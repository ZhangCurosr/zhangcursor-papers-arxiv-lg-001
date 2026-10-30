# A NEW AFIB DETECTION METHOD WITH DEEP NEURAL NETWORKS

Sara Gjorgjieva<sup>1</sup> <sup>2</sup> <sup>[0009-0009-2014-4515]</sup>, Ana Angjelevska<sup>1</sup> <sup>2</sup>, Dimitar Trajanov<sup>2</sup> <sup>[0000-0002-3105-6010]</sup>, Marjan Gusev<sup>1</sup> <sup>2</sup> <sup>[0000-0003-0351-9783]</sup>

<sup>1</sup>Innovation Dooel, Skopje, North Macedonia <sup>2</sup>Ss. Cyril and Methodius University in Skopje, Faculty of Computer Science and Engineering, North Macedonia

{sara.gjorgjieva, ana.angjelevska}@innovation.com.mk {dimitar.trajanov, marjan.gushev}@finki.ukim.mk

Published in 16th International Conference, ICT Innovations 2024 Web Proceedings, pp. 140–153. Skopje: ICT-ACT, September 2024.

## ABSTRACT

Early detection of atrial fibrillation (AFIB), characterized by irregular heart rhythms, can potentially save thousands of lives annually [1]. Numerous studies have attempted to address this challenge using various algorithms and methodologies. This paper introduces a novel and robust method that enhances the precision of AFIB detection. Neural Networks (NN) are widely recognized for their ability to identify complex electrocardiogram (ECG) patterns, facilitating the accurate detection of irregular heart rhythms. This study explores multiple conventional machine learning algorithms alongside neural network approaches to develop a robust AFIB detection system. To evaluate the effectiveness and robustness of the proposed method, the MIT BIH arrhythmia database (MITDB) and the Long-term Atrial Fibrillation Database (LTAFDB), both available on the Physionet site. Our approach presents a novel feature engineering technique, employing specific features such as a time series of the differences in successive heart rates (dBPM) rather than traditional inter-beat intervals. The model analyzes inputs labeled into five distinct classes, using only clean intervals identified through a new labeling method for training while treating others as outliers. This strategy ensures a more accurate and reliable AFIB detection process. The best-performing model, a Deep Neural Network (DNN), achieved an F1 score of 95.18% for binary atrial fibrillation classification on a benchmark dataset distinct from the training set. Models trained using our approach have demonstrated superior performance than previously published models.

Keywords atrial fibrillation · arrhythmia · deep learning · ECG

## 1 Introduction

Atrial fibrillation (AFIB) is a type of cardiac arrhythmia that occurs in the heart’s atria due to disrupting the normal cycle of the heart’s electrical impulses. It typically arises from increased automatism in a group of cells within the atria, causing their contractions to exceed the regular rate of impulses per minute and resulting in complete electrical disorganization within the atria. These abnormal impulses override the heart’s natural pacemaker, disrupting its ability to maintain a regular rhythm and leading to an irregular heart rate [2].

AFIB prevents the heart muscle from relaxing properly between contractions, reducing its efficiency and overall performance. The condition manifests as a highly irregular pulse rate. It can lead to numerous health issues, including rapid heartbeat, weakness, and dizziness, leading to blood clots, stroke, heart failure, and other heart-related complications [2].

Detecting AFIB requires the ability to identify specific ECG characteristics. AFIB is an irregular rhythm of heartbeats that seems normal in width but varies in height and spacing without preceding polarization P-waves. Most state-of-the art studies focus on detecting an irregular rhythm in the ECG without considering the absence of P-waves, which are very difficult to detect due to their very low amplitude. An irregular rhythm may have patterns such as those classified with Bigeminy (B) and Trigeminy (T), while AFIB is an irregular rhythm without patterns.

State-of-the-art analysis shows that most Neural Network approaches use RR intervals as the time between consecutive heartbeats. In this paper, we use a strategy based on differences in heart rates between consecutive heartbeats as a novel feature proven to outperform other approaches. We used multiple techniques to achieve a robust real-time medical device application, including cross-validation and an additional validation dataset to prevent overfitting. For this purpose, we used the standard MIT-BIH arrhythmia ECG benchmark database (MITDB) [3] and the Long-Term Atrial Fibrillation ECG Database (LTAFDB) [4] available on the Physionet site [5].

The paper follows the next structure. Related work in Section 2 analyses the similar approaches. Section 3 presents the methods in this research, explaining the datasets, feature engineering, machine learning models, and evaluation methodology. Section 4 addresses the performance evaluation from the experiments. Discussion in Section 5 compares the achieved results with the related work. Section 6 addresses the conclusion and gives directions for future work.

## 2 Related Work

Search for an effective algorithm to detect AFIB results in numerous approaches. Our analysis shows the application of different signal processing and Machine Learning (ML) models, including NNs. Evaluation methods differ from using proprietary datasets via standards, such as EC57 [6] or EIC 60601-2-47 [7] to derive meaningful evaluation metrics.

Authors who use the same dataset for training and testing reveal high results, although these models are not robust enough. Applying the same model to a different patient performs very poorly. An example includes [8] declaring an F1 score of 98.97% with their NN model. A model created by combining a recurrent NN and long-short-term memory (LSTM) [9] was trained and tested on the same MITDB dataset with a 10-fold cross-validation to reach an F1 score of 98.35%.

To overcome the robustness and achieve applicability, the researchers used the patient-wise method, with totally different datasets for the training and testing. For example, [10] uses different combinations of convolutional NNs and long-short-term memory (LSTM) frameworks. Although using a 70/30 splitting criteria on the same MITDB dataset, this approach reveals an F1 score of 82%. Recurrent NNs for detecting AFIB [11] on the same dataset (MITDB) obtain an F1 score of 83.00%. A distributed deep NNs [12] was used on a proprietary dataset collecting records from multiple hospitals and wearable ECG devices to detect AFIB with a 5-fold cross-validation, achieving an F1 score of 90.70%.

Our approach belongs to the class of patient-wise data split for training and testing datasets. Contrary to other NN solutions, our approach uses features based on differences in rates from consecutive heartbeats, while the others use intervals between consecutive heartbeats.

We plan three experiments: the first using the same dataset for training and testing, the second using separate ECG benchmark tests, and the third dividing the MITDB into two equal datasets. Many other papers typically use crossvalidation splits.

## 3 Methods

## 3.1 Datasets

Databases in this research are fundamental in studying heart diseases and are widely referenced in the scientific community, publicly available from the Physionet website [13].

MITDB ECG database [3] contains 30-minute ECG recordings from 48 patients, totaling 109,483 annotated heartbeats. This research excludes four records from paced patients (102, 104, 107, and 217). Consequently, the total number of annotated heartbeats analyzed is 97,924. It is a golden benchmark for evaluating newly developed algorithms and ML models, and in this research, training the new model uses MITDB. Only 10% of records consist of AFIB rhythm episodes.

LTAFDB database [4] consists of 24 to 25-hour ECG recordings from 83 individual patients diagnosed with AFIB, with over half of the records with AFIB rhythm. This database contains 8,903,169 annotated heartbeats, excluding two paced patients (6 and 113); we excluded them, totaling with 8,682,368, but because it is enormous compared to the other dataset, we used a shortened version where we included 943,953 annotated heartbeats.

The records from both databases contain predominantly Normal Sinus Rhythm (NSR), including episodes of Bigeminy (B), Trigeminy (T), Bradycardia (SBR), Tachycardia (SVT or SVTA), and a small portion of other atrial and ventricular arrhythmia. This research excludes episodes of Ventricular Fibrillation (VFIB) and Ventricular Flutter (VFL) from the analysis. Both databases are imbalanced, which is a common challenge in medical data. Addressing this imbalance will be crucial in training and evaluating our models effectively for atrial fibrillation detection.

## 3.2 Feature Engineering

Feature engineering is crucial in developing machine learning models, especially for detecting complex conditions such as atrial fibrillation (AFIB) from heart rate data. Feature engineering aims to extract features from raw data that can effectively capture the underlying patterns and differences between normal heart rhythms and AFIB episodes. For AFIB detection, various features derived from electrocardiogram (ECG) data enhance the model’s ability to distinguish between different cardiac rhythms.

This research focuses on features derived from the intervals between heartbeats to develop robust machine-learning models for AFIB detection. These features include RR intervals and additional measures of irregularity and complexity.

• RR Intervals represent the time difference between two consecutive heartbeats. It is a fundamental feature in most AFIB detection research because RR interval variations indicate irregular heart rhythms. Analyzing the sequence of RR intervals helps identify the erratic patterns typical of AFIB [14].

• GZIP Compression Length transforms each bucket formed from the time series value into a UTF-8 character and concatenates them into a string of 41 characters. The length of the compressed string serves as a feature, providing a measure of the sequence’s complexity and variability.

• Shannon Entropy (ShEn) quantifies the unpredictability or randomness in the buckets formed from time series values. It is calculated as (1), where $n _ { i }$ is the number of values in the i-th bucket, and N is the total number of intervals. Higher entropy indicates more significant irregularity, which is characteristic of AFIB episodes.

$$
S h E n = - \sum _ { i = 0 } ^ { N - 1 } ( \frac { n _ { i } } { N } ) * l o g ( \frac { n _ { i } } { N } )\tag{1}
$$

• Sample Entropy (SaEn) is calculated using the TSFEL library processing the buckets formed from time series as input. Sample Entropy measures the complexity of time-series data, with higher values suggesting more irregular and complex patterns, typical of AFIB.

By leveraging these features, our approach aims to effectively capture the nuances of different heart rhythms, enhancing the model’s ability to distinguish between AFIB and NonAFIB episodes. This comprehensive feature engineering strategy is pivotal in improving the accuracy and reliability of AFIB detection using machine learning techniques.

To train the model effectively, we use only clean segments [15] and apply a majority rule, where intervals with more than 51% of heartbeats labeled as 1 (indicative of AFIB) are classified as AFIB.

## 3.3 Ensemble Machine Learning Models

Ensemble Machine Learning Models combine the predictions from multiple models to improve accuracy and robustness. In our earlier work [16], we analyzed several ensemble ML models, which were also employed in this study for AFIB detection.

To ensure the robustness and effectiveness of these models, we employed a 10-fold cross-validation technique along with hyperparameter tuning.

## 3.3.1 Decision Tree (DT)

creates a model based on simple decision rules inferred from the data features. It recursively splits the data into subsets based on feature values, maximizing the separation between classes at each node. Decision Trees are intuitive and easy to interpret, making them a good baseline for initial model development in AFIB detection [17]. The following hyperparameters were tuned in the optimization process: maximum depth in range (1,20), minimum samples per leaf (2,6), random state [42], criterion [gini, entropy], maximal features [sqrt,log2, None], and CCP alpha [0.0, 0.01 0.1, 0.2].

## 3.3.2 Random Forest (RF)

is an ensemble method that builds a collection of Decision Trees, each trained on a randomly selected subset of the training data. By aggregating the predictions from multiple trees, Random Forest reduces overfitting and improves generalization. This method is particularly effective in dealing with noisy data and capturing complex heart rate variability patterns indicative of AFIB. [18]. The same hyperparameter settings were used as in DT, maximum depth in range (1,20), minimum samples per leaf (2,6), random state [42], criterion [gini,entropy], maximal features [sqrt,log2, None], and CCP alpha [0.0, 0.01 0.1, 0.2]

## 3.3.3 XGBoost (XGB)

stands for Extreme Gradient Boosting as a powerful ensemble learning technique that builds decision trees in parallel, optimizing each tree based on the errors of the previous trees. XGBoost incorporates regularization techniques to prevent overfitting, making it highly effective for large datasets and complex feature spaces. Its ability to compute individual tree errors concurrently enhances model performance and generalization, which is crucial for accurately detecting AFIB episodes amidst varying heart rhythms [19]. The hyperparameter tuning optimized the learning rate in the range [0.01, 0.1, 0.16, 0.2], number of estimators [100, 200, 300], max depth range(1,20), colsample bytree [0.5, 0.7 0.9], min child weight [1,2], and regularization parameter reg alpha [0.01, 0.1, 1].

## 3.3.4 AdaBoost (AB)

stands for Adaptive Boosting that combines multiple weak learners sequentially. It assigns higher weights to misclassified instances, forcing subsequent models to focus more on these challenging cases. This iterative process improves the model’s ability to correct errors and enhances its robustness. It supports the classification of imbalanced datasets, such as those with fewer AFIB episodes compared to NonAFIB episodes, by focusing learning on the minority class [20]. We optimized the number of estimators in the range [100, 200, 300] and learning rate [0.01, 0.1, 0.2, 0.3] within the hyperparameter tuning.

## 3.3.5 CatBoost (CB)

means Categorical Boosting that handles categorical features effectively through an ordered boosting approach. It builds decision trees by considering the order of observations and incorporating categorical data directly into the model. CatBoost is known for its training efficiency and ability to handle high-cardinality categorical features, making it suitable for complex datasets encountered in AFIB detection. We selected the loss function [Logloss] and evaluation metric [AUC]. The hyperparameter tuning optimized the learning rate [0.01, 0.1, 0.2] and colsample by level [0.5, 0.7, 0.9].

## 3.4 Deep Learning Model Development

We also employed deep learning (DL) models to detect atrial fibrillation (AFIB) from ECG data.

## 3.4.1 Recurrent Neural Network (RNN)

are designed to process sequential data where the order of inputs matters, such as time series data. In AFIB detection, RNNs are advantageous because they can learn temporal dependencies and relationships between heartbeats, capturing the irregular patterns characteristic of AFIB episodes. RNNs maintain a hidden state updated at each time step, allowing them to retain information across long sequences [21]. They can effectively handle sequential data, such as ECG signals [22] because it is designed to maintain temporal dependencies between inputs.

The RNN model used in this study consists of several layers. The SimpleRNN Layer is the initial layer in the network that processes input data as a sequence of time steps, capturing temporal patterns. The dense layers are created with Leaky ReLU Activations. After the SimpleRNN layer, the model incorporates six fully connected (Dense) layers. These layers use leaky ReLU (Rectified Linear Unit) activation functions to introduce non-linearity, allowing the network to learn more complex data patterns. The Dropout Layers are interspersed between the dense layers to mitigate overfitting. These layers randomly deactivate some neurons during training, forcing the network to learn more robust features. L2 Regularization [23] is applied to the dense layers to penalize large weights, further controlling overfitting by smoothing the model. Learning Rate Scheduler is a dynamic scheduler that adjusts the learning rate during training epochs to enhance model convergence and performance. Optimizer and Loss Function is realized by the Adam optimizer for efficient gradient-based optimization and binary cross-entropy loss function to measure the error in binary classification tasks (AFIB vs. NonAFIB). The training parameters include multiple epochs with a batch size of 128. The validation set, derived from a separate test set, assesses the model’s generalization performance during training.

## 3.4.2 Siamese Neural Network (SNN)

is based on a specialized architecture that learns to measure similarity or dissimilarity between two input instances, regardless of the specific task. In the context of AFIB detection, SNNs can be trained to compare segments of ECG data and determine whether they represent similar rhythm patterns (e.g., AFIB vs. NonAFIB) [24].

The twin network architecture in the SNN consists of two identical branches, each processing ECG signals. The architecture is designed such that both branches have the same structure and shared weights, ensuring that they learn the same features. Each branch separately consists of three fully connected Dense layers and four more merged layers. The feature extraction and comparison is done during training, where both branches receive pairs of ECG signals (one set from AFIB episodes and another from NSR episodes). The outputs from both branches are combined and used for the final classification. Yje loss Function and optimization uses the binary cross-entropy loss function, suitable for binary classification tasks. The Adam optimizer updates the network weights during training, minimizing classification errors. The key advantage is the dual-branch SNN architecture that allows the model to effectively learn temporal patterns and similarities between pairs of inputs, which enhances its ability to differentiate between AFIB and NSR episodes.

## 3.4.3 Deep Neural Network (DNN)

is designed to learn and represent complex data patterns through multiple layers of neurons. It is particularly effective at extracting hierarchical features from raw input data, essential for making accurate predictions or classifications in AFIB detection. By leveraging deep architectures, DNNs can capture high-level abstractions and fine-grained details from ECG data, enhancing the model’s ability to distinguish between different heart rhythms [25].

We implement DNN architecture using Keras to extract and process features from the input data through multiple layers. The dense layers with leaky ReLU activations are crucial for capturing non-linear relationships in the input data and distinguishing between normal sinus rhythm and AFIB episodes. The |em dropout layers are applied to the dense layers to prevent overfitting. This regularization technique helps the model generalize unseen data better by randomly dropping units during training. The Optimizer and Loss Function is realized by the Adam optimizer for training and minimizes the binary cross-entropy loss function. This setup is well-suited for binary classification tasks and helps the model learn effectively. The training strategy optimizes the model’s ability to classify AFIB episodes accurately.

## 3.4.4 Hyperparemeter optimization

includes the following ranges: dropout rate [0.1 - 0.5], optimizer [adam,sgd], epochs [5-50], batch size [32 - 256], activation [relu, leaky relu] and layers config [3,9,12,18].

## 3.5 Evaluation Methodology

Evaluating the performance of machine learning models in detecting atrial fibrillation (AFIB) is critical, especially given the inherent class imbalance in AFIB datasets, where NonAFIB episodes often outnumber AFIB episodes. We use the F1 score to assess model performance, a metric particularly suited for imbalanced classification problems.

The F1 score is calculated as the harmonic mean of sensitivity (SEN) and precision, providing a balanced measure that considers both false positives and false negatives. This is especially important in medical diagnoses, where missing an AFIB episode (false negative) or incorrectly identifying a NonAFIB episode as AFIB (false positive) can have significant clinical implications. The F1 score is calculated by (2), Where: Sensitivity (SEN), also known as the True Positive Rate (TPR), measures the proportion of actual positive cases (AFIB episodes) correctly identified by the model calculated by (3), and Precision, also known as Positive Predictive Value (PPV), measures the proportion of predicted positive cases versus all positives, calculated by (3).

$$
F _ { 1 } = \frac { 2 * P P V * S E N } { P P V + S E N }\tag{2}
$$

$$
S E N = { \frac { T P } { T P + F N } } \qquad P P V = { \frac { T P } { T P + F P } }\tag{3}
$$

## 3.5.1 Duration-Based Evaluation Method

[5] evaluates the duration of correctly identified AFIB episodes instead of the heartbeat-oriented evaluation, providing a more nuanced understanding of the model’s ability to detect AFIB over varying time intervals.

Each analyzed interval is assigned a value, and a specific post-processing procedure is applied to smooth out short sequences with alternating labels. This smoothing process helps mitigate the impact of noisy predictions and more

accurately delineates the beginning and end of AFIB episodes. By refining these sequences, the model’s detection of AFIB episodes aligns more closely with clinically meaningful patterns, providing a more reliable evaluation of its real-world applicability

This dual approach ensures that the evaluation captures the model’s ability to correctly identify AFIB cases and its effectiveness in recognizing AFIB episodes amidst varying heart rhythms.

## 3.6 Experiments

We specify three experiments based on different training/validation/ testing datasets.

## 3.6.1 Experiment 1

uses a random selection of data samples and forms three subsets by an 80/10/10 split ratio. This splitting method lacks robustness and generality since it includes data from the same patient in all three datasets, so the model will be evaluated with data for the patient for which it was trained and validated. Experiment 1a uses MITDB and Experiment 1b LTAFDB.

## 3.6.2 Experiment 2

uses different datasets where the first dataset forms the training and validation dataset with a ratio of 80/20 split and a different dataset for testing. We use MITDB as the first dataset divided into training and validation datasets. LTAFDB is used as a testing dataset. This method ensures testing with a completely independent dataset, providing a robust measure of its performance in real-world scenarios and generalization capability across different datasets and patient populations. Experiment 2a uses MITDB for training and LTAFDB for testing, while Experiment 2b uses the opposite LTAFDB for training and MITDB for testing.

## 3.6.3 Experiment 3

uses the De Chazal Method, where the MITDB dataset is divided with the interpatient method into DS1 and DS2 datasets, specifically designed to handle the class imbalance present in the MITDB database, such that DS1 is used for training and DS2 is used for testing. The key aspect of this method is that patients included in the training dataset are not included in the testing set, preventing data leakage and overfitting. This split method is highly effective in balancing patient distribution. The DS1 dataset includes patients 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, and 230, and DS2 the remaining from the MITDB.

## 4 Results

Tables 1 through 5 present the results for each evaluated model across different experiments.

Table 1: Results obtained from Experiment 1a, training with MITDB, and evaluating with MITDB
<table><tr><td>Method</td><td>SEN (%)</td><td>PPV (%)</td><td>F1 (%)</td><td>Mean (%)</td><td>STD (%)</td></tr><tr><td>DT</td><td>91.14</td><td>87.71</td><td>89.39</td><td>73.02</td><td>32.66</td></tr><tr><td>RF</td><td>91.66</td><td>89.36</td><td>90.49</td><td>88.52</td><td>1.14</td></tr><tr><td>XGB</td><td>91.35</td><td>88.52</td><td>89.91</td><td>88.89</td><td>0.60</td></tr><tr><td>AB</td><td>93.10</td><td>86.84</td><td>89.86</td><td>88.92</td><td>0.62</td></tr><tr><td>CB</td><td>91.56</td><td>89.17</td><td>90.35</td><td>89.53</td><td>0.13</td></tr><tr><td>RNN</td><td>91.80</td><td>87.24</td><td>89.46</td><td>89.46</td><td>0.00</td></tr><tr><td>SNN</td><td>97.44</td><td>76.62</td><td>85.79</td><td>85.79</td><td>0.00</td></tr><tr><td>DNN</td><td>88.21</td><td>89.03</td><td>88.62</td><td>88.62</td><td>0.00</td></tr></table>

Among all the models, the Deep Neural Network (DNN) model, as shown in Table 3, which was trained on the MITDB dataset and tested on the LTAFDB dataset, stands out as the most effective. It achieved an impressive F1 score of 95.18% in detecting AFIB with separate training and testing datasets.

The DNN is the best-performing model due to its exceptional ability to generalize across different datasets. This model performed well on distinct datasets and demonstrated robustness by including cross-validation and an additional validation dataset in the evaluation process.

Table 2: Results obtained from Experiment 1b, training with LTAFDB, and evaluating with LTAFDB
<table><tr><td>Method</td><td>SEN (%)</td><td>PPV (%)</td><td>F1 (%)</td><td>Mean (%)</td><td>STD (%)</td></tr><tr><td>DT</td><td>96.56</td><td>96.95</td><td>96.75</td><td>96.55</td><td>0.13</td></tr><tr><td>RF</td><td>96.55</td><td>97.05</td><td>96.80</td><td>96.68</td><td>0.14</td></tr><tr><td>XGB</td><td>96.44</td><td>97.11</td><td>96.77</td><td>96.74</td><td>0.09</td></tr><tr><td>AB</td><td>96.82</td><td>96.69</td><td>96.75</td><td>96.68</td><td>0.06</td></tr><tr><td>CB</td><td>96.58</td><td>97.06</td><td>96.82</td><td>96.84</td><td>0.01</td></tr><tr><td>RNN</td><td>96.29</td><td>97.18</td><td>96.73</td><td>96.73</td><td>0.00</td></tr><tr><td>SNN</td><td>96.61</td><td>96.71</td><td>96.66</td><td>96.66</td><td>0.00</td></tr><tr><td>DNN</td><td>96.44</td><td>97.06</td><td>96.75</td><td>96.75</td><td>0.00</td></tr></table>

Table 3: Results obtained from Experiment 2a, training with MITDB, and evaluating with LTAFDB
<table><tr><td>Method</td><td>SEN (%)</td><td>PPV (%)</td><td>F1 (%)</td><td>Mean (%)</td><td>STD (%)</td></tr><tr><td>DT</td><td>90.90</td><td>97.01</td><td>93.85</td><td>72.99</td><td>32.65</td></tr><tr><td>RF</td><td>89.88</td><td>97.25</td><td>93.42</td><td>88.48</td><td>1.09</td></tr><tr><td>XGB</td><td>89.52</td><td>97.29</td><td>93.24</td><td>88.78</td><td>0.50</td></tr><tr><td>AB</td><td>91.78</td><td>97.17</td><td>94.40</td><td>88.85</td><td>0.57</td></tr><tr><td>CB</td><td>89.58</td><td>97.21</td><td>93.24</td><td>89.49</td><td>0.14</td></tr><tr><td>RNN</td><td>93.75</td><td>96.48</td><td>95.09</td><td>95.09</td><td>0.00</td></tr><tr><td>SNN</td><td>95.25</td><td>94.58</td><td>94.91</td><td>94.91</td><td>0.00</td></tr><tr><td>DNN</td><td>94.87</td><td>95.49</td><td>95.18</td><td>95.18</td><td>0.00</td></tr></table>

Table 4: Results obtained from Experiment 2b, training with LTAFDB, and evaluating with MITDB
<table><tr><td>Method</td><td>SEN (%)</td><td>PPV (%)</td><td>F1 (%)</td><td>Mean (%)</td><td>STD (%)</td></tr><tr><td>DT</td><td>96.43</td><td>58.60</td><td>72.89</td><td>96.52</td><td>0.13</td></tr><tr><td>RF</td><td>96.17</td><td>59.18</td><td>73.27</td><td>96.65</td><td>0.14</td></tr><tr><td>XGB</td><td>96.43</td><td>58.92</td><td>73.14</td><td>96.73</td><td>0.10</td></tr><tr><td>AB</td><td>96.92</td><td>57.46</td><td>72.15</td><td>96.65</td><td>0.06</td></tr><tr><td>CB</td><td>96.23</td><td>58.71</td><td>72.93</td><td>96.82</td><td>0.01</td></tr><tr><td>RNN</td><td>97.05</td><td>58.00</td><td>72.61</td><td>72.61</td><td>0.00</td></tr><tr><td>SNN</td><td>93.78</td><td>64.36</td><td>76.34</td><td>76.34</td><td>0.00</td></tr><tr><td>DNN</td><td>97.69</td><td>56.00</td><td>71.18</td><td>71.18</td><td>0.00</td></tr></table>

Table 5: Results obtained from Experiment 3, training with DS1, and evaluating with DS2
<table><tr><td>Method</td><td>SEN (%)</td><td>PPV (%)</td><td>F1 (%)</td><td>Mean (%)</td><td>STD (%)</td></tr><tr><td>DT</td><td>87.35</td><td>79.93</td><td>83.47</td><td>53.10</td><td>37.58</td></tr><tr><td>RF</td><td>89.20</td><td>79.74</td><td>84.21</td><td>59.59</td><td>35.96</td></tr><tr><td>XGB</td><td>83.24</td><td>81.20</td><td>82.21</td><td>80.62</td><td>1.88</td></tr><tr><td>AB</td><td>81.29</td><td>82.29</td><td>81.79</td><td>82.22</td><td>0.97</td></tr><tr><td>CB</td><td>79.67</td><td>82.23</td><td>80.93</td><td>80.51</td><td>0.59</td></tr><tr><td>RNN</td><td>91.04</td><td>75.30</td><td>82.43</td><td>82.43</td><td>0.00</td></tr><tr><td>SNN</td><td>92.14</td><td>68.87</td><td>78.82</td><td>78.82</td><td>0.00</td></tr><tr><td>DNN</td><td>88.08</td><td>73.34</td><td>80.04</td><td>80.04</td><td>0.00</td></tr></table>

## 5 Discussion

Based on the results obtained from our experiments and the reviewed studies, NNs have proven effective in detecting AFIB.

To ensure robustness, we used different variations of training and testing of the models to prove their efficiency and accuracy. Due to the imbalance in our datasets, we selected the F1 score as our performance metric, as it is well-suited for assessing model performance on skewed datasets. We also used 10-fold cross-validation on our ensembling methods, mean and standard deviation of the classifier performance, and additional hyperparameter tuning on these methods.

The success of our approach also hinges on selecting appropriate parameters and designing an optimal NN structure tailored for AFIB detection. While the SNN, RNN, and DNN methods also performed well, the CB model demonstrated superior performance in our study, with an F1 score of 96.82%. Table 6 compares the results from related research achieved by other authors. Note that our methods did not achieve the highest scores compared to the others. Still, we aimed to attain maximal robustness using a 10-fold cross-validation and addition validation dataset. We also included hyperparameter tuning to reach the best results from our models. Additionally, we applied extra techniques to prevent overfitting.

Table 6: Comparison with other papers
<table><tr><td>Related Work</td><td>Method</td><td>F1(%)</td><td>Split</td><td>Dataset</td></tr><tr><td>Martis et al. [8]</td><td>ANN</td><td>98.97</td><td></td><td>MITDB, AFDB</td></tr><tr><td>Wang et al. [10]</td><td>CNN+LSTM</td><td>82.00</td><td>70/30</td><td>MITDB</td></tr><tr><td>Teijeiro et al. [11]</td><td>RNN</td><td>83.00</td><td></td><td>PhysioNet/CinC 2017</td></tr><tr><td>Cai et al. [12]</td><td>DDNN</td><td>90.70</td><td>5 fold CV</td><td>proprietary data</td></tr><tr><td>Faust et al. [9]</td><td>RNN + LSTM</td><td>98.35</td><td>10 fold CV</td><td>AFDB</td></tr><tr><td>Artis et al. [26]</td><td>ANN</td><td>92.60</td><td></td><td>MITDB</td></tr><tr><td>Our method</td><td>RF</td><td>90.49</td><td>80/10/10+10 fold CV</td><td>MITDB</td></tr><tr><td>Our method</td><td>CB</td><td>96.82</td><td>80/10/10+10 fold CV</td><td>LTAFDB</td></tr><tr><td>Our method</td><td>DNN</td><td>95.18</td><td>separate datasets</td><td>MITDB, LTAFDB</td></tr><tr><td>Our method</td><td>SNN</td><td>76.34</td><td>separate datasets</td><td>LTAFDB, MITDB</td></tr><tr><td>Our method</td><td>RF</td><td>84.21</td><td>deChazal split</td><td>MITDB</td></tr></table>

Table 7: Optimal hyperparameters for the results achieved in Table 6
<table><tr><td>Model</td><td>Training</td><td>Testing</td><td>Optimal Parameters</td></tr><tr><td>RF</td><td>MITDB</td><td>MITDB</td><td>ccp alpha:[0.0], criterion:[gini], max depth:[12], max features:[sqrt], estimators:[300], random state [42]</td></tr><tr><td>CB</td><td>LTAFDB</td><td>LTAFDB</td><td>colsample bylevel:[0.5], evaluation metric:[AUC], learning rate:[0.1], loss function:[Logloss]</td></tr><tr><td>DNN</td><td>MITDB</td><td>LTAFDB</td><td>layers_config=[3, 9, 12, 18], dropout rate:[0.1], optimizer:[adam], epochs:[19], batch size:[128]</td></tr><tr><td>SNN</td><td>LTAFDB</td><td>MITDB</td><td>layers_config=[3, 9, 12, 18], epochs:[8], batch size:[128] optimizer:[adam]</td></tr><tr><td>RF</td><td>DS1</td><td>DS2</td><td>ccp alpha:[0.0], criterion:[entropy], max depth:[2], max features:[sqrt], estimators:[200], random state:[42]</td></tr></table>

Table 7 represents the optimal hyperparameters used to achieve the highest result for each train/test combination we used, as shown in Table 6.

## 6 Conclusion

We presented a new method for developing ML models based on newly derived features, such as Gzip length, Shannon Entropy, and Spectral Entropy, besides the series of differences in heart rates between consecutive heartbeats. Special labeling techniques divide the dataset into five classes, and the AFIB binary classification determines the start and end of an AFIB rhythm applied to the duration-based evaluation. We used the MITDB and LTAFDB datasets for the experiments and tried all the combinations to see which gave us the best results. Specifically, the decision to train the model only with clean intervals improved the model’s performance. The results outperform other research compared with the same evaluation methodology.

This research shows that DNN models have great potential in biomedical signal processing, especially for detecting AFIB. Advanced deep learning techniques are promising technology after carefully designing the underlying problem. These advancements offer hope for improving medical outcomes and patient care in heart health. Refining NN methods further could lead to even more progress in AFib detection and other vital areas of medical research. However, besides the secret ingredients in developing the model, we conclude that feature engineering improved the whole process, especially selecting the training subset from the training dataset by applying a specific labeling method.

## References

[1] Ziad Sankari and Hojjat Adeli. Heartsaver: A mobile cardiac monitoring system for auto-detection of atrial fibrillation, myocardial infarction, and atrio-ventricular block. Computers in biology and medicine, 41(4):211–220, 2011.

[2] American Heart Association. What is Atrial Fibrillation?, 2024. Last accessed online on 01.07.2024 at: https://www.heart.org/en/health-topics/atrial-fibrillation/ what-is-atrial-fibrillation-afib-or-af.

[3] George B Moody and Roger G Mark. The impact of the mit-bih arrhythmia database. IEEE engineering in medicine and biology magazine, 20(3):45–50, 2001.

[4] Simona Petrutiu, Alan V Sahakian, and Steven Swiryn. Abrupt changes in fibrillatory wave characteristics at the termination of paroxysmal atrial fibrillation in humans. Europace, 9(7):466–470, 2007.

[5] Marjan Gusev and Martina Boshkovska. Performance evaluation of atrial fibrillation detection. In 2019 42nd International Convention on Information and Communication Technology, Electronics and Microelectronics (MIPRO), pages 342–347. IEEE, 2019.

[6] ANSI/AAMI. ANSI/AAMI EC57:2012 (R2020) testing and reporting performance results of cardiac rhythm and st segment measurement algorithms, 2012. Last accessed online on 19.01.2024 at: https://webstore.ansi. org/standards/aami/ansiaamiec572012r2020.

[7] IEC. IEC 60601-2-47:2012 medical electrical equipment - part 2-47: Particular requirements for the basic safety and essential performance of ambulatory electrocardiographic systems, 2012. Last accessed online on 19.01.2024 at: https://webstore.ansi.org/standards/aami/ansiaamiiec60601472012r2016.

[8] Roshan Joy Martis, U Rajendra Acharya, Hari Prasad, Chua Kuang Chua, Choo Min Lim, and Jasjit S Suri. Application of higher order statistics for atrial arrhythmia classification. Biomedical signal processing and control, 8(6):888–900, 2013.

[9] Oliver Faust, Alex Shenfield, Murtadha Kareem, Tan Ru San, Hamido Fujita, and U Rajendra Acharya. Automated detection of atrial fibrillation using long short-term memory network with rr interval signals. Computers in biology and medicine, 102:327–335, 2018.

[10] Jiacheng Wang and Weiheng Li. Atrial fibrillation detection and ecg classification based on cnn-bilstm. arXiv preprint arXiv:2011.06187, 2020.

[11] Tomás Teijeiro, Constantino A García, Daniel Castro, and Paulo Félix. Arrhythmia classification from the abductive interpretation of short single-lead ecg records. In 2017 Computing in cardiology (cinc), pages 1–4. IEEE, 2017.

[12] Wenjuan Cai, Yundai Chen, Jun Guo, Baoshi Han, Yajun Shi, Lei Ji, Jinliang Wang, Guanglei Zhang, and Jianwen Luo. Accurate detection of atrial fibrillation from 12-lead ecg using deep neural network. Computers in biology and medicine, 116:103378, 2020.

[13] Databases. https://www.physionet.org/about/database/, 2024. Accessed: 2024-04-26.

[14] Shrikanth Rao SK, MaheshKumar H Kolekar, and Roshan Joy Martis. A deep learning approach for detecting atrial fibrillation using rr intervals of ecg. Frontiers in Biomedical Technologies, 11(2):255–264, 2024.

[15] Dimitri Dojchinovski and Marjan Gusev. Segment labeling method for ml-based afib detection. 2020.

[16] Stojancho Tudjarski, Tomislav Ignjatov, and Marjan Gusev. A new ml-based afib detector. In 2021 29th Telecommunications Forum (TELFOR), pages 1–4, 2021.

[17] Guangyu Bin, Minggang Shao, Guanghong Bin, Jiao Huang, Dingchang Zheng, and Shuicai Wu. Detection of atrial fibrillation using decision tree ensemble. In 2017 Computing in Cardiology (CinC), pages 1–4. IEEE, 2017.

[18] Alan Kennedy, Dewar D Finlay, Daniel Guldenring, Raymond R Bond, Kieran Moran, and James McLaughlin. Automated detection of atrial fibrillation using rr intervals and multivariate-based classification. Journal of electrocardiology, 49(6):871–876, 2016.

[19] Yao Chen, Xiao Wang, Yonghan Jung, Vida Abedi, Ramin Zand, Marvi Bikak, and Mohammad Adibuzzaman. Classification of short single-lead electrocardiograms (ecgs) for atrial fibrillation detection using piecewise linear spline and xgboost. Physiological measurement, 39(10):104006, 2018.

[20] Inas Salah Mahmood and Iman Abuel Maaly Abdelrahman. A comparison between different classifiers for diagnoses of atrial fibrillation. In 2019 International Conference on Computer, Control, Electrical, and Electronics Engineering (ICCCEEE), pages 1–6. IEEE, 2019.

[21] VG Sujadevi, KP Soman, and R Vinayakumar. Real-time detection of atrial fibrillation from short time single lead ecg traces using recurrent neural networks. In Intelligent systems technologies and applications, pages 212–221. Springer, 2018.

[22] Rasmus S Andersen, Abdolrahman Peimankar, and Sadasivan Puthusserypady. A deep learning approach for real-time detection of atrial fibrillation. Expert Systems with Applications, 115:465–473, 2019.

[23] Ekachai Phaisangittisagul. An analysis of the regularization between l2 and dropout in single hidden layer neural network. In 2016 7th International Conference on Intelligent Systems, Modelling and Simulation (ISMS), pages 174–179. IEEE, 2016.

[24] Yiuwai Ng, Min-Tsun Liao, Ting-Li Chen, Chih-Kuo Lee, Cheng-Ying Chou, and Weichung Wang. Few-shot transfer learning for personalized atrial fibrillation detection using patient-based siamese network with single-lead ecg records. Artificial Intelligence in Medicine, 144:102644, 2023.

[25] Luan Tran, Yanfang Li, Luciano Nocera, Cyrus Shahabi, and Li Xiong. Multifusionnet: atrial fibrillation detection with deep neural networks. AMIA Summits on Translational Science Proceedings, 2020:654, 2020.

[26] Shane G Artis, RG Mark, and GB Moody. Detection of atrial fibrillation using artificial neural networks. Master’s thesis, Massachusetts Institute of Technology, Dept. of Electrical Engineering and . . . , 1991.