# SAGE: Stability-Aware Graph-Based Ensemble Feature Selection for Explainable Postpartum Depression Risk Prediction

Md. Rokon Islam Emon

Syed Shariar Alam Shuvo

Department of Computer Science

Department of Law and Economics

Shahriar Siddique Ayon

Brunel University London

Technical University of Darmstadt

Department of Computer Science

Uxbridge UB8 3PH, United Kingdom

Darmstadt, Germany

American International University-Bangladesh

2565915@brunel.ac.uk

shuvo.ss.121@gmail.com

Dhaka, Bangladesh

Abdullah Al Mamun

shahriarayon63@gmail.com

Department of Computer Science and Engineering

Dhaka University of Engineering and Technology

Gazipur, Bangladesh

mamun.duet.bd@gmail.com

Ahnaf Atef Choudhury

Department of Information Sciences and Technology

George Mason University

Fairfax, VA, USA

achoudh9@gmu.edu

Abstract—Postpartum depression (PPD) poses a major burden on maternal and child health, especially in low- and middleincome countries where prevalence exceeds 19%. Despite advancements in machine learning for PPD prediction, current approaches are limited by opaque global explanations that lack clinical usefulness at the patient level, unstable feature selection, and poor generalization under class imbalance. We propose SAGE, a Stability-Aware Graph-Based Ensemble feature selection system that incorporates both local explainable AI and a genetically optimized artificial neural network (GA-ANN). Using a primary cohort of 766 postpartum women, SAGE combines information-theoretic relevance, PCA-based structure, and graph-based interactions with bootstrap stability weighting to identify robust and non-redundant predictors. The GA-ANN architecture, optimized using a genetic algorithm and enhanced with GAN based oversampling, achieved strong performance with 87.96% accuracy, 86.32% F1 score, and 0.88 AUC using only 16 features, outperforming baseline and other feature selection methods. Psychological and socioeconomic factors such as EPDS score, PHQ-9 score, feelings about motherhood, and abuse history are the main predictors, while demographic factors have less influence. The LIME-based explanations allow instancebased insight into selected features from the graph, enabling personalized risk assessment. The findings make SAGE a scalable, interpretable, and clinical tool for early identification of PPD in health-care limited resources.

Index Terms—Postpartum depression, Feature selection, Genetic algorithm, Data-driven, Maternal mental health

## I. INTRODUCTION

Postpartum depression (PPD) is a common and serious condition after childbirth that affects mothers and can have lasting effects on families. The global prevalence of PPD is estimated at 17.22%, with a disproportionate burden in low- and middle-income countries (19%) compared to highincome countries (13%) [1]. According to the World Health Organization, over 20% of mothers in developing countries suffer from PPD, which is associated with a higher risk of suicide and bad child outcomes such as poor growth, developmental delays, and illness [2]. A meta-analysis of 86 studies found a 22.1% prevalence of perinatal depression in rural areas and 21.1% for postnatal depression, with higher rates of 25.4% in Bangladesh due to economic hardship and limited mental health services, emphasizing the importance of early detection in South Asian resource-limited settings [3].

PPD arises from combined social, economic, obstetric, and psychological factors; low social support, marital dissatisfaction, and low income (≈ 13% higher risk) significantly increase vulnerability [4]. Obstetric complications further worsen outcomes, with higher depression rates in affected mothers (23.97%) compared to uncomplicated deliveries (16.71%) [5]. Given this complexity, artificial intelligence (AI) is essential in helping convert vast amounts of maternal healthcare data into effective clinical insights. Several recent studies have demonstrated strong performance using ensemble learning and optimized gradient boosting models, including XGBoost, for integrating obstetric and psychological factors with explainable AI capabilities [6], [7]. These advances highlight the potential of AI for early identification of mothers at risk of PPD.

Many machine learning (ML)-based PPD studies achieve high accuracy; however, their reliability is often limited by weak validation, class imbalance, and unstable feature selection. Existing graph-based selectors may overlook statistical relevance and structural variability, while ensemble approaches often lack stability guarantees, leading to inconsistent feature rankings across resamples [8]–[10]. These limitations reduce interpretability and clinical trust, preventing current methods from achieving reliable real-world deployment.

To address these limitations, this study proposes a Stability-

Aware Graph-based Ensemble (SAGE) feature selection framework that is coupled with a genetically optimized artificial neural networks (GA-ANN) and local explainable AI (XAI) for robust PPD prediction. SAGE combines information-theoretic, PCA, graph-based, and stabilityweighted feature selection. Combined with GAN oversampling, it improves GA-ANN performance, while LIME provides patient-level explanations. Key Contributions of this work:

• A novel stability-aware graph ensemble feature selector that explicitly models feature interactions and selection robustness.

• Analysis of a primary dataset of 766 postpartum women in Bangladesh shows that psychological and socioeconomic factors are the key predictors, while demographic factors are less influential.

• A GA-optimized ANN designed to enhance performance on imbalanced perinatal data using GAN-based oversampling.

• The first integration of local XAI with graph-based feature selection for patient-level PPD risk stratification.

The paper is organized as follows: Section II reviews related work, Section III presents the proposed methodology, Section IV discusses the results, and Section V concludes the study with future directions.

## II. RELATED WORK

Recent studies indicate that AI can assist in predicting the risk of antenatal and PPD; however, its performance varies across different data sources. Clapp et al. used electronic health records (EHR) of 29,168 cases and found that prepartum Edinburgh Postnatal Depression Scale (EPDS) scores significantly improve early prediction of PPD [11]. Wang et al. used random forests to determine trends in PPD between 30.9% and 29.1% in 3,174 women, and found birth weight and maternal weight to be important indicators [12]. Another study, which used recursive feature elimination on 8,454 cases, achieved an AUC of 0.91 for predicting post-partum care [13].

In studies on cesarean delivery data, XGBoost and neural network ensembles perform very well (AUC of 0.789-0.955, accuracy of 95.0%) and are even better with hyperparameter tuning [14]. But results on cesarean delivery and other postpartum data sets are not always the same. For instance, there was an increase in performance (AUC 0.897-0.733 on the original data) after using SMOTE [15]. A study of 87 papers on PPD also reported that the average accuracy was high (nearly 93.4%) [16], but it was not very rigorous either, with less than half of the papers using internal or external validation.

Zhang et al. used LASSO and variance inflation factor analysis to identify 17 key predictors from 78 variables, achieving an AUC of 0.849 with XGBoost [17]. Cellini et al. combined LASSO and Boruta feature selection. Identified 11 important predictors, including gestational weight gain, in-law relationship and sleep quality for gradient boosting [8]. A systematic review highlighted that most studies rely on sociodemographic or clinical filtering without validating stability, which limits how well the models can be applied generally [18].

Recent studies have expanded PPD prediction through diverse data modalities and validation strategies. Qi et al. demonstrated robust performance using externally validated ML models on large-scale clinical data [19]. Wang et al showed that plasma proteomics-based ML can provide objective, pre-symptomatic risk stratification [20]. As predictive models move toward clinical use, obstetric decision-support systems should provide interpretable predictions. SHAP-based methods have been widely used to interpret PPD prediction models, including XGBoost, by identifying the contributions of key factors such as anxiety, pelvic floor muscle strength, and postnatal care satisfaction [8]. SHAP is also utilized for validating clinical cut-offs (e.g., 21.5%) and making decisions regarding interventions among high-risk populations [14]. Most work still focuses on SHAP, while methods like LIME and counterfactual explanations are less commonly used.

However, current research often use weak validation, handle class imbalance poorly, and rely on unstable feature selection that ignores feature interactions and underutilizes rich postpartum features. This calls for the development of an improved and graph-oriented feature selection model with optimized models and local explanations. Applied to the NAFLD dataset with a SAGE-based Genetic Algorithm (GA) optimized ANN and GAN oversampling, the framework improved results from 98.35% accuracy and 98.64% F1-score to 98.46% accuracy and 98.73% F1-score, showing its effectiveness and novelty [21]. The following sections present the detailed methodology and results of our study.

## III. METHODOLOGY

This section provides an overview of the proposed framework, data preparation, feature engineering, and development of an optimised model for women who have given birth. Data imbalance is addressed through oversampling and XAI is used for risk and protective factor identification. The proposed prediction framework is illustrated in Figure 1.

## A. Data Collection and Preprocessing

The dataset used in this study is a primary field-collected dataset of 766 postpartum women in Bangladesh, gathered from hospital settings between March and June 2025 across urban, rural, and outpatient facilities 1. Participants aged 18–41 were recruited within 24 months of childbirth following ethical guidelines, ensuring informed consent and anonymity. Mental health status was assessed using standardized instruments, including Patient Health Questionnaire (PHQ)-2, PHQ-9, and EPDS, with 40.47% of participants screening positive for depression during pregnancy. The dataset includes diverse sociodemographic, clinical, obstetric, psychosocial, and neonatal factors, covering maternal characteristics, pregnancy history, delivery details, health conditions, and postnatal behaviors.

![](images/60ce9b1d7f585fed5e6a6c0c74502b71b3a082dcb5979a521700719938111798.jpg)  
Fig. 1. Proposed Framework for Postpartum Depression Prediction.

At first, the dataset had 51 features and 766 patient records. Irrelevant identifiers such as Patient ID were removed, along with redundant outcome columns (PHQ-9 Result and EPDS Result) since their corresponding score values were retained. There are no missing values in the dataset. Instead of being nulls, entries marked as "None" were treated as valid categorical values. There are four numerical features, and the other categorical variables were turned into numerical data using label encoding. The final dataset has 48 features and 766 instances after preprocessing.

## B. Stability-Aware Graph-Based Ensemble Feature Selection

After preprocessing, redundant features were reduced using Recursive Feature Elimination (RFE), Tree-based Feature Importance, Principal Component Analysis-Information Gain (PCA-IG), and the proposed Stability-Aware Graph Ensemble (SAGE) method. SAGE feature selection combines statistical relevance, structural variance, and graph-based interaction information in a single framework, which is weighted to account for stability in feature selection for classification problems. Information gain and PCA-based structural contribution are defined as:

$$
I G _ { j } = I ( X _ { j } ; y ) , \quad P C A _ { j } = \sum _ { k = 1 } ^ { m } | V _ { j k } | \cdot \lambda _ { k }\tag{1}
$$

While graph-based interaction importance and final ensemble scoring are given by:

$$
C _ { j } = \frac { \deg ( X _ { j } ) } { d - 1 } , \quad S c o r e _ { j } = \alpha I G _ { j } ^ { * } + \beta P C A _ { j } ^ { * } + \gamma C _ { j } ^ { * }\tag{2}
$$

Where stability-adjusted components $I G _ { j } ^ { * } , P C A _ { j } ^ { * }$ , and $C _ { j } ^ { * }$ are derived using bootstrap statistics as:

$$
S _ { j } ^ { * } = \frac { \mu _ { j } } { \sigma _ { j } + \epsilon }\tag{3}
$$

Here, $X _ { j }$ and y represent the j-th feature and target variable, respectively; $I ( X _ { j } ; y )$ denotes mutual information, $V _ { j k }$ and $\lambda _ { k }$ are PCA loadings and explained variance, $\deg ( X _ { j } )$ represents graph degree, and $\mu _ { j } , ~ \sigma _ { j }$ denote the mean and standard deviation of feature importance across $B$ bootstrap resamples. € is a smoothing constant, while $\alpha , \beta ,$ and $\gamma$ are ensemble weights satisfying $\alpha + \beta + \gamma = 1$

The proposed feature selection framework combines $^ { \mathrm { I G , } }$ PCA, and graph centrality. An undirected feature graph is constructed from the Pearson correlation matrix, where an edge $( i , j )$ is formed when $| \rho _ { i j } | \geq 0 . 5 ,$ and $C _ { j }$ denotes the normalized degree centrality. For each component, $B = 1 0 0$ bootstrap resamples are used to compute the stability-adjusted score $S _ { j } ^ { * } = \mu _ { j } / ( \sigma _ { j } + \epsilon )$ , where $\epsilon = 1 0 ^ { - 6 }$ . The component scores are min-max normalized to $[ 0 , 1 ]$ and combined using equal weights $\alpha = \beta = \gamma = 1 / 3$ , with sensitivity analysis confirming ranking robustness.

SAGE has a computational complexity of $O ( B \cdot n \cdot d +$ $d ^ { 3 } )$ , dominated by bootstrap resampling and PCA. For our dataset $( n = 7 6 6 , d = 4 8 )$ , it completes in under one second on standard hardware and is implemented in Python using scikit-learn and NetworkX.

Figure 2 presents the SAGE-based feature importance ranking, where EPDS Score (0.89) and PHQ-9 Score (0.84) emerge as the most dominant predictors. Other key factors include feeling about motherhood (0.62), occupation (0.51), education (0.50), abuse (0.44), and anger after childbirth (0.45), reflecting the strong impact of psychological and socioeconomic variables. Moderate contributions are observed from family and economic factors such as relationship with in-laws and income (≈ 0.49), while variables like support received, pregnancy loss history, and mode of delivery show lower importance (≈ 0.27–0.29). Overall, psychological and emotional factors drive prediction, with demographic and clinical variables playing a supporting role.

Based on SAGE ranking, the least important features were removed, resulting in a final dataset of 17 features (including the target) with 766 observations. Stratified 5-fold crossvalidation was used for unbiased evaluation, with each fold allocating 80% for training and 20% for testing; 20% of the training portion was reserved for validation. SAGE feature selection, CTGAN oversampling, and GA-ANN tuning were performed exclusively on each training split, with the resulting features and hyperparameters applied unchanged to the validation and test sets. Final performance was averaged across the five test folds.

## C. Model Selection and Hyperparameter Tuning

Using standard Python libraries, we evaluated multiple ML and DL models with optimized hyperparameters to improve prediction performance. Baseline models include Logistic Regression (LR), Support Vector Classifier (SVC), K-Nearest Neighbors (KNN), and Random Forest (RF), all optimized via grid search. An Artificial Neural Network (ANN) with two hidden layers (64, 32 neurons), ReLU activation, dropout (0.3), and Adam optimizer (lr=0.001) was also used for binary classification. ANN outperformed all baseline models on the full feature set, and its performance was further improved through Genetic Algorithm (GA) optimization. The GA optimizes the ANN using 20 individuals over 50 generations, with tournament selection (size 3), single-point crossover $( P _ { c } = 0 . 8 )$ , and Gaussian mutation $( P _ { m } = 0 . 2 )$ . The search covers $n \in \{ 1 6 , 3 2 , 6 4 , 1 2 8 \} , \eta \in [ 1 0 ^ { - 4 } , 1 0 ^ { - 2 } ] , d \in [ 0 . 1 , 0 . 5 ]$ $b \in \{ 1 6 , 3 2 , 6 4 \}$ , and $e \in \{ 5 0 , 1 0 0 , 2 0 0 \}$ , using mean 5-fold CV F1-score as the fitness function.

![](images/fb9291b098472b94edc12f24c01d6e517f8734b59282d93449d593864f9b7e26.jpg)  
Fig. 2. SAGE Feature Importance Ranking for Postpartum Depression Prediction.

The target variable 'Depression during pregnancy (PHQ-$2 ) ^ { \bullet }$ is imbalanced, with 59.53% negative and 40.47% positive cases. To address class imbalance, Conditional Tabular GAN (CTGAN)-based oversampling was applied exclusively within each training fold, leaving the test set untouched for unbiased evaluation. CTGAN was trained for 300 epochs with $2 5 6 \times 1 2 8$ generator/discriminator hidden dimensions, batch size 500, dropout 0.3, and mode-specific normalization for categorical variables, with early stopping based on validation performance. Synthetic samples were generated until a 1:1 minority-to-majority ratio was achieved.

## D. Explainable AI Integration

XAI improves interpretability by explaining predictions. This study uses LIME for instance-level explanations to identify case-specific feature contributions [22]. The LIME model is set up like this:

$$
\hat { g } \left( x \right) = a r g m i n _ { g \epsilon G } L \left( f , g , \pi _ { x } \right) + \Omega \left( g \right)\tag{4}
$$

In LIME, the original complex model is locally approximated by an interpretable surrogate model ${ \hat { g } } ( x ) \in G$ , obtained by minimizing a loss function $L ( f , g , \pi _ { x ^ { \prime } } )$ that balances fidelity to the original model and interpretability.

## E. Evaluation Metrics

The classification models were evaluated using standard performance metrics, including accuracy, precision, recall, F1- score, and the Area Under the Receiver Operating Characteristic Curve (AUC-ROC). These metrics were computed from the confusion matrix, which consists of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN). The AUC-ROC measures the model's ability to distinguish between classes across different decision thresholds, where values closer to 1 indicate better predictive performance.

## IV. RESULTS ANALYSIS AND DISCUSSION

All models were implemented in Python on Google Colab using TensorFlow 2.13, Scikit-learn 1.3, DEAP 1.4, and SDV 1.2 (CTGAN). Stratified 5-fold cross-validation with a fixed random seed of 42 was employed to ensure robust and reproducible evaluation. This section reports the best model, comparisons before and after feature selection, effects of GAN-based imbalance handling, and LIME-based feature explanations.

## A. Performance Evaluation of Baseline Models

Table I shows model performance using the full feature set without oversampling. GA-ANN achieves the best results (79.28% accuracy, 76.52% F1-score, 0.79 AUC), followed by ANN (77.67%, 74.48%, 0.78) and RFC (75.58%, 70.72%. 0.76). Among baseline models, LR performs best (73.43% accuracy), while SVC and KNN show lower performance, highlighting the superiority of GA-based deep learning approaches.

TABLE I  
MODEL PERFORMANCE WITH FULL FEATURE SET AND NO OVERSAMPLING
<table><tr><td>Model</td><td>Precision (%)</td><td>Recall (%)</td><td>F1- score (%)</td><td>Accuracy (%)</td><td>AUC</td></tr><tr><td>LR</td><td>69.31</td><td>65.34</td><td>68.61</td><td>73.43</td><td>0.74</td></tr><tr><td>SVC</td><td>66.87</td><td>64.78</td><td>65.04</td><td>71.54</td><td>0.72</td></tr><tr><td>KNN</td><td>66.35</td><td>64.23</td><td>64.82</td><td>71.21</td><td>0.71</td></tr><tr><td>RFC</td><td>71.36</td><td>69.67</td><td>70.72</td><td>75.58</td><td>0.76</td></tr><tr><td>ANN</td><td>75.12</td><td>73.35</td><td>74.48</td><td>77.67</td><td>0.78</td></tr><tr><td>GA-ANN</td><td>77.06</td><td>75.36</td><td>76.52</td><td>79.28</td><td>0.79</td></tr></table>

## B. Feature Selection and Oversampling for Performance Enhancement

Table II compares model performance under different feature selection methods with GAN-based oversampling. SAGEbased feature selection using 16 features achieves the best results, where GA-ANN attains 87.96% accuracy, 86.32% F1- score, and 0.88 AUC. PCA-IG with 15 features also performs strongly, with GA-ANN reaching 87.42% accuracy and 0.87

AUC. RFE + Tree-based methods using 19 features show slightly lower but competitive performance, with GA-ANN achieving 86.18% accuracy and 0.86 AUC. Overall, GA-ANN consistently outperforms other models, with SAGE providing the most effective feature representation.

TABLE II  
MODEL PERFORMANCE WITH FEATURE SELECTION AND GAN OVERSAMPLING
<table><tr><td>FS Method</td><td>Model</td><td>Precision (%)</td><td>Recall (%)</td><td>F1- score (%)</td><td>Accuracy (%)</td><td>AUC</td></tr><tr><td>RFE +</td><td>ANN</td><td>85.12</td><td>84.94</td><td>85.02</td><td>85.52</td><td>0.86</td></tr><tr><td>Tree</td><td>RFC</td><td>84.77</td><td>84.45</td><td>84.72</td><td>85.31</td><td>0.85</td></tr><tr><td>Based</td><td>GA-ANN</td><td>85.65</td><td>85.38</td><td>85.60</td><td>86.18</td><td>0.86</td></tr><tr><td rowspan="3">PCA- IG</td><td>LR</td><td>84.62</td><td>84.08</td><td>84.47</td><td>85.63</td><td>0.86</td></tr><tr><td>ANN</td><td>85.44</td><td>85.27</td><td>85.32</td><td>86.25</td><td>0.86</td></tr><tr><td>GA-ANN</td><td>86.23</td><td>85.74</td><td>86.17</td><td>87.42</td><td>0.87</td></tr><tr><td rowspan="3">SAGE</td><td>RFC</td><td>84.81</td><td>83.95</td><td>84.67</td><td>85.76</td><td>0.86</td></tr><tr><td>ANN</td><td>85.18</td><td>85.04</td><td>85.10</td><td>86.43</td><td>0.87</td></tr><tr><td>GA-ANN</td><td>86.46</td><td>85.56</td><td>86.32</td><td>87.96</td><td>0.88</td></tr></table>

Figure 3 compares the performance of different machine learning models using three feature selection methods (RFE+Tree, PCA-IG, and SAGE) after GAN-based oversampling. The evaluation includes precision, recall, F1-score, accuracy, and AUC. Overall, the SAGE feature selection method consistently achieved the strongest performance, with the GA-ANN model producing the highest results across all evaluation metrics, including 87.96% accuracy and an AUC of 0.88. These findings demonstrate that combining SAGE feature selection with GA-ANN provides the most effective classification performance among the evaluated approaches.

![](images/bef558daa0cff4275b06230afea9d5a1860bccc83e25366243a0a9be16276a23.jpg)  
Fig. 3. Comparison of Model Performance under Different Feature Selection Methods with GAN Oversampling.

## C. LIME-Based Insights into Model Decisions

The impact of features on predictions is shown using LIME tabular and feature importance plots based on SAGEselected features and the GA-ANN model. Figure 4 shows a LIME-based explanation where the model predicts Class 0 with 69% probability and Class 1 with 31%. Key protective factors include relationship with in-laws (0.94), income (0.72), age of older children (0.70), education (0.56), and husband's education (0.54), while risk factors include poor sleep (-1.74), abuse (-0.97), EPDS score (-0.66), occupation after childbirth (-0.50), and feeling about motherhood (-0.42). Overall, protective socioeconomic factors outweigh psychosocial stressors, leading to a final prediction of Class 0.

![](images/86a002f1d952a7b3c8ae47c25d889530f67b3f4ffd20406f0e3ffd775ec0ccba.jpg)  
Fig. 4. LIME-Based Local Explanation of Model Prediction.

Figure 5 presents the LIME feature importance plot for a Class 1 prediction, where key risk factors such as prior depression (PHQ2 ≈ -0.16), education level (≈ -0.11), income, PHQ-9 (≈ −0.03), EPDS (≈ −0.02), and poor sleep (≈ —0.02) drive the model toward higher risk. In contrast, factors like total children (≈ +0.06), husband's education (≈ +0.04), and post-childbirth occupation (≈ +0.04) contribute toward Class 0. Overall, negative psychological and lifestyle factors dominate, leading to a higher-risk prediction (Class 1).

![](images/46fc00478927b479f43db217c48435cfea76afa62db89ee8773278104da40b16.jpg)  
Fig. 5. Instance-Level Feature Importance Using LIME.

Compared with existing studies, SAGE provides a strong balance of predictive performance, feature parsimony, stability, and interpretability. Zhang et al. [17] reported an AUC of 0.849 using 17 predictors, whereas SAGE achieves an AUC of 0.88 with only 16 features. Cellini et al. [8] selected 11 predictors using LASSO-Boruta but did not assess selection stability, which SAGE addresses through bootstrap-weighted ensemble selection. Although Liu et al. [14] reported up to 95.0% accuracy on cesarean-specific cohorts, SAGE is evaluated on a more heterogeneous primary cohort. Moreover, SAGE combines graph-based feature selection with local LIME explanations, extending beyond the global SHAP-based interpretability used in prior studies [8], [14], [23]. SAGE demonstrates improved methodological rigor while maintaining competitive predictive performance and clinically relevant interpretability.

## V. CONCLUSION AND FUTURE WORK

This study presents SAGE, an ensemble feature selection framework for PPD prediction that incorporates a GAoptimized ANN and a local XAI. By combining multiple selection methods with stability weighting, the model identifies a reliable set of key features from high-dimensional data. The proposed system addresses the class imbalance and achieves better accuracy than traditional methods. Our findings indicate that psychological and sociodemographic factors are the main risk factors of PPD. Moreover, LIME explanations enhance the interpretability by providing patient-specific insights, which makes it suitable for clinical decision support for maternal health.

The framework should be tested across regions to assess whether similar patterns exist among women in South Asia and Africa. Longitudinal analysis of the prenatal and postnatal periods can improve the understanding of temporal risk. Multimodal data (EHR, wearable data and clinical text) can be used to enhance predictive performance, while stability-aware feature selection strategies can still be applied. The system should be extended to other perinatal mental health problems and clinical systems.

## DATA AVAILABILITY

The dataset is available on Mendeley at the following link: https://data.mendeley.com/datasets/nzgnsrgsg5/1

## REFERENCES

[1] S. A. Amer, N. A. Zaitoun, H. A. Abdelsalam, A. Abbas, M. S. Ramadan, H. M. Ayal, S. E. A. Ba-Gais, N. M. Basha, A. Allahham, E. B. Agyenim et al., “Exploring predictors and prevalence of postpartum depression among mothers: Multinational study," BMC Public Health, vol. 24, no. 1, p. 1308, 2024.

[2] WorldHealthOrganization,"Maternalmental health," https://www.who.int/teams/mental-health-and-substance-use/ promotion-prevention/maternal-mental-health, n.d., accessed: 2026-04- 19.

[3] T. Pan, Y. Zeng, X. Chai, Z. Wen, X. Tan, and M. Sun, "Global prevalence of perinatal depression and its determinants among rural women: A systematic review and meta-analysis," Depression and Anxiety, vol. 2024, no. 1, p. 1882604, 2024.

[4] X.-W. Yang, X.-L. Jiang, and Y.-L. Wu, "Clinical investigation of postpartum depression risk factors and screening predictors," World Journal of Psychiatry, vol. 16, no. 2, p. 113101, 2026.

[5] Z. Wang, J. Liu, H. Shuai, Z. Cai, X. Fu, Y. Liu, X. Xiao, W. Zhang, E. Krabbendam, S. Liu et al., "Mapping global prevalence of depression among postpartum women," Translational psychiatry, vol. 11, no. 1, p. 543, 2021.

[6] X. Huang, L. Zhang, C. Zhang, J. Li, and C. Li, "Postpartum depression risk prediction using explainable machine learning algorithms," Frontiers in Medicine, vol. 12, p. 1565374, 2025.

[7] G. M. I. Alam, T. Biswas, S. A. Tanim, and M. Mridha, "An explainable analytics framework for predicting diabetes in women using convolutional neural networks," Healthcare Analytics, p. 100422, 2025.

[8] P. Cellini, A. Pigoni, G. Delvecchio, C. Moltrasio, and P. Brambilla, “Machine learning in the prediction of postpartum depression: A review," Journal of Affective Disorders, vol. 309, pp. 350–357, 2022.

[9] J. S. Mathew and G. Ramasamy, "Graph convolutional networks for predicting postpartum depression: A symptom-based analysis," in 2025 International Conference on Emerging Technologies in Computing and Communication (ETCC). IEEE, 2025, pp. 1–6.

[10] N. Harshitha, K. Reethika, S. Krishnaveni, and M. Sahu, "Emocare: Xai enabled emotion detection and health guidance for pregnancy and postpartum wellness," in 2025 8th International Conference on Trends in Electronics and Informatics (ICOEI). IEEE, 2025, pp. 743–749.

[11] M. A. Clapp, V. M. Castro, P. Verhaak, T. H. McCoy, L. L. Shook, A. G. Edlow, and R. H. Perlis, “Stratifying risk for postpartum depression at time of hospital discharge," American Journal of Psychiatry, vol. 182, no. 6, pp. 551–559, 2025.

[12] Y. Wang, P. Yan, G. Wang, Y. Liu, J. Xiang, Y. Song, L. Wei, P. Chen, and J. Ren, “Trajectory on postpartum depression of chinese women and the risk prediction models: A machine-learning based three-wave follow-up research," Journal of Affective Disorders, vol. 365, pp. 185– 192, 2024.

[13] C. Wakefield and M. G. Frasch, “Predicting patients requiring treatment for depression in the postpartum period using common electronic medical record data available antepartum," AJPM focus, vol. 2, no. 3, p. 100100, 2023.

[14] H. Liu, A. Dai, Z. Zhou, X. Xu, K. Gao, Q. Li, S. Xu, Y. Feng, C. Chen, C. Ge et al., “An optimization for postpartum depression risk assessment and preventive intervention strategy based machine learning approaches," Journal of Affective Disorders, vol. 328, pp. 163–174, 2023.

[15] Z. Ma, M. Horvath, D. M. Stamilio, K. Sekyere, and M. N. Gurcan, “Building a machine learning model to predict postpartum depression from electronic health records in a tertiary care setting," Journal of Clinical Medicine, vol. 14, no. 18, p. 6644, 2025.

[16] M. Alkhateeb, A. Nayeem, A. Ahmed, M. Alsahli, J. Sheikh, and A. Abd-Alrazaq, “Ai for detecting and predicting postpartum depression: Scoping review," Journal of Medical Internet Research, vol. 28, p. e77376, 2026.

[17] R. Zhang, Y. Liu, Z. Zhang, R. Luo, and B. Lv, "Interpretable machine learning model for predicting postpartum depression: retrospective study," JMIR Medical Informatics, vol. 13, p. e58649, 2025.

[18] J. Xia, C. Chen, X. Lu, T. Zhang, T. Wang, Q. Wang, and Q. Zhou, "Artificial intelligence-oriented predictive model for the risk of postpartum depression: a systematic review," Frontiers in Public Health, vol. 13, p. 1631705, 2025.

[19] W. Qi, Y. Wang, Y. Wang, S. Huang, C. Li, H. Jin, J. Zuo, X. Cui, Z. Wei, Q. Guo et al., "Prediction of postpartum depression in women: development and validation of multiple machine learning models," Journal of Translational Medicine, vol. 23, no. 1, p. 291, 2025.

[20] S. Wang, R. Xu, G. Li, S. Liu, J. Zhu, and P. Gao, "A plasma proteomicsbased model for identifying the risk of postpartum depression using machine learning," Journal of Proteome Research, vol. 24, no. 2, p. 824, 2025.

[21] M. E. Hossain, B. C. Das, S. S. Ayon, J. Mia, M. M. Hasan, and M. Khan, “Explainable ai for nafld prediction using clinical and lifestyle data," in 2026 5th International Conference on Electrical, Computer & Telecommunication Engineering (ICECTE). IEEE, 2026, pp. 1–6.

[22] S. S. Ayon, A. Al Mamun, M. E. Hossain, W. Alamro, Y. M. Allawi, N. N. I. Prova, M. S. U. Miah, S. M. Sultan, and A. Abadleh, "Explainable ai framework for improved thalassemia mental health classification and feature selection," PLoS One, vol. 21, no. 1, p. e0341168, 2026.

[23] K. S. Sharif, M. Abubakkar, M. M. Uddin, and A. A. Khaled, "A comparative framework integrating hybrid convolutional and unified graph neural networks for accurate parkinson's disease classification," in 2024 7th International Seminar on Research of Information Technology and Intelligent Systems (ISRITI). IEEE, 2024, pp. 31–37.