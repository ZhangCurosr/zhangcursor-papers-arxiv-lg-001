# Beyond Binary Classification: Multiclass Machine Learning for Diabetes and Prediabetes Risk Prediction Using NHANES

Aman Mujahid<sup>a</sup>, Anita Penkova<sup>b</sup>

<sup>a</sup>Thomas Lord Department ofComputer Science, Viterbi School ofEngineering, University ofSouthern California, Los Angeles, 90089, CA, USA <sup>b</sup>Department of Aerospace and Mechanical Engineering, Viterbi School of Engineering, University of Southern California, Los Angeles, 90089, CA, USA

## Abstract

Detecting diabetes risk early is important for preventing disease progression. Most existing machine learning approaches focus6 on binary classification, which limits their ability to capture intermediate risk states such as prediabetes. We develop a supervised learning framework to classify adults as normal, prediabetic, or diabetic using National Health and Nutrition Examination Survey<sup>0</sup> data. The framework uses routinely collected demographic, clinical, and lifestyle variables from 15,356 adults, with glycated hemoglobin (HbA1c) used only to define labels and excluded from input features to prevent leakage. We compare conventional<sup>v</sup> machine learning and transformer-based models and examine two binary formulations: normal versus prediabetes, and normal<sup>o</sup> versus combined prediabetes and diabetes. Models are evaluated on a held-out test set using precision, sensitivity, specificity, and F1 score, with 5-fold cross-validation and 95% confidence intervals assessing stability. In the multiclass setting, performance is broadly consistent across models, with accuracy up to 0.72 and macro-averaged F1 of 0.70. Performance is stronger for the norma and diabetes categories, with normal-class F1 reaching 0.79 and diabetes sensitivity reaching 0.80, and XGBoost achieving a diabetes precision of 0.73 and specificity of 0.96. Prediabetes remains the most challenging class, with F1 scores of 0.54 to 0.56 fo conventional models and 0.53 to 0.55 for transformers, reflecting its overlap with both neighboring states. For this class, XGBoost achieves a precision of 0.49, sensitivity of 0.65, and specificity of 0.74. When the task is reduced to normal versus prediabetes, F1 scores rise modestly to 0.62 to 0.64. A more substantial improvement is observed under a broader screening formulation combining<sup>s</sup> prediabetes and diabetes, with F1 scores of 0.73 to 0.74 across all models and XGBoost reaching a sensitivity of 0.78 and specificity of 0.77. These findings suggest that feature overlap along the biological continuum between normal and diabetic states contributes more to the dificulty of prediabetes detection than model architecture. Multiclass prediction provides a more clinically informative representation of metabolic risk progression but highlights a trade-of between clinically meaningful stratification and predictive accuracy.

Keywords: Prediabetes, Diabetes Risk Prediction, Multiclass Classification, Machine Learning, NHANES, Gradient Boosting, Tabular Transformers

## 1. Introduction

Diabetes mellitus remains a major public health challenge worldwide, with prevalence continuing to increase due to demographic shifts and behavioral risk factors. Although longterm complications of diabetes, which include cardiovascular<sub>a</sub> disease, kidney dysfunction, and vision impairment, are well documented, the condition often advances quietly over time. A lot of individuals pass through a prediabetic phase marked by impaired glucose regulation and few or no symptoms. As a result, many people who are already at elevated risk remain unaware of their condition to the point where it reaches an advanced stage. Detecting abnormal patterns in blood glucose earlier can make a big diference by enabling timely preventive interventions.

Prediabetes represents a state of transition between normal glucose control and diabetes, where early changes in behavior and daily habits can reduce the risk of progression. Evidence from the Diabetes Prevention Program demonstrated that both lifestyle intervention and metformin reduced the incidence of type 2 diabetes among individuals at elevated risk, with lifestyle intervention producing the greater reduction [1]. Despite its clinical importance, this condition is frequently overlooked since routine screening practices often merge it with normal glycemic status. Therefore, many predictive models frame diabetes risk as a binary problem, diferentiating only the presence and absence of disease. Keeping prediabetes separate, as in treating it as a distinct category, paints a clearer picture of the risk and opens the door to faster treatment.

Glycemic status can be assessed using several commonly collected biomarkers such as fasting plasma glucose, oral glucose tolerance tests, and glycated hemoglobin (HbA1c). Clinical guidelines define clear HbA1c ranges corresponding to normal glycemic control, prediabetes, and diabetes [2], which makes it a practical tool for classifying individuals within a large population. For models designed to assess risk before a clinical di agnosis is made, however, including HbA1c as a predictor can leak information that directly defines the outcome, making performance look stronger than it would in a real screening setting.

Machine learning methods have been used for quite a while to assess diabetes risk by detecting patterns across a wide range of demographic, clinical, and behavioral information. Compared with conventional statistical models, these methods can better capture complex and nonlinear associations in observational health data. Although there have been great advances in machine learning for the field of diabetes prediction, most studies that have been done so far rely on binary classification, limiting their ability to model intermediate metabolic states like prediabetes.

To fill these gaps, we used data from the United States National Health and Nutrition Examination Survey (NHANES), a large observational dataset that includes demographic, clinical, and lifestyle information from adults across the general population. Using this data, we developed a supervised learning framework that classifies individuals as normal, prediabetic, or diabetic based on established HbA1c criteria, while excluding HbA1c from the input features to prevent label leakage. We evaluated multiple gradient boosting models, a soft voting ensemble, and three transformer-based architectures, and reported macro-averaged metrics to focus on balanced performance across classes. We also ran an exploratory subgroup analysis with fairness-aware reweighting to check how consistently prediabetes is detected among the demographic groups. By comparing multiclass and binary screening tasks, we tested whether the main challenge comes from model capacity or from overlap between adjacent glycemic states. This work makes the following contributions:

• We show that prediabetes remains dificult to separate from adjacent glycemic states across both conventional machine learning models and transformer-based architectures, suggesting that a substantial portion of the limitation arises from feature overlap within the metabolic continuum rather than insuficient model complexity.

• We show that prediction performance is influenced more strongly by problem formulation than by model selection, with substantial gains observed when prediabetes and diabetes are merged into a broader abnormal glycemic category.

• We distinguish realistic screening-oriented prediction from diagnostic reconstruction by excluding HbA1c from the predictor set, allowing model performance to reflect information available prior to formal diagnosis.

## 2. Related Work

## 2.1. Machine Learning Approaches for Diabetes Risk Prediction

Much of the machine learning work on diabetes has focused on finding algorithms that can improve prediction accuracy and better separate individuals with diabetes from those without it. For instance, Xu and Wang combined random forest based feature selection with XGBoost and showed that this approach performed better than several standard classification methods on the Pima Indians Diabetes dataset [3]. Deberneh and Kim used electronic health records to predict whether patients would be normal, prediabetic, or diabetic in the following year, and reported that ensemble models performed better than individual classifiers [4]. More recent studies have also used large population-level datasets such as NHANES, where XGBoost achieved the highest AUC among several compared classifiers [5; 6], and others have combined several models through ensemble and stacking frameworks [7; 8]. Collectively, these studies demonstrate that boosting based methods consistently achieve strong performance on large population datasets such as NHANES. However, despite diferences in algorithms and cohorts, they almost universally formulate diabetes prediction as a binary classification problem, leaving intermediate metabolic states comparatively underexplored.

## 2.2. Machine Learningfor Early Detection ofPrediabetes

Recent work has increasingly focused on identifying prediabetes before it progresses to established diabetes. Kopitar et al. used electronic health record data to detect impaired fasting glucose and undiagnosed diabetes [9], Kulkarni et al. used electrocardiography measurements to detect patterns associated with prediabetes and diabetes [10], and Li et al. found that XGBoost could provide an interpretable estimate of prediabetes risk from population-survey variables [11]. More recent studies have treated prediabetes more directly as its own prediction target. Hu et al. developed a nomogram for predicting 5-year incident prediabetes in Chinese adults [12], and Almadhoun and Burhanuddin compared several classifiers for distinguishing normal from prediabetic cases [13]. Fan et al. extended this direction to multiclass diagnosis by separating nondiabetic, prediabetic, and diabetic patients with an interpretable residual-network model [14]. Their analysis also showed that prediabetes overlaps with neighboring glycemic states in the raw feature space, although their model, which used HbA1c as an input, still separated the three classes with high accuracy. However, many approaches either predict future onset, combine abnormal glycemic states into a single group, or retain highly diagnostic biomarkers such as HbA1c as input features.

## 2.3. Transformer-Based and Attention-Based Models for Dia betes Prediction

Recent studies have explored transformer-based architectures for diabetes prediction using structured clinical data. Tab-Transformer has been applied to binary diabetes classification using tabular medical variables [15; 16], and tabular transformers have been extended to more specialized settings, includ ing longitudinal clinical and DXA-derived features [17] and metagenomic profiles [18]. Oba et al. applied TabNet to annual health-screening records to predict whether Type 2 diabetes status would worsen over time [19]. Their results also showed that strong boosting methods remained competitive, suggesting that attention-based models do not necessarily outperform conventional ensemble methods on structured healthcare data. It remains unclear whether transformer-based models can better capture the overlapping feature patterns associated with prediabetes while avoiding direct label leakage from HbA1c.

## 2.4. Fairness and Bias in Healthcare Machine Learning

Concerns about fairness are particularly relevant in the case of diabetes risk prediction. When certain populations are underrepresented in training datasets, prediction models may produce less reliable results for those groups, and several researchers have emphasized that evaluating algorithmic fairness should be considered an important step when developing machine learning systems for healthcare applications [20]. Existing prediction tools such as the Framingham Ofspring Risk Score and the Prediabetes Risk Test have been shown to produce diferent levels of risk estimation for diferent populations when evaluated using large national datasets such as NHANES [21]. Other work has also highlighted the importance of examining error rates and calibration across demographic groups when evaluating healthcare prediction models [22]. Fairness evaluation also involves inherent trade-ofs, since diferent group-level criteria cannot always be satisfied simultaneously [23].

## 3. Methods

## 3.1. Dataset

We used data from the National Health and Nutrition Examination Survey (NHANES), a program conducted by the National Center for Health Statistics that combines interviews, physical examinations, and laboratory testing to assess the health and nutritional status of people in the United States [24]. For this study, we combined six NHANES cycles spanning 2007-2008 through 2017-2018. Data from the diferent NHANES components were merged using the participant identifier SEQN.

The dataset included demographic details such as age, sex, and race or ethnicity, body measurements like body mass index and waist circumference, blood pressure readings, and laboratory measurements including HbA1c, fasting glucose, and lipid-related variables. We also used questionnaire data that capture lifestyle factors such as smoking, physical activity, sedentary behavior, and family history of diabetes. Race and ethnicity data were taken from the RIDRETH1 variable and harmonized across cycles. This variable classifies participants as Mexican American, Other Hispanic, Non-Hispanic White, Non-Hispanic Black, or Other Race Including Multi-Racial, with Non-Hispanic Asian participants grouped within the Other Race category. For blood pressure, repeated systolic and diastolic measurements were averaged to get a single value for each participant. After merging all the selected components, the dataset contained a total of 19,307 participants.

Participants missing values in key variables, including age, body mass index, systolic blood pressure, diastolic blood pressure, and HbA1c, were removed. We also limited the dataset to adults aged 18 years and older. After applying these filtering steps, the final cohort used for the multiclass formulation included 15,356 individuals.

Glycemic status was defined using HbA1c thresholds based on standard clinical guidelines [2]. For the primary multiclass formulation, individuals with HbA1c below 5.7% were labeled as normal, individuals with $5 . 7 \% < = \mathrm { H b A l c } < 6 . 5 \%$ were labeled as prediabetic, and individuals with HbA1c >= 6.5% were labeled as diabetic. This resulted in 9,249 normal cases, 4,279 prediabetes cases, and 1,828 diabetes cases, representing 60.2%, 27.9%, and 11.9% of the final cohort, respectively.

<table><tr><td>Variable Type</td><td>Category</td><td>Variables</td></tr><tr><td>Continuous Vari- ables</td><td>Demographics</td><td>Age</td></tr><tr><td rowspan="5"></td><td>Body Measure-</td><td>Body Mass Index, Waist</td></tr><tr><td>ments Blood Pressure</td><td>Circumference Systolic Blood Pressure,</td></tr><tr><td></td><td>Diastolic Blood Pressure</td></tr><tr><td>Laboratory</td><td>Fasting Glucose, Total Cholesterol, HDL, LDL,</td></tr><tr><td>Lifestyle</td><td>Triglycerides Sedentary Behavior</td></tr><tr><td>Categorical Vari- ables</td><td>Demographics</td><td>Sex, Race or Ethnicity</td></tr><tr><td></td><td>Lifestyle</td><td>Smoking Status, Physical</td></tr><tr><td></td><td>Family History</td><td>Activity Indicators Family History of Diabetes</td></tr></table>

Table 1: Continuous and categorical variables used in the dataset
<table><tr><td>Class</td><td>Samples</td><td>Percentage</td></tr><tr><td>Normal</td><td>9,249</td><td>60.2%</td></tr><tr><td>Prediabetes</td><td>4,279</td><td>27.9%</td></tr><tr><td>Diabetes</td><td>1,828</td><td>11.9%</td></tr></table>

Table 2: Class distribution

Since HbA1c was used to define the outcome labels, it was excluded from the predictor set during model training to avoid label leakage. Fasting glucose was kept as a predictor, even though it is also a glycemic biomarker. Unlike HbA1c, it was not used to set the class boundaries, which means keeping it does not put the outcome definition back into the input features. HbA1c tracks average glucose over several months, whereas fasting glucose is a single reading taken at one point in time. Therefore it is connected to the labeling criterion without being equivalent to it, and it is the kind of measurement that is usu ally on hand in a screening setting. One complication is that NHANES measures fasting glucose only in the morning fasting subsample, so it is missing for a substantial proportion of participants. Rather than restricting the cohort to fasting participants, which would have shrunk the sample and altered its compo sition, we imputed missing values using the same median ap proach applied to the other continuous variables. A summary of the continuous and categorical variables used in this study is shown in Table 1, and the distribution of the target classes in the final dataset is presented in Table 2.

## 3.2. Experimental Pipeline

## 3.2.1. Feature Processing and Encoding

Missing values in continuous variables were filled using the median value of the corresponding feature, since median imputation is relatively robust to outliers and better preserves the distribution of skewed clinical variables compared to mean imputation. The median values were computed on the full cohort before the split between the train and test and since they do not use outcome information, this does not introduce label leakage. Although alternative imputation strategies may influence model behavior to some extent, evaluating sensitivity to diferent imputation methods was outside the scope of the current study. For categorical variables, missing entries were treated as a separate category rather than removing those samples from the dataset.

Variables such as sex, race or ethnicity, smoking behavior, physical activity indicators, and family history of diabetes were converted into numerical form using integer-based label encoding. Label encoding was selected primarily because several of the models used in this study are tree-based methods, which accommodate integer-encoded categorical variables directly without relying on Euclidean distances, and it also helped keep the feature space compact compared to one-hot encoding.

Continuous variables were standardized after preprocessing and feature construction. While tree-based models are generally less afected by feature scaling, standardization was retained because transformer based architectures benefited from normalized continuous features, while applying the same preprocessing to all models ensured consistency across experiments.

## 3.2.2. Feature Engineering

Additional features were created to capture metabolic and cardiovascular patterns that may not be fully reflected by the original variables alone. The triglyceride-to-HDL ratio was included because high triglycerides combined with low HDL cholesterol are often linked to insulin resistance. Although both variables were already included separately, the ratio captures their combined efect and may represent metabolic risk more directly. Pulse pressure was also calculated from systolic and diastolic blood pressure because the gap between these values may reflect vascular stress that is not fully captured by either measure alone. Interaction terms between age and body mass index, and between age and waist circumference, were added to capture how body composition may afect diabetes risk diferently across age groups.

Several binary clinical indicators were also created, including obesity status, elevated blood pressure, elevated fasting glucose, high triglycerides, and low HDL cholesterol. Elevated fasting glucose (≥100 mg/dL) and elevated blood pressure (≥130/80 mmHg) followed American Diabetes Association (ADA) criteria [2], while obesity was defined as BMI ≥30 kg/m<sup>2</sup>, high triglycerides as ≥150 mg/dL, and low HDL cholesterol as <40 mg/dL. Although derived from existing variables, these indicators are not fully redundant because they represent clinically meaningful cut points that may help separate risk groups. The same engineered feature set was used across all prediction tasks to keep the multiclass and binary formulations comparable.

## 3.2.3. Train-Test Splitting and Class Weighting

The dataset was divided into training and testing sets using a 75/25 split, with stratified sampling applied to preserve the original class distribution across the glycemic categories. A fixed random seed of 42 was used throughout the experiments to support reproducibility.

Class imbalance was addressed using weights calculated from the training data, giving greater influence to underrepresented classes during model optimization, as recommended for learning from skewed class distributions [25]. Class weights were computed using the balanced formulation implemented in scikit-learn, where each class weight is inversely proportional to its frequency in the training data. Oversampling methods such as SMOTE were not applied because the goal was to evaluate model performance under the natural distribution of the NHANES cohort, and synthetic samples could introduce artificial combinations of demographic, metabolic, and lifestyle characteristics that may not reflect real participants.

## 3.2.4. Experimental Formulations

All prediction tasks in this study were built using the same preprocessing pipeline, feature set, train-test splitting strategy, and evaluation procedure. The main formulation treated glycemic prediction as a three-class classification problem involving normal, prediabetes, and diabetes categories.

To further examine the dificulty of identifying intermediate metabolic states, two additional binary settings were evaluated. The first focused only on distinguishing normal individuals from those with prediabetes, excluding diabetic participants from the dataset. It is intended as a controlled contrast that isolates the boundary between normal and prediabetic states, not as a standalone screening model, since diabetic individuals cannot be identified and excluded in advance in a real screening setting. Secondly, we combined prediabetes and diabetes into a single abnormal glycemic category and compared it against normal individuals in a broader screening-oriented setting.

## 3.3. Modeling Approach

## 3.3.1. XGBoost

We used XGBoost [26] as the primary model in this study because of its strong performance on structured tabular data and its ability to capture nonlinear relationships and feature interactions across demographic, clinical, and lifestyle variables.

Hyperparameters were selected by randomized search with 5-fold cross-validation, using a fixed seed of 42 and balanced sample weights during fitting. Table 3 reports the full search space. The binary formulations used shallower trees, lower learning rates, and a wider regularization space than the mul ticlass setting, and shared identical grids with each other. For the multiclass XGBoost model, the search was run for 40 iterations and optimized using macro F1. For the two binary formulations, the search was run for 80 iterations and optimized using positive-class F1, where the positive class was prediabetes in the normal versus prediabetes task and the combined abnormal group in the normal versus abnormal glycemic state task. Model tuning was done only on the training data, and final performance was evaluated on the separate test set.

<table><tr><td>Hyperparameter</td><td>Multiclass</td><td>Binary Formulations</td></tr><tr><td>Tree depth</td><td>3,4,5</td><td>2,3,4</td></tr><tr><td>Learning rate</td><td>0.03, 0.05, 0.07</td><td>0.01, 0.03, 0.05</td></tr><tr><td>Number of trees</td><td>400, 600, 800</td><td>500, 700, 900</td></tr><tr><td>Row subsampling</td><td>0.8, 0.9, 1.0</td><td>0.7, 0.8, 0.9</td></tr><tr><td>Column subsampling</td><td>0.8, 0.9, 1.0</td><td>0.7, 0.8, 0.9</td></tr><tr><td>Minimum child weight</td><td>1,3,5</td><td>3,5,7, 10</td></tr><tr><td>Minimum split gain</td><td>0,0.1, 0.3</td><td>0.1, 0.3, 0.5, 1.0</td></tr><tr><td>L1 regularization</td><td></td><td>0, 0.5, 1, 2</td></tr><tr><td>L2 regularization</td><td></td><td>1,3,5, 10</td></tr><tr><td>Maximum delta step</td><td></td><td>0,1,3,5</td></tr></table>

Table 3: Hyperparameter search space for the primary XGBoost model. Dashes indicate parameters not included in the corresponding search.

## 3.3.2. Fairness-Aware Reweighting and Baseline Model

We evaluated a fairness-aware weighting strategy for the primary multiclass XGBoost model, in which sample weights were adjusted using both class labels and race or ethnicity group membership. For comparison, a baseline multiclass XGBoost model was trained using the same data split and hyperparameter settings but with only standard class-based weighting. This allowed the impact of fairness-aware reweighting to be evaluated independently from other modeling changes.

## 3.3.3. Comparison Models and Ensemble Methods

LightGBM [27] and CatBoost [28] were included as additional gradient boosting models. To keep the comparison across boosting frameworks fair, both were tuned using the same randomized search procedure applied to XGBoost, with the XG-Boost ranges mapped onto the corresponding hyperparameters in each library and the same cross-validation scheme, scoring metric, and class-weighting strategy. To limit computational cost, LightGBM and CatBoost used reduced grid density and sampling budgets. LightGBM sampled 25 configurations in the multiclass setting and 40 in the binary formulations, and CatBoost sampled 20 in all settings. Since LightGBM grows trees leaf-wise, the number of leaves was tuned jointly with tree depth. CatBoost has no counterpart to minimum split gain, maximum delta step, or L1 regularization, and its leaf-size constraint was omitted to limit search complexity. CatBoost also handled categorical variables directly through its native categorical feature processing. HistGradientBoosting, Random Forest [29], and Logistic Regression were not tuned and instead served as fixed baselines, ofering reference points that do not depend on the tuning budget. Class imbalance was handled the same way across all models.

We also explored a soft voting ensemble approach by averaging the predicted class probabilities from the primary boosting models, with final predictions assigned based on the highest combined probability score.

## 3.3.4. Transformer-Based Modelsfor Tabular Data

To broaden the comparison beyond conventional machine learning models, we evaluated three transformer-based architectures designed for structured tabular data, to examine whether attention mechanisms could capture relationships among demographic, clinical, and lifestyle variables that may not be fully represented by tree-based methods. TabTransformer uses self-attention to learn contextual representations of categorical variables [30]. FT-Transformer extends this idea by representing both categorical and continuous variables as feature tokens and applying attention across the full feature sequence [31]. SAINT further incorporates attention across both features and samples, allowing each patient representation to draw context from other records within the same batch [32]. We used a supervised SAINT variant without contrastive pretraining. Each transformer architecture was evaluated across all three prediction settings.

All transformer models used the same leakage-aware predictor set as the conventional models. Categorical variables were integer-encoded and mapped to learned embeddings, while continuous variables were standardized using statistics calculated from the training data. Class imbalance was addressed using weighted cross-entropy loss based on the class distribution in the training set. We optimized all transformer models with AdamW using a learning rate of $1 0 ^ { - 3 } .$ , weight decay of $1 0 ^ { - 4 }$ a batch size of 256, and a dropout rate of 0.2. Each model used a multilayer perceptron classification head, applied to the flattened categorical embeddings combined with the continuous variables for TabTransformer and to a learnable classification token for FT-Transformer and SAINT. The main architectural and training settings for each model are summarized in Table 4.

## 3.3.5. Model Interpretation

To better understand model behavior, we used SHAP [33] to examine feature contributions, focusing primarily on the prediabetes class. SHAP values were computed on a subset of the test data to keep computation manageable while still capturing overall model behavior, and summary plots and mean absolute SHAP values were used to identify the most influential predictors. Because post-hoc explanations do not provide a complete or causal account of model behavior, and interpretability itself can be defined in diferent ways, the results were interpreted cautiously [34; 35]. In particular, correlated metabolic and anthropometric features may redistribute importance across related variables, which can influence the ranking of individual predictors without necessarily changing the overall predictive behavior of the model.

## 3.4. Evaluation Metrics

Model performance was evaluated using accuracy, precision, recall, specificity, and F1 score. These measures are computed from the number of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN):

$$
\mathrm { P r e c i s i o n } = { \frac { T P } { T P + F P } }\tag{1}
$$

$$
\mathrm { R e c a l l } = { \frac { T P } { T P + F N } }\tag{2}
$$

$$
{ \mathrm { S p e c i f i c i t y } } = { \frac { T N } { T N + F P } }\tag{3}
$$

Recall is equivalent to sensitivity. For the multiclass task, sensitivity and specificity are computed separately for each

<table><tr><td>Model</td><td>Feature Representation</td><td>Transformer Configuration</td><td>Training</td></tr><tr><td>TabTransformer</td><td>16-d categorical embeddings with continuous features</td><td>2 encoder layers, 4 heads</td><td>40 epochs, batch size 256</td></tr><tr><td>FT-Transformer</td><td>32-d categorical and continuous feature tokens</td><td>3 encoder layers, 4 heads</td><td>50 epochs, batch size 256</td></tr><tr><td>SAINT</td><td>32-d feature tokens with feature and sample attention</td><td>3 attention stages, 4 heads</td><td>50 epochs, batch size 256</td></tr></table>

Table 4: Summary of transformer model architectures and training configurations.

class, treating that class as positive and the remaining two as negative. The F1 score combines precision and recall as their harmonic mean:

$$
\mathrm { F 1 } = 2 \cdot \frac { \mathrm { P r e c i s i o n } \cdot \mathrm { R e c a l l } } { \mathrm { P r e c i s i o n } + \mathrm { R e c a l l } }\tag{4}
$$

Because the dataset is imbalanced, particularly for the diabetes category, we treat macro-averaged F1 as the primary metric:

$$
\mathrm { M a c r o - F 1 } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { F } 1 _ { i }\tag{5}
$$

Here, N is the number of classes and $\mathrm { F } 1 _ { i }$ is the F1 score of class i. Unlike weighted F1, macro F1 averages the per-class scores without weighting by class frequency, so every glycemic state carries equal weight and strong results on the majority normal class cannot mask weak detection of the smaller prediabetes and diabetes groups. We also report balanced accuracy, computed as the average recall across classes, as a complement to macro F1.

Class-wise precision, recall, and F1 scores were additionally reported, and confusion matrices were used to identify common misclassification patterns, particularly whether prediabetes cases were being confused with normal or diabetic samples. For the binary formulations, precision, recall, and F1 score were reported with respect to the positive class.

To examine the consistency of model performance across diferent data splits, we also performed 5-fold stratified crossvalidation using macro F1 score as the primary metric, and approximate 95% confidence intervals were calculated from the fold-level macro F1 scores. As shown in Table 5, the treebased and boosting models performed similarly, with narrow and largely overlapping confidence intervals that point to stable behavior across diferent partitions of the dataset. Logistic Regression was the exception. Its mean macro F1 of 0.6662 sits below the other models and its confidence interval does not overlap with them, which fits the expectation that a linear model captures less of the nonlinear structure separating the three glycemic classes. Since the intervals overlap so heavily among the tree-based and boosting models, the small gaps in macro F1 among them are best read as descriptive rather than as firm evidence that any one model outperforms the rest.

## 4. Results

## 4.1. Multiclass Model Performance

We evaluated all models on a separate test set. Overall, performance was consistent across models, with accuracy ranging from 0.69 to 0.72 and macro F1 scores from 0.68 to 0.70. Bal anced accuracy also remained close to standard accuracy across models, ranging from 0.71 to 0.72. This suggests that the reported performance was not driven only by the majority normal class, but reflected relatively balanced recall across the three glycemic groups.

<table><tr><td>Model</td><td>Macro F1</td><td>95% Confidence Interval</td></tr><tr><td>XGBoost</td><td>0.6980 ± 0.0076</td><td>[0.6913, 0.7047]</td></tr><tr><td>LightGBM</td><td> $0 . 6 9 3 0 \pm 0 . 0 0 6 4$ </td><td>[0.6874, 0.6986]</td></tr><tr><td>CatBoost</td><td> $0 . 6 9 9 8 \pm 0 . 0 0 9 6$ </td><td>[0.6914, 0.7082]</td></tr><tr><td>HistGradientBoosting</td><td> $0 . 6 9 3 8 \pm 0 . 0 0 5 7$ </td><td>[0.6888, 0.6988]</td></tr><tr><td>Random Forest</td><td> $0 . 6 9 8 0 \pm 0 . 0 0 6 4$ </td><td>[0.6924, 0.7036]</td></tr><tr><td>Logistic Regression</td><td> $0 . 6 6 6 2 \pm 0 . 0 0 7 8$ </td><td>[0.6594, 0.6730]</td></tr></table>

Table 5: Cross-validation stability analysis using macro F1 score. Approximate 95% confidence intervals were calculated from the 5-fold cross-validation results.

Full results for the boosting, ensemble, and baseline models are reported in Table 6. XGBoost achieved an accuracy of 0.71, balanced accuracy of 0.72, and a macro F1 score of 0.70. The model performed well on the normal and diabetes classes, with strong precision and relatively balanced recall. As expected, prediabetes remained the most dificult class to classify. The model achieved a precision of 0.49, a sensitivity of 0.65 and a specificity of 0.74 for this class, reflecting the dificulty in separating prediabetes from both normal and diabetic cases without HbA1c. Although precision for the prediabetes class remained moderate, the relatively higher recall is important in a screening context, where failing to identify at-risk individuals may be more problematic than generating additional false positives. Most prediabetes misclassifications involved the normal class rather than the diabetes class, as shown in the confusion matrices in Figure 1. For diabetes, XGBoost reached a specificity of 0.96, so few individuals without diabetes were labeled diabetic.

The remaining models showed similar performance patterns. Once LightGBM and CatBoost were tuned under the same search procedure, both landed very close to XGBoost, each reaching an accuracy of 0.70 and a macro F1 of 0.69. For prediabetes, XGBoost, the soft voting ensemble, HistGradientBoosting, and Random Forest all reached an F1 of 0.56, with Light-GBM and CatBoost just behind at 0.55. Sensitivity for this class varied a little more, from 0.62 for Random Forest up to 0.66 for HistGradientBoosting. The diabetes class was handled consistently, with sensitivity of 0.79 to 0.80 for XGBoost, LightGBM, CatBoost, and the ensemble, and specificity between 0.95 and 0.97 across all models. For the normal class, F1 held at 0.78 for the boosting models and the ensemble and rose slightly to 0.79 for Random Forest. No single boosting model was clearly best across every class, and the gaps between them were small.

![](images/2cf24a00ddd11cbf869de4c54215898e526afc6bbf8219a5e19265024754baaa.jpg)

![](images/9deb74d46872b5ccb1697607f585faee94d559bfb9b5e7764ad23dd93ce089b4.jpg)

![](images/11f9b08c901d8abaaed8dd32d866a26d7ff53c4c7b0926309758f66fbfbd06a0.jpg)

Figure 1: Confusion matrices for XGBoost across the multiclass and binary prediction formulations.
<table><tr><td>Model</td><td></td><td>ced Acd </td><td>Macro F1</td><td>rmal Precision</td><td>mal Recall</td><td>mal Spe ecificity</td><td>Normal F1</td><td>rediabetes Precisio</td><td>Prediabetes Recall</td><td>Prediabetes Spe ecificity</td><td>Prediabetes F1</td><td>Diabetes Precision</td><td>Diabetes Recall</td><td>Diabetes Specificity</td><td>Diabetes F1</td></tr><tr><td>XGBoost</td><td>0.71</td><td></td><td>0.70</td><td>0.86</td><td>0.71 0.71</td><td>0.83</td><td>0.78</td><td>0.49</td><td>0.65</td><td>0.74</td><td>0.56</td><td>0.73</td><td>0.80</td><td>0.96</td><td>0.76</td></tr><tr><td>LightGBM CatBoost</td><td>0.70 0.70</td><td></td><td>0.69</td><td>0.86</td><td></td><td>0.82 0.83</td><td>0.78 0.78</td><td>0.48</td><td>0.64</td><td>0.773</td><td>0.55 0.55</td><td>0.71 0.70</td><td>0.7</td><td>0.96 0.95</td><td></td></tr><tr><td>Ensemble</td><td>0.70</td><td>0.72</td><td>0.69</td><td>0.86</td><td>0.71 0.71</td><td>0.83</td><td>0.78</td><td>0.48 0.49</td><td>0.64</td><td>0.73</td><td>0.56</td><td>0.77</td><td></td><td></td><td>0.774</td></tr><tr><td>HistGradientBoosting</td><td>0.70</td><td>0.71</td><td>0.70</td><td>0.86 0.86</td><td>0.71</td><td>0.82</td><td>0.78</td><td>0.49</td><td>0.65 0.66</td><td>0.73</td><td>0.56</td><td></td><td>0.80</td><td>0.96</td><td>0.75</td></tr><tr><td>Random Forest</td><td>0.72</td><td>0.71</td><td>0.70</td><td>0.84</td><td>0.75</td><td>0.79</td><td>0.79</td><td>0.50</td><td>0.62</td><td>0.76</td><td>0.56</td><td>0.76</td><td>0.77 0.77</td><td>0.96 0.97</td><td>0.75 0.76</td></tr><tr><td>Logistic Regression</td><td>0.69</td><td>0.71</td><td>0.68</td><td>0.85</td><td>0.70</td><td>0.82</td><td>0.77</td><td>0.47</td><td>0.64</td><td>0.73</td><td>0.54</td><td>0.69</td><td>0.79</td><td>0.95</td><td>0.73</td></tr></table>

Table 6: Performance comparison across the boosting, ensemble, and baseline models, including overall metrics, balanced accuracy, and class-wise precision, recall, specificity, and F1 scores for normal, prediabetes, and diabetes. Recall and specificity are computed for each class by treating that class as positive and the other two classes as negative.

## 4.2. Binary Formulationsfor Prediabetes Detection

The two binary formulations produced diferent levels of difficulty. Relative to the multiclass setting, where prediabetes F1 ranged from approximately 0.54 to 0.56, the normal versus prediabetes formulation led to a modest improvement, with F1 scores ranging from 0.62 to 0.64. A larger increase was observed in the broader screening-oriented formulation, where prediabetes and diabetes were grouped into a single abnormal glycemic category. In this setting, F1 scores reached approximately 0.73. Simplifying the prediction task consistently improved performance across all evaluated models, and the substantially larger gain from combining prediabetes and diabetes indicates that the models more readily distinguish general metabolic abnormality than the finer distinction between normal and prediabetes. Table 7 summarizes model performance across both formulations. Balanced accuracy followed the same pattern, ranging from 0.72 to 0.74 for normal versus prediabetes and from 0.77 to 0.78 for normal versus abnormal glycemic state. Specificity ranged from 0.70 to 0.77 in the normal versus prediabetes task and from 0.77 to 0.80 in the normal versus abnormal task, with Random Forest highest in both and XGBoost at 0.71 and 0.77, respectively.

## 4.3. Transformer-Based Results Across Prediction Formulations

As shown in Table 8, the transformer models produced results broadly comparable to the conventional machine learning baselines, but did not lead to a clear improvement in overall performance.

Among the transformer models, TabTransformer achieved the highest macro F1 score of 0.69 and the highest prediabetes F1 score of 0.55. FT-Transformer achieved the highest overall accuracy and balanced accuracy, reaching 0.71 and 0.72, respectively. SAINT achieved a balanced accuracy of 0.71, although its overall accuracy and macro F1 score were slightly lower than those of the other transformer models.

Prediabetes remained the most dificult class across all three architectures, with F1 scores ranging from 0.53 to 0.55. Under the normal versus prediabetes formulation, prediabetes F1 increased to approximately 0.63 for all three transformer models. Here, FT-Transformer achieved the highest accuracy at 0.73, while SAINT achieved the highest prediabetes recall at 0.77. Performance improved further under the normal versus abnormal formulation. Abnormal-class F1 reached 0.73 for TabTransformer and FT-Transformer and 0.74 for SAINT. FT-Transformer and SAINT both reached the highest accuracy at 0.77 and the highest balanced accuracy at 0.78, and both attained a ROC-AUC of 0.87. Specificity ranged from 0.72 to 0.79 for prediabetes in the multiclass task, from 0.70 to 0.73 in the normal versus prediabetes task, and from 0.74 to 0.76 in the normal versus abnormal task. None of the three architectures consistently outperformed the boosting models, and all struggled to separate prediabetes from adjacent glycemic states, indicating that increasing model complexity alone does not overcome the intrinsic overlap between intermediate metabolic states in the available feature space.

## 4.4. Efect of Problem Formulation

Comparing the three formulations shows that the main difficulty in detecting prediabetes comes from overlap between neighboring glycemic states. As shown in Figure 1, diabetes is identified more cleanly than prediabetes in the multiclass setting, while most of the uncertainty appears around the normal and prediabetes boundary. This suggests that the model captures stronger metabolic signals for diabetes, but has more difficulty separating early-stage risk from normal variation.

The binary normal versus prediabetes formulation makes this issue clearer. By removing diabetes from the task, the model focuses only on early-stage detection, but the confusion matrix still shows overlap between the two groups. The model correctly identifies 816 of the 1,070 prediabetes cases, a sensitivity of 0.76, but it also classifies 663 of the 2,312 normal individuals as prediabetic, giving a specificity of 0.71. This pattern matches the results in Table 7, where sensitivity for prediabetes is stronger than precision, and shows that the boundary remains dificult to define.

<table><tr><td>Formulation</td><td>Model</td><td>Accuracy</td><td>Balanced Accuracy</td><td>Precision</td><td>Recall</td><td>Specificity</td><td>F1 Score</td><td>ROC-AUC</td></tr><tr><td>Normal vs Prediabetes</td><td>XGBoost</td><td>0.73</td><td>0.74</td><td>0.55</td><td>0.76</td><td>0.71</td><td>0.64</td><td>0.8182</td></tr><tr><td>Normal vs Prediabetes</td><td>LightGBM</td><td>0.73</td><td>0.74</td><td>0.55</td><td>0.76</td><td>0.71</td><td>0.64</td><td>0.8169</td></tr><tr><td>Normal vs Prediabetes</td><td>CatBoost</td><td>0.73</td><td>0.73</td><td>0.55</td><td>0.75</td><td>0.72</td><td>0.64</td><td>0.8144</td></tr><tr><td>Normal vs Prediabetes</td><td>Ensemble</td><td>0.73</td><td>0.74</td><td>0.55</td><td>0.76</td><td>0.72</td><td>0.64</td><td>0.8178</td></tr><tr><td>Normal vs Prediabetes</td><td>HistGradientBoosting</td><td>0.73</td><td>0.74</td><td>0.55</td><td>0.76</td><td>0.72</td><td>0.64</td><td>0.8169</td></tr><tr><td>Normal vs Prediabetes</td><td>Random Forest</td><td>0.74</td><td>0.72</td><td>0.58</td><td>0.68</td><td>0.77</td><td>0.62</td><td>0.8055</td></tr><tr><td>Normal vs Prediabetes</td><td>Logistic Regression</td><td>0.71</td><td>0.72</td><td>0.54</td><td>0.74</td><td>0.70</td><td>0.62</td><td>0.8016</td></tr><tr><td>Normal vs Abnormal Glycemic State</td><td>XGBoost</td><td>0.77</td><td>0.78</td><td>0.69</td><td>0.78</td><td>0.77</td><td>0.73</td><td>0.8674</td></tr><tr><td>Normal vs Abnormal Glycemic State</td><td>LightGBM</td><td>0.77</td><td>0.78</td><td>0.69</td><td>0.78</td><td>0.77</td><td>0.73</td><td>0.8690</td></tr><tr><td>Normal vs Abnormal Glycemic State</td><td>CatBoost</td><td>0.77</td><td>0.77</td><td>0.69</td><td>0.77</td><td>0.78</td><td>0.73</td><td>0.8698</td></tr><tr><td>Normal vs Abnormal Glycemic State</td><td>Ensemble</td><td>0.77</td><td>0.78</td><td>0.69</td><td>0.78</td><td>0.77</td><td>0.73</td><td>0.8696</td></tr><tr><td>Normal vs Abnormal Glycemic State</td><td>HistGradientBoosting</td><td>0.78</td><td>0.77</td><td>0.70</td><td>0.76</td><td>0.79</td><td>0.73</td><td>0.8672</td></tr><tr><td>Normal vs Abnormal Glycemic State</td><td>Random Forest</td><td>0.78</td><td>0.78</td><td>0.71</td><td>0.76</td><td>0.80</td><td>0.73</td><td>0.8622</td></tr><tr><td>Normal vs Abnormal Glycemic State</td><td>Logistic Regression</td><td>0.77</td><td>0.77</td><td>0.69</td><td>0.78</td><td>0.77</td><td>0.73</td><td>0.8574</td></tr></table>

Table 7: Performance comparison across targeted binary formulations. Metrics are reported for the positive class in each setup, corresponding to prediabetes in the first formulation and to the combined abnormal glycemic state, which includes both prediabetes and diabetes, in the second. Specificity is the proportion of normal individuals correctly classified as normal. Balanced accuracy is also reported to account for class imbalance between the normal and positive classes.
<table><tr><td>Formulation</td><td>Model</td><td>Accuracy</td><td>Balanced Accuracy</td><td>Macro F1</td><td>Precision</td><td>Recall</td><td>Specificity</td><td>F1</td><td>ROC-AUC</td></tr><tr><td>Multiclass</td><td>TabTransformer</td><td>0.70</td><td>0.71</td><td>0.69</td><td>0.49</td><td>0.61</td><td>0.75</td><td>0.55</td><td>0.86</td></tr><tr><td>Multiclass</td><td>FT-Transformer</td><td>0.71</td><td>0.72</td><td>0.68</td><td>0.50</td><td>0.56</td><td>0.79</td><td>0.53</td><td>0.86</td></tr><tr><td>Multiclass</td><td>SAINT</td><td>0.68</td><td>0.71</td><td>0.67</td><td>0.46</td><td>0.61</td><td>0.72</td><td>0.53</td><td>0.85</td></tr><tr><td>Normal vs Prediabetes</td><td>TabTransformer</td><td>0.72</td><td>0.73</td><td>0.71</td><td>0.55</td><td>0.75</td><td>0.71</td><td>0.63</td><td>0.80</td></tr><tr><td>Normal vs Prediabetes</td><td>FT-Transformer</td><td>0.73</td><td>0.73</td><td>0.71</td><td>0.56</td><td>0.74</td><td>0.73</td><td>0.63</td><td>0.81</td></tr><tr><td>Normal vs Prediabetes</td><td>SAINT</td><td>0.72</td><td>0.73</td><td>0.70</td><td>0.54</td><td>0.77</td><td>0.70</td><td>0.63</td><td>0.81</td></tr><tr><td>Normal vs Abnormal Glycemic State</td><td>TabTransformer</td><td>0.76</td><td>0.77</td><td>0.76</td><td>0.67</td><td>0.79</td><td>0.74</td><td>0.73</td><td>0.86</td></tr><tr><td>Normal vs Abnormal Glycemic State</td><td>FT-Transformer</td><td>0.77</td><td>0.78</td><td>0.77</td><td>0.68</td><td>0.79</td><td>0.76</td><td>0.73</td><td>0.87</td></tr><tr><td>Normal vs Abnormal Glycemic State</td><td>SAINT</td><td>0.77</td><td>0.78</td><td>0.77</td><td>0.68</td><td>0.80</td><td>0.75</td><td>0.74</td><td>0.87</td></tr></table>

Table 8: Transformer-based model performance across the multiclass, normal versus prediabetes, and normal versus abnormal glycemic state formulations. For the multiclass task, precision, recall, specificity, and F1 refer to the prediabetes class, with specificity computed by treating prediabetes as positive and the remaining two classes as negative. For the binary tasks, these metrics refer to prediabetes in the normal versus prediabetes formulation and to the combined abnormal glycemic state in the normal versus abnormal formulation, and specificity is the proportion of normal individuals correctly classified as normal.

The normal versus abnormal glycemic state formulation shows the clearest separation because prediabetes and diabetes are combined into a single positive class. This reduces the burden on the model by asking it to detect broader metabolic abnormality rather than distinguish between intermediate and advanced glycemic states. Here XGBoost reaches a sensitivity of 0.78 and a specificity of 0.77, so most abnormal cases are detected without misclassifying a large share of normal individuals. The progressive improvement observed when moving from multiclass prediction to broader abnormality screening suggests that the principal source of error lies in distinguishing intermediate metabolic states rather than detecting glycemic abnormality in general.

## 4.5. UMAP Feature-Space Visualization

To examine how the glycemic classes were distributed in the feature space, we applied Uniform Manifold Approximation and Projection (UMAP) [36] to the same leakage-aware predictor set used for classification, with settings given in the caption of Figure 2.

As shown in Figure 2, the projection contains several local groupings, but these groups do not clearly correspond to the three glycemic classes. Some regions contain higher concentrations of normal or diabetic participants, although substantial mixing remains throughout the projection. Prediabetes is particularly dispersed across regions containing both normal and diabetic participants rather than forming a separate group. This pattern lines up with the lower F1 scores that prediabetes produced across the models. We read this agreement as supporting evidence rather than a direct explanation, because UMAP is a nonlinear, exploratory projection that does not preserve the exact geometry of the feature space and can stretch or compress distances. For that reason, the groupings seen here should not be taken as distinct clinical subgroups.

![](images/0cefcedc6191e1452b60e93b28630e016d113c95bc3cd843cc0ed25db4c562c2.jpg)  
Figure 2: UMAP visualization of the normal, prediabetes, and diabetes groups in the test data. HbA1c and participant identifiers were excluded, numeric variables were standardized, and categorical variables were one-hot encoded. UMAP was fitted on the training set and the held-out test set was projected into two dimensions, using 30 neighbors, a minimum distance of 0.1, Euclidean distance, and a random seed of 42.

<table><tr><td>Encoding</td><td>Accuracy</td><td>Balanced Accuracy</td><td>Macro F1</td><td>Prediabetes F1</td></tr><tr><td>Label</td><td>0.71</td><td>0.72</td><td>0.70</td><td>0.56</td></tr><tr><td>One-Hot</td><td>0.71</td><td>0.72</td><td>0.69</td><td>0.55</td></tr><tr><td>Target</td><td>0.72</td><td>0.72</td><td>0.70</td><td>0.56</td></tr></table>

Table 9: Encoding sensitivity analysis for the multiclass XGBoost model.

## 4.6. Encoding Sensitivity Analysis

To examine whether the categorical encoding strategy affected the main multiclass results, we compared the primary XGBoost model using label encoding, one-hot encoding, and target encoding. As shown in Table 9, performance changed only marginally across the three methods, so the main conclusions of our study hold regardless of how categorical variables are represented. The prediabetes F1 stayed modest under every encoding, which points back to feature overlap rather than encoding choice as the source of the dificulty.

## 4.7. Fairness-Aware Reweighting

We also assessed how fairness-aware reweighting afected the consistency of prediabetes detection across racial groups in the multiclass setting. In this context, the recall of subgroups captures the proportion of true prediabetes cases within a given group that the model correctly identified. Relative to the baseline, the fairness-aware XGBoost model did not distribute prediabetes recall more evenly across race groups. Recall increased for Non-Hispanic White and Other Race participants but decreased for Other Hispanic and Non-Hispanic Black participants, while Mexican American participants stayed about the same, so the gap between the highest and lowest groups remained similar. The analysis also considered only recall rather than a fuller range of fairness measures. The results are therefore best read as exploratory evidence that reweighting alone shifts recall between groups rather than equalizing it, not as a comprehensive fairness assessment spanning all groups and error types. Figure 3 reports prediabetes recall across race groups for the baseline and fairness-aware models.

## 5. Model Interpretation

Figure 4 shows the SHAP summary plot for the prediabetes class in the XGBoost model, where features are ordered by their mean absolute SHAP value and the spread of values reflects their influence on individual predictions. The most influential predictors include fasting glucose, the interaction between age and body mass index, age, and LDL cholesterol, followed by waist circumference, the triglyceride to HDL ratio, and body mass index. Fasting glucose is the strongest predictor by a wide margin. It is not used to define the labels, but it is closely related to HbA1c, so part of its contribution reflects glycemic information the two measures share.

![](images/bf85c35282e4315bdbe9114d2204661d7602abe40615f0fc5e2a8548ef2919ae.jpg)  
Figure 3: Prediabetes recall across race or ethnicity groups for the baseline and fairness-aware multiclass XGBoost models.

The efect of fasting glucose is not uniform across its range. Very high values strongly push predictions away from prediabetes, which is consistent with these individuals being classified as diabetic instead, while low to moderate values contribute modestly toward prediabetes. The interaction between age and body mass index shows a clear pattern, which is that low values reduce predicted risk and high values increase it. Age and LDL cholesterol show consistent positive contributions, with higher values associated with increased predicted risk, which aligns with established clinical patterns.

Lipid variables also contribute to the predictions. A higher triglyceride-to-HDL ratio is associated with increased risk, while higher HDL cholesterol tends to reduce it. This reflects known mechanisms of metabolic dysfunction, where elevated triglycerides and reduced HDL are linked to insulin resistance. Waist circumference, pulse pressure, and systolic and diastolic blood pressure show smaller and less consistent efects. For waist circumference, higher values tend to reduce predicted risk, which most likely reflects information already captured by the age interaction terms rather than a protective efect. Physical activity and sedentary behavior variables contribute relatively little, and the binary clinical threshold flags do not appear among the 20 most influential features, which suggests that their information is already represented by the continuous measurements.

Overall, the SHAP analysis shows that the model relies mainly on glycemic, age, and body size measures, with lipid and lifestyle features playing a smaller role. The major predictors identified by SHAP exhibit continuous rather than discrete relationships with risk. This supports the view that glycemic status reflects a metabolic continuum rather than sharply separated categories.

![](images/f5f48a5e8381fbde2f8d548a07c013e95e069366b7f93dc922e7b79e87631fb6.jpg)  
Figure 4: SHAP summary plot illustrating feature contributions to the prediabetes class in the XGBoost model. Laboratory features are LBXGLU for fasting glucose, LBDLDL for LDL cholesterol, LBDHDD for HDL cholesterol, LBXTR for triglycerides, and LBXTC for total cholesterol. Body and blood pressure features are BMXBMI for body mass index, BMXWAIST for waist circumference, and SBP and DBP for systolic and diastolic blood pressure. Demographic and lifestyle features are RIDAGEYR for age, RIDRETH1 for race or ethnicity, SMQ040 for current cigarette smoking, SMQ020 for having smoked at least 100 cigarettes in life, PAD680 for minutes of sedentary activity, PAQ665 for moderate recreational activities, and MCQ300C for a close relative having diabetes. Engineered features are Age\_BMI and Age\_Waist for the interactions of age with body mass index and waist circumference, TG\_to\_HDL for the triglyceride-to-HDL ratio, and PulsePressure for pulse pressure.

## 6. Discussion

The findings show that both conventional machine learning models and transformer-based architectures perform well in predicting normal, prediabetes, and diabetes using NHANES data, with macro F1 scores around 0.70 across the boosting and ensemble models. This shows that the selected demographic, clinical, and lifestyle variables carry enough information to distinguish among the three glycemic states even without using HbA1c as an input feature.

None of the transformer-based models clearly outperformed the strongest conventional baselines, so attention-based architectures alone do not overcome the dificulty of separating the intermediate glycemic state. This agrees with Oba et al. [19], where XGBoost and CatBoost outperformed the attentionbased TabNet on health screening records. Given the close performance of conventional and transformer-based approaches, further gains likely depend more on richer feature sets or improved cohort design than on switching architectures.

Prediabetes remained the hardest class to predict across models, which reflects the biological continuum between normal glycemic regulation and diabetes, where fixed clinical thresholds are applied to a gradual process of metabolic change. Individuals near the normal-prediabetes or prediabetes-diabetes boundaries may therefore have similar demographic, metabolic, and lifestyle profiles despite being assigned to diferent classes. Fan et al. [14] reported a prediabetes F1 of 0.88, but with HbA1c among the inputs and only eight prediabetic test cases, suggesting that much of the diference reflects excluding HbA1c from the predictors rather than limitations in model capacity.

Fairness-aware reweighting did not make prediabetes recall more consistent across race groups, and instead shifted it between groups. In a healthcare context, this is important because uneven performance across demographic groups can translate into unequal opportunities for early detection and prevention. This suggests that reweighting alone is not enough, and that fairness considerations belong in model development itself, not only in evaluation performed after the fact.

Across the evaluated model families and prediction formulations, the overlap between adjacent glycemic states contributes more to prediction dificulty than diferences in model architecture. The normal versus abnormal formulation reached an area under the curve of 0.87, compared with 0.82 for binary diabetes prediction from NHANES lifestyle variables [5] and 0.90 in a high-risk NHANES subgroup [6], although targets and predictors difer across these studies. The models are therefore better at detecting general glycemic abnormality than at isolating prediabetes as a distinct intermediate state. One limitation of the current formulation is that standard multiclass classification treats normal, prediabetes, and diabetes as separate categories, even though they represent an ordered metabolic progression. Not all classification errors are equally serious, since misclassifying a prediabetic individual as normal is less severe than misclassifying a diabetic individual as normal, yet conventional multiclass approaches penalize both equally. Future work could explore ordinal classification or cost-sensitive learning to better reflect this structure.

## 7. Limitations

First, the outcome labels were defined using HbA1c thresholds, which is a practical and commonly used approach, but it does not fully reflect how diabetes and prediabetes are assessed in clinical practice, where diagnosis often involves mul tiple tests, medical history, and repeated measurements over time. In addition, NHANES is cross-sectional, so the models developed here focus on classification and do not provide insight into how individuals transition from prediabetes to diabetes, which may also obscure early predictive signals. Excluding HbA1c, while necessary to prevent leakage, may also remove a clinically dominant predictor, thereby lowering achievable performance. However, fasting glucose remained in the predictor set despite its biological relationship with HbA1c and may partially contribute to reconstructing glycemic status. It is also measured only in the fasting subsample, so imputed values were used elsewhere. Since it was the strongest predictor in the SHAP analysis, a check restricted to participants with measured fasting glucose would clarify how far the reported importance depends on imputation. External validation on an independent cohort was not performed, limiting the generalizability of the findings.

The fairness-aware reweighting analysis remained exploratory and was limited to the primary multiclass XGBoost model. It focused mainly on subgroup recall and did not evaluate other fairness-related outcomes such as subgroup precision, false positive rates, calibration, or intersectional diferences across combined demographic categories. Finally, the study relies on the variables available in NHANES and evaluates a selected group of conventional and transformer-based models.

## 8. Conclusion and Future Work

This study examined how well conventional machine learning models and transformer-based architectures can classify individuals as normal, prediabetic, or diabetic using NHANES data. Across the boosting models XGBoost, LightGBM, and CatBoost, comparable performance was achieved within all evaluation metrics. Even without using HbA1c as an input feature, the models were able to distinguish between the three groups reasonably well. Prediabetes remained the most dificult class, although recall for this group was still relatively strong, which is important for early detection. Specificity for diabetes stayed between 0.95 and 0.97 across models, so few individuals without diabetes were misclassified as diabetic.

Problem formulation mattered more than model choice, particularly when modeling intermediate disease states such as prediabetes. Among both model families, performance was strongest when prediabetes and diabetes were grouped together, meaning the models are better at recognizing general glycemic abnormality than identifying prediabetes as a separate intermediate state. Several extensions to this work are possible. Adding features such as dietary habits, medication use, or more detailed clinical variables may help improve separation between classes, especially for prediabetes. Longitudinal modeling and clinically actionable risk thresholds for early-stage detection are worth evaluating as well. Applying these models to external datasets, and ideally to longitudinal data, would show how well they generalize and whether they can support practical use in real-world settings.

## 9. Data Availability

The data used in this study are derived from the National Health and Nutrition Examination Survey, a public resource made available by the National Center for Health Statistics [24]. This study used the 2007-2008 to 2017-2018 cycles.

## 10. Acknowledgements

The authors thank the National Center for Health Statistics for making the NHANES data publicly available. This study is a secondary analysis of de-identified NHANES data. The Research Ethics Review Board of the National Center for Health Statistics approved NHANES and written informed consent was obtained from all adult participants. No further ethical approval was needed. This research did not receive specific grants from funding agencies in the public, commercial or non-profit sectors. The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## References

[1] W. C. Knowler, E. Barrett-Connor, S. E. Fowler, R. F. Hamman, J. M. Lachin, E. A. Walker, D. M. Nathan, Diabetes Prevention Program Research Group, Reduction in the incidence of type 2 diabetes with lifestyle intervention or metformin, New England Journal of Medicine 346 (6) (2002) 393–403. doi:10.1056/NEJMoa012512.

[2] American Diabetes Association Professional Practice Committee for Diabetes, 2. Diagnosis and Classification of Diabetes: Standards of Care in Diabetes— 2026, Diabetes Care 49 (Supplement 1) (2026) S27–S49. doi:10.2337/dc26-S002.

[3] Z. Xu, Z. Wang, A risk prediction model for type 2 diabetes based on weighted feature selection of random forest and XGBoost ensemble classifier, in: 2019 Eleventh International Conference on Advanced Computational Intelligence (ICACI), Guilin, China, 2019, pp. 278–283. doi:10.1109/ICACI.2019.8778622.

[4] H. M. Deberneh, I. Kim, Prediction of type 2 diabetes based on machine learning algorithm, International Journal of Environmental Research and Public Health 18 (6) (2021) 3317. doi:10.3390/ijerph18063317.

[5] E. Riveros Perez, B. Avella-Molano, Learning from the machine: is diabetes in adults predicted by lifestyle variables? A retrospective predictive modelling study of NHANES 2007–2018, BMJ Open 15 (3) (2025) e096595. doi:10.1136/bmjopen-2024-096595.

[6] X. Yang, M. Yao, J. Huang, Z. Cheng, T. Sun, Machine learning predicts diabetes risk in high-risk populations: analysis of National Health and Nutrition Examination Survey data, Archives of Medical Science 22 (2) (2026) 708–720. doi:10.5114/aoms/209547.

[7] L. Jiang, Z. Xia, R. Zhu, H. Gong, J. Wang, J. Li, L. Wang, Diabetes risk prediction model based on community follow-up data using machine learning, Preventive Medicine Reports 35 (2023) 102358. doi:10.1016/j.pmedr.2023.102358.

[8] W. Li, Y. Peng, K. Peng, Diabetes prediction model based on GA-XGBoost and stacking ensemble algorithm, PLOS ONE 19 (9) (2024) e0311222. doi:10.1371/journal.pone.0311222.

[9] L. Kopitar, P. Kocbek, L. Cilar, A. Sheikh, G. Štiglic, Early detection of type 2 diabetes mellitus using machine learning-based prediction models, Scientific Reports 10 (2020) 11981. doi:10.1038/s41598-020-68771-z.

[10] A. R. Kulkarni, A. A. Patel, K. V. Pipal, S. G. Jaiswal, M. T. Jaisinghani, V. Thulkar, L. Gajbhiye, P. Gondane, A. B. Patel, M. Mamtani, H. Kulkarni, Machinelearning algorithm to non-invasively detect diabetes and pre-diabetes from electrocardiogram, BMJ Innovations 9 (1) (2023) 32–42. doi:10.1136/bmjinnov-2021-000759.

[11] X. Li, F. Ding, L. Zhang, S. Zhao, Z. Hu, Z. Ma, F. Li, Y. Zhang, Y. Zhao, Y. Zhao, Interpretable machine learning method to predict the risk of pre-diabetes using a national-wide cross-sectional data: evidence from CHNS, BMC Public Health 25 (1) (2025) 1145. doi:10.1186/s12889-025-22419-7.

[12] Y. Hu, Y. Han, Y. Liu, Y. Cui, Z. Ni, L. Wei, C. Cao, H. Hu, Y. He, A nomogram model for predicting 5-year risk of prediabetes in Chinese adults, Scientific Reports 13 (1) (2023) 22523. doi:10.1038/s41598-023-50122-3.

[13] M. B. Almadhoun, M. A. Burhanuddin, Optimizing feature selection and machine learning algorithms for early detection of prediabetes risk: Comparative study, JMIR Bioinformatics and Biotechnology 6 (2025) e70621. doi:10.2196/70621.

[14] H. Fan, Z. Li, N. Yan, H. Liu, An interpretable progressive residual network for automated multiclass diabetes diagnosis, Scientific Reports 16 (1) (2026) 20468. doi:10.1038/s41598-026-51603-x.

[15] S. Mukherjee, D. De, XAI-IoT based real-time diabetes prediction using TabTransformer, Medicine in Novel Technology and Devices 30 (2026) 100435. doi:10.1016/j.medntd.2026.100435.

[16] Kevin, N. K. Thariq, E. Irwansyah, Diabetes risk classification in medical data using TabTransformer, in: 2025 6th International Conference on Artificial Intelligence and Data Sciences (AiDAS), 2025, pp. 50–55. doi:10.1109/AIDAS67696.2025.11213882.

[17] S. Khan, M. R. Biswas, Z. Shah, Early prediction of type 2 diabetes using multimodal data and tabular transformers, in: 2025 3rd International Conference on Foundation and Large Language Models (FLLM), 2025, pp. 592–599. doi:10.1109/FLLM67465.2025.11391053.

[18] N. Wang, M. Wu, W. Gu, C. Dai, Z. Shao, K. P. Subbalakshmi, MSFT-Transformer: A multistage fusion tabular transformer for disease prediction using metagenomic data, Briefings in Bioinformatics 26 (3) (2025) bbaf217. doi:10.1093/bib/bbaf217.

[19] Y. Oba, T. Tezuka, M. Sanuki, Y. Wagatsuma, Interpretable prediction of diabetes from tabular health screening records using an attentional neural network, in: 2021 IEEE 8th International Conference on Data Science and Advanced Analytics (DSAA), 2021, pp. 1–11. doi:10.1109/DSAA53316.2021.9564151.

[20] S. V. Chinta, Z. Wang, A. Palikhe, X. Zhang, A. Kashif, M. A. Smith, J. Liu, W. Zhang, AI-driven healthcare: A review on ensuring fairness and mitigating bias, PLOS Digital Health 4 (5) (2025) e0000864. doi:10.1371/journal.pdig.0000864.

[21] H. T. Cronjé, A. Katsiferis, L. K. Elsenburg, T. O. Andersen, N. H. Rod, T.-L. Nguyen, T. V. Varga, Assessing racial bias in type 2 diabetes risk prediction algorithms, PLOS Global Public Health 3 (5) (2023) e0001556. doi:10.1371/journal.pgph.0001556.

[22] K. Chakradeo, I. Huynh, S. B. Balaganeshan, O. L. Dollerup, H. Gade-Jørgensen, S. K. Laupstad, M. Malham, T.-L. Nguyen, A. Hulman, T. V. Varga, Navigating fairness aspects of clinical prediction models, BMC Medicine 23 (1) (2025) 567. doi:10.1186/s12916-025- 04340-3.

[23] J. Kleinberg, S. Mullainathan, M. Raghavan, Inherent trade-ofs in the fair determination of risk scores, in: 8th Innovations in Theoretical Computer Science Conference (ITCS 2017), Vol. 67 of Leibniz International Proceedings in Informatics (LIPIcs), 2017, pp. 43:1–43:23. doi:10.4230/LIPIcs.ITCS.2017.43.

[24] National Center for Health Statistics, National health and nutrition examination survey, https://wwwn.cdc.gov/nchs/nhanes/, accessed 9 October 2026 (2026).

[25] H. He, E. A. Garcia, Learning from imbalanced data, IEEE Transactions on Knowledge and Data Engineering 21 (9) (2009) 1263–1284. doi:10.1109/TKDE.2008.239.

[26] T. Chen, C. Guestrin, XGBoost: A scalable tree boosting system, in: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, ACM, 2016, pp. 785–794. doi:10.1145/2939672.2939785.

[27] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, T.-Y. Liu, LightGBM: A highly eficient gradient boosting decision tree, in: Proceedings of the 31st International Conference on Neural Information Processing Systems, Curran Associates Inc., Red Hook, NY, USA, 2017, pp. 3149–3157.

[28] L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, A. Gulin, CatBoost: Unbiased boosting with categorical features, in: Proceedings of the 32nd International Conference on Neural Information Processing Systems, Curran Associates Inc., Red Hook, NY, USA, 2018, pp. 6639– 6649.

[29] L. Breiman, Random forests, Machine Learning 45 (1) (2001) 5–32. doi:10.1023/A:1010933404324.

[30] X. Huang, A. Khetan, M. Cvitkovic, Z. Karnin, Tab-Transformer: Tabular data modeling using contextual

embeddings, arXiv preprint arXiv:2012.06678 (2020). doi:10.48550/arXiv.2012.06678.

[31] Y. Gorishniy, I. Rubachev, V. Khrulkov, A. Babenko, Revisiting deep learning models for tabular data, in: Advances in Neural Information Processing Systems, Vol. 34, 2021, pp. 18932–18943.

[32] G. Somepalli, M. Goldblum, A. Schwarzschild, C. B. Bruss, T. Goldstein, SAINT: Improved neural networks for tabular data via row attention and contrastive pre-training, arXiv preprint arXiv:2106.01342 (2021). doi:10.48550/arXiv.2106.01342.

[33] S. M. Lundberg, S.-I. Lee, A unified approach to interpreting model predictions, in: Proceedings of the 31st International Conference on Neural Information Processing Systems, Curran Associates Inc., Red Hook, NY, USA, 2017, pp. 4768–4777.

[34] F. Doshi-Velez, B. Kim, Towards a rigorous science of interpretable machine learning, arXiv preprint arXiv:1702.08608 (2017). doi:10.48550/arXiv.1702.08608.

[35] Z. C. Lipton, The mythos of model interpretability, Communications of the ACM 61 (10) (2018) 36–43. doi:10.1145/3233231.

[36] L. McInnes, J. Healy, J. Melville, UMAP: Uniform manifold approximation and projection for dimension reduction, arXiv preprint arXiv:1802.03426 (2020). doi:10.48550/arXiv.1802.03426.