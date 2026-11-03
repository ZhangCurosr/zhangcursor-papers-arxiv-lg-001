# A Hybrid Visitor Prediction Framework using Interpretable Temporal features, Integrating IQR and Anomaly Detection in Lakes in Nebraska

Israt Zarin

School of Computing

University of Nebraska–Lincoln

Lincoln, Nebraska 68588, USA

izarin2@huskers.unl.edu

Christopher Chizinski

School of Natural Resources

University of Nebraska–Lincoln

Lincoln, Nebraska 68583, USA

cchizinski2@unl.edu

## Abstract

For efective resource management, infrastructure planning, and sustainable tourism, it is really important to predict accurate visitors. This paper estimates the daily visitation of Nebraska lakes using spatial, basic temporal, and interpretable temporal features. Moreover, we evaluate the base models with multiple metrics. Using the best three models, our method will ensemble hybrid models and choose the best hybrid technique. Further, we investigate the impact of lag features and how the models work with diferent outlier detection techniques such as Interquartile Range(IQR) and Anomaly detection. The models are evaluated using regression metrics (MAE, MSE, RMSE, MAPE,and R<sup>2</sup>). Experiment result shows that, after deducted anomaly, it reduces MAE, MSE,RMSE,and MAPE. After adding lag, it is showing some positive efect on the result. Moreover, Anomaly identification techniques is more better than IQR for improving performance and preserve data. The reported anomaly-based

cross-validation results favored the voting ensemble over the compared base models on the evaluated metrics. In anomaly-based cross-validation, VOTE improved prediction errors compared with MLP (256,128,64); improvements varied in the single-split comparison. Specifically, the hybrid model reduced MAE by 9.82%, MSE by 3.12%, RMSE by 1.76%, and MAPE by 6.77%. In the reported cross-validation comparison, the best anomaly-based hybrid model had lower MAE, MSE, RMSE, and MAPE than the best IQR-based hybrid model by 9.81%, 34.30%, 18.99%, and 0.66%, respectively.

Keywords: Spatial–temporal feature, Hybrid ensemble learning, Anomaly detection, IQR, Lag features

## 1 Introduction

The management of recreation areas, parks, museums, and shielded landscapes is becoming more dependent on time and space-rich data on when and where individuals visit[1][2]. Old techniques, including manual counters, intercept survey methods and periodic survey methods, have been useful in giving snapshots but do not meet the management demands of the scale, frequency or geographic location distribution of current visitation due to the labour-intensive nature and expensive tools of these methodologies[3], their high cost when used at high frequencies, their spatially limited nature and their inability to capture shortterm fluctuations in behaviour and behavioural shifts across multiple sites at once.

The mobile/GPS mobility data has reached maturity over the last 10 years and has made it possible to track human movement continuously and on a fine scale, so it is now possible to observe the daily, weekly, and seasonal patterns of human movement over a large number of sites at the same time [2].

Now a days, machine learning techniques are used to estimate visitors. With the development of the field, there has been a variety of research in terms of forecasting visitors[4][5] [6] and being efective in predicting planning, waste management, safety, and conservation planning. Cultural institutions and destinations: Studies show that tree-based models (e.g.,

Gradient Boosting, Random Forest) and sequence models can be even better than classical baselines by taking advantage of calendar structure (weekday, holidays, season) and exogenous drivers (weather, events). To illustrate, visit the forecasting at museums. It has found recurring patterns, and special-day efects are better described using ensembles than using linear baselines [4], whereas studies of lakes and parks have found that decision trees and decision-related classifiers can be very successful at classifying the expected demand [5]. Similar to city-level demand prediction, in urban areas, hybrid deep learning, namely, LSTM with gradient boosting, has demonstrated better results in city-level predictions as compared to the ARIMA-like models, indicating the worth of both nonlinear features interaction and short-term memory [7][8]. Syntheses of the tourism-ML literature also record the trend to data-driven, hybrid, and ensemble methods with more power to their content and stacked learners to elevate accuracy and stability [9]. An emerging body of work on the problem of sparse geolocation and regional modeling demonstrates how the addition of spatial structure (i.e. graph-based relationships between places) can improve the level of prediction where direct observations are incomplete or heterogeneous across locations [10][11]

This study is needed due to the fact that erroneous or noisy mobility information may result in erroneous information on how often visitors are visited and thereby alter funding and stafing, as well as policy decisions in managing recreational resources. As an example, raw spikes or drops in mobility indicators can either overestimate or underestimate the true visitation patterns, thereby giving false direction on conservation planning or investing in infrastructure [12]. By efectively reducing the problem through interquartile range (IQR), strong anomaly detection, the accuracy of machine learning models and the credibility of the decisions based on the data are enhanced. Not only that, but previous studies seldom test the behaviour of ensemble approaches after the removal of anomalies, so it is unclear which factor is responsible for greater improvement in performance: data cleaning or the type of model.

Meanwhile, scholars have also grown sensitive to the data-quality facts of mobility streams.

Raw signals may consist of duplicated pings, GPS drift, device/platform peculiarities, and transmission gaps, which may be passed up into modelling pipelines. According to empirical studies, transport and smartphone traces may contain systematic noise and platform diferences, which, when not mitigated, will result in biased downstream analytics and poor performance [13][14]. Expansive reviews of the spatio-temporal mining emphasise that robust pipelines should incorporate quality management and anomaly handling as opposed to more advanced models, particularly when resources are allocated, and policies are based on outputs[15]. Simultaneously, the spatio-temporal analytics society has suggested multi-scale anomaly detection on human-activity data and quality screening on volunteered geographic information, showing that the detection and pre-filtering of abnormal observations can significantly enhance the trustworthiness of the following model [16][17].

Although there is an evident improvement, there are four gaps. To begin with, the anomaly handling is not very often explicit. The vast majority of tourism-forecasting pipelines do not explicitly detect outliers, either by pre-aggregating or unintentionally cleaning data points, however outliers can overstate or understate estimated visitation and behave to reinforce the efects of downstream features[18]. Second, these comparisons of models are frequently limited: most studies compare a single family (e.g., tree ensembles or a particular deep network) and do not compare the variability of diferent regressors with similar splits and metrics, which makes it dificult to separate the benefits of data and the benefits of the model choice. Moreover, a single model can be biased. Third, the use of interpretable features: despite recent models that focus on it, most visitor-count analyses continue to work with flat tabular features and cannot be generalised between sites and seasons, and they don’t focus much on interpretable features[10][14]. Lastly, when the models are showing close results, how to identify which one is best?

The research proposes using extra temporal features and lag features, an interquartile range (IQR), and an anomaly-aware system; it cleaned dataset and predicted visitor numbers from extensive mobility datasets. The research design consists of two distinct phases.

The first stage employs three unsupervised anomaly detection methods, including Isolation Forest, Local Outlier Factor, and One-Class SVM, to remove abnormal data points while preserving genuine visitation patterns [19][20][18]. Moreover, at the same time, applying IQR to remove outliers helped to find out which process is more efective. The second stage performs identical model evaluations between base model candidates like tree-based models, linear and regularised models, and neural networks [21][22][23], which receive the same data partitions and evaluation metrics. The research design compares model performance across anomaly removal and IQR filtering, while also considering diferences among mode types. The research applies existing evidence from museums and lakes, marine parks, and urban tourism to create daily visitation estimates for recreation management decisions [4][5] [7][9][10][11] and also examined them with hybrid methods. It also used multiple metrics for evaluation so that it would be easy to choose between models whose performances are very similar. This study examines five factors relative to the accuracy of visitor prediction and the strength of model evaluation. First, it investigated the possibility of better prediction by using anomaly detection and the Interquartile Range (IQR) method without compromising data quality and investigated which method was more efective. Second, it evaluated whether hybrid machine learning models can outperform individual machine learning models. Third, it examines whether we need a single metric to assess models or whether a full multi-metric framework is needed for good assessment. Fourth, it examined the role of interpretable temporal features and lag variables as additional factors to increase the accuracy of predictions. Finally, if several models have similar predictive capabilities, it determines which model offers the best possible predictions in terms of accuracy, reliability and robustness. It develops a hybrid method that helps agencies convert noisy mobility data into reliable analytical tools for their stafing, conservation, and infrastructure planning needs.

## 2 Materials and Methods

The study presents an anomaly-conscious machine learning model to predict visitor activity at the recreational lakes within Nebraska based on large-scale mobility data. Its workflow consists of five main stages that include data acquisition and preprocessing, feature engineering, anomaly detection, model development and evaluation, and visualisation.

The workflow presented in diagram 1 represents the end-to-end modelling pipeline that is used in this research. We begin with the summarised visitation data and then proceed to carry out data cleaning and preprocessing to achieve consistency and reliability. This is followed by a well-structured aspect of feature engineering, in which spatial features, pure temporal features, and interpretable temporal features (like calendar and hourly visit patterns) are generated, and explicit leakage control is implemented to avoid using information that may unreasonably overpower predictive power. An ensemble of single regression models is then trained and evaluated (and can include tree-based, linear, and regularised models and neural networks), and the best-performing models are picked. The chosen models then act as base learners in a hybrid modelling phase, where they are both voted on and stacked, and ridge regression and gradient boosting are also used as meta-learners so that the individual models can be strengthened and weaknesses can be complementary to each other. Lastly, a wide range of regression- and classification-based measures, such as MSE, MAE, R2, precision, recall, and AUC, are used to evaluate model performance in terms of an integrated measure of predictive accuracy and predictive strength in various data regimes.

Figure file not uploaded:

Picture/0.png

Figure 1: Overview of the lake-visitation prediction workflow, including preprocessing, feature engineering, anomaly handling, base-model evaluation, and hybrid-model development.

## 2.1 Data Collection and Cleaning

The data will be gathered through the process of data collection and preprocessing. The data employed in the present research were obtained based on the AirSage mobility platform that combines anonymous and geographical visitation data throughout the waterbody in the state of Nebraska. The records are the estimated number of visitors to a site in a day and other metrics, including the time spent and the time information. In this project, we used 123627 records of 227 lakes that were gathered and processed. The data had blanks, duplicate records, and inconsistencies of identifiers that necessitated massive cleaning. The preprocessing procedures were:

## 2.2 Dataset preprocessing

In this research, we used a cleaned lake-visitation dataset where each row contains a lake identifier (poi id), date (YYYYMMDD), and target visitation count. Missing data are removed from the dataset. Features are divided into two categories. We used a cleaned, daily lake-visitation dataset where each record contains a lake identifier (poi id), a date (date, parsed from YYYYMMDD), and the target variable count (daily visitation). Records with missing data were removed. If the dataset contained an alternative lake identifier column (e.g., POI ID, lake id, id), it was renamed to poi id. To provide a consistent numeric representation of lake identity, we created poi id code using categorical encoding of poi id.

• Change of date attributes to a standard format YYYYMMDD.

• Assuring a unitary numeric representation of visitor numbers and visit durations.

• Elimination of zero-value records or incomplete entries.

• Arranging the records in order of poi id (point of interest identifier) and date to maintain temporal order.

It was used to clean data, meaning that it provided data uniformity and reliability before training the models.

## 2.3 Feature Engineering

At first we divided the features into two categories: spatial and temporal.

## 2.3.1 Spatial

Spatial features are demographic structures that select numeric columns from numeric columns, such as region/district, age, and race, and numeric counts from demographic prefixes, such as home \*, male \*, female \*, and income \*. Spatial features remain stable over time, which allows the model to capture long-term patterns. They are also called static features.

## 2.3.2 Temporal

Temporal features mainly capture dynamic patterns such as hourly visits. A few temporal and contextual characteristics were created after cleaning to record periodic and behavioural dynamics of visitation, such as weekly rhythms and seasonal cycles. Those features are derived from the current date, such as the start of the week, the end of the week, the season, the month, and visitation a day before. This enables the model to learn short-term patterns.

• Calendar information: weekend, month, and day of week (is weekend).

• Seasonal grouping: winter (Dec-Feb), spring (Mar- May), summer (Jun- Aug), and fall (Sep-Nov).

• Lag features: lag1 (number of visitors the day before) and lag7 (number of visitors the week before), a short-run change of direction.

To prevent data leakage and prevent biased learning, highly correlated features were removed. Some of the features are white, income 60 75, income 75 100, female 45 49, and so on. By eliminating these highly correlated features, we ensure predictive performance is meaningful and improve model robustness. The top 20 correlated features are described in the diagram.

![](images/f91ea55fbbdc354a62a753bb2b83a04d762c4e5de2fd8d1462a19c8b9de53caf.jpg)  
Figure 2: Heatmap for the top 20 highly correlated features

## 2.4 Remove Outliers

## 2.4.1 Anomaly Detection

The anonymous dataset can contain duplicate signals, transmission errors, and irregular patterns. To identify this type of pattern, three anomaly detection methods are used. Diferent methods can give diferent results because there must be changes in high false positive results. To avoid that, multiple anomaly detection algorithms must be used to identify the true anomalies, which helps us to avoid relying on one algorithm.

• Isolation Forest (IF) – isolates anomalous points by recursively partitioning the feature space.

• Local Outlier Factor (LOF) – measures the local density deviation of a sample relative to its neighbours.

• One-Class Support Vector Machine (OCSVM) – separates normal instances from anomalies by learning a tight boundary around the majority data.

Each algorithm produced an anomaly score. A consensus-based voting approach was

applied, where any data point labelled as anomalous by at least two detectors was removed:

$$
A _ { i } = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f ~ a t ~ l e a s t ~ t w o ~ d e t e c t o r s ~ m a r k ~ s a m p l e ~ } i \mathrm { ~ a s ~ a n o m a l o u s } , } \\ { 0 , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.
$$

## 2.4.2 Interquartile Range (IQR)

In order to identify and remove outliers, Interquartile Range (IQR) filtering was applied to the numeric predictor variables before model training. This preprocessing step reduced the influence of extreme values and improved the quality of the dataset used for machine learning model development. Figure 3 indicates the amount of data removed by the two techniques in preprocessing. Anomaly detection removed just 6,181 rows (5.00%) of the data, whereas 43,411 rows (35.11%) removed by IQR filtering. That indicates that the anomaly detection technique can preserve a large amount of data compared to IQR filtering.

Figure file not uploaded:

$$
\mathtt { P i c t u r e / P i c t u r e \_ 2 / 1 . p n g }
$$

Figure 3: The number of rows that were removed from the data by the IQR filtering and anomaly detection during data pre-processing.

## 2.5 Base Model Candidates

To evaluate performance in diferent condition we have selected diferent models. In this research, we considered three types of models: tree-based and ensemble models, linear and regularised models, and neural network models.

## 2.5.1 Tree-based and Ensemble models

Enable capturing non-linear patterns, we have used Random Forest (RF) and Extra Trees Regressor (ETR). RF helps to reduce variance. Besides, ETR improves generalisation. In addition, we have also employed a Gradient Boosting Regressor (GBR) to capture additive nonlinear relationships and a Decision Tree (DT) to provide hierarchical decision structures.

## 2.5.2 Linear and Regularized Models

In order to consider linear baselines, we selected linear regression (LR) and ridge regression. Ridge regression used L2 to overcome overfitting. Moreover, Huber Regression and Bayesian Ridge Regression are also included to improve robustness and stability

## 2.5.3 Neural Network Models

We used more than one Multi-Layer Perceptron (MLP) architecture of diferent layers, such as (64), (128, 64), and (256, 128, 64) hidden units. Standardised inputs, ReLU activation, early stopping, and adaptive learning rates were also used to train all neural models so that they are stable in their convergence. They are very eficient at capturing high-dimensional nonlinear patterns from spatial and temporal features.

The three best-performing models were selected for subsequent ensemble evaluation.

## 2.5.4 Evaluation Metrics

We evaluate our proposed method using both regression metrics and classification metrics. Base models are so strong, and diferences in prediction accuracy and loss can be very minimal, making it hard to identify the best models. Multiple metrics help to overcome that issue[24][25][26]. Besides, the regression metrics are used to indicate the similarity of the predicted and observed visitation counts, whereas the classification metrics are used to define whether the model can be used to identify the most relevant, highest-demand periods that are important in the management decisions.

Mean Absolute Error (MAE). MAE examines the average prediction errors in terms of target visitation[27][28]. It is simple to interpret and use in nature as the average absolute deviation between mean predicted counts and counts.Lower values indicate that the model’s performance is higher.

$$
\mathrm { M A E } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left| y _ { i } - \hat { y } _ { i } \right| .\tag{1}
$$

where,

• $n$ is the total number of observations.

$y _ { i }$ is the actual (observed) value.

• $\hat { y } _ { i }$ is the predicted value.

Mean Squared Error (MSE) MSE computes the average squared residual, which indicates that a higher value degrades the model’s performance[27]. Moreover, MSE is more sensitive to rare extreme errors and may be dominated by a small number of large residuals. MSE is more susceptible to large-scale extreme errors and could be dominated by relatively few large residuals.

$$
\mathrm { M S E } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( y _ { i } - \hat { y } _ { i } \right) ^ { 2 } .\tag{2}
$$

where,

• $n$ is the total number of observations.

$y _ { i }$ is the actual (observed) value.

• $\hat { y } _ { i }$ is the predicted value.

Coeficient of Determination $( R ^ { 2 } )$ . $R ^ { 2 }$ explains how correctly the model predict visitation.[29] Higher values indicate strong power.

$$
R ^ { 2 } = 1 - \frac { \sum _ { i = 1 } ^ { n } \left( y _ { i } - \hat { y } _ { i } \right) ^ { 2 } } { \sum _ { i = 1 } ^ { n } \left( y _ { i } - \bar { y } \right) ^ { 2 } } ,\tag{3}
$$

where ¯y denotes the mean of the ground-truth values.

where,

$y _ { i }$ is the actual (observed) value.

• $\hat { y } _ { i }$ is the predicted value.

• $\bar { y }$ is the mean of the actual (observed) values.

Root Mean Square Error Root Mean Squared Error (RMSE) measures the average magnitude of the prediction error[30]. It is the square root of the Mean Squared Error (MSE), giving the error in the same units as the target variable. The lower the RMSE the better the predictive performance.

$$
{ \mathrm { R M S E } } = { \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left( y _ { i } - { \hat { y } } _ { i } \right) ^ { 2 } } }\tag{4}
$$

where,

• $n$ is the total number of observations.

• $y _ { i }$ is the actual (observed) value.

• $\hat { y } _ { i }$ is the predicted value.

Mean Absolute Percentage Error (MAPE) The Mean Absolute Percentage Error (MAPE) indicates the average prediction error as a percentage of the actual values[31]. It is useful for comparing the accuracy of prediction within diferent data sets at diferent scales. The smaller the value, the more accurate the predictions.

$$
\mathrm { M A P E } = \frac { 1 0 0 } { n } \sum _ { i = 1 } ^ { n } \left| \frac { y _ { i } - \hat { y } _ { i } } { y _ { i } } \right|\tag{5}
$$

where,

• n is the total number of observations.

• $y _ { i }$ is the actual (observed) value.

• $\hat { y } _ { i }$ is the predicted value.

## 2.6 Meta Model

For creating an efective model, we constructed a hybrid metamodel using the top 3 base models that provide the highest performance. Using the same features and the same traintest split, we combined them using two ensemble strategies: (i) Voting regression to reduce variance, (ii) stacking regression to overcome systematic errors. In stacking, we used multiple meta-learners (e.g., Ridge and Gradient Boosting) and trained them using cross-validation. Hybrid models perform better because they integrate complementary predictors, making both models less biased and less variable and increasing the generalisation.

## 3 Results

Table 1 and 2 described the MAE, MSE, RMSE, MAPE, and $\mathrm { R ^ { 2 } }$ of the best baseline models before applying the temporal feature named the lag feature. This scenario is captured before and after applying IQR and anomaly detection techniques.

In term of MAE, both Interquartile Range (IQR) filtering and anomaly removal showed improvement. The lag value indicates the past value of a particular feature. Lag features improved several models, although not all models benefited in every evaluation setting. For example, for the best base predictor before using IQR removal technique and considering the lag feature, the MLP (256,128,64) model decreased the test MAE from 1.705 to 1.323.

Moreover, after applying IQR filtering for the same base model, the test $\mathrm { R ^ { 2 } }$ value increased from 0.991 to 0.997 with a decrease in the test MAE value from 0.640 to 0.613, and a similar improvement was also observed in other metrics such as MSE, RMSE and MAPE. Likewise, after applying IQR and incorporating lag features, considering metrics like MSE, RMSE, and MAPE showed improvement for other predictors like MLP(128,64) and ETR. Moreover, some models, such as ETR, showed a substantially higher MSE before applying IQR filtering (2162.719), and the value dropped significantly to 3.011 that indicated that the IQR filtering method was quite efective in reducing the outliers and improved the predictive performance of some models, and all models performed well with the lag feature.

After anomaly detection, several models showed lower prediction errors, although improvements were not uniform across metrics or lag settings. Table 2 compares MAE, MSE, RMSE, and MAPE before and after anomaly removal. With lag features after anomaly removal, Bayesian Ridge achieved the lowest MAE (0.880), while the MLP achieved the lowest MSE (2.292), RMSE (1.514), and MAPE (3.605%). Moreover, significant improvement was shown in MLP(256,128,64) as MAE dropped from 1.043 to 1.003, MSE reduced from 4.243 to 2.292, and other metrics like RMSE and RMSE also deceased through Anomaly removal and incorporating lag features. This result indicates that after using the anomaly identification approach, the lag feature improved performance. Table 2 indicates that before using the anomaly process, all model were providing strong performance in terms of MAE and MSE WITH LAG conditions. Moreover, many models like Ridge, in terms of MAE, MSE,RMSE, and MAPE showed efective performance. This result reveals that the anomaly detection process enhanced performance for many models.

Figure 5 compares actual and predicted visitation for the base models, while Figure 3 compares the number of observations removed by the two preprocessing methods.

Using the best three performing base models as inputs, the hybrid models were constructed and evaluated. Figures 5 and 6, and Table 3 showed that the impact of anomaly detection showed strong predictive performance with both base and meta-learners. Instead of low MAE and RMSE in base models, hybrid meta-models were more efective by achieving less error in terms of MAE and MAPE. For the single-split comparison, the voting model achieved an MAE of 0.862 and MAPE of 3.325%; the lowest base-model MAE was 0.880, while the lowest base-model MAPE was 3.605%. 5 and 6 showed that, after applying anomaly detection, the diference between the predicted and the actual value for all models (base models and hybrid models) was very minimal, but 3 showed that STACK GBR had an MAE of 0.951 and MAPE of 3.406%; these results are not uniformly lower than those of the base models.

<sub>on</sub> <sub>of</sub> <sub>representative</sub> m<sup>odels</sup> <sup>before</sup> <sup>and</sup> <sup>after</sup> <sup>IQR</sup> <sup>filtering</sup> <sup>with</sup> <sup>and</sup> <sup>wi</sup>
<table><tr><td>Stage</td><td>Model</td><td colspan="2">MAE</td><td colspan="2">MSE</td><td colspan="2">RMSE</td><td colspan="2">MAPE (%)</td><td colspan="2">R2</td></tr><tr><td></td><td></td><td>No Lag</td><td>With Lag</td><td>No Lag</td><td>With Lag</td><td>No Lag</td><td>With Lag</td><td>No Lag</td><td>With Lag</td><td>No Lag</td><td>With Lag</td></tr><tr><td>Before</td><td>MLP (256,128,64)</td><td>1.705</td><td>1.323</td><td>11.738</td><td>5.715</td><td>3.426</td><td>2.391</td><td>5.001</td><td>4.818</td><td>1.000</td><td>1.000</td></tr><tr><td>IQR</td><td>MLP (128,64)</td><td>1.404</td><td>1.426</td><td>6.458</td><td>6.833</td><td>2.541</td><td>2.614</td><td>4.545</td><td>4.805</td><td>1.000</td><td>1.000</td></tr><tr><td>Filtering</td><td>ETR</td><td>4.468</td><td>4.746</td><td>1638.830</td><td>2162.719</td><td>40.482</td><td>46.505</td><td>1.732</td><td>1.828</td><td>0.994</td><td>0.992</td></tr><tr><td>After</td><td>MLP (256,128,64)</td><td>0.640</td><td>0.613</td><td>3.363</td><td>1.020</td><td>1.834</td><td>1.010</td><td>3.004</td><td>2.826</td><td>0.991</td><td>0.997</td></tr><tr><td>IQR</td><td>MLP (128,64)</td><td>0.733</td><td>0.677</td><td>4.018</td><td>1.259</td><td>2.004</td><td>1.122</td><td>3.548</td><td>3.232</td><td>0.989</td><td>0.997</td></tr><tr><td>Filtering</td><td>ETR</td><td>0.559</td><td>0.550</td><td>7.428</td><td>3.011</td><td>2.725</td><td>1.735</td><td>1.823</td><td>1.859</td><td>0.980</td><td>0.992</td></tr></table>

<sub>n</sub> <sub>of</sub> <sub>representative</sub> <sub>models</sub> <sub>before</sub> <sub>and</sub> <sub>after</sub> <sub>ano</sub>m<sup>aly</sup> <sup>removal</sup> <sup>with</sup> <sup>and</sup> <sup>with</sup>
<table><tr><td>Stage</td><td>Model</td><td colspan="2">MAE</td><td colspan="2">MSE</td><td colspan="2">RMSE</td><td colspan="2">MAPE (%)</td><td colspan="2">R²</td></tr><tr><td></td><td></td><td>No Lag</td><td>With Lag</td><td>No Lag</td><td>With Lag</td><td>No Lag</td><td>With Lag</td><td>No Lag</td><td>With Lag</td><td>No Lag</td><td>With Lag</td></tr><tr><td>Before</td><td>Bayes</td><td>0.890</td><td>0.891</td><td>2.949</td><td>4.494</td><td>1.717</td><td>2.120</td><td>3.574</td><td>3.617</td><td>1.000</td><td>1.000</td></tr><tr><td>Anomaly</td><td>Ridge</td><td>0.957</td><td>1.211</td><td>3.161</td><td>5.589</td><td>1.778</td><td>2.364</td><td>3.859</td><td>4.681</td><td>1.000</td><td>1.000</td></tr><tr><td>Removal</td><td>MLP (256,128,64)</td><td>1.600</td><td>1.437</td><td>8.076</td><td>7.956</td><td>2.842</td><td>2.821</td><td>4.704</td><td>4.798</td><td>1.000</td><td>1.000</td></tr><tr><td>After</td><td>Bayes</td><td>0.889</td><td>0.880</td><td>3.946</td><td>2.396</td><td>1.987</td><td>1.548</td><td>3.714</td><td>3.668</td><td>1.000</td><td>1.000</td></tr><tr><td>Anomaly</td><td>Ridge</td><td>0.913</td><td>0.886</td><td>4.002</td><td>2.411</td><td>2.000</td><td>1.553</td><td>3.815</td><td>3.682</td><td>1.000</td><td>1.000</td></tr><tr><td>Removal</td><td>MLP (256,128,64)</td><td>1.043</td><td>1.003</td><td>4.243</td><td>2.292</td><td>2.060</td><td>1.514</td><td>3.611</td><td>3.605</td><td>1.000</td><td>1.000</td></tr></table>

Table 3 and Fig. 4 compared the predictive performance of the base models and the proposed hybrid models using five evaluation metrics: $R ^ { 2 }$ , MAE, MSE, RMSE, and MAPE. Among the hybrid models in Table 3, STACK GBR achieved the highest $R ^ { 2 }$ and lowest MSE and RMSE; VOTE achieved the lowest MAE, while STACK Ridge achieved the lowest MAPE. No single hybrid model led every metric. The other hybrid models were competitive on selected metrics, although not all outperformed the base models. These results indicate that the value of hybrid learning depends on the evaluation metric and setting. Moreover, Fig. 7 presents the scatter plot for predicted vs actual result for hybrid meta-models.

Figure file not uploaded:

Picture\_3/metrics11.png

Figure 4: Anomaly -based evaluation of hybrid meta-models using $\mathrm { R ^ { 2 } }$ , MAE, MSE, RMSE, and MAPE.

K-fold validation summary results are shown in Table 4. Diferent split results after applying anomaly detection techniques are shown in Figure 8indicating that the base model and hybrid model performances were similar and all models achieve higher accuracy $( R ^ { 2 } > 9 9 \% )$ Table 5 reports aggregate model metrics rather than fold-level fluctuations. Conversely, ensemble approaches enhance stability: the voting model achieved the lowest MAE, MSE,

![](images/a6626376b773c7c1c1d9af7d0a4c24935767752d1c8978d35be0ae11ac63db71.jpg)  
(a) IQR Technique

![](images/488e24c3be66ef6c897642580ab15b6b51071c33b80e780c3e4528ef56a23d8c.jpg)  
(b) Anomaly Detection Technique

Figure 5: Actual vs. predicted values for the base model after applying (a) the IQR technique and (b) the anomaly detection technique.  
![](images/8908657d4ff3d33c2d31b38a9bfe6397c9b0611643391fddd17122fc17714fad.jpg)  
(a) IQR Technique

![](images/f9172b207564fe8a53317a07e70e1d64ba4211975334b009c7a9a0289e244b3d.jpg)  
(b) Anomaly Detection Technique  
Figure 6: Actual vs. predicted values for the Hybrid after applying (a) the IQR technique and (b) the anomaly detection technique.

Table 3: Performance comparison of the top three base models and top three hybrid models after anomaly removal with lag features.
<table><tr><td>Category</td><td>Model</td><td>MAE</td><td>MSE</td><td> $R ^ { 2 }$ </td><td>RMSE</td><td>MAPE (%)</td></tr><tr><td rowspan="3">Base</td><td>MLP (256,128,64)</td><td>1.003</td><td>2.292</td><td>1.0000</td><td>1.514</td><td>3.605</td></tr><tr><td>Ridge</td><td>0.886</td><td>2.411</td><td>1.0000</td><td>1.553</td><td>3.682</td></tr><tr><td>Bayesian Ridge</td><td>0.880</td><td>2.396</td><td>1.0000</td><td>1.548</td><td>3.668</td></tr><tr><td rowspan="3">Hybrid</td><td>STACK_GBR</td><td>0.951</td><td>3.099</td><td>0.999946</td><td>1.760</td><td>3.406</td></tr><tr><td>VOTE</td><td>0.862</td><td>3.687</td><td>0.999936</td><td>1.920</td><td>3.325</td></tr><tr><td>STACK_Ridge</td><td>0.865</td><td>3.721</td><td>0.999935</td><td>1.929</td><td>3.310</td></tr></table>

![](images/e01a2829941362fe01735542aed0861da30bb6891c643118a06ee18bcecab23d.jpg)  
Figure 7: Scatter plot for actual vs. predicted values for voting, Ridge-based stacking, and GBR-based stacking

RMSE, and MAPE and the highest $R ^ { 2 }$ in the anomaly-based cross-validation comparison. In the reported anomaly-based cross-validation comparison, VOTE achieved the best results on the evaluated metrics. Specifically, the hybrid model reduced MAE by 9.82%, MSE by 3.12%, RMSE by 1.76%, and MAPE by 6.77%, whereas STACK Ridge gave almost identical but higher error across folds, although with somewhat lower accuracy. Overall, the K-fold analysis favored VOTE among the evaluated models. From 4 and 5 depicted that, the best anomaly-based hybrid model (VOTE) reduced the MAE from 0.9070 to 0.8180 (9.81% improvement), the MSE from 3.3776 to 2.2191 (34.30% improvement), the RMSE from 1.8351 to 1.4866 (18.99% improvement), and the MAPE from 3.5409% to 3.5176% (0.66% improvement) compared with the best IQR-based hybrid model (STACK Ridge).Figure 10 compares test MAE for the base and hybrid models after anomaly removal, illustrating the reported performance diference. Moreover, the MSE, RMSE, and MAPE of the hybrid model were also the lowest. When comparing the K-fold validation results of the IQR and anomaly detection techniques, Figure 9 shows that the anomaly detection technique performed better than IQR, with a 9.81% improvement in MAE.

Table 4: K-fold cross-validation performance of hybrid models after IQR-based outlier removal.
<table><tr><td>Hybrid Models</td><td>MAE↓</td><td> $\overline { { \mathbf { M S E } \downarrow } }$ </td><td>RMSE↓</td><td>MAPE  $( \% ) \downarrow$ </td><td> $\overline { { \mathbf { R } ^ { 2 } \dag } }$ </td></tr><tr><td>STACK_Ridge</td><td>0.906990</td><td>3.377647</td><td>1.835120</td><td>3.540943</td><td>0.999987</td></tr><tr><td>VOTE</td><td>2.205798</td><td>300.798465</td><td>16.927260</td><td>3.392077</td><td>0.998871</td></tr><tr><td>STACK_GBR</td><td>4.531633</td><td>750.901946</td><td>23.888775</td><td>7.099465</td><td>0.997227</td></tr><tr><td colspan="6"></td></tr><tr><td colspan="6"><img src="images/258a15b455c9c3088a9f9d60929a5c2c6d7d345f0bc80cc774fed3662ad64716.jpg"/></td></tr></table>

Figure 8: Cross-validation performance (MAE and $\mathrm { R } ^ { 2 } )$ of base and hybrid models

When bias is close to zero, it is good for the model, and when it is negative or positive, bias indicates underprediction or overprediction. Diagram 11, and 12 depicted that, considering IQR, for the hybrid model, negative bias was showing, and variance was showing less than the base model. When compared the variance with the anomaly identification technique was much greater; this indicated that after applying the anomaly detection technique, the mode could capture more diverse prediction patterns. After applying the anomaly identification technique, the hybrid model is more eficient, as it reduced systematic error (bias) while maintaining the same error variance. . Diagram 13 showed that both models are centred to zero, indicating less prediction error. Moreover, the hybrid model showed a narrower distribution, suggesting less variability and more consistent predictions across samples, thus enhancing overall predictive reliability.

Table 5: K-fold performance comparison of the base and hybrid models after anomaly removal
<table><tr><td>Model</td><td>MAE</td><td>MSE</td><td>RMSE</td><td>MAPE (%)</td><td> $R ^ { 2 }$ </td></tr><tr><td colspan="6">Base Models</td></tr><tr><td>MLP (256,128,64)</td><td>0.907074</td><td>2.290680</td><td>1.513177</td><td>3.773141</td><td>0.999803</td></tr><tr><td>Bayes</td><td>0.880754</td><td>2.618634</td><td>1.612985</td><td>3.973741</td><td>0.999774</td></tr><tr><td>Ridge</td><td>0.880911</td><td>2.619215</td><td>1.613165</td><td>3.973542</td><td>0.999774</td></tr><tr><td colspan="6">Hybrid Models</td></tr><tr><td>VOTE</td><td>0.818019</td><td>2.219127</td><td>1.486550</td><td>3.517616</td><td>0.999809</td></tr><tr><td>STACK_GBR</td><td>0.865331</td><td>2.370002</td><td>1.501891</td><td>3.574142</td><td>0.999796</td></tr><tr><td>STACK_Ridge</td><td>1.033668</td><td>4.946523</td><td>2.102236</td><td>4.295037</td><td>0.999572</td></tr><tr><td colspan="6"></td></tr><tr><td colspan="5">Figure file not uploaded: Picture_3/MAE2.png</td></tr></table>

Figure 9: Using K-fold validation comparison of hybrid model test MAE using IQR filtering and anomaly removal

![](images/95c8b1bb8366cc09c8de213172f431472126f0345012a6eda11a45740e409e06.jpg)  
Figure 10: Base model vs Hybrid model using Test MAE

![](images/cff1af6d83809f398cbafb1f5c82a502e0d59cbd76d08bb965ddd6ce769ffdb1.jpg)  
(a) IQR Bias

![](images/5c759a466c2b00ccde55e159608dcdf29081b38e4051b042dcca29c61d4580d2.jpg)  
(b) Anomaly Bias  
Figure 11: Bias comparison of single and hybrid models using (a) IQR and (b) anomaly detection techniques.

![](images/f69cd4b81d736631c26401ee7099844ae55e5077385c0218ce8954e3fd33f945.jpg)

![](images/6c726c14d404da000b07359e682ae53426b2cd01d6ff36916b725d1f36c5a790.jpg)  
Figure 12: Prediction variance of single and hybrid models using (a) IQR and (b) anomaly detection techniques.

![](images/44023f56f3cf280807d31164f672906fcd9dfc6fb610b902663a76b8f4ff1e00.jpg)  
Figure 13: Error distribution of single vs. hybrid models

## 4 Discussion

## Conclusions

This study investigated the efectiveness of combining spatial and temporal features and lag features with ensemble learning to improve visitation count prediction accuracy. Experimental results showed that lag features and anomaly removal reduced prediction error in several settings, with high reported $R ^ { 2 }$ values for many models. Among individual learners, Bayes,Ridge,and multilayer perceptrons achieved consistently high performance. Moreover, hybrid approaches further enhanced robustness and error stability. Anomaly detection removed fewer records than IQR filtering and produced lower errors for the best hybrid models in the reported cross-validation comparison; the difering retained datasets should be considered when interpreting this result.

The hybrid methods showed diferent strengths across evaluation metrics. In the reported anomaly-based cross-validation results, VOTE achieved the lowest MAE, MSE, RMSE, and MAPE and the highest $R ^ { 2 }$ among the compared models. These findings support the usefulness of ensemble methods in this evaluation, without establishing that every hybrid model outperformed every base model. The reported cross-validation summary favored VOTE on the evaluated metrics; however, the table does not provide fold-by-fold variability, so stability across folds cannot be established from those summary values alone. The k-fold validation experiment investigated that the VOTE method worked efectively. Moreover, bias–variance analysis revealed that the proposed hybrid framework substantially decreases prediction bias without increasing variance, indicating improved reliability for real-world deployment.

Overall, the results confirm that integrating anomaly detection, temporal lag features, and hybrid learning provides a robust and accurate framework for large-scale spatiotemporal visitation modelling. Future work may extend this framework by incorporating extra features such as weather, adaptive ensemble weighting, and uncertainty quantification to further enhance operational decision support for resource planning and environmental management.

## References

[1] Noam Shoval and Rein Ahas. The use of tracking technologies in tourism research :the first decade. Tourism Geographies, 18(5):587–606, 2016.

[2] Vincent D Blondel, Adeline Decuyper, and Gautier Krings. A survey of results on mobile phone datasets analysis. EPJ data science, 4(1):10, 2015.

[3] Anne Hardy, Sarah Hyslop, Kate Booth, Brady Robards, Jagannath Aryal, Ulrike Gretzel, and Richard Eccleston. Tracking tourists’ travel with smartphone-based gps technology: a methodological discussion. Information Technology & Tourism, 17(3):255–274, 2017.

[4] Norman Yap, Mingwei Gong, Ranesh Naha, and Aniket Mahanti. Machine learningbased modelling for museum visitations prediction. In Proceedings of the Conference, pages 1–7, 10 2020. doi: 10.1109/ISNCC49221.2020.9297182.

[5] Abang Zainoren, Wan Fairos Wan Yaacob, Syerina Azlin Md Nasir, Serah Jaya, and Suhaili Mokhtar. Using machine learning to predict visitors to totally protected areas in sarawak, malaysia. Sustainability, 14:1–16, 02 2022. doi: 10.3390/su14052735.

[6] Jiaying Chen, Cheng Li, Liyao Huang, and Weimin Zheng. Tourism demand forecasting: a deep learning model based on spatial-temporal transformer. Tourism Review, 80(3): 648–663, 2025.

[7] Abang Zainoren, Wan Fairos Wan Yaacob, Syerina Azlin Md Nasir, Serah Jaya, and Suhaili Mokhtar. Using machine learning to predict visitors to totally protected areas in sarawak, malaysia. Sustainability, 14:1–16, 02 2022. doi: 10.3390/su14052735.

[8] Athanasios Salamanis, Georgia Xanthopoulou, Dionysios Kehagias, and Dimitrios Tzovaras. Lstm-based deep learning models for long-term tourism demand forecasting. Electronics, 11(22):3681, 2022.

[9] Jos´e Carlos Sancho N´u˜nez, Juan A. G´omez-Pulido, and Rafael Ram´ırez. Machine learning applied to tourism: A systematic review. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 14, 07 2024. doi: 10.1002/widm.1549.

[10] Julian Lemmel, Zahra Babaiee, Marvin Kleinlehner, Ivan Majic, Philipp Neubauer, Johannes Scholz, Radu Grosu, and Sophie Neubauer. Prediction of tourism flow with sparse geolocation data, 08 2023.

[11] Ali Louati, Hassen Louati, Meshal Alharbi, Elham Kariri, Turki Khawaji, Yasser Almubaddil, and Sultan Aldwsary. Machine learning and artificial intelligence for a sustainable tourism: A case study on saudi arabia. Information, 15(9), 2024. ISSN 2078- 2489. doi: 10.3390/info15090516. URL https://www.mdpi.com/2078-2489/15/9/516.

[12] Zhenlong Li, Huan Ning, Fengrui Jing, and M Naser Lessani. Understanding the bias of mobile location data across spatial scales and over time: a comprehensive analysis of safegraph data in the united states. Plos one, 19(1):e0294430, 2024.

[13] Daqing Zhang, Nan Li, Zhi-Hua Zhou, Chao Chen, Lin Sun, and Shijian Li. ibat: detecting anomalous taxi trajectories from gps traces. In Proceedings of the 13th international conference on Ubiquitous computing, pages 99–108, 2011.

[14] Song Gao. Spatio-temporal analytics for exploring human mobility patterns and urban dynamics in the mobile age. Spatial Cognition & Computation, 15(2):86–114, 2015.

[15] Ali Hamdi, Khaled Shaban, Abdelkarim Erradi, Amr Mohamed, Shakila Rumi, and Wei Shao. Spatiotemporal data mining: A survey on challenges and open problems, 03 2021.

[16] Ximeng Cheng, Zhiqian Wang, Xuexi Yang, Liyan Xu, and Yu Liu. Multi-scale detection and interpretation of spatio-temporal anomalies of human activities represented by timeseries. Computers, Environment and Urban Systems, 88:101627, 07 2021. doi: 10.1016/ j.compenvurbsys.2021.101627.

[17] Yanan Xin. Anomaly detection for volunteered geographic information: a case study of safecast data. International Journal of Geographical Information Science, 36(7):1423– 1442, 2022.

[18] Marco AF Pimentel, David A Clifton, Lei Clifton, and Lionel Tarassenko. A review of novelty detection. Signal processing, 99:215–249, 2014.

[19] Yang Cao, Haolong Xiang, Hang Zhang, Ye Zhu, and Kai Ming Ting. Anomaly detection based on isolation mechanisms: A survey. Machine Intelligence Research, 22(5):849–865, 2025.

[20] Guilherme O Campos, Arthur Zimek, J¨org Sander, Ricardo JGB Campello, Barbora Micenkov´a, Erich Schubert, Ira Assent, and Michael E Houle. On the evaluation of unsupervised outlier detection: measures, datasets, and an empirical study. Data mining and knowledge discovery, 30(4):891–927, 2016.

[21] Philipp Probst, Marvin N Wright, and Anne-Laure Boulesteix. Hyperparameters and tuning strategies for random forest. Wiley Interdisciplinary Reviews: data mining and knowledge discovery, 9(3):e1301, 2019.

[22] Alexey Natekin and Alois Knoll. Gradient boosting machines, a tutorial. Frontiers in neurorobotics, 7:21, 2013.

[23] Minmin He and Xiyuan Qian. Forecasting tourist arrivals using stl-xgboost method. Tourism Economics, page 13548166241313411, 2025.

[24] Hoda Khoshvaght, Ratish Ramyad Permala, Amir Razmjou, and Mehdi Khiadani. A critical review on selecting performance evaluation metrics for supervised machine learning models in wastewater quality prediction. Journal of Environmental Chemical Engineering, 13(6):119675, 2025.

[25] Oona Rainio, Jarmo Teuho, and Riku Kl´en. Evaluation metrics and statistical tests for machine learning. Scientific Reports, 14(1):6086, 2024.

[26] Maxim Vladimirovich Shcherbakov, Adriaan Brebels, Nataliya Lvovna Shcherbakova, Anton Pavlovich Tyukov, Timur Alexandrovich Janovsky, Valeriy Anatol’evich Kamaev, et al. A survey of forecast error measures. World Applied Sciences Journal, 24(24):171– 176, 2013.

[27] Boran Sekeroglu, Yoney Kirsal Ever, Kamil Dimililer, and Fadi Al-Turjman. Comparative evaluation and comprehensive analysis of machine learning models for regression problems. Data Intelligence, 4(3):620–652, 2022.

[28] Tianfeng Chai and Roland R Draxler. Root mean square error (rmse) or mean absolute error (mae)?–arguments against avoiding rmse in the literature. Geoscientific model development, 7(3):1247–1250, 2014.

[29] Davide Chicco, Matthijs J Warrens, and Giuseppe Jurman. The coeficient of determination r-squared is more informative than smape, mae, mape, mse and rmse in regression analysis evaluation. Peerj computer science, 7:e623, 2021.

[30] Timothy O Hodson. Root mean square error (rmse) or mean absolute error (mae): When to use them or not. Geoscientific Model Development Discussions, 2022:1–10, 2022.

[31] Ziyi Tian, Xiao Wang, Yan Wang, and Jae Ho Lee. Enhancing museum visitor forecasting using deep learning and sentiment analysis: A transformer-based approach for sustainable management. Plos one, 20(11):e0335623, 2025.