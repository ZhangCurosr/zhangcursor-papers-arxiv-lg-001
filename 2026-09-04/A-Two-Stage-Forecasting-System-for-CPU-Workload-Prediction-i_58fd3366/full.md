# A Two-Stage Forecasting System for CPU Workload Prediction in Private Clouds

Ashir Javeed<sup>1</sup>, Anton Borg<sup>∗1</sup>, H˚akan Grahn<sup>1</sup>, Lars Lundberg<sup>1</sup>, Dhyey Patel<sup>2</sup>, and Sogand Shirinbab<sup>2</sup>

<sup>1</sup>Department of Computer Science, Blekinge Institute of Technology, 371 79, Sweden <sup>2</sup>Ericsson AB, 371 33 Karlskrona, Sweden

## Abstract

Accurate cloud resource forecasting is essential for proactive resource provisioning, maintaining Quality of Service (QoS), and reducing operational costs in dynamic cloud environments. The existing forecasting approaches predominantly estimate future CPU workload directly from historical resource traces, which often overlook the relationship between customer service demand and subsequent resource consumption. This study proposes a two-stage integrated forecasting model that explicitly models this dependency by first forecasting customer service requests, expressed as Transactions Per Second (TPS), and subsequently estimating future CPU workload from the TPS forecast. Both the forecasting component and resource prediction component employed the XGBoost model within a cascaded learning architecture, complemented by adaptive online retraining using an expanding-window strategy to address concept drift in continuously evolving cloud workloads. The proposed work was evaluated using real-world traces collected from a private cloud environment comprising ten applications. Experimental results demonstrate robust forecasting performance by achieving Symmetric Mean Absolute Percentage Error (SMAPE) below 7% for most applications, with the best-performing application achieving an MAE of 0.7372, RMSE of 1.1866, SMAPE of 3.57%, and an $R ^ { 2 }$ of 0.9185. Horizon-wise drift analysis confirmed stable recursive forecasting behavior with controlled error accumulation across a 60-step prediction horizon. Compared with the conventional direct CPU forecasting method, the newly developed two-stage integrated model gives improved forecasting robustness, computational eficiency, and interpretability, making it well-suited for proactive resource management and intelligent auto-scaling in cloud computing environments.

## 1 Introduction

Cloud computing has experienced tremendous growth in recent years, driven by lower operating costs and increased productivity [1]. The recent advancements in artificial intelligence (AI) and Big data also contribute to the steadily rising demand for cloud computing, as these technologies require significant computational resources [2]. To maintain the Quality of Service (QoS), cloud service providers should allocate and manage computing resources eficiently while respecting energy and sustainability factors. Over-provisioning of cloud resources costs in terms of energy waste, and under-provisioning afects QoS [3]. Therefore, cloud computing resources such as processing (CPU) power, memory, and storage should be allocated eficiently to meet cloud customers’ needs and energy. Consequently, maintaining smooth cloud operations and eficiency requires accurate forecasting of future demand for cloud resources.

Cloud service providers rely on workload forecasting to proactively allocate computing resources and maintain QoS under dynamic workload conditions [4]. In this regard, historical data can be utilized for building a predictive model for forecasting an estimated demand of computing resources such as CPU, storage, memory, and network bandwidth for future needs. However, accurately estimating future demand for computing resources is not an easy task because workload in data centers is highly dynamic and rapidly changing for demanded resources. For instance, 5% to 80% variation in cloud workload is observed in Alibaba cloud datacentres [5]. Similar variation in cloud workload is also studied in this study, where data is obtained from a private cloud services provider company. Developing an eficient method that can accurately address high variance in cloud resource demand requires an in-depth understanding and knowledge of workload patterns [6]. Through acquired knowledge, we can construct an algorithm that is robust and precise for forecasting the future workload of the cloud.

Generally, cloud data is sequential in nature and collected over time, which turns resource forecasting into a time-series problem [7]. Therefore, researchers have utilized statistical methods such as the ARIMA model and machine learning models, including linear regression, decision, and most recent methods based on deep learning, such as recurrent neural network (RNN) and Long Short-Term Memory (LSTM), to deal with the problem as a time series task [8, 9]. These models successfully predict future cloud workload, but they do not account for the uncertainty of such forecasts due to limited data on cloud customer behavior. Furthermore, few studies in the literature considered the factor of services requested by the cloud customers for future workload forecasting [10, 11, 12]. Additionally, forecasting resource requirements across cloud applications based on customer demand remains a challenge for cloud researchers. Moreover, the presence of high fluctuations across diferent services based on customers’ demand at various time-steps, and their impact on cloud resources such as CPU utilization in the future, poses a significant challenge for accurate forecasting of future resource (CPU). Hence, there is a need for a mechanism that accounts for customer service requests at a given time step for a particular cloud application and accurately forecasts the future resources needed to handle customer demand. Existing forecasting methods directly model future CPU utilization from historical CPU traces. However, CPU consumption is primarily a downstream consequence of incoming service demand. Ignoring this causal dependency limits forecasting robustness under highly dynamic cloud workloads.

To overcome these limitations, this study proposes a two-stage integrated forecasting system for proactive CPU resource prediction in cloud environments. We build on our previous work [10], where customer requests were used as input features to train a machine learning model for CPU workload prediction in cloud applications. In the present study, a dedicated forecasting model is developed to predict future customer requests, which are subsequently utilized as input features for downstream CPU workload estimation. The integration of the CPU model with the forecasting model provides a comprehensive understanding of feature (customer requests) interactions over time, thereby enhancing the model’s forecasting capabilities. Contrary to computationally expensive deep sequential forecasting architectures such as LSTM and Transformer-based forecasting models [8], the proposed XGBoost-based framework supports lightweight adaptive learning with lower retraining overhead and faster inference for streaming cloud environments. The contributions and findings of our work are summarised as follows:

• The study presents an integrated forecasting system that models customer service requests and downstream CPU utilization in two sequential learning stages.

• Developed a TPS-driven forecasting mechanism that captures workload dynamics prior to CPU estimation for proactive cloud resource management.

• Adaptive online retraining and rolling maximum (Rolling Max) smoothing were integrated into the proposed system to enhance forecasting stability under highly dynamic cloud computing workloads.

• Demonstrated computationally eficient forecasting using XGBoost-based learning suitable for lightweight streaming deployment.

• Validated the proposed system using real-world private cloud traces and showed improved forecasting accuracy and robustness compared with conventional forecasting approaches.

The rest of the study is structured as follows. Section 2 presents related work. Section 3 presents the methodology. The experimental results and the statistical analysis are elaborated in Section 4. Section 5 provides a detailed discussion of the given work. Finally, Section 6 presents the conclusion.

## 2 Related Work

Resource forecasting in cloud data center is a critical component for eficient resource management; therefore, significant research attention has been devoted in the literature to address this problem. Researchers have employed diferent methodologies based on statistics, conventional machine learning and state-ofthe-art novel techniques for accurate resource forecasting. Statistical models are primarily based on mathematical principles that help to identify the trends and patterns in the data. However, these models struggle to capture the patterns when the data contains complex and nonlinear relationships. Consequently, accurate resource forecasting in cloud environments becomes challenging when relying on traditional statistical models such as ARIMA [13]. ARIMA is based on AutoRegressive (AR) and Moving Average (MA) principles, which are effective in capturing stationary trends and seasonality. Therefore, ARIMA has limited capability in capturing nonlinear relationships and complex patterns in cloud resource forecasting data [14].

On the other hand, ML models demonstrate superior capability in learning nonlinear workload patterns for eficient cloud resource forecasting over statistical models. ML models are sometimes computationally expensive because they require feature engineering [15] to learn complex pattern in the data and hyperparameter tuning to improve performance [16]. For instance, in literature diferent ML models are utilized for resource prediction and forecasting in cloud environment such as random forest [17], XGBoost [18], K-nearest neighbors regression [19], support vector regression [20], and ensemble [21, 9]. Although machine learning approaches improve forecasting capability under nonlinear cloud workload conditions, their performance often depends heavily on feature engineering and hyperparameter optimization. Furthermore, many ML-based forecasting systems primarily focus on prediction accuracy, ofering limited adaptability to evolving workload dynamics in streaming cloud environments.

A. Rathee et al. proposed an integrated model (XGBoost and Red Fox Optimization Algorithm (RFOA)) for optimized resource allocation in a cloud environment. In their proposed model, RFOA algorithm was deployed to optimize the hyperparameters of XGboost model. The performance of the proposed model was compared with the baseline and genetic algorithm (GA)-based models. Experimental results displayed that proposed model achieved superior performance in term of accuracy and resource utilization while reduced computational cost and improve reliability of cloud services [18]. G. Sripathi and D. A. Khan presented a proactive resource allocation framework using support vector machine (SVM) regression model for cloud computing environment. The developed framework employed workload and geographic information to predict QoS requirements and dynamic allocated resources across geo-distributed cloud locations. The proposed method helps in eficient resource utilization, reduced overprovisioning and underutilization, and enhanced overall system responsiveness and QoS performance. Experimental results demonstrated significant improvements in prediction accuracy, precision, recall, and F1-score compared to existing methods [20].

Furthermore, N. Roy et al. presented an autoscaling predictive method for workload forecasting in cloud environment for improved Quality of Service (QoS). Their predictive model is based on a second-order autoregressive moving average method that accurately forecasts future workload and satisfies the application QoS while keeping operational costs low [22]. Y. C. Chang et al. developed an algorithm based on Recurrent Neural Network (RNN) model for eficient workload forecasting in cloud computing servers. Their proposed method helps to proactive resource allocation through predicting future workload in advance that improve resource management in the cloud [23]. Z. Chen et al. proposed a hybrid Fuzzy Neural Network (FNN) with ensemble modelling approach for accurate forecasting of future workload in cloud. The developed model combined fuzzy logic and ensemble learning for better capturing the complex workload patterns in the cloud environment that helps in accurate forecasting of resources and eficient cloud resource management [24]. Z. Chen et al. developed an algorithm, namely the Prediction Algorithm for Cloud Computing Workloads, based on a deep learning technique for cloud workload forecasting. The performance of the conventional RNN model was improved by integrating a Top-Sparse Autoencoder (TSA) with a Gated Recurrent Unit (GRU). TSA helps by extracting the useful workload patterns from the data, and the future workload is accurately predicted with GRU. Their proposed algorithm was evaluated on the real-world workload traces from Google and Alibaba Cloud. The experimental results showed that the proposed algorithm outperformed the traditional RNN model in workload forecasting accuracy [25]. Although deep learning models improve nonlinear sequence modeling capability, they typically require computationally expensive retraining, large-scale datasets, and substantial hardware resources, limiting their practicality for lightweight adaptive cloud deployment.

In 2021, M. A. Razz et al. designed a hybrid auto-scaling framework for workload forecasting in cloud computing environment. The developed framework is based on an ensemble technique that employs several ML models, such as a support vector machine, a decision tree, a random forest, and a K-nearest neighbors. To identify the dynamic workload variations and better Quality of Service (QoS), their framework introduced a burst-aware auto-scaling mechanism that improved resource utilization via vertical and horizontal scaling and lowered computational overhead. The proposed framework also analyzed the service requests by the cloud users to tackle the sudden workload burst efficiently [26]. S. Karimunnisa et al. proposed a workload forecasting model by employing deep learning techniques to tackle dynamic workload patterns in cloud computing. Their proposed model integrated Deep Belief Network (DBN) with the Lion Algorithm (LA) that optimized the hidden neuron of the DBN model for improve prediction accuracy. The study experimental results demonstrated that proposed model outperformed existing state-of-the-art forecasting approaches [27].

Recently in 2025, A. Rossi et al. presented a Bayesian deep learning (BDL) model that forecast future workload of a cloud to improve better resource allocation and QoS reliability in cloud computing environments. The presented model introduced univariate and bivariate prediction models using Hybrid Bayesian Neural Networks (BNNs) and probabilistic Long Short-Term Memory (LSTM) networks to forecast multiple cloud resources. The developed method also employed transfer learning techniques to improve adaptability across diferent cloud data centres. Experimental evaluation done using Google and Alibaba cloud datasets demonstrated improved forecasting accuracy and better service-level performance compared to traditional forecasting models [28]. Table 1 presents the summary of studies discussed in the literature that consists of study references, key contribution, methodology and findings of the given study.

Most current cloud forecasting research uses historical resource traces to directly estimate future CPU consumption. However, inbound workload intensity and customer service demand are intrinsically what cause CPU consumption. As a result, in cloud settings, direct forecasting methods might not adequately capture the causal relationship between workload evolution and downstream infrastructure response.

Although significant progress has been made in cloud resource forecasting, several challenges remain: statistical models often struggle to capture complex, nonlinear workload patterns, and machine learning and deep learning approaches typically require substantial computational resources, extensive hyperparameter tuning, and large-scale training data. In addition, many existing methods primarily focus on improving prediction accuracy while inadequately addressing real-time adaptability and eficient resource management in cloud environments. Therefore, there is a need for a simple, eficient forecasting system that accurately predicts cloud resource (CPU) demand in a private cloud system while maintaining computational eficiency. In contrast to traditional direct forecasting techniques, which estimate future CPU utilization solely from historical CPU traces, the proposed two-stage integrated system decomposes the forecasting problem into two sequential learning stages: customer request forecasting and downstream CPU estimation. The proposed cascaded formulation explicitly models the causal relationship between incoming service demand and application-level resource consumption (CPU). Because CPU utilization is mostly determined by workload intensity, forecasting TPS before CPU prediction allows the system to detect workload evolution sooner than direct resourceforecasting approaches. Furthermore, decoupling workload dynamics from infrastructure response simplifies forecasting, increases interpretability, and enhances flexibility in highly dynamic private cloud environments.

## 3 Methodology

This section presents the proposed two-stage integrated forecasting system for proactive CPU resource prediction in private cloud environments. The system integrates multivariate Transactions Per Second (TPS) forecasting with a downstream CPU estimation model using a unified machine learning pipeline. Unlike traditional approaches that directly map historical CPU usage to future demand, the proposed method explicitly models customer service requests as an intermediate representation, enabling earlier and more robust prediction of CPU utilization under dynamic workload conditions.

<sub>mparative</sub> <sub>su</sub>m<sup>mary</sup> <sup>of</sup> <sup>cloud</sup> <sup>resource</sup> <sup>forecas</sup>
<table><tr><td>Ref.</td><td>Methodology</td><td>Forecast Target</td><td>Adaptive Learning</td><td>Dataset / Environment</td><td>Key Limitation</td></tr><tr><td>[9]</td><td>Ensemble Learning</td><td>CPU workload prediction</td><td>Yes</td><td>Cloud applications</td><td>Direct CPU forecasting without explicit workload causality modeling.</td></tr><tr><td>[13]</td><td>Statistical (ARIMA)</td><td>Cloud workload</td><td>No</td><td>Cloud workload traces</td><td>Effective for stationary trends but weak for nonlinear and dynamic cloud workloads.</td></tr><tr><td>[17]</td><td>Random Forest</td><td>Resource utilization</td><td>No</td><td>Cloud resource traces</td><td>Requires feature engineering and lacks online adaptability.</td></tr><tr><td>[18]</td><td>XGBoost + RFOA</td><td>Resource allocation</td><td>Partial</td><td>Cloud computing environment</td><td>Hyperparameter optimization adds computational overhead.</td></tr><tr><td>[19]</td><td>KNN Regression</td><td>Workload prediction</td><td>No</td><td>Dynamic cloud workloads</td><td>Sensitive to neighborhood selection and workload variance.</td></tr><tr><td>[20]</td><td>SVM Regression</td><td>QoS-aware resource allocation</td><td>Partial</td><td>Geo-distributed cloud environment</td><td>Computational overhead increases with workload scale and feature complexity.</td></tr><tr><td>[21]</td><td>Ensemble ML</td><td>Cloud resource forecasting</td><td>No</td><td>Cloud workload traces</td><td>Improved accuracy but increased model complexity and retraining cost.</td></tr><tr><td>[22]</td><td>ARIMA-based forecasting</td><td>Autoscaling workload</td><td>No</td><td>Cloud application traces</td><td>Limited support for rapidly fluctuating workloads.</td></tr><tr><td>[23]</td><td>Deep Learning (RNN)</td><td>Workload forecasting</td><td>No</td><td>Cloud server workloads</td><td>High retraining cost and limited long-term dependency learning</td></tr><tr><td>[24]</td><td>Hybrid FNN + Ensemble</td><td>Cloud workload</td><td>No</td><td>Cloud computing environment</td><td>Increased architectural complexity and computational cost.</td></tr><tr><td>[25]</td><td>TSA + GRU</td><td>Cloud workload prediction</td><td>No</td><td>Google and Alibaba traces</td><td>Requires large-scale training data and expensive retraining.</td></tr><tr><td>[26]</td><td>Hybrid Ensemble + Burst-aware Scaling</td><td>Dynamic workload forecasting</td><td>Partial</td><td>Cloud autoscaling environment</td><td>Ensemble framework increases deployment complexity in streaming environments</td></tr><tr><td>[27]</td><td>DBN + Lion Algorithm</td><td>Workload prediction</td><td>No</td><td>Cloud workload traces</td><td>Deep learning optimization introduces high computational overhead.</td></tr><tr><td>[28]</td><td>Bayesian DL (BNN + Probabilistic LSTM)</td><td>Multi-resource forecasting</td><td>Partial</td><td>Google and Alibaba datasets</td><td>High computational complexity and expensive adaptive retraining.</td></tr><tr><td>Current Study</td><td>two-stage integrated XGBoost forecasting</td><td>TPS + CPU forecasting</td><td>Yes</td><td>Private cloud environment</td><td>Adaptive forecasting with explicit workload-to-CPU causal modeling and reduced retraining overhead.</td></tr></table>

The architecture consists of two stages: (i) TPS Forecasting Module and (ii) CPU Resource Estimation Module, as illustrated in Figure 1. The system operates in a walk-forward streaming setting with periodic retraining using an expanding-window strategy to handle concept drift in cloud environments.

![](images/77cf6d8e5fe13f601a363d6041f8a12c119baac239f1f2287fdbe4a09e5a49ee.jpg)  
Figure 1: Architecture of the proposed two-stage integrated forecasting system with adaptive expanding-window retraining for proactive CPU resource prediction in private cloud.

## 3.1 Problem Formulation

Let the multivariate TPS observation at time step o be defined as:

$$
X _ { o } = [ x _ { o } ^ { ( 1 ) } , x _ { o } ^ { ( 2 ) } , \dots , x _ { o } ^ { ( m ) } ] ,\tag{1}
$$

where m denotes the number of TPS-related features.

The corresponding CPU utilization is represented as:

$$
y _ { o } \in \mathbb { R } .\tag{2}
$$

The objective is to predict future CPU utilization over horizon h using historical TPS observations.

## 3.2 Stage 1: TPS Forecasting Module

A supervised learning dataset is constructed using a sliding historical window of size hw = 512:

$$
\mathbf { S } _ { o } = [ \mathbf { x } _ { o - h w + 1 } , \hdots , \mathbf { x } _ { o } ] \in \mathbb { R } ^ { h w \times m } .\tag{3}
$$

A multi-output XGBoost regression model learns the mapping:

$$
f _ { \theta } ( \mathbf { S } _ { o } ) = X _ { o + 1 } .\tag{4}
$$

Multi-step forecasting is performed recursively:

$$
\begin{array} { r } { \hat { X } _ { o + i } = f _ { \theta } ( \hat { \mathbf { S } } _ { o + i - 1 } ) , \quad i = 1 , 2 , \ldots , h , } \end{array}\tag{5}
$$

The resulting TPS forecast trajectory is:

$$
\hat { X } _ { o : o + h } = [ \hat { X } _ { o + 1 } , \dots , \hat { X } _ { o + h } ] .\tag{6}
$$

## 3.3 Stage 2: CPU Resource Estimation Module

The CPU estimation model is trained independently on original TPS feature vectors, as given in Equation 1. When training the CPU target value is defined using a rolling maximum window:

$$
\tilde { y } _ { o } = \operatorname* { m a x } ( y _ { o + 1 } , y _ { o + 2 } , \dots , y _ { o + \beta } ) ,\tag{7}
$$

which captures near-future peak CPU demand while reducing sensitivity to short-term fluctuations.

The CPU estimation model is trained as:

$$
g _ { \phi } ( X _ { o } ) = \tilde { y } _ { o } .\tag{8}
$$

The training dataset is defined as:

$$
\mathcal { D } _ { C P U } = \{ ( X _ { o } , \tilde { y } _ { o } ) \} _ { o = 1 } ^ { N } .\tag{9}
$$

## 3.4 Integrated Prediction Mechanism

During inference, the CPU model directly consumes the forecasted TPS feature from the first stage, without using recursive state representations:

$$
\hat { y } _ { o + i } = g _ { \phi } ( \hat { X } _ { o + i } ) , \quad i = 1 , 2 , \ldots , h .\tag{10}
$$

The final CPU forecast trajectory is:

$$
\hat { Y } _ { o : o + h } = [ \hat { y } _ { o + 1 } , \dots , \hat { y } _ { o + h } ] .\tag{11}
$$

The TPS forecasting model is periodically retrained using an expandingwindow strategy after every r=1000 observations. The retraining interval (r=1000) was empirically selected to balance adaptation speed, prediction accuracy, and computational overhead. More frequent retraining increases computational cost with only marginal gains in accuracy, whereas less frequent retraining delays adaptation to changes in workload patterns. This strategy enables the model to eficiently adapt to evolving workload distributions while mitigating concept drift in streaming cloud environments.

Algorithm 1 summarizes the complete workflow of the proposed system. The algorithm first trains a multivariate TPS forecasting model using a slidingwindow representation of historical workload data. It then independently trains a CPU estimation model using instantaneous TPS feature vectors paired with a forward-looking rolling maximum CPU target. During deployment, the system operates in a walk-forward manner where TPS values are recursively forecasted over the prediction horizon. These predicted TPS values are then directly passed to the CPU model to estimate future CPU utilization. Periodic retraining of the TPS model using an expanding window ensures adaptability to non-stationary workload dynamics while maintaining computational eficiency.

Algorithm 1: Adaptive Two-Stage TPS-Driven CPU Forecasting   
Input: TPS data X, CPU data Y, window size hw, horizon ${ \overline { { h , } } }$ rolling   
window $\beta ,$ retraining interval r   
Output: Predicted CPU trajectory $\hat { \mathbf { Y } } _ { o : o + h }$   
Construct TPS training set $\mathcal { D } _ { T P S } = \{ ( \mathbf { S } _ { o } , X _ { o + 1 } ) \}$ and train TPS model   
$f _ { \theta } .$   
Construct CPU training set $\mathcal { D } _ { C P U } = \{ ( X _ { o } , \tilde { y } _ { o } ) \}$ , where   
$\tilde { y } _ { o } = \operatorname* { m a x } ( y _ { o + 1 } , \dots , y _ { o + \beta } )$ , and train CPU model $g _ { \phi }$   
for each time step o do   
Append newly observed TPS and CPU data.   
if o mod $r = 0$ then   
Retrain $f _ { \theta }$ using the expanding-window dataset.   
Initialize $\hat { \mathbf { S } } _ { o } \gets \mathbf { S } _ { o } .$   
for $i = 1$ to h do   
$\hat { X } _ { o + i } = f _ { \theta } ( \hat { \mathbf { S } } _ { o + i - 1 } )$   
Update $\hat { \mathbf { S } } _ { o + i }$ with $\hat { X } _ { o + i }$   
$\widehat { \tilde { y } } _ { o + i } = g _ { \phi } ( \hat { X } _ { o + i } )$   
$\hat { \mathbf { Y } } _ { o : o + h } = [ \widehat { \tilde { y } } _ { o + 1 } , \dots , \widehat { \tilde { y } } _ { o + h } ]$   
return $\hat { \mathbf { Y } } _ { o : o + h }$

## 3.5 Dataset Collection and Environment

The dataset used in this study was collected from a private cloud environment. Resource (CPU) utilization, along with the cloud services requested by customers, was recorded at 1-minute intervals. Data were collected for one day, resulting in a total of 1,440 samples, which was insuficient for the scope of this study. Therefore, the dataset was expanded by replicating and mirroring the original data points, increasing its size by approximately three times to obtain 8,514 samples, representing roughly six days of data.

The applications are organized into pods in the cloud computing environment, where a pod represents a group of one or more applications deployed together and managed as a single operational unit. In this study, we focused on applications whose CPU utilization is primarily influenced by user service requests. Four pods, A, B, C, and D, were selected from the private Cloud. Pods A and C each contain one application, pod B contains three (B1, B2, B3), and pod D contains five (D1–D5). Confidentiality agreements with the data provider prevent the disclosure of the specific definitions of certain attributes (e.g., specific service-level metrics or measurement units). However, selected features align with those explored in similar feature-based studies (e.g., Javeed et al. [9], Wang et al. [29]), which typically capture user service request patterns. Table 2 summarizes the details of each pod, including the number of input features per application, and the CPU utilization statistics: range, mean, and standard deviation (SD). Moreover, the CPU utilization range in Table 2 is multiplied by a constant factor to hide sensitive data information.

Table 2: Description of the selected applications.
<table><tr><td>Pod</td><td>Application</td><td>Features (TPS)</td><td>CPU Range</td><td>Mean</td><td>SD</td></tr><tr><td>A</td><td>A</td><td>5</td><td>03.8342 – 49.6500</td><td>30.8940</td><td>06.6864</td></tr><tr><td></td><td>B1</td><td>5</td><td>67.6155 -275.625</td><td>187.890</td><td>47.4330</td></tr><tr><td>B</td><td>B2</td><td>5</td><td>04.8462 – 12.2025</td><td>06.4925</td><td>01.5356</td></tr><tr><td></td><td>B3</td><td>5</td><td>06.6900– 18.1230</td><td>11.1893</td><td>02.2032</td></tr><tr><td>C</td><td>C</td><td>5</td><td>02.2169– 61.1310</td><td>24.6960</td><td>11.4692</td></tr><tr><td></td><td>D1</td><td>5</td><td>0.04910–0.08250</td><td>0.06660</td><td>0.00840</td></tr><tr><td></td><td>D2</td><td>5</td><td>0.01940–0.15590</td><td>0.06660</td><td>0.05390</td></tr><tr><td>D</td><td>D3</td><td>5</td><td>00.8825 – 02.5704</td><td>01.5798</td><td>0.68970</td></tr><tr><td></td><td>D4</td><td>5</td><td>0.00045-0.01425</td><td>00.0048</td><td>0.00530</td></tr><tr><td></td><td>D5</td><td>5</td><td>0.00255 – 0.01125</td><td>00.0053</td><td>0.00230</td></tr></table>

The incoming service requests from users to the private cloud environment are shown in Figure 2. Although timestamps are not explicitly modeled, the data is treated as a sequential time series using observation order. Therefore, the x-axis of Figure 2 represents the observation index corresponding to individual data samples for the given service request. The y-axis shows the number of customer service requests (TPS). The true values on the y-axis for requested services were anonymized to protect the privacy of data.

All experiments were conducted using Python (version 3.12.3) on a local machine. The details of the computing environment are summarized in Table 3.

Table 3: Setup of computing environment.
<table><tr><td>System properties</td><td>Specifications</td></tr><tr><td>CPU</td><td>Intel(R) Core i5-1145G7 @ 2.60GHz</td></tr><tr><td>RAM</td><td>32GB</td></tr><tr><td>OS</td><td>Windows 10 Enterprise</td></tr><tr><td>Storage</td><td>475GB</td></tr><tr><td>IDE</td><td>Visual Studio 1.92</td></tr></table>

![](images/5773fcfbe2cef459afdb7f28895ce8eb1708460a666beb3245ba3c2354901ef2.jpg)  
Figure 2: Flow of incoming customer service requests (TPS) to the private cloud

## 3.6 Validation and Evaluation Metrics

For the problem addressed in this study, we employed a walk-forward validation scheme, which is well-suited to time-dependent data and forecasting tasks that require temporal ordering to be preserved [30]. In the walk-forward validation method, the model is iteratively trained on a historical window while evaluated on the subsequent unseen observation period. This validation method closely simulates real-world deployment conditions [31]. Hence, this procedure eliminates the risk of information leakage from future data and provides a realistic estimate of predictive performance under non-stationary dynamics [32]. Initially, the first 25% of the dataset was used to train the two-stage forecasting system. During walk-forward validation, predictions were generated sequentially for incoming observations, while the training window was expanded and the model was retrained after every r=1000 observations. This strategy preserves temporal ordering while reducing computational overhead. To assess the feasibility of real-time deployment, the execution time of each forecasting iteration was also recorded during inference. The average runtime is computed as:

$$
R u n t i m e _ { a v g } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } t _ { i }\tag{12}
$$

where $t _ { i }$ represents the execution time for the $i ^ { t h }$ forecasting iteration.

The accuracy of regression-based problems is measured in terms of error [33]. Therefore, we have selected several error metrics to rigorously evaluate the eficiency of the constructed integrated forecasting model, such as root mean square error (RMSE) [34], mean underestimate error (MUE) [35], mean overestimate error (MOE) [36], mean absolute error (MAE), and symmetric mean absolute percentage error (SMAPE). SMAPE was included because it provides a scaleindependent and symmetric evaluation of forecasting accuracy, making it suitable for dynamic cloud workloads with fluctuating CPU utilization patterns. In addition, MUE and MOE were selected to analyze asymmetric underestimation and overestimation behaviors in cloud resource provisioning scenarios. Because the CPU utilization ranges difer across the evaluated applications $( \mathrm { e . g . , }$ A, B1, C, and D1), relative metrics such as rMUE and rMOE were employed to enable fair comparisons across workloads with diferent scales. In addition to error metrics, we have also employed R-squared $( R ^ { 2 } )$ to assess how well the constructed forecast system predictions match the observed CPU load. By comparing predicted values with actual CPU load, the performance of the proposed system is evaluated while accounting for the total data variability of the cloud environment. The mathematical formulas for the selected metrics are as follows:

$$
R M S E = \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( y _ { i } - \hat { y } _ { i } ) ^ { 2 } }\tag{13}
$$

$$
\mathrm { M A E } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left| y _ { i } - \hat { y } _ { i } \right|\tag{14}
$$

$$
S M A P E = \frac { 1 0 0 } { n } \sum _ { i = 1 } ^ { n } \frac { \left| y _ { i } - \hat { y } _ { i } \right| } { \left( \left| y _ { i } \right| + \left| \hat { y } _ { i } \right| \right) / 2 }\tag{15}
$$

$$
r M U E = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \frac { \operatorname* { m a x } ( 0 , y _ { i } - \hat { y } _ { i } ) } { y _ { i } }\tag{16}
$$

$$
r M O E = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \frac { \operatorname* { m a x } ( 0 , \hat { y } _ { i } - y _ { i } ) } { y _ { i } }\tag{17}
$$

Where $y _ { i }$ is the actual CPU value, $\hat { y } _ { i }$ is the predicted CPU value by the ML model, and n is the number of data points. In Equation (16), max $( 0 , y _ { i } - { \hat { y } } _ { i } )$ ensures that only the underestimation errors (where the actual value is greater than the predicted value) are considered, similarly, in Equation (17), max(0, $\hat { y } _ { i } -$ $y _ { i } )$ ensures that only the overestimation errors (where the predicted value is greater than the actual value) are considered.

$$
R ^ { 2 } = 1 - \frac { \sum _ { i = 1 } ^ { n } ( y _ { i } - \hat { y } _ { i } ) ^ { 2 } } { \sum _ { i = 1 } ^ { n } ( y _ { i } - \bar { y } ) ^ { 2 } }\tag{18}
$$

$R ^ { 2 }$ is the coeficient of determination that helps in evaluating the goodnessof-fit of the proposed forecasting system. When $\bar { R ^ { 2 } }$ equals 1, it means the model perfectly predicts the CPU load without any error, explaining all the variance in the actual CPU load, and when $R ^ { 2 }$ equals 0, it means the model does not explain any of the variance in the CPU load and performs no better than simply predicting the mean of the observed CPU load. Even in some cases, $R ^ { 2 }$ can be negative $( R ^ { 2 } < 0 )$ ; that means the model performs worse than predicting the mean value for all data points.

## 4 Experimental Results and Analysis

This section presents a detailed description of the experiments and results conducted to evaluate the performance of the newly developed two-stage integrated system for forecasting cloud resource (CPU). Customer service requests (TPS) are used as input features to forecast future CPU workloads for given cloud applications. The experimental results are presented in several subsections. Subsection 4.1 presents the overall performance of the proposed integrated forecasting model across all applications using multiple evaluation metrics. Subsection 4.2 provides a detailed drift analysis of the forecasting model across the applications. Finally, Subsection 4.3 presents the performance of the CPU model that was trained on original TPS values to predict the resource (CPU) across the cloud applications.

## 4.1 Performance Analysis of Proposed Two-stage Forecasting Model

This section presents a detailed experimental analysis of the newly developed two-stage integrated forecasting system for the given ten cloud applications. The performance of the proposed forecasting model was evaluated using several metrics, including Mean Absolute Error (MAE), Root Mean Square Error (RMSE), Symmetric Mean Absolute Percentage Error (SMAPE), Relative Mean Underestimation Error (rMUE), Relative Mean Overestimation Error (rMOE), the coeficient of determination $( R ^ { 2 } )$ , and the average forecasting runtime in milliseconds. The XGBoost hyperparameter configuration used in the proposed forecasting model is first presented, followed by the forecasting performance results summarized in Table 5. The first column lists the cloud applications, while the second column indicates the rolling maximum window size (Win.) used for CPU resource smoothing. The remaining columns report the corresponding evaluation metrics.

Prior to evaluating the forecasting performance, the XGBoost regression model was configured using a fixed set of hyperparameters for this experiment. The model employed 300 boosting estimators with a maximum tree depth of 6 and a learning rate of 0.05, providing a balance between model complexity and generalization capability. To mitigate overfitting, both the row subsampling ratio (subsample) and the feature subsampling ratio (colsample bytree) were set to 0.8. The regression objective was specified as reg:squarederror. Furthermore, the histogram-based tree construction algorithm (tree method = hist) was employed to improve computational eficiency, while parallel execution across all available CPU cores (n jobs = -1) reduced the training time. A fixed random seed (random state = 42) was used to ensure reproducibility of the experimental results. These parameters come from the Python implementations. The complete hyperparameter configuration is summarized in Table 4.

The results from Table 5 suggest that the proposed two-stage forecasting model achieved satisfactory resource (CPU) prediction accuracy for the given cloud applications. The MAE values remained below 1.5 across eight applications, except for application B1, which exhibited a larger prediction error due to its higher workload variability. The lowest MAE was observed for application D5 (0.0005), followed by D4 (0.0011) and D1 (0.0020), which indicates the capability of the proposed model to accurately capture relatively stable workload patterns. Similarly, RMSE values followed a trend consistent with MAE, suggesting that large forecasting deviations were efectively controlled for most applications.

Table 4: XGBoost hyperparameter configuration used in the proposed two-stage forecasting model.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Random State (random_state)</td><td>42</td></tr><tr><td>Number of Estimators (n_estimators)</td><td>300</td></tr><tr><td>Maximum Tree Depth (max_depth)</td><td>6</td></tr><tr><td>Learning Rate (learning-rate)</td><td>0.05</td></tr><tr><td>Row Subsampling (subsample)</td><td>0.8</td></tr><tr><td>Feature Subsampling (colsample_bytree)</td><td>0.8</td></tr><tr><td>Objective Function</td><td>reg:squarederror</td></tr><tr><td>Tree Construction Method (tree_method)</td><td>hist</td></tr><tr><td>Parallel Processing (n_-jobs)</td><td>-1</td></tr></table>

Table 5: Prediction accuracy and computational performance of the proposed two-stage forecasting model across ten cloud applications
<table><tr><td> ${ \overline { { \mathrm { A p p . } } } }$ </td><td>Win.</td><td>MAE</td><td>RMSE</td><td>SMAPE (%)</td><td>rMUE</td><td>rMOE</td><td> $\overline { { \mathrm { R } ^ { 2 } } }$ </td><td> $\overline { { R T _ { a v g } } }$  (ms)</td></tr><tr><td> $\overline { { \mathrm A } }$ </td><td>6</td><td>0.7372</td><td>1.1866</td><td>3.5697</td><td>0.0222</td><td>0.0129</td><td>0.9185</td><td>186.90</td></tr><tr><td>B1</td><td>4</td><td>6.4567</td><td>10.9866</td><td>4.9595</td><td>0.0332</td><td>0.0148</td><td>0.7843</td><td>181.91</td></tr><tr><td>B2</td><td>15</td><td>0.2784</td><td>0.4674</td><td>5.9456</td><td>0.0145</td><td>0.0481</td><td>0.7994</td><td>184.76</td></tr><tr><td>B3</td><td>9</td><td>0.4631</td><td>0.7766</td><td>5.4935</td><td>0.0303</td><td>0.0235</td><td>0.6398</td><td>232.44</td></tr><tr><td>C</td><td>13</td><td>1.3694</td><td>2.8348</td><td>6.3037</td><td>0.0155</td><td>0.0579</td><td>0.7538</td><td>178.50</td></tr><tr><td>D1</td><td>15</td><td>0.0020</td><td>0.0033</td><td>4.0781</td><td>0.0301</td><td>0.0090</td><td>0.6326</td><td>197.56</td></tr><tr><td>D2</td><td>14</td><td>0.0107</td><td>0.0236</td><td>25.4845</td><td>0.0268</td><td>0.4832</td><td>0.6185</td><td>193.35</td></tr><tr><td>D3</td><td>11</td><td>0.1131</td><td>0.2430</td><td>10.9928</td><td>0.0130</td><td>0.1258</td><td>0.6945</td><td>263.70</td></tr><tr><td>D4</td><td>15</td><td>0.0011</td><td>0.0023</td><td>45.7334</td><td>0.0414</td><td>1.3155</td><td>0.6370</td><td>193.34</td></tr><tr><td>D5</td><td>15</td><td>0.0005</td><td>0.0008</td><td>12.2366</td><td>0.0202</td><td>0.1256</td><td>0.6924</td><td>196.02</td></tr></table>

Investigating the efects of rolling maximum-window selection for resource (CPU) smoothing is also crucial for given cloud applications. For all applications, the rolling maximum window size ranged from 4 to 15. This illustrates how varying workload characteristics necessitate varying degrees of smoothing depending on CPU utilization. With comparatively small window sizes (4 and 6), Applications B1 and A achieved ideal performance, indicating the presence of short-term workload changes. On the other hand, applications B2, D1, D4, and D5 needed larger windows of 15, suggesting that more smoothing was helpful for reducing noise and identifying long-term workload trends. These results confirm that the suggested pre-processing mechanism is adaptable.

In terms of percentage-based forecasting accuracy, we selected the SMAPE metric as given in Equation (15). The SMAPE values remained below 11% for most of the applications which confirms the robustness of the forecasting framework under diferent workload scales. Applications A, B1, B2, B3, C, and D1 achieved SMAPE values below 7%, indicating highly accurate relative predictions. However, applications D2 and D4 recorded significantly larger SMAPE values of 25.48% and 45.73%, respectively. Although D4 exhibits the largest SMAPE (45.73%), its absolute MAE is only 0.0011. Therefore, the large percentage error is primarily caused by the extremely small CPU utilization values rather than poor forecasting accuracy.

From Equations (16) and (17), we computed the relative mean underestimation error (rMUE) and relative mean overestimation error (rMOE) for all the given applications using the proposed two-stage integrated forecasting model. These metrics help us to understand the behavior of the proposed forecasting model in terms of overestimation and underestimation. The rMUE remained consistently low across all applications, ranging between 0.0130 and 0.0414. This indicates that the model rarely produced substantial under-predictions. Similarly, the rMOE values were relatively small for most applications, demonstrating balanced forecasting behavior.

Nevertheless, applications D2 and D4 displayed comparatively higher overestimation tendencies with rMOE values of 0.4832 and 1.3155. Such behavior suggests that the forecasting model overestimates workloads when CPU utilization levels are extremely low. From a cloud resource management perspective, moderate overestimation is generally preferable to underestimation because it reduces the risk of service-level agreement (SLA) violations caused by insuficient resource allocation.

Furthermore, the values of the coeficient of determination $( R ^ { 2 } )$ varied between 0.6185 and 0.9185 for all cloud applications. Application A had the greatest value $( R ^ { 2 } = 0 . 9 1 8 5 )$ , meaning that the suggested forecasting model accounted for over 91% of the workload variance. The majority of cloud applications showed a good correlation between expected and actual CPU consumption patterns with $R ^ { 2 }$ values above 0.69. Lower $R ^ { 2 }$ values for D2, D4, and B3 indicate the existence of erratic, hard-to-predict workload changes.

In addition to prediction accuracy, the computational eficiency of the proposed forecasting model was evaluated using Equation (12), which computes the average runtime required to generate resource (CPU) predictions. The average runtime ranged from 178.50 ms to 263.70 ms, with a mean runtime of approximately 201 ms across all applications. These results indicate that the newly developed forecasting model can generate predictions in less than one quarter of a second, making it suitable for real-time cloud resource provisioning and auto-scaling applications. Although application D3 required the highest runtime (263.70 ms), the computational overhead remains suficiently low for practical deployment in dynamic cloud environments.

The presented forecasting model facilitates the proactive resource provisioning in private cloud systems due to its low sub-second runtime. Precise CPU estimates reduce SLA violations and increase resource utilization eficiency by enabling auto-scaling techniques to distribute resources prior to workload peaks.

Since application A ofers a sample example of the forecasting behavior seen throughout the examined cloud applications, only its forecasting findings are graphically given for the sake of conciseness.

![](images/9099dde7ad56eaaf4409cca95ecc0059eb7e7e085c3b66f0d72b889902e2f1a3.jpg)  
Figure 3: Performance analysis of the proposed two-stage integrated forecasting model for the application A, where a forecasting horizon of 60 corresponds to a 1-hour-ahead resource prediction.

Figure 3 presents the forecasting performance of the proposed two-stage integrated forecasting model for application A. The black curve represents the actual normalized CPU Max values, whereas the red curve corresponds to the forecasts generated by the proposed model. The vertical blue dashed line indicates the transition from the training phase to the testing phase, where the initial 25% of the observations were used for model training, and the remaining 75% were reserved for walk-forward evaluation. The forecasting framework employs a rolling window of 512 observations to predict future TPS values over a 60-step forecasting horizon, which the CPU prediction model subsequently uses to estimate future CPU Max utilization.

A visual examination of the testing area shows that the expected and actual CPU workload match quite well. Abrupt workload shifts, high-utilization peaks, and low-utilization periods are only a few of the long-term workload dynamics that the suggested framework efectively captures. Specifically, during periods of relatively low CPU activity, the model maintains constant performance while precisely tracking the significant utilization spikes around observations 4,200 and 7,100. The shaded overestimation and underestimation regions’ modest extent further suggests that predicting discrepancies is minimal for the majority of the evaluation period.

These findings are further supported by the quantitative results shown in the Figure 3 for application A. With an MAE of 0.7372, an RMSE of 1.1866, an SMAPE of 3.57%, and an $R ^ { 2 }$ value of 0.9185, the proposed model efectively explained over 91% of the variability in CPU use. The high coeficient of determination and low forecasting errors indicate that the suggested framework efectively preserves workload characteristics over the forecasting horizon. These findings show that the proposed forecasting model preserves the temporal features of CPU usage dynamics while maintaining forecasting accuracy over long prediction horizons.

Overall, the experimental findings show that the newly developed two-stage forecasting model strikes a good balance between operational robustness, computational eficiency, and forecasting accuracy. Across a variety of cloud applications, the model continuously maintained low forecasting errors, high explanatory power, and quick execution times. These capabilities make the proposed forecasting model eficient for proactive resource provisioning, intelligent autoscaling, and real-time CPU workload predictions in dynamic cloud computing settings.

## 4.2 Drift Analysis of Forecasting Model

Although the overall forecasting accuracy provides a global assessment of model performance, it does not reveal how prediction errors evolve as the forecasting horizon increases. In recursive multi-step forecasting, prediction errors generated at earlier steps are propagated to subsequent predictions, which may lead to the accumulation of forecasting drift over the time. Therefore, a drift analysis was conducted to evaluate the stability and robustness of the proposed two-stage forecasting system across the forecasting horizon.

The following anonymized TPS features were examined for this purpose: Customer Service Request 1 (CSR1), Customer Service Request 2 (CSR2), Customer Service Request 3 (CSR3), Customer Service Request 4 (CSR4), and Customer Service Request 5(CSR5). The original service identifiers were anonymized to preserve the confidentiality of customer applications and cloud services while retaining the statistical characteristics required for forecasting and resource estimation analyses.

Figure 4 illustrates the evolution of forecasting drift across a 60-step prediction horizon using horizon-wise MAE values, starting from 0 to 59 steps where 0 represents the first predicted value. The figure provides a direct visualization of recursive error propagation and highlights the long-term stability characteristics of each TPS trafic stream. For every TPS feature, there is a monotonic rise in predicting error as shown in Figure 4, supporting the anticipated build-up of uncertainty in recursive multi-step forecasting. CRS2 and CSR4 show the sharpest development, with MAE values rising steadily along the forecasting horizon and reaching the maximum error magnitudes at longer prediction intervals. This trend suggests that these trafic streams have more intricate temporal dynamics and are more susceptible to recurrent forecasting errors. On the other hand, CSR1, CSR3, CSR5 show significantly slower drift increase. Their MAE curves show better long-range predictability and higher temporal consistency as they progressively rise and stabilize over longer time periods.

Furthermore, none of the TPS features show an abrupt divergence or an exponential rise in error. Rather, the drift curves remain smooth and roughly linear, suggesting that prediction uncertainty accumulates in a regulated way. This behavior indicates that the forecasting model does not exhibit catastrophic recursive error propagation and remains stable throughout the entire prediction horizon. Therefore, the drift growth study confirms the usefulness of the proposed TPS forecasting component as an input generator for downstream CPU resource prediction and demonstrates its resilience.

![](images/041f33cf79bbef56886794e1f45d6212bfaeecf990aae191fa88571df2042e0f.jpg)  
Figure 4: Evolution of forecasting drift over a 60-step (0-59) prediction horizon measured by horizon-wise MAE. Error increases progressively for all TPS trafic streams, with CSR2 and CSR4 exhibiting the largest drift, while CSR1, CSR3, and CSR5 maintain comparatively stable long-term prediction performance.

## 4.3 Standalone Evaluation of the CPU Prediction Model

Unlike Section 4.1, which evaluates the performance of the complete two-stage forecasting system, this section evaluates the CPU prediction model independently. The objective is to assess the intrinsic prediction accuracy and computational eficiency of the CPU model without the influence of errors propagated from the TPS forecasting stage, thereby distinguishing the contribution of the CPU prediction component from that of the overall forecasting system.

Table 6: XGBoost hyperparameter configuration
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Random State (random_state)</td><td>42</td></tr><tr><td>Number of Estimators (n_estimators)</td><td>300</td></tr><tr><td>Maximum Tree Depth (max_depth)</td><td>4</td></tr><tr><td>Learning Rate (learning-rate)</td><td>0.05</td></tr><tr><td>Row Subsampling (subsample)</td><td>0.8</td></tr><tr><td>Feature Subsampling (colsample_bytree)</td><td>0.8</td></tr><tr><td colspan="2">Objective Function reg:squarederror</td></tr></table>

For the standalone CPU prediction experiment, the XGBoost regression

model was configured using a separate set of hyperparameters to evaluate the intrinsic prediction capability of the CPU model. The model employed 300 boosting estimators with a maximum tree depth of 4 and a learning rate of 0.05. To improve generalization and reduce overfitting, both the row subsampling ratio (subsample) and the feature subsampling ratio (colsample bytree) were fixed at 0.8. The regression objective was specified as reg:squarederror, and a fixed random seed of 42 was used to ensure reproducibility of the experimental results. The complete hyperparameter configuration is presented in Table 6. Unlike to Experiment 4.1, where the XGBoost model employed parallel processing $( n _ { - } j o b s )$ as shown in Table 4, Experiment 4.3 (Table 6) focuses solely on CPU resource prediction without the TPS forecasting stage. Consequently, parallel processing $( n _ { - } j o b s )$ was not utilized, as the experiment was designed to evaluate only the CPU prediction model.

The performance of the resource prediction (CPU) model was evaluated using multiple error metrics, including MAE, RMSE, SMAPE, rMUE, rMOE, and the coeficient of determination $( R ^ { 2 } )$ , across nine cloud applications. In addition, the average response time $( R T _ { a v g } )$ was measured to assess the computational eficiency of the prediction process. The results are summarized in Table 7.

Table 7: Performance of CPU model for resource prediction using original TPS features for given cloud applications
<table><tr><td>App.</td><td>Win</td><td>MAE</td><td>RMSE</td><td>SMAPE (%)</td><td>rMUE</td><td>rMOE</td><td> $\overline { { \mathrm { ~ \mathrm { R } ^ { 2 } } } }$ </td><td> $\overline { { R { T _ { a v g } ( \mathrm { m s } ) } } }$ </td></tr><tr><td>A</td><td>6</td><td>0.216</td><td>0.409</td><td>1.0400</td><td>0.0099</td><td>0.0109</td><td>0.9903</td><td>1.138591</td></tr><tr><td>B1</td><td>4</td><td>2.232</td><td>3.528</td><td>1.7866</td><td>0.0158</td><td>0.0183</td><td>0.9776</td><td>4.530044</td></tr><tr><td>B2</td><td>15</td><td>0.111</td><td>0.216</td><td>2.3929</td><td>0.0224</td><td>0.0249</td><td>0.9567</td><td>7.673111</td></tr><tr><td>B3</td><td>9</td><td>0.242</td><td>0.377</td><td>2.9342</td><td>0.0265</td><td>0.0305</td><td>0.9145</td><td>5.336268</td></tr><tr><td>C</td><td>13</td><td>0.553</td><td>1.123</td><td>2.4841</td><td>0.0258</td><td>0.0269</td><td>0.9611</td><td>1.042598</td></tr><tr><td>D1</td><td>15</td><td>0.001</td><td>0.001</td><td>1.8528</td><td>0.0183</td><td>0.0189</td><td>0.9438</td><td>1.631443</td></tr><tr><td>D2</td><td>14</td><td>0.002</td><td>0.005</td><td>6.2378</td><td>0.0345</td><td>0.0847</td><td>0.9821</td><td>0.842303</td></tr><tr><td>D3</td><td>11</td><td>0.026</td><td>0.061</td><td>3.1652</td><td>0.0259</td><td>0.0346</td><td>0.9809</td><td>0.836380</td></tr><tr><td>D4</td><td>15</td><td>0.000</td><td>0.001</td><td>19.7295</td><td>0.1064</td><td>0.3074</td><td>0.9737</td><td>0.787613</td></tr><tr><td>D5</td><td>15</td><td>0.000</td><td>0.000</td><td>7.4061</td><td>0.0546</td><td>0.0807</td><td>0.9328</td><td>0.771310</td></tr></table>

The consistently high $R ^ { 2 }$ values, ranging from 0.9145 to 0.9903 across the evaluated cloud applications, indicate that the CPU model achieved high prediction accuracy. With an $R ^ { 2 }$ value of 0.9903, a low MAE of 0.216, and a SMAPE of only 1.04%, application A had the best predictive performance, demonstrating outstanding agreement between the projected and real CPU usage statistics. Applications D2 and D3 also demonstrated outstanding predictive skills, sustaining relatively low prediction errors while obtaining $R ^ { 2 }$ values above 0.98.

The prediction performance of B1, despite having the highest absolute error of any application $( \mathrm { M A E } = 2 . 2 3 2$ and $\mathrm { R M S E } = 3 . 5 2 8 )$ ), B1 nevertheless had a good coeficient of determination $( R ^ { 2 } = 0 . 9 7 7 6 )$ , suggesting that the model was successful in capturing the general patterns in CPU consumption. While the SMAPE values of 2.39% and 2.93%, respectively, show acceptable relative prediction accuracy, the MAE values for B2 and B3 stayed below 0.25.

An intriguing finding on the pod D applications. The MAE and RMSE for Applications D4 and D5 were very low, indicating negligible absolute prediction errors. Nonetheless, the SMAPE, rMUE, and rMOE values for these applications were significantly greater. Specifically, D4 exhibited the highest SMAPE (19.73%), rMUE (10.64%), and rMOE (30.74%) among all evaluated applications. This behavior can be attributed to the relatively small magnitude of the actual CPU utilization values in these workloads, where even minor prediction deviations can result in large percentage-based errors. Therefore, the elevated relative error metrics for D4 and D5 should be interpreted in the context of their very low absolute resource utilization levels rather than as indicators of poor model performance.

Additionally, Applications A, B1, B2, B3, D1, and D3 have rMUE values below 3%, and the associated rMOE is also below 4%. These findings show that the model regularly generates predictions with modest overestimation and underestimation behavior that are near the observed CPU usage values. The model’s appropriateness for near real-time resource prediction in cloud settings is demonstrated by the average predicted reaction time, which spans from 0.77 ms to 7.67 ms across all applications. B2 needed the longest average reaction time of 7.67 ms, whereas applications D4 and D5 had the lowest response times of 0.79 ms and 0.77 ms, respectively. However, every prediction time stays within a few milliseconds, indicating that the CPU prediction model is lightweight.

In conclusion, the findings show that the CPU prediction model generates highly accurate and computationally eficient resource projections across a wide range of cloud applications. The model’s continuously high $R ^ { 2 }$ values, low absolute errors, and millisecond-level forecast latency show that it can enable proactive resource management and dynamic scaling decisions in cloud computing.

## 5 Discussion

The newly developed two-stage integrated forecasting model successfully supports proactive cloud resource prediction, as the experimental evaluation confirms. The proposed forecasting methodology continuously attained excellent prediction accuracy while preserving computational eficiency across a wide range of cloud applications by fusing customer service request (TPS) forecasting with CPU workload prediction. The conventional resource prediction approaches directly estimate future CPU workload using timestamps or historical data; the proposed forecasting model first predicts future customer service requests (TPS) behavior and subsequently estimates resource (CPU) demand. This hierarchical modeling strategy enables the forecasting model to capture the causal relationship between workload dynamics and CPU utilization more efectively.

One of the major findings of this study is the consistently low forecasting error across most applications. The obtained MAE, RMSE, and SMAPE values indicate that the proposed forecasting model accurately models both high-load and moderate-load applications despite the workload characteristics. The high coeficient of determination $( R ^ { 2 } )$ observed for nearly all applications further confirms that the proposed forecasting model successfully captures the TPS variability of cloud workloads. Applications exhibiting lower prediction accuracy such as B3, D2, and D4 that are correspond to workloads with highly irregular CPU utilization patterns or extremely low CPU values, making them inherently more dificult to predict. Nevertheless, even in these challenging cases, the forecasting performance remained stable without catastrophic degradation.

Additional proof of the proposed forecasting strategy’s resilience is provided by the drift study. Prediction errors in recursive multi-step forecasting inevitably accumulate as the forecasting horizon lengthens. Rather than exponential drift accumulation, the slow rise in MAE indicates regulated error propagation as shown in Figure 4. In practical cloud resource provisioning, where auto-scaling decisions are directly impacted by prediction reliability over several future intervals, this feature is especially crucial.

A comparison of the results presented in Tables 5 and 7 provides valuable insight into the impact of workload forecasting on the overall prediction framework. It should be noted that these tables correspond to two diferent experimental settings. In the first experiment, the proposed two-stage integrated system uses forecasted TPS values generated by the forecasting model as inputs to the CPU prediction model. Consequently, the final CPU prediction accuracy is influenced by both the forecasting uncertainty and the regression error, leading to relatively lower prediction performance. In contrast, the second experiment evaluates the CPU prediction model using the original TPS values, thereby isolating the performance of the resource prediction model from forecasting errors. As expected, this configuration achieves substantially lower MAE and RMSE values and higher $R ^ { 2 }$ values across all cloud applications. The observed performance gap between the two experiments quantifies the efect of forecasting uncertainty on downstream resource prediction. Despite relying on predicted TPS values, the integrated forecasting system still maintains satisfactory prediction accuracy and sub-second execution time, demonstrating that the proposed forecasting component introduces only a limited degradation in CPU prediction performance. These findings confirm the robustness of the two-stage architecture and indicate that the forecasting errors are suficiently controlled to support practical proactive resource provisioning and auto-scaling in dynamic cloud environments.

The proposed forecasting model’s computational eficiency is another noteworthy result. While the CPU prediction model itself runs in a matter of milliseconds, the forecasting stage takes just about 200 ms to construct a full 60-step prediction horizon. The two-stage forecasting system enables dynamic resource allocation and real-time cloud monitoring systems without causing appreciable scheduling delays due to its low computational overhead. As a result, the proposed forecasting approach can be implemented in real-world cloud settings where prediction latency is crucial.

The proposed forecasting approach has a number of benefits over several current cloud resource prediction techniques, which mostly concentrate on one-step forecasting or direct CPU estimation. First, the two-stage architecture improves interpretability and forecasting flexibility by explicitly modeling workload evolution before resource prediction. Second, the incorporation of drift analysis ofers a thorough assessment of long-horizon forecasting stability, which is frequently disregarded in earlier research. Third, the proposed approach takes into account forecasting robustness, computing eficiency, and prediction accuracy all at once, which increases its applicability to real-world cloud resource management situations.

Despite these encouraging results, several limitations remain. The experimental evaluation is based on TPS-driven workload data collected from a private cloud environment and focuses exclusively on CPU resource prediction. Other critical cloud resources, such as memory, storage, network bandwidth, and energy consumption, were not considered in this study. Furthermore, the forecasting system was evaluated using historical workload traces under relatively stable operational conditions, and the dataset size and workload diversity were limited.

In future work, we plan to extend the analysis using larger-scale datasets collected over longer time periods with more diverse and intensive workload patterns. Adaptive online learning mechanisms may be required to maintain prediction accuracy under unexpected workload anomalies, hardware failures, or concept drift during long-term deployments. However, a key limitation of the current expanding-window retraining strategy is its computational overhead. As the number of observations increases over time, the retraining cost of the forecasting model also increases due to the continuously growing training dataset. Future research will therefore focus on extending the proposed approach to multivariate resource prediction, simultaneously modeling CPU, memory, storage, and network utilization. The robustness of long-term forecasting can be further improved by incorporating transformer-based sequence models, continual learning strategies, and advanced drift adaptation techniques. Additionally, integrating the proposed forecasting framework with reinforcement learningbased auto-scaling mechanisms represents a promising direction toward fully autonomous cloud resource management systems.

## 6 Conclusion

This study presented a two-stage integrated forecasting model for proactive resource (CPU) prediction in cloud computing environments. The conventional approaches directly forecast future CPU workload from historical resource traces. At the same time, the proposed forecasting model explicitly models the relationship between customer service demand and resource consumption by first forecasting future Transactions Per Second (TPS) and subsequently estimating downstream CPU workload. The proposed technique uses the XGBoost model in both components (forecasting and resource prediction). It includes adaptive online retraining via an expanding-window strategy to continually adapt to changing workload patterns while keeping computational overhead low.

The proposed approach was extensively evaluated using real-world workload traces collected from a private cloud infrastructure comprising ten cloud applications. The two-stage integrated forecasting model achieved accurate CPU forecasting across various workloads in cloud applications, with a median Symmetric Mean Absolute Percentage Error (SMAPE) value of 5.9% for the complete forecasting model and a R<sup>2</sup> value between 0.9145 − 0.9903 for the CPU estimation module. The average forecasting latency for the integrated forecasting pipeline was less than 264 ms and less than 1 ms for CPU estimate, showing that it is suitable for real-time deployment in cloud auto-scaling systems. Furthermore, horizon-wise drift study revealed that recursive forecasting errors were comparatively stable across a 60-step prediction horizon for 3 out of 5 features, demonstrating the feasibility of the proposed cascaded forecasting architecture.

For cloud resource management, the proposed forecasting approach provides a number of useful benefits. Cloud providers can anticipate workload changes earlier than with traditional direct forecasting methods by predicting customer demand before estimating resource requirements. This enables proactive resource provisioning, reduces Service Level Agreement (SLA) violations, improves resource utilization, and lowers operating costs. Additionally, XGBoost ofers a computationally eficient alternative to deep learning-based forecasting models.

## Authorship contribution

Ashir Javeed: Conceptualization, Methodology, Validation, Writing – original draft. Anton Borg: Methodology, Writing – review & editing. H˚akan Grahn: Methodology, Validation, Supervision, Resources, Writing – review & editing. Lars Lundberg: Methodology, Supervision, Writing – review & editing. Dhyey Patel: Resources, Data Curation, Writing – review & editing. Sogand Shirinbab: Resources, Data Curation, Writing – review & editing.

## Funding

This work is funded in part by the project “Green Clouds - Load Prediction and Optimization in Private Cloud Systems”, funded by the Knowledge Foundation (grant: 20220215) in Sweden.

## Acknowledgements

The authors thank Ericsson AB, Karlskrona, Sweden for providing data and technical support for this work.

## Data availability

The data used in the study is confidential.

## Conflict of interest/Competing interests

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## References

[1] S. D. M¨uller, S. R. Holm, J. Søndergaard, Benefits of cloud computing: literature review in a maturity model perspective, Communications of the Association for Information Systems 37 (1) (2015) 42.

[2] B. Marr, Tech Trends in Practice: The 25 technologies that are driving the 4th Industrial Revolution, John Wiley & Sons, Chichester, UK, 2020.

[3] M. M. Al-Sayed, Workload time series cumulative prediction mechanism for cloud resources using neural machine translation technique, Journal of Grid Computing 20 (2) (2022) 16.

[4] E. Patel, D. S. Kushwaha, An integrated deep learning prediction approach for eficient modelling of host load patterns in cloud computing, Journal of Grid Computing 21 (1) (2023) 5.

[5] W. Chen, K. Ye, Y. Wang, G. Xu, C.-Z. Xu, How does the workload look like in production cloud? analysis and clustering of workloads on alibaba cluster trace, in: 2018 IEEE 24th International Conference on Parallel and Distributed Systems (ICPADS), IEEE, 2018, pp. 102–109.

[6] M. Xu, L. Yang, Y. Wang, C. Gao, L. Wen, G. Xu, L. Zhang, K. Ye, C. Xu, Practice of alibaba cloud on elastic resource provisioning for largescale microservices cluster, Software: Practice and Experience 54 (1) (2024) 39–57.

[7] M. A. Villegas, D. J. Pedregal, J. R. Trapero, A support vector machine for model selection in demand forecasting applications, Computers & industrial engineering 121 (2018) 1–7.

[8] A. Esposito, R. Maisto, G. Capasso, Forecasting cloud workload using arima, varima, and deep recurrent models, in: International Conference on Broadband and Wireless Computing, Communication and Applications, Springer, 2025, pp. 49–58.

[9] A. Javeed, A. Borg, H. Grahn, L. Lundberg, D. Patel, S. Shirinbab, Improving cloud eficiency: A machine learning-based stacking model for cpu utilization prediction, in: 2025 8th International Conference on Data Science and Machine Learning Applications (CDMA), IEEE, 2025, pp. 120–125.

[10] A. Javeed, A. Borg, H. Grahn, L. Lundberg, D. Patel, S. Shirinbab, CPU load prediction of heterogeneous cloud applications using lightweight models, in: 7th International Conference on Cloud Computing and Artificial Intelligence Technologies and Applications (CloudTech 2025), 2025, accepted, to appear.

[11] S. J, I. S. R, Predictopticloud: A hybrid framework for predictive optimization in hybrid workload cloud task scheduling, Simulation Modelling Practice and Theory 134 (2024) 102946. doi:https://doi.org/10.1016/ j.simpat.2024.102946.

[12] T. Wang, S. Ferlin, M. Chiesa, Predicting cpu usage for proactive autoscaling, in: Proceedings of the 1st Workshop on Machine Learning and Systems, 2021, pp. 31–38.

[13] R. N. Calheiros, E. Masoumi, R. Ranjan, R. Buyya, Workload prediction using arima model and its impact on cloud applications’ qos, IEEE transactions on cloud computing 3 (4) (2014) 449–458.

[14] V. I. Kontopoulou, A. D. Panagopoulos, I. Kakkos, G. K. Matsopoulos, A review of arima vs. machine learning approaches for time series forecasting in data driven networks, Future Internet 15 (8) (2023) 255.

[15] A. Zheng, A. Casari, Feature engineering for machine learning: principles and techniques for data scientists, ” O’Reilly Media, Inc.”, Beijing; Boston, 2018.

[16] S. Albahli, Eficient hyperparameter tuning for predicting student performance with bayesian optimization, Multimedia tools and applications 83 (17) (2024) 52711–52735.

[17] K. Valarmathi, S. Kanaga Suba Raja, Resource utilization prediction technique in cloud using knowledge based ensemble random forest with lstm model, Concurrent Engineering 29 (4) (2021) 396–404.

[18] A. Rathee, S. Dalal, K. B. Adem, Optimizing cloud resource allocation with an integrated xgboost–rfoa model, Journal of Cloud Computing (2026).

[19] F. Mart´ınez, M. P. Fr´ıas, M. D. P´erez-Godoy, A. J. Rivera, Dealing with seasonality by narrowing the training set in time series forecasting with knn, Expert systems with applications 103 (2018) 38–48.

[20] G. Sripathi, D. A. Khan, Proactive resource allocation framework with svm regression for cloud environments, SN Computer Science 7 (1) (2026) 90.

[21] S. Banerjee, S. Roy, S. Khatua, Eficient resource utilization using multistep-ahead workload prediction technique in cloud: S. banerjee et al., The Journal of Supercomputing 77 (9) (2021) 10636–10663.

[22] N. Roy, A. Dubey, A. Gokhale, Eficient autoscaling in the cloud using predictive models for workload forecasting, in: 2011 IEEE 4th international conference on cloud computing, IEEE, 2011, pp. 500–507.

[23] Y.-C. Chang, R.-S. Chang, F.-W. Chuang, A predictive method for workload forecasting in the cloud environment, in: Advanced Technologies, Embedded and Multimedia for Human-centric Computing: HumanCom and EMC 2013, Springer, Dordrecht, 2013, pp. 577–585.

[24] Z. Chen, Y. Zhu, Y. Di, S. Feng, Self-adaptive prediction of cloud resource demands using ensemble model and subtractive-fuzzy clustering based fuzzy neural network, Computational intelligence and neuroscience 2015 (1) (2015) 919805.

[25] Z. Chen, J. Hu, G. Min, A. Y. Zomaya, T. El-Ghazawi, Towards accurate prediction for high-dimensional and highly-variable cloud workloads with deep learning, IEEE Transactions on Parallel and Distributed Systems 31 (4) (2019) 923–934.

[26] M. A. Razzaq, J. A. Mahar, M. Ahmad, N. Saher, A. Mehmood, G. S. Choi, Hybrid auto-scaled service-cloud-based predictive workload modeling and analysis for smart campus system, IEEE Access 9 (2021) 42081–42089.

[27] S. Karimunnisa, A. Gopu, T. P. Rao, M. Ayyadurai, E. Kumar, A novel workload forecasting model for cloud computing using alaa-dbn algorithm, Multimedia Tools and Applications 84 (13) (2025) 11383–11407.

[28] A. Rossi, A. Visentin, D. Carraro, S. Prestwich, K. N. Brown, Forecasting workload in cloud computing: towards uncertainty-aware predictions and transfer learning, Cluster computing 28 (4) (2025) 258.

[29] H. Wang, J. Pannereselvam, L. Liu, Y. Lu, X. Zhai, H. Ali, Cloud workload analytics for real-time prediction of user request patterns, in: 2018 IEEE 20th International Conference on High Performance Computing and Communications; IEEE 16th International Conference on Smart City; IEEE 4th International Conference on Data Science and Systems (HPCC/SmartCity/DSS), IEEE, 2018, pp. 1677–1684.

[30] R. J. Hyndman, G. Athanasopoulos, Forecasting: Principles and Practice, 3rd Edition, OTexts, Melbourne, Australia, 2021. URL https://otexts.com/fpp3/

[31] C. Bergmeir, R. J. Hyndman, B. Koo, A note on the validity of crossvalidation for evaluating autoregressive time series prediction, Computational Statistics & Data Analysis 120 (2018) 70–83. doi:10.1016/j.csda.

2017.11.003. URL https://doi.org/10.1016/j.csda.2017.11.003

[32] Y.-y. A. Lau, Z. Shao, D.-Y. Yeung, Fast and slow streams for online time series forecasting without information leakage, in: The Thirteenth International Conference on Learning Representations, 2025.

[33] D. Saxena, J. Kumar, A. K. Singh, S. Schmid, Performance analysis of machine learning centered workload prediction models for cloud, IEEE Transactions on Parallel and Distributed Systems 34 (4) (2023) 1313–1330.

[34] G. A. Hofmann, K. S. Trivedi, M. Malek, A best practice guide to resource forecasting for computing systems, IEEE Transactions on Reliability 56 (4) (2007) 615–628.

[35] P. Nawrocki, M. Smendowski, Optimization of the use of cloud computing resources using exploratory data analysis and machine learning, Journal of Artificial Intelligence and Soft Computing Research 14 (4) (2024) 287–308.

[36] H. Liao, T.-y. Liu, J. Guo, B. Huang, D. Yang, J. Ding, Retrospecting available cpu resources: Smt-aware scheduling to prevent sla violations in data centers, IEEE Transactions on Parallel and Distributed Systems 36 (1) (2025) 67–83. doi:10.1109/TPDS.2024.3494879.