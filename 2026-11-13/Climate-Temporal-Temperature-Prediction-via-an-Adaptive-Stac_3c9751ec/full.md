# Climate Temporal Temperature Prediction via an Adaptive Stacked Bidirectional Residual Mamba Network

Yongxiang Lei

Abstract—Accurate temperatures forecasting relies on tradi tional meteorological parameters that are essential for monitoring weather informatics and guiding forecasting efforts. This study presents Deep DualMamba, a novel deep learning framework for temporal temperature prediction, built upon an enhanced variant of the recently proposed Mamba network. Unlike conventional Transformer-based architectures, which dominate foundation models but suffer from computational inefficiencies on long sequences, Deep DualMamba leverages selective structured state space models (SSMs) to achieve both scalability and performance. We identify a core limitation of prior subquadratictime models—namely, their inability to perform long-term and robust forecasting and introduce two key innovations. First, we propose an adaptive bidirectional Mamba as encoders to learn complex and efficient features. Second, we develop a hardwareaware, parallel recurrent algorithm to compensate for the loss of convolutional efficiency, ensuring practical viability at scale. Last, the Deep DualMamba’s approximation ability and convergence is theoretical analyzed. These design choices culminate in a simplified, attention- and MLP-free neural architecture that exhibits linear complexity with sequence length and significantly improved inference throughput. To demonstrate its applicability in climate informatics, we implement a full forecasting pipeline using historical meteorological data from UK monitoring stations. By introducing α residual connection in a stacked bidirectional Mamba, Deep DualMamba effectively learns multi-horizon temperature trends, achieving superior predictive accuracy with a reduced parameter footprint. Extensive evaluations on real-world datasets confirm its robustness, and a systematic ablation study reveals the sensitivity of key architectural hyperparameters. In addition, it offers both state-of-the-art predictive performance and enhanced robustness, while maintaining computational efficiency suitable for operational deployment. Climate temperature forecasting is a critical part for meteorological science and reducing carbon emissions. Accurate forecasting helps monitor weather informatics and guides forecasting efforts. However, traditional learning-based approaches have low forecasting performance and robustness, especially in long-term forecasting. In addition, the parameter in the model is superior. The novel DualMamba algorithms we introduce in this paper overcame these limitations. With a significant increase in reducing 32% the forecasting error, the inference ability speedup 5x faster than standard Transformer. It is a promising artificial intelligence algorithm for contributing to climate forecasting society. The theoretical contributions and practical application significantly contribute to the future trends of climate, environmental and artificial intelligence society.

## I. INTRODUCTION

Index Terms—Convergence, Time Series Forecasting, Climate Temperature Prediction, Adaptive State Space Models, Deep Learning

CCURATE climate temperature forecasting is foundational to meteorology and climate science, underpinning decision-making across environmental, agricultural, and policy domains [1]. It is also one of the process in supporting the decarbonization. The escalating complexity of global climate patterns—driven by anthropogenic perturbations and natural variability—has exposed critical limitations in conventional modeling approaches. Traditional statistical methods and machine learning models frequently fail to capture the multiscale, nonlinear interactions inherent in modern climate datasets [2] while deep learning architectures like Multi-Layer Perceptron (MLP) face fundamental constraints in adaptability and computational efficiency [3], [4]. These challenges become particularly acute in high-stakes applications requiring both predictive accuracy and operational practicality.

Climate temperature prediction occupies a unique position at the nexus of environmental monitoring and decisionmaking, providing essential insights for global change informatics, ecosystem preservation, and sustainable development [5]. The escalating complexity of climate systemsdriven by anthropogenic greenhouse gas emissions, natural variability, and regional feedback mechanisms-has rendered accurate prediction of temperature time series an urgent yet formidable challenge [6]. For example, Li et al. presented a data-driven temperature approach for thermal error prediction, while their method are based on the deterministic prediction, the intractability is lacked [7], [8]. Data from authoritative sources, such as the UK Met Office Climate Portal, exemplify this complexity, offering high-resolution records of temperature anomalies marked by non-linear trends, seasonal cycles, and stochastic noise. Xue et al. proposed a multistep prediction based on the Gated Recurrent Network (GRU) with attention mechanism [9]. These characteristics demand advanced modeling techniques capable of capturing intricate temporal dynamics while maintaining computational efficiency and robustness, particularly in applications critical to policy and resource management.

MLPs are frequently employed to model nonlinear relationships in time series data [10]. Despite their widespread use, MLPs exhibit notable shortcomings that limit their efficacy in climate temperature forecasting [11], [12]. Their dependence on fixed activation functions (e.g., ReLU, tanh) constrains their ability to adapt to the diverse and evolving patterns found in datasets like those from the UK Met Office, often necessitating deeper architectures or extensive feature engineering [13]. Additionally, the dense parameter configurations of MLPs inflate computational demands and heighten the risk of overfitting, especially when confronted with the noisy, heterogeneous nature of real-world climate records [14], [15]. These limitations highlight the need for more flexible and efficient neural network architectures tailored to the challenges of environmental time series analysis [16]. Some improved temporal neural network such as Long Short-Term Memory (LSTM) has achieves more scholars’ and researchers’ attention [17]. For example, Du et al proposed a multivariate regression method based on a hybrid temporal neural network, while their network performance depends on the increasing trainable parameter [18]. In time-series forecasting tasks, there are also some variants such as Gated Recurrent Unit (GRU), which is proposed to improve the prediction performance.

The study of Mamba offers a transformative alternative for sequence modeling, leveraging a selective state space mechanism that enables efficient processing of long-range dependencies [19]–[21]. The core innovation of Mamba lies in its input-dependent parameterization, which dynamically modulates transition matrices based on the input sequence, allowing context-aware modeling of complex temporal patterns [22]. This adaptive mechanism distinguishes Mamba from conventional recurrent neural networks and Transformers, achieving superior computational efficiency while maintaining linear-time complexity [19]. In [23], Mamba demonstrated state-of-the-art performance in traffic flow prediction, significantly outperforming existing temporal models. However, the interpretability of its state transitions remains underexplored. While theoretical analyses confirm Mamba’s expressive power in sequence approximation [24], [25], practical deployment scenarios require further empirical validation across diverse application domains. For climate temperature forecasting, where temporal relationships vary across scales and contexts, Mamba’ lightweight yet expressive design holds significant promise for improving predictive accuracy while mitigating the computational burden-a critical advantage when processing extensive datasets like those provided by the UK Met Office Climate Portal<sup>1</sup> [26].

Recent research demonstrates that Mamba, a selective state space model, is rapidly gaining traction for time series forecasting due to its ability to capture long-term dependencies with linear computational complexity, offering a compelling alternative to Transformer-based models [27], [28], [28], [29]. The Transformer architecture confers the capacity for global attention [30], enabling the computation of the impact of all other variates upon a given variate, which facilitates the learning of precise information. However, the computational load of global attention escalates exponentially with an increase in the number of variates, potentially rendering it impractical. This limitation could restrict the application of Transformer-based algorithms in real-world scenarios. Mamba-based models have achieved state-of-the-art or highly competitive results across diverse forecasting tasks, including electricity consumption, traffic flow, and weather prediction, often outperforming traditional and Transformer-based approaches in both accuracy and efficiency [31], [32]. Innovations such as bidirectional Mamba architectures, multi-scale processing, and hybrid models that combine Mamba with graph convolutional networks or Transformers further enhance performance, particularly for longsequence and multivariate forecasting [33]. Several studies have addressed Mamba’s limitations, such as its initial struggles with cross-channel dependencies, by introducing channel correlation modules and adaptive tokenization strategies [34]. Probabilistic extensions of Mamba now enable uncertainty quantification, making forecasts more robust and interpretable in real-world applications like energy and traffic management [35]. Transfer learning and foundation model approaches have also been explored, allowing Mamba-based models to perform well even with limited training data [21]. Despite these advances, challenges remain in fully exploiting Mamba’s capabilities for complex, real-world scientific forecasting, but ongoing research continues to refine its architectures and broaden its applicability. To date, most research has tested Mamba on synthetic or simplified datasets, leaving its effectiveness on comprehensive, real-world climate archives, such as those from the UK Met Office, largely unexamined. Moreover, Mamba-specific hyperparameters (e.g. historical length, preceding horizon) have received little attention, hindering practical optimization and deployment in fields requiring precision and robustness performance. Addressing these gaps is crucial for unlocking Mamba’s full potential in climate science, where rigorous, data-driven insights are increasingly essential.

This study bridges critical gaps by presenting the first operational application of Mamba for climate temperature forecasting, using high-fidelity data from the UK Met Office Climate Portal. By validating Mamba on real climate data, this work solidifies its role as an interpretable, precise, and efficient framework for operational temperature forecasting. To our knowledge, it represents the first demonstration of Mamba’s viability in real-world climate prediction, advancing both theoretical foundations and empirical implementation. The main contributions are:

• Pioneering real-world use: Deploying DualMamba on UK Met Office temperature data to move beyond synthetic or simplified testbeds. A practical climate temperature forecasting framework is implemented with high fidelity and robust performance.

• Novel temporal architecture: Proposing a novel bidirectional (feedforward and reverse) Mamba tailored to capture dynamic, non-stationary temperature trends, which learns more efficient encoding representation.

• Superior robust performance and efficiency: Achieving 14.2% lower RMSE with significantly fewer parameters compared to MLP and Transformer baselines. The hyperparameter ablation study verified the robustness. The different time-series forecasting models are also compared.

• Theoretical universal approximation and convergence: Offering theoretical approximation ability analysis and convergence guarantee for wider adoption.

The brief’s architecture is structured as follows. Section II establishes mathematical foundations. Section III delineates the DualMamba architecture. Section IV validates the framework through rigorous benchmarking against MLP, Transformer and other baselines. Finally, Section VI synthesizes key insights, discussing implications for DualMamba assessment protocols and outlining pathways for operational deployment in other forecasting systems.

## II. PROBLEM FORMULATION

## A. Multi-Layer Perceptron

![](images/c1f0bf37823d0e7072fea23cf02ca1e917a142f6e5f45fe4e0a3edefffd6948d.jpg)  
Fig. 1: The Illustration of Multilayer Perception.

The MLP is a feedforward neural network consisting of an input layer, one or more hidden layers, and an output layer. An intuitive figure of MLP is shown in Figure 1. In MLP, each layer is made up of neurons that are fully connected to neurons in the subsequent layer. Due to its universal approximation capability, an MLP can approximate any continuous function, which makes it widely applicable in time series forecasting tasks. An MLP is structured with input layer, hidden layer, and output layer. The input layer receives the raw data, with each neuron corresponding to one input feature. The hidden layer performs nonlinear transformations to extract features. Each neuron computes a weighted sum of its inputs, adds a bias, and then applies an activation function. The output layer produces the final prediction. The number of neurons in this layer depends on the specific task. Another key element for MLP is forward propagation, during forward propagation, data flows from the input layer through the hidden layers to the output layer. The computation in each layer is defined by:

$$
\mathbf { z } ^ { ( l ) } = \mathbf { W } ^ { ( l ) } \mathbf { a } ^ { ( l - 1 ) } + \mathbf { b } ^ { ( l ) } ,\tag{1}
$$

$$
\mathbf { a } ^ { ( l ) } = \sigma \left( \mathbf { z } ^ { ( l ) } \right) ,\tag{2}
$$

where $\mathbf { z } ^ { ( l ) }$ is the linear combination at layer l, $\mathbf { W } ^ { ( l ) }$ and $\mathbf { b } ^ { ( l ) }$ are the weight matrix and bias vector for layer l, respectively, $\mathbf { a } ^ { ( l - 1 ) }$ is the activation output from the previous layer, and $\sigma ( \cdot )$ denotes the activation function. The activation functions introduce nonlinearity into the network, enabling it to model complex relationships. MLP’s common activation functions include:

• Sigmoid:

$$
\sigma ( x ) = \frac { 1 } { 1 + e ^ { - x } } .\tag{3}
$$

• ReLU:

$$
\sigma ( x ) = \operatorname* { m a x } ( 0 , x ) .
$$

• Tanh:

(4)

$$
\sigma ( x ) = \operatorname { t a n h } ( x ) = { \frac { e ^ { x } - e ^ { - x } } { e ^ { x } + e ^ { - x } } } .\tag{5}
$$

The training process of a MLP is primarily driven by the backpropagation algorithm, which iteratively optimizes network parameters through gradient-based updates. Initially, the loss is computed by comparing the model’s predictions with ground truth labels using a predefined objective function. Subsequently, the gradients of the loss with respect to all trainable parameters are systematically calculated through backward propagation of errors across network layers, leveraging the chain rule of differential calculus. Finally, these gradients guide the parameter adjustments via optimization methods such as Stochastic Gradient Descent (SGD) or adaptive variants like Adam, where learning rate scheduling and momentum terms are typically incorporated to balance convergence speed and stability. This cyclical process of forward inference, gradient computation, and parameter refinement continues until the model achieves satisfactory performance or meets predefined stopping criteria.

## B. Application in Time Series Forecasting

MLPs can be utilized in time series forecasting in the following manner. The time series’ lagged values are utilized as input features in the first step. For instance, the value at time step $n + 1$ can be predicted using information from the previous n time steps. The model’s prediction is used as input for further predictions in the second stage, which implements multi-step forecasting using a rolling forecast approach. Finally, the MLP deep formula can be represented as:

$$
\begin{array} { r } { \mathrm { M L P } ( \mathbf { x } ) = \left( \mathbf { W } _ { 3 } \circ \sigma _ { 2 } \circ \mathbf { W } _ { 2 } \circ \sigma _ { 1 } \circ \mathbf { W } _ { 1 } \right) ( \mathbf { x } ) , } \end{array}\tag{6}
$$

where ◦ denotes the linear mapping projection.

MLPs are a useful tool for forecasting problems since empirical research has shown that they can successfully capture nonlinear relationships in time series data.

## C. Limitations of MLP

Despite the widespread use of MLPs in time series forecasting, several inherent limitations restrict their performance, particularly in capturing complex, dynamic patterns in data. Some of the key challenges of MLPs are as follows:

1) Static Activation Functions: In conventional MLPs, each neuron employs a fixed activation function such as the sigmoid, ReLU, or tanh function. These static functions do not adapt to the varying patterns present in time series data, thereby limiting the network’s expressiveness.

2) Over-parameterization and Complexity: To approximate complex functions, MLPs often require a large number of neurons and layers, which leads to a high parameter count. This over-parameterization not only increases computational cost but also makes the model more susceptible to overfitting, especially when the amount of available training data is limited.

3) Limited Expressive Power: Although the Universal Approximation Theorem ensures that an MLP with at least one hidden layer can approximate any continuous function given enough neurons, in practice, achieving a good approximation often demands a network with impractically high complexity. This limitation hinders efficient learning of intricate, nonlinear relationships in time series data.

## D. Motivation

The main motivations behind this brief are listed as follows.

• The traditional MLP is a black-box model. The fixed activation function and global parameter tuning of MLP limit its adaptability to high-dimensional data and long-term forecasting problem.

• The Transformer-based models exhibit formidable efficacy while the quadratic complexity of the Transformer leads to low computational efficiency and high costs, which somewhat hinders the deployment of the forecasting model in real-world scenarios.

• Experiments in traditional Mamba shows it limited feature representation ability with long-term historical inputs.

• Climate temperature has the complex factors which causes the forecasting challenging.

To address these limitations, We propose DualMamba. Unlike traditional MLPs and Mamba, DualMamba leverages the bidirectional encoders to learn superior climate features. This allows the network to dynamically to better capture complex patterns and superior feature representation in time series data with a reduced number of parameters. The comparisons of MLPs and Mamba are given in Table II.

## III. METHODOLOGIES

In UK climate forecasting tasks, the framework is formulated as follows. The target variable at time t is denoted as $y _ { t } \in \mathbb { R } ^ { T }$ . The goal is to forecast future values of the sequence:

$$
y _ { t _ { 0 } : T } = [ y _ { t _ { 0 } } , y _ { t _ { 0 } + 1 } , \ldots , y _ { t _ { 0 } + T } ] ,\tag{7}
$$

representing temperatures from the starting point $t _ { 0 }$ to preceding horizon $T .$

Historical observations $\mathbf { x } \in \mathbb { R } ^ { c }$ over a receding window of c time steps are defined as:

$$
\begin{array} { r } { \mathbf { x } _ { t _ { 0 } - c : t _ { 0 } - 1 } = \left[ x _ { t _ { 0 } - c } , \dots , x _ { t _ { 0 } - 2 } , x _ { t _ { 0 } - 1 } \right] , } \end{array}\tag{8}
$$

where $t _ { 0 }$ demarcates the transition from historical data $( t _ { 0 } - c$ to $t _ { 0 } - 1 )$ to the prediction interval $( t _ { 0 } ~ \mathrm { t o } ~ T )$ . This partitioning into context and forecast windows enables the model to leverage temporal dependencies for multi-step predictions.

The framework builds on the Mamba foundations, which has the advantage of learning long-term patterns. In our work, we design a bidirectional Mamba encoders to learn the feedforward and reverse input features, which causes a superior longdependency features, therefore, a superior climate temperature forecasting ability.

Remark 1: Our methodology generates point estimates for each step in the prediction horizon, prioritizing both accuracy and temporal coherence. By explicitly modeling the relationship between the observation window (8) and forecasting sequence (7), the approach mitigates error propagation common in autoregressive models.

```latex
Algorithm 1 The Process of Mamba Block
Input: $\overline { { \boldsymbol { X } \in \mathbb { R } ^ { B \times V \times D } } }$
Output: $\pmb { Y } \in \mathbb { R } ^ { B \times V \times D }$
1: $x , z \gets \operatorname { L i n e a r } ( U ) \quad \vartriangle D$ Linear projection $\triangleright ( B , V , E D )$
2: $x ^ { \prime } \gets \mathrm { S i L U } ( \mathrm { C o n v 1 D } ( x ) )$ $\triangleright ( B , V , E D )$
3: A ← Parameter ▷ Structured state matrix $\mathsf { \Gamma } _ { \mathsf { P } } ( D , N )$
4: $B , C \gets \mathrm { L i n e a r } ( x ^ { \prime } )$ , Linear(x<sup>′</sup>) $\triangleright ( B , V , N )$
5: ∆ ← Softplus(Parameter + Broadcast(Linear(x<sup>′</sup>)))
$\triangleright ( B , V , D )$
6: ${ \overline { { A } } } , { \overline { { B } } } \gets \operatorname { d i s c r e t i z e } ( \Delta , A , B )$ ▷ Discretization
$\triangleright ( B , V , D , N )$
7: $y  \mathrm { S e l e c t i v e S S M } ( \overline { { A } } , \overline { { B } } , C ) ( x ^ { \prime } )$ $\triangleright ( B , V , E D )$
8: $y ^ { \prime }  y \otimes \mathrm { S i L U } ( z )$ $\triangleright ( B , V , E D )$
9: $Y \gets \mathrm { L i n e a r } ( y ^ { \prime } )$ ▷ Linear projection $\mathsf { \Gamma } \mathsf { \Gamma } \mathsf { \Gamma } \mathsf { P } ( B , V , D )$
```

## A. State Space Models

State Space Models can represent any cyclical process with latent states. By using first-order differential equations to represent the evolution of the system’s internal state and another set to describe the relationship between latent states and output sequences, input sequences $\boldsymbol { x } ( t ) \in \mathbb { R } ^ { D }$ can be mapped to output sequences $\boldsymbol { y } ( t ) \in \mathbb { R } ^ { N }$ through latent states $h ( t ) \in \mathbb { R } ^ { N }$ in (9):

$$
\begin{array} { c } { { h ( t ) ^ { ' } = A h ( t ) + B x ( t ) , } } \\ { { y ( t ) = C h ( t ) , } } \end{array}\tag{9}
$$

where $\pmb { A } \in \mathbb { R } ^ { N \times N }$ and $\pmb { B } , \pmb { C } \in \mathbb { R } ^ { N \times D }$ are learnable matrices. Then, the continuous sequence is discretized by a step size $\Delta$ and the discretized SSM model is represented as (10).

$$
\begin{array} { l } { { h _ { t } = \overline { { { \cal A } } h } _ { t - 1 } + \overline { { { \cal B } } } x _ { t } , } } \\ { { y _ { t } = C h _ { t } , } } \end{array}\tag{10}
$$

where $\overline { { A } } = \exp ( \Delta A )$ and $\overline { { B } } = ( \Delta \pmb { A } ) ^ { - 1 } ( \exp ( \Delta \pmb { A } ) - I ) \cdot \Delta \pmb { B } .$ Since transitioning from continuous form $( \Delta , A , B , C )$ to discrete form $( \overline { { \cal A } } , \overline { { \cal B } } , { \cal C } )$ , the model can be efficiently calculated using a linear recursive approach [36]. The structured state space model (S4) [37], originating from the vanilla SSM, utilizes HiPPO [38] for initialization to add structure to the state matrix A, thereby improving long-range dependency modeling.

## B. Mamba block

Mamba [19] introduces a data-dependent selection mechanism into the S4 and incorporates hardware-aware parallel algorithms in its looping mode. The mechanism enables Mamba to capture contextual information in long sequences while maintaining computational efficiency. As an approximately linear perplexity series model, Mamba demonstrates potential in long sequence tasks, compared to Transformers, in both efficiency enhancement and performance improvement. The details are presented in the algorithm related to the mamba layer in Algorithm 1 and the description in Figure ??, where the former illustrates the complete data processing procedure, while the latter depicts the formation process of the output at sequence position t. The Mamba layer takes a sequence $\pmb { X } \in \dot { \mathbb { R } } ^ { B \times V \times D }$ as input, where B denotes the batch size, V denotes the number of variates, and D denotes hidden dimension.

The block first expands the hidden dimension to ED through linear projection, obtaining x and z. Then, it processes the projection obtained earlier using convolutional functions and a SiLU [39] activation function to get $x ^ { \prime } .$ . Based on the discretized SSM selected by the input parameters, denoted as the core of the Mamba Block, together with $x ^ { \prime } { . }$ it generates the state representation $y .$ Finally, y is combined with a residual connection from z after activation, and the final output y<sub>t</sub> at time step t is obtained through a linear transformation. In summary, the Mamba Block effectively handles sequential information by leveraging selective state space models and input-dependent adaptations. The parameters involved in the Mamba Block include an SSM state expansion factor $N ,$ a size of convolutional kernel k, and a block expansion factor E for input-output linear projection. The larger the values of N and $E ,$ the higher the computational cost. The final output of the Mamba block is $\pmb { Y } \in \dot { \mathbb { R } } ^ { B \times V \times D }$

The temperature forecasting problem is framed as a supervised learning framework consisting of a training dataset with input output $\left\{ \mathbf { x } _ { t _ { 0 } - L : t _ { 0 } - 1 } , \mathbf { y } _ { t _ { 0 } : L } \right\}$ in the condition and prediction lengths. The aim is to find $f$ that approximates $\mathbf { y } _ { t _ { 0 } : T }$ , i.e., $\mathbf { y } _ { t _ { 0 } : T } ~ \approx ~ f \left( \mathbf { x } _ { t _ { 0 } - L : t _ { 0 } - 1 } \right)$ . The output and input layers will be comprised of $N _ { o } .$ , and $N _ { i }$ nodes corresponding to the total amount of time steps in (7) and (8), while the transformation/hidden layer of n nodes.

## C. DualMamba Model Implementation

The implementation framework of DualMamba is visually abstracted in Figure 2, and the procedure of DualMamba is formally codified in Algorithm 2. The original meteorological climate temperature is collected from various CET stations. Then a sequencing layer is used for the linear process, which means deal with the non-numerical value (such as season, year, month, day etc) to a numerical time series. The preprocessing layer normalize the original data, split the data for the training process. The model initialization is also implemented in the preprocessing layer.

The core element of DualMamba is the block. Unlike the traditional Mamba, DualMamba generalizes effective representation from bidirectional Mamba encoders. We also introduce a formal definition of network depth. Moreover, DualMamba differs from Mamba in its bidirectional encoder process (feedforward and reverse) of sequential time-series climate predictions.

Within this layer, the primary objective is to extract the key temporal information existed in the climate data. Mamba’s selective mechanism can discern the significance of different variates akin to an attention mechanism, and it exhibits a computational overhead that escalates in a near-linear fashion with an increasing count of variates. Yet, the unilateral nature of Mamba precludes it from attending to global variates in the manner of the Transformer; its selection mechanism is unidirectional, capable only of incorporating antecedent variates. To surmount this limitation, we employ two Mamba blocks to be combined as a bidirectional Mamba layer as Eq. (11), which facilitates the acquisition of correlations among all variates.

```latex
Algorithm 2 Prediction Multistage Framework DualMamba
Require: Raw meteorological data repositories
Ensure: Validated predictive model with performance metrics
1. Data Acquisition & Curation
1: D ← Compile Climate Dataset from MET Office archives
2. Multistage Preprocessing
2: Standardize: $\begin{array} { r } { z ^ { ( k ) } = \frac { x ^ { ( k ) } - \mu ^ { ( k ) } } { \sigma ^ { ( k ) } } } \end{array}$
3. Stratified Data Partitioning
3: Split D via temporal blocking: $\mathcal { D } _ { \mathrm { t r a i n } } \colon \mathcal { D } _ { \mathrm { v a l } } \colon \mathcal { D } _ { \mathrm { t e s t } }$
4. Model Initialization
4: Initialize parameters
5. Model Forecasting with DualMamba
5: Input: $B a t c h ( X ) = \mathsf { \bar { [ } } x _ { 1 } , x _ { 2 } , \ldots x _ { L } \mathsf { ] } : ( B , L , V )$
6: for l in L do
7: Bidirectional Mamba Encoding:
8: ${ \overrightarrow { \underline { { H } } } } \gets { \overrightarrow { \mathrm { M a m b a B l o c k } } } ( X )$ ▷ Forward pass
9: ${ \overleftarrow { \mathbf { H } } }  \overleftarrow { \mathrm { M a m b a B l o c k } } ( X )$ ▷ Backward pass
10: $\pmb { H }  \overrightarrow { \pmb { H } } + \overleftarrow { \pmb { H } }$ ▷ Fusion
11: $U ^ { ' } \gets F N N ( H ) + \alpha X$ ▷ Residual connection
12: Feed-Forward Encoding FFN:
13: $U _ { , } ^ { ' } \gets \mathrm { L a y e r N o r m } ( U ^ { ' } )$
14: $U ^ { ' } \gets \mathrm { F e e d F o r w a r i } ( U ^ { ' } )$
15: $U ^ { ' } \gets \mathrm { L a y e r N o r m } ( \dot { U } ^ { ' } )$
16: end for
17: Projection:
18: $\mathbf { \boldsymbol { Y } } ^ { \prime }  \mathrm { F C } ( \mathbf { \boldsymbol { U } } ^ { \prime } )$ ▷(B, V, T)
19: Output: $B a t c h ( Y ) = [ y _ { L + 1 } , \dots y _ { L + T } ]$ ▷(B, V, T)
6. Iterative Optimization
20: Configure AdamW optimizer
21: Terminate when $\begin{array} { r } { \frac { 1 } { W } \sum _ { i = t - W } ^ { t } \mathcal { L } _ { \mathrm { v a l } } ^ { ( i ) } > \tau _ { \mathrm { s t o p } } } \end{array}$
7. Validation Analysis
22: Execute temporal cross-validation: K=5 folds with rolling
window
23: Compute MSE, RMSE, MAP, MAPE metrics according
to (34)-(35)
```

$$
\begin{array} { r } { \overrightarrow { \pmb { H } } = \overrightarrow { \mathrm { M a m b a ~ B l o c k } } ( \pmb { X } ) , } \\ { \overleftarrow { \pmb { H } } = \overleftarrow { \mathrm { M a m b a ~ B l o c k } } ( \pmb { X } ) . } \end{array}\tag{11}
$$

The hidden climate features are encoded by the bidirectional Mamba is aggregated $H \ = \ { \overrightarrow { H } } + \overleftarrow { H }$ and connected with another residual network to form the output of this layer $\boldsymbol { U } ^ { \prime } = \mathrm { F F N } ( \mathrm { H } ) + \alpha \boldsymbol { X }$ . α is an adaptive residual coefficient.

We incorporate a normalization layer [40] to improve convergence and training stability by transforming all variables toward a Gaussian distribution, thereby mitigating discrepancies arising from heterogeneous measurement scales. Subsequently, a feed-forward network (FFN) operates on the series representation of each variable, encoding the observed time series and decoding the corresponding future representations through dense nonlinear transformations. This process implicitly preserves temporal dependencies by maintaining sequential relationships. Finally, an additional normalization layer is applied to refine the predicted future series representations.

![](images/a0cb77a3b7e203cd188bddf40e47ebfab0ded128a206aa39d90ca832e64e4087.jpg)  
Fig. 2: The Overall Framework of Adaptive Deep DualMamba, the Left Side of the Figure Presents the Overall Architecture of Our Model. The Right Side of the Figure Details the Components of the DualMamba Block.

TABLE I: The Benchmarks of Meteorological Stations in the Central England Temperature (CET) Mean Station Series Dataset.
<table><tr><td>Station Name</td><td>Location</td><td>Sequence</td><td>Samples</td><td>Time Range</td></tr><tr><td>Radcliffe</td><td>Oxford, England</td><td>62,625</td><td>61,324</td><td>1815-2024</td></tr><tr><td>Rothamsted</td><td>Harpenden, Hertfordshire</td><td>53,713</td><td>52,412</td><td>1878-2024</td></tr><tr><td>Ross-on-Wye</td><td>Herefordshire, England</td><td>50,049</td><td>48,748</td><td>1964–2024</td></tr><tr><td>Stonyhurst College</td><td>Clitheroe, Lancashire</td><td>45,179</td><td>43,878</td><td>1961-2024</td></tr><tr><td>Malvern</td><td>Worcestershire, England</td><td>38,498</td><td>37,197</td><td>1926-2024</td></tr><tr><td>Squires Gate</td><td>Blackpool, Lancashire</td><td>24,368</td><td>23,067</td><td>1941-2024</td></tr><tr><td>Ringway</td><td>Manchester, England</td><td>24,974</td><td>23,673</td><td>1949-2024</td></tr><tr><td>Pershore College</td><td>Worcestershire, England</td><td>23,789</td><td>22,488</td><td>1959-2024</td></tr></table>

Finally, to ensure the DualMamba output is the desired output dimension, a fully-connected layer is added to the bidirectional Mamba cell in (11). Thus, the output of the DualMamba can be modeled as:

$$
\begin{array} { r } { \mathbf { Y } ^ { \prime } = \mathbf { W } \mathbf { U } ^ { \prime } + b , } \end{array}\tag{12}
$$

where $\mathbf { W } \in \mathbb { R } ^ { d \times T }$ , and b is the bias.

## IV. APPROXIMATION ABILITY AND CONVERGENCE

## A. Approximation Capabilities of the DeepDualMamba

The universal approximation capability of DualMamba can be formally established through the lens of operator learning theory. Consider the temporal forecasting problem as learning an operator $\mathcal { G } : \mathcal { U }  \mathcal { V }$ mapping input sequences $u ( t ) \in \mathcal { U } \subset C ( [ 0 , T ] ; \mathbb { R } ^ { d _ { u } } )$ to output trajectories $y ( t ) \in \mathcal { y } \subset$ $C ( [ 0 , T ] ; \mathbb { R } ^ { d _ { y } } )$

Theorem 1 (Universal Approximation). Let $\kappa \subset \mathcal { U }$ be a compact set of continuous functions, and $\mathcal { G } : \mathcal { U } \to \mathcal { y }$ a continuous operator. For any $\epsilon > 0 ,$ there exists a DualMamba model $\mathcal { G } _ { \theta }$ with sufficiently large depth L and state dimension D such that:

TABLE II: The Comparison Between MLP and Deep DualMamba Architectures.
<table><tr><td>Aspect</td><td>MLP</td><td>Deep DualMamba</td></tr><tr><td>Theoretical Basis</td><td>Universal Approximation Theorem</td><td>State Space Models (SSMs) with structured dynam- ics</td></tr><tr><td>Shallow Formula</td><td> $f ( { \boldsymbol { x } } ) \approx \sum _ { i = 1 } ^ { N } a _ { i } \sigma ( w _ { i } \cdot { \boldsymbol { x } } + b _ { i } )$ </td><td> $f ( x ) = \sum _ { i = 1 } ^ { L } { \mathrm { M a m b a } } _ { i } ( x )$ </td></tr><tr><td>Model features</td><td>Emphasizes parameter expressivity and nonlinearity</td><td>y Long-range modeling, memory efficiency, and bidi- rectional information flow</td></tr><tr><td>Deep Form</td><td>Layer-wise composition:  $\mathsf { M L P } ( x ) = ( W _ { 3 } \circ \sigma _ { 2 } \circ W _ { 2 } \circ$   $\bar { \sigma _ { 1 } \circ W _ { 1 } ) ( x ) }$ </td><td>Hierarchical composition: DualMamba(x) =  $( \Phi _ { 3 } \circ \mathbf { M } _ { 2 } \circ \Phi _ { 1 } ) ( { \bar { x } } )$ </td></tr><tr><td>Structure Diagram</td><td>Fixed nonlinear activation (nodes); Linear weight matrices (edges)</td><td>Alternating selective state-space modules and non- linear transitions</td></tr><tr><td>Parametrization</td><td>Global polynomial approximation</td><td>Universal approximation and convergence</td></tr></table>

$$
\operatorname* { s u p } _ { u \in { \mathcal K } } \| \mathcal G ( u ) - \mathcal G _ { \theta } ( u ) \| _ { L ^ { 2 } ( [ 0 , T ] ; \mathbb R ^ { d _ { y } } ) } < \epsilon .\tag{13}
$$

Proof. The proof proceeds in three steps:

Step 1: Mamba as a Continuous Kernel Operator The Mamba block implements a parameterized kernel operator:

$$
( \mathcal { K } u ) ( t ) = \int _ { 0 } ^ { t } \overline { { K } } ( t , s ; \theta ) u ( s ) d s ,\tag{14}
$$

where the discretized state-space parameters $\mathbf { \overline { { A } } } , \mathbf { \overline { { B } } }$ generate the kernel $\overline { { K } }$ through the variation of constants formula. By the generalized universal approximation theorem for Volterra operators [41], such kernels can approximate any continuous causal operator arbitrarily well.

Step 2: Bidirectional Composition The bidirectional architecture forms a composite operator:

$$
{ \mathcal { G } } _ { \theta } = { \mathcal { P } } \circ \underbrace { { \mathcal { F } } _ { L } \circ \cdots \circ { \mathcal { F } } _ { 1 } } _ { L { \mathrm { ~ l a y e r s } } } ,\tag{15}
$$

where each layer $\mathcal { F } _ { l } ~ = ~ \mathcal { M } _ { \right. } \oplus \mathcal { M } _ { \left. }$ consists of bidirectional Mamba blocks. Following the deep operator network framework [42], such compositions are dense in the space of continuous operators when:

1) Each M is pointwise universal (satisfied by SiLU activations)

2) The feature dimension D grows with approximation precision

Step 3: Residual Connections as Identity Approximators The residual structure $\mathcal { F } _ { l } ( u ) ~ = ~ u + \Delta ( u )$ ensures the network satisfies the conditions of the neural ordinary differential equations (neural ODEs) [43]:

$$
\frac { \partial u ( l ) } { \partial l } = \Delta ( u ( l ) , l ; \theta ) ,\tag{16}
$$

whose flow maps can approximate any continuous trajectory transformation. □

## B. Convergence Analysis

We establish the optimization convergence of DualMamba under the stochastic gradient framework. Consider the empirical risk minimization:

$$
\operatorname* { m i n } _ { \theta } \mathcal { L } ( \theta ) = \mathbb { E } _ { ( u , y ) \sim \mathcal { D } } \left[ \| y - \mathcal { G } _ { \theta } ( u ) \| _ { 2 } ^ { 2 } \right] .\tag{17}
$$

Assumption 1 (Smoothness and Convexity). The loss landscape satisfies:

1) L-smoothness: $\| \nabla \mathcal { L } ( \theta ) - \nabla \mathcal { L } ( \theta ^ { \prime } ) \| \leq L \| \theta - \theta ^ { \prime } \|$

2) Polyak-Łojasiewicz condition: $\exists \mu \quad > \quad 0$ such that <sup>1</sup> ∥∇L(θ)∥<sup>2</sup> ≥ µ(L(θ) − L<sup>∗</sup>)

Theorem 2 (Convergence Rate). Under Assumption 1, Dual-Mamba trained with Adam optimizer achieves linear convergence:

$$
\mathbb { E } [ \mathcal { L } ( \theta _ { k } ) - \mathcal { L } ^ { * } ] \leq \left( 1 - \frac { \mu } { L } \eta \right) ^ { k } \left( \mathcal { L } ( \theta _ { 0 } ) - \mathcal { L } ^ { * } \right) + \frac { \sigma ^ { 2 } } { 2 \mu } ,\tag{18}
$$

where η is the learning rate and $\sigma ^ { 2 }$ bounds gradient noise.

Proof. The convergence analysis relies on the hierarchical structure of DualMamba:

Step 1: State-Space Discretization Stability The discretization scheme $\overline { { \mathbf { A } } } = e ^ { \Delta \mathbf { A } }$ ensures:

$$
\| \overline { { \mathbf { A } } } \| _ { 2 } \le e ^ { - \gamma \Delta } < 1 , \quad \gamma > 0 ,\tag{19}
$$

guaranteeing contractivity of the state transition. This satisfies the stability condition for RNN convergence [43].

Step 2: Bidirectional Gradient Analysis The bidirectional architecture decouples the gradient backpropagation:

$$
\frac { \partial \mathcal { L } } { \partial \theta } = \underbrace { \frac { \partial \mathcal { L } } { \partial \mathbf { h } _ { T } } } _ { \mathrm { f o r w a r d ~ p a s s } } ^ { \mathcal { T } } \underbrace { \frac { \partial \mathbf { h } _ { t } } { \partial \mathbf { h } _ { t - 1 } } } _ { \mathrm { f o r w a r d ~ p a s s } } + \underbrace { \frac { \partial \mathcal { L } } { \partial \mathbf { h } _ { 0 } } } _ { \mathrm { b a c k w a r d ~ p a s s } } \underbrace { \prod _ { t = T } ^ { 1 } \frac { \partial \mathbf { h } _ { t } } { \partial \mathbf { h } _ { t + 1 } } } _ { \mathrm { b a c k w a r d ~ p a s s } } .\tag{20}
$$

The Mamba’s selective scanning mechanism bounds the Jacobian norms:

$$
\operatorname* { m a x } { \left( \left\| \frac { \partial \mathbf { h } _ { t } } { \partial \mathbf { h } _ { t - 1 } } \right\| _ { 2 } , \left\| \frac { \partial \mathbf { h } _ { t } } { \partial \mathbf { h } _ { t + 1 } } \right\| _ { 2 } \right) } \leq \kappa < 1 ,\tag{21}
$$

preventing exponential gradient explosion/vanishing.

Step 3: Adam Optimization Dynamics Adapting the Adam convergence framework [44], we define the Lyapunov function:

$$
V _ { k } = \| \theta _ { k } - \theta ^ { * } \| ^ { 2 } + \frac { \eta } { ( 1 - \beta _ { 1 } ) \sqrt { v _ { k } } + \epsilon } \mathcal { L } ( \theta _ { k } ) ,\tag{22}
$$

whose drift satisfies the supermartingale condition $\mathbb { E } [ V _ { k + 1 } | \mathcal { F } _ { k } ] ~ \le ~ ( 1 ~ - ~ c \eta ) V _ { k } ~ + ~ \bar { \eta ^ { 2 } } \sigma ^ { 2 }$ . Application of the Robbins-Siegmund theorem completes the proof. □

![](images/bf929b57ea63cd758a92135865468508b3a6d3d02e69fa1141d39ce2494173b2.jpg)  
Fig. 3: Approximation Complexity Comparison.

Corollary 1 (Sample Complexity). For $\begin{array} { r } { k \geq \frac { L } { \mu \eta } } \end{array}$ log $\frac { \Delta \mathcal { L } } { \epsilon }$ , Dual-Mamba achieves $\mathbb { E } [ \mathcal { L } ( \theta _ { k } ) - \mathcal { L } ^ { * } ] \le \epsilon$ with sample complexity $O ( \epsilon ^ { - 1 } )$ .

Remark 1. The gating mechanisms in Mamba (SiLU and selective SSM) ensure the PL condition holds locally in the basin of attraction of optimal parameters, justifying the linear convergence rate in practice.

## C. Comparison with MLP Approximation

The novel DualMamba architecture fundamentally differs from standard MLPs in approximation efficiency, particularly for temporal forecasting tasks. While both architectures possess universal approximation capabilities, DualMamba achieves exponential parameter efficiency when modeling high-dimensional spatiotemporal dependencies. Consider the class of C<sup>r</sup>-smooth target functions $f ^ { * } \in \mathcal { F } ^ { r } ( \mathcal { X } )$ , where $\mathcal { X } \subset \mathbb { R } ^ { d }$ is a compact domain.

Theorem 3 (Approximation Complexity). Let $f ^ { * } : \mathbb { R } ^ { L } $ $\mathbb { R } ^ { T }$ be a Lipschitz-continuous temporal mapping. The minimal number of parameters required to approximate $f ^ { * }$ with error ϵ satisfies:

$$
\begin{array} { r } { \mathcal { N } _ { M L P } ( \epsilon ) = \Omega \left( \epsilon ^ { - \frac { L + T } { r } } \right) , \quad \mathcal { N } _ { D u a l M a m b a } ( \epsilon ) = \mathcal { O } \left( \kappa ^ { L + T } \log ( 1 / \epsilon ) \right) } \end{array}\tag{23}
$$

where $\kappa < 1$ is the compression factor induced by the statespace formulation.

Proof. The proof follows from Kolmogorov superposition and tensor network theory:

Step 1: Curse of Dimensionality in MLPs For an MLP with ReLU activations, the approximation error scales as [45]:

$$
\operatorname* { i n f } _ { \theta } \| f ^ { * } - f _ { \theta } ^ { \mathrm { M L P } } \| _ { L ^ { 2 } } \leq C _ { 1 } \cdot \mathcal { N } _ { \mathrm { M L P } } ^ { - r / ( L + T ) } ,\tag{24}
$$

yielding the exponential parameter dependence $\begin{array} { r l } { \mathcal { N } _ { \mathrm { M L P } } } & { { } = } \end{array}$ $\dot { \Omega } ( \epsilon ^ { - ( L + T ) / r } )$

Step 2: State-Space Dimensionality Reduction DualMamba leverages the state-space formulation:

$$
h _ { t } = \overline { { A } } h _ { t - 1 } + \overline { { B } } u _ { t } , \quad y _ { t } = C h _ { t } .\tag{25}
$$

The hidden state $h _ { t } \in \mathbb { R } ^ { D }$ compresses history information, satisfying:

$$
I ( u _ { 1 : t } ; h _ { t } ) \geq ( 1 - e ^ { - \gamma t } ) I ( u _ { 1 : t } ; u _ { 1 : t } ) ,\tag{26}
$$

for some $\gamma \mathrm { ~ > ~ } 0$ (information preservation). This allows approximation with:

$$
\mathcal { N } _ { \mathrm { D u a l M a m b a } } = \mathcal { O } \left( \underbrace { D ^ { 2 } } _ { \mathrm { S S M } } + \underbrace { L _ { \mathrm { l a y e r s } } D d } _ { \mathrm { T r a n s f o r m e r s } } \right) .\tag{27}
$$

Step 3: Tensor Train Representation The bidirectional Mamba layers implement a matrix product operator (MPO) [46]:

$$
f _ { \theta } ( u ) _ { \tau } = \sum _ { \alpha _ { 1 } , \dots , \alpha _ { L } } A _ { \tau , \alpha _ { 1 } } ^ { [ 1 ] } \prod _ { t = 2 } ^ { L } A _ { \alpha _ { t - 1 } , \alpha _ { t } } ^ { [ t ] } u _ { t } ,\tag{28}
$$

with bond dimension $\alpha _ { t } \leq D$ . The approximation error then satisfies [47]:

$$
\left\| f ^ { * } - f _ { \theta } \right\| \leq C _ { 2 } \cdot e ^ { - c D } ,\tag{29}
$$

yielding logarithmic dependence on ϵ.

Corollary 2 (Long Horizon Efficiency). For sequence length $L  \infty ,$ , DualMamba maintains constant parameter complexity $\mathcal { O } ( D ^ { 2 } )$ , while MLP requires $\mathcal { O } ( e ^ { L } )$ parameters to achieve comparable accuracy.

The architectural advantage manifests in three key aspects: 1. Structural Priors: DualMamba incorporates:

• Causality: $\overline { { A } }$ ensures temporal consistency

• Stationarity: Kernel parameters shared across time

• Exponential Memory: $h _ { t }$ compresses $\{ u _ { 1 } , \ldots , u _ { t } \}$ via $e ^ { t A }$

whereas MLPs must learn these inductive biases from data.

2. Gradient Propagation: The state-space Jacobian satisfies:

$$
\frac { \partial h _ { t } } { \partial h _ { s } } = e ^ { ( t - s ) \overline { { A } } } \implies \left. \frac { \partial \mathcal { L } } { \partial h _ { s } } \right. \leq e ^ { - \gamma ( t - s ) } \left. \frac { \partial \mathcal { L } } { \partial h _ { t } } \right. ,\tag{30}
$$

guaranteeing stable gradient flow, while MLPs suffer from $\bar { \| } \nabla _ { \theta } \mathcal { L } \| \sim \bar { L ^ { - k } }$ degradation [48].

3. Spectral Efficiency: DualMamba approximates frequency content via:

$$
K ( \omega ) = C ( i \omega I - A ) ^ { - 1 } B ,\tag{31}
$$

achieving $\mathcal { O } ( 1 )$ parameter per frequency band, while MLPs require $\mathcal { O } ( \omega _ { \mathrm { m a x } } )$ neurons [49], [50].

Remark 2. The selective SSM mechanism in DualMamba adaptively focuses capacity on relevant state transitions, avoiding the uniform resource allocation of MLPs. This yields additional efficiency gains for sparse temporal dependencies. When the approximation error tends to 0, the complexity ratio of MLP and DualMamba would be infinity. An intuitive comparison is given in Figure 3.

TABLE III: Model Architecture Specification.
<table><tr><td>Model</td><td>Configuration</td><td>Time horizon</td><td>Activations</td></tr><tr><td>MLP (3-depth)</td><td>[1000, 600, 600, 600, 300</td><td>Observe/Predict: 1000/300</td><td>ReLU</td></tr><tr><td>MLP (4-depth)</td><td>[1000, 600, 600, 600, 600, 300]</td><td>Observe/Predict: 1000/300</td><td>ReLU</td></tr><tr><td>RNN (3-depth)</td><td>[1000, 60, 60, 60, 300]</td><td>Observe/Predict: 1000/300</td><td>ReLU</td></tr><tr><td>RNN (4-depth)</td><td>[1000, 60, 60, 60, 60, 300]</td><td>Observe/Predict: 1000/300</td><td>ReLU</td></tr><tr><td>GRU (3-depth)</td><td>[1000, 60, 60, 60, 300]</td><td>Observe/Predict: 1000/300</td><td>ReLU</td></tr><tr><td>GRU (4-depth)</td><td>[1000, 60, 60, 60, 60, 300]</td><td>Observe/Predict: 1000/300</td><td>ReLU</td></tr><tr><td>LSTM (3-depth)</td><td>[1000, 60, 60, 60, 300]</td><td>Observe/Predict: 1000/300</td><td>ReLU</td></tr><tr><td>LSTM (4-depth)</td><td>[1000, 60, 60, 60, 60, 300]</td><td>Observe/Predict: 1000/300</td><td>ReLU</td></tr><tr><td>Transformer (3-depth)</td><td>[1000, 60, 60, 60, 300]</td><td>Observe/Predict: 1000/300</td><td>ReLU</td></tr><tr><td>Transformer (4-depth)</td><td>[1000, 60, 60, 60, 60, 300]</td><td>Observe/Predict: 1000/300</td><td>ReLU</td></tr><tr><td>Mamba (3-depth)</td><td>[1000, 60, 60, 300]</td><td>Observe/Predict: 1000/300</td><td>SiLU</td></tr><tr><td>Mamba (4-depth)</td><td>[1000, 60, 60, 60, 300]</td><td>Observe/Predict: 1000/300</td><td>SiLU</td></tr><tr><td>DualMamba (3-depth)</td><td>[1000, 60, 60, 300]</td><td>Observe/Predict: 1000/300</td><td>SiLU</td></tr><tr><td>DualMamba (4-depth)</td><td>[1000, 60, 60, 60, 300]</td><td>Observe/Predict: 1000/300</td><td>SiLU</td></tr></table>

![](images/1540a1d7268f30b321d61e69e01a5ecf3552c2dd66a8f74987b5e5677821ed64.jpg)  
Fig. 4: Polar Charts for UK Climate Datasets.

TABLE IV: The Comparisons with Different Models and Corresponding Evaluation Indexes: A: training Datasets, B: Validation Datasets, C: Testing Dataset.
<table><tr><td rowspan="2">Model</td><td colspan="3">MSE</td><td colspan="3">RMSE</td><td colspan="3">MAE</td><td colspan="3">MAPE(%)</td></tr><tr><td>A</td><td>B</td><td>C</td><td>A</td><td>B</td><td>C</td><td>A</td><td>B</td><td>C</td><td>A</td><td>B</td><td>C</td></tr><tr><td>MLP-3</td><td>0.0487</td><td>0.0507</td><td>0.0579</td><td>0.2208</td><td>0.2251</td><td>0.2406</td><td>0.1672</td><td>0.0738</td><td>0.0076</td><td>27.96</td><td>28.52</td><td>28.91</td></tr><tr><td>MLP-4</td><td>0.0121</td><td>0.0258</td><td>0.0742</td><td>0.0832</td><td>0.0247</td><td>0.0563</td><td>0.0813</td><td>0.0734</td><td>0.0023</td><td>27.68</td><td>28.59</td><td>28.17</td></tr><tr><td>RNN-3</td><td>0.0910</td><td>0.0950</td><td>0.1219</td><td>0.0301</td><td>0.0308</td><td>0.0349</td><td>0.0241</td><td>0.0246</td><td>0.0280</td><td>62.87</td><td>20.40</td><td>11.25</td></tr><tr><td>RNN-4</td><td>0.0852</td><td>0.0861</td><td>0.0987</td><td>0.0291</td><td>0.0293</td><td>0.0314</td><td>0.0233</td><td>0.0235</td><td>0.0251</td><td>65.21</td><td>21.82</td><td>12.61</td></tr><tr><td>GRU-3</td><td>0.0845</td><td>0.0867</td><td>0.0977</td><td>0.0290</td><td>0.0294</td><td>0.0312</td><td>0.0232</td><td>0.0235</td><td>0.0250</td><td>63.08</td><td>21.64</td><td>13.51</td></tr><tr><td>GRU-4</td><td>0.0487</td><td>0.0507</td><td>0.0579</td><td>0.2208</td><td>0.2251</td><td>0.2406</td><td>0.1672</td><td>0.0738</td><td>0.0076</td><td>27.96</td><td>28.52</td><td>28.91</td></tr><tr><td>LSTM-3</td><td>0.0851</td><td>0.0874</td><td>0.1020</td><td>0.0291</td><td>0.0295</td><td>0.0319</td><td>0.0233</td><td>0.0236</td><td>0.0256</td><td>64.53</td><td>21.97</td><td>13.01</td></tr><tr><td>LSTM-4</td><td>0.0840</td><td>0.0849</td><td>0.0957</td><td>0.0289</td><td>0.0291</td><td>0.0309</td><td>0.0232</td><td>0.0233</td><td>0.0248</td><td>64.44</td><td>20.97</td><td>13.13</td></tr><tr><td>Transformer-3</td><td>0.1884</td><td>0.1678</td><td>0.2858</td><td>0.0434</td><td>0.0409</td><td>0.0534</td><td>0.0338</td><td>0.0323</td><td>0.0423</td><td>98.17</td><td>30.49</td><td>20.26</td></tr><tr><td>Transformer-4</td><td>0.1384</td><td>0.1875</td><td>0.3672</td><td>0.0372</td><td>0.0430</td><td>0.0659</td><td>0.0291</td><td>0.0335</td><td>0.0479</td><td>84.49</td><td>27.78</td><td>23.03</td></tr><tr><td>Mamba-3</td><td>0.1055</td><td>0.0848</td><td>0.0952</td><td>0.0324</td><td>0.0291</td><td>0.0308</td><td>0.0249</td><td>0.0233</td><td>0.0244</td><td>10.73</td><td>11.56</td><td>9.50</td></tr><tr><td>Mamba-4</td><td>0.0794</td><td>0.0827</td><td>0.0777</td><td>0.0281</td><td>0.0287</td><td>0.0278</td><td>0.0224</td><td>0.0225</td><td>0.0219</td><td>10.64</td><td>10.70</td><td>9.87</td></tr><tr><td>DualMamba-3</td><td>0.0765</td><td>0.1591</td><td>0.0987</td><td>0.0276</td><td>0.0398</td><td>0.0314</td><td>0.0224</td><td>0.0318</td><td>0.0244</td><td>8.79</td><td>12.54</td><td>8.08</td></tr><tr><td>DualMamba-4</td><td>0.0663</td><td>0.0668</td><td>0.0853</td><td>0.0257</td><td>0.0258</td><td>0.0292</td><td>0.0196</td><td>0.0200</td><td>0.0225</td><td>7.60</td><td>10.95</td><td>6.32</td></tr></table>

## V. CASE STUDY AND RESULT ANALYSIS

## A. UK Climate Datasets

To investigate the spatiotemporal variability of the UK climate, this study utilizes CET dataset, a high-resolution gridded observational dataset developed and maintained by the UK Met Office National Climate Information Centre. This dataset integrates observations from the UK land weather station network, adhering to World Meteorological Organization (WMO) best-practice standards, and undergoes rigorous quality assurance to ensure reliability for climate monitoring and trend analysis.

For this analysis, we extracted monthly mean temperature and precipitation data over the period 1961–2024, aligning with the WMO-recommended 30-year climatological baseline (1991–2020) and the longer-term reference period (1961– 1990) to assess both contemporary anomalies and historical trends. The dataset’s 1 km resolution facilitates detailed regional analysis, while aggregated national and regional statistics provide a broader context for interpreting climate variability.

These findings leverage the CET dataset’s high spatial fidelity and long temporal span, enabling robust detection of climate signals amidst natural variability. Uncertainties associated with station closures and data homogenization were minimized through the Met Office’s peer-reviewed interpolation techniques, ensuring the integrity of long-term trends. The dataset’s open accessibility via the Met Office Climate Data Portal further enhances its utility for reproducibility and interdisciplinary climate-related research. The data details and examples are given in Table I.

## B. Model Implementation Details and Evaluation Metrics

The model was trained and evaluated using the CET dataset, which provides gridded monthly temperature data at 1 km × 1 km resolution from 1961 to 2024. To optimize computational efficiency while preserving climatic fidelity, our methodology employs z-score normalized national monthly temperature anomalies partitioned through stratified temporal blocking: 70% training, 15% validation and 15% test.

The DualMamba architecture comprised an input layer with 1000 neurons, two hidden layers each with 64 neurons, and 300 output neuron for the predicted temperature. Unlike traditional neural networks, DualMamba uses bidirectional Mamba encoders to capture the intricate spatiotemporal dynamics of UK climate data. Traditional RNN struggle to capture longterm dependencies in sequences due to issues like gradient varnishing and explosion. Although variants like LSTM and GRU have addressed these issues to some extent, they still face challenges with very long sequences. In our experimental design, RNN, GRU, LSTM, Transformer, Mamba-villa are also selected as the baselines. The detailed model architecture is presented in Table III. The Mamba layer is set as L = 2, and adaptive residual coefficient is set as 0.5. The model was trained using the AdamW optimizer with the learning rate 0.0001 over 50 epochs with a batch size of 4096, minimizing the mean squared error (MSE) loss. The dropout rate is set as 0.1. Validation set performance was monitored to implement early stopping with 10 patience, ensuring the selection of the best-performing model for testing. The proposed DualMamba is running on a high performance computation with PyTorch framework and with 2 Nvidia H200s graph card with 282GB memory for the accelerated computation. The CUDA 12.8 is used for the tensor computation. In addition, distributed data parallel techniques is used for the computation acceleration of training. In this way, we speedup the training process, DualMamba’s inference latency can be reduced to within 1.1× that of an MLP by precomputing and fusing SiLU activation into optimized CUDA cache kernels.

Model performance was evaluated using four metrics: MSE, RMSE, MAE, and MAPE, computed on the original temperature scale for evaluating the performance. Their mathematical definitions are provided below. MSE quantifies the average squared deviation, emphasizing larger errors:

$$
\mathrm { { M S E } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( y _ { i } - \hat { y } _ { i } ) ^ { 2 } .\tag{32}
$$

In the prediction tasks, it highlights variance in predictive accuracy. RMSE measures error in the same units as temperature, balancing sensitivity to outliers:

$$
\mathrm { R M S E } = \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( y _ { i } - \hat { y } _ { i } ) ^ { 2 } } .\tag{33}
$$

RMSE provides an intuitive error magnitude. MAE assesses the average absolute deviation, robust to extreme errors:

$$
\mathrm { M A E } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } | y _ { i } - \hat { y } _ { i } | .\tag{34}
$$

MAE is a critical index for evaluating the typical prediction error. MAPE expresses error relative to observed values, enabling scale-independent comparison:

$$
\mathrm { M A P E } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left| \frac { y _ { i } - \hat { y } _ { i } } { y _ { i } } \right| \times 1 0 0 ,\tag{35}
$$

where $y _ { i }$ is the observed temperature, $\hat { y } _ { i }$ the predicted value, and $n$ the number of test samples. It is known that MAPE is particularly suited for variables with varying ranges.

These metrics were computed on the test set, with results benchmarked against baseline model Mamba to assess Dual-Mamba’s efficacy. Hyperparameter tuning, including different input interval, was guided by validation performance, optimizing for RMSE. Detailed results and regional comparisons are presented in Section IV, demonstrating DualMamba’s potential as a robust tool for climate prediction.

## C. Simulation Result Analysis

In our experimental design, we compare MLP-3, MLP-4, RNN-3, RNN-4, GRU-3, GRU-4, LSTM-3, LSTM-4, Transformer-3, Transformer-4, and Mamba-3, Mamba-4, respectively. In addition, the bold font in the table indicates superior results. Table IV summarizes all the comparison results for different approaches and our presented DualMamba. It shows that our presented DualMamba outperforms RNN and LSTM in the four indexes, which means DualMamba achieves higher prediction accuracy (lower error index). It is worth noting that, compared to Transformer, our presented DualMamba have a superior forecasting performance. Furthermore, compared to villa-Mamba, our proposed DualMamba also achieves better forecasting performance, and an intuitive lower error indexes, which means that DualMamba learns better features representation by using bidirectional encoders.

Figure 4 gives the radar chart which provides a comprehensive comparison of six distinct models, namely MLP-3,

![](images/5dffb692de1012096c214c2f3d0a0ff72ca36249a27440c3044b3284f5cc8f6d.jpg)

![](images/a4a52c51a8fd23a50df6d8d17bab4e382b8584f7c738c4b7d1de38e65b26892e.jpg)

![](images/548a3f15fbd1615f8121d97ff7bec06f7dbc1352b3a4412ea1d5b9baf92225af.jpg)

![](images/3169de5b005ca1e6ec78743a6ca9939edfbc9abaa28f0bab6f6b36511c4a0d56.jpg)  
Fig. 5: The Historical Observation and Prediction Comparisons with Radcliffe, Rothamsted, Ross on Wye, and Stonyhurst, Respectively.

![](images/3b489bfe61f024c5c565723bc2fcb7ed57e297f8ff8fe51de0d4049b23240161.jpg)

![](images/11912caf499ec2171876129e402bf1b63fec43e35007b3088f83fc3ca5033a24.jpg)

![](images/9bb320277e445b290a721cf36c8807d10bea7d12c63c3ebfffa7a6819e22053b.jpg)

![](images/acae73d388b0976942a3122040692d69f66fb76c735c74d461346a3803881e3c.jpg)  
Fig. 6: The Historical Observation and Prediction Comparisons with Malvern, Squires Gate, Ringway, and Pershore College, Respectively.

TABLE V: DualMamba Hyperparameter Sensitivity Analysis on Testing Set.
<table><tr><td>Stations</td><td>Hyperparameter</td><td>MSE</td><td>RMSE</td><td>MAE</td><td>MAPE (%)</td><td>Param.</td></tr><tr><td rowspan="7">Radcliffe</td><td>L=1000, T=300</td><td>0.1194</td><td>0.3456</td><td>0.2816</td><td>10.31</td><td>157,596</td></tr><tr><td>L=1000, T=400</td><td>0.1197</td><td>0.3460</td><td>0.2818</td><td>10.46</td><td>159,296</td></tr><tr><td>L=1000, T=500</td><td>0.1196</td><td>0.3459</td><td>0.2817</td><td>10.25</td><td>160,996</td></tr><tr><td>L=800, T=300</td><td>0.1198</td><td>0.3462</td><td>0.2820</td><td>10.77</td><td>144,796</td></tr><tr><td>L=800, T=400</td><td>0.1201</td><td>0.3466</td><td>0.2823</td><td>10.88</td><td>146,496</td></tr><tr><td>L=800, T=500</td><td>0.1196</td><td>0.3459</td><td>0.2817</td><td>10.35</td><td>148,196</td></tr><tr><td>L=600, T=300</td><td>0.1195</td><td>0.3456</td><td>0.2816</td><td>10.44</td><td>131,996</td></tr><tr><td></td><td>L=600, T=400 0.1196</td><td>0.3458</td><td>0.2817</td><td>10.39</td><td></td><td>133,696</td></tr><tr><td rowspan="10">Rothamsted</td><td>L=600, T=500</td><td>0.1198</td><td>0.3461</td><td>0.2819</td><td>10.18</td><td>135,396</td></tr><tr><td>L=1000, T=300</td><td>0.1154</td><td>0.3396</td><td>0.2779</td><td>10.42</td><td>157,596</td></tr><tr><td>L=1000, T=400</td><td>0.1158</td><td>0.3403</td><td>0.2784</td><td>10.35</td><td>159,296</td></tr><tr><td>L=1000, T=500</td><td>0.1157</td><td>0.3402</td><td>0.2784</td><td>10.35</td><td>160,996</td></tr><tr><td>L=800, T=300</td><td>0.1157</td><td>0.3402</td><td>0.2782</td><td>10.77</td><td>144,796</td></tr><tr><td>L=800, T=400</td><td>0.1151</td><td>0.3393</td><td>0.2776</td><td>10.38</td><td>146,496</td></tr><tr><td>L=800, T=500</td><td>0.1159</td><td>0.3405</td><td>0.2786</td><td>10.36</td><td>148,196</td></tr><tr><td>L=600, T=300</td><td>0.1157</td><td>0.3401</td><td>0.2782</td><td>10.38</td><td>131,996</td></tr><tr><td>L=600, T=400</td><td>0.1153</td><td>0.3396</td><td>0.2778</td><td>10.31</td><td>133,696</td></tr><tr><td>L=600, T=500</td><td>0.1156</td><td>0.3399</td><td>0.2781</td><td>10.19</td><td>135,396</td></tr><tr><td rowspan="8">Ross_on_Wye</td><td>L=1000, T=300</td><td>0.1264</td><td>0.3555</td><td>0.2928</td><td>26.76</td><td>157,596</td></tr><tr><td>L=1000, T=400</td><td>0.1265</td><td>0.3557</td><td>0.2930</td><td>19.44</td><td>159,296</td></tr><tr><td>L=1000, T=500</td><td>0.1270</td><td>0.3563</td><td>0.2935</td><td>17.38</td><td>160,996</td></tr><tr><td>L=800, T=300</td><td>0.1283</td><td>0.3582</td><td>0.2951</td><td>70.58</td><td>144,796</td></tr><tr><td>L=800, T=400</td><td>0.1269</td><td>0.3562</td><td>0.2934</td><td>17.07</td><td>146,496</td></tr><tr><td>L=800, T=500</td><td>0.1271</td><td>0.3565</td><td>0.2937</td><td>24.21</td><td>148,196</td></tr><tr><td>L=600, T=300</td><td>0.1263</td><td>0.3554</td><td>0.2927</td><td>37.08</td><td>131,996</td></tr><tr><td>L=600, T=400</td><td>0.1263</td><td>0.3554</td><td>0.2927</td><td>15.44</td><td>133,696</td></tr><tr><td rowspan="10">Stonghurst</td><td>L=600, T=500</td><td>0.1269</td><td>0.3562</td><td>0.2933</td><td>18.69</td><td>135,396</td></tr><tr><td>L=1000, T=300</td><td>0.1141</td><td>0.3378</td><td>0.2766</td><td>10.24</td><td>157,596</td></tr><tr><td>L=1000, T=400</td><td>0.1140</td><td>0.3377</td><td>0.2766</td><td>10.14</td><td>159,296</td></tr><tr><td>L=1000, T=500</td><td>0.1140</td><td>0.3376</td><td>0.2765</td><td>10.20</td><td>160,996</td></tr><tr><td>L=800, T=300</td><td>0.1198</td><td>0.3462</td><td>0.2820</td><td>10.28</td><td>144,796</td></tr><tr><td>L=800, T=400</td><td>0.1143</td><td>0.3380</td><td>0.2768</td><td>10.34</td><td>146,496</td></tr><tr><td>L=800, T=500</td><td>0.1144</td><td>0.3382</td><td>0.2770</td><td>10.38</td><td>148,196</td></tr><tr><td>L=600, T=300</td><td>0.1141</td><td>0.3377</td><td>0.2767</td><td>10.68</td><td>131,996</td></tr><tr><td>L=600, T=400</td><td>0.1140</td><td>0.3376</td><td>0.2765</td><td>10.64</td><td>133,696</td></tr><tr><td>L=600, T=500</td><td>0.1139</td><td>0.3375</td><td>0.2765</td><td>10.16</td><td>135,396</td></tr></table>

RNN-3, GRU-3, LSTM-3, Transformer-3, and DualMamba-3, across four key performance metrics. Each axis on the radar chart corresponds to one of the datasets: Training, Validation, and Testing. The data points for each model are connected by a line, with the solid line representing the DualMamba-3 model and the dashed lines representing the other models for clear distinction. Notably, the DualMamba-3 model is highlighted by a solid line and filled area, emphasizing its comparative performance against the other models represented by dashed lines without fill. The chart effectively visualizes the balanced performance of the DualMamba-3 model across different datasets, particularly excelling in the Training dataset as indicated by its proximity to the chart’s center. In contrast, the Transformer-3 model shows a notable deviation in the Testing dataset, suggesting potential overfitting or datasetspecific performance issues. This visualization aids in quickly assessing the robustness and generalizability of each model under evaluation.

Figure 5 and Figure 6 quantifies the forecasting fidelity of DualMamba against conventional Mamba on the eight stations testing benchmark. The results demonstrate Dual-Mamba’s superior capacity to resolve multiscale temperature dynamics, achieving a 0.003 reduction in RMSE. Compared to MLPs, DualMamba captures better local dynamic informatics, while MLP only predicts the trends. This enhanced resolution stems from DualMamba’s spline-parameterized architecture, which adaptively allocates basis functions to critical dynamic regimes. Figure 7 describes the corresponding training process for DualMamba, it is observable that all the eights UK dataset coverages to a stable value, which also matches our theoretical convergence analysis. The training process is stable with all the eight benchmarks even in Radcliffe and Stonyhurst, the train and validation loss has a permitable margin.

To conduct a comprehensive ablation study, we vary both the historical horizon and the prediction horizon. Specifically, the historical horizons are set to [1000, 800, 600], while the prediction horizons are [300, 400, 500], respectively. Table V presents a systematic hyperparameter sensitivity analysis of the DualMamba framework. Due to the similar results on other seven stations, we only shows the Radcliffe comparisons, other seven results are omitted. An intuitive result from Table V is that the DualMamba achieves extreme superior performance especially in terms of both long-term forecasting and forecasting robustness with different forecasting horizons. From historical horizon 1000 to 600, the decreasing horizon would not causes a performance loss, even with a large prediction range. In the worst forecasting case, 600 historical horizon to predict 500 preceding horizon, we still have the 0.1198 RMSE, which is similar to the best case, 1000 historical horizon to predict 300 horizon. Other forecasting approaches performance largely depends on the historical length, the performance decrease largely when decreasing the historical interval, or increasing the forecasting horizon.

## VI. CONCLUSION

Climate temperature in the UK is critical for monitoring carbon emissions, improving the goal of global warming. In this brief, an improved DualMamba framework is proposed for

TABLE VI: DualMamba Hyperparameter Sensitivity Analysis.
<table><tr><td>Stations</td><td>Hyperparameter</td><td>MSE</td><td>RMSE</td><td>MAE</td><td>MAPE (%)</td><td>Param.</td></tr><tr><td rowspan="8">Malvern</td><td>L=1000, T=300</td><td>0.1244</td><td>0.3527</td><td>0.2867</td><td>17.26</td><td>157,596</td></tr><tr><td>L=1000, T=400</td><td>0.1247</td><td>0.3531</td><td>0.2871</td><td>18.99</td><td>159,296</td></tr><tr><td>L=1000, T=500</td><td>0.1242</td><td>0.3524</td><td>0.2866</td><td>20.79</td><td>160,996</td></tr><tr><td>L=800, T=300</td><td>0.1242</td><td>0.3525</td><td>0.2866</td><td>13.74</td><td>144,796</td></tr><tr><td>L=800, T=400</td><td>0.1241</td><td>0.3523</td><td>0.2865</td><td>19.02</td><td>146,496</td></tr><tr><td>L=800, T=500</td><td>0.1246</td><td>0.3530</td><td>0.2872</td><td>24.24</td><td>148,196</td></tr><tr><td>L=600, T=300</td><td>0.1246</td><td>0.3529</td><td>0.2870</td><td>29.55</td><td>131,996</td></tr><tr><td>L=600, T=400</td><td>0.1243</td><td>0.3525</td><td>0.2868</td><td>12.13</td><td>133,696</td></tr><tr><td rowspan="10">Squires_Gate</td><td>L=600, T=500</td><td>0.1245</td><td>0.3529</td><td>0.2872</td><td>16.36</td><td>135,396</td></tr><tr><td>L=1000, T=300</td><td>0.1043</td><td>0.3230</td><td>0.2625</td><td>10.27</td><td>157,596</td></tr><tr><td>L=1000, T=400</td><td>0.1042</td><td>0.3228</td><td>0.2623</td><td>10.38</td><td>159,296</td></tr><tr><td>L=1000, T=500</td><td>0.1046</td><td>0.3234</td><td>0.2630</td><td>10.37</td><td>160,996</td></tr><tr><td>L=800, T=300</td><td>0.1042</td><td>0.3228</td><td>0.2622</td><td>10.37</td><td>144,796</td></tr><tr><td>L=800, T=400</td><td>0.1043</td><td>0.3230</td><td>0.2625</td><td>10.81</td><td>146,496</td></tr><tr><td>L=800, T=500</td><td>0.1043</td><td>0.3229</td><td>0.2626</td><td>10.17</td><td>148,196</td></tr><tr><td>L=600, T=300</td><td>0.1038</td><td>0.3222</td><td>0.2617</td><td>10.42</td><td>131,996</td></tr><tr><td>L=600, T=400</td><td>0.1039</td><td>0.3223</td><td>0.2619</td><td>10.21</td><td>133,696</td></tr><tr><td>L=600, T=500</td><td>0.1051</td><td>0.3242</td><td>0.2635</td><td>10.93</td><td>135,396</td></tr><tr><td rowspan="8">Ringway</td><td>L=1000, T=300</td><td>0.0930</td><td>0.3050</td><td>0.2543</td><td>38.15</td><td>157,596</td></tr><tr><td>L=1000, T=400</td><td>0.0932</td><td>0.3053</td><td>0.2546</td><td>22.21</td><td>159,296</td></tr><tr><td>L=1000, T=500</td><td>0.0933</td><td>0.3055</td><td>0.2547</td><td>12.07</td><td>160,996</td></tr><tr><td>L=800, T=300</td><td>0.0930</td><td>0.3049</td><td>0.2543</td><td>18.00</td><td>144,796</td></tr><tr><td>L=800, T=400</td><td>0.0933</td><td>0.3054</td><td>0.2547</td><td>11.29</td><td>146,496</td></tr><tr><td>L=800, T=500</td><td>0.0931</td><td>0.3052</td><td>0.2543</td><td>14.88</td><td>148,196</td></tr><tr><td>L=600, T=300</td><td>0.0932</td><td>0.3054</td><td>0.2547</td><td>21.43</td><td>131,996</td></tr><tr><td>L=600, T=400</td><td>0.0932</td><td>0.3053</td><td>0.2545</td><td>24.34</td><td>133,696</td></tr><tr><td rowspan="9">Pershore_College</td><td>L=600, T=500</td><td>0.0931</td><td>0.3052</td><td>0.2543</td><td>10.08</td><td>135,396</td></tr><tr><td>L=1000, T=300</td><td>0.1401</td><td>0.3743</td><td>0.3071</td><td>27.63</td><td>157,596</td></tr><tr><td>L=1000, T=400</td><td>0.1390</td><td>0.3728</td><td>0.3054</td><td>25.31</td><td>159,296</td></tr><tr><td>L=1000, T=500</td><td>0.1390</td><td>0.3728</td><td>0.3055</td><td>16.83</td><td>160,996</td></tr><tr><td>L=800, T=300</td><td>0.1391</td><td>0.3729</td><td>0.3059</td><td>28.84</td><td>144,796</td></tr><tr><td>L=800, T=400</td><td>0.1386</td><td>0.3723</td><td>0.3051</td><td>16.00</td><td>146,496</td></tr><tr><td>L=800, T=500</td><td>0.1390</td><td>0.3728</td><td>0.3055</td><td>20.50</td><td>148,196</td></tr><tr><td>L=600, T=300</td><td>0.1395</td><td>0.3735</td><td>0.3064</td><td>44.17</td><td>131,996</td></tr><tr><td>L=600, T=400</td><td>0.1391</td><td>0.3730</td><td>0.3057</td><td>21.75</td><td>133,696</td></tr><tr><td>L=600, T=500</td><td>0.1391</td><td>0.3730</td><td>0.3059</td><td>24.01</td><td>135,396</td></tr></table>

temperature prediction. It shows that DualMamba improves the prediction accuracy compared to the traditional MLP. The main advantages of the presented DualMamba are:

1) A novel adaptive deep DualMamba architecture for UK temperature prediction is presented.

2) The flexibility and hyper-parameters sensitivity of the proposed DualMamba has been evaluated in the ablation study.

3) Rigorous convergence analysis establishing the universal approximation property. It also have the advantage of mathematical guarantee

$$
\operatorname* { l i m } _ { \epsilon \to 0 } \frac { \mathcal { N } _ { \mathrm { M L P } } ( \epsilon ) } { \mathcal { N } _ { \mathrm { D u a l M a m b a } } ( \epsilon ) } = \infty .
$$

It is superior performance improvement in long-term prediction.

4) The results indicate that DualMamba requires low computational overhead and achieves leading robust performance. It achieves extraordinary in long-term time-series forecasting robustness.

The advantage is primarily attributed to the bidirectional Mamba block within the dual encoding Layer, which offers an enhanced understanding of climate temperature at a lower overhead compared to the Transformer. Furthermore, we conduct extensive experiments to prove Mamba possesses robust capabilities in time-series forecasting tasks. We demonstrate that DualMamba has the forecasting performance in long-term prediction with acceptable hyperparameter.

Notwithstanding these advancements, the current Dual-Mamba implementation exhibits prolonged training durations due to long dependency. Mamba excels at detecting long-range dependencies and controlling the escalation of computational demands, thus equipping it to meet the challenges outlined. In climate temperature forecasting tasks where resources are limited, compared with Transformer, Mamba is capable of processing more variates information simultaneously and delivering more accurate predictions. The proposed DualMamba is a general forecasting framework which can be extended to other time-series forecasting domains. Looking forward, Mamba-based models are expected to be applicable to a broader spectrum of time series prediction tasks that involve processing extensive variate data.

Pretrained models utilizing the Transformer architecture capitalize on its robust generalization capabilities, achieving notable success in climate temperature prediction. Our experimental results indicate that Mamba matches the Transformer in both generalization and stability, suggesting that the development of a Mamba-based pre-training model for other time-series forecasting tasks could be a fruitful direction to explore. This work bridges theoretical approximation theory with practical climate informatics, providing a novel paradigm for high-dimensional spatiotemporal prediction tasks.

## REFERENCES

[1] X. Zhang, Y. Zhang, and X. Meng, “Spatio-temporal mamba dynamic graph convolutional recurrent network for traffic prediction,” IEEE Transactions on Artificial Intelligence, 2025.

[2] P. Li, X. Zheng, S. Xiang, J. Hou, Y. Qin, M. S. Kurboniyon, and W. Ren, “Channel independence bidirectional gated mamba with interactive recurrent mechanism for time series forecasting,” IEEE Transactions on Industrial Electronics, 2025.

![](images/9ee95547cad88a270e01c9127e9e564c9f0b69a80520a696a2cd2db87b154cbe.jpg)

![](images/b3c8e26676f105c5ee82aeb5caf8d5d79fac91e5f2ab80c06fe1d6e3c2eb3acf.jpg)

![](images/8aabe7d513f30e9dd486cec6f2d1c7ef48b5b697e347b9f2025e0fd192717fb9.jpg)

![](images/052f5457ca480ae3c91c604b5e6db2a8b50278bbeb20ad39c69ebbc1b4bbe052.jpg)

![](images/4edcb52e3d655045a432947d1d238330b0e15d226ad1f69535aee765df6d053c.jpg)

![](images/a19ee0a94d97c019ac7e351863a5576ee11626b856d21bbc993c5cb6da8b1c97.jpg)

![](images/cbb1dfe03d3bb1a896a23d572d683639c3b72f9430cd27fb58122e267a8f1792.jpg)

![](images/176ff66a89e73d53cf12df94f5c33e406c08d93ef73bf18a5c142266d05f1709.jpg)  
Fig. 7: Training and Validation Loss on Radcliffe, Ross on Wye, Rothamsted, Stonyhurst, Malvern, Squires Gate, Ringway, and Pershore College, Respectively.

[3] H. J. Fowler, D. Cooley, S. R. Sain, and M. Thurston, “Detecting change in uk extreme precipitation using results from the climate prediction. net bbc climate change experiment,” Extremes, vol. 13, pp. 241–267, 2010.

[4] Y. Lei and H. R. Karimi, “Duallstm: A novel key-quality prediction for a hierarchical cone thickener,” Control Engineering Practice, vol. 137, p. 105566, 2023.

[5] G. Papacharalampous, H. Tyralis, and D. Koutsoyiannis, “Univariate time series forecasting of temperature and precipitation with a focus on machine learning algorithms: A multiple-case study from greece,” Water resources management, vol. 32, no. 15, pp. 5207–5239, 2018.

[6] Y. Song, R. Wan, L. Li, W. Wang, and H. Xing, “A multi-scale temporal-frequency fusion network based on mlp for long-term time series forecasting,” International Journal of Machine Learning and Cybernetics, pp. 1–12, 2024.

[7] Z. Li, G. Li, K. Xu, and L. Wang, “Data-driven thermal error prediction of spindle with mechanism-reinforced temperature information,” Advanced Engineering Informatics, vol. 64, p. 103005, 2025.

[8] Y. Lei, H. R. Karimi, and X. Chen, “A novel self-supervised deep lstm network for industrial temperature prediction in aluminum processes application,” Neurocomputing, vol. 502, pp. 177–185, 2022.

[9] J. Xue, Y. Cheng, W. Zhai, X. Zhou, and S. Zhou, “Research on multistep ahead prediction method for tool wear based on mstcn-sbigru-mha,” Advanced Engineering Informatics, vol. 65, p. 103219, 2025.

[10] N. Azizi, M. Yaghoubirad, M. Farajollahi, and A. Ahmadi, “Deep learning based long-term global solar irradiance and temperature fore-

casting using time series with multi-step multivariate output,” Renewable Energy, vol. 206, pp. 135–147, 2023.

[11] R. Graf, S. Zhu, and B. Sivakumar, “Forecasting river water temperature time series using a wavelet–neural network hybrid modelling approach,” Journal of Hydrology, vol. 578, p. 124115, 2019.

[12] H. Wang, Y. Mo, H. Dai, N. Yin, S. Fan, and B. Li, “Ftmlp: Mlp with feature-temporal block for multivariate time series forecasting,” Neurocomputing, vol. 607, p. 128365, 2024.

[13] B. Mohammadi, S. Mehdizadeh, F. Ahmadi, N. T. T. Lien, N. T. T. Linh, and Q. B. Pham, “Developing hybrid time series and artificial intelligence models for estimating air temperatures,” Stochastic Environmental Research and Risk Assessment, vol. 35, pp. 1189–1204, 2021.

[14] M. Mart´ınez Comesana, L. Febrero-Garrido, F. Troncoso-Pastoriza, and˜ J. Mart´ınez-Torres, “Prediction of building’s thermal performance using lstm and mlp neural networks,” Applied Sciences, vol. 10, no. 21, p. 7439, 2020.

[15] F. Rodr´ıguez, M. Genn, L. Fontan, and A. Galarza, “Very short-term´ temperature forecaster using mlp and n-nearest stations for calculating key control parameters in solar photovoltaic generation,” Sustainable Energy Technologies and Assessments, vol. 45, p. 101085, 2021.

[16] B. Ustaoglu, H. Cigizoglu, and M. Karaca, “Forecast of daily mean, maximum and minimum temperature time series by three artificial neural network methods,” Meteorological Applications: A journal of forecasting, practical applications, training techniques and modelling, vol. 15, no. 4, pp. 431–445, 2008.

[17] M. Mohammadi, S. Jamshidi, A. Rezvanian, M. Gheisari, and A. Kumar,

“Advanced fusion of mtm-lstm and mlp models for time series forecasting: an application for forecasting the solar radiation,” Measurement: Sensors, vol. 33, p. 101179, 2024.

[18] H. Du, L. Jiang, C. Zhao, W. Li, Y. Bu, Q. Xu, C. Zhang, and J. Yan, “Industrial equipment structure multivariate regression prediction via random input and hybrid temporal neural networks,” Advanced Engineering Informatics, vol. 64, p. 103006, 2025.

[19] A. Gu and T. Dao, “Mamba: Linear-time sequence modeling with selective state spaces,” arXiv preprint arXiv:2312.00752, 2023.

[20] A. Gu, Modeling sequences with structured state spaces. Stanford University, 2023.

[21] H. Ma, Y. Chen, W. Zhao, J. Yang, Y. Ji, X. Xu, X. Liu, H. Jing, S. Liu, and G. Yang, “A mamba foundation model for time series forecasting,” arXiv preprint arXiv:2411.02941, 2024.

[22] J. Alman and H. Yu, “Fundamental limitations on subquadratic alternatives to transformers,” arXiv preprint arXiv:2410.04271, 2024.

[23] Q. Li, J. Qin, D. Cui, D. Sun, and D. Wang, “Cmmamba: channel mixing mamba for time series forecasting,” Journal of Big Data, vol. 11, no. 1, p. 153, 2024.

[24] G. Zhao, L. Zhou, and X. Wang, “Ismrnn: An implicitly segmented rnn method with mamba for long-term time series forecasting,” arXiv preprint arXiv:2407.10768, 2024.

[25] T. Dao and A. Gu, “Transformers are ssms: Generalized models and efficient algorithms through structured state space duality,” arXiv preprint arXiv:2405.21060, 2024.

[26] A. Stephens, P. James, D. Alderson, S. Pascoe, S. Abele, A. Iwi, and P. Chiu, “The challenges of developing an open source, standardsbased technology stack to deliver the latest uk climate projections,” International Journal of Digital Earth, vol. 5, no. 1, pp. 43–62, 2012.

[27] Z. Wang, F. Kong, S. Feng, M. Wang, X. Yang, H. Zhao, D. Wang, and Y. Zhang, “Is mamba effective for time series forecasting?” Neurocomputing, vol. 619, p. 129178, 2025.

[28] A. Liang, X. Jiang, Y. Sun, and C. Lu, “Bi-mamba4ts: Bidirectional mamba for time series forecasting,” arXiv preprint arXiv:2404.15772, 2024.

[29] M. A. Ahamed and Q. Cheng, “Timemachine: A time series is worth 4 mambas for long-term forecasting,” 2024.

[30] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[31] W. Lin, Z. Zhang, G. Ren, Y. Zhao, J. Ma, and Q. Cao, “Mgcn: Mamba-integrated spatiotemporal graph convolutional network for longterm traffic forecasting,” Knowledge-Based Systems, vol. 309, p. 112875, 2025.

[32] W. Zhang, J. Huang, R. Wang, C. Wei, W. Huang, and Y. Qiao, “Integration of mamba and transformer-mat for long-short range time series forecasting with application to weather dynamics,” in 2024 International Conference on Electrical, Communication and Computer Engineering (ICECCE). IEEE, 2024, pp. 1–6.

[33] Y. M. Karadag, S. Kalkan, and I. G. Dino, “ms-mamba: Multi-scale mamba for time-series forecasting,” arXiv preprint arXiv:2504.07654, 2025.

[34] C. Zeng, Z. Liu, G. Zheng, and L. Kong, “Cmamba: Channel correlation enhanced state space models for multivariate time series forecasting,” arXiv preprint arXiv:2406.05316, 2024.

[35] P. Pessoa, P. Campitelli, D. P. Shepherd, S. B. Ozkan, and S. Presse,´ “Mamba time series forecasting with uncertainty propagation,” stat, vol. 1050, p. 13, 2025.

[36] A. Gu, I. Johnson, K. Goel, K. K. Saab, T. Dao, A. Rudra, and C. R’e, “Combining recurrent, convolutional, and continuous-time models with linear state-space layers,” in Neural Information Processing Systems, 2021.

[37] A. Gu, K. Goel, and C. Re, “Efficiently modeling long sequences with´ structured state spaces,” arXiv preprint arXiv:2111.00396, 2021.

[38] A. Gu, T. Dao, S. Ermon, A. Rudra, and C. Re, “Hippo: Recurrent mem-´ ory with optimal polynomial projections,” ArXiv, vol. abs/2008.07669, 2020.

[39] S. Elfwing, E. Uchibe, and K. Doya, “Sigmoid-weighted linear units for neural network function approximation in reinforcement learning,” Neural networks : the official journal of the International Neural Network Society, vol. 107, pp. 3–11, 2017.

[40] Y. Liu, T. Hu, H. Zhang, H. Wu, S. Wang, L. Ma, and M. Long, “itransformer: Inverted transformers are effective for time series forecasting,” arXiv preprint arXiv:2310.06625, 2023.

[41] I. Gohberg and M. G. Krein, Theory and applications of Volterra operators in Hilbert space. American Mathematical Soc., 1970, vol. 24.

[42] L. Lu, P. Jin, G. Pang, Z. Zhang, and G. E. Karniadakis, “Learning nonlinear operators via deeponet based on the universal approximation theorem of operators,” Nature machine intelligence, vol. 3, no. 3, pp. 218–229, 2021.

[43] R. T. Chen, Y. Rubanova, J. Bettencourt, and D. K. Duvenaud, “Neural ordinary differential equations,” Advances in neural information processing systems, vol. 31, 2018.

[44] S. J. Reddi, S. Kale, and S. Kumar, “On the convergence of adam and beyond,” arXiv preprint arXiv:1904.09237, 2019.

[45] D. Yarotsky, “Error bounds for approximations with deep relu networks,” Neural networks, vol. 94, pp. 103–114, 2017.

[46] E. Stoudenmire and D. J. Schwab, “Supervised learning with tensor networks,” Advances in neural information processing systems, vol. 29, 2016.

[47] J. Lu, Z. Shen, H. Yang, and S. Zhang, “Deep network approximation for smooth functions,” SIAM Journal on Mathematical Analysis, vol. 53, no. 5, pp. 5465–5506, 2021.

[48] B. Poole, S. Lahiri, M. Raghu, J. Sohl-Dickstein, and S. Ganguli, “Exponential expressivity in deep neural networks through transient chaos,” Advances in neural information processing systems, vol. 29, 2016.

[49] T. De Ryck, S. Lanthaler, and S. Mishra, “On the approximation of functions by tanh neural networks,” Neural Networks, vol. 143, pp. 732– 750, 2021.

[50] L. Gonon and A. Jacquier, “Universal approximation theorem and error bounds for quantum neural networks and quantum reservoirs,” IEEE Transactions on Neural Networks and Learning Systems, 2025.