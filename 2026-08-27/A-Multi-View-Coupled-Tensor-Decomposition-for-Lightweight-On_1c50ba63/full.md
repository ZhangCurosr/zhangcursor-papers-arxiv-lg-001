# A Multi-View Coupled Tensor Decomposition for Lightweight Online Adaptive Traffic Prediction

Quan Yu, Jie Ni, Yu-Hong Dai, Xiongjun Zhang

Abstract—Accurate online traffic prediction is essential for intelligent transportation systems, where forecasting must be performed continuously under imperfect sensing conditions. Missing observations and anomalous disturbances make this task challenging, particularly when prediction relies on a single traffic view. This paper proposes a Multi-View Coupled Tensor Decomposition (MVCTD) model for online traffic prediction from imperfect multi-view observations, such as speed, flow, and occupancy. The proposed model uses coupled tensor decomposition to build a structured latent forecasting space, in which shared spatial structures across traffic views and view-specific temporal dynamics are jointly modeled. A group sparse regularization is further introduced to capture correlated abnormal responses induced by real traffic anomalies and thus reduce their influence on forecasts. For streaming deployment, MVCTD performs iterative refinement only on the current latent tensor, while the remaining model variables are updated by lightweight closed-form steps based on summarized historical information, thereby avoiding repeated optimization over the full historical sequence. Experiments on real-world traffic datasets demonstrate that MVCTD achieves accurate forecasts with favorable runtime under severe missingness, confirming its suitability for online traffic prediction.

Index Terms—Online traffic prediction, multi-view, coupled tensor decomposition, missing data, anomaly modeling

## I. INTRODUCTION

O NLINE traffic prediction is a fundamental task in Intelligent Transportation Systems (ITS), where traffic states must be continuously inferred from streaming measurements and extrapolated over future horizons [1]. Accurate forecasts support proactive traffic control, congestion mitigation, and route guidance [2]. Modern sensing and communication infrastructures provide large-scale traffic measurements from urban road networks, but these data are high-dimensional, strongly spatio-temporal, and often available only in an imperfect form. As a result, effective online prediction requires not only modeling recurrent traffic dynamics, but also coping with missing observations, abnormal disturbances, and stringent computational constraints [3].

Traffic forecasting has been studied extensively in the literature. Classical time-series models, including Autoregressive (AR), Autoregressive Integrated Moving Average (ARIMA), and Seasonal ARIMA, model future values from historical temporal dependencies and have been applied to short-term traffic flow and occupancy prediction [1], [4], [5], [6], [7]. However, these models are primarily designed for univariate or low-dimensional series and therefore have limited ability to capture network-wide spatial correlations. Vector Autoregression (VAR) extends autoregressive modeling to multiple correlated variables, and low-rank factorization has further improved its scalability for spatio-temporal data. For example, Bayesian Temporal Matrix Factorization (BTMF) combines matrix factorization with a Gaussian VAR process to regularize temporal factors and improve prediction under missing data [3]. Related low-rank and dynamical system models have also been used in broader spatio-temporal forecasting applications [8], [9], [10], [11].

Deep learning methods provide another major line of work by learning nonlinear temporal and spatial dependencies from data [12]. Representative examples include deep belief networks [13], recurrent neural networks and their gated variants such as long short-term memory and gated recurrent unit models [14], [15], [16], [17], [18], [19], [20], convolutional neural networks [21], [22], [23], and graph convolutional networks that explicitly incorporate road-network topology [24], [25], [26], [27], [28], [29]. These models are powerful when sufficient clean training data and computational resources are available. Nevertheless, their performance can deteriorate when online inputs are heavily incomplete or contaminated, and the cost of updating the model for newly arriving data may be high in streaming deployment.

Despite these advances, online prediction from real-world traffic streams remains challenging for three reasons. First, traffic observations are inherently multi-view: variables such as speed, flow, and occupancy describe the same evolving network state from complementary perspectives [30], [31]. Forecasting each view separately may ignore useful crossview information, whereas naively stacking all variables may obscure view-specific temporal behavior. Second, streaming observations are often incomplete and occasionally corrupted by accidents, sensor faults, or other irregular events [32], [33]. Treating imputation, anomaly filtering, and forecasting as separate preprocessing and prediction stages can propagate estimation errors into the final forecast. Third, online deployment requires fast adaptation after new partial observations arrive. Repeatedly re-optimizing the entire historical sequence is computationally unattractive, especially when the history length grows over time.

These issues motivate a forecasting framework that couples multiple traffic views, learns from incomplete observations, separates regular dynamics from anomalous disturbances, and updates efficiently in a streaming setting. To this end, we propose a Multi-View Coupled Tensor Decomposition (MVCTD) model for lightweight online adaptive traffic prediction. MVCTD represents daily traffic states as third-order tensors indexed by road segment, time interval, and traffic view, and decomposes the regular traffic component into a shared spatial dictionary and day-specific latent temporal tensors through coupled tensor factorization. Forecasting is then performed in the latent space, where shared cross-view structures and view-dependent temporal dynamics are modeled together with temporal regularity and sparse abnormal components. For streaming deployment, MVCTD adopts a two-stage scheme that initializes the model from historical incomplete data and then performs lightweight online adaptation by refining only the current latent tensor while updating the remaining variables through closed-form steps based on summarized historical information.

The main contributions of this paper are summarized as follows:

• We propose MVCTD, a multi-view coupled tensor decomposition framework that constructs a structured latent forecasting space for online traffic prediction. It couples traffic views through a shared spatial dictionary and viewdependent latent temporal factors, enabling cross-view complementarities to be exploited while retaining viewspecific temporal dynamics.

• We develop a robust latent learning formulation that recovers predictive traffic states from incomplete and anomaly corrupted observations within a single optimization problem. The formulation combines latent space autoregression with cross-view group sparse anomaly modeling and latent temporal regularization for global periodicity and local smoothness, enabling recurrent traffic dynamics to be learned while reducing the risk of treating corrupted observations as regular traffic patterns.

• We design a lightweight two-stage optimization scheme for streaming deployment. In the offline stage, the model is initialized from historical incomplete data. In the online stage, only the current latent tensor is refined, while the remaining model variables are updated via closedform solutions. This avoids repeated optimization over the full historical sequence, thereby reducing computational overhead and enabling efficient online adaptation.

The remaining parts of this paper are organized as follows. Section II introduces the tensor algebra and notation used throughout the paper. Section III presents the proposed MVCTD model. Section IV develops the corresponding optimization algorithm. Section V reports the experimental results, and Section VI concludes the paper. The proof of the main theorem is deferred to the supplementary material.

## II. PRELIMINARIES

This section introduces the notation and tensor operations used in the proposed MVCTD framework. We focus on the algebraic tools needed for coupled tensor factorization and latent temporal regularization.

For a positive integer $n ,$ let $[ n ] = \{ 1 , 2 , \dots , n \}$ . Scalars, vectors, matrices, and tensors are denoted by lowercase letters, boldface lowercase letters, uppercase letters, and calligraphic letters, respectively. The fields of real and complex numbers are denoted by R and $\mathbb { C } .$ . For tensors $x , y \in$ $\mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ , let $\mathcal { X } _ { i j k }$ and $\mathcal { V } _ { i j k }$ denote their $( i , j , k )$ -th entries, respectively; the inner product is defined as $\langle \mathcal { X } , \mathcal { Y } \rangle =$ $\begin{array} { r } { \sum _ { i = 1 } ^ { n _ { 1 } } \sum _ { j = 1 } ^ { \dot { n } _ { 2 } } \sum _ { k = 1 } ^ { \dot { n _ { 3 } } } \chi _ { i j k } \mathcal { V } _ { i j k } } \end{array}$ , with induced Frobenius norm $\| \mathcal { X } \| = \sqrt { \langle \mathcal { X } , \mathcal { X } \rangle }$ . For a matrix $X , \| X \| ,$ denotes its nuclear norm, i.e., the sum of its singular values. The Discrete Fourier Transform (DFT) of a vector x is denoted by $\hat { \mathbf { x } } ;$ for a tensor $x , { \hat { x } }$ denotes the tensor obtained by applying the DFT along the third dimension. The indicator function $\mathbb { I } _ { \mathbb { S } } ( x )$ equals one if $x \in \mathbb { S }$ and zero otherwise. The sign function is defined as sgn $( x ) = 1 { \mathrm { ~ i f ~ } } x > 0 , \operatorname { s g n } ( x ) = 0 { \mathrm { ~ i f ~ } } x = 0$ , and $\operatorname { s g n } ( x ) = - 1$ if $x < 0$ . For $\ b { x } , \ b { y } \in \mathbb { R } ^ { n }$ , the circular convolution of x and y is defined as $\mathbf { \boldsymbol { x } } \bullet \mathbf { \boldsymbol { y } } = \mathcal { C } ( \mathbf { \boldsymbol { x } } ) \mathbf { \boldsymbol { y } }$ , where $\mathcal { C } ( \pmb { x } )$ is the circulant matrix given by

$$
\mathcal { C } ( \pmb { x } ) = \left[ \begin{array} { c c c c } { x _ { 1 } } & { x _ { n } } & { \cdots } & { x _ { 2 } } \\ { x _ { 2 } } & { x _ { 1 } } & { \cdots } & { x _ { 3 } } \\ { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { x _ { n } } & { x _ { n - 1 } } & { \cdots } & { x _ { 1 } } \end{array} \right] .
$$

Definition 2.1 (t-product $I 3 4 l ) .$ Let $\mathcal { X } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ and $\mathcal { V } \in \mathbb { R } ^ { n _ { 2 } \times n _ { 4 } \times n _ { 3 } }$ . Their t-product $\mathcal { Z } = \mathcal { X } \ast \mathcal { Y } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 4 } \times n _ { 3 } }$ is defined by $\begin{array} { r } { \mathcal { Z } ( i , j , : ) = \sum _ { k = 1 } ^ { n _ { 2 } } \mathcal { X } ( i , k , : ) \bullet \mathcal { V } ( k , j , : ) } \end{array}$

Definition 2.2 (conjugate transpose $I 3 4 J )$ : The conjugate transpose of $\mathcal { X } \in \mathbb { C } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ , denoted by $\mathcal { X } ^ { \top } \in \mathbb { C } ^ { n _ { 2 } \times n _ { 1 } \times n _ { 3 } }$ is obtained by conjugate transposing each frontal slice and reversing the order of the frontal slices from the second to the last.

Definition 2.3 (f-diagonal tensor $I ^ { 3 4 } I ) \colon \mathrm { ~ \bf ~ A ~ }$ tensor is fdiagonal if each of its frontal slices is a diagonal matrix.

Definition 2.4 (identity tensor $I 3 4 l ) .$ : The identity tensor $\mathcal { T } \in$ $\mathbb { R } ^ { n \times n \times n _ { 3 } }$ has the identity matrix as its first frontal slice and zero matrices as all remaining frontal slices.

Definition 2.5 (orthogonal tensor $I ^ { 3 4 } I ) \colon \textbf { A }$ tensor $\mathcal { Q } ^ { \mathrm { ~ \tiny ~ \in ~ } }$ R<sup>n×n×n3</sup> is orthogonal if $\mathcal { Q } ^ { \top } \ast \mathcal { Q } = \mathcal { Q } \ast \mathcal { Q } ^ { \top } = \mathcal { I } .$

The t-product admits an analogue of the matrix singular value decomposition.

Theorem 2. $1 ( t { \cdot } S V D ~ / 3 4 J ) ;$ : For any tensor $\mathcal { X } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ its tensor Singular Value Decomposition $( \mathrm { t } { - } \mathrm { S V D } )$ is given by $\mathcal { X } = \mathcal { U } * \mathcal { S } * \bar { \mathcal { V } } ^ { \top }$ , where $\mathcal { U } \in \mathbb { R } ^ { n _ { 1 } \times n _ { 1 } \times n _ { 3 } }$ and $\boldsymbol { \mathcal { V } } \in \mathbb { R } ^ { n _ { 2 } \times n _ { 2 } \times n _ { 3 } }$ are orthogonal tensors, and $S \in \mathbb { R } ^ { n _ { 1 } \times n _ { 2 } \times n _ { 3 } }$ is f-diagonal. If X has tensor tubal rank r as defined in Definition 2.6, its skinny t-SVD is $\mathcal { X } = \mathcal { U } _ { r } * S _ { r } * \mathcal { V } _ { r } ^ { \top }$ , where $\mathcal { U } _ { r } , \mathcal { S } _ { r }$ , and $\mathcal { V } _ { r }$ retain the first r singular tubes.

Definition 2.6 (tensor tubal rank $I 3 5 { \cal I } ) .$ Given the t-SVD $\mathcal { X } = \mathcal { U } \ast \mathcal { S } \ast \mathcal { V } ^ { \intercal }$ , the tensor tubal rank is the number of nonzero singular tubes in $s ,$ i.e., ran $\mathfrak { i } _ { t } ( \mathcal { X } ) = \# \{ i : \mathcal { S } ( i , i , : ) \neq \mathbf { 0 } \}$

Lemma 2.1: [36, Theorem 3.1] The eigenvalues of a circulant matrix $\mathcal { C } ( z )$ generated by any vector $z \in \mathbb { R } ^ { n }$ are given by the components of $F z$ , where $F$ is the DFT matrix.

Lemma 2.2: [37] For any vectors x, $\textbf { \textit { y } } \in \mathbb { R } ^ { n }$ , one has $\widehat { { \pmb x } \bullet { \pmb y } } = \hat { \pmb x } \circ \hat { \pmb y }$ and $\| \pmb { x } \| ^ { 2 } \overset { \cdot } { = } \| \hat { \pmb { x } } \| ^ { 2 } / n$ , where ◦ denotes the Hadamard product.

## III. MVCTD FOR LIGHTWEIGHT ONLINE ADAPTIVETRAFFIC FORECASTING

This section develops the proposed Multi-View Coupled Tensor Decomposition (MVCTD) framework for online traffic prediction from imperfect multi-view streams. We first formalize the rolling forecasting task and then present the two stages of MVCTD. Stage I initializes a structured latent predictive representation from historical incomplete observations, while Stage II adapts this representation online as new partial traffic tensors arrive.

## A. Problem Formulation

Let $\mathscr { X } _ { d } \in \mathbb { R } ^ { N \times T \times V }$ denote the complete traffic state tensor on day d, where N, T, and V are the numbers of road segments, daily time intervals, and traffic variables, respectively. The complete tensor is not fully observed in practice. Instead, the predictor receives $\mathcal { M } _ { d } ~ = ~ \mathcal { P } _ { \Omega _ { d } } ( \mathcal { X } _ { d } )$ , where $\Omega _ { d }$ is the observed index set and $\mathcal { P } _ { \Omega _ { d } }$ denotes the projection that keeps entries in $\Omega _ { d }$ and sets the remaining entries to zero. Given the partial history $\{ \mathcal { M } _ { d } \} _ { d = 1 } ^ { D }$ , the goal is to predict the next complete traffic tensor $\chi _ { D + 1 }$ and then repeat this procedure as new partial observations arrive, as illustrated in Fig. 1.

![](images/fd60f16ceb7ed78a1480bef84ee2524e1c410d902a05bb9a9e0d4638b3512722.jpg)  
Fig. 1: Schematic illustration of rolling online traffic forecasting with incomplete observations.

This setting requires the model to use complementary information across traffic variables, recover predictive states from incomplete measurements, reduce the influence of nonrecurrent anomalies, and update efficiently in a rolling deployment. Accordingly, MVCTD does not treat completion, anomaly suppression, and prediction as separate preprocessing and forecasting steps. Instead, it couples them within a single latent tensor formulation so that multi-view information can directly support forecasting under imperfect observations.

## B. Stage I: Offline Initialization

Given the historical partial tensors $\{ \mathcal { M } _ { d } \} _ { d = 1 } ^ { D }$ defined in Section III-A, the offline stage learns the initial components required for online prediction. It estimates regular latent traffic patterns, separates sparse abnormal disturbances, and initializes the temporal prediction model used in the subsequent streaming stage.

Unified Optimization Framework. Let $f _ { \mathcal { X } }$ map a historical window of length L to the current traffic state, i.e., $\hat { \mathcal { X } } _ { d } ~ =$ $f _ { \mathcal { X } } ( \mathcal { X } _ { d - L } , \ldots , \mathcal { X } _ { d - 1 } ; \Theta _ { \mathcal { X } } )$ . Directly fitting this predictor to the raw stream is unreliable because the available observations are incomplete and may contain anomalous events. We therefore introduce a regular component $\mathcal { O } _ { d }$ and an anomaly component $\mathcal { E } _ { d } ,$ , and learn the predictive state jointly with completion and anomaly separation:

$$
\begin{array} { r l } { \operatorname* { m i n } } & { \underset { \underset { \ r { d = L + 1 } } { \underbrace { \sum _ { d = L + 1 } ^ { D } } } } { \underbrace { \sum _ { d = L + 1 } ^ { D } \| \mathcal { X } _ { d } - f _ { \mathcal { X } } ( \mathcal { X } _ { d - L } , \ldots , \mathcal { X } _ { d - 1 } ; \Theta _ { \mathcal { X } } ) \| ^ { 2 } } } } \\ & { + \sum _ { d = 1 } ^ { D } \big [ \mathcal { R } _ { 1 } ( \mathcal { O } _ { d } ) + \mathcal { R } _ { 2 } ( \mathcal { E } _ { d } ) \big ] } \\ { \mathrm { s . t . } } & { \mathcal { X } _ { d } = \mathcal { O } _ { d } + \mathcal { E } _ { d } , \mathcal { P } _ { \Omega _ { d } } ( \mathcal { X } _ { d } ) = \mathcal { M } _ { d } , d \in [ D ] . } \end{array}\tag{1}
$$

Here, $\mathcal { R } _ { 1 }$ imposes structural regularization on the regular component $\mathcal { O } _ { d } .$ , while $\mathcal { R } _ { 2 }$ penalizes the anomaly component $\mathcal { E } _ { d } .$ . The parameter $\gamma > 0$ balances the forecasting fidelity term and the regularization terms.

Coupled Tensor Factorization. The regular component $\mathcal { O } _ { d }$ should preserve the shared road-network structure across days while allowing different traffic variables to exhibit viewdependent temporal behavior. Since all daily traffic tensors are generated over the same road network, it is natural to use a common spatial dictionary to describe recurring spatial patterns. Meanwhile, day-to-day variations and view-specific temporal evolution are represented by day-specific latent temporal factors. The t-product further couples the factor tensors along the view dimension, enabling interactions among traffic variables to be encoded within the decomposition rather than treating each view independently. We therefore factorize the regular component as

$$
\mathcal { O } _ { d } = \mathcal { A } \ast \mathcal { Z } _ { d } , \quad \forall d \in [ D ] ,
$$

where $\mathcal { A } \ \in \ \mathbb { R } ^ { N \times R \times V }$ is a shared spatial dictionary and $\mathcal { Z } _ { d } \in \mathbb { R } ^ { R \times T \times V }$ stores the day-specific latent temporal factors. This representation moves forecasting from the raw observation space to a structured latent space, where cross-view complementarities are encoded by the coupled dictionary and recurrent temporal patterns are represented more compactly.

The low-rank nature of $\mathcal { O } _ { d }$ can be promoted through sparsity of the latent coefficients. Specifically, Yu et al. [38, Theorem 3.2] established the equivalence between tensor tubal rank and group sparsity of the core coefficients under an orthogonal basis:

$$
\begin{array} { r l r } & { } & { \mathrm { \ r a n k } _ { t } ( \mathcal { O } _ { d } ) = \operatorname* { m i n } \big \{ \| \mathcal { Z } _ { d } \| _ { \mathrm { H S } } ^ { 0 } : \mathcal { O } _ { d } = \mathcal { A } \ast \mathcal { Z } _ { d } , \mathcal { A } ^ { \top } \ast \mathcal { A } = \mathbb { Z } , } \\ & { } & { \mathcal { A } \in \mathbb { R } ^ { N \times R \times V } , \mathcal { Z } _ { d } \in \mathbb { R } ^ { R \times T \times V } \big \} , } \end{array}
$$

where $\begin{array} { r } { \| \mathcal { Z } _ { d } \| _ { \mathrm { H S } } ^ { 0 } = \sum _ { r = 1 } ^ { R } \| \mathcal { Z } _ { d } ( r , \cdot , \cdot ) \| ^ { 0 } . } \end{array}$

Combining these physically motivated structures with theoretically grounded constraints, and using the $\ell _ { p }$ -norm relaxation for the $\ell _ { 0 } { \cdot } \mathrm { n o r m }$ , we reformulate problem (1) as follows:

$$
\begin{array} { r l } { \operatorname* { m i n } } & { \gamma \sum _ { d = L + 1 } ^ { D } \| \mathcal { Z } _ { d } - f _ { \mathcal { Z } } ( \mathcal { Z } _ { d - L } , \ldots , \mathcal { Z } _ { d - 1 } ; \Theta _ { \mathcal { Z } } ) \| ^ { 2 } } \\ & { + \sum _ { d = 1 } ^ { D } \left[ \lambda _ { 1 } \| \mathcal { Z } _ { d } \| _ { \mathrm { H S } } ^ { p } + \lambda _ { 2 } \| \mathcal { E } _ { d } \| _ { \mathrm { T F } } ^ { p } \right] } \\ { \mathrm { s . t . } } & { \mathcal { X } _ { d } = \mathcal { A } \ast \mathcal { Z } _ { d } + \mathcal { E } _ { d } , \mathcal { P } _ { \Omega _ { d } } ( \mathcal { X } _ { d } ) = \mathcal { M } _ { d } , d \in [ D ] , } \\ & { \mathcal { A } ^ { \top } \ast \mathcal { A } = \mathcal { Z } . } \end{array}
$$

Here, $\lambda _ { 1 } , \lambda _ { 2 } { \it \Delta \phi } > 0$ are regularization parameters. The first term measures prediction error in the latent space rather than in the high-dimensional traffic tensor space. Since $R \ll N$ and $\mathcal { E } _ { d }$ absorbs sparse corruptions, latent-space forecasting reduces the dimension of the prediction model and makes the learned dynamics less sensitive to abnormal observations. The penalty $\begin{array} { r } { \| \dot { \boldsymbol { \mathcal { Z } } } _ { d } \| _ { \mathrm { H S } } ^ { p } = \sum _ { r = 1 } ^ { R } \| \mathcal { Z } _ { d } ( r , : , : ) \| ^ { p } } \end{array}$ promotes a compact regular component with low tubal rank, while $\| \mathcal { E } _ { d } \| _ { \mathrm { T F } } ^ { p } =$ $\begin{array} { r } { \sum _ { n = 1 } ^ { N } { \sum _ { t = 1 } ^ { T } \dot { \| \mathcal { E } _ { d } ( n , t , : ) \| ^ { p } } } } \end{array}$ induces group sparsity across the view dimension [39], allowing correlated abnormal responses to be separated from regular traffic dynamics.

Subset Autoregression. Standard AR models typically rely on a continuous window of lags $( \mathbf { e . g . , } 1 , \ldots , p )$ , which often leads to over-parameterization and the curse of dimensionality, particularly when capturing long-term dependencies. To address this, we characterize the temporal dynamics of $\mathcal { Z } _ { d }$ using an SAR strategy. This approach is parsimonious: by explicitly selecting a sparse set of physically meaningful time lags ${ \mathcal { L } } : = \{ h _ { l } \} _ { l = 1 } ^ { \bar { L } ^ { - } } \left( \mathrm { e . g . } \right)$ daily or weekly cycles) rather than consecutive ones, the model captures multi-scale periodicities with fewer parameters. Here, $\bar { \boldsymbol { L } } = | \boldsymbol { \mathcal { L } } |$ denotes the number of selected lags, and $L \ : = \ \operatorname* { m a x } { \mathcal { L } }$ denotes the minimum history length required by the forecasting loss. Consequently, the evolution of the latent tensor is formulated as

$$
\mathcal { Z } _ { d } \approx \sum _ { l = 1 } ^ { \bar { L } } w ^ { l } \mathcal { Z } _ { d - h _ { l } } ,
$$

where $w ^ { l } \in \mathbb { R }$ is the autoregressive coefficient associated with lag $h _ { l }$

Laplacian Convolutional Priors. Traffic states usually contain both global periodicity and local temporal continuity. Inspired by [40], these two patterns can be characterized by a circulant matrix nuclear norm and a temporal convolution penalty. By imposing on the traffic state itself, the corresponding priors take the form

$$
\lambda _ { 3 } \| \mathcal { C } \diamond \mathcal { X } _ { d } \| _ { * } + \lambda _ { 4 } \| \ell \boxtimes \mathcal { X } _ { d } \| _ { \mathrm { R F } } ,
$$

where $\lambda _ { 3 } , \lambda _ { 4 } > 0$ are regularization parameters, and

$$
\ell \triangleq ( 2 \tau , \underbrace { - 1 , \ldots , - 1 } _ { \tau } , 0 , \ldots , 0 , \underbrace { - 1 , \ldots , - 1 } _ { \tau } ) ^ { \intercal } \in \mathbb { R } ^ { T }
$$

is the temporal convolution kernel. The term $\| \mathcal { C } \diamond \mathcal { X } _ { d } \| _ { * } =$ $\begin{array} { r } { \sum _ { n = 1 } ^ { N } \sum _ { v = 1 } ^ { \bar { V } } \| \mathcal { C } ( \mathcal { X } _ { d } ( n , : , v ) ) \| , } \end{array}$ <sub>∗</sub> promotes global low-rank periodicity, and $\begin{array} { r } { \Vert \ell \boxtimes \mathcal { X } _ { d } \Vert _ { \mathrm { R F } } = \sum _ { n = 1 } ^ { N } \sum _ { v = 1 } ^ { V } \Vert \ell \bullet \mathcal { X } _ { d } ( n , : , v ) \Vert } \end{array}$ promotes local smoothness.

In MVCTD, however, these temporal priors are imposed on $\mathcal { Z } _ { d }$ rather than directly on $\mathcal { X } _ { d }$ . Since prediction is performed in the latent space, regularizing $\mathcal { Z } _ { d }$ directly encourages extrapolatable recurrent patterns. Moreover, $\mathcal { X } _ { d }$ may include sparse anomalies represented by $\mathcal { E } _ { d } ;$ smoothing the raw state could therefore mix abnormal disturbances into regular dynamics.

The following bound supports this latent-space regularization strategy by showing that temporal regularity of the recovered traffic state can be controlled through the latent coefficients.

Theorem 3.1: Let $\mathcal { A } \in \mathbb { R } ^ { N \times R \times V }$ and $\mathcal { Z } \in \mathbb { R } ^ { R \times T \times V }$ , and let $\ell \in \mathbb { R } ^ { T }$ be a convolution kernel. If $\mathcal { X } = \mathcal { A } \ast \mathcal { Z }$ , then the following inequality holds.

(1) $\| \mathcal { C } \circ \mathcal { X } \| _ { * } \leq \zeta \| \mathcal { C } \circ \mathcal { Z } \| _ { * } ;$

(2) $\| \ell \boxtimes \mathcal { X } \| _ { \mathrm { R F } } \le \zeta \| \ell \boxtimes \mathcal { Z } \| _ { \mathrm { R F } } .$

Here $\begin{array} { r } { \zeta = \operatorname* { m a x } _ { r } \sum _ { n = 1 } ^ { N } \sum _ { v = 1 } ^ { V } | A ( n , r , v ) | . } \end{array}$

Motivated by this bound and the robustness considerations above, we reformulate the Laplacian convolutional priors in the latent space as

$$
\lambda _ { 3 } \| \mathcal { C } \diamond \mathcal { Z } _ { d } \| _ { * } + \lambda _ { 4 } \| \ell \boxtimes \mathcal { Z } _ { d } \| _ { \mathrm { R F } } .
$$

Integrated Model Formulation. Combining the components above yields the concrete objective for the offline initialization stage:

$$
\begin{array} { r l } { \underset { \Theta } { \mathrm { m i n } } } & { ~ \underset { \mathcal { C } = L + 1 } { \mathrm { \sum } } ~ \overset { D } { \underset { l = l - 1 } { \mathrm { \sum } } } w ^ { l } \mathcal { Z } _ { d - h _ { l } } \big \| ^ { 2 } + \underset { d = 1 } { \overset { D } { \sum } } \big [ \underset { \mathrm { L o w - r a n k } } { \underbrace { \lambda _ { 1 } \| \mathcal { Z } _ { d } \| _ { \mathrm { H S } } ^ { p } } } } \\ & { + \underset { \mathrm { A n o m a l y } } { \underbrace { \lambda _ { 2 } \| \mathcal { E } _ { d } \| _ { \mathrm { T F } } ^ { p } } } + \underset { \mathrm { G l o b a l p e r i o d i c i t y } } { \underbrace { \lambda _ { 3 } \| \mathcal { C } \diamond \mathcal { Z } _ { d } \| _ { \mathrm { s } } } } + \underset { \mathrm { L o c a l s m o o t h n e s s } } { \underbrace { \lambda _ { 4 } \| \mathcal { E } \boxtimes \mathcal { Z } _ { d } \| _ { \mathrm { H F } } } } \big ] } \\ { \mathrm { s . t . } } & { ~ \mathcal { X } _ { d } = \mathcal { A } \ast \mathcal { Z } _ { d } + \mathcal { E } _ { d } , ~ \mathcal { P } _ { \Omega _ { d } } ( \mathcal { X } _ { d } ) = \mathcal { M } _ { d } , ~ d \in [ D ] , } \\ & { ~ \mathcal { A } ^ { \top } \ast \mathcal { A } = \mathcal { L } , } \end{array}
$$

where $\boldsymbol { \Theta } = \{ \{ \mathcal { X } _ { d } , \mathcal { Z } _ { d } , \mathcal { E } _ { d } \} _ { d = 1 } ^ { D } , \{ w ^ { l } \} _ { l = 1 } ^ { \bar { L } } , A \}$ denotes the set of all learnable parameters.

(2)

## C. Stage II: Lightweight Online Adaptive Prediction

After offline initialization, MVCTD operates in a rolling manner. At each new day, the current model first produces a prior forecast and then corrects the latent state once the partial observation becomes available. Only the current latent tensor is optimized iteratively, while the global model components are updated separately.

Forecasting. Before observing day $D + 1$ , MVCTD predicts the latent tensor $\mathcal { Z } _ { D + 1 } ^ { + }$ using the learned SAR relation. With the current spatial basis $A _ { D }$ and autoregressive coefficients $w _ { D } ^ { l } .$ , the prior traffic forecast is

$$
\mathcal { X } _ { D + 1 } ^ { + } = \mathcal { A } _ { D } \ast \mathcal { Z } _ { D + 1 } ^ { + } , \mathrm { ~ w h e r e ~ } \mathcal { Z } _ { D + 1 } ^ { + } = \sum _ { l = 1 } ^ { \bar { L } } w _ { D } ^ { l } \mathcal { Z } _ { D + 1 - h _ { l } } ^ { \star } .\tag{3}
$$

Online Inference and Model Adaptation. When the partial observation $\mathcal { M } _ { D + 1 } = \mathcal { P } _ { \Omega _ { D + 1 } } ( \mathcal { X } _ { D + 1 } )$ is received, MVCTD refines the current latent state and updates the model parameters through three decoupled subproblems.

• Latent state inference: The current latent tensor $\mathcal { Z } _ { D + 1 } ^ { \star }$ is estimated by solving

$$
\begin{array} { r l } { \underset { \mathcal { Z } , \varepsilon } { \operatorname* { m i n } } } & { \lambda _ { 1 } \| \mathcal { Z } \| _ { \mathrm { H S } } ^ { p } + \lambda _ { 2 } \| \mathcal { E } \| _ { \mathrm { T F } } ^ { p } + \lambda _ { 3 } \| \mathcal { C } \diamond \mathcal { Z } \| _ { * } } \\ & { + \lambda _ { 4 } \| \ell \boxtimes \mathcal { Z } \| _ { R F } + \gamma \| \mathcal { Z } - \mathcal { Z } _ { D + 1 } ^ { + } \| ^ { 2 } } \\ { \mathrm { s . t . } } & { \mathcal { P } _ { \Omega _ { D + 1 } } ( \mathcal { A } _ { D } * \mathcal { Z } + \mathcal { E } ) = \mathcal { M } _ { D + 1 } . } \end{array}\tag{4}
$$

• SAR coefficients update: The SAR coefficients are updated by minimizing the cumulative latent prediction error:

$$
\operatorname* { m i n } _ { w ^ { l } } \sum _ { d = L + 1 } ^ { D + 1 } \big \| \mathcal { Z } _ { d } ^ { \star } - \sum _ { l = 1 } ^ { \bar { L } } w ^ { l } \mathcal { Z } _ { d - h _ { l } } ^ { \star } \big \| ^ { 2 } .\tag{5}
$$

• Spatial basis update: The shared spatial basis is then updated to incorporate the latest recovered state:

$$
\operatorname* { m i n } _ { \boldsymbol { A } } \sum _ { d = 1 } ^ { D + 1 } \| \boldsymbol { A } \ast \mathcal { Z } _ { d } ^ { \star } + \boldsymbol { \mathcal { E } } _ { d } ^ { \star } - \boldsymbol { \mathcal { X } } _ { d } ^ { \star } \| ^ { 2 } , \quad \mathrm { s . t . } \quad \boldsymbol { A } ^ { \top } \ast \boldsymbol { \mathcal { A } } = \mathbb { Z } .\tag{6}
$$

Recursion. The updated parameters $A _ { D + 1 }$ and $w _ { D + 1 } ^ { l }$ are used to roll the model forward and generate the next forecast:

$$
\mathcal { X } _ { D + 2 } ^ { + } = \mathcal { A } _ { D + 1 } \ast \mathcal { Z } _ { D + 2 } ^ { + } , \mathrm { ~ w h e r e ~ } \mathcal { Z } _ { D + 2 } ^ { + } = \sum _ { l = 1 } ^ { \bar { L } } w _ { D + 1 } ^ { l } \mathcal { Z } _ { D + 2 - h _ { l } } ^ { \star } .
$$

Remark 3.1 (Computational efficiency): The efficiency of the online stage comes from separating current state inference from global parameter updates. When a new partial observation arrives, the previously inferred latent tensors $\{ \mathcal { Z } _ { 1 } ^ { \star } , \ldots , \mathcal { Z } _ { D } ^ { \star } \}$ are kept fixed, and iterative optimization is performed only for the current day latent inference problem. The SAR coefficients and the shared spatial basis are then updated through closed-form steps using summarized historical information. In this way, MVCTD avoids re-optimizing the full historical sequence during online adaptation.

The full workflow of MVCTD, from offline initialization to online adaptive prediction, is summarized in Fig. 2.

D. Theoretical Unification of Multi-View Coupled Tensor Decomposition

To clarify the representational scope of the proposed tproduct model, we relate it to standard matrix decompositions of individual traffic views. The following result shows that the coupled tensor representation can embed both independently decomposed views and the special case where all views share a common spatial basis.

Theorem 3.2: Let $\mathcal { X } \in \mathbb { R } ^ { N \times T \times V }$ be a third-order tensor. Suppose that for each view $v \in [ V ]$ , the corresponding frontal slice admits a matrix decomposition

$$
X ^ { ( v ) } : = \mathcal { X } ( : , : , v ) = A _ { v } Z _ { v } ,
$$

where $A _ { v } \in \mathbb R ^ { N \times r _ { v } }$ and $Z _ { v } \ \in \ \mathbb R ^ { r _ { v } \times T }$ . Then there exist an integer $\begin{array} { r } { p \leq \sum _ { v = 1 } ^ { V } r _ { v } } \end{array}$ and two tensors $\mathcal { A } \in \mathbb { R } ^ { N \times p \times V }$ and $\mathcal { Z } \in \mathbb { R } ^ { p \times T \times V }$ such that

$$
\chi = \mathcal { A } \ast \mathcal { Z } .
$$

More specifically, the following two cases can be embedded in this t-product form:

• Independent-view case. When the views are allowed to have distinct spatial bases, one may set $\begin{array} { r } { p = \sum _ { v = 1 } ^ { V } r _ { v } } \end{array}$ and construct the Fourier domain slices for each $k \in [ V ]$ as

$$
\hat { A } ^ { ( k ) } = \big [ A _ { 1 } A _ { 2 } \cdots A _ { V } \big ] , \quad \hat { Z } ^ { ( k ) } = \left[ \begin{array} { c } { \omega _ { k } ^ { 0 } Z _ { 1 } } \\ { \vdots } \\ { \omega _ { k } ^ { V - 1 } Z _ { V } } \end{array} \right] ,
$$

where $\omega _ { k }$ denotes the corresponding Fourier phase factor. • Shared-basis case. If all views share an identical spatial basis, namely $A _ { v } ~ = ~ A$ for all v, the representation reduces to $p = r _ { 1 }$ . In this case, A reduces to the form

$$
A ^ { ( 1 ) } = A , \quad A ^ { ( v ) } = 0 , v \neq 1 ,
$$

and the frontal slices of Z satisfy $Z ^ { ( v ) } = Z _ { v }$ . Consequently, the coupled tensor model recovers the classical joint matrix decomposition form.

This result shows that the proposed t-product representation covers both fully view-specific and shared-basis multi-view decompositions. Hence, MVCTD can model traffic variables that share common road-network structure while retaining view-dependent temporal patterns.

## IV. OPTIMIZATION ALGORITHM

To solve the non-convex offline model in (2), we develop an Alternating Direction Method of Multipliers (ADMM) based alternating algorithm. Two auxiliary variables, $\mathcal { V } _ { d } = \mathcal { Z } _ { d }$ and $\mathcal { C } _ { d } ~ = ~ \mathcal { Z } _ { d } .$ , are introduced to decouple the latent temporal regularization and SAR fitting terms from the update of ${ \mathcal { Z } } _ { d } .$ The resulting augmented Lagrangian function is given by

$$
\begin{array} { l } { \displaystyle \mathbb { I } _ { \eta } \big ( \mathcal { X } _ { d } , \mathcal { A } , \mathcal { Z } _ { d } , \mathcal { Y } _ { d } , \mathcal { C } _ { d } , \mathcal { E } _ { d } , w ^ { l } ; \mathcal { P } _ { d } , \mathcal { C } _ { d } , \mathcal { R } _ { d } \big ) = \sum _ { d = 1 } ^ { D } \big [ \lambda _ { 1 } \| \mathcal { Z } _ { d } \| _ { \mathbb { H } ^ { s } } ^ { p } } \\ { \displaystyle + \lambda _ { 2 } \| \mathcal { E } _ { d } \| _ { \mathbb { T } ^ { p } } ^ { 1 } + \lambda _ { 3 } \| \mathcal { E } \circ \mathcal { Y } _ { d } \| _ { s } + \lambda _ { 4 } \| \ell \boxtimes \mathcal { Y } _ { d } \| _ { n F } \big ] + } \\ { \displaystyle \gamma \sum _ { d = L + 1 } ^ { D } \big \| \mathcal { Z } _ { d } - \sum _ { \iota = 1 } ^ { \tilde { L } } w ^ { \iota } { \mathcal { C } } _ { d - h \iota } \big \| ^ { 2 } + \sum _ { d = 1 } ^ { D } \big [ \big \langle A + \mathcal { Z } _ { d } + \mathcal { E } _ { d } - \mathcal { X } _ { d } , } \\ { \displaystyle \mathcal { P } _ { d } \big \rangle + \frac { \eta _ { 1 } } { 2 } \big \| A \ast \mathcal { Z } _ { d } + \mathcal { E } _ { d } - \mathcal { X } _ { d } \big \| ^ { 2 } \big ] + \sum _ { d = 1 } ^ { D } \big [ \big \langle \mathcal { Y } _ { d } - \mathcal { Z } _ { d } , \mathcal { Q } _ { d } \big \rangle + } \\  \displaystyle \frac { \eta _ { 2 } } { 2 } \| \mathcal { Y } _ { d } - \mathcal { Z } _ { d } \| ^ { 2 } \big ] + \sum _ { d = 1 } ^ { D } \big [ \big \langle \mathcal { C } _ { d } - \mathcal { Z } _ { d } , \mathcal { R } _ { d } \big \rangle + \frac { \eta _ { 3 } } { 2 } \| \mathcal { C } _ { d } - \mathcal { Z } _ { d } \| \end{array}
$$

Here, $\mathcal { P } _ { d } , \mathcal { Q } _ { d } .$ and $\mathcal { R } _ { d }$ are dual variables, and $\eta _ { i } > 0 , \ i \in [ 3 ]$ are penalty parameters. The primal variables are updated by minimizing this augmented Lagrangian with respect to one block at a time, followed by the dual variable updates. The main subproblems are derived below.

Computing $\lambda _ { d } \colon$ The sub-problem to update $\mathcal { X } _ { d }$ is

$$
\operatorname * { a r g m i n } _ { \mathcal { P } _ { \Omega _ { d } } ( \mathcal { X } _ { d } ) = \mathcal { M } _ { d } } \frac { \eta _ { 1 } } { 2 } \| \mathcal { A } \ast \mathcal { Z } _ { d } + \mathcal { E } _ { d } - \mathcal { X } _ { d } + \mathcal { P } _ { d } / \eta _ { 1 } \| ^ { 2 } .\tag{7}
$$

Clearly, the optimal solution to the $\mathcal { X } _ { d }$ sub-problem is given by

$$
\mathcal { X } _ { d } = \mathcal { P } _ { \Omega _ { d } ^ { c } } \big ( \boldsymbol { \mathcal { A } } * \mathcal { Z } _ { d } + \boldsymbol { \mathcal { E } } _ { d } + \mathcal { P } _ { d } / \eta _ { 1 } \big ) + \mathcal { M } _ { d } .\tag{8}
$$

![](images/f552d1cfe970758328cf96a117bf7a46239d4c71e054254a97cf83e602b2018a.jpg)  
Fig. 2: The proposed MVCTD framework.

Computing A: The sub-problem to update A is

$$
\begin{array} { r } { \underset { \mathcal { A } ^ { \top } \ast \mathcal { A } = \mathcal { T } } { \arg \operatorname* { m i n } } \sum _ { d = 1 } ^ { D } \frac { \eta _ { 1 } } { 2 } \| \mathcal { A } \ast \mathcal { Z } _ { d } + \mathcal { E } _ { d } - \mathcal { X } _ { d } + \mathcal { P } _ { d } / \eta _ { 1 } \| ^ { 2 } } \\ { = \underset { \mathcal { A } ^ { \top } \ast \mathcal { A } = \mathcal { T } } { \arg \operatorname* { m a x } } \ \langle \mathcal { A } , \displaystyle \sum _ { d = 1 } ^ { D } \big ( \eta _ { 1 } \mathcal { X } _ { d } - \eta _ { 1 } \mathcal { E } _ { d } - \mathcal { P } _ { d } \big ) \ast \mathcal { Z } _ { d } ^ { \top } \big \rangle . } \end{array}\tag{9}
$$

By using [38, Lemma 4.1], the optimal solution of (9) is given by

$$
\mathcal { A } = \mathcal { U } _ { \mathcal { A } } \ast \mathcal { V } _ { \mathcal { A } } ^ { \top } ,\tag{10}
$$

where $\mathcal { U } _ { A }$ and $\mathcal { V } _ { A }$ are obtained by the skinny t-SVD decomposition: $\begin{array} { r } { \sum _ { d = 1 } ^ { D } \left( \eta _ { 1 } \mathscr { X } _ { d } - \eta _ { 1 } \mathscr { E } _ { d } - \dot { \mathcal { P } } _ { d } \right) * \mathscr { Z } _ { d } ^ { \top } = \mathcal { U } _ { A } * \mathscr { S } _ { A } * \mathscr { V } _ { A } ^ { \top } } \end{array}$

Computing $\mathcal { Z } _ { d } \mathbf { : }$ The sub-problem to update $\mathcal { Z } _ { d }$ i

$$
\begin{array} { r l } {  { \operatorname * { a r g m i n } _ { \mathcal { Z } _ { d } } \| \mathcal { Z } _ { d } \| _ { \mathrm { H S } } ^ { p } + \gamma \mathbb { 1 } _ { ( L , \infty ) } ( d ) \| \mathcal { Z } _ { d } - \sum _ { l = 1 } ^ { L } w ^ { l } \mathcal { C } _ { d - h _ { l } } \| ^ { 2 } } } \\ & { \ +  \mathcal { A } \neq \mathcal { E } _ { d } + \mathcal { E } _ { d } - \mathcal { X } _ { d } , \mathcal { P } _ { d }  + \frac { \eta _ { 1 } } { 2 } \| A * \mathcal { Z } _ { d } + \mathcal { E } _ { d } } \\ & { \ - \mathcal { X } _ { d } \| ^ { 2 } +  \mathcal { Y } _ { d } - \mathcal { Z } _ { d } , \mathcal { Q } _ { d }  + \frac { \eta _ { 2 } } { 2 } \| \mathcal { Y } _ { d } - \mathcal { Z } _ { d } \| ^ { 2 } + } \\ & { \ \qquad \mathcal { C } _ { d } - \mathcal { Z } _ { d } , \mathcal { R } _ { d }  + \frac { \eta _ { 3 } } { 2 } \| \mathcal { C } _ { d } - \mathcal { Z } _ { d } \| ^ { 2 } } \\ & { = \underset { \mathcal { Z } _ { d } } { \operatorname { a r g m i n } } \lambda _ { 1 } \| \mathcal { Z } _ { d } \| _ { \mathrm { H S } } ^ { p } + \frac { \lambda _ { \mathcal { Z } } } { 2 } \| \mathcal { Z } _ { d } - \tilde { \mathcal { Z } } / \lambda _ { \mathcal { Z } } \| ^ { 2 } , } \end{array}\tag{11}
$$

where $\begin{array} { r c l } { { \lambda _ { Z } } } & { { = } } & { { 2 \gamma \mathbb { 1 } _ { ( L , \infty ) } ( d ) \ + \ \eta _ { 1 } \ + \ \eta _ { 2 } \ + \ \eta _ { 3 } } } \end{array}$ and $\begin{array} { r l } { \tilde { \mathcal { Z } } } & { { } = } \end{array}$ $\begin{array} { r } { 2 \gamma \mathbb { 1 } _ { ( L , \infty ) } ( d ) \sum _ { l = 1 } ^ { L } w ^ { l } \mathcal { C } _ { d - h _ { l } } + \eta _ { 1 } \mathcal { A } ^ { \top } \ast \left( \mathcal { X } _ { d } - \mathcal { E } _ { d } \right) - \mathcal { A } ^ { \top } \ast \mathcal { P } _ { d } + } \end{array}$ $\eta _ { 2 } { \mathcal { Y } } _ { d } + { \mathcal { Q } } _ { d } + \eta _ { 3 } { \mathcal { C } } _ { d } + { \mathcal { R } } _ { d } .$ . By [41, Appendix A], for each $r \in [ R ]$ the optimal solution of (11) is given by

$$
\mathcal { Z } _ { d } \left( r , : , : \right) = \left\{ \begin{array} { l l } { \operatorname* { P r o x } _ { \frac { \lambda _ { 1 } } { \lambda _ { \mathcal { Z } } } \mid \cdot \mid ^ { p } } \left( \frac { \tilde { z } } { \lambda z } \right) \cdot \tilde { \mathcal { Z } } \left( r , : , : \right) / \tilde { z } , } & { \mathrm { i f ~ } \tilde { z } \neq 0 ; } \\ { 0 , } & { \mathrm { i f ~ } \tilde { z } = 0 , } \end{array} \right.\tag{12}
$$

where $\tilde { z } = \| \tilde { \mathcal { Z } } \left( r , : , : \right) \|$

Computing $y _ { d } \mathbf { : }$ The sub-problem to update $\mathcal { V } _ { d }$ is

$$
\underset { \mathcal { V } _ { d } } { \arg \operatorname* { m i n } } \ \lambda _ { 3 } \sum _ { r = 1 } ^ { R } \sum _ { v = 1 } ^ { V } \| \mathcal { C } \big ( \mathcal { V } _ { d } ( r , : , v ) \big ) \| _ { * } + \lambda _ { 4 } \sum _ { r = 1 } ^ { R } \sum _ { v = 1 } ^ { V }\tag{13}
$$

According to Lemmas 2.1 and 2.2, each component $\textbf { \scriptsize { y } } : =$ $\mathcal { D } _ { d } ( r , : , v )$ of the tensor $\mathcal { \partial } _ { d }$ is updated independently as follows:

$$
\begin{array} { r l r } & { } & { \underset { \pmb { y } } { \arg \operatorname* { m i n } } \ \lambda _ { 3 } \| \mathcal { C } \big ( \pmb { y } \big ) \| _ { * } + \lambda _ { 4 } \| \pmb { \ell \bullet y } \| ^ { 2 } + \frac { \eta _ { 2 } } { 2 } \| \pmb { y } - \pmb { h } \| ^ { 2 } } \\ & { } & { = \underset { \pmb { y } } { \arg \operatorname* { m i n } } \ \lambda _ { 3 } \| \pmb { \hat { y } } \| _ { 1 } + \frac { \lambda _ { 4 } } { T } \| \hat { \pmb { \ell } } \circ \pmb { \hat { y } } \| ^ { 2 } + \frac { \eta _ { 2 } } { 2 T } \| \hat { \pmb { y } } - \hat { \pmb { h } } \| ^ { 2 } , } \end{array}\tag{14}
$$

where $\textbf { \em h } = \mathcal { Z } _ { d } ( r , : , v ) - \mathcal { Q } _ { d } ( r , : , v ) / \eta _ { 2 }$ . On each $\hat { y } _ { i } ,$ , the optimization problem is given by

$$
\begin{array} { r l } & { \underset { \hat { y } _ { i } } { \arg \operatorname* { m i n } } ~ \lambda _ { 3 } T \big | \hat { y } _ { i } \big | + \frac { 2 \lambda _ { 4 } \hat { \ell } _ { i } ^ { 2 } + \eta _ { 2 } } { 2 } \big ( \hat { y } _ { i } - \frac { \eta _ { 2 } \hat { h } _ { i } } { 2 \lambda _ { 4 } \hat { \ell } _ { i } ^ { 2 } + \eta _ { 2 } } \big ) ^ { 2 } } \\ & { = \mathrm { s g n } \big ( \frac { \eta _ { 2 } \hat { h } _ { i } } { 2 \lambda _ { 4 } \hat { \ell } _ { i } ^ { 2 } + \eta _ { 2 } } \big ) \cdot \operatorname* { m a x } \big \{ \big | \frac { \eta _ { 2 } \hat { h } _ { i } } { 2 \lambda _ { 4 } \hat { \ell } _ { i } ^ { 2 } + \eta _ { 2 } } \big | - \frac { \lambda _ { 3 } T } { 2 \lambda _ { 4 } \hat { \ell } _ { i } ^ { 2 } + \eta _ { 2 } } , 0 \big \} . } \end{array}\tag{15}
$$

Computing $\mathcal { C } _ { d } \mathbf { : }$ The sub-problem to update $\mathcal { C } _ { d }$ is

$$
\begin{array} { r l r } & { } & { \arg \operatorname* { m i n } _ { \mathcal { C } _ { d } } \gamma \displaystyle \sum _ { d = L + 1 } ^ { D } \left\| { \mathcal { Z } _ { d } } - \sum _ { l = 1 } ^ { \bar { L } } w ^ { l } { \mathcal { C } _ { d - h _ { l } } } \right\| ^ { 2 } + \left. { { \mathcal { C } _ { d } } - { \mathcal { Z } _ { d } } , { \mathcal { R } _ { d } } } \right. } \\ & { } & { \quad \quad \quad \quad \quad \quad \quad \quad + \frac { \eta _ { 3 } } { 2 } \| { \mathcal { C } _ { d } } - { \mathcal { Z } _ { d } } \| ^ { 2 } . \quad ( 1 6 ) } \end{array}
$$

Let $J ( \mathcal { C } _ { d } )$ denote the objective function in (16). The gradient $\nabla { c _ { d } } J ( \mathcal { C } _ { d } ) = 0$ is computed as

$$
\eta _ { 3 } ( \mathcal { C } _ { d } - \mathcal { Z } _ { d } ) + \mathcal { R } _ { d } + \gamma \sum _ { l = 1 } ^ { \bar { L } } \mathbb { 1 } _ { ( L , D ] } ( d + h _ { l } ) \cdot \big [ - 2 w ^ { l } \big ( \mathcal { Z } _ { d + h _ { l } } -
$$

$$
\sum _ { j = 1 } ^ { \bar { L } } w ^ { j } \mathcal { C } _ { d + h _ { l } - h _ { j } } ) \big ] = 0 .\tag{17}
$$

This yields the update rule for $\mathcal { C } _ { d }$ given by

$$
\mathcal { C } _ { d } = \big ( \eta _ { 3 } \mathcal { Z } _ { d } - \mathcal { R } _ { d } + 2 \gamma \sum _ { l = 1 } ^ { \bar { L } } \mathbb { 1 } _ { ( L , D ] } ( d + h _ { l } ) w ^ { l } \big ( \mathcal { Z } _ { d + h _ { l } } -
$$

$$
\sum _ { j \neq l } w ^ { j } \mathcal C _ { d + h _ { l } - h _ { j } } \big ) \big ) / \big ( \eta _ { 3 } + 2 \gamma \sum _ { l = 1 } ^ { \bar { L } } \mathbb { 1 } _ { ( L , D ] } ( d + h _ { l } ) ( w ^ { l } ) ^ { 2 } \big ) .\tag{18}
$$

Computing $\mathcal { E } _ { d } \mathbf { : }$ The sub-problem to update $\mathcal { E } _ { d }$ is

arg min $\lambda _ { 2 } \| \mathcal { E } _ { d } \| _ { \mathrm { T F } } ^ { p } + \frac { \eta _ { 1 } } { 2 } \| \mathcal { A } \ast \mathcal { Z } _ { d } + \mathcal { E } _ { d } - \mathcal { X } _ { d } + \mathcal { P } _ { d } / \eta _ { 1 } \| ^ { 2 } .$ E<sub>d</sub>

(19)

Analogous to the update of $\mathcal { Z } _ { d } .$ , for each $n \in [ N ] , \ t \in [ T ]$ the optimal solution of (19) is given by

$$
\mathcal { E } _ { d } \left( n , t , : \right) = \left\{ \begin{array} { l l } { \operatorname* { P r o x } _ { \frac { \lambda _ { 2 } } { \eta _ { 1 } } | \cdot | ^ { p } } \left( \tilde { e } \right) \cdot \tilde { \mathcal { E } } \left( n , t , : \right) / \tilde { e } , } & { \mathrm { i f ~ } \tilde { e } \neq 0 ; } \\ { 0 , } & { \mathrm { i f ~ } \tilde { e } = 0 , } \end{array} \right.\tag{20}
$$

where $\tilde { \mathcal { E } } = \mathcal { X } _ { d } - \mathcal { A } * \mathcal { Z } _ { d } - \mathcal { P } _ { d } / \eta _ { 1 }$ and $\tilde { e } = \| \tilde { \mathcal { E } } \left( n , t , : \right) \|$

Computing $w ^ { l } \colon$ The sub-problem to update $w ^ { l }$ is

$$
\underset { w ^ { l } } { \arg \operatorname* { m i n } } \ \sum _ { d = L + 1 } ^ { D } \left\| \mathcal Z _ { d } - \sum _ { l = 1 } ^ { \bar { L } } w ^ { l } \mathcal C _ { d - h _ { l } } \right\| ^ { 2 } .\tag{21}
$$

Let $\pmb { w } = [ w ^ { 1 } , \dots , w ^ { \bar { L } } ] ^ { \top } \in \mathbb { R } ^ { \bar { L } }$ denote the coefficient vector. 1 This is a linear least squares problem with the closed-form 12 solution given by

$$
w = F ^ { - 1 } g ,
$$

where the entries of $F ~ \in ~ \mathbb { R } ^ { \bar { L } \times \bar { L } }$ and $\textbf {  { g } } \in \ \mathbb { R } ^ { \bar { L } }$ are computed by $\begin{array} { r } { F ( k , l ) \ = \ \sum _ { d = L + 1 } ^ { D } \langle { \mathcal C } _ { d - h _ { k } } , { \mathcal C } _ { d - h _ { l } } \rangle } \end{array}$ and $\mathbf { \pmb { g } } ( k )$ =<sup>1</sup> $\scriptstyle \sum _ { d = L + 1 } ^ { D } \langle \mathcal { Z } _ { d } , \mathcal { C } _ { d - h _ { k } } \rangle$ 15 15 , respectively.

Based on the derived update rules, we summarize the complete batch ADMM procedure for the offline initialization in Algorithm 4.1.

Although Algorithm 4.1 provides a reliable initialization, applying it repeatedly after each new observation would be too expensive for streaming prediction. We therefore use a lightweight online procedure. The prior $\mathcal { Z } _ { D + 1 } ^ { + }$ from (3) is used as a warm start, and ADMM is run only for the current day inference problem (4). After the current latent tensor is obtained, the SAR coefficients and the spatial dictionary are refreshed through the closed-form problems (5) and (6). This procedure is summarized in Algorithm 4.2. In practice, a full retraining or an incremental/global learning strategy such as [42] can be triggered when the validation error becomes large.

Complexity Analysis. Let $K _ { \mathrm { o f f } }$ and $K _ { \mathrm { o n } }$ denote the numbers of ADMM iterations in Algorithms 4.1 and 4.2, respectively. The offline stage is dominated by t-product operations, FFT based temporal shrinkage, and the SAR least-squares update. The offline initialization requires $\mathcal { O } \big ( K _ { \mathrm { o f f } } \big ( D \bar { V } R T ( N \bar { \mathbf {  { t } } }$ log $T + \bar { L } ^ { 2 } ) + \bar { L } ^ { 3 } ) \big )$ . For small R and ${ \bar { L } } ,$ this cost scales nearly linearly with the historical data size.

Algorithm 4.1 Stage I: Offline Initialization via Batch   
ADMM.   
Input: Incomplete tensors $\{ \mathcal { M } _ { d } \} _ { d = 1 } ^ { D } ,$ regularization parame   
ters γ and $\{ \lambda _ { i } \} _ { i = 1 } ^ { 4 } .$ Set $k \gets 0 .$   
1 Update $\mathcal { X } _ { d } ^ { k + \mathrm { 1 } }$ via (8);   
2 Update $\mathcal { A } ^ { \breve { k } + 1 }$ via (10);   
3 Update $\mathcal { Z } _ { d } ^ { k + 1 }$ via (12);   
4 Update $\mathcal { V } _ { d } ^ { \breve { k } + 1 }$ via (15);   
5 Update $\mathcal { C } _ { d } ^ { \breve { k } + 1 }$ via (18);   
6 Update $\mathcal { E } _ { d } ^ { \breve { k } + 1 }$ via (20);   
7 Update $\mathbf { \Delta } \overset { \sim } { w ^ { l } } ^ { k + 1 }$ via (22);   
8 Update Lagrangian multipliers and penalty parameters via   
$\left( \begin{array} { l } { \mathcal { P } _ { d } ^ { k + 1 } = \mathcal { P } _ { d } ^ { k } + \eta _ { 1 } ^ { k } ( \mathcal { A } _ { . } ^ { k + 1 } * \mathcal { Z } _ { d . } ^ { k + 1 } + \mathcal { E } _ { d } ^ { k + 1 } - \mathcal { X } _ { d } ^ { k + 1 } ) ; } \end{array} \right.$   
$\mathcal { Q } _ { d } ^ { \breve { k } + 1 } = \mathcal { Q } _ { d } ^ { \breve { k } } + \eta _ { 2 } ^ { \breve { k } } ( \mathcal { y } _ { d } ^ { k + 1 } - \breve { \mathcal { Z } } _ { d } ^ { k + 1 } ) ;$   
$\mathcal { R } _ { , d } ^ { \tilde { k } + 1 } = \mathcal { R } _ { , d } ^ { \tilde { k } } + \eta _ { 3 } ^ { \bar { k } } ( \mathcal { C } _ { d } ^ { \tilde { k } + 1 } - \mathcal { Z } _ { d } ^ { \tilde { k } + 1 } ) ;$   
$\eta _ { i } ^ { k + 1 } = \beta \eta _ { i } ^ { \bar { k } } , i \in [ 3 ] .$   
(23)   
9 If a termination criterion is met, set $\mathcal { X } _ { d } ^ { \star } : = \mathcal { X } _ { d } ^ { k + 1 } , \mathcal { E } _ { d } ^ { \star } : = \mathcal { E } _ { d } ^ { \dot { k } + 1 } ,$   
$\mathcal { A } _ { D } : = \mathcal { A } ^ { k + 1 } , \mathcal { Z } _ { d } ^ { \star } : = \mathcal { Z } _ { d } ^ { k + 1 } , w _ { D } ^ { l } : = w ^ { l ^ { k + 1 } }$ ; else, set $k \gets$   
$k + 1 ,$ return to Step 1;   
Output: Recover tensors $\{ \mathcal { X } _ { d } ^ { \star } \} _ { d = 1 } ^ { D } ,$ , anomaly tensor $\{ \mathcal { E } _ { d } ^ { \star } \} _ { d = 1 } ^ { D } ,$   
dictionary tensor $A _ { D } .$ , latent tensors $\{ \mathcal { Z } _ { d } ^ { \star } \} _ { d = 1 } ^ { D } .$ , and   
autoregressive coefficients $\{ w _ { D } ^ { l } \} _ { l = 1 } ^ { \bar { L } } .$   
Algorithm 4.2 Stage II: Lightweight Online Adaptive Predic   
tion   
Input: Tensors $\mathcal { A } _ { D } , \{ \mathcal { Z } _ { d } ^ { \star } \} _ { d = 1 } ^ { D } ,$ and $\{ w _ { D } ^ { l } \} _ { l = 1 } ^ { \bar { L } }$ from Algorithm   
4.1.   
10 Predict $\mathcal { Z } _ { D + 1 } ^ { + }$ and $\boldsymbol { \mathcal { X } } _ { D + 1 } ^ { + }$ via (3);   
1 If the prediction horizon is reached, terminate; otherwise,   
proceed to the next step;   
12 Initialize $\mathcal { Z } _ { D + 1 } ^ { 0 } = \mathcal { Z } _ { D + 1 } ^ { + }$ and $\mathscr { X } _ { D + 1 } ^ { 0 } = \mathscr { P } _ { \Omega ^ { c } } ( \mathcal { Z } _ { D + 1 } ^ { + } ) + \mathscr { M } _ { D + 1 }$   
; // Warm-start initialization   
13 Update $\mathcal { Z } _ { D + 1 } ^ { \star } , \mathcal { E } _ { D + 1 } ^ { \star }$ by solving model (4) via ADMM ;   
$/ / \cot$ imize for current day only   
4 Sequentially update $w _ { D + 1 } ^ { l } , l \in [ \bar { L } ]$ and $A _ { D + 1 }$ via (5) and (6);   
$D \gets D + 1 ;$   
Output: Predict stream $\mathcal { X } _ { D + 1 } ^ { + } , \mathcal { X } _ { D + 2 } ^ { + } , . . . .$

At each online step, ADMM updates only the latent tensor of the newly arrived day, after which the model parameters are refreshed in closed form. The per-step complexity is

$$
\mathcal { O } \big ( K _ { \mathrm { o n } } ( V N R T + R V T \log T ) + \bar { L } ^ { 2 } R V T + \bar { L } ^ { 3 } \big ) .
$$

Compared with batch re-optimization, the online update avoids the multiplicative factor $D$ in the iterative part, making MVCTD suitable for streaming prediction.

We now establish the convergence of the proposed Algorithm 4.1 through the following theorem.

Theorem 4.1: Let $\{ \Theta ^ { k } \} _ { k = 1 } ^ { \infty }$ denote the sequence generated by Algorithm 4.1, where $\begin{array} { r l r l } \Theta ^ { k }  & { { } } & { = } \end{array}$ $\{ \boldsymbol { \mathcal { X } } _ { d } ^ { k } , \boldsymbol { \mathcal { A } } ^ { k } , \boldsymbol { \mathcal { Z } } _ { d } ^ { k } , \boldsymbol { \bar { \mathcal { Y } } } _ { d } ^ { k } , \boldsymbol { \mathcal { C } } _ { d } ^ { k } , \boldsymbol { \bar { \mathcal { E } } } _ { d } ^ { k } , \boldsymbol { w } ^ { l ^ { k } } , \boldsymbol { \mathcal { P } } _ { d } ^ { k } , \boldsymbol { \mathcal { Q } } _ { d } ^ { k } , \boldsymbol { \mathcal { R } } _ { d } ^ { k } \}$ . If the dual sequences $\{ \mathcal { Q } _ { d } ^ { k } \} _ { k = 1 } ^ { \infty }$ and $\{ \mathcal { R } _ { d } ^ { k } \} _ { k = 1 } ^ { \infty }$ are bounded, then every accumulation point of $\{ \Theta ^ { k } \} _ { k = 1 } ^ { \infty }$ satisfies the Karush-Kuhn-Tucker (KKT) conditions of problem (2).

## V. NUMERICAL EXPERIMENTS

This section evaluates MVCTD on real-world traffic prediction tasks with incomplete multi-view observations. All experiments are conducted in MATLAB 2025b on a workstation with an Intel Core i5-12500H CPU (2.50 GHz) and 16 GB RAM.

## A. Experimental Settings

1) Implementation Details: For all competing methods, we use the parameter settings recommended in their original papers. For MVCTD, the regularization parameters are empirically set to $\lambda _ { 1 } = 2 e { - } 1 , \lambda _ { 2 } = 5 e { - } 1 , \lambda _ { 3 } = 1 e { - } 2 , \lambda _ { 4 } = 2 e { - } 1$ and $\gamma = 5 e { \mathrm { - } } 1$ . The penalty growth factor is set to $\beta = 1 . 1 5$ The SAR lag set is fixed as ${ \mathcal { L } } = \{ 7 \}$ , so that $\bar { L } = | \mathcal { L } | = 1$ and $L = \operatorname* { m a x } \mathcal L = 7 .$ The optimization is terminated if the following criterion is satisfied for five consecutive iterations:

$$
\begin{array} { r l } & { \mathrm { E r r : = m a x \left\{ \| \mathcal { A } ^ { k + 1 } \ast \mathcal { Z } _ { d } ^ { k + 1 } - \mathcal { A } ^ { k } \ast \mathcal { Z } _ { d } ^ { k } \| _ { \infty } , \right. } } \\ & { \quad \quad \quad \quad \left. \| \mathcal { E } _ { d } ^ { k + 1 } - \mathcal { E } _ { d } ^ { k } \| _ { \infty } , \ \| \mathcal { X } _ { d } ^ { k + 1 } - \mathcal { X } _ { d } ^ { k } \| _ { \infty } \right\} < 1 e - 3 . } \end{array}
$$

The maximum number of iterations is set to 200.

2) Traffic Datasets: We use two real-world traffic datasets from the Caltrans Performance Measurement System $( \mathrm { P e M S } ) ^ { 1 }$ Both datasets contain volume, occupancy, and speed measurements at a 5-minute resolution, forming a three-view traffic tensor. Occupancy values are rescaled from [0, 1] to [0, 100] before evaluation.

• PeMS-D4. This dataset contains measurements from 307 loop detectors in the San Francisco Bay Area (District 4) from January 1 to February 28, 2018. It contains 59 days and 16, 992 time intervals, yielding a tensor of size $3 0 7 \times 1 6 , 9 9 2 \times 3$

• PeMS-D8. This dataset contains measurements from 170 sensors in the San Bernardino area (District 8) from July 1 to August 31, 2016. It contains 62 days and 17, 856 time intervals, yielding a tensor of size $1 7 0 \times 1 7 , 8 5 6 \times 3$

3) Evaluation Metrics: Prediction accuracy is evaluated by Mean Absolute Percentage Error (MAPE) and Root Mean Square Error (RMSE):

$$
\begin{array} { r l } & { { \mathrm { M A P E } } = \displaystyle \frac { 1 } { B } \sum _ { i , j , k } { \big \lvert \frac { \mathcal { M } _ { i j k } - \mathcal { X } _ { i j k } } { \mathcal { M } _ { i j k } } \big \rvert \times 1 0 0 } , } \\ & { { \mathrm { R M S E } } = \sqrt { \displaystyle \frac { 1 } { B } \sum _ { i , j , k } { \big ( \mathcal { M } _ { i j k } - \mathcal { X } _ { i j k } \big ) ^ { 2 } } } , } \end{array}
$$

where B is the number of evaluated entries, and $\mathcal { M } _ { i j k }$ and $\mathcal { X } _ { i j k }$ denote the ground-truth and predicted values at index $( i , j , k )$ , respectively.

4) Baseline Models: We compare MVCTD with six representative baselines covering low-rank factorization, Bayesian modeling, dynamical system based modeling, and deep learning.

• HTMF [43]. A temporal matrix factorization method that incorporates Hankel matrices into a low-dimensional latent space. It jointly optimizes factor matrices and Hankel structures via alternating minimization, enabling robust modeling of sparse and nonstationary traffic series.

• BTMF [3]. A Bayesian temporal factorization model that integrates low-rank decomposition with a VAR process in a probabilistic graphical model. It captures global and local temporal consistencies for robust prediction and uncertainty estimation under missing data.

• circDMDsp [44]. An anti-circulant dynamic mode decomposition method with sparsity constraints. It models traffic evolution as a high-dimensional nonlinear dynamical system and extracts interpretable spatio-temporal coherent structures for reconstruction and prediction.

• PatchTST [45]. A Transformer based model for multivariate time-series forecasting with a channelindependent architecture and patch-level segmentation. It reduces computational complexity while preserving local temporal semantics.

• PreSTGNet [46]. A spatiotemporal graph neural network with a two-stage pre-training and fine-tuning framework. It learns transferable representations from traffic networks and adapts them to the target forecasting task.

AmTGBiM [47]. A Mamba based model for long-term traffic prediction that combines an adaptive masked spatial Transformer with dynamic graph attention. It captures long-range temporal dependencies and dynamic spatial correlations among traffic sensors.

## B. Online Traffic Forecasting

We evaluate online forecasting performance on PeMS-D4 and PeMS-D8. Following the rolling prediction setting in Fig. 1, each experiment uses a 15-day historical window for training and a 5-day horizon for online prediction. To emulate sensor failures, 80% of the input entries are randomly masked. This setting evaluates whether a predictor can remain accurate and efficient when the incoming multi-view traffic stream is incomplete. Since circDMDsp, PatchTST, PreSTGNet and AmTGBiM cannot directly handle incomplete inputs, they are evaluated only on fully observed data and used as reference baselines.

Tables I and II report the forecasting results under fully observed and severely incomplete settings, respectively. When no entries are missing, MVCTD consistently obtains the lowest MAPE and RMSE on both datasets and across all three traffic views. The improvement is evident against both factorizationbased and deep forecasting baselines. For example, on PeMS-D8, MVCTD reduces the best competing MAPE from 21.65 to 11.20 for volume and from 25.77 to 13.19 for occupancy, while also achieving the smallest speed error.

The superiority of MVCTD is preserved under severe data sparsity. With 80% missing entries, MVCTD still ranks first on every reported metric in Table II. Its performance degradation from the fully observed case is also limited: the volume MAPE increases by 1.06 on PeMS-D4 and 0.95 on PeMS-D8, and the occupancy MAPE increases by less than one percentage point on both datasets. By comparison, HTMF and BTMF exhibit much larger prediction errors under the same missing rate, highlighting the robustness of MVCTD in highly incomplete online forecasting scenarios.

In addition to accuracy, MVCTD is computationally efficient. Under the fully observed setting, it requires 101 seconds on PeMS-D4 and 49 seconds on PeMS-D8, which is lower than all competing methods. Under the 80% missing rate, the running time further decreases to 72 seconds and 38 seconds, respectively. Compared with the fastest baseline in Table II, MVCTD is about 26 times faster on PeMS-D4 and more than 33 times faster on PeMS-D8. These results indicate that MVCTD provides an effective accuracy–efficiency balance for online traffic prediction.

TABLE I: Comparison of forecasting performance with full observations.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="3">MAPE</td><td colspan="3">RMSE</td><td rowspan="2">Time (s)</td></tr><tr><td>Volume</td><td>Occupancy</td><td>Speed</td><td>Volume</td><td>Occupancy</td><td>Speed</td></tr><tr><td rowspan="7">PeMS-D4</td><td>MVCTD</td><td>14.42</td><td>17.91</td><td>5.24</td><td>36.18</td><td>2.46</td><td>4.83</td><td>101</td></tr><tr><td>HTMF</td><td>32.41</td><td>59.13</td><td>6.80</td><td>55.13</td><td>3.00</td><td>5.54</td><td>4384</td></tr><tr><td>BTMF</td><td>36.52</td><td>64.11</td><td>7.43</td><td>67.78</td><td>3.31</td><td>5.99</td><td>663</td></tr><tr><td>circDMDsp</td><td>36.44</td><td>50.33</td><td>8.10</td><td>62.70</td><td>3.22</td><td>7.50</td><td>425</td></tr><tr><td>PatchTST</td><td>50.34</td><td>59.21</td><td>8.37</td><td>68.50</td><td>3.45</td><td>7.01</td><td>2686</td></tr><tr><td>PreSTGNet</td><td>38.08</td><td>51.09</td><td>9.18</td><td>70.94</td><td>3.81</td><td>7.44</td><td>23160</td></tr><tr><td>AmTGBiM</td><td>38.68</td><td>53.66</td><td>9.54</td><td>68.00</td><td>3.58</td><td>7.82</td><td>1418</td></tr><tr><td rowspan="7">PeMS-D8</td><td>MVCTD</td><td>11.20</td><td>13.19</td><td>3.89</td><td>36.99</td><td>2.25</td><td>4.20</td><td>49</td></tr><tr><td>HTMF</td><td>23.96</td><td>25.77</td><td>5.30</td><td>51.48</td><td>2.64</td><td>4.70</td><td>3227</td></tr><tr><td>BTMF</td><td>24.21</td><td>33.09</td><td>5.87</td><td>57.22</td><td>2.87</td><td>5.19</td><td>1757</td></tr><tr><td>circDMDsp</td><td>21.65</td><td>26.78</td><td>6.06</td><td>50.43</td><td>2.64</td><td>5.79</td><td>289</td></tr><tr><td>PatchTST</td><td>26.85</td><td>32.49</td><td>5.96</td><td>58.30</td><td>2.85</td><td>5.38</td><td>2127</td></tr><tr><td>PreSTGNet</td><td>36.97</td><td>35.11</td><td>6.66</td><td>68.99</td><td>3.35</td><td>5.82</td><td>8409</td></tr><tr><td>AmTGBiM</td><td>23.93</td><td>35.66</td><td>7.55</td><td>61.47</td><td>3.03</td><td>6.56</td><td>855</td></tr></table>

TABLE II: Comparison of forecasting performance under 80% missing rate.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="3">MAPE</td><td colspan="3">RMSE</td><td rowspan="2">Time (s)</td></tr><tr><td>Volume</td><td>Occupancy</td><td>Speed</td><td>Volume</td><td>Occupancy</td><td>Speed</td></tr><tr><td rowspan="3">PeMS-D4</td><td>MVCTD</td><td>15.48</td><td>18.71</td><td>5.81</td><td>38.60</td><td>2.55</td><td>5.16</td><td>72</td></tr><tr><td>HTMF</td><td>35.68</td><td>52.79</td><td>7.18</td><td>58.62</td><td>2.94</td><td>5.68</td><td>2501</td></tr><tr><td>BTMF</td><td>36.59</td><td>65.04</td><td>7.60</td><td>69.45</td><td>3.33</td><td>6.07</td><td>1871</td></tr><tr><td rowspan="3">PeMS-D8</td><td>MVCTD</td><td>12.15</td><td>13.81</td><td>4.49</td><td>38.87</td><td>2.29</td><td>4.50</td><td>38</td></tr><tr><td>HTMF</td><td>27.46</td><td>37.09</td><td>6.28</td><td>57.33</td><td>3.83</td><td>5.19</td><td>3629</td></tr><tr><td>BTMF</td><td>23.87</td><td>32.11</td><td>5.83</td><td>57.30</td><td>2.85</td><td>5.12</td><td>1257</td></tr></table>

Fig. 3 further compares the 5-day prediction trajectories of average traffic volume on PeMS-D8 with full observations. The ground-truth series presents a pronounced daily pattern, where sharp peak periods alternate with low volume intervals from Day 16 to Day 20. MVCTD reproduces this pattern with smaller amplitude and timing errors, particularly around the rapid rising and falling edges of the daily peaks. HTMF and BTMF capture the coarse periodic trend but deviate more noticeably near peak intervals, leading to either amplified fluctuations or attenuated peak values. The deep forecasting baselines, including PatchTST, PreSTGNet, and AmTGBiM, generally follow the global trend but produce smoother curves, which makes them less responsive to short term peak changes. circDMDsp shows a similar smoothing effect and misses part of the peak-to-valley variation. These trajectory-level differences are consistent with the volume MAPE and RMSE comparisons in Table I.

## C. Model Analysis

1) Parameter Analysis: We evaluate the sensitivity of MVCTD to $\{ \lambda _ { i } \} _ { i = 1 } ^ { 4 }$ and γ on PeMS-D8 under a 40% missing rate. As shown in Fig. 4, the model is more sensitive to $\lambda _ { 1 }$ and $\lambda _ { 2 } .$ , which control the latent low-rank and anomaly sparsity terms, respectively. The temporal regularization parameters $\lambda _ { 3 }$ and $\lambda _ { 4 }$ are relatively stable over a broad range, while $\gamma$ has a localized favorable region. Based on these results, we set $\lambda _ { 1 } = 2 e { \mathrm { - } } 1 .$ , λ = 5e-1, $\lambda _ { 3 } = 1 e { \mathrm { - } } 2 .$ $\lambda _ { 4 } = 2 e \mathrm { - } 1$ , and $\gamma = 5 e { \mathrm { - } } 1$

2) Ablation Study: Table III reports the ablation results on PeMS-D8 with a 40% missing rate. Removing the anomaly sparsity term $( \lambda _ { 2 } )$ leads to the largest degradation, increasing volume MAPE from 11.31 to 22.94. The global periodicity $\left( \lambda _ { 3 } \right)$ and local smoothness $( \lambda _ { 4 } )$ terms also contribute to the final performance, as removing either term weakens the prediction accuracy across the three views. The full model achieves the best or tied-best results on nearly all metrics, confirming the effectiveness of the proposed regularization design.

TABLE III: Ablation study of different regularization terms on the PeMS-D8 dataset under 40% missing rate.
<table><tr><td rowspan="2">Method</td><td colspan="3">MAPE</td><td colspan="3">RMSE</td></tr><tr><td>Volume</td><td>Occupancy</td><td>Speed</td><td>Volume</td><td>Occupancy</td><td>Speed</td></tr><tr><td>w/o Anomaly</td><td>22.94</td><td>42.25</td><td>5.48</td><td>52.74</td><td>3.71</td><td>5.54</td></tr><tr><td>w/o Global Periodicity</td><td>12.22</td><td>14.22</td><td>3.98</td><td>38.00</td><td>2.25</td><td>4.29</td></tr><tr><td>w/o Local Smoothness</td><td>11.61</td><td>14.07</td><td>3.97</td><td>37.85</td><td>2.27</td><td>4.28</td></tr><tr><td>Proposed</td><td>11.31</td><td>13.26</td><td>3.94</td><td>37.50</td><td>2.25</td><td>4.24</td></tr></table>

3) Effectiveness of Multi-View Modeling: To evaluate the benefit of multi-view coupling, we compare MVCTD with a single-view variant trained independently for each traffic attribute. As shown in Table IV, MVCTD consistently outperforms the single-view variant on PeMS-D8, reducing speed MAPE from 4.67 to 4.49. On PeMS-D4, although the singleview variant obtains a slightly lower volume RMSE, MVCTD achieves lower MAPE values for all three views and reduces volume MAPE from 15.92 to 15.48. These results verify that coupling volume, occupancy, and speed provides useful complementary information under incomplete observations.

TABLE IV: Performance comparison between single-view and multi-view modeling strategies.
<table><tr><td rowspan="2">Datasets</td><td rowspan="2">Strategy</td><td colspan="3">MAPE (%)</td><td colspan="3">RMSE</td></tr><tr><td>Volume</td><td>Occupancy</td><td>Speed</td><td>Volume</td><td>Occupancy</td><td>Speed</td></tr><tr><td rowspan="2">PeMS-D4</td><td>Single-View</td><td>15.92</td><td>19.87</td><td>6.02</td><td>37.86</td><td>2.61</td><td>5.37</td></tr><tr><td>Proposed</td><td>15.48</td><td>18.71</td><td>5.81</td><td>38.60</td><td>2.55</td><td>5.16</td></tr><tr><td rowspan="2">PeMS-D8</td><td>Single-View</td><td>12.24</td><td>14.03</td><td>4.67</td><td>39.17</td><td>2.40</td><td>4.75</td></tr><tr><td>Proposed</td><td>12.15</td><td>13.81</td><td>4.49</td><td>38.87</td><td>2.29</td><td>4.50</td></tr></table>

4) Effectiveness of Subset Autoregression: Table V compares different autoregressive lag structures. Using only the immediate previous day ({1}) gives the weakest performance, indicating that longer-term dependencies are important for traffic forecasting. The continuous window $( \{ 1 , \ldots , 7 \} )$ substantially reduces the errors, but the proposed sparse lag structure performs better on most metrics. Although its volume RMSE is slightly higher, it achieves the lowest errors in the other reported views. This result suggests that subset autoregression captures the dominant periodic dependence while avoiding redundant intermediate lags.

![](images/4c00b3047566e860aa6460f27835c6374b522f8ee89e70daace32a7babce54a8.jpg)  
Fig. 3: Time series prediction of average traffic volume on the PeMS-D8 dataset with full observations.

![](images/1bd131e18ddfbbd852d605cb18aaf78bcdaa9849f26770471a0efc7f80052dbd.jpg)

![](images/a47daab41c075fd2af687111e0ef728ad7e667c5f646544eb861fa0694dede99.jpg)

![](images/95c01a043d30ad413a3203f07f054da06eb03d6944d8f4aeb6c3c047b627e3b3.jpg)

![](images/e03413b06033e8347992fcdc7d66038187ff7282068c0d3298fda5ae1283022a.jpg)  
Fig. 4: Parameter analysis of MVCTD with respect to $\lambda _ { i }$ and $\gamma$ on the PeMS-D8 dataset under 40% missing rate.

![](images/35060fcaccadbfeee31d4251d50d76b8a68699f7fb780ccf4b947b351b3ae15c.jpg)

TABLE V: Impact of different autoregressive lag structures on prediction performance.
<table><tr><td rowspan="2">Time Lag Set</td><td colspan="3">MAPE</td><td colspan="3">RMSE</td></tr><tr><td>Volume</td><td>Occupancy</td><td>Speed</td><td>Volume</td><td>Occupancy</td><td>Speed</td></tr><tr><td>{1}</td><td>22.74</td><td>26.77</td><td>5.62</td><td>60.62</td><td>3.03</td><td>5.44</td></tr><tr><td> $\{ 1 , \ldots , 7 \}$ </td><td>12.32</td><td>14.56</td><td>4.18</td><td>37.15</td><td>2.28</td><td>4.33</td></tr><tr><td>Proposed</td><td>11.31</td><td>13.26</td><td>3.94</td><td>37.50</td><td>2.25</td><td>4.24</td></tr></table>

5) Convergence Analysis: Fig. 5 shows the convergence curves of Algorithm 4.1 on PeMS-D8 under different missing rates. The error fluctuates during the first 20 iterations, which is expected for alternating minimization on a non-convex objective. It then decreases rapidly and reaches the stopping criterion within 90 iterations in all cases. This indicates that the proposed batch solver is practical for offline initialization.

![](images/98d62ad9994677491caaba16c652aa98b8264527dab5a17260db2151e4a8eefd.jpg)  
(a) MR = 0%

![](images/c14063c0a8110aa3c0a3003722d122cef1cd715b3c6ec43820bb57c58688a1c3.jpg)  
(b) MR = 40%

![](images/17fc9506d16907a8f187fd0468642138a4388950db794c2ef21ce7e640d947ec.jpg)  
(c) MR = 80%  
Fig. 5: Convergence curves of Algorithm 4.1 on the PeMS-D8 dataset under different Missing Rates (MR).

## VI. CONCLUSIONS

This paper proposed MVCTD, a multi-view coupled tensor decomposition model for online traffic prediction from incomplete and anomaly corrupted observations. MVCTD represents regular traffic states through a shared spatial dictionary and view-dependent latent temporal factors, so that complementary information among volume, occupancy, and speed can be exploited in a low-dimensional forecasting space. The model also integrates missing data recovery, anomaly separation, latent temporal regularization, and subset autoregression into a unified predictive formulation. To support streaming deployment, we developed a two-stage optimization scheme. The offline stage learns the initial coupled tensor representation from historical data, while the online stage refines only the current latent tensor and updates the remaining model parameters through lightweight closed-form steps based on summarized historical information. Experiments on PeMS-D4 and PeMS-D8 show that MVCTD achieves accurate and efficient prediction under severe missingness, and the ablation studies verify the contributions of anomaly modeling, temporal regularization, multi-view coupling, and subset autoregression. Future work will incorporate richer traffic-related factors, such as weather and network topology, and develop more adaptive online learning strategies for large-scale traffic systems.

## REFERENCES

[1] Z. Xu, Z. Lv, B. Chu, and J. Li, “Fast autoregressive tensor decomposition for online real-time traffic flow prediction,” Knowl.-Based Syst., vol. 282, p. 111125, Dec. 2023.

[2] M. Andres and R. Nair, “A predictive-control framework to address bus bunching,” Transp. Res. B: Methodol., vol. 104, pp. 123–148, Oct. 2017.

[3] X. Chen and L. Sun, “Bayesian temporal factorization for multidimensional time series prediction,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 44, no. 9, pp. 4659–4673, Sep. 2022.

[4] M. S. Ahmed and A. R. COOK, “Analysis of freeway traffic time-series data by using box-Jenkins techniques,” Transport. Res. Rec., vol. 722, pp. 1–9, 1979.

[5] M. M. Hamed, H. R. Al-Masaeid, and Z. M. B. Said, “Short-term prediction of traffic volume in urban arterials,” J. Transp. Eng., vol. 121, no. 3, pp. 249–254, May 1995.

[6] S. Lee and D. B. Fambro, “Application of subset autoregressive integrated moving average model for short-term freeway traffic volume forecasting,” Transport. Res. Rec., vol. 1678, no. 1, pp. 179–188, Jan. 1999.

[7] B. M. Williams, P. K. Durvasula, and D. E. Brown, “Urban freeway traffic flow prediction: Application of seasonal autoregressive integrated moving average and exponential smoothing models,” Transport. Res. Rec., vol. 1644, no. 1, pp. 132–141, Jan. 1998.

[8] K. Takeuchi, H. Kashima, and N. Ueda, “Autoregressive tensor factorization for spatio-temporal predictions,” in Proc. IEEE Int. Conf. Data Min. IEEE, Nov. 2017, pp. 1105–1110.

[9] P. Jing, Y. Su, X. Jin, and C. Zhang, “High-order temporal correlation model learning for time-series prediction,” IEEE Trans. Cybern., vol. 49, no. 6, pp. 2385–2397, Jun. 2019.

[10] H. Wang and L. Zhang, “High-dimensional low-rank three-way tensor autoregressive time series predictor,” Expert Syst. Appl., vol. 299, p. 130104, Mar. 2026.

[11] Z. Luan, D. Sun, H. Wang, and L. Zhang, “Efficient online prediction for high-dimensional time series via joint tensor Tucker decomposition,” J. Mach. Learn. Res., vol. 26, no. 261, pp. 1–30, 2025.

[12] X. Yin, G. Wu, J. Wei, Y. Shen, H. Qi, and B. Yin, “Deep learning on traffic prediction: Methods, analysis, and future directions,” IEEE Trans. Intell. Transp. Syst., vol. 23, no. 6, pp. 4927–4943, Jun. 2022.

[13] W. Huang, G. Song, H. Hong, and K. Xie, “Deep architecture for traffic flow prediction: Deep belief networks with multitask learning,” IEEE Trans. Intell. Transp. Syst., vol. 15, no. 5, pp. 2191–2201, Oct. 2014.

[14] R. Yasdi, “Prediction of road traffic using a neural network approach,” Neural. Comput. Appl., vol. 8, no. 2, pp. 135–142, May 1999.

[15] R. More, A. Mugal, S. Rajgure, R. B. Adhao, and V. K. Pachghare, “Road traffic prediction and congestion control using artificial neural networks,” in Proc. Int. Conf. Comput., Anal. Secur. Trends. IEEE, Dec. 2016, pp. 52–57.

[16] X. Ma, Z. Tao, Y. Wang, H. Yu, and Y. Wang, “Long short-term memory neural network for traffic speed prediction using remote microwave sensor data,” Transp. Res. C, Emerg. Technol., vol. 54, pp. 187–197, May 2015.

[17] I. Sutskever, O. Vinyals, and Q. V. Le, “Sequence to sequence learning with neural networks,” in Proc. 28th Int. Conf. Neural Inf. Process. Syst., vol. 27. MIT Press, Dec. 2014, pp. 3104–3112.

[18] Z. Fu, Y. Wu, and X. Liu, “A tensor-based deep LSTM forecasting model capturing the intrinsic connection in multivariate time series,” Appl. Intell., vol. 53, no. 12, pp. 15 873–15 888, Nov. 2022.

[19] J.-M. Yang, Z.-R. Peng, and L. Lin, “Real-time spatiotemporal prediction and imputation of traffic status based on LSTM and graph laplacian regularized matrix factorization,” Transp. Res. C, Emerg. Technol., vol. 129, p. 103228, Aug. 2021.

[20] R. Fu, Z. Zhang, and L. Li, “Using LSTM and GRU neural network methods for traffic flow prediction,” in Proc. 31st Youth Acad. Annu. Conf. Chin. Assoc. Autom. (YAC). IEEE, Nov. 2016, pp. 324–328.

[21] J. Zhang, Y. Zheng, D. Qi, R. Li, X. Yi, and T. Li, “Predicting citywide crowd flows using deep spatio-temporal residual networks,” Artif. Intell., vol. 259, pp. 147–166, Jun. 2018.

[22] Y. Liu, Z. Liu, and R. Jia, “DeepPF: A deep learning based architecture for metro passenger flow prediction,” Transp. Res. C, Emerg. Technol., vol. 101, pp. 18–34, Apr. 2019.

[23] S. Deng, S. Jia, and J. Chen, “Exploring spatial–temporal relations via deep convolutional neural networks for traffic flow prediction with incomplete data,” Appl. Soft Comput., vol. 78, pp. 712–721, May 2019.

[24] A. Ali, I. Ullah, S. Ahmad, Z. Wu, J. Li, and X. Bai, “An attention-driven spatio-temporal deep hybrid neural networks for traffic flow prediction in transportation systems,” IEEE Trans. Intell. Transp. Syst., vol. 26, no. 9, pp. 14 154–14 168, Sep. 2025.

[25] Y. Chen and X. M. Chen, “A novel reinforced dynamic graph convolutional network model with data imputation for network-wide traffic flow prediction,” Transp. Res. C, Emerg. Technol., vol. 143, p. 103820, Oct. 2022.

[26] S. Guo, Y. Lin, N. Feng, C. Song, and H. Wan, “Attention based spatialtemporal graph convolutional networks for traffic flow forecasting,” Proc. AAAI Conf. Artif. Intell., vol. 33, no. 01, pp. 922–929, Jul. 2019.

[27] X. Huang, Y. Ye, X. Yang, and L. Xiong, “Multi-view dynamic graph convolution neural network for traffic flow prediction,” Expert Syst. Appl., vol. 222, p. 119779, Jul. 2023.

[28] H. Wang, R. Zhang, X. Cheng, and L. Yang, “Hierarchical traffic flow prediction based on spatial-temporal graph convolutional network,” IEEE Trans. Intell. Transp. Syst., vol. 23, no. 9, pp. 16 137–16 147, Sep. 2022.

[29] J. Zhu, X. Han, H. Deng, C. Tao, L. Zhao, P. Wang, T. Lin, and H. Li, “KST-GCN: A knowledge-driven spatial-temporal graph convolutional network for traffic forecasting,” IEEE Trans. Intell. Transp. Syst., vol. 23, no. 9, pp. 15 055–15 065, Sep. 2022.

[30] Y. He, Y. Jia, Y. Jia, C. An, Z. Lu, and J. Xia, “An integrated intra-view and inter-view framework for multiple traffic variable data simultaneous recovery,” IEEE Trans. Intell. Transp. Syst., vol. 25, no. 11, pp. 17 200– 17 217, Nov. 2024.

[31] S. Yang, K. Kalpakis, and A. Biem, “Detecting road traffic events by coupling multiple timeseries with a nonparametric Bayesian method,” IEEE Trans. Intell. Transp. Syst., vol. 15, no. 5, pp. 1936–1946, Oct. 2014.

[32] X. Feng, H. Zhang, C. Wang, and H. Zheng, “Traffic data recovery from corrupted and incomplete observations via spatial-temporal TRPCA,” IEEE Trans. Intell. Transp. Syst., vol. 23, no. 10, pp. 17 835–17 848, Oct. 2022.

[33] B.-Z. Li, X.-L. Zhao, X. Chen, M. Ding, and R. Wen Liu, “Convolutional low-rank tensor representation for structural missing traffic data imputation,” IEEE Trans. Intell. Transp. Syst., vol. 25, no. 11, pp. 18 847– 18 860, Nov. 2024.

[34] M. E. Kilmer and C. D. Martin, “Factorization strategies for third-order tensors,” Linear Algebra Appl., vol. 435, no. 3, pp. 641–658, Aug. 2011.

[35] M. E. Kilmer, K. Braman, N. Hao, and R. C. Hoover, “Third-order tensors as operators on matrices: A theoretical and computational framework with applications in imaging,” SIAM J. Matrix Anal. Appl., vol. 34, no. 1, pp. 148–172, Jan. 2013.

[36] R. M. Gray, Toeplitz and Circulant Matrices: A Review. Boston, MA, USA: Now, 2006.

[37] S. L. Brunton and J. N. Kutz, Data-Driven Science and Engineering: Machine Learning, Dynamical Systems, and Control. Cambridge University Press, May 2022.

[38] Q. Yu, Y.-H. Dai, and M. Bai, “Spectral-spatial extraction through layered tensor decomposition for hyperspectral anomaly detection,” SIAM J. Imaging Sci., vol. 19, no. 2, pp. 807–838, Apr. 2026.

[39] Q. Yu, Y.-H. Dai, L.-B. Cui, S. Gelareh, and M. Bai, “A unified spatiotemporal and coupled multi-view tensor framework for simultaneous traffic data reconstruction and event detection,” Transp. Res. C, Emerg. Technol., vol. 193, Dec. 2026.

[40] X. Chen, Z. Cheng, H. Cai, N. Saunier, and L. Sun, “Laplacian convolutional representation for traffic time series imputation,” IEEE Trans. Knowl. Data Eng., vol. 36, no. 11, pp. 6490–6502, Nov. 2024.

[41] L. Pan and X. Chen, “Group sparse optimization for images recovery using capped folded concave functions,” SIAM J. Imaging Sci., vol. 14, no. 1, pp. 1–25, jan 2021.

[42] D. Deng, C. Shahabi, U. Demiryurek, L. Zhu, R. Yu, and Y. Liu, “Latent space model for road networks to predict time-varying traffic,” in Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining, ser. KDD ’16. ACM, Aug. 2016, pp. 1525–1534.

[43] X. Chen, X.-L. Zhao, and C. Cheng, “Forecasting urban traffic states with sparse data using hankel temporal matrix factorization,” INFORMS J. Comput., vol. 37, no. 4, pp. 945–961, Jul. 2025.

[44] X. Wang and L. Sun, “Anti-circulant dynamic mode decomposition with sparsity-promoting for highway traffic dynamics analysis,” Transp. Res. C, Emerg. Technol., vol. 153, p. 104178, Aug. 2023.

[45] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, “A time series is worth 64 words: Long-term forecasting with transformers,” in Proc. Int. Conf. Learn. Represent., 2023.

[46] S. Zhang, Z. Liu, C. Wo, J. Qian, Y. Liu, and H. O. Gao, “Spatiotemporal graph neural network for traffic flow prediction based on a two-stage pretraining and fine-tuning framework,” Adv. Eng. Inf., vol. 73, p. 104572, Jul. 2026.

[47] Y. Xu, Z. Meng, and X. Jiang, “AmTGBiM: A Mamba-based model for long-term traffic flow prediction combining adaptive masked spatial transformer and dynamic graph attention,” J. King Saud Univ. - Comput. Inf. Sci., May 2026.