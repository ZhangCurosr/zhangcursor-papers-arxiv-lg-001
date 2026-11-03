# CONTROL FOUNDATION MODELS ACROSS CHANGING BLOCK DIAGRAMS: GENERALIZING TIME-SERIES PRE-DICTION TO DYNAMIC GRAPHS

Yuji Okamoto<sup>∗</sup>   
RIKEN   
2-1 Hirosawa, 351-0198, Japan   
yuji.0001@gmail.com   
Masaki Inoue   
Keio University,   
3-14-1 Hiyoshi, Kohoku-ku,   
Yokohama, Kanagawa 223-8522, Japan   
minoue@appi.keio.ac.jp

Kenta Hoshino Institute of Science Tokyo, 2-12-1 Ookayama, Meguro-ku, Tokyo 152-8550, Japan hoshino@comp.isct.ac.jp

## ABSTRACT

In the prediction and control of physical systems, combining records from different experiments and operations is expected to supply information that no single experiment provides on its own. Time-series foundation models, which learn a shared predictor from diverse time series, offer a framework for treating such records as a common training resource. In addition, advances in query-based prediction models, which switch between known signals and target signals within the same time series, have made it possible to handle response prediction and control-input generation in a unified manner. However, in control-experiment data, the connections between signals change with the experimental stage, such as system identification, filtering, and feedback control. Such data require a formulation that handles signal values and connection structures dynamically and in a unified way. In this study, we formulate a control foundation model. It defines a data space of dynamic graphs that contain connections, signal values, and missing values, together with an operation on this space that specifies the known signals and the prediction targets. This allows response prediction and control-input generation to be described as a common masked prediction problem, even for experimental data with different connections and recorded signals. The Block Diagram Transformer (BDT), which implements this formulation, combines attention along the time axis with a graph neural network that follows the block-diagram connections. It can use the pretrained parameters of existing time-series foundation models as initial values. Trained on control-experiment data that mix multiple experimental settings, the proposed model stabilized an unstable system through feedback control. This is difficult to achieve by fine-tuning existing time-series foundation models, and control became possible only with the introduction of BDT. Ongoing work applying these results to a wide range of target systems is also beginning to bear fruit.

## 1 INTRODUCTION

In real-world control, data on the input and output signals of a plant accumulate through identification experiments, controller tuning, and routine operation. Efforts have been made to exploit such data for new identification and controller tuning (Bittencourt et al., 2015; Campi et al., 2002). Reinforcement-learning-based control has likewise evolved toward offline reinforcement learning, which uses experience gained in past trials to learn policies (Levine et al., 2020). Data obtained from diverse experiments and operations should be treated as a common training resource and exploited for prediction, estimation, and control beyond the purposes for which they were originally acquired.

Reusing such data requires capturing the connections under which each time-series signal was generated and how it is used. In system identification, which estimates system parameters from the input and output signals of a system, it has long been emphasized that the difference in signal generation between open loop and closed loop affects the conditions for estimating the plant dynamics (Forssell & Ljung, 1999). This discussion has been extended to systems of multiple interconnected components, and further to settings in which only some of the internal signals can be measured (Van den Hof et al., 2013; Dankers et al., 2016). Formulations that use experimental data directly for identification, prediction, and control have also been developed for the experimental conditions of individual linear time-invariant systems (Markovsky, 2017; van Waarde et al., 2020). Formulations have thus been developed for each specific experimental condition and purpose. However, a unified formulation that handles experimental data collected under different conditions and the design problems of prediction and control within a single framework has yet to be established.

To address this issue, we extend the data space from time series of signal values to dynamic graphs based on block diagrams, and represent the design problems of prediction and control in a unified manner. This space consists of dynamic graphs in which each signal component is a vertex, the connections between signals are directed edges defined at each time step, and each vertex carries its signal values and whether they are missing. On this space, we define an operation that specifies the prediction targets and the known signals, and treat prediction and control as a common masked prediction problem. This allows us to construct a Control Foundation Model (CFM) that jointly learns response prediction and control-input generation from experimental data with different connections and recorded signals. CFM aims to perform prediction and control without additional training, according to the connections and purpose of each experiment.

To build a predictor based on this formulation, we leverage time-series foundation models pretrained on diverse series. Time-series foundation models have been developed as a framework for predicting new series from given histories and covariates without additional training (Ansari et al., 2025). For the inter-signal operations of this predictor, we construct the Block Diagram Transformer (BDT), which reflects the block diagram at each time step. BDT processes the history of each signal with attention along the time axis, and along the signal axis restricts the attention scope according to the block-diagram connections. By keeping the same parameter configuration as the corresponding time-series foundation model, BDT inherits its pretrained parameters and uses them for predictions that reflect the connection structure.

In experiments on an unstable linear system, we confirmed that our approach improved the prediction accuracy of control inputs compared with fine-tuning a time-series foundation model. In feedback control, we also obtained a decaying response close to that of linear-quadratic-Gaussian (LQG) control. This work unifies a data representation that preserves differences in experimental conditions with an operation that specifies the design problems of prediction and control, providing a framework for jointly learning prediction and control from diverse experimental data.

The contributions of this paper are threefold:

• Unified formulation of experimental data and prediction/control problems. We define a data space of dynamic graphs containing connections, signal values, and missing values, together with an operation on this space that specifies the intended use, thereby generalizing masked prediction on time series. This allows response prediction and control-input generation using data from different control experiments to be treated as a common learning problem.

• Architecture of the control foundation model. We construct BDT, which combines attention along the time axis with graph operations based on block diagrams. BDT inherits the pretrained parameters of the corresponding time-series foundation model while reflecting the connections at each experimental stage in its predictions.

• Demonstration on an unstable linear system. Experiments on a single unstable linear system confirmed improved prediction accuracy of control inputs and a decaying response close to that of LQG control in feedback control.

## 2 RELATED WORKS

## 2.1 GENERALIZATION OF TIME-SERIES MODELS

Time-series foundation models have been developed as a framework in which a predictor is pretrained on diverse series and applied to unseen series without additional training (Das et al., 2024; Ansari et al., 2024; Woo et al., 2024). This is achieved by keeping the pretrained parameters fixed while capturing series-specific patterns of variation from the observed history through attention along the time axis and reflecting them in the predictions. The recently proposed Chronos-2 and TimesFM-3 perform multivariate forecasting that exploits information across series by combining attention along the time axis with attention along the channel axis (Ansari et al., 2025; Jain & Sen, 2026). Time-series foundation models are thus evolving to exploit not only the temporal variation of individual series but also the relationships among series. In the control field, studies that perform response prediction and state estimation as extensions of system identification have also been proposed (Forgione et al., 2023; Busetto et al., 2024).

Meanwhile, query-based designs, which select the time steps and variables to be predicted at inference time and handle different inference tasks with the same model, have also been studied. These models, notably UniMASK and MTM, use a masked-learning framework to select the time steps and channels to be predicted and flexibly predict them (Carroll et al., 2022; Wu et al., 2023). This modeling has led to time-series prediction models that handle covariates and prediction targets in a unified manner. A an application of such models, $\mathbf { M } ^ { \mathrm { 3 } } \mathbf { P } \mathbf { C }$ performs model predictive control that combines the generation of action candidates with the prediction of their outcomes to evaluate and select candidates (Wen et al., 2025). However, these approaches model only a single system, and their generalization to unseen systems has not been demonstrated.

## 2.2 TIME-SERIES PREDICTION USING GRAPH STRUCTURES

For traffic and sensor networks, the problem of predicting future values from the connection structure and the observation history at each site has been studied, where the network is represented as a graph with observation sites as nodes and their connections as edges. For this problem, models that combine temporal feature extraction with inter-site information aggregation by graph neural networks have been proposed (Yu et al., 2018; Guo et al., 2019). More recently, research on foundation models, including OpenCity, has pretrained spatiotemporal representations that account for adjacency structures shared across multiple systems, aiming at prediction transfer and adaptation to different systems (Li et al., 2025; Zhong et al., 2025). Furthermore, query-based designs that specify the time steps and sites to be inferred and handle tasks such as prediction and imputation with a common model have also been studied (Yuan et al., 2025). However, while these models are highly flexible along the graph dimension, they have a fixed prediction length along the time dimension, and their generalization along the time dimension remains unclear.

In this work, we extend the framework of learning from time-series data of different systems and performing prediction and control with the same model to control experiments whose connection structure changes over time. We represent experimental data with different sequence lengths and numbers of signals as dynamic graphs based on block diagrams, and handle response prediction and control-input generation in a unified manner, including differences in connections and recorded signals.

## 3 BACKGROUND

Let $a _ { s } \in \mathbb { R } ^ { n _ { a } }$ denote the prediction targets and $b _ { s } \in \mathbb { R } ^ { n _ { b } }$ the covariates. We define a model that predicts the H points starting at time t from the history of the preceding L points and the known future covariates as follows:

$$
p _ { \theta } \big ( a _ { t : t + H - 1 } \ | \ a _ { t - L : t - 1 } , \ b _ { t - L : t - 1 } , \ b _ { t : t + H - 1 } \big )\tag{3.1}
$$

By pretraining the shared parameters $\theta$ on diverse series, time-series foundation models can predict future signal values from past history data in a zero-shot manner.

![](images/0eb3d6871fbf522fb06a55cdace89d55ab7b7c4e8ef9a1bbb07e90cf2f97e31a.jpg)  
Figure 1: Changes in signal connections and available data across experimental stages. The top row shows the block diagrams for identification, state estimation, stabilization, and tracking, and the bottom row shows schematic time series corresponding to each stage. P, K, and KF denote the plant, the controller, and the Kalman filter, respectively. Vertical dashed lines indicate stage boundaries, and shading indicates signal components that cannot be observed.

Recent time-series foundation models (Ansari et al., 2025) embed the time-series data $a _ { * } , b _ { * }$ and their masks patch by patch, and predict the future by combining Transformers along both the time axis and the channel axis. Let $h _ { i , t } ^ { l }$ denote the hidden state of channel i at patch position t. The attention part of a Transformer layer has the following structure:

$$
\begin{array} { r l } & { \left( h _ { i , 1 } ^ { l + 1 / 2 } , \ldots , h _ { i , L + H } ^ { l + 1 / 2 } \right) = \mathrm { T i m e A t t n } \bigl ( h _ { i , 1 } ^ { l } , \ldots , h _ { i , L + H } ^ { l } \bigr ) , \ \qquad i \in \{ 1 , \ldots , n _ { a } + n _ { b } \} , } \\ & { \left( h _ { 1 , t } ^ { l + 1 } , \ldots , h _ { n _ { a } + n _ { b } , t } ^ { l + 1 } \right) = \mathrm { G r o u p A t t n } \Bigl ( h _ { 1 , t } ^ { l + 1 / 2 } , \ldots , h _ { n _ { a } + n _ { b } , t } ^ { l + 1 / 2 } \Bigr ) , \quad t \in \{ 1 , \ldots , L + H \} . } \end{array}\tag{3.2}
$$

This mechanism makes the prediction model applicable to variable dimensions of the prediction targets and covariates. In what follows, we write $T \triangleq L + H$ for the combined length of the input (history) range and the prediction range, and $n \triangleq n _ { a } + n _ { b }$ for the combined dimension of the prediction targets and covariates.

The key question here is what the prediction targets are and what the covariates are in control engineering. Prediction and control-input generation in control can be viewed uniformly as conditional prediction, which infers unknown signals from known signals. Treating the input as known and inferring the observations yields response prediction, whereas treating the observation history and the reference values as known and inferring the input yields control-input generation. In other words, instead of describing the two with separate input-output formats, both can be expressed by switching between known components and prediction targets on a common time series (Wu et al., 2023). To express this unification, we collect the time series of all signals into $\boldsymbol { x } \in \mathbb { R } ^ { T \times n }$ , and let M denote the index set representing the query (specification of prediction targets) and $M ^ { C }$ its complement, which specifies the known components. Prediction and control are described as different settings of M under the same conditional prediction model and training objective:

$$
\begin{array} { r l } { \mathrm { P r e d i c t i o n ~ m o d e l : } \quad p _ { \theta } ( x _ { M } \mid x _ { M ^ { c } } ) , } & { } \\ { \mathrm { L o s s ~ f u n c t i o n : } \quad \mathrm { L o s s } ( \theta ) \triangleq \mathbb { E } _ { ( x , M ) \sim \mathcal { D } } \left[ - \log p _ { \theta } ( x _ { M } \mid x _ { M ^ { c } } ) \right] . } \end{array}\tag{3.3}
$$

Here, $M \subseteq \{ 1 , \dots , T \} \times \{ 1 , \dots , n \}$ , and $x _ { M }$ and $x _ { M } c$ denote the prediction targets and the known components, respectively. D is the training distribution over pairs of a time series x and a prediction-target specification M.

![](images/7bb9f14a7ba2f93a369bdee325855b6fe60f6d7d4df311a17e69c25e778e00ef.jpg)  
Figure 2: Unified representation of control time-series data using block diagrams, and the control foundation model.

In control-system design, however, the results of earlier stages are carried over to later ones: the plant dynamics are identified, unobservable states are estimated, and this information is then used for stabilization and tracking (Fig. 1). In this process, not only the prediction targets but also the mechanisms that process the signals and their connections change. For example, the input u during identification is applied externally without using the observation y, whereas it is determined using the estimated state xˆ obtained from the observations in stabilization, and additionally the reference signal r in tracking. The connections between these signals and the mechanisms that process them are represented by block diagrams (top row of Fig. 1). In this work, we treat the connections at each experimental stage as block diagrams together with the time-series data, and formulate a foundation model for prediction and control across different stages.

## 4 METHOD

## 4.1 UNIFYING CONTROL TIME-SERIES DATA WITH BLOCK DIAGRAMS

We formulate the block diagram described in the previous section as a graph that represents the connections between signals. Each signal component is a vertex, and the connection through which one signal is fed into a mechanism that generates another signal is represented as a directed edge. We associate the signal values at each time step with the vertices and describe changes in the connection structure across experimental stages as changes in the edge set. We thereby define control time-series data, which treat block diagrams and signal values jointly, as the following dynamic graph.

Definition 1 (Control time-series data with block diagrams). We define control time-series data, comprising the connection structure between signals, the signal values, and whether each value is missing, as the following dynamic graph:

$$
\mathcal { G } = ( V , E _ { t } , X _ { t } ) _ { t = 1 } ^ { T } , \qquad E _ { t } \subseteq V \times V , \qquad X _ { t } = ( x _ { t } , \eta _ { t } ) \in \mathbb { R } ^ { | V | } \times \{ 0 , 1 \} ^ { | V | } .\tag{4.1}
$$

Here, $V$ is the set of vertices corresponding to the signal components, and $E _ { t }$ is the set that represents each connection from signal i to signal j as a directed edge $( i , j ) . x _ { t }$ denotes the signal values and $\eta _ { t }$ the missingness mask; $\eta _ { t , i } = 1$ indicates that the value is recorded, and $\eta _ { t , i } = 0$ indicates that it is missing.

This formulation allows experimental stages that differ in connection structure and recorded signals to be represented over a common vertex set V. Open-loop identification has the connection $u  y ,$ whereas tracking control based on state estimation additionally has $u , y  \hat { x }$ , corresponding to the inputs to the state estimator, and $r , { \hat { x } }  u ,$ , corresponding to the inputs to the controller (top row of Fig. 1). Here, a connection between vector signals denotes the collection of edges between their corresponding components. The estimated state xˆ and the reference signal r, which are not recorded during the identification stage, retain their vertices and are represented by setting the missingness mask to $\eta _ { t , i } = 0$ (bottom row of Fig. 1). In this way, differences in connection structure and missing signals are represented by $E _ { t }$ and $\eta _ { t }$ , respectively, so that the data from all stages are handled in a common format. Below, we define the prediction model and the training objective for the control time-series data $\mathcal { G } .$

## 4.2 PREDICTION MODEL AND LOSS FUNCTION WITH BLOCK DIAGRAMS

With this unified representation of control time-series data, prediction problems are unified in a form that accounts for experimental settings that time-series foundation models could not represent. As with time-series data, we denote the query that specifies the prediction targets by $M \subset \{ 1 , \cdot \cdot \cdot , T \} \times V$ and define the unified prediction model and its loss function as follows:

$$
\begin{array} { r l } { \mathrm { P r e d i c t i o n ~ m o d e l : } \quad } & { p _ { \theta } ( \mathcal { G } _ { M } \mid \mathcal { G } _ { M ^ { c } } ) , } \\ { \mathrm { L o s s ~ f u n c t i o n : } \quad } & { \mathrm { L o s s } ( \theta ) \triangleq \mathbb { E } _ { ( \mathcal { G } , M ) \sim \mathcal { D } } \left[ - \log p _ { \theta } ( \mathcal { G } _ { M } \mid \mathcal { G } _ { M ^ { c } } ) \right] . } \end{array}\tag{4.2}
$$

Here, $\mathcal { G } _ { M }$ and $\mathcal { G } _ { M ^ { C } }$ denote data that retain the vertex set and the connection information at each time step and contain the signal values corresponding to M and $M ^ { C }$ , respectively. The prediction targets are the signal values corresponding to ${ \bar { M } } ;$ the known signal values in $M ^ { C }$ and the block diagram are given as the conditions for prediction.

By defining masking operations and predictive distributions on control time-series data that include the connection structure, this formulation extends the unified treatment of prediction and control to experimental stages with different connection structures. We refer to a model that pretrains shared parameters on diverse control time-series data and addresses prediction and control problems within this framework as a control foundation model. Conventional time-series prediction is included as a special case of the control foundation model.

Theorem 1. If the connection structure and the time-series variables are independent, Eq. (4.2) reduces to Eq. (3.3).

Proof. In this proof, we omit the notation for the missingness indicator, which is common to both models. By the independence of the time-series variables and the connection structure,

$$
\begin{array} { r l } { p _ { \theta } ( \mathcal { G } _ { M } \mid \mathcal { G } _ { M ^ { c } } ) = \displaystyle \frac { p _ { \theta } ( x , V , E _ { 1 : T } ) } { p _ { \theta } ( x _ { M ^ { c } } , V , E _ { 1 : T } ) } } \\ { ~ } & { = \displaystyle \frac { p _ { \theta } ( x ) p _ { \theta } ( V , E _ { 1 : T } ) } { p _ { \theta } ( x _ { M ^ { c } } ) p _ { \theta } ( V , E _ { 1 : T } ) } } \\ { ~ } & { = p _ { \theta } ( x _ { M } \mid x _ { M ^ { c } } ) . } \end{array}
$$

Therefore, Eq. (4.2) reduces to Eq. (3.3).

## 4.3 ARCHITECTURE OF THE BLOCK DIAGRAM TRANSFORMER

We propose the Block Diagram Transformer (BDT) as a concrete architecture that realizes the control foundation model. BDT processes the history of each signal with attention along the time axis and, along the channel axis, uses a graph neural network (GNN) based on the block diagram at each time step. By replacing the Group Attention in Eq. (3.2) with this graph operation, BDT reflects the connection structure in the information exchange between signals.

$$
\begin{array} { r l } & { \left( h _ { i , 1 } ^ { l + 1 / 2 } , \ldots , h _ { i , T } ^ { l + 1 / 2 } \right) = \mathrm { T i m e A t t n } \bigl ( h _ { i , 1 } ^ { l } , \ldots , h _ { i , T } ^ { l } \bigr ) , \qquad i \in \{ 1 , \ldots , | V | \} , } \\ & { \quad \left( h _ { 1 , t } ^ { l + 1 } , \ldots , h _ { | V | , t } ^ { l + 1 } \right) = \mathrm { G N N } \Bigl ( h _ { 1 , t } ^ { l + 1 / 2 } , \ldots , h _ { | V | , t } ^ { l + 1 / 2 } \ | \ ( V , E _ { t } ) \Bigr ) , \quad t \in \{ 1 , \ldots , T \} . } \end{array}\tag{4.3}
$$

This architecture includes the attention structure of conventional time-series foundation models as a special case.

Theorem 2. Let the GNN in $E q .$ (4.3) be a Graph Transformer that uses the same parameters as the Group Attention in Eq. (3.2), with only an attention-scope restriction by a graph mask added. Then, Eq. (3.2) is a special case of $E q .$ . (4.3) in which the graph $( V , E _ { t } )$ at every time step is the complete graph.

Proof. We define the graph mask corresponding to the connection structure at time t as

$$
( B _ { t } ) _ { i j } = { \left\{ \begin{array} { l l } { 0 , } & { ( j , i ) \in E _ { t } { \mathrm { ~ o r ~ } } i = j , } \\ { - \infty , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }
$$

Letting H be the matrix whose rows are the hidden states $h _ { i } ^ { \top }$ , the operation of each head is given by

$$
\mathrm { G r a p h A t t n } ( H ; B _ { t } ) = \mathrm { s o f t m a x } _ { \mathrm { r o w } } \left( \frac { ( H W _ { Q } ) ( H W _ { K } ) ^ { \top } } { \sqrt { d _ { k } } } + B _ { t } \right) H W _ { V } ,
$$

where $W _ { Q } , W _ { K } , W _ { V }$ are weight matrices and $d _ { k }$ is the key dimension. For the complete graph, $B _ { t } = 0$ at all time steps, so the operation of each head coincides with that of the corresponding head of Group Attention. Because the operations other than the attention-scope restriction are also shared, the entire operation along the channel axis coincides with Group Attention. Since the attention along the time axis is shared as well, Eq. (4.3) coincides with Eq. (3.2). □

In this architecture, Graph Attention is an operation that modifies only the attention mask of Group Attention, and the parameter structure is shared. Therefore, the pretrained parameters of the corresponding time-series foundation model can be transferred directly to BDT.

In the following, we use this improved model to conduct numerical experiments on prediction problems specific to control engineering.

## 5 RESULTS

Even for the same input and output signals, the input is applied externally during identification but is generated from observations during control. Using trajectories that span identification through control of a single unstable linear system, we evaluated response prediction, input prediction, and feedback control with the predicted inputs. We compared pretrained Chronos-2, its fine-tuned version, and CFM implemented with BDT. CFM was given the known block diagrams corresponding to the identification and control segments, together with the switching time.

Fine-tuned Chronos-2 and CFM started from the same pretrained weights. To preserve the relative scale of inputs and outputs in the model inputs, both were trained without per-channel input/output normalization. Below, we report the CFM configuration that achieved the smallest root mean square (RMS) of the closed-loop output on the test trajectories, and fine-tuned Chronos-2 with the matched learning rate and maximum number of updates. These reported configurations were selected after inspecting the test results. The experimental and training procedures and the results for all search configurations are summarized in Appendix A.

## 5.1 ACCURACY OF INPUT AND OUTPUT PREDICTION

We predicted future outputs or inputs from the histories of the stored test trajectories. For response prediction, the known future inputs were also given; for input prediction, only past inputs and outputs were given. CFM handled both tasks by changing the specification of known components and prediction targets.

In output prediction, fine-tuned Chronos-2 and CFM achieved nearly the same accuracy, whereas in input prediction, CFM achieved a smaller error (Table 1). The two models, which are close in response-prediction accuracy, differed when inferring the inputs.

Table 1: Multi-step prediction performance. Mean ± sample standard deviation of the per-trajectory root-meansquare error (RMSE) over 100 test trajectories. The input-prediction statistics include both the identification segment with the M-sequence and the LQG control segment.
<table><tr><td>Method</td><td>Output-prediction RMSE</td><td>Input-prediction RMSE</td></tr><tr><td>Chronos-2</td><td>0.4235 ± 0.2507</td><td> $1 . 8 8 1 4 \pm 0 . 8 0 0 8$ </td></tr><tr><td>fine-tuned Chronos-2</td><td>0.0289 ± 0.0042</td><td> $1 . 6 7 9 3 \pm 0 . 7 3 3 3$ </td></tr><tr><td>CFM</td><td> $0 . 0 2 9 0 \pm 0 . 0 0 3 5$ </td><td> $\mathbf { 0 . 7 7 7 2 } \pm \mathbf { 0 . 0 2 7 1 }$ </td></tr></table>

![](images/b9f5ae755898da7ec3198cc8dae170d679da7caf7517c6676673b67831c1ce6d.jpg)  
Figure 3: Feedback control response with predicted inputs. Output y (top) and input u (bottom) of a representative trajectory. Shading indicates the identification segment. LQG is a reference that uses the known model and the observation at the current time step, whereas the learned models use observations up to the previous time step. The displayed range is limited to compare the decaying responses, but the evaluation metrics use all time points, including those outside this range. Full-range plots are shown in Appendix A.

## 5.2 FEEDBACK CONTROL WITH PREDICTED INPUTS

Next, we performed feedback control that drives the system to the origin. At each time step, the model predicted an input sequence, and its first element was applied to the plant. Because the resulting observation is appended to the history before the next prediction, each model uses, after control starts, a history generated by its own inputs.

On a representative trajectory starting from the same identification history and the same state at the start of control, CFM exhibited a decaying response close to the LQG reference, whereas fine-tuned Chronos-2 retained large-amplitude oscillations (Fig. 3). Across all six configurations with matched learning rates and maximum numbers of updates, CFM also achieved a smaller closed-loop output RMS aggregated over all test trajectories (Table A1 in Appendix A).

These results indicate that reflecting the stage-dependent connections in prediction is effective for extending time-series foundation models from response prediction to control-input generation.

## 6 CONCLUSION

We formulated prediction and control as a common prediction problem by introducing dynamic graphs that jointly represent the connections between signals, signal values, and missingness, together with an operation that specifies known signals and prediction targets. This formulation provides a foundation for treating control experiments with different connections and recorded signals as a common training resource. Its concrete architecture, the Block Diagram Transformer (BDT), inherits the pretrained parameters of a time-series foundation model while incorporating block-diagram connections into cross-series attention.

In experiments on a single unstable linear system, the proposed model reduced the control-input prediction error and the closed-loop output RMS while maintaining response-prediction accuracy comparable to that of a fine-tuned time-series foundation model. The significance of this result is most evident in closed-loop control simulations: whereas the closed-loop system with the existing model failed to stabilize the unstable plant, the closed-loop system with the proposed method achieved stabilization. These results indicate that handling stage-dependent connection structures is effective for extending time-series prediction to control-input generation.

This work has already been extended to unseen system instances, multi-input multi-output (MIMO) systems, and joint learning from linear systems and rigid-body arms, and we are evaluating its generalization across different systems and signal dimensions. These studies examine to what extent learning from multiple experiments can compensate for information lacking in individual experiments, and the range over which response prediction, input prediction, and closed-loop control can be performed without additional training. Our framework aims to reuse control-experiment records beyond the purposes for which they were acquired and to learn prediction and control jointly.

## ACKNOWLEDGMENTS

This work was supported by JST Moonshot R&D (JPMJMS2021), JST Research and Development Program for Next-generation Edge AI Semiconductors (JPMJES2511), JSPS KAKENHI (JP25K00148, JP25K01254, JP25H02626, JP26K14994, JP23K03902), a project (JPNP14004) com missioned by the New Energy and Industrial Technology Development Organization (NEDO), and RIKEN TRIP initiative (AGIS).

## REFERENCES

Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, Jasper Zschiegner, Danielle C. Maddix, Hao Wang, Michael W. Mahoney, Kari Torkkola, Andrew Gordon Wilson, Michael Bohlke-Schneider, and Yuyang Wang. Chronos: Learning the language of time series. Transactions on Machine Learning Research, 2024. ISSN 2835-8856.

Abdul Fatir Ansari, Oleksandr Shchur, Jaris Küken, Andreas Auer, Boran Han, Pedro Mercado, Syama Sundar Rangapuram, Huibin Shen, Lorenzo Stella, Xiyuan Zhang, Mononito Goswami, Shubham Kapoor, Danielle C. Maddix, Pablo Guerron, Tony Hu, Junming Yin, Nick Erickson, Prateek Mutalik Desai, Hao Wang, Huzefa Rangwala, George Karypis, Yuyang Wang, and Michael Bohlke-Schneider. Chronos-2: From univariate to universal forecasting. Technical report, Amazon Web Services, 2025.

André C. Bittencourt, Alf J. Isaksson, Daniel Peretzki, and Krister Forsman. An algorithm for finding process identification intervals from normal operating data. Processes, 3(2):357–383, 2015.

Riccardo Busetto, Valentina Breschi, Marco Forgione, Dario Piga, and Simone Formentin. In-context learning of state estimators. IFAC-PapersOnLine, 58(15):145–150, 2024.

M. C. Campi, A. Lecchini, and S. M. Savaresi. Virtual reference feedback tuning: A direct method for the design of feedback controllers. Automatica, 38(8):1337–1346, 2002.

Micah Carroll, Orr Paradise, Jessy Lin, Raluca Georgescu, Mingfei Sun, David Bignell, Stephanie Milani, Katja Hofmann, Matthew Hausknecht, Anca Dragan, and Sam Devlin. Uni[MASK]: Unified inference in sequential decision problems. In Advances in Neural Information Processing Systems, volume 35, pp. 35365–35378, 2022.

Arne Dankers, Paul M. J. Van den Hof, Xavier Bombois, and Peter S. C. Heuberger. Identification of dynamic models in complex networks with prediction error methods: Predictor input selection. IEEE Transactions on Automatic Control, 61(4):937–952, 2016.

Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou. A decoder-only foundation model for time-series forecasting. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 10148–10167. PMLR, 2024.

Marco Forgione, Filippo Pura, and Dario Piga. From system models to class models: An in-context learning paradigm. IEEE Control Systems Letters, 7:3513–3518, 2023.

Urban Forssell and Lennart Ljung. Closed-loop identification revisited. Automatica, 35(7):1215–1241, 1999.

Shengnan Guo, Youfang Lin, Ning Feng, Chao Song, and Huaiyu Wan. Attention based spatial-temporal graph convolutional networks for traffic flow forecasting. Proceedings ofthe AAAI Conference on Artificial Intelligence, 33(01):922–929, 2019.

Ayush Jain and Rajat Sen. TimesFM-3: A zero-shot foundation model for multivariate forecasting. Google Research blog, August 2026.

Sergey Levine, Aviral Kumar, George Tucker, and Justin Fu. Offline reinforcement learning: Tutorial, review, and perspectives on open problems. arXiv preprint arXiv:2005.01643, 2020.

Zhonghang Li, Long Xia, Lei Shi, Yong Xu, Dawei Yin, and Chao Huang. Open spatio-temporal foundation models for traffic prediction. ACM Transactions on Intelligent Systems and Technology, art. 3773912, October 2025.

Ivan Markovsky. A missing data approach to data-driven filtering and control. IEEE Transactions on Automatic Control, 62(4):1972–1978, 2017.

Paul M. J. Van den Hof, Arne Dankers, Peter S. C. Heuberger, and Xavier Bombois. Identification of dynamic models in complex networks with prediction error methods—basic methods for consistent module estimates. Automatica, 49(10):2994–3006, 2013.

Henk J. van Waarde, Jaap Eising, Harry L. Trentelman, and M. Kanat Camlibel. Data informativity: A new perspective on data-driven analysis and control. IEEE Transactions on Automatic Control, 65(11):4753–4768, 2020.

Kehan Wen, Yutong Hu, Yao Mu, and Lei Ke. M<sup>3</sup>P C: Test-time model predictive control for pretrained masked trajectory model. In The Thirteenth International Conference on Learning Representations, 2025.

Gerald Woo, Chenghao Liu, Akshat Kumar, Caiming Xiong, Silvio Savarese, and Doyen Sahoo. Unified training of universal time series forecasting transformers. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pp. 53140–53164. PMLR, 2024.

Philipp Wu, Arjun Majumdar, Kevin Stone, Yixin Lin, Igor Mordatch, Pieter Abbeel, and Aravind Rajeswaran. Masked trajectory models for prediction, representation, and control. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pp. 37607– 37623. PMLR, 2023.

Bing Yu, Haoteng Yin, and Zhanxing Zhu. Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting. In Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence, IJCAI-18, pp. 3634–3640. International Joint Conferences on Artificial Intelligence Organization, 2018.

Yuan Yuan, Chonghua Han, Jingtao Ding, Guozhen Zhang, Depeng Jin, and Yong Li. Diffusion transformers as open-world spatiotemporal foundation models. In Advances in Neural Information Processing Systems, volume 38, 2025.

Siru Zhong, Junjie Qiu, Yangyu Wu, Xingchen Zou, Zhongwen Rao, Bin Yang, Chenjuan Guo, Hao Xu, and Yuxuan Liang. Learning to factorize spatio-temporal foundation models. In Advances in Neural Information Processing Systems, volume 38, 2025.

## A DETAILS OF EXPERIMENTS AND EVALUATION ON A SINGLE UNSTABLE SYSTEM

## A.1 EXPERIMENTAL SYSTEM, TRAINING, AND EVALUATION PROCEDURE

We generated trajectories from a single unstable linear system and split them into training, validation, and test sets at the trajectory level.

The state and observation equations are

$$
x _ { t + 1 } = A _ { d } x _ { t } + B _ { d } u _ { t } + w _ { t } , \qquad y _ { t } = C x _ { t } + D u _ { t } + v _ { t }\tag{A.1}
$$

with state dimension $^ { 3 , }$ one input and one output, and $D = 0$ . The continuous-time system was discretized by zero-order hold with step size $\Delta t = 0 . 1$ . The fixed system ID is 967; the largest real part of the eigenvalues of the continuous-time matrix is approximately +0.0744768, and the spectral radius of the discrete-time matrix is approximately 1.0074755. The plant is open-loop unstable.

All trajectories use the same $A _ { d } , B _ { d } , C , D ,$ , while the initial state, excitation phase, and noise vary.

Item Setting   
Number of trajectories and 1000 trajectories: train 800, validation 100, test 100. The stored ID   
split assignment is used   
Initial state $x _ { 0 } \sim \mathcal { N } ( 0 , I _ { 3 } )$   
Process noise $w _ { t } \sim \mathcal { N } ( 0 , 1 0 ^ { - 4 } I _ { 3 } )$   
Observation noise $v _ { t } \sim \mathcal { N } ( 0 , 1 0 ^ { - 4 } )$   
Train/validation length 512 points, $t = \dot { 0 } , \dots , 5 1 1$   
Stored test length 768 points, $t = 0 , \ldots , 7 6 7$   
Identification input 9-stage M-sequence with amplitude $\pm 1 .$ period 511, SciPy   
max\_len\_seq $( 9 , \tt t a p s = [ 5 ] )$ , phase varied per trajectory   
Switching time for Selected per trajectory from {16, 32, . . . , 496}   
train/validation   
Switching time for test 256 for all trajectories   
System seed 2026091801, fixed ID 967   
Trajectory seed SeedSequence generated from 2026091802 and the trajectory ID   
Switch seed 2026091702   
M-sequence phase seed 2026091703 and the trajectory ID   
Test extension noise seed 2026091704 and the trajectory ID

The teacher trajectories for training are excited by the M-sequence before the switch and controlled by known-model LQG after the switch. The LQR weights are $Q = I _ { 3 } , R = 1$ , and the steady-state Kalman filter uses the noise covariances above. The teacher LQG determines $u _ { t }$ from the estimated state that incorporates the current observation $y _ { t } .$ . In contrast, closed-loop control by the learned models determines $u _ { t }$ from observations up to $t - 1$ . Known-model $\mathrm { L Q G }$ and the learned models therefore differ in both model knowledge and observation timing.

The storage order is $( u , y )$ . Simulation arrays are in double precision, and the training signals stored in Arrow are float32. The stored NPZ files hold the matrix values and noise sequences in double precision, and the manifest holds the split information.

The matrices of the discrete-time plant used are shown below (displayed values are rounded).

$$
A _ { d } = \left[ \begin{array} { c c c } { 1 . 0 1 4 2 2 7 2 2 5 2 } & { 0 . 0 6 1 5 0 7 9 3 5 0 } & { - 0 . 0 5 5 9 7 1 9 6 3 8 } \\ { - 0 . 0 1 9 6 1 6 8 8 8 7 } & { 0 . 8 9 8 7 2 8 6 3 1 9 } & { - 0 . 0 6 4 3 0 0 2 0 9 5 } \\ { 0 . 0 6 8 5 9 8 5 8 4 2 } & { 0 . 0 4 5 0 8 6 7 2 6 9 } & { 1 . 0 0 8 1 2 4 4 9 0 0 } \end{array} \right] , \qquad B _ { d } = \left[ \begin{array} { c c c } { - 0 . 0 0 3 5 5 6 9 5 7 3 } \\ { 0 . 0 8 5 2 4 6 2 5 5 7 } \\ { - 0 . 0 2 2 9 0 1 2 6 7 2 } \end{array} \right] ,\tag{A.2}
$$

$$
C = [ 0 . 2 7 3 2 3 4 4 4 0 8 \quad 0 . 0 9 0 6 5 5 1 2 0 2 \quad - 0 . 5 4 0 3 8 6 8 5 9 1 ] , \qquad D = 0 .\tag{A.3}
$$

## A.1.1 PREDICTION TASKS AND CONNECTION STRUCTURE

For each origin c, the observed history $u _ { 0 : c - 1 } , y _ { 0 : c - 1 }$ is given as input, and 16 points are predicted. We use the expanding history up to the origin rather than a fixed-length sliding window. Short histories within a batch are left-padded with NaN, and only the valid part is passed to the model.

<table><tr><td>Task</td><td>Known information</td><td>Prediction target</td></tr><tr><td>y prediction</td><td>Past u/y and future input  $u _ { c : c + 1 5 }$ </td><td> $y _ { c : c + 1 5 }$ </td></tr><tr><td>u prediction</td><td>Past u/y only</td><td> $u _ { c : c + 1 5 }$ </td></tr></table>

y prediction, which uses future u, and u prediction, which uses no future covariates, are trained in the same batch. Series other than the prediction targets are masked out of the loss. Because u prediction also includes the M-sequence input in the identification segment, the u errors in the table should not be interpreted as input errors in the control segment alone.

Fine-tuned Chronos-2 and CFM use the same Chronos-2 initial weights and parameter configuration, and all parameters are fine-tuned. Fine-tuned Chronos-2 allows attention between series within a group. CFM applies an inter-series attention mask corresponding to the known block diagram, allowing the path u→y in the identification segment and u↔y in the control segment. Self-paths are retained. The switching time is given in the data; this is not an experiment in which the model estimates the switching time.

In the implementation, y→u is allowed for a history patch once its end exceeds the switching time, and for a prediction patch when the prediction origin is at or after the switching time. Inter-series connections at the REG token positions are allowed. Hence, the mask is not simply a uniform static adjacency matrix but depends on the patch position and the known switching time.

## A.1.2 COMMON TRAINING SETTINGS

Each run started from the original Chronos-2 weights. The 6000-step and 12000-step runs are also independent runs; the latter is not a continuation from the 6000-step checkpoint.

<table><tr><td>Item</td><td>Common setting</td></tr><tr><td>Backbone Input patch / stride / output 16 / 16 / 16</td><td>Chronos-2, 12 layers, 12 heads, hidden size 768, FFN 3072</td></tr><tr><td>patch</td><td></td></tr><tr><td>Prediction length</td><td>16, with REG token 21 1evels: 0.01, 0.05, 0.10, 0.15, . . ., 0.95, 0.99</td></tr><tr><td>Quantiles Point prediction</td><td>Quantile 0.5</td></tr><tr><td>Train origins</td><td>64, 80, . . . , 496: 28 origins, 800×28=22,400 windows</td></tr><tr><td>Validation origins</td><td>128,256,384,496:100×4=400 windows</td></tr><tr><td>Optimizer</td><td>AdamW, weight decay 0, default betas=(0.9,0.999),</td></tr><tr><td>LR schedule</td><td> $\mathrm { e p s } { = } 1 0 ^ { - 8 }$  cosine, warmup 25 updates, minimum  $\mathrm { L R { = } 2 \times 1 0 ^ { - 6 } }$ </td></tr><tr><td>Gradient clipping</td><td>norm 1.0</td></tr><tr><td></td><td>Dropout / attention dropout 0 / 0.1; eval mode during evaluation</td></tr><tr><td>tion</td><td>Batch / gradient accumula- Current batch 16, accumulation 1</td></tr><tr><td>Validation frequency</td><td>Every 100 updates</td></tr><tr><td>Checkpoint selection</td><td>Minimum val/target_loss within the run</td></tr><tr><td>Precision and resources</td><td>float32, a single GH200 GPU per run</td></tr><tr><td>rent grid</td><td>Training seed for the cur- 42, shared across all conditions</td></tr></table>

The training loss is the pinball loss over the 21 quantiles. The median is used for point prediction.

## A.1.3 EVALUATION PROCEDURE AND PERFORMANCE METRICS

Evaluation conditions Let N be the number of evaluation trajectories, $n _ { x } , n _ { u } , n _ { y }$ the state, input, and output dimensions, and ∆t the sampling interval. Let $T _ { \mathrm { i d } } ^ { }$ be the length of the identification segment, $T _ { \mathrm { c t r l } }$ the length of the control evaluation segment, and $T _ { \mathrm { a l l } } = T _ { \mathrm { i d } } + T _ { \mathrm { c t r l } }$ the stored trajectory length. Let H be the prediction length, δ the re-prediction interval, $T _ { \mathrm { p r e d } }$ the trajectory length used for prediction evaluation, and C the set of prediction origins.

Prediction evaluation (Table 1 in the main text) At each origin of the stored teacher trajectories, the measured history is given and the future output or input is predicted. Histories generated by the model itself under closed-loop control are not used. The evaluation of input prediction includes both the identification segment with the M-sequence and the LQG control segment.

For task $q \in \{ y , u \}$ and trajectory $i ,$ the RMSE and its mean and sample standard deviation across trajectories are defined as follows.

$$
r _ { i , q } = \sqrt { \frac { 1 } { | \mathcal { C } | H } \sum _ { c \in \mathcal { C } } \sum _ { h = 0 } ^ { H - 1 } ( \hat { q } _ { c + h | c } ^ { ( i ) } - q _ { c + h } ^ { ( i ) } ) ^ { 2 } } ,\tag{A.4}
$$

$$
\bar { r } _ { q } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } r _ { i , q } , \qquad s _ { q } = \sqrt { \frac { 1 } { N - 1 } \sum _ { i = 1 } ^ { N } ( r _ { i , q } - \bar { r } _ { q } ) ^ { 2 } } .\tag{A.5}
$$

Table 1 in the main text reports $\bar { r } _ { q } \pm s _ { q }$

Closed-loop evaluation (Fig. 3 in the main text) The same plant, identification history, state at the start of control, and noise sequences are used for every method. The true state is used only to initialize the simulator and is not given to the learned models. At each control time step, the future input sequence is predicted from past inputs and outputs, and the first input is applied to the environment. The resulting observation is appended to the history, and re-prediction is performed at the next time step. After control starts, each model uses the history produced by its own control. The objective is regulation to the origin. The design and observation timing of the LQG reference are described in "Experimental system, training, and evaluation procedure" (Section A.1).

The magnitudes of the output and input are evaluated by the RMS aggregated over all trajectories, all control time steps, and all components.

$$
J _ { q } = \left[ \frac { 1 } { N T _ { \mathrm { c t r l } } n _ { q } } \sum _ { i = 1 } ^ { N } \sum _ { t = T _ { \mathrm { i d } } } ^ { T _ { \mathrm { a l l } } - 1 } \lVert q _ { t } ^ { ( i ) } \rVert _ { 2 } ^ { 2 } \right] ^ { 1 / 2 } , \qquad q \in \{ y , u \} .\tag{A.6}
$$

As an auxiliary metric, we define the deviation from the corresponding LQG reference.

$$
E _ { q } = \left[ \frac { 1 } { N T _ { \mathrm { c t r l } } n _ { q } } \sum _ { i = 1 } ^ { N } \sum _ { t = T _ { \mathrm { i d } } } ^ { T _ { \mathrm { a l l } } - 1 } \Vert q _ { t } ^ { ( i ) } - q _ { t , \mathrm { L Q G } } ^ { ( i ) } \Vert _ { 2 } ^ { 2 } \right] ^ { 1 / 2 } .\tag{A.7}
$$

$J _ { q }$ represents the magnitude relative to the origin, and $E _ { q }$ the difference from the LQG reference. In both cases, squared values are aggregated over all trajectories before taking the square root.

## Specific values in this experiment

<table><tr><td>Variable / setting</td><td>Value</td></tr><tr><td> $N$ </td><td>100</td></tr><tr><td> $( n _ { x } , n _ { u } , n _ { y } )$ </td><td>(3, 1, 1)</td></tr><tr><td> $\Delta t$ </td><td>0.1</td></tr><tr><td> $T _ { \mathrm { p r e d } }$ </td><td>512</td></tr><tr><td> $\mathcal { C }$ </td><td>{64, 80, . . . , 496} (28 origins)</td></tr><tr><td> $H$ </td><td>16</td></tr><tr><td></td><td>Prediction evaluation range t = 64, . . . , 511 of each trajectory (448 points)</td></tr><tr><td> $T _ { \mathrm { i d } }$ </td><td>256</td></tr><tr><td> $T _ { \mathrm { c t r l } }$ </td><td>512</td></tr><tr><td> $T _ { \mathrm { a l l } }$ </td><td>768</td></tr><tr><td> $\delta$ </td><td>1</td></tr><tr><td>Trajectory in Fig. 3 of the Trajectory ID 1 main text</td><td></td></tr></table>

The prediction length H was fixed across all conditions. No saturation was applied to the control input.

## A.2 REASONS FOR REMOVING PER-CHANNEL NORMALIZATION

If the input and output are standardized independently for each history, the relative scale between the signals can no longer be recovered from the standardized signal values alone. Let $\mu _ { u } , \mu _ { y }$ be the means of the respective series over the same observed history segment and $s _ { u } , s _ { y }$ their positive standard deviations. Then

$$
\tilde { u } = \frac { u - \mu _ { u } } { s _ { u } } , \qquad \tilde { y } = \frac { y - \mu _ { y } } { s _ { y } }
$$

holds. As a simple example, the proportional relation $u = k y$ becomes, after standardization,

$$
\tilde { u } = k \frac { s _ { y } } { s _ { u } } \tilde { y } + \frac { k \mu _ { y } - \mu _ { u } } { s _ { u } } .
$$

When non-constant series exactly satisfy $u = k y , k > 0$ over the same history segment, $\mu _ { u } = k \mu _ { y }$ and $s _ { u } = k s _ { y }$ imply

$$
\tilde { u } = \tilde { y } .
$$

In this case, the standardized signal values coincide for different positive gains, so the magnitude of the original proportionality coefficient cannot be distinguished from those values alone.

In this experiment, to preserve the relative scale between the raw signals in the model input, we replaced the input/output processing of fine-tuned Chronos-2 and CFM with the identity map. The removed processing consists of per-history, per-series standardization, the asinh transform, and the inverse transform of the output; the LayerNorm layers inside the Transformer were retained. Chronos-2 vanilla used the original input/output processing.

The above is the rationale for the choice of preprocessing; it is not a claim that a full model, including reintroduction of statistics or inverse transformation of the output, cannot handle proportional relations. In addition, since vanilla and the fine-tuned versions differ simultaneously in whether they were trained and in preprocessing, we do not interpret their performance difference as the effect of removing normalization alone.

## A.3 RESULTS BY TRAINING CONFIGURATION

## A.3.1 SEARCH SPACE AND SELECTION PROCEDURE

Fine-tuned Chronos-2 and CFM were trained independently over the same search space. Each run started from the original pretrained weights, and the common settings follow "Prediction tasks and connection structure" (Section A.1).

<table><tr><td>Item</td><td>Search space</td></tr><tr><td>Learning rate Maximum number of up- {6000, 12000}</td><td> $\{ 3 \times 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , 3 \times 1 0 ^ { - 4 } \}$ </td></tr><tr><td>dates</td><td></td></tr><tr><td>Batch size</td><td>16 (fixed)</td></tr><tr><td>Number of configurations 6 per model, 12 in total</td><td></td></tr></table>

Let A be the set of CFM configurations, $\kappa _ { a }$ the set of saved updates for configuration a, and $\theta _ { a , k }$ the weights at update k. The checkpoint of each run was selected by validation loss, and the adopted configuration was selected by its closed-loop output RMS.

$$
k ^ { \star } ( a ) \in \mathop { \mathrm { a r g } \operatorname* { m i n } } _ { k \in { \cal K } _ { a } } \mathcal { L } _ { \mathrm { v a l } } ( \theta _ { a , k } ) , \qquad a ^ { \star } \in \mathop { \mathrm { a r g } \operatorname* { m i n } } _ { a \in \mathcal { A } } J _ { y } \bigl ( \theta _ { a , k ^ { \star } ( a ) } \bigr ) .\tag{A.8}
$$

$J _ { y }$ was computed on the test trajectories. For fine-tuned Chronos-2 in the main text, we used the same learning rate, maximum number of updates, and batch size as the adopted CFM.

<table><tr><td>Model</td><td>Input/output normalization</td><td>Adopted learning Max. updates rate</td><td></td><td>Adopted checkpoint</td></tr><tr><td>Chronos-2 vanilla</td><td>Yes</td><td></td><td></td><td>Original pretrained weights</td></tr><tr><td>fine-tuned Chronos-2 No</td><td></td><td> $1 0 ^ { - 4 }$ </td><td>12000</td><td>Update 10800</td></tr><tr><td>CFM</td><td>No</td><td> $1 0 ^ { - 4 }$ </td><td>12000</td><td>Update 11800</td></tr></table>

A.3.2 CLOSED-LOOP COMPARISON UNDER MATCHED TRAINING CONFIGURATIONS

Table A1: Closed-loop output RMS under matched training configurations. The checkpoint of each run was selected by validation loss. The maximum number of updates differs from the number of updates of the checkpoint actually selected.

<table><tr><td>Learning rate</td><td>Max. updates</td><td>fine-tuned Chronos-2</td><td>CFM</td></tr><tr><td>3e-05</td><td>6,000</td><td>160.37797</td><td>7.92710</td></tr><tr><td>3e-05</td><td>12,000</td><td>152.49778</td><td>7.80909</td></tr><tr><td>0.0001</td><td>6,000</td><td>34.22627</td><td>1.79427</td></tr><tr><td>0.0001</td><td>12,000</td><td>11.04592</td><td>1.27467</td></tr><tr><td>0.0003</td><td>6,000</td><td>12.22290</td><td>1.34924</td></tr><tr><td>0.0003</td><td>12,000</td><td>10.02856</td><td>1.74633</td></tr></table>

## A.3.3 DETAILS OF ALL SEARCH CONFIGURATIONS

The evaluation metrics follow "Evaluation procedure and performance metrics" (Section A.1).

<table><tr><td>Configuration</td><td>LR</td><td>Max. updates</td><td>Min. validation loss</td><td>Selected update</td><td>Test y RMS</td><td>Test u RMS</td></tr><tr><td>fine-tuned Chronos-2</td><td>3e-05</td><td>6000</td><td>8.60341</td><td>5900</td><td>160.37797</td><td>22.85499</td></tr><tr><td>fine-tuned Chronos-2</td><td>3e-05</td><td>12000</td><td>6.03964</td><td>9700</td><td>152.49778</td><td>24.12714</td></tr><tr><td>fine-tuned Chronos-2</td><td>0.0001</td><td>6000</td><td>8.50126</td><td>3700</td><td>34.22627</td><td>18.32350</td></tr><tr><td>fine-tuned Chronos-2</td><td>0.0001</td><td>12000</td><td>6.68355</td><td>10800</td><td>11.04592</td><td>12.09465</td></tr><tr><td>fine-tuned Chronos-2</td><td>0.0003</td><td>6000</td><td>9.78136</td><td>3700</td><td>12.22290</td><td>5.08303</td></tr><tr><td>fine-tuned Chronos-2</td><td>0.0003</td><td>12000</td><td>9.09383</td><td>11900</td><td>10.02856</td><td>5.95962</td></tr><tr><td>CFM</td><td>3e-05</td><td>6000</td><td>4.83451</td><td>6000</td><td>7.92710</td><td>30.45850</td></tr><tr><td>CFM</td><td>3e-05</td><td>12000</td><td>2.24564</td><td>11800</td><td>7.80909</td><td>27.96356</td></tr><tr><td>CFM</td><td>0.0001</td><td>6000</td><td>3.63282</td><td>6000</td><td>1.79427</td><td>2.67895</td></tr><tr><td>CFM</td><td>0.0001</td><td>12000</td><td>2.86828</td><td>11800</td><td>1.27467</td><td>2.28126</td></tr><tr><td>CFM</td><td>0.0003</td><td>6000</td><td>3.38390</td><td>5900</td><td>1.34924</td><td>1.97067</td></tr><tr><td>CFM</td><td>0.0003</td><td>12000</td><td>2.72442</td><td>11700</td><td>1.74633</td><td>1.93835</td></tr></table>

In every configuration matched by learning rate and maximum number of updates, the closed-loop output RMS of CFM was smaller than that of fine-tuned Chronos-2.

The control performance of the adopted CFM and its difference from the LQG reference are shown below.
<table><tr><td> $J _ { y }$ </td><td> $J _ { u }$ </td><td> $E _ { y }$ </td><td> $E _ { u }$ </td></tr><tr><td>1.27467</td><td>2.28126</td><td>0.79346</td><td>1.28305</td></tr></table>

A.3.4 INPUT PREDICTION ERROR AND CLOSED-LOOP OUTPUT RMS FOR ALL CONFIGURATIONS

Table A2: Input prediction error and closed-loop output RMS for all 12 configurations. Input prediction is reported as the mean ± sample standard deviation of per-trajectory RMSE, evaluated on the same 100 test trajectories and 28 origins as in Table 1.
<table><tr><td>Method</td><td>Learning rate</td><td>Max. updates</td><td>Input prediction RMSE</td><td>Closed-loop output RMS</td></tr><tr><td>fine-tuned Chronos-2</td><td>3e-05</td><td>6,000</td><td> $1 . 7 0 1 2 \pm 0 . 7 3 9 3$ </td><td>160.37797</td></tr><tr><td>fine-tuned Chronos-2</td><td>3e-05</td><td>12,000</td><td>1.5970 ± 0.8323</td><td>152.49778</td></tr><tr><td>fine-tuned Chronos-2</td><td>0.0001</td><td>6,000</td><td>1.7109 ± 0.7382</td><td>34.22627</td></tr><tr><td>fine-tuned Chronos-2</td><td>0.0001</td><td>12,000</td><td>1.6793 ± 0.7333</td><td>11.04592</td></tr><tr><td>fine-tuned Chronos-2</td><td>0.0003</td><td>6,000</td><td>1.7423 ± 0.7500</td><td>12.22290</td></tr><tr><td>fine-tuned Chronos-2</td><td>0.0003</td><td>12,000</td><td>1.6906 ± 0.7297</td><td>10.02856</td></tr><tr><td>CFM</td><td>3e-05</td><td>6,000</td><td>0.6922 ± 0.0158</td><td>7.92710</td></tr><tr><td>CFM</td><td>3e-05</td><td>12,000</td><td>0.1922 ± 0.0500</td><td>7.80909</td></tr><tr><td>CFM</td><td>0.0001</td><td>6,000</td><td>0.7410 ± 0.0226</td><td>1.79427</td></tr><tr><td>CFM</td><td>0.0001</td><td>12,000</td><td>0.7772 ± 0.0271</td><td>1.27467</td></tr><tr><td>CFM</td><td>0.0003</td><td>6,000</td><td>0.7208 ± 0.0210</td><td>1.34924</td></tr><tr><td>CFM</td><td>0.0003</td><td>12,000</td><td>0.7868 ± 0.0274</td><td>1.74633</td></tr></table>

## A.3.5 FULL-RANGE CLOSED-LOOP RESPONSES

## A.3.6 REPRODUCIBILITY INFORMATION

The experiment ID is single-unstable-mseq-20260918. The data-generation seeds are given in "Experimental system, training, and evaluation procedure" (Section A.1), the training seed and settings in "Prediction tasks and connection structure" (Section A.1), and the adopted configurations in "Results by training configuration" (Section A.3).

![](images/c8c1e56015eabbc215530fbfa2041ebb29b8bd40363c898ccb8bd8a1009db68e.jpg)  
Figure A1: Full-range display of the same representative trajectory as in Fig. 3. The output (top) and input (bottom) are shown over their full range on a symmetric logarithmic axis in the left column and as a zoomed view on a linear axis in the right column. Gray: Chronos-2 vanilla; blue: fine-tuned Chronos-2; orange: CFM; black: identification and LQG reference response. The display range in the right column does not indicate an input constraint. No input saturation was applied in this experiment.

<table><tr><td>Item</td><td>Environment</td></tr><tr><td></td><td>GPU for training and cur- NVIDIA GH200, 1 per run</td></tr><tr><td>rent evaluation PyTorch / CUDA</td><td>2.13 / 13</td></tr><tr><td>Transformers / Lightning</td><td>5.14.1 / 2.6.6</td></tr><tr><td>Training precision</td><td>float32</td></tr></table>

Code, data, and settings are summarized in the list of reference materials.

## C EVALUATING THE EFFECT OF THE GRAPH MASK ON CAUSAL RELATIONS

## C.1 EVALUATION APPROACH

In predicting control time series, not only the correlations between signals but also the causal direction, i.e., which signals each signal is generated from, is meaningful. The graph mask restricts the attention scope between series on the basis of this known structure. In this evaluation, we examine whether the imposed constraints are reflected in the dependencies of the predictions of the model as a whole.

The evaluation fixes all other conditions, modifies only the observed history of some signals, and compares the resulting changes in the predictions. For signals that, by construction, have no influence on the prediction targets, we check that the predictions are invariant to such modifications. This test is separate from the magnitude of the prediction error: it checks the consistency between the known causal structure and the dependencies used by the model. Below, we carry out this evaluation using an identification task as a concrete example.

## C.2 VERIFICATION ON IDENTIFICATION

As a concrete example, we use $0 \leq t < 2$ s of one SISO identification example from Stage 2. In this identification task, the generation rule for the input u does not refer to the output y, and the causal direction is $u  y .$ . Therefore, for the model in which $y  u$ is prohibited, we check whether the prediction of u remains unchanged when only the past y is modified. The verification covers only the identification segment and does not include the control segment.

The signals present in this segment are u and $y ;$ the reference signal r is absent. With rows as receivers, columns as sources, and the order $( u , y )$ , the additive masks are

$$
B _ { t } ^ { \mathrm { d y n a m i c } } = \left[ { \begin{array} { c c } { 0 } & { - \infty } \\ { 0 } & { 0 } \end{array} } \right] , \qquad B _ { t } ^ { \mathrm { c o m p l e t e } } = \left[ { \begin{array} { c c } { 0 } & { 0 } \\ { 0 } & { 0 } \end{array} } \right]
$$

and they are constant throughout the segment. Here, 0 denotes that attention is allowed and $- \infty$ that it is prohibited; dynamic prohibits $y  u$

For the 200 recorded points of the identification segment, we predicted $u _ { c }$ one step ahead at 168 time steps with origins $c = 3 2 , \ldots , 1 9 9$ . The input at each origin is the recorded $u _ { 0 : c - 1 } , y _ { 0 : c - 1 }$ . Between the two inputs being compared, only the past outputs were changed to

$$
\widetilde { y } _ { j } = - 3 y _ { j } + 1 , \qquad 0 \leq j < c
$$

while the past $u ,$ time, graph, missingness and prediction-target specification, and the weights of each model were kept fixed. Both $u _ { c }$ and $y _ { c }$ at the prediction time were hidden, and predictions were not fed back into the history for the next origin. This is a comparison of predictions on the same recorded history, not a closed-loop evaluation in which the plant is re-simulated according to the modified $y .$

## C.3 RESULTS AND INTERPRETATION

<table><tr><td>Model</td><td>Maximum absolute difference in in- Time steps with identical predic- put prediction before and after mod- tions before and after modification ification</td></tr><tr><td>dynamic 0</td><td>168 / 168</td></tr><tr><td>complete</td><td>0.23722672 0 / 168</td></tr></table>

In this identification example, the predictions of dynamic before and after the modification were byte-identical at all 168 time steps, whereas those of complete changed at every time step. For dynamic, invariance to modification of the output history was confirmed for the prohibited direction $y  u ,$ yielding prediction dependencies consistent with the specified causal direction. For complete, whose configuration allows dependence in this direction, the prediction of u changed. The invariance here refers to agreement between the predictions before and after the output modification; it does not imply agreement with the recorded ground-truth u. The dynamic model in Fig. C1 also has prediction errors with respect to the ground-truth u.

The models compared are selected checkpoints trained separately in Stage 2. The batch size is 64 for both, while the learning rates differ: $3 \times 1 0 ^ { - 5 }$ for dynamic and $1 0 ^ { - 5 }$ for complete. For dynamic, we used the version that also applies the graph constraints to the REG tokens; models after the implementation change that fully connects the REG tokens to one another are not included. What this evaluation shows is the consistency between the known causal direction and the prediction dependencies in this identification example. Because the weights and learning rates of the two model differ, the observed difference is not isolated as a causal effect of the graph mask alone.

Comparison conditions, raw arrays, and check results · Vector version of Fig. C1

![](images/73785a91bdb5cb49ccbfde253c57c399426810783438f5d6e2c170e2ceb0cd8e.jpg)

![](images/82edcd04919642594bd87c50f6ca2d5fa794af59a2ea327b9e60b92a5a2861af.jpg)  
Figure C1: Graph mask and the dependence of input prediction on the output history. a: Additive masks for dynamic and complete. The boxes correspond to $y  u$ . b: Recorded u over the entire identification segment (black), prediction with the original y (blue), and prediction with the modified y (orange dashed). The bottom panel shows the original y and ye. c: Zoomed view of u for both models over the same interval, 1.47–1.57 s, with the same vertical axis. The gray band indicates the zoomed range. This range was chosen as 0.05 s on either side of 1.52 s, where the prediction difference of complete is largest within the identification segment, and does not represent a typical interval.