# An Efficient Machine Learning Approach for Degradation Forecasting in AEM Water Electrolysis

Marco Veneriano   
MaLGa Center   
Universita degli Studi di Genova\`   
Genoa, Italy   
marco.veneriano@edu.unige.it

Andrea Riva Antares Electrolysis S.r.l. Genoa, Italy andrea.riva@antares-electrolysis.com

Ani Gjergji   
MaLGa Center   
Universita degli Studi di Genova\`   
Genoa, Italy   
ani.gjergji@edu.unige.it   
Vito Paolo Pastore   
MaLGa Center   
Universita degli Studi di Genova\`   
Genoa, Italy   
vito.paolo.pastore@unige.it

Sebastiano Bellani Antares Electrolysis S.r.l. Genoa, Italy sebastiano.bellani@antares-electrolysis.com

Matteo Santacesaria   
MaLGa Center   
Universita degli Studi di Genova\`   
Genoa, Italy   
matteo.santacesaria@unige.it

Abstract—This study provides a data-driven analysis of a novel dataset of single-cell Anion Exchange Membrane water electrolyzers (AEMWE), operated under constant current load across multiple heterogeneous experimental campaigns. We train and evaluate a range of machine learning models with different complexity, including linear baselines, LSTMs and CNNs, to perform medium-term forecasting of the cell voltage degradation curve. The models are assessed within a rigorous training and evaluation framework specifically designed for heterogeneous industrial data.

Index Terms—Anion Exchange Membrane Water Electrolysis, Machine Learning, Voltage degradation curve, Deep Learning for time series

## I. INTRODUCTION

Green hydrogen production via water electrolysis is widely regarded as a key technology for reducing emissions in both industrial and civilian sectors. Anion Exchange Membrane Water Electrolysis (AEMWE) has recently emerged as a promising alternative to conventional Proton Exchange Membrane (PEM) and alkaline systems. AEMWE offers the potential for lower costs by reducing the reliance on fluorinated polymers and precious metals, while also mitigating their associated environmental impact. However, a major limitation for largescale deployment is the limited long-term stability of current membranes, often limiting system lifetime. For this reason, devising systems capable of robust predictive health and performance monitoring is crucial to optimize maintenance and avoid critical failures. In this work, we present the design and evaluation of a data-driven framework, applied to an inhouse AEMWE single-cell dataset, with the goal of developing predictive models for degradation dynamics on a budget, and providing insights applicable to industrial settings.

In recent years, a growing body of work has explored data-driven degradation modeling in electrochemical energy systems, primarily focusing on PEM fuel cells and PEM water electrolyzers. Most of these works focus on short-term prediction or remaining useful life estimation (RUL), often using deep architectures such as CNN–LSTM or attention-based models [1]–[4]. AEMWE-related machine learning studies have mainly addressed performance prediction and operatingpoint optimization [6], [11], rather than explicit forecasting of degradation trajectories over time, with only a limited number of works addressing it [10]. Complementary to datadriven approaches, numerous physics-based models have been developed to describe electrochemical performance and degradation mechanisms in water electrolyzers, typically grounded in electrochemical kinetics, transport phenomena, and material aging laws [12]. For PEM water electrolyzers, several works review and develop degradation modeling strategies that combine mechanistic descriptions of membrane thinning, catalyst dissolution, and gas crossover with simplified polarization equations to predict efficiency loss and lifetime under realistic operating conditions [13]. Building on these concepts, recent studies propose layer-unspecific physical models that extract quasi-steady polarization curves from operating data and track the temporal evolution of effective resistance and exchange current density, yielding millivolt-scale voltage prediction errors and enabling lifetime forecasts with uncertainty quantification [8]. Hybrid approaches have recently been adopted, like Physics-Informed Neural Networks (PINNs). These provide an alternative that improves consistency and interpretability through electrochemical and transport priors, such as temperature prediction for PEM cells [9]. For AEMs hybrid models are adopted in [14], to focus on the evolution of hydroxide conductivity under prolonged alkaline exposure as key indicator of chemical and structural breakdown. While these physics-based models offer interpretability and extrapolation capabilities, they require extensive characterization of material properties, careful calibration, and often assume quasi-stationary or slowly varying operating conditions [7].

Considering a heterogeneous dataset of AEMWE singlecell aging campaigns provided by Antares Electrolysis, this study aims to model AEMWE degradation by forecasting the future evolution of the voltage curve over medium-term horizons. The dataset comprises 12 distinct experimental runs, with different setup configurations (e.g. membranes, catalyst layers, etc.), each lasting between 40 and 100 hours under different configurations. Importantly, the time series considered correspond to the initial phase of operation, characterized by transient dynamics and significantly higher degradation rates compared to steady mid-life conditions. As a result, this makes the forecasting task particularly challenging and distinct from typical degradation studies, which often focus on quasistationary regimes [1]–[3].

Accordingly, the adopted modeling choices are driven by the need for architectures that can generalize to unseen experimental conditions, remain robust to noise, and perform effectively in a data-scarce regime. Specifically, we formulate a multihorizon forecasting problem in which a 24-hour observation window of voltage measurements is used to predict the future voltage at horizons of 3, 6, 12, 18, and 24 hours.

To this end, the main contributions of this work are summarized as follows:

• We introduce and analyze a unique dataset of AEMWE degradation trajectories, capturing transient early-life dynamics across multiple heterogeneous experimental runs.

• We propose a rigorous framework for training and evaluation in data-scarce settings, featuring a Leave-One-Group-Out cross-validation protocol specifically tailored for industrial deployment.

• We systematically compare machine learning models of varying complexity, ranging from linear and non-linear baselines to shallow regularized networks and deep temporal architectures (LSTMs, 1D-CNNs), to identify the optimal trade-off between expressive power and stability.

## II. EXPERIMENTAL DATASET AND PROBLEM DEFINITION

## A. Experimental Dataset and Notation

The experimental analysis is based on a dataset of electrochemical time series provided by Antares Electrolysis, obtained from constant-current aging campaigns on a 5 cm<sup>2</sup> Anion Exchange Membrane (AEM) single cell. These tests were performed under heterogeneous operating protocols using a custom test station specifically engineered to characterize cell degradation behavior. The complete dataset comprises 12 distinct experimental runs, each lasting between 40 and 100 hours, with a primary focus on capturing cell voltage evolution over time.

Formally, the dataset is defined as a collection of N experiments:

$$
\mathcal { D } = \{ E _ { 1 } , \ldots , E _ { N } \} ,\tag{1}
$$

![](images/f641ebf380db98c799bdf5ded77f7828ecb83848ffb5980a8e10ee0959104719.jpg)  
Fig. 1. Representative raw voltage profile from a single experiment.

where each individual experiment $E _ { i }$ is represented by the tuple:

$$
E _ { i } = \left( \{ ( t _ { k } ^ { ( i ) } , v _ { k } ^ { ( i ) } ) \} _ { k = 1 } ^ { K _ { i } } , \ \mathbf { m } ^ { ( i ) } \right) ,\tag{2}
$$

where $K _ { i }$ is the number of samples of experiment i. Here, $t _ { k } ^ { ( i ) } \in [ 0 , \mathcal { T } _ { i } ]$ denotes the discrete measurement timestamp, $v _ { k } ^ { ( i ) } \in$ R represents the observed cell voltage at that timestamp, and $\mathbf { m } ^ { ( i ) } \in \mathbb { R } ^ { d _ { m } }$ is a vector of static metadata detailing the specific initial experimental conditions of the run (e.g. membrane properties, catalyst layers, etc.).

A representative raw voltage profile from the dataset is illustrated in Fig. 1. The degradation trajectory reveals complex, non-linear dynamics characterized by distinct local trends. Although environmental and operational metadata $\mathbf { m } ^ { ( i ) }$ are recorded, the metadata vector is omitted from the core modeling pipeline, reducing the objective to a purely univariate timeseries forecasting task. This was motivated by a preliminary exploratory analysis, where metadata were incorporated as features and coupled to the flattened input window. These experiments showed no benefit or a decreasing of predictive performance, indicating that, in our setting, metadata static features provide only a negligible predictive signal for the temporal degradation horizon considered.

## B. Preprocessing

The raw signals are irregularly sampled, with higher acquisition frequency during transient regimes characterized by rapid signal variations, resulting in a non-uniform temporal grid. Additionally, the measurements are affected by significant high-frequency noise. To obtain a uniformly sampled and noise-robust representation suitable for learning, the following preprocessing pipeline is applied independently to each experiment:

1) Gaussian smoothing, used to attenuate high-frequency noise while preserving long-term degradation trends.

2) Cubic interpolation: a cubic interpolant is fitted to the smoothed signal, yielding a continuous-time function

$$
\tilde { V } ^ { ( i ) } : [ 0 , \mathcal { T } _ { i } ]  \mathbb { R } .\tag{3}
$$

3) Uniform resampling, producing a discrete-time signal

$$
V _ { n } ^ { ( i ) } : = \tilde { V } ^ { ( i ) } ( n \Delta t ) , \quad n = 0 , \dots , T _ { i } ,\tag{4}
$$

with fixed sampling interval $\Delta t = 5$ minutes. Note that $T _ { i }$ may change depending on i, since the experiments have different duration.

This temporal resolution reflects the fact that electrochemical degradation evolves on timescales significantly longer than one hour, making finer sampling unnecessary for the prediction task.

## C. Final dataset construction

A key challenge is the strong heterogeneity across experiments, which differ in duration, voltage range, membrane and catalyst properties, and noise characteristics. To enable learning across experiments of varying length, we construct a dataset of fixed-size input-output pairs using a sliding-window approach.

a) Sliding windows: We define the window length L and stride $f$ as

$$
L = \frac { 2 4 \mathrm { ~ h ~ } } { \Delta t } , \quad f = \frac { 1 \mathrm { ~ h ~ } } { \Delta t } .\tag{5}
$$

For each experiment, overlapping windows are extracted as

$$
\widetilde W _ { i , j } = \big ( V _ { n _ { j } ^ { ( i ) } } ^ { ( i ) } , \dots , V _ { n _ { j } ^ { ( i ) } + L - 1 } ^ { ( i ) } \big ) , \quad n _ { j } ^ { ( i ) } = j f ,\tag{6}
$$

where $i = 1 , \ldots , N$ is the experiment index.

The window length corresponds to a full 24-hour cycle, allowing the models to capture daily variations in operating conditions. The chosen stride results in highly overlapping windows; to prevent data leakage, all windows extracted from a single experiment are assigned to the same split during evaluation.

b) Prediction task: The goal is to learn a mapping

$$
f _ { \theta } : \mathbb { R } ^ { L }  \mathbb { R }\tag{7}
$$

that predicts a scalar summary of the future system state given a past observation window.

Rather than predicting a single future value, we adopt a direct multi-horizon formulation. Let $H \in \{ 3 , 6 , 1 2 , 1 8 , 2 4 \}$ denote the prediction horizon and $H _ { s } = H / \Delta t$ . The target associated with $\widetilde { W } _ { i , j }$ is defined as

$$
\tilde { y } _ { i , j } = \frac { 1 } { A } \sum _ { n = n _ { j } ^ { ( i ) } + L + H _ { s } - A } ^ { n _ { j } ^ { ( i ) } + L + H _ { s } - 1 } V _ { n } ^ { ( i ) } , \quad A = \frac { 1 } { 4 } H _ { s } .\tag{8}
$$

This definition corresponds to averaging over the final portion of the prediction horizon, yielding a robust estimate of the future regime and mitigating the effect of high-frequency noise.

c) Normalization: To account for distributional differences across experiments and avoid data leakage, we adopt a per-window normalization scheme. For each window $\widetilde { W } _ { i , j } ,$ we compute its mean $\mu _ { i , j }$ and standard deviation $\sigma _ { i , j }$ , and define

$$
W _ { i , j } = \frac { \widetilde { W } _ { i , j } - \mu _ { i , j } } { \sigma _ { i , j } } , \quad y _ { i , j } = \frac { \tilde { y } _ { i , j } - \mu _ { i , j } } { \sigma _ { i , j } } .\tag{9}
$$

This local standardization emphasizes relative temporal dynamics while ensuring robustness to variations in absolute voltage scale across experiments.

d) Final dataset: The final dataset is

$$
\mathcal { D } _ { f } = \{ ( W _ { i , j } , y _ { i , j } ) \} ,\tag{10}
$$

where the experiment index $i = 1 , \ldots , N$ is kept in order to implement the evaluation procedure we will introduce later. Note that, since each experiment i has different duration, the index j will range on an index set depending on i.

## III. METHODOLOGY AND ARCHITECTURES

The forecasting task is characterized by a severe low-data regime, with the final dataset $\mathcal { D } _ { f }$ containing approximately 200–500 samples depending on the prediction horizon. This setting introduces a high risk of overfitting and makes careful control of model complexity essential.

To address this challenge, we adopt a strict cross-experiment evaluation protocol together with a restrained modeling strategy.

## A. Evaluation Protocol

The dataset consists of $N = 1 2$ independent experimental runs, each generating multiple highly overlapping samples via a sliding-window procedure. Standard random splits are not appropriate in this setting, as they would mix samples from the same experiment across training and test sets, leading to data leakage and overly optimistic performance estimates.

We therefore adopt a Leave-One-Group-Out Cross-Validation (LOGO-CV) strategy, where each experimental run defines a group. Let $\mathcal { E } ~ = ~ \{ E _ { 1 } , \ldots , E _ { N } \}$ denote the set of experiments. For each fold k, we define:

• Test set: all samples derived from $E _ { k }$ ;

• Training set: all samples derived from $\mathcal { E } \setminus \{ E _ { k } \}$

Each model is trained from scratch on the training folds and evaluated on the held-out experiment. Final performance is reported as the average across all folds, providing an estimate of cross-experiment generalization, while avoiding the leakage of data between training and evaluation.

## B. Validation and Model Selection

Due to the limited number of independent experiments, we do not employ an additional validation split. Preliminary experiments showed that such splits lead to high-variance performance estimates and unstable model selection, as results depend strongly on the specific choice of validation experiments.

TABLE I  
GLOBAL PERFORMANCE SUMMARY: COMPARISON OF MODELS ACROSS TIME HORIZONS (MAE).
<table><tr><td rowspan=1 colspan=1>Horizon</td><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>Linear</td><td rowspan=1 colspan=1>Ridge</td><td rowspan=1 colspan=1>RF</td><td rowspan=1 colspan=1>Shallow NN</td><td rowspan=1 colspan=1>LSTM</td><td rowspan=1 colspan=1>1D-CNN</td></tr><tr><td rowspan=2 colspan=1>24h</td><td rowspan=1 colspan=1>Norm</td><td rowspan=1 colspan=1>2.708</td><td rowspan=1 colspan=1>2.567</td><td rowspan=1 colspan=1>2.960 ± 0.015</td><td rowspan=1 colspan=1>1.816 ± 0.015</td><td rowspan=1 colspan=1>1.977 ± 0.205</td><td rowspan=1 colspan=1>1.845 ± 0.158</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathbf { c } \mathbf { V } } }$ </td><td rowspan=1 colspan=1>2.308</td><td rowspan=1 colspan=1>1.909</td><td rowspan=1 colspan=1> $2 . 3 6 2 \pm 0 . 0 0 7$ </td><td rowspan=1 colspan=1>1.000 ± 0.005</td><td rowspan=1 colspan=1> $1 . 0 6 3 \pm 0 . 1 4 6$ </td><td rowspan=1 colspan=1>1.284 ± 0.221</td></tr><tr><td rowspan=2 colspan=1>18h</td><td rowspan=1 colspan=1>Norm</td><td rowspan=1 colspan=1>2.386</td><td rowspan=1 colspan=1>2.127</td><td rowspan=1 colspan=1> $\overline { { 2 . 2 5 1 \pm 0 . 0 2 3 } }$ </td><td rowspan=1 colspan=1> $1 . 7 3 3 \pm 0 . 0 1 9$ </td><td rowspan=1 colspan=1> $\mathbf { 1 . 7 3 0 \pm 0 . 0 3 5 }$ </td><td rowspan=1 colspan=1>1.756 ± 0.059</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathbf { c } \mathbf { V } } }$ </td><td rowspan=1 colspan=1>2.001</td><td rowspan=1 colspan=1>1.710</td><td rowspan=1 colspan=1>1.772 ± 0.024</td><td rowspan=1 colspan=1>1.150 ± 0.020</td><td rowspan=1 colspan=1> $\mathbf { 1 . 0 9 1 \pm 0 . 0 5 9 }$ </td><td rowspan=1 colspan=1>1.359 ± 0.074</td></tr><tr><td rowspan=2 colspan=1>12h</td><td rowspan=1 colspan=1>Norm</td><td rowspan=1 colspan=1>2.064</td><td rowspan=1 colspan=1>1.614</td><td rowspan=1 colspan=1>1.727 ± 0.025</td><td rowspan=1 colspan=1>1.471 ± 0.004</td><td rowspan=1 colspan=1>1.562 ± 0.041</td><td rowspan=1 colspan=1>1.543 ± 0.007</td></tr><tr><td rowspan=1 colspan=1>cV</td><td rowspan=1 colspan=1>1.868</td><td rowspan=1 colspan=1>1.393</td><td rowspan=1 colspan=1>1.483 ± 0.014</td><td rowspan=1 colspan=1>1.180 ± 0.010</td><td rowspan=1 colspan=1>1.157 ± 0.039</td><td rowspan=1 colspan=1>1.278 ± 0.013</td></tr><tr><td rowspan=2 colspan=1>6h</td><td rowspan=1 colspan=1>Norm</td><td rowspan=1 colspan=1>1.612</td><td rowspan=1 colspan=1>1.020</td><td rowspan=1 colspan=1> $\overline { { 1 . 0 4 2 \pm 0 . 0 1 1 } }$ </td><td rowspan=1 colspan=1>0.970 ± 0.009</td><td rowspan=1 colspan=1> $\overline { { 1 . 1 5 4 \pm 0 . 0 3 5 } }$ </td><td rowspan=1 colspan=1>1.025 ± 0.009</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathrm { ~ c v ~ } } }$ </td><td rowspan=1 colspan=1>1.566</td><td rowspan=1 colspan=1>0.867</td><td rowspan=1 colspan=1>0.908 ± 0.009</td><td rowspan=1 colspan=1>0.850 ± 0.010</td><td rowspan=1 colspan=1> $\overline { { 0 . 9 6 5 \pm 0 . 0 2 8 } }$ </td><td rowspan=1 colspan=1>0.887 ± 0.010</td></tr><tr><td rowspan=2 colspan=1>3h</td><td rowspan=1 colspan=1>Norm</td><td rowspan=1 colspan=1>1.251</td><td rowspan=1 colspan=1>0.644</td><td rowspan=1 colspan=1>0.683 ± 0.002</td><td rowspan=1 colspan=1>0.675 ± 0.003</td><td rowspan=1 colspan=1>0.790 ± 0.045</td><td rowspan=1 colspan=1>0.668 ± 0.002</td></tr><tr><td rowspan=1 colspan=1> $\overline { { \mathrm { \Omega c V } } }$ </td><td rowspan=1 colspan=1>1.235</td><td rowspan=1 colspan=1>0.575</td><td rowspan=1 colspan=1>0.607 ± 0.002</td><td rowspan=1 colspan=1>0.620 ± 0.000</td><td rowspan=1 colspan=1>0.690 ± 0.036</td><td rowspan=1 colspan=1>0.587 ± 0.010</td></tr></table>

Instead, all architectures and hyperparameters are set based on standard practice (see Section III-C). These configurations remain constant across all cross-validation folds. This restrained strategy reduces the risk of overfitting to the evaluation protocol and ensures a consistent comparison across models.

We emphasize that the objective of this study is not to achieve state-of-the-art performance through extensive hyperparameter tuning, but rather to evaluate the robustness and relative inductive biases of different modeling approaches in a low-data regime.

## C. Models and Architectures

We consider a hierarchy of models with increasing expressive power, ranging from linear baselines to deep neural architectures. All models are implemented in Python, using scikit-learn for classical methods and TensorFlow for neural architectures. Models are trained using the mean squared error loss; neural models are optimized with Adam [15] (lr $= ~ 1 0 ^ { - 3 } )$ , employing a learning rate scheduler to improve convergence and early stopping to prevent overfitting.

## 1) Baselines:

a) Global linear trend: As a simple heuristic, we assume that the degradation trend observed in the input window remains constant over the prediction horizon. For each sample, we fit

$$
v ( t ) = \alpha t + \beta ,\tag{11}
$$

and extrapolate to the target time t<sub>target</sub>:

$$
\hat { y } = \alpha t _ { \mathrm { t a r g e t } } + \beta .\tag{12}
$$

b) Ridge regression: We model the target as a linear function of past observations:

$$
\begin{array} { r } { \hat { y } = \mathbf { w } ^ { \top } \mathbf { x } + b , } \end{array}\tag{13}
$$

where x is the flattened input window. Parameters are estimated using ridge regression with $L ^ { 2 }$ regularization $( \alpha = 1 )$ ensuring stability under strong collinearity in the input features.

c) Random Forest: We include a Random Forest regressor as a strong non-parametric baseline for tabular representations of the input windows. The model is configured with 300 trees, minimum leaf size of 5, and $\sqrt { L }$ feature subsampling, where $L = 2 8 8$ is the input dimensionality. This configuration provides a balance between variance reduction and robustness in low-sample regimes.

2) Shallow neural network: We consider a shallow fully connected ReLU Neural Network, implemented with strong regularization. The architecture consists of a single hidden layer with 128 ReLU units, $L ^ { 2 }$ normalization, and dropout (rate 0.2).

## 3) Deep temporal models:

a) 1D Convolutional Neural Network (1D-CNN): The 1D-CNN captures local temporal dependencies through convolutional filters. The architecture consists of a single convolutional layer with 16 filters and kernel size 5, followed by max pooling (pool size 2), dropout (rate 0.3), and a fully connected layer with 16 ReLU units. This design reduces parameter count while preserving sensitivity to local structure in the time series.

b) Long Short-Term Memory network (LSTM): The LSTM explicitly models sequential dependencies through recurrent dynamics. The input sequence is processed and summarized via the final hidden state. The model consists of a single LSTM layer with 8 units and dropout (rate 0.2), followed by a fully connected layer with 4 ReLU units and dropout (rate 0.2), and a final linear output layer.

Although more expressive in principle, the LSTM is intentionally kept small due to the low-data regime. Empirically, increasing model capacity led to degraded performance, suggesting strong overfitting sensitivity.

## IV. EXPERIMENTAL RESULTS AND DISCUSSION

## A. Performance metrics

Model performance is evaluated using the Mean Absolute Error (MAE). Given ground truth targets $y _ { i }$ and predictions $\hat { y } _ { i } ,$ , the MAE is defined as

$$
\mathrm { M A E } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \left| y _ { i } - \hat { y } _ { i } \right| .\tag{14}
$$

We report MAE both in normalized units and in physical units (centiVolts, cV), obtained by inverting the per-window normalization (see Eq. 9), along with their standard deviations across five different training and evaluation runs. Reporting results in physical units facilitates interpretation in the context of electrochemical performance and degradation.

## B. Results

We present results across five different time horizons (3, 6, 12, 18, 24 hours) in Table I. The best performing baseline is Ridge Regression by a clear margin: the global linear trend struggles to capture the full complexity of the degradation phenomenon, while the Random Forest shows weaker performance at longer horizons.

The Shallow Neural Network achieves the most consistent performance across all time scales, as well as having lower standard deviation between runs, compared to temporal models, which highlights more robust predictions. Regarding deep temporal models, the LSTM performs well on longer time horizons but shows degraded performance on shorter time windows, while the 1D-CNN provides stable results across all horizons, with slightly higher error than the Shallow Neural Network.

Regarding the centiVolts MAE, the errors remain consistently bounded across all forecasting horizons, without exhibiting a systematic increase at longer time scales. This suggests that the models are learning physically meaningful signal variations rather than overfitting short-term noise, and that the improvement in normalized performance translates into stable accuracy in the original voltage scale.

## V. CONCLUSIONS

The proposed framework proves robust in handling heterogeneous and data-scarce conditions, supporting its applicability in industrial settings. In this regime, simple and strongly regularized models such as the Shallow Neural Network achieve the most consistent performance across time horizons. The 1D-CNN provides competitive results, indicating that convolutional temporal biases are effective across different forecasting ranges. In contrast, the LSTM shows less stable performance, particularly on shorter horizons, and appears more sensitive to capacity and training choices.

Overall, the results suggest that, under severe data scarcity and noisy electrochemical time series, simple regularized neural architectures offer the most reliable trade-off between performance and stability in our setting.

## ACKNOWLEDGEMENTS

Funded by the European Union. This work is supported by the Horizon Europe Grant Agreements No. 101251004 (NEREUS) and No. 101251223 (SCALE-AEM), through the Clean Hydrogen Partnership and its members. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the Clean Hydrogen Joint Undertaking. Neither the European Union nor the granting authority can be held responsible for them. This work was also supported by the Italian Ministry of the Environment and Energy Security (MASE) through the Mission Innovation 2.0 project “Sole di notte” (ID: MI ERE 00192), CUP: F33C25001220001.

## REFERENCES

[1] F. Zhang, M. Ni, S. Tai, B. Zu, F. Xi, Y. Shen, B. Wang, Z. Qin, R. Wang, T. Guo, K. Jiao, ”Machine learning assisted health status analysis and degradation prediction of aging proton exchange membrane fuel cells.” Applied Energy 384 (2025): 125483. DOI: 10.1016/j.apenergy.2025.125483.

[2] C. Jia, H. He, J. Zhou, K. Li, J. Li, Z. Wei, ”A performance degradation prediction model for PEMFC based on bi-directional long short-term memory and multi-head self-attention mechanism.” International Journal of Hydrogen Energy 60 (2024): 133-146. DOI: 10.1016/j.ijhydene.2024.02.181.

[3] B. Xu, W. Ma, W. Wu, Y. Wang, Y. Yang, J. Li, X. Zhu, Q. Liao, ”Degradation prediction of PEM water electrolyzer under constant and start-stop loads based on CNN-LSTM.” Energy and AI 18 (2024): 100420. DOI: 10.1016/j.egyai.2024.100420.

[4] L. Hongwei, Q. Binxin, H. Zhicheng, L. Junnan, Y. Yue, L. Guolong, ”An interpretable data-driven method for degradation prediction of proton exchange membrane fuel cells based on temporal fusion transformer and covariates.” international journal of hydrogen energy 48.66 (2023): 25958-25971. DOI:10.1016/j.ijhydene.2023.03.316.

[5] S. Li, Y. Ma, Y. Nie, H. Yuan, S. Zhou, P. Bao, X. Wei, J. Zhu, H. Dai, ”A Review on Data-Driven-Based Remaining Useful Life Prediction Methods for Proton Exchange Membrane Fuel Cells.” Energy Technology 14.4 (2026): e202502564. DOI: 10.1002/ente.202502564.

[6] M. M. Kabir, Y. Choden, S. Phuntsho, L. Tijing, H. K. Shon, ”Predictive machine learning optimization of anion exchange membrane water electrolysis systems.” Desalination (2025): 119198. DOI: 10.1016/j.desal.2025.119198.

[7] A. Majumdar, M. Haas, I. Elliot, S. Nazari, ”Control and controloriented modeling of PEM water electrolyzers: A review.” International Journal of Hydrogen Energy 48.79 (2023): 30621-30641. DOI: 10.1016/j.ijhydene.2023.04.204.

[8] F. Dittmar, T. Lickert, T. Smolinka, K. Pinkwart, J. Tubke, ”Develop-¨ ment and validation of a layer-unspecific physical model for voltage degradation to predict the remaining useful life of proton exchange membrane water electrolyzers.” Journal of Power Sources 668 (2026): 239122. DOI: 10.1016/j.jpowsour.2025.239122.

[9] I. Zerrougui, Z. Li, D. Hissel. ”Physics-Informed Neural Network for modeling and predicting temperature fluctuations in proton exchange membrane electrolysis.” Energy and AI 20 (2025): 100474. DOI: 10.1016/j.egyai.2025.100474.

[10] R. J. van der Horst, A. Silani, M. Khosravi, ”Control-Oriented Hybrid Modeling of AEM Electrolyzer via Residual Error Correction.” IEEE Transactions on Industry Applications (2026). DOI: 10.1109/TIA.2026.3656156.

[11] T. Wang, J. Wang, C. Zhang, P. Wang, Z. Ren, H. Guo, Z. Wu, F. Wang, ”Direct operational data-driven workflow for dynamic voltage prediction of commercial alkaline water electrolyzers based on artificial neural network (ANN).” Fuel 376 (2024): 132624. DOI: 10.1016/j.fuel.2024.132624.

[12] Riccardo Venturino, Alessio D’Alessandro, Laura Traversone, Fiammetta Rita Bianchi, Barbara Bosio, ”Development of a predictive model for performance analysis of anion exchange membrane electrolysers.” Electrochimica Acta (2026): 148753. DOI: 10.1016/j.electacta.2026.148753.

[13] A. Makhsoos, M. Kandidayeni, B. G. Pollet, L. Boulon, ”Proton exchange membrane water electrolyzers degradation models review: implications for power allocation and energy management.” Journal of Power Sources 655 (2025): 238003. DOI: 10.1016/j.jpowsour.2025.238003.

[14] William Schertzer, Mohammed Al Otmi Janani Sampath, Ryan P. Lively, Rampi Ramprasad, ”AI-Assisted Physics-Informed Predictions of Degradation Behavior of Polymeric Anion Exchange Membranes.” The Journal of Physical Chemistry B 130.5 (2026): 1684-1693. DOI: 10.1021/acs.jpcb.5c07063.

[15] D. P. Kingma and J. Ba, ”Adam: A method for stochastic optimization,” arXiv preprint arXiv:1412.6980, 2014. DOI: 10.48550/arXiv.1412.6980.