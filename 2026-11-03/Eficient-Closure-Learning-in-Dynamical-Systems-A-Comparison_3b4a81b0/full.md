# Eficient Closure Learning in Dynamical Systems: A Comparison of Data-Assimilated and Embedded Approaches

Rileigh Bandy<sup>1</sup>, Teresa Portone<sup>∗1</sup>, Mohammad Khalil<sup>1</sup>, Eric Crislip<sup>2</sup>, and Kyle Neal<sup>1</sup>

<sup>1</sup>Sandia National Laboratories

<sup>2</sup>Department of Statistics, The Ohio State University, Columbus, Ohio

November 3, 2026

## Abstract

Increased computing power has transformed computational science and improved predictive capabilities for complex systems, but many phenomena still exceed current simulation capabilities or require rapid evaluations for fast design iterations. Closure models address this challenge by approximating unresolved efects in reduced or incomplete dynamical systems. Data-driven closures ofer a flexible mechanism for improving model accuracy and applicability, but approaches that embed the closure within a computational model for construction and training phases can be computationally expensive and cause numerical instabilities in the computational model during training. To avoid the challenges posed by embedded approaches, staged approaches to closure-model learning have been developed. The first stage uses data assimilation to infer state and closure trajectories from observations. The second stage trains a machine learning closure model ofline using the assimilated data, thereby avoiding repeated simulator calls during training. In this work, we investigate the cost and accuracy tradeofs between embedded and staged data-driven closure modeling approaches for a synthetic epidemiological SIQR model with a nonlinear state-dependent quarantine closure. The results illustrate a practical accuracy-eficiency trade-of: for this exemplar, embedded training provides the strongest average accuracy, while the staged approach yields comparable qualitative behavior with a moderate increase in error, substantially fewer model evaluations, and improved optimization robustness

## 1 Introduction

Increased computational power has enabled high-fidelity simulation across a wide range of scientific and engineering applications, but significant challenges remain in constructing accurate and complete mathematical models. A central issue is the closure problem, in which unknown quantities appear in the governing equations and must be modeled in order to close the system. These unknown quantities may arise from unresolved scales, incomplete physics, or coarse-grained representations. Prominent examples of closure problems include Reynolds stresses in Reynolds-averaged Navier–Stokes equations, subgrid-scale stresses in large eddy simulations, and constitutive relations in continuum mechanics [1, 2, 3].

Traditional approaches to constructing closure models rely on dimensional analysis, asymptotic reasoning, scaling arguments, and empirical calibration [4, 2, 3, 5]. These methods have produced robust and widely used models that perform well within specific calibration regimes. However, they often result in relatively simple algebraic or low-dimensional functional forms, which can limit expressiveness and adaptability. Consequently, such closures may struggle to generalize across regimes or capture complex nonlinear interactions in many physical and biological systems [6].

These limitations motivate more flexible data-driven approaches to closure modeling. Recent advances in machine learning, together with increasing availability of high-fidelity simulation and experimental data, have enabled the construction of closure models based on expressive function approximators such as neural networks [7, 8, 9, 10, 11, 12, 13, 14]. Such approaches can capture nonlinear dependencies and latent structures that are dificult to encode using traditional closure modeling paradigms. However, significant challenges remain when neural closures are trained while embedded directly within the governing equations. Each loss or validation evaluation requires a forward model solve, and training may require diferentiating through the numerical solver. Embedded training can also be numerically fragile: as the neural-network parameters evolve, the learned closure may alter the system dynamics in ways that make the initial-value problem dificult to solve, for example by inducing stifness, nonphysical states, or extremely small stable time steps. These challenges become more pronounced when multiple training trajectories or parameter scenarios are used to improve generalizability, because each scenario requires an additional model solve during optimization.

To address these challenges, staged approaches to data-driven closure modeling have been developed [15, 16, 17, 18, 19]. The central idea is to decouple closure inference from closure-function training. In the first stage, data assimilation is used to infer closure-consistent state and closure trajectories from observations by augmenting the physical state with a latent closure variable. In the second stage, a regression or machine-learning model is trained ofline to map resolved states to the inferred closure values. After training, the learned closure can be embedded into the governing equations for prediction. This workflow avoids repeated forward simulations during neural-network optimization and does not require the simulator to be diferentiable.

Despite the practical appeal of staged closure learning, the accuracy–cost trade-ofs relative to embedded neural-closure training are not yet fully understood. This paper therefore presents a systematic comparison of these tradeofs in the context of a synthetic compartment-based epidemiological model, where an unknown closure represents a state-dependent transition rate from infectious to quarantined populations. Because the study is synthetic, the true closure is known and can be used for diagnostic evaluation, although the methods themselves use only noisy observations of population state variables. For this study, we find that the embedded approach achieves lower average held-out calibration error and lower average calibration closure error, but requires 9× greater model evaluations for training than the staged approach.

The remainder of the paper is organized as follows. Section 2 reviews the closure-learning problem, embedded neural closure modeling, and the data-assimilation tools used for staged closure inference. Section 3 defines the comparison protocol between embedded and staged approaches, including data usage, neuralnetwork optimization, performance metrics, computational-cost accounting, and robustness measures. Section 4 presents the synthetic SIQR numerical study and compares the two methods in terms of held-out calibration accuracy, diagnostic closure error, training-stage model evaluations, failed neural-network starts, and validation-scenario generalization. Section 5 summarizes the main findings and discusses future direc tions.

## 2 Background

## 2.1 The closure problem

Many systems of interest can be represented as dynamical systems whose resolved state $\boldsymbol { x } ( t ) \in \mathbb { R } ^ { n _ { x } }$ evolves according to

$$
\frac { d \boldsymbol { x } ( t ) } { d t } = R ( \boldsymbol { x } ( t ) , \boldsymbol { \xi } ( \boldsymbol { x } ( t ) ) ) ,\tag{1}
$$

where R denotes the known resolved dynamics and $\xi$ denotes an unknown closure. The closure may represent unresolved physics, model-form error, coarse-graining efects, missing source terms, or unresolved couplings. The closure problem arises because the governing equations cannot be evaluated without specifying $\xi .$ In many applications, the functional form of $\xi$ is unavailable or only approximately known.

The goal of data-driven closure learning is to construct a parameterized approximation $\xi _ { \theta }$ from observations of the resolved state or other derived quantities of interest. Once learned, the closure model is embedded into the dynamical system to produce the closed predictive model

$$
\frac { d \boldsymbol { x } ( t ) } { d t } = R ( \boldsymbol { x } ( t ) , \xi _ { \theta } ( \boldsymbol { x } ( t ) ) ) .\tag{2}
$$

Suppose observations are available at times $\{ t _ { i } \} _ { i = 0 } ^ { N _ { t } - 1 }$ . A linear observation model can be defined as

$$
d _ { i } = H x ( t _ { i } ) + \epsilon _ { i } , \qquad \epsilon _ { i } \stackrel { i i d } { \sim } { \mathcal { N } } ( 0 , \Gamma ) ,\tag{3}
$$

where H maps the resolved state to the observed quantities and Γ is the observation-error covariance. For any given training procedure, a subset of all observation times may be employed (e.g., for test-train-validation split). In the following discussion we denote a subset of observation-time indices as $\mathcal { T } ,$ , and we specify which indices are used for diferent training procedures in detail in Section 3.

## 2.2 Embedded training

Embedded training places the parametric closure model directly inside the governing equations during optimization. For a neural-network closure $\xi _ { \theta }$ , the embedded model is

$$
\frac { d x ( t ; \theta ) } { d t } = R ( x ( t ; \theta ) , \xi _ { \theta } ( x ( t ; \theta ) ) ) ,\tag{4}
$$

where $x ( t ; \theta )$ denotes the model trajectory obtained with closure parameters θ. For Gaussian observation errors, embedded training minimizes the negative log-likelihood of the training observations. Up to additive constants independent of θ, this objective is

$$
\mathcal { L } _ { \mathrm { e m b } } ( \theta ) = \frac { 1 } { 2 } \sum _ { i \in \mathcal { I } } \left( d _ { i } - H x ( t _ { i } ; \theta ) \right) ^ { \top } \Gamma ^ { - 1 } \left( d _ { i } - H x ( t _ { i } ; \theta ) \right) .\tag{5}
$$

This approach is closely related to universal diferential equations, which augment mechanistic models with trainable function approximators [20]. A key advantage of embedded training is that the closure is optimized according to its direct efect on the observable state trajectory. However, each loss or validation evaluation requires a forward solve of the dynamical system. Gradient-based optimization may also require diferentiating through the numerical time integrator. These costs become more pronounced when the data includes multiple scenarios (e.g., multiple initial conditions), because each scenario requires a unique model solve. Embedded training can also be numerically fragile because intermediate neural-network parameters may alter the dynamics in ways that make the initial-value problem dificult to solve. For example, the learned closure may induce stifness, nonphysical states, exploding trajectories, or extremely small time steps to maintain numerical stability.

## 2.3 Staged training

Staged training separates closure inference from closure-function training. The first stage uses data assimilation to infer a time-dependent closure trajectory that is consistent with the observations. The second stage trains a regression model ofline to map resolved states to the inferred closure values. After this supervised training step, the learned closure is embedded into the governing equations for prediction. Thus, staged training invokes the expensive computational model during data assimilation, but avoids repeated model calls during neural-network optimization. It also does not require the computational model to be diferentiable with respect to closure-model parameters.

## 2.3.1 Step 1: Data assimilation by state augmentation, EnKF, and smoothing

The staged approach considered here estimates the latent closure by augmenting the physical state with the closure variable. Define the augmented state as

$$
z ( t ) = \Big [ \mathop { x } ( t ) \Big ] .\tag{6}
$$

The physical state evolves according to the known model evaluated with a time-dependent closure, while the closure is assigned artificial stochastic dynamics:

$$
\frac { d \boldsymbol { x } ( t ) } { d t } = \boldsymbol { R } ( \boldsymbol { x } ( t ) , \boldsymbol { \xi } ( t ) ) ,\tag{7}
$$

$$
d \xi ( t ) = \sigma _ { \xi } d W ( t ) .\tag{8}
$$

Here $W ( t )$ is a Wiener process and $\sigma _ { \xi }$ controls the closure process-noise magnitude. The stochastic closure dynamics are not intended to be a mechanistic model for the closure. Instead, they provide flexibility for the data-assimilation procedure to attribute model-data mismatch to an unresolved closure process.

The observation model for the augmented state is

$$
d _ { i } = \mathcal { H } z ( t _ { i } ) + \epsilon _ { i } , \qquad \epsilon _ { i } \overset { i i d } { \sim } \mathcal { N } ( 0 , \Gamma ) .\tag{9}
$$

If only the physical state is observed, then $\mathcal { H } = \lceil H  \quad 0 \rceil$

We use an ensemble Kalman filter (EnKF) [21] to sequentially update an ensemble representation of the augmented state. At each observation time, the ensemble is first propagated through the augmented dynamics in (7)–(8). The forecast ensemble is then corrected using the discrepancy between the predicted and observed data. Because the closure variable is part of the augmented state, this update adjusts both the resolved physical variables and the latent closure through their ensemble-estimated correlations with the observations. The closure process-noise scale $\sigma _ { \xi }$ can be fixed or estimated from the observations used in the assimilation window. In this work, $\sigma _ { \xi }$ is selected by maximizing an EnKF innovation likelihood, following the innovation-based covariance-parameter estimation approach of [22].

Filtering estimates at time $t _ { i }$ use observations only up to that time. For closure learning, however, it is useful for the inferred closure labels to reflect information from the full assimilated time window. We therefore apply a fixed-interval ensemble smoother [23]. The smoother retrospectively updates stored ensembles at previous output times using later observation innovations and ensemble-estimated temporal correlations. After all observations in the assimilation window have been processed, the resulting smoothed ensemble provides closure-consistent estimates of both the resolved state and the latent closure.

## 2.3.2 Step 2: Ofline closure training

The full smoothed ensemble contains more information than its mean. In particular, the ensemble spread provides an estimate of uncertainty in both the resolved state and the inferred closure. This uncertainty could be propagated into the learned closure model by training an uncertainty-aware regressor on the ensemble rather than only on the ensemble mean. For example, the ensemble at each time could be used to define a local likelihood for Bayesian neural-network training, or multiple closure networks could be trained on ensemble-derived samples to form an ensemble closure model. Such approaches would allow uncertainty in the data-assimilated closure estimates to be carried into the final autonomous predictions. In this work, however, we use only the smoothed ensemble mean. This deterministic reduction is chosen to maintain a direct comparison with the deterministic embedded neural-network closure.

The data-assimilated closure-training pairs are

$$
\mathcal { D } _ { \mathrm { D A } } = \left\{ \left( \bar { x } _ { i } ^ { s } , \bar { \xi } _ { i } ^ { s } \right) : i \in \mathcal { I } \right\} ,\tag{10}
$$

where $\bar { x } _ { i } ^ { s }$ and $\bar { \xi } _ { i } ^ { s }$ denote the smoothed ensemble means of the resolved state and inferred closure, respectively. A neural network $\xi _ { \theta }$ is then trained ofline to approximate the map from resolved state to inferred closure:

$$
\theta ^ { * } = \mathop { \mathrm { a r g m i n } } _ { \theta } \frac { 1 } { 2 } \sum _ { i \in \mathcal { I } } \left| \xi _ { \theta } ( \bar { x } _ { i } ^ { s } ) - \bar { \xi } _ { i } ^ { s } \right| ^ { 2 } .\tag{11}
$$

After training, the learned closure is inserted into the original dynamical model for new predictions:

$$
\frac { d x ( t ) } { d t } = R ( x ( t ) , \xi _ { \theta ^ { * } } ( x ( t ) ) ) .\tag{12}
$$

## 3 Comparison Methodology

This section defines the experimental protocol used to compare embedded closure training with staged dataassimilated closure training. The general mathematical approach for both methods is given in Section 2. Here, we specify how the observation data are split, how each method uses those data, how neural-network training is performed, and how accuracy, cost, and robustness are measured.

## 3.1 Data usage and held-out evaluation

Both methods are evaluated using the same noisy observations and the same randomized train, validation, and test time-index split. The observation-time indices are partitioned into $\mathcal { T } _ { \mathrm { t r a i n } } , ~ \mathcal { T } _ { \mathrm { v a l } } .$ , and $\mathcal { T } _ { \mathrm { t e s t } }$ . The split fractions are 60%, 20%, and 20%, respectively. The test indices are held out from all training, data assimilation, hyperparameter selection, early stopping, and model-selection procedures.

For embedded training, the neural closure is trained directly against the noisy observations indexed by $\mathcal { T } _ { \mathrm { t r a i n } }$ . The validation observations indexed by $\mathcal { T } _ { \mathrm { v a l } }$ are used for early stopping, random-start selection, and derivative-based local refinement accept-or-reject decisions. The test observations indexed by $\mathcal { T } _ { \mathrm { t e s t } }$ are used only for final evaluation.

For staged training, data assimilation is performed once using the union of the original training and validation observations. Thus, the assimilation index set is

$$
{ \mathcal { T } } _ { \mathrm { a s s i m } } = { \mathcal { T } } _ { \mathrm { t r a i n } } \cup { \mathcal { T } } _ { \mathrm { v a l } } .\tag{13}
$$

The original test observations are never assimilated. The closure process-noise parameter $\sigma _ { \xi }$ is estimated using only observations indexed by $\mathcal { I } _ { \mathrm { a s s i m } }$ . After the EnKF and smoother are run, pseudo-training pairs are selected at the original training times and pseudo-validation pairs are selected at the original validation times:

$$
\begin{array} { r } { \mathcal { D } _ { \mathrm { p s e u d o - t r a i n } } = \left\{ \left( \bar { x } _ { i } ^ { s } , \bar { \xi } _ { i } ^ { s } \right) : i \in \mathcal { T } _ { \mathrm { t r a i n } } \right\} , } \end{array}\tag{14}
$$

$$
\begin{array} { r } { \mathcal D _ { \mathrm { p s e u d o - v a l } } = \left\{ \left( \bar { x } _ { i } ^ { s } , \bar { \xi } _ { i } ^ { s } \right) : i \in \mathcal { T } _ { \mathrm { v a l } } \right\} . } \end{array}\tag{15}
$$

The prefix “pseudo” emphasizes that the supervised learning targets $\bar { \xi } _ { i } ^ { s }$ are not directly observed closure values, but are inferred quantities produced by the EnKF and smoother. The staged approach uses the original train–validation index split to organize these inferred pairs, but the pseudo-training and pseudovalidation sets are not independent raw datasets. Because both sets are produced by a single assimilation and smoothing procedure over $\mathcal { I } _ { \mathrm { a s s i m } }$ , they are temporally correlated through the forecast dynamics, ensemble covariances, and retrospective smoothing updates. Thus, pseudo-validation is used only for internal staged neural-network selection, such as early stopping or random-start selection decisions. It is not treated as an independent measure of predictive performance. Final accuracy comparisons are made only on the original held-out observations indexed by $\mathcal { T } _ { \mathrm { t e s t } }$ , which are excluded from assimilation, training, validation, and model selection.

## 3.2 Closure-model training objectives

Both approaches use the same parametric closure-model class. They difer in the objective used to construct the closure and in whether the dynamical-system simulator is invoked during closure-model training.

In the embedded approach, the closure is trained according to its efect on the observed state trajectory. For a candidate closure $\xi _ { \theta } .$ , the model trajectory $x ( t ; \theta )$ is obtained by solving (4). The embedded training objective is the observation-space negative log-likelihood over the training indices, up to additive constants independent of θ:

$$
\mathcal { L } _ { \mathrm { e m b } } ( \theta ) = \frac { 1 } { 2 } \sum _ { i \in \mathcal { I } _ { \mathrm { t r a i n } } } \left( d _ { i } - H x ( t _ { i } ; \theta ) \right) ^ { T } \Gamma ^ { - 1 } \left( d _ { i } - H x ( t _ { i } ; \theta ) \right) .\tag{16}
$$

Multistart and stopping criteria evaluations use the analogous observation-space loss over $\mathcal { T } _ { \mathrm { v a l } }$

In the staged approach, the closure is trained as an ofline supervised regression problem using the data-assimilated state–closure pairs. The staged training objective is

$$
\mathcal { L } _ { \mathrm { s t a g e d } } ( \theta ) = \frac { 1 } { 2 } \sum _ { i \in \mathcal { I } _ { \mathrm { t r a i n } } } \left. \xi _ { \theta } \bigl ( \bar { x } _ { i } ^ { s } \bigr ) - \bar { \xi } _ { i } ^ { s } \right. ^ { 2 } .\tag{17}
$$

Multistart and stopping criteria evaluations use the analogous supervised loss over $\mathcal { D } _ { \mathrm { p s e u d o - v a l } }$ . This supervised training step does not call the dynamical-system simulator. After training, the selected staged closure is embedded into the governing equations for future predictions.

## 3.3 Performance metrics

Both approaches are evaluated on the same held-out test observations. The primary calibration metric is the likelihood-scaled observation space test loss

$$
\mathcal { L } _ { \mathrm { t e s t } } = \frac { 1 } { 2 } \sum _ { i \in \mathcal { I } _ { \mathrm { t e s t } } } \left( d _ { i } - H x _ { \mathrm { p r e d } } ( t _ { i } ) \right) ^ { T } \Gamma ^ { - 1 } \left( d _ { i } - H x _ { \mathrm { p r e d } } ( t _ { i } ) \right) ,\tag{18}
$$

where $x _ { \mathrm { p r e d } } ( t _ { i } )$ denotes the state trajectory produced by the trained closure model. This metric evaluates predictive agreement with raw observations that were not used during training, data assimilation, validation, or model selection.

Because the numerical study is synthetic, the true closure is known and can be used for diagnostic evaluation. For an index set I, the closure error is measured by

$$
\mathrm { R M S E } _ { \xi } ( \mathcal { T } ) = \left( \frac { 1 } { | \mathcal { T } | } \sum _ { i \in \mathcal { T } } | \xi _ { \theta } ( x _ { \mathrm { t r u e } } ( t _ { i } ) ) - \xi _ { \mathrm { t r u e } } ( x _ { \mathrm { t r u e } } ( t _ { i } ) ) | ^ { 2 } \right) ^ { 1 / 2 } .\tag{19}
$$

This closure RMSE is a diagnostic metric and would not be available in a real closure-learning application where $\xi _ { \mathrm { t r u e } }$ is unknown.

To assess generalization, trained closures are also embedded into systems with parameter values diferent from those used during calibration. For each of these validation scenarios, state prediction error is measured by

$$
\mathrm { R M S E } _ { x } = \left( \frac { 1 } { N _ { t } } \sum _ { i = 0 } ^ { N _ { t } - 1 } \left. \boldsymbol { x } _ { \mathrm { p r e d } } ( t _ { i } ) - \boldsymbol { x } _ { \mathrm { t r u e } } ( t _ { i } ) \right. _ { 2 } ^ { 2 } \right) ^ { 1 / 2 } .\tag{20}
$$

Closure prediction error in each validation scenario is measured analogously using ${ \mathrm { R M S E } } _ { \xi }$ over the validation trajectory.

## 3.4 Computational cost and robustness accounting

Computational cost is measured by the number of computational model evaluations required to construct each closure model. This accounting is intended to isolate the portion of each workflow that would become expensive for large, legacy, or high-fidelity simulators.

For embedded training, the cost includes all forward solves of the dynamical system used during closuremodel optimization and model selection. Each evaluation of the embedded training or validation objective requires a computational model evaluation with the current closure parameters. Thus, embedded cost scales directly with the number of candidate closure models evaluated during training.

For staged training, the cost includes computational model evaluations used during the data-assimilation stage, including any evaluations used to select data-assimilation hyperparameters and the final assimilation and smoothing pass. The subsequent supervised closure-regression step is not included in the total cost because it evaluates only the ofline regression objective on the assimilated state–closure pairs. This distinction reflects the central computational diference between the two workflows: embedded training repeatedly calls the simulator during neural-network optimization, whereas staged training shifts simulator use into a preprocessing stage and then performs closure-function training ofline.

Robustness is measured by the number of failed candidate training runs. A candidate is counted as failed if it produces a solver failure, nonfinite state, or nonfinite loss. This metric is included because embedded optimization can generate intermediate closures that alter the dynamical system in unstable or nonphysical ways. In contrast, after the data-assimilation stage, the staged neural-network optimization is a supervised regression problem, so unstable intermediate closure models do not impact computational model performance during training.

## 3.5 Replicate analysis

To assess the variability in comparisons of accuracy and cost, we perform replicate trainings herein, where each replicate features a new noisy observation realization, a new random train-validation-test split, and

a new seed for multistart neural-network training. Both methods are evaluated on the same held-out test observations per replicate realization. These replicate results are aggregated and presented in the numerical results in Section 4.

## 4 Numerical study

## 4.1 SIQR epidemiological exemplar

We compare embedded and staged closure training using a synthetic compartmental epidemiological model with susceptible, infectious, quarantined, and removed compartments. The state is

$$
\begin{array} { r } { \boldsymbol { x } ( t ) = \left[ \boldsymbol { s } ( t ) \quad \boldsymbol { i } ( t ) \quad \boldsymbol { q } ( t ) \quad \boldsymbol { r } ( t ) \right] ^ { T } , } \end{array}\tag{21}
$$

where each component is a population fraction. The true dynamics are

$$
\frac { d s } { d t } = - \beta s i ,\tag{22}
$$

$$
\frac { d i } { d t } = \beta s i - \gamma i - \xi ( s , i ) i ,\tag{23}
$$

$$
\frac { d q } { d t } = \xi ( s , i ) i - \delta q ,\tag{24}
$$

$$
\frac { d r } { d t } = \gamma i + \delta q .\tag{25}
$$

The unknown closure $\xi$ controls the transition from infectious to quarantined individuals. The true synthetic closure is

$$
\xi ( s ( t ) , i ( t ) ) = s ( t ) ^ { 1 / 2 } \cdot \frac { 1 } { 2 } \left[ 1 + \operatorname { t a n h } \left( \frac { i ( t ) N _ { \mathrm { p o p } } - 3 \times 1 0 ^ { 5 } } { 0 . 0 5 N _ { \mathrm { p o p } } } \right) \right] .\tag{26}
$$

This closure represents a smooth, saturating quarantine response. When the infectious population count is low, the transition into quarantine is small. As infections increase, the quarantine rate rises, representing increased testing, self-isolation, or public-health response. The hyperbolic tangent produces a smooth transition near $3 \times 1 0 ^ { 5 }$ infectious individuals. The dependence on $s ( t ) ^ { \bar { 1 } / 2 }$ reduces the quarantine response as the susceptible pool is depleted.

For the dataset used in training, the population size is $N _ { \mathrm { p o p } } = 2 \times 1 0 ^ { 6 }$ . The initial counts are 500 infectious individuals, 10 quarantined individuals, 10 removed individuals, and the remainder susceptible. The nondimensional initial condition is therefore

$$
i ( 0 ) = \frac { 5 0 0 } { N _ { \mathrm { p o p } } } ,\tag{27}
$$

$$
q ( 0 ) = \frac { 1 0 } { N _ { \mathrm { p o p } } } ,\tag{28}
$$

$$
r ( 0 ) = \frac { 1 0 } { N _ { \mathrm { p o p } } } ,\tag{29}
$$

$$
s ( 0 ) = 1 - i ( 0 ) - q ( 0 ) - r ( 0 ) .\tag{30}
$$

The transition rate parameters in the model are set at $\beta = 0 . 1 5 , \gamma = 0 . 0 1 3$ , and $\delta = 0 . 0 1$ . The time evolution of this system is shown in Figure 1.

Noisy observations are generated as

$$
d _ { i } = H x ( t _ { i } ) + \epsilon _ { i } , \qquad \epsilon _ { i } \sim \mathcal { N } ( 0 , \sigma _ { \epsilon } ^ { 2 } I ) ,\tag{31}
$$

with $H = I \in \mathbb { R } ^ { 4 }$ . The measurement noise is assumed known and is set to $\sigma _ { \epsilon } ^ { 2 } = 1 0 ^ { - 6 }$

![](images/aa5b33dd4b2928cffc5b1f72253be701c6191746677a19d9d188f0c7aff9c353.jpg)

![](images/48aa089e40fb1e94f80f564cba7a10d360ca85a6ddca4892cb73b47d834d4551.jpg)  
Figure 1: True SIQR system dynamics and closure trajectory used to generate the synthetic observations.

## 4.1.1 SIQR embedded closure model

The embedded approach replaces the unknown quarantine closure with a neural network $\xi _ { \theta }$ . The embedded SIQR dynamics are

$$
\frac { d s } { d t } = - \beta s i ,\tag{32}
$$

$$
\frac { d i } { d t } = \beta s i - \gamma i - \xi _ { \theta } { \left( s , i , q , r \right) } i ,\tag{33}
$$

$$
\frac { d q } { d t } = \xi _ { \theta } ( s , i , q , r ) i - \delta q ,\tag{34}
$$

$$
\frac { d r } { d t } = \gamma i + \delta q .\tag{35}
$$

The positive sign in the quarantined equation reflects that the learned closure transfers individuals from the infectious compartment into the quarantined compartment. The state variable trajectory induced by a given neural network parameter vector $\theta$ is denoted by $x ( t ; \theta )$ . The network is trained against noisy state observations using the embedded trajectory loss described in Section 3.2. Training, validation, random-start selection, and local-refinement accept-or-reject decisions follow the comparison protocol in Section 3. The specific optimizer, initialization, and solver settings used in this study are given in Section 4.2.

## 4.1.2 SIQR staged data-assimilated closure model

For the staged data-assimilated approach, the SIQR state is augmented with a time-dependent latent closure variable. The augmented state is

$$
\hat { z } ( t ) = \left[ s ( t ) \quad i ( t ) \quad q ( t ) \quad r ( t ) \quad \hat { \xi } ( t ) \right] ^ { T } .\tag{36}
$$

The augmented dynamics used during data assimilation are

$$
\frac { d s } { d t } = - \beta s i ,\tag{37}
$$

$$
\frac { d i } { d t } = \beta s i - \gamma i - \hat { \xi } ( t ) i ,\tag{38}
$$

$$
\frac { d q } { d t } = \hat { \xi } ( t ) i - \delta q ,\tag{39}
$$

$$
\frac { d \boldsymbol { r } } { d t } = \boldsymbol { \gamma } \boldsymbol { i } + \boldsymbol { \delta q } ,\tag{40}
$$

$$
d \hat { \xi } ( t ) = \sigma _ { \xi } d W ( t ) .\tag{41}
$$

The EnKF ensemble size is $N _ { \mathrm { e n s } } = 1 0 0$ . The closure process noise is applied only to the closure component. The observation covariance is $\Gamma = \sigma _ { \epsilon } ^ { 2 } I _ { 4 }$ . The assimilation uses observations indexed by $\mathcal { T } _ { \mathrm { a s s i m } } = \mathcal { T } _ { \mathrm { t r a i n } } \cup \mathcal { T } _ { \mathrm { v a l } }$ The original test observations indexed by $\mathcal { T } _ { \mathrm { t e s t } }$ are not assimilated.

The EnKF and fixed-interval smoother produce smoothed ensemble means $\bar { z } _ { i } ^ { s } = [ \bar { x } _ { i } ^ { s } , \bar { \xi } _ { i } ^ { s } ] ^ { T }$ . For the SIQR problem, the smoothed physical-state mean is $\bar { x } _ { i } ^ { s } = [ \bar { s } _ { i } ^ { s } , \bar { i } _ { i } ^ { s } , \bar { q } _ { i } ^ { s } , \bar { r } _ { i } ^ { s } ] ^ { \top }$ . The staged neural network is trained ofline using the supervised loss

$$
\mathcal { L } _ { \mathrm { s t a g e d } } ( \theta ) = \frac { 1 } { 2 } \sum _ { i \in \mathcal { I } _ { \mathrm { t r a i n } } } \left| \xi _ { \theta } \big ( \bar { s } _ { i } ^ { s } , \bar { i } _ { i } ^ { s } , \bar { q } _ { i } ^ { s } , \bar { r } _ { i } ^ { s } \big ) - \bar { \xi } _ { i } ^ { s } \right| ^ { 2 } .\tag{42}
$$

The pseudo-validation loss is computed using the same expression over $\mathcal { T } _ { \mathrm { v a l } }$ . After training, the learned staged closure is embedded into the SIQR equations for further predictions:

$$
\frac { d s } { d t } = - \beta s i ,\tag{43}
$$

$$
\frac { d i } { d t } = \beta s i - \gamma i - \xi _ { \theta } { \left( s , i , q , r \right) } i ,\tag{44}
$$

$$
\frac { d q } { d t } = \xi _ { \theta } ( s , i , q , r ) i - \delta q ,\tag{45}
$$

$$
\frac { d r } { d t } = \gamma i + \delta q .\tag{46}
$$

## 4.2 Numerical implementation

The code used to generate the numerical results in this section is available at https://github.com/ sandialabs/STAGED\_DDCM-Staged\_Embedded\_Comparison. All numerical experiments are implemented in Python using NumPy, SciPy, and packages in the JAX ecosystem. Synthetic SIQR trajectories and noisy observation data are generated using SciPy’s solve ivp method with the RK45 timestepper. The true sys tem is solved on the time interval [0, 200] days. Observations are stored every $\Delta t = 0 . 2 5$ days, giving 801 observation times including both endpoints.

The embedded and staged closure networks use the same architecture, which was selected through a manual search using the supervised training data from the staged approach. Each network takes the four instantaneous state variables $( s , i , q , r )$ as input and has two hidden layers with two nodes per layer, sigmoid hidden-layer activations, and one scalar output. The output layer is unconstrained. Weights are initialized using Xavier uniform initialization and biases are initialized to zero. Both approaches use 10 random starts.

Neural-network training uses a two-stage derivative-based local optimization procedure. Each random start is first trained with AdamW, an adaptive first-order method [24]. The resulting candidate is then refined using L-BFGS [25] as a quasi-Newton local polishing step. The L-BFGS-refined candidate is accepted only if it improves the relevant validation objective. For embedded training, this validation objective is the raw-observation validation loss. For staged training, this validation objective is the EnKF-derived pseudovalidation loss.

AdamW uses learning rate $1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 6 }$ , and gradient clipping with maximum global norm 1. Each AdamW run is allowed at most 2000 iterations. Validation is performed every 100 AdamW iterations with early-stopping patience of 4 validation checks, so training stops after at most 400 AdamW iterations without validation improvement. After AdamW, L-BFGS is run for at most 500 iterations.

Embedded closure simulations are computed using Difrax with the Dopri5 solver and an adaptive step size controller. The same Difrax solver settings are used when the trained staged closure is embedded into the SIQR equations for predictions after training. A neural-network start is counted as failed if the ODE solve fails, produces nonfinite states or losses, or returns the large solver-failure penalty used by the training code.

The EnKF and smoother are implemented in custom NumPy code. The EnKF ensemble size is $N _ { \mathrm { e n s } } =$ 100. The augmented SIQR forecast model uses a forward-Euler update with internal step size $\Delta t _ { \mathrm { i n t } } = 0 . 2 5$ days. The initial closure value in the augmented state is $\hat { \xi } ( 0 ) = 0$ . The closure process noise is applied only to the closure component of the augmented state.

For the staged approach, the EnKF assimilates only the original training and validation observations. The original test observations are never assimilated. The closure process-noise scale $\sigma _ { \xi }$ is estimated separately for each replicate using an EnKF likelihood computed from the training and validation observations. For this likelihood optimization, every tenth training-or-validation observation is used. The process-noise optimization uses SciPy’s minimize with the Nelder–Mead method, initial value $2 . 0 \times 1 0 ^ { - 5 }$ , bounds $[ 1 0 ^ { - 8 } , 0 . 5 ]$ , and tolerance $1 0 ^ { - 5 }$ . After estimating $\sigma _ { \xi } .$ , a final EnKF assimilation is performed using all training and validation observations. Smoothing is employed in the staged experiments. The staged neural network is trained using only the EnKF-derived state–closure pairs and does not call the SIQR simulator during supervised optimization.

For each replicate in the study, independent random-number streams are used for observation noise, train–validation–test splitting, EnKF sampling, embedded neural-network initialization, and staged neuralnetwork initialization. Specifically, replicate-dependent seeds are used for observation noise, data splitting, EnKF sampling, embedded training starts, and staged training starts.

## 4.3 Calibration comparison

The study uses 50 independent replicates. Both methods are evaluated on the same held-out test observations for each replicate. The reported test loss is the likelihood-scaled observation-space loss defined in Section 3.3. Because the study is synthetic, we also report closure RMSE on the test indices. This closure RMSE is a diagnostic metric and would not be available in a real closure-modeling application.

Figure 2 compares the learned state trajectories from the embedded and staged approaches across replicates. Both methods reproduce the qualitative SIQR dynamics quite well. Figure 3 compares the learned closure trajectories with the true synthetic closure. Again, both methods recover the broad temporal structure of the closure.

![](images/b329cc26da8094d8cd0ddb5507b94843ae17391e68a885738faae9e5f676ba8c.jpg)  
Figure 2: Embedded and staged state trajectories across 50 replicates compared with the true SIQR trajectory.

Figure 4 summarizes calibration accuracy across the 50 replicates, where calibration performance is reported using held-out observation-space test loss and synthetic closure RMSE on the calibration test indices. The embedded approach achieves the lower average held-out test loss in this example, with mean test loss $3 . 9 2 \times 1 0 ^ { 2 } \pm 1 . 3 5 \times 1 0 ^ { 2 }$ . The staged approach has a larger mean held-out test loss of $1 . 7 5 \times 1 0 ^ { 3 } \pm 1 . 2 3 \times 1 0 ^ { 3 }$ The embedded approach also has a lower average calibration closure RMSE of $1 . 1 8 \times 1 0 ^ { - 3 } \pm 1 . 1 7 \times 1 0 ^ { - 3 } ,$ compared with $1 . 7 0 \times 1 0 ^ { - 3 } \pm 7 . 4 2 \times 1 0 ^ { - 4 }$ for the staged approach. Although the embedded approach is more accurate on average, the staged errors remain within the same order of magnitude for this calibration problem. Thus, the calibration results indicate a measurable but not prohibitive loss in average accuracy for the staged approach.

![](images/767fff2158e5a58360caf85c37842027d985140a2edfd7ace24c19684a530209.jpg)  
Figure 3: Learned closure trajectories from embedded and staged replicates compared with the true synthetic closure.

![](images/6fc17430d86c5a0608f6edc1c70dd20191854443ace050d93e1af14363a170d8.jpg)  
Figure 4: Distribution of calibration errors over the 50 independent replicates for the embedded and staged approaches. Each violin shows the empirical distribution of the replicate-level error metric for one method, with width indicating relative density. Horizontal lines inside each violin indicate the $2 5 ^ { \mathrm { t h } } , 5 0 ^ { \mathrm { t h } }$ , and $7 5 ^ { \mathrm { t h } }$ percentiles of the replicate distribution.

The computational-cost and robustness comparison presented in Table 1 favors the staged approach. The embedded approach requires $2 . 4 1 \times 1 0 ^ { 4 } \pm 7 . 4 1 \times 1 0 ^ { 3 }$ training-stage model evaluations. The staged approach requires $2 . 7 8 \times 1 0 ^ { 3 } \pm 3 . 5 9 \times 1 0 ^ { 2 }$ training-stage model evaluations. This corresponds to an approximately 9× reduction in model evaluations. The robustness comparison also favors the staged approach. The embedded approach has 215 failed neural-network starts across the 50 replicates with 10 multi-starts each, corresponding to a roughly 40% failure rate, whereas the staged approach has no failed starts.

<table><tr><td>Method</td><td>Training model evaluations</td><td>Failed starts</td></tr><tr><td>Embedded</td><td> $\overline { { 2 . 4 1 \times 1 0 ^ { 4 } \pm 7 . 4 1 \times 1 0 ^ { 3 } } }$ </td><td>215</td></tr><tr><td>STAGED DDCM</td><td> $2 . 7 8 \times 1 0 ^ { 3 } \pm 3 . 5 9 \times 1 0 ^ { 2 }$ </td><td>0</td></tr></table>

Table 1: Comparison of embedded and staged closure-training approaches over 50 replicates. Computational cost is measured by training-stage model evaluations, where results are presented as $\mu \pm \sigma$ over the 50 replicates. Robustness is measured by the total number of failed neural-network starts.

As shown in Table 2, the staged cost is dominated by process-noise maximum-likelihood estimation. The process-noise MLE requires $2 . 6 8 \times 1 0 ^ { 3 } \pm 3 . 5 9 \times 1 0 ^ { 2 }$ model evaluations, while the final EnKF assimilation requires 100 model evaluations. The supervised staged neural-network training is ofline and does not invoke the SIQR simulator. This result indicates that further eficiencies may be gained in the process-noise MLE step of the data assimilation process.

<table><tr><td>Component</td><td>Model evaluations</td></tr><tr><td>Process-noise MLE</td><td> $2 . 6 8 \times 1 0 ^ { 3 } \pm 3 . 5 9 \times 1 0 ^ { 2 }$ </td></tr><tr><td>Final EnKF assimilation</td><td> $1 . 0 0 \times 1 0 ^ { 2 } \pm 0$ </td></tr><tr><td>Total STAGED DDCM</td><td> $2 . 7 8 \times 1 0 ^ { 3 } \pm 3 . 5 9 \times 1 0 ^ { 2 }$ </td></tr></table>

Table 2: Training-stage model-evaluation accounting for the staged approach. The total is the sum of model evaluations used for process-noise maximum-likelihood estimation and the final EnKF assimilation. The subsequent supervised neural-network training step does not call the SIQR simulator and is therefore excluded from this simulator-dependent cost.

## 4.4 Validation-scenario generalization comparison

We also evaluate how the learned closures generalize when embedded into SIQR systems with parameter values diferent from the calibration scenario. The validation scenarios are

$$
\mathrm { C a s e ~ 1 } \mathrm { : } \quad \beta = 0 . 3 0 , \quad \gamma = 0 . 0 2 , \quad \delta = 0 . 0 1 ,\tag{47}
$$

$$
\mathrm { C a s e \ 2 } \mathrm { : } \quad \beta = 0 . 1 0 , \quad \gamma = 0 . 0 1 , \quad \delta = 0 . 0 1 ,\tag{48}
$$

$$
\mathrm { C a s e ~ 3 } \mathrm { : } \quad \beta = 0 . 1 5 , \quad \gamma = 0 . 1 0 , \quad \delta = 0 . 0 1 .\tag{49}
$$

These cases and their relation to the calibration scenario are illustrated through a susceptible-infectious phase diagram in Figure 5. Case 1 is a more transmissible scenario with a shorter infectious period. Case 2 remains closer to the calibration trajectory and is closer to an interpolation regime in state space. Case 3 induces stronger extrapolation through a substantially accelerated recovery rate.

![](images/ebb62837db9b23ba546559c2e978ef0e6d2207f54c02c19be186de2afd3319ad.jpg)  
Figure 5: Calibration and validation scenario trajectories in the susceptible-infectious projection.

Figures 6 and 7 show state and closure trajectories for the validation scenarios. Both embedded and staged closures perform similarly across scenarios, with the greatest qualitative variability in results for Case 3, which constitutes the greatest extrapolation from the calibration scenario. Qualitative variability seems similar for the other two cases, which are closer to the calibration scenario.

Figure 8 summarizes validation scenario accuracy across the 50 replicates. The validation metrics are consistent with the calibration results but also show that the staged approach can remain competitive. For Case 1, the embedded approach has mean state RMSE $9 . 4 3 \times 1 0 ^ { - 3 }$ and mean closure $\mathrm { R M S E ~ 5 . 0 9 } \times 1 0 ^ { - 3 }$ . For the same case, the staged approach has mean state RMSE $1 . 1 5 \times 1 0 ^ { - 2 }$ and mean closure RMSE $5 . 7 4 \times 1 0 ^ { - 3 }$ These values are slightly larger for the staged approach but are close in magnitude, and the median errors in Case 1 are favorable to the staged approach. For Case 2, the embedded approach has lower mean and median state and closure errors. For Case 3, both methods become less accurate, and the embedded approach again has lower average errors. Overall, the validation scenarios suggest that staged training can produce useful closures outside the calibration trajectory, but with greater replicate-to-replicate variability than embedded training in this example.

Closure, validation scenarios  
![](images/e196b10c0053a5711936b80cf7f69b5ecbb70271654d1ff29827986d1d4e9447.jpg)

Population ratios, validation scenarios  
![](images/0ea16f94f2d17d7db084dc339c8d2f03ace252fb96acbef912c3d29af6bd4e9f.jpg)  
Figure 6: Embedded and staged state trajectories under validation scenarios.  
Figure 7: Embedded and staged learned closure trajectories under validation scenarios.

## 4.5 Discussion

Taken together, these results show an accuracy–eficiency tradeof in the SIQR exemplar. The embedded approach is more accurate on average for this inexpensive low-dimensional system, but the staged approach attains the same qualitative behavior with substantially fewer model evaluations and no failed neural-network starts. The staged approach therefore sacrifices some average accuracy in exchange for lower computational cost and improved optimization robustness.

Validation scenario errors

![](images/3753776152d678e3cb390be138110a6a0b610564ea49f464660ce4d33dbb7467.jpg)  
Figure 8: Distribution of validation scenario errors over the 50 independent replicates for the embedded and staged approaches. Each violin shows the empirical distribution of the replicate-level error metric for one method, with width indicating relative density. Horizontal lines inside each violin indicate the $2 5 ^ { \mathrm { t h } } , 5 0 ^ { \mathrm { t h } }$ and $7 5 ^ { \mathrm { t h } }$ percentiles of the replicate distribution.

Whether this sacrifice is worthwhile depends on the computational setting. For inexpensive diferentiable models, embedded training may be preferable. For expensive or nondiferentiable models, or for models where numerical instability can be easily induced, the reduced model-evaluation cost and improved robustness of staged training may be more important than the moderate increase in error observed here. Because this comparison is based on one synthetic SIQR example, the numerical values should not be interpreted as universal. Instead, they illustrate the type of accuracy–cost trade-of that can arise when replacing embedded neural-closure training with a staged data-assimilated workflow.

## 5 Conclusions

This work compared embedded and staged data-assimilated approaches for learning neural closure models in dynamical systems, focusing on accuracy, computational cost, and optimization robustness. Embedded training optimizes the closure through trajectory-level mismatch, whereas staged training uses data assimilation to infer closure-consistent trajectories before fitting the closure through ofline supervised regression.

In the synthetic SIQR example considered here, embedded training produced the lowest average held-out calibration error and the lowest average diagnostic closure error. However, the staged approach produced qualitatively similar trajectories and closure behavior with errors that were often comparable in magnitude. Thus, the results should be interpreted as an accuracy–eficiency tradeof rather than as a definitive ranking of the two approaches. For this low-dimensional and inexpensive exemplar, the accuracy advantage of embedded training may justify its additional computational cost. For more expensive, nondiferentiable, legacy, or numerically fragile simulators, the reduced model-evaluation cost and improved robustness of staged training may be more important than the moderate increase in error observed in this example.

The computational and robustness diferences were substantial. Across 50 independent replicates, staged training required approximately 9× fewer training-stage model evaluations and eliminated failed neuralnetwork starts. In contrast, embedded training yielded an approximately 40% training failure rate due numerical challenges caused by intermediate closure model parameter guesses leading to instability, solver failures, etc. Staged training avoids this issue by separating computational model evaluations during data

assimilation from ofline neural-network optimization.

Both approaches produced generally stable predictions under parameter perturbations. Embedded training generally achieved lower mean state and closure errors, while staged training remained competitive in some scenarios and captured meaningful features of the nonlinear quarantine response. The larger staged mean errors were partly driven by outliers, indicating greater replicate-to-replicate variability. These findings suggest that staged learning can generalize beyond the calibration trajectory, although prediction errors may accumulate when trajectories depart from the calibration manifold.

Future work should extend the comparison to more complex and high-dimensional systems, incorporate uncertainty in the inferred closure, and use multiple calibration scenarios to improve generalization. Further investigation should also account for architecture selection and hyperparameter tuning, particularly the potential to reuse assimilated datasets for ofline model development without additional simulator evaluations. These extensions will help establish when the computational and robustness advantages of staged training outweigh the accuracy gains of embedded optimization.

## Acknowledgments

Sandia National Laboratories is a multi-mission laboratory managed and operated by National Technology & Engineering Solutions of Sandia, LLC (NTESS), a wholly owned subsidiary of Honeywell International Inc., for the U.S. Department of Energy’s National Nuclear Security Administration (DOE/NNSA) under contract DE-NA0003525. This written work is authored by an employee of NTESS. The employee, not NTESS, owns the right, title and interest in and to the written work and is responsible for its contents. Any subjective views or opinions that might be expressed in the written work do not necessarily represent the views of the U.S. Government. The publisher acknowledges that the U.S. Government retains a nonexclusive, paid-up, irrevocable, world-wide license to publish or reproduce the published form of this written work or allow others to do so, for U.S. Government purposes. The DOE will provide public access to results of federally sponsored research in accordance with the DOE Public Access Plan. This paper describes objective technical results and analysis. Any subjective views or opinions that might be expressed in the paper do not necessarily represent the views of the U.S. Department of Energy or the United States Government. This work was supported by the Laboratory Directed Research and Development program (project 233077) at Sandia National Laboratories, a multimission laboratory managed and operated by National Technology and Engineering Solutions of Sandia LLC, a wholly owned subsidiary of Honeywell International Inc. for the U.S. Department of Energy’s National Nuclear Security Administration under contract DE- NA0003525.

## References

[1] Vishal Srivastava and Karthik Duraisamy. Generalizable physics-constrained modeling using learning and inference assisted by feature-space engineering. Physical Review Fluids, 6(12):124602, 2021.

[2] Paul A Durbin. Some recent developments in turbulence closure modeling. Annual Review of Fluid Mechanics, 50:77–103, 2018.

[3] K Hanjali´c. Advanced turbulence closure models: a view of current status and future prospects. International Journal of Heat and Fluid Flow, 15(3):178–203, 1994.

[4] Guy L Bergel, David Montes de Oca Zapiain, and Vicente Romero. Neural network ensembles and uncertainty estimation for predictions of inelastic mechanical deformation using a finite element methodneural network approach. Data-Centric Engineering, 4:e23, 2023.

[5] Jaideep Ray, Matthew Barone, Stefan Domino, Tania Banerjee, and Sanjay Ranka. Verification of data-driven models of physical phenomena using interpretable approximation. Technical report, Sandia National Laboratories (SNL-NM), Albuquerque, NM (United States), 2021.

[6] Shaowu Pan and Karthik Duraisamy. Data-driven discovery of closure models. SIAM Journal on Applied Dynamical Systems, 17(4):2381–2413, 2018.

[7] Eric J. Parish and Karthik Duraisamy. A paradigm for data-driven predictive modeling using field inversion and machine learning. Journal of Computational Physics, 305:758–774, January 2016.

[8] Shaowu Pan and Karthik Duraisamy. Data-Driven Discovery of Closure Models. SIAM Journal on Applied Dynamical Systems, 17(4):2381–2413, January 2018.

[9] Abhinav Gupta and Pierre F. J. Lermusiaux. Neural closure models for dynamical systems. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences, 477(2252):20201004, August 2021.

[10] W.A. Porteous, M.T.P. Laiu, and C.D. Hauck. Data-Driven, Structure-Preserving Approximations to Entropy-Based Moment Closures for Kinetic Equations. Communications in Mathematical Sciences, 21(4):885–913, 2023.

[11] Juntao Huang, Yingda Cheng, Andrew J. Christlieb, and Luke F. Roberts. Machine learning moment closure models for the radiative transfer equation I: Directly learning a gradient based closure. Journal of Computational Physics, 453:110941, March 2022.

[12] Stephan Rasp, Michael S. Pritchard, and Pierre Gentine. Deep learning to represent subgrid processes in climate models. Proceedings of the National Academy of Sciences, 115(39):9684–9689, September 2018.

[13] Eric Parish, David S Ching, Nathan E Miller, Steven J Beresh, and Matthew F Barone. Turbulence modeling for compressible flows using discrepancy tensor-basis neural networks and extrapolation detection. In AIAA SciTech 2023 Forum, page 2126, 2023.

[14] Benjamin Sanderse, Panos Stinis, Romit Maulik, and Shady E. Ahmed. Scientific machine learning for closure models in multiscale problems: A review. Foundations of Data Science, 7(1):298–337, March 2025.

[15] Kyle D Neal, Mohammad Khalil, and Teresa Portone. Investigating model form error estimation for sparse data. In Annual Conference of the PHM Society, volume 16, 2024.

[16] Eric Crislip, Mohammad Khalil, Teresa Portone, Oksana Chkrebtii, and Kyle Neal. Closure term estimation in spatiotemporal models of dynamical systems. arXiv preprint arXiv:2511.20869, 2025.

[17] Abhinav Subramanian and Sankaran Mahadevan. Bayesian estimation of discrepancy in dynamics model prediction. Mechanical Systems and Signal Processing, 123:351–368, 2019.

[18] Abhinav Subramanian and Sankaran Mahadevan. Error estimation in coupled multi-physics models. Journal of Computational Physics, 395:19–37, 2019.

[19] Abhinav Subramanian and Sankaran Mahadevan. Model error propagation in coupled multiphysics systems. AIAA Journal, 58(5):2236–2245, 2020.

[20] Christopher Rackauckas, Yingbo Ma, Julius Martensen, Collin Warner, Kirill Zubov, Rohit Supekar, Dominic Skinner, Ali Ramadhan, and Alan Edelman. Universal diferential equations for scientific machine learning. arXiv preprint arXiv:2001.04385, 2020.

[21] Geir Evensen. The ensemble kalman filter: Theoretical formulation and practical implementation. Ocean dynamics, 53:343–367, 2003.

[22] Dick P Dee. On-line estimation of error covariance parameters for atmospheric data assimilation. Monthly weather review, 123(4):1128–1145, 1995.

[23] Geir Evensen and Peter Jan van Leeuwen. An Ensemble Kalman Smoother for Nonlinear Dynamics. Monthly Weather Review, 128(6):1852–1867, June 2000.

[24] Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization, 2019.

[25] Dong C. Liu and Jorge Nocedal. On the limited memory BFGS method for large scale optimization. Mathematical Programming, 45(1):503–528, August 1989.