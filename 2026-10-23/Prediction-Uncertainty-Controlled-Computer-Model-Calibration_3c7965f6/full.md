# Prediction Uncertainty Controlled Computer Model Calibration

Ryan Heejoon Byun<sup>1,2</sup>, Dayeon Yoon<sup>1,2</sup>, and Won Chang<sup>∗1,2</sup>

<sup>1</sup>Department of Statistics, Seoul National University

<sup>2</sup>Institute for Data Innovation in Science, Seoul National University

## Abstract

Computer-model calibration is usually undertaken in service of prediction, yet neither Bayesian Kennedy–O’Hagan (KOH) calibration nor amortized simulation-based inference is built to control the predictive error of the calibrated simulator. When parameters are weakly identified or the data–model discrepancy is misspecified, the simulator-output bands induced by a nominal posterior level can be far from nominal. We propose a two-layer framework that targets predictive miscoverage directly. First, a conditional flow-matching model learns an amortized posterior for the calibration parameters, trained on simulator outputs perturbed with Gaussian-process discrepancies of varying structure. Second, conformal risk control arranges posterior draws for the target observation into nested candidate sets and selects the smallest one whose fixed-rank pointwise bands attain a prescribed time-averaged miscoverage on a synthetic calibration sample. Under exchangeability of calibration and target observations, the bands control expected miscoverage in finite samples. In a synthetic benchmark, the proposed bands are roughly 20% the width of KOH–MCMC bands with miscoverage near the 10% target. In calibrating an age-structured SEIR model to the 2010 Malawi measles outbreak, they are again sharper and near-nominal, and the weeks they fail to cover coincide with the asymmetric epidemic decline, exposing structural inadequacy of the mechanistic model.

Keywords: Computer model calibration; Conformal risk control; Flow matching; Simulationbased inference

## 1 Introduction

Computer models—deterministic simulators of complex physical, biological, and engineering systems—are indispensable scientific instruments in fields ranging from epidemiology and climate science to hydrology and astrophysics. Their practical utility depends on calibration: learning unknown simulator inputs from observations so that the calibrated simulator can support prediction, intervention design, and scientific inference (Kennedy and O’Hagan, 2001, Bayarri et al., 2007, Higdon et al., 2008, Brynjarsd´ottir and O’Hagan, 2014, Plumlee, 2017). These downstream tasks require more than accurate parameter point estimates; they require predictive uncertainty that is neither misleadingly narrow nor unnecessarily difuse. For example, Lim et al. (2025) developed a more accurate method for estimating epidemic initial conditions but identified wide credible intervals and below-nominal empirical coverage as limitations requiring further work. Motivated by this concern, we focus on predictive uncertainty for simulator outputs: intervals should accurately identify scientifically relevant trajectories.

An increasingly attractive approach to calibration is to learn the inverse map from simulator outputs to input parameters. This inverse-model approach has been developed for highdimensional dependent outputs using deep neural networks (Bhatnagar et al., 2022) and can be viewed within the broader framework of simulation-based inference (SBI), in which simulations rather than a tractable likelihood provide the information needed for inference (Cranmer et al., 2020). Modern neural SBI methods train a conditional density estimator on simulated parameter–output pairs and thereby amortize the cost of inference: after a single training stage, the fitted model can rapidly return an approximate posterior for each new observation (Papamakarios et al., 2019, Papamakarios and Murray, 2016, Greenberg et al., 2019), which makes SBI compelling for repeated, sequential, or time-sensitive calibration problems.

Amortization, however, does not by itself make posterior predictive uncertainty reliable. Under weak parameter identifiability, data–model discrepancy, or approximation error in the learned inverse map, some calibration parameters can be recovered imprecisely, and propagating the resulting parameter distribution through the simulator can produce predictive intervals that are excessively wide, too narrow, or otherwise poorly aligned with their nominal level (Hermans et al., 2022). Standard SBI procedures approximate parameter distributions; they do not control the predictive error of their simulator pushforwards. The conventional Kennedy–O’Hagan (KOH) Bayesian formulation (Kennedy and O’Hagan, 2001) shares this predictive limitation while introducing a computational one: its posterior is commonly explored by Markov chain Monte Carlo (MCMC), which can be expensive and must generally be rerun for every new observation (Bhatnagar et al., 2022, Cranmer et al., 2020). Moreover, nominal posterior probability levels and the repeated-sample coverage of simulator-output bands are diferent quantities, and parameter–discrepancy nonidentifiability or misspecification can further compromise inference (Brynjarsd´ottir and O’Hagan, 2014, Syring and Martin, 2019). Figure 1 previews how KOH– MCMC and a na¨ıve SBI construction can both yield conservative simulator-output bands for the same measles outbreak. We consequently take amortized SBI as our methodological starting point and use MCMC-based KOH calibration as a benchmark.

Our central contribution is to introduce prediction-accuracy-based conformal risk control (CRC) into neural SBI. We combine CRC (Angelopoulos et al., 2024) with flow-matching posterior estimation (FMPE) (Dax et al., 2023)—a scalable neural posterior estimator based on continuous normalizing flows (Lipman et al., 2023) with competitive performance on established SBI benchmarks (Dax et al., 2023, Lueckmann et al., 2021). Rather than accepting a fixed posterior probability region, CRC selects the calibrated parameter set by directly evaluating a predictive miscoverage loss after parameter draws are propagated through the simulator, identifying the region of the parameter space most relevant for accurate prediction. Under exchangeability between the CRC calibration examples and a future target, the resulting simulator-output intervals control expected miscoverage in finite samples. This prediction-driven correction is related in spirit to flow-based conformal forecasting (Lee et al., 2026), but our target is the simulator pushforward of an amortized parameter posterior in a computer-model calibration problem. This prediction-driven calibration is practical only because the posterior is amortized: CRC requires an approximate posterior for each of hundreds of synthetic calibration observations conditioned on the target, which amounts to a few hundred ODE integrations with a trained flow but would demand as many separate MCMC runs under the standard KOH workflow, repeated for every new observation. Amortization is therefore not merely a computational convenience but the condition that makes prediction-accuracy-based risk control feasible.

We also address the discrepancy sensitivity of the inverse map. Following the learning-withnoise strategy of Bhatnagar et al. (2022), we train FMPE on simulator outputs contaminated by draws from a flexible heterogeneous Gaussian-process discrepancy family and generate the separate CRC calibration sample under the corresponding discrepancy mechanism. The framework therefore combines rapid amortized parameter inference with a second, prediction-focused layer that corrects the size of the simulator-output set, and it can be applied to SBI methods other than FMPE.

The remainder of the paper is organized as follows. Section 2 reviews KOH and simulationbased calibration, conformal risk control, and flow matching. Section 3 develops the proposed framework. Section 4 reports a simulation study using the synthetic model of Bhatnagar et al. (2022), Section 5 applies the framework to the Malawi measles outbreak, and Section 6 concludes.

## 2 Background

## 2.1 Kennedy–O’Hagan (KOH) Calibration and Simulation-Based Inference

The standard KOH calibration framework, proposed by Kennedy and O’Hagan (2001) and developed in many subsequent works (see, e.g., Bayarri et al., 2007, Higdon et al., 2008, Tuo and Wu, 2015, Plumlee, 2017, 2019), is based on the observation model

$$
Z = Y ( \theta ) + \delta ,\tag{1}
$$

where $\theta \in \Theta \subset \mathbb { R } ^ { d }$ denotes the calibration parameter vector; $Y ( \theta ) \in \mathbb { R } ^ { T }$ is the deterministic simulator output at T discrete time points or spatial locations; $Z \in \mathbb { R } ^ { T }$ denotes the corresponding observational data; and $\delta \in \mathbb { R } ^ { T }$ is a data–model discrepancy term capturing systematic mismatch between the simulator and reality, including observational errors. Given a prior $p ( \theta )$ and the likelihood $p ( Z \mid \theta )$ induced by (1), KOH calibration targets the posterior $p ( \theta \mid Z ) \propto p ( Z \mid \theta ) p ( \theta )$ , which quantifies parametric uncertainty; propagating posterior draws through $Y ( \theta )$ yields a posterior distribution for the simulator output. This forward formulation is statistically delicate: θ and the discrepancy can be nonidentifiable without careful prior specification or experimental design, and ignoring or misspecifying the discrepancy can bias inference on the physical parameters (Brynjarsd´ottir and O’Hagan, 2014, Tuo and Wu, 2015, Plumlee, 2017). It is also computationally demanding, since a suficiently long MCMC chain must be run for the observation at hand and repeated when a new observation arrives.

A separate issue concerns the predictive meaning of the resulting bands. A nominal posterior probability level describes uncertainty under the assumed model; it does not, by itself, guarantee that a band covers observed outcomes at the same rate under repeated sampling (Syring and Martin, 2019), and conventional KOH calibration contains no step that tunes simulatoroutput bands to a prespecified predictive-miscoverage loss. Figure 1a illustrates this with the KOH–MCMC baseline for the Malawi measles outbreak analyzed in Section 5. Its nominal 90% pointwise band has empirical miscoverage 0.0345 among weeks with at least 300 reported cases (coverage of approximately 96.55%). The broad band, particularly around the epidemic peak, illustrates the cost in sharpness when posterior uncertainty is propagated without an explicit predictive-risk calibration step.

The inverse-model approach reverses the computational direction and learns calibration parameters directly from simulator outputs; Bhatnagar et al. (2022), for example, train a deep neural network on simulated parameter–output pairs for time-series calibration. More broadly, inference from an implicit simulator is described as simulation-based inference (Cranmer et al., 2020). In neural posterior estimation, one simulates pairs $( \theta , x )$ from the joint model and fits a conditional density $q _ { \phi } ( \theta \mid x )$ to approximate $p ( \theta \mid x )$ (Papamakarios and Murray, 2016, Greenberg et al., 2019); the training cost is paid once, after which $q _ { \phi } ( \theta \mid Z )$ can be sampled for any new observation at negligible additional cost. SBI nevertheless changes the computational mechanism, not the predictive objective. If a component of θ is weakly identified, the learned conditional distribution is inaccurate, or the observed discrepancy difers from those represented during training, the approximate posterior may be displaced or overly dispersed; passing all of its draws through $Y ( \theta )$ transfers that error to the output scale. A nominal credible region from KOH–MCMC and a nominal probability region from SBI therefore share a crucial limitation: neither is constructed to guarantee that the corresponding simulator-output interval attains its nominal predictive coverage $( \mathrm { s e e } , \mathrm { e . g . }$ , Hermans et al., 2022, Syring and Martin, 2019).

Figure 1b makes the SBI limitation concrete using approximate posterior samples from a trained flow-matching model for the same Malawi outbreak. We retain the 90% of generated parameter draws with the highest estimated conditional posterior density, pass them through the SEIR simulator, and form pointwise bands from the forward trajectories. The band misses none of the evaluated weeks, but its average pointwise width is 6,078.54 cases, indicating overcoverage (see Section S4 of the Supplementary Material). This distinction motivates the two-layer structure of our framework: flow-matching SBI supplies rapid approximate parameter draws, while CRC chooses among the induced simulator-output sets using prediction error on a separate synthetic calibration sample, targeting the quantity that matters downstream—predictive miscoverage—with explicit finite-sample risk control under exchangeability.

## 2.2 Conformal Risk Control

Conformal risk control (CRC) extends split conformal prediction from ensuring marginal coverage to controlling a user-specified expected loss, retaining finite-sample validity under exchangeability (Angelopoulos et al., 2024, Vovk et al., 2005, Romano et al., 2019, Angelopoulos and Bates, 2023). Let $\mathcal { D } _ { \mathrm { c a l } } : = \{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n }$ denote a separate calibration dataset and $\left( X _ { n + 1 } , Y _ { n + 1 } \right)$ a new test point, where $X _ { i }$ represents the input features and $Y _ { i }$ the response. Throughout, we use the term calibration in two senses: computer model calibration (learning simulator parameters θ) and CRC calibration (using a separate sample $\mathcal { D } _ { \mathrm { c a l } }$ to tune statistical risk).

Let $\Lambda \subseteq \mathbb { R }$ be a nonempty, totally ordered set of admissible threshold values whose largest element $\lambda _ { \mathrm { { m a x } } } = \operatorname* { s u p } \Lambda$ exists, and let $\{ \widehat { C } _ { \lambda } \} _ { \lambda \in \Lambda }$ be a family of prediction sets in which larger λ yields more conservative sets. For each observation i, define a bounded loss

$$
L _ { i } ( \lambda ) : = L \Big ( \widehat { C } _ { \lambda } ( X _ { i } ) , Y _ { i } \Big ) , \qquad L _ { i } ( \lambda ) \in ( - \infty , B ] ,
$$

where B is a known upper bound; $L _ { i } ( \lambda )$ measures the discrepancy between $\widehat { C } _ { \lambda } ( X _ { i } )$ and $Y _ { i } \ { \mathrm { ( e . g . } }$ ， empirical miscoverage). The goal is a data-dependent threshold $\widehat { \lambda }$ such that $\mathbb { E } [ L _ { n + 1 } ( \widehat { \lambda } ) ] \leq \alpha$ for a target level $\alpha \in ( 0 , 1 )$ . Assuming each $L _ { i } ( \lambda )$ is non-increasing and right-continuous in λ with $L _ { i } ( \lambda _ { \operatorname* { m a x } } ) \leq \alpha$ almost surely, let $\begin{array} { r } { \widehat { R } _ { n } ( \lambda ) : = n ^ { - 1 } \sum _ { i = 1 } ^ { n } L _ { i } ( \lambda ) } \end{array}$ be the empirical calibration risk and

![](images/6247c917c3abf5547abd978b1fe262b3fd7612ecec31b41567d634d217b2c05e.jpg)

(a) KOH–MCMC posterior simulator-output band.  
![](images/5e197f24cfed4516f832f7102d4cc34604e514cb5ae00534ae6a6bbbdf3ef2f9.jpg)  
(b) Flow-matching SBI with fixed posterior-density retention and no CRC.

Figure 1: Predictive bands for the 2010 Malawi measles outbreak. Panel (a) shows the nominal 90% pointwise KOH–MCMC band: empirical miscoverage is 3.45% on the evaluated weeks and average pointwise width is 3,526.89 cases. Panel (b) shows a flow-matching SBI band obtained by retaining the highest-density 90% of posterior parameter draws: empirical miscoverage is 0% and average width is 6,078.54 cases. Blue dashed curves are medians of simulator outputs, green dotted curves are band limits, and red solid curves are observed weekly cases; orange shading marks missed observations when present. Miscoverage is evaluated only at weeks with at least 300 observed cases.

select

$$
\widehat { \lambda } = \operatorname* { i n f } \left\{ \lambda \in \Lambda : \ \frac { n } { n + 1 } \widehat { R } _ { n } ( \lambda ) + \frac { B } { n + 1 } \leq \alpha \right\} = \operatorname* { i n f } \left\{ \lambda \in \Lambda : \ \widehat { R } _ { n } ( \lambda ) \leq \alpha _ { \mathrm { a d j } } \right\} ,\tag{2}
$$

where $\alpha _ { \mathrm { a d j } } = ( n + 1 ) \alpha / n - B / n$ is the finite-sample adjusted target level, with $\widehat { \lambda } = \lambda _ { \mathrm { m a x } }$ if the set is empty. Angelopoulos et al. (2024) show that if $L _ { i } ( \lambda ) , i = 1 , \ldots , n + 1$ , are exchangeable, then $\mathbb { E } [ L _ { n + 1 } ( \widehat { \lambda } ) ] \leq \alpha$ . In our setting, CRC thus provides a rigorous mechanism for constructing prediction intervals with controlled expected miscoverage when nominal credible levels cannot be trusted.

## 2.3 Flow Matching

We approximate the target distribution of the calibration parameters using flow matching, which learns a time-dependent vector field transporting a simple reference distribution to a complex target (Lipman et al., 2023). We use the Gaussian conditional probability path

$$
p _ { t } ( x \mid z ) = \mathcal { N } \big ( x ; t z , ( 1 - ( 1 - \sigma _ { \operatorname* { m i n } } ) t ) ^ { 2 } I _ { d } \big ) , \qquad t \in [ 0 , 1 ] ,
$$

where $x \in \mathbb { R } ^ { d }$ is the state variable, $z \sim p _ { \mathrm { d a t a } }$ is a target sample, and $\sigma _ { \operatorname* { m i n } } \in ( 0 , 1 )$ is a small variance floor. The trajectory starts from standard Gaussian noise at $t = 0$ and concentrates around z as $t  1$ , so that marginally $p _ { 0 } ( x ) = \mathcal { N } ( 0 , I _ { d } )$ and $p _ { 1 } ( x ) \approx p _ { \mathrm { d a t a } }$ . A neural network $v _ { t } ^ { \phi } ( x )$ with weights $\phi$ is trained with the conditional flow matching (CFM) objective

$$
\mathcal { L } _ { \mathrm { C F M } } ( \phi ) = \mathbb { E } _ { t \sim \pi , z \sim p _ { \mathrm { d a t a } } , x \sim p _ { t } ( \cdot | z ) } \left[ \left\| v _ { t } ^ { \phi } ( x ) - v _ { t } ^ { \mathrm { f o w } } ( x \mid z ) \right\| ^ { 2 } \right] , \qquad v _ { t } ^ { \mathrm { f o w } } ( x \mid z ) = \frac { z - ( 1 - \sigma _ { \operatorname* { m i n } } ) x } { 1 - ( 1 - \sigma _ { \operatorname* { m i n } } ) t } ,
$$

where $\pi ( t )$ is a sampling distribution over [0, 1] and $v _ { t } ^ { \mathrm { { f l o w } } }$ is the closed-form conditional vector field generating the Gaussian path. This objective is a tractable, simulation-free surrogate for learning the target transport (Lipman et al., 2023, 2024); it is minimized by stochastic gradient descent on minibatch Monte Carlo estimates without solving an ODE during training. After training, approximate samples are generated by integrating the learned ordinary diferential

equation (ODE)

$$
X _ { 0 } \sim { \mathcal { N } } ( 0 , I _ { d } ) , \qquad d X _ { t } = v _ { t } ^ { \phi } ( X _ { t } ) d t , \qquad t \in [ 0 , 1 ] ,
$$

and taking the terminal state $X _ { 1 }$

Because our goal is to sample from the conditional posterior given observed covariates $y \ ( \mathrm { e . g . }$ , simulator outputs), we condition the vector field on y and employ classifier-free guidance (CFG) at sampling time,

$$
\tilde { v } _ { t } ( x \mid y ) = ( 1 - w _ { t } ) v _ { t } ^ { \phi } ( x \mid \emptyset ) + w _ { t } v _ { t } ^ { \phi } ( x \mid y ) , \qquad w _ { t } > 1 ,
$$

where ∅ denotes a learnable unconditional null-embedding (Ho and Salimans, 2022, Zheng et al., 2023): on condition-dropped training examples the raw condition $y = [ A , Z ] \in \mathbb { R } ^ { a + T }$ , with $a =$ dim(A), is replaced by a trainable vector $e _ { \emptyset } \in \mathbb { R } ^ { a + T }$ that is optimized jointly with the network, so the conditional and unconditional fields share all other parameters. A general formulation of flow matching, including marginal and conditional paths, objective equivalence, and guided flows, is provided in Section S1.2 of the Supplementary Material.

## 3 Prediction Uncertainty Controlled Calibration

This section develops our framework for constructing simulation-based calibration sets with finitesample control over expected predictive miscoverage. Throughout, the observed sequence, simulator output, and discrepancy have common dimension T: for $\theta \in \Theta , Y ( \theta ) = ( Y _ { 1 } ( \theta ) , \ldots , Y _ { T } ( \theta ) ) ^ { \top }$ $\delta = ( \delta _ { 1 } , \ldots , \delta _ { T } ) ^ { \top }$ , and $Z = Y ( \theta ) + \delta = ( Z _ { 1 } , \dots , Z _ { T } ) ^ { \top } \in \mathbb { R } ^ { T }$ . When the loss is evaluated only at scientifically relevant coordinates, ${ \mathcal { T } } ( Z ) \subseteq \{ 1 , \ldots , T \}$ denotes their deterministic index set and $T ( Z ) = | { \mathcal { I } } ( Z ) | ;$ ; otherwise $\mathcal { T } ( Z ) = \{ 1 , . . . , T \}$ and $T ( Z ) = T$ . Let $\alpha \in ( 0 , 1 )$ denote the target risk level, n the size of the CRC calibration dataset, and N the number of amortized posterior draws used to construct each candidate parameter set.

To tune the conformal parameter sets, we construct a synthetic CRC calibration dataset $\mathcal { D } _ { n } : = \{ ( \tilde { A } _ { i } , \tilde { Z } _ { i } , \theta _ { i } ) \} _ { i = 1 } ^ { n }$ that mimics the joint distribution of the parameters and the observables: parameters $\theta _ { i }$ are drawn from the trained flow-matching posterior approximation conditioned on the target observation, and synthetic observations $\tilde { Z } _ { i } = Y ( { \theta _ { i } } ) + \tilde { \delta } _ { i }$ are formed with simulated discrepancies $\tilde { \delta } _ { i }$ drawn from a suficiently flexible distribution, since exact knowledge of the discrepancy law is not assumed. In some problems we optionally extract auxiliary summaries $\tilde { A } _ { i } = A ( \theta _ { i } ) \in \mathcal { A }$ , forward-model outputs assumed to be observed exactly. The construction of $\mathcal { D } _ { n }$ is detailed in Section 3.4.2. Section 3.1 presents the miscoverage loss and parameterset constructions, Section 3.3 the conditional flow-matching architecture, and Section 3.4 the discrepancy-simulation strategy.

## 3.1 Miscoverage Risk

For the i-th calibration sample, define $\mathcal { T } _ { i } : = \mathcal { T } ( \tilde { Z } _ { i } )$ and $T _ { i } : = | { \mathcal { T } } _ { i } |$ . Let $\Lambda \subseteq \mathbb { R }$ be a nonempty, totally ordered set of admissible thresholds with largest element $\lambda _ { \mathrm { m a x } } = \operatorname* { s u p } \Lambda$ . Denote the candidate parameter set given an auxiliary summary A and observed data Z by $\mathcal { C } _ { \lambda } ( A , Z ) , \lambda \in \Lambda$ and require the family to be nested: $\lambda \leq \lambda ^ { \prime }$ implies $\mathcal { C } _ { \lambda } ( A , Z ) \subseteq \mathcal { C } _ { \lambda ^ { \prime } } ( A , Z )$ . Let $m _ { i , \lambda } : = | \mathcal { C } _ { \lambda } ( \tilde { A } _ { i } , \tilde { Z } _ { i } ) |$ and let $N _ { i } ^ { \mathrm { r e f } } : = m _ { i , \lambda _ { \operatorname* { m a x } } } \geq 1$ be the reference set size. Define the fixed lower and upper tail ranks by

$$
k _ { i } ^ { - } : = \left\lceil N _ { i } ^ { \mathrm { r e f } } \frac { \rho } { 2 } \right\rceil , \qquad q _ { i } ^ { + } : = \operatorname* { m a x } \left\{ 1 , \left\lfloor N _ { i } ^ { \mathrm { r e f } } \left( 1 - \frac { \rho } { 2 } \right) \right\rfloor \right\} , \qquad k _ { i } ^ { + } : = N _ { i } ^ { \mathrm { r e f } } - q _ { i } ^ { + } + 1 ,
$$

where $\rho \in ( 0 , 1 )$ is a user-specified interval-tail hyperparameter; $k _ { i } ^ { - }$ is the rank from the lower tail and $k _ { i } ^ { + }$ the rank from the upper tail. These ranks are computed once from $N _ { i } ^ { \mathrm { r e f } }$ and reused for every candidate λ.

For any finite parameter set Θ and coordinate $u ,$ let $Y _ { u , ( r ) } ( \Theta ) , 1 \le r \le | \Theta |$ , denote the r-th order statistic of $\{ Y _ { u } ( \theta ) : \theta \in \Theta \}$ . For a candidate set with enough retained draws, i.e., $m _ { i , \lambda } \geq k _ { i } ^ { - } + k _ { i } ^ { + } - 1$ , define the fixed-rank prediction interval at coordinate u by $[ b _ { i u } ^ { - } ( \lambda ) , b _ { i u } ^ { + } ( \lambda ) ]$ where $b _ { i u } ^ { - } ( \lambda ) : = Y _ { u , ( k _ { i } ^ { - } ) } ( \mathcal { C } _ { \lambda } ( \tilde { A } _ { i } , \tilde { Z } _ { i } ) ) , r _ { i , \lambda } ^ { + } : = m _ { i , \lambda } - k _ { i } ^ { + } + 1$ , and $b _ { i u } ^ { + } ( \lambda ) : = Y _ { u , ( r _ { i . \lambda } ^ { + } ) } ( \mathcal { C } _ { \lambda } ( \tilde { A } _ { i } , \tilde { Z } _ { i } ) )$ . The fixed-rank empirical miscoverage loss is then defined as follows. If $T _ { i } = 0$ , set $L _ { i } ( \lambda ) = 0$ . If $T _ { i } > 0$ but $m _ { i , \lambda } < k _ { i } ^ { - } + k _ { i } ^ { + } - 1$ , set $L _ { i } ( \lambda ) = 1$ . Otherwise,

$$
L _ { i } ( \lambda ) : = \frac { 1 } { T _ { i } } \sum _ { u \in \mathcal { T } _ { i } } \mathbb { I } \Bigl \{ \tilde { Z } _ { i u } \notin \bigl [ b _ { i u } ^ { - } ( \lambda ) , b _ { i u } ^ { + } ( \lambda ) \bigr ] \Bigr \} .\tag{3}
$$

The fixed-rank construction guarantees monotonicity. Because the candidate sets are nested in $\lambda ,$ adding draws can only decrease the $k _ { i } ^ { - }$ -th smallest and increase the $k _ { i } ^ { + } { \cdot } \mathrm { t h }$ largest simulated value, so $b _ { i u } ^ { - } ( \lambda )$ is non-increasing and $b _ { i u } ^ { + } ( \lambda )$ non-decreasing in λ and the intervals expand monotonically with λ. Consequently the threshold selected by CRC in equation (2) favors the narrowest bands that satisfy the empirical constraint.

We introduce two formulations for the candidate sets $ { \mathcal { C } } _ { \lambda } ( A , Z )$ . For each conditioning pair $( A , Z )$ , consider the conditional ODE of Section 2.3, $d X _ { t } \ = \ v _ { t } ^ { \phi } ( X _ { t } \ \vert \ A , Z ) d t , \ X _ { 0 } \ = \ x _ { 0 }$ , and let $\psi _ { 0 , t } ^ { \phi } ( x _ { 0 } \mid A , Z ) : = X _ { t }$ denote the induced flow map, with set-valued pushforward $\psi _ { 0 , t } ^ { \phi } ( U \mid$ $A , Z ) : = \{ \psi _ { 0 , t } ^ { \phi } ( u \mid A , Z ) : u \in U \}$ for finite $U \subset \mathbb { R } ^ { d }$ . In both formulations, increasing λ expands the set, driving down the fixed-rank miscoverage loss.

(i) Radius-based set. This approach truncates the base distribution prior to the flow transformation. Let ${ \cal { S } } _ { N } = \{ \xi _ { j } \} _ { j = 1 } ^ { N }$ <sub>1</sub> be an ensemble of N base samples $\xi _ { j } \overset { i i d } { \sim } \mathcal { N } ( 0 , I _ { d } )$ and $B _ { \lambda } ( 0 ) = \{ x \in$ $\mathbb { R } ^ { d } : \| x \| _ { 2 } \leq \lambda \}$ the $_ { L _ { 2 } - \mathrm { b a l l } }$ of radius λ. We define

$$
\mathcal { C } _ { \lambda } ( A , Z ) : = \psi _ { 0 , 1 } ^ { \phi } \big ( B _ { \lambda } ( 0 ) \cap S _ { N } \mid A , Z \big ) ,
$$

with admissible threshold domain $\Lambda _ { \mathrm { r a d } } = [ 0 , \operatorname* { m a x } _ { 1 \leq j \leq N } \| \xi _ { j } \| _ { 2 } ]$ , whose largest element retains the complete base ensemble.

(ii) Posterior-based set. Alternatively, we threshold the conditional posterior density. Let $\Theta _ { N } ^ { \phi } ( A , Z ) : = \psi _ { 0 , 1 } ^ { \phi } ( S _ { N } \mid A , Z )$ be the generated parameter ensemble and, for any finite set Θ, let $Q _ { \rho } \{ f ( \theta ) : \theta \in \Theta \}$ denote the empirical ρ-quantile of f over Θ. The posterior-based set retains the highest-density draws up to $\lambda \in [ 0 , 1 ]$ :

$$
\mathcal { C } _ { \lambda } ( A , Z ) : = \left\{ \theta \in \Theta _ { N } ^ { \phi } ( A , Z ) : \log p _ { 1 } ^ { \phi } ( \theta \mid A , Z ) \geq Q _ { 1 - \lambda } \Bigl \{ \log p _ { 1 } ^ { \phi } ( \vartheta \mid A , Z ) : \vartheta \in \Theta _ { N } ^ { \phi } ( A , Z ) \Bigr \} \right\} ,
$$

with $\Lambda _ { \mathrm { p o s } } = [ 0 , 1 ] ;$ in practice, $ { \mathcal { C } } _ { \lambda } ( A , Z )$ consists of the top $\lceil \lambda N \rceil$ generated draws ranked by conditional log-density. The conditional log-density is computed along the probability flow ODE via the continuous normalizing-flow (CNF) identity (Chen et al., 2018, Lipman et al., 2023)

$$
{ \frac { d } { d t } } \log p _ { t } ^ { \phi } ( X _ { t } \mid A , Z ) = - \operatorname { d i v } v _ { t } ^ { \phi } ( X _ { t } \mid A , Z ) ,\tag{4}
$$

where $X _ { t }$ traces the trajectory from $X _ { 0 } \sim p _ { 0 } ^ { \phi } ( \theta \mid A _ { \mathrm { o b s } } , Z _ { \mathrm { o b s } } ) = \mathcal { N } ( 0 , I _ { d } )$ to $X _ { 1 } = \theta \sim p _ { 1 } ^ { \phi } ( \theta \ |$ $A _ { \mathrm { o b s } } , Z _ { \mathrm { o b s } } )$

## 3.2 Selection of the trimming parameter.

After fitting the flow, the tail hyperparameter $\rho$ is selected separately for each construction $j \in \{ { \mathrm { r a d } } , { \mathrm { p o s } } \}$ using simulated tuning data generated by the mechanism of Section 3.4.2, independently of the final CRC calibration sample: a tuning CRC sample of size $n _ { \mathrm { t u n e } } = 1 0 0$ and a validation sample of size $n _ { v } = 5 0 0$ . For each $\rho$ on the grid $\mathcal { G } = \{ 0 . 0 1 , 0 . 0 2 , \hdots , 0 . 2 0 \}$ , we compute a tuning threshold $\widehat { \lambda } _ { j } ^ { \mathrm { t u n e } } ( \rho )$ from equation (2) with $n _ { \mathrm { t u n e } }$ in place of $n ,$ and evaluate it on the validation sample using two criteria: the validation miscoverage and the mean pointwise interval length,

$$
\begin{array} { l } { { \displaystyle \widehat { R } _ { j , \mathrm { v a l } } ( \rho ) = \frac { 1 } { n _ { v } } \sum _ { v = 1 } ^ { n _ { v } } L _ { j v } ^ { \mathrm { v a l } } \left( \widehat { \lambda } _ { j } ^ { \mathrm { t u n e } } ( \rho ) ; \rho \right) , } } \\ { { \displaystyle W _ { j } ( \rho ) = \frac { 1 } { n _ { v } T } \sum _ { v = 1 } ^ { n _ { v } } \sum _ { u = 1 } ^ { T } \left[ b _ { j v u } ^ { + } \left( \widehat { \lambda } _ { j } ^ { \mathrm { t u n e } } ( \rho ) ; \rho \right) - b _ { j v u } ^ { - } \left( \widehat { \lambda } _ { j } ^ { \mathrm { t u n e } } ( \rho ) ; \rho \right) \right] , } } \end{array}\tag{5}
$$

where $L _ { j v } ^ { \mathrm { v a l } }$ is the loss in equation (3) for validation observation v and $b _ { j v u } ^ { \pm }$ are the corresponding interval endpoints at coordinate u; we set $W _ { j } ( \rho ) = + \infty$ if any validation interval lacks enough retained draws to be defined. The first criterion ensures that the tuned procedure attains the target risk on fresh data, and the second measures sharpness. We select the admissible value with the sharpest intervals,

$$
\widehat { \rho } _ { j } \in \underset { \rho \in \mathcal { G } _ { j } ^ { \mathrm { a d m } } } { \arg \operatorname* { m i n } } W _ { j } ( \rho ) , \qquad \mathcal { G } _ { j } ^ { \mathrm { a d m } } = \{ \rho \in \mathcal { G } : \widehat { R } _ { j , \mathrm { v a l } } ( \rho ) \leq \alpha , W _ { j } ( \rho ) < \infty \} ,\tag{6}
$$

and the rule yields no selection if $\mathcal { G } _ { j } ^ { \mathrm { a d m } }$ is empty. We then freeze $\rho _ { j } = \widehat { \rho } _ { j }$ , generate a fresh final CRC calibration sample, and recompute the final threshold $\widehat { \lambda } _ { j }$ . Conditional on the fitted model and tuning information, the CRC bound retains its form provided the assumptions of Section 2.2 hold for the final calibration and target losses (Angelopoulos et al., 2024).

## 3.3 Deep Approximate Distribution with Flow Matching

We parameterize the conditional vector field $v _ { t } ^ { \phi } ( x \mid A , Z )$ by a deep neural network with inputs $( x , A , Z , t )$ , where $x \in \mathbb { R } ^ { d }$ is the latent state at time $t \in [ 0 , 1 ]$ (Dax et al., 2023, Lipman et al., 2023, 2024). The conditioning information $( t , A , Z )$ enters through a context vector $c _ { t } = [ h _ { t } , h _ { Z } , A ]$

The time embedding $h _ { t } \in \mathbb { R } ^ { 2 5 6 }$ is obtained by passing random Fourier features of t (Rahimi and Recht, 2007) through a two-layer SiLU network. The sequence representation $h _ { Z }$ is built by dividing $Z \in \mathbb { R } ^ { T }$ into overlapping patches of width 7 and stride 2, linearly embedding each patch into $\mathbb { R } ^ { 1 2 8 }$ , prepending a learned start-of-sequence token, adding positional encodings, and passing the result through a four-layer Transformer encoder (Vaswani et al., 2017). The encoder output is aggregated by learned-query multihead-attention pooling with $q = 4$ queries and 4 heads (Lee et al., 2019), projected back to $\mathbb { R } ^ { 1 2 8 }$ , and augmented with two geometric features, $\log \{ 1 + \operatorname* { m a x } _ { u } Z _ { u } \}$ and the normalized location of the maximum, giving $h _ { Z } \in \mathbb { R } ^ { 1 3 0 }$

Given x and $c _ { t } .$ , the backbone consists of $N _ { \mathrm { b l k } } = 4$ fixed-width residual blocks of hidden dimension 256 with Adaptive Layer Normalization (AdaLN) and Swish-Gated Linear Units (SwiGLU) (Shazeer, 2020, Peebles and Xie, 2023): in each block, the context vector produces a scale and shift applied to the layer-normalized state, which is then passed through a gated linear unit with dropout and added residually. The final normalized state is concatenated with $c _ { t }$ and passed through d coordinate-specific output heads to produce $v _ { t } ^ { \phi } ( x \mid A , Z ) \in \mathbb { R } ^ { d }$ . The context projections, residual output projections, and output heads are initialized to zero, so each block initially acts as the identity and the initial vector field is zero, which improves numerical stability early in training. The complete layer-by-layer specification is provided in Section S1.1 of the Supplementary Material.

## 3.4 Handling Data-Model Discrepancy

Data-model discrepancy—a mismatch between the model output $Y ( \theta )$ and the observed data Z even at the best input setting—is a central challenge in computer model calibration, and ignoring it can bias inference on θ (Bhatnagar et al., 2022, Brynjarsd´ottir and O’Hagan, 2014, Tuo and Wu, 2015). To make the flow-matching model robust to discrepancy, we train it on model outputs contaminated with synthetic discrepancy terms sampled from Gaussian processes with varying hyperparameters. Exposure to a diverse array of discrepancies encourages the learned inverse map to identify features that are informative for recovering the physical parameters while accounting for discrepancy.

## 3.4.1 Training Data Generation

We first draw a pool of $N _ { \mathrm { p a r a m } }$ parameter configurations $\theta _ { k } , k \in \{ 1 , \dots , N _ { \mathrm { p a r a m } } \}$ , via Latin Hypercube Sampling (LHS) over the prior bounds (McKay et al., 1979) and evaluate the simulator to obtain the clean trajectories $Y ( \theta _ { k } )$ and auxiliary outputs $\tilde { A } _ { k } = A ( \theta _ { k } )$ . Separately, we generate a pool of $N _ { \mathrm { n o i s e } }$ raw discrepancy realizations $\tilde { \delta } _ { j } , j \in \{ 1 , \dots , N _ { \mathrm { n o i s e } } \}$ , from a zero-mean Gaussian process (GP) (Bhatnagar et al., 2022) whose marginal variance scales with the temporal variability of the clean output. Writing $\begin{array} { r } { \sigma _ { Y } ^ { 2 } ( \pmb { \theta } ) = \frac { 1 } { T } \sum _ { u = 1 } ^ { T } ( Y _ { u } ( \pmb { \theta } ) - \bar { Y } ( \pmb { \theta } ) ) ^ { 2 } } \end{array}$ with $\begin{array} { r } { \bar { Y } ( \pmb { \theta } ) = \frac { 1 } { T } \sum _ { u = 1 } ^ { T } Y _ { u } ( \pmb { \theta } ) } \end{array}$ ， the covariance between indices $u _ { 1 } , u _ { 2 } \in \{ 1 , . . . , T \}$ is

$$
K ( u _ { 1 } , u _ { 2 } ; \pmb \theta ) = \sigma _ { Y } ^ { 2 } ( \pmb \theta ) \left[ \zeta \mathbb { I } ( u _ { 1 } = u _ { 2 } ) + \kappa \exp \left( - \frac { | u _ { 1 } - u _ { 2 } | } { \tau } \right) \right] .\tag{7}
$$

The prefactor $\sigma _ { Y } ^ { 2 } ( \pmb { \theta } )$ keeps the discrepancy magnitude commensurate with the amplitude of each trajectory, so that the nugget $\zeta ,$ partial sill $\kappa ,$ and range $\tau$ control the noise-to-signal ratio, the relative strength of the temporally correlated component, and the correlation length, respectively. These hyperparameters are sampled via LHS from user-specified bounds for each realization. The training set pairs every parameter configuration k with every noise realization $j ,$

$$
\begin{array} { r } { \tilde { Z } _ { k , j , u } = Y _ { u } ( \theta _ { k } ) + \tilde { \delta } _ { j , u } , } \end{array}\tag{8}
$$

with contaminated states clipped to their theoretical domain $\left( \mathrm { e . g . } \right.$ , nonnegative counts), yielding $\mathcal { D } _ { \mathrm { t r a i n } } = \{ ( \tilde { A } _ { k } , \tilde { Z } _ { k , j } , \theta _ { k } ) \}$ of size $N _ { \mathrm { t r a i n } } = N _ { \mathrm { p a r a m } } \times N _ { \mathrm { n o i s e } }$

## 3.4.2 Calibration Data Generation for Conformal Risk Control

After training, the n CRC calibration samples in $\mathcal { D } _ { n }$ are generated independently, conditional on the fitted flow and the observed conditioning pair. We draw $\theta _ { i } \sim p _ { 1 } ^ { \phi } ( \theta \mid A _ { \mathrm { o b s } } , Z _ { \mathrm { o b s } } )$ via ODE integration, each requiring only a forward pass through the trained flow rather than a new posterior computation, clamp the draws to their physical bounds, and pass each through the simulator to obtain $Y ( \theta _ { i } )$ and ${ \tilde { A } } _ { i }$ . Rather than reusing the training noise pool, we independently sample new GP hyperparameters and generate a unique raw GP draw $\tilde { \delta } _ { i }$ for each $\theta _ { i }$ , and then apply a nonnegative, problem-specific state-dependent envelope $M _ { u } \{ Y ( \theta _ { i } ) \}$ to reproduce the heteroscedastic scale pattern of the target observation:

$$
\tilde { Z } _ { i , u } = Y _ { u } ( \theta _ { i } ) + M _ { u } \{ Y ( \theta _ { i } ) \} \tilde { \delta } _ { i , u } , \qquad u = 1 , \ldots , T .\tag{9}
$$

The scale at coordinate u may depend on the entire clean trajectory; both studies below use

$$
M _ { u } \{ Y ( \theta ) \} = \left\{ \frac { Y _ { u } ( \theta ) - \underset { 1 \leq s \leq T } { \operatorname* { m i n } } Y _ { s } ( \theta ) } { \underset { 1 \leq s \leq T } { \operatorname* { m a x } } Y _ { s } ( \theta ) - \underset { 1 \leq s \leq T } { \operatorname* { m i n } } Y _ { s } ( \theta ) } \right\} ^ { 1 / 2 } ,\tag{10}
$$

so that discrepancies are attenuated in low-signal regions. After modulation, the sequence is clipped to its physical bounds. The synthetic calibration samples are therefore independent draws from

$$
P _ { \mathrm { c a l } } ( A , Z , \theta ) = p _ { \delta , M } \{ Z \mid Y ( \theta ) \} \mathbb { I } \{ A = A ( \theta ) \} p ( \theta ) ,\tag{11}
$$

where $p ( \theta ) = p _ { 1 } ^ { \phi } ( \theta \mid A _ { \mathrm { o b s } } , Z _ { \mathrm { o b s } } )$ and $p _ { \delta , M }$ is the conditional law induced jointly by the raw GP, the envelope M, and clipping.

The two discrepancy mechanisms serve diferent purposes. Unmodulated discrepancies in $\mathcal { D } _ { \mathrm { t r a i n } }$ are a data-augmentation device that makes the learned inverse map robust to severe discrepancies across the full trajectory, whereas the envelope in equation (9) makes the synthetic calibration data reproduce the state-dependent variability anticipated in the observed data, encouraging exchangeability of the target observation with the calibration data. The finite-sample CRC conclusion rests on the working assumption that the target observation $\left( { \cal A } _ { \mathrm { o b s } } , { \cal Z } _ { \mathrm { o b s } } \right)$ and the n synthetic observations $\{ ( \tilde { A } _ { i } , \tilde { Z } _ { i } ) \} _ { i = 1 } ^ { n }$ are jointly exchangeable under this envelope-modulated mechanism. Since the loss $L _ { i } ( \lambda )$ depends only on the observations and not on $\theta _ { i } .$ , the CRC result of Section 2.2 applies under this assumption (Angelopoulos et al., 2024).

## 4 Simulation Study

We evaluate the proposed framework using the synthetic computer model and observations of Bhatnagar et al. (2022). Unlike the application in Section 5, the synthetic output $Y ( \theta )$ is available in closed form without a numerical solver, providing an inexpensive testbed. Throughout this section, the target miscoverage level is $\alpha = 0 . 1$

![](images/186e818fc01f2833d0f1a9ffc1aa226201f1c48ea285dd97b2929656538f3b2f.jpg)  
Figure 2: Synthetic observation and representative model outputs from the simulation-study forward model.

## 4.1 Synthetic Model Outputs and Observational Data

The deterministic forward model produces smooth unimodal curves governed by θ = $( \theta _ { 1 } , \theta _ { 2 } , \theta _ { 3 } ) ^ { \top } \in [ 0 , 1 ] ^ { 3 }$ . For $T = 5 3$ time points $\begin{array} { r } { s ( u ) = \frac { 4 ( u - 1 ) } { T - 1 } - 2 } \end{array}$ equally spaced on $[ - 2 , 2 ]$ , the output at index u is

$$
Y _ { u } ( \pmb { \theta } ) = 0 . 3 + \frac { \theta _ { 1 } + 0 . 3 } { \sqrt { 2 \pi ( \theta _ { 3 } + 0 . 1 ) } } \exp \left[ - \frac { ( s ( u ) - \theta _ { 2 } + 0 . 5 ) ^ { 2 } } { \theta _ { 3 } + 0 . 1 } \right] ,
$$

where $\theta _ { 1 }$ controls the amplitude, $\theta _ { 2 }$ the peak location, and $\theta _ { 3 }$ the dispersion. Representative outputs and the synthetic observation are shown in Figure 2.

All datasets share the GP discrepancy family of Section 3.4, with the kernel in equation (7) and hyperparameters sampled via LHS from range proportion $\tau _ { \mathrm { p r o p } } \in [ 0 . 0 5 , 0 . 2 0 ]$ (so that $\tau =$ $\tau _ { \mathrm { p r o p } } \cdot T )$ , partial sill $\kappa \in [ 0 . 0 5 ^ { 2 } , 0 . 2 5 ^ { 2 } ]$ , and nugget $\zeta \in [ 0 . 0 0 5 ^ { 2 } , 0 . 0 2 5 ^ { 2 } ]$

Training data. We draw $N _ { \mathrm { t r a i n } } = 5 { , } 0 0 0$ parameter configurations via LHS over $[ 0 , 1 ] ^ { 3 }$ and $N _ { \mathrm { n o i s e } } = 3 0 \mathrm { G P }$ hyperparameter vectors, and pair every configuration with every noise realization via equation (8), yielding 150,000 contaminated trajectories. As discussed in Section 3.4, the training discrepancies are left unscaled by any state-dependent envelope so that the network sees severe discrepancies at all signal levels.

Synthetic observation data. We fix a ground-truth parameter $\pmb { \theta } ^ { * }$ and draw a raw GP discrepancy $\varepsilon _ { \mathrm { o b s } }$ with covariance given by equation (7) at $\pmb { \theta } ^ { * }$ , modulated by the envelope of equation (10):

$$
Z _ { \mathrm { o b s } , u } = Y _ { u } ( \pmb \theta ^ { * } ) + M _ { u } \{ Y ( \pmb \theta ^ { * } ) \} \varepsilon _ { \mathrm { o b s } , u } , \qquad u = 1 , \dots , T .\tag{12}
$$

Calibration data. A separate CRC calibration dataset $\mathcal { D } _ { n }$ of size $n = 1 0 0$ is built following Section 3.4.2: parameters are sampled through the trained flow map $\psi _ { 0 , 1 } ^ { \phi } ( \cdot \mid A _ { \mathrm { o b s } } , Z _ { \mathrm { o b s } } )$ and clamped to $[ 0 , 1 ] ^ { 3 }$ , and each draw receives a unique envelope-modulated GP discrepancy as in equation (9). The synthetic target and the CRC calibration observations thus deliberately share the same discrepancy mechanism (GP family, hyperparameter law, and envelope), isolating performance under the nominal specification; matching the discrepancy mechanisms does not by itself establish joint exchangeability, which remains a working assumption (Section 3.4.2). The thresholds $\widehat { \lambda } _ { \mathrm { { r a d } } }$ and $\widehat { \lambda } _ { \mathrm { { p o s } } }$ are calibrated once on $\mathcal { D } _ { n }$ and held fixed thereafter.

## 4.2 Implementation and Evaluation Protocol

We use the conditional vector-field architecture of Section 3.3 with $\sigma _ { \mathrm { m i n } } = 0 . 1$ . Posterior samples are generated by integrating the conditional ODE with the Euler scheme over 100 equally spaced steps on [0, 1], using classifier-free guidance scale w = 1.2 with an interval-based scheduler (Ho and Salimans, 2022, Zheng et al., 2023, Kynk¨a¨anniemi et al., 2024), and the generated parameters are mapped back to the original scale and clamped to $[ 0 , 1 ] ^ { 3 }$ . For both set constructions we draw $N = 1 0 0$ latent base samples $\xi _ { j } \overset { \mathrm { i i d } } { \sim } \mathcal { N } ( 0 , I _ { d } )$ and transport them through the flow; the same latent pool is shared across all n calibration samples so that diferences among parameter sets are driven by the conditioning observations rather than base-space sampling noise.

The miscoverage loss of equation (3) uses the independently selected values $\rho _ { \mathrm { r a d } } = \rho _ { \mathrm { p o s } } = 0 . 0 3$ (Section 3.2) and the relevant-coordinate set $\mathcal { T } ( Z ) = \{ u : Z _ { u } \geq \eta \}$ with $\eta = 0 . 3 0 1$ . Since the clean simulator has baseline 0.3 and minimum attainable peak $0 . 3 \substack { + 0 . 3 / \sqrt { 2 \pi ( 1 . 1 ) } \approx 0 . 4 1 4 }$ over $[ 0 , 1 ] ^ { 3 }$ , η lies only 0.001 (about 0.88% of the minimum dynamic range) above the baseline and excludes only essentially flat background coordinates. For the radius-based sets, the ODE trajectories of all N latent draws conditioned on each calibration observation are precomputed together with the basespace norms $\| \xi _ { j } \| _ { 2 }$ , reducing the threshold search to a filtering operation; for the posterior-based sets, the conditional log-densities computed via equation (4) are cached alongside the generated parameters. In both cases $\widehat { \lambda }$ is obtained by applying Brent’s root-finding algorithm (Brent, 1973) to $\bar { L } ( \lambda ) - \alpha _ { \mathrm { a d j } }$ , where $\begin{array} { r } { \bar { L } ( \lambda ) = n ^ { - 1 } \sum _ { i = 1 } ^ { n } L _ { i } ( \lambda ) } \end{array}$ and $\alpha _ { \mathrm { a d j } } = ( n + 1 ) \alpha / n - 1 / n ;$ the search is carried out over $\lambda \in \left( 0 , \operatorname* { m a x } _ { 1 \leq j \leq N } \| \xi _ { j } \| _ { 2 } \right]$ for the radius-based sets and over $\lambda \in ( 0 . 0 0 1 , 0 . 9 9 9 )$ for the

posterior-based sets, with tolerance $1 0 ^ { - 4 }$

We use repeated held-out simulation to assess how each procedure behaves for repeated measurements of the same system. Holding $\pmb { \theta } ^ { * }$ fixed, we conduct $N _ { \mathrm { e x p } } ~ = ~ 2 0 0$ experiments, regenerating an independent envelope-modulated GP discrepancy with newly sampled hyperparameters according to equation (12) in each. The CRC thresholds are estimated once from $\mathcal { D } _ { n }$ and frozen before the held-out experiments, mimicking deployment to future observations. For each held-out observation we construct parameter sets of maximum size $N ^ { \mathrm { r e f } } = 1 0 0$ and the associated predictive bands, and record (i) empirical miscoverage, (ii) average pointwise interval length (sharpness), and (iii) RMSE between the held-out observation and the band median (center accuracy). For comparison, a KOH–MCMC baseline is rerun separately for every held-out observation and its pointwise (1 − α) posterior band is constructed, reproducing the observationspecific, non-amortized Bayesian workflow (MCMC specifications and convergence diagnostics are given in Section S3.1 of the Supplementary Material).

## 4.3 Results

Figure 3 displays the predictive bands produced by the radius-based, posterior-based, and MCMC methods for a single representative synthetic observation. Here, we intentionally show a case where the radius-based and posterior-based methods lead to almost the same result. The blue dashed curve is the median of the simulator outputs over the parameter set, the green dotted curves trace the pointwise 90% prediction interval, the red solid curve is the synthetic observation, and orange shading marks time points where the observation falls outside the interval (and exceeds η). On this draw, the MCMC band envelops the observation entirely at the cost of roughly 7× longer pointwise intervals and roughly twice the RMSE between the predictive median and the observation, whereas the two CRC bands enclose the observation much more tightly.

Figure 4 summarizes the three metrics across the $N _ { \mathrm { e x p } } = 2 0 0$ experiments. The radius-based set, calibrated at $\hat { \lambda } _ { \mathrm { { r a d } } } = 1 . 3 0 7 8$ , attains a mean miscoverage of 0.1222, close to the target, with some dispersion (75% quantile 0.1989). The posterior-based set, calibrated to retain the top $\hat { \lambda } _ { \mathrm { p o s } } = 4 6 . 4 7 \%$ of draws by conditional log-density, is more conservative with mean miscoverage 0.0721 (75% quantile 0.1132). The MCMC baseline attains zero miscoverage in all 200 experiments, reflecting bands wide enough to envelop almost every realization regardless of how the discrepancy is distributed. The RMSE and interval-length panels make the cost of this overcoverage explicit: the CRC methods achieve mean RMSE 0.0089 (radius) and 0.0085 (posterior), roughly a third of the MCMC mean of 0.0222, and mean pointwise interval lengths of 0.0363 and 0.0426, about 20% of the MCMC mean of 0.2073—a pattern consistent with parameter–discrepancy nonidentifiability widening posterior uncertainty (Brynjarsd´ottir and O’Hagan, 2014).

![](images/6cc81c71577d17fd50f8295b154f86b6571fc845a8be9e97af27d24cc2d0a2f3.jpg)

(a) Radius-based construction: miscoverage 0.0943, RMSE 0.0119, average pointwise interval length 0.0337.  
![](images/6feb166a6339a62937d359582c45ee29d006179572c5e78e30fa6d7fc9e546d3.jpg)

(b) Posterior-based construction: miscoverage 0.0943, RMSE 0.0118, average pointwise interval length 0.0339.  
![](images/56d1ae1d4a87e31d09b1059357e9162a7ffc7e9699caa343f21f68195302d45f.jpg)  
(c) MCMC baseline: miscoverage 0.0000, RMSE 0.0232, average pointwise interval length 0.2058.  
Figure 3: Prediction bands for the same representative synthetic observation under the two CRC constructions and the KOH–MCMC baseline, with target miscoverage $\alpha = 0 . 1 $ . Line and shading conventions are described in the text.

![](images/2d55cc963f13a1896cbcb9b979c0bf6a65b7eca899f81358a2c7dc9bb504eba5.jpg)  
(a) Miscoverage

![](images/2468c12d826367b3adffff02cf4e09f8fc8b06ec3653baf29d9e20912b48fe27.jpg)  
(b) RMSE

![](images/ae7c1f3a63bc08629b30b79dd9ef33abea56f57e9a544c996048ac6bed28ea04.jpg)  
(c) Average interval length  
Figure 4: Distributions of evaluation metrics across 200 independent experiments: empirical miscoverage, RMSE between the predictive median and the synthetic observation, and average pointwise interval length. The horizontal dashed line in panel (a) marks the target level α = 0.1.

These results illustrate the central trade-of targeted by the framework. The MCMC bands reflect posterior uncertainty under an assumed Gaussian observation model but are not calibrated to a prespecified predictive-risk target (Syring and Martin, 2019); they are over-wide on average and cannot adapt to the realized discrepancy. The CRC methods calibrate the interval width directly against the empirical miscoverage on synthetic data generated under the same flexible GP discrepancy family (Angelopoulos et al., 2024), producing intervals that are tight, adaptive, and near the nominal target. An ablation that bypasses the CRC step and instead retains a fixed 90% of the generated draws under either selection rule (Section S4 of the Supplementary Material) indicates that this behavior is driven by the CRC calibration step rather than by the set constructions themselves: for the representative observation of Figure 3, both fixed-retention rules attain zero miscoverage but produce average interval lengths of 0.0597, about 1.8 times those of the corresponding CRC bands, indicating overcoverage.

![](images/342acd5dea254cdf42c2f758bf673b47be5e3589435838a9f3eeb40ed8f2a250.jpg)  
Figure 5: Observed measles data and representative SEIR model outputs.

## 5 Application to a Measles SEIR Model

We now apply the framework to a real-data calibration problem: inferring the parameters of an age-structured Susceptible–Exposed–Infected–Recovered (SEIR) model from the 2010 measles outbreak in Malawi (Minetti et al., 2013). This application is more challenging than the simulation study in three respects. First, the forward model is a system of 84 coupled ODEs whose evaluation is far more expensive than the closed-form curve. Second, the posterior is non-identifiable along certain directions in parameter space, complicating both inference and discrepancy modeling. Third, the data are real, so the true data-generating mechanism is unknown and the discrepancy assumption underlying the exchangeability argument can only be assessed indirectly. Our goal is to demonstrate finite-sample miscoverage control under these realistic conditions while remaining computationally favorable relative to the Bayesian baseline.

## 5.1 Measles Outbreak Data and Age-Structured SEIR Model

The data, originally analyzed by Minetti et al. (2013) and shown in Figure 5, consist of reported weekly measles case counts $Z _ { \mathrm { o b s } }$ at $T = 5 3$ epochs spanning 52 weeks (364 days) and the empirical proportions of cases across three coarse age groups (0.5–5, 5–15, and 15+ years), $A _ { \mathrm { o b s } } = ( 0 . 4 2 , 0 . 3 0 , 0 . 2 8 ) \in \Delta ^ { 2 } \subset \mathbb { R } ^ { 3 }$ , where $\Delta ^ { 2 }$ is the 2-simplex. The time series $Z _ { \mathrm { o b s } }$ is treated as the noisy observation subject to discrepancy, while $A _ { \mathrm { o b s } }$ plays the role of the auxiliary summary A, assumed to be observed exactly.

We adopt an age-structured SEIR formulation (Shea et al., 2014) in which the host population is partitioned into $K = 2 1$ age classes (a six-month band for 0.5–1 year, one-year bands up to 19–20 years, and a single class for 20+ years). For each class $i ,$ let $S _ { i } , E _ { i } , I _ { i } , R _ { i }$ denote the number of susceptible, exposed, infectious, and recovered individuals. The dynamics are

$$
\begin{array} { l l } { \displaystyle \frac { d S _ { i } } { d t } = - \lambda _ { i } ( t ) S _ { i } , } & { \displaystyle \frac { d E _ { i } } { d t } = \lambda _ { i } ( t ) S _ { i } - \sigma E _ { i } , } \\ { \displaystyle \frac { d I _ { i } } { d t } = \sigma E _ { i } - \gamma I _ { i } , } & { \displaystyle \frac { d R _ { i } } { d t } = \gamma I _ { i } , } \end{array}\tag{13}
$$

with latent and infectious periods both fixed at seven days, $\sigma = \gamma = 1 / 7$ (Shea et al., 2014). The age-specific force of infection is density dependent,

$$
\lambda _ { i } ( t ) = \beta \sum _ { j = 1 } ^ { K } W _ { i j } I _ { j } ( t ) ,\tag{14}
$$

where $\beta > 0$ is a per-pair transmission rate and $W = ( W _ { i j } )$ is the Who-Acquires-Infection-From-Whom (WAIFW) matrix. Following Anderson and May (1985), Shea et al. (2014), $W _ { i j }$ depends only on $| i - j |$ , decreasing from 1 on the diagonal to $2 / 3$ for $| i - j | \ge 4$ (the full specification is given in Section S2 of the Supplementary Material), encoding contact rates that are highest within an age class and decline with age separation.

$\mathrm { A t } t = 0 .$ , the susceptible age distribution is a discretized Gamma density: $S _ { i } ( 0 )$ is proportional to the Gamma $( a _ { \mathrm { s h } } , a _ { \mathrm { r t } } )$ density at the class midpoint, normalized so that $\textstyle \sum _ { i = 1 } ^ { K } S _ { i } ( 0 ) = N _ { 0 }$ , the pre-outbreak susceptible population size. A small fraction $f _ { E } \in ( 0 , 1 )$ of each class is moved into the exposed compartment to seed the epidemic, $E _ { i } ( 0 ) = \mathrm { m a x } \{ f _ { E } \cdot S _ { i } ( 0 ) , 1 \}$ , with $S _ { i } ( 0 )$ adjusted accordingly and $I _ { i } ( 0 ) = R _ { i } ( 0 ) = 0$ . Because the elapsed time between epidemic onset and the first reporting week is unknown, we introduce a warm-up duration $t _ { 0 } > 0 \colon$ the system is integrated from $t = 0 \mathrm { ~ t o ~ } t = t _ { 0 } .$ , and the simulator is then evaluated at $t = t _ { 0 } + 7 ( u - 1 )$ days for $u = 1 , \dots , T$ . The resulting weekly case-count sequence is the simulator output $Y ( \pmb { \theta } ) \in \mathbb { R } ^ { T }$ and the time-summed age-group proportions of the infectious compartment over the observation window,

$$
a _ { g } = \frac { \sum _ { i \in \mathcal { G } _ { g } } \int _ { t _ { 0 } } ^ { t _ { 0 } + 3 6 4 } I _ { i } ( t ) d t } { \sum _ { i = 1 } ^ { K } \int _ { t _ { 0 } } ^ { t _ { 0 } + 3 6 4 } I _ { i } ( t ) d t } , \qquad g \in \{ 1 , 2 , 3 \} ,
$$

with $\mathcal { G } _ { g }$ the index sets of the three age groups, form the auxiliary summary $A ( \pmb \theta ) = ( a _ { 1 } , a _ { 2 } , a _ { 3 } ) \in$

<table><tr><td>Parameter</td><td>Definition</td><td>Prior support</td></tr><tr><td> $\beta$ </td><td>Per-pair transmission rate  $\left( \mathrm { d a y ^ { - 1 } } \right)$ </td><td> $[ 0 . 5 \times 1 0 ^ { - 6 } , \ 2 . 5 \times 1 0 ^ { - 6 } ]$ </td></tr><tr><td> $N _ { 0 }$ </td><td>Susceptible population size</td><td> $[ 1 . 4 \times 1 0 ^ { 5 } , ~ 3 \times 1 0 ^ { 5 } ]$ </td></tr><tr><td> $f _ { E }$ </td><td>Initial exposed fraction</td><td> $[ 1 0 ^ { - 4 } , ~ 1 0 ^ { - 3 } ]$ </td></tr><tr><td> $a _ { \mathrm { s h } }$ </td><td>Shape of age distribution</td><td>[0.8, 1.4]</td></tr><tr><td> $a _ { \mathrm { r t } }$ </td><td>Rate of age distribution</td><td>[0.09, 0.39]</td></tr><tr><td> $t _ { 0 }$ </td><td>Warm-up duration (days)</td><td>[0.1, 10]</td></tr></table>

Table 1: Latin hypercube bounds for the SEIR calibration parameters.

$\Delta ^ { 2 } .$ . The calibration target is the six-dimensional vector $\pmb \theta = ( \beta , N _ { 0 } , f _ { E } , a _ { \mathrm { s h } } , a _ { \mathrm { r t } } , t _ { 0 } ) ^ { \top }$ , sampled from a Latin hypercube design with the bounds in Table 1.

## 5.2 Discrepancy, Synthetic Data, and Implementation

We assume that the discrepancy between $Y ( \pmb \theta )$ and the observed weekly cases is generated by the heterogeneous GP family of Section 3.4, with hyperparameters sampled via LHS from bounds calibrated to the scale of the measles data: $\tau _ { \mathrm { p r o p } } \in [ 0 . 1 , 0 . 3 ] , \kappa \in [ 0 . 1 0 ^ { 2 } , 0 . 4 0 ^ { 2 } ] , \mathrm { a n d } \zeta \in [ 0 . 0 1 ^ { 2 } , 0 . 0 4 ^ { 2 } ]$ ; $A ( \pmb \theta )$ is not subject to discrepancy. The training set is constructed exactly as in Section 3.4.1, with $N _ { \mathrm { t r a i n } } = 5 { , } 0 0 0 \mathrm { L H S }$ parameter configurations from Table 1, $N _ { \mathrm { n o i s e } } = 3 0 \mathrm { G P }$ hyperparameter vectors, no state-dependent envelope, and 150,000 contaminated training sequences. The CRC calibration dataset $\mathcal { D } _ { n }$ of size $n = 1 0 0$ follows Section 3.4.2: parameters are drawn through the trained flow map conditioned on $( A _ { \mathrm { o b s } } , Z _ { \mathrm { o b s } } )$ , clamped to the support in Table 1, passed through the SEIR model, and contaminated with a unique GP discrepancy modulated by the envelope of equation (10), giving independent draws from equation (11). Under the working assumption that the real-world discrepancy is itself a realization from this heteroscedastic GP family, $\mathcal { D } _ { n }$ and the target observation are exchangeable.

The flow-matching architecture, sampling ODE, parameter-set constructions, and evaluation metrics are identical to those of Section 4.2. The signal-relevance threshold is $\eta = 3 0 0$ weekly cases, the level below which apparent miscoverage carries little epidemiological consequence, and the tuning procedure of Section 3.2 selects $\rho _ { \mathrm { r a d } } = 0 . 0 3$ and $\rho _ { \mathrm { p o s } } = 0 . 0 5$ . The thresholds $\widehat { \lambda } _ { \mathrm { { r a d } } }$ and $\widehat { \lambda } _ { \mathrm { { p o s } } }$ are calibrated once on $\mathcal { D } _ { n }$ and held fixed. As a baseline, KOH calibration is run by MCMC for the real observation to sample from the posterior predictive distribution, from which pointwise (1 − α) credible intervals are constructed; MCMC specifications and convergence diagnostics are given in Section S3.2 of the Supplementary Material.

<table><tr><td>Method</td><td>Miscoverage</td><td>RMSE</td><td>Avg. interval length</td></tr><tr><td>Radius-based</td><td>0.1379</td><td>940.77</td><td>2540.26</td></tr><tr><td>Posterior-based</td><td>0.1379</td><td>908.40</td><td>3157.48</td></tr><tr><td>MCMC baseline</td><td>0.0345</td><td>947.96</td><td>3526.89</td></tr><tr><td>Target (α = 0.1)</td><td>0.1000</td><td></td><td></td></tr></table>

Table 2: Empirical miscoverage, RMSE between the median predicted weekly cases and the observed measles outbreak, and average pointwise interval length for the radius-based parameter set, posterior-based parameter set, and MCMC baseline. The target miscoverage level is α = 0.1.

## 5.3 Results

Table 2 reports empirical miscoverage, RMSE between the median predicted trajectory and the observed weekly cases, and average pointwise interval length for the two proposed constructions and the MCMC baseline. Figure 6 displays the proposed bands overlaid on the Malawi data, with miscovered weeks highlighted in orange; the KOH–MCMC band was shown in Figure 1a.

Both proposed parameter sets achieve an empirical miscoverage of 0.1379, moderately above the target α = 0.1 and consistent with the CRC guarantee $\mathbb { E } [ L _ { n + 1 } ( \hat { \lambda } ) ] ~ \leq ~ \alpha .$ , with average interval lengths of 2540.26 and 3157.48 cases. The MCMC baseline attains a miscoverage of 0.0345, less than half the nominal level, with the widest band (3526.89 cases); this pattern is consistent with the sensitivity of posterior uncertainty to the assumed discrepancy specification (Brynjarsd´ottir and O’Hagan, 2014, Syring and Martin, 2019). The CRC procedure instead calibrates $\hat { \lambda }$ directly against the empirical miscoverage of synthetic data generated under the heterogeneous GP family, sidestepping the need for any single parametric noise model to be exactly correct. The RMSE values of the three methods are comparable. The ablation study in Section S4 of the Supplementary Material isolates the role of the CRC step on these data: retaining a fixed 90% of the generated draws without CRC, under either selection rule, attains zero miscoverage on the evaluated weeks but produces bands considerably wider than both the CRC and KOH–MCMC bands, indicating overcoverage (cf. Figure 1b).

A further feature of the framework is that the miscovered weeks, shown as orange bands in Figure 6, localize where the SEIR model struggles most. In both CRC panels the orange bands cluster around the peak and in the late tail, regions corresponding to the asymmetric fallof of the real outbreak—a pattern that the approximately symmetric, bell-shaped SEIR trajectories cannot reproduce regardless of parameter choice. The CRC framework therefore supplies not only a coverage guarantee but also a model-diagnostic signal: when the calibrated interval fails, the model is structurally inadequate. The MCMC procedure ofers no analogous diagnostic because its credible intervals do not target coverage; its overcoverage reflects the same structural inadequacy but provides no statistical mechanism for surfacing it. In applications where the validity of the predictive interval is itself of scientific interest, this combination of finite-sample guarantees and built-in diagnostics supports the proposed approach.

![](images/ea318c942b2b8c2511115aa07ab0c73157a22c1a9bfafca59c6a48a89dfe8825.jpg)

(a) Radius-based construction.  
![](images/8224956cd3fe2089a12079757fdd8fcf976c5f7e5ce9ae9e5c8a7824a20c3e3b.jpg)  
(b) Posterior-based construction.  
Figure 6: CRC prediction intervals for the Malawi measles outbreak. The blue dashed line is the median of the generated SEIR trajectories, the green dotted lines are the pointwise prediction bounds, and the red solid line is the observed series. Orange bands mark weeks at which the observed value falls outside the prediction interval and exceeds the signal-relevance threshold $\eta = 3 0 0$

## 6 Discussion

This paper has developed a framework for computer model calibration that combines amortized neural posterior approximation via flow matching with conformal risk control to deliver finitesample miscoverage guarantees on the simulator pushforward of posterior samples. In both a synthetic benchmark with known ground truth and a real-data calibration of an age-structured SEIR model to the 2010 Malawi measles outbreak, the proposed parameter sets attained nearnominal coverage with substantially sharper bands, whereas a standard KOH–MCMC baseline overcovered. The contrast reflects a diference in objective rather than in implementation quality: a Bayesian analysis under a Gaussian observation model reports credible intervals as a byproduct of posterior sampling and has limited capacity to account for structural mismatch between a mechanistic model and reality, whereas our framework targets predictive miscoverage directly and narrows the parameter set only as much as the calibration data demand.

Two features make the framework well-suited to applied calibration. First, it is amortized: the flow-matching posterior is trained once and queried rapidly for any new observation, in contrast to MCMC, which must be rerun each time (Cranmer et al., 2020, Dax et al., 2023)—a consequential advantage for real-time surveillance and sequential calibration where data accrue continuously. More fundamentally, amortization is what makes the CRC layer possible at all: the calibration and tuning samples together require on the order of $n { + } n _ { \mathrm { t u n e } } { + } n _ { v }$ conditional posterior approximations per target observation, a cost that is negligible for a trained flow but prohibitive for observation-specific MCMC. Second, it supplies an interpretable diagnostic: clusters of time points at which the calibrated interval fails to cover localize the regions where the mechanistic model is structurally inadequate, directing attention toward the aspects of the model most in need of refinement.

Several limitations should be acknowledged. The finite-sample guarantee relies on exchangeability between the synthetic calibration set and the real observation (Angelopoulos et al., 2024), which holds under the working assumption that the true discrepancy lies in the user-specified

GP family—a condition that cannot be verified from a single observation. The quality of the amortized posterior depends on the size and coverage of the training simulation pool, and higherdimensional parameter spaces may require larger training budgets (Cranmer et al., 2020, Lueckmann et al., 2021). Finally, the framework controls marginal miscoverage averaged over time; pointwise or conditional miscoverage at specific time points is not guaranteed.

Several extensions suggest themselves. Although exchangeability is untestable in the strict sense, partial diagnostics comparing summary statistics of the real observation against their empirical distribution across the calibration set—for instance, a kernel two-sample test on residual sequences (Gretton et al., 2012)—would help flag gross violations. Mondrian conformal prediction (Vovk et al., 2005) or localized conformal variants (Guan, 2023) could deliver control that adapts to local dificulty, a multi-channel extension would accommodate several observation channels with distinct discrepancy structures, and moving beyond the fixed-kernel GP family to non-stationary or non-Gaussian discrepancies, or learning the discrepancy family adaptively, would improve robustness under misspecification. More broadly, the calibration literature has long recognized discrepancy between model and reality as the principal source of inferential difficulty (Kennedy and O’Hagan, 2001, Brynjarsd´ottir and O’Hagan, 2014, Tuo and Wu, 2015). The framework presented here identifies the region of parameter space most relevant for prediction by calibrating prediction intervals against a flexible family of discrepancies, and we hope it proves useful as a robust complement to classical KOH calibration for scientific problems in which mechanistic models meet imperfect data.

## Acknowledgments

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Ministry of Science and ICT (MSIT) (grant number [RS-2025-00523567,RS-2026-25607210]), the Global-LAMP Program of the NRF grant funded by the Ministry of Education of the Republic of Korea (grant number [RS-2023-00301976]).

## Supplementary Material

Supplementary material available online contains the full flow-matching architecture, training details, and a general formulation of flow matching (Section S1), the WAIFW matrix (Section S2), MCMC specifications and diagnostics (Section S3), and an ablation study without conformal risk control (Section S4). The source code can be downloaded from github.com/ryanb61901/Prediction-Uncertainty-Controlled-Computer-Model-Calibration.

## Data Availability Statement

The data that support the findings of this study are available from the authors upon reasonable request.

## References

Anderson, R. M. and R. M. May (1985). Age-related changes in the rate of disease transmission: Implications for the design of vaccination programmes. The Journal of Hygiene 94 (3), 365– 436.

Angelopoulos, A. N. and S. Bates (2023). Conformal prediction: A gentle introduction. Foundations and Trends in Machine Learning 16 (4), 494–591.

Angelopoulos, A. N., S. Bates, A. Fisch, L. Lei, and T. Schuster (2024). Conformal risk control. In The Twelfth International Conference on Learning Representations.

Bayarri, M., J. Berger, J. Cafeo, G. Garcia-Donato, F. Liu, J. Palomo, R. Parthasarathy, R. Paulo, J. Sacks, and D. Walsh (2007). Computer model validation with functional output. The Annals of Statistics 35(5), 1874–1906.

Bhatnagar, S., W. Chang, S. Kim, and J. Wang (2022). Computer model calibration with time series data using deep learning and quantile regression. SIAM/ASA Journal on Uncertainty Quantification 10 (1), 1–26.

Brent, R. P. (1973). Algorithms for Minimization without Derivatives. Prentice-Hall.

Brynjarsd´ottir, J. and A. O’Hagan (2014). Learning about physical parameters: The importance of model discrepancy. Inverse Problems 30 (11), 114007.

Carpenter, B., A. Gelman, M. D. Hofman, D. Lee, B. Goodrich, M. Betancourt, M. Brubaker, J. Guo, P. Li, and A. Riddell (2017). Stan: A probabilistic programming language. Journal of Statistical Software 76(1), 1–32.

Chen, R. T. Q., Y. Rubanova, J. Bettencourt, and D. K. Duvenaud (2018). Neural ordinary diferential equations. In Advances in Neural Information Processing Systems, Volume 31, pp. 6571–6583.

Cranmer, K., J. Brehmer, and G. Louppe (2020). The frontier of simulation-based inference. Proceedings of the National Academy of Sciences 117(48), 30055–30062.

Dax, M., J. Wildberger, S. Buchholz, S. R. Green, J. H. Macke, and B. Sch¨olkopf (2023). Flow matching for scalable simulation-based inference. In Advances in Neural Information Processing Systems, Volume 36.

Greenberg, D., M. Nonnenmacher, and J. Macke (2019). Automatic posterior transformation for likelihood-free inference. In Proceedings of the 36th International Conference on Machine Learning, Volume 97 of Proceedings of Machine Learning Research, pp. 2404–2414. PMLR.

Gretton, A., K. M. Borgwardt, M. J. Rasch, B. Sch¨olkopf, A. Smola, and K.-R. M¨uller (2012). A kernel two-sample test. Journal of Machine Learning Research 13, 723–773.

Guan, L. (2023). Localized conformal prediction: A generalized inference framework for conformal prediction. Biometrika 110 (1), 33–50.

Hermans, J., A. Delaunoy, F. Rozet, A. Wehenkel, V. Begy, and G. Louppe (2022). A crisis in simulation-based inference? beware, your posterior approximations can be unfaithful. Transactions on Machine Learning Research.

Higdon, D., J. Gattiker, B. Williams, and M. Rightley (2008). Computer model calibration using high-dimensional output. Journal of the American Statistical Association 103 (482), 570–583.

Ho, J. and T. Salimans (2022). Classifier-free difusion guidance. arXiv preprint arXiv:2207.12598.

Hofman, M. D. and A. Gelman (2014). The No-U-Turn sampler: Adaptively setting path lengths in Hamiltonian Monte Carlo. Journal of Machine Learning Research 15 (47), 1593–1623.

Kennedy, M. C. and A. O’Hagan (2001). Bayesian calibration of computer models. Journal of the Royal Statistical Society: Series B (Statistical Methodology) 63 (3), 425–464.

Kynk¨a¨anniemi, T., M. Aittala, T. Karras, S. Laine, T. Aila, and J. Lehtinen (2024). Applying guidance in a limited interval improves sample and distribution quality in difusion models. In Advances in Neural Information Processing Systems, Volume 37.

Lee, J., Y. Lee, J. Kim, A. Kosiorek, S. Choi, and Y. W. Teh (2019). Set Transformer: A framework for attention-based permutation-invariant neural networks. In Proceedings of the 36th International Conference on Machine Learning, Volume 97 of Proceedings of Machine Learning Research, pp. 3744–3753. PMLR.

Lee, J., C. Xu, and Y. Xie (2026). Flow-based conformal prediction for multi-dimensional time series. In The Fourteenth International Conference on Learning Representations.

Lim, D., K. T. Ko, H. Hong, H. Lee, B. Choi, W. Chang, S. Choi, and J. K. Kim (2025). A history-dependent approach for accurate initial condition estimation in epidemic models. PLOS Computational Biology 21 (9), e1013438.

Lipman, Y., R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le (2023). Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations.

Lipman, Y., M. Havasi, P. Holderrieth, N. Shaul, M. Le, B. Karrer, R. T. Q. Chen, D. Lopez-Paz, H. Ben-Hamu, and I. Gat (2024). Flow matching guide and code. arXiv preprint arXiv:2412.06264.

Lueckmann, J.-M., J. Boelts, D. S. Greenberg, P. J. Gon¸calves, and J. H. Macke (2021). Benchmarking simulation-based inference. In Proceedings of the 24th International Conference on Artificial Intelligence and Statistics, Volume 130 of Proceedings of Machine Learning Research, pp. 343–351. PMLR.

McKay, M. D., R. J. Beckman, and W. J. Conover (1979). A comparison of three methods for selecting values of input variables in the analysis of output from a computer code. Technometrics 21 (2), 239–245.

Minetti, A., M. Kagoli, A. Katsulukuta, H. Huerga, A. Featherstone, H. Chiotcha, D. Noel, C. Bopp, L. Sury, R. Fricke, M. Iscla, N. Hurtado, T. Ducomble, S. Nicholas, S. Kabuluzi, R. F. Grais, and F. J. Luquero (2013). Lessons and challenges for measles control from unexpected large outbreak, Malawi. Emerging Infectious Diseases 19(2), 202–209.

Papamakarios, G. and I. Murray (2016). Fast ϵ-free inference of simulation models with Bayesian conditional density estimation. In Advances in Neural Information Processing Systems, Volume 29, pp. 1028–1036.

Papamakarios, G., D. C. Sterratt, and I. Murray (2019). Sequential neural likelihood: Fast likelihood-free inference with autoregressive flows. In Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics, Volume 89 of Proceedings of Machine Learning Research, pp. 837–848. PMLR.

Peebles, W. and S. Xie (2023). Scalable difusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205.

Plumlee, M. (2017). Bayesian calibration of inexact computer models. Journal of the American Statistical Association 112 (519), 1274–1285.

Plumlee, M. (2019). Computer model calibration with confidence and consistency. Journal of the Royal Statistical Society: Series B (Statistical Methodology) 81 (3), 519–545.

Rahimi, A. and B. Recht (2007). Random features for large-scale kernel machines. In Advances in Neural Information Processing Systems, Volume 20, pp. 1177–1184.

Romano, Y., E. Patterson, and E. Candes (2019). Conformalized quantile regression. In Advances in Neural Information Processing Systems, Volume 32.

Shazeer, N. (2020). GLU variants improve transformer. arXiv preprint arXiv:2002.05202.

Shea, K., M. J. Tildesley, M. C. Runge, C. J. Fonnesbeck, and M. J. Ferrari (2014). Adaptive management and the value of information: Learning via intervention in epidemiology. PLOS Biology 12 (10), e1001970.

Syring, N. and R. Martin (2019). Calibrating general posterior credible regions. Biometrika 106 (2), 479–486.

Tuo, R. and C. J. Wu (2015). Eficient calibration for imperfect computer models. The Annals of Statistics 43 (6), 2331–2352.

Vaswani, A., N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin (2017). Attention is all you need. In Advances in Neural Information Processing Systems, Volume 30.

Vovk, V., A. Gammerman, and G. Shafer (2005). Algorithmic Learning in a Random World. Springer.

Zheng, Q., M. Le, N. Shaul, Y. Lipman, A. Grover, and R. T. Q. Chen (2023). Guided flows for generative modeling and decision making. arXiv preprint arXiv:2311.13443.

## Supplementary Material

## S1 Model Details

## S1.1 Flow-Matching Architecture

This section gives the complete specification of the network summarized in Section 3.3 of the main text. To parameterize the conditional posterior of the calibration parameters, we utilize a deep neural network to approximate the target conditional vector field $v _ { t } ^ { \phi } ( x \mid A , Z )$ (Dax et al., 2023, Lipman et al., 2023, 2024). Let θ ∈ R $\boldsymbol { \theta } \in \mathbb { R } ^ { d }$ denote the target calibration parameter and $x \in \mathbb { R } ^ { d }$ the continuous latent state at integration time $t \in [ 0 , 1 ]$ . The conditional vector field is defined as a mapping

$$
v _ { t } ^ { \phi } ( \cdot \mid \cdot , \cdot ) : \mathbb { R } ^ { d } \times \mathcal { A } \times \mathbb { R } ^ { T } \times [ 0 , 1 ] \to \mathbb { R } ^ { d } , ( x , A , Z , t ) \mapsto v _ { t } ^ { \phi } ( x \mid A , Z )
$$

governed by trainable network weights $\phi .$ In practice, we parameterize $v _ { t } ^ { \phi } ( x \mid A , Z )$ as a neural network with inputs $( x , A , Z , t )$

Let $a = \dim ( A )$ , with $a = 0$ when no auxiliary covariates are available. We inject information on $( t , A , Z )$ through the context vector

$$
c _ { t } = [ h _ { t } , h _ { Z } , A ] \in \mathbb { R } ^ { r _ { c } } ,
$$

where $h _ { t } \in \mathbb { R } ^ { r _ { t } }$ is the time embedding and $h _ { Z }$ is the sequence representation defined below.

Let $\omega \in \mathbb { R } ^ { r _ { t } / 2 }$ be a vector of frequencies sampled once from $\mathcal { N } ( 0 , \sigma _ { \omega } ^ { 2 } I _ { r _ { t } / 2 } )$ and then held fixed (Rahimi and Recht, 2007). Define

$$
r ( t ) = \sqrt { 2 } [ \sin ( 2 \pi t \omega ) ^ { \top } , \cos ( 2 \pi t \omega ) ^ { \top } ] ^ { \top } \in \mathbb { R } ^ { r _ { t } } ,
$$

and pass it through the two-layer SiLU network

$$
\begin{array} { r } { h _ { t } = W _ { t , 2 } \mathrm { S i L U } ( W _ { t , 1 } r ( t ) + b _ { t , 1 } ) + b _ { t , 2 } \in \mathbb { R } ^ { r _ { t } } , } \end{array}
$$

where $W _ { t , 1 } \in \mathbb R ^ { 2 r _ { t } \times r _ { t } } , \ W _ { t , 2 } \ \in \ \mathbb R ^ { r _ { t } \times 2 r _ { t } } , \ b _ { t , 1 } \ \in \ \mathbb R ^ { 2 r _ { t } }$ , and $b _ { t , 2 } \in \mathbb { R } ^ { r _ { t } }$ . In our implementation, $r _ { t } = 2 5 6$ and $\sigma _ { \omega } = 1 6$ (Lipman et al., 2024).

To construct $h _ { Z }$ , we first divide $Z \in \mathbb { R } ^ { T }$ into $L = \lfloor ( T - p ) / s \rfloor + 1$ overlapping patches $P _ { \ell } ( Z ) \in$ $\mathbb { R } ^ { p }$ of width $p = 7$ and stride $s = 2$ . Each patch is embedded as $e _ { \ell } = W _ { \mathrm { p a t c h } } P _ { \ell } ( Z ) + b _ { \mathrm { p a t c h } } \in \mathbb { R } ^ { r _ { Z } }$ where $W _ { \mathrm { p a t c h } } \in \mathbb { R } ^ { r _ { z } \times p } , b _ { \mathrm { p a t c h } } \in \mathbb { R } ^ { r _ { z } }$ , and $r _ { Z } = 1 2 8$ . After a learned start-of-sequence token in $\mathbb { R } ^ { r _ { Z } }$ is prepended and positional encodings are added, a four-layer Transformer encoder maps the sequence to

$$
H ( Z ) \in \mathbb { R } ^ { ( L + 1 ) \times r _ { Z } } .
$$

We aggregate $H ( Z )$ using learned-query multihead-attention pooling (Vaswani et al., 2017, Lee et al., 2019). Let $Q _ { \mathrm { p o o l } } \in \mathbb { R } ^ { q \times r _ { Z } }$ contain $q = 4$ learned query vectors, let $H _ { a } = 4$ be the number of attention heads, and let $r _ { h } ^ { \mathrm { a t t } } = r _ { Z } / H _ { a } = 3 2$ be the dimension of each head. For head $j \in \{ 1 , \dots , H _ { a } \}$ , define

$$
O _ { j } = \mathrm { s o f t m a x } \left( \frac { ( Q _ { \mathrm { p o o l } } W _ { Q , j } ) ( H ( Z ) W _ { K , j } ) ^ { \top } } { \sqrt { r _ { h } ^ { \mathrm { a t t } } } } \right) H ( Z ) W _ { V , j } \in \mathbb { R } ^ { q \times r _ { h } ^ { \mathrm { a t t } } } ,
$$

where $W _ { Q , j } , W _ { K , j } , W _ { V , j } \ \in \ \mathbb { R } ^ { r _ { Z } \times r _ { h } ^ { \mathrm { a t t } } }$ are learned query, key, and value projections. The head outputs are concatenated columnwise and projected as

$$
O = { \mathrm { C o n c a t } } ( O _ { 1 } , \ldots , O _ { H _ { a } } ) W _ { O } + { \bf 1 } _ { q } b _ { O } ^ { \top } \in \mathbb { R } ^ { q \times r _ { Z } } ,
$$

where $W _ { O } \ \in \ \mathbb { R } ^ { r _ { Z } \times r _ { Z } } , \ b _ { O } \ \in \ \mathbb { R } ^ { r _ { Z } }$ , and ${ \mathbf { 1 } } _ { q }$ is the length-q vector of ones. For a matrix $O _ { i }$ Flatten $( O ) : = ( O _ { 1 , : } ^ { \top } , \ldots , O _ { q , : } ^ { \top } ) ^ { \top } \in \mathbb { R } ^ { q r _ { Z } }$ denotes rowwise vectorization. The pooled sequence embedding is therefore

$$
h _ { \mathrm { s e q } } = W _ { \mathrm { p r o j } } \operatorname { F l a t t e n } ( O ) + b _ { \mathrm { p r o j } } \in \mathbb { R } ^ { r _ { Z } } ,
$$

with $W _ { \mathrm { p r o j } } \in \mathbb { R } ^ { r z \times q r z }$ and $b _ { \mathrm { p r o j } } \in \mathbb { R } ^ { r z }$

We augment this learned representation with two geometric features. Let $u ^ { * } =$ $\arg \operatorname* { m a x } _ { u \in \{ 1 , \ldots , T \} } Z _ { u }$ , and define $\tilde { Z } _ { \mathrm { m a x } } = \log \{ 1 + \operatorname* { m a x } _ { u } Z _ { u } \}$ and $\tilde { Z } _ { \mathrm { a r g } } = ( u ^ { * } - 1 ) / T$ , where the

subtraction by one matches the zero-based index used in the implementation. Then

$$
h _ { Z } = [ h _ { \mathrm { s e q } } , \tilde { Z } _ { \mathrm { m a x } } , \tilde { Z } _ { \mathrm { a r g } } ] \in \mathbb R ^ { r _ { Z } + 2 } , \qquad r _ { c } = r _ { t } + r _ { Z } + 2 + a .
$$

Given x and $c _ { t } .$ the vector-field backbone uses $N _ { \mathrm { b l k } } = 4$ fixed-width residual blocks with hidden dimension $r _ { b } = 2 5 6$ , Adaptive Layer Normalization (AdaLN), and Swish-Gated Linear Units (SwiGLU) (Shazeer, 2020, Peebles and Xie, 2023). The initial state is $h ^ { ( 0 ) } = W _ { x } x + b _ { x } \in$ $\mathbb { R } ^ { r _ { b } }$ , where $W _ { x } \in \mathbb { R } ^ { r _ { b } \times d }$ and $b _ { x } \in \mathbb { R } ^ { r _ { b } }$ . In block $m \in \{ 1 , \ldots , N _ { \mathrm { b l k } } \}$ , the context-dependent scale and shift are

$$
\begin{array} { r } { \bigg [ \gamma _ { m } ( c _ { t } ) \bigg ] = W _ { c , m } \mathrm { S i L U } ( c _ { t } ) + b _ { c , m } \in \mathbb { R } ^ { 2 r _ { b } } , } \\ { \left\lfloor \beta _ { m } ( c _ { t } ) \right\rfloor = W _ { c , m } \mathrm { S i L U } ( c _ { t } ) + b _ { c , m } \in \mathbb { R } ^ { 2 r _ { b } } , } \end{array}
$$

where $W _ { c , m } \in \mathbb { R } ^ { 2 r _ { b } \times r _ { c } }$ and $b _ { c , m } \in \mathbb { R } ^ { 2 r _ { b } }$ . The block then computes

$$
\tilde { h } ^ { ( m ) } = \mathrm { L a y e r N o r m } ( h ^ { ( m - 1 ) } ) \odot \{ 1 + \gamma _ { m } ( c _ { t } ) \} + \beta _ { m } ( c _ { t } ) ,
$$

$$
u _ { m } = W _ { \mathrm { v a l } , m } \tilde { h } ^ { ( m ) } + b _ { \mathrm { v a l } , m } , \qquad g _ { m } = W _ { \mathrm { g a t e } , m } \tilde { h } ^ { ( m ) } + b _ { \mathrm { g a t e } , m } ,
$$

$$
h ^ { ( m ) } = h ^ { ( m - 1 ) } + W _ { \mathrm { o u t } , m } \mathrm { D r o p o u t } \{ u _ { m } \odot \mathrm { S i L U } ( g _ { m } ) \} + b _ { \mathrm { o u t } , m } ,
$$

where $W _ { \mathrm { v a l } , m } , W _ { \mathrm { g a t e } , m } , W _ { \mathrm { o u t } , m } \in \mathbb { R } ^ { r _ { b } \times r _ { b } }$ , their corresponding biases lie in $\mathbb { R } ^ { r _ { b } }$ , and $\odot$ denotes elementwise multiplication. Here LayerNorm, Dropout, and SiLU denote the standard layernormalization, dropout, and sigmoid-weighted linear-unit operations, respectively. Finally, $\mathrm { L a y e r N o r m } ( h ^ { ( N _ { \mathrm { b l k } } ) } )$ is concatenated with $c _ { t }$ and passed through d coordinate-specific output heads $g _ { 1 } ^ { \phi } , \ldots , g _ { d } ^ { \phi }$ , giving

$$
v _ { t } ^ { \phi } ( x \mid A , Z ) = \bigl ( g _ { 1 } ^ { \phi } ( [ \mathrm { L a y e r N o r m } ( h ^ { ( N _ { \mathrm { b u k } } ) } ) , c _ { t } ] ) , \bigr . . . , g _ { d } ^ { \phi } ( [ \mathrm { L a y e r N o r m } ( h ^ { ( N _ { \mathrm { b u k } } ) } ) , c _ { t } ] ) \bigr ) ^ { \top } \in \mathbb { R } ^ { d } .
$$

The context projections and $W _ { \mathrm { o u t } , m }$ are initialized to zero, so each residual block initially acts as the identity. The output heads are separately initialized to zero, which makes the initial vector-field output $v _ { t } ^ { \phi }$ equal to zero and improves numerical stability early in training.

The sequence encoder is a Transformer with 4 layers, 4 attention heads, feedforward dimension 512, and dropout rate 0.1 (Vaswani et al., 2017). The model is trained on the 150,000

contaminated trajectories using the conditional flow-matching objective with $\sigma _ { \mathrm { m i n } } = 0 . 1$

## S1.2 Flow Matching Details

This section summarizes the general flow-matching framework used in the main text. Flow matching learns a continuous-time transport from a simple reference distribution $p _ { \mathrm { i n i t } }$ to a target distribution $p _ { \mathrm { d a t a } }$ by regressing a time-dependent vector field (Lipman et al., 2023). Let $p _ { t } ( x \mid z )$ be a conditional probability path with endpoints $p _ { 0 } ( \cdot \mid z ) = p _ { \mathrm { i n i t } }$ and $p _ { 1 } ( \cdot \mid z ) = \mathcal { N } ( z , \sigma _ { \mathrm { m i n } } ^ { 2 } I ) \approx \delta _ { z }$ A conditional vector field $v _ { t } ^ { \mathrm { f l o w } } ( \cdot \mid z )$ generates this path if the solution of

$$
d X _ { t } = v _ { t } ^ { \mathsf { f l o w } } ( X _ { t } \mid z ) d t ,
$$

with $X _ { 0 } \sim p _ { 0 } ( \cdot \mid z )$ satisfies $X _ { t } \sim p _ { t } ( \cdot \mid z )$ for each $t \in [ 0 , 1 ]$

Integrating over $z \sim p _ { \mathrm { d a t a } }$ yields the marginal path

$$
p _ { t } ( x ) : = \int p _ { t } ( x \mid z ) p _ { \mathrm { d a t a } } ( z ) d z ,
$$

with endpoints $p _ { 0 } = p _ { \mathrm { i n i t } }$ and $p _ { 1 } = \mathcal { N } ( 0 , \sigma _ { \mathrm { m i n } } ^ { 2 } I ) * p _ { \mathrm { d a t a } } \approx p _ { \mathrm { d a t a } }$ . The corresponding marginal vector field is

$$
v _ { t } ^ { \mathrm { t a r g e t } } ( x ) = \int v _ { t } ^ { \mathrm { f l o w } } ( x \mid z ) \frac { p _ { t } ( x \mid z ) p _ { \mathrm { d a t a } } ( z ) } { p _ { t } ( x ) } d z ,
$$

which drives

$$
X _ { 0 } \sim p _ { \mathrm { i n i t } } , \qquad d X _ { t } = v _ { t } ^ { \mathrm { t a r g e t } } ( X _ { t } ) d t , \qquad X _ { 1 } { \dot { \sim } } p _ { \mathrm { d a t a } } .
$$

The ideal objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { F M } } ( \phi ) = \mathbb { E } _ { t \sim \pi , x \sim p _ { t } } \left[ \Vert v _ { t } ^ { \phi } ( x ) - v _ { t } ^ { \mathrm { t a r g e t } } ( x ) \Vert ^ { 2 } \right] , } \end{array}
$$

but $v _ { t } ^ { \mathrm { t a r g e t } }$ is often intractable. In practice, one minimizes the conditional objective

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { C F M } } ( \phi ) = \mathbb { E } _ { t \sim \pi , z \sim p _ { \mathrm { d a t a } } , x \sim p _ { t } ( \cdot \vert z ) } \left[ \Vert v _ { t } ^ { \phi } ( x ) - v _ { t } ^ { \mathrm { f o w } } ( x \mid z ) \Vert ^ { 2 } \right] , } \end{array}
$$

which has the same gradient target under standard regularity conditions, i.e., $\nabla _ { \phi } \mathcal { L } _ { \mathrm { C F M } } ( \phi ) =$

$\nabla _ { \phi } { \mathcal { L } } _ { \mathrm { F M } } ( \phi )$ (Lipman et al., 2023, 2024).

After training, approximate samples are obtained by solving

$$
X _ { 0 } \sim p _ { \mathrm { i n i t } } , \qquad d X _ { t } = v _ { t } ^ { \phi } ( X _ { t } ) d t , \qquad t \in [ 0 , 1 ] ,
$$

and taking $X _ { 1 }$

For Gaussian probability paths,

$$
p _ { t } ( x \mid z ) = \mathcal { N } ( x ; \alpha _ { t } z , \beta _ { t } ^ { 2 } I _ { d } ) ,
$$

with smooth schedules $\left( \alpha _ { t } , \beta _ { t } \right)$ satisfying $\alpha _ { 0 } = 0 , \alpha _ { 1 } = 1 , \beta _ { 0 } = 1$ , and $\beta _ { 1 } = \sigma _ { \operatorname* { m i n } } .$ , the conditional velocity is

$$
v _ { t } ^ { \mathrm { f o w } } ( x \mid z ) = \left( \dot { \alpha } _ { t } - \frac { \dot { \beta } _ { t } } { \beta _ { t } } \alpha _ { t } \right) z + \frac { \dot { \beta } _ { t } } { \beta _ { t } } x .
$$

In practice, training uses Monte Carlo estimates of the CFM objective. At each update, we draw a minibatch of target–condition pairs $\left( z _ { b } , y _ { b } \right)$ from the training data, times $t _ { b } \sim \pi ,$ and independent noise vectors $\varepsilon _ { b } \sim \mathcal { N } ( 0 , I _ { d } )$ ; we then form $x _ { t _ { b } } = t _ { b } z _ { b } + \{ 1 - ( 1 - \sigma _ { \operatorname* { m i n } } ) t _ { b } \} \varepsilon _ { b }$ and evaluate the closed-form target $v _ { t _ { b } } ^ { \mathrm { f l o w } } ( x _ { t _ { b } } \ | \ z _ { b } )$ . The network parameters $\phi$ are updated by backpropagation to minimize the minibatch mean squared error between this target and $v _ { t _ { b } } ^ { \phi } \left( x _ { t _ { b } } \mid y _ { b } \right)$ , thereby learning the conditional vector field over the full probability path without solving an ODE during training.

For conditional generation, guided flows interpolate unconditional and conditional fields:

$$
\widetilde v _ { t } ( x \mid y ) = \big ( 1 - w _ { t } \big ) v _ { t } ^ { \mathrm { t a r g e t } } ( x \mid \emptyset ) + w _ { t } v _ { t } ^ { \mathrm { t a r g e t } } ( x \mid y ) , \qquad w _ { t } > 1 ,
$$

which increases adherence to the condition y at sampling time (Ho and Salimans, 2022, Zheng et al., 2023).

## S2 WAIFW Matrix for the Age-Structured SEIR Model

We adopt a structured WAIFW matrix in which the entry between age classes i and $j$ depends only on $| i - j |$ , following Anderson and May (1985) and Shea et al. (2014):

$$
W _ { i j } = \left\{ \begin{array} { l l } { 1 , } & { \left| i - j \right| = 0 , } \\ { 1 1 / 1 2 , } & { \left| i - j \right| = 1 , } \\ { 5 / 6 , } & { \left| i - j \right| = 2 , } \\ { 3 / 4 , } & { \left| i - j \right| = 3 , } \\ { 2 / 3 , } & { \left| i - j \right| \geq 4 . } \end{array} \right.
$$

This of-diagonal decay encodes the empirical observation that contact rates are highest within an age class and decline gradually with age separation.

## S3 MCMC Specifications and Diagnostics

The conventional Bayesian calibration baselines reported in the main text were implemented in CmdStan (Carpenter et al., 2017) using the No-U-Turn Sampler (NUTS), an adaptive Hamiltonian Monte Carlo algorithm (Hofman and Gelman, 2014). For both the simulation study and the real-data application, we ran four independent chains with max treedepth= 10 and adapt delta= 0.85. The bounded parameter declarations listed below had no additional priordensity statements in the Stan programs and therefore induce uniform priors over their respective supports.

In both analyses, the time-series likelihood was a Gaussian process centered at the relevant forward-model trajectory, following the KOH discrepancy formulation (Kennedy and O’Hagan, 2001). In the notation of the main text,

$$
Z \mid \theta , \tau , \kappa , \zeta \sim { \mathcal N } _ { T } \{ Y ( \theta ) , \Sigma ( \theta , \tau , \kappa , \zeta ) \} ,\tag{S1}
$$

where

$$
\Sigma _ { u v } ( \pmb \theta , \tau , \kappa , \zeta ) = \sigma _ { Y } ^ { 2 } ( \pmb \theta ) \left[ \zeta \mathbb { I } ( u = v ) + \kappa \exp \left\{ - \frac { | u - v | } { \tau } \right\} \right] , \qquad u , v = 1 , \dots , T .\tag{S2}
$$

Here $\tau$ is the correlation range, κ is the partial sill, ζ is the nugget, and $\sigma _ { Y } ^ { 2 } ( \pmb { \theta } )$ is the temporal variance of the forward trajectory, as defined in Section 3.4.1 of the main text. The covariance hyperparameters had independent priors κ ∼ Inv-Gamma(2, 1) and $\zeta \sim \mathrm { { I n v - G a m m a } ( 2 , 1 ) }$ , where the second argument is the scale parameter. The prior support for $\tau$ difered between the two analyses as specified below.

## S3.1 Simulation Study

For the simulation study, the calibration parameters had independent priors

$$
\theta _ { 1 } , \theta _ { 2 } , \theta _ { 3 } \stackrel { \mathrm { i n d } } { \sim } \mathrm { U n i f o r m } ( 0 , 1 ) ,
$$

and the $\mathrm { G P }$ range parameter had prior

$$
\tau \sim \mathrm { U n i f o r m } ( 0 . 0 5 \times 5 3 , 0 . 2 0 \times 5 3 ) = \mathrm { U n i f o r m } ( 2 . 6 5 , 1 0 . 6 0 ) .
$$

The mean in equation (S1) was the Gaussian density-shaped forward trajectory used in the main text,

$$
Y ( \pmb \theta , s ) = 0 . 3 + \frac { \theta _ { 1 } + 0 . 3 } { \sqrt { 2 \pi ( \theta _ { 3 } + 0 . 1 ) } } \exp \biggl \{ - \frac { ( s - \theta _ { 2 } + 0 . 5 ) ^ { 2 } } { \theta _ { 3 } + 0 . 1 } \biggr \} ,
$$

evaluated at $T = 5 3$ equally spaced values of s on [−2, 2].

Each of the four chains was run for 100,000 iterations: the first 30,000 iterations were discarded as warmup and the remaining 70,000 iterations were retained for posterior inference. Thus, the four chains contributed 280,000 post-warmup draws in total. Figure S1 shows the MCMC diagnostics for a single synthetic observation data: Specifically, the chain-specific marginal posterior densities and post-warmup traces for the three forward-model parameters and the three GP covariance hyperparameters. The density curves from the four chains largely overlap, and the traces show no visually apparent chain-specific drift.

![](images/4520599550880d42cd51a37339ae3deefec1b272b4ddb54ca500c08d9a5878be.jpg)

![](images/23388072cd7a323f8fd17d2b68d3fa68aa1a7e0cf9eadb0c7a2d74727353e81f.jpg)

![](images/b348602080704d651cb4ddb40097456a5499a68ac323ca6592c11d5c46d1b8fd.jpg)

![](images/dd1ca817294358c9ca6dc711391d627117ade11d3fd7acc4abd706239499cdf4.jpg)

![](images/d3ded4120acf3b35f95e860e2c3184f33efc8741cd798c1b1ef4a0758050d477.jpg)

![](images/805b511e72240ab7c0b81249f3f4f4da160e83fc19e5d3e866678b2db68c3f39.jpg)

![](images/307065a5c79b8446cf95fe4b1e1205e8bcfbf9c0c6266db92405bfef939b52a6.jpg)

![](images/e8a5cd70edb3439681dcfa5e9ac1d95bda9ffbe298705c5abf9ab91d5806a9e2.jpg)

![](images/e846761bcb1b35e0c4b502b8e9903b937e68574eb343e38ce628beac0639d4b6.jpg)

![](images/df3a0504decf9f8a6328753cda16457cc5fd3af0551e21c93ec965846c7ccdb8.jpg)

![](images/657619b7efd8f7a7e77e7cd675d5d365f57f3de5b319e979540929d8c2317d5d.jpg)

![](images/766341e14353cdc750169c177d74b18cdf8f3cc6851ef913080f5ff8f5ff64e6.jpg)  
Figure S1: MCMC diagnostics for the simulation study. The left column displays chain-specific marginal posterior density estimates, and the right column displays the corresponding postwarmup traces for $\theta _ { 1 } , \theta _ { 2 } , \theta _ { 3 } , \tau , \kappa ,$ and ζ. Each color represents one of the four NUTS chains.

## S3.2 Real-Data Application

For the measles application, the six forward-model parameters had independent uniform priors induced by the following bounds:

$$
\begin{array} { r l } & { \beta \sim \mathrm { U n i f o r m } ( 0 . 5 \times 1 0 ^ { - 6 } , 2 . 5 \times 1 0 ^ { - 6 } ) , \quad N _ { 0 } \sim \mathrm { U n i f o r m } ( 1 . 4 \times 1 0 ^ { 5 } , 3 . 0 \times 1 0 ^ { 5 } ) , } \\ & { f _ { E } \sim \mathrm { U n i f o r m } ( 1 0 ^ { - 4 } , 1 0 ^ { - 3 } ) , \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { a _ { \mathrm { r t } } \sim \mathrm { U n i f o r m } ( 0 . 0 9 , 0 . 3 9 ) , \quad \quad \quad t _ { 0 } \sim \mathrm { U n i f o r m } ( 0 . 1 , 1 0 ) . } \end{array}\tag{S3}
$$

For this analysis, the GP range parameter had prior

$$
\tau \sim \mathrm { U n i f o r m } ( 0 . 1 \times 5 3 , 0 . 3 \times 5 3 ) = \mathrm { U n i f o r m } ( 5 . 3 , 1 5 . 9 ) .
$$

The weekly infected-population time series followed equations (S1)–(S2), with the deterministic age-structured SEIR trajectory replacing the synthetic Gaussian density-shaped mean. The age-proportion observations supplied a second likelihood component. Using the age proportions obtained from the SEIR trajectory (A(θ)), we defined the likelihood of the observed proportions $A _ { \mathrm { o b s } } = ( 0 . 4 2 , 0 . 3 0 , 0 . 2 8 ) ^ { \top }$ as

$$
A _ { \mathrm { o b s } } \mid \theta \sim \mathrm { D i r i c h l e t } \{ e _ { 2 } A ( \theta ) \} , \qquad e _ { 2 } = 3 0 0 .\tag{S4}
$$

The time-series and age-proportion likelihood contributions were conditionally independent given the forward-model parameters and GP hyperparameters.

Each of the four chains was run for 10,000 iterations, with 3,000 warmup iterations followed by 7,000 retained draws. The combined post-warmup sample therefore contained 28,000 draws. Figure S2 reports MCMC diagnostics for the observed measles outbreak data: Specifically, the chain-specific marginal posterior densities and post-warmup traces for the six SEIR parameters and the three GP covariance hyperparameters. The four density curves show substantial overlap, and the retained traces show no obvious long-term chain-specific drift.

![](images/3bee35718911ad1155d3d7a6887059ea47c9b671d2d050cbd765a2b4a8ed4a05.jpg)  
Figure S2: MCMC diagnostics for the real-data application. The left column displays chainspecific marginal posterior density estimates, and the right column displays the corresponding post-warmup traces for $\beta , N _ { 0 } , f _ { E } , a _ { \mathrm { s h } } , a _ { \mathrm { r t } } , t _ { 0 } , \tau , \kappa .$ , and $\zeta .$ Each color represents one of the four NUTS chains.

## S4 Ablation Study for Parameter-Set Selection

We conducted an ablation study to examine the empirical behavior of the prediction intervals when the conformal risk control step is removed. In the full procedure, the retention threshold is selected by CRC (Angelopoulos et al., 2024) to control the expected miscoverage loss. In this ablation, we instead bypass this calibration step and retain a fixed proportion of candidate draws. Specifically, we retain the top 90% of candidate draws according to two simple, non-conformal selection rules. The radius rule retains the draws with the smallest Euclidean norm of the latent sample, whereas the posterior rule retains the draws with the largest posterior density values.

After this uncalibrated filtering step, all remaining components of the pipeline are kept unchanged. The retained draws are propagated through the simulator, transformed back to the original data scale, clipped to the parameter bounds, and evaluated using the same fixed-rank empirical miscoverage loss as in the main analysis. Thus, this ablation isolates the efect of replacing the CRC-selected parameter set with a fixed 90% parameter set. The resulting miscoverage values should therefore be interpreted as empirical diagnostics, not as conformal risk-controlled guarantees.

Table S1: Ablation study without conformal risk control. Instead of selecting the retention threshold by CRC, each rule retains a fixed 90% of candidate draws. RMSE is computed between the observed trajectory and the median of the model outputs, and the average interval length is averaged over time.
<table><tr><td>Data setting</td><td>Filtering rule</td><td>Miscoverage</td><td>RMSE</td><td>Avg. interval length</td></tr><tr><td>Simulated data</td><td>Radius</td><td>0.0000</td><td>0.0119</td><td>0.0597</td></tr><tr><td>Simulated data</td><td>Posterior</td><td>0.0000</td><td>0.0119</td><td>0.0597</td></tr><tr><td>Real data</td><td>Radius</td><td>0.0000</td><td>1041.64</td><td>6578.77</td></tr><tr><td>Real data</td><td>Posterior</td><td>0.0000</td><td>1077.62</td><td>6078.54</td></tr></table>

For the simulated data, both uncalibrated filtering rules attain zero empirical miscoverage, with nearly identical RMSE and average interval length. A similar pattern appears in the realdata analysis: both rules attain zero empirical miscoverage, with similar RMSE and average interval length.

These results suggest that, in these particular examples, retaining a fixed 90% of candidate draws produces intervals with empirical miscoverage similar across the two non-conformal filtering criteria. However, unlike the CRC-based procedure, this fixed-retention ablation does not adapt the threshold to satisfy a prescribed risk level and does not provide the finite-sample risk guarantee. The ablation therefore supports the role of CRC as the component that converts a reasonable parameter-set construction into a calibrated prediction procedure, rather than merely serving as another heuristic filtering rule.

![](images/f910f7eb865c9bdf58e8bd9b4cfa001be56ed54acf8f4ca7986ed3d693867f76.jpg)  
(a) Radius-based filtering.

![](images/2efde643dc85ab0ed7439d3c6ec6a7930cea483f1fe6883a00acb05e96d2d08f.jpg)  
(b) Posterior-based filtering.  
Figure S3: Ablation results for the simulated-data experiment. Both panels retain 90% of the candidate draws and display the median model output, the 90% fixed-rank prediction interval, and the synthetic observation.

![](images/5071f5ea93fc2e062a6a7375e623b46fe5ae73b3c3eb917581f99666d56f8fe2.jpg)  
(a) Radius-based filtering.

![](images/82745f562a991d5fd9353423a143f76f6a731358c2dc82203f03c8476bfb13c4.jpg)  
(b) Posterior-based filtering.  
Figure S4: Ablation results for the real-data experiment. Both panels retain 90% of the candidate draws and display the median model output, the 90% fixed-rank prediction interval, and the observed data.