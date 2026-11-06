# Conformal Prediction for Time-Dependent PDEs

Joshua Stiller<sup>∗</sup> LMU Munich, MCML

Annika Schneider TU Munich, Helmholtz AI, relAI, MCML

Eyke Hüllermeier LMU Munich, MCML, DFKI

## Abstract

Uncertainty quantification is crucial in scientific machine learning, where models inform safety-critical tasks such as flood forecasting, financial risk management, and thermal control in machines. Conformal prediction provides distributionfree coverage guarantees, but in time-dependent settings common to physics and engineering, these guarantees can break down, leading to systematic undercoverage. We study this problem in the context of surrogate models for time-dependent physical systems described by partial differential equations. We prove that in a function space setting, distributions at arbitrarily close times can be mutually singular, making exact coverage guarantees impossible. We further show that in discretized settings, the total variation distance between distributions often saturates exponentially fast with the grid resolution, mandating restrictions on the spatial discretization to ensure coverage. Finally, for certain settings, we show how to use principled weighted conformal prediction to obtain finite-sample coverage guarantees over growing time horizons.

## 1 Introduction

Many problems in physics and engineering, such as weather prediction, aerodynamics, and financial modeling, are governed by partial differential equations (PDEs). Classical numerical solvers are accurate but computationally expensive, often scaling poorly with the dimensionality of the data. AI-based surrogate models have emerged as a promising alternative, providing fast approximations of PDE solutions. Prominent examples include physics-informed neural networks (PINNs) [29], DeepONets [22], and Neural Operators [20, 21].

Despite these advances, surrogate models still lack principled mechanisms for uncertainty quantification, which are vital, as model predictions often inform safety-critical decisions in science and engineering. Conformal prediction (CP) [33] provides a principled framework, producing distribution-free uncertainty sets with guaranteed marginal coverage. These guarantees, however, rely on exchangeability between calibration and test samples, a condition that is frequently violated in time-dependent PDEs.

Non-stationarity in time-dependent PDEs Let u denote the solution of a time-dependent PDE at time t. In practice, we are interested in predicting $u _ { t + \delta }$ for several time steps $\delta > 0$ , beyond the available training and calibration data. Unless u<sub>t</sub> is a stationary process, test samples follow a different distribution than observed calibration samples, breaking the exchangeability assumption required by conformal prediction (see Figure E.1). This type of non-stationarity is ubiquitous: sudden shocks (e.g., stock market crashes), long-term structural changes (e.g., climate trends), and limited development windows (e.g., laboratory testing) all produce systematic shifts in the data distribution (see Figure 1).

![](images/765348d30c166b8ab687426cfa2f9e94e5b6e73e8ed136ccf0f16bcdb5159bee.jpg)

![](images/ab3c05f7e1c7b9d330ec61e36342d1bc6fb8e586b621be1b8210858ee67c419a.jpg)

![](images/7171af8186d716cf1ea6278001ab9d10e902476fa80cd2a0b6711af9324f614d.jpg)  
Figure 1: Examples of different types of non-stationarity that can arise in time-dependent systems. (a) Sudden distribution shift due to external shocks, illustrated by a stock market crash. (b) Longterm structural changes, as in climate time series. (c) Limited observation window during system development, where calibration and testing occur on different parts of the trajectory.

Why existing methods fall short Existing CP approaches for time-dependent settings either rely on assumptions like mixing, which generally fail for PDE dynamics, or relax to asymptotic guarantees that average coverage over time and cannot guarantee per-step coverage [9, 10]. In theory, weighted conformal prediction (WCP) [32, 3] provides finite-sample coverage under distribution shift, but it requires likelihood ratios $p _ { \mathrm { t e s t } } / p _ { \mathrm { c a l } }$ that are generally unavailable for PDE surrogates. As a result, no current approach provides finite-sample coverage at every future time step. This raises a structural question: when can CP guarantees hold for time-dependent PDEs, and how should CP be applied to obtain them? We answer with theoretical limits and practical guidelines.

## Our contributions

1. We show that in function-space, already for the heat equation with Gaussian initial conditions, the total variation (TV) distance between solutions at any two distinct times is maximal. This rules out non-exchangeable CP guarantees on the function-space view common in the neural operator literature.

2. We show that for discretized PDEs the TV distance often approaches 1 exponentially fast with the grid granularity. This gives a practical guideline: rather than calibrating on the full prediction grid, we should restrict to a coarser grid or the spatial subregion of interest, as we do in our real-world experiments.

3. We identify assumptions under which closed-form likelihood ratios and finite-sample WCP guarantees at every future time step exist in the discretized setting and further lift this discrete guarantee to the true continuous solution.

We empirically validate these results on synthetic PDEs and three real-world datasets.

## 2 Related work

CP for PDEs is closely related to conformal prediction for time series. Existing approaches can be roughly categorized as follows:

Access to exchangeable data Some work treats entire solution trajectories as exchangeable units (e.g., Ma et al. [23] for Neural Operators, Moya et al. [25] for DeepONets, or Gray et al. [13] and Gopakumar et al. [12, 11] for general surrogate models). These approaches fail when extrapolated to future time steps beyond the observed trajectory, as the distribution of the solution changes with time.

Relaxing to asymptotic guarantees Instead of requiring marginal coverage at every time step, asymptotic coverage provides guarantees on average over time. The seminal work on adaptive conformal prediction [9] tightens or loosens conformal bands based on past miscoverage. Several refinements have been proposed [e.g., 1, 37, 10]. For non-critical tasks and slowly drifting time series, this can be a viable approach. In many PDE problems, however, undercoverage at a single time step can be fatal and it is a priori unknown how fast changes appear.

Exploiting additional properties To extend CP beyond exchangeability, some work assumes additional properties on the data. For example, stationarity, (strong) mixing of the process or residuals, or requirements on the accuracy of the predictive model [e.g., 27, 37, 2, 38, 36]. Depending on the assumptions, both asymptotic and per-time-step guarantees can be obtained.

Distributional assumptions Generally, CP is distribution-free, but when certain properties of the distribution are known, they can be exploited to obtain coverage guarantees without exchangeability. In full generality, Barber et al. [3] show that coverage can be lower bounded in terms of the total variation (TV) distance between calibration and test distributions. Farinhas et al. [8] extend this to conformal risk control. Under covariate or label shift, more specialized approaches achieve marginal coverage when the distribution shift can be quantified [32, 28]. In practice, these methods are rarely used in a principled way, as the required density ratios are typically unknown. A related approach comes from Harris and Liu [15], which replaces global exchangeability with a local assumption across nearby time points.

## 3 Background

## 3.1 Conformal prediction

Conformal prediction (CP) is a framework for constructing prediction sets with marginal finite-sample coverage guarantees [33]. In the standard split setting, a model is trained on a dataset $\mathcal { D } _ { \mathrm { t r a i n } }$ and calibrated on a dataset $\mathcal { D } _ { \mathrm { c a l } }$ , yielding a set-valued predictor such that, for a test sample $( x _ { \mathrm { t e s t } } , y _ { \mathrm { t e s t } } )$

$$
\begin{array} { r } { \mathbb { P } \big ( y _ { \mathrm { t e s t } } \in \mathcal { C } ( x _ { \mathrm { t e s t } } ) \big ) ~ \ge ~ 1 - \alpha , } \end{array}
$$

where α denotes the miscoverage level. This guarantee relies on exchangeability of calibration and test samples, meaning that their joint distribution is invariant under permutations. When exchangeability is violated, coverage may fail.

Conformal prediction beyond exchangeability When calibration and test distributions differ but are related through a likelihood ratio, weighted conformal prediction (WCP) provides a natural extension of standard CP [32]. A canonical example is covariate shift, where calibration samples $( x _ { i } , y _ { i } )$ for $i \in \{ 1 , \ldots , n \}$ are drawn from $\mathcal { P } _ { X } \times \mathcal { P } _ { Y \mid X }$ , while test samples $( x _ { \mathrm { t e s t } } , y _ { \mathrm { t e s t } } )$ follow $\mathcal { Q } _ { X } \times \mathcal { P } _ { Y \mid X }$ , i.e., the conditional distribution of $Y$ given X remains unchanged. In this setting, valid coverage guarantees can be recovered by reweighting the calibration quantiles according to the density ratio

$$
w _ { i } \ \propto \ \frac { p _ { \mathrm { t e s t } } ( x _ { i } ) } { p _ { \mathrm { c a l } } ( x _ { i } ) } , \quad \forall x _ { i } \in \mathcal { D } _ { \mathrm { c a l } } \cup \{ x _ { \mathrm { t e s t } } \} , \quad \sum _ { i } w _ { i } = 1 ,\tag{1}
$$

where $p _ { \mathrm { c a l } }$ and $p _ { \mathrm { t e s t } }$ denote the densities of ${ \mathcal { P } } _ { X }$ and $\mathcal Q _ { X }$ (see Appendix F for more details).

In the most general case, Barber et al. [3] and, similarly, Farinhas et al. [8] provide corrections for the conformal guarantees based on the TV distance between calibration and test distribution:<sup>1</sup>

$$
\mathbb { P } \big ( y _ { \mathrm { t e s t } } \in \mathcal { C } ( x _ { \mathrm { t e s t } } ) \big ) \ \ge \ 1 - \alpha - \sum _ { i = 1 } ^ { n } \tilde { w } _ { i } d _ { \mathrm { T V } } ( z , z ^ { i } ) ,\tag{2}
$$

where $\tilde { w } _ { i } = w _ { i } / ( w _ { 1 } + \cdot \cdot \cdot + w _ { n } + 1 )$ , with $w _ { i } \in [ 0 , 1 ] , z = ( ( x _ { 1 } , y _ { 1 } ) , \ldots , ( x _ { n } , y _ { n } ) , ( x _ { \mathrm { t e s t } } , y _ { \mathrm { t e s t } } ) )$ and $z ^ { i }$ arises from permuting the test pair in z with the $i ^ { \mathrm { t h } }$ calibration pair. These approaches are rarely used in a principled way because they require knowing the corresponding distributions. We show when and how they can be used on PDE problems to regain coverage over time.

## 3.2 PDEs as operator mappings

Many dynamical systems in physics and engineering can be described by evolution equations of the form

$$
\frac { \partial u } { \partial t } ( { \bf x } , t ) = \mathcal { L } _ { \bf x } u ( { \bf x } , t ) ,
$$

![](images/762dc884fb8db3ee8bb66f9afc5649381532e9c5d9ca31a95ce1943d6312da24.jpg)  
Figure 2: Overview of the theoretical results. The colors differentiate between function space (blue) and discretized space (yellow).

where u : $\Omega \times [ 0 , \infty )  \mathbb { R }$ is the state variable, x denotes spatial coordinates on a bounded domain $\Omega \subset \mathbb { R } ^ { d } , t \geq 0$ is the time, and $\mathcal { L } _ { x }$ is a (possibly nonlinear) differential operator acting on the spatial variable x. In this paper, we are interested in the Cauchy-type problem, where we consider boundary conditions on Ω<sup>¯</sup> and initial conditions $u ( { \pmb x } , 0 )$ from some Banach space of functions $( A , \parallel \cdot \parallel _ { A } )$ and are interested in a solution u $( { \pmb x } , t )$ in some Banach space of functions $\left( \mathcal { U } _ { t } , \parallel \cdot \parallel _ { \mathcal { U } _ { t } } \right)$ . We will further only consider well-posed problems with a time-invariant differential operator, where we can define solution operators $\bar { \mathcal { G } } _ { t } : \mathcal { A } \overset { \left. } { \right. } \mathcal { U } _ { t } , \mathcal { G } _ { t } ( a ) = u ( \cdot , t )$ that uniquely map an initial condition to a solution function $\boldsymbol { u } ( \boldsymbol { x } , t )$ and the map $t \mapsto \mathcal { G } _ { t }$ is continuous in t. In the rest of the paper, we will assume that all functions come from the same space, so $\mathcal { A } = \mathcal { U } _ { t }$ for all $t \geq 0$ , to simplify the notation, but the results apply more generally.

## 4 WCP for time-dependent PDEs

We begin by specifying the conformal setup, followed by the main theoretical contributions which we summarize in Figure 2.

## 4.1 Problem setting

From initial conditions to solutions To apply CP in the PDE setting, we first specify the underlying distributions. Let u(x, t) be the solution of a PDE and $S _ { \delta } : \mathcal { U } \to \mathcal { U }$ be the solution operator mapping $\boldsymbol { u } ( \boldsymbol { x } , t )$ to $u ( { \pmb x } , t + \dot { \delta } )$ for any $t \geq 0$ . We may assume the initial condition (or in practice: our initial measurement) is distributed according to $u ( { \pmb x } , 0 ) \sim \mathcal { P } _ { 0 }$ . Then, $S _ { \delta }$ induces a pushforward measure<sup>2</sup> $\mathcal { P } _ { \delta } : = ( S _ { \delta } ) _ { \# } \mathcal { P } _ { 0 }$ . If we now consider consecutive time points, our distribution changes over time:

$$
{ \mathcal { P } } _ { 0 } ~ { \xrightarrow { S _ { \delta } } } ~ { \mathcal { P } } _ { \delta } ~ { \xrightarrow { S _ { \delta } } } ~ { \mathcal { P } } _ { 2 \delta } ~ { \xrightarrow { S _ { \delta } } } ~ \cdot \cdot \cdot
$$

Thus, we obtain a sequence of probability distributions $\{ \mathcal { P } _ { k \delta } \} _ { k \in \mathbb { N } }$ on the same function space, evolving under the PDE dynamics. Calibration and test data drawn from different $\mathcal { P } _ { t }$ are therefore not exchangeable: although they live in the same function space, their distributions shift with time.

Datasets Datasets of size n typically consist of pairs $( u _ { i } ( { \pmb x } , t ) , u _ { i } ( { \pmb x } , t + \delta ) )$ ), for $i \in \{ 1 , \ldots , n \}$ generated from initial conditions $u _ { i } ( { \pmb x } , 0 ) \sim \mathcal { P } _ { 0 }$ . In practice, the $u _ { i }$ can either arise from measurements or numerical simulations. In both simulations and real-world settings, the states $u _ { i } ( { \pmb x } , t )$ are discretized on a grid with p points, yielding vectors in R<sup>p</sup>. In this case, we omit the spatial dependence and denote the discretized solution at time t by $\mathbf { \boldsymbol { \mathsf { u } } } _ { i } ^ { t } : = ( u _ { i } ( x _ { 0 } , t ) , \ldots , u _ { i } ( x _ { p - 1 } , \mathbf { \boldsymbol { \mathsf { \hat { t } } } } ) ) ^ { T } \in \mathbf { \bar { \mathbb { R } } } ^ { p }$ (bold font). Nevertheless, since neural operators can generalize across grids, the function-space viewpoint remains relevant and is commonly adopted in the literature [15, 13, 24]. Accordingly, we consider both the functional representation (Section 4.2) and the discretized setting (Section 4.3). Throughout the paper we assume that our calibration dataset consists of pairs $( u _ { \mathrm { c a l } , i } ( { \pmb x } , t _ { \mathrm { c a l } } ) , u _ { \mathrm { c a l } , i } ( { \pmb x } , t _ { \mathrm { c a l } } + \delta ) )$ from some time $t _ { \mathrm { c a l } }$ and our test dataset consists of pairs $( u _ { \mathrm { t e s t } , i } ( { \pmb x } , t ) , u _ { \mathrm { t e s t } , i } ( { \pmb x } , t + \delta ) )$ for some $t \geq t _ { \mathrm { c a l } }$ . We omit the subscripts cal and test for notational simplicity.

## 4.2 Distribution shifts in function spaces

We will now consider CP on PDEs, starting with the function-space perspective, where the data points $u _ { i } ( { \pmb x } , t )$ are elements of a suitable function space. As the data is not exchangeable, we must resort to a more general CP method. WCP is not suitable because the function-space densities<sup>3</sup> cannot be evaluated, but if the TV distance between the distributions at times t and $t + \delta$ were moderate, CP coverage guarantees for the prediction at $t + \delta$ could still be recovered using the approach from Equation (2). The following theorem shows that even for simple problems, like the heat equation with Gaussian initial distribution, the TV distance between the solution distributions $\mathcal { P } _ { t } , \mathcal { P } _ { t + \delta }$ at different time points $t , t + \delta$ is always maximal $d _ { \mathrm { T V } } ( \mathcal { P } _ { t } , \mathcal { P } _ { t + \delta } ) = 1$ , for all $t \ge 0 , \delta > 0$ . This is representative of a broader phenomenon that $^ { 6 6 } [ . . . ]$ measures in infinite-dimensional spaces have a strong tendency of being mutually singular.” [14]. As a direct consequence, regular CP and any method requiring distributional similarity between calibration and test fails. In particular, exact WCP via Equation (1) is inapplicable.

Theorem 4.1. Consider the one-dimensional heat equation $\partial _ { t } u = \partial _ { x x } u$ on the domain $\Omega = ( 0 , 1 )$ with Dirichlet boundary conditions, $t \geq 0$ . Suppose the initial condition $u _ { 0 } ( x )$ is sampledfrom a Gaussian randomfield $\mathcal { P } _ { 0 } = \mathcal { N } ( \mathbf { 0 } , ( I - \Lambda ) ^ { - 1 } )$ , where Λ is the Laplace operator on Ω. Then, the TV distance between the solution distributions at any two time points is maximal:

$$
d _ { \mathrm { T V } } ( \mathcal { P } _ { t } , \mathcal { P } _ { t + \delta } ) = 1 , \quad \forall t \ge 0 , \ : \delta > 0 .
$$

Proofsketch: The proof uses the Feldman–Hájek theorem [6], which shows that two Gaussian measures on a Hilbert space are either equivalent or mutually singular. We then calculate the Cameron–Martin spaces of both measures and show they differ, which tells us that the measures cannot be equivalent. The full proof is provided in Appendix G.1. □

We will now discuss how, despite the issue above, coverage guarantees can be achieved in practice.

## 4.3 Coverage guarantees in practice

Theorem 4.1 shows that a pure function-space perspective renders TV-based corrections vacuous, even for arbitrarily close time points. This is not a contradiction to practical CP on PDE surrogate models: in practice, we always observe or simulate on a finite number of degrees of freedom. Therefore, we shift the perspective to the discretized state $\mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } u _ { i } ^ { t } \in \mathbb { R } ^ { p }$ . As we still consider the non-exchangeable setting, we either need additional inductive biases or rely on TV-based corrections. We will consider both approaches, starting with the former.

While CP is inherently distribution-free, related methods such as WCP [32, 8] give up this property in exchange for relaxing the exchangeability assumption. We will now identify a setting relevant to time-dependent PDEs where closed-form likelihood ratios are available so that WCP can be applied.

First, note that in physics, Gaussianity is often assumed for the initial distribution, because the aggregate effect of many small independent perturbations forming the initial condition is approximately Gaussian by the central limit theorem. This is well supported by the literature [21, 30, 12, 39, 11].

Second, observe that when considering linear PDEs solved with standard numerical schemes, the time evolution is affine $\pmb { u } _ { i } ^ { t + \delta } = \pmb { A } _ { \delta } \pmb { u } _ { i } ^ { t } + \pmb { b } _ { \delta }$ , with $\ b { A } _ { \delta } \in \mathbb { R } ^ { p \times p }$ and $b _ { \delta } \in \mathbb { R } ^ { p }$ . If the initial condition is Gaussian, ${ \pmb u } _ { i } ^ { 0 } \sim \mathcal { N } ( { \pmb \mu } _ { 0 } , \dot { \pmb \Sigma } _ { 0 } )$ , then $\mathbf { \Delta } \mathbf { u } _ { i } ^ { t }$ remains Gaussian for all $t \geq 0$ , with mean $\pmb { \mu } _ { t }$ and covariance $\Sigma _ { t }$ evolving explicitly. Consequently, for linear PDEs with Gaussian initial distribution the calibration and test distributions admit closed-form likelihood ratios, enabling WCP.

Proposition 4.2. Consider a linear evolution equation on a bounded domain with linear boundary conditions and a deterministic source term. If the calibration dataset at time $t _ { \mathrm { c a l } }$ and a test sample $u _ { \mathrm { t e s t } } ^ { t }$ at time $t \geq t _ { \mathrm { c a l } }$ are generatedfrom the same affine numerical scheme on independent initial conditions sampled from a Gaussian distribution, then we can construct a likelihood-weighted conformal predictor such tha

$$
\mathbb { P } \big ( \boldsymbol { u } _ { \mathrm { t e s t } } ^ { t + \delta } \in \mathcal { C } _ { t + \delta } ^ { \mathrm { W C P } } ( \boldsymbol { u } _ { \mathrm { t e s t } } ^ { t } ) \big ) \geq 1 - \alpha .
$$

![](images/c321f6889b8d2eb81ee06e232caeef0a07c860ce757cbc44b827448f9c24e8b8.jpg)  
(a) Drift PDE, Proposition 4.4

![](images/b493bea266711a6b15138bcfb0c41cc5e11c2dd5512c7cc15d607c1e10351fe5.jpg)  
(b) Backward heat equation, Proposition 4.7  
Figure 3: Visualization for Proposition 4.4 and Proposition 4.7. (a) Coverage decays exponentially with grid granularity for the baselines, while WCP grows infinite bands (dotted lines). The theoretical lower bound follows Barber et al. [3]. (b) Coverage of the analytical solution of the backward heat equation with numerical conformal bands. With finer granularity, coverage over the analytic solution converges to the target as the reconstruction error vanishes. Also here, the coverage decline effect of (a) would start dominating at higher granularity. More in Appendix H.4.

Proofsketch: As the initial distribution is Gaussian and the PDE linear, we can show that the solution at each time step is Gaussian with known mean and covariance. With this, we calculate the exact weights for WCP. The full proof is provided in Appendix G.2. □

Remark 4.3 (Other initial distributions). This result extends to any other initial distribution that is closed under affine transformations (e.g., Cauchy, Laplace, and Logistic). We include additional experiments sampling from different initial distributions, including heavy-tailed Cauchy initial conditions, in Appendix K.

We only know the input-time distribution, not the prediction-time one (Gaussian with unknown moments), so the distribution itself cannot be used to infer uncertainty directly.

Even though WCP is theoretically attainable in the setting we just identified, the resulting conformal bands can become too large in practice to provide useful information. In the next section, we present a way to alleviate this: We show that the grid granularity is a powerful lever for controlling the TV distance and conformal band size, and with this, the applicability of WCP.

## 4.4 Effect of grid granularity

So far we worked with $\mathbf { \Delta } \mathbf { \boldsymbol { u } } _ { i } ^ { t }$ at arbitrary granularity. Refining the grid recovers the function-space limit, where Theorem 4.1 gives $d _ { \mathrm { T V } } = 1$ , raising the question of how granularity affects CP. We analyze this question for the drift PDE which represents slight non-stationary behavior. The following proposition shows that for this PDE, the TV distance approaches 1 exponentially fast in the grid size, so standard CP approaches become inapplicable even under slight drifts.

Proposition 4.4. Consider the drift PDE $\partial _ { t } u ( x , t ) = c$ , with $x \in ( 0 , 1 ) , t \geq 0 , c \in \mathbb { R }$ , and compatible boundary conditions. Define z and $z ^ { i }$ as in Equation (2) with n i.i.d. calibration pairs at times $( t _ { 1 } , t _ { 1 } + \delta )$ and a test pair at time $( t _ { 2 } , t _ { 2 } + \delta )$ ) from an affine scheme on a grid with p points and Gaussian initial conditions. Then, the total variation distance between the laws ofz and $z ^ { i }$ is

$$
d _ { \mathrm { T V } } \left( z , z ^ { i } \right) ~ = ~ 2 \Phi \Bigl ( { \frac { \sqrt { 2 p c ^ { 2 } } \left| t _ { 1 } - t _ { 2 } \right| } { 2 } } \Bigr ) - 1 ,
$$

where Φ is the cumulative distribution function of the standard normal distribution. Hence, $d _ { \mathrm { T V } } ( z , z ^ { i } ) \to 1$ exponentially fast in the grid size p and with $i t ,$ the coverage guarantee from Barber et al. $I 3 J$ vanishes exponentially.

Moreover, for likelihood-ratio weighting as in Equation (7) evaluated at the distribution centers $\pmb { x } _ { \mathrm { t e s t } } : = \mathbb { E } [ \pmb { u } _ { n + 1 } ^ { t _ { 2 } } ]$ and $\pmb { x } _ { \mathrm { c a l } } : = \mathbb { E } [ \pmb { u } _ { i } ^ { \top _ { 1 } } ]$ ], the test weight satisfies

$$
w _ { \mathrm { t e s t } } = 1 - O \big ( n e ^ { - p c ^ { 2 } ( t _ { 1 } - t _ { 2 } ) ^ { 2 } } \big ) , \quad w _ { \mathrm { c a l } } = O \big ( e ^ { - p c ^ { 2 } ( t _ { 1 } - t _ { 2 } ) ^ { 2 } } \big ) .
$$

Hence, the weighted conformal bands become infinite at an exponential rate.

Proofsketch: The TV distance follows from Theorem 4.5 with ${ \pmb w } = { \bf 1 }$ and $\Sigma _ { 0 } = I _ { p } ,$ and for the weights, we calculate the explicit density ratio. The full proof is provided in Appendix G.6. □

We visualize Proposition 4.4 in Figure 3a. The drift example provides a good heuristic that a slight non-stationarity can lead to exponential decay of the coverage guarantees for general (non)linear PDEs. We extend this result to general linear PDEs in Proposition G.1.

The exponential growth in p relies on white-noise initial conditions, where every grid point carries independent information about the drift. In real data, neighboring grid points are usually correlated, which requires a more general statement. The following theorem replaces the grid size $p$ by an effective dimension $p _ { \mathrm { e f f } } ,$ which is small when the drift points along directions of high initial variance. Restricting CP to a subregion of the domain reduces the effective dimension and thereby improves the CP guarantees, though other approaches can be more suitable depending on the covariance structure of the data.

Theorem 4.5. Consider the drift PDE on a grid with p points, $\partial _ { t } \pmb { u } ^ { t } = c \pmb { w }$ with $t \geq 0$ , drift strength $c \in \mathbb { R }$ , and drift profile w $\in \mathbb { R } ^ { p } \left( e . g . , w = 1 \right.$ for a constant $d r i f t ) ,$ , and let the initial conditions be non-degenerate Gaussian, ${ \pmb u } ^ { 0 } \sim \mathcal { N } ( { \pmb \mu } _ { 0 } , { \pmb \Sigma } _ { 0 } )$ . For z and $z ^ { i }$ as in Proposition 4.4,

$$
d _ { \mathrm { T V } } \bigl ( z , z ^ { i } \bigr ) \ = \ 2 \Phi \Bigl ( \frac { \sqrt { 2 p _ { \mathrm { e f f } } ~ c ^ { 2 } } \left| t _ { 1 } - t _ { 2 } \right| } { 2 } \Bigr ) - 1 , \quad w i t h ~ p _ { \mathrm { e f f } } : = w ^ { \top } \Sigma _ { 0 } ^ { - 1 } w .
$$

Proofsketch: Whitening the covariance reduces the statement to the equal-covariance identity used for Proposition 4.4. The full proof is provided in Appendix G.5. □

For ${ \pmb w } = { \bf 1 }$ and $\Sigma _ { 0 } = I _ { p }$ , we have $p _ { \mathrm { e f f } } = p ,$ which recovers Proposition 4.4. Thus, the TV distance is dependent on the smoothness of the initial state and how the PDE drifts relative to it, even approaching 1 exponentially fast in the grid size.

Knowing that many PDEs of interest are non-stationary, and that fine grids are often required for accurate predictions, it may seem futile to apply non-exchangeable CP on PDEs at all. However, while the full grid is usually necessary for prediction, we often do not need uncertainty quantification over the whole space. Referencing our real-world examples in Section 5, we may find that the temperature of Lake Maggiore is most relevant at depths where certain fish are living, or that bleaching of the FRAP target is most interesting in the middle of the plate where the laser is focused. In such cases, Theorem 4.5 tells us to restrict CP to coherent regions of interest, such as the center of the bleach spot or a few neighboring depths in the lake, and leverage the more favorable properties of the marginal distribution. We demonstrate this in our experiments.

## 4.5 Applications and extensions of the framework

The previous theorem showed that non-exchangeable CP should only be applied to a subset of the grid to avoid exponential decay of the coverage guarantees and uninformative conformal bands. While WCP methods “only” suffer from vacuous prediction intervals instead of undercoverage, they face additional practical challenges: For numerical data, we can evolve the exact mean and covariance from the initial distribution. For real-world measurements, however, the true mean and covariance must be estimated from data, introducing additional uncertainty. While it is common in CP literature that some assumptions cannot be checked on real-world data (e.g., [32, 3]), we nonetheless want to give a theoretical outlook on how to account for this estimation uncertainty and recover coverage guarantees when additional assumptions can be made.

Proposition 4.6 (Robust WCP under Gaussian distributions). Under the assumptions ofProposition 4.2, suppose we additionally observe independent estimation samples $\{ u _ { j } ^ { t _ { \mathrm { c a l } } } \} _ { j = 1 } ^ { m _ { 0 } }$ and $\{ { \pmb u } _ { j } ^ { t } \} _ { j = 1 } ^ { m _ { 1 } }$ with $m _ { k } > p f o r k \in \{ 0 , 1 \}$ , and fix an arbitrary estimation uncertainty $\eta > 0$ . Then, we can construct a conformal prediction set $C _ { t + \delta } ^ { \mathrm { r o b } }$ with coverage guarantee

$$
\mathbb { P } \big ( \boldsymbol { u } _ { \mathrm { t e s t } } ^ { t + \delta } \in C _ { t + \delta } ^ { \mathrm { r o b } } ( \boldsymbol { u } _ { \mathrm { t e s t } } ^ { t } ) \big ) \ \geq \ 1 - \eta - \alpha .
$$

Smaller η will usually correspond to wider bands.

Proofsketch: We use the sample distributions of the estimated means and covariances to construct a $1 - \eta$ confidence region over the true means and covariances. Then, we take the worst case quantiles over these regions. The full proof can be found in Appendix G.3. □

Without additional structure on the covariance, this procedure is intractable in practice, and only of theoretical value. Exploring tractable problem classes $( \mathbf { e . g . }$ , with covariances diagonal in Fourier space) is left for future work. Empirically, such corrections are often unnecessary, as mean and covariance estimates converge rapidly with the number of samples. In particular, we find that constructing prediction bands from estimated mean and covariance yields consistent coverage, whereas estimating the full distribution shift, as the original WCP does, often leads to undercoverage.

## 4.6 Recovering guarantees for the true solution

While we saw in Section 4.2 that function-space coverage is generally intractable, numerical schemes typically come with deterministic error bounds of order $\check { O ( } h ^ { k } )$ in the grid size $h .$ . The following proposition lifts a discrete CP set $C _ { h }$ to a conformal set on the true solution by inflating $C _ { h }$ with this error: the convergence rate carries over (even though unknown constants typically prevent computing $C$ explicitly), and depends only on the prediction granularity rather than that used for $C _ { h }$ . See Figure 3b for an empirical demonstration.

Proposition 4.7. Let $u ( \cdot , t )$ be a PDE solution in some function space $( \mathcal { U } , \Vert \cdot \Vert _ { \mathcal { U } } )$ and ${ \mathbf { } } _ { { \mathbf { } } _ { } u _ { } ^ { t } }$ be its approximation via some numerical scheme on grid size h. Let $C _ { h }$ be a conformal prediction setfor the discretized state with coverage $1 - \alpha$ and assume that an error bound $\| u ( \cdot , t ) - \dot { I } _ { h } u ^ { t } \| _ { \mathcal { U } } \leq \dot { \varepsilon _ { h , t } } ( u _ { 0 } )$ is available via some reconstruction operator $I _ { h } : \mathbb { R } ^ { p }  \mathcal { U } ( e . g . $ ., piecewise interpolation). Then we can define a conformal set for the function-space setting, $C \subset { \mathcal { U } }$ , with $\mathbb { P } \big ( u ( \cdot , t ) \bar { \in } { \cal C } \big ) \ \geq \ 1 - \alpha$ by

$$
C : = \left\{ v \in \mathcal { U } : \exists z \in C _ { h } \mathit { w i t h } \ \| v - I _ { h } z \| _ { \mathcal { U } } \leq \varepsilon _ { h , t } ( u _ { 0 } ) \right\} .\tag{3}
$$

Proofsketch: We use the error bound to extend the conformal set for the numerical solution using the triangle inequality. The full proof is given in Appendix G.4 and a visualization in Figure 3b.

## 5 Experiments

We demonstrate our theoretical results on both synthetic and real-world data, in particular the coverage of WCP with Gaussian likelihood ratio

Experimental setup For the synthetic case, we choose the general second-order PDE framework, which covers many well-known PDEs $u _ { t } = a u _ { x x } + b u _ { x } + c u , x \in \left( 0 , 2 \pi \right)$ $t \geq 0$ where a, $b , c \in \mathbb { R }$ are tunable parameters and $u _ { 0 }$ is sampled from a multivariate Gaussian. We target setups where the PDE becomes more unstable over time $( a < 0 )$ . Otherwise, the PDE smooths out over time, leading to trivial coverage at future time steps. Hence, we consider $a = \{ - 0 . 0 0 5 , - 0 . 0 0 7 5 , - 0 . 0 1 \}$ and for the remaining parameters we choose $b , c \in \{ - 0 . 5 , 0 , 0 . 5 \}$ . See Figure H.1 for exemplary data. For real-world scenarios, we use pulsed-thermography measurements from Wei et al. [34], diffusion dynamics from FRAP-microscopy (Fluorescence recovery after photobleaching) [35] and heat-diffusion from sensors in Lake Maggiore under seasonal drift [31] (details in Appendices I.1, I.2, and I.3). We consider the following tasks, targeting 90% coverage:<sup>4</sup> (1) Recalibration case: Starting from $t _ { 0 } = 0 ,$ , we generate states at time steps $t _ { i } = t _ { 0 } + i \delta$ . We use pairs $\left( { { t } _ { i } } , { { t } _ { i + 1 } } \right)$ for calibration and test on $\left( t _ { i + 1 } , t _ { i + 2 } \right)$ for $i \in \{ 0 , \ldots , 1 8 \}$ (we also include the calibration time coverage as the first step). The inverse scenario is obtained by reversing time. (2) General case: As above, but no recalibration is allowed after $t _ { 0 } .$ corresponding to deployment at an unknown future time.

![](images/627da7294564e8b378c204b71413cc83c7ec20e39b1831379868c22e0087ac95.jpg)  
Figure 4: Coverages for varying a and fixed $b = 0 , c = 0 . 5$ on the recalibration task. The 90% target is marked by a black dashed line. For WCP Gauss and WCP Learned, the fraction of samples with infinite bandwidth $n _ { \infty }$ is indicated by green and pink dotted lines, respectively. Coverage holds for some parameters, but small variations induce undercoverage at later time steps. Only WCP Gauss meets the target throughout.

Baselines We define four baselines for our experiments. The first is a naïve implementation (Supr.) which depends on exchangeability [7]. Secondly, we use WCP with learned weights, as described in Tibshirani et al. [32] (WCP Learned), which serves as our main baseline. Thirdly, we use the LSCI method [15], a novel approach for CP on PDEs (note Appendix D). Finally, we use Dynamically-Tuned Adaptive Conformal Inference (DtACI) by Gibbs and Candès [10], an updated version of the original Adaptive CP [9] with automatic parameter selection. However, as adaptive CP requires an evaluation of the error at every step, this method is only applicable to the recalibration task. We compare these to our WCP method (WCP Gauss), which only estimates mean and covariance for each model input to construct the weights. In cases where the distributional dissimilarity of $u _ { t }$ and $u _ { t + \delta }$ is too large, the WCP methods predict infinite bands. We report the fraction of these samples as $n _ { \infty }$ . A band is wider the larger the likelihood ratio of its test sample, and infinite once the normalized test weight exceeds α (Appendix J.3). Note that in safety-critical applications, reporting trivial bands is often a more valuable result than delivering bands with undercoverage.

Additional experiments To demonstrate the effect of our derived results, Figure 3 provides visualizations of Proposition 4.4 and Proposition 4.7. More details in Appendix H.4. We additionally include an ablation study in Appendix H.3 that stress-tests WCP Gauss under reduced sample sizes and increased dimensionality. As expected, when the number of samples is insufficient relative to the problem dimensionality, instability in covariance estimation can lead to undercoverage. Coverage is stable across random calibration/test splits and, for the pulsed-thermography data, across plate assignments with retrained surrogates (Appendix J.1).

Results Exemplary figures for synthetic and real-world coverage results can be seen in Figure 4 and Figure I.2 respectively. Results for all remaining parameter combinations (incl. tables) are provided in Appendix $\bar { \mathsf { K } } . ^ { 5 }$ Across both synthetic and real-world tasks, WCP Gauss is the only method that consistently attains the target coverage. In general, WCP Gauss can estimate the weights more easily than WCP Learned, and DtACI would require longer time windows to adapt (which may require costly measurements). This does not mean that coverage with other methods is impossible. While other methods can achieve coverage in specific settings or with careful tuning or more expressive models (e.g., for WCP Learned), this behavior does not generalize reliably. In practice, small changes such as grid resolution, PDE parameters, or evaluation time steps can already lead to undercoverage. This issue is often hidden in smoothing forward problems, where residuals shrink over time, but becomes apparent in inverse problems, unstable dynamics, or blow-up regimes. Overall, WCP is the only method providing formal guarantees, and we can see empirically that this is a clear advantage as soon as our system exhibits significant dynamics.

## 6 Limitations

The convergence results for increasing grid granularity (Proposition 4.4) should be read as a heuristic for (non)linear PDEs: with increasing granularity, even small drifts can cause the TV distance to deteriorate rapidly. Of course, one can construct PDEs with no drift at all, which may exhibit more favorable TV behavior. Similar considerations apply to the vacuous TV bound in function spaces. Identifying the conditions under which these effects occur more generally is left for future work. Finally, while the lifting proposition (Proposition 4.7) is independent of the choice of PDE, the proposed WCP method (Proposition 4.2 together with Proposition 4.6) relies strictly on linearity and loses its coverage guarantees for nonlinear PDEs, where coverage depends on the PDE parameter (Appendix J.2).

## 7 Discussion

Conformal prediction for time-dependent physical phenomena is often constrained by nonexchangeable data. In this work, we investigated whether coverage guarantees can be maintained beyond the exchangeability assumption and show that this depends strongly on the problem representation. On function spaces, measures are typically mutually singular, making coverage guarantees unattainable. On discretized domains, however, we derived how weighted CP can be applied to linear PDEs to obtain coverage guarantees. We further show that these guarantees can be preserved for real-world measurements, and that CP should be restricted to a coarse, low-dimensional grid to provide meaningful guarantees. A natural next step is to extend these results to specific classes of (non)linear PDEs.

## Acknowledgments and Disclosure of Funding

The authors gratefully acknowledge the computational and data resources provided by the Leibniz Supercomputing Centre (LRZ). JS acknowledges funding from the Munich Center for Machine Learning (MCML), as well as the support he received from the METEOR Research Training Group, which is sponsored by the German Research Foundation. AS is supported by the DAAD programme Konrad Zuse Schools of Excellence in Artificial Intelligence, sponsored by the Federal Ministry of Research, Technology and Space.

## References

[1] Anastasios Angelopoulos, Emmanuel Candès, and Ryan J Tibshirani. Conformal PID control for time series prediction. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 23047–23074. Curran Associates, Inc., 2023. doi: 10.52202/075280-1000. URL https://proceedings.neurips.cc/paper\_files/p aper/2023/file/47f2fad8c1111d07f83c91be7870f8db-Paper-Conference.pdf.

[2] Andreas Auer, Martin Gauch, Daniel Klotz, and Sepp Hochreiter. Conformal prediction for time series with modern Hopfield networks. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 56027–56074. Curran Associates, Inc., 2023. doi: 10.52202/075280-2444. URL https://proceedings.neurips.cc/paper \_files/paper/2023/file/aef75887979ae1287b5deb54a1e3cbda-Paper-Conference.pdf.

[3] Rina Foygel Barber, Emmanuel J. Candès, Aaditya Ramdas, and Ryan J. Tibshirani. Conformal prediction beyond exchangeability. The Annals ofStatistics, 51(2):816–845, April 2023. ISSN 0090-5364, 2168-8966. doi: 10.1214/23-AOS2276. URL https://projecteuclid.org/journals/annals-of-statistic s/volume-51/issue-2/Conformal-prediction-beyond-exchangeability/10.1214/23-AOS 2276.full.

[4] S.S. Barsov and V.V. Ul’yanov. Estimates of the proximity of Gaussian measures. Soviet Mathematics–Doklady, 34:462–466, 1987.

[5] Lukas Biewald. Experiment Tracking with Weights and Biases, 2020. URL https://www.wandb.com/.

[6] Giuseppe Da Prato and Jerzy Zabczyk. Stochastic Equations in Infinite Dimensions. Cambridge University Press, 1 edition, December 1992. ISBN 978-0-521-38529-9 978-0-521-05980-0 978-0-511-66622-3. URL https://www.cambridge.org/core/product/identifier/9780511666223/type/book.

[7] Jacopo Diquigiovanni, Matteo Fontana, and Simone Vantini. Conformal prediction bands for multivariate functional data. Journal of Multivariate Analysis, 189:104879, May 2022. ISSN 0047259X. doi: 10.1016/j.jmva.2021.104879. URL https://linkinghub.elsevier.com/retrieve/pii/S004725 9X21001573.

[8] António Farinhas, Chrysoula Zerva, Dennis Ulmer, and Andre Martins. Non-exchangeable conformal risk control. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun, editors, International Conference on Learning Representations, pages 50952–50966, 2024. URL https://proceedings.ic lr.cc/paper\_files/paper/2024/file/de04896f011beff76c91e094f72727f4-Paper-Confe rence.pdf.

[9] Isaac Gibbs and Emmanuel Candès. Adaptive conformal inference under distribution shift. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P. S. Liang, and J. Wortman Vaughan, editors, Advances in Neural Information Processing Systems, volume 34, pages 1660–1672. Curran Associates, Inc., 2021. URL https://procee dings.neurips.cc/paper\_files/paper/2021/file/0d441de75945e5acbc865406fc9a2559-P aper.pdf.

[10] Isaac Gibbs and Emmanuel J. Candès. Conformal inference for online prediction with arbitrary distribution shifts. Journal of Machine Learning Research, 25(162):1–36, 2024. URL https://jmlr.org/papers/ v25/22-1218.html.

[11] Vignesh Gopakumar, Ander Gray, Lorenzo Zanisi, Timothy Nunn, Daniel Giles, Matt Kusner, Stanislas Pamela, and Marc Peter Deisenroth. Calibrated physics-informed uncertainty quantification. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu, editors, Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 20103–20141. PMLR, July 2025. URL https://proceedings.mlr.press/v267/gopakumar25a.html.

[12] Vignesh Gopakumar, Ander Gray, Joel Oskarsson, Lorenzo Zanisi, Daniel Giles, Matt J Kusner, Stanislas Pamela, and Marc Peter Deisenroth. Uncertainty quantification of surrogate models using conformal prediction. Machine Learning: Science and Technology, 7(1):015025, 2026. URL http://dx.doi.org /10.1088/2632-2153/ae2e7b.

[13] Ander Gray, Vignesh Gopakumar, Sylvain Rousseau, and Sebastien Destercke. Guaranteed Prediction Sets for Functional Surrogate Models. In Silvia Chiappa and Sara Magliacane, editors, Proceedings of the Forty-first Conference on Uncertainty in Artificial Intelligence, volume 286 of Proceedings ofMachine Learning Research, pages 1569–1585. PMLR, July 2025. URL https://proceedings.mlr.press/v2 86/gray25a.html.

[14] Martin Hairer. An Introduction to Stochastic PDEs, July 2023. URL http://arxiv.org/abs/0907.4 178. arXiv:0907.4178 [math.PR].

[15] Trevor Harris and Yan Liu. Locally adaptive conformal inference for operator models, July 2025. URL http://arxiv.org/abs/2507.20975.

[16] Haiyun He and Ziv Goldfeld. Information-theoretic generalization bounds for deep neural networks. IEEE Transactions on Information Theory, 71(8):6227–6247, 2025. doi: 10.1109/TIT.2025.3568697. URL https://doi.org/10.1109/TIT.2025.3568697.

[17] Wolfgang Karl Härdle and Léopold Simar. Applied Multivariate Statistical Analysis. Springer International Publishing, Cham, 2019. ISBN 978-3-030-26005-7 978-3-030-26006-4. doi: 10.1007/978-3-030-26006-4. URL http://link.springer.com/10.1007/978-3-030-26006-4.

[18] Lazaros Karageorgiou and Kyprianos-Iason Prodromidis. Limit profile for the high-temperature Curie-Weiss model, October 2025. URL http://arxiv.org/abs/2510.18793.

[19] Olivier Ledoit and Michael Wolf. A well-conditioned estimator for large-dimensional covariance matrices. Journal ofMultivariate Analysis, 88(2):365–411, February 2004. ISSN 0047259X. doi: 10.1016/S0047-2 59X(03)00096-4. URL https://linkinghub.elsevier.com/retrieve/pii/S0047259X030009 64.

[20] Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Neural Operator: Graph Kernel Network for Partial Differential Equations, March 2020. URL http://arxiv.org/abs/2003.03485.

[21] Zongyi Li, Nikola Borislavov Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial differential equations. In International Conference on Learning Representations, 2021. URL https://openreview .net/forum?id=c8P9NQVtmnO.

[22] Lu Lu, Pengzhan Jin, Guofei Pang, Zhongqiang Zhang, and George Em Karniadakis. Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators. Nature Machine Intelligence, 3(3):218–229, March 2021. ISSN 2522-5839. doi: 10.1038/s42256-021-00302-5. URL https://www.nature.com/articles/s42256-021-00302-5.

[23] Ziqi Ma, David Pitt, Kamyar Azizzadenesheli, and Anima Anandkumar. Calibrated uncertainty quantification for operator learning via conformal prediction. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. doi: 10.48550/arXiv.2402.01960. URL https://openreview.net/forum?id=cGpe gxy12T.

[24] Amirhossein Mollaali, Gabriel Zufferey, Gonzalo Constante-Flores, Christian Moya, Can Li, Meng Yue, and Guang Lin. Conformalized prediction of post-fault voltage trajectories using pre-trained and finetuned attention-driven neural operators. Neural Networks, 192:107809, 2025. doi: 10.1016/J.NEUNET.2025.10 7809. URL https://doi.org/10.1016/j.neunet.2025.107809.

[25] Christian Moya, Amirhossein Mollaali, Zecheng Zhang, Lu Lu, and Guang Lin. Conformalized-DeepONet: a distribution-free framework for uncertainty quantification in deep operator networks. Physica D: Nonlinear Phenomena, 471:134418, January 2025. ISSN 01672789. doi: 10.1016/j.physd.2024.134418. URL https://linkinghub.elsevier.com/retrieve/pii/S0167278924003683.

[26] Robb J. Muirhead. Aspects of Multivariate Statistical Theory. Wiley Series in Probability and Statistics. Wiley, 1 edition, March 1982. ISBN 978-0-471-09442-5 978-0-470-31655-9. doi: 10.1002/9780470316559. URL https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316559.

[27] Roberto I. Oliveira, Paulo Orenstein, Thiago Ramos, and João Vitor Romano. Split Conformal Prediction and Non-Exchangeable Data. Journal of Machine Learning Research, 25(225):1–38, 2024. URL http: //jmlr.org/papers/v25/23-1553.html.

[28] Aleksandr Podkopaev and Aaditya Ramdas. Distribution-free uncertainty quantification for classification under label shift. In Cassio de Campos and Marloes H. Maathuis, editors, Proceedings ofthe Thirty-Seventh Conference on Uncertainty in Artificial Intelligence, volume 161 of Proceedings of Machine Learning Research, pages 844–853. PMLR, July 2021. URL https://proceedings.mlr.press/v161/podko paev21a.html.

[29] M. Raissi, P. Perdikaris, and G.E. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal ofComputational Physics, 378:686–707, February 2019. ISSN 00219991. doi: 10.1016/j.jcp.2018 .10.045. URL https://linkinghub.elsevier.com/retrieve/pii/S0021999118307125.

[30] Fabricio Dos Santos, Tara Akhound-Sadegh, and Siamak Ravanbakhsh. Physics-informed transformer networks. In The Symbiosis of Deep Learning and Differential Equations III, 2023. URL https: //openreview.net/forum?id=zu80h9YryU.

[31] Kara Scheu, Oliver Fringer, Derek Fong, and Stephen Monismith. Observation data and numerical model files for river plumes in Pallanza Bay of Lake Maggiore, Italy, April 2018. URL https://zenodo.org /record/1206187.

[32] Ryan J Tibshirani, Rina Foygel Barber, Emmanuel Candes, and Aaditya Ramdas. Conformal prediction under covariate shift. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d’Alché Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper\_files/paper/2019/file/8fb21ee7a220752 6da55a679f0332de2-Paper.pdf.

[33] Vladimir Vovk, Alexander Gammerman, and Glenn Shafer. Algorithmic Learning in a Random World. Springer International Publishing, Cham, 2022. ISBN 978-3-031-06648-1 978-3-031-06649-8. URL https://link.springer.com/10.1007/978-3-031-06649-8.

[34] Ziang Wei, Ahmad Osman, Bernd Valeske, and Xavier Maldague. Pulsed thermography dataset for training deep learning models. Applied Sciences, 13(5), 2023. ISSN 2076-3417. doi: 10.3390/app13052901. URL https://www.mdpi.com/2076-3417/13/5/2901.

[35] Victor Wåhlstrand Skärström, Annika Krona, Niklas Lorén, and Magnus Röding. DeepFRAP: fast fluorescence recovery after photobleaching data analysis using deep neural networks. Journal of Microscopy, 282(2):146–161, May 2021. ISSN 0022-2720, 1365-2818. doi: 10.1111/jmi.12989. URL https://onlinelibrary.wiley.com/doi/10.1111/jmi.12989.

[36] Chen Xu and Yao Xie. Conformal prediction interval for dynamic time-series. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pages 11559–11569. PMLR, July 2021. URL https: //proceedings.mlr.press/v139/xu21h.html.

[37] Chen Xu and Yao Xie. Sequential predictive conformal inference for time series. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 38707–38727. PMLR, July 2023. URL https://proceedings.mlr. press/v202/xu23r.html.

[38] Chen Xu, Hanyang Jiang, and Yao Xie. Conformal prediction for multi-dimensional time series by ellipsoidal sets. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 55076–55099. PMLR, July 2024. URL https://proceedings.mlr.press/v235/xu24m.html.

[39] Anthony Zhou and Amir Barati Farimani. Predicting change, not states: An alternate framework for neural PDE surrogates. Computer Methods in Applied Mechanics and Engineering, 441:117990, June 2025. ISSN 00457825. doi: 10.1016/j.cma.2025.117990. URL https://linkinghub.elsevier.com/retr ieve/pii/S0045782525002622.

## A LLM disclaimer

Large language models (LLMs) were used to support the development and execution of this project. They contributed to improving the clarity and readability of the manuscript, suggesting alternative phrasings, helping with the code, offering feedback on mathematical arguments, and providing ideas throughout the research and coding process. All research contributions, results, and final formulations were manually verified.

## B Licenses of external assets

All external assets used in this work are publicly available. The pulsed-thermography dataset of Wei et al. [34] is released under CC-BY 4.0 and hosted on $\mathrm { K a g g l e } . ^ { 6 }$ The Lake Maggiore thermistor data of Scheu et al. [31] is archived on Zenodo under CC-BY 4.0.<sup>7</sup> The FRAP dataset and reference code of Wåhlstrand Skärström et al. [35] are archived on Zenodo under CC-BY 4.0.<sup>8</sup>

## C Example: a Gaussian random field on a function space

A probability distribution on a function space is very similar to a distribution on $\mathbb { R } ^ { p }$ . Where a draw from a multivariate Gaussian $\mathcal { N } ( \boldsymbol { \mu } , \boldsymbol { \Sigma } )$ on $\mathbb { R } ^ { p }$ produces a vector of $\dot { p }$ numbers, a draw from a distribution on $L ^ { 2 } ( \Omega )$ produces an entire function u $: \Omega \to \mathbb { R }$ . Each sample is a function over Ω, and different draws give different functions.

The simplest and most common such distributions are Gaussian randomfields (GRFs), the infinitedimensional analogue of multivariate Gaussians. Like a Gaussian on $\mathbb { R } ^ { \bar { p } }$ , a GRF is determined by two ingredients: a mean function $m \in L ^ { 2 } ( \Omega )$ ) playing the role of $\mu ,$ and a covariance operator C playing the role of Σ. The covariance operator $C : L ^ { \mathsf { \bar { 2 } } } ( \Omega ) \to L ^ { 2 } ( \dot { \Omega } )$ must be positive, symmetric, and trace class [14]. The trace-class condition is what makes the infinite sum of variances finite, ensuring samples actually lie in $L ^ { 2 } ( \Omega )$ . We write u $\sim \mathcal { N } ( m , C )$

The most useful way to picture a sample is via its Karhunen–Loève expansion. If $( \mu _ { n } , e _ { n } ) _ { n \geq 1 }$ are the eigenvalues and eigenfunctions of $C \left( \operatorname { s o } C e _ { n } = \mu _ { n } e _ { n } \right)$ , then a draw $u \sim \mathcal { N } ( \mathbf { 0 } , \dot { C } )$ can be written as

$$
u ( x ) \ = \ \sum _ { n = 1 } ^ { \infty } \sqrt { \mu _ { n } } \xi _ { n } e _ { n } ( x ) , \qquad \xi _ { n } \overset { \mathrm { i i d } } { \sim } \mathcal { N } ( 0 , 1 ) ,\tag{4}
$$

i.e., a random superposition of the basis functions $e _ { n }$ with independent Gaussian coefficients whose variances $\mu _ { n }$ are set by $C$ . The decay rate of $\mu _ { n }$ controls how rough or smooth the samples look: fast decay suppresses high-frequency $e _ { n }$ and gives smooth samples, slow decay gives rough ones.

For Theorem 4.1, we take $\Omega = ( 0 , 1 ) , m = \mathbf { 0 }$ , and $C = ( I - \Lambda ) ^ { - 1 }$ with Λ the Laplace operator under homogeneous Dirichlet boundary conditions. The eigenfunctions $e _ { n } ( x ) = { \sqrt { 2 } } \sin ( n \pi x )$ are sine modes of increasing frequency, and the eigenvalues $\mu _ { n } { \overset { . } { = } } 1 / ( 1 + ( n \pi ) ^ { 2 } )$ decay like $\dot { 1 } / n ^ { 2 } .$ . The trace condition $\textstyle \sum _ { n } \mu _ { n } < \infty$ holds, so $\mathcal { N } ( \mathbf { 0 } , C )$ is a well-defined GRF on $\dot { L } ^ { 2 } ( \Omega )$ . Each sample is a continuous function on (0, 1) that vanishes at the endpoints (because all $e _ { n }$ do), with most of its energy in the low-frequency modes.

## D Validating local exchangeability

To have formal guarantees with the LSCI approach from [15], it is required that the model’s residuals $r _ { t }$ are locally exchangeable. Considering this for the simplest setup of two time points t and $t + \delta$ for some $\delta > 0$ , this means that it must hold that

$$
d _ { \mathrm { T V } } \big ( \binom { r _ { t } } { r _ { t + \delta } } , \binom { r _ { t + \delta } } { r _ { t } } \big ) \big ) \leq d ( t , t + \delta )
$$

for a pre-metric d on the time domain. Note, however, that since we do not have access to the laws of the random vectors above, we cannot reason about their TV distance. Even though in Proposition 4.2

we derive the laws of the solutions $u _ { t } .$ , it is not clear how to reason about the above left-hand side without further assumptions (like independence of the residuals over time, which is not plausible as the solution has a clear time-dependence in all non-stationary PDEs).

## E Additional figures

![](images/30bde63b2abb661b0ffd6a0739d54f53eaa33f6dd14bb9b2dfa4ecbfe944fbcf.jpg)

![](images/f7dea65061dd28f4c61ddcf2f40ac0220b7a52e61e14e5ca014545d60cb0f53a.jpg)

![](images/1cb973361976bd5f361f8dd28fe9538a80b824b566e87567d74f21424c7ce4c5.jpg)  
Figure E.1: CP bands of the backward heat equation calibrated at $t = 0$ . Each panel shows one time point of the solution $u ( x , t )$ and CP bands over the spatial domain x. While at time $t = \delta$ , the solution is properly covered, coverage is lost at later time steps.

## F Weighted conformal prediction formula

For completeness, we recall the explicit form of the weighted split conformal predictor of Tibshirani et al. [32] that underlies our WCP Gauss method (Proposition 4.2).

Given a calibration set $\mathcal { D } _ { \mathrm { c a l } } = \{ ( \boldsymbol { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n } , \mathrm { a }$ fixed score function $S ,$ and unnormalized weights $w _ { i }$ on $\begin{array} { r } { \mathcal { D } _ { \mathrm { c a l } } \cup \{ \pmb { x } _ { \mathrm { t e s t } } \} \left( \mathbf { e } . \mathbf { g } . \right. } \end{array}$ ., Equation (1) or, in our case, Equation (7)), define the normalized weights

$$
\tilde { w } _ { i } : = \frac { w _ { i } } { \sum _ { j = 1 } ^ { n } w _ { j } + w _ { \mathrm { t e s t } } } , \ i = 1 , \dots , n , \qquad \tilde { w } _ { n + 1 } : = \frac { w _ { \mathrm { t e s t } } } { \sum _ { j = 1 } ^ { n } w _ { j } + w _ { \mathrm { t e s t } } } ,\tag{5}
$$

where $w _ { \mathrm { t e s t } }$ is the weight evaluated at $\scriptstyle { \mathbf { x } } _ { \mathrm { t e s t } }$ . Writing $S _ { i } : = S ( \pmb { x } _ { i } , y _ { i } )$ for the calibration scores, the weighted split conformal prediction set at level $1 - \alpha$ is

$$
\mathcal { C } ^ { \mathrm { W C P } } ( \boldsymbol { x } _ { \mathrm { t e s t } } ) : = \left\{ y \in \mathbb { R } \ : \ S ( \boldsymbol { x } _ { \mathrm { t e s t } } , y ) \leq Q _ { 1 - \alpha } \Big ( \sum _ { i = 1 } ^ { n } \tilde { w } _ { i } \delta _ { S _ { i } } + \tilde { w } _ { n + 1 } \delta _ { + \infty } \Big ) \right\} ,\tag{6}
$$

where $\delta _ { a }$ denotes a point mass at a and $Q _ { 1 - \alpha } ( \mu )$ the $( 1 - \alpha )$ )-quantile of the discrete measure $\mu .$ The atom $\mathrm { a t } + \infty$ forces ${ \mathcal { C } } ^ { \mathrm { W C P } } = \mathbb { R }$ whenever $\tilde { w } _ { n + 1 } > \alpha , \mathrm { i . e . }$ , whenever the test point is so unlike the calibration set that no finite quantile carries enough mass. This corresponds to the infinite bands reported as $n _ { \infty }$ in our experiments. With weights set to the true density ratio between test and calibration distributions, the predictor satisfies $\mathbb { P } \big ( y _ { \mathrm { t e s t } } \in \mathcal { C } ^ { \mathrm { W C P } } ( \pmb { x } _ { \mathrm { t e s t } } ) \big ) \geq \bar { 1 } - \alpha \left[ 3 2 \right]$ . Standard split CP (Section 3.1) is recovered by setting $w _ { i } \equiv 1$ , which collapses all normalized weights to $1 / ( n + 1 )$ ).

Gaussian WCP algorithm Algorithm 1 below instantiates the WCP set in Equation (6) using Gaussian density estimates at calibration and prediction times, with covariance estimated by the Ledoit–Wolf shrinkage estimator [19] for stability in the high-dimensional regime. We list the algorithm with different sets for calibration and test. In practice, it should be common to use the same trajectories at different time points for calibration and testing. We show both in our real-world tasks, where the pulsed-thermography and FRAP datasets use the same trajectories, but Lake Maggiore splits the data (at least it splits them into early-spring and late-spring).

Algorithm 1: WCP for time-dependent PDEs.   
Input: calibration pairs $\{ ( \pmb { u } _ { i } ^ { t _ { \mathrm { c a l } } } , \pmb { u } _ { i } ^ { t _ { \mathrm { c a l } } + \delta } ) \} _ { i = 1 } ^ { n }$ , test-time inputs $\{ \tilde { u } _ { j } ^ { t } \} _ { j = 1 } ^ { m + 1 }$ at time $t \geq t _ { \mathrm { c a l } }$ (with   
$\tilde { \boldsymbol { u } } _ { m + 1 } ^ { t }$ the query), nonconformity score S, miscoverage level α.   
Output: prediction set $\mathcal { C } _ { t + \delta } ^ { \mathrm { W C P } } ( \tilde { \boldsymbol { u } } _ { m + 1 } ^ { t } )$   
$\mathcal { D } _ { t _ { \mathrm { c a l } } }  \{ \boldsymbol { u } _ { i } ^ { t _ { \mathrm { c a l } } } \} _ { i = 1 } ^ { n } , \ \mathcal { D } _ { t }  \{ \boldsymbol { \tilde { u } } _ { j } ^ { t } \} _ { j = 1 } ^ { m }$   
for $\tau \in \{ t _ { \mathrm { c a l } } , t \}$ do   
$\begin{array} { r } { \hat { \pmb { \mu } } _ { \tau }  \frac { 1 } { | \mathcal { D } _ { \tau } | } \sum _ { \pmb { u } \in \mathcal { D } _ { \tau } } \pmb { u } } \end{array}$ // sample mean   
$\hat { \Sigma } _ { \tau }  \mathrm { L e d o i t W o l f } ( \mathcal { D } _ { \tau } )$ // Ledoit–Wolf shrinkage   
for $i = 1 , \ldots , n$ do   
$w _ { i } \gets \varphi ( \pmb { u } _ { i } ^ { t _ { \mathrm { c a l } } } ; \hat { \pmb { \mu } } _ { t } , \hat { \Sigma } _ { t } ) / \varphi ( \pmb { u } _ { i } ^ { t _ { \mathrm { c a l } } } ; \hat { \pmb { \mu } } _ { t _ { \mathrm { c a l } } } , \hat { \Sigma } _ { t _ { \mathrm { c a l } } } )$ // importance weight   
$S _ { i } \gets S ( \boldsymbol { u } _ { i } ^ { t _ { \mathrm { c a l } } } , \boldsymbol { u } _ { i } ^ { t _ { \mathrm { c a l } } + \delta } )$ // calibration score   
$w _ { m + 1 } \gets \varphi ( \tilde { \pmb { u } } _ { m + 1 } ^ { t } ; \hat { \pmb { \mu } } _ { t } , \hat { \pmb { \Sigma } } _ { t } ) / \varphi ( \tilde { \pmb { u } } _ { m + 1 } ^ { t } ; \hat { \pmb { \mu } } _ { t _ { \mathrm { c a l } } } , \hat { \pmb { \Sigma } } _ { t _ { \mathrm { c a l } } } )$ // test weight   
$\tilde { w } _ { i } \gets w _ { i } / ( \sum _ { j } w _ { j } + w _ { m + 1 } ) , ~ \tilde { w } _ { n + 1 } \gets w _ { m + 1 } / ( \sum _ { j } w _ { j } + w _ { m + 1 } )$ // normalize, cf.   
Equation (5)   
$\begin{array} { r } { \hat { q } _ { 1 - \alpha }  Q _ { 1 - \alpha } \big ( \sum _ { i = 1 } ^ { n } \tilde { w } _ { i } \delta _ { S _ { i } } + \tilde { w } _ { n + 1 } \delta _ { + \infty } \big ) } \end{array}$   
return $\mathcal { C } _ { t + \delta } ^ { \mathrm { W C P } } ( \tilde { \pmb { u } } _ { m + 1 } ^ { t } ) = \{ \pmb { y } : S ( \tilde { \pmb { u } } _ { m + 1 } ^ { t } , \pmb { y } ) \leq \hat { q } _ { 1 - \alpha } \}$

## G Proofs

## G.1 Proof of Theorem 4.1

Theorem 4.1. Consider the one-dimensional heat equation $\partial _ { t } u = \partial _ { x x } \ i$ on the domain $\Omega = ( 0 , 1 )$ with Dirichlet boundary conditions, $t \geq 0 .$ . Suppose the initial condition $u _ { 0 } ( x )$ is sampledfrom a Gaussian randomfield $\mathcal { \dot { P } } _ { 0 } = \mathcal { N } ( \mathbf { 0 } , ( I - \Lambda ) ^ { - 1 } )$ , where Λ is the Laplace operator on Ω. Then, the TV distance between the solution distributions at any two time points is maximal:

$$
d _ { \mathrm { T V } } ( \mathcal { P } _ { t } , \mathcal { P } _ { t + \delta } ) = 1 , \quad \forall t \ge 0 , \ : \delta > 0 .
$$

Proof. Notice that our whole setup is on a Hilbert space. We begin by showing that $( I - \Lambda ) ^ { - 1 }$ is a well-defined covariance operator on $L ^ { 2 } ( \Omega )$ . For this, according to [14, Proposition 4.17], the operator must be positive, symmetric and trace class. As both I and Λ are positive and symmetric, so is their sum. $\mathbf { A } \mathbf { s } - \pmb { \Lambda }$ is positive, it follows that $( I - \Lambda )$ is invertible, due to strictly positive eigenvalues. Further, the eigenvalues of Λ with Dirichlet boundary conditions are given by $\lambda _ { n } = ( { \overset { \smile } { n \pi } } ) ^ { 2 }$ , with corresponding eigenfunctions $e _ { n } ( x ) = { \sqrt { 2 } }$ sin(nπx), which form an orthonormal basis of $L ^ { 2 } ( \Omega )$ . Therefore, the eigenvalues of $( I - \dot { \mathbf { A } } ) ^ { - 1 }$ are given by $\mu _ { n } = 1 / ( 1 + ( n \pi ) ^ { 2 } )$ and as

$$
\sum _ { n = 1 } ^ { \infty } \mu _ { n } < \infty ,
$$

we conclude that $( I - \Lambda ) ^ { - 1 }$ is a trace class operator and thus defines a Gaussian measure on $L ^ { 2 } ( \Omega )$

Now our proof will be based on the Feldman–Hájek theorem [6, Theorem 2.23], which gives a characterization of when two Gaussian measures on a Hilbert space are either equivalent or mutually singular. We will briefly state the whole chain of reasoning, and then provide the necessary details.

We will show that our measure at all times is Gaussian. By the Feldman–Hájek theorem, two Gaussian measures $\mathcal { N } ( m _ { 1 } , C _ { 1 } )$ and $\mathcal { N } ( m _ { 2 } , C _ { 2 } )$ on a Hilbert space are either equivalent or mutually singular. A necessary condition for equivalence is that the Cameron–Martin spaces, as given by $C ^ { 1 / 2 }$ , of the two measures are equal as sets [6, Theorem 2.23]. Thus, if the ranges of the covariance operators $C _ { 1 } ^ { 1 / 2 }$ and $C _ { 2 } ^ { 1 / 2 }$ are not equal, then the measures are mutually singular and their TV distance is 1.

Calculating the covariance operators Starting with a measure $\mu _ { 0 }$ of the initial distribution, the heat equation induces a semigroup $S ( t ) = \exp ( t \mathbf { \Lambda } )$ , which maps the initial condition $u _ { 0 }$ to the solution at time t, i.e., $\left. u ( \cdot , t ) \right| ^ { - } = \bar { S } ( t ) \dot { u } _ { 0 }$ . Therefore, the measure $\mu _ { t }$ of $u ( \cdot , t )$ is induced by the pushforward of $\mu _ { 0 }$ under $S ( t ) , { \mathrm { i . e . , } } \mu _ { t } = S ( t ) _ { \# } \mu _ { 0 } . { \mathrm { ~ A s ~ } } S ( t )$ is linear, $\mu _ { t }$ is also a Gaussian measure with mean 0 and covariance operator

$$
C _ { t } = S ( t ) ( I - \Lambda ) ^ { - 1 } S ( t ) ^ { * } = \exp ( t \Lambda ) ( I - \Lambda ) ^ { - 1 } \exp ( t \Lambda ) ,
$$

where $S ( t ) ^ { * }$ denotes the adjoint of $S ( t )$ [14, Chap. 4.3]. As we have seen above, the eigenvalues of $( \pmb { I } - \pmb { \Lambda } ) ^ { - 1 }$ are given by $\mu _ { n } = 1 / ( 1 + ( n \pi ) ^ { 2 } )$ , with corresponding eigenfunctions $e _ { n } ( x )$ . Further, the eigenfunctions of Λ are also given by $e _ { n } ( x )$ , with corresponding eigenvalues $\lambda _ { n } = \dot { - } ( n \pi ) ^ { 2 }$ . Lastly, by functional calculus, the eigenfunctions of $\exp ( t \Lambda )$ are also given by $e _ { n } ( x )$ with corresponding eigenvalues $\nu _ { n } = \exp ( - t ( n \overline { { \pi } } ) ^ { 2 } )$ ). With this, we can compute

$$
\begin{array} { l } { { C _ { t } e _ { n } = \displaystyle \exp ( t \mathbf { A } ) ( I - \mathbf { A } ) ^ { - 1 } \exp ( t \mathbf { A } ) e _ { n } } } \\ { { \displaystyle \qquad = ( \exp ( - t ( n \pi ) ^ { 2 } ) ) \left( \frac { 1 } { 1 + ( n \pi ) ^ { 2 } } \right) ( \exp ( - t ( n \pi ) ^ { 2 } ) ) e _ { n } = \frac { \exp ( - 2 t ( n \pi ) ^ { 2 } ) } { 1 + ( n \pi ) ^ { 2 } } e _ { n } . } } \end{array}
$$

Thus, the eigenvalues of $C _ { t }$ are given by $\lambda _ { n } ( t ) = \nu _ { n } ^ { 2 } \mu _ { n } = \exp ( - 2 t ( n \pi ) ^ { 2 } ) / ( 1 + ( n \pi ) ^ { 2 } )$ , with corresponding eigenfunctions $e _ { n } ( x )$ .

Calculating the Cameron–Martin spaces The functions $e _ { n } ( x )$ form an orthonormal basis of $L ^ { 2 } ( \Omega )$ , so we can express every element $f \in L ^ { 2 } ( \Omega )$ as

$$
f = \sum _ { n = 1 } ^ { \infty } c _ { n } e _ { n } , \quad \sum _ { n = 1 } ^ { \infty } c _ { n } ^ { 2 } < \infty .
$$

The Cameron–Martin space $H _ { t }$ of $\mathcal { P } _ { t }$ is given by the range of $\boldsymbol { C } _ { t } ^ { 1 / 2 }$ , which is given by

$$
\mathrm { R a n } ( C _ { t } ^ { 1 / 2 } ) = \left\{ C _ { t } ^ { 1 / 2 } f \mid f \in L ^ { 2 } ( \Omega ) \right\} = \left\{ \sum _ { n = 1 } ^ { \infty } \sqrt { \lambda _ { n } ( t ) } c _ { n } e _ { n } \mid f \in L ^ { 2 } ( \Omega ) \right\} .
$$

Therefore, $g \in H _ { t }$ if and only if g can be expressed as

$$
g = \sum _ { n = 1 } ^ { \infty } d _ { n } e _ { n } , \quad \sum _ { n = 1 } ^ { \infty } \frac { d _ { n } ^ { 2 } } { \lambda _ { n } ( t ) } < \infty .
$$

Inserting the expression for $\lambda _ { n } ( t )$ , we see that $g \in H _ { t }$ if and only if

$$
\sum _ { n = 1 } ^ { \infty } d _ { n } ^ { 2 } \frac { 1 + ( n \pi ) ^ { 2 } } { \exp ( - 2 t ( n \pi ) ^ { 2 } ) } < \infty .
$$

Showing that the Cameron–Martin spaces are not equal Now it is easy to see that for any $t \geq 0$ $\delta > 0$ , the Cameron–Martin spaces $H _ { t }$ and $H _ { t + \delta }$ are not equal. For example, the function

$$
h ( x ) = \sum _ { n = 1 } ^ { \infty } \exp ( - ( t + \delta ) ( n \pi ) ^ { 2 } ) e _ { n } ( x )
$$

is an element of $H _ { t }$ , as

$$
\sum _ { n = 1 } ^ { \infty } ( \exp ( - ( t + \delta ) ( n \pi ) ^ { 2 } ) ) ^ { 2 } \frac { 1 + ( n \pi ) ^ { 2 } } { \exp ( - 2 t ( n \pi ) ^ { 2 } ) } = \sum _ { n = 1 } ^ { \infty } ( 1 + ( n \pi ) ^ { 2 } ) \exp ( - 2 \delta ( n \pi ) ^ { 2 } ) < \infty ,
$$

but it is not an element of $H _ { t + \delta }$ , as

$$
\sum _ { n = 1 } ^ { \infty } ( \exp ( - ( t + \delta ) ( n \pi ) ^ { 2 } ) ) ^ { 2 } \frac { 1 + ( n \pi ) ^ { 2 } } { \exp ( - 2 ( t + \delta ) ( n \pi ) ^ { 2 } ) } = \sum _ { n = 1 } ^ { \infty } ( 1 + ( n \pi ) ^ { 2 } ) = \infty .
$$

Therefore, by the Feldman–Hájek theorem, the measures $\mathcal { P } _ { t }$ and $\mathcal { P } _ { t + \delta }$ are mutually singular, and their TV distance is 1. □

## G.2 Proof of Proposition 4.2

Proposition 4.2. Consider a linear evolution equation on a bounded domain with linear boundary conditions and a deterministic source term. If the calibration dataset at time $t _ { \mathrm { c a l } }$ and a test sample $u _ { \mathrm { t e s t } } ^ { t }$ at time $t \geq t _ { \mathrm { c a l } }$ are generated from the same affine numerical scheme on independent initial conditions sampled from a Gaussian distribution, then we can construct a likelihood-weighted conformal predictor such that

$$
\mathbb { P } \big ( \boldsymbol { u } _ { \mathrm { t e s t } } ^ { t + \delta } \in \mathcal { C } _ { t + \delta } ^ { \mathrm { W C P } } ( \boldsymbol { u } _ { \mathrm { t e s t } } ^ { t } ) \big ) \geq 1 - \alpha .
$$

Proof. By assumption, the numerical time–δ map is affine: there exist a matrix $\ b { A } \in \mathbb { R } ^ { p \times p }$ and vector $\pmb { b } \in \mathbb { R } ^ { p }$ such that

$$
\begin{array} { r } { \pmb { u } ^ { t + \delta } = \pmb { A } \pmb { u } ^ { t } + \pmb { b } . } \end{array}
$$

Iterating yields for $k \in \mathbb N$

$$
\boldsymbol { u } ^ { t _ { \mathrm { c a l } } + k \delta } = \boldsymbol { A } ^ { k } \boldsymbol { u } ^ { t _ { \mathrm { c a l } } } + \sum _ { j = 0 } ^ { k - 1 } \boldsymbol { A } ^ { j } \boldsymbol { b } ,
$$

and in particular each ${ \mathbf { } } u ^ { t }$ is an affine function of $\mathbf { \Delta } _ { \pmb { u } } ^ { 0 }$

Since ${ \pmb u } ^ { 0 } \sim \mathcal { N } ( { \pmb \mu } _ { 0 } , { \pmb \Sigma } _ { 0 } )$ and the multivariate normal family is closed under affine transformations, we have for every $t \geq 0$

$$
\pmb { u } ^ { t } \sim \mathcal { N } ( \pmb { \mu } _ { t } , \pmb { \Sigma } _ { t } )
$$

for some mean $\pmb { \mu } _ { t }$ and covariance $\Sigma _ { t } \ ( \mathrm { e . g . } , \mu _ { t } \ = \ M _ { t } \mu _ { 0 } + c _ { t }$ and $\Sigma _ { t } \ : = \ : M _ { t } \Sigma _ { 0 } M _ { t } ^ { \top }$ for the corresponding affine representation $\pmb { u } ^ { t } = \pmb { M } _ { t } \pmb { u } ^ { 0 } + \pmb { c } _ { t } )$ . Hence the calibration and test laws admit Lebesgue densities $p _ { t _ { \mathrm { c a l } } }$ and $p _ { t }$ on $\mathbb { R } ^ { p }$ and the likelihood ratio

$$
r ( \pmb { x } ) : = \frac { p _ { t } ( \pmb { x } ) } { p _ { t _ { \mathrm { c a l } } } ( \pmb { x } ) }\tag{7}
$$

is available in closed form. In particular, weighting by Equation (7) corresponds to importance weights proportional to $p _ { t } / p _ { t _ { \mathrm { c a l } } }$

Now apply likelihood-weighted split conformal prediction (weighted CP) with these weights to any fixed score function. The standard validity theorem for weighted split CP (under covariate shift with known likelihood ratio) implies that the resulting predictor $\mathcal { C } _ { t + \delta } ^ { \mathrm { W C P } }$ satisfies

$$
\mathbb { P } \big ( \boldsymbol { u } ^ { t + \delta } \in \mathcal { C } _ { t + \delta } ^ { \mathrm { W C P } } \big ) ~ \ge ~ 1 - \alpha
$$

for each target time index $t \geq t _ { \mathrm { c a l } }$ . This proves the claim.

## G.3 Proof of Proposition 4.6

Proposition 4.6 (Robust WCP under Gaussian distributions). Under the assumptions ofProposition 4.2, suppose we additionally observe independent estimation samples $\{ u _ { j } ^ { t _ { \mathrm { c a l } } } \} _ { j = 1 } ^ { n _ { 0 } }$ and $\{ \pmb { u } _ { j } ^ { t } \} _ { j = 1 } ^ { m _ { 1 } }$ 1 with $m _ { k } > p f o r k \in \{ 0 , 1 \}$ , and fix an arbitrary estimation uncertainty $\eta > 0$ . Then, we can construct a conformal prediction set $C _ { t + \delta } ^ { \mathrm { r o b } }$ with coverage guarantee

$$
\mathbb { P } \big ( \boldsymbol { u } _ { \mathrm { t e s t } } ^ { t + \delta } \in C _ { t + \delta } ^ { \mathrm { r o b } } ( \boldsymbol { u } _ { \mathrm { t e s t } } ^ { t } ) \big ) \ \geq \ 1 - \eta - \alpha .
$$

Smaller η will usually correspond to wider bands.

Proof. As this proof is purely statistical, we will use random variables $X _ { i }$ instead of PDE-solutions $u _ { i } ^ { t }$ to ease the notation with more familiar symbols. In particular, we will use $Z _ { i } ^ { ( 0 ) }$ and $Z _ { i } ^ { ( 1 ) }$ for the estimation samples of the calibration and test distribution, respectively.

The proof extends WCP [32] to use worst case weights over a confidence region. Specifically, we know that we can construct a valid conformal set with the true parameters $\bar { \theta ^ { \star } } = ( \bar { \mu _ { 0 } } , \Sigma _ { 0 } , \mu _ { 1 } , \Sigma _ { 1 } )$ using WCP. Now we construct a $( 1 - \eta )$ confidence region Θ for the unknown true parameters $\theta ^ { \star }$ using the covariate-only samples. The robust construction then takes a worst-case weighted conformal threshold (meaning the threshold that yields the largest conformal set over Θ) computed using the WCP weights

$$
p _ { i , \theta } ( x ) = \frac { w _ { \theta } ( X _ { i } ) } { \sum _ { j = 1 } ^ { n } w _ { \theta } ( X _ { j } ) + w _ { \theta } ( x ) } , i = 1 , \dots , n , \qquad p _ { n + 1 , \theta } ( x ) = \frac { w _ { \theta } ( x ) } { \sum _ { j = 1 } ^ { n } w _ { \theta } ( X _ { j } ) + w _ { \theta } ( x ) } ,
$$

where $w _ { \theta } = d P _ { 1 } ^ { \theta } / d P _ { 0 } ^ { \theta }$ is the covariate likelihood ratio of normal distributions with parameters $\theta , X _ { i }$ are calibration samples and x is the test sample. On the event $\{ \theta ^ { \star } \in \Theta \}$ , the robust set contains the oracle $\mathbf { W C P }$ set. The stated coverage then follows by a union bound.

Throughout, $\varphi ( \cdot ; \pmb { \mu } , \pmb { \Sigma } )$ denotes the p-variate Gaussian density with mean $\pmb { \mu }$ and covariance $\pmb { \Sigma } \succ 0$

Getting the (1 − η) confidence region for $\theta ^ { \star }$ From the covariate-only samples $\{ Z _ { j } ^ { ( k ) } \} _ { j = 1 } ^ { m _ { k } }$ for $k \in \{ 0 , 1 \}$ , define the sample mean and sample covariance

$$
\widehat { \pmb { \mu } } _ { k } : = \frac { 1 } { m _ { k } } \sum _ { j = 1 } ^ { m _ { k } } Z _ { j } ^ { ( k ) } , \qquad \widehat { \pmb { \Sigma } } _ { k } : = \frac { 1 } { m _ { k } - 1 } \sum _ { j = 1 } ^ { m _ { k } } ( Z _ { j } ^ { ( k ) } - \widehat { \pmb { \mu } } _ { k } ) ( Z _ { j } ^ { ( k ) } - \widehat { \pmb { \mu } } _ { k } ) ^ { \top } .
$$

Since $m _ { k } > p$ and the data are Gaussian, $\widehat { \Sigma } _ { k }$ is positive definite almost surely [26, Thm. 3.1.4].

Fix numbers $\eta _ { 1 } , \eta _ { 2 } , \eta _ { 3 } , \eta _ { 4 } > 0$ with $\eta _ { 1 } + \eta _ { 2 } + \eta _ { 3 } + \eta _ { 4 } \leq \eta .$

Mean confidence sets. By Härdle and Simar [17, Cor. 5.3], for each $k \in \{ 0 , 1 \}$ , the Hotelling $T ^ { 2 }$ statistic

$$
T _ { k } ^ { 2 } : = m _ { k } \big ( \widehat { \pmb { \mu } } _ { k } - { \pmb \mu } _ { k } \big ) ^ { \top } \widehat { \pmb { \Sigma } } _ { k } ^ { - 1 } \big ( \widehat { \pmb { \mu } } _ { k } - { \pmb \mu } _ { k } \big )
$$

satisfies

$$
\frac { m _ { k } - p } { p ( m _ { k } - 1 ) } T _ { k } ^ { 2 } \sim F _ { p , m _ { k } - p } .
$$

Let $c _ { k } ^ { ( \mu ) }$ be the $\left( 1 - \eta _ { 3 + k } \right)$ quantile of $\frac { p ( m _ { k } - 1 ) } { m _ { k } - p } F _ { p , m _ { k } - p }$ and define

$$
\mathcal { M } _ { k } : = \left\{ \pmb { \mu } \in \mathbb { R } ^ { p } : m _ { k } ( \widehat { \pmb { \mu } } _ { k } - \pmb { \mu } ) ^ { \top } \widehat { \pmb { \Sigma } } _ { k } ^ { - 1 } ( \widehat { \pmb { \mu } } _ { k } - \pmb { \mu } ) \leq c _ { k } ^ { ( \mu ) } \right\} .
$$

Then $\mathcal { M } _ { k }$ is compact and $\mathbb { P } ( \mu _ { k } \in \mathcal { M } _ { k } ) \ge 1 - \eta _ { 3 + k }$ . Notice that $\mathcal { M } _ { k }$ is an ellipsoid around the sample mean.

Covariance confidence sets. For the sample covariance matrix, we know $W _ { k } = ( m _ { k } - 1 ) \widehat { \Sigma } _ { k } \sim$ Wishar $\bar { \mathbf { \zeta } } _ { p } ( \mathbf { \Sigma } \Sigma _ { k } , m _ { k } - 1 )$ [17, Thm. 5.7 (c)]. The eigenvalues of $\boldsymbol { \Sigma } _ { k } ^ { - 1 / 2 } \boldsymbol { \widehat { \Sigma } } _ { k } \boldsymbol { \Sigma } _ { k } ^ { - 1 / 2 }$ have a distribution independent of $\Sigma _ { k } \ [ 1 7$ , Thm. 5.5]. Choose constants $0 < a _ { k } < b _ { k } ^ { " } <$ such that

$$
\mathbb { P } \Big ( a _ { k } I \preceq \Sigma _ { k } ^ { - 1 / 2 } \widehat { \Sigma } _ { k } \Sigma _ { k } ^ { - 1 / 2 } \preceq b _ { k } I \Big ) \geq 1 - \eta _ { k + 1 } .
$$

This is well-defined, as the eigenvalues of $\boldsymbol { \Sigma } _ { k } ^ { - 1 / 2 } \widehat { \boldsymbol { \Sigma } } _ { k } \boldsymbol { \Sigma } _ { k } ^ { - 1 / 2 }$ are almost surely strictly positive and finite. Hence $L _ { k } : = \lambda _ { \operatorname* { m i n } } ( \Sigma _ { k } ^ { - 1 / 2 } \widehat \Sigma _ { k } \Sigma _ { k } ^ { - 1 / 2 } )$ and $U _ { k } : = \lambda _ { \operatorname* { m a x } } \bigl ( \Sigma _ { k } ^ { - 1 / 2 } \widehat \Sigma _ { k } \Sigma _ { k } ^ { - 1 / 2 } \bigr )$ are well-defined random variables with support on a subset of $( 0 , \infty )$ .   
Define

$$
\mathcal { S } _ { k } : = \left\{ \Sigma \in \mathbb { S } _ { + + } ^ { p } : \frac { 1 } { b _ { k } } \widehat { \Sigma } _ { k } \preceq \Sigma \preceq \frac { 1 } { a _ { k } } \widehat { \Sigma } _ { k } \right\} .
$$

Each set $\scriptstyle { S _ { k } }$ is compact. Indeed, $\boldsymbol { S _ { k } }$ is bounded since, for any $\pmb { \Sigma } \in S _ { k }$ , its spectral norm satisfies

$$
\| \Sigma \| _ { 2 } = \lambda _ { \operatorname* { m a x } } ( \Sigma ) ,
$$

which is bounded above by the largest eigenvalue allowed by the defining inequality of $S _ { k }$

Closedness follows as well. Let $( \Sigma _ { n } ) _ { n \in \mathbb { N } } \subset S _ { k }$ be a convergent sequence with limit $\Sigma _ { \infty }$ . By definition of $\boldsymbol { S _ { k } }$ , for every n and every vector $z \in \mathbb { R } ^ { p }$

$$
z ^ { \top } \left( { \frac { 1 } { a _ { k } } } { \widehat { \Sigma } } _ { k } - \Sigma _ { n } \right) z \geq 0 .
$$

By continuity of the quadratic form, passing to the limit yields

$$
z ^ { \top } \left( { \frac { 1 } { a _ { k } } } { \widehat { \Sigma } } _ { k } - \Sigma _ { \infty } \right) z \geq 0 \quad { \mathrm { f o r ~ a l l ~ } } z \in \mathbb { R } ^ { p } .
$$

An analogous argument applies to the lower bound in the definition of $\scriptstyle { S _ { k } }$ . Hence $\pmb { \Sigma } _ { \infty } \in S _ { k }$ , and $\scriptstyle { S _ { k } }$ is closed.

Since $\scriptstyle { S _ { k } }$ is both closed and bounded in the space of symmetric matrices, it is compact. Moreover, under the stated assumptions, the matrices in $\scriptstyle { S _ { k } }$ are almost surely bounded away from singularity.

Define the joint confidence region

$$
\begin{array} { r } { \Theta : = \mathcal { M } _ { 0 } \times \mathcal { S } _ { 0 } \times \mathcal { M } _ { 1 } \times \mathcal { S } _ { 1 } . } \end{array}
$$

By a union bound,

$$
\mathbb { P } ( \theta ^ { \star } \in \Theta ) \geq 1 - \eta .\tag{8}
$$

Let $E : = \{ \theta ^ { \star } \in \Theta \}$

Defining the weights and quantiles For $\boldsymbol { \theta } = ( \mu _ { 0 } , \boldsymbol { \Sigma } _ { 0 } , \mu _ { 1 } , \boldsymbol { \Sigma } _ { 1 } ) \in \Theta$ , define the Gaussian likelihood ratio

$$
w _ { \boldsymbol { \theta } } ( x ) : = \frac { \varphi ( x ; \pmb { \mu } _ { 1 } , \pmb { \Sigma } _ { 1 } ) } { \varphi ( x ; \pmb { \mu } _ { 0 } , \pmb { \Sigma } _ { 0 } ) } .
$$

For a test covariate value $x \in \mathbb { R } ^ { p }$ , define the associated WCP weights

$$
p _ { i , \theta } ( x ) : = \frac { w _ { \theta } ( X _ { i } ) } { \sum _ { j = 1 } ^ { n } w _ { \theta } ( X _ { j } ) + w _ { \theta } ( x ) } , \quad i = 1 , \dots , n , \qquad p _ { n + 1 , \theta } ( x ) : = \frac { w _ { \theta } ( x ) } { \sum _ { j = 1 } ^ { n } w _ { \theta } ( X _ { j } ) + w _ { \theta } ( x ) } .
$$

Then $p _ { i , \theta } ( x ) \geq 0$ and $\begin{array} { r } { \sum _ { i = 1 } ^ { n + 1 } p _ { i , \theta } ( x ) = 1 } \end{array}$

Let $\theta ^ { \star }$ denote the true parameters and $w ^ { \star } : = w _ { \theta ^ { \star } }$ . The oracle WCP set is the WCP set obtained by using $w ^ { \star }$ in the weights above, and we denote it by ${ \widehat { C } } _ { \theta ^ { \star } }$ below.

Fix $x \in \mathbb { R } ^ { p }$ and a candidate label $y \in \mathbb R$ . Let $S _ { i } , i = 1 , \ldots , n$ , be the nonconformity scores for the sample $( X _ { i } , Y _ { i } )$

For each $\theta \in \Theta$ and test point $( x , y )$ with score $S _ { n + 1 }$ , define the discrete probability measure on $[ 0 , \infty ]$

$$
\widehat { \nu } _ { \theta } ^ { ( x , y ) } : = \sum _ { i = 1 } ^ { n } p _ { i , \theta } ( x ) \delta _ { S _ { i } } + p _ { n + 1 , \theta } ( x ) \delta _ { \infty } ,\tag{9}
$$

and define the corresponding CDF on R by

$$
F _ { \theta } ( t ; x , y ) : = \widehat { \nu } _ { \theta } ^ { ( x , y ) } \big ( ( - \infty , t ] \big ) = \sum _ { i = 1 } ^ { n } p _ { i , \theta } ( x ) \mathbf { 1 } \{ S _ { i } \leq t \} , \qquad t \in \mathbb { R } ,
$$

since the atom at contributes no mass to $( - \infty , t ]$ for finite t.

Define the (1 α) weighted quantile as

$$
q _ { \theta } ( x , y ) : = \operatorname* { i n f } \Big \{ t \in \mathbb { R } : F _ { \theta } ( t ; x , y ) \geq 1 - \alpha \Big \} .\tag{10}
$$

Equivalently, $q _ { \theta } ( x , y )$ is the quantile of level $1 - \alpha$ under the discrete law $\widehat { \nu } _ { \theta } ^ { ( x , y ) }$ in Equation (9).

Robust threshold and robust prediction set Define the robust threshold as the worst-case quantile over $\Theta ;$

$$
q _ { \mathrm { r o b } } ( x , y ) : = \operatorname* { s u p } _ { \theta \in \Theta } q _ { \theta } ( x , y ) .\tag{11}
$$

Define the robust WCP prediction set by

$$
\widehat { C } _ { \mathrm { r o b } } ( x ) : = \Big \{ y \in \mathbb { R } : S _ { n + 1 } \leq q _ { \mathrm { r o b } } ( x , y ) \Big \} .\tag{12}
$$

Also, define for any fixed $\theta \in \Theta$ , the WCP set

$$
\widehat { C } _ { \theta } ( x ) : = \Big \{ y \in \mathbb { R } : S _ { n + 1 } \leq q _ { \theta } ( x , y ) \Big \} .
$$

In particular, ${ \widehat { C } } _ { \theta } ,$ ⋆ is the oracle WCP set.

On the event $E = \{ \theta ^ { \star } \in \Theta \}$ , we have for all $( x , y )$

$$
q _ { \mathrm { r o b } } ( x , y ) = \operatorname* { s u p } _ { \theta \in \Theta } q _ { \theta } ( x , y ) \ \geq \ q _ { \theta ^ { \star } } ( x , y ) ,
$$

and therefore, for all $x ,$

$$
\widehat { C } _ { \mathrm { r o b } } ( x ) \supseteq \widehat { C } _ { \theta ^ { \star } } ( x ) \qquad \mathrm { o n } E .\tag{13}
$$

Robust coverage Let $F : = \{ Y _ { n + 1 } \notin \widehat { C } _ { \mathrm { r o b } } ( X _ { n + 1 } ) \}$ . By Equation (13),

$$
F \cap E \subseteq \{ Y _ { n + 1 } \not \in { \widehat { C } } _ { \theta ^ { \star } } ( X _ { n + 1 } ) \} \cap E .
$$

Hence,

$$
\begin{array} { r } { \mathbb { P } ( F ) \leq \mathbb { P } \Big ( Y _ { n + 1 } \not \in \widehat { C } _ { \theta ^ { \star } } ( X _ { n + 1 } ) \Big ) + \mathbb { P } ( E ^ { c } ) . } \end{array}
$$

By the oracle WCP guarantee (i.e., the covariate-shift WCP coverage result applied with the true likelihood ratio $w ^ { \star } = w _ { \theta ^ { \star } } )$

$$
\begin{array} { r } { \mathbb { P } \Big ( Y _ { n + 1 } \in \widehat { C } _ { \theta ^ { \star } } ( X _ { n + 1 } ) \Big ) \geq 1 - \alpha , } \end{array}
$$

so $\mathbb { P } ( Y _ { n + 1 } \not \in \widehat { C } _ { \theta ^ { \star } } ( X _ { n + 1 } ) ) \leq \alpha$ . Also, by Equation $( 8 ) , \mathbb { P } ( E ^ { c } ) \le \eta$ . Therefore,

$$
\begin{array} { r } { \mathbb { P } \Big ( Y _ { n + 1 } \in \widehat { C } _ { \mathrm { r o b } } ( X _ { n + 1 } ) \Big ) \geq 1 - \eta - \alpha . } \end{array}
$$

## G.4 Proof of Proposition 4.7

Proposition 4.7. Let $u ( \cdot , t )$ be a PDE solution in some function space $( \mathcal { U } , \Vert \cdot \Vert _ { \mathcal { U } } )$ and ${ \mathbf { } } _ { { \mathbf { } } _ { } u _ { } ^ { t } }$ be its approximation via some numerical scheme on grid size h. Let $C _ { h }$ be a conformal prediction setfor the discretized state with coverage $1 - \alpha$ and assume that an error bound $\| u ( \cdot , t ) - \hat { I } _ { h } u ^ { t } \| _ { \mathcal { U } } \leq \dot { \varepsilon _ { h , t } } ( u _ { 0 } )$ is available via some reconstruction operator $I _ { h } : \mathbb { R } ^ { p }  \mathcal { U } ( e . g . $ , piecewise interpolation). Then we can define a conformal setfor thefunction-space setting, $C \subset { \mathcal { U } } ,$ , with $\mathbb { P } \big ( u ( \cdot , t ) \bar { \in } { \cal C } \big ) \ \geq \ 1 - \alpha$ by

$$
C : = \left\{ v \in \mathcal { U } : \exists z \in C _ { h } \mathit { w i t h } \ \| v - I _ { h } z \| _ { \mathcal { U } } \leq \varepsilon _ { h , t } ( u _ { 0 } ) \right\} .\tag{3}
$$

Proof. Let $( \Omega , { \mathcal { F } } , { \mathcal { P } } )$ be an underlying probability space supporting the random initial condition $u _ { 0 } : \Omega  \mathcal { U }$ and hence the (random) PDE solution at time t,

$$
u ( \cdot , t ) : \Omega \to \mathcal { U } ,
$$

as well as the random numerical state

$$
u ^ { t } : \Omega \to \mathbb { R } ^ { p } .
$$

Assume $I _ { h } : \mathbb { R } ^ { p }  \mathcal { U }$ is measurable and that the deterministic error bound assumed in Proposition 4.7 holds for all $\omega \in \Omega$

$$
\begin{array} { r } { \left\| u ( \cdot , t ) ( \omega ) - I _ { h } u ^ { t } ( \omega ) \right\| _ { \mathcal { U } } \leq \varepsilon _ { h , t } ( u _ { 0 } ( \omega ) ) . } \end{array}\tag{14}
$$

Let $C _ { h } \subset \mathbb { R } ^ { p }$ be a (measurable) prediction set such that

$$
\mathbb { P } \big ( \boldsymbol { u } ^ { t } \in C _ { h } \big ) \ge 1 - \alpha .
$$

Define the lifted set $C \subset { \mathcal { U } }$ by Equation (3), i.e.

$$
C : = \Big \{ u ( \cdot , t ) \in \mathcal { U } : \exists z \in C _ { h } \mathrm { ~ s . t . ~ } \| u ( \cdot , t ) - I _ { h } z \| _ { \mathcal { U } } \leq \varepsilon _ { h , t } ( u _ { 0 } ) \Big \} ,
$$

where the right-hand side is interpreted pointwise in ω through the random radius $\varepsilon _ { h , t } \big ( u _ { 0 } ( \omega ) \big )$ .

Consider the event $E : = \{ \omega : \boldsymbol { u } ^ { t } ( \omega ) \in C _ { h } \} \in \mathcal { F }$ . Fix $\omega \in E$ . Then $z : = { \boldsymbol { \mathsf { u } } } ^ { t } ( \omega ) \in C _ { h }$ . By the pathwise bound Equation (14),

$$
\begin{array} { r } { \big \| u ( \cdot , t ) ( \omega ) - I _ { h } z \big \| _ { \mathcal { U } } = \big \| u ( \cdot , t ) ( \omega ) - I _ { h } u ^ { t } ( \omega ) \big \| _ { \mathcal { U } } \leq \varepsilon _ { h , t } ( u _ { 0 } ( \omega ) ) . } \end{array}
$$

Hence, by definition of $C ,$ we have $u ( \cdot , t ) ( \omega ) \in C$ . Since $\omega \in E$ was arbitrary, this proves the event inclusion

$$
\{ \omega : { \boldsymbol u } ^ { t } ( \omega ) \in C _ { h } \} \subseteq \{ \omega : u ( \cdot , t ) ( \omega ) \in C \} .
$$

Taking probabilities yields

$$
\mathbb { P } ( u ( \cdot , t ) \in C ) \geq \mathbb { P } ( u ^ { t } \in C _ { h } ) \geq 1 - \alpha ,
$$

which is the desired marginal coverage statement.

## G.5 Proof of Theorem 4.5

Theorem 4.5. Consider the drift PDE on a grid with p points, $\partial _ { t } \pmb { u } ^ { t } = c \pmb { w }$ with $t \geq 0 ,$ , drift strength $c \in \mathbb { R }$ , and drift profile w $\in \mathbb { R } ^ { p } ( e . g . , w = 1$ for a constant drift), and let the initial conditions be non-degenerate Gaussian, ${ \pmb u } ^ { 0 } \sim \mathcal { N } ( { \pmb \mu } _ { 0 } , { \pmb \Sigma } _ { 0 } )$ . For z and $z ^ { i }$ as in Proposition 4.4,

$$
d _ { \mathrm { T V } } \bigl ( z , z ^ { i } \bigr ) \ = \ 2 \Phi \Bigl ( \frac { \sqrt { 2 p _ { \mathrm { e f f } } ~ c ^ { 2 } } \left| t _ { 1 } - t _ { 2 } \right| } { 2 } \Bigr ) - 1 , \quad w i t h p _ { \mathrm { e f f } } : = w ^ { \top } \Sigma _ { 0 } ^ { - 1 } w .
$$

Proof. We write $\pmb { v } : = c \pmb { w }$ for the drift vector. In addition to the TV distance, we derive the weights for likelihood-ratio weighting, which we use in the proof of Proposition 4.4. The drift PDE $\partial _ { t } \ b { u } ^ { t } = \ b { v }$ has the solution

$$
\begin{array} { r } { \pmb { u } ^ { t } = \pmb { u } ^ { 0 } + t \pmb { v } . } \end{array}
$$

Since ${ \pmb u } ^ { 0 } \sim \mathcal { N } ( { \pmb \mu } _ { 0 } , { \pmb \Sigma } _ { 0 } )$ , we have for every $t \geq 0 ,$

$$
\begin{array} { r } { \pmb { u } ^ { t } \sim \mathcal { N } ( m _ { t } , \pmb { \Sigma } _ { 0 } ) , \qquad m _ { t } : = \pmb { \mu } _ { 0 } + t \pmb { v } , } \end{array}
$$

i.e., the drift shifts the mean and leaves the covariance unchanged.

Reduction to TV distance on inputs For $m \in \{ 1 , 2 \}$ , each pair $( \boldsymbol { u } _ { i } ^ { t _ { m } } , \boldsymbol { u } _ { i } ^ { t _ { m } + \delta } )$ can be written as

$$
( \boldsymbol { \mathbf { \mathit { u } } } _ { i } ^ { t _ { m } } , \boldsymbol { \mathbf { \mathit { u } } } _ { i } ^ { t _ { m } + \delta } ) = \left( \boldsymbol { \mathbf { \mathit { u } } } _ { i } ^ { t _ { m } } , \mathcal { G } _ { \delta } ( \boldsymbol { \mathbf { \mathit { u } } } _ { i } ^ { t _ { m } } ) \right)
$$

for some solution operator $\mathcal { G } _ { \delta } .$ . Consequently, the joint distribution of $( \pmb { u } _ { i } ^ { t _ { m } } , \pmb { u } _ { i } ^ { t _ { m } + \delta } )$ is supported on a p-dimensional submanifold of $\mathbb { R } ^ { 2 p }$ . This is easiest to see in the case $p = 1$ , where the joint distribution is concentrated on the graph of $\mathcal { G } _ { \delta } ( x )$ in the x–y plane.

The only difference between the joint distributions of $( { \boldsymbol { \mathbf { \mathit { u } } } _ { i } ^ { t _ { 1 } } } , { \boldsymbol { \mathbf { \mathit { u } } } _ { i } ^ { t _ { 1 } + \delta } } )$ and $( { \pmb u } _ { i } ^ { t _ { 2 } } , { \pmb u } _ { i } ^ { t _ { 2 } + \delta } )$ lies in how probability mass is distributed along this graph. In particular, any measurable set either has probability zero under both joint distributions (if it does not intersect the graph), or its probability is entirely determined by the corresponding subset of x-values.

Define

$$
\pmb { \xi } _ { m } : = \mathscr { L } ( \pmb { u } _ { i } ^ { t _ { m } } ) \quad \mathrm { a n d } \quad \pmb { \nu } _ { m } : = \mathscr { L } \big ( \pmb { u } _ { i } ^ { t _ { m } } , \pmb { u } _ { i } ^ { t _ { m } + \delta } \big ) , \qquad m \in \{ 1 , 2 \} ,
$$

where $\mathcal { L }$ denotes the law of a random variable, and define the mapping

$$
T ( x ) : = \big ( x , \mathcal { G } _ { \delta } ( x ) \big ) .
$$

Then $\pmb { \nu } _ { m } = T _ { \# } \pmb { \xi } _ { m }$ . Since $T$ is injective, with inverse given by projection onto the first component restricted to $\mathrm { g r a p h } ( \mathcal { G } _ { \delta } )$ , it follows that

$$
\lVert \pmb { \nu } _ { 1 } - \pmb { \nu } _ { 2 } \rVert _ { \mathrm { T V } } = \lVert \pmb { \xi } _ { 1 } - \pmb { \xi } _ { 2 } \rVert _ { \mathrm { T V } } .
$$

From here on, we will use $\hat { z } : = ( { \boldsymbol { \mathbf { \mathit { u } } } } _ { 1 } ^ { t _ { 1 } } , \ldots , { \boldsymbol { \mathbf { \mathit { u } } } } _ { n } ^ { t _ { 1 } } , { \boldsymbol { \mathbf { \mathit { u } } } } _ { n + 1 } ^ { t _ { 2 } } )$ and $\hat { z } ^ { i }$ for $\hat { z }$ when the ith and last entry are swapped.

Exact TV distance for equal-covariance Gaussians As each $\mathbf { \Delta } \mathbf { \boldsymbol { u } } _ { i } ^ { t }$ is multivariate normal and independent of the others, the joint distributions of zˆ and $\hat { z } ^ { i }$ are again multivariate normal with the same block-diagonal covariance ${ \cal I } _ { n + 1 } \otimes { \pmb \Sigma } _ { 0 } \in \mathbb { R } ^ { p ( n + 1 ) \times p ( n + 1 ) }$ . The mean of zˆ is $\bar { \pmb { \mu } } : =$ $( \pmb { m } _ { t _ { 1 } } , \dots , \pmb { m } _ { t _ { 1 } } , \pmb { m } _ { t _ { 2 } } ) \in \mathbb { R } ^ { p ( n + 1 ) }$ , consisting of n blocks $\mathbf { \nabla } m _ { t _ { 1 } }$ followed by one block $\mathbf { \nabla } m _ { t _ { 2 } }$ . The mean $\bar { \pmb { \mu } } ^ { \ i }$ of $\hat { z } ^ { i }$ equals $\bar { \pmb { \mu } }$ with the ith and last block swapped.

TV distance is invariant under bijective measurable maps, so we may apply the whitening map $\pmb { x } \mapsto ( \pmb { I _ { n + 1 } } \otimes \pmb { \Sigma } _ { 0 } ^ { - 1 / 2 } )$ x to both distributions, which turns the covariance into $I _ { p ( n + 1 ) }$ . For identity covariances we use the following identity, which is used in various papers (e.g., He and Goldfeld [16], Karageorgiou and Prodromidis [18]) and attributed to Barsov and Ul’yanov [4] (as we cannot find access to the original source we use the citation directly from other papers that used the identity).

$$
d _ { \mathrm { T V } } ( z , z ^ { i } ) = d _ { \mathrm { T V } } ( \hat { z } , \hat { z } ^ { i } ) = d _ { \mathrm { T V } } (  { \mathcal { N } } (  { \mu _ { 1 } } , I _ { p ( n + 1 ) } ) ,  { \mathcal { N } } (  { \mu _ { 2 } } , I _ { p ( n + 1 ) } ) ) = 2 \Phi ( \frac { \left\| \mu _ { 1 } - \mu _ { 2 } \right\| _ { 2 } } { 2 } ) - 1
$$

Together with the whitening, it gives

$$
d _ { \mathrm { T V } } ( z , z ^ { i } ) = d _ { \mathrm { T V } } ( \hat { z } , \hat { z } ^ { i } ) = 2 \Phi \Big ( \frac { \big \| ( I _ { n + 1 } \otimes \Sigma _ { 0 } ^ { - 1 / 2 } ) ( \bar { \mu } - \bar { \mu } ^ { i } ) \big \| _ { 2 } } { 2 } \Big ) - 1 .
$$

The vector $\bar { \pmb { \mu } } - \bar { \pmb { \mu } } ^ { i }$ is zero except for the ith block, $\pmb { m } _ { t _ { 1 } } - \pmb { m } _ { t _ { 2 } } = ( t _ { 1 } - t _ { 2 } ) \pmb { v }$ , and the last block, $( t _ { 2 } - t _ { 1 } ) \pmb { v }$ . Hence

$$
\left\| ( I _ { n + 1 } \otimes \Sigma _ { 0 } ^ { - 1 / 2 } ) ( \bar { \pmb { \mu } } - \bar { \pmb { \mu } } ^ { i } ) \right\| _ { 2 } ^ { 2 } = 2 ( t _ { 1 } - t _ { 2 } ) ^ { 2 } \pmb { v } ^ { \top } \Sigma _ { 0 } ^ { - 1 } \pmb { v }
$$

and

$$
d _ { \mathrm { T V } } \big ( z , z ^ { i } \big ) = 2 \Phi \Big ( \frac { \sqrt { 2 v ^ { \top } \Sigma _ { 0 } ^ { - 1 } v } \left| t _ { 1 } - t _ { 2 } \right| } { 2 } \Big ) - 1 .
$$

Moreover, since $1 - \Phi ( z ) \le e ^ { - z ^ { 2 } / 2 }$ , we get

$$
\begin{array} { r } { 1 - d _ { \mathrm { T V } } ( z , z ^ { i } ) = 2 \Big ( 1 - \Phi \big ( \frac { \sqrt { 2 v ^ { \top } \Sigma _ { 0 } ^ { - 1 } v } | t _ { 1 } - t _ { 2 } | } { 2 } \big ) \Big ) \leq 2 \exp \Big ( - \frac { ( t _ { 1 } - t _ { 2 } ) ^ { 2 } v ^ { \top } \Sigma _ { 0 } ^ { - 1 } v } { 4 } \Big ) , } \end{array}
$$

so $d _ { \mathrm { T V } } ( z , z ^ { i } ) \to 1$ exponentially fast in $( t _ { 1 } - t _ { 2 } ) ^ { 2 } v ^ { \top } \Sigma _ { 0 } ^ { - 1 } v$ . With ${ \pmb v } = c { \pmb w }$ , we have $\pmb { v } ^ { \top } \pmb { \Sigma } _ { 0 } ^ { - 1 } \pmb { v } =$ $c ^ { 2 } p _ { \mathrm { e f f } }$ , which gives Theorem 4.5.

Likelihood ratio at the centers Let $p _ { t }$ denote the density of $\mathcal { N } ( \boldsymbol { m } _ { t } , \pmb { \Sigma } _ { 0 } )$ and $\Delta : = \pmb { m } _ { t _ { 2 } } - \pmb { m } _ { t _ { 1 } } =$ $( t _ { 2 } - t _ { 1 } ) \pmb { v }$ . The Gaussian likelihood ratio is

$$
r ( \pmb { x } ) : = \frac { p _ { t _ { 2 } } ( \pmb { x } ) } { p _ { t _ { 1 } } ( \pmb { x } ) } = \mathrm { e x p } \Big ( \Delta ^ { \top } \Sigma _ { 0 } ^ { - 1 } \pmb { x } - \frac { 1 } { 2 } \big ( \pmb { m } _ { t _ { 2 } } ^ { \top } \Sigma _ { 0 } ^ { - 1 } m _ { t _ { 2 } } - \pmb { m } _ { t _ { 1 } } ^ { \top } \Sigma _ { 0 } ^ { - 1 } m _ { t _ { 1 } } \big ) \Big ) .
$$

Evaluating at $\pmb { x } _ { \mathrm { t e s t } } : = \pmb { m } _ { t _ { 2 } }$ and $\pmb { x } _ { \mathrm { c a l } } : = \pmb { m } _ { t _ { 1 } }$ gives

$$
\begin{array} { r } { \log r ( \pmb { m } _ { t _ { 2 } } ) = \frac { 1 } { 2 } \Delta ^ { \top } \pmb { \Sigma } _ { 0 } ^ { - 1 } \Delta , \qquad \log r ( \pmb { m } _ { t _ { 1 } } ) = - \frac { 1 } { 2 } \Delta ^ { \top } \pmb { \Sigma } _ { 0 } ^ { - 1 } \Delta . } \end{array}
$$

Since $\begin{array} { r } { \Delta ^ { \top } \Sigma _ { 0 } ^ { - 1 } \Delta = ( t _ { 1 } - t _ { 2 } ) ^ { 2 } \pmb { v } ^ { \top } \pmb { \Sigma } _ { 0 } ^ { - 1 } \pmb { v } = : \kappa } \end{array}$ , we obtain

$$
r (  { \mathbf { x } } _ { \mathrm { t e s t } } ) = e ^ { \kappa / 2 } , \qquad r (  { \mathbf { x } } _ { \mathrm { c a l } } ) = e ^ { - \kappa / 2 } .
$$

Normalized weights If the calibration set contains n points with ratio $r ( \pmb { x } _ { \mathrm { c a l } } )$ and we normalize over these n points plus the test point, then

$$
w _ { \mathrm { t e s t } } = \frac { r ( x _ { \mathrm { t e s t } } ) } { n r ( x _ { \mathrm { c a l } } ) + r ( x _ { \mathrm { t e s t } } ) } = \frac { e ^ { \kappa } } { n + e ^ { \kappa } } = 1 - O \big ( n e ^ { - \kappa } \big ) ,
$$

and for any such calibration point,

$$
w _ { \mathrm { c a l } } = \frac { r ( { \bf x } _ { \mathrm { c a l } } ) } { n r ( { \bf x } _ { \mathrm { c a l } } ) + r ( { \bf x } _ { \mathrm { t e s t } } ) } = \frac { 1 } { n + e ^ { \kappa } } = O ( e ^ { - \kappa } ) .
$$

## G.6 Proof of Proposition 4.4

Proposition 4.4. Consider the drift $P D E \partial _ { t } u ( x , t ) = c .$ with $x \in ( 0 , 1 ) , t \geq 0 , c \in \mathbb { R }$ , and compatible boundary conditions. Define $z \widetilde { a n d } z ^ { i }$ as in Equation (2) with n i.i.d. calibration pairs at times $( t _ { 1 } , t _ { 1 } + \delta )$ and a test pair at time $( t _ { 2 } , t _ { 2 } + \delta )$ from an affine scheme on a grid with p points and Gaussian initial conditions. Then, the total variation distance between the laws ofz and z<sup>i</sup> is

$$
d _ { \mathrm { T V } } \left( z , z ^ { i } \right) ~ = ~ 2 \Phi \Bigl ( { \frac { \sqrt { 2 p c ^ { 2 } } \left| t _ { 1 } - t _ { 2 } \right| } { 2 } } \Bigr ) - 1 ,
$$

where Φ is the cumulative distribution function of the standard normal distribution. Hence, $d _ { \mathrm { T V } } ( z , z ^ { i } ) \to 1$ exponentially fast in the grid size p and with it, the coverage guarantee from Barber et al. [3] vanishes exponentially.

Moreover, for likelihood-ratio weighting as in Equation (7) evaluated at the distribution centers $\pmb { x } _ { \mathrm { t e s t } } : = \mathbb { E } [ \pmb { u } _ { n + 1 } ^ { t _ { 2 } } ]$ and $\pmb { x } _ { \mathrm { c a l } } : = \mathbb { E } [ \pmb { u } _ { i } ^ { \top _ { 1 } } ]$ ], the test weight satisfies

$$
w _ { \mathrm { t e s t } } = 1 - O \big ( n e ^ { - p c ^ { 2 } ( t _ { 1 } - t _ { 2 } ) ^ { 2 } } \big ) , \quad w _ { \mathrm { c a l } } = O \big ( e ^ { - p c ^ { 2 } ( t _ { 1 } - t _ { 2 } ) ^ { 2 } } \big ) .
$$

Hence, the weighted conformal bands become infinite at an exponential rate.

Proof. Proposition 4.4 is the special case ${ \pmb w } = { \bf 1 }$ and $\Sigma _ { 0 } = I _ { p }$ of the proof in Appendix G.5. Then $p _ { \mathrm { e f f } } = p$ and $\kappa = p c ^ { 2 } ( t _ { 1 } - t _ { 2 } ) ^ { 2 }$ , which gives the TV distance, its exponential convergence in p, and the weights. □

## G.7 General TV distance bound

Proposition G.1 (Extension of Proposition 4.4). Consider a linear partial differential equation discretized on a spatial grid with p points. Let $\mathbf { u } ^ { t _ { 1 } } , \mathbf { u } ^ { t _ { 2 } } \in \mathbb { R } ^ { p }$ be the Gaussian numerical solutions $\mathcal { N } ( \mu _ { t _ { 1 } } , \Sigma _ { t _ { 1 } } )$ and $\mathcal { N } ( \mu _ { t _ { 2 } } , \Sigma _ { t _ { 2 } } )$ at two distinct time steps.

Suppose there exist uniform spectral bounds $0 < \underline { { \lambda } } \le \overline { { \lambda } } < \infty$ such that for $t \in \{ t _ { 1 } , t _ { 2 } \}$ , the eigenvalues of the covariance matrices satisfy $\lambda ( \Sigma _ { t } ) \in [ \underline { { \lambda } } , \overline { { \lambda } } ]$ . Then the total variation distance between the laws ofthe numerical solutions satisfies

$$
d _ { \mathrm { T V } } \big ( \mathbf { u } ^ { t _ { 1 } } , \mathbf { u } ^ { t _ { 2 } } \big ) \leq \frac { 1 } { 2 } \sqrt { p \left( \frac { \overline { { \lambda } } } { \underline { { \lambda } } } - 1 + \log \frac { \overline { { \lambda } } } { \underline { { \lambda } } } \right) + \frac { \| \mu _ { t _ { 1 } } - \mu _ { t _ { 2 } } \| ^ { 2 } } { \underline { { \lambda } } } } .
$$

For a consistent discretization of a smooth Gaussian random field, $\underline { { \lambda } } \to 0$ as p grows, so the uniform spectral bounds only hold for a fixed range of grid sizes.

Proof. The result follows by bounding the Kullback–Leibler (KL) divergence and applying Pinsker’s inequality. Let $P = \mathcal { N } ( \pmb { \mu } _ { t _ { 1 } } , \pmb { \Sigma } _ { t _ { 1 } } )$ and $Q = \mathcal { N } ( \mu _ { t _ { 2 } } , \Sigma _ { t _ { 2 } } )$ . By Pinsker’s inequality,

$$
d _ { \mathrm { T V } } ( P , Q ) \leq { \sqrt { { \frac { 1 } { 2 } } \operatorname { K L } ( P \| Q ) } } .
$$

The KL divergence for $p \cdot$ dimensional Gaussian measures is well known as

$$
\operatorname { K L } ( P \| Q ) = { \frac { 1 } { 2 } } \left[ \operatorname { t r } ( \Sigma _ { t _ { 2 } } ^ { - 1 } \Sigma _ { t _ { 1 } } ) - p + \log { \frac { \operatorname* { d e t } \Sigma _ { t _ { 2 } } } { \operatorname* { d e t } \Sigma _ { t _ { 1 } } } } + ( \mu _ { t _ { 1 } } - \mu _ { t _ { 2 } } ) ^ { \top } \Sigma _ { t _ { 2 } } ^ { - 1 } ( \mu _ { t _ { 1 } } - \mu _ { t _ { 2 } } ) \right] .
$$

First, we bound the terms related to the covariance structure. Under the assumption that the eigenvalues of $\Sigma _ { t _ { 1 } }$ and $\Sigma _ { t _ { 2 } }$ are bounded by $\underline { { \lambda } }$ and $ { \overline { { \lambda } } } ,$ the trace of the product satisfies

$$
\mathrm { t r } \big ( \Sigma _ { t _ { 2 } } ^ { - 1 } \Sigma _ { t _ { 1 } } \big ) \leq p \cdot \lambda _ { \operatorname* { m a x } } \big ( \Sigma _ { t _ { 2 } } ^ { - 1 } \big ) \lambda _ { \operatorname* { m a x } } \big ( \Sigma _ { t _ { 1 } } \big ) \leq p \frac { \overline { { \lambda } } } { \underline { { \lambda } } } ,
$$

where the first inequality follows from von Neumann’s trace inequality as the matrices are symmetric and positive definite.

Similarly, the log-determinant ratio is bounded by the sum of the logs of the eigenvalue ratios:

$$
\log \frac { \operatorname* { d e t } \Sigma _ { t _ { 2 } } } { \operatorname* { d e t } \Sigma _ { t _ { 1 } } } = \sum _ { i = 1 } ^ { p } \log \frac { \lambda _ { i } ( \Sigma _ { t _ { 2 } } ) } { \lambda _ { i } ( \Sigma _ { t _ { 1 } } ) } \leq p \log \frac { \overline { { \lambda } } } { \underline { { \lambda } } } .
$$

Next, the quadratic form involving the mean shift is bounded using the spectral norm of the inverse covariance matrix:

$$
( \mu _ { t _ { 1 } } - \mu _ { t _ { 2 } } ) ^ { \top } \Sigma _ { t _ { 2 } } ^ { - 1 } ( \mu _ { t _ { 1 } } - \mu _ { t _ { 2 } } ) \leq \lambda _ { \operatorname* { m a x } } ( \Sigma _ { t _ { 2 } } ^ { - 1 } ) \Vert \mu _ { t _ { 1 } } - \mu _ { t _ { 2 } } \Vert ^ { 2 } \leq \frac { \Vert \mu _ { t _ { 1 } } - \mu _ { t _ { 2 } } \Vert ^ { 2 } } { \underline { { \lambda } } } .
$$

Substituting these bounds into the KL divergence expression, we obtain

$$
\mathrm { K L } ( P \| Q ) \leq \frac { 1 } { 2 } \left[ p \left( \frac { \overline { { \lambda } } } { \underline { { \lambda } } } - 1 + \log \frac { \overline { { \lambda } } } { \underline { { \lambda } } } \right) + \frac { \| \mu _ { t _ { 1 } } - \mu _ { t _ { 2 } } \| ^ { 2 } } { \underline { { \lambda } } } \right] .
$$

Applying the square root from Pinsker’s inequality yields the stated bound.

## H Experimental details

We tracked our results with Weights & Biases [5].

## H.1 Methods

Supr. The Supr. method follows the suggestion by Diquigiovanni et al. [7]. We calculate the scores at calibration time as the maximum over the spatial dimension x. Then, at test time the respective quantile of these scores is added to each spatial location of the test prediction to generate the conformal interval. No further fine-tuning is necessary.

![](images/ce7d55e43659cdb57456fae01bd3f19f01d52dd187edd502a7cafb19ac1e6765.jpg)  
Figure H.1: Exemplary coverage plots for the methods (from left to right) Supr., LSCI, WCP Learned, and WCP Gauss (Ours) for a sample at time step 10 with parameters $a = - 0 . 0 0 7 5 , b = 0 , c = 0 . 5 .$

LSCI The LSCI method is difficult to fine-tune. In particular, we have four different parameters: λ, the embedding dimension, the metric, and the number of samples to use for the conformal bands. As the computational burden is high for this method, we opted for a mid-range setup, and chose (following the original work) $\lambda = { \bar { 3 } } .$ , embedding dimension 20, the supremum norm, and varied the number of samples between 500 and 1000, depending on what gave the closest to the target coverage for the first time step. This fine-tuning would not be feasible in practice, but as it favors the baseline we compare against, it does not harm the validity of our method.

DtACI To showcase adaptive methods, we use Dynamically-Tuned Adaptive Conformal Inference (DtACI) by Gibbs and Candès [10]. They dynamically adjust their α value according to previous miscoverage to satisfy the original $1 - \alpha$ guarantee. In comparison to previous methods [e.g., 9], this method does not need manual tuning of the correction parameter, and the paper suggests calculation for some default parameters which we followed. For the rest of the free parameters, we followed the examples from their paper and chose 8 agents with γ one of [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128] and a window size of 5 (due to the short time frame we observe).

WCP Learned The WCP Learned method is a WCP method according to the original proposal of Tibshirani et al. [32]. There, they suggest estimating the (unknown) weights by the probabilistic output of a classifier. We experimented with multiple versions of LogisticRegression, Random Forests, and Support Vector Machines. We found gradient boosting to yield the best results in our experiments (n\_estimators=50, max\_depth=3, learning\_rate=0.1). There might be better models, e.g., certain deep learning models could model the underlying distributions better. However, the quality of a model depends on the data, i.e., a working model on one dataset might undercover on another. Also, as the models need to be trained at every prediction step, large deep learning models are practically infeasible.

WCP Gauss Our proposed method is also based on Tibshirani et al. [32]. However, instead of estimating the weights, we use the knowledge of the underlying distribution to calculate them precisely. The mean and covariance are always estimated from the data using the average for the mean and Ledoit–Wolf [19] for the covariance.

## H.2 Data generation

We generated multiple synthetic datasets from a general second-order formulation for periodic PDEs, each discretized with finite-difference schemes with exact exponential matrix evolution. Initial conditions were sampled from Gaussian processes with covariance $6 2 5 ( - \partial _ { x } ^ { 2 } + 2 5 I ) ^ { - 2 }$ to provide smooth but nontrivial trajectories. Each dataset is stored as a compressed .npz file containing $( n _ { \mathrm { s a m p l e s } } , n _ { t } + 1 , n _ { x } )$ trajectories together with grid and metadata. The PDE looks as follows

$$
\begin{array} { r l } & { u _ { t } ( x , t ) = a u ( x , t ) _ { x x } + b u ( x , t ) _ { x } + c u ( x , t ) , ~ x \in ( 0 , 2 \pi ) , \ t \geq 0 , } \\ & { \qquad u ( x , 0 ) = u _ { 0 } ( x ) , ~ x \in ( 0 , 2 \pi ) , } \end{array}
$$

with periodic boundary conditions, and where $a , b , c \in \mathbb { R }$ are tunable parameters. We sampled data from all combinations of $a ~ \in ~ \{ - 0 . 0 0 5 , - 0 . 0 0 7 5 , - 0 . 0 1 \} , ~ b ~ \in ~ \{ - 0 . 5 , 0 , 0 . 5 \}$ , and $c \in \{ - 0 . 5 , 0 , 0 . 5 \}$ . Note that we focused on negative values for $^ { a , }$ as positive values tend to smooth the trajectory over time which leads to trivial coverage even for naïve CP. We sampled 21 time steps (incl. initial condition) between 0 and 1 (for the recalibration task, we extended the time to 2, as otherwise all methods cover too easily) and sampled 5000 trajectories for training, 500 for validation, 5000 for calibration, and 1000 for testing, and use a spatial resolution of $n _ { x } = 4 8$ . As surrogate, we train a 1D Fourier Neural Operator [21] (3 layers, 12 Fourier modes, 32 hidden channels) to predict one time step ahead, using AdamW (learning rate $1 0 ^ { - 3 }$ , cosine schedule) on the MSE loss for 5 epochs.

## H.3 Ablation study

The critical part of the WCP Gauss method is the accurate estimation of the covariance. If the dimensionality increases or the number of samples for estimation drops, we expect instability in the covariance estimation and, therefore, instability in the coverage results. In Figure H.2 we see an example of the synthetic experiment when we increase grid granularity (i.e., the dimensionality of the problem) or reduce the number of calibration samples. The same would hold for reducing the number of test samples, but we omit this from further plots so as not to overload the appendix. We can see that a higher granularity results in an earlier onset of infinite bands (as suggested in Proposition 4.4) and that the method becomes unstable when the number of estimation samples is too low for the dimensionality of the problem.

## H.4 Visualizations of theoretical results

Proposition 4.7 We use the backward heat equation to visualize this proposition. As we do not know the exact approximation error, we cannot construct conformal sets as described in the proposition. It is, however, reasonable to assume that the miscoverage should decrease as the approximation error drops (because what we would theoretically add to the numerical conformal intervals will get smaller with finer granularity). Therefore, we sampled numerical trajectories for the backward heat equation (with parameter $\nu = - 0 . 0 0 1$ and the granularities seen in the figure) as described in Appendix H.2. For each granularity, we used the sampled initial conditions to generate the analytic solution from its Fourier representation using 16 modes for sine and cosine each. We then proceeded as in the synthetic case (general task), calibrated with the numerical data, predicted from numerical input, and checked coverage with the analytical output. The depicted coverages come from the first time step after the calibration (at later time steps, the negative effect from Proposition 4.4 reduced the observed effect, and the methods started undercovering or getting infinite bandwidth).

Proposition 4.4 To show that a higher granularity implies a higher TV distance and faster infinite bandwidth, we sampled from the PDE in the proposition (with drift parameter 1 and the granularities seen in the figure) as described in Appendix H.2. For each granularity, we calculated the coverage as in the synthetic case (general task). The depicted time step is 10, as this was the time step where the effect for our chosen PDE was best visible (earlier time steps would need higher granularity to drop in coverage, and later ones dropped too quickly). The theoretical lower bound was calculated as described in Barber et al. [3] as $\begin{array} { r } { 1 - \bar { \alpha } - \sum _ { i = 1 } ^ { n } \tilde { w } _ { i } { \dot { d } } _ { \mathrm { T V } } ( z , z ^ { i } ) } \end{array}$ with equal weights $w _ { i }$ . Here, z and $z ^ { i }$ are described in Proposition 4.4.

Correlated initial conditions Figure H.3 evaluates the TV distance of Theorem 4.5 for the constant drift v = 1 under correlated initial conditions. If the initial field and the drift are smooth, neighboring grid points are strongly correlated and $\boldsymbol { v } ^ { \top } \Sigma _ { 0 } ^ { - 1 }$ v stays bounded as the grid is refined, so the TV distance levels off below 1. A drift that is rougher than the initial field, such as a step, still drives the TV distance to 1.

## I Real-world examples

## I.1 Pulsed-thermography data

The dataset of Wei et al. [34] contains 19 samples of 2D surfaces measured over 1810 time steps. We allocate 3 samples for training a Fourier Neural Operator [21], 1 for validation, 13 for calibration and testing. 2 samples were excluded due to large outlier noise. We crop the borders of the measurement data, since they correspond to non-heated background regions (see Figure I.1). We then determine the time frame where cooling begins, by finding the frame with the largest temperature and use the 20 following time frames as the dataset. As we consider an inverse problem, we flip the time axis. For the recalibration task, we additionally use a time stride (i.e., we remove every second frame from the data), because otherwise all methods always covered (as they provide temporally very fine-grained data). Because the dataset is small, we divide each surface into square patches of size $1 0 \times 1 0$ and treat them as identically distributed samples (to mitigate dependency between the patches, we leave an empty space of size 5 to all neighboring patches). While this induces some dependency between patches, and the variations in the metal plates (e.g., differently drilled backside holes) introduce slight distribution shifts, both effects are minor for a real-world measurement dataset and have a negligible impact on CP performance. We further do not split the trajectories into different calibration and test sets. We calibrate at an earlier time than we test, but for the same trajectories. This also introduces dependency structures, but again mitigates the low dataset size and mirrors the real-world setting, where the same processes are observed over time. We can see in Figure I.2 that WCP Gauss is the only method with consistent coverage over time.

## I.2 Lake Maggiore thermistor data

The Lake Maggiore thermistor dataset of Scheu et al. $[ 3 1 ] ^ { 9 }$ contains 1-minute-resolution thermistor chain measurements at multiple depths in Pallanza Bay. We use 50 nights in early spring for calibration and 14 nights in late spring for testing. A 1D Fourier Neural Operator [21] (3 layers, 3 Fourier modes, 32 hidden channels) is trained on 15 further early-spring nights (and validated on 5) to predict one time step ahead, for up to 200 epochs with early stopping. We subsample to 30-minute resolution across 7 depth sensors and extract nighttime windows as independent trajectories of vertical heat diffusion without solar forcing, which fits the linear-PDE setting of Proposition 4.2. We z-scored with the global mean and standard deviation of the training nights. As for the pulsed-thermography data, we frame the prediction as an inverse problem and reverse the time axis. Following the discussion after Theorem 4.5, we restrict CP to the middle depth sensors (27, 30, and 35 m above bed) rather than the full water column, exploiting the more favorable marginal distribution. Figure I.3 shows the resulting coverages for both tasks: WCP Gauss again maintains the target coverage almost everywhere (note that with a test size of 14, single-step deviations from 90% are statistically expected).

## I.3 FRAP data

The FRAP dataset of Wåhlstrand Skärström et al. [35]<sup>10</sup> contains fluorescence-recovery-afterphotobleaching experiments on 32 wt% sucrose solutions, whose post-bleach intensity field obeys the standard diffusion equation and again fits Proposition 4.2. We train a 2D Fourier Neural Operator [21] on 400 frame pairs from 20 simulated recovery fields of size $2 5 6 \times 2 5 6$ pixels, and use 20 real experiments for calibration and test (we use different time steps for calibration and test, not different trajectories). The training fields are produced by a spectral diffusion simulator with randomly sampled diffusion coefficient, bleach depth, additive-noise level, and bleach radius according to the original paper. Real recoveries are computed as $\Delta F / F _ { 0 }$ relative to the per-pixel pre-bleach mean, percentile-and-MAD clipped to suppress imaging outliers, Gaussian-smoothed with $\sigma { = } 0 . 6 \mathrm { p x }$ , and per-trajectory z-scored before training. The 5 5 box filter is applied only at CP score extraction and is not seen by the FNO. From every $2 5 6 \times 2 5 6$ frame we extract $\phantom { - } 1 3 \times 3$ grid of pixels around the bleach center, spaced 5 pixels apart and each smoothed with a $5 \times 5$ box filter, yielding a 9-dimensional time series. The time axis is then truncated, time-strided, and reversed to frame the problem as an inverse one. Following the discussion after Theorem 4.5, restricting CP to the center of the bleach region (where the laser is focused and the recovery dynamics are concentrated) rather than to the full $2 5 6 \times 2 5 6$ field exploits the more favorable marginal distribution. Figure I.4 shows the resulting coverages together with sample recovery snapshots. Only WCP Gauss attains non-trivial target coverage at every time step on both tasks (with the same test-size caveat as above).

## J Robustness experiments

## J.1 Multi-seed stability

Figure J.1 repeats the general task of the synthetic experiment (Figure K.5, $a = - 0 . 0 0 7 5 )$ over 10 reshuffled calibration/test splits and the pulsed-thermography experiment over 10 plate-level reshuffles of the train/validation/calibration/test assignment, each with a retrained surrogate. Since each seed uses a different plate assignment and surrogate, the pulsed-thermography curves differ from those in Figure I.2. The standard deviation across seeds is small and the ranking of the methods is the same as in the single-seed figures. LSCI is omitted due to its computational cost.

## J.2 Nonlinear PDEs

We add a cubic reaction term to the synthetic PDE, $u _ { t } = a u _ { x x } + c u - \gamma u ^ { 3 }$ with $a = - 0 . 0 0 7 5$ and $b = 0 .$ , where $\gamma = 0$ recovers the linear PDE. For $\gamma > 0 .$ , the solutions are no longer Gaussian and Proposition 4.2 does not apply. Figure J.2 shows that coverage then depends on the PDE parameters: for $c = 0 . 5$ , WCP Gauss attains the target for all $\gamma ,$ while for $c = 3$ the solutions saturate into a bimodal distribution and at $\gamma = 1 0$ the coverage of WCP Gauss drops to about 0.5. Coverage on nonlinear PDEs is therefore not guaranteed and can be lost by changing the parameters.

## J.3 Bandwidths

The bandwidth of WCP Gauss is determined by the likelihood ratio of the test sample. A test sample that is likely under the test distribution but unlikely under the calibration distribution has a large weight $\tilde { w } _ { n + 1 }$ , which moves the weighted quantile in Equation (6) upwards. Once $\tilde { w } _ { n + 1 } > \alpha$ , the band is infinite. Figure J.3 shows this for the synthetic PDE: the width increases monotonically with the likelihood ratio, and infinite bands occur exactly for the samples with $\tilde { w } _ { n + 1 } > \alpha$

## J.4 Other initial distributions

We further show results when starting sampling from other initial distributions. In Figure K.7, we can see coverages when initially sampling from a Laplace distribution and in Figure K.8 when sampling from a Logistic distribution. As the results are almost equivalent to those with Gaussian initial conditions, we only show the graphs for $b = 0$ and refrain from further plots or tables. In Figure K.9, we show coverages of WCP Gauss (with Cauchy instead of Gaussian likelihood-ratio weights) when sampling from heavy-tailed Cauchy initial conditions over multiple seeds. The method stays stable over the seed distribution.

## K Additional results on synthetic data

We present the additional results (plots and tables) for the synthetic experiments. For the recalibration task, we find them in Figure K.1 and Table K.1 for $b = - 0 . 5$ , in Figure K.2 and Table K.2 for $b = 0$ and in Figure K.3 and Table K.3 for $b = 0 . 5$ . For the general task, we find them in Figure K.4 and Table K.4 for $b = - 0 . 5$ , in Figure K.5 and Table K.5 for $b = 0$ , and in Figure K.6 and Table K.6 for $b = 0 . 5$

All results show similar structures. If the parameter set yields a simple dynamic, all methods cover even far beyond the calibration points. But when the dynamics get rougher, all but the WCP Gauss method undercover. WCP Gauss refrains from false predictions and instead yields infinite bands. DtACI yields worse results than is usually seen in the literature. The reason is that we observe only a short prediction horizon. When using adaptive CP on stock market data for example, thousands of time steps are available and changes happen slowly.

## L Computing resources

All the experiments were conducted on a mix of CPUs and GPUs (A40, A100, H100), based on the availability of our academic cluster. Most runs are comfortably executed on CPUs, but for the image data, GPUs speed up the process. No run takes longer than a few minutes. For the LSCI method, it is highly advised to use the computation cache that JAX provides. Otherwise, these calculations can take far longer, even on GPUs. Memory and storage do not exceed 10 GB per dataset. The full research project required more compute than the experiments reported in the paper, due to failed runs, parameter tuning, or experiments that were not relevant enough for the final cut.

Table K.1: Coverage and mean bandwidth for the recalibration task for varying a, c and fixed b = 0.5. We also report the fraction of samples where infinite bands were reported $n _ { \infty }$ . The mean bandwidth is taken over finite bands only. The gray font is chosen for better readability.
<table><tr><td colspan="2"></td><td></td><td colspan="4">c = −0.5</td><td colspan="4">c = 0.0</td><td colspan="5">c = 0.5</td></tr><tr><td colspan="2"></td><td>T:</td><td>1</td><td>5</td><td>10 15</td><td>20</td><td>1</td><td>5</td><td>10</td><td>15 20</td><td>1</td><td>5</td><td>10</td><td>15</td><td>20</td></tr><tr><td colspan="2">Supr.</td><td>Cov. Bw.</td><td>.91 .0125</td><td>.92 .0109</td><td>.91 .90 .0089 .0073</td><td>.90 .0065</td><td>.91 .0133</td><td>.91 .0134</td><td>.90 .0135</td><td>.91 .89 .0145 .0161</td><td>.91 .0142</td><td>.89 .0171</td><td>.88 .0243</td><td>.89 .0419</td><td>.88 .0793</td></tr><tr><td colspan="2">LSCI</td><td>Cov. Bw.</td><td>.96 .0121</td><td>.96 .0069</td><td>.92 .91 .0052 .0052</td><td>.90 .0056</td><td>.96 .0126</td><td>.95 .91 .0108 .0101</td><td>.94 .0148</td><td>.86 .0152</td><td>.97 .0133</td><td>.96 .0167</td><td>.97 .0238</td><td>.97 .0508</td><td>.96 .0988</td></tr><tr><td colspan="2">0005 DtACI</td><td>Cov. Bw.</td><td>.91 .0125</td><td>.97 .0117</td><td>.96 .96 .0094 .0080</td><td>.94 .0068</td><td>.91 .0133</td><td>.93 .0129</td><td>.93 .93 .0131 .0138</td><td>.86 .0150</td><td>.91 .0142</td><td>.88 .0145</td><td>.89 .0212</td><td>.91 .0344</td><td>.91 .0650</td></tr><tr><td colspan="2">ⅡI WCP a Learned</td><td>Cov. Bw.</td><td>.90 .0117</td><td>.89 .0100 .0083</td><td>.88 .89 .0070</td><td>.89</td><td>.90 .89 .0124</td><td>.89</td><td>.89</td><td>.87</td><td>.90</td><td>.88</td><td>.88</td><td>.89</td><td>.87</td></tr><tr><td rowspan="3">WCP Gauss (Ours)</td><td rowspan="2"></td><td rowspan="2">n∞ Cov.</td><td></td><td></td><td></td><td>.0063 0%</td><td></td><td>.0125 .0128</td><td>.0138</td><td>.0158</td><td>.0133</td><td>.0165</td><td>.0238</td><td>.0410</td><td>.0768</td></tr><tr><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0% 0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td></tr><tr><td rowspan="2">Bw.</td><td rowspan="2">.91</td><td>.91</td><td>.91</td><td>.93</td><td>.98</td><td>.91 .91</td><td>.90</td><td>.93</td><td>.97</td><td>.91</td><td>.92</td><td>.91</td><td>.89</td><td>.86</td></tr><tr><td>.0127</td><td>.0106 0%</td><td>.0087 0% 0%</td><td>.0080 .0086</td><td>.0134</td><td>.0138 0%</td><td>.0135</td><td>.0158</td><td>.0218</td><td>.0143</td><td>.0192</td><td>.0283</td><td>.0430 .0708</td></tr><tr><td colspan="2">Supr.</td><td>n∞ 0% Cov. Bw.</td><td>.91 .0126</td><td>.92 .0110</td><td>.91 .87</td><td>0% .81</td><td>0% .91 .91</td><td>0% .89</td><td>0% .84</td><td>0% .75</td><td>0% .91</td><td>0% .89</td><td>0% .88</td><td>1% .87</td><td>2% .83</td></tr><tr><td colspan="2">LSCI</td><td>Cov.</td><td>.95</td><td>.0093 .95 .89</td><td>.0089 .83</td><td>.0122 .84</td><td>.0133 .0136 .96 .93</td><td>.0142 .86</td><td>.0179 .81</td><td>.0316 .77</td><td>.0142 .98</td><td>.0172 .94</td><td>.0251 .91</td><td>.0461 .89</td><td>.1042 .85</td></tr><tr><td colspan="2">0.0075 DtACI</td><td>Bw. Cov.</td><td>.0122 .91</td><td>.0070 .0068 .97 .96</td><td>.0086 .88</td><td>.0129 .64</td><td>.0127 .0110 .91 .93</td><td>.0132 .91</td><td>.0182 .73</td><td>.0333 .47</td><td>.0134 .91</td><td>.0142 .88</td><td>.0256 .88</td><td>.0584 .78</td><td>.1263 .69</td></tr><tr><td colspan="2">I WCP</td><td>Bw. Cov.</td><td>.0126 .90 .89</td><td>.0118 .0097 .88</td><td>.0090 .85</td><td>.0111 .81</td><td>.0133 .0130 .90 .89</td><td>.0137 .88</td><td>.0165 .81</td><td>.0279 .74</td><td>.0142 .90</td><td>.0146 .88</td><td>.0220 .87</td><td>.0394 .85</td><td>.0914 .82</td></tr><tr><td colspan="2">Learned</td><td>Bw. n∞</td><td>.0118 0%</td><td>.0102 .0088 0% 0%</td><td>.0087 0%</td><td>.0122 0%</td><td>.0124 .0128 0% 0%</td><td>.0138 0%</td><td>.0175 0%</td><td>.0314 0%</td><td>.0133 0%</td><td>.0168 0%</td><td>.0246 0%</td><td>.0442 0%</td><td>.1017 0%</td></tr><tr><td colspan="2">WCP Gauss (Ours)</td><td>Cov. Bw. n∞ 0%</td><td>.92 .0127 0%</td><td>.91 .0107 .0091</td><td>.90 .98 .0112</td><td>.96 .0146</td><td>.92 .91 .0136 .0140</td><td>.89 .0141</td><td>.96 .0223</td><td>.97 .0395</td><td>.91 .0144</td><td>.91 .0192</td><td>.90 .0276</td><td>.88 .0453</td><td>.89 .1134</td></tr><tr><td colspan="2">Supr.</td><td>Cov. Bw.</td><td>.91 .0126</td><td>.92 .0112</td><td>.87 .72 .0102 .0154</td><td>.69 .0320</td><td>.91 .0133</td><td>.91 .84 .0139 .0160</td><td>.65 .0312</td><td>.60 .0829</td><td>.91 .0143</td><td>.89 .0174</td><td>0% 2% .84 .0274</td><td>5% .68 .0696</td><td>6% .59 .2281</td></tr><tr><td colspan="2">LSCI</td><td>Cov. Bw.</td><td>.95 .0123</td><td>.92 .0075 .0090</td><td>.78 .75 .0161</td><td>.78 .0349</td><td>.96 .0127</td><td>.89 .74 .0094 .0151</td><td>.68 .0328</td><td>.68 .0897</td><td>.97 .0134</td><td>.89 .0140</td><td>.80 .0304</td><td>.74 .0810</td><td>.66 .2472</td></tr><tr><td colspan="2">001 DtACI</td><td>Cov. Bw.</td><td>.91 .0126</td><td>.96 .90 .0118 .0102</td><td>.47 .0137</td><td>.43 .0282</td><td>.91 .0133 .0130</td><td>.93 .75 .0149</td><td>.34 .0275</td><td>.33 .0731</td><td>.91 .0143</td><td>.87 .0147</td><td>.70 .0242</td><td>.38 .0617</td><td>.32 .2040</td></tr><tr><td colspan="2">II WCP a Learned</td><td>Cov. Bw.</td><td>.90 .0118</td><td>.89 .84 .0103 .0099</td><td>.72 .0154</td><td>.70 .0320</td><td>.90 .0124</td><td>.89 .0129</td><td>.81 .64 .0155 .0310</td><td>.61 .0829</td><td>.90 .0134</td><td>.88 .0171</td><td>.83 .0269</td><td>.66 .0685</td><td>.59 .2275</td></tr><tr><td colspan="2">WCP</td><td>n∞ 0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0% 0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td></tr><tr><td colspan="2">Gauss (Ours)</td><td>Cov. Bw.</td><td>.92 .91 .0128 .0111</td><td>.97 .0125</td><td>.93 .0179</td><td>.94 .0394</td><td>.92 .0137</td><td>.91 .0144</td><td>.93 .0190</td><td>.95 .0390 .1034</td><td>.93</td><td>.92 .0146</td><td>.92 .0196 .0318</td><td>.91 .93 .0876</td><td>.93</td></tr></table>

Table K.2: Coverage and mean bandwidth for the recalibration task for varying a, c and fixed $b = 0 . 0 .$ We also report the fraction of samples where infinite bands were reported $n _ { \infty } .$ . The mean bandwidth is taken over finite bands only. The gray font is chosen for better readability.
<table><tr><td colspan="2"></td><td colspan="4">c = −0.5</td><td colspan="3"></td><td colspan="4">c = 0.0</td><td colspan="4"> $c = 0 . 5$ </td></tr><tr><td colspan="2"></td><td>T:</td><td>1</td><td>5</td><td>10 15</td><td>20</td><td>1</td><td>5</td><td>10</td><td>15</td><td>20</td><td>1</td><td>5</td><td>10</td><td>15</td><td>20</td></tr><tr><td colspan="2">Supr.</td><td>Cov. Bw.</td><td>.91 .0125</td><td>.92 .0108</td><td>.92 .0089</td><td>.92 .91 .0073 .0063</td><td>.91 .0132</td><td></td><td>.91 .0133</td><td>.91 .91 .0136 .0144</td><td>.89 .0157</td><td>.91 .0141</td><td>.90 .0172</td><td>.90 .0249</td><td>.89 .0417</td><td>.88 .0798</td></tr><tr><td colspan="2">LSCI</td><td>Cov. Bw.</td><td>.96 .0120</td><td>.97</td><td>.93 .90 .0048</td><td>.88</td><td>.96 .0124</td><td>.95 .0126</td><td>.93 .0098</td><td>.89</td><td>.87</td><td>.97</td><td>.97</td><td>.98</td><td>.97</td><td>.96</td></tr><tr><td colspan="2">000 DtACI</td><td>Cov.</td><td>.91</td><td>.0073 .97</td><td>.0049 .96 .97</td><td>.0051 .95</td><td>.91</td><td>.94</td><td>.94</td><td>.0110 .95</td><td>.0145 .91</td><td>.0131 .91</td><td>.0187 .89</td><td>.0235 .91</td><td>.0495 .91</td><td>.0972 .90</td></tr><tr><td colspan="2">9. Ⅱ WCP</td><td>Bw. Cov.</td><td>.0125 .90</td><td>.0116 .90</td><td>.0091 .0076 .90 .90</td><td>.0065 .89</td><td>.0132 .90</td><td>.0127 .90</td><td>.0128 .90</td><td>.0135 .89</td><td>.0149 .88</td><td>.0141 .90</td><td>.0144 .89</td><td>.0204 .89</td><td>.0336 .88</td><td>.0634 .88</td></tr><tr><td rowspan="3">8 Learned WCP</td><td rowspan="2"></td><td>Bw. .0116</td><td>.0099</td><td>.0083</td><td>.0069</td><td>.0061</td><td>.0122</td><td>.0124</td><td>.0129</td><td></td><td></td><td>.0132.0164</td><td>.0241</td><td></td><td></td></tr><tr><td> $n _ { \infty }$  0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>.0137 0%</td><td>.0152</td><td></td><td></td><td>.0394</td><td>.0759</td></tr><tr><td colspan="2">Cov.</td><td>.91</td><td>.91</td><td>.91 .93</td><td>.97</td><td>.91</td><td>.92</td><td>.91</td><td>0% .93 .95</td><td>0% .91</td><td>0% .92</td><td>0% .91</td><td>0% .90</td><td>0% .87</td></tr><tr><td colspan="2" rowspan="2">Gauss (Ours) Supr.</td><td>Bw.  $n _ { \infty }$ </td><td>.0125</td><td>.0103</td><td>.0086 .0078</td><td>.0078</td><td>.0133</td><td>.0135</td><td>.0135</td><td>.0153 .0192</td><td>.0143</td><td>.0188</td><td>.0276</td><td>.0437</td><td>.0681</td></tr><tr><td>0%</td><td>0%</td><td>0% .92 .92</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0% 0%</td><td>0%</td><td>0%</td><td>1%</td><td>2%</td></tr><tr><td colspan="2">LSCI</td><td>Bw. .0125 Cov. .96</td><td>.0109 .95</td><td>.0092 .89</td><td>.88 .0087 .82</td><td>.78 .91 .0116 .81</td><td>.0132 .0136 .96 .93</td><td>.91 .0144 .85</td><td>.91 .83 .0172 .76</td><td>.72 .0300 .76</td><td>.91 .0142 .96</td><td>.90 .0173</td><td>.89 .0257</td><td>.88 .84 .0460 .1006</td></tr><tr><td colspan="2">07</td><td>Bw. .0121 Cov. .91</td><td>.0068 .97</td><td>.0062</td><td>.0079</td><td>.0121 .0125</td><td>.0097 .94</td><td>.0111 .92</td><td>.0164</td><td>.0316</td><td>.95 .0132 .0156</td><td>.91 .0250 </td><td>.86 .0567</td><td>.83 .1235</td></tr><tr><td colspan="2">DtACI 0</td><td>Bw. .0125 Cov. .90</td><td>.0117</td><td>.96 .0093</td><td>.89 .0086</td><td>.64 .91 .0107 .0132</td><td>.0128</td><td>.0133</td><td>.73 .0159</td><td>.48 .91 .0269 .0142</td><td>.89 .0144</td><td>.89 .0210</td><td>.85 .0382</td><td>.66 .0876</td></tr><tr><td colspan="2">∥WCP a Learned</td><td>Bw. .0116</td><td>.90 .0101</td><td>.90 .0086</td><td>.86 .77 .0085</td><td>.90 .0116 .0123</td><td>.90 .0127</td><td>.89 .0137</td><td>.81 .0168</td><td>.71 .90 .0298 .0132</td><td>.89 .0164</td><td>.89 .0248</td><td>.87 .0449</td><td>.84 .0991</td></tr><tr><td colspan="2">WCP Gauss (Ours)</td><td>n∞ 0% Cov. .91 Bw. .0126</td><td>0% .91 .0105 .0094</td><td>0% .92</td><td>0% .97</td><td>0% 0% .96 .92</td><td>0% .91</td><td>0% .92</td><td>0% .95</td><td>0% 0% .95 .92</td><td>0% .92</td><td>0% .91</td><td>0% .88</td><td>0% .89</td></tr><tr><td colspan="2">Supr.</td><td> $n _ { \infty }$  0% Cov. .91</td><td>0% .92</td><td>0% .89</td><td>0% .70</td><td>.0140 .0135 0% 0% .66 .91</td><td>0% .91</td><td>0% .86</td><td>.0153 .0220 1% .63</td><td>.0368 2% 0% .59 .91</td><td>.0144 .0189 0% .90</td><td>.0279 2% .87</td><td>.0447 5%</td><td>.1084 6%</td></tr><tr><td colspan="2">LSCI</td><td>Bw. Cov.</td><td>.0125 .0111 .95 .92</td><td>.0100 .76</td><td>.0147 .75</td><td>.0307 .77</td><td>.0133 .0137 .96 .87</td><td>.0156 .76</td><td>.0300 .67</td><td>.0796 .0143 .69 .96</td><td>.0175 .89</td><td>.0275 .80</td><td>.69 .0669 .74</td><td>.59 .2207 .66</td></tr><tr><td colspan="2">DtACI</td><td>Bw. .0121 Cov. .91</td><td>.0074 .97</td><td>.0085 .91</td><td>.0154 .48</td><td>.0342 .43</td><td>.0126 .0097 .91 .94</td><td>.0145 .80</td><td>.0314 .35</td><td>.0882 .33 .91</td><td>.0133 .0148 .88</td><td>.0294 .70</td><td>.0792 .39</td><td>.2418 .32</td></tr><tr><td colspan="2">001 II</td><td>Bw. .0125 Cov. .90</td><td>.0118 .90</td><td>.0099 .86</td><td>.0133 .70</td><td>.0272 .0133 .67 .90</td><td>.0129 .89</td><td>.0144 .84</td><td>.0266 .63</td><td>.0707 .59 .90</td><td>.0143 .0145 .89</td><td>.0229 .86</td><td>.0591 .68</td><td>.1976 .57</td></tr><tr><td colspan="2">WCP a Learned</td><td>Bw. .0117</td><td>.0103</td><td>.0095</td><td>.0147</td><td>.0308 0% 0%</td><td>.0124 .0129 0%</td><td>.0152 0%</td><td>.0299</td><td>.0797</td><td>.0132 .0169</td><td>.0270</td><td>.0663</td><td>.2184</td></tr><tr><td colspan="2"></td><td>n∞ 0% Cov. .92</td><td>0% .91</td><td>0% .97</td><td>0% .97</td><td>.93</td><td>.92 .91</td><td>.94</td><td>0% .95</td><td>0% .95</td><td>0% 0% .92 .92</td><td>0% .90</td><td>0%</td><td>0%</td></tr><tr><td colspan="2">WCP Gauss (Ours)</td><td>Bw. .0128</td><td>.0108</td><td>.0132</td><td>.0183</td><td>.0378</td><td>.0135</td><td>.0137 .0192</td><td>.0376</td><td>.1030</td><td>.0145 .0190</td><td>.0301</td><td>.89 .0770</td><td>.91 .2767</td></tr></table>

![](images/d1353c2c24d9be4b282a6eadf0e2a1445bc1eb4226981c0a9b67090bc5f63559.jpg)

![](images/adb103acb3f70f2a2d516a0f0ef268cef70f1ac34f6ece796bd453493a4a76d0.jpg)

![](images/76e9365fa4657f86c7bc8b71a9363eda77dff624f86ae51dec8eff77c51cd798.jpg)

![](images/97e31063ed330d1847adb4be655af982e3fbfd8fe7636facef9da3bbb8678bd2.jpg)

![](images/b919266224a30ae6b28805f3403c65b680f08d275529fdffacf675c693e6b64d.jpg)

![](images/5ca0632c1559ca9d2189bcb694948fbc7a1cb56eb45b87e4783779fe4980a160.jpg)

![](images/2c4d15716812ddee0a198977d34ebed1dad0f48dfa0ea475661dd07b9c0d5d1e.jpg)

![](images/5c4a96a9e0b34af2b129e463b771250e343c0a4cd5a6641ae5f5c35e6aac4fe7.jpg)

![](images/ab807d8eee34d8f82fb350abfffd219665d046e5dbb4c6c1a547ff127a5e95e9.jpg)

![](images/fc0659242b2d7b196c6923325b5eb2d64ea4e0f321ea67825fd70a0fc16c94e6.jpg)  
Figure H.2: Ablation study for method stability. We use data from the synthetic task (no recalibration) with parameters $a = - 0 . 0 0 7 5 , b = 0 , c = 0$ . From top to bottom, the rows correspond to 5000, 4000, 3000, 2000, and 1000 calibration samples. Each plot reports coverage (left) and the fraction of infinite conformal bands (right) over time, where color corresponds to grid granularity. We see that higher granularity leads to earlier onset of infinite bands, which aligns with Proposition 4.4. When the number of samples is insufficient relative to the dimensionality, covariance estimation degrades and undercoverage can occur.

![](images/c0aaf3621689ef16f5059118310c42cf16027a6efb930ae1c6b5137ea415f772.jpg)  
Figure H.3: TV distance of Theorem 4.5 at $| t _ { 1 } - t _ { 2 } | = 0 . 5$ over the grid granularity. For the constant drift v = 1 under white noise and the covariance of the granularity experiment in Figure 3a, $\mathbf { 1 } ^ { \top } \pmb { \Sigma } _ { 0 } ^ { - 1 } \mathbf { 1 } = p$ and both curves coincide. For the smooth Gaussian random field $\mathcal { N } ( 0 , 6 2 5 ( - \partial _ { x } ^ { 2 } +$ $2 5 I ) ^ { - 2 } )$ on (0, 2π), evaluated on the grid, the TV distance levels off below 1 for the constant drift, but tends to 1 for a step drift $( v _ { j } = 1$ on the left half of the domain, 0 otherwise).

![](images/72298b37c6a4681e298a3f716d967e52cc59d9ebf532b34535c68ebc5e7d696b.jpg)

![](images/c7fea74ffa4f51d89892ec885de272ce12ce55a8c5e683b257f3573732c27b82.jpg)

![](images/accad7d0bd68ed8a116381e1d3f678f5a3c162e1aa1708294efbaa35d548209b.jpg)

Figure I.1: Exemplary frames from the real-world pulsed-thermography dataset. The left image shows a raw data frame from the beginning of the cooling period. The middle image shows how we cropped the frames to remove the background. The right image shows how the data looks after 100 time steps.  
![](images/ad0b3c3ace776db06144c93cb42b0526c1e3b8f3b27faa80e9f3c422e32a6345.jpg)

![](images/ddf158bdc8af42eb55b53e4ee86b821e097609681c80115174f2145fabc1ac29.jpg)  
Figure I.2: Coverages for the pulsed-thermography dataset for the recalibration task (left) and the general task (right). The 90% coverage target and, for WCP Gauss (resp. Learned), the fraction of samples with infinite bandwidth $n _ { \infty }$ are marked as a black dashed line and a green (resp. pink) dashed line, respectively. We see that all methods achieve target coverage at first, but only WCP Gauss can preserve coverage over time.

![](images/74a8932acca71c4e8ee450859392719148cf869cfcd100335c5ee5a917768250.jpg)  
Figure I.3: Coverages for the Lake Maggiore nighttime cooling dataset for the recalibration task (left) and the general task (right). The 90% target is marked by a black dashed line. For WCP Gauss (resp. Learned), the fraction of samples with infinite bandwidth $n _ { \infty }$ is shown as a green (resp. pink) dotted line. WCP Gauss and LSCI maintain the target coverage on both tasks. Deviations at single time steps are statistically expected given the test size of 14.

![](images/76dee277d2bff545d265a258f4c80031ca1278865591fec45d8ed9d7861757ca.jpg)

![](images/2c2dcd6b03e0776de296712a25758e54cebc00af495bea277e6f8ac32d1d840e.jpg)

![](images/9e96ad9910a884629733d86d76467112ced854f903f6688c6045e965eb4d5277.jpg)

![](images/22f4fd6044181083e1b8f8490eab9dae3dc9c9f3646e3a37a06cf6f0878df572.jpg)  
Figure I.4: (Top) Coverages for the FRAP dataset for the recalibration task (left) and the general task (right). The 90% target is marked by a black dashed line. For WCP Gauss (resp. Learned), the fraction of samples with infinite bandwidth $n _ { \infty }$ is shown as a green (resp. pink) dotted line. (Bottom) Recovery field snapshots from a real FRAP experiment on 32 wt% sucrose solution, showing the normalized fluorescence change $\Delta F / F _ { 0 }$ just after photobleaching (t = 0) and after partial recovery $\left( t = 5 \right)$ . The dashed circle marks the bleach center. WCP Gauss is the only method that provides non-trivial target coverage for every time step on both tasks, where WCP Learned immediately provides trivial bands.

a)  
b)  
![](images/821ac7be1126b66d96a5802a19a5001f4a753604081789f3355e246893e044bf.jpg)  
Figure J.1: Multi-seed stability. (a) Coverages for the general task of the synthetic PDE $\begin{array} { r } { ( a = - 0 . 0 0 7 5 . } \end{array}$ $b = 0 )$ for $c \in \{ - 0 . 5 , 0 , 0 . 5 \}$ , mean one standard deviation over 10 reshuffled calibration/test splits. (b) Coverages for the pulsed-thermography dataset for the recalibration task (left) and the general task (right), mean one standard deviation over 10 plate-level reshuffles with a retrained surrogate per seed. The 90% target is marked by a black dashed line. For WCP Gauss (resp. Learned), the fraction of samples with infinite bandwidth $n _ { \infty }$ is shown as a green (resp. pink) dotted line.

![](images/f4ba3bb4ad5de394442b2aef5605ac3b72d191f29f9c9c1a1dc1f3397688ac80.jpg)  
Figure J.2: Nonlinear ablation for $u _ { t } = a u _ { x x } + c u - \gamma u ^ { 3 } ( a = - 0 . 0 0 7 5 , b = 0 ) .$ . (a) Coverages, mean one standard deviation over 10 reshuffled calibration/test splits, for $c = 0 . 5$ (top row) and $c = 3$ (bottom row). The 90% target is marked by a black dashed line. For WCP Gauss (resp. Learned), the fraction of samples with infinite bandwidth $n _ { \infty }$ is shown as a green (resp. pink) dotted line. (b) Marginal distribution of the middle grid point at $t = 1$ for $c = 3$ with a Gaussian fit.

![](images/772780981c711d194020f4bf2cdf50135e528dee7823d480d7871ee4563f6dc8.jpg)  
Figure J.3: Bandwidth of WCP Gauss per test sample against the test log likelihood-ratio log $r ( x _ { \mathrm { t e s t } } )$ for the synthetic PDE $( a = - 0 . 0 0 7 5 , b = 0 , c = 0 . 5 ,$ , general task) at the first time steps with infinite bands. Infinite bands are marked by triangles at the top of each panel.

![](images/e287f700f1e2f68e5ca6ade81d41a1a0662ff402b1c966073ea41ea17cda13c0.jpg)  
Figure K.1: Coverages for the recalibration task for varying a, c and fixed b = 0.5 across increasing prediction horizon. The 90% coverage target is marked as black dashed line and the fraction of samples with infinite bandwidth $n _ { \infty }$ for WCP Learned and WCP Gauss as dotted line in their respective color.

![](images/8974508a118d04792c251dde39eaa030eaa296bd204728c6498b76eb6fd73fe6.jpg)  
Figure K.2: Coverages for the recalibration task for varying a, c and fixed b = 0 across increasing prediction horizon. The 90% coverage target is marked as black dashed line and the fraction of samples with infinite bandwidth $n _ { \infty }$ for WCP Learned and WCP Gauss as dotted line in their respective color.

![](images/f0ee409f7a7e270c4b92f2580785e006a88c805d230337ad1a2724d9802c1163.jpg)  
Figure K.3: Coverages for the recalibration task for varying a, c and fixed b = 0.5 across increasing prediction horizon. The 90% coverage target is marked as black dashed line and the fraction of samples with infinite bandwidth $n _ { \infty }$ for WCP Learned and WCP Gauss as dotted line in their respective color.

Table K.3: Coverage and mean bandwidth for the recalibration task for varying a, c and fixed $b = 0 . 5 .$ We also report the fraction of samples where infinite bands were reported $n _ { \infty } .$ . The mean bandwidth is taken over finite bands only. The gray font is chosen for better readability.
<table><tr><td colspan="2"></td><td colspan="5">c = −0.5</td><td colspan="4"> $c = 0 . 0$ </td><td colspan="4"> $c = 0 . 5$ </td><td colspan="4"></td></tr><tr><td colspan="2"></td><td>T:</td><td>1</td><td>5</td><td>10</td><td>15</td><td>20</td><td>1</td><td>5</td><td>10</td><td>15</td><td>20</td><td>1</td><td></td><td>5</td><td>10</td><td>15</td><td>20</td></tr><tr><td colspan="2">Supr.</td><td>Cov.</td><td>.91</td><td>.92</td><td>.92</td><td>.91</td><td>.91</td><td>.91</td><td>.91</td><td>.91</td><td>.91</td><td>.89</td><td></td><td>.91 .0144</td><td>.89 .0172</td><td>.89 .0248</td><td>.89 .0426</td><td>.89</td></tr><tr><td colspan="2">LSCI</td><td>Bw. Cov.</td><td>.0127 .96</td><td>.0109 .95</td><td>.0090 .93</td><td>.0075 .92</td><td>.0067 .90</td><td>.0134 .97</td><td>.0135 .94</td><td>.0139 .91</td><td>.0147 .89</td><td>.0164 .86</td><td></td><td>.96 .95</td><td></td><td>.97</td><td>.97</td><td>.0809 .97</td></tr><tr><td colspan="2">00 DtACI</td><td>Bw. Cov.</td><td>.0100 .91</td><td>.0067 .97</td><td>.0053 .96</td><td>.0053 .96</td><td>.0056 .95</td><td>.0102 .91</td><td>.0096 .94</td><td>.0095 .93</td><td>.0127 .93</td><td>.0150 .89</td><td>.0108 .91</td><td>.0137 .88</td><td>.90</td><td>.0238</td><td>.0505 .91</td><td>.0986 .90</td></tr><tr><td colspan="2">9 Ⅱ WCP</td><td>Bw. Cov.</td><td>.0127 .90</td><td>.0117 .89</td><td>.0096</td><td>.0080</td><td>.0068</td><td>.0134</td><td>.0130 .89</td><td>.0133 .89</td><td>.0139 .89</td><td>.0154</td><td></td><td>.0144 .0146</td><td>.0213</td><td></td><td>.0346</td><td>.0638</td></tr><tr><td colspan="2">a Learned</td><td>Bw.</td><td>.0117</td><td>.0101</td><td>.89 .0084</td><td>.89 .0072</td><td>.89 .0064</td><td>.90 .0124</td><td>.0127</td><td>.0129</td><td>.0140</td><td>.87 .0157</td><td></td><td>.90 .0133 .0168</td><td>.89 .0238</td><td>.88</td><td>.89 .0403</td><td>.88 .0766</td></tr><tr><td rowspan="3">WCP Gauss (Ours)</td><td rowspan="2"></td><td> $n _ { \infty }$ </td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td></tr><tr><td>Cov. .91</td><td></td><td>.91</td><td>.91</td><td>.91</td><td>.97</td><td>.91</td><td>.91</td><td>.92</td><td>.93</td><td>.95</td><td>.91</td><td>.91</td><td>.91</td><td></td><td>.87</td></tr><tr><td>Bw.</td><td>.0129</td><td>.0106</td><td>.0089</td><td>.0074</td><td>.0080</td><td>.0134</td><td>.0137</td><td>.0143</td><td>.0155</td><td></td><td>.0199</td><td>.0144</td><td>.0187</td><td>.0276</td><td></td><td>.0683</td><td>.89 .0431</td></tr><tr><td colspan="2">Supr.</td><td> $n _ { \infty }$  Cov.</td><td>0% .91</td><td>0% .92</td><td>0% .91</td><td>0% .87</td><td>0% .79</td><td>0% .91</td><td>0% .91</td><td>0% .90</td><td>0% .85</td><td>0% .72</td><td>0% .91</td><td>0% .89</td><td>0% .88</td><td>.88</td><td>1%</td><td>2% .84</td></tr><tr><td colspan="2">LSCI</td><td>Bw. Cov.</td><td>.0127 .96</td><td>.0110 .94</td><td>.0093 .88</td><td>.0092 .85</td><td>.0123 .81</td><td>.0134 .96</td><td>.0137 .91</td><td>.0145 .86</td><td>.0182 .80</td><td>.0316 .75</td><td>.0144 .97</td><td>.0174 .92</td><td>.0256</td><td>.0472</td><td>.1050</td><td></td></tr><tr><td colspan="2">07 DtACI</td><td>Bw. Cov.</td><td>.0092 .91</td><td>.0070 .96</td><td>.0068 .94</td><td>.0086 .87</td><td>.0129 .67</td><td>.0098 .91</td><td>.0091 .93</td><td>.0135</td><td>.0177</td><td>.0333</td><td>.0102</td><td>.0129</td><td>.92 .0255</td><td></td><td>.87 .0580 .1262</td><td>.84</td></tr><tr><td colspan="2">0. ∥WCP</td><td>Bw. Cov.</td><td>.0127 .90</td><td>.0118 .89</td><td>.0098 .89</td><td>.0090 .86</td><td>.0112</td><td>.0134 .90</td><td>.0131 .89</td><td>.91</td><td>.74 .0138.0168</td><td>.49 .0282</td><td>.91</td><td>.87 .0144.0147</td><td>.89 .0219</td><td></td><td>.80 .67 .0396.0908</td><td></td></tr><tr><td colspan="2">a Learned</td><td>Bw. n∞</td><td>.0118 0%</td><td>.0102 0%</td><td>.0088</td><td>.0090</td><td>.78 .0122</td><td>.0125</td><td>.0128</td><td>.88</td><td>.83 .0136.0179</td><td>.71 .0314</td><td>.90 .0133</td><td>.89 .0170</td><td>.0244</td><td>.88</td><td>.87 .0456</td><td>.81 .1014</td></tr><tr><td colspan="2">WCP Gauss (Ours)</td><td>Cov. Bw.</td><td>.91 .0128</td><td>.91 .0108 .0102</td><td>0% .94</td><td>0% .97</td><td>0% .97</td><td>0% .91</td><td>0% .91</td><td>0% .93</td><td>0% .96</td><td>0% .99</td><td>0% .91</td><td>0% .91</td><td></td><td>0% .90</td><td>0% .92</td><td>0% .92</td></tr><tr><td colspan="2">Supr.</td><td> $n _ { \infty }$  Cov.</td><td>0% .91</td><td>0% .91</td><td>0% .87</td><td>0%</td><td>.0113 .0153 0%</td><td>.0134 0% .91</td><td>.0139 0%</td><td>.0163 0%</td><td>.0231 1%</td><td>.0436 2%</td><td>.0143 0%</td><td>0%</td><td>.0188 2%</td><td>.0274</td><td>.0559 5%</td><td>.1276 6%</td></tr><tr><td colspan="2">LSCI</td><td>Bw. Cov.</td><td>.0127 .95</td><td>.0112 .90</td><td>.0103 .79</td><td>.74 .0155 .77</td><td>.69 .0319 .75</td><td>.0135 .96</td><td>.90 .0138 .86</td><td>.84 .0161 .69</td><td>.64 .0313 .68</td><td>.61 .0825 .70</td><td>.91 .0144 .96</td><td>.89 .0175 .86</td><td>.85 .0277</td><td>.0699</td><td>.70 .59 .2270</td><td></td></tr><tr><td colspan="2">001 DtACI</td><td>Bw. Cov.</td><td>.0088 .91</td><td>.0071 .96</td><td>.0091 .89</td><td>.0163</td><td>.0349 .44</td><td>.0090 .91</td><td>.0094 .93</td><td>.0144 .78</td><td>.0331 .36</td><td>.0907 .34</td><td>.0093 .91</td><td>.0124</td><td>.0305</td><td>.81</td><td>.72 .0802</td><td>.67 .2482</td></tr><tr><td colspan="2">II WCP</td><td>Bw. Cov.</td><td>.0127 .90</td><td>.0118</td><td>.0104</td><td>.47 .0138</td><td>.0282 .69</td><td>.0135 .90</td><td>.0131 .89</td><td>.0150</td><td>.0277 .64</td><td>.0731 .61</td><td></td><td>.0144 .0149</td><td>.86</td><td>.69 .0241</td><td>.39 .0616</td><td>.32 .2033</td></tr><tr><td colspan="2">a Learned</td><td>Bw.</td><td>.0118</td><td>.89 .0104</td><td>.84 .0100</td><td>.73 .0155</td><td>.0319</td><td>.0125</td><td>.0132</td><td>.81 .0155</td><td>.0312</td><td>.0825</td><td></td><td>.89 .0134</td><td>.89 .0172</td><td>.84 .0271</td><td>.67 .0685</td><td>.58 .2266</td></tr><tr><td colspan="2">WCP Gauss (Ours)</td><td> $n _ { \infty }$  Cov.</td><td>0% .91</td><td>0% .91</td><td>0% .98</td><td>0% .97</td><td>0% .96</td><td>0% .91</td><td>0% .91</td><td></td><td>0% .97</td><td>0% .97</td><td>0% .99</td><td>0% .91</td><td>0% .91</td><td>0% .93</td><td>0% .95</td><td>0% .93</td></tr></table>

![](images/4fdb5f619eb4f0b2db672480c0acea77d3089397fa8d22733c51c7f44a23711e.jpg)  
Figure K.4: Mean coverages for the general task (no recalibration) for varying a, c and fixed b = 0.5 across an increasing prediction horizon. The 90% coverage target is marked as black dashed line and the fraction of samples with infinite bandwidth $n _ { \infty }$ for WCP Learned and WCP Gauss as dotted line in their respective color.

![](images/4e69afd8e506b4e9b958ab0c94811930f8c843fe88b3775fece4122516ee4d31.jpg)  
Figure K.5: Mean coverages for the general task (no recalibration) for varying a, c and fixed b = 0 across increasing prediction horizon. The 90% coverage target is marked as black dashed line and the fraction of samples with infinite bandwidth $n _ { \infty }$ for WCP Learned and WCP Gauss as dotted line in their respective color.

![](images/6c5133092350123c10b5b919c4e1e55fa77f2ead05e61b5f22a7acc7def38bf7.jpg)  
Figure K.6: Mean coverages for the general task (no recalibration) for varying $a , c$ and fixed $b = 0 . 5$ across increasing prediction horizon. The 90% coverage target is marked as black dashed line and the fraction of samples with infinite bandwidth $n _ { \infty }$ for WCP Learned and WCP Gauss as dotted line in their respective color.

Table K.4: Coverage and mean bandwidth for the general task (no recalibration) for varying a, c and fixed $b = - 0 . 5$ . We also report the fraction of samples where infinite bands were reported $n _ { \infty } .$ . The mean bandwidth is taken over finite bands only. A dash (–) marks cells where every band was infinite. The gray font is chosen for better readability.
<table><tr><td colspan="2"></td><td colspan="5">c = −0.5</td><td colspan="5">c = 0.0</td><td colspan="5"> $c = 0 . 5$ </td></tr><tr><td colspan="2"></td><td>1</td><td>5</td><td>10</td><td>15</td><td>20</td><td>1</td><td>5</td><td>10</td><td>15</td><td>20</td><td>1</td><td>5</td><td>10</td><td>15</td><td>20</td></tr><tr><td colspan="2">Supr.</td><td>Cov.</td><td>.91</td><td>.93</td><td>.95 .97</td><td>.98</td><td>.91</td><td>.91</td><td>.91</td><td>.90</td><td>.90</td><td>.91</td><td>.89</td><td>.85</td><td>.78</td><td>.69</td></tr><tr><td rowspan="2">LSCI</td><td>Bw.</td><td>.0129</td><td>.0129</td><td>.0129</td><td>.0129</td><td>.0129</td><td>.0132</td><td>.0132</td><td>.0132</td><td>.0132</td><td>.0132</td><td>.0136</td><td>.0136</td><td>.0136 .0136</td><td></td><td>.0136</td></tr><tr><td>Cov.</td><td>.96</td><td>.98</td><td>.99</td><td>.99</td><td>.99</td><td>.96</td><td>.95</td><td>.92</td><td>.87</td><td>.72</td><td>.96</td><td>.90</td><td>.72</td><td>.42</td><td>.11</td></tr><tr><td rowspan="2">0.0005 1 WCP</td><td>Bw.</td><td>.0122</td><td>.0122</td><td>.0123</td><td>.0123</td><td>.0123</td><td>.0124</td><td>.0124</td><td>.0124</td><td>.0124</td><td>.0124</td><td>.0127</td><td>.0127</td><td>.0126</td><td>.0125</td><td>.0124</td></tr><tr><td>Cov.</td><td>.90</td><td>.90</td><td>.90</td><td>.91</td><td>.91</td><td>.90</td><td>.90</td><td>.89</td><td>.89</td><td>.87</td><td>.90</td><td>.89</td><td>.87</td><td>.84</td><td>.82</td></tr><tr><td rowspan="2">∥ Learned a</td><td>Bw.</td><td>.0119</td><td>.0111</td><td>.0103</td><td>.0095</td><td>.0087</td><td>.0122</td><td>.0123</td><td>.0122</td><td>.0124</td><td>.0122</td><td>.0127</td><td>.0136</td><td>.0153</td><td>.0165</td><td>.0184</td></tr><tr><td>n∞</td><td>0%</td><td>0% .91</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td></tr><tr><td rowspan="2">WCP Gauss (Ours)</td><td>Cov.</td><td>.91</td><td>.0115</td><td>.91 .0104</td><td>.90</td><td>.88</td><td>.91</td><td>.91</td><td>.91</td><td>.91</td><td>.93</td><td>.91</td><td>.92</td><td>.93</td><td>.99</td><td>1.00</td></tr><tr><td>Bw.</td><td>.0129 0%</td><td>0%</td><td>0%</td><td>.0093 0%</td><td>.0083 0%</td><td>.0133 0%</td><td>.0133 0%</td><td>.0133 0%</td><td>.0136</td><td>.0148</td><td>.0137</td><td>.0164</td><td>.0191</td><td>.0231</td><td>.0264</td></tr><tr><td rowspan="2">Supr.</td><td>n∞ Cov.</td><td>.91</td><td>.93</td><td>.95</td><td>.97</td><td>.98</td><td>.91</td><td>.91</td><td></td><td>1%</td><td>11%</td><td>0%</td><td>0%</td><td>20%</td><td>85%</td><td>&gt;99%</td></tr><tr><td>Bw.</td><td>.0128</td><td>.0128</td><td>.0128</td><td>.0128</td><td>.0128</td><td>.0132</td><td>.0132</td><td>.91 .0132</td><td>.90 .0132</td><td>.88 .0132</td><td>.91</td><td>.89 .0137</td><td>.84 .0137</td><td>.75 .0137</td><td>.55 .0137</td></tr><tr><td rowspan="2">00075 LSCI</td><td>Cov.</td><td>.96</td><td>.98</td><td>.97</td><td>.97</td><td>.90</td><td>.96</td><td>.94</td><td>.86</td><td>.65</td><td>.27</td><td>.0137 .96</td><td>.90</td><td>.59</td><td>.17</td><td>.01</td></tr><tr><td>Bw.</td><td>.0122</td><td>.0123</td><td>.0123</td><td>.0123</td><td>.0123</td><td>.0125</td><td>.0125</td><td>.0125</td><td>.0125</td><td>.0125</td><td>.0128</td><td>.0127</td><td>.0126</td><td>.0125</td><td>.0124</td></tr><tr><td rowspan="2">WCP II Learned</td><td>Cov.</td><td>.90</td><td>.90</td><td>.90</td><td>.90</td><td>.89</td><td>.90</td><td>.90</td><td>.88</td><td>.88</td><td>.85</td><td>.90</td><td>.89</td><td>.87</td><td>.84</td><td>.79</td></tr><tr><td>Bw.</td><td>.0119</td><td>.0111</td><td>.0104</td><td>.0095</td><td>.0087</td><td>.0123</td><td>.0124</td><td>.0123</td><td>.0125</td><td>.0122</td><td>.0127</td><td>.0137</td><td>.0154</td><td>.0165</td><td>.0180</td></tr><tr><td rowspan="2">a WCP</td><td>n∞</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td></tr><tr><td>Cov.</td><td>.91</td><td>.91</td><td>.90</td><td>.89</td><td>.88</td><td>.91</td><td>.91</td><td>.91</td><td>.94</td><td>1.00</td><td>.91</td><td>.92</td><td>.97</td><td>1.00</td><td>1.00</td></tr><tr><td rowspan="2">Gauss (Ours) Supr.</td><td>Bw.</td><td>.0129</td><td>.0115</td><td>.0104</td><td>.0093</td><td>.0086</td><td>.0133</td><td>.0134</td><td>.0136</td><td>.0157</td><td>.0201</td><td>.0138</td><td>.0165</td><td>.0204</td><td>.0252</td><td></td></tr><tr><td>n∞ Cov.</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>25%</td><td>94%</td><td>0%</td><td>0%</td><td>45%</td><td>99%</td><td>100%</td></tr><tr><td colspan="2"></td><td>.91 Bw. .0129</td><td>.93 .0129</td><td>.95 .0129</td><td>.96 .0129</td><td>.98 .0129</td><td>.91 .0132</td><td>.91 .0132</td><td>.90 .0132</td><td>.88 .0132</td><td>.73 .0132</td><td>.91 .0137</td><td>.88 .0137</td><td>.83 .0137</td><td>.67 .0137</td><td>.06 .0137</td></tr><tr><td colspan="2">LSCI</td><td>Cov. .95</td><td>.96</td><td>.93</td><td>.79</td><td>.41</td><td>.95</td><td>.93</td><td>.75</td><td>.26</td><td>.01</td><td>.96</td><td>.86</td><td>.39</td><td>.02</td><td>.00</td></tr><tr><td colspan="2">0001 WCP</td><td>Bw. .0122</td><td>.0123</td><td>.0123</td><td>.0123</td><td>.0123</td><td>.0125</td><td>.0125</td><td>.0125</td><td>.0125</td><td>.0125</td><td>.0128</td><td>.0127 .88</td><td>.0126 .86</td><td>.0126 .82</td><td>.0124 .47</td></tr><tr><td rowspan="2">Learned ⅡI a n∞ Cov.</td><td>Cov. Bw.</td><td>.90 .0119</td><td>.90 .0111</td><td>.89 .0103</td><td>.88 .0095</td><td>.80 .0087</td><td>.90 .0123</td><td>.89 .0123</td><td>.88 .0124 .0124</td><td>.85</td><td>.65 .0125</td><td>.90 .0127</td></table>

Table K.5: Coverage and mean bandwidth for the general task (no recalibration) for varying a, c and fixed $b = 0 . 0$ . We also report the fraction of samples where infinite bands were reported $n _ { \infty } .$ . The mean bandwidth is taken over finite bands only. A dash (–) marks cells where every band was infinite. The gray font is chosen for better readability.
<table><tr><td colspan="2"></td><td colspan="5">c = −0.5</td><td colspan="5">c = 0.0</td><td colspan="5"> $c = 0 . 5$ </td></tr><tr><td colspan="2"></td><td>1</td><td>5</td><td>10</td><td>15</td><td>20</td><td>1</td><td>5</td><td>10</td><td>15</td><td>20</td><td>1</td><td>5</td><td>10</td><td>15</td><td>20</td></tr><tr><td colspan="2">Supr.</td><td>Cov.</td><td>.91</td><td>.93</td><td>.95 .97</td><td>.98</td><td>.91</td><td>.91</td><td>.91</td><td>.91</td><td>.90</td><td>.91</td><td>.89</td><td>.83</td><td>.79</td><td>.70</td></tr><tr><td rowspan="2">LSCI</td><td>Bw.</td><td>.0128</td><td>.0128</td><td>.0128</td><td>.0128</td><td>.0128</td><td>.0132</td><td>.0132</td><td>.0132</td><td>.0132</td><td>.0132</td><td>.0136</td><td>.0136</td><td>.0136 .0136</td><td></td><td>.0136</td></tr><tr><td>Cov.</td><td>.96</td><td>.98</td><td>.98</td><td>.99</td><td>1.00</td><td>.95</td><td>.94</td><td>.91</td><td>.86</td><td>.75</td><td>.96</td><td>.90</td><td>.73</td><td>.45</td><td>.13</td></tr><tr><td rowspan="2">0.0005 1 WCP</td><td>Bw.</td><td>.0121</td><td>.0121</td><td>.0122</td><td>.0122</td><td>.0122</td><td>.0123</td><td>.0123</td><td>.0123</td><td>.0123</td><td>.0123</td><td>.0127</td><td>.0126</td><td>.0125</td><td>.0124</td><td>.0123</td></tr><tr><td>Cov.</td><td>.90</td><td>.89</td><td>.90</td><td>.91</td><td>.91</td><td>.90</td><td>.90</td><td>.89</td><td>.89</td><td>.88</td><td>.90</td><td>.89</td><td>.88</td><td>.85</td><td>.82</td></tr><tr><td rowspan="2">∥ Learned a</td><td>Bw.</td><td>.0118</td><td>.0110</td><td>.0102</td><td>.0095</td><td>.0087</td><td>.0122</td><td>.0122</td><td>.0123</td><td>.0123</td><td>.0122</td><td>.0127</td><td>.0138</td><td>.0151</td><td>.0167</td><td>.0178</td></tr><tr><td>n∞</td><td>0%</td><td>0% .90</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td></tr><tr><td rowspan="2">WCP Gauss (Ours)</td><td>Cov.</td><td>.92</td><td>.0112</td><td>.91 .0103</td><td>.90</td><td>.90</td><td>.91</td><td>.91</td><td>.91</td><td>.92</td><td>.93</td><td>.92</td><td>.92</td><td>.94</td><td>.99</td><td>1.00</td></tr><tr><td>Bw.</td><td>.0129 0%</td><td>0%</td><td>0%</td><td>.0091 0%</td><td>.0083 0%</td><td>.0133 0%</td><td>.0133 0%</td><td>.0134 0%</td><td>.0138</td><td>.0151</td><td>.0138</td><td>.0164</td><td>.0196</td><td>.0250</td><td>.0275</td></tr><tr><td rowspan="2">Supr.</td><td>n∞ Cov.</td><td>.91</td><td>.93</td><td>.95</td><td>.97</td><td>.98</td><td>.91</td><td>.91</td><td></td><td>0%</td><td>11%</td><td>0%</td><td>0%</td><td>20%</td><td>85%</td><td>&gt;99%</td></tr><tr><td>Bw.</td><td>.0128</td><td>.0128</td><td>.0128</td><td>.0128</td><td>.0128</td><td>.0132</td><td>.0132</td><td>.91 .0132</td><td>.90 .0132</td><td>.88</td><td>.91</td><td>.89 .0136</td><td>.83 .0136</td><td>.76 .0136</td><td>.56</td></tr><tr><td rowspan="2">00075 LSCI</td><td>Cov.</td><td>.96</td><td>.97</td><td>.97</td><td>.95</td><td>.92</td><td>.96</td><td>.93</td><td>.86</td><td>.63</td><td>.0132 .27</td><td>.0136 .96</td><td>.88</td><td>.58</td><td>.16</td><td>.0136 .01</td></tr><tr><td>Bw.</td><td>.0122</td><td>.0122</td><td>.0122</td><td>.0123</td><td>.0123</td><td>.0124</td><td>.0124</td><td>.0124</td><td>.0124</td><td>.0124</td><td>.0127</td><td>.0126</td><td>.0125</td><td>.0125</td><td>.0123</td></tr><tr><td rowspan="2">WCP II Learned</td><td>Cov.</td><td>.90</td><td>.90</td><td>.90</td><td>.91</td><td>.91</td><td>.90</td><td>.90</td><td>.88</td><td>.88</td><td>.86</td><td>.90</td><td>.89</td><td>.88</td><td>.83</td><td>.80</td></tr><tr><td>Bw.</td><td>.0118</td><td>.0111</td><td>.0102</td><td>.0095</td><td>.0087</td><td>.0122</td><td>.0122</td><td>.0122</td><td>.0123</td><td>.0123</td><td>.0127</td><td>.0139</td><td>.0153</td><td>.0164</td><td>.0184</td></tr><tr><td rowspan="2">a WCP</td><td>n∞</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td></tr><tr><td>Cov.</td><td>.92</td><td>.90</td><td>.91</td><td>.89</td><td>.89</td><td>.91</td><td>.91</td><td>.92</td><td>.95</td><td>1.00</td><td>.92</td><td>.92</td><td>.97</td><td>1.00</td><td>1.00</td></tr><tr><td rowspan="2">Gauss (Ours)</td><td>Bw.</td><td>.0129</td><td>.0113</td><td>.0103</td><td>.0092</td><td>.0085</td><td>.0133</td><td>.0133</td><td>.0137</td><td>.0160</td><td>.0188</td><td>.0138</td><td>.0165</td><td>.0207</td><td>.0240</td><td></td></tr><tr><td>n∞ Cov.</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>25%</td><td>94%</td><td>0%</td><td>0%</td><td>45%</td><td>99%</td><td>100%</td></tr><tr><td colspan="2">Supr.</td><td>.91 Bw. .0128</td><td>.93 .0128</td><td>.95 .0128</td><td>.96 .0128</td><td>.97 .0128</td><td>.91 .0132</td><td>.91 .0132</td><td>.90 .0132</td><td>.88 .0132</td><td>.76 .0132</td><td>.91 .0136</td><td>.89 .0136</td><td>.82 .0136</td><td>.68</td><td>.09 .0136</td></tr><tr><td colspan="2">LSCI</td><td>Cov. .96</td><td>.96</td><td>.92</td><td>.77</td><td>.42</td><td>.96</td><td>.92</td><td>.71</td><td>.26</td><td>.01</td><td>.96</td><td>.85</td><td>.40</td><td>.0136 .03</td><td>.00</td></tr><tr><td colspan="2">0001 WCP</td><td>Bw. .0122</td><td>.0122</td><td>.0123</td><td>.0123</td><td>.0123</td><td>.0124</td><td>.0124</td><td>.0124</td><td>.0124</td><td>.0124</td><td>.0127</td><td>.0127 .89</td><td>.0126 .87 .82</td><td>.0125</td><td>.0124 .52</td></tr><tr><td rowspan="2">Learned ⅡI a n∞ Cov.</td><td>Cov. Bw.</td><td>.90 .0119</td><td>.89 .0111</td><td>.89 .0102</td><td>.90 .0095</td><td>.84 .0088</td><td>.90 .0122</td><td>.89 .0122</td><td>.88 .0123</td><td>.86 .0123</td><td>.71 .0125</td><td>.90 .0127 .0139</td></table>

Table K.6: Coverage and mean bandwidth for the general task (no recalibration) for varying a, c and fixed $b = 0 . 5$ . We also report the fraction of samples where infinite bands were reported $n _ { \infty } .$ . The mean bandwidth is taken over finite bands only. A dash (–) marks cells where every band was infinite. The gray font is chosen for better readability.
<table><tr><td colspan="2"></td><td colspan="5">c = −0.5</td><td colspan="5">c = 0.0</td><td colspan="5"> $c = 0 . 5$ </td></tr><tr><td colspan="2"></td><td>1</td><td>5</td><td>10</td><td>15</td><td>20</td><td>1</td><td>5</td><td>10</td><td>15</td><td>20</td><td>1</td><td>5</td><td>10</td><td>15</td><td>20</td></tr><tr><td colspan="2">Supr.</td><td>Cov.</td><td>.91</td><td>.93</td><td>.95 .97</td><td>.98</td><td>.91</td><td>.91</td><td>.91</td><td>.91</td><td>.90</td><td>.91</td><td>.89</td><td>.84</td><td>.77</td><td>.68</td></tr><tr><td rowspan="2">LSCI</td><td>Bw.</td><td>.0129</td><td>.0129</td><td>.0129</td><td>.0129</td><td>.0129</td><td>.0133</td><td>.0133</td><td>.0133</td><td>.0133</td><td>.0133</td><td>.0137</td><td>.0137</td><td>.0137 .0137</td><td></td><td>.0137</td></tr><tr><td>Cov.</td><td>.96</td><td>.98</td><td>.98</td><td>1.00 1.00</td><td></td><td>.97</td><td>.95</td><td>.91</td><td>.84</td><td>.74</td><td>.96</td><td>.90</td><td>.73</td><td>.41</td><td>.15</td></tr><tr><td rowspan="2">0.0005 WCP</td><td>Bw.</td><td>.0122</td><td>.0122</td><td>.0122</td><td>.0123</td><td>.0123</td><td>.0124</td><td>.0124</td><td>.0124</td><td>.0124</td><td>.0124</td><td>.0127</td><td>.0126</td><td>.0125</td><td>.0124</td><td>.0123</td></tr><tr><td>Cov.</td><td>.90</td><td>.90</td><td>.90</td><td>.90</td><td>.91</td><td>.90</td><td>.89</td><td>.88</td><td>.88</td><td>.87</td><td>.90</td><td>.89</td><td>.87</td><td>.85</td><td>.82</td></tr><tr><td rowspan="2">∥ Learned a</td><td>Bw.</td><td>.0119</td><td>.0109</td><td>.0103</td><td>.0095</td><td>.0087</td><td>.0123</td><td>.0122</td><td>.0122</td><td>.0123</td><td>.0122</td><td>.0127</td><td>.0140</td><td>.0149</td><td>.0167</td><td>.0182</td></tr><tr><td>n∞</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td></tr><tr><td rowspan="2">WCP Gauss (Ours)</td><td>Cov.</td><td>.91</td><td>.91</td><td>.90</td><td>.89</td><td>.89</td><td>.91</td><td>.91</td><td>.91</td><td>.91</td><td>.93</td><td>.91</td><td>.92</td><td>.94</td><td>.99</td><td>1.00</td></tr><tr><td>Bw.</td><td>.0130</td><td>.0113 0%</td><td>.0102 0%</td><td>.0092</td><td>.0084</td><td>.0134</td><td>.0134</td><td>.0135</td><td>.0138</td><td>.0149</td><td>.0138</td><td>.0158</td><td>.0196</td><td>.0232</td><td>.0241</td></tr><tr><td rowspan="2">Supr.</td><td>n∞ Cov.</td><td>0% .91</td><td>.93</td><td>.95</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>11%</td><td>0%</td><td>0%</td><td>20%</td><td>86%</td><td>&gt;99%</td></tr><tr><td>Bw.</td><td>.0129</td><td>.0129</td><td>.0129</td><td>.97 .0129</td><td>.98 .0129</td><td>.91 .0133</td><td>.91 .0133</td><td>.90 .0133</td><td>.89 .0133</td><td>.88</td><td>.91</td><td>.88</td><td>.83</td><td>.73</td><td>.52</td></tr><tr><td rowspan="2">LSCI</td><td>Cov.</td><td>.96</td><td>.97</td><td>.97</td><td>.96</td><td>.91</td><td>.97</td><td>.94</td><td>.86</td><td>.62</td><td>.0133 .29</td><td>.0137 .96</td><td>.0137 .88</td><td>.0137 .57</td><td>.0137</td><td>.0137 .01</td></tr><tr><td>Bw.</td><td>.0122</td><td>.0123</td><td>.0123</td><td>.0123</td><td>.0123</td><td>.0124</td><td>.0125</td><td>.0125</td><td>.0125</td><td>.0125</td><td>.0127</td><td>.0127</td><td>.0126</td><td>.15 .0125</td><td>.0124</td></tr><tr><td rowspan="2">00075 WCP II Learned</td><td>Cov.</td><td>.90</td><td>.90</td><td>.90</td><td>.89</td><td>.89</td><td>.89</td><td>.90</td><td>.88</td><td>.87</td><td>.85</td><td>.90</td><td>.89</td><td>.86</td><td>.84</td><td>.78</td></tr><tr><td>Bw.</td><td>.0119</td><td>.0110</td><td>.0102</td><td>.0095</td><td>.0087</td><td>.0123</td><td>.0123</td><td>.0122</td><td>.0121</td><td>.0123</td><td>.0127</td><td>.0140</td><td>.0151</td><td>.0170</td><td>.0179</td></tr><tr><td rowspan="2">a WCP</td><td>n∞</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td></tr><tr><td>Cov.</td><td>.91</td><td>.91</td><td>.90</td><td>.89</td><td>.88</td><td>.91</td><td>.91</td><td>.91</td><td>.95</td><td>1.00</td><td>.91</td><td>.92</td><td>.97</td><td>1.00</td><td>1.00</td></tr><tr><td rowspan="2">Gauss (Ours)</td><td>Bw.</td><td>.0129</td><td>.0113</td><td>.0103</td><td>.0092</td><td>.0086</td><td>.0134</td><td>.0134</td><td>.0137</td><td>.0156</td><td>.0203</td><td>.0139</td><td>.0160</td><td>.0208</td><td>.0244</td><td></td></tr><tr><td>n∞</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>25%</td><td>94%</td><td>0%</td><td>0%</td><td>44%</td><td>99%</td><td>100%</td></tr><tr><td rowspan="2">Supr. LSCI</td><td>Cov. Bw.</td><td>.91</td><td>.93</td><td>.95</td><td>.96</td><td>.97</td><td>.91</td><td>.91</td><td>.90</td><td>.88</td><td>.75</td><td>.91</td><td>.88</td><td>.82</td><td>.64</td><td>.07</td></tr><tr><td></td><td>.0129</td><td>.0129</td><td>.0129</td><td>.0129</td><td>.0129</td><td>.0133</td><td>.0133</td><td>.0133</td><td>.0133</td><td>.0133</td><td>.0138</td><td>.0138</td><td>.0138</td><td>.0138</td><td>.0138</td></tr><tr><td rowspan="2">0001 WCP</td><td>Cov. Bw.</td><td>.96</td><td>.96</td><td>.94</td><td>.78</td><td>.42</td><td>.97</td><td>.92</td><td>.71</td><td>.26</td><td>.01</td><td>.97</td><td>.84</td><td>.38</td><td>.02</td><td>.00</td></tr><tr><td>Cov.</td><td>.0123 .90</td><td>.0123 .90</td><td>.0123 .89</td><td>.0123 .88</td><td>.0124 .81</td><td>.0125 .90</td><td>.0125 .89</td><td>.0125 .87</td><td>.0125 .82</td><td>.0125 .64</td><td>.0128 .90</td><td>.0127 .89</td><td>.0126 .85</td><td>.0125 .78</td><td>.0124 .39</td></tr></table>

![](images/996c4ec62c02228935d93fc98aae4cbbe023d9d88bd8ac7f15aaaae0b4dbcce4.jpg)  
Figure K.7: Coverages for the general task for varying a, c and fixed b = 0 with Laplace initial distribution across increasing prediction horizon. The 90% coverage target is marked as black dashed line and the fraction of samples with infinite bandwidth $n _ { \infty }$ for WCP Learned and WCP Gauss as dotted line in their respective color.

![](images/d59efda8ee376acf9dbb065b4307f11bfc4d32bde67489604daa95feaf7f8f1f.jpg)  
Figure K.8: Coverages for the general task for varying a, c and fixed b = 0 with Logistic initial distribution across increasing prediction horizon. The 90% coverage target is marked as black dashed line and the fraction of samples with infinite bandwidth $n _ { \infty }$ for WCP Learned and WCP Gauss as dotted line in their respective color.

![](images/0970e1a9fe80939a8ea14a1ee565cd909ebcc1b2f8a0a200146d9c65a7da4ddf.jpg)  
Figure K.9: Coverages of WCP with Cauchy likelihood-ratio weights for the general task for varying $a , c$ and fixed $b = 0$ with heavy-tailed elliptical Cauchy initial distribution, mean one standard deviation over 10 reshuffled calibration/test splits. The 90% coverage target is marked as black dashed line and the fraction of samples with infinite bandwidth $n _ { \infty }$ as dotted line. The Cauchy likelihood ratios stay bounded, so no infinite bands occur. The heavy tails increase the variance of the coverage across splits, and for $c = - 0 . 5$ the mean coverage falls slightly below the target at late time steps.