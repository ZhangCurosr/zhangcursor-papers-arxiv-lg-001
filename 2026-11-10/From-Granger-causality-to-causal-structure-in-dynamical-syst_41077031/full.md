# From Granger causality to causal structure in dynamical systems

V´ıctor Elvira, Senior Member, IEEE

The idea of testing whether the past of one signal improves the prediction of another can be traced back to Wiener. It was Granger who formalized the concept [1]. Granger causality has become one of the most widely used tools to infer directed interactions between the components of a system. The name however can be misleading: it detects predictive relations among the measured signals, not causal relations among the mechanisms that generate them. This article starts by reviewing recent advances in observation-space Granger analysis. It then follows a path to directed structure in latent dynamical models, and asks under which conditions that structure can be read causally.

## INTRODUCTION

Multivariate time series arise in a wide range of problems in science and engineering. A recurring question is which components of a system influence the future of others: which brain region drives another, how respiration and arterial pressure influence heart rate [2], or which mode of atmospheric variability precedes another across seasons [3]. In many applications, only observational data are available (i.e., interventions are not possible), which makes it difficult to draw strong conclusions about the causal mechanisms of the system. Graphical approaches instead infer directed or undirected relations among its components. In signal processing, this line of work is increasingly connected with graph signal processing and topology inference [4], [5], as well as with the broader causal-inference literature [6]. This tutorial takes a complementary route: it starts from Granger causality and moves the graph from the measured signals to the latent dynamics that generate them.

Granger causality (GC) [1] has been one of the most widely used frameworks for this task for more than half a century. Component $j$ Granger-causes component i if the past of $j$ improves the prediction of i when conditioning on the past of all other components [7]. Under a vector autoregressive (VAR) model, this corresponds to testing blocks of autoregressive coefficients, whose sparsity pattern defines a directed graph over the observed variables. VAR-based Granger analysis is computationally tractable and widely used across econometrics, neuroscience, and signal processing [7]–[9]. Significant work has been done recently on high-dimensional VAR models [10], [11].

![](images/8d015b3e48c2d28e2ec6fca49c0eadb6aef0b2807939e7add8b393d4b96406db.jpg)  
Fig. 1. The causal ladder of this tutorial. From left to right: (1) the Granger graph of the observed process; (2) the directed graph of the latent dynamics; (3) the latent graph once it is identifiable; and (4) its causal reading, obtained by interpreting the transition equations as structural mechanisms. The labels below the arrows indicate what is needed to move up one level.

Observation-space Granger analysis becomes unreliable when the process is only indirectly observed. For instance, temporal filtering, subsampling, and measurement noise in general transform VAR processes into VARMA processes [12], [13], while unobserved common drivers induce spurious edges. Nonlinear dynamics introduce additional mismatch, since VAR models capture only linear dependencies. Recent causal-discovery methods such as PCMCI extend observation-space analysis to high-dimensional and nonlinear settings [14], but still assume that all relevant drivers are observed. These limitations motivate explicit latent dynamical models.

State-space models (SSMs) separate latent dynamics from observations. In a linear Gaussian SSM, a sparse transition matrix A defines a directed graph over latent components, where $A _ { i j } \neq 0$ encodes a one-step-ahead influence of state component j on state component i. Unlike observation-space Granger estimates, this graph is defined on the latent dynamics, separately from the observation model [15], [16]. This interpretation extends to nonlinear SSMs, where the directed graph describes which latent components enter the transition model of each component. In some cases, SSM parameters are identifiable only up to similarity transformations, so interpreting the latent graph requires additional information, such as a known observation model or structural constraints on the model parameters. The latent transitions can also be given an interventional meaning if they represent structural equations (e.g., known physical unobserved processes). This view is formalized for continuous-time systems by structural dynamical causal models (SDCMs) [17]. This tutorial follows the resulting causal ladder (Figure 1): from observation-space Granger structure, through latent directed structure, to a causal reading of the latent graph.

Examples: A first example considers the cardiovascular setup in Figure 2, involving three latent physiological processes: x<sub>1</sub> (respiratory activity), x<sub>2</sub> (heart-rate variability), and x<sub>3</sub> (blood-pressure variability). This is a simplified version of the setting in [2, p. 303], where omitting respiration can induce apparent interactions between cardiovascular variables. The true latent structure is

![](images/d4c5dcb29202b32d2df0b991a9caee73b005c81d5fb7d586edbfe59257a8753f.jpg)  
Fig. 2. Running example: three-node dynamical system used throughout the paper. The latent respiratory process $x _ { 1 }$ drives both the future of heart rate $\left( x _ { 2 } \right)$ and blood pressure $( x _ { 3 } )$ . Only $y _ { 2 }$ and y<sub>3</sub> are measured. Observation-space analysis may therefore infer a spurious directed link between $y _ { 2 }$ and $y _ { 3 }$ due to the shared hidden driver $x _ { 1 }$ , motivating the transition to latent state-space inference.

$$
x _ { 1 } \longrightarrow x _ { 2 } , \qquad x _ { 1 } \longrightarrow x _ { 3 } , \qquad x _ { 2 } \not  x _ { 3 } , \qquad x _ { 3 } \not  x _ { 2 } ,\tag{1}
$$

so that $x _ { 1 }$ acts as a hidden common driver of both $x _ { 2 }$ and $x _ { 3 }$ . Observation-space analysis restricted to $( y _ { 2 } , y _ { 3 } )$ therefore induces a spurious directed dependence between the observed processes, motivating the transition from Granger analysis in $\mathbf { \mathscr { y } } _ { t }$ to latent state-space modeling.

A second example, the five-node chain $x _ { 1 }  x _ { 2 }  x _ { 3 }  x _ { 4 }  x _ { 5 }$ , has a different nature: when x<sub>3</sub> is unobserved, it is a hidden mediator instead of a hidden common cause, since $x _ { 2 }$ does affect $x _ { 4 }$ , but only through $x _ { 3 }$

## TIME-SERIES MODELS AND GRANGER CAUSALITY

## Granger causality

Definition: We consider a stationary stochastic process $\{ { y } _ { t } \} _ { t \ge 1 }$ , where $\pmb { y } _ { t } = ( y _ { t , 1 } , \dots , y _ { t , n _ { y } } ) ^ { \top } \in \mathbb { R } ^ { n _ { y } }$ stationarity ensures that the prediction-error variances below do not depend on t [1]. Denote by $\mathcal { H } _ { t - 1 }$ the information set containing the past of all components up to time $t - 1$ , and by $\mathcal { H } _ { t - 1 } \setminus y . _ { , j }$ the same set except the past of component $j .$ . The optimal predictor of $y _ { t , i }$ given a generic information set H is denoted $\mathcal { P } ( y _ { t , i } \mid \mathcal { H } )$

Definition 1 (Granger causality [1]): Component $y . , j$ is Granger causal for $y . , i$ if

$$
\mathrm { v a r } [ y _ { t , i } - \mathcal { P } ( y _ { t , i } \mid \mathcal { H } _ { t - 1 } ) ] < \mathrm { v a r } [ y _ { t , i } - \mathcal { P } ( y _ { t , i } \mid \mathcal { H } _ { t - 1 } \setminus y _ { \cdot , j } ) ] .\tag{2}
$$

This is a predictive notion of directed influence: it does not require, nor directly imply, that $j$ is a cause of i in the interventional sense [18]. Definition 1 is the weak variant (WLGC), using only past information. The strong variant (SLGC) [19], [20] extends the information set to contemporaneous variables, capturing instantaneous interactions.

Pairwise, conditional, and block variants: The definition in (2) is full-conditional, conditioning on the full past of all components. The pairwise variant restricts the information set to the past of $y . , i$ and $y . , j$ alone and is therefore prone to spurious detections (false positives) under hidden common drivers (Figure 2) [7]. Block variants apply the same comparison to groups of components: a group A Grangercauses a group B if removing the past of A from the full information set increases the prediction error of ${ \bf \nabla } _ { { \bf { y } } _ { B } ; }$ in VAR models this is a joint test of zero cross-block coefficients [8].

The VAR parametric setting: The optimal predictors in (2) are conditional expectations, which in practice require a working model to be computable. The most widely used model for GC is the vector autoregressive (VAR) model of order $p \mathrm { : }$

$$
\pmb { y } _ { t } = \sum _ { k = 1 } ^ { p } \pmb { \Phi } _ { k } \pmb { y } _ { t - k } + \pmb { e } _ { t } , \qquad \pmb { e } _ { t } \sim \mathcal { N } ( \mathbf { 0 } , \pmb { \Sigma } ) ,\tag{3}
$$

where $\Phi _ { k } \in \mathbb { R } ^ { n _ { y } \times n _ { y } }$ is the coefficient matrix at lag $k ,$ and $e _ { t }$ is a white-noise innovation with covariance Σ. Within a correctly specified VAR model, component $j$ Granger-causes component i if

$$
\Phi _ { k } ( i , j ) \neq 0 \quad \mathrm { f o r ~ s o m e } \ k \in \{ 1 , \ldots , p \} ,\tag{4}
$$

where $\Phi _ { k } ( i , j )$ is the coefficient from component $j$ at lag $k$ to component i at time t. The resulting directed graph $\mathcal { G } _ { y }$ has an edge $j  i$ whenever (4) holds. Geweke [21] introduced a strength measure of Granger causality based on the ratio of prediction error variances:

$$
\mathcal { F } _ { j  i } = \ln \frac { \mathrm { v a r } [ y _ { t , i } - \mathcal { P } ( y _ { t , i } \mid \mathcal { H } _ { t - 1 } \setminus y . , _ { j } ) ] } { \mathrm { v a r } [ y _ { t , i } - \mathcal { P } ( y _ { t , i } \mid \mathcal { H } _ { t - 1 } ) ] } .\tag{5}
$$

$\mathcal { F } _ { j  i } \geq 0$ , with $\mathcal { F } _ { j  i } = 0$ if and only if $j$ does not Granger-cause i. Thus, Granger causality in the VAR setting reduces to a coefficient-exclusion test, assessed via F-tests or likelihood-ratio statistics [7], [8].

Assumptions and limitations: The VAR-Granger framework relies on several assumptions:

• Stability: $\rho ( \Phi _ { \mathrm { c o m p } } ) < 1$ , where $\rho ( \cdot )$ is the spectral radius (maximum eigenvalue modulus) and $\Phi _ { \mathrm { c o m p } }$ the companion matrix of (3); this rules out explosive processes.

• Known uniform lag order $p \mathrm { : }$ misspecified $p$ biases the graph.

• Causal sufficiency: all relevant drivers are observed; an unmeasured common driver appears as a spurious Granger link (Figure 2).

• Perfect observation: $\mathbf { \mathscr { y } } _ { t }$ is free of measurement noise, temporal aggregation, and subsampling.

The assumptions are often violated in practice [7], producing spurious or missing Granger links. Hidden variables can affect Granger interpretation in two complementary ways. In the cardiovascular example, $x _ { 2 } \left. x _ { 1 } \right. x _ { 3 }$ with $x _ { 1 }$ hidden is a hidden common cause: intervening on $x _ { 2 }$ leaves $x _ { 3 }$ unchanged, yet a Granger link between them may appear. In the five-node chain, a hidden $x _ { 3 }$ is a hidden mediator: intervening on $x _ { 2 }$ does affect $x _ { 4 }$ , but only through $x _ { 3 }$ , and Granger analysis reports a direct link $x _ { 2 } \to x _ { 4 } ;$ the causal effect is real, and only its reading as a direct edge is spurious.

VARMA models: A vector autoregressive moving-average (VARMA) process of order $( p , q )$ extends the VAR by adding lagged innovations,

$$
y _ { t } = \sum _ { k = 1 } ^ { p } \Phi _ { k } y _ { t - k } + e _ { t } + \sum _ { \ell = 1 } ^ { q } \Theta _ { \ell } e _ { t - \ell } , \qquad e _ { t } \sim \mathcal { N } ( \mathbf { 0 } , \Sigma ) .\tag{6}
$$

VAR models are the subclass with $q = 0$ , and under standard invertibility conditions a VARMA process admits a VAR representation of infinite order, VAR(∞) [8, Ch. 11].

## State-space models

State-space models (SSMs) describe dynamical systems through a latent state $\pmb { x } _ { t } = ( x _ { t , 1 } , \dots , x _ { t , n _ { x } } ) ^ { \top } \in$ $\mathbb { R } ^ { n _ { x } }$ and an observed signal $\pmb { y } _ { t } = ( y _ { t , 1 } , \dots , y _ { t , n _ { y } } ) ^ { \top } \in \mathbb { R } ^ { n _ { y } }$ . In its simplest form, an SSM is specified by a transition density $p ( \pmb { x } _ { t } \mid \pmb { x } _ { t - 1 } )$ and an observation density $p ( \pmb { y } _ { t } \mid \pmb { x } _ { t } )$ , allowing nonlinear, non-Gaussian, and more general dynamical models. We focus first on the linear-Gaussian case,

$$
{ \pmb x } _ { t } = { \pmb A } { \pmb x } _ { t - 1 } + { \pmb w } _ { t } , \qquad { \pmb w } _ { t } \sim { \mathcal { N } } ( \mathbf { 0 } , { \pmb Q } ) ,\tag{7}
$$

$$
\begin{array} { r } { \pmb { y } _ { t } = \pmb { H } \pmb { x } _ { t } + \pmb { v } _ { t } , \qquad \pmb { v } _ { t } \sim \mathcal { N } ( \mathbf { 0 } , \pmb { R } ) , } \end{array}\tag{8}
$$

with initial state $\pmb { x } _ { 0 } \sim \mathcal { N } ( \pmb { \mu } _ { 0 } , \pmb { \Sigma } _ { 0 } )$ and mutually independent noise sequences. Here A governs the latent dynamics, H maps states to observations, Q is the state-noise covariance, and R is the observation-noise covariance. Filtering, smoothing, and prediction are available in closed form through the Kalman filter and Rauch–Tung–Striebel smoother [22], [23].

Any stationary $\mathrm { L G } \mathrm { - } \mathrm { S S M } , \rho ( A ) < 1$ , induces an observation process that is a linear transformation of a VAR(1) process and admits a VARMA representation [8, Sec. 11.6.1]. Conversely, every VARMA process admits an SSM representation, for instance by stacking lags of $\mathbf { \mathscr { y } } _ { t }$ and $e _ { t }$ into a companion state. Thus, LG-SSMs and VARMA models generate the same class of linear stationary processes, but the parametrizations are not one-to-one [8], [24].

## OBSERVATION-SPACE GRANGER CAUSALITY IN HIGH-DIMENSIONAL TIME SERIES

In VAR models, the computational and statistical problems arise quickly when the number of series $n _ { y }$ is large $( p n _ { y } ^ { 2 }$ scalar parameters). Granger causality in a VAR model is encoded in the sparsity pattern of the coefficient matrices: component $j$ Granger-causes component i if and only if $\Phi _ { k } ( i , j ) \neq 0$ for some $k \in \{ 1 , \ldots , p \}$ . Collecting $\{ \Phi _ { k } \} _ { k = 1 } ^ { p }$ into $\pmb { \Phi } = [ \pmb { \Phi } _ { 1 } \ \cdot \ \cdot \ \pmb { \Phi } _ { p } ]$ , the per-lag Granger graphs and the aggregate Granger graph are

$$
\mathcal { G } _ { y } ^ { ( k ) } ( i , j ) = \mathbb { 1 } [ \Phi _ { k } ( i , j ) \neq 0 ] , \qquad \mathcal { G } _ { y } \ = \ \bigcup _ { k = 1 } ^ { p } \mathcal { G } _ { y } ^ { ( k ) } ,\tag{9}
$$

where the union is taken entrywise. Estimating $\mathcal { G } _ { y }$ therefore reduces to support recovery of Φ [7]. This section reviews two families of methods: sparsity-inducing penalties on the coefficients, and strategies that reduce the effective conditioning set. Both families target the same graph $\mathcal { G } _ { y }$ but bypass the conditional Granger test (one full-conditional regression per pair $( i , j ) )$ , which is computationally and statistically challenging in high dimensions: penalized estimators read off $\mathcal { G } _ { y }$ from the support of a single regularized estimate of $\Phi .$ , while conditioning-set reduction methods perform explicit independence tests on a datadriven reduced parent set.

## Regularized VAR estimation

The methods in this subsection recover $\mathcal { G } _ { y }$ from a single estimate of the VAR coefficients Φ obtained via penalized loss minimization, rather than through separate pairwise Granger tests.

Lasso-penalized VAR: A natural approach to high-dimensional support recovery is to add an $\ell _ { 1 }$ penalty to the least-squares formulation of the VAR(p) model. Writing the model in regression form with $\mathbf { Y }$ stacking the observations and X the lagged regressors, the Lasso-VAR estimator [10] solves

$$
\widehat \Phi = \arg \operatorname* { m i n } _ { \Phi } \ \frac { 1 } { T } \| \pmb { Y } - \pmb { X } \pmb { \Phi } ^ { \top } \| _ { F } ^ { 2 } + \lambda \| \pmb { \Phi } \| _ { 1 } .\tag{10}
$$

This estimator is formally standard $\ell _ { 1 }$ -penalized regression on a stacked design matrix. However, unlike the classical i.i.d. setting, the rows of X are temporally dependent, and the resulting estimation theory depends on spectral properties of the underlying process. The penalty $\begin{array} { r } { \| \Phi \| _ { 1 } = \sum _ { i , j , k } | \Phi _ { k } ( i , j ) | } \end{array}$ promotes sparsity in the coefficients. Under Gaussian innovations with $\pmb { \Sigma } = \sigma ^ { 2 } \pmb { I }$ , minimizing the squared-error loss is equivalent to minimizing the negative log-likelihood, so (10) can be read as a penalized maximumlikelihood estimator. For stable Gaussian VAR processes $( \rho ( \Phi _ { \mathrm { c o m p } } ) < 1 )$ with sparse coefficients, Basu & Michailidis [10, Sec. 4] bound the estimation error of the Lasso; the bounds depend on the extremes of the spectral density, so stronger temporal dependence demands more samples.

Structured sparsity and latent factor corrections: A natural refinement under the GC perspective is to enforce sparsity jointly across lags by grouping the coefficients $\{ \Phi _ { 1 } ( i , j ) , \ldots , \Phi _ { p } ( i , j ) \}$ for each pair $( i , j )$ , leading to group-sparse penalties that select or discard entire directed edges across all lags, i.e., promoting sparsity on $\mathcal { G } _ { y }$ [25], [26]. Another generalization is block Granger causality, where the grouping is over blocks of dimensions rather than lags: components are partitioned into groups, and the penalty selects or discards the influence of a whole group on each target component at once [27].

Beyond pair-level sparsity, a few shared latent factors can induce spurious Granger connectivity. Basu et al. [11] address this by decomposing the transition matrix of a VAR(1) as $\Phi = L ^ { * } + R ^ { * }$ , with low-rank $L ^ { * }$ (capturing latent effects) and sparse $\pmb { R } ^ { * }$ (the interpretable Granger structure): the latent effects are accounted for, but not modeled as dynamical processes.

## Reducing the conditioning set: domain-guided and data-driven approaches

Unlike penalized estimators, the methods below perform explicit conditional-independence tests. Each test conditions on a reduced set of variables, chosen from domain knowledge or from the data. Highdimensional Granger analysis suffers from two related problems: large conditioning sets reduce statistical power, and strong autocorrelation can make additional predictive contributions from causal parents harder to detect [14]. The approaches below address this by restricting the conditioning set itself, either through structural constraints or data-driven parent selection.

Domain-guided conditioning: Partial knowledge of the physics or topology of the system (e.g., which components can Granger-cause others directly) can be used to decide in advance which components are plausible direct causes. Each conditional Granger test then conditions on a smaller, physically meaningful set. For instance, in intracavitary electrograms recorded during atrial fibrillation [28], electrodes at known positions record the propagation of electrical activation, and electrophysiology rules out certain connections, such as direct links between distant electrodes. Luengo et al. [28] build the graph as a directed tree: they select as root the electrode with the largest number of significant outgoing pairwise Granger links, and then accept a candidate child of each node only if its Granger causality remains significant given the children already accepted (GS-CaRe); the local variant (LS-CaRe) only considers pairwise-significant links to electrodes within a distance that grows up to a user-defined maximum, linking a distant electrode only if no closer one is connected. Restricting candidates to physically admissible links increases computational efficiency and removes spurious long-range links, at the price that the result is only as good as the prior knowledge it relies on.

Data-driven conditioning: PCMCI: When no structural prior is available, the conditioning sets must be learned from the data. PCMCI [14] constructs a sparse set of candidate parents for each target and tests directed links within this reduced space. It proceeds in two steps. First (PC step), a conditionselection stage iteratively prunes an initial candidate set using conditional independence tests, yielding a small set $\widehat { \mathcal { P } } ( y _ { t , i } )$ approximating the relevant parents. Second (MCI step), each candidate link $y _ { t - \tau , j } $ $y _ { t , i }$ , corresponding to the entry $\mathcal { G } _ { \pmb { y } } ^ { ( \tau ) } ( i , j )$ , is tested via the momentary conditional independence (MCI) criterion:

$$
y _ { t - \tau , j } \perp \perp y _ { t , i } \Bigm | \widehat { \mathcal { P } } ( y _ { t , i } ) \setminus \{ y _ { t - \tau , j } \} , \widehat { \mathcal { P } } ( y _ { t - \tau , j } ) .\tag{11}
$$

Conditioning on the parents of the target $y _ { t , i }$ isolates the direct effect of the candidate source, while conditioning on the parents of the source $y _ { t - \tau , j }$ mitigates inflated test statistics caused by strong autocorrelation [14]. PCMCI supports nonparametric conditional-independence tests, including conditional mutual information, closely related to transfer entropy $T _ { j \to i } = I ( y _ { t , i } ; y _ { \cdot , j } ^ { - } \mid y _ { \cdot , i } ^ { - } )$ [29], the informationtheoretic analogue of Granger causality (with $T _ { j  i } = { \textstyle \frac { 1 } { 2 } } \mathcal { F } _ { j  i }$ for jointly Gaussian processes [30], where ${ \mathcal { F } } _ { j  i }$ is the Geweke measure in (5)). This captures nonlinear dependencies beyond VAR-based methods. PCMCI+ [31] further extends the framework to contemporaneous links $( \tau = 0 )$ , analogous to strong Granger causality [19] in the VAR setting.

## FROM OBSERVATION-SPACE GRANGER TO STATE-SPACE REPRESENTATIONS

Vector autoregression is the natural home of Granger causality, but its graphical interpretation relies on strong modeling assumptions. Figure 3 summarizes four failure modes motivating the transition to latent state-space models: preprocessing can convert VAR into VARMA, finite-order truncation biases the retained coefficients, VARMA non-causality no longer corresponds to sparsity, and latent confounders induce spurious edges, as in the running example of Figure 2.

## Preprocessing and noisy measurements can convert VAR to VARMA

Several common preprocessing operations in general turn a VAR into a VARMA, breaking the model assumption of the methods in the previous section (Figure 4). Observing only a subset of the components of a VAR has the same effect [12]; we return to it below, since it is the hidden-variable problem.

Subsampling: A VAR(p) process observed at every m-th step is in general a VARMA [20] [8, Sec. 11.6.2], as an exception, a VAR(1) remains VAR(1), but with transition matrix $\Phi _ { 1 } ^ { m }$ , whose graph can differ. Solo [20, Sec. II-B] shows that strong Granger causality survives subsampling unless the components become independent, while no such guarantee holds for weak Granger causality. A similar mechanism appears when a continuous-time system is sampled, which matters for the causal reading of the latent graph discussed later.

![](images/f29505398cbd498e140948bd638cbecf511a8b82ee92675da49adef63e45d1b1.jpg)

![](images/6adf19c1efeca16918857b2afc1ad15f21c1e35dfabd1682cdc5da8ba5c485eb.jpg)

![](images/f3736ae5f9171c3eb26cce0823fa33a3d5d37405d20467b4750a242d81d35455.jpg)

![](images/7cf060c1e11b77dd09913876b24066877addeaab80a213df371897abcc202b84.jpg)

Fig. 3. Four failure modes of VAR-based Granger causality. The first three concern the parametrization of the observed process. (a) Preprocessing: filtering, subsampling, or measurement noise in general convert a $\operatorname { V A R } ( p )$ into a $\mathbf { V A R M A } ( p ^ { \prime }$ , q) process. (b) Model truncation: finite-order VAR fits truncate the induced $\operatorname { V A R } ( \infty )$ representation, biasing $\hat { \Phi } _ { k }$ and distorting $\mathcal { G } _ { y } .$ (c) VARMA opacity: in VARMA models, Granger non-causality is not encoded as sparsity in Φ $\qquad \mathrm { { o r } \ \Theta } .$ , but as a rational identity involving both [32]. The fourth failure mode, (d) Latent confounding, is qualitatively different: hidden drivers alter the information set itself, inducing spurious edges in $\mathcal { G } _ { y }$ that observation-space estimators assuming causal sufficiency cannot remove. Together, these effects motivate the transition to latent state-space modeling.  
![](images/97bc713f8397364e6f293c63fe5ef596501f65d1287cc6e6edeb3dcfea6beb5c.jpg)  
Fig. 4. Three common operations in general turn a VAR process into a VARMA: temporal subsampling, linear filtering, and additive measurement noise. Any finite-order VAR fit to the transformed series is misspecified.

Filtering: Linear time-invariant (LTI) filtering can likewise turn a VAR process into a VARMA process [12], so that accurately representing the filtered process by a pure VAR can require a substantially higher model order.

Measurement noise: If the underlying process evolves as a VAR but is observed with additive independent noise, $\tilde { \pmb { y } } _ { t } = \pmb { y } _ { t } + \pmb { v } _ { t } .$ , then $\tilde { \mathbf { y } } _ { t }$ admits a VARMA representation whose MA component is generated by the noise innovation [20, Sec. III]; see also [8, Ch. 11].

## Truncation pathology

Under standard invertibility conditions, a VARMA process admits a $\mathrm { V A R } ( \infty )$ representation, so any finite-order $\operatorname { V A R } ( p )$ fit truncates this expansion. The omitted high-lag terms carry predictive content that spills into the retained coefficients $\{ \Phi _ { k } \} _ { k = 1 } ^ { p }$ , potentially biasing the Granger graph both ways:

low-lag coefficients may absorb high-lag effects (spurious edges), while real influence spread across several lags can drop below detection (missing edges) [8, Ch. 15]. Information criteria [8] and Bayesian order inference [33] can identify a reasonable finite $p ,$ but residual serial correlation generally remains whenever the true process has an infinite VAR representation. Moreover, the guarantees of the previous section assume a correctly specified finite-order VAR; how sparse VAR graph recovery behaves under VARMA-induced truncation remains much less understood.

## Innovations-based Granger and the residual gap

Barnett & Seth [12] avoid VAR truncation by representing the observation process with an LG-SSM and computing Granger causality directly from the innovation covariance produced by the Kalman filter. In steady state, the one-step predictive covariance of $\mathbf { \mathscr { y } } _ { t }$ given $\pmb { y } _ { 1 : t - 1 }$ is $\pmb { \Sigma } = \pmb { H } \pmb { V } \pmb { H } ^ { \top } + \pmb { R }$ , where V is the unique stabilizing solution of the discrete algebraic Riccati equation (DARE), not the observation noise R. The Geweke measure becomes

$$
\mathcal { F } _ { j  i } = \ln \frac { | \widetilde { \Sigma } _ { i i } | } { | \Sigma _ { i i } | } ,\tag{12}
$$

where $\widetilde { \pmb { \Sigma } }$ is the analogous innovation covariance from the LG-SSM reduced by removing $y _ { j }$ from the observation vector [12], [21]. This computes exact observation-space Granger causality without finiteorder VAR approximation. However, the resulting graph is still the observation-space graph $\mathcal { G } _ { y }$ , not the latent $\mathcal { G } _ { x }$ . The link $( i , j )$ is absent if and only if

$$
{ \pmb { H } } _ { i } \left( { \pmb { A } } - { \pmb { K } } { \pmb { H } } \right) ^ { k } { \pmb { K } } _ { j } = \mathbf { 0 } , \quad k = 0 , 1 , \ldots , n _ { x } - 1 ,\tag{13}
$$

where $\pmb { H } _ { i }$ denotes the i-th row of H, $K _ { j }$ the j-th column of K, and $\pmb { K } = \pmb { A } \pmb { V } \pmb { H } ^ { \top } \pmb { \Sigma } ^ { - 1 }$ is the steady-state Kalman gain in predictor form (mapping the innovation into $\widehat { \pmb { x } } _ { t + 1 | t } )$

## An attempt to interpret VARMA models as Granger graphs

Since preprocessing generally turns the VAR model into a VARMA one, the next question is whether the VARMA parametrization itself admits a transparent GC graph interpretation. Boudjellaba et al. [32] consider a stationary invertible block-VARMA $\Phi ( z ) \pmb { y } _ { t } = \Theta ( z ) \pmb { e } _ { t }$ partitioned as $\pmb { y } = ( \pmb { y } _ { 1 } , \pmb { y } _ { 2 } )$ , where $\Phi ( z )$ and $\Theta ( z )$ are the AR and MA matrix polynomials with blocks $\{ \Phi _ { i j } ( z ) \}$ and $\{ \Theta _ { i j } ( z ) \}$ . They show that the Granger non-causality ${ \pmb y } _ { 1 } \not \to { \pmb y } _ { 2 }$ holds if and only if $\Phi _ { 2 1 } ( z ) - \Theta _ { 2 1 } ( z ) \Theta _ { 1 1 } ( z ) ^ { - 1 } \Phi _ { 1 1 } ( z ) \equiv 0 $ as a rational identity in z. Two consequences follow. First, the AR and MA contributions can cancel each other so that $\textstyle \pmb { y } _ { 1 } \not \to \pmb { y } _ { 2 }$ even when both $\Phi _ { 2 1 }$ and $\Theta _ { 2 1 }$ are nonzero. Second, an MA cross-block alone can create a Granger link in the absence of any AR cross-coupling. Thus, the VARMA parametrization is opaque with respect to the Granger graph: sparsity in either coefficient block is neither necessary nor sufficient for Granger non-causality, precluding any direct reading of the Granger graph from Φ or Θ alone.

## Latent confounding

Latent confounding is central to causal inference: an unmeasured common driver of two observed components makes any observation-space notion of directed influence ambiguous [18], [34]. Unlike preprocessing, truncation, or VARMA misspecification, which distort the parametrization, confounding compromises the information set itself. When a common latent driver acts on multiple observed components, its shared variability can be read by observation-space estimators as direct predictive influence and produce edges that no underlying dynamical mechanism contains.

In causal terminology, hidden variables may act either as confounders (common causes) or as unobserved mediators along causal pathways; throughout this tutorial, both situations are treated under the broader perspective of hidden-variable effects on the observation-space graph. In both cases, the hiddenvariable effect can disappear when the omitted process is included explicitly in the latent state, motivating the transition from observation-space analysis on $\mathbf { \mathscr { y } } _ { t }$ to latent state-space modeling.

Suppose that in Figure 2, $x _ { 2 }$ and $x _ { 3 }$ are affected by $x _ { 1 }$ one step later, and that only $\pmb { y } _ { t } = ( y _ { t , 2 } , y _ { t , 3 } ) ^ { \top }$ is observed while $x _ { t , 1 }$ remains hidden. Although no direct interaction exists between $x _ { t , 2 }$ and $x _ { t , 3 } .$ , the shared influence of ${ \boldsymbol { x } } _ { t - 1 , 1 }$ may induce predictive dependence between the observed components. Observationspace methods therefore tend to infer a spurious edge between $y _ { t , 2 }$ and $y _ { t , 3 }$

This failure concerns methods that assume causal sufficiency, such as Granger causality, Lasso-VAR, and PCMCI; tsFCI [35] and LPCMCI [36] drop this assumption and can flag a dependence as possibly due to a hidden common cause, but do not recover the hidden process or its links.

## LEARNING AND INTERPRETING STRUCTURE IN THE LATENT SPACE

We now move to latent dynamical models, where the directed graph is defined by the transition structure of the hidden state itself rather than by predictive relations among observations $\mathbf { \mathscr { y } } _ { t }$ . The central object of this section is the directed graph $\mathcal { G } _ { x }$ defined by the sparsity of the transition kernel of the latent dynamics. We first formalize this graphical interpretation in LG-SSMs and pose the penalized maximum-likelihood estimation problem.

## Directed structure in latent dynamics

In the LG-SSM, the latent state evolves as a VAR(1) on the unobserved process, $\pmb { x } _ { t } = \pmb { A } \pmb { x } _ { t - 1 } + \pmb { w } _ { t }$ . The transition matrix $\pmb { A } \in \mathbb { R } ^ { n _ { x } \times n _ { : } }$ <sup>x</sup> encodes the one-step latent dynamics: entry $A _ { i j }$ quantifies the influence

of component $j$ at time $t - 1$ on component i at time t. Equivalently, $A _ { i j } \neq 0$ if and only if component $j$ appears in the structural equation of component i, $\begin{array} { r } { x _ { t , i } = \sum _ { j } A _ { i j } x _ { t - 1 , j } + w _ { t , i } } \end{array}$ . Its sparsity pattern thus can be read as a directed graph $\mathcal { G } _ { x }$ over the $n _ { x }$ latent components [15]:

$$
j \to i \in { \mathcal G } _ { x } \quad \Longleftrightarrow \quad A _ { i j } \ne 0 .\tag{14}
$$

The directed graph $\mathcal { G } _ { x }$ is the latent counterpart of the observation-space Granger graph $\mathcal { G } _ { y }$ in (9). Under this SSM, the Granger graph on $\mathbf { \mathscr { y } } _ { t }$ is a joint function of A, H, Q, and R (13), mixing latent dynamics with measurement effects. Note that higher-order dynamics $\begin{array} { r } { \pmb { x } _ { t } = \sum _ { k = 1 } ^ { p } \pmb { A } _ { k } \pmb { x } _ { t - k } + \pmb { w } _ { t } } \end{array}$ can be rewritten as a first-order SSM via companion-form augmentation at the cost of a larger state dimension [37].

## Learning sparse latent dynamics

The log-likelihood of A can be computed by marginalizing over the state trajectory $\pmb { x } _ { 1 : T }$ as [23]

$$
\log p ( \pmb { y } _ { 1 : T } \mid A ) = - \sum _ { t = 1 } ^ { T } \Bigl [ \frac { 1 } { 2 } \log \operatorname* { d e t } ( 2 \pi \pmb { S } _ { t } ) + \frac { 1 } { 2 } \pmb { z } _ { t } ^ { \top } \pmb { S } _ { t } ^ { - 1 } \pmb { z } _ { t } \Bigr ] ,\tag{15}
$$

where $z _ { t } = y _ { t } - H A { \widehat { x } } _ { t - 1 }$ is the innovation at time t and $S _ { t }$ is its covariance, both obtained by running the Kalman filter with a given A. A common approach to recover a sparse A is to solve the penalized problem

$$
\widehat { \pmb { A } } = \arg \operatorname* { m i n } _ { \pmb { A } } \underbrace { - \log p ( \pmb { y } _ { 1 : T } \mid \pmb { A } ) } _ { \mathrm { d a t a \ f i t \ v i a \ K F } } + \underbrace { \lambda _ { A } \Omega ( \pmb { A } ) } _ { \mathrm { s t r u c t u r a l \ p e n a l t y } } ,\tag{16}
$$

where Ω is a (possibly non-differentiable) convex penalty promoting a desired property of the directed graph $\mathcal { G } _ { x }$ : entry-wise sparsity $( \Omega ( A ) = \| A \| _ { 1 }$ <sub>1</sub>, the default), block-sparsity for pair-level Granger structure, low rank via the nuclear norm, or stability via the operator-norm ball. The recursive dependence of $\widehat { \pmb { x } } _ { t - 1 }$ and $S _ { t }$ on A makes the problem highly nonconvex; although gradients are available, EM-based majorization (GraphEM) and block alternating majorization-minimization (DGLASSO) are used in this case.

## Structure learning in LG-SSMs

We now present three methods within the LG-SSM framework: two yield point estimates of the graph as in [15], [16], and one takes a fully Bayesian approach that quantifies uncertainty over the graph topology (see also Table I).

GraphEM (learning the directed graph) [15]: GraphEM computes the MAP estimate of A, with the other LG-SSM parameters known, under a penalty $\lambda _ { A } \Omega ( A ) { \mathrm { ~ } } ( \mathbf { e } . \mathbf { g } . , \ell _ { 1 } { \mathrm { ~ n o r m } } )$ . It modifies the EM algorithm for SSMs [38]: the E-step runs the Kalman filter and RTS smoother with the current $\boldsymbol { A } ^ { ( i ) }$ to build a minorizer of the log-likelihood, and the M-step minimizes the penalized surrogate. Under mild conditions, the objective decreases monotonically and every cluster point of the iterates is a critical point [15, Thm. 1].

![](images/9a0f42300fdf5cd77c0a7c2f1afe532635965c30975aad0571ef154e3a304a5b.jpg)  
Fig. 5. Observation-space sparsity versus latent-space sparsity. Data are generated from the sparse LG-SSM shown in panel (c): a 5-dimensional Markov-1 latent process with sparse transition matrix A (chain $1  2  3  4  5$ plus self-loops) and sparse state-noise precision graph G<sub>P</sub> (edges 1−3 and 2−4), observed through identity H with additive measurement noise. The progression 56 $( \mathrm { O L S } ) \to 2 5$ (Lasso) $\mid  9 + 2$ (latent SSM) illustrates the parametrization gap between observation-space and latent-space representations. We use $T = 2 \cdot 1 0 ^ { 5 }$ samples. (a) $\mathrm { O L S } \ \mathrm { V A R } ( p = 3 )$ fit on the observations, together with the residual precision graph. Only arrows arriving at time t are displayed. Because the noisy observation process is VARMA, the induced finite-order VAR representation becomes dense even though the latent dynamics are sparse. (b) Row-wise Lasso- $\mathsf { V A R } ( p = 3 )$ on the same data. The $\ell _ { 1 }$ penalty removes many weak coefficients, but several lag-2 and lag-3 links persist, reflecting truncation effects and the effective infinite-order dependence induced in observation space. (c) DGLASSO [16] recovers the true sparse dual graph $\left( { \mathcal { G } } _ { x } , { \mathcal { G } } _ { P } \right)$ in latent space.

DGLASSO (learning directed and undirected graphs) [16]: DGLASSO jointly estimates A and the state-noise precision $P = Q ^ { - 1 }$ by minimizing the negative marginal log-likelihood (15) plus $\lambda _ { A } \| A \| _ { 1 } +$ $\lambda _ { P } \Vert P \Vert _ { 1 }$ <sub>1</sub>, with a block alternating majorization-minimization scheme that converges to a critical point under mild conditions [16, Thm. 2]. As in the graphical Lasso [39], [40], the off-diagonal zeros of P define an undirected graph $\mathcal { G } _ { P }$ of conditional dependencies between noise components, so DGLASSO returns a dual description: dynamical influences in $\mathcal { G } _ { x }$ and contemporaneous dependencies in $\mathcal { G } _ { P }$

Figure 5 illustrates the parametrization gap with a small simulation. A sparse Markov-1 SSM with 9 transition entries and 2 undirected precision edges (panel (c)) induces in observation space as a much denser VAR representation: 50 directed OLS-VAR coefficients plus 6 residual-precision edges (panel (a)), or 25 coefficients after $\ell _ { 1 }$ regularization (panel (b)). The lag-2 and lag-3 “ghost” edges persist under Lasso because they reflect the effective higher-order dependence induced in observation space by the latent dynamics and measurement noise.

Bayesian formulations: A complementary perspective is to treat A as a random variable and compute its posterior distribution $p ( A \mid { \pmb y } _ { 1 : T } )$ , quantifying uncertainty over both the edge weights and the graph topology, i.e., also $p ( \mathcal { G } _ { x } \mid \pmb { y } _ { 1 : T } )$ . SpaRJ [41] is a fully Bayesian approach for the LG-SSM based on reversible-jump Markov chain Monte Carlo (RJMCMC) [42], interpreting the sparsity pattern of $A , \mathcal { G } _ { x } .$

as a model indicator: each possible support defines a different model, and the chain explores model space and parameter space hierarchically. Transition kernels add or remove edges (model jumps) and update the nonzero entries (within-model moves); the Kalman filter provides the marginal likelihood for the acceptance ratio. The output samples can be processed for instance to estimate the probability $\operatorname* { P r } ( A _ { i j } \neq 0 \ | \ y _ { 1 : T } )$ for each directed edge.

In the overall picture, the Bayesian route reports uncertainty over graphs instead of a thresholded estimate; its priors play the role of the structural information (sparsity priors for the $\ell _ { 1 }$ penalty, zero prior mass for masked edges), and a non-identified model can show up as a posterior spread over the similarity class. Particle MCMC and $\mathrm { S M C ^ { 2 } }$ extend it to nonlinear SSMs by replacing the Kalman likelihood with particle-filter estimates [43]. The cost (one filtering pass per iteration over up to $2 ^ { n _ { x } ^ { 2 } }$ supports) limits it to moderate $n _ { x }$

## Beyond linear-Gaussian dynamics

Many systems of scientific interest are however nonlinear (e.g., chaotic physical systems, biological oscillators); forcing a linear model distorts both the inferred dynamics and the recovered graph. Recent approaches in dynamical systems have explored sparse nonlinear representations and polynomial approximations for discovering governing equations from data [44]. GraphGrad [45] extends LG-SSMs by replacing linear transitions with sparse polynomial dynamics, which can be interpreted as polynomial approximations of more general transition functions. Note that several canonical dynamical systems of practical interest (Lorenz 63 and 96, Lotka-Volterra predator-prey models, compartmental epidemiological models such as SIR and SEIR) admit exactly polynomial, often quadratic, state dynamics.

GraphGrad parametrizes the transition as

$$
\pmb { x } _ { t } \sim p ( \pmb { x } _ { t } \mid \pmb { x } _ { t - 1 } ; \pmb { C } ) = \mathcal { N } \big ( \pmb { C } m ( \pmb { x } _ { t - 1 } ; \pmb { D } ) , \pmb { Q } \big ) ,\tag{17}
$$

where ${ \pmb m } ( { \pmb x } ; { \pmb D } ) \in \mathbb { R } ^ { M }$ is the vector of all monomials of degree at most d in x, whose exponents are encoded in the columns of $\pmb { D } \in \mathbb { N } _ { 0 } ^ { n _ { x } \times M }$ (fixed once $n _ { x }$ and d are chosen), and $C \in \mathbb { R } ^ { n _ { x } \times M }$ is the learnable coefficient matrix [45]. Explicitly, the ℓ-th column $D _ { : , \ell } = [ D _ { 1 \ell } , \ldots , D _ { n _ { x } \ell } ] ^ { \top }$ of D is the exponent vector that defines the ℓ-th monomial,

$$
m ( \boldsymbol { x } ; D ) = \big [ m _ { 1 } ( \boldsymbol { x } ; D ) , \dots , m _ { M } ( \boldsymbol { x } ; D ) \big ] ^ { \top } , \qquad m _ { \ell } ( \boldsymbol { x } ; D ) = \prod _ { j = 1 } ^ { n _ { x } } x _ { j } ^ { D _ { j \ell } } ,\tag{18}
$$

and the columns of D enumerate all exponent vectors with total degree $\begin{array} { r } { \sum _ { j = 1 } ^ { n _ { x } } D _ { j \ell } \leq d , } \end{array}$ , so that $M =$ $\textstyle { \binom { n _ { x } + d } { d } }$ . The column of zeros yields the constant monomial $m _ { \ell } = 1$ . The LG-SSM in (7) is the special

TABLE I  
COMPARISON OF GRAPH-BASED SSM ESTIMATION METHODS.
<table><tr><td></td><td>GraphEM [15]</td><td>DGLASSO [16]</td><td>GraphGrad [45]</td></tr><tr><td>Model class</td><td>LG-SSM</td><td>LG-SSM</td><td>General (nonlinear) SSM</td></tr><tr><td>Target parameter(s) Graph type</td><td>A Directed</td><td>A,  $P = Q ^ { - 1 }$  Directed + undirected</td><td>C (polynomial coeff.) Directed</td></tr><tr><td>Inference alg.</td><td>Kalman filter/smoother</td><td>Kalman filter/smoother</td><td>Differentiable particle filter</td></tr><tr><td>Optimization alg. Sparsity penalty</td><td>EM + proximal M-step  $\ell _ { 1 }$  on A</td><td>Block alternating MM  $\ell _ { 1 }$  on A and P</td><td>Proximal gradient descent  $\ell _ { 1 }$  on  $C$ </td></tr></table>

case of (17) with $d = 1$ (up to the constant term), with $C = A$ and $m ( { \pmb x } _ { t - 1 } ; { \pmb D } ) = { \pmb x } _ { t - 1 }$ . The directed graph $\mathcal { G } _ { x }$ is determined by the sparsity pattern of C: component $j$ directly influences component i iff at least one monomial involving $x _ { j }$ has a nonzero coefficient in the i-th row of C,

$$
j \to i \ \in \ { \mathcal { G } } _ { x } \quad \iff \quad \exists \ell : D _ { j \ell } > 0 \ { \mathrm { a n d } } \ C _ { i \ell } \neq 0 .\tag{19}
$$

For example, the Euler–Maruyama discretization of the Lorenz 63 system is exactly of this form with $d = 2 \cdot$ : each state equation combines linear terms with at most one bilinear term $( x _ { 1 } x _ { 3 } \ \mathrm { o r } \ x _ { 1 } x _ { 2 } )$ , and the induced graph contains every directed edge except 3 → 1.

The MAP estimate solves

$$
\widehat { \pmb { C } } = \arg \operatorname* { m i n } _ { \pmb { C } } \Big ( - \log p ( \widehat { \pmb { y } _ { 1 : T } \mid \pmb { C } } ) + \lambda _ { C } \| \pmb { C } \| _ { 1 } \Big ) ,\tag{20}
$$

where $\widehat { p ( \pmb { y } _ { 1 : T } \mid C ) }$ is a differentiable approximation of the intractable likelihood obtained with a differentiable particle filter (DPF) [45].

Measurement-noise experiment: We compare observation-space VAR-GC against a penalized-likelihoo LG-SSM estimator on an LG-SSM with $n _ { x } = 5$ latent processes arranged in a directed chain $x _ { 1 } $ $x _ { 2 }  x _ { 3 }  x _ { 4 }  x _ { 5 } .$ , with diagonal coefficients $A _ { i , i } = 0 . 2$ and off-diagonal coefficients $A _ { i + 1 , i } = 0 . 6 ,$ all components observed $( H = I )$ , with additive observation noise and varying noise level $\sigma _ { R } .$ When a VAR is fitted directly to noisy observations, the lagged regressors are themselves contaminated by noise (an errors-in-variables problem), which attenuates the estimated autoregressive coefficients toward zero as σ<sub>R</sub> increases. True directed effects therefore become progressively harder to detect. By contrast, the LG-SSM models the observation noise explicitly. Table II shows the resulting failure mode: VAR-GC progressively loses true edges, while GraphEM [15] retains all true edges across the tested noise levels. However, with $T = 5 0 0$ and the edge rule $| \widehat { A } _ { i j } | > 0 . 0 5$ , GraphEM also reports small spurious entries, whose magnitude grows with $\sigma _ { R } ,$ yielding lower precision and $F _ { 1 }$ than VAR-GC. Thus, explicitly modeling measurement noise prevents false negatives, but sparse graph recovery based on $\ell _ { 1 }$ regularization from short noisy series still requires adequate thresholding.

TABLE II  
MEASUREMENT-NOISE EXPERIMENT: ADDITIVE OBSERVATION NOISE INDUCES FALSE NEGATIVES IN VAR-BASED GRANGER CAUSALITY ESTIMATION. FIVE-NODE CHAIN, $T = 5 0 0 , \alpha = 0 . 0 5$ WITH BONFERRONI CORRECTION. THE ATTENUATED COEFFICIENT DENOTES THE POPULATION OLS VALUE ASSOCIATED WITH EACH TRUE EDGE UNDER MEASUREMENT NOISE. RESULTS ARE AVERAGED OVER 500 INDEPENDENT MONTE CARLO RUNS. RECALL (R), PRECISION (P) AND $F _ { 1 }$ ARE POOLED OVER RUNS; A GRAPHEM EDGE IS DECLARED WHEN $| \widehat { A } _ { i j } | > 0 . 0 5 , i \neq j$
<table><tr><td rowspan="2"> $\sigma _ { R }$ </td><td rowspan="2">att. coeff.</td><td colspan="2">VAR-GC</td><td rowspan="2"></td><td colspan="3">GraphEM</td></tr><tr><td>R</td><td>P  $F _ { 1 }$ </td><td>R</td><td>P</td><td> $F _ { 1 }$ </td></tr><tr><td>0.0 0.5</td><td>0.60 0.49-0.53</td><td>1.00 1.00 1.00</td><td>0.99 0.98 0.96</td><td>1.00 0.99 0.98</td><td>1.00 1.00</td><td>0.60 0.49</td><td>0.75 0.66</td></tr></table>

## Identifiability

For a generic LG-SSM, different parameter triples $( A , H , Q )$ can generate the same marginal distribution over ${ \mathbf { } } ^ { y } \mathrm { _ { 1 : T } } \mathrm { : }$ for any invertible $\pmb { T } \in \mathbb { R } ^ { n _ { x } \times n _ { x } }$ , the transformed system $( T A T ^ { - 1 } , H T ^ { - 1 } , T Q T ^ { \top } )$ with initial distribution $\mathcal { N } ( T \mu _ { 0 } , T \Sigma _ { 0 } T ^ { \top } )$ , produces identical observations. This similarity class has $n _ { x } ^ { 2 }$ degrees of freedom, and the transformation generally changes the sparsity pattern of A [24].

Observation geometry: When H is known and full column rank with $n _ { y } \ \geq \ n _ { x }$ , the constraint ${ \pmb { H } } { \pmb { T } } ^ { - 1 } = { \pmb { H } }$ forces ${ \boldsymbol { \mathbf { T } } } = { \boldsymbol { \mathbf { I } } } ,$ eliminating the similarity ambiguity (including the common setting $n _ { x } = n _ { y } ,$ $H = I )$ ; identifiability of A then depends on the usual statistical conditions, such as sufficient excitation of the state dynamics. When $n _ { y } < n _ { x }$ , H cannot have full column rank, leaving a non-trivial family of T with $\pmb { H } \pmb { T } ^ { - 1 } = \pmb { H }$ , and hence of equivalent models, and additional restrictions are then required [24].

Structural restrictions: When structural information about the latent system is available, it can be encoded as a hard restriction on the parameter space, which rules out some of the equivalent models:

• Structural masks on A: imposing $A _ { i j } ~ = ~ 0$ for physically impossible connections rules out every equivalent model whose transition matrix $\pmb { T } \pmb { A } \pmb { T } ^ { - 1 }$ does not keep these zeros, as with the spatial neighborhood constraints of [28].

• Independent state noise: a diagonal $Q$ removes only part of the ambiguity: with $Q$ fixed, $T Q T ^ { \top } = Q$ still allows the continuous family ${ \pmb T } = { \pmb Q } ^ { 1 / 2 } { \pmb O } { \pmb Q } ^ { - 1 / 2 }$ , O orthogonal. Latent rotations remain, and reducing the ambiguity to signed permutations requires breaking them, e.g., with structural masks or non-Gaussian noise.

• Canonical parametrizations: canonical forms, such as the observable echelon form, fix one represen tative per similarity class by imposing a known pattern of zeros and ones on H and part of A [24], [46]. Under standard regularity conditions this makes the model identifiable, but the zeros are imposed by the parametrization, so the sparsity pattern of the canonical A cannot be read as $\mathcal { G } _ { x }$

Sparse representatives when structure is unknown: When such restrictions are unavailable, the natural fallback is to estimate A under a sparsity-inducing penalty, as reviewed above. The likelihood remains constant along the similarity class [24], so the model is still not identifiable [47]; the penalty only selects, among equally good fits, one with small $\ell _ { 1 }$ norm. Permutations and diagonal scalings of the latent state preserve the sparsity pattern of A, so a sparse representative can be unique at most up to these transformations; fixing them requires information that maps latent components to physical variables, such as a known H or a structural mask. Beyond that, we are not aware of a general uniqueness result: support-recovery guarantees for sparse regression [48], [49] apply once the model is identified, not to the selection within a class.

What can be read from the recovered graph: The eigenvalues of A and the spectral density of $\mathbf { \mathscr { y } } _ { t }$ (hence the observation-space Granger graph) are invariant across the similarity class. However, the sparsity pattern of A is not, so reading $j  i$ as an interaction between physical variables requires anchoring the latent coordinates, e.g., through a known H or a structural mask. The recovered graph $\mathcal { G } _ { x }$ therefore says more than an observation-space Granger graph, because it describes the latent dynamics and can avoid the failure modes of the previous section under correct model specification. It says less than a causal graph, which additionally requires reading the transition equations as structural mechanisms, as discussed next.

Nonlinear constraints: We illustrate how structural constraints improve graph recovery under nonlinear dynamics and model misspecification. Following Luengo et al. [28], a 10-pole circular catheter yielding nine bipolar channels records synthetic intracardiac electrograms generated by coupled FitzHugh–Nagumo oscillators. The dynamics are nonlinear, so neither the VAR nor the LG-SSM assumptions hold exactly. The ground-truth propagation graph is known, and physiology provides an additional constraint: electrical wavefronts propagate locally, so electrode i can directly influence only spatial neighbors $( d _ { \operatorname* { m a x } } = 1 )$ . The dominant factor is therefore not whether inference is performed in observation or latent space, but whether the admissible graph class is structurally constrained.

We compare pairwise GC, full-conditional GC, PCMCI [14], LS-CaRe [28], and GraphEM with the same spatial mask imposed after each M-step. For PCMCI we run two variants as an ablation: a linear partial-correlation test $( \mathrm { P C M C I _ { l i n } } )$ and a nonlinear k-NN conditional mutual information test $( \mathrm { P C M C I _ { n l } } )$ The purpose is to separate the contribution of the conditional-independence test from that of the structural constraint. Table III shows that the best method in every row uses the locality mask. The two PCMCI variants obtain similar $F _ { 1 }$ scores, so the choice of test does not explain their low accuracy. LS-CaRe performs best for single-rotor and circular propagation, while GraphEM performs best for flat wavefronts. GraphEM is less robust in the single-rotor regime.

TABLE III  
NONLINEAR-CONSTRAINTS EXPERIMENT: $F _ { 1 }$ SCORES ON FITZHUGH–NAGUMO ELECTROGRAM DATA [28]. THE BEST-PERFORMING METHOD IN EVERY SETTING USES THE SPATIAL LOCALITY CONSTRAINT $d _ { \operatorname* { m a x } } = 1 \ : ( \mathrm { L S } { \cdot } \mathrm { C A R E } .$ GRAPHEM). $\mathbf { P C M C I } _ { \mathrm { l i n } }$ : LINEAR PARTIAL CORRELATION AND $\mathrm { P C M C I _ { n l } } \colon$ NONLINEAR. BOLD: BEST METHOD PER ROW.
<table><tr><td rowspan="2">Pattern SNR</td><td rowspan="2"></td><td colspan="4">No constraint</td><td colspan="2"> $d _ { \operatorname* { m a x } } = 1$ </td></tr><tr><td>Pair</td><td>Full</td><td> $\mathbf { P C M C I } _ { \mathrm { l i n } }$ </td><td> $\mathrm { P C M C I _ { n l } }$ </td><td> $_ \mathrm { L S - C a R e }$ </td><td>GraphEM</td></tr><tr><td rowspan="3">Single</td><td>10 dB</td><td>0.63</td><td>0.50</td><td>0.13</td><td>0.16</td><td>1.00</td><td>0.64</td></tr><tr><td>20 dB</td><td>0.58</td><td>0.75</td><td>0.18</td><td>0.13</td><td>1.00</td><td>0.50</td></tr><tr><td>40 dB</td><td>0.44</td><td>0.62</td><td>0.14</td><td>0.12</td><td>0.90</td><td>0.53</td></tr><tr><td rowspan="3">Circular</td><td>10 dB</td><td>0.09</td><td>0.00</td><td>0.05</td><td>0.18</td><td>0.71</td><td>0.60</td></tr><tr><td>20 dB</td><td>0.17</td><td>0.13</td><td>0.18</td><td>0.15</td><td>0.88</td><td>0.57</td></tr><tr><td>40 dB</td><td>0.23</td><td>0.21</td><td>0.16</td><td>0.19</td><td>0.82</td><td>0.67</td></tr><tr><td rowspan="3">Flat</td><td>10 dB</td><td>0.56</td><td>0.17</td><td>0.22</td><td>0.20</td><td>0.71</td><td>0.80</td></tr><tr><td>20 dB</td><td>0.60</td><td>0.59</td><td>0.24</td><td>0.19</td><td>0.50</td><td>0.63</td></tr><tr><td>40 dB</td><td>0.47</td><td>0.58</td><td>0.16</td><td>0.12</td><td>0.56</td><td>0.63</td></tr></table>

![](images/0f53f01c5b23a85acf228e5c72e875e416a181008f061f9ce0359130ea35256e.jpg)  
Fig. 6. Conceptual relation between SCMs, SDCMs, and state-space models. SCMs encode static causal mechanisms; SDCMs extend them to dynamical systems with feedback and interventions on trajectories; discretizing an SDCM yields the transition equations of an SSM, which can also be read structurally on its own.

## Causal interpretation of the latent graph

The methods presented above recover directed structure from latent dynamics: $j \to i$ holds iff $A _ { i j } \neq 0$ in the linear case (14), or iff $x _ { t - 1 , j }$ appears in $f _ { i } ( { \pmb x } _ { t - 1 } )$ in the polynomial case (19). A discrete-time SSM can be given a structural causal interpretation directly, by treating each componentwise transition equation as an autonomous mechanism that an intervention replaces while the others remain unchanged. In discrete time, a hard intervention on component i means externally fixing $x _ { t , i } = c$ while all other components evolve under their own structural equations. Structural dynamical causal models (SDCMs) [17] provide the corresponding formalism for continuous-time dynamical systems, where interventions act on differential equations. Two further questions are separate from this interpretation: whether the recovered graph is the graph of the system (identifiability), and, when the discrete-time model is meant to represent an underlying continuous-time system, how its transition graph relates to the instantaneous couplings of that system (discretization). Figure 6 summarizes the relation between SCMs, SDCMs, and SSMs.

From SCMs to SDCMs: A structural causal model (SCM) [18] assigns to each variable $X _ { i }$ a structural equation $X _ { i } = f _ { i } ( \mathrm { { P a } } ( X _ { i } ) , U _ { i } )$ with exogenous noise $U _ { i } ;$ the causal graph is defined by the parent sets $\mathrm { P a } ( X _ { i } )$ . SDCMs extend this idea to dynamical systems: each component i of the state is governed by a dynamic structural equation,

$$
\dot { X } _ { i } ( t ) = g _ { i } \big ( X _ { \mathrm { P a } ( i ) } ( t ) , U _ { i } ( t ) \big ) ,\tag{21}
$$

where $X _ { \mathrm { P a } ( i ) }$ denotes the components driving the i-th mechanism. The key property is modularity: each equation represents an autonomous mechanism that can be intervened upon independently of the others. Unlike static SCMs, SDCMs naturally accommodate feedback: cyclic mechanisms in the rolled graph $( \mathrm { e . g . , ~ } x _ { 1 }  x _ { 2 }  x _ { 3 }  x _ { 1 } )$ are allowed because the time-unrolled graph $\cdot \cdot \cdot  x _ { t , 1 }  x _ { t + 1 , 2 } $ $x _ { t + 2 , 3 }  x _ { t + 3 , 1 }  \cdot \cdot \cdot$ remains acyclic by temporal order.

Continuous and discrete time: Consider a linear SDCM ${ \dot { \pmb x } } ( t ) = { \pmb G } { \pmb x } ( t ) + { \pmb u } ( t )$ , where G is the drift matrix and ${ \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } { \mathbf { } } \mathbf { } { \mathbf { } } \mathbf { } { } { \mathbf { } } \mathbf { } { } { \mathbf { } } \mathbf { } { } \mathbf { } { } { \mathbf { } } \mathbf { } { } \mathbf { } \mathbf { } { } \mathbf { }  \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { }  \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \Xi  \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \Xi \mathbf { } \mathbf { } \mathbf { } \mathbf { } \Xi \mathbf { } \mathbf { } \mathbf { } \mathbf { } \Xi \mathbf { } \mathbf { } \mathbf { } \Xi \mathbf { } \mathbf { } \mathbf { } \Xi \mathbf { } \mathbf { } \mathbf { } \mathbf { } \Xi \mathbf { } \mathbf { } \mathbf { } \Xi \mathbf { } \mathbf { } \mathbf \Xi { } \mathbf \Xi { } \mathbf \Xi \mathbf { } \mathbf { } \mathbf \Xi \mathbf { } \Xi \mathbf { } \mathbf \Xi \mathbf { } \mathbf \Xi \Xi \mathbf { } \mathbf \Lambda \mathbf { } \mathbf \Lambda \Lambda \mathbf { } \mathbf \Lambda \mathbf { } \mathbf \Lambda \Lambda \mathbf \Lambda \mathbf { } \mathbf \Lambda \Lambda \mathbf \Lambda \Lambda \mathbf \Lambda \Lambda \mathbf  \mathbf \mathbf { \Lambda \Lambda \Lambda \mathbf \Lambda \Lambda \mathbf } \mathbf \mathbf \Lambda \mathbf \Lambda \mathbf \Lambda \mathbf \Lambda \Lambda \mathbf \Lambda \mathbf \Lambda \mathbf \Lambda \Lambda \mathbf \mathbf \Lambda \mathbf \Lambda \mathbf \Lambda \mathbf \Lambda \mathbf \Lambda \mathbf \Lambda \mathbf \Lambda \mathbf \mathbf \Lambda \mathbf \Lambda \mathbf \Lambda \mathbf \mathbf \Lambda \mathbf \mathbf \Lambda \mathbf \mathbf \Lambda \mathbf \Lambda \mathbf \mathbf \mathbf \Lambda \mathbf \Lambda \mathbf \mathbf \mathbf \Lambda \mathbf \mathbf \mathbf \Lambda \mathbf $ is a continuous-time stochastic forcing process. An Euler step gives $\pmb { A } = \pmb { I } + \Delta t \pmb { G }$ whose off-diagonal zero pattern is exactly that of G. Exact sampling instead gives $\begin{array} { r } { \pmb { A } = \exp ( \pmb { G } \Delta t ) = } \end{array}$ $\begin{array} { r } { I + \Delta t \mathbf { G } + \frac { 1 } { 2 } \Delta t ^ { 2 } \mathbf { G } ^ { 2 } + \cdot \cdot \cdot } \end{array}$ , where directed paths of length k in G contribute through the term $\Delta t ^ { k } G ^ { k } / k ! $ an entry $A _ { i j }$ with $G _ { i j } = 0$ whose shortest path from $j$ to i has length k is therefore generally nonzero, with leading order $\Delta t ^ { k }$ . Likewise, a diagonal instantaneous noise covariance generally yields a nondiagonal discrete-time noise covariance, with off-diagonal terms appearing at order $\Delta t ^ { 2 }$ . The graph of a sampled transition therefore describes dependencies accumulated over a sampling interval and need not coincide with the graph of instantaneous couplings; this can be seen as the continuous-time counterpart of subsampling a VAR process. If the drift of the cardiovascular example of Figure 2 has the fork structure $x _ { 2 } \left. x _ { 1 } \right. x _ { 3 }$ , there is no directed path between any pair of distinct components that is not already connected by an edge, so exact sampling adds no off-diagonal edge; if the drift of the five-node chain is itself a chain, exact sampling adds, e.g., the edge 1 → 3 through the term $\frac { 1 } { 2 } \Delta t ^ { 2 } ( G ^ { 2 } ) _ { 3 1 }$ . For the polynomial model, an Euler step likewise gives a coefficient matrix C with the parent structure of the continuous-time mechanism.

Requirements for a causal interpretation of the transition graph: Interpreting the directed transition graph causally requires additional assumptions:

1) Continuous-time correspondence. If the SSM is obtained by sampling an underlying continuous-time system, the graph of the exact discrete-time transition (in the linear case, $\pmb { A } = \exp ( \pmb { G } \Delta t ) )$ may contain additional edges arising from indirect paths in the instantaneous causal graph of the continuous-time system, as discussed above.

2) Independent noise components. The transition matrix A captures only directed dependencies across time. If the innovations of different state components are dependent, additional contemporaneous dependencies are present that are not represented in the transition graph. Thus, interpreting the graph of A as the full causal structure requires independent innovations; otherwise, the innovation dependence must also be represented, as in models that jointly estimate the transition and innovation graphs. If Q is block diagonal, innovations are independent across blocks but may remain dependent within each block; the transition graph alone therefore does not represent the dependencies within those blocks.

3) Causal sufficiency. The latent state $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { t } }$ must contain all causally relevant processes. Otherwise the transition parameters absorb indirect pathways and mediated effects. The hidden-intermediate-node experiment illustrates this directly: omitting $x _ { 3 }$ leads VAR-GC to infer spurious direct edges, and with $n _ { x } = 4$ GraphEM cannot recover the missing node. The choice of latent dimension $n _ { x }$ and observation operator H is therefore itself a causal modeling assumption: it defines which processes are treated as part of the causal system.

These requirements concern the causal interpretation of the transition graph and are distinct from identifiability, which determines whether the transition graph can be recovered from the observations.

## PRACTICAL GUIDELINES AND ILLUSTRATIVE EXAMPLES

## Choosing a method: decision guide

The methods surveyed in this tutorial span a broad spectrum of assumptions and complexity. The decision logic can be summarized as two sequential questions: first, whether observation-space assumptions are credible (i.e., no preprocessing-induced VARMA structure and no plausible latent confounders); second, given the chosen space, whether dimensionality (observation branch) or nonlinearity (latent branch) is the dominant constraint.

Observation-space assumptions: Observation-space methods require that the observed process admits a VAR representation and that no relevant driver is hidden. Both assumptions fail in common situations: filtering, subsampling, or averaging generally produce a VARMA process, and unobserved drivers cannot be corrected by observation-space estimators that assume causal sufficiency; methods that allow latent confounders, such as tsFCI [35] and LPCMCI [36], can flag them but do not recover the latent structure. When latent confounders are plausible, SSMs offer a way to model the latent processes explicitly. Within the observation-space setting, sparsity penalties (Lasso-VAR) and constraint-based discovery (PCMCI) are useful when $n _ { y }$ is large relative to $T$ or when the series are strongly autocorrelated; classical F-tests on OLS-VAR remain a reasonable baseline otherwise.

Identifiability and causal interpretation: Two questions should be kept apart. Identifiability asks whether the latent graph can be recovered from the data; it usually requires a known observation model or structural constraints, since sparsity penalties alone only select one representative among equivalent models. Causal interpretation asks what the graph means: it can be read causally when the transition equations are taken as structural mechanisms with their own drivers and no relevant process is omitted; otherwise it describes predictive structure in latent space.

Computational complexity and scalability: Lasso-VAR estimates $n _ { y } ^ { 2 } p$ coefficients, and PCMCI has a complexity polynomial in $n _ { y }$ and the maximum lag [14]. In latent-state methods, each Kalman filter pass costs $\mathcal { O } ( T ( n _ { x } ^ { 3 } + n _ { y } ^ { 3 } ) )$ [41]: GraphEM runs a Kalman filter and smoother at each EM iteration, DGLASSO twice per iteration [16]; both run inner solvers which may dominate the complexity. SpaRJ runs one Kalman filter pass per MCMC iteration, which accounts for most of its run time [41]. GraphGrad instead runs a differentiable particle filter at each gradient step, each pass being linear in $T$ and in the number of particles [45].

Software: The MVGC toolbox [9] implements time- and frequency-domain Granger causality for VAR models, while a state-space formulation is developed in [12]; tigramite implements PCMCI and its nonlinear extensions [14]. Reference implementations for the latent-state methods are available in the authors’ repositories.

## Hidden intermediate node experiment

We use the same five-node chain as in the measurement-noise experiment $( x _ { 1 }  x _ { 2 }  x _ { 3 }  x _ { 4 }  x _ { 5 } ,$ $A _ { i , i } = 0 . 2 , A _ { i + 1 , i } = 0 . 6 )$ , but now observe only $x _ { 1 } , x _ { 2 } , x _ { 4 } , x _ { 5 }$ . Observation noise is Gaussian with covariance 0.25 I, and we generate $T = 5 0 { , } 0 0 0$ samples. The sub-chain $x _ { 2 }  x _ { 3 }  x _ { 4 }$ is the hiddenmediator case, the counterpart of the hidden common cause in Fig. 2: the effect of $x _ { 2 }$ on $x _ { 4 }$ is real, but observation-space methods report it as a direct link that bypasses the missing node.

Observation-space methods. We apply full-conditional Granger causality with $p = 5$ lags and Bonferroni correction at $\alpha = 0 . 0 5$ (GCCA toolbox [50]), and PCMCI with linear partial-correlation tests [14]. Both recover the two true observable edges $( x _ { 1 }  x _ { 2 } , x _ { 4 }  x _ { 5 } )$ , but also infer three spurious direct edges (red in Fig. 7(b)). These are predictive dependencies induced by the hidden mediator, not finite-sample artifacts: the F-statistic of $x _ { 2 } \to x _ { 4 }$ grows from 19.1 at $T = 2 { , } 0 0 0$ to 799.8 at $T = 5 0 { , } 0 0 0$

![](images/53f963a6c33b6a44e516838fa3397933c841ebc4917e12d5f6345befa9b9257f.jpg)  
Fig. 7. Hidden intermediate node experiment. (a) True latent graph $\mathcal { G } _ { x }$ . Blue nodes are latent components indirectly measured through noisy observations; x<sub>3</sub> (gray dashed) is completely hidden. (b) Observation-space Granger causality applied to the observed processes $\pmb { y } _ { t } = ( y _ { 1 } , y _ { 2 } , y _ { 4 } , y _ { 5 } ) ^ { \top }$ . Besides the true observable edges, three spurious direct edges appear (red) due to the hidden mediator x<sub>3</sub>. (c) GraphEM reconstructs the latent transition graph using the same $\pmb { y } _ { t }$ and recovers the true $\mathcal { G } _ { x } .$ . Results are from a single run at $T = 5 0 { , } 0 0 0$

State-space approach. GraphEM [15] with $n _ { x } = 5$ , known H, and $\ell _ { 1 } { + } \mathrm { s t a b i l i t y }$ regularization $( \lambda _ { A } =$ 0.5) recovers the full latent chain including $x _ { 3 }$ (Fig. 7(c)). Declaring a GraphEM edge $j  i$ when $| \widehat { A } _ { i j } | > 0 . 0 5 , i \neq j$ (the rule used in all experiments), GraphEM attains precision, recall and $F _ { 1 }$ equal to 1, whereas Granger causality, scored against the direct observable edges $x _ { 1 } \to x _ { 2 }$ and $x _ { 4 } \to x _ { 5 }$ , has recall 1 but precision 0.40 $( F _ { 1 } = 0 . 5 7 )$ . This requires specifying the latent dimension and the observation operator correctly; with $n _ { x } = 4 .$ , GraphEM cannot recover the missing latent node.

The two approaches target different graphical objects. Observation-space Granger methods estimate the predictive graph of $\mathbf { \nabla } _ { \mathbf { \boldsymbol { y } } _ { t } } .$ , while the SSM formulation targets the latent transition graph $\mathcal { G } _ { x } .$ . All experiments in this tutorial are illustrations of specific mechanisms with known ground truth, not a benchmark. They do not address misspecified $n _ { x }$ or $H ,$ , the choice of regularization parameters without ground truth, scalability, or real data; for comparisons we refer to the original publications [14]–[16], [28], [45].

## Related approaches and their place on the causal ladder

Graph signal processing (GSP) usually processes signals on a known graph, e.g., graph VARMA models for forecasting [51]; related work on network topology inference instead learns the graph from the signals [5], [52], with variants for directed [53]–[55] and time-varying [56] graphs and for nonlinear dependencies [57]. Convergent cross mapping and directed-information tests address causal discovery in nonlinear and discrete time series [58], [59]. Most of these methods place the graph on the observed signals, i.e., on the first step of the ladder in Fig. 1. Their connection with latent state-space models is beyond the scope of this tutorial.

Other families go beyond a graph among the observed signals, but their nodes need not be physical variables. Koopman methods [60] map the state through a set of functions in which the nonlinear dynamics become approximately linear. The resulting transition graph is therefore defined over these functions, whereas in the SSMs of this tutorial the nodes are the state components. Dynamic causal modeling [61], [62] also models latent dynamics and the measurement process explicitly, but in its classical form selects among a few hypothesized graphs by Bayesian model evidence instead of learning a sparse graph. Neural Granger methods [63], in contrast, remain in the observation space, whereas temporal causal representation learning recovers latent causal processes, identifiable up to permutation and componentwise transformations under nonstationarity or non-Gaussianity assumptions [64].

## CONCLUSION AND OUTLOOK

This tutorial has examined a progression from classical Granger causality on observed time series to causal structure in latent dynamical systems. A central message is that these approaches do not estimate the same graphical object: observation-space methods target predictive structure in $\mathbf { \nabla } _ { \mathbf { \eta } _ { \left. \mathbf { \lambda } \right. } } \mathbf { \eta } _ { \left. \mathbf { \lambda } _ { \mathbf { \lambda } } \right. } \mathbf { \nabla } _ { \mathbf { \eta } _ { \left. \mathbf { \lambda } \right. } } \mathbf { \eta } _ { \left. \mathbf { \lambda } _ { \mathbf { \lambda } } \right. } \mathbf { \nabla } _ { \mathbf { \eta } _ { \left. \mathbf { \lambda } \right. } } \mathbf { \eta } _ { \left. \mathbf { \lambda } _ { \lambda } \right. } \mathbf { \nabla } _ { \mathbf { \eta } _ { \left. \mathbf { \lambda } \right. } } \mathbf { \eta } _ { \left. \mathbf { \lambda } _ { \lambda } \right. }$ while state-space formulations target the transition structure of an assumed model with latent structure, which in many applications is closer to the physical mechanisms that generate the data. Each step along this progression requires stronger assumptions: separating latent dynamics from observation effects, identifying the latent representation, and reading the transition equations as structural mechanisms. When the latent model samples an underlying continuous-time system, the recovered graph describes dependencies accumulated over one sampling interval, which may include indirect paths of the instantaneous causal graph. Specifying the latent representation is itself a causal modeling assumption.

An important open problem is the extension of the last two steps of the causal ladder to nonlinear systems: while identifiability is comparatively well characterized in linear Gaussian SSMs, it is unclear under which conditions nonlinear latent dynamical systems admit identifiable and causally interpretable representations.

## REFERENCES

[1] C. W. Granger, “Investigating causal relations by econometric models and cross-spectral methods,” Econometrica: Journal of the Econometric Society, vol. 37, no. 3, pp. 424–438, 1969.

[2] A. Porta and L. Faes, “Wiener–Granger causality in network physiology with applications to cardiovascular control and neuroscience,” Proceedings of the IEEE, vol. 104, no. 2, pp. 282–309, 2016.

[3] V. Elvira, E. Chouzenoux, J. Cerda, and G. Camps-Valls, “Graphs in state-space models for granger causality in climate science,” arXiv preprint arXiv:2307.10703, 2023.

[4] V. N. Ioannidis, Y. Shen, and G. B. Giannakis, “Semi-blind inference of topologies and dynamical processes over dynamic graphs,” IEEE Transactions on Signal Processing, vol. 67, no. 9, pp. 2263–2274, 2019.

[5] G. Mateos, S. Segarra, A. G. Marques, and A. Ribeiro, “Connecting the dots: Identifying network structure via graph signal processing,” IEEE Signal Processing Magazine, vol. 36, no. 3, pp. 16–43, 2019.

[6] J. Runge, A. Gerhardus, G. Varando, V. Eyring, and G. Camps-Valls, “Causal inference for time series,” Nature Reviews Earth & Environment, vol. 4, no. 7, pp. 487–505, 2023.

[7] A. Shojaie and E. B. Fox, “Granger causality: A review and recent advances,” Annual Review of Statistics and Its Application, vol. 9, no. 1, pp. 289–319, 2022.

[8] H. Lutkepohl,¨ New Introduction to Multiple Time Series Analysis, Springer, Berlin, 2005.

[9] L. Barnett and A. K. Seth, “The MVGC multivariate Granger causality toolbox: A new approach to Granger-causal inference,” Journal of Neuroscience Methods, vol. 223, pp. 50–68, 2014.

[10] S. Basu and G. Michailidis, “Regularized estimation in sparse high-dimensional time series models,” The Annals of Statistics, pp. 1535–1567, 2015.

[11] S. Basu, X. Li, and G. Michailidis, “Low rank and structured modeling of high-dimensional vector autoregressions,” IEEE Transactions on Signal Processing, vol. 67, no. 5, pp. 1207–1222, 2019.

[12] L. Barnett and A. K. Seth, “Granger causality for state-space models,” Physical Review E, vol. 91, no. 4, pp. 040101, 2015.

[13] D. Paulin and V. Elvira, “Scalable estimation of varma models,” arXiv preprint arXiv:2608.06340, 2026.

[14] J. Runge, P. Nowack, M. Kretschmer, S. Flaxman, and D. Sejdinovic, “Detecting and quantifying causal associations in large nonlinear time series datasets,” Science advances, vol. 5, no. 11, pp. eaau4996, 2019.

[15] V. Elvira and E. Chouzenoux, “Graphical inference in linear-Gaussian state-space models,” <sup>´</sup> IEEE Trans. Signal Process., vol. 70, pp. 4757–4771, 2022.

[16] E. Chouzenoux and V. Elvira, “Sparse graphical linear dynamical systems,” Journal of Machine Learning Research, vol. 25, no. 223, pp. 1–53, 2024.

[17] S. Bongers, T. Blom, and J. M. Mooij, “Causal modeling of dynamical systems,” arXiv preprint arXiv:1803.08784, 2018.

[18] J. Pearl, Causality, Cambridge university press, 2009.

[19] P. E. Caines, “Weak and strong feedback free processes,” IEEE Transactions on Automatic Control, vol. 21, no. 5, pp. 737–739, 1976.

[20] V. Solo, “On causality i: sampling and noise,” in 2007 46th IEEE Conference on Decision and Control. IEEE, 2007, pp. 3634–3639.

[21] J. F. Geweke, “Measures of conditional linear dependence and feedback between time series,” Journal of the American Statistical Association, vol. 79, no. 388, pp. 907–915, 1984.

[22] R. E. Kalman, “A new approach to linear filtering and prediction problems,” Journal of Basic Engineering, vol. 82, pp. 35–45, 1960.

[23] S. Sarkk ¨ a and L. Svensson, ¨ Bayesian filtering and smoothing, vol. 17, Cambridge university press, 2023.

[24] L. Ljung, System Identification: Theory for the User, Prentice Hall, Upper Saddle River, NJ, 2nd edition, 1999.

[25] A. C. Lozano, N. Abe, Y. Liu, and S. Rosset, “Grouped graphical Granger modeling methods for temporal causal modeling,” in Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2009, pp. 577–586.

[26] A. Bolstad, B. D. Van Veen, and R. Nowak, “Causal network inference via group sparse regularization,” IEEE Transactions on Signal Processing, vol. 59, no. 6, pp. 2628–2641, 2011.

[27] S. Basu, A. Shojaie, and G. Michailidis, “Network granger causality with inherent grouping structure,” The Journal of Machine Learning Research, vol. 16, no. 1, pp. 417–453, 2015.

[28] D. Luengo, G. Rios-Munoz, V. Elvira, C. Sanchez, and A. Artes-Rodriguez, “Hierarchical algorithms for causality retrieval in atrial fibrillation intracavitary electrograms,” IEEE Journal of Biomedical and Health Informatics, vol. 23, no. 1, pp. 143–155, 2019.

[29] T. Schreiber, “Measuring information transfer,” Physical Review Letters, vol. 85, no. 2, pp. 461–464, 2000.

[30] L. Barnett, A. B. Barrett, and A. K. Seth, “Granger causality and transfer entropy are equivalent for gaussian variables,” Physical review letters, vol. 103, no. 23, pp. 238701, 2009.

[31] J. Runge, “Discovering contemporaneous and lagged causal relations in autocorrelated nonlinear time series datasets,” in Conference on uncertainty in artificial intelligence. Pmlr, 2020, pp. 1388–1397.

[32] H. Boudjellaba, J.-M. Dufour, and R. Roy, “Testing causality between two vectors in multivariate autoregressive moving average models,” Journal of the American Statistical Association, vol. 87, no. 420, pp. 1082–1090, 1992.

[33] R. L. Binks, S. E. Heaps, M. Panagiotopoulou, Y. Wang, and D. J. Wilkinson, “Bayesian inference on the order of stationary vector autoregressions,” Bayesian Analysis, 2024, Advance publication.

[34] P. Spirtes, C. Glymour, and R. Scheines, Causation, Prediction, and Search, MIT Press, Cambridge, MA, 2nd edition, 2000.

[35] D. Entner and P. O. Hoyer, “On causal discovery from time series data using FCI,” in Proceedings of the Fifth European Workshop on Probabilistic Graphical Models (PGM), 2010, pp. 121–128.

[36] A. Gerhardus and J. Runge, “High-recall causal discovery for autocorrelated time series with latent confounders,” Advances in neural information processing systems, vol. 33, pp. 12615–12625, 2020.

[37] E. Chouzenoux and V. Elvira, “Graphical inference in non-Markovian linear-Gaussian state-space models,” in Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2024), Seoul, South Korea, Apr. 2024.

[38] R. H. Shumway and D. S. Stoffer, “An approach to time series smoothing and forecasting using the EM algorithm,” Journal of Time Series Analysis, vol. 3, no. 4, pp. 253–264, 1982.

[39] J. Friedman, T. Hastie, and R. Tibshirani, “Sparse inverse covariance estimation with the graphical lasso,” Biostatistics, vol. 9, no. 3, pp. 432–441, 2008.

[40] S. L. Lauritzen, Graphical models, Number 17 in Oxford Statistical Science Series. Oxford University Press, 1996.

[41] B. Cox and V. Elvira, “Sparse bayesian estimation of parameters in linear-gaussian state-space models,” IEEE Transactions on Signal Processing, vol. 71, pp. 1922–1937, 2023.

[42] P. J. Green, “Reversible jump Markov chain Monte Carlo computation and Bayesian model determination,” Biometrika, vol. 82, no. 4, pp. 711–732, 1995.

[43] N. Kantas, A. Doucet, S. S. Singh, J. Maciejowski, and N. Chopin, “On particle methods for parameter estimation in state-space models,” Statistical Science, vol. 30, no. 3, pp. 328–351, 2015.

[44] S. L. Brunton, J. L. Proctor, and J. N. Kutz, “Discovering governing equations from data by sparse identification of nonlinear dynamical systems,” Proceedings of the national academy of sciences, vol. 113, no. 15, pp. 3932–3937, 2016.

[45] B. Cox, E. Chouzenoux, and V. Elvira, “Graphgrad: Efficient estimation of sparse polynomial representations for general <sup>´</sup> state-space models,” IEEE Transactions on Signal Processing, 2025.

[46] A. Bryutkin, M. E. Levine, I. Urteaga, and Y. Marzouk, “Canonical bayesian linear system identification,” arXiv preprint arXiv:2507.11535, 2025.

[47] P. Gustafson, “What are the limits of posterior distributions arising from nonidentified models, and why should we care?,” Journal of the American Statistical Association, vol. 104, no. 488, pp. 1682–1695, 2009.

[48] P. Zhao and B. Yu, “On model selection consistency of Lasso,” Journal of Machine Learning Research, vol. 7, pp. 2541–2563, 2006.

[49] M. J. Wainwright, “Sharp thresholds for high-dimensional and noisy sparsity recovery using $\ell _ { 1 }$ -constrained quadratic programming (Lasso),” IEEE Transactions on Information Theory, vol. 55, no. 5, pp. 2183–2202, 2009.

[50] A. K. Seth, “A MATLAB toolbox for Granger causal connectivity analysis,” Journal of Neuroscience Methods, vol. 186, no. 2, pp. 262–273, 2010.

[51] E. Isufi, A. Loukas, N. Perraudin, and G. Leus, “Forecasting time series with VARMA recursions on graphs,” IEEE Transactions on Signal Processing, vol. 67, no. 18, pp. 4870–4885, 2019.

[52] X. Dong, D. Thanou, M. Rabbat, and P. Frossard, “Learning graphs from data: A signal representation perspective,” IEEE Signal Processing Magazine, vol. 36, no. 3, pp. 44–63, 2019.

[53] J. Mei and J. M. F. Moura, “Signal processing on graphs: Causal modeling of unstructured data,” IEEE Transactions on Signal Processing, vol. 65, no. 8, pp. 2077–2092, Apr. 2017.

[54] A. G. Marques, S. Segarra, and G. Mateos, “Signal processing on directed graphs: The role of edge directionality when processing and learning from network data,” IEEE Signal Processing Magazine, vol. 37, no. 6, pp. 99–116, 2020.

[55] C. Cui, P. Banelli, and P. M. Djuric, “Topology inference of directed graphs by Gaussian processes with sparsity constraints,”´ IEEE Transactions on Signal Processing, vol. 72, pp. 2147–2159, 2024.

[56] B. Zaman, L. M. Lopez-Ramos, D. Romero, and B. Beferull-Lozano, “Online topology identification from vector autoregressive time series,” IEEE Transactions on Signal Processing, vol. 69, pp. 210–225, 2021.

[57] G. Giannakis, Y. Shen, and G. Karanikolas, “Topology identification and learning over graphs: Accounting for nonlinearities and dynamics,” Proceedings of the IEEE, vol. 106, no. 5, pp. 787–807, May 2018.

[58] K. Butler, G. Feng, and P. M. Djuric, “On causal discovery with convergent cross mapping,” ´ IEEE Transactions on Signal Processing, vol. 71, pp. 2595–2607, 2023.

[59] A. Theocharous, G. G. Gregoriou, P. Sapountzis, and I. Kontoyiannis, “Temporally causal discovery tests for discrete time series and neural spike trains,” IEEE Transactions on Signal Processing, vol. 72, pp. 1333–1347, 2024.

[60] S. L. Brunton, M. Budisiˇ c, E. Kaiser, and J. N. Kutz, “Modern Koopman theory for dynamical systems,” ´ SIAM Review, vol. 64, no. 2, pp. 229–340, 2022.

[61] K. J. Friston, L. Harrison, and W. Penny, “Dynamic causal modelling,” NeuroImage, vol. 19, no. 4, pp. 1273–1302, 2003.

[62] K. J. Friston, A. M. Bastos, A. Oswal, B. Van Wijk, C. Richter, and V. Litvak, “Granger causality revisited,” Neuroimage, vol. 101, pp. 796–808, 2014.

[63] A. Tank, I. Covert, N. Foti, A. Shojaie, and E. B. Fox, “Neural Granger causality,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 8, pp. 4267–4279, 2022.

[64] W. Yao, Y. Sun, A. Ho, C. Sun, and K. Zhang, “Learning temporally causal latent processes from general temporal data,” in International Conference on Learning Representations (ICLR), 2022.

## BIOGRAPHY

V´ıctor Elvira (Senior Member, IEEE) is a Distinguished Research Professor at the Basque Centre for Applied Mathematics (BCAM), Bilbao, Spain, and a Professor with a Personal Chair in Statistics and Data Science at the School of Mathematics, University of Edinburgh. He received his Ph.D. in statistical signal processing from the University of Cantabria (Spain) in 2011. His research interests are in statistical signal processing, computational statistics, and probabilistic machine learning, with a focus on state-space models, Monte Carlo methods, and graph-based formulations for time-series data. He is an ELLIS Fellow, Fulbright Fellow, Marie Curie Fellow, Leverhulme Fellow, Alan Turing Fellow, and recipient of the 2024 EURASIP Early Career Award. He is the Principal Investigator of an ARIA-funded project (2025–2028) on probabilistic forecasting of climate tipping points.