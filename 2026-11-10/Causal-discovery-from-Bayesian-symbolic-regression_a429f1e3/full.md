# Causal discovery from Bayesian symbolic regression

Roger Guimerà,1, 2, 3, \* Kai-Hendrik Cohrs,4 Sergio Pedreira,1 Marta Sales-Pardo,1, 2, † and Gustau Camps-Valls4,

1Department of Chemical Engineering, Universitat Rovira i Virgili, Tarragona, Spain

2Center for Computational Science and Applied Mathematics (ComSCIAM), Universitat Rovira i Virgili, Tarragona, Spain 3ICREA, Barcelona, Spain

4Image Processing Laboratory (IPL), Universitat de València, València, Spain

Causal discovery from observational data is central to scientific reasoning, yet established constraint-based and score-based methods often recover only a Markov equivalence class and provide no signal in bivariate or strongly autocorrelated regimes typical in some areas of science. Here, we recast causal discovery as a Bayesian symbolic regression problem. Given a set of candidate parents for each target variable, the Bayesian symbolic regression samples closed-form expressions from a posterior over symbolic models. The fraction of sampled expressions in which each candidate variable appears defines a posterior inclusion probability that we use as the causal score. This replaces pointwise model selection by an ensemble of plausible mechanisms. We evaluate the method on coupled nonlinear bivariate dynamics, on the CauseMe benchmark suite spanning linear, chaotic, nonlinear, higher-dimensional and high-noise settings, and on a four-variable biogeoscience structural causal model under Gaussian, heavy-tailed and mixed-noise regimes. Our approach is competitive with, and often outperforms, specialized causal discovery algorithms. Additionally, the method returns closedform mechanisms leading to explicit causality and interpretability.

## I. INTRODUCTION

Scientific understanding is built on mechanisms that describe how interventions on one part of a system propagate through the rest. Moving from statistical association to structural causation is a long-standing endeavor that has reshaped many disciplines1-3. In disciplines where true experiments are seldom possible, causal reasoning is essential to attribute change to forcings, to identify indirect connections, and to evaluate process-based and data-driven models on physical and mechanistic rather than purely predictive grounds. This is the case for Earth and climate sciences4–6, neuroscience7,8, epidemiology9, and economics10, to name a few examples.

Dominant causal discovery paradigms rely on conditionalindependence tests over a presumed faithful distribution and return graphs only up to a Markov equivalence class2,11. In the bivariate setting, where one wishes to distinguish X → Y from Y → X, the marginal independence structure is the same for both directions and constraint-based methods provide no signal. To break this symmetry, one can exploit structural restrictions on the functional form (additive noise models12,13, LiNGAM14), invariance under environments15,16, or algorithmic information principles17–19. Approaches in the latter family can assign complexity to causal graphs—usually measured as the causal graph Kolmogorov complexity approximated through kernel embeddings20,21 or stochastic codes22,23and can therefore use the minimum description length (MDL) principle for causal graph selection. Still other approaches operate in the time domain through Granger causality24, convergent cross mapping25, and conditional-independence variants tailored to autocorrelation26,27.

Here, we argue that probabilistic symbolic regression approaches28,29 offer a better solution to causal discovery, even in the bivariate case. Indeed, from dynamical systems, we know that the equilibrium states of a first-order ordinary differential (ODE) system on which we intervene can be recovered using a structural causal model (SCM)30. Interestingly, under standard stability assumptions, recovering the differential (or difference) equation that governs a process therefore amounts to recovering the SCM of its equilibrium or stationary regime. This result suggests that symbolic regression is then not only a predictive surrogate but a constructive route to the structural mechanisms targeted by causal inference6.

With this motivation in mind, here we use methods that explore the space of closed-form mathematical expressions to find causal relationships. In particular, we use the Bayesian machine scientist (BMS)28, which explores this space with a Markov chain Monte Carlo (MCMC) sampler over symbolic expressions and assigns each expression a plausibility based on its description length³1. The output is a sample taken from the posterior distribution of plausible mechanisms. We define the causal score of a candidate parent variable in the causal graph as the variable's posterior inclusion frequency, that is, the fraction of sampled expressions in which the variable appears. This score handles many candidate parents and lagged predictors, is robust to high noise levels, accommodates autocorrelation, and remains informative in regimes where several expressions fit the data. Besides, our method simultaneously delivers a causal graph and explicit functional forms for plausible mechanisms.

This paper makes three contributions. First, a formalization of causal discovery as Bayesian symbolic regression, in which the causal score of each candidate parent is its posterior inclusion frequency in the ensemble of sampled mechanisms. Second, an identifiability analysis that connects causal recovery to the recently characterized learnability transition of probabilistic model selection³2, separating a learnable low-noise regime, a transitional regime, and an unlearnable high-noise regime. Third, an evaluation on a wide variety of problems, from bivariate nonlinear dynamics, to biogeoscience structural causal models, and standard benchmark suites26,27. In all of them causalBMS attains the highest or very competitive performance compared to specialized methods elsewhere, while additionally returning the closed-form mechanism that produced the data or very good approximations to those.

## II. CAUSAL SCORES FROM BAYESIAN SYMBOLIC REGRESSION

## A. Bayesian symbolic regression from time series

We consider a vector of observable variables $\mathbf { x } ( t ) \in \mathbb { R } ^ { d } , t =$ $1 , \ldots , T$ . Each component $x ^ { i }$ is assumed to be generated by a structural equation33

$$
x ^ { i } ( t ) = f _ { i } ^ { * } \bigl ( \mathbf { x } _ { \mathrm { P a } ( i ) } ( t - 1 ) , \ldots , \mathbf { x } _ { \mathrm { P a } ( i ) } ( t - \tau _ { \mathrm { m a x } } ) \bigr ) + \varepsilon ^ { i } ( t ) ,\tag{1}
$$

with $\operatorname { P a } ( i ) \subseteq \{ 1 , \ldots , d \}$ the set of causal parents of $x ^ { i }$ and $\varepsilon ^ { i } \sim \mathcal { N } ( 0 , \sigma )$ an exogenous Gaussian noise term assumed mutually independent across i and t. In this setting, the goal of causal discovery is to recover $\{ \mathrm { P a } ( i ) \} _ { i = 1 } ^ { d }$ , or its lagged extension, from the data $\boldsymbol { \mathcal { D } } = \{ { \mathbf { x } } ( t ) \} _ { t = 1 } ^ { T } .$ In contrast, the goal of symbolic regression is originally to recover $f _ { i } ^ { * }$ in closed form. The key observation is that, under the stated conditions, causal discovery becomes easier if one first solves the symbolic regression problem.

Indeed, we note that Eq. (1) also admits a structural causal interpretation. For a deterministic dynamical system $\dot { \mathbf { x } } = \mathbf { F } ( \mathbf { x } )$ with a globally stable equilibrium, the parent set induced by F at equilibrium defines an SCM that is consistent with the do-calculus on perfect interventions30. The same construction applies to discrete-time systems satisfying Eq. (1) under standard stationarity conditions. Recovering $f _ { i } ^ { * }$ therefore yields, under these assumptions, both a predictive equation and the structural mechanism that defines the causal effect of intervening on the parents of $x ^ { i } .$ This intuition that symbolic regression drives causal discovery can thus be properly formalized.

The space of admissible expressions for the i-th component is denoted $\mathcal { F } _ { i }$ Following standard practice in symbolic regression28,29,32, we represent closed-form expressions $f _ { i } \in \mathcal { F } _ { i }$ as labeled trees whose internal nodes are operators drawn from a fixed alphabet $\mathcal { O } = \{ + , \times , \exp , \log , \sin , . . . \}$ and whose leaves are variables or real parameters $\pmb { \theta } \in \Theta _ { f _ { i } } \subseteq \mathbb { R } ^ { k _ { f _ { i } } }$ Such alphabet is often selected (restricted) by domain knowledge and process understanding. In symbolic regression approaches such as $\mathrm { S I N D y } ^ { 3 4 }$ , the model is restricted to linear combinations of certain basis functions. The selection of these basis functions is critical and, in general, limits the expressiveness of the approach35. By contrast, probabilistic symbolic regression and the BMS28,29 impose no such restrictions and allow any function combination and composition from the alphabet.

For a target $x ^ { i }$ and an expression $f _ { i } \in \mathcal { F } _ { i } ,$ the BMS uses probabilistic arguments to assign to each candidate expression $f _ { i } \in \mathcal { F } _ { i }$ the posterior probability

$$
\begin{array} { r l } { p ( f _ { i } | \mathcal { D } ) \ = \ \displaystyle \frac { 1 } { p ( \mathcal { D } ) } \int _ { \Theta _ { f _ { i } } } p ( \mathcal { D } | f _ { i } , \theta ) p ( \theta | f _ { i } ) p ( f _ { i } ) \mathrm { d } \theta } & { } \\ { \displaystyle \ = \ \frac { \exp \big [ - \mathcal { H } ( f _ { i } ) \big ] } { Z } , } \end{array}\tag{2}
$$

with description length (or information Hamiltonian $^ { 1 ^ { 3 6 } ) }$ $\mathcal { H } ( f _ { i } ) = - \log p ( \mathcal { D } , f _ { i } )$ , evidence (or partition function) $Z =$ $p ( \mathcal { D } )$ , parameter prior $p ( \theta | f _ { i } )$ , and structural prior $p ( f _ { i } ) ^ { 2 8 , 3 7 }$ The structural prior can either be learned from a corpus of mathematical expressions extracted from scientific texts under a maximum-entropy assumption $^ { 1 ^ { 2 8 } }$ or chosen to encode other forms of knowledge about the target expression $^ { 3 8 }$ . Under the assumption of Gaussian exogenous noise $\varepsilon ^ { i } ( t ) \sim \mathcal { N } ( 0 , \sigma )$ and using Laplace's approximation around the maximumlikelihood estimator $\theta ^ { * } =$ arg maxθ $p ( \mathcal { D } | f _ { i } , \theta _ { f _ { i } } )$ , the description length is

$$
\mathcal { H } ( f _ { i } ) \approx \frac { B ( f _ { i } ) } { 2 } - \log p ( f _ { i } ) ,\tag{3}
$$

where $B ( f _ { i } ) = - 2 \log p ( \mathcal { D } | f _ { i } , \theta ^ { * } ) + ( k _ { f _ { i } } + 1 )$ log T is the Bayesian information criterion $^ { 3 9 }$ of $f _ { i } , k _ { f _ { i } }$ is the number of parameters of $f _ { i } { } ^ { 4 0 }$ , and T is the number of observed time points. Note that Eq. (3) makes the connection with the MDL twopart code $3 7$ explicit: $B ( f _ { i } ) / 2$ encodes data and parameters, while $- \log p ( f _ { i } )$ encodes the structural complexity of the expression.

The BMS explores the space $\mathcal { F } _ { i }$ by generating a Metropolis chain over expression trees with three local moves: root addition or removal, elementary-tree replacement, and node replacement28. Parallel tempering across a temperature ladder $\left\{ T _ { k } \right\}$ stabilizes the exploration of the rugged $\bar { \mathcal { H } } { - } \mathrm { l a n d s c a p e } ^ { 2 8 }$ and the chain returns a set $\mathcal { S } _ { i } = \{ f _ { i } ^ { ( j ) } \} _ { j = 1 } ^ { M }$ of expressions sampled from $p ( f _ { i } | \mathcal { D } )$ . Note that this is a key aspect of our approach, as averages over the chain of models $\mathcal { S } _ { i }$ are an approximation to an average over the posterior, which is in general a better estimator of any observable than those provided by any single mode $1 ^ { 2 9 }$

## B. From sampled expressions to causal scores

Each sampled expression $f _ { i } ^ { ( j ) } \in \mathcal { S } _ { i }$ specifies a set of parents that plausibly enter the causal mechanism for $x ^ { i } .$ Pooling the expressions in the chain $\mathcal { S } _ { i }$ yields a posterior estimate of the probability that a candidate parent participates in the generating equation. Specifically, for a target $x ^ { i }$ and a candidate parent $x ^ { k } ( t - \tau )$ , the posterior inclusion frequency is

$$
\pi _ { k \to i } ( \tau ) = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \mathbb { I } \Big [ x ^ { k } ( t - \tau ) \in \mathcal { L } \big ( f _ { i } ^ { ( j ) } \big ) \Big ] ,\tag{4}
$$

where $\mathcal { L } ( f )$ denotes the set of leaves of expression $f$ that are observable variables, I is the indicator function, and the sum is over the sampled expressions in $\mathcal { S } _ { i }$ Marginalizing over lags,

Algorithm 1. causalBMS: Causal discovery from Bayesian symbolic regression

Require: Data $\boldsymbol { \mathcal { D } } = \{ \mathbf { x } ( t ) \} _ { t = 1 } ^ { T } ;$ alphabet ${ \mathcal { O } } ;$ corpus-derived prior $p ( f ) ;$ maximum lag $\tau _ { \mathrm { { m a x } } } ;$ chain length M; tempering ladder $\{ T _ { k } \} _ { k = 1 } ^ { K }$

Ensure: Causal score matrix û $\in [ 0 , 1 ] ^ { d \times d }$ and ensembles of equations $\{ \mathcal { S } _ { i } \} _ { i = 1 } ^ { d }$

1: for $i = 1$ to d do

Construct candidate predictor set $\mathcal { X } _ { i } = \{ x ^ { k } ( t - \tau ) : k \in$ $\{ 1 , \dots , d \} , \tau \in \{ 0 , \dots , \tau _ { \operatorname* { m a x } } \} \} \backslash \{ x ^ { i } ( t ) \}$

3: Run parallel-tempered Metropolis MCMC on $\mathcal { F } _ { i }$ targeting $p ( f | \mathcal { D } ) \propto \exp [ - \mathcal { H } ( f ) ]$ via Eq. (3).

4:Collect post-burn-in samples $\mathcal { S } _ { i } = \{ f _ { i } ^ { ( j ) } \} _ { j = 1 } ^ { M }$ from the cold replica.

5: for $k = 1$ to $d , k \neq i$ do

6: $\begin{array} { r } { \pi _ { k \to i }  \frac { 1 } { M } \sum _ { j } \mathbb { I } \Big [ \exists \tau : x ^ { k } ( t - \tau ) \in \mathrm { V a r } ( f _ { i } ^ { ( j ) } ) \Big ] . } \end{array}$

7: end for

8: end for

9: return $\Pi = ( \pi _ { k  i } ) _ { k , i }$ and the ensembles $\mathcal { S } _ { i } .$

we obtain a variable-level score independent of lag

$$
\pi _ { k \to i } = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \mathbb { I } \Big [ \exists \boldsymbol { \tau } : \boldsymbol { x } ^ { k } ( t - \boldsymbol { \tau } ) \in \mathcal { L } \big ( f _ { i } ^ { ( j ) } \big ) \Big ] ,\tag{5}
$$

which we use as the score of the inferred causal relationship $k  i . \mathrm { ~ \ } \mathrm { \bf ~ B y }$ thresholding $\pi _ { k \to i }$ for all variables, we obtain a directed causal graph; by ranking edges in the causal graph by $\pi _ { k \to i } ,$ we obtain precision-recall and ROC curves. The full procedure is summarized in Algorithm 1.

## C. Consistency, equation learnability and identifiability of causal parents

The description length is a consistent model-selection criterion, meaning that, as $T \to \infty ,$ the posterior $p ( f _ { i } | \mathcal { D } )$ concentrates on the model that minimizes $\mathcal { H } ( f _ { i } )$ over $\mathcal { F } _ { i } ,$ which happens to be the ground-truth model $f _ { i } ^ { * }$ as long as $f _ { i } ^ { * } \in$ $\mathcal { F } _ { i } ^ { 3 \bar { 2 } , 3 7 , 3 9 }$ . We summarize the implications for the inclusionfrequency score.

Proposition 1 (Asymptotic recovery of the causal parents). Assume that (i) the true mechanism $f _ { i } ^ { * }$ is in $\mathcal { F } _ { i } ;$ (ii) the prior $p ( f _ { i } )$ assigns strictly positive mass to $f _ { i } ^ { * }$ ; (iii) the exogenous noise σ is fnite; and (iv) the MCMC sampler is ergodic. Then $\pi _ { k \to i } \to \mathbb { I } [ k \in \mathrm { P a } ( i ) ]$ with probability approaching one as $T $ ∞and $M \to \infty .$

For a fixed and finite T, however, recovery is not guaranteed at all noise levels even when assumptions $( i ) , ( i i )$ , and (iv) above hold. Indeed, Fajardo-Fontiveros et al.32 show that probabilistic model selection over closed-form expressions exhibits a sharp transition between a low-noise regime in which the true model has the smallest description length and a high-noise regime in which a trivial model becomes more plausible. Since probabilistic model selection is Bayes optimal, no other method will be able to recover the true generating model in the high-noise regime. An upper bound (that is typically tight) to the transition noise is

$$
\pmb { \sigma } ^ { \times } \approx \left( \frac { \langle \delta ^ { 2 } \rangle T } { 2 \Delta _ { M } ^ { * } + ( k _ { i } ^ { * } - 1 ) \log T } \right) ^ { 1 / 2 } ,\tag{6}
$$

where $k _ { i } ^ { * }$ is the number of parameters of $f _ { i } ^ { * } , \Delta _ { M } ^ { * } = \log p ( f _ { i } ^ { c } ) -$ lo $\mathrm { g } p ( f _ { i } ^ { * } )$ is the gap in structural description length between $f _ { i } ^ { * }$ and the trivial model $f _ { i } ^ { c } = \mathrm { a r g } \operatorname* { m a x } _ { f _ { i } } p ( f _ { i } )$ (which, in our maximum entropy prior corresponds to a constant $f _ { i } ^ { c } = \theta _ { 0 } )$ and $\langle { \delta ^ { 2 } } \rangle$ is the variance of $f _ { i } ^ { * }$ over the support of the predictors $3 \dot { 2 }$ . This leads us to a second proposition.

Proposition 2 (Assymptotic causal unidentifiability). For $\cdot \ f -$ nite T and $\sigma \to \infty ,$ the posterior $p ( f _ { i } | \mathcal { D } )$ concentrates around the trivial model $f _ { i } ^ { c } = \mathrm { a r g } \operatorname* { m a x } _ { f _ { i } } p ( f _ { i } )$ , the inclusion frequencies $\pi _ { k \to i }$ approach the prior expectations $\mathbb { E } _ { p ( f _ { i } ^ { c } ) } \big [ \mathbb { I } [ x ^ { k } \in$ $\mathcal { L } ( f _ { i } ^ { c } ) ] ]$ independently of the true $\mathrm { P a } ( i )$ , and the causal score loses al discriminative power.

Yet, whether causal parents can be detected at finite $\sigma >$ $\sigma ^ { \times }$ , where the exact true model cannot possibly be identified from the data alone, is an open empirical question—in principle, it is possible that, even when the ground-truth model $f _ { i } ^ { * }$ is unlearnable, some other model with the correct causal structure maximizes the posterior. We return to this issue below in section III A.

## III. EXPERIMENTAL RESULTS

We evaluate our causal discovery approach based on BMS (causalBMS) across a wide range of datasets, to understand how the approach performs under relevant scenarios of nonlinear processes, strong couplings, non-stationarity, highdimensional scenarios, and different levels and diversity of noise sources. All aspects have been identified as critical for causal discovery⁴. In particular, we evaluate the method on: (i) a coupled nonlinear bivariate dynamical system; (ii) the CauseMe benchmark suite spanning linear, chaotic, nonlinear, high-dimensional and high-noise settings; and (iii) a biogeoscience structural causal model of shortwave radiation, air temperature, gross primary productivity and ecosystem respiration under Gaussian, heavy-tailed and mixed-noise regimes.

We compare causalBMS to standard methods in the literature $( \mathrm { P C M C I ^ { + 2 6 , 4 1 } }$ $\mathrm { V A R - L i N G A M ^ { 1 4 , 4 2 } }$ , convergent cross mapping $\begin{array} { r l } {  { ( \mathbf { C C M } ) ^ { 2 5 } } } \end{array}$ , and a wide variety of methods in the CauseMe platform leaderboard²7), and through standard metrics of performance (F1 score, precision-recall scores and the ROC AUC metric). We also study the robustness of symbolic regression to noise in the coupled nonlinear bivariate dynamical system and in the biogeoscience selected case study.

The experimental setup requires running causalBMS multiple times and averaging the results. Computational cost is dominated by the MCMC sweep across targets; in our experiments we use a minimum of $\bar { M } \sim 1 0 ^ { 4 }$ retained samples after burn-in, parallel tempering with a minimum of $K = 1 0$ replicas, and a tree-size cap $| f _ { i } | \le 5 0$ nodes.

![](images/40f45928b8cc7a323db2da8676729a3b2b3eb042574ef0dea2c45508da94cc20.jpg)  
FIG. 1. Causal and functional recovery in unidirectionally coupled nonlinear dynamics. (a-d) Sample observed trajectories xt and $y _ { t }$ for the system in Eq. (7). (e-h) Causal scores $\pi _ { k \to i } ( \tau )$ obtained by causalBMS from Eq. (4). (i-1) Minimum description length expressions sampled by causalBMS (green) reproduce the true self-influence (black). Grey points are the $N = 2 0 0 0$ observations. Equation labels in the legend are the closed-form expressions inferred by the chain. We consider four noise regimes: low noise $( \sigma = 0 . 0 4 $ , first column); moderate noise $( \sigma = 0 . 4 ,$ second column); high noise $( \sigma = 4 ,$ third column); very high noise $( \sigma = 1 0 ^ { \circ }$ , fourth column).

## A. Causal and functional recovery in unidirectionally coupled nonlinear dynamics

We start by considering a bivariate coupled system that combines self-excitation with a unidirectional coupling and is governed by the following equations

$$
\begin{array} { r l } & { x _ { t } = 3 . 4 x _ { t - 1 } \bigl ( 1 - x _ { t - 1 } ^ { 2 } \bigr ) \exp ( - x _ { t - 1 } ^ { 2 } ) + \varepsilon _ { t } ^ { x } , } \\ & { y _ { t } = 3 . 4 y _ { t - 1 } \bigl ( 1 - y _ { t - 1 } ^ { 2 } \bigr ) \exp ( - y _ { t - 1 } ^ { 2 } ) + 0 . 5 x _ { t - 1 } y _ { t - 1 } + \varepsilon _ { t } ^ { y } , } \end{array}\tag{7}
$$

with $\varepsilon \sim \mathcal { N } ( 0 , \sigma )$ The causal pair $( X \to Y , Y \not \to X )$ is hard for linear and Granger-style methods, which see strong autocorrelation in both series. Indeed, for moderate levels of noise $\sigma ^ { 2 } = 0 . 4 ,$ these methods fail to detect the correct causal links6. Here, we investigate whether our Bayesian symbolic regression approach can detect the correct causal relationships and governing equations at different noise levels (Fig. 1). The levels of noise we investigate go from low, where autocorrelation and cross-correlation between both time series are strong (Fig. 1a), to very high, where autocorrelation in the $x _ { t }$ time series is much weaker, and the dynamics of $y _ { t }$ is dominated by some sporadic large bursts (Fig. 1d). In all cases, we let causalBMS explore time lags $\tau = 1 , 2 , 3$ to evaluate whether autocorrelation in the time series leads to failure to distinguish direct from indirect temporal effects or other forms of lag misidentification.

As we show in Fig. 1e-h, causalBMS correctly and unequivocally identifies causal parents and lags at all levels of noise, with lagged causal scores $\pi _ { k \to i } ( \tau )$ close to one when a causal link $k  i$ exists at τ, and close to zero when it does not. Only when the autocorrelation in the $x _ { t }$ series becomes very weak, does the corresponding score decrease to $\pi _ { X \to X } ( 1 ) = 0 . 7 4$ , still clearly above 0.5 and above the true negative edges.

Our causalBMS draws from the posteriors over equations for $x _ { t }$ and $y _ { t }$ separately and reads directionality from the inclusion frequencies in Eq. (5). The ensemble assigns lag τ = 1 in X a high inclusion probability inside the equation for $y _ { t } .$ and Y a very low inclusion probability inside the equation for $x _ { t } ,$ recovering the correct causal effect. Beyond directionality, the method recovers, or at least approximates, the functional form of the interaction (Fig. 1i-1). At low noise $( \sigma = 0 . 0 4 )$ , the true $f _ { i } ^ { * }$ is well approximated within the observed range of x(t) (Fig. 1i), which is relatively narrow and prevents causalBMS from learning the limiting behavior and, thus, the exact functional form $f _ { i } ^ { * }$ . Conversely, at high noise $( \sigma \geq 4 ,$ Fig. 1k,1) the signal-to-noise ratio is so low that σ is above $\sigma ^ { \times }$ , as defined in Eq. (6), and the true $f _ { i } ^ { * }$ cannot be learned by any method³2. Despite this, the method still detects the correct causal effect. Finally, moderate noise levels $( \sigma = 0 . 4 )$ lead to exact recovery of the $f _ { i } ^ { * } ( \mathrm { F i g . \ 1 j } )$ because, in this regime, the signal-tonoise ratio is high enough to bring the data into the learnable regime and, at the same time, the observed range of $x ( t )$ is large enough that the limiting behavior of $f _ { i } ^ { * }$ is visible.

Thus, this example illustrates three important and somewhat counterintuitive features of causal discovery based on Bayesian symbolic regression: (i) causalBMS has no problem in detecting unidirectional causal relationships, even in scenarios where the state of the art in causal discovery methods fail6; (ii) causal effects are detectable even in the unlearnable symbolic regression phase; and (iii) more noise can actually increase the signal-to-noise ratio by increasing the range of observed values in the time series.

## B. CauseMe benchmarks

Next, we evaluate causalBMS on the CauseMe benchmark suite curated for the NeurIPS 2019 Causality for Climate competition26,27. The suite stresses three orthogonal axes: dynamical complexity (non-linearity of the dynamics and dimensionality d of the system), sample size T, and signal-to-noise ratio. Here, we consider the following datasets:

• Linear baselines. The linear-VAR family $( d = 3 .$ T = 150) provides a controlled baseline of linear models; the linear-VAR\_noisy configuration adds strong exogenous noise for the relatively small samples available.

• Nonlinear and chaotic systems. Coupled logistic maps (logistic-largenoise, $d = 3$ , medium and strong noise) test the high-noise transitional regime, while nonlinear-VAR with polynomial dependencies probes scalability across system dimensionality $d \in$ {3,5, 10} at $T = 3 0 0$

Each dataset consists of 250 time-series sets from which the causal graph must be inferred. We compare causalBMS with the 20 top performers for each dataset in the CauseMe leaderboard (Fig. 2).

Across all setups, causalBMS is consistently among the top performers in F1-score, area under the precision-recall curve (AUC-PR), and area under the ROC curve (AUC) (Fig. 2). Indeed, causalBMS is the top performer in 13 out of 21 datasetmetric pairs, the second-best in another four pairs, and consistently close to the top performer when it is not the best. Our approach works particularly well in nonlinear systems. The robustness in the strong-noise variants also supports using the score based on the probabilistic approach and the complete posterior distribution.

## C. A biogeoscience case study: temperature, productivity, and respiration

Finally, we consider a four-variable causal graph rooted in ecosystem physiology, which we choose because it violates some assumptions of causalBMS and is therefore particularly challenging for our approach. The structural model follows the simulated case study of Runge et al.43, in which incoming shortwave radiation $R _ { g }$ drives gross primary productivity GPP through air temperature $T _ { \mathrm { a i r } } ,$ and ecosystem respiration

![](images/3c52ca470019e07affe5ed758e0ab1a52878faf66a8086bd16cbda0704e7df1c.jpg)  
FIG. 2. Performance of causalBMS on the CauseMe benchmark. For the experiments described in §III B, we show: (a) area under the ROC curve; (b) area under the precision–recall curve; and (c) F1- score. The green bars are the causalBMS scores; the grey ranges span the top-20 entries reported by CauseMe for each dataset. causalBMS sits at or near the top of every column and consistently falls within the upper range of competing methods, especially for the nonlinear and strong-noise variants.

$R _ { \mathrm { e c o } }$ responds nonlinearly to $T _ { \mathrm { a i r } }$ and GPP through a $Q _ { 1 0 }$ -like kernel. The system encodes the standard light-temperature partitioning used to separate net ecosystem exchange into assimilation and respiration44,45 and is a controlled surrogate for the eddy-covariance setting routinely analyzed on FLUXNET sites46. The lag-one summary graph contains eight directed edges, including four cross-influences $( R _ { g } \to T _ { \mathrm { a i r } } , R _ { g } \to \mathrm { G P P }$ $T _ { \mathrm { a i r } } {  } \mathrm { G P P } , T _ { \mathrm { a i r } } {  } R _ { \mathrm { e c o } } , \mathrm { G P P } {  } R _ { \mathrm { e c o } } )$ and four autoregressive self-edges. The discrete-time SCM is, with seasonal phase $\phi _ { t } = t \pi / 3 6 5$ and i.i.d. fluctuations $\eta _ { t } ^ { i } .$

$$
R _ { g } ( t ) = 2 8 0 \sin ^ { 2 } \phi _ { t } + 5 0 \left| \sin \phi _ { t } \right| \eta _ { t } ^ { R _ { g } } ,\tag{8}
$$

$$
T _ { \mathrm { a i r } } ( t ) = 0 . 8 T _ { \mathrm { a i r } } ( t - 1 ) + 0 . 0 2 R _ { g } ( t ) + 5 \eta _ { t } ^ { T } ,
$$

$$
\mathrm { G P P } ( t ) = \left| 0 . 2 \mathrm { G P P } ( t - 1 ) + 0 . 0 0 2 T _ { \mathrm { a i r } } ( t ) R _ { g } ( t ) + 3 \eta _ { t } ^ { G } \right| ,
$$

$$
R _ { \mathrm { e c o } } ( t ) = \Big | 0 . 3 R _ { \mathrm { e c o } } ( t - 1 ) + 0 . 9 \mathrm { G P P } ( t ) 0 . 8 ^ { 0 . 1 2 ( T _ { \mathrm { a i r } } ( t ) - 1 5 ) } + 2 \eta _ { t } ^ { R } \Big | .
$$

![](images/51e510474f0b28a1111470f56ed997d3ed7953bbec50cb5b76d3359f43792075.jpg)

![](images/bdc3c25e193c1e7585aa527e606fc90449a526da60c09c9df186195b9696d2ff.jpg)

![](images/7e3eb23dd1d20c5269c2c1018ba7d7a24b1bd9888f16bb8b98ccd585d3efde62.jpg)  
FIG. 3. Multi-method comparison on the biogeoscience SCM of Eq. (8). F1-score (left), AUC-PR (middle), and AUC-ROC (right) on the cross-influence entries of the lag-one summary graph, averaged over twenty replicates per noise regime; error bars are one standard deviation. Regimes (left to right inside each panel): Gaussian fluctuations at the reference scale, Gaussian fluctuations at twice the reference scale. Student-t fluctuations with v = 4 degrees of freedom and a mixed regime with an additional cubic term on the $T _ { \mathrm { a i r } }$ fluctuations. causalBMS (green) is among the leading methods for F1-score and AUC-ROC and often ties for the best performer on these metrics. CCM (yellow) is the strongest method for AUC-PR throughout. In AUC-ROC, a linear Granger-causality test extended with contemporaneous terms (purple) matches or slightly exceeds every other method, including causalBMS, in all four regimes. VAR-LiNGAM (red) is consistently the weakest method in AUC-ROC and AUC-PR, while PCMCI+ (blue) shows more erratic performance across noise regimes in F1-score.

Each simulation produces $T = 8 0 0$ daily samples followed by a low-radiation seasonal mask that retains only the active season. We compare five methods spanning the constraint-, structural-, dynamics- and regression-based families: PCMCI+ with partial-correlation independence tests at $\tau _ { \operatorname* { m a x } } = 1 ^ { 2 6 , 4 1 }$ $\mathrm { V A R - L i N G A M ^ { 1 4 , 4 2 } }$ , convergent cross mapping $\begin{array} { r l } {  { ( \mathbf { C C M } ) ^ { 2 5 } } } \end{array}$ , a linear Granger-causality test extended with contemporaneous (lag-zero) terms to allow for instantaneous influences in Eq. $( 8 ) ^ { 4 7 }$ , and causalBMS.

We evaluate on four noise regimes: Gaussian fluctuations at the reference scale σ = 1 and at $\sigma = 2$ , heavy-tailed Studentt fluctuations with v = 4 degrees of freedom, and a mixed regime in which the temperature fluctuation includes an additive cubic distortion $0 . 2 \bar { 5 } \varepsilon ^ { 3 }$ as in the Runge example43, modeling occasional extremes. Each method is run on five independent replicates per regime; we report the F1-score, the area under the precision-recall curve (AUC-PR) and the area under the ROC curve (AUC-ROC) computed on the twelve offdiagonal entries of the lag-one summary graph (Fig. 3).

We note that this experiment is singularly challenging for the causalBMS for at least two reasons. First, in some experiments, the additive noise terms are not Gaussian. Second, and more significantly, functional identification is misspecified in Eq. (8) because of the contemporaneous (lag zero) influence terms. For example, temperature $T _ { \mathrm { a i r } }$ is causally affected only by radiation $R _ { g } ,$ but because of the lack of temporal resolution, the effect of $R _ { g }$ is measured as instantaneous. Similarly, the causal effect of $T _ { \mathrm { a i r } }$ and $R _ { g }$ on GPP and $R _ { \mathrm { e c o } }$ is also measured as instantaneous. Therefore, $T _ { \mathrm { a i r } }$ can be written as a function of $R _ { g } ,$ but also as a function of GPP and of $R _ { \mathrm { e c o } }$

Despite these intrinsic difficulties, causalBMS remains competitive with the strongest methods on most of the metrics in Fig. 3. In F1-score, causalBMS sits within the leading cluster together with CCM and linear Granger across all regimes and is often the best or close-to-best performer. In AUC-PR, causalBMS is in the second tier, alongside PCMCI+ and linear Granger, all below CCM, which is the strongest method on this metric in all regimes. On AUC-ROC, linear Granger with contemporaneous terms matches or exceeds every method on average in all four regimes, but causalBMS is always a very close second and statistically indistinguishable from the top performer. VAR-LiNGAM is consistently the weakest method in AUC-ROC across all four regimes, reflecting its inability to represent the multiplicative and exponential terms. In F1-score, however, its performance is comparable to, and in some regimes higher than, PCMCI+, whose linear partial-correlation test is itself much more uncertain across noise regimes than either causalBMS or linear Granger.

That a simple contemporaneous-term linear Granger test matches causalBMS in edge detection reflects a property of this benchmark: the mechanisms themselves in Eq. (8) are close enough to linear that a correctly-specified, lowparameter significance test is a statistically efficient detector of an edge’s presence.

Unlike PCMCI+, VAR-LiNGAM, linear Granger and CCM, which return only edge weights, causalBMS returns the closed-form expressions that produced those weights, together with a diagnostic (Fig. 4d-f) that identifies, without reference to the ground truth, which of its recovered mechanisms remain incompletely specified. Table I compares the MDL expression obtained by the causalBMS for each target to the ground-truth mechanism in Eq. (8).

For $R _ { \mathrm { { e c o } } } .$ the MDL expression not only captures the causal parents correctly but also matches exactly the ground truth expression, up to some error in the coefficients. This optimal outcome is possible because $R _ { \mathrm { e c o } }$ has the only functional dependency that is not misspecified in Eq. (8). For $T _ { \mathrm { a i r } }$ and GPP, the MDL models also capture the leading terms in the ground-truth causal mechanism– $- T _ { \mathrm { a i r } } ( t - 1 )$ and the product $R _ { g } ( t ) T _ { \mathrm { a i r } } ( t )$ , respectively. However, the models also include other terms with incorrect causal parents. The fact that the causalBMS prefers models that are more complex than the ground truth leads us to hypothesize that they must be more predictive than the ground truth. Indeed, as per Eq. (3), the only way a more complex model (lower log $p ( f _ { i } ) )$ is acceptable is if it reduces the BIC term of the description length even more. In Fig. 4a-c, we show that this is indeed the case—the identified MDL expressions have, for all three targets, lower prediction error than the ground-truth expressions. This counterintuitive result is a consequence of the misspecification in Eq. (8). Consider, for example, $T _ { \mathrm { a i r } } .$ Since, functionally, $T _ { \mathrm { a i r } }$ can be solved for analytically from GPP and $R _ { \mathrm { { e c o } } } .$ it is in general better to use a combination of these expressions to estimate $T _ { \mathrm { a i r } }$ , since such a combination will tend to cancel the noise terms and therefore lead to more accurate estimates.

TABLE I. Equations recovered by causalBMS on the biogeoscience SCM. For each target, we show the ground truth model and the MDL expression identified by the causalBMS in the reference Gaussian-noise regime.
<table><tr><td>True mechanism (from Eq. (8))</td><td>MDL causalBMS expression</td></tr><tr><td> $T _ { \mathrm { a i r } }$   $0 . 8 T _ { \mathrm { a i r } } ( t - 1 ) + 0 . 0 2 R _ { g } ( t )$ </td><td> $0 . 8 2 T _ { \mathrm { a i r } } ( t - 1 ) + 0 . 8 4 \mathrm { G P P } ( t ) - 0 . 0 2 3 T _ { \mathrm { a i r } } ( t - 1 ) R _ { \mathrm { e c o } } ( t )$ </td></tr><tr><td>GPP  $0 . 2 \mathrm { G P P } ( t - 1 ) + 0 . 0 0 2 T _ { \mathrm { a i r } } ( t ) R _ { g } ( t )$ </td><td> $1 . 1 \cdot 1 0 ^ { - 5 } \left( 6 2 0 R _ { \mathrm { e c o } } ( t ) + R _ { g } ( t ) T _ { \mathrm { a i r } } ( t ) \right) \left( \mathrm { G P P } ( t - 1 ) + R _ { \mathrm { e c o } } ( t ) - 2 . 0 R _ { \mathrm { e c o } } ( t - 1 ) + T _ { \mathrm { a i r } } ( t ) + 6 4 \right)$ </td></tr><tr><td> $R _ { \mathrm { e c o } }$   $0 . 3 R _ { \mathrm { e c o } } ( t - 1 ) + 0 . 9 \mathrm { G P P } ( t ) 0 . 8 ^ { 0 . 1 2 \left( T _ { \mathrm { a i r } } ( t ) - 1 5 \right) }$ </td><td> $0 . 3 2 R _ { \mathrm { e c o } } ( t - 1 ) + 1 . 3 \mathrm { G P P } ( t ) 0 . 9 7 ^ { T _ { \mathrm { a i r } } ( t ) }$ </td></tr></table>

![](images/3d8835cfd7c327157e14dbd446d09155cb7ed5faea3bd6de12f30bea070efe04.jpg)  
FIG. 4. Properties of minimum description length models identified by the causalBMS for the biogeoscience model of Eq. (8). After obtaining the MDL model for each target with the causalBMS, we generate test data with the same SCM model of Eq. (8) and test the predictive abilities and correlation structure of the models. (a-c) Predicted versus observed values of the target variables for the MDL model (green) and the ground-truth model (grey). (d-f) Correlations between residuals and predictors for each target variable and each predictor identified by the MDL model. We report the BH-corrected p-values for a permutation test with HSIC assessing remaining correlation between residuals and chosen predictors.

We now test whether each recovered mechanism fully accounts for its own selected inputs. For this, we compute the

Hilbert-Schmidt independence criterion (HSIC) between the residuals of each target and every predictor chosen in the respective MDL expression (Table I). We assess the significance using a permutation test. We run 1000 permutations and apply the Benjamini-Hochberg (BH) correction to account for multiple testing. This assessment is a diagnostic for the adequacy of the functional form that could apply in principle to the fitted equation of any method, regardless of how its inputs were selected. For $T _ { \mathrm { a i r } }$ and GPP, every predictor entering their MDL expressions remains significantly dependent on the residual after correction $( p _ { \mathrm { B H } } \le 0 . 0 1 2 4$ , Fig. 4d-e). This shows that neither expression fully captures the target's dependence on its own selected inputs. This is consistent with their larger deviation from the ground-truth mechanism in Table I. At the same time, none of the three predictors of the $R _ { \mathrm { e c o } }$ expression show significant residual dependence $( p _ { \mathrm { B H } } \ge 0 . 2 0 2 , \mathrm { F i g . ~ 4 f ) }$ This agrees with its near-exact recovery of both the groundtruth mechanism and RMSE (Fig. 4c). Because this diagnostic requires no access to the ground-truth model, it offers a practical, general-purpose check on whether a fitted expression (from causalBMS or any regression-based causal discovery method) is fully specified, complementing aggregate error metrics such as RMSE with a finer-grained structural test of residual adequacy.

## IV. DISCUSSION

We show that casting causal discovery as Bayesian symbolic regression turns the BMS posterior over expressions into a causal score via variable-inclusion frequencies. The score preserves the description length semantics of the BMS, extends naturally to many candidate parents and lagged predictors, and remains informative through the learnability transition characterized in Ref.32. Because each sampled expression directly enumerates its predictors, the score does not require comparing candidate factorizations or selecting a single best equation, and exploits the full posterior of plausible mechanisms.

Three limitations qualify the present results. First, identifiability is conditional on closed-form representability of the true mechanism; outside this class the inclusion frequencies remain well defined as a feature-selection signal but lose the consistency guarantee of Proposition 1. Second, the BMS sampler scales with the number of candidate parents; current implementations comfortably handle the benchmark dimensions $( d \leq 1 0 , T \leq 3 0 0 )$ but require additional ingredients to reach the high-dimensional settings of some real data. Third, the noise threshold in Eq. (6) depends on quantities $( \langle { \delta ^ { 2 } } \rangle , \Delta _ { M } ^ { * } )$ that involve the unknown true model and must be estimated from data; we use it as a diagnostic rather than as a hard guarantee.

Several extensions are natural and are likely to alleviate these limitations. Combining causalBMS with constraintbased skeleton discovery11,26 would exploit the strengths of both families: independence tests prune the candidate parent sets, and symbolic regression orients edges and recovers mechanisms. A similar approach is exploited by the recent TOPIC framework48, which infers a topological order from algorithmic mutual information and identifies edges that compress the data under a candidate ordering; embedding causalBMS as the per-node scoring oracle’ in such a topological pipeline would extend the method to higher dimensions while preserving its mechanism-level output. Embedding causalBMS inside hybrid modeling pipelines would provide explicit mechanism candidates for downstream physical evaluation, and the ODE-SCM correspondence of Ref.30 suggests a route from discovered difference equations to interventional predictions.

[1] J. Pearl, Causal inference in statistics: An overview, Stat. Surv. 3, 96 (2009).

[2] J. Peters, D. Janzing, and B. Schölkopf, Elements of Causal Inference: Foundations and Learning Algorithms (The MIT

In any case, we have shown that causal BMS often performs better than the state of the art. Moreover, the principal scientific gain of causalBMS over the alternatives is that it not only provides edge ranking in the causal graph, but also the explicit functional form of the surviving expressions, which is a much richer description of mechanisms, allows to test for remaining correlations, and is typically an additional object of interest in science.

## ACKNOWLEDGMENTS

This work is supported by the European Research Council (ERC) through grant USMILE (grant agreement No. 855187); the European Commission through the Horizon Europe project ELIAS (grant agreement No. 101120237); projects PID2022-142600NB-I00, PID2022-142600NB-I00, and PID2025-176129NB-I00 from MCIN / AEI / 10.13039 / 501100011033; and by the Government of Catalonia (2021SGR-633).

## AUTHOR CONTRIBUTIONS

RG, KHC, MSP, and GCV designed the research. RG, KHC, SP, and GCV wrote code and run simulations. All authors discussed and interpreted the results, worked on the visualizations, contributed to writing and revising the manuscript, and approved the final version.

## CODE AND DATA AVAILABILITY

The code for the BMS is openly available31. Data for the CauseMe benchmark are available from the CauseMe platform49. All other data used in the manuscript were synthetically generated as described in the text.

## COMPETING INTERESTS DECLARATION

The authors declare that they have no competing interests.

Press, Cambridge, MA, 2017) https://mitpress.mit. edu/ 9780262037310/elements-of-causal-inference/.

[3] J. Pearl, Causality: Models, Reasoning, and Inference, 2nd ed. (Cambridge University Press, New York, NY, 2009).

[4] J. Runge, S. Bathiany, E. Bollt, G. Camps-Valls, D. Coumou, E. Deyle, C. Glymour, M. Kretschmer, M. D. Mahecha, J. Muñoz-Marí, E. H. van Nes, J. Peters, R. Quax, M. Reichstein, M. Scheffer, B. Schölkopf, P. Spirtes, G. Sugihara, J. Sun, K. Zhang, and J. Zscheischler, Inferring causation from time series in Earth system sciences, Nat. Commun. 10, 2553 (2019).

[5] M. Reichstein, G. Camps-Valls, B. Stevens, M. Jung, J. Denzler, N. Carvalhais, and Prabhat, Deep learning and process understanding for data-driven Earth system science, Nature 566, 195 (2019).

[6] G. Camps-Valls, A. Gerhardus, U. Ninad, G. Varando, G. Martius, E. Balaguer-Ballester, R. Vinuesa, E. Diaz, L. Zanna, and J. Runge, Discovering causal relations and equations from data, Phys. Rep. 1044, 1 (2023).

[7] A. T. Reid, D. B. Headley, R. D. Mill, R. Sanchez-Romero, L. Q. Uddin, D. Marinazzo, D. J. Lurie, P. A. Valdés-Sosa, S. J. Hanson, B. B. Biswal, V. Calhoun, R. A. Poldrack, and M. W. Cole, Advancing functional connectivity research from association to causation, Nat. Neurosci. 22, 1751 (2019).

[8] S. H. Siddiqi, K. P. Kording, J. Parvizi, and M. D. Fox, Causal mapping of human brain function, Nat. Rev. Neurosci. 23, 361 (2022).

[9] M. A. Hernán and J. M. Robins, Causal Inference: What If (Chapman & Hall/CRC, Boca Raton, FL, 2020) https: //miguelhernan.org/whatifbook.

[10] J. Hicks, Causality in Economics (Australian National University Press, Canberra, 1980).

[11] P. Spirtes, C. Glymour, and R. Scheines, Causation, Prediction, and Search, 2nd ed. (The MIT Press, Cambridge, MA, 2000).

[12] P. O. Hoyer, D. Janzing, J. M. Mooij, J. Peters, and B. Schölkopf, Nonlinear causal discovery with additive noise models, in Adv. Neural Inf. Process. Syst., Vol. 21 (2009) pp. 689–696, https: //papers.nips.cc/paper\_files/paper/2008/hash/ f7664060cc52bc6f3d620bcedc94a4b6-Abstract.html.

[13] J. M. Mooij, J. Peters, D. Janzing, J. Zscheischler, and B. Schölkopf, Distinguishing cause from effect using observational data: Methods and benchmarks, J. Mach. Learn. Res. 17, 1 (2016), https://www.jmlr.org/papers/v17/ 14-518.html.

[14] S. Shimizu, P. O. Hoyer, A. Hyvärinen, and A. Kerminen, A linear non-Gaussian acyclic model for causal discovery, J. Mach. Learn. Res. 7, 2003 (2006), https://www. jmlr.org/ papers/v7/shimizu06a.html.

[15] J. Peters, P. Bühlmann, and N. Meinshausen, Causal inference by using invariant prediction: Identification and confidence intervals, J. R. Stat. Soc. B 78, 947 (2016).

[16] C. Heinze-Deml, J. Peters, and N. Meinshausen, Invariant causal prediction for nonlinear models, J. Causal Inference 6, 20170016 (2018).

[17] D. Janzing, J. Mooij, K. Zhang, J. Lemeire, J. Zscheischler, P. Daniušis, B. Steudel, and B. Schölkopf, Informationgeometric approach to inferring causal directions, Artif. Intell. 182–183, 1 (2012).

[18] X. Sun, D. Janzing, and B. Schölkopf, Causal reasoning by evaluating the complexity of conditional densities with kernel methods, Neurocomputing 71, 1248 (2008).

[19] H. Reichenbach, The Direction of Time (University of California Press, Berkeley, CA, 1991) originally published in 1956.

[20] J. Mitrovic, D. Sejdinovic, and Y. W. Teh, Causal inference via kernel deviance measures, in Adv. Neural Inf. Process. Syst., Vol. 31 (2018) pp. 6986–6994, https: //papers.nips.cc/paper\_files/paper/2018/hash/ 73fed7fd472e502d8908794430511f4d-Abstract.html.

[21] A. Pérez-Suay and G. Camps-Valls, Causal inference in geoscience and remote sensing from observational data, IEEE Trans. Geosci. Remote Sens. 57, 1502 (2019).

[22] K. Budhathoki and J. Vreeken, Causal inference by stochastic complexity, arXiv:1702.0677610.48550/arXiv.1702.06776 (2017).

[23] A. Marx and J. Vreeken, Telling cause from effect by local and global regression, Knowl. Inf. Syst. 60, 1277 (2019)

[24] C. W. J. Granger, Investigating causal relations by econometric models and cross-spectral methods, Econometrica 37, 424 (1969).

[25] G. Sugihara, R. May, H. Ye, C.-h. Hsieh, E. Deyle, M. Fogarty, and S. Munch, Detecting causality in complex ecosystems, Science 338, 496 (2012).

[26] J. Runge, P. Nowack, M. Kretschmer, S. Flaxman, and D. Sejdinovic, Detecting and quantifying causal associations in large nonlinear time series datasets, Sci. Adv. 5, eaau4996 (2019).

[27] J. Runge, X.-A. Tibau, M. Bruhns, J. Muñoz-Marí, and G. Camps-Valls, The causality for climate competition, in Proceedings of the NeurIPS 2019 Competition and Demonstration Track, Proc. Mach. Learn. Res., Vol. 123 (PMLR, 2020) pp. 110–120, https://proceedings.mlr.press/ v123/runge20a.html.

[28] R. Guimerà, I. Reichardt, A. Aguilar-Mogas, F. A. Massucci, M. Miranda, J. Pallarès, and M. Sales-Pardo, A bayesian machine scientist to aid in the solution of challenging scientific problems, Sci. Adv. 6, eaav6971 (2020).

[29] R. Guimerà and M. Sales-Pardo, Bayesian symbolic regression: automated equation discovery from a physicist's perspective, Philos. Trans. R. Soc. A 384, 20250089 (2026).

[30] J. M. Mooij, D. Janzing, and B. Schölkopf, From ordinary differential equations to structural causal models: The deterministic case, in Proceedings of the Twenty-Ninth Conference on Uncertainty in Artificial Intelligence (UAI) (AUAI Press, Arlington, VA, 2013) pp. 440–448.

[31] https://bitbucket.org/rguimera/ machine-scientist/, accessed: 2026-09-18.

[32] O. Fajardo-Fontiveros, I. Reichardt, H. R. De Los Ríos, J. Duch, M. Sales-Pardo, and R. Guimerà, Fundamental limits to learning closed-form mathematical models from data, Nat. Commun. 14, 1043 (2023).

[33] For simplicity, here we ignore possible contemporaneous effects, but we consider a case where such effects exist later in our experimental results section.

[34] S. L. Brunton, J. L. Proctor, and J. N. Kutz, Discovering governing equations from data by sparse identification of nonlinear dynamical systems, Proc. Natl. Acad. Sci. U.S.A. 113, 3932 (2016).

[35] O. Cabanas-Tirapu, S. Cobo-Lopez, S. E. Sanchez, F. L. Rohwer, M. Sales-Pardo, and R. Guimerà, Integral bayesian symbolic regression for optimal discovery of governing equations from scarce and noisy data (2025), arXiv:2511.14388 [physics.data-an].

[36] T. A. Enßlin, Information theory for fields, Ann. Phys. (Berlin) 531, 1800127 (2019).

[37] P. D. Grünwald, The Minimum Description Length Principle (The MIT Press, Cambridge, MA, 2007).

[38] C. Fox, N. D. Tran, F. N. Nacion, S. Sharlin, and T. R. Josephson, Incorporating background knowledge in symbolic regression using a computer algebra system, Mach. Learn.: Sci. Technol. 5, 025057 (2024).

[39] G. Schwarz, Estimating the dimension of a model, Ann. Stat. 6, 461 (1978).

[40] The +1 term corresponds to one additional parameter of the probabilistic model, namely, the standard deviation σ of the exogenous noise.

[41] C. Krich, J. Runge, D. G. Miralles, M. Migliavacca, O. Perez-Priego, T. El-Madany, A. Carrara, and M. D. Mahecha, Estimating causal networks in biosphere-atmosphere interaction with the pcmci approach, Biogeosciences 17, 1033 (2020).

[42] A. Hyvärinen, K. Zhang, S. Shimizu, and P. O. Hoyer, Estimation of a structural vector autoregression model using non-Gaussianity, J. Mach. Learn. Res. 11, 1709 (2010), https: //www.jmlr.org/papers/v11/hyvarinen10a.html.

[43] J. Runge, A. Gerhardus, G. Varando, V. Eyring, and G. Camps-Valls, Causal inference for time series, Nat. Rev. Earth Environ. 4, 487 (2023).

[44] M. Reichstein, E. Falge, D. Baldocchi, D. Papale, M. Aubinet, P. Berbigier, C. Bernhofer, N. Buchmann, T. Gilmanov, A. Granier, et al., On the separation of net ecosystem exchange into assimilation and ecosystem respiration: Review and improved algorithm, Glob. Change Biol. 11, 1424 (2005).

[45] M. D. Mahecha, M. Reichstein, N. Carvalhais, G. Lasslop, H. Lange, S. I. Seneviratne, R. Vargas, C. Ammann, M. A. Arain, A. Cescatti, I. A. Janssens, M. Migliavacca, L. Montagnani, and A. D. Richardson, Global convergence in the temper-

ature sensitivity of respiration at ecosystem level, Science 329, 838 (2010).

[46] G. Pastorello, C. Trotta, E. Canfora, H. Chu, D. Christianson, Y.-W. Cheah, C. Poindexter, J. Chen, A. Elbashandy, M. Humphrey, et al., The fluxnet2015 dataset and the oneflux processing pipeline for eddy covariance data, Sci. Data 7, 225 (2020).

[47] A. Shojaie and E. B. Fox, Granger causality: A review and recent advances, Annu. Rev. Stat. Appl. 9, 289 (2022).

[48] S. Xu, S. Mameche, and J. Vreeken, Information-theoretic causal discovery in topological order, in Proceedings of The 28th International Conference on Artificial Intelligence and Statistics, Proc. Mach. Learn. Res., Vol. 258, edited by Y. Li, S. Mandt, S. Agrawal, and E. Khan (PMLR, 2025) pp. 2008– 2016.

[49] J. Runge, X.-A. Tibau, M. Bruhns, J. Muñoz-Marí, and G. Camps-Valls, The causality for climate competition, in Proceedings of the NeurIPS 2019 Competition and Demonstration Track, Proc. Mach. Learn. Res., Vol. 123 (PMLR, 2020) pp. 110–120, https://proceedings.mlr.press/ v123/runge20a.html.