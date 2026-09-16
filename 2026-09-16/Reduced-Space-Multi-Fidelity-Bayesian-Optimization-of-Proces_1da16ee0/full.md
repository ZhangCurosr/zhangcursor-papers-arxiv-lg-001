# Reduced-Space Multi-Fidelity Bayesian Optimization of Process Simulation Models

Niki Triantafyllou, Andrea Bernardi, and Maria M. Papathanasiou<sup>⋆</sup>

Sargent Centre for Process Systems Engineering, Department of Chemical Engineering, Imperial College London, UK {niki.triantafyllou20, a.bernardi13, maria.papathanasiou11}@imperial.ac.uk

Abstract. Optimizing industrial process flowsheets is often computationally prohibitive due to the high cost of rigorous simulations and the curse of dimensionality inherent in complex design spaces. To address these challenges, we present a reduced-space multi-fidelity Bayesian optimization (RS-MFBO) framework designed for high-dimensional, expensive black-box functions. The approach integrates Global Sensitivity Analysis (GSA) for dimensionality reduction with a fidelity-augmented Gaussian process that captures correlations between low-cost approximations and expensive high-fidelity evaluations. A cost-aware acquisition strategy, augmented with cooldown and promotion mechanisms, adaptively guides the allocation of samples across fidelities. The framework is validated on two distinct industrial process simulators: a plasmid DNA bioprocess in SuperPro Designer and a green fuel synthesis plant in Aspen HYSYS. Results across diverse economic and physical objectives demonstrate that the proposed method substantially reduces the number of high-fidelity simulator evaluations while maintaining competitive optimization performance compared to single-fidelity baselines. These results highlight RS-MFBO as a scalable, simulator-agnostic approach for cost-constrained black-box optimization.

Keywords: Multi-fidelity Bayesian Optimization · Bayesian Optimization · Gaussian Processes · Active Learning.

## 1 Introduction

Systematic process design and optimization aim to determine operating and design conditions that maximize process performance (e.g., throughput, yield, energy eficiency) or minimize cost and environmental impact. (Bio)chemical process simulators such as Aspen and SuperPro Designer are widely used for these problems owing to their extensive model libraries, integrated physicalproperty databases, and user-friendly interfaces. While such simulators provide high-fidelity (HI) representations of process behavior, their governing equations are not exposed, so derivatives are unavailable and gradient-based methods are inapplicable. Consequently, sequential-modular flowsheet optimization relies on derivative-free, sample-eficient strategies.

Bayesian optimization (BO) ofers a principled framework for optimizing expensive black-box functions under limited evaluations [8,1,16,23,9]. However, in flowsheet optimization settings, BO faces three key challenges: (i) high-dimensional design spaces; (ii) costly and occasionally non-convergent simulator evaluations (necessitating random restarts or discarding failed runs); and (iii) operational constraints that restrict feasible operating regions. Constrained BO is therefore required to keep exploration within known safe limits [3]. At the same time, single-fidelity simulation-based BO is ineficient when simulator calls are expensive and unreliable, while surrogate-only BO tends to lose accuracy when extrapolating beyond its training domain [23].

Multi-fidelity Bayesian optimization (MFBO) mitigates the challenges of costly simulator evaluations and imperfect surrogate models by combining multiple sources of information, typically a cheap low-fidelity (LO) approximation and an expensive high-fidelity (HI) simulator, within a unified probabilistic multifidelity model [11,12,5,4]. By jointly learning cross-fidelity correlations and using cost-aware acquisition functions that balance exploration, exploitation, and fidelity accuracy, MFBO eficiently allocates evaluations under a fixed computational budget.

Even with MFBO, high dimensionality remains a bottleneck in flowsheet optimization. To address this, global sensitivity analysis (GSA) can be employed to eliminate low-influence variables prior to optimization. GSA provides an interpretable, model-agnostic means of quantifying the contribution and interaction efects of each input on the objective function, enabling an explainable form of dimensionality reduction.

Building on our earlier work on reduced-space single-fidelity BO [23], we now introduce a reduced-space multi-fidelity Bayesian optimization (RS-MFBO) framework that couples surrogate and simulator evaluations through a fidelityaugmented Gaussian process with a cross-fidelity kernel. A cost-aware acquisition policy incorporating cooldown and promotion mechanisms adaptively balances LO and HI queries throughout the search. We assess the algorithmic robustness of this framework against two benchmark case studies utilizing different (bio)chemical process simulators: (a) plasmid DNA production in Super-Pro Designer with 18 mixed continuous and discrete decision variables, and (b) green fuel (dimethyl ether) production in Aspen HYSYS with 14 mixed continuous and discrete decision variables. By benchmarking against single-fidelity and surrogate-based baselines across twelve diverse objectives (six per case study), we show that RS-MFBO achieves competitive optimization performance with significantly reduced computational overhead (Figure 1).

![](images/4351056718aa683fff3a00bc805abda593dd445b06f7c91bedcf4660dce202fa.jpg)

Dimethyl Ether Production  
![](images/9e0223bf882826abb3a7fbd55b6f17dcc47d004d7cfe90febd80d1449e26b23c.jpg)  
Fig. 1. High-fidelity black-box simulator models: (a) pDNA bioprocess (SuperPro Designer), (b) green fuel synthesis (Aspen HYSYS) [24].

## 2 Related Work

High-Dimensional Bayesian Optimization. Scaling BO to high-dimensional design spaces (D > 20) is a longstanding challenge. A common strategy involves embedding the high-dimensional problem into a lower-dimensional latent subspace. This includes random embedding methods like REMBO [25] and HeSBO [15], as well as recent extensions for discrete search spaces that project integer variables into continuous latent manifolds [9]. Other approaches, such as TuRBO [7], manage dimensionality by restricting the search to local trust regions. While numerically efective, these embedding and projection methods operate in latent spaces that lack physical meaning. In process systems engineering, maintaining interpretability is crucial. Decision-makers need to understand which specific physical variables (e.g., temperatures, flow rates) drive performance. This motivates our use of Global Sensitivity Analysis (GSA) [18] to identify the active subspace directly in the native coordinate system, preserving engineering insight.

BO in Process Systems Engineering. Bayesian optimization is gaining traction in the chemical engineering community as a sample-eficient alternative to derivative-free solvers. Recent applications cover a broad spectrum of process systems problems, including reactor design [19], bioprocess development [13], and molecular discovery [20,14]. Specialized frameworks like ENTMOOT [22] have also been developed to handle tree-based surrogate models within BO. Although some of these studies have successfully introduced multi-fidelity strategies to specific domains, such hierarchical data structures are often neglected in broader process optimization tasks, where single-fidelity surrogates remain the standard.

Multi-Fidelity Acquisition Strategies. Multi-fidelity BO (MFBO) integrates information sources of varying costs and accuracies. Existing methods often assume a discrete fidelity hierarchy (e.g., MISO [17]) or rely on informationtheoretic acquisition functions, such as Multi-Fidelity Entropy Search (MUMBO) or Knowledge Gradient (MF-KG) [26]. While these advanced acquisition functions ofer rigorous theoretical bounds, they incur significant computational overhead due to the need for numerical integration or inner optimization loops. In our context, where high-fidelity simulations require only 5–45 seconds, the walltime cost of optimizing an expensive acquisition function (like KG) can exceed the cost of the simulation itself, negating the benefits of MFBO. To ensure the framework remains practical for intermediate-cost simulators, we adopt the continuous fidelity relaxation proposed by Wu et al. [26], but pair it with a lightweight, analytical cost-aware UCB strategy instead of the computationally expensive Knowledge Gradient.

## 3 Methodology

## 3.1 Problem Setting

The optimization task seeks to identify process design and operating variables $\mathbf { x } \in \mathcal { X }$ that optimize a high-fidelity (HI) simulator objective. Formally, the problem is expressed as a constrained black-box optimization:

$$
\begin{array} { r l } & { \underset { \mathbf { x } \in \mathcal { X } _ { \mathrm { f e a s } } } { \operatorname* { m i n } } f _ { \mathrm { H I } } ( \mathbf { x } ) } \\ & { \mathrm { s . t . } \quad g _ { j } ( \mathbf { x } ) \leq 0 , \quad j = 1 , \dots , m , } \end{array}\tag{1}
$$

where $\mathcal { X } _ { \mathrm { f e a s } } \subseteq \mathcal { X }$ denotes the feasible input domain satisfying all operational and physical constraints $g _ { j } ( \mathbf { x } )$ . Evaluations of $f _ { \mathrm { H I } }$ are computationally expensive and derivative-free, motivating the use of a low-fidelity surrogate $f _ { \mathrm { L O } }$ that approximates f<sub>HI</sub> at reduced cost. The proposed multi-fidelity Bayesian optimization (MFBO) framework (Algorithm 1) adaptively allocates evaluations between $f _ { \mathrm { L O } }$ and $f _ { \mathrm { H I } }$ to eficiently solve (1) under a finite computational budget.

## 3.2 Input-Augmented Multi-Fidelity Gaussian Process

We model both fidelities jointly as a single Gaussian process (GP) defined over an augmented input space $( \mathbf { x } , s )$ , where $s \in [ 0 , 1 ]$ represents the fidelity level $( s { = } 0$ for low fidelity and $s { = } 1$ for high fidelity). This continuous formulation captures smooth correlations between fidelities while avoiding the need for separate surrogate models.

The prior over the latent function $f ( \mathbf { x } , s )$ is

$$
f ( \mathbf { x } , s ) \sim \mathcal { G P } \big ( m ( \mathbf { x } , s ) , k \big ( ( \mathbf { x } , s ) , ( \mathbf { x } ^ { \prime } , s ^ { \prime } ) \big ) \big ) ,\tag{2}
$$

with a constant mean $m ( { \bf x } , s ) = \mu _ { 0 }$ . Let $\mathbf { z } = ( \mathbf { x } , s )$ and $\mathbf { z } ^ { \prime } = ( \mathbf { x } ^ { \prime } , s ^ { \prime } )$ . The covariance follows the linear-truncated fidelity kernel used in BoTorch’s multifidelity Gaussian process implementation [2,26]:

$$
k ( \mathbf { z } , \mathbf { z } ^ { \prime } ) = k _ { \mathrm { u n b i a s e d } } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) + ( 1 - s ) ( 1 - s ^ { \prime } ) ( 1 + s s ^ { \prime } ) ^ { p } k _ { \mathrm { b i a s e d } } ( \mathbf { x } , \mathbf { x } ^ { \prime } ) .\tag{3}
$$

Here, $k _ { \mathrm { u n b i a s e d } }$ and $k _ { \mathrm { b i a s e d } }$ are $\mathrm { M a t e r n { - } 5 / 2 }$ kernels applied to normalized inputs with standardized outputs, and $p$ is a kernel hyperparameter. Since this work uses only the binary fidelity levels $s \in \{ 0 , 1 \}$ , the fidelity-dependent factor reduces to one for low–low covariance and to zero whenever at least one point is high fidelity.

For the same process input x, this gives

$$
k \big ( ( \mathbf { x } , 0 ) , ( \mathbf { x } , 0 ) \big ) = k _ { \mathrm { u n b i a s e d } } ( \mathbf { x } , \mathbf { x } ) + k _ { \mathrm { b i a s e d } } ( \mathbf { x } , \mathbf { x } ) ,\tag{4}
$$

$$
k { \big ( } ( \mathbf { x } , 1 ) , ( \mathbf { x } , 1 ) { \big ) } = k _ { \mathrm { u n b i a s e d } } ( \mathbf { x } , \mathbf { x } ) ,\tag{5}
$$

$$
k { \big ( } ( \mathbf { x } , 0 ) , ( \mathbf { x } , 1 ) { \big ) } = k _ { \mathrm { u n b i a s e d } } ( \mathbf { x } , \mathbf { x } ) .\tag{6}
$$

Thus, the low-fidelity source is represented as a biased but correlated information source rather than as an independent surrogate. The shared covariance component couples the two fidelities, while the bias component captures the discrepancy associated with the lower-fidelity source.

Conditioning on observed data $\mathcal { D } = \{ ( \mathbf { x } _ { i } , s _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ yields the posterior predictive distribution

$$
p ( f _ { * } \mid \mathbf { x } _ { * } , s _ { * } , \mathcal { D } ) = \mathcal { N } \big ( \mu _ { * } ( \mathbf { x } _ { * } , s _ { * } ) , \sigma _ { * } ^ { 2 } ( \mathbf { x } _ { * } , s _ { * } ) \big ) ,\tag{7}
$$

with standard GP mean and variance updates.

This input-augmented formulation is related to autoregressive and co-kriging multi-fidelity models [12], in which low- and high-fidelity responses are modeled as correlated but non-identical functions. The joint covariance allows observations at either fidelity to contribute to the posterior. In particular, low-fidelity evaluations can reduce uncertainty in high-fidelity predictions when the two information sources are suficiently correlated, while retaining a distinct highfidelity response surface.

Algorithm 1 Reduced-space cost-aware multi-fidelity Bayesian optimization   
Require: Feasible domain ${ \overline { { \mathcal { X } _ { \mathrm { f e a s } } } } } ,$ budget ${ \overline { { B , } } }$ fidelity costs $c _ { \mathrm { L O } } < c _ { \mathrm { H I } } .$ , UCB parameters   
$( \beta , \alpha )$ , cooldown period H, promotion interval $P ,$ and top-K candidates   
1: GSA screening: compute total-order Sobol’ indices $S _ { i } ^ { \mathrm { T } }$ and retain dominant vari  
ables to form reduced space $\mathcal { X } _ { \mathrm { f e a s } } ^ { \prime }$   
2: Initialization: sample Sobol’ points at both fidelities in $\mathcal { X } _ { \mathrm { f e a s } } ^ { \prime } ;$ evaluate $f _ { \mathrm { L O } }$ and   
f<sub>HI</sub>; set total cost C   
3: while $C < B$ do   
4: Fit MF-GP   
5: Maximize $\mathrm { U C B } _ { s } ( \mathbf { x } ) = \mu _ { s } ( \mathbf { x } ) + \beta \sigma _ { s } ( \mathbf { x } )$ for $s \in \{ \mathrm { L O } , \mathrm { H I } \}$   
6: Compute cost-adjusted scores $J _ { s } = \mathrm { U C B } _ { s } / c _ { s } ^ { \alpha }$   
7: if cooldown reached or J<sub>HI</sub> $\geq J _ { \mathrm { L O } }$ then   
8: Evaluate $f _ { \mathrm { H I } } ( \mathbf { x } _ { \mathrm { H I } } )$   
9: else   
10: Evaluate $f _ { \mathrm { L O } } ( \mathbf { x } _ { \mathrm { L O } } )$   
11: end if   
12: if iteration mod $P = 0$ then   
13: Promote top-K LO points to HI and re-evaluate (duplicates removed)   
14: end if   
15: end while   
16: return best observed high-fidelity objective

## 3.3 Cost-Aware Acquisition with Cooldown and Promotion

At iteration t, we optimize the upper-confidence-bound (UCB) acquisition separately at each fidelity and score them cost-adjustedly:

$$
\mathrm { U C B } _ { s } ( \mathbf { x } ) = \mu _ { s } ( \mathbf { x } ) + \beta \sigma _ { s } ( \mathbf { x } ) , \quad s \in \{ \mathrm { L O } , \mathrm { H I } \} ,\tag{8}
$$

$$
J _ { s } = \frac { \mathrm { U C B } _ { s } ( \mathbf { x } ) } { c _ { s } ^ { \alpha } } .\tag{9}
$$

The fidelity with the larger $J _ { s }$ is selected, but a high-fidelity (HI) evaluation is forced every H steps to prevent over-reliance on the low-fidelity (LO) surrogate (cooldown). Every P steps, the top-K LO points with the highest predicted improvement are promoted for HI reevaluation. For minimization objectives, the acquisition function is applied to the transformed objective $- f .$

Before promotion, candidate LO points are checked for redundancy against existing HI evaluations. We remove any point $\mathbf { x } _ { \mathrm { n e w } }$ that lies within a small $\ell _ { \infty } -$ norm distance of an existing HI point $\mathbf { x } _ { \mathrm { o l d } } \mathbf { : }$

$$
\left\| \mathbf { x } _ { \mathrm { n e w } } - \mathbf { x } _ { \mathrm { o l d } } \right\| _ { \infty } = \operatorname* { m a x } _ { i } \left| x _ { \mathrm { n e w } , i } - x _ { \mathrm { o l d } , i } \right| < \varepsilon ,
$$

with $\varepsilon = 1 0 ^ { - 6 }$ in normalized coordinates. This deduplication avoids redundant simulator calls and numerical instability due to repeated evaluations at nearly identical conditions. The surviving points are then evaluated at high fidelity, and their results are incorporated into the dataset used for subsequent model updates. The total computational budget is updated accordingly.

The MFBO loop proceeds until the total computational budget is exhausted or no further improvement in the best observed high-fidelity objective is detected over a fixed number of iterations, indicating convergence.

## 3.4 Reduced Space via Variance-Based Global Sensitivity Analysis

To alleviate the curse of dimensionality and improve model conditioning, we perform a variance-based global sensitivity analysis (GSA) using total-order Sobol’ indices [18]. Let $f ( \mathbf { x } )$ denote the model output and $\operatorname { V a r } ( f )$ its total variance over the input domain X . For each input variable $x _ { i } ,$ the total-order index $S _ { i } ^ { \mathrm { T } }$ quantifies its overall contribution (main efect and all interactions) to the output variance:

$$
S _ { i } ^ { \mathrm { T } } = 1 - \frac { \operatorname { V a r } _ { \mathbf { x } _ { \sim i } } ( \mathbb { E } [ f ( \mathbf { x } ) \mid \mathbf { x } _ { \sim i } ] ) } { \operatorname { V a r } ( f ) } ,\tag{10}
$$

where $\mathbf x _ { \sim i }$ denotes all inputs except $x _ { i }$ . Variables with negligible $S _ { i } ^ { \mathrm { T } }$ have limited influence on the objective and can be safely fixed at nominal values.

We retain the subset of variables ${ \mathcal { T } } = \{ i : S _ { i } ^ { \mathrm { T } } \geq \tau \}$ , where $\tau$ is a userspecified total-order sensitivity threshold. Variables not included in I are fixed at their nominal values.

## 4 Experimental Setup

## 4.1 Case Studies

To validate the generalizability of RS-MFBO, we apply the framework to two distinct industrial process simulators (Figure 1) representing diferent problem classes.

Case Study I: Biopharmaceutical Process (SuperPro Designer). We optimize a plasmid DNA (pDNA) production process [24]. The system is modeled as a steady-state representation of a batch process in SuperPro Designer. This setup captures the aggregated material and energy balances of a complete production cycle, involving strictly sequential unit operations (fermentation, lysis, chromatography).

– Search Space: The problem involves D = 18 mixed continuous and discrete variables $( \mathrm { e . g . }$ , equipment sizing, batch scheduling parameters, flow rates).

Objectives: We optimize six KPIs: (1) Production Cost, (2) Operating Expenditure (OpEx), (3) Capital Expenditure (CapEx), (4) Batch Size, (5) Batch Time, and (6) Cycle Time. This set captures the trade-ofs between cost minimization and throughput maximization under strict scheduling constraints. Two linear input constraints enforce the increasing duration of fermentation stages:

$$
t _ { \mathrm { f l a s k } } \ \leq \ t _ { \mathrm { s e e d } } \ \leq \ t _ { \mathrm { m a i n } }
$$

Case Study II: Chemical Synthesis Process (Aspen HYSYS). We optimize a green fuel synthesis plant producing Dimethyl Ether (DME) from captured $\mathrm { C O _ { 2 } }$ and green $\mathrm { H _ { 2 } }$ . Modeled in Aspen HYSYS, this is a steady-state continuous process characterized by complex thermodynamic interactions and rigorous recycle loops, which frequently induce simulator non-convergence.

– Search Space: The problem involves D = 14 mixed continuous and discrete variables $( \mathrm { e . g . }$ , reactor pressures, column tray counts, recycle ratios).

– Objectives: We optimize six KPIs including: (1) DME Production Rate, (2) Energy Eficiency, (3) Carbon Eficiency, (4) OpEx, (5) CapEx, and (6) Production Cost. This selection tests the algorithm’s ability to handle highly non-convex thermodynamic landscapes.

## 4.2 Baselines

We compare the proposed RS-MFBO against five baselines:

1. Sobol’ Sampling: Quasi-random search (performance lower bound).

2. BO (Standard): Vanilla high-fidelity Bayesian Optimization on the full input space.

3. RS-BO: Single-fidelity BO operating on the reduced subspace defined by GSA.

4. RS-ANN-BO: A surrogate-based approach where BO optimizes the lowfidelity ANN on the reduced subspace defined by GSA, validated once at the end.

5. RS-ANN-MILP: Deterministic global optimization of the ReLU ANN surrogate on the reduced subspace defined by GSA via Mixed-Integer Linear Programming [6].

## 4.3 Implementation Details

A VBA-Python COM interface was developed to enable automated communication with the SuperPro Designer and Aspen HYSYS simulators. Experiments were conducted using a fixed total cost budget. High-fidelity simulator evaluations were assigned a normalized cost $c _ { \mathrm { H I } } = 1 0 . 0$ , while low-fidelity surrogate evaluations were assigned $c _ { \mathrm { L O } } \approx 1 0 ^ { - 1 }$ . GSA screening was performed using the SALib library [10] using a total-order Sobol’ index threshold of $\tau = 0 . 0 1$ . The Multi-Fidelity GPs were implemented in BoTorch [2] using a linear-truncated kernel structure. UCB parameters were set to $\beta \ : = \ : 1 5 . 0$ and $\alpha \ : = \ : 0 . 1$ , with cooldown period $H = 1 0$ , promotion interval $P = 1 0 .$ , and $K = 1$ candidate promoted per interval. Each run was initialised with $N _ { \mathrm { H I } } = 2$ and $N _ { \mathrm { L O } } = 2$ Sobol’ points per fidelity. All results are reported as the median and interquartile range (IQR) over 10 random seeds. Neural network training was GPU-accelerated, while Bayesian optimization and MILP were executed on CPU.

Apart from the pure Sobol’ sampling baseline, all methods are initialized with the best-performing design points identified by the GSA-based screening procedure, whereas the vanilla BO baseline starts from independent Sobol’ samples. The implementation of the proposed RS-MFBO framework is available at https://github.com/nikitrian/Reduced-space\_Bayesian\_Optimization.

![](images/e1270bcdeb2786da4e7a9cb7b1a47809bf372baf94c3f5285656cb5773a463c9.jpg)

(b)  
![](images/7a0afbc7da32e5b008b769369572b40ed9a18f381eab24c807a8bde314b5559c.jpg)

![](images/183fdfb5ad687d8538f3a51e9dab85661dc95f9f2cbc9fc42e1729013ba728f4.jpg)

(d)  
![](images/87d2fff94690af1b22dbcc96153cc258def8af57d6750cbaeba0502af9510efc.jpg)

(e)  
![](images/9f7ff9f0f29e5742db203b46436dee60f0ac2944af802611639d77708461d8d5.jpg)

(f)  
![](images/9b8df808700c77c5b3b9638f8705a3370c068b94508f5c148eacbae6cf8ed788.jpg)  
Fig. 2. Optimization results for all objectives: best observed high-fidelity value vs. function evaluations for Case Study I. Surrogate-based methods are validated with the simulator.

## 5 Results and Discussion

The proposed reduced-space MFBO framework was evaluated against five singleand surrogate-based optimization baselines across two distinct industrial process case studies: a batch plasmid DNA (pDNA) production process modeled in SuperPro Designer and a continuous green fuel (dimethyl ether) synthesis plant modeled in Aspen HYSYS. In total, twelve diverse key performance indicators (six per case study) were optimized, covering economic metrics like capital and operating expenditures, as well as physical performance indicators such as batch time, cycle time, and carbon eficiency. Each objective was optimized independently under identical initial designs and total cost budgets to ensure a fair comparison of sample eficiency across methods. Critical variables and lowfidelity ANN surrogates were obtained via GSA screening as described in the methodology.

## 5.1 Overall optimization performance

The reduced-space multi-fidelity BO (RS-MFBO) demonstrates robust convergence across all objectives in both case studies, despite the distinct nature of their optimization landscapes.

In the pDNA case study, the problem involves strictly sequential unit operations and scheduling constraints such as the non-decreasing fermentation times which are enforced across all benchmark methods $( t _ { \mathrm { f l a s k } } \leq t _ { \mathrm { s e e d } } \leq t _ { \mathrm { m a i n } } )$

![](images/3840f9c3f152e91feecff44a062f4ee19aded7468f2096c99aa73b931fcf5ea1.jpg)

![](images/cb91b67b399cc1b0e29893f2d679c4fd91f8747f276c3aa1e625d2cac25929ab.jpg)

![](images/35827535cad0833fc98a8e56aecc248a2ac09057990facd0c8fda212b16d4bfe.jpg)

![](images/c04efbd9b2139df710c9f3e7ae73c75dba2a57c80922bbfe704ffd841f125354.jpg)

![](images/9e404734572855ddf4cc06186b1f17c7558b0cbd91519a578d94eb8cdb670039.jpg)  
Fig. 3. Optimization results for all objectives for case study II: best observed highfidelity value vs. function evaluations. Surrogate-based methods are validated with the simulator.

Optimization trajectories for the pDNA bioprocess are presented in Figure 2. RS-MFBO demonstrates robust convergence across most objectives, though the results highlight the inherent trade-ofs of dimensionality reduction. For economic objectives such as Production Cost (Fig. 2f), RS-MFBO matches the final median performance of the high-fidelity baselines, converging to the optimum within the first 10 to 15 evaluations. Notably, for CapEx (Fig. 2d), RS-MFBO achieves a visibly lower final median value than both vanilla BO and reduced-space BO (RS-BO), suggesting that the multi-fidelity exploration helped escape local optima that trapped the greedy single-fidelity methods. However, in the case of OpEx (Fig. 2e), the reduced-space methods (both RS-BO and RS-MFBO) slightly underperform compared to the full-space vanilla BO. This behavior is expected in reduced-space optimization; by fixing non-dominant variables to nominal values, the optimizer operates on a restricted manifold that may exclude the true global optimum if those non-critical variables have small but non-zero efects.

Performance on operational KPIs varies by dificulty. For Batch Size (Fig. 2a), all optimization-based methods converge to a similar high-throughput solution, significantly outperforming random sampling. In contrast, Cycle Time (Fig. 2c) proves to be an easy optimization landscape where even the baseline Sobol sampling identifies the optimal solution immediately. In these simpler tasks, RS-MFBO still ofers value by confirming optimality with fewer expensive checks than standard BO. Throughout these experiments, the pure surrogate-based baselines (RS-ANN-BO and RS-ANN-MILP) appear as single points in the figures. While these methods incur the lowest computational cost, their accuracy is unreliable. RS-ANN-MILP often identifies designs that appear optimal in silico but degrade significantly when validated against the true high-fidelity simulator, highlighting the necessity of the feedback loop provided by RS-MFBO.

The optimization results for the continuous DME synthesis process are summarized in Figure 3. This case study poses a distinct challenge due to complex thermodynamic recycle loops that frequently destabilize the process simulation. RS-MFBO leverages the smooth low-fidelity surrogate to navigate this landscape eficiently. This is most evident in the Energy Eficiency maximization (Fig. 3b). Here, RS-MFBO exploits the high-quality region identified by the low-fidelity surrogate to initialize the search near 80% eficiency, avoiding the initial exploration phase that delays standard BO and RS-BO, which start significantly lower (between 70% and 76%). While the single-fidelity methods eventually climb to competitive values, RS-MFBO maintains a performance lead throughout the budget, reaching peak eficiencies of approximately 84%.

Similar advantages are observed for the DME Production Rate (Fig. 3a), where RS-MFBO identifies the optimal operating region (approx. 4.0 kg/s) almost immediately. In contrast, standard vanilla BO shows a delayed convergence, requiring over 40 function evaluations to match the performance that RS-MFBO achieves in fewer than 10. For economic objectives like OpEx (Fig. 3d) and CapEx (Fig. 3e), the method exploits the high quality of the initial surrogate to flatten the cost curve instantly. While vanilla BO eventually converges to a similar cost, it incurs a significant regret penalty during the initial exploration phase. The diference in convergence speed between RS-MFBO and the single-fidelity baselines across these objectives highlights the method’s ability to handle highdimensional search spaces efectively.

## 5.2 Efect of reduced-space modeling

The comparison between vanilla Bayesian optimization (BO), reduced-space BO (RS-BO), and RS-MFBO highlights the benefits of GSA-based dimensionality reduction across both simulators. In several SuperPro objectives, RS-BO converges faster than vanilla BO, indicating that restricting the search to the most influential variables can improve sample eficiency. RS-MFBO inherits this advantage and further augments it with a fidelity-aware GP and cost-aware acquisition policy, yielding the best overall cost-performance trade-of. From a modeling standpoint, operating in the reduced space improves GP conditioning and leads to more stable hyperparameter estimates. In the full 18-dimensional pDNA problem and the 14-dimensional DME problem, kernel hyperparameter optimization is often ill-conditioned, leading to over-smoothing and large posterior variance. By retaining only the most influential variables (typically 6–8 depending on the objective), the reduced-space formulation produced more stable marginal likelihood fits and better-calibrated uncertainty estimates. Furthermore, the reduced space significantly accelerates the internal acquisition optimization step, cutting its wall-time by approximately 40–60% on average compared to the full-space operation.

![](images/7e34ac7dbc38d5748de910981a4b951c69a5d3163bdef5d5c14be1b29dfd8d4e.jpg)  
Fig. 4. High fidelity (HI) evaluations vs. iteration for the pDNA case study.

## 5.3 Acquisition dynamics and use of high-fidelity evaluations

Analysis of the cumulative high-fidelity (HI) simulator evaluations reveals a characteristic “staircase” usage profile for RS-MFBO across all 12 objectives (see Figures 4, 5). The flat regions of this profile correspond to extended phases of low-fidelity exploration where the acquisition function exploits the cheap surrogate or explores regions with high surrogate uncertainty. Vertical jumps occur only when the cooldown mechanism forces a correction or when the promotion heuristic identifies a promising candidate from the low-fidelity model. This behavior arises naturally from the cost-aware acquisition policy, which prioritizes the cheaper ANN surrogate unless the cost-adjusted UCB score or the cooldown interval indicates the need for high-fidelity refinement.

Across both the SuperPro and Aspen HYSYS case studies, RS-MFBO uses substantially fewer HI simulator calls than the single-fidelity BO baselines. While vanilla BO and reduced-space single-fidelity BO query the simulator at every iteration, RS-MFBO performs the majority of its evaluations on the low-fidelity surrogate and relies on selective high-fidelity queries to maintain surrogate accuracy. The $\ell _ { \infty } .$ -based deduplication step further avoids redundant re-evaluations at nearly identical conditions. Overall, RS-MFBO attains competitive or improved final objective values while using approximately 65–80% fewer high-fidelity evaluations than pure BO under the same total cost budget. This eficiency is consistent across both case studies.

## 5.4 Computational time

Figure 6 compares the wall-time for all methods. As expected, pure BO incurs the highest computational cost because every step involves a simulator call in the full input space. In the SuperPro Designer case study (Fig. 6a), individual simulations take approximately 10–15 seconds. However, the total runtime for vanilla BO is disproportionately high. This is primarily because optimizing the acquisition function within the full 18-dimensional space, subject to the two constraints, is computationally intensive. Reduced-space single-fidelity BO (RS-BO) reduces wall-time relative to BO by operating in a smaller subspace, but its reliance on high-fidelity evaluations still dominates the overall runtime.

![](images/26ade9329c4a206054a8169a3ad7f026aac5cd972078dae18f4353a5bd301318.jpg)

![](images/0c61248f4ffff4435986f2a1d107c23a9a7b892043738176fa71151e5093f2e4.jpg)

![](images/8c120ae930ed3e1f15a54da1b1647267b9b02ff24120472069f84741445c1590.jpg)

![](images/6b7f2b0e2ca52aded8bbd09aedddfb0233e13f1b82be3a5ffdcad08d4b444297.jpg)

![](images/ff51d93248b45a9d5ab74fc7eb1984dfba38f5e4e1aa4f3bfc92944b94e136fe.jpg)

Fig. 5. High fidelity (HI) evaluations vs. iteration for the DME case study.  
![](images/ddf98953d2d785855f53634910184ef6bd891c4b9c215a612a58654c99662659.jpg)

![](images/a741ef94c4855f4ac5800619bff52ae066216ca0e1ae6ea6efd16f46fffe6da2.jpg)  
Fig. 6. Execution time by method and objective.

In the Aspen HYSYS case study (Fig. 6b), simulation times vary significantly (ranging from 5 to 45 seconds) depending on the dificulty of converging the recycle loops. Here, RS-MFBO consistently lies among the fastest methods. The wall-time reduction stems from two factors: (i) most iterations query the cheap low-fidelity surrogate, avoiding the heavy overhead of unstable simulator calls, and (ii) GSA restricts both the GP fit and the acquisition optimization to a reduced number of variables. In many objectives, RS-MFBO achieves nearbest objective values within a fraction of the runtime of BO and RS-BO, making it more suitable for time- and budget-constrained simulation models. The surrogate-only methods (RS-ANN-BO and RS-ANN-MILP) are the fastest in absolute terms because they avoid simulator calls altogether (except for final validation). However, their performance is ultimately bounded by the accuracy of the ANN. When the surrogate misrepresents the true global optimum, the surrogate-only methods converge to suboptimal designs. In contrast, RS-MFBO ofers a better trade-of between computational cost and solution quality by systematically combining low- and high-fidelity information.

## 6 Conclusion

We presented a reduced-space multi-fidelity Bayesian optimization framework tailored to simulation-based models and evaluated it on two diferent simulators. The combination of GSA-based dimensionality reduction, a fidelity-augmented Gaussian process with a cross-fidelity kernel, and a cost-aware acquisition strategy with cooldown and promotion enabled eficient exploration while limiting the number of expensive simulator calls. Across the KPIs studied, the method achieved competitive final performance while using substantially fewer HI evaluations than single-fidelity Bayesian optimization.

Future work includes extending the fidelity hierarchy beyond two levels, and developing learned fidelity selection policies for broader applicability.

Acknowledgments. NT and MMP would like to thank Ben Lyons, Benoit Chachuat and Cleo Kontoravdi for their contributions and input on the construction of the flowsheets and insights on the case studies. NT is thankful for the Marit Mohn Scholarship awarded by the Department of Chemical Engineering, Imperial College London.

## References

1. Archetti, F., Candelieri, A.: Bayesian optimization and data science, vol. 849. Springer (2019)

2. Balandat, M., Karrer, B., Jiang, D.R., Daulton, S., Letham, B., Wilson, A.G., Bakshy, E.: BoTorch: A Framework for Eficient Monte-Carlo Bayesian Optimization. In: Advances in Neural Information Processing Systems 33 (2020)

3. Candelieri, A.: Sequential model based optimization of partially defined functions under unknown constraints. Journal of Global Optimization 79(2), 281–303 (2021)

4. Candelieri, A., Ponti, A., Archetti, F.: Fair and green hyperparameter optimization via multi-objective and multiple information source bayesian optimization. Machine Learning 113(5), 2701–2731 (2024)

5. Candelieri, A., Ponti, A., Archetti, F., Sabatella, A.: Multiple Information Source Bayesian Optimization. Springer (2025)

6. Ceccon, F., Jalving, J., Haddad, J., Thebelt, A., Tsay, C., Laird, C.D., Misener, R.: Omlt: optimization & machine learning toolkit. J. Mach. Learn. Res. 23(1) (Jan 2022)

7. Eriksson, D., Pearce, M., Gardner, J., Turner, R.D., Poloczek, M.: Scalable global optimization via local bayesian optimization. Advances in neural information processing systems 32 (2019)

8. Garnett, R.: Bayesian optimization. Cambridge University Press (2023)

9. Hernández-Morales, G., Cansino-Loeza, B., Jiménez-Gutiérrez, A., Zavala, V.M.: Simulation-based optimization over discrete spaces using projection to continuous latent spaces (2025), https://arxiv.org/abs/2510.14206

10. Iwanaga, T., Usher, W., Herman, J.: Toward salib 2.0: Advancing the accessibility and interpretability of global sensitivity analyses. Socio-Environmental Systems Modelling 4, 18155 (May 2022). https://doi.org/10.18174/sesmo.18155

11. Kandasamy, K., Dasarathy, G., Schneider, J., Póczos, B.: Multi-fidelity bayesian optimisation with continuous approximations. In: 34th International Conference on Machine Learning, ICML 2017. pp. 2861–2878. International Machine Learning Society (IMLS) (2017)

12. Kennedy, M.C., O’Hagan, A.: Predicting the output from a complex computer code when fast approximations are available. Biometrika 87(1), 1–13 (2000)

13. Martens, A., Neufang, M., Butté, A., von Stosch, M., del Rio Chanona, A., Helleckes, L.M.: Holistic bioprocess development across scales using multi-fidelity batch bayesian optimization (2025), https://arxiv.org/abs/2508.10970

14. McDonald, M.A., Koscher, B.A., Canty, R.B., Zhang, J., Ning, A., Jensen, K.F.: Bayesian optimization over multiple experimental fidelities accelerates automated discovery of drug molecules. ACS central science 11(2), 346–356 (2025)

15. Nayebi, A., Munteanu, A., Poloczek, M.: A framework for bayesian optimization in embedded subspaces. In: International Conference on Machine Learning. pp. 4752–4761. PMLR (2019)

16. Paulson, J.A., Tsay, C.: Bayesian optimization as a flexible and eficient design framework for sustainable process systems. Current Opinion in Green and Sustainable Chemistry 51, 100983 (2025)

17. Poloczek, M., Wang, J., Frazier, P.: Multi-information source optimization. Advances in neural information processing systems 30 (2017)

18. Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M., Tarantola, S.: Variance based sensitivity analysis of model output. design and estimator for the total sensitivity index. Computer Physics Communications 181(2), 259–270 (2010)

19. Savage, T., Basha, N., McDonough, J., Krassowski, J., Matar, O., del Rio Chanona, E.A.: Machine learning-assisted discovery of flow reactor designs. Nature Chemical Engineering 1(8), 522–531 (2024)

20. Shields, B.J., Stevens, J., Li, J., Parasram, M., Damani, F., Alvarado, J.I.M., Janey, J.M., Adams, R.P., Doyle, A.G.: Bayesian reaction optimization as a tool for chemical synthesis. Nature 590(7844), 89–96 (2021)

21. Song, J., Chen, Y., Yue, Y.: A general framework for multi-fidelity bayesian optimization with gaussian processes. In: The 22nd International Conference on Artificial Intelligence and Statistics. pp. 3158–3167. PMLR (2019)

22. Thebelt, A., Tsay, C., Lee, R., Sudermann-Merx, N., Walz, D., Shafei, B., Misener, R.: Tree ensemble kernels for bayesian optimization with known constraints over mixed-feature spaces. Advances in Neural Information Processing Systems 35, 37401–37415 (2022)

23. Triantafyllou, N., Lyons, B., Bernardi, A., Chachuat, B., Kontoravdi, C., Papathanasiou, M.M.: Comparative assessment of simulation-based and surrogatebased approaches to flowsheet optimization using dimensionality reduction. Computers & Chemical Engineering 189, 108807 (2024)

24. Triantafyllou, N., Sarkis, M., Krassakopoulou, A., Shah, N., Papathanasiou, M.M., Kontoravdi, C.: Uncertainty quantification for gene delivery methods: A roadmap for pdna manufacturing from phase i clinical trials to commercialization. Biotechnology Journal 19(1), 2300103 (2024)

25. Wang, Z., Hutter, F., Zoghi, M., Matheson, D., De Feitas, N.: Bayesian optimization in a billion dimensions via random embeddings. Journal of Artificial Intelligence Research 55, 361–387 (2016)

26. Wu, J., Toscano-Palmerin, S., Frazier, P.I., Wilson, A.G.: Practical multi-fidelity bayesian optimization for hyperparameter tuning. In: Uncertainty in Artificial Intelligence. pp. 788–798. PMLR (2020)