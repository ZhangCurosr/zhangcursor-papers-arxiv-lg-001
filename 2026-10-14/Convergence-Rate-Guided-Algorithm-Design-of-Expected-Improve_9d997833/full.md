# Convergence-Rate-Guided Algorithm Design of Expected Improvement for Continuous Simulation Optimization

Songhao Wang<sup>1,3</sup>, Jianglin Xia<sup>1</sup>, Haowei Wang<sup>2</sup>, and Szu Hui Ng<sup>2</sup>

<sup>1</sup>College of Business, Southern University of Science and Technology, Shenzhen, CHINA   
<sup>2</sup>Department of Industrial Systems Engineering and Management, National University of Singapore, Singapore, SINGAPORE <sup>3</sup>Corresponding author: wangsh2021@sustech.edu.cn

## Abstract

Expected improvement (EI) based on stochastic kriging (SK) surrogate is widely used in continuous simu lation optimization, but unknown heteroscedastic noise on simulation output makes three basic design choices consequential: the incumbent used to define improvement, the replication level at each sampled point, and the terminal recommendation. Existing methods typically either assume the noise levels are known or choose these components separately through heuristic rules or empirical calibration. We instead derive a high-probability upper bound on simple regret and systematically choose the three components within general rule classes to tighten this bound. We first establish a finite-sample confidence bound for the SK predictor with unknown and estimated noise levels as a foundation for the three design choices. We then analyze general posterior-quantile families for both the incumbent and the terminal recommendation. For the incumbent, minimizing the regret bound selects a lower quantile coeficient that matches the coeficient in the SK predictor confidence bound. For the terminal recommendation, minimizing the same bound instead selects the sampled point with the best upper posterior quantile under the same coeficient. These opposite quantile choices reflect the diferent roles of search and implementation in simulation optimization. Intermediate simulations are exploratory with an optimistic incumbent, while the terminal recommendation is implemented in the real system and is therefore relatively conservative about the remaining posterior uncertainty. Under mild conditions satisfied by common kernels, minimizing a tractable kernel-specific bound yields Θ(log T) replications. For the squared-exponential kernel, the resulting simple regret is Oe(T<sup>−1/2</sup>), matching the polynomial dependence on the raw simulation budget in the minimax lower bound. For a Matérn kernel, the derived rate also matches the corresponding confidence-bound rate under known homoscedastic noise up to logarithmic factors. To our knowledge, this is the first finite-budget high-probability simple-regret guarantee of this form for a continuous-domain EI-type simulation optimization method that learns unknown heteroscedastic variances through replications.

## 1 Introduction

Simulation optimization studies decision problems in which candidate decision points are evaluated by running simulation models. It is widely used when the system of interest is analytically intractable, or too complex to optimize directly. Representative applications include manufacturing and production scheduling, healthcare resource planning, transportation, and supply chain management (see, e.g., Amaran et al. 2016, Wang and Demeulemeester 2023, Osorio and Bierlaire 2013). A typical simulation optimization procedure begins with a decision-maker providing a certain amount of computational resources. An algorithm then allocates these resources among selected decision points to run experiments. Each run produces a random output. Once the budget is exhausted, a final recommended solution is required to be implemented in the real-world system.

Surrogate-based algorithms are a common and eficient strategy for simulation optimization. They fit a surrogate model to the simulation results collected so far and use it to direct subsequent simulations toward inputs that appear promising or remain uncertain. Gaussian process (GP), or kriging, models are especially useful. They provide both a predictor and a measure of predictive uncertainty (Williams and Rasmussen 2006). Stochastic kriging (SK) extends this model by separating uncertainty about the response surface from the intrinsic noise in stochastic simulation outputs (Ankenman et al. 2010). GP-based search has consequently become an important methodology in the simulation literature, including discrete optimization via simulation, Bayesian optimization with correlated sampling, and continuous optimization via simulation (see, e.g., Sun et al. 2014, Xie et al. 2016, Pearce et al. 2022, Wang et al. 2020, Wang et al. 2025). Within this broader class, expected improvement (EI) is one of the most widely used sequential selection rules. Introduced in eficient global optimization (EGO) for deterministic black-box functions, EI selects the next design by its expected improvement over a current best value (also called incumbent value). It combines predicted performance and predictive uncertainty in a simple closed-form criterion (Jones et al. 1998). Several approaches adapt this idea to stochastic simulation, such as sequential kriging optimization (SKO, Huang et al. 2006), expected quantile improvement (EQI, Picheny et al. 2013), and two-stage stochastic optimization (TSSO, Quan et al. 2013).

EI approaches have also been used in many areas outside simulation optimization, including engineering design and computer science, typically presented in bandit-style problems (Shahriari et al. 2015). However, simulation optimization difers from these settings in two important respects. The first diference is the noise structure. Many theoretical analyses in GP bandits assume a known, homoscedastic, or otherwise simple noise model (see, e.g., Srinivas et al. 2010). In stochastic simulation, the output variance is often input-dependent and unknown. Queueing simulations provide a classical example. Response variability can change substantially with trafic intensity, and nearly saturated queues are known to produce highly heteroscedastic simulation responses (Cheng and Kleijnen 1999). Similar noise structures arise in manufacturing, epidemic management, and other operational systems (Binois et al. 2018). In this setting, a single run cannot determine whether an observation reflects the underlying expected response or merely local simulation variability. Replication at the same input is therefore essential for estimating local variance and informing the surrogate model about the reliability of a sample mean. The second diference lies in the performance criterion. Much of the GP-bandit literature studies cumulative regret because each sampled input is treated as a realized decision. Simulation optimization instead treats the sampled inputs as exploratory experiments. As only the design implemented at the end matters, the relevant criterion is simple regret, the optimality gap of the final recommendation. It must therefore certify the quality of that recommendation, not the sequence of sampled points.

These simulation-specific features turn three seemingly routine implementation choices into central design decisions: the incumbent value, the replication allocation, and the terminal recommendation. It is worth emphasizing that these dificulties do not arise in the same form when evaluations are noise-free. Some may also disappear under simpler or fully specified noise structures. In deterministic EI, for example, the incumbent can be taken directly as the best objective value observed so far (Jones et al. 1998, Bull 2011). Replication is unnecessary because repeated evaluations at the same input return the same value, and the terminal recommendation can simply be the best input evaluated during the search. Once simulation outputs exhibit a complex noise structure, however, every historical response is contaminated by potentially heteroscedastic noise. The three choices that are natural in deterministic EI therefore become nontrivial and potentially consequential.

Consider first how noise afects the choice of the current best. Using either a raw observation or a sample mean from several replications may select an incumbent point that appears exceptionally good only because of favorable noise, even though its true mean performance is much worse. Such a downward distortion can make genuine improvements appear unlikely and misdirect the subsequent search. Existing studies therefore use more stable information supplied by the GP model, such as the best posterior prediction or a posterior quantile. A broad family of such choices can be written as $\xi _ { n } ^ { ( c _ { n } ) } : = \operatorname* { m i n } \{ \mu _ { n } ( x ) - c _ { n } \sigma _ { n } ( x ) \}$ , where x represents the decision point, $\mu _ { n } ( x )$ and $\sigma _ { n } ( x )$ are the SK prediction and predictive standard deviation, respectively, and $c _ { n } \in \mathbb { R }$ is the quantile coeficient after selecting n decision points. The choice $c _ { n } = 0$ gives the best-posterior-prediction incumbent used in noisy-EI analyses (Wang and de Freitas 2014, Gupta et al. 2022). SKO sets $c _ { n } ~ = ~ - 1$ , whereas EQI uses a prespecified $c _ { n } \leq 0 $ ; both choices correspond to upper quantiles (Huang et al. 2006, Picheny et al. 2013, Jalali et al. 2017). Although these choices have intuitive justifications, their coeficients are typically prescribed in advance or treated as tuning parameters based on convention or empirical performance.

Similar discretion appears in the other two design decisions. How to allocate a limited simulation budget across candidate designs is itself a central problem in simulation optimization (Chen et al. 2000, Ankenman et al. 2010, Binois et al. 2019). In stochastic EI, the number of replications assigned to a design determines the accuracy of its sample mean and local noise-variance estimate. Under a fixed raw simulation budget, it also determines how many distinct designs the algorithm can explore. Existing EI procedures use several diferent mechanisms, including equal allocation, repeated sampling of past points, separate allocation rules such as optimal computing budget allocation (OCBA), or user-specified stopping thresholds (Huang et al. 2006, Picheny et al. 2013, Quan et al. 2013, Pedrielli et al. 2020, Jalali et al. 2017). As with the incumbent choice discussed above, quantities such as equal allocation replication sizes, allocation-stage budgets, and stopping thresholds often remain external inputs whose values must be fixed in advance or calibrated empirically. Terminal recommendations are similarly heterogeneous: diferent methods return the sampled design with the best sample mean, the best GP prediction, or the best posterior quantile (Picheny et al. 2013, Quan et al. 2013, Jalali et al. 2017).

Taken together, these alternatives allow many combinations of design choices. This flexibility, however, can leave practitioners uncertain about which combination to use or what guarantee it provides. A more systematic and coherent approach is therefore needed to select these components while reducing reliance on heuristic or empirically calibrated settings. Although empirical studies have compared diferent algorithms numerically (Jalali et al. 2017), they do not provide a unified theoretical basis for selecting among them. We address this need from a theoretical perspective. Specifically, we begin with a generic EI procedure in which all three components are left general within broad rule classes. We then derive a high-probability upper bound on the simple regret of this generic procedure. The three design choices are systematically selected to tighten this bound. To our knowledge, no existing continuousdomain EI method provides a comparable high-probability simple-regret bound under heteroscedastic noise with unknown input-dependent variances estimated from replications.

In our work, the simple regret bound for the algorithm is built upon a finite-sample confidence bound for the SK predictor. At each selected design, independent replications provide a sample mean and a sample-variance estimate of the unknown noise variance. As a sample variance can underestimate the true variance, directly treating it as known can make the confidence band too narrow. We derive a correction factor for this underestimation and obtain a confidence event that holds jointly over the entire design space and all simulation iterations. The resulting confidence coeficient, denoted by $\Lambda _ { n }$ after n sampled designs, becomes the common calibration scale for the search and recommendation rules. To our knowledge, this is the first finite-budget SK confidence bound for adaptive designs with unknown heteroscedastic noise estimated from replications. Unlike Chen et al. (2024), which treats estimated variances asymptotically, our bound provides finite-sample coverage and directly supports the EI regre analysis.

Building on this confidence guarantee for the SK predictor, we derive a simple-regret upper bound for a generic EI procedure. The bound is generic because none of the three design components has yet been fixed. It accommodates the global linear-quantile incumbent family described above, an analogous family of terminal rules that use the same linear posterior-quantile form to select among sampled points, and any admissible replication size. We then select the three design components to tighten this certificate. Within the general global linear-quantile incumbent family, minimizing the incumbent-dependent part of the certificate selects $c _ { n } = \Lambda _ { n }$ . The resulting lower-quantile incumbent keeps a region in contention whenever the SK confidence bound cannot yet rule out that it contains an optimizer. Within the general family of terminal posterior-quantile rules, the terminal-dependent part is minimized by an upper quantile with coeficient equal to the final confidence width of the SK predictor. This choice therefore accounts conservatively for its remaining uncertainty. Unlike the first two design choices, the replication size depends on the specific covariance kernel used by the GP. After determining the first two parameters, we therefore introduce mild conditions on the kernel that are satisfied by standard kernels, including squared-exponential (SE), MatÃ©rn, and polynomial kernels. Under these conditions, we show that the optimal replication size has order Θ(log T) given a total budget of T simulation replications. Thus, the simple-regret analysis yields an optimistic incumbent for search, logarithmic replication for learning unknown heteroscedastic variances while preserving spatial exploration, and a conservative sampled-point recommendation for implementation.

Under these calibrated design choices, we obtain the optimized high-probability simple-regret guarantee. For SE kernels, the resulting simple regret is of order $\tilde { O } ( T ^ { - 1 / 2 } )$ , matching the polynomial dependence in the minimax lower bound for noisy GP optimization. For Matérn kernels, the rate is ${ \tilde { O } } ( T ^ { - \frac { 2 \nu - d } { 2 ( 2 \nu + d ) } } )$ ), where $\tilde { O }$ suppresses logarithmic factors, ν is the smoothness parameter of the Matérn kernel, and d is the dimension of the design space. It also matches, up to logarithmic factors, existing confidence-based regret bounds under the simpler setting of known homoscedastic noise, demonstrating the strength of our guarantee (Gupta et al. 2022, Vakili et al. 2021b). We highlight that our bound is a stronger and more decision-relevant form of guarantee for simulation optimization than Bayesian simple regret (Takeno et al. 2025, Wang et al. 2025), which averages over random functions, noises, and algorithmic randomness. In addition, we find that estimating the unknown heteroscedastic variances throug logarithmic replication introduces at most an additional logarithmic factor into the regret bound. It therefore leaves the leading polynomial dependence on $T$ unchanged.

The contributions of this paper are fourfold. First, we derive a finite-sample confidence bound for the SK pre dictor. This bound is not specific to EI. It can also support other SK-based sequential-design methods. Second, we analyze a general family of global linear-quantile incumbents. This family includes the best posterior predic tion and common posterior-quantile choices. Our analysis identifies the quantile level that removes a potentially nonvanishing term in the simple-regret certificate. Thus, the quantile level is determined by theory rather than treated as a tuning parameter. Third, we study replication as a budget-allocation decision. Under mild kernel conditions, we show that $b _ { T } = \Theta ( \log T )$ is order-optimal for the resulting regret bound. Fourth, We analyze a general family of terminal posterior quantiles over actually sampled designs. The confidence-matched upper quantile is a bound-minimizing choice for the displayed uniform certificate and yields a finite-budget high-probability simple-regret guarantee. To our knowledge, this is the first such bound for an EI-type simulation optimization method with unknown heteroscedastic noise.

The remainder of the paper is organized as follows. Sections 2 and 3 review the related literature and introduce the necessary background on SK and EI. Section 4 presents the generic EI procedure with general design choices and establishes the confidence bound for the SK predictor. Section 5 derives the simple-regret certificate for the generic EI procedure. Section 6 uses this certificate to select the three design components and establish the resulting convergence rates. Section 7 presents the numerical experiments, and Section 8 concludes the paper. The proofs of all the theoretical results are presented in the Appendix.

## 2 Related Works

Jones et al. (1998) introduced the EGO framework and made EI its central infill criterion for expensive deterministic black-box optimization. In this noise-free setting, Vazquez and Bect (2010) established asymptotic convergence properties of EI. Bull (2011) later derived simple-regret rates for EI and modified epsilon-greedy EI algorithms. We next review literature in two related areas: EI in simulation optimization and noisy EI theory in GP bandits.

## 2.1 EI in simulation optimization

In stochastic simulation, direct use of EGO is inadequate because simulation outputs are noisy. This motivated several extensions of EI. Sequential kriging optimization (SKO) modifies EI through augmented expected improvement (Huang et al. 2006). Its incumbent is the SK prediction at an “efective best” sampled point identified by minimizing an upper posterior quantile. Its heteroscedastic implementation studied by Jalali et al. (2017) assigns a fixed replication for selected design and returns the sampled point with the smallest posterior quantile. EQI defines improvement directly through posterior quantiles (Picheny et al. 2013). Its incumbent and final recommendation are both based on the smallest upper posterior quantile among previously sampled points. It may revisit a sampled design and add replications until the updated EQI falls below a chosen fraction of its initial value. The quantile level, replication increment, and stopping fraction are tuning parameters. TSSO separates search from allocation (Quan et al. 2013). Its modified EI uses the SK prediction at the design with the smallest sample mean as its in cumbent. A fixed iteration budget is divided between the new design and OCBA allocation among sampled designs (Chen et al. 2000). The final recommendation has the smallest sample mean. eTSSO retains these incumbent and recommendation but makes the iteration budget adaptive (Pedrielli et al. 2020).

Our work also relates to other GP-based simulation optimization frameworks. GP-based search constructs a sampling distribution for discrete optimization via simulation and establishes global convergence (Sun et al. 2014). Knowledge-gradient methods incorporate correlated beliefs and common random numbers (Xie et al. 2016, Pearce et al. 2022). GP random search (GPRS) studies continuous optimization via simulation and proves a convergence rate under known heteroscedastic noise (Wang et al. 2025). These methods are not EI procedures, but they show how surrogate modeling, sampling, and noise estimation can be integrated in other simulation optimization frameworks.

Replication has also been studied outside the EI literature. Classical SK experiment design allocates a fixed budget over a fixed set of sites to improve prediction accuracy (Ankenman et al. 2010). Binois et al. (2019) use integrated mean squared prediction error criterion to choose between a new design and a replicate. Their objective is global surrogate accuracy rather than optimization regret. GPRS proves convergence and a rate under known heteroscedastic noise variances, while its unknown-variance implementation is studied only numerically (Wang et al. 2025). CGLO requires every sampled design to receive a growing number of replications and establishes asymptotic convergence under known noise variances (Meng et al. 2022).

Prior research in this area has established several types of theoretical results. OCBA provides asymptotic allocation theory for finite alternatives (Chen et al. 2000). In a normal ranking-and-selection model, Ryzhov (2016) characterize the asymptotic allocations of EI and relate them to OCBA. eTSSO and several GP search methods also establish asymptotic guarantees under their respective sampling structures. The missing result is a finite-budget analysis that jointly determines the incumbent, replication order, and final recommendation when unknown heteroscedastic variances are estimated from replications. Existing quantile levels, batch sizes, and allocation thresholds are largely treated as tuning parameters. Jalali et al. (2017) also show that an inappropriate recommendation rule can ofset the gains achieved during search.

## 2.2 Noisy EI theory in GP bandits

As noted earlier, noisy GP-bandit theory considers a related but diferent setting. It usually treats each observation as one round, assumes a fixed noise model, and focuses on cumulative regret. Replication allocation and final identification are therefore not its main objects.

One line of work rescales and inflates the posterior uncertainty used inside EI, forcing more exploration. (Wang and de Freitas 2014, Gupta et al. 2022, Hu et al. 2025). The drawback is that the acquisition function is no longer the ordinary EI computed from the fitted GP posterior, and the theoretically required inflation can hurt empirical performance (Takeno et al. 2025). The second route randomizes the incumbent, for example by using the maximum of a posterior sample path (Takeno et al. 2025). This avoids deterministic variance inflation, but it introduces additional algorithmic randomness and usually requires approximate posterior sampling in continuous or large input spaces. Moreover, the available guarantee is limited to weaker Bayesian regret, averaged over the GP posterior, observation noise, and algorithmic randomness, rather than a high-probability simple-regret guarantee for a returned design.

Beyond EI, noisy GP-bandit theory is more developed. GP-UCB uses optimism and controls high-probability cumulative regret through confidence widths and maximum information gain (Srinivas et al. 2010). Improved GP-UCB and GP-TS provide frequentist RKHS guarantees (Chowdhury and Gopalan 2017), while GP-PIMS provides a Bayesian posterior-sampling guarantee (Takeno et al. 2024). For pure exploration, maximum variance reduction returns the posterior-mean maximizer and achieves an order-optimal simple-regret bound up to logarithmic factors for common kernels (Vakili et al. 2021a, Cai and Scarlett 2021). MINI-META and BTS-RED also repeat observations, but target cumulative regret, computational cost, or a prescribed variance threshold (Calandriello et al. 2022, Dai et al. 2023).

The distinction is therefore both statistical and decision-oriented. Noisy EI theory in GP bandits explains how an incumbent or acquisition rule can control regret under a specified observation model. Our setting additionally requires the algorithm to learn that model through replication and to return a single sampled design. To our knowledge, existing bandit results do not use one finite-budget, high-probability simple-regret bound to select the incumbent, replication level, and terminal recommendation together.

## 3 Preliminaries

## 3.1 Stochastic Kriging and Expected Improvement

We consider the minimization of an unknown latent mean response $f : \mathcal { X } $ R with the following assumption on design space $\mathcal { X }$

Assumption 3.1. The design space X is a nonempty compact subset of $\mathbb { R } ^ { d }$

SK models the latent response surface f by a $\mathrm { G P } , \mathcal { G P } ( \mu _ { 0 } , k _ { \theta } )$ , where $\mu _ { 0 }$ is a prior mean and $k _ { \theta }$ is a positive definite kernel with hyperparameter θ. As this paper adopts a frequentist perspective under which f is treated as a fixed function in a Hilbert space (see details later in Section 3.3), we follow the common convention in this line of analysis by using a zero-mean GP as the working surrogate, i.e., $\mu _ { 0 } = 0$ . Popular choices of kernel functions include the SE ker nel $\begin{array} { r } { k _ { \ell } ^ { \mathrm { S E } } ( x , x ^ { \prime } ) = \exp \left( - \frac { 1 } { 2 } \rho _ { \ell } ^ { 2 } ( x , x ^ { \prime } ) \right) } \end{array}$ , and the Matérn-ν kernel $\begin{array} { r } { k _ { \ell , \nu } ^ { \mathrm { M a t } } ( x , x ^ { \prime } ) = \frac { 2 ^ { 1 - \nu } } { \Gamma ( \nu ) } \left( \sqrt { 2 \nu } \rho _ { \ell } ( x , x ^ { \prime } ) \right) ^ { \nu } K _ { \nu } \left( \sqrt { 2 \nu } \rho _ { \ell } ( x , x ^ { \prime } ) \right) } \end{array}$ where $K _ { \nu }$ is the modified Bessel function of the second kind and $\begin{array} { r } { \rho _ { \ell } ( x , x ^ { \prime } ) = \left( \sum _ { j = 1 } ^ { d } \frac { ( x _ { j } - x _ { j } ^ { \prime } ) ^ { 2 } } { \ell _ { j } ^ { 2 } } \right) ^ { 1 / 2 } } \end{array}$ . For notational simplicity, the kernel variance scale is normalized to one. A separate scale parameter can be incorporated into θ if needed.

In stochastic simulation, evaluating a decision point x produces random output $Y ( x ) = f ( x ) + \varepsilon ( x )$ , where $\varepsilon ( x )$ is a normal noise following $N ( 0 , v ( x ) )$ . The variance function $v ( x )$ depends on the input location. Given n decision points $x _ { 1 } , \ldots , x _ { n }$ with $b _ { i }$ replications at $x _ { i } .$ denote the sample mean and sample variance at $x _ { i }$ as ${ \bar { Y } } _ { i }$ and $S _ { i } ^ { 2 }$ , respectively. The SK posterior mean and variance are

$$
\mu _ { n } ( x ) = k _ { n } ( x ) ^ { \top } ( K _ { n } + R _ { n } ) ^ { - 1 } \bar { Y } _ { 1 : n } ,\tag{1}
$$

$$
\sigma _ { n } ^ { 2 } ( x ) = k _ { \theta } ( x , x ) - k _ { n } ( x ) ^ { \top } ( K _ { n } + R _ { n } ) ^ { - 1 } k _ { n } ( x ) ,\tag{2}
$$

where $\bar { Y } _ { 1 : n } ~ = ~ ( \bar { Y } _ { 1 } , \ldots , \bar { Y } _ { n } ) ^ { \top } , ~ k _ { n } ( x ) ~ = ~ \big ( k _ { \theta } ( x , x _ { 1 } ) , \ldots , k _ { \theta } ( x , x _ { n } ) \big ) ^ { \top }$ , the matrix $K _ { n } ~ = ~ \left( k _ { \theta } ( x _ { i } , x _ { j } ) \right) _ { i , j = 1 } ^ { n }$ represents the extrinsic uncertainty about the response surface, and $R _ { n } = \mathrm { d i a g } ( S _ { 1 } ^ { 2 } / b _ { 1 } , . . . , S _ { n } ^ { 2 } / b _ { n } )$ represents the intrinsic uncertainty due to stochastic simulation noise.

Given the posterior distribution, EI function at $x$ is

$$
\operatorname { E I } _ { n } ( x ; \xi _ { n } ) = \operatorname { \mathbb { E } } _ { n } { \big [ } I _ { n } ( x ; \xi _ { n } ) { \big ] } = \operatorname { \mathbb { E } } _ { n } { \big [ } { \big ( } \xi _ { n } - f ( x ) { \big ) } _ { + } { \big ] } ,
$$

where $\mathbb { E } _ { n } [ \cdot ]$ denotes conditional expectation given results of the n decision points, and $\xi _ { n }$ is the incumbent (current best) value. EI computes the expected improvement of response at x with respect to $\xi _ { n } . \mathrm { ~ H f ~ } \sigma _ { n } ( x ) > 0$ , this has the closed form

$$
\operatorname { E I } _ { n } ( x ; \xi _ { n } ) = \sigma _ { n } ( x ) \tau \biggl ( { \frac { \xi _ { n } - \mu _ { n } ( x ) } { \sigma _ { n } ( x ) } } \biggr ) ,
$$

where $\tau ( z ) = z \Phi ( z ) + \phi ( z )$ , and Φ and ϕ are the standard normal cumulative distribution function (cdf) and probability density function (pdf). If $\sigma _ { n } ( x ) = 0 , { \mathrm { E I } } _ { n } ( x ; \xi _ { n } ) = \big ( \xi _ { n } - \mu _ { n } ( x ) \big ) _ { - }$ +

## 3.2 Information Gain

Information gain quantifies the reduction in uncertainty about the latent response values provided by the observations. Let $A = \left\{ x _ { 1 } , \ldots , x _ { m } \right\} \subset { \mathcal { X } }$ be a finite set of decision points, and let $f _ { A } = ( f ( x _ { 1 } ) , \ldots , f ( x _ { m } ) ) ^ { \top }$ . Suppose the corresponding sample-mean observations are $\bar { Y } _ { A } = f _ { A } + \eta _ { A }$ , where $\eta _ { A } \sim N ( 0 , R _ { A } )$ , and $R _ { A } = \mathrm { d i a g } ( r _ { 1 } ^ { 2 } , . . . , r _ { m } ^ { 2 } )$ is a diagonal noise matrix. The entries $r _ { i } ^ { 2 }$ may difer across decision points, reflecting heteroscedastic simulation noise and possibly diferent numbers of replications. Let $K _ { A } = \big ( k _ { \theta } ( x _ { i } , x _ { j } ) \big ) _ { i , j = 1 } ^ { m }$ . The information gain from observing $\bar { Y } _ { A }$ is the mutual information between $\bar { Y } _ { A }$ and $f _ { A } { \mathrm { : } }$

$$
\Gamma ( \bar { Y } _ { A } ; f _ { A } ) = H ( \bar { Y } _ { A } ) - H ( \bar { Y } _ { A } \mid f _ { A } ) = \frac { 1 } { 2 } \log \operatorname * { d e t } \left( I _ { m } + R _ { A } ^ { - 1 / 2 } K _ { A } R _ { A } ^ { - 1 / 2 } \right) ,
$$

since both $f _ { A }$ and $\bar { Y } _ { A }$ are Gaussian. Information gain is connected to posterior variance. For a sequential ordering $x _ { 1 } , \ldots , x _ { m }$ with noise variances $r _ { 1 } ^ { 2 } , \ldots , r _ { m } ^ { 2 }$ , the chain rule gives

$$
\Gamma ( \bar { Y } _ { A } ; f _ { A } ) = \frac { 1 } { 2 } \sum _ { i = 1 } ^ { m } \log \left( 1 + \frac { \sigma _ { i - 1 } ^ { 2 } ( x _ { i } ) } { r _ { i } ^ { 2 } } \right) ,
$$

where $\sigma _ { i - 1 } ^ { 2 } ( x _ { i } )$ is the GP posterior variance at $x _ { i }$ before observing the ith sample mean. This identity is the main reason information gain appears in regret analysis: it provides a way to control the cumulative posterior uncertaint at the points selected by the algorithm.

## 3.3 Reproducing Kernel Hilbert Spaces

The frequentist analysis of GP-based optimization usually measures smoothness through the reproducing kernel Hilbert space (RKHS) associated with the kernel. Let $k : \mathcal { X } \times \mathcal { X } \to \mathbb { R }$ be a positive definite kernel. The RKHS induced by k, denoted by $\mathcal { H } _ { k }$ , is a Hilbert space of functions $f : \mathcal { X } $ R such that, for every $x \in \mathcal { X }$ , the evaluation functional $f \mapsto f ( x )$ is continuous. Equivalently, there exists a function $\boldsymbol { k } ( \cdot , x ) \in \mathcal { H } _ { k }$ satisfying the reproducing property

$$
f ( x ) = \langle f , k ( \cdot , x ) \rangle _ { \mathcal { H } _ { k } } , \qquad \forall f \in \mathcal { H } _ { k } .
$$

By Cauchy’s inequality, $| f ( x ) | \leq \| f \| _ { \mathcal { H } _ { k } } \sqrt { k ( x , x ) }$ . When the kernel is normalized so that $k ( x , x ) \leq 1$ , this implies $| f ( x ) | \leq \| f \| _ { \mathcal { H } _ { k } }$ . The RKHS norm quantifies the complexity of the latent response surface. A smaller RKHS norm means that the function is smoother or more compatible with the kernel. Throughout this paper, we make the following assumption on function $f .$

Assumption 3.2. The objective function $f \in \mathcal { H } _ { k _ { \theta } }$ with $\| f \| _ { \mathcal { H } _ { k _ { \theta } } } \le B _ { f }$ . The kernel of the RKHS is is continuous on $\mathcal { X } \times \mathcal { X }$ . In addition, the kernel is normalized and there is a fixed constant $k _ { \operatorname* { m i n } { } } > 0$ such that $k _ { \mathrm { m i n } } \le k _ { \theta } ( x , x ) \le 1$ $f o r \ a l l \ x \in { \mathcal { X } }$

For SE kernels, the associated RKHS contains very smooth functions; for Matérn kernels, the smoothness is controlled by the parameter ν. In both cases, the RKHS assumption provides the regularity condition needed to connect GP posterior uncertainty with finite-budget regret guarantees.

## 4 Generic EI Framework and Finite-Sample SK Confidence

We begin with a generic EI procedure in which the incumbent, the replication size, and the final recommendation remain unspecified within broad rule classes. We first state the procedure and identify where these three components enter in Section 4.1. We then establish a finite-sample confidence bound for the SK predictor used by the procedure in Section 4.2. The next section uses this confidence bound to derive a simple-regret certificate for this generic procedure.

Throughout this section, $\mu _ { n }$ and $\sigma _ { n }$ denote the SK posterior mean and standard deviation after n simulation batches. The kernel hyperparameter is assumed fixed and known, and θ is suppressed from the notation. This is a standard convention in theoretical analyses of Bayesian and simulation optimization (Bull 2011, Wang et al. 2025). In our experiments, the hyperparameters are updated using the sequentially collected data.

## 4.1 A Generic EI Procedure

In this paper, we consider a simulation optimization problem with a total raw simulation budget T. The algorithm sequentially selects decision points and allocates $b _ { T }$ simulation replications to each point. For expositional simplicity, we use a common replication size $b _ { T }$ to every selected point. Nonetheless, the framework and its order-wise guarantees require only that the replication numbers at diferent points have the same asymptotic order in the total budget T, rather than being exactly equal. After the budget is exhausted, it returns a final recommendation ${ \hat { x } } _ { T }$ With this setting, we use $\begin{array} { r } { N _ { T } = \left\lfloor \frac { T } { b _ { T } } \right\rfloor } \end{array}$ to represent the maximum number of distinct points with budget T. We further require that the budget must cover at least two distinct designs with at least two replications each. Thus $2 \leq b _ { T } \leq \lfloor T / 2 \rfloor$ , which gives $2 \leq N _ { T } \leq \lfloor T / 2 \rfloor$ . In this generic algorithm, any value of $b _ { T }$ within the range above is permitted. We next detail the general families of the incumbent value and the recommendation in this procedure.

## 4.1.1 Global linear-quantile incumbents

In deterministic EI, the incumbent is usually the best observed value. This choice is not suitable in stochastic sim ulation because observations contain random errors. We instead analyze a family of posterior-quantile incumbents. Let ${ \bf c } = ( c _ { 0 } , \ldots , c _ { N _ { T } - 1 } ) \in \mathbb { R } ^ { N _ { T } }$ be a deterministic coeficient sequence. It may depend on $T , \delta ,$ , and the known model constants, but not on the observed simulation outputs. After n batches, define

$$
Q _ { n } ^ { ( c _ { n } ) } ( x ) : = \mu _ { n } ( x ) - c _ { n } \sigma _ { n } ( x ) , g _ { n + 1 } ^ { \mathrm { c } } : = \arg \operatorname* { m i n } _ { x \in \mathcal { X } } Q _ { n } ^ { ( c _ { n } ) } ( x ) , \xi _ { n } ^ { \mathrm { c } } : = Q _ { n } ^ { ( c _ { n } ) } ( g _ { n + 1 } ^ { \mathrm { c } } ) = \operatorname* { m i n } _ { x \in \mathcal { X } } Q _ { n } ^ { ( c _ { n } ) } ( x ) .
$$

Equivalently, $Q _ { n } ^ { ( c _ { n } ) } ( x )$ is the posterior quantile at level $\Phi ( - \boldsymbol { c } _ { n } )$ under the SK model. The incumbent is chosen as $\xi _ { n } ^ { \mathbf { c } }$ and the point selected by EI is $\boldsymbol { x } _ { n + 1 } ^ { \mathbf { c } } \in \arg \operatorname* { m a x } _ { \boldsymbol { x } \in \mathcal { X } } \mathrm { E I } _ { n } ( \boldsymbol { x } ; \boldsymbol { \xi } _ { n } ^ { \mathbf { c } } )$ . Continuity of the kernel and compactness of X ensure that the minima and maxima in these definitions are attained. Ties are broken arbitrarily. Here, $c _ { n } > 0 , c _ { n } = 0$ and $c _ { n } < 0$ correspond respectively to lower-quantile, posterior-mean, and upper-quantile incumbents, representing optimistic, neutral, and uncertainty-penalizing treatments of posterior uncertainty when deciding the incumbent.

Many incumbent rules in the literature can be written in this general form or are closely related to it. Setting $c _ { n } = 0$ gives the widely used global best-posterior-mean incumbent (Wang and de Freitas 2014). Some methods instead optimize the quantile criterion only over previously sampled points. For example, SKO and EQI optimize upper quantile functions with negative $c _ { n }$ coeficients in (4.1.1) over the sampled points. Whether the incumbent is optimized over the full domain or the set of sampled points often reflects computational considerations. The former requires continuous optimization, whereas the latter requires only a finite comparison. A similar distinction arises in a related GP based knowledge-gradient (KG) methods. Although their sampling principle difers from EI, they also depend on a current best value. Due to practical considerations such as computational resources, implementations adopting either the global best posterior mean or the best sampled posterior can both be found in the literature (Wu and Frazier 2016, Jalali et al. 2017).

We optimize the incumbent over the full domain for two reasons. First, for commonly used diferentiable kernels, the continuous problem can be handled in practice using multistart gradient-based solvers such as L-BFGS-B, whose computational cost is often modest relative to expensive simulation runs. Second, full-domain optimization is aligned with our objective of tightening the simple-regret certificate. With a larger feasible set, it may lead to better convergence behavior. In addition, in our theoretical analysis, it permits direct comparison with the unknown optimizer and avoids an additional sampled-set approximation term.

## 4.1.2 Final recommendation

After the simulation budget is exhausted, the algorithm must return one concrete decision point. To enhance the decision maker’s trust in the algorithm’s final recommendation, we consider the following three aspects. First, we select a previously evaluated point as the final recommendation as with the simulation literature (Jalali et al. 2017). This allows the decision maker to observe the efectiveness of the returned strategy through historical simulation data to enhance their confidence in the recommendation. Second, unlike many classical methods that convert cumulative regret to simple regret by randomly returning a previously sampled point, we select from these points based on their performance, which better justifies our choice. Third, as mentioned earlier, when recommending the current best to the decision maker, we account for the impact of residual uncertainty and deliberately avoid solutions with excessive uncertainty.

Let $\mathcal { X } _ { T } ^ { \mathbf { c } } = \{ x _ { 1 } ^ { \mathbf { c } } , \ldots , x _ { N _ { T } } ^ { \mathbf { c } } \}$ be the sampled set produced by the coeficient sequence c. The returned design is

$$
\hat { x } _ { T } ^ { \mathbf { c } } \in \arg \operatorname* { m i n } _ { z \in { \mathcal { X } _ { T } ^ { \mathbf { c } } } } \left\{ \mu _ { N _ { T } } ( z ) + d _ { T } \sigma _ { N _ { T } } ( z ) \right\} ,\tag{3}
$$

where $d _ { T } \in \mathbb { R }$ is a recommendation coeficient. Similar to the general family of incumbents, the choices $d _ { T } = 0 ,$ $d _ { T } > 0 .$ , and $d _ { T } < 0$ correspond to the smallest sampled posterior mean, an upper posterior quantile, and a lower posterior quantile, respectively. We leave its value unspecified here and select it from the simple-regret certificate in Section 5.

The generic EI procedure is summarized in Algorithm 1.

Algorithm 1 Generic Expected Improvement with Stochastic Kriging   
Require: $T , b _ { T } , v _ { 0 } , \mathbf { c } , d _ { T }$   
1: Set $N _ { T } = \lfloor T / b _ { T } \rfloor$ and $\mathcal { D } _ { 0 } = \mathcal { D } ;$ compute $\kappa _ { T }$ from (4).   
2: for $n = 0 , 1 , \dots , N _ { T } - 1$ do   
3: Compute the SK posterior $( \mu _ { n } , \sigma _ { n } )$ from $\mathcal { D } _ { n }$   
4: Compute the incumbent value: $\begin{array} { r } { \xi _ { n } ^ { \mathbf { c } } = \operatorname* { m i n } _ { x \in \mathcal { X } } \{ \mu _ { n } ( x ) - c _ { n } \sigma _ { n } ( x ) \} } \end{array}$   
5: Select the next decision point: $\boldsymbol { x } _ { n + 1 } ^ { \mathbf { c } } \in \arg \operatorname* { m a x } _ { \boldsymbol { x } \in \mathcal { X } } \mathrm { E I } _ { n } ( \boldsymbol { x } ; \boldsymbol { \xi } _ { n } ^ { \mathbf { c } } )$   
6: Run $b _ { T }$ independent replications at $x _ { n + 1 } ^ { \mathbf { c } }$ and compute $\bar { Y } _ { n + 1 }$ and $S _ { n + 1 } ^ { 2 }$   
7: Set $\widetilde v _ { n + 1 } = \operatorname* { m a x } \{ v _ { 0 } , S _ { n + 1 } ^ { 2 } \}$ and $r _ { n + 1 } ^ { 2 } = \widetilde { v } _ { n + 1 } / b _ { T }$   
8: Update $\mathcal { D } _ { n + 1 } = \mathcal { D } _ { n } \cup \{ ( x _ { n + 1 } ^ { \mathbf { c } } , \bar { Y } _ { n + 1 } , r _ { n + 1 } ^ { 2 } ) \}$   
9: Return $\hat { x } _ { T } ^ { \mathbf { c } }$ according to (3)

## 4.2 Confidence Bound for SK Prediction

In simulation settings with heteroscedastic normal noise, the variance function $v ( x )$ is unknown and may vary over the design space. We make the following assumption on the noise levels:

Assumption 4.1. There exist constants $v _ { \star }$ and $V _ { \star }$ such that

$$
0 < v _ { \star } \leq v ( x ) \leq V _ { \star } < \infty , \qquad x \in \mathcal { X } .
$$

We highlight that this assumption only requres the noise variances to be uniformly bounded. Neither v<sub>⋆</sub> nor $V _ { \star }$ is assumed to be known to the algorithm.

At each selected decision point, the algorithm collects a batch of $b _ { T }$ independent replications. At the location selected for batch $q ,$ we can compute the sample mean $\bar { Y } _ { q }$ and sample variance $S _ { q } ^ { 2 }$ from these replications. Let $\mathcal { H } _ { q - 1 }$ contain all information available before this batch, including its selected location $x _ { q } .$ . Conditional on $\mathcal { H } _ { q - 1 }$ , under Gaussian simulation noise,

$$
\bar { Y } _ { q } \sim N \left( f ( x _ { q } ) , \frac { v ( x _ { q } ) } { b _ { T } } \right) , \qquad \frac { ( b _ { T } - 1 ) S _ { q } ^ { 2 } } { v ( x _ { q } ) } \sim \chi _ { b _ { T } - 1 } ^ { 2 } ,
$$

and $\bar { Y } _ { q }$ and $S _ { q } ^ { 2 }$ are conditionally independent.

In conventional SK implementations, the unknown intrinsic variance $ { \boldsymbol { v } } (  { \boldsymbol { { x } } } _ { q } )$ is estimated by $S _ { q } ^ { 2 }$ , so $S _ { q } ^ { 2 } / b _ { T }$ is used as a plug-in estimate of the noise variance of the sample mean $\bar { Y } _ { q }$ (Ankenman et al. 2010). We use a regularized estimate $\widetilde { v } _ { q } : = \operatorname* { m a x } \{ v _ { 0 } , S _ { q } ^ { 2 } \}$ . The plug-in estimate of the noise variance of $\bar { Y } _ { q }$ is $\begin{array} { r } { r _ { q } ^ { 2 } = \frac { \widetilde { v } _ { q } } { b _ { T } } } \end{array}$ . Here, $v _ { 0 }$ is a positive variance floor independently of the unknown noise bounds. It regularizes the plug-in noise matrix and need not be a lower bound on $v ( x )$ . Practically, it plays a role similar to a nugget or jitter term to stabilize covariance-matrix inversion when the estimated noise variances are very small.

The confidence analysis must account for possible underestimation of $ { \boldsymbol { v } } (  { \boldsymbol { { x } } } _ { q } )$ with sample variances. Let $\nu _ { T } : =$ $b _ { T } - 1$ and $\begin{array} { r } { \alpha _ { T } : = \frac { \delta } { 4 N _ { T } } } \end{array}$ . For $p \in ( 0 , 1 )$ ), let $\chi _ { \nu , p } ^ { 2 }$ denote the p-quantile of a chi-squared random variable with ν degrees of freedom. Define

$$
\kappa _ { T } : = \operatorname* { m a x } \left\{ 1 , \frac { \nu _ { T } } { \chi _ { \nu _ { T } , \alpha _ { T } } ^ { 2 } } \right\} , \qquad h _ { T } : = \operatorname* { m a x } \left\{ 1 , \frac { \chi _ { \nu _ { T } , 1 - \alpha _ { T } } ^ { 2 } } { \nu _ { T } } \right\} .\tag{4}
$$

These two factors are used to control potential under or over estimate of the noise variance. Specifically, $\kappa _ { T }$ accounts for sample-variance estimates that are too small and could otherwise produce overly narrow confidence bands, while $h _ { T }$ controls estimates that are too large and can inflate posterior uncertainty. In addition, define $M : = \operatorname* { m a x } \{ v _ { 0 } , V _ { \star } \}$ , and $W _ { T } : = h _ { T } M$ , which are used in the performance analysis.

Lemma 4.2. With probability at least $1 - \delta / 2$ , the following inequalities hold for every $q \leq N _ { T }$ :

$$
\frac { v ( x _ { q } ) } { \kappa _ { T } } \leq S _ { q } ^ { 2 } \leq h _ { T } v ( x _ { q } ) , \qquad v ( x _ { q } ) \leq \kappa _ { T } \widetilde v _ { q } .
$$

In addition, under Assumption $\begin{array} { r } { 4 \cdot 1 , \ \frac { v _ { 0 } } { b _ { T } } \leq r _ { q } ^ { 2 } \leq \frac { W _ { T } } { b _ { T } } } \end{array}$

Although this lemma is valid for every admissible $b _ { T }$ , the sizes of $\kappa _ { T }$ and $h _ { T }$ determine how much the under and over estimation of the noise variance influences the SK confidence bound. We therefore first characterize how κ and $h _ { T }$ depends on the replication level.

Lemma 4.3. Fix $\delta \in ( 0 , 1 )$ , and let $T \to \infty$ . For each $T ,$ let b<sub>T</sub> be an integer satisfying $\begin{array} { r } { 2 \leq b _ { T } \leq \left\lfloor \frac { T } { 2 } \right\rfloor } \end{array}$ . Then the following statements hold:

1. If $\dot { { b } } _ { T } = o ( \log T )$ , then $\kappa _ { T } , h _ { T } \longrightarrow \infty$

2. If $b _ { T } = \Theta ( \log T )$ , then $\kappa _ { T } , h _ { T } = O ( 1 )$

3. If $\frac { b _ { T } } { \log T } \longrightarrow \infty$ , then $\kappa _ { T } , h _ { T } \longrightarrow 1$

In particular, $i f b _ { T } = b \ge 2$ is fixed, then $\kappa _ { T } = \Theta \left( T ^ { 2 / ( b - 1 ) } \right) \ a n d \ h _ { T } = \Theta ( \log T )$

Lemma 4.3 isolates the efect of replication on the variance-under / overestimation correction. A sublogarithmic replication level makes these corrections diverge, whereas logarithmic replication keeps them bounded. A superlogarithmic level can make the corrections approach one, but it also leaves fewer design evaluations for spatial exploration. We next show how these factors enters the SK confidence width. Section 6.3 then combines this efect with the number of design evaluations and the information-gain terms in the simple-regret bound.

With n batch observations, the data used by the surrogate model are ${ \mathcal { D } } _ { n } = \{ ( x _ { q } , { \bar { Y } } _ { q } , r _ { q } ^ { 2 } ) : q = 1 , \ldots , n \}$ , where $r _ { q } ^ { 2 } = \widetilde { v } _ { q } / b _ { T }$ . The SK posterior mean and variance are therefore the quantities in equations (1) and (2), with $\overset { \underset { \mathrm { ~ \scriptsize ~ 1 ~ } } { } } { R _ { n } } = \mathrm { d i a g } ( r _ { 1 } ^ { 2 } , \ldots , r _ { n } ^ { 2 } )$ . Thus, the surrogate uses the standard SK covariance formula with clipped plug-in estimates of the intrinsic variances. The correction $\kappa _ { T }$ is incorporated into the confidence coeficient for the SK predictor rather than into the plug-in noise matrix. We derive the following confidence bound for the SK predictor.

Theorem 4.4. Under Assumptions 3.1, 3.2, and $4 . 1 ,$ for $n = 1 , \dots , N _ { T }$ , define

$$
\gamma _ { n } ^ { \mathrm { S K } } = \operatorname* { s u p } _ { 1 \leq m \leq n } \operatorname* { s u p } _ { a _ { 1 } , \ldots , a _ { m } \in \mathcal { X } } \frac { 1 } { 2 } \log \operatorname* { d e t } \left( I _ { m } + \frac { b _ { T } } { v _ { 0 } } K _ { A } \right) , \qquad \gamma _ { 0 } ^ { \mathrm { S K } } = 0 ,\tag{5}
$$

where $K _ { A } = \left( k _ { \theta } ( a _ { i } , a _ { j } ) \right) _ { i , j = 1 } ^ { m }$ . Let

$$
\Lambda _ { n } = B _ { f } + \sqrt { 2 \kappa _ { T } \left( \gamma _ { n } ^ { \mathrm { S K } } + \log \frac { 2 } { \delta } \right) } , \qquad n = 0 , \dots , N _ { T } .\tag{6}
$$

Then $( \Lambda _ { n } ) _ { n = 0 } ^ { N _ { T } }$ is nondecreasing. With probability at least $1 - \delta$ , all variance inequalities in Lemma $4 . 2$ and the following confidence inequalities hold jointly:

$$
| f ( \boldsymbol { x } ) - \mu _ { n } ( \boldsymbol { x } ) | \leq \Lambda _ { n } \sigma _ { n } ( \boldsymbol { x } ) , \qquad \forall \boldsymbol { x } \in \mathcal { X } , \quad \forall n = 0 , \ldots , N _ { T } .
$$

The proof of this lemma uses conditional independence of Gaussian sample mean and variance, and a timeuniform self-normalized inequality (Kirschner and Krause 2018). This theorem provides the statistical foundation for the subsequent algorithm design. The uncertainty from estimating the unknown variances enters through $\kappa _ { T }$ , accounting for variance underestimation in the confidence coeficient $\Lambda _ { n } .$ . The upper-tail factor $h _ { T }$ controls overestimation of the noise variances and is used in the subsequent simple-regret analysis to bound posterior uncertainty. When the variances are known, we will have $\kappa _ { T } = 1$ , and the result reduces to the standard timeuniform RKHS confidence bound, up to confidence-allocation constants. If the variance is constant, the bound in Theorem 4.4 gives its usual homoscedastic form (Kirschner and Krause 2018).

## 5 A Unified Simple-Regret Certificate

Algorithm 1 leaves the incumbent sequence, replication size, and final recommendation unspecified within their respective rule classes. We now use the finite-sample confidence guarantee in Theorem 4.4 to derive a common highprobability simple-regret certificate before selecting any of these three components. For any $b _ { T } , \mathbf { c } = \left( c _ { 0 } , \dots , c _ { N _ { T } - 1 } \right)$ and $d _ { T }$ , define

$$
\begin{array} { c } { { \mathcal { E } _ { T } : = \left\{ | f ( x ) - \mu _ { n } ( x ) | \leq \Lambda _ { n } \sigma _ { n } ( x ) , \quad \forall x \in \mathcal { X } , \quad n = 0 , \ldots , N _ { T } \right\} , } } \\ { { \mathcal { V } _ { T } : = \left\{ \frac { v ( x _ { q } ) } { \kappa _ { T } } \leq S _ { q } ^ { 2 } \leq h _ { T } v ( x _ { q } ) , \quad q = 1 , \ldots , N _ { T } \right\} . } } \end{array}
$$

The event $\nu _ { T }$ implies $v ( x _ { q } ) / \kappa _ { T } \leq \widetilde { v } _ { q } \leq W _ { T }$ for every q. By Theorem 4.4, $\mathcal { G } _ { T } : = \mathcal { E } _ { T } \cap \mathcal { V } _ { T }$ has probability at least $1 - \delta$ . All subsequent deterministic inequalities are stated on $\mathcal { G } _ { T }$

The derivation proceeds in two stages. We first compare the posterior uncertainties at the incumbent point and EI-selected points and invert the EI function to obtain a regret certificate for each selected point in Section 5.1. We then use uniform variance bounds and information gain to identify a well-certified sampled point and compare it with the terminal recommendation. Combining these two stages yields the final simple-regret certificate in Section 5.2.

## 5.1 Regret certificate for a sampled point

For any coeficient sequence c, as the terminal recommendation is restricted to sampled points, we first derive a regret certificate for each point selected by EI. For each $n < N _ { T }$ , define $\begin{array} { r } { \rho _ { n } ^ { \mathbf { c } } : = \frac { \sigma _ { n } ( x _ { n + 1 } ^ { \mathbf { c } } ) } { \sigma _ { n } ( g _ { n + 1 } ^ { \mathbf { c } } ) } } \end{array}$ . The denominator is positive because every point has a positive noise proxy. The following lemma shows that the EI-selected point cannot have less posterior uncertainty than the incumbent point.

Lemma 5.1. For every $n = 0 , \ldots , N _ { T } - 1 , \rho _ { n } ^ { \mathbf { c } } \geq 1 . \ E q u i v a l e n t l y , \ \sigma _ { n } ( g _ { n + 1 } ^ { \mathbf { c } } ) \leq \sigma _ { n } ( x _ { n + 1 } ^ { \mathbf { c } } ) .$

This comparison follows from the global minimization of the linear quantile and therefore holds for every real coeficient $c _ { n }$ in the general incumbent family. To convert this comparison into a regret certificate, we next invert the function τ appearing in EI. It is a prerequisite for bounding on the gap between each sampled point and the incumbent.

Lemma 5.2. For $c \in \mathbb { R }$ and $r \geq 1$ , let $q _ { c } ( r )$ be the unique real number satisfying $\begin{array} { r } { \tau ( - q _ { c } ( r ) ) = \frac { \tau ( - c ) } { r } } \end{array}$ . Then $q _ { c } ( r ) \geq c$ and

$$
q _ { c } ( r ) \leq \left\{ \begin{array} { l l } { \sqrt { c ^ { 2 } + 2 \log r } , } & { c \geq 0 , } \\ { \sqrt { 2 \log r } , } & { c < 0 . } \end{array} \right.
$$

For each step $n ,$ write $s _ { n } ^ { g } : = \sigma _ { n } ( g _ { n + 1 } ^ { \mathbf { c } } )$ and $s _ { n } ^ { x } : = \sigma _ { n } ( x _ { n + 1 } ^ { \mathbf { c } } )$ . Lemma 5.1 gives $\rho _ { n } ^ { \mathbf { c } } = s _ { n } ^ { x } / s _ { n } ^ { g } \ge 1$ , allowing Lemma 5.2 to control the gap between the selected point and the incumbent. For $\lambda > 0$ , let $z _ { \lambda }$ be the unique solution of $\begin{array} { r } { \frac { \phi ( z _ { \lambda } ) } { \Phi ( z _ { \lambda } ) } = \lambda } \end{array}$ , and set $c _ { \lambda } ^ { \circ } : = - z _ { \lambda }$ . Define

$$
A ( \lambda , c ) : = \left\{ \begin{array} { l l } { 0 , } & { c \geq \lambda , } \\ { \lambda - c , } & { c _ { \lambda } ^ { 0 } < c < \lambda , } \\ { \displaystyle \frac { \tau ( - c ) } { \Phi ( z _ { \lambda } ) } , } & { c \leq c _ { \lambda } ^ { 0 } . } \end{array} \right.
$$

Based on this definition, the following lemma provides regret certificate for a sampled EI point.

Lemma 5.3. For every $n = 0 , \dots , N _ { T } - 1$ ，

$$
\begin{array} { r } { f ( x _ { n + 1 } ^ { \mathbf { c } } ) - f ^ { \star } \leq \left\{ \Lambda _ { n } + q _ { c _ { n } } ( \rho _ { n } ^ { \mathbf { c } } ) \right\} \sigma _ { n } ( x _ { n + 1 } ^ { \mathbf { c } } ) + ( \xi _ { n } ^ { \mathbf { c } } - f ^ { \star } ) _ { + } , } \end{array}\tag{7}
$$

$$
( \xi _ { n } ^ { \mathbf { c } } - f ^ { \star } ) _ { + } \leq ( \Lambda _ { n } - c _ { n } ) _ { + } \sigma _ { n } ( x ^ { \star } ) ,\tag{8}
$$

$$
\begin{array} { r } { ( \xi _ { n } ^ { \mathbf { c } } - f ^ { \star } ) _ { + } \leq A ( \Lambda _ { n } , c _ { n } ) \sigma _ { n } ( x _ { n + 1 } ^ { \mathbf { c } } ) . } \end{array}\tag{9}
$$

Consequently,

$$
f ( x _ { n + 1 } ^ { \mathbf { c } } ) - f ^ { \star } \leq \lbrace \Lambda _ { n } + q _ { c _ { n } } ( \rho _ { n } ^ { \mathbf { c } } ) + A ( \Lambda _ { n } , c _ { n } ) \rbrace \sigma _ { n } ( x _ { n + 1 } ^ { \mathbf { c } } ) .\tag{10}
$$

A general incumbent can lie above $f ^ { \star }$ even on the SK predictor confidence event. This produces the term $( \xi _ { n } ^ { \mathbf { c } } - f ^ { \star } ) _ { + } \mathrm { ~ i n ~ } ( 7 )$ . Equation (8) shows the direct efect of the incumbent coeficient, but it depends on the posterior variance at the unknown optimizer. The information-gain argument used in GP-based optimization literature instead controls posterior variances at sampled points. Therefore, (9) transfers this uncertainty term to the EIselected point at the cost of the factor $A ( \Lambda _ { n } , c _ { n } )$ . The proof in Appendix F further indicates that no smaller coeficient than $A ( \Lambda _ { n } , c _ { n } )$ exists under the definition of the incumbent and the SK predictor confidence bound in Theorem 4.4. It remains to control the posterior standard deviations at the selected points and connect a well-certified sampled point to the terminal recommendation.

## 5.2 From sampled points to final simple regret

The pointwise certificate in (10) still depends on the data-dependent variance ratio $\rho _ { n } ^ { \mathbf { c } }$ . We first replace this ratio with a uniform bound and then control the accumulated posterior uncertainties at the selected points.

Lemma 5.4. For every $x \in \mathcal { X }$ and $n = 0 , \ldots , N _ { T }$

$$
\frac { k _ { \mathrm { m i n } } } { 1 + n b _ { T } / v _ { 0 } } \leq \sigma _ { n } ^ { 2 } ( x ) \leq 1 , 1 \leq \rho _ { n } ^ { \mathbf { c } } \leq \sqrt { \frac { 1 + n b _ { T } / v _ { 0 } } { k _ { \mathrm { m i n } } } } .
$$

Define

$$
\ell _ { n } : = \log \left( \frac { 1 + n b _ { T } / v _ { 0 } } { k _ { \operatorname* { m i n } } } \right) \mathrm { ~ a n d ~ } \bar { q } _ { n } ( c ) : = \left\{ \begin{array} { l l } { \sqrt { c ^ { 2 } + \ell _ { n } } , } & { c \geq 0 , } \\ { \sqrt { \ell _ { n } } , } & { c < 0 . } \end{array} \right.
$$

Lemmas 5.2 and 5.4 imply $q _ { c _ { n } } ( \rho _ { n } ^ { \mathbf { c } } ) \leq \bar { q } _ { n } ( c _ { n } )$ , removing the data-dependent ratio from the pointwise certificate. It remains to control the posterior standard deviations at the selected points, which is achieved through information gain.

Lemma 5.5. For the points selected under c,

$$
\sum _ { n = 0 } ^ { N _ { T } - 1 } \sigma _ { n } ^ { 2 } ( x _ { n + 1 } ^ { \mathbf { c } } ) \leq \frac { 2 \gamma _ { N _ { T } } ^ { \mathrm { S K } } } { \log ( 1 + b _ { T } / W _ { T } ) } .
$$

As in standard BO analyses, Lemma 5.5 controls the cumulative posterior uncertainty at the points when they are selected. Together with the pointwise certificate, it ensures that at least one sampled point has both a small posterior standard deviation and a small regret certificate. To collect the coeficient-dependent terms, define

$$
D _ { T } ( \mathbf { c } ) : = \operatorname* { m a x } _ { 0 \leq n < N _ { T } } \left\{ \Lambda _ { n } + \bar { q } _ { n } ( c _ { n } ) + A ( \Lambda _ { n } , c _ { n } ) \right\} .
$$

We also define the uncertainty scales used to compare this well-certified sampled point with the terminal recommendation:

$$
\eta _ { T } : = \sqrt { \frac { 2 \gamma _ { N r } ^ { \mathrm { S K } } } { N _ { T } \log ( 1 + b _ { T } / W _ { T } ) } } , \omega _ { T } : = \sqrt { \frac { W _ { T } } { b _ { T } + W _ { T } } } , \mathrm { a n d } \zeta _ { T } : = \operatorname* { m i n } \{ \eta _ { T } , \omega _ { T } \} .
$$

Lemma 5.3 and Lemma 5.5 together guarantee a sampled point whose simple regret is at most $D _ { T } ( \mathbf { c } ) \eta _ { T }$ . After all observations have been incorporated, this point has posterior standard deviation at most $\zeta _ { T } = \mathrm { m i n } \{ \eta _ { T } , \omega _ { T } \}$ , while every sampled point, including the terminal recommendation, has posterior standard deviation at most ω . Because the terminal recommendation minimizes its linear posterior quantile over the sampled points, it can be compared with this well-certified point. Applying the terminal SK predictor confidence bound then gives the following result.

Theorem 5.6. Let $b _ { T }$ be any integer satisfying $\begin{array} { r } { 2 \leq b _ { T } \leq \left| \frac { T } { 2 } \right| } \end{array}$ . Under Assumptions $\begin{array} { r } { \boldsymbol { 1 } \sim \mathcal { 3 } , } \end{array}$ for each fixed deterministic incumbent coeficient sequence c and $d _ { T }$ , with probability at least $1 - \delta$

$$
f \big ( \hat { x } _ { T } ^ { \mathbf { c } , d _ { T } } \big ) - f ^ { \star } \le D _ { T } ( \mathbf { c } ) \eta _ { T } + \big ( \Lambda _ { N _ { T } } + d _ { T } \big ) _ { + } \zeta _ { T } + \big ( \Lambda _ { N _ { T } } - d _ { T } \big ) _ { + } \omega _ { T } .\tag{11}
$$

Theorem 5.6 links the three design components through one finite-budget simple-regret certificate. The first term is generated by the search process and contains the incumbent-dependent factor $D _ { T } ( \mathbf { c } )$ . The last two terms measure the transfer from a well-certified sampled point to the terminal recommendation and contain the terminal coeficient $d _ { T }$ . The replication size afects all three terms through $N _ { T } , \kappa _ { T } , \Lambda _ { n } , \gamma _ { N _ { T } } ^ { \mathrm { S K } }$ , and the uncertainty scales η<sub>T</sub>, $\zeta _ { T }$ , and $\omega _ { T }$ . Because none of the three components has yet been calibrated, the result applies to the generic EI procedure. The next section selects their parameters to tighten this certificate.

## 6 Bound-Guided Design Choices and Regret Rates

This section uses Theorem 5.6 as a common criterion for selecting the three design components. The first two subsections calibrate the terminal and incumbent coeficients from their respective parts of the certificate. Unlike these two coeficients, the replication size depends on the covariance kernel. After fixing the first two components, we therefore select the replication size by minimizing a tractable upper bound under mild conditions for kernels. The final subsection derives the resulting rates for specific kernel classes.

## 6.1 Choice of the terminal coeficient

For fixed c and $b _ { T }$ , define $G _ { T } ( d ) : = ( \Lambda _ { N _ { T } } + d ) _ { + } \zeta _ { T } + ( \Lambda _ { N _ { T } } - d ) _ { + } \omega _ { T }$ . Because $\zeta _ { T } \leq \omega _ { T }$

$$
\begin{array} { r } { G _ { T } ( d ) - G _ { T } ( \Lambda _ { N _ { T } } ) = \left\{ \begin{array} { l l } { ( \Lambda _ { N _ { T } } - d ) \omega _ { T } - 2 \Lambda _ { N _ { T } } \zeta _ { T } , } & { d \leq - \Lambda _ { N _ { T } } , } \\ { ( \Lambda _ { N _ { T } } - d ) ( \omega _ { T } - \zeta _ { T } ) , } & { - \Lambda _ { N _ { T } } \leq d \leq \Lambda _ { N _ { T } } , } \\ { ( d - \Lambda _ { N _ { T } } ) \zeta _ { T } , } & { d \geq \Lambda _ { N _ { T } } . } \end{array} \right. } \end{array}
$$

All three expressions are nonnegative. In the first case, $\Lambda _ { N _ { T } } - d \geq 2 \Lambda _ { N _ { T } }$ , so the diference is at least $2 \Lambda _ { N _ { T } } ( \omega _ { T } - \zeta _ { T } )$ Therefore, $d _ { T } = \Lambda _ { N _ { T } }$ is a global bound-minimizing choice for the displayed uniform certificate in (11). It yields the sampled design with the smallest posterior upper confidence value. Henceforth, we select $d _ { T } = \Lambda _ { N _ { T } }$ to return the final recommendation written as $\hat { x } _ { T } ^ { \mathbf { c } } : = \hat { x } _ { T } ^ { \mathbf { c } , \Lambda _ { N _ { T } } }$

## 6.2 Choice of the incumbent coeficient

Having calibrated the terminal coeficient, we next select the incumbent coeficient sequence c. Under $d _ { T } = \Lambda _ { N _ { T } }$ and the feasible range $2 \leq b _ { T } \leq \lfloor T / 2 \rfloor$ , the simple regret has the form:

$$
f ( \hat { x } _ { T } ^ { \mathbf { c } } ) - f ^ { \star } \leq 2 \left\{ 2 \Lambda _ { N _ { T } } + D _ { T } ( \mathbf { c } ) \right\} \sqrt { \frac { b _ { T } \gamma _ { N _ { T } } ^ { \mathrm { S K } } } { T \log ( 1 + b _ { T } / W _ { T } ) } } .
$$

With this form, we observe that to minimize the bound for simple regret, we need to select c to minimize $D _ { T } ( \mathbf { c } )$ This result is summarized in the following lemma.

Lemma 6.1. For every $n = 0 , \dots , N _ { T } - 1$ and $c \in \mathbb { R }$ , we have $\bar { q } _ { n } ( c ) + A ( \Lambda _ { n } , c ) \geq \bar { q } _ { n } ( \Lambda _ { n } )$

Recall that $D _ { T } ( { \bf c } ) = \mathrm { m a x } _ { 0 \leq n < N _ { T } } \left\{ \Lambda _ { n } + { \bar { q } } _ { n } ( c _ { n } ) + A ( \Lambda _ { n } , c _ { n } ) \right\}$ and $A ( \Lambda _ { n } , \Lambda _ { n } ) = 0$ . Consequently, the choice $c _ { n } \ = \Lambda _ { n }$ minimizes every term $\Lambda _ { n } + \bar { q } _ { n } ( c _ { n } ) + A ( \Lambda _ { n } , c _ { n } )$ in the definition of $D _ { T } ( \mathbf { c } )$ . Hence, the sequence $\mathbf { c } =$ $( \Lambda _ { 0 } , \dots , \Lambda _ { N _ { T } - 1 } )$ minimizes $D _ { T } ( \mathbf { c } )$ over $\mathbb { R } ^ { N _ { T } }$ . For this choice, we omit the superscript c from the selected points, sampled set, and final recommendation.

Corollary 6.2. Set $c _ { n } = \Lambda _ { n } ~ f o r ~ n = 0 , \ldots , N _ { T } - 1$ and $d _ { T } = \Lambda _ { N _ { T } }$ . Write $\hat { x } _ { T } : = \hat { x } _ { T } ^ { \mathbf { c } , \Lambda _ { N _ { T } } }$ . Under the assumptions of Theorem 5.6, with probability at least $1 - \delta$

$$
f ( \hat { x } _ { T } ) - f ^ { \star } \le 8 \sqrt { \frac { \left[ \Lambda _ { N _ { T } } ^ { 2 } + \log \left( \frac { 1 + T / v _ { 0 } } { k _ { \operatorname* { m i n } } } \right) \right] b _ { T } \gamma _ { N _ { T } } ^ { \mathrm { S K } } } { T \log ( 1 + b _ { T } / W _ { T } ) } } .
$$

Thus, the proposed algorithm uses $c _ { n } = \Lambda _ { n }$ during search and $d _ { T } = \Lambda _ { N _ { T } }$ at termination. We highlight that the calibrated incumbent and terminal rules treat uncertainty in opposite ways. During the search, EI uses the best lower posterior quantile as its incumbent, whereas the terminal rule returns the sampled point with the smallest upper posterior quantile. This contrast reflects the diferent roles of intermediate simulation experiments and the final implemented decision. As intermediate evaluations are exploratory and are not themselves deployed in the real system, the incumbent can treat remaining uncertainty optimistically and keep potentially good regions under consideration. The final recommendation, by contrast, is intended for real-world implementation and therefore treats remaining uncertainty conservatively.

## 6.3 Kernel-Dependent Choice of the Replication Size

Unlike the incumbent and terminal coeficients, the replication size depends on the covariance kernel used by the SK predictor. Increasing the replication size improves the estimation of the local noise variances and reduces the noise variance of each sample mean. However, under a fixed raw simulation budget, it also reduces the number of EI iterations. The efect of the latter reduction is governed by the kernel-specific information gain. To make this tradeof explicit, consider any candidate integer $\begin{array} { r } { b \in \mathcal { B } _ { T } : = \check { \left\{ 2 , \dots , \left\lfloor \frac { T } { 2 } \right\rfloor \right\} } } \end{array}$ . Define $\begin{array} { r } { N _ { T } ( b ) : = \lfloor \frac { T } { b } \rfloor , \nu _ { b } : = b - 1 } \end{array}$ $\begin{array} { r } { \alpha _ { T } ( b ) : = \frac { \delta } { 4 N _ { T } ( b ) } , \kappa _ { T } ( b ) : = \operatorname* { m a x } \left\{ 1 , \frac { \nu _ { b } } { \chi _ { \nu _ { b } , \alpha _ { T } ( b ) } ^ { 2 } } \right\} , h _ { T } ( b ) : = \operatorname* { m a x } \left\{ 1 , \frac { \chi _ { \nu _ { b } , 1 - \alpha _ { T } ( b ) } ^ { 2 } } { \nu _ { b } } \right\} } \end{array}$ , and $W _ { T } ( b ) : = M h _ { T } ( b )$ . For the same candidate $b ,$ denote $\gamma _ { T } ( b )$ and $\dot { \Lambda } _ { T } ( b )$ as the information gain and the confidence coeficient for the SK predictor with $b _ { T } = b , n = N _ { T } ( b )$ and $\kappa _ { T } = \kappa _ { T } ( b )$ in equations (5) and (6). Thus, increasing b reduces the number of EI evaluations $N _ { T } ( b )$ , but it also makes the sample-variance correction $\kappa _ { T } ( b )$ smaller and reduces the noise variance of each sample mean to order $1 / b$ in $\gamma _ { T } ( b )$

After setting $c _ { n } = \Lambda _ { n }$ and $d _ { T } = \Lambda _ { N _ { T } }$ using the quantities associated with the same candidate $b ,$ Corollary 6.2 gives the following certificate:

$$
\mathcal { R } _ { T } ( b ) : = 8 \sqrt { \frac { \left[ \Lambda _ { T } ^ { 2 } ( b ) + \log \left( \frac { 1 + T / v _ { 0 } } { k _ { \mathrm { m i n } } } \right) \right] b \gamma _ { T } ( b ) } { T \log \left( 1 + b / W _ { T } ( b ) \right) } } .\tag{12}
$$

For every predetermined $b \in B _ { T }$ , Corollary 1 gives $f ( \hat { x } _ { T } ) - f ^ { \star } \leq \mathcal { R } _ { T } ( b )$ with probability at least $1 - \delta .$ . The exact maximum information gain in (12) is generally unavailable in closed form. However, there exist upper bounds in the literature for the information gain. Define $\begin{array} { r } { U _ { T } ( b ) : = \frac { b N _ { T } ( b ) } { v _ { 0 } } } \end{array}$ . For a kernel $k ,$ the information-gain bound can be denoted as $G _ { k } ( U _ { T } ( b ) )$ (Vakili et al. 2021b). Such kernel-specific information-gain bounds are mainly derived by using the smoothness of the kernel or the decay of its eigenvalues. Their main purpose is to describe how information gain grows as the number of observations increases. Although these bounds remain valid when only a small number of observations are available, they may then be loose. To obtain a fair comparison among diferent choices of $b _ { T }$ and $N _ { T } .$ , we derive the following bound for $\gamma _ { T } ( b )$ that can be tighter with smaller $N _ { T }$

Lemma 6.3. Under Assumption 3.2, for any candidate replication number $b ,$

$$
\gamma _ { T } ( b ) \leq \frac { N _ { T } ( b ) } { 2 } \log \left( 1 + \frac { b } { v _ { 0 } } \right) .
$$

Unlike the kernel-specific bound, Lemma 6.3 only uses the number of batch observations and the normalization of the kernel. The result can be interpreted as follows: each batch mean contributes at most $\begin{array} { r } { \frac { 1 } { 2 } \log \left( 1 + \frac { b } { v _ { 0 } } \right) } \end{array}$ to the information gain, and there are at most $N _ { T } ( b )$ such observations. This bound becomes useful when b is large and the number of batches is small. We then combine the two bounds and define

$$
\overline { { \gamma } } _ { T } ^ { k } ( b ) : = \operatorname* { m i n } \Bigg \{ G _ { k } \big ( U _ { T } ( b ) \big ) , \frac { N _ { T } ( b ) } { 2 } \log \bigg ( 1 + \frac { b } { v _ { 0 } } \bigg ) \Bigg \} .
$$

After determining the incumbent and terminal coeficients, we introduce the following mild conditions on the kernel-specific information-gain bound.

Assumption 6.4. Let $\begin{array} { r } { g _ { T } : = G _ { k } \left( \frac { T } { v _ { 0 } } \right) } \end{array}$ . The kernel-specific bound $G _ { k }$ satisfies the following conditions:

1. $G _ { k }$ is nondecreasing, and there is a constant $C _ { k } < \infty$ such that $G _ { k } ( 2 u ) \leq C _ { k } G _ { k } ( u )$ for all suficiently large u;

2. log $T = { \cal { O } } ( g _ { T } )$

3. $g _ { T } = o \left( { \sqrt { T \log \log T } } \right)$

The first condition requires the spectral information-gain bound to change by at most a constant factor when its argument changes by a constant factor. The second condition ensures that the information-gain term is at least of logarithmic order. The third condition requires it to grow more slowly than $\sqrt { T \log \log T }$ . These conditions are satisfied by most standard kernel classes. They include finite-rank kernels, such as linear and fixed-degree polynomial kernels; kernels with exponentially decaying eigenvalues, including the SE kernel; and kernels with suficiently fast polynomial eigenvalue decay. In the last class, the required growth condition holds when the polynomial decay exponent is greater than two, which includes Matérn kernels with $\nu > d / 2$ . Finite-rank and exponential-eigendecay kernels have logarithmic or polylogarithmic information gain, while suficiently fast polynomial eigendecay gives a sub-square-root information-gain order. These properties ensure that the three conditions hold. Detailed verification for these kernel classes is provided in Appendix M.

We next choose the order of the replication size by minimizing a tractable kernel-specific upper bound. For each candidate $b ,$ replace the exact information gain with its upper bound $\overline { { \gamma } } _ { T } ^ { k } ( b )$ . Then, the optimal replication number preferred by the displayed bound is

$$
b _ { T } ^ { \mathrm { b d } } \in \arg \operatorname* { m i n } _ { b \in B _ { T } } \overline { { \mathcal { R } } } _ { T } ^ { k } ( b )
$$

where $\overline { { \mathcal { R } } } _ { T } ^ { k } ( b )$ is the upper bound of $\mathcal { R } _ { T } ( b )$ with $\gamma _ { T } ( b )$ replaced by its upper bound $\overline { { \gamma } } _ { T } ^ { k } ( b )$ in (12). The optimal order of $b _ { T }$ is presented in the following theorem.

Theorem 6.5. For $\delta \in ( 0 , 1 )$ , under Assumptions ${ \mathit { 3 . 1 , 3 . 2 , 4 . 1 , } }$ and $6 . 4 ,$ as $\begin{array} { r } { T  \infty , b _ { T } ^ { \mathrm { b d } } = \Theta ( \log \frac { T } { \delta } ) } \end{array}$

From this theorem and its proof, we find that replication smaller than logarithmic order produces a large correction $\kappa ,$ whereas replication larger than logarithmic order sacrifices spatial exploration without improving the leading order of the bound. Together with the calibrated c and $d _ { T }$ , these choices form a coherent algorithmic design: an optimistic incumbent for exploration, logarithmic replication for variance learning, and a conservative recommendation for implementation.

Under the rule $b _ { T } = \Theta ( \log ( T / \delta ) )$ , Corollary 6.2 gives

$$
f ( \hat { x } _ { T } ) - f ^ { \star } = O \left( g _ { T } \sqrt { \frac { \log ( T / \delta ) } { T \log \log ( T / \delta ) } } \right) .\tag{13}
$$

This bound is consistent with related results in the literature. For variance-rescaled noisy EI, an $O ( \gamma _ { T } \sqrt { T } )$ cumulative-regret bound has been established (see discussions in Takeno et al. 2025). If one of the sampled points is returned uniformly at random, its conditional expected regret is therefore $O ( \gamma _ { T } / \sqrt { T } )$ , matching the polynomial order of our bound. Notably, these results assume known homoscedastic noise, whereas our guarantee accommodates unknown heteroscedastic variances learned through replications, further demonstrating the strength of our result. The next subsection makes the resulting guarantee explicit by deriving simple-regret rates for diferent kernels.

## 6.4 Kernel-specific rates and comparison with lower bounds

We now substitute the corresponding kernel-specific information-gain bounds into (13). Constants depending on $B _ { f } , v _ { \mathrm { m i n } } , \bar { v } _ { \mathrm { ; } }$ , and the kernel parameters are suppressed. The notation $\widetilde O$ suppresses logarithmic factors.

SE kernels. According to Vakili et al. (2021b), the information gain bound is $g _ { T } = O \left( ( \log T ) ^ { d + 1 } \right)$ . Therefore,

$$
f ( \hat { x } _ { T } ) - f ^ { \star } = O \left( \frac { ( \log T ) ^ { d + 3 / 2 } } { \sqrt { T \log \log T } } \right) .\tag{14}
$$

This polynomial order can be compared with the minimax lower bound under SE kernel. Even when the Gaussian noise variance $v _ { 0 } > 0$ is known and constant, Cai and Scarlett (2021, Theorem 1) show that an algorithm needs

$$
T = \Omega \left( { \frac { v _ { 0 } } { \varepsilon ^ { 2 } } } \left( \log { \frac { B _ { f } } { \varepsilon } } \right) ^ { d / 2 } \log { \frac { 1 } { \delta } } \right)
$$

observations to attain simple regret at most ε with probability at least $1 - \delta$ over the SE RKHS ball. For fixed $B _ { f }$ $v _ { 0 }$ , and $\delta ,$ the unavoidable polynomial dependence is $T ^ { - 1 / 2 }$ . The lower bound permits adaptive sampling and an arbitrary returned point, and therefore also applies to the present setting. Hence, (14) is minimax optimal in its polynomial dependence on the raw simulation budget, up to logarithmic factors.

Matérn kernels. For a Matérn kernel with $\nu > d / 2$ , the information gain bound is $g _ { T } = O \left( T ^ { \frac { d } { 2 \nu + d } } ( \log T ) ^ { \frac { 2 \nu } { 2 \nu + d } } \right)$ Substitution into (13) yields

$$
\begin{array} { r } { f ( \hat { x } _ { T } ) - f ^ { \star } = \widetilde O \left( T ^ { - \frac { 2 \nu - d } { 2 ( 2 \nu + d ) } } \right) . } \end{array}
$$

As mentioned above, related bounds are available under known homoscedastic noise (no replication is required), so comparing our result with them helps understand the efect of learning heteroscedastic noise through replications. The comparison shows that the price of learning unknown heteroscedastic noise enters through logarithmic factors. For instance, with known homoscedastic noise, the same information-gain argument gives

$$
f ( \hat { x } _ { T } ) - f ^ { \star } = O \left( \frac { ( \log T ) ^ { d + 1 } } { \sqrt { T } } \right) ,
$$

for SE kernels. Thus, under the present analysis, estimating the unknown local variances changes the SE bound by the logarithmic factor $\sqrt { \frac { \log T } { \log \log T } }$ , but does not change its polynomial dependence on the raw simulation budget. Similar results can be found for the Matérn kernels.

## 7 Numerical Experiments

We conduct two complementary numerical studies. The first uses generated objectives that lie exactly in the assumed RKHS, with known kernel hyperparameters, to examine the finite-budget convergence behavior and the three design choices identified by the regret analysis. The second compares our regret certificate guided EI (CertEI) with SKO, EQI, and TSSO, first on the same RKHS objectives and then on four standard test functions. The RKHS study isolates the mechanisms analyzed by the theory, whereas the standard test functions examine the robustness of the resulting algorithm beyond the matched-kernel setting.

## 7.1 Controlled RKHS Experiments

We first use controlled RKHS objectives to examine two questions: whether the finite-budget behavior of CertEI is consistent with the upper-bound rates derived in Section 6.4, and how its three design choices afect its empirica performance.

## 7.1.1 Controlled setting.

For each dimension $d \in \{ 2 , 4 \}$ and each of the Matérn-5/2 and SE kernels, we construct one objective function $f ( x )$ from the corresponding RKHS on decision space $[ 0 , 1 ] ^ { d }$ in the following form:

$$
f ( x ) = \sum _ { j = 1 } ^ { 1 0 0 } w _ { j } k ( x , z _ { j } ) .
$$

A simulation replication at x follows $Y ( x ) = f ( x ) + \tau ( x ) Z .$ where $Z \sim N ( 0 , 1 )$ independently and $\tau ( x )$ varies with the objective value. Specifically, following the construction in Jalali et al. (2017), we let the noise standard deviation increase linearly with the objective value. Denote $R _ { f } = f _ { \mathrm { m a x } } - f ^ { \star }$ as the response range and define $\begin{array} { r } { u ( x ) = \mathrm { c l i p } \left( \frac { f ( x ) - f ^ { \star } } { R _ { f } } , 0 , 1 \right) } \end{array}$ . We then set $\tau ( x ) = R _ { f } \left[ 0 . 7 5 + 2 . 2 5 u ( x ) \right]$ . Consequently, the noise standard deviation equals $0 . 7 5 \dot { R _ { f } }$ at a global minimizer and increases to $3 R _ { f }$ at the largest objective value. The response GP uses the kernel and hyperparameters employed to construct the objective, and these hyperparameters are held fixed throughout each run. Following related theoretical studies (see, e.g., Wang et al. 2025), the known hyperparameters are used to align the experiment closely with the conditions of theoretical analysis and isolate the convergence behavior of the algorithm. Thus, the response-GP hyperparameters are known, but the input-dependent simulation variance must still be learned from replications. For each combination of test function and budget, we perform 100 independent runs. Performance is measured by the simple regret $\mathrm { G A P } ( T ) = f ( \widehat { x } _ { T } ) - f ^ { \star }$ . The complete objective construction and related experimental details are reported in Appendix O.

## 7.1.2 Empirical evidence of the regret bound

Figure 1 compares the average simple regret of CertEI over 100 macro-replications with the polynomial parts of the corresponding theoretical rates. The dashed lines represent the regret bound, with orders of $T ^ { - 3 / 1 4 }$ and $T ^ { - 1 / 2 }$ in two dimensions, and $T ^ { - 1 / 1 8 }$ and $T ^ { - 1 / 2 }$ in four dimensions, for the Matérn-5/2 and SE kernels, respectively. Each dashed line is normalized to agree with CertEI at the first displayed budget. We can observe that the mean simple regret decreases throughout the displayed budget range in all four settings. In addition, the empirical values remain below the corresponding reference curves. For example, on the four-dimensional Matérn-5/2 objective, the mean GAP decreases from 2.5861 at $T = e ^ { 8 }$ to 0.1017 at $\bar { T } = e ^ { 1 0 }$ . This finite-budget behavior is consistent with the theoretical orders. As these orders are upper bounds, empirical regret often decreases faster than the stated theoretical order, as observed both in our experiments and in related studies (Wang et al. 2025).

![](images/7909cbbc959772bfa941415b05cbc71217e5be47cbd4237ec4c490f373d97449.jpg)  
(a) 2d Matérn-5/2

![](images/8f07eb81da79328a1305e2969220721918cd66a09db805fc283c0733abd15ef6.jpg)  
(b) 2d squared exponential

![](images/f6d7522d5a0fb8ca5fbb2a166f22b7c35e90a7cfa349b4cb9063601c99c0534f.jpg)  
(c) 4d Matérn-5/2

![](images/ac29103ef450a0aa69229bf490da08f9310943e6d2c032e895026771f77597e2.jpg)  
(d) 4d squared exponential  
Figure 1: Average simple regret of CertEI on the RKHS objectives.

## 7.1.3 Ablation of the design choices

We next conduct ablation experiments to evaluate the efect of each design choice. To isolate the efect, we consider three sets of experiments, varying only one design choice at a time while holding all other experimental details fixed. For the EI incumbent, we compare min $. x \{ \mu _ { n } ( x ) - \Lambda _ { n } \sigma _ { n } ( x ) \}$ , min ${ } _ { x } \mu _ { n } ( x )$ , and min ${ \bf \Phi } _ { x } \{ \mu _ { n } ( x ) + \sigma _ { n } ( x ) \}$ . For the replication size, we compare the orders O(1), O(log log T), O(log T), and ${ \cal O } ( T ^ { 1 / 2 } )$ . For the final recommendation, we apply the three terminal scores $\mu _ { N _ { T } } ( x ) - \Lambda _ { N _ { T } } \sigma _ { N _ { T } } ( x )$ , and $\mu _ { N _ { T } } ( x ) , \mu _ { N _ { T } } ( x ) + \Lambda _ { N _ { T } } \sigma _ { N _ { T } } ( x )$ to the set of sampled designs. The comparison levels are chosen to reflect both common practice and systematic deviations from our theoretically motivated choices. The alternative incumbent and final-recommendation rules are commonly used in the literature. For replication, O(1) corresponds to the fixed equal-replication rule used in prior studies, while O(log log T) and $O ( T ^ { 1 / 2 } )$ are smaller and larger, respectively, than our O(log T) choice. They represent diferent tradeofs between more decision points and more precise estimates at each point. To keep the presentation compact, Figure 2 reports only the average GAP of the four-dimensional Matérn-5/2 instance. Results for the two-dimensiona Matérn-5/2, two-dimensional SE, and four-dimensional SE objectives are reported in Appendix O.

In Figure 2 (a), the calibrated lower-confidence incumbent consistently produces the smallest average GAP. Under small simulation budgets, all three rules perform similarly. As the budget increases, however, the other two rules converge more slowly, with the upper-quantile incumbent exhibiting the slowest convergence. This indicates that they do not explore the decision space suficiently to locate the region containing the global minimizer, which is consistent with our theoretical analysis. In Figure 2 (b), the calibrated logarithmic replication schedule attains the smallest mean GAP. The two small replications choices preserve more search opportunities but provide noisier local estimates, while the square-root choice can leave too few adaptive decisions. Similar to our theoretical analysis, they all attain lower convergence rates. In Figure 2 (c), the proposed final rule and the posterior-mean rule are nearly indistinguishable on this test, both much better than the lower quantile criterion. Thus, the proposed final rule avoids the deterioration caused by rewarding residual uncertainty. We also report an additional four-dimensional Matérn-5/2 experiment showing the advantage of our rule over selecting the smallest posterior mean in Appendix O.4.1. Under larger noise, the same budget leaves greater residual uncertainty in the candidate solutions, making the diference between the two rules more pronounced. The results show that explicitly penalizing this uncertainty can produce a more reliable final recommendation.

![](images/ff5608731802f6e7cf5f126faba7feb4cadd35ad9c69138da86d06843a1f65e1.jpg)  
(a) EI incumbent

![](images/97de02826a5143e6f5c8de02fe8857cf6806cd6c05171a17541d71a5603baace.jpg)  
(b) Replication order

![](images/e75dd1227d71729f23d5f2ba64a76d6e9e02fc957a82dc3c6641364ea893032c.jpg)  
(c) Final recommendation  
Figure 2: Controlled ablations on the four-dimensional Matérn-5/2 objective.

The preceding one-factor-at-a-time ablations isolate the efect of each design choice. In Appendix O.5, we present a fully crossed experiment over all levels of the three design choices considered in the preceding ablations. This yields $3 \times 4 \times 3 = 3 6$ algorithmic configurations, which we evaluate on the RKHS objectives. As a complement to the one-factor-at-a-time ablations, this experiment examines whether the selected design choices remain efective when used jointly. The results are consistent with the individual ablations. Across all tested settings, the proposed configuration consistently ranks among the top two of the 36 configurations.

## 7.2 Comparison with Existing Algorithms

We next compare CertEI with commonly used EI algorithms for stochastic simulations, namely $\mathrm { S K O ^ { 1 } }$ , EQI, and TSSO. We first compare them under the RKHS objectives in the last subsection with an ideal setting of known GP hyperparameter. We then use four synthetic test functions to evaluate the algorithms when the GP hyperparameters must be learned during optimization. Each test is evaluated using 100 macro-replications.

The four synthetic functions are six-hump Camelback, rescaled Branin, Hills, and Hartmann-6. The Camelback and Branin functions are low-dimensional multimodal problems with several well-separated minima. The Hills function provides a more densely multimodal landscape with many competing local optima, while Hartmann-6 examines performance on a higher-dimensional multimodal problem. Together, these functions cover optimization landscapes with diferent dimensions and degrees of multimodality. As in the preceding RKHS experiments, we add Gaussian noise to these functions following the construction of Jalali et al. (2017). The complete settings of these functions are provided in Appendix O.6.

Figure 3 reports the comparison results.

We can observe that at the smaller budgets, the relative performance of the four algorithms varies across the six problems, and CertEI is not always the best-performing method. As the budget increases, however, CertEI exhibits the most consistent reduction in average GAP. From approximately $T = e ^ { 8 . 5 }$ through $T = e ^ { 1 0 }$ , it attains the smallest average GAP on all six problems. In particular, CertEI continues to improve on the four analytical functions after some competing methods begin to flatten, and it achieves the smallest average GAP on every problem at $T = e ^ { 1 0 }$ . The advantage is most pronounced on Camelback and rescaled Branin, while the Hills problem remains dificult for all four methods and exhibits substantial variation across macro-replications. CertEI also maintains its improvement on the higher-dimensional Matérn-5/2, SE, and Hartmann-6 problems, indicating that the observed performance is not limited to low-dimensional test functions.

Overall, these experiments provide complementary evidence for both the convergence behavior and the design choices underlying CertEI. On the fixed RKHS objectives, the empirical GAP decreases consistently with, and often faster than, the rates suggested by the theoretical upper bounds. The one-factor-at-a-time and fully crossed ablations further show that the proposed incumbent, replication order, and final recommendation form a robust combination. In comparison with existing EI algorithms, CertEI exhibits the most consistent improvement as the budget increases and attains the smallest average GAP on all six test problems. Taken together, the results support the proposed algorithmic design and the competitive finite-budget performance of CertEI.

![](images/9ac822f6c34c2111e7f7acf24a55df5b19925c683305f0f0c7bcc7a7513f4bb4.jpg)  
(a) 4d Matérn-5/2 objective

![](images/33c583ce467b63f9ba8379610de186fcba15870da355e9a04d06d1e805d5f824.jpg)  
(b) 4d SE objective

![](images/f97f3782adc7d53ea8a559b0b6a5ebc4314f54f718ba828bb38e70aff102c4e7.jpg)  
(c) 2d six-hump Camelback

![](images/cc2d29f0be3644210dbc3feda34c55491f4c6823b1d4f75ce235a6c67b43285e.jpg)  
(d) 2d rescaled Branin

![](images/69a412dbb1e92993b9cdb5e812554a4bf3beb2c966c6abc0cf522266aa128ff3.jpg)  
(e) 2d Hills

![](images/635a0e5bde02a77c6b2a864bdd34373b8f19482f2181271b6892d43baa1bab93.jpg)  
(f) 6d Hartmann-6  
Figure 3: Average GAP of the four algorithms

## 8 Summary

This paper studies how to design an EI algorithm for continuous simulation optimization when the simulation noise is heteroscedastic and unknown. Designing such an algorithm requires specifying three basic components: the incumbent, the replication size, and the final recommendation. Existing theoretical and empirical studies generally either assume that the heteroscedastic noise variances are known or determine these three components separately using heuristic rules. In contrast, we take a theory-guided approach: we derive a high-probability simple regret bound and use it as a common criterion to systematically determine all three design choices. This bound is built on a finite-sample confidence inequality for the SK predictor when the noise variances are estimated from replications. Within the rule classes considered, to tighten the regret bound, the resulting method uses a lowerquantile incumbent to preserve exploration, Θ(log T) replications to balance variance estimation and spatial search, and a conservative upper-quantile recommendation over sampled designs. The resulting simple regret is ${ \widetilde O } ( T ^ { - 1 / 2 } )$ for SE kernels, matching the polynomial dependence in the minimax lower bound, while the Matérn rate matches its known-homoscedastic confidence-bound counterpart up to logarithmic factors. The numerical results and ablation studies are consistent with the theoretical convergence behavior and support the three design choices. Comparisons with existing common EI algorithms further show the competitive empirical performance of our calibrated design choices.

## A Proof of Lemma 4.2

Proof. Proof of Lemma 4.2. Conditional on the history $\begin{array} { r } { \mathcal { H } _ { q - 1 } , \ Z _ { q } : = \ \frac { \nu _ { T } S _ { q } ^ { 2 } } { \upsilon ( x _ { q } ) } \ \sim \ \chi _ { \nu _ { T } } ^ { 2 } } \end{array}$ , where $\nu _ { T } = b _ { T } - 1$ . By the definition of the chi-squared quantile, $\mathbb { P } \big ( Z _ { q } < \chi _ { \nu _ { T } , \alpha _ { T } } ^ { 2 } \big | \mathcal { H } _ { q - 1 } \big ) \le$ α<sub>T</sub> and $\mathbb { P } \big ( Z _ { q } > \chi _ { \nu _ { T } , 1 - \alpha _ { T } } ^ { 2 } \big | \mathcal { H } _ { q - 1 } \big ) \le \alpha _ { T }$ . In addition, according to the definition of $\kappa _ { T }$ and $h _ { T } ,$ , we have $1 / \kappa _ { T } \leq \chi _ { \nu _ { T } , \alpha _ { T } } ^ { 2 } / \nu _ { T }$ and h<sub>T</sub> $\ge \chi _ { \nu _ { T } , 1 - \alpha _ { T } } ^ { 2 } / \nu _ { T }$ . Thus, we

have

$$
\mathbb { P } \bigg ( S _ { q } ^ { 2 } < \frac { v ( x _ { q } ) } { \kappa _ { T } } \bigg ) \leq \mathbb { P } \Bigg ( S _ { q } ^ { 2 } < \frac { v ( x _ { q } ) \chi _ { v _ { T } , \alpha _ { T } } ^ { 2 } } { \nu _ { T } } \Bigg ) \leq \alpha _ { T } , \mathbb { P } \big ( S _ { q } ^ { 2 } > v ( x _ { q } ) h _ { T } \big ) \leq \mathbb { P } \Bigg ( S _ { q } ^ { 2 } > \frac { v ( x _ { q } ) \chi _ { v _ { T } , 1 - \alpha _ { T } } ^ { 2 } } { \nu _ { T } } \Bigg ) \leq \alpha _ { T } .
$$

Therefore, except on an event with conditional probability at most $2 \alpha _ { T }$ , we have $\begin{array} { r } { \frac { v ( x _ { q } ) } { \kappa r } \leq S _ { q } ^ { 2 } \leq h _ { T } v ( x _ { q } ) } \end{array}$ . Taking the union bound over these events on all $N _ { T }$ batches gives, with probability at least $1 ^ { ' } - \delta / 2$

$$
\frac { v ( x _ { q } ) } { \kappa _ { T } } \leq S _ { q } ^ { 2 } \leq h _ { T } v ( x _ { q } ) , \qquad \forall q \leq N _ { T } .
$$

On this event, $v ( x _ { q } ) \leq \kappa _ { T } S _ { q } ^ { 2 } \leq \kappa _ { T } \widetilde { v } _ { q } .$ , and

$$
v _ { 0 } \leq \widetilde { v } _ { q } = \operatorname* { m a x } \{ v _ { 0 } , S _ { q } ^ { 2 } \} \leq \operatorname* { m a x } \{ v _ { 0 } , h _ { T } V _ { \star } \} \leq h _ { T } \operatorname* { m a x } \{ v _ { 0 } , V _ { \star } \} = W _ { T } .
$$

Dividing by $b _ { T }$ proves the last ineuqality in Lemma 4.2.

## B Proof of Lemma 4.3

Proof. Proof of Lemma 4.3. For a sequence of candidate replication numbers $b = b _ { T }$ , write $\nu _ { T } : = b _ { T } - 1$ , and $\begin{array} { r } { L _ { T } : = \log \frac { 4 N _ { T } } { \delta } } \end{array}$ . Then $\alpha _ { T } = e ^ { - L _ { T } }$ . We first prove the three statements for $\kappa _ { T }$ separately.

Case 1: $: \ b _ { T } = o ( \log T )$ . We first establish a lower bound on the lower tail of a chi-squared random variable. Let $\begin{array} { r } { Q _ { \nu } = \sum _ { j = 1 } ^ { \nu } Z _ { j } ^ { 2 } \sim \chi _ { \nu } ^ { 2 } } \end{array}$ , where $Z _ { 1 } , \ldots , Z _ { \nu }$ are independent standard normal random variables. There exists a constant $c _ { 0 } > 0$ such that

$$
\mathbb { P } ( | Z _ { 1 } | \le a ) \ge c _ { 0 } a , \qquad 0 < a \le 1 .\tag{15}
$$

We can select this $c _ { 0 }$ as follows. As the standard normal density is continuous and strictly positive on $[ - 1 , 1 ]$ , we have

$$
\mathbb { P } ( | Z _ { 1 } | \le a ) = \int _ { - a } ^ { a } \phi ( z ) d z \ge 2 a \operatorname* { m i n } _ { | z | \le 1 } \phi ( z ) ,
$$

and (15) follows by taking $c _ { 0 } : = 2 \phi ( 1 ) > 0$

If $0 < a \le 1$ , then $\{ | Z _ { j } | \le a , \ j = 1 , \ldots , \nu \} \subseteq \{ Q _ { \nu } \leq \nu a ^ { 2 } \}$ . Hence, by independence,

$$
\mathbb { P } ( Q _ { \nu } \le \nu a ^ { 2 } ) \ge \prod _ { j = 1 } ^ { \nu } \mathbb { P } ( | Z _ { j } | \le a ) \ge ( c _ { 0 } a ) ^ { \nu } .\tag{16}
$$

$$
\nu = \nu _ { T }
$$

$$
\begin{array} { r } { a _ { T } : = c _ { 0 } ^ { - 1 } \exp \left( - \frac { L _ { T } } { \nu _ { T } } \right) } \end{array}
$$

$$
b _ { T } = o ( \log T )
$$

$$
\begin{array} { r } { L _ { T } = \log { \frac { 4 N _ { T } } { \delta } } = \Theta ( \log T ) } \end{array}
$$

$$
\begin{array} { r } { N _ { T } = \frac { T } { b _ { T } } } \end{array}
$$

$$
1 \leq \nu _ { T } = b _ { T } - 1 = o ( \log T )
$$

$$
\begin{array} { r } { \frac { L _ { T } } { \nu _ { T } } \longrightarrow \infty } \end{array}
$$

$$
a _ { T } \leq 1
$$

Substitution of a into (16) gives P $\left( Q _ { \nu _ { T } } \leq \nu _ { T } a _ { T } ^ { 2 } \right) \geq ( c _ { 0 } a _ { T } ) ^ { \nu _ { T } } = \exp ( - L _ { T } ) = \alpha _ { T }$ . By the definition of the lower α<sub>T</sub>-quantile, $\chi _ { \nu _ { T } , \alpha _ { T } } ^ { 2 } \leq \nu _ { T } a _ { T } ^ { 2 }$ . Therefore,

$$
\kappa _ { T } \geq a _ { T } ^ { - 2 } = c _ { 0 } ^ { 2 } \exp \left( \frac { 2 L _ { T } } { \nu _ { T } } \right) .\tag{17}
$$

Since $L _ { T } / \nu _ { T } \to \infty , \kappa _ { T } \longrightarrow \infty$

Case 2: $b _ { T } = \Theta ( \log T )$ . In this case, $\begin{array} { r } { N _ { T } = \Theta \left( \frac { T } { \log T } \right) } \end{array}$ , so $L _ { T } = \Theta ( \log T )$ . Since $\nu _ { T } = b _ { T } - 1 = \Theta ( \log T )$ , there exists a constant $c _ { 1 } > 0$ such that $\frac { \nu _ { T } } { L _ { T } } \geq c _ { 1 }$ for all suficiently large T.

We use a standard Chernof bound for the lower tail of a chi-squared random variable. Let $Q _ { \nu } \sim \chi _ { \nu } ^ { 2 }$ . For any $s \in ( 0 , 1 )$ , according to Ghosh (2021), we have

$$
\begin{array} { r } { \mathbb { P } ( Q _ { \nu } \le s \nu ) \le \exp \left\{ - \nu I ( s ) \right\} , } \end{array}\tag{18}
$$

where $\begin{array} { r } { I ( s ) : = \frac { s - 1 - \log s } { 2 } } \end{array}$ . Because $I ( s ) \longrightarrow \infty \mathrm { \ a s \ } s \downarrow 0$ , we can choose a fixed $s _ { 0 } \in ( 0 , 1 )$ such that $c _ { 1 } I ( s _ { 0 } ) > 1$ . It follows that, $\nu _ { T } I ( s _ { 0 } ) > L _ { T }$ for all suficiently large T. Hence,

$$
\begin{array} { r } { \mathbb { P } \left( Q _ { \nu _ { T } } \le s _ { 0 } \nu _ { T } \right) \le \exp \left\{ - \nu _ { T } I ( s _ { 0 } ) \right\} < e ^ { - L _ { T } } = \alpha _ { T } . } \end{array}
$$

Therefore, $\chi _ { \nu _ { T } , \alpha _ { T } } ^ { 2 } \geq s _ { 0 } \nu _ { T }$ . It follows that $\begin{array} { r } { \kappa _ { T } \leq \operatorname* { m a x } \left\{ 1 , \frac { 1 } { s _ { 0 } } \right\} = \frac { 1 } { s _ { 0 } } } \end{array}$ . Thus, $\kappa _ { T } = O ( 1 )$

Case 3: $b _ { T } / \log T \to \infty$ . Since $\begin{array} { r } { N _ { T } \leq T , L _ { T } = \log \frac { 4 N _ { T } } { \delta } = O ( \log T ) } \end{array}$ . On the other hand, $\nu _ { T } = b _ { T } - 1$ satisfies $\frac { \nu _ { T } } { \log T } \longrightarrow \infty$ . Hence, $\begin{array} { r } { \frac { L _ { T } } { \nu _ { T } } \longrightarrow 0 } \end{array}$ . The Laurent–Massart lower-tail inequality gives, for every $x > 0$

$$
\begin{array} { r } { \mathbb { P } \left( Q _ { \nu _ { T } } \leq \nu _ { T } - 2 \sqrt { \nu _ { T } x } \right) \leq e ^ { - x } . } \end{array}
$$

Taking $x = L _ { T }$ gives

$$
\begin{array} { r } { \mathbb { P } \left( Q _ { \nu _ { T } } \leq \nu _ { T } - 2 \sqrt { \nu _ { T } L _ { T } } \right) \leq e ^ { - L _ { T } } = \alpha _ { T } . } \end{array}
$$

Therefore, $\chi _ { \nu _ { T } , \alpha _ { T } } ^ { 2 } \geq \nu _ { T } - 2 \sqrt { \nu _ { T } L _ { T } }$ . For all suficiently large $T ,$ the right-hand side is positive, and hence

$$
1 \leq \kappa _ { T } \leq \frac { \nu _ { T } } { \nu _ { T } - 2 \sqrt { \nu _ { T } L _ { T } } } = \frac { 1 } { 1 - 2 \sqrt { L _ { T } / \nu _ { T } } } .
$$

Since $\begin{array} { r } { \frac { L _ { T } } { \nu _ { T } } \longrightarrow 0 } \end{array}$ , the right-hand side converges to one. Therefore, $\kappa _ { T } \longrightarrow 1$

Fixed replication number. Finally, suppose that $b _ { T } = b \geq 2$ is fixed, and write $\nu : = b - 1$ . The chi-squared density is

$$
f _ { \nu } ( x ) = \frac { x ^ { \nu / 2 - 1 } e ^ { - x / 2 } } { 2 ^ { \nu / 2 } \Gamma ( \nu / 2 ) } , \qquad x > 0 .
$$

As $x \downarrow 0 , e ^ { - x / 2 } = 1 + o ( 1 )$ , and hence

$$
\mathbb { P } ( Q _ { \nu } \leq x ) = \int _ { 0 } ^ { x } f _ { \nu } ( y ) d y = { \frac { x ^ { \nu / 2 } } { 2 ^ { \nu / 2 } \Gamma ( \nu / 2 + 1 ) } } \{ 1 + o ( 1 ) \} .
$$

Therefore, as $\alpha \downarrow 0 , \chi _ { \nu , \alpha } ^ { 2 } = \Theta \left( \alpha ^ { 2 / \nu } \right)$

For fixed $\begin{array} { r } { b , N _ { T } = \left\lfloor \frac { T } { b } \right\rfloor = \Theta ( T ) } \end{array}$ , so $\begin{array} { r } { \alpha _ { T } = \frac { \delta } { 4 N _ { T } } = \Theta ( T ^ { - 1 } ) } \end{array}$ . Consequently,

$$
\kappa _ { T } = \Theta \left( \alpha _ { T } ^ { - 2 / ( b - 1 ) } \right) = \Theta \left( T ^ { 2 / ( b - 1 ) } \right) .
$$

We next establish the corresponding results for $h _ { T }$ . For notation simplicity, define $q _ { T } : = \chi _ { \nu _ { T } , 1 - \alpha _ { T } } ^ { 2 }$ , and $\begin{array} { r } { r _ { T } : = \frac { L _ { T } } { \nu _ { T } } } \end{array}$ The Chernof bound in Ghosh (2021) gives, when $s > 1$

$$
\mathbb { P } ( Q _ { \nu } \geq s \nu ) \leq \exp \{ - \nu I ( s ) \} .
$$

Suppose first that $b _ { T } = o ( \log T )$ , so $r _ { T } \to \infty$ . Fix $\varepsilon \in ( 0 , 1 )$ and set $s _ { T } = ( 2 + \varepsilon ) r _ { T }$ . Since $I ( s ) \sim s / 2$ as $s \to \infty$

$$
\frac { \nu _ { T } I ( s _ { T } ) } { L _ { T } } \longrightarrow 1 + \frac { \varepsilon } { 2 } > 1 .
$$

Thus, for all suficiently large $T , s _ { T } \nu _ { T } = ( 2 + \varepsilon ) r _ { T } \nu _ { T } = ( 2 + \varepsilon ) L _ { T }$ . The Chernof bound above gives $\mathbb { P } ( Q _ { \nu _ { T } } \geq ( 2 + \varepsilon ) L _ { T } ) \leq$ $e ^ { - \nu _ { T } I ( s _ { T } ) } < e ^ { - L _ { T } } = \alpha _ { T }$ . Hence, $q _ { T } \leq ( 2 + \varepsilon ) L _ { T }$ for all suficiently large T. In addition, $x _ { T } = ( 2 - \varepsilon ) L _ { T }$ , we have

$$
\mathbb { P } ( Q _ { \nu _ { T } } > x _ { T } ) \ge \mathbb { P } ( Q _ { 1 } > x _ { T } ) \ge \int _ { x _ { T } } ^ { x _ { T } + 1 } { \frac { e ^ { - u / 2 } } { \sqrt { 2 \pi u } } } d u \ge { \frac { e ^ { - ( x _ { T } + 1 ) / 2 } } { \sqrt { 2 \pi ( x _ { T } + 1 ) } } } > e ^ { - L _ { T } } = \alpha _ { T }
$$

for all suficiently large T. Hence, $q _ { T } \geq ( 2 - \varepsilon ) L _ { T }$ . As ε is arbitrary, $q _ { T } \sim 2 L _ { T }$ , and therefore

$$
h _ { T } \sim \frac { 2 L _ { T } } { \nu _ { T } } = \Theta ( \frac { \log T } { b _ { T } - 1 } ) \longrightarrow \infty .
$$

In particular, for fixed $b _ { T } = b \geq 2 , h _ { T } = \Theta ( \log T )$

If $b _ { T } = \Theta ( \log T )$ , then r<sub>T</sub> is bounded as both $L _ { t }$ and ν<sub>T</sub> are $\Theta ( T )$ . Choose a fixed $s > 1$ such that $I ( s ) > r _ { T }$ eventually. This leads to $\mathbb { P } ( Q _ { \nu } \ge s \nu _ { T } ) \le \exp \{ - \nu _ { T } I ( s ) \} \le \exp \{ - \nu _ { T } r _ { T } \} = \alpha _ { T }$ . Hence, $q _ { T } \le s \nu _ { T }$ , so $1 \leq h _ { T } \leq s$ and $h _ { T } = \Theta ( 1 )$

Finally, if $b _ { T } / \log T \to \infty .$ , as shown in Case $3 , r _ { T }  0$ . For every fixed $\varepsilon > 0 , I ( 1 + \varepsilon ) > r _ { T }$ eventually. Setting $s = 1 + \epsilon$ and following the similar analysis above, we have $1 \leq h _ { T } \leq 1 + \varepsilon$ . Consequently, $h _ { T } \to 1$ . This completes the proof. □

## C Proof of Theorem 4.4

Proof. Proof of Theorem 4.4. Let V<sub>T</sub> be the event defined in the main text. The proof of Lemma 4.2 gives $\mathbb { P } ( \mathcal { V } _ { T } ^ { c } ) \le \delta / 2$ . We first define an auxiliary noise scale that is valid on every sample path, including paths outside $\nu _ { T }$

Before observing the simulation results at the qth point, the previous points have been observed and $x _ { q }$ has been selected. For the analysis, we construct the filtration by first revealing the sample variance $S _ { q } ^ { 2 }$ and then the sample mean $\bar { Y } _ { q }$ . Let $\mathcal { H } _ { q - 1 / 2 }$ denote the information available after this first reveal. Define $\begin{array} { r } { \varepsilon _ { q } : = \bar { Y } _ { q } - f ( x _ { q } ) , d _ { q } ^ { 2 } : = \frac { v ( x _ { q } ) } { b _ { T } } } \end{array}$ $\begin{array} { r } { r _ { q } ^ { 2 } : = \frac { \widetilde { v } _ { q } } { b _ { T } } } \end{array}$ , and $\begin{array} { r } { r _ { q } ^ { \circ } : = \operatorname* { m a x } \left\{ r _ { q } , \frac { d _ { q } } { \sqrt { \kappa _ { T } } } \right\} } \end{array}$ . The location $x _ { q } .$ , the sample variance, and these scales are measurable with respect to $\mathcal { H } _ { q - 1 / 2 }$ . Under Gaussian sampling, the sample mean and sample variance within a batch are independent. Thus, conditional on $\mathcal { H } _ { q - 1 / 2 } , \varepsilon _ { q } \sim N ( 0 , d _ { q } ^ { 2 } )$ . Moreover, the definition of $r _ { q } ^ { \circ }$ gives $d _ { q } ^ { 2 } \leq \kappa _ { T } ( r _ { q } ^ { \circ } ) ^ { 2 }$ . Therefore, for every $t \in \mathbb { R }$

$$
\mathbb { E } \big [ \mathrm { e x p } ( t \varepsilon _ { q } ) \big | \mathcal { H } _ { q - 1 / 2 } \big ] = \mathrm { e x p } \left( \frac { t ^ { 2 } d _ { q } ^ { 2 } } { 2 } \right) \leq \mathrm { e x p } \left( \frac { t ^ { 2 } \kappa _ { T } ( r _ { q } ^ { \circ } ) ^ { 2 } } { 2 } \right) .
$$

Hence, $\varepsilon _ { q }$ is conditionally $\sqrt { \kappa _ { T } } r _ { q } ^ { \circ }$ -sub-Gaussian.

Let $ { \mu ^ { \circ } } _ { n }$ and $\boldsymbol { \sigma } _ { n } ^ { \circ }$ be the SK prediction and predictive variance formed with $R _ { n } ^ { \circ } : = \operatorname { d i a g } ( ( r _ { 1 } ^ { \circ } ) ^ { 2 } , \ldots , ( r _ { n } ^ { \circ } ) ^ { 2 } )$ , and let $\begin{array} { r } { \Gamma _ { n } ^ { \circ } : = \frac { 1 } { 2 } } \end{array}$ log det $\left( I _ { n } + ( R _ { n } ^ { \circ } ) ^ { - \bar { 1 } / 2 } K _ { n } ( R _ { n } ^ { \circ } ) ^ { - 1 / 2 } \right)$ . We now apply the RKHS confidence inequality of Kirschner and Krause (2018, Lemma 7) to the enlarged filtration. In the notation of that lemma, set $\rho _ { q } = \sqrt { \kappa _ { T } } r _ { q } ^ { \circ } , \Sigma _ { n } = \kappa _ { T } R _ { n } ^ { \circ } .$ $\lambda = \kappa _ { T } ^ { - 1 }$ , and $\begin{array} { r } { \delta _ { 0 } = \frac { \delta } { 2 } } \end{array}$ . Then $\lambda \Sigma _ { n } = R _ { n } ^ { \circ }$ , so the estimator in that lemma is $ { \mu ^ { \circ } } _ { n }$ . Its confidence-width function equals $\sqrt { \kappa _ { T } } \sigma _ { n } ^ { \circ } ,$ , and

$$
\log \left( \frac { 1 } { \delta _ { 0 } } \frac { \operatorname* { d e t } ( \lambda \Sigma _ { n } + K _ { n } ) ^ { 1 / 2 } } { \operatorname* { d e t } ( \lambda \Sigma _ { n } ) ^ { 1 / 2 } } \right) = \Gamma _ { n } ^ { \circ } + \log \frac { 2 } { \delta } .
$$

Therefore, with probability at least $1 - \delta / 2$ , simultaneously for all $n \geq 1$ and all $x \in \mathcal { X }$

$$
| f ( x ) - \mu _ { n } ^ { \circ } ( x ) | \leq \left[ B _ { f } + \sqrt { 2 \kappa _ { T } \left( \Gamma _ { n } ^ { \circ } + \log \frac 2 \delta \right) } \right] \sigma _ { n } ^ { \circ } ( x ) .\tag{19}
$$

Denote the event that this bound holds as $\mathcal { E } _ { T } ^ { \circ }$

By construction,

$$
( r _ { q } ^ { \circ } ) ^ { 2 } \geq r _ { q } ^ { 2 } = \frac { \operatorname* { m a x } \{ v _ { 0 } , S _ { q } ^ { 2 } \} } { b _ { T } } \geq \frac { v _ { 0 } } { b _ { T } } .
$$

Therefore, $\begin{array} { r } { ( R _ { n } ^ { \circ } ) ^ { - 1 } \preceq \frac { b _ { T } } { v _ { 0 } } I _ { n } } \end{array}$ . Using the identity d $\operatorname { e t } ( I + A B ) = \operatorname* { d e t } ( I + B A )$ , we can rewrite

$$
\Gamma _ { n } ^ { \circ } = { \frac { 1 } { 2 } } \log \operatorname * { d e t } \left( I _ { n } + K _ { n } ^ { 1 / 2 } ( R _ { n } ^ { \circ } ) ^ { - 1 } K _ { n } ^ { 1 / 2 } \right) .
$$

The Loewner-order bound on $( R _ { n } ^ { \circ } ) ^ { - 1 }$ gives $K _ { n } ^ { 1 / 2 } ( R _ { n } ^ { \circ } ) ^ { - 1 } K _ { n } ^ { 1 / 2 } \preceq \frac { b _ { T } } { v _ { 0 } } K _ { n }$ . Since the map $A \mapsto \log \operatorname* { d e t } ( I + A )$ is monotone on the positive semidefinite cone, $\Gamma _ { n } ^ { \circ } \leq { \frac { 1 } { 2 } }$ log det $\begin{array} { r } { \left( I _ { n } + \frac { b _ { T } } { v _ { 0 } } K _ { n } \right) } \end{array}$ . By the definition of the replicated-noise information-gain envelope $\Gamma _ { n } ^ { \circ } \leq \gamma _ { n } ^ { \mathrm { S K } } , \forall n \leq N _ { T }$ . Substituting this bound into (19) gives the coeficient $\Lambda _ { n }$ in Theorem 4.4.

On $\nu _ { T }$ , Lemma 4.2 gives $d _ { q } ^ { 2 } \le \kappa _ { T } r _ { q } ^ { 2 }$ for every observed batch. Therefore, $r _ { q } ^ { \circ } = r _ { q }$ , and the auxiliary estimator is exactly the clipped plug-in SK predictor: $( \mu _ { n } ^ { \circ } , \sigma _ { n } ^ { \circ } ) = ( \mu _ { n } , \sigma _ { n } ) , n = 1 , \ldots , N _ { T } . ^ { \circ } \mathrm { A t } n = 0$ , the reproducing property and the kernel normalization give

$$
| f ( x ) | \leq \| f \| _ { \mathcal { H } _ { k _ { \theta } } } \sqrt { k _ { \theta } ( x , x ) } \leq B _ { f } \sigma _ { 0 } ( x ) \leq \Lambda _ { 0 } \sigma _ { 0 } ( x ) .
$$

All variance and confidence inequalities hold jointly on $\nu _ { T } \cap \mathcal { E } _ { T } ^ { \circ }$ . Their total failure probability is at most

$$
\mathbb { P } \left\{ ( \mathcal { V } _ { T } \cap \mathcal { E } _ { T } ^ { \circ } ) ^ { c } \right\} \leq \frac { \delta } { 4 } + \frac { \delta } { 4 } + \frac { \delta } { 2 } = \delta .
$$

The sequence $\gamma _ { n } ^ { \mathrm { S K } }$ is nondecreasing by definition, so $\Lambda _ { n }$ is also nondecreasing.

## D Proof of Lemma 5.1

Proof. Proof of Lemma 5.1. Fix $n < N _ { T }$ and abbreviate $c = c _ { n } , \xi = \xi _ { n } ^ { \mathbf { c } } , g = g _ { n + 1 } ^ { \mathbf { c } }$ , and $x = x _ { n + 1 } ^ { \mathbf { c } }$ . For every $y \in { \mathcal { X } }$ , global quantile minimality gives $\xi \le \mu _ { n } ( y ) - c \sigma _ { n } ( y )$ . Hence, $\begin{array} { r } { \frac { \xi - \mu _ { n } ( y ) } { \sigma _ { n } ( y ) } \leq - c . } \end{array}$ and the monotonicity of τ gives E $\left[ { } _ { n } ( y ; \xi ) \leq \sigma _ { n } ( y ) \tau ( - c ) \right]$ . At g, equality holds: EI ${ } _ { n } ( g ; \xi ) = \sigma _ { n } ( g ) \tau ( - c )$ . Since x maximizes EI,

$$
\sigma _ { n } ( x ) \tau ( - c ) \geq \operatorname { E I } _ { n } ( x ; \xi ) \geq \operatorname { E I } _ { n } ( g ; \xi ) = \sigma _ { n } ( g ) \tau ( - c ) .
$$

The function τ is strictly positive, so the common factor cancels and $\sigma _ { n } ( x ) \geq \sigma _ { n } ( g )$

## E Proof of Lemma 5.2

Proof. Proof of Lemma 5.2. The function τ is continuous and strictly increasing because $\tau ^ { \prime } ( z ) = \Phi ( z ) > 0$ , and its range is $( 0 , \infty )$ . Hence the equation defining $q _ { c } ( r )$ has a unique solution. Since $r \geq 1 , \tau ( - q _ { c } ( r ) ) \leq \tau ( - c )$ , which implies $q _ { c } ( r ) \geq c$

For $u \geq 0$ , define $H ( u ) : = e ^ { u ^ { 2 } / 2 } \tau ( - u )$ . Diferentiation of H gives

$$
H ^ { \prime } ( u ) = e ^ { u ^ { 2 } / 2 } \left\{ u \phi ( u ) - ( 1 + u ^ { 2 } ) \Phi ( - u ) \right\} .
$$

Since $\begin{array} { r } { ( 1 + u ^ { 2 } ) \Phi ( - u ) - u \phi ( u ) = \int _ { u } ^ { \infty } ( z - u ) ^ { 2 } \phi ( z ) d z \geq 0 , H ^ { \prime } ( u ) \leq 0 } \end{array}$ . Suppose first that $c \geq 0$ . Since $q _ { c } ( r ) \geq c$ and H is nonincreasing,

$$
r = \frac { \tau ( - c ) } { \tau ( - q _ { c } ( r ) ) } \geq \exp \left( \frac { q _ { c } ^ { 2 } ( r ) - c ^ { 2 } } { 2 } \right) .
$$

Taking logarithms proves $q _ { c } ( r ) \leq \sqrt { c ^ { 2 } + 2 \log r } .$

Now suppose that $c < 0$ . If $q _ { c } ( r ) \leq 0$ , then $q _ { c } ( r ) \leq \sqrt { 2 \log r }$ is immediate. If $q _ { c } ( r ) > 0$ , monotonicity of $\tau$ and the preceding tail-ratio bound with lower endpoint zero give

$$
r = \frac { \tau ( - c ) } { \tau ( - q _ { c } ( r ) ) } \geq \frac { \tau ( 0 ) } { \tau ( - q _ { c } ( r ) ) } \geq \exp \left( \frac { q _ { c } ^ { 2 } ( r ) } { 2 } \right) .
$$

This proves the second case.

## F Proof of Lemma 5.3

Proof. Proof of Lemma 5.3. Fix $n < N _ { T }$ and use the abbreviations from the proof of Lemma 5.1. Let $s _ { x } : = \sigma _ { n } ( x )$ $\begin{array} { r } { s _ { g } : = \sigma _ { n } ( g ) , \rho : = \frac { s _ { x } } { s _ { g } } } \end{array}$ . By Lemma $5 . 1 , \rho \ge 1$ . EI optimality and the equality at the quantile minimizer give

$$
s _ { x } \tau \left( \frac { \xi - \mu _ { n } ( x ) } { s _ { x } } \right) = \mathrm { E I } _ { n } ( x ; \xi ) \geq \mathrm { E I } _ { n } ( g ; \xi ) = s _ { g } \tau ( - c ) = s _ { x } \frac { \tau ( - c ) } { \rho } .
$$

By the definition of $q _ { c } ( \rho )$ and the monotonicity of $\tau ,$ we have $\begin{array} { r } { \frac { \xi - \mu _ { n } ( x ) } { s _ { x } } \ge - q _ { c } ( \rho ) } \end{array}$ , or equivalently, $\mu _ { n } ( x ) - \xi \leq q _ { c } ( \rho ) s _ { x }$ On $\mathcal { E } _ { T } , f ( x ) \leq \mu _ { n } ( x ) + \Lambda _ { n } s _ { x }$ . Therefore,

$$
f ( x ) - f ^ { \star } \le f ( x ) - \xi + ( \xi - f ^ { \star } ) _ { + } \le \{ \Lambda _ { n } + q _ { c } ( \rho ) \} s _ { x } + ( \xi - f ^ { \star } ) _ { + } .
$$

This proves the first inequality in Lemma 5.3.

The definition of $\xi$ and the confidence bound at $x ^ { \star }$ give

$$
\xi \leq \mu _ { n } ( x ^ { \star } ) - c \sigma _ { n } ( x ^ { \star } ) \leq f ^ { \star } + ( \Lambda _ { n } - c ) \sigma _ { n } ( x ^ { \star } ) .
$$

Taking positive parts proves the second inequality in Lemma 5.3.

It remains to replace the variance at $x ^ { \star }$ by $s _ { x }$ . We first derive the scalar comparison needed for this step. For notation simplicity, let $\lambda = \Lambda _ { n }$ and we first consider the scenario when $c < \lambda$ . At any point y with $\sigma _ { n } ( y ) > 0$ define $\begin{array} { r } { u : = \frac { \bar { \xi } - \mu _ { n } ( y ) } { \sigma _ { n } ( y ) } , t : = \frac { \xi - f ( y ) } { \sigma _ { n } ( y ) } } \end{array}$ . Global quantile minimality gives u $\leq - c$ , while the confidence inequality gives $| u - t | \leq \lambda .$

We next show that

$$
\begin{array} { r } { \operatorname { E I } _ { n } ( y ; \xi ) \ge K _ { \lambda , c } ( \xi - f ( y ) ) _ { + } , } \end{array}\tag{20}
$$

where $\begin{array} { r } { K _ { \lambda , c } : = \operatorname* { i n f } _ { - \lambda < u \leq - c } \frac { \tau ( u ) } { u + \lambda } } \end{array}$ . If $t \leq 0$ , then $( \xi - f ( y ) ) _ { + } = 0$ , and the claimed inequality follows from the nonnegativity of EI. Suppose now that $t > 0$ . Then $t \leq u + \lambda$ , and hence $- \lambda < u \le - c$ . By the definition of $K _ { \lambda , c } ,$ $\tau ( u ) \geq K _ { \lambda , c } ( u + \lambda ) \geq K _ { \lambda , c } t .$ Multiplying by $\sigma _ { n } ( y )$ gives

$$
\begin{array} { r } { \operatorname { E I } _ { n } ( y ; \xi ) = \sigma _ { n } ( y ) \tau ( u ) \ge K _ { \lambda , c } \sigma _ { n } ( y ) t = K _ { \lambda , c } ( \xi - f ( y ) ) _ { + } . } \end{array}
$$

To calculate this infimum in $K _ { \lambda , c } ,$ let $\begin{array} { r } { R _ { \lambda } ( u ) : = \frac { \tau ( u ) } { u + \lambda } } \end{array}$ . Since $\tau ^ { \prime } ( u ) = \Phi ( u )$ , we have $\begin{array} { r } { R _ { \lambda } ^ { \prime } ( u ) = \frac { \lambda \Phi ( u ) - \phi ( u ) } { ( u + \lambda ) ^ { 2 } } } \end{array}$ . The ratio $\phi ( u ) / \Phi ( u )$ is strictly decreasing from +∞ to zero. Hence, there is a unique point $z _ { \lambda }$ satisfying $\begin{array} { r } { \frac { \phi ( z _ { \lambda } ) } { \Phi ( z _ { \lambda } ) } = \lambda . } \end{array}$ Moreover,

$$
( z _ { \lambda } + \lambda ) \Phi ( z _ { \lambda } ) = z _ { \lambda } \Phi ( z _ { \lambda } ) + \lambda \Phi ( z _ { \lambda } ) = z _ { \lambda } \Phi ( z _ { \lambda } ) + \phi ( z _ { \lambda } ) = \tau ( z _ { \lambda } ) .
$$

Since $\Phi ( z _ { \lambda } ) > 0$ and $\tau ( z _ { \lambda } ) > 0$ , it follows that $z _ { \lambda } + \lambda > 0$ , and hence $z _ { \lambda } > - \lambda$ . Finally,

$$
R _ { \lambda } ^ { \prime } ( u ) = \frac { \Phi ( u ) } { ( u + \lambda ) ^ { 2 } } \left( \lambda - \frac { \phi ( u ) } { \Phi ( u ) } \right) .
$$

Therefore, $R _ { \lambda }$ is decreasing on $( - \lambda , z _ { \lambda } )$ and increasing on $( z _ { \lambda } , \infty )$ . I $\mathrm { ~ f ~ } - c \geq z _ { \lambda }$ , the interval $( - \lambda , - c ]$ contains $z _ { \lambda }$ and $K _ { \lambda , c } = R _ { \lambda } ( z _ { \lambda } ) = \Phi ( z _ { \lambda } ) . \mathrm { I f } - z _ { \lambda } < c < \lambda$ , the interval ends before $z _ { \lambda }$ , and $\begin{array} { r } { K _ { \lambda , c } = R _ { \lambda } ( - c ) = \frac { \tau ( - c ) } { \lambda - c } } \end{array}$ . The two expressions agree at $c = - z _ { \lambda }$

Apply (20) at $y = x ^ { \star }$ with $\lambda = \Lambda _ { n }$ . EI optimality gives

$$
K _ { \Lambda _ { n } , c } ( \boldsymbol { \xi } - \boldsymbol { f } ^ { \star } ) _ { + } \leq \mathrm { E I } _ { n } ( x ^ { \star } ; \boldsymbol { \xi } ) \leq \mathrm { E I } _ { n } ( x ; \boldsymbol { \xi } ) .
$$

Global quantile minimality also gives $\mathrm { E I } _ { n } ( x ; \xi ) \leq s _ { x } \tau ( - c )$ . Therefore,

$$
( \xi - f ^ { \star } ) _ { + } \leq \frac { \tau ( - c ) } { K _ { \Lambda _ { n } , c } } s _ { x } = A ( \Lambda _ { n } , c ) s _ { x } .
$$

We next consider the scenario when $c \geq \Lambda _ { n } ,$ the second inequality in Lemma 5.3 already gives $( \xi - f ^ { \star } ) _ { + } = 0$ which is the first case in the definition of A. This proves the third inequality in Lemma 5.3. Substituting it into the first inequality proves the final certificate in that lemma. The calculation of $K _ { \lambda , c }$ also shows that no smaller coeficient than $A ( \Lambda _ { n } , c )$ exists under the two scalar conditions u $\leq - c$ and $| u - t | \leq \lambda$

## G Proof of Lemma 5.4

Proof. Proof of Lemma 5.4. The upper bound $\sigma _ { n } ^ { 2 } ( x ) \leq 1$ follows from $k _ { \theta } ( x , x ) \leq 1$ and the fact that conditioning cannot increase posterior variance.

For the lower bound, write $k _ { n } ( x ) = \left( k _ { \theta } ( x , x _ { 1 } ) , \ldots , k _ { \theta } ( x , x _ { n } ) \right) ^ { \top }$ . The joint prior covariance of $f ( x )$ and $( f ( x _ { 1 } ) , \ldots , f ( x _ { n } ) )$ is positive semidefinite:

$$
\left( \begin{array} { c c } { k _ { \theta } ( x , x ) } & { k _ { n } ( x ) ^ { \top } } \\ { k _ { n } ( x ) } & { K _ { n } } \end{array} \right) \succeq 0 .
$$

Since $k _ { \theta } ( x , x ) \ge k _ { \mathrm { m i n } } > 0$ , its Schur complement gives $K _ { n } - k _ { n } ( x ) k _ { n } ( x ) ^ { \top } / k _ { \theta } ( x , x ) \succeq 0$ . Also, $\begin{array} { r } { R _ { n } \ \succeq \ \frac { v _ { 0 } } { b _ { T } } I _ { n } } \end{array}$ . Let $r _ { 0 } ^ { 2 } : = v _ { 0 } / b _ { T }$ . Then

$$
K _ { n } + R _ { n } \succeq k _ { n } ( x ) k _ { n } ( x ) ^ { \top } / k _ { \theta } ( x , x ) + r _ { 0 } ^ { 2 } I _ { n } .
$$

Taking inverses reverses the Loewner order. With $u = k _ { n } ( x )$ ，

$$
u ^ { \top } ( K _ { n } + R _ { n } ) ^ { - 1 } u \leq u ^ { \top } ( u u ^ { \top } / k _ { \theta } ( x , x ) + r _ { 0 } ^ { 2 } I _ { n } ) ^ { - 1 } u = \frac { k _ { \theta } ( x , x ) \| u \| ^ { 2 } } { k _ { \theta } ( x , x ) r _ { 0 } ^ { 2 } + \| u \| ^ { 2 } } .
$$

Kernel normalization and Cauchy’s inequality imply $| k _ { \theta } ( x , x _ { i } ) | ^ { 2 } \leq k _ { \theta } ( x , x ) k _ { \theta } ( x _ { i } , x _ { i } ) \leq k _ { \theta } ( x , x ) , \mathrm { s o } \| u \| ^ { 2 } \leq n k _ { \theta } ( x , x )$ Therefore,

$$
\sigma _ { n } ^ { 2 } ( x ) = k _ { \theta } ( x , x ) - u ^ { \top } ( K _ { n } + R _ { n } ) ^ { - 1 } u \ge \frac { k _ { \theta } ( x , x ) ^ { 2 } r _ { 0 } ^ { 2 } } { k _ { \theta } ( x , x ) r _ { 0 } ^ { 2 } + \| u \| ^ { 2 } } \ge \frac { k _ { \theta } ( x , x ) r _ { 0 } ^ { 2 } } { r _ { 0 } ^ { 2 } + n } \ge \frac { k _ { \operatorname* { m i n } } } { 1 + n b _ { T } / v _ { 0 } } .
$$

For $n = 0$ , both bounds follow from $\sigma _ { 0 } ^ { 2 } ( x ) = k _ { \theta } ( x , x )$

Combining the upper bound at $x _ { n + 1 } ^ { \mathbf { c } }$ with the lower bound at $g _ { n + 1 } ^ { \mathbf { c } }$ gives $\begin{array} { r } { \rho _ { n } ^ { \mathbf { c } } \leq \sqrt { \frac { 1 + n b _ { T } / v _ { 0 } } { k _ { \operatorname* { m i n } } } } } \end{array}$ . The lower bound $\rho _ { n } ^ { \mathbf { c } } \geq 1$ is Lemma 5.1. Lemma 5.2 then gives $q _ { c _ { n } } ( \rho _ { n } ^ { \mathbf { c } } ) \leq \bar { q } _ { n } ( c _ { n } )$ in both cases $c _ { n } \geq 0$ and $c _ { n } < 0$ □

## H Proof of Lemma 5.5

Proof. Proof of Lemma 5.5. For brevity, write $s _ { n } : = \sigma _ { n } ( x _ { n + 1 } ^ { \mathbf { c } } )$ . The determinant chain rule for the realized clipped plug-in SK predictor gives

$$
\Gamma _ { N _ { T } } ( X _ { N _ { T } } , R _ { N _ { T } } ) = \frac { 1 } { 2 } \sum _ { n = 0 } ^ { N _ { T } - 1 } \log \left( 1 + \frac { s _ { n } ^ { 2 } } { r _ { n + 1 } ^ { 2 } } \right) .
$$

On $\mathcal { V } _ { T } , r _ { n + 1 } ^ { 2 } \le W _ { T } / b _ { T }$ . Hence,

$$
\log \left( 1 + \frac { s _ { n } ^ { 2 } } { r _ { n + 1 } ^ { 2 } } \right) \geq \log \left( 1 + \frac { b _ { T } } { W _ { T } } s _ { n } ^ { 2 } \right) .
$$

Since $0 \leq s _ { n } ^ { 2 } \leq 1$ , concavity gives

$$
\log \left( 1 + \frac { b _ { T } } { W _ { T } } s _ { n } ^ { 2 } \right) \geq s _ { n } ^ { 2 } \log \left( 1 + \frac { b _ { T } } { W _ { T } } \right) .
$$

It follows that

$$
\Gamma _ { N _ { T } } ( X _ { N _ { T } } , R _ { N _ { T } } ) \geq \frac { 1 } { 2 } \log \left( 1 + \frac { b _ { T } } { W _ { T } } \right) \sum _ { n = 0 } ^ { N _ { T } - 1 } s _ { n } ^ { 2 } .
$$

Also, $R _ { N _ { T } } \mathrm { ~ \textbar { ~ } { ~ } ~ } v _ { 0 } / b _ { T } I _ { N _ { T } }$ . The log-determinant comparison used in the proof of Theorem 4.4 therefore gives $\Gamma _ { N _ { T } } ( X _ { N _ { T } } , R _ { N _ { T } } ) \leq \gamma _ { N _ { T } } ^ { \mathrm { S K } }$ . Combining the last two displays proves

$$
\sum _ { n = 0 } ^ { N _ { T } - 1 } s _ { n } ^ { 2 } \le \frac { 2 \gamma _ { N _ { T } } ^ { \mathrm { S K } } } { \log { ( 1 + b _ { T } / W _ { T } ) } } .
$$

## I Proof of Theorem 5.6

Proof. Proof of Theorem 5.6. We work on $\mathcal { G } _ { T }$ . For brevity, we write $s _ { n } : = \sigma _ { n } ( x _ { n + 1 } ^ { \mathbf { c } } ) , N : = N _ { T }$ and $\lambda _ { T } : = \Lambda _ { N }$ Lemma 5.5 gives $\begin{array} { r } { \sum _ { n = 0 } ^ { N - 1 } s _ { n } ^ { 2 } \le \frac { 2 \gamma _ { N } ^ { \mathrm { S K } } } { \log ( 1 + b _ { T } / W _ { T } ) } } \end{array}$ . Choose $j \in \{ 0 , \dots , N - 1 \}$ such that $s _ { j }$ is no larger than the rootmean-square of these values. Since $N \geq 1$ and $\lfloor x \rfloor \ge x / 2$ for every $x \ge 1$ , applied with $x = T / b _ { T }$ we have, $\begin{array} { r } { N _ { T } = \left| \frac { T } { b _ { T } } \right| \ge \frac { T } { 2 b _ { T } } } \end{array}$ . It follows that

$$
s _ { j } \leq \eta _ { T } \leq 2 \sqrt { \frac { b _ { T } \gamma _ { N } ^ { \mathrm { S K } } } { T \log \left( 1 + b _ { T } / W _ { T } \right) } } .\tag{21}
$$

By Lemmas 5.3 and 5.4 and the definition of $D _ { T } ( \mathbf { c } )$

$$
f ( x _ { j + 1 } ^ { \mathbf { c } } ) - f ^ { \star } \leq D _ { T } ( \mathbf { c } ) s _ { j } \leq D _ { T } ( \mathbf { c } ) \eta _ { T } .\tag{22}
$$

We next control the terminal posterior variance at every sampled location. When the qth point is added, the one-step posterior variance update at its own location gives

$$
\sigma _ { q } ^ { 2 } ( x _ { q } ^ { \mathbf { c } } ) = \sigma _ { q - 1 } ^ { 2 } ( x _ { q } ^ { \mathbf { c } } ) - \frac { \sigma _ { q - 1 } ^ { 4 } ( x _ { q } ^ { \mathbf { c } } ) } { \sigma _ { q - 1 } ^ { 2 } ( x _ { q } ^ { \mathbf { c } } ) + r _ { q } ^ { 2 } } = \frac { \sigma _ { q - 1 } ^ { 2 } ( x _ { q } ^ { \mathbf { c } } ) r _ { q } ^ { 2 } } { \sigma _ { q - 1 } ^ { 2 } ( x _ { q } ^ { \mathbf { c } } ) + r _ { q } ^ { 2 } } .
$$

Lemma 5.4 gives $\sigma _ { q - 1 } ^ { 2 } ( x _ { q } ^ { \mathbf { c } } ) \leq 1$ . Since $\mathcal { G } _ { T } \subseteq \mathcal { V } _ { T } , r _ { q } ^ { 2 } \leq W _ { T } / b _ { T }$ , and therefore $\begin{array} { r } { \sigma _ { q } ^ { 2 } ( x _ { q } ^ { \mathbf { c } } ) \leq \frac { W _ { T } } { b _ { T } + W _ { T } } = \omega _ { T } ^ { 2 } } \end{array}$

For the fixed kernel and the realized sequence of appended positive noise scales, posterior variance is nonincreasing as further observations are added. Hence,

$$
\sigma _ { N } ( x _ { q } ^ { \mathbf { c } } ) \leq \omega _ { T } , \qquad q = 1 , \dots , N .\tag{23}
$$

For $x _ { j + 1 } ^ { \mathbf { c } }$ selected in (21), posterior-variance monotonicity also gives $\sigma _ { N } ( x _ { j + 1 } ^ { \mathbf { c } } ) \leq \sigma _ { j } ( x _ { j + 1 } ^ { \mathbf { c } } ) = s _ { j } \leq \eta \tau$ <sub>T</sub>. Combining the last inequality with (23) yields

$$
\sigma _ { N } ( x _ { j + 1 } ^ { \mathbf { c } } ) \leq \zeta _ { T } .\tag{24}
$$

Fix any $d _ { T } \in \mathbb { R }$ and abbreviate $\hat { x } : = \hat { x } _ { T } ^ { \mathbf { c } , d _ { T } }$ . Since xˆ minimizes $\mu _ { N } ( z ) + d _ { T } \sigma _ { N } ( z )$ over the sampled set and $x _ { j + 1 } ^ { \mathbf { c } }$ belongs to that set, the confidence inequalities on $\mathcal { G } _ { T } \subseteq \mathcal { E } _ { T }$ give

$$
\begin{array} { r l } & { f ( \hat { x } ) - f ^ { \star } \leq \mu _ { N } ( \hat { x } ) + \lambda _ { T } \sigma _ { N } ( \hat { x } ) - f ^ { \star } } \\ & { \qquad = \{ \mu _ { N } ( \hat { x } ) + d _ { T } \sigma _ { N } ( \hat { x } ) \} + ( \lambda _ { T } - d _ { T } ) \sigma _ { N } ( \hat { x } ) - f ^ { \star } } \\ & { \qquad \leq \mu _ { N } ( x _ { j + 1 } ^ { \mathbf { c } } ) + d _ { T } \sigma _ { N } ( x _ { j + 1 } ^ { \mathbf { c } } ) + ( \lambda _ { T } - d _ { T } ) \sigma _ { N } ( \hat { x } ) - f ^ { \star } } \\ & { \qquad \leq f ( x _ { j + 1 } ^ { \mathbf { c } } ) - f ^ { \star } + ( \lambda _ { T } + d _ { T } ) \sigma _ { N } ( x _ { j + 1 } ^ { \mathbf { c } } ) + ( \lambda _ { T } - d _ { T } ) \sigma _ { N } ( \hat { x } ) } \\ & { \qquad \leq f ( x _ { j + 1 } ^ { \mathbf { c } } ) - f ^ { \star } + ( \lambda _ { T } + d _ { T } ) _ { + } \sigma _ { N } ( x _ { j + 1 } ^ { \mathbf { c } } ) + ( \lambda _ { T } - d _ { T } ) _ { + } \sigma _ { N } ( \hat { x } ) . } \end{array}
$$

Because $\hat { x }$ is sampled, (23) gives $\sigma _ { N } ( \hat { x } ) \leq \omega _ { T }$ . Together with (22) and (24), this proves

$$
f ( \hat { x } _ { T } ^ { \mathbf { c } , d _ { T } } ) - f ^ { \star } \leq D _ { T } ( \mathbf { c } ) \eta _ { T } + ( \lambda _ { T } + d _ { T } ) _ { + } \zeta _ { T } + ( \lambda _ { T } - d _ { T } ) _ { + } \omega _ { T } .\tag{25}
$$

When $d _ { T } = \lambda _ { T } = \Lambda _ { N _ { T } }$ T

$$
f ( \hat { x } _ { T } ^ { \mathbf { c } , \Lambda _ { N _ { T } } } ) - f ^ { \star } \le D _ { T } ( \mathbf { c } ) \eta _ { T } + 2 \Lambda _ { N _ { T } } \zeta _ { T } \le \{ D _ { T } ( \mathbf { c } ) + 2 \Lambda _ { N _ { T } } \} \eta _ { T } .
$$

Using (21), we obtain

$$
f ( \hat { x } _ { T } ^ { \mathbf { c } , \Lambda _ { N _ { T } } } ) - f ^ { \star } \leq 2 \left\{ 2 \Lambda _ { N _ { T } } + D _ { T } ( \mathbf { c } ) \right\} \sqrt { \frac { b _ { T } \gamma _ { N _ { T } } ^ { \mathrm { S K } } } { T \log ( 1 + b _ { T } / W _ { T } ) } } .
$$

Since $\mathbb { P } ( \mathcal { G } _ { T } ) \ge 1 - \delta _ { : }$ , all stated conclusions follow.

## J Proof of Corollary 6.2

Proof. Proof of Corollary 6.2. Set $c _ { n } = \Lambda _ { n }$ and $d _ { T } = \Lambda _ { N _ { T } }$ for every $n < N _ { T }$ . Then $A ( \Lambda _ { n } , c _ { n } ) = 0$ and, because $\Lambda _ { n }$ and $\ell _ { n }$ are nondecreasing,

$$
D _ { T } ( \mathbf { c } ) = \operatorname* { m a x } _ { n < N _ { T } } \Big \{ \Lambda _ { n } + \sqrt { \Lambda _ { n } ^ { 2 } + \ell _ { n } } \Big \} \le \Lambda _ { N _ { T } } + \sqrt { \Lambda _ { N _ { T } } ^ { 2 } + \log \left( \frac { 1 + T / v _ { 0 } } { k _ { \operatorname* { m i n } } } \right) } .
$$

Let

$$
J _ { T } : = \sqrt { \Lambda _ { N _ { T } } ^ { 2 } + \log \left( \frac { 1 + T / v _ { 0 } } { k _ { \mathrm { m i n } } } \right) } .
$$

Since $\Lambda _ { N _ { T } } \leq J _ { T }$

$$
2 \left\{ 2 \Lambda _ { N _ { T } } + D _ { T } ( \mathbf { c } ) \right\} \leq 2 ( 3 \Lambda _ { N _ { T } } + J _ { T } ) \leq 8 J _ { T } .
$$

Substitution into Theorem 5.6 gives the claimed bound.

## K Proof of Lemma 6.1

Proof. Proof of Lemma 6.1. For ease of exposition, we write $\lambda : = \Lambda _ { n } > 0$ in this proof. We first show that, $\forall c \in \mathbb { R }$ $A ( \lambda , c ) \geq ( \lambda - c ) _ { + }$ . This holds with equality when $c \geq \lambda$ or $c _ { \lambda } ^ { \circ } < c < \lambda$ by the definition of A. Suppose that $c \leq c _ { \lambda } ^ { \circ } = - z _ { \lambda }$ . Since $\tau ^ { \prime } ( u ) = \Phi ( u )$ and $\tau ^ { \prime \prime } ( u ) = \phi ( u ) > 0$ , convexity of $\tau$ function and the definition of $z _ { \lambda }$ give

$$
\tau ( - c ) \geq \tau ( z _ { \lambda } ) + \Phi ( z _ { \lambda } ) ( - c - z _ { \lambda } ) = ( \lambda - c ) \Phi ( z _ { \lambda } ) ,
$$

where the equality uses $\phi ( z _ { \lambda } ) = \lambda \Phi ( z _ { \lambda } )$ and hence $\tau ( z _ { \lambda } ) = ( z _ { \lambda } + \lambda ) \Phi ( z _ { \lambda } )$ . The latter identity also implies $z _ { \lambda } + \lambda > 0$ so $c _ { \lambda } ^ { \circ } < \lambda$ . Dividing by $\Phi ( z _ { \lambda } ) > 0$ yields

$$
A ( \lambda , c ) = { \frac { \tau ( - c ) } { \Phi ( z _ { \lambda } ) } } \geq \lambda - c = ( \lambda - c ) _ { + } .
$$

Next, the definition of $\bar { q } _ { n }$ can be written as $\bar { q } _ { n } ( c ) = \sqrt { ( c _ { + } ) ^ { 2 } + \ell _ { n } } = \left\| ( c _ { + } , \sqrt { \ell _ { n } } ) \right\| _ { 2 }$ , where $\ell _ { n } \geq 0$ . Thus $\bar { q } _ { n }$ is nondecreasing. The triangle inequality gives, for any $c _ { 1 } , c _ { 2 } \in \mathbb { R }$

$$
| \bar { q } _ { n } ( c _ { 2 } ) - \bar { q } _ { n } ( c _ { 1 } ) | \leq | ( c _ { 2 } ) _ { + } - ( c _ { 1 } ) _ { + } | \leq | c _ { 2 } - c _ { 1 } | .
$$

Therefore, $\bar { q } _ { n }$ is 1-Lipschitz.

Thus, If $c < \lambda$ , we have $\bar { q } _ { n } ( \lambda ) - \bar { q } _ { n } ( c ) \leq \lambda - c$ . Combining the bound for A, we have

$$
\bar { q } _ { n } ( c ) + A ( \lambda , c ) \geq \bar { q } _ { n } ( c ) + \lambda - c \geq \bar { q } _ { n } ( \lambda ) .
$$

If $c \geq \lambda$ , then $A ( \lambda , c ) = 0$ , and monotonicity gives the same inequality.

## L Proofs for the Replication-Number Analysis

The replication analysis below uses the calibrated consequence of Theorem 5.6 obtained by setting $d _ { T } = \Lambda _ { N _ { T } }$ . This gives exactly the bound in Corollary 6.2 and hence the same replication objective. We first define some notations used in the replication-number analysis:

$$
F _ { T } ( b ) : = \frac { N _ { T } ( b ) } { 2 } \log \left( 1 + \frac { b } { v _ { 0 } } \right) .
$$

$$
\overline { { \Lambda } } _ { T } ^ { k } ( b ) : = B _ { f } + \sqrt { 2 \kappa _ { T } ( b ) \left( \overline { { \gamma } } _ { T } ^ { k } ( b ) + \log \frac { 2 } { \delta } \right) } .
$$

## L.1 Proof of the Finite-Batch Information-Gain Bound

Proof. Proof of Lemma 6.3. First fix $1 \leq m \leq N _ { T } ( b )$ and any collection of points $A = ( a _ { 1 } , \ldots , a _ { m } ) \in { \mathcal { X } } ^ { m }$ . Let $\lambda _ { 1 } , \ldots , \lambda _ { m }$ be the nonnegative eigenvalues of the positive semidefinite matrix $K _ { A }$ . Then

$$
{ \frac { 1 } { 2 } } \log \operatorname* { d e t } \left( I _ { m } + { \frac { b } { v _ { 0 } } } K _ { A } \right) = { \frac { 1 } { 2 } } \sum _ { i = 1 } ^ { m } \log \left( 1 + { \frac { b } { v _ { 0 } } } \lambda _ { i } \right) .
$$

The function $\begin{array} { r } { z \longmapsto \log \left( 1 + \frac { b } { v _ { 0 } } z \right) } \end{array}$ is concave on $[ 0 , \infty )$ . Jensen’s inequality therefore gives

$$
\frac { 1 } { 2 } \sum _ { i = 1 } ^ { m } \log \left( 1 + \frac { b } { v _ { 0 } } \lambda _ { i } \right) \leq \frac { m } { 2 } \log \left( 1 + \frac { b } { v _ { 0 } } \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \lambda _ { i } \right) .
$$

The sum of the eigenvalues equals the trace:

$$
\sum _ { i = 1 } ^ { m } \lambda _ { i } = \operatorname { t r } ( K _ { A } ) = \sum _ { i = 1 } ^ { m } k _ { \theta } ( a _ { i } , a _ { i } ) .
$$

Since the kernel is normalized so that $k _ { \theta } ( x , x ) \leq 1$ for every $x \in \mathcal { X } , \operatorname { t r } ( K _ { A } ) \leq m$ . Consequently,

$$
\frac { 1 } { 2 } \log \operatorname* { d e t } \left( I _ { m } + \frac { b } { v _ { 0 } } K _ { A } \right) \leq \frac { m } { 2 } \log \left( 1 + \frac { b } { v _ { 0 } } \right) \leq \frac { N _ { T } ( b ) } { 2 } \log \left( 1 + \frac { b } { v _ { 0 } } \right) .
$$

The last expression does not depend on the points in A or on $m \le N _ { T } ( b )$ . Taking the supremum over all admissible m and A proves

$$
\gamma _ { T } ( b ) \leq \frac { N _ { T } ( b ) } { 2 } \log \left( 1 + \frac { b } { v _ { 0 } } \right) .
$$

## L.2 Proof of the General-Kernel Replication Theorem

Proof. Proof of Theorem 6.5. Fix $\delta \in ( 0 , 1 )$ and write $\begin{array} { r } { g _ { T } : = G _ { k } \left( \frac { T } { v _ { 0 } } \right) } \end{array}$ . For every $b \in \mathcal { B } _ { T } , T - b < b N _ { T } ( b ) \leq T$ . Since $b \le T / 2 , \ { \frac { T } { 2 } } \le b N _ { T } ( b ) \le T$ . Therefore, $\begin{array} { r } { \frac { T } { 2 v _ { 0 } } \leq U _ { T } ( b ) \leq \frac { T } { v _ { 0 } } } \end{array}$ . As $G _ { k }$ is nondecreasing, $\begin{array} { r } { G _ { k } \left( \frac { T } { 2 v _ { 0 } } \right) \le G _ { k } \bigl ( U _ { T } ( b ) \bigr ) \le g _ { T } } \end{array}$ The condition $G _ { k } ( 2 u ) \leq C _ { k } G _ { k } ( u )$ gives, for all suficiently large T, $\begin{array} { r } { g _ { T } \leq C _ { k } G _ { k } \left( \frac { T } { 2 v _ { 0 } } \right) } \end{array}$ . Hence,

$$
G _ { k } \big ( U _ { T } ( b ) \big ) = \Theta ( g _ { T } )\tag{26}
$$

uniformly over $b \in B _ { T }$

We first evaluate the bound at a logarithmic replication number. Fix any constant $c > 0$ and define $b _ { 0 } : =$ ⌈c log T⌉. For all suficiently large $T , b _ { 0 } \in B _ { T }$ . Moreover, $\begin{array} { r } { N _ { T } ( b _ { 0 } ) = \Theta \left( \frac { T } { \log T } \right) } \end{array}$ , and

$$
F _ { T } ( b _ { 0 } ) = \frac { N _ { T } ( b _ { 0 } ) } { 2 } \log \left( 1 + \frac { b _ { 0 } } { v _ { 0 } } \right) = \Theta \left( \frac { T \log \log T } { \log T } \right) .
$$

Moreover, denote $L _ { b }$ as $\log ( 4 N _ { T } ( b ) / \delta )$ . By the Laurent–Massart upper-tail inequality (Laurent and Massart 2000, eq. (4.3)) and the definition of $h _ { T } ( b )$ , we have

$$
1 \leq h _ { T } ( b ) \leq 1 + 2 \sqrt { \frac { L _ { b } } { \nu _ { b } } } + 2 \frac { L _ { b } } { \nu _ { b } } .
$$

Thus, $h _ { T } ( b _ { 0 } ) = O ( 1 ) , W _ { T } ( b _ { 0 } ) = \Theta ( 1 )$ , and l $\mathrm { { y g } } ( 1 + { b _ { 0 } } / { W _ { T } ( b _ { 0 } ) } ) = \Theta ( \log \log T )$

By assumption, $g _ { T } = o \left( { \sqrt { T \log \log T } } \right)$ . Since

$$
\sqrt { T \log \log T } = o \left( \frac { T \log \log T } { \log T } \right) ,
$$

we have $g _ { T } = o \left( F _ { T } ( b _ { 0 } ) \right)$ ). Together with (26), this gives

$$
\begin{array} { r } { \overline { { \gamma } } _ { T } ^ { k } ( b _ { 0 } ) = G _ { k } \big ( U _ { T } ( b _ { 0 } ) \big ) = \Theta ( g _ { T } ) . } \end{array}
$$

Furthermore, log $\begin{array} { r } { \frac { 3 N _ { T } ( b _ { 0 } ) } { \delta } = \Theta ( \log T ) } \end{array}$ . Since $b _ { 0 } = \Theta ( \log T )$ , Lemma 4.3 gives $\kappa _ { T } ( b _ { 0 } ) = { \cal O } ( 1 )$ . The assumption log $T = { \cal { O } } ( g _ { T } )$ then implies

$$
\left( \overline { { \Lambda } } _ { T } ^ { k } ( b _ { 0 } ) \right) ^ { 2 } + \log \left( \frac { 1 + T / v _ { 0 } } { k _ { \mathrm { m i n } } } \right) = O ( g _ { T } ) .
$$

Therefore,

$$
\left( \overline { { \mathcal { R } } } _ { T } ^ { k } ( b _ { 0 } ) \right) ^ { 2 } = O \left( \frac { g _ { T } ^ { 2 } \log T } { T \log \log T } \right) .
$$

Since $g _ { T } ^ { 2 } = o ( T$ log log T), we obtain

$$
\left( \overline { { \mathcal { R } } } _ { T } ^ { k } ( b _ { 0 } ) \right) ^ { 2 } = o ( \log T ) .\tag{27}
$$

We next show that a minimizer cannot lie in the range where the finite-batch bound is active, $\mathrm { i } , \mathrm { e } . , F _ { T } ( b ) \leq$ $G _ { k } ( U _ { T } ( b ) )$ and thus $\overline { { \gamma } } _ { T } ^ { k } ( b ) = F _ { T } ( b )$ . Suppose that, for some $b \in \mathcal { B } _ { T } , \overline { \gamma } _ { T } ^ { k } ( b ) = F _ { T } ( b )$ . Then

$$
\frac { b \overline { { \gamma } } _ { T } ^ { k } ( b ) } { T \log ( 1 + b / W _ { T } ( b ) ) } = \frac { b N _ { T } ( b ) } { 2 T } \frac { \log ( 1 + b / v _ { 0 } ) } { \log ( 1 + b / W _ { T } ( b ) ) } .
$$

Because $\begin{array} { r } { b N _ { T } ( b ) \ge \frac { T } { 2 } } \end{array}$ and $W _ { T } ( b ) \geq M \geq v _ { 0 }$ , we have

$$
\frac { b \overline { { \gamma } } _ { T } ^ { k } ( b ) } { T \log ( 1 + b / W _ { T } ( b ) ) } \geq \frac { 1 } { 4 } .
$$

Also,

$$
\left( \overline { { \Lambda } } _ { T } ^ { k } ( b ) \right) ^ { 2 } + \log \left( \frac { 1 + T / v _ { 0 } } { k _ { \operatorname* { m i n } } } \right) \geq \log \left( \frac { 1 + T / v _ { 0 } } { k _ { \operatorname* { m i n } } } \right) .
$$

It follows that

$$
\left( \overline { { \mathcal { R } } } _ { T } ^ { k } ( b ) \right) ^ { 2 } \geq 1 6 \log \left( \frac { 1 + T / v _ { 0 } } { k _ { \mathrm { m i n } } } \right) = \Omega ( \log T ) .\tag{28}
$$

Comparison of (27) and (28) shows that, for all suficiently large $T ,$ no minimizer can lie in the range where the finite-batch bound is active.

It remains to minimize the bound over replication numbers for which $\overline { { { \gamma } } } _ { T } ^ { k } ( b ) = G _ { k } \big ( U _ { T } ( b ) \big )$ . By $( 2 6 ) , \overline { { \gamma } } _ { T } ^ { k } ( b ) =$ $\Theta ( g _ { T } )$ uniformly over this range. Since $\kappa _ { T } ( b ) \geq 1$ , log $T = { \cal { O } } ( g _ { T } )$ , we have

$$
\left( \overline { { \Lambda } } _ { T } ^ { k } ( b ) \right) ^ { 2 } + \log \left( \frac { 1 + T / v _ { 0 } } { k _ { \mathrm { m i n } } } \right) = \Theta \left( \kappa _ { T } ( b ) g _ { T } \right) .
$$

Consequently,

$$
\left( \overline { { \mathcal { R } } } _ { T } ^ { k } ( b ) \right) ^ { 2 } = \Theta \left( \frac { g _ { T } ^ { 2 } } { T } H _ { T } ( b ) \right) ,\tag{29}
$$

where

$$
H _ { T } ( b ) : = \frac { \kappa _ { T } ( b ) b } { \log ( 1 + b / W _ { T } ( b ) ) } .
$$

Since $\begin{array} { r } { h _ { T } ( b ) \geq 1 , H _ { T } ( b ) \geq \underline { { H _ { T } ( b ) } } : = \frac { \kappa _ { T } ( b ) b } { \log ( 1 + b / M ) } } \end{array}$ . For $b = \Theta ( \log T )$ , both $\kappa _ { T } ( b )$ and $h _ { T } ( b )$ are bounded above by constants and below by one. Thus,

$$
H _ { T } ( b ) = \Theta \left( { \frac { \log T } { \log \log T } } \right) .\tag{30}
$$

We now consider replication numbers smaller than logarithmic order. Suppose that $b = o ( \log T )$ . Recall that $\nu _ { b } = b - 1$ , and $\begin{array} { r } { L _ { b \mathrm { ~ } } = \log \frac { 4 N _ { T } ( b ) } { \delta } } \end{array}$ . Since δ is fixed and $\begin{array} { r } { b \ = \ o ( \log T ) , \ L _ { b \ } = \Theta ( \log T ) , \ \frac { L _ { b \ } } { \nu _ { b \ } } \ \longrightarrow \ \infty } \end{array}$ . According to Lemma 4.3, $\begin{array} { r } { \kappa _ { T } ( b ) \geq c _ { 0 } ^ { 2 } \exp \left( \frac { 2 L _ { b } } { \nu _ { b } } \right) \longrightarrow \infty } \end{array}$

Since $b = o ( \log T )$ , we also have $b \leq L _ { b }$ for all suficiently large $T ,$ and therefore log $\begin{array} { r } { \left( 1 + \frac { b } { M } \right) = O ( \log L _ { b } ) } \end{array}$ . Thus,

$$
\frac { H _ { T } ( b ) } { L _ { b } / \log L _ { b } } \geq c _ { 1 } \exp ( \frac { 2 L _ { b } } { \nu _ { b } } ) \frac { b } { L _ { b } } \frac { \log L _ { b } } { \log ( 1 + b / M ) } \geq c _ { 2 } \frac { \exp ( \frac { 2 L _ { b } } { \nu _ { b } } ) } { \frac { 2 L _ { b } } { \nu _ { b } } } \longrightarrow \infty .
$$

Since $L _ { b } = \Theta ( \log T )$ , this gives

$$
\underline { { H } } _ { T } ( b ) \gg \frac { \log T } { \log \log T } .\tag{31}
$$

Thus, a replication number smaller than logarithmic order gives a larger bound than (30).

Finally, suppose that $\frac { b } { \log T } \longrightarrow \infty$ . Since $\begin{array} { r } { \kappa _ { T } ( b ) \geq 1 , \underline { { H _ { T } ( b ) } } \geq \frac { \bar { b } } { \log \left( 1 + b / M \right) } } \end{array}$ . Let $s _ { T } : = { \frac { b } { \log T } }$ . Then $s _ { T } \to \infty$ , and

$$
\frac { H _ { T } ( b ) } { \log T / \log \log T } \geq \frac { s _ { T } \log \log T } { \log \left( 1 + s _ { T } \log T / M \right) } .
$$

The denominator satisfies

$$
\log \left( 1 + s _ { T } \log T / M \right) \leq C + \log s _ { T } + \log \log T
$$

for a constant C. Therefore,

$$
\frac { s _ { T } \log \log T } { C + \log s _ { T } + \log \log T } \longrightarrow \infty ,
$$

and hence

$$
\underline { { H } } _ { T } ( b ) \gg \frac { \log T } { \log \log T } .\tag{32}
$$

Thus, a replication number larger than logarithmic order also gives a larger bound than (30). Since $H _ { T } ( b ) \geq \underline { { H } } _ { T } ( b )$ these lower bounds also apply to $H _ { T } ( b )$ . Therefore, neither sublogarithmic nor superlogarithmic replication can minimize the bound.

To summarize, we have shown that a minimizing sequence cannot satisfy $b _ { T } ^ { \mathrm { b d } } = o ( \log T ) \ \mathrm { o r } \ \frac { b _ { T } ^ { \mathrm { b d } } } { \log T } \ \longrightarrow \ \infty$ . We finally verify that the preceding comparisons cover arbitrary minimizing sequences, including those for which the ratio to log T has no limit. Let $\begin{array} { r } { r _ { T } : = \frac { b _ { T } ^ { \mathrm { b d } } } { \log T } } \end{array}$ . If $b _ { T } ^ { \mathrm { b d } }$ were not of order log T, then either lim in $\dot { \mathbf { \rho } } _ { T  \infty } r _ { T } = 0$ or lim su $) _ { T  \infty } r _ { T } = \infty$ . In the first case, there exists a subsequence $T _ { j } \to \infty$ such that $r _ { T _ { j } }  0$ . Equations (29), (30), and (31) then imply that the logarithmic candidate $b _ { 0 } ( T _ { j } )$ gives a strictly smaller bound than $b _ { T _ { j } } ^ { \mathrm { b d } }$ for all suficiently large j. This contradicts the definition of $b _ { T _ { j } } ^ { \mathrm { b d } }$ . The same contradiction follows from (32) if a subsequence satisfies $r _ { T _ { j } }  \infty$ . Therefore, we have

$$
0 < \operatorname* { l i m } _ { T \to \infty } \operatorname* { i n f } _ { T \to \infty } r _ { T } < \operatorname* { l i m } _ { T \to \infty } r _ { T } < \infty ,
$$

which is equivalent to $b _ { T } ^ { \mathrm { b d } } = \Theta ( \log T )$

## M Kernel Classes Satisfying the Conditions of Theorem 6.5

This section verifies the three conditions in Theorem 6.5 for several commonly used kernel classes. Recall that $\begin{array} { r } { g _ { T } = G _ { k } \left( \frac { T } { v _ { 0 } } \right) } \end{array}$ , where $G _ { k }$ is a deterministic spectral upper bound on the maximum information gain. We verify the conditions below.

## M.1 Finite-rank kernels

Suppose that the kernel has finite rank r. Taking $\textit { D } = \textit { r }$ in Theorem 3 of Vakili et al. (2021b) removes the spectral-tail term. Under the normalization $k _ { \theta } ( x , x ) \leq 1$ , we may use

$$
G _ { k } ( u ) = \frac { r } { 2 } \log \left( 1 + \frac { u } { r } \right) .
$$

For fixed $r ,$ this function is nondecreasing and

$$
\frac { G _ { k } ( 2 u ) } { G _ { k } ( u ) } = \frac { \log ( 1 + 2 u / r ) } { \log ( 1 + u / r ) } = O ( 1 ) .
$$

Thus, the first condition holds.

Moreover, $g _ { T } = \Theta ( \log T )$ , because $v _ { 0 }$ is fixed. Therefore, log $T = { \cal { O } } ( g _ { T } )$ , so the second condition holds. Finally,

$$
{ \frac { g _ { T } } { \sqrt { T \log \log T } } } = O \left( { \frac { \log T } { \sqrt { T \log \log T } } } \right) \longrightarrow 0 .
$$

Hence, the third condition also holds.

This class includes linear kernels and fixed-degree polynomial kernels on bounded finite-dimensional domains.

## M.2 Kernels with exponential eigenvalue decay

The eigenvalues of this type of kernel satisfy $\lambda _ { j } \le C _ { 1 } \exp \left( - C _ { 2 } j ^ { \beta _ { e } } \right)$ for some constants $C _ { 1 } , C _ { 2 } > 0$ and $\beta _ { e } > 0$ Under the regularity conditions of Vakili et al. (2021b), the corresponding information-gain bound can be written as

$$
G _ { k } ( u ) = C _ { k } \log ( 1 + u ) [ 1 + \log ( 1 + u ) ] ^ { 1 / \beta _ { e } } .\tag{33}
$$

The function in (33) is nondecreasing. Also, for suficiently large $u ,$

$$
\frac { G _ { k } ( 2 u ) } { G _ { k } ( u ) } = \left[ \frac { \log ( 1 + 2 u ) } { \log ( 1 + u ) } \right] ^ { 1 + 1 / \beta _ { e } } \longrightarrow 1 .
$$

Thus, the first condition of Theorem 6.5 holds.

For fixed $v _ { 0 } .$ ,

$$
g _ { T } = G _ { k } \left( { \frac { T } { v _ { 0 } } } \right) = \Theta \left( ( \log T ) ^ { 1 + 1 / \beta _ { e } } \right) .
$$

Since $1 + 1 / \beta _ { e } > 1$ , log $T = { \cal { O } } ( g _ { T } )$ , which verifies the second condition.

For the third condition,

$$
\frac { g _ { T } } { \sqrt { T \log \log T } } = O \left( \frac { ( \log T ) ^ { 1 + 1 / \beta _ { e } } } { \sqrt { T \log \log T } } \right) \longrightarrow 0 .
$$

Hence, this class of kernel satisfies all three conditions of Theorem 6.5.

The SE kernel on a compact subset of $\mathbb { R } ^ { d }$ is an important example. Its eigenvalue decay leads to

$$
G _ { \mathrm { S E } } ( u ) = O \left( [ \log ( 1 + u ) ] ^ { d + 1 } \right) .
$$

## M.3 Kernels with polynomial eigenvalue decay

The eigenvalues of this type of kernel satisfy $\lambda _ { j } \le C _ { 1 } j ^ { - \beta _ { p } }$ for some $\beta _ { p } > 1$ . Under the regularity conditions in Vakili et al. (2021b),

$$
G _ { k } ( u ) = C _ { k } u ^ { 1 / \beta _ { p } } \left[ \log ( 1 + u ) \right] ^ { 1 - 1 / \beta _ { p } } .\tag{34}
$$

The function in (34) is nondecreasing. Moreover,

$$
\frac { G _ { k } ( 2 u ) } { G _ { k } ( u ) } = 2 ^ { 1 / \beta _ { p } } \left[ \frac { \log ( 1 + 2 u ) } { \log ( 1 + u ) } \right] ^ { 1 - 1 / \beta _ { p } } \longrightarrow 2 ^ { 1 / \beta _ { p } } .
$$

Thus, the first condition holds.

For fixed v<sub>0</sub>,

$$
g _ { T } = \Theta \left( T ^ { 1 / \beta _ { p } } ( \log T ) ^ { 1 - 1 / \beta _ { p } } \right) .
$$

Since $\beta _ { p } > 1$ , the polynomial term dominates log $T ,$ and hence log $T = O ( g _ { T } )$ . Therefore, the second condition holds.

For the third condition,

$$
\frac { g _ { T } } { \sqrt { T \log \log T } } = \Theta \left( T ^ { 1 / \beta _ { p } - 1 / 2 } \frac { ( \log T ) ^ { 1 - 1 / \beta _ { p } } } { \sqrt { \log \log T } } \right) .
$$

This quantity converges to zero whenever $\begin{array} { r } { \frac { 1 } { \beta _ { p } } < \frac { 1 } { 2 } } \end{array}$ , or equivalently, $\beta _ { p } > 2$ . Thus, kernels with polynomial eigenvalue decay satisfying $\beta _ { p } > 2$ meet all three conditions of Theorem 6.5.

A Matérn kernel with smoothness parameter ν in dimension d has polynomial eigenvalue decay with $\begin{array} { r } { \beta _ { p } = \frac { 2 \nu + d } { d } } \end{array}$ Therefore, $\beta _ { p } > 2$ is equivalent to $\begin{array} { r } { \nu > \frac { d } { 2 } . } \end{array}$ . In this case,

$$
G _ { \mathrm { M a t } } ( u ) = O \left( u ^ { \frac { d } { 2 \nu + d } } [ \log ( 1 + u ) ] ^ { \frac { 2 \nu } { 2 \nu + d } } \right) ,
$$

and all three conditions of Theorem 6.5 are satisfied.

## N Derivation of the Rate after Choosing the Replication Number and Kernels

Under the conditions of Theorem 6.5, choose $b _ { T } = \Theta ( \log T )$ . Then $\begin{array} { r } { N _ { T } = \Theta \left( \frac { T } { \log T } \right) } \end{array}$ and, by Lemma $4 . 3 , \kappa _ { T } = { \cal O } ( 1 )$ and $h _ { T } = O ( 1 )$ . Since M is fixed and finite, $W _ { T } = \Theta ( 1 )$ , and $\log ( 1 + \dot { b _ { T } } / W _ { T } ^ { ' } ) = \Theta ( \log \log T )$ . The proof of Theorem 6.5 shows that $\overline { { \gamma } } _ { T } ^ { k } ( b _ { T } ) = \Theta ( g _ { T } )$ . Since log $T = { \cal { O } } ( g _ { T } )$ , the definition of the confidence coeficient gives

$$
\left( \overline { { \Lambda } } _ { T } ^ { k } ( b _ { T } ) \right) ^ { 2 } + \log \left( \frac { 1 + T / v _ { 0 } } { k _ { \mathrm { m i n } } } \right) = O ( g _ { T } ) .
$$

Substitution into Corollary 6.2 yields

$$
f ( \hat { x } _ { T } ) - f ^ { \star } = O \left( \sqrt { \frac { g _ { T } \log T g _ { T } } { T \log \log T } } \right) = O \left( g _ { T } \sqrt { \frac { \log T } { T \log \log T } } \right) .
$$

For a SE kernel,

$$
g _ { T } = O \left( ( \log T ) ^ { d + 1 } \right) .
$$

Therefore,

$$
f ( \hat { x } _ { T } ) - f ^ { \star } = O \left( ( \log T ) ^ { d + 1 } \sqrt { \frac { \log T } { T \log \log T } } \right) = O \left( \frac { ( \log T ) ^ { d + 3 / 2 } } { \sqrt { T \log \log T } } \right) .
$$

For a Matérn kernel with smoothness parameter $\nu ,$

$$
g _ { T } = O \left( T ^ { \frac { d } { 2 \nu + d } } ( \log T ) ^ { \frac { 2 \nu } { 2 \nu + d } } \right) .
$$

Consequently,

$$
\begin{array} { r } { f ( \widehat { x } _ { T } ) - f ^ { \star } = \widetilde { O } \left( T ^ { \frac { d } { 2 \nu + d } } T ^ { - 1 / 2 } \right) = \widetilde { O } \left( T ^ { - \frac { 2 \nu - d } { 2 ( 2 \nu + d ) } } \right) . } \end{array}
$$

This rate decreases polynomially when $\begin{array} { r } { \nu > \frac { d } { 2 } } \end{array}$

## O Experiment Settings and Additional Results

## O.1 Objective Functions and Experimental Settings

This subsection provides the complete construction and implementation details for the controlled RKHS experiments in Section 7.1.

For each dimension $d \in \{ 2 , 4 \}$ , the decision space is $\mathcal { X } _ { d } = [ 0 , 1 ] ^ { d }$ . We consider a $\mathrm { S E }$ kernel and a Matérn-5/2 kernel, both with isotropic lengthscale $\ell = 0 . 2$ . For each combination of dimension and kernel, we construct one fixed objective as the finite kernel expansion $\begin{array} { r } { f _ { d , k } ( x ) = \sum _ { j = 1 } ^ { 1 0 0 } w _ { d , j } k ( x , z _ { d , j } ) } \end{array}$ . Consequently, $f _ { d , k }$ belongs exactly to the RKHS induced by k. Within each dimension, the two kernels use the same 100 basis points and standardnormal weights. Separate basis points and weights are generated for the two- and four-dimensional experiments. Each objective is fixed and used in every macro-replication. The true extrema used to compute the simple regret are obtained from a 262,144-point Sobol screen followed by multistart local optimization. Tables 1 and 2 report the resulting minimizers, objective ranges, and RKHS norms (Here $R _ { f } = f _ { \mathrm { m a x } } - f ^ { \star }$ and $B _ { f } = \| f \| _ { \mathcal { H } _ { k } } )$ .

Table 1: Properties of the two-dimensional RKHS objectives
<table><tr><td>Kernel</td><td> $x _ { 1 } ^ { \star }$ </td><td> $x _ { 2 } ^ { \star }$ </td><td> $f ^ { \star }$ </td><td> $f _ { \mathrm { m a x } }$ </td><td> $R _ { f }$ </td><td> $B _ { f }$ </td></tr><tr><td>Matérn-5/2</td><td>0.058755</td><td>0.952746</td><td>-2.945546</td><td>3.072480</td><td>6.018026</td><td>6.931367</td></tr><tr><td>Squared exponential</td><td>0.609984</td><td>0.537395</td><td>-3.081055</td><td>3.369256</td><td>6.450311</td><td>6.935835</td></tr></table>

Table 2: Properties of the fixed four-dimensional RKHS objectives.
<table><tr><td>Kernel</td><td> $x ^ { \star }$ </td><td> $f ^ { \star }$ </td><td> $f _ { \mathrm { m a x } }$ </td><td> $R _ { f }$ </td><td> $B _ { f }$ </td></tr><tr><td>Matérn-5/2</td><td>(0.680540,0.617090,0.067218,0.626605)</td><td>-5.390637</td><td>3.219524</td><td>8.610161</td><td>11.255655</td></tr><tr><td>Squared exponential</td><td>(0.678059,0.621087,0.066322,0.619068)</td><td>-6.000295</td><td>3.217247</td><td>9.217542</td><td>11.318412</td></tr></table>

The two-dimensional experiments begin with 20 Latin-hypercube points. Following the work of Jalali et al. (2017), we discretize the space to optimize the EI function. In two dimensional case, the discretized set includes 2,048 scrambled Sobol candidates at each iteration. The four-dimensional experiments begin with 40 Latin-hypercube points and 8,192 discrete scrambled Sobol candidates. In four dimensions, to improve the quality of EI maximization, we add a follow-up gradient search. Specifically, in each iteration, we first select the ten candidates with the largest acquisition values and then adopt L-BFGS-B strting from these ten points. For the replication-order ablation, every four-dimensional arm uses 20 initial points rather than 40. This is to keep the ${ \cal O } ( \bar { T } ^ { 1 / 2 } )$ schedule feasible at the smallest ablation budget.

In the standard CertEI configuration, the replication size is $b _ { T } ~ = ~ \rho \log T$ , rounded upward to the nearest multiple of five. We use $\rho = 8$ for the Matérn- $- 5 / 2$ kernel and $\rho = 6$ for the SE kernel. In addition, the theoretical confidence coeficient is defined through the exact maximum information gain, which is not directly computable. In the numerical experiments, we therefore replace it with a tractable kernel-specific proxy combining its known asymptotic order from Vakili et al. (2021b) with the finite-batch bound in Lemma 6.1.

## O.2 Additional EI-incumbent ablation

We first report the three EI-incumbent ablations in the remaining three scenarios.

Consistent with the four-dimensional Matérn-5/2 result in the main text, the CertEI incumbent attains the smallest average GAP at every reported budget in all three additional settings. As the budget increases, the other two incumbent choices converge more slowly, where the upper-quantile incumbent is generally the worst.

## O.3 Additional Replication-order ablation

We report the remaining replication-order experiments.

For both two-dimensional objectives, O(log T) attains the lowest average GAP at all three budgets. Its advantage over O(log log T) is relatively small for the Matérn- $5 / 2$ objective but more pronounced for the SE objective. By contrast, $O ( T ^ { 1 / 2 } )$ performs poorly on the Matérn-5/2 objective at the smallest budget because its large replication number leave only a few adaptive decisions after initialization. For the four-dimensional SE objective, ${ \cal O } ( \log T )$ has the lowest average GAP at ${ \bar { T } } = e ^ { 1 0 . 5 }$ and $T = e ^ { 1 1 }$ , whereas $O ( T ^ { 1 / 2 } )$ performs best at $T = \stackrel { \cdot } { e } ^ { 1 1 . 5 }$ . Thus, a larger replication order can become competitive when the available budget is suficiently large, but it is less efective at the two smaller budgets. Together with the four-dimensional Matérn-5/2 result in the main text, these experiments support ${ \cal O } ( \log T )$ as a stable compromise between the number of searched locations and the precision obtained at each location.

![](images/7ee0fbc6a9d21db14378d6aa98d135d821f72358f0a2ecdbe58d4d1903194883.jpg)  
(a) Two-dimensional Matérn-5/2.

![](images/e8c20728e327cbeb519bd11ed996454e5fcd228df66df2887bd6edfcb4e64cad.jpg)  
(b) Two-dimensional SE.

![](images/490d4848da759d9d424acd4e47eb2cda7517a4b42f81f08ed188c9a100b0218e.jpg)  
(c) Four-dimensional SE.  
Figure 4: Additional EI-incumbent ablations

## O.4 Additional Final-recommendation ablation

We further examine the final-recommendation rule on the three additional instances.

Figure 6 shows that the calibrated upper quantile and posterior-mean rules perform comparably on these instances, like the one reported in the main text. In contrast, the optimistic lower quantile rule has a larger average GAP throughout, with the separation becoming especially pronounced for the four-dimensional SE instance at large budgets. In this case, we see that the GAP of the lower-quantile recommendation become worse when $T$ increases. As T increases, $\Lambda _ { { N _ { T } } }$ also increases, strengthening the uncertainty reward in the optimistic lower-quantile score $\mu _ { N _ { T } } ( x ) - \Lambda _ { N _ { T } } \sigma _ { N _ { T } } ( x )$ . In the four-dimensional SE setting, substantial residual uncertainty remains, so the larger coeficient makes this rule increasingly likely to select a highly uncertain but truly poor sampled point, causing the average GAP to rise.

## O.4.1 A diagnostic terminal-selection test

We report an additional experiment designed to distinguish our final recommendation rule from returning the sampled point with the smallest posterior mean. When simulation noise is high, more posterior uncertainty remains at termination under the same budget, making the efect of the uncertainty adjustment easier to observe. This experiment illustrates such a case.

The minimization domain is $\mathcal { X } = [ 0 , 1 ] ^ { 4 }$ . The objective is a finite kernel expansion $\begin{array} { r } { f ( \boldsymbol { x } ) = \sum _ { j = 1 } ^ { 1 4 } w _ { j } k _ { 5 / 2 } ( \boldsymbol { x } , \boldsymbol { c } _ { j } ) } \end{array}$ where $k _ { 5 / 2 }$ is the isotropic Matérn- ${ \cdot } 5 / 2$ kernel with lengthscale $\ell = 0 . 2$ . We divide the 14 kernel centers $c _ { j }$ into three groups: safe, global, and decoy. The three groups create regions with diferent combinations of objective quality and optimization dificulty. The safe basin is broad and supported by several nearby centers, making it relatively easy to locate and estimate, but it is deliberately suboptimal. The global basin is deeper but more localized, while the decoy basins provide isolated suboptimal regions that can appear attractive under noisy posterior estimates. This construction creates a nontrivial terminal-selection problem: the algorithm must distinguish a reliably good sampled point from points whose posterior means appear favorable but remain highly uncertain. Let $\mathbf { 1 } _ { 4 } = ( 1 , 1 , 1 , 1 ) ^ { \top }$ , let $c _ { \mathrm { s } } = 0 . 3 0 1 _ { 4 }$ , and let $e _ { i }$ denote the ith coordinate vector. The relative coeficient magnitudes of the safe, global, and decoy components are 1 : 14 : 6. Table 3 reports the normalized coeficients used in the experiment. The objective function is normalized to have response range $R _ { f } = f _ { \mathrm { m a x } } - f ^ { \star } = 1$ , and its RKHS norm

![](images/8612bb7dc7689eae3d9f62b74dccfc2bdc44dbe32a84a56f52aca9bdbfa5bef2.jpg)  
(a) Two-dimensional Matérn-5/2.

![](images/4e95af1e8867a094f374123845120856954f740a1e95dd8d6e2160650dadfd87.jpg)

(b) Two-dimensional SE.  
![](images/0247b742844bbbb4da5d0356cb282b9df0f320ad6d243dd8beb2de5ddb66d372.jpg)  
(c) Four-dimensional SE.  
Figure 5: Average GAP under the four replication orders for the additional objective functions

Table 3: Kernel centers and normalized coeficients of the diagnostic objective.
<table><tr><td>Component</td><td>Kernel centers</td><td>Weight per center</td></tr><tr><td>Safe basin</td><td> $c _ { \mathrm { s } }$  and the eight axis offsets  $c _ { \mathrm { s } } \pm 0 . 1 1 e _ { i } , i = 1 , \ldots , 4$ </td><td>-0.06986611</td></tr><tr><td>Global basin</td><td>(0.72,0.72,0.72,0.72)</td><td>-0.97812556</td></tr><tr><td>Decoy basins</td><td>(0.76, 0.24, 0.76, 0.24), (0.24, 0.76, 0.24, 0.76), (0.78, 0.78, 0.22, 0.22), and (0.22,0.22,0.78,0.78)</td><td>-0.41919667</td></tr></table>

is $\| f \| _ { \mathcal { H } } = 1 . 4 2 3 1 2 6 6 9 .$ . Its global minimizer is $x ^ { \star } = ( 0 . 7 1 8 3 3 8 , 0 . 7 1 8 3 3 8 , 0 . 7 1 8 3 3 8 , 0 . 7 1 8 3 3 8 )$ . We add independent homoscedastic Gaussian noise on ${ f \colon \epsilon \sim \mathcal N \bigl ( 0 , ( 3 R _ { f } ) ^ { 2 } \bigr ) }$ . Thus, the noise standard deviation is $\sigma _ { \epsilon } = 3 R _ { f }$ . It is worth noting that homoscedasticity is imposed only in the data-generating process. The algorithm is not informed that the noise variance is constant and continues to estimate it at each sampled point using the replication-based procedure. The results are reported in the following figure.

Figure 7 shows a clear separation among the three rules. The conservative recommendation attains a sub stantially smaller GAP than posterior-mean selection throughout the tested budget range, while the optimistic recommendation performs worst. Under uniformly high noise, a favorable posterior mean can still be associated with substantial residual uncertainty; penalizing that uncertainty therefore helps prevent an insuficiently supported design from being returned.

![](images/09705ae262344e8638bbc70e4bba0451fba299d6bafd158c5b8aa78d0912d2e1.jpg)  
(a) Two-dimensional Matérn-5/2.

![](images/1dfbc9424559e863808573005fba2d07844ae574312b2fe4b48b762b7a26db3a.jpg)

(b) Two-dimensional SE.  
![](images/f8b44553be11dd42e7c816d6a3a8fb4672e600ce7dec5d37fce88d274e330a56.jpg)  
(c) Four-dimensional SE.

Figure 6: Average GAP under the three final-recommendation rules  
![](images/56f630204f437b08037f4d2114d03a9ac6e80314bb51a219b127249ec9b863a8.jpg)

![](images/5e082339de007f7b796494d0c9ea96e7aac1d7b8d7ea8df7d84df0ccbf36b06b.jpg)

![](images/056ffae93968e599fa75610c0bd060bbc1ca3b5a8b0ea92afda1c67916432f21.jpg)  
Figure 7: GAP distributions at three representative budgets

## O.5 Joint Ablation of the Three Design Choices

We present the crossed experiment that varies all three design choices simultaneously. This gives $3 \times 4 \times 3 = 3 6$ configurations for each function–budget pair. We conduct the crossed experiment on the two-dimensional Matérn-$5 / 2 ,$ , two-dimensional SE, and four-dimensional Matérn-5/2 objectives at $T = e ^ { 1 0 . 5 }$ and $T = e ^ { 1 1 . 5 }$ . The results are shown in the following Figures 8, 9, and 10. In these figures, panel rows correspond to the two budgets and panel columns to the three terminal rules. Within each panel, rows correspond to EI incumbents and columns to replication orders. Each displayed cell is the average GAP over 100 macro-replications.

We observe that the joint ablation yields conclusions consistent with the preceding one-factor-at-a-time experiments. In every tested function–budget setting, the best-performing configurations use ${ \cal O } ( \log T )$ replication together with the proposed lower-quantile incumbent. For the final recommendation, the optimistic lower-quantile rule performs substantially worse than the other two alternatives, whereas the posterior-mean rule and our proposed conservative upper-quantile rule yield similar results. More importantly, the complete configuration formed by our three recommended design choices (outlined in red in the figure) consistently ranks among the top two of all 36 configurations in every tested setting. These results show that the three choices also perform robustly when use jointly, rather than being efective only when the other components are held fixed.

![](images/8cf654159d1d5b53259cea715c0b2acff9b5870588dd08d90b1b0ba8662f5607.jpg)  
Figure 8: Joint ablation on the two-dimensional Matérn-5/2 objective

![](images/b36596518e69c6277aefa48007d07899f6163f24bf7e9a223e669fef34b9b80c.jpg)  
Figure 9: Joint ablation on the two-dimensional SE objective

![](images/f3f89a1a29e9fba7f0831060c47527f66727fcbd978e6f2f30d395e2a3faeec7.jpg)  
Figure 10: Joint ablation on the four-dimensional Matérn-5/2 objective

## O.6 Settings for the Synthetic Test Functions

We consider the six-hump Camelback, rescaled Branin, Hills, and Hartmann-6 functions. All four algorithms use a Matérn-5/2 kernel functions. The GP hyperparameters are initialized by maximum likelihood and re-estimated after every sequential stage using weakly regularized maximum a posteriori estimation.

The standard deviation of the simulation noise is $\tau ( x ) = R _ { f } \{ a + b u ( x ) \}$ , where $a > 0 , b < 0$ , and $a + b > 0$ Therefore, $\tau ( x )$ decreases from $a R _ { f }$ near a global minimizer to $( a + b ) R _ { f }$ near the upper end of the objective range. This deliberately places the largest noise near the global optimum, producing the same qualitative adverse noise pattern considered in Jalali et al. (2017), although the noise magnitudes used here are diferent. The problem-specific settings are summarized in Table 4.

Table 4: Settings for the analytical test functions. The nominal replication size is ρ log T.
<table><tr><td>Problem</td><td>d</td><td>(a, b)</td><td>ρ</td><td>Candidates</td></tr><tr><td>Six-hump Camelback</td><td>2</td><td>(4, −1.75)</td><td>8</td><td>2,048</td></tr><tr><td>Rescaled Branin</td><td>2</td><td>(4,−3)</td><td>6</td><td>2,048</td></tr><tr><td>Hills</td><td>2</td><td>(8, −1.5)</td><td>8</td><td>2,048</td></tr><tr><td>Hartmann-6</td><td>6</td><td>(7.2, −3.5)</td><td>8</td><td>10,000</td></tr></table>

## References

Amaran S, Sahinidis NV, Sharda B, Bury SJ (2016) Simulation optimization: a review of algorithms and applications. Annals of Operations Research 240(1):351–380.

Ankenman B, Nelson BL, Staum J (2010) Stochastic kriging for simulation metamodeling. Operations research 58(2):371–382.

Binois M, Gramacy RB, Ludkovski M (2018) Practical heteroscedastic gaussian process modeling for large simulation experiments. Journal of Computational and Graphical Statistics 27(4):808–821.

Binois M, Huang J, Gramacy RB, Ludkovski M (2019) Replication or exploration? sequential design for stochastic simulation experiments. Technometrics 61(1):7–23, URL http://dx.doi.org/10.1080/00401706.2018.1469433.

Bull AD (2011) Convergence rates of eficient global optimization algorithms. Journal of Machine Learning Research 12(10).

Cai X, Scarlett J (2021) On lower bounds for standard and robust gaussian process bandit optimization. International Conference on Machine Learning, 1216–1226 (PMLR).

Calandriello D, Carratino L, Lazaric A, Valko M, Rosasco L (2022) Scaling Gaussian Process optimization by evaluating a few unique candidates multiple times. Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, 2523–2541 (PMLR), URL https://proceedings.mlr.press/v162/ calandriello22a.html.

Chen CH, Lin J, Yücesan E, Chick SE (2000) Simulation budget allocation for further enhancing the eficiency of ordinal optimization. Discrete Event Dynamic Systems 10(3):251–270.

Chen X, Zhang Y, Xie G, Zhang J (2024) A uniform error bound for stochastic kriging: Properties and implications on simulation experimental design. ACM Transactions on Modeling and Computer Simulation 35(1):1–43.

Cheng RC, Kleijnen JP (1999) Improved design of queueing simulation experiments with highly heteroscedastic responses. Operations research 47(5):762–777.

Chowdhury SR, Gopalan A (2017) On kernelized multi-armed bandits. Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, 844–853 (PMLR), URL https:// proceedings.mlr.press/v70/chowdhury17a.html.

Dai Z, Nguyen QP, Tay SS, Urano D, Leong R, Low BKH, Jaillet P (2023) Batch Bayesian Optimization for replicable experimental design. Advances in Neural Information Processing Systems, volume 36, 36476–36506, URL http://dx. doi.org/10.52202/075280-1584.

Ghosh M (2021) Exponential tail bounds for chisquared random variables. Journal of Statistical Theory and Practice 15(2):35

Gupta S, Rana S, Venkatesh S, et al. (2022) Regret bounds for expected improvement algorithms in gaussian process bandit optimization. International Conference on Artificial Intelligence and Statistics, 8715–8737 (PMLR).

Hu S, Wang H, Dai Z, Low BKH, Ng SH (2025) Adjusted expected improvement for cumulative regret minimization in noisy bayesian optimization. Journal of Machine Learning Research 26(46):1–33.

Huang D, Allen TT, Notz WI, Zeng N (2006) Global optimization of stochastic black-box systems via sequential kriging meta-models. Journal of global optimization 34(3):441–466.

Jalali H, Van Nieuwenhuyse I, Picheny V (2017) Comparison of kriging-based algorithms for simulation optimization with heterogeneous noise. European Journal of Operational Research 261(1):279–301.

Jones DR, Schonlau M, Welch WJ (1998) Eficient global optimization of expensive black-box functions. Journal of Global optimization 13(4):455–492.

Kirschner J, Krause A (2018) Information directed sampling and bandits with heteroscedastic noise. Conference On Learning Theory, 358–384 (PMLR).

Laurent B, Massart P (2000) Adaptive estimation of a quadratic functional by model selection. Annals of statistics 1302–1338.

Meng Q, Wang S, Ng SH (2022) Combined global and local search for optimization with Gaussian Process models. INFORMS Journal on Computing 34(1):622–637, URL http://dx.doi.org/10.1287/ijoc.2021.1078.

Osorio C, Bierlaire M (2013) A simulation-based optimization framework for urban transportation problems. Operations Research 61(6):1333–1345, URL http://dx.doi.org/10.1287/opre.2013.1226.

Pearce MAL, Poloczek M, Branke J (2022) Bayesian optimization allowing for common random numbers. Operations Research 70(6):3457–3472, URL http://dx.doi.org/10.1287/opre.2021.2208.

Pedrielli G, Wang S, Ng SH (2020) An extended two-stage sequential optimization approach: Properties and performance. European Journal of Operational Research 287(3):929–945.

Picheny V, Ginsbourger D, Richet Y, Caplin G (2013) Quantile-based optimization of noisy computer experiments with tunable precision. Technometrics 55(1):2–13.

Quan N, Yin J, Ng SH, Lee LH (2013) Simulation optimization via kriging: a sequential search using expected improvemen with computing budget constraints. IIE Transactions 45(7):763–780.

Ryzhov IO (2016) On the convergence rates of expected improvement methods. Operations Research 64(6):1515–1528, URL http://dx.doi.org/10.1287/opre.2016.1494.

Shahriari B, Swersky K, Wang Z, Adams RP, De Freitas N (2015) Taking the human out of the loop: A review of bayesian optimization. Proceedings of the IEEE 104(1):148–175.

Srinivas N, Krause A, Kakade S, Seeger M (2010) Gaussian process optimization in the bandit setting: no regret and experimental design. Proceedings of the 27th International Conference on International Conference on Machine Learning, 1015–1022.

Sun L, Hong LJ, Hu Z (2014) Balancing exploitation and exploration in discrete optimization via simulation through a gaussian process-based search. Operations Research 62(6):1416–1438, URL http://dx.doi.org/10.1287/opre.2014. 1315.

Takeno S, Inatsu Y, Karasuyama M, Takeuchi I (2024) Posterior sampling-based bayesian optimization with tighter bayesian regret bounds. Proceedings of the 41st International Conference on Machine Learning, 47510–47534.

Takeno S, Inatsu Y, Karasuyama M, Takeuchi I (2025) Regret analysis of posterior sampling-based expected improvement for bayesian optimization. Transactions on Machine Learning Research 2025.

Vakili S, Bouziani N, Jalali S, Bernacchia A, Shiu Ds (2021a) Optimal order simple regret for gaussian process bandits. Advances in Neural Information Processing Systems 34:21202–21215.

Vakili S, Khezeli K, Picheny V (2021b) On information gain and regret bounds in gaussian process bandits. Proceedings of the 24th International Conference on Artificial Intelligence and Statistics, volume 130 of Proceedings of Machine Learning Research, 82–90 (PMLR).

Vazquez E, Bect J (2010) Convergence properties of the expected improvement algorithm with fixed mean and covariance functions. Journal of Statistical Planning and inference 140(11):3088–3095.

Wang J, Clark SC, Liu E, Frazier PI (2020) Parallel bayesian global optimization of expensive functions. Operations Research 68(6):1850–1865.

Wang L, Demeulemeester E (2023) Simulation optimization in healthcare resource planning: A literature review. IISE Transactions 55(10):985–1007.

Wang X, Hong LJ, Jiang Z, Shen H (2025) Gaussian process-based random search for continuous optimization via simulation. Operations Research 73(1):385–407.

Wang Z, de Freitas N (2014) Theoretical analysis of bayesian optimisation with unknown gaussian process hyper-parameters. arXiv preprint arXiv:1406.7758 .

Williams CK, Rasmussen CE (2006) Gaussian processes for machine learning, volume 2 (MIT press Cambridge, MA).

Wu J, Frazier P (2016) The parallel knowledge gradient method for batch bayesian optimization. Advances in neural information processing systems 29.

Xie J, Frazier PI, Chick SE (2016) Bayesian optimization via simulation with pairwise sampling and correlated prior beliefs. Operations Research 64(2):542–559, URL http://dx.doi.org/10.1287/opre.2016.1480.