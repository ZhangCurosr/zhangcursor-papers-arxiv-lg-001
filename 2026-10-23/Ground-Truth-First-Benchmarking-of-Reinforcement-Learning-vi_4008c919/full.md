# Ground-Truth-First Benchmarking of Reinforcement Learning via Converse Optimality: From Smooth, Fully Observed Systems to Non-Smooth or Partially Observed Tasks

Gr´egoire Ouerdane<sup>1,2</sup> Sinan Ibrahim<sup>1,2</sup> Stefan Streif<sup>3</sup> Pavel Osinenko<sup>1,2,4</sup>

<sup>1</sup>Center for Engineering Systems and Sciences, Skoltech, 30 Bolshoy Boulevard, bld. 1, Moscow, 121205, Russia <sup>2</sup>Central University, 7 Gasheka St., bld. 1, Moscow, 123056, Russia

<sup>3</sup>Technische Universit¨at Chemnitz, Straße der Nationen 62, D-09111 Chemnitz, Germany

<sup>4</sup>Sirius University of Science and Technology, 1 Olimpiyskiy Avenue, Sirius, 354340, Russia

Contact email: p.osinenko@gmail.com

## Abstract

Benchmarking in reinforcement learning is usually relative: algorithms are ranked by returns on environments with unknown optima. Distance to optimality remains unobservable, while conclusions may depend on reward design, stochasticity, hyperparameters, environment selection, and simulator details. We develop a ground-truth-first framework based on converse optimality, constructing each task with a certified optimal value function and policy withheld from learners and used only for evaluation. The framework unifies two modes: sensitivity benchmarking samples certified environments from continuous-time generators to assess robustness to task variation, while oracle-referenced comparison uses native discrete-time generators to evaluate policies via optimality gap, regret, and policy error. Continuous-time models support physically interpretable sensitivity studies but introduce dependence on the simulator step (∆t) and numerical scheme. Native discrete-time models align the environment, oracle, and learner on the same transition law, eliminating external integration choices as a confounder. We reproduce quantitative results covering optimizer sensitivity and oracle-referenced comparisons on serial planar arms, dynamically extended nonholonomic vehicles, and quadrotor-inspired tasks. The methodology extends beyond smooth afine Gaussian systems: virtual-input pullback handles non-afine raw-action interfaces and finite-covariance non-Gaussian disturbances. A non-smooth converse optimality formulation enables certified benchmarks with kinked value functions and switching surfaces. For partially observed evaluation, an information-state formulation reduces benchmarking to converse optimality on a suficient statistic and provides an information-penalty reference. Code, fixtures, and diagnostics are available on the public project page.

## Contents

1 Introduction

2 Related Work

3 Ground-Truth-First Fixtures

4 Mode I – Continuous-Time Sensitivity Benchmarking 7   
4.1 Continuous-time converse generators for sensitivity 7   
4.2 Sensitivity evaluation protocol . . 8   
4.3 Sensitivity experiment 8   
5 Mode II – Native Discrete-Time Oracle-Referenced Comparison 9   
5.1 Deterministic discrete-time generator 9   
5.2 Stochastic discrete-time generator . 10   
5.3 Example families 10   
5.4 Oracle-referenced evaluation protocol 11   
5.5 Oracle-referenced experiment 12   
6 Extension I: ConverseQuad – Non-Afine Actuation and Non-Gaussian Gusts 13   
6.1 Virtual Actuator Coordinates 14   
6.2 Raw Action Interface 15   
6.3 Cost Pullback and Oracle Preservation 15   
6.4 Finite-Covariance Non-Gaussian Gusts 16   
6.5 Reported Results 16   
7 Extension II: Non-Smooth Converse Optimality 17   
7.1 Motivation: why non-smooth values matter 18   
7.2 Background: viscosity solutions and Clarke subdiferentials . 18   
7.3 Problem Setup and Assumptions 19   
7.4 Theoretical Results 21   
7.5 General Constructive Procedure 23   
7.6 Construction with a Quadratic Control Cost 25   
7.7 Remarks on Problem Geometry and Multiple Switching Surfaces 26   
7.8 Problem Definition and Generator Pipeline 28   
7.9 Example: Piecewise-Quadratic Value Function 28   
7.10 Validation . 32   
7.11 Dataset Construction . 33   
7.12 Numerical Experiments on the Non-Smooth Benchmark 34   
7.13 Remark on Higher-Dimensional Extensions . 36   
8 Extension III: Partially Observed MDPs 36   
8.1 Information States and Reduction . 37   
8.2 Converse Optimality on the Information State 38   
8.3 Quadratic Information-State Construction 38   
8.4 Realizing the Information State as an Output System 39   
8.5 Benchmark Protocol 40   
8.6 LQG Reference and Information Penalty 40   
9 Discussion 41   
10 Conclusion 42   
A Reproducibility and Released Assets 45   
B Impact Statement 45   
C Proofs for the Smooth Theory 45   
C.1 Results for the Stochastic Discrete-Time Setting . 47   
C.2 Continuous-Time Converse Derivation 50   
D Proofs for the Non-Smooth Theory 50   
E Proofs for the Information-State Theory 53   
F Generator Templates and Derivations 55   
F.1 Deterministic Discrete-Time Homotopy Construction 55   
F.2 Quadratic-Gaussian Energy Identity Derivation 56   
G Benchmark Generation, Experimental Implementation, and Reproducibility 57   
G.1 Generator Validation and Experimental Implementation 57   
G.2 Oracle Evaluation, Artifacts, and Reproducibility Checklist 61

## 1 Introduction

Performance evaluation has become the default empirical language of reinforcement learning (RL). A new method is introduced, evaluated on benchmark environments, and compared against a changing set of baselines. This protocol is valuable, but it leaves a basic control-theoretic quantity hidden: distance to optimality. When the optimal policy is unknown, a benchmark can tell us that one algorithm achieved a larger return than another; it cannot tell us whether that algorithm is near optimal, whether all compared methods are poor, or whether the ranking is an artifact of reward scaling, random seeds, task weighting, environment selection, or simulator conventions. This limitation is not superficial; it is fundamentally structural. Deep RL conclusions are well-known to depend strongly on implementation choices, evaluation protocols, random seeds, hyperparameters, and statistical reporting practices [1, 2, 3]. For example, aggregate rankings may fluctuate significantly when tasks are weighted or normalized diferently [4]. Recent critiques therefore argue that benchmark scores alone are insuficient evidence for broad algorithmic claims; controlled scientific tests are needed to explain how and when algorithms work [5]. For RL control, two particular questions arise: does a learning pipeline remain reliable when the environment varies? and how far is a learned policy from the true optimum on a fixed task? Conventional benchmarks struggle with both when the optimum is unavailable.

Converse optimality ofers a direct alternative. It reverses the usual direction of optimal control: rather than solving a Bellman or Hamilton–Jacobi–Bellman (HJB) equation, one starts from a desired value function and policy and constructs environments for which they are optimal. The environment is therefore generated together with its certificate, turning evaluation from relative scoring to absolute calibrated measurement: optimality gaps, regret, Bellman residuals, and sensitivity across distributions of certified tasks. The central claim of this paper is that converse optimality transforms RL benchmarks into known-answer experimental instruments: it tells not only which method returns the most, but why a method fails, how far it is from optimal, whether conclusions survive controlled environment variation, and whether the strongest baseline is genuinely near the certified optimum.

We organize this view into two complementary modes. In sensitivity benchmarking, a continuous-time generator defines a distribution over certified systems, turning environment variation into an experimental variable that tests pipeline consistency; the learner experiences a discretized version whose behavior depends on the integration step, which is precisely what sensitivity studies are meant to probe. In oracle-referenced comparison, native discrete-time fixtures define the transition law, oracle $( V ^ { \star } , \pi ^ { \star } )$ , and evaluation protocol in the same formalism, removing the discretization confound and enabling absolute evaluation by optimality gap, regret, Bellman residual, and policy error.

Time formalism is part of this design. Continuous-time generators produce physically interpretable non-linear systems, but the MDP experienced by a digital learner depends on the integration step $\Delta t$ and numerical scheme. This dependence is not a defect; it makes continuous-time generators well-suited for sensitivity studies. Native discrete-time generators define the transition law, oracle, and evaluation protocol in the same formalism, removing the discretization confound. Thus, continuous-time generators are natural for sensitivity, whereas native discrete-time fixtures are natural for absolute algorithm comparison.

![](images/c73c8117e17d668971c7b58cacef9908610dcff0d80ccb0594d7523a60ce6323.jpg)  
Figure 1: Ground-truth-first benchmark-generation pipeline. A converse construction samples certified dynamics and costs, validates the optimality certificate, exports an environment fixture, trains learners without oracle access, and evaluates learned policies against the hidden pair (V<sup>⋆</sup>, π<sup>⋆</sup>).

The framework is not limited to smooth, afine, Gaussian, fully observed examples. The present work extends it along three independent axes. First, non-afine raw-action interfaces and finite-covariance non-Gaussian disturbances are handled by virual-input pullback, and demonstrated by the example of the ConverseQuad benchmark. Second, non-smooth value functions are managed via viscosity solutions and Clarke subdiferentials, enabling benchmarks with kinked values and switching surfaces. Third, partially observed evaluation is reduced to converse optimality on a suficient information state, with a linear-quadratic Gaussian (LQG) informationpenalty reference. These extensions share the same certificate logic; they difer in the interface the learner sees and the regularity of the certified value.

Our contribution is methodological. We review the foundations of converse optimality and contribute: (i) a ground-truth-first fixture formalism exporting certified values and policies; (ii) a separation between continuous-time sensitivity families and native discrete-time oraclereferenced fixtures; (iii) an extension to non-afine and non-Gaussian tasks, including Converse-Quad; (iv) a non-smooth converse theory via viscosity solutions and Clarke subdiferentials; (v) an information-state formulation for partially observed benchmarks; and (vi) a validation and evaluation protocol using Bellman consistency, closed-loop checks, and Common Random Numbers (CRN) schedules, demonstrated on optimizer sensitivity and on robotic control families. Section 2 reviews related work, Sections 3––5 develop the framework, Sections 6––8 present the extensions, and Sections 9––10 discuss limitations and open problems. The accompanying project page provides code, fixtures, results, and diagnostics.

## 2 Related Work

RL benchmark suites. Shared environment suites—ALE, MuJoCo, Gym, Procgen, Mini-Grid, bsuite, D4RL [6, 7, 8, 9, 10, 11, 12]—enable large-scale empirical comparison but generally lack known optimal policies. Their primary signal is relative return, and a method may outperform baselines while remaining far from optimal.

Empirical reliability and statistical practice. RL conclusions depend heavily on implementation details, random seeds, and statistical reporting [1, 2, 3]. Benchmark scores alone are insuficient evidence for broad claims [5]. We adopt this diagnostic perspective but change the experimental substrate: generating tasks with certified optima makes absolute error observable.

Classical optimal control and inverse optimal control. Classical optimal control fixes dynamics and cost and solves for a value function and policy, typically via the HJB equation or dynamic programming [13, 14]. This is intractable for general nonlinear systems due to the curse of dimensionality. Inverse optimal control and inverse reinforcement learning take the opposite direction: they infer a cost or reward from observed trajectories or demonstrations [15, 16]. Converse optimality difers from both. It neither solves a forward problem nor infers a cost from data; it prescribes a value function and constructs dynamics and state costs for which the prescribed pair is provably optimal. This makes it uniquely suited to benchmark generation, because the optimal reference is an input to the construction rather than an output of an intractable computation.

Converse optimality. Converse optimality was formalized for continuous-time deterministic systems by Nevistic and Primbs [17], using HJB certificates, and extended to discrete-time systems by G¨ohrt et al. [18], using Bellman-equation conditions. Yaremenko et al. [19] turned the discrete-time construction into a coordinate-free parametric generator for RL benchmarks, and Ibrahim et al. [20] extended it to stochastic control-afine systems with direct oracle metrics. We build on this line. We reuse its certificate logic, its quadratic specializations, and its CRN evaluation protocol, and we extend it in three directions: non-smooth values, non-afine raw actions with non-Gaussian disturbances, and information-state formulations for partially observed evaluation.

Non-smooth optimal control and viscosity solutions. Non-smooth value functions arise naturally in optimal control whenever the control enters linearly without a quadratic penalty (bang-bang control), when the horizon is minimized rather than discounted, when state or control constraints are active, or when the dynamics switch between modes. In these regimes the value function is typically only locally Lipschitz, with kinks along switching manifolds where the classical gradient does not exist. The standard machinery replaces pointwise derivatives with viscosity inequalities [21, 22] and classical gradients with Clarke subdiferentials [23, 24]; when the resulting optimal feedback is discontinuous, the closed loop is interpreted as a diferential inclusion in the Filippov sense [25]. Existing converse constructions, however, assume candidate values that are at least $C ^ { 1 }$ or $C ^ { 2 }$ [17, 18, 19], which excludes precisely the switching-surface and bang-bang structures that make learning hard. We therefore develop a non-smooth converse theory in which the smooth HJB certificate is replaced by a viscosity–Clarke condition, yielding certified benchmarks whose optimal values have known kinks and whose optimal policies may be discontinuous.

Partial observability and information-state reduction. Partially observed control is classically reduced to a fully observed problem on an information state: a statistic of the observation–action history that is suficient for dynamic programming [26, 27]. In finite-state POMDPs this statistic is a belief vector over the hidden state; in linear–Gaussian problems it is the Kalman filter’s conditional mean and covariance; in general nonlinear continuous-state models it is an infinite-dimensional belief density. Recurrent policies approximate this reduction in practice [28, 29], and memory-oriented benchmarks probe it empirically [30], but they do not provide a certified optimal reference. A further complication is that a full-state oracle may exploit information the learner cannot access from its history, so comparing a history-based policy against a latent-state optimum conflates learning error with an information penalty. No converse construction has been formulated on an information state with a certified output-feedback oracle. We address this by reducing partially observed benchmarking to converse optimality on a suficient statistic, certifying the oracle in the learner’s own information set, and isolating the information penalty analytically in an LQG reference.

## 3 Ground-Truth-First Fixtures

A standard RL benchmark provides an environment but not the optimal return. $\mathrm { A }$ ground-truthfirst benchmark adds a certificate that a particular value function $V ^ { \star }$ and policy $\pi ^ { \star }$ are indeed optimal. We work within the usual Markov decision process (MDP) framework with state space $s ,$ action space ${ \mathcal { A } } ,$ transition kernel $P ,$ stage cost $^ { c , }$ and discount γ. For a stationary policy π, the value function is

$$
V ^ { \pi } ( s ) = \mathbb { E } \left[ \sum _ { k = 0 } ^ { \infty } \gamma ^ { k } c ( s _ { k } , \pi ( s _ { k } ) ) \middle | s _ { 0 } = s \right] ,
$$

and the optimal value function satisfies $\begin{array} { r } { V ^ { \star } ( s ) = \operatorname* { i n f } _ { \pi } V ^ { \pi } ( s ) } \end{array}$ . We state the certificate for this discounted, fully observed, smooth discrete-time case, and generalize it in Sections $_ { 6 - 8 }$ to nonsmooth values, non-afine interfaces, and information states.

Cost and reward conventions. All certificates are stated in the cost convention, where lower is better and $V ^ { \star }$ is nonnegative for nonnegative stage costs. When results are reported in the reward convention $r = - c ,$ so that ${ V } _ { r } ^ { \star } = - { V } ^ { \star }$ and the optimality gap $\operatorname { G a p } ^ { r } = - \mathrm { G a p } ^ { c }$ , so that zero is optimal and more negative values indicate larger suboptimality. The convention used in each experiment is stated explicitly in the corresponding caption. This dual convention matters because the Bellman certificate is naturally a cost identity, whereas standard RL libraries report rewards.

Optimality certificate. To check optimality, we use the Bellman residual

$$
\mathcal { R } _ { V } ( s , a ) = c ( s , a ) + \gamma \int V ( s ^ { \prime } ) P ( \mathrm { d } s ^ { \prime } \mid s , a ) - V ( s ) .
$$

The optimality certificate states: if for all states and actions $\mathcal { R } _ { V } ( s , \pi ( s ) ) = 0$ and ${ \mathcal { R } } _ { V } ( s , a ) \geq$ $0 ,$ then $V = V ^ { \star }$ and $\pi$ is optimal (see Appendix $\mathrm { C } )$ . The certificate is checked analytically where possible, and numerically on a declared validation set $\mathcal { G }$ with tolerance $\epsilon _ { B } .$ . Accepted fixtures record the maximum residual, the validation domain, and the numerical tolerance in the validation record. This makes the certificate independently auditable: an external party can reload a fixture, recompute the residual, and verify that the prescribed pair is consistent with the transition law and cost.

Fixtures. A benchmark fixture

$$
\mathfrak { B } _ { \theta } = ( \mathcal { S } , \mathcal { A } , P _ { \theta } , c _ { \theta } , \gamma , V _ { \theta } ^ { \star } , \pi _ { \theta } ^ { \star } , \Omega _ { \theta } )
$$

packages the certified MDP with a validation record. The tuple may additionally carry the time formalism, the regularity class of $V ^ { \star }$ , the action interface the learner sees, and the observation map. The agent trains on the MDP $( S , { \mathcal { A } } , P _ { \theta } , c _ { \theta } , \gamma )$ ; $V _ { \theta } ^ { \star }$ and $\pi _ { \theta } ^ { \star }$ are withheld as evaluation oracles. When $\theta$ is fixed, the fixture provides a single known-answer task. When $\theta$ is sampled from a distribution, one obtains a distribution over certified $M D P s ,$ enabling the two modes in Table 1.

<table><tr><td>Mode</td><td>Time formalism</td><td>Scientific question</td><td>Typical outputs</td></tr><tr><td>Sensitivity robustness</td><td>Continuous-time generator</td><td>Does a learning pipeline remain reli- able when the environment varies?</td><td>Relative suboptimality dis- tributions, mean and vari- ance tests, train-test shift studies</td></tr><tr><td>Oracle- referenced comparison</td><td>time fixture</td><td>Native discrete- How far is a learned policy from the certified optimum on a fixed fixture?</td><td>Optimality gap, regret, Bell- man residual, policy error, paired CRN confidence in- tervals</td></tr></table>

Table 1: Two complementary benchmark modes enabled by ground-truth-first fixtures. Mode I is typically instantiated with a continuous-time generator and Mode II with a native discretetime fixture; Section 4 and Section 5 explain why.

What the extensions change. The extensions in Sections 6–8 modify one element of this picture each, while leaving the fixture concept unchanged. Non-smooth values replace the pointwise Bellman residual with a viscosity–Clarke condition, so that the certificate remains valid at kinks and switching surfaces. Non-afine raw actions and finite-covariance non-Gaussian disturbances change the action interface and the disturbance law, while preserving the same certificate in virtual-input coordinates. Partial observability replaces the state s with a suficient information state b, so that the oracle is attainable from the learner’s history rather than from a hidden state; this also makes the information penalty separable from the learning gap. In all three cases, the task and its certificate are constructed together, and the reference pair is withheld until evaluation. The remainder of the paper develops the two modes (Sections 4–5), the three extensions (Sections 6–8), and validation and reproducibility protocols.

## 4 Mode I – Continuous-Time Sensitivity Benchmarking

The certificate of Section 3 guarantees optimality when certain conditions hold. To turn that certificate into a usable benchmark, one needs converse generator templates—recipes that, given a desired value function V and policy, produce the dynamics and costs that satisfy the certificate. This section develops Mode I for continuous-time sensitivity studies. Mode II (Section 5) develops native discrete-time fixtures for absolute comparison, and Sections 6–8 extend both to non-afine interfaces, non-smooth values, and partial observability. Modes I and II together cover the smooth, fully observed, afine-action case; the extensions relax each of these in turn, while leaving the certificate logic unchanged.

## 4.1 Continuous-time converse generators for sensitivity

Consider a continuous-time, control-afine system ${ \dot { s } } = f ( s ) + g ( s ) a$ with state $s \in \mathbb { R } ^ { n }$ and action $a \in \mathbb { R } ^ { m }$ . We prescribe a positive definite $C ^ { 1 }$ value function V . A converse construction [19] uses two designer-chosen smooth fields: a vector field $H ( s )$ shaping the drift, and a matrix field $G ( s )$ directing the control. The generator outputs

$$
g ( s ) = G ( s ) , \quad f ( s ) = \frac { 1 } { 4 } G ( s ) \big ( \mathcal { L } _ { G } V ( s ) \big ) ^ { \top } - H ( s ) , \quad c ( s , a ) = \mathcal { L } _ { H } V ( s ) + \| a \| _ { 2 } ^ { 2 } ,
$$

where ${ \mathcal { L } } _ { G } V ( s )$ is the Lie derivative of V along the columns of G. A completion of squares (Appendix C.2) gives the Hamiltonian as $\begin{array} { r } { \| a + \frac { 1 } { 2 } ( \mathcal { L } _ { G } V ( s ) ) ^ { \top } \| _ { 2 } ^ { 2 } } \end{array}$ , yielding the certified optimal policy $\begin{array} { r } { a ^ { \star } ( s ) = - \frac { 1 } { 2 } ( { \mathcal { L } } _ { G } V ( s ) ) ^ { \top } } \end{array}$

From generator to sensitivity benchmark. The fields H and G are knobs that parametrize a rich family of certified nonlinear control problems (stable/unstable, underactuated/fully actuated, stif/slack). Sampling them from a distribution produces a distribution over certified

![](images/02bc6baf4fdc222c8e0857fa2891ae335c2f0114839a906a91d2f5d5ca3b763f.jpg)  
Figure 2: Reproduced from Yaremenko et al. [19]. Sensitivity (top) and robustness (bottom) test structures. The symbol ◦ indicates that the policy is substituted into the test problem, closing the loop. The problem sample is split evenly at the junction.

MDPs—the substrate for sensitivity and robustness studies. Because the certificate is continuoustime, the digital RL agent experiences a discretized version whose dynamics depend on the step size $\Delta t$ and integration method. This dependency is not a bug: it makes environment variation and numerical realization part of the experimental variable, precisely what sensitivity studies require. It is also why Mode I is used for sensitivity and not for absolute comparison. Mode II, and its extensions in Sections $6 { - } 7 ,$ removes this confound by working natively in discrete time, so that learner, oracle, and metrics share the same transition law.

## 4.2 Sensitivity evaluation protocol

A continuous-time generator defines a distribution over certified environments indexed by θ. For a sampled environment θ and a learning procedure, a trained policy $\pi \theta$ is obtained. The relative suboptimality (RS) normalizes the incurred cost by the known optimal cost:

$$
\operatorname { R S } ( \pi _ { \theta } ; \theta , s _ { 0 } ) = { \frac { J _ { \theta } ( \pi _ { \theta } ; s _ { 0 } ) } { V _ { \theta } ^ { \star } ( s _ { 0 } ) + \epsilon } } ,
$$

where $J _ { \theta }$ is the cumulative cost and $\epsilon > 0$ is a small constant that guards against division by zero. RS is task-aware: it measures excess cost as a multiple of the optimum, independent of raw task scale. The cost convention is defined in Section $s ;$ reward-convention plots flip the sign.

Sampling N environments and training on each yields a sample {RS }. Two procedures are compared by their RS distributions—the question is not who wins on one task, but whether one procedure systematically produces a diferent suboptimality profile across tasks. An F-test is applied for variance (consistency) and a t-test for means (average performance). The protocol is: sample environments, train, compute RS with the known $V _ { \theta } ^ { \star }$ , and test. This turns environment variation from a nuisance into a controlled, statistically measurable experimental variable.

## 4.3 Sensitivity experiment

The sensitivity protocol is illustrated with the continuous-time case study of a generator producing a sample of 160 unstable three-dimensional control problems, each with a known optimal value [19]. Two PPO pipelines, identical in all respects except the optimizer (Adam vs. RAdam), were trained on independent splits of 80 environments.

Relative suboptimality is computed for each trained policy $\pi _ { \theta }$ . An F-test comparing the two RS distributions gave a variance ratio $F \approx 2 . 5 9 \ ( p \approx 1 . 7 \times 1 0 ^ { - 5 } )$ , rejecting the null of equal variance at any conventional significance level.

<table><tr><td>Quantity</td><td>Value</td></tr><tr><td>Generator family</td><td>Unstable 3D continuous-time converse-optimal systems</td></tr><tr><td>Learning algorithm</td><td>PPO [31]</td></tr><tr><td>Perturbation under test</td><td>Adam vs. RAdam optimizer</td></tr><tr><td>Environment sample size</td><td>160 generated problems, split into 80 per optimizer arm</td></tr><tr><td>Normalization</td><td>Relative suboptimality using each environment&#x27;s known optimal value</td></tr><tr><td>Test statistic</td><td>Variance ratio F ≈ 2.59</td></tr><tr><td>p-value</td><td> $\approx 1 . 7 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Interpretation</td><td>Optimizer choice materially changed outcome variance across certi- fied environments</td></tr></table>

Table 2: Continuous-time sensitivity case study. Two PPO pipelines difered only in optimizer. The measured quantity is relative suboptimality over a generated family of certified environments. Values are reported from the source; the stated test concerns the variance of relative suboptimality.

This result does not mean one optimizer is uniformly “better.” It shows that the optimizer choice materially alters the spread of suboptimality across environments—a measure of pipeline consistency. A return-only benchmark would have obscured this efect entirely. The conversegenerated family makes sensitivity a measurable and statistically testable property. The result is a reported sensitivity study; independent reproduction requires the original generator schedule and training seeds, which are discussed in Appendix G.

## 5 Mode II – Native Discrete-Time Oracle-Referenced Comparison

Mode II uses native discrete-time generators. Unlike Mode I, the transition law itself is the benchmark: learner, oracle, and metrics share the same map, which removes the discretization confound and makes these fixtures suitable for absolute comparison. We proceed from deterministic to stochastic dynamics, then to example families, then to the evaluation protocol. Extensions to non-afine actions and non-Gaussian noise (Section 6), non-smooth values (Section 7), and partial observability (Section 8) preserve the same certificate logic; only the interface and the regularity class change.

## 5.1 Deterministic discrete-time generator

Consider a deterministic, control-afine, discrete-time system $s _ { k + 1 } = f ( s _ { k } ) + g ( s _ { k } ) a _ { k }$ , with state $s \in \mathbb { R } ^ { n }$ and action $a \in \mathbb { R } ^ { m }$ . The stage cost $c ( s , a ) = q ( s ) + \rho ( a )$ is separable, with $q \geq 0$ and $\rho$ strictly convex with $\rho ( 0 ) = 0$ . For a given input-coupling function $g$ and control cost $\rho ,$ , the converse optimality problem asks: given a candidate optimal value function $V ^ { \star }$ , find a drift f such that $V ^ { \star }$ satisfies the Bellman equation for the considered system.

G¨ohrt et al. [18] provided a complete answer. A pair $( V ^ { \star } , q )$ is admissible—i.e., there exists a drift f realizing $V ^ { \star }$ as the optimal value function—if and only if $V ^ { \star } ( s ) - q ( s ) \geq 0$ for all $s \in \mathbb { R } ^ { n }$ The suficiency proof is constructive and uses a homotopy argument (see Appendix F.1). When the value function, the state cost, and the control cost are quadratic, with $V ^ { \star } ( s ) = s ^ { \top } P s .$ $q ( s ) = s ^ { \top } Q s , \rho ( a ) = a ^ { \top }$ Ra and $P , Q , R \succ 0$ , the admissible drift can be written explicitly as

$$
f ( s ) = H ( s ) ^ { - 1 / 2 } S ( s ) ( P - Q ) ^ { 1 / 2 } s , \qquad S ( s ) ^ { \top } S ( s ) = I ,\tag{1}
$$

where $H ( s ) = \left( P ^ { - 1 } + g ( s ) R ^ { - 1 } g ( s ) ^ { \top } \right) ^ { - 1 }$ . The orthogonal field $S ( s )$ can be chosen freely and serves as a design knob for injecting nonlinear coupling while preserving optimality. The optimal policy is $a ^ { \star } ( s ) = - \big ( R + g ( s ) ^ { \top } P g ( s ) \big ) ^ { - 1 } g ( s ) ^ { \top } P f ( s )$

## 5.2 Stochastic discrete-time generator

The converse framework can also be extended to stochastic systems $s _ { k + 1 } = f ( s _ { k } ) + g ( s _ { k } ) a _ { k } +$ $w _ { k }$ with additive Gaussian noise $w _ { k } \sim \mathcal { N } ( 0 , \Sigma )$ [20]. The optimal value function satisfies the stochastic Bellman equation

$$
V ^ { \star } ( s ) = \operatorname* { i n f } _ { a } \big \{ c ( s , a ) + \gamma \mathbb { E } \big [ V ^ { \star } ( f ( s ) + g ( s ) a + w ) \big ] \big \} .
$$

An analogous necessary-and-suficient optimality condition holds (see Appendix C). The cost convention of Section 3 is used throughout.

For benchmark generation, the most useful instance is the quadratic-Gaussian case with value function $V ^ { \star } ( s ) = s ^ { \top } P s + b .$ noise certainty equivalence ofset $b ,$ and quadratic stage costs. The stochastic Bellman equation reduces to a single energy equation

$$
\begin{array} { r } { s ^ { \top } ( P - Q ) s = f ( s ) ^ { \top } H _ { \gamma } ( s ) f ( s ) , } \end{array}\tag{2}
$$

with an explicit state-dependent metric $H _ { \gamma } \succ 0$ , which any drift must satisfy to make $V ^ { \star }$ optimal. Equation (2) is not a constraint to be checked after the fact; it is the design equation. Any drift satisfying it yields a certified fixture, and the free parameter—the orthogonal field $S ( s )$ below— injects nonlinearity without breaking the certificate. Two explicit parameterizations exist: a metric-normalized form $f ( s ) = H _ { \gamma } ^ { - 1 / 2 } S ( s ) ( P - Q ) ^ { 1 / 2 } s$ with an orthogonal field $S ( s )$ , and a square-root form. Both inject bounded nonlinear coupling while preserving the energy identity exactly. A control-strength homotopy $g _ { p } ( s ) = p g ( s ) , \ p \in ( 0 , 1 ]$ varies authority, conditioning, and task dificulty without invalidating the certificate; larger p improves the oracle but amplifies sensitivity to action error, the control-authority paradox analysed in Section 5.5.

In native discrete time, the transition law itself is the benchmark: learner, oracle, and metrics share the same map. This eliminates discretization confounds and makes these fixtures ideal for absolute, calibrated algorithm comparison (Mode II).

## 5.3 Example families

The following families instantiate a general recipe rather than exhaust it. Each family highlights a particular design dimension such as state-dependent actuation, dynamic extension, or nonafine interfaces that the generator can accommodate. Complexity is introduced through certified parameters: link count, module count, dynamic-extension depth, coupling strength, control authority, actuator coordinates, and disturbance structure. A certified latent MDP can be wrapped with partial or image-valued observations, but an output-feedback oracle requires an information-state certificate (Section 8); a hidden-state oracle is not automatically attainable from the learner’s history.

State-dependent actuation and nonlinear coupling. As a concrete example, consider a first family which models a serial n-link planar arm with state $s = [ \theta _ { 1 } , \omega _ { 1 } , \ldots , \theta _ { n } , \omega _ { n } ] ^ { \top } \in \mathbb { R } ^ { 2 n }$ and action $a = [ \tau _ { 1 } , \ldots , \tau _ { n } ] ^ { \top } \in \mathbb { R } ^ { n }$ . The input map is configuration dependent with $[ g ( s ) ] _ { 2 i , i } = \cos \theta _ { i }$ for $i = 1 , \ldots , n ,$ with all other entries zero. The drift is generated by

$$
f _ { p } ( s ) = H _ { \gamma } ^ { ( p ) } ( s ) ^ { - 1 / 2 } S _ { p } ( s ) ( P - Q ) ^ { 1 / 2 } s .\tag{3}
$$

The orthogonal field $S _ { p } ( s )$ is a product of Givens rotations containing tanh terms that create bounded velocity-dependent and neighbor-coupling efects. A sinusoidal term gives a gravity-like bias. The family is non-linear, scalable in dimension, and certified by construction.

Dynamic extension and delayed actuation. The second example illustrates dynamic extension with a nonholonomic vehicle family [20], where for module i, the state is $s _ { i }$ = $[ x _ { i } , y _ { i } , \phi _ { i } , v _ { i } ^ { ( 0 ) } , \ldots , v _ { i } ^ { ( r _ { v } - 1 ) } , \omega _ { i } ^ { ( 0 ) } , \ldots , \omega _ { i } ^ { ( r _ { \omega } - 1 ) } ] ^ { \intercal }$ and the action is $a _ { i } = [ a _ { v , i } , a _ { \omega , i } ] ^ { \top }$ . The kinematic states evolve as

$$
x _ { i , k + 1 } = x _ { i , k } + \tau v _ { i , k } ^ { ( 0 ) } \cos \phi _ { i , k } , \quad y _ { i , k + 1 } = y _ { i , k } + \tau v _ { i , k } ^ { ( 0 ) } \sin \phi _ { i , k } , \quad \phi _ { i , k + 1 } = \phi _ { i , k } + \tau \omega _ { i , k } ^ { ( 0 ) } .\tag{4}
$$

Velocity chains shift upward, with control entering only at the highest derivative. Modules are stacked through the block-diagonal matrix $G ,$ and obtain the overall state $\begin{array} { r } { s _ { k + 1 } = f _ { p } ( s _ { k } ) \ : + } \end{array}$ $p G u _ { k } + w _ { k }$ . Dynamic extension introduces delayed actuation, long credit-assignment paths, and stif closed-loop behavior, stressing RL algorithms diferently from the arm family.

Non-afine actuation and non-Gaussian gusts. A third family, ConverseQuad, is presented in Section 6. It exposes a state-dependent aerial actuation model through a nonlinear raw-action coordinate and admits finite-covariance non-Gaussian disturbances while preserving the certified optimum in virtual-input coordinates.

## 5.4 Oracle-referenced evaluation protocol

Fixtures are validated (see Appendix G) before export. Four absolute metrics are measured.

Normalized optimality gap (tail-corrected). The tail-corrected cost gap is

$$
\mathrm { G a p } _ { T } ^ { c } ( \hat { \pi } ; s _ { 0 } ) = \frac { \mathbb { E } \left[ \sum _ { k = 0 } ^ { T - 1 } \gamma ^ { k } c ( s _ { k } , \hat { \pi } ( s _ { k } ) ) + \gamma ^ { T } V ^ { \star } ( s _ { T } ) \right] - V ^ { \star } ( s _ { 0 } ) } { | V ^ { \star } ( s _ { 0 } ) | + \epsilon } ,
$$

where $\epsilon > 0$ guards against division by zero. The tail term removes truncation bias, yielding zero for the optimal policy (up to numerical precision) and nonnegative values for suboptimal policies. Under the reward convention, the optimality gap is $G ^ { \star } = - V ^ { \star }$ , so zero is optimal and more negative values indicate larger suboptimality. Each plot states explicitly whether Gap<sup>c</sup> or Gap<sup>r</sup> is used.

Discounted regret. Regret accumulates the per-step excess cost incurred by πˆ relative to the optimal policy. It is a trajectory-level diagnostic that complements the scalar gap. Lower cost regret is better.

$$
\operatorname { R e g r e t } _ { T } ^ { c } ( \hat { \pi } ; s _ { 0 } ) = \mathbb { E } \left[ \sum _ { k = 0 } ^ { T - 1 } \gamma ^ { k } \Big ( c ( s _ { k } , \hat { \pi } ( s _ { k } ) ) - c ( s _ { k } , \pi ^ { \star } ( s _ { k } ) ) \Big ) \right] .
$$

If a reward-convention regret is plotted, it is defined by $\mathrm { R e g r e t } _ { T } ^ { r } = - \mathrm { R e g r e t } _ { T } ^ { c }$ , and the sign convention is stated in the corresponding caption.

Policy error and Bellman residual. Policy error is $\mathbb { E } _ { s \sim \mu } \| \hat { \pi } ( s ) - \pi ^ { \star } ( s ) \| _ { 2 } ^ { 2 }$ . The Bellman residual $\mathcal { R } _ { \hat { V } }$ is reported when a critic $\hat { V }$ is available.

CRN. All policies are evaluated on a shared pre-generated schedule of start states and noise streams. Paired bootstrap over trajectories yields confidence intervals, ensuring that diferences reflect algorithms, not noise realizations.

![](images/8e86f56a3b8a4b344c342364a291de2b97d5f4b8c27500ff6568ffec14fcf9fa.jpg)  
Figure 3: Comparing algorithm absolute performance. Discounted reward bars $( r = - c )$ with 95% CIs under CRN. Colors correspond to $p \in \{ 0 . 5 , 0 . 6 , 0 . 8 \}$ .

## 5.5 Oracle-referenced experiment

PPO, A2C, SAC, TD3, and DDPG are evaluated on ConverseArm and NVDEx native discretetime fixtures under a CRN schedule. The three regimes probe diferent failure modes: the serial\_random tests broad occupancy coverage with diverse initial states, and serial\_fixed tests narrow stationary distributions around a fixed initial state, and NVDEx tests long-horizon delayed actuation $( T = 7 0 0 )$ . Figure 3 shows that the oracle-referenced ranking is strongly regime-dependent. On the serial-random fixtures, TD3 has the best point estimate in both optimality gap and regret, followed by SAC and PPO; A2C and DDPG remain substantially farther from the certified optimum.

The ranking changes on the serial-fixed fixtures. PPO is closest to the oracle by a large margin, while SAC and TD3 exhibit large regret despite their strong performance in many standard continuous-control benchmarks. This indicates that the fixed-initial-state regime creates a narrow, stationary occupancy distribution in which PPO’s clipped on-policy updates are more stable, whereas of-policy bootstrapping and replay do not provide the same advantage.

The NVDEx regime shows the strongest separation: PPO is near the certified optimum, while DDPG, SAC, TD3, and A2C remain far from zero. This reflects the vehicle’s dynamic extension: delayed actuation, a longer horizon, and error propagation through feedback-sensitive dynamics. The result does not claim universal PPO superiority; rather, it shows that this fixture exposes a failure mode where delayed actuation and long-horizon conditioning challenge of-policy baselines under the reproduced protocol.

Together, Figures 3 and 10 show that certified fixtures turn RL evaluation from relative ranking into diagnostic measurement. No single algorithm dominates: TD3 excels on serialrandom, while PPO excels on serial-fixed and NVDEx. Distributional plots expose lower-tail failures hidden by means. Because the optimum is known, one can measure not only which method performs best, but also how far each is from optimality and how error is distributed across runs and fixtures.

In the serial-random regime, A2C is concentrated far below zero, while DDPG has a wide distribution with a severe negative tail. PPO, SAC, and TD3 are shifted closer to the oracle, with TD3 having the best central performance. This supports the interpretation that randomized initial states favour algorithms that can exploit broad transition coverage, while still revealing instability in DDPG.

In the serial-fixed regime, PPO’s distribution is shifted closest to zero, whereas SAC and TD3 exhibit lower-tail failures. Thus, the performance change from serial-random to serial-fixed is not only a change in mean score; it is a change in the shape of the error distribution. The fixed regime appears to favour stable on-policy improvement over replay-based reuse.

![](images/4cee30bf75e066c54d6ecd94f0f4071e05c8816977dcf37c950bbc150ef6fe61.jpg)

Figure 4: Serial n-link planar arm (ConverseArm): absolute oracle-referenced performance (optimality gap) across algorithms (rows) and control authority p (columns), with separate panels for n.  
![](images/777b972632a98dffb897db0a6cb1f55f07a248f93f5a7c9493554c3efe2e9dbc.jpg)  
Figure 5: Serial n-link planar arm (ConverseArm): absolute oracle-referenced performance (regret) across algorithms (rows) and control authority p (columns), with separate panels for n.

The NVDEx panel is the clearest example of why oracle-referenced distributions are useful. PPO remains concentrated near the optimum, while the other baselines show broad distributions and severe negative tails. Since NVDEx contains dynamic extension and delayed actuation, these tails indicate that small policy or value-estimation errors can be amplified over long horizons. The certified value function makes this failure severity measurable in oracle units.

## 6 Extension I: ConverseQuad – Non-Afine Actuation and Non-Gaussian Gusts

The fixture methodology is not limited to afine actions or Gaussian disturbances. This section develops ConverseQuad, a certified aerial family in which the RL agent acts through a nonlinear raw-actuator coordinate while the optimum is constructed in a latent afine coordinate and pulled back exactly. The family exposes a failure mode distinct from ConverseArm and NVDEx: policy errors are transformed by the actuator Jacobian before entering the dynamics, so methods that propagate critic-gradient errors through the action map are more exposed, and finite-covariance non-Gaussian gusts increase trajectory variability without changing the certified optimum. The construction is summarized in the following subsections; the state decomposition, input map, and energy identity are derived inline in Sections 6.1–6.4. This section summarizes the construction and reports the results available in the source paper [20].

![](images/fb1c5415a4efccd4a03dfb86c241d420ab585f3889d4915525cd4afe0d98e706.jpg)

![](images/4d525ede8cc44187078d89c6ec930be7c31ea6240299db115c71f83ae72ec8ad.jpg)  
Figure 6: Left: reward-convention optimality gap to the certified optimum, where zero is optimal and more negative values are worse. Right: cost regret under matched CRN evaluation, where lower is better. Bars show means and error bars show paired-bootstrap 95% confidence intervals. The logarithmic vertical scales show order-of-magnitude diferences across algorithms and regimes.

What this family tests. ConverseQuad isolates one interface change and one disturbance change while leaving the rest of the quadratic-Gaussian (QG) construction of Section 5 intact. On the interface side, the agent acts in raw actuator coordinates $u ,$ and the environment is nonafine in $u ;$ the certified optimum is defined in a virtual coordinate $\bar { u } = \phi ( u )$ and pulled back exactly. Because the actuator map is smooth but nonlinear, a policy error in raw coordinates is transformed by the Jacobian $\phi ^ { \prime } ( u ) = 1 + 3 \eta u ^ { 2 }$ before entering the dynamics, which amplifies critic-gradient errors for methods that backpropagate through the action. On the disturbance side, the noise law is any zero-mean finite-covariance distribution, not necessarily Gaussian; a Gaussian mixture is used in the reported experiments to model gust-like perturbations. This stresses value estimation rather than control, since the Bellman minimizer is unchanged.

## 6.1 Virtual Actuator Coordinates

Let the error state be

$$
s = [ x , y , z , \phi , \theta , \psi , v _ { x } , v _ { y } , v _ { z } , \omega _ { x } , \omega _ { y } , \omega _ { z } ] ^ { \top } \in \mathbb { R } ^ { 1 2 } .\tag{5}
$$

The first three components are the translational position error; $\phi , \theta , \psi$ are the roll, pitch, and yaw errors; $v _ { x } , v _ { y } , v _ { z }$ are the linear velocity errors; and $\omega _ { x } , \omega _ { y } , \omega _ { z }$ are the angular velocity errors. Let $\bar { u } = [ \bar { u } _ { 1 } , \bar { u } _ { 2 } , \bar { u } _ { 3 } , \bar { u } _ { 4 } ] ^ { \top } \in \mathbb { R } ^ { 4 }$ denote a virtual thrust-deviation input around hover; the first component is a collective thrust perturbation, and the remaining three are roll, pitch, and yaw moment inputs. With sampling time $\tau > 0$ , mass m $> 0$ , and inertias $J _ { x } , J _ { y } , J _ { z } > 0$ , define

$$
g _ { Q } ( s ) = \tau g _ { \mathrm { c o n t } } ( s ) ,\tag{6}
$$

where

$$
g _ { \mathrm { c o n t } } ( s ) = \left[ \begin{array} { c c c c c } { 0 _ { 6 \times 4 } } & & & & & \\ { - \frac { 1 } { m } \sin \theta } & { 0 } & { 0 } & { 0 } \\ { \frac { 1 } { m } \sin \phi \cos \theta } & { 0 } & { 0 } & { 0 } \\ { \frac { 1 } { m } ( \cos \phi \cos \theta - 1 ) } & { 0 } & { 0 } & { 0 } \\ { 0 } & { \frac { 1 } { J _ { x } } } & { 0 } & { 0 } \\ { 0 } & { 0 } & { \frac { 1 } { J _ { y } } } & { 0 } \\ { 0 } & { 0 } & { 0 } & { \frac { 1 } { J _ { z } } } \end{array} \right] .\tag{7}
$$

The first six rows are zero because position and attitude errors do not change instantaneously; thrust couples to translational velocity through the attitude-dependent terms in rows seven to nine, and moments couple to angular velocity in rows ten to twelve. The virtual-input dynamics are

$$
s _ { k + 1 } = f _ { p } ( s _ { k } ) + p g _ { Q } ( s _ { k } ) \bar { u } _ { k } + w _ { k } , \qquad p \in ( 0 , 1 ] .\tag{8}
$$

The parameter p is the control-authority homotopy of Section 5.

## 6.2 Raw Action Interface

The RL agent acts through a raw command $u \in \mathbb { R } ^ { 4 }$ , mapped to the virtual input by

$$
\bar { u } = \phi ( u ) , \qquad \phi _ { i } ( u _ { i } ) = u _ { i } + \eta u _ { i } ^ { 3 } , \qquad \eta > 0 .\tag{9}
$$

Since $\phi _ { i } ^ { \prime } ( u _ { i } ) = 1 + 3 \eta u _ { i } ^ { 2 } > 0 ,$ ϕ is a component-wise $C ^ { 2 }$ difeomorphism. The map models a stifening actuator: small commands are nearly linear, while large commands are amplified. The environment seen by the RL agent is

$$
s _ { k + 1 } = f _ { p } ( s _ { k } ) + p g _ { Q } ( s _ { k } ) \phi ( u _ { k } ) + w _ { k } .\tag{10}
$$

For $\eta > 0$ , this dynamics is non-afine in the raw action because

$$
\frac { \partial ^ { 2 } } { \partial u _ { i } ^ { 2 } } \big ( g _ { Q } ( s ) \phi ( u ) \big ) = 6 \eta u _ { i } g _ { Q , i } ( s ) ,\tag{11}
$$

where $g _ { Q , i } ( s )$ is the ith column of $g _ { Q } ( s )$ . The parameter η is a dificulty knob: $\eta  0$ recovers the afine QG case of Section 5, and larger η increases the curvature of the actuator map.

## 6.3 Cost Pullback and Oracle Preservation

The virtual-action quadratic cost is

$$
c _ { \bar { u } } \bigl ( s , \bar { u } \bigr ) = s ^ { \top } Q s + \bar { u } ^ { \top } R \bar { u } , \qquad Q \succeq 0 , R \succ 0 .\tag{12}
$$

The raw-action cost is the pullback

$$
c _ { u } ( s , u ) = s ^ { \top } Q s + \phi ( u ) ^ { \top } R \phi ( u ) .\tag{13}
$$

Therefore,

$$
\begin{array} { r l } & { \underset { u } { \mathop { \operatorname* { m i n } } } \left\{ c _ { u } ( s , u ) + \gamma \mathbb { E } V ^ { \star } ( f _ { p } ( s ) + p g _ { Q } ( s ) \phi ( u ) + w ) \right\} } \\ & { \quad \quad = \underset { \bar { u } } { \mathop { \operatorname* { m i n } } } \left\{ s ^ { \top } Q s + \bar { u } ^ { \top } R \bar { u } + \gamma \mathbb { E } V ^ { \star } ( f _ { p } ( s ) + p g _ { Q } ( s ) \bar { u } + w ) \right\} . } \end{array}\tag{14}
$$

Thus the raw-action MDP is action-coordinate equivalent to the virtual afine MDP. If $\bar { u } _ { p } ^ { \star } ( s )$ is the certified virtual optimal action, then

$$
\boldsymbol { u } _ { p } ^ { \star } ( s ) = \boldsymbol { \phi } ^ { - 1 } ( \bar { \boldsymbol { u } } _ { p } ^ { \star } ( s ) ) .\tag{15}
$$

For $V ^ { \star } ( s ) = s ^ { \top } P s + b$ , define

$$
B _ { Q } ^ { ( p ) } ( s ) = R + \gamma p ^ { 2 } g _ { Q } ( s ) ^ { \top } P g _ { Q } ( s ) ,\tag{16}
$$

$$
H _ { Q } ^ { ( p ) } ( s ) = \gamma \Big ( P - \gamma p ^ { 2 } P g _ { Q } ( s ) ( B _ { Q } ^ { ( p ) } ( s ) ) ^ { - 1 } g _ { Q } ( s ) ^ { \top } P \Big ) .\tag{17}
$$

These reduce to $B ( s )$ and $H _ { \gamma } ( s )$ of Section 5 when $g _ { Q }$ is replaced by g and $p = 1$ . Choose

$$
f _ { p } ( s ) = \left( H _ { Q } ^ { ( p ) } ( s ) \right) ^ { - 1 / 2 } S _ { p } ( s ) ( P - Q ) ^ { 1 / 2 } s , \qquad S _ { p } ( s ) ^ { \top } S _ { p } ( s ) = I .\tag{18}
$$

Then

$$
f _ { p } ( s ) ^ { \top } H _ { Q } ^ { ( p ) } ( s ) f _ { p } ( s ) = s ^ { \top } ( P - Q ) s ,\tag{19}
$$

so the Bellman energy identity holds exactly. The optimal virtual input is

$$
\begin{array} { r } { \bar { u } _ { p } ^ { \star } ( s ) = - \gamma p \big ( B _ { Q } ^ { ( p ) } ( s ) \big ) ^ { - 1 } g _ { Q } ( s ) ^ { \top } P f _ { p } ( s ) . } \end{array}\tag{20}
$$

## 6.4 Finite-Covariance Non-Gaussian Gusts

Let $w _ { k }$ be any zero-mean disturbance with finite covariance Σ. For example,

$$
\boldsymbol { w } _ { k } \sim ( 1 - \epsilon ) \mathcal { N } ( 0 , \Sigma _ { 1 } ) + \epsilon \mathcal { N } ( 0 , \Sigma _ { 2 } ) , \qquad \Sigma = ( 1 - \epsilon ) \Sigma _ { 1 } + \epsilon \Sigma _ { 2 } .\tag{21}
$$

The Gaussian mixture is only an illustration; the certificate does not require Gaussianity. Since $V ^ { \star }$ is quadratic,

$$
\begin{array} { r } { \mathbb { E } [ V ^ { \star } ( z + w ) ] = z ^ { \top } P z + \mathrm { t r } ( P \Sigma ) + b . } \end{array}\tag{22}
$$

The Bellman minimizer therefore depends on the disturbance only through its covariance. The value ofset remains

$$
b = \frac { \gamma } { 1 - \gamma } \operatorname { t r } ( P \Sigma ) .\tag{23}
$$

The certificate requires the disturbance to be zero-mean, of finite covariance, and independent across time. Heteroscedastic, state-dependent, or temporally correlated noise is not covered by the present construction; extending the certificate to such disturbances would require modifying the energy identity and the value ofset accordingly.

## 6.5 Reported Results

The results in this subsection are reproduced from the source paper [20]; no new training runs were performed for this manuscript. Table 3 reports the oracle-referenced metrics for PPO, A2C, SAC, TD3, and DDPG on ConverseQuad under CRN, using the same optimality-gap and regret definitions as Section 5.4. The reward-convention optimality gap is reported; zero is optimal and more negative values indicate larger suboptimality.

ConverseQuad exposes a failure mode distinct from ConverseArm and NVDEx. The certificate is constructed in a virtual thrust/moment input, whereas the agent acts through a nonlinear raw-actuator coordinate. Policy errors are therefore transformed by the actuator map before entering the dynamics, and deterministic actor-critic methods become more exposed to criticgradient and action-coordinate errors. In the reported results, PPO and SAC form the leading group on both metrics, TD3 is intermediate, and A2C and DDPG are least reliable. The finitecovariance non-Gaussian disturbances increase trajectory and return variability while leaving the oracle unchanged, producing noisier value estimates for learned critics. The ordering should be read as a diagnostic of the raw-actuator interface under the reproduced protocol, not as a universal ranking: on ConverseArm, the ranking is regime-dependent, with TD3 leading under diverse initial states and PPO leading under fixed initial states and on NVDEx.

<table><tr><td></td><td>Algorithm Optimality gap</td><td>Optimality gap 95% CI</td><td>Regret</td><td>Regret 95% CI</td></tr><tr><td>A2C</td><td>-1.44</td><td>[-1.76, -1.16]</td><td> $2 . 1 3 \times 1 0 ^ { 1 }$ </td><td> $[ 1 . 8 4 \times 1 0 ^ { 1 } , 2 . 4 7 \times 1 0 ^ { 1 } ]$ </td></tr><tr><td>DDPG</td><td>-2.18</td><td>[-2.53, -1.85]</td><td> $3 . 3 5 \times 1 0 ^ { 1 }$ </td><td> $[ 2 . 9 2 \times 1 0 ^ { 1 } , 3 . 8 6 \times 1 0 ^ { 1 } ]$ </td></tr><tr><td>PPO</td><td> $- 5 . 9 2 \times 1 0 ^ { - 1 }$ </td><td> $[ - 7 . 4 0 \dot { \times } 1 0 ^ { - 1 } , - 4 . 6 8 \dot { \times } 1 0 ^ { - 1 } ]$ </td><td>8.52</td><td> $[ 7 . 4 7 , 9 . 6 6 ]$ </td></tr><tr><td>SAC</td><td> $- 6 . 0 9 \times 1 0 ^ { - 1 }$ </td><td> $\left[ - 7 . 6 4 \times 1 0 ^ { - 1 } , - 4 . 7 6 \times 1 0 ^ { - 1 } \right]$ </td><td>9.03</td><td> $[ 7 . { \dot { 6 0 } } , 1 . 0 6 \times \mathrm { \dot { 1 } 0 ^ { 1 } } ]$ </td></tr><tr><td>TD3</td><td> $- 7 . 6 2 \times 1 0 ^ { - 1 }$ </td><td> $[ - 9 . 3 0 \times 1 0 ^ { - 1 } , - 6 . 2 4 \times 1 0 ^ { - 1 } ]$ </td><td> $1 . 2 3 \times 1 0 ^ { 1 }$ </td><td> $[ 1 . 0 0 \times 1 0 ^ { 1 } , 1 . 5 1 \times \bar { 1 0 ^ { 1 } } ]$ </td></tr></table>

Table 3: ConverseQuad oracle-referenced metrics under CRN. Reproduced from [20]. Optimality gap uses the reward convention: zero is optimal and more negative values are worse; regret is in cost units, where smaller is better. Intervals are the source’s reported 95% paired-bootstrap intervals.

The remaining question—how the ranking changes as the actuator nonlinearity η and the mixture parameter ϵ are varied—is not resolved by the reported results and is left to future work. A systematic sweep over these two knobs would separate the contribution of interface nonlinearity from disturbance non-Gaussianity, both of which are present simultaneously in the current ConverseQuad configuration.

![](images/6bd7d11a610deb9e3238883db9b47f7a3efadb0089b988b6790b29172daf8e2e.jpg)

![](images/d06d314a185d2117572c2ddfb7b5a2751ef86492e1bb5ce9a48d46eda2f4db07.jpg)

![](images/a7b13573508b5d07102a7fafd7cc1304ebd23d6f6df37f15bf60238601d4c706.jpg)

![](images/e8a2e6eb0004f82d2695497042d46eb9c9d75b81c557a37053cbf6fb926feaae.jpg)  
Figure 7: Compact oracle-referenced summaries. (a) Per-experiment optimality gap and regret show how algorithm rankings change with occupancy measure, stifness, and actuation geometry. (b) ConverseQuad regret under nonlinear raw actuation and non-Gaussian disturbances. (c) Macro optimality gap; the closer the values are to zero, the better.

Forward link. The next extension (Section 7) changes the regularity class of $V ^ { \star }$ itself, rather than the action interface or the disturbance law. The certificate logic is unchanged: the task and its reference pair are constructed together, and the reference is withheld until evaluation.

## 7 Extension II: Non-Smooth Converse Optimality

The fixture viewpoint extends to value functions that are merely locally Lipschitz, such as piecewise-quadratic templates with kinks along switching surfaces. This section develops nonsmooth converse optimality. We replace the smooth HJB certificate of Sections 4–6 with a viscosity–Clarke condition: classical gradients are replaced by Clarke subdiferentials, and the pointwise HJB equation is replaced by its viscosity formulation. The construction is deterministic and continuous-time. A homotopy parameter selects a generalized gradient branch on the switching surface, generating families of systems that share the same value function but difer in the sharpness of the non-smooth behaviour. We work the construction end-to-end for a twodimensional piecewise-quadratic value function with a straight switching line, and we validate the resulting fixture analytically.

## 7.1 Motivation: why non-smooth values matter

Bang-bang control, minimum-time problems, saturated policies, state and control constraints, and hybrid or switching systems all produce optimal value functions that are locally Lipschitz but not diferentiable everywhere. When the control enters linearly and the cost does not penalize control quadratically, the optimal policy is typically bang-bang, and the value function has kinks along switching manifolds where the classical gradient does not exist [24]. In these regimes, the classical HJB equation is ill-defined, and the smooth converse constructions of Sections 4–6 do not apply.

Without a certified ground truth, a learner’s poor performance on such a task cannot be attributed to the non-smoothness of the optimum rather than to approximation or optimization error. Standard RL benchmarks can exhibit non-smooth value functions, but absent a reference they cannot distinguish genuine non-smoothness from overfitting. Non-smooth converse optimality closes this gap: it produces tasks whose optimal value has a known kink and whose optimal policy may be discontinuous, so that a learned policy can be measured against a certified reference in exactly the regime that stresses smoothness assumptions.

## 7.2 Background: viscosity solutions and Clarke subdiferentials

Viscosity solutions. When the optimal value function is not diferentiable, the classical HJB equation is replaced by its viscosity formulation. Consider a first-order Hamilton–Jacobi equation

$$
F ( s , V ( s ) , D V ( s ) ) = 0 , \quad s \in \Omega ,
$$

where $\Omega \subset \mathbb { R } ^ { n }$ is open and $F : \Omega \times \mathbb { R } \times \mathbb { R } ^ { n }  \mathbb { R }$ is continuous. A continuous function $V$ is a viscosity subsolution if for every $C ^ { 1 }$ test function $\phi$ and every local maximum $s _ { 0 }$ of $V - \phi$ 2

$$
F ( s _ { 0 } , V ( s _ { 0 } ) , D \phi ( s _ { 0 } ) ) \leq 0 ,
$$

and a viscosity supersolution if for every local minimum $s _ { 0 }$ of $V - \phi$

$$
F ( s _ { 0 } , V ( s _ { 0 } ) , D \phi ( s _ { 0 } ) ) \geq 0 .
$$

V is a viscosity solution if it is both [21, 22]. For optimal control, the optimal value function is the unique viscosity solution of

$$
\operatorname* { s u p } _ { a \in \mathbb { A } } \big \{ - \nabla V ^ { * } ( s ) \cdot f ( s , a ) - c ( s , a ) \big \} = 0 ,
$$

subject to appropriate boundary conditions. Uniqueness follows from the comparison principle and is what allows the viscosity formulation to certify optimality without classical diferentiability.

Clarke subdiferentials. For a locally Lipschitz function $V ,$ , the Clarke generalized directional derivative is

$$
V ^ { \circ } ( s ; v ) : = \operatorname* { l i m } _ { y \to s , t \downarrow 0 } \operatorname* { l i m } _ { t } - V ( y ) - V ( y ) \quad
$$

and the Clarke subdiferential is

$$
\partial _ { C } V ( s ) : = \big \{ \zeta \in \mathbb { R } ^ { n } : V ^ { \circ } ( s ; v ) \geq \langle \zeta , v \rangle \ \forall v \in \mathbb { R } ^ { n } \big \} .
$$

The set $\partial _ { C } V ( s )$ is nonempty, compact, and convex. It reduces to the singleton $\{ \nabla V ( s ) \}$ where $V$ is smooth, and to the convex hull of limiting gradients from adjacent smooth regions at a switching surface. A locally Lipschitz V is a viscosity solution of the HJB equation if and only if for every s and every $p \in \partial _ { C } V ( s )$ the corresponding HJB inequality holds [24]. This equivalence is what makes the viscosity framework constructive: it allows the optimality certificate to be checked pointwise using generalized gradients rather than test functions.

When the optimal policy is discontinuous, the closed-loop dynamics $\dot { s } = f ( s ) + g ( s ) a ^ { * } ( s )$ is interpreted as a diferential inclusion in the sense of Filippov [32]. Solutions exist and are unique under mild regularity, so a discontinuous optimal feedback still defines a well-posed benchmark.

Relation to the smooth case. When V is continuously diferentiable, $\partial _ { C } V ( s ) = \{ \nabla V ( s ) \}$ and the viscosity–Clarke condition collapses to the classical pointwise HJB equation used in Sections $4 { - } 6$ . The non-smooth formulation is therefore a strict generalization of the smooth converse theory, not a replacement for it.

## 7.3 Problem Setup and Assumptions

Let the state space be $\mathbb { R } ^ { n }$ and the control space be $\mathbb { A } \subset \mathbb { R } ^ { m }$ . We consider control-afine dynamical systems of the form

$$
\dot { s } ( t ) = f ( s ( t ) ) + g ( s ( t ) ) a ( t ) , \quad s ( 0 ) = s _ { 0 } ,\tag{24}
$$

where $a ( \cdot )$ is an admissible control signal taking values in $\mathbb { A } .$ . We use the cost convention of Section 3: the running cost $c : \mathbb { R } ^ { n } \times \mathbb { A } $ R is nonnegative, and lower cost is better.

For a stationary policy $\pi : \mathbb { R } ^ { n }  \mathbb { A }$ , the infinite-horizon cost functional is

$$
V ^ { \pi } ( s _ { 0 } ) = \int _ { 0 } ^ { \infty } c ( s ( t ) , \pi ( s ( t ) ) ) d t ,\tag{25}
$$

and the optimal value function is

$$
V ^ { * } ( s _ { 0 } ) = \operatorname* { i n f } _ { \pi } V ^ { \pi } ( s _ { 0 } ) .\tag{26}
$$

In the converse optimality setting we instead begin with a prescribed candidate $V : \mathbb { R } ^ { n } $ R and aim to construct $( f , g , c )$ such that $V = V ^ { * }$ and the minimizer of the Hamiltonian is a certified optimal feedback. Throughout this section, V denotes the prescribed candidate, and $V ^ { * }$ denotes the true optimal value; the goal of the construction is to make them coincide.

Assumption 7.3.1 (System and control structure). (a) The control set $\mathbb { A } \subset \mathbb { R } ^ { m }$ is compact and nonempty.

(b) The dynamics $f : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ and the control matrix field $g : \mathbb { R } ^ { n }  \mathbb { R } ^ { n \times m }$ are continuous and locally Lipschitz in the state variable: for any compact $S \subset \mathbb { R } ^ { n }$ there exist $L _ { f , S } , L _ { g , S } > 0$ such that for all $s _ { 1 } , s _ { 2 } \in S$ 7

$$
\| f ( s _ { 1 } ) - f ( s _ { 2 } ) \| \leq L _ { f , S } \| s _ { 1 } - s _ { 2 } \| , \qquad \| g ( s _ { 1 } ) - g ( s _ { 2 } ) \| \leq L _ { g , S } \| s _ { 1 } - s _ { 2 } \| .
$$

(c) The running cost $c ( s , a )$ is continuous, nonnegative, and bounded from below.

(d) The control cost $a \mapsto c ( s , a )$ is coercive for each $s ,$ in the sense that inf $\dot { \cdot } _ { a \in \mathbb { A } } c ( s , a ) \to \infty$ as $\left\| a \right\| \to \infty$ whenever A is unbounded; when A is compact this condition is automatic.

Compactness of the control set guarantees the existence of minimizing controls in the pointwise optimization problems arising in the HJB equation. Lipschitz continuity of the dynamics ensures existence and uniqueness of trajectories for admissible controls. Coercivity of the control cost is used in Lemma 7.4.1 to ensure that the pointwise minimizer is unique and that the stationarity condition is both necessary and suficient. These assumptions are standard in dynamic programming and HJB theory and are required for the validity of the dynamic programming principle.

Assumption 7.3.2 (Value function regularity). The candidate value function $V : \mathbb { R } ^ { n }  \mathbb { R }$ satisfies:

(a) V is continuous and locally Lipschitz.

(b) V is therefore diferentiable almost everywhere (Rademacher), and at points of non-diferentiability the Clarke subdiferential $\partial _ { C } V ( s )$ exists and is nonempty.

(c) For the candidate optimal feedback $a ^ { * } ( s )$ constructed in this section, the corresponding closed-loop trajectory s(·) satisfies

$$
\int _ { 0 } ^ { \infty } c { \bigl ( } s ( t ) , a ^ { * } ( s ( t ) ) { \bigr ) } d t < + \infty .
$$

Local Lipschitz continuity is the natural regularity class for value functions arising in optimal control: it guarantees almost-everywhere diferentiability, and it is preserved under the operations that appear in the Bellman equation. The Clarke subdiferential then provides a generalized gradient suitable for non-smooth analysis and yields HJB conditions consistent with viscosity solution theory (Section 7.2). We do not require V to be convex or piecewise quadratic; those are special cases used for the worked example in Section 7.9.

Definition 7.3.1 (Viscosity HJB condition)). The triple $( f , g , c )$ realizes V as an optimal value function if the following conditions hold:

(i) (Subsolution) For every $s \in \mathbb { R } ^ { n }$ there exists $p \in \partial _ { C } V ( s )$ such that

$$
\operatorname* { i n f } _ { a \in \mathbb { A } } \left\{ c ( s , a ) + p \cdot \left( f ( s ) + g ( s ) a \right) \right\} \leq 0 .
$$

(ii) (Supersolution) For every $s \in \mathbb { R } ^ { n }$ and every $p \in \partial _ { C } V ( s )$ ，

$$
\operatorname* { i n f } _ { a \in \mathbb { A } } \left\{ c ( s , a ) + p \cdot \left( f ( s ) + g ( s ) a \right) \right\} \geq 0 .
$$

(iii) (Measurable optimal feedback) There exists a measurable function $a ^ { * } : \mathbb { R } ^ { n } \to \mathbb { A }$ and a measurable selection $p ( s ) \in \partial _ { C } V ( s )$ such that for almost every s,

$$
a ^ { \ast } ( s ) \in \arg \operatorname* { m i n } _ { a \in \mathbb { A } } \big \{ c ( s , a ) + p ( s ) \cdot ( f ( s ) + g ( s ) a ) \big \} ,
$$

and the minimum value equals zero for almost every s.

Definition 7.3.1 is the non-smooth analogue of the Bellman residual certificate of Section 3. In the smooth case, the residual $\mathcal { R } _ { V } ( s , a ) \geq 0$ for all a and $\mathcal { R } _ { V } ( s , \pi ( s ) ) = 0$ are replaced by the subsolution and supersolution inequalities above, evaluated on elements of the Clarke subdiferential. If the viscosity HJB condition holds, then V is a viscosity solution of the Hamilton–Jacobi– Bellman equation associated with the control problem [22, 33], and by uniqueness of viscosity solutions $V = V ^ { * }$ . The feedback $a ^ { * }$ in condition (iii) is then an optimal policy. Well-posedness of the closed-loop system when $a ^ { * }$ is discontinuous is established separately in Lemma 7.4.3.

Given a non-smooth candidate V satisfying Assumption 2, three questions arise, and they structure the rest of the section: does there exist a triple $( f , g , c )$ meeting the viscosity condition (Section 7.4); can explicit formulas for $( f , g , c )$ be given in terms of $V$ and its Clarke subdiferential (Section 7.5); and can the construction be parameterized to produce families of systems that all share the same V (Section 7.9)?

When V is $C ^ { 1 } , \partial _ { C } V ( s ) = \{ \nabla V ( s ) \}$ and the viscosity condition collapses to the classical pointwise HJB equation. The non-smooth formulation is therefore a strict generalization of the smooth converse theory of Sections 4–6 and of the discrete-time constructions in [17, 18].

## 7.4 Theoretical Results

We now establish the theoretical foundations of non-smooth converse optimality for deterministic control-afine systems. Throughout, we assume that the candidate value function V satisfies Assumption 7.3.2 (locally Lipschitz, with Clarke subdiferential $\partial _ { C } V ( s )$ defined everywhere), and that the system components $( f , g , c )$ satisfy Assumption 7.3.1. Full proofs are given in Appendix D; we state the results and give proof sketches here.

Lemma 7.4.1 (First-order optimality condition). Let Assumptions 7.3.1–7.3.2 hold, and let the stage cost be $c ( s , a ) = q ( s ) + \rho ( a )$ with $\rho \in C ^ { 2 }$ strictly convex. Assume that the action space is unconstrained, $\mathbb { A } = \mathbb { R } ^ { m }$ , and that V is a viscosity solution of the HJB equation. Suppose there exist measurable functions $a ^ { * } : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ and $p : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ with $p ( s ) \in \partial _ { C } V ( s )$ such that the minimum in the viscosity HJB condition of Definition 7.3.1 is attained and equals zero for almost every s. Then, for almost every s, the first-order condition for the pointwise minimization over a holds:

$$
\nabla _ { a } \rho ( a ) \big | _ { a = a ^ { * } ( s ) } + g ( s ) ^ { \top } p ( s ) = 0 .\tag{27}
$$

Proof sketch. For a fixed s where the minimum is attained, the function $a \mapsto \rho ( a ) + p ( s ) \cdot g ( s ) a$ is strictly convex and coercive on $\mathbb { R } ^ { m }$ , so its unique minimizer $a ^ { * } ( s )$ satisfies the unconstrained first-order condition. The state cost $q ( s )$ and the drift term $p ( s ) \cdot f ( s )$ do not depend on a. The full argument is in Appendix D. □

Remark. When A is a compact subset of $\mathbb { R } ^ { m }$ and the minimizer lies on the boundary, condition (27) is replaced by a variational inequality. The stationarity conditions in Lemma 7.4.2 and Theorem 7.4.4 below are therefore stated under the convention that (i) is either satisfied by an interior minimizer or holds with equality of directional derivatives on the feasible cone at $a ^ { * } ( s )$

Lemma 7.4.2 (Suficient condition for optimality). Let Assumptions 7.3.1–7.3.2 hold. Let V be locally Lipschitz, and suppose there exist measurable functions $a ^ { * } : \mathbb { R } ^ { n } \to \mathbb { A }$ and $p : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ with $p ( s ) \in \partial _ { C } V ( s )$ such that for almost every $s \in \mathbb { R } ^ { n }$

(i) (Stationarity) $\nabla _ { a } \rho ( a ) | _ { a = a ^ { * } ( s ) } + g ( s ) ^ { \top } p ( s ) = 0$ , understood in the variational sense of the remark above when the minimizer is on the boundary;

(ii) (Hamiltonian optimality) $a ^ { * } ( s )$ attains the infimum

$$
\operatorname* { i n f } _ { a \in \mathbb { A } } \left\{ c ( s , a ) + p ( s ) \cdot ( f ( s ) + g ( s ) a ) \right\} = 0 ;
$$

(iii) (Viscosity supersolution) for every $\tilde { p } \in \partial _ { C } V ( s )$

$$
\operatorname* { i n f } _ { a \in \mathbb { A } } \left\{ c ( s , a ) + \tilde { p } \cdot \left( f ( s ) + g ( s ) a \right) \right\} \geq 0 .
$$

Then V is the optimal value function for the control problem with dynamics ${ \dot { s } } = f ( s ) + g ( s ) a$ and running cost $^ { c , }$ and $a ^ { * }$ is an optimal feedback policy.

Proof sketch. Conditions (ii) and (iii) state that V satisfies the viscosity HJB condition of Definition 7.3.1. By uniqueness of viscosity solutions, V coincides with the optimal value function. Condition (i) identifies $a ^ { * } ( s )$ as the minimizer of the Hamiltonian for the selected generalized gradient. The full argument is in Appendix D. □

Remark (Recovering the smooth case). When V is continuously diferentiable, $\partial _ { C } V ( s ) =$ = $\{ \nabla V ( s ) \}$ and the viscosity conditions reduce to the classical HJB equation. Lemma 7.4.1 then recovers the standard first-order optimality condition, and Lemma 7.4.2 becomes the classical verification theorem. The non-smooth formulation is therefore a strict generalization of the smooth case, not a parallel theory.

Lemma 7.4.3 (Measurability of the optimal feedback). Let Assumptions 7.3.1–7.3.2 hold, and suppose a measurable selection $p ( s ) \in \partial _ { C } V ( s )$ is given.

(i) The set-valued map

$$
\Phi ( s ) : = \arg \operatorname* { m i n } _ { a \in \mathbb { A } } \left\{ c ( s , a ) + p ( s ) \cdot ( f ( s ) + g ( s ) a ) \right\}
$$

is nonempty, compact-valued, and weakly measurable; equivalently, its graph is a Borel subset of $\mathbb { R } ^ { n } \times \mathbb { A }$

(ii) Φ admits a measurable selection $a ^ { * } : \mathbb { R } ^ { n }  \mathbb { A } , i . e .$ , a Lebesgue-measurable function with $a ^ { * } ( s ) \in \Phi ( s )$ for almost every s.

(iii) The closed-loop vector field $\dot { s } = f ( s ) + g ( s ) a ^ { * } ( s )$ is measurable and locally bounded, and therefore admits Filippov solutions [32] from any initial condition.

Proof sketch. Since $c , f , g$ , and p are measurable in s and continuous in a, the integrand is a Carath´eodory function. The control set A is compact, so the argmin correspondence is nonempty and compact-valued, and its graph is Borel. The Kuratowski–Ryll-Nardzewski selection theorem then gives a Borel-measurable selector, and Filippov’s existence theorem gives solutions of the closed loop. The full argument is in Appendix D. □

Theorem 7.4.4 (Non-smooth continuous-time converse optimality). Let Assumptions $\ 7 . 3 . 1 -$ 7.3.2 hold, and let an input-coupling matrix $g : \mathbb { R } ^ { n }  \mathbb { R } ^ { n \times m }$ , a stage cost $c ( s , a ) = q ( s ) + \rho ( a )$ with q continuous and $\rho \in C ^ { 2 }$ strictly convex, and a candidate locally Lipschitz value function $V : \mathbb { R } ^ { n }  \mathbb { R }$ be given. The following statements are equivalent:

(a) There exists a drift $f : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ such that the triple $( f , g , c )$ realizes V as the optimal value function for the control-afine system ${ \dot { s } } = f ( s ) + g ( s ) a$

(b) There exist a drift $f : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ , a measurable policy $a ^ { * } : \mathbb { R } ^ { n } \to \mathbb { A }$ , and a measurable selection $p : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ with $p ( s ) \in \partial _ { C } V ( s )$ such that for almost every $s \in \mathbb { R } ^ { n }$ :

(i) $\nabla _ { a } \rho ( a ) | _ { a = a ^ { \ast } ( s ) } + g ( s ) ^ { \top } p ( s ) = 0 ;$

(ii) $\begin{array} { r } { \operatorname* { i n f } _ { a \in \mathbb { A } } \{ c ( s , a ) + p ( s ) \cdot ( f ( s ) + g ( s ) a ) \} = 0 } \end{array}$ , attained at $a ^ { * } ( s )$

(iii) for every $\begin{array} { r } { \tilde { p } \in \partial _ { C } V ( s ) , \operatorname* { i n f } _ { a \in \mathbb { A } } \{ c ( s , a ) + \tilde { p } \cdot \left( f ( s ) + g ( s ) a \right) \} \geq 0 . } \end{array}$

Whenever (b) holds, the policy $a ^ { * }$ is optimal and V is the optimal value function. Conversely, whenever (a) holds, a triple satisfying (b) exists.

Proof sketch. $\mathrm { ( b ) \ \Rightarrow \ ( a ) }$ is Lemma $7 . 4 . 2 . \ ( { \mathrm { a } } ) \Rightarrow ( { \mathrm { b } } )$ uses the fact that the true optimal value function $V ^ { * } = V$ is a viscosity solution of the HJB equation, so the viscosity conditions hold with some measurable selection $p ( s ) \in \partial _ { C } V ( s )$ ; the stationarity condition is then Lemma 7.4.1, and measurability of the selection is Lemma 7.4.3. The full proof is in Appendix D. □

Role of Lemma 7.4.3. Although Lemma 7.4.3 does not appear explicitly in the statement of Theorem 7.4.4, it is what makes the converse framework well-posed. Definition 7.3.1 and Lemma 7.4.2 both require a measurable optimal feedback, and Lemma 7.4.3 guarantees that one exists and that the resulting closed loop is a Filippov system. This is what allows the generated benchmark to be simulated even when the optimal policy is discontinuous.

The theorem is constructive. Section 7.5 turns it into a six-step procedure for a given candidate V, and Section 7.9 instantiates the procedure for a two-dimensional piecewise-quadratic value function.

## 7.5 General Constructive Procedure

We now turn Theorem 7.4.4 into a systematic procedure that, given a candidate locally Lipschitz value function V, produces a triple $( f , g , c )$ for which V is optimal in the viscosity sense. The procedure makes no assumption about the specific form of the control cost $\rho$ beyond strict convexity and $C ^ { 2 }$ regularity; the quadratic case is treated in Section 7.6 and instantiated in Section 7.9.

The procedure uses Assumptions 7.3.1–7.3.2. The input-coupling matrix $g : \mathbb { R } ^ { n }  \mathbb { R } ^ { n \times m }$ is prescribed and satisfies Assumption 7.3.1(b). The control set A is compact and nonempty, or $\mathbb { A } = \mathbb { R } ^ { m }$ in the unconstrained case. The candidate $V : \mathbb { R } ^ { n }  \mathbb { R }$ is locally Lipschitz and its Clarke subdiferential $\partial _ { C } V ( s )$ is explicitly characterized. Among the six steps below, Steps 1–3 are design choices: they depend on $V ,$ on the geometry of the switching surfaces, and on a gain matrix K that shapes the uncontrolled dynamics. Steps 4–6 are then determined: the state cost, the supersolution check, and the closed-loop well-posedness follow from the design choices.

Step 1. Subdiferential analysis. Compute the Clarke subdiferential $\partial _ { C } V ( s )$ for every $s \in$ $\mathbb { R } ^ { n }$ . Identify:

• smooth regions—open sets on which $\partial _ { C } V ( s ) = \{ \nabla V ( s ) \}$ is a singleton;

• switching surfaces—lower-dimensional manifolds on which V is non-diferentiable and $\partial _ { C } V ( s )$ has more than one element.

On each switching surface, express $\partial _ { C } V ( s )$ as the convex hull of finitely many limiting gradients from adjacent smooth regions. Introduce local parameters $\alpha \in [ 0 , 1 ] ^ { k }$ (a single scalar α for a codimension-one surface) that parametrize this convex hull. These parameters are homotopy knobs: diferent values of α produce diferent admissible systems that share the same value function V.

Step 2. Selection of a generalized gradient. Choose a measurable selection $p : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ with $p ( s ) \in \partial _ { C } V ( s )$ for every s. On smooth regions the selection is unique, $p ( s ) = $ $\nabla V ( s )$ . On switching surfaces the selection may depend on the parameters α introduced in Step 1. Diferent selections yield diferent admissible systems but leave V unchanged.

Step 3. Construction of the drift. Choose a locally Lipschitz, positive definite gain matrix $K ( s ) \succ 0$ , and set

$$
f ( s ) : = - K ( s ) p ( s ) .\tag{28}
$$

The negative sign makes V decrease along the uncontrolled dynamics: since $p ( s ) \cdot f ( s ) =$ $- p ( s ) ^ { \intercal } K ( s ) p ( s ) \leq 0$ , the drift performs generalized steepest descent in the K-weighted norm. Other choices of $f$ satisfying the Hamiltonian optimality condition are possible; the discrete-time constructions of Section 5 use square-root and metric-normalized parameterizations that also satisfy the corresponding energy identity.

Step 4. Determination of the state cost. With the drift fixed by (28), the optimal feedback $a ^ { * }$ is determined by the stationarity condition of Lemma 7.4.1. For general strictly convex $\rho \in C ^ { 2 }$ , the equation

$$
\nabla _ { a } \rho ( a ) \big | _ { a = a ^ { * } ( s ) } + g ( s ) ^ { \top } p ( s ) = 0\tag{29}
$$

implicitly defines $a ^ { * } ( s )$ . Because $\rho$ is strictly convex and coercive, the solution exists, is unique, and is measurable in s by Lemma 7.4.3. The state cost $q ( s )$ is then determined by the Hamiltonian optimality condition (condition (ii) of Theorem 7.4.4): for the chosen $f$ and $a ^ { * }$

$$
0 = q ( s ) + \rho ( a ^ { \ast } ( s ) ) + p ( s ) \cdot \big ( f ( s ) + g ( s ) a ^ { \ast } ( s ) \big ) ,\tag{30}
$$

which gives

$$
q ( s ) = - \rho ( a ^ { * } ( s ) ) - p ( s ) \cdot \big ( f ( s ) + g ( s ) a ^ { * } ( s ) \big ) .\tag{31}
$$

With $f ( s ) = - K ( s ) p ( s )$ , the drift term $p ( s ) \cdot f ( s )$ is non-positive. The sign of the remaining term depends on the control cost; $q ( s )$ is computed from (31) and checked for nonnegativity. In the quadratic case this check is automatic, as shown in Section 7.6.

Step 5. Verification of the viscosity conditions. With $f , a ^ { * }$ , and q determined as above, verify the viscosity HJB condition of Definition 7.3.1.

• Subsolution. By construction, for the selected $p ( s ) \in \partial _ { C } V ( s )$ ，

$$
\operatorname* { i n f } _ { a \in \mathbb { A } } \left\{ c ( s , a ) + p ( s ) \cdot \left( f ( s ) + g ( s ) a \right) \right\} = 0 ,
$$

and $a ^ { * } ( s )$ attains the infimum. This follows directly from (29) and the definition of $q ( s )$

• Supersolution. For every $\tilde { p } \in \partial _ { C } V ( s )$

$$
\operatorname* { i n f } _ { a \in \mathbb { A } } \left\{ c ( s , a ) + \tilde { p } \cdot \left( f ( s ) + g ( s ) a \right) \right\} \geq 0 .
$$

This inequality is not automatic. It couples the choice of $K ( s )$ to the geometry of $\partial _ { C } V ( s )$ and to the input map $g ( s )$ . It must be checked for the specific V and the chosen $K ( s )$ . In the quadratic case, the check reduces to a lower bound on $K \colon$ this is derived in Section 7.6 and used in the worked example of Section 7.9. If the supersolution condition fails, one may enlarge $K ( s )$ and recompute $q ( s )$ ; Theorem 7.4.4 guarantees that a suficiently large $K ( s )$ exists whenever the coeficient of K in the supersolution inequality is positive.

Step 6. Well-posedness of the closed loop. The optimal feedback $a ^ { * } ( s )$ may be discontinuous on switching surfaces. The closed-loop dynamics $\dot { s } = f ( s ) + g ( s ) a ^ { * } ( s )$ is therefore interpreted as a diferential inclusion in the Filippov sense [32]. By Lemma 7.4.3, the right-hand side is measurable and locally bounded, so absolutely continuous Filippov solutions exist from any initial condition. This is what allows the generated fixture to be simulated even when the optimal policy jumps.

The procedure yields a triple $( f , g , c )$ satisfying all hypotheses of Theorem 7.4.4, which certifies that V is the optimal value function and $a ^ { * }$ is an optimal feedback. Sections 7.6–7.9 specialize the procedure to quadratic control costs and work the construction end-to-end for a two-dimensional piecewise-quadratic value function.

## 7.6 Construction with a Quadratic Control Cost

We now specialize the general procedure of Section 7.5 to the quadratic control cost used in the benchmark examples. Let

$$
\begin{array} { r } { \rho ( a ) = \frac { 1 } { 2 } a ^ { \top } R a , \qquad R \succ 0 , } \end{array}\tag{32}
$$

and assume the control space is unconstrained, $\mathbb { A } = \mathbb { R } ^ { m }$ . The compact case is discussed in the remark at the end of this subsection. The drift is chosen as in the general procedure,

$$
f ( s ) = - K ( s ) p ( s ) , \qquad K ( s ) \succ 0 ,\tag{33}
$$

where $p ( s ) \in \partial _ { C } V ( s )$ is the measurable selection of Step 2 in Section 7.5. Because $\rho$ is quadratic, the stationarity condition Ra $+ g ( s ) ^ { \top } p ( s ) = 0$ solves explicitly for the optimal action:

$$
\begin{array} { r } { { \boldsymbol { a } } ^ { * } ( { \boldsymbol { s } } ) = - R ^ { - 1 } { \boldsymbol { g } } ( { \boldsymbol { s } } ) ^ { \top } { \boldsymbol { p } } ( { \boldsymbol { s } } ) . } \end{array}\tag{34}
$$

Substituting (33) and (34) into the Hamiltonian optimality condition (30) and solving for q(s) gives

$$
\begin{array} { r l } & { q ( s ) = - \frac { 1 } { 2 } { a } ^ { * } ( s ) ^ { \top } R { a } ^ { * } ( s ) - p ( s ) \cdot \left( f ( s ) + g ( s ) { a } ^ { * } ( s ) \right) } \\ & { \qquad = \frac { 1 } { 2 } \left\| g ( s ) ^ { \top } p ( s ) \right\| _ { R ^ { - 1 } } ^ { 2 } + \left\| p ( s ) \right\| _ { K ( s ) } ^ { 2 } . } \end{array}\tag{35}
$$

The right-hand side is manifestly nonnegative because $R ^ { - 1 } \succ 0$ and $K ( s ) \succ 0$ . The state cost therefore satisfies the positivity requirement automatically.

Automatic satisfaction of the first two conditions. With the choices (33), (34), and (35), the stationarity condition (condition (i) of Lemma 7.4.2) holds by construction, and the Hamiltonian optimality condition (condition (ii)) holds because, for the selected $p ( s )$

$$
\operatorname* { i n f } _ { a \in \mathbb { R } ^ { m } } \left\{ c ( s , a ) + p ( s ) \cdot \left( f ( s ) + g ( s ) a \right) \right\} = c ( s , a ^ { * } ( s ) ) + p ( s ) \cdot \left( f ( s ) + g ( s ) a ^ { * } ( s ) \right) = 0 ,
$$

as a direct algebraic verification confirms. The only remaining requirement is the supersolution inequality (condition (iii) of Lemma 7.4.2).

Supersolution condition. For fixed s and $\tilde { p } \in \partial _ { C } V ( s )$ , define

$$
\begin{array} { r } { \kappa _ { \tilde { p } } ( a ) : = q ( s ) + \frac { 1 } { 2 } a ^ { \top } R a + \tilde { p } \cdot ( f ( s ) + g ( s ) a ) . } \end{array}
$$

This expression is strictly convex and coercive in $^ { a , }$ so it attains a unique minimum at

$$
\begin{array} { r } { { a } _ { \tilde { p } } ^ { * } ( s ) : = - { R } ^ { - 1 } { g } ( s ) ^ { \top } \tilde { p } . } \end{array}
$$

Substituting this minimizer and using (33) and (35) gives

$$
\begin{array} { r } { \underset { \boldsymbol { a } \in \mathbb { R } ^ { m } } { \operatorname* { i n f } } \kappa _ { \tilde { p } } ( \boldsymbol { a } ) = \frac { 1 } { 2 } \left\| g ( \boldsymbol { s } ) ^ { \top } p ( \boldsymbol { s } ) \right\| _ { R ^ { - 1 } } ^ { 2 } - \frac { 1 } { 2 } \left\| g ( \boldsymbol { s } ) ^ { \top } \tilde { p } \right\| _ { R ^ { - 1 } } ^ { 2 } + \left\| p ( \boldsymbol { s } ) \right\| _ { K ( \boldsymbol { s } ) } ^ { 2 } - \langle \tilde { p } , p ( \boldsymbol { s } ) \rangle _ { K ( \boldsymbol { s } ) } . } \end{array}\tag{36}
$$

The supersolution condition is $( 3 6 ) \geq 0$ for every $\tilde { p } \in \partial _ { C } V ( s )$

The case $\tilde { p } = p ( s ) . \quad \mathrm { A t } \ \tilde { p } = p ( s )$ , the right-hand side of (36) equals

$$
\begin{array} { r } { \frac { 1 } { 2 } \| g ^ { \top } p \| _ { R ^ { - 1 } } ^ { 2 } - \frac { 1 } { 2 } \| g ^ { \top } p \| _ { R ^ { - 1 } } ^ { 2 } + \| p \| _ { K } ^ { 2 } - \langle p , p \rangle _ { K } = 0 . } \end{array}
$$

The condition is therefore trivially satisfied at the selected gradient. The nontrivial constraint comes from the other elements of the Clarke subdiferential, i.e., from $\tilde { p } \in \partial _ { C } V ( s ) \backslash \{ p ( s ) \}$ . This is the analogue of the fact that, in the smooth case, the Bellman residual is identically zero at the optimal action and nonnegative at all others.

Scalar gain. Let $K ( s ) = k I$ with $k > 0$ . Then (36) becomes

$$
\begin{array} { r } { k \big ( \| p \| ^ { 2 } - \widetilde { p } \cdot p \big ) \geq \frac { 1 } { 2 } \big \| g ^ { \top } \widetilde { p } \big \| _ { R ^ { - 1 } } ^ { 2 } - \frac { 1 } { 2 } \big \| g ^ { \top } p \big \| _ { R ^ { - 1 } } ^ { 2 } \qquad \forall \widetilde { p } \in \partial _ { C } V ( s ) \setminus \{ p \} . } \end{array}\tag{37}
$$

The right-hand side does not depend on k. Suppose

$$
c _ { \operatorname* { m i n } } ( s ) : = \operatorname* { m i n } _ { \tilde { p } \in \partial _ { C } V ( s ) \backslash \{ p ( s ) \} } \left\{ \| p ( s ) \| ^ { 2 } - { \tilde { p } } \cdot p ( s ) \right\} > 0 .\tag{38}
$$

Then the coeficient of k is positive on the nontrivial part of the subdiferential, and (37) holds if and only if

$$
k \geq k _ { \operatorname* { m i n } } ( s ) : = \frac { \operatorname* { m a x } _ { \tilde { \boldsymbol { p } } \in \partial _ { C } V ( s ) \backslash \{ \boldsymbol { p } ( s ) \} } \left\{ \frac { 1 } { 2 } \| \boldsymbol { g } ( s ) ^ { \top } \tilde { \boldsymbol { p } } \| _ { R ^ { - 1 } } ^ { 2 } \right\} - \frac { 1 } { 2 } \| \boldsymbol { g } ( s ) ^ { \top } \boldsymbol { p } ( s ) \| _ { R ^ { - 1 } } ^ { 2 } } { c _ { \operatorname* { m i n } } ( s ) } .\tag{39}
$$

The numerator is finite because $\partial _ { C } V ( s )$ is compact. If (38) fails for the chosen s, then either the selected generalized gradient $p ( s )$ must be replaced by another element of $\partial _ { C } V ( s )$ for which the denominator is positive, or a state-dependent gain $K ( s )$ must be used. A state-dependent $K ( s )$ can be chosen so that the quadratic form in $\langle \tilde { p } , p \rangle _ { K }$ dominates the $R ^ { - 1 }$ -weighted diference in (36); the scalar bound (39) is the worst-case special case.

Summary and forward reference. The construction yields a certified triple $( f , g , c )$ once the supersolution lower bound (39) is satisfied. In smooth regions, $c _ { \mathrm { m i n } } ( s )$ is undefined (the subdiferential is a singleton), and the supersolution condition holds trivially. The constraint is therefore active only on switching surfaces. Section 7.9 instantiates the bound (39) for the two-dimensional piecewise-quadratic family, where $c _ { \mathrm { m i n } } ( s ) = 8 ( 1 - \alpha ^ { 2 } ) y ^ { 2 } > 0$ on the switching line for every $\alpha < 1$ and every $y \ne 0$

Remark (compact control sets). When $\mathbb { A } \subset \mathbb { R } ^ { m }$ is compact and the unconstrained minimizer $a ^ { * } ( s ) = - R ^ { - 1 } g ( s ) ^ { \top } p ( s )$ lies outside $\mathbb { A } ,$ the optimal action is the projection of $a ^ { * } ( s )$ onto $\mathbb { A } .$ , and the stationarity condition is replaced by a variational inequality on the normal cone of A at the projected point. The remaining construction is unchanged: the supersolution inequality must still be checked for every $\tilde { p } \in \partial _ { C } V ( s )$ , and the lower bound on K is obtained from the same computation with the projected action. The compact case is not needed for the benchmark examples in this paper and is left to future work.

## 7.7 Remarks on Problem Geometry and Multiple Switching Surfaces

Geometric interpretation via Lie derivatives. In continuous-time control-afine systems, the quantity $g ( s ) ^ { \top } p ( s )$ appearing in the optimal feedback (34) admits a natural geometric interpretation: it is a vector of Lie derivatives of $V$ along the columns of the input-coupling matrix. Let $g _ { 1 } ( s ) , \ldots , g _ { m } ( s )$ be the columns of $g ( s )$ . Each $g _ { i }$ describes the instantaneous direction in which the state moves when the i-th control input is applied. For a smooth $V ,$ the Lie derivative of V along $g _ { i }$ is the directional derivative

$$
\begin{array} { r } { \mathcal { L } _ { g _ { i } } V ( s ) = \nabla V ( s ) \cdot g _ { i } ( s ) , } \end{array}
$$

and the vector $\mathcal { L } _ { g } V ( s ) = ( \mathcal { L } _ { g _ { 1 } } V ( s ) , \ldots , \mathcal { L } _ { g _ { m } } V ( s ) ) ^ { \top }$ collects the rates of change of $V$ in all actuated directions. In smooth continuous-time converse optimality, the optimal feedback is

$$
\boldsymbol { a } ^ { * } ( \boldsymbol { s } ) = - \boldsymbol { R } ^ { - 1 } \mathcal { L } _ { g } \boldsymbol { V } ( \boldsymbol { s } ) = - \boldsymbol { R } ^ { - 1 } \boldsymbol { g } ( \boldsymbol { s } ) ^ { \top } \nabla \boldsymbol { V } ( \boldsymbol { s } ) ,\tag{40}
$$

see [17, 19]. The right-hand side is the steepest descent of V projected onto the subspace spanned by the columns of $g ( s )$ , weighted by the inverse of the control-cost matrix $R ^ { - 1 }$ . The same structure appears in the discrete-time construction of Section $5 ,$ where the optimal feedback $a ^ { \ast } ( s ) = - \gamma B ( s ) ^ { - 1 } g ( s ) ^ { \top } P f ( s )$ is the R<sup>−1</sup>-weighted projection of the value gradient onto the actuated subspace, with the metric $B ( s ) = R + \gamma g ( s ) ^ { \top } P g ( s )$ replacing $R .$

The Clarke extension. With a measurable selection $p ( s ) \in \partial _ { C } V ( s )$ , the scalar $p ( s ) \cdot g _ { i } ( s )$ is a selected Lie derivative drawn from the set

$$
{ \mathcal { L } } _ { g _ { i } } ^ { \mathrm { C l a r k e } } V ( s ) : = \{ \zeta \cdot g _ { i } ( s ) : \zeta \in \partial _ { C } V ( s ) \} .
$$

Because $\partial _ { C } V ( s )$ is compact and convex and $\zeta \mapsto \zeta \cdot g _ { i } ( s )$ is a linear functional, this set is a closed bounded interval in $\mathbb { R }$ . For a fixed selection $p ( s )$ , the i-th coordinate of the vector $g ( s ) ^ { \top } p ( s )$ is an element of ${ \mathcal { L } } _ { g _ { i } } ^ { \mathrm { C l a r k e } } V ( s )$ . The optimal feedback

$$
a ^ { * } ( s ) = - R ^ { - 1 } g ( s ) ^ { \top } p ( s )
$$

therefore steers the system opposite to the chosen projection of the generalized gradient onto the control subspace. On smooth regions the selection is unique and coincides with $\nabla V ( s )$ on a switching surface, $\partial _ { C } V ( s )$ is the convex hull of limiting gradients from adjacent smooth regions, so each ${ \mathcal { L } } _ { g _ { i } } ^ { \mathrm { C l a r k e } } V ( s )$ is the interval spanned by the corresponding classical Lie derivatives.

Worked example. For the piecewise-quadratic value of Section 7.9 with $g = I$ , the subdiferential on the switching line is $\partial _ { C } V ( 0 , y ) = \{ ( ( 2 \alpha - 1 ) 2 y , ~ 4 y ) : \alpha \in [ 0 , 1 ] \}$ . The first actuated direction gives the interval

$$
{ \mathcal { L } } _ { g _ { 1 } } ^ { \mathrm { C l a r k e } } V ( 0 , y ) = [ - 2 y , ~ 2 y ] ~ ( y > 0 ) ,
$$

of length $4 | y |$ , while the second actuated direction is a singleton, $\mathcal { L } _ { g _ { 2 } } ^ { \mathrm { C l a r k e } } V ( 0 , y ) = \{ 4 y \}$ . The width of the first interval quantifies the severity of the non-smoothness: it is zero at the origin, grows linearly away from the switching surface, and is the geometric quantity that the homotopy parameter α selects a point from. The optimal policy chooses a specific element of this interval, and varying α continuously varies the strength of the optimal feedback on the switching line while preserving global optimality.

Supersolution in Lie-derivative form. The supersolution condition of Lemma 7.4.2,

$$
\operatorname* { i n f } _ { a \in \mathbb { A } } \left\{ c ( s , a ) + \tilde { p } \cdot \left( f ( s ) + g ( s ) a \right) \right\} \geq 0 \qquad \forall \tilde { p } \in \partial _ { C } V ( s ) ,
$$

can also be written in Lie-derivative notation. In the quadratic case, using the explicit expression (36), we obtain

$$
\begin{array} { r l r l } & { - \frac { 1 } { 2 } \big \| \mathcal L _ { g } ^ { \tilde { p } } V \big \| _ { R ^ { - 1 } } ^ { 2 } + \frac { 1 } { 2 } \big \| \mathcal L _ { g } ^ { p } V \big \| _ { R ^ { - 1 } } ^ { 2 } + \| p \| _ { K } ^ { 2 } - \langle \tilde { p } , p \rangle _ { K } \geq 0 } & & { \forall \tilde { p } \in \partial _ { C } V ( s ) , } \end{array}\tag{41}
$$

where $\mathcal { L } _ { g } ^ { \tilde { p } } V = g ( s ) ^ { \top } \tilde { p }$ and $\begin{array} { r } { \mathcal { L } _ { g } ^ { p } V = g ( s ) ^ { \top } p ( s ) } \end{array}$ are the tested and the selected Lie-derivative vectors, and $\| \cdot \| _ { R ^ { - 1 } } , \| \cdot \| _ { K }$ are the corresponding weighted norms. This inequality has a clean geometric meaning: the actuated component of the selected gradient must not be much smaller than the largest actuated component available in the subdiferential. If it were, no gain $K ( s ) \succ 0$ would sufice, and the controller would be “too soft” on the switching surface. In the scalar-gain case $K = k I$ , this is exactly the requirement $c _ { \mathrm { m i n } } ( s ) > 0$ derived in Section 7.6: if the selected Lie derivative is close to the boundary of ${ \mathcal { L } } _ { g _ { i } } ^ { \mathrm { C l a r k e } } V ( s )$ , the denominator shrinks and the required k grows.

Multiple switching surfaces and parameterization. When the candidate value function has several switching surfaces (which may also intersect), the Clarke subdiferential at a point is the convex hull of the limiting gradients from all adjacent smooth regions. The corresponding Lie-derivative set becomes a polytope parametrized by a vector $\alpha \in [ 0 , 1 ] ^ { k }$ with k independent barycentric coordinates, where $k + 1$ is the number of adjacent smooth regions. The construction of Section 7.6 extends verbatim, since it depends only on the fact that $\partial _ { C } V ( s )$ is a compact convex set, and yields a k-parameter family

$$
\big \{ ( f _ { \alpha } , q _ { \alpha } , a _ { \alpha } ^ { * } ) \big \} _ { \alpha \in [ 0 , 1 ] ^ { k } } ,
$$

all sharing the same optimal value function V . Each component of α independently controls the sharpness of the non-smooth behavior along a particular switching surface, providing a multidimensional dificulty knob for benchmark generation. Whether the supersolution condition (41) holds for every α in the family is a geometric question: it depends on the relative orientation of the switching surfaces and on the spectrum of $g ( s ) ^ { \top } g ( s )$ . The construction is therefore not automatic in the multi-surface case, but the certificate logic is unchanged, and each candidate α can be verified by the same check as in the scalar case.

## 7.8 Problem Definition and Generator Pipeline

Each problem instance is a non-smooth, continuous-time, deterministic, control-afine system

$$
\begin{array} { r } { \dot { s } = f ( s ) + g ( s ) a , \qquad s \in \mathbb { R } ^ { n } , \ a \in \mathbb { R } ^ { m } , } \end{array}
$$

with separable running cost $c ( s , a ) \ : = \ : q ( s ) + \rho ( a )$ , where ${ \rho } ( a ) \ = \ \textstyle { \frac { 1 } { 2 } } a ^ { \top } R a$ with $R \succ 0 .$ , and $q ( s )$ is determined by (35). The drift is constructed as $f ( s ) = - K ( s ) p ( s )$ with a positive definite locally Lipschitz gain $K ( s ) \succ 0$ and a measurable selection $p ( s ) \in \partial _ { C } V ( s )$ . The optimal feedback is $a ^ { * } ( s ) = - R ^ { - 1 } g ( s ) ^ { \top } p ( s )$ , and the triple $( f , g , c )$ satisfies the viscosity HJB condition of Definition 7.3.1 by Theorem 7.4.4. The certified optimal value is the prescribed $V ,$ and the optimal policy is $\pi ^ { * } ( s ) = a ^ { * } ( s )$

A benchmark family is a parametric collection of such problems,

$$
{ \mathcal { F } } _ { \psi } = \{ M _ { \psi } : \psi \in \Psi \} ,
$$

where each instance $M _ { \psi } = ( f _ { \psi } , g _ { \psi } , q _ { \psi } , \rho , R , V _ { \psi } ^ { * } , \pi _ { \psi } ^ { * } )$ is a fully specified certified problem. The parameter space Ψ encodes tunable dificulty knobs—dimension, control authority, coupling strength, and non-smoothness severity through the homotopy parameters α—while preserving the analytical solution. The certified pair $( V _ { \psi } ^ { * } , \pi _ { \psi } ^ { * } )$ is derived from the construction, not sampled independently.

Algorithm 1 implements the six-step procedure of Section 7.5 for the quadratic control case of Section 7.6. It assumes the candidate V is piecewise quadratic with a finite number of smooth regions, so that the Clarke subdiferential admits an analytical characterization. For a codimension-one switching surface the selection parameter is a scalar $\alpha \in [ 0 , 1 ]$ ; for k independent switching surfaces it is a vector $\alpha \in [ 0 , 1 ] ^ { k }$

Two validation checks are worth emphasizing. The subsolution residual in (i) verifies that the selected gradient $p ( s )$ achieves a Hamiltonian infimum of zero, which is automatic by construction but sensitive to numerical implementation. The supersolution residual in (ii) verifies that no other element of $\partial _ { C } V ( s )$ produces a negative Hamiltonian, which is the only nontrivial constraint of the construction. In the scalar-gain case $K = k I$ , the residual in (ii) is controlled by the lower bound (39); enlarging k until the residual passes is the standard workflow. In the worked example of Section 7.9, (ii) holds for any $k > 0$ because the right-hand side of the general supersolution inequality is nonpositive on [0, 1], and the enlargement loop is not invoked. Validation details are reported in Section 7.10.

## 7.9 Example: Piecewise-Quadratic Value Function

We apply the construction procedure of Section 7.5 to a family of two-dimensional piecewisequadratic value functions. The family is simple enough to admit a full analytic characterization of the Clarke subdiferential and the supersolution condition, yet rich enough that its members span a continuum from smooth to sharply kinked. We present the general family first, then work the construction end-to-end for a specific configuration, and finally derive the general supersolution condition that identifies which members of the family are certified.

```latex
Algorithm 1 Converse-optimal benchmark generator (non-smooth, quadratic control).
Require: Family type, parameter ranges, number of instances N, gain type (constant or state
dependent), homotopy parameter vector $\alpha ,$ seeds.
1: for each instance $i = 1$ to N do
2: Sample parameter vector ψ from the family distribution $\mathcal { P } _ { \Psi }$
3: Extract $( Q , R , g , K )$ and, if applicable, the piecewise-quadratic matrices from ψ<sub>i</sub>.
4: Compute $\partial _ { C } V ( s )$ analytically on smooth regions and on each switching surface.
5: Choose a measurable selection $p ( s ) \in \partial _ { C } V ( s )$ , fixing the homotopy parameter α on each
switching surface.
6: Construct drift $f ( s ) = - K ( s ) p ( s )$ and optimal policy $a ^ { * } ( s ) = - R ^ { - 1 } g ( s ) ^ { \top } p ( s )$
7: Compute state cost $q ( s )$ via (35).
8: Validate on a grid G:
(i) Subsolution residual. ma $\begin{array} { r } { \mathrm { x } _ { s \in \mathcal { G } } \left| \operatorname* { i n f } _ { a } \{ c ( s , a ) + p ( s ) \cdot ( f ( s ) + g ( s ) a ) \} \right| < \varepsilon _ { \mathrm { s u b } } . } \end{array}$
(ii) Supersolution residual. min $. . \mathrm { { } } s \in { \mathcal { G } } \operatorname* { i n f } _ { \tilde { p } \in \partial _ { C } V ( s ) }$ in $\mathrm { f } _ { a } \{ c ( s , a ) + \tilde { p } \cdot ( f ( s ) + g ( s ) a ) \} > - \varepsilon _ { \mathrm { s u p } } .$
(iii) State cost nonnegativity. min $_ { \cdot s \in { \mathcal { G } } } q ( s ) \geq 0 .$
(iv) Closed-loop boundedness. Trajectories of $\dot { s } = f ( s ) + g ( s ) a ^ { * } ( s )$ from representative
initial states remain bounded over the validation horizon.
9: If (ii) fails, enlarge $K ( s )$ using the lower bound (39) and return to Step 5.
10: Export fixture (YAML) with parameters, seeds, validation metadata, and oracle evaluators
for $( V ^ { * } , \pi ^ { * } )$
11: end for
12: return Dataset of N validated fixtures.
```

The piecewise-quadratic family. Consider the control-afine system

$$
\dot { s } = f ( s ) + a , \qquad a \in \mathbb { R } ^ { 2 } , \qquad g ( s ) \equiv I ,
$$

with quadratic control cost $\begin{array} { r } { \rho ( a ) = \frac { 1 } { 2 } \| a \| ^ { 2 } , \mathrm { i . e . , } R = I } \end{array}$ . The state-dependent input matrix is set to the identity to keep the algebra transparent; the general case g(s) follows from the same construction with g carried through the pullback formula of Section 7.6. Let the candidate value function be

$$
V ( s ) = \left\{ \begin{array} { l l } { x ^ { 2 } + 2 \beta x y + 2 y ^ { 2 } , } & { x \ge 0 , } \\ { 2 x ^ { 2 } + 2 \gamma x y + 2 y ^ { 2 } , } & { x < 0 , } \end{array} \right.\tag{42}
$$

with parameters $| \beta | < \sqrt { 2 }$ and $| \gamma | < 2 .$ . These bounds are exactly the positive-definiteness conditions on the two quadratic pieces: the right-hand matrix $P _ { + } = { \binom { 1 } { \beta } } \beta \atop { 2 } \qquad $ has determinant $2 - \beta ^ { 2 }$ , and the left-hand matrix $P _ { - } = \left( \begin{array} { l l } { { 2 } } & { { \gamma } } \\ { { \gamma } } & { { 2 } } \end{array} \right)$ has determinant $4 - \gamma ^ { 2 }$

Each quadratic piece is strictly convex. The two pieces agree at $x = 0 \ ( V ( 0 , y ) = 2 y ^ { 2 } )$ , so V is continuous. It is diferentiable everywhere except on the switching line $x = 0$ , where the gradient jumps. It is not globally convex when $\beta \neq \gamma ;$ the construction uses only local Lipschitz continuity and the Clarke subdiferential, not convexity. The kink sharpness is controlled by $| \beta - \gamma | \colon$ : when $\beta = \gamma ,$ the family reduces to the smooth quadratic $V ( s ) = x ^ { 2 } + 2 \beta x y + 2 y ^ { 2 }$ on both sides.

Step 1: Subdiferential analysis. The Clarke subdiferential of V is

$$
\partial _ { C } V ( s ) = \left\{ \begin{array} { l l } { \{ ( 2 x + 2 \beta y , \ 2 \beta x + 4 y ) ^ { \top } \} , } & { x > 0 , } \\ { \{ ( 4 x + 2 \gamma y , \ 2 \gamma x + 4 y ) ^ { \top } \} , } & { x < 0 , } \\ { \mathrm { c o } \{ ( 2 \beta y , \ 4 y ) ^ { \top } , \ ( 2 \gamma y , \ 4 y ) ^ { \top } \} , } & { x = 0 . } \end{array} \right.
$$

On the switching line, the subdiferential can be parametrized by $\alpha \in [ 0 , 1 ]$ as

$$
\begin{array} { r } { \partial _ { C } V ( 0 , y ) = \{ \left( 2 ( \alpha \beta + ( 1 - \alpha ) \gamma ) y , \ 4 y \right) ^ { \top } | \alpha \in [ 0 , 1 ] \} . } \end{array}\tag{43}
$$

Let $c _ { \alpha } : = \alpha \beta + ( 1 - \alpha ) \gamma$ , so that $\partial _ { C } V ( 0 , y ) = \{ ( 2 c _ { \alpha } y , 4 y ) ^ { \top } : \alpha \in [ 0 , 1 ] \}$ . The extreme values $\alpha = 1$ and $\alpha = 0$ correspond to the right-hand and left-hand limiting gradients $p ^ { + } = ( 2 \beta y , 4 y )$ and $p ^ { - } = ( 2 \gamma y , 4 y )$

Step 2: Selection. We select the right-hand limiting gradient on the switching line, i.e., the selection

$$
\begin{array} { r } { p ( s ) = \left\{ \begin{array} { l l } { ( 2 x + 2 \beta y , 2 \beta x + 4 y ) ^ { \top } , } & { x > 0 , } \\ { ( 4 x + 2 \gamma y , 2 \gamma x + 4 y ) ^ { \top } , } & { x < 0 , } \\ { ( 2 \beta y , 4 y ) ^ { \top } , } & { x = 0 . } \end{array} \right. } \end{array}\tag{44}
$$

This corresponds to $\alpha = 1$ in (43). Other selections are treated in the homotopy discussion below and enter the general supersolution condition derived below.

Step 3: Drift. Choose $K ( s ) = k I$ with $k > 0 ,$ , so that $f ( s ) = - k p ( s )$ . Since $p ( s ) \cdot f ( s ) =$ $- k \| p ( s ) \| ^ { 2 } \leq 0$ , V decreases along the uncontrolled dynamics; the gain k quantifies how much of the stabilization is done by the natural dynamics rather than by the control. In the explicit formulas below we fix $k = 1$

Step 4: Optimal feedback and state cost. Since $R = I$ , the stationarity condition reduces to $a + p ( s ) = 0$ , giving $a ^ { * } ( s ) = - p ( s )$ . Substituting into the Hamiltonian optimality condition and solving for $q ( s )$ yields

$$
\begin{array} { r } { q ( s ) = \left( \frac { 1 } { 2 } + k \right) \| p ( s ) \| ^ { 2 } . } \end{array}\tag{45}
$$

The right-hand side is manifestly nonnegative, so the state cost is admissible for any $k > 0$

Step 5: Viscosity verification. On smooth regions x $\neq 0$ , the subdiferential is a singleton and the classical HJB equation holds by construction. On the switching line $x = 0$ , the subsolution property holds by construction; the nontrivial condition is the supersolution. We compute it in general below and specialize to the worked configuration afterward.

General supersolution condition. Using the general expression (36) with $g = I , R = I$ $K = k I$ , and substituting $p = ( 2 c _ { \alpha } y , 4 y )$ and $\widetilde { p } = ( 2 c _ { \alpha ^ { \prime } } y , 4 y )$ for fixed $\alpha , \alpha ^ { \prime } \in [ 0 , 1 ]$ , gives

$$
\operatorname* { i n f } _ { a \in \mathbb { R } ^ { 2 } } \kappa _ { \widetilde { p } } ( a ) = 2 y ^ { 2 } \left( c _ { \alpha } - c _ { \alpha ^ { \prime } } \right) \big [ ( 1 + 2 k ) c _ { \alpha } + c _ { \alpha ^ { \prime } } \big ] ,\tag{46}
$$

where $c _ { \alpha } = \alpha \beta { + } ( 1 { - } \alpha ) \gamma$ and $c _ { \alpha ^ { \prime } } = \alpha ^ { \prime } \beta + ( 1 { - } \alpha ^ { \prime } ) \gamma$ . The supersolution condition of Definition 7.3.1 requires (46) to be nonnegative for every $\alpha ^ { \prime } \in [ 0 , 1 ]$

When the selection is $\alpha = 1 \ ( \sec \alpha = \beta )$ and $\beta \geq \gamma ,$ the factor $c _ { \alpha } - c _ { \alpha ^ { \prime } }$ is nonnegative for all $\alpha ^ { \prime } { \mathrm { . } }$ , and the condition reduces to

$$
( 1 + 2 k ) \beta + c _ { \alpha ^ { \prime } } \geq 0 \qquad \forall c _ { \alpha ^ { \prime } } \in [ \gamma , \beta ] .\tag{47}
$$

The left-hand side is linear in $c _ { \alpha ^ { \prime } } .$ , so it attains its minimum on the interval $[ \gamma , \beta ]$ at $c _ { \alpha ^ { \prime } } = \gamma$ The condition therefore simplifies to the scalar inequality

$$
( 1 + 2 k ) \beta \geq - \gamma .\tag{48}
$$

This is the general certificate condition for the family (42): any $( \beta , \gamma , k )$ satisfying $| \beta | < { \sqrt { 2 } } ,$ $| \gamma | < 2 , k > 0$ , and (48) yields a certified system with value function V and optimal feedback $a ^ { * } = - p$ . In the smooth case $\beta = \gamma , ( 4 8 )$ reduces to $( 1 + 2 k ) \beta \geq - \beta$ , which is $( 2 + 2 k ) \beta \geq 0 ;$ for $\beta > 0$ this holds automatically, and for $\beta = 0$ the configuration is the isotropic quadratic.

For the case $\beta < \gamma$ , the analysis is symmetric: the factor $c _ { \alpha } - c _ { \alpha ^ { \prime } }$ is nonpositive, and the condition becomes $( 1 + 2 k ) \beta + c _ { \alpha ^ { \prime } } \leq 0$ for all $c _ { \alpha ^ { \prime } } \in [ \beta , \gamma ]$ , giving $( 1 + 2 k ) \beta \leq - \gamma .$ . We state the condition only for $\beta \geq \gamma$ , which covers all configurations used in the numerical experiments.

Specific worked configuration $( \beta = 1 , \gamma = - 1 , k = 1 , \alpha = 1 )$ . Take $\beta = 1$ and $\gamma = - 1$ Both satisfy the positive-definiteness constraints $( | \beta | < \sqrt { 2 } , | \gamma | < 2 )$ . The certificate condition (48) becomes $( 1 + 2 k ) \geq 1$ , which holds for any $k > 0 ;$ with $k = 1$ the margin is large. The candidate value function is

$$
V ( s ) = \left\{ { { x } ^ { 2 } } + 2 x y + 2 y ^ { 2 } , \quad x \geq 0 , \right.\tag{49}
$$

The Clarke subdiferential on the switching line is

$$
\partial _ { C } V ( 0 , y ) = \{ ( 2 ( 2 \alpha - 1 ) y , ~ 4 y ) ^ { \top } : \alpha \in [ 0 , 1 ] \} ,
$$

with $\alpha = 1$ giving the right-hand limiting gradient $( 2 y , 4 y ) ^ { \top }$ and $\alpha = 0$ the left-hand limiting gradient $( - 2 y , 4 y ) ^ { \top }$ . The selected gradient on the switching line is $p ( 0 , y ) = ( 2 y , 4 y ) ^ { \top }$ . The drift, optimal feedback, and state cost are

$$
\begin{array} { r l r l r } { f ( 0 , y ) = - ( 2 y , 4 y ) ^ { \top } , } & { } & { a ^ { \ast } ( 0 , y ) = - ( 2 y , 4 y ) ^ { \top } , } & { } & { q ( 0 , y ) = \frac { 3 } { 2 } \| p ( 0 , y ) \| ^ { 2 } , } \end{array}
$$

and on the smooth regions $f$ and $a ^ { * }$ coincide with $- p ( s )$ with $k = 1$ . The resulting closed-loop matrices are

$$
P _ { 1 } = \left( { 2 \atop 2 } \& \right) , \qquad P _ { 2 } = \left( { 4 \atop - 2 } \& \right) ,
$$

with $f = - P _ { 1 } s$ for $x > 0$ and $f = - P _ { 2 } s$ for $x < 0$ . Both $- P _ { 1 }$ and $- P _ { 2 }$ are Hurwitz, so the closed loop is locally asymptotically stable on each side. Substituting $\beta = 1 , \gamma = - 1$ into (46) reproduces the specific form

$$
\operatorname* { i n f } _ { a } \kappa _ { \tilde { p } } ( a ) = 8 ( 1 - \alpha ^ { \prime 2 } ) y ^ { 2 } \geq 0 ,
$$

where $\tilde { p } = ( 2 ( 2 \alpha ^ { \prime } - 1 ) y , 4 y ) ^ { \top }$ . The inequality holds with equality at $\alpha ^ { \prime } = 1 \ ( \mathrm { w h e r e } \ \tilde { p } = p )$ and is strict for $\alpha ^ { \prime } < 1$

Homotopy parameterization. The parameter α selects an element of the Clarke subdiferential on the switching line and generates a family of admissible systems that all share the same value function V . For fixed $\alpha ,$ the drift, optimal control, and state cost on the switching line are

$$
\begin{array} { r } { f _ { \alpha } ( 0 , y ) = - k p _ { \alpha } ( 0 , y ) , \qquad a _ { \alpha } ^ { * } ( 0 , y ) = - p _ { \alpha } ( 0 , y ) , \qquad q _ { \alpha } ( 0 , y ) = \left( \frac { 1 } { 2 } + k \right) \lVert p _ { \alpha } ( 0 , y ) \rVert ^ { 2 } , } \end{array}
$$

with $p _ { \alpha } ( 0 , y ) = ( 2 c _ { \alpha } y , 4 y ) ^ { \top }$ . The drift and cost on the smooth regions are unchanged. Every member of the family is certified whenever the pair $( \beta , \gamma , k )$ satisfies (48) with the selected $\alpha$ in place of the right-hand limit. In particular, the limiting cases $\alpha = 1$ and $\alpha = 0$ correspond to right-continuous and left-continuous drifts, and intermediate values interpolate between them, all preserving the same value function and the same global optimum.

Explicit formulas $( \beta = 1 , \gamma = - 1 , k = 1 , \alpha = 1 )$ . Collecting the expressions above,

$$
f ( { \boldsymbol { s } } ) = { \left\{ \begin{array} { l l } { - ( 2 x + 2 y , ~ 2 x + 4 y ) ^ { \top } , } & { x \geq 0 , } \\ { - ( 4 x - 2 y , ~ - 2 x + 4 y ) ^ { \top } , } & { x < 0 , } \end{array} \right. } \qquad { \boldsymbol { f } } ( 0 , y ) = - ( 2 y , ~ 4 y ) ^ { \top } ,
$$

$$
a ^ { * } ( s ) = \left\{ \begin{array} { l l } { - ( 2 x + 2 y , 2 x + 4 y ) ^ { \top } , } & { x > 0 , } \\ { - ( 4 x - 2 y , - 2 x + 4 y ) ^ { \top } , } & { x < 0 , } \\ { - ( 2 y , 4 y ) ^ { \top } , } & { x = 0 , } \end{array} \right.
$$

$$
q ( s ) = \left\{ \begin{array} { l l } { \frac { 3 } { 2 } \big ( ( 2 x + 2 y ) ^ { 2 } + ( 2 x + 4 y ) ^ { 2 } \big ) , } & { x \geq 0 , } \\ { \frac { 3 } { 2 } \big ( ( 4 x - 2 y ) ^ { 2 } + ( - 2 x + 4 y ) ^ { 2 } \big ) , } & { x < 0 , } \end{array} \right. \quad \quad c ( s , a ) = q ( s ) + \frac { 1 } { 2 } \| a \| ^ { 2 } .
$$

The closed-loop dynamics $\dot { s } = - ( k + 1 ) p ( s )$ inherits the discontinuity across $x \ = \ 0$ and is interpreted in the Filippov sense; Lemma 7.4.3 guarantees existence of absolutely continuous solutions from any initial condition.

Relation to the numerical experiments. The numerical experiments of Section 7.12 sample the family (42) with configurations $( \beta , \gamma , k )$ reported in Table 4. The certificate condition (48) is satisfied by the smooth baseline $\beta = \gamma = 0$ , by the matched-kink configurations $( \beta , \gamma ) = ( 1 , - 1 )$ and $( 0 . 7 , - 0 . 7 )$ , and by the intermediate configuration. One reported configuration, $( \beta , \gamma , k ) =$ $( 1 . 4 , - 1 . 9 , 0 . 0 1 )$ , sits at the boundary of the admissible region: substituting into (48) gives $( 1 + 0 . 0 2 ) \cdot 1 . 4 = 1 . 4 2 8$ against a required 1.9, so the scalar-gain certificate is not satisfied with the reported parameters. We therefore exclude this boundary configuration from the certified results in the numerical subsection and treat it as an open case in the discussion of future work.

## 7.10 Validation

Each generated instance undergoes three validation checks on a grid G that includes points on the switching line $x = 0$ . The checks verify the three nontrivial conditions of the non-smooth certificate: the subsolution equality on the selected gradient, the supersolution inequality over the full Clarke subdiferential, and well-posedness of the closed loop. A fixture is exported only if all three pass.

1. Subsolution residual. For each test point $s \in \mathcal G$ , evaluate

$$
H _ { p } ( s ) : = \operatorname* { i n f } _ { a \in \mathbb { R } ^ { 2 } } \{ q ( s ) + { \frac { 1 } { 2 } } \| a \| ^ { 2 } + p ( s ) \cdot ( f ( s ) + a ) \} ,
$$

where $p ( s )$ is the selected generalized gradient used in the generator. By construction, $H _ { p } ( s ) = 0$ . Numerically we verify

$$
\operatorname* { m a x } _ { s \in \mathcal { G } } | H _ { p } ( s ) | < 1 0 ^ { - 1 2 } .
$$

This is a sanity check on the implementation of $f , q ,$ and the selection $p ;$ it is automatic in exact arithmetic.

2. Supersolution residual. The nontrivial condition of the certificate is that for every $\tilde { p } \in \partial _ { C } V ( s )$ 2

$$
\begin{array} { r } { H _ { \widetilde { p } } ( s ) : = \displaystyle \operatorname* { i n f } _ { a \in \mathbb { R } ^ { 2 } } \left\{ q ( s ) + \frac { 1 } { 2 } \| a \| ^ { 2 } + \widetilde { p } \cdot \left( f ( s ) + a \right) \right\} \geq 0 . } \end{array}
$$

We evaluate the worst-case residual over a finite sampling of the extreme points of $\partial _ { C } V ( s )$ (on the switching line, the two limiting gradients $p ^ { + }$ and $p ^ { - } :$ ; on smooth regions, the unique gradient). The instance passes if

$$
\operatorname* { m i n } _ { s \in \mathcal { G } } \operatorname* { m i n } _ { \tilde { p } \in \partial _ { C } ^ { \mathrm { s a m p l e } } V ( s ) } H _ { \tilde { p } } ( s ) > - \varepsilon _ { \mathrm { s u p } } ,
$$

with $\varepsilon _ { \mathrm { s u p } } = 1 0 ^ { - 1 0 }$ . On smooth regions, this condition holds with equality at $\tilde { p } = p ( s )$ and is strictly positive at every other extreme. On the switching line, the residual is controlled analytically by the certificate condition (48); instances that fail this condition are rejected before export, as in the dataset construction below.

3. Boundedness. The closed-loop system $\dot { s } = f ( s ) + a ^ { * } ( s )$ is integrated over a fixed time horizon using a standard ODE solver. Trajectories from a set of representative initial states remain within a bounded region, confirming that the optimal policy stabilizes the system and that the numerical implementation is well-behaved. Instances whose trajectories leave the validation domain or accumulate numerical instability are rejected.

We do not check $B ( s ) = R + g ( s ) ^ { \top } P ( s ) g ( s ) \succ 0$ . That condition belongs to the discrete-time quadratic-Gaussian construction of Section 5, where it ensures strict convexity of the Bellman integrand and positive definiteness of the discounted metric $H _ { \gamma }$ . In continuous time, strict convexity of the control cost is guaranteed by $R \succ 0$ alone, and no discounted metric is involved.

## 7.11 Dataset Construction

A dataset of certified benchmark instances is built by sampling the family parameters from specified distributions and rejecting any instance that fails the certificate. For the piecewisequadratic family of Section 7.9, the parameters are the kink coeficients $( \beta , \gamma )$ , the passive drift gain k, and the selection parameter α on the switching line:

$$
\begin{array} { r } { \beta \sim \mathcal { U } ( - \sqrt { 2 } , \sqrt { 2 } ) , \qquad \gamma \sim \mathcal { U } ( - 2 , 2 ) , \qquad k \sim \mathcal { U } ( 0 . 2 , 2 . 0 ) , \qquad \alpha \sim \mathcal { U } ( 0 , 1 ) . } \end{array}
$$

The bounds $| \beta | < \sqrt { 2 }$ and $| \gamma | < 2$ ensure that both quadratic pieces $P ^ { + }$ (with determinant $2 - \beta ^ { 2 } )$ and P<sup>−</sup> (with determinant $4 - \gamma ^ { 2 } )$ are positive definite.

A sampled instance is accepted only if it satisfies the certificate condition (48). For the right-continuous selection $\alpha = 1$ used in the numerical experiments, this is the scalar inequality $( 1 + 2 k ) \beta \geq - \gamma$ . For a general selection parameter $\alpha ,$ the same computation gives a condition linear in $c _ { \alpha } = \alpha \beta + ( 1 - \alpha ) \gamma ;$ a sampled instance is accepted if the resulting supersolution inequality holds at both $\tilde { p } = p ^ { + }$ and $\tilde { p } = p$ <sup>−</sup> (the two extreme points of the Clarke subdiferential on the switching line). Instances that fail are discarded and resampled. This rejection ensures that every exported fixture carries a valid non-smooth optimality certificate.

For each accepted instance, the piecewise expressions for $f , a ^ { * }$ , and $q$ are computed analytically and exported as a YAML fixture, together with the parameter vector, deterministic seeds for evaluation, and reference evaluators for V and $a ^ { * }$ . The resulting collection provides a testbed for reinforcement learning algorithms on a continuum of certified non-smooth optimal control tasks, spanning the range from smooth $( \beta = \gamma )$ to sharply kinked $( | \beta - \gamma |$ large) with controlled passive drift and homotopy selection.

## 7.12 Numerical Experiments on the Non-Smooth Benchmark

We now report numerical experiments on a discretized instantiation of the non-smooth benchmark. The RL agent interacts with an explicit Euler discretization of the continuous-time dynamics at $\Delta t = 0 . 0 5 \mathrm { s }$ over a horizon of 200 steps. This mirrors the Mode I setup of Section 4: the continuous-time certificate is exact, while the digital learner experiences a discretization whose step size is part of the experimental variable.

Experimental setup. The evaluation uses the piecewise-quadratic family (42), with the rightcontinuous selection $\alpha = 1$ on the switching line. Five configurations $( \beta , \gamma , k )$ were evaluated, listed in Table 4.
<table><tr><td>Config.</td><td>β</td><td>γ k</td></tr><tr><td>C0</td><td>0.0 0.0</td><td>0.2</td></tr><tr><td>C1</td><td>1.0</td><td>-1.0 0.1</td></tr><tr><td>C2</td><td>1.0 -1.0</td><td>1.0</td></tr><tr><td>C3</td><td>0.7 -0.7</td><td>0.2</td></tr><tr><td>C4</td><td>1.4 -1.9</td><td>0.01</td></tr></table>

Table 4: Certified configurations of the piecewise-quadratic family. The first four satisfy (1 + $2 k ) \beta \geq - \gamma$ from (48). The smooth baseline $\beta = \gamma = 0$ is certified trivially.

Each algorithm is trained for 30,000 environment steps, corresponding to 150 episodes, with no warm-up phase. This budget is deliberately limited to expose diferences in sample eficiency. The policy network is a two-hidden-layer MLP with 64 units per layer and tanh activations, shared across all algorithms. Evaluation uses a paired protocol with 15 fixed initial states, each evaluated under the deterministic greedy policy. Five independent training seeds per algorithm– configuration pair yield 5 estimates, and the per-seed metrics are averaged over the 15 initial states. Confidence intervals are obtained by bootstrap resampling (1,000 samples).

The metrics are the normalized optimality gap and the absolute regret of Section 5.4. Since the benchmark is deterministic and the horizon is long enough for the system to converge, the regret coincides with the raw excess cost $J ^ { \pi } ( s _ { 0 } ) - V ^ { * } ( s _ { 0 } )$

Aggregate results. Table 5 reports aggregates. The table below reports the four certified configurations listed in Table 4, with confidence intervals.

Interpretation. Of-policy methods with replay (SAC and DDPG) achieve the lowest gaps and regrets, followed by TD3, while the on-policy methods (PPO and A2C) are significantly worse. Two efects drive this ordering. First, of-policy algorithms reuse past transitions many times within the 30,000-step budget, efectively seeing a larger dataset; on-policy methods discard each rollout after a single update. Under the discontinuous optimal policy that the nonsmooth configurations produce, this sample-eficiency advantage matters: the policy must discover large actions near the switching line, and a replay bufer concentrates updates in the regions where the optimal feedback jumps. Second, the clipped surrogate objective of PPO and the synchronous updates of A2C are less efective at representing the sharp jump in the optimal control than the entropy-regularized exploration of SAC or the deterministic policy gradient of DDPG.

The smooth baseline C0 shows the smallest gap between on-policy and of-policy methods, confirming that the divergence is driven by the interaction between the kink and the algorithm’s update rule, not by the algorithms’ general performance on smooth control tasks. The medium configuration C3 sits between the extremes, and its intermediate gap confirms that both the kink sharpness $| \beta - \gamma |$ and the passive gain k act as independent dificulty knobs. The matched-kink pair C1 and C2 difer only in k (0.1 versus 1.0), and the improvement from C1 to C2 for every algorithm is consistent with the interpretation that larger passive drift reduces the burden on the learned controller.

<table><tr><td>Algorithm</td><td>Normalized OptGap</td><td>Regret</td></tr><tr><td>SAC</td><td>1.05 [0.73, 1.36]</td><td>2.39 [1.96, 2.83]</td></tr><tr><td>DDPG</td><td>1.18 [0.80, 1.55]</td><td>2.70 [2.17, 3.23]</td></tr><tr><td>TD3</td><td>5.04 [3.05, 7.03]</td><td>7.64 [5.22, 10.07]</td></tr><tr><td>PPO</td><td>5.14 [2.06, 8.23]</td><td>13.39 [7.80, 18.99]</td></tr><tr><td>A2C</td><td>12.97 [0, 26.47]</td><td>22.06 [3.99, 40.11]</td></tr></table>

Table 5: Aggregate non-smooth benchmark results over the certified configurations C0–C4. Mean and 95% CI via paired bootstrap (5 seeds × 15 initial states per configuration). Lower is better for both metrics.

Normalized OptGap (average over configs)  
![](images/777852e4a70fac6f84336f4cd3e5d9a3460c13b3d8e993a6bb31f122fd17f35f.jpg)  
Figure 8: Normalized optimality gap for each algorithm on the non-smooth mobile-robot benchmark. Error bars indicate 95% bootstrap confidence intervals.

Relation to the stochastic benchmarks. The non-smooth benchmark reveals a ranking that difers from the stochastic smooth families of Sections 5 and 6. On ConverseArm with fixed initial states and on NVDEx, PPO is the strongest algorithm; on ConverseQuad, PPO and SAC lead. Here, SAC and DDPG lead and PPO is substantially worse. This is not a contradiction: the non-smooth configurations are deterministic and have a discontinuous optimal policy, whereas the stochastic families are smooth and have continuous optimal policies. The two benchmark classes therefore probe diferent failure modes of the same algorithms, and the certified optimum allows both to be measured in the same optimality units.

(a)  
![](images/e475567c8034e066e1157991172ee8ce9aeabffbc7d94edf7994ba2f08f6a1ef.jpg)

![](images/2a582b27a20f9fefb95841f90b86785133104ec13af9fd08b528d7b79398011f.jpg)  
(b)  
Figure 9: Compact heatmap summaries. (a) Normalized optimality gap across the parameter space; darker shades indicate larger suboptimality. (b) Regret for the same fized parameters; darker shades correspond to higher cumulative excess cost.

## 7.13 Remark on Higher-Dimensional Extensions

This piecewise-quadratic family example demonstrates that the non-smooth theoretical framework can be turned into a concrete benchmark generator. Extending the pipeline to non-smooth value functions of higher dimension (e.g., piecewise quadratic with multiple switching surfaces) and to discrete-time dynamics is the subject of ongoing work.

## 8 Extension III: Partially Observed MDPs

The fixtures of Sections 4–7 expose the state to the learner. In many safety-critical applications— robotic manipulation under occlusion, vehicle control with intermittent GPS, aerial navigation with noisy sensors—the learner receives only observations, and the underlying state is hidden. This section extends the ground-truth-first framework to partially observed evaluation. The central idea is to replace the physical state s in the converse construction with an information state $b ,$ the control-theoretic analogue of a belief state: a statistic of the observation–action history that is suficient for dynamic programming. The certificate is then constructed on $b ,$ and the optimal output-feedback policy is certified in the learner’s own information set rather than in a hidden latent state.

The extension has two layers. The first layer is a fully observed Markov decision process (MDP) in b, to which the converse constructions of Sections 5–6 apply without modification. The second layer is an output process that generates observations and updates b recursively. The learner sees only observations and rewards; the private evaluator maintains b and applies the certified feedback $a ^ { * } ( b )$ . This separation is what allows partially observed benchmarks to inherit the certified-optimum property of their fully observed counterparts.

A full-state oracle that observes s is generally not an admissible reference for an outputfeedback learner, because it may exploit information the learner cannot access. The same initial belief and cost objective decompose as

$$
\underbrace { J ( \pi ) - J _ { \mathrm { f u l l } } ^ { * } } _ { \mathrm { t o t a l ~ d i f f e r e n c e } } = \underbrace { J ( \pi ) - J _ { \mathrm { o u t } } ^ { * } } _ { \mathrm { l e a r n i n g ~ g a p } } + \underbrace { J _ { \mathrm { o u t } } ^ { * } - J _ { \mathrm { f u l l } } ^ { * } } _ { \mathrm { i n f o r m a t i o n ~ p e n a l t y } } ,\tag{50}
$$

where $J _ { \mathrm { o u t } } ^ { * }$ is the optimal value over output-feedback policies that use only the learner’s history. Reporting the total diference against a full-state oracle conflates learning error with an information penalty the learner cannot remove. The constructions below certify $J _ { \mathrm { o u t } } ^ { * }$ directly, and the LQG reference of Section 8.6 supplies an analytic $J _ { \mathrm { f u l l } } ^ { * }$ so that the information penalty can be measured separately.

## 8.1 Information States and Reduction

Let $X _ { k }$ be a hidden state, $O _ { k }$ an observation, $A _ { k }$ an action, and $C _ { k } \geq 0$ a stage cost. Let

$$
H _ { k } = ( O _ { 0 } , A _ { 0 } , C _ { 0 } , O _ { 1 } , \dots , A _ { k - 1 } , C _ { k - 1 } , O _ { k } )
$$

denote the observation–action–cost history available to the learner before choosing $A _ { k }$ . An output-feedback policy is a measurable map $A _ { k } = \pi _ { k } ( H _ { k } )$ . Including past costs in the history matters whenever the learner receives rewards, since the reward channel can carry information about the hidden state.

Throughout the partially observed extension, all dynamics, costs, and observation models are time-homogeneous, and all policies are stationary. The Bellman equations below are stated in their stationary form under this assumption.

Definition 8.1.1 (Suficient information state). A measurable map $B _ { k } = \phi _ { k } ( H _ { k } )$ is a suficient information state $i f ,$ under every admissible policy, there exist common kernels $K _ { k }$ and cost functions $c _ { k }$ such that

$$
\operatorname* { P r } \bigl ( B _ { k + 1 } \in { \cal D } \mid H _ { k } , A _ { k } \bigr ) = K _ { k } ( { \cal D } \mid B _ { k } , A _ { k } ) , \qquad \mathbb { E } \bigl [ C _ { k } \mid H _ { k } , A _ { k } \bigr ] = c _ { k } ( B _ { k } , A _ { k } )\tag{51}
$$

for every measurable set D and every admissible $( H _ { k } , A _ { k } )$

Definition 8.1.1 is the control-theoretic version of the belief-state reduction of a POMDP. For a classical POMDP with hidden state $X _ { k }$ , the canonical information state is the conditional density $b _ { k } ( x ) = p _ { X _ { k } \mid H _ { k } } ( x \mid h _ { k } )$ . In finite-state POMDPs, $b _ { k }$ is a probability vector on the simplex; in linear-Gaussian problems, it is the Kalman filter’s conditional mean and covariance; in general nonlinear continuous-state models, it is an infinite-dimensional object. The construction below is exact whenever $B _ { k }$ is finite-dimensional and Markov; when it is not, the benchmark must be built on a deliberately designed finite-dimensional information state, as discussed in Section 8.4.

Theorem 8.1.1 (Information-state reduction). Let $B _ { k }$ be a suficient information state in the sense of Definition 8.1.1. Then the optimal output-feedback control problem is equivalent to a fully observed $M D P$ with state $B _ { k }$ . In particular, the optimal value can be written as a function $v ^ { * } ( b )$ satisfying

$$
v ^ { * } ( b ) = \operatorname* { m i n } _ { a \in \mathbb { R } ^ { m } } \Bigl \{ c ( b , a ) + \gamma \int _ { \mathbb { R } } v ^ { * } ( b _ { + } ) p _ { b } ( b _ { + } \mid b , a ) d b _ { + } \Bigr \} .\tag{52}
$$

Ifa measurable minimizer $\pi ^ { * } ( b )$ exists, then the output-feedback policy $a _ { k } = \pi ^ { * } ( \phi _ { k } ( h _ { k } ) )$ is optimal among all nonanticipative output-feedback policies.

The proof is in Appendix E. The essential point is that no new dynamic-programming principle is needed: Definition 8.1.1 says that the transition kernel and the expected stage cost depend on the history only through $B _ { k }$ , so the backward-induction argument for the fully observed Bellman equation applies verbatim once s is replaced by b.

## 8.2 Converse Optimality on the Information State

Theorem 8.1.1 reduces the output-feedback problem to a fully observed Markov problem on the information state. We can therefore apply the converse constructions of Sections 5–6 directly to b. Throughout this subsection, we assume that the information-state dynamics is afine in the virtual input,

$$
B _ { k + 1 } = f ( B _ { k } ) + g ( B _ { k } ) A _ { k } + \Omega _ { k + 1 } ,\tag{53}
$$

where $\{ \Omega _ { k } \}$ is i.i.d., zero-mean, and has finite covariance $\Sigma _ { b }$ , and that the stage cost is separable,

$$
c ( b , a ) = q ( b ) + \rho ( a ) ,\tag{54}
$$

with $q \geq 0$ and $\rho$ strictly convex. These are the same structural assumptions used for the fully observed stochastic construction; they are now stated on b rather than on s.

Theorem 8.2.1 (Converse optimality on the information state). Let the information-state dynamics (53) and cost (54) satisfy the assumptions above, and let $v ^ { * } : \mathbb { R } ^ { d } $ R be a candidate value function satisfying the same regularity and integrability conditions as in the fully observed stochastic construction (Section 5). If for every $b \in \mathbb { R } ^ { d }$ there exists a policy $\pi ^ { * } ( b )$ such that

$$
\pi ^ { * } ( b ) \ i s \ a \ B e l l m a n \ a r g m i n : \pi ^ { * } ( b ) \in \arg \operatorname* { m i n } _ { a \in \mathbb R ^ { m } } \Bigl \{ \rho ( a ) + \gamma \mathbb E \bigl [ v ^ { * } ( f ( b ) + g ( b ) a + \Omega ) \bigr ] \Bigr \} ,\tag{55}
$$

$$
\boldsymbol { v } ^ { * } ( b ) = \boldsymbol { q } ( b ) + \rho ( \pi ^ { * } ( b ) ) + \gamma \mathbb { E } \big [ \boldsymbol { v } ^ { * } ( \boldsymbol { f } ( b ) + \boldsymbol { g } ( b ) \pi ^ { * } ( b ) + \Omega ) \big ] ,\tag{56}
$$

then $v ^ { * }$ is the optimal information-state value function, and the output-feedback policy $A _ { k } =$ $\pi ^ { * } ( B _ { k } )$ is optimal among all output-feedback policies.

The proof applies the Bellman-operator contraction argument of the fully observed stochastic theorem (Appendix C) on the information state. Because $B _ { k }$ is a suficient statistic by Definition 8.1.1, the value function of the output-feedback problem coincides with the value function of the Markov problem on $b ,$ and optimality in the latter transfers to the former.

## 8.3 Quadratic Information-State Construction

The most useful specialization is again quadratic. Choose

$$
\begin{array} { r } { q ( b ) = b ^ { \top } Q b , \qquad \rho ( a ) = a ^ { \top } R a , \qquad v ^ { * } ( b ) = b ^ { \top } P b + v _ { 0 } , } \end{array}\tag{57}
$$

where $P \succ 0 , R \succ 0 , Q \succeq 0$ , and $P - Q \succeq 0$ . Let

$$
B _ { \gamma } ( b ) = R + \gamma g ( b ) ^ { \top } P g ( b ) , \qquad H _ { \gamma } ( b ) = \gamma \big ( P ^ { - 1 } + \gamma g ( b ) R ^ { - 1 } g ( b ) ^ { \top } \big ) ^ { - 1 } .\tag{58}
$$

Then, by the same completion-of-squares calculation as in the fully observed stochastic case, the optimal information-state feedback is

$$
a ^ { * } ( b ) = - \gamma B _ { \gamma } ( b ) ^ { - 1 } g ( b ) ^ { \top } P f ( b ) ,\tag{59}
$$

and the Bellman identity reduces to the energy equation

$$
\begin{array} { r } { \boldsymbol { b } ^ { \top } ( \boldsymbol { P } - \boldsymbol { Q } ) \boldsymbol { b } = \boldsymbol { f } ( \boldsymbol { b } ) ^ { \top } \boldsymbol { H } _ { \gamma } ( \boldsymbol { b } ) \boldsymbol { f } ( \boldsymbol { b } ) . } \end{array}\tag{60}
$$

The value ofset is $\begin{array} { r } { v _ { 0 } = \frac { \gamma } { 1 - \gamma } \operatorname { t r } ( P \Sigma _ { b } ) } \end{array}$ . Any drift of the metric-normalized form

$$
f ( b ) = H _ { \gamma } ( b ) ^ { - 1 / 2 } S ( b ) ( P - Q ) ^ { 1 / 2 } b , \qquad S ( b ) ^ { \top } S ( b ) = I ,\tag{61}
$$

satisfies (60), so the same parameterizations as in Section 5 apply with s replaced by b.

Corollary 8.3.1 (Quadratic information-state benchmark). Suppose $B _ { k }$ is a finite-dimensional suficient information state satisfying (53), and the cost is quadratic as in (57). Choose any input matrix field $g ( b )$ and any orthogonal field $S ( b )$ , and define f by (61). Then $v ^ { * } ( b ) = b ^ { \top } P b + v _ { 0 }$ with $\begin{array} { r } { v _ { 0 } = \frac { \gamma } { 1 - \gamma } \operatorname { t r } ( P \Sigma _ { b } ) } \end{array}$ is the optimal information-state value function, and the optimal outputfeedback controller is $A _ { k } = a ^ { * } ( B _ { k } )$ with $a ^ { * }$ given by (59).

The corollary is the POMDP analogue of the QG specialization used for ConverseArm, NVDEx, and ConverseQuad. Every construction of Section 5.3 and Section 6.1 carries over once the physical state is replaced by a suficient information state, and the certificate logic is unchanged.

## 8.4 Realizing the Information State as an Output System

The construction above is stated in terms of b. For a learner-facing benchmark, b must be reconstructed from the observation–action–reward history rather than observed directly. This requires an output realization: a pair $( \eta , \tau )$ where $\eta ( o \mid b , a )$ is an observation density and

$$
b _ { + } = \tau ( b , a , o _ { + } )\tag{62}
$$

is a deterministic filter recursion. The induced transition kernel on b is

$$
\kappa ( D \mid b , a ) = \int _ { \mathbb { O } } \mathbb { 1 } \left\{ \tau ( b , a , o ) \in D \right\} \eta ( o \mid b , a ) d o ,\tag{63}
$$

and the realization is compatible with (53) if this kernel equals the one induced by the noise density $p _ { \Omega }$

Proposition 8.4.1 (Innovation realization). Let the target information-state dynamics be (53) with noise density $p _ { \Omega }$ . Take the observation space $\mathbb { O } = \mathbb { R } ^ { d }$ , reveal the innovation $O _ { k + 1 } = \Omega _ { k + 1 }$ and define the filter update

$$
\tau ( b , a , o ) = f ( b ) + g ( b ) a + o .\tag{64}
$$

Then the pair $( \eta , \tau )$ with $\eta ( o \mid b , a ) = p _ { \Omega } ( o )$ realizes (53).

The innovation realization is mathematically clean and gives an exact output process whose information state is finite-dimensional by design. It does not, however, model a physically natural hidden-state process: it is a deliberately designed benchmark whose suficient statistic is known in closed form. This is the design principle of the POMDP extension. The construction is not the most general nonlinear POMDP, and it does not claim a nonlinear separation theorem; it constructs a benchmark for which the optimal output-feedback policy is certified in the learner’s own information set.

What the learner sees. The environment maintains $b _ { k }$ internally. At reset, the learner receives $o _ { 0 }$ and the environment returns rewards $r _ { k } = - c ( b _ { k } , a _ { k } )$ . The field info returned by the environment must not contain $b _ { k }$ , the drift $f ,$ the input map g, the matrices $P , Q , R .$ , the orthogonal field S, or the filter update τ. If the source code of the environment is available to the learner, a strong student or an LLM can recover the information-state recursion; a serious benchmark therefore keeps the final evaluation seeds and the private filter parameters outside the learner’s notebook.

Making the interface less transparent. The innovation realization is exact but transparent if $f , g .$ , and τ are revealed. A less transparent variant composes the observation with a smooth bijection $\chi : \mathbb { R } ^ { d }  \mathbb { R } ^ { d }$

$$
o _ { k + 1 } = \chi ( \Omega _ { k + 1 } ) , \qquad \tau ( b , a , o ) = f ( b ) + g ( b ) a + \chi ^ { - 1 } ( o ) .\tag{65}
$$

The induced kernel on b is unchanged, so the certificate is preserved.

## 8.5 Benchmark Protocol

A student-facing POMDP benchmark is implemented as follows. The environment stores $b _ { k }$ internally and returns only $o _ { k }$ , the reward $r _ { k } = - c ( b _ { k } , a _ { k } )$ , and termination flags. It never returns $b _ { k }$ . The private evaluator computes the certified action $a ^ { * } ( b _ { k } )$ from (59) and the certified value $v ^ { * } ( b _ { k } ) = b _ { k } ^ { \top } P b _ { k } + v _ { 0 }$

The learner-facing API is

$$
o _ { 0 } = \mathrm { r e s e t } ( ) , \qquad ( o _ { k + 1 } , r _ { k } , \mathrm { d o n e , i n f o } ) = \mathrm { s t e p } ( a _ { k } ) ,\tag{66}
$$

where the info dictionary may contain diagnostic signals but must not contain $b _ { k }$ or any of the certificate parameters. The evaluation objective is the discounted cost

$$
J ^ { \pi } ( b _ { 0 } ) = \mathbb { E } _ { \pi } \Big [ \sum _ { k = 0 } ^ { T - 1 } \gamma ^ { k } c ( B _ { k } , A _ { k } ) \Big | B _ { 0 } = b _ { 0 } \Big ] ,\tag{67}
$$

truncated at horizon $T$ . The certified reference is

$$
J ^ { * } ( b _ { 0 } ) = v ^ { * } ( b _ { 0 } ) - \gamma ^ { T } \mathbb { E } \big [ v ^ { * } ( B _ { T } ) \ | \ B _ { 0 } = b _ { 0 } , A _ { k } = a ^ { * } ( B _ { k } ) \big ] ,\tag{68}
$$

and the normalized relative suboptimality is

$$
\mathrm { R S } ( b _ { 0 } , \pi ) = \frac { J ^ { \pi } ( b _ { 0 } ) - J ^ { * } ( b _ { 0 } ) } { | J ^ { * } ( b _ { 0 } ) | + \epsilon } .\tag{69}
$$

CRN are used as in Section 5.4: all evaluated policies see the same start beliefs and the same innovation streams.

A useful decomposition follows directly from (50). When a full-state reference is available (for example, from a linear-Gaussian hidden model), the evaluator reports both $J ^ { \pi } - J _ { \mathrm { o u t } } ^ { * }$ and the analytic information penalty $J _ { \mathrm { o u t } } ^ { * } \mathrm { ~ - ~ } J _ { \mathrm { f u l l } } ^ { * }$ separately. This separates learning error from information loss and is what distinguishes an information-consistent POMDP benchmark from one whose oracle exploits hidden information.

## 8.6 LQG Reference and Information Penalty

To make the information-penalty decomposition quantitative, consider the noisy-sensor LQG reference

$$
X _ { k + 1 } = A X _ { k } + G A _ { k } + W _ { k + 1 } , \qquad Y _ { k } = C X _ { k } + V _ { k } ,\tag{70}
$$

with process covariance $W = 1 0 ^ { - 4 } I _ { ☉ }$ , sensor variance $\sigma _ { y } ^ { 2 } ,$ quadratic stage cost $X ^ { \top } Q X + R A ^ { 2 }$ discount $\gamma \in ( 0 , 1 )$ , and parameters

$$
A = \left( { \begin{array} { c c } { 1 } & { 0 . 2 } \\ { 0 } & { 1 } \end{array} } \right) , \quad G = \left( { 0 . 0 2 } \right) , \quad C = I , \quad Q = \mathrm { d i a g } ( 1 , 0 . 1 ) , \quad R = 0 . 1 .
$$

The Kalman filter is the finite-dimensional information state $b _ { k } = ( \mu _ { k } , E )$ , where $\mu _ { k } = \operatorname { \mathbb { E } } [ X _ { k } \mid$ $H _ { k } ]$ and E is the stationary posterior covariance solving

$$
\Xi = A E A ^ { \top } + { \cal W } , \qquad E = \Xi - \Xi C ^ { \top } ( C \Xi C ^ { \top } + \sigma _ { y } ^ { 2 } ) ^ { - 1 } C \Xi .
$$

The controller Riccati equation $P = Q + \gamma A ^ { \top } P A - \gamma ^ { 2 } A ^ { \top } P G ( R + \gamma G ^ { \top } P G ) ^ { - 1 } G ^ { \top } P A$ determines the optimal full-state value

$$
V _ { \mathrm { f u l l } } ^ { * } ( x ) = x ^ { \top } P x + \frac { \gamma } { 1 - \gamma } \operatorname { t r } ( P W ) .
$$

Averaged over the same initial belief, $\begin{array} { r } { \mathbb { E } V _ { \mathrm { f u l l } } ^ { * } ( X _ { 0 } ) = \mu _ { 0 } ^ { \top } P \mu _ { 0 } + \mathrm { t r } ( P E ) + \frac { \gamma } { 1 - \gamma } \mathrm { t r } ( P W ) } \end{array}$ . The optimal output-feedback value satisfies

$$
v ^ { * } ( \mu ) = \mu ^ { \top } P \mu + \mathrm { t r } ( P E ) + \frac { \gamma } { 1 - \gamma } \mathrm { t r } ( P W ) + \frac { \gamma } { 1 - \gamma } \mathrm { t r } ( P K ^ { \top } M K E ) ,
$$

where K is the LQG gain and $M = R + \gamma G ^ { \top } P G$ . The information penalty is therefore

$$
J _ { \mathrm { o u t } } ^ { * } - J _ { \mathrm { f u l l } } ^ { * } = \frac { \gamma } { 1 - \gamma } \mathrm { t r } \big ( P K ^ { \top } M K E \big ) ,\tag{71}
$$

a closed-form expression that scales with the posterior covariance $E ,$ i.e., with sensor uncertainty.

The exact result is Theorem 8.2.1: if $B _ { k }$ is a finite-dimensional suficient information state with afine dynamics (53), the quadratic construction of Section 8.3 certifies $v ^ { * }$ and $a ^ { * }$ exactly, and the output-feedback policy is optimal in the learner’s information class. What the construction does not do is solve arbitrary nonlinear continuous-state POMDPs. For a general hidden-state model, the belief density $x \mapsto p _ { X _ { k } | H _ { k } } ( x \mid h _ { k } )$ is infinite-dimensional, and the value function is a functional on densities rather than a matrix on $\mathbb { R } ^ { d }$ . The matrix formulas of this section apply only after a finite-dimensional information state has been specified. This is a design principle, not a defect: a POMDP benchmark is useful precisely when its suficient statistic is known, and the construction makes the choice explicit.

Forward link. The three extensions of Sections 6–8 each change one element of the framework: the action interface and disturbance law in Extension I, the regularity class of the value function in Extension II, and the observation interface in Extension III. The certificate logic is unchanged.

## 9 Discussion

The framework trades broad ecological coverage for calibration: converse-generated fixtures provide known-answer control problems with measurable optimality gaps, not replacements for open-ended simulators or real-world platforms. The families studied here exhibit nonlinear drift, state-dependent actuation, delayed input channels, and stif closed-loop behaviour, with dificulty tunable via dimension, dynamic extension, coupling strength, control authority, and disturbance structure. They remain constructed tasks; strong performance indicates diagnostic capability under certified dynamics, not deployment readiness.

What each extension covers, and what it does not. Extension I (Section 6) shows that the certificate survives a nonlinear raw-action coordinate and a non-Gaussian finite-covariance disturbance law. What it does not cover is state-dependent, heteroscedastic, or temporally correlated noise, all of which would break the moment-only argument behind (22). The reported ConverseQuad results are a single configuration in η and ϵ; a systematic sweep over actuator nonlinearity and mixture weight is left to future work.

Extension II (Section 7) gives a viscosity–Clarke certificate for value functions that are locally Lipschitz but not diferentiable. The theory is continuous-time; the numerical experiments of Section 7.12 use an explicit Euler discretization, which inherits the discretization confound of Mode I. A native discrete-time non-smooth construction would eliminate this confound, but the discrete-time analogue of the viscosity inequality is not standard and is left open. The supersolution condition couples the gain matrix K to the geometry of the Clarke subdiferential, and the scalar-gain version requires the denominator (38) to be positive.

Extension III (Section 8) constructs a finite-dimensional information-state benchmark whose optimal output-feedback policy is certified in the learner’s own information set. The construction does not solve arbitrary nonlinear continuous-state POMDPs. For a general hidden-state model, the belief density is infinite-dimensional, and the value function is a functional on densities rather than a matrix on R<sup>d</sup>. The innovation realization of Proposition 8.4.1 is exact but not physically natural: it is a deliberately designed benchmark whose suficient statistic is known by construction, not a solution to the general partial-observability problem.

## 10 Conclusion

We presented a ground-truth-first framework for reinforcement learning benchmarking based on converse optimality. The central idea is to make the optimum part of the benchmark specification rather than an unknown target outside it. This supports two complementary modes: continuoustime generators for sensitivity and robustness studies, and native discrete-time fixtures for oraclereferenced comparison by optimality gap, regret, Bellman residual, and policy error. Continuoustime and discrete-time generators play distinct roles: the former provide physically interpretable probes but inherit simulator-step dependence, while the latter remove timestep ambiguity for absolute comparison.

The paper combines a review of the converse-optimality lineage with three extensions of the certificate logic. Extension I extends the construction to non-afine raw actions and finitecovariance non-Gaussian disturbances, with the ConverseQuad aerial family as the concrete instantiation. Extension II replaces the smooth HJB certificate with a viscosity–Clarke condition, enabling certified benchmarks with kinked value functions and switching surfaces, and reports a discrete-time instantiation on a piecewise-quadratic family. Extension III reduces partially observed evaluation to converse optimality on a suficient information state, certifies the optimal output-feedback policy in the learner’s information class, and supplies an analytic LQG information penalty that separates learning error from information loss.

## References

[1] Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, and David Meger. Deep reinforcement learning that matters. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018.

[2] Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, and Marc G. Bellemare. Deep reinforcement learning at the edge of the statistical precipice. In Advances in Neural Information Processing Systems (NeurIPS), volume 34, 2021.

[3] Andrew Patterson, Samuel Neumann, Martha White, and Adam White. Empirical design in reinforcement learning. Journal of Machine Learning Research, 25(318):1–63, 2024.

[4] David Balduzzi, Karl Tuyls, Julien Perolat, and Thore Graepel. Re-evaluating evaluation. Advances in Neural Information Processing Systems, 31, 2018.

[5] Scott M Jordan, Adam White, Bruno Castro Da Silva, Martha White, and Philip S Thomas. Position: Benchmarking is limited in reinforcement learning research. arXiv preprint arXiv:2406.16241, 2024.

[6] Marc G Bellemare, Yavar Naddaf, Joel Veness, and Michael Bowling. The arcade learning environment: An evaluation platform for general agents. Journal of artificial intelligence research, 47:253–279, 2013.

[7] Emanuel Todorov, Tom Erez, and Yuval Tassa. Mujoco: A physics engine for model-based control. In 2012 IEEE/RSJ international conference on intelligent robots and systems, pages 5026–5033. IEEE, 2012.

[8] Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. Openai gym. arXiv preprint arXiv:1606.01540, 2016.

[9] Karl Cobbe, Chris Hesse, Jacob Hilton, and John Schulman. Leveraging procedural generation to benchmark reinforcement learning. In International conference on machine learning, pages 2048–2056. PMLR, 2020.

[10] Maxime Chevalier-Boisvert, Bolun Dai, Mark Towers, Rodrigo Perez-Vicente, Lucas Willems, Salem Lahlou, Suman Pal, Pablo Samuel Castro, and J K Terry. Minigrid & miniworld: Modular & customizable reinforcement learning environments for goal-oriented tasks. Advances in Neural Information Processing Systems, 36:73383–73394, 2023.

[11] Ian Osband, Yotam Doron, Matteo Hessel, John Aslanides, Eren Sezener, Andre Saraiva, Katrina McKinney, Tor Lattimore, Csaba Szepesvari, Satinder Singh, et al. Behaviour suite for reinforcement learning. arXiv preprint arXiv:1908.03568, 2019.

[12] Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine. D4RL: Datasets for deep data-driven reinforcement learning. arXiv preprint arXiv:2004.07219, 2020.

[13] Richard Bellman. Dynamic Programming. Princeton University Press, Princeton, NJ, 1957.

[14] Frank L Lewis, Draguna Vrabie, and Kyriakos G Vamvoudakis. Reinforcement learning and feedback control: Using natural decision methods to design optimal adaptive controllers. IEEE Control Systems Magazine, 32(6):76–105, 2012.

[15] Andrew Y Ng, Stuart Russell, et al. Algorithms for inverse reinforcement learning. In Icml, volume 1, page 2, 2000.

[16] Pieter Abbeel and Andrew Y Ng. Apprenticeship learning via inverse reinforcement learning. In Proceedings of the twenty-first international conference on Machine learning, page 1, 2004.

[17] Vesna Nevistic and James A Primbs. Constrained nonlinear optimal control: a converse hjb approach. California Institute of Technology, Pasadena, CA, 91125:96–021, 1996.

[18] Thomas G¨ohrt, Pavel Osinenko, and Stefan Streif. Converse optimality for discrete-time systems. IEEE Transactions on Automatic Control, 65(5):2257–2264, 2019.

[19] Grigory Yamerenko, Sinan Ibrahim, Francisco Moreno-Mora, Pavel Osinenko, and Stefan Streif. Generating informative benchmarks for reinforcement learning. IEEE Control Systems Letters, 2025.

[20] Sinan Ibrahim, Gr´egoire Ouerdane, Hadi Salloum, Henni Ouerdane, Stefan Streif, and Pavel Osinenko. Benchmarking reinforcement learning via stochastic converse optimality: Generating systems with known optimal policies. arXiv preprint arXiv:2603.17631, 2026.

[21] Michael G Crandall and Pierre-Louis Lions. Viscosity solutions of hamilton-jacobi equations. Transactions of the American mathematical society, 277(1):1–42, 1983.

[22] Martino Bardi, Italo Capuzzo Dolcetta, et al. Optimal control and viscosity solutions of Hamilton-Jacobi-Bellman equations, volume 12. Springer, 1997.

[23] Frank H Clarke. Optimization and nonsmooth analysis. SIAM, 1990.

[24] Francis Clarke. Nonsmooth analysis in control theory: a survey. European Journal of Control, 7(2-3):145–159, 2001.

[25] Aleksei Fedorovich Filippov. Diferential equations with discontinuous right-hand side. Matematicheskii sbornik, 93(1):99–128, 1960.

[26] Richard D Smallwood and Edward J Sondik. The optimal control of partially observable markov processes over a finite horizon. Operations research, 21(5):1071–1088, 1973.

[27] Leslie Pack Kaelbling, Michael L Littman, and Anthony R Cassandra. Planning and acting in partially observable stochastic domains. Artificial intelligence, 101(1-2):99–134, 1998.

[28] Matthew J Hausknecht and Peter Stone. Deep recurrent q-learning for partially observable mdps. In AAAI fall symposia, volume 45, page 141, 2015.

[29] Steven Kapturowski, Georg Ostrovski, John Quan, Remi Munos, and Will Dabney. Recurrent experience replay in distributed reinforcement learning. In International conference on learning representations, 2018.

[30] Steven Morad, Ryan Kortvelesy, Matteo Bettini, Stephan Liwicki, and Amanda Prorok. Popgym: Benchmarking partially observable reinforcement learning. arXiv preprint arXiv:2303.01859, 2023.

[31] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[32] AF Filippov. Existence and general properties of solutions of discontinuous systems. In Diferential Equations with Discontinuous Righthand Sides, pages 48–122. Springer, 1988.

[33] Wendell H Fleming and H Mete Soner. Controlled Markov processes and viscosity solutions. Springer, 2006.

[34] Steven E Shreve and Dimitri P Bertsekas. Universally measurable policies in dynamic programming. Mathematics of Operations Research, 4(1):15–30, 1979.

[35] Charalambos D Aliprantis and Kim C Border. Infinite dimensional analysis: a hitchhiker’s guide. Springer, 2006.

[36] A. F. Filippov. Diferential Equations with Discontinuous Righthand Sides. Kluwer, 1988.

## A Reproducibility and Released Assets

The framework is intended to be reproducible as a seedable benchmark generator rather than only as a fixed list of hand-authored environments. The public project page: https: $: / / \mathrm { g t f r } ]$ github.io/Benchmarking-of-Reinforcement-Learning-via-Converse-Optimality/ hosts the paper, code links, fixtures, experimental summaries, and visual diagnostics. A complete release contains generator code, random seeds for sampled fixtures, validation grids, accepted and rejected instance metadata, CRN evaluation schedules, baseline configurations, software versions, and scripts for reproducing the tables.

For the experiments summarized in the main text, the relevant reproducibility objects are the certified fixture parameters, the validation tolerance, the horizon, the start-state distribution, the disturbance streams, and the baseline hyperparameter files. Since the oracle is explicit, independent reproduction can verify both the reported returns and the Bellman consistency of each exported environment.

## B Impact Statement

This work aims to improve the reliability and interpretability of RL experimentation. Its main positive impact is methodological: it can help researchers distinguish relative performance from distance to optimality and design more diagnostic experiments. Potential negative impact is limited but not absent. If known-answer fixtures are overused as narrow leaderboards, they could encourage overfitting to generated families or create unwarranted confidence about deployment readiness. We therefore frame the method as a complement to broader benchmarks and realworld validation, not a substitute for them.

## C Proofs for the Smooth Theory

A benchmark fixture is useful only if we can verify that a given pair $( V , \pi )$ is indeed optimal. The natural verification tool is the Bellman residual. For any candidate value function $V .$

$$
\mathcal { R } _ { V } ( s , a ) : = c ( s , a ) + \gamma \int _ { S } V ( s ^ { \prime } ) P ( \mathrm { d } s ^ { \prime } \mid s , a ) - V ( s ) .
$$

The residual measures how much V fails to satisfy the Bellman equation at a given state-action pair. If $V = V ^ { \star }$ , then in $\mathrm { f } _ { a } \mathscr { R } _ { V } ( s , a ) = 0$ for all s and the infimum is attained by any optimal action. More generally, the residual provides a pointwise certificate that can be checked without solving the full MDP. Proposition 1 makes this precise.

Definition C.0.1 (Bellman operators). For a given $M D P \left( S , \mathcal { A } , P , c , \gamma \right)$ , we define the optimal Bellman operator $T$ and the policy evaluation operator $T ^ { \pi }$ as follows. For any function $V$ : ${ \mathcal { S } }  \mathbb { R }$

$$
( T V ) ( s ) : = \operatorname* { i n f } _ { a \in \mathcal { A } } \Big \{ c ( s , a ) + \gamma \int _ { S } ^ { V } ( s ^ { \prime } ) P ( \mathrm { d } s ^ { \prime } | s , a ) \Big \} , \quad ( T ^ { \pi } V ) ( s ) : = c ( s , \pi ( s ) ) + \gamma \int _ { S } ^ { V } ( s ^ { \prime } ) P ( \mathrm { d } s ^ { \prime } | s , \pi ( s ) ) .
$$

The Bellman operators formalize two fundamental questions in dynamic programming. The optimal Bellman operator $T$ asks: “What is the smallest total cost that can be achieved from state s, assuming that the future cost from the next state onward is given by $V ? ^ { \mathfrak { g } }$ . The true optimal value function $V ^ { \star }$ is the unique fixed point of this operator, i.e. $V ^ { \star } = T V ^ { \star } ;$ this is the Bellman equation. The policy evaluation operator $T ^ { \pi }$ asks: “If I follow policy π forever, what total cost will I incur from state s if the future cost is evaluated according to $V ? ^ { \mathfrak { g } }$ . The value function of $\pi$ is its unique fixed point: $V ^ { \pi } = T ^ { \pi } V ^ { \pi }$

Proposition C.0.1 (Optimality certificate). Let $\mathbb { B } _ { w }$ be the weighted sup-norm Banach space defined by $\mathbb { B } _ { w } : = \{ V : \mathcal { S }  \mathbb { R }$ measurable $\| V \| _ { w } < \infty \}$ , with $\begin{array} { r } { \| V \| _ { w } = \operatorname* { s u p } _ { s \in \mathcal { S } } \frac { | V ( s ) | } { w ( s ) } } \end{array}$ and w some suitable weight function. Let $T$ and $T ^ { \pi }$ be the Bellman operators from $ D e f i n i t i o n ~ 1 ,$ well-defined on $\mathbb { B } _ { w . }$ , and admitting unique fixed points $V ^ { \star }$ and $V ^ { \pi }$ . For any locally Lipschitz function $V \in \mathbb { B } _ { w }$ and any admissible measurable policy $\pi _ { ; }$ , the following are equivalent:

1. $V = V ^ { \star } = V ^ { \pi } ; i . e .$ , V is the true optimal value function and π is an optimal policy.

2. For all $s \in { \mathcal { S } }$ and all $a \in A ,$ , we have $\mathcal { R } _ { V } ( s , \pi ( s ) ) = 0$ and $\mathcal { R } _ { V } ( s , a ) \geq 0$

Proof. We prove the equivalence.

$( 1 ) \Rightarrow ( 2 )$ . Assume that $V = V ^ { \star } = V ^ { \pi }$ , and that π is an optimal policy. Since $V ^ { \star }$ is the optimal value function, it satisfies the Bellman equation

$$
V ^ { \star } ( s ) = \operatorname* { i n f } _ { a \in { \mathcal A } } \left\{ c ( s , a ) + \gamma \int _ { \mathcal S } V ^ { \star } ( s ^ { \prime } ) P ( \mathrm { d } s ^ { \prime } \mid s , a ) \right\} .
$$

Therefore, for every $( s , a ) \in \mathcal S \times \mathcal A$

$$
c ( s , a ) + \gamma \int _ { S } V ^ { \star } ( s ^ { \prime } ) P ( \mathrm { d } s ^ { \prime } \mid s , a ) - V ^ { \star } ( s ) \geq 0 ,
$$

that is, $\mathcal { R } _ { V } ( s , a ) \geq 0$ . Since π is optimal and $V ^ { \pi } = V ^ { \star }$ , the policy $\pi$ attains the Bellman infimum at every state:

$$
V ^ { \star } ( s ) = c ( s , \pi ( s ) ) + \gamma \int _ { S } V ^ { \star } ( s ^ { \prime } ) { \cal P } ( \mathrm { d } s ^ { \prime } \mid s , \pi ( s ) ) .
$$

Hence, $\mathcal { R } _ { V } ( s , \pi ( s ) ) = 0$ . Thus condition (2) holds.

$( 2 ) \Rightarrow ( 1 )$ . Assume now that for all $s \in { \mathcal { S } }$ and all $a \in \mathcal { A } , \mathcal { R } _ { V } ( s , a ) \geq 0$ , and $\mathcal { R } _ { V } ( s , \pi ( s ) ) = 0$ By definition of the residual, the inequality $\mathcal { R } _ { V } ( s , a ) \geq 0$ means that

$$
V ( s ) \leq c ( s , a ) + \gamma \int _ { S } V ( s ^ { \prime } ) P ( \mathrm { d } s ^ { \prime } \mid s , a ) \qquad \forall a \in \mathcal { A } .
$$

Taking the infimum over a gives $V ( s ) \leq ( T V ) ( s )$ . On the other hand, evaluating the residual at the action $\pi ( s )$ and using $\mathcal { R } _ { V } ( s , \pi ( s ) ) = 0$ , we obtain

$$
V ( s ) = c ( s , \pi ( s ) ) + \gamma \int _ { \cal S } V ( s ^ { \prime } ) P ( \mathrm { d } s ^ { \prime } \mid s , \pi ( s ) ) = ( T ^ { \pi } V ) ( s ) .
$$

Since

$$
( T V ) ( s ) = \operatorname* { i n f } _ { a \in { \cal A } } \left\{ c ( s , a ) + \gamma \int _ { \cal S } V ( s ^ { \prime } ) { \cal P } ( \mathrm { d } s ^ { \prime } \mid s , a ) \right\} ,
$$

and the action $a = \pi ( s )$ achieves equality, we conclude that $( T V ) ( s ) = ( T ^ { \pi } V ) ( s ) = V ( s )$ Therefore, $V = T V$ and $V = T ^ { \pi } V$

The Bellman operators $T$ and $T ^ { \pi }$ each admit unique fixed points. Hence V must coincide simultaneously with the optimal value function and the value function associated with $\pi ,$ so that $V = V ^ { \star } = V ^ { \pi }$ . Therefore π is an optimal policy and condition (1) holds. This completes the proof. □

Condition 2 of Proposition 1 requires the residual to vanish on the candidate policy and to be non-negative everywhere else. Together they force V to be the fixed point of the optimal Bellman operator and π to attain the minimum. When V is smooth and the action space is unconstrained, the residual inequality reduces to the classical Hamilton–Jacobi–Bellman equation together with a first-order stationarity condition; when V is only locally Lipschitz, the same logic applies with Clarke subdiferentials replacing classical gradients (see Appendix D). With the certificate in hand, we can now define the central object of the benchmarking framework.

## C.1 Results for the Stochastic Discrete-Time Setting

To guarantee that the stochastic Bellman equation is well-posed and that the first-order optimality condition is valid, we consider the following set of assumptions [20].

Assumption C.1.1 (Well-posedness and regularity). The following holds on $\mathbb { R } ^ { n }$ :

1. $q : \mathbb { R } ^ { n } \to \mathbb { R } _ { + }$ is continuous.

2. $\rho : \mathbb { R } ^ { m }  \mathbb { R } _ { + }$ is $C ^ { 2 }$ and µ-strongly convex: $\nabla ^ { 2 } \rho ( a ) \succeq \mu I$ for all a.

3. $f , g$ are locally Lipschitz with at most polynomial growth.

4. The candidate value function $V : \mathbb { R } ^ { n }  \mathbb { R } \ i s \ C ^ { 2 }$ and convex.

Assumption C.1.2 (Growth and integrability). There exist constants $\eta > 0$ and an integer $\ell \geq 1$ such that for all $s \in \mathbb { R } ^ { n }$ ，

$$
| V ( s ) | + \| \nabla V ( s ) \| + \| \nabla ^ { 2 } V ( s ) \| \le \eta \bigl ( 1 + \| s \| ^ { \ell } \bigr ) ,
$$

and, for each fixed $( s , a ) \in \mathbb { R } ^ { n } \times \mathbb { R } ^ { m }$ , the expectations $\mathbb { E } [ \| \nabla V ( f ( s ) + g ( s ) a + w ) \| ]$ and $\mathbb { E } [ \| \nabla ^ { 2 } V ( f ( s ) +$ $g ( s ) a + w ) \| ]$ are finite under $w \sim \mathcal { N } ( 0 , \Sigma )$

Lemma C.1.1 (Expectation-gradient interchange). Under Assumptions 1 and 2, for all $s \in \mathbb { R } ^ { n }$ and $a \in \mathbb { R } ^ { m }$ ,

$$
\begin{array} { r } { \nabla _ { a } \mathbb { E } [ V ( f ( s ) + g ( s ) a + w ) ] = \mathbb { E } \Big [ g ( s ) ^ { \top } \nabla V ( f ( s ) + g ( s ) a + w ) \Big ] . } \end{array}
$$

Proof. Fix $s \in \mathbb { R } ^ { n }$ and define $\phi ( a , w ) = V ( f ( s ) + g ( s ) a + w )$ . By the chain rule, we have

$$
\nabla _ { a } \phi ( a , w ) = g ( s ) ^ { \top } \nabla V ( f ( s ) + g ( s ) a + w )
$$

To interchange the derivative and the expectation, we verify the conditions of the dominated convergence theorem. Let $e _ { 1 } , \ldots , e _ { m }$ be the canonical basis vectors of $\mathbb { R } ^ { m }$ . Without loss of generality, we choose a sequence $\left\{ t _ { \nu } \right\}$ converging to 0, such that we can define sequences $\{ a _ { \nu } ^ { ( i ) } \}$ through $a _ { \nu } ^ { ( i ) } = a + t _ { \nu } e _ { i }$ converging to a along each coordinate axis i.By the mean value theorem, for each axis $i ,$ there exists $\xi _ { \nu } ^ { ( \bar { i } ) }$ on the line segment between $a _ { \nu } ^ { ( i ) }$ and a such that

$$
\phi ( a _ { \nu } ^ { ( i ) } , w ) - \phi ( a , w ) = \nabla _ { a } \phi ( \xi _ { \nu } ^ { ( i ) } , w ) \cdot ( a _ { \nu } ^ { ( i ) } - a ) ,
$$

$$
\mathrm { ~ a n d ~ } | \phi ( a _ { \nu } ^ { ( i ) } , w ) - \phi ( a , w ) | \leq \| \nabla _ { a } \phi ( \xi _ { \nu } ^ { ( i ) } , w ) \| \cdot \| a _ { \nu } ^ { ( i ) } - a \| .
$$

Now, let us bound $\| \nabla _ { a } \phi ( \xi _ { \nu } ^ { ( i ) } , w ) \|$ for each axis i. From Assumption 2, we obtain:

$$
\| \nabla _ { a } \phi ( \xi _ { \nu } ^ { ( i ) } , w ) \| \le \| g ( s ) \| \cdot \| \nabla V ( f ( s ) + g ( s ) \xi _ { \nu } ^ { ( i ) } + w ) \| ,
$$

and hence,

$$
\| \nabla _ { a } \phi ( \xi _ { \nu } ^ { ( i ) } , w ) \| \le \| g ( s ) \| \cdot \eta ( 1 + \| f ( s ) + g ( s ) \xi _ { \nu } ^ { ( i ) } + w \| ^ { \ell } ) .
$$

Since $\xi _ { \nu } ^ { ( i ) }$ lies in a bounded neighborhood of a (for ν suficiently large), and $f , g$ have at most polynomial growth (Assumption 1.3), there exist constants $C _ { 1 } ( s )$ and $C _ { 2 } ( s )$ such that

$$
\begin{array} { r } { \| f ( s ) + g ( s ) \xi _ { \nu } ^ { ( i ) } + w \| \le C _ { 1 } ( s ) + C _ { 2 } ( s ) \| \xi _ { \nu } ^ { ( i ) } \| + \| w \| \le C ( s , a ) + \| w \| , } \end{array}
$$

where $C ( s , a )$ is independent of ν and $\begin{array} { r l } { \xi _ { \nu } ^ { ( i ) } } & { { } ( \mathrm { i . e . } } \end{array}$ , uniform over the neighborhood of $a )$ . This uniformity ensures the subsequent growth bound

$$
\| \nabla _ { a } \phi ( \xi _ { \nu } ^ { ( i ) } , w ) \| \leq \eta \| g ( s ) \| \big ( 1 + \big ( C ( s , a ) + \| w \| \big ) ^ { \ell } \big ) \leq K ( s , a ) ( 1 + \| w \| ^ { \ell } ) ,
$$

where $K ( s , a ) ( 1 + \| w \| ^ { \ell } )$ is independent of $\nu .$ This provides a single bound applicable to the entire sequence. The growth bound $\beta : w \mapsto K ( s , a ) ( 1 + \| w \| ^ { l } )$ is measurable.

Let $\mu _ { \Sigma }$ be the Gaussian measure. By definition of the expectation, the Lebesgue integral of $\beta$ weighted by the Gaussian density is

$$
\int _ { \mathbb { R } ^ { n } } \beta ( w ) d \mu _ { \Sigma } ( w ) = \mathbb { E } [ K ( s , a ) ( 1 + \| w \| ^ { l } ) ] .
$$

The linearity of the expectation yields

$$
\mathbb { E } [ K ( s , a ) ( 1 + \| w \| ^ { l } ) ] = K ( s , a ) ( 1 + \mathbb { E } [ \| w \| ^ { l } ] ) ,
$$

and hence,

$$
\int _ { \mathbb { R } ^ { n } } \beta ( w ) d \mu _ { \Sigma } ( w ) = K ( s , a ) \left( 1 + \int _ { \mathbb { R } ^ { n } } \| w \| ^ { \ell } d \mu _ { \Sigma } ( w ) \right) .
$$

As all Gaussian moments are finite, we have:

$$
\int _ { \mathbb { R } ^ { n } } \| w \| ^ { \ell } d \mu _ { \Sigma } ( w ) = \mathbb { E } [ \| w \| ^ { \ell } ] < \infty
$$

for all $\ell > 0$ . Hence,

$$
\int _ { \mathbb { R } ^ { n } } \beta ( w ) d \mu _ { \Sigma } ( w ) < \infty ,
$$

and the growth bound $\beta$ is Lebesgue-integrable with respect to the Gaussian measure. Since $\xi _ { \nu } ^ { ( i ) } \to a$ , continuity gives $\nabla _ { a } \phi ( \xi _ { \nu } ^ { ( i ) } , w ) \to \nabla _ { a } \phi ( a , w )$ pointwise. Hence, the bound

$$
\| \nabla _ { a } \phi ( \xi _ { \nu } ^ { ( i ) } , w ) \| \le K ( s , a ) ( 1 + \| w \| ^ { \ell } )
$$

provides a suitable dominating function, and by the dominated convergence theorem, we eventually obtain

$$
\operatorname* { l i m } _ { \nu \to \infty } \mathbb { E } [ \nabla _ { a } \phi ( \xi _ { \nu } ^ { ( i ) } , w ) ] = \mathbb { E } [ \nabla _ { a } \phi ( a , w ) ] .\tag{∗}
$$

On the other hand, by definition of the gradient, we have

$$
\nabla _ { a } \mathbb { E } [ \phi ( a , w ) ] = \Big ( \frac { \partial } { \partial a _ { 1 } } \mathbb { E } [ \phi ( a , w ) ] , \dots , \frac { \partial } { \partial a _ { m } } \mathbb { E } [ \phi ( a , w ) ] \Big ) .
$$

The directional derivative along the i-th axis is

$$
\frac { \partial } { \partial a _ { i } } \mathbb { E } [ \phi ( a , w ) ] = \operatorname* { l i m } _ { \nu \to \infty } \mathbb { E } \Big [ \frac { \phi ( a + t _ { \nu } e _ { i } , w ) - \phi ( a , w ) } { t _ { \nu } } \Big ] .
$$

Along the i-th axis, we also have

$$
\frac { \phi ( a + t _ { \nu } e _ { i } , w ) - \phi ( a , w ) } { t _ { \nu } } = \nabla _ { a } \phi ( \xi _ { \nu } ^ { ( i ) } , w ) \cdot e _ { i } .
$$

Combining these expressions yields

$$
\operatorname* { l i m } _ { \nu \to \infty } \mathbb { E } \Big [ \frac { \phi ( a + t _ { \nu } e _ { i } , w ) - \phi ( a , w ) } { t _ { \nu } } \Big ] = \operatorname* { l i m } _ { \nu \to \infty } \mathbb { E } [ \nabla _ { a } \phi ( \xi _ { \nu } ^ { ( i ) } , w ) \cdot e _ { i } ] = \mathbb { E } [ \nabla _ { a } \phi ( a , w ) \cdot e _ { i } ] ,
$$

where the last equality follows from the dominated convergence argument (∗) above. But, $\begin{array} { r } { \nabla _ { a } \phi ( a , w ) \cdot e _ { i } = \frac { \partial } { \partial a _ { i } } \phi ( a , w ) } \end{array}$ . Therefore, $\begin{array} { r } { \frac { \partial } { \partial a _ { i } } \mathbb { E } [ \phi ( a , w ) ] = \mathbb { E } [ \frac { \partial } { \partial a _ { i } } \bar { \phi } ( a , w ) ] } \end{array}$ . Finally, we get

$$
\begin{array} { r l r } & { } & { \nabla _ { a } \mathbb { E } [ \phi ( a , w ) ] = \Big ( \mathbb { E } [ \frac { \partial } { \partial a _ { 1 } } \phi ( a , w ) ] , \ldots , \mathbb { E } [ \frac { \partial } { \partial a _ { m } } \phi ( a , w ) ] \Big ) } \\ & { } & { = \mathbb { E } \Big ( \frac { \partial \phi } { \partial a _ { 1 } } ( a , w ) , \ldots , \frac { \partial \phi } { \partial a _ { m } } ( a , w ) \Big ) = \mathbb { E } [ \nabla _ { a } \phi ( a , w ) ] . } \end{array}
$$

Theorem C.1.2 (Stochastic converse optimality). Fix a stage cost $c ( s , a ) = q ( s ) + \rho ( a )$ with $q , \rho$ satisfying Assumption 1, discount factor $\gamma \in ( 0 , 1 )$ , noise covariance $\Sigma \succ 0$ , input coupling $g : \mathbb { R } ^ { n }  \mathbb { R } ^ { n \times m }$ , and a candidate value function $V ^ { * } \in C ^ { 2 } ( \mathbb { R } ^ { n } )$ satisfying Assumption 2. Then there exists a drift f : $\mathbb { R } ^ { n } \to \mathbb { R } ^ { n }$ (i.e. a system $s _ { k + 1 } = f ( s _ { k } ) + g ( s _ { k } ) a _ { k } + w _ { k }$ with $w _ { k } \sim \mathcal { N } ( 0 , \Sigma ) \}$ 1 for which $V ^ { * }$ is the optimal value function if and only if there exists a policy $\pi ^ { * } : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ such that for every $s \in \mathbb { R } ^ { n }$ :

$$
\begin{array} { r } { \nabla _ { a } \Big [ \rho ( a ) + \gamma \mathbb { E } \big [ V ^ { * } \big ( f ( s ) + g ( s ) a + w \big ) \big ] \Big ] \Big \rvert _ { a = a ^ { * } } = 0 , } \end{array}\tag{72}
$$

$$
V ^ { * } ( s ) = q ( s ) + \rho ( a ^ { * } ) + \gamma \mathbb { E } \big [ V ^ { * } \big ( f ( s ) + g ( s ) a ^ { * } + w \big ) \big ] .\tag{73}
$$

When such $\pi ^ { * }$ exists, it is an optimal policy and $V ^ { * }$ satisfies the stochastic Bellman equation given in Subsection 5.2 for all $s \in \mathbb { R } ^ { n }$

Proof. Suficiency. Suppose there exist a policy $\pi ^ { * } : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ and a drift $f : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ such that for every $s \in \mathbb { R } ^ { n }$ , equations (72) and (73) hold, and $\pi ^ { * } ( s _ { k } ) = a _ { k } ^ { * }$ for all $s _ { k } \in \mathbb { R } ^ { n }$ . For a fixed $s \in \mathbb { R } ^ { n }$ we define the Bellman integrand:

$$
\Phi _ { s } ( a ) : = q ( s ) + \rho ( a ) + \gamma \mathbb { E } \bigl [ V ^ { * } ( f ( s ) + g ( s ) a + w ) \bigr ]
$$

By the Lemma, we have:

$$
\nabla _ { a } \Phi _ { s } ( a ) = \nabla _ { a } \rho ( a ) + \gamma \mathbb { E } \big [ g ( s ) ^ { \top } \nabla V ^ { * } ( f ( s ) + g ( s ) a + w ) \big ] .
$$

Equations $( 4 ) \AA - \bigl ( 5 \bigr )$ give $\nabla _ { a } \Phi _ { s } ( a ) | _ { a = a ^ { * } } = 0$ and imply that the Bellman identity holds with action $a ^ { * }$ . In particular,

$$
V ^ { * } ( s ) = q ( s ) + \rho ( a ^ { * } ) + \gamma \mathbb { E } \big [ V ^ { * } ( f ( s ) + g ( s ) a ^ { * } + w ) \big ] = \Phi _ { s } ( a ^ { * } ) .
$$

Since the Bellman equation involves the minimization of $\Phi _ { s } ( a )$ with respect to $^ { a , }$ it follows that this minimum is attained at $a ^ { * }$ . Hence:

$$
\begin{array} { l } { \displaystyle \operatorname* { m i n } _ { a \in \mathbb { R } ^ { m } } \Phi _ { s } ( a ) = \Phi _ { s } ( a ^ { * } ) } \\ { \displaystyle \qquad = q ( s ) + \rho ( a ^ { * } ) + \gamma \mathbb { E } \big [ V ^ { * } ( f ( s ) + g ( s ) a ^ { * } + w ) \big ] } \\ { \displaystyle \qquad = V ^ { * } ( s ) . } \end{array}
$$

Thus, $\pi ^ { * }$ is an optimal policy and $V ^ { * }$ is the optimal value function for the drift $f .$

Now fix any stationary policy π and define its Bellman operator:

$$
( T ^ { \pi } V ) ( s ) : = q ( s ) + \rho ( \pi ( s ) ) + \gamma \mathbb { E } \left[ V { \bigl ( } f ( s ) + g ( s ) \pi ( s ) + w ) { \bigr ) } \right] .
$$

By definition of the minimum, mi ${ \Omega } _ { a } \Phi _ { s } ( a ) \leq \Phi _ { s } ( \pi ( s ) )$ , hence $V ^ { * } ( s ) \leq ( T ^ { \pi } V ^ { * } ) ( s )$ for all $s .$ Applying $T ^ { \pi }$ to both sides of $V ^ { * } \leq T ^ { \pi } V ^ { * }$ and using the monotonicity of $T ^ { \pi }$ (which follows from the positivity of costs), we obtain $T ^ { \pi } V ^ { * } \leq ( T ^ { \pi } ) ^ { 2 } V ^ { * }$ . By induction, $V ^ { * } \leq ( T ^ { \pi } ) ^ { k } V ^ { * }$ for all $k \in \mathbb N$ . Since $T ^ { \pi }$ is a γ-contraction in an appropriate weighted norm $( \mathrm { e . g . }$ the sup norm under bounded costs, or a weighted norm under the discounted criterion), then by the Banach fixed-point theorem, $( T ^ { \pi } ) ^ { k } V ^ { * }  V ^ { \pi }$ , the value function of policy π. Taking limits preserves the inequality, yielding $V ^ { * } \leq V ^ { \pi }$ on R<sup>n</sup>, proving the optimality of $V ^ { * }$ and $\pi ^ { * }$

Necessity. Assume $V ^ { * }$ optimal, so for each s the minimizer $\pi ^ { * } ( s )$ of the Bellman equation exists. By the measurable selection theorem $[ 3 4 ] , \pi ^ { * }$ is chosen to be measurable, and hence admissible as a feedback policy. At a minimum, we have $\nabla _ { a } \Phi _ { s } ( \pi ^ { * } ( s ) ) = 0$ . Lemma 1 justifies interchanging the derivative and the expectation, yielding (72). The Bellman identity (73) holds by the definition of $\pi ^ { * } ( s ) = a ^ { * }$ as the minimizer of the Bellman equation. □

## C.2 Continuous-Time Converse Derivation

Recall the continuous-time control-afine system ${ \dot { s } } = f ( s ) + g ( s ) a$ with running cost $c ( s , a )$ . For a candidate value function $V \in C ^ { 1 }$ , the Hamiltonian is

$$
H ( s , a ) = c ( s , a ) + \nabla V ( s ) ^ { \top } \big ( f ( s ) + g ( s ) a \big ) .
$$

The generator [19] prescribes two chosen smooth fields $\boldsymbol { H } : \mathbb { R } ^ { n }  \mathbb { R } ^ { n }$ and $G : \mathbb { R } ^ { n }  \mathbb { R } ^ { n \times m }$ and defines

$$
g ( s ) = G ( s ) ,\tag{74}
$$

$$
f ( s ) = \frac { 1 } { 4 } G ( s ) \left( \mathcal { L } _ { G } V ( s ) \right) ^ { \top } - H ( s ) ,\tag{75}
$$

$$
c ( s , a ) = \mathcal { L } _ { H } V ( s ) + \| a \| _ { 2 } ^ { 2 } .\tag{76}
$$

Here $\mathcal { L } _ { G } V ( s ) = \nabla V ( s ) ^ { \top } G ( s )$ is a row vector in $\mathbb { R } ^ { 1 \times m }$ , and $\mathcal { L } _ { H } V ( s ) = \nabla V ( s ) ^ { \top } H ( s )$ is a scalar. Substituting (74)–(76) into the Hamiltonian,

$$
\begin{array} { r l } & { H ( s , a ) = { \mathscr L } _ { H } V ( s ) + \| a \| _ { 2 } ^ { 2 } + \nabla V ( s ) ^ { \top } \Big ( \frac { 1 } { 4 } G ( s ) \big ( { \mathscr L } _ { G } V ( s ) \big ) ^ { \top } - H ( s ) + G ( s ) a \Big ) } \\ & { \qquad = { \mathscr L } _ { H } V ( s ) + \| a \| _ { 2 } ^ { 2 } + \frac { 1 } { 4 } \nabla V ( s ) ^ { \top } G ( s ) \big ( { \mathscr L } _ { G } V ( s ) \big ) ^ { \top } - \nabla V ( s ) ^ { \top } H ( s ) + \nabla V ( s ) ^ { \top } G ( s ) a . } \end{array}
$$

Because $\begin{array} { r } { \mathcal { L } _ { G } V ( s ) = \nabla V ( s ) ^ { \top } G ( s ) } \end{array}$ , we have

$$
\nabla V ( s ) ^ { \top } G ( s ) \big ( { \mathcal L } _ { G } V ( s ) \big ) ^ { \top } = \| { \mathcal L } _ { G } V ( s ) \| _ { 2 } ^ { 2 } , \qquad \nabla V ( s ) ^ { \top } H ( s ) = { \mathcal L } _ { H } V ( s ) .
$$

Therefore the terms $\mathcal { L } _ { H } V ( s )$ cancel, leaving

$$
H ( s , a ) = \| a \| _ { 2 } ^ { 2 } + { \frac { 1 } { 4 } } \| { \mathcal { L } } _ { G } V ( s ) \| _ { 2 } ^ { 2 } + \left( { \mathcal { L } } _ { G } V ( s ) \right) a .
$$

The right-hand side is a quadratic form in a. Completing the square,

$$
H ( s , a ) = \big \| { a + \textstyle \frac { 1 } { 2 } \big ( \mathcal { L } _ { G } V ( s ) \big ) ^ { \top } } \big \| _ { 2 } ^ { 2 } .
$$

The unique minimizer of this expression is obtained by making the squared norm vanish, i.e.

$$
\begin{array} { r } { a ^ { \star } ( s ) = - \frac { 1 } { 2 } \left( \mathcal { L } _ { G } V ( s ) \right) ^ { \top } , } \end{array}
$$

which is exactly the equation given in the main text. When V and the constructed cost are positive definite, Proposition 1 guarantees that V is the globally optimal value function for the generated system.

## D Proofs for the Non-Smooth Theory

Proof of Lemma 7.4.1 (First-order optimality condition). Fix any $s \in \mathbb { R } ^ { n }$ for which the minimum in the viscosity HJB condition is attained and equals zero, and let $p = p ( s ) \in \partial _ { C } V ( s )$ be the corresponding generalized gradient. Define the function

$$
\Phi _ { s } ( a ) : = q ( s ) + \rho ( a ) + p \cdot \left( f ( s ) + g ( s ) a \right) .
$$

By assumption, $a ^ { * } ( s )$ is a global minimizer of $\Phi _ { s }$ over $\mathbb { R } ^ { m }$ and $\Phi _ { s } ( a ^ { * } ( s ) ) = 0$ . Since $q ( s )$ and $p \cdot f ( s )$ do not depend on $a ,$ , minimizing $\Phi _ { s }$ is equivalent to minimizing

$$
h ( a ) : = \rho ( a ) + p \cdot g ( s ) a , \qquad a \in \mathbb { R } ^ { m } .
$$

The function $\rho$ is strictly convex and $C ^ { 2 }$ by Assumption 7.3.1. Moreover, $\rho$ is coercive, so $h ( a )  \infty \mathrm { { a s } } \| a \|  \infty$ . Hence the sublevel sets of $h$ are compact, and a global minimum of h exists. Strict convexity guarantees that this minimum is unique. Consequently, $a ^ { * } ( s )$ is the unique global minimizer of $h .$

Because the minimization is over the whole space $\mathbb { R } ^ { m }$ (no constraints), the first-order necessary condition for optimality of a smooth, unconstrained objective function must hold:

$$
\nabla h ( a ) \big | _ { a = a ^ { * } ( s ) } = \nabla _ { a } \rho ( a ) \big | _ { a = a ^ { * } ( s ) } + g ( s ) ^ { \top } p = 0 .
$$

Strict convexity of $\rho$ implies that the Hessian $\nabla ^ { 2 } h ( a ) = \nabla ^ { 2 } \rho ( a )$ is positive definite everywhere; therefore the first-order condition is also suficient for global optimality. This establishes (27) for almost every s. □

Proof of Lemma $\widetilde { 7 } . 4 . 2 \ ( S u f \mathscr { k } c i e n t$ condition for optimality). Conditions (b) and (c) together state that for almost every s:

• there exists $p ( s ) \in \partial _ { C } V ( s )$ for which

$$
\operatorname* { i n f } _ { a \in \mathbb { A } } \left\{ c ( s , a ) + p ( s ) \cdot ( f ( s ) + g ( s ) a ) \right\} = 0 ,
$$

and the infimum is realized by $a ^ { * } ( s )$ (condition (b));

• for every $\tilde { p } \in \partial _ { C } V ( s )$

$$
\operatorname* { i n f } _ { a \in \mathbb { A } } \left\{ c ( s , a ) + \tilde { p } \cdot \left( f ( s ) + g ( s ) a \right) \right\} \geq 0
$$

(condition (c)).

Thus V satisfies precisely the viscosity HJB condition of Definition 7.3.1: the subsolution property holds with equality for the particular generalized gradient $p ( s )$ , and the supersolution property is exactly condition (c). Moreover, condition (b) guarantees that a measurable optimal feedback $a ^ { * }$ exists and that the closed-loop system admits solutions (the well-posedness of the feedback is ensured when $a ^ { * }$ is discontinuous by interpreting solutions in the sense of Filippov; see Lemma 7.4.3).

Because V satisfies the viscosity HJB equation, it is a viscosity solution of the Hamilton– Jacobi–Bellman equation associated with the control problem. Under Assumptions 7.3.1–7.3.2, the classical uniqueness results for viscosity solutions [22, 33] imply that V is the unique viscosity solution. Since the true optimal value function $V ^ { * }$ is also a viscosity solution of the same HJB equation (see, e.g., [22]), we must have $V = V ^ { * }$ . Therefore $V$ is the optimal value function.

Condition (a) (stationarity) is automatically fulfilled because $a ^ { * } ( s )$ minimizes the Hamiltonian over $\mathbb { R } ^ { m }$ and the control cost $\rho$ is smooth and strictly convex; it is stated separately for clarity and for use in constructive procedures. The policy $a ^ { * }$ that attains the minimum is thus optimal. □

Proof of Lemma $\ 7 . 4 . 3$ (Measurability of the optimal feedback). (a) Non-emptiness, compactness, and measurability of $\Phi ( s )$ . Fix $s \in \mathbb { R } ^ { n }$ . The function

$$
\psi _ { s } ( a ) : = c ( s , a ) + p ( s ) \cdot \left( f ( s ) + g ( s ) a \right)
$$

is continuous in $a ,$ , because $c$ is continuous (Assumption 7.3.1), f and $g$ are locally Lipschi $^ { \mathrm { ~ Z ~ , ~ } }$ and the inner product with $p ( s )$ is linear. The control set A is compact and non-empty (Assumption 7.3.1). By the Extreme Value Theorem, $\psi _ { s }$ attains its minimum on $\mathbb { A } .$ Hence, $\Phi ( s )$ is non-empty.

The set $\Phi ( s ) = \psi _ { s } ^ { - 1 } ( \{ \operatorname* { m i n } \psi _ { s } \} )$ is the inverse image of the closed singleton $\{ \operatorname* { m i n } \psi _ { s } \}$ under a continuous map $\psi _ { s } ;$ therefore it is closed. A closed subset of a compact is compact, so $\Phi ( s )$ is compact.

We now show that $\Phi$ is measurable $( \mathrm { i . e . }$ , its graph is a Borel subset of $\mathbb { R } ^ { n } \times \mathbb { A } )$ . For each fixed $a \in \mathbb { A }$ , the map $s \mapsto \psi _ { s } ( a )$ is Lebesgue-measurable. Indeed, c is continuous, $f$ and $g$ are locally Lipschitz (hence measurable), and $p ( s )$ is measurable by hypothesis; the sum of compositions of measurable functions is measurable. Thus $( s , a ) \mapsto \psi _ { s } ( a )$ is a Carath´eodory function (measurable in s for fixed $^ { a , }$ and continuous in a for fixed $s )$ . For a Carath´eodory objective and a constant compact constraint set, the argmin multifunction is measurable; its graph is a Borel subset of the product space $\mathbb { R } ^ { n } \times \mathbb { A } \ \lvert 3 5 \rvert$

(b) Existence of a measurable selection. From part (a), the correspondence $\Phi ( s )$ has non-empty compact (hence closed) values and its graph Gr(Φ) is a Borel subset of $\mathbb { R } ^ { n } \times \mathbb { A }$ . The spaces $\mathbb { R } ^ { n }$ and $\mathbb { A } \subset \mathbb { R } ^ { m }$ are Polish (complete separable metric spaces). For a correspondence with closed values taking values in a Polish space, the property that its graph is a Borel set is equivalent to weak measurability [35]. Hence, $\Phi ( s )$ is weakly measurable. The Kuratowski–Ryll-Nardzewski measurable selection theorem states that every weakly measurable set-valued map with non-empty closed values in a Polish space admits a Borel-measurable selection. Specifically, if $\mathrm { G r } ( \Phi )$ is measurable (which it is, being Borel-measurable) and $\Phi ( s )$ is non-empty and closed for all s, then there exists a Borel-measurable function $a ^ { * } : \mathbb { R } ^ { n } \to \mathbb { A }$ such that $a ^ { * } ( s ) \in \Phi ( s )$ for almost every s. Since every Borel-measurable function is Lebesgue-measurable, we obtain the required measurable selection, which we take as the optimal feedback.

(c) Filippov solutions of the closed-loop system. We define the closed-loop vector field $F ( s ) : = f ( s ) + g ( s ) a ^ { * } ( s )$ . The functions $f$ and $g$ are continuous (so Borel-measurable) and the function $a ^ { * }$ is Lebesgue-measurable. The composition of continuous and measurable functions is measurable, so $F$ is Lebesgue-measurable. The functions $f , g$ are locally Lipschitz, so they are locally bounded. The action set $\mathbb { A }$ is compact by assumption, and therefore $a ^ { * } ( s )$ stays within a bounded set. Hence for any bounded neighborhood of a state, the product $g ( s ) a ^ { * } ( s )$ is bounded, and $F ( s )$ is bounded. Thus $F$ is locally bounded.

For a vector field that is measurable and locally bounded, Filippov’s theorem [36] states that for every initial condition $s _ { 0 } \in \mathbb { R } ^ { n }$ there exists an absolutely continuous trajectory $s ( t )$ defined on some interval [0, T] such that

$$
\dot { s } ( t ) \in K [ F ] ( s ( t ) ) \quad \mathrm { f o r ~ a l m o s t ~ a l l } ~ t ,
$$

where $K [ F ]$ denotes the Filippov set. Thus the closed-loop system admits a solution in the sense of Filippov. If $a ^ { * }$ is continuous, the Filippov set reduces to $\{ F ( s ) \}$ and the trajectory is a classical Carath´eodory solution. □

Proof of Theorem $7 . 4 . 4$ (Non-smooth continuous-time deterministic converse optimality). (b) $\Rightarrow ( a ) - S u f f c i e n c y .$ If statement (b) holds, we are given a drift $f ,$ a measurable policy $a ^ { * }$ , and a measurable selection $p ( s ) \in \partial _ { C } V ( s )$ satisfying conditions $( \mathrm { i } ) { - } ( \mathrm { i i i } )$ . These are exactly the hypotheses of Lemma 7.4.2 (suficient condition for optimality). Lemma 7.4.2 therefore applies and guarantees that, for the control-afine system ${ \dot { s } } = f ( s ) + g ( s ) a$ with running cost $c ,$ the function V is the optimal value function and $a ^ { * }$ is an optimal feedback policy. Hence statement (a) is true.

$( a ) \Rightarrow ( b ) - N e c e s s i t y .$ . Assume that (a) holds, i.e., there exists a drift $f _ { 0 }$ such that V is the optimal value function for the system $\dot { s } = f _ { 0 } ( s ) + g ( s ) a$ with cost c. Because the control problem satisfies Assumptions 7.3.1–7.3.2, the optimal value function is a viscosity solution of the associated Hamilton–Jacobi–Bellman equation [22, 33]. Hence the triple $( f _ { 0 } , g , c )$ satisfies the viscosity HJB condition of Definition 7.3.1: there exist a measurable policy $a ^ { * } : \mathbb { R } ^ { n } \to \mathbb { A }$ and a measurable selection $p ( s ) \in \partial _ { C } V ( s )$ such that for almost every s

$$
\operatorname* { i n f } _ { a \in \mathbb { A } } \left\{ c ( s , a ) + p ( s ) \cdot \left( f _ { 0 } ( s ) + g ( s ) a \right) \right\} = 0 , \qquad a ^ { * } ( s ) \mathrm { ~ a t t a i n s ~ t h e ~ i n f i m u m } ,
$$

and for every $\tilde { p } \in \partial _ { C } V ( s )$

$$
\operatorname* { i n f } _ { a \in \mathbb { A } } \left\{ c ( s , a ) + \tilde { p } \cdot \left( f _ { 0 } ( s ) + g ( s ) a \right) \right\} \geq 0 .
$$

Thus conditions (ii) and (iii) of statement (b) hold with the drift $f = f _ { 0 }$

It remains to verify the stationarity condition (i). The hypotheses of Lemma 7.4.1 (first-order optimality) are fulfilled: $V$ is a viscosity solution, the control cost $\rho$ is strictly convex and $C ^ { 2 }$ and the selections $a ^ { * }$ , p achieve the zero infimum. Lemma 7.4.1 therefore yields

$$
\nabla _ { a } \rho ( a ) \big | _ { a = a ^ { * } ( s ) } + g ( s ) ^ { \top } p ( s ) = 0 ,
$$

which is precisely condition (i) of statement (b). Consequently, the same drift $f _ { 0 }$ together with the selections $a ^ { * }$ and $p$ satisfies all three requirements of (b). This completes the proof of the equivalence. □

## E Proofs for the Information-State Theory

Proof of Theorem 8.1.1 (Information-state reduction). Fix a finite horizon T and define $v _ { T } ( b ) : =$ 0. For $t = T - 1 , \dots , 0$ , define recursively

$$
v _ { t } ( b ) : = \operatorname* { m i n } _ { a \in \mathbb { R } ^ { m } } \left\{ c ( b , a ) + \gamma \int _ { \mathbb { B } } v _ { t + 1 } ( b _ { + } ) p _ { b } ( b _ { + } \mid b , a ) d b _ { + } \right\} .\tag{77}
$$

We claim by backward induction on t that $v _ { t } ( b )$ is the minimal conditional cost-to-go from any history $h _ { t }$ satisfying $\phi _ { t } ( h _ { t } ) = b$ . The claim holds at $t = T$ because the terminal cost is zero. Suppose it holds at $t + 1$ . Fix a history $h _ { t }$ with $\phi _ { t } ( h _ { t } ) = b$ . For any output-feedback action $a = \pi _ { t } ( h _ { t } )$ , Definition 8.1.1 gives

$$
\begin{array} { r l } & { \mathbb { E } \big [ C _ { t } + \gamma v _ { t + 1 } ( B _ { t + 1 } ) \mid H _ { t } = h _ { t } , A _ { t } = a \big ] } \\ & { \qquad = c ( b , a ) + \gamma \int _ { \mathbb { B } } v _ { t + 1 } ( b _ { + } ) p _ { b } ( b _ { + } \mid b , a ) d b _ { + } . } \end{array}\tag{78}
$$

The right-hand side depends on the history only through $b = \phi _ { t } ( h _ { t } )$ and on the action a. Minimizing over a yields exactly $( 7 7 )$ . Hence no dependence on the full history remains once $b$ is fixed. This proves the claim for the finite horizon $T ,$

Now let $T \to \infty$ . The Bellman operator

$$
( \mathcal T u ) ( b ) : = \operatorname* { m i n } _ { a \in \mathbb { R } ^ { m } } \left\{ c ( b , a ) + \gamma \int _ { \mathbb { R } } u ( b _ { + } ) p _ { b } ( b _ { + } \mid b , a ) d b _ { + } \right\}\tag{79}
$$

is a γ-contraction in the supremum norm on bounded value functions, because $c \geq 0$ is bounded and $\gamma \in ( 0 , 1 )$ . By the Banach fixed-point theorem, T has a unique fixed point $v ^ { \star }$ in the space of bounded measurable functions, and the finite-horizon iterates (77) converge to $v ^ { \star }$ uniformly. Therefore $v ^ { \star }$ satisfies (52).

It remains to transfer the result to output-feedback policies. Let $\pi ^ { \star } ( b )$ be a measurable minimizer of the right-hand side of (52), whose existence is guaranteed by the measurableselection argument of Lemma 7.4.3 applied to the information-state dynamics. Define the outputfeedback policy $A _ { k } = \pi ^ { \star } ( \phi _ { k } ( H _ { k } ) )$ ). For every history $h _ { k }$

$$
\begin{array} { r } { \mathbb { E } \big [ C _ { k } + \gamma v ^ { \star } ( B _ { k + 1 } ) \mid H _ { k } = h _ { k } , A _ { k } = \pi ^ { \star } ( \phi _ { k } ( h _ { k } ) ) \big ] = v ^ { \star } ( \phi _ { k } ( h _ { k } ) ) , } \end{array}\tag{80}
$$

which is the Bellman identity evaluated at the minimizer. Any other nonanticipative history policy π satisfies the Bellman inequality and therefore cannot achieve a lower conditional costto-go. This proves optimality of $\pi ^ { \star }$ among all nonanticipative output-feedback policies. □

Proof of Theorem 8.2.1 (Converse optimality on the information state). The proof mirrors the proof of Theorem C.1.2 (Appendix C) with the physical state s replaced by the information state b and the noise w replaced by Ω. We state the diferences explicitly.

Suficiency. Suppose there exist a policy $\pi ^ { \star } : \mathbb { R } ^ { d }  \mathbb { R } ^ { m }$ and a drift $f$ such that for every $b \in \mathbb { R } ^ { d }$ conditions (55)–(56) hold. Define the Bellman integrand

$$
\begin{array} { r } { \Phi _ { b } ( a ) : = q ( b ) + \rho ( a ) + \gamma \mathbb { E } \big [ { v } ^ { \star } ( f ( b ) + g ( b ) a + \Omega ) \big ] . } \end{array}\tag{81}
$$

By the expectation-gradient interchange lemma (Lemma C.1.1, which holds verbatim with Ω in place of w under the same growth and integrability assumptions on $v ^ { \star } )$ , the gradient of $\Phi _ { b }$ with respect to $a$ is

$$
\nabla _ { a } \Phi _ { b } ( a ) = \nabla _ { a } \rho ( a ) + \gamma \mathbb { E } \big [ g ( b ) ^ { \top } \nabla v ^ { \star } ( f ( b ) + g ( b ) a + \Omega ) \big ] .\tag{82}
$$

The Bellman argmin condition (55) gives $\nabla _ { a } \Phi _ { b } ( a ) \big | _ { a = \pi ^ { \star } ( b ) } = 0$ , and condition (56) gives $\Phi _ { b } ( \pi ^ { \star } ( b ) ) =$ $v ^ { \star } ( b )$ . Therefore

$$
\operatorname* { m i n } _ { a \in \mathbb { R } ^ { m } } \Phi _ { b } ( a ) = \Phi _ { b } ( \pi ^ { \star } ( b ) ) = v ^ { \star } ( b ) ,\tag{83}
$$

so $v ^ { \star }$ satisfies the information-state Bellman equation (52) and $\pi ^ { \star }$ attains the minimum.

Now fix any stationary policy π and define the policy-evaluation operator

$$
( \mathcal T ^ { \pi } u ) ( b ) : = q ( b ) + \rho ( \pi ( b ) ) + \gamma \mathbb E \bigl [ u ( f ( b ) + g ( b ) \pi ( b ) + \Omega ) \bigr ] .\tag{84}
$$

By the minimum property, $v ^ { \star } \leq T ^ { \pi } v ^ { \star }$ for all b. Applying $\tau ^ { \pi }$ to both sides and using monotonicity gives $\mathcal T ^ { \pi } v ^ { \star } \leq ( \mathcal T ^ { \pi } ) ^ { 2 } v ^ { \star }$ , and by induction $v ^ { \star } \leq ( T ^ { \pi } ) ^ { k } v ^ { \star }$ for all $k \in \mathbb N$ . Under the same weighted-norm regularity assumed for the fully observed stochastic construction (Section 5), $\tau ^ { \pi }$ is a γ-contraction on the corresponding weighted Banach space, and the iterates $( T ^ { \pi } ) ^ { k } v ^ { \star }$ converge to $v ^ { \pi }$ , the value function of $\pi$ . Taking limits preserves the inequality, giving $v ^ { \star } \leq v ^ { \pi }$ on $\mathbb { R } ^ { d }$ . Hence $v ^ { \star }$ is the optimal information-state value function and $\pi ^ { \star }$ is optimal.

Necessity. Assume $v ^ { \star }$ is the optimal information-state value function, so for each b the minimizer in (52) exists. Measurability of $\pi ^ { \star }$ follows from Lemma 7.4.3 applied to the Carath´eodory integrand $\Phi _ { b } ( a )$ . At the minimum, $\nabla _ { a } \Phi _ { b } ( \pi ^ { \star } ( b ) ) = 0$ , which is (55). The Bellman identity (56) holds by the definition of $\pi ^ { \star } ( b )$ as the minimizer.

Transfer to output-feedback. $\mathrm { B y }$ Theorem 8.1.1, the infimum over history policies equals the infimum over information-state policies. The composite policy $A _ { k } = \pi ^ { \star } ( \phi _ { k } ( H _ { k } ) )$ is admissible and attains $v ^ { \star }$ , so it is optimal among all output-feedback policies. □

Scope of the infinite-horizon certificate. The quadratic construction establishes the stated Bellman identity and the pointwise minimizing action. To conclude that the candidate equals the infinite-horizon optimal value on an unbounded information-state space, a separate verification argument is required; the bounded-function contraction argument alone does not establish this conclusion for the quadratic candidate. The unbounded-cost case therefore requires appropriate weighted-space or integrability and transversality assumptions. We distinguish this verification step from the algebraic construction of the benchmark.

Proof of Corollary 8.3.1 (Quadratic information-state benchmark). Substituting the quadratic forms

$$
\begin{array} { r } { q ( b ) = b ^ { \top } Q b , \quad \rho ( a ) = a ^ { \top } R a , \quad v ^ { \star } ( b ) = b ^ { \top } P b + v _ { 0 } } \end{array}
$$

into the information-state Bellman equation (52) and completing the square with respect to a yields the same algebraic identity as in the fully observed stochastic case (Appendix F.2), with s replaced by b. Specifically, the Bellman integrand satisfies

$$
\Phi _ { b } ( a ) = b ^ { \top } Q b + \gamma ( f ( b ) + g ( b ) a ) ^ { \top } P ( f ( b ) + g ( b ) a ) + \gamma \operatorname { t r } ( P \Sigma _ { b } ) + \gamma v _ { 0 } + a ^ { \top } R a + ( \mathrm { t e r m s ~ l i n e a r ~ i n ~ } a ) ,
$$

and minimizing over a gives the optimal feedback

$$
a ^ { \star } ( b ) = - \gamma B _ { \gamma } ( b ) ^ { - 1 } g ( b ) ^ { \top } P f ( b ) , \qquad B _ { \gamma } ( b ) = R + \gamma g ( b ) ^ { \top } P g ( b ) .
$$

The minimized value is $f ( b ) ^ { \top } H _ { \gamma } ( b ) f ( b ) + b ^ { \top } Q b + \gamma \operatorname { t r } ( P \Sigma _ { b } ) + \gamma v _ { 0 }$ , and the identity $b ^ { \top } P b + v _ { 0 }$ = mi $\Omega _ { a } \Phi _ { b } ( a )$ holds if and only if

$$
b ^ { \top } ( P - Q ) b = f ( b ) ^ { \top } H _ { \gamma } ( b ) f ( b ) , \qquad v _ { 0 } = \frac { \gamma } { 1 - \gamma } \operatorname { t r } ( P \Sigma _ { b } ) .
$$

The metric-normalized parameterization $f ( b ) = H _ { \gamma } ( b ) ^ { - 1 / 2 } S ( b ) ( P - Q ) ^ { 1 / 2 } b$ satisfies the first identity by orthogonality of S, and the value ofset satisfies the second by construction. Theorem 8.2.1 then certifies $v ^ { \star }$ and $a ^ { \star }$ as the optimal pair. □

Proof of Proposition 8.4.1 (Innovation realization). Let $\tau ( b , a , o ) = f ( b ) + g ( b ) a + o$ and $\eta ( o \mid b , a ) = p _ { \Omega } ( o )$ . The induced transition kernel on $b ,$ defined in (63), is

$$
\begin{array} { c } { { \kappa ( D \mid b , a ) = \displaystyle \int _ { \mathbb { O } \mid \mathcal { C } \{ \tau ( b , a , o ) \in D \} \ : p _ { \Omega } ( o ) \ : d o } { } d o } } \\ { { = \displaystyle \int _ { \mathbb { O } \mid \mathcal { C } \{ f ( b ) + g ( b ) a + o \in D \} \ : p _ { \Omega } ( o ) \ : d o } . } } \end{array}\tag{85}
$$

Change variables $b _ { + } = f ( b ) + g ( b ) a + o .$ , so $o = b _ { + } - f ( b ) - g ( b ) a$ and $d o = d b _ { + }$ . Then

$$
\kappa ( D \mid b , a ) = \int _ { \mathbb { O } } p _ { \Omega } ( b _ { + } - f ( b ) - g ( b ) a ) d b _ { + } ,\tag{86}
$$

which is exactly the transition kernel induced by the information-state dynamics (53). Hence $( \eta , \tau )$ realizes (53). The induced law of $B _ { k + 1 }$ given $( B _ { k } , A _ { k } ) = ( b , a )$ coincides with the law under the afine dynamics, and the expected stage cost $\mathbb { E } [ C _ { k } \mid H _ { k } , A _ { k } ] = c ( B _ { k } , A _ { k } )$ is measurable from $( B _ { k } , A _ { k } )$ by construction. Therefore $B _ { k }$ satisfies Definition 8.1.1 and the realization is valid.

## F Generator Templates and Derivations

## F.1 Deterministic Discrete-Time Homotopy Construction

Consider deterministic dynamics

$$
s _ { k + 1 } = f ( s _ { k } ) + g ( s _ { k } ) a _ { k }
$$

with separable cost $c ( s , a ) = q ( s ) + \rho ( a )$ and discount factor absorbed into the notation for this subsection. G¨ohrt et al. [18] show that a candidate pair $( V ^ { \star } , q )$ is admissible if and only if

$$
V ^ { \star } ( s ) - q ( s ) \geq 0\tag{87}
$$

on the state domain. The suficiency argument is constructive. The Bellman equation requires

$$
V ^ { \star } ( s ) - q ( s ) = \operatorname* { m i n } _ { a } \{ \rho ( a ) + V ^ { \star } ( f ( s ) + g ( s ) a ) \} .
$$

For each $s ,$ the right-hand side is the minimum of a nonnegative function of the next-state point $f ( s ) + g ( s ) a$ and the control efort $\rho ( a )$ . The homotopy construction continuously deforms feasible next-state targets on the level set

$$
\{ z : V ^ { \star } ( z ) \leq V ^ { \star } ( s ) - q ( s ) \}
$$

while balancing the control cost $\rho ( u )$ . In the quadratic case,

$$
V ^ { \star } ( s ) = s ^ { \top } P s , \qquad q ( s ) = s ^ { \top } Q s , \qquad \rho ( a ) = a ^ { \top } R a ,
$$

with $0 \preceq Q \prec P$ and $R \succ 0$ , the admissibility condition is automatic and the construction becomes explicit. Define

$$
H ( s ) = \left( P ^ { - 1 } + g ( s ) R ^ { - 1 } g ( s ) ^ { \top } \right) ^ { - 1 } .
$$

Then any drift of the form

$$
f ( s ) = H ( s ) ^ { - 1 / 2 } S ( s ) ( P - Q ) ^ { 1 / 2 } s , \qquad S ( s ) ^ { \top } S ( s ) = I ,\tag{88}
$$

satisfies the deterministic energy identity

$$
s ^ { \top } ( P - Q ) s = f ( s ) ^ { \top } H ( s ) f ( s ) .
$$

The optimal feedback is

$$
a ^ { \star } ( s ) = - \left( R + g ( s ) ^ { \top } P g ( s ) \right) ^ { - 1 } g ( s ) ^ { \top } P f ( s ) .
$$

The orthogonal field $S ( s )$ is a design degree of freedom: it changes the drift geometry while preserving the certificate.

## F.2 Quadratic-Gaussian Energy Identity Derivation

Let

$$
V ^ { \star } ( s ) = s ^ { \top } P s + b , \qquad q ( s ) = s ^ { \top } Q s , \qquad \rho ( a ) = a ^ { \top } R a ,
$$

with $P \succ 0 , 0 \preceq Q \prec P$ , and $R \succ 0$ . For zero-mean noise with covariance Σ,

$$
\mathbb { E } [ V ^ { \star } ( f + g a + w ) ] = ( f + g a ) ^ { \top } P ( f + g a ) + \operatorname { t r } ( P \Sigma ) + b .
$$

The Bellman integrand is

$$
\begin{array} { r l } & { \Phi _ { s } ( a ) = s ^ { \top } Q s + a ^ { \top } R a + \gamma ( f + g a ) ^ { \top } P ( f + g a ) + \gamma \operatorname { t r } ( P \Sigma ) + \gamma b } \\ & { \qquad = s ^ { \top } Q s + \gamma f ^ { \top } P f + 2 \gamma a ^ { \top } g ^ { \top } P f + a ^ { \top } ( R + \gamma g ^ { \top } P g ) a + \gamma \operatorname { t r } ( P \Sigma ) + \gamma b . } \end{array}\tag{89}
$$

Define

$$
B _ { \gamma } ( s ) = R + \gamma g ( s ) ^ { \top } P g ( s ) .
$$

Since $B _ { \gamma } ( s ) \succ 0$ , the unique minimizer is

$$
a ^ { \star } ( s ) = - \gamma B _ { \gamma } ( s ) ^ { - 1 } g ( s ) ^ { \top } P f ( s ) .
$$

Substituting the minimizer gives

$$
\begin{array} { c } { { \displaystyle { \operatorname* { m i n } _ { a } \Phi _ { s } ( a ) = s ^ { \top } Q s + \gamma f ^ { \top } P f - \gamma ^ { 2 } f ^ { \top } P g B _ { \gamma } ^ { - 1 } g ^ { \top } P f + \gamma \mathrm { t r } ( P \Sigma ) + \gamma b } } } \\ { { = s ^ { \top } Q s + f ^ { \top } H _ { \gamma } f + \gamma \mathrm { t r } ( P \Sigma ) + \gamma b , } } \end{array}\tag{90}
$$

where

$$
H _ { \gamma } ( s ) = \gamma \left( P - \gamma P g ( s ) B _ { \gamma } ( s ) ^ { - 1 } g ( s ) ^ { \top } P \right) .
$$

Thus $\begin{array} { r } { V ^ { \star } ( s ) = \operatorname* { m i n } _ { a } \Phi _ { s } ( a ) } \end{array}$ holds if

$$
s ^ { \top } ( P - Q ) s = f ( s ) ^ { \top } H _ { \gamma } ( s ) f ( s )
$$

and

$$
b = { \frac { \gamma } { 1 - \gamma } } \operatorname { t r } ( P \Sigma ) .
$$

By the matrix inversion lemma,

$$
P - \gamma P g ( R + \gamma g ^ { \top } P g ) ^ { - 1 } g ^ { \top } P = ( P ^ { - 1 } + \gamma g R ^ { - 1 } g ^ { \top } ) ^ { - 1 } ,
$$

so $H _ { \gamma } ( s ) \succ 0$ . This justifies the metric-normalized and square-root parameterizations.

## G Benchmark Generation, Experimental Implementation, and Reproducibility

This appendix describes the validation, implementation, and reproducibility protocol used for the reported benchmark experiments. The released benchmark code is organized into two selfcontained experimental modes. Mode I implements continuous-time sensitivity studies and symbolic generator demonstrations, while Mode II implements native discrete-time oracle-referenced fixtures. Both modes follow the same separation principle: the learner interacts only with the exposed environment, whereas the certified value function, optimal policy, validation traces, and oracle diagnostics are used only for benchmark validation and post-training evaluation.

<table><tr><td>Mode</td><td>Implementation path</td></tr><tr><td>Mode I: continuous-time sensitivity and generator studies codemode1/</td><td></td></tr><tr><td>Mode II: native discrete-time oracle-referenced fixtures</td><td>Codemode2/Code/</td></tr></table>

## G.1 Generator Validation and Experimental Implementation

Validation checks for native discrete-time fixtures. The stochastic discrete-time generator in Algorithm 2 produces each fixture together with a validation trace. Each exported instance undergoes three checks.

1. Symmetric positive-definiteness of the Bellman action metric. For the quadratic– Gaussian construction, the Bellman integrand contains

$$
B _ { \gamma } ( s ) = R + \gamma g ( s ) ^ { \top } P g ( s ) .
$$

The condition $B _ { \gamma } ( s ) \succ 0$ is required for two reasons. First, it makes the Bellman integrand strictly convex in the action, so the first-order optimality condition identifies the unique global minimizer $a ^ { \star } ( s )$ . Second, it keeps the discounted metric

$$
H _ { \gamma } ( s ) = \gamma \left( P - \gamma P g ( s ) B _ { \gamma } ( s ) ^ { - 1 } g ( s ) ^ { \top } P \right)
$$

positive definite, which is necessary for the energy equation

$$
s ^ { \top } ( P - Q ) s = f ( s ) ^ { \top } H _ { \gamma } ( s ) f ( s )
$$

to be well-posed and numerically stable. In the implementation, this condition is checked on a prescribed bounded validation domain D. Instances that violate the numerical lowereigenvalue tolerance are rejected.

2. Bellman residual. The Bellman residual check verifies that the constructed dynamics, costs, value, and policy satisfy the certified optimality identity. For each grid point $s \in \mathcal G$ the implementation computes

$$
\delta _ { \mathrm { B e l l } } ( s ) = | V ^ { \star } ( s ) - ( q ( s ) + \rho ( a ^ { \star } ( s ) ) + \gamma \mathbb { E } [ V ^ { \star } ( f ( s ) + g ( s ) a ^ { \star } ( s ) + w ) ] ) | .
$$

For the quadratic–Gaussian case, the expectation is evaluated analytically through the covariance term. An instance passes this check if

$$
\operatorname* { m a x } _ { s \in \mathcal { G } } \delta _ { \mathrm { B e l l } } ( s ) < \varepsilon _ { \mathrm { B e l l } } ,
$$

where $\varepsilon _ { \mathrm { B e l l } }$ is the numerical tolerance recorded with the fixture. This check prevents implementation-level errors in the oracle, drift, cost, or noise ofset.

3. Closed-loop boundedness. The closed-loop system under $\pi ^ { \star }$ is simulated over the validation horizon and domain. This verifies that the constructed instance is not only algebraically certified but also numerically usable for finite-horizon RL evaluation. Trajectories that diverge, leave the accepted validation domain, or accumulate numerical instability cause the candidate fixture to be rejected.

Validated native discrete-time fixtures are specified through version-controlled configuration artifacts containing the generator parameters, cost matrices, disturbance covariance, discount factor, oracle definitions for $V ^ { \star }$ and $\pi ^ { \star }$ , validation tolerances, and evaluation seeds.

Algorithm 2 Converse-Optimal Benchmark Generator (quadratic–Gaussian case)   
1: Input: family type, parameter ranges, number of instances N, parameterization (sqrt or metric), nonlin  
earity knob α, seeds   
2: for each hyperparameter tuple ψ do   
3: Sample parameter vector ψ<sub>i</sub> from the family-specific distribution $P _ { \Psi }$   
4: Extract $( n , m , Q , R , P , \gamma , \Sigma )$ and input map g from ψ<sub>i</sub>   
5: Verify $B _ { \gamma } ( s ) = R + \gamma g ( s ) ^ { \top } P g ( s ) \succsim 0$ on the validation domain D   
6: Choose a bounded direction field <sup>˜</sup>f(s) or orthogonal field $S ( s )$   
7: Construct f by the selected converse parameterization and compute $a ^ { \star } ( s )$   
8: Validate the Bellman identity on G and test closed-loop boundedness under $a ^ { \star }$   
9: Export the accepted fixture with oracle accessors, validation metadata, and evaluation seeds   
10: end for   
11: Output: accepted fixture set, validator, oracle evaluators, baseline scripts, and plotting scripts

Mode I: continuous-time sensitivity experiment. Mode I evaluates whether an RL training pipeline remains stable under controlled variation of certified continuous-time systems. The main sensitivity experiment compares two PPO pipelines that difer only in the PyTorch optimizer used inside Stable-Baselines3, namely Adam and RAdam. The system-generation schedule is fixed by generator seeds, while neural-network initialization and optimizer stochasticity are not frozen. Thus, the experiment measures optimizer-induced outcome variability under the same generated-system schedule rather than a single deterministic training trace.

Each generated Mode-I sensitivity instance is a three-dimensional, three-input continuoustime system

$$
\dot { s } = f _ { \theta } ( s ) + g _ { \theta } ( s ) a , \qquad s \in \mathbb { R } ^ { 3 } , \quad a \in \mathbb { R } ^ { 3 } .
$$

The generator uses the converse-HJB template

$$
f _ { \theta } ( \boldsymbol { s } ) = - \xi _ { \theta } ( \boldsymbol { s } ) + \frac { 1 } { 4 } g _ { \theta } ( \boldsymbol { s } ) g _ { \theta } ( \boldsymbol { s } ) ^ { \top } \nabla V ( \boldsymbol { s } ) , \qquad V ( \boldsymbol { s } ) = \| \boldsymbol { s } \| _ { 2 } ^ { 2 } , \qquad \xi _ { \theta } ( \boldsymbol { s } ) = 1 . 2 \nabla V ( \boldsymbol { s } ) .
$$

The input matrix $g _ { \boldsymbol { \theta } } ( s )$ is generated symbolically using a truncated trigonometric expansion. The continuous-time system is exposed to the RL agent through a Gymnasium wrapper using explicit Euler integration. The reported optimizer-sensitivity study uses horizon 200, integration step 0.01, initial state $s _ { 0 } = ( - 1 . 5 , - 1 . 5 , - 1 . 5 ) ^ { \top }$ , action bounds $[ - 1 0 , 1 0 ] ^ { 3 }$ , observation bounds $[ - 1 0 , 1 0 ] ^ { 3 }$ , and training reward

$$
r ( s , a ) = - \| s \| _ { 2 } ^ { 2 } - \| a \| _ { 2 } ^ { 2 } ,
$$

with numerical clipping to keep training traces bounded.

The Adam–RAdam comparison contains 80 generated-system runs per optimizer and 160 runs in total. The run schedule is generated from the seed/count pairs

$$
( 2 , 1 0 ) , \qquad ( 4 , 1 0 ) , \qquad ( 5 , 2 0 ) , \qquad ( 5 , 4 0 ) ,
$$

where (seed, M) denotes M sequentially generated systems after resetting the generator to the specified seed. The reproduction command is

cd "code mode $1 "$

python3 main.py

The TensorBoard directory is multi\_opt\_full\_experiment/. Stored CSV outputs for the 160- run study are provided under Combined\_CSVs/, and the analysis scripts compute the reported variance, histogram, boxplot, and kernel-density diagnostics:

python3 analysis\_scripts/tensorboard\_fscore.py

python3 analysis\_scripts/tensorboard\_hist.py

python3 analysis\_scripts/tensorboard\_boxplot.py

python3 analysis\_scripts/Final\_Rewards\_Distribution\_KDE.py

All PPO hyperparameters are shared between Adam and RAdam:
<table><tr><td>Quantity</td><td>Value</td><td>Comment</td></tr><tr><td>Policy class</td><td>MlpPolicy</td><td>Stable-Baselines3 PPO</td></tr><tr><td>Feature extractor</td><td>CustomMLP</td><td>feature dimension 128</td></tr><tr><td>Policy network</td><td>[64, 64]</td><td>shared architecture</td></tr><tr><td>Value network</td><td>[64, 64]</td><td>shared architecture</td></tr><tr><td>Optimizer</td><td>Adam or RAdam</td><td>only varied factor</td></tr><tr><td>Learning rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td>shared</td></tr><tr><td>Rollout length</td><td>2048</td><td>n_steps</td></tr><tr><td>Batch size</td><td>64</td><td>shared</td></tr><tr><td>Epochs per update</td><td>10</td><td>shared</td></tr><tr><td>Discount</td><td>0.99</td><td>shared</td></tr><tr><td>GAE parameter</td><td>0.95</td><td>shared</td></tr><tr><td>PPO clip range</td><td>0.1</td><td>shared</td></tr><tr><td>Entropy coefficient</td><td>0.01</td><td>shared</td></tr><tr><td>Gradient clipping</td><td>0.3</td><td>max_grad_norm</td></tr><tr><td>Training budget</td><td>600000 steps</td><td>per generated system</td></tr></table>

Mode I also includes a symbolic N-crank generator demonstration. This experiment is not used as the main oracle-referenced comparison in the paper; it demonstrates that the continuoustime converse template can generate physically interpretable nonlinear control families. The implemented example uses $N = 4$ coupled cranks with state

$$
\begin{array} { r } { s = ( x _ { 1 } , y _ { 1 } , \ldots , x _ { N } , y _ { N } ) ^ { \top } \in \mathbb { R } ^ { 2 N } , } \end{array}
$$

where $x _ { i }$ and $y _ { i }$ are the angular position and angular velocity of crank i, and $u _ { i }$ is the torque applied at the i-th pivot. Adjacent cranks are coupled through spring-like trigonometric terms. The symbolic construction uses

$$
f ( s ) = \frac { 1 } { 4 } G ( s ) L _ { G } V ( s ) ^ { \top } - H ( s ) , \qquad c ( s , u ) = L _ { H } V ( s ) + u ^ { \top } u ,
$$

with analytic optimal policy

$$
\boldsymbol { u } ^ { \star } ( \boldsymbol { s } ) = - \frac { 1 } { 2 } L _ { G } \boldsymbol { V } ( \boldsymbol { s } ) ^ { \top } .
$$

The training interface exposes PPO, A2C, and $\mathrm { S A C }$

python3 train\_all\_crank.py --run\_both

and the evaluator compares learned controllers with the analytic optimal controller and the uncontrolled baseline $u = 0$

python3 eval\_all\_with\_sim\_cranks.py

The evaluator exports state trajectories, control signals, value-function traces, stage-cost traces, animations, and CSV files under plots/ and plots/data/.

Mode II: native discrete-time oracle-referenced fixtures. Mode II uses native discretetime fixtures, so the environment transition law, oracle, learner interface, and evaluation metrics are defined in the same formalism. This removes the additional ambiguity introduced by choosing a numerical integration step for an underlying continuous-time system.

The Mode-II code contains two reported benchmark families. ConverseArm is a stochastic serial planar arm with state-dependent actuation. For n links,

$$
\begin{array} { r } { s = ( \theta _ { 1 } , \omega _ { 1 } , \ldots , \theta _ { n } , \omega _ { n } ) ^ { \top } \in \mathbb { R } ^ { 2 n } , \qquad a = ( \tau _ { 1 } , \ldots , \tau _ { n } ) ^ { \top } \in \mathbb { R } ^ { n } . } \end{array}
$$

The cost and value oracle are

$$
c ( s , a ) = s ^ { \top } Q s + a ^ { \top } R a , \qquad V ^ { \star } ( s ) = s ^ { \top } P s + c _ { \Sigma } , \qquad c _ { \Sigma } = \frac { \gamma } { 1 - \gamma } \operatorname { t r } ( P \Sigma ) .
$$

The input matrix is configuration dependent:

$$
[ g ( s ) ] _ { 2 i , i } = \cos ( \theta _ { i } ) , \qquad i = 1 , \dots , n ,
$$

with all other entries zero. The transition has the form

$$
\begin{array} { r } { s _ { k + 1 } = f _ { p } ( s _ { k } ) + p g ( s _ { k } ) a _ { k } + w _ { k } , \qquad w _ { k } \sim \mathcal { N } ( 0 , \Sigma ) , } \end{array}
$$

where $p$ is the control-authority parameter and $f _ { p }$ is generated by the stochastic quadratic converse construction. The environment returns reward $r _ { k } = - c ( s _ { k } , a _ { k } )$ , while all oracle diagnostics are computed in the cost convention and converted to reward convention only for plotting when explicitly stated.

NVDEx is the dynamically extended nonholonomic-vehicle family. The implementation name is NUDEx-v0; the manuscript uses the family name NVDEx. Each module has state

$$
s _ { i } = ( x _ { i } , y _ { i } , \phi _ { i } , v _ { i } ^ { ( 0 ) } , \ldots , v _ { i } ^ { ( r _ { v } - 1 ) } , \omega _ { i } ^ { ( 0 ) } , \ldots , \omega _ { i } ^ { ( r _ { \omega } - 1 ) } ) ^ { \top } .
$$

The kinematic variables evolve as

$$
\begin{array} { c } { { x _ { i , k + 1 } = x _ { i , k } + \tau v _ { i , k } ^ { ( 0 ) } \cos \phi _ { i , k } , y _ { i , k + 1 } = y _ { i , k } + \tau v _ { i , k } ^ { ( 0 ) } \sin \phi _ { i , k } , } } \\ { { { } } } \\ { { \phi _ { i , k + 1 } = \phi _ { i , k } + \tau \omega _ { i , k } ^ { ( 0 ) } . } } \end{array}
$$

The velocity-extension chains shift forward in discrete time, and the action enters only at the highest derivative. This creates delayed actuation and longer credit-assignment paths while retaining the certified quadratic oracle.

Both Mode-II families expose exact oracle methods through the benchmark object:

$$
\mathtt { o r a c l e . a c t i o n } ( s ) = \pi ^ { \star } ( s ) , \qquad \mathtt { o r a c l e . v a l u e } ( s ) = V ^ { \star } ( s ) , \qquad \mathtt { o r a c l e . q . v a l u e } ( s , a ) = Q ^ { \star } ( s , a ) .
$$

The oracle-referenced baselines are

$$
\{ \mathrm { P P O , A 2 C , S A C , T D 3 , D D P G } \} .
$$

The released scripts use Stable-Baselines3 implementations with the following principal hyperparameters:
<table><tr><td></td><td>Algorithm Main hyperparameters</td><td>Type</td></tr><tr><td>PPO</td><td>n_steps=1024,batch_size=256,n_epochs=10,  $\begin{array} { r l r } { \mathrm { l r } } & { { } = } & { 3 \mathrm { ~ \cdot ~ } 1 0 ^ { - 4 } } \end{array}$   $\mathtt { e n t \_ c o e f = 1 0 ^ { - 4 } , \mathtt { c l i p = 0 . 2 } }$ </td><td>, On-policy</td></tr><tr><td>A2C</td><td> $\mathtt { n _ { - } s t e p s { = } 2 0 , l r = 3 \cdot 1 0 ^ { - 4 } }$  , ent_coef=10−4</td><td>On-policy</td></tr><tr><td>SAC</td><td>batch_size=256, buffer=700000, 1earning_starts=10000, lr = 3 · 10−4, Off-policy τ = 0.005, ent_coef=auto</td><td></td></tr><tr><td>TD3</td><td>batch_size=256, buffer=700000, 1earning_starts=25000, lr = 3 · 10−4, Off-policy  $\tau = 0 . 0 0 5 , \mathtt { p o l i c y \_ d e l a y = 2 }$ </td><td></td></tr><tr><td>DDPG</td><td>batch_size=256, buffer=700000, learning_starts=25000, lr = 3· 10−4, Off-policy τ = 0.005</td><td></td></tr></table>

The exact fixture grids, algorithm schedules, training budgets, seeds, and dificulty presets used by the released experiments are stored in the YAML configuration files and shell launchers. This avoids hard-coding a separate table in the manuscript that could drift from the executable benchmark configuration.

## G.2 Oracle Evaluation, Artifacts, and Reproducibility Checklist

Cost and reward conventions. All oracle definitions are naturally expressed in the cost convention. Since the Gymnasium environments return rewards $r = - c ,$ the implementation records the convention used by each diagnostic. For a learned deterministic policy ${ \hat { \pi } } ,$ rollout horizon $T ,$ and initial state $s _ { 0 } .$ , the tail-corrected cost $\mathrm { g a p }$ is

$$
\mathrm { G a p } _ { T } ^ { c } ( \hat { \pi } ; s _ { 0 } ) = \frac { \mathbb { E } \left[ \sum _ { k = 0 } ^ { T - 1 } \gamma ^ { k } c ( s _ { k } , \hat { \pi } ( s _ { k } ) ) + \gamma ^ { T } V ^ { \star } ( s _ { T } ) \right] - V ^ { \star } ( s _ { 0 } ) } { | V ^ { \star } ( s _ { 0 } ) | + \epsilon } .
$$

This quantity is zero for the certified optimal policy up to numerical precision and is nonnegative for policies worse than the oracle. Equivalently, if $G ^ { \star } = - V ^ { \star }$ denotes the reward-value oracle, the reward-convention optimality gap is

$$
\mathrm { G a p } _ { T } ^ { r } ( \hat { \pi } ; s _ { 0 } ) = \frac { \mathbb { E } \left[ \sum _ { k = 0 } ^ { T - 1 } \gamma ^ { k } r ( s _ { k } , \hat { \pi } ( s _ { k } ) ) + \gamma ^ { T } G ^ { \star } ( s _ { T } ) \right] - G ^ { \star } ( s _ { 0 } ) } { | G ^ { \star } ( s _ { 0 } ) | + \epsilon } .
$$

In this convention, zero is optimal and more negative values indicate larger suboptimality. The paper reports explicitly whether a plot uses Gap<sup>c</sup> or Gap<sup>r</sup> .

The discounted cost regret is

$$
\operatorname { R e g r e t } _ { T } ^ { c } ( \hat { \pi } ; s _ { 0 } ) = \mathbb { E } \left[ \sum _ { k = 0 } ^ { T - 1 } \gamma ^ { k } \Big ( c ( s _ { k } , \hat { \pi } ( s _ { k } ) ) - c ( s _ { k } , \pi ^ { \star } ( s _ { k } ) ) \Big ) \right] .
$$

Lower cost regret is better. If a reward-convention regret is plotted, it is defined by

$$
\mathrm { R e g r e t } _ { T } ^ { r } = - \mathrm { R e g r e t } _ { T } ^ { c } ,
$$

and the sign convention is stated in the corresponding caption.

CRN evaluation schedule. For each stochastic fixture $\theta ,$ the evaluation schedule is generated before policy evaluation:

$$
\begin{array} { r } { \mathcal { E } _ { \theta } = \left\{ \left( s _ { 0 } ^ { ( j ) } , w _ { 0 : T - 1 } ^ { ( j ) } \right) \right\} _ { j = 1 } ^ { M } . } \end{array}
$$

All algorithms are evaluated on exactly the same ${ \mathcal { E } } _ { \theta }$ . If the environment is deterministic, the schedule reduces to matched start states. If the environment is stochastic, the same disturbance stream is replayed for each policy. This gives a paired design: the diference between two algorithms on trajectory $j$ is evaluated under the same initial condition and randomness. In the implementation, evaluation seeds are deterministic ofsets of the training seed, for example

$$
{ \mathrm { e v a l u a t i o n ~ s e c d } } = { \mathrm { t r a i n i n g ~ s e c d } } + 1 0 0 0 0 , \qquad { \mathrm { C R N ~ m e t r i c ~ s e c d } } = { \mathrm { t r a i n i n g ~ s e c d } } + 2 0 0 0 0 .
$$

For NVDEx, the evaluation callback uses deterministic policy actions, evaluates at fixed environmentstep intervals, and averages over multiple CRN-matched evaluation episodes.

Confidence intervals are computed by paired bootstrap:

1. Sample trajectory indices $j _ { 1 } , \dots , j _ { M }$ with replacement.

2. Recompute the metric diference using the resampled paired trajectories.

3. Repeat for $B _ { \mathrm { b o o t } }$ bootstrap replicates.

4. Report the empirical 2.5% and 97.5% quantiles.

The released evaluation schedule includes the seeds and serialized start-state/noise arrays needed to reproduce paired evaluation.

Bellman-consistency diagnostics. The implementation also logs Bellman-consistency quantities using the known value. In the cost convention, a one-step diagnostic is

$$
\delta _ { k } ^ { c } = V ^ { \star } ( s _ { k } ) - \left( c ( s _ { k } , a _ { k } ) + \gamma V ^ { \star } ( s _ { k + 1 } ) \right) .
$$

For oracle actions under the exact transition law, this residual is zero in expectation up to numerical and Monte Carlo error. These diagnostics are used for validation and sanity checking; they are not exposed to the learner during training.

Output artifacts. The released experiments produce artifacts at three levels. At the generator level, they include configuration files, generator seeds, accepted/rejected instance logs, and validation outputs. At the learning level, they include baseline training scripts, hyperparameter files, trained-model directories, TensorBoard traces, and rollout CSVs. At the evaluation level, they include oracle metrics, CRN schedules, bootstrap summaries, heatmaps, trajectory plots, and diagnostic figures.

The main output locations are:

• Mode-I optimizer-sensitivity traces under multi\_opt\_full\_experiment/ and aggregated CSVs under Combined\_CSVs/.

• Mode-I statistical plots under analysis\_scripts/, including variance-test, histogram, boxplot, and KDE diagnostics.

• N-crank trajectories, control traces, value traces, stage-cost traces, animations, and CSVs under plots/ and plots/data/.

• ConverseArm logs, models, rollout CSVs, and figures under the released arm experiment directories.

• NVDEx logs, CSV summaries, and plotting scripts under nvdexruns/ and its evaluation subdirectories.

## Reproducibility checklist. The project website and repository include:

1. Generator source code and version hash.

2. Fixture/configuration files containing $( P , Q , R , \Sigma , g , f , \gamma , V ^ { \star } , \pi ^ { \star } )$ or executable equivalents.

3. Generator seeds and accepted/rejected instance logs.

4. Validation-domain specifications and tolerance values.

5. CRN evaluation schedules.

6. Baseline training scripts and hyperparameter files.

7. Metric scripts for optimality gap, regret, policy error, Bellman residual, and bootstrap intervals.

8. Figure-generation scripts for tables, heatmaps, trajectories, and distributional plots.

9. Environment, Python, CUDA, and dependency versions.

10. Hardware and wall-clock summaries for the reported runs: Intel® Core<sup>TM</sup> i9-9900 CPU, RTX 3080 12 GB, and 16 GB DDR4 RAM.

This structure makes the benchmark independently auditable. Another group can regenerate or reload fixtures, verify Bellman consistency, train policies without oracle access, and reproduce oracle-referenced evaluation using the same paired start-state and disturbance schedules. Figure 10 shows that the mean values in Table 6 do not fully describe the failure modes. Several algorithm–regime pairs have heavy negative tails, meaning that some evaluated runs or fixtures are not merely slightly suboptimal but orders of magnitude farther from the certified optimum.

![](images/39a5f9e1b74b91bce2655fa02bde10e8ee2eca130253ee41db549aa0fc14f470.jpg)

Figure 10: Distribution of reward-convention optimality gaps by regime and algorithm. Points correspond to available run/fixture evaluation rows; violin widths show the empirical distribution of optimality gaps. The dashed line marks the certified optimum at zero. Black horizontal markers indicate central tendency. Unlike Figure 6, this is not a confidence-interval plot; it is a distributional diagnostic that exposes variability and lower-tail failures.
<table><tr><td>Experiment</td><td>Algorithm</td><td>Optimality gap</td><td>Optimality gap 95% CI</td><td>Regret (mean)</td><td>Regret 95% CI</td></tr><tr><td>Serial-random</td><td>A2C</td><td>-2.36e+03</td><td>[-2.79e+03, -1.93e+03]</td><td>8.24e+03</td><td>[6.83e+03, 9.58e+03]</td></tr><tr><td>Serial-random</td><td>DDPG</td><td>-2.29e+03</td><td>[-3.83e+03, -9.12e+02]</td><td>9.28e+03</td><td>[3.50e+03, 1.63e+04]</td></tr><tr><td>Serial-random</td><td>PPO</td><td>-3.32e+02</td><td>[-8.45e+02, -4.92e+01]</td><td>1.51e+03</td><td>[1.92e+02, 3.93e+03]</td></tr><tr><td>Serial-random</td><td>SAC</td><td>-1.65e+02</td><td>[-3.47e+02, -4.63e+01]</td><td>6.90e+02</td><td>[1.72e+02, 1.53e+03]</td></tr><tr><td>Serial-random</td><td>TD3</td><td>-5.85e+01</td><td>[-8.64e+01, -3.57e+01]</td><td>2.14e+02</td><td>[1.25e+02, 3.35e+02]</td></tr><tr><td>Serial-fixed</td><td>A2C</td><td>-1.11e+03</td><td>[-2.15e+03, -4.58e+02]</td><td>5.78e+04</td><td>[3.12e+04, 9.23e+04]</td></tr><tr><td>Serial-fixed</td><td>DDPG</td><td>-6.48e+02</td><td>[-1.34e+03, -2.51e+02]</td><td>3.06e+04</td><td>[1.61e+04, 5.22e+04]</td></tr><tr><td>Serial-fixed</td><td>PPO</td><td>-6.46e+01</td><td>[-9.56e+01, -3.68e+01]</td><td>4.08e+03</td><td>[2.30e+03, 6.14e+03]</td></tr><tr><td>Serial-fixed</td><td>SAC</td><td>-1.51e+03</td><td>[-2.44e+03, -8.00e+02]</td><td>9.03e+04</td><td>[5.18e+04, 1.33e+05]</td></tr><tr><td>Serial-fixed</td><td>TD3</td><td>-1.53e+03</td><td>[-2.69e+03, -7.61e+02]</td><td>8.31e+04</td><td>[5.05e+04, 1.23e+05]</td></tr><tr><td>NVDEx</td><td>A2C</td><td>-5.60e+03</td><td>[-1.33e+04, -1.06e+03]</td><td>5.94e+02</td><td>[1.34e+02, 1.33e+03]</td></tr><tr><td>NVDEx</td><td>DDPG</td><td>-1.46e+04</td><td>[-3.13e+04, -3.37e+03]</td><td>3.22e+03</td><td>[4.76e+02, 7.39e+03]</td></tr><tr><td>NVDEx</td><td>PPO</td><td>-1.58e+01</td><td>[-3.07e+01, -4.57e+00]</td><td>2.97e+00</td><td>[7.01e-01, 5.82e+00]</td></tr><tr><td>NVDEx</td><td>SAC</td><td>-7.76e+03</td><td>[-1.56e+04, -1.15e+03]</td><td>1.56e+03</td><td>[1.13e+02, 3.13e+03]</td></tr><tr><td>NVDEx</td><td>TD3</td><td>-1.22e+04</td><td>[-2.24e+04, -4.20e+03]</td><td>2.78e+03</td><td>[6.96e+02, 5.67e+03]</td></tr></table>

Table 6: Oracle-referenced comparison on native discrete-time fixtures. Metrics are mean and 95% CI via paired bootstrap under CRN. For optimality gap, zero is optimal and values closer to zero are better; for regret, smaller is better.