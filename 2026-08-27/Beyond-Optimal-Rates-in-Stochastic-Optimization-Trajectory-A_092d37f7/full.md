# Beyond Optimal Rates in Stochastic Optimization: Trajectory-Adaptive Stopping Rules

Liviu Aolaritei<sup>1</sup>, Lucas Lévy<sup>3</sup>, Francis Bach<sup>3</sup>, and Michael I. Jordan<sup>1,2,3</sup>

<sup>1</sup>Department of Electrical Engineering and Computer Sciences, UC Berkeley, USA

<sup>2</sup>Department of Statistics, UC Berkeley, USA

<sup>3</sup>Inria, École Normale Supérieure, PSL Research University, France

## Abstract

Stochastic gradient descent (SGD) is typically analyzed at a deterministic horizon chosen before the algorithm is run, even though practical stopping decisions are made adaptively by inspecting the evolving trajectory. This mismatch creates a fundamental certification problem: fixed-time guarantees do not generally remain valid at data-dependent stopping times, while deterministic horizons derived from worst-case bounds can be highly conservative. We address this problem for strongly convex stochastic optimization by constructing fully observable, trajectory-adaptive upper confi dence sequences for the squared distance of the last iterate to the optimizer and the suboptimality of a weighted average. These bounds hold simultaneously over time, attain the optimal 1/t decay rate up to iterated-logarithmic factors in the worst case, and adapt to the realized stochastic gradients, allowing SGD to stop as soon as a prescribed accuracy is certified without sacrificing statistical validity. Our approach treats the evolving SGD trajectory as a sequential experiment whose observations provide evidence about the unknown optimization error. To formalize this perspective, we develop new recursive confidence-sequence techniques and a general time-uniform empirical Bernstein in equality for adapted processes with time-varying conditional means and predictable ranges that may grow without bound. We further extend these confidence-sequence constructions to minibatch SGD, with the empirical Bernstein bounds exploiting the realized second-moment structure within each minibatch. Numerical experiments show that the resulting stopping rules can require several orders of magnitude fewer iterations than natural deterministic horizons.

## 1 Introduction

Stochastic optimization is a foundational computational paradigm across machine learning, operations research, statistics, and scientific computing. When objectives are expectations under unknown distributions or finite sums over prohibitively large datasets, stochastic gradient methods make optimization possible using only samples or noisy first-order information [41, 32]. Their scalability has made them the principal computational workhorses of modern machine learning [5]. Yet a basic operational question remains unresolved: when should a stochastic optimization algorithm stop?

Existing theoretical analyses of stochastic gradient descent (SGD) are predominantly ex ante. Before the algorithm is run, one specifies a stepsize schedule and a deterministic horizon T, and then studies its performance at that horizon. This perspective has produced a deep understanding of stochastic firstorder methods: it has identified optimal rates, clarified the roles of averaging and strong convexity, and provided principled guidance for selecting stepsizes [34, 4, 36, 26]. Its guarantees, however, are attached to horizons chosen independently of the realized run. In other words, they answer the forward-looking question of what can be guaranteed if SGD is run until a predetermined time $T ,$ but not the online question of whether the trajectory observed so far provides enough evidence that the algorithm should stop.

In practice, however, this online question is unavoidable. Practitioners alter stepsizes when progress slows and stop a run when the observed trajectory appears to have stabilized. Yet the quantities that would directly certify optimization progress are usually unknown: the objective is often a population loss defined by an expectation under an unknown distribution, and its optimizer and optimal value are unknown. Consequently, neither the true suboptimality nor the distance to the optimizer is directly observable during the run. Practitioners therefore rely on observable proxies, such as empirical losses computed from validation samples [35], and use these proxies together with the evolving stochastic trajectory to decide whether to tune, continue, or stop the algorithm. These proxies are themselves random and dependent on the particular samples used to construct them. They may be informative, but favorable behavior of a proxy is not itself a certificate for suboptimality or distance to the optimizer.

Existing fixed-horizon guarantees do not certify this adaptive mode of operation: a high-probability statement proved at a deterministic time $T$ does not, in general, remain valid at a data-dependent stopping time selected after repeatedly inspecting the iterates, stochastic gradients, or validation losses [16, 1, 39]. The stochastic-oracle assumptions may remain unchanged, but the stopping time is now coupled to the random fluctuations revealed along the observed run. Among the many iterations examined, an unusually favorable fluctuation may be precisely what triggers termination. Therefore, a guarantee calibrated for one predetermined time does not automatically account for this selection over time.

One can preserve validity by fixing the stopping horizon in advance. Given a fixed-time highprobability bound and a target accuracy ε, one may solve for a time $T _ { \varepsilon }$ at which the bound falls below ε, and then run SGD for exactly that many iterations. This produces a valid certified stopping rule, but typically a very conservative one. Because $T _ { \varepsilon }$ is obtained by inverting a worst-case guarantee before the trajectory is observed, it reflects the most adverse behavior permitted by the assumptions rather than the behavior of the realized run. In practice, SGD can perform substantially better than these guarantees predict, sometimes by several orders of magnitude, so the resulting deterministic horizon may require many more iterations than would be suficient on the realized run. This limitation is not resolved by choosing a stepsize schedule that attains the optimal worst-case rate: such optimality describes performance over a broad problem class, whereas stopping asks whether the particular run has already reached a prescribed accuracy. Existing theory is therefore indispensable for ex ante algorithm design, but it is not itself an online stopping mechanism.

This leaves a clear gap between rigorous but potentially conservative fixed-horizon guarantees and adaptive early-stopping procedures that generally lack a certificate. Bridging this gap requires more than another fixed-time convergence bound. It requires a theory in which the evolving trajectory is itself allowed to determine, with statistical validity, when suficient progress has been made. This suggests a diferent perspective on stochastic optimization: a running SGD trajectory can be viewed as a sequential experiment. Its iterates and stochastic gradients do not merely update the decision variable; they also provide evidence about the unknown optimization error. The stopping problem then becomes one of determining whether the observed trajectory has produced enough evidence to certify that a prescribed accuracy has been reached. This leads to the central question of the paper:

Can SGD certify, from its own realized trajectory, when it should stop?

The statistical theory underlying this question has deep roots in anytime-valid sequential inference, which studies how to construct inferential guarantees that remain valid when observations are examined sequentially and the decision to continue or stop depends on what has already been observed [47, 49, 39, 38]. In the present setting, the relevant objects are confidence sequences: observable, trajectorydependent upper bounds that, with high probability, bound a target quantity, such as suboptimality or distance to the optimizer, simultaneously at every iteration [7, 27, 24]. SGD may therefore be monitored continuously and stopped at the first time the bound falls below a prescribed tolerance, while the certificate remains valid at the selected stopping time. Constructing sharp confidence sequences for SGD requires optimization-specific tools because the unknown errors evolve recursively with the iterates and are driven by the same stochastic gradients from which the certificates must be constructed.

A first step toward meeting this challenge was taken by Aolaritei and Jordan [2], who developed anytime-valid confidence sequences and certified stopping rules for SGD under general convexity and smooth nonconvexity. Specifically, that work constructed fully observable certificates for suboptimality in the convex setting and for stationarity in the smooth nonconvex setting. These certificates allow the algorithm to be monitored continuously and stopped at a data-dependent time while retaining a valid guarantee.

In the present paper, we substantially expand this line of work by developing a considerably sharper theory for strongly convex stochastic optimization that exploits the contractive structure of the $\mathrm { { d y - } }$ namics. Retaining trajectory adaptivity and time-uniform validity in this setting requires new constructions, including a recursive confidence-sequence argument and an empirical Bernstein theory for adapted processes with time-varying conditional means and growing predictable ranges. The resulting fully observable certificates bound the squared distance of the last iterate to the optimizer and the suboptimality of a weighted-average iterate. They recover the canonical $1 / t$ rate, up to the generally unavoidable log log t factors associated with time-uniform validity [8, 24], and adapt to the stochastic behavior observed along the realized trajectory. In our numerical experiments, the corresponding stopping rules terminate orders of magnitude earlier than natural deterministic horizons obtained by inverting conventional fixed-time guarantees.

## 1.1 Problem formulation

We consider stochastic optimization problems of the form

$$
\operatorname* { m i n } _ { x \in \mathcal { X } } f ( x ) ,
$$

where $\boldsymbol { \mathcal { X } } \subseteq \mathbb { R } ^ { d }$ is a closed convex set and $f : \mathcal { X } \to \mathbb { R }$ is diferentiable.<sup>1</sup> Throughout, we assume that the objective satisfies the following condition.

Assumption 1.1 (Strong convexity). The function $f$ is µ-strongly convex on $\mathcal { X } .$ , meaning that

$$
f ( y ) \geq f ( x ) + \langle \nabla f ( x ) , y - x \rangle + { \frac { \mu } { 2 } } \left. y - x \right. ^ { 2 } , \qquad \forall x , y \in { \mathcal { X } } .
$$

Under Assumption 1.1, the minimizer is unique; we denote it by $x ^ { \star }$ . We study projected stochastic gradient descent (SGD), defined recursively by

$$
x _ { t + 1 } = \Pi _ { \mathcal { X } } \big ( x _ { t } - \eta _ { t } g _ { t } \big ) , \qquad t \geq 1 ,\tag{1}
$$

and initialized from $x _ { 1 } \in \mathcal { X }$ almost surely. Here, $\Pi _ { \mathcal { X } }$ denotes the Euclidean projection onto $\mathcal { X } , \{ g _ { t } \} _ { t \ge 1 }$ are stochastic gradients, and $\{ \eta _ { t } \} _ { t \ge 1 }$ is a stepsize sequence. We work with the natural filtration

$$
\mathcal { F } _ { 0 } : = \sigma ( x _ { 1 } ) , \qquad \mathcal { F } _ { t } : = \sigma ( x _ { 1 } , g _ { 1 } , \dotsc , x _ { t } , g _ { t } ) , \qquad t \geq 1 .
$$

Thus, $\mathcal { F } _ { t }$ contains the complete SGD trajectory observed through iteration t. We assume that each stepsize is selected using only information available before the corresponding stochastic gradient is observed.

Assumption 1.2 (Predictable stepsizes). For every $t \geq 1$ , the stepsize $\eta _ { t }$ is $\mathcal { F } _ { t - 1 } { \mathrm { - m e a s u r a b l e } }$

Assumption 1.2 permits both deterministic and data-dependent stepsize rules, provided they depend only on the past trajectory. It includes constant and diminishing schedules, as well as adaptive choices based on previously observed iterates and stochastic gradients (e.g., AdaGrad-type schedules and related adaptive methods [13, 50]). The stochastic gradients satisfy the following standard unbiasedness and tail conditions.

Assumption 1.3 (Stochastic gradients). For every $t \geq 1$ , the stochastic gradient $g _ { t }$ satisfies:

(i) Unbiasedness:

$$
\mathbb { E } [ g _ { t } \mid \mathcal { F } _ { t - 1 } ] = \nabla f ( x _ { t } ) .
$$

(ii) Conditional sub-Gaussian noise: defining $\xi _ { t } : = g _ { t } - \nabla f ( x _ { t } )$ , there exists a constant $\sigma ^ { 2 } > 0$ such that, for every $\lambda \in \mathbb { R }$ and every $u \in \mathbb { R } ^ { d }$ 2

$$
\mathbb { E } \left[ \exp \bigl ( \lambda \langle u , \xi _ { t } \bigr \rangle \bigr ) \big | \mathcal { F } _ { t - 1 } \right] \leq \exp \left( \frac { \lambda ^ { 2 } \sigma ^ { 2 } \left\| u \right\| ^ { 2 } } { 2 } \right) \qquad \mathrm { a l m o s t ~ s u r e l y } .\tag{2}
$$

Assumption 1.3(i) is the standard conditional unbiasedness requirement in stochastic approximation [32]. Assumption 1.3(ii) imposes a conditional sub-Gaussian bound on the gradient noise while allowing its distribution to vary along the trajectory. This condition is satisfied by Gaussian and many lighttailed noise models. It also holds whenever $\| g _ { t } \| \leq G$ almost surely, in which case one may take $\sigma ^ { 2 } = G ^ { 2 }$ by the conditional Hoefding lemma [23].

To ensure that the resulting contraction factors are positive, we impose an upper bound on the stepsizes after an initial period.

Assumption 1.4 (Upper bound on stepsizes). There exists a deterministic integer $t _ { 0 } \geq 1$ such that

$$
0 < \eta _ { t } < \frac { 1 } { 2 \mu } \qquad \mathrm { a l m o s t ~ s u r e l y ~ f o r ~ e v e r y ~ } t \ge t _ { 0 } .
$$

Assumption 1.4 is satisfied by the standard diminishing stepsize schedules used for strongly convex stochastic optimization. It is imposed only from the deterministic time $t _ { 0 }$ onward, allowing arbitrary predictable stepsizes during the initial phase of the algorithm. The index $t _ { 0 }$ serves as the starting point of our time-uniform analysis.

Our goal is to provide anytime-valid, trajectory-adaptive guarantees for the progress of SGD. Given the stochastic gradients, predictable stepsizes, and observed iterates, we seek upper bounds on suitable performance measures that remain valid simultaneously over the entire run. We formalize this objective through the following definition.

Definition 1.5 (Anytime-valid upper confidence sequence). Let $\{ Q _ { t } \} _ { t \ge t _ { 0 } }$ be a real-valued stochastic process. For any $\alpha \in ( 0 , 1 )$ , a sequence of random variables $\{ U _ { t } ( \alpha ) \} _ { t \ge t _ { 0 } }$ is an anytime-valid upper confidence sequence for $\{ Q _ { t } \} _ { t \ge t _ { 0 } }$ at level $1 - \alpha$ if

$$
\mathbb { P } \left( \forall t \ge t _ { 0 } : \ Q _ { t } \le U _ { t } ( \alpha ) \right) \ge 1 - \alpha .
$$

When $Q _ { t }$ represents a performance measure of the current optimization trajectory, such as the squared distance to the optimizer or a notion of suboptimality, Definition 1.5 provides a sequence of upper bounds that are valid simultaneously over time. Consequently, the bounds may be monitored continuously and evaluated at a time selected from the observed trajectory without invalidating the coverage guarantee.

To evaluate our confidence sequences, we assume access to the problem constants that enter their construction. In particular, the strong-convexity parameter $\mu$ is assumed known, and $\sigma ^ { 2 }$ is a known valid upper bound on the conditional sub-Gaussian proxy in Assumption 1.3(ii). In the bounded-gradient setting considered later, we assume that a known constant $G$ satisfies $\| g _ { t } \| \leq G$ almost surely, in which case one may take $\sigma ^ { 2 } = G ^ { 2 }$ as explained earlier. The fully observable constructions additionally require an observable upper bound $R _ { 0 }$ on $\| \boldsymbol { x } _ { t _ { 0 } } - \boldsymbol { x } ^ { \star } \| ^ { 2 }$ ; under bounded gradients, the choice $R _ { 0 } = G ^ { 2 } / \mu ^ { 2 }$ is always valid. By contrast, $x ^ { \star } , f ( x ^ { \star } )$ , and the performance measures considered below are not assumed known. Importantly, $G$ and $R _ { 0 }$ need only be valid upper bounds rather than tightly calibrated values: the numerical experiments in Section 5.5 show that, for the empirical-Bernstein confidence sequences, the efect of conservative choices of these quantities is largely transient.

We now introduce the two performance measures considered in this paper and the stopping rules induced by their corresponding upper confidence sequences.

Squared distance to the optimizer. Our first performance measure is

$$
Z _ { t } : = \| x _ { t } - x ^ { \star } \| ^ { 2 } .\tag{3}
$$

This quantity measures the accuracy of the current iterate and therefore defines a natural last-iterate stopping criterion: ideally, the algorithm would stop as soon as $Z _ { t } \leq \varepsilon$ . Since $x ^ { \star }$ is unknown, however, this criterion is not directly observable. An anytime-valid upper confidence sequence provides a natural trajectory-adaptive surrogate.

At time $t ,$ after observing $g _ { t } .$ , the algorithm computes $x _ { t + 1 }$ , so $Z _ { t + 1 }$ is $\mathcal { F } _ { t }$ -measurable. We therefore seek a computable, $\mathcal { F } _ { t } .$ -measurable upper bound for $Z _ { t + 1 }$ . Suppose that $\{ U _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha ) \} _ { t \geq t _ { 0 } }$ is an anytimevalid upper confidence sequence for $\{ Z _ { t + 1 } \} _ { t \ge t _ { 0 } }$ with this property. Then $U _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha )$ certifies the accuracy of the newly computed iterate $x _ { t + 1 }$ , leading to the stopping rule

$$
\tau _ { \varepsilon } ^ { \mathrm { d i s t } } ( \alpha ) : = \operatorname* { i n f } \left\{ t \geq t _ { 0 } : U _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha ) \leq \varepsilon \right\} ,\tag{4}
$$

with the convention inf $\sigma : = \infty$ . Consequently, $\tau _ { \varepsilon } ^ { \mathrm { d i s t } } ( \alpha )$ is a stopping time with respect to $\{ \mathcal { F } _ { t } \} _ { t \ge t _ { 0 } }$ When finite, it returns $x _ { \tau _ { \varepsilon } ^ { \mathrm { d i s t } } ( \alpha ) + 1 }$ , the first iterate certified to lie within squared distance $\varepsilon$ of $x ^ { \star }$ .

Weighted average suboptimality. Our second performance measure is the linearly weighted average suboptimality

$$
\bar { F } _ { t } : = \frac { 1 } { S _ { t } } \sum _ { s = t _ { 0 } } ^ { t } s \bigl ( f ( x _ { s } ) - f ( x ^ { \star } ) \bigr ) , \qquad S _ { t } : = \sum _ { s = t _ { 0 } } ^ { t } s , \qquad t \geq t _ { 0 } .\tag{5}
$$

The corresponding weighted average iterate is

$$
\bar { x } _ { t } : = \frac { 1 } { S _ { t } } \sum _ { s = t _ { 0 } } ^ { t } s x _ { s } .\tag{6}
$$

By convexity of $f ,$

$$
f ( { \bar { x } } _ { t } ) - f ( x ^ { \star } ) \leq { \bar { F } } _ { t } .
$$

Thus, an upper bound on $\bar { F } _ { t }$ certifies the suboptimality of the observable iterate ${ \bar { x } } _ { t }$ . The linearly increasing weights emphasize later iterates and were introduced in [26], together with the stepsize $\eta _ { t } = 2 / ( \mu ( t + 1 ) )$ , to recover the optimal convergence rate for strongly convex stochastic optimization; we adopt the same stepsize for this performance metric.

Suppose that $\left\{ U _ { t } ^ { \mathrm { s u b } } ( \alpha ) \right\} _ { t \geq t _ { 0 } }$ is an anytime-valid upper confidence sequence for $\{ \bar { F } _ { t } \} _ { t \ge t _ { 0 } }$ and that $U _ { t } ^ { \mathrm { s u b } } ( \alpha )$ is $\mathcal { F } _ { t } .$ -measurable for every t > $t \geq t _ { 0 }$ . Using the same target accuracy $\varepsilon > 0$ , define

$$
\tau _ { \varepsilon } ^ { \mathrm { s u b } } ( \alpha ) : = \operatorname* { i n f } \left\{ t \geq t _ { 0 } : U _ { t } ^ { \mathrm { s u b } } ( \alpha ) \leq \varepsilon \right\} ,\tag{7}
$$

with the convention inf $\sigma : = \infty$ . When finite, this rule returns $\bar { x } _ { \tau _ { \varepsilon } ^ { \mathrm { s u b } } ( \alpha ) }$ , the first weighted average iterate certified to be ε-suboptimal.

The validity of the two stopping rules follows directly from the simultaneous-coverage property.

Proposition 1.6 (Certified stopping rules). Fix $\alpha \in ( 0 , 1 )$ and $\varepsilon > 0$

(i) If $\{ U _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha ) \} _ { t \geq t _ { 0 } }$ is an anytime-valid upper confidence sequence for $\{ Z _ { t + 1 } \} _ { t \ge t _ { 0 } }$ , then

$$
\mathbb { P } \left( \{ \tau _ { \varepsilon } ^ { \mathrm { d i s t } } ( \alpha ) < \infty \} \cap \left\{ \left\| x _ { \tau _ { \varepsilon } ^ { \mathrm { d i s t } } ( \alpha ) + 1 } - x ^ { \star } \right\| ^ { 2 } > \varepsilon \right\} \right) \leq \alpha .
$$

(ii) If $\{ U _ { t } ^ { \mathrm { s u b } } ( \alpha ) \} _ { t \geq t _ { 0 } }$ is an anytime-valid upper confidence sequence for $\{ \bar { F } _ { t } \} _ { t \ge t _ { 0 } }$ , then

$$
\begin{array} { r } { \mathbb { P } \left( \{ \tau _ { \varepsilon } ^ { \mathrm { s u b } } ( \alpha ) < \infty \} \cap \left\{ f \bigl ( \bar { x } _ { \tau _ { \varepsilon } ^ { \mathrm { s u b } } ( \alpha ) } \bigr ) - f ( x ^ { \star } ) > \varepsilon \right\} \right) \leq \alpha . } \end{array}
$$

Thus, for each stopping rule separately, with probability at least $1 - \alpha$ , whenever the stopping time is finite, the returned iterate satisfies the prescribed accuracy requirement.

The result follows immediately from Definition 1.5. We therefore omit the proof and refer to $[ 2 ,$ Theorem 3.1], where the same stopping-time argument is carried out. The aim of the current paper is to construct the upper confidence sequences $U _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha )$ and $U _ { t } ^ { \mathrm { s u b } } ( \alpha )$ explicitly from the observed iterates, stochastic gradients, and stepsizes $\left\{ \left( x _ { s } , g _ { s } , \eta _ { s } \right) \right\} _ { s = t _ { 0 } } ^ { t }$

## 1.2 Contributions

Our main contributions can be summarized as follows:

(i) Confidence sequences under sub-Gaussian noise. Through a time-uniform Hoefding argument, we construct fully observable, trajectory-adaptive confidence sequences for the two performance measures introduced above. These confidence sequences can be computed online from the observed stochastic gradients and stepsizes. In the worst case, they match the optimal $1 / t$ rate for strongly convex stochastic optimization, up to the iterated-logarithmic cost of time-uniform validity, while remaining sensitive to the observed trajectory and therefore potentially smaller than the corresponding worst-case bounds.

1. Squared distance to the optimizer. We construct an anytime-valid upper confidence sequence $\{ \widehat { U } _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha ) \} _ { t \geq t _ { 0 } }$ whose stochastic term depends explicitly on the conditional sub-Gaussian variance proxy $\sigma ^ { 2 }$ . If $\| g _ { t } \| \leq G$ almost surely and $\eta _ { t } = 1 / ( \mu t )$ , then its worst-case decay satisfies

$$
\widehat { U } _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha ) = O \left( \frac { G ^ { 2 } } { \mu ^ { 2 } } \frac { ( \log ( 1 / \alpha ) + \log \log ( e t ) ) } { t } \right) .\tag{8}
$$

2. Weighted average suboptimality.<sup>2</sup> Building on the observable distance confidence sequence, we construct an anytime-valid upper confidence sequence $\{ \widehat { U } _ { t } ^ { \mathrm { s u b } } ( \alpha ) \} _ { t \geq t _ { 0 } }$ for weighted average suboptimality. If $\| g _ { t } \| \leq G$ almost surely and $\eta _ { t } = 2 / ( \mu ( t + 1 ) )$ , then its worst-case decay satisfies

$$
\widehat { U } _ { t } ^ { \mathrm { s u b } } ( \alpha ) = O \left( \frac { G ^ { 2 } } { \mu } \frac { ( \log ( 1 / \alpha ) + \log \log ( e t ) ) } { t } \right) .\tag{9}
$$

(ii) Refined confidence sequences under bounded stochastic gradients. Through a novel time-uniform empirical Bernstein argument (see item (iv)), we refine both confidence sequences under the assumption that $\| g _ { t } \| \leq G$ almost surely. The resulting bounds replace the fixed worstcase variance proxy $\sigma ^ { 2 } = G ^ { 2 }$ in the Hoefding construction by the realized squared stochasticgradient magnitudes $\| g _ { t } \| ^ { 2 }$ , together with an additional correction that decays at the faster $t ^ { - 3 / 2 }$ rate, up to iterated-logarithmic factors. The refined confidence sequences are fully observable and preserve the worst-case rates in equations (8) and (9). At the same time, they adapts more sharply to the observed trajectory: in our numerical experiments, the resulting confidence sequences are generally one order of magnitude smaller than their Hoefding-based counterparts. This refinement constitutes the central contribution of the paper.

(iii) Minibatch extensions. We extend both constructions to minibatch SGD with b conditionally independent stochastic gradients per iteration. For the Hoefding confidence sequences, minibatching reduces the conditional sub-Gaussian variance proxy from $\sigma ^ { 2 }$ to $\sigma ^ { 2 } / b$ . For the empirical Bernstein confidence sequences, we develop a minibatch analogue that pools the observed quadratic variation of the individual stochastic gradients and adapts to the largest eigenvalue of the realized second-moment matrix of each minibatch. The minibatch empirical Bernstein boundary has a leading term that improves at the usual $1 / \sqrt { b }$ rate, together with a lower-order correction that improves at the faster $1 / b$ rate. In our numerical experiments, the improvement relative to the corresponding worst-case bounds grows with the minibatch size, reaching between two and three orders of magnitude for the larger minibatch sizes considered. The gap relative to existing ex ante guarantees with explicit constants can be substantially larger still; see Figure 1.

(iv) A time-uniform empirical Bernstein inequality with growing predictable range. To support the refinement in item (ii), we establish a general time-uniform empirical Bernstein inequality for adapted processes with time-varying conditional means and predictable increment ranges that may grow without bound. This feature is essential for our SGD application: in both the squared-distance and weighted-average suboptimality analyses, the relevant predictable ranges can scale as $\sqrt { t }$ under the worst-case convergence rate. Existing empirical Bernstein confidencesequence constructions, such as Howard et al. [24, Theorem 4], allow the conditional means to vary but assume a fixed range scale, and therefore do not directly cover this setting. When the predictable range evolves over time, the admissible exponential parameter evolves with it. Starting from the pointwise exponential inequality of Fan et al. [15, Lemma 4.1], we overcome this dificulty through predictable truncation and simultaneous stitching over dyadic scales of the observed quadratic process and the running predictable range. The resulting explicit boundary adapts jointly to both quantities and is nondecreasing in each of them.

![](images/de5504c90846ab68a432c1b28b73f9055a7e5914a5ef9506783678b888ee36b1.jpg)

![](images/6dbaa299009eac907426d9b735f49c713639f22c8ee8734599457a181b8ccba6.jpg)  
Figure 1: Trajectory-adaptive certification of SGD on an SVM problem with minibatch size $b = 2 5 6$ and confidence level 99%. H and EB denote our Hoefding- and empirical-Bernstein-based confidence sequences, respectively; worst-case is the trajectory-independent version of our Hoefding construction obtained by replacing the realized squared stochastic-gradient norms $\| g _ { t } \| ^ { 2 }$ by their uniform upper bound $G ^ { 2 }$ , while RSS12 denotes the finite-horizon high-probability bound of Rakhlin et al. [36].<sup>4</sup> Left: squared distance to the optimizer. Right: suboptimality of the weighted average. The gray curves show the corresponding ground truth. In the right panel, crosses and vertical lines mark the first iteration at which each bound falls below the target $\varepsilon = 1 0 ^ { - 3 }$ , and hence the time at which that bound first certifies the prescribed accuracy. Further details are provided in Section 5.

## 1.3 Related work

We first discuss the statistical foundations of our approach and their historical connection to stochastic approximation, before turning to the relevant background in strongly convex stochastic optimization.

Anytime-valid inference and stochastic approximation. The statistical framework underlying our analysis is anytime-valid sequential inference, which develops measures of evidence and uncertainty that remain valid under continuous monitoring and data-dependent stopping. Its motivating pathology is classical: a procedure may be statistically valid at every predetermined sample size and yet lose validity when observations are repeatedly inspected and sampling stops after a favorable fluctuation. An early illustration appears in Feller’s 1940 discussion of extra-sensory perception, where repeated sampling could turn chance fluctuations into apparently significant evidence [16]. Anscombe later famously described this practice as “sampling to a foregone conclusion” [1], while Robbins placed the same phenomenon within the broader theory of sequential experimentation [39]. The martingale foundations go back to Ville [47], while Wald developed the classical theory of sequential hypothesis testing [49]. Subsequent work by Darling, Robbins, Siegmund, Lai, and others developed confidence sequences that are valid at any and all times, boundary-crossing methods, and tests of power one [7, 9, 40, 42, 27]. After remaining a comparatively small niche for decades, these ideas reemerged rapidly around 2017 under the modern framework of safe anytime-valid inference [38]. During 2018– 2020, several independent lines of work with closely aligned motivations and techniques developed the modern theory of e-values, and the term itself was jointly adopted in its present measure-theoretic sense in 2020; see [48, 44, 19, 37] and the references therein. The terminology is recent, but the underlying objects trace back to likelihood ratios, nonnegative martingales, and Ville’s betting interpretation.

A central technical ingredient in this framework is the construction of time-uniform boundaries that control a stochastic process uniformly over time. Such boundaries are commonly obtained from exponential supermartingales through mixtures or stitching. The method of mixtures integrates over the exponential tuning parameter; in the literature on self-normalized processes, this technique is also known as pseudo-maximization and provides a way to adapt to a random intrinsic time [10, 11, 12]. Stitching instead allocates tuning parameters and error probabilities across successive intrinsic-time epochs and combines the resulting bounds through a union bound. Epoch-based arguments of this kind go back at least to Darling and Robbins [8], while the approach was developed systematically for modern confidence sequences by Howard et al. [24]. Our empirical Bernstein analysis follows the stitching route, but must also accommodate a predictable range that evolves with the SGD trajectory; we therefore stitch simultaneously over the observed quadratic process and the running range scale.

This history also includes a close connection to stochastic approximation. Robbins and Monro introduced stochastic approximation [41], while the Robbins–Siegmund almost-supermartingale theorem later became a standard tool for establishing the convergence of stochastic approximation algorithms [43]. Lai likewise contributed to adaptive stochastic approximation jointly with Robbins [28]. The two traditions therefore share important intellectual origins, but their modern theories developed largely separately. Aolaritei and Jordan [2] brought these perspectives together by constructing trajectoryadaptive confidence sequences and certified stopping rules for convex and smooth nonconvex SGD. The present paper continues this line of work in the strongly convex setting, where contractivity permits substantially sharper certificates but also requires new recursive and self-normalized sequential arguments.

Strongly convex stochastic optimization. Classical work by Polyak and Juditsky established the asymptotic optimality of trajectory averaging under suitable regularity conditions [34]. Building on this asymptotic theory, Bach and Moulines [4] gave nonasymptotic analyses of standard and averaged SGD for smooth convex objectives, showing that inverse-time stepsizes attain the optimal rate under strong convexity, while slower decay combined with averaging is more robust to stepsize calibration and remains efective in the absence of strong convexity. Subsequent work obtained optimal $O ( 1 / T )$ rates through specialized algorithms and averaging schemes. Hazan and Kale [22] proposed an epoch-based method for strongly convex stochastic optimization without smoothness assumptions, while Ghadimi and Lan [17, 18] developed accelerated and multistage schemes for strongly convex composite problems, together with optimal or nearly optimal complexity bounds and large-deviation guarantees. For standard SGD, Rakhlin et al. [36] showed that smoothness yields the optimal $O ( 1 / T )$ rate, while sufix averaging recovers the same rate without smoothness. They also obtained a finite-horizon high-probability squared-distance bound valid simultaneously for all $t \leq T$ , with an additional iterated-logarithmic factor. Further weighted-averaging schemes attaining the optimal rate were developed by Lacoste-Julien et al. [26], using linearly increasing weights for projected stochastic subgradient descent, and by Nedic and Lee [31] for stochastic mirror descent. Extensions include adaptation to unknown local strong convexity in logistic regression [3] and trajectory averaging on Riemannian manifolds for geodesically strongly convex objectives [46].

The gap between averaged and last-iterate performance motivated a complementary line of work. For nonsmooth strongly convex objectives, Shamir and Zhang [45] established an $O ( \log T / T )$ lastiterate rate in expectation and proposed an online averaging scheme attaining the optimal $O ( 1 / T )$ rate. Harvey et al. [20] proved the corresponding high-probability last-iterate bound, established its tightness through a matching deterministic example, and obtained an optimal $O ( 1 / T )$ high-probability guarantee for sufix averaging. For the linearly weighted average of Lacoste-Julien et al. [26], Harvey et al. [21] established the optimal $O ( \log ( 1 / \delta ) / T )$ high-probability rate. Jain et al. [25] instead designed horizon-dependent stepsize sequences that transfer optimal guarantees for averaged SGD to the last iterate. More recently, Liu and Zhou [29] developed a unified last-iterate analysis in expectation and with high probability across general domains, composite objectives, non-Euclidean geometries, and both convex and strongly convex settings.

Recent work has derived high-probability convergence rates for strongly convex SGD that hold uniformly over time. Pham et al. [33] obtained such a rate for the last-iterate squared distance, matching the order in (8), while Chen et al. [6] established the corresponding order in (9) for the lastiterate objective gap under smoothness. Although time-uniform, these guarantees remain ex ante and worst-case: their envelopes depend only on the iteration count and global problem parameters, rather than adapting to the realized trajectory. For the accuracy-based stopping problem considered here, time uniformity alone therefore ofers no advantage over a finite-horizon guarantee: thresholding either bound yields a deterministic worst-case iteration budget, while the time-uniform bound may be more conservative because of its additional iterated-logarithmic or constant-factor cost, as illustrated by the comparison between Pham et al. [33] and Rakhlin et al. [36]. The confidence sequences developed here instead adapt to the stochastic fluctuations observed along the realized trajectory, while providing fully observable certificates that can be evaluated throughout the run and used for online stopping, with the same decay orders in the worst case. This distinction is quantitatively substantial in our experiments: in Figure 1, our trajectory-adaptive distance certificate is roughly five orders of magnitude smaller than the finite-horizon bound of Rakhlin et al. [36], with an even larger gap relative to the time-uniform bound of Pham et al. [33].

## 1.4 Organization and notation

Organization. Section 2 develops our confidence sequences under conditional sub-Gaussian noise using a time-uniform Hoefding argument. Section 3 refines these constructions under bounded stochastic gradients through a time-uniform empirical Bernstein approach. Section 4 extends both approaches to minibatch SGD. Finally, Section 5 numerically evaluates the resulting confidence sequences and their certified stopping rules. All proofs are deferred to Appendix A.

Notation. For $x , y \in \mathbb { R } ^ { d } , \| x \|$ denotes the Euclidean norm and $\langle x , y \rangle$ the standard inner product. For a closed convex set $\mathcal { X } \subseteq \mathbb { R } ^ { d } , \Pi _ { \mathcal { X } } ( z )$ denotes the Euclidean projection of z onto . For a symmetric matrix $M , \lambda _ { \mathrm { m a x } } ( M )$ denotes its largest eigenvalue. Given a filtration $\{ \mathcal { F } _ { t } \} _ { t \ge 0 }$ , a process $\{ X _ { t } \}$ is adapted if $X _ { t }$ is $\mathcal { F } _ { t } { \mathrm { - m e a s u r a b l e } }$ and predictable if $X _ { t }$ is $\mathcal { F } _ { t - 1 } { \mathrm { - m e a s u r a b l e } }$ . Conditional expectations are written $\mathbb { E } [ \cdot \mid \mathcal { F } _ { t - 1 } ]$ , and we adopt the convention inf $\mathcal { O } = \infty$ . Unless a base is indicated explicitly, all logarithms are natural. For two nonnegative sequences $a _ { t }$ and $b _ { t }$ , we write $a _ { t } = O ( b _ { t } )$ if there exists a constant $C < \infty$ such that $a _ { t } \leq C b _ { t }$ throughout the stated range; we write $a _ { t } \lesssim b _ { t }$ synonymously with $a _ { t } = O ( b _ { t } )$ and $a _ { t } \asymp b _ { t }$ if $a _ { t } \lesssim b _ { t }$ and $b _ { t } \lesssim a _ { t }$ . Unless stated otherwise, implicit constants may depend on fixed problem parameters but are independent of $t , \alpha$ , and the minibatch size b whenever these quantities vary in the corresponding statement. All probabilities and expectations are taken with respect to the underlying algorithmic randomness.

## 2 Confidence Sequences under Sub-Gaussian Noise

We construct anytime-valid upper confidence sequences for the two performance measures of interest: the last-iterate squared distance $Z _ { t + 1 }$ and the suboptimality of the weighted-average iterate, $f ( { \bar { x } } _ { t } ) -$ $f ( x ^ { \star } )$ . Although the two guarantees arise from diferent SGD recursions, both reduce to controlling partial sums of directional gradient noise terms. We first isolate the common time-uniform concentration argument, then construct the observable distance confidence sequence, which will in turn be used to make the suboptimality confidence sequence fully observable.

The common probabilistic ingredient underlying both constructions is a conditional exponential moment bound of sub-Gaussian form. Consider an adapted sequence of real-valued random variables $\{ X _ { t } \} _ { t \ge t _ { 0 } }$ and a nonnegative predictable sequence $\{ v _ { t } \} _ { t \ge t _ { 0 } }$ . Suppose that, for some $\gamma \in \mathbb { R }$ ，

$$
\mathbb { E } \left[ \exp \left( \lambda X _ { t } \right) \big | \mathcal { F } _ { t - 1 } \right] \leq \exp \left( 2 ^ { \gamma } \lambda ^ { 2 } v _ { t } \right) \qquad \mathrm { a l m o s t ~ s u r e l y }
$$

for every $\lambda > 0$ and every $t \geq t _ { 0 }$ . The predictable quantity $v _ { t }$ serves as a conditional variance proxy for $X _ { t }$ , using only information available before $X _ { t }$ is observed, while $\gamma$ records the constant arising in the particular application. For each fixed $\lambda > 0$ , the conditional exponential bound implies that

$$
\exp \left( \lambda \sum _ { s = t _ { 0 } } ^ { t } X _ { s } - 2 ^ { \gamma } \lambda ^ { 2 } \sum _ { s = t _ { 0 } } ^ { t } v _ { s } \right)
$$

is a nonnegative supermartingale. The cumulative variance proxy $\scriptstyle \sum _ { s = t _ { 0 } } ^ { t } v _ { s }$ records the conditional fluctuation scale accumulated over the process. In sequential inference, such a cumulative process is often called an intrinsic time: the scale of the concentration boundary for the partial sum adapts to the accumulated variance proxy, rather than being indexed solely by the iteration count.

This exponential supermartingale is the starting point for a time-uniform concentration bound. Ville’s inequality controls its running maximum and, after rearranging, yields for each fixed $\lambda > 0$ a linear upper boundary for the partial sums that holds simultaneously over time [47]. No single choice of λ, however, is well calibrated across all possible scales of the intrinsic time: diferent values of $\lambda \ : \mathrm { y }$ ield sharper boundaries in diferent ranges of the cumulative variance proxy. Dyadic stitching combines these fixed-λ boundaries over successive dyadic scales, while allocating the error probability across the scales. The result is a curved boundary that adapts to the realized intrinsic time and remains valid uniformly over the infinite horizon.

The following lemma formalizes this construction in the precise form used throughout the section. It is a dyadic specialization of the stitching framework developed by Howard et al. [24].

Lemma 2.1 (Anytime-valid stitched sub-Gaussian bound). Let $\{ \mathcal { F } _ { t } \} _ { t \ge t _ { 0 } - 1 }$ be a filtration. Let $\{ X _ { t } \} _ { t \ge t _ { 0 } }$ be adapted, and let $\{ v _ { t } \} _ { t \ge t _ { 0 } }$ be nonnegative and predictable. Define

$$
\bar { X } _ { t _ { 0 } - 1 } : = 0 , \qquad \bar { X } _ { t } : = \sum _ { s = t _ { 0 } } ^ { t } X _ { s } , \qquad V _ { t _ { 0 } - 1 } : = 0 , \qquad V _ { t } : = \sum _ { s = t _ { 0 } } ^ { t } v _ { s } .
$$

Fix $\gamma \in \mathbb { R }$ . Suppose that, for every $\lambda > 0$ and every $t \geq t _ { 0 }$

$$
\begin{array} { r } { \mathbb { E } \left[ \exp \left( \lambda X _ { t } - 2 ^ { \gamma } \lambda ^ { 2 } v _ { t } \right) \Big | \mathcal { F } _ { t - 1 } \right] \leq 1 . } \end{array}
$$

For $t \geq t _ { 0 } - 1$ , let

$$
V _ { t , \mathrm { e f f } } : = \operatorname* { m a x } \{ V _ { t } , 1 \} , \qquad m _ { t } : = \lceil \log _ { 2 } V _ { t , \mathrm { e f f } } \rceil .
$$

Then, for every $\alpha \in ( 0 , 1 )$ ,

$$
\mathbb { P } \left( \forall t \geq t _ { 0 } : \ \bar { X } _ { t } \leq 2 ^ { ( \gamma + 3 ) / 2 } \sqrt { V _ { t , \mathrm { e f f } } \left( \log \frac { 1 } { \alpha } + \log ( ( m _ { t } + 1 ) ( m _ { t } + 2 ) ) \right) } \right) \geq 1 - \alpha .
$$

Lemma 2.1 provides the generic concentration step, but applying it to SGD requires a problemspecific analysis of the algorithmic dynamics. For each performance measure, we derive a recursion that isolates the contribution of the gradient noise as a partial sum of directional terms, and then establish the required conditional exponential moment bound using a predictable variance proxy.

## 2.1 Last-iterate squared distance to the optimizer

We begin by constructing an upper confidence sequence for the last-iterate squared distance to the optimizer. Recall from (3) that $Z _ { t } = \| x _ { t } - x ^ { \star } \| ^ { 2 }$ . To account for the time-varying contraction induced by strong convexity, define

$$
a _ { t } : = 1 - 2 \mu \eta _ { t } , \qquad A _ { t _ { 0 } - 1 } : = 1 , \qquad A _ { t } : = \prod _ { s = t _ { 0 } } ^ { t } a _ { s } ^ { - 1 } , \quad t \geq t _ { 0 } .\tag{10}
$$

By Assumption 1.4, $a _ { t } \in \mathsf { \Gamma } ( 0 , 1 )$ almost surely for every $t \geq t _ { 0 }$ , so $A _ { t }$ is well defined. Following the notation of Lemma 2.1, for $t \geq t _ { 0 }$ define the adapted term

$$
X _ { t } : = 2 A _ { t } \eta _ { t } \langle x ^ { \star } - x _ { t } , \xi _ { t } \rangle , \qquad \bar { X } _ { t } : = \sum _ { s = t _ { 0 } } ^ { t } X _ { s } .
$$

With these definitions, Lemma C.2 yields the following decomposed upper bound on the last-iterate distance: for every $t \geq t _ { 0 }$

$$
Z _ { t + 1 } \leq \frac { 1 } { A _ { t } } \left[ Z _ { t _ { 0 } } + \sum _ { s = t _ { 0 } } ^ { t } A _ { s } \eta _ { s } ^ { 2 } \left. g _ { s } \right. ^ { 2 } + \bar { X } _ { t } \right] .\tag{11}
$$

This decomposition follows from the standard squared-distance Lyapunov argument for projected stochastic approximation; see, e.g., Nemirovski et al. [32, Section 2.1]. A closely related unrolling of the contractive SGD recursion for the classical stepsize $\eta _ { t } = 1 / ( \mu t )$ appears in Rakhlin et al. [36, Appendix B.7, Lemma 6]; the decomposition (11) records the same telescoping mechanism for general predictable stepsizes.

Thus, following (11), the last-iterate distance to the optimizer is controlled by an initialization term, an observable accumulation of squared stochastic gradients, and a partial sum of directional gradient noise terms. A time-uniform bound on $X _ { t }$ therefore translates directly into an upper confidence sequence for $Z _ { t + 1 }$ . To control this partial sum, define

$$
v _ { t } ^ { \mathrm { d i s t } } : = \sigma ^ { 2 } A _ { t } ^ { 2 } \eta _ { t } ^ { 2 } Z _ { t } , \qquad t \geq t _ { 0 } .
$$

The sequence $\{ v _ { t } ^ { \mathrm { d i s t } } \}$ is predictable because $x _ { t } , \eta _ { t }$ , and $A _ { t }$ are $\mathcal { F } _ { t } .$ <sub>−1</sub>-measurable. Applying the conditional sub-Gaussian bound in Assumption 1.3(ii) with the predictable direction $2 A _ { t } \eta _ { t } ( x ^ { \star } - x _ { t } )$ gives

$$
\mathbb { E } \left[ e ^ { \lambda X _ { t } } \Big | \mathcal { F } _ { t - 1 } \right] \le e ^ { 2 \lambda ^ { 2 } v _ { t } ^ { \mathrm { d i s t } } } \qquad \mathrm { a l m o s t ~ s u r e l y } ,
$$

for every $\lambda \geq 0$ and every $t \geq t _ { 0 }$ . Thus, $v _ { t } ^ { \mathrm { d i s t } }$ is the predictable intrinsic-time increment in the generic construction, and Lemma 2.1 applies with $\gamma = 1$

To express the resulting confidence sequences compactly, for $v \geq 0$ define

$$
v _ { \mathrm { e f f } } : = \mathrm { m a x } \{ v , 1 \} , \qquad m ( v ) : = \lceil \log _ { 2 } v _ { \mathrm { e f f } } \rceil ,\tag{12}
$$

and, for $\alpha \in ( 0 , 1 )$ , let

$$
\mathfrak { H } _ { \alpha } ( v ) : = \sqrt { v _ { \mathrm { e f f } } \left( \log \frac { 1 } { \alpha } + \log \left( ( m ( v ) + 1 ) ( m ( v ) + 2 ) \right) \right) } .\tag{13}
$$

The function ${ \mathfrak { H } } _ { \alpha }$ collects the common dependence of the stitched boundaries on the intrinsic time and its dyadic scale, and is nondecreasing in v. For the distance process, define

$$
V _ { t } ^ { \mathrm { d i s t } } : = \sum _ { s = t _ { 0 } } ^ { t } v _ { s } ^ { \mathrm { d i s t } } , \qquad t \geq t _ { 0 } .
$$

This is the cumulative intrinsic time associated with the directional gradient noise terms. Since the conditional exponential bound corresponds to $\gamma = 1$ , Lemma 2.1 guarantees that

$$
\bar { X } _ { t } \leq 4 \mathfrak { H } _ { \alpha } \left( V _ { t } ^ { \mathrm { d i s t } } \right)
$$

simultaneously for every $t \geq t _ { 0 }$ with probability at least $1 - \alpha$ . Combining this bound with (11) gives the following proposition.

Proposition 2.2 (Confidence sequence for $\| x _ { t + 1 } - x ^ { \star } \| ^ { 2 } )$ . Suppose Assumptions 1.1–1.4 hold. Then, for every $\alpha \in ( 0 , 1 )$ , the process

$$
U _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha ) : = \frac { 1 } { A _ { t } } \left[ Z _ { t _ { 0 } } + \sum _ { s = t _ { 0 } } ^ { t } A _ { s } \eta _ { s } ^ { 2 } \left. g _ { s } \right. ^ { 2 } + 4 \mathfrak { H } _ { \alpha } \left( V _ { t } ^ { \mathrm { d i s t } } \right) \right] , \qquad t \ge t _ { 0 } ,\tag{14}
$$

is an upper confidence sequence for the last-iterate squared distance to the optimizer, in the sense that

$$
\begin{array} { r } { \mathbb { P } \left( \forall t \geq t _ { 0 } : \ Z _ { t + 1 } \leq U _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha ) \right) \geq 1 - \alpha . } \end{array}
$$

Although the squared-gradient accumulation in (14) is observable from the SGD trajectory, the confidence sequence is not yet fully observable: the initial distance $Z _ { t _ { 0 } }$ is unknown, and the the boundary term $4 \mathfrak { H } _ { \alpha } \left( V _ { t } ^ { \mathrm { d i s t } } \right)$ depends, through $V _ { t } ^ { \mathrm { d i s t } }$ , on the unknown distances $\{ Z _ { s } \} _ { s = t _ { 0 } } ^ { t }$ . If is compact, an immediate remedy is to replace every $Z _ { s }$ by the uniform bound diam $( \mathcal { X } ) ^ { 2 }$ . This produces a fully observable confidence sequence, but discards the contraction of the iterates toward the optimizer. As shown in Remark 2.5, the resulting bound exhibits $t ^ { - 1 / 2 }$ polynomial decay, rather than the $\mathrm { n e a r - } 1 / t$ decay achieved by the recursive construction below.

To obtain a sharper observable bound, we exploit the simultaneous-coverage guarantee of Proposition 2.2. Define the event

$$
E _ { \alpha } : = \left\{ \forall t \geq t _ { 0 } : \ Z _ { t + 1 } \leq U _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha ) \right\} .
$$

On $E _ { \alpha } ,$ which has probability at least $1 - \alpha ,$ every subsequent unknown distance is bounded by the corresponding upper confidence bound, that is, $Z _ { s } \leq U _ { s } ^ { \mathrm { d i s t } } ( \alpha )$ for all $s \geq t _ { 0 } + 1$ . The bounds $U _ { s } ^ { \mathrm { d i s t } } ( \alpha )$ are themselves not observable, so they cannot be inserted directly into $V _ { t } ^ { \mathrm { d i s t } }$ . Instead, starting from an

observable upper bound on $Z _ { t _ { 0 } }$ , we construct observable envelopes $\widehat { U } _ { s } ^ { \mathrm { d i s t } } ( \alpha )$ recursively. Suppose that, up to the current time, these envelopes satisfy $U _ { s } ^ { \mathrm { d i s t } } ( \alpha ) \leq \widehat { U } _ { s } ^ { \mathrm { d i s t } } ( \alpha )$ on $E _ { \alpha }$ . Then

$$
Z _ { s } \le U _ { s } ^ { \mathrm { d i s t } } ( \alpha ) \le \widehat { U } _ { s } ^ { \mathrm { d i s t } } ( \alpha ) ,
$$

so the observable envelopes can be used to upper bound the unknown distances in the intrinsic-time process. The resulting observable intrinsic time dominates $V _ { t } ^ { \mathrm { d i s t } }$ and, since ${ \mathfrak { H } } _ { \alpha }$ is nondecreasing, produces a boundary term that dominates 4 ${ \mathfrak { H } } _ { \alpha } \left( V _ { t } ^ { \mathrm { d i s t } } \right)$ . Substituting this larger boundary term into (14) yields the next observable envelope and preserves the required domination, thereby closing the construction recursively.

We now formalize this construction. Fix $\alpha \in ( 0 , 1 )$ , and let $R _ { 0 }$ be any $\mathcal { F } _ { t _ { 0 } }$ <sub>−1</sub>-measurable quantity satisfying $R _ { 0 } \geq Z _ { t _ { 0 } }$ almost surely. Define recursively

$$
R _ { t _ { 0 } } ^ { \mathrm { { d i s t } } } : = R _ { 0 } , \qquad R _ { s } ^ { \mathrm { { d i s t } } } : = \widehat { U } _ { s } ^ { \mathrm { { d i s t } } } ( \alpha ) , \quad s \geq t _ { 0 } + 1 .\tag{15}
$$

For every $t \geq t _ { 0 }$ , define the observable intrinsic time

$$
\widehat { V } _ { t } ^ { \mathrm { d i s t } } : = \sum _ { s = t _ { 0 } } ^ { t } \sigma ^ { 2 } A _ { s } ^ { 2 } \eta _ { s } ^ { 2 } R _ { s } ^ { \mathrm { d i s t } } ,\tag{16}
$$

and, using the notation in (12) and the stitched boundary function in (13), define the observable upper envelope

$$
\widehat { U } _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha ) : = \frac { 1 } { A _ { t } } \left[ R _ { 0 } + \sum _ { s = t _ { 0 } } ^ { t } A _ { s } \eta _ { s } ^ { 2 } \left. g _ { s } \right. ^ { 2 } + 4 \mathfrak { H } _ { \alpha } \left( \widehat { V } _ { t } ^ { \mathrm { d i s t } } \right) \right] , \qquad t \geq t _ { 0 } .\tag{17}
$$

The recursion is well defined. When the recursion is evaluated at time t, the proxies $R _ { t _ { 0 } } ^ { \mathrm { d i s t } } , \ldots , R _ { t } ^ { \mathrm { d } }$ ist are already available: $R _ { t _ { 0 } } ^ { \mathrm { d i s t } } = R _ { 0 }$ , while, for $t \geq t _ { 0 } + 1 , R _ { t } ^ { \mathrm { d i s t } } = \widehat { U } _ { t } ^ { \mathrm { d i s t } } ( \alpha )$ was computed at time $t - 1$ They determine the observable intrinsic time $\widehat { V } _ { t } ^ { \mathrm { d i s t } }$ , and hence the boundary term $4 \Im _ { \alpha } \left( \widehat { V } _ { t } ^ { \mathrm { d i s t } } \right)$ and the next envelope $\widehat { U } _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha ) = R _ { t + 1 } ^ { \mathrm { d i s t } }$ . The following theorem establishes that the resulting process is fully observable and retains the simultaneous-coverage guarantee.

Theorem 2.3 (Observable confidence sequence for $\| x _ { t + 1 } - x ^ { \star } \| ^ { 2 } )$ . Suppose Assumptions 1.1–1.4 hold, and let $R _ { 0 }$ be an $\mathcal { F } _ { t _ { 0 } - 1 ^ { - } } \mathrm { m e a s u r a b l e }$ quantity satisfying $R _ { 0 } \geq Z _ { t _ { 0 } }$ almost surely. Then, for every $\alpha \in$ $( 0 , 1 )$ , the recursively defined process $\{ \widehat { U } _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha ) \} _ { t \geq t _ { 0 } }$ is well defined and fully observable from $R _ { 0 }$ and the SGD trajectory. Moreover,

$$
\begin{array} { r } { \mathbb { P } \left( \forall t \geq t _ { 0 } : \ Z _ { t + 1 } \leq \widehat { U } _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha ) \right) \geq 1 - \alpha . } \end{array}
$$

A key feature of Theorem 2.3 is that it accommodates any predictable stepsize sequence satisfying the standing conditions, including stepsizes chosen adaptively from the observed trajectory. The confidence sequence is therefore not tied to a particular schedule or to an a priori convergence-rate calculation. To verify that the construction nevertheless recovers the expected behavior under a standard choice, we next specialize to $\eta _ { t } = 1 / ( \mu t )$ and, for simplicity, assume that the stochastic gradients are uniformly bounded by G. This condition also provides a deterministic observable initialization: unbiasedness and conditional Jensen’s inequality imply $\| \nabla f ( x _ { t _ { 0 } } ) \| \le G$ , while strong convexity and first-order optimality yield $\mu \left\| \boldsymbol { x } _ { t _ { 0 } } - \boldsymbol { x } ^ { \star } \right\| \leq \| \nabla f ( \boldsymbol { x } _ { t _ { 0 } } ) |$ . Consequently,

$$
Z _ { t _ { 0 } } \leq R _ { 0 } : = { \frac { G ^ { 2 } } { \mu ^ { 2 } } } \qquad \mathrm { a l m o s t \ s u r e l y } .\tag{18}
$$

The following proposition shows that, under this classical specialization, the observable envelope decays at the $1 / t$ rate in the worst case, up to an iterated-logarithmic factor typical of anytime-valid analysis.

![](images/caacb9cfa150de55f07dcd4a1f5ea10c9db46b6bc0ee3b66155cd1a297c6e5fd.jpg)

Proposition 2.4 (Decay rate of the observable confidence sequence for $\| \boldsymbol { x } _ { t + 1 } - \boldsymbol { x } ^ { \star } \| ^ { 2 } )$ . Fix $\alpha \in ( 0 , 1 )$ and suppose Assumptions 1.1–1.4 hold. Let $t _ { 0 } = 3$ and $\eta _ { t } = 1 / ( \mu t )$ for every $t \geq 3$ , and suppose that $\| g _ { t } \| \leq G$ almost surely for every $t \geq 3$ . Initialize the recursive construction using $R _ { 0 }$ from (18). Define

$$
B : = 1 + \frac { G ^ { 2 } } { \mu ^ { 2 } } + \frac { \sigma ^ { 2 } } { \mu ^ { 2 } } , \qquad L _ { t } ( \alpha ) : = 1 + \log \frac { 1 } { \alpha } + \log \log ( e t ) + \log \log ( e B ) .
$$

Then there exists a universal constant $^ 5 ~ C > 0$ such that, almost surely,

$$
\widehat { U } _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha ) \leq C B \frac { L _ { t + 1 } ( \alpha ) } { t + 1 } \qquad \mathrm { f o r ~ e v e r y ~ } t \geq 4 .
$$

Proposition 2.4 reveals why the recursive construction preserves the strongly convex rate. The recursion does more than replace the latent distances by observable upper bounds: each newly computed envelope is fed back into the subsequent intrinsic-time increments. In this sense, the construction is self-localizing, since a shrinking certified radius leads to a corresponding reduction in the variance scale governing future deviations. Under $\eta _ { t } = 1 / ( \mu t )$ , this feedback reduces the intrinsic-time growth from order $t ^ { 3 }$ under a fixed global radius to order $t ^ { 2 }$ , up to slowly varying factors. Because the stitched boundary scales with the square root of the intrinsic time, it is therefore of order $t ,$ again up to slowly varying factors. After normalization by $A _ { t } \asymp t ^ { 2 }$ , this yields the $1 / t$ decay established above. The following remark makes this improvement explicit by comparing the recursive construction with the simpler diameter-based substitution.

Remark 2.5 (Decay rate under the diameter-based observable bound). Suppose the conditions of Proposition 2.4 hold and that $\mathcal { X }$ is compact with diameter $D ,$ so that $Z _ { t } \le D ^ { 2 }$ . Then, under $\eta _ { t } = 1 / ( \mu t )$ with $t _ { 0 } = 3$ and $A _ { t } = t ( t - 1 ) / 2$

$$
V _ { t } ^ { \mathrm { { d i s t } } } \leq \sigma ^ { 2 } D ^ { 2 } \sum _ { s = 3 } ^ { t } A _ { s } ^ { 2 } \eta _ { s } ^ { 2 } \lesssim t ^ { 3 } .
$$

Consequently, using the monotonicity of ${ \mathfrak { H } } _ { \alpha } .$

$$
\frac { 4 \mathfrak { H } _ { \alpha } \left( V _ { t } ^ { \mathrm { d i s t } } \right) } { A _ { t } } \lesssim t ^ { - 1 / 2 } \sqrt { L _ { t } ( \alpha ) } .
$$

Thus, directly replacing the distances $Z _ { t }$ by the diameter bound yields only a $t ^ { - 1 / 2 } \mathrm { - t y p e }$ decay, up to stitched-logarithmic factors, whereas the recursive observable envelope preserves the near-1/t rate.

The preceding proposition was stated for the stepsize $\eta _ { t } = 1 / ( \mu t )$ . In the sequel, it is also useful to have the analogous rate statement for the common alternative schedule $\eta _ { t } = 2 / ( \mu ( t + 1 ) )$ . We record this variant here.

Remark 2.6 (Alternative polynomial stepsize). The same rate conclusion holds for the stepsize

$$
\eta _ { t } = \frac { 2 } { \mu ( t + 1 ) } .
$$

In this case, $1 - 2 \mu \eta _ { t } = ( t - 3 ) / ( t + 1 )$ , so the natural analysis start time is $t _ { 0 } = 4$ . Following similar steps to those in the proof of Proposition 2.4, there exists a universal constant $C < \infty$ such that, for every $t \geq 4$ 2

$$
\widehat { U } _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha ) \leq C B \frac { L _ { t + 1 } ( \alpha ) } { t + 1 } \qquad \mathrm { a l m o s t ~ s u r e l y } .
$$

The proof is similar to the proof of Proposition 2.4 and is omitted.

## 2.2 Suboptimality of the weighted-average iterate

We next consider the suboptimality of the weighted-average iterate and construct an upper confidence sequence for $f ( { \bar { x } } _ { t } ) - f ( x ^ { \star } )$ . Recalling the definitions in (5) and (6), the analysis follows the same basic structure as for the last-iterate distance to the optimizer: the objective gap is bounded by an initialization term, an observable accumulation of squared stochastic gradients, and a partial sum of directional gradient noise terms that can again be controlled using Lemma 2.1.

For this construction, we take $t _ { 0 } \geq 4$ and specialize to the stepsize $\eta _ { t } = 2 / ( \mu ( t + 1 ) )$ considered in Remark 2.6. This choice is naturally aligned with the linear weights defining $\bar { x } _ { t } { : }$ after the one-step suboptimality inequality is weighted by $t ,$ the resulting distance terms telescope across consecutive times. Define

$$
E _ { t } : = t \langle x ^ { \star } - x _ { t } , \xi _ { t } \rangle , \qquad \bar { E } _ { t } : = \sum _ { s = t _ { 0 } } ^ { t } E _ { s } , \qquad t \ge t _ { 0 } .
$$

The resulting weighted decomposition is recorded in Lemma C.3 and gives

$$
f ( \bar { x } _ { t } ) - f ( x ^ { \star } ) \leq \frac { 1 } { S _ { t } } \left[ \frac { 1 } { \mu } \sum _ { s = t _ { 0 } } ^ { t } \| g _ { s } \| ^ { 2 } + \frac { \mu t _ { 0 } ( t _ { 0 } - 1 ) } { 4 } Z _ { t _ { 0 } } + \bar { E } _ { t } \right] .\tag{19}
$$

This decomposition follows from the standard weighted telescoping argument for strongly convex SGD; see Lacoste-Julien et al. [26, Section 3.2]. We retain the directional gradient-noise term explicitly, rather than taking expectations, for the anytime-valid analysis.

The first term on the right-hand side of (19) is observable from the SGD trajectory, the second is an initialization term, and the final term accumulates the gradient noise in the directions $x ^ { \star } - x _ { t }$ Thus, as in the distance analysis, a time-uniform upper bound on $\bar { E } _ { t }$ translates directly into an upper confidence sequence for the weighted-average suboptimality.

To control this final term, define the predictable intrinsic-time increment

$$
v _ { t } ^ { \mathrm { s u b } } : = \sigma ^ { 2 } t ^ { 2 } Z _ { t } , \qquad t \geq t _ { 0 } .\tag{20}
$$

The sequence $\left\{ E _ { t } \right\}$ is adapted, while $\{ v _ { t } ^ { \mathrm { s u b } } \}$ is predictable because $x _ { t }$ is $\mathcal { F } _ { t - 1 }$ -measurable. Applying the conditional sub-Gaussian bound in Assumption 1.3(ii) with the predictable direction $t ( x ^ { \star } - x _ { t } )$ yields

$$
\mathbb { E } \left[ e ^ { \lambda E _ { t } } \Big | \mathcal { F } _ { t - 1 } \right] \leq \exp \left( \frac { \lambda ^ { 2 } v _ { t } ^ { \mathrm { s u b } } } { 2 } \right) \qquad \mathrm { a l m o s t ~ s u r e l y }
$$

for every $\lambda \geq 0$ and every $t \geq t _ { 0 }$ . Since the coeficient in the conditional exponent is $1 / 2$ , this is precisely the setting of Lemma 2.1 with $\gamma = - 1$

Using the notation in (12) and the stitched-boundary function in (13), define, for every $t \geq t _ { 0 }$

$$
V _ { t } ^ { \mathrm { s u b } } : = \sum _ { s = t _ { 0 } } ^ { t } v _ { s } ^ { \mathrm { s u b } } .
$$

In this application, $V _ { t } ^ { \mathrm { s u b } }$ is the cumulative intrinsic time associated with the weighted directional-noise terms. Since the conditional exponential bound corresponds to $\gamma = - 1$ , Lemma 2.1 guarantees that

$$
\bar { E } _ { t } \leq 2 \mathfrak { H } _ { \alpha } \left( V _ { t } ^ { \mathrm { s u b } } \right)
$$

simultaneously for every $t \geq t _ { 0 }$ with probability at least $1 - \alpha$ . Combining this bound with (19) gives the following proposition.

Proposition 2.7 (Confidence sequence for $f ( { \bar { x } } _ { t } ) - f ( x ^ { \star } ) )$ . Suppose Assumptions 1.1–1.4 hold. Let $t _ { 0 } \geq 4$ and $\eta _ { t } = 2 / ( \mu ( t + 1 ) )$ for every $t \geq t _ { 0 }$ . Then, for every $\alpha \in ( 0 , 1 )$ , the process

$$
U _ { t } ^ { \mathrm { s u b } } ( \alpha ) : = \frac { 1 } { S _ { t } } \left[ \frac 1 \mu \sum _ { s = t _ { 0 } } ^ { t } \| g _ { s } \| ^ { 2 } + \frac { \mu t _ { 0 } ( t _ { 0 } - 1 ) } { 4 } Z _ { t _ { 0 } } + 2 \mathfrak { H } _ { \alpha } \left( V _ { t } ^ { \mathrm { s u b } } \right) \right] , \qquad t \ge t _ { 0 } ,\tag{21}
$$

is an upper confidence sequence for the weighted-average suboptimality, in the sense that

$$
\begin{array} { r } { \mathbb { P } \left( \forall t \geq t _ { 0 } : \ f ( \bar { x } _ { t } ) - f ( x ^ { \star } ) \leq U _ { t } ^ { \mathrm { s u b } } ( \alpha ) \right) \geq 1 - \alpha . } \end{array}
$$

Proposition 2.7 is not yet fully observable, because both the initialization term and the intrinsic time depend on the unknown distances to the optimizer. The key diference from the preceding subsection is that no second recursive construction is needed. Theorem 2.3 already provides observable upper bounds on the entire distance sequence, which can be substituted directly into the suboptimality bound. We allocate error probability $\alpha / 2$ to the distance confidence sequence and $\alpha / 2$ to the stitched bound for the weighted directional-noise process. On the intersection of the two corresponding events, every unknown distance is dominated by its observable envelope, and the resulting suboptimality bound holds simultaneously over time. A union bound then gives overall coverage at least $1 - \alpha$ . This coupling has only a mild cost: replacing α by $\alpha / 2$ adds log $2$ to the confidence logarithms appearing inside the square-root boundaries.

Recall the observable distance proxies defined in (15). To make their dependence on the confidence level explicit, for $\beta \in ( 0 , 1 )$ write

$$
R _ { t _ { 0 } } ^ { \mathrm { d i s t } } ( \beta ) : = R _ { 0 } , \qquad R _ { s } ^ { \mathrm { d i s t } } ( \beta ) : = \widehat { U } _ { s } ^ { \mathrm { d i s t } } ( \beta ) , \quad s \geq t _ { 0 } + 1 .\tag{22}
$$

On the simultaneous-coverage event of Theorem 2.3 at level $\beta ,$ these proxies satisfy

$$
Z _ { s } \leq R _ { s } ^ { \mathrm { d i s t } } ( \beta ) \qquad \mathrm { f o r ~ e v e r y ~ } s \geq t _ { 0 } .
$$

For $t \geq t _ { 0 }$ , define the observable intrinsic time

$$
\widehat { V } _ { t } ^ { \mathrm { s u b } } ( \beta ) : = \sum _ { s = t _ { 0 } } ^ { t } \sigma ^ { 2 } s ^ { 2 } R _ { s } ^ { \mathrm { d i s t } } ( \beta ) .\tag{23}
$$

On the distance-coverage event at level $\beta ,$ the observable intrinsic time dominates its latent counterpart for every $t \geq t _ { 0 }$ . Since ${ \mathfrak { H } } _ { \beta }$ is nondecreasing,

$$
2 \mathfrak { H } _ { \beta } \left( V _ { t } ^ { \mathrm { s u b } } \right) \leq 2 \mathfrak { H } _ { \beta } \left( \widehat { V } _ { t } ^ { \mathrm { s u b } } ( \beta ) \right) \qquad \mathrm { f o r ~ e v e r y ~ } t \geq t _ { 0 } .
$$

The same distance proxies also control the initialization term, since $Z _ { t _ { 0 } } \le R _ { t _ { 0 } } ^ { \mathrm { d i s t } } ( \beta ) = R _ { 0 }$ . Thus, the distance confidence sequence supplies both ingredients needed to make the suboptimality certificate observable. Given $\alpha \in ( 0 , 1 )$ , define the observable suboptimality envelope

$$
\widehat { U } _ { t } ^ { \mathrm { s u b } } ( \alpha ) : = \frac { 1 } { S _ { t } } \left[ \frac { 1 } { \mu } \sum _ { s = t _ { 0 } } ^ { t } \| g _ { s } \| ^ { 2 } + \frac { \mu t _ { 0 } ( t _ { 0 } - 1 ) } { 4 } R _ { 0 } + 2 \mathfrak { H } _ { \alpha / 2 } \left( \widehat { V } _ { t } ^ { \mathrm { s u b } } ( \alpha / 2 ) \right) \right] , \qquad t \geq t _ { 0 } .\tag{24}
$$

The construction is observable: for $s \geq t _ { 0 } + 1$ , the proxy $R _ { s } ^ { \mathrm { d i s t } } ( \beta ) = \widehat { U } _ { s } ^ { \mathrm { d i s t } } ( \beta )$ is $\mathcal { F } _ { s }$ <sub>−1</sub>-measurable and is therefore available before $g _ { s }$ is observed. At the end of iteration t, all distance proxies in (22) with index at most t and all gradients $g _ { s }$ with $s \leq t$ are available, so (23) and (24) define an $\mathcal { F } _ { t }$ -measurable quantity.

Theorem 2.8 (Observable confidence sequence for $f ( \bar { x } _ { t } ) - f ( x ^ { \star } ) )$ . Suppose Assumptions 1.1–1.4 hold. Let $t _ { 0 } \geq 4$ and $\eta _ { t } = 2 / ( \mu ( t + 1 ) )$ for every $t \geq t _ { 0 }$ , and let $R _ { 0 }$ be an $\mathcal { F } _ { t _ { 0 } - 1 }$ -measurable quantity satisfying $R _ { 0 } \geq Z _ { t _ { 0 } }$ almost surely. Then, for every $\alpha \in ( 0 , 1 )$ , the process $\{ \widehat { U } _ { t } ^ { \mathrm { s u b } } ( \alpha ) \} _ { t \geq t _ { 0 } }$ is fully observable and satisfies

$$
\begin{array} { r } { \mathbb { P } \left( \forall t \geq t _ { 0 } : \ f ( \bar { x } _ { t } ) - f ( x ^ { \star } ) \leq \widehat { U } _ { t } ^ { \mathrm { s u b } } ( \alpha ) \right) \geq 1 - \alpha . } \end{array}
$$

We now turn to the decay of the observable suboptimality confidence sequence under the same bounded-gradient specialization used in Proposition 2.4. In particular, the deterministic initialization $R _ { 0 } = G ^ { 2 } / \mu ^ { 2 }$ from (18) remains available, although any sharper observable upper bound on $Z _ { t _ { 0 } }$ may be used instead. Since the present construction uses the stepsize $\eta _ { t } = 2 / ( \mu ( t + 1 ) )$ , the rate analysis combines Proposition 2.4 with its extension in Remark 2.6. The following proposition shows that the resulting observable suboptimality envelope recovers the optimal $1 / t$ rate in the worst case, up to iterated-logarithmic factors.

Proposition 2.9 (Decay rate of the observable confidence sequence for $f ( { \bar { x } } _ { t } ) - f ( x ^ { \star } ) )$ . Fix $\alpha \in ( 0 , 1 )$ and suppose Assumptions 1.1–1.4 hold. Let $R _ { 0 } = G ^ { 2 } / \mu ^ { 2 } , t _ { 0 } = 4$ , and $\eta _ { t } = 2 / ( \mu ( t + 1 ) )$ for every $t \geq 4$ and suppose that $\| g _ { t } \| \leq G$ almost surely for every $t \geq 4$ . Define

$$
B : = 1 + \frac { G ^ { 2 } } { \mu ^ { 2 } } + \frac { \sigma ^ { 2 } } { \mu ^ { 2 } } , \qquad B _ { \mathrm { s u b } } : = 1 + \mu B = 1 + \mu + \frac { G ^ { 2 } } { \mu } + \frac { \sigma ^ { 2 } } { \mu } ,
$$

and, for $t \geq 4$ , let

$$
L _ { t } ^ { \mathrm { s u b } } ( \alpha ) : = 1 + \log \frac { 1 } { \alpha } + \log \log ( e t ) + \log \log ( e B ) + \log \log ( e B _ { \mathrm { s u b } } ) .
$$

Then there exists a universal constant $C > 0$ such that, almost surely,

$$
\widehat { U } _ { t } ^ { \mathrm { s u b } } ( \alpha ) \leq C B _ { \mathrm { s u b } } \frac { L _ { t } ^ { \mathrm { s u b } } ( \alpha ) } { t } \qquad \mathrm { f o r ~ e v e r y ~ } t \geq 4 .
$$

Proposition 2.9 shows that passing from observable distance bounds to an observable suboptimality certificate does not degrade the optimal worst-case rate. Indeed, the $\mathrm { n e a r - } 1 / s$ distance envelopes make the suboptimality intrinsic time grow as $t ^ { 2 } ,$ up to slowly varying factors; the corresponding stitched boundary therefore grows as $t ,$ and division by $S _ { t } \asymp t ^ { 2 }$ yields the $\mathrm { n e a r - } 1 / t$ decay. Directly bounding objective suboptimality also improves the dependence on the strong-convexity parameter by one power: the terms $G ^ { 2 } / \mu ^ { 2 }$ and $\sigma ^ { 2 } / \mu ^ { 2 }$ in the distance certificate become $G ^ { 2 } / \mu$ and $\sigma ^ { 2 } / \mu$ . Consequently, when $\mu$ is small, the suboptimality certificate can become informative at a substantially earlier stage of the optimization trajectory.

## 3 Refined Confidence Sequences under Bounded Gradients

The preceding section constructed anytime-valid upper confidence sequences for the last-iterate squared distance $Z _ { t + 1 }$ and the suboptimality of the weighted-average iterate, $f ( \bar { x } _ { t } ) - f ( x ^ { \star } )$ , under a conditional sub-Gaussian assumption on the gradient noise. Under bounded stochastic gradients, conditional Hoefding’s lemma verifies this assumption with $\sigma ^ { 2 } = G ^ { 2 }$ , so those confidence sequences remain valid. With this choice, however, the intrinsic times entering the observable distance and suboptimality confidence sequences are driven by the worst-case proxy $G ^ { 2 }$ and therefore do not adapt to the realized magnitudes of the stochastic gradients. Because $G$ is a uniform bound, it can be substantially larger than the gradient magnitudes observed along a typical trajectory, making the stochastic contribution to the resulting certificates unnecessarily large.

The natural refinement is to let the stochastic boundaries respond to the gradients actually observed along the trajectory. Under the bounded-gradient assumption, an empirical Bernstein construction makes this possible by replacing the fixed proxy $G ^ { 2 }$ with quadratic processes involving the realized values of $\| g _ { t } \| ^ { 2 }$ . The relevant directional terms, however, have ranges that evolve with the SGD trajectory, so a fixed-range empirical Bernstein bound is not suficient for our purposes. We therefore develop a time-uniform empirical Bernstein inequality for a growing predictable range, which will provide the common concentration argument for both the last-iterate distance and weighted-average suboptimality. This development builds on the empirical Bernstein supermartingale construction of Howard et al. [24] and the exponential inequality of Fan et al. [15] underlying it. As emphasized in [24], this approach is technically closer to the literature on self-normalized bounds [10] than to classical empirical Bernstein arguments such as Maurer and Pontil [30]. The latter combine a concentration bound for the sample mean involving the true variance with a separate concentration bound for the sample variance. By contrast, Howard et al. [24] construct an exponential supermartingale that directly relates the centered process to an online empirical variance.

We impose the following assumption throughout this section.

Assumption 3.1 (Bounded stochastic gradients). There exists $G < \infty$ such that

$$
\left\| g _ { t } \right\| \leq G \qquad { \mathrm { a l m o s t ~ s u r e l y ~ f o r ~ a l l ~ } } t \geq t _ { 0 } .
$$

As shown in (18), Assumption 3.1 always yields the observable initialization $R _ { 0 } = G ^ { 2 } / \mu ^ { 2 }$ . This universal choice will be used below, although sharper observable bounds on $Z _ { t _ { 0 } }$ may be available in specific applications and can be substituted directly into the construction.

The concentration mechanism remains the same as in the preceding section: a one-step conditional exponential bound is lifted to a time-uniform boundary through a supermartingale argument and stitching. What changes is the compensating term. Rather than using a predictable variance proxy, the empirical Bernstein inequality below uses the realized square of the increment itself. This is the key mechanism that will allow the resulting confidence sequences to adapt to the stochastic gradients observed along the trajectory.

The following lemma isolates this one-step ingredient in the precise form needed below. Starting from a pointwise exponential inequality, it yields after conditional centering a bound in which the quadratic compensation depends on the realized square $X ^ { 2 }$ , while the admissible tuning parameter is restricted by the increment range b. These two features will drive the time-uniform construction that follows. To state the lemma, for $\theta \in [ 0 , 1 )$ define

$$
\psi _ { \mathrm { E } } ( \theta ) : = - \log ( 1 - \theta ) - \theta .\tag{25}
$$

We are now ready to state the lemma.

Lemma 3.2 (Conditional exponential moment bound). For every $\theta \in [ 0 , 1 )$ and every $x \geq - 1$ 2

$$
\exp \left( \theta x - \psi _ { \mathrm { E } } ( \theta ) x ^ { 2 } \right) \leq 1 + \theta x .\tag{26}
$$

Moreover,

$$
\psi _ { \mathrm { E } } ( \theta ) \leq { \frac { \theta ^ { 2 } } { 2 ( 1 - \theta ) } } .\tag{27}
$$

Consequently, let  be a sigma-field, let X be an integrable real-valued random variable satisfying $| X | \leq b$ almost surely for some deterministic $b > 0 .$ , and let $\lambda \in [ 0 , 1 / b )$ . Then

$$
\mathbb { E } \left[ \exp \left( \lambda \big ( X - \mathbb { E } [ X \mid \mathcal { G } ] \big ) - \frac { \lambda ^ { 2 } X ^ { 2 } } { 2 ( 1 - \lambda b ) } \right) \Bigg | \mathcal { G } \right] \leq 1 \qquad \mathrm { a l m o s t ~ s u r e l y } .\tag{28}
$$

The pointwise inequality (26) was first established in the proof of Fan et al. [15, Lemma 4.1]. In Howard et al. [24, Theorem 4], this inequality is applied conditionally to deviations from a predictable reference sequence, and the corresponding squared deviations are accumulated to construct a timeuniform empirical Bernstein confidence sequence. Equation (28) records the one-step conditional form needed here, with the predictable reference specialized to zero and the quadratic coeficient simplified using (27). A self-contained proof is provided in the appendix.

We now lift the one-step inequality in Lemma 3.2 to a time-uniform bound. Two evolving scales enter the construction. First, the quadratic compensation in (28) accumulates the realized squared increments. Second, the admissible tuning parameter depends on the range of the increment. In our applications this range is predictable and may grow with the trajectory, without admitting any finite uniform bound over the infinite horizon. A time-uniform boundary must therefore adapt simultaneously to the accumulated quadratic process and to the largest predictable range encountered so far. We achieve this by stitching over dyadic scales of both quantities.

For $v , b \geq 0$ , define the efective quadratic-process and range scales

$$
v _ { \mathrm { e f f } } : = \mathrm { m a x } \{ v , 1 \} , \qquad b _ { \mathrm { e f f } } : = \mathrm { m a x } \{ b , 1 \} ,\tag{29}
$$

and the corresponding dyadic indices

$$
k ( v ) : = \lceil \log _ { 2 } v _ { \mathrm { e f f } } \rceil , \qquad \ell ( b ) : = \lceil \log _ { 2 } b _ { \mathrm { e f f } } \rceil .\tag{30}
$$

Both indices take values in $\{ 0 , 1 , 2 , \ldots \}$ . As in Lemma 2.1, the unit floor prevents negative dyadic indices and keeps the boundary well defined when either argument is zero. For $\alpha \in ( 0 , 1 )$ , define

$$
\Gamma _ { \alpha } ( v , b ) : = \log \frac { 1 } { \alpha } + \log \bigl ( ( k ( v ) + 1 ) ( k ( v ) + 2 ) \bigr ) + \log \bigl ( ( \ell ( b ) + 1 ) ( \ell ( b ) + 2 ) \bigr ) ,\tag{31}
$$

and let

$$
\mathfrak { B } _ { \alpha } ( v , b ) : = 2 \sqrt { v _ { \mathrm { e f f } } \Gamma _ { \alpha } ( v , b ) } + 2 b _ { \mathrm { e f f } } \Gamma _ { \alpha } ( v , b ) .\tag{32}
$$

The two logarithmic scale terms in $\Gamma _ { \alpha }$ account for the simultaneous stitching over the quadratic process and the predictable range. The resulting boundary is the empirical Bernstein counterpart of the stitched sub-Gaussian boundary from the preceding section, with the additional range scale allowing the predictable increment bound to evolve without any uniform upper bound over time. To the best of our knowledge, this is the first time-uniform empirical Bernstein bound that allows the predictable range to vary with the history and grow without any finite uniform bound over the infinite horizon.

Theorem 3.3 (Empirical Bernstein bound with a growing predictable range). Fix a deterministic integer $t _ { 0 } \geq 1$ . Let $\{ \mathcal { F } _ { t } \} _ { t \ge t _ { 0 } - 1 }$ be a filtration. Let $\{ H _ { t } \} _ { t \ge t _ { 0 } }$ be a real-valued integrable adapted process, and let $\{ c _ { t } \} _ { t \ge t _ { 0 } }$ be an almost surely finite,<sup>6</sup> nonnegative predictable process such that

$$
| H _ { t } | \leq c _ { t } \qquad \mathrm { a l m o s t ~ s u r e l y ~ f o r ~ e v e r y ~ } t \geq t _ { 0 } .
$$

Define

$$
\overline { { H } } _ { t _ { 0 } - 1 } : = 0 , \qquad \overline { { H } } _ { t } : = \sum _ { s = t _ { 0 } } ^ { t } \left( H _ { s } - \mathbb { E } [ H _ { s } \mid \mathcal { F } _ { s - 1 } ] \right) ,
$$

and

$$
W _ { t _ { 0 } - 1 } : = 0 , \qquad W _ { t } : = \sum _ { s = t _ { 0 } } ^ { t } H _ { s } ^ { 2 } , \qquad C _ { t } : = \operatorname* { m a x } _ { t _ { 0 } \leq s \leq t } c _ { s } , \qquad t \geq t _ { 0 } .
$$

Then, for every $\alpha \in ( 0 , 1 )$ ,

$$
\begin{array} { r } { \mathbb { P } \left( \forall t \geq t _ { 0 } : \overline { { H } } _ { t } \leq \mathfrak { B } _ { \alpha } ( W _ { t } , C _ { t } ) \right) \geq 1 - \alpha . } \end{array}\tag{33}
$$

Moreover, the map $( v , b ) \mapsto \mathfrak { B } _ { \alpha } ( v , b )$ is nondecreasing in each argument.

Theorem 3.3 will now be used to refine the distance and weighted-suboptimality confidence sequences developed in the preceding section. We treat the two cases separately, beginning with the last-iterate distance to the optimizer.

## 3.1 Last-iterate squared distance to the optimizer

We begin with the last-iterate distance to the optimizer. Recall from Lemma C.2 that the distance analysis reduces to controlling the partial sum of the directional gradient noise terms $X _ { t } = 2 A _ { t } \eta _ { t } \langle x ^ { \star } -$ $x _ { t } , \xi _ { t } \rangle$ . To bring this term within the scope of Theorem 3.3, define the corresponding uncentered directional stochastic-gradient term

$$
H _ { t } ^ { \mathrm { d i s t } } : = 2 A _ { t } \eta _ { t } \langle x ^ { \star } - x _ { t } , g _ { t } \rangle , \qquad t \geq t _ { 0 } .\tag{34}
$$

By conditional unbiasedness, $H _ { t } ^ { \mathrm { { d i s t } } } - \mathbb { E } \left[ H _ { t } ^ { \mathrm { { d i s t } } } \Big | \mathcal { F } _ { t - 1 } \right] = X _ { t }$ . Thus, the centered process appearing in Theorem 3.3 is exactly the stochastic term already isolated by the distance decomposition. Under Assumption 3.1,

$$
\left| H _ { t } ^ { \mathrm { d i s t } } \right| \leq 2 G A _ { t } \eta _ { t } \sqrt { Z _ { t } } .
$$

For the canonical stepsize $\eta _ { t } \asymp 1 / t$ , we have $A _ { t } \asymp t ^ { 2 }$ , while the distance confidence sequence of the preceding section gives $Z _ { t } \lesssim L _ { t } / t$ . Hence the predictable range can grow as $\sqrt { t L _ { t } }$ , and in particular need not remain bounded over the infinite horizon. This is precisely the setting covered by Theorem 3.3. Accordingly, define the latent quadratic and range processes

$$
V _ { t } ^ { \mathrm { { d i s t } , E B } } : = \sum _ { s = t _ { 0 } } ^ { t } { \left( H _ { s } ^ { \mathrm { { d i s t } } } \right) ^ { 2 } } = 4 \sum _ { s = t _ { 0 } } ^ { t } { A _ { s } ^ { 2 } \eta _ { s } ^ { 2 } \langle g _ { s } , x ^ { \star } - x _ { s } \rangle ^ { 2 } } ,\tag{35}
$$

$$
B _ { t } ^ { \mathrm { { d i s t } , E B } } : = 2 G \operatorname* { m a x } _ { t _ { 0 } \leq s \leq t } A _ { s } \eta _ { s } \sqrt { Z _ { s } } .\tag{36}
$$

These are the distance-specific counterparts of the quadratic process $W _ { t }$ and the running predictable range $C _ { t }$ in Theorem 3.3. The generic theorem also requires $H _ { t } ^ { \mathrm { d i s t } }$ to be integrable. Since $Z _ { t } \le G ^ { 2 } / \mu ^ { 2 }$ almost surely, it is enough to assume

$$
\mathbb { E } \left[ A _ { t } \eta _ { t } \right] < \infty \qquad \mathrm { f o r ~ e v e r y ~ d e t e r m i n i s t i c ~ } t \ge t _ { 0 } .\tag{37}
$$

This mild technical condition is automatic for the deterministic stepsize schedules considered below.

Combining Lemma C.2 with Theorem 3.3 gives the latent empirical Bernstein confidence sequence

$$
U _ { t + 1 } ^ { \mathrm { d i s t } , \mathrm { E B } } ( \alpha ) : = \frac { 1 } { A _ { t } } \left[ Z _ { t _ { 0 } } + \sum _ { s = t _ { 0 } } ^ { t } A _ { s } \eta _ { s } ^ { 2 } \left. g _ { s } \right. ^ { 2 } + \mathfrak { B } _ { \alpha } \left( V _ { t } ^ { \mathrm { d i s t } , \mathrm { E B } } , B _ { t } ^ { \mathrm { d i s t } , \mathrm { E B } } \right) \right] , \qquad t \ge t _ { 0 } ,\tag{38}
$$

which satisfies

$$
\begin{array} { r } { \mathbb { P } \left( \forall t \geq t _ { 0 } : \ Z _ { t + 1 } \leq U _ { t + 1 } ^ { \mathrm { d i s t , E B } } ( \alpha ) \right) \geq 1 - \alpha . } \end{array}
$$

This is the empirical Bernstein counterpart of Proposition 2.2 from the preceding section. The argument follows the same overall logic, with Theorem 3.3 providing the corresponding concentration step. We defer the formal proof of this confidence-sequence guarantee to Proposition B.1 in the appendix.

Remark 3.4 (Comparison with the sub-Gaussian construction). This empirical Bernstein refinement admits a direct comparison with the sub-Gaussian confidence sequence from Proposition 2.2. The leading square-root term in $\mathfrak { B } _ { \alpha } \left( V _ { t } ^ { \mathrm { d i s t , E B } } , B _ { t } ^ { \mathrm { d i s t , E B } } \right)$ is the analogue of $4 \mathfrak { H } _ { \alpha } \left( V _ { t } ^ { \mathrm { d i s t } } \right)$ . Under the boundedgradient choice $\sigma ^ { 2 } = G ^ { 2 }$ , the latter uses the fixed worst-case scale $G ^ { 2 }$ , whereas the former is driven by the realized directional quantities $\langle g _ { s } , x ^ { \star } - x _ { s } \rangle ^ { 2 } \leq \| g _ { s } \| ^ { 2 } Z _ { s }$ . In the worst case, when $\| g _ { s } \| = G$ and the directional bound is attained, $V _ { t } ^ { \mathrm { d i s t , E B } } = 4 V _ { t } ^ { \mathrm { d i s t } }$ , so the leading factor 2 in $\mathfrak { B } _ { \alpha } \left( V _ { t } ^ { \mathrm { d i s t , E B } } , B _ { t } ^ { \mathrm { d i s t , E B } } \right)$ exactly recovers the outer factor 4 in $4 \mathfrak { H } _ { \alpha } \left( V _ { t } ^ { \mathrm { d i s t } } \right)$ , up to the additional iterated-logarithmic factor required to stitch over the predictable range. Away from this worst case, the leading term in $\mathfrak { B } _ { \alpha } \left( V _ { t } ^ { \mathrm { d i s t , E B } } , B _ { t } ^ { \mathrm { d i s t , E B } } \right)$ adapts to the realized stochastic-gradient magnitudes and directions, as desired. Beyond this additional logarithmic factor, the only new additive contribution is the linear range term in B<sub>α</sub> $\left( V _ { t } ^ { \mathrm { d i s t , E B } } , B _ { t } ^ { \mathrm { d i s t , E B } } \right)$ For the observable construction below, Proposition 3.6 shows that the corresponding range contribution is lower order, scaling as $( L _ { t } / t ) ^ { 3 / 2 }$ rather than $L _ { t } / t$

The bound is not yet observable, however, because $Z _ { t _ { 0 } }$ , the directions $x ^ { \star } - x _ { s } .$ , and the distances entering $B _ { t } ^ { \mathrm { d i s t , E B } }$ all depend on the unknown optimizer. We remove this dependence using the same recursive domination principle as in the preceding section, now applied simultaneously to the two arguments of $\mathfrak { B } _ { \alpha }$ . We formalize this construction. Fix $\alpha \in ( 0 , 1 )$ , and let $R _ { 0 }$ be any $\mathcal { F } _ { t _ { 0 } - 1 } .$ -measurable quantity satisfying $R _ { 0 } \geq Z _ { t _ { 0 } }$ almost surely. Define recursively the observable distance envelopes

$$
R _ { t _ { 0 } } ^ { \mathrm { E B } } : = R _ { 0 } , \qquad R _ { s } ^ { \mathrm { E B } } : = \widehat { U } _ { s } ^ { \mathrm { d i s t , E B } } ( \alpha ) , \quad s \geq t _ { 0 } + 1 .\tag{39}
$$

For $t \geq t _ { 0 }$ , set

$$
\widehat { V } _ { t } ^ { \mathrm { d i s t , E B } } : = 4 \sum _ { s = t _ { 0 } } ^ { t } A _ { s } ^ { 2 } \eta _ { s } ^ { 2 } \left. g _ { s } \right. ^ { 2 } R _ { s } ^ { \mathrm { E B } } ,\tag{40}
$$

$$
\widehat { B } _ { t } ^ { \mathrm { d i s t , E B } } : = 2 G \operatorname* { m a x } _ { t _ { 0 } \leq s \leq t } A _ { s } \eta _ { s } \sqrt { R _ { s } ^ { \mathrm { E B } } } ,\tag{41}
$$

and define

$$
\widehat { U } _ { t + 1 } ^ { \mathrm { d i s t , E B } } ( \alpha ) : = \frac { 1 } { A _ { t } } \left[ R _ { 0 } + \sum _ { s = t _ { 0 } } ^ { t } A _ { s } \eta _ { s } ^ { 2 } \| g _ { s } \| ^ { 2 } + \mathfrak { B } _ { \alpha } \left( \widehat { V } _ { t } ^ { \mathrm { d i s t , E B } } , \widehat { B } _ { t } ^ { \mathrm { d i s t , E B } } \right) \right] .\tag{42}
$$

The recursion is sequentially well defined: when the envelope at time t + 1 is computed, all $R _ { s } ^ { \mathrm { E B } }$ with $s \leq t$ are already available. As in the sub-Gaussian construction, simultaneous domination of the unknown distances $Z _ { s }$ by these observable distance envelopes yields observable upper bounds on both the latent quadratic process and the predictable range; monotonicity of $\mathfrak { B } _ { \alpha }$ then propagates the domination through the recursion.

Theorem 3.5 (Refined observable confidence sequence for $\| \boldsymbol { x } _ { t + 1 } - \boldsymbol { x } ^ { \star } \| ^ { 2 } )$ . Suppose Assumptions 1.1– 1.4 and 3.1 hold, and suppose (37) is satisfied. Let $R _ { 0 }$ be an $\mathcal { F } _ { t _ { 0 } }$ <sub>−1</sub>-measurable quantity satisfying $R _ { 0 } \geq Z _ { t _ { 0 } }$ almost surely. Then, for every $\alpha \in ( 0 , 1 )$ , the recursively defined process $\{ \widehat { U } _ { t + 1 } ^ { \mathrm { d i s t , E B } } ( \alpha ) \} _ { t \geq t _ { 0 } }$ is well defined and fully observable from $R _ { 0 }$ and the SGD trajectory. Moreover,

$$
\begin{array} { r } { \mathbb { P } \left( \forall t \geq t _ { 0 } : \ Z _ { t + 1 } \leq \widehat { U } _ { t + 1 } ^ { \mathrm { d i s t , E B } } ( \alpha ) \right) \geq 1 - \alpha . } \end{array}
$$

The observable construction preserves the trajectory adaptation highlighted in Remark 3.4: the quadratic process (40) depends on the realized values $\| g _ { s } \| ^ { 2 }$ , while the recursive observable distance envelopes $R _ { s } ^ { \mathrm { E B } }$ remove the remaining dependence on the unknown optimizer. The global bound G remains only in the predictable-range process (41) and, when used, in the canonical initialization $R _ { 0 } = G ^ { 2 } / \mu ^ { 2 }$ . It remains to understand the cost of accommodating the growing predictable range. The empirical Bernstein boundary introduces the additional linear range term in (32), which has no counterpart in the sub-Gaussian construction. The next proposition shows that this term is of lower order and that the full observable confidence sequence retains the near-1/t decay established in the preceding section. Define

$$
B _ { \mathrm { E B } } : = 1 + \frac { G ^ { 2 } } { \mu ^ { 2 } } ,\tag{43}
$$

and, for $t \geq t _ { 0 }$

$$
L _ { t } ^ { \mathrm { E B } } ( \alpha ) : = 1 + \log \frac { 1 } { \alpha } + \log \log ( e t ) + \log \log ( e B _ { \mathrm { E B } } ) .\tag{44}
$$

To isolate the contribution of the linear range term in (32), define

$$
\mathcal { R } _ { t } ^ { \mathrm { E B } } ( \alpha ) : = \frac { 2 \operatorname* { m a x } \left\{ \widehat { B } _ { t } ^ { \mathrm { d i s t , E B } } , 1 \right\} } { A _ { t } } \Gamma _ { \alpha } \left( \widehat { V } _ { t } ^ { \mathrm { d i s t , E B } } , \widehat { B } _ { t } ^ { \mathrm { d i s t , E B } } \right) .\tag{45}
$$

With these quantities in place, we are ready to state the proposition.

Proposition 3.6 (Decay rate of the refined observable confidence sequence for $\| \boldsymbol { x } _ { t + 1 } - \boldsymbol { x } ^ { \star } \| ^ { 2 } )$ . Fix $\alpha \in ( 0 , 1 )$ and suppose the conditions of Theorem 3.5 hold. Let $R _ { 0 } = G ^ { 2 } / \mu ^ { 2 } , \eta _ { t } = 1 / ( \mu t )$ , and $t _ { 0 } = 3$ Then there exist universal constants $C , C _ { \mathrm { r } } < \infty$ such that, almost surely,

$$
\widehat { U } _ { t + 1 } ^ { \mathrm { d i s t , E B } } ( \alpha ) \leq C B _ { \mathrm { E B } } \left[ \frac { L _ { t + 1 } ^ { \mathrm { E B } } ( \alpha ) } { t + 1 } + \left( \frac { L _ { t + 1 } ^ { \mathrm { E B } } ( \alpha ) } { t + 1 } \right) ^ { 2 } \right] , \qquad t \geq 3 ,\tag{46}
$$

and, whenever $t \geq 4$ and $L _ { t } ^ { \mathrm { E B } } ( \alpha ) \leq t .$

$$
\mathcal { R } _ { t } ^ { \mathrm { E B } } ( \alpha ) \leq C _ { \mathrm { r } } B _ { \mathrm { E B } } \left( \frac { L _ { t } ^ { \mathrm { E B } } ( \alpha ) } { t } \right) ^ { 3 / 2 } .\tag{47}
$$

Proposition 3.6 yields two useful conclusions. First, once $L _ { t + 1 } ^ { \mathrm { E B } } ( \alpha ) \leq t + 1$ , the observable confidence sequence satisfies

$$
\widehat { U } _ { t + 1 } ^ { \mathrm { d i s t , E B } } ( \alpha ) \leq 2 C B _ { \mathrm { E B } } \frac { L _ { t + 1 } ^ { \mathrm { E B } } ( \alpha ) } { t + 1 } ,
$$

thereby recovering the same near-1/t worst-case decay rate as the sub-Gaussian construction. For fixed α and fixed problem parameters, this condition holds for all suficiently large t. Second, (47) shows that the additional contribution induced by the growing predictable range is of lower order than the leading $L _ { t } ^ { \mathrm { E B } } ( \alpha ) / t$ term.

Remark 3.7 (Alternative polynomial stepsize). The same conclusion holds for the stepsize

$$
\eta _ { t } = \frac { 2 } { \mu ( t + 1 ) } .
$$

In this case, $1 - 2 \mu \eta _ { t } = ( t - 3 ) / ( t + 1 )$ , so the natural analysis start time is $t _ { 0 } = 4$ . Following similar steps to those in the proof of Proposition 3.6, there exists a universal constant $C <$ such that, almost surely, for every $t \geq 4$ ,

$$
\widehat { U } _ { t + 1 } ^ { \mathrm { d i s t , E B } } ( \alpha ) \leq C B _ { \mathrm { E B } } \left[ \frac { L _ { t + 1 } ^ { \mathrm { E B } } ( \alpha ) } { t + 1 } + \left( \frac { L _ { t + 1 } ^ { \mathrm { E B } } ( \alpha ) } { t + 1 } \right) ^ { 2 } \right] .
$$

The analogue of (47) holds as well, with a diferent universal constant. The proof is similar and is omitted.

## 3.2 Suboptimality of the weighted-average iterate

We now turn to the weighted-average suboptimality. Recall from Lemma C.3 that the suboptimality analysis reduces to controlling the partial sum of the directional gradient noise terms $E _ { t } = t \left. \boldsymbol { x } ^ { \star } - \boldsymbol { x } _ { t } , \boldsymbol { \xi } _ { t } \right.$ To bring this term within the scope of Theorem 3.3, define the corresponding uncentered directional stochastic-gradient term

$$
H _ { t } ^ { \mathrm { s u b } } : = t \langle x ^ { \star } - x _ { t } , g _ { t } \rangle , \qquad t \geq t _ { 0 } .\tag{48}
$$

By conditional unbiasedness, $H _ { t } ^ { \mathrm { s u b } } - \mathbb { E } \left[ H _ { t } ^ { \mathrm { s u b } } \Big | \mathcal { F } _ { t - 1 } \right] = E _ { t }$ . Thus, the centered process appearing in Theorem 3.3 is exactly the stochastic term already isolated by the suboptimality decomposition. Under Assumption 3.1,

$$
\left| H _ { t } ^ { \mathrm { s u b } } \right| \leq G t { \sqrt { Z _ { t } } } .
$$

As in the distance construction, the bound $Z _ { t } \lesssim L _ { t } / t$ implies that the predictable range can grow as $\sqrt { t L _ { t } }$ and need not remain bounded over the infinite horizon. Accordingly, define the latent quadratic and range processes

$$
V _ { t } ^ { \mathrm { s u b , E B } } : = \sum _ { s = t _ { 0 } } ^ { t } \left( H _ { s } ^ { \mathrm { s u b } } \right) ^ { 2 } = \sum _ { s = t _ { 0 } } ^ { t } s ^ { 2 } \langle g _ { s } , x ^ { \star } - x _ { s } \rangle ^ { 2 } ,\tag{49}
$$

$$
B _ { t } ^ { \mathrm { s u b , E B } } : = G \operatorname* { m a x } _ { t _ { 0 } \leq s \leq t } s \sqrt { Z _ { s } } .\tag{50}
$$

These are the corresponding quadratic and predictable-range processes for the suboptimality analysis. Moreover, since $Z _ { t } \le G ^ { 2 } / \mu ^ { 2 }$ almost surely, $H _ { t } ^ { \mathrm { s u b } }$ is integrable for every deterministic $t \geq t _ { 0 }$

Combining Lemma C.3 with Theorem 3.3 gives the latent empirical Bernstein confidence sequence

$$
U _ { t } ^ { \mathrm { s u b , E B } } ( \alpha ) : = \frac { 1 } { S _ { t } } \left[ \frac { 1 } { \mu } \sum _ { s = t _ { 0 } } ^ { t } \| g _ { s } \| ^ { 2 } + \frac { \mu t _ { 0 } ( t _ { 0 } - 1 ) } { 4 } Z _ { t _ { 0 } } + \mathfrak { B } _ { \alpha } \left( V _ { t } ^ { \mathrm { s u b , E B } } , B _ { t } ^ { \mathrm { s u b , E B } } \right) \right] , \qquad t \geq t _ { 0 } ,\tag{51}
$$

which satisfies

$$
\begin{array} { r } { \mathbb { P } \left( \forall t \geq t _ { 0 } : \ f ( \bar { x } _ { t } ) - f ( x ^ { \star } ) \leq U _ { t } ^ { \mathrm { s u b , E B } } ( \alpha ) \right) \geq 1 - \alpha . } \end{array}
$$

This is the empirical Bernstein counterpart of Proposition 2.7 from the preceding section. The argument follows the same overall logic, with Theorem 3.3 providing the corresponding concentration step. We defer the formal proof of this confidence-sequence guarantee to Proposition B.2 in the appendix.

As in the distance construction, the latent bound is not observable because $Z _ { t _ { 0 } }$ , the directions $x ^ { \star } - x _ { s }$ and the distances entering $B _ { t } ^ { \mathrm { s u b , E B } }$ depend on the unknown optimizer. Here this dependence can be removed directly using the observable distance confidence sequence constructed above. Let $R _ { 0 }$ be any $\mathcal { F } _ { t _ { 0 } - 1 }$ -measurable quantity satisfying $R _ { 0 } \geq Z _ { t _ { 0 } }$ almost surely. For $\beta \in ( 0 , 1 )$ , define the observable distance envelopes

$$
R _ { t _ { 0 } } ^ { \mathrm { d i s t , E B } } ( \beta ) : = R _ { 0 } , \qquad R _ { s } ^ { \mathrm { d i s t , E B } } ( \beta ) : = \widehat { U } _ { s } ^ { \mathrm { d i s t , E B } } ( \beta ) , \quad s \geq t _ { 0 } + 1 .\tag{52}
$$

For $t \geq t _ { 0 }$ , set

$$
\widehat { V } _ { t } ^ { \mathrm { s u b , E B } } ( \beta ) : = \sum _ { s = t _ { 0 } } ^ { t } s ^ { 2 } \left\| g _ { s } \right\| ^ { 2 } R _ { s } ^ { \mathrm { d i s t , E B } } ( \beta ) ,\tag{53}
$$

$$
\widehat { B } _ { t } ^ { \mathrm { s u b , E B } } ( \beta ) : = G \operatorname* { m a x } _ { t _ { 0 } \leq s \leq t } s \sqrt { R _ { s } ^ { \mathrm { d i s t , E B } } ( \beta ) } .\tag{54}
$$

Given $\alpha \in ( 0 , 1 )$ , define

$$
\widehat { U } _ { t } ^ { \mathrm { s u b , E B } } ( \alpha ) : = \frac { 1 } { S _ { t } } \left[ \frac { 1 } { \mu } \sum _ { s = t _ { 0 } } ^ { t } \| g _ { s } \| ^ { 2 } + \frac { \mu t _ { 0 } ( t _ { 0 } - 1 ) } { 4 } R _ { 0 } + \mathfrak { B } _ { \alpha / 2 } \left( \widehat { V } _ { t } ^ { \mathrm { s u b , E B } } ( \alpha / 2 ) , \widehat { B } _ { t } ^ { \mathrm { s u b , E B } } ( \alpha / 2 ) \right) \right] , \qquad t \geq t _ { 0 } .\tag{55}
$$

As in the distance construction, the observable distance envelopes dominate the unknown distances simultaneously and therefore yield observable upper bounds on both latent arguments of $\mathfrak { B } _ { \alpha / 2 }$ . The use of confidence level $\alpha / 2$ accounts for the simultaneous validity of the distance and suboptimality confidence sequences.

Theorem 3.8 (Refined observable confidence sequence for $f ( \bar { x } _ { t } ) - f ( x ^ { \star } ) )$ . Suppose Assumptions 1.1– 1.4 and 3.1 hold. Let $t _ { 0 } \geq 4$ and $\eta _ { t } = 2 / ( \mu ( t + 1 ) )$ for every $t \geq t _ { 0 }$ , and let $R _ { 0 }$ be an $\mathcal { F } _ { t _ { 0 } - 1 }$ -measurable quantity satisfying $R _ { 0 } \geq Z _ { t _ { 0 } }$ almost surely. Then, for every $\alpha \in ( 0 , 1 )$ , the process $\{ \widehat { U } _ { t } ^ { \mathrm { s u b , E B } } ( \alpha ) \} _ { t \geq t _ { 0 } }$ is well defined and fully observable from $R _ { 0 }$ and the SGD trajectory. Moreover,

$$
\begin{array} { r } { \mathbb { P } \left( \forall t \geq t _ { 0 } : \ f ( \bar { x } _ { t } ) - f ( x ^ { \star } ) \leq \widehat { U } _ { t } ^ { \mathrm { s u b , E B } } ( \alpha ) \right) \geq 1 - \alpha . } \end{array}
$$

The observable construction preserves the same trajectory adaptation: the quadratic process (53) depends on the realized values $\| g _ { s } \| ^ { 2 }$ , while the observable distance envelopes remove the remaining dependence on the unknown optimizer. As in the distance analysis, it remains to quantify the cost of the growing predictable range. Recall $B _ { \mathrm { E B } }$ from (43), and define

$$
B _ { \mathrm { s u b } } ^ { \mathrm { E B } } : = 1 + \mu B _ { \mathrm { E B } } = 1 + \mu + \frac { G ^ { 2 } } { \mu } ,\tag{56}
$$

and, for $t \geq t _ { 0 }$

$$
L _ { t } ^ { \mathrm { s u b , E B } } ( \alpha ) : = 1 + \log \frac { 2 } { \alpha } + \log \log ( e t ) + \log \log ( e B _ { \mathrm { E B } } ) + \log \log \left( e B _ { \mathrm { s u b } } ^ { \mathrm { E B } } \right) .\tag{57}
$$

To isolate the contribution of the linear range term in (32), define

$$
\mathcal { R } _ { t } ^ { \mathrm { s u b , E B } } ( \alpha ) : = \frac { 2 \operatorname* { m a x } \left\{ \widehat { B } _ { t } ^ { \mathrm { s u b , E B } } ( \alpha / 2 ) , 1 \right\} } { S _ { t } } \Gamma _ { \alpha / 2 } \left( \widehat { V } _ { t } ^ { \mathrm { s u b , E B } } ( \alpha / 2 ) , \widehat { B } _ { t } ^ { \mathrm { s u b , E B } } ( \alpha / 2 ) \right) .\tag{58}
$$

With these quantities in place, we are ready to state the proposition.

Proposition 3.9 (Decay rate of the refined observable confidence sequence for $f ( { \bar { x } } _ { t } ) - f ( x ^ { \star } ) )$ . Fix $\alpha \in ( 0 , 1 )$ and suppose Assumptions 1.1–1.4 and 3.1 hold. Let $R _ { 0 } = G ^ { 2 } / \mu ^ { 2 } , t _ { 0 } = 4$ , and $\eta _ { t } = 2 / ( \mu ( t { + } 1 ) )$ 0 for every $t \geq 4$ . Then there exist universal constants $C , C _ { \mathrm { r } } < \infty$ such that, almost surely,

$$
\widehat { U } _ { t } ^ { \mathrm { s u b , E B } } ( \alpha ) \leq C B _ { \mathrm { s u b } } ^ { \mathrm { E B } } \left[ \frac { L _ { t } ^ { \mathrm { s u b , E B } } ( \alpha ) } { t } + \left( \frac { L _ { t } ^ { \mathrm { s u b , E B } } ( \alpha ) } { t } \right) ^ { 2 } \right] , \qquad t \geq 4 ,\tag{59}
$$

and, whenever $L _ { t } ^ { \mathrm { s u b , E B } } ( \alpha ) \leq t$

$$
\mathcal { R } _ { t } ^ { \mathrm { s u b , E B } } ( \alpha ) \leq C _ { \mathrm { r } } B _ { \mathrm { s u b } } ^ { \mathrm { E B } } \left( \frac { L _ { t } ^ { \mathrm { s u b , E B } } ( \alpha ) } { t } \right) ^ { 3 / 2 } .\tag{60}
$$

Proposition 3.9 yields the same two conclusions as in the distance analysis. First, once $L _ { t } ^ { \mathrm { s u b , E B } } ( \alpha ) \leq$ t, the observable confidence sequence satisfies

$$
\widehat { U } _ { t } ^ { \mathrm { s u b , E B } } ( \alpha ) \leq 2 C B _ { \mathrm { s u b } } ^ { \mathrm { E B } } \frac { L _ { t } ^ { \mathrm { s u b , E B } } ( \alpha ) } { t } ,
$$

thereby recovering the same near-1/t worst-case decay rate as the sub-Gaussian construction. For fixed α and fixed problem parameters, this condition holds for all suficiently large t. Second, (60) shows that the additional contribution induced by the growing predictable range is of lower order than the leading $L _ { t } ^ { \mathrm { s u b , E B } } ( \alpha ) / t$ term.

## 4 Minibatch Extensions

The confidence sequences developed in the preceding two sections admit natural extensions to minibatch SGD. Fix an integer $b \geq 1$ . At every iteration $t \geq 1$ , conditionally on $\mathcal { F } _ { t - 1 }$ , suppose that we observe independent and identically distributed stochastic gradients $g _ { t } ^ { ( 1 ) } , \ldots , g _ { t } ^ { ( b ) }$ satisfying

$$
\mathbb E \left[ \boldsymbol g _ { t } ^ { ( i ) } \left| \mathcal F _ { t - 1 } \right. \right] = \nabla f ( \boldsymbol x _ { t } ) , \qquad i = 1 , \dots , b .\tag{61}
$$

Let $\mathcal { F } _ { t } : = \sigma \left( \mathcal { F } _ { t - 1 } , g _ { t } ^ { ( 1 ) } , \ldots , g _ { t } ^ { ( b ) } \right)$ denote the filtration generated by the history through iteration t. Define the minibatch gradient and its noise by

$$
\bar { g } _ { t } : = \frac { 1 } { b } \sum _ { i = 1 } ^ { b } g _ { t } ^ { ( i ) } , \qquad \bar { \xi } _ { t } : = \bar { g } _ { t } - \nabla f ( x _ { t } ) ,\tag{62}
$$

and let the SGD update use $\bar { g } _ { t }$ in place of $g _ { t }$

For the sub-Gaussian confidence sequences of Section 2, the efect of minibatching is immediate. Suppose, in addition, that the individual noises $\xi _ { t } ^ { ( i ) } : = g _ { t } ^ { ( i ) } - \nabla f ( x _ { t } )$ satisfy the conditional sub-Gaussian condition in Assumption 1.3 with the same variance proxy $\sigma ^ { 2 }$ . Conditional independence then implies that their average $\bar { \xi } _ { t }$ satisfies the same conditional sub-Gaussian bound with variance proxy $\sigma ^ { 2 } / b$ Consequently, the confidence sequences of Section 2 extend directly to minibatch SGD by replacing $\sigma ^ { 2 }$ with $\sigma ^ { 2 } / b$ in the corresponding variance processes and replacing $g _ { t }$ with $\bar { g } _ { t }$ in the SGD recursions.

The empirical Bernstein construction of Section 3 allows the information contained in the minibatch to be used more finely. The sub-Gaussian construction summarizes the stochastic fluctuations through the fixed conditional variance proxy $\sigma ^ { 2 } / b$ . By contrast, the empirical Bernstein argument can be applied to the b individual observations before they are averaged. Conditional independence allows the corresponding one-step exponential bounds to be combined while retaining the sum of the individual realized quadratic contributions. Thus, the leading square-root term enjoys the usual $1 / \sqrt { b }$ minibatch reduction while continuing to adapt to the realized directional second moments within each minibatch, whereas the linear range term carries an explicit factor $1 / b$

This observation yields a general minibatch extension of Theorem 3.3, in which the centered process is formed from the minibatch averages while the realized quadratic process retains the individual observations within each minibatch. The resulting form will apply directly to both the distance and suboptimality confidence sequences.

Proposition 4.1 (Minibatch empirical Bernstein bound with a growing predictable range). Fix deterministic integers $t _ { 0 } \geq 1$ and $b \geq 1$ , and let $\{ \mathcal { F } _ { t } \} _ { t \ge t _ { 0 } - 1 }$ be a filtration. For every $t \geq t _ { 0 } .$ , let $H _ { t } ^ { ( 1 ) } , \dots , H _ { t } ^ { ( b ) }$ be integrable, <sub>t</sub>-measurable random variables that are conditionally independent given $\mathcal { F } _ { t - 1 }$ and have a common conditional mean $m _ { t } : = \mathbb { E } \left[ H _ { t } ^ { ( i ) } \mid { \mathcal { F } } _ { t - 1 } \right]$ , for $i = 1 , \ldots , b$ . Let $\{ c _ { t } \} _ { t \ge t _ { 0 } }$ be an almost surely finite,<sup>7</sup> nonnegative predictable process such that $\left| H _ { t } ^ { ( i ) } \right| \le c _ { t }$ almost surely for every $t \ \geq \ t _ { 0 }$ and $i = 1 , \dots , b$ . Define

$$
\overline { { H } } _ { t _ { 0 } - 1 } ^ { \mathrm { M B } } : = 0 , \qquad \overline { { H } } _ { t } ^ { \mathrm { M B } } : = \sum _ { s = t _ { 0 } } ^ { t } \left( \frac { 1 } { b } \sum _ { i = 1 } ^ { b } H _ { s } ^ { ( i ) } - m _ { s } \right) ,
$$

and

$$
W _ { t _ { 0 } - 1 } ^ { \mathrm { M B } } : = 0 , \qquad W _ { t } ^ { \mathrm { M B } } : = \sum _ { s = t _ { 0 } } ^ { t } \sum _ { i = 1 } ^ { b } \left( H _ { s } ^ { ( i ) } \right) ^ { 2 } , \qquad C _ { t } : = \operatorname* { m a x } _ { t _ { 0 } \le s \le t } c _ { s } .
$$

Then, for every $\alpha \in ( 0 , 1 )$ ,

$$
\mathbb { P } \left( \forall t \geq t _ { 0 } : \ \overline { { H } } _ { t } ^ { \mathrm { M B } } \leq \frac { 1 } { b } \mathfrak { B } _ { \alpha } \left( W _ { t } ^ { \mathrm { M B } } , C _ { t } \right) \right) \geq 1 - \alpha .\tag{63}
$$

For $b = 1$ , Proposition 4.1 reduces exactly to Theorem 3.3. We next specialize Proposition 4.1 to the SGD setting. For this purpose, we impose the minibatch analogue of Assumption 3.1.

Assumption 4.2 (Bounded minibatch stochastic gradients). There exists $G < \infty$ such that

$$
\left\| g _ { t } ^ { ( i ) } \right\| \leq G \qquad { \mathrm { a l m o s t ~ s u r e l y ~ f o r ~ a l l ~ } } t \geq t _ { 0 } { \mathrm { ~ a n d ~ } } i = 1 , \ldots , b .
$$

We now specialize Proposition 4.1 to the SGD confidence sequences, beginning with the last-iterate distance. For every $t \geq t _ { 0 }$ , define the observable within-minibatch second-moment matrix

$$
\widehat { \Sigma } _ { t } : = \frac { 1 } { b } \sum _ { i = 1 } ^ { b } g _ { t } ^ { ( i ) } { ( g _ { t } ^ { ( i ) } ) } ^ { \top } .\tag{64}
$$

As in the single-gradient construction of Section 3, set

$$
H _ { s } ^ { ( i ) , \mathrm { d i s t } } : = 2 A _ { s } \eta _ { s } \langle x ^ { \star } - x _ { s } , g _ { s } ^ { ( i ) } \rangle , \qquad i = 1 , \ldots , b .
$$

Their averaged centered increment is exactly the stochastic term arising from the distance decomposition with $\bar { g } _ { s }$ in place of $g _ { s }$ . Moreover,

$$
\sum _ { i = 1 } ^ { b } \Big ( H _ { s } ^ { ( i ) , \mathrm { d i s t } } \Big ) ^ { 2 } = 4 b A _ { s } ^ { 2 } \eta _ { s } ^ { 2 } ( x _ { s } - x ^ { \star } ) ^ { \top } \widehat \Sigma _ { s } ( x _ { s } - x ^ { \star } ) ,
$$

while Assumption 4.2 gives

$$
\left| H _ { s } ^ { ( i ) , \mathrm { d i s t } } \right| \leq 2 G A _ { s } \eta _ { s } { \sqrt { Z _ { s } } } .
$$

Thus, the minibatch quadratic process retains the realized directional second moment encoded by ${ \widehat \Sigma } _ { s }$ rather than reducing the stochastic fluctuations to a fixed variance proxy.

To make the resulting bound fully observable, we use the same recursive domination argument as in the distance construction of Section 3. Since

$$
( x _ { s } - x ^ { \star } ) ^ { \top } \widehat { \Sigma } _ { s } ( x _ { s } - x ^ { \star } ) \leq \lambda _ { \operatorname* { m a x } } ( \widehat { \Sigma } _ { s } ) Z _ { s } ,\tag{65}
$$

the unknown distance can again be replaced by a previously constructed distance envelope. Fix α $( 0 , 1 )$ , and let $R _ { 0 }$ be any $\mathcal { F } _ { t _ { 0 } - 1 }$ <sub>1</sub>-measurable quantity satisfying $R _ { 0 } \geq Z _ { t _ { 0 } }$ almost surely. Define recursively

$$
R _ { t _ { 0 } } ^ { \mathrm { E B , M B } } : = R _ { 0 } , \qquad R _ { s } ^ { \mathrm { E B , M B } } : = \widehat { U } _ { s } ^ { \mathrm { d i s t , E B , M B } } ( \alpha ) , \quad s \geq t _ { 0 } + 1 .\tag{66}
$$

For every $t \geq t _ { 0 }$ , set

$$
\widehat { V } _ { t } ^ { \mathrm { d i s t , E B , M B } } : = 4 b \sum _ { s = t _ { 0 } } ^ { t } A _ { s } ^ { 2 } \eta _ { s } ^ { 2 } \lambda _ { \mathrm { m a x } } ( \widehat { \Sigma } _ { s } ) R _ { s } ^ { \mathrm { E B , M B } } ,\tag{67}
$$

$$
\widehat { B } _ { t } ^ { \mathrm { d i s t , E B , M B } } : = 2 G \operatorname* { m a x } _ { t _ { 0 } \leq s \leq t } A _ { s } \eta _ { s } \sqrt { R _ { s } ^ { \mathrm { E B , M B } } } ,\tag{68}
$$

and define

$$
\widehat { U } _ { t + 1 } ^ { \mathrm { d i s t , E B , M B } } ( \alpha ) : = \frac { 1 } { A _ { t } } \left[ R _ { 0 } + \sum _ { s = t _ { 0 } } ^ { t } A _ { s } \eta _ { s } ^ { 2 } \left. \bar { g } _ { s } \right. ^ { 2 } + \frac { 1 } { b } \mathfrak { B } _ { \alpha } \left( \widehat { V } _ { t } ^ { \mathrm { d i s t , E B , M B } } , \widehat { B } _ { t } ^ { \mathrm { d i s t , E B , M B } } \right) \right] .\tag{69}
$$

As in the single-gradient case, the construction is recursive: $R _ { s } ^ { \mathrm { E B , M B } }$ is available before the minibatch at iteration s is observed, and the quantities above determine the distance envelope for the next iterate. The following corollary is the minibatch analogue of Theorem 3.5.

Corollary 4.3 (Refined observable minibatch confidence sequence for $\| \boldsymbol { x } _ { t + 1 } - \boldsymbol { x } ^ { \star } \| ^ { 2 } )$ . Suppose Assumptions 1.1–1.4 and 4.2 hold, and suppose (37) is satisfied. Let $R _ { 0 }$ be an $\mathcal { F } _ { t _ { 0 } - 1 }$ -measurable quantity satisfying $R _ { 0 } \ \geq \ Z _ { t _ { 0 } }$ almost surely. Then, for every $\alpha \in ( 0 , 1 )$ , the recursively defined process $\{ \widehat { U } _ { t + 1 } ^ { \mathrm { d i s t , E B , M B } } ( \alpha ) \} _ { t \geq t _ { 0 } }$ is well defined and fully observable from $R _ { 0 }$ and the minibatch SGD trajectory. Moreover,

$$
\mathbb { P } \left( \forall t \geq t _ { 0 } : \ Z _ { t + 1 } \leq \widehat { U } _ { t + 1 } ^ { \mathrm { d i s t , E B , M B } } ( \alpha ) \right) \geq 1 - \alpha .
$$

The spectral relaxation (65) makes the directional quadratic process observable while retaining information about the geometry of the realized minibatch. Indeed, the latent quantity depends on the unknown direction $x _ { s } - x ^ { \star }$ , whereas $\lambda _ { \operatorname* { m a x } } ( \widehat { \Sigma } _ { s } )$ provides a direction-free upper bound determined entirely by the observed stochastic gradients. Moreover,

$$
\lambda _ { \operatorname* { m a x } } ( \widehat { \Sigma } _ { s } ) \leq \mathrm { t r } ( \widehat { \Sigma } _ { s } ) = \frac { 1 } { b } \sum _ { i = 1 } ^ { b } \left\| g _ { s } ^ { ( i ) } \right\| ^ { 2 } \leq G ^ { 2 } ,
$$

so the resulting bound can be substantially sharper than replacing the quadratic process by the uniform worst-case factor $G ^ { 2 }$ , particularly when the realized gradients are small or their energy is spread across several directions. This refinement also remains computationally tractable: since $\widehat { \Sigma } _ { s }$ has rank at most $b ,$ its largest eigenvalue can be computed from a $b \times b$ Gram matrix, without forming or diagonalizing a full d d matrix.

We next turn to weighted-average suboptimality. As in the single-gradient construction of Section 3, the stochastic term in the weighted suboptimality decomposition is controlled through the directional observations

$$
H _ { s } ^ { ( i ) , \mathrm { s u b } } : = s \langle x ^ { \star } - x _ { s } , g _ { s } ^ { ( i ) } \rangle , \qquad i = 1 , \dots , b .
$$

Their averaged centered increment is $s \langle x ^ { \star } - x _ { s } , \xi _ { s } \rangle$ , and

$$
\sum _ { i = 1 } ^ { b } \Big ( H _ { s } ^ { ( i ) , \mathrm { s u b } } \Big ) ^ { 2 } = b s ^ { 2 } \big ( x _ { s } - x ^ { \star } \big ) ^ { \top } \widehat \Sigma _ { s } ( x _ { s } - x ^ { \star } ) ,
$$

while Assumption 4.2 gives

$$
\left| H _ { s } ^ { ( i ) , \mathrm { s u b } } \right| \leq G s { \sqrt { Z _ { s } } } .
$$

Thus, as in the distance construction, the latent quadratic and range processes can be made observable using the same spectral relaxation together with the previously constructed minibatch distance envelopes. For a confidence level $\beta \in ( 0 , 1 )$ ), define

$$
R _ { t _ { 0 } } ^ { \mathrm { d i s t , E B , M B } } ( \beta ) : = R _ { 0 } , \qquad R _ { s } ^ { \mathrm { d i s t , E B , M B } } ( \beta ) : = \widehat { U } _ { s } ^ { \mathrm { d i s t , E B , M B } } ( \beta ) , \quad s \geq t _ { 0 } + 1 .\tag{70}
$$

For every $t \geq t _ { 0 }$ , set

$$
\widehat { V } _ { t } ^ { \mathrm { s u b , E B , M B } } ( \beta ) : = b \sum _ { s = t _ { 0 } } ^ { t } s ^ { 2 } \lambda _ { \mathrm { m a x } } ( \widehat { \Sigma } _ { s } ) R _ { s } ^ { \mathrm { d i s t , E B , M B } } ( \beta ) ,\tag{71}
$$

$$
\widehat { B } _ { t } ^ { \mathrm { s u b , E B , M B } } ( \beta ) : = G \operatorname* { m a x } _ { t _ { 0 } \leq s \leq t } s \sqrt { R _ { s } ^ { \mathrm { d i s t , E B , M B } } ( \beta ) } .\tag{72}
$$

Given $\alpha \in ( 0 , 1 )$ , define

$$
\begin{array} { r l } & { \widehat { U } _ { t } ^ { \mathrm { s u b , E B , M B } } ( \alpha ) : = \displaystyle \frac { 1 } { S _ { t } } \Bigg [ \frac { 1 } { \mu } \displaystyle \sum _ { s = t _ { 0 } } ^ { t } \| \bar { g } _ { s } \| ^ { 2 } + \frac { \mu t _ { 0 } ( t _ { 0 } - 1 ) } { 4 } R _ { 0 } } \\ & { \qquad + \displaystyle \frac { 1 } { b } \mathfrak { B } _ { \alpha / 2 } \left( \widehat { V } _ { t } ^ { \mathrm { s u b , E B , M B } } ( \alpha / 2 ) , \widehat { B } _ { t } ^ { \mathrm { s u b , E B , M B } } ( \alpha / 2 ) \right) \Bigg ] , \qquad t \geq t _ { 0 } . } \end{array}\tag{73}
$$

The following corollary is the minibatch analogue of Theorem 3.8.

Corollary 4.4 (Refined observable minibatch confidence sequence for $f ( { \bar { x } } _ { t } ) - f ( x ^ { \star } ) )$ ). Suppose $\mathrm { A s } _ { - }$ sumptions 1.1–1.4 and 4.2 hold. Let $t _ { 0 } \geq 4$ and $\eta _ { t } = 2 / ( \mu ( t + 1 ) )$ for every $t \geq t _ { 0 }$ , and let $R _ { 0 }$ be an $\mathcal { F } _ { t _ { 0 } }$ <sub>−1</sub>-measurable quantity satisfying $R _ { 0 } \geq Z _ { t _ { 0 } }$ almost surely. Then, for every $\alpha \in ( 0 , 1 )$ , the process $\{ \widehat { U } _ { t } ^ { \mathrm { s u b , E B , M B } } ( \alpha ) \} _ { t \geq t _ { 0 } }$ is well defined and fully observable from $R _ { 0 }$ and the minibatch SGD trajectory. Moreover,

$$
\begin{array} { r } { \mathbb { P } \left( \forall t \geq t _ { 0 } : \ f ( \bar { x } _ { t } ) - f ( x ^ { \star } ) \leq \widehat { U } _ { t } ^ { \mathrm { s u b , E B , M B } } ( \alpha ) \right) \geq 1 - \alpha . } \end{array}
$$

When $b = 1$ , we have $\widehat { \Sigma } _ { t } = g _ { t } g _ { t } ^ { \top }$ and $\lambda _ { \operatorname* { m a x } } ( \widehat { \Sigma } _ { t } ) = \| g _ { t } \| ^ { 2 }$ , so the minibatch distance and suboptimality constructions reduce exactly to their single-gradient empirical Bernstein counterparts in Section 3. For general $b ,$ the factor b in the observable quadratic processes combines with the prefactor $1 / b$ outside the empirical Bernstein boundary to yield the usual $1 / \sqrt { b }$ reduction in the leading square-root term, up to the iterated-logarithmic factors from stitching, while the linear range term carries the explicit factor $1 / b .$

## 5 Experiments

To study the empirical performance of our confidence sequences, we run a series of experiments on a primal support vector machine (SVM) problem, using both real-world and synthetic data. Section 5.1 details the optimization problem, the experimental protocol, and the bounds we compare against. Section 5.2 then discusses the introductory figure (Figure 1), which illustrates the improvement in certified stopping on a single instance. Section 5.3 (Figure 2) varies the minibatch size $b ,$ exhibiting its efect on the relative behavior of the bounds. Section 5.4 (Figure 3) compares three synthetic datasets of increasing dificulty while keeping all constants entering the bounds fixed, thereby isolating trajectory adaptivity. We conclude with two verifications. Section 5.5 (Figures 4 and 5) checks the sensitivity of our bounds to conservatively chosen constants, and Section 5.6 (Figure 6) examines whether the same behavior persists in a single-pass regime beyond the finite-dataset formulation used in the preceding experiments.

## 5.1 Experimental setup

We consider the SVM primal optimization problem

$$
f ( x ) : = \frac { \mu } { 2 } \| x \| ^ { 2 } + \mathbb { E } _ { z } \big [ \operatorname* { m a x } \{ 0 , 1 - z ^ { \top } x \} \big ] ,\tag{74}
$$

where $z$ follows a probability distribution $\mathbb { Q }$ on $\mathbb { R } ^ { d }$ and satisfies $\| z \| \leq M$ almost surely. This problem is µ-strongly convex and nonsmooth. For a minibatch size $b \geq 1$ , at each time $t \geq 1$ we sample $z _ { t } ^ { ( 1 ) } , \ldots , z _ { t } ^ { ( b ) }$ independently from $\mathbb { Q }$ , conditionally on $\mathcal { F } _ { t - 1 }$ , and define

$$
g _ { t } ^ { ( i ) } = \mu x _ { t } - z _ { t } ^ { ( i ) } \mathbf { 1 } \{ 1 - ( z _ { t } ^ { ( i ) } ) ^ { \top } x _ { t } \geq 0 \} , \qquad \bar { g } _ { t } = \frac { 1 } { b } \sum _ { i = 1 } ^ { b } g _ { t } ^ { ( i ) } , \qquad i = 1 , \ldots , b ,
$$

where $\bar { g } _ { t }$ is used in the SGD update. For $b = 1$ , this reduces to the single-gradient setting; in that case, we simply write $g _ { t }$ . Since (74) is not diferentiable in general, each $g _ { t } ^ { ( i ) }$ is an unbiased stochastic subgradient rather than a stochastic gradient. Although we assume diferentiability throughout the paper for notational convenience, the same arguments apply with unbiased stochastic subgradients.

For the explicit constants, first notice that

$$
\frac { \mu } { 2 } \| x ^ { \star } \| ^ { 2 } \leq f ( x ^ { \star } ) \leq f ( 0 ) = 1 .
$$

We therefore take $\mathcal { X } = \{ x \in \mathbb { R } ^ { d } : \| x \| \leq \sqrt { 2 / \mu } \}$ , which contains $x ^ { \star }$ and hence has the same minimizer as the unconstrained problem (74). On $x ,$

$$
\| g _ { t } ^ { ( i ) } \| \leq \sqrt { 2 \mu } + M = : G \qquad \mathrm { a l m o s t ~ s u r e l y } , \qquad i = 1 , \ldots , b .
$$

Thus the bounded-gradient conditions required by the empirical-Bernstein-based confidence sequences hold. In particular, the conditional Hoefding lemma gives the valid minibatch sub-Gaussian proxy $\sigma ^ { 2 } = G ^ { 2 } / b$ for $\bar { g } _ { t }$ . Finally, we use $R _ { 0 } = 8 / \mu$ , the squared diameter of $x ,$ as a deterministic bound on $Z _ { t _ { 0 } }$

Data and protocol. We run experiments on the covertype.binary dataset, obtained from the LIBSVM data websitem<sup>8</sup> and on synthetic datasets generated with scikit-learn’s make\_classification.<sup>9</sup> A bias term is appended to every dataset, and dense features are standardized to zero mean and unit variance.

To account for the fact that we work with finite datasets, we take $\mathbb { Q }$ to be the uniform distribution over the dataset, rather than the unknown data-generating distribution. Sampling independently from $\mathbb { Q } .$ , i.e., with replacement, makes the empirical risk problem a special case of our general setting. To confirm that the performance of our confidence sequences is not an artifact of this special case, we also conduct a true single-pass experiment in which SGD uses each data point exactly once, directly targeting the expected risk under the data-generating distribution. The full details and results of this experiment are discussed in Section 5.6.

In all experiments we set $\mu = 0 . 1$ , choose the confidence level $1 - \alpha = 9 9 \%$ , use $t _ { 0 } = 4$ , and take the stepsize schedule $\eta _ { t } = 2 / ( \mu ( t + 1 ) )$ ). For the finite-dataset experiments, the ground-truth quantities $x ^ { \star }$ and $f ( x ^ { \star } )$ are computed beforehand with LIBLINEAR [14]. The resulting numerical error is several orders of magnitude below the smallest values appearing in our plots, so these quantities can be treated as exact.

Each figure reports a single trajectory of SGD. This is deliberate: our goal is to illustrate how the confidence sequences and the resulting stopping certificates evolve along an individual realized trajectory, rather than to average away their trajectory dependence across runs. Repeating the experiments with several seeds yields essentially identical qualitative behavior, and we therefore report a single run. Similarly, the horizon $T _ { \mathrm { m a x } }$ at which our plots stop is arbitrary: all the confidence sequences we display remain valid beyond it and do not depend on it.

Compared bounds. For both the squared distance to the optimum $Z _ { t + 1 } = \| x _ { t + 1 } - x ^ { \star } \| ^ { 2 }$ and the suboptimality $f ( \bar { x } _ { t } ) - f ( x ^ { \star } )$ of the weighted average, we compare the quantities listed in Table 1. The two worst-case baselines are obtained from our Hoefding-based confidence sequences by replacing the realized squared-gradient terms by their a priori upper bound $G ^ { 2 }$ and propagating this replacement through the recursive distance envelopes. With $\sigma ^ { 2 } = G ^ { 2 }$ denoting the individual-gradient sub-Gaussian proxy, so that $\sigma ^ { 2 } / b$ is the corresponding minibatch proxy, this yields

$$
\breve { U } _ { t _ { 0 } } ^ { \mathrm { d i s t } } ( \alpha ) : = R _ { 0 } , \qquad \breve { U } _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha ) : = \frac { 1 } { A _ { t } } \Biggl ( R _ { 0 } + G ^ { 2 } \sum _ { s = t _ { 0 } } ^ { t } A _ { s } \eta _ { s } ^ { 2 } + 4 \mathfrak { H } _ { \alpha } \Biggl ( \frac { \sigma ^ { 2 } } { b } \sum _ { s = t _ { 0 } } ^ { t } A _ { s } ^ { 2 } \eta _ { s } ^ { 2 } \breve { U } _ { s } ^ { \mathrm { d i s t } } ( \alpha ) \Biggr ) \Biggr )\tag{75}
$$

<table><tr><td>Label</td><td></td><td>Quantity Anytime-valid</td><td>Trajectory-adaptive</td><td>Reference</td></tr><tr><td rowspan="6">H  $Z _ { t }$  EB</td><td>Ground truth</td><td> $\| x _ { t } - x ^ { \star } \| ^ { 2 }$ </td><td></td><td></td></tr><tr><td> $\widehat { U } _ { t } ^ { \mathrm { d i s t } } ( \alpha )$ </td><td></td><td>yes</td><td>Theorem 2.3 10</td></tr><tr><td></td><td> $\widehat { U } _ { t } ^ { \mathrm { d i s t , E B , M B } } ( \alpha )$ </td><td></td><td>Corollary 4.3</td></tr><tr><td>worst-case</td><td> $\check { U } _ { t } ^ { \mathrm { d i s t } } ( \alpha )$ </td><td></td><td>(75)</td></tr><tr><td>RSS12</td><td></td><td></td><td>[36]</td></tr><tr><td>Ground truth</td><td> $f ( { \bar { x } } _ { t } ) - f ( x ^ { \star } )$ </td><td></td><td></td></tr><tr><td rowspan="4">H  $\bar { F } _ { t }$  EB</td><td></td><td></td><td></td><td>Theorem 2.8</td></tr><tr><td></td><td> $\widehat { U } _ { t } ^ { \mathrm { s u b } } ( \alpha )$   $\widehat { U } _ { t } ^ { \mathrm { s u b , E B , M B } } ( \alpha )$ </td><td></td><td></td></tr><tr><td>worst-case</td><td></td><td>yes</td><td>yes</td></tr><tr><td></td><td> $\check { U } _ { t } ^ { \mathrm { s u b } } ( \alpha )$ </td><td>yes</td><td>no</td></tr></table>

Table 1: Quantities compared in the experiments. All bounds are plotted at confidence level $1 - \alpha = 0 . 9 9$ and use the same trajectory.

for squared distance, and

$$
\breve { U } _ { t } ^ { \mathrm { s u b } } ( \alpha ) : = \frac { 1 } { S _ { t } } \left( \frac { t - t _ { 0 } + 1 } { \mu } G ^ { 2 } + \frac { \mu t _ { 0 } ( t _ { 0 } - 1 ) } { 4 } R _ { 0 } + 2 \mathfrak { H } _ { \alpha / 2 } \left( \frac { \sigma ^ { 2 } } { b } \sum _ { s = t _ { 0 } } ^ { t } s ^ { 2 } \breve { U } _ { s } ^ { \mathrm { d i s t } } \left( \frac { \alpha } { 2 } \right) \right) \right)\tag{76}
$$

for suboptimality. Under the deterministic stepsize schedule used in our experiments, these sequences are deterministic and trajectory-independent. They are also well defined recursively: the distance sequence is initialized at $\check { U } _ { t _ { 0 } } ^ { \mathrm { d i s t } } ( \alpha ) ~ = ~ R _ { 0 }$ , and the suboptimality sequence uses the already defined distance sequence at level $\alpha / 2$ . Moreover, since $\| \bar { g } _ { t } \| ^ { 2 } \leq G ^ { 2 }$ almost surely and ${ \mathfrak { H } } _ { \alpha }$ is nondecreasing, the same recursive domination argument used for the Hoefding-based confidence sequences shows that the worst-case baselines remain valid confidence sequences. Comparing H with the corresponding worstcase baseline therefore isolates the efect of adapting the Hoefding-based construction to the realized trajectory.

For the distance to the optimum, we additionally compare against the high-probability bound of Rakhlin et al. [36, Proposition 1], which states that, for any fixed horizon $T \geq 4$ , with probability at least $1 - \alpha$ , simultaneously for all $t \leq T$

$$
Z _ { t } \le \frac { ( 6 2 4 \log ( \log ( T ) / \alpha ) + 1 ) G ^ { 2 } } { \mu ^ { 2 } t } .\tag{77}
$$

This bound stems from the same standard one-step recursion underlying our distance analysis, but difers from $\check { U } ^ { \mathrm { d i s t } }$ in several respects. First, (77) is stated in closed form, at the price of a multiplicative constant that is not necessarily optimized, whereas $\check { U } ^ { \mathrm { d i s t } }$ is defined recursively. Second, (77) does not account for the variance reduction due to minibatching, whereas $\check { U } ^ { \mathrm { d i s t } }$ uses the minibatch variance proxy $\sigma ^ { 2 } / b$ . The efect of this diference will be illustrated in Section 5.3 by varying b. Third, (77) is a finite-horizon simultaneous guarantee rather than a confidence sequence in our sense, since the horizon $T$ must be fixed in advance. We take $T = T _ { \mathrm { m a x } }$ , the most favorable valid choice for covering the entire plotted horizon. More recently, Pham et al. [33] obtained an infinite-horizon time-uniform analogue that removes the need to specify T, but with a larger leading constant (1008 instead of 624). Since our goal is to compare against a favorable explicit literature benchmark, we therefore use (77) in the experiments.

The bound (77) is stated for the stepsize schedule $\eta _ { t } = 1 / ( \mu t )$ , while our experiments use $\eta _ { t } =$ $2 / ( \mu ( t { + } 1 ) )$ ). Both belong to the standard $1 / t$ stepsize regime for strongly convex stochastic optimization, and we use (77) as the corresponding explicit literature benchmark. Overall, the two baselines serve diferent purposes: RSS12 provides an explicit finite-horizon literature benchmark, while $\check { U } ^ { \mathrm { d i s t } }$ is the natural non-adaptive counterpart of our Hoefding-based confidence sequence under the same stepsize schedule and minibatch variance proxy.

For the suboptimality of the weighted average, we are not aware of a high-probability bound in the literature stated with explicit constants. Harvey et al. [21] establish such a bound for the same linearly weighted averaging scheme and stepsize schedule, but only up to unspecified absolute constants, which makes a direct quantitative comparison impossible without further analysis. We therefore compare only against the worst-case baseline $\check { U } _ { t } ^ { \mathrm { s u b } }$ above.

## 5.2 Introductory plot

We begin with an experiment illustrating our main findings. We run SGD on covertype.binary with minibatch size $b = 2 5 6$ . Figure 1 reports, on log-log axes, the squared distance to the optimum (left) and the suboptimality of the weighted average (right), with the bounds of Table 1 evaluated along the same trajectory. On the right panel, we mark for each bound the first time it drops below the threshold $\varepsilon = 1 0 ^ { - 3 }$ . Stopping SGD at this time certifies, with probability $1 - \alpha$ , that the returned weighted average satisfies $f ( { \bar { x } } _ { t } ) - f ( x ^ { \star } ) \leq \varepsilon$

Note first that $\check { U } ^ { \mathrm { d i s t } }$ lies several orders of magnitude below RSS12. As explained in Section 5.1, this reflects the larger explicit constants in RSS12 together with the fact that it does not account for the variance reduction due to minibatching. The role of minibatching is examined separately in Section 5.3.

Turning to the confidence sequences, the Hoefding-based bound $\widehat { U } ^ { \mathrm { s u b } }$ (H) certifies the threshold 6.1 times earlier than the worst-case baseline, while the empirical Bernstein minibatch bound $\widehat { U } ^ { \mathrm { s u b , E B , M B } }$ (EB) does so 312 times earlier. In terms of gradient lookups, this corresponds to $5 . 1 \times 1 0 ^ { 6 }$ lookups for EB, $2 . 6 \times 1 0 ^ { 8 }$ for H, and $1 . 6 \times 1 0 ^ { 9 }$ for the worst-case baseline, all for the same confidence level and target accuracy.

These improvements illustrate the adaptivity of our confidence sequences, which make use of observed quantities along the trajectory rather than their a priori upper bounds. The further improvement of the empirical Bernstein sequence over the Hoefding-based one reflects its ability to exploit the realized second-moment structure within each minibatch, rather than relying only on the fixed sub-Gaussian variance proxy. Overall, the experiment shows that trajectory adaptation can substantially tighten the resulting certificates while preserving their anytime-valid guarantees.

## 5.3 Efect of the minibatch size

We now repeat the previous experiment on covertype.binary with $b \in \{ 1 , 3 2 , 1 0 2 4 \}$ , all other parameters being unchanged. All three runs are given the same budget of $1 0 ^ { 7 }$ gradient lookups. Although the performance of SGD may difer, the goal of this experiment is not to isolate how fast the algorithm converges, but how tightly its progress can be certified. The question of choosing the minibatch size b is beyond the scope of this paper; here we only study its efect on the confidence sequences.

Figure 2 reports the results. The ordering of the curves remains the same across all three experiments, and the overall picture of Section 5.2 is unchanged. We make the following additional observations:

![](images/6fee2be79e8cbaee460a7c8ffd2f87ff350c4edab84703edd35bf96abb2e219b.jpg)  
Figure 2: SGD on the covertype.binary dataset for minibatch sizes $b = 1$ (left), 32 (middle), and 1024 (right). Top row: squared distance to the optimum. Bottom row: suboptimality of the weighted average. All panels use the conventions of Figure 1: ground truth (solid gray line), worst-case bound (dash-dotted blue line), H (solid orange line with squares), and EB (solid red line with triangles); RSS12 (dotted blue line) is shown in the distance panels only.

• The bound of Rakhlin et al. [36] is about one order of magnitude above our worst-case baseline at $b = 1$ . As b grows, the gap widens to three orders of magnitude at $b = 1 0 2 4$ . At $b = 1$ the diference is primarily due to the explicit constants, while for larger values of b the gap widens because $\check { U } ^ { \mathrm { d i s t } }$ captures the variance reduction due to minibatching through the proxy $\sigma ^ { 2 } / b ,$ , whereas RSS12 does not.

• Hoefding-based H is indistinguishable from worst-case at $b = 1$ . The adaptive contribution in H comes from retaining the realized squared minibatch-gradient norms $\| \bar { g } _ { s } \| ^ { 2 }$ in the deterministic accumulation, whereas the recursive deviation term uses the fixed variance proxy $\sigma ^ { 2 } / b .$ At $b = 1$ 2 the latter dominates, so replacing $\| g _ { s } \| ^ { 2 }$ by $G ^ { 2 }$ changes little. As b increases, the deviation term shrinks and the trajectory-dependent accumulation becomes relatively more important, making the efect of adaptivity increasingly visible.

• Empirical-Bernstein-based EB improves on H by about one order of magnitude already at $b = 1$ and this gap widens with b. Unlike H, whose deviation term is controlled by the fixed variance proxy $\sigma ^ { 2 } / b$ , EB exploits the realized second-moment structure within each minibatch through $\lambda _ { \operatorname* { m a x } } ( \widehat { \Sigma } _ { t } )$ . The contribution of each minibatch to the leading square-root term therefore carries the usual $1 / \sqrt { b }$ scaling while adapting to $\lambda _ { \operatorname* { m a x } } ( \widehat { \Sigma } _ { t } ) \leq \mathrm { t r } ( \widehat { \Sigma } _ { t } ) \leq G ^ { 2 }$ . In addition, the lower-order range term carries an explicit factor $1 / b$ and hence decreases faster with the minibatch size.

• The gap between the ground truth and the worst-case baseline also widens with b. The EB confidence sequence nevertheless recovers a growing share of this gap. Define

$$
\rho _ { t } : = \frac { \log ( \check { U } _ { t } ^ { \mathrm { s u b } } ( \alpha ) / \widehat { U } _ { t } ^ { \mathrm { s u b , E B , M B } } ( \alpha ) ) } { \log ( \check { U } _ { t } ^ { \mathrm { s u b } } ( \alpha ) / ( f ( \bar { x } _ { t } ) - f ( x ^ { \star } ) ) ) } ,
$$

![](images/bb8c53e09c368d84abf49946a324292cb9d9dc1dd161f44b029ecabaef56f34e.jpg)

![](images/4076860693c040ac27f2cfce5c67a663da2ec670c7d040d67f0b313fdb10a70b.jpg)

Figure 3: SGD on three datasets generated with make\_classification, with increasing class separability from A to C. Left: squared distance to the optimum. Right: suboptimality of the weighted average. Each panel shows the ground truth (solid gray lines) and EB (solid red lines) for each dataset, with increasing shade darkness from A to C and each curve labelled accordingly, together with the worst-case bound (dash-dotted blue line); RSS12 (dotted blue line) is shown in the distance panel only.
<table><tr><td></td><td>class_sep  $\mathtt { f l i p \_ y }$ </td></tr><tr><td>A (low separability) 0.2</td><td>0.1</td></tr><tr><td>B (medium separability) 1.0</td><td>0.02</td></tr><tr><td>C (high separability) 5.0</td><td>0.0</td></tr></table>

Table 2: Parameters of each generated dataset.

the fraction of the gap between the worst-case bound and the ground truth recovered by EB on a logarithmic scale. At the end of the budget, we obtain $\rho = 2 4 . 8 \%$ , 43.6%, and 53.8% for b = 1, 32, 1024, respectively.

## 5.4 Adaptivity across problem instances

In this experiment, we aim to isolate the trajectory adaptivity of the empirical Bernstein confidence sequence. We generate three datasets with make\_classification, of increasing class separability, and report the evolution of the ground-truth quantities $Z _ { t }$ and $f ( \bar { x } _ { t } ) - f ( x ^ { \star } )$ , together with EB (Figure 3).

The dataset parameters are given in Table 2. For all three, we set n\_samples $= 1 0 ^ { 5 }$ , n\_features = 20, and n\_clusters\_per\_class = 1. Each dataset is then rescaled so that G has the same value across all three. Since $\mu , \alpha , b = 2 5 6 .$ , and $R _ { 0 }$ are also fixed, the worst-case bound is identical across the three instances.

On these instances, SGD makes faster progress on the more highly separated datasets, and the confidence sequences follow the same ordering. The relationship between the geometry of the data and the performance of SGD on the SVM primal problem is outside the scope of this paper; what matters here is that this variation in performance is invisible to the worst-case analysis but reflected by the trajectory-adaptive confidence sequences. The same deterministic bound applies to all three runs, whereas EB separates them and, after $1 0 ^ { 6 }$ SGD iterations, is roughly 17 times tighter for C than for A in suboptimality: $\widehat { U } ^ { \mathrm { s u b , E B , M B } } = 1 . 9 \times 1 0 ^ { - 6 }$ for C, compared with $3 . 2 \times 1 0 ^ { - 5 }$ for A.

![](images/e0f4981d64f7ecc5bf7cce4c84ea09d0cfad1a02a69164573044ca44aaa9b3d6.jpg)

![](images/c6c855d80b307a07de6b4c9f7a622a0f081facce590a769dac5adb2d4df18d21.jpg)  
Figure 4: SGD on dataset B from Table 2, with $b = 2 5 6$ , comparing the bounds obtained with the correct gradient bound G and with G overestimated by a factor of 10. Solid and dash-dotted lines correspond to the correct value, while dashed lines in lighter shades correspond to the overestimated one. Each panel shows the ground truth (solid gray line), the worst-case bound (blue line), H (orange line with squares), and EB (red line with triangles).

This behavior is consistent with the empirical Bernstein construction, which adapts to the realized second-moment structure within each minibatch through $\lambda _ { \operatorname* { m a x } } ( \widehat { \Sigma } _ { t } )$ . Better separated datasets have fewer samples with an active hinge loss, and in our experiments this is accompanied by substantially smaller realized values of $\lambda _ { \operatorname* { m a x } } ( \widehat { \Sigma } _ { t } )$ . The average value of $\lambda _ { \operatorname* { m a x } } ( \widehat { \Sigma } _ { t } )$ over the last 1% of the run decreases by approximately 73% from A to B, and by approximately 98% from A to C. A deterministic bound depending on G alone cannot distinguish these trajectories.

## 5.5 Robustness to parameter misspecification

The bounds considered above depend on deterministic problem constants such as $G , \sigma ^ { 2 }$ , and $R _ { 0 }$ , whose roles difer across the diferent constructions. These quantities are rarely available exactly in practice. The situation is not symmetric: choosing a value that is too small can invalidate the corresponding guarantee, whereas a conservative upper bound preserves validity at the price of tightness. The practical question is therefore what is lost when these constants are chosen too conservatively. We do not vary $\mu ,$ since it enters the stepsize schedule and changing it would therefore also change the SGD trajectory itself.

Misspecification of the gradient bound G. We first compare the bounds computed with the correct value of G to the same bounds computed with G overestimated by a factor of 10. For H and the worst-case baseline, we use the valid individual-gradient proxy $\sigma ^ { 2 } = G ^ { 2 }$ , so overestimating G by a factor of 10 also overestimates $\sigma ^ { 2 }$ by a factor of 100. The experiment is run on dataset B from the previous subsection, with the SGD trajectory itself unchanged. Figure 4 reports both families of curves.

The worst-case bounds degrade as expected for both squared distance and suboptimality. Their leading asymptotic terms scale with $G ^ { 2 }$ , and the misspecified curves lie approximately a factor of 100 above their well-specified counterparts. A similarly strong efect is observed for Hoefding-based H, reflecting the dependence of its deviation term on the fixed variance proxy $\sigma ^ { 2 } / b$ rather than on the realized second-moment structure of the stochastic gradients.

Empirical-Bernstein-based EB behaves diferently. Its leading deviation term adapts to the realized second-moment structure within each minibatch through $\lambda _ { \operatorname* { m a x } } ( \widehat { \Sigma } _ { t } )$ , while $G$ enters only through the lower-order range term of the empirical Bernstein boundary. The misspecified and well-specified EB curves are consequently asymptotically equivalent. They difer noticeably only at first and agree within a factor of 1.06 after $1 0 ^ { 7 }$ iterations, for both $\widehat { U } ^ { \mathrm { d i s t , E B , M B } } \ \mathrm { a n d } \ \widehat { U } ^ { \mathrm { s u b , E B , M B } }$

![](images/4edb3c2bc06095ed9971aed54a443746cd168cc7849106be9e433f35f17daf19.jpg)

![](images/5c61fad0542ad84fa4ce8afc71d1617cec9dbbedca105a3605dbe906741ced2d.jpg)  
Figure 5: SGD on make\_classification dataset, $b = 2 5 6$ , comparing EB computed with the correct value of $R _ { 0 }$ and with $R _ { 0 }$ over-estimated by factors 10, 100 and 1000. Each panel shows the ground truth (solid gray line) and the four confidence sequences (red lines, in increasingly light shades and increasingly sparse dotting as the over-estimation factor grows, as indicated in the legend).

Misspecification of the initial distance bound $R _ { 0 }$ . We repeat the experiment with $R _ { 0 }$ overestimated by factors 10, 100, and 1000, while keeping all other quantities fixed, and report EB in each case in Figure 5. Unlike G, $R _ { 0 }$ enters through the initialization of the recursive distance envelope, and its influence therefore diminishes as the trajectory-dependent terms accumulate. For the squared-distance confidence sequence, the efect is pronounced initially but vanishes rapidly: with $R _ { 0 }$ overestimated by a factor of 1000, the confidence sequence initially difers from the correctly specified one by the same factor, but their relative diference is only 4% after 100 iterations and 0.01% after $1 0 ^ { 3 }$ iterations.

Taken together, these two experiments show that, along the trajectories considered here, the efect of conservative choices of G and $R _ { 0 }$ on EB is largely transient. This reflects the trajectory adaptivity of the empirical Bernstein construction: its leading deviation term is driven by realized second-moment quantities, while G enters through the lower-order range term and the influence of $R _ { 0 }$ is rapidly attenuated by the recursive construction. In practice, this suggests that safe upper bounds on G and $R _ { 0 }$ need not be specified very precisely for the resulting EB certificates to become largely insensitive to them later in the run.

## 5.6 True single-pass

All experiments above sample with replacement from the empirical distribution of a finite dataset, which makes (74) the empirical risk and allows $x ^ { \star }$ to be computed beforehand. We now turn to the population setting, where the observations are modeled as i.i.d. samples from an unknown datagenerating distribution P. Under this standard model, processing the dataset once in an order chosen independently of the observations has the same distribution as sequential i.i.d. sampling from P. Two consequences follow: the minimizer of the expected risk cannot be computed beforehand, so no ground truth is available, and the run is necessarily limited by the available sample size.

We run this experiment on covertype.binary, which contains N = 581012 observations, with minibatch size $b = 2 5 6$ , and report the evolution of worst-case, H, and EB in Figure 6. Using complete minibatches, the run stops after $\lfloor N / b \rfloor = 2 2 6 9 \mathrm { S G D }$ iterations, with no observation reused. The picture is essentially the same as the one observed throughout this section. For suboptimality, $\widehat { U } _ { t } ^ { \mathrm { s u b } }$ is 6.4 times tighter than $\check { U } _ { t } ^ { \mathrm { s u b } }$ , while $\widehat { U } _ { t } ^ { \mathrm { s u b , E B , M B } }$ is $2 . 7 \times 1 0 ^ { 2 }$ times tighter. For comparison, in the experiment of Section 5.2, the corresponding factors are also 6.4 and $2 . 7 \times 1 0 ^ { 2 }$ after the same number of iterations. The improvements reported above are therefore not an artifact of repeatedly sampling from a finite empirical distribution and persist in this single-pass expected-risk setting.

![](images/153978fedfff68b867dfd7640422f88f1a67c4f6fe353880d0f498814d8b449f.jpg)

![](images/bef2be2f8f6353482b985e5318b57203e75206d97da704de47de19b096bd06bc.jpg)  
Figure 6: SGD on the covertype.binary dataset in a true single-pass regime, with $b = 2 5 6$ . Each panel shows the worst-case bound (dash-dotted blue line), H (solid orange line with squares), and EB (solid red line with triangles).

## 5.7 Summary of the empirical findings

As shown in these experiments, on real-world data our confidence sequences can certify a target accuracy several orders of magnitude earlier than the trajectory-independent bounds against which they are compared. This improvement reflects their ability to adapt to the realized SGD trajectory. When all problem constants are fixed and only the problem instance varies, the worst-case bounds remain unchanged while our confidence sequences adapt to the instance. Minibatching further amplifies these diferences, and our bounds recover a growing share of the gap between the worst-case bounds and the ground truth as b increases.

The two constructions nevertheless behave diferently. The empirical-Bernstein-based sequences are consistently tighter than their Hoefding-based counterparts in our experiments and appear particularly attractive when the stochastic gradients can be uniformly bounded. Their leading deviation terms adapt to realized second-moment quantities, while the efects of conservative choices of G and $R _ { 0 }$ become negligible later in the runs considered above. By contrast, the Hoefding-based sequences retain an asymptotic dependence on the fixed sub-Gaussian variance proxy $\sigma ^ { 2 }$

Even with these improvements, a considerable gap remains between the confidence sequences and the ground truth in the experiments where the latter is available. Whether this residual gap can be reduced through a sharper analysis of the SGD dynamics is an interesting question that we leave open.

## Acknowledgements

Liviu Aolaritei acknowledges support from the Swiss National Science Foundation through the Postdoc.Mobility Fellowship (grant agreement P500PT\_222215). Funded in part by the European Union (ERC-2022-SYG-OCEAN-101071601). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

## References

[1] F. J. Anscombe. Fixed-sample-size analysis of sequential observations. Biometrics, 10(1):89–100, 1954.

[2] L. Aolaritei and M. I. Jordan. Stopping rules for stochastic gradient descent via anytime-valid confidence sequences. arXiv preprint arXiv:2512.13123, 2025.

[3] F. Bach. Adaptivity of averaged stochastic gradient descent to local strong convexity for logistic regression. The Journal of Machine Learning Research, 15(1):595–627, 2014.

[4] F. Bach and E. Moulines. Non-asymptotic analysis of stochastic approximation algorithms for machine learning. Advances in Neural Information Processing Systems, 24, 2011.

[5] L. Bottou, F. E. Curtis, and J. Nocedal. Optimization methods for large-scale machine learning. SIAM review, 60(2):223–311, 2018.

[6] K. Chen, Y. Feng, and T. Wang. Revisiting stochastic gradient descent for strongly convex objectives: Tight uniform-in-time bounds. Systems & Control Letters, 212:106419, 2026.

[7] D. A. Darling and H. Robbins. Confidence sequences for mean, variance, and median. Proceedings of the National Academy of Sciences, 58(1):66–68, 1967.

[8] D. A. Darling and H. Robbins. Iterated logarithm inequalities. Proceedings of the National Academy of Sciences, 57(5):1188–1192, 1967.

[9] D. A. Darling and H. Robbins. Some nonparametric sequential tests with power one. Proceedings of the National Academy of Sciences, 61(3):804–809, 1968.

[10] V. H. de la Peña, M. J. Klass, and T. L. Lai. Self-normalized processes: Exponential inequalities, moment bounds and iterated logarithm laws. The Annals of Probability, 32(3):1902–1933, 2004.

[11] V. H. de la Peña, M. J. Klass, and T. L. Lai. Pseudo-maximization and self-normalized processes. Probability Surveys, 4:172–192, 2007.

[12] V. H. de la Peña, T. L. Lai, and Q.-M. Shao. Self-normalized Processes: Limit Theory and Statistical Applications. Springer, 2009.

[13] J. Duchi, E. Hazan, and Y. Singer. Adaptive subgradient methods for online learning and stochastic optimization. Journal of machine learning research, 12(7), 2011.

[14] R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang, and C.-J. Lin. Liblinear: A library for large linear classification. the Journal of machine Learning research, 9:1871–1874, 2008.

[15] X. Fan, I. Grama, and Q. Liu. Exponential inequalities for martingales with applications. Electronic Journal of Probability, 20:1–22, 2015.

[16] W. K. Feller. Statistical aspects of ESP. The Journal of Parapsychology, 4(2):271, 1940.

[17] S. Ghadimi and G. Lan. Optimal stochastic approximation algorithms for strongly convex stochastic composite optimization i: A generic algorithmic framework. SIAM Journal on Optimization, 22(4):1469–1492, 2012.

[18] S. Ghadimi and G. Lan. Optimal stochastic approximation algorithms for strongly convex stochastic composite optimization, ii: shrinking procedures and optimal algorithms. SIAM Journal on Optimization, 23(4):2061–2089, 2013.

[19] P. Grünwald, R. de Heide, and W. Koolen. Safe testing. Journal of the Royal Statistical Society Series B: Statistical Methodology, 86(5):1091–1128, 2024.

[20] N. J. Harvey, C. Liaw, Y. Plan, and S. Randhawa. Tight analyses for non-smooth stochastic gradient descent. In Conference on Learning Theory, pages 1579–1613. PMLR, 2019.

[21] N. J. Harvey, C. Liaw, and S. Randhawa. Simple and optimal high-probability bounds for stronglyconvex stochastic gradient descent. arXiv preprint arXiv:1909.00843, 2019.

[22] E. Hazan and S. Kale. Beyond the regret minimization barrier: an optimal algorithm for stochastic strongly-convex optimization. In Proceedings of the 24th Annual Conference on Learning Theory, pages 421–436. JMLR Workshop and Conference Proceedings, 2011.

[23] W. Hoefding. Probability inequalities for sums of bounded random variables. Journal of the American statistical association, 58(301):13–30, 1963.

[24] S. R. Howard, A. Ramdas, J. McAulife, and J. Sekhon. Time-uniform, nonparametric, nonasymptotic confidence sequences. The Annals of Statistics, 49(2):1055–1080, 2021.

[25] P. Jain, D. M. Nagaraj, and P. Netrapalli. Making the last iterate of SGD information theoretically optimal. SIAM Journal on Optimization, 31(2):1108–1130, 2021.

[26] S. Lacoste-Julien, M. Schmidt, and F. Bach. A simpler approach to obtaining an O(1/t) convergence rate for the projected stochastic subgradient method. arXiv preprint arXiv:1212.2002, 2012.

[27] T. L. Lai. On confidence sequences. The Annals of Statistics, pages 265–280, 1976.

[28] T. L. Lai and H. Robbins. Adaptive design and stochastic approximation. The Annals of Statistics, 7(6):1196–1221, 1979.

[29] Z. Liu and Z. Zhou. Revisiting the last-iterate convergence of stochastic gradient methods. In International Conference on Learning Representations, volume 2024, pages 40394–40428, 2024.

[30] A. Maurer and M. Pontil. Empirical bernstein bounds and sample variance penalization. In Proceedings of the 22nd Annual Conference on Learning Theory, 2009.

[31] A. Nedic and S. Lee. On stochastic subgradient mirror-descent algorithm with weighted averaging. SIAM Journal on Optimization, 24(1):84–107, 2014.

[32] A. Nemirovski, A. Juditsky, G. Lan, and A. Shapiro. Robust stochastic approximation approach to stochastic programming. SIAM Journal on Optimization, 19(4):1574–1609, 2009.

[33] T. Pham, A. Rinaldo, and P. Sarkar. Time-uniform concentration bounds for iterative algorithms. arXiv preprint arXiv:2511.18273, 2025.

[34] B. T. Polyak and A. B. Juditsky. Acceleration of stochastic approximation by averaging. SIAM Journal on Control and Optimization, 30(4):838–855, 1992.

[35] L. Prechelt. Early stopping-but when? In Neural Networks: Tricks of the trade, pages 55–69. Springer, 1998.

[36] A. Rakhlin, O. Shamir, and K. Sridharan. Making gradient descent optimal for strongly convex stochastic optimization. In International Conference on Machine Learning, pages 1571–1578, 2012.

[37] A. Ramdas and R. Wang. Hypothesis testing with e-values. Foundations and Trends® in Statistics, 1(1–2):1–390, 2025.

[38] A. Ramdas, P. Grünwald, V. Vovk, and G. Shafer. Game-theoretic statistics and safe anytime-valid inference. Statistical Science, 38(4):576–601, 2023.

[39] H. Robbins. Some aspects of the sequential design of experiments. Bulletin of the American Mathematical Society, 58(5):527–535, 1952.

[40] H. Robbins. Statistical methods related to the law of the iterated logarithm. The Annals of Mathematical Statistics, 41(5):1397–1409, 1970.

[41] H. Robbins and S. Monro. A stochastic approximation method. The Annals of Mathematical Statistics, pages 400–407, 1951.

[42] H. Robbins and D. Siegmund. Boundary crossing probabilities for the Wiener process and sample sums. The Annals of Mathematical Statistics, pages 1410–1429, 1970.

[43] H. Robbins and D. Siegmund. A convergence theorem for non negative almost supermartingales and some applications. In Optimizing methods in statistics, pages 233–257. Elsevier, 1971.

[44] G. Shafer. Testing by betting: A strategy for statistical and scientific communication. Journal of the Royal Statistical Society: Series A (Statistics in Society), 184(2):407–431, 2021.

[45] O. Shamir and T. Zhang. Stochastic gradient descent for non-smooth optimization: Convergence results and optimal averaging schemes. In International Conference on Machine Learning, pages 71–79. PMLR, 2013.

[46] N. Tripuraneni, N. Flammarion, F. Bach, and M. I. Jordan. Averaging stochastic gradient descent on Riemannian manifolds. In Conference On Learning Theory, pages 650–687. PMLR, 2018.

[47] J. Ville. Étude Critique de la Notion de Collectif. Gauthier-Villars, Paris, 1939.

[48] V. Vovk and R. Wang. E-values: Calibration, combination and applications. The Annals of Statistics, 49(3):1736–1754, 2021.

[49] A. Wald. Sequential tests of statistical hypotheses. The Annals of Mathematical Statistics, 16(2): 117–186, 1945.

[50] R. Ward, X. Wu, and L. Bottou. Adagrad stepsizes: Sharp convergence over nonconvex landscapes. Journal of Machine Learning Research, 21(219):1–30, 2020.

## A Proofs

## A.1 Proofs of Section 2

## A.1.1 Proof of Lemma 2.1

For each $\lambda > 0$ , define

$$
M _ { t } ( \lambda ) : = \exp \left( \lambda \bar { X } _ { t } - 2 ^ { \gamma } \lambda ^ { 2 } V _ { t } \right) , \qquad t \geq t _ { 0 } - 1 ,
$$

with $M _ { t _ { 0 } - 1 } ( \lambda ) = 1$ . By the conditional exponential bound,

$$
\begin{array} { r } { \mathbb { E } [ M _ { t } ( \lambda ) \mid \mathcal { F } _ { t - 1 } ] = M _ { t - 1 } ( \lambda ) \mathbb { E } \left[ \exp \left( \lambda X _ { t } - 2 ^ { \gamma } \lambda ^ { 2 } v _ { t } \right) \Big | \mathcal { F } _ { t - 1 } \right] \leq M _ { t - 1 } ( \lambda ) . } \end{array}
$$

Thus $\{ M _ { t } ( \lambda ) \} _ { t \ge t _ { 0 } - 1 }$ is a nonnegative supermartingale. By Ville’s inequality, for every $c > 0$

$$
\begin{array} { r } { \mathbb { P } \left( \exists t \geq t _ { 0 } - 1 : \lambda \bar { X } _ { t } - 2 ^ { \gamma } \lambda ^ { 2 } V _ { t } \geq c \right) \leq e ^ { - c } . } \end{array}
$$

Equivalently,

$$
\mathbb { P } \left( \exists t \geq t _ { 0 } - 1 : \bar { X } _ { t } \geq \frac { c } { \lambda } + 2 ^ { \gamma } \lambda V _ { t } \right) \leq e ^ { - c } .\tag{78}
$$

We now stitch these fixed-λ bounds over dyadic ranges of the variance process. Define

$$
I _ { 0 } : = [ 0 , 1 ] , \qquad I _ { m } : = ( 2 ^ { m - 1 } , 2 ^ { m } ] , \qquad m \ge 1 .
$$

For $m \geq 0$ , set

$$
\alpha _ { m } : = \frac { \alpha } { ( m + 1 ) ( m + 2 ) } , \qquad c _ { m } : = \log \frac { 1 } { \alpha _ { m } } = \log \frac { 1 } { \alpha } + \log \bigl ( ( m + 1 ) ( m + 2 ) \bigr ) .
$$

Then $\textstyle \sum _ { m = 0 } ^ { \infty } \alpha _ { m } = \alpha$ . For each $m \geq 0$ , choose

$$
\lambda _ { m } : = { \sqrt { \frac { c _ { m } } { 2 ^ { \gamma } 2 ^ { m } } } } .
$$

Applying (78) with $( \lambda , c ) = ( \lambda _ { m } , c _ { m } )$ gives

$$
\mathbb { P } \left( \exists t \geq t _ { 0 } - 1 : \bar { X } _ { t } \geq \frac { c _ { m } } { \lambda _ { m } } + 2 ^ { \gamma } \lambda _ { m } V _ { t } \right) \leq \alpha _ { m } .
$$

Suppose now that $V _ { t } \in I _ { m }$ . Then $V _ { t } \le 2 ^ { m }$ , so on the complement of the above event,

$$
\bar { X } _ { t } < \frac { c _ { m } } { \lambda _ { m } } + 2 ^ { \gamma } \lambda _ { m } 2 ^ { m } .
$$

By the choice of $\lambda _ { m }$ ,

$$
{ \frac { c _ { m } } { \lambda _ { m } } } = { \sqrt { 2 ^ { \gamma } c _ { m } 2 ^ { m } } } , \qquad 2 ^ { \gamma } \lambda _ { m } 2 ^ { m } = { \sqrt { 2 ^ { \gamma } c _ { m } 2 ^ { m } } } .
$$

Hence

$$
\bar { X } _ { t } < 2 \sqrt { 2 ^ { \gamma } c _ { m } 2 ^ { m } } .
$$

If $m \geq 1$ , then $V _ { t } > 2 ^ { m - 1 }$ , and so $2 ^ { m } < 2 V _ { t }$ . Therefore

$$
\bar { X } _ { t } < 2 \sqrt { 2 ^ { \gamma } c _ { m } 2 V _ { t } } = 2 ^ { ( \gamma + 3 ) / 2 } \sqrt { V _ { t } c _ { m } } .
$$

If $m = 0$ , then $V _ { t , \mathrm { e f f } } = 1$ , and

$$
\begin{array} { r } { \bar { X } _ { t } < 2 \sqrt { 2 ^ { \gamma } c _ { 0 } } \leq 2 ^ { ( \gamma + 3 ) / 2 } \sqrt { V _ { t , \mathrm { e f f } } c _ { 0 } } . } \end{array}
$$

Thus, for every $m \geq 0$ , outside an event of probability at most $\alpha _ { m }$

$$
V _ { t } \in I _ { m } \quad \Longrightarrow \quad \bar { X } _ { t } < 2 ^ { ( \gamma + 3 ) / 2 } \sqrt { V _ { t , \mathrm { e f f } } c _ { m } } .
$$

Taking a union bound over $m \geq 0$ , with probability at least $\textstyle 1 - \sum _ { m = 0 } ^ { \infty } \alpha _ { m } = 1 - \alpha$ , the preceding implication holds simultaneously for all $m \geq 0$ and all $t \geq t _ { 0 } - 1$ . On this event, fix any $t \geq t _ { 0 }$ and let

$$
m _ { t } : = \left\lceil \log _ { 2 } V _ { t , \mathrm { e f f } } \right\rceil .
$$

Then $V _ { t } \in I _ { m _ { t } }$ , and therefore

$$
\bar { X } _ { t } \leq 2 ^ { ( \gamma + 3 ) / 2 } \sqrt { V _ { t , \mathrm { e f f } } \left( \log \frac { 1 } { \alpha } + \log \left( ( m _ { t } + 1 ) ( m _ { t } + 2 ) \right) \right) } .
$$

Since $t \geq t _ { 0 }$ was arbitrary, the result follows.

## A.1.2 Proof of Proposition 2.2

By Lemma C.2, for every $t \geq t _ { 0 }$ 2

$$
Z _ { t + 1 } \leq \frac { 1 } { A _ { t } } \left[ Z _ { t _ { 0 } } + \sum _ { s = t _ { 0 } } ^ { t } A _ { s } \eta _ { s } ^ { 2 } \left. g _ { s } \right. ^ { 2 } + \bar { X } _ { t } \right] .
$$

It remains to control $X _ { t }$ uniformly over time. Recall that

$$
X _ { t } = 2 A _ { t } \eta _ { t } \langle x ^ { \star } - x _ { t } , \xi _ { t } \rangle .
$$

Since $x _ { t } , \ n _ { t }$ , and $A _ { t }$ are $\mathcal { F } _ { t - 1 }$ -measurable, applying Assumption 1.3(ii) with the predictable direction $2 A _ { t } \eta _ { t } ( x ^ { \star } - x _ { t } )$ gives

$$
\mathbb { E } \left[ e ^ { \lambda X _ { t } } \Big | \mathcal { F } _ { t - 1 } \right] \leq \exp \left( 2 \lambda ^ { 2 } \sigma ^ { 2 } A _ { t } ^ { 2 } \eta _ { t } ^ { 2 } Z _ { t } \right)
$$

for every $\lambda \geq 0$ . Thus, the sequences $\{ X _ { t } \}$ and $\{ v _ { t } ^ { \mathrm { d i s t } } \}$ satisfy the hypotheses of Lemma 2.1 with

$$
v _ { t } ^ { \mathrm { d i s t } } = \sigma ^ { 2 } A _ { t } ^ { 2 } \eta _ { t } ^ { 2 } Z _ { t } , \qquad \gamma = 1 .
$$

Applying Lemma 2.1 therefore yields

$$
\begin{array} { r } { \mathbb { P } \left( \forall t \geq t _ { 0 } : \ \bar { X } _ { t } \leq 4 \mathfrak { H } _ { \alpha } \left( V _ { t } ^ { \mathrm { d i s t } } \right) \right) \geq 1 - \alpha . } \end{array}
$$

Substituting this bound into the preceding decomposition and using the definition of $U _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha )$ in (14) proves the result.

## A.1.3 Proof of Theorem 2.3

We first check that the recursion is well-defined. The quantity $\widehat { U } _ { t _ { 0 } + 1 } ^ { \mathrm { d i s t } } ( \alpha )$ is defined explicitly from $R _ { 0 }$ and the data at time $t _ { 0 }$ . Inductively, if

$$
\widehat { U } _ { t _ { 0 } + 1 } ^ { \mathrm { d i s t } } ( \alpha ) , \ldots , \widehat { U } _ { t } ^ { \mathrm { d i s t } } ( \alpha )
$$

have been defined, then $\widehat { V } _ { t } ^ { \mathrm { d i s t } }$ is also defined, which in turn defines $\widehat { U } _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha )$ . Thus the sequence is recursively well-defined. Moreover, each $\widehat { U } _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha )$ depends only on $R _ { 0 }$ and the observed trajectory up to time $t ,$ so the construction is fully observable. Now let

$$
U _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha ) : = \frac { 1 } { A _ { t } } \left[ \mathcal { Z } _ { t _ { 0 } } + \sum _ { s = t _ { 0 } } ^ { t } A _ { s } \eta _ { s } ^ { 2 } \left. g _ { s } \right. ^ { 2 } + 4 \mathfrak { H } _ { \alpha } \left( V _ { t } ^ { \mathrm { d i s t } } \right) \right]
$$

be the adaptive confidence sequence from Proposition 2.2, where

$$
V _ { t } ^ { \mathrm { { d i s t } } } = \sum _ { s = t _ { 0 } } ^ { t } \sigma ^ { 2 } A _ { s } ^ { 2 } \eta _ { s } ^ { 2 } Z _ { s } .
$$

Set

$$
E _ { \alpha } : = \left\{ \forall t \geq t _ { 0 } : \ Z _ { t + 1 } \leq U _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha ) \right\} .
$$

$\mathrm { B y }$ Proposition 2.2, $\mathbb { P } ( E _ { \alpha } ) \ge 1 - \alpha$ . We will show that on $E _ { \alpha }$

$$
U _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha ) \leq \widehat { U } _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha ) \qquad \mathrm { f o r ~ a l l ~ } t \geq t _ { 0 } .
$$

This will imply

$$
Z _ { t + 1 } \leq U _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha ) \leq { \widehat { U } } _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha ) \qquad { \mathrm { f o r ~ a l l ~ } } t \geq t _ { 0 }
$$

on $E _ { \alpha }$ , and therefore establish the claim. By (12) and (13), the map $v \mapsto \mathfrak { H } _ { \alpha } ( v )$ is nondecreasing on $[ 0 , \infty )$ . We now prove by induction on $t \geq t _ { 0 }$ that on $E _ { \alpha }$

$$
V _ { t } ^ { \mathrm { d i s t } } \leq \widehat { V } _ { t } ^ { \mathrm { d i s t } } \qquad \mathrm { a n d } \qquad U _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha ) \leq \widehat { U } _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha ) .
$$

For the base case $t = t _ { 0 }$ , since $R _ { 0 } \geq Z _ { t _ { 0 } }$ , we have

$$
V _ { t _ { 0 } } ^ { \mathrm { d i s t } } = \sigma ^ { 2 } A _ { t _ { 0 } } ^ { 2 } \eta _ { t _ { 0 } } ^ { 2 } Z _ { t _ { 0 } } \le \sigma ^ { 2 } A _ { t _ { 0 } } ^ { 2 } \eta _ { t _ { 0 } } ^ { 2 } R _ { 0 } = \widehat { V } _ { t _ { 0 } } ^ { \mathrm { d i s t } } .
$$

By monotonicity,

$$
\mathfrak { H } _ { \alpha } \left( V _ { t _ { 0 } } ^ { \mathrm { d i s t } } \right) \leq \mathfrak { H } _ { \alpha } \left( \widehat { V } _ { t _ { 0 } } ^ { \mathrm { d i s t } } \right) .
$$

Using also $Z _ { t _ { 0 } } \leq R _ { 0 }$ , we obtain

$$
U _ { t _ { 0 } + 1 } ^ { \mathrm { d i s t } } ( \alpha ) = \frac { 1 } { A _ { t _ { 0 } } } \left[ Z _ { t _ { 0 } } + A _ { t _ { 0 } } \eta _ { t _ { 0 } } ^ { 2 } \left. g _ { t _ { 0 } } \right. ^ { 2 } + 4 \mathfrak { H } _ { 0 } \left( V _ { t _ { 0 } } ^ { \mathrm { d i s t } } \right) \right] \leq \frac { 1 } { A _ { t _ { 0 } } } \left[ R _ { 0 } + A _ { t _ { 0 } } \eta _ { t _ { 0 } } ^ { 2 } \left. g _ { t _ { 0 } } \right. ^ { 2 } + 4 \mathfrak { H } _ { 0 } \left( \widehat { V } _ { t _ { 0 } } ^ { \mathrm { d i s t } } \right) \right]
$$

Now fix $t \geq t _ { 0 } + 1$ , and suppose that for every $s = t _ { 0 } , \ldots , t - 1$ 2

$$
V _ { s } ^ { \mathrm { d i s t } } \leq \widehat { V } _ { s } ^ { \mathrm { d i s t } } \qquad \mathrm { a n d } \qquad U _ { s + 1 } ^ { \mathrm { d i s t } } ( \alpha ) \leq \widehat { U } _ { s + 1 } ^ { \mathrm { d i s t } } ( \alpha ) .
$$

Since we are on $E _ { \alpha } .$ , for each $s = t _ { 0 } + 1 , \ldots , t$

$$
Z _ { s } \leq U _ { s } ^ { \mathrm { d i s t } } ( \alpha ) \leq \widehat { U } _ { s } ^ { \mathrm { d i s t } } ( \alpha ) .
$$

Therefore,

$$
V _ { t } ^ { \mathrm { d i s t } } = \sigma ^ { 2 } A _ { t _ { 0 } } ^ { 2 } \eta _ { t _ { 0 } } ^ { 2 } Z _ { t _ { 0 } } + \sum _ { s = t _ { 0 } + 1 } ^ { t } \sigma ^ { 2 } A _ { s } ^ { 2 } \eta _ { s } ^ { 2 } Z _ { s } \leq \sigma ^ { 2 } A _ { t _ { 0 } } ^ { 2 } \eta _ { t _ { 0 } } ^ { 2 } R _ { 0 } + \sum _ { s = t _ { 0 } + 1 } ^ { t } \sigma ^ { 2 } A _ { s } ^ { 2 } \eta _ { s } ^ { 2 } \widehat { U } _ { s } ^ { \mathrm { d i s t } } ( \alpha ) = \widehat { V } _ { t } ^ { \mathrm { d i s t } } .
$$

By monotonicity again,

$$
{ \mathfrak { H } } _ { \alpha } \left( V _ { t } ^ { \mathrm { d i s t } } \right) \leq { \mathfrak { H } } _ { \alpha } \left( { \widehat { V } } _ { t } ^ { \mathrm { d i s t } } \right) .
$$

Using also $Z _ { t _ { 0 } } \leq R _ { 0 }$ , we conclude that

$$
U _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha ) = \frac { 1 } { A _ { t } } \left[ Z _ { t _ { 0 } } + \sum _ { s = t _ { 0 } } ^ { t } A _ { s } \eta _ { s } ^ { 2 } \left\| g _ { s } \right\| ^ { 2 } + 4 \mathfrak { H } _ { \alpha } \left( V _ { t } ^ { \mathrm { d i s t } } \right) \right] \leq \frac { 1 } { A _ { t } } \left[ R _ { 0 } + \sum _ { s = t _ { 0 } } ^ { t } A _ { s } \eta _ { s } ^ { 2 } \left\| g _ { s } \right\| ^ { 2 } + 4 \mathfrak { H } _ { \alpha } \left( \widehat { V } _ { t } ^ { \mathrm { d i s t } } \right) \right]
$$

This closes the induction. Hence, on $E _ { \alpha }$ ,

$$
Z _ { t + 1 } \leq { \widehat { U } } _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha ) \qquad { \mathrm { f o r ~ a l l ~ } } t \geq t _ { 0 } .
$$

Therefore,

$$
\mathbb { P } \left( \forall t \geq t _ { 0 } : \ Z _ { t + 1 } \leq { \widehat { U } } _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha ) \right) \geq \mathbb { P } ( E _ { \alpha } ) \geq 1 - \alpha .
$$

This completes the proof.

## A.1.4 Proof of Proposition 2.4

Fix $\alpha \in ( 0 , 1 )$ and write $L _ { t } : = L _ { t } ( \alpha )$ . Since $B \geq 1$ , the sequence $L _ { t }$ is nondecreasing and satisfies $L _ { t } \geq 1$ for all $t \geq 1$ . For convenience, throughout the proof define

$$
\begin{array} { r } { \widehat { V } _ { t , \mathrm { e f f } } ^ { \mathrm { d i s t } } : = \operatorname* { m a x } \left\{ \widehat { V } _ { t } ^ { \mathrm { d i s t } } , 1 \right\} , \qquad \widehat { m } _ { t } ^ { \mathrm { d i s t } } : = m \left( \widehat { V } _ { t } ^ { \mathrm { d i s t } } \right) = \left\lceil \log _ { 2 } \widehat { V } _ { t , \mathrm { e f f } } ^ { \mathrm { d i s t } } \right\rceil . } \end{array}
$$

For $\eta _ { t } = 1 / ( \mu t )$ and $t _ { 0 } = 3$ , we have

$$
1 - 2 \mu \eta _ { t } = 1 - \frac { 2 } { t } = \frac { t - 2 } { t } , \qquad t \geq 3 .
$$

Therefore

$$
A _ { t } = \prod _ { s = 3 } ^ { t } { \frac { s } { s - 2 } } = { \frac { t ( t - 1 ) } { 2 } } , \qquad t \geq 3 .
$$

In particular,

$$
{ \frac { 1 } { A _ { t } } } = { \frac { 2 } { t ( t - 1 ) } } .
$$

We first bound the deterministic part of the recursive envelope. Define

$$
D _ { t } : = \frac { 1 } { A _ { t } } \left[ R _ { 0 } + \sum _ { s = 3 } ^ { t } A _ { s } \eta _ { s } ^ { 2 } \left. g _ { s } \right. ^ { 2 } \right] , \qquad t \ge 3 .
$$

Since

$$
A _ { s } \eta _ { s } ^ { 2 } = { \frac { s ( s - 1 ) } { 2 } } \cdot { \frac { 1 } { \mu ^ { 2 } s ^ { 2 } } } = { \frac { s - 1 } { 2 \mu ^ { 2 } s } } \leq { \frac { 1 } { 2 \mu ^ { 2 } } } ,
$$

The almost-sure bound $\| g _ { t } \| \leq G$ gives

$$
\sum _ { s = 3 } ^ { t } A _ { s } \eta _ { s } ^ { 2 } \left. g _ { s } \right. ^ { 2 } \leq \frac { G ^ { 2 } } { 2 \mu ^ { 2 } } ( t - 2 ) .
$$

Using $R _ { 0 } \leq G ^ { 2 } / \mu ^ { 2 }$ , we obtain

$$
D _ { t } \leq \frac { 2 } { t ( t - 1 ) } \left[ \frac { G ^ { 2 } } { \mu ^ { 2 } } + \frac { G ^ { 2 } } { 2 \mu ^ { 2 } } ( t - 2 ) \right] = \frac { G ^ { 2 } } { \mu ^ { 2 } } \frac { 1 } { t - 1 } .
$$

Thus, for every $t \geq 3 .$

$$
D _ { t } \leq B \frac { 1 } { t - 1 } \leq 2 B \frac { L _ { t + 1 } } { t + 1 } ,
$$

where we used $B \geq G ^ { 2 } / \mu ^ { 2 } , L _ { t + 1 } \geq 1$ , and $1 / ( t - 1 ) \leq 2 / ( t + 1 )$ . Hence,

$$
D _ { t } \leq 2 B \frac { L _ { t + 1 } } { t + 1 } , \qquad t \geq 3 .\tag{79}
$$

Now set $K : = 4 0 9 6$ . We prove by induction that

$$
\widehat { U } _ { s } ^ { \mathrm { d i s t } } ( \alpha ) \leq K B \frac { L _ { s } } { s } , \qquad s \geq 4 .
$$

For the base case $s = 4$ , note that $A _ { 3 } = 3$ and $\eta _ { 3 } = 1 / ( 3 \mu )$ , so $A _ { 3 } ^ { 2 } \eta _ { 3 } ^ { 2 } = 1 / \mu ^ { 2 }$ . Therefore,

$$
\widehat { V } _ { 3 } ^ { \mathrm { d i s t } } = \sigma ^ { 2 } A _ { 3 } ^ { 2 } \eta _ { 3 } ^ { 2 } R _ { 0 } \leq \frac { \sigma ^ { 2 } } { \mu ^ { 2 } } \frac { G ^ { 2 } } { \mu ^ { 2 } } \leq B ^ { 2 } .
$$

Since $B \geq 1$ , this implies $\widehat { V } _ { 3 , \mathrm { e f f } } ^ { \mathrm { d i s t } } = \operatorname* { m a x } \{ \widehat { V } _ { 3 } ^ { \mathrm { d i s t } } , 1 \} \leq B ^ { 2 }$ . Consequently,

$$
\widehat { m } _ { 3 } ^ { \mathrm { d i s t } } = \left\lceil \log _ { 2 } \widehat { V } _ { 3 , \mathrm { e f f } } ^ { \mathrm { d i s t } } \right\rceil \leq 1 + 2 \log _ { 2 } B .
$$

Using $\log _ { 2 } B \leq 2$ log B for $B \geq 1$ , we get $\widehat { m } _ { 3 } ^ { \mathrm { d i s t } } + 2 \leq 3 + 2 \log _ { 2 } B \leq 4 ( 1 + \log B )$ . Hence,

$$
\begin{array} { r } { \log \big ( ( \widehat { m } _ { 3 } ^ { \mathrm { d i s t } } + 1 ) ( \widehat { m } _ { 3 } ^ { \mathrm { d i s t } } + 2 ) \big ) \leq 2 \log ( \widehat { m } _ { 3 } ^ { \mathrm { d i s t } } + 2 ) \leq 2 \log 4 + 2 \log ( 1 + \log B ) . } \end{array}
$$

Since log $( 1 + \log B ) = \log \log ( e B ) \leq L _ { 4 }$ and $L _ { 4 } \geq 1$ , the preceding display gives

$$
\log \left( ( \widehat { m } _ { 3 } ^ { \mathrm { d i s t } } + 1 ) ( \widehat { m } _ { 3 } ^ { \mathrm { d i s t } } + 2 ) \right) \leq 5 L _ { 4 } .
$$

Therefore, by (13),

$$
4 \mathfrak { H } _ { \alpha } \left( \widehat { V } _ { 3 } ^ { \mathrm { d i s t } } \right) \leq 4 \sqrt { B ^ { 2 } \left( \log \frac { 1 } { \alpha } + 5 L _ { 4 } \right) } \leq 1 0 B L _ { 4 } ,
$$

where we used $\log ( 1 / \alpha ) \leq L _ { 4 }$ and $L _ { 4 } \geq 1$ . Now,

$$
\widehat { U } _ { 4 } ^ { \mathrm { d i s t } } ( \alpha ) = \frac { 1 } { A _ { 3 } } \left[ R _ { 0 } + A _ { 3 } \eta _ { 3 } ^ { 2 } \left. g _ { 3 } \right. ^ { 2 } + 4 \mathfrak { H } _ { \alpha } \left( \widehat { V } _ { 3 } ^ { \mathrm { d i s t } } \right) \right] .
$$

Using $A _ { 3 } = 3 , \eta _ { 3 } = 1 / ( 3 \mu ) , R _ { 0 } \le G ^ { 2 } / \mu ^ { 2 }$ , and $\| g _ { 3 } \| \leq G ,$ we obtain

$$
\widehat { U } _ { 4 } ^ { \mathrm { d i s t } } ( \alpha ) \leq \frac { 1 } { 3 } \left[ \frac { G ^ { 2 } } { \mu ^ { 2 } } + \frac { G ^ { 2 } } { 3 \mu ^ { 2 } } + 1 0 B L _ { 4 } \right] .
$$

Because $B \geq G ^ { 2 } / \mu ^ { 2 }$ and $L _ { 4 } \geq 1$ , this yields

$$
\widehat { U } _ { 4 } ^ { \mathrm { d i s t } } ( \alpha ) \leq 4 B L _ { 4 } \leq K B \frac { L _ { 4 } } { 4 } ,
$$

since $K = 4 0 9 6 \geq 1 6$ . This proves the base case.

Now fix $t \geq 4$ and assume that

$$
\widehat { U } _ { s } ^ { \mathrm { d i s t } } ( \alpha ) \leq K B \frac { L _ { s } } { s } , \qquad s = 4 , \dots , t .
$$

We prove the same bound for $s = t + 1$ . By the recursive definition of the observable intrinsic time,

$$
\widehat { V } _ { t } ^ { \mathrm { d i s t } } = \sigma ^ { 2 } A _ { 3 } ^ { 2 } \eta _ { 3 } ^ { 2 } R _ { 0 } + \sum _ { s = 4 } ^ { t } \sigma ^ { 2 } A _ { s } ^ { 2 } \eta _ { s } ^ { 2 } \widehat { U } _ { s } ^ { \mathrm { d i s t } } ( \alpha ) .
$$

As shown above, the initial term satisfies $\sigma ^ { 2 } A _ { 3 } ^ { 2 } \eta _ { 3 } ^ { 2 } R _ { 0 } \leq B ^ { 2 }$ . For the summation term, we have

$$
A _ { s } ^ { 2 } \eta _ { s } ^ { 2 } = \left( \frac { s ( s - 1 ) } 2 \right) ^ { 2 } \frac 1 { \mu ^ { 2 } s ^ { 2 } } = \frac { ( s - 1 ) ^ { 2 } } { 4 \mu ^ { 2 } } \le \frac { s ^ { 2 } } { 4 \mu ^ { 2 } } .
$$

Using the induction hypothesis,

$$
\sum _ { s = 4 } ^ { t } \sigma ^ { 2 } A _ { s } ^ { 2 } \eta _ { s } ^ { 2 } \widehat { U } _ { s } ^ { \mathrm { d i s t } } ( \alpha ) \leq \frac { K \sigma ^ { 2 } B } { 4 \mu ^ { 2 } } \sum _ { s = 4 } ^ { t } s L _ { s } \leq \frac { K \sigma ^ { 2 } B } { 4 \mu ^ { 2 } } t ^ { 2 } L _ { t } \leq \frac { K } { 4 } B ^ { 2 } t ^ { 2 } L _ { t } ,
$$

where we used the monotonicity of $L _ { s }$ and the fact that $\sigma ^ { 2 } / \mu ^ { 2 } \le B$ . Since $t \geq 4 , B \geq 1 , L _ { t } \geq 1$ , and $K \geq 1$ , we also have

$$
B ^ { 2 } \leq K B ^ { 2 } t ^ { 2 } L _ { t } \qquad \mathrm { a n d } \qquad 1 \leq K B ^ { 2 } t ^ { 2 } L _ { t } .
$$

Therefore the initial term and the efective truncation by 1 are both absorbed into the same scale, and

$$
\widehat { V } _ { t , \mathrm { e f f } } ^ { \mathrm { d i s t } } \leq K B ^ { 2 } t ^ { 2 } L _ { t } .\tag{80}
$$

We next control the stitched logarithmic factor. From (80) and $K = 4 0 9 6 = 2 ^ { 1 2 }$ ，

$$
\widehat { m } _ { t } ^ { \mathrm { d i s t } } = \left\lceil \log _ { 2 } \widehat { V } _ { t , \mathrm { e f f } } ^ { \mathrm { d i s t } } \right\rceil \leq 1 3 + 2 \log _ { 2 } B + 2 \log _ { 2 } t + \log _ { 2 } L _ { t } .
$$

Therefore, $\widehat { m } _ { t } ^ { \mathrm { d i s t } } + 2 \leq 1 6 + 2 \log _ { 2 } B + 2 \log _ { 2 } t + \log _ { 2 } L _ { t } . \mathrm { S i n c e } \log _ { 2 } y \leq 2 \log y \mathrm { ~ f o r ~ } y \geq 1$

$$
\widehat { m } _ { t } ^ { \mathrm { d i s t } } + 2 \leq 1 6 + 4 \log B + 4 \log t + 2 \log L _ { t } .
$$

The right-hand side is bounded by $1 6 ( 1 + \log B ) ( 1 + \log t ) ( 1 + \log L _ { t } )$ . Thus,

$$
\begin{array} { r l } & { \log \bigl ( ( \widehat { m } _ { t } ^ { \mathrm { d i s t } } + 1 ) ( \widehat { m } _ { t } ^ { \mathrm { d i s t } } + 2 ) \bigr ) \leq 2 \log ( \widehat { m } _ { t } ^ { \mathrm { d i s t } } + 2 ) } \\ & { \qquad \leq 2 \log 1 6 + 2 \log ( 1 + \log B ) + 2 \log ( 1 + \log t ) + 2 \log ( 1 + \log L _ { t } ) . } \end{array}
$$

Each term on the right is controlled by $L _ { t }$ . Indeed,

$$
\log ( 1 + \log B ) = \log \log ( e B ) \leq L _ { t } , \qquad \log ( 1 + \log t ) = \log \log ( e t ) \leq L _ { t } ,
$$

and, since $L _ { t } \geq 1$ , we have $\log ( 1 + \log L _ { t } ) \leq L _ { t }$ . Also $2 \log { 1 6 } \leq 6 L _ { t }$ . Hence

$$
\log \ ( ( \widehat { m } _ { t } ^ { \mathrm { d i s t } } + 1 ) ( \widehat { m } _ { t } ^ { \mathrm { d i s t } } + 2 ) ) \leq 1 2 L _ { t } .
$$

Since $\log ( 1 / \alpha ) \leq L _ { t }$ , we have

$$
\log \frac { 1 } { \alpha } + \log \left( ( \widehat { m } _ { t } ^ { \mathrm { d i s t } } + 1 ) ( \widehat { m } _ { t } ^ { \mathrm { d i s t } } + 2 ) \right) \leq 1 3 L _ { t } .
$$

Combining this with (80), we obtain

$$
4 \tilde { \Phi } _ { \alpha } \left( \widehat { V } _ { t } ^ { \mathrm { d i s t } } \right) = 4 \sqrt { \hat { V } _ { t , \mathrm { e f f } } ^ { \mathrm { d i s t } } \left( \log \frac { 1 } { \alpha } + \log ( ( \hat { m } _ { t } ^ { \mathrm { d i s t } } + 1 ) ( \hat { m } _ { t } ^ { \mathrm { d i s t } } + 2 ) ) \right) } \leq 4 \sqrt { K B ^ { 2 } t ^ { 2 } L _ { t } \cdot 1 3 L _ { t } } = 4 \sqrt { 1 3 K } B t L _ { t }
$$

Dividing by $A _ { t } = t ( t - 1 ) / 2$ gives

$$
\frac { 4 \Im _ { \alpha } \left( \widehat { V } _ { t } ^ { \mathrm { d i s t } } \right) } { A _ { t } } \le 3 0 \sqrt { K } B \frac { L _ { t } } { t - 1 } .
$$

For $t \geq 4$ , we have $1 / ( t - 1 ) \leq 5 / ( 3 ( t + 1 ) )$ , and therefore

$$
\frac { 4 \Im _ { \alpha } \left( \widehat { V } _ { t } ^ { \mathrm { d i s t } } \right) } { A _ { t } } \le 5 0 \sqrt { K } B \frac { L _ { t + 1 } } { t + 1 } ,\tag{81}
$$

where we also used $L _ { t } \leq L _ { t + 1 }$

Finally, by definition,

$$
\widehat { U } _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha ) = D _ { t } + \frac { 4 \mathfrak { H } _ { \alpha } \left( \widehat { V } _ { t } ^ { \mathrm { d i s t } } \right) } { A _ { t } } .
$$

Using (79) and (81), we get

$$
\widehat { U } _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha ) \leq \left( 2 + 5 0 \sqrt { K } \right) B \frac { L _ { t + 1 } } { t + 1 } .
$$

Since $K = 4 0 9 6 , \sqrt { K } = 6 4$ , and $2 + 5 0 { \sqrt { K } } = 3 2 0 2 \leq 4 0 9 6 = K$ , we conclude that

$$
\widehat { U } _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha ) \leq K B \frac { L _ { t + 1 } } { t + 1 } .
$$

This closes the induction. Substituting K = 4096 gives the claimed bound.

## A.1.5 Proof of Proposition 2.7

Proof. By Lemma C.3, for every $t \geq t _ { 0 }$

$$
f ( \bar { x } _ { t } ) - f ( x ^ { \star } ) \leq \frac { 1 } { \mu S _ { t } } \sum _ { s = t _ { 0 } } ^ { t } \| g _ { s } \| ^ { 2 } + \frac { \mu t _ { 0 } ( t _ { 0 } - 1 ) } { 4 S _ { t } } Z _ { t _ { 0 } } + \frac { 1 } { S _ { t } } \bar { E } _ { t } .
$$

It remains to control $\bar { E } _ { t }$ uniformly over time. Recall that

$$
E _ { t } = t \langle x ^ { \star } - x _ { t } , \xi _ { t } \rangle .
$$

Since $x _ { t }$ is $\mathcal { F } _ { t - 1 }$ -measurable, applying Assumption 1.3(ii) with the predictable direction $t ( x ^ { \star } - x _ { t } )$ gives

$$
\mathbb { E } \left[ e ^ { \lambda E _ { t } } \Big | \mathcal { F } _ { t - 1 } \right] \leq \exp \left( \frac { \lambda ^ { 2 } \sigma ^ { 2 } t ^ { 2 } Z _ { t } } { 2 } \right)
$$

for every $\lambda \geq 0$ . Thus, the increments $E _ { t } = \bar { E } _ { t } - \bar { E } _ { t - 1 }$ satisfy the hypotheses of Lemma 2.1 with

$$
v _ { t } ^ { \mathrm { s u b } } = \sigma ^ { 2 } t ^ { 2 } Z _ { t } , \qquad \gamma = - 1 .
$$

Applying Lemma 2.1 with $\bar { E } _ { t }$ in place of ${ \bar { X } } _ { t }$ and $V _ { t } ^ { \mathrm { s u b } }$ in place of V<sub>t</sub> yields

$$
\begin{array} { r } { \mathbb { P } \left( \forall t \geq t _ { 0 } : \ \bar { E } _ { t } \leq 2 \Im _ { \alpha } \left( V _ { t } ^ { \mathrm { s u b } } \right) \right) \geq 1 - \alpha . } \end{array}
$$

Substituting this bound into the preceding decomposition and using the definition of $U _ { t } ^ { \mathrm { s u b } } ( \alpha )$ in (21) proves the result. □

## A.1.6 Proof of Theorem 2.8

We first note that, for each $t \geq t _ { 0 }$ , the quantity $\widehat { U } _ { t } ^ { \mathrm { s u b } } ( \alpha )$ depends only on $R _ { 0 }$ , the observed trajectory up to time $t ,$ and the recursive observable bounds

$$
\widehat { U } _ { t _ { 0 } + 1 } ^ { \mathrm { d i s t } } ( \alpha / 2 ) , \ldots , \widehat { U } _ { t } ^ { \mathrm { d i s t } } ( \alpha / 2 ) .
$$

Hence $\widehat { U } _ { t } ^ { \mathrm { s u b } } ( \alpha )$ is fully observable. Now let $U _ { t } ^ { \mathrm { s u b } } ( \alpha / 2 )$ be the adaptive confidence sequence from Proposition 2.7. Set

$$
E _ { 1 } : = \Big \{ \forall t \ge t _ { 0 } : ~ f ( \bar { x } _ { t } ) - f ( x ^ { \star } ) \le U _ { t } ^ { \mathrm { s u b } } ( \alpha / 2 ) \Big \} .
$$

By Proposition 2.7, $\mathbb { P } ( E _ { 1 } ) \ge 1 - \alpha / 2$ . Also set

$$
E _ { 2 } : = \Big \{ \forall t \geq t _ { 0 } : ~ Z _ { t + 1 } \leq \widehat { U } _ { t + 1 } ^ { \mathrm { d i s t } } ( \alpha / 2 ) \Big \} .
$$

By Theorem 2.3, $\mathbb { P } ( E _ { 2 } ) \ge 1 - \alpha / 2$ . We now show that on $E _ { 2 } , U _ { t } ^ { \mathrm { s u b } } ( \alpha / 2 ) \le \widehat { U } _ { t } ^ { \mathrm { s u b } } ( \alpha )$ for all $t \geq t _ { 0 }$ Indeed, on $E _ { 2 }$ , we have $Z _ { s } \le \widehat { U } _ { s } ^ { \mathrm { d i s t } } ( \alpha / 2 )$ for every $s \geq t _ { 0 } + 1$ , and also $Z _ { t _ { 0 } } \leq R _ { 0 }$ b<sub>.</sub> <sub>Therefore,</sub>

$$
V _ { t } ^ { \mathrm { s u b } } = \sigma ^ { 2 } t _ { 0 } ^ { 2 } Z _ { t _ { 0 } } + \sum _ { s = t _ { 0 } + 1 } ^ { t } \sigma ^ { 2 } s ^ { 2 } Z _ { s } \leq \sigma ^ { 2 } t _ { 0 } ^ { 2 } R _ { 0 } + \sum _ { s = t _ { 0 } + 1 } ^ { t } \sigma ^ { 2 } s ^ { 2 } \widehat { U } _ { s } ^ { \mathrm { d i s t } } ( \alpha / 2 ) = \widehat { V } _ { t } ^ { \mathrm { s u b } } ( \alpha / 2 ) .
$$

Since ${ \mathfrak { H } } _ { \alpha / 2 }$ is nondecreasing, it follows that

$$
2 \mathfrak { H } _ { \alpha / 2 } \left( V _ { t } ^ { \mathrm { s u b } } \right) \le 2 \mathfrak { H } _ { \alpha / 2 } \left( \widehat { V } _ { t } ^ { \mathrm { s u b } } ( \alpha / 2 ) \right) .
$$

Using also $Z _ { t _ { 0 } } \leq R _ { 0 }$ , we obtain

$$
U _ { t } ^ { \mathrm { s u b } } ( \alpha / 2 ) \leq \frac { 1 } { \mu S _ { t } } \sum _ { s = t _ { 0 } } ^ { t } \| g _ { s } \| ^ { 2 } + \frac { \mu t _ { 0 } ( t _ { 0 } - 1 ) } { 4 S _ { t } } R _ { 0 } + \frac { 2 } { S _ { t } } \mathfrak { H } _ { \alpha / 2 } \left( \widehat { V } _ { t } ^ { \mathrm { s u b } } ( \alpha / 2 ) \right) = \widehat { U } _ { t } ^ { \mathrm { s u b } } ( \alpha ) .
$$

Therefore, on $E _ { 1 } \cap E _ { 2 }$ ，

$$
f ( \bar { x } _ { t } ) - f ( x ^ { \star } ) \leq U _ { t } ^ { \mathrm { s u b } } ( \alpha / 2 ) \leq \widehat { U } _ { t } ^ { \mathrm { s u b } } ( \alpha ) \qquad \mathrm { f o r ~ a l l ~ } t \geq t _ { 0 } .
$$

Hence

$$
\begin{array} { r } { \mathbb { P } \left( \forall t \geq t _ { 0 } : \ f ( \bar { x } _ { t } ) - f ( x ^ { \star } ) \leq \widehat { U } _ { t } ^ { \mathrm { s u b } } ( \alpha ) \right) \geq \mathbb { P } ( E _ { 1 } \cap E _ { 2 } ) \geq 1 - \alpha , } \end{array}
$$

where the last step follows from the union bound. This completes the proof.

## A.1.7 Proof of Proposition 2.9

Throughout the proof, $C < \infty$ denotes a universal constant whose value may change from line to line. Write $L _ { t } ^ { \mathrm { s u b } } : = L _ { t } ^ { \mathrm { s u b } } ( \alpha )$ . The sequence $L _ { t } ^ { \mathrm { s u b } }$ is nondecreasing and satisfies $L _ { t } ^ { \mathrm { s u b } } \geq 1$ . For convenience, throughout the proof define

$$
\widehat { V } _ { t , \mathrm { e f f } } ^ { \mathrm { s u b } } ( \alpha / 2 ) : = \operatorname* { m a x } \left\{ \widehat { V } _ { t } ^ { \mathrm { s u b } } ( \alpha / 2 ) , 1 \right\} , \qquad \widehat { m } _ { t } ^ { \mathrm { s u b } } ( \alpha / 2 ) : = m \left( \widehat { V } _ { t } ^ { \mathrm { s u b } } ( \alpha / 2 ) \right) = \left\lceil \log _ { 2 } \widehat { V } _ { t , \mathrm { e f f } } ^ { \mathrm { s u b } } ( \alpha / 2 ) \right\rceil .
$$

For every $t \geq 4$

$$
S _ { t } = \sum _ { s = 4 } ^ { t } s = \frac { t ( t + 1 ) } { 2 } - 6 \geq \frac { t ^ { 2 } } { 4 } .
$$

By definition of the observable weighted-average suboptimality bound,

$$
\widehat { U } _ { t } ^ { \mathrm { s u b } } ( \alpha ) = \frac { 1 } { S _ { t } } \left( \frac 1 \mu \sum _ { s = 4 } ^ { t } \Vert g _ { s } \Vert ^ { 2 } + 3 \mu R _ { 0 } + 2 \mathfrak { H } _ { \alpha / 2 } \left( \widehat { V } _ { t } ^ { \mathrm { s u b } } ( \alpha / 2 ) \right) \right) , \qquad t \ge 4 ,
$$

where we used $\mu t _ { 0 } ( t _ { 0 } - 1 ) / 4 = 3 \mu$

We first bound the deterministic terms. Since $\| g _ { s } \| \leq G$ and, by the standing initialization, $R _ { 0 } \leq$ $G ^ { 2 } / \mu ^ { 2 }$ , we have

$$
\frac { 1 } { S _ { t } } \frac { 1 } { \mu } \sum _ { s = 4 } ^ { t } \| g _ { s } \| ^ { 2 } \leq \frac { G ^ { 2 } } { \mu } \frac { t } { S _ { t } } \leq 4 \frac { G ^ { 2 } } { \mu } \frac { 1 } { t } ,
$$

and

$$
\frac { 3 \mu R _ { 0 } } { S _ { t } } \leq 3 \frac { G ^ { 2 } } { \mu } \frac { 1 } { S _ { t } } \leq 1 2 \frac { G ^ { 2 } } { \mu } \frac { 1 } { t ^ { 2 } } \leq 1 2 \frac { G ^ { 2 } } { \mu } \frac { 1 } { t } .
$$

Since $B _ { \mathrm { s u b } } \geq G ^ { 2 } / \mu$ and $L _ { t } ^ { \mathrm { s u b } } \geq 1$ , it follows that

$$
\frac { 1 } { S _ { t } } \left( \frac { 1 } { \mu } \sum _ { s = 4 } ^ { t } \| g _ { s } \| ^ { 2 } + 3 \mu R _ { 0 } \right) \leq C B _ { \mathrm { s u b } } \frac { L _ { t } ^ { \mathrm { s u b } } } { t } .\tag{82}
$$

It remains to control the observable boundary term. By Remark 2.6, applied with confidence level $\alpha / 2$ , there exists a universal constant $C < \infty$ such that, for every $s \geq 5$

$$
\widehat { U } _ { s } ^ { \mathrm { d i s t } } ( \alpha / 2 ) \le C B \frac { L _ { s } ( \alpha / 2 ) } { s } ,
$$

where $L _ { s } ( \cdot )$ is the logarithmic factor from Proposition 2.4. Since

$$
L _ { s } ( \alpha / 2 ) = L _ { s } ( \alpha ) + \log 2 \leq C L _ { s } ^ { \mathrm { s u b } } ( \alpha ) ,
$$

we obtain

$$
\begin{array} { r } { \widehat { U } _ { s } ^ { \mathrm { d i s t } } ( \alpha / 2 ) \le C B \frac { L _ { s } ^ { \mathrm { s u b } } } { s } , \qquad s \ge 5 . } \end{array}
$$

Using the definition of the observable intrinsic time, we get

$$
\begin{array} { r } { \widehat { V } _ { t } ^ { \mathrm { s u b } } ( \alpha / 2 ) = 1 6 \sigma ^ { 2 } R _ { 0 } + \displaystyle \sum _ { s = 5 } ^ { t } \sigma ^ { 2 } s ^ { 2 } \widehat { U } _ { s } ^ { \mathrm { d i s t } } ( \alpha / 2 ) \le 1 6 \sigma ^ { 2 } \displaystyle \frac { G ^ { 2 } } { \mu ^ { 2 } } + C \sigma ^ { 2 } B \displaystyle \sum _ { s = 5 } ^ { t } s L _ { s } ^ { \mathrm { s u b } } } \\ { \le 1 6 \sigma ^ { 2 } \displaystyle \frac { G ^ { 2 } } { \mu ^ { 2 } } + C \sigma ^ { 2 } B t ^ { 2 } L _ { t } ^ { \mathrm { s u b } } . } \end{array}
$$

The first term is bounded by $C \mu ^ { 2 } B ^ { 2 }$ , because

$$
\sigma ^ { 2 } \frac { G ^ { 2 } } { \mu ^ { 2 } } = \mu ^ { 2 } \left( \frac { \sigma ^ { 2 } } { \mu ^ { 2 } } \right) \left( \frac { G ^ { 2 } } { \mu ^ { 2 } } \right) \leq \mu ^ { 2 } B ^ { 2 } .
$$

The second term is bounded by $C \mu ^ { 2 } B ^ { 2 } t ^ { 2 } L _ { t } ^ { \mathrm { s u b } }$ , since

$$
\sigma ^ { 2 } B = \mu ^ { 2 } \left( { \frac { \sigma ^ { 2 } } { \mu ^ { 2 } } } \right) B \leq \mu ^ { 2 } B ^ { 2 } .
$$

Therefore, using $t \geq 4$ and $L _ { t } ^ { \mathrm { s u b } } \geq 1$ 2

$$
\widehat { V } _ { t } ^ { \mathrm { s u b } } ( \alpha / 2 ) \le C \mu ^ { 2 } B ^ { 2 } t ^ { 2 } L _ { t } ^ { \mathrm { s u b } } .
$$

Since $B _ { \mathrm { s u b } } = 1 + \mu B$ , we have $\mu B \le B _ { \mathrm { s u b } }$ and $1 \leq B _ { \mathrm { s u b } }$ . Hence

$$
\mu ^ { 2 } B ^ { 2 } t ^ { 2 } L _ { t } ^ { \mathrm { s u b } } \leq B _ { \mathrm { s u b } } ^ { 2 } t ^ { 2 } L _ { t } ^ { \mathrm { s u b } } , \qquad 1 \leq B _ { \mathrm { s u b } } ^ { 2 } t ^ { 2 } L _ { t } ^ { \mathrm { s u b } } .
$$

Thus the efective intrinsic time satisfies

$$
\begin{array} { r } { \widehat { V } _ { t , \mathrm { e f f } } ^ { \mathrm { s u b } } ( \alpha / 2 ) \leq C B _ { \mathrm { s u b } } ^ { 2 } t ^ { 2 } L _ { t } ^ { \mathrm { s u b } } . } \end{array}\tag{83}
$$

Therefore,

$$
\widehat { m } _ { t } ^ { \mathrm { s u b } } ( { \alpha } / { 2 } ) \leq C + 2 \log _ { 2 } B _ { \mathrm { s u b } } + 2 \log _ { 2 } t + \log _ { 2 } L _ { t } ^ { \mathrm { s u b } } .
$$

The same elementary logarithmic bound used in the proof of Proposition 2.4 then gives

$$
\log \Big ( \big ( \widehat { m } _ { t } ^ { \mathrm { s u b } } ( \alpha / 2 ) + 1 \big ) \big ( \widehat { m } _ { t } ^ { \mathrm { s u b } } ( \alpha / 2 ) + 2 \big ) \Big ) \leq C L _ { t } ^ { \mathrm { s u b } } .
$$

The logarithmic factor on the left is bounded in this way because the preceding display introduces only iterated logarithms of t, $B _ { \mathrm { s u b } }$ , and $L _ { t } ^ { \mathrm { s u b } }$ , all of which are absorbed by the definition of $L _ { t } ^ { \mathrm { s u b . } }$ ; the same definition also gives $\log ( 2 / \alpha ) \leq C L _ { t } ^ { \mathrm { s u b } }$ . Using the definition of ${ \mathfrak { H } } _ { \alpha / 2 }$ and (83), we obtain

$$
2 \mathfrak { H } _ { \alpha / 2 } \left( \widehat { V } _ { t } ^ { \mathrm { s u b } } ( \alpha / 2 ) \right) \le 2 \sqrt { C B _ { \mathrm { s u b } } ^ { 2 } t ^ { 2 } L _ { t } ^ { \mathrm { s u b } } \cdot C L _ { t } ^ { \mathrm { s u b } } } \le C B _ { \mathrm { s u b } } t L _ { t } ^ { \mathrm { s u b } } .
$$

Dividing by $S _ { t } \geq t ^ { 2 } / 4$ gives

$$
\frac { 2 \mathfrak { H } _ { \alpha / 2 } \left( \widehat { V } _ { t } ^ { \mathrm { s u b } } ( \alpha / 2 ) \right) } { S _ { t } } \leq C B _ { \mathrm { s u b } } \frac { L _ { t } ^ { \mathrm { s u b } } } { t } .\tag{84}
$$

Combining (82) and (84), we conclude that

$$
\widehat { U } _ { t } ^ { \mathrm { s u b } } ( \alpha ) \leq C B _ { \mathrm { s u b } } \frac { L _ { t } ^ { \mathrm { s u b } } } { t } , \qquad t \geq 4 ,
$$

almost surely. This proves the proposition.

## A.2 Proofs of Section 3

## A.2.1 Proof of Lemma 3.2

The claim is immediate for $\theta = 0 .$ , so fix $\theta \in ( 0 , 1 )$ . For $x \geq - 1$ , define

$$
h _ { \theta } ( x ) : = \log ( 1 + \theta x ) - \theta x + \psi _ { \mathrm { E } } ( \theta ) x ^ { 2 } .
$$

The logarithm is well defined because $1 + \theta x \geq 1 - \theta > 0$ , and (26) is equivalent to $h _ { \theta } ( x ) \ \geq \ 0$ Diferentiating gives

$$
h _ { \theta } ^ { \prime } ( x ) = x \left( 2 \psi _ { \mathrm { E } } ( \theta ) - \frac { \theta ^ { 2 } } { 1 + \theta x } \right) .\tag{85}
$$

The power-series representation

$$
\psi _ { \mathrm { E } } ( \theta ) = \sum _ { m = 2 } ^ { \infty } { \frac { \theta ^ { m } } { m } }
$$

implies

$$
{ \frac { \theta ^ { 2 } } { 2 } } \leq \psi _ { \mathrm { E } } ( \theta ) \leq { \frac { \theta ^ { 2 } } { 2 ( 1 - \theta ) } } .\tag{86}
$$

The upper bound is (27).

For $x \geq 0$ , the term in parentheses in (85) is at least $2 \psi _ { \mathrm { E } } ( \theta ) - \theta ^ { 2 } \ge 0$ . Thus $h _ { \theta }$ is nondecreasing on $[ 0 , \infty )$ , and $h _ { \theta } ( x ) \geq h _ { \theta } ( 0 ) = 0$ there. For $x \in [ - 1 , 0 ]$ , set

$$
q _ { \theta } ( x ) : = 2 \psi _ { \mathrm { E } } ( \theta ) - \frac { \theta ^ { 2 } } { 1 + \theta x } .
$$

The function $q _ { \theta }$ is nondecreasing, since

$$
q _ { \theta } ^ { \prime } ( x ) = \frac { \theta ^ { 3 } } { ( 1 + \theta x ) ^ { 2 } } > 0 .
$$

By (86), $q _ { \theta } ( - 1 ) \leq 0$ and $q _ { \theta } ( 0 ) \geq 0$ . Hence $q _ { \theta }$ changes sign at most once, from nonpositive to nonnegative. Since $x \le 0$ , equation (85) shows that $h _ { \theta }$ first increases and then decreases on $[ - 1 , 0 ]$ . Its minimum is therefore attained at an endpoint. Finally,

$$
h _ { \theta } ( - 1 ) = \log ( 1 - \theta ) + \theta + \psi _ { \mathrm { E } } ( \theta ) = 0 , \qquad h _ { \theta } ( 0 ) = 0 .
$$

This proves (26).

For the conditional statement, let $m : = \mathbb { E } [ X \mid { \mathcal { G } } ]$ . Apply (26) with $\theta = \lambda b$ and $x = X / b$ to obtain

$$
\exp \left( \lambda X - \frac { \psi _ { \mathrm { E } } ( \lambda b ) } { b ^ { 2 } } X ^ { 2 } \right) \leq 1 + \lambda X .
$$

Taking conditional expectations and multiplying by the $\mathcal { G }$ -measurable factor $e ^ { - \lambda m }$ gives

$$
\mathbb { E } \left[ \exp \left( \lambda ( X - m ) - \frac { \psi _ { \mathrm { E } } ( \lambda b ) } { b ^ { 2 } } X ^ { 2 } \right) \Big | \mathscr { G } \right] \leq e ^ { - \lambda m } ( 1 + \lambda m ) \leq 1 .
$$

Here $\lambda m \ge - \lambda b > - 1$ , and the last inequality follows from $( 1 + y ) e ^ { - y } \leq 1$ for $y > - 1$ . By (27),

$$
\frac { \psi _ { \mathrm { E } } ( \lambda b ) } { b ^ { 2 } } \leq \frac { \lambda ^ { 2 } } { 2 ( 1 - \lambda b ) } .
$$

Replacing the quadratic coeficient by the larger quantity on the right decreases the exponential integrand and proves (28).

## A.2.2 Proof of Theorem 3.3

For $k , \ell \geq 0 ,$ , define

$$
\overline { { { v } } } _ { k } : = 2 ^ { k } , \qquad \overline { { { b } } } _ { \ell } : = 2 ^ { \ell } .
$$

We stitch over the dyadic intervals

$$
I _ { 0 } ^ { \mathrm { v } } : = [ 0 , 1 ] , \qquad I _ { k } ^ { \mathrm { v } } : = ( 2 ^ { k - 1 } , 2 ^ { k } ] , \qquad k \geq 1 ,
$$

and

$$
I _ { 0 } ^ { \mathrm { b } } : = [ 0 , 1 ] , \qquad I _ { \ell } ^ { \mathrm { b } } : = ( 2 ^ { \ell - 1 } , 2 ^ { \ell } ] , \qquad \ell \geq 1 .
$$

For every $k , \ell \geq 0 ,$ , set

$$
\alpha _ { k , \ell } : = \frac { \alpha } { ( k + 1 ) ( k + 2 ) ( \ell + 1 ) ( \ell + 2 ) }
$$

and

$$
d _ { k , \ell } : = \log \frac { 1 } { \alpha _ { k , \ell } } = \log \frac { 1 } { \alpha } + \log \bigl ( ( k + 1 ) ( k + 2 ) \bigr ) + \log \bigl ( ( \ell + 1 ) ( \ell + 2 ) \bigr ) .
$$

Since

$$
\sum _ { j = 0 } ^ { \infty } { \frac { 1 } { ( j + 1 ) ( j + 2 ) } } = 1 ,
$$

we have

$$
\sum _ { k , \ell \geq 0 } \alpha _ { k , \ell } = \alpha .
$$

For each pair $( k , \ell )$ , define

$$
q _ { k , \ell } : = \sqrt { \frac { 2 d _ { k , \ell } } { \overline { { v } } _ { k } } } , ~ \lambda _ { k , \ell } : = \frac { q _ { k , \ell } } { 1 + \overline { { b } } _ { \ell } q _ { k , \ell } } .\tag{87}
$$

Then

$$
\begin{array} { r } { 0 < \lambda _ { k , \ell } \overline { { b } } _ { \ell } < 1 . } \end{array}
$$

Fix $k , \ell \geq 0 .$ . Since $c _ { s }$ is $\mathcal { F } _ { s - 1 }$ -measurable for every s, the running maximum $C _ { s } = \operatorname* { m a x } _ { t _ { 0 } \leq r \leq s } c _ { r }$ is also <sub>s−1</sub>-measurable. Consequently,

$$
J _ { s } ^ { ( \ell ) } : = \mathbb { 1 } \{ C _ { s } \leq \overline { { b } } _ { \ell } \}
$$

is predictable. Define

$$
\overline { { H } } _ { t _ { 0 } - 1 } ^ { ( \ell ) } : = 0 , \qquad \overline { { H } } _ { t } ^ { ( \ell ) } : = \sum _ { s = t _ { 0 } } ^ { t } J _ { s } ^ { ( \ell ) } \left( H _ { s } - \mathbb { E } [ H _ { s } \mid \mathcal { F } _ { s - 1 } ] \right) ,
$$

and

$$
W _ { t _ { 0 } - 1 } ^ { ( \ell ) } : = 0 , \qquad W _ { t } ^ { ( \ell ) } : = \sum _ { s = t _ { 0 } } ^ { t } J _ { s } ^ { ( \ell ) } H _ { s } ^ { 2 } .
$$

For $t \geq t _ { 0 } - 1$ , let

$$
M _ { t } ^ { k , \ell } : = \exp \left( \lambda _ { k , \ell } \overline { { { H } } } _ { t } ^ { ( \ell ) } - \frac { \lambda _ { k , \ell } ^ { 2 } } { 2 ( 1 - \lambda _ { k , \ell } \overline { { { b } } } _ { \ell } ) } W _ { t } ^ { ( \ell ) } \right) .
$$

If $J _ { s } ^ { ( \ell ) } = 1$ , then

$$
\begin{array} { r } { | H _ { s } | \le c _ { s } \le C _ { s } \le \overline { { b } } _ { \ell } , } \end{array}
$$

whereas $J _ { s } ^ { ( \ell ) } H _ { s } = 0$ if $J _ { s } ^ { ( \ell ) } = 0$ . Hence

$$
| J _ { s } ^ { ( \ell ) } H _ { s } | \leq \overline { { b } } _ { \ell } .
$$

Moreover, predictability gives

$$
\begin{array} { r } { \mathbb E [ J _ { s } ^ { ( \ell ) } H _ { s } \mid \mathcal F _ { s - 1 } ] = J _ { s } ^ { ( \ell ) } \mathbb E [ H _ { s } \mid \mathcal F _ { s - 1 } ] . } \end{array}
$$

Applying Lemma 3.2 conditionally with $X = J _ { s } ^ { ( \ell ) } H _ { s } , b = \overline { { b } } _ { \ell } .$ , and $\lambda = \lambda _ { k , \ell }$ yields

$$
\mathbb { E } [ M _ { s } ^ { k , \ell } \mid \mathcal { F } _ { s - 1 } ] \le M _ { s - 1 } ^ { k , \ell } .
$$

Thus $\{ M _ { t } ^ { k , \ell } \} _ { t \geq t _ { 0 } - 1 }$ is a nonnegative supermartingale with $M _ { t _ { 0 } - 1 } ^ { k , \ell } = 1$ . By Ville’s inequality,

$$
\mathbb { P } \Bigg ( \exists t \geq t _ { 0 } : \ \overline { { H } } _ { t } ^ { ( \ell ) } \geq \frac { d _ { k , \ell } } { \lambda _ { k , \ell } } + \frac { \lambda _ { k , \ell } } { 2 ( 1 - \lambda _ { k , \ell } \overline { { b } } _ { \ell } ) } W _ { t } ^ { ( \ell ) } \Bigg ) \leq \alpha _ { k , \ell } .\tag{88}
$$

We now stitch these fixed-λ bounds over the dyadic ranges of $W _ { t }$ and $C _ { t }$ . Because $C _ { t }$ is nondecreasing, $C _ { t } \leq \overline { { b } } _ { \ell }$ implies

$$
J _ { s } ^ { ( \ell ) } = 1 \qquad \mathrm { f o r ~ e v e r y ~ } t _ { 0 } \leq s \leq t .
$$

Therefore,

$$
\overline { { { H } } } _ { t } ^ { ( \ell ) } = \overline { { { H } } } _ { t } , \qquad W _ { t } ^ { ( \ell ) } = W _ { t }
$$

at every time t satisfying $C _ { t } \leq \overline { { b } } _ { \ell }$ . Taking a union bound in (88) over all $k , \ell \geq 0$ , with probability at least

$$
1 - \sum _ { k , \ell \geq 0 } \alpha _ { k , \ell } = 1 - \alpha ,
$$

we have

$$
\overline { { H } } _ { t } < \frac { d _ { k , \ell } } { \lambda _ { k , \ell } } + \frac { \lambda _ { k , \ell } } { 2 ( 1 - \lambda _ { k , \ell } \overline { { b } } _ { \ell } ) } W _ { t }\tag{89}
$$

simultaneously for every $k , \ell \geq 0$ and every $t \geq t _ { 0 }$ such that $C _ { t } \leq \overline { { b } } _ { \ell }$

Work on this event and fix $t \geq t _ { 0 }$ . Let

$$
W _ { t , \mathrm { e f f } } : = \operatorname* { m a x } \{ W _ { t } , 1 \} , \qquad C _ { t , \mathrm { e f f } } : = \operatorname* { m a x } \{ C _ { t } , 1 \} ,
$$

and let

$$
k _ { t } : = k ( W _ { t } ) , \qquad \ell _ { t } : = \ell ( C _ { t } ) .
$$

Then

$$
W _ { t } \in I _ { k _ { t } } ^ { \mathrm { v } } , \qquad C _ { t } \in I _ { \ell _ { t } } ^ { \mathrm { b } } ,
$$

and therefore

$$
W _ { t } \le \overline { { v } } _ { k _ { t } } \le 2 W _ { t , \mathrm { e f f } } , \qquad C _ { t } \le \overline { { b } } _ { \ell _ { t } } \le 2 C _ { t , \mathrm { e f f } } .
$$

Moreover,

$$
d _ { k _ { t } , \ell _ { t } } = \Gamma _ { \alpha } ( W _ { t } , C _ { t } ) .
$$

Applying (89) with $\left( k , \ell \right) = \left( k _ { t } , \ell _ { t } \right)$ gives

$$
\overline { { H } } _ { t } < \frac { d _ { k _ { t } , \ell _ { t } } } { \lambda _ { k _ { t } , \ell _ { t } } } + \frac { \lambda _ { k _ { t } , \ell _ { t } } } { 2 ( 1 - \lambda _ { k _ { t } , \ell _ { t } } \overline { { b } } _ { \ell _ { t } } ) } W _ { t } \leq \frac { d _ { k _ { t } , \ell _ { t } } } { \lambda _ { k _ { t } , \ell _ { t } } } + \frac { \lambda _ { k _ { t } , \ell _ { t } } \overline { { v } } _ { k _ { t } } } { 2 ( 1 - \lambda _ { k _ { t } , \ell _ { t } } \overline { { b } } _ { \ell _ { t } } ) } .
$$

By the choice of $\lambda _ { k _ { t } , \ell _ { t } }$ in (87),

$$
\frac { d _ { k _ { t } , \ell _ { t } } } { \lambda _ { k _ { t } , \ell _ { t } } } + \frac { \lambda _ { k _ { t } , \ell _ { t } } \overline { { v } } _ { k _ { t } } } { 2 ( 1 - \lambda _ { k _ { t } , \ell _ { t } } \overline { { b } } _ { \ell _ { t } } ) } = \sqrt { 2 d _ { k _ { t } , \ell _ { t } } \overline { { v } } _ { k _ { t } } } + d _ { k _ { t } , \ell _ { t } } \overline { { b } } _ { \ell _ { t } } .
$$

Consequently,

$$
\overline { { H } } _ { t } < \sqrt { 2 d _ { k _ { t } , \ell _ { t } } \overline { { v } } _ { k _ { t } } } + d _ { k _ { t } , \ell _ { t } } \overline { { b } } _ { \ell _ { t } } \leq 2 \sqrt { W _ { t , \mathrm { e f f } } d _ { k _ { t } , \ell _ { t } } } + 2 C _ { t , \mathrm { e f f } } d _ { k _ { t } , \ell _ { t } } = \mathfrak { B } _ { \alpha } ( W _ { t } , C _ { t } ) .
$$

Since $t \geq t _ { 0 }$ was arbitrary, this proves (33). Monotonicity follows because all terms in (29)–(32) are nondecreasing in the corresponding argument.

## A.2.3 Proof of Theorem 3.5

The recursion is well defined: at time $t ,$ all $R _ { s } ^ { \mathrm { E B } }$ with $s \leq t$ have already been computed, and therefore $( 4 0 ) \mathrm { - } ( 4 2 )$ define nonnegative $\mathcal { F } _ { t }$ -measurable quantities. Hence the resulting process is fully observable. Let

$$
\mathcal { E } _ { \alpha } ^ { \mathrm { E B } } : = \left\{ \forall t \geq t _ { 0 } : \ Z _ { t + 1 } \leq U _ { t + 1 } ^ { \mathrm { d i s t , E B } } ( \alpha ) \right\} .
$$

By Proposition B.1, $\mathbb { P } ( \mathcal { E } _ { \alpha } ^ { \mathrm { E B } } ) \geq 1 - \alpha$ . We prove by induction that, on this event, for every $t \geq t _ { 0 }$

$$
V _ { t } ^ { \mathrm { d i s t , E B } } \leq \widehat { V } _ { t } ^ { \mathrm { d i s t , E B } } , \qquad B _ { t } ^ { \mathrm { d i s t , E B } } \leq \widehat { B } _ { t } ^ { \mathrm { d i s t , E B } } , \qquad U _ { t + 1 } ^ { \mathrm { d i s t , E B } } ( \alpha ) \leq \widehat { U } _ { t + 1 } ^ { \mathrm { d i s t , E B } } ( \alpha ) .\tag{90}
$$

For $t = t _ { 0 }$ , Cauchy–Schwarz and $Z _ { t _ { 0 } } \leq R _ { 0 } = R _ { t _ { 0 } } ^ { \mathrm { E B } }$ give

$$
V _ { t _ { 0 } } ^ { \mathrm { d i s t , E B } } \leq 4 A _ { t _ { 0 } } ^ { 2 } \eta _ { t _ { 0 } } ^ { 2 } \left. g _ { t _ { 0 } } \right. ^ { 2 } R _ { t _ { 0 } } ^ { \mathrm { E B } } = \widehat { V } _ { t _ { 0 } } ^ { \mathrm { d i s t , E B } } ,
$$

and $B _ { t _ { 0 } } ^ { \mathrm { d i s t , E B } } \leq \widehat { B } _ { t _ { 0 } } ^ { \mathrm { d i s t , E B } }$ . Monotonicity of $\mathfrak { B } _ { \alpha }$ then implies the final inequality in (90).

Now fix $t \geq t _ { 0 } + 1$ and suppose the claim holds through time t 1. On $\mathcal { E } _ { \alpha } ^ { \mathrm { E B } }$ , for every $s = t _ { 0 } + 1 , \ldots , t ,$

$$
Z _ { s } \leq U _ { s } ^ { \mathrm { d i s t , E B } } ( \alpha ) \leq \widehat { U } _ { s } ^ { \mathrm { d i s t , E B } } ( \alpha ) = R _ { s } ^ { \mathrm { E B } } .
$$

Together with $Z _ { t _ { 0 } } \leq R _ { t _ { 0 } } ^ { \mathrm { E B } }$ , this yields

$$
V _ { t } ^ { \mathrm { d i s t , E B } } \leq 4 \sum _ { s = t _ { 0 } } ^ { t } A _ { s } ^ { 2 } \eta _ { s } ^ { 2 } \left. g _ { s } \right. ^ { 2 } R _ { s } ^ { \mathrm { E B } } = \widehat { V } _ { t } ^ { \mathrm { d i s t , E B } } ,
$$

$$
B _ { t } ^ { \mathrm { d i s t , E B } } \leq 2 G \operatorname* { m a x } _ { t _ { 0 } \leq s \leq t } A _ { s } \eta _ { s } \sqrt { R _ { s } ^ { \mathrm { E B } } } = \widehat { B } _ { t } ^ { \mathrm { d i s t , E B } } .
$$

Monotonicity of $\mathfrak { B } _ { \alpha }$ and $Z _ { t _ { 0 } } \leq R _ { 0 }$ then give

$$
U _ { t + 1 } ^ { \mathrm { d i s t , E B } } ( \alpha ) \leq \widehat { U } _ { t + 1 } ^ { \mathrm { d i s t , E B } } ( \alpha ) .
$$

This closes the induction. Hence, on $\mathcal { E } _ { \alpha } ^ { \mathrm { E B } }$

$$
Z _ { t + 1 } \leq \widehat { U } _ { t + 1 } ^ { \mathrm { d i s t , E B } } ( \alpha ) \qquad \mathrm { f o r ~ e v e r y ~ } t \geq t _ { 0 } .
$$

The theorem follows because $\mathbb { P } ( \mathcal { E } _ { \alpha } ^ { \mathrm { E B } } ) \geq 1 - \alpha$

## A.2.4 Proof of Proposition 3.6

Throughout the proof, $C < \infty$ denotes a universal constant whose value may change from line to line. For convenience, throughout the proof define

$$
\widehat { V } _ { t , \mathrm { e f f } } ^ { \mathrm { d i s t , E B } } : = \operatorname* { m a x } \left\{ \widehat { V } _ { t } ^ { \mathrm { d i s t , E B } } , 1 \right\} , \qquad \widehat { B } _ { t , \mathrm { e f f } } ^ { \mathrm { d i s t , E B } } : = \operatorname* { m a x } \left\{ \widehat { B } _ { t } ^ { \mathrm { d i s t , E B } } , 1 \right\} .
$$

For the specified stepsize,

$$
A _ { t } = \prod _ { s = 3 } ^ { t } { \frac { s } { s - 2 } } = { \frac { t ( t - 1 ) } { 2 } } , \qquad t \geq 3 .
$$

Consequently,

$$
A _ { t } \geq \frac { t ^ { 2 } } { 4 } , A _ { t } \eta _ { t } ^ { 2 } = \frac { t - 1 } { 2 \mu ^ { 2 } t } \leq \frac { 1 } { 2 \mu ^ { 2 } } , A _ { t } ^ { 2 } \eta _ { t } ^ { 2 } = \frac { ( t - 1 ) ^ { 2 } } { 4 \mu ^ { 2 } } \leq \frac { t ^ { 2 } } { 4 \mu ^ { 2 } } , A _ { t } \eta _ { t } = \frac { t - 1 } { 2 \mu } \leq \frac { t } { 2 \mu } .\tag{91}
$$

In particular, (37) is automatic.

Write

$$
L _ { t } : = L _ { t } ^ { \mathrm { E B } } ( \alpha ) , \qquad q _ { t } : = \frac { L _ { t } } { t } , \qquad \Phi _ { t } : = q _ { t } + q _ { t } ^ { 2 } .
$$

As in the proof of Proposition 2.4, we use an induction argument. We claim that, for a suficiently large universal constant K,

$$
\widehat { U } _ { s } ^ { \mathrm { d i s t , E B } } ( \alpha ) \leq K B _ { \mathrm { E B } } \Phi _ { s } , \qquad s \geq 4 .\tag{92}
$$

We first verify the claim at $s = 4$ . Since $A _ { 3 } \eta _ { 3 } = 1 / \mu$ , the definitions (40)–(41) give

$$
\widehat { V } _ { 3 , \mathrm { e f f } } ^ { \mathrm { d i s t , E B } } \leq 4 B _ { \mathrm { E B } } ^ { 2 } , \qquad \widehat { B } _ { 3 , \mathrm { e f f } } ^ { \mathrm { d i s t , E B } } \leq 2 B _ { \mathrm { E B } } .
$$

Consequently,

$$
\begin{array} { r } { k \left( \widehat { V } _ { 3 } ^ { \mathrm { d i s t , E B } } \right) + 2 \leq C \left( 1 + \log B _ { \mathrm { E B } } \right) , \qquad \ell \left( \widehat { B } _ { 3 } ^ { \mathrm { d i s t , E B } } \right) + 2 \leq C \left( 1 + \log B _ { \mathrm { E B } } \right) , } \end{array}
$$

and hence

$$
\Gamma _ { \alpha } \left( \widehat { V } _ { 3 } ^ { \mathrm { d i s t , E B } } , \widehat { B } _ { 3 } ^ { \mathrm { d i s t , E B } } \right) \leq C L _ { 4 } .
$$

Using these bounds in (42), together with $R _ { 0 } \leq B _ { \mathrm { E B } } ,$ yields

$$
\widehat { U } _ { 4 } ^ { \mathrm { d i s t , E B } } ( \alpha ) \leq C B _ { \mathrm { E B } } \left( 1 + \sqrt { L _ { 4 } } + L _ { 4 } \right) \leq C B _ { \mathrm { E B } } L _ { 4 } .
$$

Since $L _ { 4 } \geq 1$ and $\Phi _ { 4 } \ge L _ { 4 } / 4$ , this proves (92) at $s = 4$ for some constant K.

Now fix $t \geq 4$ and suppose (92) holds through time t. The deterministic part of (42) satisfies

$$
\begin{array} { r l r } & { } & { \displaystyle \frac { 1 } { A _ { t } } \left[ R _ { 0 } + \sum _ { s = 3 } ^ { t } A _ { s } \eta _ { s } ^ { 2 } \left. g _ { s } \right. ^ { 2 } \right] \leq \frac { 2 } { t ( t - 1 ) } \left[ \frac { G ^ { 2 } } { \mu ^ { 2 } } + \frac { t - 2 } { 2 } \frac { G ^ { 2 } } { \mu ^ { 2 } } \right] } \\ & { } & { \quad \quad = \frac { G ^ { 2 } / \mu ^ { 2 } } { t - 1 } \leq 2 B _ { \mathrm { E B } } q _ { t + 1 } . \quad } \end{array}\tag{93}
$$

We next control the two arguments of the empirical Bernstein boundary. We claim that

$$
\widehat { V } _ { t , \mathrm { e f f } } ^ { \mathrm { d i s t , E B } } \leq C K B _ { \mathrm { E B } } ^ { 2 } t ^ { 2 } L _ { t } ( 1 + q _ { t } ) ,\tag{94}
$$

$$
\widehat { B } _ { t , \mathrm { e f f } } ^ { \mathrm { d i s t , E B } } \leq C \sqrt { K } B _ { \mathrm { E B } } t \sqrt { q _ { t } ( 1 + q _ { t } ) } .\tag{95}
$$

For (94), the contribution of $s = 3$ is at most $4 B _ { \mathrm { E B } } ^ { 2 }$ . For $s \geq 4$ , the induction hypothesis, (91), and $G ^ { 2 } / \mu ^ { 2 } \leq B _ { \mathrm { E B } }$ give

$$
4 A _ { s } ^ { 2 } \eta _ { s } ^ { 2 } \left. g _ { s } \right. ^ { 2 } \widehat { U } _ { s } ^ { \mathrm { d i s t , E B } } ( \alpha ) \leq K B _ { \mathrm { E B } } ^ { 2 } s ^ { 2 } \Phi _ { s } .
$$

Since $L _ { s }$ is nondecreasing,

$$
\sum _ { s = 4 } ^ { t } s ^ { 2 } \Phi _ { s } = \sum _ { s = 4 } ^ { t } \left( s L _ { s } + L _ { s } ^ { 2 } \right) \le C \left( t ^ { 2 } L _ { t } + t L _ { t } ^ { 2 } \right) = C t ^ { 2 } L _ { t } ( 1 + q _ { t } ) .
$$

The initial contribution and the unit floor are absorbed by the right-hand side of (94).

For (95), the contribution of $s = 3$ is at most $2 B _ { \mathrm { E B } }$ . For $s \geq 4 .$ , the induction hypothesis and (91) give

$$
2 G A _ { s } \eta _ { s } \sqrt { \hat { U } _ { s } ^ { \mathrm { d i s t , E B } } ( \alpha ) } \leq C \sqrt { K } B _ { \mathrm { E B } } s \sqrt { \Phi _ { s } } \leq C \sqrt { K } B _ { \mathrm { E B } } \left( \sqrt { s L _ { s } } + L _ { s } \right) ,
$$

where we used $( G / \mu ) \sqrt { B _ { \mathrm { E B } } } \leq B _ { \mathrm { E B } }$ . By the monotonicity of $L _ { s }$ and the inequality

$$
{ \sqrt { q } } + q \leq 2 { \sqrt { q ( 1 + q ) } } , \qquad q \geq 0 ,
$$

we obtain

$$
\operatorname* { m a x } _ { 4 \leq s \leq t } \Big ( \sqrt { s L _ { s } } + L _ { s } \Big ) \leq 2 t \sqrt { q _ { t } ( 1 + q _ { t } ) } .
$$

The initial contribution and the unit floor are again absorbed, proving (95).

We now bound the logarithmic factor. By (94)–(95) and the definitions in (30),

$$
\begin{array} { r } { k \left( \widehat { V } _ { t } ^ { \mathrm { d i s t , E B } } \right) + 2 \leq C \left[ 1 + \log K + \log B _ { \mathrm { E B } } + \log t + \log L _ { t } + \log ( 1 + q _ { t } ) \right] , } \\ { \ell \left( \widehat { B } _ { t } ^ { \mathrm { d i s t , E B } } \right) + 2 \leq C \left[ 1 + \log K + \log B _ { \mathrm { E B } } + \log t + \log L _ { t } + \log ( 1 + q _ { t } ) \right] . } \end{array}
$$

Since log $( ( j + 1 ) ( j + 2 ) ) \leq 2 \log ( j + 2 )$ for $j \geq 0$ , these bounds yield

$$
\Gamma _ { \alpha } \left( \widehat { V } _ { t } ^ { \mathrm { d i s t , E B } } , \widehat { B } _ { t } ^ { \mathrm { d i s t , E B } } \right) \leq C \left[ 1 + \log \frac { 1 } { \alpha } + \log ( 1 + \log K ) + \log \log ( e B _ { \mathrm { E B } } ) \right.
$$

$$
+ \log \log ( e t ) + \log ( 1 + \log L _ { t } ) + \log ( 1 + \log ( 1 + q _ { t } ) ) { \Biggr ] } .
$$

The definition of $L _ { t }$ and the elementary bounds

$$
\begin{array} { r } { \log ( 1 + \log L _ { t } ) \leq L _ { t } , \qquad \log ( 1 + \log ( 1 + q _ { t } ) ) \leq L _ { t } } \end{array}
$$

therefore imply

$$
\begin{array} { r } { \Gamma _ { \alpha } \left( \widehat { V } _ { t } ^ { \mathrm { d i s t , E B } } , \widehat { B } _ { t } ^ { \mathrm { d i s t , E B } } \right) \leq C \rho _ { K } L _ { t } , \qquad \rho _ { K } : = 1 + \log ( 1 + \log K ) . } \end{array}\tag{96}
$$

Combining (91), (94), and (96), the square-root contribution satisfies

$$
\begin{array} { r } { \frac { 2 } { A _ { t } } \sqrt { \widehat { V } _ { t , \mathrm { e f f } } ^ { \mathrm { d i s t } , \mathrm { E B } } \Gamma _ { \alpha } \left( \widehat { V } _ { t } ^ { \mathrm { d i s t } , \mathrm { E B } } , \widehat { B } _ { t } ^ { \mathrm { d i s t } , \mathrm { E B } } \right) } \leq C \sqrt { K \rho _ { K } } B _ { \mathrm { E B } } q _ { t } \sqrt { 1 + q _ { t } } } \\ { \leq C \sqrt { K \rho _ { K } } B _ { \mathrm { E B } } \left( q _ { t } + q _ { t } ^ { 2 } \right) . } \end{array}\tag{97}
$$

Similarly, the linear range contribution satisfies

$$
\mathcal { R } _ { t } ^ { \mathrm { E B } } ( \alpha ) \leq C \sqrt { K } \rho _ { K } B _ { \mathrm { E B } } q _ { t } \sqrt { q _ { t } ( 1 + q _ { t } ) } .\tag{98}
$$

For every $q _ { t } \geq 0$ , the right-hand side of (98) is bounded by a constant multiple of $B _ { \mathrm { E B } } ( q _ { t } + q _ { t } ^ { 2 } )$ ; whenever $q _ { t } \leq 1$ , it is bounded by a constant multiple of $B _ { \mathrm { E B } } q _ { t } ^ { 3 / 2 }$

Combining (93), (97), and (98), and using

$$
q _ { t } \leq \frac { t + 1 } { t } q _ { t + 1 } \leq \frac 4 3 q _ { t + 1 } , \qquad t \geq 3 ,
$$

which implies $\Phi _ { t } \le ( 1 6 / 9 ) \Phi _ { t + 1 }$ , gives

$$
\widehat { U } _ { t + 1 } ^ { \mathrm { d i s t , E B } } ( \alpha ) \leq \left[ C + C \sqrt { K \rho _ { K } } + C \sqrt { K } \rho _ { K } \right] B _ { \mathrm { E B } } \Phi _ { t + 1 } .
$$

Since $\rho _ { K } = o ( \sqrt { K } )$ , a suficiently large universal K makes the bracketed coeficient at most K. This closes the induction, and proves (46). Moreover, whenever $L _ { t + 1 } ^ { \mathrm { E B } } ( \alpha ) \leq t + 1$ , we have $q _ { t + 1 } \leq 1$ and therefore $\Phi _ { t + 1 } \leq 2 q _ { t + 1 }$

Finally, whenever $t \geq 4$ and $q _ { t } \leq 1$ , the sharper form of (98) gives (47) after fixing the universal induction constant K. For fixed α and fixed problem parameters,

$$
L _ { t } ^ { \mathrm { E B } } ( \alpha ) = O ( 1 + \log \log t ) ,
$$

so $q _ { t } \to 0$ and $q _ { t } ^ { 3 / 2 } = o ( q _ { t } )$ . This also confirms the lower-order interpretation stated after the proposition.

## A.2.5 Proof of Theorem 3.8

Fix $\alpha \in ( 0 , 1 )$ and set $\beta : = \alpha / 2$ . For the prescribed deterministic stepsize, condition (37) is automatic, so Theorem 3.5 applies at confidence level $\beta .$ . Define

$$
\mathcal { E } _ { \mathrm { s u b } } ^ { \mathrm { E B } } : = \Big \{ \forall t \geq t _ { 0 } : ~ f ( \bar { x } _ { t } ) - f ( x ^ { \star } ) \leq U _ { t } ^ { \mathrm { s u b , E B } } ( \beta ) \Big \}
$$

and

$$
\mathcal { E } _ { \mathrm { d i s t } } ^ { \mathrm { E B } } : = \left\{ \forall s \geq t _ { 0 } + 1 : \ Z _ { s } \leq \widehat { U } _ { s } ^ { \mathrm { d i s t , E B } } ( \beta ) \right\} .
$$

By Proposition B.2 and Theorem 3.5, each event has probability at least $1 - \beta$

On $\mathcal { E } _ { \mathrm { d i s t } } ^ { \mathrm { E B } }$ , equation (52) and the initialization $Z _ { t _ { 0 } } \leq R _ { 0 }$ imply

$$
Z _ { s } \leq R _ { s } ^ { \mathrm { d i s t , E B } } ( \beta ) \qquad \mathrm { f o r ~ e v e r y ~ } s \geq t _ { 0 } .
$$

Consequently, Cauchy–Schwarz gives, for every $t \geq t _ { 0 }$

$$
\begin{array} { r l } & { { \cal V } _ { t } ^ { \mathrm { s u b , E B } } \le \displaystyle \sum _ { s = t _ { 0 } } ^ { t } s ^ { 2 } \| g _ { s } \| ^ { 2 } Z _ { s } \le \widehat { V } _ { t } ^ { \mathrm { s u b , E B } } ( \beta ) , } \\ & { { \cal B } _ { t } ^ { \mathrm { s u b , E B } } \le G \operatorname* { m a x } _ { t _ { 0 } \le s \le t } ^ { \operatorname* { m a x } } s \sqrt { R _ { s } ^ { \mathrm { d i s t , E B } } ( \beta ) } = \widehat { B } _ { t } ^ { \mathrm { s u b , E B } } ( \beta ) . } \end{array}
$$

By the monotonicity of $\mathfrak { B } _ { \beta }$ in both arguments and $Z _ { t _ { 0 } } \leq R _ { 0 }$ , it follows that

$$
U _ { t } ^ { \mathrm { s u b , E B } } ( \beta ) \leq \widehat { U } _ { t } ^ { \mathrm { s u b , E B } } ( \alpha ) \qquad \mathrm { f o r ~ e v e r y ~ } t \geq t _ { 0 } .
$$

Therefore, on $\mathcal { E } _ { \mathrm { s u b } } ^ { \mathrm { E B } } \cap \mathcal { E } _ { \mathrm { d i s t } } ^ { \mathrm { E B } }$ ，

$$
f ( \bar { x } _ { t } ) - f ( x ^ { \star } ) \leq \widehat { U } _ { t } ^ { \mathrm { s u b , E B } } ( \alpha ) \qquad \mathrm { f o r ~ e v e r y ~ } t \geq t _ { 0 } .
$$

The union bound gives

$$
\begin{array} { r } { \mathbb { P } \left( \mathcal { E } _ { \mathrm { s u b } } ^ { \mathrm { E B } } \cap \mathcal { E } _ { \mathrm { d i s t } } ^ { \mathrm { E B } } \right) \ge 1 - 2 \beta = 1 - \alpha , } \end{array}
$$

which proves the theorem.

## A.2.6 Proof of Proposition 3.9

Throughout the proof, $C < \infty$ denotes a universal constant whose value may change from line to line. For convenience, throughout the proof define

$$
\widehat { V } _ { t , \mathrm { e f f } } ^ { \mathrm { s u b , E B } } ( \beta ) : = \operatorname* { m a x } \left\{ \widehat { V } _ { t } ^ { \mathrm { s u b , E B } } ( \beta ) , 1 \right\} , \qquad \widehat { B } _ { t , \mathrm { e f f } } ^ { \mathrm { s u b , E B } } ( \beta ) : = \operatorname* { m a x } \left\{ \widehat { B } _ { t } ^ { \mathrm { s u b , E B } } ( \beta ) , 1 \right\} .
$$

Set

$$
\beta : = \frac { \alpha } { 2 } , \qquad L _ { t } : = L _ { t } ^ { \mathrm { s u b , E B } } ( \alpha ) , \qquad q _ { t } : = \frac { L _ { t } } { t } .
$$

The sequence $L _ { t }$ is nondecreasing and satisfies $L _ { t } \geq 1$

Following Remark 3.7, we have:

$$
\widehat { U } _ { s } ^ { \mathrm { d i s t , E B } } ( \beta ) \leq C B _ { \mathrm { E B } } \left[ \frac { L _ { s } ^ { \mathrm { E B } } ( \beta ) } { s } + \left( \frac { L _ { s } ^ { \mathrm { E B } } ( \beta ) } { s } \right) ^ { 2 } \right] , \qquad s \geq 5 .\tag{99}
$$

$\mathrm { B y }$ (57), $L _ { s } ^ { \mathrm { E B } } ( \beta ) \leq L _ { s } ,$ , and hence

$$
R _ { s } ^ { \mathrm { d i s t , E B } } ( \beta ) \leq C B _ { \mathrm { E B } } \left( q _ { s } + q _ { s } ^ { 2 } \right) , \qquad s \geq 5 .\tag{100}
$$

At the initial index,

$$
R _ { 4 } ^ { \mathrm { d i s t , E B } } ( \beta ) = R _ { 0 } = \frac { G ^ { 2 } } { \mu ^ { 2 } } .
$$

We next control the two arguments of the empirical Bernstein boundary. We claim that, for every $t \geq 4$ 2

$$
\widehat { V } _ { t , \mathrm { e f f } } ^ { \mathrm { s u b , E B } } ( \beta ) \leq C \left( B _ { \mathrm { s u b } } ^ { \mathrm { E B } } \right) ^ { 2 } t ^ { 2 } L _ { t } ( 1 + q _ { t } ) ,\tag{101}
$$

$$
\widehat { B } _ { t , \mathrm { e f f } } ^ { \mathrm { s u b , E B } } ( \beta ) \leq C B _ { \mathrm { s u b } } ^ { \mathrm { E B } } t \sqrt { q _ { t } ( 1 + q _ { t } ) } .\tag{102}
$$

For (101), the contribution of $s = 4$ is bounded by

$$
1 6 G ^ { 2 } R _ { 0 } = \frac { 1 6 G ^ { 4 } } { \mu ^ { 2 } } \leq 1 6 \left( B _ { \mathrm { s u b } } ^ { \mathrm { E B } } \right) ^ { 2 } .
$$

For $s \geq 5 ,$ , Assumption 3.1 and (100) give

$$
\begin{array} { r } { s ^ { 2 } \left. g _ { s } \right. ^ { 2 } R _ { s } ^ { \mathrm { d i s t , E B } } ( \beta ) \le C G ^ { 2 } B _ { \mathrm { E B } } \left( s L _ { s } + L _ { s } ^ { 2 } \right) . } \end{array}
$$

Moreover,

$$
\begin{array} { r } { G ^ { 2 } B _ { \mathrm { E B } } \leq \mu ^ { 2 } B _ { \mathrm { E B } } ^ { 2 } \leq \left( B _ { \mathrm { s u b } } ^ { \mathrm { E B } } \right) ^ { 2 } , } \end{array}
$$

because $G ^ { 2 } \leq \mu ^ { 2 } B _ { \mathrm { E B } }$ . Since $L _ { s }$ is nondecreasing,

$$
\sum _ { s = 5 } ^ { t } \Big ( s L _ { s } + L _ { s } ^ { 2 } \Big ) \leq C \left( t ^ { 2 } L _ { t } + t L _ { t } ^ { 2 } \right) = C t ^ { 2 } L _ { t } ( 1 + q _ { t } ) .
$$

The initial contribution and the unit floor are therefore absorbed by the right-hand side of (101).

For (102), the contribution of $s = 4$ is at most

$$
4 G \sqrt { R _ { 0 } } = \frac { 4 G ^ { 2 } } { \mu } \leq 4 B _ { \mathrm { s u b } } ^ { \mathrm { E B } } .
$$

For $s \geq 5 .$ , equation (100) yields

$$
G s \sqrt { R _ { s } ^ { \mathrm { d i s t , E B } } ( \beta ) } \leq C G \sqrt { B _ { \mathrm { E B } } } s \sqrt { q _ { s } ( 1 + q _ { s } ) } .
$$

Since

$$
G \sqrt { B _ { \mathrm { E B } } } \leq \mu B _ { \mathrm { E B } } \leq B _ { \mathrm { s u b } } ^ { \mathrm { E B } } ,
$$

and

$$
s ^ { 2 } q _ { s } ( 1 + q _ { s } ) = s L _ { s } + L _ { s } ^ { 2 } \leq t L _ { t } + L _ { t } ^ { 2 } = t ^ { 2 } q _ { t } ( 1 + q _ { t } ) ,
$$

we obtain (102), after absorbing the initial contribution and the unit floor.

We now control the logarithmic factor. The definitions in (30), together with (101)–(102), imply

$$
\begin{array} { r } { k \left( \widehat { V } _ { t } ^ { \mathrm { s u b , E B } } ( \beta ) \right) + 2 \leq C \left[ 1 + \log B _ { \mathrm { s u b } } ^ { \mathrm { E B } } + \log t + \log L _ { t } + \log ( 1 + q _ { t } ) \right] , } \\ { \ell \left( \widehat { B } _ { t } ^ { \mathrm { s u b , E B } } ( \beta ) \right) + 2 \leq C \left[ 1 + \log B _ { \mathrm { s u b } } ^ { \mathrm { E B } } + \log t + \log L _ { t } + \log ( 1 + q _ { t } ) \right] . } \end{array}
$$

Using log $( ( j + 1 ) ( j + 2 ) ) \leq 2 \log ( j + 2 )$ for $j \geq 0$ , we conclude that

$$
\begin{array} { r l } & { \Gamma _ { \beta } \left( \widehat { V } _ { t } ^ { \mathrm { s u b , E B } } ( \beta ) , \widehat { B } _ { t } ^ { \mathrm { s u b , E B } } ( \beta ) \right) \leq C \bigg [ 1 + \log \frac { 2 } { \alpha } + \log \log \left( e B _ { \mathrm { s u b } } ^ { \mathrm { E B } } \right) + \log \log ( e t ) } \\ & { \qquad + \log ( 1 + \log L _ { t } ) + \log ( 1 + \log ( 1 + q _ { t } ) ) \bigg ] . } \end{array}
$$

The definition of $L _ { t }$ controls the first four terms. Furthermore, since $L _ { t } \geq 1$

$$
\log ( 1 + \log L _ { t } ) \leq L _ { t } ,
$$

and, because log $( 1 + x ) \leq x$ for every $x \geq 0$

$$
\log ( 1 + \log ( 1 + q _ { t } ) ) \leq \log ( 1 + q _ { t } ) \leq q _ { t } = \frac { L _ { t } } { t } \leq L _ { t } .
$$

Hence

$$
\begin{array} { r } { \Gamma _ { \beta } \left( \widehat { V } _ { t } ^ { \mathrm { s u b , E B } } ( \beta ) , \widehat { B } _ { t } ^ { \mathrm { s u b , E B } } ( \beta ) \right) \leq C L _ { t } . } \end{array}\tag{103}
$$

Since

$$
S _ { t } = \sum _ { s = 4 } ^ { t } s = \frac { t ( t + 1 ) } { 2 } - 6 \geq \frac { t ^ { 2 } } { 4 } , \qquad t \geq 4 ,
$$

combining (101) and (103) gives

$$
\frac { 2 } { S _ { t } } \sqrt { \widehat { V } _ { t , \mathrm { e f f } } ^ { \mathrm { s u b , E B } } ( \beta ) \Gamma _ { \beta } \left( \widehat { V } _ { t } ^ { \mathrm { s u b , E B } } ( \beta ) , \widehat { B } _ { t } ^ { \mathrm { s u b , E B } } ( \beta ) \right) } \leq C B _ { \mathrm { s u b } } ^ { \mathrm { E B } } q _ { t } \sqrt { 1 + q _ { t } } \leq C B _ { \mathrm { s u b } } ^ { \mathrm { E B } } \left( q _ { t } + q _ { t } ^ { 2 } \right) .\tag{104}
$$

Similarly, (102) and (103) yield

$$
\mathcal { R } _ { t } ^ { \mathrm { s u b , E B } } ( \alpha ) \leq C B _ { \mathrm { s u b } } ^ { \mathrm { E B } } q _ { t } \sqrt { q _ { t } ( 1 + q _ { t } ) } .\tag{105}
$$

For every $q _ { t } \geq 0$ , the right-hand side is bounded by a constant multiple of $B _ { \mathrm { s u b } } ^ { \mathrm { E B } } ( q _ { t } + q _ { t } ^ { 2 } )$ ; whenever $q _ { t } \leq 1$ , it is bounded by a constant multiple of $B _ { \mathrm { s u b } } ^ { \mathrm { E B } } q _ { t } ^ { 3 / 2 }$

It remains to control the deterministic terms in (55). Assumption 3.1, $R _ { 0 } = G ^ { 2 } / \mu ^ { 2 }$ , and $S _ { t } \geq t ^ { 2 } / 4$ give

$$
\frac { 1 } { S _ { t } } \left[ \frac { 1 } { \mu } \sum _ { s = 4 } ^ { t } \| g _ { s } \| ^ { 2 } + 3 \mu R _ { 0 } \right] \leq \frac { C } { t } \frac { G ^ { 2 } } { \mu } \leq C B _ { \mathrm { s u b } } ^ { \mathrm { E B } } q _ { t } ,
$$

where we used $L _ { t } \geq 1$ . Combining this bound with (104) and (105) gives the stronger estimate

$$
\widehat { U } _ { t } ^ { \mathrm { s u b , E B } } ( \alpha ) \leq C B _ { \mathrm { s u b } } ^ { \mathrm { E B } } \left( q _ { t } + q _ { t } ^ { 2 } \right) , \qquad t \geq 4 .
$$

This proves (59). Moreover, whenever $L _ { t } \leq t ,$ , we have $q _ { t } \leq 1$ , and therefore $q _ { t } + q _ { t } ^ { 2 } \le 2 q _ { t }$ . Under the same condition, the sharper form of (105) gives (60).

Finally, for fixed α and fixed problem parameters,

$$
L _ { t } ^ { \mathrm { s u b , E B } } ( \alpha ) = O ( 1 + \log \log t ) ,
$$

so $q _ { t } \to 0$ and $q _ { t } ^ { 3 / 2 } = o ( q _ { t } )$ . This also confirms the lower-order interpretation stated after the proposition.

## A.3 Proofs of Section 4

## A.3.1 Proof of Proposition 4.1

For brevity, write

$$
H _ { t } : = \frac { 1 } { b } \sum _ { i = 1 } ^ { b } H _ { t } ^ { ( i ) } .
$$

We follow the predictable-truncation and stitching proof of Theorem 3.3. Fix a dyadic range level $\ell \geq 0$ let $\overline { { b } } _ { \ell } : = 2 ^ { \ell }$ , and define the predictable truncation

$$
J _ { t } ^ { ( \ell ) } : = \mathbb { 1 } \left\{ C _ { t } \leq \overline { { b } } _ { \ell } \right\} .
$$

For every $\lambda \in [ 0 , 1 / \overline { { b } } _ { \ell } )$ , Lemma 3.2 applied conditionally to $J _ { t } ^ { ( \ell ) } H _ { t } ^ { ( i ) }$ gives

$$
\mathbb { E } \Bigg [ \exp \left( \lambda J _ { t } ^ { ( \ell ) } ( H _ { t } ^ { ( i ) } - m _ { t } ) - \frac { \lambda ^ { 2 } J _ { t } ^ { ( \ell ) } } { 2 \big ( 1 - \lambda \overline { { b _ { \ell } } } \big ) } \big ( H _ { t } ^ { ( i ) } \big ) ^ { 2 } \right) \Big | \mathcal { F } _ { t - 1 } \Bigg ] \leq 1 .
$$

Because $J _ { t } ^ { ( \ell ) }$ is $\mathcal { F } _ { t - 1 }$ -measurable and $H _ { t } ^ { ( 1 ) } , \dots , H _ { t } ^ { ( b ) }$ are conditionally independent given $\mathcal { F } _ { t - 1 }$ , the corresponding exponential factors are conditionally independent. Multiplying their conditional expectations therefore gives

$$
\mathbb { E } \Bigg [ \exp \left( \lambda b J _ { t } ^ { ( \ell ) } ( H _ { t } - m _ { t } ) - \frac { \lambda ^ { 2 } J _ { t } ^ { ( \ell ) } } { 2 \big ( 1 - \lambda \overline { { b } } _ { \ell } \big ) } \sum _ { i = 1 } ^ { b } ( H _ { t } ^ { ( i ) } ) ^ { 2 } \right) \Big | \mathcal { F } _ { t - 1 } \Bigg ] \leq 1 .
$$

Thus, for every fixed intrinsic-time and range scale, the exponential process used in the proof of Theorem 3.3 remains a nonnegative supermartingale after replacing its centered increment by $b J _ { t } ^ { ( \ell ) } ( H _ { t } - m _ { t } )$ and its observed quadratic increment by $\begin{array} { r } { J _ { t } ^ { ( \ell ) } \sum _ { i = 1 } ^ { b } ( H _ { t } ^ { ( i ) } ) ^ { 2 } } \end{array}$ . Repeating the same dyadic stitching over the intrinsic-time and range scales gives, with probability at least $1 - \alpha$ , simultaneously for every $t \geq t _ { 0 }$

$$
b \sum _ { s = t _ { 0 } } ^ { t } ( H _ { s } - m _ { s } ) \leq \mathfrak { B } _ { \alpha } \left( W _ { t } ^ { \mathrm { M B } } , C _ { t } \right) .
$$

Since $m _ { s } = \mathbb { E } [ H _ { s } \mid { \mathcal { F } } _ { s - 1 } ]$ , division by b proves (63).

## B Additional Results

Proposition B.1 (Refined confidence sequence for $\| x _ { t + 1 } - x ^ { \star } \| ^ { 2 } )$ . Suppose Assumptions 1.1–1.4 and 3.1 hold, and suppose (37) is satisfied. Let $V _ { t } ^ { \mathrm { d i s t , E B } }$ and $B _ { t } ^ { \mathrm { d i s t , E B } }$ be defined as in (35)–(36). Then, for every $\alpha \in ( 0 , 1 )$ , the process $U _ { t + 1 } ^ { \mathrm { d i s t , E B } } ( \alpha )$ defined in (38) satisfies

$$
\begin{array} { r } { \mathbb { P } \left( \forall t \geq t _ { 0 } : \ Z _ { t + 1 } \leq U _ { t + 1 } ^ { \mathrm { d i s t , E B } } ( \alpha ) \right) \geq 1 - \alpha . } \end{array}
$$

Proof. By Lemma C.2, for every $t \geq t _ { 0 }$

$$
A _ { t } Z _ { t + 1 } \leq Z _ { t _ { 0 } } + \sum _ { s = t _ { 0 } } ^ { t } A _ { s } \eta _ { s } ^ { 2 } \left. g _ { s } \right. ^ { 2 } + \sum _ { s = t _ { 0 } } ^ { t } X _ { s } .
$$

By (34) and conditional unbiasedness,

$$
H _ { s } ^ { \mathrm { d i s t } } - \mathbb { E } \left[ H _ { s } ^ { \mathrm { d i s t } } \Big | \mathcal { F } _ { s - 1 } \right] = X _ { s } .
$$

Moreover, Assumption 3.1 gives

$$
\left| H _ { s } ^ { \mathrm { d i s t } } \right| \leq 2 G A _ { s } \eta _ { s } \sqrt { Z _ { s } } .
$$

By construction, $\{ H _ { s } ^ { \mathrm { d i s t } } \} _ { s \geq t _ { 0 } }$ is adapted. Since $A _ { s } , \eta _ { s } .$ , and $Z _ { s }$ are $\mathcal { F } _ { s }$ <sub>−1</sub>-measurable, the process

$$
c _ { s } ^ { \mathrm { d i s t } } : = 2 G A _ { s } \eta _ { s } \sqrt { Z _ { s } }
$$

is predictable and almost surely finite. Moreover, the quadratic process and running predictable range associated with $\{ H _ { s } ^ { \mathrm { d i s t } } \} _ { s \geq t _ { 0 } }$ in Theorem 3.3 are precisely $V _ { t } ^ { \mathrm { d i s t , E B } }$ and $B _ { t } ^ { \mathrm { d i s t , E B } }$ , respectively. Finally, since $Z _ { s } \le G ^ { 2 } / \mu ^ { 2 }$ almost surely, condition (37) ensures that $H _ { s } ^ { \mathrm { d i s t } }$ is integrable for every deterministic $s \geq t _ { 0 }$ . Applying Theorem 3.3 therefore gives, with probability at least $1 - \alpha ,$ simultaneously for every $t \geq t _ { 0 }$

$$
\sum _ { s = t _ { 0 } } ^ { t } X _ { s } \leq \mathfrak { B } _ { \alpha } \left( V _ { t } ^ { \mathrm { d i s t , E B } } , B _ { t } ^ { \mathrm { d i s t , E B } } \right) .
$$

Substituting this inequality into the distance decomposition and dividing by $A _ { t }$ yields

$$
Z _ { t + 1 } \leq U _ { t + 1 } ^ { \mathrm { d i s t , E B } } ( \alpha )
$$

simultaneously for every $t \geq t _ { 0 }$ , which proves the result.

Proposition B.2 (Refined confidence sequence for $f ( \bar { x } _ { t } ) - f ( x ^ { \star } ) )$ . Suppose Assumptions 1.1–1.4 and 3.1 hold. Let $t _ { 0 } \geq 4$ and $\eta _ { t } = 2 / ( \mu ( t + 1 ) )$ for every $t \geq t _ { 0 }$ . Let $V _ { t } ^ { \mathrm { s u b , E B } }$ and $B _ { t } ^ { \mathrm { s u b , E B } }$ be defined as in (49)–(50). Then, for every $\alpha \in ( 0 , 1 )$ , the process $U _ { t } ^ { \mathrm { s u b , E B } } ( \alpha )$ defined in (51) satisfies

$$
\begin{array} { r } { \mathbb { P } \left( \forall t \geq t _ { 0 } : \ f ( \bar { x } _ { t } ) - f ( x ^ { \star } ) \leq U _ { t } ^ { \mathrm { s u b , E B } } ( \alpha ) \right) \geq 1 - \alpha . } \end{array}
$$

Proof. Lemma C.3 gives, for every $t \geq t _ { 0 }$

$$
f ( \bar { x } _ { t } ) - f ( x ^ { \star } ) \leq \frac { 1 } { \mu S _ { t } } \sum _ { s = t _ { 0 } } ^ { t } \| g _ { s } \| ^ { 2 } + \frac { \mu t _ { 0 } ( t _ { 0 } - 1 ) } { 4 S _ { t } } Z _ { t _ { 0 } } + \frac { 1 } { S _ { t } } \sum _ { s = t _ { 0 } } ^ { t } E _ { s } .\tag{106}
$$

By (48) and conditional unbiasedness,

$$
H _ { s } ^ { \mathrm { s u b } } - \mathbb { E } \left[ H _ { s } ^ { \mathrm { s u b } } \Big | \mathcal { F } _ { s - 1 } \right] = E _ { s } .
$$

Moreover, the predictable range associated with $H _ { s } ^ { \mathrm { s u b } }$ has running maximum $B _ { t } ^ { \mathrm { s u b , E B } }$ , while the corresponding quadratic process is

$$
\sum _ { s = t _ { 0 } } ^ { t } \left( H _ { s } ^ { \mathrm { s u b } } \right) ^ { 2 } = V _ { t } ^ { \mathrm { s u b , E B } } .
$$

Theorem 3.3 therefore implies that, with probability at least $1 - \alpha$ , simultaneously for every $t \geq t _ { 0 } .$

$$
\sum _ { s = t _ { 0 } } ^ { t } E _ { s } \leq \mathfrak { B } _ { \alpha } \left( V _ { t } ^ { \mathrm { s u b , E B } } , B _ { t } ^ { \mathrm { s u b , E B } } \right) .
$$

Substituting this inequality into (106) proves the proposition.

## C Supporting Lemmas

Lemma C.1 (One-step contractive recursion). Under Assumptions 1.1 and 1.4, the iterates satisfy

$$
Z _ { t + 1 } \leq ( 1 - 2 \mu \eta _ { t } ) Z _ { t } + \eta _ { t } ^ { 2 } \left. g _ { t } \right. ^ { 2 } + 2 \eta _ { t } \langle x ^ { \star } - x _ { t } , \xi _ { t } \rangle , \qquad \forall t \geq t _ { 0 } .\tag{107}
$$

Proof. The proof follows the standard one-step analysis of projected stochastic approximation; see, $\mathrm { e . g . }$ , Nemirovski et al. [32, Section 2.1]. We retain the directional gradient noise term explicitly for the anytime-valid analysis. Since $x ^ { \star } \in \mathcal { X }$ , the nonexpansiveness of the projection gives

$$
Z _ { t + 1 } = \left\| x _ { t + 1 } - x ^ { \star } \right\| ^ { 2 } = \left\| \Pi _ { \mathcal { X } } ( x _ { t } - \eta _ { t } g _ { t } ) - x ^ { \star } \right\| ^ { 2 } \leq \left\| x _ { t } - \eta _ { t } g _ { t } - x ^ { \star } \right\| ^ { 2 } .
$$

Expanding the square,

$$
Z _ { t + 1 } \leq Z _ { t } - 2 \eta _ { t } \langle x _ { t } - x ^ { \star } , g _ { t } \rangle + \eta _ { t } ^ { 2 } \left. g _ { t } \right. ^ { 2 } .
$$

Now, writing $g _ { t } = \nabla f ( x _ { t } ) + \xi _ { t }$

$$
Z _ { t + 1 } \leq Z _ { t } - 2 \eta _ { t } \langle x _ { t } - x ^ { \star } , \nabla f ( x _ { t } ) \rangle + 2 \eta _ { t } \langle x ^ { \star } - x _ { t } , \xi _ { t } \rangle + \eta _ { t } ^ { 2 } \left. g _ { t } \right. ^ { 2 } .
$$

It remains to lower-bound the deterministic gradient term. By the gradient strong monotonicity property,

$$
\langle \nabla f ( x _ { t } ) - \nabla f ( x ^ { \star } ) , x _ { t } - x ^ { \star } \rangle \geq \mu \left. x _ { t } - x ^ { \star } \right. ^ { 2 } = \mu Z _ { t } .
$$

Hence

$$
\langle \nabla f ( x _ { t } ) , x _ { t } - x ^ { \star } \rangle = \langle \nabla f ( x _ { t } ) - \nabla f ( x ^ { \star } ) , x _ { t } - x ^ { \star } \rangle + \langle \nabla f ( x ^ { \star } ) , x _ { t } - x ^ { \star } \rangle .
$$

By the first-order optimality condition at $x ^ { \star }$ , we have $\langle \nabla f ( x ^ { \star } ) , x _ { t } - x ^ { \star } \rangle \geq 0$ . Therefore,

$$
\langle \nabla f ( x _ { t } ) , x _ { t } - x ^ { \star } \rangle \geq \mu Z _ { t } .
$$

Substituting back proves (107).

Lemma C.2 (Last-iterate distance to the optimizer decomposition). Suppose Assumptions 1.1 and 1.4 hold, and let $\{ A _ { t } \} _ { t \ge t _ { 0 } - 1 }$ be defined as in (10). For $t \geq t _ { 0 }$ , define

$$
X _ { t } : = 2 A _ { t } \eta _ { t } \langle x ^ { \star } - x _ { t } , \xi _ { t } \rangle , \qquad \bar { X } _ { t } : = \sum _ { s = t _ { 0 } } ^ { t } X _ { s } .
$$

Then, for every $t \geq t _ { 0 }$

$$
Z _ { t + 1 } \leq \frac { 1 } { A _ { t } } \left[ Z _ { t _ { 0 } } + \sum _ { s = t _ { 0 } } ^ { t } A _ { s } \eta _ { s } ^ { 2 } \left. g _ { s } \right. ^ { 2 } + \bar { X } _ { t } \right] .\tag{108}
$$

Proof. A closely related unrolling of the contractive SGD recursion for the classical stepsize $\eta _ { t } = 1 / ( \mu t )$ appears in Rakhlin et al. [36, Appendix B.7, Lemma $6 ] ;$ the decomposition below records the same telescoping mechanism for general predictable stepsizes. By Lemma C.1, for every $s \geq t _ { 0 }$

$$
Z _ { s + 1 } \leq ( 1 - 2 \mu \eta _ { s } ) Z _ { s } + \eta _ { s } ^ { 2 } \left\| g _ { s } \right\| ^ { 2 } + 2 \eta _ { s } \langle x ^ { \star } - x _ { s } , \xi _ { s } \rangle .
$$

Multiplying by A<sub>s</sub> and using $A _ { s } ( 1 - 2 \mu \eta _ { s } ) = A _ { s - 1 }$ , we obtain

$$
\begin{array} { r } { A _ { s } Z _ { s + 1 } \leq A _ { s - 1 } Z _ { s } + A _ { s } \eta _ { s } ^ { 2 } \vert \vert g _ { s } \vert \vert ^ { 2 } + X _ { s } . } \end{array}
$$

Summing over $s = t _ { 0 } , \ldots , t$ gives

$$
A _ { t } Z _ { t + 1 } - A _ { t _ { 0 } - 1 } Z _ { t _ { 0 } } \leq \sum _ { s = t _ { 0 } } ^ { t } A _ { s } \eta _ { s } ^ { 2 } \left. g _ { s } \right. ^ { 2 } + \bar { X } _ { t } .
$$

Since $A _ { t _ { 0 } - 1 } = 1$ , we obtain

$$
A _ { t } Z _ { t + 1 } \leq Z _ { t _ { 0 } } + \sum _ { s = t _ { 0 } } ^ { t } A _ { s } \eta _ { s } ^ { 2 } \vert \vert g _ { s } \vert \vert ^ { 2 } + \bar { X } _ { t } .
$$

Finally, Assumption 1.4 implies $A _ { t } > 0$ , so dividing by $A _ { t }$ proves (108).

Lemma C.3 (Weighted-average suboptimality decomposition). Suppose Assumption 1.1 holds. Let $t _ { 0 } \geq 4$ and $\eta _ { t } = 2 / ( \mu ( t + 1 ) )$ for every $t \geq t _ { 0 }$ . Define

$$
E _ { t } : = t \langle x ^ { \star } - x _ { t } , \xi _ { t } \rangle , \qquad \bar { E } _ { t } : = \sum _ { s = t _ { 0 } } ^ { t } E _ { s } .
$$

Then, for every $t \geq t _ { 0 }$

$$
f ( \bar { x } _ { t } ) - f ( x ^ { \star } ) \leq \frac { 1 } { \mu S _ { t } } \sum _ { s = t _ { 0 } } ^ { t } \| g _ { s } \| ^ { 2 } + \frac { \mu t _ { 0 } ( t _ { 0 } - 1 ) } { 4 S _ { t } } Z _ { t _ { 0 } } + \frac { 1 } { S _ { t } } \bar { E } _ { t } .
$$

Proof. The proof builds on the weighted telescoping argument of Lacoste-Julien et al. [26, Section 3.2], adapted here to retain the directional gradient noise term pathwise rather than taking expectations. Let

$$
E _ { t _ { 0 } - 1 } : = 0 , \qquad E _ { s } : = s \left. x ^ { \star } - x _ { s } , \xi _ { s } \right. \qquad \bar { E } _ { t } : = \sum _ { s = t _ { 0 } } ^ { t } E _ { s } , \qquad t \geq t _ { 0 } .
$$

Since $x ^ { \star } \in \mathcal { X }$ , the nonexpansiveness of the projection $\mathrm { g i }$ ves

$$
Z _ { t + 1 } = \| x _ { t + 1 } - x ^ { \star } \| ^ { 2 } = \| \Pi _ { \mathcal { X } } ( x _ { t } - \eta _ { t } g _ { t } ) - x ^ { \star } \| ^ { 2 } \leq \| x _ { t } - \eta _ { t } g _ { t } - x ^ { \star } \| ^ { 2 } .
$$

Expanding the square,

$$
Z _ { t + 1 } \leq Z _ { t } - 2 \eta _ { t } \langle x _ { t } - x ^ { \star } , g _ { t } \rangle + \eta _ { t } ^ { 2 } \left. g _ { t } \right. ^ { 2 } .
$$

Writing $g _ { t } = \nabla f ( x _ { t } ) + \xi _ { t }$ and using strong convexity,

$$
\left. \nabla f ( { x } _ { t } ) , { x } _ { t } - { x } ^ { \star } \right. \geq f ( { x } _ { t } ) - f ( { x } ^ { \star } ) + \frac { \mu } { 2 } Z _ { t } ,
$$

we obtain

$$
2 \eta _ { t } \big ( f ( x _ { t } ) - f ( x ^ { \star } ) \big ) \leq \eta _ { t } ^ { 2 } \| g _ { t } \| ^ { 2 } + ( 1 - \mu \eta _ { t } ) Z _ { t } - Z _ { t + 1 } + 2 \eta _ { t } \langle x ^ { \star } - x _ { t } , \xi _ { t } \rangle .
$$

Now substitute $\eta _ { t } = 2 / ( \mu ( t + 1 ) )$ . Since

$$
\frac { \eta _ { t } } { 2 } = \frac { 1 } { \mu ( t + 1 ) } , \qquad \frac { 1 - \mu \eta _ { t } } { 2 \eta _ { t } } = \frac { \mu ( t - 1 ) } { 4 } , \qquad \frac { 1 } { 2 \eta _ { t } } = \frac { \mu ( t + 1 ) } { 4 } ,
$$

it follows that

$$
f ( x _ { t } ) - f ( x ^ { \star } ) \leq \frac { \| g _ { t } \| ^ { 2 } } { \mu ( t + 1 ) } + \frac { \mu } { 4 } ( ( t - 1 ) Z _ { t } - ( t + 1 ) Z _ { t + 1 } ) + \langle x ^ { \star } - x _ { t } , \xi _ { t } \rangle .
$$

Multiplying by t and using $t / ( t + 1 ) \leq 1$ , we obtain

$$
t \big ( f ( x _ { t } ) - f ( x ^ { \star } ) \big ) \leq \frac { 1 } { \mu } \| g _ { t } \| ^ { 2 } + \frac { \mu } { 4 } \big ( t ( t - 1 ) Z _ { t } - t ( t + 1 ) Z _ { t + 1 } \big ) + t \big \langle x ^ { \star } - x _ { t } , \xi _ { t } \big \rangle .
$$

Summing from $s = t _ { 0 }$ to $t ,$ and using

$$
\sum _ { s = t _ { 0 } } ^ { t } ( s ( s - 1 ) Z _ { s } - s ( s + 1 ) Z _ { s + 1 } ) = t _ { 0 } ( t _ { 0 } - 1 ) Z _ { t _ { 0 } } - t ( t + 1 ) Z _ { t + 1 } ,
$$

we obtain

$$
\sum _ { s = t _ { 0 } } ^ { t } s \big ( f ( x _ { s } ) - f ( x ^ { \star } ) \big ) \leq \frac { 1 } { \mu } \sum _ { s = t _ { 0 } } ^ { t } \| g _ { s } \| ^ { 2 } + \frac { \mu } { 4 } \Big ( t _ { 0 } ( t _ { 0 } - 1 ) Z _ { t _ { 0 } } - t ( t + 1 ) Z _ { t + 1 } \Big ) + \bar { E } _ { t } .
$$

Dropping the negative terminal term yields

$$
\sum _ { s = t _ { 0 } } ^ { t } s \big ( f ( x _ { s } ) - f ( x ^ { \star } ) \big ) \leq \frac { 1 } { \mu } \sum _ { s = t _ { 0 } } ^ { t } \| g _ { s } \| ^ { 2 } + \frac { \mu t _ { 0 } ( t _ { 0 } - 1 ) } { 4 } Z _ { t _ { 0 } } + \bar { E } _ { t } .
$$

Dividing by $S _ { t }$ and using the convexity of $f$ we obtain

$$
f ( \bar { x } _ { t } ) - f ( x ^ { \star } ) \leq \frac { 1 } { \mu S _ { t } } \sum _ { s = t _ { 0 } } ^ { t } \| g _ { s } \| ^ { 2 } + \frac { \mu t _ { 0 } ( t _ { 0 } - 1 ) } { 4 S _ { t } } Z _ { t _ { 0 } } + \frac { 1 } { S _ { t } } \bar { E } _ { t } .\tag{109}
$$