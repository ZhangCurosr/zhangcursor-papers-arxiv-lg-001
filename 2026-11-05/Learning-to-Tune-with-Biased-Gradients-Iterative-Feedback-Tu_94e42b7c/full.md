# Learning to Tune with Biased Gradients: Iterative Feedback Tuning of Model-Free Control on Throttle Valves

Felix Mart´ ´ınez-Jimenez´

Sander Miller-Murphy

Tobie Ondoa-Essomba

Emmanuel Witrant

Univ. Grenoble Alpes, CNRS, GIPSA-lab, Grenoble, France

FDMTZ01@OUTLOOK.COM

SANDER.MILLER.MURPHY@GMAIL.COM

ONDOADIMITRI3@GMAIL.COM

EMMANUEL.WITRANT@UNIV-GRENOBLE-ALPES.FR

## Abstract

Iterative feedback tuning (IFT) learns the parameters of a fixed-structure controller by stochastic gradient descent. At each iteration, two closed-loop experiments are run on the plant: the first one measures the tracking error, and the second one, driven by this error, provides the sensitivities from which the gradient of the cost is estimated. This estimate is unbiased for linear plants only. We study the biased case on automotive electronic throttle valves, whose friction and stiction vary between valves and runs, controlled by an intelligent proportional (model-free) controller, which is an incremental PI controller. We show that the mismatch of the controller state at the start of the two experiments, which stiction makes unavoidable, adds an error inversely proportional to the amplitude of the second experiment. On digital twins of ten valves, the IFT gradient is a descent direction in only 69% of the low-cost region, friction being the cause, and all update rules plateau and then drift. On two hardware valves, the best update rule depends on the noise structure of the gradient. On the first valve, where one gradient component is buried in noise, Adam outperforms Gauss–Newton, its normalisation and its momentum each improving one parameter. On the second valve, where both components are consistent, the same normalisation amplifies a direction that Gauss–Newton rightly discounts, and Adam drifts to a controller worse than the initial one. The twins reproduce the mechanism but not this ranking.

Keywords: iterative feedback tuning; biased stochastic gradients; Adam; friction; model-free control; sim-to-real

## 1. Introduction

Data-driven controller tuning casts the design of a fixed-structure controller as the minimisation of a closed-loop cost J(ϑ) that can only be queried through experiments on the plant. Iterative feedback tuning (IFT) (Hjalmarsson et al., 1998; Hjalmarsson, 2002) is the stochastic-gradient member of this family: closed-loop experiments provide at each iteration an estimate gˆ of ∇J, which is unbiased when the plant is linear time-invariant and the noise is zero-mean. Other approaches avoid gradients, either by solving a model-reference identification problem from a single data set, as virtual reference feedback tuning (Campi et al., 2002) and correlation-based tuning (Karimi et al., 2008), or by treating J as a black box, as Bayesian optimisation (Berkenkamp et al., 2016). IFT remains attractive because it needs no model, scales to many parameters and has the structure of the stochastic optimisation methods of machine learning, including adaptive ones such as Adam (Kingma and Ba, 2015). IFT has been applied to the intelligent PID controllers of model-free control in simulation (Baciu et al., 2021) and with Adam-like updates (Baciu and Lazar, 2023). Munteanu et al. (2025) report practical difficulties of its real-time implementation, in particular its sensitivity to noisy gradient estimates and to the step size.

The unbiasedness of ${ \hat { g } } ,$ however, rests on two assumptions that physical plants routinely violate: the closed-loop map must be linear, and the experiments of one iteration must start from the same state. On nonlinear plants, the classical recommendation is a small second experiment that probes the linearisation along the first one (Sjoberg and Agarwal¨ , 1996; Hjalmarsson, 1998). Mechanical systems with Coulomb friction and stiction have no such linearisation, and their equilibrium is a set rather than a point. For a generic biased gradient oracle, Ajalloeian and Stich (2020) show that stochastic gradient descent only converges to a neighbourhood of a stationary point whose size scales with the bias. What is missing is an account, for a concrete plant, of where the bias comes from, how large it is where tuning actually operates, how it interacts with the update rule, and whether a digital twin can predict it.

We address these questions on electronic throttle valves (ETVs), low-cost servos with a preloaded return spring, Coulomb friction and stiction. Their friction is strongly asymmetric, which motivated dedicated models (Pujol et al., 2016) and adaptive or friction-compensating controllers (Canudas de Wit et al., 2001; Jiao et al., 2014), and it varies significantly between valves of the same reference and between repetitions of the same test (Witrant et al., 2023). The controller is the intelligent proportional (iP) controller of model-free control (Fliess and Join, 2013), whose two gains are usually tuned by trial and error (Yahagi and Kajiwara, 2022) and which has been applied to the same benchmark by Belhadjoudja et al. (2023). Preliminary real-time experiments with IFT and an Adam-based variant on this benchmark improved a hand-tuned controller but stalled after a few iterations and showed no benefit of Adam; this paper explains both observations.

Section 2 shows that the initial-state mismatch caused by stiction adds to the IFT gradient an error proportional to $1 / c ,$ where c is the amplitude of the second experiment. Section 3 quantifies the bias on digital twins of ten valves identified from previous data (Witrant et al., 2023) and isolates its cause, and Section 4 compares Gauss–Newton (GN), trust-region, sign and Adam updates on these twins. Section 5 reports hardware experiments on two valves: repeatability, a finite-difference check of the gradient, repeated tuning runs that separate the effects of normalisation and momentum, and a comparison with the twin under the same protocol. The firmware, the host code, the data and the scripts reproducing all figures and tables are publicly available (Mart´ınez-Jimenez et al.´ , 2026).

## 2. Model-free control and iterative feedback tuning

Let y be the measured valve angle, r its reference and $e \ : = \ : y \ : - \ : r$ the tracking error. Following Belhadjoudja et al. (2023), the ultra-local model of the iP controller is $\dot { e } = F + \alpha u$ , where $F$ gathers everything unknown and $\alpha > 0$ is a design parameter. With the backward-Euler estimate $\hat { F } _ { k } = ( e _ { k } - e _ { k - 1 } ) / T _ { s } - \alpha u _ { k - 1 }$ and the control law $\boldsymbol { u } _ { k } = - ( K \boldsymbol { e } _ { k } + \hat { F } _ { k } ) / \alpha$ , where $K > 0$ sets the desired error dynamics $\dot { e } = - K e$ , the controller reads

$$
u _ { k } = u _ { k - 1 } - q _ { 0 } e _ { k } - q _ { 1 } e _ { k - 1 } , \qquad q _ { 0 } = \frac { 1 + T _ { s } K } { \alpha T _ { s } } , \quad q _ { 1 } = - \frac { 1 } { \alpha T _ { s } } .\tag{1}
$$

It is therefore an incremental PI controller, $u = C ( z ) ( r - y )$ with $C ( z ) = { \frac { q _ { 0 } + q _ { 1 } z ^ { - 1 } } { 1 - z ^ { - 1 } } } = K _ { p } +$ $\frac { K _ { i } T _ { s } } { 1 - z ^ { - 1 } }$ , where $K _ { p } = - q _ { 1 } = 1 / ( \alpha T _ { s } )$ and $K _ { i } = ( q _ { 0 } + q _ { 1 } ) / T _ { s } = K / ( \alpha T _ { s } )$ are the proportional and integral gains. The map is invertible, $K = - ( q _ { 0 } + q _ { 1 } ) / ( q _ { 1 } T _ { s } )$ and $\alpha = - 1 / ( q _ { 1 } T _ { s } )$ , and $K , \alpha > 0$ if and only if $q _ { 1 } < 0 < q _ { 0 } + q _ { 1 }$ . With $T _ { s } = 5 0 \mathrm { m s }$ , the hand-tuned controller of the bench, $( K , \alpha ) = ( 5 , 2 . 5 )$ , corresponds to $( q _ { 0 } , q _ { 1 } ) = ( 1 0 , - 8 )$

Because of the spring preload, the IFT experiments are run in deviation around an operating angle $\theta _ { A }$ , from an equilibrium $( \theta _ { A } , u _ { A } )$ reached in closed loop. Experiment 1 applies a reference $r _ { 1 }$ consisting of one up and one down step, $\theta _ { A }  \theta _ { B }  \theta _ { A } ( r _ { 1 } = \theta _ { B }$ on samples 10 to 109 of $N = 2 0 0 .$ , i.e. 5 s out of 10 s), and records the output $y _ { 1 }$ , the error $e _ { 1 } = y _ { 1 } - r _ { 1 }$ and the input deviation $\Delta u _ { 1 } = u _ { 1 } - u _ { A }$ . The cost and its gradient with respect to $\rho = [ q _ { 0 } q _ { 1 } ] ^ { \top }$ are, for $N$ samples,

$$
J ( \rho ) = \frac { 1 } { 2 N } \sum _ { k = 1 } ^ { N } \bigl ( e _ { 1 , k } ^ { 2 } + \lambda \Delta u _ { 1 , k } ^ { 2 } \bigr ) , \qquad \nabla J = \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \bigl ( e _ { 1 , k } \psi _ { k } ^ { y } + \lambda \Delta u _ { 1 , k } \psi _ { k } ^ { u } \bigr ) ,\tag{2}
$$

with $\lambda ~ = ~ 0 . 0 1$ and the output and input sensitivities $\psi ^ { y } ~ = ~ \partial y / \partial \rho$ and $\psi ^ { u } ~ = ~ \partial u / \partial \rho$ For a linear plant modelled by the transfer function $G ( z )$ , with the sensitivity function $S = ( 1 + G C ) ^ { - 1 }$ and the complementary sensitivity function $T = G C S$ , these sensitivities are $\psi ^ { y } = - \Phi T e _ { 1 }$ and $\psi ^ { u } = - \Phi C S e _ { 1 }$ , where $\Phi ( z ) = C ^ { - 1 } \partial C / \partial \rho = ( q _ { 0 } + q _ { 1 } z ^ { - 1 } ) ^ { - 1 } [ 1 z ^ { - 1 } ] ^ { \top }$ (Hjalmarsson et al., 1998). They involve the unknown $T$ and CS applied to a signal, which is exactly what a closedloop experiment computes. To obtain them without a model, experiment 2 therefore feeds the error of experiment 1 back as reference, $r _ { 2 } = \theta _ { 2 } - c e _ { 1 }$ , where $c > 0$ is its excitation amplitude $( c = 1$ and $\theta _ { 2 } = \theta _ { A }$ in standard IFT), and the sensitivities are estimated by

$$
\hat { \psi } ^ { y } = \frac { 1 } { c } \Phi \left( y _ { 2 } - \theta _ { 2 } \right) , \qquad \hat { \psi } ^ { u } = \frac { 1 } { c } \Phi \left( u _ { 2 } - u _ { 2 , 0 } \right) ,\tag{3}
$$

where $u _ { 2 , 0 }$ is the input at the start of experiment 2. The gradient estimate and the Gauss–Newton approximation of the Hessian of J are then

$$
\hat { g } = \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \bigl ( e _ { 1 , k } \hat { \psi } _ { k } ^ { y } + \lambda \Delta u _ { 1 , k } \hat { \psi } _ { k } ^ { u } \bigr ) , \qquad \hat { R } = \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \bigl ( \hat { \psi } _ { k } ^ { y } \hat { \psi } _ { k } ^ { y } { } ^ { \top } + \lambda \hat { \psi } _ { k } ^ { u } \hat { \psi } _ { k } ^ { u } { } ^ { \top } \bigr ) ,\tag{4}
$$

and the standard IFT update is the Gauss–Newton step $\rho _ { i + 1 } = \rho _ { i } - \gamma \hat { R } ^ { - 1 } \hat { g }$ . The pole of Φ is $1 / ( 1 + T _ { s } K )$ , so that the filters are stable if and only if $K > 0$ . Since only deviations are used, $\theta _ { 2 }$ may differ from $\theta _ { A }$ , a special case of the filtered second experiment of Prochazka et al. ´ (2005).

IFT assumes that both experiments start from the same state, which a plant with stiction does not allow. The valve rests at the target angle for any net torque within the stiction band, so that the integral state of the controller at the start of experiment $2 , u _ { 2 , 0 }$ , differs from $u _ { A }$ by an amount δ that depends on the history and is not controlled. Seen from a linear model, the valve is then held by a friction torque equivalent to an input offset $d = - \delta$ , which is released as soon as the valve moves. The following result quantifies the effect of such an offset; H denotes the unit step (Heaviside) sequence.

Proposition 1 Let the plant be $y - \theta _ { 2 } = G ( z ) ( u - u _ { 2 , 0 } + d H )$ during experiment 2, where G is linear and d is a constant offset, the loop being stable. Then the estimates (3) satisfy $\hat { \psi } ^ { y } = $ $\begin{array} { r } { \psi ^ { y } + \frac { d } { c } \Phi G S } \end{array}$ H and $\begin{array} { r } { \hat { \psi } ^ { u } = \psi ^ { u } - \frac { d } { c } \Phi T H } \end{array}$ , and

$$
\hat { g } - \nabla J = \frac { d } { c } b , \qquad b = \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \Bigl ( e _ { 1 , k } ( \Phi G S H ) _ { k } - \lambda \Delta u _ { 1 , k } ( \Phi T H ) _ { k } \Bigr ) .\tag{5}
$$

The proof, given in Appendix A, follows from linearity. The error scales as $1 / c ,$ so that the small second experiment recommended for smooth nonlinear plants amplifies it, whereas a large one drives the valve into the input saturation and the stops. The vector b depends only on $\rho ,$ G and experiment 1: if E $\lfloor d \rfloor \ne 0$ , for instance when the valve tends to stick on the same side, the error is a systematic bias, and if $\mathbb { E } [ d ] = 0$ it is an additional variance that does not decrease with the length of the experiment. Finally, the offset term is a step response, concentrated at low frequencies where the gain of Φ is $1 / ( q _ { 0 } + q _ { 1 } ) = \alpha / K$ , the inverse of the integral gain; the error is thus amplified precisely for the sluggish controllers that a tuning algorithm explores when it reduces K or increases α. In reality the offset is not constant, since the valve may stick again during the experiment, which adds a state-dependent error; Section 3 quantifies the total effect.

The parametrisation matters for coordinate-wise methods. The first update of Adam moves every coordinate by approximately its step size $\mu ,$ whatever the magnitude of the gradient. In $\left( q _ { 0 } , q _ { 1 } \right)$ coordinates, the integral gain $K _ { i } T _ { s } = q _ { 0 } + q _ { 1 } = K / \alpha$ is the small difference of two large numbers, $1 0 - 8 = 2$ at the hand-tuned point, so that a first step with $\mu = 1$ that decreases $q _ { 0 }$ and $q _ { 1 }$ by one unit each cancels the integral action. We therefore optimise $\vartheta = ( \log K , \log \alpha )$ , which keeps $K$ and α positive and makes the steps relative, with $\hat { y } _ { \vartheta } = Q ^ { \top } \hat { g } , \hat { R } _ { \vartheta } = Q ^ { \top } \hat { R } Q$ and $Q = \partial \rho / \partial \vartheta .$ Five update rules are compared. GN, $\vartheta _ { i + 1 } = \vartheta _ { i } - \gamma \hat { R } _ { \vartheta } ^ { - 1 } \hat { g } _ { \vartheta } .$ , is the standard IFT update, and clipped GN limits its step to $\lVert \vartheta _ { i + 1 } - \vartheta _ { i } \rVert \leq 0 . 2$ . The sign step $\vartheta _ { i + 1 } = \vartheta _ { i } - h \mathrm { s i g n } ( \hat { g } _ { \vartheta } )$ (Bernstein et al., 2018) normalises each coordinate but has no memory. Adam (Kingma and Ba, 2015) combines this normalisation with momentum,

$$
m _ { i } = \beta _ { 1 } m _ { i - 1 } + ( 1 - \beta _ { 1 } ) \hat { g } \phi _ { i } , v _ { i } = \beta _ { 2 } v _ { i - 1 } + ( 1 - \beta _ { 2 } ) \hat { g } _ { i , i } ^ { 2 } , \vartheta _ { i + 1 } = \vartheta _ { i } - \mu \frac { m _ { i } / ( 1 - \beta _ { 1 } ^ { i } ) } { \sqrt { v _ { i } / ( 1 - \beta _ { 2 } ^ { i } ) } + \epsilon } ,\tag{6}
$$

where squares, square roots and ratios are taken element-wise, m<sub>0</sub> $= v _ { 0 } = 0 , \beta _ { 1 } = 0 . 9 , \beta _ { 2 } =$ 0.999 and $\epsilon = 1 0 ^ { - 8 }$ . “Adam-GN” applies (6) to the GN direction $\hat { R } _ { \vartheta } ^ { - 1 } \hat { g } _ { \vartheta }$ instead of $\hat { g } _ { \vartheta } ^ { \phantom { \dagger } }$ , as in the preliminary experiments. Comparing the sign step and Adam separates the effects of normalisation and momentum.

## 3. Digital twins and reliability of the gradient

The benchmark of Witrant et al. (2023) consists of throttle valves of the same reference whose return spring opens the valve; a DC motor driven by a PWM signal in [0, 255] closes it, and the angle θ is measured by a magnetic sensor at $T _ { s } = 5 0$ ms. Each valve has a PRBS record and a slow PWM staircase with several up–down cycles. Following Pujol et al. (2016), the shaft dynamics are

$$
\dot { \theta } = \omega , \qquad \dot { \omega } = - a _ { 1 } ( \theta - \theta _ { s } ) - a _ { 2 } ^ { \sigma } \omega - a _ { 3 } ^ { \sigma } \mathrm { s i g n } ( \omega ) + a _ { 4 } v ,\tag{7}
$$

with $v = - \mathrm { P W M }$ and $\sigma \in \{ + , - \}$ the direction of motion. We extended the initial model of Pujol et al. (2016) with a spring equilibrium $\theta _ { s }$ beyond the upper stop (preload), a Karnopp stiction model (Karnopp, 1985) with direction-dependent breakaway levels $a _ { 3 s } ^ { \pm }$ , and inelastic stops. The eleven parameters are fitted by differential evolution followed by the Nelder–Mead (downhill simplex) method on the PRBS and the first staircase cycle, the other cycles being kept for crossvalidation (Appendix B). The simulation RMS errors are $0 . 9 \mathrm { - } 3 . 5 ^ { \circ }$ on the PRBS and $4 . 9 \mathrm { - } 1 3 . 9 ^ { \circ }$ in cross-validation. The latter are dominated by the positions at which the valve sticks, which vary from cycle to cycle in the data themselves: the closed position of one valve varies between $2 ^ { \circ }$ and $3 0 ^ { \circ }$ . The twins are therefore behavioural models of the stick–slip phenomenology.

![](images/e8827df23899b34079ab04d9039b2db3363c5ae244507d9fa8bddd4e329f7541.jpg)

![](images/7d94980bd575af61a235fdee4370246b784f55bac88811d858a9001680daa5a4.jpg)

![](images/2b0d9ee3e5569ccbc3ae1d5ea414ae42e9fd047917ec637f09d81885cf5d79c7.jpg)  
Figure 1: Twins. (a) Cosine between the IFT and true gradients for one valve (⋆: hand-tuned controller; contours: log J). (b) Descent fraction averaged over the ten twins when removing each nonlinearity. (c) Effect of the amplitude c of experiment 2.

On each twin, the IFT gradient $\hat { g } ( \vartheta ) ~ ( c = 1$ , experiment 30–60–30<sup>◦</sup>) is compared with the gradient $\nabla J ( \vartheta )$ of the true cost, obtained by central finite differences, on a $1 0 \times 1 0$ log-spaced grid $\mathcal { G }$ covering $K \in [ 0 . 1 , 3 0 ]$ and $\alpha \in [ 1$ , 20]. The low-cost half of the grid is $\mathcal { G } _ { - } = \{ \vartheta \in \mathcal { G } : J ( \vartheta ) \leq$ median<sub>G</sub> J}, and we define the descent fraction

$$
D = \frac { 1 } { | \mathcal { G } _ { - } | } \sum _ { \vartheta \in \mathcal { G } _ { - } } \mathbf { 1 } \big [ \hat { g } ( \vartheta ) ^ { \top } \nabla J ( \vartheta ) > 0 \big ] ,\tag{8}
$$

i.e. the proportion of low-cost points where an infinitesimal step along −gˆ decreases J. On the linearised twins, the cosine between both gradients exceeds 0.99999 at all 177 points where the loop is stable, which validates the implementation. On the actual twins, the descent fraction is only 0.69 on average (0.54–0.80 across valves), and the wrong directions concentrate in the valley of the cost around the hand-tuned controller (Fig. 1a), i.e. where a tuning algorithm operates.

Removing one nonlinearity at a time identifies the cause (Fig. 1b). The input saturation plays no role (0.70) and symmetrising the damping helps slightly (0.75), whereas removing Coulomb friction and stiction restores a descent fraction of 0.93, and of 1.00 for nine valves out of ten. Consistently with Proposition 1, a smaller second experiment is counter-productive (Fig. 1c): the descent fraction falls to 0.60 for $c = 0 . 5$ , where the valve remains stuck more often, while $c = 5$ drives it into the saturation and the stops (0.63). No amplitude removes the bias.

## 4. Update rules under biased gradients

Each rule is run for 40 iterations from $( K , \alpha ) \ : = \ : ( 5 , 2 . 5 )$ on the ten twins in three scenarios: a nominal one; a “noise” scenario, with a measurement noise of standard deviation $0 . 5 ^ { \circ }$ and friction coefficients scaled by $e ^ { 0 . 2 \xi } , \xi \sim \mathcal { N } ( 0 , 1 )$ , drawn independently for each experiment; and a “harsh”

scenario with $2 ^ { \circ }$ and $e ^ { 0 . 4 \xi }$ . Eight seeds are used per valve. Performance is measured on the noisefree nominal twin by the normalised gap $g _ { i } = ( J ( \vartheta _ { i } ) - J ^ { \star } ) / ( J ( \vartheta _ { 0 } ) - J ^ { \star } )$ , J<sup>⋆</sup> being the minimum over a $3 1 \times 3 1$ grid, and a run is counted as failed if the parameters become inadmissible or the validation error explodes.

For all well-tuned rules, the median gap reaches about 0.5 within 10–15 iterations and then stalls (Table 2 and Fig. 5, Appendix C), which is the behaviour expected from biased stochastic gradient descent (Ajalloeian and Stich, 2020). Continuing the iterations is harmful: after 40 iterations, 19– 46% of the runs end with a higher cost than the initial controller, even without noise. Rejecting the steps that increase the measured cost removes this tail but also stops the progress, since the measured cost is itself corrupted by the friction variability. Large GN steps $( \gamma \geq 0 . 3 )$ fail in 20–51% of the runs; clipping removes the failures but not the drift.

With $\mu = 0 . 0 2$ , Adam never fails, reaches the same median gap as GN and has a much smaller tail, the 90th percentile of $g _ { 4 0 }$ being 1.76 against 3.69 for GN $( \gamma = 0 . 1 )$ under noise, and 2.29 against 6.90 in the harsh scenario. Under moderate noise, Adam is significantly better than GN with $\gamma = 0 . 1$ but not than the sign step, so that its benefit comes from the bounded, normalised step. In the harsh scenario, it also outperforms the sign step, because the first moment averages out when the biased gradient changes sign around the valley, which shrinks the updates and delays the drift (Appendix C). Adam-GN brings no benefit, and far from the optimum the fixed relative step of Adam is a drawback.

Finally, decreasing J did not improve a validation test spanning $1 5 { - } 6 5 ^ { \circ }$ (median RMS ratio 0.94–1.16), because the IFT experiment (30–60<sup>◦</sup>) excites a different friction regime: the IFT experiment must span the operating range of interest, which we do on the bench.

## 5. Hardware experiments

## 5.1. Experiment server and protocol

The firmware turns the Arduino of the benchmark into an experiment server driven by a Python host over the serial link. It implements the iP law (1) with the saturated $u _ { k - 1 }$ in $\hat { F } _ { k }$ , which provides anti-windup, and fixed-rate scheduling. Each run starts with a 3 s closed-loop settling phase at its initial angle, and experiment 2 is generated on board from the stored $e _ { 1 }$ , so that both experiments share the same sampling instants. The host computes the estimates and the updates and, every second iteration, runs the validation test of Belhadjoudja et al. (2023), a $1 5 / 6 5 ^ { \circ }$ square reference of 20 s. The experiments were run on valves 9 and 0 of the benchmark, called A and B below, starting from the hand-tuned controller; a PI controller previously used on this bench (Witrant et al., 2023), equivalent to $( K , \alpha ) = ( 4 . 4 1 , 5 . 0 8 )$ , serves as a second reference.

## 5.2. Repeatability: the cost is set by where the valve sticks

We measure the dispersion of the cost over n repetitions with fixed controller gains by its coefficient of variation $\mathrm { C V } = s _ { J } / \bar { J } ,$ where J<sup>¯</sup> and $s _ { J }$ are the sample mean and standard deviation of the measured costs. With the $3 0 – 6 0 ^ { \circ }$ experiment of the twin study, ten repetitions at the hand-tuned controller give $\mathrm { C V } = 1 9 \%$ on valve A. This dispersion is not due to measurement noise: J is determined by the controller state $u _ { A }$ at the end of the settling phase (correlation −0.90), which spreads over the stiction band with a standard deviation of 17 PWM units for the same angle. The two experiments of one iteration therefore start from integral states that differ by 17.5 PWM units on average, so that the offset d of Proposition 1 is observed directly. Running experiment 1 over the validation range $( 1 5  – 6 5 ^ { \circ } )$ and experiment 2 around $\theta _ { 2 } = 4 0 ^ { \circ }$ , away from both stops, reduces the CV to 6.3%. The price is a less informative gradient. For each of the ten repetitions, we compute the angle between its estimate $\hat { g }$ and the mean of the ten estimates: $0 ^ { \circ }$ means that the estimate points in the mean direction, $9 0 ^ { \circ }$ that a step along it does not move the parameters in the mean direction at all. At $3 0 { - } 6 0 ^ { \circ }$ , the angles are small (median $7 ^ { \circ }$ , maximum $3 3 ^ { \circ } )$ , so that every iteration moves the parameters in nearly the same direction. $\mathrm { A t } 1 5  – 6 5 ^ { \circ }$ , they range from $2 3 ^ { \circ }$ to $9 0 ^ { \circ }$ (median $4 7 ^ { \circ } ) { \mathrm { : } }$ : the direction of a single step is then largely set by the run-to-run dispersion. All subsequent results use the $1 5 / 6 5 ^ { \circ }$ protocol, which also aligns the IFT cost with the validation test.

Valve B behaves differently under the same protocols. Its CV is 5.7% at $3 0 { - } 6 0 ^ { \circ }$ and 4.3% at 15– $6 5 ^ { \circ }$ , and at $1 5 { - } 6 5 ^ { \circ }$ all single-shot gradients lie within $1 0 ^ { \circ }$ of their mean $\hat { g } = [ - 0 . 5 \pm 1 . 7 , - 1 3 . 0 \pm$ 1.9] in (log $K , \log \alpha )$ . The two valves thus offer two regimes of the same plant: on valve A, the K component of the gradient is buried in noise, whereas on valve B both components are consistent from one iteration to the next.

## 5.3. Gradient check on valve A

To check the IFT gradient against a model-free reference, the cost was also measured at $\vartheta \pm h \mathbf { e } _ { j }$ $h = 0 . 3$ , for each coordinate $j$ of $\vartheta ,$ , each evaluation being one run of experiment 1; the central differences $( J ( \vartheta + h \mathbf { e } _ { j } ) - J ( \vartheta - h \mathbf { e } _ { j } ) ) / ( 2 h )$ estimate the gradient of the measured cost, friction effects included. At the hand-tuned and PI controllers of valve A, eight interleaved repetitions of these finite-difference experiments and of the IFT experiments $( c \in \{ 0 . 5 , 1 , 2 \} )$ ) were performed (Fig. 6, Appendix D). At the hand-tuned controller, the finite differences give $\nabla J = \left[ - 4 . 8 \pm \right.$ $1 . 2 , - 5 . 7 \pm 1 . 8 ]$ (mean ± standard error), while IFT with $c = 1$ gives $[ 1 . 0 \pm 1 . 0 , - 5 . 2 \pm 1 . 0 ] \colon$ the α component is correct, but the K component is missed. The cosine between both remains positive (0.64, with a 90% bootstrap interval of $[ 0 . 3 3 , 0 . 8 7 ] ,$ ), yet a step of 0.3 along $- \hat { g } / \lVert \hat { g } \rVert$ does not decrease the measured cost. Eight repetitions of J before and after the step differ by −0.4% only, and a one-sided Welch t-test of the hypothesis that the cost decreased gives a p-value of 0.46, i.e. the decrease is indistinguishable from the run-to-run dispersion: the plateau of Section 4 is observed directly. At the PI controller, farther from the optimum, the IFT direction is reliable (cosine 0.84–0.88 for all c) and the same step decreases J by 5.1% $( p = 0 . 0 7 )$ .

## 5.4. Tuning: the best update rule depends on the valve

On valve A, GN $( \gamma = 0 . 1 )$ and Adam $( \mu = 0 . 0 5 )$ were each run twice for 20 iterations, and the sign step once with $h = 0 . 0 2$ and once with $h = 0 . 0 5 ( \mathrm { F i g . ~ 2 a , c } )$ . The measured J (21–31) is useless as a progress indicator, whereas the validation RMSE is reproducible to within $0 . 0 2 \mathrm { - } 0 . 0 8 ^ { \circ }$ (Table 3, Appendix D) and decreases steadily. Each method converges to its own reproducible region, GN to (6.9, 2.9) and Adam to $( 6 . 2 – 6 . 4 , 3 . 4 – 3 . 6 )$ , with a validation RMSE over the last three evaluations of 6.84–6.91<sup>◦</sup> for GN and $6 . 7 4 \substack { - 6 . 7 5 ^ { \circ } }$ for Adam, from $7 . 2 0 – 7 . 2 7 ^ { \circ }$ initially.

These runs separate the two ingredients of Adam. The α component of $\hat { g }$ is consistently negative, but $\hat { R } ^ { - 1 }$ shrinks this direction, so that GN moves log α by only 0.007 per iteration on average, against 0.016–0.019 for Adam and a steady increase for the sign step, which has no momentum: the normalisation frees the direction that GN under-weights. The K component of ${ \hat { g } } ,$ on the other hand, is dominated by noise (Fig. 6). GN reverses its K step in 37–53% of the iterations and travels 2.3–2.6 times the net distance in log K, and the sign step with $h = 0 . 0 2$ performs a random walk $( K \in [ 4 . 8 , 5 . 2 ]$ , path ratio 10), whereas Adam extracts the weak drift (reversals 11–32%, path ratio $1 . 2 \mathrm { - } 1 . 4 )$ . A larger normalised step does not reproduce this behaviour: with $h = 0 . 0 5$ , the sign step reaches $\alpha = 3 . 3 7$ within six iterations, as Adam does, but $K$ wanders between 4.8 and 6.1 (42% reversals) and the final RMSE $( 6 . 8 5 ^ { \circ } )$ is that of GN. In this low signal-to-noise regime, Adam thus combines a conditioning effect of the normalisation on one coordinate and a variance-reduction effect of the momentum on the other. The improvement of the quadratic criterion comes with more overshoot, however (Fig. 3a): Adam gives the lowest RMSE but 29% overshoot on the upward steps, against 20% and the fastest settling for GN (Table 3). A masked criterion that ignores the transient (Lequin et al., 2003) would trade these differently.

On valve B, the ranking is reversed (Fig. 2b,c and Table 3). GN was run three times and converged each time to the same controller, (6.06, 3.36), (6.25, 3.40) and (6.15, 3.40), whose validation RMSE is $6 . 7 1 ^ { \circ }$ , against $7 . 2 5 ^ { \circ }$ for the hand-tuned controller. Adam was run twice and drifted both times to (3.0, 5.8), whose RMSE, $7 . 4 6 ^ { \circ }$ , is worse than that of the hand-tuned controller, although its best iterate along the way, (4.27, 4.42), is as good as GN (6.72<sup>◦</sup>). With a low integral gain $K / \alpha \approx 0 . 5$ , the final Adam controller stays above the set point for about half a second after the upward step, instead of a brief spike for GN, and hits the lower stop after the downward step (Fig. 3b). The drift is not an artefact of the validation test. Ten repetitions of the IFT experiment at both end points give $J = 2 0 . 6 \pm 0 . 9$ at the GN end point and $2 1 . 6 \pm 1 . 1$ at the Adam end point (Welch test, $p = 0 . 0 3 )$ , and the IFT gradient at the Adam end point, $[ - 1 . 0 \pm 1 . 8 , + 7 . 8 \pm 2 . 1 ]$ , correctly points back towards a larger $K$ and a smaller α. Adam has thus overshot a region that the gradient identifies correctly. On this valve, the K component of $\hat { g }$ is small but consistently positive (in 80–85% of the iterations, mean +2 against about −11 for the α component). GN, through the coupling in $\hat { R } ^ { - 1 }$ , increases K by about 0.01 per iteration despite the sign of $\hat { g } _ { K }$ , whereas a coordinate-wise normalisation gives this small component the same weight as the α component. The sign step with $h = 0 . 0 5$ isolates this effect: without momentum, it also decreases $K$ steadily, to 3.0, and its validation RMSE rises from 6.75 to $7 . 2 2 ^ { \circ }$ , while α oscillates around 5, where the α component of $\hat { g }$ changes sign. Adam follows the same path and, because of its momentum, carries α beyond this point, to 5.8. On valve B, the normalisation is therefore the cause of the drift, and the momentum amplifies it.

## 5.5. Twin versus hardware

The protocol was also run on the twins of both valves, with $0 . 1 ^ { \circ }$ noise and ±20% friction variability. The twin of valve A reproduces the cost level of the $3 0 { - } 6 0 ^ { \circ }$ experiment $( J = 1 7 . 6$ against 17.0), the ranking of the controllers of Table 3, their overshoot and the cosines of Fig. 6a, but not the structure of the run-to-run variability (CV of 6% and 16% at 30/60 and 15/65, against 19% and 6% on hardware). On this twin, Adam degrades the validation error by decreasing K, as on valve B. The twin of valve B fails more clearly: at the hand-tuned controller, it predicts a stick–slip limit cycle between 36 and $9 0 ^ { \circ }$ around the $6 5 ^ { \circ }$ set point, which the valve does not exhibit, and hence a validation RMSE of $1 2 . 6 ^ { \circ }$ instead of $7 . 3 ^ { \circ }$ . It ranks the Adam end point first $( 6 . 7 ^ { \circ } )$ and the GN end point last among the tuned controllers (7.3<sup>◦</sup>), the reverse of the hardware. The twins thus expose the mechanism but cannot select the update rule for a given unit.

(a) valve A  
![](images/b44e5b162eebaa5fe44f25c543b0155dfe85b6e06bc05f7499185ba9e30f429f.jpg)

(b) valve B  
![](images/a6e72e217969d2af1d22d54a3e9add72b652e7f9cc9e72eb0d686116d14ffb75.jpg)

(c) (α, K); solid A, dashed B  
![](images/a3349b5a42dd5525e52ebf5c762f6e9c9a4ee754363f28563626758ba859f2c0.jpg)  
Figure 2: Hardware tuning runs (15/65<sup>◦</sup>, 20 iterations): validation RMSE every second iteration on valve A (a) and valve B (b), and trajectories in the $( \alpha , K )$ plane $( \mathrm { c } ; \star \colon$ hand-tuned start; markers: end points).

(a) valve A  
![](images/46bfe84a201bb3d5f47cce3759fac1f1c00f7d4e64d686399bf290bf4c0e401e.jpg)

(b) valve B  
![](images/83f41ba1edcd74874ad59ef8b644edbf4e86d68dbe5c9da894da2e1d5cb4bcc4.jpg)  
time (s)  
Figure 3: Closed-loop responses during the validation test (one up and one down step) for the handtuned controller, the end points of GN and Adam, and, on valve B, the best Adam iterate; $( K , \alpha )$ in parentheses.

## 6. Discussion and conclusion

IFT is stochastic gradient descent with an experimental oracle, and on a plant with stiction this oracle is biased. On throttle valves, the bias stems from the initial-state mismatch that stiction imposes on the two experiments of one iteration (Proposition 1); it was quantified on digital twins and observed on hardware. As for any biased stochastic gradient method, progress stalls and continuing may degrade the controller, so that a stopping rule based on a separate validation test, rather than on the noisy IFT cost, is indispensable; on valve B, it would have stopped Adam close to its best evaluation, at the level reached by GN.

Two lessons seem transferable to other data-driven tuning problems on mechanical plants. The first is that the design of the experiment matters more than the update rule: spanning the operating range of interest and running the second experiment away from the stops reduced the dispersion of the cost from 19 to 6% on valve A and made the IFT cost consistent with the validation criterion. The second is that adaptive updates are neither good nor bad in themselves: their coordinate-wise normalisation replaces the GN metric, which helps when that metric discounts a reliable direction and hurts when it discounts an unreliable one, and their momentum pays off only when a weak trend is buried in noise. Two valves of the same reference and the same protocol produced both situations, and their twins did not predict which one would occur; the twin of valve B even ranked the controllers in reverse order.

The hardware study is limited to two valves and two or three runs per method and valve, and the proposition covers a constant offset only. Natural continuations are a gradient-free baseline such as Bayesian optimisation (Berkenkamp et al., 2016), friction-aware estimators (dithering, repeated settling, or a correction of (5) with an estimate of d), and masked criteria (Lequin et al., 2003) to reduce the overshoot.

## References

A. Ajalloeian and S. U. Stich. On the convergence of SGD with biased gradients. arXiv preprint arXiv:2008.00051, 2020.

A. Baciu and C. Lazar. Iterative feedback tuning of model-free intelligent PID controllers. Actuators, 12(2):56, 2023. doi: 10.3390/act12020056.

A. Baciu, C. Lazar, and C.-F. Caruntu. Iterative feedback tuning of model-free controllers. In Proceedings of the 25th International Conference on System Theory, Control and Computing (ICSTCC), pages 467–472, 2021.

M. C. Belhadjoudja, M. Maghenem, and E. Witrant. On the intelligent proportional controller applied to linear systems. In Proceedings ofthe 62nd IEEE Conference on Decision and Control (CDC), pages 938–943, 2023.

F. Berkenkamp, A. P. Schoellig, and A. Krause. Safe controller optimization for quadrotors with Gaussian processes. In Proceedings of the IEEE International Conference on Robotics and Automation (ICRA), pages 491–496, 2016.

J. Bernstein, Y.-X. Wang, K. Azizzadenesheli, and A. Anandkumar. signSGD: compressed optimisation for non-convex problems. In Proceedings of the 35th International Conference on Machine Learning (ICML), volume 80 of Proceedings of Machine Learning Research, pages 560–569, 2018.

M. C. Campi, A. Lecchini, and S. M. Savaresi. Virtual reference feedback tuning: a direct method for the design of feedback controllers. Automatica, 38(8):1337–1346, 2002. doi: 10.1016/S0005-1098(02)00032-8.

C. Canudas de Wit, I. Kolmanovsky, and J. Sun. Adaptive pulse control of electronic throttle. In Proceedings of the 2001 American Control Conference, volume 4, pages 2872–2877, 2001. doi: 10.1109/ACC.2001.946335.

M. Fliess and C. Join. Model-free control. International Journal of Control, 86(12):2228–2252, 2013. doi: 10.1080/00207179.2013.810345.

H. Hjalmarsson. Control of nonlinear systems using iterative feedback tuning. In Proceedings of the 1998 American Control Conference, pages 2083–2087, 1998.

H. Hjalmarsson. Iterative feedback tuning – an overview. International Journal ofAdaptive Control and Signal Processing, 16(5):373–395, 2002. doi: 10.1002/acs.714.

H. Hjalmarsson, M. Gevers, S. Gunnarsson, and O. Lequin. Iterative feedback tuning: theory and applications. IEEE Control Systems Magazine, 18(4):26–41, 1998. doi: 10.1109/37.710876.

X. Jiao, J. Zhang, and T. Shen. An adaptive servo control strategy for automotive electronic throttle and experimental validation. IEEE Transactions on Industrial Electronics, 61(11):6275–6284, 2014. doi: 10.1109/TIE.2014.2311398.

A. Karimi, M. Butcher, and R. Longchamp. Model-free precompensator tuning based on the correlation approach. IEEE Transactions on Control Systems Technology, 16(5):1013–1020, 2008.

D. Karnopp. Computer simulation of stick-slip friction in mechanical dynamic systems. Journal of Dynamic Systems, Measurement, and Control, 107(1):100–103, 1985. doi: 10.1115/1.3140698.

D. P. Kingma and J. Ba. Adam: a method for stochastic optimization. In Proceedings of the 3rd International Conference on Learning Representations (ICLR), 2015.

O. Lequin, M. Gevers, M. Mossberg, E. Bosmans, and L. Triest. Iterative feedback tuning of PID parameters: comparison with classical tuning rules. Control Engineering Practice, 11(9):1023– 1033, 2003. doi: 10.1016/S0967-0661(02)00303-9.

F. Mart´ınez-Jimenez, S. Miller-Murphy, T. Ondoa-Essomba, and E. Witrant. Learning to tune with´ biased gradients: code, firmware and data for iterative feedback tuning of model-free control on throttle valves. Zenodo, 2026. URL https://doi.org/10.5281/zenodo.23493908.

M.-D. Munteanu, D.-I. Pintilie, and C. Lazar. Practical issues in the implementation of IFT algorithm for model-free controllers. In Proceedings of the 25th International Conference on Control Systems and Computer Science (CSCS), pages 80–87, 2025.

H. Prochazka, M. Gevers, B. D. O. Anderson, and C. Ferrera. Iterative feedback tuning for robust ´ controller design and optimization. In Proceedings of the 44th IEEE Conference on Decision and Control and European Control Conference, pages 3602–3607, Seville, Spain, 2005. doi: 10.1109/CDC.2005.1582721.

G. Pujol, Y. Vidal, L. Acho, and A. N. Vargas. Asymmetric modelling and control of an electronic throttle. International Journal ofNumerical Modelling: Electronic Networks, Devices and Fields, 29(2):192–204, 2016. doi: 10.1002/jnm.2063.

J. Sjoberg and M. Agarwal. Model-free repetitive control design for nonlinear systems. In¨ Proceedings of the 35th IEEE Conference on Decision and Control, volume 4, pages 2824–2829, Kobe, Japan, 1996.

E. Witrant, I. D. Landau, and M.-P. Vaillant. A data-driven control methodology applied to throttle valves. Control Engineering Practice, 139:105634, 2023.

S. Yahagi and I. Kajiwara. Non-iterative data-driven tuning of model-free control based on an ultra-local model. IEEE Access, 10:72773–72784, 2022.

## Appendix A. Proof of Proposition 1

Write all signals of experiment 2 in deviation from $( \theta _ { 2 } , u _ { 2 , 0 } ) \colon \tilde { y } = y _ { 2 } - \theta _ { 2 } , \tilde { u } = u _ { 2 } - u _ { 2 , 0 }$ and $\tilde { r } = r _ { 2 } - \theta _ { 2 } = - c e _ { 1 }$ . Since $u _ { 2 , 0 }$ is the initial value of the integral state and the loop starts at rest with $e = 0$ , the incremental law (1) gives $\tilde { u } = C ( \tilde { r } - \tilde { y } )$ with zero initial conditions, while the plant gives $\tilde { y } = G ( \tilde { u } + d H )$ . Eliminating u˜ yields $\tilde { y } = T \tilde { r } + G S d H$ and $\tilde { u } = C S \tilde { r } - T d H$ . Applying $\Phi / c$ and using $\psi ^ { y } = - \Phi T e _ { 1 }$ and $\psi ^ { u } = - \Phi C S e _ { 1 }$ gives the expressions of $\hat { \psi } ^ { y }$ and ${ \hat { \psi } } ^ { u }$ . Since $\hat { g }$ is obtained by inserting $\hat { \psi } ^ { y }$ and $\hat { \psi } ^ { u }$ in the right-hand side of (2), which is linear in them, (5) follows. A measurement noise n in experiment 2 adds $\textstyle { { \frac { 1 } { c } } \Phi S n }$ and $- { \frac { 1 } { c } } \Phi C S n$ in the same way, which is the classical result of Hjalmarsson et al. (1998); the offset term is a deterministic, low-frequency analogue of this noise term, with the same 1/c scaling, but not zero-mean. □

## Appendix B. Digital twins

The model (7) is integrated with 50 sub-steps per sampling period and a zero-order hold on v. When $\omega = 0$ , the valve stays at rest as long as the net torque $- a _ { 1 } ( \theta - \theta _ { s } ) + a _ { 4 } v$ lies in $[ - a _ { 3 s } ^ { - } , a _ { 3 s } ^ { + } ]$ . A reversal of ω caused by the friction term within a sub-step is replaced by a stop, whereas a reversal caused by the other torques is integrated normally, and the stops are inelastic. The identification criterion is the sum of the RMS simulation errors on the PRBS (511 samples) and on the first staircase cycle, the latter with weight 0.5. One valve has no staircase record and is identified on the PRBS only; it is the outlier of the ablation study. The identified friction asymmetry changes sign between valves, which indicates that the preload and the friction levels are only partially identifiable. Fig. 4 shows a typical fit and Table 1 gives the parameters of the ten twins; the large spread of the friction parameters, sometimes over two orders of magnitude, reflects both the dispersion of the valves and their partial identifiability. Valves A and B of Section 5 are valves 9 and 0.

Table 1: Parameters of the twins (angles in degrees, time in s, input in PWM units) and RMS simulation errors (<sup>◦</sup>) on the PRBS, on the staircase cycle used for identification (fit) and on the cross-validation cycles (CV); valve 6 has no staircase record and was identified on the PRBS only.
<table><tr><td>valve</td><td>a1</td><td> $\theta _ { s }$ </td><td> $a _ { 2 } ^ { + }$ </td><td> $a _ { 2 } ^ { - }$ </td><td> $a _ { 3 } ^ { + }$ </td><td> $a _ { 3 } ^ { - }$ </td><td> $a _ { 3 s } ^ { + }$ </td><td> $a _ { 3 s } ^ { - }$ </td><td>a4</td><td> $\theta _ { \mathrm { { m i n } } }$ </td><td> $\theta _ { \mathrm { m a x } }$ </td><td>|PRBS</td><td>fit</td><td>CV</td></tr><tr><td>0|</td><td>63</td><td>197</td><td>52</td><td>37</td><td>816</td><td>13377</td><td>1154</td><td>16949</td><td>364</td><td>1.3</td><td>90|</td><td>2.1</td><td>4.9</td><td>10.1</td></tr><tr><td>1</td><td>1655</td><td>177</td><td>1000</td><td>357</td><td>133820</td><td>87401</td><td>187340</td><td>108673</td><td>4200</td><td>0.0</td><td>81</td><td>2.7</td><td>6.1</td><td>6.6</td></tr><tr><td>2</td><td>59</td><td>107</td><td>41</td><td>27</td><td>619</td><td>9099</td><td>1724</td><td>9297</td><td>186</td><td>0.0</td><td>82</td><td>2.6</td><td>6.7</td><td>13.1</td></tr><tr><td>3</td><td>82</td><td>101</td><td>71</td><td>15</td><td>1966</td><td>94</td><td>5898</td><td>107</td><td>73</td><td>1.6</td><td>81</td><td>3.5</td><td>9.3</td><td>10.5</td></tr><tr><td>5</td><td>44</td><td>162</td><td>23</td><td>11</td><td>3266</td><td>230</td><td>4245</td><td>331</td><td>78</td><td>13</td><td>82</td><td>2.6</td><td>6.5</td><td>10.5</td></tr><tr><td>6</td><td>238</td><td>263</td><td>1000</td><td>33</td><td>15545</td><td>45951</td><td>35261</td><td>46232</td><td>1284</td><td>0.0</td><td>79</td><td>0.9</td><td></td><td></td></tr><tr><td>7</td><td>313</td><td>82</td><td>190</td><td>127</td><td>2784</td><td>50544</td><td>4126</td><td>81420</td><td>869</td><td>13</td><td>79</td><td>2.1</td><td>7.9</td><td>13.2</td></tr><tr><td>9</td><td>69</td><td>88</td><td>36</td><td>20</td><td>42</td><td>4482</td><td>72</td><td>7371</td><td>112</td><td>0.4</td><td>81</td><td>2.7</td><td>6.0</td><td>13.8</td></tr><tr><td>10</td><td>85</td><td>5168</td><td>67</td><td>22</td><td>6737</td><td>716</td><td>10614</td><td>1006</td><td>171</td><td>0.0</td><td>81</td><td>2.84.5</td><td></td><td>9.2</td></tr><tr><td>11</td><td>32</td><td>278</td><td>49</td><td>45</td><td>3397</td><td>5263</td><td>5027</td><td>6081</td><td>225</td><td>0.6</td><td>90</td><td></td><td>1.73.3</td><td>4.9</td></tr></table>

PRBS (identification)  
![](images/3b956f8fc670353b7f263abb2be9afb77dba39e8fe456c38f41ca1ce6369d114.jpg)

PWM staircase (left: identification, right: cross-validation)  
![](images/0db79c185393384a666ebf1188a4a7658402f98f496b520ab8c61b9f4839ef1d.jpg)  
Figure 4: Twin of one valve: PRBS used for identification (top) and PWM staircase (bottom; identification on the left of the dashed line, cross-validation on the right).

## Appendix C. Additional results on the twins

Table 2: Update rules on the ten twins after 40 iterations (80 runs per cell): failure rate, median and 90th percentile of the normalised cost gap $( g = 1 )$ : no improvement; $g = 0 \colon$ optimum).
<table><tr><td>update</td><td>h</td><td>fail</td><td>noise  $\tilde { g } _ { 4 0 }$ </td><td> $g _ { 4 0 } ^ { 9 0 }$ </td><td>fail</td><td>harsh  $\tilde { g } _ { 4 0 }$ </td><td> $g _ { 4 0 } ^ { 9 0 }$ </td></tr><tr><td>GN</td><td>0.1</td><td>0%</td><td>0.61</td><td>3.69</td><td>0%</td><td>0.81</td><td>6.90</td></tr><tr><td>GN</td><td>0.5</td><td>36%</td><td>0.54</td><td>9.10</td><td>51%</td><td>0.82</td><td>13.14</td></tr><tr><td> $\mathrm { G N } , \Vert \Delta \Vert \leq 0 . 2$ </td><td>0.3</td><td>0%</td><td>0.85</td><td>6.81</td><td>0%</td><td>0.76</td><td>11.75</td></tr><tr><td>sign step</td><td>0.02</td><td>0%</td><td>0.62</td><td>2.72</td><td>0%</td><td>0.62</td><td>2.72</td></tr><tr><td>ADAM</td><td>0.02</td><td>0%</td><td>0.50</td><td>1.76</td><td>0%</td><td>0.50</td><td>2.29</td></tr><tr><td>ADAM</td><td>0.05</td><td>0%</td><td>0.57</td><td>3.05</td><td>0%</td><td>0.79</td><td>4.26</td></tr><tr><td>ADAM on  $\hat { R } ^ { - 1 } \hat { g }$ </td><td>0.05</td><td>0%</td><td>0.53</td><td>3.66</td><td>0%</td><td>0.68</td><td>4.27</td></tr></table>

The initial controller changes the picture. From an aggressive controller $( K = 3 0 , \alpha = 1 . 2 )$ GN reaches a median gap of 0.22 in 25 iterations, and 0.01 with clipping, against 0.88 for Adam with $\mu = 0 . 0 2$ and 0.04 with $\mu = 0 . 0 5$ . From a sluggish one $( K = 0 . 3 , \alpha = 1 5 )$ , unclipped GN fails in 38% of the runs, while clipped GN and Adam with $\mu = 0 . 0 5$ reach 0.09 and 0.12. The parametrisation matters as well: applied in $\left( q _ { 0 } , q _ { 1 } \right)$ coordinates, Adam-GN fails within ten iterations for 60% of the twins with $\mu = 1$ and for all of them with $\mu = 3 0$ , by cancelling the integral action.

![](images/af26b42627f079998d1c8177bd6dca92f2c0f4ad9786802e026431acc1481d59.jpg)  
Figure 5: Normalised cost gap along the iterations on the twins (solid: median over valves and seeds; dotted: 90th percentile).

An acceptance test that rejects any step increasing the measured cost leaves a median gap of 0.84–1.00 after 40 iterations, i.e. it stops the progress almost entirely.

The statistical comparisons of Section 4 are paired Wilcoxon signed-rank tests over valves and seeds on the gap averaged over the last five iterations. Under noise, the median difference between Adam $( \mu = 0 . 0 2 )$ and GN $( \gamma = 0 . 1 ) \mathrm { i s } - 0 . 0 6 ( p < 1 0 ^ { - 3 } )$ , while the differences with the sign step $( p = 0 . 2 0 )$ and with GN at $\gamma = 0 . 0 5 ( p = 0 . 9 9 )$ are not significant. In the harsh scenario, Adam improves on the sign step by $- 0 . 0 9 ( p < 1 0 ^ { - 3 } )$ and on GN at $\gamma = 0 . 0 5$ by only −0.01.

## Appendix D. Hardware protocol and additional results

The set-up consists of a throttle valve with a single return spring that opens it, a DC motor driven by a 10 A H-bridge shield, an Arduino Mega 2560 and a magnetic angle sensor calibrated so that the closed stop reads $0 ^ { \circ } ;$ ; the PWM saturates at 255. The worst sampling delay observed with the firmware was $1 2 \mu \mathrm { s }$ , with no overrun.

The firmware accepts a RUN command whose arguments are the type of run (experiment 1, experiment 2 generated on board as $r _ { 2 } = \theta _ { 2 } - c e _ { 1 }$ , validation, or hold), $K , \alpha , c , \theta _ { A } , \theta _ { B } , N$ , the settling time, $u _ { \mathrm { m a x } }$ and $\theta _ { 2 } .$ , and a CAL command for the sensor calibration. Each run starts with 3 s of closed-loop settling at its initial angle. Experiment 1 follows $\theta _ { A }  \theta _ { B }  \theta _ { A }$ over $N = 2 0 0$ samples (10 s), with $3 0 / 6 0 ^ { \circ }$ in the twin study and $1 5 / 6 5 ^ { \circ }$ with $\theta _ { 2 } = 4 0 ^ { \circ }$ in the hardware study, and $\lambda = 0 . 0 1$

For the gradient check, each controller is evaluated by eight interleaved repetitions of the IFT experiments for $c \in \{ 0 . 5 , 1 , 2 \}$ and of the four finite-difference experiments (±0.3 in log K and log α); the cosines are given with 2000-sample bootstrap intervals, and the descent test compares eight repetitions of J at ϑ and at $\vartheta - 0 . 3 \hat { g } / \| \hat { g } \|$ with a one-sided Welch test. The tuning runs have 20 iterations with a validation test every second iteration, with $\gamma = 0 . 1$ for GN, $\mu = 0 . 0 5$ for Adam and $h \in \{ 0 . 0 2 , 0 . 0 5 \}$ for the sign step. The path ratio is $\sum _ { i } | \Delta$ log $K _ { i } | / |$ log K<sub>20</sub> − log $K _ { 0 } |$ , and the reversal rate is the fraction of iterations at which the sign of ∆ log K changes.

Fig. 6 details the gradient check of Section 5 on valve A, and Table 3 gives the results of the validation tests on both valves and on their twins.

(a) IFT vs finite diferences  
![](images/002970f0684a364281c2abe6a093151c4962d16ce7b716afaddcf61b93ed4605.jpg)

(b) step −0.3 g/̂ ‖g‖̂  
![](images/d95e7225ddc5098d71144d3e8a77af9fca13dbcebfa8f7f77695c52aad50efbb.jpg)  
Figure 6: Hardware, valve A, $1 5 / 6 5 ^ { \circ }$ . (a) Cosine between the IFT gradient and the finite-difference gradient of the measured cost (bars: 90% bootstrap intervals; crosses: twin of the same valve under the same protocol). (b) Measured change of J after a step of 0.3 along $- \hat { g } / \| \hat { g } \| ( c = 1 )$ and p-value of a one-sided Welch test.

Table 3: Validation test on hardware (valve A: 3 repetitions; valve B: 6 repetitions, end points of two runs pooled): RMSE $( { \mathrm { m e a n } } \pm { \mathrm { s t d } } )$ , mean overshoot and 2% settling time of the upward steps, mean steady-state error; last column: twin of the same valve.
<table><tr><td>valve</td><td>controller  $( K , \alpha )$ </td><td>RMSE (°)</td><td>overshoot (%)</td><td>settling (s)</td><td> $| e _ { \infty } |$  (°)</td><td>twin RMSE</td></tr><tr><td rowspan="4">A</td><td>hand-tuned (5.00, 2.50)</td><td>7.26±0.06</td><td>7</td><td>1.59</td><td>0.7</td><td>7.78</td></tr><tr><td>GN, end (6.96, 2.92)</td><td>6.82±0.02</td><td>20</td><td>0.64</td><td>0.4</td><td>7.65</td></tr><tr><td>Adam, end (6.16, 3.63)</td><td>6.73±0.03</td><td>29</td><td>1.62</td><td>1.0</td><td>7.56</td></tr><tr><td>PI 2021 (4.41, 5.08)</td><td>7.01±0.08</td><td>18</td><td>2.05</td><td>1.7</td><td>7.76</td></tr><tr><td rowspan="5">B</td><td>hand-tuned (5.00, 2.50)</td><td>7.25±0.03</td><td>8</td><td>1.24</td><td>0.3</td><td>12.57</td></tr><tr><td>GN, end (6.15, 3.40)</td><td>6.71±0.03</td><td>36</td><td>0.69</td><td>0.3</td><td>7.27</td></tr><tr><td>Adam, best (4.27, 4.42)</td><td>6.72±0.06</td><td>39</td><td>1.34</td><td>0.5</td><td>7.00</td></tr><tr><td>Adam, end (3.02, 5.78)</td><td>7.46±0.06</td><td>26</td><td>1.82</td><td>0.4</td><td>6.68</td></tr><tr><td>PI 2021 (4.41, 5.08)</td><td>7.14±0.08</td><td>40</td><td>1.70</td><td>0.5</td><td>6.73</td></tr></table>