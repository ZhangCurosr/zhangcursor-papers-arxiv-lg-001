# Mixed-Integer Nonlinear Differentiable Predictive Control for Underground Pumped Hydro Energy Storage Systems

Honghui Zheng<sup>1</sup>, Jan Boldock´ y´<sup>2</sup>, Yury Dvorkin<sup>1</sup> and Jan Drgo´ naˇ <sup>1</sup>

Abstract— This paper extends Mixed-Integer Differentiable Predictive Control (MI-DPC) to multi-modal discrete decisions and nonconvex polynomial dynamics arising in Underground Pumped Hydro Energy Storage Systems (UPHES). A neural policy mapping problem parameters to continuous setpoints and integer mode selections via a Gumbel-Softmax layer is trained in a self-supervised manner by differentiating the expectation of the finite horizon control objective through the nonlinear dynamics model. Three methodological contributions enable this extension: a parallel differentiable simulator that preserves gradient magnitude, a Transformer encoder that captures long-range temporal dependencies, and a Gumbel-Softmax temperature annealing schedule that regularizes the combinatorial search. We demonstrate the framework on dayahead scheduling of a UPHES, a large-scale mixed-integer optimal control problem with nonlinear unit performance curves and volume-head coupling. MI-DPC achieves only 1.6% suboptimality relative to a piecewise mixed-integer quadratic programming baseline, while providing five orders of magnitude speedup in online scheduling time.

## I. INTRODUCTION

Parametric mixed-integer nonlinear programs (MINLPs) with nonlinear dynamics and discrete mode decisions arise broadly in optimal control problems, yet due to their NPhard [1] nature remain computationally prohibitive for realtime solution, even with state-of-the-art solvers. This paper addresses a representative instance: day-ahead scheduling of an Underground Pumped Hydro Energy Storage (UPHES) system [2], [3], where at each of 24 hourly steps the unit selects among three mutually exclusive modes (pump, idle, turbine) and determines continuous power setpoints subject to polynomial unit performance curves (UPCs), cubic volume–head coupling, and time-varying electricity prices.

Existing approaches sacrifice model fidelity for tractability. Mixed-integer programming (MIP) formulations replace nonlinear mappings with piecewise-linear approximations [4] or their chance-constrained extensions [5], and global affine surrogates [6], enabling branch-and-bound solvers at the cost of systematic error that grows in high-curvature regions of the dynamics. Alternative strategies like Bayesian optimization [7] or multi-fidelity simulators [6] improve solution accuracy but require computational budgets incompatible with real-time operation. A decision-focused learning framework [8] trains a neural network to guide the recursive linearization of the MINLP into differentiable quadratic programs, yet still solves an optimization problem at inference and inherits integer-variable errors from initializations.

From a control perspective, the scheduling problem is a parametric MINLP: given an initial state and price forecast, compute a state-input trajectory subject to nonlinear dynamics and integer constraints. Classical explicit model predictive control (MPC) enumerates polyhedral regions of the parameter space [9], but the number of regions grows exponentially with horizon and constraint count [10]. Neural approximations of MPC learn the parameter-to-control map via supervised learning [11], [12], but require data labels of pre-solved optimal instances, which can be prohibitively expensive to generate for MIPs [13], [14].

To alleviate these issues, Differentiable Predictive Control (DPC) [15], [16] introduced an MPC-inspired self-supervised training of a parametric control policy via automatic differentiation over known dynamics, enabling scalable offline pre-training and fast online inference. However, extending DPC to mixed-integer problems requires gradients through discrete mode decisions, although argmax is nondifferentiable almost everywhere. Gumbel-Softmax [17], [18] with the straight-through estimator [19] provides a workaround: the forward pass uses hard one-hot decisions, while the backward pass follows a differentiable softmax surrogate. Boldocky et al. [20] used this approach to introduce MI-´ DPC for a thermal energy system with linear dynamics and MLP policies. Concurrent works extend MI-DPC to nonlinear dynamics with binary decisions for data center cooling [21] and to battery dispatch with nondifferentiable degradation models [22]. Other self-supervised learning-tooptimize approaches for MINLP, including with feasibility guarantees [23], and mixed-integer MPC [24], exploit similar differentiable approximations to integer policies, yet consider different settings from the MI-DPC framework here. A key consideration in the proposed Gumbel-Softmax policy representation is the temperature τ [17]: high values encourage exploration but bias the relaxation, whereas low values sharpen decisions at the cost of weaker, noisier gradients and a greater risk of premature mode commitment, similar to relaxation collapse in differentiable architecture search [25]. To the best of our knowledge, no prior work has studied this trade-off systematically in the context of MI-DPC.

Contributions. We extend the MI-DPC framework [20] to UPHES scheduling, where polynomial UPCs and volume– head coupling introduce nonlinear, non-convex dynamics. To handle these challenges we introduce three methodological contributions: (i) a parallel differentiable simulator that preserves training-time gradient magnitude through nonlinear dynamics; (ii) a Transformer policy in place of an MLP; and (iii) a two-stage temperature schedule for more stable discrete mode learning. To facilitate adoption and reproducibility, we make the code open-source<sup>1</sup>.

## II. PROBLEM FORMULATION

We consider the day-ahead scheduling of an Underground Pumped Hydroelectric Storage (UPHES) unit operating as a price taker on the electricity market over a horizon of $N = 2 4$ hours with hourly time steps $\Delta t = 1$ 1 h. At each hour $t \in { \mathcal { T } } = \{ 1 , \ldots , N \}$ the unit selects one of three mutually exclusive operational modes: idle (I), turbine (T), or pump (P). The scheduling problem constitutes a mixedinteger nonlinear program (MINLP):

$$
\underset { p _ { t } ^ { T } , p _ { t } ^ { P } , m _ { t } } { \mathrm { m a x i m i z e } } \sum _ { t = 1 } ^ { N } \Bigl ( ( p _ { t } ^ { T } + p _ { t } ^ { P } ) \lambda _ { t } ^ { \mathrm { D A } } - C _ { \mathrm { o p } } ( p _ { t } ^ { T } + p _ { t } ^ { P } ) ^ { 2 } \Bigr )\tag{1a}
$$

$$
\mathrm { s . t . } \quad m _ { t } \in \{ - 1 , 0 , 1 \}\tag{1b}
$$

$$
p _ { \operatorname* { m i n } } ^ { T } ( h _ { t } ) [ m _ { t } ] _ { + } \leq p _ { t } ^ { T } \leq p _ { \operatorname* { m a x } } ^ { T } ( h _ { t } ) [ m _ { t } ] _ { + }\tag{1c}
$$

$$
p _ { \operatorname* { m i n } } ^ { P } ( h _ { t } ) \left[ - m _ { t } \right] _ { + } \leq p _ { t } ^ { P } \leq p _ { \operatorname* { m a x } } ^ { P } ( h _ { t } ) \left[ - m _ { t } \right] _ { + }\tag{1d}
$$

$$
q _ { t } = f _ { T } ^ { \mathrm { U P C } } ( p _ { t } ^ { T } , h _ { t } ) [ m _ { t } ] _ { + } + f _ { P } ^ { \mathrm { U P C } } ( p _ { t } ^ { P } , h _ { t } ) [ - m _ { t } ] _ { + }\tag{1e}
$$

$$
v _ { \mathrm { l o w } , t } = v _ { \mathrm { l o w } } ^ { \mathrm { i n i t } } + \Delta t \sum _ { s = 1 } ^ { t } q _ { s }\tag{1f}
$$

$$
v _ { \mathrm { l o w } , t } = f ^ { \mathrm { v o l } } ( h _ { t } )\tag{1g}
$$

$$
0 \leq v _ { \mathrm { l o w } , t } \leq v _ { \mathrm { m a x } }\tag{1h}
$$

$$
h _ { \operatorname* { m i n } } \leq h _ { t } \leq h _ { \operatorname* { m a x } }\tag{1i}
$$

$$
v _ { { \mathrm { l o w } } , N } \leq v _ { \mathrm { l o w } } ^ { \mathrm { t a r g e t } }\tag{1j}
$$

where $\lambda _ { t } ^ { \mathrm { D A } }$ is the day-ahead price and $C _ { \mathrm { o p } }$ is a quadratic op erational cost coefficient. The integer m<sub>t</sub> encodes the mode (1: turbine, 0: idle, −1: pump), with $[ m _ { t } ] _ { + } : = \operatorname* { m a x } ( 0 , m _ { t } )$ and $[ - m _ { t } ] _ { + } : = \operatorname* { m a x } ( 0 , - m _ { t } )$ . Power is signed as in [8]: $p _ { t } ^ { T } \geq 0$ is generation and $p _ { t } ^ { P } \leq 0$ is consumption, so the pump bounds satisfy $p _ { \operatorname* { m i n } } ^ { P } ( \bar { h } _ { t } ) \leq p _ { \operatorname* { m a x } } ^ { P } ( h _ { t } ) \bar { \leq } 0$ and the revenue in (1a) is negative when pumping. Constraints (1c)– (1d) enforce head-dependent power limits, while (1e) obtains the flow rate q<sub>t</sub> from mode-specific unit performance curves (UPCs) $f _ { m } ^ { \mathrm { U P C } } ( p , h )$ , which are bivariate polynomials fitted to experimental Francis pump-turbine data [26]. The maximum reservoir capacity $v _ { \mathrm { m a x } }$ is imposed by (1h).

The lower-reservoir volume $v _ { \mathrm { l o w } , t }$ evolves via cumulative flow (1f), accumulating net flow over all preceding time steps. The nonlinear volume–head coupling (1g) arises from reservoir geometry [4]. Constraint (1j) preserves long-term water balance. The polynomial UPCs, nonconvex volume– head map, and integer modes render (1) computationally challenging. Mixed-integer quadratic programming (MIQP) reformulations using either global linearization (MIQP-GL) or piecewise-bilinear approximations (MIQP-PW) are commonly employed, trading fidelity for tractability [8].

Beyond the technical constraints, the operator faces two market-driven costs incorporated in the training loss (Section III-E): a system imbalance penalty under a doublepricing settlement [27], [28] (shortages penalized at a premium, surpluses compensated at a discount), and a target volume penalty that monetizes any water left in the lower reservoir above the target in (1j), i.e., an end-of-horizon stateof-charge deficit, at the median day-ahead price.

## III. METHODOLOGY

This section reformulates the MINLP (1) as a differentiable program within the DPC framework [16], [20]. Nonlinear bounds (1c)–(1d) and dynamics (1e)–(1g) are handled in a differentiable simulator, using straight-through estimators for hard projections that block gradients. The target constraint (1j) and market settlement terms enter the loss (13), while Section III-C introduces the integrality mechanism for $m _ { t }$ . Figure 1 demonstrates the MI-DPC pipeline.

## A. Parametric Formulation

The MINLP (1) is parameterized by the initial hydraulic state and day-ahead price trajectory:

$$
\pmb { \xi } = [ h ^ { \mathrm { i n i t } } , v ^ { \mathrm { i n i t } } , \lambda _ { 1 } ^ { \mathrm { D A } } , \dots , \lambda _ { N } ^ { \mathrm { D A } } ] ^ { \top } \in \mathbb { R } ^ { N + 2 }\tag{2}
$$

A neural policy $\pi _ { \theta } : \mathbb { R } ^ { N + 2 }  [ 0 , 1 ] ^ { N \times 2 } \times \mathbb { R } ^ { N \times 3 }$ maps ξ directly to continuous power ratios $\mathbf { u } _ { c } \in \lbrack 0 , 1 \rbrack ^ { N \times 2 }$ and discrete mode logits $\ell \in \mathbb { R } ^ { N \times 3 } \colon [ \pmb { \mathscr { u } } _ { c } , \ell ] ^ { \top } = \pi _ { \theta } ( \pmb { \xi } )$ . Training minimizes the expected penalized loss over $\mathcal { P } _ { \xi }$ :

$$
\operatorname* { m i n } _ { \theta } \mathbb { E } _ { \boldsymbol { \xi } \sim \mathcal { P } _ { \boldsymbol { \xi } } } \Big [ \mathcal { L } \big ( \pi _ { \boldsymbol { \theta } } ( \boldsymbol { \xi } ) ; \boldsymbol { \xi } \big ) \Big ] ,\tag{3}
$$

where $\mathcal { L }$ combines negative profit and soft constraint penalties, discussed further in Section III-E.

## B. Neural Policy Architecture

The policy uses a Transformer encoder with two output heads. At each hour, the input token $\boldsymbol { z } _ { t } : = [ \bar { \lambda } _ { t } , \bar { h } , \bar { v } ] ^ { \top }$ contains the min-max normalized price $\bar { \lambda } _ { t }$ and repeated initial states h, <sup>¯</sup> v¯. Tokens are projected by a small MLP, augmented with sinusoidal positional encodings, and processed over the full horizon by a multi-layer Transformer encoder [29]. Two MLP heads then output the continuous power ratios $\pmb { u } _ { c } \in [ 0 , 1 ] ^ { N \times 2 }$ via sigmoid and the mode logits $\boldsymbol { \ell } \in \mathbb { R } ^ { N \times 3 }$ The discrete head biases are initialized to log $\left( \hat { p } _ { m } \right)$ , where $\hat { p } _ { m } \in \{ \hat { p } _ { \mathrm { p u m p } } , \ : \hat { p } _ { \mathrm { i d l e } } , \ : \hat { p } _ { \mathrm { t u r b } } \}$ is the mode distribution from empirical optimal solutions. This makes the initial softmax match the prior and reduces infeasible mode combinations [30].

## C. Differentiable Integer Relaxation

The differentiable integer relaxation block in Fig. 1 maps mode logits to hard modes while preserving gradients.

![](images/d1786ceb68d0264032ed5206f590c8d98a62a89cc61ce4a05e76abb41bf4db59.jpg)

![](images/608afc299607e3b5423ecbac303fed2c2565504298efe3a002ba1ea7cb39de87.jpg)  
Fig. 1. MI-DPC pipeline with Softmax STE. Problem parameters are mapped by a neural policy to continuous power ratios and mode logits. A Gumbel-Softmax STE yields hard mode decisions, the differentiable simulator rolls out the nonlinear UPHES dynamics, and the Lagrangian loss combines negative ex-post profit with soft constraint penalties. Green arrows denote the forward pass; red arrows denote backward gradient flow through the STE and simulator

![](images/21c05f056e91b866169a69d7ef50d385a2d0afcc8a4b86b3ccfccdd703fa3c96.jpg)  
Fig. 2. Gumbel-Softmax samples on the probability simplex for decreasing temperature coefficient τ. Vertices correspond to pump $( \mathrm { P } ) ,$ idle (I), and turbine (T). Lower τ concentrates samples near the vertices.  
Fig. 3. Standard clamp vs. straight-through (STE) clamp. (a) Both produce identical forward outputs. (b) The standard clamp zeros the gradient outside $[ a , b ] ,$ while the STE clamp preserves unit gradient everywhere.

a) Training with Gumbel-Softmax: During training, mode logits $\boldsymbol { \ell } _ { t } \in \mathbb { R } ^ { 3 }$ are discretized via the Gumbel-Softmax straight-through estimator [17]. Given i.i.d. Gumbel noise $g _ { i }$ the soft probabilities are

$$
\widetilde { s } _ { t , i } = \frac { \exp \bigl ( ( \ell _ { t , i } + g _ { i } ) / \tau \bigr ) } { \sum _ { j = 1 } ^ { 3 } \exp \bigl ( ( \ell _ { t , j } + g _ { j } ) / \tau \bigr ) } ,\tag{4}
$$

where $\tau > 0$ is the temperature coefficient. The straightthrough estimator [19] applies argmax in the forward pass to obtain a hard one-hot $\bar { \pmb { s } } _ { t } \in \{ 0 , 1 \} ^ { 3 }$ , while routing backward gradients through $\tilde { \pmb { s } } _ { t } \in ( 0 , 1 ) ^ { 3 }$ . The scalar mode is

$$
m _ { t } = \bar { \pmb { s } } _ { t } ^ { \top } [ - 1 , 0 , 1 ] ^ { \top } .\tag{5}
$$

At inference, Gumbel noise is removed and modes are selected deterministically via $\bar { \pmb { s } } _ { t } = \mathrm { o n e . h o t } ( \operatorname { a r g m a x } _ { i } \ \ell _ { t , i } )$

b) Temperature annealing: The temperature τ controls exploration versus commitment [18]. We keep $\tau { = } \tau _ { 0 }$ during warm-up and then decrease it exponentially toward $\tau _ { \mathrm { e n d } } \mathrm { : }$

$$
\tau ( e ) = \left\{ \begin{array} { l l } { \tau _ { 0 } , } & { e < e _ { w } , } \\ { \tau _ { 0 } \bigg ( \displaystyle \frac { \tau _ { \mathrm { e n d } } } { \tau _ { 0 } } \bigg ) ^ { ( e - e _ { w } ) / ( E - e _ { w } ) } , } & { \mathrm { o t h e r w i s e , } } \end{array} \right.\tag{6}
$$

where E is the total number of epochs and $e _ { w }$ is the warmup epochs. As illustrated in Fig. 2, high τ in warm-up encourages exploration, while low τ at the end yields nearhard decisions and narrows the training-inference gap.

## D. Differentiable Simulator

The differentiable simulator block in Fig. 1 unrolls the nonlinear dynamics while preserving end-to-end gradients.

a) Straight-through clamp: A standard clamp $\Pi _ { [ a , b ] } ( x ) : = \operatorname* { m i n } ( \operatorname* { m a x } ( x , a ) , b )$ zeros the gradient whenever a state saturates a bound,

$$
\frac { \partial \Pi _ { [ a , b ] } ( x ) } { \partial x } = 1 ( a < x < b ) ,\tag{7}
$$

where $1 ( \cdot ) \in \{ 0 , 1 \}$ is the indicator function, cutting off the training signal precisely when the policy is out of bounds. We replace all projections with a straight-through clamp [19]: identical to $\Pi _ { [ a , b ] }$ in the forward pass, but with gradient defined as

$$
\frac { \partial \tilde { \Pi } _ { [ a , b ] } ( x ) } { \partial x } : = 1 ,\tag{8}
$$

so when gradients flow through saturated states, the policy continues to receive a training signal at the boundary, as in Fig. 3. The projection also ensures the profit surrogate is computed over physically feasible trajectories; without it, the policy could exploit arbitrage along infeasible states, producing a misleading training signal (see Section III-E).

b) Power scaling: The normalized continuous outputs $\mathbf { \boldsymbol { u } } _ { c }$ are scaled to the feasible power vector p using headdependent bounds and mode indicators. For turbine mode,

$$
p _ { t } ^ { T } = \big ( p _ { \mathrm { m i n } } ^ { T } ( h _ { t } ) + u _ { c , t } ^ { T } ( p _ { \mathrm { m a x } } ^ { T } ( h _ { t } ) - p _ { \mathrm { m i n } } ^ { T } ( h _ { t } ) ) \big ) [ m _ { t } ] _ { + } .\tag{9}
$$

The pump expression is analogous, using $[ - m _ { t } ] _ { + }$

c) Parallel rollout: Let $\mathbf { x } : = \mathbf { v }$ denote the state variable (reservoir volume) and $\mathbf { s } : = \mathbf { h }$ the coupling variable (head). Given control variable p (power), the flow rate yields ${ \bf q } =$ $f ^ { \mathrm { U P C } } ( \mathbf { p } , \mathbf { s } , \mathbf { m } )$ as in (1e), the volume trajectory v is

$$
\mathbf { v } = v ^ { \mathrm { i n i t } } \mathbf { 1 } ^ { \top } + \Delta t \mathbf { q } \mathbf { M } ^ { \top } ,\tag{10}
$$

where $\mathbf { M } \in \mathbb { R } ^ { T \times T }$ is lower-triangular $( M _ { i j } : = { \bf 1 } [ j \leq i ] )$ , so $\begin{array} { r } { ( \mathbf q \mathbf { M } ^ { \top } ) _ { t } = \sum _ { k = 1 } ^ { t } q _ { k } } \end{array}$ accumulates flow causally, recovering the sequential update $v _ { t } = v _ { t - 1 } + \Delta t q _ { t }$ from (1f). Let f denote the dynamics and g the states coupling function:

$$
f ( \pmb { u } _ { c } , \mathbf { m } , \mathbf { s } ) : = v ^ { \mathrm { i n i t } } \mathbf { 1 } ^ { \top } + \Delta t f ^ { \mathrm { U P C } } ( \mathbf { p } , \mathbf { s } , \mathbf { m } ) \mathbf { M } ^ { \top } .\tag{11}
$$

$$
g ( \mathbf { x } ) : = ( f ^ { \mathrm { v o l } } ) ^ { - 1 } ( \mathbf { x } ) .\tag{12}
$$

Here $g$ recovers hydraulic head from volume via (1g). A natural baseline is the sequential rollout as shown in

Algorithm 1 Sequential Rollout Simulator   
Require: Initial condition $x _ { 0 } = v ^ { \mathrm { i n i t } } , \ s _ { 0 } = h ^ { \mathrm { i n i t } } ;$ control input $^ { u _ { c } ; }$ mode   
indicators m; STE projection clamp Π<sup>˜</sup>   
1: for $t = 1 , \ldots , N$ do   
2: $\hat { x } _ { t } \gets f ( u _ { c , t } , m _ { t } , s _ { t - 1 } )$ ▷ one-step dynamics   
3: $x _ { t } \gets \tilde { \Pi } ( \hat { x } _ { t } )$ ▷ STE clamp   
4: $s _ { t } \gets g ( x _ { t } )$ ▷ recover coupling variable   
5: end for   
6: return $( \mathbf { x } , { \hat { \mathbf { x } } } , \mathbf { s } )$ ▷ projected, raw, coupling variables   
Algorithm 2 Parallel Rollout Simulator   
Require: Initial condition ${ \hat { s } } _ { 0 } = { h } ^ { \mathrm { i n i t . } } ;$ control input $^ { u _ { c } ; }$ mode indicators   
m; STE projection clamp Π<sup>˜</sup>   
— Pass 1: frozen initial state —   
1: $\hat { \mathbf { x } } \gets f ( \pmb { u } _ { c } , \mathbf { m } , \hat { s } _ { 0 } \mathbf { 1 } ^ { \top } )$ ▷ batch dynamics with fixed state   
2: x ← Π(<sup>˜</sup> xˆ) ▷ STE clamp to feasible set   
3: s ← g(x) ▷ recover coupling variable   
— Pass 2: refined state —   
4: $\hat { \mathbf { x } } \gets f ( \pmb { u } _ { c } , \mathbf { m } , \mathbf { s } )$ ▷ re-evaluate with recovered state   
5: $\mathbf { x } \gets \tilde { \Pi } ( \hat { \mathbf { x } } )$ ▷ STE clamp   
6: s ← g(x) ▷ output map   
7: return $( \mathbf { x } , { \hat { \mathbf { x } } } , \mathbf { s } )$ ▷ projected, raw, coupling variables

Algorithm 1, which propagates the clamped state from step t to step t+1. Although exact, backpropagation through this N-step causal chain is analogous to backpropagation through time (BPTT) in recurrent networks, where gradient magnitudes decay exponentially with horizon length [31], [32]. The parallel rollout shown in Algorithm 2 replaces the sequential loop with a two-pass fixed-point estimate: Pass 1 evaluates all time steps in parallel using a frozen initial state, and Pass 2 refines with the recovered coupling variable. This breaks the N-step dependency chain, trading some simulator fidelity for stronger gradient signal.

## E. Loss Function Design

The loss combines profit and feasibility penalties:

$$
\begin{array} { r } { \mathcal { L } = - L ^ { \mathrm { e x } } + \mathcal { L } ^ { \mathrm { f e a s } } . } \end{array}\tag{13}
$$

a) Ex-post profit surrogate: Let $p _ { t } ^ { \mathrm { { o p t } } }$ denote scheduled net power from (9) and $p _ { t } ^ { \mathrm { s i m } }$ the realized net power from the differentiable simulator. The ex-post profit is

$$
\begin{array} { r l } & { { \cal L } ^ { \mathrm { e x } } = \underbrace { \sum _ { t = 1 } ^ { N } p _ { t } ^ { \mathrm { s i m } } \lambda _ { t } ^ { \mathrm { D A } } } _ { \mathrm { r e v e n u e } } - \underbrace { C _ { \mathrm { o p } } \sum _ { t = 1 } ^ { N } ( p _ { t } ^ { \mathrm { s i m } } ) ^ { 2 } } _ { \mathrm { o p e r a t i o n a l ~ c o s t } } } \\ & { \quad \quad - \underbrace { \omega _ { \mathrm { S U } } \displaystyle \sum _ { t = 1 } ^ { N } ( p _ { t } ^ { \mathrm { s i m } } - p _ { t } ^ { \mathrm { o p t } } ) \lambda _ { t } ^ { \mathrm { S I } } } _ { \mathrm { s y s t e m ~ i n b a l a n c e ~ p e n a l i p } } - \underbrace { \omega _ { \mathrm { T V } } \alpha \bar { \lambda } [ v _ { \mathrm { l o w } , N } - v _ { \mathrm { l o w } } ^ { \mathrm { t a r g e t } } ] } _ { \mathrm { t a r g e t ~ v o l u m e ~ p e n a l i p } } + , } \end{array}\tag{14}
$$

where $\lambda _ { t } ^ { \mathrm { S I } }$ is the system-imbalance price $( 2 \lambda _ { t } ^ { \mathrm { D A } }$ for shortages, $0 . 5 \lambda _ { t } ^ { \mathrm { D A } }$ for surpluses), λ<sup>¯</sup> is the median day-ahead price, α converts volume to energy, $\omega _ { \mathrm { S I } }$ and $\omega _ { \mathrm { T V } }$ are penalty weights, and $v _ { \mathrm { l o w } , N }$ is the final reservoir volume. The last two terms are the market costs of Section II: schedule deviations and excess of $v _ { \mathrm { l o w } , N }$ over $v _ { \mathrm { l o w } } ^ { \mathrm { t a r g e t } }$

Algorithm 3 MI-DPC Policy Optimizatoin   
Require: Neural control policy $\pi _ { \boldsymbol { \theta } } ;$ parametric distribution $\overline { { \mathcal { P } _ { \xi } ; } }$ number of   
epochs E; temperature schedule τ(e)   
1: for $e = 1 , \ldots , E$ do   
2: $\tau  \tau ( e )$ ▷ temp. annealing   
3: for each mini-batch $\{ \pmb { \xi } ^ { ( i ) } \} \sim \mathcal { P } _ { \pmb { \xi } }$ do   
4: $( { \pmb u } _ { c } , { \pmb \ell } )  { \pi } _ { \pmb \theta } ( { \pmb \xi } )$ ▷ neural policy   
5: $m \gets \mathrm { G u m b e l S T E } ( \ell , \tau )$ ▷ integer relaxation   
6: $( \mathbf { x } , \hat { \mathbf { x } } , \mathbf { s } ) \gets \mathrm { S i m u l a t e } ( { \pmb u } _ { c } , m )$ ▷ Alg. 2   
7: $\mathcal { L } \gets - L ^ { \mathrm { e x } } + \mathcal { L } ^ { \mathrm { f e a s } }$ ▷ Eqs. (14)–(15)   
8: $\nabla _ { \theta } \mathcal { L }  \mathrm { b a c k p r o p } ( \mathcal { L } )$   
9: Clip $\| \nabla _ { \theta } \mathcal { L } \| _ { \infty }$ and update θ via AdamW optimizer   
10: end for   
11: end for   
12: return Trained policy π<sub>θ</sub>

b) Feasibility penalties: The soft feasibility term penalizes violations of the raw volume $v _ { t } ^ { \operatorname { r a w } }$ and head $h _ { t } ^ { \operatorname* { r a w } }$ , i.e., the components of xˆ and $g ( \hat { { \bf x } } )$ in Algorithm 2:

$$
\begin{array} { r l r } {  { \mathcal { L } ^ { \mathrm { f e a s } } = \kappa _ { v } ^ { - } \sum _ { t = 1 } ^ { N } [ - v _ { t } ^ { \mathrm { r a w } } ] _ { + } + \kappa _ { v } ^ { + } \sum _ { t = 1 } ^ { N } [ v _ { t } ^ { \mathrm { r a w } } - v _ { \mathrm { m a x } } ] _ { + } } } \\ & { } & { + \kappa _ { h } ^ { - } \sum _ { t = 1 } ^ { N } [ h _ { \mathrm { m i n } } - h _ { t } ^ { \mathrm { r a w } } ] _ { + } + \kappa _ { h } ^ { + } \sum _ { t = 1 } ^ { N } [ h _ { t } ^ { \mathrm { r a w } } - h _ { \mathrm { m a x } } ] _ { + } , } \\ & { } & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad . } \end{array}\tag{15}
$$

where $\kappa _ { v } ^ { \pm }$ and $\kappa _ { h } ^ { \pm }$ are penalty weights. Together, the STE clamp and feasibility penalties form a two-sided training signal: the clamp keeps the profit surrogate (14) honest by enforcing feasibility in the forward pass, while the penalties push raw states back inside bounds in the backward pass.

## F. Training Strategy

Algorithm 3 summarizes the policy optimization procedure. Each epoch samples mini-batches from ${ \mathcal P } _ { \xi } ,$ rolls out the differentiable pipeline, and updates θ by backpropagation with gradient clipping. The temperature schedule is given by (6), while hyperparameters are provided in Section IV.

## IV. CASE STUDY

The case study uses a representative Belgian UPHES plant with hydraulic head $h \in [ 5 0 , 9 9 ]$ m and reservoir capacity $v _ { \mathrm { m a x } } { = } 5 8 8 , 0 0 0 \ \mathrm { m ^ { 3 } }$ ; UPCs are from laboratory measurements on a reduced-scale Francis pump-turbine [26], [33]. Dayahead prices are from the Belgian Elia TSO [34], with operational cost at 0.4 $\mathrm { E U R / M W ^ { 2 } }$ and imbalance multipliers of $2 \times$ (shortage) and 0.5× (surplus). The Transformer policy uses mode-logit bias log ˆp with $\hat { p } { = } [ 0 . 4 0 , 0 . 1 5 , 0 . 4 5 ]$ derived from optimal MIQP-PW mode distributions. Training uses AdamW optimizer, batch 32, 25 epochs on 10,000 scenarios sampled from a distribution fitted to 2024 Elia prices, excluding the evaluation set. Key hyperparameters: $\tau _ { 0 } { = } 1 0  \tau _ { \mathrm { e n d } } { = } 0 . 0 8 ( e _ { w } { = } 0 . 3 5 E ,$ i.e. 9 of 25 epochs); $\kappa _ { v } = \kappa _ { h } = 5 0 ; \ \omega _ { \mathrm { { S I } } } = \omega _ { \mathrm { { T V } } } = 1$ . The 19 held-out evaluation dates are selected by K-medoids clustering on 2024 Elia prices.

The MI-DPC pipeline is implemented using the Neuro-MANCER library [35], an open-source differentiable programming framework for parametric constrained optimization built on PyTorch 2.9.1 (CUDA 13.0). MIQP baselines are solved with Gurobi 13.0.0. All experiments run on an Intel Core Ultra 9 275HX with 32 GB RAM and an NVIDIA GeForce RTX 5070 Laptop GPU.

![](images/31f5d559cf5be27dbff54b14ed805ebcb743c80a833f76c4fbf108b7abb7a88f.jpg)  
Fig. 4. MI-DPC converged schedule for a representative day. Top: power dispatch (red, MW) and day-ahead price (blue, C/MWh); shaded red bands indicate the feasible turbine (positive) and pump (negative) power regions. Bottom: hydraulic head (orange) and reservoir volume (green); dashed lines mark the volume bounds and end-of-day target.

## A. Benchmark Comparison

Table I shows the performance of MI-DPC against MIQP-GL and MIQP-PW on the 19 held-out benchmark days. All methods are scored by the same ex-post profit (14) under the exact nonlinear simulator: the MIQP baselines optimize (1a) under approximate dynamics, so their schedules $p _ { t } ^ { \mathrm { { o p t } } }$ deviate from the realized $p _ { t } ^ { \mathrm { s i m } }$ and incur imbalance and target-volume penalties, whereas MI-DPC trains on (14) directly. MI-DPC statistics are mean and standard deviation over 47 seeds; the MIQP baselines are deterministic, with the optimal gap set at 1% and maximum solution time at 1 hour.

TABLE I  
MEAN EX-POST PROFIT (EUR/DAY) ACROSS 19 BENCHMARK DAYS.
<table><tr><td>Method</td><td>Profit (EUR/day)</td><td>Training time</td><td>Inference time</td></tr><tr><td>MIQP-GL</td><td>1,997</td><td></td><td>1.91 s</td></tr><tr><td>MIQP-PW</td><td>2,530</td><td></td><td>918.89 s</td></tr><tr><td>MI-DPC</td><td>2,489 ± 71</td><td>161 s</td><td>4.9 ms</td></tr></table>

MI-DPC achieves 2,489±71 EUR/day, a 24.6% improvement over MIQP-GL and within 1.6% of MIQP-PW. Its key advantage is online speed: 4.9 ms per day (390× faster than MIQP-GL and over five orders of magnitude faster than MIQP-PW), making it the only method compatible with real-time re-dispatch as market conditions update. MI-DPC requires 161 s of one-time offline training, whereas the MIQP baselines solve from scratch at every invocation.

Fig. 4 shows the converged MI-DPC schedule for a representative day. The dispatch lies strictly within the feasible turbine and pump power regions at every hour, confirming constraint satisfaction. The policy concentrates on pumping during low-price hours and turbining during peak-price hours, exploiting day-ahead price arbitrage, while head and volume stay within bounds and end near the target.

## B. Ablation Study

We ablate three design choices (policy architecture, temperature schedule, and training simulator), each evaluated over 47 random seeds on the same 19 benchmark days. Unless stated otherwise, all runs use the Transformer backbone, annealed temperature, and parallel dynamics from Table I. Fig. 5 shows the per-seed profit distributions for all three ablations.

a) Policy architecture: The Transformer achieves the highest mean ex-post profit (2,489±71 EUR/day), followed by Bi-LSTM $( 2 , 4 2 1 \pm 1 3 3 )$ , MLP $( 2 , 3 5 3 \pm 1 4 3 )$ , and CNN $( 2 , 0 8 1 \pm 1 8 7 ) ;$ see Fig. 5(a). Self-attention captures the longrange price dependencies due to arbitrage.

b) Temperature schedule: The two-stage annealing schedule (6) outperforms fixed temperature $\mathit { \Pi } ( \tau \ = \ 0 . 0 8$ throughout): 2,489±71 vs. 2,285±163 EUR/day (Fig. 5(b)). Annealing also cuts cross-seed standard deviation (163 to 71 EUR/day): warm-up regularizes the search.

c) Training simulator: The parallel rollout (Algorithm 2) dominates the sequential rollout (Algorithm 1): 2,489 ± 71 vs. 2,142 ± 51 EUR/day (Fig. 5(c)). Backpropagation through the N-step causal chain in Algorithm 1 is equivalent to BPTT, causing vanishing-gradient effects that degrade policy learning. The sequential rollout is also 12× slower to train (2,133 vs. 161 s).

## V. CONCLUSION

This paper extended MI-DPC to nonconvex polynomial dynamics and multi-modal discrete decisions for UPHES scheduling via a parallel differentiable simulator, a Transformer encoder, and a two-stage temperature schedule. The MI-DPC framework achieves ex-post profit within 1.6% of a piecewise MIQP baseline at millisecond inference times, a five orders of magnitude acceleration compatible with realtime re-dispatch. Ablations show that the Transformer captures long-range arbitrage dependencies, temperature annealing prevents premature mode commitment, and the parallel simulator preserves gradient magnitude.

Future work includes addressing uncertainties arising from plant-model mismatch and price forecasts. From a theoretical perspective, we aim to derive rigorous feasibility guarantees for mixed-integer optimal control problems with nonconvex polynomial dynamics via tractable safety filters.

## REFERENCES

[1] P. Bendotti, P. Fouilhoux, and C. Rottner, “On the complexity of the unit commitment problem,” Annals of Operations Research, vol. 274, no. 1, pp. 119–130, 2019.

[2] S. Rehman, L. M. Al-Hadhrami, and M. M. Alam, “Pumped hydro energy storage system: A technological review,” Renewable and Sustainable Energy Reviews, vol. 44, pp. 586–598, 2015.

[3] A. Blakers, M. Stocks, B. Lu, and C. Cheng, “A review of pumped hydro energy storage,” Progress in Energy, vol. 3, no. 2, 2021.

[4] J.-F. Toubeau, S. Iassinovski, E. Jean, J.-Y. Parfait, J. Bottieau, Z. De Greve, and F. Vall \` ee, “Non-linear hybrid approach for the ´ scheduling of merchant underground pumped hydro energy storage,” IET Generation, Transmission & Distribution, vol. 13, no. 21, pp. 4798–4808, 2019.

![](images/30e58158a08e8a0463dcb199169d8cb62d44f7c258724b1b8027f701860ab9be.jpg)

(b) Temperature  
![](images/e2fc1600550c0e1d3b15077237eeb7ff18d4c9f9b921d314cc9ec9c7733a70a2.jpg)

(c) Dynamics  
![](images/3bac0309c8b50112dedb634e23f8dd7fd1dbd5f3daf3d16692a544d4c042fb75.jpg)  
Fig. 5. Ablation study: per-seed ex-post profit distributions (47 seeds each). Violin bodies show KDE density; horizontal lines mark the median and interquartile range. The dashed horizontal line indicates the MIQP-PW mean. (a) Architecture: the Transformer dominates, with MLP and Bi-LSTM as closest competitors. (b) Temperature schedule: annealing reduces cross-seed variance and improves mean profit. (c) Training dynamics: the parallel simulator substantially outperforms the sequential rollout simulator.

[5] J.-F. Toubeau, Z. De Greve, P. Goderniaux, F. Vall \` ee, and K. Bruninx,´ “Chance-constrained scheduling of underground pumped hydro energy storage in presence of model uncertainties,” IEEE Transactions on Sustainable Energy, vol. 11, no. 3, pp. 1516–1527, 2019.

[6] P. Favaro, M. Dolanyi, F. Vall´ ee, and J.-F. Toubeau, “Neural network´ informed day-ahead scheduling of pumped hydro energy storage,” Energy, vol. 289, p. 129999, 2024.

[7] M. Gobert, J. Gmys, J.-F. Toubeau, N. Melab, D. Tuyttens, and F. Vallee, “Parallel bayesian optimization for optimal scheduling of ´ underground pumped hydro-energy storage systems,” in 2022 IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW). IEEE, 2022, pp. 790–797.

[8] H. Zheng, P. Favaro, Y. Dvorkin, and J. Drgona, “Accelerating un-ˇ derground pumped hydro energy storage scheduling with decisionfocused learning,” IEEE Transactions on Sustainable Energy, 2026.

[9] A. Bemporad, M. Morari, V. Dua, and E. N. Pistikopoulos, “The explicit linear quadratic regulator for constrained systems,” Automatica, vol. 38, no. 1, pp. 3–20, 2002.

[10] F. Borrelli, A. Bemporad, and M. Morari, Predictive control for linear and hybrid systems. Cambridge University Press, 2017.

[11] B. Karg and S. Lucia, “Efficient representation and approximation of model predictive control laws via deep learning,” IEEE transactions on cybernetics, vol. 50, no. 9, pp. 3866–3878, 2020.

[12] M. Hertneck, J. Kohler, S. Trimpe, and F. Allg ¨ ower, “Learning¨ an approximate model predictive controller with guarantees,” IEEE Control Systems Letters, vol. 2, no. 3, pp. 543–548, 2018.

[13] A. Cauligi, P. Culbertson, E. Schmerling, M. Schwager, B. Stellato, and M. Pavone, “Coco: Online mixed-integer control via supervised learning,” IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 1447–1454, 2022.

[14] D. Bertsimas and B. Stellato, “Online mixed-integer optimization in milliseconds,” INFORMS Journal on Computing, vol. 34, no. 4, pp. 2229–2248, 2022.

[15] J. Drgona, K. Kiˇ s, A. Tuor, D. Vrabie, and M. Klauˇ co, “Differen-ˇ tiable predictive control: Deep learning alternative to explicit model predictive control for unknown nonlinear systems,” Journal of Process Control, vol. 116, pp. 80–92, 2022.

[16] J. Drgona, A. Tuor, and D. Vrabie, “Learning constrained parametricˇ differentiable predictive control policies with guarantees,” IEEE Transactions on Systems, Man, and Cybernetics: Systems, vol. 54, no. 6, pp. 3596–3607, 2024.

[17] E. Jang, S. Gu, and B. Poole, “Categorical reparameterization with gumbel-softmax,” arXiv preprint arXiv:1611.01144, 2016.

[18] C. J. Maddison, A. Mnih, and Y. W. Teh, “The concrete distribution: A continuous relaxation of discrete random variables,” arXiv preprint arXiv:1611.00712, 2016.

[19] Y. Bengio, N. Leonard, and A. Courville, “Estimating or propagating ´ gradients through stochastic neurons for conditional computation,” arXiv preprint arXiv:1308.3432, 2013.

[20] J. Boldocky, S. D. Javan, M. Gulan, M. M ´ onnigmann, and J. Drgo ¨ na,ˇ

“Learning to solve parametric mixed-integer optimal control problems via differentiable predictive control,” arXiv:2506.19646, 2025.

[21] J. Boldocky, C. Faulkner, E. Michael, M. Gulan, A. Tuor, and´ J. Drgona, “Data center chiller plant optimization via mixed-integer ˇ nonlinear differentiable predictive control,” SSRN 5764791, 2026.

[22] E. Safarzadeh Ravajiri, J. Drgona, M. Mehrtash, and B. F.ˇ Hobbs, “End-to-end battery dispatch with exact rainflow degradation via mixed-integer differentiable predictive control,” arXiv preprint arXiv:2609.12968, 2026.

[23] B. Tang, E. B. Khalil, and J. Drgona, “Learning to optimize for mixed-ˇ integer non-linear programming with feasibility guarantees,” arXiv preprint arXiv:2410.11061, 2024.

[24] J. Adamek, L. Luken, and S. Lucia, “Enabling robust mixed-integer¨ nonlinear model predictive control via self-supervised learning and combinatorial integral approximation,” Journal of Process Control, vol. 159, p. 103636, 2026.

[25] Y. Ichikawa, “Controlling continuous relaxation for combinatorial optimization,” Advances in Neural Information Processing Systems, vol. 37, pp. 47 189–47 216, 2024.

[26] T. Mercier, J. Jomaux, E. De Jaeger, and M. Olivier, “Provision of primary frequency control with variable-speed pumped-storage hydropower,” 2017 IEEE Manchester PowerTech, pp. 1–6, 2017.

[27] L. Vandezande, L. Meeus, R. Belmans, M. Saguan, and J.-M. Glachant, “Well-functioning balancing markets: A prerequisite for wind power integration,” Energy policy, vol. 38, no. 7, 2010.

[28] J. Bottieau, L. Hubert, Z. De Greve, F. Vall\` ee, and J.-F. Toubeau,´ “Very-short-term probabilistic forecasting for a risk-aware participation in the single price imbalance settlement,” IEEE Transactions on Power Systems, vol. 35, no. 2, pp. 1218–1230, 2019.

[29] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[30] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, “Focal loss´ for dense object detection,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 2980–2988.

[31] B. List, L.-W. Chen, K. Bali, and N. Thuerey, “Differentiability in unrolled training of neural physics simulators on transient dynamics,” Computer Methods in Applied Mechanics and Engineering, 2025.

[32] J. Xu, V. Makoviychuk, Y. Narang, F. Ramos, W. Matusik, A. Garg, and M. Macklin, “Accelerated policy learning with parallel differentiable simulation,” arXiv preprint arXiv:2204.07137, 2022.

[33] Multitel, “Smartwater,” https://www.multitel.be/projets/smartwater/, Sep. 2022.

[34] Elia Group, “Belgian bidding zone day-ahead reference price,” https: //www.elia.be/en/grid-data/transmission/day-ahead-reference-price, Nov 2025.

[35] J. Drgona, A. Tuor, J. Koch, M. Shapiro, B. Jacob, and D. Vrabie, “NeuroMANCER: Neural Modules with Adaptive Nonlinear Constraints and Efficient Regularizations,” https://github.com/pnnl/ neuromancer, 2023.