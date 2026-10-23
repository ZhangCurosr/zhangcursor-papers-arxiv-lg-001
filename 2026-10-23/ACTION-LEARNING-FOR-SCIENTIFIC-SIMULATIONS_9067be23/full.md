# ACTION LEARNING FOR SCIENTIFIC SIMULATIONS

Sarah Liaw<sup>1,2∗</sup> Yuanqi Du<sup>3</sup> Yilun Du<sup>1,2</sup>

<sup>1</sup>Kempner Institute <sup>2</sup>Harvard University <sup>3</sup>Microsoft Research New England

<sup></sup> Website § Code

## ABSTRACT

Understanding and controlling how systems evolve is central to science and engineering, where many problems require reasoning about complete trajectories. Existing approaches typically learn local dynamics or generate trajectories, but objectives constructed from individual transitions can miss dependencies across time, while generative models do not necessarily provide a tractable objective for evaluating and optimizing arbitrary paths. We study action learning, the problem of recovering a stochastic action from trajectory observations, yielding a common objective for reasoning about dynamical systems. We introduce LEAP (Learned Energy on Action Paths), which anchors the unknown action to a tractable reference process and learns the remaining correction by comparing observed trajectories with reference trajectories. This construction yields a differentiable objective on path space that can capture both local dynamical structure and nonadditive dependencies across time. Once learned, the same action can be flexibly reused for path ranking and inference, mechanism steering, constrained optimization and sampling. Across stochastic physical systems, many-body orbital dynamics and molecular simulations, LEAP recovers underlying dynamics, captures dependencies across time and enables transfer across inference objectives and physical conditions. We show that action learning is a promising framework for turning trajectory observations into reusable objectives for scientific inference and control.

## 1 FROM TRAJECTORY SAMPLES TO PATH INFERENCE

![](images/e6d8c3947db85634d6d69a210fe2067c528fa885b2cdedd08479a39e189c819f.jpg)  
(a) Local dynamics learning. One-step transition laws induce a likelihood over complete trajectories.

![](images/ac708142d10638cff869012bf023212606a321558e7e4dfc3c80d255e4ebc690.jpg)  
(b) Action learning. LEAP learns relative path costs from complete trajectory comparisons.  
Figure 1: Local dynamics learning versus action learning. Local models learn short-range transition laws that induce a path likelihood, whereas LEAP learns an action from complete trajectories while retaining structured local dynamics when appropriate.

Understanding how systems move between states requires reasoning about the trajectories that connect them. In molecular dynamics, two conformations may be connected by several transition routes that visit different intermediate configurations (Bolhuis et al., 2002; Shaw et al., 2010); in robotics, agents act through sequences of decisions (Kalakrishnan et al., 2011; Janner et al., 2022). In these settings, the object of interest is often the trajectory itself: which paths realize a transition, which mechanisms are plausible and which paths remain likely under perturbations or constraints.

Answering these questions requires reasoning about complete paths. To infer representative transitions or impose new preferences, we need an objective that assigns probabilities or costs to trajectories. When the dynamics are known, stochastic action functionals provide such an objective.

In statistical physics and stochastic control, actions such as the Onsager–Machlup functional, Freidlin–Wentzell rate functional and path integral control costs define probabilities or costs on path space (Onsager & Machlup, 1953; Freidlin & Wentzell, 2012; Kappen, 2005; Adib, 2007). Once the action is available, we can use it for downstream tasks like constrained inference and sampling.

In many applications, however, we do not know the action. Instead, we observe trajectories from simulations or experiments, such as molecular dynamics, sequences of state transitions, policy rollouts or logs of stochastic processes (Lindorff-Larsen et al., 2011; Shaw et al., 2010; Glielmo et al., 2021). Classical trajectory inference methods typically assume a known action, while learned methods predict local dynamics, learn representations or generate trajectories directly (Chen et al., 2019; Li et al., 2020; Song et al., 2021; Jing et al., 2024b). These objects are useful, but they do not directly give the reusable path objective needed for downstream trajectory inference. We therefore call the task of recovering an evaluable action from trajectory samples action learning.

Existing learned methods often model dynamics locally by fitting one-step transition laws and using the resulting local model to evaluate complete trajectories (Chua et al., 2018). This requirement can become restrictive when trajectories are selected by global events, e.g., successful transitions or task-level constraints. A local predictor trained on such data estimates the one-step conditionals of the selected process, which can differ from the underlying dynamics. Errors in local transition loglikelihoods can also accumulate along a path and then distort the induced optimization landscape. Since path-level quantities do not have to decompose into independent local decisions, optimizing locally plausible segments can still produce a poor local minimum; Fig. 1 shows this distinction.

We propose LEAP (Learned Energy on Action Paths), a framework for learning a stochastic action directly from trajectory samples. LEAP treats the action itself as the learning target, rather than a local predictor or trajectory generator. It anchors the unknown action to a tractable reference process and learns the remaining correction from complete trajectories, yielding an evaluable objective on path space. In structured settings, this correction recovers changes in the underlying dynamics; more generally, it can capture nonadditive interactions across transitions. Once learned, the action can be reused with preferences that were not specified during training. For example, a reward for visiting an intermediate basin can favor a different transition route, while a geometric penalty can discourage paths outside admissible regions, in both cases without retraining the learned action.

Our approach applies broadly to trajectories from dynamical systems, including molecular simulations, stochastic physical systems and logged sequential processes (Shaw et al., 2010; Kalakrishnan et al., 2011; Janner et al., 2022). We evaluate LEAP across controlled physical systems, a 31-body orbital benchmark with 186-dimensional states and an alanine dipeptide molecular benchmarks. These experiments test physical recovery, nonadditive dependence across time and reuse across inference objectives, with the double well and orbital systems providing exact actions for evaluation.

Our contributions are as follows: (i) we formulate action learning as learning a reusable objective directly on complete trajectories; (ii) we introduce LEAP, which learns a stochastic action by comparing observed trajectories with samples from a tractable reference process; (iii) we establish theoretical guarantees for recovering the action and show that dependencies across time can result in trajectory objectives that cannot be represented by sums of local transition costs; and (iv) we demonstrate recovery of underlying physical dynamics from terminally selected trajectories, modeling of nonadditive dependencies in many-body systems, and reuse of a fixed learned action across molecular inference tasks and physical conditions.

## 1.1 RELATED WORK

Sampling trajectories from known dynamics. Classical methods for sampling trajectories usually assume access to the dynamics, energy landscape or action. When trajectories of interest occur often, it is sufficient to just do direct simulation. When relevant paths are rare, e.g., because an energy barrier separates metastable states, methods can use known physical models to identify or sample transition pathways. The string method evolves discretized curves on an energy landscape (E et al., 2002), while minimum action methods optimize action functionals derived from stochastic dynamics (Freidlin & Wentzell, 2012). Stochastic control methods construct trajectory objectives from known dynamics and costs to sample rare transitions (Kappen, 2005; Todorov, 2009; Williams et al., 2017; Holdijk et al., 2023; Du et al., 2024). More recent methods use learned force fields or pretrained generative models to get energies or dynamics for transition inference (Yuan et al., 2024;

![](images/2a95a9ad1315f1f1030e8478d0ae372fe571f20480a5023a706d09d751cf8899.jpg)  
Figure 2: LEAP. Given data trajectories and a tractable reference process, LEAP learns a centered path-space correction and combines it with the reference action to obtain $S _ { \theta } ( \bar { \Phi } ; c )$

Raja et al., 2025; Amin et al., 2025). These methods typically rely on available physical models, learned dynamics or predefined objective to perform inference. LEAP targets the complementary setting of learning the trajectory action itself from observations, enabling subsequent inference without access to the underlying target dynamics at all.

Learning from trajectories. Learned trajectory models usually target evolution or generation. Neural differential equation and latent dynamics models learn evolution from observations (Chen et al., 2019; Kidger et al., 2021; Hafner et al., 2019), while score-based, flow matching, diffusion and action matching methods learn vector fields or scores that transport or generate samples from state, trajectory or marginal distributions (Song et al., 2021; Jing et al., 2024b;a; Neklyudov et al., 2023). These methods model evolution or generation, but they do not generally return the stochastic action needed for path optimization, mechanism steering or Metropolis-Hastings correction. Energy-based models have also been used for planning by learning energies over individual state transitions and composing them along a trajectory (Du et al., 2020). Relatedly, reinforcement learning and imitation learning learn rewards or policies from demonstrations, e.g., maximum-entropy inverse reinforcement learning and adversarial imitation learning (Ziebart et al., 2008; Ho & Ermon, 2016; Fu et al., 2018; Nuti et al., 2023). LEAP also learns from trajectories, but targets a different object: a stochastic action on complete paths, i.e., the trajectory-level objective used directly for optimization, control and sampling, which can then be reused across new downstream objectives.

## 2 LEARNING ACTION FUNCTIONALS FROM TRAJECTORY SAMPLES

LEAP learns a differentiable stochastic action $S _ { \theta } ( \Phi ; c )$ directly from trajectory samples, providing a reusable objective for evaluating and optimizing trajectories. The action assigns a scalar objective to each complete trajectory, with lower action corresponding to higher probability under the observed path distribution. LEAP requires trajectory observations but not the underlying drift, potential or simulator likelihood. We construct the action by introducing a tractable reference process and learning a correction that describes how the observed path distribution differs from the reference. This formulation can represent both local dynamical structure and dependencies across time, yielding an objective that can be evaluated and reused for new inference tasks. See Alg. 1 for the algorithm.

## 2.1 TARGET ACTION AND TRAINING OBJECTIVE

Let $\Phi = ( x _ { 0 } , x _ { 1 } , \ldots , x _ { N } )$ denote a discretized trajectory with $t _ { k } = k \Delta t$ and horizon $T = N \Delta t$ Let c denote the conditioning variables, $\mathrm { e . g . }$ , endpoints, terminal regions, horizon or diffusion level. We observe trajectory samples $\Phi \sim p _ { \mathrm { d a t a } } ( \cdot \mid c )$ for each context.

Target action. Following stochastic action formalisms such as Onsager–Machlup and Freidlin– Wentzell theory, we use the negative log path density as the trajectory objective (Onsager & Machlup, 1953; Freidlin & Wentzell, 2012):

$$
S ^ { * } ( \Phi ; c ) = - \log p _ { \mathrm { d a t a } } ( \Phi \mid c ) + \alpha ( c ) ,\tag{1}
$$

where $\alpha ( c )$ is constant across paths with the same context and therefore does not affect their relative action, including for path ranking, sampling or representative path optimization (§2.4).

Reference-based decomposition. Directly evaluating $p _ { \mathrm { d a t a } } ( \Phi \mid c )$ is generally not possible from trajectory samples. LEAP therefore introduces a tractable reference process with density $p _ { \mathrm { r e f } } ( \Phi \mid c )$

providing an evaluable baseline action against which the observed dynamics can be characterized. The unknown action can be expressed as the reference action plus a correction that captures how the observed path distribution differs from the reference. This correction admits a physical interpretation when the processes differ in drift, while more generally allowing dependencies across transitions that cannot be represented by additive local objectives.

Specifically, the correction is the path log-density ratio

$$
G ^ { * } ( \Phi ; c ) = \log \frac { p _ { \mathrm { d a t a } } ( \Phi \mid c ) } { p _ { \mathrm { r e f } } ( \Phi \mid c ) } .\tag{2}
$$

With $S _ { \mathrm { r e f } } ( \Phi ; c ) = - \log p _ { \mathrm { r e f } } ( \Phi \mid c )$ , the target action is

$$
S ^ { * } ( \Phi ; c ) = \underbrace { S _ { \mathrm { r e f } } ( \Phi ; c ) } _ { \mathrm { r e f e r e n c e \ a c t i o n } } - \underbrace { G ^ { * } ( \Phi ; c ) } _ { \mathrm { l e a r n e d \ c o r r e c t i o n } } + \alpha ( c ) .\tag{3}
$$

We estimate $G ^ { * }$ from comparisons between observed and reference trajectories and combine it with the reference action. The resulting $S _ { \theta }$ is the learned stochastic action used for trajectory inference.

Learning the action correction. To estimate $G ^ { * }$ , we compare observed trajectories with those drawn from the reference process (Gutmann & Hyvarinen¨ , 2010). For each context $c ,$ we draw one observed trajectory $\Phi ^ { + } \sim p _ { \mathrm { d a t a } } ( { \cdot } \mid c )$ and K reference trajectories $\Phi _ { k } ^ { - } \sim p _ { \mathrm { r e f } } ( \cdot \mid c )$ . We parameterize the correction by $G _ { \theta } ( \Phi ; c )$ and optimize the setwise comparison objective, which has the InfoNCE form (van den Oord et al., 2019)

$$
\mathcal { L } _ { K } ( \theta ) = - \mathbb { E } \left[ \log \frac { \exp G _ { \theta } ( \Phi ^ { + } ; c ) } { \exp G _ { \theta } ( \Phi ^ { + } ; c ) + \sum _ { k = 1 } ^ { K } \exp G _ { \theta } ( \Phi _ { k } ^ { - } ; c ) } \right] .\tag{4}
$$

This objective has the desired correction as its population solution. Under the conditions in $\ S \operatorname { A } . 1$ every population minimizer satisfies

$$
G _ { \theta ^ { * } } ( \Phi ; c ) = G ^ { * } ( \Phi ; c ) + a ( c ) ,\tag{5}
$$

for every $K \geq 1$ . The context-dependent term cancels from each comparison, so the objective identifies the part of the correction that varies across trajectories.

Because the comparison objective leaves this context-dependent constant undetermined, we fix the additive convention after training by centering the correction under the reference distribution,

$$
\bar { G } _ { \theta } ( c ) = \mathbb { E } _ { \Phi \sim p _ { \mathrm { r e f } } ( \cdot | c ) } \left[ G _ { \theta } ( \Phi ; c ) \right] , \qquad \widetilde { G } _ { \theta } ( \Phi ; c ) = G _ { \theta } ( \Phi ; c ) - \bar { G } _ { \theta } ( c ) .\tag{6}
$$

The learned action is then

$$
S _ { \theta } ( \Phi ; c ) = - \log p _ { \mathrm { r e f } } ( \Phi \mid c ) - \widetilde G _ { \theta } ( \Phi ; c ) .\tag{7}
$$

At the population solution, this recovers the data action up to a constant depending on the context, which does not affect path rankings or optimization within that context.

## 2.2 REFERENCE PROCESS

The reference defines the baseline path measure and the inductive bias of the learned action. This means we need a tractable path density, efficient sampling and sufficient probability mass over the trajectories represented in the data. While its drift can include an available nominal drift or force model, LEAP does not necessarily require one. For rare-event trajectories, we condition the reference on the same event as the data so that both contain trajectories selected by the same condition.

We use a discrete-time Gaussian Markov reference because its transition law is analytic and parameterized directly by drift and diffusion. Under shared diffusion, differences in drift induce an exact path log-likelihood ratio, giving the learned correction a direct dynamical interpretation. We define

$$
\begin{array} { r } { x _ { k + 1 } = x _ { k } + b _ { \mathrm { r e f } } ( x _ { k } , t _ { k } ; c ) \Delta t + \sqrt { 2 D \Delta t } \xi _ { k } , \qquad \xi _ { k } \sim \mathcal { N } ( 0 , I ) , } \end{array}\tag{8}
$$

where $b _ { \mathrm { r e f } }$ is the reference drift and D is the diffusion coefficient. This induces the path density $p _ { \mathrm { r e f } } ( \Phi \mid c )$ and reference action $S _ { \mathrm { r e f } } ( \Phi ; c )$ . For endpoint or terminal-region conditioning, p<sub>ref</sub> denotes the corresponding conditional density over the interior knots; its normalizer contributes only a context-dependent constant (see §A.3).

![](images/51d4b641fa7953a3e189a21c8140f243c6c16fb574448c46d74190edd76a40b0.jpg)  
Figure 3: Downstream trajectory inference with the learned action. Once $S _ { \theta } ( \Phi ; c )$ is learned, the same action supports representative path inference, mechanism steering (tilting), constrained inference and sampling.

## 2.3 LOCAL AND GLOBAL ACTION CORRECTIONS

When the data and reference processes are Gaussian Markov processes with the same diffusion coefficient and differ only in drift, their exact path log-likelihood ratio decomposes over transitions. We use this structure when the drift difference is the quantity of interest. Let $v _ { k } = ( x _ { k + 1 } - x _ { k } ) / \Delta t$ and let u parameterize $b _ { \mathrm { d a t a } } - b _ { \mathrm { r e f } }$ . The correction is

$$
G _ { \theta } ( \Phi ; c ) = \sum _ { k = 0 } ^ { N - 1 } \left[ \underbrace { \frac { 1 } { 2 D } u _ { \theta } ( x _ { k } , t _ { k } ; c ) ^ { \top } \left( v _ { k } - b _ { \mathrm { r e f } } ( x _ { k } , t _ { k } ; c ) \right) } _ { \mathrm { v e l o c i t y ~ a l i g m m e n t } } - \underbrace { \frac { 1 } { 4 D } \left\| u _ { \theta } ( x _ { k } , t _ { k } ; c ) \right\| ^ { 2 } } _ { \mathrm { n o r m a l i z a t i o n ~ c o r t } } \right] \Delta t .\tag{9}
$$

When the data and reference processes also share the initial law and time grid, and differ only through their drift, Eq. (9) is the exact path log-likelihood ratio for $b _ { \mathrm { d a t a } } = b _ { \mathrm { r e f } } + u ^ { * }$ ; see §A.2. For stopped or padded trajectories, we sum only over active transitions as described in the appendix.

More generally, LEAP does not require a Gaussian drift parameterization. When the log density ratio between the data and reference distributions decomposes across transitions, LEAP-Local learns a scalar correction $g _ { \theta }$ for each transition and sums these corrections along the trajectory,

$$
G _ { \theta } ^ { \mathrm { l o c a l } } ( \Phi ; c ) = \sum _ { k = 0 } ^ { N - 1 } g _ { \theta } ( x _ { k } , x _ { k + 1 } , t _ { k } ; c ) .\tag{10}
$$

This factorization can fail when information is shared across the trajectory. For example, marginalizing a latent variable that remains fixed throughout a trajectory can couple multiple transitions. LEAP-Global therefore models the correction directly from the complete path,

$$
G _ { \theta } ^ { \mathrm { g l o b a l } } ( \Phi ; c ) = r _ { \theta } ( \Phi ; c ) .\tag{11}
$$

LEAP-Hybrid combines the two parameterizations,

$$
G _ { \theta } ^ { \mathrm { h y b r i d } } ( \Phi ; c ) = G _ { \theta } ^ { \mathrm { l o c a l } } ( \Phi ; c ) + G _ { \theta } ^ { \mathrm { g l o b a l } } ( \Phi ; c ) .\tag{12}
$$

The global term captures interactions across transitions that cannot be written as an additive transition correction. When both terms are learned jointly, only their sum is identified, so interpreting the local term as a physical drift correction requires additional restrictions.

For conservative systems, we optionally parameterize the total drift as the gradient of a potential,

$$
b _ { \mathrm { r e f } } ( \boldsymbol { x } , t ; c ) + u _ { \theta } ( \boldsymbol { x } , t ; c ) = - \nabla U _ { \theta } ( \boldsymbol { x } ; c ) .\tag{13}
$$

## 2.4 DOWNSTREAM TRAJECTORY INFERENCE

Because $S _ { \theta }$ is an action on complete trajectories, we can reuse it after training. The same learned object can rank candidate paths, infer representative trajectories, incorporate mechanism preferences, impose soft constraints and define trajectory distributions. See §B.2 for details.

Trajectory ranking and representative inference. Given feasible trajectories Γ(c), e.g., paths satisfying boundary conditions, we can rank candidate trajectories by their action or infer the lowestaction representative trajectory by solving

$$
\Phi ^ { * } ( c ) = \arg \operatorname* { m i n } _ { \Phi \in \Gamma ( c ) } S _ { \theta } ( \Phi ; c ) .\tag{14}
$$

Algorithm 1 LEAP training   
Require: dataset $\mathcal { D } = \{ ( \Phi _ { i } ^ { + } , c _ { i } ) \}$ , reference process $p _ { \mathrm { r e f } } ( \Phi \mid c )$ , reference count K   
1: while not converged do   
2: sample minibatch $\{ ( \Phi _ { i } ^ { + } , c _ { i } ) \} _ { i = 1 } ^ { B } \subset \mathcal { D }$   
3: for $\bar { i } = 1$ to B do   
4: sample $\{ \Phi _ { i , k } ^ { - } \} _ { k = 1 } ^ { K } \sim p _ { \mathrm { r e f } } ( \cdot \mid c _ { i } )$   
5: evaluate $G _ { i } ^ { + } \gets G _ { \theta } ( \Phi _ { i } ^ { + } ; c _ { i } )$ and $G _ { i , k } ^ { - } \gets G _ { \theta } ( \Phi _ { i , k } ^ { - } ; c _ { i } )$   
6: end for   
7: update $\theta$ by a gradient step on the minibatch estimate of Eq. (4)   
8: end while   
9: return learned action $S _ { \theta }$ in $\operatorname { E q . } \left( 7 \right)$

Mechanism steering and constraints. We steer the representative path by adding trajectory-level functionals. Given a desired mechanism preference $M ( \mathbf { \dot { \Phi } } )$ and a constraint penalty $C ( \Phi )$ , we solve

$$
\arg \operatorname* { m i n } _ { \Phi \in \Gamma ( c ) } \left[ S _ { \theta } ( \Phi ; c ) - \lambda _ { M } M ( \Phi ) + \lambda _ { C } C ( \Phi ) \right] .\tag{15}
$$

Changing $\lambda _ { M }$ or $\lambda _ { C }$ changes the inference problem without retraining the model.

Compositional trajectory distributions. The learned action defines an unnormalized trajectory distribution. Given mechanism preferences $M _ { j }$ with weights $\gamma _ { j }$ and penalties $C _ { \ell }$ with weights $\lambda _ { \ell } .$

$$
\pi ( \Phi \mid c ) \propto \exp \left( - S _ { \theta } ( \Phi ; c ) + \sum _ { j } \gamma _ { j } M _ { j } ( \Phi ) - \sum _ { \ell } \lambda _ { \ell } C _ { \ell } ( \Phi ) \right) .\tag{16}
$$

Sampling from $\pi$ gives a family of plausible paths weighted by the specified preferences/penalties.

Metropolis-Hastings (MH) sampling. At fixed context, $S _ { \theta }$ defines an unnormalized distribution over paths. For a proposal kernel $q ( \Phi ^ { \prime } \mid \Phi )$ on $\Gamma ( c )$ , the MH acceptance probability is

$$
{ \cal A } ( \Phi , \Phi ^ { \prime } ) = \operatorname * { m i n } \left\{ 1 , \exp \bigl ( - S _ { \theta } ( \Phi ^ { \prime } ; c ) + S _ { \theta } ( \Phi ; c ) \bigr ) { \frac { q ( \Phi \mid \Phi ^ { \prime } ) } { q ( \Phi ^ { \prime } \mid \Phi ) } } \right\} .\tag{17}
$$

For symmetric proposals, this reduces to $A ( \Phi , \Phi ^ { \prime } ) =$ min $\big \{ 1 , \exp \big ( - S _ { \theta } \big ( \Phi ^ { \prime } ; c \big ) + S _ { \theta } \big ( \Phi ; c \big ) \big ) \big \}$ . Any context-dependent additive constant in $S _ { \theta }$ cancels in the acceptance ratio.

## 3 WHY LEAP RECOVERS AN ACTION

We give three guarantees for action learning. The first recovers the data action from the comparison objective, the second identifies the drift correction under the structured model, and the third bound downstream inference error from action error. Full assumptions and proofs are in $\ S \mathrm { A }$

The comparison objective recovers the action. Fix a context $c .$ Assume that the data and reference distributions have positive densities on the same support, the candidates are conditionally independent and the score class contains $G ^ { * }$ . Every population minimizer of Eq. (4) satisfies

$$
G _ { \theta ^ { * } } ( \Phi ; c ) = \log \frac { p _ { \mathrm { d a t a } } ( \Phi \mid c ) } { p _ { \mathrm { r e f } } ( \Phi \mid c ) } + a ( c ) .\tag{18}
$$

I $\begin{array} { r } { \mathrm { ~ f ~ } \mathbb { E } _ { p _ { \mathrm { r e f } } } | G ^ { * } | < \infty } \end{array}$ , reference centering then gives

$$
S _ { \theta ^ { * } } ( \Phi ; c ) = S ^ { * } ( \Phi ; c ) + \beta ( c ) .\tag{19}
$$

The context-dependent constant keeps optimization solutions and MH ratios the same.

The structured correction recovers drift differences. For Euler–Maruyama dynamics with the same diffusion, time step and initial distribution, changing the drift changes the path log-likelihood by the structured correction in Eq. (9). Matching the unconditioned path distribution identifies the drift at states and times visited by the process. If data and reference trajectories are selected using the terminal event, this changes the path log-likelihood ratio by condition-dependent constants. Recovering the unconditioned drift from selected trajectories needs variation across terminal conditions and the structured drift parameterization. See §4.1 for the recovery; §A.2 and A.3 for derivations.

![](images/961a3196d2b8aa53c054134138f4ce94a2be570771d5affa1c36774c5720d7cd.jpg)

<table><tr><td></td><td>Drift MSE↓</td><td colspan="2">True path cost ↓</td></tr><tr><td>Method</td><td></td><td> $T = 4$ </td><td> $T = 1 6$ </td></tr><tr><td>Kramers-Moyal</td><td>0.813</td><td></td><td></td></tr><tr><td>Cubic velocity reg.</td><td>0.775</td><td>0.319</td><td>0.613</td></tr><tr><td>Zero field</td><td>0.0759</td><td>0.264</td><td>0.445</td></tr><tr><td>Geometric path</td><td></td><td>0.264</td><td>0.264</td></tr><tr><td>Analytic drift</td><td>0</td><td>0.253</td><td>0.250</td></tr><tr><td>LEAP</td><td> $\mathbf { 3 . 3 \times 1 0 ^ { - 4 } }$ </td><td>0.253</td><td>0.251</td></tr></table>

Figure 5: Double well recovery.  
Table 1: Physical recovery and path inference. Drift error and path cost under new endpoints and horizons.

Action error controls later inference. Fix a context c and an inference domain $\Gamma ( c )$ . Suppose

$$
\operatorname* { s u p } _ { \Phi \in \Gamma ( c ) } | S _ { \theta } ( \Phi ; c ) - S ^ { * } ( \Phi ; c ) - \beta ( c ) | \le \varepsilon .
$$

For the same added functional $H _ { \ast }$ , let $F _ { \theta } = S _ { \theta } + H$ and $F ^ { * } = S ^ { * } + H$ . If $\widehat { \Phi }$ is δ-suboptimal for $F _ { \theta }$

$$
F ^ { * } ( \widehat \Phi ) - \operatorname* { i n f } _ { \Phi \in \Gamma ( c ) } F ^ { * } ( \Phi ) \leq 2 \varepsilon + \delta .\tag{20}
$$

If $\pi _ { \boldsymbol { \theta } } \propto e ^ { - F _ { \boldsymbol { \theta } } }$ and $\pi ^ { * } \propto e ^ { - F ^ { * } }$ have finite, positive normalizing constants, then

$$
\left| \log { \frac { \pi _ { \theta } ( \Phi \mid c ) } { \pi ^ { * } ( \Phi \mid c ) } } \right| \leq 2 \varepsilon .\tag{21}
$$

Thus the same action error bounds both the objective value of an inferred trajectory and the change in its normalized trajectory density.

## 4 EXPERIMENTS

Our experiments evaluate four properties of action learning. We show that LEAP can recover physical dynamics from selected trajectories, capture dependencies across multiple transitions, improve as the reference moves closer to the data and reuse the learned action under new physical conditions and inference requests. See §B for the full experimental details.

## 4.1 RECOVERING DYNAMICS UNDER TERMINAL SELECTION

We first show that LEAP recovers the underlying drift when training trajectories are selected by their final state. We use the overdamped diffusion $d X _ { t } = b ( X _ { t } ) d t + \sqrt { 2 D } d W _ { t }$ with $b ( x ) = x - x ^ { 3 }$ and $U ( x ) = ( x ^ { 2 } - 1 ) ^ { 2 } / 4$ . In this double well setting, trajectories start at $x _ { 0 } = - 1$ and are kept when $\dot { x } _ { T } \in [ 0 . 9 , 1 . 1 ]$ at $T = 3$ We use 5,000 selected trajectories per seed with $D = 0 . 2 0$ and $\Delta t = 0 . 0 1$ , and a Brownian reference conditioned on the same terminal window. Conditioning on the final state changes the distribution of one-step increments, so local regression does not directly estimate the original drift. LEAP instead compares complete selected paths with the matched reference and learns the drift through the structured action correction.

Drift recovery. We compare LEAP with Kramers-Moyal, which estimates drift from binned onestep increments, and cubic velocity regression using $\{ 1 , \dot { x } , \dot { x } ^ { 2 } , x ^ { 3 } \}$ , which contains the exact drift x− $x ^ { 3 } .$ LEAP achieves drift MSE $( 3 . { \dot { 3 } } \pm { \dot { 2 } } . 7 ) \times 1 0 ^ { - 4 }$ , compared to 0.8132±0.0030 for Kramers-Moyal, $0 . 7 7 5 2 { \scriptstyle \pm 0 . 0 0 4 3 }$ for cubic regression and 0.0759 for the zero field (Fig. 5 and Table 1). Even though cubic regression can represent the exact drift, it fits the one-step dynamics of the terminally selected trajectories, which follow a Doob h-transformed drift that differs from the original drift (§A.3).

Path inference under new endpoints. Next, we keep the learned potential fixed and change the endpoint from the training region to the saddle $x _ { T } = 0 \nonumber$ , and vary the horizon. At $T = 4$ LEAP gives true path cost $0 . \bar { 2 } 5 3 \bar { 3 } \pm 0 . 0 0 0 1$ versus the analytic value 0.2532. At $T = 1 6 ,$ , LEAP gives $0 . 2 5 0 6 \pm 0 . 0 0 0 6$ versus 0.2500, while cubic velocity regression gives $0 . 6 1 2 6 \pm 0 . 0 0 7 9$ and the geometric path gives 0.2637. The same learned action can still infer near-optimal paths after changing both the endpoint and the horizon.

Reference choice. We retrain LEAP with different references on an unconditioned double well, where the exact path density is available. Changing the reference drift from 0 to 0.5b reduces the correction error $\bar { \mathrm { N R M S E } } _ { \Delta G }$ from 0.66 to 0.223, where $\mathrm { N R M S E } _ { \Delta G }$ compares predicted and exact correction differences within each context and a constant prediction gives 1 (§B.1), showing that a closer reference leaves a smaller correction to learn. See §D for the full sweep.

## 4.2 RECOVERING A NONADDITIVE TRAJECTORY ACTION

Next, we test whether LEAP can learn a correction that depends on several transitions. These settings may appear when an unobserved physical property is fixed during a trajectory, like an unknown mass in orbital dynamics, friction coefficient in Langevin dynamics or material parameter in a mechanical system, and its effect accumulates over the full path. We model this by sampling $z \in \{ - 1 , + 1 \}$ once and changing the solar mass to $m _ { \odot } ( 1 + z \eta )$ . The same unobserved z is fixed for the entire trajectory, so recovering the observed path likelihood needs combining information across time. We use a 31-body orbital system initialized from JPL HORIZONS, giving a 186-dimensional state space (Giorgini et al., 1997).

Conditional on z, the path density factorizes over transitions. After marginalizing z, the exact datato-reference correction becomes

$$
\begin{array} { r } { G ^ { \ast } ( \Phi ; c ) = \log \left[ \frac { 1 } { 2 } e ^ { \ell _ { + 1 } ( \Phi ; c ) } + \frac { 1 } { 2 } e ^ { \ell _ { - 1 } ( \Phi ; c ) } \right] - \ell _ { 0 } ( \Phi ; c ) , } \end{array}\tag{22}
$$

where $\begin{array} { r } { \ell _ { z } = \sum _ { k } \ell _ { z , k } } \end{array}$ and $\ell _ { 0 }$ uses the nominal solar mass. Defining $d _ { k } = ( \ell _ { + 1 , k } - \ell _ { - 1 , k } ) / 2$ and $m _ { k } = ( \ell _ { + 1 , k } { \overline { { + } } } \ " \ell _ { - 1 , k } ) / 2 - \ell _ { 0 , k }$ gives

$$
G ^ { * } ( \Phi ; c ) = \sum _ { k } m _ { k } + \log \cosh A , \qquad A = \sum _ { k } d _ { k } .\tag{23}
$$

Its mixed derivatives satisfy $\partial ^ { 2 } G ^ { * } / ( \partial d _ { i } \partial d _ { j } ) > 0$ for $i \neq j ,$ whereas a sum of separate transition corrections has zero mixed derivatives. Therefore no additive transition model can represent the exact correction; see $\ S \mathrm { A . 4 }$

Local and complete-path models. We evaluate whether the correction can be recovered from individual transitions or needs information from the whole trajectory. LEAP-Local uses the local correction from Eq. (10). As a baseline with different supervision, Transition NCE applies noisecontrastive estimation (Gutmann & Hyvarinen ¨ , 2010) directly to data and reference transitions using the same transition inputs. LEAP-Global uses the full trajectory and features accumulated from the nominal gravitational dynamics and its sensitivity to the solar mass (Table 2). These inputs do not have the hidden variable z or the exact correction $G ^ { * }$

LEAP-Local and Transition NCE stay at the constant prediction level, with NRMSE 1.0013 and 1.0008, while LEAP-Global reaches 0.228, i.e., it accounts for about 95% of the squared variation in the exact correction differences. This shows that the correction cannot be recovered from separate transitions and requires information across the trajectory. Consistent with this result, Fig. 6 shows that LEAP-Global follows the exact log cosh A dependence and recovers the hidden mass perturbation as $1 . 0 2 \times 1 0 ^ { - 4 }$ , compared to the true $1 . 0 4 \times \dot { 1 } 0 ^ { - 4 }$ . As a trajectory diffusion comparison, GeoTDM (Han et al., 2024) is trained separately on the same data and reference paths, and the difference in its objectives gives AUROC 0.4999; see §E.5.

Accumulated features. We ablate these features while keeping a complete-path model, using binary and categorical neural ratio estimation (NRE) objectives (Miller et al., 2024). Without them, binary NRE gives NRMSE $1 . 0 0 0 0 0 \pm 3 \times 1 0 ^ { - 5 }$ . Adding them reduces error to $0 . 1 7 5 \pm 0 . 0 1 1$ with binary NRE and $0 . 2 2 8 \pm 0 . 0 1 5$ with categorical NRE (Table 3). Both objectives improve, showing that the accumulated information from Eq. (23) drives the decrease. Because both are ratio estimators on complete paths, the binary result also shows that action learning is not tied exactly to the categorical objective in Eq. (4).

Scaling. At 42 bodies and 256 transitions, each path has 64,764 coordinates. LEAP-Global reaches NRMSE 0.111, while LEAP-Local stays at 1.0015. Thus, even as the system grows to more bodies and longer trajectories, LEAP-Global can accumulate the small effect of the hidden mass perturbation across transitions, while models that treat each transition separately still stay near the constant prediction baseline. See §E.4 for the full sweep.

![](images/50ce60da3df8fdf053eb3439d88e10065a25025046bf970d16be7a91ed1cd7fb.jpg)

<table><tr><td>Model</td><td>Input</td><td> $\mathrm { N R M S E } _ { \Delta G }$  →</td></tr><tr><td>LEAP-Local</td><td>One transition</td><td>1.0013±0.0015</td></tr><tr><td>Transition NCE</td><td>One transition</td><td>1.0008±0.0013</td></tr><tr><td>LEAP-Global</td><td>Complete path + features</td><td> $\mathbf { 0 . 2 2 8 { \scriptstyle \pm 0 . 0 1 1 } }$ </td></tr></table>

Table 2: Ablation on trajectory information. Models using one transition at a time remain near the constant prediction level, while the complete-path model reduces NRMSE.  
Figure 6: Recovery of the nonadditive orbital correction. LEAP-Global follows the exact log cosh A dependence across held-out trajectories.
<table><tr><td>Objective</td><td>Input</td><td>NRMSE△G ↓</td></tr><tr><td>Categorical NRE (LEAP)</td><td>Transformer + features</td><td>0.228±0.015</td></tr><tr><td>Binary NRE</td><td>Path encoder</td><td>1.00000±3×10−5</td></tr><tr><td>Binary NRE</td><td>Transformer + features</td><td>0.175±0.011</td></tr></table>

Table 3: Ablation on accumulated features. Adding the accumulated features reduces error for both binary and categorical ratio objectives.

## 4.3 REUSING LEARNED ACTIONS ACROSS CONDITIONS AND OBJECTIVES

In molecular simulation, trajectories may come from a high-dimensional simulator but be analyzed in a small set of collective coordinates, i.e., the effective dynamics are not available in closed form. These coordinates are used to identify metastable states and describe transition pathways. They are also used in enhanced sampling and restrained simulations, where one may require a path to pass through a particular basin or compare conformational pathways across temperatures. We therefore evaluate whether LEAP can learn one action in these reduced coordinates and reuse it across these different settings. We use alanine dipeptide represented by the backbone angles (ϕ, ψ) on the Ramachandran torus (Vargas et al., 2002). OpenMM Langevin simulations provide 169 reactive C7eq→C7ax trajectories at 500 K and 613 at 600 K (Eastman et al., 2023).

Completion from partial observations. We keep the learned action fixed and optimize the missing states. When the endpoints and three interior states are observed, LEAP gives circular MSE 0.0931± 0.0016 at 600 K, compared to $0 . 1 1 9 \pm 0 . 0 1 2$ for conditional DDPM (Ho et al., 2020) and 0.125 ± 0.011 for conditional flow matching (Lipman et al., 2023). Other learned actions that share the same reference, from velocity regression and binary NRE, reach similar error (0.0900 and 0.0897), so with dense observations the gain over the generators comes from optimizing a learned path objective. With endpoints only, the observations constrain the path less and the learned action matters more. Among action-based methods, LEAP then gives the lowest error, 0.6225 compared to 0.6609 for velocity regression and 0.6682 for binary NRE. That said, the conditional generators, which are trained with completion masks, achieve lower error in this setting. LEAP is trained only on complete trajectories and uses the same learned action for both observation patterns, so the observation pattern can change without training a new model. Fig. 7a shows one example, with full results in §F.

Transfer across temperatures. We keep the learned correction fixed and use the diffusion, temperature and reference process of the target system. Training at 600 K gives mean within-context AUROC $0 . 9 5 7 9 \pm 0 . 0 0 5 7$ at 500 K, compared to 0.9403 ± 0.0099 for direct 500 K training, likely because the 600 K training set is larger (463 vs 130 trajectories). Training at 500 K gives $0 . 9 7 1 9 \pm 0 . 0 0 6 8$ at 600 K, compared to $0 . { \dot { 9 } } 7 5 4 \pm 0 . 0 0 1 2$ for direct 600 K training. The learned correction therefore remains useful after changing temperature.

Adding preferences and constraints. Next, using the same learned 600 K action, we change the inference objective. Increasing the basin preference weight from 0 to 20 increases the mean preference M from 0.0658 to 0.2517, while the projected transition cost increases from 0.9389 to 0.9809 nats per active step. We also impose $| \phi | \leq \mathrm { i } 5 0 ^ { \circ }$ while keeping the same basin preference. In Fig. 7c, the preference-only path enters the excluded angular region, while the constrained path stays within the bound and reaches the same endpoint. Thus, we can add preferences and geometric constraints without retraining the learned action.

## 5 CONCLUSION

In this work, we introduced action learning, which treats the stochastic action as a reusable learning target for trajectory data, and LEAP, which learns this action by comparing complete trajectories.

Observed End  
a Completion  
![](images/c24373013d0ee22310981823755d53252475d69734e6b5686a7a8a7e731814f3.jpg)  
φ(degrees)

b Preference weight  
![](images/a5e8696116934daef73054e27ba9442af72372c39418a996e1b2b91f909eba2d.jpg)  
φ (degrees)  
λ = 0 λ = 20 λ = 5 λ = 50

c Bound + preference  
![](images/1b5006c516dd094d26795ea8c8ad8360c67334538068a19d3cfd9056cb2e251f.jpg)  
φ(degrees)  
Preference only Both

Figure 7: Reuse of a fixed molecular action. All panels use the same 600 K LEAP action. (a) Completion from endpoints and three interior observations. (b) Increasing the basin preference raises M from 0.23 to 0.81 for the shown path. (c) Adding $| \phi | \leq 1 5 0 ^ { \circ }$ changes the route while preserving the terminal condition. Diamonds and squares are endpoints, dots are saved frames and the background is empirical free energy.

In the double well setting, the learned action recovers the underlying physical drift and supports path inference under new endpoints and horizons. In the orbital benchmark, it captures interactions across trajectories induced by a persistent latent regime that cannot be represented by additive transition models. In molecular coordinates, the same learned action enables completion, temperature transfer, basin preferences and angular constraints without retraining. These experiments show that learning an objective on path space provides a common interface between trajectory data and diverse downstream inference problems, including settings where the reduced dynamics are unavailable in closed form. Looking forward, we hope action learning encourages a broader view for trajectory modeling, in particular, shifting the focus to learning reusable objectives over complete paths.

Limitations. LEAP requires sufficient overlap between the reference and observed path distributions, as poor overlap makes the learned correction harder to estimate accurately. While the structured correction in Eq. (9) is exact for Gaussian Markov processes with shared diffusion, state-dependent diffusion requires more likelihood ratio terms. Finally, projecting dynamics onto reduced coordinates can introduce memory and friction or diffusion that depends on the state (Ma et al., 2016; Vroylandt & Monmarche´, 2022). Our alanine experiments therefore learn an effective action in (ϕ, ψ) rather than the full molecular action

## IMPACT STATEMENT

This paper presents work whose goal is to advance the field of Machine Learning. Learned trajectory objectives may help analyze and steer dynamical systems in science, robotics, planning and other sequential domains. As with other learned objectives, they could also be used to steer systems toward harmful outcomes, so deployment requires domain validation and careful attention to the data.

## ACKNOWLEDGEMENTS

SL is supported by a Kempner Graduate Fellowship.

## REFERENCES

Artur B. Adib. Stochastic actions for diffusive dynamics: Reweighting, sampling, and minimization. The Journal of Physical Chemistry B, 112(19):5910–5916, November 2007. ISSN 1520-5207. doi: 10.1021/jp0751458. URL http://dx.doi.org/10.1021/jp0751458.

Ishan Amin, Sanjeev Raja, and Aditi Krishnapriyan. Towards fast, specialized machine learning force fields: Distilling foundation models via energy hessians, 2025. URL https://arxiv. org/abs/2501.09009.

Peter G Bolhuis, David Chandler, Christoph Dellago, and Phillip L Geissler. Transition path sampling: Throwing ropes over rough mountain passes, in the dark. Annual review of physical chemistry, 53(1):291–318, 2002.

Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, and David Duvenaud. Neural ordinary differential equations, 2019. URL https://arxiv.org/abs/1806.07366.

Kurtland Chua, Roberto Calandra, Rowan McAllister, and Sergey Levine. Deep reinforcement learning in a handful of trials using probabilistic dynamics models. In Proceedings of the 32nd International Conference on Neural Information Processing Systems, NIPS’18, pp. 4759–4770, Red Hook, NY, USA, 2018. Curran Associates Inc.

Yilun Du, Toru Lin, and Igor Mordatch. Model-based planning with energy-based models. In Leslie Pack Kaelbling, Danica Kragic, and Komei Sugiura (eds.), Proceedings of the Conference on Robot Learning, volume 100 of Proceedings of Machine Learning Research, pp. 374–383. PMLR, 30 Oct–01 Nov 2020. URL https://proceedings.mlr.press/v100/du20a. html.

Yuanqi Du, Michael Plainer, Rob Brekelmans, Chenru Duan, Frank Noe, Carla P Gomes, Alan Aspuru-Guzik, and Kirill Neklyudov. Doob’s lagrangian: A sample-efficient variational approach to transition path sampling. Advances in Neural Information Processing Systems, 37:65791– 65822, 2024.

Weinan E, Weiqing Ren, and Eric Vanden-Eijnden. String method for the study of rare events. Physical Review B, 66(5), August 2002. ISSN 1095-3795. doi: 10.1103/physrevb.66.052301. URL http://dx.doi.org/10.1103/PhysRevB.66.052301.

Peter Eastman, Raimondas Galvelis, Raul P. Pel´ aez, Charlles R. A. Abreu, Stephen E. Farr, Emilio´ Gallicchio, Anton Gorenko, Michael M. Henry, Frank Hu, Jing Huang, Andreas Kramer, Julien¨ Michel, Joshua A. Mitchell, Vijay S. Pande, Joao PGLM Rodrigues, Jaime Rodriguez-Guerra,˜ Andrew C. Simmonett, Sukrit Singh, Jason Swails, Philip Turner, Yuanqing Wang, Ivy Zhang, John D. Chodera, Gianni De Fabritiis, and Thomas E. Markland. Openmm 8: Molecular dynamics simulation with machine learning potentials, 2023. URL https://arxiv.org/abs/2310. 03121.

Mark I. Freidlin and Alexander D. Wentzell. Random Perturbations ofDynamical Systems, volume 260 of Grundlehren der mathematischen Wissenschaften. Springer, Berlin, Heidelberg, 3 edition, 2012. ISBN 978-3-642-25847-3. doi: 10.1007/978-3-642-25847-3.

Justin Fu, Katie Luo, and Sergey Levine. Learning robust rewards with adversarial inverse reinforcement learning, 2018. URL https://arxiv.org/abs/1710.11248.

J. D. Giorgini, D. K. Yeomans, A. B. Chamberlin, P. W. Chodas, R. A. Jacobson, M. S. Keesey, J. H. Lieske, S. J. Ostro, E. M. Standish, and R. N. Wimberly. JPL’s On-Line Solar System Ephemeris and Data Service. In Bulletin of the American Astronomical Society, volume 28, pp. 1099, September 1997.

I. V. Girsanov. On transforming a certain class of stochastic processes by absolutely continuous substitution of measures. Theory of Probability & Its Applications, 5(3):285–301, 1960. doi: 10.1137/1105027. URL https://doi.org/10.1137/1105027.

Aldo Glielmo, Brooke E. Husic, Alex Rodriguez, Cecilia Clementi, Frank Noe, and Alessandro´ Laio. Unsupervised learning methods for molecular simulation data. Chemical Reviews, 121:9722 – 9758, 2021. URL https://api.semanticscholar.org/CorpusID:233743060.

Michael Gutmann and Aapo Hyvarinen. Noise-contrastive estimation: A new estimation princi-¨ ple for unnormalized statistical models. In Yee Whye Teh and Mike Titterington (eds.), Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, volume 9 of Proceedings of Machine Learning Research, pp. 297–304, Chia Laguna Resort, Sardinia, Italy, 13–15 May 2010. PMLR. URL https://proceedings.mlr.press/v9/ gutmann10a.html.

Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 2555–2565. PMLR, 09–15 Jun 2019. URL https://proceedings.mlr.press/v97/hafner19a.html.

Jiaqi Han, Minkai Xu, Aaron Lou, Haotian Ye, and Stefano Ermon. Geometric trajectory diffusion models. Advances in Neural Information Processing Systems, 37:25628–25662, 2024.

Jonathan Ho and Stefano Ermon. Generative adversarial imitation learning, 2016. URL https: //arxiv.org/abs/1606.03476.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS ’20, Red Hook, NY, USA, 2020. Curran Associates Inc. ISBN 9781713829546.

Lars Holdijk, Yuanqi Du, Ferry Hooft, Priyank Jaini, Berend Ensing, and Max Welling. Stochastic optimal control for collective variable free sampling of molecular transition paths. Advances in Neural Information Processing Systems, 36:79540–79556, 2023.

Michael Janner, Yilun Du, Joshua B. Tenenbaum, and Sergey Levine. Planning with diffusion for flexible behavior synthesis, 2022. URL https://arxiv.org/abs/2205.09991.

Bowen Jing, Bonnie Berger, and Tommi Jaakkola. Alphafold meets flow matching for generating protein ensembles, 2024a. URL https://arxiv.org/abs/2402.04845.

Bowen Jing, Hannes Stark, Tommi Jaakkola, and Bonnie Berger. Generative modeling of¨ molecular dynamics trajectories. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 40534–40564. Curran Associates, Inc., 2024b. doi: 10.52202/ 079017-1282. URL https://proceedings.neurips.cc/paper\_files/paper/ 2024/file/478b06f60662d3cdc1d4f15d4587173a-Paper-Conference.pdf.

Mrinal Kalakrishnan, Sachin Chitta, Evangelos Theodorou, Peter Pastor, and Stefan Schaal. Stomp: Stochastic trajectory optimization for motion planning. In 2011 IEEE international conference on robotics and automation, pp. 4569–4574. IEEE, 2011.

H J Kappen. Path integrals and symmetry breaking for optimal control theory. Journal ofStatistical Mechanics: Theory and Experiment, 2005(11):P11011–P11011, November 2005. ISSN 1742- 5468. doi: 10.1088/1742-5468/2005/11/p11011. URL http://dx.doi.org/10.1088/ 1742-5468/2005/11/P11011.

Patrick Kidger, James Foster, Xuechen Li, and Terry J Lyons. Neural sdes as infinite-dimensional gans. In Marina Meila and Tong Zhang (eds.), Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 5453–5463. PMLR, 18–24 Jul 2021. URL https://proceedings.mlr.press/v139/ kidger21b.html.

Xuechen Li, Ting-Kam Leonard Wong, Ricky T. Q. Chen, and David Duvenaud. Scalable gradients for stochastic differential equations. In Silvia Chiappa and Roberto Calandra (eds.), Proceedings ofthe Twenty Third International Conference on Artificial Intelligence and Statistics, volume 108 of Proceedings of Machine Learning Research, pp. 3870–3882. PMLR, 26–28 Aug 2020. URL https://proceedings.mlr.press/v108/li20i.html.

Kresten Lindorff-Larsen, Stefano Piana, Ron O. Dror, and David E. Shaw. How fast-folding proteins fold. Science, 334(6055):517–520, 2011. doi: 10.1126/science.1208351. URL https://www. science.org/doi/abs/10.1126/science.1208351.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=PqvMRDCJT9t.

Lina Ma, Xiantao Li, and Chun Liu. The derivation and approximation of coarse-grained dynamics from langevin dynamics. The Journal of Chemical Physics, 145(20), November 2016. ISSN 1089-7690. doi: 10.1063/1.4967936. URL http://dx.doi.org/10.1063/1.4967936.

Benjamin Kurt Miller, Christoph Weniger, and Patrick Forre. Contrastive neural ratio estimation for´ simulation-based inference, 2024. URL https://arxiv.org/abs/2210.06170.

Kirill Neklyudov, Rob Brekelmans, Daniel Severo, and Alireza Makhzani. Action matching: Learning stochastic dynamics from samples, 2023. URL https://arxiv.org/abs/2210. 06662.

Felipe Nuti, Tim Franzmeyer, and Joao F. Henriques. Extracting reward functions from diffusion˜ models, 2023. URL https://arxiv.org/abs/2306.01804.

L. Onsager and S. Machlup. Fluctuations and irreversible processes. Phys. Rev., 91:1505–1512, Sep 1953. doi: 10.1103/PhysRev.91.1505. URL https://link.aps.org/doi/10.1103/ PhysRev.91.1505.

Sanjeev Raja, Martin S<sup>ˇ</sup>´ıpka, Michael Psenka, Tobias Kreiman, Michal Pavelka, and Aditi S. Krishnapriyan. Action-minimization meets generative modeling: Efficient transition path sampling with the onsager-machlup functional, 2025. URL https://arxiv.org/abs/2504. 18506.

David E. Shaw, Paul Maragakis, Kresten Lindorff-Larsen, Stefano Piana, Ron O. Dror, Michael P. Eastwood, Joseph A. Bank, John M. Jumper, John K. Salmon, Yibing Shan, and Willy Wriggers. Atomic-level characterization of the structural dynamics of proteins. Science, 330(6002):341– 346, 2010. doi: 10.1126/science.1187409. URL https://www.science.org/doi/abs/ 10.1126/science.1187409.

Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations, 2021. URL https://arxiv.org/abs/2011.13456.

Emanuel Todorov. Efficient computation of optimal actions. Proceedings ofthe National Academy of Sciences of the United States of America, 106(28):11478–11483, 2009. doi: 10.1073/pnas. 0710743106. URL https://doi.org/10.1073/pnas.0710743106.

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding, 2019. URL https://arxiv.org/abs/1807.03748.

Rubicelia Vargas, Jorge Garza, Benjamin P. Hay, and David A. Dixon. Conformational study of the alanine dipeptide at the mp2 and dft levels. The Journal ofPhysical Chemistry A, 106(13):3213– 3218, 2002. doi: 10.1021/jp013952f. URL https://doi.org/10.1021/jp013952f.

Hadrien Vroylandt and Pierre Monmarche. Position-dependent memory kernel in generalized´ langevin equations: Theory and numerical estimation. The Journal of Chemical Physics, 156 (24), June 2022. ISSN 1089-7690. doi: 10.1063/5.0094566. URL http://dx.doi.org/ 10.1063/5.0094566.

Grady Williams, Andrew Aldrich, and Evangelos A. Theodorou. Model predictive path integral control: From theory to parallel computation. Journal of Guidance, Control, and Dynamics, 40(2):344–357, 2017. doi: 10.2514/1.G001921. URL https://doi.org/10.2514/1. G001921.

Eric C.-Y. Yuan, Anup Kumar, Xingyi Guan, Eric D. Hermes, Andrew S. Rosen, Judit Zador,´ Teresa Head-Gordon, and Samuel M. Blau. Analytical ab initio hessian from a deep learning potential for transition state optimization. Nature Communications, 15(1), 10 2024. doi: 10.1038/s41467-024-52481-5.

Brian D. Ziebart, Andrew Maas, J. Andrew Bagnell, and Anind K. Dey. Maximum entropy inverse reinforcement learning. In Proceedings ofthe 23rd National Conference on Artificial Intelligence - Volume 3, AAAI’08, pp. 1433–1438. AAAI Press, 2008. ISBN 9781577353683.

## APPENDIX

The appendix is organized as follows. §A presents the theoretical results on action recovery, Gaussian drift corrections, conditioning, nonadditive actions and downstream inference. § B describes the shared evaluation metrics, training procedures and inference protocols. § C provides additional details on the double well experiments, while § D examines reference choice, temporal resolution and training budget. §E presents the orbital results, model comparisons and scaling experiments. Finally, §F details the molecular experiments, including completion, temperature transfer, preferences, constraints and additional validation.

## A THEORY AND DERIVATIONS

We additionally provide the assumptions and derivations in § 3, which includes population action recovery, the structured Gaussian correction and drift interpretation, matched trajectory selection, nonadditive actions and downstream stability.

Notation. Fix a context c and let $\Omega _ { c }$ be the trajectory space. The densities $p _ { \mathrm { d a t a } } ( \cdot \mid c )$ and $p _ { \mathrm { r e f } } ( \cdot \mid c )$ are defined with respect to the same base measure on $\bar { \Omega } _ { c } .$ . We use the target correction $G ^ { * }$ and target action $S ^ { * }$ defined in Eqs. (1) and (2).

## A.1 ACTION RECOVERY FROM TRAJECTORY COMPARISONS

Recall that LEAP learns the correction by identifying one observed trajectory among $K$ reference trajectories drawn under the same context. At the population level, this objective recovers the trajectory density ratio for every $K \geq 1$

Fix a context c and write $p = p _ { \mathrm { d a t a } } ( \cdot | \ c )$ and $q = p _ { \mathrm { r e f } } ( \cdot | \ c )$ , so that

$$
G ^ { * } ( \Phi ; c ) = \log \frac { p ( \Phi ) } { q ( \Phi ) } .
$$

Draw an index I uniformly from $\{ 0 , \ldots , K \}$ . Conditional on $I = i$ , independently draw

$$
\Phi _ { i } \sim p , \qquad \Phi _ { j } \sim q , \quad j \neq i .\tag{A.1}
$$

The corresponding population objective is

$$
\mathcal { L } _ { K } ( G ; c ) = - \mathbb { E } \left[ \log \frac { \exp G ( \Phi _ { I } ; c ) } { \sum _ { j = 0 } ^ { K } \exp G ( \Phi _ { j } ; c ) } \Bigg | c \right] .\tag{A.2}
$$

By symmetry, this is the population form of Eq. (4).

Theorem A.1 (Population action recovery). Suppose p and q are positive on the same support, the candidates in $E q . \ ( \mathrm { A } . 1 )$ are conditionally independent and the model class contains $G ^ { * }$ . Then, for every $K \geq 1$ , every population minimizer satisfies

$$
G _ { \theta ^ { * } } ( \Phi ; c ) = G ^ { * } ( \Phi ; c ) + a ( c )\tag{A.3}
$$

almost everywhere, where $a ( c )$ is independent $o f \Phi$ $\begin{array} { r } { I f \mathbb { E } _ { q } | G ^ { * } | < \infty , } \end{array}$ , then the reference-centered action satisfies

$$
S _ { \theta ^ { * } } ( \Phi ; c ) = S ^ { * } ( \Phi ; c ) + \beta ( c )\tag{A.4}
$$

for a constant $\beta ( c )$ independent of Φ.

Proof. By Bayes’ rule,

$$
\mathbb { P } ( I = i \mid \Phi _ { 0 : K } , c ) = \frac { p ( \Phi _ { i } ) \prod _ { j \neq i } q ( \Phi _ { j } ) } { \sum _ { \ell = 0 } ^ { K } p ( \Phi _ { \ell } ) \prod _ { j \neq \ell } q ( \Phi _ { j } ) } = \frac { p ( \Phi _ { i } ) / q ( \Phi _ { i } ) } { \sum _ { \ell = 0 } ^ { K } p ( \Phi _ { \ell } ) / q ( \Phi _ { \ell } ) } = \frac { \exp G ^ { * } ( \Phi _ { i } ; c ) } { \sum _ { \ell = 0 } ^ { K } \exp G ^ { * } ( \Phi _ { \ell } ; c ) } .\tag{A.5}
$$

The comparison loss is minimized when the model matches this posterior. Therefore, for any two candidates $\Phi _ { i }$ and $\Phi _ { j } , G _ { \theta ^ { * } } ( \Phi _ { i } ; c ) - G _ { \theta ^ { * } } ( \Phi _ { j } ; c ) = G ^ { * } ( \Phi _ { i } ; c ) - G ^ { * } ( \Phi _ { j } ; c )$ , which implies $G _ { \theta ^ { * } } ( \Phi ; c ) = G ^ { * } ( \Phi ; c ) + \tilde { a ( c ) }$ almost everywhere. Then, the constant $a ( c )$ cancels from the softmax.

Reference centering then gives

$$
\widetilde { G } _ { \theta ^ { * } } ( \Phi ; c ) = G _ { \theta ^ { * } } ( \Phi ; c ) - \mathbb { E } _ { q } G _ { \theta ^ { * } } ( \Phi ^ { \prime } ; c ) = G ^ { * } ( \Phi ; c ) - \mathbb { E } _ { q } G ^ { * } ( \Phi ^ { \prime } ; c ) .\tag{A.6}
$$

Substituting into Eq. (7),

$$
S _ { \theta ^ { * } } ( \Phi ; c ) = - \log q ( \Phi ) - \widetilde G _ { \theta ^ { * } } ( \Phi ; c ) = - \log p ( \Phi ) + \mathbb { E } _ { q } G ^ { * } ( \Phi ^ { \prime } ; c ) = S ^ { * } ( \Phi ; c ) + \beta ( c ) ,\tag{A.7}
$$

where $\beta ( c ) = \mathbb { E } _ { q } G ^ { * } ( \Phi ^ { \prime } ; c ) - \alpha ( c )$

Reference centering selects an additive convention for the learned correction. With finite data, correction accuracy then depends on reference coverage, the number of reference trajectories, model capacity and optimization.

## A.2 STRUCTURED CORRECTION AND DRIFT IDENTIFICATION

Consider the Euler–Maruyama process

$$
x _ { k + 1 } = x _ { k } + b ( x _ { k } , t _ { k } ; c ) \Delta t + \sqrt { 2 D \Delta t } \xi _ { k } , \qquad \xi _ { k } \overset { \mathrm { i i d } } { \sim } \mathcal { N } ( 0 , I ) .\tag{A.8}
$$

Let $v _ { k } = ( x _ { k + 1 } - x _ { k } ) / \Delta t$ . Its transition density is

$$
q _ { b } ( x _ { k + 1 } \mid x _ { k } , c ) = ( 4 \pi D \Delta t ) ^ { - d / 2 } \exp \left[ - \frac { \Delta t } { 4 D } \| v _ { k } - b ( x _ { k } , t _ { k } ; c ) \| ^ { 2 } \right] ,\tag{A.9}
$$

so for a fixed initial state,

$$
- \log p _ { b } ( \Phi \mid c ) = C _ { \mathrm { E M } } ( c ) + \sum _ { k = 0 } ^ { N - 1 } \frac { \Delta t } { 4 D } \| v _ { k } - b ( x _ { k } , t _ { k } ; c ) \| ^ { 2 } ,\tag{A.10}
$$

where $\begin{array} { r } { C _ { \mathrm { E M } } ( c ) = \frac { N d } { 2 } \log ( 4 \pi D \Delta t ) } \end{array}$

Suppose the data and reference processes share the diffusion, time grid and initial law and differ via $b _ { \mathrm { d a t a } } = b _ { \mathrm { r e f } } + u ^ { * }$ . Writing $u _ { k } = u ^ { \ast } ( x _ { k } , t _ { k } ; c )$ , the Gaussian likelihood ratio gives

$$
G ^ { * } ( \Phi ; c ) = \sum _ { k = 0 } ^ { N - 1 } \left[ \frac { u _ { k } ^ { \top } \left( v _ { k } - b _ { \mathrm { r e f } } ( x _ { k } , t _ { k } ; c ) \right) } { 2 D } - \frac { \| u _ { k } \| ^ { 2 } } { 4 D } \right] \Delta t .\tag{A.11}
$$

Recall that this is the structured correction in Eq. (9). Subtracting it from the reference action gives

$$
S _ { \mathrm { r e f } } ( \Phi ; c ) - G ^ { * } ( \Phi ; c ) = C _ { \mathrm { E M } } ( c ) + \sum _ { k = 0 } ^ { N - 1 } { \frac { \Delta t } { 4 D } } \left\| v _ { k } - b _ { \mathrm { d a t a } } ( x _ { k } , t _ { k } ; c ) \right\| ^ { 2 } .\tag{A.12}
$$

Thus the structured correction parameterizes the drift difference exactly under the shared-diffusion assumptions. This discrete likelihood ratio is the Euler–Maruyama analogue of the shared-diffusion Girsanov likelihood ratio (Girsanov, 1960).

Proposition A.2 (Drift identification under shared diffusion). Let $P _ { b }$ and $P _ { \widetilde { b } }$ be the unconditioned Euler–Maruyama trajectory distributions with the same initial law, step size $\Delta t > 0$ and diffusion coefficient $\dot { D } > 0$ . Then

$$
\mathrm { K L } ( P _ { b } \| P _ { \overline { { b } } } ) = \frac { \Delta t } { 4 D } \sum _ { k = 0 } ^ { N - 1 } \mathbb { E } _ { P _ { b } } \left[ \left\| b ( X _ { k } , t _ { k } ; c ) - \widetilde { b } ( X _ { k } , t _ { k } ; c ) \right\| ^ { 2 } \right] .\tag{A.13}
$$

Therefore, equality of the trajectory distributions implies equality of their drifts at states and times visited by the process.

Proof. The KL chain rule decomposes the path divergence into expected transition divergences. Because the corresponding Gaussian transitions have covariance $2 D \Delta t I$

$$
{ \mathrm { K L } } \left( q _ { b } ( \cdot \ | \ X _ { k } , c ) \| q _ { \widetilde { b } } ( \cdot \ | \ X _ { k } , c ) \right) = \frac { \Delta t } { 4 D } \left\| b ( X _ { k } , t _ { k } ; c ) - \widetilde { b } ( X _ { k } , t _ { k } ; c ) \right\| ^ { 2 } .
$$

Summing over k gives Eq. (A.13). If the path distributions are equal, the KL divergence is zero, so the drift difference vanishes almost surely on every occupied state-time pair. □

For structured LEAP, $b = b _ { \mathrm { r e f } } + u ^ { * }$ and $\widetilde { b } = b _ { \mathrm { r e f } } + u _ { \theta ^ { * } }$ . Recovering the unconditioned path distribution then identifies $u _ { \theta ^ { * } } = u ^ { * }$ on the support of the observed states and times.

## A.3 MATCHED CONDITIONING AND SELECTED TRAJECTORIES

Trajectory datasets are often restricted to a fixed endpoint or terminal event. When the data and reference use the same condition, conditioning changes their path log-likelihood ratio only by a constant within that condition.

Proposition A.3 (Matched conditioning preserves action differences). Let $Y = \tau ( \Phi )$ be a trajectory statistic. Suppose the data and reference admit conditional densities given $Y = y$ with respect to the same base measure and both assign positive density to $Y = y .$ . Then, for almost every Φ with $\tau ( \Phi ) = y$

$$
\log \frac { p _ { \mathrm { d a t a } } ( \Phi \mid Y = y , c ) } { p _ { \mathrm { r e f } } ( \Phi \mid Y = y , c ) } = \log \frac { p _ { \mathrm { d a t a } } ( \Phi \mid c ) } { p _ { \mathrm { r e f } } ( \Phi \mid c ) } + \log \frac { p _ { \mathrm { r e f } } ^ { Y } ( y \mid c ) } { p _ { \mathrm { d a t a } } ^ { Y } ( y \mid c ) } .\tag{A.14}
$$

Proof. For $\tau ( \Phi ) = y { \mathrm { . } }$

$$
\displaystyle p _ { \mathrm { d a t a } } ( \Phi \mid Y = y , c ) = { \frac { p _ { \mathrm { d a t a } } ( \Phi \mid c ) } { p _ { \mathrm { d a t a } } ^ { Y } ( y \mid c ) } } , \qquad p _ { \mathrm { r e f } } ( \Phi \mid Y = y , c ) = { \frac { p _ { \mathrm { r e f } } ( \Phi \mid c ) } { p _ { \mathrm { r e f } } ^ { Y } ( y \mid c ) } } .
$$

Taking their log ratio gives $\operatorname { E q . } \ ( \mathrm { A . 1 4 } )$ . The last term depends on y and $c .$ Therefore conditioning preserves action differences between trajectories with the same condition. □

Setting $Y = X _ { N }$ gives exact endpoint conditioning and setting $Y = \mathbb { 1 } \{ X _ { N } \in B \}$ and conditioning on $Y = 1$ gives conditioning on a terminal region.

Conditioning preserves complete path likelihood ratios up to a constant, but it changes the one-step transition distribution. For the terminal event $\mathcal { E } = \{ X _ { N } \in B \}$ , define $h _ { k } ( x ; c ) = P \bar { ( } \mathcal { E } \mid X _ { k } = x , c \bar { ) }$ The conditioned transition kernel is

$$
q _ { k } ^ {  { \varepsilon } } ( x ^ { \prime } \mid x , c ) = q _ { k } ( x ^ { \prime } \mid x , c ) \frac { h _ { k + 1 } ( x ^ { \prime } ; c ) } { h _ { k } ( x ; c ) } .\tag{A.15}
$$

For smooth positive h in continuous time, the corresponding drift is

$$
b ^ { \mathcal { E } } ( x , t ) = b ( x , t ) + 2 D \nabla \log h ( x , t ) .\tag{A.16}
$$

Thus local regression on selected trajectories estimates $b ^ { \varepsilon }$

Remark A.4 (Drift identification under fixed endpoints). Matched conditioning alone does not identify the unconditioned drift. Consider constant drift $b ( x , t ) = v$ relative to zero drift. The path log-likelihood ratio is

$$
\log \frac { p _ { v } ( \Phi \mid x _ { 0 } ) } { p _ { 0 } ( \Phi \mid x _ { 0 } ) } = \sum _ { k = 0 } ^ { N - 1 } \left[ \frac { v ^ { \top } ( x _ { k + 1 } - x _ { k } ) } { 2 D } - \frac { \Delta t \| v \| ^ { 2 } } { 4 D } \right] = \frac { v ^ { \top } ( x _ { N } - x _ { 0 } ) } { 2 D } - \frac { T \| v \| ^ { 2 } } { 4 D } .\tag{A.17}
$$

The intermediate states cancel from the ratio, which leaves only the initial and final states $( x _ { 0 } , x _ { N } )$ When both endpoints are fixed, the ratio is therefore the same for every trajectory connecting them, so it is not possible to distinguish different values of v from these trajectories alone. Thus, Prop. A.2 establishes drift identification for unconditioned trajectory distributions. We separately evaluate drift recovery from trajectories selected by their final states in §4.1.

## A.4 NONADDITIVE ACTION FROM A PERSISTENT LATENT REGIME

A latent physical parameter may remain fixed for an entire trajectory without appearing in the observed state. Marginalizing this parameter can introduce interactions between transitions, resulting in an action that cannot simply be expressed as a sum of individual transition costs.

Let $z \in \{ - 1 , + 1 \}$ be sampled once with equal probability and held fixed for the complete trajectory. Conditional on z, assume the path density factorizes as $\begin{array} { r } { p _ { z } ( \Phi \mid c ) = \prod _ { k = 0 } ^ { N - 1 } q _ { z , k } ( x _ { k + 1 } \mid x _ { k } , c ) } \end{array}$ , with reference $\begin{array} { r } { p _ { 0 } ( \Phi ~ \vert ~ c ) = \prod _ { k = 0 } ^ { N - 1 } q _ { 0 , k } ( x _ { k + 1 } ~ \vert ~ x _ { k } , c ) } \end{array}$ . Any shared initial density cancels from the likelihood ratio.

Define $\ell _ { z , k } = \log q _ { z , k } ( x _ { k + 1 } \mid x _ { k } , c )$ and $\ell _ { 0 , k } = \log q _ { 0 , k } ( x _ { k + 1 } \mid x _ { k } , c )$ and let

$$
d _ { k } = \frac { \ell _ { + 1 , k } - \ell _ { - 1 , k } } { 2 } , \qquad m _ { k } = \frac { \ell _ { + 1 , k } + \ell _ { - 1 , k } } { 2 } - \ell _ { 0 , k } .\tag{A.18}
$$

Proposition A.5 (Marginalizing latent dynamics induces nonadditive actions). Let $p _ { \mathrm { d a t a } } ( \Phi \mid c ) =$ $\begin{array} { r } { \frac { 1 } { 2 } p _ { + 1 } ( \Phi \mid c ) + \frac { 1 } { 2 } p _ { - 1 } ( \bar { \Phi } \mid c ) } \end{array}$ . Then

$$
G ^ { * } ( \Phi ; c ) = \sum _ { k = 0 } ^ { N - 1 } m _ { k } + \log \cosh \left( \sum _ { k = 0 } ^ { N - 1 } d _ { k } \right) .\tag{A.19}
$$

For any $i \neq j$

$$
\frac { \partial ^ { 2 } G ^ { * } } { \partial d _ { i } \partial d _ { j } } = \mathrm { s e c h } ^ { 2 } \left( \sum _ { k = 0 } ^ { N - 1 } d _ { k } \right) > 0 .\tag{A.20}
$$

Therefore, on any region where $d _ { i }$ and $d _ { j }$ can vary independently, $G ^ { * }$ cannot be represented by a twice differentiable additive model $\begin{array} { r } { G ( \Phi ; \bar { c } ) = \sum _ { k } g _ { k } ( d _ { k } , m _ { k } ) } \end{array}$

Proof. Using the conditional factorization,

$$
G ^ { * } ( \Phi ; c ) = \log \left[ \frac 1 2 \exp \left( \sum _ { k } \ell _ { + 1 , k } - \ell _ { 0 , k } \right) + \frac 1 2 \exp \left( \sum _ { k } \ell _ { - 1 , k } - \ell _ { 0 , k } \right) \right] .
$$

By Eq. (A.18), $\ell _ { + 1 , k } - \ell _ { 0 , k } = m _ { k } + d _ { k }$ and $\ell _ { - 1 , k } - \ell _ { 0 , k } = m _ { k } - d _ { k }$ . Therefore,

$$
G ^ { * } ( \Phi ; c ) = \sum _ { k } m _ { k } + \log \left[ \frac { 1 } { 2 } e ^ { \sum _ { k } d _ { k } } + \frac { 1 } { 2 } e ^ { - \sum _ { k } d _ { k } } \right] = \sum _ { k } m _ { k } + \log \cosh \left( \sum _ { k } d _ { k } \right) .
$$

Let $A = \textstyle \sum _ { k } d _ { k }$ . Then $\partial G ^ { * } / \partial d _ { i } = \operatorname { t a n h } A .$ , so for $\begin{array} { r } { i \neq j , \frac { \partial ^ { 2 } G ^ { * } } { \partial d _ { i } \partial d _ { i } } = \mathrm { s e c h } ^ { 2 } A > 0 } \end{array}$

In contrast, any additive model satisfies $\begin{array} { r } { \partial ^ { 2 } ( \sum _ { k } g _ { k } ( d _ { k } , m _ { k } ) ) / ( \partial d _ { i } \partial d _ { j } ) = 0 \mathrm { f o r } i \neq j . } \end{array}$

## A.5 DOWNSTREAM STABILITY UNDER ACTION REUSE

Fix a context c and feasible set Γ(c) and suppose

$$
\operatorname* { s u p } _ { \Phi \in \Gamma ( c ) } | S _ { \theta } ( \Phi ; c ) - S ^ { * } ( \Phi ; c ) - \beta ( c ) | \le \varepsilon .\tag{A.21}
$$

For any added path functional $H ,$ , define $F _ { \theta } = S _ { \theta } + H$ and $F ^ { * } = S ^ { * } + H$

Theorem A.6 (Stability under action reuse). $I f { \widehat { \Phi } }$ is δ-suboptimalfor $F _ { \theta } ,$ , then

$$
F ^ { * } ( \widehat \Phi ; c ) - \operatorname* { i n f } _ { \Phi \in \Gamma ( c ) } F ^ { * } ( \Phi ; c ) \leq 2 \varepsilon + \delta .\tag{A.22}
$$

$H \pi _ { \theta } \propto e ^ { - F _ { \theta } }$ and $\pi ^ { * } \propto e ^ { - F ^ { * } }$ havefinite positive normalizing constants, then

$$
\left| \log { \frac { \pi _ { \theta } ( \Phi \mid c ) } { \pi ^ { * } ( \Phi \mid c ) } } \right| \leq 2 \varepsilon \qquad \forall \Phi \in \Gamma ( c ) .\tag{A.23}
$$

Proof. Write $e ( \Phi ) = S _ { \theta } ( \Phi ; c ) - S ^ { * } ( \Phi ; c ) - \beta ( c ) , \mathbf { s o } \ | e ( \Phi ) | \le \varepsilon . \ \mathrm { I f } \ \widehat { \Phi }$ is δ-suboptimal for $F _ { \theta } { \mathrm { . } }$ , then for every $\Phi \in \Gamma ( c ) , F ^ { * } ( \widehat \Phi ; c ) \leq F ^ { * } ( \Phi ; c ) + e ( \Phi ) - e ( \widehat \Phi ) + \delta \leq F ^ { * } ( \Phi ; c ) + 2 \varepsilon + \delta .$ which gives Eq. (A.22) after taking the inf.

For the normalized distributions,

$$
\log \frac { \pi _ { \theta } ( \Phi \mid c ) } { \pi ^ { * } ( \Phi \mid c ) } = - e ( \Phi ) - \log \mathbb { E } _ { \pi ^ { * } } [ e ^ { - e ( \Phi ) } ] .
$$

Since $| e | \le \varepsilon$ , the second term has magnitude at most ε, giving Eq. (A.23).

Because the same H appears in both objectives, these bounds continue to hold after adding new preferences or constraints.

## B EXPERIMENTAL DETAILS

We describe the evaluation metrics, training settings and inference procedures used across experiments in §4.

Compute resources. We run each training job on a single NVIDIA GPU (H200, H100, A100) with up to 8 CPU cores and 64 GB of memory. The reference study in §D runs on CPUs.

## B.1 METRICS AND UNCERTAINTY

Drift recovery. When the true drift is known, we report

$$
\mathrm { M S E } _ { b } = \frac { 1 } { n _ { x } } \sum _ { i = 1 } ^ { n _ { x } } \left\| \widehat { b } ( x _ { i } ) - b ^ { * } ( x _ { i } ) \right\| ^ { 2 } .\tag{B.1}
$$

We evaluate all methods on the same $n _ { x }$ states. For the double well, we use 301 states uniformly spanning [−1, 1].

Action correction error. Because the action is identified up to a context-dependent constant, we compare correction differences within each context. For $i , j \in I _ { c } ,$ , let $\Delta G _ { i j } = \mathsf { \bar { G } } ( \Phi _ { i } ; c ) - G ( \Phi _ { j } ; c )$ We define

$$
\mathrm { N R M S E } _ { \Delta G } = \left[ \frac { \sum _ { c } n _ { c } ^ { - 2 } \sum _ { i , j \in I _ { c } } \left( \Delta G _ { \theta , i j } - \Delta G _ { i j } ^ { * } \right) ^ { 2 } } { \sum _ { c } { n _ { c } ^ { - 2 } \sum _ { i , j \in I _ { c } } \left( \Delta G _ { i j } ^ { * } \right) ^ { 2 } } } \right] ^ { 1 / 2 } .\tag{B.2}
$$

Exact correction differences give 0 and a constant prediction gives 1.

Trajectory discrimination and ordering. We report within-context AUROC for separating data and reference trajectories. The metric does not measure correction scale, so we also use $\mathrm { \Delta N R M S E } _ { \Delta G }$ when we need numerical action differences.

Path quality. When the true drift is known, we evaluate inferred paths with

$$
Q _ { \mathrm { t r u e } } ( \Phi ) = \frac { \Delta t } { 4 } \sum _ { k = 0 } ^ { N - 1 } \left. \frac { x _ { k + 1 } - x _ { k } } { \Delta t } - b ^ { * } ( x _ { k } ) \right. ^ { 2 } .\tag{B.3}
$$

For the double well, $b ^ { * } ( x ) = x - x ^ { 3 }$ . We use this cost only for evaluation; each method performs inference with its own objective.

Uncertainty. We use at least three independent random seeds for every experiment. For all the tables, we report the sample mean±sample standard deviation.

## B.2 PATH INFERENCE

Algorithm 2 Path optimization with a learned LEAP action   
Require: learned action $S _ { \theta } ( \Phi ; c )$ , initialization $\Phi ^ { ( 0 ) } \in \Gamma ( c )$ , optional rewards $\{ M _ { j } \}$ , optional   
penalties $\{ C _ { \ell } \}$   
1: define   
$F ( \Phi ; c ) = S _ { \theta } ( \Phi ; c ) - \sum _ { j } \gamma _ { j } M _ { j } ( \Phi ) + \sum _ { \ell } \lambda _ { \ell } C _ { \ell } ( \Phi )$   
2: initialize $\Phi  \Phi ^ { ( 0 ) }$   
3: for r = 1 to R do   
4: update the free trajectory coordinates by one optimizer step on $F ( \Phi ; c )$   
5: project onto hard constraints when required   
6: if using knot redistribution then   
7: redistribute knots and reevaluate F before the next optimizer step   
8: end if   
9: end for   
10: return optimized path $\Phi ^ { * }$

<table><tr><td></td><td>Double well</td><td>Orbital</td><td>Alanine</td></tr><tr><td>Training paths</td><td>5,000</td><td>1,016</td><td>130 / 463</td></tr><tr><td>Reference paths K</td><td>8</td><td>4</td><td>8</td></tr><tr><td>Training budget</td><td>500 epochs</td><td>100 epochs</td><td>1,200 updates</td></tr><tr><td>Batch size</td><td>64</td><td>8</td><td>32</td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 3 }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Training seeds</td><td>3</td><td>3</td><td>3</td></tr></table>

Table 4: Training settings for the principal experiments. Molecular counts correspond to the 500 K and 600 K training sets. Orbital model inputs are described in §E.2.

<table><tr><td>Method</td><td>Definition</td></tr><tr><td>Kramers-Moyal</td><td>Binned mean local increment</td></tr><tr><td>Cubic velocity regression</td><td>Cubic least-squares fit to local increments</td></tr><tr><td>Zero field</td><td> $b ( x ) = 0$ </td></tr><tr><td>Geometric path</td><td>Fixed waiting and linear crossing path</td></tr><tr><td>Analytic drift</td><td>Exact drift and action</td></tr><tr><td>LEAP</td><td>Complete path comparison objective</td></tr></table>

Table 5: Double well methods and controls.

## B.3 TRAINING AND MODEL SELECTION

Table 4 summarizes the main settings. Additionally, for the double well, we use a potential network with three hidden layers of width 512, GELU activations and LayerNorm. We train with AdamW, weight decay $1 0 ^ { - 5 }$ , cosine learning rate decay to $1 0 ^ { - 5 }$ and gradient clipping at 1. For the orbital experiments, we use validation stopping with patience 20. § F gives the molecular architecture, reference and regularization settings.

## C DOUBLE WELL EXPERIMENTS

We provide additional details on the double well experiments, including trajectory generation, baselines, path inference and the analytic cost for long horizons.

## C.1 TRAJECTORY CONSTRUCTION AND BASELINES

Data and reference. We simulate from $x _ { 0 } = - 1 { \mathrm { ~ t o ~ } } T = 3$ with $D = 0 . 2 0$ and $\Delta t = 0 . 0 1$ and retain trajectories with $x _ { T } \in \left[ 0 . 9 , 1 . 1 \right]$ . Each seed uses 5,000 selected training trajectories. The reference is Brownian motion with the same diffusion and terminal condition. We sample $x _ { T }$ from its terminal density restricted to [0.9, 1.1], then sample the corresponding Gaussian bridge. The data and reference use the same terminal selection event as in Prop. A.3.

LEAP model. We parameterize the drift by a scalar potential, $b _ { \theta } ( x ) = - d U _ { \theta } ( x ) / d x$ . Training settings are given in Table 4.

Local drift baselines. Kramers-Moyal averages $( x _ { k + 1 } - x _ { k } ) / \Delta t$ in state bins of width $2 / 3 0 0$ Cubic velocity regression fits the same local target by least squares using $\{ 1 , x , x ^ { 2 } , x ^ { 3 } \}$ . This basis contains the exact drift $x - x ^ { 3 }$ , so its error does not come from function class misspecification. Both methods use the same selected trajectories as LEAP. By §A.3, terminal selection changes their local regression target.

## C.2 PATH INFERENCE UNDER NEW BOUNDARY CONDITIONS

Inference setting. We keep the learned potential fixed and change the endpoint and horizon. Training paths end in [0.9, 1.1] at $T = 3 ,$ , whereas inference fixes $x _ { 0 } = - 1$ and $x _ { T } = 0$ . We evaluate $T \in \{ 1 , 4 , 8 , 1 6 \}$ and optimize only the free path coordinates.

<table><tr><td></td><td>Drift MSE↓</td><td colspan="2">True path cost ↓</td></tr><tr><td>Method</td><td></td><td>T = 4</td><td>T = 16</td></tr><tr><td>Kramers-Moyal</td><td>0.8132±0.0030</td><td></td><td></td></tr><tr><td>Cubic velocity reg.</td><td>0.7752±0.0043</td><td>0.3189±0.0008</td><td>0.6126±0.0079</td></tr><tr><td>Zero field</td><td>0.0759</td><td>0.2637</td><td>0.4454</td></tr><tr><td>Geometric path</td><td></td><td>0.2637</td><td>0.2637</td></tr><tr><td>Analytic drift</td><td>0</td><td>0.2532</td><td>0.2500</td></tr><tr><td>LEAP</td><td> $\mathbf { ( 3 . 3 \pm 2 . 7 ) \times 1 0 ^ { - 4 } }$ </td><td>0.2533±0.0001</td><td>0.2506±0.0006</td></tr></table>

Table 6: Physical recovery and path inference. Drift error and path cost under new endpoints and horizons.

Optimization. We use the same four fixed linear and waiting initializations for every differentiable objective and run 500 L-BFGS iterations. Each method selects the candidate with lowest value under its own objective. We then evaluate the returned path using the ground-truth cost $Q \mathrm { { _ { t r u e } } }$ from Eq. (B.3). For LEAP, we differentiate through the learned drift during path optimization.

## C.3 LONG-HORIZON BARRIER SCALE

For

$$
S _ { D } ^ { T } ( \phi ) = \frac { 1 } { 4 D } \int _ { 0 } ^ { T } \left| \dot { \phi } ( t ) + U ^ { \prime } ( \phi ( t ) ) \right| ^ { 2 } d t ,
$$

with $\phi ( 0 ) = - 1$ and $\phi ( T ) = 0 \quad$ , the barrier $U ( 0 ) - U ( - 1 ) = 1 / 4$ sets the long-horizon cost. Using $| \dot { \phi } + U ^ { \prime } | ^ { 2 } = | \dot { \phi } - U ^ { \prime } | ^ { 2 } + 4 U ^ { \prime } \dot { \phi } .$

$$
D S _ { D } ^ { T } ( \phi ) = U ( \phi ( T ) ) - U ( \phi ( 0 ) ) + \frac { 1 } { 4 } \int _ { 0 } ^ { T } | \dot { \phi } - U ^ { \prime } ( \phi ) | ^ { 2 } d t ~ \geq U ( 0 ) - U ( - 1 ) = \frac { 1 } { 4 } .
$$

The reverse gradient path $\dot { \phi } = U ^ { \prime } ( \phi )$ approaches this bound as $T \to \infty ,$ so

$$
\operatorname * { l i m } _ { T  \infty } \operatorname * { i n f } _ { \phi ( 0 ) = - 1 \atop \phi ( T ) = 0 } D S _ { D } ^ { T } ( \phi ) = \frac { 1 } { 4 } .
$$

This limit explains why the optimal action approaches the barrier value for long horizons. At finite horizon T, we evaluate the optimized discrete path under the true action.

## D REFERENCE CHOICE AND ROBUSTNESS

We test sensitivity to the reference process on an unconditioned double well with an exact discrete path density. We use $b ( x ) = x - x ^ { \bar { 3 } } , D = 0 . 1 2$ and $X _ { 0 } \sim \mathcal { N } ( - 0 . 8 , 0 . 1 5 ^ { 2 } )$ . Each of the seeds uses 256 training, 96 validation and 128 test trajectories. Unless varied below, we use 24 transitions with $\Delta t = 0 . 0 4 \dot { , } T = 0 . 9 6$ and a zero-drift reference with $D _ { \mathrm { r e f } } = 0 . 2 0$

Transition NCE compares each observed next state with reference next states sampled from the same current state. Gaussian MLE maximizes the Gaussian one-step transition likelihood and learns a scalar diffusion coefficient. Both use the same drift network as LEAP. Because this setting is unconditioned and Markov, the transition factorization is exact and transition-level estimators are well specified, so we expect the three estimators to perform similarly. The point of this ablation is to study the sensitivity to the reference; the advantages of LEAP appear under terminal selection (§4.1) and persistent latent variables (§4.2).

Reference choice. Adding physical drift to the reference decreases LEAP correction NRMSE from 0.66 at $a _ { \mathrm { r e f } } = 0 \mathrm { t o } 0 . 2 2 3$ at $a _ { \mathrm { r e f } } = 0 . 5$ and 0.079 at $a _ { \mathrm { r e f } } = 1 ( \mathrm { T a b l e } 7 )$ .

Temporal and observation robustness. Across the three temporal grids, the three estimators reach similar correction error (Table 8). Increasing the hidden integration resolution from 2 to 8 substeps changes LEAP AUROC from 0.9171 to 0.9216. Observation noise decreases AUROC from 0.9156 on clean trajectories to 0.5948 at $\sigma _ { \mathrm { o b s } } = 0 . 1 0$

Data and reference budget. Increasing the training set from 32 to 256 trajectories decreases LEAP NRMSE from 0.87 to 0.66. Increasing K from 1 to 4 improves the most, with small changes after

<table><tr><td>Setting</td><td>LEAP</td><td>Transition NCE</td><td>Gaussian MLE</td></tr><tr><td colspan="4">Reference diffusion</td></tr><tr><td> $D _ { \mathrm { r e f } } = 0 . 0 6$ </td><td> $0 . 3 1 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $0 . 2 4 { \scriptstyle \pm 0 . 1 3 }$ </td><td> $0 . 3 4 7 { \scriptstyle \pm 0 . 0 1 4 }$ </td></tr><tr><td> $D _ { \mathrm { r e f } } = 0 . 1 2$ </td><td> $0 . 5 8 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $0 . 6 4 \pm \mathrm { 0 . 1 2 }$ </td><td> $0 . 6 4 4 { \scriptstyle \pm 0 . 0 8 4 }$ </td></tr><tr><td> $D _ { \mathrm { r e f } } = 0 . 2 0$ </td><td> $0 . 6 6 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $0 . 7 1 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $0 . 7 2 9 { \scriptstyle \pm 0 . 0 8 4 }$ </td></tr><tr><td> $D _ { \mathrm { r e f } } = 0 . 3 2$ </td><td> $0 . 7 9 5 { \scriptstyle \pm 0 . 0 8 9 }$ </td><td> $0 . 7 7 { \scriptstyle \pm 0 . 1 0 }$ </td><td> $0 . 8 5 5 { \scriptstyle \pm 0 . 0 6 2 }$ </td></tr><tr><td> $D _ { \mathrm { r e f } } = 0 . 4 8$ </td><td> $0 . 9 3 9 { \scriptstyle \pm 0 . 0 4 0 }$ </td><td> $0 . 8 3 7 { \scriptstyle \pm 0 . 0 8 0 }$ </td><td> $0 . 9 3 1 { \scriptstyle \pm 0 . 0 3 4 }$ </td></tr><tr><td colspan="4"> $R e f e r e n c e d r i f t$   $b _ { \mathrm { r e f } } = a _ { \mathrm { r e f } } b$ </td></tr><tr><td> $a _ { \mathrm { r e f } } = 0 . 2 5$ </td><td> $0 . 4 2 { \scriptstyle \pm 0 . 1 5 }$ </td><td> $0 . 4 5 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $0 . 4 8 6 { \scriptstyle \pm 0 . 0 6 6 }$ </td></tr><tr><td> $a _ { \mathrm { r e f } } = 0 . 5 0$ </td><td> $0 . 2 2 3 { \scriptstyle \pm 0 . 0 6 3 }$ </td><td> $0 . 2 6 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $0 . 3 5 6 { \scriptstyle \pm 0 . 0 2 8 }$ </td></tr><tr><td> $a _ { \mathrm { r e f } } = 0 . 7 5$ </td><td> $0 . 1 2 5 5 { \scriptstyle \pm 0 . 0 0 5 4 }$ </td><td> $0 . 1 2 1 { \scriptstyle \pm 0 . 0 2 9 }$ </td><td> $0 . 3 2 5 { \scriptstyle \pm 0 . 0 1 3 }$ </td></tr><tr><td> $a _ { \mathrm { r e f } } = 1 . 0 0$ </td><td> $0 . 0 7 9 { \scriptstyle \pm 0 . 0 4 4 }$ </td><td> $0 . 0 3 8 4 { \scriptstyle \pm 0 . 0 0 8 7 }$ </td><td> $0 . 3 1 7 0 { \scriptstyle \pm 0 . 0 0 6 2 }$ </td></tr></table>

Table 7: Reference choice. Correction NRMSE ↓. The diffusion sweep uses $b _ { \mathrm { r e f } } = 0 ;$ ; the drift sweep uses $D _ { \mathrm { r e f } } = 0 . 2 0$ . Contrastive models use $K = 4$
<table><tr><td>Setting</td><td>LEAP</td><td>Transition NCE</td><td>Gaussian MLE</td></tr><tr><td> $\Delta t = 0 . 0 2 , N = 4 8$ </td><td> $0 . 8 2 8 { \scriptstyle \pm 0 . 0 9 6 }$ </td><td> $0 . 7 6 9 { \scriptstyle \pm 0 . 0 8 1 }$ </td><td> $0 . 7 3 0 { \scriptstyle \pm 0 . 0 3 3 }$ </td></tr><tr><td> $\Delta t = 0 . 0 4 , N = 2 4$ </td><td> $0 . 6 6 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $0 . 7 1 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $0 . 7 2 9 { \scriptstyle \pm 0 . 0 8 4 }$ </td></tr><tr><td> $\Delta t = 0 . 0 8 , N = 1 2$ </td><td> $0 . 6 7 \pm \mathrm { 0 . 1 5 }$ </td><td> $0 . 6 7 1 { \scriptstyle \pm 0 . 0 7 0 }$ </td><td> $0 . 6 6 { \scriptstyle \pm 0 . 1 3 }$ </td></tr></table>

Table 8: Temporal resolution. Correction NRMSE ↓, mean ± sample SD at fixed $T = 0 . 9 6$

$K = 4$ . Under the default setting, selecting the path with the lowest predicted action yields regret of 0.0075 nats for LEAP, compared to 0.0230 nats for Gaussian MLE, measured using the true action.

## E ORBITAL EXPERIMENTS

We give the exact orbital correction and controls for path structure, likelihood parameterization and comparison objective, using the setting in § 4.2. The benchmark has 31 bodies, a 186-dimensional state and 32 transitions. We split initial contexts chronologically and evaluate on held-out contexts. During training, we resimulate data and reference trajectories from the 254 training initial states at every epoch, using 4 data trajectories per state (or 16 and 32 in Table 10). Therefore, the reported training trajectory counts are per epoch.

## E.1 EXACT TRAJECTORY ACTION

For each trajectory, we draw $z \in \{ - 1 , + 1 \}$ uniformly and set the solar mass to $m _ { \odot } ( 1 + z \eta )$ . The same unobserved z remains fixed for every transition; the reference uses the nominal mass $z = 0$

Let $\mu _ { z } ( x )$ be the Velocity-Verlet update under regime z and let Σ be the shared transition covariance. Conditional on the initial context,

$$
\begin{array} { r l } { \ell _ { z } ( \Phi ) = \displaystyle \sum _ { k = 0 } ^ { N - 1 } \log \mathcal { N } \left( x _ { k + 1 } ; \mu _ { z } ( x _ { k } ) , \Sigma \right) , \quad G ^ { * } ( \Phi ) } & { = \log \left[ \frac { 1 } { 2 } e ^ { \ell _ { + 1 } ( \Phi ) } + \frac { 1 } { 2 } e ^ { \ell _ { - 1 } ( \Phi ) } \right] - \ell _ { 0 } ( \Phi ) . } \end{array}\tag{E.1}
$$

Marginalizing the shared variable combines likelihoods over entire trajectories, producing the nonadditive correction in $\ S \mathrm { A . 4 }$

## E.2 PATH STRUCTURE AND ESTIMATOR CONTROLS

We compare model architectures and estimation objectives to study the importance of capturing dependencies across time. LEAP-Local sums corrections from individual transitions, while Transition NCE uses the same transition information without the structured LEAP parameterization. LEAP-Global learns a correction from the entire trajectory, using accumulated features derived from the nominal gravitational dynamics. These features use the sensitivity $J _ { k }$ of the nominal transition mean to the solar mass and the standardized innovations $e _ { k }$ . Defining $\dot { \gamma } _ { k } = e _ { k } ^ { \top } J _ { k }$ and $\begin{array} { r } { I = \sum _ { k } | J _ { k } | ^ { 2 } } \end{array}$ , the features are $\begin{array} { r } { Z = \sum _ { k } \gamma _ { k } / \sqrt { I } , Z ^ { 2 } , \log ( 1 + I ) } \end{array}$ and the mean of $\gamma _ { k } ^ { 2 } / | J _ { k } | ^ { 2 }$ . They do not contain the latent regime z or the exact correction $G ^ { * }$ , but assume that the perturbed parameter, the solar mass, is known.

<table><tr><td>Model</td><td>Input</td><td>NRMSE△G ↓</td></tr><tr><td>LEAP-Local</td><td>One transition</td><td>1.0013±0.0015</td></tr><tr><td>Transition NCE</td><td>One transition</td><td>1.0008±0.0013</td></tr><tr><td>LEAP-Global</td><td>Complete path + accumulated features</td><td>0.228±0.011</td></tr><tr><td>LEAP-Hybrid</td><td>Local + complete path</td><td> $0 . 2 3 2 { \scriptstyle \pm 0 . 0 1 4 }$ </td></tr><tr><td>Gaussian MLE</td><td>State + time</td><td> $1 1 . 9 2 { \scriptstyle \pm 0 . 6 6 }$ </td></tr><tr><td>History Gaussian MLE</td><td>Causal history</td><td> $1 0 . 1 6 2 { \scriptstyle \pm 0 . 0 9 3 }$ </td></tr><tr><td colspan="3">Objective control with complete path model</td></tr><tr><td>Binary NRE</td><td>Path encoder</td><td> $1 . 0 0 0 0 0 { \scriptstyle \pm 3 \times 1 0 ^ { - 5 } }$ </td></tr><tr><td>Binary NRE</td><td>Transformer + accumulated features</td><td>0.175±0.011</td></tr><tr><td>Categorical NRE (LEAP)</td><td>Transformer + accumulated features</td><td> $0 . 2 2 8 { \scriptstyle \pm 0 . 0 1 5 }$ </td></tr></table>

Table 9: Orbital controls. Correction NRMSE ↓ on the 31-body benchmark; a constant prediction gives 1. Bold is the lowest mean within each control block.

<table><tr><td>Setting</td><td>Training</td><td> $\mathrm { N R M S E } _ { \Delta G } \downarrow$ </td></tr><tr><td> $K = 4$ </td><td>validation stop</td><td> $0 . 2 2 0 8 { \scriptstyle \pm 0 . 0 0 5 6 }$ </td></tr><tr><td> $K = 4$ </td><td>275 epochs</td><td> $0 . 2 2 0 7 { \scriptstyle \pm 0 . 0 1 9 }$ </td></tr><tr><td> $K = 1 6$ </td><td>validation stop</td><td> $0 . 1 4 7 { \scriptstyle \pm 0 . 0 1 7 }$ </td></tr><tr><td> $K = 3 2$ </td><td>validation stop</td><td> $0 . 1 3 8 { \scriptstyle \pm 0 . 0 3 2 }$ </td></tr><tr><td>Training data</td><td></td><td></td></tr><tr><td>Training paths</td><td> $\mathrm { L E A P  – G l o b a l }$ </td><td>Binary NRE</td></tr><tr><td>4,064</td><td> $0 . 1 4 9 { \scriptstyle \pm 0 . 0 6 9 }$ </td><td> $0 . 1 2 9 { \scriptstyle \pm 0 . 0 1 7 }$ </td></tr><tr><td>8,128</td><td> $0 . 1 1 0 { \scriptstyle \pm 0 . 0 3 3 }$ </td><td> $0 . 1 1 9 { \scriptstyle \pm 0 . 0 1 5 }$ </td></tr></table>

Table 10: Comparison and data budgets. Correction NRMSE ↓. Increasing K changes the number of reference paths per observed trajectory; the lower panel increases the number of training paths. Separate runs from Table 9.

LEAP-Hybrid combines local and global corrections. We also compare against Gaussian MLE, which predicts a Gaussian distribution for the next state given the current state and time, and History Gaussian MLE, which additionally conditions on previous states using a causal GRU. Finally, we compare binary and categorical neural ratio estimation objectives to assess the effect of the training objective independently of the model architecture. Results are reported in Table 9.

Path structure. Transition-level models remain at NRMSE 1, while LEAP-Global reaches 0.228 (Table 9). The Gaussian likelihood models also perform poorly, as the exact correction in Eq. (E.1) involves a mixture over a latent regime shared across the trajectory. These results therefore show the importance of modeling dependencies across time.

Objective control. We also assess whether action recovery depends on the choice of contrastive objective. Using the same Transformer and accumulated features, binary and categorical ratio estimation reach NRMSE 0.175 and 0.228, respectively, showing that both objectives can recover the underlying action correction. On the other hand, binary ratio estimation with a generic path encoder stays near NRMSE 1. This shows that accurate action estimation can be achieved with different ratio estimation objectives, while incorporating relevant physical structure substantially improves recovery.

## E.3 COMPARISON AND DATA BUDGETS

Increasing K from 4 to 16 decreases LEAP-Global NRMSE from 0.2208 to 0.147, while extending the K = 4 model to 275 epochs leaves it at 0.2207 (Table 10). Increasing the training set from 4,064 to 8,128 paths further decreases LEAP-Global NRMSE from 0.149 to 0.110.

<table><tr><td>Transitions</td><td>Path coords.</td><td>LEAP-Local</td><td>LEAP-Global</td></tr><tr><td>31 bodies</td><td></td><td></td><td></td></tr><tr><td>32</td><td>6,138</td><td> $0 . 9 9 9 4 { \scriptstyle \pm 1 . 2 \times 1 0 ^ { - 3 } }$ </td><td> $0 . 2 5 8 4 { \scriptstyle \pm 0 . 0 0 9 7 }$ </td></tr><tr><td>64</td><td>12,090</td><td> $0 . 9 9 8 5 { \scriptstyle \pm 7 \times 1 0 ^ { - 4 } }$ </td><td> $0 . 1 2 7 { \scriptstyle \pm 0 . 0 1 5 }$ </td></tr><tr><td>128</td><td>23,994</td><td> $1 . 0 0 5 2 { \scriptstyle \pm 0 . 0 0 6 6 }$ </td><td> $0 . 1 1 3 { \scriptstyle \pm 0 . 0 2 7 }$ </td></tr><tr><td>256</td><td>47,802</td><td> $0 . 9 9 6 5 { \scriptstyle \pm 7 \times 1 0 ^ { - 4 } }$ </td><td> $0 . 1 1 7 { \scriptstyle \pm 0 . 0 5 5 }$ </td></tr><tr><td>42 bodies</td><td></td><td></td><td></td></tr><tr><td>32</td><td>8,316</td><td> $1 . 0 0 0 1 { \scriptstyle \pm 7 \times 1 0 ^ { - 4 } }$ </td><td> $0 . 2 5 1 { \scriptstyle \pm 0 . 0 1 6 }$ </td></tr><tr><td>64</td><td>16,380</td><td> $1 . 0 0 1 1 { \scriptstyle \pm 1 . 6 \times 1 0 ^ { - 3 } }$ </td><td> $0 . 1 4 4 { \scriptstyle \pm 0 . 0 2 4 }$ </td></tr><tr><td>128</td><td>32,508</td><td> $0 . 9 9 9 7 { \scriptstyle \pm 1 . 4 \times 1 0 ^ { - 3 } }$ </td><td> $0 . 1 6 3 { \scriptstyle \pm 0 . 0 5 3 }$ </td></tr><tr><td>256</td><td>64,764</td><td> $1 . 0 0 1 5 { \scriptstyle \pm 1 . 7 \times 1 0 ^ { - 3 } }$ </td><td> $0 . 1 1 1 { \scriptstyle \pm 0 . 0 1 0 }$ </td></tr></table>

Table 11: Scaling with system size and path length. Correction NRMSE ↓. Each configuration uses its exact mixture correction and $\dot { K } = 4 .$ . Separate runs from Table 9.
<table><tr><td>Objective difference</td><td>Error ↓</td><td> $\mathbf { A U R O C } \uparrow$ </td><td>Spearman ↑</td></tr><tr><td>Denoising</td><td> $1 . 0 0 0 0 { \scriptstyle \pm 2 . 3 \times 1 0 ^ { - 6 } }$ </td><td> $0 . 4 9 9 9 { \scriptstyle \pm 2 \times 1 0 ^ { - 4 } }$ </td><td></td></tr><tr><td>Variational</td><td> $0 . 9 8 8 3 { \scriptstyle \pm 1 \times 1 0 ^ { - 4 } }$ </td><td> $0 . 4 9 9 9 { \scriptstyle \pm 2 \times 1 0 ^ { - 4 } }$ </td><td> $0 . 0 0 6 6 { \scriptstyle \pm 0 . 0 0 5 5 }$ </td></tr></table>

Table 12: Trajectory diffusion baseline. GeoTDM objective differences on the orbital benchmark with 31 bodies. The denoising objective is evaluated using correction NRMSE, while the variational objective is evaluated using affine relative error.

## E.4 LONGER PATHS AND LARGER SYSTEMS

We vary the number of bodies and transitions and evaluate each setting against its exact mixture correction. At 42 bodies and 256 transitions, each path contains 64,764 coordinates. LEAP-Global reaches NRMSE 0.111, while LEAP-Local remains at 1.0015 (Table 11). The global model therefore maintains low correction error as the state dimension and trajectory length increase.

## E.5 TRAJECTORY DIFFUSION BASELINE

We adapt GeoTDM (Han et al., 2024) to the orbital benchmark and train separate trajectory diffusion models on the data and reference paths. We use the difference between their denoising or variational objectives as an estimate of $G ^ { * }$ . In Table 12, both give chance AUROC (0.4999) and the variational estimate has Spearman correlation 0.0066 with $G ^ { \bar { * } }$ . Thus trajectories with larger exact corrections do not systematically receive larger GeoTDM estimates.

## F MOLECULAR INFERENCE AND ACTION REUSE

Below we give additional details for the setting in §4.3.

## F.1 MOLECULAR DATA AND MODELS

Data and reference. We use $1 3 0 / 1 7 / 2 2$ train/validation/test trajectories at 500 K and $4 6 3 / 7 3 / 7 7$ at 600 K, with simulation replicas assigned to disjoint splits. Vacuum Langevin dynamics use a 1 fs integration step and we save the backbone angles $( \phi , \psi )$ every 0.1 ps. Paths have at most 201 frames.

We estimate a binned angular drift $\widehat { b } ( x )$ and effective diffusion $D _ { \mathrm { e f f } }$ from the training replicas and define

$$
b _ { \mathrm { r e f } } ( x , t ) = \widehat { b } ( x ) + 1 5 \frac { \mathrm { w r a p } ( x _ { B } - x ) } { T - t + 2 \Delta t } .
$$

We condition reference paths on the same reactive event as the data and evaluate wrapped Gaussian transition densities by summing periodic images.

Models and controls. We embed angles periodically and parameterize LEAP with a drift network of three width 128 hidden layers and a scalar residual with two width 64 hidden layers. LEAP uses

<table><tr><td>Method</td><td>Sparse completion Endpoints only</td><td></td></tr><tr><td>Reference action</td><td> $0 . 1 0 3 1 { \scriptstyle \pm 0 . 0 0 1 6 }$ </td><td> $0 . 8 1 8 3 { \scriptstyle \pm 0 . 0 0 4 2 }$ </td></tr><tr><td>Untrained action</td><td> $0 . 0 9 4 7 { \scriptstyle \pm 0 . 0 0 0 0 }$ </td><td> $0 . 7 9 5 3 { \scriptstyle \pm 0 . 0 0 2 7 }$ </td></tr><tr><td>Velocity regression</td><td> $0 . 0 9 0 0 { \scriptstyle \pm 0 . 0 0 0 3 }$ </td><td> $0 . 6 6 0 9 { \scriptstyle \pm 0 . 0 0 3 2 }$ </td></tr><tr><td>Binary NRE</td><td> $0 . 0 8 9 7 { \scriptstyle \pm 0 . 0 0 0 2 }$ </td><td> $0 . 6 6 8 2 { \scriptstyle \pm 0 . 0 1 1 8 }$ </td></tr><tr><td>Conditional DDPM</td><td> $0 . 1 1 9 { \scriptstyle \pm 0 . 0 1 2 }$ </td><td> $0 . 4 0 9 { \scriptstyle \pm 0 . 0 3 4 }$ </td></tr><tr><td>Conditional flow matching</td><td> $0 . 1 2 5 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $0 . 3 5 0 { \scriptstyle \pm 0 . 0 1 2 }$ </td></tr><tr><td>LEAP</td><td> $0 . 0 9 3 1 { \scriptstyle \pm 0 . 0 0 1 6 }$ </td><td> $0 . 6 2 2 5 { \scriptstyle \pm 0 . 0 0 8 0 }$ </td></tr></table>

Table 13: Molecular completion at 600 K. Circular MSE ↓ on unobserved frames for the same observation patterns and durations.

<table><tr><td>Train</td><td>Test</td><td>LEAP</td><td>Binary NRE</td><td>Velocity</td><td>Untrained</td></tr><tr><td>500</td><td>500</td><td> $0 . 9 4 0 3 { \scriptstyle \pm 0 . 0 0 9 9 }$ </td><td> $0 . 9 3 3 7 { \scriptstyle \pm 0 . 0 0 3 0 }$ </td><td> $0 . 9 4 2 7 { \scriptstyle \pm 0 . 0 0 3 3 }$ </td><td> $0 . 8 3 8 5 { \scriptstyle \pm 0 . 0 0 5 0 }$ </td></tr><tr><td>600</td><td>500</td><td> $\mathbf { 0 . 9 5 7 9 { \scriptstyle \pm 0 . 0 0 5 7 } }$ </td><td> $0 . 9 4 8 9 { \scriptstyle \pm 0 . 0 0 5 1 }$ </td><td> $0 . 9 4 3 7 { \scriptstyle \pm 0 . 0 0 2 2 }$ </td><td> $0 . 8 3 8 5 { \scriptstyle \pm 0 . 0 0 5 0 }$ </td></tr><tr><td>500</td><td>600</td><td> $\mathbf { 0 . 9 7 1 9 { \scriptstyle \pm 0 . 0 0 6 8 } }$ </td><td> $0 . 9 3 8 6 { \scriptstyle \pm 0 . 0 0 2 6 }$ </td><td> $0 . 9 5 3 3 { \scriptstyle \pm 0 . 0 0 2 5 }$ </td><td> $0 . 8 8 4 7 { \scriptstyle \pm 0 . 0 0 2 8 }$ </td></tr><tr><td>600</td><td>600</td><td> $0 . 9 7 5 4 { \scriptstyle \pm 0 . 0 0 1 2 }$ </td><td> $0 . 9 6 9 2 { \scriptstyle \pm 0 . 0 0 0 7 }$ </td><td> $0 . 9 5 5 9 { \scriptstyle \pm 0 . 0 0 2 9 }$ </td><td> $0 . 8 8 4 7 { \scriptstyle \pm 0 . 0 0 2 8 }$ </td></tr></table>

Table 14: Temperature transfer. Within-context AUROC ↑. Bold marks LEAP evaluated at a temperature different from its training temperature.

K = 8 reference paths drawn from 32 successful references per initial state. Binary NRE uses the same inputs and network capacity, while velocity regression fits wrapped one-step increments with the same drift network.

For conditional generation, we compare with DDPM (Ho et al., 2020) and flow matching (Lipman et al., 2023) using the same molecular coordinates, observation masks and active durations. Both use a temporal convolutional network with width 96 and seven residual layers. DDPM uses a 100-step cosine schedule and flow matching uses 100 Euler steps at inference.

## F.2 COMPLETION CONTROLS

We evaluate two conditioning settings. In the first, we observe the endpoints together with three interior frames; in the second, we observe only the endpoints and trajectory duration. For methods that optimize a learned action, we use eight shared initializations, run 400 steps of projected Adam and select the path with the lowest action. Conditional generators produce eight candidate trajectories, which are selected using the reference action. In both settings, candidate selection does not use the held out target frames.

With endpoints and three interior observations, LEAP achieves circular MSE 0.0931, compared with 0.119 for conditional DDPM and 0.125 for conditional flow matching (Table 13). With endpoints only, LEAP gives the lowest error among action-based methods (0.6225, compared with 0.6609 for velocity regression and 0.6682 for binary NRE), while the conditional generators achieve lower reconstruction error. The same LEAP checkpoint is used for both observation patterns.

## F.3 TEMPERATURE TRANSFER

We evaluate each learned correction at both temperatures without retraining. At the target temperature, we use the corresponding reference process and effective diffusion.

Training at 600 K gives AUROC 0.9579 at 500 K, while training at 500 K gives 0.9719 at 600 K (Table 14). Thus the learned correction transfers across temperatures when combined with the target reference process.

<table><tr><td></td><td> $\lambda = 0$ </td><td> $\lambda = 5$ </td><td> $\lambda = 2 0$ </td><td> $\lambda = 5 0$ </td></tr><tr><td>Preference  $M \uparrow$ </td><td> $0 . 0 6 5 8 { \scriptstyle \pm 0 . 0 0 5 9 }$ </td><td> $0 . 1 0 9 5 { \scriptstyle \pm 0 . 0 0 8 8 }$ </td><td> $0 . 2 5 1 7 { \scriptstyle \pm 0 . 0 0 4 5 }$ </td><td> $0 . 3 6 4 1 { \scriptstyle \pm 0 . 0 1 1 6 }$ </td></tr><tr><td>Transition cost↓</td><td> $0 . 9 3 8 9 { \scriptstyle \pm 0 . 0 5 4 5 }$ </td><td> $0 . 9 1 5 9 { \scriptstyle \pm 0 . 0 2 2 8 }$ </td><td> $0 . 9 8 0 9 { \scriptstyle \pm 0 . 0 2 3 1 }$ </td><td> $1 . 3 7 7 4 { \scriptstyle \pm 0 . 0 2 5 2 }$ </td></tr></table>

Table 15: Preference reuse with a fixed LEAP action. Increasing λ raises the basin preference while changing the unguided projected transition cost in nats per active step.

## F.4 PREFERENCE AND CONSTRAINT REUSE

We keep the learned 600 K action fixed and add a periodic preference centered at $\begin{array} { r l } { c _ { M } } & { { } = } \end{array}$ $( - 6 0 ^ { \circ } , \bar { - } 4 0 ^ { \circ } )$ ,

$$
M ( \Phi ) = \frac { 1 } { N - 1 } \sum _ { k = 1 } ^ { N - 1 } \exp \left[ \frac { 1 } { s _ { M } ^ { 2 } } \sum _ { j = 1 } ^ { 2 } ( \cos ( x _ { k , j } - c _ { M , j } ) - 1 ) \right] , \qquad s _ { M } = 3 0 ^ { \circ } ,
$$

with inference objective $J _ { \lambda } ( \Phi ) = S _ { \theta } ( \Phi ) - \lambda M ( \Phi )$ . We evaluate $\lambda \in \{ 0 , 5 , 2 0 , 5 0 \}$ on the same $7 7$ endpoint contexts.

Increasing λ from 0 to 20 raises the mean preference from 0.0658 to 0.2517 while the transition cost changes from 0.9389 to 0.9809 (Table 15). $\mathrm { { A t } } \lambda = 5 0$ , the preference increases further to 0.3641 with a larger transition cost of 1.3774.

Constraints and composed requests. We also keep the learned action fixed while imposing $| \phi _ { k } | \le 1 5 0 ^ { \circ }$ and $| \phi _ { k + 1 } - \phi _ { k } | \stackrel { - } { \le } \pi - 1 0 ^ { - 3 }$ through projection during inference. We combine these bounds with basin preferences, route constraints and additional observed states by changing only the inference objective or feasible set.

Molecular geometry verification. Reconstructed trajectories preserve molecular geometry at both temperatures, with torsion errors below $0 . 2 4 ^ { \circ }$ , relative bond length changes below 1.5%, and preserved $\mathrm { C } _ { \alpha }$ chirality and trans peptide geometry.