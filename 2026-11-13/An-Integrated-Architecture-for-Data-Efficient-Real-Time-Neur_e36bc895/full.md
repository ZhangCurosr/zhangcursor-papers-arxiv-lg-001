# An Integrated Architecture for Data-Efficient Real-Time Neural Imitation of Nonlinear Time-Optimal Model Predictive Control

Maarten De Baecke<sup>1</sup>, Flavia Sofia Acerbo<sup>1</sup>, Andrea Giusti<sup>2</sup>, and Wilm Decre´<sup>1</sup>

Abstract— Nonlinear time-optimal model predictive control (TOMPC) enables highly performant control of constrained systems, but its computational cost prohibits real-time use on fast dynamical systems. Neural network imitation of the TOMPC control law can reduce online computational costs, yet existing work mostly targets quadratic cost formulations in simulation and addresses the challenges of neural controllers in isolation. This work imitates a long-horizon nonconvex TOMPC with hard terminal constraints, whose high computational cost limits training-data generation. We present (i) self-informed sampling (SIFS), a data-efficient expert-sampling strategy that uses a learned time-to-go to query the expert only where the policy deviates from time-optimal behavior, (ii) a complete training and control architecture that jointly addresses dataset correlation, suboptimal expert labels, covariate shift, constraint satisfaction and model mismatch, and (iii) real-world validation on a Furuta pendulum. SIFS matches standard dataset aggregation performance in roughly one-third of the training time and with less than half the training data. The resulting controller matches the region of attraction of the expert TOMPC while raising the control frequency from 1.4 Hz to 1 kHz. On hardware, the controller swings up and stabilizes the pendulum along trajectories closely matching the expert TOMPC solution.

## I. INTRODUCTION

Model predictive control (MPC) has emerged as a powerful and widely applicable control strategy. In particular, nonlinear time-optimal MPC (TOMPC) seeks to steer nonlinear systems to a desired state in minimum time, thereby achieving highly performant control while accounting for system dynamics and constraints [1], [2]. Nevertheless, for nonlinear systems with fast dynamics, the high computational cost of online nonconvex optimization results in computation times often orders of magnitude higher than the available sampling interval [3]. A promising approach to overcome this limitation is to imitate the MPC control law with a neural network (NN), often through Behavioral Cloning (BC) [4], trained offline on MPC solutions [5]. Given a sufficiently small NN that accurately reproduces the MPC control law, MPC-like control can be achieved at the high control frequencies required by fast dynamical systems. Such a controller is hereafter referred to as a neural network model predictive controller (NNMPC).

Existing literature already demonstrates real-time approximation of nonlinear MPC, but almost exclusively for quadratic-tracking objectives, and predominantly in simulation [6], [7]. Hardware demonstrations remain scarce, with existing works limited to either simpler objectives, soft or no terminal constraints or comparatively simple platforms [8]–[10]. This work therefore tackles imitation of more computationally challenging MPC formulations, specifically a long-horizon, time-optimal formulation with hard terminal constraints. A key characteristic of such problems is that the computational cost of the MPC formulation not only restricts online deployment, but also strongly limits the generation of training data. Moreover, time-optimal control laws tend toward bang-bang behavior. Accurately approximating such laws with a smooth NN requires dense data near the switching points, increasing the demand for expert data. This makes conventional expert-data collection approaches such as Dataset Aggregation (DAgger) [11] particularly costly, motivating more efficient expert-sampling strategies. Finally, existing works address individual challenges of NNMPC, such as dataset correlation [12], suboptimal expert labels [13], controller adaptability [10], constraint satisfaction [6] and covariate shift [11], but a complete architecture integrating these solutions is currently lacking, leaving the design of a reliable NNMPC system fragmented across the literature. Therefore, this work aims at (i) presenting an expertsampling algorithm, referred to as self-informed sampling (SIFS), as a data-efficient alternative to DAgger specifically designed for time-optimal MPC, (ii) providing a complete training and control architecture that jointly addresses dataset correlation, suboptimal expert labels, controller adaptability, constraint satisfaction, covariate shift and efficient expert sampling, and (iii) providing real-world validation by deploying the resulting controller on a physical Furuta pendulum.

## II. PRIOR WORK

Many approaches have been proposed to overcome the computational limitations of MPC. One approach shifts the computational burden offline by storing precomputed optimal control solutions in a lookup table and interpolating them online, known as explicit MPC [14]. Neural networks offer a scalable alternative by learning a continuous mapping from states to control inputs. While NNMPC appears promising, several challenges remain.

Dataset Correlation. As argued in [12], MPC datasets typically violate the i.i.d. assumption underlying standard imitation learning [11], due to strong correlations both within and across trajectories. To mitigate this, [12] proposes an Optimally-Sampled Dataset algorithm, subsampling the existing dataset so as to minimize biases that hinder generalization.

Suboptimality and Modality. Nonconvex MPC can yield locally optimal or multimodal demonstrations. Consequently, different solutions may assign different actions to the same state, resulting in conflicting supervision. Under a mean squared error objective, these actions are effectively averaged, yielding a policy that follows neither solution and may therefore exhibit suboptimal behavior.

Works such as [15] address this by filtering out suboptimal samples, yet works such as [16] argue that such samples can still provide valuable state-space coverage and should not be discarded outright. An alternative is weighted BC, which assigns each sample a data-dependent weight based on its estimated importance [17]. Such weights can steer the learned solution towards one mode (preferably the most optimal one), while simultaneously retaining state-space coverage.

Nevertheless, to compute these weights, existing approaches require substantial additional training complexity, such as an auxiliary discriminator network [17] or bilevel optimization over ranked reference trajectories [13].

Data Efficiency. Although imitating an MPC policy via BC is straightforward, covariate shift [11] should be taken into account. Several strategies have been proposed to mitigate this issue, including augmenting the training data with perturbed states through noise injection [18], and DAgger [11]. The latter iteratively executes the learned policy, queries the expert for actions at the visited states, and aggregates these newly labeled samples into the training set for subsequent retraining.

Nevertheless, all such methods are sample-inefficient as they do not distinguish between states that require additional supervision and those already handled well by the policy [19]. Since NN inference is computationally inexpensive compared to querying the expert, the learned policy can be used to identify states where its performance remains suboptimal. Expert demonstrations are then generated only for these states, focusing data collection on regions where additional supervision is most beneficial. This approach mitigates covariate shift, while simultaneously improving sample efficiency. Previous work in data-efficient sampling employs epistemic uncertainty [19] using ensembles [20] or Monte Carlo (MC) dropout [21]. Similar to DAgger, these methods iteratively expand the training dataset, but query expert labels only for visited states with high epistemic uncertainty. A related approach, STRIDE (Sampling Trajectories for the Robust Imitation and Dynamic Evaluation) [22], avoids uncertainty estimation and instead samples states based on the distance between closed-loop NNMPC and MPC trajectories, combined with grid-based validation over the task space. Despite their different state-selection criteria, we argue that both epistemic uncertainty and STRIDE share three fundamental limitations.

First, their selection metrics are indirect proxies for control quality: epistemic uncertainty can be overconfident [20], while trajectory distance assumes that states far from expert trajectories are poorly controlled and nearby states are controlled well. Thus, queries may miss failure-prone states and waste samples on well-controlled ones. Second, neither metric accounts for the sensitivity of the system to control errors, treating equivalent errors identically despite potentially different consequences. Third, neither explicitly considers how control errors propagate through the system dynamics: an error that is readily corrected at the next timestep is treated the same as one that causes instability. Moreover, STRIDE requires evaluation against a grid whose size grows combinatorially with task-space dimension, limiting its scalability to higher-dimensional problems.

Constraint Satisfaction. Unlike classical MPC, naive neural imitation does not guarantee strict satisfaction of the originally imposed state and input constraints [23]. Previous work proposes to enhance the neural training loss function with an additional penalty to include constraint violation into the learning gradient. This can be done via the penalty method (quadratically penalizing constraint violations), or logarithmic barrier methods (a loss term whose gradient grows unboundedly as the solution approaches the constraint boundaries) [6], [24]. Notably, works employing such methods simultaneously train the network to learn the dual variables. Finally, Control Barrier Functions (CBFs) have been shown to be effective in guaranteeing constraint satisfaction when applied to the neural network output [25].

Adaptability. Unlike classical MPC, whose behavior can be adjusted online by retuning cost or model parameters, a trained NN policy offers no direct mechanism for tuning. Adapting expert parameters requires regenerating the dataset and retraining from scratch. Recent work [10] mitigates this by learning an auxiliary NN alongside the nominal policy that predicts how the control output should change under parameter perturbations, enabling adaptation without retraining. Nevertheless, this requires substantial additional training effort, as accurate sensitivity estimates are crucial for performance but significantly increase the computational burden [10].

## III. METHODOLOGY

This work expands on prior work by providing a complete training and control architecture that jointly addresses the challenges of NNMPC, discussed in the previous section. A schematic representation of this architecture is provided in Fig. 1. First, an expert MPC I is used to generate new data points, which are iteratively aggregated to the dataset II . The resulting dataset is then filtered using an optimal sampling algorithm III to remove correlated and redundant samples. Each remaining data point is assigned a weight IV to reduce the influence of conflicting expert labels, and the NN is subsequently trained V on the weighted dataset using BC. The current policy is evaluated through closed-loop rollouts VI , with the proposed SIFS algorithm VII identifying poorly performing states. The expert MPC is queried only on these states. The resulting expert demonstrations are added to the dataset, completing the data-aggregation loop. The NNMPC output is adjusted by an ancillary controller VIII for plant–model mismatch compensation and a safety filter IX for constraint satisfaction. The following sections detail each component.

![](images/fb9d108994a6739c46708ca5c1a796f7215654d144e020796fc21da57c60f5d1.jpg)  
Fig. 1. Schematic outline of the proposed integrated architecture. Roman numerals refer to the blocks as introduced in the text.

## A. Expert Model Predictive Controller

The expert MPC considered here involves a long-horizon, nonconvex, time-optimal formulation with a hard terminal constraint, where time-optimality is realized using an exponentially weighted objective following [2]. Unlike quadratictracking MPC, it exploits the full actuation range to reach the target in minimum time without requiring a reference trajectory. Nevertheless, the considered time-optimal formulation is sensitive to initialization and computationally demanding.

For TOMPC, the time-to-go (T2G) naturally serves as a value function, defined as the first time the predicted MPC trajectory from an initial state $\mathbf { x } _ { 0 }$ lies within a tolerance ξ of the reference $\mathbf { x } _ { f } \colon \mathrm { T 2 G } ( \mathbf { x } _ { 0 } ) = \Delta t \cdot \operatorname* { m i n } \{ k \colon \| \mathbf { x } _ { k } - \mathbf { x } _ { f } \| \leq \xi \}$ where ∆t is the sampling time and $\mathbf { X } _ { k }$ is the predicted state at step k of the MPC solution.

## B. Correlation between expert labels

Since expert data generated by an MPC are strongly correlated, raw MPC datasets often oversample regions containing little variation in the optimal control law [12]. This adds redundancy to the dataset and simultaneously skews the cost minimization, making the overall trained policy worse. Therefore, our proposed framework builds upon the approach of [12], in which the NN is trained on an Optimally-Sampled

Dataset, preserving the information content of the MPC policy at a prescribed resolution while eliminating redundant data points.

## C. Conflicting expert labels

Since the nonconvex optimal control problem may converge to different local optima, nearby states can receive conflicting expert labels. To mitigate this, this work employs weighted BC, multiplying each quadratic error term in the training loss by a weight, thereby giving higher-quality samples greater influence during training.

To efficiently compute weights, we propose a lightweight T2G-based weighting strategy, leveraging T2G as a direct measure of trajectory optimality under a time-optimal objective. Concretely, we assign each training sample a weight $w _ { \mathbf { k } \mathrm { n n } } ( \mathbf { x } _ { i } )$ based on its T2G relative to the distance-weighted average T2G of its M nearest neighbors, excluding samples from the same trajectory:

$$
\alpha _ { i j } = e ^ { - \frac { \| \mathbf { x } _ { i } - \mathbf { x } _ { j } \| ^ { 2 } } { \sigma ^ { 2 } } } , \quad \overline { { \mathrm { T 2 G } } } ( \mathbf { x } _ { i } ) = \frac { \sum _ { j \in \mathcal { N } _ { M } ( i ) } \alpha _ { i j } \mathrm { T 2 G } ( \mathbf { x } _ { j } ) } { \sum _ { j \in \mathcal { N } _ { M } ( i ) } \alpha _ { i j } } ,\tag{1}
$$

$$
w _ { \mathrm { k n n } } ( \mathbf { x } _ { i } ) = \frac { 1 } { 1 + e ^ { p \left( \mathrm { T 2 G } ( \mathbf { x } _ { i } ) - \overline { { \mathrm { T 2 G } } } ( \mathbf { x } _ { i } ) \right) } } \cdot q + ( 1 - q ) .\tag{2}
$$

Here, $\alpha _ { i j }$ is a distance-based Gaussian weight, $\mathcal { N } _ { M } ( i )$ denotes the M nearest neighbors of $\mathbf { X } _ { i } ,$ excluding points from its own trajectory; σ controls the influence of distance in state space, while $p \in \mathbb { R } ^ { + }$ and $q \in [ 0 , 1 ]$ control the sigmoid sharpness and lower bound, respectively. By incorporating distance-based weighting in the computation of T2G, trajectory points in sparsely sampled regions retain higher influence, even when they appear suboptimal relative to their neighbors. This balances state-space coverage with the suppression of suboptimal expert data, although conflicting labels from equally time-optimal modes remain unresolved.

## D. Self-Informed Sampling

We introduce self-informed sampling as a data-efficient sampling strategy that addresses the limitations of existing approaches discussed in Section II. Concretely, we augment the NN with an additional output head that predicts a value function, more specifically the T2G. The T2G decreases by one sampling interval per time step along a time-optimal trajectory, providing a direct indicator of deviations from optimal behavior without requiring the expert controller.

Since the T2G is learned, its predictions contain noise and occasional harmless discontinuities. The informativestate criterion should therefore tolerate short-term deviations and isolated jumps as long as the overall decreasing trend remains consistent. Conversely, persistent deviations, such as stagnation, gradual increases, or strongly irregular behavior, should be detected and flagged from their onset. To this end, we design a simple algorithm to identify segments that follow the expected linear T2G evolution and flag transitions between them (see Fig. 2). Since the exact state causing the deviation may be missed, states preceding a flagged state are also considered as candidates. Candidates are then sampled for expert querying with a probability that increases with (i) their proximity to a flag, (ii) how early they occur in the rollout, as early errors propagate through the remainder of the trajectory, and (iii) the severity of the corresponding rollout, which quantifies how much the predicted T2G deviates from the expected decrease of one sampling interval per time step.

![](images/f090cdf25c535dc8630db37c7360a2c455138cd43b25b87a6fc558ca4a4ff989.jpg)  
Fig. 2. Predicted time-to-go (T2G) along a rollout for the expert MPC (red) and a partially trained NNMPC (yellow). While the NNMPC prediction is noisy, it closely follows the expected trend except for the highlighted transition (dashed line), where it jumps to a different trajectory. This deviation indicates suboptimal behavior and identifies the states most informative for expert querying.

Unlike epistemic-uncertainty or trajectory-distance criteria, which judge the policy at a single state through an indirect proxy, the proposed criterion evaluates the T2G over an entire rollout. Because the network receives only the current state as input, with no access to its previous T2G estimate, it cannot produce a consistent T2G sequence by simply decrementing an earlier output. A T2G prediction that decreases by exactly one sampling interval per step is therefore unlikely to occur by chance, and this likelihood decreases rapidly with the rollout length. Such a trend is unlikely to arise unless the network has (i) learned the T2G of the visited states and (ii) predicts controls that steer the system to the correct next state. This avoids overconfidence in poorly handled states, as well as unnecessary queries at states that are already well learned. Furthermore, locally suboptimal actions with negligible effect on the trajectory leave the trend intact and are not flagged, whereas errors that propagate through the dynamics break it.

## E. Controller Adaptability

To compensate for plant–model mismatch without retraining on real-world data, the NNMPC is augmented with a lightweight ancillary controller that corrects its control action online. Two options are considered: a dynamic-inversion compensator (DIC) and a Linear Quadratic Regulator (LQR).

1) Dynamic-Inversion Compensator: Following the inverse-dynamics transfer approach of [26], the DIC compensates for plant–model mismatch by computing an input ¯u that reproduces, on an estimated plant model, the output dynamics induced by the nominal NNMPC input $u _ { \mathrm { n o m } }$ on the nominal model. Unlike learned inverse models, the DIC analytically inverts the plant dynamics, similar to input–output feedback linearization [27].

Consider a single-input single-output (SISO) control-affine system $\dot { \mathbf { x } } = f ( \mathbf { x } ; \pmb { \theta } ) + g ( \mathbf { x } ; \pmb { \theta } ) \boldsymbol { u }$ with parameters θ and output $y = h ( \mathbf { x } )$ . Differentiating the output r (relative degree) times until the input explicitly appears yields

$$
y ^ { ( r ) } = a ( { \bf x } ; \theta ) + b ( { \bf x } ; \theta ) u , \qquad a = L _ { f } ^ { r } h , \ b = L _ { g } L _ { f } ^ { r - 1 } h ,\tag{3}
$$

with $L _ { f }$ and $L _ { g }$ the Lie derivatives along f and $g .$

At each control step, the nominal model (with model parameters $\theta _ { \mathrm { n } } )$ is first evaluated using $u _ { \mathrm { n o m } }$ to determine the desired output derivative: $\nu = a ( \mathbf { x } ; \theta _ { \mathrm { n } } ) + b ( \mathbf { x } ; \theta _ { \mathrm { n } } ) u _ { \mathrm { n o m } }$ . The DIC then analytically inverts the estimated plant dynamics (with $\hat { \boldsymbol { \theta } }$ the estimate of the true plant parameters) to find the input ¯u that reproduces this derivative on the real plant:

$$
\bar { u } = \frac { \nu - a ( { \bf x } ; \hat { \theta } ) } { b ( { \bf x } ; \hat { \theta } ) } , \qquad b ( { \bf x } ; \hat { \theta } ) \ne 0 .\tag{4}
$$

For multiple-input, multiple-output (MIMO) systems, exact matching of only some output dynamics may conflict with the overall NNMPC objective (also regulating other output dynamics). Therefore, the applied input is obtained by blending the nominal and dynamically compensated inputs: $u = u _ { \mathrm { n o m } } + \zeta \left( \bar { u } - u _ { \mathrm { n o m } } \right)$ , with $\zeta \in [ 0 , 1 ]$

The DIC thus requires a control-affine model with a well-defined relative degree and a nonzero input coefficient b throughout the operating region, as well as an estimate $\hat { \boldsymbol { \theta } }$ of the plant parameters.

2) Linear Quadratic Regulator: A more generic alternative employs an ancillary LQR, inspired by its use in the tube-based MPC framework of [28]. At each time step, the state predicted by the nominal model given the previous state and input, x¯, is compared with the measured next state x of the real plant. The LQR then directly converts this observed mismatch into a corrective input:

$$
u = u _ { \mathrm { n o m } } + \mathbf { K } ( \bar { \mathbf { x } } - \mathbf { x } ) ,\tag{5}
$$

where K is the LQR gain, computed using the linearized nominal model.

## F. Constraint Satisfaction

We address constraint satisfaction at two levels: additional terms in the training loss encourage the neural network to learn the constraints imposed by the original MPC formulation, while a differentiable safety filter in the final network layer formally guarantees constraint satisfaction.

To steer the raw network output towards the feasible region, the training loss combines a logarithmic barrier with a quadratic penalty:

$$
\begin{array} { r } { L _ { \mathrm { b a r r i e r } } = - \operatorname* { m e a n } \Big [ \log ( u _ { \mathrm { r a w } } - u _ { \mathrm { m i n , s a f e } } + \varepsilon ) \qquad } \\ { + \log ( u _ { \mathrm { m a x , s a f e } } - u _ { \mathrm { r a w } } + \varepsilon ) \Big ] , } \end{array}\tag{6}
$$

$$
\begin{array} { r } { L _ { \mathrm { p e n a l t y } } = \mathrm { m e a n } \left[ \mathrm { m a x } ( 0 , u _ { \mathrm { r a w } } - u _ { \mathrm { m a x , s a f e } } ) ^ { 2 } \right. \qquad } \\ { \left. + \mathrm { m a x } ( 0 , u _ { \mathrm { m i n , s a f e } } - u _ { \mathrm { r a w } } ) ^ { 2 } \right] . } \end{array}\tag{7}
$$

Here, $u _ { \mathrm { r a w } }$ denotes the neural network output before the safety filter, $u _ { \mathrm { m i n / m a x , s a f e } }$ denote the corresponding safe bounds imposed by the safety filter and ε ensures numerical stability. The barrier term is evaluated only within the safe region; outside it, its contribution is zero.

The network is not trained to jointly predict the dual variables [6], [24], as this would complicate data generation. Without a dual-variable term, the logarithmic barrier (6) alone proved insufficient in our experiments: it provides no gradient when the raw output violates the safe bounds, leading to unstable training. The quadratic penalty (7) complements the barrier by providing a gradient in the infeasible region, thereby mitigating this instability.

The safety filter projects the raw network output onto the set of inputs that keep the system within its constraints at all future times: $\begin{array} { r } { u _ { \mathrm { s a f e } } = \arg \operatorname* { m i n } _ { u \in \mathcal { U } _ { \mathrm { s } } ( \mathbf { x } ) } \| u - u _ { \mathrm { r a w } } \| ^ { 2 } } \end{array}$ , with $\mathcal { U } _ { \mathrm { s } } ( \mathbf { x } ) = \{ u \in \mathcal { U } : f _ { d } ( \mathbf { x } , u ) \in \mathcal { C } \}$ , C a control-invariant (safe) set and $f _ { d } ( { \bf x } , u )$ the discrete system dynamics. Ideally, $\mathcal { C }$ is the maximal control-invariant set, making $\mathcal { U } _ { \mathrm { s } } ( \mathbf { x } )$ maximally permissive and the filter nonconservative, without additional constraints on the MPC [29]. If this set is intractable, CBFs [25] provide a conservative alternative. The same CBF constraint should then be included in the expert MPC, preventing the NNMPC from learning infeasible control actions.

## IV. EXPERIMENTAL SETUP

## A. Furuta Pendulum and Expert Controller

To demonstrate the capability of the proposed framework, we consider the imitation of a TOMPC for control of a Furuta pendulum. The state vector and control input are defined as $\mathbf { x } = ( \theta _ { 1 } , \theta _ { 2 } , \dot { \theta } _ { 1 } , \dot { \theta } _ { 2 } )$ and $u = \ddot { \theta } _ { 1 }$ , with $\theta _ { 1 }$ and $\theta _ { 2 }$ denoting the base and pendulum-arm angles, respectively, and $\theta _ { 2 } = \pi$ rad corresponding to the inverted position. The continuoustime nonlinear dynamics $\dot { \mathbf { x } } = f ( \mathbf { x } ) + g ( \mathbf { x } ) \cdot u$ follow [30]. The TOMPC formulation itself is:

$$
\begin{array} { r l } { \displaystyle \operatorname* { m i n } _ { { \bf x } , { \bf u } , s } } & { ~ \displaystyle \sum _ { k = 0 } ^ { N - 1 } \left[ \left( { \boldsymbol \gamma } ^ { { \bf x } , k } \right) \cdot \| { \bf x } _ { k } - { \bf x } _ { f } \| _ { 2 } ^ { 2 } + w _ { u } \cdot u _ { k } ^ { 2 } + w _ { s } \cdot s ^ { 2 } \right] } \\ { \mathrm { s u b j e c t ~ t o : } } & { { \bf x } _ { k + 1 } = f _ { d } ( { \bf x } _ { k } , u _ { k } ) } \\ & { { \bf x } _ { 0 } = { \bf x } _ { \mathrm { i n i t } } , \quad { \bf x } _ { N } = { \bf x } _ { f } } \\ & { \theta _ { 1 , \mathrm { m i n } } \le \theta _ { 1 , k } \le \theta _ { 1 , \mathrm { m a x } } , \qquad \forall k \in \{ 0 , \ldots , N \} } \\ & { u _ { \mathrm { m i n } } \le u _ { k } \le u _ { \mathrm { m a x } } , \quad \quad \forall k \in \{ 0 , \ldots , N - 2 \} } \\ & { - s \le u _ { N - 1 } \le s , \quad s \ge 0 . } \end{array}\tag{8}
$$

The MPC predicts over a finite horizon of $N = 6 0$ steps with sampling time $\Delta t = 0 . 0 3 \mathrm { ~ s ~ }$ . The state $\mathbf { x } _ { k } \in \mathbb { R } ^ { 4 }$ and control input $u _ { k } \in \mathbb { R }$ are optimized toward the target state $\mathbf { x } _ { f } \in \mathbb { R } ^ { 4 }$ , subject to the input bounds $u _ { \mathrm { m i n } } \le u _ { k } \le u _ { \mathrm { m a x } }$ and arm-angle bounds $\theta _ { 1 , \mathrm { { m i n } } } \leq \theta _ { 1 , k } \leq \theta _ { 1 , \mathrm { { m a x } } }$ . The objective uses $w _ { u } = 1 \ \mathrm { t o }$ penalize control effort and $w _ { s } = 1 0 ^ { 5 }$ to penalize the slack variable $s \in \mathbb { R }$ , while $\gamma = 1 5 0$ denotes the base weight and $\kappa = 3 . 0$ the exponent of the time-dependent weighting. The discrete-time transition $\mathbf { x } _ { k + 1 } = f _ { d } ( \mathbf { x } _ { k } , u _ { k } )$ is obtained via fourth-order Runge-Kutta (RK4) integration of the continuous-time dynamics using timestep ∆t.

The MPC is formulated to swing up and stabilize the Furuta pendulum within a single prediction horizon, while respecting box constraints on the base angle and control input. The objective is to achieve this task in minimum time while converging the base angle to its nominal upright position. Time-optimality is implemented using an exponential weight [1], [2]. A vanishing terminal control input is used to encourage steady-state behavior upon reaching the target terminal state. The optimal control problem is transcribed using multiple shooting with Fatrop [31], within the Impact library [32], leveraging its C-API for faster execution.

Empirically, a good initial guess is crucial for solver convergence. Following [33], we therefore maintain a growing database of previously computed feasible trajectories. A KD-tree queries this database for nearby trajectories, which provide warm starts for subsequent optimizations.

## B. NNMPC Training Setup

The NN is a Multilayer Perceptron (MLP), with three fully connected hidden layers of 512 nodes each and tanh activations, mapping the current state $\mathbf { X } _ { k }$ and target state $\mathbf { X } _ { f }$ (eight inputs) to the T2G and the nominal control output u<sub>k,raw</sub> $u _ { k , \ast }$ (two outputs). The latter is subsequently passed through the safety layer to obtain the applied control input $u _ { k , \mathrm { s a f e } }$ . The multi-objective loss is formulated as:

$$
L _ { \mathrm { t o t a l } } = \lambda _ { u } L _ { u } + \lambda _ { \mathrm { T 2 G } } L _ { \mathrm { T 2 G } } + \lambda _ { b } L _ { \mathrm { b a r r i e r } } + \lambda _ { p } L _ { \mathrm { p e n a l t y } } ,\tag{9}
$$

where each term $L _ { i }$ is computed as a weighted mean $\frac { \sum w _ { \mathrm { k n n } } \cdot \ell _ { i } } { \sum w _ { \mathrm { k n n } } }$ with $w _ { \mathrm { k n n } }$ the k-nearest neighbors weights of Section III-C, and $\lambda _ { i }$ hyperparameters. The imitation error functions employed are: $\ell _ { u } = ( u _ { \mathrm { s a f e } } - u _ { \mathrm { e x p e r t } } ) ^ { 2 } , \quad \ell _ { \mathrm { T 2 G } } = | \mathrm { T 2 G } - \mathrm { T 2 G } _ { \mathrm { e x p e r t } } |$ The barrier and penalty terms are defined in Section III-F.

## C. Training Procedure

First, the network is pre-trained on an initial grid-based dataset of 15,601 data points for 1200 epochs, using a learning rate of $8 \times 1 0 ^ { - 4 }$ . Then, the iterative training procedure presented in Section III-D is run for 1441 iterations, aggregating 1,189,320 data points, of which 708,312 remained after optimal sampling (Section III-B). Training is done using minibatches of 4096 samples, with 70 epochs per iteration and at most 15 new trajectories (900 data points) added per iteration with a fixed learning rate of $1 0 ^ { - 6 }$

## V. CLOSED-LOOP VALIDATION

## A. Simulation

Fig. 3 compares the convergence performance of the expert MPC and NNMPC over 1000 randomly sampled initial states within the region of interest. Each state is evaluated independently with both controllers over a 3.6-second evaluation window, twice the MPC prediction horizon. Both controllers successfully converge in 91.5% of the cases, while only the NNMPC succeeds in 4.2% and only the MPC in 4.3%.

Fig. 3a shows that the NNMPC takes slightly longer to reach the reference state, with a median time of 1.62 s compared to 1.35 s for the MPC. However, Fig. 3b reveals a nonnegligible number of cases in which the NNMPC reaches the reference faster than the expert MPC. This indicates that the expert MPC is not always time-optimal and that the NNMPC can generalize beyond the demonstrated trajectories, in some cases achieving faster convergence. The remaining difference in convergence time may result from insufficient training or suboptimal handling of conflicting labels in the MPC-generated dataset.

![](images/87f0b7c6940200173ca640d867acdcdc8e22b19b35447bb205f03d644fa63cd2.jpg)  
Time to reach reference (s)

(a) Distribution of convergence times for the NNMPC and the expert MPC.  
![](images/98880208852bec3f7f46947a6f8d40df4f215e2382c9b2e180d0d9721294381a.jpg)  
(b) Distribution of pairwise convergence-time differences.  
Fig. 3. Performance comparison of the expert MPC and NNMPC over 1000 randomly sampled initial states.

## B. Computation time

Table I compares the per-iteration computation time of the original MPC with the proposed NNMPC executed in real time using CPU (Intel Core i7-9850H) and GPU (Nvidia Quadro P1000 Mobile) inference. When deployed on CPU, the NNMPC controller supports a maximum control frequency of ≈100 Hz. On GPU, inference consistently remains below 1 ms, theoretically enabling control frequencies up to 1 kHz. This represents a substantial improvement compared to the original MPC, whose maximum reliable control frequency of approximately 1.36 Hz severely limits its use for real-time control.

## VI. COMPONENT VALIDATION

The individual building blocks of the proposed NNMPC architecture are evaluated through simulation. Closed-loop performance is quantified by randomly sampling initial states from a region of interest and recording whether the controller successfully reaches the terminal state within 3.6 seconds. The percentage of trajectories that fail to reach the terminal state within this time is referred to as the failed trajectory percentage.

TABLE I  
PER-ITERATION COMPUTATION TIME DURING PENDULUM SWING-UP AND THE RESULTING MAXIMUM RELIABLE CONTROL FREQUENCY.
<table><tr><td rowspan="2"></td><td colspan="3">Computation time (ms)</td><td rowspan="2">Max. reliable freq. (Hz)</td></tr><tr><td>Mean</td><td>Std. dev.</td><td>Max</td></tr><tr><td>MPC</td><td>114.165</td><td>150.132</td><td>732.676</td><td>1.36</td></tr><tr><td>NNMPC (CPU)</td><td>4.839</td><td>0.809</td><td>10.576</td><td>94.6</td></tr><tr><td>NNMPC (GPU)</td><td>0.409</td><td>0.094</td><td>0.998</td><td>1002</td></tr></table>

## A. Conflicting expert labels

Fig. 4 displays the results of ablating the proposed T2Gbased weighting strategy of Section III-C, showing that its removal leads to a nonnegligible increase in overall failedtrajectory percentage.

## B. Self-Informed Sampling

Fig. 5 evaluates the proposed self-informed sampling (SIFS) strategy by replacing the state-selection component of Section III-D with DAgger and MC-dropout (implemented according to [19]). DAgger samples uniformly, and MCdropout uses epistemic uncertainty estimation based on 20 forward passes with a dropout probability of 30%. As shown, SIFS achieves a notably faster reduction in failure rate. Fig. 5b demonstrates that both DAgger and MC-dropout become increasingly less data- and time-efficient compared to SIFS. For this particular example, comparable performance is reached approximately three times faster and with less than half the required training data.

## C. Controller Adaptability

Both the nominal and ancillary controllers were evaluated on systematically perturbed plant models. Fig. 6 shows that both ancillary architectures improve robustness overall, with LQR performing better for some perturbations and DIC for others. The best choice therefore depends on the plant and the expected model mismatch. Rather than advocating a particular ancillary architecture, these results demonstrate that simple ancillary controllers can provide useful adaptability to model mismatch without retraining.

## VII. HARDWARE VALIDATION

Finally, the NNMPC is deployed on a physical Furuta pendulum, using the C++ deployment framework ImpaC++t of [34]. The base is driven by a stepper motor, and the pendulum angle is measured by an incremental encoder (4096 ticks/rev). Both the motor controller and the encoder interface are connected via CANopen over EtherCAT, using the SOEM master library. The I/O thread runs at 1 kHz and the NNMPC at 250 Hz. The full pipeline runs on a laptop with standard Linux using a GPU for the network evaluation.

![](images/07cd5a0fc19c513ea7296c8243f0af89ab40bfbf1864856cc6bd0c355fe8a2d0.jpg)  
Fig. 4. Evolution of the failed trajectory percentage over training iterations, with and without the proposed weighting strategy.

SIFS training time (h)  
![](images/6ccc94f95938c91f94f633e58fa0e6bfb665ca1117b5ba66c8bcb6ca7a420e47.jpg)

![](images/ee5ae41440421a0b2b2b7eb008dc3614579311d844003d5c1860312736164225.jpg)  
(a) Failure rate over training it- (b) Additional expert queries and training erations. time needed to match the SIFS failure rate.  
Fig. 5. Training progress of self-informed sampling (SIFS), DAgger, and MC-dropout.

The controller performance for the physical pendulum was assessed through a swing-up task. Fig. 7 shows that the real-world state trajectory closely matches the simulated results. The main discrepancy is a high-frequency oscillatory control effort near the inverted equilibrium, attributable to sensor noise and residual plant–model mismatch. This did not compromise stability and could be further suppressed with a low-pass filter on the NNMPC output. Fig. 8 displays the swing-up task on the physical setup.

## VIII. CONCLUSION

We presented an integrated architecture for imitating TOMPC with a neural network, which combines state-of-theart techniques for dataset correlation, constraint satisfaction, and covariate shift with three original contributions tailored to TOMPC: self-informed sampling using a jointly learned T2G, a lightweight T2G-based weighting of conflicting expert labels, and ancillary controllers that compensate for plant–model mismatch without retraining.

On a Furuta pendulum, self-informed sampling matched the performance of state-of-the-art methods with less than half the training data and roughly one third of the training time, while the T2G-based weighting markedly reduced the failure rate. The resulting NNMPC attains a region of attraction on par with the expert MPC at an over 700-fold lower worst-case computation time (on GPU), and the ancillary controllers substantially improved generalization to perturbed plant models in simulation. On hardware, running at 250 Hz, the controller swings up and stabilizes the pendulum along trajectories closely matching simulation.

![](images/0fc5232a179f7522942e49baa205776ec04e86483cb0360a8418ceb317fdd164.jpg)  
Fig. 6. Each model parameter (m , L , b , l , J<sup>ˆ</sup> ; notation as in [30]) was systematically perturbed via $p  p \cdot ( 1 + \delta / 2 )$ , where δ denotes the normalized deviation, varying between −1 and 1. For every parameter setting, 200 rollouts were performed from randomly sampled initial states. Each cell reports the percentage of successful trajectories.

![](images/157074ca6db5b1f58492914e54c3edd92a02ec39f77e00739336840b45dfe84d.jpg)

̇ θ<sub>1</sub> hardware ̇ θ<sub>1</sub> simulation ̇ θ<sub>1</sub> MPC (expert) ̇ θ<sub>2</sub> hardware ̇ θ<sub>2</sub> simulation ̇ θ<sub>2</sub> MPC (expert)  
![](images/c032d265ba41576b96a7e6ec2fcd73d786af867e92d393fb79bcfde42fe9122f.jpg)

![](images/53c2953a7af045dbfb59fff5e7f471ec6d0a7fae3e8ef98902c572460c7bb56f.jpg)  
Fig. 7. State and control trajectories of the expert MPC solution, the NNMPC in simulation, and the NNMPC deployed on the physical pendulum. The solid hardware curves show one representative run. The shaded bands show the minimum-to-maximum envelope over four additional hardware reruns.

The remaining performance gap may stem from insufficient training or conflicting expert labels. Future work could explore multimodal architectures, such as Mixture Density Networks [35], and continual learning to limit training-time growth as the dataset expands.

![](images/d487da4fe9644f816dd1af41665d9f92fd0c9b7d12792b284dc60b2f630a8bec.jpg)  
Fig. 8. Swing-up and stabilization starting from the downward equilibrium. Timestamps correspond to the final frame in each overlaid sequence.

## APPENDIX

Videos demonstrating the experimental results and further implementation details are available on the project webpage: https://maartendebaecke.github.io/PW/neural-tompc/.

## ACKNOWLEDGMENT

The authors thank Alvaro Javier Florez Martinez for his support with the MPC design and the IMPACT software and Mathias Bos for his support with the hardware validation.

## REFERENCES

[1] R. Verschueren, H. J. Ferreau, A. Zanarini et al., “A stabilizing nonlinear model predictive control scheme for time-optimal point-topoint motions,” in 2017 IEEE 56th Annual Conference on Decision and Control (CDC), 2017, pp. 2525–2530.

[2] A. Florez, W. Decre, and J. Swevers, “Practical approaches for time-´ optimal mpc: Dual stage and artificial reference methods,” in 2025 European Control Conference (ECC), 2025, pp. 358–363.

[3] D. Dirckx, M. Bos, B. Vandewal et al., “Asap-mpc: an asynchronous update scheme for online motion planning with nonlinear model predictive control,” Auton. Robots, vol. 49, no. 1, Mar. 2025.

[4] M. Bain and C. Sammut, “A framework for behavioural cloning,” in Machine Intelligence 15, Intelligent Agents [St. Catherine’s College, Oxford, July 1995]. GBR: Oxford University, 1999, p. 103–129.

[5] B. Karg and S. Lucia, “Efficient representation and approximation of model predictive control laws via deep learning,” IEEE Transactions on Cybernetics, vol. 50, no. 9, pp. 3866–3878, 2020.

[6] N. R. Mohanty, P. Ubare, V. Patne et al., “Embedded dnn-based nonlinear mpc with guaranteed constraint satisfaction,” Neural Computing and Applications, vol. 37, no. 18, p. 12305–12328, Mar. 2025.

[7] T. Li and K. You, “Neural mpc for quadrotor trajectory tracking on embedded gpus,” Control Theory and Technology, vol. 24, pp. 271– 281, 2026.

[8] A. Tagliabue and J. P. How, “Efficient deep learning of robust policies from mpc using imitation and tube-guided data augmentation,” IEEE Transactions on Robotics, vol. 40, pp. 4301–4321, 2024.

[9] C. Kong, B. Wang, L. He et al., “Ann-mpc: An adaptive neural network-based nonlinear model predictive control method for robust trajectory tracking of quadrotors,” IEEE Robotics and Automation Letters, vol. 10, no. 10, pp. 10 966–10 973, 2025.

[10] H. Hose, A. Grafe, and S. Trimpe, “Parameter-adaptive approxi-¨ mate MPC: Tuning neural-network controllers without retraining,” in Proceedings of the 6th Annual Learning for Dynamics & Control Conference, ser. Proceedings of Machine Learning Research, A. Abate, M. Cannon, K. Margellos et al., Eds., vol. 242. PMLR, 15–17 Jul 2024, pp. 349–360.

[11] S. Ross, G. Gordon, and D. Bagnell, “A reduction of imitation learning and structured prediction to no-regret online learning,” in Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, ser. Proceedings of Machine Learning Research, G. Gordon, D. Dunson, and M. Dud´ık, Eds., vol. 15. Fort Lauderdale, FL, USA: PMLR, 11–13 Apr 2011, pp. 627–635.

[12] A. Castillo, E. Pryor, A. El Fathi et al., “Neural networks for onchip model predictive control: A method to build optimized training datasets and its application to type-1 diabetes,” IEEE Transactions on Cybernetics, vol. 55, no. 7, pp. 3062–3072, 2025.

[13] S. Zhang, Z. Cao, D. Sadigh et al., “Confidence-aware imitation learning from demonstrations with varying optimality,” in Advances in Neural Information Processing Systems, M. Ranzato, A. Beygelzimer, Y. Dauphin et al., Eds., vol. 34. Curran Associates, Inc., 2021, pp. 12 340–12 350.

[14] A. Bemporad, M. Morari, V. Dua et al., “The explicit linear quadratic regulator for constrained systems,” Automatica, vol. 38, no. 1, pp. 3– 20, 2002.

[15] Y. Zhang, Y. Xie, H. Liu et al., “Scizor: A self-supervised approach to data curation for large-scale imitation learning,” arXiv:2505.22626, 2025, preprint.

[16] M. Yang, S. Levine, and O. Nachum, “TRAIL: Near-optimal imitation learning with suboptimal data,” in International Conference on Learning Representations, 2022.

[17] H. Xu, X. Zhan, H. Yin et al., “Discriminator-weighted offline imitation learning from suboptimal demonstrations,” in Proceedings of the 39th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, K. Chaudhuri, S. Jegelka, L. Song et al., Eds., vol. 162. PMLR, 17–23 Jul 2022, pp. 24 725–24 742.

[18] M. Laskey, J. Lee, R. Fox et al., “Dart: Noise injection for robust imitation learning,” in Proceedings of the 1st Annual Conference on Robot Learning, ser. Proceedings of Machine Learning Research, S. Levine, V. Vanhoucke, and K. Goldberg, Eds., vol. 78. PMLR, 13–15 Nov 2017, pp. 143–156.

[19] Y. Cui, D. Isele, S. Niekum et al., “Uncertainty-aware data aggregation for deep imitation learning,” in 2019 International Conference on Robotics and Automation (ICRA). IEEE Press, 2019, p. 761–767.

[20] B. Lakshminarayanan, A. Pritzel, and C. Blundell, “Simple and scalable predictive uncertainty estimation using deep ensembles,” in Proceedings of the 31st International Conference on Neural Information Processing Systems, ser. NIPS’17. Red Hook, NY, USA: Curran Associates Inc., 2017, p. 6405–6416.

[21] Y. Gal and Z. Ghahramani, “Dropout as a bayesian approximation: Representing model uncertainty in deep learning,” in Proceedings of The 33rd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, M. F. Balcan and K. Q. Weinberger, Eds., vol. 48. New York, New York, USA: PMLR, 20–22 Jun 2016, pp. 1050–1059.

[22] A. Giusti, M. Monsieur, and B. Mrak, “STRIDE: Investigating efficient sampling strategies for neural approximations of optimal control,” in Proceedings of the 23rd International Conference on Informatics in Control, Automation and Robotics (ICINCO), vol. 1, 2026, pp. 166– 173.

[23] T. Zieger, A. Savchenko, T. Oehlschlagel¨ et al., “Towards safe neural network supported model predictive control,” IFAC-PapersOnLine, vol. 53, no. 2, pp. 5246–5251, 2020, 21st IFAC World Congress.

[24] S. Adhau, V. V. Naik, and S. Skogestad, “Constrained neural networks for approximate nonlinear model predictive control,” in 2021 60th IEEE Conference on Decision and Control (CDC), 2021, pp. 295– 300.

[25] A. D. Ames, X. Xu, J. W. Grizzle et al., “Control barrier function based quadratic programs for safety critical systems,” IEEE Transactions on Automatic Control, vol. 62, no. 8, pp. 3861–3876, 2017.

[26] P. Christiano, Z. Shah, I. Mordatch et al., “Transfer from simulation to real world through learning deep inverse dynamics model,” arXiv preprint arXiv:1610.03518, 2016.

[27] J.-J. E. Slotine and W. Li, Applied Nonlinear Control. Englewood Cliffs, NJ: Prentice-Hall, 1991.

[28] S. Kim, J. Lee, K. Han et al., “Bridging the sim-to-real gap: A control framework for imitation learning of model predictive control,” International Journal of Control, Automation, and Systems, vol. 24, no. 3, p. 425–443, Feb. 2026.

[29] K.-C. Hsu, H. Hu, and J. F. Fisac, “The safety filter: A unified view of safety-critical control in autonomous systems,” Annual Review of Control, Robotics, and Autonomous Systems, vol. 7, no. 1, pp. 47–72, 2024.

[30] B. Cazzolato and Z. Prime, “On the dynamics of the furuta pendulum,” Journal of Control Science and Engineering, vol. 2011, 04 2011.

[31] L. Vanroye, A. Sathya, J. De Schutter et al., “Fatrop: A fast constrained optimal control problem solver for robot trajectory optimization and control,” in 2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2023, pp. 10 036–10 043.

[32] A. Florez, A. Astudillo, W. Decre´ et al., “Impact: A toolchain for nonlinear model predictive control specification, prototyping, and deployment,” IFAC-PapersOnLine, vol. 56, no. 2, p. 3164–3169, 2023.

[33] T. S. Lembono, A. Paolillo, E. Pignat et al., “Memory of motion for warm-starting trajectory optimization,” IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 2594–2601, 2020.

[34] M. Bos, W. Decre, and J. Swevers, “ImpaC++t: A C++ toolchain´ for seamless deployment of nonlinear model predictive control with Impact,” Presented at the 45th Benelux Meeting on Systems and Control, Lommel, Belgium, Mar. 2026, abstract in the Book of Abstracts.

[35] M. Okamoto, J. Ren, Q. Mao et al., “Approximation of nonlinear model predictive control using mixture density networks,” in Proc. Found. Comput.-Aided Process Oper./Chem. Process Control (FO-CAPO/CPC), 2023.