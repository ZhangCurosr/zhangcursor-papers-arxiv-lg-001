# Causal World Models under Physical Parameter Shifts

Chu Lin<sup>1</sup> and Yi Sun<sup>2,∗</sup>

<sup>1</sup>Department of Computer Science, <sup>2</sup>Department of Electrical Engineering,

The City College of New York, New York, NY 10031, USA

<sup>∗</sup>Corresponding author: ysun@ccny.cuny.edu

## Abstract

World models predict how embodied systems evolve under actions, but their accuracy can degrade when physical conditions move outside the training distribution. We study two components of causal modeling: which inputs a predictor may use, and how its mechanism computes their efects. In a synthetic benchmark with shortcut features, a frozen dependency mask derived from paired interventions reduces mean out-of-distribution (OOD) RMSE from 1.769 for a fully connected predictor to 0.697, close to the generator-oracle mask (0.694). For robot link-mass extrapolation in CausalWorld, with disjoint training and test mass ranges, we develop finger-local world models in which link masses enter inertia and dynamics-bias terms through explicit afine coeficients, actions act through either a learned torque network or the known PD controller, and learned residuals capture contact-related efects. Compared with a separately trained control whose mass inputs are fixed at training means, the known-PD model turns out-of-range masses into a consistent predictive gain: median time-averaged joint R over 150-step rollouts rises from 0.609 to 0.727, with improvement in all ten task–seed runs (exact two-sided sign test, p ≈ 0.002). In a shared 100-step forecast, the learned-torque model is competitive with the strongest recurrent robotic world model (RWM) configuration (median joint R<sup>2</sup> of 0.755–0.768 versus 0.752), whereas supplying out-of-range masses sharply degrades the MLP and recurrent predictors. These results indicate that how a physical parameter enters the dynamics mechanism determines whether its out-of-range value helps or harms prediction.

Keywords: embodied world models; causal dependencies; structured dynamics; physical parameter extrapolation; mass-afine mechanisms.

## 1 Introduction

World models are used to predict the evolution of embodied systems as they execute actions [1]. Their prediction accuracy may degrade when physical conditions deviate from the training environment. Recent theory connects generalization across multi-step goals to learned predictive world models [2]. Embodied applications require these models to remain useful under changing conditions, such as when a robot handles a diferent payload or operates with diferent physical properties. Pearl’s structural causal model (SCM) framework separates structural dependencies from the functions governing those relationships [3]. This separation motivates two questions for a world model: which inputs should a predictor use, and how should its dynamics mechanism compute their efects?

Causal dynamics models, such as CDL [4] and recent structure-distribution approaches [5], learn or estimate dependencies between current state, action, and next state, and use them to restrict irrelevant inputs to the dynamics predictor. Such constraints remove direct prediction paths from excluded shortcut inputs. However, if the OOD shift involves a true causal factor, such as a heavier robot link, the relevant dependency must remain. The computation performed after reading the shifted value is then determined by the mechanism. Input selection alone does not specify how this function should extrapolate. Recent context-aware mechanical models and structured physical world models further motivate examining how parameter information enters the dynamics mechanism [6, 7].

We develop a robot world-model configuration that assigns physical parameters and control inputs distinct roles: link masses enter inertia and dynamics bias through explicit afine coeficients, actions produce torque through the known PD controller, and a neural residual models contact-related efects without directly receiving robot masses. The dependencies are fixed to each finger’s local state and action, with shared object context. We compare this configuration with an MLP and mass-afine dynamics with learned torque under the same input routing. Each model is paired with a separately trained mass-information control: mass inputs are removed from the MLP or fixed at their training means in the structured models. A complementary synthetic experiment uses an intervention-derived mask to evaluate shortcut dependence. Figure 1 summarizes the robot architecture and prediction procedure.

![](images/a4838c79ae48cc76f986b8c771d2d33bdd5ad558303b3259dc3f38460c350a69.jpg)  
Figure 1: Overview of the robot world model. Three finger mechanisms predict joint accelerations under fixed input dependencies; one mechanism family is used per model. A separate object MLP predicts object-state changes. Joint integration and object updates produce future states, which are fed back for rollout prediction. Training uses simulator trajectories; evaluation uses link masses outside the training range.

The experiments connect dependency selection to physical parameter extrapolation. Frozen interventionderived masks reduce mean synthetic OOD RMSE by 60.6%. In CausalWorld, supplying out-of-range link masses to an unstructured MLP or to a recurrent robotic world model sharply degrades prediction, whereas the known-PD mass-afine model improves its median time-averaged joint ${ \dot { R } } ^ { 2 }$ over 150 steps from 0.609 for its fixed-mass-input control to 0.727. The median within-run benefit of mass information is 0.121 under the heavy-mass shift, compared with 0.009 within the training mass range. In a shared 100-step benchmark, the learned-torque model is competitive with the strongest recurrent baseline (median joint $R ^ { 2 }$ of 0.755 with actual masses and 0.768 with fixed mass inputs, versus 0.752), and the known-PD model retains a positive mass benefit in every run.

Our contributions are summarized as follows:

• A mass-informed robot world-model design. We develop finger-local mass-afine dynamics with learned-torque and known-PD variants. Masses enter inertia and bias through explicit coeficients, while torque and residual networks use mass-excluded inputs. Joint integration and a separate object-state predictor generate future trajectories.

• Competitive OOD prediction in a shared CausalWorld benchmark. Over a common 100-step forecast, the learned-torque model reaches median joint $R ^ { 2 }$ of 0.755 with actual masses and 0.768 with fixed mass inputs, compared with 0.752 for the strongest tested RWM configuration, with slightly lower joint NRMSE and object-position RMSE and without using observation history. The accuracy margins are small; the larger diference lies in how the models respond to out-of-range mass inputs, which sharply degrade the MLP and recurrent predictors.

• Consistent predictive benefit from out-of-range masses. Separately trained mass-information controls show that the known-PD design improves median 150-step mean joint $R ^ { 2 }$ from 0.609 to 0.727. Actual masses improve prediction in all ten task–seed runs in both the 150-step ablation and the 100-step benchmark; the latter yields a median paired mass gain of 0.071 (exact two-sided sign test, $p \approx 0 . 0 0 2$ in each case).

## 2 Problem Statement

Given state $s _ { t } ,$ , action $a _ { t }$ , robot link masses $m ,$ and other physical context $c ,$ we predict

$$
\hat { s } _ { t + 1 } = F _ { \theta , G } ( s _ { t } , a _ { t } , m , c ) ,\tag{1}
$$

where G specifies permitted input dependencies and θ parameterizes the mechanisms. The robot mass study assumes that the current link masses and the simulator’s low-level control law are available. The networks learn dynamics coeficients from state-prediction supervision, using the current mass values to specify the physical condition. A rollout starts from an observed state and repeatedly applies $F _ { \theta , G }$ using predicted states and the recorded action sequence.

We distinguish two sources of distribution shift. In a shortcut shift, an input correlated with an output during training changes its relationship to that output; excluding an irrelevant input removes that prediction path. In physical parameter extrapolation, a relevant parameter takes values outside the training support. The model must retain its dependence on that parameter and compute its efect at unseen values. The synthetic study evaluates fixed masks in a benchmark containing shortcut shifts. The robot study isolates a link-mass shift under a shared dependency structure.

The robot evaluation asks both whether predictions are accurate and whether actual mass information improves them. For mechanism family r and task–seed run $b ,$ we define the predictive benefit of mass as

$$
\Delta _ { r , b } = \overline { { R ^ { 2 } } } _ { r , b } ^ { \mathrm { a c t u a l } } - \overline { { R ^ { 2 } } } _ { r , b } ^ { \mathrm { c o n t r o l } } ,\tag{2}
$$

where $\overline { { R ^ { 2 } } }$ is the mean joint $R ^ { 2 }$ across the evaluated rollout steps. The controls and score aggregation are specified below. This separates absolute prediction accuracy from the additional value of providing the physical parameter.

## 3 Related Work

Causal dynamics and dependency constraints. SCMs separate dependencies from the mechanisms generating variables [3]. Theoretical studies connect robustness under specified distribution shifts to causal world-model knowledge, including settings in which actions afect the environment [8, 9]. CDL learns causal input dependencies for dynamics prediction and task-independent state abstraction [4]; FCDL refines the structure through quantized state–action subgroups [10]. More recently, CADY estimates a distribution over causal structures and samples masks for a probabilistic dynamics model, evaluating robustness to corrupted inputs and changes in robot dynamics [5]. CoDA uses local causal structure for counterfactual data augmentation [11], while BECAUSE learns causal representations to address confounding in ofline model-based reinforcement learning [12]. We fix the dependency structure to study the functional response to a supplied physical parameter outside its training range.

Composable and masked world models. WM3C uses language-guided compositional causal components to model unseen environments [13]. Causal-JEPA uses object-level latent masking to encourage interactiondependent prediction and evaluates visual reasoning and predictive control [14]. These approaches organize latent information and interactions. Our study uses recorded physical states and known link masses, focusing on how a dynamics mechanism turns an out-of-range parameter into a useful change in its predicted trajectory.

Structured mechanics and physical parameterization. DeLaN embeds Lagrangian mechanics into learned dynamics [15, 16]; classical inertial identification uses the linear dependence of inverse dynamics on inertial parameters [17]. Structured Mechanical Models and APHYNITY combine known physical structure with learned components [18, 19]. Recent extensions are particularly relevant: Context-Aware DeLaN combines recurrent online identification with contextual mechanical models and residual dynamics for a robot arm under varying loads [6]; Floating-Base DeLaN imposes branch-induced sparsity and physical consistency on learned inertia matrices [20]. Both directly build on DeLaN. Our configuration makes measured masses explicit afine coeficients of inertia and bias, routes actions through a known controller, and assesses parameter utility using separately trained mass-information controls.

Physical world models and extrapolation. Architectural functional form influences neural extrapolation [21]. Wang et al. study structured physical world models with learned energy, dissipation, known drive channels, and separable intervention-parameter maps in electromagnetic and fluid systems [7]. This combines imposed physical structure with learned system-specific functions. We study contact-rich robot prediction under link-mass changes, using a mass-afine joint model with a separately learned state- and velocity-dependent bias and an object-state predictor. The target is the predictive benefit of known masses over finite robot rollouts.

History-based adaptation and robotic prediction. UP-OSI estimates physical parameters from state–action history, while CaDM conditions dynamics on a learned context; RMA uses an adaptation module for policy control [22–24]. Context-Aware DeLaN further couples online identification to structured mechanics [6]. RWM uses recurrent history and autoregressive training for long-horizon robotic prediction [1]. These methods motivate separating inference of unknown context from use of a supplied parameter: in our mass experiments the actual link masses and low-level control law are available, and the question is how the dynamics representation uses them.

Evaluation and baseline methods. PETS combines probabilistic ensemble dynamics, trajectory sampling, and model-predictive control [25]. CausalWorld provides configurable robot and object properties for embodied transfer studies [26]. We use CausalWorld for both environment-transfer prediction with a PETSstyle dynamics baseline and the link-mass mechanism study, with separate metrics for the two protocols. Appendix A identifies the sources of established components and our design choices.

## 4 Methodology

## 4.1 Fixed dependencies and model decomposition

The robot world model contains three separately parameterized finger mechanisms and one object-state MLP. Each finger uses its own three joint positions, three velocities, three actions, four mass entries, and shared object state and context. Its mechanism predicts three joint accelerations, which are integrated to update that finger’s joint state. The object MLP reads all joint positions, velocities, and actions together with the object state and context, and predicts a 13-dimensional state increment. It receives robot-mass efects through the evolving state. This input routing is fixed across the three mechanism families; each model learns its own network weights.

For the synthetic dependency experiment, the mask is estimated from paired interventions. For each intervened input i, we average the absolute output change and normalize each output column:

$$
E _ { i j } = \frac { 1 } { N _ { i } } \sum _ { n : I _ { n } = i } | y _ { n , j } ^ { \mathrm { c f } } - y _ { n , j } ^ { \mathrm { b a s e } } | , \quad S _ { i j } = \frac { E _ { i j } } { \operatorname* { m a x } _ { k } E _ { k j } + 1 0 ^ { - 9 } } , \quad A _ { i j } = \mathbf { 1 } \{ S _ { i j } \geq 1 0 ^ { - 3 } \operatorname* { m a x } _ { k , l } \} .\tag{3}
$$

Scores use the training intervention pool; the mask is frozen before fitting the predictor. A shared scalar encoder, edge-specific linear maps, and output heads give

$$
\hat { y } _ { j } = g _ { \theta , j } \left( \sum _ { i } A _ { i j } W _ { i j } \phi _ { \theta } ( x _ { i } ) \right) .\tag{4}
$$

The robot structure is specified by finger-local routing, whereas the synthetic mask is intervention-derived.

## 4.2 Mass-afine dynamics mechanisms

The MLP baseline maps the permitted inputs to joint acceleration. The structured alternatives separate the known mass values from learned state-dependent coeficients. For one finger, let $\mu _ { r } = m _ { r } / ( 0 . 0 3 \mathrm { k g } )$ . We form

$$
\mathcal { M } ( { \boldsymbol { q } } , m ) = L _ { 0 } ( { \boldsymbol { q } } ) L _ { 0 } ( { \boldsymbol { q } } ) ^ { \top } + \sum _ { r = 1 } ^ { 4 } \mu _ { r } L _ { r } ( { \boldsymbol { q } } ) L _ { r } ( { \boldsymbol { q } } ) ^ { \top } + 1 0 ^ { - 4 } I ,\tag{5}
$$

$$
h _ { \theta } ( q , \dot { q } , m ) = b _ { 0 } ( q , \dot { q } ) + \sum _ { r = 1 } ^ { 4 } \mu _ { r } b _ { r } ( q , \dot { q } ) .\tag{6}
$$

Each $L _ { r }$ is lower triangular with diagonal entries so $\mathrm { f t p l u s } ( v ) + 1 0 ^ { - 3 }$ , giving positive-definite M for nonnegative masses. The bias coeficients are learned separately from the inertia coeficients. $h _ { \theta }$ models position- and velocity-dependent dynamics bias. All coeficient networks depend on state, while mass enters through the explicit coeficients $\mu _ { r }$

The learned-torque configuration computes

$$
\ddot { q } = { D _ { \alpha } } { \mathcal { M } } ^ { - 1 } \big [ \tau _ { \theta } ( q , \dot { q } , a ) + c _ { \theta } ( q , \dot { q } , z ) - h _ { \theta } \big ] ,\tag{7}
$$

where z contains object state and context, and $D _ { \alpha }$ is a learned positive diagonal acceleration scale initialized from training acceleration statistics. The torque network and contact-related residual $c _ { \theta }$ do not directly take robot mass as input. The residual is learned end to end from state-prediction error. The terms M and $h _ { \theta }$ are afine in mass; solving the dynamics introduces the corresponding nonlinear dependence of acceleration on mass.

## 4.3 Known PD control and state updates

The known-PD configuration maps normalized action a to target joint angles $q ^ { * }$ using the simulator’s joint limits, and computes

$$
\tau _ { \mathrm { P D } } = \mathrm { c l i p } _ { \tau _ { \mathrm { m a x } } } \big [ \mathrm { c l i p } _ { \tau _ { \mathrm { m a x } } } \big ( K _ { p } ( q ^ { * } - q ) - K _ { d } { \dot { q } } \big ) - K _ { d , s } { \dot { q } } \big ] .\tag{8}
$$

Clipping is elementwise. The controller uses $K _ { p } = 1 0$ $K _ { d } = ( 0 . 1 , 0 . 3 , 0 . 0 0 1 )$ , $K _ { d , s } = ( 0 . 0 8 , 0 . 0 8 , 0 . 0 4 )$ per finger, and $\tau _ { \mathrm { m a x } } = 0 . 3 6$ N m. This reproduces the simulator’s PD torque, safety damping, and torque limits [27]. The rule uses action and joint state; mass afects its rollout response through state feedback. The mass-afine dynamics in physical units are

$$
\ddot { q } = ( S \mathcal { M } S ) ^ { - 1 } \big [ \tau _ { \mathrm { P D } } + \tau _ { \mathrm { m a x } } c _ { \theta } ( q , \dot { q } , z ) - \tau _ { \mathrm { m a x } } h _ { \theta } \big ] ,\tag{9}
$$

where the fixed diagonal scale S is the square root of the per-joint ratio of training torque standard deviation to training acceleration standard deviation. Actions enter this mechanism through the known controller; networks learn the inertia, dynamics bias, and contact-related residual. This configuration specifies both the torque source and the physical scaling of the learned terms.

For all mechanisms, accelerations are softly limited, followed by semi-implicit integration:

$$
\tilde { \dot { q } } = a _ { \mathrm { m a x } } \operatorname { t a n h } ( \ddot { q } / a _ { \mathrm { m a x } } ) , \quad \hat { \dot { q } } _ { t + 1 } = v _ { \mathrm { m a x } } \operatorname { t a n h } \biggl ( \frac { \dot { q } _ { t } + \Delta t \tilde { \ddot { q } } } { v _ { \mathrm { m a x } } } \biggr ) , \quad \hat { q } _ { t + 1 } = q _ { t } + \Delta t \hat { \dot { q } } _ { t + 1 } .\tag{10}
$$

Limits and arithmetic are coordinatewise. The object MLP predicts a normalized increment that is transformed back to state units and added to the current object state. Object quaternions are normalized during rollouts. Joint and object predictions form the next state in Eq. (1).

## 4.4 Training and mass-information controls

Models are first trained on one-step transitions and then fine-tuned on free-running trajectories. The multi-step objective combines teacher-forced one-step error with rollout error:

$$
\mathcal { L } = \mathbb { E } \mathrm { \mathrm { ~ M S E } } \left( \frac { \hat { s } _ { t + 1 } ^ { \mathrm { T F } } - s _ { t + 1 } } { \sigma _ { \Delta s } } \right) + \frac { \lambda } { H } \sum _ { k = 1 } ^ { H } \mathbb { E } \mathrm { \mathrm { ~ M S E } } \left( \frac { \hat { s } _ { t + k } ^ { \mathrm { r o l l } } - s _ { t + k } } { \sigma _ { s } } \right) ,\tag{11}
$$

where MSE averages squared components, $\lambda = 1$ , and normalization statistics are fitted to training data. Each rollout begins with an observed state and then uses its own predictions. Model selection uses validation trajectories from the training mass domain.

We use three complementary mass-information controls. First, a separately trained MLP removes robot-mass inputs. Second, each structured model has a separately trained counterpart in which mass inputs are overwritten by their training means during both training and evaluation. These counterparts use the same varying-mass simulator trajectories as the corresponding actual-mass models. Third, an inference-time diagnostic replaces the trained MLP’s test mass inputs with training means while retaining its weights. This last comparison measures the efect of the shifted input on an already fitted predictor.

## 5 Simulation Results

## 5.1 Experimental setup and metrics

The synthetic benchmark has two interacting objects, 32 inputs, six next-state outputs, and 17 distractor inputs. Its OOD split changes physical inputs and shortcut relationships. We compare dense, random, intervention-derived, and generator-oracle masks using the same predictor architecture, data, and loss. Random masks match the oracle’s edge count. Inputs and outputs are standardized using training statistics, and RMSE is computed in those standardized output coordinates over ten seeds.

The robot experiment uses CausalWorld pushing and picking tasks [26]. Nine movable-link masses vary, while three base masses remain fixed, giving 12 mass inputs. Each varying mass is sampled within its simulator space A bounds $( \ell _ { r } , h _ { r } )$ as

$$
m _ { r } = \ell _ { r } + u _ { r } ( h _ { r } - \ell _ { r } ) , \qquad u _ { r } \sim U ( 0 , 0 . 4 ) \mathrm { f o r ~ t r a i n i n g } , \quad u _ { r } \sim U ( 0 . 6 , 1 ) \mathrm { f o r ~ O O D ~ t e s t i n g } .\tag{12}
$$

The mean test-to-training mass ratio is approximately 1.85. Both domains are inside platform space A, with a gap between the light and heavy training/test subranges. Per task–seed run, 1,200 light-mass episodes are split into 864 training, 96 validation, and 240 in-distribution (ID) test episodes; 300 separate heavy-mass episodes form the OOD test. Each episode has 150 steps at 250 Hz, and action targets change every 50 steps.

The 31-dimensional state contains nine joint positions, nine velocities, and the object’s position, quaternion, linear velocity, and angular velocity. Other context consists of object mass, three size coordinates, and object, floor, and stage friction. Network inputs are standardized using training statistics; the PD rule uses joint positions and velocities in physical units. Finger and coeficient networks have two SiLU hidden layers of width 128; the object MLP has three hidden layers of width 256. Acceleration limits use ten times the training 0.999 quantile of absolute velocity increments divided by $\Delta t ;$ velocity limits use three times the maximum absolute training velocity. Table 1 gives the robot training settings.

Table 1: Robot training settings. Actual-mass and fixed-mass-input PD models use the same candidate grid and select their settings separately on light-mass validation data.
<table><tr><td>Setting</td><td>One-step training</td><td>MLP / learned torque roll- Known-PD rollout out</td><td></td></tr><tr><td>Optimizer / batch</td><td>Adam / 256</td><td>Adam / 128</td><td>Adam / 128</td></tr><tr><td>Learning rate</td><td> $1 0 ^ { - 3 }$ </td><td> $3 \times 1 0 ^ { - 4 }$ </td><td> $\{ 1 0 ^ { - 4 } , 3 \times 1 0 ^ { - 5 } \}$ </td></tr><tr><td>Horizon</td><td>1</td><td>50</td><td>10, then 50</td></tr><tr><td>Training budget</td><td>80 epochs</td><td>1,500 updates</td><td>1,500 updates per stage</td></tr><tr><td>Validation interval / patience</td><td>Every epoch / 10</td><td>Every 100 updates  $/ ~ 5$ </td><td>Every 25 updates  $/ 8$ </td></tr></table>

One-step learning rates are halved after three validation plateaus; the gradient-norm limit is 1.0. Rollout validation uses 512 windows. Within the PD family, actual-mass and fixed-mass-input counterparts begin one-step training from identical weights and matched minibatch order; each final model is selected by its own validation loss. The configurations in Table 1 define the mechanism comparisons.

For each rollout step k, joint coordinates are scaled by their training standard deviations. With $z _ { n , k , j }$ denoting scaled coordinate j in test episode n and $\bar { z } _ { k , j }$ its mean over test episodes, we compute

$$
R _ { k } ^ { 2 } = 1 - \frac { \sum _ { n , j } ( \hat { z } _ { n , k , j } - z _ { n , k , j } ) ^ { 2 } } { \sum _ { n , j } ( z _ { n , k , j } - \bar { z } _ { k , j } ) ^ { 2 } } , \qquad \overline { { R ^ { 2 } } } = \frac { 1 } { H } \sum _ { k = 1 } ^ { H } R _ { k } ^ { 2 } .\tag{13}
$$

At each step, $R _ { k } ^ { 2 } = 1$ denotes exact joint-state prediction, $R _ { k } ^ { 2 } = 0$ matches the per-step test-mean baseline, and a negative value indicates greater squared error than that baseline. The 18 coordinates include joint positions and velocities. The mass-information ablation uses $H = 1 5 0$ from the episode’s initial state; the RWM comparison uses a common $H = 1 0 0$ forecast after an observed history. We also report joint normalized RMSE (NRMSE) and object-position RMSE:

$$
\mathrm { N R M S E } _ { \mathrm { j o i n t } } = \sqrt { \frac { 1 } { 1 8 E H } \sum _ { n = 1 } ^ { E } \sum _ { k = 1 } ^ { H } \sum _ { j = 1 } ^ { 1 8 } \left( \frac { \hat { s } _ { n , k , j } - s _ { n , k , j } } { \sigma _ { j } } \right) ^ { 2 } } , \qquad \mathrm { R M S E } _ { \mathrm { o b j } } = \sqrt { \frac { 1 } { 3 E H } \sum _ { n = 1 } ^ { E } \sum _ { k = 1 } ^ { H } \| \hat { p } _ { n , k } - p _ { n , k } \| _ { 2 } ^ { 2 } s _ { n , k , j } } .\tag{14}
$$

where E is the number of test episodes, $\sigma _ { j }$ is a training-set joint-state standard deviation, and $p$ is the object’s three-dimensional position. Joint NRMSE expresses joint-state error in training-standard-deviation units, making position and velocity errors comparable on a dimensionless scale. Object RMSE measures the root-mean-square error over the three position coordinates and is converted from meters to centimeters in the comparison table. Higher $\overline { { R ^ { 2 } } }$ and lower errors indicate better predictions.

We report medians of run-level metrics over two tasks and five seeds. Mass benefit is computed within each run using Eq. (2), then summarized by $\Delta _ { \mathrm { m a s s } } = \mathrm { m e d i a n } _ { b } \Delta _ { r , b }$ and the number of positive diferences. A gain against a named baseline uses the same paired aggregation. A median paired gain can difer from the diference of the two model-score medians. These gains are changes in $R ^ { 2 }$ units, not percentage improvements. Counts are descriptive: task runs within a seed share mass and action sampling streams. Where we summarize the direction of paired diferences, we also report an exact two-sided sign test over the ten runs (10/10 positive diferences give p ≈ 0.002; 8/10 give p ≈ 0.11); because runs within a seed are not fully independent, these p-values are indicative only. Absolute $R ^ { 2 }$ values are normalized by the variance of each test split, so comparisons between the ID and OOD splits rely on paired within-split diferences rather than on absolute scores. The synthetic standardized RMSE and the environment-transfer skill score use their own targets and definitions, specified with their respective results.

## 5.2 Prediction with fixed dependency masks

Table 2 shows that the intervention-derived mask reduces mean OOD RMSE from 1.769 to 0.697, a 60.6% reduction, with improvement in all ten seeds. The oracle-mask error is 0.694. The random sparse mask has an error of 1.604, indicating that the selected dependency pattern matters beyond applying a generic sparse mask. These comparisons establish the value of the frozen dependency constraint for prediction in this synthetic setting.

Table 2: Synthetic OOD prediction. Values are ten-seed means in training-standardized output coordinates. Lower is better.
<table><tr><td>Input mask</td><td>OOD RMSE</td></tr><tr><td>Dense</td><td>1.769</td></tr><tr><td>Random sparse</td><td>1.604</td></tr><tr><td>Intervention-derived, frozen</td><td>0.697</td></tr><tr><td>Generator oracle reference</td><td>0.694</td></tr></table>

The predictor uses a 16-dimensional scalar encoding, 16-dimensional edge messages, encoder hidden width 64, and output-head hidden width 128. All mask variants use squared prediction error and the same efect-normalized paired-response loss. Adam uses learning rate $3 \times 1 0 ^ { - 3 }$ , batch size 2,048, weight decay $1 0 ^ { - 6 }$ , and at most 300 epochs with validation patience 40. The paired-loss weight increases from 0.25 to 1 after 20 epochs. Validation combines observational prediction and paired-response scores. This holds the learning procedure fixed while comparing the available input dependencies.

## 5.3 Mass-information ablation across mechanism configurations

Table 3 ablates mass information within each mechanism family under the heavy-mass shift. The known-PD model uses the actual mass to improve its median score from 0.609 to 0.727. The controls retain the same varying-mass trajectories and remove or fix the mass supplied to the predictor, so the comparison measures the value of that input within each family. The MLP’s separately trained no-mass control obtains 0.721, compared with $\overline { { R ^ { 2 } } } = - 0 . 8 0 3$ with actual mass. Replacing test masses with training means in the fitted actual-mass MLP gives 0.717. This diagnostic shows why input availability and useful parameter dependence are evaluated separately.

Table 3: Mass-information ablation under link-mass extrapolation. Values are medians of 150-step mean joint $R ^ { 2 }$ over two tasks and five seeds. Within each row, the actual-mass model is compared with a separately trained control of the same mechanism family. Higher is better.
<table><tr><td>Mechanism</td><td>Actual mass</td><td>Ablated mass</td><td>Ablation definition</td></tr><tr><td>MLP</td><td>-0.803</td><td>0.721</td><td>Mass inputs removed</td></tr><tr><td>Mass-affine, learned torque</td><td>0.734</td><td>0.733</td><td>Mass inputs fixed at training means</td></tr><tr><td>Mass-affine, known PD</td><td>0.727</td><td>0.609</td><td>Mass inputs fixed at training means</td></tr></table>

All controls use the same varying-mass simulator data as their actual-mass counterparts; only access to mass information is changed. The structured controls use training-mean masses in training and testing; the MLP control removes mass inputs. Validation selects settings within each family as specified in Table 1. The inference-only MLP substitution score, 0.717, is a separate diagnostic.

The mass-afine known-PD model obtains a median $\overline { { R ^ { 2 } } }$ of 0.727, compared with 0.609 for its fixed-mass-input counterpart. Actual mass improves $\overline { { R ^ { 2 } } }$ in all ten task–seed runs, with a mean within-run gain of 0.120 (median 0.121; exact two-sided sign test, $p \approx 0 . 0 0 2 )$ . The learned-torque model achieves medians of 0.734 and 0.733 for actual and fixed mass, respectively, with positive within-run mass benefit in three of ten runs. The known-PD configuration consistently converts supplied out-of-range mass into predictive benefit under this protocol. Its fixed-mass-input control is, however, markedly weaker than the learned-torque control (0.609 versus 0.733). One plausible explanation is that, with torque fixed by the known controller, the inertia and bias terms are the only channels through which the model can represent heavier links, so the PD mechanism relies on mass information more than the learned-torque model does. The mass benefit should therefore be read together with the absolute scores. Comparisons between mechanism families concern complete configurations, including their scaling and training settings; the matched controls quantify the benefit of mass information within each family.

## 5.4 Mass benefit across domains and rollout horizons

Figure 2 shows the known-PD mass ablation over the full prediction horizon. Actual mass has a higher median score than fixed mass at every reported step. At step 150, the medians are 0.518 and 0.439, respectively. The positive benefit therefore extends to the end of the evaluated rollout.

The benefit is larger when the parameter moves outside the training range. Taking the actual-minus-control diference within each run and then its median gives 0.121 on heavy-mass episodes and 0.009 on ID episodes. The OOD benefit exceeds the ID benefit in every task–seed run. This connects the result to the intended application: when the robot’s physical parameters change beyond the observed range, the structured mechanism makes the known parameter useful for trajectory prediction.

(a) Known-PD mass ablation  
![](images/1e0bed8dac10c70f4d5ee5b61b3bbee5d1232a1eeae34237b8deb3285e41c6ab.jpg)

(b) Mass-information benefit  
![](images/10ac3f47f84a8f1b28463ec6088d8d339a87f5d467955bd47a5ef4dc5550fc24.jpg)  
Figure 2: Predictive value of actual mass within the known-PD mechanism. (a) Points are medians of per-step scores across the ten task–seed runs; lines connect recorded evaluation steps. (b) Bars show medians of within-run diferences in 150-step mean joint $R ^ { 2 }$ . These paired diferences are aggregated separately from the model-score medians in Table 3.

## 5.5 Comparison with recurrent robotic world models

We compare the structured mechanisms with Robotic World Model (RWM) [1], a recurrent predictor trained with autoregressive state feedback. We adapt its oficial dynamics implementation to the same CausalWorld state, action, and context variables. RWM uses a two-layer GRU with 256 hidden units and Gaussian prediction heads. We evaluate training forecast lengths of 8 and 50 steps, each with actual-mass and no-mass inputs. This evaluates recurrent prediction alongside explicit physical parameterization; Appendix B.2 gives the adaptation and training settings.

Every model predicts the same states $s _ { 3 3 } , \ldots , s _ { 1 3 2 }$ from the observed state s<sub>32</sub> and recorded actions $a _ { 3 2 } , \dots , a _ { 1 3 1 }$ . RWM additionally receives the preceding observed history. The forecast contains 100 steps at 250 Hz (0.4 s), with predicted states fed back throughout. All models use the same episode splits and training-derived normalization. Existing structured-model checkpoints are evaluated on this shared window, and RWM checkpoints are selected using ID validation. Table 4 reports all evaluated configurations. Comparisons across families evaluate complete configurations with their respective history, controller knowledge, scaling, and training settings; within-family controls quantify mass-information benefit.

The learned-torque mass-afine model matches or slightly improves all three reported median prediction metrics relative to both no-mass RWM configurations, by margins of 0.003–0.016 in median $\overline { { R ^ { 2 } } }$ . With actual masses, it achieves $\overline { { R ^ { 2 } } } = 0 . 7 5 5$ , joint NRMSE of 0.449, and object-position RMSE of 2.668 cm, compared with 0.752, 0.457, and 2.842 cm for RWM trained with a 50-step forecast. Its fixed-mass-input variant obtains the best absolute scores in Table 4: 0.768, 0.440, and 2.650 cm. Given these small margins, we interpret the learned-torque configuration as competitive with the strongest recurrent baseline, while using no observation history; its accuracy does not depend on access to the actual masses.

The known-PD mechanism converts out-of-range mass into a positive predictive gain in all ten task–seed runs: median $\overline { { R ^ { 2 } } }$ rises from 0.677 with fixed mass inputs to 0.746 with actual masses, and the median paired gain is 0.071. Relative to RWM without mass inputs trained with an eight-step forecast, the actual-mass known-PD model improves $\overline { { R ^ { 2 } } }$ by a median paired 0.027, with a positive diference in eight of ten runs. Its median joint NRMSE is 0.461, compared with 0.478 for that RWM configuration. Extending RWM’s training forecast to 50 steps raises its no-mass median $\overline { { R ^ { 2 } } }$ to 0.752. The paired mass comparisons identify the distinctive result of the known-PD design: actual masses improve prediction consistently, while supplying them to the recurrent predictors or the learned-torque model does not yield a positive median mass benefit in this benchmark.

Table 4: CausalWorld link-mass extrapolation: a shared 100-step forecast after 32 observed history steps. Entries are medians over ten task–seed runs. $N _ { \mathrm { t r } }$ denotes RWM’s training forecast length. The final column is the median paired gain from actual mass relative to the control immediately above it. Bold marks the best value in each metric column.
<table><tr><td>Model</td><td>Mass input</td><td> $\overline { { R ^ { 2 } } } \uparrow$ </td><td>Joint NRMSE↓</td><td>Object RMSE (cm)↓</td><td> $\Delta _ { \mathrm { m a s s } }$  ↑</td></tr><tr><td>Hold-state reference</td><td>None</td><td>-0.320</td><td>1.055</td><td>3.835</td><td></td></tr><tr><td rowspan="2">RWM  $( N _ { \mathrm { t r } } = 8 )$ </td><td>None</td><td>0.718</td><td>0.478</td><td>2.913</td><td></td></tr><tr><td>Actual</td><td>-0.648</td><td>1.182</td><td>7.311</td><td>↓1.367</td></tr><tr><td rowspan="2">RWM  $( N _ { \mathrm { t r } } = 5 0 )$ </td><td>None</td><td>0.752</td><td>0.457</td><td>2.842</td><td></td></tr><tr><td>Actual</td><td>-0.540</td><td>1.136</td><td>5.744</td><td>↓1.293</td></tr><tr><td rowspan="2">Mass-affine, learned torque (A)</td><td>Mean</td><td>0.768</td><td>0.440</td><td>2.650</td><td></td></tr><tr><td>Actual</td><td>0.755</td><td>0.449</td><td>2.668</td><td>↓0.010</td></tr><tr><td rowspan="2">Mass-affine, known PD (B)</td><td>Mean</td><td>0.677</td><td>0.520</td><td>2.830</td><td></td></tr><tr><td>Actual</td><td>0.746</td><td>0.461</td><td>2.816</td><td>↑0.071</td></tr></table>

All models share test episodes, forecast origins, future actions, and scoring. RWM additionally uses the observed history; B uses the known controller. RWM “None” zeros the standardized robot-mass inputs; “Mean” fixes masses during training and evaluation. Positive mass gains occur in 0/10, 0/10, 2/10, and 10/10 runs for RWM (8), RWM (50), A, and ${ \bar { \mathrm { B } } } ,$ respectively. Arrows show the direction and magnitude of paired mass efects, not changes between the displayed medians. The 150-step ablation in Table 3 starts from the episode’s initial state.

## 5.6 Environment-transfer comparison with a probabilistic ensemble baseline

To test dependency constraints with an established dynamics-model family, we compare an interventionmasked predictor with a PETS-style probabilistic ensemble [25] on CausalWorld pushing and picking. This complementary benchmark trains on simulator space A and tests on space B. Each task–seed run uses 8,000 source examples and 4,000 target examples, with a 20% source holdout for ID testing and an independent source intervention pool for mask estimation. Each prediction maps an initial physical description and a nine-dimensional action to the simulator observation after holding that action for 50 environment steps. Data collection retains examples in which the object moves at least 5 mm; action pools are constructed separately in each domain using the same movement criterion. Appendix B specifies the predictor settings and score.

Table 5 reports mean target-domain prediction skill across ten seeds per task. Restricting the ensemble’s inputs using the intervention-derived graph raises mean skill from 0.1407 to 0.5335. The predictor with output-specific masks obtains 0.6122. This comparison shows that useful dependency selection extends to a probabilistic ensemble, alongside the masked predictor. The ensemble uses PETS-style bootstrapped Gaussian dynamics networks, and the comparison evaluates their predictive dynamics. The score and 50-step transition protocol difer from the joint-rollout mass ablation in Table 3.

Table 5: External dynamics-baseline comparison on CausalWorld space A→B. Values are mean prediction skill relative to the source-training mean, S (higher is better), over ten seeds per task. All rows share the same data, targets, and scoring procedure. This is a complementary environment-transfer benchmark.
<table><tr><td>Predictor</td><td>Pushing</td><td>Picking</td><td>Overall</td></tr><tr><td>PETS-style probabilistic ensemble</td><td>0.2321</td><td>0.0492</td><td>0.1407</td></tr><tr><td>PETS-style ensemble + graph input selection</td><td>0.6831</td><td>0.3839</td><td>0.5335</td></tr><tr><td>Intervention-masked predictor</td><td>0.7508</td><td>0.4737</td><td>0.6122</td></tr></table>

## 6 Conclusions and Future Work

We developed a robot world-model design combining mass-afine dynamics, alternative torque mechanisms, and learned residuals under fixed finger-local dependencies. Frozen intervention-derived masks improve synthetic OOD prediction. In the shared CausalWorld benchmark, the learned-torque configuration is competitive with the strongest tested RWM configuration, with or without mass inputs. Supplying out-ofrange masses sharply degrades the MLP and recurrent predictors, whereas the known-PD configuration makes actual mass information consistently useful: median 150-step mean joint R<sup>2</sup> increases from 0.609 to 0.727 relative to its fixed-mass-input control, and positive mass benefit is observed in all ten runs of both the 150-step ablation and the 100-step benchmark.

The present evaluation assumes available link masses and a known low-level controller for the PD variant, and covers forecasts of 0.4–0.6 s. The robot dependency structure is specified by finger-local routing rather than learned, and the evaluation uses two tasks with five seeds each. Longer deployments, uncertain parameter estimates, and downstream control remain to be evaluated. Future work will examine intervention-based learning objectives and energy-consistent relationships between inertia and bias, with matched torque-model comparisons to isolate how controller knowledge afects mass use. These results show that explicit mechanism design can turn out-of-range physical parameters into useful information for embodied prediction.

## References

[1] C. Li, A. Krause, and M. Hutter. Robotic World Model: A Neural Network Simulator for Robust Policy Optimization in Robotics. arXiv preprint, arXiv:2501.10100, 2025. https://arxiv.org/abs/2501.10100.

[2] J. Richens, T. Everitt, and D. Abel. General agents need world models. ICML, PMLR 267:51659–51687, 2025. https://proceedings.mlr.press/v267/richens25a.html.

[3] J. Pearl. Causal inference in statistics: An overview. Statistics Surveys, 3:96–146, 2009. https://ftp.cs.ucla. edu/pub/stat\_ser/r350.pdf.

[4] Z. Wang, X. Xiao, Z. Xu, Y. Zhu, and P. Stone. Causal Dynamics Learning for Task-Independent State Abstraction. ICML, PMLR 162:23151–23180, 2022. https://proceedings.mlr.press/v162/wang22ae.html.

[5] A. Murillo-González, J. Xu, and L. Liu. Learning Causal Structure Distributions for Robust Planning. IEEE Robotics and Automation Letters, 2025. https://arxiv.org/abs/2508.06742.

[6] L. Schulze, J. Peters, and O. Arenz. Context-Aware Deep Lagrangian Networks for Model Predictive Control. IROS, 2025. https://arxiv.org/abs/2506.15249.

[7] Y. Wang, P. Priye, L. Wei, and H. Ling. Stable and Counterfactually Robust Physical World Models from Imposed Structure and Learned Physics. arXiv preprint, arXiv:2610.00280, 2026. https://arxiv.org/abs/2610.00280.

[8] J. Richens and T. Everitt. Robust agents learn causal world models. ICLR, 2024. https://arxiv.org/abs/24 02.10877.

[9] M. Ceriscioli and K. Mohan. Agents Robust to Distribution Shifts Learn Causal World Models Even Under Mediation. NeurIPS, 38, 2025. https://papers.neurips.cc/paper\_files/paper/2025/hash/6fe10a4c0d6806 09f0560920bd9ade4a-Abstract-Conference.html.

[10] I. Hwang, Y. Kwak, S. Choi, B.-T. Zhang, and S. Lee. Fine-Grained Causal Dynamics Learning with Quantization for Improving Robustness in Reinforcement Learning. ICML, PMLR 235:20842–20870, 2024. https://proceedi ngs.mlr.press/v235/hwang24b.html.

[11] S. Pitis, E. Creager, and A. Garg. Counterfactual Data Augmentation using Locally Factored Dynamics. NeurIPS, 33, 2020. https://arxiv.org/abs/2007.02863.

[12] H. Lin, W. Ding, J. Chen, L. Shi, J. Zhu, B. Li, and D. Zhao. BECAUSE: Bilinear Causal Representation for Generalizable Ofline Model-based Reinforcement Learning. NeurIPS, 37, 2024. https://arxiv.org/abs/2407 .10967.

[13] X. Wang and B. Huang. Modeling Unseen Environments with Language-guided Composable Causal Components in Reinforcement Learning. ICLR, 2025. https://proceedings.iclr.cc/paper\_files/paper/2025/hash/79d8 6433c2acd12b6fa98553435d226e-Abstract-Conference.html.

[14] H. Nam, Q. Le Lidec, L. Maes, Y. LeCun, and R. Balestriero. Causal-JEPA: Learning World Models through Object-Level Latent Masking. ICML, 2026. https://arxiv.org/abs/2602.11389.

[15] M. Lutter, C. Ritter, and J. Peters. Deep Lagrangian Networks: Using Physics as Model Prior for Deep Learning. ICLR, 2019. https://arxiv.org/abs/1907.04490.

[16] M. Lutter and J. Peters. Combining physics and deep learning to learn continuous-time dynamics models. The International Journal of Robotics Research, 42(3), 2023. https://doi.org/10.1177/02783649231169492.

[17] C. G. Atkeson, C. H. An, and J. M. Hollerbach. Estimation of Inertial Parameters of Manipulator Loads and Links. The International Journal of Robotics Research, 5(3):101–119, 1986. https://doi.org/10.1177/027836 498600500306.

[18] J. K. Gupta, K. Menda, Z. Manchester, and M. Kochenderfer. Structured Mechanical Models for Robot Learning and Control. L4DC, PMLR 120:328–337, 2020. https://proceedings.mlr.press/v120/gupta20a.html.

[19] Y. Yin, V. Le Guen, J. Dona, E. de Bézenac, I. Ayed, N. Thome, and P. Gallinari. Augmenting Physical Models with Deep Networks for Complex Dynamics Forecasting. ICLR, 2021. https://arxiv.org/abs/2010.04456.

[20] L. Schulze, J. D. Negri, V. Barasuol, V. S. Medeiros, M. Becker, J. Peters, and O. Arenz. Floating-Base Deep Lagrangian Networks. arXiv preprint, arXiv:2510.17270, 2025. https://arxiv.org/abs/2510.17270.

[21] K. Xu, M. Zhang, J. Li, S. S. Du, K.-I. Kawarabayashi, and S. Jegelka. How Neural Networks Extrapolate: From Feedforward to Graph Neural Networks. ICLR, 2021. https://arxiv.org/abs/2009.11848.

[22] W. Yu, J. Tan, C. K. Liu, and G. Turk. Preparing for the Unknown: Learning a Universal Policy with Online System Identification. RSS, 2017. https://arxiv.org/abs/1702.02453.

[23] K. Lee, Y. Seo, S. Lee, H. Lee, and J. Shin. Context-aware Dynamics Model for Generalization in Model-Based Reinforcement Learning. ICML, PMLR 119:5757–5766, 2020. https://proceedings.mlr.press/v119/lee20g. html.

[24] A. Kumar, Z. Fu, D. Pathak, and J. Malik. RMA: Rapid Motor Adaptation for Legged Robots. RSS, 2021. https://arxiv.org/abs/2107.04034.

[25] K. Chua, R. Calandra, R. McAllister, and S. Levine. Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models. NeurIPS, 2018. https://arxiv.org/abs/1805.12114.

[26] O. Ahmed, F. Träuble, A. Goyal, A. Neitz, Y. Bengio, B. Schölkopf, M. Wüthrich, and S. Bauer. CausalWorld: A Robotic Manipulation Benchmark for Causal Structure and Transfer Learning. ICLR, 2021. https://arxiv. org/abs/2010.04296.

[27] CausalWorld contributors. TriFingerRobot: PD control and safety torque implementation. CausalWorld source documentation, version 1.2.0. Accessed October 7, 2026. https://causal-world.readthedocs.io/en/latest/ \_modules/causal\_world/envs/robot/trifinger.html.

[28] C. Li and RWM contributors. Oficial Robotic World Model implementation, rsl\_rl\_rwm. Commit 18eebcdd71 45284c8d5eed5d8ed1a4b96c649693. https://github.com/leggedrobotics/rsl\_rl\_rwm/tree/18eebcdd7145 284c8d5eed5d8ed1a4b96c649693.

## Appendix A Component Sources and Design Choices

The design uses established dependency selection, positive-definite matrix parameterization, afine inertial dependence, hybrid modeling, and PD control. Our configuration combines these components through finger-local routing, explicit mass entry in inertia and bias, mass-excluded torque and residual-network inputs, and the known simulator controller. The mass-information controls assess this configuration’s use of the supplied physical parameter. Table 6 separates component sources from the design and evaluation choices in this study.

Table 6: Sources of established components and this study’s specific design and evaluation choices.
<table><tr><td>Component</td><td>Established foundation</td><td>Work developed in this study</td></tr><tr><td>Input dependencies</td><td>Causal dynamics and dependency selection [4, 10]</td><td>Frozen intervention masks in the synthetic benchmark; shared finger-local constraints in robot models</td></tr><tr><td>Dynamics mechanisms</td><td>Positive-definite inertia parameterization and inertial identification [15, 17, 18]</td><td>Mass-affine inertia and state-dependent bias; torque and contact-related residual networks receive no direct mass input.</td></tr><tr><td>Known control</td><td>Hybrid modeling and simulator PD control [18, 19, 26, 27]</td><td>A known-PD mechanism in physical units, integrated with learned mass-affine terms and an object-state</td></tr><tr><td>Evaluation</td><td>Controlled simulation and input ablation</td><td>predictor. Disjoint link-mass ranges, separately trained mass- information controls, and comparison of ID and OOD mass benefits.</td></tr></table>

## Appendix B Complementary Benchmark Details

## B.1 Environment-transfer predictors and metric

The environment-transfer data record simulator state variables and physical descriptors together with nine action coordinates. Constant source features are removed, and remaining inputs and output observations are standardized using source-training statistics. The two tasks use ten seeds each. Per run, 20% of the 8,000 source examples form an ID test set; the remaining examples are split 85:15 for model fitting and validation. A separate pool of 4,000 paired source interventions estimates dependencies. The efect threshold is 0.03 times the largest measured efect, and action dependencies are kept open because the paired pool intervenes on state variables rather than actions. No target prediction scores are used in model validation.

The PETS-style baseline has five bootstrapped probabilistic MLPs, each with two hidden layers of width 200 and SiLU activations. Each network predicts a mean and diagonal log variance, is trained with Gaussian negative log likelihood, and uses Adam with learning rate $1 0 ^ { - 3 }$ , batch size 512, at most 300 epochs, and validation patience 30. Point predictions average the five network means. Its graph variant retains an input if the estimated mask connects it to at least one output; the masked predictor instead uses output-specific dependencies. All variants use the same source observations and target observations. Graph variants additionally use the independent source intervention pool, and the comparison evaluates their complete predictive configurations.

The intervention-masked predictor uses an eight-dimensional scalar encoding, 16-dimensional edge messages, and encoder and output-head hidden widths of 64. It is trained with squared prediction error, Adam at learning rate $3 \times 1 0 ^ { - 3 }$ , batch size 512, at most 300 epochs, and validation patience 30. The reported 0.6122 configuration does not add a paired-response loss. The paired data are used to estimate its mask, and the mask is held fixed during predictor training.

For standardized target observation ${ \tilde { y } } _ { n }$ and prediction $\hat { \tilde { y } } _ { n } ,$ the reported score is

$$
S = 1 - \frac { \sum _ { n } \| \hat { \tilde { y } } _ { n } - \tilde { y } _ { n } \| _ { 2 } ^ { 2 } } { \sum _ { n } \| \tilde { y } _ { n } - \bar { \tilde { y } } _ { \mathrm { t r a i n } } \| _ { 2 } ^ { 2 } } ,\tag{15}
$$

where $\bar { \tilde { y } } _ { \mathrm { t r a i n } }$ is the source-training mean. This measures prediction skill against the source-training constant predictor; Eq. (13) uses the per-step test mean for joint $R ^ { 2 }$ . Task means and the overall mean aggregate run-level scores. The three benchmark rows share this definition and target set.

## B.2 RWM implementation and shared-window evaluation

The comparison adapts the oficial RWM state-dynamics module [1, 28] at commit 18eebcdd7145284c8d5e ed5d8ed1a4b96c649693. Its two-layer, 256-unit GRU feeds separate mean and log-standard-deviation heads, each with a 128-unit ReLU hidden layer. The adapter uses the oficial sampled autoregressive regression and variance-bound losses. The input has 59 coordinates: 31 states, nine actions, 12 robot masses, and seven object/environment context values. No-mass variants zero all 12 standardized mass slots during training and evaluation, retaining the input dimension and initialization of their actual-mass counterparts. This benchmark evaluates the state predictor; policy optimization and auxiliary reward or termination heads are outside its prediction task.

RWM receives 32 observed state–action pairs and trains to forecast either 8 or 50 future states. Adam uses learning rate $1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 5 }$ , batch size 1,024, a gradient-norm limit of 1.0, and variance-bound weight 1.0. Each configuration has a budget of 50,000 gradient updates, with ID validation every 250 updates. Early stopping uses 20 validation checks without improvement after at least 2,000 updates. Selection minimizes joint NRMSE over the common 100-step ID validation forecast. Both training horizons use this selection protocol. The four structured configurations use their existing validation-selected checkpoints from Table 1.

For evaluation, RWM is initialized with $( s _ { 1 } , a _ { 1 } ) , \dotsc , ( s _ { 3 2 } , a _ { 3 2 } )$ , producing ${ \hat { s } } _ { 3 3 } ;$ later steps feed predicted states and the recorded next action back into the model. Structured models start directly from $s _ { 3 2 }$ . All models predict $s _ { 3 3 }$ through $s _ { 1 3 2 }$ without future observed states. RWM feeds back its predicted mean at evaluation, and the shared evaluation applies quaternion normalization. The hold-state reference repeats s at every forecast step. All 300 OOD episodes and 240 ID episodes are evaluated per task–seed run. Non-finite predictions count as failures. Table $7$ separates the task medians and reports ID accuracy under the same window.

Table 7: Additional shared-window scores. ID columns aggregate ten task–seed runs; each OOD task column aggregates its five seeds. All scores are median 100-step mean joint $R ^ { 2 }$ (higher is better).
<table><tr><td>Model</td><td>Mass input</td><td>ID</td><td>OOD pushing</td><td>OOD picking</td></tr><tr><td>Hold-state reference</td><td>None</td><td>-0.332</td><td>-0.327</td><td>-0.316</td></tr><tr><td>RWM (8)</td><td>None</td><td>0.642</td><td>0.715</td><td>0.719</td></tr><tr><td>RWM (8) RWM (50)</td><td>Actual</td><td>0.626</td><td>-0.850</td><td>-0.447</td></tr><tr><td>RWM (50)</td><td>None Actual</td><td>0.635 0.622</td><td>0.752 -0.658</td><td>0.751 -0.421</td></tr><tr><td>Mass-affine, learned torque (A)</td><td>Mean</td><td>0.701</td><td>0.764</td><td>0.774</td></tr><tr><td>Mass-affine, learned torque (A)</td><td>Actual</td><td>0.693</td><td>0.751</td><td>0.756</td></tr><tr><td>Mass-affine, known PD (B)</td><td>Mean</td><td>0.620</td><td>0.661</td><td>0.682</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Mass-affine, known PD (B)</td><td>Actual</td><td>0.635</td><td>0.745</td><td>0.754</td></tr></table>