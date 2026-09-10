# Beyond Conventional Federated Learning via High-Order Regularization

Alireza Kabgani, Masoud Ahookhosh

Department of Mathematics, University of Antwerp, Antwerp, Belgium Email: alireza.kabgani@uantwerp.be, masoud.ahookhosh@uantwerp.be

## Abstract

Federated clients that perform several local optimization steps can return parameter displacements with widely different magnitudes. The quadratic regularization of FedProx grows linearly with displacement and therefore offers limited control over the contrast between ordinary and unusually large client movements. We here introduce HiFedProx, which replaces the quadratic penalty with a scale-matched power-type regularizer indexed by $p \geq 2 .$ . All powers have the same regularization-gradient magnitude at a reference displacement ${ \dot { R } } ,$ while every $p > 2$ gives a weaker response below R and a stronger response above it. An exact affine reference calculation shows that increasing p compresses relative displacement disparities, although very large powers approach fixed-radius behavior and increase local curvature. HiFedProx combines this geometry with finite-budget stochastic client optimization and same-minibatch Armijo backtracking. In paired five-seed experiments on a frozen 60-writer FEMNIST subset, a common-parameter study over $p \in \{ 2 , 3 , 4 , 5 , 6 , 7 , 8 \}$ shows similar clean-training performance but substantial gains under composite stress. The lowest moderate- and severe-stress losses occur at $p = 7$ and $p = 6$ , improving over $p = 2$ by 11.44% and 23.16%, respectively. Although displacement-tail ratios continue to decrease through $p = 8 ,$ predictive performance peaks in an intermediate range and Armijo trial cost increases with $p .$ These results indicate that the exponent should be calibrated rather than maximized. In our experiments, $p = 5 – 7$ provides the most useful range.

## Index Terms

federated learning, HiFedProx, high-order regularization, Armijo backtracking, client displacement

## I. INTRODUCTION

Federated learning (FL) trains a model by solving a distributed optimization problem in which multiple clients collaboratively train a shared model from distributed data without transmitting their raw examples to a central server. We consider a system consisting of one central server and $N \geq 1$ clients, where client i stores a local dataset $\mathcal { D } _ { i } = ( z _ { i j } ) _ { j = 1 } ^ { n _ { i } }$ and minimizes the empirical objective $f _ { i }$ given by

$$
f _ { i } ( w ) = \frac { 1 } { n _ { i } } \sum _ { j = 1 } ^ { n _ { i } } \ell ( w ; z _ { i j } ) .\tag{1}
$$

The learning task is to solve the client-uniform optimization problem

$$
\operatorname* { m i n } _ { w \in \mathbb { R } ^ { d } } F ( w ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } f _ { i } ( w ) ,\tag{2}
$$

where each client contributes equally to the global objective regardless of its local sample size.

A widely used approach is Federated Averaging (FedAvg), in which the server broadcasts the current model to a subset of clients, each client performs several local stochastic-gradient updates, and the resulting local models are averaged to obtain the next global iterate [1]. While multiple local updates substantially reduce communication, they also amplify the effects of statistical heterogeneity and optimization variability, often leading to highly unequal client displacements. In particular, some clients may consistently move much farther from the global model because of heterogeneous data distributions, aggressive loca optimization, or larger local workloads. This imbalance motivates optimization methods that explicitly account for disparities in local update magnitudes while preserving the communication efficiency of federated learning.

To moderate excessive client displacements, FedProx augments each client’s local optimization problem with a quadratic proximal term [2], i.e.,

$$
\operatorname* { m i n } _ { u \in \mathbb { R } ^ { d } } \ Q _ { i } ( u ; w ^ { t } ) : = f _ { i } ( u ) + \frac { \mu } { 2 } \| u - w ^ { t } \| ^ { 2 } ,\tag{3}
$$

where $w ^ { t }$ denotes the current global model. The quadratic regularization penalizes deviations from $w ^ { t } .$ , and increasing the parameter $\mu$ uniformly strengthens this penalty for all client displacements. We instead ask whether the regularizer can remain relatively mild for typical client displacements while responding more aggressively to unusually large ones.

Motivated by the need for a more flexible regularization of client displacements, we generalize the quadratic proximal term to a family of high-order regularizers inspired by recent advances in high-order proximal-point methods [3]–[10]. Specifically, we consider the local optimization models

$$
\operatorname* { m i n } _ { u \in \mathbb { R } ^ { d } } \quad Q _ { i , p } ( u ; w ^ { t } ) : = f _ { i } ( u ) + \frac { \mu } { p R ^ { p - 2 } } \| u - w ^ { t } \| ^ { p } ,\tag{4}
$$

where $p \geq 2$ and $\mu > 0$ controls the regularization strength, $R > 0$ is a reference displacement scale, and $p$ provides an additional degree of flexibility. The normalization is chosen so that the magnitude of the regularization gradient equals $\mu R$ whenever $\| u - w ^ { t } \| = R ,$ independently of $p .$ Consequently, compared with the scale-matched quadratic regularizer $( \mathrm { i } . \mathrm { e } . , p = 2 )$ every choice $p > 2$ yields a milder regularization response for displacements below R and a stronger response for displacements above R. We restrict attention to $p \geq 2$ because the subquadratic regime $1 < p < 2$ reverses this behavior, i.e., it penalizes small displacements more strongly and large displacements more weakly than the quadratic model.

We incorporate the family (4) into a finite-budget stochastic client solver with same-minibatch Armijo backtracking and call the resulting method HiFedProx. All choices of $p \geq 2$ use the same client initialization, sampling rule, local optimization procedure, server aggregation, and stochastic-gradient budget and only the radial regularization term changes. The method remains first order and requires neither Hessians nor higher-order derivatives. The case $p = 2$ gives the safeguarded quadratic comparator, while values $p > 2$ produce high-order variants.

The paper makes three main contributions. First, we introduce a scale-matched family of high-order proximal regularizer for federated learning and show how the parameters $\mu , R ,$ and $p$ separately control the strength, crossover scale, and shape of the regularization response. Second, we develop HiFedProx, a complete first-order federated procedure with finite loca budgets and capped same-minibatch Armijo backtracking. We also derive an exact affine reference response and a smooth fixed-batch safeguard result that clarify the radial mechanism and the role of backtracking. $T h i r d ,$ we conduct a paired FEMNIST exponent-sensitivity study over $p \in \{ 2 , 3 , 4 , 5 , 6 , 7 , 8 \}$ with matched gradient and communication budgets. The results identify a useful intermediate range, around $p = 5 – 7$ in the present protocol: larger powers continue to compress displacement disparities, but predictive performance saturates and then deteriorates while line-search cost increases.

The remainder of this paper is organized as follows. In Section II, we discuss the related works. In Section III, we introduce scale-matched regularization and HiFedProx. In Section IV, the geometry and local safeguard are discussed. Section V presents the experimental design, Section VI reports the FEMNIST results, and Section VII discusses the choice of $p$ and the limitations. Section VIII concludes the paper.

## II. RELATED WORK

FedProx is the closest related method, as it augments each client’s local objective with a quadratic proximal regularizer while maintaining a single shared global model [2]. Similarly, pFedMe employs quadratic proximal subproblems, but uses them to learn personalized client models rather than a single global model [11]. More recently, adaptive proximal methods have focused on dynamically adjusting the coefficient of a quadratic regularizer while retaining its quadratic form [12]. Our work differs in that it generalizes the quadratic proximal regularizer to a scale-matched family of high-order regularizers. Rather than varying only the regularization coefficient, we introduce the exponent $p$ as an additional design parameter, with the regularizers calibrated to have the same gradient magnitude at a prescribed reference displacement R.

Several federated optimization methods address sources of performance degradation other than the choice of proximal regularizer. SCAFFOLD, FedNova, and FedVARP mitigate client drift, heterogeneous local computation, and variance arising from partial client participation, respectively [13]–[15]. These approaches are complementary to the high-order proximal regularization considered in this work. In a different direction, nonsmooth $\ell _ { 1 }$ -based regularizers promote sparse, coordinatewise client updates [16], whereas our regularization is smooth, isotropic, and based on high-order Euclidean norms.

Federated second-order methods, such as FedNL, exploit Hessian information to construct cubic Taylor models for local optimization [17]. In contrast, our approach modifies the proximal regularization in the client objective while remaining entirely first-order. Furthermore, server-side clipping limits client updates only after local training has been completed, whereas our high-order proximal regularization influences the optimization trajectory throughout local training; see, for example, analyses of clipped FedAvg [18].

## III. SCALE-MATCHED REGULARIZATION AND HIFEDPROX

## A. Federated objective and scale matching

In this section, we consider the client-uniform optimization problem (2), in which each client contributes equally to the global objective regardless of its local sample size. The client sampling and model aggregation procedures follow the same client-uniform convention. The high-order regularizer in (4) is applied only during local training; predictive loss and accuracy are evaluated using the resulting model without the regularization term.

Let us define the power regularizer and the radial map

$$
\phi _ { p , R , \mu } ( d ) = \frac { \mu } { p R ^ { p - 2 } } \left\| d ^ { p } \right\| , \quad J _ { p } ( d ) = \left\{ \left\| d ^ { p - 2 } \right\| d , ~ d \neq 0 , \right.
$$

where its gradient and magnitude are given by

$$
\nabla \phi _ { p , R , \mu } ( d ) = \mu R ^ { 2 - p } J _ { p } ( d ) ,\tag{5}
$$

$$
\| \nabla \phi _ { p , R , \mu } ( d ) \| = \mu R \left( \frac { \| d \| } { R } \right) ^ { p - 1 } .\tag{6}
$$

All powers therefore have regularization-gradient magnitude $\mu R$ at $\| d \| = R .$ . Comparing a general power with the quadratic response gives

$$
\frac { \| \nabla \phi _ { p , R , \mu } ( d ) \| } { \| \nabla \phi _ { 2 , R , \mu } ( d ) \| } = \left( \frac { \| d \| } { R } \right) ^ { p - 2 } .\tag{7}
$$

For $p = 3 ,$ , the magnitude is $\mu \left\| d \right\| ^ { 2 } / R .$ , compared with $\mu \left. d \right.$ for $p = 2$ . Figure 1 shows the normalized responses. The parameter R is the crossover at which they agree, $\mu$ rescales the response, and $p$ controls its shape. Only the gradient magnitude is matched at $R ,$ , and the regularizer values and curvatures remain power dependent. The parameter $R$ is a crossover scale, not a clipping radius or hard constraint.

![](images/46ed072c04347d1f754acd48cfcae89640ccfc33bd33abaaa40634cfe3a7dea4.jpg)  
Fig. 1. Normalized quadratic and pth-order regularization-gradient magnitudes. Both equal one at $\| d \| / R = 1$

## B. Finite-budget client update

Training proceeds in communication rounds. At round t, the server has the current model $w ^ { t }$ and samples a subset $S _ { t } \subseteq \{ 1 , \ldots , N \}$ of $m : = | S _ { t } |$ clients uniformly without replacement. The server sends $w ^ { t }$ to the selected clients. Each selected client starts from $w ^ { t }$ , returns a finite-budget point $u _ { i } ^ { t } .$ , and communicates the displacement $\Delta _ { i } ^ { t } = u _ { i } ^ { t } - w ^ { t }$ . The equal-weight server update is

$$
w ^ { t + 1 } = w ^ { t } + \frac { 1 } { | S _ { t } | } \sum _ { i \in S _ { t } } \Delta _ { i } ^ { t } = \frac { 1 } { | S _ { t } | } \sum _ { i \in S _ { t } } u _ { i } ^ { t } .\tag{8}
$$

For a nonempty minibatch $B \subseteq { \mathcal { D } } _ { i }$ <sub>i</sub>, let

$$
Q _ { i , p } ( u ; w ^ { t } , B ) = f _ { i } ( u ; B ) + \phi _ { p , R , \mu } ( u - w ^ { t } ) .
$$

Let $\mathrm { A D } _ { u } Q _ { i , p } ( u ; w ^ { t } , B )$ denote the direction returned by automatic differentiation for the complete regularized batch model. At differentiable points, this is $\nabla _ { u } Q _ { i , p } ( u ; w ^ { t } , B )$ . Algorithm 1 uses this direction and evaluates the current value and every trial value on the same minibatch. If no trial passes the test, the client keeps its current iterate.

Note that each local iteration uses one backward pass. The current objective value is obtained from the same forward computation, and each Armijo trial adds one further forward evaluation. The backtracking procedure therefore adds neither backward passes nor communication. Moreover, let us emphasize that setting $p = 2$ gives the safeguarded quadratic comparator. Every value $p \geq 2$ uses the same client optimizer, sampling rule, aggregation rule, and gradient budget, and only the radial power changes. No variant uses Hessians or higher-order derivatives.

Algorithm 1 HiFedProx (High-order Fed-Prox)   
Require: $w _ { . } ^ { t } , \mathcal { D } _ { i } , p , \mu , R , K _ { i } ^ { t } , \alpha _ { 0 } , c , \rho , J$   
1: u ← w<sup>t</sup>   
2: for $k = 0 , \ldots , K _ { i } ^ { t } - 1$ do   
3: sample nonempty $B _ { k } \subseteq { \mathcal { D } } _ { i }$   
4: $g \gets \mathrm { A D } _ { u } Q _ { i , p } ( u ; w ^ { t } , B _ { k } )$   
5: $q  Q _ { i , p } ( u ; w ^ { t } , B _ { k } ) ; \alpha  \alpha _ { 0 }$   
6: for $j = 0 , \dots , J$ do   
7: $v  u - \alpha g$   
8: if $Q _ { i , p } ( v ; w ^ { t } , B _ { k } ) \leq q - c \alpha \| g \| ^ { 2 }$ then   
9: $u  v ;$ break   
10: end if   
11: if $j < J$ then   
12: $\alpha \gets \rho \alpha$   
13: end if   
14: end for   
15: end for   
16: return $\boldsymbol { u } - \boldsymbol { w } ^ { t }$

## IV. GEOMETRY AND LOCAL SAFEGUARD

The following calculation isolates the radial effect before curvature, stochastic sampling, and repeated local steps enter. It treats $\boldsymbol { g } \in \mathbb { R } ^ { d }$ as a prescribed vector. When the loss is differentiable, g can be chosen as a client gradient.

Proposition 1 (Affine client-model response). Let $p \geq 2 , \mu , R > 0 ,$ and $g \in \mathbb { R } ^ { d }$ . The problem

$$
\operatorname* { m i n } _ { d } \left\{ \langle g , d \rangle + { \frac { \mu } { p R ^ { p - 2 } } } \left\| d \right\| ^ { p } \right\}\tag{9}
$$

has the unique solution $d _ { p } ( 0 ) = 0$ and, $f o r \ g \neq 0$

$$
d _ { p } ( g ) = - R \left( { \frac { \lVert g \rVert } { \mu R } } \right) ^ { 1 / ( p - 1 ) } { \frac { g } { \lVert g \rVert } } .\tag{10}
$$

For nonzero $g _ { i } , g _ { j }$

$$
{ \frac { \| d _ { p } ( g _ { i } ) \| } { \| d _ { p } ( g _ { j } ) \| } } = \left( { \frac { \| g _ { i } \| } { \| g _ { j } \| } } \right) ^ { 1 / ( p - 1 ) } .\tag{11}
$$

Proof. The objective is coercive and strictly convex. For $g \neq 0 ,$ , the optimality condition $g + \mu R ^ { 2 - p } \left\| d \right\| ^ { p - 2 } d = 0$ shows that d points opposite to $g .$ Taking norms and restoring the direction gives (10) and division gives (11). □

Writing $\theta ( g ) = \left\| g \right\| / ( \mu R )$ gives the dimensionless response

$$
\frac { \| d _ { p } ( g ) \| } { R } = \theta ( g ) ^ { 1 / ( p - 1 ) } .\tag{12}
$$

The value $\theta = 1$ gives the common displacement R. For every $p > 2$ , the high-order displacement is larger than the quadratic one when $0 < \theta < 1$ , equal to it when $\theta = 1$ , and smaller when $\theta > 1$ . Moreover, $\mathrm { i f } \ \| g _ { i } \| > \| g _ { j } \|$ , then (11) is strictly smaller than $\| g _ { i } \| / \| g _ { j } \|$ , and this compression becomes stronger as $p$ increases. The transformation preserves ordering but does not uniformly shrink all displacements. If $s _ { ( 1 ) } \leq \cdots \leq s _ { ( M ) }$ are ordered direction norms, the corresponding affine displacement norms satisfy

$$
r _ { ( k ) } ^ { ( p ) } = R \left( \frac { s _ { ( k ) } } { \mu R } \right) ^ { 1 / ( p - 1 ) } .\tag{13}
$$

This identity motivates the upper-tail displacement diagnostic used in the experiments.

Larger powers strengthen disparity compression, but the limiting behavior warns against the rule “larger is always better.” For every fixed $g \neq 0 ,$

$$
\operatorname* { l i m } _ { p \to \infty } | | d _ { p } ( g ) | | = R .\tag{14}
$$

Thus, the affine response approaches a fixed-radius step and becomes nearly insensitive to the magnitude of $g .$ . Equivalently,

$$
\phi _ { p , R , \mu } ( d ) = \frac { \mu R ^ { 2 } } { p } ( \frac { \| d \| } { R } ) ^ { p }  \{ 0 , \quad \| d \| \leq R ,\tag{15}
$$

pointwise. Very large powers therefore approximate a hard displacement-radius constraint: they can suppress extreme movements, but may also erase useful magnitude information.

The pth-order regularizer also becomes more curved farther away from the server model. On $\{ d : \| d \| \leq D \}$ , its gradient is Lipschitz with constant

$$
L _ { \phi } ( D ) = \mu ( p - 1 ) \left( \frac { D } { R } \right) ^ { p - 2 } .\tag{16}
$$

For $D > R ,$ this quantity grows rapidly with p, i.e., the larger powers can require smaller accepted steps and more backtracking.   
For any fixed $D < R , L _ { \phi } ( D ) \to 0$ as $p  \infty$ , although the curvature need not decrease monotonically for moderate powers.

For a smooth fixed-batch objective, let $g = \nabla Q ( u )$ and suppose that $\nabla Q$ is L<sub>Q</sub>-Lipschitz on the trial segment. The descent lemma ensures that

$$
Q ( u - \alpha g ) \leq Q ( u ) - \alpha \left( 1 - \frac { L _ { Q } \alpha } { 2 } \right) \left. g \right. ^ { 2 } .\tag{17}
$$

Hence, every $\alpha \leq 2 ( 1 - c ) / L _ { Q }$ , passes the Armijo test with parameter c [19]. Algorithm 1 employs this acceptance rule as a finite-budget numerical safeguard.

These formulas motivate the two displacement diagnostics used in the following. Groupwise means show whether the response is selective, and an upper-tail-to-median ratio measures relative spread among the clients selected in each round.

The affine formula describes a reference subproblem rather than a complete federated convergence result. Even with exact client solutions and full participation, equal averaging for $p > 2$ is not generally a common-step gradient method on the average high-order envelope: the factor relating a client displacement to its envelope gradient depends on that displacement norm. Finite local budgets and partial participation introduce further errors. We therefore use the analysis to explain the radial mechanism and the role of backtracking, while leaving the outer convergence to future work.

## V. EXPERIMENTAL DESIGN

## A. Data, model, and stress conditions

The primary study uses frozen FEMNIST caches constructed from 60 writers, with each writer representing a single client. The caches comprise 8,064 training, 995 calibration, and 1,021 evaluation examples. The pooled training, calibration, and evaluation sets each contain all 62 FEMNIST character classes [20]. Although the calibration and evaluation sets are disjoint from the training set, they are drawn from the same writers. Consequently, the experiments evaluate generalization to new examples from known clients rather than to previously unseen clients. Each image is stored as a flattened $1 4 \times 1 4$ grayscale array with pixel values in [0, 1] and reshaped before being passed to the network.

The classifier has two $3 \times 3$ convolutional layers with unit padding and 16 and 32 output channels, respectively. Each convolution is followed by ReLU and $2 \times 2$ max pooling. The resulting $3 2 \times 3 \times 3$ representation is flattened and passes through a 64-unit fully connected ReLU layer and a 62-logit output layer. The network has 27,326 trainable parameters. Predictive performance is evaluated using cross-entropy loss and classification accuracy. The client regularizer is used only during loca training.

The training objective is nonconvex but smooth within each fixed activation and pooling pattern, with nondifferentiable boundaries introduced by ReLU and max pooling. At each local step, PyTorch automatic differentiation returns a direction for the complete regularized minibatch objective, following its documented rules in nondifferentiable elementary operations [21]. At differentiable points, this direction is the ordinary gradient. Proposition 1 applies to any prescribed direction and therefore provides a reference calculation for the direction returned by automatic differentiation. It does not describe the complete nonlinear client trajectory. The inequality (17) provides us with a smooth fixed-batch motivation for backtracking, while the experiment uses the capped same-minibatch rule as a numerical safeguard at every local step.

In each communication round, 18 of the 60 clients are sampled uniformly without replacement. Ordinary selected clients use three local iterations with initial step-size 0.5. In each stressed condition, 12 clients are selected once for each seed and remain in the stress-test group throughout all 100 rounds. For a fixed seed, this group is shared among the compared methods. For every stress-test client, all training labels are replaced according to $y \mapsto \left( y + 1 \right)$ mod 62, while the calibration and evaluation labels remain unchanged. Under moderate stress, these clients use nine local iterations and an initial step-size 1.0. Under severe stress, they use nine local iterations and initial step-size 2.0. The remaining clients retain the normal settings in Table I. This composite intervention is designed to induce a persistent group of comparatively large client displacements. The resulting displacement separation is measured in Section VI rather than assumed.

## B. Exponent protocol, pairing, and evaluation

The reference pair $( \mu , R ) = ( 0 . 0 3 , 1 )$ was selected by the original 60-round clean-plus-moderate calibration for the quadratic and cubic variants. We keep this pair fixed for

$$
p \in \{ 2 , 3 , 4 , 5 , 6 , 7 , 8 \} ,
$$

TABLE I  
PRIMARY FEMNIST PROTOCOL. IN THE STRESSED CONDITIONS, THE MULTIPLIERS APPLY ONLY TO 12 CLIENTS SELECTED ONCE PER SEED AND HELD FIXED THROUGHOUT TRAINING.
<table><tr><td>Item</td><td>Setting</td></tr><tr><td>Training / calibration / evaluation</td><td> $8 0 6 4 / 9 9 5 / 1 0 2 1$ </td></tr><tr><td>Writers / pooled classes</td><td> $6 0 / 6 \dot { 2 }$ </td></tr><tr><td>Rounds / clients per round</td><td> $1 0 \dot { 0 } / 1 8$ </td></tr><tr><td>Minibatch / ordinary local steps</td><td> $1 6 / 3$ </td></tr><tr><td>Initial local step / Armijo trial</td><td>0.5</td></tr><tr><td>Clean</td><td>labels unchanged, step ×1, local steps ×1</td></tr><tr><td>Moderate stress</td><td>cyclic label corruption, step  $\times 2 ,$  local steps ×3</td></tr><tr><td>Severe stress</td><td>cyclic label corruption, step ×4, local steps ×3</td></tr><tr><td>Powers in sensitivity study</td><td> $p \in \{ 2 , \bar { 3 } , 4 , 5 , 6 , 7 , 8 \}$ </td></tr><tr><td>Evaluation seeds</td><td> $3 0 0 1 { - } 3 0 0 5$ </td></tr></table>

allowing the sensitivity study to isolate the effect of the exponent. Although any real exponent $p \geq 2$ is theoretically admissible, we restrict our numerical study to the consecutive integers for a transparent and systematic comparison. Notice that the real-valued powers $p \geq 2$ may further refine the choice of $p$ and are left for future investigation.

For each condition and seed, all powers share the initial network, stress-test clients, participating clients in every round, and client-local minibatch streams. Their local-step, backward-pass, and communication budgets are therefore identical. The Armijo parameters are $c = 1 0 ^ { - 4 } , \rho = 0 . 5$ , and $J = 1 2$ , so each local search tests at most 13 step sizes. The direction, current value, and all trial values within one local search use the same minibatch. If all trials fail, the beginning-of-step iterate is restored.

The primary outcomes are pooled cross-entropy and pooled accuracy on the 1,021 evaluation examples. We also report the linearly interpolated tenth percentile of the 60 writer accuracies. For the $m = 1 8$ selected displacement norms in round $t ,$ ordered as $r _ { ( 1 ) } ^ { t } \le \cdots \le r _ { ( m ) } ^ { t } ,$ define

$$
T _ { t } = \frac { r _ { ( \lceil 0 . 9 m \rceil ) } ^ { t } } { \operatorname* { m a x } \{ r _ { ( \lceil 0 . 5 m \rceil ) } ^ { t } , 1 0 ^ { - 1 2 } \} } .\tag{18}
$$

The reported tail statistic averages $T _ { t }$ over rounds. For $m = 1 8$ , the numerator and denominator are the 17th and 9th ordered norms. Larger values indicate a more dominant upper displacement tail.

For selected paired comparisons, we enumerate all $5 ^ { 5 } = 3 1 2 5$ ordered bootstrap resamples of the five seedwise loss differences and report the 2.5% and 97.5% percentiles of the resampled means [22]. Because the exponent study is exploratory, involving many possible pairwise contrasts, we emphasize effect sizes, intervals, and paired directions rather than treating every comparison as a separate hypothesis test.

## VI. PRIMARY FEMNIST RESULTS

## A. Predictive performance and exponent sensitivity

Table II reports the common-parameter exponent study. Clean losses remain close among all powers, ranging from 2.2610 to 2.2922. Under moderate stress, loss decreases from 2.7208 at $p = 2$ to its minimum 2.4095 at $p = 7 ,$ , then rises to 2.4328 at $p = 8 .$ . Under severe stress, loss improves through $p = 6 ,$ , reaching 2.6235, and then increases to 2.6772 and 2.6701 at $p = 7$ and $p = 8 .$ . The accuracy results show the same broad intermediate-power advantage, although their exact optima differ: $p = 5$ gives the highest moderate-stress accuracy, while $p = 6$ gives the highest severe-stress accuracy.

TABLE II  
FINAL FEMNIST RESULTS OVER FIVE PAIRED SEEDS, REPORTED AS MEAN ± SAMPLE STANDARD DEVIATION. ALL POWERS USE THE COMMON SETTING $( \mu , R ) = ( 0 . 0 3 , 1 )$
<table><tr><td>p</td><td>Clean loss</td><td>Moderate loss</td><td>Severe loss</td><td>Moderate acc. (%)</td><td>Severe acc. (%)</td></tr><tr><td>2</td><td> $2 . 2 8 3 3 \pm 0 . 1 0 0 0$ </td><td> $2 . 7 2 0 8 \pm 0 . 1 0 8 7$ </td><td> $3 . 4 1 4 1 \pm 0 . 0 9 1 3$ </td><td> $3 8 . 2 8 \pm 5 . 8 7$ </td><td> $1 1 . 5 4 \pm 4 . 8 9$ </td></tr><tr><td>3</td><td> $2 . 2 9 2 2 \pm 0 . 0 9 4 3$ </td><td> $2 . 5 7 7 0 \pm 0 . 1 7 2 2$ </td><td> $2 . 9 5 5 2 \pm 0 . 1 0 5 5$ </td><td> $4 1 . 2 7 \pm 7 . 6 1$ </td><td> $3 0 . 7 5 \pm 5 . 8 7$ </td></tr><tr><td>4</td><td> $\mathbf { 2 . 2 6 1 0 \pm 0 . 0 6 3 7 }$ </td><td> $2 . 4 9 2 8 \pm 0 . 1 0 6 6$ </td><td> $2 . 7 9 9 1 \pm 0 . 1 1 4 4$ </td><td> $4 3 . 7 2 \pm 3 . 8 9$ </td><td> $3 6 . 0 6 \pm 4 . 0 1$ </td></tr><tr><td>5</td><td> $2 . 2 6 9 0 \pm 0 . 0 5 7 9$ </td><td> $2 . 4 1 8 8 \pm 0 . 1 0 0 5$ </td><td> $2 . 7 0 5 9 \pm 0 . 1 0 2 2$ </td><td> ${ \bf 4 5 . 7 0 \pm 3 . 0 5 }$ </td><td> $3 9 . 0 0 \pm 2 . 2 4$ </td></tr><tr><td>6</td><td> $2 . 2 6 4 0 \pm 0 . 0 7 0 7$ </td><td> $2 . 4 2 1 8 \pm 0 . 0 9 8 5$ </td><td> $\mathbf { 2 . 6 2 3 5 \pm 0 . 0 5 6 1 }$ </td><td> $4 5 . 3 9 \pm 1 . 9 9$ </td><td> ${ \bf 4 0 . 8 8 \pm 1 . 2 9 }$ </td></tr><tr><td>7</td><td> $2 . 2 6 3 4 \pm 0 . 0 4 1 4$ </td><td> $\mathbf { 2 . 4 0 9 5 \pm 0 . 0 8 5 1 }$ </td><td> $2 . 6 7 7 2 \pm 0 . 0 7 9 0$ </td><td> $4 5 . 4 1 \pm 2 . 6 0$ </td><td> $3 9 . 8 4 \pm 2 . 7 4$ </td></tr><tr><td>8</td><td> $2 . 2 7 7 2 \pm 0 . 0 7 8 0$ </td><td> $2 . 4 3 2 8 \pm 0 . 0 5 4 0$ </td><td> $2 . 6 7 0 1 \pm 0 . 0 4 4 3$ </td><td> $4 4 . 9 8 \pm 2 . 4 3$ </td><td> $4 0 . 2 0 \pm 1 . 0 2$ </td></tr></table>

Figure 2 makes the intermediate-power plateau visible. On the clean-plus-moderate calibration-style score, $p = 7$ has the lowest mean value, 2.3365, while $p = 5$ and $p = 6$ remain close at 2.3439 and 2.3429. The moderate-stress advantage of $p = 7$ over $p = 5$ and $p = 6$ is small, and the corresponding paired bootstrap intervals contain zero. Under severe stress, $p = 6$ improves over $p = 5$ by 0.0825 on average; the exhaustive paired bootstrap interval is [0.0221, 0.1210], and four of five seeds favor $p = 6$ . Its mean severe-stress loss is also below those of $p = 7$ and $p = 8 .$ , although those pairwise intervals contain zero. The robust conclusion is therefore an intermediate-power plateau rather than a uniquely optimal exponent: additional tail compression beyond this range does not translate into uniformly better prediction.

![](images/ab9a85980be6f451a7287eab0b24a691d044b0c16f3ed4deaf0085f28eae1c5f.jpg)  
Fig. 2. Mean final pooled loss versus p; bars show one sample standard deviation over five paired seeds.

## B. Displacement compression and local cost

The displacement-tail statistic decreases monotonically with $p$ in every condition; see Table III and Fig. 3. Under severe stress, it falls from 4.936 at $p = 2$ to 2.086 at $p = 8 ,$ , a reduction of about 57.7%. The stress-test-to-unmodified mean-displacement ratio likewise decreases from 5.21 to 1.91. Thus, increasing $p$ continues to equalize relative displacement magnitudes even after predictive loss has stopped improving.

TABLE III  
MEAN DISPLACEMENT-TAIL STATISTIC AND SEVERE-STRESS ARMIJO TRIAL-POINT COST. THE FINAL COLUMN IS RELATIVE TO $p = 2 .$
<table><tr><td>p</td><td>Clean tail</td><td>Moderate tail</td><td>Severe tail</td><td>Severe trials</td><td>Trial increase</td></tr><tr><td>2</td><td>1.448</td><td>3.070</td><td>4.936</td><td>7835.0</td><td>0.00%</td></tr><tr><td>3</td><td>1.428</td><td>2.561</td><td>3.506</td><td>8472.2</td><td>8.13%</td></tr><tr><td>4</td><td>1.395</td><td>2.210</td><td>2.763</td><td>8810.8</td><td>12.45%</td></tr><tr><td>5</td><td>1.362</td><td>2.015</td><td>2.437</td><td>9214.6</td><td>17.61%</td></tr><tr><td>6</td><td>1.343</td><td>1.905</td><td>2.284</td><td>9553.6</td><td>21.93%</td></tr><tr><td>7</td><td>1.327</td><td>1.817</td><td>2.172</td><td>9764.6</td><td>24.63%</td></tr><tr><td>8</td><td>1.302</td><td>1.777</td><td>2.086</td><td>10069.6</td><td>28.52%</td></tr></table>

Figure 4 shows how the tail reduction is produced. Relative to $p = 2 ,$ , intermediate powers increase the mean displacement of unmodified clients while decreasing that of stress-test clients, so the group separation narrows from both directions. For larger powers, the unmodified-client mean also declines. This turnover supports a selective-response interpretation rather than uniform shrinkage and helps explain why the smallest tail at $p = 8$ does not result in the best predictive loss.

All powers use the same number of local iterations, backward passes, and client–server messages. Their measured computationa difference is the number of Armijo trial-point forward evaluations. Under severe stress, this count grows from 7835.0 at $p = 2$ to 9553.6 at $p = 6 ,$ 9764.6 at $p = 7 ,$ , and 10069.6 at $p = 8$ . No run records a complete line-search failure or zero-step fallback. The writer-level tenth-percentile accuracy improves sharply from 2.35% at $p = 2$ to between 18.49% and 20.15% for $p = 5 \mathrm { - } 8$ under severe stress, but it does not improve monotonically with the exponent.

The numerical pattern therefore matches the theoretical tradeoff. Increasing p compresses relative displacement disparities, but beyond an intermediate range the additional compression no longer improves the prediction and continues to increase local search cost.

![](images/f98c95358389e353a560d85a278a1985c8fcde99f398e3d0fb5275b01fe78ccd.jpg)  
Fig. 3. Mean upper-tail displacement ratio versus $p ;$ bars show one sample standard deviation.

![](images/b6be0279214dd3a4c4708d700b4494d990f9bed2b87f17327492bfacbc420b80.jpg)

![](images/985e26a4fd3a5d8225ca2044bb706d86705aea37fe8e63e1f86976a05da16dc1.jpg)  
Fig. 4. Mean client-displacement norms versus p under moderate and severe stress. Bars show one sample standard deviation over five seeds. Increasing p first allows more movement in the unmodified group while progressively reducing stress-test-client movement. At the largest powers, the unmodified-client mean also turns downward, consistent with the fixed-radius affine limit.

## VII. DISCUSSION: CHOOSING THE EXPONENT

We would like to emphasize that there is no universally optimal exponent. The parameter $p$ controls how sharply the regularizer changes around R: larger powers are flatter below the crossover and steeper above it. In the affine model, they drive all nonzero displacement magnitudes toward $R ,$ while the local curvature outside the crossover grows rapidly. Consequently, choosing p balances three effects: suppression of extreme movements, preservation of informative magnitude differences, and local backtracking cost.

For the present FEMNIST protocol, the useful range is approximately $p = 5 \mathrm { - } 7 .$ , and no single exponent dominates every criterion. The choice $p = 5$ is the conservative option because it has the lowest local cost within this range and gives the highest moderate-stress accuracy. The choice $p = 6$ is preferable when severe large-displacement robustness is the main priority, since it gives the lowest severe-stress loss and highest severe-stress accuracy. The choice $p = 7$ gives the lowest moderate-stress loss and the best clean-plus-moderate score, although its advantage over $p = 5$ and $p = 6$ is small. The result at $p = 8$ rules out the simplistic conclusion that a larger exponent is always better: it gives the smallest displacement tail but does not improve predictive performance and requires the most trial evaluations. Table IV summarizes the observed tradeoffs in this protocol.

In practice, $p$ should be selected jointly with $\mu$ and R by validation. A reasonable rule is to choose the smallest exponent whose validation loss lies within a prescribed tolerance of the best observed value, thereby avoiding unnecessary curvature and overcompression. The current comparison deliberately keeps $( \mu , R )$ fixed to isolate the exponent, so it does not establish the jointly optimal triplet $( p , \mu , R )$ . A stronger tuning study should use several calibration seeds and keep the five evaluation seed untouched.

The scope remains limited to one $^ { 6 0 }$ -writer subset, one architecture, five seeds, and a composite stress intervention that changes labels, local steps, and initial step sizes together. The study does not test unseen writers, isolate individual stress components, or explore fractional powers. Finally, the affine calculation and smooth fixed-batch safeguard do not prove convergence of the implemented changing-minibatch ReLU procedure with partial participation.

TABLE IV  
EMPIRICAL EXPONENT CHOICES IN THE PRESENT PROTOCOL.
<table><tr><td>Priority</td><td>Observed choice</td></tr><tr><td>Lowest cost within the plateau</td><td> $p = 5$ </td></tr><tr><td>Best severe-stress performance</td><td> $p = 6$ </td></tr><tr><td>Best moderate-stress loss</td><td> $p = 7$ </td></tr><tr><td>Strongest tail compression</td><td> $p = 8$ </td></tr></table>

## VIII. CONCLUSION

We introduced HiFedProx, a first-order federated procedure based on a scale-matched family of power-type client regularizers. The exponent p controls the radial shape without changing the regularization force at the reference displacement R. The affine response shows that larger powers compress relative displacement disparities and approach a fixed-radius behavior, while the curvature analysis predicts increasing local search difficulty outside the crossover region.

The paired FEMNIST study over $p \in \{ 2 , 3 , 4 , 5 , 6 , 7 , 8 \}$ confirms this tradeoff. Predictive performance improves from the quadratic model to an intermediate high-order range: $p = 7$ provides us with the lowest moderate-stress loss, $p = 5$ gives the highest moderate-stress accuracy, and $p = 6$ results in the best severe-stress loss and accuracy. Our biggest choice of $p = 8$ further reduces displacement tails but does not improve predictive performance and increases Armijo trial cost. Thus, the exponent should be calibrated rather than maximized. In the present experiment, $p = 5 – 7$ provides the most useful range.

## REFERENCES

[1] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. Aguera y Arcas, “Communication-efficient learning of deep networks from decentralized¨ data,” in Proc. AISTATS, 2017, pp. 1273–1282.

[2] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, “Federated optimization in heterogeneous networks,” in Proc. MLSys, 2020, pp. 429–450.

[3] Y. Nesterov, “Inexact high-order proximal-point methods with auxiliary search procedure,” SIAM J. Optim, vol. 31, pp. 2807–2828, 2021.

[4] Y. Nesterov, “Inexact accelerated high-order proximal-point Methods,” Math. Program., vol. 197, pp. 1–26, 2023.

[5] M. Ahookhosh and Y. Nesterov, “High-order methods beyond the classical complexity bounds: inexact high-order proximal-point methods,” Math. Prog., vol. 208, pp. 365–407, 2024.

[6] A. Kabgani and M. Ahookhosh, “Moreau envelope and proximal-point methods under the lens of high-order regularization,” Set-Valued Var. Anal., vol. 33, no. 4, Art. no. 47, pp. 1–35, 2025.

[7] A. Kabgani and M. Ahookhosh, “ItsDEAL: Inexact two-level smoothing descent algorithms for weakly convex optimization,” arXiv:2501.02155, 2025.

[8] A. Kabgani and M. Ahookhosh, “ItsOPT: An inexact two-level smoothing framework for nonconvex optimization via high-order Moreau envelope,” SIAM J. Optim., in press.

[9] M. Ahookhosh, A. Iusem, A. Kabgani, and F. Lara, “Asymptotic convergence analysis of high-order proximal-point methods beyond sublinear rates,” SIAM J. Optim., in press.

[10] A. Kabgani, F. Lara and M. Ahookhosh, “Robust learning meets quasar-convex optimization: Inexact high-order proximal-point methods,” arXiv:2605.08474, 2026

[11] C. T. Dinh, N. H. Tran, and T. D. Nguyen, “Personalized federated learning with Moreau envelopes,” in Proc. NeurIPS, 2020.

[12] A. Hajarizadeh and S. Gupta, “DynaMu: Loss-guided adaptive proximal regularization for federated learning,” in Proc. IEEE/CVF CVPR Workshops, 2026, pp. 3321–3329.

[13] S. P. Karimireddy et al., “SCAFFOLD: Stochastic controlled averaging for federated learning,” in Proc. ICML, 2020, pp. 5132–5143.

[14] J. Wang, Q. Liu, H. Liang, G. Joshi, and H. V. Poor, “Tackling the objective inconsistency problem in heterogeneous federated optimization,” in Proc. NeurIPS, 2020.

[15] D. Jhunjhunwala, P. Sharma, A. Nagarkatti, and G. Joshi, “FedVARP: Tackling the variance due to partial client participation in federated learning,” in Proc. UAI, 2022, pp. 906–916.

[16] Y. Shi, Y. Zhang, P. Zhang, Y. Xiao, and L. Niu, “Federated learning with ℓ<sub>1</sub> regularization,” Pattern Recognit. Lett., vol. 172, pp. 15–21, 2023.

[17] M. Safaryan, R. Islamov, X. Qian, and P. Richtarik, “FedNL: Making Newton-type methods applicable to federated learning,” in ´ Proc. ICML, 2022, pp. 18959–19010.

[18] X. Zhang, X. Chen, M. Hong, S. Wu, and J. Yi, “Understanding clipping for federated learning: Convergence and client-level differential privacy,” in Proc. ICML, 2022, pp. 26048–26067.

[19] L. Armijo, “Minimization of functions having Lipschitz continuous first partial derivatives,” Pacific J. Math., vol. 16, no. 1, pp. 1–3, 1966.

[20] S. Caldas et al., “LEAF: A benchmark for federated settings,” arXiv:1812.01097, 2018

[21] PyTorch Contributors, “Autograd mechanics,” PyTorch Documentation, accessed Jul. 2026. [Online]. Available: https://docs.pytorch.org/docs/stable/notes autograd.html

[22] B. Efron, “Bootstrap methods: Another look at the jackknife,” Ann. Statist., vol. 7, no. 1, pp. 1–26, 1979.

[23] S. Holm, “A simple sequentially rejective multiple test procedure,” Scand. J. Statist., vol. 6, no. 2, pp. 65–70, 1979.