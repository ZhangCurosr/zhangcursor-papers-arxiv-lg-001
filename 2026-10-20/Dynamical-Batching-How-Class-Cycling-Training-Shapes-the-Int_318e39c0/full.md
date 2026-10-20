# Dynamical Batching: How Class-Cycling Training Shapes the Internal Structure of Neural Networks

Nicol´as Ratier-Werbin,<sup>1,</sup> <sup>∗</sup> Samuel Lozano,<sup>1,</sup> <sup>2,</sup> <sup>∗</sup> and Miguel Ruiz-Garc´ıa<sup>1,</sup> <sup>2</sup>

<sup>1</sup>Departamento de Estructura de la Materia, F´ısica T´ermica y Electr´onica,

Universidad Complutense de Madrid, 28040 Madrid, Spain

<sup>2</sup>Grupo Interdisciplinar de Sistemas Complejos, Universidad Complutense de Madrid, 28040 Madrid, Spain (Dated: October 20, 2026)

We investigate whether cyclically varying the class composition of training batches can modify both the performance and the internal organization of neural networks. Motivated by nonequilibrium systems, in which cycling between related sub-goals can make switching between their solutions increasingly eficient, we introduce Dynamical Batching (DB), a sampling protocol that periodically emphasizes each class while leaving the loss and the optimizer unchanged, and compare it with Uniform Batching (UB), in which every mini-batch has equal class proportions. In the regime where standard training fails, DB changes the outcome: on a three-arm spiral, a capacity-limited MLP that plateaus far below perfect accuracy under UB fits the training data under DB. Tracking individual weights, we find that under DB the change between class-specialized states is carried by a smaller set of weights than under UB, and that this set is markedly more persistent from cycle to cycle, on every synthetic dataset we test and in a wider network that fits the data under both protocols. This organization is a signature of the protocol rather than a predictor of its benefit: it appears equally where DB leaves accuracy unchanged. On CIFAR-10, the same protocol modestly improves the test accuracy of a channel-narrowed ResNet-10, a standard residual architecture, and of a second capacity-limited convolutional network. These results show that how class emphasis is distributed in time can shape both the solution a network reaches and the internal structure of the network.

## I. INTRODUCTION

Neural networks are typically optimized under a fixed training objective, yet the trajectory followed to optimize that objective is itself part of the learning process. Choices such as the ordering of examples, curriculum, learning-rate schedule, optimizer, and batch construction can change which solutions are reached even when the final objective is unchanged [1–5]. This raises a broader question: how does a training protocol change not only which function a network learns, but also where in parameter space diferent parts of that function are implemented?

The question is particularly interesting for tasks that contain several related sub-problems. In a multiclass classifier, for example, a single network must implement class-specific decision rules while sharing a finite parameter budget. More generally, multitask learning asks a shared model to solve several objectives whose gradients may not point in the same direction. When task gradients conflict, optimizing their sum can produce ineficient or destructive interference, motivating methods that explicitly rebalance, project, or otherwise manipulate task gradients [6–8]. Continual-learning methods approach a related problem from the opposite temporal regime: when tasks are presented sequentially, parameter isolation, masks, pruning, or architectural expansion can protect task-specific computations from interference [9–14], and whether tasks are presented in blocks or interleaved changes what is learned [15]. These approaches demonstrate that which parameters are allowed to change is consequential, but they typically impose parameter isolation explicitly or address tasks one after another.

Here we study a diferent regime: the architecture and the objective are unchanged, but optimization pressure is deliberately cycled between related sub-problems. The protocol is closely related to dynamical loss functions [16, 17], an approach motivated by oscillating strategies in biological systems, materials science and physics [18– 23], and distinct from learned, teacher-driven loss schedules [24]. A dynamical loss periodically changes the class weights of the cross-entropy loss, and this can rescue under-parametrized networks that fail under a static loss and can improve generalization in over-parametrized ones [16, 17]. This process can be analyzed through Hessian and neural-tangent-kernel dynamics [16]. Our questions are complementary: can a similar temporal forcing, implemented through mini-batch composition, rescue underparametrized networks that fail under Uniform Batching (UB), in which every mini-batch has equal class proportions? And does it reorganize the parameters so that switching between class-specific solutions becomes more economical? Acting on batch composition rather than on the loss leaves the per-example loss and the optimizer untouched, so the protocol can be dropped into any training pipeline without modification.

This question connects to a broader literature on modularity. Modularity is commonly understood as an organization in which subsets of a system support distinguishable functions while interacting relatively weakly with other subsets. In neural networks, modularity can be defined structurally through connectivity [25] or functionally through subsets of weights required for particular computations [26, 27]. Standard networks can exhibit some specialization without being explicitly modular, but the extent to which useful functional modules emerge from ordinary training remains an active question. In particular, Csord´as et al. [26] found that common architectures often fail to reuse the same submodules across related functions, while work on superposition has emphasized that finite representational capacity can instead lead multiple features to share neurons or directions [28, 29]. We therefore do not assume that a network becomes modular in the graph-theoretic sense. Instead, we ask whether the parameter changes associated with switching between sub-problems become localized: a weaker notion than modularity, but a directly measurable one.

Our central hypothesis is that periodically emphasizing each class concentrates the class-to-class change of the network onto a small, stable set of weights. If so, moving from one class-specialized state to another should involve fewer weights, largely the same ones every time, and a shorter distance in parameter space. We test this prediction with three complementary measurements: the size of the switching set, the smallest fraction of weights that carries most of the class-to-class change; its persistence, the overlap between the switching sets of consecutive cycles; and the switching distance between class-specialized states.

We first map the operating regime of Dynamical Batching (DB) on a three-arm spiral, where a capacityconstrained MLP that plateaus under UB reaches nearperfect training accuracy under a suitable amplitude– period combination. We then follow the same training runs over time: the class-to-class change concentrates on a small set of weights that persists from cycle to cycle, while the distance between class-specialized states decreases, although the drift of ordinary training also decreases. Finally, we test whether the accuracy efect extends beyond the toy setting with a channel-narrowed ResNet-10 on CIFAR-10 and three further convolutional architectures. Our contributions are:

1. A batch-level protocol that rescues a capacitylimited network. Without changing the loss, optimizer or architecture, DB lets a one-hidden-layer MLP that plateaus under UB fit the spiral; a grid over amplitude and period shows a broad region of gain bordered by a failure region.

2. Weight-level measures of class switching. We define a drift-corrected switching set, its persistence, and a switching distance, and track them against a UB control that uses identical bookkeeping.

3. Persistent, localized switching as a signature of the protocol. Under DB, the switching set is smaller and approximately twice as persistent as under UB across all five synthetic datasets, and the same pattern holds for a wider network that fits the data under both protocols.

4. Evidence beyond the toy setting. On CIFAR-10, DB modestly improves the test accuracy of a channel-narrowed ResNet-10, acting as a regularizer; a plain CNN with a narrow fully connected bottleneck shows a similar gain, while two wider networks show none beyond seed-to-seed noise.

## II. DYNAMICAL BATCHING

## A. Protocol

Figure 1 shows the DB protocol schematically. Each mini-batch is built with a time-dependent number of samples from each class. Panel (a) shows the fraction of samples drawn from each class over time, at low and high amplitude. Panel (b) is a cartoon of parameter space: the three colored cones represent regions where the class of that color is correctly classified, and the lines display training trajectories.

Our working hypothesis is that under UB a capacitylimited network reaches a compromise point that satisfies no class fully, whereas DB forces the network to visit a diferent region of parameter space each time a diferent class is emphasized and, as training progresses, the trajectory approaches a region where all classes are classified correctly. Moving from one class-emphasized region to another should then require increasingly small changes in parameter space, which we test directly in Sec. IV.

To integrate the oscillations into standard optimizers without altering the loss function (standard cross-entropy optimized with SGD or Adam), we implement the protocol as a dataset sampling schedule. For a dataset of C classes and period T, one focal class $c ^ { * } ( t ) \ = \ \lfloor t / T \rfloor$ mod C is emphasized during its window. Its relative sampling weight $g ( t _ { T } )$ follows a triangular ramp of amplitude $A \geq 1 { : }$

$$
\begin{array}{c} g ( t _ { T } ) = \left\{ { 1 + m { t _ { T } } , } \atop { 2 A - 1 - m { t _ { T } } , }  & { { 0 \leq t _ { T } \leq T / 2 , } } \end{array} \right.\tag{1}
$$

where $t _ { T } ~ = ~ t$ mod T and $m = 2 ( A - 1 ) / T$ . We call C consecutive periods a cycle. The class proportions are $\begin{array} { r } { p _ { c ^ { * } } ( t ) = \frac { g ( t _ { T } ) } { g ( t _ { T } ) + C - 1 } } \end{array}$ for the focal class and $p _ { j } ( t ) =$ $\frac { 1 } { g ( t _ { T } ) + C - 1 }$ for each of the other $C - 1$ classes (Fig. 1a). $\overset { \cdot } { A } = 1$ recovers UB.

Each mini-batch of nominal size B is built by deterministic stratification: class i contributes

$$
n _ { i } ( t ) = \lfloor p _ { i } ( t ) B \rfloor\tag{2}
$$

examples, drawn with replacement within the class. The class composition is therefore fixed at every step; only the choice of samples within each class is random.<sup>1</sup> In the final 5% of training steps all batches are uniform, so that every run ends under the same objective.

![](images/a8becc786089f37c4d0b2dfc6975562ae2f7a636edc61979d264cb0f69291771.jpg)

![](images/092eaae37f91828dbd62c746ce41489a1af370e5d33c6a8011fb9a20e5db20e8.jpg)  
FIG. 1. Learning by Dynamical Batching (DB). (a) Class proportion $p _ { i } ( t )$ , the fraction of each mini-batch drawn from class i, over two training cycles $\left( C = 3 \right)$ . At low amplitude (top, A = 5) every batch still contains all three classes: the focal-class proportion rises from $1 / 3$ to $0 . 7 1$ , while the other two fall to 0.14. At high amplitude (bottom, $A = 7 0 )$ the emphasized class dominates the batch, its proportion saturating near 1, while the other two nearly vanish. A cycle is $C$ consecutive periods, one sweep through the classes. (b) Schematic of the parameter space (illustrative, not data). Each class-specialized objective is satisfied in an extended, cone-shaped region (one cone per class, colored as in (a)), and the cones meet only near a common apex (Optimum, star). The shape only encodes that each class alone is satisfied in a large region of parameter space, while all classes together are satisfied only in a much smaller region. In the under-parametrized regime, Uniform Batching (UB, dotted gray) moves from initialization (circle) to a point that satisfies no class fully (Suboptimal, cross). DB (solid green) alternates between the cones, each period pulling the trajectory toward one class’s region before the next pulls it toward another.

## B. Experimental setup

To understand the learning dynamics in detail, we use a three-arm spiral, a multi-class variant of the classic two-spirals benchmark [30], with 100 training points per class and another 100 test points per class, generated by the same procedure with a diferent random seed; each training seed draws a new dataset and a new initialization (Appendix A). We train two one-hidden-layer MLPs with ReLU activations and Glorot initialization [31]: a small network with $H \ = \ 5 0$ hidden units (303 parameters; under-parametrized for this task) and a large one with $H \ : = \ : 5 0 0$ (3,003 parameters; over-parametrized). All main-text MLP experiments use 125,000 training steps, batch size $B = 5 0$ and Adam [32] (learning rate 0.002, $\beta = ( 0 . 9 , 0 . 9 9 9 )$ $\epsilon = 1 0 ^ { - 8 } )$ , with zero weight decay and no gradient clipping.

## III. DYNAMICAL BATCHING SUCCEEDSWHERE UNIFORM BATCHING PLATEAUS

We ask whether DB can improve learning, and under which conditions. Figure 2 maps the accuracy gain over the A = 1 (UB) baseline across amplitude and period (5 seeds per cell). A broad region of positive test-accuracy gain emerges for moderate amplitudes and periods, where the oscillating emphasis lets the network leave the poor solution reached under UB. At large amplitudes combined with short periods a failure region appears, consistent with the emphasis shifting faster than the network can consolidate. For the small network $\left( H = 5 0 \right)$ the best cell of the grid, $( A , T ) = ( 2 0 , 3 2 0 )$ , improves test accuracy by +37.9 points over the grid’s UB baseline. Re-running this configuration and its UB control with 20 seeds each gives $7 0 . 5 \pm 8 . 9 \%$ training and $5 9 . 2 \pm 7 . 7 \%$ test accuracy under UB, against $9 9 . 9 \pm 0 . 3 \%$ and $9 5 . 1 \pm 1 . 3 \%$ under DB (Table I). The UB shortfall is not a matter of slower convergence within our training budget: UB training accuracy gains less than 10 points over the last two thirds of training and remains far below DB throughout $( \mathrm { A p \mathrm { - } }$ pendix A 1, Fig. 5a), and doubling the budget to 250,000 steps raises it only to $7 5 . 3 { \pm } 1 1 . 3 \%$ (Table II). For the large network $( H = 5 0 0 )$ , UB already fits the training set, and DB improves test accuracy only slightly, by at most +1.7 points in the grid, at $( A , T ) = ( 1 0 , 1 5 0 )$ ; with 20 seeds, test accuracy is $9 6 . 6 \pm 1 . 5 \%$ under UB and $9 8 . 2 \pm 1 . 2 \%$

under DB.

Why does UB plateau when the same network can fit the data under DB? One candidate is a gradient-conflict plateau, related to the conflicting-gradient problem of multi-task learning [7]. When per-class gradients disagree, the balanced gradient $\begin{array} { r } { \frac { 1 } { C } \dot { \sum _ { i } } \nabla { \mathcal { L } _ { i } } } \end{array}$ can vanish at a point where no individual class is satisfied $( \nabla { \mathcal { L } } _ { i } \neq 0$ for every i), so the mean step of UB is zero there, whereas a class-focused batch follows a single, undiluted $\nabla { \mathcal { L } } _ { i }$ . If $\begin{array} { r } { \sum _ { i } \nabla { \mathcal { L } } _ { i } = 0 } \end{array}$ and the per-class gradients have equal norm, their mean pairwise cosine must equal $- 1 / ( C - 1 )$ , i.e. an angle of 120<sup>◦</sup> for $C = 3$ (Appendix B 2). We measured this cosine for gradients computed on fixed, single-class probe sets, with all C gradients evaluated at the same parameters (Appendix B 2, Fig. 8a,b). Under UB it sits at or close to −0.5 throughout training, both in the small network that plateaus near 71% training accuracy and in the large network that fits the data. Under DB, evaluated at the class-emphasized states, the class gradients remain strongly opposed but do not cancel (cosine from $\approx - 0 . 3 9$ early in training to ≈ −0.45 after ≈ 40–50 cycles). Two conclusions follow. First, because the cancelling arrangement is equally present when UB succeeds, it cannot by itself explain the plateau of the small network: it indicates only that UB training stays close to a stationary point of the balanced loss, which is poor in the small network and good in the large one. Second, DB does not remove the conflict between classes; it changes which combination of the conflicting gradients the optimizer follows. The DB batch weights the classes unequally, so its expected gradient is $\sum _ { i } p _ { i } ( t ) \nabla { \mathcal { L } } _ { i }$ with $p _ { c ^ { * } } ( t ) > 1 / C$ . Even at a point where the balanced sum vanishes, this step equals $\begin{array} { r } { \dot { \sum } _ { i } \big ( p _ { i } ( t ) - \frac { 1 } { C } \big ) \nabla \mathcal { L } _ { i } } \end{array}$ , which is generically nonzero: the network keeps moving where UB would stall. We regard this as a hypothesis for how DB leaves the poor solution, not as its explanation.

## IV. CLASS SWITCHING LOCALIZES ONTO A SMALL, PERSISTENT SET OF WEIGHTS

We quantify how the protocol organizes the weights with three measures. Let $\theta _ { i } ( c )$ be the average of the parameter vector over the $T / 2$ training steps in the central half, $[ T / 4 , 3 T / 4 )$ , of class i’s period in cycle c, where class i dominates the batch.

To ask which weights change when the emphasis moves from one class to the next, we measure for each weight k how much it varies across the $C$ class-focused periods of a cycle: $a _ { k } ( c )$ , the standard deviation of $\theta _ { i , k } ( c )$ across classes $i ,$ after removing the part of that variation that is simply steady drift across the consecutive class windows (Appendix B, Eq. (B2)). The switching set $\boldsymbol { S } ( \boldsymbol { c } )$ is the smallest set of weights that together carry 80% of the total variation $\textstyle \sum _ { k } a _ { k } ^ { 2 } ( c ) ;$ ; all other weights form the scafold. We report its size as a fraction of all n weights, $| S ( c ) | / n$ . The ordering between protocols is unchanged for thresholds of 50% and 90% (Appendix B, Fig. 8d).

Whether the switch is carried by the same weights every cycle is measured by the Jaccard overlap of consecutive switching sets,

$$
J ( c ) = \frac { | S ( c ) \cap S ( c - 1 ) | } { | S ( c ) \cup S ( c - 1 ) | } ,\tag{3}
$$

the number of weights that are switching weights in both cycles, divided by the number that are switching weights in at least one of them.

The simplest measure of how far the network moves when it switches classes is the switching distance, the mean distance over the $C ( C - 1 ) / 2$ class pairs,

$$
d ( c ) = \frac { 2 } { C ( C - 1 ) } \sum _ { i > j } \lVert \pmb { \theta } _ { i } ( c ) - \pmb { \theta } _ { j } ( c ) \rVert _ { 2 } .\tag{4}
$$

UB uses the same schedule and the same bookkeeping, but with $A = 1$ , so class sampling stays uniform. $\pmb \theta _ { i } ( c )$ is then not a class-emphasized state but the average parameter vector over the corresponding window of ordinary training, and its oscillation, switching set and distance measure the fluctuations and drift of ordinary training over the same windows. UB is therefore the natural control for how concentrated and how stable the motion is, but not for how large it is: DB moves between genuinely diferent class solutions and UB does not, so the distance between class states is expected to be larger under DB.

## A. Results

Throughout this section the small network is trained at $( A , T ) = \hat { ( 2 0 , 3 2 0 ) } , ^ { 2 }$ and the UB control is the identical run with $A = 1$

By the end of training, switching between classes under DB is carried by fewer weights than the fluctuations of UB training, and largely by the same weights every cycle (Fig. 3a,b, Table I). Under DB, 80% of the class-to-class variation is carried by only 14% of the weights after ≈ 30 cycles and by $1 9 \pm 6 \%$ at the end of training; under UB, the same fraction of the ordinary training fluctuations requires about a third of the weights throughout (32 ± 4% at the end; Fig. 3a). The participation ratio of $\pmb \theta _ { i } -$ $\theta _ { j }$ , a threshold-free measure of the same concentration, agrees in direction but separates the protocols only late in training (Appendix B, Fig. 8c). A small switching set could still be redrawn in every cycle, but it is not: consecutive switching sets overlap with a Jaccard index that peaks near 0.8 around cycle 30 and ends at 0.59±0.13 under DB, against $0 . 3 3 \pm 0 . 0 6$ under UB (Fig. 3b). Both measures are thus most favorable around cycle 30 and relax partially afterwards, while staying well separated from UB. At the same time, the distance between classemphasized states falls about 4.5-fold, from 2.9 to 0.64 ± 0.14 (Fig. 3c). This decrease is not by itself specific to DB: the drift of ordinary training over the same windows also falls, from ≈ 1.6 to 0.24 ± 0.19, mostly within the first few cycles, and the DB distance remains above it, as it must. Relative to that drift, the DB distance still shrinks, from roughly five times the UB value around cycle 10 to 2.7 times at the end. What is specific to DB is that the class states remain distinct while the change between them is carried by a small, persistent set of weights. The three quantities are complementary: a network could switch through a small set of weights that is redrawn every cycle, or through a stable set that moves ever farther. Figure 3e sketches the picture they give together.

Dynamical Batching  
Uniform Batching  
![](images/113387af83fa90d4040dad54fa2846aa78e99a6eb34ac98c9dfdfc43b41d54dc.jpg)

(a)  
![](images/f014ccbb210812da9870c73e53ffcb5c1365cf926114cf1f20c3e20851c2374d.jpg)  
FIG. 2. Dynamical Batching (DB) on the three-arm spiral. (a) Accuracy gain in percentage points over the Uniform Batching (UB, A = 1) run with the same period T, across oscillation amplitude A and period T, for the small $( H = 5 0 )$ and large $( H = 5 0 0 )$ networks; training and test gain shown separately. Color scales difer between rows. Each cell averages 5 seeds; the yellow and green boxes mark the UB baseline (A = 1) and the best DB configuration, $( A , T ) = ( 2 0 , 3 2 0 )$ , used in (b) and in Sec. IV. For the small network, a broad region of positive gain reaches +37.87 points in test accuracy, next to a failure region at large A and small T. The large network already reaches 99.00% train and 95.93% test accuracy under UB in this grid, and DB improves test accuracy slightly (up to +1.74 points) in a narrow band at low amplitude, just above the failure region. (b) Decision boundaries for a single UB run $( A = 1 )$ and a single DB run at $( A , T ) = ( 2 0 , 3 2 0 )$ : the network is evaluated on a grid and shaded by predicted class, with training (circles) and test (crosses) samples colored by true label. UB produces a fragmented boundary that does not follow the arms (test accuracy 56.60%); DB recovers the spiral structure (test accuracy 94.47%).

This organization is not a by-product of the accuracy gap of the small network. In the large network $( H = 5 0 0$ $( A , T ) = ( 1 0 , 1 5 0 ) )$ ), both protocols fit the training set (mean training accuracy $\geq 9 9 . 9 \%$ , and the first two observations hold more strongly: under DB the switching set shrinks to $3 \pm 3 \%$ of the weights, against $1 5 \pm 6 \%$ under UB, and its persistence reaches $0 . 7 0 \pm 0 . 2 0$ , against $0 . 3 4 \pm 0 . 1 4$ . The distance between class states again falls, from 3.6 to $0 . 1 8 \pm 0 . 1 2$ , but here it closely tracks the drift of ordinary training throughout and ends close to it $( 0 . 1 3 \pm 0 . 1 1$ ; Appendix B 4, Fig. 9, Table IV).

The first two observations also largely generalize across data geometries (Appendix A, Table III). On five synthetic datasets trained with the same schedule, persistence is 1.8–2.2 times higher under DB than under UB in every case. The switching set is 26–46% smaller under DB on Spiral, Checkerboard, Dartboard and Rings, and marginally smaller, within the seed-to-seed spread, on Blobs, which UB solves almost perfectly. Neither measure tracks the accuracy gain, however. The same smaller, persistent set appears where DB rescues the network, on Spiral and on Rings (test accuracy $7 9 . 0 \pm 6 . 7 \%$ against $6 5 . 7 \pm 1 0 . 4 \%$ , and where it leaves accuracy unchanged within noise, on Checkerboard, Dartboard and Blobs.

Figure 3d illustrates the functional side of this picture in one representative DB run. Late in training, accuracy is at ceiling for most of the period, drops in its middle part while the loss rises, and recovers within 5–10 steps once the emphasis on the focal class has decreased, a small fraction of the $T = 3 2 0$ steps of the period. Early in training, accuracy stays between ≈ 0.35 and ≈ 0.5 for almost the whole period. The abrupt onset and end of the dip (t ≈ 116 and $t \approx 2 0 5 )$ coincide with the flooring of Eq. (2): at these steps the number of examples of each non-focal class drops from 3 to 2 per batch, and later returns to 3. The non-focal classes never vanish from the batch (at the peak, $g = A = 2 0$ , the batch holds 45 focal-class examples and 2 of each other class), but the step from 3 to 2 is a one-third cut in each non-focal class, which is why its efect on the loss is so abrupt. The dip therefore marks the interval in which the other classes are most under-represented, and the recovery is the functional counterpart of the switch that Fig. 3a–c quantify in weight space. Fast recovery late in training is what one would expect from a switch carried by fewer, persistent weights, but it also coincides with the network having learned the task; our measurements show that the two views change together, not that one causes the other.

![](images/bb3439a5fefdedac38d627e4bda9dd1027c607aa4453a59dedb35ca72eca2c4b.jpg)

(a)  
![](images/cc2995f099ca1fbfecb748b626a10578618a159af7fc89c5edea3d96e787fd25.jpg)

(b)  
![](images/82a0d739bc941a6c2c67341e9f9cd1366e691bdb89832e571e56efb7acae7b9e.jpg)

(c)  
![](images/cfda823667f4855f0c6e68475cd06ac09bc4e17259d5c51b140e8b9fe77b202f.jpg)

(d)  
(e)  
![](images/9e1b94079980cedda5053c563b254cef6401356388fe952721842364d03e4623.jpg)

![](images/f5ced456e0e6ec19a79960f2f310c6dd2d20b0b3beedc0ee48085ff4b45e7106.jpg)  
FIG. 3. Class switching concentrates on a small, persistent set of weights, and the distance between class states shrinks over training. Spiral, small network, $( A , T ) = ( 2 0 , 3 2 0 ) , B = 5 0$ . One cycle is $3 \times 3 2 0 = 9 6 0$ steps, giving 130 cycles, of which the first 122 are dynamical and the rest fall in the uniform final 5% and are excluded from the analysis. Solid green: Dynamical Batching (DB); dashed amber: Uniform Batching $( \mathrm { U B } , A = 1 )$ (a) Switching-set size $| S | / n \colon$ the fraction of all weights needed to carry 80% of the class-to-class variation. (b) Persistence of the switching set: the fraction of weights switching in either cycle that switch in both. (c) Switching distance d (Eq. (4)), the mean of $\lVert { \pmb \theta } _ { i } - { \pmb \theta } _ { j } \rVert _ { 2 }$ over the three class pairs. (d) Full-dataset training accuracy (black, left axis) and training loss (coral red, right axis; note the diferent loss scales) over one focal-class period $( T = 3 2 0$ steps; the horizontal axis counts steps from the start of that period), early and late in one DB run. The dashed vertical line marks the middle of the period $( t = T / 2 = 1 6 0 )$ , where the focal class is most emphasized. The step-like changes of the loss coincide with the steps at which the number of samples of each non-focal class per batch changes (Eq. (2)). (e) Weight-level schematic (no data): early in training the class-to-class change is spread over many weights and redrawn every cycle; late in training it is carried by a small set of weights (highlighted) that is largely the same in every cycle, while the remaining scafold barely oscillates. In (a)–(c), each point is the mean over 20 seeds and bands are ±1 s.d. across seeds.

TABLE I. Spiral, small network, $( A , T ) = ( 2 0 , 3 2 0 )$ , mean ± s.d. over 20 seeds. Accuracies are measured at the end of training (after the uniform final phase); switching measures at the last cycle of each protocol’s regime (the last dynamical cycle for DB, the last cycle for UB). $| { \cal { S } } | / n \colon$ switching-set size; J: persistence of the switching set; d: switching distance (Eq. (4)).
<table><tr><td>Method Train acc. (%) Test acc. (%)</td><td></td><td></td><td> $| S | / n$ </td><td>Persistence J</td><td> $d$ </td></tr><tr><td>UB</td><td> $7 0 . 5 \pm 8 . 9$ </td><td> $5 9 . 2 \pm 7 . 7$ </td><td> $0 . 3 2 \pm 0 . 0 4$ </td><td> $0 . 3 3 \pm 0 . 0 6$ </td><td> $0 . 2 4 \pm 0 . 1 9$ </td></tr><tr><td>DB</td><td> $9 9 . 9 \pm 0 . 3 $ </td><td> $9 5 . 1 \pm 1 . 3$ </td><td> $0 . 1 9 \pm 0 . 0 6$ </td><td> $0 . 5 9 \pm 0 . 1 3$ </td><td> $0 . 6 4 \pm 0 . 1 4$ </td></tr></table>

## V. A CHANNEL-NARROWED CNN ON CIFAR-10

The measurements above concern a single hidden layer trained on synthetic data. To check whether the accuracy efect of Sec. III carries over to a standard architecture and dataset, we apply the same batch-composition schedule [Eq. (1)] to a ResNet-10 [33] trained on CIFAR-10 [34] $( C = 1 0 )$ The network has four stages of one basic residual block each, with all channel widths scaled by a factor of 0.15 to reduce capacity. It is trained for 100,000 steps with batch size 128 using SGD with momentum 0.9 and weight decay $5 \times 1 0 ^ { - 4 } ;$ ; the learning rate starts at 0.0679 and is divided by 10 after one and two thirds of training. We scan 14 amplitudes from A = 1 to $A = 2 5 0$ and 15 periods from $T = 1 0$ to $T = 1 5 0 0$ steps (values listed in Appendix C), with 5 seeds per configuration. Accuracies are measured at the end of training on 5,000- example subsets of the training and test sets. Under UB, averaged over the 15 cells of the A = 1 column, the network reaches $1 0 0 . 0 0 { \pm } 0 . 0 0 \%$ training and 79.84±0.49% test accuracy: unlike the small spiral network, it fits its training set. All values in this section are mean ± s.d. over seeds. In the heatmaps, the gain of each cell is measured relative to the UB cell $( A = 1 )$ with the same period; the numbers quoted in the text instead use the mean of the whole A = 1 column, which pools all UB runs. Both baselines give the same gain for the best configuration to within 0.01 points (Appendix C). We do not repeat the weight-level analysis of Sec. IV here and leave it to future work.

A band of positive test-accuracy gain runs along the edge of the failure region, at large amplitudes and periods of one to a few hundred steps (Fig. 4): 29 of the 195 DB configurations exceed the UB test accuracy, 13 of them by more than half a point. The maximum test accuracy is 81.27 ± 0.38% at A = 125 and T = 200, a gain of +1.43 points, about 2.3 times the combined seed-to-seed s.d. of the two configurations (0.62), and neighboring cells along the same band reach comparable values [81.01% at $( A , T ) = ( 7 5 , 1 0 0 )$ , 80.98% at (100, 150) and 80.86% at (150, 200)]. Because this is the maximum over a grid selected on test accuracy, it is biased upward; the band of positive gain, rather than the single best cell, is the evidence for the efect. The gain comes with a decrease in training accuracy, to $9 4 . 6 5 \pm 0 . 5 8 \%$ for that configuration, and every configuration that improves test accuracy by more than half a point has training accuracy between 90% and 99.5%, whereas configurations that still fit the training set gain at most 0.35 points: here the oscillations act as a regularizer, trading training fit for generalization, rather than rescuing an optimization failure as the small network on the spiral. At large amplitudes combined with short periods, both training and test accuracy collapse, as in the failure region of the spiral, and the boundary of this region moves to longer periods as the amplitude grows. The size of the gain is comparable to that of the over-parametrized spiral network (+1.74 points), not to the rescue of the under-parametrized one (+37.87 points).

The architecture sweep of Appendix C (Table V) indicates which models benefit. Alongside the narrow ResNet-10 we tested a plain CNN whose capacity is limited by a 50-unit fully connected bottleneck, a VGG-style Myrtle5 network, and the same ResNet-10 at half width. The plain CNN shows the largest and broadest gain $( + 2 . 0 0 \pm 0 . 9 6 $ points at $A = 7 5 , T = 1 5 0 0$ , with 42 configurations improving test accuracy by more than one point), again with lower training accuracy; it does not quite fit its training set under UB (98.30%). Myrtle5 and the half-width ResNet-10 gain at most about half a point, within noise. The benefit is therefore not restricted to networks that fail to fit their data, since the narrow ResNet-10 fits it; across the four architectures it appears in the two narrowest ones, which are also those with the lowest UB test accuracy. With two architectures on each side, we read this as consistent with, not as a demonstration of, a benefit concentrated in capacity-limited models.

## VI. DISCUSSION

Our results show that Dynamical Batching (DB), relative to Uniform Batching (UB), changes both whether a capacity-limited network reaches a good solution and how the change between class-specialized states is organized in its weights. DB introduces no modular architecture, routing mechanism, task mask, or sparsity penalty; it only changes which class receives the strongest optimization pressure. Under DB, the change between class-emphasized states is carried by a set of weights that is about twice as persistent from cycle to cycle as under UB on every dataset we tested, and smaller on all of them.

This picture connects three lines of work. First, dynamic-objective methods show that changing the optimization target over time can improve trainability and generalization [16]; we find that the same can be achieved through batch composition alone. Second, multitasklearning work identifies interference between task gradients as a central optimization obstacle [6–8]; our probegradient measurements show that per-class gradients nearly cancel under UB, equally when training plateaus and when it succeeds, so conflict alone does not explain the plateau, and that DB changes how the conflicting gradients are combined (Appendix B 2). Third, modularity and continual-learning work shows that allocating diferent functions to distinguishable parameter subsets can reduce interference [10, 11, 26], and optimization is known to concentrate in small subspaces or subnetworks [35, 36]. Our experiments suggest a bridge: temporal class cycling lets a network carry the class-to-class change in a stable subset of its parameters, without that subset being imposed.

The connection to nonequilibrium training is suggestive. Periodically varying related goals can produce modular organization without explicitly selecting for it [37], can facilitate adaptation [38], and has an optimal time scale [39]; Falk et al. [22] showed that periodic target switching can teach a physical material to change functionality through a small subset of design variables. We observe a related efect in gradient-trained networks: repeated exposure to diferent but related class objectives is associated with a small, persistent parameter footprint for switching. In our setting, however, this footprint is not a set of class-specific modules.

Several limitations are important. First, the weightlevel evidence comes from one-hidden-layer MLPs on synthetic three-class tasks; on CIFAR-10 we show only an accuracy efect. Second, neither the size nor the persistence of the switching set tracks the accuracy gain. Third, our control is ordinary UB training, whose window-towindow motion is largely stochastic; a periodic emphasis on class-agnostic data subsets would separate efects of class structure from those of periodic forcing per se. Fourth, our notion of localization is weaker than the functional modularity of Csord´as et al. [26] and Hod et al. [27], and our measurements are correlational: a causal test would freeze the switching set late in training and ask whether switching and recovery fail, compared with freezing the same number of random weights. Fifth, the protocol requires a meaningful partition of the data, and amplitude and period interact strongly; configurations were selected on test accuracy, and we have not compared DB with other regularizers or class-reweighting schemes on CIFAR-10.

![](images/b3b48a401aca224fcdeb06b24f972ec13a4a347a9c6b3bb6cb4c272fb9da1865.jpg)

![](images/0568a205f401ed0cc875f3eebc8c6e6605031dedc2fce69cc771248abe035fcc.jpg)  
FIG. 4. Accuracy change on a channel-narrowed ResNet-10. Gain in training (left) and test (right) accuracy, in percentage points, relative to the Uniform Batching (UB) run (A = 1) with the same period T (UB averaged over all periods: train 100.00%, test 79.84%), across amplitude A and period T, for a ResNet-10 with channel widths scaled by 0.15, trained on CIFAR-10. Each cell averages 5 seeds. The color scale is clipped at −30 points: dark blue cells are runs that collapse, mostly to chance level. Axes are categorical; only a subset of the tested values is labeled.

Overall, the main result is not that cyclic training universally improves neural networks. Rather, it suggests that how optimization pressure is distributed in time can shape the internal organization of the solution: a fixed architecture trained under a changing sequence of related sub-problems can develop a small, persistent parameter response to those sub-problems.

## DATA AVAILABILITY

The protocol is fully specified in Sec. II [Eqs. (1)–(2)], the weight-level measures in Sec. IV and Appendix B [Eq. (B2)], and the probe-gradient test in Sec. III and Appendix B 2. Dataset generation, grid-search configurations and baselines for the synthetic datasets and CNN architectures are reported in Appendixes A and C. The synthetic datasets are generated by the released code, and CIFAR-

10 is publicly available [34]. Code and configuration files to reproduce all results are available at https://github. com/samuellozanoiglesias/dynamical\_batching and archived on Zenodo [40].

## USE OF AI TOOLS

Generative AI tools were used to help refine hypotheses about the mechanism behind the observed accuracy gains, to provide feedback on experimental design (e.g., the choice of control conditions and ablations), to search for and summarize related literature while writing the Introduction and Discussion, to suggest structural edits to the manuscript, and to polish the writing. They were not used to generate datasets, to formulate or prove mathematical claims, to clean or reformat data, or to analyze data. The authors reviewed all AI-assisted content and take full responsibility for the content of this article.

## ACKNOWLEDGMENTS

The project that gave rise to these results received the support of a fellowship from Fundaci´on Ram´on Areces. We acknowledge support from Ram´on y Cajal program (RYC2021-032055-I) funded by MCIN/AEI/10.13039/501100011033 and by European Union NextGenerationEU/PRTR, a Research Grant from HFSP (Ref.-No: RGEC33/2024) with the award DOI (https://doi.org/10.52044/HFSP.RGEC332024.pc.gr. 194170) and grant PID2023-147067NB-I00 funded by MCIU/AEI/10.13039/501100011033 and by ERDF/EU. Samuel Lozano is a Graduate Fellow of the Madrid City Council (Spain) at the Residencia de Estudiantes, 2025-2026.

## Appendix A: Additional Toy Experiments

Throughout the Appendices, DB denotes Dynamical Batching and UB Uniform Batching $( A = 1 )$ . This appendix expands the toy-network experiments of the main text. We first show the training curves behind the spiral results, then test robustness to batch size on the spiral, and finally evaluate the weight-level measures on four additional synthetic datasets. The purpose is not to claim universal optimal hyper-parameters, but to determine whether the qualitative pattern survives changes in data geometry.

## 1. Training curves on the spiral: UB plateaus, DB does not

Figure 5 shows training and test accuracy over the whole run for the 20-seed experiments of Sec. IV and Appendix B 4. For the small network (panel a), UB rises during the $\mathrm { f i r s t } \approx 2 0 { , } 0 0 0$ steps and then improves only slowly: over the last two thirds of training its training accuracy gains less than 10 points, ending at $7 0 . 5 \pm 8 . 9 \%$ $( 5 9 . 2 \pm 7 . 7 \% \mathrm { t e s t } )$ , with a wide spread across seeds. The same network under DB passes the UB curve after ≈ 20,000 steps and ends at $9 9 . 9 \pm 0 . 3 \%$ training and 95.1 ± 1.3% test accuracy. For the large network (panel b), both protocols fit the training set, and UB does so faster than DB; DB ends with slightly higher test accuracy $( 9 8 . 2 \pm 1 . 2 \%$ against $9 6 . 6 \pm 1 . 5 \% )$

UB accuracy in the small network is still increasing slowly at the end of training, so we tested whether a longer budget closes the gap. Doubling the training length to 250,000 steps, with everything else unchanged (20 seeds per protocol, uniform final 5%), barely moves UB: training accuracy rises to $7 5 . 3 \pm 1 1 . 3 \%$ and test accuracy to $6 2 . 4 \pm 1 0 . 8 \%$ , that is, by 4.8 and 3.2 points for 125,000 additional steps. DB reaches $1 0 0 . 0 \pm 0 . 0 \%$ training and $9 4 . 9 \pm 1$ .4% test accuracy, so the gap remains 24.7 points in training and 32.5 points in test accuracy (Table II). The diference is therefore not a matter of DB reaching the UB solution sooner within a fixed budget. The weight-level picture is also unchanged at the longer budget: under DB the switching set is about half the size of that under UB and about 1.6 times as persistent. All other small-network results use 125,000 steps, for consistency across datasets and protocols.

## 2. Batch-size robustness on the spiral

We swept period T and amplitude A for two network widths $( H = 5 0$ and $H = 5 0 0$ hidden units) on the threearm spiral, with the batch size increased from $B = 5 0$ to $B = 2 0 0$ and the optimizer and architectures unchanged. The corresponding grid maps (Fig. 6, the analogue of Fig. 2a without decision boundaries) show the same qualitative structure: a broad region of positive gain and a failure region at large amplitudes and short periods.

For the small network $( H = 5 0 )$ the UB baseline reaches 70.20% training and 55.60% test accuracy, and the configuration at $( \bar { A } , T ) = ( 7 0 , 5 5 0 )$ reaches 99.33% training and 94.93% test accuracy. For the large network $( H = 5 0 0 )$ , the UB baseline reaches 99.33% training and 96.13% test accuracy, while the best tested oscillating configuration reaches 97.33% test accuracy at $( A , T ) =$ (80, 500). The numerical optima difer from those of the main-text grid because the batch size and sweep ranges difer; we therefore treat the grid search as evidence for a broad operating regime rather than for a universal optimal $( A , T )$

Batch composition is deterministic (Eq. (2)), so the protocol contains no multinomial sampling noise at any batch size. What the batch size changes is the efect of the flooring: the realized composition, and the amplitude above which non-focal classes vanish from the batch $( g ( t _ { T } ) > B - C + 1 )$ , both depend jointly on A, B and C. That the phase structure survives a fourfold change in B suggests that it is not tied to a particular flooring pattern. The failure region does move: at $\mathrm { B } { = } 2 0 0$ it recedes to somewhat shorter periods, most visibly for the large network, so the boundary depends on the batch size as well as on A and T.

## 3. Weight-level measures across five toy datasets

To test whether the pattern of Sec. IV is specific to the spiral geometry, we repeated the measurements on five synthetic benchmarks. All use the small MLP of the main text (one hidden layer, 50 units, ReLU activations, Glorot initialization) trained for 125,000 steps with $B = 5 0$ , with 20 seeds per configuration.

• Spiral: 3 classes, 100 points per class. Class j follows an arm with angular ofset $2 \pi j / 3 { : }$ its points have radii evenly spaced from 0.01 to 1 and angles evenly spaced over an interval of length 4π (two full turns), and Gaussian noise of standard deviation 0.2 is added to each angle.

• Blobs: 3 classes, 100 points per class. Fifteen Gaussian modes are equally spaced on a circle of radius 0.6 and assigned to the three classes in rotation, so that each class is a mixture of 5 modes $7 2 ^ { \circ }$ apart and neighboring modes belong to diferent classes; noise standard deviation 0.05.

• Checkerboard: 3 classes, 100 points per class, drawn from the cells of a $7 \times 7$ grid whose labels alternate among the three classes; noise standard deviation 0.1.

• Dartboard: 3 classes, 100 points per class. The disc between radii 0.1 and 1.0 is divided into 3 radial rings and 12 angular sectors, and the cell in ring r and sector s is assigned class $( r + s )$ mod 3. Points are sampled uniformly in area within the central part of each cell, leaving an empty margin of 15% of the cell’s radial and angular width on each side; Gaussian noise of standard deviation 0.03 is then added to both coordinates.

TABLE II. Spiral, small network, $( A , T ) = ( 2 0 , 3 2 0 )$ , trained for 250,000 steps (twice the budget used elsewhere), $\mathrm { m e a n } \pm \mathrm { s . d . }$ over 20 seeds. Columns as in Table I.
<table><tr><td>Method Train acc. (%) Test acc. (%)</td><td></td><td></td><td> $| S | / n$ </td><td>Persistence J</td><td>d</td></tr><tr><td>UB</td><td> $7 5 . 3 \pm 1 1 . 3$ </td><td> $6 2 . 4 \pm 1 0 . 8$ </td><td> $0 . 2 9 \pm 0 . 0 8$ </td><td> $0 . 3 4 \pm 0 . 0 6$ </td><td> $0 . 1 8 \pm 0 . 1 3$ </td></tr><tr><td>DB</td><td> $1 0 0 . 0 \pm 0 . 0$ </td><td> $9 4 . 9 \pm 1 . 4$ </td><td> $0 . 1 5 \pm 0 . 0 7$ </td><td> $0 . 5 5 \pm 0 . 1 9$ </td><td> $0 . 3 8 \pm 0 . 1 7$ </td></tr></table>

![](images/87f6e1a5c71f020e592362f85c2aadf045fb0c2066bd4b5b944819d1d288cfed.jpg)

![](images/e2c04cfa729694292ac650dfbb009630b67e405a0b554078def7b9f5a7f263f0.jpg)  
FIG. 5. Training curves on the spiral. Full-dataset training (dark) and test (light) accuracy over training for UB (amber) and DB (green); lines are the mean and bands ±1 s.d. across 20 seeds. The final 5% of training is uniform for both protocols. (a) Small network $( H = 5 0 ) , ( A , T ) = ( 2 0 , 3 2 0 )$ . After an initial rise, UB levels of, gaining less than 10 points of training accuracy over the last two thirds of training, and ends at $7 0 . 5 \pm 8 . 9 \%$ training and $5 9 . 2 \pm 7 . 7 \%$ test accuracy. DB overtakes it after $\approx 2 0 { , } 0 0 0$ steps and ends at $9 9 . 9 \pm 0 . 3 \%$ and $9 5 . 1 \pm 1 . 3 \%$ . The fast oscillation of the DB curves follows the periodic class emphasis. (b) Large network (H = 500), $( A , T ) = ( 1 0 , 1 5 0 )$ . Both protocols fit the training set $( 9 9 . 9 \pm 0 . 5 \%$ UB, $1 0 0 . 0 \pm 0 . 0 \%$ DB), UB faster than DB; test accuracy ends at 96.6 ± 1.5% (UB) and $9 8 . 2 \pm 1 . 2 \%$ (DB).

• Rings: 3 classes, 100 points per class. Each class consists of 2 concentric rings, interleaved with those of the other classes, starting at radius 0.1 with a radial separation of 0.15 between consecutive rings; noise standard deviation 0.05.

Figure 7 shows the five datasets. All five are normalized to the region $[ - 1 , 1 ] \times [ - 1 , 1 ]$ , so the networks receive inputs on the same scale for every geometry. For every dataset, the test set (100 points per class) is generated by the same procedure as the training set with a diferent random seed. Each training seed also sets the dataset seed, so each seed draws a new dataset (training and test sets) and a new initialization, and the spread across seeds reported throughout includes dataset-to-dataset variability. For the spiral, whose radii are evenly spaced, training and test points therefore lie at the same radii and difer only in their angular noise.

For all datasets we use $( A , T ) = ( 2 0 , 3 2 0 )$ , the values selected for the main-text spiral experiment. This is a transfer test rather than a per-dataset optimization, so Table III should not be read as a comparison of the maximum attainable accuracy on each dataset.

Three observations follow from Table III.

a. The switching set is more persistent under DB on every dataset. On all five datasets the switching set is more persistent under DB than under UB, by a factor of 1.8–2.2.

b. The switching set is smaller under DB on every dataset. DB needs fewer weights than UB to carry 80% of the class-to-class variation on Rings (0.13 vs. 0.24), Spiral (0.19 vs. 0.32), Checkerboard (0.16 vs. 0.26) and Dartboard (0.23 vs. 0.31), a reduction of 26–46%, and marginally fewer on Blobs (0.16 vs. 0.20, within the seedto-seed spread). The concentration reported for the spiral in Sec. IV therefore transfers to all five geometries, although its magnitude varies.

c. Neither measure tracks the accuracy gain. DB improves test accuracy substantially on Spiral (+35.9 points) and Rings (+13.3), and leaves it unchanged within seedto-seed variability on Checkerboard (+1.2), Dartboard (+1.5) and Blobs (−1.1). Persistence is higher under DB on every dataset, and the switching set is smaller, so both properties appear with a gain (Spiral, Rings) and without one (Checkerboard, Dartboard, and Blobs). They are therefore a signature of the protocol, not of its accuracy benefit. The switching distance is larger under DB on every dataset, as expected, since only DB moves between class-specialized solutions (Sec. IV).

Training set  
![](images/44893b6e89ec0d8629731b08fa9b711331bb5fe400fb7bbc6dff967cdc99679d.jpg)

Test set  
![](images/3a3bb2a94217dc4d4bf87b815d19e7734d534f70c4a72389870156141d589fb3.jpg)

![](images/3a4af65fe642a11e632c676191301e10a2994f420e71b3aa641c624001c292e7.jpg)  
Amplitude (A)

![](images/de6dcf807401b1c9d5dc9c674393a4b4c2ff40a351f0f2e5c5eeac4808f57249.jpg)  
Amplitude (A)  
FIG. 6. Grid search on the spiral at a larger batch size $( B = 2 0 0 )$ . Accuracy gain in percentage points over the UB run $( A = 1 )$ with the same period T, across oscillation amplitude A and period T, for the small $\left( H = 5 0 \right)$ and large $( H = 5 0 0 )$ networks; training and test gain are shown separately. As for $B = 5 0$ in the main text, a broad region of positive gain unlocks the capacity-constrained network, bordered by a failure region at extreme amplitudes and short periods where the emphasis shifts faster than the network can consolidate. Each cell represents the average of 5 independent seeds.

(a)  
![](images/8600aba27a48a37fe4165f74be4959ac7d578d7521573698fcc9a5ae3e2c5cb8.jpg)

(b)  
![](images/a57ae5f6c6011c23d7f9ca88256b2c1f7790dd4821465cfb7faf722c7780c7ea.jpg)  
(c)

![](images/0201d04abe2733591eb152aa1f65a65d694009c58cf214971b7a7b29d74d5746.jpg)

(d)  
![](images/b3d76d82c6270981a1d211941ce5136a1c8a3a57e001c085881795a25ca97520.jpg)  
(e)

![](images/1f44049aa06ec872dba612e2641acbdd49c9d165631585fc7f63460b616872c9.jpg)  
FIG. 7. The five synthetic datasets. Training samples (circles) and test samples (crosses), colored by class, for (a) Spiral, (b) Blobs, (c) Checkerboard, (d) Dartboard and (e) Rings, each with 3 classes and 100 training and 100 test points per class, all within the region $[ - 1 , 1 ] \times [ - \dot { 1 } , 1$ ]. Each panel shows one realization (seed 0) generated from the exact configuration used for training; every training seed draws a new realization.

Taken together, the five datasets show that under DB the switch is carried by largely the same weights from cycle to cycle on all five geometries, and by fewer weights than under UB (within noise on Blobs), whereas the accuracy gain is specific to some of them. Nor is this organization a by-product of an accuracy gap: the set is smaller and more persistent under DB on Checkerboard and Dartboard, where both protocols reach the same accuracy, just as on Spiral and Rings; it is also about twice as persistent on Blobs, which UB already solves; and it holds in the large network of Appendix B 4, where both protocols fit the data.

TABLE III. Accuracy and weight-level measures across five toy datasets, $( A , T ) = ( 2 0 , 3 2 0 )$ , mean $\pm \ : \mathrm { s . d }$ . over 20 seeds. Accuracies are measured at the end of training (after the uniform final phase); switching measures at the last cycle of each protocol’s regime. $| { \cal { S } } | / n ;$ switching-set size; $J { \boldsymbol { : } }$ persistence of the switching set; d: switching distance (Sec. IV).
<table><tr><td>Dataset</td><td>Method Train acc. (%) Test acc. (%)</td><td></td><td> $| S | / n$ </td></tr><tr><td rowspan="2">Spiral</td><td>UB</td><td> $7 0 . 5 \pm 8 . 9$ </td><td> $5 9 . 2 \pm 7 . 7$   $0 . 3 2 \pm 0 . 0 4 \ 0 . 3 3 \pm 0 . 0 6 \ 0 . 2 4 \pm 0 . 1 9$ </td></tr><tr><td>DB</td><td> $9 9 . 9 \pm 0 . 3$ </td><td> $9 5 . 1 \pm 1 . 3$   $0 . 1 9 \pm 0 . 0 6 0 . 5 9 \pm 0 . 1 3 0 . 6 4 \pm 0 . 1 4$ </td></tr><tr><td rowspan="2">Blobs</td><td>UB</td><td> $9 9 . 8 \pm 0 . 4$   $9 7 . 8 \pm 1 . 1$ </td><td> $0 . 2 0 \pm 0 . 0 8 \ 0 . 3 5 \pm 0 . 1 0 \ 0 . 1 9 \pm 0 . 0 6$ </td></tr><tr><td>DB</td><td> $9 9 . 8 \pm 0 . 4$   $9 6 . 7 \pm 1 . 6$ </td><td> $0 . 1 6 \pm 0 . 1 0 ~ 0 . 7 1 \pm 0 . 1 0 ~ 0 . 3 5 \pm 0 . 0 8$ </td></tr><tr><td rowspan="2">Checkerboard UB</td><td></td><td> $9 7 . 8 \pm 2 . 7$   $8 9 . 7 \pm 4 . 1$ </td><td> $0 . 2 6 \pm 0 . 0 7 \ 0 . 3 3 \pm 0 . 0 6 \ 0 . 1 7 \pm 0 . 0 9$ </td></tr><tr><td>DB</td><td> $9 7 . 6 \pm 1 . 4$   $9 0 . 9 \pm 1 . 7$ </td><td> $0 . 1 6 \pm 0 . 0 8 ~ 0 . 7 3 \pm 0 . 1 4 ~ 0 . 5 3 \pm 0 . 1 2$ </td></tr><tr><td rowspan="2">Dartboard</td><td>UB</td><td> $9 6 . 8 \pm 1 . 5$   $8 3 . 8 \pm 3 . 0$ </td><td> $0 . 3 1 \pm 0 . 0 3 \ 0 . 3 4 \pm 0 . 0 6 \ 0 . 1 8 \pm 0 . 0 3$ </td></tr><tr><td>DB</td><td> $9 7 . 3 \pm 1 . 4$   $8 5 . 3 \pm 2 . 3$ </td><td> $0 . 2 3 \pm 0 . 0 4 \ 0 . 6 1 \pm 0 . 0 5 \ 0 . 4 4 \pm 0 . 0 6$ </td></tr><tr><td rowspan="2">Rings</td><td>UB</td><td> $8 0 . 4 \pm 9 . 7$   $6 5 . 7 \pm 1 0 . 4$ </td><td> $0 . 2 4 \pm 0 . 0 8 ~ 0 . 3 1 \pm 0 . 0 7 ~ 0 . 3 3 \pm 0 . 1 7$ </td></tr><tr><td>DB</td><td> $9 4 . 2 \pm 4 . 2$   $7 9 . 0 \pm 6 . 7$ </td><td> $0 . 1 3 \pm 0 . 0 3 ~ 0 . 6 3 \pm 0 . 0 5 ~ 0 . 4 7 \pm 0 . 0 7$ </td></tr></table>

## Appendix B: Weight-level switching analysis

Unless stated otherwise, results use the setting of Sec. IV: small network $( H = 5 0 ) , ( A , T ) = ( 2 0 , \bar { 3 } 2 0 )$ 20 seeds per protocol, dynamical cycles only. The large network $( H = 5 0 0 )$ uses $( A , T ) = ( 1 0 , 1 5 0 )$ , the best configuration of its grid in Fig. 2a, also with 20 seeds per protocol (277 cycles, of which the first 264 are dynamical). Solid green is DB and dashed amber is UB; bands are ±1 s.d. across seeds.

## 1. Definition of the switching set

Let $q _ { i } \in \{ 0 , \ldots , C - 1 \}$ be the position of class i in the cycle, $\begin{array} { r } { m _ { k } ( c ) = \frac { 1 } { C } \sum _ { i } \theta _ { i , k } ( c ) } \end{array}$ the cycle mean of weight k (since each $\theta _ { i , k } ( \bar { c } )$ is itself a time average, $m _ { k } ( c )$ is the average of weight k over the $C T / 2$ central steps of the cycle), and $D _ { k } ( c ) = m _ { k } ( c ) - m _ { k } ( c - 1 )$ its drift between consecutive cycles. Because the C class windows are consecutive in time, a weight that drifts by $D _ { k }$ per cycle moves by $D _ { k } / C$ from one class window to the next, and would difer between class states even without any classspecific change. For each weight we therefore remove a linear trend across the cycle, with its slope set by this drift, and quantify class-to-class switching by the variance

of the residuals:

$$
u _ { i , k } ( c ) = \theta _ { i , k } ( c ) - \alpha _ { k } ( c ) - \beta _ { k } ( c ) q _ { i } , \qquad \beta _ { k } ( c ) = \frac { D _ { k } ( c ) } { C } ,\tag{B1}
$$

$$
\alpha _ { k } ( c ) = m _ { k } ( c ) - \beta _ { k } ( c ) \frac { C - 1 } { 2 } , \qquad a _ { k } ^ { 2 } ( c ) = \frac { 1 } { C } \sum _ { i } u _ { i , k } ( c ) ^ { 2 } .\tag{B2}
$$

The intercept centers the residuals, $\begin{array} { r } { \sum _ { i } u _ { i , k } ( c ) = 0 } \end{array}$ so $a _ { k } ^ { 2 } ( c )$ is the variance across classes of the de-trended states; equivalently, $u _ { i , k } ( c ) = \theta _ { i , k } ( c ) - m _ { k } ( c ) - D _ { k } ( c ) \left( q _ { i } - \right.$ $( C - 1 ) / 2 ) / C$ . In the first cycle of each regime there is no previous cycle, and no trend is removed. All quantities are computed from the stored class states $\pmb \theta _ { i } ( c )$ of every cycle. The switching set $\boldsymbol { S } ( \boldsymbol { c } )$ is the smallest set of weights with $\begin{array} { r } { \sum _ { k \in S ( c ) } a _ { k } ^ { 2 } ( c ) \overset { \cdot } { \geq } 0 . 8 \overset { \cdot } { \sum _ { k } } a _ { k } ^ { 2 } ( c ) } \end{array}$ , obtained by ranking weights by $a _ { k } ^ { 2 } ( c )$

a. Why the slope is taken from the previous cycle. Two simpler choices fail for identifiable reasons. Without any trend removal, $u _ { i , k } = \theta _ { i , k } - m _ { k }$ , pure drift produces $u _ { i , k } = D _ { k } \left( q _ { i } - ( C - 1 ) / 2 \right) / C \mathrm { : }$ a class pattern that repeats identically every cycle. A network that only drifts would then show a switching set made of its fastest-drifting weights, and that set would appear highly persistent. Fitting the slope by least squares to the $\bar { C }$ class states of the current cycle removes this drift, but it also removes two of the C degrees of freedom, the mean and the slope, leaving only $C - 2$ residual directions. For $C = 3$ a single direction is left: with classes indexed by their position in the cycle, $u . , \kappa \propto \left( 1 , - 2 , 1 \right)$ ) and $a _ { k } ^ { 2 } = ( \theta _ { 0 , k } - 2 \theta _ { 1 , k } +$ $\theta _ { 2 , k } ) ^ { 2 } / 1 \overset { . } { 8 }$ . A weight that separates the first and the last class of the cycle, pattern $( + 1 , 0 , - 1 )$ , would then have $a _ { k } = 0$ and never enter the switching set, and the result would depend on which class the cycle happens to start with. Within one cycle, a class pattern that is linear in cycle position cannot be distinguished from drift. The drift $D _ { k } ( c )$ is estimated from outside the cycle, from the previous cycle mean, and is zero under a closed limit cycle whatever the oscillation amplitude, which is why Eq. (B2) separates the two.

## 2. Per-class gradient geometry

a. What is measured. For each class k, $\nabla \mathcal L _ { k } ( \pmb \theta )$ is the gradient, with respect to all network parameters, of the mean cross-entropy over a probe set that contains only examples of class k. The probe sets are drawn once, before training, and kept fixed throughout; with 100 training examples per class, each contains all training examples of its class. The gradient is a diagnostic computed on stored parameters, not the gradient of any training mini-batch. This distinction matters for UB. A UB mini-batch mixes the three classes, and its gradient is approximately the balanced sum $\begin{array} { r } { \frac { 1 } { C } \sum _ { k } \nabla { \mathcal { L } _ { k } } ; } \end{array}$ comparing such mixed gradients with each other would say nothing about conflict between classes. The probe gradients instead isolate the contribution of each class, for UB and DB alike.

In each cycle, and at each of the C class states $\pmb \theta _ { i } ( c )$ used for the switching measures (Sec. IV), we evaluate all C probe gradients $\check { \nabla } \mathcal { L } _ { k } ( \pmb \theta _ { i } ( c ) )$ at that same point and take their mean pairwise cosine; the reported value is the mean over the C states of the cycle. Evaluating the C gradients at the same parameters is what the argument below requires. Evaluating each class gradient at its own window center instead compares gradients at diferent parameters. Under DB, whose class states are far apart early in training, this produces an apparent cosine near 0 in the first cycles that disappears when the gradients are evaluated at the same point.

$b . \quad W h y \ – 1 / ( C - 1 )$ At a stationary point of the balanced objective, $\begin{array} { r } { \sum _ { k } \nabla { \mathcal L } _ { k } \ = \ 0 } \end{array}$ Taking the inner product with $\nabla { \mathcal { L } } _ { j }$ gives $\begin{array} { r } { \| \nabla \mathcal { L } _ { j } \| ^ { 2 } = - \sum _ { k \neq j } \langle \nabla \mathcal { L } _ { k } , \nabla \mathcal { L } _ { j } \rangle } \end{array}$ so for equal norms the mean pairwise cosine is exactly $- 1 / ( C - 1 )$ . A cosine of −0.5 therefore indicates that the per-class gradients nearly cancel, so that the balanced step is a small residual of large, opposing class gradients. One might suspect that this cancellation is confined to the output layer: given the current hidden features, the output layer solves a convex problem that balanced sampling brings quickly to its optimum, where its perclass gradients cancel whatever the quality of the features. Our measurements do not support this: restricted to the hidden-layer parameters or to the readout parameters alone, the cosine takes the same values as over all parameters, in both networks, close to −0.5 under UB and ≈ −0.45 under DB.

The converse also deserves care. A mean cosine of $- 1 / ( C - 1 )$ implies exact cancellation only for gradients of equal norm; the cosine alone is therefore evidence of near-cancellation, not a proof of it. Moreover, at a classemphasized DB state the optimizer follows the weighted sum $\begin{array} { r } { \sum _ { k } p _ { k } \nabla \mathcal { L } _ { k } } \end{array}$ , so near-stationarity of that weighted objective does not require the unweighted sum to vanish; a cosine above −0.5 under DB is therefore expected and should not be read as reduced conflict.

c. Results. Under UB $( \mathrm { F i g . 8 a , b ) }$ , the per-class gradients cancel throughout training in both networks, whether training ultimately succeeds or not. The cancelling arrangement is therefore the generic state of balanced training at this scale. It indicates that UB training stays close to a stationary point of the balanced loss over the time scale of a cycle, a poor one in the small network and a good one in the large network, and is not by itself the signature of a failed run. What distinguishes the small network is that it stays at a poor solution while in this arrangement (Sec. III, Fig. 5a). Under DB, the per-class gradients at the class-emphasized states are almost as strongly opposed $( \approx - 0 . 3 9$ early in training, $\approx - 0 . 4 5$ after ≈ 40–50 cycles in both networks) but do not cancel. DB therefore does not remove the conflict between classes. What it changes is the combination of the conflicting gradients that the optimizer follows: at each point, the batch weights the focal class more heavily, so the step is not the near-cancelling balanced sum. Whether this is what lets DB leave the poor solution remains a hypothesis.

## 3. Robustness of the concentration measure

The participation ratio, a threshold-free measure of how many weights carry the displacement between two class states, agrees in direction with the switching-set size, but separates the two protocols only late in training (Fig. 8c). The switching-set size depends on the energy threshold, but its ordering between the protocols does not: at 50%, 80% and 90% it ends at 0.07, 0.19 and 0.30 under DB against 0.10, 0.32 and 0.47 under UB (Fig. 8d).

## 4. Persistence does not require an accuracy gap: the large network

In the small network, DB fits the data and UB does not, so diferences in weight dynamics could in principle be a by-product of the accuracy gap. The large network removes this confound: both protocols fit the training set (Fig. 5b, Table IV): training accuracy $9 9 . 9 \pm 0 . 5 \%$ under UB and 100.0±0.0% under DB, test accuracy 96.6±1.5% and $9 8 . 2 \pm 1 . 2 \%$ , respectively (20 seeds).

The first two main-text observations hold, and are stronger, when there is no accuracy gap to explain them (Fig. 9a,b). The third needs the same qualification as in the main text: the distance between class states falls, but so does the drift of ordinary UB training over the same windows (from 2.8 to 0.13), and the two end close to each other (Fig. 9c).

## Appendix C: Additional CNN architectures on CIFAR-10

We repeated the grid search of Sec. V on three further architectures to determine which models benefit from the

![](images/6d0210e7cca72d420c7e380d6213c59f5cf3ea0f1716ef597e3a469727f76cfb.jpg)

![](images/67f29ce237a607bba778b90c6151b964f8a1a5027e7374b5e13ffaff007268f0.jpg)

![](images/2c84b4758e550888e60cab0e5af75df36eb889ceb22cf0bb0c6f0fd028b946a0.jpg)

![](images/9e8c58479d39fc8196094b629b10b3134fa3823bda1462e6c374fad397dd6b96.jpg)  
FIG. 8. Additional measures on the spiral. Solid green: DB; dashed amber: UB; bands ±1 s.d. across 20 seeds. (a) Mean pairwise cosine between the per-class probe gradients $\nabla \mathcal { L } _ { k } ,$ , each computed on the training examples of a single class, with all three gradients evaluated at the same parameters (Appendix B 2), small network $( H = 5 0 )$ Under UB it sits at $- 1 / ( C - 1 ) = - 0 . 5$ (dotted) throughout training, while training plateaus near 71% (Fig. 5a). Under DB it starts at $\approx - 0 . 3 9$ and settles at ≈ −0.45 after ≈ 40 cycles: the class gradients remain strongly opposed but do not cancel. (b) The same for the large network $( H = 5 0 0 )$ . UB again sits close $\mathrm { t o \ : - 0 . 5 }$ (between −0.50 and −0.48), although it fits the training set $( \geq 9 9 . 8 \% )$ the arrangement is generic under UB, not a signature of failure. DB stays near −0.40 for the first ≈ 25 cycles and settles at $\approx - 0 . 4 5$ after ≈ 50 cycles. (c) Participation ratio $\begin{array} { r } { \mathrm { P R } _ { i j } = ( \sum _ { k } \Delta _ { k } ^ { 2 } ) ^ { 2 } / \sum _ { k } \Delta _ { k } ^ { 4 } } \end{array}$ of $\Delta = \pmb { \theta } _ { i } - \pmb { \theta } _ { j }$ [41], the efective number of weights carrying the displacement, small network. Under DB it falls from 90 to 28, agreeing in direction with $| S | / n$ (Fig. 3a); under UB it drops within the first cycles and then rises to 40, with large seed-to-seed variability, so the two protocols separate only after cycle 75–80. (d) Switching-set size $| S | / n$ under DB (solid green) and UB (dashed amber) for energy thresholds of 50% (thin, lower), 80% (thick, main text) and 90% (thin, upper). After the first few cycles, each DB curve lies below the UB curve of the same threshold, ending at 0.07, 0.19 and 0.30 against 0.10, 0.32 and 0.47: DB needs fewer weights than UB to carry the same fraction of the class-to-class variation, whichever fraction is chosen.  
TABLE IV. Spiral, large network $( H = 5 0 0 )$ $( A , T ) = ( 1 0 , 1 5 0 )$ , mean $\pm \ : \mathrm { s . d . }$ over 20 seeds. Columns as in Table I.

$$
| S | / n
$$

$$
d
$$

$$
9 9 . 9 \pm 0 . 5
$$

$$
1 0 0 . 0 \pm 0 . 0
$$

$$
9 6 . 6 \pm 1 . 5
$$

$$
9 8 . 2 \pm 1 . 2 
$$

$$
0 . 1 5 \pm 0 . 0 6
$$

$$
0 . 0 3 \pm 0 . 0 3
$$

$$
0 . 3 4 \pm 0 . 1 4
$$

$$
0 . 1 3 \pm 0 . 1 1
$$

$$
0 . 7 0 \pm 0 . 2 0
$$

protocol: a plain CNN with a fully connected bottleneck, a VGG-style Myrtle5 network, and the ResNet-10 of the main text at half width.

$$
0 . 1 8 \pm 0 . 1 2
$$

All architectures are trained for 100,000 steps with batch size 128 using SGD with momentum 0.9 and weight decay $5 \times 1 0 ^ { - 4 }$ without gradient clipping or Nesterov momentum. The initial learning rate is 0.05 for the plain CNN and 0.0679 for all other architectures, and in every case it is divided by 10 at steps 33,333 and 66,666, one and two thirds of training. Every grid covers the same 14 amplitudes, A ∈ {1, 5, 10, 20, 30, 50, 75, 100, 125, 150, 175, 200, 225, 250}, and 15 periods, $T \in \{ 1 0 , 1 5 , 2 0 , 3 0 , 4 0 , 5 0$ 75, 100, 150, 200, 400, 800, 1000, 1200, 1500} steps, that is, 210 configurations of which 195 are DB $( A > 1 )$ . Training and test accuracy are evaluated at the end of training on 5,000-example subsets of the training and test sets. The UB baseline is the mean over the 15 cells of the $A = 1$ column, whose runs difer only by seed-to-seed and runto-run variability, and its uncertainty is the s.d. over all runs in that column. In the heatmaps (Figs. 4 and 10), the gain of each cell is instead computed relative to the $A = 1$ cell with the same period T. The two baselines difer only by the variability between A = 1 cells (UB test accuracy 79.61–80.16% across periods for the narrow

ResNet-10), and the best-cell gains they give difer by at most 0.32 points: +1.42 against +1.43 for the narrow ResNet-10, +2.15 against +2.00 for the plain CNN, +0.21 against +0.53 for Myrtle5 and +0.08 against +0.22 for the half-width ResNet-10 (same-T against pooled baseline). No conclusion depends on this choice. Uncertainties of single configurations are s.d. across seeds; for a diference we quote the combined s.d. $\sqrt { s _ { \mathrm { b a s e } } ^ { 2 } + s _ { \mathrm { b e s t } } ^ { 2 } }$ . All “best” values are maxima over the grid and are therefore biased upward by the selection, equally for every architecture. All heatmap’s points include the average over 5 independent seeds.

A test-accuracy gain beyond the seed-to-seed variability appears only in the two narrowest models, the ResNet-10 at width 0.15 (main text) and the plain CNN with a 50-unit bottleneck, and in both it comes with lower training accuracy. The two wider models, Myrtle5 and the ResNet-10 at width 0.5, show no gain beyond noise. Whether a model fits its training set under UB does not separate the two groups: the narrow ResNet-10 reaches 100% training accuracy under UB and still benefits. With two architectures on each side, the protocol appears useful primarily in capacity-limited networks, but we have not isolated whether width, the resulting test accuracy, or another property of these models is the relevant variable, nor the mechanism in these CNNs. In every architecture, large amplitudes combined with short periods make training collapse, often to chance level, and the failure boundary moves to longer periods as the amplitude grows (Figs. 4 and 10).

(a)  
![](images/ab875efb59bfe05f8c9ee7208053627383d39b2b940df4ab639a21879a1f8039.jpg)

![](images/74f9224d2413a719c2cc91ba40f725cb80e0edfc0ffe166ffbfd4f63f1c1801a.jpg)

(c)  
![](images/0f44deaa7604169a6ae8b10b2bbb5bb3e7e55ee79934d2eb82e73ac52b50096a.jpg)  
FIG. 9. Studying the structure of the network (analogous to Fig. 3a–c) for the large network $( H = 5 0 0 )$ , where both protocols fit the training set. $( A , T ) = ( 1 0 , 1 5 0 )$ , giving 277 cycles, of which 264 are dynamical. (a) Switching-set size $| { \cal { S } } | / n \colon$ DB drops to 0.06 within a few cycles and ends at 0.03; UB stays at 0.15–0.25 and ends at 0.15. The concentration is stronger than in the small network (Fig. 3a). (b) Persistence of the switching set (Jaccard overlap of consecutive cycles): DB starts below UB (0.1) while the switching set is being formed, rises to 0.7 by cycle ≈ 35 and stays at 0.55–0.75; UB stays at 0.30–0.34. (c) Switching distance: DB falls from 3.6 to 0.18, about twentyfold, closely tracking the drift of ordinary training over the same windows (UB, from 2.8 to 0.13).

## 1. Plain CNN with a fully connected bottleneck

Three convolutional stages with 16, 32 and 64 channels, $3 \times 3$ kernels, ReLU activations, no batch normalization, max pooling after every stage and no residual connections, followed by a 50-unit fully connected bottleneck before the 10-way classifier. Its capacity is limited by the bottleneck, like that of the small spiral network by its hidden layer.

Results: The UB baseline reaches $9 8 . 3 0 \pm 0 . 2 0 \%$ training and $7 5 . 9 7 \pm 0 . 7 5 \%$ test accuracy, so the network does not quite fit its training set. A broad region of positive test gain appears [Fig. 10(a)]: 85 of the 195 DB configurations improve on the UB test accuracy, 42 of them by more than one point. The best DB test accuracy is 77.96±0.60% (at $A = 7 5 , T = 1 5 0 0 )$ , a gain of $+ 2 . 0 0 { \pm } 0 . 9 6 $ points, and several other cells come within 0.1 points of it $[ ( A , T ) = ( 5 0 , 1 5 0 0 ) , ( 1 0 0 , 1 5 0 )$ and (100, 400)]. Training accuracy in the best configuration falls t $, 9 0 . 0 7 \pm 0 . 3 5 \%$ ， and the highest DB training accuracy, $9 8 . 1 9 \pm 0 . 1 8 \%$ (at $A = 5 , T = 3 0 )$ , is within noise of the baseline: as for the narrow ResNet-10, the protocol acts as a regularizer.

## 2. Myrtle5

A VGG-style Myrtle5 network [42]: four convolutional stages with one block each, channel widths doubling from 64 up to 512, 3 × 3 kernels, batch normalization, and pooling at the last stage.

Results: The UB baseline reaches $1 0 0 . 0 0 \pm 0 . 0 0 \%$ training and $8 5 . 1 0 \pm 0 . 4 7 \%$ test accuracy. Under DB, training accuracy stays at 100.00% wherever training does not collapse, and the best test accuracy is 85.64 ± 0.59% (at $A = 5 , T = 1 5 )$ . The diference, $+ 0 . 5 3 \pm 0 . 7 5$ points, is smaller than its combined s.d., so we treat it as no efect. The distinction from the capacity-limited models rests on this ratio of gain to seed noise, not on how the configurations were selected, which is the same for all architectures: +1.43 points is about 2.3 times its combined s.d. (0.62) and +2.00 about twice (0.96), whereas for the two wider networks the gain is below its combined s.d.

## 3. ResNet-10 at half width

The ResNet-10 of the main text, four stages of one basic block each, with width multiplier 0.5 instead of 0.15.

Results: The UB baseline reaches 100.00 ± 0.00% training and 85.98±0.48% test accuracy. The best DB test accuracy is $8 6 . 2 0 \pm 0 . 0 7 \% \left( \mathrm { a t } A = 1 0 , T = 5 0 \right)$ , a diference of $+ 0 . 2 2 \pm 0 . 4 8$ points. No DB configuration improves test accuracy by more than half a point. Widening the same architecture from 0.15 to 0.5 thus raises the UB test accuracy by about six points and removes the benefit of the protocol.

TABLE V. Architectures tested on CIFAR-10 and their response to DB. Test-accuracy gains beyond seed-to-seed noise appear only in the two capacity-limited architectures (top two rows). “Base” values are UB $( A = 1 )$ means over the 15 cells of the $A = 1$ column; “Best test” is the maximum over the DB configurations $( A > 1 )$ of the grid, and “Train at best” the training accuracy of that same configuration. Every configuration is run with 5 seeds. Values are mean $\pm \ \mathrm { s . d . }$ over seeds (accuracies in %); the test gain is given with the combined s.d. ${ \sqrt { s _ { \mathrm { b a s e } } ^ { 2 } + s _ { \mathrm { b e s t } } ^ { 2 } } } .$
<table><tr><td>Architecture</td><td>Fits train (UB)? Base train</td><td></td><td>Base test</td><td>Best test</td><td>(A,T)</td><td>Train at best</td><td>Test gain</td></tr><tr><td>ResNet-10 (width 0.15) (main text)</td><td>Yes</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $7 9 . 8 4 \pm 0 . 4 9$ </td><td> $8 1 . 2 7 \pm 0 . 3 8$ </td><td>(125,200)</td><td> $9 4 . 6 5 \pm 0 . 5 8$ </td><td>Yes  $( + 1 . 4 3 \pm 0 . 6 2 )$ </td></tr><tr><td>Plain  $\mathrm { C N N } { \mathrm { ~ + ~ } } \mathrm { F C }$  bottleneck (50)</td><td>No</td><td> $9 8 . 3 0 \pm 0 . 2 0 $ </td><td> $7 5 . 9 7 \pm 0 . 7 5$ </td><td> $7 7 . 9 6 \pm 0 . 6 0$ </td><td>(75,1500)</td><td> $9 0 . 0 7 \pm 0 . 3 5$ </td><td>Yes  $( + 2 . 0 0 \pm 0 . 9 6 )$ </td></tr><tr><td>Myrtle5 (base width 64)</td><td>Yes</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $8 5 . 1 0 \pm 0 . 4 7$ </td><td>85.64 ± 0.59</td><td>(5,15)</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td>No  $( + 0 . 5 3 \pm 0 . 7 5 )$ </td></tr><tr><td>ResNet-10 (width 0.5)</td><td>Yes</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td>85.98 ± 0.48 86.20 ± 0.07</td><td></td><td>(10,50)</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td>No  $( + 0 . 2 2 \pm 0 . 4 8 )$ </td></tr></table>

![](images/bde64ccdc1e5b5987fa8795096437fa6b44f97d5a8ef8974f644e8a1fed9728f.jpg)

![](images/8de4963d910d50b28e2c3bc88b98857bfbc16a304e826b47f7a0bdee788e533d.jpg)

![](images/4692693503816e4b93f2289f8ade738630b9dda89d0ba17c31df037fab8e6a64.jpg)

![](images/98e66b7712383021819868d78c48fb16aa74923b5a1d476e87b4b60b184ef67e.jpg)

![](images/b4dd401f1c9a49b29b9fe0ca184c9dc40acec44b20b956700260acf116829f61.jpg)

![](images/714a65eeb228d2ca96e0241879c005bf37dc276afee0fba78c65e0cd22bff3f8.jpg)  
FIG. 10. Grid searches for the additional CNN architectures. Gain in training (left) and test (right) accuracy, in percentage points, relative to the UB run $( A = 1 )$ with the same period T, across amplitude A and period T, on CIFAR-10. Each cell averages 5 seeds. The color scale is clipped at −30 points; dark blue cells are runs that collapse. Axes are categorical, as in Fig. 4. (a) Plain CNN with a 50-unit fully connected bottleneck (UB: train 98.30%, test 75.97%): a broad region of positive test gain at intermediate to large amplitudes and periods of about a hundred steps or more, with reduced training accuracy. (b) Myrtle5 (UB: 100.00%, 85.10%). (c) ResNet-10 at width 0.5 (UB: 100.00%, 85.98%). In (b) and (c) the tes accuracy outside the failure region is flat to within seed-to-seed noise, and the training set is fitted wherever training does not collapse.

[1] Y. Bengio, J. Louradour, R. Collobert, and J. Weston, Curriculum learning, in Proceedings of the 26th Annual International Conference on Machine Learning (2009) pp. 41–48.

[2] L. N. Smith, Cyclical learning rates for training neural networks, in IEEE Winter Conference on Applications of Computer Vision (WACV) (2017) pp. 464–472.

[3] B. Neyshabur, Implicit Regularization in Deep Learning, Ph.D. thesis, Toyota Technological Institute at Chicago (2017), arXiv:1709.01953.

[4] J. M. Cohen, S. Kaur, Y. Li, J. Z. Kolter, and A. Talwalkar, Gradient descent on neural networks typically occurs at the edge of stability, in International Conference on Learning Representations (2021).

[5] J. M. Cohen, A. Damian, A. Talwalkar, J. Z. Kolter, and J. D. Lee, Understanding optimization in deep learning with central flows, in International Conference on Learning Representations (2025) arXiv:2410.24206.

[6] O. Sener and V. Koltun, Multi-task learning as multiobjective optimization, in Advances in Neural Information Processing Systems, Vol. 31 (2018).

[7] T. Yu, S. Kumar, A. Gupta, S. Levine, K. Hausman, and C. Finn, Gradient surgery for multi-task learning, in Advances in Neural Information Processing Systems, Vol. 33 (2020).

[8] Z. Chen, V. Badrinarayanan, C.-Y. Lee, and A. Rabinovich, GradNorm: Gradient normalization for adaptive loss balancing in deep multitask networks, in Proceedings of the 35th International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 80 (PMLR, 2018) pp. 794–803.

[9] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska, D. Hassabis, C. Clopath, D. Kumaran, and R. Hadsell, Overcoming catastrophic forgetting in neural networks, Proceedings of the National Academy of Sciences 114, 3521 (2017).

[10] J. Serr\`a, D. Sur´ıs, M. Miron, and A. Karatzoglou, Overcoming catastrophic forgetting with hard attention to the task, in Proceedings of the 35th International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 80 (PMLR, 2018) pp. 4548–4557.

[11] A. Mallya, D. Davis, and S. Lazebnik, Piggyback: Adapting a single network to multiple tasks by learning to mask weights, in Proceedings of the European Conference on Computer Vision (ECCV) (2018) pp. 67–82.

[12] A. Mallya and S. Lazebnik, PackNet: Adding multiple tasks to a single network by iterative pruning, in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2018) pp. 7765–7773.

[13] A. A. Rusu, N. C. Rabinowitz, G. Desjardins, H. Soyer, J. Kirkpatrick, K. Kavukcuoglu, R. Pascanu, and R. Hadsell, Progressive neural networks, arXiv preprint arXiv:1606.04671 (2016).

[14] M. Ruiz-Garcia, Model architecture can transform catastrophic forgetting into positive transfer, Scientific Reports 12, 10736 (2022).

[15] T. Flesch, J. Balaguer, R. Dekker, H. Nili, and C. Summerfield, Comparing continual task learning in minds and machines, Proceedings of the National Academy of Sciences 115, E10313 (2018).

[16] M. Ruiz-Garcia, G. Zhang, S. S. Schoenholz, and A. J. Liu, Tilting the playing field: Dynamical loss functions for machine learning, in Proceedings of the 38th International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 139 (PMLR, 2021) pp. 9157–9167.

[17] E. Lav´ın-Pallero and M. Ruiz-Garc´ıa, Dynamical loss functions shape landscape topography and improve learning in artificial neural networks, Journal of Statistical Mechanics: Theory and Experiment 2025, 123402 (2025).

[18] A. Murugan and H. M. Jaeger, Bioinspired nonequilibrium search for novel materials, MRS Bulletin 44, 96 (2019).

[19] A. Murugan, K. Husain, M. J. Rust, C. Hepler, J. Bass, J. M. J. Pietsch, P. S. Swain, S. G. Jena, J. E. Toettcher, A. K. Chakraborty, et al., Roadmap on biology in time varying environments, Physical Biology 18, 041502 (2021).

[20] V. Mustonen and M. L¨assig, From fitness landscapes to seascapes: non-equilibrium dynamics of selection and adaptation, Trends in Genetics 25, 111 (2009).

[21] J. Cairns, F. Borse, T. Mononen, T. Hiltunen, and V. Mustonen, Strong selective environments determine evolutionary outcome in time-dependent fitness seascapes, Evolution Letters 6, 266 (2022).

[22] M. J. Falk, J. Wu, A. Matthews, V. Sachdeva, N. Pashine, M. L. Gardel, S. R. Nagel, and A. Murugan, Learning to learn by using nonequilibrium training protocols for adaptable materials, Proceedings of the National Academy of Sciences 120, e2219558120 (2023).

[23] M. Ruiz-Garc´ıa, A. J. Liu, and E. Katifori, Tuning and jamming reduced to their minima, Physical Review E 100, 052608 (2019).

[24] L. Wu, F. Tian, Y. Xia, Y. Fan, T. Qin, J.-H. Lai, and T.-Y. Liu, Learning to teach with dynamic loss functions, in Advances in Neural Information Processing Systems, Vol. 31 (2018) pp. 6466–6477.

[25] D. Filan, S. Casper, S. Hod, C. Wild, A. Critch, and S. Russell, Clusterability in neural networks, arXiv preprint arXiv:2103.03386 (2021).

[26] R. Csord´as, S. van Steenkiste, and J. Schmidhuber, Are neural nets modular? Inspecting functional modularity through diferentiable weight masks, in International Conference on Learning Representations (2021).

[27] S. Hod, D. Filan, S. Casper, A. Critch, and S. Russell, Quantifying local specialization in deep neural networks, arXiv preprint arXiv:2110.08058 (2021).

[28] N. Elhage, T. Hume, C. Olsson, N. Schiefer, T. Henighan, S. Kravec, Z. Hatfield-Dodds, R. Lasenby, D. Drain, C. Chen, R. Grosse, S. McCandlish, J. Kaplan, D. Amodei, M. Wattenberg, and C. Olah, Toy models of superposition, Transformer Circuits Thread (2022), arXiv:2209.10652.

[29] A. Scherlis, K. Sachan, A. S. Jermyn, J. Benton, and B. Shlegeris, Polysemanticity and capacity in neural networks, arXiv preprint arXiv:2210.01892 (2022).

[30] K. J. Lang and M. J. Witbrock, Learning to tell two spirals apart, in Proceedings of the 1988 Connectionist Models Summer School (Morgan Kaufmann, 1988) pp. 52–59.

[31] X. Glorot and Y. Bengio, Understanding the dificulty of training deep feedforward neural networks, in Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics, Proceedings of Machine Learning Research, Vol. 9 (PMLR, 2010) pp. 249–256.

[32] D. P. Kingma and J. Ba, Adam: A method for stochastic optimization, in International Conference on Learning Representations (2015).

[33] K. He, X. Zhang, S. Ren, and J. Sun, Deep residual learning for image recognition, in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2016) pp. 770–778.

[34] A. Krizhevsky, Learning Multiple Layers of Features from Tiny Images, Tech. Rep. (University of Toronto, 2009).

[35] G. Gur-Ari, D. A. Roberts, and E. Dyer, Gradient descent happens in a tiny subspace, arXiv preprint arXiv:1812.04754 (2018).

[36] J. Frankle and M. Carbin, The lottery ticket hypothesis: Finding sparse, trainable neural networks, in International Conference on Learning Representations (2019).

[37] N. Kashtan and U. Alon, Spontaneous evolution of modularity and network motifs, Proceedings of the National Academy of Sciences 102, 13773 (2005).

[38] M. Parter, N. Kashtan, and U. Alon, Facilitated variation: How evolution learns from past environments to generalize

to new environments, PLoS Computational Biology 4, e1000206 (2008).

[39] V. Sachdeva, K. Husain, J. Sheng, S. Wang, and A. Murugan, Tuning environmental timescales to evolve and maintain generalists, Proceedings of the National Academy of Sciences 117, 12693 (2020).

[40] N. Ratier-Werbin, S. Lozano, and M. Ruiz-Garc´ıa, Dynamical Batching: code and configuration files, Zenodo (2026), version v1.0.1, doi:10.5281/zenodo.23471602; all versions: doi:10.5281/zenodo.23349474.

[41] A. Ingrosso and S. Goldt, Data-driven emergence of convolutional structure in neural networks, Proceedings of the National Academy of Sciences 119, e2201854119 (2022).

[42] V. Shankar, A. Fang, W. Guo, S. Fridovich-Keil, J. Ragan-Kelley, L. Schmidt, and B. Recht, Neural kernels without tangents, in Proceedings of the 37th International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 119 (PMLR, 2020) pp. 8614– 8623.