# From Biological Curves to Soft Modular Robots: Learning-Based Inverse Design with Uncertainty-Guided Refinement

Dixiang Wang, Minghao Zha, Arifur Rahman, Yinhong Qin, Yitao Jiang, Shiyang Lu, Zebing Mao, Muhao Chen, Luyang Zhao

Abstract— Animals exhibit diverse shape-changing behaviors, from the undulatory swimming of eels to the arching movements of inchworms. Reproducing these biological shapes with modular soft robots offers a promising approach to developing versatile bioinspired robotic systems. However, translating a desired shape into a feasible robotic configuration remains challenging. Existing cable-driven modular soft robots rely on computationally expensive physics-based modeling and combinatorial search for inverse design, making it difficult to efficiently reproduce diverse target shapes. To address this challenge, we present a learning-based real-to-sim-to-real framework that translates biological curves into robotic configurations through surrogateassisted inverse design and uncertainty-guided refinement. The proposed method combines learned inverse prediction with targeted candidate exploration, using a fast forward surrogate to evaluate and select suitable designs. The learned forward surrogate reduces deformation prediction time from 1.53 × 10<sup>5</sup> ms to 0.0008 ms, achieving approximately a 1.9 × 10<sup>8</sup>-fold speedup. At the same time, the proposed inverse design method reduces design time from $\mathbf { 1 . 9 9 \times } \mathbf { \bar { 1 0 ^ { 6 } } }$ ms to 4.18 ms. On 5,000 test samples generated from previously unseen threading patterns, the method achieves a 93% shape reconstruction success rate under surrogate-based evaluation. Physical demonstrations with four bioinspired robots further illustrate the translation of biological curves into realizable robotic shapes.

## I. INTRODUCTION

Nature exhibits a rich diversity of shape-changing behaviors, from the swimming motions of eels [1] to the slithering of snakes [2] and the arching movement of inchworms [3]. Despite their distinct biological mechanisms, many of these deformations can be represented as geometric curves, provid ing a common description of diverse biological shapes [4], [5]. This observation motivates a real-to-sim-to-real approach to bio-inspired robot design, in which shapes observed in nature are abstracted into target curves, translated into feasible robotic configurations through simulation, and reproduced by physical robots, as illustrated in Fig. 1. Soft robots are particularly suitable for this purpose, as their compliant structures enable flexible and adaptable deformations [6]–[9]. However, translating a desired geometric curve into a physically realizable robotic shape requires identifying appropriate structural configurations and actuation parameters [10]–[12].

SoftSnap, a previously developed cable-driven modular soft robotic platform, provides one possible approach to realizing such target curves [13]. Each untethered module integrates onboard actuation, control, and power, enabling different bending curves through variations in cable-threading patterns and cable lengths. Multiple modules can be assembled using customized snap-together connectors to construct more complex robotic shapes. To predict the resulting deformation, SoftSnap employs a physics-based quasi-static forward model that uses iterative numerical optimization to minimize bending energy under a cable-length constraint. Although this model captures the deformation behavior, repeated forward evaluations can be computationally expensive (152.75 s in average). The original platform also supports inverse design through a search-based approach that enumerates possible threading configurations, filters geometrically infeasible candidates, and evaluates the remaining configurations using the physics-based forward model. However, the number of possible threading configurations grows combinatorially with the number of ribs and available holes. Consequently, the inverse design process becomes computationally expensive due to both the large number of candidate configurations and the repeated numerical optimization required for forward evaluation. Restricting the number of evaluated candidates can reduce computational cost but may overlook configurations that better reproduce the target shape.

Learning-based methods offer a promising approach to reducing the computational cost of repeated physics-based evaluations. Previous studies have used neural networks to predict soft robot deformation [14]–[16], estimate cable tensions for desired tip positions in soft manipulators [17], and perform inverse design using learned forward surrogates [18]–[21]. However, inverse design for the SoftSnap platform presents additional challenges due to its mixed discretecontinuous design space and non-unique inverse mapping [22]. Specifically, different combinations of cable-threading patterns and cable lengths can produce identical or nearly identical shapes, represented by their bending-angle vectors. This one-to-many relationship makes direct inverse prediction challenging [23], [24]. Conventional regression methods, such as those using mean squared error, may average cable lengths associated with different valid solutions, while selecting the most probable hole at each rib may combine threading choices from different feasible configurations. As a result, the predicted threading pattern and cable length may fail to reproduce the target shape accurately. Exhaustively evaluating alternative configurations, however, would reintroduce substantial computational costs. These challenges motivate an approach that combines fast inverse prediction with targeted candidate exploration and efficient surrogatebased evaluation.

To enable efficient inverse design within this real-to-simto-real workflow, we propose a surrogate-assisted approach that combines learning-based prediction with uncertaintyguided candidate refinement. First, a large-scale dataset is generated using the physics-based quasi-static model and curated to remove invalid and unstable samples. A forward surrogate is then trained to efficiently predict robot deformation given a cable-threading pattern and target cable length. Next, an inverse neural network is trained to jointly predict the cable-threading pattern and cable length from a target curve represented by a bending-angle vector. The inverse model is initially trained using supervised classification and regression losses, followed by end-to-end fine-tuning through a pretrained, frozen forward surrogate to minimize angular reconstruction error. Once trained, the inverse model generates an initial design, which is further improved through uncertainty-guided candidate refinement. Specifically, uncertain threading positions are identified using classificationlogit margins, and alternative configurations are explored with adaptive search budgets. For more uncertain cases, local cable-length adjustments are also introduced. The forward surrogate then evaluates the resulting candidates and selects the configuration with the lowest predicted reconstruction error. Finally, physical demonstrations of biologically inspired soft robots illustrate the framework’s potential for translating target geometric curves into realizable robotic shapes.

![](images/f5a086dd44af700c6dabf269576a3fae56a056eb62c5fdf3286787d9b0ac5d64.jpg)  
Fig. 1. Overview of the animal-inspired real-to-sim-to-real design framework. Left: Key frames are extracted from real videos, and the curve is projected onto our skeleton to obtain an angle sequence. Middle: Our trained inverse model, combined with the refinement filter, maps the angle sequence to a threading pattern and target string length. Right: The threading pattern and target string length are applied to the physical skeleton, which is controlled to imitate the creature’s motion.

The main contributions of this work are summarized as follows: (a) A data-driven forward modeling framework that integrates physics-based data generation, data curation, and a learned forward surrogate to efficiently predict robot deformation without repeated numerical optimization during inference. (b) A surrogate-assisted inverse design approach that combines learning-based inverse prediction with uncertaintyguided candidate refinement to selectively explore alternative cable-threading patterns and cable lengths, improving targetshape reconstruction without exhaustive search.

## II. PROBLEM FORMULATION

This study focuses on the quasi-static deformation of a single cable-driven soft robotic module from the previously developed SoftSnap platform [13]. As illustrated in Fig. 2, the module consists of 12 ribs arranged along a flexible central spine and actuated by a single cable. By varying the cable-threading pattern and shortening the cable, relative rotations are induced between adjacent ribs, producing different bending shapes.

![](images/10ad393d2bcb24ad269a0b479e233b8d172cd7542a3dd2228be0fb570be5cad0.jpg)  
Fig. 2. Geometry and cable routing of a SoftSnap module (dimensions in mm). (a) Overview of the 12-rib skeleton. (b) Top view showing the overall dimensions and rib spacing. (c) Geometry and positions of the six threading holes on each rib. (d) Geometry of the two guide holes on the central spine between adjacent ribs. (e) Example of an R3-to-R0 cable-routing transition.

Each rib contains six physical threading holes, labeled L1–L3 and R1–R3, with two additional guide holes located along the central spine between adjacent ribs. The cablethreading pattern is denoted by $\mathbf { p } = ( p _ { 1 } , \ldots , p _ { 1 2 } )$ , where $p _ { i } \in \mathcal { P } = \{ L 0 , L 1 , L 2 , L 3 , R 0 , R 1 , R 2 , R 3 \}$ specifies one of the eight routing choices at the ith rib. Unlike the six physical threading holes, L0 and R0 represent effective nearcenter routing choices designed to produce bending angles close to zero between adjacent ribs. In the physical system, these near-center routing choices are implemented using the innermost rib holes and the two guide holes along the central spine. The cable passes through the first guide hole to the opposite side of the rib and then through the second guide hole to return to its original side. For example, an R3-to-R0 transition is implemented using the R1 hole and the two guide holes. Similarly, an R0-to-L0 transition is physically implemented as an R0-to-R0 routing, while an L0-to-R0 transition is implemented as an L0-to-L0 routing, preserving the original cable-routing side.

The target cable length is denoted by L. The resulting shape is represented by an 11-dimensional bending-angle vector $\pmb { \alpha } = ( \alpha _ { 1 } , \dots , \alpha _ { 1 1 } )$ , where $\alpha _ { i }$ denotes the relative bending angle between the ith and (i + 1)th ribs.

## A. Physics-Based Forward Modeling

The forward problem predicts the resulting robot shape given a cable-threading pattern p and target cable length L. This mapping is expressed as:

$$
\begin{array} { r } { \alpha = f _ { \mathrm { p h y s } } ( \mathbf { p } , L ) , } \end{array}\tag{1}
$$

where $f _ { \mathrm { p h y s } }$ denotes the physics-based quasi-static forward model. Following the original SoftSnap model [13], the equilibrium configuration is obtained by minimizing the bending energy subject to a fixed total cable-length constraint:

$$
\operatorname* { m i n } _ { \alpha } \quad \sum _ { i = 1 } ^ { 1 1 } \alpha _ { i } ^ { 2 } \quad \mathrm { s . t . } \quad \quad \sum _ { i = 1 } ^ { 1 1 } \ell _ { i } = L ,\tag{2}
$$

where $\ell _ { i }$ denotes the cable-segment length between the ith and (i + 1)th ribs, determined by their threading choices and relative bending angle. The resulting equilibrium bendingangle vector α defines the output of the physics-based forward model.

## B. Inverse Design Problem

Given a target shape represented by a bending-angle vector $\ b { \alpha } ^ { * }$ , the inverse design problem seeks a cable-threading pattern p and target cable length L that reproduce the desired deformation. Unlike the forward problem, the inverse mapping is non-unique: different combinations of threading patterns and cable lengths may produce identical or nearly identical bending-angle vectors. Moreover, some target shapes may be physically unattainable due to geometric and actuation constraints. We therefore formulate inverse design as a constrained optimization problem:

$$
\left( \mathbf { p } ^ { * } , L ^ { * } \right) \in \arg \operatorname* { m i n } _ { ( \mathbf { p } , L ) \in \mathcal { D } } \mathrm { R M S E } \left( f _ { \mathrm { p h y s } } ( \mathbf { p } , L ) , \alpha ^ { * } \right) ,\tag{3}
$$

where D denotes the admissible design space containing feasible threading configurations and cable lengths. The angular reconstruction error is defined as $\mathrm { R M S E } ( \alpha , \alpha ^ { * } ) =$ $\textstyle { \sqrt { \frac { 1 } { 1 1 } \sum _ { i = 1 } ^ { 1 1 } ( \alpha _ { i } - \alpha _ { i } ^ { * } ) ^ { 2 } } }$

The objective is to identify a feasible design that accurately reproduces the target shape, rather than recovering a unique set of design parameters. However, directly solving this problem requires searching a large mixed discrete-continuous design space and repeatedly evaluating the physics-based forward model. This motivates the surrogate-assisted inverse design approach presented in the following section.

## III. METHODOLOGY

The proposed framework consists of four main components: physics-based data generation and curation, forward surrogate modeling, two-stage inverse model training, and uncertainty-guided candidate refinement.

## A. Data Generation and Curation

To train the forward and inverse models, a large-scale dataset is generated using the physics-based quasi-static model described in Section II. Each sample consists of a cable-threading pattern p, a target cable length $L ,$ and the corresponding bending-angle vector α. For each threading pattern, the cable is progressively shortened from its initial length $L _ { \mathrm { m a x } }$ through a sequence of contraction steps. At each step, the constrained optimization problem in Eq. 2 is solved using the equilibrium configuration from the previous step as the initial guess. This procedure generates multiple deforma tion states for each threading pattern. However, the physicsbased optimization does not explicitly enforce all geometric feasibility constraints. Consequently, some generated states may exhibit excessive bending, collisions between adjacent ribs, or numerical instability. To improve dataset quality, a curation process is introduced to remove geometrically invalid, redundant, and unstable samples.

![](images/3ead7e5eb7f3a285e0f6b9d169c2939ebbd72a3ee09e8bff6d533a5ea3ad4251.jpg)

![](images/feaa78f09eb266e0100a5e2200335fce3e55b76005c1162474fb95b98cef5927.jpg)  
Fig. 3. Illustrations of $0 . 2 \mathrm { ~ < ~ } \mathrm { R M S E }$ and RMSE $< 0 . 0 3$ . Black and gray line segments represent ground-truth and predicted rods, respectively; red and blue curves represent the corresponding ground-truth and predicted arcs.

First, each bending angle is constrained to $\alpha _ { i } \in$ [−1.5, 1.5] rad to exclude configurations with excessive bending that may cause collisions between adjacent ribs. In addition, changes in bending-angle vectors between consecutive contraction steps are measured as $\begin{array} { r l } { d _ { t } } & { { } = } \end{array}$ $\begin{array} { r } { \sqrt { \frac { 1 } { 1 1 } \sum _ { i = 1 } ^ { 1 1 } \Big ( \alpha _ { i } ^ { ( t ) } - \alpha _ { i } ^ { ( t - 1 ) } \Big ) ^ { 2 } } } \end{array}$ . As shown in Fig. 3, samples with $d _ { t } < 0 . \dot { 0 3 }$ rad are considered redundant due to negligible deformation changes, while samples with $d _ { t } \geq 0 . 2$ rad are treated as potentially numerically unstable. Only samples satisfying $0 . 0 3 \leq d _ { t } < 0 . 2$ rad are retained; otherwise, the remaining contraction steps for the current threading pattern are terminated.

![](images/3c4f9550dec90fc4169531a0685fcddf73135b0791454ea845353aaaa86d8114.jpg)  
Fig. 4. Forward surrogate training framework.

## B. Forward Surrogate Modeling

The forward surrogate approximates the physics-based mapping from cable-threading patterns and cable lengths to bending-angle vectors, avoiding repeated numerical optimization during inference. As illustrated in Fig. 4, the discrete threading pattern is represented using one-hot encoding. Each of the 12 rib positions has eight possible routing choices, corresponding to L0–L3 and R0–R3 as defined in Section II. The pattern is encoded as a matrix $\mathrm { ~ { ~ \bf ~ P ~ } ~ } \in$ $\{ 0 , 1 \} ^ { 8 \times 1 2 }$ , where each row represents the selected routing choice at one rib. The matrix is flattened and concatenated with the continuous cable length L to form the forward model input. Before concatenation, $L$ is standardized using z-score normalization as follows:

$$
{ \cal L } _ { i n p u t } = \frac { L - \mu _ { L } } { \sigma _ { L } + 1 0 ^ { - 8 } }\tag{4}
$$

where $\mu _ { L }$ and $\sigma _ { L }$ denote the mean and standard deviation, respectively, of the string lengths in the training set.

A multilayer perceptron (MLP) is employed to learn the nonlinear mapping between the design parameters and the resulting deformation:

$$
\begin{array} { r } { \hat { \alpha } = F _ { \phi } ( \mathbf { P } , L ) , } \end{array}\tag{5}
$$

where $\phi$ denotes the trainable parameters of the forward surrogate and $\hat { \pmb { \alpha } } \in \mathbb { R } ^ { 1 1 }$ is the predicted bending-angle vector.

The forward model is trained using the simulated bendingangle vectors as supervision. The training loss is defined as the mean squared error between the predicted and simulated bending angles:

$$
\mathcal { L } _ { \mathrm { f o r w a r d } } = \frac { 1 } { 1 1 } \left\| F _ { \phi } ( \mathbf { P } , L ) - \alpha \right\| _ { 2 } ^ { 2 } .\tag{6}
$$

Once trained, the forward surrogate provides efficient deformation predictions and is used for both inverse model fine-tuning and candidate evaluation.

## C. Two-Stage Inverse Model Training

Given a target bending-angle vector $\ b { \alpha } ^ { * }$ , the inverse model aims to jointly predict the corresponding cable-threading pattern and target cable length. Since the design parameters include both discrete and continuous variables, the inverse model performs classification and regression.

The inverse network is expressed as:

$$
\begin{array} { r } { ( { \bf Z } , \hat { L } ) = I _ { \boldsymbol { \theta } } ( { \boldsymbol \alpha } ^ { * } ) , } \end{array}\tag{7}
$$

where θ denotes the trainable parameters of the inverse network, $\mathbf { Z } \in \mathbb { R } ^ { 1 2 \times 8 }$ contains the routing classification logits, and $\hat { L }$ is the predicted cable length.

The inverse model is trained in two stages.

Stage 1: Supervised Pretraining. The inverse network is first trained using the threading patterns and cable lengths from the generated dataset. Cross-entropy loss is employed for routing classification, while a regression loss is used for cable-length prediction. The supervised training objective is expressed as:

$$
\mathcal { L } _ { \mathrm { s u p } } = \frac { 1 } { 1 2 } \sum _ { i = 1 } ^ { 1 2 } \mathrm { C E } ( \mathbf { Z } _ { i } , p _ { i } ) + \lambda _ { L } \mathcal { L } _ { \mathrm { l e n g t h } } ( \hat { L } , L ) ,\tag{8}
$$

where CE denotes the cross-entropy loss for the categorical routing choices, and $\lambda _ { L } = 1$ balances the classification and regression objectives.

Stage 2: End-to-End Fine-Tuning. To directly optimize shape reconstruction accuracy, the pretrained forward surrogate is incorporated into the inverse training process. During this stage, the forward surrogate parameters $\phi$ are frozen, while the inverse model parameters θ remain trainable.

To maintain differentiability, the predicted routing logits are converted into a continuous probability matrix using Softmax:

$$
P _ { i , k } ^ { \mathrm { s o f t } } = \frac { \exp ( Z _ { i , k } ) } { \sum _ { j = 1 } ^ { 8 } \exp ( Z _ { i , j } ) } .\tag{9}
$$

The resulting probability matrix and predicted cable length are passed through the frozen forward surrogate:

$$
\hat { \alpha } = F _ { \phi } ( \mathrm { \bf P } ^ { \mathrm { s o f t } } , \hat { L } ) .\tag{10}
$$

The angular reconstruction loss is defined as:

$$
\mathcal { L } _ { \mathrm { r e c o n } } = \frac { 1 } { 1 1 } \left. \hat { \pmb { \alpha } } - \pmb { \alpha } ^ { * } \right. _ { 2 } ^ { 2 } .\tag{11}
$$

The reconstruction loss is backpropagated through the frozen forward surrogate to update the inverse network. This allows the inverse model to be optimized according to the resulting deformation rather than solely matching the design parameters recorded in the training dataset.

![](images/24dcc78d83419d32670c7b01824a9f309691b95ea029ffe4b3c07d39b6275000.jpg)  
Fig. 5. Two-stage inverse model training. Stage 1 performs supervised pretraining using a combined classification and regression loss. Stage 2 minimizes angular reconstruction error through a pretrained, frozen forward surrogate, updating only the inverse model parameters.

## D. Uncertainty-Guided Candidate Refinement

Once trained, the inverse model generates an initial threading pattern and cable length for a given target shape. However, due to the non-unique inverse mapping, directly accepting a single predicted configuration may result in inaccurate shape reconstruction.

To improve reconstruction accuracy without exhaustive search, an uncertainty-guided candidate refinement strategy is introduced. Rather than exploring the entire design space, this strategy focuses on routing positions where the inverse model exhibits greater prediction ambiguity.

For each rib position i, the inverse network produces eight classification logits $Z _ { i , k }$ . The uncertainty of the routing prediction is estimated using the gap between the two largest logits:

$$
g _ { i } = Z _ { i , ( 1 ) } - Z _ { i , ( 2 ) } ,\tag{12}
$$

where $Z _ { i , ( 1 ) }$ and $Z _ { i , ( 2 ) }$ denote the largest and second-largest logits, respectively. A smaller gap indicates greater ambiguity between the two most likely routing choices. We select the R ribs with the highest uncertainty scores $g _ { i }$ for further search. For each selected rib, we then choose the top-k candidate holes based on $Z _ { i , ( k ) }$ . The resulting search space contains $k ^ { R }$ candidate configurations.

In addition to threading-pattern refinement, local cablelength adjustments are introduced for hard samples. The candidate lengths are generated as:

$$
L ^ { ( j ) } = \hat { L } + \Delta L , \quad \Delta L \in \{ 0 , \pm 4 \ m m , \pm 8 \ m m \} .\tag{13}
$$

The resulting threading patterns and cable lengths form a candidate set C that includes the initial inverse prediction.

For each candidate $( \mathbf { p } ^ { ( j ) } , L ^ { ( j ) } ) \in \mathcal { C }$ , the learned forward surrogate reconstructs the corresponding bending-angle vector:

$$
\begin{array} { r } { \hat { \pmb { \alpha } } ^ { ( j ) } = F _ { \phi } \left( \mathrm { o n e h o t } ( \mathbf { p } ^ { ( j ) } ) , L ^ { ( j ) } \right) . } \end{array}\tag{14}
$$

The final configuration is selected as the candidate with

the lowest surrogate-predicted angular reconstruction error:

$$
( \mathbf { p } ^ { \mathrm { r e f } } , L ^ { \mathrm { r e f } } ) \in \arg \operatorname* { m i n } _ { ( \mathbf { p } , L ) \in \mathcal { C } } \mathrm { R M S E } \left( F _ { \phi } ( \mathrm { o n e h o t } ( \mathbf { p } ) , L ) , \alpha ^ { * } \right) .\tag{15}
$$

By selectively exploring uncertain routing positions and evaluating candidates through the learned forward surrogate, the proposed method refines the initial inverse prediction without exhaustively evaluating the combinatorial design space.

## IV. EXPERIMENTS

The experiments evaluate three questions: (1) whether a learned forward surrogate can accurately and efficiently replace repeated physics-based deformation predictions; (2) whether the proposed inverse design method can reconstruct target shapes for previously unseen threading patterns; and (3) how supervised pretraining and uncertainty-guided candidate refinement contribute to inverse design performance.

## A. Experimental Setup and Dataset

The dataset is generated using the physics-based quasistatic SoftSnap model [13] and the data curation procedure described in Section III. Each sample contains a threading pattern p, target cable length $L ,$ and corresponding bendingangle vector α $\mathbf { \Lambda } \in \mathbb { R } ^ { 1 1 }$ . The training dataset contains approximately 5M samples from around 115k distinct threading patterns, where the data were split into training and validation sets at a ratio of 95:5. An independent test set of 5k samples is generated from threading patterns excluded from training. Since each pattern can be evaluated at multiple cable lengths, the pattern-level separation prevents exact threading configurations from appearing in both the training and test sets.

To examine data efficiency, models are additionally trained using subsets of 10k, 100k, 500k, 1M, and 3M samples, in addition to the full 5-million-sample dataset. The test set is held fixed across training sizes.

The primary evaluation metric is the angular root-meansquare error (RMSE) defined in Section II. Shape reconstruction is considered successful when the error is below $\epsilon = 0 . 0 3$ rad, as the reconstructed curves closely resemble the target curves within this threshold. For the forward task, predictions are compared with the equilibrium angles generated by the physics-based simulator. For the inverse task, the reported model-comparison results use the learned forward surrogate to reconstruct the angles produced by predicted threading patterns and cable lengths; these results are therefore referred to as surrogate-based inverse evaluations.

Runtime is measured separately for a single forward prediction and for the complete inverse design procedure, including candidate generation and evaluation. All measurements are performed on a system equipped with an NVIDIA H200 GPU and an Intel Xeon Platinum 8580 CPU, using Python 3.11.16, PyTorch 2.11.0, and CUDA 12.8, with a batch size of 131,072.

## B. Forward Surrogate Evaluation

The forward task predicts the 11-dimensional bendingangle vector from a threading pattern and cable length. Six model families are compared: k-nearest neighbors (kNN), CNN, XGBoost, Transformer encoder, Transformer decoder, and MLP. All models are evaluated on the held-out test set. Table I summarizes their accuracy, reported inference time, and model size. Among the evaluated learning-based models, the MLP obtains the lowest angular RMSE (0.0077 rad) and highest success rate (96.02%), and is therefore selected as the forward surrogate used in subsequent inverse design experiments. Compared with the Transformer decoder, its RMSE is reduced from 0.0186 to 0.0077 rad. Compared with the original physics-based forward optimizer, which requires 152.75 seconds for a matched $( \mathbf { p } , L )$ query, the learned MLP surrogate requires 0.0008 milliseconds, corresponding to a $1 . 9 \times 1 0 ^ { 8 }$ -fold speedup. Increasing the training set from 10k to 5M samples improved prediction success from approximately 74% to 93%, with performance gradually saturating at approximately 96% as the dataset size increased to 5 million samples.

TABLE I  
PERFORMANCE COMPARISON OF FORWARD SURROGATE MODELS.
<table><tr><td>Model</td><td>RMSE (rad)</td><td>Success Rate</td><td>Time (ms)</td><td>Size (MB)</td></tr><tr><td>kNN (Baseline)</td><td>0.2016</td><td>0.0778</td><td>12.2777</td><td>231.9694</td></tr><tr><td>CNN</td><td>0.0219</td><td>0.8454</td><td>0.1534</td><td>1.194</td></tr><tr><td>XGBoost</td><td>0.0346</td><td>0.6520</td><td>47.6386</td><td>206.38</td></tr><tr><td>Transformer (Enc)</td><td>0.0325</td><td>0.7126</td><td>0.0490</td><td>0.067</td></tr><tr><td>Transformer (Dec)</td><td>0.0186</td><td>0.8692</td><td>0.0471</td><td>0.148</td></tr><tr><td>MLP (Proposed)</td><td>0.0077</td><td>0.9602</td><td>0.0008</td><td>130.592</td></tr></table>

## C. Inverse Design Evaluation

Inverse design predicts a threading pattern and cable length from a target bending-angle vector. Since multiple parameter configurations can reproduce similar shapes, the evaluation focuses on the reconstructed shape instead of exact recovery of the original threading labels. The returned discrete configurations are evaluated using the forward surrogate and the angular RMSE defined in Section II.

Table II compares the original SoftSnap inverse design method with several learning-based approaches, including kNN, a decoder-only Transformer, and MLPs with different training strategies and candidate refinement. The original SoftSnap method achieves an average runtime of $1 . 9 9 \times 1 0 ^ { 6 }$ ms per target and a reconstruction success rate of 100%. Among the supervised-only baselines, the decoder-only Transformer achieves the highest success rate of 78.08%, which increases to 85.06% with candidate refinement. The two-stage MLP achieves reconstruction success rates of 79.44% without refinement and 93.18% with candidate refinement. With $R = 3$ and $k = 6 ,$ , the refined model achieves the highest success rate among the methods compared in this table, with an average runtime of 4.182 ms per sample. Our experiments show that increasing R and k can further improve the success rate. For example, setting $R = 6$ and $k = 8$ increases the success rate to 94.48%, but at a substantially higher runtime of 3967.094 ms per sample. These results suggest that supervised initialization and candidate refinement provide complementary improvements in inverse design performance. Fig. 6 illustrates two examples in which candidate refinement corrects inaccurate initial threading choices and improves shape reconstruction.

![](images/fe3f346e11bd5c8dfc5df53524531ae0c990e6c1141279224f3cbaa87a8b5675.jpg)

![](images/7cb454eec32548c899c464bb4bdc48c15f2edf795a480a245db55209d88dc247.jpg)  
Fig. 6. Two test examples with the largest improvement in shape reconstruction error from initial prediction to uncertainty-guided candidate refinement.

## V. PHYSICAL VALIDATION

To evaluate the physical realization of the proposed realto-sim-to-real workflow, four bioinspired robot configurations are considered: inchworm, caterpillar, snake, and brittle star, as shown in Fig. 7. For each case, a representative frame of the biological motion is used to extract and normalize a target curve, which is represented by an 11-dimensional bending-angle vector. The trained inverse model, followed by uncertainty-guided candidate refinement, determines a cablethreading pattern and target cable length for reproducing the desired shape. The selected configuration is then implemented using SoftSnap modules.

To examine whether the resulting robotic configurations can generate locomotion, the physical robots are driven using prescribed cyclic motor commands. The inchwormand caterpillar-inspired robots each use one motor, while the snake-inspired robot uses three modules with three independently commanded motors. The brittle star-inspired robot uses two motors driven by the same signal. The corresponding motor commands are shown in Fig. 8. These commands are specified for the locomotion demonstrations; the inverse design model predicts cable-threading patterns and target cable lengths rather than full time-varying control sequences.

TABLE II  
PERFORMANCE COMPARISON OF INVERSE DESIGN METHODS.
<table><tr><td>Method</td><td>Search strategy</td><td>RMSE (rad)</td><td>Success</td><td>Time (ms)</td><td>Size (MB)</td></tr><tr><td>Original SoftSnap inverse search</td><td>Physics-based search</td><td>0.0973</td><td>1.000</td><td> $\overline { { 1 . 9 9 \times 1 0 ^ { 6 } } }$ </td><td></td></tr><tr><td>kNN</td><td>None</td><td>0.1297</td><td>0.1246</td><td>52.723</td><td>2059.937</td></tr><tr><td>CNN</td><td>None</td><td>0.0459</td><td>0.4496</td><td>0.220</td><td>0.532</td></tr><tr><td>XGBoost</td><td>None</td><td>0.0748</td><td>0.2438</td><td>0.212</td><td>59.354</td></tr><tr><td>Transformer (Encoder)</td><td>None</td><td>0.0413</td><td>0.5052</td><td>0.002</td><td>1.571</td></tr><tr><td>Transformer (Decoder)</td><td>None</td><td>0.0250</td><td>0.7808</td><td>0.084</td><td>4.871</td></tr><tr><td>MLP (one-stage)</td><td>None</td><td>0.0346</td><td>0.6290</td><td>0.016</td><td>132.426</td></tr><tr><td>MLP (end-to-end)</td><td>None</td><td>0.0304</td><td>0.6990</td><td>0.020</td><td>132.426</td></tr><tr><td>MLP (two-stage)</td><td>None</td><td>0.0230</td><td>0.7944</td><td>0.124</td><td>132.426</td></tr><tr><td>Transformer (Decoder) + refinement</td><td>Uncertainty-guided</td><td>0.0179</td><td>0.8506</td><td>3.602</td><td>4.871</td></tr><tr><td>MLP (end-to-end) + refinement</td><td>Uncertainty-guided</td><td>0.0150</td><td>0.8760</td><td>3.277</td><td>132.426</td></tr><tr><td>MLP (two-stage) + refinement [proposed]</td><td>Uncertainty-guided</td><td>0.0099</td><td>0.9318</td><td>4.182</td><td>132.426</td></tr></table>

![](images/3b331c9a433d08addcb0f92ea326f998fb366bad546126dbaffe659e33622c24.jpg)  
Fig. 7. Real-to-sim-to-real physical validation of four bioinspired robots: (a) inchworm, (b) caterpillar, (c) snake, and (d) brittle star. Biological shapes are abstracted into target curves (Real), mapped to cable-threading patterns and target lengths through inverse design (Sim), and realized using physical SoftSnap modules (Real).

Five locomotion trials were conducted for each robot, with three actuation cycles per trial for the inchworm and snake and five for the caterpillar and brittle star. Each cycle lasted 1.6 s, except for the snake (3.0 s). The average speeds were 4.25, 4.08, 1.09, and 9.50 cm/s, corresponding to 6.80, 6.52, 3.27, and 15.20 cm/cycle for the inchworm, caterpillar, snake, and brittle star, respectively. These results demonstrate locomotion capability across different robotic configurations. These demonstrations illustrate the feasibility of translating biological target curves into physically realizable SoftSnap configurations capable of producing bioinspired deformations and locomotion. The evaluation focuses on representative shape realization and physical motion rather than on reproducing complete biological gait trajectories.

![](images/81ff7f4611a36236316e093f78248f98237cc4fcc80ac668ba83002fcce3bb38.jpg)  
(a)

![](images/af7135198f9ff2d9d88c1082459f15c0746b12829128ccc3c9d455d34dbd566d.jpg)  
(b)  
Fig. 8. Motor command signals for physical validation. (a) Commands for the inchworm-, caterpillar-, and brittle star-inspired robots; the two brittle star motors receive the same signal. (b) Three independently commanded motors for the snake-inspired robot.

## VI. CONCLUSION AND FUTURE WORK

This paper presented a learning-based real-to-sim-to-real framework for translating biological curves into physically realizable soft modular robots. Building on the SoftSnap platform, the proposed approach combines a learned forward surrogate with an inverse model and uncertainty-guided candidate refinement to address the computational challenges of physics-based optimization and combinatorial inverse search. This enables efficient inverse design without exhaustively searching all possible threading configurations.

Experiments on previously unseen threading patterns demonstrate that the proposed refinement improves the reconstruction success rate from 79.44% to 93.18% under surrogate-based evaluation. With an average runtime of 4.182 ms per sample, the proposed inverse design method achieves an approximately $4 . 7 6 \times 1 0 ^ { 5 }$ -fold speedup over the original physics-based approach. The learned forward surrogate further accelerates deformation prediction by more than eight orders of magnitude, enabling rapid candidate evaluation. Physical demonstrations with four bioinspired robots illustrate the feasibility of translating biological target curves into physically realizable soft modular robots. The current framework focuses on quasi-static shape reconstruction within the studied SoftSnap configuration space, without accounting for the effects of external forces such as gravity and friction or the robot’s dynamics. Future work will extend the framework to incorporate these physical effects into dynamic simulations, enabling more realistic predictions of robot motion and interactions with the environment.

## REFERENCES

[1] E. D. Tytell, “Kinematics and hydrodynamics of linear acceleration in eels, anguilla rostrata,” Proceedings of the Royal Society B: Biological Sciences, vol. 271, no. 1557, pp. 2535–2540, 12 2004.

[2] B. C. Jayne, “What defines different modes of snake locomotion?” Integrative and Comparative Biology, vol. 60, no. 1, pp. 156–170, 07 2020.

[3] W. Ausrivong, A. Srisuchinnawong, and P. Manoonpong, “Inchworminspired adaptive multimodal neural control for an autonomous inspection robot,” in 2026 IEEE International Conference on Robotics and Automation (ICRA), 2026, pp. 17 185–17 192.

[4] A. J. Ijspeert, “Biorobotics: Using robots to emulate and investigate agile locomotion,” Science, vol. 346, no. 6206, pp. 196–203, 2014.

[5] W. Hu, G. Z. Lum, M. Mastrangeli, and M. Sitti, “Small-scale softbodied robot with multimodal locomotion,” Nature, vol. 554, no. 7690, pp. 81–85, 2018.

[6] D. Rus and M. T. Tolley, “Design, fabrication and control of soft robots,” Nature, vol. 521, no. 7553, pp. 467–475, 2015.

[7] X. Dong, X. Luo, H. Zhao, C. Qiao, J. Li, J. Yi, L. Yang, F. J. Oropeza, T. S. Hu, Q. Xu et al., “Recent advances in biomimetic soft robotics: fabrication approaches, driven strategies and applications,” Soft Matter, vol. 18, no. 40, pp. 7699–7734, 2022.

[8] C. Laschi, B. Mazzolai, and M. Cianchetti, “Soft robotics: Technologies and systems pushing the boundaries of robot abilities,” Science Robotics, vol. 1, no. 1, p. eaah3690, 2016.

[9] L. Zhao, Y. Jiang, M. Chen, K. Bekris, and D. Balkcom, “Modular shape-changing tensegrity-blocks enable self-assembling robotic structures,” Nature Communications, vol. 16, no. 1, p. 5888, 2025.

[10] F. Chen, Z. Song, S. Chen, G. Gu, and X. Zhu, “Morphological design for pneumatic soft actuators and robots with desired deformation behavior,” IEEE Transactions on Robotics, vol. 39, no. 6, pp. 4408– 4428, 2023.

[11] R. J. WebsterIII and B. A. Jones, “Design and kinematic modeling of constant curvature continuum robots: A review,” The International Journal of Robotics Research, vol. 29, no. 13, pp. 1661–1683, 2010.

[12] G. Maloisel, E. Knoop, C. Schumacher, and M. Bacher, “Automated¨ routing of muscle fibers for soft robots,” IEEE Transactions on Robotics, vol. 37, no. 3, pp. 996–1008, 2021.

[13] L. Zhao, Y. Jiang, C.-Y. She, M. Chen, and D. Balkcom, “Softsnap: Rapid prototyping of untethered soft robots using snap-together modules,” Soft Robotics, vol. 12, no. 6, pp. 698–707, 2025.

[14] S. I. Beaber, Z. Liu, and Y. Sun, “Physics-guided deep learning enabled surrogate modeling for pneumatic soft robots,” IEEE Robotics and Automation Letters, vol. 9, no. 12, pp. 11 441–11 448, 2024.

[15] J. M. Bern, Y. Schnider, P. Banzet, N. Kumar, and S. Coros, “Soft robot control with a learned differentiable model,” in 2020 3rd IEEE International Conference on Soft Robotics (RoboSoft), 2020, pp. 417– 423.

[16] T. Navez, E. Menager, P. Chaillou, O. Goury, A. Kruszewski, and ´ C. Duriez, “Modeling, embedded control, and design of soft robots using a learned condensed fem model,” IEEE Transactions on Robotics, vol. 41, pp. 2441–2459, 2025.

[17] M. Giorelli, F. Renda, M. Calisti, A. Arienti, G. Ferri, and C. Laschi, “Neural network and jacobian method for solving the inverse statics of a cable-driven soft arm with nonconstant curvature,” IEEE Transactions on Robotics, vol. 31, no. 4, pp. 823–834, 2015.

[18] D. Tong, Z. Hao, M. Liu, and W. Huang, “Inverse design of snapactuated jumping robots powered by mechanics-aided machine learning,” IEEE Robotics and Automation Letters, vol. 10, no. 2, pp. 1720– 1727, 2025.

[19] J. Liu, J. H. Low, Q. Q. Han, M. Lim, D. Lu, C.-H. Yeow, and Z. Liu, “Simulation data driven design optimization for reconfigurable soft gripper system,” IEEE Robotics and Automation Letters, vol. 7, no. 2, pp. 5803–5810, 2022.

[20] Y. Yao, L. He, and P. Maiolino, “Spada: A toolbox of designing soft pneumatic actuators for shape matching based on surrogate modeling,” Robotics Reports, vol. 2, no. 1, p. rorep.2023.0029, 2024.

[21] S. Yi, X. Bai, A. Singh, J. Ye, M. T. Tolley, and X. Wang, “Codesign of soft gripper with neural physics,” in Proceedings of The 9th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, J. Lim, S. Song, and H.-W. Park, Eds., vol. 305. PMLR, 27–30 Sep 2025, pp. 4313–4327.

[22] N. Ansari, H.-P. Seidel, and V. Babaei, “Mixed integer neural inverse design,” ACM Trans. Graph., vol. 41, no. 4, Jul. 2022.

[23] D. Liu, Y. Tan, E. Khoram, and Z. Yu, “Training deep neural networks for the inverse design of nanophotonic structures,” ACS Photonics, vol. 5, no. 4, pp. 1365–1369, 04 2018.

[24] L. Ardizzone, J. Kruse, S. Wirkert, D. Rahner, E. W. Pellegrini, R. S. Klessen, L. Maier-Hein, C. Rother, and U. Kothe, “Analyzing inverse¨ problems with invertible neural networks,” 2019. [Online]. Available: https://arxiv.org/abs/1808.04730