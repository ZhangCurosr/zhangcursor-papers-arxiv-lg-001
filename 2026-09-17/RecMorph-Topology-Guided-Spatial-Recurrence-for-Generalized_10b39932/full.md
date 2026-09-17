# RecMorph: Topology-Guided Spatial Recurrence for Generalized Morphology Control

Quanrui Rao<sup>1</sup>, Yong Liu<sup>2</sup>, Xueming Xiao<sup>3</sup>, Yingbo Luo<sup>1</sup>, Kun Wu<sup>2</sup>, Zhenyu Xu<sup>2</sup>, and Meibao Yao<sup>1,∗</sup>

Abstract— Generalized morphology control requires a single policy to transform information across limbs with different physical roles, coordinate whole-body motion, and remain efficient as body size grows. Existing communication mechanisms address these requirements only partially. We introduce RecMorph, a topology-guided spatial recurrent architecture that uses recurrent sequence computation to jointly perform cross-limb communication and representation transformation. A depth-first traversal converts the kinematic tree into a morphology-derived sequence, along which shared bidirectional transitions progressively transform limb information before action decoding. Residual preservation, RMS normalization, and input-dependent channel modulation stabilize this repeated spatial transformation, yielding linear token complexity at fixed model width and depth. Across five UNIMAL tasks, RecMorph achieves the strongest mean final training performance among the evaluated generalized morphology controllers and the highest measured inference throughput on FT, while generalizing to unseen variations and bodies with up to 30 limbs. We further migrate representative generalized controllers from UNIMAL benchmarks to a four-platform quadruped setting. RecMorph achieves the best macro-averaged performance under nominal and high friction, reduces nominal velocity RMSE by 43.5% relative to specialist MLPs, and one shared policy completes 40 physical Go1/Go2 trials without falls. These results show that topology-guided recurrent transformation provides an effective and efficient communication mechanism for Generalized Morphology Control and remains effective when transferred from procedural bodies to physical robot platforms. Code and experimental resources are publicly available at https: //github.com/quanruirao/RecMorph.

## I. INTRODUCTION

A robot’s morphology determines how local motion contributes to whole-body behavior. The same joint velocity can propel one body, stabilize another, or destabilize a third. Learning one controller across these bodies therefore requires more than accommodating a variable number of actuators: information expressed in one limb’s physical context must be converted into representations that are useful for controlling other parts of the body. This creates three coupled requirements for Generalized Morphology Control: the policy must transform information across heterogeneous limb contexts, coordinate the complete body, and keep this communication efficient as the number of limbs grows. Meeting these requirements enables control knowledge to be reused across robot structures without designing and training an independent controller for every body.

![](images/2eb8799e0a141b41ae829d132102438f52a73b49ee64664d979e069a89ef0094.jpg)  
Fig. 1. Motivation and overview of RecMorph. RecMorph uses topologyordered recurrent transformation to achieve cross-limb contextualization, whole-body communication, and linear token complexity.

Existing controllers address different parts of this prob lem [1]–[4]. Graph message passing follows the robot kinematic structure and provides efficient local communication. Whole-body interaction is obtained by increasing messagepassing depth, so already aggregated neighborhood representations are repeatedly mixed as the receptive field expands, which can weaken limb-specific information. Dense selfattention removes this local receptive-field restriction and gives every limb direct access to every other limb, but requires quadratic pairwise token interactions. Moreover, a source limb contributes essentially the same projected value representation to different targets, with target specificity entering primarily through attention weights. There is no process of transformation into target-specific limb or movement representations. Morphology-conditioned projec tions improve this local representation mapping, but retain dense attention for body-wide aggregation and its associated pairwise computation. In this paper, cross-limb contextualization denotes constructing a target-conditioned representation from information originating in different limb-local physical contexts. The operation may be realized through direct targetdependent aggregation, morphology-conditioned transformation, progressive propagation, or a combination of these mechanisms. We use cross-limb representation compatibility to describe whether the resulting heterogeneous limb rep resentations can support shared downstream communication and action decoding. These limitations leave a clear need for a communication operator that performs explicit cross-limb contextualization, provides whole-body reach, and scales linearly with morphology size.

These requirements motivate recurrent sequence computation. A recurrent state carries information across an ordered body representation using a number of transitions that grows linearly with sequence length, while each transition simultaneously transforms the propagated feature before it reaches another limb. Communication and representation transformation are therefore performed by the same operator. Our proposed RecMorph exploits this property by serializing the kinematic tree with a depth-first traversal and applying shared bidirectional recurrence along the resulting morphology-derived sequence. Information originating from a source limb is progressively transformed before contributing to downstream limb representations, while the two recurrent directions provide whole-body communication within each block. Repeated spatial transformation also introduces a practical challenge: propagated context must remain stable without overwriting the local state of the current limb. Rec-Morph therefore preserves limb-local information through a residual pathway, controls feature scale with RMSNorm, and uses input-dependent channel modulation to regulate the propagated representation before shared action decoding. At fixed width and block depth, the resulting computation scales linearly with the number of limb tokens.

Most generalized morphology controllers are developed and compared on procedurally generated benchmarks such as UNIMAL, where heterogeneous bodies already share a unified simulation and control interface. It remains unclear whether architectural advantages observed in this setting persist when these controllers are transferred to standard robot platforms with different dynamics and low-level control conventions. We therefore migrate representative generalized morphology controllers to a common quadruped benchmark spanning Go1, Go2, ANYmal-B, and ANYmal-C. Each platform is mapped to the same token-based policy interface, while its native joint ordering, reference configuration, lowlevel gains, and safety limits are preserved in the actuation layer. This setting allows the controller architectures to be compared across standard robot platforms under a common training protocol. RecMorph retains a substantial advantage over the evaluated generalized controllers under nominal and high friction, and the shared-policy design is further deployed on physical Go1 and Go2.

## Our contributions are threefold:

(i) We introduce RecMorph, a generalized controller that converts robot topology into a recurrent communication sequence. Shared bidirectional transitions perform both crosslimb communication and representation transformation, providing whole-body interaction with linear token complexity at fixed network width and depth.

(ii) We develop a stabilized spatial recurrent block that preserves local limb information during repeated transformation. UNIMAL experiments show that RecMorph combines strong control performance with high inference throughput, while controlled studies identify the contributions of recurrent stabilization and morphology-derived ordering and demonstrate generalization to unseen and substantially larger bodies.

(iii) We transfer representative Generalized Morphology Control methods from procedural UNIMAL benchmarks to a common four-quadruped setting and develop the interface required for shared-policy control across heterogeneous robot platforms. RecMorph achieves the strongest aggregate performance under nominal and high friction and is further deployed with one shared policy on physical Go1 and Go2.

## II. RELATED WORK

Generalized morphology control. Graph-based controllers encode robot structure directly in the policy. NerveNet propagates messages between connected body modules [1], while shared modular policies perform bottomup and top-down communication over the kinematic tree [2]. Token-based controllers instead construct global interactions among limb representations. MetaMorph applies self-attention to DFS-ordered limb tokens [3]; SWAT and Body Transformer introduce additional structural constraints into attention [5], [6]; and GCNT combines morphology encoding with graph-based communication [7]. ModuMorph generates morphology-conditioned projections to adapt limb representations before global aggregation [4]. RecMorph introduces a recurrent communication operator in which robot topology determines the processing order and shared bidirectional transitions progressively transform information between limb contexts. An operator-level comparison with attention- and graph-based morphology controllers is provided in Appendix B.

Legged robot control. Learning-based locomotion has enabled robust deployment of legged robots through largescale simulation, system identification, actuator modeling, domain randomization, and policy optimization [8]–[11]. Additional methods improve robustness and versatility through motion imitation, online adaptation, and explicit adaptation to variations in terrain and system dynamics [10], [12]. These advances primarily optimize locomotion for a fixed embod iment, where the observation representation, action parameterization, nominal configuration, and low-level control interface are designed for a specific robot. In contrast, generalized morphology control introduces an additional requirement: a single policy must preserve effective state aggregation and action decoding across multiple bodies with distinct mechanical structures. Our quadruped evaluation studies this cross-platform setting by training shared policies across Go1, Go2, ANYmal-B, and ANYmal-C and comparing them with both generalized morphology baselines and independently trained platform-specific controllers.

## III. TOPOLOGY-GUIDED SPATIAL RECURRENCE

## A. Problem Formulation

We consider a family of robot morphologies indexed by k, where morphology k is represented by a rooted kinematic tree $G _ { k } ~ = ~ ( V _ { k } , E _ { k } )$ with $D _ { k }$ limb tokens. At control step t, limb i is described by local proprioception $o _ { k , t } ^ { i }$ and available morphology attributes $m _ { k } ^ { i }$ . A shared policy π produces actions for the active actuators, while padded tokens are masked from action probabilities, entropy, and value aggregation. The learning objective is

![](images/43c11fc45943a1d78865691211f371373c0c5c85cd8f8c2fb41be97b9a65225b.jpg)  
Fig. 2. RecMorph architecture. A morphology-derived traversal defines the within-step communication order, and stabilized bidirectional recurrence propagates whole-body context before shared action decoding.

$$
\operatorname* { m a x } _ { \theta } \mathbb { E } _ { k \sim p ( k ) , \pi _ { \theta } } \left[ \sum _ { { t \geq 0 } } \gamma ^ { t } r _ { k } ( s _ { t } , a _ { t } ) \right] ,\tag{1}
$$

where robot-specific dynamics and rewards are retained while the policy parameters are shared across morphologies.

To expose the policy to morphology structure, we define a deterministic DFS preorder $p _ { k }$ over $G _ { k }$ and construct

$$
\begin{array} { r } { x _ { j } ^ { 0 } = \sqrt { d } E \left[ o _ { k , t } ^ { p _ { k } ( j ) } , m _ { k } ^ { p _ { k } ( j ) } \right] , } \end{array}\tag{2}
$$

where $E ( \cdot )$ is the shared token encoder. The traversal determines the spatial communication order, while the inverse permutation restores decoded actions to the physical actuator ordering. Terrain-aware UNIMAL tasks additionally encode the local height field and fuse it with contextualized limb features before decoding. Actor and critic use separate parameters, and the critic aggregates values only over active limbs. The primary formulation assumes a rooted kinematic tree; an extension to closed-loop morphologies using spanning-tree serialization is evaluated in Appendix L.

## B. Stabilized Bidirectional Spatial Recurrence

Topology-guided serialization determines where information is propagated; the recurrent block determines how that information is transformed and integrated at each limb. A useful spatial communication operator must therefore propagate body-wide context without overwhelming the limbspecific representation that ultimately drives action decoding. RecMorph realizes this principle through a stabilized bidirectional recurrent block that combines normalized feature transformation, state-dependent modulation, and residual information preservation.

For the primary BiRNN instantiation, each block first applies RMSNorm [13],

$$
N ( v ) = w \odot \frac { v } { \sqrt { \mathrm { m e a n } ( v ^ { 2 } ) + \epsilon } } ,\tag{3}
$$

where each normalization layer maintains an independent learnable scale. Given the representation $x _ { j } ^ { \ell }$ of limb token j at block $\ell ,$ two parallel projections construct the recurrent input and a state-dependent modulation signal:

$$
u _ { j } = \mathrm { S i L U } \big ( W _ { x } N _ { x } ( x _ { j } ^ { \ell } ) \big ) , \qquad z _ { j } = W _ { z } x _ { j } ^ { \ell } .\tag{4}
$$

The recurrent input is propagated in both directions along the morphology-derived sequence,

$$
\overrightarrow { h } _ { j } = \operatorname { t a n h } ( A _ {  } \overrightarrow { h } _ { j - 1 } + B _ {  } u _ { j } + b _ {  } ) ,\tag{5}
$$

$$
\begin{array} { r } { \overleftarrow { h } _ { j } = \operatorname { t a n h } ( A _ {  } \overleftarrow { h } _ { j + 1 } + B _ {  } u _ { j } + b _ {  } ) . } \end{array}\tag{6}
$$

The corresponding boundary conditions are

$$
\stackrel { \right. } { h } _ { 0 } = { \bf 0 } , \qquad \stackrel { \left. } { h } _ { D _ { k } + 1 } = { \bf 0 } .\tag{7}
$$

The boundary states are reinitialized at every policy invocation, so the recurrent state represents spatial communication within the current body configuration rather than temporal memory across control steps. The two directions are concatenated as

$$
y _ { j } = \left[ \overrightarrow { h } _ { j } ; \overleftarrow { h } _ { j } \right] ,\tag{8}
$$

providing each limb with context from both sides of the serialized morphology.

RecMorph then conditions the propagated context on the current limb representation before residual integration. We first define the limb-conditioned modulation vector as

$$
\tilde { g } _ { j } = \mathrm { S i L U } ( z _ { j } ) , \qquad g _ { j } = [ \tilde { g } _ { j } ; \tilde { g } _ { j } ] .\tag{9}
$$

The duplicated modulation vector matches the dimensionality of the concatenated forward and backward recurrent representation. The block output is then

$$
x _ { j } ^ { \ell + 1 } = N _ { o } \big ( x _ { j } ^ { \ell } + W _ { o } N _ { g } ( y _ { j } \odot g _ { j } ) \big ) .\tag{10}
$$

This construction separates contextual transport from limb-conditioned contextual modulation. Bidirectional recurrence aggregates information across the body, whereas channel-wise modulation determines which components of that context are emphasized for the current limb. Applying modulation before normalization preserves its effect on the relative composition of the contextual representation, while the residual pathway retains a direct limb-local signal throughout the recurrent stack. RMS normalization further controls the scale of repeatedly transformed features, yielding a stable interface between successive spatial communication blocks. The resulting block can therefore be viewed as a structured contextualization operator: topology determines the communication order, recurrence transports information along that order, and modulated residual integration determines how the transported context modifies each limblocal representation. This decomposition is important for generalized morphology control, where the same communication mechanism must remain applicable across bodies with different numbers and arrangements of limbs.

![](images/85e2c604c9dbd4e74f95405c9105de58beab188e7cfd9ae945b8eabe715cd28e.jpg)  
Fig. 3. UNIMAL evaluation tasks: flat terrain, incline, exploration, variable terrain, and obstacle traversal.

The primary UNIMAL configuration stacks four such blocks with an embedding width of 128 and a hidden width of 256 in each direction. Variants such as BiLSTM and BiGRU modify only the recurrent transition while preserving the normalization, modulation, residual pathway, and decoder interface. This controlled substitution isolates the choice of recurrent operator from the stabilization architecture.

## C. Topology-Guided Feature Transport

The serialized morphology turns recurrent propagation into a structured within-step transport process. To expose the recurrent transport core, consider a linearized form in which normalization, modulation, residual integration, and bias terms are temporarily omitted. Partition the bidirectional output projection as $W _ { o } = [ W _ { o , \right. } W _ { o , \left. } ]$ . The contextual contribution at token i can then be written as

$$
\Delta x _ { i } = W _ { o , \right. } \sum _ { j = 1 } ^ { i } A _ { \right. } ^ { i - j } B _ { \right. } u _ { j } + W _ { o , \left. } \sum _ { j = i } ^ { D _ { k } } A _ { \left. } ^ { j - i } B _ { \left. } u _ { j } .\tag{11}
$$

Hence, information from token j reaches token i through repeated applications of a shared recurrent transition, with the number of applications determined by their separation in the serialized morphology. In the nonlinear controller, the influence is governed by products of state-dependent transition Jacobians. RecMorph realizes cross-limb contextualization through this progressive source-to-target transformation. A source feature is repeatedly transformed as it traverses the morphology-derived sequence, so its contribution to a target limb depends on the learned transitions and the portion of the body through which it propagates.

DFS keeps limbs within the same subtree contiguous and places many joints along an articulation chain near one another in the serialized order. Bidirectional propagation therefore supports efficient exchange within kinematic branches while still exposing each token to context from both sides of the body-derived sequence.

The traversal is not unique: different sibling orders preserve the same physical morphology while changing sequence adjacency. For a permutation P and token-to-action network $F _ { \theta }$ , the action mean in physical actuator order is

$$
\mu _ { \theta , P } ( X ) = P ^ { \top } F _ { \theta } ( P X ) ,\tag{12}
$$

with the same reindexing applied to masks and morphol ogy attributes. During augmentation, children of each parent are independently permuted once per episode and the order is applied consistently throughout the rollout. This exposes the policy to multiple topology-preserving serializations while maintaining a token-to-actuator correspondence.

We evaluate globally randomized token orders, which disrupt local subtree structure and provide a stronger intervention on the communication path. At fixed hidden width h and block count L, recurrent computation scales as $O ( L D _ { k } h ^ { 2 } )$ , with state storage linear in the number of limb tokens. The experiments in Sec. IV examine how the morphology-derived order affects control performance, ordering robustness, throughput, and action relevance. A linearized derivation of the sequence-distance-dependent recurrent transport operator is provided in Appendix C.

## IV. GENERALIZED MORPHOLOGY CONTROL

This section evaluates RecMorph on UNIMAL from two complementary perspectives: its ability to combine strong task performance, high inference efficiency, zero-shot generalization, and scaling to larger morphologies; and the roles of recurrent stabilization, morphology-derived ordering, and cross-limb transformation in producing these gains.

## A. Benchmark and Training Protocol

We evaluate RecMorph on 100 UNIMAL morphologies with up to 12 limbs in MuJoCo [14], [15]. The five tasks in Fig. 3 span flat-terrain locomotion (FT), incline traversal, exploration, obstacle traversal, and locomotion over curved slopes, steps, and rugged surfaces (varied terrain, VT). A single generalized morphology policy is trained across all bodies within each task. Terrain-aware tasks additionally provide local height-field observations.

Policies are trained with PPO [16] for approximately 100 million environment interactions. We report online training return as mean ± sample standard deviation over four independent policy seeds. We compare against representative generalized morphology controllers spanning global attention with MetaMorph [3] and MetaMorph\* [4], structureaware attention with SWAT [5], contextual modulation with ModuMorph [4], graph communication with NerveNet [1] and GCNT [7], embodiment-aware attention with Body Transformer (BoT) [6], morphology-informed heterogeneous graph communication with MI-HGNN [17], and shared modular recurrence with SMR [18]. We also include two singlerobot baselines. The single-robot (fair) baseline trains an independent MLP policy for each morphology using the same per-robot training budget as the multi-morphology setting. The single-robot (10M) baseline trains an independent MLP policy for each morphology for 10 million steps and serves as an approximate single-morphology upper reference. For BoT and MI-HGNN, we construct policy adaptations compatible with the common generalized morphology control interface.

![](images/cfa0d02019b36ad9495d40272f8ff7d76e7a82e184a33e257cae99cd7c5d0344.jpg)

Fig. 4. Performance of representative generalized morphology controllers across the five UNIMAL tasks.  
![](images/e6493e6461686bf980e89516d6d66dd7a41a41313fa212873101406d50a4f030.jpg)  
Fig. 5. Analysis of the number of parameters (params) and frames per second (FPS) for different methods on FT.

## B. Control Performance and Efficiency

As shown in Fig. 4, the RecMorph variant attains the highest mean on every task. Relative to the strongest non-RecMorph mean in each column, the gains are 4.9% on FT, 48.1% on incline, 9.1% on obstacles, 23.7% on exploration, and 15.1% on VT. BiRNN leads FT and incline, while BiGRU leads obstacles and exploration, and BiLSTM leads VT. The recurrent transition is therefore a selectable component within the same spatial-transport architecture. The comparable performance of BiRNN, BiLSTM, and BiGRU further indicates that the benefit is associated with the shared spatial-transport design rather than a particular recurrent cell.

In the comparative experiments with baselines, RecMorph employed a four-layer bidirectional sequential model. Figure 5 compares model size and measured inference through put on FT. RecMorph(BiRNN) achieves the highest FPS among the evaluated controllers while remaining within a comparable parameter budget, indicating that recurrent spatial transport provides favorable runtime scaling in addition to its control-performance gains. Taken together, these results place RecMorph at the intended performance–efficiency operating point: recurrent whole-body communication achieves the strongest evaluated control performance together with the highest measured inference throughput.

## C. Zero-Shot Generalization

We evaluate whether the learned policy transfers to unseen morphology variations without additional training. As shown in Fig. 6, across these zero-shot perturbations, RecMorph consistently retains strong transfer performance relative to the evaluated generalized morphology baselines. The advantage extends from dynamics and kinematics changes to previously unseen topology graphs, showing that the learned spatial communication rule generalizes beyond the exact training embodiments. These results support topology-guided recurrence as a transferable structural prior rather than a mechanism specialized to a fixed morphology collection. A complementary stress test under single-limb observation dropout is reported in Appendix K.

![](images/571ea0fc7912aefb9c1831ced8d638078d915e7c1b95e696e57afbd65569b84c.jpg)  
Fig. 6. Zero-shot transfer under dynamics, kinematic, and topology changes (four policy seeds; bars show mean and error bars show seed SD).

## D. Scaling to Larger Bodies

Larger bodies test whether a shared transition remains useful when both the communication sequence and the action space grow. The larger-body setting increases the maximum limb count from 12 to 30 and the average from approximately 10 to 25. Figure 7 shows RecMorph learning effective locomotion in this setting and achieving a higher training-return curve than the attention-based controllers. The single-robot reference is an independently trained MLP with a 10-millionstep budget per body; its horizontal line is a final-return reference, not a learning curve. Returns also decrease for the independent MLP reference, showing that larger bodies increase control difficulty beyond the communication operator. Together with zero-shot transfer, this study tests both adaptation-free reuse and shared-policy learning over longer body descriptions. Additional throughput measurements and reference-baseline statistics for the larger-body setting are reported in Appendix J.

![](images/e31eec3b330619e5e16cbb745ca493efcb485ad0742787c0f8114cf009d06ec0.jpg)  
Fig. 7. FT learning on larger morphologies with up to 30 limbs. The single-robot reference is independently trained for 10M steps per body.

## E. Stabilizing Repeated Spatial Transformation

The recurrent operator repeatedly transforms information before it reaches later limbs, making preservation of local state and feature scale central to the architecture. We therefore isolate the components introduced to stabilize this repeated spatial transformation.

TABLE I  
COMPONENT ABLATION. “PARAM.-MATCHED” DENOTES A PLAIN BIRNN WITH A PARAMETER COUNT MATCHED TO RECMORPH.
<table><tr><td>Variant</td><td>Residual</td><td>RMSNorm</td><td>Channel Modulation</td><td>Return</td></tr><tr><td>Plain BiRNN</td><td></td><td></td><td></td><td> $2 1 8 8 . 6 \pm 4 1 8 . 7$ </td></tr><tr><td>Param.-matched</td><td></td><td></td><td></td><td> $2 7 8 5 . 2 \pm 4 8 8 . 5$ </td></tr><tr><td>+ Residual</td><td>√</td><td></td><td></td><td> $3 5 7 7 . 0 \pm 2 6 0 . 7$ </td></tr><tr><td>+ RMSNorm</td><td>√</td><td>√</td><td></td><td> $4 0 2 7 . 7 \pm 1 1 0 . 6$ </td></tr><tr><td>Full RecMorph</td><td>√</td><td>√</td><td>√</td><td> ${ \bf 4 3 6 7 . 4 \pm 7 1 . 6 }$ </td></tr></table>

Table I shows a progressive improvement from ordinary recurrence to the full controller. The residual path increases return from 2188.6 to 3577.0, and adding RMSNorm raises it to 4027.7. Channel Modulation provides a further 8.4% return improvement and 17.0% learning-curve-area improvement over residual+RMSNorm. This supports a complementary role for information preservation, normalization, and input-dependent channel modulation. The nested interventions quantify each component’s effect conditional on the preceding design. The parameter-matched plain model reaches 2785.2 with 3.164 million actor–critic parameters, compared with RecMorph’s 3.174 million. A same-V100 diagnostic gives comparable training throughput, 700 versus 698 FPS. The stabilized controller therefore achieves higher return at comparable capacity and measured throughput.

## F. Role of Morphology-Derived Order

Topology provides RecMorph with more than token identity: it determines the order in which source information is transformed before reaching a target. We therefore vary the traversal while holding the recurrent operator fixed.

![](images/aea2f0085490e708b503f3a74e52b6d4dc2045331b47ce16e8ba7a6a2e2587c7.jpg)  
Fig. 8. Effect of morphology-derived traversal order. DFS and BFS preserve ordering, whereas global randomization degrades RecMorph learning.

Figure 8 compares the effects of depth-first search (DFS), breadth-first search (BFS), and global random ordering on RecMorph. Both tree-derived traversals give stronger Rec-Morph learning curves than global random order, with DFS leading BFS. The relevant prior is therefore the organization of communication by body structure. Breadth-first order groups limbs by depth, while DFS keeps subtrees contiguous; either retains regularities that an arbitrary token sequence disrupts. DFS supplies a communication order, but a tree admits multiple equivalent sibling orders. We evaluate this choice while keeping the physical robot and actuator mapping fixed. Each of 100 robots is tested under 50 unseen sibling permutations; augmentation samples an equivalent order once per training episode.

TABLE II  
SIBLING-ORDER ROBUSTNESS. C: CANONICAL; P: PERMUTED.
<table><tr><td>Train / seed</td><td>C test</td><td>P test</td><td>Drop (%)</td><td>Tail</td></tr><tr><td>C / 1</td><td>1095.0</td><td>399.3</td><td>63.5</td><td>319.1</td></tr><tr><td>C /  1409</td><td>1064.8</td><td>413.8</td><td>61.1</td><td>319.3</td></tr><tr><td>P / 1</td><td>1369.3</td><td>1268.1</td><td>7.4</td><td>1164.4</td></tr><tr><td>P /  1409</td><td>1138.3</td><td>1094.0</td><td>3.9</td><td>993.7</td></tr></table>

Table II reveals both the strength and flexibility of the sequence prior. Canonical policies depend strongly on their training order, losing 61.1–63.5% under sibling permutations. Episode-level augmentation reduces the loss to 3.9–7.4% and improves the worst-decile return. Canonical evaluation also improves in these two runs. The intervention therefore offers a practical way to make the learned controller less dependent on an arbitrary serialization choice. These results measure learned ordering robustness using episode evaluation, separately from the online training benchmark.

## G. Cross-Limb Transformation and Action Relevance

We examine how recurrent spatial transport changes the compatibility of limb representations with a shared action decoder. Matched linear probes are fitted before and after communication on four independently trained policies.

TABLE III  
LINEAR PROBES BEFORE AND AFTER RECURRENT COMMUNICATION.
<table><tr><td>Probe</td><td>Pre</td><td>Post</td><td>∆</td></tr><tr><td>Robot-ID acc.</td><td> $8 4 . 2 8 \pm 1 . 2 6$ </td><td> $6 1 . 4 6 \pm 3 . 1 9$ </td><td> ${ \bf - 2 2 . 8 2 \pm 3 . 6 3 }$ </td></tr><tr><td>Limb-ID acc.</td><td> $9 5 . 4 9 \pm 0 . 3 9$ </td><td> $8 6 . 4 5 \pm 2 . 6 2$ </td><td> $\mathbf { - 9 . 0 5 \pm 2 . 4 4 }$ </td></tr><tr><td>Action  $R ^ { 2 }$ </td><td> $- 0 . 0 7 1 \pm 0 . 0 2 6$ </td><td> $0 . 3 5 4 \pm 0 . 0 2 1$ </td><td> $\mathbf { + 0 . 4 2 5 \pm 0 . 0 2 8 }$ </td></tr><tr><td>Action NRMSE</td><td> $1 . 0 3 3 \pm 0 . 0 1 3$ </td><td></td><td> $0 . 8 0 2 \pm 0 . 0 1 4 - 0 . 2 3 1 \pm 0 . 0 1 5$ </td></tr></table>

As shown in Table III, communication substantially reduces explicit morphology identity while improving action predictability. Robot-ID and limb-ID balanced accuracy decrease from $8 4 . 2 8 \pm 1 . 2 6$ to $6 1 . 4 6 \pm 3 . 1 9$ and from $9 5 . 4 9 \pm$ 0.39 to $8 6 . 4 5 \pm 2 . 6 2$ , respectively. Meanwhile, action $R ^ { 2 }$ increases from −0.071±0.026 to $0 . 3 5 4 { \pm } 0 . 0 2 1$ , and NRMSE decreases from $1 . 0 3 3 \pm 0 . 0 1 3$ to $0 . 8 0 2 \pm 0 . 0 1 4$ . Together, the probes show that recurrent transport reduces explicit source-identity information while producing representations that are more predictive of the policy action. This behavior is consistent with the intended cross-limb transformation. Complementary feature-space analysis is provided in Appendix F.

## V. CROSS-PLATFORM GENERALIZED CONTROL

The UNIMAL experiments establish the performance, efficiency, and generalization of RecMorph under a unified procedural morphology and control interface. We extend this evaluation to standard quadruped platforms, where controller architectures must operate across different body dynamics and low-level control conventions. A matched benchmark spanning Go1, Go2, ANYmal-B, and ANYmal-C evaluates whether the advantage of RecMorph persists beyond the UNIMAL setting, followed by physical deployment of one shared policy on Go1 and Go2.

## A. Transfer to Four Quadruped Platforms

We transfer RecMorph, MetaMorph, ModuMorph, Body-Transformer, and GCNT to the same four-platform benchmark, with each method sharing one policy across Go1, Go2, ANYmal-B, and ANYmal-C. Each robot is represented through a common limb-token observation and action interface. A platform adapter maps native joint states into the shared token order and maps the active policy outputs back to the corresponding native joints. Robot-specific reference poses, low-level gains, joint limits, and safety constraints remain in the actuation layer. The resulting interface allows the same generalized controller architecture to operate across all four platforms while preserving the low-level control settings required by each robot. The specialist MLP baseline is trained independently for each robot, whereas all generalized-controller baselines share one policy across the four platforms. All methods use 128 environments per robot, 32 rollout steps, and 10,000 PPO iterations. Four independent random seeds are used for every method.

Table IV reports four-platform macro results under nominal static/dynamic friction (0.8, 0.6). RecMorph provides the best tracking, progress, stability, and fall-rate trade-off among shared and specialist controllers. Relative to independently trained per-robot MLPs, RecMorph reduces velocity RMSE from $0 . 2 7 6 \pm 0 . 1 1 6$ to $0 . 1 5 6 \pm 0 . 0 3 8$ m/s (43.5%), reduces tilt RMS from $( 5 . 5 2 \pm 0 . 5 5 ) ^ { \circ }$ to $( 3 . 0 0 \pm 0 . 6 2 ) ^ { \circ }$ (45.6%), and lowers the fall rate from $1 7 . 0 4 \pm 1 1 . 4 7 \%$ to $6 . 9 7 \pm 1 1 . 1 1 \% \ ( 5 9 . 1 \% )$ , while increasing progress from $9 2 . 1 5 { \pm } 7 . 0 6 $ to 97.11±0.82 m (5.4%). RecMorph has the best mean on all four metrics among the shared-policy baselines. Relative to ModuMorph, the strongest shared controller in nominal velocity tracking, RecMorph reduces mean RMSE by 75.9% and approximately doubles mean progress.

Table V compares all six controllers under low, nominal, and high friction. Every entry reports the mean and sample SD across independent policy seeds. Static/dynamic coefficients are (0.3, 0.2), (0.8, 0.6), and (1.2, 1.0), respectively.

Across the four evaluated platforms, RecMorph achieves the best macro-averaged performance on all four metrics under both nominal and high friction, outperforming the shared-controller baselines and independently trained specialist MLPs. Under high friction, it achieves 0.133 ± 0.023 m/s velocity RMSE, $9 8 . 4 0 \pm 1 . 9 7$ m progress, (2.69 ± $0 . 8 5 ) ^ { \circ }$ tilt RMS, and a $( 6 . 7 4 \pm 1 2 . 2 6 ) \%$ fall rate using a single shared policy. These results demonstrate that the aggregate advantage of RecMorph extends across two contact settings, supporting effective policy sharing across the evaluated embodiments without requiring separate platformspecific policies. Under low friction, RecMorph retains advantages over several shared-controller baselines, achieving lower mean velocity RMSE and greater mean progress than MetaMorph, BodyTransformer, and GCNT. Compared with specialist MLPs, it achieves slightly lower mean velocity RMSE (0.907 ± 0.042 versus $0 . 9 1 9 \pm 0 . 1 9 8 \mathrm { m } / \mathrm { s } )$ and lower mean tilt RMS $( ( 2 0 . 9 5 \pm 5 . 4 5 ) ^ { \circ }$ versus $( 2 9 . 2 1 \pm 9 . 7 5 ) ^ { \circ } )$ , with only a modest reduction in mean progress $( 1 2 . 9 9 \pm 3 . 2 7 $ versus $1 5 . 2 2 \pm 1 8 . 0 2 \mathrm { m } )$ . This competitiveness does not extend to fall resistance: RecMorph has a higher fall rate, (77.14 ± 14.24)% versus $( 2 8 . 2 5 \pm 2 8 . 1 7 ) \%$ . ModuMorph achieves better low-friction means on all four metrics, although its observed seed-to-seed SD is substantially larger for velocity RMSE and progress. These results show that the performance advantage of RecMorph persists when generalized controllers are transferred from procedural UNIMAL morphologies to standard quadruped platforms, supporting the transferability of its recurrent cross-limb transformation beyond the original benchmark setting.

## B. Physical Deployment on Go1 and Go2

We deploy one separately trained shared RecMorph policy on physical Go1 and Go2 to assess whether the highlevel mapping can be reused without platform-specific policy retraining. Each condition contains ten 15-s trials, all completed without a reported fall. Detailed deployment training settings and trial-level success statistics are provided in Appendices N and O.

The deployed checkpoint uses policy seed 1409 and a 64- token, 32-feature interface with 12 active actuator tokens per quadruped. The hardware experiment demonstrates reuse of a shared policy on two physical platforms. Active tokens contain base velocity, projected gravity, velocity commands, joint states, previous actions, and joint-limit features. The shared decoder produces joint-position offsets according to

![](images/0b5b9b290ff5d5bc630a333004ea160f04a2fb0c551de73ee9a20ff7be62be5f.jpg)  
Fig. 9. Shared-policy deployment on physical Go1 and Go2 robots. All four conditions use the same RecMorph policy

TABLE IV  
NOMINAL-FRICTION QUADRUPED BENCHMARK.
<table><tr><td>Method</td><td>Controller scope</td><td>Velocity RMSE (m/s) ↓</td><td>Progress (m) ↑</td><td>Tilt RMS (°) ↓</td><td>Fall rate (%) ↓</td></tr><tr><td>RecMorph(BiRNN)</td><td>Shared</td><td> $\mathbf { 0 . 1 5 6 \pm 0 . 0 3 8 }$ </td><td> $\mathbf { 9 7 . 1 1 \pm 0 . 8 2 }$ </td><td> $\mathbf { 3 . 0 0 \pm 0 . 6 2 }$ </td><td> ${ \bf 6 . 9 7 \pm 1 1 . 1 1 }$ </td></tr><tr><td>Per-robot MLP</td><td>Specialist</td><td> $0 . 2 7 6 \pm 0 . 1 1 6$ </td><td> $9 2 . 1 5 \pm 7 . 0 6$ </td><td> $5 . 5 2 \pm 0 . 5 5$ </td><td> $1 7 . 0 4 \pm 1 1 . 4 7$ </td></tr><tr><td>ModuMorph</td><td>Shared</td><td> $0 . 6 4 8 \pm 0 . 5 8 5$ </td><td> $4 8 . 4 9 \pm 5 8 . 7 0$ </td><td> $3 . 6 1 \pm 2 . 1 7$ </td><td> $4 3 . 3 5 \pm 9 . 4 1$ </td></tr><tr><td>MetaMorph</td><td>Shared</td><td> $0 . 7 7 9 \pm 0 . 0 1 6$ </td><td> $2 9 . 2 7 \pm 0 . 6 5$ </td><td> $5 . 1 4 \pm 0 . 3 2$ </td><td> $3 4 . 2 6 \pm 0 . 7 8$ </td></tr><tr><td>BodyTransformer</td><td>Shared</td><td> $0 . 9 6 5 \pm 0 . 1 4 3$ </td><td> $1 5 . 8 1 \pm 8 . 8 5$ </td><td> $1 6 . 5 4 \pm 0 . 5 6$ </td><td> $9 6 . 6 9 \pm 3 . 2 5$ </td></tr><tr><td>GCNT</td><td>Shared</td><td> $1 . 1 7 4 \pm 0 . 1 6 0$ </td><td> $- 6 . 0 0 \pm 1 0 . 7 2$ </td><td> $1 7 . 1 1 \pm 4 . 0 4$ </td><td> $9 9 . 9 4 \pm 0 . 0 8$ </td></tr></table>

TABLE V

CROSS-PLATFORM RESULTS UNDER FRICTION VARIATION.
<table><tr><td>Method</td><td>Friction</td><td>Velocity RMSE (m/s) ↓</td><td>Progress (m) ↑</td><td>Tilt RMS (°) ↓</td><td>Fall rate (%) ↓</td></tr><tr><td>RecMorph(BiRNN)</td><td>Low</td><td> $0 . 9 0 7 \pm 0 . 0 4 2$ </td><td> $1 2 . 9 9 \pm 3 . 2 7$ </td><td> $2 0 . 9 5 \pm 5 . 4 5$ </td><td> $7 7 . 1 4 \pm 1 4 . 2 4$ </td></tr><tr><td></td><td>Nominal</td><td> $\mathbf { 0 . 1 5 6 \pm 0 . 0 3 8 }$ </td><td> ${ \bf 9 7 . 1 1 \pm 0 . 8 2 }$ </td><td> ${ \bf 3 . 0 0 \pm 0 . 6 2 }$ </td><td> ${ \bf 6 . 9 7 \pm 1 1 . 1 1 }$ </td></tr><tr><td></td><td>High</td><td> $\mathbf { 0 . 1 3 3 \pm 0 . 0 2 3 }$ </td><td> ${ \bf 9 8 . 4 0 \pm 1 . 9 7 }$ </td><td> ${ \bf 2 . 6 9 \pm 0 . 8 5 }$ </td><td> ${ \bf 6 . 7 4 \pm 1 2 . 2 6 }$ </td></tr><tr><td>Per-robot MLP</td><td>Low</td><td> $0 . 9 1 9 \pm 0 . 1 9 8$ </td><td> $1 5 . 2 2 \pm 1 8 . 0 2$ </td><td> $2 9 . 2 1 \pm 9 . 7 5$ </td><td> ${ \bf 2 8 . 2 5 \pm 2 8 . 1 7 }$ </td></tr><tr><td></td><td>Nominal</td><td> $0 . 2 7 6 \pm 0 . 1 1 6$ </td><td> $9 2 . 1 5 \pm 7 . 0 6$ </td><td> $5 . 5 2 \pm 0 . 5 5$ </td><td> $1 7 . 0 4 \pm 1 1 . 4 7$ </td></tr><tr><td></td><td>High</td><td> $0 . 3 3 2 \pm 0 . 0 9 3$ </td><td> $8 5 . 3 2 \pm 3 . 3 7$ </td><td> $8 . 5 4 \pm 1 . 2 4$ </td><td> $2 5 . 7 4 \pm 1 2 . 2 3$ </td></tr><tr><td>ModuMorph</td><td>Low</td><td> $\mathbf { 0 . 7 9 3 \pm 0 . 3 6 4 }$ </td><td> $\mathbf { 2 9 . 6 3 \pm 3 2 . 8 9 }$ </td><td> ${ \bf 4 . 5 2 \pm 2 . 5 8 }$ </td><td> $4 9 . 8 2 \pm 0 . 2 6$ </td></tr><tr><td></td><td>Nominal</td><td> $0 . 6 4 8 \pm 0 . 5 8 5$ </td><td> $4 8 . 4 9 \pm 5 8 . 7 0$ </td><td> $3 . 6 1 \pm 2 . 1 7$ </td><td> $4 3 . 3 5 \pm 9 . 4 1$ </td></tr><tr><td></td><td>High</td><td> $0 . 6 5 4 \pm 0 . 5 7 6$ </td><td> $4 7 . 8 3 \pm 5 7 . 7 5$ </td><td> $3 . 9 1 \pm 2 . 5 6$ </td><td> $4 6 . 4 9 \pm 4 . 9 6$ </td></tr><tr><td>MetaMorph</td><td>Low</td><td> $0 . 9 9 5 \pm 0 . 0 0 3$ </td><td> $1 . 4 3 \pm 0 . 2 6$ </td><td> $5 . 7 3 \pm 0 . 3 3$ </td><td> $3 6 . 0 9 \pm 1 7 . 6 4$ </td></tr><tr><td></td><td>Nominal</td><td> $0 . 7 7 9 \pm 0 . 0 1 6$ </td><td> $2 9 . 2 7 \pm 0 . 6 5$ </td><td> $5 . 1 4 \pm 0 . 3 2$ </td><td> $3 4 . 2 6 \pm 0 . 7 8$ </td></tr><tr><td></td><td>High</td><td> $0 . 7 4 8 \pm 0 . 0 0 8$ </td><td> $3 1 . 8 4 \pm 0 . 9 2$ </td><td> $6 . 6 2 \pm 0 . 9 1$ </td><td> $4 4 . 9 5 \pm 1 . 1 9$ </td></tr><tr><td>BodyTransformer</td><td>Low</td><td> $1 . 0 0 0 \pm 0 . 0 1 3$ </td><td> $3 . 8 3 \pm 1 . 4 7$ </td><td> $1 3 . 2 4 \pm 2 . 0 3$ </td><td> $9 4 . 0 2 \pm 4 . 8 8$ </td></tr><tr><td></td><td>Nominal</td><td> $0 . 9 6 5 \pm 0 . 1 4 3$ </td><td> $1 5 . 8 1 \pm 8 . 8 5$ </td><td> $1 6 . 5 4 \pm 0 . 5 6$ </td><td> $9 6 . 6 9 \pm 3 . 2 5$ </td></tr><tr><td></td><td>High</td><td> $0 . 9 7 2 \pm 0 . 1 5 8$ </td><td> $1 5 . 2 8 \pm 1 0 . 0 4$ </td><td> $1 6 . 8 5 \pm 0 . 9 5$ </td><td> $9 6 . 8 9 \pm 2 . 9 5$ </td></tr><tr><td>GCNT</td><td>Low</td><td> $1 . 1 8 9 \pm 0 . 2 1 1$ </td><td> $- 1 1 . 5 8 \pm 1 6 . 2 4$ </td><td> $1 3 . 1 0 \pm 7 . 4 7$ </td><td> $9 9 . 0 6 \pm 1 . 3 3 $ </td></tr><tr><td></td><td>Nominal</td><td> $1 . 1 7 4 \pm 0 . 1 6 0$ </td><td> $- 6 . 0 0 \pm 1 0 . 7 2$ </td><td> $1 7 . 1 1 \pm 4 . 0 4$ </td><td> $9 9 . 9 4 \pm 0 . 0 8$ </td></tr><tr><td></td><td>High</td><td> $1 . 2 1 1 \pm 0 . 0 9 3$ </td><td> ${ \cdot } 7 . 0 8 \pm 4 . 5 6 $ </td><td> $1 6 . 5 8 \pm 4 . 7 3$ </td><td> $9 9 . 4 2 \pm 0 . 8 2 $ </td></tr></table>

$$
q _ { \mathrm { t a r g e t } , k } = q _ { \mathrm { d e f a u l t } , k } + 0 . 2 5 a _ { \theta , k } .\tag{13}
$$

The deployment policy runs at 50 Hz and is trained on a mixture of flat and micro-rough terrain with corrupted proprioceptive observations. Thus, the same learned high-level controller is reused across Go1 and Go2 while the low-level interface respects their physical differences. The deployment therefore evaluates the same shared-policy formulation on physical hardware without changing the learned policy across platforms. Trial-level deployment statistics and the success criterion are reported in Appendix O.

## VI. DISCUSSION

RecMorph shows that recurrent sequence computation provides a useful mechanism for Generalized Morphology Control because communication and cross-limb representation transformation are performed by the same operator. Robot topology determines the order of transformation, while a shared recurrent transition progressively converts sourcelimb information into representations used by other limb controllers. The comparable behavior of BiRNN, BiLSTM, and BiGRU indicates that the central mechanism extends beyond a particular recurrent cell. The quadruped experiments add a second result: generalized controller architectures developed on procedural morphology benchmarks can be transferred to standard robot platforms through a shared token/action interface while retaining platform-specific low-level actuation.

The experiments also expose concrete limits of the current formulation. First, RecMorph commits to a fixed topologyconsistent traversal, while the sibling-order study shows that different valid serializations can lead to different learned transformations. Learning task-dependent traversal orders or combining several topology-consistent routes is therefore a natural extension. Second, linear computational complexity does not prevent the source-to-target transformation path from growing with body size; hierarchical recurrence over limbs, subtrees, and branches could shorten this path for substantially larger robots. Third, the low-friction results separate morphology generalization from dynamics adaptation, motivating recurrent transformations conditioned on online estimates of contact, friction, and actuator response.

## VII. CONCLUSION

We introduced RecMorph for Generalized Morphology Control, using robot topology to organize recurrent crosslimb transformation. Shared bidirectional transitions progressively convert limb-local information into target-relevant representations, providing whole-body communication with linear token complexity at fixed network width and depth. UNIMAL experiments establish the performance, efficiency, and generalization of this mechanism, while the crossplatform study shows that its advantage survives the transfer of generalized controllers to standard quadruped platforms and physical Go1/Go2.

## ACKNOWLEDGMENT

This work was supported by the National Natural Science Foundation of China (Grant No. 52472448); the Deep Earth Probe and Mineral Resources Exploration National Science and Technology Major Project (Grant Nos. 2024ZD1000804 and 2024ZD1000802); the Scientific and Technological Research Project of the Education Department of Jilin Province (Grant No. JJKH20261598KJ); and the Natural Science Foundation of Jilin Province (Grant No. 20260205078GH).

## REFERENCES

[1] T. Wang, R. Liao, J. Ba, and S. Fidler, “Nervenet: Learning structured policy with graph neural networks,” in International conference on learning representations, 2018.

[2] W. Huang, I. Mordatch, and D. Pathak, “One policy to control them all: Shared modular policies for agent-agnostic control,” in International Conference on Machine Learning. PMLR, 2020, pp. 4455–4464.

[3] A. Gupta, L. Fan, S. Ganguli, and L. Fei-Fei, “Metamorph: Learning universal controllers with transformers,” arXiv preprint arXiv:2203.11931, 2022.

[4] Z. Xiong, J. Beck, and S. Whiteson, “Universal morphology control via contextual modulation,” in International Conference on Machine Learning. PMLR, 2023, pp. 38 286–38 300.

[5] S. Hong, D. Yoon, and K.-E. Kim, “Structure-aware transformer policy for inhomogeneous multi-task reinforcement learning,” in Interna tional Conference on Learning Representations, 2022.

[6] C. Sferrazza, D.-M. Huang, F. Liu, J. Lee, and P. Abbeel, “Body transformer: Leveraging robot embodiment for policy learning,” arXiv preprint arXiv:2408.06316, 2024.

[7] Y. Luo, M. Yao, and X. Xiao, “Gcnt: Graph-based transformer policies for morphology-agnostic reinforcement learning,” arXiv preprint arXiv:2505.15211, 2025.

[8] J. Tan, T. Zhang, E. Coumans, A. Iscen, Y. Bai, D. Hafner, S. Bohez, and V. Vanhoucke, “Sim-to-real: Learning agile locomotion for quadruped robots,” arXiv preprint arXiv:1804.10332, 2018.

[9] J. Hwangbo, J. Lee, A. Dosovitskiy, D. Bellicoso, V. Tsounis, V. Koltun, and M. Hutter, “Learning agile and dynamic motor skills for legged robots,” Science robotics, vol. 4, no. 26, p. eaau5872, 2019.

[10] A. Kumar, Z. Fu, D. Pathak, and J. Malik, “Rma: Rapid motor adaptation for legged robots,” arXiv preprint arXiv:2107.04034, 2021.

[11] N. Rudin, D. Hoeller, P. Reist, and M. Hutter, “Learning to walk in minutes using massively parallel deep reinforcement learning,” in Conference on robot learning. PMLR, 2022, pp. 91–100.

[12] X. B. Peng, E. Coumans, T. Zhang, T.-W. Lee, J. Tan, and S. Levine, “Learning agile robotic locomotion skills by imitating animals,” arXiv preprint arXiv:2004.00784, 2020.

[13] B. Zhang and R. Sennrich, “Root mean square layer normalization,” Advances in neural information processing systems, vol. 32, 2019.

[14] A. Gupta, S. Savarese, S. Ganguli, and L. Fei-Fei, “Embodied intelligence via learning and evolution,” Nature communications, vol. 12, no. 1, p. 5721, 2021.

[15] E. Todorov, T. Erez, and Y. Tassa, “Mujoco: A physics engine for model-based control,” in 2012 IEEE/RSJ international conference on intelligent robots and systems. IEEE, 2012, pp. 5026–5033.

[16] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.

[17] D. Butterfield, S. S. Garimella, N.-J. Cheng, and L. Gan, “Mi-hgnn: Morphology-informed heterogeneous graph neural network for legged robot contact perception,” in 2025 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2025, pp. 10 110–10 116.

[18] L. Engwegen, D. Brinks, and W. Bohmer, “Modular recurrence in¨ contextual mdps for universal morphology control,” arXiv preprint arXiv:2506.08630, 2025.

[19] R. C. de Amorim and V. Makarenkov, “Improving clustering quality evaluation in noisy gaussian mixtures,” Neurocomputing, p. 133330, 2026.

A Morphology as a Structural Prior for Cross-Limb Representation Compatibility

![](images/0278fac610916594fe50fd275550b829a9940653930c24e9923c2f467b095396.jpg)  
Fig. 10. Performance after removing selected topology-dependent morphology attributes from the UNIMAL observation space. Topology-derived token ordering is retained where applicable.

Generalized morphology control requires information expressed in different limb-local physical contexts to become useful for target-specific action prediction. We refer to this general computation as cross-limb contextualization: a target representation is constructed from source-limb information through target-dependent weighting, morphology-conditioned transformation, progressive propagation, or a combination of these operations. The resulting ability of heterogeneous limb representations to support common downstream computation is referred to as cross-limb representation compatibility. MetaMorph, ModuMorph, NerveNet, and RecMorph realize this computation in different ways. MetaMorph relies primarily on state-dependent global aggregation, ModuMorph additionally adapts limb projections according to morphology, NerveNet performs graph-local message transformation, and RecMorph progressively transforms information along a morphology derived sequence.

## A.1 Sensitivity to Topology-Dependent Morphology Attributes

UNIMAL morphology observations combine hardware descriptors with fields encoding relative position and orientation. We isolate the contribution of these explicit topology-dependent attributes by removing the fields listed in Table IX while preserving the remaining observation interface. Architecture-level structural cues, including morphology-derived token ordering, are retained. This intervention therefore measures controller sensitivity to explicit topology-dependent morphology attributes.

MetaMorph combines DFS-ordered limb tokens with Transformer-based global communication and a learned positional embedding. We additionally evaluate MetaMorph\*, which removes the positional encoding while retaining the same token ordering and attention architecture. For both variants, we then remove the selected topology-dependent morphology attributes defined in Table IX. Figure 10 shows that this attribute intervention does not produce a systematic decrease in return across the five tasks, indicating limited sensitivity to these explicit morphology fields under the evaluated protocol.

$$
\hat { \mathbf { E } } _ { k , t } = \mathbf { E } _ { k , t } + \mathbf { W _ { p o s } }\tag{14}
$$

ModuMorph introduces morphology-conditioned limb projections and morphology-conditioned attention through a hypernetwork. Under the same attribute-removal intervention, its return also remains broadly stable across the evaluated tasks. Fig. 10 indicates that the selected explicit topology-dependent attributes are not the sole source of useful morphology conditioning in this setting; the remaining morphology descriptors continue to support body-dependent feature transformation.

## A.2 Cross-Limb Representation Compatibility

Limb observations correspond to different mechanical roles across morphologies, so a shared controller must transform them into representations that can be jointly compared, propagated, and decoded by common downstream computation. We refer to this property as cross-limb representation compatibility. Different controller families realize this compatibility through different computational mechanisms: shared projection and global attention in MetaMorph, morphology-conditioned mappings in ModuMorph, graph-local communication in NerveNet, and morphology-guided recurrent transport in RecMorph.

We first analyze MetaMorph from this perspective. MetaMorph projects each raw limb observation $\mathbf { S _ { M P } } _ { k , t } ^ { i }$ into a latent embedding using a shared linear layer. Although this projection maps all limb observations into the same feature dimension, it does not explicitly account for limb-specific coordinate frames or physical roles before aggregation. The subsequent global aggregation is performed by Multi-Head Self-Attention. The output $\bar { \mathbf { Z } _ { k , t } ^ { i , ( l ) } }$ for limb i at the ${ l ^ { t h } }$ layer at time t is as seen in Equation 16. Detailed analysis of MetaMorph is provided in Appendix B.1.

$$
\hat { \mathbf { r } } _ { k , t } ^ { i } = \sum _ { j \neq i } \mathbf { R } _ { j , i } ^ { ( l ) } ^ { \top } \mathbf { Z } _ { k , t } ^ { j , ( l - 1 ) } + \left( \mathbf { R } _ { i , i } ^ { ( l ) } + \mathbf { I } \right) ^ { \top } \mathbf { Z } _ { k , t } ^ { i , ( l - 1 ) }\tag{15}
$$

$$
\mathbf { Z } _ { k , t } ^ { i , ( l ) } = \mathbf { W } _ { \mathbf { M X } } ^ { ( l ) } ^ { \top } \hat { \Gamma } _ { k , t } ^ { i } + \mathbf { b } _ { \mathbf { O } } ^ { ( l ) }\tag{16}
$$

Here, $\mathbf { R } _ { j , i } ^ { ( l ) }$ denotes the attention-based aggregation weight from limb $j$ to limb i. For a given attention head, the value projection of source limb $j$ is shared across target limbs. Target dependence enters through the attention coefficients, which determine how strongly the same source representation contributes to different targets. MetaMorph therefore performs target-dependent aggregation, while the source-to-target feature transformation itself remains implicit in the weighted mixing operation.

ModuMorph can be interpreted differently. It uses a hypernetwork to generate limb-wise projection weights $\tilde { \mathbf { W } } _ { \mathbf { e m b _ { H N } } } ^ { i }$ and biases $\mathbf { b } _ { \mathbf { e m b _ { H N } } } ^ { i }$ from the morphological context $\mathbf { C } _ { k }$ . The input embedding of limb i is given by

$$
{ \bf { E } } _ { \mathbf { H } \mathbf { N } _ { k , t } ^ { i } } = \left( { \tilde { \mathbf { W } } _ { \mathbf { e m b } _ { \mathbf { H N } } } ^ { i } } ^ { \top } { \mathbf { S } } _ { \mathbf { M } \mathbf { P } _ { k , t } ^ { i } } + \mathbf { b } _ { \mathbf { e m b } _ { \mathbf { H N } } } ^ { i } \right) \cdot \sqrt { D _ { \mathrm { e m b } } }\tag{17}
$$

Compared with a single shared projection, these morphology-conditioned limb-wise projections can make local limb representations more stable and more compatible before aggregation. In this sense, ModuMorph improves cross-limb representation compatibility through morphology-conditioned projections.

ModuMorph also differs from standard self-attention in its aggregation mechanism. Its queries and keys are generated from the morphological context $\mathbf { C } _ { k }$ , resulting in time-invariant aggregation weights $\mathbf { R } _ { \mathbf { H N } , j , i } ^ { ( l ) }$ . For limb i, the aggregation can be written as

$$
\tilde { \mathbf { r } } _ { k , t } ^ { i } = \sum _ { j \neq i } \mathbf { R } _ { \mathbf { H } \mathbf { N } , j , i } ^ { ( l ) } \top \mathbf { Z } _ { k , t } ^ { j , ( l - 1 ) } + \left( \mathbf { R } _ { \mathbf { H } \mathbf { N } _ { i , i } ^ { ( l ) } } + \mathbf { I } \right) ^ { \top } \mathbf { Z } _ { k , t } ^ { i , ( l - 1 ) }\tag{18}
$$

$$
\mathbf { Z } _ { \mathbf { H } \mathbf { N } _ { k , t } } ^ { i , ( l ) } = \mathbf { W } _ { \mathbf { M X } } ^ { ( l ) } ^ { \top } \tilde { \Gamma } _ { k , t } ^ { i } + \mathbf { b } _ { \mathbf { O } } ^ { ( l ) }\tag{19}
$$

Because $\mathbf { R } _ { \mathbf { H N } , j , i } ^ { ( l ) }$ is conditioned on morphology rather than recomputed solely from instantaneous observations, Because $\mathbf { R } _ { \mathrm { H N } , j , i } ^ { ( l ) }$ is conditioned on morphology, ModuMorph produces a morphology-dependent aggregation pattern that remains fixed for a given body. ModuMorph addresses part of this representation mismatch by generating morphology-conditioned limb projections. Each source limb can therefore enter aggregation through a body-dependent feature map. The subsequent whole-body communication still uses dense attention, so morphology primarily adapts the local feature transformation while global source-to-target aggregation remains pairwise.

This analysis leads to our central design perspective. Effective generalized morphology control requires both cross-limb contextualization and body-wide information aggregation. Existing attention-based controllers provide global aggregation, while morphology-conditioned hypernetworks improve the compatibility of limb-local features. RecMorph builds on this perspective by using morphology as a structural prior for recursive topological transport, enabling limb-local information to be progressively transformed and aggregated along the robot’s kinematic structure.

## B Universal Controller Analysis

## B.1 MetaMorph

This section analyzes MetaMorph from the perspective of cross-limb contextualization and aggregation. The goal is to characterize how its self-attention operator mixes limb-local features. While MetaMorph includes a positional encoding layer intended for morphological identification, our analysis aligns with recent findings [4] suggesting that this encoding acts primarily as a context-dependent bias rather than a robust topological descriptor. Hence we omit the positional encoding layer. This configuration is referred to as MetaMorph\*. The policy network first projects the raw observation $\mathbf { S _ { M P } } _ { k , t } ^ { i } \in \mathbb { R } ^ { D _ { \mathrm { o b s } } }$ of each limb i of robot k at time step t into a latent space $\mathbb { R } ^ { D _ { \mathrm { { c m t } } } }$ <sup>b</sup> . For a single robot k, the observation is $\mathbf { S _ { M P } } _ { k , t } \in \mathbb { \tilde { R } } ^ { D _ { \mathrm { o b s } } \times D _ { k } }$

$$
\mathbf { E } _ { k , t } = \left( \mathbf { W _ { e m b } ^ { \top } S _ { M P _ { \left. k , t \right. } } } + \mathbf { b _ { e m b } } \right) \cdot \sqrt { \mathbf { \mathit { D } } _ { \mathrm { e m b } } }\tag{20}
$$

Where $\mathbf { W _ { e m b } } \in \mathbb { R } ^ { D _ { \mathrm { o b s } } \times D _ { \mathrm { e m b } } }$ and $\mathbf { b _ { e m b } } \in \mathbb { R } ^ { D _ { \mathrm { e m b } } }$ are the weight and bias of the linear embedding layer, respectively, and they are shared parameters. While this projection maps all limb observations into a common feature dimension, the same transformation is applied to every limb. Therefore, this embedding layer does not explicitly account for limb-specific coordinate frames, joint roles, or morphology-dependent local contexts before attention-based aggregation. The multiplier $\sqrt { D _ { \mathrm { e m b } } }$ scales the embedded values to counteract the increase in variance in high-dimensional space.

Each column of the embedded matrix $\mathbf { E } _ { k , t } \in \mathbb { R } ^ { D _ { \mathrm { e m b } } \times D _ { k } }$ corresponds to a limb’s feature vector. This matrix is fed into an $L ^ { \mathrm { t h } } { \mathrm { - l a y e r } }$ Transformer Encoder, with the input to the first layer being $\mathbf { X } _ { k , t } ^ { ( 1 ) }$ . Since MetaMorph uses Multi-Head Self-Attention in each layer, we have $\mathbf { Q } _ { k , t } ^ { ( l ) } = \mathbf { K } _ { k , t } ^ { ( l ) } = \mathbf { V } _ { k , t } ^ { ( l ) } = \mathbf { X } _ { k , t } ^ { ( l ) }$ . The encoder layers model dependencies between limbs via the self-attention mechanism. For analytical clarity, we use a linearized abstraction of the attention mixing operator and omit normalization and feed-forward nonlinearities. The resulting derivation is intended to characterize the structure of cross-token aggregation.

For layer $l \in \{ 1 , \ldots , L \}$ and attention head $h \in \{ 1 , \ldots , D _ { h } \}$ , let $D _ { q }$ and $D _ { v }$ denote the query/key and value dimensions, respectively. Define

$$
\mathbf { Q } _ { h } ^ { ( l ) } = \mathbf { W } _ { \mathbf { Q } } ^ { ( l ) , h ^ { \top } } \mathbf { Q } _ { k , t } ^ { ( l ) } ,\tag{21}
$$

$$
\mathbf { K } _ { h } ^ { ( l ) } = \mathbf { W } _ { \mathbf { K } } ^ { ( l ) , h ^ { \top } } \mathbf { K } _ { k , t } ^ { ( l ) } ,\tag{22}
$$

$$
\mathbf { V } _ { h } ^ { ( l ) } = \mathbf { W } _ { \mathbf { V } } ^ { ( l ) , h ^ { \top } } \mathbf { V } _ { k , t } ^ { ( l ) } ,\tag{23}
$$

where $\mathbf { W _ { Q } ^ { ( l ) , h } } , \mathbf { W _ { K } ^ { ( l ) , h } } \in \mathbb { R } ^ { D _ { \mathrm { e m b } } \times D _ { q } }$ and $\mathbf { W } _ { \mathbf { V } } ^ { ( l ) , h } \in \mathbb { R } ^ { D _ { \mathrm { e m b } } \times D _ { \tau } }$

Throughout this analysis, the first relation index denotes the source token and the second denotes the target token. We therefore represent the attention matrix in source-by-target orientation:

$$
\Omega _ { k , t } ^ { ( l ) , h } = \mathrm { s o f t m a x } _ { \mathrm { c o l } } \left( \frac { { \mathbf { K } _ { h } ^ { ( l ) } } ^ { \top } \mathbf { Q } _ { h } ^ { ( l ) } } { \sqrt { D _ { q } } } \right) ,\tag{24}
$$

where the softmax is applied independently to each target column. The output of head h is

$$
\mathbf { H } _ { k , t } ^ { ( l ) , h } = \mathbf { V } _ { h } ^ { ( l ) } \Omega _ { k , t } ^ { ( l ) , h } .\tag{25}
$$

Let $\alpha _ { j , i } ^ { ( l ) , h }$ denote the entry in row $j$ and column i of $\Omega _ { k , t } ^ { ( l ) , h }$ . It represents the attention weight from source limb $j$ to target limb $i ,$ and therefore satisfies

$$
\sum _ { j = 1 } ^ { D _ { k } } \alpha _ { j , i } ^ { ( l ) , h } = 1 .\tag{26}
$$

Decomposing ${ \bf { H } } _ { k , t }$ by limb yields:

$$
\begin{array} { l } { { \displaystyle { \bf H } _ { k , t } ^ { ( l ) , h } = \left[ { \bf H } _ { k , t } ^ { 1 , ( l ) , h } , { \bf H } _ { k , t } ^ { 2 , ( l ) , h } , \cdots , { \bf H } _ { k , t } ^ { D _ { k } , ( l ) , h } \right] } \ ~ } \\ { { \displaystyle ~ = \left[ \sum _ { j = 1 } ^ { D _ { k } } \alpha _ { j , 1 } ^ { ( l ) , h } { \bf W } _ { \bf V } ^ { ( l ) , h } { \bf V } _ { k , t } ^ { \dag } , \sum _ { j = 1 } ^ { D _ { k } } \alpha _ { j , 2 } ^ { ( l ) , h } { \bf W } _ { \bf V } ^ { ( l ) , h } { } ^ { \top } { \bf V } _ { k , t } ^ { j , ( l ) } , \cdots , \sum _ { j = 1 } ^ { D _ { k } } \alpha _ { j , D _ { k } } ^ { ( l ) , h } { \bf W } _ { \bf V } ^ { ( l ) , h } { } ^ { \top } { \bf V } _ { k , t } ^ { j , ( l ) } \right] } } \end{array}\tag{27}
$$

(28)

Concatenating the outputs of all heads along the feature dimension gives:

$$
\mathbf { H } _ { k , t } ^ { ( l ) } = \left[ { \mathbf { H } _ { k , t } ^ { ( l ) , 1 } } ^ { \top } ; { \mathbf { H } _ { k , t } ^ { ( l ) , 2 } } ^ { \top } ; \ldots ; { \mathbf { H } _ { k , t } ^ { ( l ) , D _ { h } } } ^ { \top } \right] ^ { \top } \in \mathbb { R } ^ { ( D _ { h } \cdot D _ { v } ) \times D _ { k } } .\tag{29}
$$

Thus, the output of the multi-head attention mechanism at layer l is:

$$
\mathbf { M } _ { k , t } ^ { ( l ) } = \mathbf { W } _ { \mathbf { H } } ^ { ( l ) } ^ { \top } \mathbf { H } _ { k , t } ^ { ( l ) }\tag{30}
$$

$$
= \mathbf { W _ { H } ^ { ( l ) } } ^ { \top } \left[ \mathbf { H } _ { k , t } ^ { { ( l ) } , 1 } { } ^ { \top } ; \mathbf { H } _ { k , t } ^ { { ( l ) } , 2 } { } ^ { \top } ; \dots ; \mathbf { H } _ { k , t } ^ { { ( l ) } , { D _ { h } } } { } ^ { \top } \right] ^ { \top }\tag{31}
$$

$$
= \sum _ { h = 1 } ^ { D _ { h } } \mathbf { W _ { H } ^ { ( l ) , h } } ^ { \top } \mathbf { H } _ { k , t } ^ { ( l ) , h }\tag{32}
$$

where $\mathbf { W } _ { \mathbf { H } } ^ { ( l ) } \ \in \ \mathbb { R } ^ { ( D _ { h } \cdot D _ { v } ) \times D _ { \mathrm { c m b } } }$ is the integration weight for the multi-head attention output, which can be expanded row-wise as:

$$
\mathbf { W _ { H } ^ { ( l ) } } = \left[ { \mathbf { W _ { H } ^ { ( l ) , 1 } } } ^ { \top } ; { \mathbf { W _ { H } ^ { ( l ) , 2 } } } ^ { \top } ; \cdots ; { \mathbf { W _ { H } ^ { ( l ) , D _ { h } } } } ^ { \top } \right] ^ { \top }\tag{33}
$$

where $\mathbf { W } _ { \mathbf { H } } ^ { ( l ) , i } \in \mathbb { R } ^ { D _ { v } \times D _ { \mathbf { e m b } } }$

The output of the multi-head attention mechanism for limb i of robot k at time t and layer l is:

$$
\mathbf { M } _ { k , t } ^ { i , ( l ) } = \sum _ { h = 1 } ^ { D _ { h } } \mathbf { W } _ { \mathbf { H } } ^ { ( l ) , h } { } ^ { \top } \sum _ { j = 1 } ^ { D _ { k } } \alpha _ { j , i } ^ { ( l ) , h } \mathbf { W } _ { \mathbf { V } } ^ { ( l ) , h } { } ^ { \top } \mathbf { V } _ { k , t } ^ { j , ( l ) }\tag{34}
$$

$$
= \sum _ { j = 1 } ^ { D _ { k } } \left( \sum _ { h = 1 } ^ { D _ { h } } \alpha _ { j , i } ^ { ( l ) , h } \mathbf { W } _ { \mathbf { H } } ^ { ( l ) , h } \mathbf { W } _ { \mathbf { V } } ^ { ( l ) , h } \mathbf { \Phi } ^ { \top } \right) \mathbf { V } _ { k , t } ^ { j , ( l ) }\tag{35}
$$

$$
\mathbf { \Sigma } = \sum _ { j = 1 } ^ { D _ { k } } { \mathbf { R } _ { j , i } ^ { ( l ) } } ^ { \top } { \mathbf { V } _ { k , t } ^ { j , ( l ) } }\tag{36}
$$

Define $\begin{array} { r } { \mathbf { R } _ { j , i } ^ { ( l ) } = \sum _ { h = 1 } ^ { D _ { h } } \alpha _ { j , i } ^ { ( l ) , h } \mathbf { W } _ { \mathbf { V } } ^ { ( l ) , h } \mathbf { W } _ { \mathbf { H } } ^ { ( l ) , h } , \mathbf { R } _ { j , i } ^ { ( l ) } \in \mathbb { R } ^ { D _ { \mathrm { e m b } } \times D _ { \mathrm { e m b } } } } \end{array}$ represents the multi-head aggregated effective relation matrix from limb j to limb i in layer l of the Transformer. It quantifies the influence strength of limb $j ^ { \circ } \mathbf { s }$ latent embedding feature $\mathbf { V } _ { k , t } ^ { j , ( l ) }$ on limb i during feature extraction in layer l.

Next, $\mathbf { M } _ { k , t } ^ { ( l ) }$ is passed through a residual connection and normalization layer, then fed into a feed-forward neural network layer, followed by another residual connection and normalization layer to obtain the output of the $l ^ { \mathrm { t h } }$ Transformer encoder layer:

$$
\mathbf { Z } _ { k , t } ^ { ( l ) } = \mathbf { W _ { O } ^ { ( l ) } } ^ { \top } \left( \mathbf { M } _ { k , t } ^ { ( l ) } + \mathbf { X } _ { k , t } ^ { ( l ) } \right) + \mathbf { b } _ { \mathbf { O } } ^ { ( l ) } + \mathbf { M } _ { k , t } ^ { ( l ) } + \mathbf { X } _ { k , t } ^ { ( l ) }\tag{37}
$$

$$
\mathbf { \Gamma } = \mathbf { W _ { M X } ^ { ( l ) } } ^ { \top } \left( \mathbf { M } _ { k , t } ^ { ( l ) } + \mathbf { X } _ { k , t } ^ { ( l ) } \right) + \mathbf { b _ { O } ^ { ( l ) } }\tag{38}
$$

$$
= \left[ \mathbf { W _ { M X } ^ { ( l ) } } ^ { \top } \left( \sum _ { j = 1 } ^ { D _ { k } } \mathbf { R _ { \hat { j } , 1 } ^ { ( l ) } } ^ { \top } \mathbf { V } _ { k , t } ^ { j , ( l ) } + \mathbf { X } _ { k , t } ^ { 1 , ( l ) } \right) \right. , \cdots \left. , \mathbf { W _ { M X } ^ { ( l ) } } ^ { \top } \left( \sum _ { j = 1 } ^ { D _ { k } } \mathbf { R _ { \hat { j } , D _ { k } } ^ { ( l ) } } ^ { \top } \mathbf { V } _ { k , t } ^ { j , ( l ) } + \mathbf { X } _ { k , t } ^ { D _ { k } , ( l ) } \right) \right] + \mathbf { b _ { \hat { 0 } } ^ { ( l ) } }\tag{39}
$$

where $\mathbf { W } _ { \mathbf { O } } ^ { ( l ) } \in \mathbb { R } ^ { D _ { \mathbf { e m b } } \times D _ { \mathbf { e m b } } }$ and $\mathbf { b } _ { \mathbf { O } } ^ { ( l ) } \in \mathbb { R } ^ { D _ { \mathbf { e m b } } }$ are the weight and bias of the feed-forward neural network layer, respectively. Let $\mathbf { W } _ { \mathbf { M X } } ^ { ( l ) } = \mathbf { W } _ { \mathbf { O } } ^ { ( l ) } + \mathbf { I }$ . Then the output for limb i at encoder layer l and time t is:

$$
\mathbf { Z } _ { k , t } ^ { i , ( l ) } = \mathbf { W _ { M X } ^ { ( l ) } } ^ { \top } \left( \sum _ { j = 1 } ^ { D _ { k } } { \mathbf { R } _ { j , i } ^ { ( l ) } } ^ { \top } \mathbf { V } _ { k , t } ^ { j , ( l ) } + \mathbf { X } _ { k , t } ^ { i , ( l ) } \right) + \mathbf { b _ { O } ^ { ( l ) } }\tag{40}
$$

$$
= \mathbf { W _ { M X } ^ { ( l ) } } ^ { \top } \left( \sum _ { j = 1 } ^ { D _ { k } } { \mathbf { R } _ { j , i } ^ { ( l ) } } ^ { \top } \mathbf { X } _ { k , t } ^ { j , ( l ) } + \mathbf { X } _ { k , t } ^ { i , ( l ) } \right) + \mathbf { b _ { O } ^ { ( l ) } }\tag{41}
$$

$$
= \mathbf { W } _ { \mathbf { M X } } ^ { ( l ) } ^ { \top } \left( \sum _ { j = 1 } ^ { i - 1 } \mathbf { R } _ { j , i } ^ { ( l ) } ^ { \top } \mathbf { X } _ { k , t } ^ { j , ( l ) } + \left( \mathbf { R } _ { i , i } ^ { ( l ) } ^ { \top } + { \mathbf { I } } \right) \mathbf { X } _ { k , t } ^ { i , ( l ) } + \sum _ { j = i + 1 } ^ { D _ { k } } \mathbf { R } _ { j , i } ^ { ( l ) } ^ { \top } \mathbf { X } _ { k , t } ^ { j , ( l ) } \right) + \mathbf { b } _ { 0 } ^ { ( l ) }\tag{42}
$$

For the first encoder layer $( l = 1 ) , \mathbf { X } _ { k , t } ^ { ( 1 ) } = \mathbf { E } _ { k , t }$ . The output for limb i in the first encoder layer at time t is:

(43)

$$
\begin{array} { r l } { \mathcal { T } _ { \mathrm { e } , \mathrm { e } } ^ { ( 1 ) } , } & { \mathcal { W } _ { \mathrm { s h } } ^ { ( 1 ) , \top } \left( \frac { \rho ^ { - 1 } - 1 } { 2 } \mathbf { R } _ { s h } ^ { ( 1 ) , \top } \mathbf { R } _ { s h } ^ { ( 1 ) } + \left( \mathbf { R } _ { s h } ^ { ( 1 ) , \top } \mathbf { \Phi } _ { \mathrm { H } } ^ { ( 1 ) , \top } + \frac { \rho ^ { - 1 } } { 2 \rho + 1 } \mathbf { R } _ { s h } ^ { ( 1 ) , \top } \mathbf { R } _ { s h } ^ { ( 2 ) } \right) + \mathbf { R } _ { 0 } ^ { ( 1 ) , \top } \right. } \\ & { \left. - \mathbf { W } _ { \mathrm { s h } } ^ { ( 1 ) , \top } \left( \frac { \rho ^ { - 1 } } { \rho - 1 } \mathbf { R } _ { s h } ^ { ( 1 ) , \top } \mathbf { W } _ { \mathrm { s h } } ^ { ( 1 ) , \top } \mathbf { S } _ { \mathbf { W } } \right) ^ { - 1 } + \left( \mathbf { R } _ { s h } ^ { ( 1 ) , \top } + 1 \right) \mathbf { W } _ { \mathrm { s h } } ^ { ( 1 ) , \top } \mathbf { S } _ { \mathbf { W } } \right) ^ { - 1 } + \frac { \rho ^ { - 1 } } { 2 \rho - 1 } \mathbf { R } _ { s h } ^ { ( 1 ) , \top } \mathbf { W } _ { \mathrm { s h } } ^ { ( 1 ) , \top } \sqrt { \mathcal { B } _ { \mathbf { W } } } } \\ & { \quad + \mathbf { W } _ { \mathrm { s h } } ^ { ( 1 ) , \top } \left( \frac { \rho ^ { - 1 } } { \rho - 1 } \mathbf { R } _ { s h } ^ { ( 1 ) , \top } \mathbf { P } _ { \mathrm { s h } } + \mathbf { B } _ { \mathbf { W h } } \right) \sqrt { \mathcal { B } _ { \mathbf { W } } } - \mathbf { W } _ { \mathrm { s h } } ^ { ( 1 ) , \top } } \\ &  - \mathbf { W } _  \mathrm  s h  \end{array}\tag{44}
$$

(45)

(46)

We define

$$
\hat { \mathbf { R } } _ { j , i } ^ { ( l ) } = \mathbf { W _ { e m b } } \mathbf { R } _ { j , i } ^ { ( l ) }\tag{47}
$$

$$
\mathbf { b } _ { \mathbf { z } } ^ { ( l ) } = \mathbf { W } _ { \mathbf { u x } } ^ { ( l ) } ^ { \top } \left( \sum _ { h = 1 } ^ { D _ { h } } { \mathbf { W } _ { H } ^ { ( l ) , h } } ^ { \top } \mathbf { W } _ { \mathbf { v } } ^ { ( l ) , h } \mathbf { \Sigma } ^ { \top } \mathbf { b } _ { \mathbf { e m b } } + \mathbf { b } _ { \mathbf { e m b } } \right) \sqrt { D _ { \mathbf { e m b } } } + \mathbf { b } _ { \mathbf { O } } ^ { ( l ) }\tag{48}
$$

For encoder layers $l , \mathbf { X } _ { k , t } ^ { ( l ) } = \mathbf { Z } _ { k , t } ^ { ( l - 1 ) }$ . The output for limb i at encoder layer l and time t is:

$$
\mathbf { Z } _ { k , t } ^ { i , ( l ) } = \mathbf { W } _ { \mathrm { M X } } ^ { ( l ) } \ ^ { \top } \left( \sum _ { j = 1 } ^ { i - 1 } \mathbf { R } _ { j , i } ^ { ( l ) } { } ^ { \top } \mathbf { Z } _ { k , t } ^ { j , ( l - 1 ) } + \left( \mathbf { R } _ { i , i } ^ { ( l ) } { } ^ { \top } + \mathbf { I } \right) \mathbf { Z } _ { k , t } ^ { i , ( l - 1 ) } + \sum _ { j = i + 1 } ^ { D _ { k } } \mathbf { R } _ { j , i } ^ { ( l ) } { } ^ { \top } \mathbf { Z } _ { k , t } ^ { j , ( l - 1 ) } \right) + \mathbf { b } _ { 0 } ^ { ( l ) }\tag{49}
$$

The expansion shows that MetaMorph performs direct, state-dependent aggregation across all limb tokens. The effective relation matrix $\mathbf { R } _ { j , i } ^ { ( l ) }$ varies with the current observation, providing a flexible global contextualization mechanism. RecMorph introduces a complementary structural bias by constraining feature transport to a morphology-derived order and repeatedly applying shared recurrent transitions along that path.

The final Transformer encoder layer (L) outputs the high-dimensional feature representation of action decisions, $\mathbf { Z } _ { k , t } ^ { ( L ) }$ Since some complex tasks provide terrain information, MetaMorph performs feature encoding on the global perception information $\mathbf { S } _ { \mathbf { G } _ { k , t } }$ and concatenates it with $\mathbf { Z } _ { k , t } ^ { ( L ) }$ along the feature dimension. The concatenated vector is then passed through an output decoder layer to obtain the predicted action distribution:

$$
\mathbf { A } _ { k , t } = \mathbf { W } _ { \mathbf { d e c o d e r } } ^ { \top } \left( \mathbf { Z } _ { k , t } ^ { ( L ) } \oplus \left( \mathbf { W _ { G } } ^ { \top } \mathbf { S _ { G } } _ { k , t } + \mathbf { b _ { G } } \right) \right) + \mathbf { b _ { d e c o d e r } }\tag{50}
$$

$$
= \mathbf { W _ { d e c o d e r } ^ { 1 } } \mathbf { \widetilde { Z } } _ { k , t } ^ { ( L ) } + \mathbf { W _ { d e c o d e r } ^ { 2 } } \mathbf { \widetilde { \Psi } } \left( \mathbf { W _ { G } } ^ { \top } \mathbf { S _ { G } } _ { k , t } + \mathbf { b _ { G } } \right) + \mathbf { b _ { d e c o d e r } }\tag{51}
$$

$$
= \mathbf { W _ { d e c o d e r } ^ { 1 } } \mathbf { \Psi } ^ { \top } \mathbf { Z } _ { k , t } ^ { ( L ) } + \mathbf { W _ { d e c o d e r } ^ { 2 } } \mathbf { \Psi } ^ { \top } \mathbf { W _ { G } } \mathbf { \Psi } ^ { \top } \mathbf { S _ { G } } _ { k , t } + \mathbf { W _ { d e c o d e r } ^ { 2 } } \mathbf { \Psi } ^ { \top } \mathbf { b _ { G } } + \mathbf { b _ { d e c o d e r } }\tag{52}
$$

$$
= \mathbf { W _ { A } ^ { 1 } } ^ { \top } \mathbf { Z } _ { k , t } ^ { ( L ) } + \mathbf { W _ { A } ^ { 2 } } ^ { \top } \mathbf { S _ { G } } _ { k , t } + \mathbf { b _ { A } }\tag{53}
$$

where $\mathbf { W _ { G } } ~ \in ~ \mathbb { R } ^ { D _ { G } \times D _ { G } }$ and $\mathbf { b } _ { \mathbf { G } } ~ \in ~ \mathbb { R } ^ { D _ { G } }$ are the weight and bias for encoding the global perception information, respectively. $\mathbf { W _ { d e c o d e r } } ~ \in ~ \mathbb { R } ^ { ( D _ { \mathbf { e m b } } + D _ { G } ) \times D _ { \mathbf { o u t } } }$ and $\mathbf { b _ { d e c o d e r } } \in \mathbb { R } ^ { D _ { \mathbf { o u t } } }$ <sup>t</sup> are the weight and bias of the output decoder layer, respectively. The symbol ⊕ denotes the concatenation operation along the feature dimension. $\mathbf { W _ { d e c o d e r } }$ can be split as $\mathbf { W _ { d e c o d e r } } = \left\lceil \mathbf { W _ { d e c o d e r } ^ { 1 } } ^ { \top } ; \mathbf { W _ { d e c o d e r } ^ { 2 } } ^ { \top } \right\rceil ^ { \top }$ , where $\mathbf { W _ { d e c o d e r } ^ { 1 } } \in \mathbb { R } ^ { D _ { \mathbf { e m b } } \times D _ { \mathbf { o u t } } } , \mathbf { W _ { d e c o d e r } ^ { 2 } } \in \mathbb { R } ^ { D _ { \mathbf { G } } \times D _ { \mathbf { o u t } } }$ . By simplifying the weights and combining biases, we define: $\mathbf { W _ { A } ^ { 1 } } = \mathbf { W _ { d e c o d e r } ^ { 1 } } \mathbf { W _ { A } ^ { 2 } } = \mathbf { W _ { G } } \mathbf { W _ { d e c o d e r } ^ { 2 } } \mathbf { , b _ { A } } = \mathbf { W _ { d e c o d e r } ^ { 2 } } \mathbf { \Phi _ { b _ { G } } } + \mathbf { b _ { d e c o d e r } } .$

From the expression for $\mathbf { A } _ { k , t } ,$ it is evident that the action output by the Transformer encoder is further adjusted based on the global perception information to obtain the final action distribution.

## B.2 Modumorph

This section derives the aggregation structure of ModuMorph. Compared with MetaMorph, ModuMorph introduces two morphology-conditioned components: a hypernetwork that generates limb-wise input/output projections, and an attention module whose queries and keys are conditioned on the morphological context. We analyze these components to show how ModuMorph improves the compatibility of limb-local features before aggregation.

For robot k at time step t, in addition to the raw observation $\mathbf { S _ { M P \boldsymbol { k } , \boldsymbol { t } } } ,$ ModuMorph introduces a morphological context observation. The morphological context observation $\mathbf { C } _ { k }$ extracts morphology-related information from $\mathbf { S _ { M P } } _ { k , t } ^ { i }$ . For robot k, let $\mathbf { C } _ { k } = [ \mathbf { C } _ { k } ^ { 1 } , \dots , \mathbf { C } _ { k } ^ { D _ { k } } ] \in \mathbb { R } ^ { D _ { \mathrm { c t } \mathbf { x } } \times D _ { k } }$ denote the limb-wise morphology context. The network encodes $\mathbf { C } _ { k }$ via a multi-layer MLP encoder to generate $\mathbf { H } \mathbf { N _ { e m b } } _ { k , t }$ and $\mathbf { H N _ { a t t } } _ { k , t }$ , used for hypernetwork parameter generation and the attention mechanism, respectively:

$$
\mathbf { H } \mathbf { N _ { e m b } } _ { k , t } = \mathbf { R e L U } \left( \mathbf { W _ { e m b _ { C } } ^ { \top } C } _ { k } + \mathbf { b _ { e m b _ { C } } } \right)\tag{54}
$$

$$
\mathbf { H } \mathbf { N _ { a t t } } _ { k , t } = \mathbf { R e L U } \left( \mathbf { W } _ { \mathbf { a t t _ { C } } } ^ { \top } \mathbf { C } _ { k } + \mathbf { b _ { a t t _ { C } } } \right)\tag{55}
$$

where $\mathbf { W _ { e m b _ { C } } } , \mathbf { W _ { a t t _ { C } } } \in \mathbb { R } ^ { D _ { \mathrm { c t x } } \times D _ { \mathrm { e m b _ { c t } } } }$ and $\mathbf { b _ { e m b _ { C } } } , \mathbf { b _ { a t t _ { C } } } \in \mathbb { R } ^ { D _ { \mathrm { e m b _ { c t x } } } }$ x are the weights and biases of the morphology-context encoders, respectively.

Unlike MetaMorph’s shared embedding layer, ModuMorph utilizes a hypernetwork to dynamically generate projection weights $\mathbf { W _ { e m b _ { H N } } }$ and biases $\mathbf { b _ { e m b _ { H N } } }$ tailored to each limb:

$$
\mathbf { W _ { e m b _ { H N } } } = \mathbf { W _ { e m b _ { W } } ^ { \top } H N _ { e m b \boldsymbol { k } , t } } + \mathbf { b _ { e m b _ { W } } }\tag{56}
$$

$$
\mathbf { b _ { e m b _ { H N } } } = \mathbf { W _ { e m b _ { b } } ^ { \top } H N _ { e m b k , t } } + \mathbf { b _ { e m b _ { b } } }\tag{57}
$$

where $\begin{array} { r l r l r l r } { \mathbf { W } _ { \mathrm { e m b } _ { \mathrm { w } } } } & { \in } & { \mathbb { R } ^ { D _ { \mathrm { a n b } _ { \mathrm { t o t } } } \times ( D _ { \mathrm { a n b } } - D _ { \mathrm { a n b } } ) } , \ \mathbf { W } _ { \mathrm { e m b } _ { \mathrm { b } } } } & { \in } & { \mathbb { R } ^ { D _ { \mathrm { a n b } _ { \mathrm { t o t } } } \times D _ { \mathrm { a n b } } } , \ \mathbf { b } _ { \mathrm { e m b } _ { \mathrm { w } } } } & { \in } & { \mathbb { R } ^ { ( D _ { \mathrm { a s } } - D _ { \mathrm { a n b } } ) } , \mathbf { b } _ { \mathrm { e m b } _ { \mathrm { b } } } } & { \in } & { \mathbb { R } ^ { D _ { \mathrm { a n b } _ { \mathrm { t o t } } } } , \ \mathbf { W } _ { \mathrm { e m b } _ { \mathrm { n b } } } } & { \in } \end{array}$ ${ \mathbb { R } } ^ { ( D _ { \mathbf { o b s } } \cdot D _ { \mathbf { e m b } } ) \times D _ { k } }$ can be reshaped into $\tilde { \mathbf { W } } _ { \mathbf { e m b } _ { \mathbf { H N } } } \doteq \perp \perp _ { \mathbf { \theta } } \mathbf { { \Sigma } } _ { \mathbf { { \mathrm { R } } } } \mathbf { { \Sigma } } _ { \mathbf { { \mathrm { k } } } } \times D _ { \mathbf { \mathbf { \theta } } \mathbf { b } \mathbf { { \mathrm { s } } } } \times D _ { \mathbf { \mathbf { e m b } } }$ . Using $\tilde { \mathbf { W } } _ { \mathbf { e m b } _ { \mathbf { H N } } }$ and $\mathbf { b _ { e m b _ { H N } } }$ as the weight and bias of the linear embedding layer enables limbs of different morphological structure to possess specialized feature projection functions. Using these generated parameters, ModuMorph assigns different projection functions to different limbs according to their morphological context. This can make limb-local features more compatible before aggregation, because observations from different modules are no longer processed only by a single shared embedding matrix.

$$
\mathbf { E } _ { \mathbf { H } \mathbf { N } k , t } = \left( \tilde { \mathbf { W } } _ { \mathbf { e m b } _ { \mathbf { H } \mathbf { N } } } ^ { \top } \mathbf { S } _ { \mathbf { M P } k , t } + \mathbf { b _ { e m b } } _ { \mathbf { H } \mathbf { N } } \right) \cdot \sqrt { D _ { \mathrm { e m b } } }\tag{58}
$$

The linear embedding output for limb i of robot k at time step t is:

$$
{ \bf { E } } _ { \mathbf { H } \mathbf { N } _ { k , t } ^ { i } } = \left( { \tilde { \mathbf { W } } _ { \mathbf { e m b } _ { \mathbf { H N } } } ^ { i } } ^ { \top } { \mathbf { S } } _ { \mathbf { M } \mathbf { P } _ { k , t } ^ { i } } + \mathbf { b } _ { \mathbf { e m b } _ { \mathbf { H N } } } ^ { i } \right) \cdot \sqrt { D _ { \mathrm { e m b } } }\tag{59}
$$

where $\tilde { \mathbf { W } }$ is obtained by reshaping W, thus they are essentially the same matrix, and:

$$
\mathbf { W _ { e m b _ { H N } } ^ { \it i } } = \mathbf { W _ { e m b _ { W } } ^ { \top } H } \mathbf { N _ { e m b } } _ { k , t } ^ { \it i } + \mathbf { b _ { e m b _ { W } } }\tag{60}
$$

$$
\mathbf { \Sigma } = \mathbf { W } _ { \mathbf { e m b } _ { \mathrm { W } } } ^ { \top } \mathrm { R e L U } \left( \mathbf { W } _ { \mathbf { e m b } _ { \mathrm { C } } } ^ { \top } \mathbf { C } _ { k } ^ { i } + \mathbf { b } _ { \mathbf { e m b } _ { \mathrm { C } } } \right) + \mathbf { b } _ { \mathbf { e m b } _ { \mathrm { W } } }\tag{61}
$$

$$
\mathbf { b _ { e m b _ { H N } } ^ { \textit { i } } } = \mathbf { W _ { e m b _ { b } } ^ { \top } H N _ { e m b { k , t } } } + \mathbf { b _ { e m b _ { b } } }\tag{62}
$$

$$
= \mathbf { W } _ { \mathbf { e m b } _ { \mathrm { b } } } ^ { \top } \mathrm { R e L U } \left( \mathbf { W } _ { \mathbf { e m b } _ { \mathrm { C } } } ^ { \top } \mathbf { C } _ { k } ^ { i } + \mathbf { b } _ { \mathbf { e m b } _ { \mathrm { C } } } \right) + \mathbf { b } _ { \mathbf { e m b } _ { \mathrm { b } } }\tag{63}
$$

Thus, $\mathbf { E _ { H N } } _ { k , t } ^ { i } ,$ the linear embedding for limb i, is generated based on its own morphological context observation $\mathbf { C } _ { k } ^ { i }$

ModuMorph does not use self-attention in its Transformer Encoder. Instead, it uses $\mathbf { Q } _ { k , t } ^ { ( l ) } = \mathbf { K } _ { k , t } ^ { ( l ) } = \mathbf { H } \mathbf { N } _ { \mathrm { a t t } k , t } , \mathbf { V } _ { k , t } ^ { ( l ) } = \mathbf { X } _ { k , t } ^ { ( l ) }$ The output $\mathbf { E _ { H N } } _ { k , }$ from the linear embedding layer serves as the input $\mathbf { X } _ { k , t } ^ { ( 1 ) }$ to the first Transformer encoder layer. For subsequent layers $\left( l \geq 2 \right)$ , the input $\mathbf { X } _ { k . t } ^ { ( l ) }$ is the output $\mathbf { Z } _ { k , t } ^ { ( l - 1 ) }$ from the previous layer. Following the derivations in Appendix B.1, the output of head $h \in \{ 1 , \ldots , \tilde { D _ { h } } \}$ in the multi-head attention mechanism for layer $l \in \{ 1 , \ldots , L \}$ is:

$$
\mathbf { H } _ { \mathbf { H } \mathbf { N } _ { k , t } ^ { ( l ) , h } } = \left( \mathbf { W } _ { \mathbf { V } } ^ { ( l ) , h ^ { \top } } \mathbf { V } _ { k , t } ^ { ( l ) } \right) \boldsymbol { \Omega } _ { \mathbf { H } \mathbf { N } _ { k , t } } ( l ) , h\tag{64}
$$

where

$$
\begin{array} { r l } & { \Omega _ { \mathbf { H } \mathbf { N } _ { k , t } ^ { ( l ) , h } } = \operatorname { s o f t m a x } _ { \mathrm { c o l } } \left( \frac { \left( \mathbf { W } _ { \mathbf { K } } ^ { ( l ) , h ^ { \top } } \mathbf { K } _ { k , t } ^ { ( l ) } \right) ^ { \top } \left( \mathbf { W } _ { \mathbf { Q } } ^ { ( l ) , h ^ { \top } } \mathbf { Q } _ { k , t } ^ { ( l ) } \right) } { \sqrt { D _ { q } } } \right) } \\ & { = \operatorname { s o f t m a x } _ { \mathrm { c o l } } \left( \frac { \left( \mathbf { W } _ { \mathbf { K } } ^ { ( l ) , h ^ { \top } } \mathbf { H } \mathbf { N } _ { \mathbf { a t } { k } , t } \right) ^ { \top } \left( \mathbf { W } _ { \mathbf { Q } } ^ { ( l ) , h ^ { \top } } \mathbf { H } \mathbf { N } _ { \mathbf { a t } { k } , t } \right) } { \sqrt { D _ { q } } } \right) } \end{array}\tag{65}
$$

(66)

Let $\alpha _ { \mathrm { H N } _ { j , i } } ^ { } ( l ) , h$ denote the entry in row j and column i of $\Omega _ { \mathrm { H N } , k , t } ^ { ( l ) , h }$ . It represents the morphology-conditioned attention weight from source limb j to target limb i. Then, the output for limb i in the first Transformer encoder layer at time t is:

$$
\mathbf { Z } _ { \mathrm { H N } _ { k , t } ^ { i } } ( 1 ) = \mathbf { W } _ { \mathrm { M X } } ^ { ( 1 ) ^ { \top } } \left( \sum _ { j = 1 } ^ { i - 1 } \hat { \mathbf { R } } _ { \mathrm { H N } _ { j , i } } ^ { ( 1 ) } \mathbb { T } \mathbf { S } _ { \mathrm { W } _ { k , t } ^ { j } } + \left( \hat { \mathbf { R } } _ { \mathrm { H N } _ { i , i } , i } ^ { ( 1 ) } + \bar { \mathbf { W } } _ { \mathrm { e m b a x } } ^ { i } \right) ^ { \top } \mathbf { S } _ { \mathrm { M P } _ { k , t } ^ { i } } + \sum _ { j = i + 1 } ^ { D _ { k } } \hat { \mathbf { R } } _ { \mathrm { H N } _ { j , i } } ^ { ( 1 ) } \mathbb { T } \mathbf { S } _ { \mathrm { M P } _ { k , t } ^ { j } } \right) \sqrt { D _ { \mathrm { e m b } } } + \mathbf { b } _ { \mathbf { H } \mathbf { N } _ { k } ^ { i } } ( 1 )\tag{67}
$$

where

$$
\hat { \mathbf { R } } _ { \mathbf { H N } , j , i } ^ { ( l ) } = \tilde { \mathbf { W } } _ { \mathbf { e m b } _ { \mathrm { H N } } } ^ { j } \mathbf { R } _ { \mathbf { H N } , j , i } ^ { ( l ) }\tag{68}
$$

$$
\mathbf { \Xi } = \tilde { \mathbf { W } } _ { \mathbf { e m b _ { H N } } } ^ { j } \sum _ { h = 1 } ^ { D _ { h } } \alpha _ { \mathrm { H N } _ { j , i } ^ { ( l ) , h } } \mathbf { W } _ { \mathbf { V } } ^ { ( l ) , h } \mathbf { W } _ { H } ^ { ( l ) , h }\tag{69}
$$

$$
\mathbf { b } _ { \mathbf { H } \mathbf { N } _ { \mathbf { z } } } ^ { i , ( l ) } = \mathbf { W } _ { \mathbf { M } \mathbf { X } } ^ { ( l ) } \left( \sum _ { h = 1 } ^ { D _ { h } } \sum _ { j = 1 } ^ { D _ { k } } \alpha _ { \mathbf { H } \mathbf { N } _ { j , i } ^ { ( l ) , h } } \mathbf { W } _ { \mathbf { H } } ^ { ( l ) , h } \mathbf { W } _ { \mathbf { V } } ^ { ( l ) , h } \mathbf { W } _ { \mathbf { V } } ^ { \top } \mathbf { b } _ { \mathbf { e m b } _ { \mathbf { H } \mathbf { N } } } ^ { j } + \mathbf { b } _ { \mathbf { e m b } _ { \mathbf { H } \mathbf { N } } } ^ { i } \right) \sqrt { \mathbf { D } _ { \mathbf { e m b } } } + \mathbf { b } _ { \mathbf { O } } ^ { ( l ) }\tag{70}
$$

For encoder layers $l \geq 2 ,$ , where $\mathbf { X } _ { k , t } ^ { ( l ) } = \mathbf { Z } _ { k , t } ^ { ( l - 1 ) }$ , the output for limb i in layer l at time t is:

$$
\mathbf { Z _ { H N } } _ { k , t } ^ { i , ( l ) } = \mathbf { W _ { M N } } ^ { ( l ) }  { \mathop { ( \sum _ { j = 1 } ^ { i - 1 } \mathbf { R _ { H N , j , i } ^ { ( l ) } } \mathbf { \bar { T } } } } _ { k , t } ^ { j , ( l - 1 ) } + ( \mathbf { R _ { H N } } _ { i , i } ^ { ( l ) } + \mathbf { I } ) ^ { \top } \mathbf { Z } _ { k , t } ^ { i , ( l - 1 ) } + \sum _ { j = i + 1 } ^ { D _ { k } } \mathbf { R _ { H N , j , i } ^ { ( l ) } } \mathbf { \bar { Z } } _ { k , t } ^ { j , ( l - 1 ) } )  + \mathbf { b _ { 0 } ^ { ( l ) } }\tag{71}
$$

Because ${ \bf R } _ { \mathrm { H N } , j , i } ^ { ( l ) }$ is conditioned on morphology, ModuMorph provides a morphology-dependent aggregation pattern that remains fixed for a given body. Together with its generated limb-wise projections, this mechanism improves crosslimb representation compatibility before and during global aggregation. RecMorph uses morphology differently: kinematic structure determines the path along which shared recurrent transitions transport and contextualize limb features.

Similarly, the output decoder layer in ModuMorph dynamically generates projection weights $\mathbf { W _ { d e c o d e r _ { H N } } }$ and biases $\mathbf { b _ { d e c o d e r _ { H N } } }$ for each limb via a hypernetwork:

$$
\mathbf { W _ { d e c o d e r _ { H N } } } = \mathbf { W } _ { \mathbf { d e c o d e r _ { W } } } ^ { \top } \mathbf { H } \mathbf { N _ { e m b k , \ t } } + \mathbf { b _ { d e c o d e r _ { W } } }\tag{72}
$$

$$
\mathbf { b _ { d e c o d e r _ { H N } } } = \mathbf { W _ { d e c o d e r _ { b } } ^ { \top } H N _ { e m b \boldsymbol { k } , \boldsymbol { t } } } + \mathbf { b _ { d e c o d e r _ { b } } }\tag{73}
$$

where W<sub>decoder</sub> $\begin{array} { r l r l r l } & { \in } & { \mathbb { R } ^ { D _ { \mathrm { e n t } _ { \mathrm { G r } } } \times ( ( D _ { \mathrm { e n t } } + D _ { G } ) \cdot D _ { \mathrm { o u t } } ) } , \mathbf { W } _ { \mathrm { d e c o d e r } _ { \mathrm { b } } } } & { \in } & { \mathbb { R } ^ { D _ { \mathrm { e n t } _ { \mathrm { G r } } } \times D _ { \mathrm { o u t } } } , \mathbf { b } _ { \mathrm { d e c o d e r } _ { \mathrm { b } } } } & { \in } & { \mathbb { R } ^ { ( D _ { \mathrm { e n t } } + D _ { G } ) \cdot D _ { \mathrm { o u t } } } , \mathbf { b } _ { \mathrm { d e c o d e r } _ { \mathrm { b } } } } & { \in } \end{array}$ $\mathbb { R } ^ { D _ { \mathbf { o u t } } } . \mathbf { W } _ { \mathrm { d e c o d e r H N } } \in \mathbb { R } ^ { ( ( D _ { \mathbf { e m b } } + D _ { G } ) D _ { \mathbf { o u t } } ) \times D _ { k } }$ . Each column is reshaped into a limb-specific decoder matrix $\ddot { \mathbf { W } } _ { \mathrm { d e c o d e r H N } } ^ { i } \in$ $\mathbb { R } ^ { ( D _ { \mathrm { e m b } } + D _ { G } ) \times D _ { \mathrm { o u t } } }$ . The generated parameters are reshaped into limb-specific decoder weights and biases, providing each limb with a morphology-conditioned action projection from the shared contextual representation. For tasks providing global perception, ModuMorph encodes the global perception information $\mathbf { S } _ { \mathbf { G } k , t }$ , concatenates it with $\mathbf { Z _ { H N } } _ { k , t } ^ { \left( L \right) }$ along the feature dimension, and finally passes the concatenated vector through the output decoder layer to obtain the predicted action distribution:

$$
\mathbf { A } _ { \mathrm { H N } k , t } = \mathbf { W _ { \mathrm { d e c o d e r } _ { \mathrm { H N } } } } ^ { \top } \left( \mathbf { Z } _ { \mathrm { H N } _ { k , t } ^ { \left( L \right) } } \oplus \left( \mathbf { W _ { G } } ^ { \top } \mathbf { S } _ { \mathbf { G } k , t } + \mathbf { b } _ { \mathbf { G } } \right) \right) + \mathbf { b } _ { \mathbf { d e c o d e r } _ { \mathrm { H N } } }\tag{74}
$$

$$
= \mathbf { W _ { d e c o d e r _ { \mathrm { H N } } } ^ { 1 } } ^ { \top } \mathbf { Z _ { H N } } _ { k , t } ^ { ( L ) } + \mathbf { W _ { d e c o d e r _ { \mathrm { H N } } } ^ { 2 } } ^ { \top } \left( \mathbf { W _ { G } } ^ { \top } \mathbf { S _ { G } } _ { k , t } + \mathbf { b _ { G } } \right) + \mathbf { b _ { d e c o d e r _ { \mathrm { H N } } } }\tag{75}
$$

$$
= \mathbf { W _ { d e c o d e r _ { H N } } ^ { 1 } } \mathbf { \Sigma ^ { \top } } \mathbf { Z _ { H N } } _ { k , t } ^ { ( L ) } + \mathbf { W _ { d e c o d e r _ { H N } } ^ { 2 } } \mathbf { \Sigma ^ { \top } } \mathbf { W _ { G } } \mathbf { \Sigma ^ { \top } } \mathbf { S _ { G } } _ { k , t } + \mathbf { W _ { d e c o d e r _ { H N } } ^ { 2 } } \mathbf { \Sigma ^ { \top } } \mathbf { b _ { G } } + \mathbf { b _ { d e c o d e r _ { H N } } }\tag{76}
$$

$$
= { \mathbf { W } _ { A _ { \mathrm { H N } } } ^ { 1 } } ^ { \top } { \mathbf { Z } _ { \mathbf { H N } _ { k , t } ^ { ( L ) } } } + { \mathbf { W } _ { A _ { \mathrm { H N } } } ^ { 2 } } ^ { \top } { \mathbf { S } _ { \mathbf { G } _ { k , t } } } + { \mathbf { b } _ { A _ { \mathrm { H N } } } }\tag{77}
$$

## B.3 NerveNet

We analyze a linearized abstraction of the NerveNet-style message-passing operator to contrast graph-local aggregation with RecMorph’s ordered recurrent transport.

For robot k at time step $t ,$ let $\mathbf { h } _ { k , t } ^ { i , ( l ) } \in \overline { { \mathbb { R } } } ^ { D _ { \mathrm { n o d e } } }$ denote the hidden state feature vector of limb i at the $l ^ { \mathrm { t h } }$ message-passing iteration. The initial state $\mathbf { h } _ { k , t } ^ { i , ( 0 ) }$ is derived from the linear embedding of the raw observation $\mathbf { S _ { M P } } _ { k , t } ^ { i }$

In NerveNet, the graph propagation relies on the adjacency matrix $\mathbf { a } \in \mathbb { R } ^ { D _ { k } \times D _ { k } }$ defined by the robot’s morphology, where $a _ { i , j }$ indicates the connection strength from limb $j$ to limb $i .$ The update rule for node i involves message generation, aggregation, self-feature transformation, and a residual update. To isolate the neighborhood aggregation structure, we analyze a linearized form of the message-passing update and omit nonlinear activations and normalization from the algebraic expansion.

First, each node projects its current feature into a message space. For a neighboring limb $j ,$ the generated message is:

$$
\mathbf { m } _ { k , t } ^ { j , ( l ) } = \mathbf { W _ { p r o j } } ^ { \top } \mathbf { h } _ { k , t } ^ { j , ( l ) } + \mathbf { b _ { p r o j } }\tag{78}
$$

where $\mathbf { W _ { p r o j } } \in \mathbb { R } ^ { D _ { \mathbf { n o d e } } \times D _ { \mathbf { m s } \mathbf { \xi } } }$ and $\mathbf { b _ { p r o j } } \in \mathbb { R } ^ { D }$ <sup>msg</sup> are shared projection parameters.

The messages from all neighboring nodes $j \in \mathcal { N } ( i )$ are then aggregated via isotropic weighted summation based on the adjacency matrix:

$$
\bar { \mathbf { m } } _ { k , t } ^ { i , ( l ) } = \sum _ { j \in \mathcal { N } ( i ) } a _ { i , j } \mathbf { m } _ { k , t } ^ { j , ( l ) } = \sum _ { j \in \mathcal { N } ( i ) } a _ { i , j } \left( \mathbf { W _ { p r o j } } ^ { \top } \mathbf { h } _ { k , t } ^ { j , ( l ) } + \mathbf { b _ { p r o j } } \right)\tag{79}
$$

Simultaneously, node i performs a linear transformation on its own feature representation:

$$
\mathbf { s } _ { k , t } ^ { i , ( l ) } = \mathbf { W _ { s e l f } } ^ { \top } \mathbf { h } _ { k , t } ^ { i , ( l ) } + \mathbf { b _ { s e l f } }\tag{80}
$$

The self-feature $\mathbf { s } _ { k , t } ^ { i , ( l ) }$ and the aggregated neighbor message m¯ $\mathbf { \chi } _ { \cdot k , t } ^ { i , ( l ) }$ are concatenated and passed through a shared messagefusion layer. Let $\mathbf { W _ { m s g } } \in \mathbb { R } ^ { ( D _ { \mathbf { n o d e } } + D _ { \mathbf { m s g } } ) \times D _ { \mathbf { n o d e } } }$ be the weight matrix of this fusion layer. We can split $\mathbf { W _ { m s g } }$ along the feature dimension as $\mathbf { W _ { m s g } } = \mathbf { \left[ W _ { m s g . s e l f } \right]} ^ { \top } ; \mathbf { W _ { m s g . n e i g h } } ^ { \top }  ^ { \top }$ . The fused output $\mathbf { z } _ { k , t } ^ { i , ( l ) }$ is:

$$
\mathbf { z } _ { k , t } ^ { i , ( l ) } = \mathbf { W _ { m s g } } ^ { \top } \left( \mathbf { s } _ { k , t } ^ { i , ( l ) } \oplus \bar { \mathbf { m } } _ { k , t } ^ { i , ( l ) } \right) + \mathbf { b _ { m s g } }\tag{81}
$$

$$
\mathbf { \Sigma } = \mathbf { W _ { m s g . s e l f } } ^ { \top } \mathbf { s } _ { k , t } ^ { i , ( l ) } + \mathbf { W _ { m s g . n e i g h } } ^ { \top } \bar { \mathbf { m } } _ { k , t } ^ { i , ( l ) } + \mathbf { b _ { m s g } }\tag{82}
$$

Finally, NerveNet utilizes a residual connection via an output projection layer to update the node representation for the next iteration:

$$
\mathbf { h } _ { k , t } ^ { i , ( l + 1 ) } = \mathbf { h } _ { k , t } ^ { i , ( l ) } + \mathbf { W _ { o u t } } ^ { \top } \mathbf { z } _ { k , t } ^ { i , ( l ) } + \mathbf { b _ { o u t } }\tag{83}
$$

By substituting the expanded forms of $\mathbf { s } _ { k , t } ^ { i , ( l ) }$ and $\bar { \mathbf { m } } _ { k , t } ^ { i , ( l ) }$ into the final update equation, we obtain the holistic expression:

$$
\begin{array} { r l } & { \mathbf { h } _ { k , t } ^ { i , ( l + 1 ) } = \mathbf { h } _ { k , t } ^ { i , ( l ) } + \mathbf { W _ { o u t } } ^ { \top } \left( \mathbf { W _ { \mathrm { m s g , s e l f } } } ^ { \top } \left( \mathbf { W _ { \mathrm { s e l f } } } ^ { \top } \mathbf { h } _ { k , t } ^ { i , ( l ) } + \mathbf { b _ { \mathrm { s e l f } } } \right) \right. } \\ & { \qquad \left. + \mathbf { W _ { \mathrm { m s g , n e i g h } } } ^ { \top } \left( \displaystyle \sum _ { j \in \mathcal { N } ( i ) } a _ { i , j } \left( \mathbf { W _ { \mathrm { p r o j } } } ^ { \top } \mathbf { h } _ { k , t } ^ { j , ( l ) } + \mathbf { b _ { \mathrm { p r o j } } } \right) \right) + \mathbf { b _ { \mathrm { m s g } } } \right) + \mathbf { b _ { \mathrm { o u t } } } } \\ & { = \left( \mathbf { I } + \mathbf { W _ { o u t } } ^ { \top } \mathbf { W _ { \mathrm { m s g , s e l f } } } ^ { \top } \mathbf { W _ { \mathrm { s e l f } } } ^ { \top } \mathbf { h } _ { k , t } ^ { i , ( l ) } \right. } \\ & { \quad \quad \left. + \left( \mathbf { W _ { o u t } } ^ { \top } \mathbf { W _ { \mathrm { m s g , n e i g h } } } ^ { \top } \mathbf { W _ { \mathrm { p r o j } } } ^ { \top } \right) \displaystyle \sum _ { j \in \mathcal { N } ( i ) } a _ { i , j } \mathbf { h } _ { k , t } ^ { j , ( l ) } + \mathbf { B _ { \mathrm { t o t a l } } ^ { ( l ) } } \right. } \end{array}\tag{84}
$$

(85)

To reveal the algebraic essence of the GNN propagation, we define the composite transformation matrices:

$$
\mathbf { W _ { S T } } = \mathbf { W _ { s e l f } } \mathbf { W _ { m s g . s e l f } } \mathbf { W _ { o u t } }\tag{86}
$$

$$
\mathbf { W _ { M T } } = \mathbf { W _ { p r o j } } \mathbf { W _ { m s g , n e i g h } } \mathbf { W _ { o u t } }\tag{87}
$$

and the consolidated bias term:

$$
\begin{array} { r l } & { \mathbf { B _ { \mathrm { t o t a l } } ^ { ( l ) } } = \mathbf { W _ { \mathrm { o u t } } } ^ { \top } \mathbf { W _ { \mathrm { m s g . s e l f } } } ^ { \top } \mathbf { b _ { \mathrm { s e l f } } } } \\ & { \qquad + \mathbf { W _ { \mathrm { o u t } } } ^ { \top } \mathbf { W _ { \mathrm { m s g . n e i g h } } } ^ { \top } \left( \displaystyle \sum _ { j \in \mathcal { N } ( i ) } a _ { i , j } \mathbf { b _ { p r o j } } \right) + \mathbf { W _ { \mathrm { o u t } } } ^ { \top } \mathbf { b _ { \mathrm { m s g } } } + \mathbf { b _ { \mathrm { o u t } } } } \end{array}\tag{88}
$$

This simplifies the NerveNet update mechanism into the following mathematically transparent form:

$$
\mathbf { h } _ { k , t } ^ { i , ( l + 1 ) } = \left( \mathbf { I } + \mathbf { W _ { S T } } ^ { \top } \right) \mathbf { h } _ { k , t } ^ { i , ( l ) } + \mathbf { W _ { M T } } ^ { \top } \sum _ { j \in \mathcal { N } ( i ) } a _ { i , j } \mathbf { h } _ { k , t } ^ { j , ( l ) } + \mathbf { B _ { t o t a l } ^ { ( l ) } }\tag{89}
$$

Interpretation of the Message-Passing Operator. The derived update shows that NerveNet aggregates neighboring information through the weighted summation $\bar { \sum _ { j \in \mathcal { N } ( i ) } a _ { i , j } \mathbf { h } _ { k , t } ^ { j , ( l ) } }$ followed by a shared projection $\mathbf { W _ { M T } }$ . Increasing graphmessage-passing depth expands the receptive field, but each additional layer aggregates representations that already contain neighborhood mixtures from previous layers. Consequently, global communication and repeated neighborhood mixing grow together with message passing depth. RecMorph also performs repeated transformations, but the repetition occurs along an ordered hidden-state trajectory inside one spatial sequence operator instead of through successively deeper neighborhoodpooling layers.

## C Linearized Analysis of Topology-Guided Recurrent Transport

Under the linearized analysis, RMS normalization and the SiLU input nonlinearity are absorbed into an effective input to-hidden map. Relative to Sec. III, the recurrent parameters correspond to

$$
\begin{array} { r l } & { \overrightarrow { \mathbf { W } } _ { h } ^ { ( l ) ^ { \top } } \equiv A _ { \right. } ^ { ( l ) } , } \\ & { \overleftarrow { \mathbf { W } } _ { h } ^ { ( l ) ^ { \top } } \equiv A _ { \left. } ^ { ( l ) } , } \end{array}
$$

$$
\begin{array} { r } { \vec { \bf W } _ { x } ^ { ( l ) } { } ^ { \top } \equiv B _ {  } ^ { ( l ) } W _ { x } ^ { ( l ) } , } \end{array}\tag{90}
$$

$$
\begin{array} { r } { \{ \overline { { \mathbf { W } } } _ { x } ^ { ( l ) } \} ^ { \top } \equiv B _ {  } ^ { ( l ) } W _ { x } ^ { ( l ) } . } \end{array}\tag{91}
$$

Let $\mathbf { X } _ { k , t } ^ { i , ( l ) }$ denote the representation presented to the recurrent transition at token i and block l. The output projection $\mathbf { W } _ { \mathrm { o u t } } ^ { ( l ) } ^ { \top }$ corresponds to the linearized bidirectional contextual projection before residual integration. Bias terms are omitted from the source-dependent influence expansion because they contribute additive terms independent of the source-token index.

The policy network first projects the raw observation $\mathbf { S _ { M P } } _ { k , t }$ for each limb of robot k at time step t onto a latent space using a linear embedding layer to obtain the embedded vector $\mathbf { E } _ { k , t }$ . The embedded limb vectors $\mathbf { E } _ { k , t }$ are then fed into the BiRNN. Unlike the global, parallel attention of the Transformer, the BiRNN models inter-limb dependencies via bidirectional recursive information propagation along the morphological topological order. We use Depth-First Search (DFS) by default to form the morphological topological order, as DFS naturally places physically connected limbs consecutively in the sequence, preserving the locality of limb combinations. This ordering allows the recurrent backbone to propagate information through sequence neighborhoods that often correspond to physically connected or nearby modules. We denote the input to the $l ^ { \mathrm { t h } }$ layer as $\mathbf { X } _ { k , t } ^ { ( l ) }$ , where $\mathbf { X } _ { k , t } ^ { ( 1 ) } = \mathbf { E } _ { k , \iota }$ <sub>t</sub>.

The BiRNN layer consists of two independent directions. The forward recurrence processes the sequence from 1 toward $D _ { k }$ , while the backward recurrence processes it from $D _ { k }$ toward 1. Consequently, token i incorporates context from tokens $1 , \ldots , i$ through the forward state and from tokens $i , \ldots , D _ { k }$ through the backward state.

$$
\overrightarrow { \mathbf { h } } _ { k , t } ^ { i , ( l ) } = \sigma \left( \overrightarrow { \mathbf { W } } _ { \mathbf { h } } ^ { ( l ) } ^ { \top } \overrightarrow { \mathbf { h } } _ { k , t } ^ { i - 1 , ( l ) } + \overrightarrow { \mathbf { W } } _ { \mathbf { x } } ^ { ( l ) } ^ { \top } \mathbf { X } _ { k , t } ^ { i , ( l ) } + \overrightarrow { \mathbf { b } } ^ { ( l ) } \right)\tag{92}
$$

$$
\overleftarrow { \mathbf { h } } _ { k , t } ^ { i , ( l ) } = \sigma \left( \overleftarrow { \mathbf { W } } _ { h } ^ { ( l ) } { } ^ { \top } \overleftarrow { \mathbf { h } } _ { k , t } ^ { i + 1 , ( l ) } + \overleftarrow { \mathbf { W } } _ { x } ^ { ( l ) } { } ^ { \top } \mathbf { X } _ { k , t } ^ { i , ( l ) } + \overleftarrow { \mathbf { b } } ^ { ( l ) } \right)\tag{93}
$$

Here, $\vec { \mathbf { h } } _ { k , t } ^ { i , ( l ) }$ and $\mathbf { \widetilde { h } } _ { k , t } ^ { i , ( l ) }$ represent the hidden states for limb i in the forward and backward processes, respectively. $\widehat { \mathbf { W } } _ { \mathbf { x } } ^ { ( l ) }$ and $\overleftarrow { \mathbf { W } } _ { \mathbf { x } } ^ { ( l ) }$ are the projection matrices from input to hidden state for the forward and backward processes, respectively. $\widehat { \mathbf { W } } _ { \mathbf { h } } ^ { ( l ) }$ and $\smash { \overleftarrow { \mathbf { W } } _ { \mathbf { h } } ^ { ( l ) } }$ are the recurrent weight matrices between hidden states for the forward and backward processes, respectively. $\vec { \mathbf { b } } ^ { ( l ) }$ and $\overleftarrow { \mathbf { b } } ^ { ( l ) }$ are the bias vectors for the forward and backward processes, respectively. σ is a nonlinear activation function. For derivation coherence, we assume the initial states $\vec { \mathbf { h } } _ { k , t } ^ { 0 , ( l ) }$ and $\mathbf { \widetilde { h } } _ { k , t } ^ { D _ { k } + 1 , ( l ) }$ are zero vectors and approximate σ as a linea function.

At layer l, the output $\mathbf { Z } _ { k , t } ^ { i , ( l ) }$ for limb i is obtained by concatenating the forward and backward hidden states and applying a linear projection:

$$
\mathbf { Z } _ { k , t } ^ { i , ( l ) } = \mathbf { W _ { o u t } ^ { ( l ) } } ^ { \top } \left( \overrightarrow { \mathbf { h } } _ { k , t } ^ { i , ( l ) } \oplus \overleftarrow { \mathbf { h } } _ { k , t } ^ { i , ( l ) } \right) + \mathbf { b _ { o u t } ^ { ( l ) } }\tag{94}
$$

where $\oplus$ denotes vector concatenation, $\mathbf { W } _ { \mathbf { o u t } } ^ { ( l ) }$ is the output projection matrix, and $\mathbf { b } _ { \mathbf { o u t } } ^ { ( l ) }$ is the output projection bias. At this point, $\mathbf { Z } _ { k , t } ^ { i , ( l ) }$ already incorporates information from the entire morphological sequence. To characterize the sequence distance-dependent transport induced by the BiRNN, we recursively expand the hidden states.

Taking the forward process as an example, the hidden state $\vec { \mathbf { h } } _ { k , t } ^ { i , ( l ) }$ for limb i can be expanded as a cumulative function of inputs from preceding limbs. Approximating by ignoring the nonlinear effects of the activation function for linear analysis, we have:

$$
\begin{array} { l } { { \displaystyle { \overrightarrow { \bf h } } _ { k , t } ^ { i , ( l ) } \approx \sum _ { j = 1 } ^ { i } \left( \overrightarrow { \bf W } _ { { \bf h } } ^ { ( l ) } { } ^ { \top } \right) ^ { i - j } \overrightarrow { \bf W } _ { { \bf x } } ^ { ( l ) } ^ { \top } } \mathbf { X } _ { k , t } ^ { j , ( l ) } } \ ~  \\ { { \displaystyle ~ = \overrightarrow { \bf W } _ { { \bf x } } ^ { ( l ) } { } ^ { \top } { \bf X } _ { k , t } ^ { i , ( l ) } + \overrightarrow { \bf W } _ { { \bf h } } ^ { ( l ) } { } ^ { \top } \overrightarrow { \bf W } _ { { \bf x } } ^ { ( l ) } ^ { \top } { \bf X } _ { k , t } ^ { i - 1 , ( l ) } + \cdots + \left( \overrightarrow { \bf W } _ { { \bf h } } ^ { ( l ) } { } ^ { \top } \right) ^ { i - 1 } \overrightarrow { \bf W } _ { { \bf x } } ^ { ( l ) } ^ { \top } } \mathbf { X } _ { k , t } ^ { 1 , ( l ) } } \end{array}\tag{95}
$$

(96)

Similarly, the backward hidden state expands as:

$$
\mathbf { \overleftarrow { h } } _ { k , t } ^ { i , ( l ) } \approx \sum _ { j = i } ^ { D _ { k } } { \left( \mathbf { \overleftarrow { W } } _ { \mathbf { h } } ^ { ( l ) } \right) ^ { \top } } \mathbf { \overleftarrow { W } } _ { \mathbf { x } } ^ { ( l ) } \mathbf { \overbar { \mathbf { X } } } _ { k , t } ^ { j , ( l ) }\tag{97}
$$

The output feature $\mathbf { Z } _ { k , t } ^ { i , ( l ) }$ for limb i of robot k at time t in the $l ^ { \mathrm { t h } }$ BiRNN layer takes the form

$$
\begin{array} { r l } & { \mathbf { Z } _ { k , t } ^ { ( l , l ) } = \mathbf { W } _ { \mathrm { o u t } } ^ { ( l ) } [ \overset { , } { \mathbf { \overbrace { h } } _ { k , t } ^ { i } } \oplus \overset { , } { \mathbf { \overbrace { h } } _ { k , t } ^ { i , ( l ) } } + \mathbf { \dot { b } } _ { \mathrm { o u t } } ^ { ( l ) }  } \\ & { \qquad = \overset { , } { \mathbf { \overbrace { W } } _ { \mathrm { o u t } } ^ { ( l ) } } ^ { \top } \displaystyle \sum _ { j = 1 } ^ { i } ( \begin{array} { l } { \overrightarrow { \mathbf { W } } _ { \mathbf { h } } ^ { ( l ) } ^ { \top } } \end{array} ) ^ { i - j } \overset { , } { \overbrace { \mathbf { W } _ { \mathbf { x } } ^ { ( l ) } ^ { \mathcal { I } } } ^ { \mathcal { I } , ( l ) } } ^ { \mathcal { I } , ( l ) } + \overset { , } { \mathbf { \overbrace { W } } _ { \mathrm { o u t } } ^ { ( l ) } } \displaystyle \sum _ { j = i } ^ { T } ( \overleftarrow { \mathbf { W } } _ { \mathbf { h } } ^ { ( l ) } ^ { \top } ) ^ { j - i } \overset { , } { \overleftarrow { \mathbf { W } } _ { \mathbf { x } } ^ { ( l ) } } ^ { \mathcal { I } , ( l ) } + \mathbf { b } _ { \mathrm { o u t } } ^ { ( l ) } } \\ &  \qquad = \displaystyle \sum _ { j = 1 } ^ { i - 1 } \overrightarrow { \mathbf { W } } _ { \mathrm { o u t } } ^ { ( l ) } ^ { \top } ( \overrightarrow { \mathbf { W } } _ { \mathbf { h } } ^ { ( l ) } ^ { \top } ) ^ { i - j } \overrightarrow { \mathbf { W } } _ { \mathbf { x } } ^ { ( l ) } ^ { \top } \mathbf { X } _ { k , t } ^ { ( j ) } + ( \overrightarrow { \mathbf { W } } _ { \mathrm { o u t } } ^  ( \end{array}\tag{98}
$$

(99)

(100)

Here, $\left( \mathbf { W _ { h } ^ { ( l ) } } ^ { \top } \right) ^ { i - j } \mathrm { ~ a n d ~ } \left( \mathbf { W _ { h } ^ { ( l ) } } ^ { \top } \right) ^ { j - i }$ are defined as equivalent recurrent propagation weight matrices. For $j < i ,$ the source-to-target contribution is weighted by

$$
\overrightarrow { \mathbf { W } } _ { \mathrm { o u t } } ^ { ( l ) } ( \overrightarrow { \mathbf { W } } _ { h } ^ { ( l ) } ) ^ { \top } \overrightarrow { \mathbf { W } } _ { x } ^ { ( l ) }  { \top } ,
$$

with the analogous backward expression for $j > i ,$ . Thus, the number of repeated recurrent transformations is determined by the source-target separation in the serialized morphology. This gives the serialized morphology a direct role in shaping the depth of feature transport between limbs. This analysis characterizes how repeated shared transitions progressively contextualize limb-local representations before action decoding. Unlike input-dependent global attention, the recurrent transition parameters are shared along the morphology-derived sequence, providing a consistent structural bias across bodies.

The high-dimensional feature representation $\mathbf { Z } _ { k , t } ^ { ( L ) }$ output by the $L ^ { \mathrm { t h } }$ BiRNN encoder layer serves as the preliminary action decision feature. Similar to the Transformer version, when global terrain information is available, RecMorph encodes the global perception information $\mathbf { S } _ { \mathbf { G } k , t }$ and concatenates it with $\mathbf { Z } _ { k , t } ^ { ( L ) }$ , finally passing it through a decoder layer to obtain the action distribution prediction:

$$
\mathbf { A } _ { k , t } = \mathbf { W } _ { \mathbf { d e c o d e r } } ^ { \top } \left( \mathbf { Z } _ { k , t } ^ { ( L ) } \oplus \left( \mathbf { W _ { G } } ^ { \top } \mathbf { S _ { G } } _ { k , t } + \mathbf { b _ { G } } \right) \right) + \mathbf { b _ { d e c o d e r } }\tag{101}
$$

$$
= \mathbf { W _ { d e c o d e r } ^ { 1 } } \mathbf { \widetilde { Z } } _ { k , t } ^ { ( L ) } + \mathbf { W _ { d e c o d e r } ^ { 2 } } \mathbf { \widetilde { \Psi } } \left( \mathbf { W _ { G } } ^ { \top } \mathbf { S _ { G } } _ { k , t } + \mathbf { b _ { G } } \right) + \mathbf { b _ { d e c o d e r } }\tag{102}
$$

$$
= \mathbf { W _ { A } ^ { 1 } } ^ { \top } \mathbf { Z } _ { k , t } ^ { ( L ) } + \mathbf { W _ { A } ^ { 2 } } ^ { \top } \mathbf { S _ { G } } _ { k , t } + \mathbf { b _ { A } }\tag{103}
$$

where $\mathbf { W _ { G } }$ are the global perception encoding parameters, and $\mathbf { W _ { d e c o d e r } }$ are the decoder layer parameters. The decoder matrix is partitioned into $\mathbf { W _ { d e c o d e r } ^ { 1 } }$ for processing proprioceptive features and $\mathbf { W _ { d e c o d e r } ^ { 2 } }$ for encoded environmental features. In summary, topology-guided bidirectional recurrence provides an ordered feature-transport operator whose effective transformation depth varies with the relative position of limb tokens in the morphology-derived sequence. The resulting contextual representation is subsequently combined with task-specific exteroceptive information, and decoded into the shared action space.

## D Implementation Details in the UNIMAL space

The UNIMAL experiments use a common PPO optimization protocol across methods, with benchmark architecture settings matched to prior generalized morphology controllers where applicable. In this experiment, within each task, all morphologies are optimized through the same shared actor–critic using the PPO configuration in Table VI; no additional morphology-specific optimization procedure is introduced. Table VI summarizes the shared PPO configuration. For methods using KL-based early stopping, the threshold δ is selected from {0.03, 0.05} using the same task-specific search space; the selected values are reported in Table VII. RecMorph-specific architectural settings are listed in Table VIII.

<table><tr><td>Hyperparameter Name</td><td>Hyperparameter Value</td></tr><tr><td>Num of Random Seeds</td><td>4</td></tr><tr><td>Discount γ</td><td>0.99</td></tr><tr><td>GAE Parameter λ</td><td>0.95</td></tr><tr><td>Policy Epochs</td><td>8</td></tr><tr><td>Batch Size Number of Parallel Environments</td><td>5120 32</td></tr><tr><td>Total Timesteps</td><td>1 × 108</td></tr><tr><td>Optimizer</td><td>Adam</td></tr><tr><td>Initial Learning Rate</td><td></td></tr><tr><td>Learning Rate Schedule</td><td>0.0003</td></tr><tr><td>Warmup Iterations</td><td>Linear warmup and cosine decay</td></tr><tr><td></td><td>5</td></tr><tr><td>Gradient Clipping (l2 norm) Value Loss Coefficient</td><td>0.5 0.2</td></tr></table>

TABLE VI  
HYPERPARAMETER SETTINGS FOR PPO IN UNIMAL.

<table><tr><td>Environment</td><td>MetaMorph*</td><td>SWAT</td><td>ModuMorph</td><td>RecMorph</td></tr><tr><td>FT</td><td>0.05</td><td>0.05</td><td>0.05</td><td>0.05</td></tr><tr><td>INCLINE</td><td>0.03</td><td>0.03</td><td>0.05</td><td>0.05</td></tr><tr><td>VT</td><td>0.03</td><td>0.03</td><td>0.03</td><td>0.03</td></tr><tr><td>OBSTACLES</td><td>0.03</td><td>0.03</td><td>0.03</td><td>0.03</td></tr><tr><td>EXPLORATION</td><td>0.03</td><td>0.03</td><td>0.03</td><td>0.03</td></tr></table>

TABLE VII  
OPTIMAL VALUE OF THE EARLY STOPPING THRESHOLD FOR EACH METHOD.

<table><tr><td>Hyperparameter Name</td><td colspan="2">Hyperparameter Value</td></tr><tr><td>Linear Projector layers</td><td rowspan="6">1</td><td>1 128</td></tr><tr><td>Linear hidden dim Backbone layers</td><td>4</td></tr><tr><td>Sequence model hidden dim</td><td>256</td></tr><tr><td>Normalization</td><td>RMSNorm</td></tr><tr><td>Feature modulation</td><td>SiLU channel modulation</td></tr><tr><td>Sequence model activation Decoder layers</td><td></td></tr><tr><td></td><td>tanh</td></tr></table>

TABLE VIII  
HYPERPARAMETER SETTINGS FOR RECMORPH IN UNIMAL.

## E Details of morphological information removal in UNIMAL design space

<table><tr><td>Feature Category</td><td>Full Morphology</td><td>Attribute-Ablated Morphology Observation</td></tr><tr><td>Limb Model (Topology-Dependent)</td><td>body-pos body-ipos body_iquat</td><td></td></tr><tr><td>Limb Hardware (Topology-Independent)</td><td>geom_quat body_mass</td><td>body_mass</td></tr><tr><td>Joint Model (Topology-Dependent)</td><td>body_shape jnt_pos</td><td>body_shape</td></tr><tr><td>Joint Hardware (Topology-Independent)</td><td>joint_range joint_axis gear</td><td>joint_range joint_axis gear</td></tr></table>

TABLE IX  
MORPHOLOGY ATTRIBUTES RETAINED UNDER THE TOPOLOGY-ATTRIBUTE ABLATION.  
Note: Topology-dependent features reflect relative positional and orientational relationships between limbs, while Topology-independent features only describe inherent hardware properties.

We partition the morphology observation into topology-dependent geometric attributes and topology-independent hardware descriptors. The ablation removes the selected position- and orientation-related fields listed in Table IX while retaining mass, body shape, joint range, joint axis, and gear parameters. Architecture-level structural information, including morphology derived token ordering, is unchanged.

## F Feature-Space Analysis of Cross-Limb Contextualization

![](images/92cbbf7cd9428375d15b9dbfef5a7a894abc93b3a63a6210f15c2057b29eafe5.jpg)  
Fig. 11. Feature-space visualization before and after recurrent transport. Limb-identity clusters become less separated after communication, consistent with increased cross-limb contextualization.

We analyze how recursive propagation changes the structure of limb features. We sample robots from the training distribution and run each instance for 500 steps, using the first 200 steps for pre-equilibration and the remaining 300 steps for feature collection. We extract features from two stages: pre-recursion features before the BiRNN and routed features during recursive propagation.

We use the Calinski–Harabasz (CH) index [19] to quantify the separability of limb-identity clusters, complemented by PCA visualizations. A higher CH value indicates that features are more separable by limb identity, while a lower value indicates that features from different limbs become more mixed in the latent space. The CH index decreases from 3747.81 before recurrence to 1436.39 after recurrent transport, indicating reduced separability of limb-identity clusters. PCA shows the same qualitative trend. These observations complement the frozen-encoder probes in Sec. IV-G: communication reduce explicit identity structure while the main-text probes show that the resulting features become more predictive of the policy actions.

I ncl i ne  
![](images/6e0757192d901b93ef1ef6ed1a25b5e462462ae3246606c38ccd90df9445995b.jpg)  
Fig. 12. The performance of RecMorph on five terrains after removing topology-dependent morphology attributes from the UNIMAL space.

We next apply the same attribute-removal intervention to RecMorph while retaining its DFS-derived communication order. Figure 12 shows task-dependent sensitivity to the selected topology-dependent morphology attributes. FT changes little, whereas larger differences appear on the remaining tasks, with the strongest observed effect on Incline. These results show that RecMorph draws morphological information from two complementary sources: explicit morphology attributes embedded in each token and topology-derived structure encoded by the recurrent communication order.

## H Backbone Extensibility with Bidirectional Mamba2

![](images/7638df4ff28ff6404578059dbcb973c96e9dc9b8e26a24bf192242ea9ba8cccf.jpg)  
FT

![](images/3dabfd163c7c25bc3398453d15e7d4395318e969c1e3d8b358280f20dd633dfa.jpg)

![](images/90747071a20d669cf1ee12734eff6b2c8a07ba2c92049613b1c68cf8aba0b598.jpg)  
Exploration

![](images/65280cda5422d945b9dd876d6bada6501b08d814b2048c7cc1af7eaf3ef41a38.jpg)  
VT

![](images/7b97f86e8b524635d9b7b664ecb54b5fbe559cceb6fdb314d057bcfa30ed6483.jpg)  
Obstacle  
Fig. 13. Comparison between BiRNN and BiMamba2 as the underlying recursive model of RecMorph. All variants are configured with 2 layers for a fair comparison under memory constraints.

We further evaluate the backbone extensibility of RecMorph by replacing the BiRNN transition with a bidirectional Mamba2 operator. Both BiRNN and BiMamba2 are configured with two sequence layers to match the computational budget used in this comparison. As shown in Fig. 13, BiMamba2 achieves higher training returns across the five tasks, with particularly clear improvements on Incline and Obstacle. These results show that morphology-guided spatial transport is not tied to a specific recurrent cell and can accommodate alternative sequence operators.

## I Comparison with Graph-Based Message Passing

The linearized NerveNet operator in Appendix B.3 aggregates neighboring representations through a permutation-invariant neighborhood operator followed by shared feature transformations. Repeated message passing therefore expands the receptive field over the kinematic graph while preserving graph-local communication. Each additional message-passing layer aggregates representations that already contain neighborhood mixtures from earlier layers. Receptive-field growth and repeated local mixing therefore increase together with message-passing depth. RecMorph also applies repeated transformations, but these transformations occur along an ordered hidden-state trajectory within a spatial sequence operator rather than through successively deeper neighborhood pooling. The comparison with NerveNet in Fig. 4 shows that this ordered recurrent communication mechanism achieves stronger control performance on the evaluated UNIMAL tasks.

## J Scalability to Larger Morphology Graphs

To further stress-test scalability, we construct a larger UNIMAL-style dataset with up to 30 limbs and an average of 25 limbs, compared with the original dataset whose maximum and average limb counts are 12 and 10, respectively. This setting substantially increases the sequence length and control complexity relative to the original benchmark. Figure 14 shows that all methods achieve lower returns on the larger-limb dataset than on the original benchmark. This degradation should not be attributed solely to long-range information attenuation, because even the single-robot (10M) MLP baseline, which does not rely on sequential message passing, drops from 4671 on the original dataset to 1194 on the larger-limb dataset. This indicates that increasing the number of limbs also increases the intrinsic difficulty of the control problem. Despite the increased control difficulty, RecMorph retains the strongest learning curve among the evaluated shared controllers while also achieving the highest measured throughput. On the larger-body dataset, MetaMorph\*, ModuMorph, and RecMorph(BiRNN) operate at 1224, 918, and 1428 FPS, respectively. The result is consistent with the linear token-complexity motivation of recurrent spatial transport as morphology size increases.

![](images/0fa1ef9368362a0260fd9459f581496d709ee23c90e1f6e7d94889bd028cbbce.jpg)  
Fig. 14. Training results on the newly generated multi-limb dataset on FT terrains.

## K Robustness to Partial Observation Loss

![](images/9a1309e4867668548a2f53b274e87a235b58b325b4656bf6115fb99bb0746047.jpg)  
Fig. 15. FT performance under single-limb observation dropout

To evaluate robustness under realistic sensor failures, we simulate partial observation loss without altering the physical structure of the robot. Specifically, for each robot, we randomly select one limb and set its observation input to zero during execution, while keeping the physical body intact in the environment. This setup reflects real-world scenarios where sensors may malfunction while the limb continues to affect dynamics through mass, inertia, and collisions. Figure 15 shows that RecMorph retains stronger performance than the evaluated baselines under single-limb observation dropout. This behavior is consistent with body-wide contextualization allowing information from other limbs to compensate for missing local observations.

## L Extension to Closed-Loop Kinematic Structures

TABLE X  
HARDWARE DEPLOYMENT STATISTICS. EACH TRIAL LASTS 15 S. SUCCESS DENOTES COMPLETING THE TRIAL WITHOUT FALLING OR TRIGGERING EMERGENCY STOP.
<table><tr><td>Robot</td><td>Setting</td><td>Trials</td><td>Success</td></tr><tr><td>Gol</td><td>backward-right walking</td><td>10</td><td>10/10</td></tr><tr><td>Go2</td><td>forward walking</td><td>10</td><td>10/10</td></tr><tr><td>Go2</td><td>masked feet</td><td>10</td><td>10/10</td></tr><tr><td>Go2</td><td>plastic-wrapped foot</td><td>10</td><td>10/10</td></tr></table>

![](images/9f846542fb7e49db5edcae43a8e0960331e214e6d9915cba1bcd86bdcc224967.jpg)  
Fig. 16. Experiments on a generic controller conducted on an FT landscape using a dataset containing closed-loop data.

We extend RecMorph to closed-loop kinematic graphs through spanning-tree serialization. The benchmark contains 10 newly generated closed-loop morphologies and 40 tree-structured UNIMAL morphologies. For each closed-loop graph, a spanning tree is constructed first and DFS is then applied to obtain the recurrent communication order. RecMorph and the comparison controllers are trained under the same protocol. Figure 16 shows that RecMorph retains effective locomotion performance on this mixed benchmark and outperforms the evaluated baselines. The result demonstrates that spanning-tree serialization provides a practical extension of the RecMorph computation path to the evaluated closed-loop morphologies.

## M Qualitative Locomotion Visualization

We provide dynamic gait visualizations (Figure 17) comparing RecMorph with baseline methods. The visualizations qualitatively show more regular and coordinated locomotion patterns for RecMorph in the illustrated episodes. These examples are intended as qualitative complements to the quantitative benchmark results.

## N Quadruped Training and Deployment Details

a) Task and platforms.: We evaluate shared quadruped control across Unitree Go1, Unitree Go2, ANYmal-B, and ANYmal-C. Each platform has 12 actuated joints and is trained for velocity-tracking locomotion through a common tokenbased policy interface. Robot-specific joint indexing, nominal configurations, low-level gains, and safety constraints remain in the actuation layer.

b) Cross-platform training protocol.: All evaluated methods are trained with 128 environments per robot, 32 rollout steps, and 10,000 PPO iterations. Four independent random seeds are used for RecMorph and every comparison controller. Shared-policy methods jointly optimize one controller across all four platforms, whereas the specialist MLP baseline trains an independent controller for each robot.

c) Friction evaluation.: Cross-platform robustness is evaluated under low, nominal, and high static/dynamic friction coefficients of (0.3, 0.2), (0.8, 0.6), and (1.2, 1.0), respectively. The metrics reported in Tables IV and V are velocity-tracking RMSE, forward progress, body-tilt RMS, and fall rate, with all aggregate statistics computed across the four robot platforms.

d) Observation and action interface.: All platforms use a fixed token-based interface with 64 tokens and 32 features per token. Twelve tokens are active for each quadruped and the remaining tokens are padding. Padded dimensions are masked from action outputs, entropy computation, and value aggregation. Active tokens contain proprioceptive and command information available to the physical controller, including base motion, projected gravity, velocity commands, joint states, previous actions, joint-limit features, and the active mask.

e) Physical deployment.: Physical Go1 and Go2 experiments use a separately trained shared RecMorph checkpoint with policy seed 1409. The policy produces joint-position offsets according to

$$
q _ { \mathrm { t a r g e t } , k } = q _ { \mathrm { d e f a u l t } , k } + 0 . 2 5 a _ { \theta , k } .\tag{104}
$$

The controller runs at 50 Hz. Training includes flat and micro-rough terrain together with corrupted proprioceptive observations to support physical deployment. Robot-specific nominal poses, joint indexing, low-level gains, and safety limits remain unchanged.

## O Hardware Deployment Statistics

We report trial-level deployment statistics for the physical robot experiments in Table X. Each trial lasts 15 s. A trial is considered successful if the robot completes the commanded motion without falling or triggering emergency stop. Across all four evaluated hardware settings, the deployed RecMorph policy completes all 40 trials successfully.

<table><tr><td>NerveNet</td><td>X</td><td>7</td><td>下</td><td></td><td>F</td><td>瓜</td><td>环</td><td></td></tr><tr><td>MetaMorph</td><td>智</td><td>M</td><td></td><td></td><td></td><td>可</td><td></td><td></td></tr><tr><td>MetaMorph*</td><td>竹</td><td></td><td></td><td>之</td><td>公</td><td>冷</td><td></td><td>AT7</td></tr><tr><td>SWAT</td><td>8</td><td></td><td>B</td><td>花</td><td>公</td><td>8</td><td></td><td></td></tr><tr><td></td><td>S</td><td></td><td></td><td>上</td><td></td><td></td><td></td><td>百</td></tr><tr><td>ModuMorph</td><td></td><td></td><td>N</td><td></td><td>J</td><td></td><td></td><td></td></tr><tr><td>RecMorph(BiRNN)</td><td>NT</td><td></td><td>Z</td><td>R</td><td>7</td><td></td><td></td><td>XT</td></tr><tr><td>RecMorph(BiLSTM)</td><td>云</td><td></td><td>2</td><td>LZT</td><td></td><td></td><td></td><td>西</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>心</td></tr><tr><td>RecMorph(BiGRU)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>RecMorph(BiMamba2)</td><td>险</td><td></td><td>小</td><td></td><td>茶</td><td>2</td><td></td><td>2</td></tr></table>

Fig. 17. Dynamic gait graphs for RecMorph and baseline methods. These graphs are temporally continuous, proceeding sequentially from left to right.