# Macro-Action Topological Navigation under Noisy Localization using Reinforcement Learning

Simon Hakenes and Tobias Glasmachers

Institute of Neural Computation, Ruhr University Bochum, Germany simon.hakenes@ini.rub.de, tobias.glasmachers@ini.rub.de

Abstract. Navigating large, photorealistic 3D apartments from raw pixels is widely considered infeasible for plain reinforcement learning. We build an agent that does it anyway, estimating its own pose from the camera alone. The agent has to reach several target objects in sequence, and their positions change between episodes, so it must explore to find them. It builds on our earlier object-centric topological controller, which still read the agent’s true pose and its object detections from the simulator. Here we replace that true pose with an onboard, object-centric estimate. For each object we keep a bank of ORB features that, when the object is seen again, yield a rough pose measurement, which a minimal Extended Kalman Filter (EKF) fuses with a motion model. As on a real robot, the executed motions are noisy. The estimate drifts, but the agent and the nearby objects drift together, so a locally consistent pose is enough to follow each short edge and then home in visually on the target, which lets us replace full SLAM with a much smaller model, closer to how biological navigation appears to work. In the photorealistic Habitat simulator, the agent reaches its target objects from vision alone, with a pose that only needs to be locally consistent.

Keywords: Deep Reinforcement Learning · Navigation · Localization · Topological Maps · Macro Actions

## 1 Introduction

Consider how a person follows the instruction “walk to the kitchen”. They do not know the exact length of the corridor. They only recall a few landmarks and roughly how these are arranged, and this is already enough to pick a sensible route and to keep track of where they roughly are while walking. Localization here is local and approximate. It only has to be good enough to decide where to turn next and to recognize the next landmark. This is the intuition behind topological maps. A topological map represents an environment as landmarks connected by traversable links, and not as a precise metric reconstruction. Such a map does not need to be globally exact, or even globally consistent, to be useful for navigation.

There is evidence that mammals navigate in a similar way. Animal cognitive maps seem to combine a loose topological structure, i.e., which places are connected, with only local metric information, rather than one globally consistent metric frame [1,16]. The study [15] places spatial representations on a range from topological graphs to metric maps, and it points out that very diferent representations can produce the same behavior. We take this as encouragement that a globally wrong but locally consistent map can be enough.

In our previous work [8] we built a navigation agent around this approach. We showed that a simple Deep Q-Network (DQN) [13] agent can navigate large and realistic 3D scenes, which is normally considered infeasible for plain reinforcement learning (RL) from pixels. The reason is that the agent does not act on raw pixels alone. It operates on an object-centric topological map and selects macro actions, where one macro action means navigating to a known object. The map turns a long and sparse-reward navigation episode into a short sequence of elementary decisions, which is what makes the RL problem tractable. In other words, we give the agent a useful inductive prior in the form of structured world knowledge, such as objects and how to detect them, a graph, and the notion of moving through 3D space, instead of learning all of this from reward.

The task we study is sequential multi-object navigation. In each episode the agent must reach several target objects in a fixed order, the same in every episode, which it has to learn. Because the objects are placed diferently each time, it cannot memorize a route and has to find them anew, building its topological map as it explores. The closest benchmark is multi-object navigation (MultiON [23]); unlike MultiON, which reveals the next target class at each step and supplies a GPS-and-compass signal, our agent must learn which object each step denotes and localizes from vision alone.

However, that system quietly assumed perfect self-localization. It read the true position and heading direction of the agent from the simulator and used them for localization, for placing objects on the map, and for executing macro actions. This is convenient in a simulator but unrealistic for a real robot, where the pose has to be estimated from noisy sensors. The classic answer to this problem is Simultaneous Localization and Mapping (SLAM), which builds a globally consistent metric map through probabilistic updates, loop closure, and bundle adjustment. As argued above, however, brains do not seem to keep a globally consistent metric map, and our topological controller does not need one either. If the agent only has to reach the next nearby waypoint, it is enough to know roughly where it is relative to that waypoint. Avoiding global consistency is therefore both cheaper and closer to how biological navigation appears to work.

In this work we remove all ground-truth localization and test this idea directly. We add a small object-centric localizer that estimates the pose from the camera alone. For each object we keep a set of ORB image features [17] together with their estimated 3D positions. When the object is seen again, we match these features and align the matched points, which gives a rough measurement of position and heading. A minimal Extended Kalman Filter (EKF) then fuses this measurement with a simple motion model of the agent. The motion model provides yet another piece of world knowledge. Like every roboticist, we roughly know what the agent does when it moves forward or turns, so we do not have to learn these dynamics from reward, even though the executed motion is noisy and the model is therefore only approximate. The DQN is then left with the question of which object to visit, while the small model-based layer keeps track of where the agent is.

We run no loop closure and no global optimization, and each object is anchored once and then left in place, so the pose estimate drifts over time, but it drifts jointly with the nearby objects. Because a macro action only follows short edges between nearby nodes, this locally consistent pose is enough for the controller. A second reason is visual homing: as soon as the agent sees the target object, it can walk straight to it from the image alone, turning towards it in its visual array and stepping forward. Visual homing is one of the basic navigation modes also found in animals, and it needs no accurate estimate of the global position [7,15,21].

## 2 Related Work

Classical SLAM. Estimating the agent’s pose and a map of the environment at the same time is the subject of SLAM [6]. Feature-based visual systems such as ORB-SLAM [14] track local image features, often ORB [17], and build a globally consistent reconstruction through loop closure and bundle adjustment. This machinery is what makes SLAM accurate, but it is more than our topological controller needs. Bringing this much machinery to bear on a controller that only needs local consistency is like firing cannons at sparrows. In our own trials it also proved fragile in simulation, where repetitive indoor textures and sparse features make tracking and loop closure unreliable. We keep only the cheap front end, ORB features matched to per-object anchors and aligned with the Kabsch algorithm [10,22], and drop the global optimization. That a locally consistent map can be enough for navigation has also been studied under the name of relative or topometric localization [12], and earlier in topological SLAM that deliberately avoids a single global metric frame [5,11].

Localization in learned navigation. In embodied-AI benchmarks such as Habitat [19], self-localization is often sidestepped by giving the agent a GPS-and-compass signal. A common alternative is to estimate the pose from the visual stream, for example with a learned visual odometry module [25], or to learn a SLAM-like component that produces a metric map and a pose for the planner and policy, from RGB and odometry as in Active Neural SLAM [2], or from RGB-D and semantics as in goal-oriented semantic exploration [3]. These methods build a globally registered metric map with learned components. We instead keep a topological map and a small classical localizer, and never optimize it into a globally consistent metric map.

Topological navigation. Topological maps are a natural representation for navigation, because they grow easily and make planning simple. Semi-parametric topological memory [18] stores observed views as graph nodes and learns a reachability network to localize against them without an explicit metric pose, and Neural Topological SLAM [4] learns to build and reason over a topological graph for image-goal navigation. These systems learn their localization and graph reasoning, whereas we handle localization with a small classical stack and use the DQN only for the high-level choice of which object to visit.

![](images/0fa35340b5175400e39025392a1a3da654fabeb7d48ab1b91660d4889e253aa1.jpg)  
Fig. 1. Overview of the system. The decision layer (DQN over object-centric macro actions, blue) is inherited from our previous work [8]. The navigation and localization layer turns the selected object into motion and replaces the previous ground-truth pose oracle with the new localization stack (orange). An ORB landmark bank, a pose measurement, and an EKF with a motion model produce a pose estimate. It feeds the controller, and it is used to estimate the positions of detected objects, which are written to the topological map. Close to the target the controller switches to visual homing.

Macro actions. Selecting a high-level target that unfolds into a long sequence of elementary actions is a form of temporal abstraction, formalized by the options framework [20]. Unlike learned options, our macro actions are not trained from reward but built from the map and a small controller, so the RL problem reduces to choosing useful objects, scored by one shared network over the growing action set following the DRRN [9]. This design is from our previous system [8], and we ask whether it still holds once the true pose becomes a noisy, locally consistent estimate.

## 3 Method

Before going into detail, we give a high-level overview of the complete system and clarify which parts are inherited from our previous work [8] and which are new in this paper.

The method is organized into two layers (Fig. 1). The first one is the decision layer, which is the object-centric macro-action DQN from our previous work [8]. It takes the current task progress and the objects stored in the map, and decides which known object should be visited next. The second one is the navigation and localization layer. It turns that decision into actual motion, plans a shortest path over the topological map with a standard shortest-path algorithm, and drives the agent there with a small set of elementary actions (moving forward and turning). This split between the two layers is the design we keep from our previous work. Sparse-reward deep RL should not have to rediscover 3D geometry, graph search, or the efect of a forward step from raw pixels. We make these parts explicit, and we leave the DQN with the semantic problem of choosing useful objects to interact with.

What is new is the localization stack that lets this layer work without the true pose. In our previous system this layer could read the true simulator pose at every step, so localization and map construction were trivial. Here we replace this oracle with a localizer that estimates the pose from the agent’s own sensors. For each object we keep a bank of ORB landmarks, we turn re-encountered landmarks into a rough pose measurement by aligning them, and an EKF with a motion model fuses these measurements with the known geometry of the elementary actions. The motion model alone is not enough, because a forward step can be blocked by an obstacle, the agent can slide along a wall, or the actuation itself is noisy, so the predicted motion and the real motion drift apart. Because the pose is now an estimate and not ground-truth information, we also add a map-write gate that refuses to write new map entries when the pose estimate is too unreliable.

## 3.1 Macro actions and the state-action space

The agent builds and uses an object-centric topological map, both in our previous system and here. A macro action means navigating to a specific object that is already stored in this map, which turns a long and sparse-reward navigation episode into a shorter sequence of semantic choices. Each macro action runs until the agent performs a Found action, which declares that it has reached the chosen object and ends the macro.

The state and action spaces are not fixed, because the map grows while the episode runs. Let $\mathcal { O } _ { t } \subset \mathcal { V } _ { t }$ be the object nodes that are known at time t. The macro-action space is the set of these objects, $A _ { t } = \mathcal { O } _ { t }$

For an object node $v \in { \mathcal { O } } _ { t }$ , the map stores a set of image patches

$$
\mathcal { F } _ { v } = \{ f _ { v , 1 } , \hdots , f _ { v , n _ { v } } \} , \qquad f _ { v , i } \in \mathbb { R } ^ { H \times W \times 3 } .\tag{1}
$$

The task progress is encoded by a one-hot vector $g _ { t } \in \{ 0 , 1 \} ^ { N _ { T } }$ , where $N _ { T }$ is the number of targets in the sequence. The active entry indicates which step in the fixed sequence is currently active, and the agent has to learn which object that step refers to.

We keep this representation from the previous paper, because it solves the main dificulty of using a growing map with a DQN. A standard DQN has one output neuron per action [13], which assumes a fixed set of actions. Here the number of objects is not known in advance, so the network cannot output all action values at once. Instead, inspired by the DRRN [9], it evaluates one object at a time, $Q _ { \theta } ( \mathcal { F } _ { v } , g _ { t } ) \in \mathbb { R }$ , where the object is an input to the network. The same network is applied to every known object and produces one value per candidate macro action.

## 3.2 DQN architecture and policy

For each object we use up to $N _ { i }$ image patches as the action representation for robustness. All patches pass through the same convolutional feature extractor. We then combine the resulting feature vector with the progress vector through an outer product. A fully connected layer maps the combined features to a single Q-value.

At decision time the network computes one Q-value for each known object. The policy is greedy most of the time, and it explores with a Boltzmann policy [24] that samples an object from a softmax over the Q-values. Training follows the same object-wise evaluation as in [8].

## 3.3 Map representation

The map at time t is an undirected weighted graph $\mathcal { G } _ { t } = ( \nu _ { t } , \mathcal { E } _ { t } )$ . The node set $\mathcal { V } _ { t } = B _ { t } \cup \mathcal { O } _ { t }$ contains backbone nodes $B _ { t }$ and object nodes ${ \mathcal { O } } _ { t }$ . A backbone node is simply a pose that the agent has already visited. It has no visual appearance and only serves as a waypoint for navigation. An object node represents a semantic object and stores an object id, an estimated 3D position, and a set of image patches. Only object nodes are ofered to the DQN as macro actions. The backbone nodes are internal navigation scafolding.

This distinction matters. An object node is a good semantic landmark but a poor waypoint, because an object’s position can lie inside furniture, behind an occluder, or be visible but not reachable. A backbone node, in contrast, is a pose that the agent has actually occupied, so it is usually safe to navigate to. When the DQN selects an object, the controller therefore first routes to a nearby backbone node and only switches to visual homing once the object comes into view.

It is important to keep in mind that the map is not given in advance. At the start of an episode the map is empty, and the agent has usually not even seen the target objects yet, because they lie elsewhere in an apartment that typically spans several rooms. The agent first has to explore the environment and build the map, and it already navigates on this partial map while the map is still growing. After each elementary action, the current estimated pose is added as a backbone node, and nearby backbone nodes are connected.

Object nodes are added from the RGB-D image and the semantic masks. As in the previous paper, the masks and object ids come from the simulator. Object detection is a separate problem and not the topic of this paper, so we do not address it here. For every visible object, we store the RGB pixels inside its bounding box as a patch, and we back-project the depth at the object to get a 3D position in the map frame. If the object is new, a new object node is created. If it is already known, we may add a better (bigger) patch, but we do not move the stored object position afterwards. The map is therefore not globally consistent. One could object that a single global coordinate frame is a metric map through the back door. However, nothing in the controller depends on absolute coordinates. Because we never enforce global consistency, the absolute coordinates drift, but what the agent actually relies on is the relative, node-to-node geometry, which is why the global pose can drift by meters without breaking navigation.

## 3.4 ORB object landmarks

The semantic masks tell us which object is visible, but they do not give stable point correspondences over time, because a mask only labels which pixels belong to an object and does not identify the same physical point from one frame to the next. For localization we therefore keep a small bank of ORB features [17] for each object. We use ORB for practical reasons. The features are cheap compared with learned descriptors, they can be matched quickly, and they need no training.

For every visible object, ORB keypoints are detected only inside its semantic mask, so each feature belongs to exactly one object. When a feature is seen for the first time, we store its descriptor together with a 3D anchor, which we obtain by back-projecting its depth into the map frame. When the same object is seen again, we match the current descriptors against the object’s bank. Each successful match links a freshly observed point to its stored anchor on the map. These matched pairs are the input to the pose measurement described next.

Only re-found anchors are used for localization. Newly detected keypoints can extend the bank of an object for later, but they are not used as localization correspondences in the frame in which they first appear. This avoids a circular update in which a newly seen object both defines and confirms the current pose.

## 3.5 Kabsch pose measurement

The localizer turns the matched ORB correspondences into a single noisy pose measurement. Each matched feature gives two points on the ground plane. One is where the agent currently observes it, obtained by back-projecting its depth. The other is where its anchor is stored on the map. Up to the unknown pose of the agent, these two point sets should be connected by a rigid transformation. Since the agent moves on the ground plane, this motion is a 2D translation together with a single rotation around the vertical axis, i.e., an SE(2) change in position and yaw. Given at least three matches, we recover this motion with the Kabsch algorithm [10,22].

Formally, let $( p _ { i } , q _ { i } )$ with $i = 1 , \dots , M$ be the matched pairs on the ground plane, where $p _ { i }$ is the back-projected observation in the agent frame and $q _ { i }$ is the stored anchor in map coordinates. Each pair carries an inverse-distance weight $w _ { i }$ with $\textstyle \sum _ { i } w _ { i } = 1$ , so that far and therefore noisier anchors contribute less. With the weighted centroids $\begin{array} { r } { \bar { p } = \sum _ { i } w _ { i } p _ { i } } \end{array}$ and $\begin{array} { r } { \bar { q } = \sum _ { i } w _ { i } q _ { i } } \end{array}$ , the rotation follows from the SVD of the weighted cross-covariance

$$
C = \sum _ { i } w _ { i } ( p _ { i } - \bar { p } ) ( q _ { i } - \bar { q } ) ^ { \top } = U S V ^ { \top }\tag{2}
$$

as

$$
R = V \mathrm { d i a g } \big ( 1 , \mathrm { d e t } ( V U ^ { \top } ) \big ) U ^ { \top } , \qquad { \bf t } = \bar { q } - R \bar { p } .\tag{3}
$$

The determinant factor forces a proper rotation rather than a reflection, since the agent can turn but not mirror itself. The agent sits at the origin of its own frame, so t is directly the measured position, and the rotation angle of R is the measured yaw. Together they form one measurement of position and heading that is handed to the EKF.

The measurement only has to pull the predicted pose back towards the already mapped object anchors from time to time. The alignment also returns the residuals $r _ { i } = \| R p _ { i } + \mathbf { t } - q _ { i } \|$ , which tell how well the matches agree, and we keep their distribution as a measure of how much we can trust this measurement.

## 3.6 Extended Kalman filter

The EKF fuses the geometry of the elementary actions with the Kabsch measurements into a single pose estimate that is updated at every step. The motion model behind its prediction step is the piece of world knowledge mentioned in the introduction.

The pose estimate is represented by the $\operatorname { S E } ( 2 )$ state

$$
\mathbf { s } _ { t } = \left[ x _ { t } \ : z _ { t } \ : \psi _ { t } \right] ^ { \top } ,\tag{4}
$$

where $( x , z )$ are ground-plane coordinates and $\psi$ is yaw. The prediction step is tied to the commanded elementary action. For a forward step of length d,

$$
x _ { t + 1 } = x _ { t } - d \sin \psi _ { t } , \qquad z _ { t + 1 } = z _ { t } - d \cos \psi _ { t } , \qquad \psi _ { t + 1 } = \psi _ { t } .\tag{5}
$$

For elementary TurnLeft and TurnRight actions of the low-level controller, the position stays the same and the yaw changes by the known turn angle.

We deliberately add actuation noise, so that the motion model is no longer exact, especially in the yaw. The executed motion deviates from the commanded amounts by zero-mean Gaussian noise in three components, with standard deviations of $1 ^ { \circ }$ on the turn angle, 0.01 m on the forward step, and 0.005 m of sideways slip, as on a real robot. The EKF still predicts the commanded motion, so the believed pose drifts away from the true pose like real odometry.

This gives the agent a pose estimate even in frames where no object can be matched, for example when it stands close to a blank wall and sees no useful features.

The Kabsch output enters as a direct measurement whose assumed noise grows as the matches agree less, and measurements that are clearly inconsistent with the current belief are rejected. The filter therefore coasts on the motion model through unreliable visual frames and accepts a visual correction only when it fits the current estimate.

## 3.7 Map-write gating

A noisy pose creates a failure mode that did not exist when the true pose was available: if the pose is wrong, an object can be written to the map at a wrong position, and this wrong landmark then misleads all later planning and navigation. We therefore only write to the map when the current pose looks trustworthy, judged from simple signals the localizer already provides, such as how many object features were matched and how certain the EKF currently is. If the pose is not good enough, we set the observation aside and add the object later, since we would rather add an object late than in the wrong place.

## 3.8 Macro-action execution under noisy localization

To execute a macro action, the controller turns the chosen object into a navigation target. It then plans the shortest path through the graph and follows it one waypoint at a time, turning towards the next waypoint and moving forward. The controller is deliberately simple, though it still needs fallback behaviors to stay reliable. Its job is not to solve navigation in arbitrary geometry, but to use the fact that the graph splits a long route into short and tractable steps.

Close to the target object, the controller switches to visual homing. The agent reads the position of the target in the input image from the semantic mask and the depth. If the target is left or right of the image center, the agent turns towards it. If it is centered, the agent moves forward. If it is close enough, the controller signals Found, which ends the macro action. A fallback also ends the macro if the agent stalls without reaching the object, so it cannot run forever. Visual homing is reliable here, because close to the object the camera image tells the agent directly where to go, and it does not need an accurate estimate of its global position. This is the main reason why a drifting global pose does not hurt. The drift matters least exactly where the agent has to be accurate, namely right at the target. Visual homing relies on the live camera image rather than the believed pose, so it ties the agent to the real object however far the global estimate has drifted, and seeing that object again lets the localizer pull the estimate back towards its anchor.

## 4 Experiments and Results

We evaluate in the photorealistic Habitat simulator [19] on the same multi-room object-finding task as in our previous work [8]. The agent starts in an unknown apartment with an empty map and must reach a fixed sequence of target objects in the correct order. Because it does not know where the targets are, it has to explore, build the topological map online, and revisit objects it has already seen, and an episode counts as a success only when every target in the sequence has been reached. We use two reward schemes: under immediate reward the agent is rewarded each time it reaches the next target in the order, whereas under terminal reward it is rewarded only once, when the whole sequence is complete.

Terminal reward is the harder credit-assignment problem, because a long chain of correct macro actions is reinforced only at the very end, and with a single target the two schemes coincide. As before, object masks and ids are provided by the simulator, so the new element under test is purely the localization stack, not object detection. The central question is whether the macro-action navigation that previously relied on ground-truth pose still works when that pose is replaced by our noisy, locally consistent estimate.

We report the success rate and the number of elementary actions per episode, and Table 1 additionally lists the localization error as per-episode median RMSE against the ground-truth pose. Each condition and task type is trained with five independent seeds.

To see what each part contributes, we compare five conditions. Full is the complete system with the ORB, Kabsch, and EKF localizer. No-EKF is an ablation that uses the raw Kabsch measurement directly, without the motion model and the filter. Motion-Only is the complementary ablation that keeps the EKF motion model but drops the Kabsch measurement, so the pose comes from dead reckoning alone. GT-Pose is an oracle that restores the true pose and, unlike the other four conditions, runs without actuation noise, reproducing our previous system [8]. Random-Macro keeps the full localizer but picks the macro actions at random instead of using the trained DQN.

Table 1 summarizes the five conditions, Fig. 2 shows success rates and navigation cost per condition, and Fig. 3 shows the behavior of the localizer over one episode. We first look at task success under the conventional criterion, where an episode counts as a success whenever the agent declares a target Found within a small radius of it. By this measure the five conditions stay close together. Even Random-Macro, which picks its macro actions without any trained policy, reaches 84%, 62%, and 46% of the targets at one, two, and three targets. This is surprising, because the localization errors in Table 1 difer widely across the conditions, and one would expect a broken pose to hurt navigation.

The Found declarations explain the discrepancy. A wrong Found carries no penalty, so an agent that never gets to its target can give up after searching for a while and declare Found anyway, and this declaration sometimes lands inside the success radius by chance. The conventional rate counts these give-ups together with the deliberate arrivals. We therefore also report the genuine success rate, which counts only the episodes in which the agent actually reached the target and stopped at it. Under this measure the conditions separate, most clearly in the single-target and immediate-reward cells where every target is rewarded. The Full system loses little at a single target, from 93% to 85%, and although the gap grows with the sequence length, its declared successes remain largely genuine arrivals rather than give-ups. Random-Macro loses far more at every length, from 84% to 54%, from 62% to 28%, and from 46% to 25%. The gap is starkest for Random-Macro: at a single target its 84% conventional rate falls to 54% genuine, the give-ups making up the diference. The gap between the two rates is the share of successes won by chance, and it is largest for the random baseline. Under terminal reward, where only the completed sequence is rewarded, the genuine rates are low and close across the noisy conditions. Three targets under terminal reward is the hardest cell, since the whole sequence must be completed before any reward arrives; even here the Full system completes most sequences and keeps the highest genuine success of the noisy conditions.

The localization error separates the conditions more sharply still. Full keeps its heading error between 13 and 39 degrees and its position drift to a few meters across all task types. No-EKF lets the heading error grow to between 80 and 104 degrees, so its heading estimate is essentially unusable. Motion-Only drifts in position to between 3 and 4 meters. We return to both ablations in the discussion. The elementary-action counts must be read together with the success rate, since they are taken over successful episodes only. A condition that fails often, e.g., Motion-Only at three targets, can still look cheap, because the few episodes it wins are the short, easy ones. GT-Pose, the oracle, reaches every target and keeps the highest genuine success of all, so a noisy pose does cost some performance, but among the conditions that must work from a noisy pose the Full system is the strongest.

![](images/4c5ada428cb7d081b179cff3d9fc9824dba13f34bef013664cbcf7f5110ec040.jpg)  
Fig. 2. Success rate (top, with 95% confidence intervals) and elem. actions per successful episode (bottom, note the logarithmic scale) for the five conditions and the five task types. The success bars are two-toned: the light bar is the conventional success rate, and the solid inner bar is the genuine success rate, so the light remainder is the share won by give-ups, i.e., a Found, declared after the agent stalls without reaching the target, that happens to land inside the success radius. Failed episodes are excluded from the action counts because they end at the step budget, so their length reflects the budget and not the navigation.

Table 1. Comparison of the five conditions per task type, pooled over five training seeds. n is the number of evaluation episodes. Pos. and Yaw RMSE are per-episode medians taken over all episodes. Lower step counts and errors are better.
<table><tr><td>Condition</td><td>Success</td><td>Genuine succ.</td><td>Elem. actions [median]</td><td>Pos. RMSE [m]</td><td>Yaw RMSE [deg]</td><td>n</td></tr><tr><td>1 Target</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GT-Pose (oracle)</td><td>100.0%</td><td>96.9%</td><td>530</td><td>0.00</td><td>0.0</td><td>426</td></tr><tr><td>Full (ours)</td><td>92.7%</td><td>85.4%</td><td>646</td><td>1.46</td><td>13.2</td><td>314</td></tr><tr><td>No-EKF</td><td>90.3%</td><td>84.0%</td><td>737</td><td>3.80</td><td>79.7</td><td>300</td></tr><tr><td>Motion-Only</td><td>82.6%</td><td>73.4%</td><td>1257</td><td>2.67</td><td>18.4</td><td>184</td></tr><tr><td>Random-Macro</td><td>83.5%</td><td>54.1%</td><td>2643</td><td>2.01</td><td>27.6</td><td>170</td></tr><tr><td colspan="7">2 Targets, immediate reward</td></tr><tr><td>GT-Pose (oracle) 100.0%</td><td></td><td>89.2%</td><td>2411</td><td>0.00</td><td>0.0</td><td>158</td></tr><tr><td>Full (ours)</td><td>74.8%</td><td>48.7%</td><td>2818</td><td>2.08</td><td>26.5</td><td>119</td></tr><tr><td>No-EKF</td><td>70.9%</td><td>37.6%</td><td>2804</td><td>5.75</td><td>101.6</td><td>117</td></tr><tr><td>Motion-Only</td><td>69.2%</td><td>35.5%</td><td>4121</td><td>3.71</td><td>33.4</td><td>107</td></tr><tr><td>Random-Macro</td><td>61.5%</td><td>28.1%</td><td>4796</td><td>2.11</td><td>40.0</td><td>96</td></tr><tr><td colspan="7">2 Targets, terminal reward</td></tr><tr><td>GT-Pose (oracle)</td><td>100.0%</td><td>76.9%</td><td>3207</td><td>0.00</td><td>0.0</td><td>130</td></tr><tr><td>Full (ours)</td><td>61.3%</td><td>31.2%</td><td>3912</td><td>2.20</td><td>37.7</td><td>93</td></tr><tr><td>No-EKF</td><td>68.0%</td><td>36.0%</td><td>3744</td><td>5.61</td><td>97.5</td><td>100</td></tr><tr><td>Motion-Only</td><td>65.6%</td><td>31.1%</td><td>4503</td><td>3.69</td><td>35.0</td><td>90</td></tr><tr><td>Random-Macro</td><td>66.7%</td><td>36.8%</td><td>5866</td><td>2.19</td><td>40.2</td><td>87</td></tr><tr><td colspan="7">3 Targets, immediate reward</td></tr><tr><td>GT-Pose (oracle)</td><td>100.0%</td><td>79.8%</td><td>3125</td><td>0.00</td><td>0.0</td><td>109</td></tr><tr><td>Full (ours)</td><td>57.3%</td><td>27.0%</td><td>3811</td><td>2.26</td><td>36.0</td><td>89</td></tr><tr><td>No-EKF</td><td>61.0%</td><td>29.3%</td><td>4445</td><td>6.29</td><td>104.2</td><td>82</td></tr><tr><td>Motion-Only</td><td>39.2%</td><td>12.2%</td><td>4058</td><td>3.95</td><td>43.3</td><td>74</td></tr><tr><td>Random-Macro</td><td>46.4%</td><td>24.6%</td><td>5260</td><td>2.22</td><td>44.8</td><td>69</td></tr><tr><td colspan="7">3 Targets, terminal reward</td></tr><tr><td>GT-Pose (oracle)</td><td>100.0%</td><td>77.0%</td><td>3911</td><td>0.00</td><td>0.0</td><td>87</td></tr><tr><td>Full (ours)</td><td>59.2%</td><td>30.3%</td><td>4693</td><td>2.36</td><td>39.2</td><td>76</td></tr><tr><td>No-EKF</td><td>52.1%</td><td>22.5%</td><td>5402</td><td>6.50</td><td>98.7</td><td>71</td></tr><tr><td>Motion-Only</td><td>50.0%</td><td>14.7%</td><td>4028</td><td>4.39</td><td>34.7</td><td>68</td></tr><tr><td>Random-Macro</td><td>42.2%</td><td>14.1%</td><td>6393</td><td>2.46</td><td>48.6</td><td>64</td></tr></table>

## 5 Discussion and Conclusion

Our guiding question was whether macro-action topological navigation still works once the global pose is noisy, i.e., whether a pose that is only locally consistent is suficient. The experiments support a positive answer, with one qualification. The Full system reaches its targets even though its global position estimate drifts by several meters, and on the stricter genuine measure it leads the conditions that work from a noisy pose. It does not match the ground-truth oracle, and the gap widens as the sequence grows longer. We therefore do not claim that the noise is harmless, only that local consistency is enough for the controller to keep working.

![](images/fb31122f9de0e348c0afec4dc14f42b1e591b487aa321b22b94db1495eac0f14.jpg)  
(a) onboard view

![](images/d61e0494f5c7b68984e7805736d28d61ffa137077c319a691ba8bde3b8c4d65a.jpg)  
(b) map + path

![](images/f916e0d3ba4f0949a0411f57007b056a64fce2531f7910ddd20cb1f86a5dc780.jpg)  
(c) position drift

![](images/5639ddb24df475bdead20bf5dfc58b0c383568a0c36b247f297018bcdb650d35.jpg)  
elem. actions  
(d) heading error  
Fig. 3. A single successful episode of the Full system, with the onboard view (a), a bird’s-eye map showing the ground-truth (solid) and estimated (dashed) paths (b), and the position (c) and yaw (d) error over the episode. The estimated path in (b) drifts visibly away from the ground truth, so the global pose really is inconsistent. Panels (c) and (d) show the position error growing to a few meters and the heading error to roughly fifteen to thirty degrees over the episode, yet the controller still reaches each target, because the agent and the nearby objects drift together, so only their relative geometry matters.

The No-EKF ablation shows which part of the localizer matters. Without the motion model the heading estimate drifts to between 80 and 104 degrees and navigation breaks down. Hence it is the explicit motion model, and not the raw Kabsch measurement, that makes the pose usable. The low action counts of No-EKF are the selection efect from the results, since the episodes it wins are the easy ones and its failures show up in the success rate instead.

The Motion-Only ablation is the complementary cut, and it is where the actuation noise matters most. It keeps the motion model but drops the Kabsch measurement, so the pose is pure dead reckoning. Because the executed motions are noisy, as they would be on a real robot, this drift is never corrected, and the position error grows to several meters while the heading error reaches 18 to 43 degrees. In our earlier noise-free setup this condition looked almost perfect, which was an artifact of the simulator applying every commanded move exactly. With realistic actuation noise that artifact is gone, and Motion-Only now shows the unbounded drift that the Kabsch measurement is there to correct. The two ablations together separate the two jobs of the localizer. The motion model keeps the heading usable between measurements, and the measurement keeps the position from drifting away.

A natural question is whether the system needs the depth image at all. Depth enters in only two places: it back-projects the ORB matches and the detected objects into 3D anchors, and it tells visual homing when the target is close enough. Neither use needs a dense or globally accurate depth map. Because we only ever require local consistency, the metric scale that depth provides could in principle come from cheaper sources, such as triangulation of the ORB anchors across the known elementary motions, a learned monocular depth estimate, or even the apparent size of an object’s mask. The same tolerance to drift that lets us drop global SLAM should make the localizer forgiving of such a monocular substitute, which we see as a promising route to removing the depth sensor.

Three limitations stand out. First, computing ORB features inside every object mask is the most expensive part of the localizer, so the localization stack is comparatively costly, and cheaper or learned features are an obvious target for future work. Second, we still rely on the simulator for the object masks and ids, so we do not yet deal with real object detection. Adding real object detection, together with the noise it would inject into the per-object feature banks, is an important step towards a fully realistic system. Third, the macro actions are carried out by a hand-built controller that follows waypoints, homes in visually, and falls back to recovery behaviors when the agent stalls. Much of the navigation reliability, and most of the step count, rests on this controller rather than on the learned policy, which is also why even the Random-Macro baseline reaches its targets fairly often. Making such a controller robust across diverse and cluttered apartments takes a good deal of manual engineering, so learning more of this low-level execution is an appealing direction for future work.

In sum, removing the ground-truth pose oracle and navigating from a globally inaccurate but locally consistent estimate works well, though not yet as reliably as with the true pose. The cost of the ORB features and the assumption of perfect object detection are the main steps left before the system can run outside the simulator.

## References

1. Alvernhe, A., Sargolini, F., Poucet, B.: Rats build and update topological representations through exploration. Anim. Cogn. 15(3), 359–368 (May 2012).

2. Chaplot, D.S., Gandhi, D., Gupta, S., Gupta, A., Salakhutdinov, R.: Learning to Explore using Active Neural SLAM. In: International Conference on Learning Representations (ICLR) (2020)

3. Chaplot, D.S., Gandhi, D.P., Gupta, A., Salakhutdinov, R.R.: Object goal navigation using goal-oriented semantic exploration. Adv. Neural Inf. Process. Syst. 33, 4247– 4258 (2020)

4. Chaplot, D.S., Salakhutdinov, R., Gupta, A., Gupta, S.: Neural topological SLAM for visual navigation. In: 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 12875–12884. IEEE (Jun 2020).

5. Choset, H., Nagatani, K.: Topological simultaneous localization and mapping (SLAM): Toward exact localization without explicit localization. IEEE Trans. Rob. Autom. 17(2), 125–137 (Apr 2001).

6. Dissanayake, M.W.M.G., Newman, P., Clark, S., Durrant-Whyte, H.F., Csorba, M.: A solution to the simultaneous localization and map building (SLAM) problem. IEEE Trans. Rob. Autom. 17(3), 229–241 (Jun 2001).

7. Franz, M.O., Mallot, H.A.: Biomimetic robot navigation. Robot. Auton. Syst. 30(1–2), 133–153 (Jan 2000).

8. Hakenes, S., Glasmachers, T.: Deep reinforcement learning based navigation with macro actions and topological maps. In: Nicosia, G., Ojha, V., Giesselbach, S., Pardalos, P.M., Umeton, R., La Malfa, E., La Malfa, G. (eds.) Machine Learning, Optimization, and Data Science. pp. 95–108. Springer Nature Switzerland, Cham (2026).

9. He, J., Chen, J., He, X., Gao, J., Li, L., Deng, L., Ostendorf, M.: Deep Reinforcement Learning with a Natural Language Action Space. In: Proceedings of the 54th Annual

Meeting of the Association for Computational Linguistics (ACL). pp. 1621–1630 (2016).

10. Kabsch, W.: A solution for the best rotation to relate two sets of vectors. Acta Crystallographica Section A 32(5), 922–923 (1976).

11. Kuipers, B., Byun, Y.T.: A robot exploration and mapping strategy based on a semantic hierarchy of spatial representations. Rob. Auton. Syst. 8(1-2), 47–63 (Nov 1991).

12. Mazuran, M., Boniardi, F., Burgard, W., Tipaldi, G.D.: Relative Topometric Localization in Globally Inconsistent Maps. In: Robotics Research (ISRR 2015), pp. 435–451. Springer Proceedings in Advanced Robotics, Springer (2018).

13. Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A.A., Veness, J., Bellemare, M.G., Graves, A., Riedmiller, M., Fidjeland, A.K., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., Hassabis, D.: Human-level control through deep reinforcement learning. Nature 518(7540), 529–533 (Feb 2015).

14. Mur-Artal, R., Montiel, J.M.M., Tardós, J.D.: ORB-SLAM: A Versatile and Accurate Monocular SLAM System. IEEE Trans. Rob. 31(5), 1147–1163 (Oct 2015).

15. Parra-Barrero, E., Vijayabaskaran, S., Seabrook, E., Wiskott, L., Cheng, S.: A map of spatial navigation for neuroscience. Neurosci. Biobehav. Rev. 152, 105200 (Sep 2023).

16. Poucet, B.: Spatial cognitive maps in animals: new hypotheses on their structure and neural mechanisms. Psychol. Rev. 100(2), 163–182 (Apr 1993).

17. Rublee, E., Rabaud, V., Konolige, K., Bradski, G.: ORB: An eficient alternative to SIFT or SURF. In: 2011 International Conference on Computer Vision (ICCV). pp. 2564–2571 (2011).

18. Savinov, N., Dosovitskiy, A., Koltun, V.: Semi-parametric Topological Memory for Navigation. 6th International Conference on Learning Representations, ICLR 2018 - Conference Track Proceedings (Mar 2018)

19. Savva, M., Kadian, A., Maksymets, O., Zhao, Y., Wijmans, E., Jain, B., Straub, J., Liu, J., Koltun, V., Malik, J., Parikh, D., Batra, D.: Habitat: A platform for embodied AI research. In: 2019 IEEE/CVF International Conference on Computer Vision (ICCV). pp. 9338–9346. IEEE (Oct 2019).

20. Sutton, R.S., Precup, D., Singh, S.: Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning. Artif. Intell. 112(1–2), 181–211 (Aug 1999).

21. Trullier, O., Wiener, S.I., Berthoz, A., Meyer, J.A.: Biologically based artificial navigation systems: Review and prospects. Prog. Neurobiol. 51(5), 483–544 (Apr 1997).

22. Umeyama, S.: Least-squares estimation of transformation parameters between two point patterns. IEEE Transactions on Pattern Analysis and Machine Intelligence 13(4), 376–380 (1991).

23. Wani, S., Patel, S., Jain, U., Chang, A., Savva, M.: MultiON: Benchmarking Semantic Map Memory using Multi-Object Navigation. In: Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M.F., Lin, H. (eds.) Advances in Neural Information Processing Systems. vol. 33, pp. 9700–9712. Curran Associates, Inc. (2020)

24. Wiering, M.: Explorations in eficient reinforcement learning. Ph.D. thesis, University of Amsterdam (1999)

25. Zhao, X., Agrawal, H., Batra, D., Schwing, A.G.: The Surprising Efectiveness of Visual Odometry Techniques for Embodied PointGoal Navigation. In: 2021 IEEE/CVF International Conference on Computer Vision (ICCV). pp. 16107–16116 (2021).