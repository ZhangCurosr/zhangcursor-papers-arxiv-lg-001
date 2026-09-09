# Curriculum Learning as Transport: Understanding Curricula with Wasserstein Geodesics

Changho Shin\* Princeton University Princeton, NJ, USA cs1095@princeton.edu

David Alvarez-Melis   
Microsoft Research & Harvard University   
Cambridge, MA, USA   
daalvare@microsoft.com

## Abstract

Curriculum learning is governed by several coupled design choices—how difficulty is defined, how examples are ordered, how much exposure each level receives, and how quickly training moves across levels—making it hard to isolate what actually helps. We present Wasserstein curriculum paths, a simple transport-based framework that decouples these factors by representing curricula as trajectories of training distributions over discrete difficulty levels. Across a calibrated synthetic suite with 12 tasks and 33 difficulty axes, we use this framework to isolate the effects of ordering, matched exposure, endpoint smoothness, and pacing under fixed training budgets. We find that curriculum effects are strongly context-dependent: no single strategy dominates across tasks, difficulty axes, and budgets, and curricula mainly change where a fixed budget is spent most effectively. Within this framework, easy-to-hard ordering improves hard-level performance relative to exposure-matched static sampling, showing that the benefit is not explained by cumulative exposure alone. We further show that endpoint smoothness and pacing substantially affect where along the difficulty spectrum a curriculum is effective. Finally, we show that the same transport view naturally supports extensions to learned pacing through geometry and to structured difficulty spaces beyond one-imensional orderings.

## 1 Introduction

Curriculum learning aims to improve training by presenting examples in a structured order, often from easier to harder instances (Bengio et al., 2009). But in practice, curriculum effects depend on several coupled design choices: how difficulty is defined, whether training progresses easy-to-hard or hard-to-easy, how much total exposure each level receives, and how quickly the curriculum moves across levels (Hacohen & Weinshall, 2019; Wu et al., 2021; ia et al., 2026). As a result, even when a curriculum helps, it is often unclear why: are the gains due to ordering itself, greater exposure to easier examples, smoother transitions between levels, or a better pacing of a fixed training budget?

Prior curriculum methods—including self-paced selection, teacher-student schedulers, bandit approaches, and competence-based pacing—often improve performance (Kumar et al., 2010; Matiisen et al., 2020; Graves et al., 2017; Platanios et al., 2019), but they usually change several design choices at once, making it hard to isolate their contributions. Recent studies also ask more directly when ordering helps (Wu et al., 2021; Jia et al., 2026), but they do not ultimately provide a common framework for separating ordering from matched exposure, endpoint choice, pacing, and geometry within one controlled setup.

We propose Wasserstein interpolation as a unified controlled framework for studying curriculum design. By viewing a curriculum as a path of training distributions over ordered difficulty levels, Wasserstein paths provide a smooth way to move mass through nearby levels between easy-heavy and hard-heavy training distributions (McCann, 1997; Peyré &

Cuturi, 2019). This lets us vary one design choice at a time while holding the others fixed, enabling controlled comparisons of ordering, matched exposure, endpoint smoothness, pacing, and geometry within a single parameterization.

Using a calibrated synthetic suite spanning 12 tasks and 33 difficulty axes, we use this framework to isolate which components of curriculum design matter and under what conditions. Three main findings emerge:

Curriculum effects are context dependent. Across the synthetic suite, no single curriculum dominates across tasks, difficulty axes, and budgets. Instead, curricula mainly shift where a fixed training budget is used most efficiently across difficulty levels, with smooth easy-tohard progression particularly efficient on the hardest levels.

Ordering matters beyond cumulative exposure. Easy-to-hard progression improves hardlevel performance relative to an exposure-matched static baseline, whereas hard-to-easy ordering often hurts, indicating that timing matters, not just cumulative exposure.

Pacing and endpoint design matter. Even within an easy-to-hard curriculum, the same path can help or hurt depending on how sharply the endpoint distributions are concentrated and how quickly the path is traversed. Very sharp endpoints are brittle under either overly aggressive or overly conservative pacing.

Beyond these main findings, we showcase how the framework can adapt pacing through the underlying geometry and extend curriculum paths from one-dimensional orderings to structured' difficulty spaces.

Taken together, these results establish Wasserstein curricula as a principled framework for studying curriculum components, understanding when they matter, and extending curriculum design to adaptive pacing and structured difficulty spaces.

## 2 Related Work

We review the most relevant prior work here and provide further discussion in Appendix A.

Curriculum learning and difficulty-aware training. Curriculum learning studies whether organizing training examples by difficulty can improve optimization and generalization (Bengio et al., 2009). Automated variants choose what to show and when through self-paced, teacher-student, bandit, and competence-based schedules (Kumar et al., 2010; Matiisen et al. 2020; Graves et al., 2017; Platanios et al., 2019), while training dynamics and related signals have been used to estimate example difficulty (Hacohen & Weinshall, 2019; Swayamdipta et al., 2020; Shrivastava et al., 2016; Toneva et al., 2019). Wang et al. (2022) organize this literature around difficulty measurement and training scheduling, and Meng et al. (2026) combine these components in a unified dynamic framework. Recent work also asks more directly when ordering helps: Wu et al. (2021) find clearer gains under limited budgets or label noise, while Jia et al. (2026) show in LLM math post-training that the preferred direction depends on model capability, task complexity, and the difficulty metric. Our work studies a different question: given a difficulty structure, which aspects of the resulting curriculum matter? We represent curricula as paths over distributions on discrete difficulty levels and separate the difficulty geometry, endpoints, direction, cumulative exposure, and pacing. This enables controlled comparisons of factors that are often varied together, while also supporting extensions to adaptive geometry and structured difficulty spaces.

Optimal Transport Optimal transport (OT) provides a geometry for comparing probability distributions based on the cost of transporting mass between them (Villani, 2008). In this geometry, Wasserstein interpolation defines geodesic paths that smoothly transform one distribution into another (McCann, 1997; Peyré & Cuturi, 2019; Santambrogio, 2015) Prior OT-based curriculum work, especially in reinforcement learning, uses these ideas to construct or sequence tasks by generating intermediate training stages between task or environment distributions (Huang et al., 2022; Klink et al., 2022). In contrast, we use Wasserstein interpolation as an analytical framework over distributions on a fixed difficulty axis, allowing us to cleanly separate ordering, exposure, pacing, and endpoint effects within a single parameterization. This perspective focuses less on designing new curricula and more on isolating which components of curriculum design drive observed gains.

![](images/4011cff24cd2547bcdf959d2559369a53457d61651dc7b2c20e890b57be5afb6.jpg)

![](images/7a19a10559646a6ee8e871b83a40f5962f1f1675116797750a033e68c3ab5e28.jpg)

(a) Linear vs. Wasserstein interpolation.  
![](images/ea343e48672e48a27ea5517a30dd03b48159e4ec7f958d16b07a6f8834d4320d.jpg)  
(b) Curriculum heatmap.  
Figure 1: Comparison of linear and Wasserstein curricula. (a) Wasserstein interpolation moves probability mass progressively through the level geometry, whereas linear interpolation directly mixes the endpoint distributions. (b) Viewing Pt as the batch-level sampling distribution over time shows that Wasserstein places substantially more mass on intermediate levels during the transition

## 3 Preliminaries and Experimental Setup

We first introduce Wasserstein-geodesic curricula, then summarize the common protocol that turns them into a controlled study of curriculum design.

## 3.1 Wasserstein-Geodesic Curriculum

For each task, we partition the training data into L ordered difficulty levels. At training step k, each batch is sampled from a probability vector $P ^ { k } \in \Delta ^ { L - 1 }$ , where $P _ { \ell } ^ { k }$ is the probability of drawing from level l. A curriculum is therefore a sequence of sampling distributions over these levels. We represent it as a continuous path $\left\{ P _ { t } \right\} _ { t \in [ 0 , 1 ] }$ between endpoint distributions $P _ { 0 } , P _ { 1 } \in \Delta ^ { L - 1 }$ , together with a schedule $t _ { k } \in [ 0 , 1 ]$ that maps training step k to a position on the path. The path specifies which mixtures of difficulty levels are visited, and the schedule specifies how quickly training moves through them, with $P ^ { k } = P _ { t _ { k } }$

Wasserstein interpolation. We construct the path on the default one-dimensional ordering of levels: for $t \in [ \bar { 0 } , 1 ]$

$$
P _ { t } = \arg \operatorname* { m i n } _ { Q \in \Delta ^ { L - 1 } } \Big \{ ( 1 - t ) W ( Q , P _ { 0 } ) + t W ( Q , P _ { 1 } ) \Big \} ,
$$

where W is the Wasserstein distance on that fixed level geometry. This moves probability mass smoothly through adjacent levels between the two endpoint distributions.

For comparison, the linear interpolation baseline directly mixes the same endpoints,

$$
P _ { t } ^ { \mathrm { l i n } } = ( 1 - t ) P _ { 0 } + t P _ { 1 } .
$$

As illustrated in Figure 1, linear interpolation mixes the endpoint distributions directly, whereas the Wasserstein path progresses through intermediate levels, producing a more natural curriculum.

Curriculum with Wasserstein interpolation. By setting $P ^ { k } = P _ { t _ { k } }$ , we obtain curricula that move mass through nearby levels. The choice of endpoints $( P _ { 0 } , P _ { 1 } )$ determines where the curriculum starts and ends, the geometry determines which levels are treated as nearby, and the schedule $\{ t _ { k } \}$ controls how quickly training moves along the path. This makes Wasserstein curricula useful as a study lens: once the task axis is fixed, they expose several curriculum factors that can be varied cleanly, including endpoint choice, pacing, and geometry, while preserving a natural progression over the same levels. Later sections use this structure to study endpoint families and traversal rates, learned geometry in WARP, and structured difficulty spaces beyond one-dimensional orderings.

## 3.2 Common Experimental Setup

We next introduce the common setup for our experimental suite. Unless varied explicitly in later sections, we keep the task construction, models, sampling protocol, and budget calibration fixed so that the suite can be used to study ordering, matched exposure, pacing, and geometry under shared conditions. We use this calibrated synthetic suite for the main study because it supports controlled comparisons; in Appendix D, we show that in real-data SFT settings with pretrained models and noisier difficulty labels, curriculum effects are much weaker and harder to disentangle.

Tasks and difficulty control. Unless otherwise stated, we study a synthetic suite of 12 tasks, adapted in part from SynLogic (Liu et al., 2025a). For each task, we construct ordered difficulty levels by varying one difficulty axis at a time, that is, one task attribute used to order examples from easier to harder while keeping the task semantics fixed. Calibrating and validating these within-task axes is part of the contribution, since it lets us compare curricula on controlled progressions rather than heterogeneous mixtures of examples. In the main study, this yields 33 task-by-difficulty-axis conditions, because several tasks contribute multiple difficulty axes. Appendix B.1 gives the full task list, examples, and exact level constructions.

Models. We use decoder-only Transformers (Vaswani et al., 2017) throughout, with model size matched to task complexity. Most tasks use small models trained from scratch, while language-like tasks fine-tune SmolLM-135M (Ben Allal et al., 2024). Model details are reported in Appendix B.2.

Training protocol. At training step $k ,$ mini-batches are sampled from a distribution $P ^ { k } \in$ $\Delta ^ { L - 1 }$ over difficulty levels. We use static i.i.d. for the non-curriculum baseline, where batches are drawn independently from the same fixed distribution throughout training; by default, that distribution is uniform over difficulty levels. Unless varied explicitly, our default easy-to-hard runs use an easy-heavy start distribution $P _ { 0 }$ and a hard-heavy end distribution $P _ { 1 } ,$ where $( P _ { 1 } ) _ { \ell } \propto \exp ( \ell / \tau )$ and $P _ { 0 } ^ { \mathrm { ~ \cdot ~ } }$ is the symmetric reversal of $P _ { 1 } , \mathrm { i . e . , } ( P _ { 0 } ) _ { \ell } = ( P _ { 1 } ) _ { L - \ell + 1 }$ We use default temperature $\tau = 1$ and default schedule $t _ { k } = k / T$

Training budgets. Training budget is a key variable in curriculum learning (Wu et al. 2021). Rather than fixing one global step count, we calibrate small, medium, and large regimes separately for each benchmark from pilot static i.i.d. learning curves, so that the static i.i.d. baseline spans early, intermediate, and later stages of learning on that task. This keeps budget comparisons aligned across tasks instead of comparing one task at an early stage of learning and another at a much later stage. Exact budget values are listed in Appendix B.1, with the associated model and training details in Appendix B.2.

## 4 When and How Does Curriculum Learning Help?

We first study whether curriculum learning helps reliably or only in specific regimes. For this, we compare static i.i.d., linear, and Wasserstein curricula across the 33 synthetic-suite task-by-difficulty-axis conditions in the main study, and report both overall and hardestlevel accuracy.

## 4.1 When Curricula Help Is Highly Context Dependent

Setup. For each task-by-difficulty-axis condition at a given budget, we first average accuracy over 10 random-seed repetitions, then aggregate those condition-level means within each budget block. Each block contains 33 conditions.

Results. Table 1 reports the resulting summary separately for the small, medium, and large regimes. No single curriculum dominates the suite. All three methods win substantial subsets of conditions, and the ranking shifts with budget rather than settling on a universal

<table><tr><td>Curriculum</td><td>Mean overall (%)</td><td>Mean hardest (%)</td><td>Overall wins</td><td>Hardest wins</td></tr><tr><td>Small budget</td><td></td><td></td><td></td><td></td></tr><tr><td>Static i.i.d.</td><td>56.6</td><td>39.9</td><td>13</td><td>6</td></tr><tr><td>Linear</td><td>55.4</td><td>46.4</td><td>7</td><td>12</td></tr><tr><td>Wasserstein</td><td>56.6</td><td>46.3</td><td>13</td><td>15</td></tr><tr><td>Medium budget</td><td></td><td></td><td></td><td></td></tr><tr><td>Static i.i.d.</td><td>79.2</td><td>66.1</td><td>11</td><td>5</td></tr><tr><td>Linear</td><td>78.8</td><td>72.6</td><td>10</td><td>16</td></tr><tr><td>Wasserstein</td><td>79.7</td><td>72.1</td><td>12</td><td>12</td></tr><tr><td>Large budget</td><td></td><td></td><td></td><td></td></tr><tr><td>Static i.i.d.</td><td>92.7</td><td>86.0</td><td>6</td><td>1</td></tr><tr><td>Linear</td><td>92.9</td><td>89.6</td><td>12</td><td>13</td></tr><tr><td>Wasserstein</td><td>93.2</td><td>89.7</td><td>16</td><td>19</td></tr></table>

Table 1: Absolute summary for the three basic curricula, stratified by training budget. Each block contains 33 task-by-difficulty-axis conditions. The win columns denote the number of conditions in which a curriculum performs best under the corresponding accuracy metric. "Hardest" denotes the last available level for each benchmark.  
![](images/8d76d4fa7b4ce71b9306c403d3fea37d94f5ce729787d4bffd4b2c4c0cfc6ee4.jpg)

![](images/4f972d9d78a745d7ec1302425564efab8a9a5041761685c8f936cf9985f84935.jpg)

![](images/86726d37ade7b03f46d93c0e224bb2729d660257252da9efcaf5e62c60afbc59.jpg)  
Figure 2: Exposure, accuracy, and exposure-adjusted accuracy under the medium budget. Left: cumulative exposure share over easiest, middle, and hardest probe buckets. Midăle: final bucket accuracy on the same buckets. Right: exposure-adjusted accuracy on the same buckets. The middle bucket is the single middle level when the number of levels is odd and the merged middle pair when it is even, so these three probe buckets are not an exhaustive partition for five-level tasks. All panels compare only static i.i.d., linear, and Wasserstein. The corresponding profiles across all three budgets are shown in Figure 21 in Appendix C.2.

best choice. The main lesson is therefore contextual rather than universal: whether curriculum helps depends jointly on task structure, the difficulty axis, and the available budget. Appendix C.1 gives the corresponding task-wise final-accuracy breakdowns.

Finding 1. Curriculum effectiveness depends strongly on task, difficulty axis, and budget; no single strategy dominates.

## 4.2 Curricula Shift the Data-Efficiency Profile Across Difficulty

Similar overall accuracy means can hide very different learning profiles. We therefore compare curricula not only by final accuracy, but also by where they are data-efficient across difficulty levels.

Setup. Using the same setup as Section 4.1, Figure 2 summarizes medium-budget behavior on the easiest, middle, and hardest probe buckets. Exposure is the cumulative share of training mass assigned to each bucket over the full run. As a simple proxy for data efficiency, we use exposure-adjusted accuracy: final bucket accuracy divided by cumulative exposure to that bucket.

![](images/9a5513f8617a177e7a6951595cf659012e886e62080aecfac62ceb80251b763e.jpg)  
Figure 3: Ordering within the Wasserstein comparison. Each panel shows final accuracy on the easy, middle, and hard buckets for one training budget, corresponding to the first, middle, and hardest levels in the shared summary used here. "Static Matching" denotes the exposure-matched static baseline. Easy-to-hard Wasserstein consistently yields a flatter profile and stronger hard-bucket performance than either the static baseline or hard-to-easy Wasserstein.

Results. Figure 2 shows that with a fixed budget, the main difference between curricula is not a uniform gain in final accuracy, but a different data-efficiency profile across difficulty. Static i.i.d. remains strongest in final accuracy on the easiest bucket. Linear is strongest in exposure-adjusted accuracy in the middle bucket, while Wasserstein, which follows a natural smooth progression over the ordered levels, has the highest exposure-adjusted accuracy on the hardest bucket. This pattern is not driven by a few outliers: on the hard bucket, Wasserstein has higher exposure-adjusted accuracy than linear in all 99 task-byaxis-and-budget conditions. Figure 21 in Appendix C.2 shows that the same qualitative pattern persists across the smalí, medium, and large budgets. The main effect of curricu-Īum is therefore to reallocate where a fixed budget pays off, with Wasserstein especially efficient on the hardest bucket, which may matter when hard examples are scarce or costly Appendix C.2 gives the corresponding budget-wise profile view, Appendix C.3 shows the task-wise final level profiles, and Appendix C.6 revisits the hard-bucket advantage through controlled bridge-effect experiments.

Finding 2. With a fixed budget, curricula reshape the data-efficiency profile across difficulty;   
Wasserstein is especially efficient on the hardest bucket.

## 5 Does Ordering Matter?

Section 4.2 showed that curricula with the same budget can induce different exposure and accuracy profiles across difficulty. We now ask whether ordering itself matters beyond cumulative exposure, comparing the default easy-to-hard Wasserstein path, the exposurematched static baseline, and hard-to-easy Wasserstein on the same suite.

Setup. Using the same default setup as in the previous sections, we compare easy-to-hard Wasserstein, the default ordering used throughout the paper, with two controls that keep cumulative exposure fixed while changing ordering. The exposure-matched static baseline, shown as “Static Matching" in Figure 3, preserves cumulative exposure to each level but removes the temporal progression, so it uses the same fixed batch-sampling distribution throughout training. Hard-to-easy Wasserstein traverses the same path family in reverse and therefore preserves the same cumulative exposure while reversing the direction of progression.

![](images/edf168db6ec15f73df4fa86c8dc972fa97f9053bd1b91cb52dda79c6be6756d5.jpg)  
(a) Endpoints vs. temperature τ.

![](images/b1818afca664cfe3207ac9841f695cebbfb02a5aa571b2fd312240a09eaa457e.jpg)  
(b) Batch-level path for γ.

![](images/1308657ecab5d5924b21ad83b8a0bed7baf2d54eb79c1b2545975b238cb229f6.jpg)  
(c) Accuracy across (τ, γ)  
Figure 4: (a) Lower τ sharpens the endpoints; larger τ smooths them. (b) γ changes how fast training moves along the same easy-to-hard path. (c) Medium-budget sweep over 20 task-by-difficulty-axis conditions and three seeds. A broad region with moderate pacing and smoother endpoints performs best; the red dashed circles highlight collapse under very sharp endpoints with either overly aggressive or overly conservative pacing.

Results. Figure 3 shows that neither control recovers the easy-to-hard profile. The exposure-matched static baseline retains more strength on the easy bucket, but it misses the hard-bucket gains of easy-to-hard Wasserstein even when their overall averages remain similar. Hard-to-easy Wasserstein is more damaging still and often harms performance outright: it shifts strength toward the easy end and away from the hard bucket, where curricula matter most. This effect is broad rather than isolated to a few tasks: on the hardest level, forward Wasserstein exceeds reverse Wasserstein in 92 of 99 task-by-axis-and-budget conditions. The benefit of easy-to-hard progression therefore does not come from cumulative exposure alone; it comes from when examples are seen during training. Appendix C.5 studies these directional transfer patterns more directly with level-to-level transfer experiments.

Finding 3. Ordering matters beyond cumulative exposure: exposure-matched static sampling misses the hard-bucket gains of eăsy-to-hard progression, and hárd-to-easy ordering often harms.

## 6 Pacing and Smoothness Matter

Even within an easy-to-hard curriculum, do pacing and endpoint smoothness still matter? We vary endpoint smoothness (τ) and traversal speed (γ) along the same Wasserstein path.

Setup. We use seven tasks with short enough runtimes to make the full grid sweep feasible, giving 20 task-by-difficulty-axis conditions. Throughout this subsection we keep the same easy-to-hard Wasserstein path

$$
P _ { t } = \arg \operatorname* { m i n } _ { Q \in \Delta ^ { L - 1 } } \Big \{ ( 1 - t ) W ( Q , P _ { 0 } ) + t W ( Q , P _ { 1 } ) \Big \} ,
$$

and vary only its endpoint smoothness and pacing. The endpoint family and traversal are

$$
\begin{array} { r } { ( P _ { 1 } ) _ { \ell } \propto \exp ( \ell / \tau ) , \qquad ( P _ { 0 } ) _ { \ell } = ( P _ { 1 } ) _ { L - \ell + 1 } , \qquad t _ { k } = ( k / T ) ^ { \gamma } . } \end{array}
$$

Here l indexes difficulty levels, L is the number of levels, k is the training step, and T is the total number of training steps. Smaller τ makes the endpoints sharper and larger τ smooths them, while $\gamma$ controls how quickly training moves along the path: smaller values move training toward harder levels sooner, while larger values keep training longer on easier and intermediate levels. We sweep ${ \mathfrak { a } } ~ 9 \times 9$ grid over $\tau , \gamma \in$ {0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 5.0, 10.0} in the medium-budget regime only, with three seeds per condition.

Results. Figure 4(c) reveals a broad good region rather than a single best setting. Moderate pacing and smoother endpoints work well, while very sharp endpoints are brittle under both overly aggressive and overly conservative pacing. The default setting already lies in a strong region, but it is not uniquely optimal. Easy-to-hard ordering alone is therefore not enough: the same path can help or hurt depending on smoothness and pacing.

![](images/69e18855faca369863d48f25a896b81283ce348594be7ffbbe9566868e9fa045.jpg)

![](images/0a6ba46c11b4cd14e9c4286c87435d0559650e6d500e6f079396a03b843a8277.jpg)

![](images/6f2f4f339af3d7730ebaee98be3a2492382c2e199cf38c8fd27c6f34a5cf1dba.jpg)  
Figure 5: Geometry controls pacing even when endpoints stay fixed. Left: standard and an affinely rescaled learned geometry on the same level axis. Right: the resulting easy-to-hard Wasserstein trajectories. Štretching the early-to-mid region and compressing the hard end makes the path dwell longer before a sharper final handoff.

Finding 4. Pacing and endpoint smoothness matter: very sharp endpoints can fail under either overly aggressive or overly conservative pacing.

## 7 Adaptive Geometry as Learnable Pacing

If pacing matters, the next question is how to adapt it without leaving the Wasserstein framework. We do so indirectly through geometry: because Wasserstein paths depend on the underlying distances between levels, changing that geometry changes pacing while preserving the same easy-to-hard endpoints. As a prototype, we instantiate this idea as WARP (Wasserstein-Adaptive Reshaping of Paths), which learns a geometry once after a short warmup and then follows the same forward Wasserstein path on the learned locations. Appendix B.4 gives the full definition; here we only highlight the learned edge-length rule.

Setup. After a short uniform warmup, WARP probes gradient alignment between neighboring levels and converts adjacent cosine couplings into relative edge lengths, so higher similarity shortens an edge and lower similarity lengthens it.

Results. Table 2 shows that even this first prototype beats fixed Wasserstein at small and medium budgets, while matching it at large budget. This proof-of-point result shows that adapting the geometry can improve pacing along the same easy-to-hard path.

Table 2: Mean overall accuracy (%) by budget.
<table><tr><td>Method</td><td>Small</td><td>Medium</td><td>Large</td></tr><tr><td>Wasserstein</td><td>56.8</td><td>79.8</td><td>92.9</td></tr><tr><td>WARP</td><td>57.1</td><td>80.6</td><td>92.9</td></tr><tr><td>Gain</td><td>+0.2</td><td>+0.9</td><td>+0.0</td></tr></table>

## 8 Structured Difficulty Spaces Beyond One-Dimensional Orderings

So far, we have treated difficulty as a one-dimensional ordering. Here we ask whether the same framework still works when levels are connected by a richer structure. To illustrate this, we construct two structured difficulty spaces from the arithmetic-operator family, a cube and a tree, and compare static i.i.d., Wasserstein, and WARP.

Setup. We use the 2-layer transformer and compare static i.i.d., Wasserstein, and WARP over 10 seeds on the cube and tree benchmarks shown in Figure 6. In both benchmarks, each node is a level in the same local problem family, but the connectivity differs. Appendix B.5 gives the exact graph constructions and endpoint distributions.

![](images/902cf428164c19f9e544ea79b004eeecb5aee4f7eee3efebdb9edc5e598f83a4.jpg)  
Figure 6: Structured benchmarks, a cube and a tree, built from arithmetic operators. Nodes are labeled by level, and nearby callouts show example expressions.

<table><tr><td colspan="5">Cube</td></tr><tr><td>Budget</td><td>Nodes</td><td>Static i.i.d.</td><td>Wasserstein</td><td>WARP</td></tr><tr><td>Small</td><td>All</td><td>61.4</td><td>62.2</td><td>59.3</td></tr><tr><td rowspan="7">Medium</td><td>Init</td><td>82.9</td><td>77.3</td><td>70.7</td></tr><tr><td>Mid</td><td>60.8</td><td>61.3</td><td>59.2</td></tr><tr><td>End</td><td>43.7</td><td>53.0</td><td>48.4</td></tr><tr><td>All</td><td>77.6</td><td>75.1</td><td>75.6</td></tr><tr><td>Init</td><td>95.0</td><td>87.6</td><td>91.4</td></tr><tr><td>Mid</td><td>77.5</td><td>74.5</td><td>75.7</td></tr><tr><td>End</td><td>60.5</td><td>66.3</td><td>59.1</td></tr><tr><td rowspan="4">Large</td><td>All</td><td>93.3</td><td>93.2</td><td>90.5</td></tr><tr><td>Init</td><td>99.4</td><td>98.4</td><td>96.6</td></tr><tr><td>Mid</td><td>93.6</td><td>93.1</td><td>90.5</td></tr><tr><td>End</td><td>85.9</td><td>89.0</td><td>84.0</td></tr></table>

<table><tr><td colspan="5">Tree</td></tr><tr><td>Budget</td><td>Nodes</td><td>Static i.i.d.</td><td>Wasserstein</td><td>WARP</td></tr><tr><td>Small</td><td>All</td><td>62.9</td><td>67.7</td><td>71.2</td></tr><tr><td rowspan="7">Medium</td><td>Init</td><td>72.7</td><td>71.3</td><td>73.0</td></tr><tr><td>Mid</td><td>55.3</td><td>63.6</td><td>65.9</td></tr><tr><td>End</td><td>64.3</td><td>68.9</td><td>73.4</td></tr><tr><td>All</td><td>78.9</td><td>77.2</td><td>76.1</td></tr><tr><td>Init</td><td>85.8</td><td>80.7</td><td>86.0</td></tr><tr><td>Mid</td><td>73.3</td><td>75.0</td><td>72.5</td></tr><tr><td>End</td><td>80.0</td><td>77.3</td><td>75.4</td></tr><tr><td rowspan="5">Large</td><td>All</td><td>86.7</td><td>90.3</td><td>90.0</td></tr><tr><td>Init</td><td>95.3</td><td>94.5</td><td>95.4</td></tr><tr><td>Mid</td><td>83.1</td><td>88.2</td><td>86.2</td></tr><tr><td>End</td><td>86.4</td><td>90.4</td><td>90.5</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

Table 3: Accuracy (%, ten-seed mean) on the structured arithmetic-operator benchmarks. INIT/MID/END denote the root, internal nodes, and leaves for the tree, and L1, L2–L7, and L8 for the cube.

Results. Table 3 shows that geometry-aware curricula remain useful beyond onedimensional orderings, with the clearest gains near the hard end. On the cube, Wasserstein consistently improves the hard endpoint L8, while Static i.i.d. remains strongest on the easier regions and on aggregate at medium and large budgets. On the tree, both geometry-aware curricula are competitive: WARP is strongest in the small-budget regime, especially away from the root, whereas Wasserstein becomes strongest overall in the large-budget regime. These experiments show that the same path construction extends to graph-structured difficulty spaces, with effects that vary across graph, region, and training budget.

## 9 Conclusion

We use Wasserstein interpolation as a controlled lens for studying curriculum learning. This lens lets us vary calibrated difficulty axes, ordering, matched cumulative exposure, endpoint smoothness, pacing, and geometry within one framework. Taken together, our results do not support a single universally best curriculum. Instead, they show that different components of curriculum design matter in different ways: easy-to-hard ordering helps beyond cumulative exposure, pacing and endpoint design change where a fixed budget is most effective, adaptive geometry can yield gains, and structured difficulty spaces make those gains more localized and more dependent on structure. At the same time, our realdata SFT results suggest that these effects weaken substantially in settings with pretrained models, natural benchmarks, and noisier difficulty labels.

Limitations and Future Work. We intentionally constrain most of the study to from-scratch training and newly constructed tasks, reducing confounding from pretrained knowledge but leaving open how the results transfer to pretrained settings, natural data, and tasks whose difficulty structure interacts with prior knowledge. We also showcase two extensions of the framework, adaptive geometry and structured difficulty spaces, pointing to next directions such as learning geometry or pacing from transfer signaÎs, defining Wasserstein curricula on richer structures, and testing them in pretrained settings.

## Disclosure of LLM Use

We used LLM-based assistants during this project for parts of task design, code implementation, exploratory analysis, plotting and analysis code, and drafting or revising portions of the manuscript. These tools were used interactively under close author supervision. The authors made the final research decisions, ran and verified the reported experiments and analyses, checked the resulting figures and tables, and reviewed and edited all final paper text.

## Acknowledgements

The majority of this work was done while CS and DAM were at Microsoft Research. CS thanks Brenden Lake for supporting conference travel. DAM additionally acknowledges support from the Kempner Institute for the Study of Natural and Artificial Intelligence, the Aramont Fellowship Fund and NSF Award No. 2229881, NSF AI Institute for Societal Decision Making (NSF AI-SDM).

## References

Shun-ichi Amari. Information Geometry and Its Applications, volume 194 of Applied Mathematical Sciences. Springer Japan, 2016. doi: 10.1007/978-4-431-55978-8. URL https: //doi.org/10.1007/978-4-431-55978-8.

Lior Belenki, Alekh Agarwal, Tianze Shi, and Kristina Toutanova. Optimizing pre-training data mixtures with mixtures of data expert models. In Proceedings of the 3rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 32570– 32587. Association for Computational Linguistics, 2025. doi: 10.18653/v1/2025.acl-long. 1564. URL https://aclanthology.org/2025.acl-long.1564/.

Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Leandro von Werra, and Thomas Wolf. Smollm - blazingly fast and remarkably powerful. Hugging Face blog, 2024. URL https://huggingface.co/blog/smollm.

Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. Curriculum learning. In International Conference on Machine Learning, pp. 41–48. ACM, 2009. doi: 10.1145/ 1553374.1553380.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try ARC, the AI2 reasoning challenge. arXiv.org, 2018. doi: 10.48550/arXiv.1803.05457. URL https://arxiv. org/abs/1803.05457.

Simin Fan, Matteo Pagliardini, and Martin Jaggi. DOGE: Domain reweighting with generalization estimation. In International Conference on Machine Learning, pp. 12895–12915. PMLR, 2024. doi: 10.48550/arXiv.2310.15393. URL https://proceedings.mlr.press/ v235/fan24e.html.

Albert Ge, Tzu-Heng Huang, John Cooper, Avi Trost, Ziyi Chu, Satya Sai Srinath Namburi Gnvv, Ziyang Cai, Kendall Park, Nicholas Roberts, and Frederic Sala. R&b: Domain regrouping and data mixture balancing for efficient foundation model training. arXiv.org, 2025. doi: 10.48550/arXiv.2505.00358. URL https://arxiv.org/abs/2505.00358.

Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies. Transactions of the Association for Computational Linguistics, 9:346–361, 2021. doi: 10.1162/tacl\_a\_00370. URL https://acianthology.org/2021.tacl-1.21/.

Alex Graves, Marc G. Bellemare, Jacob Menick, Rémi Munos, and Koray Kavukcuoglu. Automated curriculum learning for neural networks. In International Conference on Machine Learning, pp. 1311–1320, 2017. URL https://proceedings.mlr.press/v70/graves17a. html.

Guy Hacohen and Daphna Weinshall. On the power of curriculum learning in training deep networks. In International Conference on Machine Learning, pp. 2535–2544, 2019. URL https://proceedings.mlr.press/v97/hacohen19a.html.

Peter Hase, Mohit Bansal, Peter Clark, and Sarah Wiegreffe. The unreasonable effectiveness of easy training data for hard tasks. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 7002–7024, Bangkok, Thailand, 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.378. URL https://aclanthology.org/2024.ac1-1ong.378/.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id= d7KBjmI3GmQ.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview. net/ forum?id=nZeVKeeFYf9.

Peide Huang, Mengdi Xu, Jiacheng Zhu, Laixi Shi, Fei Fang, and Ding Zhao. Curriculum reinforcement learning using optimal transport via gradual domain adaptation. In Advances in Neural Information Processing Systems 35, pp. 10656–10670. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2022. doi: 10.52202/068431-0774. URL https://arxiv.org/abs/2210.10195.

Yaning Jia, Chunhui Zhang, Xingjian Diao, Xiangchi Yuan, Z. Ouyang, and S. Vosoughi. What makes a good curriculum? disentangling the effects of data ordering on llm mathematical reasoning. Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 34472–34488, 2026. doi: 10.18653/v1/2026.acl-long.1591. URL https://arxiv.org/abs/2510.19099.

Pascal Klink, Haoyi Yang, Carlo D'Eramo, Jan Peters, and Joni Pajarinen. Curriculum reinforcement learning via constrained optimal transport. In International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp. 11341–11358. PMLR,2022. URL https://proceedings.mlr.press/v162/klink22a.html.

M. Pawan Kumar, Benjamin Packer, and Daphne Koller. Self-paced learning for latent variable models. In Neural Information Processing Systems, 2010. URL https: //proceedings. neurips.cc/paper/2010/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html.

Junteng Liu, Yuanxiang Fan, Zhuo Jiang, Han Ding, Yong Hu, Chi Zhang, Yiqi Shi, Shitong Weng, Aili Chen, Shiqi Chen, et al. Synlogic: Synthesizing verifiable reasoning data at scale for learning logical reasoning and beyond. In Advances in Neural Information Processing Systems 38, pp. 111997–112018. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2025a. doi: 10.52202/085713-3380. URL https://openreview.net/forum?id= XtNiw80Qsy. Poster.

Qian Liu, Xiaosen Zheng, Niklas Muennighoff, Guangtao Zeng, Longxu Dou, Tianyu Pang, Jing Jiang, and Min Lin. Regmix: Data mixture as regression for language model pretraining. In International Conference on Learning Representations, 2025b. doi: 10.48550/ arXiv.2407.01492. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ 5f67d864aae6115374fed7beddd119e0-Abstract-Conference.html.

Tambet Matiisen, Avital Oliver, Taco Cohen, and John Schulman. Teacher-student curriculum learning. IEEE Transactions on Neural Networks and Learning Systems, 31(9):3732–3740, 2020. doi: 10.1109/TNNLS.2019.2934906. URL https://doi.org/10.1109/TNNLS.2019. 2934906.

Robert J. McCann. A convexity principle for interacting gases. Advances in Mathematics, 128 (1):153–179, 1997. doi: 10.1006/aima.1997.1634.

Guangyu Meng, Qinkai Zeng, John P. Lalor, and Hong Yu. A psychology-based unified dynamic framework for curriculum learning. Computational Linguistics, pp. 1–49, 2026. doi: 10.1162/COLI.a.584. URL https://direct.mit.edu/coli/article/doi/10.1162/COLI.a. 584/134522/A-Psychology-based-Unified-Dynamic-Framework-for.

Jiahui Peng, Xinlin Zhuang, Jiantao Qiu, Ren Ma, Jing Yu, He Zhu, and Conghui He. Topic over source: The key to effective data mixing for language model pre-training. arXiv, 2025. doi: 10.48550/arXiv.2502.16802. URL https://arxiv.org/abs/2502.16802.

G. Peyré and Marco Cuturi. Computational optimal transport. Found. Trends Mach. Learn., 11(5-6):355–607, 2019. doi: 10.1561/2200000073.

Emmanouil Antonios Platanios, Otilia Stretcu, Graham Neubig, B. Póczos, and Tom Michael Mitchell. Competence-based curriculum learning for neural machine translation. In North American Chapter of the Association for Computational Linguistics, pp. 1162–1172. Association for Computational Linguistics, 2019. doi: 10.18653/v1/N19-1119. URL https://aclanthology.org/N19-1119/.

Filippo Santambrogio. Optimal Transport for Applied Mathematicians: Calculus of Variations, PDEs, and Modeling. Birkhäuser Cham, 2015. doi: 10.1007/978-3-319-20828-2.

Abhinav Shrivastava, Abhinav Gupta, and Ross Girshick. Training region-based object detectors with online hard example mining. In Computer Vision and Pattern Recognition, pp. 761–769. IEEE, 2016. doi: 10.1109/CVPR.2016.89.

Swabha Swayamdipta, Roy Schwartz, Nicholas Lourie, Yizhong Wang, Hannaneh Hajishirzi, Noah A. Smith, ând Yejin Choi. Dataset cartography: Mapping and diagnosing datasets with training dynamics. In Conference on Empirical Methods in Natural Language Processing, pp. 9275–9293. Association for Computational Linguistics, 2020. doi: 10.18653/v1/2020. emnlp-main.746. URL https://aclanthology.org/2020.emnlp-main.746/.

Mariya Toneva, Alessandro Sordoni, Rémi Tachet des Combes, Adam Trischler, Yoshua Bengio, and Geoffrey J. Gordon. An empirical study of example forgetting during deep neural network learning. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=BJlxm30cKm.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, pp. 5998–6008. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper\_files/paper/2017/ hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html.

Cédric Villani. Optimal transport, Old and New, volume 338. Springer Science & Business Media, 2008. ISBN 9783540710493. doi: 10.1007/978-3-540-71050-9.

Xin Wang, Yudong Chen, and Wenwu Zhu. A survey on curriculum learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(9):4555–4576, 2022. doi: 10. 1109/TPAMI.2021.3069908.URL https://doi.org/10.1109/TPAMI.2021.3069908.

Xiaoxia Wu, Ethan Dyer, and Behnam Neyshabur. When do curricula work? In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id= tW4QEInpni.

Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy Liang, Quoc V. Le, Tengyu Ma, and Adams Wei Yu. DoReMi: Optimizing data mixtures speeds up language model pretraining. In Advances in Neural Information Processing Systems 36, pp. 69798–69818. Neural Information Processing Systems Foundation, Inc. (NeurIPS), 2023. doi: 10.52202/075280-3059. URL https://proceedings.neurips.cc/paper\_files/ paper/2023/hash/dcba6be91359358c2355cd920da3fcbd-Abstract-Conference.html.

Jiasheng Ye, Peiju Liu, Tianxiang Sun, Jun Zhan, Yunhua Zhou, and Xipeng Qiu. Data mixing laws: Optimizing data mixtures by predicting language modeling performance. In International Conference on Learning Representations, 2025. doi: 10.48550/ arXiv.2403.16952. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ cc84bfabe6389d8883fc2071c848f62a-Abstract-Conference.html.

## A Extended Related Work

Curriculum learning and difficulty-aware training. Classical curriculum learning moves from easy to hard and can stabilize optimization and improve generalization (Bengio et al., 2009). Automated curriculum-learning methods choose what data to present and when: self-paced learning matches example difficulty to learner competence (Kumar et al., 2010); teacher-student curricula prioritize items with high learning progress (Matiisen et al., 2020); bandit-style schedulers adapt task proportions online (Graves et al., 2017); and competencebased schedules control how quickly harder examples are introduced (Platanios et al., 2019). Evidence suggests that pacing strongly affects outcomes (Hacohen & Weinshall, 2019), while training-dynamics diagnostics provide signals for difficulty and ambiguity (Swayamdipta et al., 2020); related heuristics include hard-example mining and forgetting-based selection (Shrivastava et al., 2016; Toneva et al., 2019). Our framework instead represents a curriculum as a path over distributions on discrete difficulty levels and separates the path itself from its pacing. This makes it possible to move probability mass smoothly through nearby levels while controlling where the curriculum travels and how quickly it advances.

Optimal transport. Optimal transport measures distances between probability distributions via the minimal cost of moving mass; in this metric, Wasserstein (displacement) interpolation follows geodesics connecting two endpoint distributions (McCann, 1997; Peyré & Cuturi, 2019; Santambrogio, 2015). OT has also been used to construct curricula in reinforcement learning by generating geodesic sequences or constrained OT plans between task distributions (Huang et al., 2022; Klink et al., 2022). Our approach is related in that it also uses Wasserstein geodesics for curriculum learning, but the emphasis here is empirical control rather than a new RL-specific scheduler. We use Wasserstein paths as a common scaffold for comparing endpoint choice, smoothness, traversal speed, and adaptive geometry on calibrated synthetic tasks and language-model reasoning benchmarks. This keeps the distributional formulation fixed while studying which curriculum-design choices matter.

Data Mixing. Designing training distributions from heterogeneous sources is an important part of large-scale model training. Static approaches fix mixture weights in advance, e.g., DoReMi (Xie et al., 2023) optimizes domain sampling via a proxy model, and DoGE (Fan et al., 2024) estimates domain importance from small runs. Recent work also predicts or balances effective mixtures from proxy runs and learned signals: RegMix regresses performance from proxy runs (Liu et al., 2025b), data mixing laws model weight-loss functions (Ye et al., 2025), mixtures of data experts approximate source contributions (Belenki et al., 2025), and regroup-and-balance strategies adapt mixtures during training (Ge et al., 2025). Other work goes beyond source-level mixing, such as topic-based mixtures (Peng et al., 2025). Our work shares the perspective of viewing curricula as data mixing, but differs in two ways: (i) we emphasize distributional paths rather than static or adaptive reweighting, and (ii) we study principled design factors such as endpoint choice and pācing, complementing existing online weighting methods.

## B Detailed Experimental Setup

We include detailed experiment setups and data examples.

## B.1 Task and Dataset Details

Each difficulty axis below is described using the writing terms from our task-to-difficultyaxis mapping. When one difficulty axis varies, the remaining task parameters stay at the dataset defaults stated in the same paragraph.

## B.1.1 k-Parity Dataset

Task. Each example is a binary string $x \in \{ 0 , 1 \} ^ { n }$ with label

$$
y = f _ { k } ( x ) = \bigoplus _ { i = 1 } ^ { k } x _ { i } .
$$

Only the leading bits matter; the remaining bits are nuisance features. We report classification accuracy.

Construction and difficulty levels. We study two difficulty axes, each with 5 levels:

• Informative-bit difficulty axis: levels use {1,2, 3, 4,5} informative prefix bits while keeping the total string length fixed at 32

• Length difficulty axis: leves use total string lengths {24, 28, 32, 36, 40} while keeping the number of informative prefix bits fixed at 3.

For the informative-bit difficulty axis, later levels strictly contain earlier ones: before the final level, the later informative positions are fixed to 0 and are released only when that level is introduced. We therefore also evaluate each level on a unique slice where the newly introduced informative bit is 1 and all later informative positions remain 0, so that each test set isolates the new bit introduced at that level.

## Example.

$$
\begin{array} { l } { I n p u t . \ \theta 1 1 \ \theta 0 1 \ 1 \ \theta 0 0 1 \ \theta 0 0 1 \ \theta 0 0 0 1 \ \theta 1 \theta 1 \ \theta 1 \ \theta 1 \ \theta 1 \ \ \theta 1 \ 1 \ \theta } \\ { T a r g e t \ o u t p u t . \ \theta } \end{array}
$$

In this string, the first three bits are 0, 1, 1, so the correct label is 0 ⊕ 1 ⊕ 1 = 0. A model that still behaves like the previous level, using only the first two bits, would predict $0 \oplus 1 = 1$ instead. This is exactly the failure that the unique-slice evaluation is designed to expose.

Dataset size. We use batch size 1000. For the informative-bit difficulty axis, the small, medium, and large budgets are 100, 200, and 300 steps. For the length difficulty axis, they are 5000, 7500, and 10000 steps.

## B.1.2 Dyck Language Dataset

Task. Each example is a prefix-completion problem for the Dyck language. We sample a valid bracket string, reveal an initial prefix, and ask the model to output the full completed sequence, not just the missing tail. We report exact-match accuracy on the completed string.

Construction and difficulty levels. We vary one factor at a time:

• Bracket-type difficulty axis: 4 levels with {1, 2,3, 4} bracket types, while total sequence length stays fixed at 20 and hidden suffix length stays fixed at 3.

• Length difficulty axis: 4 levels with total sequence lengths {14, 16, 18, 20}, while the number of bracket types stays fixed at 4 and the hidden suffix length stays fixed at 3.

• Missing-suffix difficulty axis: 4 levels with hidden suffix lengths {3, 5, 7,9}, while the number of bracket types stays fixed at 4 and the total sequence length stays fixed at 20.

## Example.

Input prefix. [[(){}]][]() ()[<<   
Target output. [[(){}]][]()()[<<>>]

The missing part is short, but the model still has to track the open brackets that remain unresolved and finish the entire balanced string correctly.

Dataset size. We use batch size 100. For the bracket-type difficulty axis, the small, medium, and large budgets are 200, 350, and 1000 steps; for the length difficulty axis they are 200, 300, and 1000; and for the missing-suffix difficulty axis they are 300, 500, and 1000.

## B.1.3 Survo Dataset

Task. Survo asks the model to fill missing entries in a partially observed 4 × 4 grid whose last row and last column give row and column sums. The unknown cells always lie in the upper-left 3 × 3 block, and we report exact-match accuracy on the completed grid.

Construction and difficulty levels. We use two difficulty axes:

• Blank-count difficulty axis: 5 levels with {1, 2, 3, 4, 5} blanks, while the grid size stays fixed at 4 × 4 and the allowed cell range stays fixed to 1–9.

• Value-range difficulty axis: 4 levels with maximum allowed cell values {6, 9, 12, 15}, while the grid size stays fixed at 4 × 4, the number of blanks stays fixed at 3, and the minimum cell value stays fixed at 1.

Example.

Input. [[X, 1, 2, 9], [6, X, 1, 10], [4, 2, X, 8], [16, 6, 5, 27]]   
Target output. [[6, 1, 2, 9], [6, 3, 1, 10], [4, 2, 2, 8], [16, 6, 5,   
27]]

The first row must begin with 6 because 6 + 1 + 2 = 9; the second row must place 3 in the middle because 6 + 3 + 1 = 10; and the third row then takes 2.

Dataset size. We use batch size 100. For both difficulty axes, the small, medium, and large budgets are 7500, 15000, and 25000 steps.

## B.1.4 Dyck Language Errors Dataset

Task. This task presents a bracket string and asks for the first position at which it becomes invalid, using 1-indexing. Errors include an unmatched closing bracket, a closing bracket of the wrong type, or a prefix that is locally valid but still unfinished at the end; in the last case the answer is one position past the end of the string. Valid strings receive —1. We report exact-match accuracy.

Construction and difficulty levels. We use two difficulty axes, each with 4 levels:

• Bracket-type difficulty axis: levels use {1,2, 3, 4} bracket types while keeping the total string length fixed at 20.

• Length difficulty axis: levels use total string lengths {10, 15, 20, 25} while keeping the number of bracket types fixed at 3.

Example.

Input. ()))(()](()( Target output. 7

The first six symbols can still be parsed consistently, but the seventh symbol is an extra closing parenthesis with nothing left to match.

Dataset size. We use batch size 100. For the bracket-type difficulty axis, we use 8,000 training examples and 1,000 evaluation examples per level, and the small, medium, and large budgets are 300, 600, and 1,000 steps. For the length difficulty axis, we keep the original setup with 2,400 training examples and 2,000 evaluation examples per level, and the small, medium, and large budgets are 750, 1500, and 3000.

## B.1.5 Calendar Scheduling Dataset

Task. Each example specifies task durations, precedence constraints, and a finite time horizon for a single-machine schedule with no overlap. The solver always picks the alphabetically earliest available task and starts it immediately after the previous task ends. The model must output the full schedule as Task: start-end pairs, and we report exact-match accuracy on that string.

Construction and difficulty levels. We vary three factors separately:

• Task-count difficulty axis: 4 levels with {4, 6, 8, 10} tasks, while the time horizon stays fixed at 20, the number of precedence constraints stays fixed at 4, and the maximum task duration stays fixed at 3.

• Horizon difficulty axis: 4 levels with time horizons {12, 16, 20, 24}, while the number of tasks stays fixed at 6, the number of precedence constraints stays fixed at 4, and the maximum task duration stays fixed at 3.

• Dependency difficulty axis: 4 levels with {2, 4, 6, 8} precedence constraints, while the number of tasks stays fixed at 6, the time horizon stays fixed at 20, and the maximum task duration stays fixed at 3.

## Example.

Input. Time horizon 0--19; durations A=1 B=1 C=1 D=2 E=2 F=1;

dependencies B->D, B->E, A->D, E->F

Target output. A:0-1 B:1-2 C:2-3 D:3-5 E:5-7 F:7-8

Tasks A, B, and C are initially available, so the alphabetical rule places them first. Task D must wait for both A and B, and task F must wait for E.

Dataset size. We use batch size 100. For the task-count difficulty axis, the small, medium, and large budgets are 2000, 3000, and 4500 steps; for the horizon difficulty axis they are 1400, 1700, and 3000; and for the dependency difficulty axis they are 2000, 3000, and 4200.

## B.1.6 Game of 24 Dataset

Task. Each instance gives four integers and a target value. The model must produce an arithmetic expression that uses each number exactly once and combines them with the standard binary operations to reach the target. Because many expressions can be correct, we report verifier-based success rate rather than exact string match.

Construction and difficulty levels. We use two difficulty axes, each with 5 levels:

• Target-value difficulty axis: levels use target values {18, 21, 24, 27, 30 }, while the number of input numbers stays fixed at four, the allowed operations stay fixed to +, —, ×, ÷, and the input-number range stays fixed at 1–15.

• Number-range difficulty axis: levels use input-number ranges {1-12,1-15,1-18,1-21,1-24}, while the target value stays fixed at 24 and the number of input numbers stays fixed at four.

The second change creates a broader and less repetitive arithmetic search space.

## Example.

Input. Numbers: 3, 11, 4, 8; target: 24

Target output. 4\*((3+11)-8)

Any verifier-equivalent expression is counted as correct; this is one valid target output.

Dataset size. We use batch size 100. For the target-value difficulty axis, the small, medium, and large budgets are 4200, 5000, and 9500 steps. For the number-range difficulty axis, they are 2500, 5000, and 9000.

## B.1.7 Goods Exchange Dataset

Task. Each example starts from a one-to-one ownership map between people and objects, followed by a short sequence of exchange statements. The model must return the final ownership map, sorted alphabetically and formatted one line per person as Person: item. We report exact-match accuracy after canonicalizing outputs to that format.

Construction and difficulty levels. We use two difficulty axes, each with 4 levels:

• People-count difficulty axis: levels use {3, 4, 5, 6} people, and therefore the same number of items, while keeping the number of exchange statements fixed at 2.

• Exchange-length difficulty axis: levels use {2, 3, 4,5} exchange statements while keeping the number of people fixed at 3.

The first broadens the state being tracked; the second lengthens the chain of ownership updates.

## Example.

Input. Initial ownership: Sophia -> coral kettle; Betty -> beige

kettle; Donna -> beige watch.

Statements: Betty asked to exchange with Donna, but Donna

refused; Sophia and Donna exchanged items;

Sophia and Betty exchanged items.

Target output. Betty: beige watch

Donna: coral kettle

Sophia: beige kettle

Here the first statement is a decoy: Donna refuses Betty's request, so ownership does not change. The next two exchanges do the real work. After Sophia and Donna exchange items, Donna holds the coral kettle and Sophia holds the beige watch; after Sophia and Betty exchange items, Betty ends with the beige watch and Sophia with the beige kettle.

Dataset size. We use batch size 100. For the people-count difficulty axis, the small, medium, and large budgets are 500, 1000, and 1500 steps. For the exchange-length difficulty axis, they are 1500, 2250, and 3000.

## B.1.8 Index Select Dataset

Task. The input is a token sequence together with a list of inclusive 0-indexed ranges. The model must concatenate the selected spans in order and output the resulting token list, space-separated. We report exact-match accuracy.

Construction and difficulty levels. We vary four factors separately:

• Source-length difficulty axis: 5 levels with source sequence lengths {10, 12, 14, 16, 18}, while selected-output length stays fixed at 8, the number of ranges stays fixed at 3, and the alphabet size stays fixed at 4.

• Selected-output-length difficulty axis: 5 levels with {4, 6, 8, 10, 12} selected tokens, while the source sequence length stays fixed at 16, the number of ranges stays fixed at 3, and the alphabet size stays fixed at 4.

• Range-count difficulty axis: 5 levels with {1,2, 3, 4, 5} ranges, while the source sequence length stays fixed at 16, the selected-output length stays fixed at 8, and the alphabet size stays fixed at 4.

• Alphabet-size difficulty axis: 5 levels with alphabet sizes {2, 3, 4, 5, 6}, while the source sequence length stays fixed at 16, the selected-output length stays fixed at 8, and the number of ranges stays fixed at 3.

These changes make the copy operation longer, more fragmented, or less repetitive.

## Example.

Input. Sequence: d a d a a b a b a d d a b c a c; ranges: 0-1, 3-5,   
10-12   
Target output. d a a a b d a b

The range 0-1 contributes d a, the range 3-5 contributes a a b, and the range 10-12 contributes d a b.

Dataset size. We use batch size 100. For source length and selected-output length, the small, medium, and large budgets are 3000, 4000, and 6000 steps. For the number of ranges they are 3600, 4000, and 6400, and for alphabet size they are 2500, 3400, and 6000.

## B.1.9 Sequence Reverse Dataset

Task. This is the standard sequence-reversal task: given a token sequence, output the same sequence in reverse order. We report exact-match accuracy on the reversed sequence.

Construction and difficulty levels. We use two difficulty axes, each with 5 levels:

• Length difficulty axis: levels use sequence lengths {6, 8, 10, 12, 14} while keeping the alphabet size fixed at 4.

• Aiphabet-size difficulty axis: levels use alphabet sizes {2, 3, 4, 5, 6} while keeping the sequence length fixed at 12.

Example.

Input.d b c b b c b b b c a a   
Target output. a a c b b b c b b c b d

Dataset size. We use batch size 100. For both difficulty axes, the small, medium, and large budgets are 600, 700, and 800 steps.

## B.1.10 Run-Length Encoding Dataset

Task. Each example contains a token sequence that must be converted to run-length form as space-separated <count> <symbol> pairs. The task is deterministic, and we report exact-match accuracy on the encoded string.

Construction and difficulty levels. We vary four properties separately:

• Length difficulty axis: 5 levels with sequence lengths {6, 7,8, 9, 10}, while the maximum number of runs stays fixed at 8, the alphabet size stays fixed at 4, and the maximum run length stays fixed at 4.

• Run-count difficulty axis: 5 levels with maximum numbers of runs {4, 5, 6, 7,8}, while the sequence length stays fixed at 12, the alphabet size stays fixed at 4, and the maximum run length stays fixed at 4.

• Alphabet-size difficulty axis: 4 levels with alphabet sizes {2, 3, 4, 5}, while the sequence length stays fixed at 12, the maximum number of runs stays fixed at 8, and the maximum run length stays fixed at 4.

• Run-length difficulty axis: 4 levels with maximum run lengths {2, 3, 4, 5}, while the sequence length stays fixed at 12, the maximum number of runs stays fixed at 8, and the alphabet size stays fixed at 4.

These difficulty axes make the compressed description longer, more segmented, or more varied.

## Example.

Input.c a a a d d d d b b b b   
Target output. 1 c 3 a 4 d 4 b

The sequence contains one c, then three a's, then four d's, and finally four b's.

Dataset size. We use batch size 100. The small, medium, and large budgets are 1000, 1500, and 2200 steps when varying sequence length; 1000, 1350, and 2100 when varying the number of runs; 1000, 1500, and 2100 when varying alphabet size; and 1300, 1500, and 2200 when varying maximum run length.

## B.1.11 Stack Operations Dataset

Task. This task simulates a LIFO stack over a finite alphabet. Inputs are sequences of PUSH and POP operations, and the model must output the final stack from top to bottom, or EMPTY if no items remain. We report exact-match accuracy.

Construction and difficulty levels. We vary four factors separately:

• Operation-count difficulty axis: 4 levels with {6, 8, 10, 12} operations, while the alphabet size stays fixed at 4 and the pop probability stays fixed at 0.4; for this difficulty axis, the stack-depth cap scales with the operation count, with a minimum cap of 4.

• Alphabet-size difficulty axis: 4 levels with alphabet sizes {2, 3, 4, 5}, while the number of operations stays fixed at 8, the maximum stack depth stays fixed at 6, and the pop probability stays fixed at 0.4.

• Maximum-depth difficulty axis: 4 levels with maximum stack depths {2, 3, 4,5}, while the number of operations stays fixed at 8, the alphabet size stays fxed at 4, and the pop probability stays fixed at 0.4.

• Pop-rate difficulty axis: 4 levels with pop probabilities {0.65, 0.5, 0.35, 0.2}, listed from easy to hard, while the number of operations stays fixed at 8, the alphabet size stays fixed at 4, and the maximum stack depth stays fixed at 6.

Later levels therefore produce longer traces and, in the last difficulty axis, more live stack contents to remember.

## Example.

Input. PUSH a; PUSH a; POP; POP; PUSH d; POP; PUSH d; PUSH d   
Target output. d d

The first two pushes are canceled by the next two pops, the next PUSH d is immediately removed, and the last two pushes remain.

Dataset size. We use batch size 100. The small, medium, and large budgets are 300, 380, and 550 steps when varying the number of operations; 280, 420, and 600 when varying alphabet size; 180, 220, and 280 when varying maximum depth; and 280, 360, and 540 when varying the pop rate.

## B.1.12 Queue Operations Dataset

Task. This task simulates a FIFO queue. Inputs are ENQUEUE and DEQUEUE operations, and the model must output the final queue from front to back, or EMPTY if the queue is empty. We report exact-match accuracy.

Construction and difficulty levels. We report three difficulty axes for this task:

• Operation-count difficulty axis: 5 levels with {12, 16, 20, 24, 28} operations while keeping the other task parameters fixed at their default values.

• Alphabet-size difficulty axis: 4 levels with alphabet sizes {4, 6, 8, 10} while keeping the remaining parameters fixed.

• Queue-capacity difficulty axis: 4 levels with capacities {2, 3, 4, 5}, listed from easy to hard, while keeping the remaining parameters fixed.

## Example.

Input. ENQUEUE b; ENQUEUE d; DEQUEUE; DEQUEUE; ENQUEUE b; DEQUEUE; ENQUEUE $\mathsf { c } ;$ ENQUEUE a; ENQUEUE d; ENQUEUE d; DEQUEUE; DEQUEUE Target output. d d

The first three dequeues remove b, then d, then the later b. The remaining front-to-back queue is d d.

Dataset size. We use batch size 100. The small, medium, and large budgets are 450, 570 and 720 steps when varying the number of operations; 300, 400, and 640 when varying alphabet size; and 280, 390, ānd 550 when varying the queue-cap parameter.

## B.2 Architecture and Training Details

All models were trained on a single NVIDIA A100 GPU.

Scratch transformer tasks Most tasks in the current synthetic sweep, including k-Parity, Dyck Language, Dyck Language Errors, Sequence Reverse, Index Select, Run-Length Encoding, Stack Operations, Queue Öperations, Survo, and Calendar Scheduling, are trained from scratch as decoder-only transformers. Most use hidden size 256, 4 layers, and 8 attention heads with AdamW, learning rate $5 \times 1 0 ^ { - 5 } .$ , weight decay 0.1, gradient clipping at 1.0, and a constant learning-rate schedule with no warmup. The main architectural exceptions are k-Parity, which uses a smaller hidden size 128 / 1-layer / 4-head model and learning rate $5 \times 1 0 ^ { - 4 }$ with batch size 1,000; Dyck Language, which uses 3 layers; Survo, which uses 6 layers; and Calendar Scheduling, which uses a larger 6-layer model with hidden size 384 and 12 attention heads. Other scratch tasks typically use train batch size 100, with task-specific evaluation batch sizes and calibrated small/medium/large budgets given in the corresponding dataset subsections and task defaults.

Pretrained language-like tasks. GAME OF 24 and GOODS EXCHANGE use HuggingFaceTB/SmolLM-135M (Ben Allal et al., 2024). We fine-tune these models with AdamW using learning rate $\dot { 5 } \times 1 0 ^ { - 5 }$ , weight decay 0.1, gradient clipping at 1.0, and a constant learning-rate schedule without warmup.

## B.3 Level-Wise Cumulative Exposure

Figure 7 shows the cumulative level-wise exposure induced over the full training trajectory with $\gamma = 1$ and τ = 1. Uniform sampling gives equal exposure to all levels, linear interpolation allocates more exposure to the boundary levels, and Wasserstein interpolation allocates more exposure to the intermediate levels. In the default one-dimensional setting, this Wasserstein path uses the path metric $d ( \ell , \ell ^ { \prime } ) = | \ell - \ell ^ { \prime } | .$ , i.e., the standard $W _ { 1 }$ geometry on ordered levels. This illustrates that even when different curricula converge to similar overall performance, the Wasserstein curriculum achieves better sample efficiency at the higher levels.

## B.4 Adaptive Geometry in WARP

Section 7 uses a simple adaptive geometry on an ordered set of difficulty levels. This can be viewed as a discrete, gradient-informed geometry on the level axis. In the default setting, adjacent levels are equally spaced; WARP instead learns nonuniform edge lengths from local gradient alignment, yielding a discrete analogue of a one-dimensional Riemannian metric. This is loosely related to information geometry, where gradient information is used to define local geometry through the Fisher information matrix (Amari, 2016). Here, we use cosine similarity between neighboring level gradients as a simpler measure of local similarity. Let $P _ { 0 } , P _ { 1 } \in \dot { \Delta } ^ { m - 1 }$ denote the easy and hard endpoint distributions, let T be the total number of training steps, and let $T _ { w } = \lfloor \rho T \rfloor$ be a short warmup, with $\rho = 0 . 0 5$ in our experiments.

![](images/14c5ed20818d5617224cabb805751a4a09d96904ef512c9e8b89f016aba24153.jpg)  
Figure 7: Cumulative level-wise exposure under different curricula. Left: uniform sampling; middle: linear interpolation; right: Wasserstein interpolation.

During warmup, WARP uses uniform sampling:

$$
P _ { \mathrm { W A R P } } ^ { k } = \mathrm { U n i f } ( \{ 1 , \ldots , m \} ) , \qquad k < T _ { w } .
$$

At the end of warmup, it probes each level to obtain gradient vectors $g _ { 1 } , \ldots , g _ { m }$ and forms the cosine-coupling matrix

$$
C _ { i j } = \frac { \langle g _ { i } , g _ { j } \rangle } { \lvert \lvert g _ { i } \rvert \rvert \lvert g _ { j } \rvert \rvert } .
$$

It then converts adjacent couplings into edge lengths

$$
\Delta _ { \ell } = 2 - \operatorname { c l i p } ( C _ { \ell , \ell + 1 } , 0 , 1 ) , \qquad \ell = 1 , \dots , m - 1 ,
$$

defines locations $x _ { 1 } = 0$ and $\boldsymbol { x } _ { \ell + 1 } = \boldsymbol { x } _ { \ell } + \boldsymbol { \Delta } _ { \ell } ,$ and follows the same forward Wasserstein geodesic on the learned locations ${ \boldsymbol { x } } = \left( x _ { 1 } , \ldots , x _ { m } \right)$

$$
\tau _ { k } = \left( \frac { k - T _ { w } } { T - T _ { w } - 1 } \right) ^ { \gamma } , \qquad P _ { \mathrm { W A R P } } ^ { k } = \mathcal { G } _ { x } \bigl ( \tau _ { k } ; P _ { 0 } , P _ { 1 } \bigr ) , \qquad k \ge T _ { w } ,
$$

where $\mathcal { G } _ { x }$ denotes Wasserstein displacement interpolation on locations x. Thus WARP changes geometry, not endpoints or direction; pacing changes as a consequence. For visualization, Figure 5 rescales the learned locations affinely to a fixed display span; this does not change the relative geometry or the induced pacing pattern.

## B.5 Details for Structured Difficulty Spaces

Section 8 uses two structured variants of the same arithmetic-operator family: an 8-node cube and a 7-node tree. Figure 6 visualizes these two benchmark structures. In the cube, each node adds a subset of three operator modifications—multiplication, parentheses, and negation—so the graph follows the natural Boolean-cube structure from the simplest add/sub benchmark to the all-three endpoint. In the tree, the root splits first into multiplication and parentheses branches, and each branch then refines into longer-expression and negation leaves. In both cases, nodes are grouped into easy, mid, and hard regions; for the cube these correspond to L1, L2-L7, and L8, while for the tree they correspond to the root, internal branch nodes, and leaves.

For the cube, the Wasserstein endpoints over L1-L8 are (0.39, 0.14, 0.14, 0.14, 0.05, 0.05, 0.05, 0.02) and(0.02, 0.05, 0.05, 0.05, 0.14, 0.14, 0.14, 0.39). For the tree, the endpoints over L1–L7 are (0.44,0.16,0.16,0.06, 0.06, 0.06, 0.06) and (0.03, 0.08, 0.08, 0.21,0.21, 0.21, 0.21). The main comparison on structured difficulty spaces reports ten seeds for static i.i.d., Wasserstein, and WARP.

![](images/9f52dca6750fe707054522a466fa21b6c296acbe8a3ba1c24ad09f462347c7b1.jpg)  
Figure 8: Wasserstein path evolution on the cube and tree structures used in Section 8. Each column corresponds to normalized training progress $t \in \{ 0 . 0 , 0 . 2 , \ldots , 1 . 0 \}$ , and each cell shows the sampling probability assigned to that node along the graph-Wasserstein path.

## C Detailed Results and Mechanism Analyses

## C.1 Task-Wise Final-Accuracy Barplots

Setup. To complement Section 4.1, we report final overall accuracy separately for each task and difficulty axis. Each figure corresponds to one task. Within a figure, panels correspond to the task's difficulty axes, the x-axis groups the small, medium, and large budgets, and colored bars compare the eight curricula: static i.i.d., linear, Wasserstein, WARP, the reverse variants, and the two static-matching baselines. Bars show the mean over 10 random-seed repetitions, and error bars show one standard deviation.

Findings. These plots expose the heterogeneity behind Table 1: the ranking shifts across tasks, difficulty axes, and budgets, rather than collapsing to a single dominant curriculum.

![](images/27b5c20dcde7396a0503655c7d1ee1f7be91e2864957382150517efb1892c882.jpg)  
Figure 9: Task-wise final overall accuracy for K-Parity. Panel titles denote the difficulty axes. Bars show mean final overall accuracy over 10 random-seed repetitions, and error bars show one standard deviation.

![](images/6fc32a867d3c32855465e66b0811a3a3b23a3cf22df3190352bddcd1bd07da89.jpg)  
Figure 10: Task-wise final overall accuracy for Dyck Language. Panel titles denote the dificulty axes. Bars show mean final overalí accuracy over 10 random-seed repetitions, and error bars show one standard deviation.

![](images/64250644c30b754bf971cdcb6e245e003856b38550688d75713ff9372847511b.jpg)  
Figure 11: Task-wise final overall accuracy for Survo. Panel titles denote the difficulty axes. Bars show mean final overall accuracy over 10 random-seed repetitions, and error bars show one standard deviation.

![](images/3fb898460634c6581b132d163c6815f8babe3e8e87ce988cee3479668d94fb74.jpg)  
Figure 12: Task-wise final overall accuracy for Dyck Language Errors. Panel titles denote the difficulty axes. Bars show mean final overall accuracy over 10 random-seed repetitions, and error bars show one standard deviation.

![](images/ac7c33536f4169cc24ea60047cf163c39b16e4b0ff0b4de324a64de1489eea83.jpg)  
Figure 13: Task-wise final overall accuracy for Calendar Scheduling. Panel titles denote the dificulty axes. Bars show mean final overall accuracy over 10 random-seed repetitions, and error bars show one standard deviation.

![](images/8495c7fed3593db06fa6d5b6db66fcac8a1efbf3923ef532843ed16ed668534c.jpg)  
Figure 14: Task-wise final overall accuracy for Game of 24. Panel titles denote the difficulty axes. Bars show mean final overall accuracy over 10 random-seed repetitions, and error bars show one standard deviation.

![](images/be4be2798ec6c64c1a9d5e8f30144078353d4eefefdbdda361f94c0d8549b9d3.jpg)  
Figure 15: Task-wise final overall accuracy for Goods Exchange. Panel titles denote the dificulty axes. Bars show mean final overall accuracy over 10 random-seed repetitions, and error bars show one standard deviation.

Index Select  
![](images/7be39580edb88b957bf5c1d67100933efef7223043b04f3bc89acd310c3fd766.jpg)

![](images/3b492d98f16bc4bde28ac3b1794b10e7957ba141b8426d592437bcb809774df2.jpg)

![](images/5fc4ba8f2d47827dc1caae53e8dc7f18eb822022aa1706dd1cd517f0cf6ed9ff.jpg)

![](images/648e3b8675460560e632e096a750fae9e729d71c73f5303e44d8ef374b65d7e4.jpg)  
Figure 16: Task-wise final overall accuracy for Index Select. Panel titles denote the difficulty axes. Bars show mean final overall accuracy over 10 random-seed repetitions, and error bars show one standard deviation

![](images/b51d96a02ae3cdee072fd71f8ec34bc6281946cc9325fbcf6bc4be8831bc282e.jpg)

Queue Operations  
![](images/a4062479437f5c2c7cb5ed9f558db3b8bf26231247dbcf41b1e952d9b620cfd3.jpg)

![](images/19f68a4efb5088b6242a46aa4e3b68dc74b7d67c6d038bfa5bab28efb5610c02.jpg)  
Figure 17: Task-wise final overall accuracy for Queue Operations. Panel titles denote the difficulty axes. Bars show mean final overall accuracy over 10 random-seed repetitions, and error bars show one standard deviation.

Run-Length Encoding  
![](images/04c5848fe6da16c0eb8ab42e59c2cb6d60e20334529c1ed3ae9cb2ce008d7426.jpg)  
Figure 18: Task-wise final overall accuracy for Run-Length Encoding. Panel titles denote the difficulty axes. Bars show mean final overall accuracy over 10 random-seed repetitions, and error bars show one standard deviation.

Sequence Reverse  
![](images/6c11436c9d4248983f019cff688f21c23bc0e37cb97d9e4e4c8a8d90cd16d3ba.jpg)  
Figure 19: Task-wise final overall accuracy for Sequence Reverse. Panel titles denote the difficulty axes. Bars show mean final overall accuracy over 10 random-seed repetitions, and error bars show one standard deviation.

![](images/1522e13f12128104a14e551f7275bea0acd7e4fbf7f8787ae49d9ebdbea69b5c.jpg)  
Figure 20: Task-wise final overall accuracy for Stack Operations. Panel titles denote the difficulty axes. Bars show mean final overall accuracy over 10 random-seed repetitions, and error bars show one standard deviation.

## C.2 Budget-Wise Exposure, Accuracy, and Exposure-Adjusted Accuracy

Setup. To complement Section 4.2, we report the same easiest, middle, and hardest probebucket summaries separately for the small, medium, and large budgets. Exposure is the cumulative share of training mass assigned to each bucket over the full run. As a simple proxy for data efficiency, we use exposure-adjusted accuracy: final bucket accuracy divided by cumulative exposure to that bucket.

Findings. The medium row reproduces Figure 2, while the small and large rows show how the same tradeoff evolves with budget. The qualitative pattern is stable: static i.i.d. remains strongest on easy-end accuracy, linear is strongest in exposure-adjusted accuracy in the middle bucket, and Wasserstein is strongest on the hardest bucket.

![](images/0006bdba46516d309cc41865ec156a5dfb2224378ffd19a623768e70e2e7b7f8.jpg)  
Figure 21: Exposure, accuracy, and exposure-adjusted accuracy across budgets. Rows denote the smāll, medium, and large budgets. Columns denote cumulative exposure share, final bucket accuracy, and exposure-adjusted accuracy on the easiest, middle, and hardest probe buckets.

## C.3 Task-Wise Final Level Profiles

Setup. To complement the task-wise endpoint barplots above, we also visualize where final performance lands across the available difficulty levels. Each figure again corresponds to one task, with rows denoting difficulty axes and columns denoting the small, medium, and large budgets. Within each panel, colored curves trace the mean final accuracy at each level for the eight curricula. Tasks with four levels stop at Level 4, while five-level tasks extend through Level 5.

Findings. These profile grids make the location of gains explicit: some curricula preserve more of the easy-end accuracy, while others shift the profile upward at the harder end. They therefore complement the budget-wise summaries and task-wise endpoint barplots by showing not only which curriculum wins overall, but also where along the task's level structure the final gains and tradeoffs occur

k-Parity  
![](images/c4a155f48075b0f1718df2b2318e5344ebe965151d5764fd400503cd31dc6b15.jpg)  
Figure 22: Task-wise final level profiles for K-Parity. Rows denote difficulty axes, columns denote budgets, and curves show mean final accuracy at each available difficulty level over 10 random-seed repetitions.

Dyck Language  
![](images/8db9b8ea06f92abe4026410ebf8ac026c59c9689038f9eb22fae5ea6e1c0f517.jpg)  
Figure 23: Task-wise final level profiles for Dyck Language. Rows denote difficulty axes, columns denote budgets, and curves show mean final accuracy at each available difficulty level over 10 random-seed repetitions.

![](images/ce708548bfda570a6611caec0c1ae620388ded7b2fa137f1cf8f885bbb5a256f.jpg)  
Figure 24: Task-wise final level profiles for Survo. Rows denote difficulty axes, columns denote budgets, and curves show mean final accuracy at each available dificulty level over 10 random-seed repetitions.

Dyck Language Errors  
![](images/06dd94ed20e1e553aa482de6850ee75946125c016bd40eb45b82604c6e4fc376.jpg)  
Figure 25: Task-wise final level profiles for Dyck Language Errors. Rows denote difficulty axes, columns denote budgets, and curves show mean final accuracy at each available difficulty level over 10 random-seed repetitions.

Calendar Scheduling  
![](images/b364f9eacfaa73cd78b0b7fe1b7c2d1db14a7ec9e37c1bb676085faf8f151aa0.jpg)  
Figure 26: Task-wise final level profiles for Calendar Scheduling. Rows denote difficulty axes, columns denote budgets, and curves show mean final ačcuracy at each available difficulty level over 10 random-seed repetitions.

Goods Exchange  
Game of 24  
![](images/f302ed3907186ffbce8087f4aad5548830a4321d839efa641eea2cc3849d9736.jpg)  
Figure 27: Task-wise final level profiles for Game of 24. Rows denote difficulty axes, columns denote budgets, and curves show mean final accuracy at each available difficulty level over 10 random-seed repetitions.

![](images/7da81462af9f73cb37796f376327843eca19e2410249468028ae2fd6313cd2aa.jpg)  
Figure 28: Task-wise final level profiles for Goods Exchange. Rows denote difficulty axes, coumns denote budgets, and curves show mean final accuracy at each available dificulty level over 10 random-seed repetitions.

Index Select  
![](images/d12ae62a3e70f2924189df26cc26da8f7e7dadcd5115130210e0b0d3f6651e7a.jpg)  
Figure 29: Task-wise final level profiles for Index Select. Rows denote difficulty axes, columns denote budgets, and curves show mean final accuracy at each available difficulty level over 10 random-seed repetitions.

Queue Operations  
![](images/d5fc804207c6de35c1241ab88cae628d4715a097dc944fa4d4489c6c9847cbdc.jpg)  
Figure 30: Task-wise final level profiles for Queue Operations. Rows denote difficulty axes, coumns denote budgets, and curves show mean final accuracy at each available difficulty level over 10 random-seed repetitions.

Run-Length Encoding  
![](images/814b551d21d87f2c20a661f09adf7eac4c346f0e1b7040da21c42eb67ea47a42.jpg)  
Figure 31: Task-wise final level profiles for Run-Length Encoding. Rows denote difficulty axes, columns denote budgets, and curves show mean final accuracy at each available difficulty level over 10 random-seed repetitions.

Sequence Reverse  
![](images/fa23a782c0e155254bc27729b8ef309372c722a0e49c90220fa814e58fde9aab.jpg)  
Figure 32: Task-wise final level profiles for Sequence Reverse. Rows denote difficulty axes, columns denote budgets, and curves show mean final accuracy at each available difficulty level over 10 random-seed repetitions.

Stack Operations  
![](images/4137862b7a7d808ac72777680e95b365835239bc250e0cdfcd4832a5e15ea623.jpg)  
Figure 33: Task-wise final level profiles for Stack Operations. Rows denote difficulty axes, columns denote budgets, and curves show mean final accuracy at each available difficulty level over 10 random-seed repetitions.

## C.4 Exploratory Analysis of When Easy-to-Hard Progression Helps

Many curriculum methods depend strongly on the chosen difficulty signal or decomposition (Platanios et al., 2019; Hacohen & Weinshall, 2019; Swayamdipta et al., 2020; Jia et al., 2026). We group the 99 task-by-axis-and-budget conditions by task family, difficulty-axis category, and their intersection, and count where Wasserstein or WARP attains the best mean among the four forward curricula

I. Tasks. We first ask the same question at the task level by grouping the 12 benchmarks into four coarse task families listed below. Unlike the difficulty-axis taxonomy below, these task families do not overlap.

• Sequence Transformation: INDEX SELECT, RUN-LENGTH ENCODING, SEQUENCE RE-VERSE

• Formal Structure: DYCK LANGUAGE, DYCK LANGUAGE ERRORS, K-PARITY

• Data-Structure Execution: QUEUE OPERATIONS, STACK OPERATIONS

• Constraint and Search: CALENDAR SCHEDULING, GAME OF 24, GOODS EXCHANGE, SURVO

<table><tr><td>Task family</td><td>Overall wins</td><td>Hardest-level wins</td></tr><tr><td>Sequence Transformation</td><td>18/30 (10 W, 8 A)</td><td>21/30 (14 W, 7 A)</td></tr><tr><td>Formal Structure</td><td>15/21 (4 W, 11 A)</td><td>13/21 (4 W, 9 A)</td></tr><tr><td>Data-Structure Execution</td><td>7/21 (2 W, 5 A)</td><td>8/21 (2 W, 6 A)</td></tr><tr><td>Constraint and Search</td><td>15/27 (7 W, 8 A)</td><td>15/27 (8 W, 7 A)</td></tr></table>

Table 4: Task-family summary of when natural easy-to-hard progression helps. We group the 12 benchmarks into four coarse task families and count the number of task-by-axisand-budget conditions for which Wasserstein or WARP attains the best mean among the four forward curricula (static i.i.d., linear, Wasserstein, WARP). The left numeric pair reports overall wins; the right reports wins on the hardest available level. 'W' denotes Wasserstein and 'A' denotes WARP.

Task findings. The task taxonomy is somewhat cleaner than the axis taxonomy, but it still does not yield a single simple rule. Natural easy-to-hard progression is strongest on Sequence Transformation tasks, where it wins in a majority of conditions overall and even more often on the hardest level. Formal Structure and the broader Constraint and Search family also show substantial support, especially at the hard end. By contrast, Data-Structure Execution remains more mixed.

II. Difficulty Axes. We next simplify the raw axis list by grouping the axes into four broad categories: Context Length (how much input must be processed), Answer Length (how much output must be produced), Entity Complexity (the number or variety of entities/attributes), and Procedural Čomplexity (the depth' or expressivity of the reasoning chain). We again examine winners both in final overall accuracy and on the hardest available level.

• Context Length: representative pairs include CALENDAR SCHEDULING / Time horizon, DYCK LANGUAGE \~/ Total sequeñce length, INDEX SELECT / Source sequence length, and SURVO / Number of blanks

• Answer Length: representative pairs include DYCK LANGUAGE / Hidden suffix length, INDEX SELECT / Selected-output length, RUN-LENGTH ENCODING / Maximum run length, and SURVO / Number of blanks

• Entity Complexity: representative pairs include DYCK LANGUAGE / Number of bracket types, GAME OF 24 / Input-number range, INDEX SELECT / Alphabet size, and SURVO / Maximum allowed cell value

• Procedural Complexity: representative pairs include CALENDAR SCHEDULING / Number of precedence constraints, GOODs ExCHANGE / Number of exchange statements, INDEX SELECT / Number of ranges, and STACK OPERATIONS / Maximum stack depth

<table><tr><td>Axis category</td><td>Overall wins</td><td>Hardest-level wins</td></tr><tr><td>Context Length</td><td>25/39 (12 W, 13 A)</td><td>23/39 (13 W, 10 A)</td></tr><tr><td>Answer Length</td><td>10/18 (6 W, 4 A)</td><td>15/18 (11 W, 4 A)</td></tr><tr><td>Entity Complexity</td><td>26/39 (10 W, 16 A)</td><td>23/39 (8 W, 15 A)</td></tr><tr><td>Procedural Complexity</td><td>11/30 (3 W, 8 A)</td><td>16/30 (8 W, 8 A)</td></tr></table>

Table 5: Difficulty-axis-category summary of when natural easy-to-hard progression helps. We group difficulty axes into four broad categories and count the number of task-byaxis-and-budget conditions for which Wasserstein or WARP attains the best mean among the four forward curricula (static i.i.d., linear, Wasserstein, WARP). The left numeric pair reports overall wins; the right reports wins on the hardest available level. 'W' denotes Wasserstein and 'A' denotes WAR. Because some axes belong to more than one category, the denominators overlap across rows.

Axis findings. The axis taxonomy helps simplify the picture, but it still does not collapse the results to a single winning axis type. Context Length and Entity Complexity contain many overall wins, suggesting that natural easy-to-hard progression often helps when an axis enlarges the amount of input/state to process or the variety of entities/attributes.

![](images/6eec90c1b13faf617404c56688fe41f5975a48cf8b522365b58023c627488fc8.jpg)

![](images/586e9bcae6078cb16974e8b4f943c1d63306d54975edf705a51f541f748f9533.jpg)  
Figure 34: Joint task-family × difficulty-axis-category view of when natural easy-to-hard progression helps. Rows denote task families and columns denote difficulty-axis categories. The left panel shows overall win rates and the right panel shows hardest-level win rates for Wasserstein or WARP against the other forward curricula. Each cell reports both the raw count and the percentage. Gray cells indicate that no task in that family has an axis of the corresponding category.

Answer Length is the strongest category on the hardest level (15/18), driven by axes such as Hidden suffix length, Maximum run length, and Selected-output length. Procedural Complexity is more mixed: it is the weakest category overall, but its hardest-level rate improves substantially, suggesting that easy-to-hard progression can still help on the difficult end without consistently improving aggregate performance.

III. Joint View. Figure 34 combines the two taxonomies by showing, for each task family and difficulty-axis category pair, the fraction of conditions in which Wasserstein or WARP attains the best mean among the four forward curricula. The left panel reports overall wins and the right reports hardest-level wins; each cell is annotated with both the raw count and the corresponding win rate.

Joint findings. The interaction view clarifies where the previous summaries line up and where they do not. Sequence Transformation is strongest when paired with Entity Complexity or Procedural Complexity axes, especially on the hardest level. Constraint and Search is strongest when paired with Context Length or Answer Length axes, but shows little support on Procedural Complexity. Formal Structure remains broadly positive across categories, though with fewer total conditions, while Data-Structure Execution stays weak across most cells These three views suggest that easy-to-hard progression is easier to characterize at the level of coarse task family than at the level of semantic axis type, but even the joint taxonomy remains only partiaily predictive.

The taxonomy views above describe where natural easy-to-hard progression can help. We next ask what mechanisms may explain those gains by examining isolated level-to-level transfer and controlled bridge effects.

## C.5 Level Transfer Experiments

We study isolated transfer between difficulty levels to ask a simple mechanism question: when source-level training helps a target level, does it mainly give the target stage a head start, or does it make target-stage learning itself faster?

Setup. For each task, difficulty axis, seed, and ordered pair of distinct levels, we train on a source level and then continue only on a destination level. Let B denote the task's per-level small budget used in the main synthetic experiments. We write s ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0} for the source-stage share, so the source stage uses sB steps while the destination stage always uses B steps. We compare each transferred run against the matched s = 0 no-transfer control. We start with lower-to-higher transfer and then return to higher-to-lower transfer when analyzing reverse ordering.

![](images/bf42b6be6bc767b8ee54a32da755b8288b04302d306496c4d261702377a21a99.jpg)  
Figure 35: Two easy-to-hard examples under the first-stage fit $t _ { \mathrm { t r } , i } ( a ; s ) \approx \alpha _ { i , s } + \beta _ { i , s } t _ { \mathrm { c t r l } , i } ( a )$ We define $t ( a )$ as the first time a target-accuracy curve reaches a. Blue: matched 0% sourceshare control. Orange shades: transferred runs with source shares s ∈ {20, 40, 60, 80, 100 }%. Curves start at the first target-stage evaluation, at 5% progress. Panel (a) is head-start dominant. Panel (b) is faster-learning dominant. All axes are in percent.

Fit by source share. For each easy-to-hard ordered pair i and each fixed $s > 0$ , let $A _ { \mathrm { t r } , i } \big ( t ; s \big )$ denote target accuracy during the target stage, where $t \in [ 0 , 1 ]$ is normalized target-stage progress. Let $A _ { \mathrm { c t r l } , i } ( \check { t } ) = A _ { \mathrm { t r } , i } ^ { \smile } ( t ; 0 )$ denote the matched no-transfer control. Over the accuracy range reached by both runs, we define $t _ { \mathrm { t r } , i } ( a ; s )$ and $t _ { \mathrm { c t r l } , i } ( a )$ as the earliest targetstage progress at which each run first reaches accuracy a, using linear interpolation between evaluations. We first fit

$$
t _ { \mathrm { t r } , i } ( a ; s ) \approx \alpha _ { i , s } + \beta _ { i , s } t _ { \mathrm { c t r l } , i } ( a ) .
$$

This gives one fitted pair $\left( \alpha _ { i , s } , \beta _ { i , s } \right)$ for each ordered pair and each source share. We then model how these coefficients vary with source share:

$$
\alpha _ { i , s } = u _ { \alpha , i } + f _ { \alpha } ( s ) , \qquad \log \beta _ { i , s } = u _ { \beta , i } + f _ { \beta } ( s ) ,
$$

where $u _ { \alpha , i }$ and $u _ { \beta , i }$ are pair-specific effects and $f _ { \alpha } , f _ { \beta }$ are quadratic functions of s. We anchor the fit at the no-transfer point $( s = 0 , \alpha = 0 , \beta = 1 )$ . Negative $\alpha _ { i , s }$ indicates a head start. Values $\beta _ { i , s } < 1$ indicate faster target-stage learning. Figure 35 shows two representative first-stage examples. Figure 36 shows how the fitted coefficients vary with source share, together with the empirical distributions.

Results. Figure 35 shows two recurring patterns: larger source share either shifts the target-stage curve earlier or steepens the remaining climb. Figure 36 shows the same pattern in aggregate. Across all easy-to-hard pairs, the fitted coefficients move from (—0.08, 0.83) at 20% source share to (—0.43, 0.45) at 100%, so both the head-start effect and the fasterlearning effect grow with source share. On hardest-target pairs, the pattern leans further toward faster learning: the fitted coefficients move from (–0.07, 0.85) at 20% to (—0.12, 0.36) at 100%. The high end also flattens: on this subset, mean β changes from 0.63 to 0.55 between 80% and 100%, while mean α changes from —0.15 to —0.11.

Finding 5. Easy-to-hard transfer helps through both head start and faster target-stage learning.   
Across source shares, faster target-stage learning is especially prominent on the hardest targets.

![](images/a1c5a32270472d68705159d34d7156976cadd94e8cfe6fb0613b335d0036a2d9.jpg)

![](images/afec721de3e00ab194acbbf6027424707befe5e5a3aa25cd9dde2b56f533e011.jpg)  
Figure 36: Transfer coefficients by source share. We estimate $\left( \alpha _ { i , s } , \beta _ { i , s } \right)$ at each observed source share from earliest hitting times on target-accuracy curves, then fit quadratic trends to the median coefficients across shares (anchored at the no-transfer point). Curves show the median-by-share trend and boxplots show the empirical distribution. Solid blue uses all easy-to-hard ordered pairs. Dashed red uses the hardest-target subset.

Directional asymmetry. Using isolated transfer at 100% source share, we compare each lower/higher level pair in both directions. Figure 37(a) shows mean final target-accuracy gain over the matched 0% control; hard-to-easy transfer is often larger, especially toward easy targets. Panels (b) and (c) show why reverse still loses: it spends early budget on hard levels, leaving less remaining margin when it reaches easier targets, so the Wassersteinover-reverse gap is small (often negative) on the easy bucket and large on the hard bucket. Reverse's extra easy-bucket gains therefore do not offset Wasserstein's larger hard-bucket gains.

![](images/27acf650764a3ece81ecffa05cfa1673e8107e544004e6d107d650d545a3a26b.jpg)

![](images/5e472761e86ec55c3144e134d8ab3bd5622b8262fb23937fb5e968e74cb4cd20.jpg)

![](images/cac6bc3f9c8adacb3ee93ed98d913ab66a5f19b5eeb1b8e3cd7b6a217ba1a884.jpg)  
Figure 37: Directional asymmetry in isolated transfer at 100% source share. Panel (a) shows mean final target-accuracy gain over matched 0% controls for each ordered pair; the upper triangle is easy-to-hard and the lower triangle is hard-to-easy. Panels (b) and (c) show Wasserstein minus reverse gaps on the easy and hard buckets as a function of final-gain asymmetry.

Finding 6. Hard-to-easy transfer is often larger than easy-to-hard transfer, but reverse Wasserstein still loses: it trains on hard levels first and reaches easier targets later with less remaining margin, so easy-bucket gains do not offset Wasserstein's larger hard-bucket gains.

## C.6 Bridge Effect Experiments

Setup. We test whether an intermediate level helps a hard target by running three-stage experiments on contiguous triples $( \ell , \ell + 1 , \ell + 2 )$ . Stage 1 trains on the source level, stage

![](images/dd4ebe223ab641e99def309269c159670d3b7dfa897fbb8bb04ab06c10d9dc67.jpg)

![](images/ab5e4c00b8de820353bd09aa3d97f43d5a623f9c77383bfcebdb53f80688b411.jpg)  
Figure 38: Bridge-share response experiments. Panel (a) shows mean final target accuracy versus bridge share, averaged over bridge triples and seeds. Panel (b) plots, for each task-by-axis setting, the median bridge-share slope over hardest-target triples against the medium-budget hard-bucket Wasserstein-linear gap.

2 uses a fixed refine budget that is split between staying on the source and moving to the bridge level, and stage 3 trains on the target level. We write $b \in \{ 0 , 0 . 2 5 , 0 . 5 , 0 . 7 5 , \breve { 1 } . 0 \}$ for the fraction of the stage-2 budget spent on the bridge level, where $b = 0$ is the skip-bridge control and $b = 1$ is the full-bridge condition. For triple i, let $A _ { i } ( b )$ denote final target accuracy. To use the full bridge-share sweep, we fit

$$
A _ { i } ( b ) \approx u _ { i } + \lambda _ { i } b ,
$$

where $u _ { i }$ is a triple-specific intercept and $\lambda _ { i }$ is the bridge-share slope. Larger $\lambda _ { i }$ means that moving more of stage 2 onto the bridge level helps more.

Results. Figure 38(a) shows an overall upward response: mean final target accuracy rises from 23.49% at 0% bridge share to 25.31% at 100%, though the sweep is not perfectly monotone. Figure 38(b) then plots, for each task-by-axis setting, the median bridge-share slope over hardest-target triples against the medium-budget hard-bucket Wasserstein-linear gap from Section 4.1. The association is positive $( r =  { \breve { 0 . 3 7 } } )$ . Settings whose hard targets improve more as bridge share increases also tend to be the settings where Wasserstein gains more over linear on the hard bucket. This is consistent with the same local-transfer picture suggested by Figure 37(a): an intermediate level can split one hard jump into two easier ones.

Finding 7. Larger bridge effects tend to align with larger Wasserstein-over-linear gains.

## D Real-World SFT Analyses

This appendix studies the pretrained SFT setting. We first evaluate real-world SFT across three bāse models, then use a synthetic-pretraining analogue to test whether pretraining itself weakens curriculum effects.

## D.1 Real-World SFT Benchmarks

Setup. We run LoRA (Hu et al., 2022) SFT on ARC (Clark et al., 2018), MMLU-STEM (the STEM subset of MMLU; Hendrycks et al. 2021), and StrategyQA (Geva et al., 2021), following the benchmark construction and difficulty labels from Hase et al. (2024). We compare $\mathsf { S m o l L M 3 - } 3 \mathsf { B } ^ { 1 } , 0 \mathsf { L M o } 2 \mathsf { - } 1 \mathsf { B } ^ { 2 }$ , and $_ { \mathrm { Q w e n } 3 - 1 . 7 \mathsf { B } ^ { 3 } }$ under the same curriculum sweep, with three seeds, learning rate $5 \times 1 0 ^ { - 6 }$ , and batch size 128.

Benchmarks and difficulty axes. We study 12 benchmark-by-axis contexts: six for ARC, three for MMLU-STEM, and three for StrategyQA. ARC uses human grade (6 levels), human difficulty (3), human Bloom level (5), human depth of knowledge (3), question length (5 quantile bins), and answer length (5 quantile bins). MMLU-STEM uses human hardness (2 levels), question length (5 quantile bins), and answer length (5 quantile bins) over the five STEM subject groups from Hase et al. (2024). StrategyQA uses decomposition length (5 levels), question length (5 quantile bins), and reasoning length (5 quantile bins). Training targets contain a short reasoning chain plus the final answer, while evaluation uses only the final answer.

Budgets. We calibrate small, medium, and large budgets separately for each dataset: ARC uses 32, 64, and 128 update steps; MMLU-STEM uses ${ \mathrm { 1 6 } } , 3 2 { \mathrm { , } }$ and 64; and StrategyQA uses 64, 96, and 128. Unless otherwise stated, we average over the same three random seeds.

Static i.i.d. Remains the Strongest Summary Baseline. Tables $6 , 7 ,$ and 8 summarize the three basic curricula over these 12 contexts for each model. Across all three models, static i.i.d. has the highest mean overall accuracy and the highest mean hardest-level accuracy in every model-budget block. Linear and Wasserstein still win isolated contexts, but the gaps are small and much less systematic than in the synthetic suite.

<table><tr><td>Curriculum</td><td>Mean overall (%)</td><td>Mean hardest (%)</td><td>Overall wins</td><td>Hardest wins</td></tr><tr><td>Small budget</td><td></td><td></td><td></td><td></td></tr><tr><td>Static i.i.d.</td><td>57.5</td><td>53.7</td><td>8</td><td>8</td></tr><tr><td>Linear</td><td>57.0</td><td>52.5</td><td>3</td><td>3</td></tr><tr><td>Wasserstein</td><td>57.1</td><td>52.7</td><td>1</td><td>1</td></tr><tr><td>Medium budget</td><td></td><td></td><td></td><td></td></tr><tr><td>Static i.i.d.</td><td>63.4</td><td>58.3</td><td>6</td><td>6</td></tr><tr><td>Linear</td><td>62.7</td><td>57.3</td><td>3</td><td>1</td></tr><tr><td>Wasserstein</td><td>62.7</td><td>58.1</td><td>3</td><td>5</td></tr><tr><td>Large budget</td><td></td><td></td><td></td><td></td></tr><tr><td>Static i.i.d.</td><td>67.6</td><td>63.4</td><td>9</td><td>6</td></tr><tr><td>Linear</td><td>66.0</td><td>62.1</td><td>1</td><td>4</td></tr><tr><td>Wasserstein</td><td>66.6</td><td>62.9</td><td>2</td><td>2</td></tr></table>

Table 6: SFT summary for the three basic curricula on ARC, MMLU-STEM, and StrategyQA using SmolLM3-3B. Each budget block contains 12 benchmark-by-axis contexts: six ARC axes, three MMLU-STEM axes, and three StrategyQA axes. "Hardest" denotes the last available level.
<table><tr><td>Curriculum</td><td>Mean overall (%)</td><td>Mean hardest (%)</td><td>Overall wins</td><td>Hardest wins</td></tr><tr><td>Small budget</td><td></td><td></td><td></td><td></td></tr><tr><td>Static i.i.d.</td><td>37.6</td><td>35.5</td><td>4</td><td>6</td></tr><tr><td>Linear</td><td>37.4</td><td>35.4</td><td>3</td><td>3</td></tr><tr><td>Wasserstein</td><td>37.5</td><td>35.5</td><td>5</td><td>3</td></tr><tr><td>Medium budget</td><td></td><td></td><td></td><td></td></tr><tr><td>Static i.i.d.</td><td>39.0</td><td>36.8</td><td>6</td><td>5</td></tr><tr><td>Linear</td><td>38.2</td><td>36.1</td><td>1</td><td>3</td></tr><tr><td>Wasserstein</td><td>38.5</td><td>36.4</td><td>5</td><td>4</td></tr><tr><td>Large budget</td><td></td><td></td><td></td><td></td></tr><tr><td>Static i.i.d.</td><td>45.6</td><td>42.8</td><td>8</td><td>6</td></tr><tr><td>Linear</td><td>44.4</td><td>41.6</td><td>3</td><td>5</td></tr><tr><td>Wasserstein</td><td>44.4</td><td>41.3</td><td>1</td><td>1</td></tr></table>

Table 7: SFT summary for the three basic curricula on ARC, MMLU-STEM, and StrategyQA using OLMo2-1B. Each budget block contains 12 benchmark-by-axis contexts: six ARC axes, three MMLU-STEM axes, and three StrategyQA axes. "Hardest" denotes the last available level.
<table><tr><td>Curriculum</td><td>Mean overall (%)</td><td>Mean hardest (%)</td><td>Overall wins</td><td>Hardest wins</td></tr><tr><td>Small budget</td><td></td><td></td><td></td><td></td></tr><tr><td>Static i.i.d.</td><td>62.6</td><td>60.4</td><td>6</td><td>5</td></tr><tr><td>Linear</td><td>62.1</td><td>60.2</td><td>3</td><td>3</td></tr><tr><td>Wasserstein</td><td>61.8</td><td>60.4</td><td>3</td><td>4</td></tr><tr><td>Medium budget</td><td></td><td></td><td></td><td></td></tr><tr><td>Static i.i.d.</td><td>68.6</td><td>66.4</td><td>5</td><td>5</td></tr><tr><td>Linear</td><td>68.1</td><td>66.0</td><td>4</td><td>5</td></tr><tr><td>Wasserstein</td><td>68.1</td><td>65.7</td><td>3</td><td>2</td></tr><tr><td>Large budget</td><td></td><td></td><td></td><td></td></tr><tr><td>Static i.i.d.</td><td>71.4</td><td>67.8</td><td>7</td><td>6</td></tr><tr><td>Linear</td><td>70.5</td><td>67.4</td><td>4</td><td>4</td></tr><tr><td>Wasserstein</td><td>70.6</td><td>67.4</td><td>1</td><td>2</td></tr></table>

Table 8: SFT summary for the three basic curricula on ARC, MMLU-STEM, and StrategyQA using Qwen3-1.7B. Each budget block contains 12 benchmark-by-axis contexts: six ARC axes, three MMLU-STEM axes, and three StrategyQA axes. "Hardest" denotes the last available level.

## D.2 Final Level Distributions, Bucket Profiles, and Remaining Efficiency Differences

Appendix D.1 showed that static i.i.d. is the strongest summary baseline in pretrained ST. We next ask whether the forward curricula still differ in exposure alločation and exposure-adjusted accuracy.

Setup. We study the medium budget and the four forward curricula: static i.i.d., linear, Wasserstein, and WARP. To keep the level view direct, we restrict to the 10 contexts whose native difficulty axes already have 3 or 5 levels, excluding only ARC human grade (6 levels) and MMLU-STEM human hardness (2 levels). Thus 3-level axes map directly to easy/middle/hard, and 5-level axes to easy/lower-mid/middle /upper-mid /hard. We report both five-position exposure/accuracy summaries and bucketed easy/middle/hard summaries with exposure-adjusted accuracy.

Figure 39 shows the five-position view. Exposure shifts much more than accuracy: linear pushes exposure toward the ends, while Wasserstein and WARP push it toward the middle, but the final accuracy curves remain close. WARP also stays close to fixed Wasserstein.

![](images/b404f47ea8b6acc9a2af634fb42f766fceabfb2a5279bd05f72830b5e9990c8a.jpg)  
Figure 39: Medium-budget SFT exposure and final accuracy on the 10 retained 3- or 5-level contexts. Rows show the three models; curves show the four forward curricula.

Figure 40 shows the bucketed view. Raw bucket accuracies remain compressed despite large exposure differences, so the exposure-adjusted view is more informative. Linear is strongest in the middle bucket, while Wasserstein and WARP are strongest on the hard bucket, but the gaps are modest and model-dependent.

![](images/0005c5db82a163ae09bb6ddade8dc795da889e3b4ee1ab7c51c103cb75878fde.jpg)  
Figure 40: Bucket-wise exposure, final accuracy, and exposure-adjusted accuracy for medium-budget SFT on the same 10 retained contexts. Rows show the three models; columns show exposure, accuracy, and exposure-adjusted accuracy under the four forward curricula.

## D.3 Easy-to-Hard Ordering Effect Is Weak in Pretrained SFT

Appendix D.2 showed that pretrained SFT largely flattens the forward-curriculum profiles.   
We next ask whether any ordering signal survives once exposure geometry is held fixed.

Setup. Figure 41 compares three Wasserstein-family controls across the small, medium, and large budgets: forward Wasserstein, reverse Wasserstein, and the exposure-matched static baseline. We keep all 12 benchmark-by-axis contexts and report eāsy, middle, and hard bucket means for each model.

![](images/68af75a917f8990633379a75b5cb24f0392993cc8c08563cdf9ace08af2e07ff.jpg)  
Figure 41: Ordering comparison within the Wasserstein family for SFT on ARC, MMLU-STEM, and StrategyQA. Rows show the three models; columns show the small, medium, and large budgets. "Static Matching" denotes the exposure-matched static baseline.

Forward ordering is no longer distinctly stronger. Across all nine model-by-budget blocks, forward Wasserstein never has the highest hard-bucket mean; the exposure-matched static baseline or the reverse-order control is always as good or better. Under pretrained SFT, changing the ordering within a fixed geometry no longer yields a robust easy-to-hard advantage.

Finding 8. Under pretrained SFT, most curriculum effects seen in the synthetic study weaken: static i.i.d. becomes the strongest summary baseline, bucket accuracies flatten, and easy-to-hard ordering no longer separates reliably. The clearest remaining difference is in exposure-adjusted data efficiency.

## D.4 Exploratory Analysis of When Easy-to-Hard Progression Helps

We next ask whether the remaining easy-to-hard wins cluster by benchmark or difficulty-axis type.

Setup. We analyze conditions in which Wasserstein or WARP attains the best mean among the four forward curricula, counting ties for all tied curricula under each metric. We then summarize these wins by benchmark and by difficulty-axis type.

![](images/6c6e29d5ab64e4966a9684a59fbc45c8b9d8e513cffb0f4432a9acff552d2a32.jpg)

![](images/610bc1fa2fa778d94d06260c31dbc576405e7dbf6c8f861bf290fca1984f4a8b.jpg)

(c) Joint view: overall  
![](images/1d5d029e5a0053e7fdc0b514956c0b8d73a7c65609782fbdd4519017e53901f8.jpg)

(d) Joint view: hardest  
![](images/160fb55252a514d359ae7a0a88ced605085ee56449efb9636eb8de449d01c7fc.jpg)  
Figure 42: Where natural easy-to-hard progression helps in SFT. Panel (a) summarizes wins by benchmark, panel (b) by axis category, and panels (c)–(d) show the joint benchmark × axis-category view for overall and hardest-level wins.

The remaining easy-to-hard wins are concentrated on ARC and human-annotated difficulty axes. Figure 42 shows only a weak concentration of wins. ARC has the most wins, especially on hardest-level accuracy, and human-annotated difficulty axes have the highest hardest-level win rate among the axis categories. However, neither pattern is consistent across both overall and hardest-level accuracy. Unlike Appendix C.4, the SFT setting does not yield a stable benchmark- or axis-level pattern.

## D.5 Controlled Synthetic Pretraining Analogue

To isolate the effect of pretraining, we vary the pretraining budget on the synthetic suite while holding the downstream tasks and evaluation protocol fixed.

Setup. In the pretraining stage, we train a shared 6-layer decoder on nine synthetic tasks from Appendix B.1. We use a uniform multitask mixture that splits each task's small-budget quota evenly across its reported difficulty axes and levels, then globally shuffles the resulting examples. We study three pretraining regimes, small, medium, and large, corresponding to 20%, 50%, and 100% of the sum of the task-wise small budgets. Downstream, we keep the earlier synthetic setup, restrict the sweep to the small budget, and remove pretraining-seen examples from the downstream train split.

Aggregate gaps narrow as pretraining grows. Table 9 shows the aggregate picture. As pretraining increases, the three basic curricula become harder to distinguish. The static baseline strengthens substantially: its mean hardest-level accuracy rises from 84.7% under small pretraining to 91.5% under medium pretraining and 93.7% under large pretraining. Even so, unlike the SFT setting, static does not become the uniformly strongest hardest-level baseline. Linear and Wasserstein continue to match or exceed it on many contexts.

<table><tr><td>Curriculum</td><td>Mean overall (%)</td><td>Mean hardest (%)</td><td>Overall wins</td><td>Hardest wins</td></tr><tr><td>Small pretraining</td><td></td><td></td><td></td><td></td></tr><tr><td>Static i.i.d.</td><td>91.3</td><td>84.7</td><td>10</td><td>6</td></tr><tr><td>Linear</td><td>91.2</td><td>88.1</td><td>13</td><td>17</td></tr><tr><td>Wasserstein</td><td>91.2</td><td>87.1</td><td>9</td><td>11</td></tr><tr><td>Medium pretraining</td><td></td><td></td><td></td><td></td></tr><tr><td>Static i.i.d.</td><td>95.0</td><td>91.5</td><td>13</td><td>13</td></tr><tr><td>Linear</td><td>95.0</td><td>92.9</td><td>17</td><td>18</td></tr><tr><td>Wasserstein</td><td>95.0</td><td>92.3</td><td>10</td><td>11</td></tr><tr><td>Large pretraining</td><td></td><td></td><td></td><td></td></tr><tr><td>Static i.i.d.</td><td>96.0</td><td>93.7</td><td>19</td><td>18</td></tr><tr><td>Linear</td><td>96.4</td><td>94.6</td><td>16</td><td>18</td></tr><tr><td>Wasserstein</td><td>96.5</td><td>94.6</td><td>14</td><td>14</td></tr></table>

Table 9: Synthetic-pretraining summary for the three basic curricula under a fixed small downstream budget. Each pretraining block contains the same 26 synthetic task-by-axis contexts. Here, "large pretraining" denotes the full multitask pretraining budget. The win columns count the number of contexts in which a curriculum attains the best mean under the corresponding metric; ties are counted for all tied curricula. "Hardest" denotes the last available level.

Level-wise profiles flatten as pretraining grows. Figure 43 shows the same narrowing in level-wise profiles. Under static i.i.d., the mean easy-hard gap falls from 10.9 points under small pretraining to 5.9 under medium pretraining and 4.8 under large pretraining; under Wasserstein, it shrinks from 5.5 to 4.5 to 3.3. This mirrors the SFT trend, though part of the narrowing here comes from easy-level saturation near 100%. Even so, the exposure-adjusted pattern survives: linear remains strongest on the middle bucket, while Wasserstein remains strongest on the hard bucket.

![](images/8c693609632a360745e13cdd1fb9783cceecb46fda76df2245411969cfa846b1.jpg)

![](images/03ceab70d05f82b20339feded26c3daea042bd214ee820e7ed2ecf918f9fe6e2.jpg)

![](images/beb9586f18f9bd1ea60055a83e60e454ae6e418956d758994201e7901aac8957.jpg)

![](images/ddb87f27f9f70c0b7bc5e7d68071d06e82d9492c01d5f47f847e156076e09e76.jpg)

![](images/e7877a26117be6193889c9e03e4805f85c1998f253d14fafe4a1c21b73cef046.jpg)

![](images/40207481b6600c955c061f13c8162bdf197c1f9077b48fd74cd5acccb1c2f649.jpg)

![](images/af285738a0dc2004997367fc32701cd0c2f4b567915666e79db524753c769a59.jpg)

![](images/9df0a5439628d608e009a24ab28f5c50821f18be6748de91cc13813642843566.jpg)

![](images/cd10c01511ca22388f240c1de992b47279c70cf52eae170c6262ca38bc2c56fc.jpg)  
Figure 43: Synthetic-pretraining analogue under a fixed small downstream budget. Rows vary pretraining amount; columns show level-wise exposure, final accuracy, and exposureadjusted accuracy for static i.i.d., linear, and Wasserstein.

Ordering weakens, but does not disappear. Figure 44 shows that the easy-to-hard advantage within the Wasserstein family shrinks steadily as pretraining increases. The Wassersteinreverse gap falls from 13.5 points under small pretraining to 2.3 under large pretraining, while the Wasserstein-exposure-matched-static gap falls from 3.9 to 0.2. The corresponding linear-reverse-linear gap similarly decreases from 8.5 to 1.6. Thus, increasing pretraining weakens ordering effects on this suite without eliminating them.

![](images/5a575b43d3b582d0dea7e9621b63dab3a5e401389cae01a58b1661ad230d1433.jpg)  
Figure 44: Ordering comparison within the Wasserstein family under a fixed small downstream budget. Panels vâry pretraining amount. "Static Matching" denotes the exposurematched static baseline.

Relation to pretrained SFT. The controlled synthetic analogue isolates pretraining as one factor behind the weaker curriculum effects observed in SFT. As pretraining increases, the differences among curricula shrink, static i.i.d. becomes stronger, and the easy-to-hard ordering advantage weakens. In the synthetic setting, these changes coincide with less remaining room for improvement, especially as easier levels approach saturation, suggesting one mechanism through which pretraining can reduce curriculum gains. Because saturation is less evident in the SFT experiments, other properties of natural post-training data may also matter.

Finding 9. Pretraining weakens downstream curriculum effects, helping explain why they are weaker in pretrained SFT; less remaining room for improvement may be one mechanism.