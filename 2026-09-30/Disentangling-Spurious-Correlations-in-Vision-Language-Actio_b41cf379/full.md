# Disentangling Spurious Correlations in Vision-Language-Action Models via Predicting Domain-Invariant Latent Lookahead

Junghyun Kim<sup>1,2∗</sup> Ngseo Kim<sup>2∗</sup> ChungWoo Lee<sup>2</sup> Seoyeon Lee<sup>2</sup> Woo-Jeong Baek<sup>5</sup> Adam Zhou<sup>1</sup> Chip Huyen<sup>1</sup> Jun-Ki Lee<sup>2†</sup> Gi-Cheon Kang<sup>3†</sup> Byoung-Tak Zhang<sup>2,4†</sup>

<sup>1</sup>OpenMind, San Francisco, CA, USA <sup>2</sup>Seoul National University, Seoul, Korea <sup>3</sup>Ajou University, Korea <sup>4</sup>Tommoro Robotics, Korea <sup>5</sup>Hyundai Motors, Korea

Abstract: Vision-Language-Action (VLA) models remain brittle under visual distribution shifts, often relying on spurious correlations tied to domain-specific factors rather than task-relevant structure. We propose Domain-Invariant Latent Lookahead (DILL), a representation-learning framework that mitigates shortcut learning in VLA policies. Our key idea is to supervise policies with domaininvariant future latents learned from domain-transformed trajectory data. A Task-Domain Encoder is trained with contrastive objectives and Gaussian disentanglement regularization to separate task-relevant structure from domain-specific visual variation. The learned encoder then provides future latents for VLA policy learning through lookahead prediction and domain disentanglement, encouraging the policy to focus on task-relevant structure rather than incidental visual factors. Counterfactual task–view evaluations show that DILL reduces shortcut reliance, while LIBERO-Plus evaluations demonstrate improved visual robustness, with 69.1% average success—11.4 percentage points above the strongest baseline. Real-world manipulation experiments further support DILL’s applicability beyond controlled simulation. Complementary latent-space diagnostics show that these behavioral gains are accompanied by representations that better preserve task-consistent structure while suppressing domain-specific variation. Our project page is available at https://dill-vla.github.io/.

Keywords: Vision-Language-Action Models, Shortcut Learning, Generalist Robot Policies, Predictive Future Latents, Domain Generalization

## 1 Introduction

A longstanding goal of robot learning is to build robots that generalize across a wide range of tasks and environments. Recent advances in Vision-Language-Action (VLA) models [1, 2, 3, 4] trained on large-scale robot datasets [5, 6] have established a promising path toward generalist robot policies by grounding actions in language and visual observations. Despite this progress, VLA models often remain brittle under distribution shift, particularly under incidental visual variation such as changes in viewpoint, background, lighting, or camera configuration [7, 8, 9], which should be irrelevant to task success. This brittleness undermines reliable out-of-distribution generalization and remains a major obstacle to practical deployment.

Such failures can be naturally understood through the lens of shortcut learning [10, 11], where policies rely on task-irrelevant visual cues that correlate with successful actions in the training distribution rather than the task-relevant semantics required for robust control. VLA models are particularly susceptible to shortcut learning for two reasons. First, robot datasets often exhibit limited withindataset diversity and are fragmented across collection sources, causing the same task or action to repeatedly co-occur with particular visual cues [11]. Second, VLA models often inherit representations from pretrained vision-language backbones [12, 13] that are optimized for visual understanding or image-language alignment rather than control, and may therefore preserve domain-specific visual information irrelevant to action generation. When mapped to actions, these representations can allow spurious visual correlations in training data to be absorbed into the policy’s decision rule [14], producing the task-substitution failure illustrated in Fig. 1: under counterfactual task–domain recomposition, the policy may execute the task associated with the observed domain.

![](images/e2d7a9d4957f7c9a02d683761f7a796550c5f55ed2863ab15d3b041a626776b9.jpg)  
Figure 1: Shortcut learning under task–domain confounding. When tasks and visual domains are spuriously correlated, a VLA may use domain-specific appearance as a shortcut for action selection. DILL reduces this failure by conditioning actions on a domain-invariant latent lookahead.

To mitigate such shortcuts, a VLA policy should learn representations that preserve task-relevant structure while discarding incidental visual variability. A natural learning signal for this purpose is future state prediction [15, 16]. Predicting how a task will evolve can encourage models to capture factors that determine future task states, such as object configuration, task progress, and actionconditioned scene changes. In this sense, future prediction provides a form of supervision that i more closely tied to control than to static visual appearance. However, future prediction alone is not sufficient. If the target is a raw future observation or an entangled visual latent, the model may still preserve domain-specific factors such as viewpoint or background that help predict future appearance but do not determine the correct action. Thus, the key question is not simply whether a VLA policy should predict the future, but what representation of the future it should predict. We argue that the predictive target should be domain-invariant: it should retain the future task structure needed for control while suppressing domain-specific factors that can act as shortcuts.

We propose Domain-Invariant Latent Lookahead (DILL), a predictive representation learning framework for robust VLA control. DILL first learns a Task-Domain Encoder from domaintransformed trajectory chunks using contrastive objectives over two pair types: task-positive pairs that preserve trajectory chunk across domain changes, and domain-positive pairs that share the same domain condition across different chunks. A Gaussian disentanglement loss further separates task and domain latents: task latents remain stable across changes in viewpoint, environment appearance, camera configuration, and visual degradation, while domain latents capture such domain-specific factors. During policy learning, the pretrained Task-Domain Encoder provides future latent supervision: the VLA policy predicts a lookahead latent aligned with the future task latent, while its current representation is regularized to be disentangled from the corresponding domain latent. The predicted lookahead latent and current representation jointly condition the action head, encouraging the policy to base its actions on task-relevant future structure rather than incidental visual factors.

The central effect of DILL is to reshape the information that the policy uses to choose actions: instead of allowing domain-specific visual cues to enter the action head as reliable proxies for the task, DILL conditions actions on a predicted future latent that is stable across domain changes and informative about the commanded behavior. This makes the learned action representation less tied to where or how a scene is observed, and more tied to the information needed to produce the correct action. Empirically, we evaluate this effect through complementary behavioral and representational analyses. Under counterfactual task–view compositions, DILL substantially reduces shortcut reliance and improves OOD task success, showing that policies are less likely to substitute the commanded task with the task spuriously associated with the observed view. On LIBERO-Plus visual perturbations, DILL achieves the highest average success rate among matched-input baselines and a substantially smaller average performance drop across camera, lighting, background, and sensornoise shifts. We further test DILL on a physical robot to assess its effectiveness beyond controlled simulation. Latent-space diagnostics further show that the final action-conditioning representation is organized more by task-consistent trajectory content than by shared visual appearance. Together, these results show that shortcut learning in VLA policies can be mitigated not only by increasing data diversity, but by explicitly shaping the predictive representation used for control.

## 2 Related Work

## 2.1 Spurious Correlations in Robot Learning

Spurious correlations [11] have emerged as an important challenge in robot learning, where poli cies may rely on incidental visual factors, such as camera viewpoint, background, and lighting, that correlate with successful actions in the training data [7, 8, 17, 9]. Prior work has addressed this issue from two perspectives: data-centric and representation-centric approaches. Data-centric approaches increase nuisance diversity through novel-view synthesis [18] or synthetic viewpoint augmentation [11]. Within representation-centric approaches, one line of work suppresses taskirrelevant factors by building invariance into visual representations, for example, by reducing sensitivity to viewpoint changes [19, 20, 21]. Another line of work injects task-relevant structure, such as spatial information [22]. Our work bridges these directions by using predictive future latents as task-relevant supervision while explicitly suppressing task-irrelevant visual factors through domaininvariant latent disentanglement.

## 2.2 World Models and Future Prediction in VLA Models

Recent VLA work has incorporated future prediction or world modeling signals to complement direct perception-to-action learning by anticipating the consequences of actions [15, 23, 24]. One line performs pixel-level imagination by generating future frames or subgoal observations before acting, as in the GR series, Ctrl-World, CoT-VLA, and SuSIE [25, 26, 27, 24, 28]. These approaches provide interpretable visual foresight, but they can be costly and prone to compounding errors. The second line moves future prediction into latent space by predicting or aligning future latent representations, including Video Prediction Policy, FLARE, VLA-JEPA, and FRAPPE [23, 15, 16, 29, 30, 31]. These methods avoid explicit pixel rollouts while encouraging representations that capture future structure useful for control. In contrast, DILL focuses on what the policy is asked to predict: rather than aligning to raw or entangled future latents, it learns future targets whose task-relevant structure is separated from domain-specific visual variation.

## 3 Method

## 3.1 Overview

Domain-Invariant Latent Lookahead (DILL) trains VLA policies with domain-invariant predictive supervision through a two-stage procedure. As shown in Fig. 2, we first learn a Task-Domain Encoder using domain-transformed trajectory data. The task encoder produces task latents that are encouraged to remain invariant across domain changes, while the domain encoder produces domain latents that capture domain-specific visual variation. Both latents are learned using contrastive objectives defined by task-positive and domain-positive pairs, together with a Gaussian disentanglement loss that encourages the task and domain latents to be separated.

We then use the learned Task-Domain Encoder to supervise VLA policy learning. For each training trajectory, the encoder provides future task and domain latents as supervisory signals. From the current observation and language instruction, the policy predicts a lookahead latent and a current representation: the former is aligned with the future task latent, while the latter is regularized to be disentangled from the domain latent. These two policy latents are fed to the action head for control, encouraging the policy to combine a predicted future task representation with a current representation discouraged from carrying domain-specific visual information. As a result, the policy is trained to rely on task-relevant future structure rather than shortcut-inducing visual cues. At test time, the policy requires only the current observation and instruction.

![](images/6a8efbdfcc8b01f5b37aff37dfa7d495c8af8ac9a6ea29b219001f29f1852a27.jpg)  
Figure 2: Overview of Domain-Invariant Latent Lookahead. We first learn a Task-Domain Encoder that maps observation chunks into task and domain latents. During policy learning, a Lookahead Predictor predicts a future task latent from the current observation and instruction, while a Current Representation Head produces a domain-disentangled current representation. The two policy latents condition the action head for control.

## 3.2 Task-Domain Latent Disentanglement

The Task-Domain Encoder learns task and domain latents from video observations: task latents are encouraged to preserve task-relevant structure, while domain latents capture domain-specific visual factors. Let $\mathbf { o } _ { t } ^ { i } = \left( o _ { t } ^ { i } , \ldots , o _ { t + T _ { v } - 1 } ^ { i } \right)$ denote an observation chunk from trajectory i, and let $\mathcal { T } _ { \eta }$ denote a domain transformation with condition η, where η specifies the full domain condition. We construct task-positive and domain-positive pairs:

$$
\underbrace { \big ( \mathcal T _ { \eta _ { 1 } } ( \mathbf o _ { t } ^ { i } ) , \mathcal T _ { \eta _ { 2 } } ( \mathbf o _ { t } ^ { i } ) \big ) } _ { \mathrm { t a s k - p o s i t i v e } } , \quad \eta _ { 1 } \ne \eta _ { 2 } , \qquad \underbrace { \big ( \mathcal T _ { \eta } ( \mathbf o _ { t } ^ { i } ) , \mathcal T _ { \eta } ( \mathbf o _ { t ^ { \prime } } ^ { j } ) \big ) } _ { \mathrm { d o m a i n - p o s i t i v e } } , \quad ( i , t ) \ne ( j , t ^ { \prime } ) .\tag{1}
$$

Task-positive pairs preserve the same observation chunk while changing the domain condition, whereas domain-positive pairs share the same domain condition across different observation chunks. The transformation families include viewpoint changes, environment appearance variation, camera heterogeneity, and visual degradation. See Appendix A.1 for details.

Let $x = \mathcal { T } _ { \eta } ( \mathbf { o } )$ denote a transformed observation chunk. The Task-Domain Encoder consists of two separate video encoders: a task encoder and a domain encoder. Each encoder maps the input observation chunk to a pooled latent vector:

$$
z ^ { \mathrm { t a s k } } = E _ { \psi } ^ { \mathrm { t a s k } } ( x ) , \qquad z ^ { \mathrm { d o m } } = E _ { \xi } ^ { \mathrm { d o m } } ( x ) .\tag{2}
$$

Implementation details are provided in Appendix A.4.

Both encoders are trained with the same InfoNCE form but with different positive pair constructions. For a minibatch of positive pairs $\{ ( x _ { i } , x _ { i } ^ { + } ) \} _ { i = 1 } ^ { B }$ , let $z _ { i }$ and $z _ { i } ^ { + }$ denote the pooled latent vectors from the corresponding encoder:

$$
\mathcal { L } _ { \mathrm { N C E } } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \exp ( \sin ( z _ { i } , z _ { i } ^ { + } ) / \tau ) } { \sum _ { k = 1 } ^ { B } \exp ( \sin ( z _ { i } , z _ { k } ^ { + } ) / \tau ) }\tag{3}
$$

where sim(·, ·) denotes cosine similarity and $\tau$ is a temperature. Applying Eq. 3 to the task encoder with task-positive pairs and to the domain encoder with domain-positive pairs gives $\mathcal { L } _ { \mathrm { t a s k } }$ and ${ \mathcal { L } } _ { \mathrm { d o m } }$ respectively.

To further disentangle the task and domain latents, we introduce a Gaussian disentanglement loss by applying SIGReg [32] to their concatenation. For each minibatch, we form a joint task-domain latent:

$$
\begin{array} { r } { u _ { i } ^ { \mathrm { t d } } = [ z _ { i } ^ { \mathrm { t a s k } } ; z _ { i } ^ { \mathrm { d o m } } ] , \qquad \mathcal { L } _ { \mathrm { d i s } } ^ { \mathrm { t d } } = \mathcal { R } _ { \mathrm { d i s } } \left( \{ u _ { i } ^ { \mathrm { t d } } \} _ { i = 1 } ^ { B } \right) , } \end{array}\tag{4}
$$

where $[ \cdot ; \cdot ]$ denotes concatenation. Unlike applying distributional regularization to each latent separately, this joint regularizer acts on the concatenated task-domain latent. Under a Gaussian approximation, matching the joint latent to an isotropic Gaussian discourages cross-covariance between the task and domain latents, thereby encouraging approximate disentanglement. Appendix A.2 provides the corresponding argument.

The task-domain disentanglement objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { T D D } } = \lambda _ { \mathrm { t a s k } } \mathcal { L } _ { \mathrm { t a s k } } + \lambda _ { \mathrm { d o m } } \mathcal { L } _ { \mathrm { d o m } } + \lambda _ { \mathrm { d i s } } ^ { \mathrm { t d } } \mathcal { L } _ { \mathrm { d i s } } ^ { \mathrm { t d } } . } \end{array}\tag{5}
$$

Task-domain encoder pretraining. We pretrain the task and domain encoders on large-scale domain-transformed trajectory chunks from ManiSkill [33], MimicGen [34], and a subset of OXE [5]. We refer to these data as the source trajectory collection. The pretrained encoders define the task and domain latent spaces used later to provide future supervision for VLA policy learning.

## 3.3 VLA Policy Learning with Domain-Invariant Latent Lookahead

During VLA policy training, the pretrained Task-Domain Encoder provides future latent supervision. Given the current observation $o _ { t }$ and language instruction $\ell ,$ the vision-language backbone [13] produces image-language tokens:

$$
H _ { t } ^ { \mathrm { v l m } } = M _ { \theta } ( o _ { t } , \ell ) .\tag{6}
$$

Two policy heads map these tokens to a predicted lookahead latent and a current representation:

$$
\begin{array} { r } { \hat { z } _ { t } ^ { \mathrm { l o o k } } = P _ { \theta } ^ { \mathrm { l o o k } } ( H _ { t } ^ { \mathrm { v l m } } ) , \qquad z _ { t } ^ { \mathrm { c u r r } } = P _ { \theta } ^ { \mathrm { c u r r } } ( H _ { t } ^ { \mathrm { v l m } } ) . } \end{array}\tag{7}
$$

Let $\mathbf { o } _ { t } ^ { + } ~ = ~ ( o _ { t + 1 } , \dots , o _ { t + T _ { v } } )$ denote the future observation chunk. Using the pretrained Task-Domain Encoder, we extract a future task target and its domain latent:

$$
z _ { t } ^ { \mathrm { t a s k , \star } } = E _ { \bar { \psi } } ^ { \mathrm { t a s k } } \left( { \bf o } _ { t } ^ { + } \right) , \qquad z _ { t } ^ { \mathrm { d o m , \star } } = E _ { \bar { \xi } } ^ { \mathrm { d o m } } \left( { \bf o } _ { t } ^ { + } \right) .\tag{8}
$$

The lookahead latent is aligned with the future task target using the InfoNCE objective in Eq. 3. For each minibatch, we instantiate Eq. 3 by using the predicted lookahead latents as anchors, $z _ { i } = \hat { z } _ { i } ^ { \mathrm { l o o k } }$ and the stop-gradient future task latents as positives, $z _ { i } ^ { + } = \mathrm { s g } ( z _ { i } ^ { \mathrm { t a s k , \star } } )$ . We denote this loss by $\mathcal { L } _ { \mathrm { l o o k } } .$

We also disentangle the current representation from the domain latent extracted by the domain encoder:

$$
\begin{array} { r } { u _ { i } ^ { \mathrm { c u r r } } = \left[ z _ { i } ^ { \mathrm { c u r r } } ; \mathrm { s g } \left( z _ { i } ^ { \mathrm { d o m } , \star } \right) \right] , \qquad \mathcal { L } _ { \mathrm { d i s } } ^ { \mathrm { c u r r } } = \mathcal { R } _ { \mathrm { d i s } } \left( \{ u _ { i } ^ { \mathrm { c u r r } } \} _ { i = 1 } ^ { B } \right) . } \end{array}\tag{9}
$$

Since the current and future chunks come from the same trajectory and domain condition, $z _ { t } ^ { \mathrm { d o m , \star } }$ provides the corresponding domain latent. The Gaussian disentanglement loss encourages $z _ { t } ^ { \mathrm { c u r r } }$ to be separated from the domain latent while preserving information useful for action prediction.

The lookahead latent provides predictive task information about the future observation chunk, while the current representation provides action-relevant information from the present input after domain disentanglement:

$$
\begin{array} { r } { \hat { a } _ { t : t + K _ { a } - 1 } = A _ { \phi } \left( \left[ \hat { z } _ { t } ^ { \mathrm { l o o k } } ; z _ { t } ^ { \mathrm { c u r r } } \right] \right) . } \end{array}\tag{10}
$$

We supervise the action chunk with behavior cloning:

$$
\mathcal { L } _ { \mathrm { a c t } } = \frac { 1 } { K _ { a } } \sum _ { k = 0 } ^ { K _ { a } - 1 } \| \hat { a } _ { t + k } - a _ { t + k } \| _ { 1 } .\tag{11}
$$

The final policy objective is

$$
\mathcal { L } _ { \mathrm { p o l i c y } } = \lambda _ { \mathrm { a c t } } \mathcal { L } _ { \mathrm { a c t } } + \lambda _ { \mathrm { l o o k } } \mathcal { L } _ { \mathrm { l o o k } } + \lambda _ { \mathrm { d i s } } ^ { \mathrm { c u r r } } \mathcal { L } _ { \mathrm { d i s } } ^ { \mathrm { c u r r } } .\tag{12}
$$

![](images/90932bd9309c9730cbb216a60cc90b5cd44b2c75f34e38985e09925c14320ba0.jpg)

Policy pretraining and adaptation. We pretrain the policy-side modules on action-labeled trajectories from the source trajectory collection. For downstream settings such as LIBERO [35] or real-world robot data, we fine-tune the policy with the same lookahead, disentanglement, and action losses. At deployment, the policy receives only the current observation and instruction. The Task-Domain Encoder is not used at inference.

## 4 Experiments

We evaluate the central claim of this paper: domain-invariant latent lookahead mitigates shortcut learning in vision-language-action policies. We combine controlled simulation, where task–domain correlations and visual shifts can be systematically manipulated, with real-world manipulation experiments. Specifically, we ask:

1. Does our approach reduce shortcut reliance under counterfactual task–view compositions?

2. Does our approach improve robustness under visual distribution shifts?

3. Do the learned latents successfully separate task structure from task-irrelevant domain factors?

## 4.1 LIBERO Shortcut Diagnostic Under Counterfactual Task–View Compositions

We first test shortcut mitigation in a controlled LIBERO diagnostic [11]. This diagnostic intentionally creates a spurious correlation between task identity and camera viewpoint during training, then breaks this correlation at test time to measure whether a policy follows the commanded task or the task spuriously associated with the observed view.

Protocol. During training, two task groups, Task-L and Task-R, are observed only from left- and right-view ranges, respectively. At test time, we swap these associations: Task-R is evaluated at the left-view boundary and Task-L at the right-view boundary. Details are in Appendix B.1.

Metrics. We report OOD success and shortcut degree. OOD success measures whether the commanded task is completed under the counterfactual view. Shortcut degree measures whether the policy instead executes the task group spuriously associated with the observed view during training. Lower shortcut degree indicates less shortcut reliance.

Compared methods. Base VLA is a behavior-cloning baseline built from DILL’s underlying VLA. It maps the current observation and instruction to actions and is trained only with action supervision, without source augmentation, latent lookahead, or task– domain supervision. We then compare two lookahead variants. Entangled latent lookahead (Entangled LA)

Figure 3: LIBERO shortcut diagnostic.

predicts future representations from the original video encoder, which does not separate task and domain factors. DILL w/o CH instead predicts disentangled future task latents, but omits the disentangled current head (CH). These two variants share policy architecture and augmented source data, isolating the choice of predictive target. Full DILL additionally disentangles the current representation used for action prediction.

Results. Figure 3 shows that Base VLA fails to complete the commanded task under swapped views (zero OOD success), while frequently executing the task associated with the observed view (shortcut degree 0.73). Thus, its failures reflect task substitution, not just difficulty acting from an unfamiliar view. MiniVLA and $\pi _ { 0 }$ exhibit the same pattern in their respective evaluations. DILL raises OOD success to 0.58 and reduces shortcut degree to 0.05, recovering commanded behavior while largely avoiding view-induced task substitution.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Original</td><td colspan="4">Visual perturbations</td><td rowspan="2">Average</td></tr><tr><td>Camera</td><td>Light</td><td>BG</td><td>Noise</td></tr><tr><td rowspan="2">OpenVLA [3]</td><td rowspan="2">76.5</td><td>0.8</td><td>8.1</td><td>34.8</td><td>15.2</td><td>14.7</td></tr><tr><td>↓75.7</td><td>↓68.4</td><td>↓41.7</td><td>↓61.3</td><td>↓61.8</td></tr><tr><td rowspan="2">WorldVLA [36]</td><td rowspan="2">79.1</td><td>0.1</td><td>43.7</td><td>17.1</td><td>10.9</td><td>18.0</td></tr><tr><td>↓79.0</td><td>↓35.4</td><td>↓62.0</td><td>↓68.2</td><td>↓61.2</td></tr><tr><td rowspan="2">UniVLA [37]</td><td rowspan="2">95.5</td><td>1.8</td><td>69.0</td><td>81.0</td><td>21.2</td><td>43.3</td></tr><tr><td>↓93.7</td><td>↓26.5</td><td>↓14.5</td><td>↓74.3</td><td>↓52.3</td></tr><tr><td rowspan="2">NORA [38]</td><td rowspan="2">87.9</td><td>2.2</td><td>45.7</td><td>58.6</td><td>12.8</td><td>29.8</td></tr><tr><td>↓85.7</td><td>↓42.2</td><td>↓29.3</td><td>↓75.1</td><td>↓58.1</td></tr><tr><td rowspan="2">OpenVLA-OFT [39]</td><td rowspan="2">95.3</td><td>10.4</td><td>76.8</td><td>93.6</td><td>49.9</td><td>57.7</td></tr><tr><td>↓84.9</td><td>↓18.5</td><td>↓1.7</td><td>↓45.4</td><td>↓37.6</td></tr><tr><td rowspan="2">Base VLA + SA</td><td rowspan="2">82.0</td><td>39.3</td><td>56.8</td><td>50.5</td><td>6.3</td><td>38.2</td></tr><tr><td>↓42.7</td><td>↓25.2</td><td>↓31.5</td><td>↓75.7</td><td>↓43.8</td></tr><tr><td rowspan="2">DILL (Ours)</td><td rowspan="2">81.6</td><td>68.4</td><td>69.4</td><td>70.0</td><td>68.7</td><td>69.1</td></tr><tr><td>↓13.2</td><td>↓12.2</td><td>↓11.6</td><td>↓12.9</td><td>↓12.5</td></tr></table>

Table 1: Zero-shot robustness evaluation on visual perturbations in LIBERO-Plus [7]. For each model, the first row reports success (%) and the second its drop from Original in percentage points. Average is the unweighted mean across the four visual categories. The bottom block matches policy architecture and augmented source data (SA: source augmentation).

The target ablation shows why future prediction alone is insufficient in this setting. Entangled LA still has zero OOD success and a shortcut degree of 0.65. Replacing its predictive targets with disentangled future task latents (DILL w/o CH) raises success to 0.44 and reduces shortcut degree to 0.06. With architecture and augmentation exposure matched, this contrast supports the importance of domain-invariant targets beyond future prediction alone. Disentangling the current representation in full DILL further raises success from 0.44 to 0.58, with little change in shortcut degree (0.06 to 0.05). Its additional benefit is therefore better execution of the commanded task, beyond the shortcut reduction already achieved by invariant lookahead targets. Additional component ablations appear in Appendix B.2.

## 4.2 LIBERO-Plus Zero-Shot Robustness Evaluation Under Visual Distribution Shifts

We next evaluate whether shortcut mitigation translates into stronger robustness under visual distribution shifts. We use the visual perturbation categories in LIBERO-Plus [7] as a controlled zero-shot evaluation: no LIBERO-Plus perturbed images are used for training. All models are adapted on the original LIBERO training split.

Compared methods. To ensure a controlled comparison, the main table includes only methods evaluated with third-person RGB observations, excluding wrist-camera images. Under this matchedinput protocol, we compare against OpenVLA [3], OpenVLA-OFT [39], WorldVLA [36], Uni-VLA [37], and NORA [38].

Results. DILL achieves the highest average success across the four visual perturbation categories (69.1%; Table 1), exceeding the strongest external baseline, OpenVLA-OFT, by 11.4 percentage points. This advantage does not come from higher original LIBERO performance: OpenVLA-OFT and UniVLA score higher without perturbations, but their average drops under visual shifts are 37.6 and 52.3 points, respectively, compared with 12.5 for DILL. DILL’s largest advantages are under camera changes (68.4% success) and sensor noise (68.7%), where all external baselines remain below 11% and 50%, respectively. OpenVLA-OFT is stronger on background and lighting changes, but DILL maintains 68.4–70.0% success across all four categories, indicating more consistent robustness across visual shifts. To test whether augmentation exposure alone accounts for this robust ness, we train Base VLA with DILL’s augmented source data while retaining the behavior-cloning objective (Base VLA + SA). This control nearly matches DILL on original LIBERO (82.0% versus 81.6%), yet its average perturbed success is much lower (38.2% versus 69.1%). DILL outperforms this control in all four categories, indicating that augmentation exposure alone does not explain its robustness gains. The full LIBERO-Plus breakdown is in Appendix B.3.

![](images/5317130f050834d94b57da94643553da6e12d16d78d240885965ab03707847dc.jpg)

![](images/02afb9742fd2d780156b01c104e7bab06ecb3946b4707667fc59aacb613a9ac7.jpg)

![](images/0136166590030b21137b0ba2f633b8b8b91abb50ac1a97d89a63c3e0ccb96f2a.jpg)  
Figure 4: Pairwise similarity diagnostics for learned latents. We compare cosine-similarity distributions for task pairs and domain pairs constructed from unseen task–domain combinations. From left to right, the panels show the task encoder, domain encoder, and final VLA latent. The task encoder should group task pairs, the domain encoder should group domain pairs, and the final VLA latent should preserve task-consistent structure while suppressing domain-specific variation.

## 4.3 Pairwise Diagnostics of Learned Latents

We also analyze whether the learned latent spaces separate task-relevant structure from domainspecific visual variation. We evaluate three representations: the task encoder output, the domain encoder output, and the final VLA latent provided to the action head. For the final VLA latent, we use the concatenated action-conditioning representation, i.e., the current policy representation together with the predicted lookahead latent.

Protocol. We compute cosine similarity between normalized latents for two types of unseen task and domain pairs. Task pairs share the same underlying trajectory content but differ in visual domain or augmentation. Domain pairs share the same visual domain or augmentation pattern but differ in trajectory content. These pair combinations are not observed during training, so the diagnostic tests whether the learned representations generalize beyond memorized pairings. A task-centric representation should assign higher similarity to task pairs than to domain pairs, while a domain-centric representation should show the opposite behavior. Additional details are provided in Appendix C.

Results. Figure 4 shows that the learned factorization behaves as intended. The task encoder assigns high similarity to task pairs and substantially lower similarity to domain pairs, with mean similarities of 0.88 and 0.35, respectively. Conversely, the domain encoder assigns high similarity to domain pairs and lower similarity to task pairs, with mean similarities of 0.94 and 0.34, indicating that the task and domain encoders capture complementary factors rather than collapsing to the same representation. The final VLA latent also remains strongly task-centric: it assigns high similarity to task pairs $( \mu = 0 . 9 4 )$ while keeping domain pairs noticeably lower $( \mu = 0 . 5 5 )$ . Since this is the representation directly provided to the action head, the result suggests that the policy is conditioned on latent features that are stable across domain changes but still discriminative across different trajectory content. This supports the mechanism behind the robustness gains in Table 1: domain-invariant latent lookahead encourages the policy to act on task-consistent structure rather than domain-specific visual cues.

## 4.4 Real-World Experiments

To test whether DILL’s benefits extend to the physical world, we use two tasks for shortcut diagnosis and three for visual robustness. The shortcut diagnostic swaps target-color/viewpoint associations learned during training to test instruction following when visual cues become misleading. We compare against Base VLA without task–domain supervision and DILL-Current, which replaces the future task target with the current task latent. Policies are fine-tuned on real-world demonstrations, while the source-pretrained Task-Domain Encoder remains frozen, without adaptation or paired view training on these tasks. Appendix D details both protocols, illustrated in Figures 9 and 10.

DILL raises counterfactual command-following success from Base VLA’s 29.7% to 80.2% and reduces observed shortcut degree from 47.2% to zero (Table 2). DILL-Current also attains 83.3% success with no observed shortcut execution, indicating that task-invariant supervision can suppress this failure mode without future prediction.

<table><tr><td rowspan="2">Model</td><td colspan="2">Shortcut diagnosis</td><td colspan="3">Robustness</td></tr><tr><td>CTF↑</td><td>SD↓</td><td>No pert. ↑</td><td>Predefined ↑</td><td>New↑</td></tr><tr><td>Base VLA</td><td> $2 9 . 7 \pm 5 . 4$ </td><td> $4 7 . 2 \pm 4 . 8$ </td><td> $8 4 . 1 \pm 5 . 5$ </td><td> $4 8 . 3 \pm 5 . 9$ </td><td> $6 4 . 0 { \pm } 6 . 4 $ </td></tr><tr><td>DILL-Current</td><td> ${ \bf 8 3 . 3 \pm 4 . 8 }$ </td><td> ${ \bf 0 . 0 \pm 0 . 0 }$ </td><td> $\mathbf { 8 8 . 9 \pm 2 . 7 }$ </td><td> $7 8 . 2 \pm 7 . 6$ </td><td> $7 7 . 2 \pm 5 . 1$ </td></tr><tr><td>DILL</td><td> $8 0 . 2 \pm 5 . 5$ </td><td> ${ \bf 0 . 0 \pm 0 . 0 }$ </td><td> ${ \bf 8 8 . 9 \pm 2 . 7 }$ </td><td> ${ \bf 8 3 . 8 \pm 4 . 2 }$ </td><td> ${ \bf 8 2 . 0 \pm 4 . 0 }$ </td></tr></table>

Table 2: Real-world shortcut diagnosis and robustness (%). CTF measures command-following under swapped target-color/viewpoint pairings; SD is shortcut degree (Section 4.1). Predefined perturbations are held-out instances of encoder-training transformation families; new perturbation (cast shadows, dynamic backgrounds, and foreground clutter) are absent from pair construction.

Future targets provide an additional benefit under visual shifts. At identical 88.9% unperturbed success, DILL exceeds DILL-Current by 5.6 and 4.8 percentage points in mean success under predefined and new perturbations, respectively. The latter are absent from encoder pair construction, indicating that lookahead’s robustness gains extend beyond the transformation families used to learn invariance.

## 5 Limitations

DILL is designed to reduce shortcut reliance induced by visual-domain factors. In our experiments, these factors include changes in viewpoint, scene appearance, camera configuration, and visual degradation. While this captures a common source of brittleness in VLA policies, it does not cover all possible spurious correlations. For example, biases in initial states, object layouts, language templates, task frequencies, or demonstrator styles may also influence action prediction without being part of the intended task semantics. Addressing such non-visual shortcuts would require defining additional nuisance factors, obtaining corresponding paired interventions, or developing objectives that can discover them more automatically.

Another important consideration is the trade-off between robustness and in-distribution performance. In the training distribution, domain-dependent cues can be highly predictive of demonstrated actions, even when they are not causally tied to the intended task semantics. Moreover, the same visual factor may play different roles across contexts: a background pattern may be a spurious cue in one setting, but part of the task-relevant scene configuration in another. Overly strong invariance may therefore suppress information that is useful for control in some situations, reducing peak indistribution performance even as it improves robustness when those cues become misleading. This suggests that robustness should not be pursued by uniformly removing domain-specific information in all cases. Instead, future work should explore adaptive objectives that preserve action-relevant factors and suppress action-irrelevant shortcuts in a context-dependent manner.

## 6 Conclusion

A central obstacle to robust VLA control is that policies can treat domain-specific appearance cues as action-relevant when they are only reliable within the training distribution. Domain-Invariant Latent Lookahead addresses this shortcut learning problem by supervising VLA representations with future latents in a space where domain-specific visual variation has been separated from taskrelevant structure. This encourages the policy to rely less on incidental appearance factors such as background, viewpoint, or lighting, and more on action-relevant scene information that remains stable across domains. Controlled simulation studies show reduced shortcut reliance and improved robustness to visual shifts, while experiments on a physical robot provide evidence of these benefits beyond simulation. Our results suggest that improving robustness in VLA models is not only a matter of scaling data or architectures, but also of shaping what the policy learns.

## Acknowledgments

This work was partly supported by grants funded by the Korean government through IITP (RS-2022-II220951-LBA/5%, RS-2022-II220953-PICA/5%, RS-2026-25553157-MIACC/10%, IITP-2026-RS-2023-00255968/10%, RS-2026-25617480/10%, and RS-2026-25552043/10%), NRF (RS-2024-00353991-SPARC/10%, RS-2023-00274280-HEI/10%, and RS-2026-25518808/10%), KEIT (RS-2025-25453780/10%), and KIAT (RS-2025-25460896/10%).

## References

[1] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, X. Chen, K. Choromanski, T. Ding, D. Driess, A. Dubey, C. Finn, et al. Rt-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023.

[2] Octo Model Team, D. Ghosh, H. Walke, K. Pertsch, K. Black, O. Mees, S. Dasari, J. Hejna, C. Xu, J. Luo, T. Kreiman, Y. Tan, L. Y. Chen, P. Sanketi, Q. Vuong, T. Xiao, D. Sadigh, C. Finn, and S. Levine. Octo: An open-source generalist robot policy. In Proceedings of Robotics: Science and Systems, Delft, Netherlands, 2024.

[3] M. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. Foster, G. Lam, P. Sanketi, Q. Vuong, T. Kollar, B. Burchfiel, R. Tedrake, D. Sadigh, S. Levine, P. Liang, and C. Finn. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024.

[4] K. Black, N. Brown, D. Driess, A. Esmail, M. Equi, C. Finn, N. Fusai, L. Groom, K. Hausman, B. Ichter, et al. π : A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

[5] O. X.-E. Collaboration, A. O’Neill, A. Rehman, A. Gupta, A. Maddukuri, A. Gupta, A. Padalkar, et al. Open X-Embodiment: Robotic learning datasets and RT-X models. https://arxiv.org/abs/2310.08864, 2023.

[6] A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.

[7] S. Fei, S. Wang, J. Shi, Z. Dai, J. Cai, P. Qian, L. Ji, X. He, S. Zhang, Z. Fei, et al. Libero-plus: In-depth robustness analysis of vision-language-action models. arXiv preprint arXiv:2510.13626, 2025.

[8] B. Zhang, J. Li, J. Shen, Y. Cai, Y. Zhang, Y. Chen, J. Dai, J. Ji, and Y. Yang. Vla-arena: An open-source framework for benchmarking vision-language-action models. arXiv preprint arXiv:2512.22539, 2025.

[9] G. Wang, C. Zhang, Q. Liu, J. Zhang, J. Cai, J. Liu, and X. Liu. Libero-x: Robustness litmus for vision-language-action models. arXiv preprint arXiv:2602.06556, 2026.

[10] R. Geirhos, J.-H. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge, and F. A. Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, 2020.

[11] Y. Xing, X. Luo, J. Xie, L. Gao, H. Shen, and J. Song. Shortcut learning in generalist robot policies: The role of dataset diversity and fragmentation. arXiv preprint arXiv:2508.06426, 2025.

[12] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.

[13] S. Bai, K. Chen, X. Liu, J. Wang, W. Ge, S. Song, K. Dang, P. Wang, S. Wang, J. Tang, H. Zhong, Y. Zhu, M. Yang, Z. Li, J. Wan, P. Wang, W. Ding, Z. Fu, Y. Xu, J. Ye, X. Zhang, T. Xie, Z. Cheng, H. Zhang, Z. Yang, H. Xu, and J. Lin. Qwen2.5-vl technical report, 2025. URL https://arxiv.org/abs/2502.13923.

[14] P. De Haan, D. Jayaraman, and S. Levine. Causal confusion in imitation learning. Advances in neural information processing systems, 32, 2019.

[15] R. Zheng, J. Wang, S. Reed, J. Bjorck, Y. Fang, F. Hu, J. Jang, K. Kundalia, Z. Lin, L. Magne, et al. Flare: Robot learning with implicit world modeling. arXiv preprint arXiv:2505.15659, 2025.

[16] J. Sun, W. Zhang, Z. Qi, S. Ren, Z. Liu, H. Zhu, G. Sun, X. Jin, and Z. Chen. Vla-jepa: Enhancing vision-language-action model with latent world model. arXiv preprint arXiv:2602.10098, 2026.

[17] X. Zhou, Y. Xu, G. Tie, Y. Chen, G. Zhang, D. Chu, P. Zhou, and L. Sun. Libero-pro: Towards robust and fair evaluation of vision-language-action models beyond memorization. arXiv preprint arXiv:2510.03827, 2025.

[18] S. Tian, B. Wulfe, K. Sargent, K. Liu, S. Zakharov, V. Guizilini, and J. Wu. View-invariant policy learning via zero-shot novel view synthesis. arXiv preprint arXiv:2409.03685, 2024.

[19] F. Liu, F. Yan, L. Zheng, C. Feng, Y. Huang, and L. Ma. Robouniview: Visual-language model with unified view representation for robotic manipulation. arXiv preprint arXiv:2406.18977, 2024.

[20] Y. Seo, J. Kim, S. James, K. Lee, J. Shin, and P. Abbeel. Multi-view masked world models for visual robotic manipulation. In International Conference on Machine Learning, pages 30613–30632. PMLR, 2023.

[21] J.-C. Pang, N. Tang, K. Li, Y. Tang, X.-Q. Cai, Z.-Y. Zhang, G. Niu, M. Sugiyama, and Y. Yu. Learning view-invariant world models for visual robotic manipulation. In The Thirteenth International Conference on Learning Representations, 2025.

[22] J. Zhang, S. Wu, X. Luo, H. Wu, L. Gao, H. T. Shen, and J. Song. Inspire: Vision-languageaction models with intrinsic spatial reasoning. arXiv preprint arXiv:2505.13888, 2025.

[23] Y. Hu, Y. Guo, P. Wang, X. Chen, Y.-J. Wang, J. Zhang, K. Sreenath, C. Lu, and J. Chen. Video prediction policy: A generalist robot policy with predictive visual representations. arXiv preprint arXiv:2412.14803, 2024.

[24] Q. Zhao, Y. Lu, M. J. Kim, Z. Fu, Z. Zhang, Y. Wu, Z. Li, Q. Ma, S. Han, C. Finn, et al. Cotvla: Visual chain-of-thought reasoning for vision-language-action models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 1702–1713, 2025.

[25] H. Wu, Y. Jing, C. Cheang, G. Chen, J. Xu, X. Li, M. Liu, H. Li, and T. Kong. Unleashing large-scale video generative pre-training for visual robot manipulation. arXiv preprint arXiv:2312.13139, 2023.

[26] C.-L. Cheang, G. Chen, Y. Jing, T. Kong, H. Li, Y. Li, Y. Liu, H. Wu, J. Xu, Y. Yang, et al. Gr-2: A generative video-language-action model with web-scale knowledge for robot manipulation. arXiv preprint arXiv:2410.06158, 2024.

[27] Y. Guo, L. X. Shi, J. Chen, and C. Finn. Ctrl-world: A controllable generative world model for robot manipulation. arXiv preprint arXiv:2510.10125, 2025.

[28] K. Black, M. Nakamoto, P. Atreya, H. Walke, C. Finn, A. Kumar, and S. Levine. Zeroshot robotic manipulation with pretrained image-editing diffusion models. arXiv preprint arXiv:2310.10639, 2023.

[29] H. Zhao, J. Wang, W. Song, S. Chen, Y. Liu, Y. Wang, H. Li, and D. Wang. Frappe: Infusing world modeling into generalist policies via multiple future representation alignment. arXiv preprint arXiv:2602.17259, 2026.

[30] M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat, Y. LeCun, and N. Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15619–15629, 2023.

[31] M. Assran, A. Bardes, D. Fan, Q. Garrido, R. Howes, M. Muckley, A. Rizvi, C. Roberts, K. Sinha, A. Zholus, et al. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.

[32] R. Balestriero and Y. LeCun. Lejepa: Provable and scalable self-supervised learning without the heuristics, 2025. URL https://arxiv.org/abs/2511.08544.

[33] S. Tao, F. Xiang, A. Shukla, Y. Qin, X. Hinrichsen, X. Yuan, C. Bao, X. Lin, Y. Liu, T. kai Chan, Y. Gao, X. Li, T. Mu, N. Xiao, A. Gurha, V. N. Rajesh, Y. W. Choi, Y.-R. Chen, Z. Huang, R. Calandra, R. Chen, S. Luo, and H. Su. Maniskill3: Gpu parallelized robotics simulation and rendering for generalizable embodied ai. Robotics: Science and Systems, 2025.

[34] A. Mandlekar, S. Nasiriany, B. Wen, I. Akinola, Y. Narang, L. Fan, Y. Zhu, and D. Fox. Mimicgen: A data generation system for scalable robot learning using human demonstrations. In 7th Annual Conference on Robot Learning, 2023.

[35] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. Advances in Neural Information Processing Systems, 36:44776–44791, 2023.

[36] J. Cen, C. Yu, H. Yuan, Y. Jiang, S. Huang, J. Guo, X. Li, Y. Song, H. Luo, F. Wang, D. Zhao, and H. Chen. WorldVLA: Towards autoregressive action world model, 2025. URL https: //arxiv.org/abs/2506.21539.

[37] Q. Bu, Y. Yang, J. Cai, S. Gao, G. Ren, M. Yao, P. Luo, and H. Li. UniVLA: Learning to act anywhere with task-centric latent actions, 2025. URL https://arxiv.org/abs/2505.06111.

[38] C.-Y. Hung, Q. Sun, P. Hong, A. Zadeh, C. Li, U.-X. Tan, N. Majumder, and S. Poria. NORA: A small open-sourced generalist vision language action model for embodied tasks, 2025. URL https://arxiv.org/abs/2504.19854.

[39] M. J. Kim, C. Finn, and P. Liang. Fine-tuning vision-language-action models: Optimizing speed and success. arXiv preprint arXiv:2502.19645, 2025.

[40] Poly Haven. Poly Haven: A free 3D asset library. https://polyhaven.com/, 2024. Textures licensed under CC0 1.0 Universal Public Domain Dedication.

[41] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Muller, J. Penna, and R. Rom- ¨ bach. SDXL: Improving latent diffusion models for high-resolution image synthesis. In International Conference on Learning Representations, 2024. URL https://openreview.net/forum? id=di52zR8xgf.

[42] Stability AI. Stable Diffusion XL Base 1.0. https://huggingface.co/stabilityai/ stable-diffusion-xl-base-1.0, 2023. Hugging Face model card.

[43] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=nZeVKeeFYf9.

## Appendix Overview

This appendix provides additional details, analyses, and supplementary experiments for DILL. Appendix A describes the method implementation and training objectives. Appendix B provides simulation benchmark details, including the LIBERO shortcut diagnostic, ablations, and the full LIBERO-Plus breakdown. Appendix C provides additional representation diagnostics. Appendix D describes the real-world evaluation protocol.

## Contents.

• Appendix A: Method details and training objectives.

• Appendix B: Simulation benchmark experiments.

• Appendix C: Representation diagnostics.

• Appendix D: Real-world evaluation protocol.

## APPENDIX

## A Additional Method Details

## A.1 Domain Transformations and Positive Pair Mining

This subsection supports Sec. 3.2 by detailing how we construct the domain-transformed positive pairs to train the Task-Domain Encoder. A domain condition η specifies the full transformation condition, including the transformation family and its sampled parameters or random seed. Taskpositive pairs vary the domain condition while preserving the same underlying observation chunk, whereas domain-positive pairs preserve the domain condition across different observation chunks.

Data sources. Task-domain encoder pretraining uses a source trajectory collection composed of MimicGen [34], ManiSkill [33], and subsets of OXE [5]. The OXE subset includes Bridge and Fractal/RT-1 trajectories. Table 3 summarizes the pretraining data. In our main experiments, the Task-Domain Encoder is pretrained once on this source trajectory collection and then kept fixed during VLA policy pretraining and downstream policy learning. This design decouples Task-Domain Encoder pretraining from downstream policy learning: downstream users can train only the policyside modules unless they choose to further adapt the Task-Domain Encoder with additional domain transformations.

Table 3: Source trajectory collection used for Task-Domain Encoder pretraining. The MimicGen row aggregates 16 task datasets, while OXE aggregates Bridge and Fractal/RT-1 subsets.
<table><tr><td>Source</td><td>Frames</td><td>Trajectories</td><td>Tasks</td></tr><tr><td>MimicGen, 16 datasets</td><td>1,010,618</td><td>3,200</td><td>16</td></tr><tr><td>ManiSkill, 2 datasets</td><td>1,632,291</td><td>14,057</td><td>17</td></tr><tr><td>OXE, Bridge + Fractal/RT-1</td><td>5,785,810</td><td>140,404</td><td>20,553</td></tr><tr><td>Total</td><td>8,428,719</td><td>157,661</td><td>20,586</td></tr></table>

For simulation data, controllable rendering enables viewpoint changes and, when segmentation masks are available, mask-based background replacement. We additionally apply image-level transformations directly to frames; these transformations are used for both simulation data and recorded trajectories such as the OXE subsets.

Transformation families. We group domain transformations into four families that reflect common visual domain shifts: viewpoint changes, environment appearance variation, camera heterogeneity, and visual degradation. A sampled domain condition can include a single transformation or a combination of transformations from these families. Table 4 summarizes the transformation types and the shared domain condition used for domain-positive pairs. Figure 5 illustrates simple singlefamily task-positive examples; the training procedure samples a broader range of transformation types, parameters, and combinations.

![](images/d351b85846c346056434570043b786020a058a18c4cca0280e689312e14f7d85.jpg)  
Anchor

![](images/d3e616127100fc6218904772bd0d112344288408884cdb463b9948a3ec346975.jpg)  
Viewpoint Changes

![](images/80169d73fff0113d51059b94f7b09bb0538a44031dbfdac87e91313bc7b64bb8.jpg)  
Environment Appearance

![](images/c5f4c7d4d4d1fd9746c69a5ce5d99bf8e1f8ad99154d7cae74fd46e1085f7068.jpg)  
Camera Heterogeneity

![](images/2d9bd1c1aaa64ed3c6eeb7ffe68bf63e168cced46ceba67d1cd9f0d046b94fe5.jpg)  
Visual Degradation  
Figure 5: Representative single-family task-positive transformations. Each column after the anchor shows the same trajectory chunk under one family of domain-specific visual factors.

Table 4: Domain transformation families used for positive pair mining. A domain condition η consists of a transformation family and its sampled parameters or random seed. The last column lists the condition shared by domain-positive pairs.
<table><tr><td>Family</td><td>Transformations</td><td>Shared condition for domain positives</td></tr><tr><td>Viewpoint changes</td><td>Rendered camera view, random crop, intrinsics change, radial distortion, warping</td><td>Camera ID or camera parameters; crop seed; focal scale; distortion and warp parameters</td></tr><tr><td>Environment appearance</td><td>Background replacement, lighting changes, color changes</td><td>Background texture IDs; mask regions; replacement mode; lighting or color parameters</td></tr><tr><td>Camera heterogeneity</td><td>Camera unprocessing, sensor noise, camera color response changes</td><td>Camera-pipeline seed; color matrices; gamma; ISO level; shot/read noise; channel gains</td></tr><tr><td>Visual degradation</td><td>Blur, weather effects, compression, pixelation, image corruptions</td><td>Corruption type, severity, and random seed</td></tr></table>

Background replacement. For ManiSkill trajectories [33] with available segmentation masks, we perform mask-based background replacement for table, floor, and wall regions. Table and floor textures are sampled from Poly Haven [40], while wall and background assets are generated with Stable Diffusion XL using text-to-image prompts for robot-relevant environments such as factories, kitchens, laboratories, workspaces, loading docks, and server rooms [41, 42]. The resulting texture pool contains 92 table textures, 163 floor textures, and 604 wall/background textures, for a total of 859 region-specific assets. At training time, textures are sampled by region and composited into the corresponding segmentation masks. For domain-positive pairs, the same texture IDs, mask regions, and replacement mode define the shared domain condition. Representative samples from the texture pool are shown in Fig. 6.

![](images/b537479824e61be49bc2126359e3caa045bc3dc836e58b77a646df1b41091c04.jpg)  
Figure 6: Representative assets for mask-based background replacement. We sample table and floor textures from Poly Haven and generate wall and background assets with Stable Diffusion XL.

Pair mining details. For the task encoder, each task-positive pair is constructed from the same episode and the same temporal window. Each training item samples two non-overlapping windows from the same episode, denoted by $\mathbf { o } _ { t _ { a } } ^ { i }$ and $\mathbf { o } _ { t _ { b } } ^ { i }$ , and samples two domain conditions $\eta _ { 1 }$ and $\eta _ { 2 }$ . We then form two task-positive pairs:

$$
\big ( \overline  { \mathscr { T } _ { \eta _ { 1 } } ( \mathbf { o } _ { t _ { a } } ^ { i } ) , \mathscr { T } _ { \eta _ { 2 } } ( \mathbf { o } _ { t _ { a } } ^ { i } ) \big ) } , \qquad \big ( \overline  { \mathscr { T } _ { \eta _ { 1 } } ( \mathbf { o } _ { t _ { b } } ^ { i } ) , \mathscr { T } _ { \eta _ { 2 } } ( \mathbf { o } _ { t _ { b } } ^ { i } ) \big ) } .\tag{13}
$$

Within each pair, the two clips share the same underlying temporal window but differ in domain condition. Across the two pairs, the temporal windows are distinct while the task identity and domain conditions are shared. After collation, the two task-positive pairs per item are flattened from $[ B , 2 , \cdot \cdot \cdot ] \mathrm { t o } [ 2 B , \cdot \cdot \cdot ]$ before applying InfoNCE. Thus, examples from the same episode but different temporal windows are included as ordinary in-batch negatives. This discourages the task encoder from solving the contrastive objective using task identity alone, while avoiding false negatives from the same underlying temporal segment.

For the domain encoder, a domain-positive pair is constructed by applying the same domain condition to two different clips. Each training item first samples a source pair $\left( \mathbf { o } _ { a } , \mathbf { o } _ { b } \right)$ , either from two non-overlapping windows of the same episode or from cross-task clips within the same camera bucket. It then samples four distinct domain-transformation types without replacement, with corresponding domain conditions $\eta _ { 1 } , \dots , \eta _ { 4 }$ , and forms four domain-positive pairs:

$$
\big ( T _ { \eta _ { m } } ( \mathbf { o } _ { a } ) , \mathcal { T } _ { \eta _ { m } } ( \mathbf { o } _ { b } ) \big ) , \qquad m = 1 , \ldots , 4 .\tag{14}
$$

Within each pair, the clips differ in trajectory content but share the same domain condition. Across the four pairs in the same item, the source clips are fixed while the domain-transformation type changes. After collation, the four domain-positive pairs per item are flattened from $\left[ B , 4 , \cdots \right]$ to $[ 4 B , \cdots ]$ before applying InfoNCE. Therefore, the domain encoder receives controlled in-batch comparisons: for an anchor under domain condition $\eta _ { m }$ , other clips from the same item but with $\eta _ { m ^ { \prime } } \neq \eta _ { m }$ appear among the standard in-batch negatives. No separate hard-negative loss is used; the dataloader simply ensures that informative non-matching domain conditions are present in the minibatch.

All transformations are applied consistently across the frames of a chunk so that each video chunk corresponds to a coherent visual domain.

## A.2 Task-Domain Latent Disentanglement Objectives

InfoNCE objective. For a minibatch of positive pairs $\{ ( x _ { i } , x _ { i } ^ { + } ) \} _ { i = 1 } ^ { B }$ , let $z _ { i }$ and $z _ { i } ^ { + }$ denote the pooled latent vectors from the corresponding encoder. We use the InfoNCE loss

$$
\mathcal { L } _ { \mathrm { N C E } } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \exp ( \sin ( z _ { i } , z _ { i } ^ { + } ) / \tau ) } { \sum _ { k = 1 } ^ { B } \exp ( \sin ( z _ { i } , z _ { k } ^ { + } ) / \tau ) } ,\tag{15}
$$

where $\mathrm { s i m } ( \cdot , \cdot )$ denotes cosine similarity and $\tau$ is a temperature. For the task encoder, the positive pairs are task-positive pairs; for the domain encoder, the positive pairs are domain-positive pairs. The resulting losses are $\mathcal { L } _ { \mathrm { t a s k } }$ and ${ \mathcal { L } } _ { \mathrm { d o m } }$

Gaussian disentanglement loss. The task and domain encoders are trained with different positive pairs, but without an additional constraint, the resulting task and domain latents can still encode overlapping information. We therefore define a Gaussian disentanglement loss that regularizes concatenated latents toward an isotropic Gaussian distribution.

For a minibatch, we construct

$$
u _ { i } ^ { \mathrm { t d } } = [ z _ { i } ^ { \mathrm { t a s k } } ; z _ { i } ^ { \mathrm { d o m } } ] , \qquad i = 1 , \ldots , B ,\tag{16}
$$

where $[ \cdot ; \cdot ]$ denotes concatenation. Here $z _ { i } ^ { \mathrm { t a s k } } , z _ { i } ^ { \mathrm { d o m } } \in \mathbb { R } ^ { D _ { z } }$ , and therefore $u _ { i } ^ { \mathrm { t d } } \in \mathbb { R } ^ { D }$ with $D = 2 D _ { z }$ We implement the disentanglement regularizer $\mathcal { R } _ { \mathrm { d i s } }$ using SIGReg [32]. Given a batch of vectors $U = \bar { \{ u _ { i } \} } _ { i = 1 } ^ { B } \subset \mathbb { R } ^ { D }$ , we sample M random unit directions $r _ { m } \sim \mathrm { U n i f } ( \mathbb { S } ^ { D - 1 } )$ , project $s _ { i , m } =$ $r _ { m } ^ { \top } u _ { i } .$ , and match each projected distribution to a standard Gaussian. Using a Gaussian kernel with bandwidth $\sigma ,$ the Epps–Pulley statistic for the projected samples $\{ s _ { i , m } \} _ { i = 1 } ^ { B }$ along direction $r _ { m }$ is

$$
\begin{array} { l } { { \displaystyle \mathrm { E P } _ { \sigma } \big ( \big \{ s _ { i , m } \big \} _ { i = 1 } ^ { B } \big ) = \frac { 1 } { B ^ { 2 } } \sum _ { i = 1 } ^ { B } \sum _ { j = 1 } ^ { B } \exp { \left( - \frac { \big ( s _ { i , m } - s _ { j , m } \big ) ^ { 2 } } { 2 \sigma ^ { 2 } } \right) } } } \\ { { - \frac { 2 \sigma } { B \sqrt { \sigma ^ { 2 } + 1 } } \sum _ { i = 1 } ^ { B } \exp { \left( - \frac { s _ { i , m } ^ { 2 } } { 2 \big ( \sigma ^ { 2 } + 1 \big ) } \right) } + \frac { \sigma } { \sqrt { \sigma ^ { 2 } + 2 } } . } } \end{array}\tag{17}
$$

The regularizer averages this statistic over random projection directions:

$$
\mathcal { R } _ { \mathrm { d i s } } ( U ) = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \mathrm { E P } _ { \sigma } \left( \{ r _ { m } ^ { \top } u _ { i } \} _ { i = 1 } ^ { B } \right) .\tag{18}
$$

The task-domain disentanglement loss is

$$
\mathcal { L } _ { \mathrm { d i s } } ^ { \mathrm { t d } } = \mathcal { R } _ { \mathrm { d i s } } \left( \{ u _ { i } ^ { \mathrm { t d } } \} _ { i = 1 } ^ { B } \right) .\tag{19}
$$

Why Gaussian matching encourages disentanglement. The Gaussian disentanglement loss does not guarantee exact independence for arbitrary distributions. Its motivation is clearest under a joint Gaussian approximation. Let

$$
u = \left[ { z \atop z ^ { a } } \right]
$$

be a jointly Gaussian random vector with covariance

$$
\Sigma = \left[ \begin{array} { c c } { { \Sigma _ { a } } } & { { C } } \\ { { C ^ { \top } } } & { { \Sigma _ { b } } } \end{array} \right] .\tag{20}
$$

If the concatenated latent is isotropic Gaussian, then $\Sigma = I ,$ , which implies

$$
\Sigma _ { a } = I , \qquad \Sigma _ { b } = I , \qquad C = 0 .\tag{21}
$$

For jointly Gaussian variables, zero cross-covariance is sufficient for independence, so

$$
p ( z ^ { a } , z ^ { b } ) = p ( z ^ { a } ) p ( z ^ { b } ) .\tag{22}
$$

Thus, driving the concatenated latent $[ z ^ { a } , z ^ { b } ]$ toward an isotropic Gaussian encourages the two components to be disentangled.

We use this argument twice. For task-domain latent disentanglement, $z ^ { a } = z ^ { \mathrm { t a s k } }$ and $z ^ { b } = z ^ { \mathrm { d o m } }$ For policy learning, $z ^ { a } ~ = ~ z ^ { \mathrm { c u r r } }$ and $z ^ { b } = z ^ { \mathrm { d o m } , \star }$ . In practice, this should be interpreted as an approximate disentanglement regularizer rather than a guarantee of exact independence.

Total disentanglement objective. The full task-domain disentanglement objective is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { T D D } } = \lambda _ { \mathrm { t a s k } } \mathcal { L } _ { \mathrm { t a s k } } + \lambda _ { \mathrm { d o m } } \mathcal { L } _ { \mathrm { d o m } } + \lambda _ { \mathrm { d i s } } ^ { \mathrm { t d } } \mathcal { L } _ { \mathrm { d i s } } ^ { \mathrm { t d } } . } \end{array}\tag{23}
$$

## A.3 Policy Objectives

Lookahead contrastive alignment. For a minibatch, the Lookahead Predictor outputs $\{ \hat { z } _ { i } ^ { \mathrm { l o o k } } \} _ { i = 1 } ^ { B }$ and the pretrained task encoder provides future task targets $\{ z _ { i } ^ { \mathrm { t a s k , \star } } \} _ { i = 1 } ^ { B }$ . We train the Lookahead Predictor with the in-batch InfoNCE loss

$$
\mathcal { L } _ { \mathrm { l o o k } } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \frac { \exp \left( \sin \left( \hat { z } _ { i } ^ { \mathrm { l o o k } } , \mathrm { s g } ( z _ { i } ^ { \mathrm { t a s k } , \star } ) \right) / \tau _ { \mathrm { l o o k } } \right) } { \sum _ { k = 1 } ^ { B } \exp \left( \sin \left( \hat { z } _ { i } ^ { \mathrm { l o o k } } , \mathrm { s g } ( z _ { k } ^ { \mathrm { t a s k } , \star } ) \right) / \tau _ { \mathrm { l o o k } } \right) } .\tag{24}
$$

The positive pair is the predicted lookahead and its corresponding future task target; for each anchor, all other future task targets in the batch serve as negatives. The stop gradient sg(·) prevents this loss from updating the pretrained task encoder.

Current Representation Head domain disentanglement. For a minibatch, let $\{ z _ { i } ^ { \mathrm { c u r r } } \} _ { i = 1 } ^ { B }$ denote current representations produced by the Current Representation Head, and let $\{ z _ { i } ^ { \mathrm { d o m , \star } } \} _ { i = 1 } ^ { B }$ denote the corresponding domain latents extracted by the pretrained domain encoder. We construct

$$
u _ { i } ^ { \mathrm { c u r r } } = [ z _ { i } ^ { \mathrm { c u r r } } ; \mathrm { s g } ( z _ { i } ^ { \mathrm { d o m , \star } } ) ] , \qquad i = 1 , \ldots , B .\tag{25}
$$

The current-domain disentanglement loss is

$$
{ \mathcal { L } } _ { \mathrm { d i s } } ^ { \mathrm { c u r r } } = { \mathcal { R } } _ { \mathrm { d i s } } \left( \{ u _ { i } ^ { \mathrm { c u r r } } \} _ { i = 1 } ^ { B } \right) .\tag{26}
$$

Under the Gaussian approximation described above, this encourages the current representation to be disentangled from the domain component while retaining information needed for action prediction.

Action loss and policy objective. The action head receives both policy latents:

$$
\begin{array} { r } { \hat { a } _ { t : t + K _ { a } - 1 } = A _ { \phi } \left( \left[ \hat { z } _ { t } ^ { \mathrm { l o o k } } ; z _ { t } ^ { \mathrm { c u r r } } \right] \right) . } \end{array}\tag{27}
$$

The behavior cloning loss is

$$
\mathcal { L } _ { \mathrm { a c t } } = \frac { 1 } { K _ { a } } \sum _ { k = 0 } ^ { K _ { a } - 1 } \| \hat { a } _ { t + k } - a _ { t + k } \| _ { 1 } .\tag{28}
$$

The full policy objective is

$$
\mathcal { L } _ { \mathrm { p o l i c y } } = \lambda _ { \mathrm { a c t } } \mathcal { L } _ { \mathrm { a c t } } + \lambda _ { \mathrm { l o o k } } \mathcal { L } _ { \mathrm { l o o k } } + \lambda _ { \mathrm { d i s } } ^ { \mathrm { c u r r } } \mathcal { L } _ { \mathrm { d i s } } ^ { \mathrm { c u r r } } .\tag{29}
$$

## A.4 Architecture and Implementation Details

Task and domain encoders. In the main text, $E _ { \psi } ^ { \mathrm { t a s k } }$ and $E _ { \xi } ^ { \mathrm { d o m } }$ denote the full task and domain encoder branches that output pooled latent vectors. Each branch consists of a VJEPA2 video backbone [31], branch-specific LoRA adapters [43], and an attention-pooling projection head.

Multi-query attention pooling. Let $H \in \mathbb R ^ { N \times D _ { v } }$ be a sequence of input tokens and let $Q \in$ $\mathbb { R } ^ { R \times D _ { q } }$ be R learned query vectors. We project queries, keys, and values using $W _ { q } \in \mathbb { R } ^ { D _ { q } \times d } ,$ $W _ { k } \in \mathbb R ^ { D _ { v } \times d }$ , and $W _ { v } \doteq \bar { \mathbb { R } ^ { D _ { v } \times D _ { p } } }$ . A multi-query attention pooler computes

$$
A = \mathrm { s o f t m a x } \left( \frac { ( Q W _ { q } ) ( H W _ { k } ) ^ { \top } } { \sqrt { d } } \right) ,\tag{30}
$$

$$
U = A H W _ { v } \in \mathbb R ^ { R \times D _ { p } } .\tag{31}
$$

The pooled tokens $U$ are flattened and further projected to the output latent dimension $D _ { z }$ :

$$
\mathrm { H e a d } ( H ) = W _ { o } \operatorname { v e c } ( U ) + b _ { o } , \qquad W _ { o } \in \mathbb { R } ^ { D _ { z } \times ( R D _ { p } ) } , \quad b _ { o } \in \mathbb { R } ^ { D _ { z } } .\tag{32}
$$

The projection heads for the task encoder, domain encoder, Lookahead Predictor, and Current Rep resentation Head all use this attention-pooling-plus-linear architecture, with separate parameters.

Policy-side heads. The policy uses a Qwen2.5-VL [13] backbone with LoRA adaptation. On top of the VLM tokens, we train two AttentiveLatentHead modules and one ResNetActionHead. The Lookahead Predictor maps VLM tokens to $\hat { z } _ { t } ^ { \mathrm { l o o k } }$ , and the Current Representation Head maps the same tokens to $z _ { t } ^ { \mathrm { c u r r } }$ . Each latent head uses 8 learned query tokens, a two-layer attentive pooler with 16 attention heads, and a linear projection to a 4096-dimensional latent.

Action head. The action head is an MLP-ResNet that maps the concatenated policy latents to an action chunk. In the dual-head setting, the action head input dimension is 4096 + 4096 = 8192. We use hidden dimension 2048, two residual MLP blocks, and output a chunk size of $K _ { a } = 5 0$

Table 5: Policy-side head budget for the VLA implementation.
<table><tr><td>Module</td><td>Parameters</td></tr><tr><td>Lookahead Predictor, AttentiveLatentHead</td><td>167.85M</td></tr><tr><td>Current Representation Head, AttentiveLatentHead</td><td>167.85M</td></tr><tr><td>ResNetActionHead, dual-head input</td><td>28.48M</td></tr><tr><td>Total policy-side heads</td><td>364.18M</td></tr></table>

## A.5 Training Stage Summary

Table 6 summarizes which modules are updated in each training stage. In the main experiments, the Task-Domain Encoder is pretrained once on the source trajectory collection and then kept fixed. This decouples reusable Task-Domain Encoder training from downstream VLA policy training.

Table 6: Training Stage Summary for Domain-Invariant Latent Lookahead.
<table><tr><td>Stage</td><td>Data</td><td>Updated modules</td><td>Objective</td></tr><tr><td>Task-domain encoder pretraining</td><td>Source trajectory collection</td><td>Task/domain encoder LoRA adapters and projection heads</td><td>LTDD</td></tr><tr><td>VLA policy pretraining</td><td>Source trajectory collection</td><td>VLM LoRA, Lookahead Predictor, Current Representation Head, action head</td><td>Lpolicy</td></tr><tr><td>Downstream policy tuning</td><td>Downstream action-labeled demonstrations (e.g., LIBERO)</td><td>VLM LoRA, Lookahead Predictor, Current Representation Head, action head</td><td>Lpolicy</td></tr></table>

Optional Task-Domain Encoder adaptation. Although the main experiments keep the pretrained Task-Domain Encoder fixed for downstream policy tuning, the same task-domain disentanglement objective could be used to adapt the task and domain encoders on additional downstream demonstration datasets. When controllable rendering or segmentation masks are unavailable, such optional adaptation would rely on image-level transformations such as cropping, warping, lighting and color changes, camera-pipeline perturbations, and visual corruptions.

## A.6 Computational cost

Task-Domain Encoder pretraining requires 273 A100 GPU-hours once; the resulting encoder is reused and remains frozen during downstream policy training. It is not used at deployment. Table 7 separates downstream tuning cost, peak training memory, and inference latency. DILL increases downstream tuning cost by 42% and peak memory by 1.5 GB relative to Base VLA, while the measured latency increases by 0.3 ms per 30-action chunk. Thus, the additional cost is concentrated in training rather than deployment.

<table><tr><td>Metric</td><td>Base VLA</td><td>DILL</td></tr><tr><td>Downstream tuning cost (× Base VLA)</td><td>1.00</td><td>1.42</td></tr><tr><td>Peak training memory (GB)</td><td>20.9</td><td>22.4</td></tr><tr><td>Inference latency (ms per 30-action chunk)</td><td>69.9</td><td>70.2</td></tr></table>

Table 7: Training and inference costs. The one-time Task-Domain Encoder pretraining cost is reported separately in the text. Latency is measured in a separate benchmark with 30-action chunks.

## B Simulation Benchmark Experiments

## B.1 LIBERO Shortcut Diagnostic

We use the LIBERO shortcut diagnostic of [11] to test whether a policy follows the commanded task or instead executes the task spuriously associated with the observed view. Unlike standard robustness evaluation, this diagnostic explicitly separates shortcut-driven task substitution from general execution failure.

Task–view confounding. The benchmark constructs two confounded task–view islands. We refer to the left-view task group as Task-L and the right-view task group as Task-R. Task-L contains LIBERO task IDs {0, 1, 3, 5, 8} and is observed only in the left-view range (10<sup>◦</sup>–25<sup>◦</sup>) during training. Task-R contains task IDs {2, 4, 6, 7, 9} and is observed only in the right-view range (55<sup>◦</sup>–70<sup>◦</sup>). At test time, we evaluate counterfactual task–view compositions by swapping these associations: Task-R is evaluated at the left viewpoint (10<sup>◦</sup>), and Task-L is evaluated at the right viewpoint (70<sup>◦</sup>). A task-faithful policy should follow the language instruction under these swapped views, whereas a shortcut-prone policy will execute the task spuriously associated with the observed view.

Metrics. We report two complementary metrics. OOD success rate measures whether the commanded task is completed under the counterfactual view. Shortcut degree measures whether the policy instead executes the task spuriously associated with the observed view during training, thereby isolating view-induced task substitution.

## B.2 Ablation Study

We ablate DILL to identify which design choices are responsible for shortcut mitigation. The central question is not whether a model can fit the confounded training distribution, but whether it can avoid using viewpoint as a proxy for task identity when the task–view association is broken. DILL is designed to address this by reshaping the information routed to the action head: a predicted lookahead latent aligned with the future task latent, and a current representation disentangled from the corresponding domain latent. The ablation study therefore asks whether each of these ingredients is necessary, and whether simpler alternatives—more augmented data or generic future prediction— are sufficient.

All variants are evaluated on the LIBERO shortcut diagnostic in Appendix B.1. This diagnostic separates three levels of generalization. In-dist. SR measures success on the original task–view training compositions. Center OOD SR evaluates each task group at an unseen midpoint viewpoint without swapping task–view association, measuring interpolation to an unseen visual domain. Counter OOD SR evaluates the counterfactual task–view swaps, where the model must follow the commanded task rather than the task associated with the observed view. Finally, shortcut degree measures how often the model follows the task spuriously associated with the observed view; lower is better.

Compared methods. We compare DILL with several ablative models:

(1) Base VLA. Base VLA is the plain behavior-cloning baseline. It uses the same VLA architecture as DILL, but directly maps the current observation-conditioned representation to actions. It does not use source augmentation, task–domain latent targets, the disentangled current head, or latent lookahead alignment. This is the minimal baseline for testing whether standard VLA behavior cloning is sufficient under task–view confounding.

(2) Base VLA + SA. This variant keeps the same behavior-cloning objective and VLA architecture as Base VLA, but trains with the same augmented source data used by DILL for task–domain encoder pretraining. This controls for the data condition: if this variant were sufficient, the gain of DILL could be attributed mainly to additional visual diversity. If not, then the improvement must come from how DILL uses the augmented data to shape the latent space, rather than from the data alone.

<table><tr><td rowspan="2">Variant</td><td colspan="4">Components</td><td colspan="5">Metrics</td></tr><tr><td>SA</td><td>TD</td><td>CH</td><td>LA</td><td>In-dist. SR↑</td><td>Center OOD SR↑</td><td>Counter OOD SR ↑</td><td>Avg. SR↑</td><td>Shortcut degree ↓</td></tr><tr><td>Base VLA</td><td></td><td></td><td></td><td></td><td>68%</td><td>22%</td><td>0%</td><td>30%</td><td>73%</td></tr><tr><td>Base VLA + SA</td><td>√</td><td></td><td></td><td></td><td>82%</td><td>10%</td><td>2%</td><td>31%</td><td>69%</td></tr><tr><td>Entangled LA</td><td>√</td><td></td><td></td><td>√</td><td>76%</td><td>24%</td><td>0%</td><td>33%</td><td>65%</td></tr><tr><td>DILL w/o CH</td><td>√</td><td>√</td><td></td><td>√</td><td>60%</td><td>44%</td><td>44%</td><td>49%</td><td>6%</td></tr><tr><td>DILL w/o LA</td><td>√</td><td>√</td><td>√</td><td></td><td>50%</td><td>62%</td><td>62%</td><td>58%</td><td>3%</td></tr><tr><td>DILL</td><td>√</td><td>√</td><td>√</td><td>√</td><td>66%</td><td>62%</td><td>58%</td><td>62%</td><td>5%</td></tr></table>

Table 8: Ablation study. All variants are evaluated on the LIBERO shortcut diagnostic under the same task–view island protocol. Component columns indicate whether each variant uses source augmentation (SA), task–domain latent targets from the task/domain encoders (TD), the disentangled current head (CH), and latent lookahead alignment (LA). In-dist. SR averages the original training compositions, Center OOD SR evaluates unseen midpoint viewpoints without swapping task identity, and Counter OOD SR evaluates counterfactual task–view swaps. Shortcut degree measures how often the policy follows the task associated with the observed view rather than the commanded task. Blank component entries indicate that the component is absent.

(3) Entangled LA. This variant adds latent lookahead prediction, but uses the representation from the original video encoder [31] as the future target instead of the task–domain latent targets. Thus, the predicted future latent is not explicitly encouraged to discard domain-specific information, and may still contain viewpoint, lighting, or background cues. This variant tests whether generic future prediction is sufficient, or whether the lookahead target must be domain-invariant.

(4) DILL w/o CH. This variant removes the disentangled current head. The model still uses taskdomain latent targets and latent lookahead alignment, but the current action-conditioning representation is no longer explicitly regularized to be disentangled from the corresponding domain latent. This tests whether domain-invariant lookahead alone is sufficient, or whether the current represen tation used by the policy also needs to be explicitly shaped to suppress domain-specific visual cues.

(5) DILL w/o LA. This variant keeps the task–domain latent targets and the disentangled current head, but removes latent lookahead alignment. In other words, the current representation is still regularized through the task–domain latent structure, but the future lookahead latent is not trained to align with the domain-invariant target. This tests whether a disentangled current representation alone can mitigate shortcut learning, or whether robust control requires explicitly aligning the predicted future latent as well.

(6) DILL. DILL is the full model, combining source augmentation, task–domain latent targets, the disentangled current head, and domain-invariant latent lookahead alignment.

Results. Table 8 shows that fitting the confounded training distribution is not sufficient for shortcutfree generalization. Base VLA reaches 68% in-distribution success, but obtains 0% Counter OOD success and a high shortcut degree of 73%. Adding source augmentation improves in-distribution success to 82% and slightly reduces shortcut degree to 69%, indicating that additional visual diversity and action pretraining are helpful for fitting the nominal tasks and can mildly reduce shortcut reliance. However, Counter OOD success remains only 2%, showing that data augmentation alone does not teach the policy which visual factors should be ignored when task–view correlations are counterfactually broken.

Entangled LA provides a second partial improvement. Compared to Base VLA + SA, it improves Center OOD success from 10% to 24% and further reduces shortcut degree from 69% to 65%, suggesting that future prediction can encourage representations that are somewhat more robust to unseen viewpoints. However, it still obtains 0% Counter OOD success. This failure is informative:

predicting a future latent is not sufficient if the target representation remains entangled with viewpoint, background, or other domain-specific cues. The lookahead target must be tied to task-relevant structure rather than to the same visual correlations present in the training distribution.

The lower half of the table isolates the DILL-specific components. DILL w/o CH, which keeps task– domain latent targets and latent lookahead alignment but removes the disentangled current head, reaches 44% Center OOD and 44% Counter OOD success while reducing shortcut degree to 6%. This is a large jump over Entangled LA, indicating that aligning lookahead prediction with task– domain latent targets removes much of the view-induced task substitution. DILL w/o LA, which keeps the disentangled current head but removes latent lookahead alignment, achieves the strongest Counter OOD success (62%) and the lowest shortcut degree (3%), but its in-distribution success drops to 50%. This suggests that the disentangled current head is highly effective at suppressing shortcut behavior, while latent lookahead alignment helps recover a better balance between nominal task execution and counterfactual generalization.

The full DILL model provides the best overall tradeoff. It matches the best Center OOD succes (62%), remains close to the best Counter OOD success (58%), keeps shortcut degree very low (5%), and achieves the highest average success across the three success metrics (62%). The key pattern is that source augmentation and generic lookahead improve some aspects of learning, but do not solve counterfactual task–view generalization. In contrast, the DILL-family variants that use task–domain latent targets sharply reduce shortcut degree, and the full model best preserves both task execution and shortcut resistance. These results support the design principle of DILL: shortcut mitigation requires not only additional visual diversity or future prediction, but a task-structured latent pathway that routes domain-invariant predictive information into action conditioning.

## B.3 LIBERO-Plus evaluation details

We evaluate methods on LIBERO-Plus [7], which tests robustness under controlled perturbations of the original LIBERO benchmark. In the main paper, we report the visual perturbation categories because they are most directly aligned with our claim about visual shortcut mitigation. Here, we provide the full seven-category breakdown. All models are evaluated zero-shot on LIBERO-Plus after training on the original LIBERO setting.

Full perturbation breakdown. Table 9 reports success rates on all LIBERO-Plus perturbation categories. Total denotes the average over Camera, Robot, Language, Light, BG, Noise, and Layout. The second row for each model reports the absolute drop relative to the original LIBERO score.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Original</td><td colspan="7">LIBERO-Plus perturbations</td><td rowspan="2">Total</td></tr><tr><td>Camera</td><td>Robot</td><td>Language Light</td><td></td><td>BG</td><td>Noise</td><td>Layout</td></tr><tr><td rowspan="2">OpenVLA [3]</td><td rowspan="2">76.5</td><td>0.8</td><td>3.5</td><td>23.0</td><td>8.1</td><td>34.8</td><td>15.2</td><td>28.5</td><td>15.6</td></tr><tr><td>↓75.7</td><td>↓73.0</td><td>↓53.5</td><td>↓68.4</td><td>↓41.7</td><td>↓61.3</td><td>↓48.0</td><td>↓60.9</td></tr><tr><td rowspan="2">WorldVLA [36]</td><td rowspan="2">79.1</td><td>0.1</td><td>27.9</td><td>41.6</td><td>43.7</td><td>17.1</td><td>10.9</td><td>38.0</td><td>25.0</td></tr><tr><td>↓79.0</td><td>↓51.2</td><td>↓37.5</td><td>↓35.4</td><td>↓62.0</td><td>↓68.2</td><td>↓41.1</td><td>↓54.1</td></tr><tr><td>UniVLA [37]</td><td>95.5</td><td>1.8</td><td>46.2</td><td>69.6</td><td>69.0</td><td>81.0</td><td>21.2</td><td>31.9</td><td>43.9</td></tr><tr><td rowspan="2">NORA [38]</td><td rowspan="2">87.9</td><td>↓93.7</td><td>↓49.3</td><td>↓25.9</td><td>↓26.5</td><td>↓14.5</td><td>↓74.3</td><td>↓63.6</td><td>↓51.6</td></tr><tr><td>2.2</td><td>37.0</td><td>65.1</td><td>45.7</td><td>58.6</td><td>12.8</td><td>62.1</td><td>39.0</td></tr><tr><td rowspan="2">OpenVLA-OFT [39]</td><td rowspan="2">95.3</td><td>↓85.7</td><td>↓50.9</td><td>↓22.8</td><td>↓42.2</td><td>↓29.3</td><td>↓75.1</td><td>↓25.8</td><td>↓48.9</td></tr><tr><td>10.4</td><td>38.7</td><td>70.5</td><td>76.8</td><td>93.6</td><td>49.9</td><td>69.9</td><td>55.8</td></tr><tr><td rowspan="2"></td><td rowspan="2"></td><td>↓84.9</td><td>↓56.6</td><td>↓24.8</td><td>↓18.5</td><td>↓1.7</td><td>↓45.4</td><td>↓25.4</td><td>↓39.5</td></tr><tr><td></td><td></td><td>2.9</td><td>69.4</td><td>70.0</td><td></td><td></td><td></td></tr><tr><td rowspan="2">DILL (Ours)</td><td rowspan="2">81.6</td><td>68.4</td><td>21.4</td><td>↓78.7</td><td></td><td></td><td>68.7</td><td>55.1</td><td>50.8</td></tr><tr><td>↓13.2</td><td>↓60.2</td><td></td><td>↓12.2</td><td>↓11.6</td><td>↓12.9</td><td>↓26.5</td><td>↓30.8</td></tr></table>

Table 9: Full LIBERO-Plus robustness breakdown. For each model, the first row shows success rate (%) on the original LIBERO benchmark and all seven LIBERO-Plus perturbation categories. The second row shows the absolute drop relative to the original score. Total denotes the official LIBERO-Plus leaderboard score.

Results. The full breakdown clarifies the scope of DILL’s robustness. DILL is strongest on the perturbations that directly change visual appearance or viewpoint, achieving the best performance on Camera (68.4%) and Noise (68.7%), and competitive performance on Light (69.4%) and BG (70.0%). This pattern matches the design of the method: domain-invariant latent lookahead is intended to suppress shortcuts tied to visual-domain cues, such as viewpoint, background texture, and image degradation. In contrast, DILL is not designed to directly solve non-visual shifts. The low Language score reflects a failure mode related to instruction generalization and language grounding, while the Robot score reflects sensitivity to robot initial-state variation. These require different invariances than the visual-domain invariance targeted by our objective.

This distinction is important for interpreting the Total score. DILL obtains a Total score of 50.8%, improving over OpenVLA, WorldVLA, UniVLA, and NORA, but remaining below OpenVLA-OFT due primarily to the Language and Robot categories. Rather than indicating a uniform robustness improvement across all axes, the result shows a more specific effect: DILL substantially reduces brittleness under visual distribution shifts, while leaving language and robot-state robustness as separate failure modes.

## C Representation Diagnostics

## C.1 Pairwise latent similarity protocol

We use pairwise latent similarity diagnostics to test whether the action-conditioning representation is organized around task content rather than domain-specific visual factors. For each representation, we compute cosine similarity between normalized latent vectors for two types of unseen task–domain pairs.

Task pairs. A task pair consists of two observations that share the same underlying trajectory content but differ in visual domain condition. A task-centric latent should assign high similarity to these pairs because the task-relevant trajectory content is preserved across the domain change.

Domain pairs. A domain pair consists of two observations that share the same visual domain condition but differ in trajectory content. A task-centric latent should assign lower similarity to these pairs because the underlying behavior is different. Conversely, a domain-centric latent would assign high similarity to domain pairs, even when the trajectory content changes.

Metric. For a task-centric action-conditioning representation, we summarize the diagnostic using the task–domain separation gap,

$$
\Delta _ { \mathrm { t a s k } } = \mathbb { E } \left[ \sin ( z _ { i } , z _ { j } ) \mid ( i , j ) \in \mathcal { P } _ { \mathrm { t a s k } } \right] - \mathbb { E } \left[ \sin ( z _ { i } , z _ { j } ) \mid ( i , j ) \in \mathcal { P } _ { \mathrm { d o m a i n } } \right] ,
$$

where $\mathcal { P } _ { \mathrm { t a s k } }$ denotes task pairs and $\mathcal { P } _ { \mathrm { d o m a i n } }$ denotes domain pairs. A larger positive gap indicates that the representation is more aligned with task content and less dominated by shared visual domain.

![](images/0c136ee8e0d1ac1b1370cd35a363c0ac3aea193fd39d69e678502b19145d40a9.jpg)

![](images/90b050c1b3d3b9ae05403204c430d2ea5a20cd17b21d726b90477147761ab47a.jpg)  
Task pair: same trajectory, different domain

![](images/f27ce720e7bf9a9c4f59a758ad451d2d7fd65445b542011532b978f2a4550df0.jpg)

![](images/e3a296c13791036d69768f43f859c93524e50f6854997b5fbcd6fba360953608.jpg)  
Domain pair: same domain, different trajectory  
Figure 7: Pair construction for latent similarity diagnostics. Task pairs test whether a latent remains stable across visual-domain changes when the underlying trajectory content is preserved. Domain pairs test whether a latent collapses observations that share visual-domain cues despite different trajectory content.

## C.2 Final action-conditioning latent

The main paper analyzes the task encoder, domain encoder, and DILL’s final action-conditioning latent (see Section 4.3). Here, we further compare the final action-conditioning latent of DILL against that of Base VLA. This comparison directly tests whether the proposed objective changes the representation used by the policy in a useful way. If DILL works as intended, the action-conditioning latent should become less organized around shared visual domain and more organized around task relevant trajectory content than the latent learned by an architecture-matched VLA trained with behavior cloning.

For Base VLA, we evaluate the final action-conditioning latent produced by the standard VLA pol icy. For DILL, we evaluate the final action-conditioning latent formed by concatenating the current representation and the predicted lookahead latent. We expect a task-centric action-conditioning latent to assign higher similarity to task pairs than to domain pairs: observations with the same trajectory content should remain close even when the visual domain changes, while observations that merely share the visual domain should remain separated when their trajectory content differs.

Interpretation. The final action-conditioning latent is the representation most directly tied to control, since it is the input used by the action head. The results in Figure 8 and Table 10 show that Base

![](images/78d2d0b953c2e986dd620c7dcf5430561a92a76a623a04fa7a3efdd7554dc47b.jpg)

![](images/3c763221b80c909a7db33d82120ea076ea607dfc3fc5ae83e43f312e70ad151c.jpg)

Figure 8: Final action-conditioning latent similarity diagnostics. We compare Base VLA and DILL using the final latent representation provided to the action head. A task-centric actionconditioning latent should assign higher similarity to task pairs than to domain pairs.
<table><tr><td>Representation</td><td>Task-pair similarity ↑</td><td>Domain-pair similarity ↓</td><td>Gap  $\Delta _ { \mathrm { t a s k } } \uparrow$ </td></tr><tr><td>Base VLA final latent</td><td>0.52</td><td>0.70</td><td>-0.18</td></tr><tr><td>DILL final latent</td><td>0.94</td><td>0.55</td><td>0.39</td></tr></table>

Table 10: Summary of final action-conditioning latent similarity. The gap $\Delta _ { \mathrm { t a s k } }$ is computed as mean task-pair similarity minus mean domain-pair similarity. Larger gap indicates a more taskcentric action-conditioning representation.

VLA exhibits an undesirable ordering for shortcut-robust behavior: its domain-pair similarity (0.70) is higher than its task-pair similarity (0.52), yielding a negative gap of −0.18. This suggests that the standard VLA latent is more strongly organized by shared visual domain than by shared trajectory content. Such a representation may make the policy more susceptible to relying on domain-specific visual cues when task identity and viewpoint are confounded.

DILL reverses this ordering. Its final action-conditioning latent assigns much higher similarity to task pairs (0.94) than to domain pairs (0.55), increasing the separation gap from −0.18 to 0.39. This indicates a substantial change in the structure of the representation used for action prediction: the latent becomes stable across domain changes when the underlying trajectory is preserved, while remaining discriminative when the trajectory content changes even within the same visual domain. The domain-pair similarity is not forced to vanish, which is expected because observations can still share scene layout and low-level visual statistics; the important point is that these shared visual domain factors no longer dominate the action-conditioning representation. These representationlevel observations complement the behavioral results: reduced shortcut reliance is accompanied by a more task-consistent geometry in the representation supplied to the action head.

## C.3 Domain predictability with linear probes

Pairwise similarity does not directly reveal whether domain attributes remain predictable from a representation. We test this using cross-task linear probes: linear classifiers trained to predict viewpoint, lighting, or sensor-noise labels from the representation supplied to the action head. Accuracy above chance indicates that these domain cues remain linearly accessible.

Mean accuracy is 64.4% for Base VLA and 25.4% for DILL, compared with a chance reference of 21.7%. DILL’s accuracy is closer to chance, suggesting reduced linear access to visual-domain information in the representation used for control. This complements the similarity analysis without establishing that all domain information has been removed.

## D Real-World Evaluation Protocol

Robot setup and training. We use a 6-DoF Universal Robots UR5e equipped with a Robotiq 2- Finger Gripper. RGB observations are collected from Intel RealSense D435i cameras at a resolution of 640 × 480. Each model is trained with 50 demonstrations per task. For DILL and DILL-Current, the source-pretrained Task-Domain Encoder remains frozen: no real-world encoder fine-tuning or paired real-view data are used for encoder training.

Counterfactual shortcut diagnostic. Cup pointing and die placement test whether policies follow the commanded target when target color conflicts with viewpoint cues. We deliberately introduce a perfect correlation between target color and camera viewpoint in the training data: red-target instructions are paired only with the left view, whereas blue-target instructions are paired only with the right view. At test time, we reverse this assignment, evaluating red-target instructions from the right view and blue-target instructions from the left view. The instruction and desired manipulation remain unchanged; only the target-color–viewpoint pairing changes. Thus, both colors and both viewpoints are observed during training, but their test combinations are held out. Figure 9 illustrates the four instructions and their training and counterfactual test views.

This protocol makes viewpoint an unreliable cue for target selection: a policy that relies on the training association may act on the wrong-colored object instead of following the instruction. Counterfactual (CTF) success measures execution of the commanded task. Shortcut degree (SD) measures execution of the target associated with the observed viewpoint during training. It therefore measures a specific task-substitution failure rather than all unsuccessful trials.

Visual robustness. Shoe upright placement, tissue pulling, and laptop closing test whether policies retain instruction-following performance under visual changes. Figure 10(a) shows training demonstrations of the three tasks, and panel (b) organizes their shared evaluation conditions. Each task is evaluated under all three conditions, with its instruction and manipulation goal unchanged.

No perturbation uses the standard scene without added visual changes. Predefined perturbation uses held-out real instances of transformation families used during Task-Domain Encoder pretraining: viewpoint, background, lighting, and sensor noise. New perturbation uses cast shadows, dynamic backgrounds, and foreground clutter, which are absent from the encoder’s positive-pair construction. Thus, “predefined” refers to the transformation family, not to prior exposure to the real evaluation images. Comparing these conditions tests transfer both within and beyond the transformation families used to learn invariance. For DILL and DILL-Current, the source-pretrained encoder remains frozen throughout real-world policy fine-tuning and evaluation.

![](images/c3bf1a351f2c6d81dc943e9eb749ccbe9b73f414e5f93a0fc632bd947b559bd9.jpg)  
Figure 9: Real-world counterfactual color–viewpoint compositions. Training pairs red-target instructions only with the left view and blue-target instructions only with the right view, creating a spurious correlation between target color and viewpoint. Counterfactual testing swaps these pairings while preserving the instruction and desired manipulation. Each row shows five frames from a training demonstration and one example of the held-out test view.

## (a) Training demonstrations (front view)

“Pick up the shoe and stand it upright.”

![](images/4c9328be42144619c120009eb51fe3b591c1b7ef6dfd0dee28f648dd30b863b6.jpg)  
“Pull a tissue out of the box.”

![](images/f66045735379e4c41b08e23e1bf60cf99acae0468c48d504e4432ac1a7422f43.jpg)

![](images/73d248c94051660e328b83447c0ea331f1270459c37e825114b8b877da5f947c.jpg)

![](images/d0c5b070367e38a199093fd6bc501b9f791b1a71812e324ea77a6871766842af.jpg)

![](images/56c64f0e57b5a8110fb3de0c558b8bb2510971065cbc924d471e7442075b771a.jpg)

![](images/2938a2b0019fd4f97ce9648c53f3a777de70ad5b479538e3409f767335d43559.jpg)

![](images/efaec8124ba57db2e12125b56d5ece340cd9b9a78ddb0b671216cf749bed4257.jpg)

![](images/bdc071a5cc3c8d85d36771be806611efab808a91cc0d18b28d1cac54cbee4d27.jpg)

![](images/c68a103d30fcd2ca081cf32d2b19af6c9233e2fe75e465d7e18e36c67e2c6724.jpg)  
“Close the laptop.”

![](images/815982cd419cfaf0d9db2e6bd140cfaa6fb109576dd75866e02f8e5660e52dfe.jpg)

![](images/ef030c0ecdb0321ceb1a68b376e9741a37f8228138e01e936a38668f5c62c09c.jpg)

![](images/3aa9863b82538c8336e4d68a802df807c8c380c56447a051ed5eb321e56d4321.jpg)

![](images/f145392b52f27223a0704268fd75cb9626e2e2ddb92755031145ca34cbae04b0.jpg)

![](images/a9be07eed11ae866512ea41c8423f7cb1f56e8a0e286cab15b48024cd5c7a72b.jpg)

![](images/fef4855dc107cfaa3407791a881703ca943d3fb752ddbb21f1b6f8a1429c35c1.jpg)

## (b) Robustness evaluation

Each task is evaluated under all three conditions; representative scenes are shown below.

No perturbation

![](images/3cb85d3dc63923696df8df14990214d9f7a504738e14a23c228bc7f6c9e8cede.jpg)  
Standard scene  
Predefined perturbations

![](images/add7d283bc3257a213190c81607d83dc6e4a76ad283a328d27804fa010d5eae5.jpg)  
New perturbations  
Viewpoint

![](images/7164fb72e791c6163d30a745f98e6b408dd1451b2a261615bd56c7cb8bc3e3eb.jpg)  
Background

![](images/48bb42bce198625a17265fdb10ba176afd83f564274e15ebfb22a52d9888c60c.jpg)  
Dynamic background

![](images/e9149ad6ea369af616a1bced889a04a23feab792d61e7989b8aac75b7388517d.jpg)  
Families used in encoder pretraining  
Foreground clutter  
Families absent from encoder pretraining

Figure 10: Real-world visual robustness tasks and evaluation conditions. (a) Five ordered frames from the training demonstration for each task: shoe upright placement, tissue pulling, and laptop closing. (b) Representative evaluation scenes: no perturbation; viewpoint and background changes from predefined transformation families; and dynamic backgrounds and foreground clutter from new families absent from Task-Domain Encoder pair construction. All three tasks are evaluated under all three conditions, with their instructions and manipulation goals unchanged. Quantitative results are reported in Table 2.