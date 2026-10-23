# Learning Robot Behavior Priors from Non-Robotic Synthetic Data

Mateusz Odrowaz-Sypniewski, Michael Amir, Jasmine Bayrooti,

Ajay Shankar, and Amanda Prorok

Abstract— Real robot data, rich across tasks and morphologies, is often treated as the bottleneck for training robot learning models, including embodied foundation models that aim for zero-shot physical transfer. In this paper, we investigate the degree to which such models can acquire necessary structure using data from systems without embodiment. To this end, we test if inexpensive, procedurally generated synthetic data can yield useful priors before a model observes any real-world behavior. We study this through robot action prediction, a natural pretraining objective and a direct test of whether structure learned from arbitrary synthetic tasks transfers to embodied settings. We introduce three data generation pipelines that model mathematical abstractions of robotic control, and use them as pretraining priors for a Transformer. We demonstrate that a model trained purely on this non-robotic data attains non-trivial zero-shot prediction performance on unseen physical robot tasks via an in-context learning mechanism, and that our specialized control priors notably improve upon generic causal priors. We also compare against a compute-matched baseline trained on real robot data, and find that it outperforms our model but only on the datasets it was trained on. These findings suggest that procedurally generated, mathematical abstractions can yield useful robotic priors, offering a potential complement to expensive embodied data collection.

## I. INTRODUCTION

Can robots learn from non-robotic data? Data-driven learning underpins recent advances in robot control [1], [2], [3], with vision-language-action (VLA) models and generalist policies relying heavily on internet-scale video, simulation, and physical demonstrations [4], [5]. While diversifying training data across embodiments is a compelling research direction [6], [7], collecting physical demonstrations is expensive, and simulators merely shift the burden to domainspecific engineering. Instead, we explore whether it is possible to learn from generic, entirely unembodied sources.

Consider action prediction, the task of predicting held-out actions from a context of state-action tuples. Intuitively, the prediction task across different robot embodiments may share basic regularities tied to general notions of control. Prior exposure to these generic, abstract dynamics could serve as a useful starting point for downstream generalization. Indeed, by operating downstream of any visual encoders, such a method can be complementary to the grounding that, say, VLAs acquire from internet-scale data.

![](images/eea67b0392b7d20e591868fd6e019ebafa268afc9bfd416fb200711dd3b35383.jpg)

![](images/3593a5725cbd3c0379907a1eafcbc242691abb26089075cddcd78593fef32744.jpg)

![](images/3a19a97ed4ed0e64f89fed6a8625557f6d5dc674def78b178546c46e0ad9206d.jpg)

![](images/8a1286ddacc744fb82df00e5e628b0640877507b414386572b953c061e37aedc.jpg)  
Coordinate: q  
Fig. 1: Predictions made by our model (→) and ground-truth actions (→) in four sample environments. The model has only ever seen priors comprised of mathematical abstractions, and never any robot data (sim or real). Average error is shown in each panel; the simulation environments use discrete unit actions (peak norm: $\sqrt { 2 } )$ , while the peak magnitude for real environments is reported in [·]. Only the first two coordinates (q<sub>1</sub>, q<sub>2</sub>) are visualized.

In this work, we demonstrate that generic, mathematical abstractions of control—generated through policy-induced synthetic rollouts—can induce transferable priors for robot action prediction without requiring any embodied data. We implement this approach using Prior-Data Fitted Networks [8], which train Transformers on synthetic datasets sampled from task priors to enable zero-shot in-context predictions<sup>1</sup>. We hypothesize that sequential decision-making tasks require specialized structural priors, as opposed to, e.g., the domain-agnostic causal models used by methods such as TabPFN [9]. Our approach to this is to develop synthetic data generators based on procedurally generated Markov Decision Processes (MDPs), Contextual Bandits, and Linear Dynamical Systems (LDS).

By utilizing only mathematical abstractions rather than physical simulations [10], [11] or real-world data, our approach departs significantly from a typical embodied learning paradigm. We evaluate across a diverse suite of held-out robotic datasets (Figure 1), and find that while a generic tab ular prior struggles to transfer to real robot domains, a model pretrained purely on our synthetic control generators achieves zero-shot action prediction well above chance, despite having never observed a robot during training. We also evaluate our approach against a compute-matched baseline trained on real, embodied robot data. The two behave differently across the split: embodied training is far stronger on the tasks it was trained on (a gap that measures what our abstractions do not supply), while our model, for which every task is equally unseen, performs consistently across both groups and overtakes the baseline on the unseen task set.

Our key contribution is in demonstrating that inexpensive synthetic data generated using purely mathematical abstractions (no sim or real data) can pretrain in-context models—enough to achieve non-trivial accuracy on fully unseen tasks. This can serve as a useful complement to, and is not a replacement for, training/fine-tuning using in-task data when available.

## II. RELATED WORK

Behavioral cloning and robot data collection. Behavioral cloning (BC) is an approach for supervised learning from expert demonstrations, by learning a state-to-action mapping from ground-truth state-action pairs. BC has been applied across diverse robotic domains [12], [13], [14], [15], [16], [2]. Recent work has scaled this paradigm by training on larger datasets [6], adopting expressive architectures such as Transformers [17] and diffusion policies [18], building policies on pretrained vision-language models [1], and action chunking [3]. These approaches remain dependent on large, diverse robot datasets; insufficient scale or diversity has been shown to degrade performance [19], [7].

Simulators and sim-to-real robot learning. A popular approach to reducing the dependence on expensive real robot data is to use software that simulates agentic environments [20], [21] and robot physics [10], [22], [11], [23]. Through techniques such as GPU parallelization, these simulators offer cheap data for training and evaluating policies, and by modeling actuators and sensors they support transfer to physical robots. Our approach is complementary: rather than simulating a specific target robot or environment, we ask whether more abstract, non-robotic data can provide a useful prior for robotic tasks.

Robot foundation models and cross-embodiment learning. Recent work has demonstrated that combining training data from different robot embodiments can improve downstream performance. Generalist policies (e.g., π<sub>0</sub> [4], GR00T [5]) train on cross-embodiment datasets, occasionally supplementing with simulation data and synthetically generated videos. Our setting differs from these works in both scale and premise: we deliberately use cheap, procedurally generated synthetic data rather than large curated robot datasets, and we ask whether our model can learn useful priors before seeing any real robot behavior.

Synthetic data for in-context prediction. Prior-Data Fitted Networks (PFNs) [8] are trained to perform in-context learning: they take in context samples and query points from a previously unseen dataset, outputting a predictive distribution over the queries in a single forward pass. TabPFN [9] popularized this approach, achieving state-ofthe-art performance on real tabular data despite training exclusively on synthetic structural causal models. This purely synthetic pretraining endows the model with a rich prior that adapts in-context to unseen datasets. Recent work extends this paradigm to reinforcement learning, training Transformers to predict optimal actions from policy rollouts, using either hand-specified environments [24], [25] or procedurally sampled ones [26]. These works transfer within simulated decision-making; we instead ask whether priors built from non-robotic abstractions reach physical embodiments, and which abstractions matter, comparing bandits, MDPs, and linear dynamics against a generic causal prior.

## III. PROBLEM SETTING

We formulate robotic action prediction as an in-context classification problem [8]. A model $q _ { \theta }$ takes in a context set $D _ { \mathrm { c t x } } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ and query inputs $D _ { \mathrm { q u e r y } } = \{ x _ { j } \} _ { j = 1 } ^ { m }$ to compute predictive distributions $q _ { \theta } ( y _ { j } \mid \ x _ { j } , D _ { \mathrm { c t x } } ) ^ { \setminus }$ in a single forward pass, requiring no gradient updates. For datasets with continuous action spaces, we tokenize actions into $K = 6 4$ discrete bins via $K \cdot$ -means [1], [27], where y represents the nearest centroid index. The model minimizes a standard negative log-likelihood (NLL) objective over a dataset distribution p<sub>D</sub>:

$$
\mathcal { L } _ { \theta } = \mathbb { E } _ { ( D _ { \mathrm { c t x } } , D _ { \mathrm { q u e r y } } ) \sim p _ { D } } \left[ - \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \log q _ { \theta } ( y _ { j } \mid x _ { j } , D _ { \mathrm { c t x } } ) \right]\tag{1}
$$

We refer to a sampled pair $( D _ { \mathrm { c t x } } , D _ { \mathrm { q u e r y } } ) \sim p _ { D }$ as a task: a single dataset drawn from a prior, split into a context set and a set of held-out queries.

We evaluate performance using Accuracy and NLL. To measure how well the model recovers the underlying continuous action space from discrete labels, we introduce an additional Relative Centroid Distance (RCD) metric. Letting $\hat { y } _ { j } ~ = ~ \mathrm { a r g } \operatorname* { m a x } _ { y } q _ { \theta } ( y ~ \mid ~ x _ { j } , D _ { \mathrm { c t x } } )$ , RCD measures the Euclidean distance between predicted and true centroids $( c ( \hat { y } _ { j } )$ and $c ( y _ { j } ) )$ , normalized by the expected distance under uniform random guessing, averaged over queries:

$$
\mathrm { R C D } = \mathbb { E } _ { ( D _ { \mathrm { c t x } } , D _ { \mathrm { q u e r y } } ) \sim p _ { D } } \left[ \frac { 1 } { m } \sum _ { j = 1 } ^ { m } \frac { \| c ( \hat { y } _ { j } ) - c ( y _ { j } ) \| _ { 2 } } { \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \| c ( k ) - c ( y _ { j } ) \| _ { 2 } } \right]\tag{2}
$$

Under this metric, perfect prediction yields 0.0, and random guessing yields 1.0.

## IV. METHOD

We pretrain models on synthetic episodes drawn from a mixture of generators, producing in-context tasks in the $( D _ { \mathrm { c t x } } , D _ { \mathrm { q u e r y } } )$ format (Section III). To isolate the effect of the pretraining data on downstream robotic transfer, we freeze the model architecture, compute budget, and training procedure across all models within each experiment group (implementation details in Section VI-D). The only variable between models then is the specific set of generators, G, that determine the task distribution $p _ { D }$ . For each new task, we draw a generator $g \sim \operatorname { U n i f o r m } ( G )$ and sample one dataset, z-scoring its features to ensure scale consistency. Intuitively, a good generator should exploit some structural properties that underpin any robotic task.

## A. Tabular (Generic) Prior

As a baseline, we use the generic tabular prior from Hollmann et al. [9]. This generator uniformly selects an underlying mechanism (either a random structural causal model (SCM), a Bayesian neural network (BNN), or a Gaussian process (GP)) to construct feature-to-label mappings, yielding static, non-sequential datasets with no embodiment structure. We retain the original maximum of 10 classes for simulated RL experiments, but modify the generator to use quantile binning with $K \ : = \ : 6 4$ classes for robotic dataset experiments, matching the $K = 6 4$ action tokens of Section III.

## B. Synthetic Control Priors

We hypothesize that the task of robotic action prediction rests in part on structural regularities that do not inherently require a robot: actions are selected as a function of state, states evolve smoothly under those actions, and the selection is consistent with some latent objective. Therefore, we introduce three non-robotic generators, each capturing a complementary axis of control structure, and ask whether a model that has internalized them in a purely mathematical setting can exploit them when they recur in embodied data:

• Markov Decision Process (MDP): temporal, policyinduced, goal-directed sequential decision-making, where the state-to-action mapping reflects long-horizon value rather than any immediately observable signal.

• Contextual Bandit: state-conditioned action selection, single-step, isolating the state-action association with the temporal structure removed entirely.

• Linear Dynamical System (LDS): linear continuous dynamics with high temporal autocorrelation, exposing the model to the smooth state evolution that is characteristic of physical systems, which the tabular MDPs lack by construction.

Together, these span a broad spectrum of fundamental control abstractions: from localized, single-step optimization (Bandits) to smooth, temporally correlated dynamics (LDS) and complex, multi-step planning (MDP). Crucially, none of them model a kinematic chain, a sensor, or a task. We randomly permute the feature dimensions of every sampled dataset, so no input index carries consistent meaning across tasks; the model must infer the relevant structure from the context set at inference, exactly as it must for an unseen robot. All three generators sample a variable number of discrete actions per dataset, up to a cap $A _ { m a x }$ matching the class count of the tabular prior in the same experiment group: $A _ { m a x } = 1 0$ for the simulated RL experiments (VMAS) and $A _ { m a x } = 6 4$ for the robotic ones.

For MDP priors, we generate episodes from diverse, randomly sampled tabular MDPs. Each MDP has 30–300 states randomly distributed in $[ 0 , 1 ] ^ { d }$ (with dimension $d \in$ $[ 1 0 , 4 0 ] )$ and $4 - A _ { m a x }$ discrete actions acting as small spatial displacements. Reward and continuous transition dynamics are jointly sampled from a multi-output Gaussian-process prior (via random Fourier features) to ensure smooth correlation across the state-action space. Next-states are snapped to the nearest discrete state to preserve the tabular structure. We solve the MDPs via truncated backward value iteration, rolling out trajectories under Boltzmann policies with random temperatures to vary exploration. Model input is the randomly permuted raw state; the label is the discrete action.

The contextual bandit generator isolates single-step, state-conditioned action selection, removing temporal dynamics entirely. Datasets consist of independent state-action pairs rather than episodic rollouts. We sample random contextual bandits with state dimension $d _ { s } ~ \in ~ [ 4 , 1 0 0 ]$ and $4 - A _ { m a x }$ discrete actions, each parameterized by a feature vector $( d _ { a } \in [ 4 , 1 6 ] )$ . A hidden reward function $f ( s , a )$ is constructed as a smooth bilinear form (of random rank) over random-Fourier-feature embeddings of states and actions, ensuring similar inputs yield similar rewards. We construct state-action pairs by sampling states i.i.d. and then drawing an action for each from a Boltzmann distribution over $f ( s , \cdot )$ with a random temperature. The model input concatenates and randomly permutes the state and chosen action features, with the latter perturbed by Gaussian noise so they do not identify the chosen action, and the labels are the chosen actions at held-out states.

The LDS generator produces continuous dynamics with high temporal autocorrelation, exposing the model to the smooth state evolution characteristic of physical systems— a structural bias absent in the tabular MDPs. We sample random linear dynamical systems where the state (dimension $d \in \ [ 4 , 1 0 0 ] )$ evolves as $x _ { t + 1 } = A x _ { t } + b _ { a } + \varepsilon$ where A is a diagonal matrix with decay rates near one (inducing strong autocorrelation), $b _ { a }$ is a fixed displacement for each of the $4 - A _ { m a x }$ discrete actions, and ε is small Gaussian noise. We roll out multiple episodes per system using a randomly sampled linear-softmax policy, providing diverse policy demonstrations over shared dynamics. The input is the raw state; the label is the discrete action.

## V. EVALUATIONS

We evaluate the models’ ability to predict next actions (given some context) on a variety of simulated and real robot data. For each dataset, we sample 512 windows of up to 2000 timesteps. Each of these windows is split in half, and we predict each half from the context of the other via crossfitting. We report the mean and standard error across three random seeds, selected at the checkpoint with highest mean accuracy over held-out data from the training distribution.

## A. Datasets

We evaluate across a spectrum of task domains: (i) six simulated multi-agent RL envs (VMAS [20]); (ii) five simulated manipulation tasks (RoboMimic [28]); and (iii) four real-world robotic datasets across manipulator, wheeled, legged, and aerial embodiments: DROID [29], A2D2 [30], GrandTour [31], and a two-multicopter flight dataset used in downwash studies [32]. Every dataset is mapped to the format of Section III: states/observations become inputs $x _ { t }$ actions become labels $y _ { t }$ (continuous actions are discretized into $K = 6 4$ centroids via K-means). A task is a window of timesteps from one dataset, split into $D _ { \mathrm { c t x } }$ and $D _ { \mathrm { q u e r y } }$

## B. Evaluation of Priors

We test three training configurations: a Tabular Prior (only the generic causal prior [9]), Control Priors (only our specialized synthetic priors: MDP, Contextual Bandit, LDS), and Tabular + Control (all four priors combined).

![](images/c57f029b44c431efb32ae4e6da5e5d5008c930f18356b6fd14756922eacb97af.jpg)

![](images/87754754ed30e4ab0f9ffdb3c6701b28a7c9ff00eb089d88dc1559cb7eabc608.jpg)

![](images/7d24ca69654b8803aec87bf15697ef03d7796bc348021de3883e4d12453194c3.jpg)  
Fig. 2: Performance of the model’s action predictions when trained using different priors for VMAS tasks and various robot datasets. We report NLL and accuracy measures in each case, and additionally report RCD when the action space is continuous. A model pretrained with any prior will already perform better than random chance (dotted line); though our control priors show a further improvement across all metrics.

![](images/f6c3bdc91e3000924b23fb5a0e6a1bc22e49133236cca0d261d34d892a36fc1e.jpg)

![](images/d023bf76a2fc4efae08d77262152a756bf5148f1ee29798472508ee0652c2e84.jpg)

TABLE I: Action prediction performance on each real-world robotic dataset for models trained using different priors, reporting mean and standard error across 3 seeds. The mean column is the grouped value plotted in Figure 2 (Real Robot); best per column is listed in bold.  
TABLE II: Inference speed and training time for all models remain roughly consistent, with some variability due to generators.
<table><tr><td>Prior</td><td>A2D2</td><td>DROID</td><td>Aerial</td><td>GrandTour</td><td>Mean</td></tr><tr><td>RCD ↓</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Tabular prior</td><td> $0 . 8 6 7 \pm 0 . 0 4 0$ </td><td> $0 . 8 9 8 \pm 0 . 0 3 1$ </td><td> $0 . 9 2 3 \pm 0 . 0 3 5$ </td><td> $0 . 9 3 1 \pm 0 . 0 8 8$ </td><td> $0 . 9 0 5 \pm 0 . 0 3 5$ </td></tr><tr><td>Control priors</td><td> $0 . 6 7 1 \pm 0 . 0 3 0$ </td><td> $\mathbf { 0 . 4 9 9 \mathop { \pm } 0 . 0 0 4 }$ </td><td> $\mathbf { 0 . 8 0 6 } \pm 0 . 0 0 3$ </td><td> $\mathbf { 0 . 8 4 9 \mathop { \pm 0 . 0 1 0 } }$ </td><td> $\mathbf { 0 . 7 0 6 } \pm 0 . 0 0 8$ </td></tr><tr><td>Tabular + control</td><td> $\mathbf { 0 . 6 6 3 \pm 0 . 0 2 8 }$ </td><td> $0 . 5 1 1 \pm 0 . 0 0 5$ </td><td> $0 . 8 2 1 \pm 0 . 0 0 2$ </td><td> $0 . 8 5 9 \pm 0 . 0 1 0$ </td><td> $0 . 7 1 4 \pm 0 . 0 0 5$ </td></tr><tr><td> $N L L \downarrow$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Tabular prior</td><td> $4 . 2 1 3 \pm 0 . 0 2 5$ </td><td> $4 . 1 9 2 \pm 0 . 0 1 3$ </td><td> $4 . 2 3 0 { \scriptstyle \pm 0 . 0 1 3 }$ </td><td> $4 . 1 4 4 \pm 0 . 0 0 7$ </td><td> $4 . 1 9 4 \pm 0 . 0 0 3$ </td></tr><tr><td>Control priors</td><td> $3 . 8 0 5 \pm 0 . 0 2 6$ </td><td> $4 . 0 4 9 \pm 0 . 0 0 7$ </td><td> $4 . 1 2 6 \pm 0 . 0 1 4$ </td><td> $3 . 9 7 8 \pm 0 . 0 0 2$ </td><td> $3 . 9 9 0 \pm 0 . 0 0 5$ </td></tr><tr><td>Tabular + control</td><td> $\mathbf { 3 . 6 5 8 \pm 0 . 0 6 3 }$ </td><td> $\mathbf { 3 . 9 9 8 \pm 0 . 0 2 0 }$ </td><td> $\mathbf { 4 . 0 3 1 } \pm 0 . 0 3 6$ </td><td> $\mathbf { 3 . 9 6 9 \mathop { \pm 0 . 0 2 6 } }$ </td><td> ${ \bf 3 . 9 1 4 \pm 0 . 0 2 8 }$ </td></tr><tr><td>Accuracy ↑</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Tabular prior</td><td> $0 . 0 3 7 \pm 0 . 0 0 8$ </td><td> $0 . 0 4 6 \pm 0 . 0 1 1$ </td><td> $0 . 0 3 8 \pm 0 . 0 1 2$ </td><td> $0 . 0 3 2 \pm 0 . 0 0 4$ </td><td> $0 . 0 3 9 \pm 0 . 0 0 9$ </td></tr><tr><td>Control priors</td><td> ${ \bf 0 . 2 4 0 \pm 0 . 0 0 3 }$ </td><td> ${ \bf 0 . 1 6 9 } \pm 0 . 0 0 3$ </td><td> $\mathbf { 0 . 1 4 1 \pm 0 . 0 0 0 }$ </td><td> $0 . 1 1 5 \pm 0 . 0 0 1$ </td><td> $\mathbf { 0 . 1 6 6 \pm } 0 . 0 0 0$ </td></tr><tr><td>Tabular + control</td><td> $0 . 2 3 4 \pm 0 . 0 1 3$ </td><td> $0 . 1 6 7 \pm 0 . 0 0 2$ </td><td> $0 . 1 3 6 \pm 0 . 0 0 1$ </td><td> ${ \bf 0 . 1 2 1 } \pm 0 . 0 0 2$ </td><td> $0 . 1 6 5 \pm 0 . 0 0 4$ </td></tr></table>

<table><tr><td>Inference Speed ↓ (context len=1000) Tabular prior Control priors Tabular + Control</td><td>7.22 ms/query</td></tr><tr><td>Training Time ↓</td><td></td></tr><tr><td>Any model</td><td>≈ 30 hours</td></tr></table>

VMAS Tasks. The VMAS tasks have 9 native discrete actions (2D planar velocities), so for all priors we use the tabular prior’s default class count of 10. From Figure 2, we see that all models achieve ${ > } 4 0 \%$ accuracy on this 9- action problem (substantially higher than random chance). The variants utilizing control priors outperform the tabularonly baseline by 7 to 9 accuracy percentage points (0.58 and 0.56 vs. 0.49), a margin consistent across all three seeds. In particular, the variant trained using only control priors achieves the highest accuracy and lowest NLL, indicating that our control abstractions transfer effectively to RL tasks in simulation.

![](images/f758c772cd013a9c4ccc8e1df87182b76b3f3032e7edaa24f7f674e6a4be80c2.jpg)  
Fig. 3: Ablation of the individual control priors on RoboMimic and realworld robotic datasets, alongside their combination and the tabular prior.

Robotic Data Tasks. For the simulated RoboMimic dataset and four real-world robot datasets (grouped together), we raise the class count of all priors to 64 to match the tokenized action spaces. In contrast to VMAS, performance of the models on the robotic environments diverges more visibly (Figure 2). The tabular-only prior drops to within a few points of the $1 / 6 4 ~ \approx ~ 0 . 0 1 6$ chance rate on realrobot datasets, whereas mixtures incorporating control priors achieve stronger results. While the absolute accuracy on these 64-class tasks remains modest (0.16 and 0.17 on RoboMimic and real robot data respectively for the control priors), achieving any meaningful predictive signal without exposure to robotic training data is already a noteworthy result. Furthermore, accuracy and NLL rankings largely align with our RCD metric, suggesting that the models capture the continuous spatial structure underlying the discrete labels. Ultimately, the Control generators generally match or exceed the Tabular + Control mixture (with the mixture ahead only on NLL), indicating that the control generators largely subsume the tabular prior’s contribution and are not merely complementary. We report a full breakdown of each of the priors on each dataset in Table I. Inference remains real-time for all models, and their training times are also comparable since their architectures are identical (Table II).

## C. Ablations

To isolate the impact of each control prior, we train models exclusively on MDP, LDS, or Contextual Bandit. As shown in Figure 3, our three generators do not contribute equally. LDS is the strongest individual prior and matches the full mixture on the real-robot suite, while MDP and Bandit each fall behind the tabular baseline on at least one metric. Note that LDS is also the only generator whose states evolve smoothly and exhibit high autocorrelation in continuous space, which suggests that what transfers is local temporal continuity in the state-action stream rather than long-horizon value structure (MDP) or state-action association alone (Bandit). Combining all three generators still gives the best NLL and accuracy on RoboMimic, and generally remains competitive in other task sets.

![](images/4741d2ed303ee4b8e4d5fb7a524aaee104ba6c9b69e729f9b5ee2834cc05857d.jpg)  
Fig. 4: Comparison of synthetic pretraining (control priors) against indomain embodied training on VMAS tasks, split by whether the baseline saw the task during training.

## D. Models Trained on Embodied Data

To ground our metrics against embodied baselines, we train two models strictly on embodied data: a VMAS Baseline trained on three VMAS environments (navigation, balance, reverse transport), and a Robotic Baseline trained on three RoboMimic tasks (can, lift, square) and two realworld datasets (A2D2, DROID). This way, each baseline is trained on a subset of the evaluation suite, with the rest held out. This allows us to assess the value of our synthetic pretraining in a regime where in-domain embodied data is available and in one where it is not. Our model remains zero-shot across all tasks.

On VMAS (Figure 4), the embodied baseline attains an accuracy of 0.83 on its training environments against our 0.55. This gap is a clear measurement of the task-specific structure that embodied data provides but generic control priors do not. However, the ranking is reversed on the baseline’s heldout tasks (0.27 vs. 0.61); the synthetic pretraining appears to generalize better in this case, despite the baseline having trained on environments that share dynamics and state-space characteristics with the held-out tasks.

![](images/5b65879bd58fc8ed8732da6e08429736510794b5c1d847a1737ef627d12b03fe.jpg)  
Fig. 5: Comparison of synthetic pretraining against embodied training on RoboMimic and real-world robotic datasets, split by whether the baseline saw the dataset during training.

Results for the robotic datasets (Figure 5) complement the simulated RL findings: the real-data baseline excels on its training datasets but falls to chance on the rest. Meanwhile, our model, which has seen none of these embodiments, stays well above chance throughout, and even overtakes the baseline on the datasets it did not train on. What the baseline learns is therefore largely specific to the embodiments it saw. We caution, however, that the severity of this collapse is not necessarily a generalizable statement about training models on (cross-)embodied data—instead, it may well reflect the limitations of our current compute budget (see scale limitations, Section VI-B). Nevertheless, we remark positively upon the consistency exhibited by a model trained only using synthetic priors on seen and unseen tasks.

![](images/16fccf5ed32cd573f5a7ca899c6efb971a65037b83607644be78ad42cc6cdbc1.jpg)  
Fig. 6: Accuracy of the control priors model against context size, averaged over three seeds.

## E. Context Scaling

We perform additional analyses to examine how the accuracy of our synthetic control priors model changes depending on the number of samples in the context. Figure 6 shows the results of this experiment for context sizes {100, 200, . . . , 1000}, with 1000 being the context size used in our main experiments. We observe that accuracy increases monotonically with context size in all three domains, indicating that the model extracts task-specific structure from the context rather than falling back on some contextindependent decision rule. VMAS (simulation) largely saturates by roughly 800 in-context samples, while the realrobot curve is still rising at 1000; our reported real-robot accuracies are therefore a lower bound on what the prior can support given more context. In practice, this acts as a tradeoff: increasing accuracy using more context is possible at the expense of compute resources and inference latency.

## VI. CONCLUSIONS & ADDENDA

“Can robots learn from non-robotic data?” We pretrained in-context models exclusively on synthetic data from mathematical abstractions (such as MDPs and LDS), and find that they exhibit non-trivial evidence of zero-shot transfer and applicability in robotic tasks. Such models have never observed any robot, simulated or real, and yet they are able to predict future actions reasonably in a range of eleven simulated and four real-world task sets (Figure 2). Evidently, using such specialized synthetic priors offers an advantage over generic causal priors (Figure 3). We contend that these provide an encouraging proof of concept. Future work should certainly build on this to explore and employ this capability further. To aid this, we present a few critical points of discussion.

![](images/8ef1e1a724f36b84bbc3c7192d94eb244e5330e653c5739a0b59da37aed2222d.jpg)  
Fig. 7: Six sample open-loop rollouts of 300 timesteps of ground-truth and predicted actions from a real-world multicopter flight. The solid circles and squares denote the start and the end positions, respectively.

The absolute accuracy of our model is relatively modest, ranging from ≈ 58 % in simulations (where random chance would be ≈ 11 %) down to ≈ 17 % in real-world scenarios (random chance ≈ 1.6 %). Baselines trained on simulated or real-world data (Figure 4 and Figure 5) invariably perform much better when they are evaluated over task sets they have seen during training. However, we find most encouragement in cases when we evaluate on tasks that are outside the training set but share similarities in dynamics, environment, and their signals. A model trained using only abstract synthetic priors appears to exhibit better generalization and consistency in its performance.

These results position abstract synthetic data as a scalable complement to existing pipelines: acquiring effective structural priors may not require expensive physical experiences or embodiment-specific simulations. Progress toward zero-shot physical AI may therefore depend less on the volume of embodied data than current scaling trends assume.

## A. Open-Loop State Predictions

Our objective of predicting future actions is useful in rolling out future states of a robot, possibly for other downstream planning and optimization tasks. To assess the raw quality of the predictions, in Figure 7, we show openloop trajectory rollouts that na¨ıvely use a chunk of predicted actions for 300 timesteps. We roll out the 2D states (positions in North and East) of a multicopter using zero-order forwardstep Euler integration, which is typically a poor approximation for high-order ODEs. In addition, note that an open-loop chunk is likely to accrue substantial errors since the model does not see new (projected) states. Nevertheless, we see that the projected final states after 300 timesteps (≈ 15 s) are reasonably close to the ground-truth states (mean error $< 0 . 4 8 \mathrm { m } )$ . The worst-case error is ≈ 1 m, which may be inadequate for some applications, but still suggests that the model is able to make sensible in-context predictions.

## B. Limitations on Scale

We observed in Figure 5 that baselines trained on some robotic tasks fail to generalize to tasks outside the training set, sometimes even performing worse than random chance. A failure in cross-embodiment transfer is not entirely surprising; however, the magnitude of the failure does warrant a qualification vis-a-vis the size/scale of our model(s). Recent \` work with large models has shown noteworthy capabilities across embodiments [5], [4]. Our model variants (used in ablations, baselines, etc.) are substantially smaller. Due to computational constraints, we compute-matched all baselines by employing the original, smaller version of TabPFN. While this enables us to train models capable of real-time inference (≈ 7 ms/query), their reduced capacity may explain the apparent lack of generalization.

Nevertheless, as we noted earlier, the equal-capacity models trained with synthetic priors retain their performance better. Future work must revisit these experiments using the newer, larger TabPFN [33] or other models with an expanded training budget. Since training at scale may constitute a fundamentally different learning paradigm, verifying whether the advantages of synthetic control priors persist for larger foundation models is a critical next step.

## C. Counter-Intuitive Fine-Tuning

We have so far referred to our priors as “pretraining” the model, with the understanding that downstream tasks might utilize the capabilities in different ways. A natural supposition is then to try to “fine-tune” such a model on (possibly limited) realistic, domain-specific data. Our initial hypothesis was that with limited in-domain data, fine-tuning a model pretrained on our general control priors would yield better performance than training a new model from scratch.

To test this, we fine-tuned our pretrained model on subsets of each robotic dataset and compared its performance to a model trained from scratch on the same data. While both models’ performance improved monotonically with subset size, their rates of improvement differed. Surprisingly, although the pretrained model naturally outperformed the from-scratch baseline on small subsets, the latter achieved higher performance on some datasets relatively quickly, after seeing as few as 1,000 to 5,000 in-domain samples.

A detailed performance evaluation is beyond the scope here; however, these initial findings suggest that the two separate training phases may not admit a na¨ıve ‘chaining’. This potential conflict likely stems from their fundamentally different objectives: pretraining aims for broad generalization by viewing each synthetic dataset only once, whereas finetuning optimizes for a given dataset through repeated exposure. While we still believe synthetic pretraining has a place in the pipeline of foundation models, additional research is required to effectively combine it with in-domain training data.

## D. Datasets, Implementation, and Reproducibility

We will release all datasets and packaged code used for this work upon acceptance.

TABLE III: Evaluation datasets. $d _ { s }$ and $d _ { a }$ are the state and action dimensions. VMAS actions are natively discrete; all others are tokenized into K = 64 bins via K-means.
<table><tr><td>Dataset</td><td>Platform</td><td> $d _ { s }$ </td><td> $d _ { a }$ </td><td>Action space</td></tr><tr><td>Simulated</td><td></td><td></td><td></td><td></td></tr><tr><td>VMAS [20]</td><td>2D multi-agent</td><td>19-27</td><td>9</td><td> $3 \times 3$  motion grid</td></tr><tr><td></td><td>RoboMimic [28] 1-2× 7-DoF Panda 53–127 7-14 ∆EE pose, gripper</td><td></td><td></td><td></td></tr><tr><td>Real-world</td><td></td><td></td><td></td><td></td></tr><tr><td>A2D2 [30]</td><td>passenger vehicle</td><td>9</td><td>3</td><td>steer, accel., brake</td></tr><tr><td>DROID [29]</td><td>7-DoF Franka</td><td>14</td><td>7</td><td>EE target, gripper</td></tr><tr><td>Aerial [32]</td><td>multicopter pair</td><td>24</td><td>4</td><td>body commands</td></tr><tr><td>GrandTour [31]</td><td>ANYmal quadruped</td><td>50</td><td>3</td><td> $v _ { x } , v _ { y } , \omega _ { z }$ </td></tr></table>

Table III summarizes the evaluation suite. From VMAS we use six scenarios spanning cooperative navigation (navigation, passage, flocking), pursuit (simple tag), and collective transport (reverse transport, balance), rolling out pretrained policies to record (state, action) trajectories; RoboMimic provides proficient-human demonstrations on five tasks (lift, can, square, transport, tool hang); transport has 127 observation dimensions, which we truncate to the first 100 to match our model’s input limit; from DROID we use one million timesteps. The downwash data records two multicopters in close proximity, with one subjected to the other’s aerodynamic wake, and its state stacks both vehicles positions and velocities over two consecutive timesteps.

Implementation. We use the official implementation of Hollmann et al. [9]<sup>2</sup>: a 12-layer Transformer with 4 attention heads. We use this earlier release rather than [33] both for compute reasons and because the latter does not release its synthetic data-generation pipeline, which our method requires. We deviate from their setup only in dataset size (2048 samples, up from 1024) and training budget (15M datasets for the robotic experiments, 9M for VMAS). The embodied baselines use the same architecture and budget with the generators replaced by real data: each roboticbaseline training set is a random contiguous window of 2048 timesteps from a training split of one of its five datasets, labeled with the evaluation K-means codebook, while the VMAS baseline samples fresh rollouts.

## ACKNOWLEDGMENT

Portions of the research code were written with assistance from Claude Code. All generated code was reviewed and tested by the authors.

## REFERENCES

[1] A. Brohan, N. Brown et al., “RT-2: Vision-language-action models transfer web knowledge to robotic control,” in arXiv preprint arXiv:2307.15818, 2023.

[2] M. Bojarski, D. D. Testa et al., “End to end learning for self-driving cars,” 2016. [Online]. Available: https://arxiv.org/abs/1604.07316

[3] T. Z. Zhao, V. Kumar, S. Levine, and C. Finn, “Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware,” in Proceedings of Robotics: Science and Systems, 2023.

[4] K. Black, N. Brown et al., “π<sub>0</sub>: A vision-language-action flow model for general robot control,” 2024. [Online]. Available: https://arxiv.org/abs/2410.24164

<sup>2</sup>https://pypi.org/project/tabpfn/0.1.11/

[5] NVIDIA, J. Bjorck et al., “GR00T N1: An open foundation model for generalist humanoid robots,” 2025. [Online]. Available: https://arxiv.org/abs/2503.14734

[6] Open X-Embodiment Collaboration, A. O’Neill et al., “Open X-Embodiment: Robotic learning datasets and RT-X models,” https: //arxiv.org/abs/2310.08864, 2023.

[7] F. Lin, Y. Hu, P. Sheng, C. Wen, J. You, and Y. Gao, “Data scaling laws in imitation learning for robotic manipulation,” arXiv preprint arXiv:2410.18647, 2024.

[8] S. Muller, N. Hollmann, S. P. Arango, J. Grabocka, and F. Hutter,¨ “Transformers can do bayesian inference,” 2024. [Online]. Available: https://arxiv.org/abs/2112.10510

[9] N. Hollmann, S. Muller, K. Eggensperger, and F. Hutter, “TabPFN: ¨ A transformer that solves small tabular classification problems in a second,” in The Eleventh International Conference on Learning Representations, 2023. [Online]. Available: https://openreview.net/ forum?id=cp5PvcI6w8

[10] NVIDIA, “Isaac Sim.” [Online]. Available: https://github.com/ isaac-sim/IsaacSim

[11] E. Todorov, T. Erez, and Y. Tassa, “MuJoCo: A physics engine for model-based control,” in 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, 2012, pp. 5026–5033.

[12] T. Zhang, Z. McCarthy, O. Jow, D. Lee, X. Chen, K. Goldberg, and P. Abbeel, “Deep imitation learning for complex manipulation tasks from virtual reality teleoperation,” in 2018 IEEE International Conference on Robotics and Automation (ICRA), 2018, pp. 5628– 5635.

[13] P. Florence, L. Manuelli, and R. Tedrake, “Self-supervised correspondence in visuomotor policy learning,” 2020. [Online]. Available: https://hdl.handle.net/1721.1/135421

[14] E. Jang, A. Irpan, M. Khansari, D. Kappler, F. Ebert, C. Lynch, S. Levine, and C. Finn, “BC-Z: Zero-shot task generalization with robotic imitation learning,” in Proceedings of the 5th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, A. Faust, D. Hsu, and G. Neumann, Eds., vol. 164. PMLR, 2022, pp. 991–1002. [Online]. Available: https://proceedings.mlr.press/v164/jang22a.html

[15] A. Loquercio, A. I. Maqueda, C. R. D. Blanco, and D. Scaramuzza, “Dronet: Learning to fly by driving,” IEEE Robotics and Automation Letters, 2018.

[16] D. A. Pomerleau, “ALVINN: an autonomous land vehicle in a neural network,” in Proceedings of the 2nd International Conference on Neural Information Processing Systems, ser. NIPS’88, 1988, pp. 305– 313.

[17] S. Dasari and A. Gupta, “Transformers for one-shot visual imitation,” in Proceedings of the 2020 Conference on Robot Learning, ser. Proceedings of Machine Learning Research, J. Kober, F. Ramos, and C. Tomlin, Eds., vol. 155. PMLR, 2021, pp. 2071–2084. [Online]. Available: https://proceedings.mlr.press/v155/dasari21a.html

[18] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” Int. J. Rob. Res., vol. 44, no. 10–11, pp. 1684–1704, 2025.

[19] A. Brohan, N. Brown et al., “RT-1: Robotics transformer for real-world control at scale,” in arXiv preprint arXiv:2212.06817, 2022.

[20] M. Bettini, R. Kortvelesy, J. Blumenkamp, and A. Prorok, “VMAS: A vectorized multi-agent simulator for collective robot learning,” The 16th International Symposium on Distributed Autonomous Robotic Systems, 2022.

[21] M. Towers, A. Kwiatkowski et al., “Gymnasium: A standard interface for reinforcement learning environments,” arXiv preprint arXiv:2407.17032, 2024.

[22] M. Mittal, P. Roth et al., “Isaac Lab - A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning,” arXiv preprint arXiv:2511.04831, 2025. [Online]. Available: https://arxiv. org/abs/2511.04831

[23] Z. Jiang, Y. Xie, K. Lin, Z. Xu, W. Wan, A. Mandlekar, L. Fan, and Y. Zhu, “DexMimicGen: Automated data generation for bimanual dexterous manipulation via imitation learning,” in 2025 IEEE International Conference on Robotics and Automation (ICRA), 2025.

[24] M. Laskin, L. Wang et al., “In-context reinforcement learning with algorithm distillation,” 2022. [Online]. Available: https: //arxiv.org/abs/2210.14215

[25] J. N. Lee, A. Xie, A. Pacchiano, Y. Chandak, C. Finn, O. Nachum, and E. Brunskill, “Supervised pretraining can learn in-context reinforce-

ment learning,” in Proceedings of the 37th International Conference on Neural Information Processing Systems, 2023.

[26] F. Wang, P. Shao, Y. Zhang, B. Yu, S. Liu, N. Ding, Y. Cao, Y. Kang, and H. Wang, “Towards large-scale in-context reinforcement learning by meta-training in randomized worlds,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. [Online]. Available: https://openreview.net/forum?id=b6ASJBXtgP

[27] N. M. Shafiullah, Z. Cui, A. A. Altanzaya, and L. Pinto, “Behavior transformers: Cloning k modes with one stone,” in Advances in Neural Information Processing Systems, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, Eds., vol. 35. Curran Associates, Inc., 2022, pp. 22 955–22 968. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/2022/file/ 90d17e882adbdda42349db6f50123817-Paper-Conference.pdf

[28] A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Mart´ın-Mart´ın, “What matters in learning from offline human demonstrations for robot manipulation,” in arXiv preprint arXiv:2108.03298, 2021.

[29] A. Khazatsky, K. Pertsch et al., “DROID: A large-scale inthe-wild robot manipulation dataset,” 2025. [Online]. Available: https://arxiv.org/abs/2403.12945

[30] J. Geyer, Y. Kassahun et al., “A2D2: Audi Autonomous Driving Dataset,” arXiv preprint arXiv:2004.06320, 2020.

[31] J. Frey, T. Tuna, F. Fu, K. Patterson, T. Xu, M. Fallon, C. Cadena, and M. Hutter, “GrandTour: A legged robotics dataset in the wild for multi-modal perception and state estimation,” arXiv preprint arXiv:2602.18164, 2026.

[32] H. Smith, A. Shankar, J. Gielis, J. Blumenkamp, and A. Prorok, “SO(2)-equivariant downwash models for close proximity flight,” IEEE Robotics and Automation Letters, vol. 9, no. 2, pp. 1174–1181, 2023.

[33] N. Hollmann, S. Muller, L. Purucker, A. Krishnakumar, M. K¨ orfer,¨ S. B. Hoo, R. T. Schirrmeister, and F. Hutter, “Accurate predictions on small data with a tabular foundation model,” Nature, 2025. [Online]. Available: https://www.nature.com/articles/s41586-024-08328-6