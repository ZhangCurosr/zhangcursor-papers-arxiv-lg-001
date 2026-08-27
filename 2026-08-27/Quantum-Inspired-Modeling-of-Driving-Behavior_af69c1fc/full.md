# Quantum-Inspired Modeling of Driving Behavior

Mohammad Elayan<sup>⋆</sup>, Omid Armantalab<sup>⋆</sup>, Wissam Kontar<sup>⋆,∗</sup>

<sup>⋆</sup>Civil & Environmental Engineering, University of Nebraska–Lincoln, Lincoln, NE, USA

## Abstract

Driver behavior is heterogeneous, context-dependent, and changes over time, and these properties shape the trafic phenomena we observe. Most models, however, fix in advance which behavioral variables interact and how. Behavior outside that form is absorbed as noise, while models flexible enough to capture it tend to lose interpretability. We introduce a quantum-inspired representation of driver behavior that combines properties usually treated separately or in part: it is continuous, probabilistic, context-dependent, history-dependent, and represents interactions among behavioral variables as learned from data. Each driver is encoded as an evolving density matrix, providing a unified representation of behavioral uncertainty, temporal evolution, and context-dependent behavioral variation. Trained without supervision on the I-24 MOTION dataset, the framework recovers three interpretable driving profiles representing three regimes: free flow, transition, and congestion. The profiles capture the behavioral range of the data and the smooth transitions drivers make between regimes as conditions change. The same representation also reproduces known macroscopic phenomena, aligning with the fundamental diagram and reproducing hysteresis loops. We also show how the representation supports practical use: it supplies context-dependent parameters to classical car-following models, and gives an autonomous vehicle a live behavioral read of the surrounding drivers with a short-horizon forecast of their motion. The framework points toward models of traffic that are interpretable and trustworthy by construction. We release an open-source toolkit on GitHub spanning data processing, training, inference, and analysis.

Keywords: Quantum Methods, Heterogeneity, Driving Behavior, Density Matrices, Trafic Flow

## 1. Introduction

Decades of trafic flow research have shown that no single behavioral description fits all drivers, all conditions, or all moments in a trip. Drivers difer, they change over time, and they respond to their surroundings. Capturing all three in a single representation has remained dificult, not because any one of them is unknown, but because doing so usually forces the modeler to commit in advance to which variables interact, in which direction, and through which functional form.

Most existing approaches commit to a representational form before estimation. Calibrated car-following models adopt a specific functional dependence of the response on a few low-order kinematic inputs, such as speed and gap; latent-class and mixture models choose a finite set of behavioral classes, such as “aggressive” or “timid”; regime-switching models pre-specify a discrete set of regimes, such as free flow, synchronized flow, and congestion, together with the kinematic and contextual variables that govern transitions between them. These specifications are necessary for estimation, but they also bound what a model can subsequently represent: the relational form is set before estimation begins, and behavior that does not fit it is treated as noise or absorbed into the residual.

We propose a diferent starting point. Behavioral heterogeneity is dificult to pre-specify: it is high-dimensional, shaped by individual preferences and experience, and expressed partly through rare events that a fixed functional form is unlikely to anticipate. These properties make it a natural candidate for free-form discovery, in which the data, rather than the modeler, reveal how the behavioral variables relate. The bottleneck, we argue, is not the choice of variables: familiar loworder kinematic quantities, such as speed and headway, remain adequate to describe what a driver is doing at a moment. What limits existing representations is how those variables are permitted to interact and how the resulting state is formed. We therefore keep a small set of behavioral variables, together with a few contextual variables describing the surrounding trafic, but change how they are represented. We lift the behavioral variables into a higher-dimensional nonlinear feature space using random Fourier features (RFF), and represent each driver’s evolving state in that space as a density matrix. The lift makes interactions among the variables accessible without specifying them in advance, the density matrix carries information forward as a continuous, probabilistic state, and the context determines how strongly each behavioral profile contributes to that state. Figure 1 sketches the properties the representation is built to combine.

![](images/0ff13e2cb89e3b8dcc44ae7e6888b90e930587ce5fbe6c801fd91970aa528f51.jpg)  
Figure 1: Conceptual motivation for the proposed representation. Drivers difer from one another, change over time, respond to context, and remain probabilistic. The framework represents these properties through an evolving densitymatrix state $\rho _ { i } ( t )$ in a nonlinear feature space, allowing behavioral profiles and within-profile modes to emerge from observed behavior and context, and ultimately supporting macroscopic trafic phenomena and prediction.

The representation is microscopic by construction: it describes how a single driver, evolves along that driver’s trajectory, and depends only on quantities observable at each instant. When applied across many drivers, the same representation reveals macroscopic structure and trafic dynamics. For instance, we show how the recovered behavioral profiles align with the properties of the fundamental diagram, how they reproduce trafic waves and hysteresis, and how stop-and-go behavior is interpreted. This link from individual behavior to aggregate trafic phenomena is not built into the model, but emerges from the data precisely because the representation was not constrained to a fixed functional form in advance.

Our goal in this paper is to ask whether behavioral heterogeneity can be represented in a form that starts from raw observations, captures the relationships among them without pre-specifying their form, and connects back to the macroscopic trafic phenomena that motivate studying behavior in the first place. We develop such a representation, demonstrate it on the I-24 MOTION dataset, and show how it integrates with classical driving models and supports prediction of human behavior in mixed trafic.

## 1.1. Related Work

Behavioral heterogeneity is a central feature of trafic flow. Diferences in how drivers choose speed, maintain spacing, and respond to disturbances shape how congestion forms, propagates, and dissipates at the macroscopic level. How this variability is represented therefore afects which behavioral patterns a model can capture, which trafic phenomena it can explain, and how useful it is for simulation and control. Research has progressively expanded the representation of heterogeneity, moving from diferences across drivers to changes within drivers, the role of trafic context, and richer probabilistic and data-driven descriptions. Many existing approaches address several fo these dimensions together, but difer in which they emphasize and how they combine them.

Drivers are inherently heterogeneous. A fundamental observation in trafic flow research is that drivers respond diferently to the same conditions and stimuli. Diferences span many aspects of driving, including car-following, gap acceptance, and acceleration, and arise from many sources, such as vehicle type, risk tolerance, perception, and stochastic variation in decision-making (Ossen and Hoogendoorn, 2011; Hamdar et al., 2019; Wagner, 2012). To accommodate this, heterogeneous trafic models progressed from fixed parameter sets to random coeficients, population distributions, driver-state distributions, and driver-specific calibration (Kim et al., 2013; Ding et al., 2022; Yao et al., 2025a). These approaches recognize that no single parameter setting describes the population, but they still treat each driver as a fixed point within it.

Drivers’ behavior changes over time. The next step was to recognize that an individual driver’s behavior is not fixed. A growing body of work shows that driving behavior evolves in response to changing trafic conditions, workload, surrounding vehicles, and strategic intent (Tavakoli et al., 2022; Battifarano and Qian, 2023). To capture this, state-based models (Tavakoli et al., 2022), latent behavioral states (Nirmale et al., 2024), and regime-switching frameworks (Ji et al., 2023) have been proposed in which drivers transition between behavioral modes over time, but the mechanism that drives the transition remains unspecified. This temporal variability extends beyond human drivers: the car-following behavior of automated vehicles also shifts across operating modes and conditions, with measurable efects on hysteresis and throughput (Zhong et al., 2024).

Behavior depends on context. If drivers change over time, what drives the change? The literature points to the surrounding trafic environment. Behavior varies with density, speed variability, local interactions, visual conditions, and the actions of neighboring vehicles (van Beinum et al., 2018; Nirmale et al., 2021; Xu et al., 2018). Recent models therefore incorporate both local conditions and broader trafic states, recognizing that behavior emerges from the interaction between driver and environment rather than from driver characteristics alone (Wang et al., 2025). Even so, drivers exposed to similar conditions can respond diferently, because context is mediated by individual preferences, experience, and perception (Lee, 2008; Oppenheim and Shinar, 2012). The challenge is to represent drivers, their behavioral evolution, and the way context shapes that evolution in one place.

Representing heterogeneous behavior. A wide range of methods have been proposed. Clustering and latent-class models group drivers into a small number of interpretable categories, providing a simple description of population-level variability (Yao et al., 2025a,b; Sun et al., 2020). Mixture models extend this by allowing behavior to be represented probabilistically rather than through hard assignments (Sun et al., 2020; Wang et al., 2024). Hidden-state approaches such as Hidden Markov Models and Dynamic Bayesian Networks add temporal evolution by modeling transitions between behavioral states over time (Xu et al., 2018; Ding et al., 2022). Behavioral embeddings and deep learning methods learn flexible representations directly from large-scale data, capturing nonlinear structure that is dificult to model explicitly (Bouhsissin et al., 2023; Wang et al.,

2024; Yao et al., 2025b). Related kernel-based approaches model context-dependent variability and uncertainty through learned similarity structures, ofering greater flexibility than deterministic formulations (Zhang et al., 2025). These models capture complex behavior, but the interactions they learn are often dificult to interpret. Other work separates population-level from agent-specific behavior, showing that aggregation can obscure meaningful individual diferences (Kontar et al., 2026). Yet a recurring trade-of appears: methods that produce intuitive descriptions rely on simplified representations, while methods that capture richer structure tend to lose interpretability. Temporal evolution, probabilistic reasoning, variable interaction, interpretability, and context dependence are usually addressed separately rather than within a single framework.

A density matrix view of behavior. Each of these methods supplies part of what driving behavior demands, but pre-defines a structural form to do so. Mixture models are probabilistic, but only as weights over component densities chosen in advance, so they record how much each pre-specified component is present, not how behavioral variables interact. Kernel methods lift inputs into a space where nonlinear structure becomes accessible, but return similarity functions or regression estimates rather than normalized probability states that can be directly interpreted, updated, and propagated through time. Hidden-state models add temporal evolution, but over a discrete set of states fixed before estimation. What is missing is a single object that is at once probabilistic, interaction-bearing, continuous in time, and free of a pre-specified relational form. A density matrix provides such an object. Recent work in quantum cognition and quantum-inspired machine learning uses density matrices to represent context-dependent behavior without reducing individuals to fixed categories (Khrennikov and Asano, 2020; Busemeyer et al., 2025). Built on a Random Fourier Feature lift, the density matrix has a structure that is learned rather than chosen: its eigenvectors define behavioral directions estimated from data, and its of-diagonal terms carry variable interactions that a mixture weight vector cannot (González et al., 2022). The Born rule turns this operator into a probability model, and the same operator evolves over a trajectory and conditions on context.

A useful representation of driving behavior should be continuous rather than discrete, probabilistic rather than deterministic, context-dependent rather than fixed, history-dependent rather than memoryless, and capable of representing behavioral variable interactions that are learned from data rather than specified by the modeler. Existing approaches address these individually or partially, but not all at once. We propose a density-matrix framework that does, and apply it to driver heterogeneity for the first time. The result is an evolving representation that admits direct interpretation through a small set of behavioral profiles, and whose context- and history-dependent dynamics connect individual behavior to known macroscopic phenomena such as the fundamental diagram and hysteresis.

A Design Philosophy for Trustworthy Behavioral Models

This paper is, in method, an AI approach to driver behavior.Its objective, however, extends beyond predictive accuracy to include transparency, interpretability, and consistency with known physical principles. We argue that, in complex interactive systems such as trafic, behavioral representation determines what can ultimately be understood about the system, and shortcomings in representation propagate into shortcomings in explanation. Existing approaches face three related limitations (Figure 2). First, driver heterogeneity is commonly reduced to fixed types or single labels, overlooking its continuous, context- and history-dependent nature. Second, both classical car-following models and many machine-learning approaches impose predefined relationships among behavioral variables, forcing observed behavior into fixed forms while treating remaining variation as noise. Third, because trafic emerges from interactions among heterogeneous drivers, these simplifications propagate to the macroscopic level and limit our understanding of trafic dynamics.

Our framework addresses these limitations by learning relationships among behavioral variables directly from the data while representing driver behavior as a continuous, probabilistic, historydependent state. Interpretability follows from the resulting structure: individual actions give rise to behavioral profiles, which in turn aggregate into recognizable trafic phenomena. This aggregation also provides a natural validation criterion. If a learned behavioral representation cannot reproduce known macroscopic trafic patterns, it is either not representative of real driving or unsuitable for explaining trafic behavior. In this sense, trustworthiness is embedded in both the representation and its validation rather than added afterward. This perspective is consistent with the broader principle that trustworthy AI emerges from modeling interactions rather than optimizing isolated properties one at a time (Cresswell, 2025).

![](images/e52d8875b487abe92527fcdb48d9af2fce7e04da1586d1000a018b7804c59b4b.jpg)  
A representation is trustworthy only ifthe behavior it learns aggregates into the traffic phenomena we already know.  
Figure 2: The design philosophy behind the framework. The same heterogeneous driver behavior can be represented in two ways. Imposing fixed interactions forces behavior into a predefined template, causing unmatched variation to appear as noise and preventing the emergence of the correct macroscopic structure. Learning interactions directly from the data preserves this variation and reproduces the observed trafic patterns while remaining interpretable. A behavioral representation is therefore trustworthy only if what it learns aggregates into known trafic phenomena.

## 1.2. Main Contributions:

The contributions of this paper are:

1. A quantum-inspired representation of driver heterogeneity. We represent each driver as an evolving density matrix in a nonlinear feature space. Mapping the behavioral variables into this space encodes variable interactions, and the resulting representation is continuous, probabilistic, context-dependent, and history-dependent, combining properties that are often treated separately in the heterogeneity literature.

2. A method for unsupervised discovery of behavioral structure. The framework recovers a small set of interpretable behavioral profiles, within-profile modes that capture finer dynamics, and continuous transitions between profiles as context evolves. None of these are imposed, all emerge from observed driver behavior and local context.

3. A constructive bridge to trafic modeling. The trained representation provides direct estimates of behavioral quantities and can be used both in driver models, such as car-following and lane-changing, and in autonomous vehicle (AV) planning as a context-aware representation of surrounding drivers.

4. An empirical demonstration on the I-24 MOTION dataset. The framework recovers interpretable behavioral profiles and modes while reproducing well-known trafic phenomena such as fundamental-diagram structure, hysteresis, and stop-and-go waves without supervision or behavioral labels.

5. An open-source implementation. We release a public toolkit covering data processing, model training, inference, and post-analysis to support future research and reproducibility.

## 2. Motivation of the Study

Three observations motivated the representation developed in this paper: (i) drivers difer from one another, yet those diferences often exhibit recurring structure, (ii) the same driver may behave diferently as context changes. and (iii) there is co-dependency among the variables in how they afect behavior. Existing approaches provide useful ways of studying these phenomena, but they are often addressed separately. We organize this section around three stylized experiments that compare a density-matrix representation against the families of methods most commonly used in the literature: methods built on static parameters, methods that assign each driver to a discrete class or to a fixed mixture over latent components, and deep learning methods that learn a perobservation state.

Drivers difer, and that diference has structure: We simulate one leader and four followers, d1 through d4. The leader brakes, holds a low speed, then accelerates. Each follower reacts differently in speed, headway, and acceleration. The scenario reproduces the stop-and-go disturbance that is ubiquitous in trafic. The followers’ gaps to the leader over time (Figure 3a) reveal four distinct following styles: d1 keeps a large gap and brakes gently, d4 keeps a short gap and brakes hard, and d2 and d3 sit in between. By design, d2 and d3 share nearly identical mean speed, headway, and acceleration, difering only in that d3 brakes harder during the leader’s maneuver. This lets us test whether a representation can distinguish two drivers that agree on average but diverge in a transient response. We fit three representations to the same data. K-means on each driver’s mean features (K = 3) merges d2 and d3 because their means coincide, discarding the diference (Figure 3b). A small neural network keeps them distinct, but only in latent coordinates that carry no behavioral meaning (Figure 3c). The density-matrix representation describes each driver as a soft mixture over three behavioral profiles: a smooth cruiser, an active follower, and an aggressive maneuverer, with within-profile modes (Figure 3d). Here, d3’s harder braking surfaces as a larger aggressive-maneuvering share. The d2–d3 diference is therefore genuine, recoverable structure, yet only the density matrix preserves it while remaining interpretable.

Drivers also change as the context changes: Heterogeneity is not only about who a driver is, but about how the same driver responds as conditions change. Here the context is the surrounding trafic density, which rises steeply into congestion between t = 26 and 56 s (Figure 4a). A leader slows as density builds, and two followers, d1 and d2, respond diferently. The first keeps a large time headway, accelerating and braking gently. The second keeps a short time headway and brakes firmly as it enters the dense region. Both settle into a low-speed regime, but along clearly diferent paths.

![](images/9526169478eda97352b5f283355ba2df67bee3faf4e3065e289012c93c9276c9.jpg)

![](images/89fbedb1778799d62fddf7db1331c19263bf9a5f89000970dbdd1e01bde1ee11.jpg)

![](images/a04fea56b4d93c8c29fc25ed8fd9510fea3e562af51c5c6ad294ab67c9d2d42a.jpg)

![](images/375bd83fa9707afabe320ad657e16653c6d1e80e1ed112f42f241e6528c0457e.jpg)  
Figure 3: Heterogeneity under three fitted representations on a common leader-follower simulation. (a) Gap to the leader over time for drivers d1–d4. (b) Hard K-means clustering K = 3 conflates d2 and d3. (c) A small neural network yields latent coordinates with no direct behavioral interpretation. (d) The density-matrix representation describes each driver as a soft mixture over three behavioral profiles with within-profile modes.

We fit three representations to these trajectories. The static representation summarizes each driver by a single calibrated IDM. It places the two drivers in diferent behavioral classes but cannot follow the change within either when context (i.e., density) changes (Figure 4b). The fixed mixture represents each driver as a constant mixture of three reference IDM regimes (relaxed, steady, and aggressive-braking). These mixture weights difer between drivers, but they are estimated once over the whole trajectory and stay fixed in time, failing to changing with context (Figure 4c). The densitymatrix representation instead carries profile weights at each instant. Its profiles–smooth cruising, active following, and aggressive maneuvering–are behavioral regimes a driver moves through, not fixed types. Unlike the static and mixture representations, it assumes no IDM or predefined functional form. The profiles and the transitions between them are learned from data. As trafic density rises, each driver’s composition shifts from smooth cruising toward active following and aggressive maneuvering (Figure 4d). The two drivers traverse the same change in context diferently: the first transitions early and gradually, the second later and more sharply.

![](images/f35bdd67312746a1d88ab137b63e010459548106e2a86ec565ba6e90f1a7b902.jpg)

![](images/83327602a8ad2694042c50f90ea64571507764e4a4791f808247a3f3ac711756.jpg)

![](images/67375054ef76d7864c9ce27c92646aa5d47a83c5609ea35842a55c6253aa47da.jpg)

![](images/6523ae90f3b9be7fb2f98bed5451af4d2b897818c3a3f66e7575cf382d78ac0e.jpg)  
Figure 4: Two simulated drivers (d1 and d2) responding to a leader as trafic density rises into congestion. (a) Speeds over time against rising density. (b) A static representation calibrates one fixed IDM per driver, placing them in rigid behavioral classes insensitive to context. (c) A fixed mixture assigns constant proportions over three reference IDM regimes, difering between drivers but not over time. (d) The density-matrix representation resolves how each driver’s composition shifts among free-form behavioral profiles (not governed by IDM formulation) over time, capturing when and how sharply each driver transitions.

Behavior depends on how variables interact: Behavioral variables do not act in isolation.

![](images/b3f463514ae42da3cd24179e27f6444bf473593b980a17502a302212c452003d.jpg)  
Figure 5: Acceleration versus headway across three speed ranges. (a) A single global regression misses the speed interaction. (b) A Gaussian mixture with few components averages across regimes. (c) A small neural network predicts accurately but yields no interpretable profile structure. (d) TThe density-matrix representation assigns observations to three profiles; the inset shows the mixture weights varying smoothly with speed, and the intermediate range (green) is captured as its own profile.

Rather, they jointly afect behavior in meaningful ways. Figure 5 plots acceleration against headway, with each observation colored by the speed at which it was recorded. The data are constructed with three speed ranges: low-speed observations (blue), intermediate-speed observations (gray), and high-speed observations (red), each generated with its own acceleration–headway relationship. The same headway corresponds to diferent accelerations depending on speed: at low and high speeds acceleration varies roughly linearly with headway, while at intermediate speed the relationship is nonlinear (hump-shaped). We fit four representations to the data. A single global regression of acceleration on headway (Figure 5a) passes through all three speed ranges at once, without distinguishing them. A Gaussian mixture model (Figure 5b), which represents the data with Gaussian components and predicts acceleration as a blend of them, smooths across the three ranges so the hump-shaped middle is flattened into the average. A small neural network trained to predict acceleration from headway and speed (Figure 5c) fits the points accurately, shown as its per-observation predictions, but exposes no profile structure. That is, the relationship it has learned is buried in the network and cannot be read of in behavioral terms. The density-matrix representation (Figure 5d) softly assigns each observation to three behavioral profiles and learns a separate relationship for each. The inset figure in Figure5d shows the mixture weights as a function of speed: as speed increases, weight passes smoothly from one profile to the next, so observations shift gradually between profiles. The hump-shaped middle range (gray points) is captured by its own profile (Profile 2 in green) rather than being averaged into the others, and the shape of each profile’s relationship is learned from the data.

Taken together, the three examples illustrate the considerations that motivated our approach. We sought a representation that could describe diferences between drivers, adapt as conditions change, and capture interactions among driving variables, while remaining interpretable. The density-matrix framework provides one way of balancing these goals through a combination of shared behavioral profiles, state evolution, and eigendecomposition. The remainder of the paper develops this framework and evaluates it on the I-24 MOTION dataset.

## 3. Methodology

This section develops a behavioral model that represents each driver as an evolving density matrix over a nonlinear feature space and explains its choices of state, profiles, context, and dynamics. The objective is a representation flexible enough to capture both how drivers difer from one another and how a single driver’s behavior changes with context, without forcing every driver into a fixed label.

## 3.1. Background

The framework developed here adopts a diferent starting point from current representations of behavioral heterogeneity. Each driver is described not by a parameter vector or a discrete label, but by a density matrix, a representation borrowed from quantum-inspired machine learning, where density matrices serve as a general probabilistic representation for nonlinear data (González et al., 2022). We adopt three structural conditions that qualify a learning framework as quantum-inspired:

Quantum-inspired modeling conditions. A learning framework is considered quantuminspired if it satisfies three structural conditions: (i) observations are encoded as normalized states in a Hilbert space, (ii) predictions are computed using a quadratic measurement rule consistent with the Born rule, and (iii) uncertainty is represented using valid density matrices that are symmetric, positive semidefinite, and trace-normalized.

A density matrix is a compact way to describe a behavioral pattern, not as a single fixed action, but as a distribution over the behaviors a driver may exhibit, together with the relationships among them. In the simplest case, the pattern reduces to one dominant behavioral mode, a characteristic way of driving. More often, a driver’s behavior is better described by several modes that coexist within the same pattern. For instance, a driver in dense trafic may alternate between smooth and aggressive car-following, and the density matrix captures both, along with how much weight each carries. Unlike an ordinary list of probabilities, it also records how the modes relate, not just how likely each one is. This represents interactions between behaviors, such as the way speed conditions the efect of spacing, rather than averaging them away. It is this property that lets a single object express both the range of behaviors a driver shows and the structure that ties them together.

Six concepts recur throughout this section and are central to the proposed framework. They are summarized in Table 1.

The combination of these properties allows the framework to capture behavioral heterogeneity at multiple levels. Diferent drivers can exhibit diferent combinations of profiles, the profile mixture can change over time as trafic conditions evolve, and each profile can contain multiple behavioral modes. The remainder of this section describes how these components are represented mathematically and estimated from data.

## 3.2. Variable Selection

Before defining the model, we select the variables it operates on. In principle, the framework can take any set of behavioral and contextual inputs, so variable selection is not strictly required. We perform it to keep the state as small as possible while retaining the information that matters: a compact state is easier to estimate, less prone to redundancy, and more readily transferred to and compared across other datasets and models. Because the task is driving behavior, the candidates are kinematic quantities describing the driver’s motion and variables describing the surrounding trafic conditions.

Table 1: Core concepts used throughout the framework.
<table><tr><td>Concept</td><td>Meaning</td></tr><tr><td>Behavioral vector</td><td>A short vector of observed driving quantities at a single instant. This is the driver&#x27;s action, measured directly.</td></tr><tr><td>Context</td><td>A vector of variables describing the surrounding traffic situation. Context is what the driver is responding to and is read from the neighborhood.</td></tr><tr><td>State</td><td>A density matrix at time t that summarizes everything the model has learned about the driver from past behavior and past context. It also evolves through context-driven predictions and observed behavior.</td></tr><tr><td>Profile</td><td>A density matrix representing a recurring pattern of driving behavior. For ex- ample, a profile can represent congested car-following, characterized generally by lower speeds and shorter spacing.</td></tr><tr><td>Profile mixture</td><td>A weighted combination of profiles for a driver at time t, where the weights depend on the current traffic context. As context changes, the contribution of each profile changes smoothly. For example, increasing traffic density may shift weight from a free-flow profile toward a congested car-following profile.</td></tr><tr><td>Modes within a profile</td><td>e Each profile can contain multiple behavioral modes, with weights indicating their relative importance. For example, a congested car-following profile may include both smooth and aggressive car-following behaviors, with different weights assigned to each mode.</td></tr></table>

Candidate variables were evaluated using four criteria: how much drivers difer from one another under comparable conditions, how strongly a variable responds to changing trafic conditions, how much it evolves over the course of a trajectory for the same driver, and how much of its variation is explained by driver characteristics. The complete candidate set, formal definitions of the criteria, and ranking results are provided in Appendix A.

The final set contains three behavioral and three contextual variables that capture complementary aspects of driving behavior: speed, spacing, jerk, local trafic density, speed entropy, and acceleration entropy. The context set favors entropy measures over averaged speeds deliberately. In congestion, the mean speed of surrounding trafic nearly coincides with the driver’s own speed, so it adds little beyond what speed already provides; the speed averages were therefore excluded as redundant. The entropy measures instead capture the variability, or disorder, in surrounding speeds and accelerations, which is both distinct from ego speed and among the few context signals that evolve over time.

The variables above longitudinal motion, vehicle spacing, driving smoothness, surrounding trafic demand, and local trafic variability. The selected variables are summarized in Table 2.

Table 2: Selected behavioral and contextual variables.
<table><tr><td>Type</td><td>Variable</td><td>Description</td></tr><tr><td>Behavioral</td><td>Speed (v)</td><td>Magnitude of longitudinal velocity (ft/s)</td></tr><tr><td>Behavioral</td><td>Spacing (∆s)</td><td>Gap to same-lane leader (ft)</td></tr><tr><td>Behavioral</td><td>Jerk (j)</td><td>Rate of change of acceleration (ft/s3)</td></tr><tr><td>Contextual</td><td>Density (d)</td><td>Vehicles within a 150 m radius</td></tr><tr><td>Contextual</td><td>Speed entropy (Hv)</td><td>Entropy of neighbors&#x27; 1 s speed changes</td></tr><tr><td>Contextual</td><td>Acceleration entropy (Ha)</td><td>Entropy of forward neighbors&#x27; accelerations</td></tr></table>

## 3.3. Driver State and Feature Representation

Let i index drivers and t index time steps within a trajectory. The behavioral vector of driver i at time t is represented by the three selected behavioral variables

$$
\begin{array} { r } { \boldsymbol { x } _ { i t } = \left[ \boldsymbol { v } , \Delta s , j \right] ^ { \top } \in \mathbb { R } ^ { 3 } , } \end{array}
$$

corresponding to speed, spacing, and jerk. These variables capture complementary aspects of driving behavior, but their relationships are often nonlinear. For example, the behavioral significance of a given spacing depends on speed, and the efect of a jerk event may vary across trafic conditions. To capture these interactions, the model does not act on the three raw variables directly. Instead, it builds a larger set of features, which are derived quantities computed from the behavioral variables into a much longer vector, giving the model room to separate behaviors that look similar in the original three. We construct these features using an RFF projection, where each feature is one component of the map:

$$
\phi _ { j } ( x ) = \cos ( w _ { j } ^ { \top } x + b _ { j } ) , \qquad j = 1 , \ldots , D ,\tag{1}
$$

where D is the number of features (the dimension of the feature space), and the projection parameters $w _ { j } \sim \mathcal { N } ( 0 , \sigma ^ { - 2 } I _ { 3 } )$ and $b _ { j } \sim \mathrm { U n i f o r m } ( 0 , 2 \pi )$ are sampled once at initialization and held fixed throughout estimation. Each feature responds to a diferent combination of speed, spacing, and jerk, so the full set covers many nonlinear efects, such as how speed scales the influence of spacing, without the modeler deciding in advance which variables interact (we illustrate this in Figure 6).

![](images/e5ec8c5ce3e079dfdaa93dd430d5faeda1ef1a71f43b73f5a2a06d2cdc2ee606.jpg)  
Figure 6: The RFF projection. The three behavioral variables (speed, spacing, jerk) form a short vector x of size $q = 3 .$ . The random Fourier projection maps it into a much longer feature vector $\phi ( x )$ of size $D \gg 3 .$ , where each feature is one nonlinear combination of the inputs, $\phi _ { j } ( x ) = \cos ( w _ { j } ^ { \top } x + b _ { j } )$ . The expanded representation lets the model separate behaviors that look similar in the original variables.

The resulting feature vector $\phi ( x _ { i t } ) \in \mathbb { R } ^ { D }$ is normalized to unit length,

$$
\tilde { \phi } ( x _ { i t } ) = \phi ( x _ { i t } ) / \lVert \phi ( x _ { i t } ) \rVert .\tag{2}
$$

This rescaling ensures that the probability later assigned to each observation through the Born rule (Section 3.5) stays properly bounded between 0 and 1.

So far, each observation is a single RFF feature vector of size D. A profile, however, is not a single action but a pattern spanning many behaviors, so we need a form that can be combined and accumulated across observations. We therefore form the outer product of the RFF feature vector with itself:

$$
\Phi ( x _ { i t } ) = \tilde { \phi } ( x _ { i t } ) \tilde { \phi } ( x _ { i t } ) ^ { \top } ,\tag{3}
$$

which re-expresses the same observation as a matrix of size $( D \times D )$ rather than a vector. This rank-one matrix is the density-matrix form of a single observation: it holds the same information as $\tilde { \phi } ( \boldsymbol { x } _ { i t } )$ , but in a form that can be summed and weighted together with others to build the profiles and evolving states defined next.

In summary, we have so far shown the building block of our representation that takes the raw behavioral variables of one observation, $x _ { i t }$ , and returns the normalized feature vector $\tilde { \phi } ( \boldsymbol { x } _ { i t } )$ and its rank-one matrix $\Phi ( x _ { i t } )$ . The input is what the driver did at a single instant; the output is that instant re-expressed in the form the rest of the framework operates on, ready to be accumulated into profiles.

## 3.4. Behavioral Profiles as Density Matrices

A profile $\rho _ { k }$ has the same form as the single-observation matrix $\Phi ( x _ { i t } )$ built above. However, instead of representing one observation, it represents a recurring pattern across many. We assume the driver population contains K such profiles, each represented by a density matrix satisfying

$$
\rho _ { k } \in \mathbb { R } ^ { D \times D } , \qquad \rho _ { k } = \rho _ { k } ^ { \top } , \quad \rho _ { k } \succeq 0 , \quad \mathrm { T r } ( \rho _ { k } ) = 1 .\tag{4}
$$

Unlike a simple average over the observations, a profile preserves the interactions among the RFF features, not just their values. Figure 7 illustrates how a density matrix encodes this, through its diagonal and of-diagonal terms.

![](images/cdbe4780b5646b778d4b7b6950c7b71785b50d0a568c41ad38d09bd93459ca80.jpg)  
Figure 7: Structure of a profile density matrix $\rho _ { k }$ , shown as a truncated $D \times D$ matrix over the RFF feature indices. Diagonal terms (blue) give the weight of each RFF feature in the profile. Of-diagonal terms (red) give the coupling between two features, which enables the profile to represent interactions among behaviors. The eigendecomposition of $\rho _ { k }$ yields its behavioral modes (green): each bar is one mode, and its height is the corresponding eigenvalue, indicating how strongly that mode contributes to the profile.

Each profile is parameterized through a low-rank factor $V _ { k } \in \mathbb { R } ^ { D \times { r } }$ , where r is the rank of the profile:

$$
\rho _ { k } = \frac { V _ { k } V _ { k } ^ { \top } } { \operatorname { T r } ( V _ { k } V _ { k } ^ { \top } ) } .\tag{5}
$$

The rank r determines how many behavioral modes a profile can represent. A rank-r matrix has at most r nonzero eigenvalues, corresponding to at most r distinct modes. Choosing $r \ll D$ limits the model to a few dominant modes, reducing the number of parameters and mitigating overfitting. Importantly, this does not reduce the D-dimensional RFF feature space; each mode remains fully nonlinear and expressive. The rank limits the number of modes, not their expressive power.

A rank-one profile represents a single behavioral mode. Although only one mode is retained, it is represented in the D-dimensional RFF space, where each feature corresponds to a nonlinear transformation of the original behavioral variables. Consequently, the resulting behavioral mode can model complex nonlinear relationships among speed, spacing, and jerk. Higher-rank profiles let several such modes coexist within the same trafic regime. For example, a profile describing carfollowing in congested trafic may contain both smooth and jerky following, with the eigenvalues indicating their relative prevalence.

This raises the question of how many profiles (K) can best describe the population. We do not pre-specify K. Instead, the model is estimated for several values $( K \in \{ 2 , 3 , 4 , 5 \} )$ , and the final choice balances how well the model fits the data, and whether the resulting profiles are distinct rather than redundant. We prefer solutions with higher likelihood, where the profiles collectively assign more probability to the behavior actually observed, while remaining distinct. These diagnostics support $K = 3$ , and are presented in detail in Appendix B.

## 3.5. Contextual Activation and State Evolution

Driver behavior depends on the situation in which the driver is acting. A driver who is steady in light trafic may become reactive in congested trafic, and the same spacing may imply diferent behavior at diferent speeds. We refer to this as context, and we represent it as a vector:

$$
c _ { i t } = \left[ d , H _ { s } , H _ { a } \right] ^ { \top } \in \mathbb { R } ^ { 3 }
$$

Context does not alter the profiles themselves; it determines how strongly each profile contributes to the driver’s current state. The contribution weight of profile k is computed through a softmax:

$$
\pi _ { k } ( c _ { i t } ) = \frac { \exp ( \beta _ { k } ^ { \top } c _ { i t } ) } { \sum _ { j = 1 } ^ { K } \exp ( \beta _ { j } ^ { \top } c _ { i t } ) } ,\tag{6}
$$

where $\beta _ { k } \in \mathbb { R } ^ { 3 }$ . The softmax is used so that the contribution weights are non-negative and sum to one, forming a valid distribution over profiles, while varying smoothly with context: as conditions on the road change, weight shifts gradually from one profile to another rather than switching abruptly. For example, a profile associated with congested trafic receives greater weight as trafic density increases.

So far, context tells us which profiles should contribute to the driver’s state at the current instant. But behavior also has memory: a driver does not reset to a context-implied state at every step, it carries over its recent behavior. The predicted state therefore blends two things: (i) the driver’s previous state, which carries this behavioral inertia, and (ii) the context-weighted combination of profiles, which pulls the state toward what the current situation implies:

$$
\rho _ { i } ^ { \mathrm { p r e d } } ( t ) = \left( 1 - \alpha \right) \rho _ { i } ( t - 1 ) + \alpha \sum _ { k = 1 } ^ { K } \pi _ { k } ( c _ { i t } ) \rho _ { k } .\tag{7}
$$

The parameter $\alpha \in ( 0 , 1 ]$ sets the balance between them. Small α keeps the driver close to its previous behavior (strong inertia), whereas large α lets context reshape the state more quickly.

Given this predicted state, we measure how well it matches what the driver actually did. The observed behavior is a feature vector, and the predicted state is a density matrix describing which behaviors the driver is expected to express. We measure the agreement between the two using the Born rule. Since the predicted state is a density matrix, the Born rule is the natural way to turn it into a probability between 0 and 1. A standard measure like a Gaussian would not fit, because it requires assuming a fixed shape for the data in advance, whereas our state makes no such assumption and instead represents behavior through learned modes and their interactions. The Born rule reads the probability directly from this state. The value is large when the observed behavior matches what the state expects and small when it does not This agreement is read directly as a probability, that serves as a likelihood for our estimation:

$$
p ( x _ { i t } \mid \rho _ { i } ^ { \mathrm { p r e d } } ( t ) ) = \tilde { \phi } ( x _ { i t } ) ^ { \top } \rho _ { i } ^ { \mathrm { p r e d } } ( t ) \tilde { \phi } ( x _ { i t } ) .\tag{8}
$$

After observing $x _ { i t }$ , the state is updated to fold in what was just seen, so the representation tracks the driver over time:

$$
\rho _ { i } ( t ) = ( 1 - \eta ) \rho _ { i } ^ { \mathrm { p r e d } } ( t ) + \eta \tilde { \phi } ( x _ { i t } ) \tilde { \phi } ( x _ { i t } ) ^ { \top } , \qquad \eta \in [ 0 , 1 ] .\tag{9}
$$

The parameter η controls how strongly new observations influence the state. Smaller values emphasize long-run behavioral tendencies, while larger values let recent observations have greater influence. Together, α and η capture two distinct mechanisms of change: adaptation to context and adaptation to experience.

## 3.6. Estimation, Regularization, and Interpretation

Up to this point we have described the form of the model: profiles, contextual contributions, and the evolving state. We now turn to what is actually learned from data. The model has three sets of unknowns. The profile factors $V _ { k }$ determine the behavioral profiles themselves, that is, what recurring patterns of driving the population contains. The context coeficients $\{ \beta _ { k } \}$ determine how the surrounding context shifts weight among those profiles. The adaptation parameter η determines how quickly a driver’s state responds to new observations. Estimation finds the values of these unknowns that make the observed driving behavior most likely under the model.

Estimation minimizes the per-observation negative log-likelihood (NLL), with an entropy penalty on the profile eigenvalues:

$$
\mathcal { L } = - \sum _ { i , t } \log p \big ( \boldsymbol { x } _ { i t } \mid \rho _ { i } ( t ) \big ) \ - \ \gamma \sum _ { k = 1 } ^ { K } H ( \lambda _ { k } ) ,\tag{10}
$$

where $H ( \lambda _ { k } )$ is the entropy of the eigenvalues $\lambda _ { k }$ of profile $\rho _ { k }$ , and $\gamma$ controls the strength of the penalty. The first term rewards profiles and contributions that explain the observed behavior. The second term encourages each profile to retain multiple behavioral modes rather than collapsing onto a single dominant one. We set $\gamma = 4$ , selected by balancing profile richness against fit (NLL), as discussed in Appendix B.

The parameters $\{ V _ { k } \} , \ \{ \beta _ { k } \}$ , and η are found using the Adam optimizer, a standard gradientbased method that adapts its step size during training. For each driver, the state is advanced step by step along the trajectory and the per-step likelihoods are summed into the total NLL, which is then reduced by adjusting the parameters. The construction $\rho _ { k } = V _ { k } V _ { k } ^ { \top } / \mathrm { T r } ( V _ { k } V _ { k } ^ { \top } )$ ensures every profile remains a valid density matrix throughout.

The persistence parameter α (Equation 7) sets how much of the previous state carries forward at each step. We fix it at a small value, $\alpha = 0 . 2$ , rather than learn it. If α is large, each state is mostly a copy of the one before it, so the model can predict the next behavior simply by repeating recent behavior. This improves NLL without the profiles learning anything meaningful. Keeping α small prevents this: it forces the state to draw on the context-weighted profiles at each step, so the profiles, rather than simple repetition, carry the behavioral structure.

Driver states are initialized as $\rho _ { i } ( 0 ) = I _ { D } / D$ , where $I _ { D }$ is the $D \times D$ identity matrix. This represents maximum uncertainty, equal weight on all directions, before any of the driver’s behavior has been observed. The state is then updated sequentially along the trajectory.

After estimation, each profile is interpreted by examining the behaviors most strongly associated with it. This reveals the combinations of speed, spacing, jerk, and surrounding trafic characteristics that define the profile, as well as any distinct behavioral modes it contains. The context coeficients $\beta _ { k }$ indicate the conditions under which a profile becomes more or less likely to govern behavior, while the evolution of $\rho _ { i } ( t )$ shows how a driver’s behavior changes over time in response to context and accumulated experience.

## 3.7. Interpreting the Learned Representation

The estimated model is interpreted at three levels, each read directly from the learned quantities.

Profiles. Each profile $\rho _ { k }$ is interpreted through its eigendecomposition: the leading eigenvectors give the profile’s dominant behavioral modes, and the eigenvalues give their relative weight. Projecting these modes back onto the behavioral variables recovers the combinations of speed, spacing, and jerk that characterize the profile, so that the density matrix is read as a concrete pattern of driving.

Context. The coeficients $\beta _ { k }$ determine the contribution weights $\pi _ { k } ( c )$ , which show the trafic conditions under which each profile becomes more or less active. Tracing $\pi _ { k } ( c )$ as context varies reveals how the surrounding trafic shifts a driver between profiles.

Evolution. The evolving state $\rho _ { i } ( t )$ shows how an individual driver’s behavior changes along a trajectory, as the contribution of each profile rises and falls in response to context and accumulated experience.

Figure 8 summarizes the full estimation pipeline.

## 4. Data and Training

This section establishes the empirical basis for the analysis that follows. We first describe the I-24 MOTION dataset and why it suits our method, then detail how the model was trained on it.

## 4.1. Dataset

Our method requires trajectory data with three properties: individual vehicle behavior resolved over time, enough surrounding vehicles to characterize each driver’s local context, and a range of trafic conditions broad enough to exercise diferent behaviors. High-resolution, spatially dense trajectory data are therefore essential. I-24 MOTION (Gloudemans et al., 2023) meets these requirements: it records every vehicle across multiple lanes of a freeway segment at high frequency, over a period spanning free flow through heavy congestion and recovery, so both individual behavior and its surrounding context are directly measurable.

The I-24 MOTION dataset provides vehicle trajectories collected from a 4.33 mile segment of Interstate 24 near Nashville, Tennessee, recorded by a 294 camera network at 25 Hz. The release used here spans 239.8 minutes of morning trafic on November 22, 2022, and contains 771,946 trajectories, of which 519,665 (67.3%) are westbound. We restrict the analysis to westbound trafic, which captures a complete free-flow to congestion to recovery cycle over the recording window. The spatiotemporal speed field in Figure 9 illustrates this structure: speeds remain high through the first portion of the recording, then collapse into a band of dense stop-and-go waves propagating against the direction of travel, before recovering toward the end.

![](images/52a399f3fd594f6137de1fd0044e33bd334f50dc3e7c4621fff9faefcb81e55c.jpg)  
Figure 8: Estimation pipeline of the quantum-inspired driver behavioral profiling framework. Trajectories are reduced to behavioral and contextual variables, the behavioral state is mapped through a RFF projection, and the per-driver density-matrix state is evolved by context-driven profile mixture and observation-driven update. Profiles and context coeficients are estimated by minimizing the NLL and interpreted through eigendecomposition.

![](images/4fcec34cc1459af1d67265188d61cb30eda9a4992895854f09676de3399be0bc.jpg)  
Figure 9: Spatio-temporal speed field for westbound I-24 trafic. Color encodes mean speed from low (red) through moderate (yellow) to high (green).

Within the westbound population, vehicles split across six coarse classes (sedan, midsize, van, pickup, semi, truck). We distinguish these classes because the model should learn from vehicles with comparable dynamics. Since large vehicles accelerate, brake, and follow diferently from passenger vehicles, we restrict the modeled population to the four passenger-vehicle classes, sedan (35.3%), midsize (34.9%), van (2.9%), and pickup (16.1%). Semis (9.6%) and trucks (1.2%) are excluded as modeled drivers. Trajectories are short overall (median duration 10.3 s, mean 14.1 s), so a minimum duration filter of 10 seconds is applied to ensure derived dynamic quantities and state evolutions can be reliably computed. The resulting eligible set contains 253,442 trajectories.

The eligible set of trajectories exhibits substantial behavioral diversity. Instantaneous speeds range from near stationary to 102 ft/s at the 90th percentile (median 49.9 ft/s), and the per lane mean speed trajectories in Figure 10a reveal a coherent slowdown afecting all four lanes during the congested interval, with modest cross lane variation in both timing and magnitude. The distribution of vehicle spacing in Figure 10b is concentrated at short following distances but extends across the full observation window, consistent with the mix of dense and free flowing periods captured by the recording.

![](images/7d3c9fb9c785016138e2ad0e2fdfad83a5a72023a447479c9d26fe90ba037b39.jpg)  
(a) Mean speed per lane over time.

![](images/df73586aca595a08a8440ddd19f717a5bf15a630048ad4fd777b42f12bde37a0.jpg)  
(b) Distribution of vehicle spacing.  
Figure 10: Speed (a) and spacing (b) characteristics of the eligible westbound population. All four lanes follow a common congestion profile, with modest cross-lane variation in onset and recovery. Spacing is concentrated at short following distances but extend across the full observation window.

## 4.2. Training Environment

The model is trained by minimizing the penalized NLL defined in Section 3.6 over all eligible trajectories. Behavioral and contextual variables are standardized, and the behavioral variables are lifted into a $D = 1 0 0$ dimensional feature space using the random Fourier map (Section 3.3). We use $K = 3$ profiles, each parameterized by a factor $V _ { k }$ of rank 10, so that every profile can express its full behavioral structure.

Training uses the Adam optimizer (learning rate 0.005) for 15 epochs. Because each driver’s state evolves sequentially, observations are processed in ordered chunks of 5,000 frames rather than shufled batches, and each driver’s state is carried across chunks within an epoch. The persistence parameter is fixed at $\alpha = 0 . 2$ and the entropy-penalty weight at $\gamma = 4 ;$ the adaptation parameter η is learned. Training was performed on Swan, the University of Nebraska–Lincoln Holland Computing Center cluster, using an Intel Xeon Gold 6348 node with 256 GB RAM.

The behavioral and contextual variables come from the selection procedure in Appendix A. The remaining settings including K, γ, D and variable subsets are examined in the ablation study in Appendix B, which supports the values used here.

## 4.3. Training Objective and Evaluation

The NLL is not a separate metric but the training objective itself (Equation 10). A low NLL means the model assigns high probability, given by the Born-rule value from Equation 8, to the behavior drivers actually exhibited.

To interpret the achieved per-observation NLL, we compare it against two reference points. The first is an uninformative model that has learned nothing about behavior. It is a maximally mixed state $\rho = I _ { D } / D$ , which assigns every observation the same probability regardless of what the driver did. This gives the worst-case $\mathrm { N L L } = \log D \approx 4 . 6 0 5$ for $D = 1 0 0$ . The second is a single density matrix we fitted to all drivers at once, with no profiles, no context dependence, and no state evolution. This static one-size-fits-all representation reaches $\mathrm { N L L } = 2 . 2 3 0$ . The full framework, with $K = 3$ profiles, context-dependent contributions, and state evolution, reaches $\mathrm { N L L } = 0 . 8 8 0$ . The reduction from 2.230 to 0.880 is the part that matters: it shows the gain comes from the profiles, context, and evolution, not from the density-matrix form alone. Equivalently, each observed behavior is on average $e ^ { 2 . 2 3 0 - 0 . 8 8 0 } \approx 3 . 9$ times more likely under the full framework than under the single static matrix.

## 5. Behavioral Analysis

This section analyzes what the trained model learned. We begin by establishing that the three profiles. Then, we explain the identity of each profile. We finally look into the distinct modes within the multi-mode profile.

## 5.1. Establishing Three Distinct Profiles

Before interpreting the profiles, we establish that they represent genuinely distinct behavioral regimes rather than arbitrary partition of the data. We characterize each profile, a density matrix $\rho _ { k } ,$ by its internal rank structure, its behavioral signature, the way its variables interact, its separation from the other profiles, and the contexts that activate it. Each is read from a diferent property of the density matrix, and together they show the three profiles are separate, interpretable regimes.

Internal rank structure. The rank of a profile determines whether it captures one behavioral mode or several. Table 3 reports the leading eigenvalues of each $\rho _ { k } \mathrm { ; }$ ; because a density matrix has trace one, the eigenvalues sum to one and act as weights on the profile’s behavioral modes. Profiles 1 and 3 are efectively rank one, with more than 99% of their weight in a single eigenvalue, so each represents one coherent behavioral pattern. Profile 2 is diferent: its leading eigenvalue is 0.85 and the next four account for a further 14%, indicating several behavioral modes coexisting within one profile. We examine this internal structure in Section 5.3.

Table 3: Top eigenvalues of each profile. Profile 2 carries multi-modal structure that the other two profiles lack.
<table><tr><td></td><td> $\lambda _ { 1 }$ </td><td> $\lambda _ { 2 }$ </td><td> $\lambda _ { 3 }$ </td><td> $\lambda _ { 4 }$ </td><td> $\lambda _ { 5 }$ </td></tr><tr><td>Profile 1</td><td>0.999</td><td>0.001</td><td> $< 1 0 ^ { - 6 }$ </td><td> $< 1 0 ^ { - 6 }$ </td><td> $< 1 0 ^ { - 6 }$ </td></tr><tr><td>Profile 2</td><td>0.852</td><td>0.077</td><td>0.034</td><td>0.020</td><td>0.011</td></tr><tr><td>Profile 3</td><td>0.998</td><td>0.002</td><td> $< 1 0 ^ { - 7 }$ </td><td> $< 1 0 ^ { - 7 }$ </td><td> $< 1 0 ^ { - 7 }$ </td></tr></table>

This distinction can be quantified by the purity of a profile,

$$
\mathcal { P } ( \rho _ { k } ) = \mathrm { T r } ( \rho _ { k } ^ { 2 } ) = \sum _ { m } \lambda _ { m } ^ { 2 } ,\tag{11}
$$

where $\lambda _ { m }$ are the eigenvalues of $\rho _ { k }$ . Recall that each profile is parameterized with rank $r = 1 0 .$ so it can place weight on at most r behavioral modes (Section 3.4). The purity then ranges from $1 / r _ { ; }$ , when weight is spread equally across all modes, to 1, when a single mode carries all of it. Purity is a standard measure of spread in quantum information. The three profiles give $\mathcal { P } ( \rho _ { 1 } ) = 0 . 9 9 9$ $\mathcal { P } ( \rho _ { 3 } ) = 0 . 9 9 7 .$ , and $\mathcal { P } ( \rho _ { 2 } ) = 0 . 7 3 4$ , confirming the eigenvalue analysis: Profiles 1 and 3 are efectively rank one, whereas Profile 2 retains substantial weight across several modes.

Separation in profile space. The profiles are also far apart as density matrices. Table 4 reports the pairwise Frobenius distances between the profiles. For reference, the maximum distance between any two rank-one density matrices is ${ \sqrt { 2 } } \approx 1 . 4 1 4$ . Profile 1 lies very far from both Profile 2 and Profile 3, reaching 93% and 97% of this maximum distance, respectively. Profiles 2 and 3 are closer to one another, at only 35% of the maximum distance. The three profiles are therefore genuinely distinct objects rather than overlapping variants of one behavior.

Table 4: Pairwise Frobenius distances between profiles. The geometric maximum for any two rank-one density matrices is ${ \sqrt { 2 } } \approx 1 . 4 1 4$
<table><tr><td></td><td>Profile 1</td><td>Profile 2</td><td>Profile 3</td></tr><tr><td>Profile 1</td><td></td><td>1.314</td><td>1.366</td></tr><tr><td>Profile 2</td><td>1.314</td><td></td><td>0.489</td></tr><tr><td>Profile 3</td><td>1.366</td><td>0.489</td><td></td></tr></table>

Behavioral signatures. We next ask what behavior each profile represents. We answer this from the data directly. Every observed frame, a single observation of one vehicle at 25 Hz, is assigned to the profile that carries the largest contribution weight there. For each profile, we then take the frames it was assigned and compute the mean and standard deviation of the raw behavioral variables: speed, spacing, and jerk (Figure 11).

The three profiles are clearly separated by speed and spacing. Profile 1 corresponds to highspeed driving with the largest spacing $( 6 2 . 2 \pm 2 8 . 5 \mathrm { f t / s } , 1 9 3 . 7 \pm 1 2 7 . 9 \mathrm { f t ) }$ . Profile 2 represents the slowest driving and the shortest spacing $( 2 7 . 4 \pm 2 0 . 7 \mathrm { f t / s } , 1 1 7 . 6 \pm 1 0 3 . 6 \mathrm { f t ) }$ . Profile 3 falls between the two on both dimensions $( 5 3 . 5 \pm 2 6 . 5 \mathrm { f t / s } , 1 5 4 . 5 \pm 1 1 3 . 1 \mathrm { f t } )$

Jerk behaves diferently. All three profiles have near-zero mean jerk and a similar spread (1.26– $\mathrm { 1 . 4 0 f t / s ^ { 3 } ) }$ , so jerk does not separate the profiles on its own. Its role instead comes through its interaction with speed. Additionally, jerk carries extra meaning for Profile 2, where positive and negative jerk distinguish the modes within the profile as discussed in (Section 5.3).

![](images/974c2ffbbaf29ff102dce68356fe33c5281783f82aa98e7d6c3bc900f91bbcab.jpg)  
Figure 11: Empirical behavioral signatures per profile. Each row shows $\mu \pm \sigma$ of speed, spacing, and jerk over the frames dominated by that profile.

Variables interactions A profile can also difer in how its variables relate, not only in their average values. Because each profile is a density matrix over the RFF feature space, the Born rule of Eq. (8) assigns a probability to any combination of speed, spacing, and jerk. Using these probabilities, we measure how strongly any two of the three variables are related within a profile through their mutual information. For two variables, mutual information is the amount of information one carries about the other: it compares their actual joint distribution against what it would be if they were independent, and equals

$$
I ( A ; B ) = \sum _ { a , b } p ( a , b ) \log \frac { p ( a , b ) } { p ( a ) p ( b ) } ,\tag{12}
$$

where $\textstyle p ( a , b )$ is obtained from the profile’s Born-rule probabilities and $p ( a ) , p ( b )$ are its marginals. It is zero when the two variables are independent and grows as they become more dependent. The results are shown in Table 5.

Table 5: Pairwise variable interaction, measured by mutual information (in nats), under each profile.
<table><tr><td>Profile  $I ( v ; \Delta s )$ </td><td> $I ( v ; j )$   $I ( \Delta s ; j )$ </td></tr><tr><td>Profile 1 0.049</td><td>0.149 0.228</td></tr><tr><td>Profile 2 0.005</td><td>0.052 0.054</td></tr><tr><td>Profile 3 0.012</td><td>0.133 0.062</td></tr></table>

Profiles 1 and 3 couple on diferent pairs of variables. Looking at the pair with the highest mutual information, Profile 1 is dominated by a spacing–jerk coupling (0.228) and Profile 3 by a speed–jerk coupling (0.133). Profile 2 shows weak coupling across all pairs, consistent with its behavior being spread over several modes rather than following one tight relationship. The framework recovers these couplings without being told which variables interact. They emerge from the fitted density matrices.

Explaining jerk through speed. The speed–jerk relationship deserves a closer look because jerk is the one behavioral variable whose marginal distribution does not separate the profiles, yet Table 5 shows it is strongly coupled to speed in Profiles 1 and 3. Figure 12 shows why. Although the profiles span a similar range of jerk values, but the speeds at which those jerk events occur difer sharply. Profile 1’s jerk events occur at high speed $( 6 0 { - } 1 2 0 \mathrm { f t / s ) }$ . Profile 2’s occur at low speed $\left( 0 { - } 4 0 \mathrm { f t / s } \right)$ Profile 3’s fall in the middle. Jerk therefore distinguishes the profiles through its relationship with speed rather than through its marginal distribution. This is the kind of interaction between variables that the density-matrix representation is built to capture. Jerk also plays a further role inside Profile 2, where it separates the profile’s internal modes; we return to this in Section 5.3.

Contextual activation. So far the profiles have been characterized by their behavior. They are also distinguished by when they arise. Each profile is activated by a diferent trafic context, through the learned coeficients $\beta _ { k }$ that link context to each profile’s contribution weight. A positive coeficient means the profile becomes more active as that context variable increases, a negative one means it is suppressed. Figure 13 reports these coeficients. Profile 2 is activated strongly by density $( \beta _ { d } = + 6 . 3 9 )$ and modestly suppressed by speed entropy. Profile 3 shows the opposite density response $( \beta _ { d } = - 5 . 1 1 )$ and is instead activated by acceleration entropy $\left( \beta _ { H _ { a } } = + 5 . 1 8 \right)$ . Profile 1 is most active when acceleration entropy is low $\left( \beta _ { H _ { a } } = - 3 . 8 1 \right)$ and is otherwise relatively insensitive to context. The three profiles are therefore triggered by distinct conditions, which reinforces that they capture diferent regimes rather than arbitrary groupings.

![](images/95b879c1003a9267884acae204b84c96e168799720d73ff072512e4815afe068.jpg)

![](images/78ac4abca2c80f5c146fc8116feab89b56bda27fad6d736aab07701f2604c3b6.jpg)

![](images/deb854504703ea77c68a6283046221a7f2bfe498bf82622eca9b09a52a54faa2.jpg)  
Figure 12: Joint distributions of speed and jerk for the three profiles. Dashed and solid contours enclose the 50% and 90% density regions, respectively.

![](images/584b0e8fedc5924edbc29848a0d01750db2a8909227d825ab761f45b13d9626a.jpg)

![](images/ca718c8b4f4389d12014586fe54181a72f749de836a73eb0f8446df9e8549104.jpg)

![](images/b70a22bfff7722cd12d395e68777a4d995897d7a3260bb1874832dde406f774e.jpg)  
Figure 13: Learned context coeficients $\beta _ { k }$ for each profile. Positive coeficients increase a profile’s contribution as the context variable rises; negative coeficients suppress it.

## 5.2. Naming the Profiles.

Each profile has now been characterized in several ways: its behavioral variable distributions, its internal structure, the way its variables interact, its distance from the other profiles, and the context that activates it. This characterization identifies each profile with a familiar driving regime. We name them accordingly.

Profile 1, free flow. This is the fastest regime with the largest spacing $( 6 2 . 2 \mathrm { f t } / \mathrm { s } ,$ 193.7 ft on average) and the most concentrated of the three $( \mathcal { P } ( \rho _ { 1 } ) = 0 . 9 9 9 )$ , meaning it captures a single, coherent way of driving. Its strongest internal coupling is between spacing and jerk $\left( I = 0 . 2 2 8 \right)$ That is, the acceleration variation is tied to how much room the driver has, rather than speed. It activates when the surrounding trafic is smooth $\left( \beta _ { H _ { a } } ~ = ~ - 3 . 8 1 \right)$ These properties describe unobstructed, high-speed cruising, a driver with space, holding a steady pace, largely unconstrained by neighbors.

Profile 2, congested stop-and-go. This is the slowest regime with the shortest spacing (27.4 ft/s, 117.6 ft). Unlike the other two, it is not a rank one profile. Its purity is far lower $( { \mathcal { P } } ( \rho _ { 2 } ) = 0 . 7 3 4 )$ and its weight is spread across several modes rather than one. It is driven almost entirely by density $( \beta _ { d } = + 6 . 3 9 )$ , reflecting congested car-following. Diferent driving behaviors emerge within this congested regime, and they are explored in Section 5.3.

Profile 3,transitional following. This regime lies between the other two in both speed and spacing (53.5 ft/s, 154.5, ft). Like Profile 1, it represents a single, consistent driving behavior $( \mathcal { P } ( \rho _ { 3 } ) ~ = ~ 0 . 9 9 7 )$ . Unlike the congested regime, it is less likely to occur as density increases $( \beta _ { d } = - 5 . 1 1 )$ but becomes more likely when acceleration entropy is high $\left( \beta _ { H _ { a } } = + 5 . 1 8 \right)$ . In other words, it appears when surrounding trafic becomes disrupted, even though the driver maintains a relatively steady pace. Its strongest interaction is between speed and jerk $\left( I = 0 . 1 3 3 \right)$ , suggesting that the driver is continuously making small speed adjustments in response to changing trafic conditions. This reflects attentive driving in unsettled but still-moving trafic.

These identities also explain the separations in Table 4. Profile 1’s large distance from the other two matches its role as the free-flow regime, which is structurally distinct from the two constrained driving regimes. Profiles 2 and 3 are much closer to each other because they describe neighboring stages of the same congestion cycle: the transitional, still-moving trafic of Profile 3 and the congested stop-and-go of Profile 2. Their proximity in profile space mirrors their proximity in trafic: drivers pass through Profile 3 on the way into and out of the Profile 2 congestion, moving smoothly between the two rather than switching abruptly between separate categories.

## 5.3. Inside Profile 2: How Stop-and-Go Driving Decomposes

Profile 2 is the only multi-modal profile in our fit (Table 3). Its top five eigenvalues are $\lambda _ { 1 } =$ 0.852, $\lambda _ { 2 } = 0 . 0 7 7$ $\lambda _ { 3 } = 0 . 0 3 4$ $\lambda _ { 4 } = 0 . 0 2 0$ , and $\lambda _ { 5 } = 0 . 0 1 1$ . Each eigenvalue points to a behavioral mode hidden inside the profile. In this subsection we examine what those modes describe.

Before going further, we exclude Modes 4 and 5 from the behavioral interpretation. They carry only a collective 3.1% of the profile’s mass $( \lambda _ { 4 } = 0 . 0 2 0 , \lambda _ { 5 } = 0 . 0 1 1 )$ , and when they do dominate a frame they tend to appear briefly and in short, scattered bursts rather than in coherent stretches that would tie them to a recognizable driving pattern. We report their eigenvalues for transparency but focus the interpretation on Modes 1–3.

Figure 14 shows the empirical speed, spacing, and jerk signatures of the three modes, computed from the frames where each mode dominates the decomposition of Profile 2.

![](images/5ea168a3c125ef3fb6285f440ecce8d1f05a3e2ebd106a27a2f7f1f8e1db18a0.jpg)

![](images/71a09a0aaf3c8f2e8564efd9b3822dd41bcd281cf12564af4fa1fefb99266dfa.jpg)

![](images/e1c05a531884c41d594a7b460b9e8c5cfdc6ba983305db42a44facb73412e01c.jpg)  
Figure 14: Empirical signatures of the three modes within Profile 2. Markers show mode means and cross-hairs indicate ±1 standard deviation for the frames associated with each mode.

Mode 1 represents steady congested following. Drivers travel at $2 8 . 4 \pm 2 0 . 6 ~ \mathrm { f t / s }$ with a spacing of $1 2 8 \pm 1 2 1$ ft, while jerk remains centered near zero $( - 0 . 0 9 \pm 1 . 2 7 ~ \mathrm { f t / s ^ { 3 } } )$ . It carries 85.2% of the profile’s mass and accounts for 93% of the frames where Profile 2 is active. In other words, Mode 1 is what congested driving looks like most of the time: a vehicle matching its leader’s pace with no hard acceleration or braking.

Mode 2 captures acceleration events. Speed and spacing are broadly similar to Mode 1, but jerk becomes strongly positive $( + 1 . 8 5 \pm 1 . 7 1 ~ \mathrm { f t / s ^ { 3 } } )$ . These are the moments when the leader pulls away or leaves the lane and the driver accelerates to close the gap. The mode appears briefly, reflecting the short duration of such events.

Mode 3 captures braking events. Its speed and spacing distributions are again similar to Mode 1, but jerk becomes strongly negative $( - 3 . 1 0 \pm 2 . 9 5 ~ \mathrm { f t / s ^ { 3 } } )$ . These episodes occur when the

driver slows down in response to a disturbance ahead in order to maintain a safe following distance.   
Mode 3 is rare $( \lambda _ { 3 } = 0 . 0 3 4 , 0 . 3 \%$ of frames) and appears only in short bursts.

Profile 2 in motion. Mode 1 is a state; Modes 2 and 3 are events. Drivers spend most of their time in Mode 1, routine car-following in congested trafic, and pass briefly through Mode 2 when they accelerate into an opening gap or Mode 3 when they brake in response to a disturbance ahead. Figure 15 shows how these three components play out along a single congested trajectory from I-24, with the dominant mode shaded at each frame.

The trajectory illustrates the state-and-event structure directly. For most of the 22 seconds the vehicle remains in Mode 1 (blue): speed stays around 20, ft/s, the gap gradually closes, and jerk remains small, reflecting the steady following that dominates this profile. The departures from Mode 1 occur at the labeled events. Near 12, s, as the vehicle finishes decelerating into the slowdown and the gap reaches its minimum, the driver releases the brake, resulting in a positive jerk peak, and Mode 2 (orange) becomes dominant. A second Mode 2 episode appears near 16, s during acceleration, as the gap widens and the vehicle speeds up resulting in another positive jerk peak. Finally, near 20, s, jerk becomes strongly negative as the driver brakes, triggering a transition to Mode 3 (red). These transitions occur with only small changes in speed and spacing, but with clear changes in jerk. This confirms that jerk is a key variable in distinguishing the modes within Profile 2, which rely on acceleration and braking response to changing trafic conditions.

![](images/0fae372fba4a043be6739f0afad1a8abb5298ab2276c35274f1e0a5d2c868269.jpg)  
Figure 15: Example Profile 2-dominated trajectory from I-24 MOTION. Speed, jerk, and spacing are shown over time, with background shading indicating the dominant mode of Profile 2 at each frame. Mode changes track the sign of jerk, which shows congested driving is separated by acceleration behavior.

## 5.4. State Evolution: Drivers Are Not Labels

So far we have described the three profiles and the modes within Profile 2. What we have not yet shown is how profile composition evolves over time. The profile mixture $\pi _ { k } ( c _ { t } )$ is not a fixed label assigned to a driver. It changes as driving conditions change. Figure 16 follows one driver over 26 seconds on I-24 as the vehicle approaches slower trafic downstream, and shows why each profile takes over when it does.

At the beginning, the driver travels at about $6 0 \mathrm { f t / s }$ with a large spacing to the leader and low surrounding density. The composition is dominated by Profile 1, the free-flow regime, which the model activates when acceleration entropy is low and the surrounding trafic is smooth. Around the 8-second mark, speed begins to fall and the surrounding motion becomes irregular. This is the condition Profile 3 responds to: it is suppressed by density but activated by acceleration entropy, so it takes over precisely in these unsettled, still-moving moments before congestion sets in. By about 13 seconds, speed has dropped below $2 0 \mathrm { f t / s }$ , density has risen sharply, and the spacing has compressed. Density is the dominant trigger of Profile 2, so as it climbs, Profile 2 becomes dominant and the driver settles into congested stop-and-go. Each takeover therefore matches the profile whose learned activation conditions are satisfied. The transitions are smooth rather than abrupt, and the model is never given these regimes as labels. They emerge from the joint structure of behavior and context.

![](images/59655c0644200c5142ddbe0fe5e8c95300196648fb0ee6c2a7c24b1ec6f41eea.jpg)  
Figure 16: State evolution for a driver entering congestion. The bottom panel shows the profile composition $\pi _ { k } ( c _ { t } )$ over time. As trafic conditions change, the driver’s state transitions smoothly from Profile 1 (free flow) to Profile 3 (transition) and finally to Profile 2 (congestion).

This type of behavioral transition is not available in classical car-following models. Models such as Newell (Newell, 2002) and the IDM (Treiber et al., 2000) assume a fixed set of driver parameters, so the same behavioral characteristics apply across both free-flow and congested conditions. Some extensions introduce time-varying behavior. The Extended Asymmetric Behavior model (Zhong et al., 2024), for example, varies time headway and minimum spacing through a predefined multiplier whose shape is sampled once per vehicle and triggered by a deceleration threshold. Although the parameters change over time,their evolution follows a prescribed pattern rather than the surrounding trafic state.

Our approach instead makes the profile composition $\pi _ { k } ( c _ { t } )$ depend on the observed context at each instant. As trafic conditions evolve, the driver moves smoothly between data-driven behavioral regimes (free flow, transition, and congestion), with transitions shaped by both the current context and recent history. And at every instant the composition is a distribution over regimes, not a hard assignment, so a driver can be mostly in one regime while still carrying weight in others. The evolution in Figure 16 is therefore a continuous, history-dependent, and probabilistic trajectory through interpretable behavioral regimes learned from data, rather than a predetermined parameter curve.

## 6. Connection to Macroscopic Trafic Phenomena

Section 5 described the profiles, modes, and state dynamics recovered by the framework. Here we examine how those findings relate to established trafic flow theory. In particular, we ask whether the learned structure aligns with well-known trafic phenomena. Such agreement would suggest that the representation is capturing meaningful aspects of trafic behavior. Two connections are especially relevant: the fundamental diagram and hysteresis.

## 6.1. Connection to Fundamental Diagram

The fundamental diagram provides a useful way to examine whether the profiles correspond to recognizable trafic regimes. If the profiles capture meaningful behavioral structure, they should align with diferent regions of the flow density relationship, even though the model was never trained on aggregate trafic measures. We construct the fundamental diagram from the westbound I-24 MOTION dataset using Edie’s generalized definitions of flow, density, and space mean speed (Edie, 1965), aggregated over 1000, ft 30s space time cells. To overlay the learned representation, we average the profile contribution weights $( \pi _ { k } ( c _ { t } ) )$ across all inference frames within each cell and color the cell according to its dominant profile.

Figure 17 shows a clear separation of regimes. Profile 1 occupies the low density, high speed portion of the diagram. Profile 3 dominates much of the region approaching capacity, while Profile 2 occupies the congested branch at higher densities. The profiles therefore appear in the same order as the trafic regimes represented in the fundamental diagram: free flow, transition, and congestion.

The model was never given the density-flow relationship or any trafic-regime labels. Its context variables include local density and the entropy of surrounding speeds and accelerations, but it was never shown how these aggregate in the density-flow plane. It was also not shown the regime to which an observation belongs. Yet the resulting profiles align closely with that macroscopic structure. The separation is weakest near capacity, where the free-flow and congested branches of the fundamental diagram physically overlap. There, observations at similar flow and density genuinely belong to diferent regimes, so profile mixing is expected. A few observations also appear of-regime away from this overlap. Such observations likely reflect cases where some drivers behaved according to a less typical profile for the local conditions, a sign of genuine behavioral heterogeneity. Sharper separation could be obtained by allowing more profiles, though we retain $K = 3$ for interpretability and profile distinctness.

## 6.2. Hysteresis

A well-known feature of trafic flow is that drivers entering congestion and drivers leaving congestion do not necessarily follow the same path in the density-flow plane. Laval (2011) showed that these hysteresis loops can be positive, negative, or negligible depending on driver behavior. If our profile composition $\pi _ { k } ( c _ { t } )$ captures how drivers move through trafic states, it should reflect this asymmetry as well.

We examine the stop and go wave shown in Figure 18 (top left). Following the method introduced in Laval (2011), we place a sequence of parallelogram regions along a representative trajectory and compute flow, density, and profile composition within each region. The resulting trajectory in the density-flow plane is shown in Figure 18 (top right).

The platoon begins in free flow, passes through moderate dense trafic, enters congestion, and then recovers. The flow density trajectory forms a negative hysteresis loop, with the acceleration branch lying above the deceleration branch. The profile composition follows the same progression: Profile 1 dominates before the wave, Profile 3 dominates during the transition into congestion, and

![](images/446ec85ffabc80a68d23111672da15a7f7e7b5a86d0f222740b765f793f25e89.jpg)  
Figure 17: Fundamental diagram for the westbound I-24 MOTION dataset, with cells colored by the dominant inferred profile. Profile 1 occupies the low density free flow region, Profile 3 dominates near capacity, and Profile 2 occupies the congested branch.

Profile 2 dominates inside the wave before the composition shifts back toward Profiles 1 and 3 during recovery.

The hysteresis becomes visible when comparing points with similar speeds. Point 4 and Point 8 are both near 20 mph, yet their profile compositions difer sharply. At Point 4, entering the trafic wave, the composition is dominated by Profile 3, the transitional following regime activated by high acceleration entropy $\left( H _ { a } \right)$ . As the driver decelerates into congestion, the surrounding trafic exhibits irregular and turbulent motion, making this profile the dominant representation. At Point 8, leaving the wave, the composition has shifted toward Profile 1, the free-flow regime. At this stage, the driver is still slow, but the surrounding trafic has begun to smooth out, so the calmer profile takes over before speed fully recovers. The two drivers share a speed but sit at diferent stages of the congestion cycle, and the representation separates them because it responds to surrounding context, not speed alone.

The post-recovery point (Point 9) reinforces this observation. Even at 53 mph, well into freeflow speeds, the composition still retains a substantial share of Profile 2, the stop-and-go profile. In contrast, Point 1, with nearly the same speed, is dominated by Profile 1. The diference is trafic history. A vehicle leaving congestion has recently experienced short spacing and stop-and-go motion. Through the persistence term α in the state update (Equation 7), this recent history continues to influence the state $\rho _ { i } ( t )$ even after trafic begins to recover. As a result, Point 9 retains a residual contribution from Profile 2, whereas Point 1 does not. This persistence explains the hysteresis: the representation reflects not only the current context and driving behavior, but also the recent driving history.

## 7. Applications to Driver and Autonomous Vehicle Models

Having shown what the representation recovers about driver behavior and trafic dynamics, we now turn to how it can be used. We consider two applications: integration with existing car-

![](images/17854c033f1ac31626b4ee15df8ddfc3ee4f961e24cace94874f3981af0ebfb0.jpg)

![](images/e54e53d5b14015c2e702e758f5bc135ea75287258970c02b7ace38c87481d010.jpg)

Profile composition along the platoon path  
![](images/a3e4a375e5abde74d67beb20e479abb49651361e3f9d280a9a0a1825a80853e7.jpg)  
Figure 18: Hysteresis on a westbound I-24 MOTION wave (lane 1). Top left: the wave on the time-space diagram. Top right: the resulting (density, flow) traced through a representative trajectory. Bottom: the average profile composition π (c ) at each point.

following models, where parameter quantities are instead derived from the driver’s evolving state, and use in autonomous and mixed trafic, where the representation serves as a compact belief state for anticipating how surrounding drivers will behave.

## 7.1. Integration with Driver Models

Driver models provide the computational framework through which behavioral assumptions are translated into vehicle motion in microscopic trafic simulation. Integrating the proposed representation with these models therefore enables existing simulations to incorporate dynamically evolving behavioral heterogeneity without fundamentally changing their underlying structure. Because carfollowing models form the basis of longitudinal vehicle interactions in most microscopic simulators, we use them to illustrate the integration.

Car-following models such as Newell (Newell, 2002), Intelligent Driver Model (Treiber et al., 2000), Krauss (Krauss et al., 1997), and Optimal Velocity Model (Bando et al., 1998) compute a vehicle’s acceleration from its current situation using a fixed set of parameters. These parameters can be broadly categorized into two groups. Target parameters define the driver’s desired steady-state behavior, such as the desired speed, preferred time headway, or jam spacing. Response parameters govern how strongly the driver reacts to deviations from these desired conditions, such as the maximum acceleration or comfortable deceleration. In their conventional formulation, both parameter groups are calibrated once for a given driver and treated as constant.

Our representation changes how the target parameters are computed. At any time t, the driver’s state $\rho _ { i } ( t )$ defines a distribution over behavior, from which we can read the average value of any behavioral quantity, for instance the driver’s speed or spacing when nearly stopped. For a behavioral quantity $^ { g , }$ which is a value computed from a behavioral vector $x ,$ this average is

$$
\mathbb { E } [ g \mid \rho _ { i } ( t ) ] = \int g ( x ) p ( x \mid \rho _ { i } ( t ) ) d x , \qquad p ( x \mid \rho _ { i } ( t ) ) = \tilde { \phi } ( x ) ^ { \top } \rho _ { i } ( t ) \tilde { \phi } ( x ) ,\tag{13}
$$

where $p ( x \mid \rho _ { i } ( t ) )$ is the probability the state gives the behavioral vector x through the Born rule. Each target parameter is obtained this way: the desired speed is the state’s average free-flow speed, and the jam spacing is the state’s average gap as speed approaches zero. Because the state evolves with context, these targets evolve with it, so the same driver has a lower desired speed inside congestion than outside ${ \mathrm { i t } } ,$ and two drivers who have the same spacing can aim for diferent speeds. The response parameters, such as maximum deceleration, are not behavioral targets and are not obtained this way. They are set separately, from vehicle limits or reference values.

At each step, the state used to compute these expectations is the same as the predicted state introduced in Section 3.5

$$
\rho _ { i } ^ { \mathrm { p r e d } } ( t ) = ( 1 - \alpha ) \rho _ { i } ( t - 1 ) + \alpha \sum _ { k = 1 } ^ { K } \pi _ { k } ( c _ { t } ) \rho _ { k } ,
$$

which carries the driver’s recent behavior through the first term and the current context through the second.

Example: The Intelligent Driver Model (IDM). IDM computes acceleration as a function of current speed $v ,$ current spacing s, and the speed diferential $\Delta v$ . Its target parameters are the desired speed $v _ { 0 }$ , time gap T, and jam spacing $s _ { 0 }$ . Its response parameters are the maximum acceleration a and comfortable deceleration b. The three target parameters are supplied by the state,

$$
\begin{array} { r } { v _ { 0 } ( t ) = \mathbb { E } [ v \mid \rho _ { i } ^ { \mathrm { p r e d } } ( t ) ] , \quad T ( t ) = \mathbb { E } [ \Delta s / v \mid \rho _ { i } ^ { \mathrm { p r e d } } ( t ) ] , \quad s _ { 0 } ( t ) = \mathbb { E } [ \Delta s \mid \rho _ { i } ^ { \mathrm { p r e d } } ( t ) , v  0 ] , } \end{array}
$$

while a and b are set separately. Substituting these into IDM gives an acceleration whose targets evolve with the state. The equation is unchanged and only the source of its targets changes, from fixed calibration to the evolving state (Figure 19).

We illustrate this integration with the example shown in Figure 20. The leader begins in free flow near 92 $\mathrm { f t / s }$ , decelerates into a stop-and-go wave with speeds oscillating near 10–20 $\mathrm { f t / s , }$ and then recovers to free flow. The first follower is a standard IDM with fixed target parameters $( v _ { 0 } = 9 2 \mathrm { f t / s } ,$ $T = 1 . 5 \mathrm { s } , s _ { 0 } = 6 . 5 \mathrm { f t } )$ . The second follower is a representation-supported IDM that reads its target parameters $v _ { 0 } ( t ) , T ( t )$ , and $s _ { 0 } ( t )$ at each step from the driver state $\rho _ { i } ( t )$ , which evolves as the surrounding context changes. Context here includes trafic density and the speed and acceleration entropy of the surrounding trafic, moving from smooth, low-density conditions in free flow, through a disrupted transition, into dense congested conditions. The bottom panel shows the resulting profile composition $\pi _ { k } ( c _ { t } )$

The result is a sensible car-following trajectory: the representation-supported follower slows as the leader slows and recovers afterward, tracking the leader throughout while adopting diferent desired speed and spacing in each regime. As the context shifts, the state moves through the three profiles, from Profile 1 in free flow, briefly through Profile 3 during the disrupted transition, to Profile 2 inside the congestion. The integration therefore works as intended and the IDM now adapts continuously to context through the evolving state.

![](images/b137eb0a990aa5359c2cbf4e4608d11ad7f6e8761457ae8773082828a79659a9.jpg)

Figure 19: Integration of the representation with IDM. The model’s equation is unchanged. Its target parameters v<sub>0</sub>, T, s<sub>0</sub> are supplied by the evolving driver state and adapt to context.  
![](images/b647f89b1612319d9ec59e6d9da0542a27cb997863fce849b38c0b30a93d4410.jpg)  
Figure 20: Representation-supported car following. A single leader (purple) drives through a free-flow, stop-and-go and recovery cycle. A standard IDM follower (dashed) uses fixed parameters. A representation-supported IDM follower (red) uses the same IDM equation but reads its target parameters ${ v _ { 0 } } ( t ) , T ( t ) , s _ { 0 } ( t )$ from the evolving state $\rho _ { i } ( t )$ . Top: speed. Middle: position. Bottom: profile composition $\pi _ { k } ( c _ { t } )$ , shifting from Profile 1 (free flow) through Profile 3 (transition) to Profile 2 (congestion) as context changes.

Car-following using acceleration from the state. Supplying target parameters still relies on a fixed model equation to turn them into acceleration. This step can be removed in a variant of our representation that includes acceleration in the behavioral vector and the speed diferential $\Delta v$ in the context. The state then represents acceleration directly, and the representation provides it

without any car-following equation:

$$
\begin{array} { r } { \dot { v } _ { i } ( t ) = \mathbb { E } \big [ \dot { v } \mid \rho _ { i } ^ { \mathrm { p r e d } } ( t ) , s , v , \Delta { v } \big ] , } \end{array}\tag{14}
$$

where $\dot { v } _ { i } ( t )$ is the average acceleration the state considers likely given the driver’s current state and context. This requires no target or response parameters and imposes no fixed functional form; the response comes entirely from the evolving state given by a model learned directly from naturalistic driving and adapts to context on its own.

## 7.2. Implications for Autonomous and Mixed Trafic

Although developed as a representation of human driving behavior, the framework has several implications for autonomous and mixed trafic systems. The representation is compact, contextdependent, history-aware, and evolves continuously across behavioral profiles. All of these are desirable properties for modeling surrounding human drivers for the benefit of an AV. Two implications follow directly from the results presented in this paper.

Forecasting surrounding drivers. The model processes a driver one frame at a time, updating the state with each new observation instead of recomputing it. This means that at any moment we hold a current state for every surrounding vehicle, and we can roll that state forward to predict what the vehicle will do next. We test this by forecasting each vehicle’s speed 3 s ahead.

We forecast the behavior of one vehicle at a time. The vehicle has been observed up to the present, and passing those observations through the model gives its current state. To forecast, we continue evolving that state forward for 3 s using the same update rule as in training. That rule has two steps at each frame:the state first moves toward the profile mixture activated by the current context,then is adjusted by the observed behavior at that frame,

$$
\rho _ { t + h } ^ { - } = \left( 1 - \alpha \right) \rho _ { t + h - 1 } + \alpha \sum _ { k } \pi _ { k } ( \hat { c } _ { t + h } ) \rho _ { k } , \qquad \rho _ { t + h } = \left( 1 - \eta \right) \rho _ { t + h } ^ { - } + \eta \tilde { \phi } ( \hat { x } _ { t + h } ) \tilde { \phi } ( \hat { x } _ { t + h } ) ^ { \top } .\tag{15}
$$

During forecasting the future context $\hat { c } _ { t + h }$ and behavior $\hat { x } _ { t + h }$ are not yet known, so we estimate them by extending the trend of the recent past using linear extrapolation. From each forecast state we then read the expected speed, evaluated at the estimated headway and jerk,

$$
\hat { v } _ { t + h } = \mathbb { E } \Big [ v \mid \rho _ { t + h } , \Delta s = \hat { s } _ { t + h } , j = \hat { j } _ { t + h } \Big ] .\tag{16}
$$

We evaluate this on lane 1 of I-24 MOTION under three diferent trafic conditions: calm, busy, and between waves, the last referring to the partially recovered region between two separate kinematic waves, where trafic has sped up but not returned to free flow. Figure 21 compares forecast and realized speed at the 3 s horizon, with mean absolute errors of 3.77, 4.51, and 4.98 mph from calm to busy conditions. For an AV, a reliable 3 s speed estimate for each surrounding vehicle provides a usable reaction window and supports proactive rather than reactive control.

The framework was trained only to represent behavior with no forecasting objective and no prediction-specific component. The forecast presented above reuses the trained model with only one change: a linear extrapolation of future inputs. The usable accuracy comes from the state, which already encodes how behavior evolves with context and history. A version built for prediction would extend the framework in two concrete ways: replacing the linear input extrapolation with a learned model of how context evolves, and adding a forecasting term to the training objective so the state is optimized to predict future behavior, not only to represent current behavior.

![](images/6941ef249326324e623e5916a23752ab030ba2749e788744d81a1f42e2948167.jpg)

![](images/4dfa202c6c2d314f2ea4266cea6b58b74be51b50fddb324af74dee378e505b51.jpg)  
Realized speed at t + 3 s (mph)

![](images/760389895ae318b6f6e09b0239e727848d50da2d2ebf286001fc7d67a86132f0.jpg)  
Realized speed at t + 3 s (mph)  
Figure 21: Short-horizon (3 s) forecast of human-driver speed from the trained state representation, evaluated on lane 1 of I-24 MOTION under calm, busy, and between-waves conditions. Each point is one vehicle; axes compare forecast and realized speed at the 3 s horizon, with the diagonal marking perfect prediction.

A behavioral read of the whole neighborhood. The same representation applied to one vehicle can be applied to every vehicle around it at once, giving an AV a real-time behavioral read of its entire neighborhood. Figure 22 shows this for lane 1 of I-24 MOTION: at a single instant, each inferred vehicle carries its own profile composition $\pi _ { k } ( c _ { t } )$ , shown as a stacked bar at its position along the corridor.

![](images/c9ab2f4e458eb9b88eba332b46c6fe965c9a5a168b8c0cfdb6f3d03831020c54.jpg)  
Figure 22: Profile composition along lane 1 of I-24 MOTION at three diferent instants. Each bar is one vehicle at its corridor position, split into profile weights $\pi _ { 1 } , \pi _ { 2 } , \pi _ { 3 }$ . Inferred vehicles cover roughly 20% of the lane-1 population.

The value lies in what this reveals. Vehicles that are close to one another in space, and therefore exposed to predominantly similar trafic context, do not appear as a behavioral monolith. Some remain concentrated in a single profile, while others occupy mixtures of multiple profiles. Behavior is not fixed by location and context alone; it also reflects each driver’s recent history, which the state retains. And because each composition is a Born-rule distribution rather than a hard label, the representation also carries the uncertainty in each driver’s behavior instead of collapsing it to a single outcome.

## 8. Conclusion and Future Work

This paper proposed a representation of driver behavior in which each driver is modeled as an evolving density matrix in a nonlinear feature space built from random Fourier features. The representation is continuous, probabilistic, context-dependent, and history-dependent, and it does not commit in advance to which behavioral variables interact or how. Behavioral profiles, the modes within them, and the mixtures that shift between them all emerge from data through unsupervised training.

Applied to the I-24 MOTION dataset, the framework recovered three behavioral profiles corresponding to familiar trafic regimes, free flow, transitional following, and congested stop-and-go, along with the modes that separate steady following from release and braking events inside congestion. It also demonstrated a clear contextual dependence on local density and the variability of surrounding speeds and accelerations. Each profile was found to carry a diferent pattern of interaction among the behavioral variables, learned rather than specified.

The framework also reproduced known macroscopic phenomena. The profiles align with the regimes of the fundamental diagram, and the state’s history dependence explains the hysteresis observed through a kinematic wave. This is more than an additional result. It is the validation principle set out at the start of the paper: a driver behavioral representation earns trust when what it learns aggregates into the trafic phenomena we already understand.

Beyond representation, the framework connects to existing modeling and control practice. For classical car-following models, the trained state supplies the target parameters those models require, demonstrated here by an IDM-based example. For automated vehicles in mixed trafic, the same representation provides a live behavioral read of the surrounding drivers and can be rolled forward to forecast their near-future motion.

## 8.1. Limitations

Several limitations arise from the scope of the study and the modeling choices.

Scope of the data and representation. The framework was evaluated on a single freeway corridor, and the modeled population was restricted to passenger vehicles. In earlier work we applied a smaller version of this representation to the Third Generation Simulation (TGSIM) dataset (Elayan and Kontar, 2026), which includes both freeway and urban intersection trajectories. We chose I-24 MOTION here for its scale and measurement quality: it resolves every vehicle across multiple lanes at high frequency over a complete congestion cycle. This allows the profiles, their internal modes, and their contextual activation to be estimated reliably. For a first treatment of this framework at this scale, that fidelity matters more than environmental variety. It does mean, however, that the profiles recovered here describe freeway driving. The dataset contains no signalized intersections, heavy weaving, or interactions with pedestrians and cyclists, so these specific profiles should not be assumed to transfer to other environments. Testing the framework on data with broader environmental coverage is a necessary step for generalization. The procedure itself, however, is general and can be retrained on new data, and it includes data-driven steps for selecting the behavioral and contextual variables and for choosing the training parameters.

Longitudinal behavior only. The behavioral vector contains speed, spacing, and jerk, all longitudinal quantities, so the representation describes car-following and not lane-changing. Lane changes are a substantial source of behavioral heterogeneity in their own right, and they are also a source the disruption the context variables register. The framework therefore sees the consequences of lane-changing in its context variables while carrying no representation of the maneuver itself. The variable-selection procedure in Appendix A did consider lateral quantities, but they scored poorly against the criteria on this dataset. Including lane-changing would require adding lateral behavioral variables and extending behavior to a mixture of discrete and continuous variables.

Simultaneous behavior and context. Behavior and context were modeled at the same time step. In practice drivers respond to their surroundings after a delay, and that delay varies across drivers and conditions. The contextual variables used here are spatially local and change smoothly relative to the state update, which makes the assumption reasonable for this analysis, but it remains an approximation.

Settings chosen empirically. Several model settings were selected rather than learned. These settings are supported by the ablation study in Appendix B, which shows the chosen values recover distinct, interpretable profiles without redundancy or over-spreading. That evidence, however, is specific to this dataset. The values were selected against I-24 MOTION freeway trafic, and there is no guarantee they are appropriate elsewhere. Applying the framework to a diferent environment would require repeating the selection procedure rather than adopting these values directly.

## 8.2. Future Directions

Three directions for future work follow from this study.

A predictive formulation. The state update was used here to represent behavior, with the forecast in Section 7.2 included as an illustration rather than as a prediction method. A predictive version would model future context jointly with future driver states and add a forecasting term to the training objective, so the state is optimized to anticipate behavior rather than only to describe it. It would also replace the linear extrapolation of future inputs with a learned model of how context evolves.

Delayed response to context. The framework could represent reaction time by separating the moment context is observed from the moment it influences the driver state, allowing the delay itself to be estimated from data and to difer across drivers.

From demonstration to simulation. Section 7.1 demonstrated how the proposed representation can be incorporated into a car-following model. A natural next step is its implementation within a full microscopic trafic simulation framework, in which each vehicle maintains its own dynamic behavioral state. Two key issues remain to be addressed. First, the computational eficiency and numerical stability of the framework must be evaluated under large-scale simulation scenarios. Second, the response parameters, such as maximum acceleration and comfortable deceleration, which were assumed to reflect vehicle capabilities in the present study, may instead require calibration for each behavioral profile. Addressing these challenges would enable a rigorous assessment of whether this framework provides a more realistic representation of trafic dynamics than conventional approaches.

Benchmarking against alternative representations. The comparisons in Section 2 were made on controlled simulations, where the true behavioral structure is known. On I-24 MOTION, the framework is measured only against reference points of its own as illustrated in . A direct comparison with latent-class, hidden-state, and embedding-based representations on the same naturalistic data is missing. It is also not straightforward, since these methods produce diferent objects, a label, a coordinate, or a probability, so likelihood alone is not a fair basis for comparison. Establishing one that captures both predictive fidelity and interpretability is a contribution in itself.

## Acknowledgement

This research is supported by the National Artificial Intelligence Research Resource (NAIRR) Pilot through award NAIRR250132 and the Jetstream2 cloud resource supported by the National Science Foundation (award NSF-OAC 2005506) at Indiana University. The authors also acknowledge the University of Nebraska Holland Computing Center for providing computational resources that contributed to this research.

## Author Contribution Statement

The authors confirm contribution to the paper as follows: study conception and design: Elayan, Armantalab and Kontar; data collection: N/A; data processing: Elayan and Kontar; analysis and interpretations of results: Elayan, Armantalab and Kontar; drafting and editing: Elayan, Armantalab and Kontar. All authors reviewed the results and approved the final version of the manuscript.

## References

Bando, M., Hasebe, K., Nakanishi, K., Nakayama, A., 1998. Analysis of optimal velocity model with explicit delay. Phys. Rev. E 58, 5429–5435. URL: https://link.aps.org/doi/10.1103/ PhysRevE.58.5429, doi:10.1103/PhysRevE.58.5429.

Battifarano, M., Qian, S., 2023. Behavioral inference from non-stationary policies: Theory and application to ridehailing drivers during covid-19 lockdowns. Transportation Research Part C: Emerging Technologies 151, 104118.

van Beinum, A., Farah, H., Wegman, F., Hoogendoorn, S., 2018. Driving behaviour at motorway ramps and weaving segments based on empirical trajectory data. Transportation Research Part C: Emerging Technologies 92, 426–441.

Bouhsissin, S., Sael, N., Benabbou, F., 2023. Driver behavior classification: A systematic literature review. IEEE Access 11, 14128–14153.

Busemeyer, J.R., Bruza, P., et al., 2025. Quantum Models of Cognition and Decision: Principles and Applications. 2nd, Cambridge University Press.

Cresswell, J.C., 2025. Trustworthy ai must account for interactions. URL: https://arxiv.org/ abs/2504.07170, arXiv:2504.07170.

Ding, Z., Xu, D., Tu, C., Zhao, H., Moze, M., Aioun, F., Guillemard, F., 2022. Driver identification through heterogeneity modeling in car-following sequences. IEEE Transactions on Intelligent Transportation Systems 23, 17143–17156.

Edie, L.C., 1965. Discussion of trafic stream measurements and definitions, in: Proceedings of the 2nd International Symposium on the Theory of Trafic Flow, London, England. pp. 139–154.

Elayan, M., Kontar, W., 2026. Behavioral heterogeneity as quantum-inspired representation. URL: https://arxiv.org/abs/2603.22729, arXiv:2603.22729.

Gloudemans, D., Wang, Y., Ji, J., Zachar, G., Barbour, W., Hall, E., Cebelak, M., Smith, L., Work, D.B., 2023. I-24 motion: An instrument for freeway trafic science. Transportation Research Part C: Emerging Technologies 155, 104311.

González, F.A., Gallego, A., Toledo-Cortés, S., Vargas-Calderón, V., 2022. Learning with density matrices and random features. Quantum Machine Intelligence 4. URL: https://doi.org/10. 1007/s42484-022-00079-9, doi:10.1007/s42484-022-00079-9.

Hamdar, S.H., Dixit, V.V., Talebpour, A., Treiber, M., 2019. A behavioral microeconomic foundation for car-following models. Transportation Research Procedia 38, 565–585.

Ji, A., Ramezani, M., Levinson, D., 2023. Joint modelling of longitudinal and lateral dynamics in lane-changing maneuvers. Transportmetrica B: Transport Dynamics 11, 996–1025.

Khrennikov, A., Asano, M., 2020. A quantum-like model of information processing in the brain. Applied Sciences 10, 707. URL: https://www.mdpi.com/2076-3417/10/2/707, doi:10.3390/ APP10020707.

Kim, I., Kim, T., Sohn, K., 2013. Identifying driver heterogeneity in car-following based on a random coeficient model. Transportation research part C: emerging technologies 36, 35–44.

Kontar, W., Kim, Y., Zhong, X., Ahn, S., 2026. Capturing behavioral heterogeneity for trafic flow: A scalable and personalized decomposition approach. Transportation Research Part B: Methodological 209, 103467.

Krauss, S., Wagner, P., Gawron, C., 1997. Metastable states in a microscopic model of trafic flow. Phys. Rev. E 55, 5597–5602. URL: https://link.aps.org/doi/10.1103/PhysRevE.55.5597, doi:10.1103/PhysRevE.55.5597.

Laval, J.A., 2011. Hysteresis in trafic flow revisited: An improved measurement method. Transportation Research Part B: Methodological 45, 385–391. URL: https://www.sciencedirect. com/science/article/pii/S0191261510001013, doi:https://doi.org/10.1016/j.trb.2010. 07.006.

Lee, J.D., 2008. Fifty years of driving safety research. Human factors 50, 521–528.

Newell, G., 2002. A simplified car-following theory: a lower order model. Transportation Research Part B: Methodological 36, 195–205. URL: https://www.sciencedirect.com/science/ article/pii/S0191261500000448, doi:https://doi.org/10.1016/S0191-2615(00)00044-8.

Nirmale, S.K., Pinjari, A.R., Chakroborty, P., 2024. A two-dimensional, multi-vehicle anticipation, and multi-stimuli based latent class framework to model driver behaviour in heterogeneous, disorderly trafic conditions. Transportation research part C: emerging technologies 160, 104458.

Nirmale, S.K., Pinjari, A.R., Sharma, A., 2021. A discrete-continuous multi-vehicle anticipation model of driving behaviour in heterogeneous disordered trafic conditions. Transportation Research Part C: Emerging Technologies 128, 103144.

Oppenheim, I., Shinar, D., 2012. A context-sensitive model of driving behaviour and its implications for in-vehicle safety systems. Cognition, Technology & Work 14, 261–281.

Ossen, S., Hoogendoorn, S.P., 2011. Heterogeneity in car-following behavior: Theory and empirics. Transportation research part C: emerging technologies 19, 182–195.

Sun, L., Wu, Z., Ma, H., Tomizuka, M., 2020. Expressing diverse human driving behavior with probabilistic rewards and online inference, in: 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), IEEE.

Tavakoli, A., Boker, S., Heydarian, A., 2022. Driver state modeling through latent variable state space framework in the wild. IEEE Transactions on Intelligent Transportation Systems 24, 1879– 1893.

Treiber, M., Hennecke, A., Helbing, D., 2000. Congested trafic states in empirical observations and microscopic simulations. Phys. Rev. E 62, 1805–1824. URL: https://link.aps.org/doi/10. 1103/PhysRevE.62.1805, doi:10.1103/PhysRevE.62.1805.

Wagner, P., 2012. Analyzing fluctuations in car-following. Transportation research part B: methodological 46, 1384–1392.

Wang, Q., Zhang, W., Song, Y., Xiao, W., 2025. Modeling and analysis of car-following behavior based on macro–micro coupling. IEEE Transactions on Intelligent Transportation Systems 27, 938–949.

Wang, Z., Lu, G., Tan, H., 2024. Driving behavior model for multi-vehicle interaction at uncontrolled intersections based on risk field considering drivers’ visual field characteristics. IEEE Transactions on Intelligent Transportation Systems 25, 15532–15546.

Xu, D., Zhao, H., Guillemard, F., Geronimi, S., Aioun, F., 2018. Aware of scene vehicles—probabilistic modeling of car-following behaviors in real-world trafic. IEEE Transactions on Intelligent Transportation Systems 20, 2136–2148.

Yao, X., Calvert, S.C., Hoogendoorn, S.P., 2025a. Driving heterogeneity identification using machine learning: A review and framework for analysis. Transportation Research Interdisciplinary Perspectives 32, 101511. doi:10.1016/j.trip.2025.101511.

Yao, X., Calvert, S.C., Hoogendoorn, S.P., 2025b. A novel framework for identifying driving heterogeneity through action patterns. IEEE Transactions on Intelligent Transportation Systems

Zhang, C., He, Z., Wu, C., Sun, L., 2025. When context is not enough: Modeling unexplained variability in car-following behavior. arXiv preprint arXiv:2507.07012 .

Zhong, X., Zhou, Y., Ahn, S., Chen, D., 2024. Understanding heterogeneity of automated vehicles and its trafic-level impact: A stochastic behavioral perspective. Transportation Research Part C: Emerging Technologies 164, 104667.

## Appendix A. Discovering the State and Context Variables

Selecting model inputs by hand makes any recovered structure partly a product of that choice. To avoid this, we select the state and context variables through a transparent, data-driven procedure, summarized here.

## A.1. Procedure

We assembled twenty candidate variables from the driver-behavior literature: twelve describing ego-vehicle motion and interactions (behavioral) and eight describing surrounding trafic (context). Each variable was evaluated using four criteria, scored on a [0, 1] scale. To ensure comparable driving conditions, observations were grouped into four equal-sized regimes based on downstream trafic speed.

Within-context spread (WCS). Measures variation across drivers within a regime, normalized by total variation. Higher values indicate greater heterogeneity.

Between-context shift (BCS). Measures variation in regime-level means, normalized by total variation. Higher values indicate stronger context sensitivity.

Temporal evolution (TE). Combines short- and long-lag autocorrelation within trajectories. Higher values indicate variables that are locally smooth while evolving over the trip.

Cross-driver consistency (CDC). Measures variation attributable to driver identity. Higher values indicate stable, distinguishable driver-specific patterns.

The four criterion scores were aggregated using two composite measures. The arithmetic mean (A) computes the average of the four scores, whereas the geometric mean (G) computes the fourth root of their product. Because the geometric mean is strongly reduced by any low score, it favors variables that perform consistently well across all criteria.

Redundant variables were then removed: if two candidates had a pairwise correlation greater than 0.85, only one was retained. Missing-data rates (NaN) are reported alongside the final scores.

## A.2. Candidate variables

Table A1 summarizes the candidate variables considered in this study

## A.3. Results

We applied the procedure to the congested westbound subset of I-24 MOTION, scoring 10,000 trajectories against a neighborhood built from the full trafic stream (Table A2). Three patterns emerge. Speed-related variables (B1, C3, C4) score highest but are mutually correlated $( | r | \approx 0 . 8 5 -$ 0.96), reflecting that in congestion a vehicle’s speed and its surroundings’ speed nearly coincide. Temporal evolution is low for most variables, the exceptions being jerk (B3) and the entropy measures (C6, C8). Omnidirectional density (C1) ranks among the strongest context variables once measured over a realistic 150 m perception radius.

Table A1: Candidate variables.
<table><tr><td>Code</td><td>Variable</td><td>Definition</td></tr><tr><td colspan="3">Behavioral variables</td></tr><tr><td>B1</td><td>Speed</td><td>Longitudinal speed</td></tr><tr><td>B2</td><td>Acceleration</td><td>Longitudinal acceleration</td></tr><tr><td>B3</td><td>Jerk</td><td>Rate of change of acceleration</td></tr><tr><td>B4</td><td>Lateral velocity</td><td>Lateral velocity</td></tr><tr><td>B5</td><td>Lateral acceleration</td><td>Lateral acceleration</td></tr><tr><td>B6</td><td>Relative speed</td><td>Ego minus leader speed</td></tr><tr><td>B7</td><td>Spacing</td><td>Gap to leader net of ego length</td></tr><tr><td>B8</td><td>Time-to-collision</td><td>Spacing divided by closing speed</td></tr><tr><td>B9</td><td>Time headway</td><td>Spacing divided by ego speed</td></tr><tr><td>B10</td><td>Acceleration std.</td><td>Acceleration standard deviation over 1 s</td></tr><tr><td>B11</td><td>Speed std.</td><td>Speed standard deviation over 5s</td></tr><tr><td>B12</td><td>Lateral std.</td><td>Lateral position standard deviation over 1 s</td></tr><tr><td colspan="3">Context variables</td></tr><tr><td>C1</td><td>Omni density</td><td>Vehicles within 150 m</td></tr><tr><td>C2</td><td>Forward density</td><td>Vehicles in forward zone</td></tr><tr><td>C3</td><td>Forward mean speed</td><td>Mean speed of forward zone</td></tr><tr><td>C4</td><td>Forward min speed</td><td>Minimum speed of forward zone</td></tr><tr><td>C5</td><td>Speed gap</td><td>Forward mean speed minus ego speed</td></tr><tr><td>C6</td><td>Speed entropy</td><td>Entropy of neighbors&#x27; speed changes</td></tr><tr><td>C7</td><td>Lateral entropy</td><td>Entropy of neighbors&#x27; lateral position changes</td></tr><tr><td>C8</td><td>Acceleration entropy</td><td>Entropy of neighbors&#x27; accelerations</td></tr></table>

Table A2: Variable ranking results based on the four evaluation criteria.
<table><tr><td>Code</td><td>Type</td><td>WCS</td><td>BCS</td><td>TE</td><td>CDC</td><td>A</td><td>G</td><td>NaN</td></tr><tr><td>B1</td><td>beh.</td><td>0.53</td><td>0.98</td><td>0.07</td><td>0.86</td><td>0.61</td><td>0.42</td><td>0.00</td></tr><tr><td>C1</td><td>ctx.</td><td>0.80</td><td>0.69</td><td>0.19</td><td>0.77</td><td>0.61</td><td>0.53</td><td>0.00</td></tr><tr><td>C4</td><td>ctx.</td><td>0.36</td><td>1.07</td><td>0.12</td><td>0.84</td><td>0.60</td><td>0.44</td><td>0.04</td></tr><tr><td>C3</td><td>ctx.</td><td>0.29</td><td>1.10</td><td>0.11</td><td>0.87</td><td>0.59</td><td>0.42</td><td>0.04</td></tr><tr><td>C2</td><td>ctx.</td><td>0.87</td><td>0.56</td><td>0.16</td><td>0.70</td><td>0.57</td><td>0.48</td><td>0.00</td></tr><tr><td>C8</td><td>ctx.</td><td>0.96</td><td>0.30</td><td>0.45</td><td>0.44</td><td>0.54</td><td>0.49</td><td>0.04</td></tr><tr><td>B7</td><td>beh.</td><td>0.98</td><td>0.26</td><td>0.11</td><td>0.58</td><td>0.48</td><td>0.36</td><td>0.16</td></tr><tr><td>C6</td><td>ctx.</td><td>0.98</td><td>0.19</td><td>0.52</td><td>0.25</td><td>0.48</td><td>0.39</td><td>0.00</td></tr><tr><td>B3</td><td>beh.</td><td>1.00</td><td>0.03</td><td>0.85</td><td>0.00</td><td>0.47</td><td>0.07</td><td>0.00</td></tr><tr><td>C7</td><td>ctx.</td><td>0.98</td><td>0.25</td><td>0.41</td><td>0.19</td><td>0.46</td><td>0.37</td><td>0.00</td></tr><tr><td>C5</td><td>ctx.</td><td>0.99</td><td>0.10</td><td>0.13</td><td>0.55</td><td>0.44</td><td>0.29</td><td>0.04</td></tr><tr><td>B6</td><td>beh.</td><td>0.99</td><td>0.12</td><td>0.17</td><td>0.45</td><td>0.43</td><td>0.31</td><td>0.16</td></tr><tr><td>B12</td><td>beh.</td><td>0.98</td><td>0.16</td><td>0.27</td><td>0.29</td><td>0.43</td><td>0.33</td><td>0.00</td></tr><tr><td>B11</td><td>beh.</td><td>0.99</td><td>0.12</td><td>0.15</td><td>0.33</td><td>0.40</td><td>0.27</td><td>0.00</td></tr><tr><td>B10</td><td>beh.</td><td>1.00</td><td>0.04</td><td>0.27</td><td>0.28</td><td>0.39</td><td>0.23</td><td>0.00</td></tr><tr><td>B8</td><td>beh.</td><td>0.99</td><td>0.14</td><td>0.14</td><td>0.30</td><td>0.39</td><td>0.28</td><td>0.16</td></tr><tr><td>B2</td><td>beh.</td><td>0.99</td><td>0.15</td><td>0.12</td><td>0.19</td><td>0.36</td><td>0.24</td><td>0.00</td></tr><tr><td>B4</td><td>beh.</td><td>0.99</td><td>0.01</td><td>0.22</td><td>0.22</td><td>0.36</td><td>0.14</td><td>0.00</td></tr><tr><td>B9</td><td>beh.</td><td>0.69</td><td>0.23</td><td>0.13</td><td>0.28</td><td>0.33</td><td>0.28</td><td>0.18</td></tr><tr><td>B5</td><td>beh.</td><td>0.99</td><td>0.02</td><td>0.16</td><td>0.11</td><td>0.32</td><td>0.14</td><td>0.00</td></tr></table>

Table A3: Selected variables.
<table><tr><td>Code Name</td><td></td><td>Rationale</td></tr><tr><td colspan="3">Behavioral</td></tr><tr><td>B1</td><td>Speed</td><td>Highest-scoring behavioral variable; strong context- sensitivity and driver consistency; no missing data.</td></tr><tr><td>B7</td><td>Spacing</td><td>Principal car-following measure; strong driver-specific signal.</td></tr><tr><td>B3</td><td>Jerk</td><td>The only strongly time-evolving behavioral variable; re- tained so the state is not static.</td></tr><tr><td colspan="3">Context</td></tr><tr><td>C1</td><td>Omni density</td><td>Strongest context variable; balanced across criteria; no missing data.</td></tr><tr><td>C6</td><td>Speed entropy</td><td>Most time-evolving context variable; captures disorder in surrounding speeds.</td></tr><tr><td>C8</td><td>Accel. entropy</td><td>Time-evolving and distinct from C6; captures disorder in surrounding accelerations.</td></tr></table>

## A.4. Selection

We retain three behavioral and three context variables Table A3), selected to capture distinct behavioral dimensions rather than simply maximize individual scores. To avoid redundancy, the forward-speed variables (C3, C4) were excluded because they largely duplicate the information contained in ego speed. Among the retained variables the strongest pairwise correlation is 0.610 (ego speed and surrounding density), and the two entropy measures are only weakly correlated $( | r ( \mathrm { C 6 } , \mathrm { C 8 } ) | = 0 . 2 5 )$ .

The behavioral variables represent motion (B1), spacing (B7), and temporal dynamics (B3), while the context variables represent trafic density (C1) and surrounding trafic disorder (C6, C8). Temporal evolution was generally weak across the candidate pool except for jerk and the entropy measures, suggesting that driver behavior under congestion is largely stationary over time.

## Appendix B. Ablation Studies

## B.1. Selecting the Number of Profiles (K)

We estimate the framework for $K \in \{ 2 , 3 , 4 , 5 \}$ under identical training conditions and select the value that balances model fit and behavioral interpretability. The results are summarized in Figure B1.

The largest improvement in fit occurs when moving from $K = 2$ to $K = 3$ , with the perobservation NLL decreasing from 0.90 to 0.88. No further improvement is obtained for $K = 4$ or $K = 5$

The profile structure indicates that three profiles are suficient. At $K = 2 .$ , transitional and congested trafic behavioral patters are conflated into a single regime. At $K = 3 .$ , a third profile emerges between free-flow and congested trafic, capturing meaningful transition between the two. Increasing K further does not uncover additional regimes. At $K = 4 .$ the extra profile is largely a duplicate of an existing transitional following profile, while at $K = 5$ an additional profile appears difuse and fails to show a clear behavioral identity as indicated by its eigenvalue spread.

The context coeficients support the same conclusion. At $K = 3$ , the profiles are activated by distinct contextual conditions, whereas the additional profiles at $K = 4$ and $K = 5$ largely mirror existing activation patterns. These results suggest that $K \ : = \ : 3$ captures the major behavioral regimes in the data without introducing redundant profiles, and we therefore use $K = 3$ in the remainder of the paper.

![](images/ed31ef5c349de8436bce4a81176d5725d7efc6fb6ed33d785454ae988c8ba1e9.jpg)

![](images/9e5a9cd0e2049da4a666ba1cb399bdc957c2f10563b76aa33c05ebf8987cfa55.jpg)

![](images/a7072617a8d83905c6612a8cadf8e1f04a788928a801769fc056a18991c27dfc.jpg)

![](images/cecab2f851ad987c4c5b40b43b151a550132d5a7d7eecb9bdc33de7deef79784.jpg)

![](images/2e56575715ba2008bf0680ed99567e5c4c7950c2f329b1400e5300a1d2fb251a.jpg)

![](images/3fe1d61843f4d932bcd0301fe4c32c477ad562cd343f2d5427497919a5959ac9.jpg)

![](images/bebd2e72a9a14209580a13e019d09d5d47dfcf345dcaed1109c4bf0e72120f78.jpg)

![](images/ba1faf719be00f7890ab0bafcacf164076c900939d5a8eaf14555f25e58d1a83.jpg)

![](images/98985e117ce844e23b83792fcec92e7cbf4f9592fe0c23df6c7d94c30e947121.jpg)

![](images/3490a6d76ad4e3e3df759e593431a3974bc84952cb62807e57cf06dfcf17cce3.jpg)

![](images/c853e2301d9c3e84635aad38be6fcddab8372867640d4449e70cd645da2b844f.jpg)

![](images/04e83043f22486e20c3355dd5168c2d6f766d014ac8e9f66f74544acb8a4d129.jpg)  
Figure B1: Selection of the number of behavioral profiles K. Columns correspond to models estimated with $K \in$ $2 , 3 , 4 , 5$ under identical training conditions. The top row shows profile eigenvalue composition, the middle row shows profile dominant mode concentration in speed–spacing space, and the bottom row shows context-activation coeficients. The per-observation NLL is reported above each column. Profile colors are not comparable across columns.

## B.2. Selecting the Regularization Weight (γ)

We estimate the framework for $\gamma \in \{ 0 , 0 . 5 , 1 , 4 , 8 \}$ under identical training conditions and select the value that lets a profile retain genuine multi-modal structure without spreading its weight so thinly that the profile loses coherence. The results are summarized in Figure B2.

![](images/ca04ef333860292b04298d6d2052596925604740a7c0f96b3be853283b7acdba.jpg)  
Figure B2: Selection of the entropy regularization weight $\gamma .$ Columns correspond to models estimated with $\gamma \in$ $\{ 0 , 0 . 5 , 1 , 4 , 8 \}$ under identical training conditions. The top row shows the profile eigenvalue spectrum $\lambda _ { k } ,$ the middle row shows the profile dominant-mode locations in speed–spacing space, and the bottom row shows context-activation coeficients. The per-observation NLL is reported above each column. Profile colors are not comparable across columns.

Fit alone does not decide the choice. The per-observation NLL is essentially flat across the range, so no value of $\gamma$ (within the examined range) is preferred on likelihood grounds. What $\gamma$ controls here is the eigenvalue structure of the profiles, and that is where the choice is made.

At $\gamma \in \{ 0 , 0 . 5 , 1 \}$ the penalty is too weak: all three profiles collapse to near rank one, with dominant eigenvalues above 0.995. At $\gamma = 4$ , the congested trafic profile develops a spread spectrum $( \lambda _ { 1 } ~ = ~ 0 . 8 5 2$ , with the next four eigenvalues carrying a further 14% of its weight). This is the structure that supports the mode decomposition reported in Section 5.3. At $\gamma = 8$ the penalty becomes too strong as the multi-modal profile’s leading eigenvalue falls to 0.663. The profile weight is smeared across many modes and no longer describes a coherent regime.

The context coeficients tell a complementary story. They are stable across the range, with the profiles activated by the same contextual conditions regardless of $\gamma$ . Regularization therefore afects the mode structure within a profile, not which contexts activate it. We choose $\gamma = 4$ because it is the only value that recovers multi-modal structure where the data support it while keeping each profile interpretable, and we use it for the remainder of the paper.

## B.3. Selecting the Behavioral Variables

We estimate the framework with three behavioral sets: $( v , \Delta s ) , ( v , j )$ , and the full set $( v , \Delta s , j )$ All other settings match the reference model. The results are summarized in Figure B3.

The per-observation NLL values are not directly comparable across ablations because each model represents a distribution over a diferent number of behavioral variables, and the NLL scales with that dimensionality. What is comparable is the profile structure the model recovers. With $( v , \Delta s )$ three profiles are recovered but the congested trafic profile loses most of its eigenvalue spread. With $( v , j )$ , the profiles are recovered in speed and jerk but spacing is no longer available for the framework to organize free flow, transition, and congestion in the speed–spacing plane where these regimes are traditionally identified.

![](images/d891faa9af693d6b82d94fb8587cbac775fce66fca8dafa047737e41e5e30515.jpg)  
Figure B3: Selection of the behavioral variable set. Columns correspond to models estimated with behavioral sets $( v , \Delta s ) , ( v , j )$ , and $( v , \Delta s , j )$ . The top row shows the profile eigenvalue spectrum, the middle row shows the profile dominant-mode locations using the two variables in that ablation, and the bottom row shows context-activation coeficients. The per-observation NLL is reported above each column and is not comparable across columns because each model represents a diferent number of behavioral variables. Profile colors are not comparable across columns.

The context coeficients also shift when a behavioral variable is dropped. Under $( v , j )$ the beta signs and magnitudes change substantially for one profile, indicating that the model is compensating for the loss of headway information by reassigning contextual signals across profiles.

Only the full behavioral set $( v , \Delta s , j )$ recovers three profiles with distinct signatures in the speed–headway plane and a spread eigenspectrum for the congested trafic profile. We therefore use the full behavioral set in the remainder of the paper.

## B.4. Selecting the Context Variables

We estimate the framework with three context sets: density alone (d), density and speed entropy $( d , H _ { v } )$ , and the full set of density, speed entropy, and acceleration entropy $( d , H _ { v } , H _ { a } )$ . The results are summarized in Figure B4.

Model fit improves as more context variables are included. The per-observation NLL is 0.91 with density alone, 0.91 with density and speed entropy, and 0.88 with the full set. The improvement

![](images/7fa173285eecd05ac7e47dae47238bd1a1e7d84cbef8b2ca4cda50c74c6ccbb7.jpg)  
Figure B4: Selection of the context variable set. Columns correspond to models estimated with density alone, density and speed entropy, and the full set of density, speed entropy, and acceleration entropy. The top row shows the profile eigenvalue spectrum, the middle row shows the profile dominant-mode locations in speed–spacing space, and the bottom row shows context-activation coeficients. The best per-observation NLL is reported above each column. Profile colors are not comparable across columns.

from adding acceleration entropy is the largest and consistent with its role in distinguishing free flow from transitional following behavior under common density.

In terms of profile structure, all three sets produce one multi-modal profile, but they disagree on which one. Under (d) and $( d , H _ { v } )$ , the spread appears in the free-flow profile rather than the congested profile.

The context coeficients also become more informative as more variables are included, with each profile responding to a diferent combination of contextual signals rather than being distinguished by density alone. We therefore use the full context set in the remainder of the paper.

## B.5. Selecting the Feature Dimension (D)

We estimate the framework at three values of the random Fourier feature dimension, $D \in$ 100, 200, 300 . The per-frame cost of the measurement update and the observation update scales as $O ( D ^ { 2 } )$ , so doubling D quadruples the per-frame work. Applied to the training scale used for the main results in this paper, $D = 2 0 0$ would require approximately 4 the wall time of $D = 1 0 0$ , and $D = 3 0 0$ approximately 9 .

As shown in Figure B5, the per-observation NLL is 0.88 at $D = 1 0 0 , 0 . 8 7$ at $D = 2 0 0$ , and 0.91 at $D = 3 0 0$ . The minor improvement at $D = 2 0 0$ over $D = 1 0 0$ is achieved at four times the cost. The profile structure is stable across the three values: each configuration recovers three profiles with distinct signatures in the speed–headway plane, comparable eigenvalue spread, and consistent context activation. We therefore use $D = 1 0 0$ in the remainder of the paper, where the fit reaches the same qualitative level as the higher-cost configurations at a fraction of the compute budget.

![](images/62fef0ce4b232a89bf3f1c5c15a12996be61a8ba4ea4304867d327eb1299c104.jpg)

![](images/58f12b63b87d383eed296d20dbd82a75c35f37ce494f1dc4145558f35a15c57a.jpg)

![](images/135747699081c7452787777793a06bb353521d107708ec54cf94c07455a4fceb.jpg)

![](images/b89d214e777c08ea94166431ba1e797537c65c94dba3fe264ba15ce703a86fc3.jpg)

![](images/6ddab2e3ab66a426797a96759d2c8eeef31a7c4acd1bf8ef5c7ede58cd671ffa.jpg)

![](images/a6ba1c0a0fecebd05a30808d61304071a8a71a371a6cc4ca695dd6221d70a7a6.jpg)

![](images/d51e12d6b5a38a92b67280d8575f3663dcab1bad263ddc58a5a5b6fdbcc30261.jpg)

![](images/f699880f0f40d3bba27f3040c5282ddb44dcf36c1c174e37066ec24daa3f5e41.jpg)

![](images/df4e5d8ac291d91bea75e06dd1745581a17e0e3bafa903d00971803f62d615b3.jpg)  
Figure B5: Selection of the feature dimension D. Columns correspond to models estimated with $D \in \{ 1 0 0 , 2 0 0 , 3 0 0 \}$ The top row shows the profile eigenvalue spectrum $\lambda _ { k } ,$ , the middle row shows the profile dominant-mode locations in speed–spacing space, and the bottom row shows context-activation coeficients. The per-observation NLL is reported above each column. Projected to the training scale used for the main results in this paper (100,000 egos, 5 epochs), the wall-clock cost is approximately $1 \times , 4 \times$ , and $9 \times$ for $D = 1 0 0 , 2 0 0 , 3 0 0$ respectively. Profile colors are not comparable across columns.