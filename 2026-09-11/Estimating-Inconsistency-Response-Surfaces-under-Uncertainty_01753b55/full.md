# Estimating Inconsistency Response Surfaces under Uncertainty in Cyber-Physical System Development

Johannes Mäkelburg ①   
Technische Universität München Munich, Germany   
johannes.maekelburg@tum.de

Tim Schwabe Technische Universität München Munich, Germany tim.schwabe@tum.de

Maribel Acosta Technische Universität München Munich, Germany maribel.acosta@tum.de

Abstract—Cyber-Physical Systems (CPS) are commonly represented through multiple interconnected models. During development, CPS consistency requires that shared model elements remain compatible across these models. Uncertainty, for example, due to sensor noise or model abstraction, changes the admissible values of model elements and can introduce inconsistencies, i.e., situations in which models can no longer be jointly satisfied. While existing approaches can determine consistency for a given uncertainty configuration, they provide limited support for systematically exploring, analyzing, and explaining inconsistency across large uncertainty spaces. We address this challenge by reformulating inconsistency as an intervention response modeling problem. Using Saltelli sampling and multi-fidelity Monte Carlo estimation, we generate intervention-response datasets and train a surrogate model that directly predicts inconsistency from the propagated uncertainty geometry. Experiments on 48 scenarios and 10 CPS domains show that the surrogate matches Monte Carlo estimates while reducing evaluation time from milliseconds to microseconds, enabling orders-of-magnitude more responsesurface evaluations within fixed computational budgets. Building on the learned response surfaces, we perform sensitivity analysis to identify dominant uncertainty drivers and introduce a gradient-based consistency recourse method to determine minimal uncertainty interventions that restore consistency. The results show that inconsistency under uncertainty can be effectively learned, analyzed, and repaired through response-surface modeling, providing a scalable foundation for uncertainty-aware consistency management in CPS development.

Index Terms—Cyber-physical systems, Consistency analysis, Uncertainty quantification, Surrogate model, Algorithmic recourse.

## I. INTRODUCTION

Modern Cyber-Physical Systems (CPS) are designed through interacting models describing structural, behavioral, and physical properties of the system [1], [2]. Consistency requires that shared model elements remain compatible across interacting models, and is crucial for reliable system design and operation [3]. At design time, each model is treated as an independent artifact whose admissible state ranges are checked for compatibility with those of interacting models. However, uncertainty due to parameter variation, sensor noise, and abstraction gaps propagates across interacting models [4], [5], potentially inducing inconsistencies at design time that are not observable at nominal operating conditions.

![](images/280444169382931ae5164be358037bac2a842f77bc8f8adeaab7b62c75a16afc.jpg)  
Fig. 1. Monte Carlo and surrogate inconsistency response surfaces for the Medical Device – Insulin Dose & Ventilator Pressure scenario. The surrogate preserves the global response geometry and consistency boundary.

As an example, consider a development-time medical system for insulin therapy consisting of two interacting (acyclic) models: a metabolic model used to estimate the patients glucose regulation and a treatment model used to determine appropriate insulin dosages. Given the current calibrations, both models are consistent as their assumed physiological state ranges overlap. However, after updating the metabolic model based on new clinical evidence, the feasible state estimates shift systematically, and the feasible state range of the metabolic model shifts and no longer overlaps with the range assumed by the treatment model. Yet the two models, each individually correct, cannot be jointly satisfied. Such inconsistencies can propagate across engineering artifacts and can ultimately lead to critical system failures [6], [7].

When uncertainty is modeled explicitly, consistency becomes a property over sets of admissible model configurations rather than single deterministic values [8]. Reasoning about this property requires propagating uncertainty through interacting models and evaluating large spaces of feasible system states [9]. This results in computationally expensive stochastic response estimation problems similar to those studied in uncertainty quantification, simulation analytics, and surrogateassisted scientific computing [10]-[12]. Recent complexity results further show that exact containment and overlap reasoning for set representations is W[1]-hard with respect to its dimension motivating learned approximation and surrogatebased analysis approaches [13]. While existing approaches can determine whether a particular configuration is consistent, they provide limited support for systematically learning and analyzing the global inconsistency behavior induced by uncertainty.

To address these challenges, we reformulate inconsistency analysis under uncertainty as an intervention-response modeling problem, where interventions (like shifting and scaling) parameterize modifications to the underlying uncertainty. Given an intervention configuration θ, the resulting inconsistency score $I ( \theta )$ defines a response surface (representing the probability of inconsistency) over the induced uncertainty space. This formulation enables systematic generation of intervention-response datasets for scalable response-function estimation, surrogate modeling, and intervention-based analysis [12]. To support scalable exploration of the resulting inconsistency landscapes, we combine multi-fidelity Monte Carlo estimation with a learned surrogate model that directly predicts inconsistency from post-intervention uncertainty geometry. Figure 1 shows the inconsistency landscape for the insulin dose scenario with respect to shift and scaling of the uncertainty region. The learned surrogate models landscape agrees quantitatively and qualitatively with the one estimated via Monte-Carlo sampling. The surrogate enables efficient approximation of computationally expensive consistency evaluations while preserving geometric dependencies induced by uncertainty propagation across interacting models [14]. Additionally, it enables large-scale sensitivity analysis and consistency recourse to be tractable. The learned response surfaces reveal nonlinear interaction effects, dominant uncertainty drivers, and intervention regions that restore cross-model consistency.

Building on this formulation, this paper makes the following contributions: (i) We formalize inconsistency under uncertainty as an intervention-response function $I ( \theta )$ over interacting CPS models. (ii) We evaluate structured surrogate architectures for learning the inconsistency response surface $I ( \theta )$ directly from post-intervention set geometry. (iii) We analyze the structural drivers of inconsistency through sensitivity analysis and intervention-response landscapes. (iv) We address consistency recourse through population- and gradient-based minimal uncertainty interventions.

## II. PRELIMINARIES

In CPS development, we represent uncertain states as sets of admissible values and analyze how uncertainty propagates across interacting components.

a) Set-Based Uncertainty Representation: We represent an uncertain model using zonotopes, which compactly encode sets of admissible states via a center point and a set of generator vectors spanning the uncertainty region. In particular we use constrained zonotopes [8], which allow representing bounded uncertainties and linear dependencies, while supporting efficient algebraic operations such as affine transformations [15], [16]

Definition 1 (Constrained Zonotope $I 8 J ) .$ Given a center $c \in \mathbb { R } ^ { n }$ , a generator matrix $G = [ g ^ { ( 1 ) } , \cdot \cdot \cdot , g ^ { ( \gamma ) } ] \in \mathbb { R } ^ { n \times \gamma }$ with $\gamma \in \mathbb { N }$ generator vectors, and constraints given by $A \in \mathbb { R } ^ { n \times \gamma }$ $b \in \mathbb { R } ^ { n }$ , a constrained zonotope $\mathcal { Z }$ is defined as

$$
\mathcal { Z } : = \Big \{ x \in \mathbb { R } ^ { n } \Big | x = c + \sum _ { i = 1 } ^ { \gamma } \xi ^ { ( i ) } g ^ { ( i ) } , A \xi = b , \xi \in [ - 1 , 1 ] ^ { \gamma } \Big \}
$$

Diagonal generators encode interval uncertainty [8], [17], while covariance-derived generators approximate probabilistic confidence regions [18].

b) Consistency under Uncertainty: Given a collection of uncertain models, each represented by a constrained zonotope, consistency requires that their uncertainty sets admit at least one jointly feasible realization.

Definition 2: Let $\mathcal { Z } = \{ \mathcal { Z } _ { 1 } , \ldots , \mathcal { Z } _ { n } \}$ denote a collection of interacting uncertainty sets. The uncertainty sets are jointly consistent iff there exists at least one jointly feasible realization: $\textstyle \bigcap _ { i = 1 } ^ { n } { \mathcal { Z } } _ { i } \neq \emptyset$

Thus, consistency under uncertainty holds when there exists at least one realization that is admissible across all uncertainty sets simultaneously.

c) Uncertainty Mappings: Dependencies between uncertainty sets are captured through uncertainty mappings (UMs), which define directed relations between source and target models. We assume that these mappings induce a directed acyclic dependency graph. As in the insulin therapy example, the physiological state ranges assumed by the metabolic and treatment models must overlap.

Definition 3: A UM is an affine function $\varphi : X _ { i }  X _ { j } .$ typically of affine form $\varphi ( x ) \ = \ F x + f ;$ that relates the admissible states of a source model to those of a target model. The mapped uncertainty set is defined as $\Phi ( \mathcal { Z } _ { j } ) = \varphi ( \mathcal { Z } _ { i } ) \cap \mathcal { Z } _ { j }$ The intersection checks whether the mapped source states are compatible with the locally admissible uncertainty regions.

Definition 4: The global feasible region is defined as the intersection of all mapped sets: $\mathcal { Z } _ { \mathrm { s y s } } = \bigcap _ { i } \Phi ( \mathcal { Z } _ { i } )$

## III. METHODOLOGY

This section reformulates uncertainty-aware consistency analysis as a response-surface modeling problem over intervention-induced uncertainty configurations. Intervention configurations θ define the intervention space, induced inconsistency I(θ) defines the response surface.

## A. CPS Representation and Inconsistency Measure

We consider a CPS as a collection of interacting models $\{ M _ { 1 } , \dots , M _ { m } \}$ , each representing a distinct aspect of the system, such as physical behavior, control logic, or sensor characteristics. Following the set-based uncertainty representation introduced in Definition 1, each model $M _ { l }$ is represented by a constrained zonotope $\mathcal { Z } _ { M _ { l } } \subseteq \mathbb { R } ^ { n }$ representing the set of admissible states of the model under uncertainty. The dimension n represents the number of independent uncertainties affecting $M _ { l }$ . Dependencies between models are captured through UMs (Definition 3), and their intersection determines the global consistency region $\mathcal { Z } _ { \mathrm { s y s } }$ (Definition 4). Figure 2 provides an overview of the consistency evaluation pipeline.

![](images/0ccb4723b49e77bbee78407badc505e2b7561526c90d88b569f17ad014ae5f8f.jpg)  
Fig. 2. Consistency reasoning pipeline: uncertainty sets are mapped and intersected into a global feasibility region, sampled to produce binary consistency scores, and aggregated into the inconsistency response $I ( \theta )$

The consistency of a CPS is evaluated in two steps: First, model-level uncertainty sets are propagated through uncertainty mappings to determine the global feasible region $\mathcal { Z } _ { \mathrm { s y s } } .$ This region contains all states that remain jointly admissible across the interacting model after accounting for all dependencies induced by the uncertainty mappings.

Given a sampled realization $\xi$ the resulting state $x ( \xi )$ is evaluated against the global feasible region $\mathcal { Z } _ { \mathrm { s y s } }$ to determine whether the system is consistent. The binary consistency score is defined as:

$$
C ( \xi ) = \left\{ { \begin{array} { l l } { 1 , } & { x ( \xi ) \in { \mathcal { Z } } _ { \mathrm { s y s } } } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } } \end{array} } \right.\tag{1}
$$

Because $C ( \xi )$ depends on stochastic realizations $\xi ,$ we aggregate over $\xi$ and directly define the inconsistency measure function

$$
I = 1 - \mathbb { E } _ { \xi \sim p ( \xi ) } \big [ C ( \xi ) \big ] .\tag{2}
$$

The function I represents the probability that the system becomes inconsistent. Evaluating I exactly requires reasoning over the full feasibility region $\mathcal { Z } _ { \mathrm { s y s } }$ , which is intractable for general CPS configurations due to the computational hardness of exact zonotope containment [13]. We therefore estimate it numerically.

## B. Estimating the Inconsistency Response

We estimate I using three estimators with different accuracy-runtime tradeoffs: Monte Carlo sampling (MC), a geometric proxy based on axis-aligned bounding boxes (AABB), and a multi-fidelity control-variate estimator (MFMC).

1) Monte Carlo (MC): MC estimates I by averaging binary inconsistency outcomes over $N _ { M C }$ sampled realizations $\xi ^ { ( k ) } \sim p ( \xi )$ •.

$$
\widehat { I } _ { \mathrm { M C } } = 1 - \frac { 1 } { N _ { M C } } \sum _ { k = 1 } ^ { N _ { M C } } C ( \xi ^ { ( k ) } ) .\tag{3}
$$

By the law of large numbers, $\widehat { I } _ { \mathrm { M C } }$ converges to I as defined in Equation 2, making it the most principled estimator. It is unbiased but computationally expensive [19] as accurate estimates require a large N. The standard error is $\sqrt { \widehat { I } ( 1 - \widehat { I } ) / N _ { M C } } .$ which upper-bounds at ≈ 0.016 (at I=0.5) for $N _ { M C } = 1 0 0 0$ This sets a label-noise floor on any surrogate fit to MC labels. Note that each $C ( \xi )$ is evaluated using an exact linear program membership test.

2) Geometric Proxy AABB: A computationally cheaper approximation replaces each zonotope $\mathcal { Z } _ { M _ { l } }$ by its axis-aligned bounding boxes (AABB) $B _ { l } .$ These AABBs are strict overapproximations that enclose the respective zonotopes but ignore their orientations and generator structures. This converts the exact set intersection into a box overlap problem, which can be evaluated efficiently using a geometric Jaccard overlap:

$$
J _ { \mathrm { A A B B } } = \frac { \mathrm { v o l } ( B _ { \cap } ) } { \mathrm { v o l } ( B _ { \cup } ) } , \quad B _ { \cap } = \bigcap _ { i } B _ { i } , \quad B _ { \cup } = \bigcup _ { i } B _ { i } .\tag{4}
$$

The resulting inconsistency approximation is then $\widehat { I } _ { \mathrm { A A B B } } = 1 -$ $J _ { \mathrm { A A B B } }$ . Because bounding boxes extend in all axis directions regardless of zonotope orientation, the union $B _ { \cup }$ grows faster than the intersection $B _ { \cap } .$ , reducing the Jaccard ratio and causing $I _ { \mathrm { A A B B } }$ to systematically overestimate inconsistency [20].

3) Multi-Fidelity Estimation (MFMC): MFMC reduces the cost of MC estimation by pairing a few-sample MC estimate with AABB as a low-fidelity control variate [14]. Since both $I _ { \mathrm { M C } }$ and $I _ { \mathrm { A A B B } }$ quantify inconsistency for the same random samples, they are correlated. This correlation can be exploited to reduce estimator variance through the correction. The resulting estimator is defined as

$$
\widehat { I } _ { \mathrm { M F } } = \widehat { I } _ { \mathrm { M C } } + \alpha \cdot \big ( \mu _ { \mathrm { A A B B } } - \widehat { I } _ { \mathrm { A A B B } } \big ) ,\tag{5}
$$

where $\widehat { I } _ { \mathrm { A A B B } }$ denotes the AABB inconsistency estimate on the same sample as $\widehat { I } _ { \mathrm { M C } } , ~ \mu _ { \mathrm { A A B B } }$ is a large-sample AABB estimate (which is cheap to obtain), and $\begin{array} { r } { \dot { \alpha } = \frac { \mathrm { C o v } \left( I _ { \mathrm { M C } } , I _ { \mathrm { A A B B } } \right) } { \mathrm { V a r } \left( I _ { \mathrm { A A B B } } \right) } } \end{array}$ is the optimal control-variate coefficient estimated from the same set [14]. Because of that, MFMC has a $\mathcal { O } ( 1 / N )$ finitesample bias. In our experiments, $r ( \widehat { I } _ { M C } , \widehat { I } _ { A A B B } ) \approx 0 . 9 4$ so the achieved variance-reduction $1 - r ^ { 2 } \approx 0 . 0 9 - 0 . 1 3$ , leading to an 8-11x variance reduction.

## C. Dataset Construction

To systematically analyze how uncertainty shapes inconsistency, we introduce an intervention configuration $\theta \in \Theta \subseteq \mathbb { R } ^ { 3 }$ that parameterizes the uncertainty geometry of each model $M _ { l }$ This induces parameterized sets ${ \mathcal { Z } } _ { M _ { l } } ( \theta )$ , a global feasibility region $\mathcal { Z } _ { \mathrm { s y s } } ( \theta )$ , and transforms the inconsistency response into a function

$$
I ( \theta ) = 1 - \mathbb { E } _ { \xi \sim p ( \xi ) } [ C ( \xi , \theta ) ]\tag{6}
$$

over the intervention space Θ.

The estimators above can evaluate $I ( \theta )$ for a fixed $\theta ,$ but large-scale analysis requires a dense coverage of Θ. We therefore construct an intervention-response dataset that serves two purposes: enabling the training of a surrogate model for efficient evaluation, and supporting downstream sensitivity analysis and recourse search.

![](images/1b8a715ef13e7814b3a435c43a2da2b754587a8fac84a90f50dbfc6a06915473.jpg)  
Fig. 3. Product-Set Transformer (PST) surrogate. Per-dimension features are projected, processed by a self-attention block, and decoded into per-dimension log-containment estimates whose sum encodes the conjunction structure of zonotope overlap.

N configurations $\theta _ { i } \in \Theta$ are samples using Saltelli sampling [21] to enable variance-based sensitivity analysis. Each configuration is labeled using MC, resulting in the dataset:

$$
\boldsymbol { D } = \{ ( \theta _ { i } , \hat { I } ( \theta _ { i } ) ) \} _ { i = 1 } ^ { N } ,\tag{7}
$$

The resulting dataset D provides a sample approximation of the inconsistency response landscape over Θ.

## D. Learning Inconsistency Response Surfaces

The problem with the previously mentioned approaches, especially MC, is that the estimation of I is expensive, and can quickly become intractable for problems requiring many evaluations of I. In fact, Froese et al. [13] show that containment of a zonotope within another (which is essential to the calculation of I) is coNP-complete and W[1]-hard in the dimension d. They further show a duality between ReLU networks and zonotopes, which motivates our surrogate model with ReLU nonlinearities.

Our proposed surrogate model takes as input a pair of zonotopes, $( \mathcal { Z } _ { 1 } , \mathcal { Z } _ { 2 } )$ , representing the source and target zonotopes. It then approximates the probability of inconsistency I, i.e.

$$
\hat { I } \approx 1 - \operatorname* { P r } ( c _ { 1 } + G _ { 1 } \xi \in \mathcal { Z } _ { 2 } ) , \qquad \xi \sim \mathcal { U } ( [ - 1 , 1 ] ^ { \gamma } ) ,\tag{8}
$$

The features to represent the zonotopes are computed in the propagated source frame (i.e., after applying the UMs), and are relative. This makes the representation invariant to a common translation, rotation, or scaling of the two zonotopes, which is the desired inductive bias for the above probability and enables the model to generalize to zonotopes of different scales.

a) Per-Dimension Features: We represent relative differences between the zonotopes per dimension using five different features. Let $\begin{array} { l c l } { \sigma _ { i } } & { = } & { r _ { 1 , i } + r _ { 2 , i } } \end{array}$ (where $r _ { k , i }$ 二 $| | G _ { k , i : } | | _ { 2 } )$ , then we define the following features: the normalized center offset $( c _ { 1 } - c _ { 2 } ) _ { i } / \sigma _ { i }$ , the source and target peraxis magnitudes $r _ { 1 , i } / { \sigma _ { i } } , r _ { 2 , i } / { \sigma _ { i } }$ , the generator-row alignment COS $\angle ( G _ { 1 , i } , G _ { 2 , i } )$ , and finally the width-ratio $r _ { 1 , i } / r _ { 2 , i }$

b) Global Features: The per-dimension features cannot capture all aspects of alignment between the zonotopes. Hence, we add additional global features: the log-volume ratio $( \sum _ { i = 1 } ^ { d }$ log $r _ { 1 , i } / r _ { 2 , i } )$ , normalized center distance $( | | c _ { 1 } -$ $c _ { 2 } | | / ( \mathsf { a v g } _ { i } ( \sigma _ { i } ) + \epsilon ) )$ , dimension $d \in \{ 2 , 3 , 4 \}$ , and two support function features: $| | c _ { 1 } - c _ { 2 } | | / ( h _ { 1 } + h _ { 2 } ) , | | c _ { 1 } - c _ { 2 } | | / h _ { 2 }$ $\begin{array} { r } { h _ { k } \ = \ \sum _ { j } | G _ { k , j } ^ { \top } v | } \end{array}$ is the support along the offset direction $v \ = \ ( c _ { 1 } - c _ { 2 } ) / | | c _ { 1 } - c _ { 2 } | |$ . These two features evaluate the agreement of the zonotopes along their offset vector instead of the coordinate axes. The rationale is that two zonotopes can be inconsistent even for a small offset if they are both narrow along v. The surrogate uses the explicit geometry of both models, supporting analysis of known configurations rather than prediction for unspecified designs.

c) Product-Set Transformer: To transform the features into an inconsistency estimate, we use a product-set transformer. An overview of the architecture is shown in Figure 3. We first project all per-dimension feature vectors to a 32- dim embedding. Then, we apply self-attention (2 heads, 64- dim feed-forward) to allow explicit interaction between the dimensions. A final per-dimension MLP outputs a logit $l _ { i } .$ The final output is then the joint non-containment probability:

$$
\hat { I } = 1 - \prod _ { i } p _ { i } \cdot p _ { g } = 1 - \exp ( \sum _ { i } \log p _ { i } + \log p _ { g } ) .\tag{9}
$$

Here, the per-dimension and global logits are transformed to probabilities using the sigmoid function $s \colon p _ { i } = s ( l _ { i } ) , p _ { g } =$ $s ( M L P ( g ) )$ , where $g$ is the global feature vector. This architecture models a noisy AND: for containment, a point must lie in both zonotopes in all dimensions. The intuition of the architecture is as follows. For diagonal generators, the joint probability factorizes into a product of per-dimension containment probabilities. But for non-diagonal generators, the probabilities of containment are statistically dependent between dimensions. Because self-attention is applied before the per-dimension factors, each factor $p _ { i }$ is conditioned on all other dimensions via attention, so their product implements an attention-parameterized factorization of the joint non-containment probability.

d) Training: We train two instances of the surrogate model. One model (PST-2D3D) for two- and threedimensional scenarios and one model (PST-4D) for fourdimensional scenarios. We train the models for 150 epochs on a mixture of real and synthetic (for 3D) zonotope scenarios using the Huber Loss and the AdamW optimizer with a learning rate of $1 0 ^ { - 3 }$ . For the ground-truth labels, we use the inconsistency calculated using the full MC scheme.

## E. Analyzing Inconsistency Response Surfaces

To systematically analyze the response surface Θ, we parametrize the modification of the uncertainty representation through an intervention vector.

![](images/e83fa00688421fd3d4b1717e5d7e7ba28edc93195c974168fcc780b9bdfd9145.jpg)  
Fig. 4. Intervention types on a zonotope (pre-intervention on the left): scaling, center shift, and correlation.

We define an intervention vector as $\boldsymbol { \theta } \ : = \ : [ s _ { u } , \Delta c _ { u } , R _ { u } ] ^ { T }$ where $s _ { u }$ parameterizes scale, $\Delta c _ { u }$ center-shift, and $R _ { u }$ correlation interventions, respectively. Figure 4 illustrates the three types of interventions. For each uncertainty u, we define $c _ { u } ( \theta ) = c _ { u } + \Delta c _ { u } , \quad G _ { u } ( \theta ) = s _ { u } G _ { u } R _ { u }$ . Generator coefficients $\xi _ { i }$ represent stochastic realizations of the uncertainty sets and are sampled during consistency evaluation. We apply an intervention as a controlled perturbation of the analysis parameter, written $\theta  w ;$ this denotes a what-if assignment to the uncertainty model.

a) Variance-Based Structural Sensitivity Analysis: To quantify the relative importance of uncertainty factors and their interactions, we employ Sobol variance decomposition. The first-order Sobol index is defined as

$$
S _ { j } = { \frac { \operatorname { V a r } _ { \theta _ { j } } \left( \mathbb { E } _ { \theta _ { \sim j } } [ I ( \pmb { \theta } ) \mid \theta _ { j } ] \right) } { \operatorname { V a r } ( I ( \pmb { \theta } ) ) } }\tag{10}
$$

quantifying the contribution of parameter $\theta _ { j }$ alone to the variance of the inconsistency function $I ( \pmb \theta )$

The total-effect index

$$
S _ { j } ^ { T } = 1 - \frac { \operatorname { V a r } _ { \pmb { \theta } _ { \sim j } } \left( \mathbb { E } [ I ( \pmb { \theta } ) | \pmb { \theta } _ { \sim j } ] \right) } { \operatorname { V a r } ( I ( \pmb { \theta } ) ) }\tag{11}
$$

captures both isolated and interaction-driven variance contributions involving $\theta _ { j }$

## F. Consistency Recourse

While intervention effects help to identify which uncertainty factors contribute the most to inconsistency, they do not indicate how consistency can be repaired. We therefore investigate how inconsistent uncertainty configurations can be repaired, while keeping the repaired parameters close to the original ones. Given an inconsistent intervention configuration $\theta ^ { * }$ with $I > l ,$ where l denotes the operational consistency threshold, the goal is to identify an intervention restoring consistency with a small change in $\theta ^ { * }$

Formally, we search for a recourse intervention

$$
\theta ^ { \prime } = \arg \operatorname* { m i n } _ { \theta \in \Theta } \| \theta - \theta ^ { * } \| _ { 2 } \quad \mathrm { s . t . } \quad \hat { I } ( \theta ) \leq l ,\tag{12}
$$

where distances are computed in the normalized intervention space.

This amounts to a continuous optimization problem, and we turn the constrained objective into an unconstrained objective with a penalty term:

$$
L ( \theta ) = \| \theta - \theta ^ { * } \| _ { 2 } ^ { 2 } + \lambda \cdot \operatorname* { m a x } ( 0 , \hat { I } ( \theta ) - l ) ^ { 2 } .
$$

We approach this optimization using both the MFMC estimator (within an evolutionary search algorithm (CMA-ES) and using finite-difference approximated gradients) and the surrogate model (which allows for direct gradient-based search since the surrogate model is fully differentiable w.r.t. θ).

The repair vector $\delta \theta = \theta ^ { \prime } - \theta ^ { * }$ quantifies which uncertainty parameters require modification to restore consistency.

## IV. EXPERIMENTAL ANALYSIS

This section analyzes the inconsistency response function $I ( \theta )$ across heterogeneous CPS domains using the dataset construction and estimation pipeline introduced in Section III-C. We first characterize the resulting inconsistency landscapes across domains before evaluating how reliably $I ( \theta )$ can be estimated, how efficiently these landscapes can be explored, and which uncertainty interventions most strongly influence inconsistency behavior. The analysis addresses the following research questions:

RQ1 How accurately and efficiently can the inconsistency response function $I ( \theta )$ be estimated using geometric, multi-fidelity, and learned estimators? (§IV-B)

RQ2 Can a learned surrogate model accurately approximate $I ( \theta )$ from generated intervention-response datasets while generalizing across unseen CPS scenarios ? (§IV-C)

RQ3 Which uncertainty parameters and parameter interactions most strongly drive inconsistency behavior across CPS domains? (§IV-D)

RQ4 Given a configuration that leads to inconsistency, what parameter intervention restores consistency? (§IV-E)

## A. Experimental Case Studies

We construct a structured intervention-response dataset spanning 48 scenarios across ten CPS domains. The scenarios cover systems engineering inconsistency situations as well as domain-specific CPS applications, including automotive systems, HVAC, industrial robotics, medical devices, railway systems, satellite systems, smart grids, water/chemical processes, and wind turbines. The dataset includes interval and probabilistic uncertainty sources derived from publicly available engineering datasets and standards, like PLEIAData [22], SWaT [23], and OpenFAST [24].

Each scenario defines an intervention space θ over zonotope-based uncertainty representations and is associated with an inconsistency response $I ( \theta )$ as defined in Equation 6. The intervention parameters control the scale, center displacement, and correlation structure of the uncertainty sets, corresponding to their main geometric degrees of freedom. The scale factor $s _ { u } \in [ 0 . 1 , 5 . 0 ]$ scales the uncertainty region compared to its nominal size, where $s _ { u } ~ < ~ 1$ generates an under approximation, and $s _ { u } > 1$ an over approximation. The normalized center shift $\Delta c _ { u } \in [ - 1 . 0 , 1 . 0 ]$ enables comparable interventions across scenarios with different uncertainty magnitudes. The correlation parameter $\rho _ { u } \in [ 0 . 0 , 0 . 9 5 ]$ ranges from independent to strongly coupled uncertainty dimensions while avoiding degenerate configurations. Intervention configurations are sampled using the Saltelli scheme [25], which systematically explores the intervention space θ by varying the intervention parameters over their respective ranges. Using $N = 2 0 4 8$ base samples, leads to 16,384 evaluations across the three intervention parameters. For all MC-based estimates, we use $N _ { M C } ~ = ~ 1 0 0 0$ samples per evaluation and set the consistency threshold $l = 0 . 5$ , i.e. inconsistent when most sampled realizations are not jointly feasible. All estimators are evaluated with an inductive, scenario-level hold-out, i.e. the surrogate (and the other estimators, although they are not trained) is evaluated only on scenarios it has not seen during training in all experiments below. Of the 48 scenarios, 35 are used for surrogate training, and 13 are held out for evaluation at the scenario level; held-out scenarios share domains with training scenarios but use distinct zonotope parameterizations and intervention configurations, evaluating within-domain generalization to unseen system configurations.

![](images/1278b97eabd1c451527988bd3831e060b91ca65d6034fe1b596e0814699ffdea.jpg)  
Fig. 5. Agreement of AABB, MFMC, and learned surrogate inconsistency estimates with the Monte Carlo reference. MFMC and the learned surrogate achieve substantially lower error and bias than AABB.

Additional implementation details, scenario specifications, dataset resources, and source code are available in the GitHub repository¹. All experiments were run on an Ubuntu server (AMD EPYC 9224, 24c/48t, 7 TiB SATA SSD), run in a Docker container limited to 16 vCPUs and 250 GiB RAM.

## B. RQ1: Estimating the Inconsistency Response Function

We evaluate how accurately and efficiently the inconsistency response function I(θ) can be estimated using geometric (AABB), multi-fidelity (MFMC), and learned estimators.

Figure 5 compares AABB, MFMC, and the learned surrogate against the MC reference. AABB shows substantially larger deviations and systematic overestimations, particularly in low-inconsistency regions $( \rho ~ = ~ 0 . 9 3 2 , ~ R ^ { 2 } ~ = ~ 0 . 8 1 5 )$ MFMC closely matches the MC reference $( \rho ~ = ~ 0 . 9 9 7$ $R ^ { 2 } = 0 . 9 9 7 )$ . The learned surrogate model also preserves the overall inconsistency structure well $( \rho = 0 . 9 9 3 , R ^ { 2 } = 0 . 9 8 3 )$ and is on par with MFMC in terms of rank correlation for 4D scenarios (+0.015). At $N _ { M C } = 1 0 0 0$ , the MC noise floor is 0.016, with an MAE of 0.011 the surrogate falls below it.

![](images/5785858c06679f3e090198f4114b0792a38a0e7cbcb3054ee3829aa9dfc5274c.jpg)  
Fig. 6. Tradeoff between estimation accuracy and computational cost across all CPS domains. MFMC achieves substantially lower error than AABB while remaining significantly faster than MC.

Figure 6 summarizes the resulting accuracy-efficiency tradeoff. MC provides the reference estimate at approximately 8998 µs per sample. AABB improves on this at approximately 1846 µs but with substantially degraded ranking quality and systematic overestimation. MFMC achieves near-reference accuracy at approximately 3306 $\mu s ,$ trading additional computation for substantially lower estimation error. The learned surrogate achieves a 935× speedup over MFMC to 3.54 µs per sample, while maintaining competitive accuracy. Notably, the MAE of the surrogate is below the standard error of the MC labels.

Overall, the results support a hierarchy among estimation paradigms: geometric over-estimators provide an inexpensive but coarse approximation with systematic bias; MFMC achieves near-reference accuracy at higher computational cost; and the learned surrogate uniquely combines competitive accuracy with orders-of-magnitude acceleration, enabling deployment at scales inaccessible to explicit numerical estimators.

## C. RQ2: Learning Inconsistency Response Surfaces

While MFMC substantially reduces the computational cost of estimating $I ( \theta )$ , large-scale exploration of interventionresponse spaces still requires repeated numerical evaluations. We therefore evaluate whether the learned surrogate model can accurately approximate the inconsistency response landscape to enable scalable exploration across unseen uncertainty configurations.

To evaluate generalization behavior, Table I reports estimator performance across different zonotope dimensions. The surrogate model maintains consistently strong performance across all evaluated dimensions, with $\rho$ values between 0.989 and 0.994 and MAE below 0.015. Unlike AABB, whose approximation quality remains largely unchanged across dimensions, the surrogate model maintains near-MFMC performance as dimensions increase. Notably, the surrogate model achieves its highest rank correlation in the 4D setting, despite having access to only 3 4D training scenarios. This indicates that the learned representation generalizes effectively beyond the dominant 2D benchmark setting , demonstrating withindomain generalization to unseen scenario configurations..

TABLE I  
ACCURACY METRICS BY ZONOTOPE DIMENSION. 95% BOOTSTRAP CIS COMPUTED BY RESAMPLING SALTELLI SAMPLES.
<table><tr><td>Dim</td><td>Method</td><td>ρ</td><td> $R ^ { 2 }$ </td><td>MAE</td></tr><tr><td rowspan="3">2D</td><td>AABB</td><td>0.932±0.001</td><td> $0 . 8 1 8 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 0 5 1 4 { \scriptstyle \pm 0 . 0 0 0 3 }$ </td></tr><tr><td>MFMC</td><td>0.997±0.000</td><td> $0 . 9 9 7 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 0 0 5 8 { \scriptstyle \pm 0 . 0 0 0 0 }$ </td></tr><tr><td>Surr.</td><td> $0 . 9 9 3 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 9 8 4 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 0 1 0 9 { \scriptstyle \pm 0 . 0 0 0 1 }$ </td></tr><tr><td rowspan="3">3D</td><td>AABB</td><td>0.949±0.003</td><td> $0 . 7 8 0 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 0 5 3 3 { \scriptstyle \pm 0 . 0 0 1 0 }$ </td></tr><tr><td>MFMC</td><td>0.994±0.001</td><td> $0 . 9 9 6 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 0 0 6 7 { \scriptstyle \pm 0 . 0 0 0 2 }$ </td></tr><tr><td>Surr.</td><td>0.989±0.001</td><td> $0 . 9 7 5 { \scriptstyle \pm 0 . 0 0 2 }$ </td><td> $0 . 0 1 5 0 { \scriptstyle \pm 0 . 0 0 0 4 }$ </td></tr><tr><td rowspan="3">4D</td><td>AABB</td><td>0.948±0.002</td><td> $0 . 7 7 1 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td> $0 . 0 4 5 2 { \scriptstyle \pm 0 . 0 0 1 0 }$ </td></tr><tr><td>MFMC</td><td>0.979±0.002</td><td> $0 . 9 9 5 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 0 0 6 8 { \scriptstyle \pm 0 . 0 0 0 2 }$ </td></tr><tr><td>Surr.</td><td> $0 . 9 9 4 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 9 8 9 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 0 0 9 8 { \scriptstyle \pm 0 . 0 0 0 2 }$ </td></tr></table>

TABLE ⅡI  
INFERENCE TIME PER SAMPLE AND ACHIEVABLE EVALUATIONS WITHIN FIXED TIME BUDGETS FOR THE 2D/3D AND 4D SURROGATE MODELS.
<table><tr><td rowspan="2">Method</td><td colspan="2">Time (µs/sample)</td><td colspan="2">Evals in 1 s</td><td colspan="2">Evals in 60 s</td></tr><tr><td>2D/3D</td><td>4D</td><td>2D/3D</td><td>4D</td><td>2D/3D</td><td>4D</td></tr><tr><td>MC</td><td>8,998</td><td>9,167</td><td>111</td><td>109</td><td>6.7 K</td><td>6.5K</td></tr><tr><td>AABB</td><td>1,846</td><td>1,627</td><td>541</td><td>614</td><td>32.5 K</td><td>36.9 K</td></tr><tr><td>MFMC</td><td>3,306</td><td>3,501</td><td>302</td><td>285</td><td>18.1 K</td><td>17.1 K</td></tr><tr><td>Surrogate</td><td>4</td><td>37</td><td>283,374</td><td>27,329</td><td>17.0M</td><td>1.6M</td></tr></table>

Table II evaluates the practical exploration capacity enabled by the different estimators. For the dominant 2D/3D benchmark setting, the surrogate model takes $4 \mu \mathrm { s }$ per evaluation, enabling approximately $2 . 8 \times 1 0 ^ { 5 }$ evaluations per second compared to roughly 300 for MFMC and 540 for AABB. Even in the 4D setting, inference remains below $4 0 , \mu \mathrm { s }$ per sample, allowing more than 27,000 evaluations per second. Within a one-minute budget, the surrogate enables evaluations up to 17 million, whereas MFMC remains below 20,000. These results demonstrate that learned surrogates fundamentally change the feasible scale of intervention-response exploration under realistic computational constraints.

Overall, the results demonstrate that learned surrogates can accurately reproduce the operational consistency boundaries of inconsistency response landscapes. At the same time, microsecond-scale inference enables exploration budgets several orders of magnitude larger than explicit MC or MFMC estimation, transforming previously infeasible large-scale intervention analysis into an interactive workflow.

Additionally, an ablation on identical features (full results in the accompanying repository¹) shows the PST improves $R ^ { 2 }$ from 0.974 for a plain MLP baseline to 0.995, with the gap widening on the harder 3D holdout (0.781 to 0.989), confirming that the attention and product-set structure contribute beyond the hand-crafted geometric features.

![](images/eadb6814edeb33a09c2d3f1287223b1eda305471947411aa8b0a19eaec746c2a.jpg)  
Fig. 7. Sobol sensitivity indices for MC and surrogate estimators.

## D. RQ3: Sensitivity and Structural Drivers of Inconsistency

Beyond predicting inconsistency values, engineers need insights into which uncertainty factors contribute most strongly to inconsistency and how these factors interact. We therefore investigate whether the learned surrogate model can accurately reproduce the sensitivity analysis that would otherwise require extensive numerical evaluation. In particular, we analyze the relative importance of different uncertainty factors, the role of parameter interactions, and how these effects are reflected in the resulting inconsistency response landscapes.

Figure 7 reports first-order $( S _ { 1 } )$ and total-effect $( S _ { T } )$ Sobol indices across all evaluated CPS domains. Each cell shows the mean index alongside a 95% bootstrap confidence interval (±); surrogate panels additionally annotate the signed difference $\Delta$ relative to MC. The narrow CIs observed across most domains (e.g. ±0.01–0.02 for Space and Water) confirm that the reported sensitivity rankings are statistically stable; the wider intervals in the Engineering domain (up to ±0.17 for ${ { R } _ { u } } )$ reflect the smaller number of available 4D scenarios rather than model instability.

Across nearly all domains, center uncertainty $\Delta c _ { u }$ exhibits the strongest first-order influence, with MC $S _ { 1 }$ values ranging from $0 . 3 6 \pm 0 . 0 6$ (Medical Device) to $0 . 4 5 \pm 0 . 0 6$ (Wind Turbine), while scale uncertainty $s _ { u }$ contributes secondary effects between 0.24–0.37. Correlation uncertainty $R _ { u }$ remains weak in isolation $( S _ { 1 } \leq 0 . 0 8 )$ but exhibits substantially larger total effects $( S _ { T } \leq 0 . 2 0 )$ , particularly in the Medical Device and Satellite domains. The resulting gap between $S _ { 1 }$ and $S _ { T }$ indicates that its influence is primarily interaction-driven rather than caused by isolated uncertainty effects.

Importantly, the surrogate reproduces the MC sensitivity structure with very high fidelity across all domains. Surrogate— MC differences satisfy $| \Delta | \leq 0 . 0 2$ for all first-order indices and $| \Delta | \leq 0 . 0 4$ for total effects across the nine CPS domains, with the largest deviation occurring in the Engineering domain $( \Delta S _ { T } ( s _ { u } ) = + 0 . 0 5 )$ , where CIs are also widest. For example, in the HVAC domain, MC yields $( S _ { 1 } , S _ { T } ) = ( 0 . 4 1 , 0 . 5 8 )$ for $\Delta c _ { u }$ , whereas the surrogate estimates (0.42, 0.60). Similarly, in the Wind Turbine domain, the surrogate accurately captures the strong center influence $( S _ { T } ~ = ~ 0 . 7 1 \pm 0 . 0 6$ vs. MC $S _ { T } = 0 . 7 0 \pm 0 . 0 7 , \Delta = + 0 . 0 1 )$ . These results indicate that the surrogate preserves both the global geometry of the inconsistency landscape and its underlying sensitivity structure.

![](images/78800ec630d8160197d54deac33d50ae7b2c2ad2eabe0c07e63b789ccb5b879f.jpg)  
Fig. 8. Intervention-response landscapes across representative CPS domains.

While the Sobol indices identify the dominant uncertainty factors, they do not reveal how inconsistency is distributed across the intervention space. Figure 8, therefore, visualizes representative intervention-response landscapes.

Across all three scenarios, scale and center uncertainty dominate first-order Sobol indices while correlation contributes only marginally (Figure 7); their relative balance differs: Building HVAC is center-led, whereas Medical Device and Satellite/Aerospace show near-equal contributions from both parameters. Nevertheless, they produce markedly different landscape geometries: HVAC contains a comparatively broad low-inconsistency region, whereas Medical Device remains highly inconsistent across most of the intervention space and exhibits only a narrow admissible region. The Satellite scenario represents the most constrained case: the inconsistency surface remains uniformly elevated above l across the entire intervention space, with no admissible region detectable.

Overall, the results reveal that inconsistency in CPS is strongly interaction-driven and cannot be fully characterized by global sensitivity measures alone. While Sobol indices identify the dominant uncertainty factors, the interventionresponse landscapes show that domains with similar sensitivity patterns can exhibit substantially different consistency regions. The learned surrogate preserves both the sensitivity structure and the response-landscape geometry sufficiently well to support scalable sensitivity and intervention analysis.

## E. RQ4: Consistency Recourse

Lastly, we want to answer how well the developed approaches can be used to repair inconsistent CPS by intervening in the uncertainty parameter space. This is important, because depending on the specific CPS, it must be as consistent as possible, or a minimal repair (i.e. small change to θ) to a predefined inconsistency level is required.

![](images/5f4e5ba3586793c886cce5ea15d93981953c853041dc7401cccc681f612f2274.jpg)  
Fig. 9. Minimal recourse trajectory for the Engineering domain. The surrogate converges directly to a point near the boundary while the FD makes suboptimal steps and overshoots. CMA-ES reaches a slightly closer point than the surrogate model with significantly more steps.

We evaluate four different search strategies for that: An evolutionary search (CMA-ES) guided by the MFMC estimator, a finite-difference gradient approximation search with MFMC (FD), a direct gradient-based search with the surrogate model, and a hybrid search where we first search using the surrogate model and then refine up to 10 steps using the FD MFMC estimator. For the derivative-free baselines (CMA-ES and FD), we set $\lambda = 1 0 0 !$ for the surrogate-based gradient search $\lambda = 4 2 . 5 . \mathrm { A l l }$ results below are assessed using a full MC estimator (4k samples).

Figure 9 illustrates the search problem. It shows the optimization trajectories of the compared methods (hybrid is not shown since the surrogate alone converged here to a valid point below $I = 0 . 5 )$ . All approaches start from $\theta ^ { * }$ and aim to find a configuration with $I \ < \ 0 . 5$ that has the smallest possible distance to $\theta ^ { * }$ . Both the surrogate model and CMA-ES guided by MFMC converge straight to the closest feasible border. However, CMA-ES does this with multiple (25 in total) costly MFMC evaluations. The surrogate slightly overshoots the boundary but then converges close to it inside the feasible region. On the contrary, the FD gradient with MFMC starts into a suboptimal direction and then overshoots widely into the feasible region and must backtrack to the boundary.

![](images/259f40a81ec3fce59e30aa31a458ccc00e2f1d7d6dc691b0882722cde3852321.jpg)  
wall-time per repair [s] (median)  
Fig. 10. Walltime vs. success rate (filled) and repair distance (hollow) with penalized minimal repair optimization.

Figure 10 shows the average success-rate and repair distance against wall-time over 13 held-out evaluation scenarios for the minimal distance repair (i.e., optimizing Eq. 12). Success rate is the number of trials in which the found solution has $I < 0 . 5 ,$ and the repair distance is the distance of the found parameters to the starting parameters. We tested three variants for CMA-ES and FD, 50, 150 or 400 MFMC evaluations as the upper limit. Hollow markers indicate repair distance, and solid ones indicate success rate. The plot shows that all approaches (except the surrogate alone) reach a high success rate of over 90%. Surrogate and Hybrid converge to a slightly higher repair distance. Yet, this comes at an order-of-magnitude faster runtime. I.e., the surrogate model finds slightly worse repairs, but at roughly 30 ms rather than over 10 seconds.

The results for the unconstrained search for a low I configuration are shown in Figure 11. The x-axis again shows the wall time, and the y-axis shows the median inconsistency reached for all methods, across the same eval scenarios as before. The surrogate alone finds configurations with small I, but significantly higher than all other approaches, which converge to a median of 0, i.e., they find fully consistent configurations. Again, the Hybrid model offers the best tradeoff here, and finds these configurations in almost half the runtime of CMA-ES, while FD needs at 150 evals to converge to the same median.

Together, these results demonstrate that both the MFMC and surrogate estimator can be effectively used to search for low-inconsistency configurations, including ones with minimal distance to the initial configuration. While the MFMC-based search offers slightly better results, the surrogate-based gradient search offers a significantly faster runtime.

## V. RELATED WORK

a) Consistency Analysis under Uncertainty: Consistency management in multi-model systems is traditionally studied in model-based systems engineering through explicit consistency relations and transformation-based preservation rules [26], [27]. These approaches operate on deterministic model states and focus on verifying or restoring consistency for fixed system configurations. More recent work considers uncertaintyaware consistency analysis [28], but remains centered on local verification rather than scalable analysis of global inconsistency behavior across uncertainty spaces. In contrast, this paper formulates inconsistency as a stochastic response function over uncertainty interventions, enabling scalable estimation, surrogate learning, sensitivity analysis, and intervention-based reasoning over inconsistency landscapes.

![](images/7a116dca6664fe0d40e3f98000a138328c2a0aa485c6ad7fa38d7104de413e1f.jpg)  
wall-time per repair [s] (median)  
Fig. 11. Absolute Runtime vs. median Inconsistency reached for unconstrained repair.

b) Response-Function Approximation: Approximating computationally expensive response functions is a central challenge in uncertainty quantification, simulation analytics, and scientific machine learning [10], [11], [29].

Multi-fidelity and variance-reduction methods approximate expensive stochastic estimators by combining inexpensive approximations with high-fidelity evaluations [12], [14], [30]. Recent work combines surrogate modeling with sensitivity analysis and explainability techniques to enable scalable exploration of complex simulation-driven systems [31].

Close to our setting, prior work has studied neural prediction of geometric overlap and convex-set properties. Yuan [32] proposes one of the earliest neural approaches for measuring the intersection of convex polyhedra, and Bao et al. [33] demonstrated that geometric polytope properties can be effectively learned from structured representations. Siamese overlap-prediction networks [34] further provide mechanisms for learning over set-valued geometric inputs.

In contrast to prior work focusing on pairwise overlap prediction or geometric property estimation, this paper studies inconsistency itself as a stochastic response function induced by uncertainty propagation across interacting multi-model systems. We combine multi-fidelity approximation with a learned surrogate operating directly on zonotope-based uncertainty representations to estimate inconsistency response surfaces over uncertainty interventions.

c) Sensitivity Analysis and Feature Importance: Global sensitivity analysis is widely used to quantify how uncertainty in model parameters affects system behavior. Variance-based approaches such as Sobol indices [35] and their estimation via Saltelli sampling [25] are commonly applied to identify influential input parameters. Janzing et al. [36] formalize feature relevance as a causal problem and argue that meaningful importance measures require interventional rather than purely observational reasoning. Similarly, Wachter et al. [37] introduce counterfactual explanations based on minimal interventions that alter model outcomes. Ustun et al. [38] frame this as algorithmic recourse: the minimal actionable change that flips an unfavorable outcome. More recently, Dyer et al. [39] propose interventionally consistent surrogate models that remain valid under distributional interventions.

TABLE III  
PRODUCT-SET TRANSFORMER ARCHITECTURE ABLATION ON THE 2D/3D SPLIT (THE 12 HELD-OUT 2D/3D SCENARIOS OF THE 13 HELD OUT OVERALL, INCLUDING THE 3D HOLDOUT; 200 EPOCHS; METRICS AT THE BEST VALIDATION CHECKPOINT). FL AT MLP IS THE LEARNED BASELINE ON IDENTICAL FEATURES.
<table><tr><td>Variant</td><td>Params</td><td>MSE</td><td> $R ^ { 2 }$ </td><td>ρ</td><td> $R _ { 3 \mathrm { D } } ^ { 2 }$ </td></tr><tr><td>PST (full)</td><td>9,394</td><td>0.00014</td><td>0.9947</td><td>0.9963</td><td>0.9887</td></tr><tr><td>Set Transformer</td><td>10,609</td><td>0.00033</td><td>0.9875</td><td>0.9926</td><td>0.9300</td></tr><tr><td>DeepSets</td><td>8,897</td><td>0.00037</td><td>0.9859</td><td>0.9910</td><td>0.9716</td></tr><tr><td>PST, no attention</td><td>850</td><td>0.00039</td><td>0.9849</td><td>0.9918</td><td>0.9793</td></tr><tr><td>Flat MLP (baseline)</td><td>7,937</td><td>0.00067</td><td>0.9742</td><td>0.9857</td><td>0.7812</td></tr></table>

Building on these perspectives, our work combines global sensitivity analysis with intervention-based reasoning and algorithmic recourse to identify dominant uncertainty drivers of inconsistency and to identify uncertainty interventions that restore consistency.

## VI. CONCLUSION

We present a scalable framework for estimating, analyzing, and repairing inconsistencies under uncertainty in CPS development. By reformulating inconsistency as an interventionresponse function I(θ), we enable systematic data generation via MFMC and Saltelli sampling, surrogate-based exploration at the microsecond scale, and consistency recourse.

Our experimental results across 48 scenarios showed that the surrogate model accurately approximates Monte Carlo inconsistency estimates and Sobol indices, models the inconsistency landscape faithfully, while reducing evaluation time from milliseconds to microseconds. Finally, the proposed recourse approach demonstrates that both MFMC and the surrogate can not only be used to explain inconsistency but also to identify uncertainty modifications that restore consistency.

Future work will investigate larger multi-model structures and higher-dimensional uncertainty representations. We also plan to study adaptive sampling strategies that focus on informative regions of the inconsistency response surface, reducing the cost of data generation and surrogate training.

## APPENDIX A

## SURROGATE ARCHITECTURE ABLATION

Table III ablates the components of the Product-Set Transformer on the 2D/3D split. The full model (attention plus product head) and the variant without the global head are indistinguishable on aggregate metrics, indicating that the per-dimension factors carry most of the signal. Removing the product head (Set Transformer) or both the product head and attention (DeepSet s) degrades accuracy; removing attention alone (PST, no attention) collapses the model to 850 parameters at a comparable cost, confirming that both the noisy-AND factorization and the attention contribute. The flat MLP on identical features matches the PST on the 2D scenarios but collapses on the harder, data-scarce 3D holdout $( R _ { 3 \mathrm { D } } ^ { 2 } ~ = ~ 0 . 7 8 1 ~ \mathrm { v s . } ~ 0 . 9 8 9 )$ , showing that the architecture's benefit is concentrated where data is scarce and the problem is hardest. Set Transformer and DeepSets reach their best validation loss within the first two epochs and degrade thereafter; all reported numbers use the best checkpoint.

## REFERENCES

[1] E. A. Lee and S. A. Seshia, Introduction to embedded systems: A cyberphysical systems approach. MIT press, 2016.

[2] A. M. Madni and M. Sievers, "Model-based systems engineering: Motivation, current status, and research opportunities," Systems Engineering, vol. 21, no. 3, pp. 172–190, 2018.

[3] A. Bhave, B. H. Krogh, D. Garlan, and B. Schmerl, "View consistency in architectures for cyber-physical systems," in IEEE/ACM second international conference on cyber-physical systems, pp. 151–160.

[4] J. Troya, N. Moreno, M. F. Bertoa, and A. Vallecillo, "Uncertainty representation in software models: a survey," Software and Systems Modeling, vol. 20, no. 4, pp. 1183–1213, 2021.

[5] J. Mäkelburg, D. Perez-Palacin, R. Mirandola, and M. Acosta, "Surveying uncertainty representation: a unified model for cyber-physical systems," Computing, vol. 108, no. 5, p. 68, 2026.

[6] Y. Mordecai and D. Dori, “Minding the cyber-physical gap: Model-based analysis and mitigation of systemic perception-induced failure," Sensors, vol. 17, no. 7, p. 1644, 2017.

[7] L. V. Nguyen, K. A. Hoque, S. Bak, S. Drager, and T. T. Johnson, "Cyber-physical specification mismatches," ACM Transactions on Cyber-Physical Systems, vol. 2, no. 4, pp. 1–26, 2018.

[8] J. K. Scott, D. M. Raimondo, G. R. Marseglia, and R. D. Braatz, “Constrained zonotopes: A new tool for set-based estimation and fault detection," Automatica, vol. 69, pp. 126–136, 2016.

[9] M. Althoff, O. Stursberg, and M. Buss, "Reachability analysis of nonlinear systems with uncertain parameters using conservative linearization," in 47th IEEE Conference on Decision and Control. IEEE, 2008, pp. 4042-4048.

[10] J. Sacks, W. J. Welch, T. J. Mitchell, and H. P. Wynn, "Design and analysis of computer experiments," Statistical science, vol. 4, no. 4, pp. 409–423, 1989.

[11] A. Forrester, A. Sobester, and A. Keane, Engineering design via surrogate modelling: a practical guide. John Wiley & Sons, 2008.

[12] B. Peherstorfer, K. Willcox, and M. Gunzburger, “Survey of multifidelity methods in uncertainty propagation, inference, and optimization,"Siam Review, vol. 60, no. 3, pp. 550–591, 2018.

[13] V. Froese, M. Grillo, C. Hertrich, and M. Stargalla, "Parameterized hardness of zonotope containment and neural network verification," in International Conference on Learning Representations, vol. 2026, 2026, pp. 46 708–46 720.

[14] B. Peherstorfer, K. Willcox, and M. Gunzburger, “Optimal model management for multifidelity monte carlo estimation," SIAM Journal on Scientific Computing, vol. 38, no. 5, pp. A3163–A3194, 2016.

[15] L. Schäfer, F. Gruber, and M. Althoff, "Scalable computation of robust control invariant sets of nonlinear systems," IEEE Transactions on Automatic Control, vol. 69, no. 2, pp. 755–770, 2023.

[16] M. Althoff, O. Stursberg, and M. Buss, “Reachability analysis of linear systems with uncertain parameters and inputs," in 46th IEEE Conference on Decision and Control. IEEE, 2007, pp. 726–732.

[17] A. Girard, “Reachability of uncertain linear systems using zonotopes," in International workshop on hybrid systems: Computation and control. Springer, 2005, pp. 291–305.

[18] W. Härdle and L. Simar, Applied multivariate statistical analysis. Springer, 2007.

[19] R. Y. Rubinstein and D. P. Kroese, Simulation and the Monte Carlo method. John Wiley & Sons, 2016.

[20] C. Ericson, Real-time collision detection. Crc Press, 2004.

[21] A. Saltelli, “Making best use of model evaluations to compute sensitivity indices," Computer physics communications, vol. 145, no. 2, pp. 280– 297, 2002.

[22] A. M. Ibarra, A. González-Vidal, and A. Skarmeta, "Pleiadata: consumption, hvac, temperature, weather and motion sensor data for smart buildings applications," Scientific Data, vol. 10, no. 1, p. 118, 2023.

[23] J. Goh, S. Adepu, K. N. Junejo, and A. Mathur, “A dataset to support research in the design of secure water treatment systems," in International conference on critical information infrastructures security. Springer, 2016, pp. 88–99.

[24] B. Jonkman, A. Platt, R. M. Mudafort, E. Branlard, M. Sprague et al., "OpenFAST: v4.0.0," 2024.

[25] A. Saltelli, P. Annoni, I. Azzini, F. Campolongo, M. Ratto, and S. Tarantola, "Variance based sensitivity analysis of model output. design and estimator for the total sensitivity index," Computer physics communications, vol. 181, no. 2, pp. 259–270, 2010.

[26] P. Stevens, "Bidirectional transformations in the large," in 2017 ACM/IEEE 20th International Conference on Model Driven Engineering Languages and Systems (MODELS). IEEE, 2017, pp. 1–11.

[27] P. Stünkel, H. König, Y. Lamo, and A. Rutle, “Comprehensive systems: a formal foundation for multi-model consistency management," Formal Aspects of Computing, vol. 33, no. 6, pp. 1067–1114, 2021.

[28] R. Jongeling and A. Vallecillo, "Uncertainty-aware consistency checking in industrial settings," in 2023 ACM/IEEE 26th International Conference on Model Driven Engineering Languages and Systems (MODELS). IEEE, 2023, pp. 73–83.

[29] G. E. Karniadakis, I. G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang, and L. Yang, “"Physics-informed machine learning," Nature Reviews Physics, vol. 3, no. 6, pp. 422–440, 2021.

[30] M. C. Kennedy and A. O'Hagan, "Predicting the output from a complex computer code when fast approximations are available," Biometrika, vol. 87, no. 1, pp. 1–13, 2000.

[31] P. Saves, P. S. Palar, M. D. Robani, N. Verstaevel, M. Garouani, J. Aligon, B. Gaudou, K. Shimoyama, and J. Morlier, "Surrogate modeling and explainable artificial intelligence for complex systems: A workflow for automated simulation exploration," arXiv preprint arXiv:2510.16742, 2025.

[32] J. Yuan, “A neural network measuring the intersection of m-dimensional convex polyhedra," Automatica, vol. 31, no. 4, pp. 517–529, 1995.

[33] J. Bao, Y.-H. He, E. Hirst, J. Hofscheier, A. Kasprzyk, and S. Majumder, “Polytopes and machine learning," International Journal of Data Science in the Mathematical Sciences, vol. 1, no. 02, pp. 181–211, 2023.

[34] X. Chen, T. Läbe, A. Milioto, T. Röhling, J. Behley, and C. Stachniss, "Overlapnet: A siamese network for computing lidar scan similarity with applications to loop closing and localization," Autonomous Robots, vol. 46, no. 1, pp. 61–81, 2022.

[35] I. M. Sobol, “Global sensitivity indices for nonlinear mathematical models and their monte carlo estimates," Mathematics and computers in simulation, vol. 55, no. 1-3, pp. 271–280, 2001.

[36] D. Janzing, L. Minorics, and P. Blöbaum, "Feature relevance quantification in explainable ai: A causal problem," in International Conference on artificial intelligence and statistics. PMLR, 2020, pp. 2907–2916.

[37] S. Wachter, B. Mittelstadt, and C. Russell, "Counterfactual explanations without opening the black box: Automated decisions and the gdpr," Harv. JL & Tech., vol. 31, p. 841, 2017.

[38] B. Ustun, A. Spangher, and Y. Liu, "Actionable recourse in linear classification," in Proceedings of the Conference on Fairness, Accountability, and Transparency, danah boyd and J. H. Morgenstern, Eds. ACM, 2019, pp. 10–19.

[39] J. Dyer, N. Bishop, Y. Felekis, F. M. Zennaro, A. Calinescu, T. Damoulas, and M. Wooldridge, "Interventionally consistent surrogates for complex simulation models,"Advances in Neural Information Processing Systems, vol. 37, pp. 21 814–21 841, 2024.