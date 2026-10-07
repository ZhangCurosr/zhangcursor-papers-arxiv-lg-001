# A Decision-Focused Neural Optimization Framework for Personalized Route Reproduction from Vehicle Trajectories

Gyeongjun Kim   
Graduate student   
Department of Smart City, Chung-Ang University, Seoul, Korea   
84 Heukseok-ro, Dongjak-gu, Seoul 156-756, Korea   
kkjunn7033@cau.ac.kr

## Yeseul Kang

Graduate student   
Department of Urban Engineering, Chung-Ang University, Seoul, Korea   
84 Heukseok-ro, Dongjak-gu, Seoul 156-756, Korea   
dptmf0307@cau.ac.kr   
Keemin Sohn\* Corresponding author   
Professor   
Department of Urban Engineering, Department of Smart City, Chung-Ang University, Seoul, Korea   
84 Heukseok-ro, Dongjak-gu, Seoul 156-756, Korea   
kmsohn@cau.ac.kr

## Abstract

To reproduce observed routes at the individual level, this study revisits the fundamental premise that travelers aim to minimize their perceived costs. Once these subjective link travel costs are identified, the route choice problem reduces to a shortest-path search without the need to enumerate alternative routes. We propose a neural pipeline that includes a perception model that embeds context covariates, which comprises individual characteristics, trip-specific attributes, and network-level traffic states, into the perceived link costs. A constrained optimization (CO) layer, which determines the shortest path (SP) based on these estimated costs, follows the perception encoder. To enable end-to-end training, we employ decision-focused learning to align the predicted shortest paths with observed routes. The implicit maximum likelihood estimation (iMLE) provides an approximate gradient of the loss function that contains the non-differentiable CO layer. Furthermore, a regularization term added to the loss function ensures proper scale of perceived link travel costs. Empirical evaluations demonstrate that the proposed framework outperforms baseline route choice models in path reproduction. The framework’s capability of retrieving perceived travel costs offers plausible interpretation for route choice motives at the individual level, providing insights into why drivers exhibit heterogeneous behavior under identical traffic conditions.

## Keywords

Route choice, Perceived travel cost, Decision-focused learning, Individual heterogeneity, Implicit maximum likelihood

## 1. Introduction

The route choice behavior has long been studied in transportation research fields (Ben-Akiva and Lerman, 1985; Bovy and Stern, 1990). Empirical evidence consistently demonstrates that travelers exhibit significant heterogeneity, choosing diverse routes even under identical traffic conditions (Jan et al., 2000; Papinski et al., 2009; Levinson and Zhu, 2013). Empirical studies further reveal that the proportion of travelers opting for the objectively minimum travel time path, even when accounting for network congestion, typically ranges between only 40% and 60% (Ramírez et al., 2021; Zhu and Levinson, 2015). To account for this behavioral diversity, discrete choice models (DCMs) have been extensively employed.

However, traditional DCMs encounter significant computational hurdles, particularly regarding the generation of choice sets. As enumerating all feasible paths in a large-scale network is practically infeasible, model estimation often relies on a pre-defined subset of routes, which entails potential biases (Ben-Akiva and Bierlaire, 1999; Bekhor et al., 2001; Frejinger et al., 2009). While the recursive logit model addresses the path enumeration problem through a link-based formulation (Fosgerau et al., 2013), it remains constrained by the independence of irrelevant alternatives (IIA) property. Consequently, logit-based route choice models struggle to accurately reproduce rational route choice behavior when candidate paths exhibit high degrees of spatial overlap. At the network level, Fosgerau et al. (2022) proposed the perturbed utility route choice (PURC) model that is also free from both the alternative enumeration and the IIA challenge. The model calibration method was also extended to utilize individual trajectories (Fosgerau et al., 2024). The extension, however, is not to retrieve the individual-level perceived link-cost from observed trajectories but to calibrate the model to retrieve link traffic volumes.

In this study, we revisit the perceived-cost minimization principle: conditional on a traveler-specific perceived link costs, the realized route is represented as the minimum cost path. Although this principle is a fundamental behavioral premise in route choice and traffic assignment, existing models typically use it indirectly through likelihood-based route probabilities (Frejinger et al., 2009), recursive value functions (Fosgerau et al., 2013), perturbed utility flow allocation (Fosgerau et al., 2022), or latent choice-set perception (Yao and Bekhor, 2022). To the best of our knowledge, few studies have directly considered an individual-level perceived link costs by embedding the full-network shortestpath (SP) finding operator in the learning objective and matching its resulting path to each observed trajectory. Including the SP finding operation makes it difficult to calibrate the model via maximizing the likelihood. Kim et al. (2026) sheds light on utilizing the principle directly for the model calibration, wherein a probit choice model was calibrated directly by the random utility maximization (RUM) theory at the individual level, not depending on the likelihood maximization that demands the evaluation of probit choice function. Whereas this previous study approximates the ‘argmax’ operation with a differentiable function, the present study keeps the shortest-path layer discrete in the forward pass and uses an iMLE-based surrogate gradient for backpropagation.

We devise a modeling framework to retrieve the latent perceived costs that lead to the observed paths. The proposed framework encodes context covariates, including individual characteristics, tripspecific attributes, and network-wide traffic states, into perceived link costs. A constrained optimization (CO) layer to determine the minimum-cost path based on the principle above, effectively reproducing the observed trajectory, is then attached to the encoder. This modeling framework is free from preparing an alternative route set that is prerequisite for other path-based DCMs. It should be noted that the proposed framework is not a likelihood-based route choice probability estimator. Rather, the framework is an intelligent route-reproduction framework that retrieves individual-specific perceived link costs and evaluates them through a full-network SP layer.

While the proposed end-to-end pipeline is conceptually intuitive and computationally efficient during inference, its training phase presents non-trivial mathematical challenges. Specifically, backpropagating gradients through a discrete SP solver requires addressing the non-differentiability. Unlike conventional supervised learning, which directly maps inputs to labels, decision-focused learning seeks to align the solution of an optimization problem with the revealed output. To overcome the gradients' vanishing or exploding nature in such neural-optimization pipelines, recent advancements have introduced techniques to enable backpropagation through CO layers (Wilder et al., 2019; Elmachtoub and Grigas, 2022; Mandi et al., 2024). In this study, we propose a robust calibration methodology by employing the implicit maximum likelihood estimation (iMLE) method. The iMLE method provides an approximate gradient of the loss function that contains nondifferentiable CO layers (Vlastelica et al., 2020; Niepert et al., 2021), thereby allowing for the calibration of the end-to-end route reproduction model and the retrieval of perceived travel costs, while maintaining the structural integrity of the underlying optimization problem.

Another critical challenge in estimating perceived link travel costs is ensuring the unique set of perceived travel times. Since multiple sets of perceived link costs can yield the identical SP solution, the estimation process is inherently ill-posed. To alleviate this non-uniqueness problem, we introduce a regularization term to the loss function when training, which ensures cost-scale anchoring. This regularization effectively anchors the estimated costs at the average travel times over trips, resulting in perceived link travel costs that align with the scale of observed travel costs. By penalizing extreme deviations, the proposed approach provides a robust and interpretable set of latent costs that maintains physical coherence.

The contributions of this study are three-fold. First, an individual-level personalized route reproduction framework is proposed. We devise a framework that operates directly on perceived link travel costs, eliminating the need for explicitly enumerating candidate routes. Second, decisionfocused learning (Mandi et al., 2024) is adapted to calibrate the proposed route-reproduction framework. The iMLE-based surrogate gradient enables end-to-end training (Niepert et al., 2021), while preserving the discrete SP solver in the forward pass. Third, the retrieval of perceived travel costs can provide a mechanistic explanation for personalized route choice behavior. Unlike existing discrete choice models that rely on aggregate stochasticity, our framework offers granular interpretability by identifying the latent cost structures that drive heterogeneous decisions under identical traffic conditions.

The remainder of this paper is organized as follows. Section 2 provides a comprehensive literature review, focusing on the evolution of route choice models within the discrete choice modeling (DCM) framework and identifying the gaps our research aims to bridge. This section also introduces several similar studies in the field of transportation and highlights the superiority of the present study. Section 3 formalizes the proposed modeling framework, details the calibration process via decisionfocused learning, and introduces the regularization scheme designed to address the cost-scale ambiguity. Section 4 describes the datasets utilized for the analysis and outlines the experimental configuration. Section 5 presents empirical results and discussions, where the proposed framework is validated against real-world vehicle trajectories to assess its performance and interpretability. Finally, Section 6 concludes the study by summarizing the key findings and discussing potential avenues for future research.

## 2. Related work

## 2.1 Discrete choice models of route choice

Route choice has traditionally been modeled within the discrete choice framework, which broadly distinguishes between path-based and link-based formulations. Path-based models represent route choice as a selection among complete origin–destination paths and are attractive because their behavioral interpretation is straightforward at the route level (Ben-Akiva and Bierlaire, 1999; Prato, 2009). However, path-based models face two persistent challenges. They require a finite candidate path set to be generated in advance, and their estimates can be sensitive to how that choice set is constructed. A second difficulty is that candidate routes often overlap substantially in real networks, which violates the independence assumptions of simple multinomial logit specifications and can lead to unrealistic substitution patterns across. To mitigate this issue, overlap-correction models such as the path size logit have been widely adopted, while later studies also explored richer choiceset generation and path representation strategies (Ben-Akiva and Bierlaire, 1999; Frejinger et al., 2009; Marra and Corman, 2020).

These limitations motivated the development of link-based model, which formulate route choice as a sequence of local link decisions on the network rather than as a single selection from a preenumerated path set. Recursive logit model (Fosgerau et al., 2013) is especially influential because it avoids explicit path enumeration, is equivalent to a multinomial logit model with an unrestricted path set, and provides a tractable framework for estimation and prediction on large networks. Subsequent extensions, including the nested recursive logit and discounted recursive logit, further improved the flexibility of the link-based framework and demonstrated its usefulness in diverse route choice settings and travel modes (Mai et al., 2015; Oyama and Hato, 2017; Zimmermann et al., 2017; Meyer de Freitas et al., 2019). Nevertheless, most existing discrete route choice models still rely on relatively simple linear utility specifications and link-additive representations, which limit their ability to capture nonlinear effects, richer trip context, and traveler-specific heterogeneity in perceived network conditions

## 2.2 Learning-based route choice

Beyond econometric and behavioral formulations, recent studies have increasingly explored learning-based approaches to relax the restrictive functional forms of classical route choice models and to better exploit rich trajectory and network data (Cantarella and de Luca, 2005; Wang et al., 2020; Zhao and Liang, 2023). On the path-based side, deep neural network variants of path size logit have been proposed to replace linear utility specifications with more flexible nonlinear mappings while retaining candidate-set-based route probabilities (Marra and Corman, 2021). These models improve representational flexibility, but they still inherit the structural dependence of pathbased approaches on route generation and candidate-set coverage

On the link-based side, inverse reinforcement learning and related sequential learning frameworks have been used to model route choice as a policy or reward-learning problem. Zhao and Liang (2023), for example, adopted an adversarial inverse reinforcement learning (AIRL) for route choice that learns a context-dependent reward function from trajectory data. Sequence-based route choice and path-inference models such as RoutesFormer further show that learned representations can improve predictive accuracy by exploiting longer-range dependency structures in observed trajectories (Qiu et al., 2024). These studies demonstrate the value of flexible, learned route-generation mechanisms and provide strong empirical baselines for comparison. However, most learning-based route choice methods remain focused on OD-conditional route distributions, link-choice policies, or latent reward functions. They do not attempt the behavioral interpretation as a driver’s perceived link cost distribution.

## 2.3 Similar Studies with Latent Travel Costs

The perturbed utility route choice (PURC) model proposed by Fosgerau et al. (2022) is especially relevant because it also avoids explicit choice-set generation and operates directly on the full network. PURC represents route choice through a perturbed utility formulation and predicts network-consistent link-use outcomes by solving a convex optimization problem. Unlike recursive logit, it is not subject to the standard IIA limitation associated with logit-based path or link choice models. More recent trajectory-based extensions further allow PURC-type models to be estimated from individual observed paths (Fosgerau et al., 2024). More recently, Rosenblad et al. (2026) further develops PURC estimation by proposing a convex inverse-estimation framework using observed link-use proportions constructed from vehicle trajectory data.

These PURC studies provide the closest full-network statistical benchmark for the present study. Nevertheless, their prediction output is an expected link-use or flow vector at each origin destination (OD) pair, rather than a deterministic reproduced path for each individual trip. Moreover, PURC is not designed to retrieve individual-specific perceived link costs that rationalizes each driver’s realized trajectory under a SP rule. In this sense, PURC and its inverse-estimation extensions are highly relevant but differ from the target output and interpretation of the proposed framework.

Recently, a parallel line of studies has integrated an optimization layer into a neural pipeline (Liu et al., 2023), into an objective function for model calibration (Sun and Arslan, 2025), or into a fullnetwork route reproduction formulation without explicit path enumeration (Fosgerau et al., 2022). These studies share the view that observed route or network outcomes can be better reproduced when the model uses latent or perturbed link costs rather than raw observed travel times alone. A key idea is that an optimization solver can act as a behavioral or operational layer that maps latent costs into observable route, tour, or flow outcomes.

Liu et al. (2023) learn latent cost structures under traffic assignment constraints by adding a userequilibrium (UE) traffic assignment layer to a neural pipeline to fit multi-day link flow observations. This shows that unobserved cost components can be inferred from demand and context data. However, the resulting travel cost is essentially a population-level cost surface shared across travelers, and thus it does not directly explain why a specific individual chooses a particular realized route. They transform the UE traffic assignment problem into a variational inequality (VI) formulation and suggest a fixed-point algorithm for solving the problem. A gradient-based algorithm is adopted for training, and the gradient is derived via the implicit differentiation theorem. Deriving the gradient, however, requires handling a large-dimensional Jacobian, which entails a substantial computationa burden.

Sun and Arslan (2025) suggest an experienced learning approach for solving a traveling salesman problem (TSP). They learn data-driven transition weights that adjust edge costs so that a routing solver generates solutions consistent with observed high-quality tours in last-mile delivery. While the learned adjustments improve realism, they are designed as a shared set of transition weights that can be reused across operations, rather than as personalized perceived link costs. In short, latent cost learning and perturbed utility modeling have progressed toward calibrating hidden network cost components from observed data, but existing approaches typically target aggregate link flows, expected link-use probabilities, and shared routing weights. This limits their ability to interpret why a particular individual selected a particular realized trajectory.

In contrast, the present study targets individual-level route reproduction by learning trip-specific perceived link costs from exogenous traveler, trip, and network-context variables and then computing the corresponding minimum perceived-cost path. The proposed framework directly uses the SP solution as the supervised object in calibration: the encoder is trained so that the path induced by the learned perceived costs matches the observed trajectory. In addition, the iMLE method employed in this study provides a surrogate gradient by repeatedly evaluating the discrete SP layer, thereby avoiding explicit differentiation of a large-dimensional equilibrium or assignment system.

## 2.4 Decision-Focused Learning and Differentiable Constrained Optimization

Many practical decision problems are solved under uncertainty because the optimization model depends on quantities that are not directly observed and therefore must be estimated from data, such as costs, demands, or rewards (Mandi et al., 2024). A common strategy is the predict-thenoptimize framework, in which a predictive model first estimates these quantities and an optimization model then computes the final decision (Bertsimas and Kallus, 2020; Mandi et al., 2024). However, better prediction accuracy does not necessarily lead to better decisions, because some estimation errors have little effect on the optimal solution while others can change the selected solution substantially even when the numerical error is small (Wilder et al., 2019; Elmachtoub and Grigas, 2022).

Decision-focused learning was introduced to address this mismatch by training predictive models according to the quality of the decisions they induce, rather than according to intermediate prediction error alone (Wilder et al., 2019; Elmachtoub and Grigas, 2022). Under this perspective, prediction and optimization are combined into a single end-to-end framework so that model parameters are updated based on downstream task performance (Wilder et al., 2019; Mandi et al., 2024). This shift is especially useful in constrained and combinatorial decision problems, where the practical objective is to obtain a high-quality decision and not merely an accurate estimate of latent model inputs (Demirović et al., 2019; Elmachtoub and Grigas, 2022; Mandi et al., 2024).

A key technical challenge, however, is that many downstream optimization problems are discrete and non-differentiable, which prevents ordinary backpropagation from being applied directly through the solver (Amos and Kolter, 2017; Vlastelica et al., 2020). Recent work on differentiable CO has addressed this issue by developing surrogate gradients, continuous relaxations, and perturbation-based methods that make it possible to train predictive models through optimization layers (Vlastelica et al., 2020; Niepert et al., 2021; Mandi et al., 2024). Among these approaches, perturbation-based methods such as iMLE are particularly appealing because they preserve the original discrete optimization problem in the forward pass while providing usable gradient estimates for end-to-end learning in the backward pass (Niepert et al., 2021). These developments provide an important methodological foundation for learning systems in which model outputs are ultimately evaluated through structured decisions rather than by prediction error alone (Mandi et al., 2024).

## 3. Methodology

## 3.1 Behavioral Assumptions

The proposed route reproduction framework is established upon two fundamental behavioral assumptions. The first concerns the derivation of subjective link-level perceptions, and the second addresses the path-selection mechanism based on those perceptions. Distinguishing these two processes clarifies how the proposed model deviates from conventional stochastic route choice formulations.

## Assumption 1 (Heterogeneous Link-Level Cost Perception):

In the proposed framework, the perceived cost of link � for traveler � is modeled as $C _ { n } ^ { a } = e ^ { y _ { n } ^ { a } }$ and $y _ { n } ^ { a } { \sim } N ( \mu _ { n } ^ { a } , \ \sigma _ { n } ^ { a } )$ , where $( \mu _ { n } ^ { a } , \sigma _ { n } ^ { a } )$ is modeled as a function of individual characteristics, trip-specific attributes, and network-level traffic states. It allows the latent cost distribution to be a lognorma and to vary across both travelers and links. This reparameterization scheme is the key departure of the proposed route choice model.

## Assumption 2 (Trip-level shortest-path finding on perceived costs):

Conditional upon the perceived link costs, the traveler is assumed to follow a deterministic costminimization objective. The path perceived cost is the linear summation of its constituent link costs. Consequently, the observed route is treated as the solution to a shortest-path optimization problem.

Collectively, these assumptions underpin the architecture of the proposed model. By bypassing probabilistic choice modeling in favor of a deterministic solver acting on latent perceived costs, the model directly mimics the intrinsic route-choosing process of travelers rather than deriving the choice probability. The subsequent sections address the mathematical complexities involved in calibrating this framework of neural-optimization pipeline.

## 3.2 Problem set-ups

Let the directed road network be represented by a graph $G = ( V , A )$ , where � and � denote the sets of nodes and directed links, respectively. We use $a \in A$ to index links and $n \in \{ 1 , \ldots , N \}$ to index trips. Each trip � is associated with an origin $o _ { n } \in V$ , a destination $d _ { n } \in V$ , a departure-time index $\pmb { x } _ { n } ^ { d e p T } \in \{ 1 , \ldots , T \}$ , and origin- and destination-specific context variables $( \pmb { x } _ { n } ^ { o } ~ \pmb { x } _ { n } ^ { d } )$

For each trip � and link �, the model assigns a personalized perceived link cost $\hat { C } _ { a , n }$ . Each trip � therefore has its own perceived link-cost vector $( \widehat { \pmb { C } } _ { n } \in \mathbb { R } ^ { | { \cal A } | } )$ , with each link cost characterized by the trip maker’s perceived travel cost $( \widehat { C } _ { a , n } ) : \widehat { \pmb { C } } _ { n } = [ \widehat { C } _ { a , n } ] _ { a \in A }$

Following the two behavioral assumptions in the previous subsection, we model route generation directly through the shortest path finding mechanism rather than computing choice probabilities.

Throughout the paper, $r _ { n }$ denotes the observed route for trip �, whereas $\hat { r } _ { n }$ denotes the route generated by the model. The learning objective is to align the generated route $\hat { r } _ { n }$ with the observed route $r _ { n }$ , which leads to decision-focused learning while the perceived link costs remain latent.

Let $\mathbf { x } _ { n }$ denote the covariate vector for trip �. Specifically, we write $\mathbf { x } _ { n } = ( x _ { n } ^ { p e r s o n } , x _ { n } ^ { t r i p } , x _ { n } ^ { n e t } )$ , where $x _ { n } ^ { p e r s o n }$ denotes traveler attributes, $\pmb { x } _ { n } ^ { t r i p }$ trip-context variables, and $\pmb { x } _ { n } ^ { n e t }$ network state information. The trip context variables $( x _ { n } ^ { t r i p } )$ ) contain spatio-temporal trip properties such as departure time $( \pmb { x } _ { n } ^ { d e p T } )$ and origin- and destination-specific context variables $( \pmb { x } _ { n } ^ { o } , \pmb { x } _ { n } ^ { d } , )$ The network state information $( x _ { n } ^ { n e t } )$ includes static and dynamic traffic states $( { \pmb x } _ { n } ^ { s t a t i c . n e t } , { \pmb x } _ { n } ^ { d y n a m i c . n e t } )$ . The encoder model maps ${ \bf x } _ { n }$ to the parameters of a perceived travel cost distribution and can be calibrated by the reparameterization scheme (Kingma and Welling, 2014). Table 1 summarizes core notations in the present study.

Table 1. Notation descriptions
<table><tr><td rowspan=1 colspan=2>Notation                                                Description</td></tr><tr><td rowspan=1 colspan=1> ${ \pmb G } = ( { \pmb V } , { \pmb A } )$ </td><td rowspan=1 colspan=1>Directed road network graph with node set V and link set A</td></tr><tr><td rowspan=1 colspan=1>a</td><td rowspan=1 colspan=1>Link index, $a \in A$ </td></tr><tr><td rowspan=1 colspan=1>n</td><td rowspan=1 colspan=1>Trip index, $n = 1 , \dots , N$ </td></tr><tr><td rowspan=1 colspan=1> $x _ { n } ^ { p e r s o n }$ </td><td rowspan=1 colspan=1>Individual-specific variables of trip n</td></tr><tr><td rowspan=1 colspan=1> $x _ { n } ^ { o } , x _ { n } ^ { d }$ </td><td rowspan=1 colspan=1>Origin- and destination-specific variables of trip n</td></tr><tr><td rowspan=1 colspan=1> $x _ { n } ^ { d e p T }$ </td><td rowspan=1 colspan=1>Departure time index of trip n</td></tr><tr><td rowspan=1 colspan=1> $x _ { n } ^ { s t a t i c . n e t } , x _ { n } ^ { d y n a m i c . n e t }$ </td><td rowspan=1 colspan=1>Static and dynamic network states of trip n</td></tr><tr><td rowspan=1 colspan=1> $x _ { n }$ </td><td rowspan=1 colspan=1>Exogenous input vector for trip $n ,$ collecting individual characteristics,trip-specific attributes, and network-level traffic states $[ { x _ { n } ^ { p e r s o n } } , { x _ { n } ^ { d e p T } } , { x _ { n } ^ { o } } , { x _ { n } ^ { d } } , { x _ { n } ^ { s t a t i c . n e t } } , { x _ { n } ^ { d y n a m i c . n e t } } ]$ </td></tr><tr><td rowspan=1 colspan=1> ${ \widehat { \pmb { C } } } _ { \pmb { n } }$ </td><td rowspan=1 colspan=1>Sampled perceived link-cost vector for trip n</td></tr><tr><td rowspan=1 colspan=1> $\pmb { \mu _ { n } } , \pmb { \Sigma _ { n } }$ </td><td rowspan=1 colspan=1>Mean vector and covariance matrix of perceived link costs for trip n</td></tr><tr><td rowspan=1 colspan=1> $\pmb { f _ { \theta } } ( \cdot )$ </td><td rowspan=1 colspan=1>Neural encoder parameterized by θ</td></tr><tr><td rowspan=1 colspan=1>ε</td><td rowspan=1 colspan=1>Perturbation noise used in perturb-and-MAP sampling</td></tr><tr><td rowspan=1 colspan=1> $\boldsymbol { r } _ { n }$ </td><td rowspan=1 colspan=1>Observed route for trip n, represented as a link-incidence vector</td></tr><tr><td rowspan=1 colspan=1> $\hat { r } _ { n }$ </td><td rowspan=1 colspan=1>Predicted route for trip n obtained from the shortest-path layer</td></tr></table>

## 3.3 Modeling Framework

## 3.3.1 Establishing model architecture

Fig. 1 illustrates the end-to-end architecture that maps covariate inputs to perceived link costs and then solves a SP problem based on the costs to derive an output path. The final output is thus the shortest path, and the difference between this path and the ground truth is minimized for training the model. Although the architecture is conceptually simple, the main difficulty lies in training the model. The CO layer to find the shortest path is included in the model pipeline but is nondifferentiable. Decision-focused learning can match the solution of a CO layer with the ground truth. The iMLE approach (Niepert et al., 2021) backpropagates the CO layer within the end-to-end model pipeline.

![](images/8565958c75a7c17670f7a3bc22cef77cfe2f61ed25becc457c68cda01305b268.jpg)  
Fig. 1 Model architecture for the proposed modelling framework

The forward pass to evaluate the proposed model has a sampling procedure as denoted in Eq (1). The sampling is to get a perceived link cost vector $( { \widehat { \pmb { C } } } _ { n } )$ . The perception model $\left[ f _ { \pmb { \theta } } ( \pmb { x } _ { n } ) \right]$ to encoded context covariates into perceived costs is approximated by a neural network and returns the mean $( \mu _ { n } \in \mathbb { R } ^ { | A | } )$ and variance $( \Sigma _ { n } \in \mathbb { R } ^ { | A | \times | A | } )$ of the Gaussian distribution. $\Sigma _ { n }$ is assumed as a diagonal matrix. To make perceived travel costs positive, the exponential function is taken to samples from the encoder. This action reflects that perceived link travel times follow the lognormal distribution.

$$
( \mu _ { n } , \Sigma _ { n } ) { = } f _ { \theta } ( x _ { n } ) , \ y _ { n } { \sim } \mathcal { N } _ { \theta } ( \mu _ { n } , \Sigma _ { n } ) , \ \widehat { \mathbf { C } } _ { n } = e ^ { y _ { n } } , \ \widehat { r } _ { n } = S P ( \widehat { \mathbf { C } } _ { n } )\tag{1}
$$

Based on this reparameterization scheme (Kingma and Welling, 2014), a perceived link-cost vector ${ \widehat { \pmb { C } } } _ { n }$ is sampled and passed to the shortest-path (SP) layer. Conditional on ${ \widehat { \pmb { C } } } _ { n }$ , the forward SP prediction is deterministic as follows

$$
\hat { r } _ { n } = S P \left( \widehat { \pmb { C } } _ { n } \right) = a r g m i n \widehat { \pmb { C } } _ { n } ^ { \top } r\tag{2}
$$

## 3.3.2 Setting up the original loss function

The original decision loss measures the discrepancy between the observed route $r _ { n }$ and the predicted route $\hat { r } _ { n }$ . In this study, the individual route-matching loss is defined as the complement of link-based intersection-over-union (IOU):

$$
\ell ( r _ { n } , \hat { r } _ { n } ) = 1 - \frac { r _ { n } ^ { \top } \hat { r } _ { n } } { 1 ^ { \top } r _ { n } + 1 ^ { \top } \hat { r } _ { n } - r _ { n } ^ { \top } \hat { r } _ { n } }\tag{3}
$$

Thus, the original decision-focused objective can be written as Eq. (4).

$$
\begin{array} { r } { L ( \pmb \theta ) = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } L _ { n } ( \pmb \theta ) , \mathrm { w h e r e } L _ { n } ( \pmb \theta ) = \mathbb { E } _ { \hat { C } _ { n } \sim \mathcal { N } ( \mu _ { n } , \Sigma _ { n } ) } [ \ell ( r _ { n } , \hat { r } _ { n } ) ] } \end{array}\tag{4}
$$

However, this objective cannot be directly optimized by ordinary backpropagation. The SP operator in Eq. (2) maps a continuous cost vector to a discrete path-incidence vector. Therefore, the loss is piecewise constant with respect to ${ \widehat { \pmb { C } } } _ { n } ,$ and its gradient is zero almost everywhere and undefined at path-switching boundaries. This non-differentiability prevents direct computation of the gradient of the loss function $\nabla _ { \widehat { \mathbf { C } } _ { n } } \ell \bigl ( r _ { n } , \mathrm { S P } ( \widehat { \pmb { C } } _ { n } ) \bigr )$

## 3.3.3 iMLE surrogate loss and gradient

Regarding the backward pass, Eq. (5) denotes the gradient of the loss function with respect to $\pmb { \theta } ,$ which is decomposed by the chain rule. Because ${ \widehat { \pmb { C } } } _ { n }$ is sampled by the reparameterization trick using the mean and standard deviation returned from the perception model $[ f _ { \pmb \theta } ( \pmb x _ { n } ) ]$ , the Jacobian, $\partial _ { \pmb { \theta } } \widehat { \pmb { C } } _ { n } ,$ is computed directly from the Jacobian, $\partial _ { \pmb { \theta } } f _ { \pmb { \theta } } ( \pmb { x } _ { n } )$ . These two Jacobians can be computed using the auto-differentiation scheme provided by any deep-learning frameworks.

$$
\nabla _ { \pmb { \theta } } L _ { n } ( \pmb { \theta } ) = \partial _ { \pmb { \theta } } \widehat { C } _ { n } ^ { \phantom { \dagger } T } \nabla _ { \widehat { C } _ { n } } L _ { n } ( \pmb { \theta } )\tag{5}
$$

On the other hand, the gradient, $\nabla _ {  { \hat { \mathbf { c } } } _ { n } } L _ { n } ( \pmb { \theta } )$ , of the loss function with respect to the perceived linkcost vector ${ \widehat { \pmb { C } } } _ { n }$ cannot be computed directly. As illustrated in Fig. 2, the CO layer maps a continuous cost vector to a discrete route. Consequently, small perturbations of ${ \widehat { \pmb { C } } } _ { n }$ usually do not change the selected route, so the loss remains unchanged over most regions of the cost domain. The loss takes the form of a piecewise constant function, and its gradient is therefore zero almost everywhere except for path-switching points. By contrast, at the path-switching boundaries, the gradient is not defined. This non-differentiability prevents the direct use of a gradient-based algorithm for minimizing the original loss. Therefore, the original loss needs to be replaced by a smooth surrogate whose gradient provides a meaningful descent direction for backpropagation through the shortest path finding layer.

![](images/45a6803ddae193e0b83d93b28f5da6289483577ad3000583fc20817933f8ed87.jpg)  
Fig. 2 Non-differentiability of the shortest-path finding layer

Niepert et al. (2021) introduced a surrogate loss ${ \mathcal { L } } _ { n }$ , that is defined for differentiation while preserving the learning signal of the original objective $( L _ { n } )$ . Prior to defining the surrogate loss, a probabilistic mass function (PMF) over routes is defined. Eq. (6) denotes a softmax PMF function that belongs to exponential family. Here, the temperature � is a hyperparameter to control the degree of sharpness.

$$
\begin{array} { r } { P _ { \widehat { c } _ { n } } ( r _ { n } ) = \frac { e ^ { - \widehat { c } _ { n } ^ { \ T } r _ { n } / \tau } } { \sum _ { r \in \mathcal { P } _ { n } } e ^ { - \widehat { c } _ { n } ^ { \ T } r / \tau } } } \end{array}\tag{6}
$$

Here, $\mathcal { P } _ { n }$ denotes the set of feasible paths connecting the origin and destination of trip �. Although this distribution is useful for deriving the iMLE surrogate gradient, it is not evaluated explicitly because enumerating all feasible paths in $\mathcal { P } _ { n }$ is computationally intractable on a large network.

Eq. (7) denotes the gradient of the log-likelihood, ln $P _ { \widehat { \pmb { c } } _ { n } } ( r _ { n } )$ , with respect to the link cost, ${ \widehat { \pmb { C } } } _ { n }$ . The simple derivation of the gradient stems from the exponential family assumption, this gradient will be used later when computing the gradient of the surrogate loss $( { \mathcal { L } } _ { n } )$

$$
\nabla _ { \widehat { c } _ { n } } \ln P _ { \widehat { c } _ { n } } ( r _ { n } ) = - r _ { n } + \mathbb { E } _ { r _ { n } \sim P _ { \widehat { c } _ { n } } ( r _ { n } ) } [ r _ { n } ]\tag{7}
$$

The surrogate loss, $\mathcal { L } _ { n } ,$ is defined by averaging the logarithm of the PMF over a chosen target distribution. The PMF of the target distribution is also chosen to be a softmax function, with a different set of link costs $( \widehat { \pmb { C } } _ { n } ^ { \prime } )$ . Eq. (8) defines the surrogate loss to be the negative expectation of log-likelihood over the target distribution $[ = P _ { \widehat { \mathbf { C } } _ { n } ^ { \prime } } ( r _ { n } ) ]$ . Minimizing the surrogate loss is equivalent to minimizing the KL-divergence between the target and incumbent PMFs, which leads the incumbent PMF as closely to the target PMF as possible.

$$
\mathcal { L } _ { n } ( \widehat { \mathbf { C } } _ { n } , \widehat { \mathbf { C } } _ { n } ^ { \prime } ) = - \mathbb { E } _ { r _ { n } \sim P _ { \widehat { c } _ { n } ^ { \prime } } ( r _ { n } ) } [ \ln P _ { \widehat { \mathbf { C } } _ { n } } ( r _ { n } ) ]\tag{8}
$$

The gradient, $\nabla _ { \widehat { \pmb { c } } _ { n } } \mathcal { L } _ { n } \big ( \widehat { \pmb { c } } _ { n } , \widehat { \pmb { c } } _ { n } ^ { \prime } \big )$ , leads to Eq. (9) by simple algebra based on Eq. (7), which will substitute for the gradient, $\nabla _ {  { \hat { \mathbf { c } } } _ { n } } L _ { n } ( \pmb { \theta } )$ , of the original loss function.

$$
\nabla _ { \widehat { C } _ { n } } \mathcal { L } _ { n } \left( \widehat { C } _ { n } , \widehat { C } _ { n } ^ { \prime } \right) = \mathbb { E } _ { r _ { n } \sim P _ { \widehat { C } _ { n } ^ { \prime } } ( r _ { n } ) } [ r _ { n } ] - \mathbb { E } _ { r _ { n } \sim P _ { \widehat { C } _ { n } } ( r _ { n } ) } [ r _ { n } ] = \mu \left( \widehat { C } _ { n } ^ { \prime } \right) - \mu \left( \widehat { C } _ { n } \right)\tag{9}
$$

Two expectations over all possible routes in Eq. (9) cannot be computed at this stage. Eq. (10) shows that the expectation of routes for a given ${ \widehat { \pmb { C } } } _ { n }$ can be approximated by the average of the MAP state with perturbation in ${ \widehat { \pmb { C } } } _ { n }$ . More concretely, if path-level Gumbel perturbations were available, perturband-MAP sampling exactly reproduces the exponential family PMF. Since all feasible paths cannot be enumerated, the implementation in Eq. (10) uses structured link-level perturbations and repeated shortest-path evaluations to approximate the required moments. The perturbation-induced distribution is therefore used as a computational device for gradient estimation. Computing the ��� state, equivalent to SP finding operation, is typically much less expensive than computing the expectation [μ(∙)] over all possible routes which is almost impossible in practice.

$$
\nabla _ { \widehat { C } _ { n } } \mathcal { L } _ { n } \left( \widehat { C } _ { n } , \widehat { C } _ { n } ^ { \prime } \right) = \mu \left( \widehat { C } _ { n } ^ { \prime } \right) - \mu \left( \widehat { C } _ { n } \right) \approx \mathbb { E } _ { \epsilon \sim p \left( \epsilon \right) } \left[ M A P \left( \widehat { C } _ { n } ^ { \prime } + \epsilon \right) - M A P ( \widehat { C } _ { n } + \epsilon ) \right]\tag{10}
$$

The gradient, $\nabla _ { \widehat { \mathbf { C } } _ { n } } \mathcal { L } _ { n } \big ( \widehat { \mathbf { C } } _ { n } , \widehat { \mathbf { C } } _ { n } ^ { \prime } \big )$ , is approximated once again by the simulated version, $\widehat { \nabla } _ { \widehat { \pmb { c } } _ { n } } \mathcal { L } _ { n } \big ( \widehat { \pmb { c } } _ { n } , \widehat { \pmb { c } } _ { n } ^ { \prime } \big )$ Eq. (11) denotes the simulated gradient with random draws $\pmb { \epsilon } _ { i } ^ { \prime } \mathsf { S } .$ . Note here that the same random draw $( \epsilon _ { i } )$ should be used for averaging both ��� states. In practice, the sample size (�) can be reduced to 1 if the batch size is sufficiently large. For this case, each element in the ��� difference vector belongs to one of -1, 0 , and +1.

$$
\begin{array} { r } { \widehat { \nabla } _ { \widehat { C } _ { n } } \mathcal { L } _ { n } \big ( \widehat { C } _ { n } , \widehat { C } _ { n } ^ { \prime } \big ) = \frac { 1 } { s } \sum _ { i = 1 } ^ { s } \bigl [ M A P \big ( \widehat { C } _ { n } ^ { \prime } + \epsilon _ { i } \big ) - M A P ( \widehat { C } _ { n } + \epsilon _ { i } ) \bigr ] , \mathrm { ~ w i t h ~ } \epsilon _ { i } \sim p ( \epsilon ) } \end{array}\tag{11}
$$

The gradient of the surrogate loss leads to the difference between two ��� route states when the incumbent and target link cost vectors are given. The next step to compute the gradient is to determine the target link travel cost vector, $\widehat { \pmb { C } } _ { n } ^ { \prime }$ , which is equivalent to choosing a target distribution. Domke (2010) described that a target distribution can be rooted in perturbation-based implicit differentiation (PID). Eq. (12) denotes the finite difference $[ \lambda \nabla _ { \boldsymbol { r _ { n } } } \ell ( \boldsymbol { r _ { n } } , \boldsymbol { \hat { r } _ { n } } ) ]$ between two link travel cost vectors.

$$
\widehat { \pmb { C } } _ { n } ^ { \prime } = \widehat { \pmb { C } } _ { n } - \lambda \nabla _ { \hat { r } _ { n } } \ell ( { r } _ { n } , \hat { { r } } _ { n } )\tag{12}
$$

Here, $\lambda > 0$ is a hyperparameter to control the perturbation intensity. The reciprocal $1 / \lambda$ incorporated into the global learning rate, �, for the gradient descent algorithm (Niepert et al., 2021). Eq. (13) shows that the stochastic gradient algorithm to update the model parameter, �, at every iteration step. Here, � is batch size.

$$
\begin{array} { r } { \pmb { \theta } ^ { \prime } = \pmb { \theta } - \frac { \eta } { B } \sum _ { n = 1 } ^ { B } \partial _ { \pmb { \theta } } \widehat { \pmb { C } } _ { n } ^ { \prime } \widehat { \nabla } _ { \widehat { C } _ { n } } \mathcal { L } _ { n } \left( \widehat { \pmb { C } } _ { n } , \widehat { \pmb { C } } _ { n } - \lambda \nabla _ { \widehat { r } _ { n } } \ell ( r _ { n } , \widehat { r } _ { n } ) \right) } \end{array}\tag{13}
$$

The last step to obtain the gradient is to determine a probabilistic distribution for perturbations (�). Niepert et al. (2021) proved that � should be drawn from the sum of Gamma (SoG) distribution, so that the Gumbel-Max trick (Papandreou and Yuille, 2010) can reproduce the softmax PMF shown in Eq. (6). According to Theorem 1 in Niepert et al. (2021), perturbing each element of link travel cost vector, ${ \widehat { \pmb { C } } } _ { n } ,$ by a random draw $\epsilon _ { i } { \sim } S o G ( \kappa , \tau , g )$ leads to perturbing the path cost, $\widehat { \pmb { C } } _ { n } ^ { \phantom { \dagger } T } r _ { n }$ , with a random draw $\zeta ( r _ { n } ) { \sim } G u m b e l ( 0 , \tau )$ , which results in the Gumbel-max trick (Jang et al., 2017). A $S o G ( \kappa , \tau , g )$ distribution has three parameters. � denotes the number of non-zero entries in a route vector to determine both parameters $( 1 / \kappa , \kappa / i )$ of ����� PDFs in ��� distribution, � is a parameter matching the temperature in the exponential family PMF shown in Eq. (6), and $_ g$ denotes how many ����� PDFs are summed in a SoG distribution. Eq. (14) denotes the PDF of ���(�, �, �), where � and $_ g$ should be a positive integer, and the adopted parameter values are provided in Appendix A.

$$
\begin{array} { r } { S o G ( \kappa , \tau , g ) = \frac { \tau } { \kappa } \big [ \sum _ { i = 1 } ^ { g } [ G a m m a ( 1 / \kappa , \kappa / i ) ] - \log g \big ] } \end{array}\tag{14}
$$

The iMLE method derives the approximate gradient of the original loss function with respect to perceived link travel costs $( { \widehat { \pmb { C } } } _ { n } )$ . In summary, the method smooths the ${ \mathsf { S P } }$ layer and derives the approximate gradient to find a local optimum, preserving learning directions of the original loss function.

Although differentiation through the discrete optimization layer is successful, the proposed framework still faces a more fundamental problem. The adopted supervision to match the estimated and observed routes cannot derive a unique set of perceived link travel costs. Intuitively, the shortest path found on a set of link travel costs does not vary by doubling all link travel costs in the set. This implies that an infinite number of perceived travel cost sets that reproduce a single ground truth route can be found. This scaling issue has not been discussed in previous iMLE studies. Niepert et al. (2021) focused only on differentiating the discrete optimization not mentioning the issue. On the other hand, Mandi et al. (2024) averted the problem by assuming that the ground truth latent variables (=perceived link cost in the present study) could be observed. Because the central objective of the proposed model is to recover behaviorally meaningful perceived link costs, we add a regularization term to stabilize estimation and reduce the scaling challenge. Section 3.3.4 describes how regularizing the distribution of travel costs releases the problem and modifies the loss function to regularize perceived link travel costs.

## 3.3.4 Regularization for cost-scale anchoring

To mitigate the cost-scale ambiguity, we regularize the output returned from the perception model for cost-scale anchoring. The baseline distribution is obtained from observed link travel times. After taking the logarithm to the observed link travel times, the mean $( \widetilde { \mu } _ { a } )$ and the standard deviation $( \tilde { \sigma } _ { a } )$ are computed. The KL-divergence between two Gaussian distributions leads to the closed form that can be easily handled while training the model.

Eq. (15) denotes the reinforced loss function [�<sup>̃</sup>(�)] with the KL-divergence $\left[ D _ { K L } ( \cdot ) \right]$ , where $\xi$ is a hyperparameter that weighs the regularization with respect to the original loss.

$$
\tilde { L } ( \pmb \theta ) = L ( \pmb \theta ) + \xi D _ { K L } ( \mathcal { N } ( \pmb \mu _ { n } , \Sigma _ { n } ) | | \mathcal { N } ( \pmb \widetilde \pi , \widetilde \Sigma ) )\tag{15}
$$

where $\widetilde \pmb { \Sigma } = \mathrm { D i a g } ( \widetilde \pmb { \sigma } ^ { 2 } )$ is the diagonal matrix containing the observed standard-deviation vector �̃. Here, $\left( \mu _ { n } , \Sigma _ { n } \right)$ is the mean and variance for reparameterization, which are returned from the perception function $[ f _ { \pmb \theta } ( \pmb x _ { n } ) ]$ approximated by a neural net for embedding input features to the perceived travel cost. On the other hand, �̃ and �<sup>̃</sup> should be given in advance so that minimizing the KL-divergence renders the perceived link travel cost not to stray away from them. The derivative of the KL-divergence with respect to � can be analytically computed.

The overall training procedure, combining perceived-cost sampling, perturb-and-MAP inference, iMLE-based surrogate gradient estimation, and KL regularization, is summarized in Algorithm 1. The values applied to the hyperparameters in the algorithm are listed in Appendix A.

Algorithm 1. Training algorithm   
Input: Training data $\mathcal { D } = \{ ( \mathbf { x } _ { n } , r _ { n } ) \} _ { n = 1 } ^ { N }$ ; observed link-wise data $( \tilde { \mu } , \tilde { \Sigma } )$ ; hyperparameters   
�(perturbation scale), �(weight for KL-divergence), � (perturbation sample size), �(Learning rate),   
and $\kappa , \tau ,$ and � (SoG parameters) (see Appendix ${ \mathsf { A } } ) .$   
Repeat until convergence   
Sample a mini batch ${ \mathcal { B } } \subset { \mathcal { D } } .$   
For each trip $n \in { \mathcal { B } } :$   
Obtain the latent perceived-cost distribution parameters:   
$( \mu _ { n } , \Sigma _ { n } ) = f _ { \theta } ( \mathbf { x } _ { n } )$   
Sample a perceived link-cost vector:   
$\begin{array} { r } { \mathbf { y } _ { n } \sim \mathcal { N } ( \mu _ { n } , \Sigma _ { n } ) , \widehat { \mathbf { C } } _ { n } = e ^ { y _ { n } } } \end{array}$   
Draw perturbations:   
$\epsilon _ { i } \sim S o G ( \kappa , \tau , g ) , i = 1 , \ldots , s .$   
Generate the predicted route through the shortest path / MAP layer:   
$\hat { r } _ { n } = \mathrm { M A P } ( \widehat { \pmb { C } } _ { n } + \epsilon _ { 1 } )$   
Compute the route-matching loss:   
$\ell ( r _ { n } , \hat { r } _ { n } )$   
Construct the target cost vector using the finite-difference perturbation:   
$\widehat { \pmb { C } } _ { n } ^ { \prime } = \widehat { \pmb { C } } _ { n } - \lambda \nabla _ { \hat { r } _ { n } } \ell ( { r } _ { n } , \hat { { r } } _ { n } )$   
Estimate the surrogate gradient with respect to ${ \hat { C } } _ { n } \mathrm { : }$   
$\widehat { \nabla } _ { \widehat { C } _ { n } } \mathcal { L } = \frac { 1 } { s } \sum _ { i = 1 } ^ { s } \bigl [ \mathrm { M A P } ( \widehat { \mathbf { C } } _ { n } ^ { \prime } + \epsilon _ { i } ) - \mathrm { M A P } ( \widehat { \mathbf { C } } _ { n } + \epsilon _ { i } ) \bigr ]$   
Compute the KL regularization term:   
$\mathcal { R } _ { n } = D _ { K L } \left( \mathcal { N } ( \mu _ { n } , \Sigma _ { n } ) \parallel \mathcal { N } ( \tilde { \mu } , \tilde { \Sigma } ) \right)$   
end for   
Compute the mini-batch gradient   
${ \bf g } _ { \theta } = \frac { 1 } { \mid \mathcal { B } \mid } \sum _ { n \in \mathcal { B } } \left[ ( \partial _ { \theta } \hat { \bf C } _ { n } ) ^ { \top } \hat { \nabla } _ { \hat { C } _ { n } } \mathcal { L } + \xi \nabla _ { \theta } R _ { n } \right] .$   
Update parameters   
$\pmb { \theta } ^ { \prime }  \pmb { \theta } - \eta \mathbf { g } _ { \theta } .$   
Return �

## 3.4 Architecture of the perception model

Fig. 3 illustrates the architectural design of the perception model, $f _ { \theta } ( \mathbf { x } _ { n } )$ , which lies in the proposed pipeline, to embed multi-dimensional covariate data, $\mathbf { x } _ { n } ,$ into the latent distribution, specifically the mean $\left( \mu _ { n } \right)$ and variance $( \Sigma _ { n } )$ of $l n \hat { \mathbf { C } } _ { n } .$ . The model takes the form of a hierarchical neural network comprising four distinct components. The first three sub-networks function as parallel encoders designed to extract high-level representations from personal information, trip context, and network state, respectively. For this multi-stream encoding, the input covariate vector, $\mathbf { x } _ { n } ,$ is partitioned into subsets $( { x _ { n } ^ { p e r s o n } } , { x _ { n } ^ { t r i p } } , { x _ { n } ^ { n e t } } )$ tailored for each respective sub-model. The final sub-network maps these three output representations into the parameters of the perceived travel cost distribution. Detailed architecture of the four sub-networks can be found in Appendix B. Furthermore, ablation tests are conducted in Section 5.2, removing each of sub-networks.

![](images/ba5a45018ae40d01c099b4a1e6dcc23f2a6b3ee28071a4e13447be9a178ed178.jpg)  
Fig. 3 Architecture of perception model $\left[ f _ { \theta } ( \mathbf { x } _ { n } ) \right]$

## 3.4.1 Personal encoder

The personal encoder maps a vector of individual characteristics $( x _ { n } ^ { p e r s o n } )$ into latent representations $( \pmb { h } _ { n } ^ { p e r s o n } )$ , which are subsequently used to derive the distribution of personalized travel costs. This process is parameterized by a feed-forward neural network as follows:

$$
\pmb { h } _ { n } ^ { p e r s o n } = g _ { \theta _ { 1 } } \big ( \pmb { x } _ { n } ^ { p e r s o n } \big ) .
$$

where $\theta _ { 1 }$ denotes the set of trainable parameters within the sub-network. The model's predictive power is maximized by incorporating a comprehensive suite of potential covariates, ranging from socio-demographic attributes such as gender, age, and income to behavioral and attitudinal variables. Furthermore, the inclusion of psychometric propensities, typically collected through stated-preference surveys, can further enhance the model's ability to capture heterogeneous decision-making results. The significance of such personal variables in uncovering the latent motives behind transportation-related choices has been extensively documented in previous studies (Kim et $\mathsf { a l . } _ { \mathsf { \Pi } }$ 2014; Lee et al., 2012; Park et al., 2011; Sohn and Yun, 2009).

In the absence of explicit socio-demographic data, as the current study relies solely on vehicle trajectory data, we focus on capturing latent driving habits implicitly embedded within individual trajectories. It is hypothesized that a driver’s speed profile serves as a proxy for their unique driving habit. For instance, an aggressive driver typically exhibits frequent and abrupt speed changes, whereas a more cautious driver tends to maintain a stable and modest speed throughout the journey. To reflect this, the proposed personal encoder $[ g _ { \theta _ { 1 } } ( \cdot ) ]$ utilizes the mean and variance of jerks, accelerations (decelerations), and longitudinal and angular speeds observed over a driver’s past trajectory as its primary input features. More concretely, for each trip, driving-habit features are computed only from trajectories of the same driver collected before the departure date of the trip.

This methodological choice is strongly supported by prior literature demonstrating that idiosyncratic driver behaviors can be effectively identified from granular vehicle trajectory data (Hallac et al., 2016; Fung et al., 2017). In particular, we adopt the feature construction strategy proposed by Chowdhury et al. (2018), who reported a driver identification accuracy of up to 82.3% using similar kinematic variables. By leveraging these high-fidelity proxies, the model can differentiate between travelers and account for individual-specific perception patterns even without access to traditional demographic information.

## 3.4.2 Trip context encoder

The perceived travel cost is further assumed to be influenced by explicit trip-related attributes, $\pmb { x } _ { n } ^ { t r i p } = [ \pmb { x } _ { n } ^ { d e p T } , \pmb { x } _ { n } ^ { o } , \pmb { x } _ { n } ^ { d } ]$ . The departure time, ${ \pmb x } _ { n } ^ { d e p T }$ , often serves as a proxy for trip purpose; for instance, trips during morning and evening peak hours typically correspond to commuting and returning home, respectively. Such temporal variations represent a unique variable that significantly affects route choice behavior. In addition, the spatial distribution of the origin and destination fundamentally shapes the traveler's choice set. To vectorize these spatial inputs, we simply add 2-D coordinates into the origin- and destination-specific variables $( x _ { n } ^ { o }$ and $\pmb { x } _ { n } ^ { d } )$ . Empirical testing verified that this simple scheme outperforms the rigorous 2-D positional encoding.

To supplement these purely locational features, we further augment $\pmb { x } _ { n } ^ { o }$ and $x _ { n } ^ { d }$ by concatenating land-use variables. Specifically, the model incorporates building floor areas, categorized by landuses, within a 500-meter radius of both the origin and destination. This provides a socio-economic context for a trip's endpoints, which can reflect the nature of the activities being pursued.

The model architecture introduces an additional mechanism to enhance embedding performance, the latent representation of personal attributes is fed into the trip context encoder. Consequently, the latent representation of the trip context $( { \pmb h } _ { n } ^ { t r i p } )$ becomes a joint function of trip-specific variables $( x _ { n } ^ { t r i p } )$ ) and latent variables $( \pmb { h } _ { n } ^ { p e r s o n } )$ for individual driving habits. The trip context encoder is parameterized by a feed-forward neural network as follows:

$$
\pmb { h } _ { n } ^ { t r i p } = g _ { \theta _ { 2 } } ( \pmb { x } _ { n } ^ { o } , \pmb { x } _ { n } ^ { d } , \pmb { x } _ { n } ^ { d e p T } , \pmb { h } _ { n } ^ { p e r s o n } )
$$

where $\theta _ { 2 }$ represents the trainable parameters of the network. This hierarchical structure allows the model to capture how the same trip context might be perceived differently depending on the individual's past driving habit.

## 3.4.3 Network state encoder

The network-wide traffic state, $\pmb { x } _ { n } ^ { n e t } = \left[ \pmb { x } _ { n } ^ { s t a t i c _ { n e t } } , \pmb { x } _ { n } ^ { d y n a m i c _ { n e t } } \right] .$ , is a fundamental determinant of a driver’s route choice behavior. This variable is categorized into static and dynamic states to capture both the structural and operational characteristics of the network. The static state, $\pmb { x } _ { n } ^ { s t a t i c \_ n e t }$ , denotes link-level geometric information, including link length, number of lanes, functional road classification, capacity, speed limits, etc. In contrast, the dynamic state, $\pmb { x } _ { n } ^ { d y n a m i c \_ n e t }$ , encompasses time-varying operational metrics such as the mean and variance of observed travel times, which are available in the specific time of day for each traveler’s departure time. Including the dynamic state as input is rational since every commercial navigation service provides such information on a real time basis.

To effectively capture the spatio-temporal propagation of traffic within the complex urban grid, the network state encoder is parameterized by a graph convolutional network (GCN):

$$
\pmb { h } _ { n } ^ { n e t } = g _ { \theta _ { 3 } } ( \pmb { x } _ { n } ^ { s t a t i c . n e t } , \pmb { x } _ { n } ^ { d y n a m i c . n e t } )
$$

where $\theta _ { 3 }$ represents the trainable parameters of graph-based architecture. GCNs have been widely adopted to model large-scale urban road networks, proving highly effective for traffic forecasting by accounting for the underlying topology (Liu et $\mathsf { a l . }$ , 2024; Bao et al., 2023; Yu et $\mathsf { a l . }$ , 2020). In Section 5.2, the contribution of adopting GCN is evaluated via ablation test.

In this framework, the network state encoder serves to inform the model of the objective traffic states. This information subsequently becomes baseline from which an individual driver’s subjective perception deviates.

## 3.4.4 The final encoder

The three latent representations are concatenated to form the input $[ { \pmb h } _ { n } ^ { p e r s o n } , { \pmb h } _ { n } ^ { t r i p } , { \pmb h } _ { n } ^ { n e t } ]$ for the final integration encoder, $g _ { \theta _ { 4 } } ( \cdot )$ . This encoder is parameterized by an additional feed-forward neural network designed to map the consolidated features into the parameters of the perceived travel cost distribution:

$$
( \mu _ { n } , \Sigma _ { n } ) = g _ { \theta _ { 4 } } ( { \pmb h } _ { n } ^ { p e r s o n } , { \pmb h } _ { n } ^ { t r i p } , { \pmb h } _ { n } ^ { n e t } )
$$

where $\theta _ { 4 }$ represents the trainable parameters of the final encoder. The derived mean $\left( \mu _ { n } \right)$ and variance $\left( \Sigma _ { n } \right)$ are subsequently utilized to sample the perceived travel costs, ${ \hat { \mathbf { C } } } _ { n }$ through the reparameterization scheme (Kingma and Welling, 2014). This trick is critical as it ensures the model pipeline remains differentiable through sampling, thereby enabling the backpropagation of gradients of the loss function during end-to-end training.

Furthermore, the output of the final encoder plays a vital role in regularizing the sampled link travel costs. By ensuring that the perceived travel costs do not deviate excessively from the empirical distribution of observed travel costs, the model maintains the behavioral rationality in interpreting the model results. This regularization mechanism releases the scaling issue of perceived link costs, effectively capturing individual subjective biases from objective traffic states.

## 4. Experimental Set-ups

## 4.1 Testbed and Data Preparation

![](images/ae4c025ff97ee4c82fba2f19063213d6157c1551f785db6263bb41f66f8c7daa.jpg)  
Fig. 4 Road network for testbed in Seoul, Korea

The empirical analysis is conducted on the road network of three major administrative districts in Seoul: Seocho-gu, Gangnam-gu, and Songpa-gu (see Fig. 4). The underlying graph $G = ( V , A )$ , where $\vert V \vert = 7 , 7 2 3$ nodes and $| A | = 1 4 , 7 6 9$ directed links. The vehicle trajectory data were provided by a commercial navigation service provider. These trajectories consist of GPS-based position sequences captured from in-vehicle units, which record coordinates from engine startup to shutdown. To transform the raw GPS points into a format suitable for route choice analysis, a map-matching algorithm was employed to snap the trajectories onto the directed links of the road network. To ensure spatial consistency, the dataset was filtered to include only trips where both the origin and destination reside within the study area. The resulting dataset contains 131,792 unique trips. Fig. 5(a) illustrates the trip length distribution, which exhibits a characteristic right-skewed profile. The distribution is characterized by a mode of 2.5 km and a mean of 4.54 km, reflecting the prevalence of relatively short-to-medium distance urban trips within the boundary of study area. Fig. 5(b) illustrates the travel time distribution that takes a similar form with the trip length distribution. The mode is 11 min. and the mean is 18.3 min.

![](images/8085d393194bb83e7cd0de260b0848ff9cf51e71ec5de17496faf28214675eb8.jpg)

![](images/4ed7666f1c5c53b22010d37f8d00986af083b8bd9dcc5e4d2474224a5908de79.jpg)  
Fig. 5 Distributions of trip length and travel times in the study area

A critical consideration in modeling real-world route choice is the influence of external information on driver behavior. While it is inherently difficult to discern the extent to which any specific observed trajectory was guided by real-time navigation services, drivers are generally presumed to have access to instantaneous traffic data. Consequently, this study adopts the operational assumption that drivers are informed of the ground-truth link travel times at their respective departure times. These observed travel times are integrated into the proposed model as primary inputs, serving a role analogous to the mean travel times employed in conventional discrete choice models. Furthermore, while it is practically impossible for a driver to maintain a perception of travel time on every link within a large-scale urban testbed, our empirical analysis suggests that drivers develop a stable subjective perception of travel times, particularly within the sub-networks encompassing their origin and destination. Detailed evidence supporting this localized perception mechanism is further discussed in Section 5.3.

A driver’s speed profile serves as a proxy for individual-specific characteristics. The mean and variance of jerks, accelerations (decelerations), and longitudinal and angular speeds are computed over a trip maker’s past trajectory. In this study, single-day trajectories are used for training and testing the proposed modelling framework. Since multi-day trajectories are available for each driver, trajectories observed before the target date to collect data for modelling.

In addition to the vehicle trajectory data, a comprehensive set of covariates was prepared to calibrate and validate the proposed model. For the trip-related attributes, land-use intensity at the trip endpoints was quantified by collecting building floor areas categorized into residential, commercial/office, and recreational uses. These data were aggregated within a 500-meter radius of each origin and destination. Utilizing the GIS-based buffering technique, we processed the spatia data provided by the ministry of land, infrastructure and transport (MOLIT) in SHAPE format to calculate the total floor area for each land-use category within the catchment areas.

The network state data were derived from two primary sources. The static network state, representing the physical infrastructure, was obtained from the official road network database released by the Korea transport database (KTDB). This includes fixed attributes such as link capacity and functional classification. The dynamic network state, capturing real-time traffic conditions, was generated by aggregating individual vehicle trajectories. This synthesized traffic information, encompassing time-varying speeds and travel times, is considered functionally equivalent to the real-time traffic information provided by commercial navigation service providers. These covariate datasets provide the spatio-temporal and socio-economic context influencing each traveler’s travel cost perception.

The complete dataset was randomly partitioned into a training set (80%) and a testing set (20%). We constructed a specialized subset to rigorously test how effectively the model reproduces observed paths in scenarios where multiple viable alternative routes exist. This subset focuses on OD pairs characterized by significant route variation, filtered according to a strict three-fold criterion. Specifically, we selected OD pairs that contained at least five observed trips and utilized at least three unique paths. To further ensure that the sampled behavior reflects a high degree of individualistic choice rather than a concentrated flow on a single dominant route, we required the ratio of unique observed routes to the total number of trips to exceed a threshold of 0.9. This process yielded a refined test dataset of 3,668 trips across 546 OD pairs, providing a stringent environment for evaluating the model's capacity to diversify route choice predictions and internalize personalized preferences.

## 4.2 Baseline Models for Comparative Analysis

We compare the proposed framework with five reference models, which are categorized based on two dimensions: formulations and modelling groundwork. Path size logit (PSL) and deep neural net-PSL (DNN-PSL) are classified into path-based models, while recursive logit (ReL) model, adversarial inverse reinforcement learning (AIRL) model, and perturbed utility route choice (PURC) model are link-based. Table 2 lists the reference models according to the corresponding taxonomy. The specification of each model is described in Appendix C to support readers’ reproduction attempts.

Table 2. Reference models by route choice formulation and modeling dimensions
<table><tr><td colspan="2">Reference model class</td><td colspan="2">Formulation dimension</td></tr><tr><td colspan="2"></td><td>Path-based model</td><td>Link-based model</td></tr><tr><td rowspan="4">Modeling dimension</td><td>Discrete choice model</td><td>PSL (Ben-Akiva and Bierlaire, 1999)</td><td>ReL (Fosgerau et al., 2013)</td></tr><tr><td>Deep learning model</td><td>DNN-PSL</td><td>AIRL</td></tr><tr><td></td><td>(Marra and Corman, 2021)</td><td>(Zhao and Liang, 2023)</td></tr><tr><td>Static optimization model</td><td></td><td>PURC</td></tr></table>

PSL is a path-based discrete choice model defined over a finite candidate path set and includes a path-size correction term to account for overlap among alternative routes. DNN-PSL extends PSL by replacing the linear utility specification with a neural-network-based representation, while retaining the same candidate-set-based probabilistic structure. For these path-based reference models, candidate routes were generated for each OD pair using the k-shortest path algorithm with k=10. The link utility is modeled by explanatory variables, and the path utility is obtained by aggregating link-level variables, including travel time, link length, number of lanes, link type, and road class.

The ReL model without path enumeration reformulates route choice as a sequence of link transitions, thereby circumventing the computational burden and potential biases associated with path set generation. It directly utilizes link-level attributes, such as travel time, length, and road class, to estimate transition probabilities. Complementing these discrete choice frameworks, the AIRL-based model treats route choice as a sequential decision-making process within a Markov decision process (MDP) environment. This approach learns context-dependent reward and policy functions directly from observed trajectories. In our implementation of AIRL, the state variables are defined by linklevel attributes, while departure time and the remaining distance to the destination are incorporated as context variables to capture the dynamic and goal-directed nature of the observed choices.

PURC model is the latest model free from enumerating alternative paths. The model belongs to static inverse optimization. Link travel time, link length, number of lanes, and road class are chosen as input for PURC. For each OD pair, PURC retrieve link flows for given demand. The individual trajectory is used to derive the link flows in the present study.

The proposed framework is to reproduce an individual route, but baseline models are all choice models or inverse optimization models. To evaluate the path reproduction performance, output of baseline models need to be go through an additional post process.

Unlike the proposed framework, PSL and DNN-PSL generate a probability distribution over a finite candidate set. To ensure a consistent and robust comparison with the observed ground-truth trajectories, we adopt an expectation-based averaging scheme rather than simple stochastic sampling. For each trip, the performance metric is computed between the ground-truth route and every individual route within the candidate set (comprising the top-10 shortest paths). These scores are then aggregated into a representative expected value, weighted by the model-estimated choice probabilities. It is important to note that the candidate set may not always encompass the actual observed route; therefore, the resulting metrics should be interpreted as candidate-set-level performance measures rather than direct, deterministic route-level prediction accuracies.

To ensure a consistent and robust comparison with link-based models, a sampling-based bestmatching scheme is devised. For each OD pair, thirty complete routes are generated through repeated stochastic rollouts, and each ground-truth route is matched with the sampled route that yields the highest overlap level. This procedure is deliberately favorable to stochastic link-based baselines because it gives credit when their sampled route set contains a route close to the observed trajectory. Therefore, the resulting scores should be interpreted as upper-bound performance.

Since PURC model cannot return deterministic path output, a conceptual comparison is necessary based on sampling. That is, the estimated optimal link-flow vector is first transformed into transition probabilities by normalizing positive outgoing flows at each node. Thirty complete routes are then sampled through the same random-walk procedure, and the most frequently generated route is selected as the PURC’s path prediction. This modal sampled route is compared with the groundtruth route using the same evaluation metrics. This treatment converts PURC’s link-flow output into a route-level prediction, allowing it to be compared using the same route-reproduction metrics as the proposed framework.

## 4.3 Evaluation Metrics

To benchmark route reproduction performance, we evaluate the similarity between the predicted and observed routes using three route-level measures. IoU is used as the primary metric, and trip length by links accuracy (TLLA) and edit distance (ED) are reported as supplementary metrics. TLLA supplements IOU by considering the length of overlapping links (Qiu et al., 2024), and ED by reflecting the order-sensitivity in route dissimilarity (Choi et al., 2021; Zhao and Liang, 2023; Qiu et al., 2024). Consequently, these measures capture (i) overlap in traversed road segments, (ii) similarity in total traveled distance, and (iii) order-aware similarity in the link sequence. Unless stated otherwise, the measures are reported as aggregate scores averaged over trips in the test set. Higher values indicate better performance for IoU and TLLA, whereas lower values indicate better performance for ED.

## Intersection-over-Union on links (IoU)

IoU measures set-level overlap between the predicted and observed link sets. Using the multi-hot link incidence vectors $\hat { r } _ { n } , r _ { n } \in \{ 0 , 1 \} ^ { | A | }$ , IoU is defined as

$$
\mathrm { I o U } ( { \widehat { r } } _ { n } , r _ { n } ) = { \frac { \mid { \widehat { r } } _ { n } \cap r _ { n } \mid } { \mid { \widehat { r } } _ { n } \cup r _ { n } \mid } } = { \frac { { \widehat { r } } _ { n } ^ { \top } r _ { n } } { \parallel { \widehat { r } } _ { n } \parallel _ { 1 } + \parallel r _ { n } \parallel _ { 1 } - { \widehat { r } } _ { n } ^ { \top } r _ { n } } } \ .
$$

IoU ranges from 0 to 1, where 1 indicates perfect overlap in observed links. Let $r _ { n }$ and $\hat { r } _ { n }$ denote the observed and predicted route, respectively.

## Trip Length by Links Accuracy (TLLA)

TLLA is chosen to measure the proportion of the observed route length that is correctly covered by the predicted route. Let �(�) denote the length of link �. TLLA for a trip � is defined as

$$
\mathrm { T L L A } _ { n } = \frac { \sum _ { e \in \hat { r } _ { n } \cap r _ { n } } l ( e ) } { \sum _ { e \in r _ { n } } l ( e ) } .
$$

A higher value indicates better route reproduction.

## Edit Distance (ED) on link sequences

Normalized edit distance is selected as a supplementing performance index to measure the ordersensitive dissimilarity between the observed and predicted link sequences. Let the observed and predicted routes for trip � be $\pi _ { n } = ( a _ { 1 } , \ldots , a _ { L _ { n } } )$ and $\hat { \pi } _ { n } = ( \hat { a } _ { 1 } , \dots , \hat { a } _ { \hat { L } _ { n } } )$ , respectively. The Levenshtein edit distance $d _ { \mathrm { E D } } ( \hat { \pi } _ { n } , \pi _ { n } )$ is first necessary to compute this index. $d _ { \mathrm { E D } } ( \hat { \pi } _ { n } , \pi _ { n } )$ is defined as the minimum number of insertions, deletions, and substitutions required to transform $\hat { \pi } _ { n }$ into $\pi _ { n }$ . To make the score comparable across trips of different lengths, the normalized edit distance is computed as follows:

$$
\mathrm { E D } _ { n } = \frac { d _ { \mathrm { E D } } ( \hat { \pi } _ { n } , \pi _ { n } ) } { \operatorname* { m a x } { ( \hat { L } _ { n } , L _ { n } ) } } .
$$

A lower value indicates better route reproduction.

## 5. Results and Discussions

## 5.1 Route Matching Performance and Comparative Analysis

Table 3 presents the comparative performance of the proposed framework against the four reference models on the test dataset. In this evaluation, the IoU is prioritized as the primary performance metric, as it provides a direct spatial measurement of link-level overlaps between the predicted and observed trajectories. To provide a multi-dimensional assessment, the TLLA and ED are incorporated as supplementary metrics, capturing route-length similarity and order-sensitive sequence consistency, respectively.

As reported in Table 3, the proposed framework consistently outperforms all reference models across the three metrics, achieving an IoU of 0.666, a TLLA of 0.926, and an ED of 0.302. Among the baselines, DNN-PSL exhibits the strongest performance within the path-based category, while AIRL emerges as the most competitive link-based alternative. It is important to contextualize these results: the path-based baselines (PSL and DNN-PSL) inherently benefit from an evaluation conducted on a pre-constrained candidate route set, whereas our model performs a direct, unconstrained prediction on the entire network. Despite this difference, the proposed model’s superior scores suggest a more faithful reconstruction of both the traversed link sequences and the overall spatial structure of observed routes. The performance of PURC, which is a static optimization model without path enumeration, lies between path- and link-based models. For TLLA, Purc is competitive with the proposed model.

These findings indicate that the decision-focused learning architecture, by internalizing latent perceived travel times, captures the underlying route choice process more effectively than conventional discrete choice, reinforcement learning, and static optimization approaches. The high TLLA and low ED scores further validate that the proposed model does not merely match isolated links but successfully preserves the sequential and topological integrity of the travelers' choices.

Table 3. Route reproduction performance on the test set.
<table><tr><td rowspan=2 colspan=2>Models</td><td rowspan=1 colspan=3>Metrics</td></tr><tr><td rowspan=1 colspan=1>loU</td><td rowspan=1 colspan=1>TLLA</td><td rowspan=1 colspan=1>ED</td></tr><tr><td rowspan=1 colspan=2>Proposed model</td><td rowspan=1 colspan=1>0.666</td><td rowspan=1 colspan=1>0.926</td><td rowspan=1 colspan=1>0.302</td></tr><tr><td rowspan=2 colspan=1>Path-based model</td><td rowspan=1 colspan=1>PSL</td><td rowspan=1 colspan=1>0.527</td><td rowspan=1 colspan=1>0.631</td><td rowspan=1 colspan=1>0.385</td></tr><tr><td rowspan=1 colspan=1>DNN-PSL</td><td rowspan=1 colspan=1>0.533</td><td rowspan=1 colspan=1>0.633</td><td rowspan=1 colspan=1>0.381</td></tr><tr><td rowspan=3 colspan=1>Link-based model</td><td rowspan=1 colspan=1>ReL</td><td rowspan=1 colspan=1>0.422</td><td rowspan=1 colspan=1>0.598</td><td rowspan=1 colspan=1>0.541</td></tr><tr><td rowspan=1 colspan=1>AIRL</td><td rowspan=1 colspan=1>0.445</td><td rowspan=1 colspan=1>0.573</td><td rowspan=1 colspan=1>0.576</td></tr><tr><td rowspan=1 colspan=1>PURC</td><td rowspan=1 colspan=1>0.489</td><td rowspan=1 colspan=1>0.901</td><td rowspan=1 colspan=1>0.476</td></tr></table>

To further scrutinize the model's capability in capturing behavioral diversity, Table 4 reports the performance metrics specifically for the diversity test set. This subset, characterized by OD pairs with multiple observed route alternatives, serves as a rigorous benchmark to evaluate whether individual-specific perceived travel times can effectively distinguish heterogeneous route-choice patterns. As shown in Table 4, the superiority of the proposed framework becomes even more pronounced in the more complex choice environment. The model achieves an IoU of 0.835, a TLLA of 0.939, and an ED of 0.153, consistently outperforming all competing methods. While the performance of the reference models also improves when restricted to this specialized subset, the magnitude of improvement for the proposed model significantly exceeds that of the baselines. Among reference models, PURC is enhanced the most when using the special subset, and for TLLA the performance of PUCR reaches that of the proposed model. DNN-PSL and AIRL remain the most competitive path-based and link-based models, respectively.

These results provide strong empirical evidence for the study's central hypothesis: when multiple routes are utilized for the same OD pair, characterizing latent, individual-specific perceived travel times allows for a more granular differentiation of route choices than is possible with existing modeling paradigms. Furthermore, the findings suggest that the proposed model does not merely converge toward an aggregate "average" route pattern. Instead, by internalizing the unique perception of each traveler through a decision-focused learning architecture, the model potentially accounts for the underlying heterogeneity that drives divergent route-choice behaviors under identical network conditions (further qualitative interpretations are provided in Sections 5.3 and 5.4).

Table 4. Route reproduction performance on the specialized diversity subset of test data.
<table><tr><td rowspan=2 colspan=2>Models</td><td rowspan=1 colspan=3>Metrics</td></tr><tr><td rowspan=1 colspan=1>loU</td><td rowspan=1 colspan=1>TLLA</td><td rowspan=1 colspan=1>ED</td></tr><tr><td rowspan=1 colspan=2>Proposed model</td><td rowspan=1 colspan=1>0.835</td><td rowspan=1 colspan=1>0.939</td><td rowspan=1 colspan=1>0.153</td></tr><tr><td rowspan=2 colspan=1>Path-based model</td><td rowspan=1 colspan=1>PSL</td><td rowspan=1 colspan=1>0.718</td><td rowspan=1 colspan=1>0.790</td><td rowspan=1 colspan=1>0.213</td></tr><tr><td rowspan=1 colspan=1>DNN-PSL</td><td rowspan=1 colspan=1>0.729</td><td rowspan=1 colspan=1>0.803</td><td rowspan=1 colspan=1>0.205</td></tr><tr><td rowspan=3 colspan=1>Link-based model</td><td rowspan=1 colspan=1>ReL</td><td rowspan=1 colspan=1>0.482</td><td rowspan=1 colspan=1>0.705</td><td rowspan=1 colspan=1>0.480</td></tr><tr><td rowspan=1 colspan=1>AIRL</td><td rowspan=1 colspan=1>0.509</td><td rowspan=1 colspan=1>0.694</td><td rowspan=1 colspan=1>0.392</td></tr><tr><td rowspan=1 colspan=1>PURC</td><td rowspan=1 colspan=1>0.768</td><td rowspan=1 colspan=1>0.929</td><td rowspan=1 colspan=1>0.215</td></tr></table>

To more rigorously evaluate the stability of the proposed framework, its predictive performance was analyzed across four distinct dimensions: the number of links per path, network path distance, path travel time, and Euclidean OD distance. Each dimension was categorized into three operational levels (low, medium, and high), and the three performance metrics (IoU, TLLA, and ED) were evaluated for each corresponding subset of the test data. Fig. 6 presents the comparative results between the proposed model and the three baseline frameworks.

The proposed model consistently outperforms all baselines across every level and dimension. While it is naturally expected that route reproduction performance might degrade as path complexity and length increase, our model demonstrates a unique and resilient trend. Specifically, the performance gap between the proposed model and the baselines widens as the path length increases.

A particularly noteworthy finding is observed in the TLLA metric, which assesses the degree of spatial overlap by considering the relative length of overlapping segments, rather than merely counting overlapping link for IoU. Unlike link- and path-based baseline models, whose performance significantly deteriorates as paths lengthen, the performance of PURC and the proposed model, as measured by TLLA, remains unchanged. The proposed model’s performance even slightly enhances with increased path length. When a path is long and complex at a high level, PURC is slightly inferior to the proposed model. This suggests that the decision-focused learning architecture is exceptionally effective at preserving the integrity of longer routes within complex urban networks. This ability to reliably reproduce long-distance trajectories constitutes a significant advantage of the proposed framework over existing route choice models.

![](images/2faeb27473afc84eda3fba0cd86d13af9679403f16257771373835aa084dd512.jpg)  
Fig. 6 The model performance comparison with respect to different dimensions

## 5.2 Ablation Analysis

To isolate the contribution of each architectural component in the perception model, an ablation analysis was conducted by systematically removing or replacing the sub-networks described in Section 3.4. Three ablation variants were constructed removing the personal, trip-context, and network state sub-networks. To examine the impact of network encoder in more detail, two more ablation settings are constructed, one removing network state features and the other replacing the GCN architecture with simple dense layers. Finally, the model without KL regularization that anchors the latent perceived cost distribution to the empirical distribution of observed travel costs.

Table 5. Route reproduction performance for ablation models on the test set
<table><tr><td colspan="1" rowspan="2">Models</td><td colspan="3" rowspan="1">Metrics</td></tr><tr><td colspan="1" rowspan="1">loU</td><td colspan="1" rowspan="1">TLLA</td><td colspan="1" rowspan="1">ED</td></tr><tr><td colspan="1" rowspan="1">Proposed model</td><td colspan="1" rowspan="1">0.666</td><td colspan="1" rowspan="1">0.926</td><td colspan="1" rowspan="1">0.302</td></tr><tr><td colspan="1" rowspan="1">Model without personal encoder</td><td colspan="1" rowspan="1">0.660</td><td colspan="1" rowspan="1">0.923</td><td colspan="1" rowspan="1">0.331</td></tr><tr><td colspan="1" rowspan="1">Model without trip-context encoder</td><td colspan="1" rowspan="1">0.658</td><td colspan="1" rowspan="1">0.921</td><td colspan="1" rowspan="1">0.308</td></tr><tr><td colspan="1" rowspan="1">Model without network state features</td><td colspan="1" rowspan="1">0.641</td><td colspan="1" rowspan="1">0.925</td><td colspan="1" rowspan="1">0.325</td></tr><tr><td colspan="1" rowspan="1">Model replacing GCN with dense layers</td><td colspan="1" rowspan="1">0.578</td><td colspan="1" rowspan="1">0.919</td><td colspan="1" rowspan="1">0.382</td></tr><tr><td colspan="1" rowspan="1">Model without KL regularization</td><td colspan="1" rowspan="1">0.505</td><td colspan="1" rowspan="1">0.910</td><td colspan="1" rowspan="1">0.481</td></tr></table>

Tables 5 summarize the ablation test results. Overall, the largest performance degradation is observed when the KL regularization term is removed. The second most influential component is the GCN-based network encoder. Replacing the graph-based architecture with simple feed-forward neural layers leads to a clear decline in route reproduction performance. This suggests that GCN plays a role of preserving the network topology.

By contrast, removing the personal encoder, trip-context encoder, or network state features causes relatively moderate degradation when testing against whole test data. However, the impact of three encoders becomes more pronounced in the more complex path environment. Table 6 lists the performance of variant models when using the specialized diversity subset of test data. In this setting, the performance degradation of the three input-removal variants is larger than that when full test data are used, suggesting that these contextual inputs become more useful when route choice ambiguity increases. This tendency is more clearly illustrated in Fig. 7. At low complexity levels, the input-removal variants perform similarly to the full model, but the gap gradually widens as route complexity increases. This suggests that individual-specific, trip-specific, and network-state information plays a complementary role that becomes increasingly important when the model must resolve more ambiguous and complex route choice situations.

Table 6. Route reproduction performance on the specialized diversity subset of test data
<table><tr><td rowspan=2 colspan=1>Models</td><td rowspan=1 colspan=3>Metrics</td></tr><tr><td rowspan=1 colspan=1>loU</td><td rowspan=1 colspan=1>TLLA</td><td rowspan=1 colspan=1>ED</td></tr><tr><td rowspan=1 colspan=1>Proposed</td><td rowspan=1 colspan=1>0.835</td><td rowspan=1 colspan=1>0.939</td><td rowspan=1 colspan=1>0.153</td></tr><tr><td rowspan=1 colspan=1>Without personal encoder</td><td rowspan=1 colspan=1>0.801</td><td rowspan=1 colspan=1>0.931</td><td rowspan=1 colspan=1>0.184</td></tr><tr><td rowspan=1 colspan=1>Without trip-context encoder</td><td rowspan=1 colspan=1>0.803</td><td rowspan=1 colspan=1>0.931</td><td rowspan=1 colspan=1>0.183</td></tr><tr><td rowspan=1 colspan=1>Without network state features</td><td rowspan=1 colspan=1>0.803</td><td rowspan=1 colspan=1>0.930</td><td rowspan=1 colspan=1>0.183</td></tr><tr><td rowspan=1 colspan=1>Without GCN</td><td rowspan=1 colspan=1>0.756</td><td rowspan=1 colspan=1>0.928</td><td rowspan=1 colspan=1>0.222</td></tr><tr><td rowspan=1 colspan=1>Without KL regularization</td><td rowspan=1 colspan=1>0.531</td><td rowspan=1 colspan=1>0.838</td><td rowspan=1 colspan=1>0.492</td></tr></table>

![](images/42194af6f24afa25454598399669f39a3fe14cc512cc448d4865829d38857f0e.jpg)  
Fig. 7 The model performance comparison for ablation with respect to different dimensions

## 5.3 Interpretation of Perceived Travel Times at Network Level

The most significant advantage of the proposed framework lies in its ability to externalize the latent perceived travel times within a driver’s decision-making process, thereby providing an interpretation of individual route-choice motives. Once properly calibrated, the model estimates personalized perceived link travel times within the testbed. However, the assumption that a driver maintains a unique and differentiated perception of travel times across the entire large-scale network is practically unrealistic.

Our empirical findings suggest that the learning process naturally regularizes a driver’s perception boundary, narrowing it down to a specific sub-area, which typically covers a corridor-like region encompassing the origin and destination. In areas remote from the chosen route, drivers do not deviate significantly from the baseline travel times provided by navigation services. This implies that travelers do not actively evaluate travel times for links irrelevant to their trip context, aligning with the principle of bounded rationality in spatial cognition.

To delineate this "perception area" and quantify the degree of personalization, we introduce a discrepancy index. This index measures the deviation of the perceived travel time $( \hat { C } _ { a , n } )$ from the observed mean travel time $( \widetilde { \mu } _ { a } )$ provided as the model input. For traveler � and link $a ,$ the timegap index $\delta _ { a , n }$ is defined as follows:

$$
\delta _ { a , n } = \frac { \mid \hat { C } _ { a , n } - \tilde { \mu } _ { a } \mid } { L _ { a } } ,
$$

where $L _ { a }$ denotes the length of link � for scaling the gap index. A higher value of $\delta _ { a , n }$ signifies a substantial deviation from the objective traffic state, indicating a strong personalized perception of that specific link. Conversely, an index value near zero suggests that the driver’s perception coincides with the observed travel time. This further suggests that the model captures a form of cognitive economy, wherein personalized cost evaluations are not internalized for network segments that fall outside the drivers’ immediate decision-making horizon.

![](images/e8ff2afeb8c5f55d619f4726c1cff02572463dbad472e4d4233064fb9ca266bb.jpg)  
Fig. 8 Spatial distribution of time gaps $( \delta _ { a , n } )$ for several chosen trips with different OD pairs

Fig. 8 illustrates the spatial distribution of the time-gap index $( \delta _ { a , n } )$ for six representative trips where the predicted and observed routes are identical. The perception area, highlighted in intense blue, appears to be spatially proportional to the Euclidean distance between the origin and destination. In all depicted cases, significant discrepancies $( \delta _ { a , n } > 0 )$ are concentrated in the vicinity of the origin, the destination, and the connecting corridor. Conversely, links situated further from this OD corridor consistently exhibit values near zero, indicating a convergence to the baseline mean travel times.

As the Euclidean OD distance increases from approximately 1.0–1.3 km to 4.1–5.4 km, the perception region is correspondingly expanded. This contraction and extension of the perception area are behaviorally plausible, reflecting that travelers scale their cognitive efforts according to trip length. Notably, this spatial pattern emerged autonomously through large-scale route data training, without the imposition of any explicit constraints or spatial priors. This suggests that the model effectively internalizes the realistic behavioral tendency of travelers to prioritize information within a localized decision-making corridor rather than considering the entire road network.

This relationship is further quantified in Fig. 9, which plots the radius of the perception area which is defined as the maximum distance encompassing all links with $\delta _ { a , n } > 0$ against the Euclidean OD distance. The results indicate a general proportionality between the trip distance and the spatial extent of the perception area. Furthermore, the variance in the radius of perception area increases as the origin and destination become more remote. This increased dispersion suggests that for longer trips, individual-specific preferences and route-choice strategies become more diverse, leading to more heterogeneous spatial configurations of the latent perception area.

![](images/6cb1f8a025634b88c944298f8b91a10145c05c6774ef5d6f667401f08d96f154.jpg)

![](images/7d02bd0208b264df260004e6e8e16d2ecf6a1c780202f2bc2eae7a484a523329.jpg)  
Fig. 9 Relationship between arial OD distance and the radius of perception area

Fig. 10(a) illustrates the link-level variations in estimated perceived travel times $( \widehat { \pmb { C } } _ { a , n } )$ . Links highlighted in intense blue represent segments where travelers’ perceptions deviate significantly from one another. Notably, freeways exhibit higher levels of perceptual variance, suggesting that drivers perceive travel times on high-capacity roads in more diverse manners. In Fig. 10(a), freeway links are delineated in yellow lines for readers’ better understanding. While some drivers may perceive their travel time to be shorter than the mean perceived travel time, others may internalize a significantly higher perceived travel time for the same segment. It is important to note that this finding is independent of other out-of-pocket costs, as no tolls are charged within the study area. This qualitative observation is further supported by the box plots in Fig. 10(c), which indicate that the interquartile range (IQR) of standard deviations for perceived travel times is substantially larger for freeways than for arterials. Beyond this distinction by road type, certain arterial segments also exhibit high levels of perceptual diversity when compared to other arterial segments, implying that specific road geometries or environmental factors may further contribute to the heterogeneity in driver perception.

In contrast, Fig. 10(b) depicts the variations in observed link travel times recorded throughout the day. There is no distinct evidence to suggest that the variance in observed travel times differs significantly between freeways and arterials. Collectively, Figs. 10(c) and 10(d) show that fluctuations across links in the standard deviation of perceived travel times are, in general, less pronounced than those in objective traffic states. This implies that the variance in actual travel times more fluctuates across links than that in perceived travel times. In particular for arterials, the IQR of standard deviations of perceived travel times is less than that of actual travel times. Although actual travel times are affected by varying traffic conditions across links, travelers tend to less consider these variations when perceiving their own travel times in complex urban road networks.

(a) Link-level Std.Dev of perceived travel times  
![](images/06def9705de3ea931721622d7da4a5b0bda88560920f544709c5951c51a08449.jpg)

(b) Link-level Std.Dev of observed link travel times  
![](images/c54db4fd7d7804e50dc992a70ed8fb63e9ebe21b3eb146bec96b1693d15212c8.jpg)  
(d) Box plots for Std.Dev of observed travel times by road type

(c) Box plots for Std.Dev of perceived travel times by road type  
![](images/e754f9f47c1baa432caf17e9b12ae2cd3ae6b7da27cff783f21dc4d184d0f5b0.jpg)

![](images/3b0e365dc1694a7b8b1f44cebc04e1d7b94babbc0b4b281dab84619b5f4ec41b.jpg)  
Fig. 10 Variations in perceived and observed link travel times

![](images/7650c05f3b620a83e7f7dfe5c4fc4644db74938b7ca9927677264ffb96f8d3a3.jpg)

## 5.4 Interpretation of Individual-Specific Route Choice Behaviors

Beyond network-level interpretations, the proposed model provides a way to interpret the latent motives underlying individual-specific route choice behaviors. The following case study demonstrates why observed routes often deviate from the objective shortest path and how travelers with identical OD pairs and departure times make different routing decisions.

![](images/c95ee45e5eccb7fdfae4a0ff791aaa6c7a926b033d8bbef66fb8bd53953c37f1.jpg)

![](images/2fd814575ab438e8b884bf38101c0514e3f0fc315b48c9aef78f1aaf42f0585f.jpg)

![](images/0677549006ca0667dfeb344bb615ca15eb7b5b4dffd8e3b0ccc814ec1ff1188d.jpg)

![](images/446d05ffd33e40ff9bd9e4639c0403eb67036799143925625c0a32bb921ad476.jpg)

(e) On-ramp bottleneck segment (zoom-in)  
(f) Merge-and-diverge section near the interchange (zoom-in)  
![](images/6f853897cee8cc657dc2e0fa098cae9d1797cba174ef59ca190048f04d137d07.jpg)  
Fig. 11 Explaining differences in individual route choices based on perceived travel times

Fig. 11 illustrates a scenario involving two travelers, A and B, sharing the same OD pair at 11:00 AM. As shown in Fig. 11(a), the objective shortest path under observed network conditions requires 11.5 minutes. However, traveler A selected a route with a travel time of 13.6 minutes, while traveler B chose a route of 12.6 minutes. Despite the objective efficiency of the 11.5-minute path, neither traveler selected it.

Figs. 11(b) and 11(c) display the estimated perceived link travel times for travelers A and B, respectively. For traveler A, the chosen route coincides perfectly with his/her subjective shortest path based on perceived travel times, even though he/she overestimates both travel times on the shortest and used routes. Specifically, traveler A perceives the chosen route to take 17.7 minutes, while estimating the objective shortest path at 25.9 minutes. A symmetrical pattern is observed for traveler B in Fig. 11(c), who underestimates the cost of the chosen route (12.0 min) while perceiving the objective shortest path as significantly more burdensome (18.3 min). These results underscore that travelers do not necessarily follow the objective traffic state; rather, they minimize travel times relative to their own idiosyncratic perception.

The mechanisms driving these deviations are further elucidated in Figs. 11(d)–(f). Fig. 11(d) plots the link-level travel times along the objective shortest path, where the observed travel times (baseline) are contrasted with the subjective perceptions of both travelers. A critical finding is that perceptual deviations are not uniform but are concentrated at specific road segments.

In Section ① (shaded in orange), both travelers perceive travel times to be substantially longer than the true congestion levels. As shown in the satellite imagery in Fig. 11(e), this segment corresponds to a freeway on-ramp prone to recurring traffic jams. The heightened uncertainty associated with merging maneuvers at this location likely causes both travelers to exaggerate the perceived delay, prompting them to avoid the objective shortest path altogether.

The differentiation between travelers A and B is primarily explained by Section ② (shaded in green), which corresponds to a weaving area near an interchange [Fig. 11(f)]. Traveler A exhibits a substantially higher perceived cost in this section, leading him/her to deviate from the objective shortest path to avoid the augmented complexities of the weaving area. In contrast, traveler B perceives Section ② as relatively endurable but chooses an earlier detour to avoid the anticipated burden of the on-ramp in Section ①.

These examples imply that behavioral heterogeneity in route choice stems from divergent individual perceptions of personalized network states. By capturing how travelers uniquely internalize the impedance of complex road geometries such as on-ramps and weaving sections, the proposed model successfully explains why individuals ignore the objective shortest path and why their chosen alternatives differ from one another even under identical external conditions.

![](images/23d8372fd0d992d1bec28a5902ec9608e38b2f1066783eb2b0ea6960787f8ce5.jpg)  
Fig. 12 Monte Carlo simulation of perceived travel times an OD pair

The proposed framework enables the sampling of latent perceived travel times through the encoder $\left[ f _ { \theta } ( \mathbf { x } _ { n } ) \right]$ within the model pipeline. To examine the stochastic nature of individual route choice, we selected for a Monte Carlo simulation three travelers sharing an identical OD pair and departure hour. Fig. 12 illustrates the results, where the observed routes are highlighted in green and the additional paths generated through 100 simulations, each case based on a unique set of sampled perceived travel times, are shown in blue. Despite conducting 100 simulations, the number of generated routes ranges from 1 to 3. The labels indicate the selection frequency of each route as the shortest path under the simulated perception of travel times.

The simulation results reveal distinct levels of route adherence among the three travelers. Traveler C exhibits the most stable behavior; all 100 sampled instances consistently converge to the observed route, which also happens to be identical to the objective shortest path. This suggests that traveler C has a highly deterministic perception of travel times with negligible fluctuation. In contrast, travelers D and E, who share the same observed route, demonstrate greater flexibility. Traveler D’s observed route remains dominant (95%), but an alternative path emerges in 5% of the simulations. Traveler E displays the highest perceptual variance, with the observed route reproduces in 81% of cases, while two additional alternatives are sampled at frequencies of 14% and 5%. These findings imply that traveler E possesses lower confidence or higher stochasticity in travel time perception compared to the others.

A key contribution of the proposed model is its ability to distinguish between intra-individua heterogeneity (variance within a single driver's perception) and inter-individual heterogeneity (differences across drivers). The heterogeneity of the former can be revealed by the variance returned by the perception encoder $\left[ f _ { \theta } (  { \mathbf { x } } _ { n } ) \right]$ in the model pipeline, and the latter can be identified by the mean travel time returned by the encoder. While travelers D and E utilized the same observed route, their simulated alternative routes are disjoint. It is not easy to separate causes of this different perception into intra- and inter-individual heterogeneity using the conventional discrete choice modeling (Bhat and Sardesai, 2006; Krueger et al., 2021). The proposed model captures these differences in perception even when revealed behaviors (=the chosen routes) are identical.

Empirically from the simulations, the generated routes are not arbitrarily dispersed across the network. Despite the Monte Carlo sampling being applied to the entire network, the model does not generate irrational alternatives far from the OD corridor. This suggests that the decision-focused learning process imposes inherent rational constraints, ensuring that the generated stochasticity remains behaviorally plausible. Overall, the model not only reproduces divergent routes across the population but also accounts for varying degrees of route adherence and confidence, among travelers sharing the same spatial and temporal trip backgrounds.

![](images/c0004e3d1a70dfcbc6b83366486ab83c47e7319724e59c2eac5dfe46e4671320.jpg)

(b) Change in input travel times by +1.5 min.  
![](images/ee0c802f24ea8f5155a87f3286622ba2f5cdea166eb7ccaa3e730bab92289edf.jpg)  
(d) Change in input travel times by+9min.

(c) Change in input travel times by +6 min.  
![](images/0b48a3ee44b6982e53514873e9788e71e38d272e4c6b52419662a027a9275f13.jpg)

![](images/c0b7c678c9d5eaf38ecf16059da7a386f4f2ea8ba3f9cc3971ad850ae36fc52b.jpg)  
Fig. 13 Counterfactual test for rerouting under perturbed link travel times

To evaluate the model’s responsiveness to dynamic changes in network states, we conducted a counterfactual test by systematically increasing the input travel times on a specific road segment. This analysis demonstrates how the proposed model, much like conventional route choice models, adapts its predictions to varying traffic conditions while preserving individual-specific behavioral properties. For this experiment, we selected three travelers (F, G, and H) who share the same OD pair and initially utilize the same observed route, as shown in Fig. 13(a).

The simulation results reveal that travelers possess distinct thresholds for rerouting under identical network disruptions. As illustrated in Fig. 13(b), a modest increase in the input travel time to 1.5 minutes triggers a reroute only for traveler H, while travelers F and G remain on their original paths. When the impedance is further increased to 6 minutes [Fig. 13(c)], traveler F opts for an alternative route, yet traveler G continues to adhere to the original route. Traveler G only deviates from the initial route once the travel time reaches a significant threshold of 9 minutes [Fig. 13(d)]. These findings indicate that even among drivers with identical outward environments, their underlying "resistance" to network changes varies substantially, reflecting diverse levels of risk aversion or route familiarity.

Beyond the decision of whether to reroute, Fig. 13 highlights significant differences in the spatial patterns of reroute. Traveler H exhibits high sensitivity, initiating a detour near the origin, far from the disrupted segment, to proactively avoid the bottleneck. In contrast, travelers F and G demonstrate lower sensitivity, sticking to the original corridor as long as possible and diverting only in the immediate vicinity of the affected links.

The results of this counterfactual test confirm that the proposed framework functions as a robust, general-purpose route choice model capable of responding to varying traffic conditions. Its ability to predict traveler-specific responses, addressing who reroutes, when (at what travel time threshold), and where (the extent of spatial deviation), offers significant practical value. In real-world scenarios such as traffic incidents, road construction, or special events that cause sudden travel time surges, the model can forecast heterogeneous rerouting behaviors. This level of granularity is essential for developing personalized traffic management strategies and for generating traffic information enroute when an unexpected event occurs.

## 6. Conclusions

This study introduces a robust framework to accurately reproduce observed trajectories at the individual level and bypasses path enumeration and overlapping problems. The integration of decision-focused learning facilitates a seamless alignment between the solution of a shortest-path finding problem and observed human behavior. This approach leverages modern machine learning techniques to enable the end-to-end calibration of latent behavioral parameters.

The empirical results demonstrate that the proposed framework consistently outperforms five baseline models in path reproduction. Its superiority is particularly evident in scenarios characterized by high route diversity, where the model excels at identifying and replicating heterogeneous choice patterns across different OD pairs. Furthermore, the model exhibits remarkable robustness against trip scale, whereas the performance of conventional models typically deteriorates as path length and travel time increase, the predictive capability of the proposed framework is reinforced, maintaining high accuracy across long-distance paths.

A significant capability of the proposed model is to retrieve latent perceived travel times. By externalizing these subjective costs, the model provides a microscopic lens through which the underlying motives of route choice can be interpreted. Simulation experiments revealed that travelers under identical traffic conditions arrive at divergent decisions due to their unique, individualized perception fields. Moreover, the analysis of these retrieved outputs confirms that a traveler’s cognitive range is spatially bound, focusing primarily on the immediate trip corridor. This offers insights into the hidden cognitive mechanisms that drive urban mobility.

The primary constraint of the present study lies in the absence of socio-demographic covariates for a more granular retrieval of personalized travel times. The integration of explicit individual-specific data in future research is expected to dramatically enhance the model’s explanatory power and predictive precision. The proposed framework, at its current state, is only for improving navigation services by individualizing the route guidance. The framework needs to be extended to a traffic assignment model that can be used in downstream transportation planning tasks.

## Acknowledgments

This research was supported by the Chung-Ang University research grant in 2026 and by the National Research Foundation of Korea (NRF) Grant funded by the Korean Government (RS-2024- 00337956).

## References

Amos, B., Kolter, J.Z., 2017. OptNet: Differentiable Optimization as a Layer in Neural Networks. In: Proceedings of the 34th International Conference on Machine Learning, PMLR 70, 136–145.

Bhat, C. R., & Sardesai, R., 2006. The impact of stop-making and travel time reliability on commute mode choice. Trans<sub>p</sub>ortation Research Part B: Methodolo<sub>g</sub>ical, 40(9), 709-730.

Bao, Y., Huang, J., Shen, Q., Cao, Y., Ding, W., Shi, Z., & Shi, Q., 2023. Spatial–temporal complex graph convolution network for traffic flow prediction. En<sub>g</sub>ineerin<sub>g</sub> A<sub>pp</sub>lications of Artificial , , 106044.

Bekhor, S., Ben-Akiva, M., Ramming, M.S., 2001. Evaluation of choice set generation algorithms for route choice models. Annals of Operations Research 144(1), 235–247.

Ben-Akiva, M., Bierlaire, M., 1999. Discrete choice methods and their applications to short-term travel decisions. In: Hall, R.W. (Ed.), Handbook of Transportation Science. Kluwer, Dordrecht, pp. 5– 33.

Ben-Akiva, M., Lerman, S.R., 1985. Discrete Choice Analysis: Theory and Application to Travel Demand. MIT Press, Cambridge, MA.

Bertsimas, D., Kallus, N., 2020. From predictive to prescriptive analytics. Management Science 66(3), 1025–1044.

Bovy, P.H.L., Stern, E., 1990. Route Choice: Wayfinding in Transport Networks. Kluwer, Dordrecht.

Cantarella, G.E., de Luca, S., 2005. Multilayer feedforward networks for transportation mode choice analysis: An analysis and a comparison with random utility models. Transportation Research Part C: Emerging Technologies 13(2), 121–155.

Choi, S., Kim, J., Yeo, H., 2021. TrajGAIL: Generating urban vehicle trajectories using generative adversarial imitation learning. Transportation Research Part C: Emerging Technologies 128, 103091.

Chowdhury, A., Chakravarty, T., Ghose, A., Banerjee, T., Balamuralidhar, P., 2018. Investigations on driver unique identification from smartphone’s GPS data alone. Journal of Advanced Transportation 2018(1), 9702730.

Demirović, E., Stuckey, P.J., Bailey, J., Chan, J., Leckie, C., Ramamohanarao, K., Guns, T., 2019. An investigation into prediction + optimisation for the knapsack problem. In: Integration of Constraint Programming, Artificial Intelligence, and Operations Research, pp. 241–257.

Domke, J., 2010. Implicit differentiation by perturbation. Advances in Neural Information Processing Systems, 23.

Elmachtoub, A.N., Grigas, P., 2022. Smart “predict, then optimize.” Management Science 68(1), 9– 26.

Fosgerau, M., Frejinger, E., Karlstrom, A., 2013. A link based network route choice model with unrestricted choice set. Transportation Research Part B 56, 70–80.

Fosgerau, M., Paulsen, M., & Rasmussen, T. K. (2022). A perturbed utility route choice model. Trans<sub>p</sub>ortation Research Part C: Emer<sub>g</sub>in<sub>g</sub> Technolo<sub>g</sub>ies, 136, 103514.

Fosgerau, M., Nielsen, N., Paulsen, M., Rasmussen, T., & Yao, R. (2024). Estimating the perturbed utility route choice model with individual-level data. Proceedings of the hEART 2024.

Frejinger, E., Bierlaire, M., Ben-Akiva, M., 2009. Sampling of alternatives for route choice modeling. Transportation Research Part B 43(10), 984–994.

Fung, N.C., Wallace, B., Chan, A.D.C., Goubran, R., Porter, M.M., Marshall, S., Knoefel, F., 2017. Driver identification using vehicle acceleration and deceleration events from naturalistic driving of older drivers. In: 2017 IEEE International Symposium on Medical Measurements and Applications (MeMeA), pp. 33–38.

Hallac, D., Sharang, A., Stahlmann, R., Lamprecht, A., Huber, M., Roehder, M., Leskovec, J., 2016. Driver identification using automobile sensor data from a single turn. In: 2016 IEEE 19th International Conference on Intelligent Transportation Systems (ITSC), pp. 953–958.

Jan, O., Horowitz, A.J., Peng, Z.R., 2000. Using global positioning system data to understand variations in path choice. Transportation Research Record 1725, 37–44.

Jang, E., Gu, S., Poole, B., 2017. Categorical reparameterization with Gumbel-Softmax. In: International Conference on Learning Representations (ICLR).

Kim, G., Kang, Y., Kock, L., Bansal, P., & Sohn, K. (2026). Scalable variational inference for multinomial probit models under large choice sets and sample sizes. Statistics and , (1), 47.

Kim, H., Kwon, S., Wu, S. K., & Sohn, K., 2014. Why do passengers choose a specific car of a metro train during the morning peak hours?. Trans<sub>p</sub>ortation research <sub>p</sub>art A: <sub>p</sub>olic<sub>y</sub> and <sub>p</sub>ractice, 61, 249- 258.

Kingma, D.P., Welling, M., 2014. Auto-encoding variational Bayes. In: International Conference on Learning Representations (ICLR).

Krueger, R., Bierlaire, M., Daziano, R. A., Rashidi, T. H., & Bansal, P., 2021. Evaluating the predictive abilities of mixed logit models with unobserved inter-and intra-individual heterogeneity. Journal of , , 100323.

Lee, I., Park, H., & Sohn, K., 2012. Increasing the number of bicycle commuters. In Proceedin<sub>g</sub>s of (Vol. 165, No. 1, pp. 63-72). Thomas Telford Ltd.

Levinson, D., Zhu, S., 2013. A portfolio theory of route choice. Transportation Research Part C: Emerging Technologies 35, 232–243.

Liu, Y., Rasouli, S., Wong, M., Feng, T., & Huang, T., 2024. RT-GCN: Gaussian-based spatiotemporal graph convolutional network for robust traffic prediction. Information Fusion, 102, 102078.

Liu, Z., Yin, Y., Bai, F., & Grimm, D. K., 2023. End-to-end learning of user equilibrium with implicit neural networks. Transportation Research Part C: Emerging Technologies, 150, 104085.

Mai, T., Frejinger, E., Fosgerau, M., 2015. A nested recursive logit model for route choice analysis. Transportation Research Part B: Methodological 75, 100–112.

Mandi, J., Kotary, J., Berden, S., Mulamba, M., Bucarey, V., Guns, T., Fioretto, F., 2024. Decisionfocused learning: Foundations, state of the art, benchmark and future opportunities. Journal of Artificial Intelligence Research 80, 1623–1701.

Marra, A.D., Corman, F., 2020. Determining an efficient and precise choice set for public transport based on tracking data. Transportation Research Part A: Policy and Practice 142, 168–186.

Marra, A.D., Corman, F., 2021. A deep learning model for predicting route choice in public transport. In: 21st Swiss Transport Research Conference (STRC 2021), Ascona, Switzerland.

Meyer de Freitas, L., Becker, H., Zimmermann, M., Axhausen, K.W., 2019. Modelling intermodal travel in Switzerland: A recursive logit approach. Transportation Research Part A: Policy and Practice 119, 200–213.

Niepert, M., Minervini, P., Franceschi, L., 2021. Implicit MLE: backpropagating through discrete exponential family distributions. Advances in Neural Information Processing Systems 34, 14567– 14579.

Oyama, Y., Hato, E., 2017. A discounted recursive logit model for dynamic gridlock network analysis. Transportation Research Part C: Emerging Technologies 85, 509–527.

Park, H., Lee, Y. J., Shin, H. C., & Sohn, K., 2011. Analyzing the time frame for the transition from leisure-cyclist to commuter-cyclist. Trans<sub>p</sub>ortation, 38(2), 305-319.

Papandreou, G., Yuille, A.L., 2010. Gaussian sampling by local perturbations. Advances in Neural Information Processing Systems 23.

Papinski, D., Scott, D.M., Doherty, S.T., 2009. Exploring the route choice decision-making process. Transportation Research Part F 12(5), 347–358.

Prato, C.G., 2009. Route choice modeling: past, present and future research directions. Journal of Choice Modelling 2(1), 65–100.

Qiu, S., Qin, G., Wong, M., Sun, J., 2024. RoutesFormer: A sequence-based route choice Transformer for efficient path inference from sparse trajectories. Transportation Research Part C: Emerging Technologies 162, 104552.

Ramírez, H.G., Leclercq, L., Chiabaut, N., Becarie, C., Krug, J., 2021. Travel time and bounded rationality in travellers’ route choice behaviour: A computer route choice experiment. Travel Behaviour and Society 22, 59–83.

Rosenblad, B., Lin, X., & Yin, Y. (2026). Inverse Estimation of The Perturbed Utility Route Choice Model. Available at SSRN 6756317.

Sohn, K., & Yun, J., 2009. Separation of car-dependent commuters from normal-choice riders in mode-choice analysis. Trans<sub>p</sub>ortation, 36(4), 423-436.

Sun, H. J. N., & Arslan, O., 2025. A contextual framework for learning routing experiences in lastmile delivery. Trans<sub>p</sub>ortation Research Part B: Methodolo<sub>g</sub>ical, 194, 103172.

Vlastelica, M., Paulus, A., Musil, V., Martius, G., Rolinek, M., 2020. Differentiation of blackbox combinatorial solvers. In: International Conference on Learning Representations (ICLR).

Wang, S., Wang, Q., Zhao, J., 2020. Deep neural networks for choice analysis: Extracting complete economic information for interpretation. Transportation Research Part C: Emerging Technologies 118, 102701.

Wilder, B., Dilkina, B., Tambe, M., 2019. Melding the data-decisions pipeline: decision-focused learning for combinatorial optimization. Proceedings of the AAAI Conference on Artificial Intelligence 33, 1658–1665.

Yao, R., & Bekhor, S. (2022). A variational autoencoder approach for choice set generation and implicit perception of alternatives in choice modeling. Trans<sub>p</sub>ortation Research Part B: , , 273-294.

Yu, B., Lee, Y., & Sohn, K., 2020. Forecasting road traffic speeds by considering area-wide spatiotemporal dependencies based on a graph convolutional neural network (GCN). Trans<sub>p</sub>ortation research part C: emerging technologies, 114, 189-204.

Zhao, Z., Liang, Y., 2023. A deep inverse reinforcement learning approach to route choice modeling with context-dependent rewards. Transportation Research Part C: Emerging Technologies 149, 104079.

Zhu, S., Levinson, D., 2015. Do people use the shortest path? An empirical test of Wardrop’s first principle. PLOS ONE 10(8), e0134322.

Zimmermann, M., Mai, T., Frejinger, E., 2017. Bike route choice modeling using GPS data without choice sets of paths. Transportation Research Part C: Emerging Technologies 75, 183–196.

## Appendix

## Appendix A. Hyperparameters and experimental settings

Table A1. Hyperparameter settings for training
<table><tr><td rowspan="2">Batch size (B)</td><td rowspan="2">Perturbat ion scale (λ)</td><td rowspan="2">Weight for KL- divergence (ξ)</td><td rowspan="2">Perturbation sample size (s)</td><td rowspan="2">Learning rate (n)</td><td colspan="3">SoG parameters</td></tr><tr><td>K</td><td>τ</td><td>g</td></tr><tr><td>500</td><td>0.8</td><td>0.01</td><td>1</td><td>1.25</td><td>50</td><td>0.2</td><td>10</td></tr></table>

In implementation, � was fixed to an empirical constant reflecting the average number of links in the observed routes, for computational simplicity.

Table A2. Constant values and dimensions of covariates
<table><tr><td rowspan=1 colspan=2>Testbed network size</td><td rowspan=2 colspan=1>Size of trainingset(N)</td><td rowspan=2 colspan=1>Size of testset</td><td rowspan=2 colspan=1>Size of sub-testset for diversity</td></tr><tr><td rowspan=1 colspan=1>Number of nodes(IV|)</td><td rowspan=1 colspan=1>Number of directed links(|AI)</td></tr><tr><td rowspan=1 colspan=1>7,723</td><td rowspan=1 colspan=1>14,769</td><td rowspan=1 colspan=1>105,434</td><td rowspan=1 colspan=1>26,358</td><td rowspan=1 colspan=1>3,668</td></tr></table>

<table><tr><td colspan="1" rowspan="1">Covariate group</td><td colspan="1" rowspan="1">Variable definition</td></tr><tr><td colspan="1" rowspan="1">Number of individual-specific variables(=driving habitvariables) $( d _ { p e r s o n } )$ </td><td colspan="1" rowspan="1">Summary statistics of driving-habit variables, including the mean, standarddeviation, and interquartile range of acceleration, deceleration, and jerk (9)</td></tr><tr><td colspan="1" rowspan="1">Number of origin-specific locationvariables $( d _ { o } )$ </td><td colspan="1" rowspan="1">Building floor area within a 500 m radius of the origin node, classified into 11building-use categories: residential, business/commercial, neighborhood living,cultural/educational, religious, accommodation/leisure, industrial/production,transportation/automobile, agricultural/livestock/environmental, public, andspecial-purpose facilities; and the x- and y-coordinates of the origin node (13)</td></tr><tr><td colspan="1" rowspan="1">Number of destination-specific variables $( d _ { d } )$ </td><td colspan="1" rowspan="1">Building floor area within a 500 m radius of the destination node, classifiedinto the same 11 building-use categories; and the x- and y-coordinates of thedestination node (13)</td></tr><tr><td colspan="1" rowspan="1">Number of staticnetwork states $( d _ { s t a t i c } )$ </td><td colspan="1" rowspan="1">Link-level attributes, including travel time, link length, number of lanes, roadclass, and x- and y-coordinates (6)</td></tr><tr><td colspan="1" rowspan="1">Number of dynamicnetwork states</td><td colspan="1" rowspan="1">Mean and standard deviation of link travel time measured at 1hour intervals(2)</td></tr><tr><td> $( d _ { d y n a m i c } )$ </td><td></td></tr></table>

Appendix B. Architecture of Perception Sub-Models
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Traveler attributesencoder $\underline { { ( g _ { \theta _ { 1 } } ) } }$ </td><td rowspan=1 colspan=1>Trip context encoder $( g _ { \boldsymbol { \theta } _ { 2 } } )$ </td><td rowspan=1 colspan=1>Traffic network encoder $( g _ { \theta _ { 3 } } )$ </td><td rowspan=1 colspan=1>Perception encoder $( g _ { \theta _ { 4 } } )$ </td></tr><tr><td rowspan=1 colspan=1>Input</td><td rowspan=1 colspan=1> $x _ { n } ^ { p e r s o n } [ B , d _ { p e r s o n } ]$ </td><td rowspan=1 colspan=1> $\begin{array} { r } { \pmb { x } _ { n } ^ { d e p T } [ B , 1 ] , } \end{array}$  $\textstyle { \pmb { x } } _ { n } ^ { o } [ B , d _ { o } ] ,$  $x _ { n } ^ { \ddot { d } } [ B , \dot { d } _ { d } ] ,$  ${ \underline { { h _ { n } ^ { p e r s o n } [ B , 3 2 ] } } }$ </td><td rowspan=1 colspan=1> $\begin{array} { r } { { \pmb x } _ { n } ^ { s t a t i c \_ n e t } [ B , | A | , d _ { s t a t i c } ] , } \end{array}$  $\pmb { x } _ { n } ^ { d y \bar { n } a m i c . n e t } \bar { [ B , | A | , d _ { d y n a m i c } ] }$ </td><td rowspan=1 colspan=1> ${ \pmb h } _ { n } ^ { p e r s o n } [ B , 3 2 ] ,$  $h _ { n } ^ { t r i p } [ B , 2 5 6 ] ,$  $\pmb { h } _ { n } ^ { n e t } [ B , | A | , 2 5 6 ]$ </td></tr><tr><td rowspan=1 colspan=1>Layers</td><td rowspan=1 colspan=1> $[ \ln { \mathsf { p u t } } ; x _ { n } ^ { p e r s o n } ]$ Linear (64)LayerNorm&amp;GELULinear (32) $\mathrm { [ O u t p u t : ~ } h _ { n } ^ { p e r s o n } ]$ </td><td rowspan=1 colspan=1>Temporal encoder[Input: $x _ { n } ^ { d e p T } ]$ Linear (256) $\mathrm { [ O u t p u t : ~ } E m d _ { h o u r } ]$ Origin encoder[Input: x0]Linear (256)LayerNorm&amp;GELU[Output: $E m d _ { o } ]$ Destination encoder[Input: x]Linear (256)LayerNorm&amp;GELU[Output: $E m d _ { d } ]$ OD fusion[Input: $E m d _ { o } , E m d _ { d } ]$ Concatenate $( E m d _ { o } , E m d _ { d } )$ Linear (256)LayerNorm&amp;GELU $[ \mathsf { O u t p u t : } E m d _ { o d } ]$ Personal fusion[Input: $E m d _ { h o u r } , E m d _ { o d } ,$  $\pmb { h } _ { n } ^ { p e r s o n } ]$  $\mathsf { C o n c a t e n a t e \ ( E m d _ { h o u r } , \ E m d _ { o d } , }$  ${ \pmb { h } } _ { n } ^ { p e r s o n } ) ~ [ B , 5 4 4 ]$ Linear (256)LayerNorm&amp;GELULinear (256)LayerNorm&amp;GELU $[ \mathsf { O u t p u t } ; h _ { n } ^ { t r i p } ]$ </td><td rowspan=1 colspan=1>Static encoder[Input: $x _ { n } ^ { s t a t i c . n e t } ]$ Linear (256)LayerNorm&amp;GELUGCN*3Linear (256)LayerNorm&amp;GELU $[ \mathsf { O u t p u t : } E m d _ { s t a t i c } ]$ Dynamic encoder[Input: $\pmb { x } _ { n } ^ { d y n a m i c . n e t } ]$ Linear (256)LayerNorm&amp;GELUGCN*3Linear (256)LayerNorm&amp;GELU[Output: $E m d _ { d y n a m i c } ]$ Fusion[Input: Emdstatic, $E m d _ { d y n a m i c } ]$ Concatenate $( E m d _ { s t a t i c } ,$  $E m d _ { d y n a m i c } ) \left[ B , | A | , 5 1 2 \right]$ Linear (256)LayerNorm&amp;GELULinear (256)LayerNorm&amp;GELU[Output: $\pmb { h } _ { n } ^ { n e t } ]$ </td><td rowspan=1 colspan=1>Fusion[Input: $\pmb { h } _ { n } ^ { p e r s o n } , \pmb { h } _ { n } ^ { t r i p } , \pmb { h } _ { n } ^ { n e t } ]$ Concatenate $( \pmb { h } _ { n } ^ { p e r s o n }$  $\pmb { h } _ { n } ^ { t r i p } , \pmb { h } _ { n } ^ { n e t } ) \left[ \boldsymbol { B } , | \boldsymbol { A } | , 5 4 4 \right]$ Linear (256)LayerNorm&amp;GELULinear (256)LayerNorm&amp;GELU[Output: $E m d _ { f u s i o n } ]$ Mean head[Input: $E m d _ { f u s i o n } ]$ Linear (256)LayerNorm&amp;GELULinear (1)[Output: $\mu _ { n } ]$ Variance head[Input: $E m d _ { f u s i o n } ]$ Linear (256)LayerNorm&amp;GELULinear (1)&amp;Softplus[Output: $\Sigma _ { n } ]$ </td></tr><tr><td rowspan=1 colspan=1>Output</td><td rowspan=1 colspan=1> ${ \pmb h } _ { n } ^ { p e r s o n } [ { \bf B } , 3 2 ]$ </td><td rowspan=1 colspan=1> $\underline { { h } } _ { n } ^ { t r i p } [ B , 2 5 6 ]$ </td><td rowspan=1 colspan=1> $h _ { n } ^ { n e t } [ B , | A | , 2 5 6 ]$ </td><td rowspan=1 colspan=1> $\mu _ { n } [ B , | A | ] , \Sigma _ { n } [ B , | A | ]$ </td></tr></table>

## Appendix C. Model specifications of reference models

Table C1. Model specifications of reference models
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Formulation</td><td rowspan=1 colspan=1>Input level</td><td rowspan=1 colspan=1>Input variables</td></tr><tr><td rowspan=1 colspan=1>PSL</td><td rowspan=1 colspan=1>Discrete choice modelwith a linear utilityspecification</td><td rowspan=1 colspan=1>Path</td><td rowspan=1 colspan=1>Path-level aggregates of link travel time, linklength, number of lanes, road class (4)</td></tr><tr><td rowspan=1 colspan=1>DNN-PSL</td><td rowspan=1 colspan=1>Discrete choice modelwith a nonlinear utilityspecification</td><td rowspan=1 colspan=1>Path</td><td rowspan=1 colspan=1>Path-level aggregates of link travel time, linklength, number of lanes, road class (4)</td></tr><tr><td rowspan=1 colspan=1>ReL</td><td rowspan=1 colspan=1>Discrete choice modelwith a linear utilityspecification</td><td rowspan=1 colspan=1>Link</td><td rowspan=1 colspan=1>Link travel time, link length, number oflanes, and road class (4)</td></tr><tr><td rowspan=2 colspan=1>AIRL</td><td rowspan=2 colspan=1>Sequential decision modelwith a nonlinear rewardspecification</td><td rowspan=2 colspan=1>Link</td><td rowspan=1 colspan=1>State variables: link travel time, link length,number of lanes, and road class (4)</td></tr><tr><td rowspan=1 colspan=1>Context variable: departure time andremaining distance to the destination (2)</td></tr><tr><td rowspan=1 colspan=1>PURC</td><td rowspan=1 colspan=1>Static optimization</td><td rowspan=1 colspan=1>Link</td><td rowspan=1 colspan=1>Link travel time, link length, number oflanes, and road class (4)</td></tr></table>