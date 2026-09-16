# NEURAL FIELD ENSEMBLES FOR AERODYNAMIC SURFACE PREDICTION: WINNING SOLUTION TO THE ONERA CRM WALL DISTRIBUTION 2025 CHALLENGE

Lionel Salesses Cenaero, Gosselies, Belgium lionel.salesses@cenaero.be

Caroline Sainvitu Cenaero, Gosselies, Belgium

Tariq Benamara Cenaero, Gosselies, Belgium

## ABSTRACT

Machine-learning surrogate models offer a promising alternative to high-fidelity Computational Fluid Dynamics (CFD) simulations for aerodynamic analysis and design. However, constructing accurate surrogates for realistic aircraft configurations remain challenging due to complex geometries, multiple flow regimes, and limited training data. This work presents the methodology that achieved first place in the ONERA CRM Wall Distribution Regression Challenge, which focuses on predicting pressure and skin-friction coefficient distributions over the NASA Common Research Model wing-body-pylonnacelle configuration under different operating conditions. The proposed approach formulates the problem as a conditional neural field mapping spatial coordinates, surface normals, and operating conditions to aerodynamic wall quantities. Fourier feature encoding, a relative squared error objective aligned with the challenge metric, ensemble learning, and k-fold cross-validation are progressively introduced to improve prediction accuracy and exploit the limited training data. Beyond presenting the final methodology, the paper documents the successive model design choices that led to the winning solution through a comprehensive ablation study and discusses several alternative approaches that were investigated but ultimately discarded. On the hidden competition test set, the proposed methodology achieves an overall score of 8.81, outperforming the strongest organizer-provided baseline, which achieved a score of 8.64, while requiring approximately three orders of magnitude fewer trainable parameters. These results illustrate that carefully designed coordinate-based neural fields constitute an efficient and robust framework for aerodynamic surrogate modeling on complex geometries under limited-data conditions.

## 1 Introduction

High-fidelity CFD simulations based on Reynolds-Averaged Navier–Stokes (RANS) equations are essential in aerodynamic design but remain computationally prohibitive for many-queries situations. This limitation has motivated the development of surrogate models capable of approximating flow solutions at a reduced computational expense. Despite recent advances in machine learning for scientific computing, constructing accurate surrogates for aerodynamic surface quantities remains challenging. The difficulty arises from the combination of complex geometries, highly nonlinear flow physics, and limited availability of high-fidelity data. In particular, transonic regimes introduce localized discontinuities associated with shock waves, while high angles of attack lead to flow separation and increased sensitivity to small variations in operating conditions. These phenomena make the construction of robust and accurate surrogate model particularly challenging.

The ONERA CRM wall distribution challenge provides a representative benchmark for aforementioned difficulties. The task consists in predicting pressure and friction fields over a realistic aircraft configuration for a wide range of aerodynamic conditions. The large number of surface points (260,774), the irregular nature of the geometry, and the diversity of flow regimes make conventional grid-based machine learning approaches difficult to apply directly. At the same time, the relatively limited number of available simulations imposes strong constraints on model complexity and generalization.

![](images/07f4dab3e887682f0797a8399329e45b716b586472f3d56f03de76c2a0b1711a.jpg)  
Figure 1: NASA Common Research Model geometry

In this work, we present the methodology that achieved the best performance on the final hidden test set of the challenge. Our approach relies on coordinate-based neural field representations, which model aerodynamic quantities as continuous functions of spatial coordinates and flow parameters. This formulation naturally accommodates irregular geometries without requiring mesh connectivity information. To improve the representation of high-frequency flow structures, the neural fields are augmented with Fourier Feature Encoding (FFE) that mitigate the spectral bias of standard Multi-Layer Perceptrons (MLP). We further show that aligning the training objective with the challenge evaluation metric is critical for achieving robust performance, particularly on the most difficult flow configurations. Finally, we demonstrate that ensemble learning combined with k-fold cross-validation provides an effective strategy for improving generalization in a low-data regime.

Challenge context. This work originates from the ONERA CRM Wall Distribution Regression Challenge, organized by Jacques Peter and Quentin Bennehard (ONERA) and hosted on the Codabench platform<sup>1</sup>. The challenge was launched in May 2025 with the objective of evaluating machine learning approaches for the prediction of aerodynamic wall quantities on a realistic aircraft configuration representative of industrial applications. The benchmark is based on the CRM-WBPN database, introduced in Peter et al. [1], containing 468 RANS simulations performed on the NASA/Boeing Common Research Model Wing-Body-Pylon-Nacelle (CRM-WBPN) geometry (see Figure 1) over a wide range of Mach numbers, Reynolds numbers and Angles of Attack (AoA). Participants were tasked with predicting wall distributions of pressure and friction coefficients from geometric descriptors and aerodynamic operating conditions.

Beyond its competitive aspect, the challenge provided a common evaluation framework for comparing a broad spectrum of surrogate modeling strategies on a challenging aerodynamic regression problem. It attracted 64 participants from both academia and industry and received a total of 75 submissions. Following the completion of the challenge, the CRM-WBPN database was publicly released, making it possible to reproduce and extend the results obtained during the competition<sup>2</sup>.

Table 1 summarizes the top-ranked solutions on the final hidden test set. The methodology presented in this paper achieved the best overall score among all submitted approaches. Beyond reporting the final solution, the objective of this paper is to analyze the successive methodological choices that led to this performance and to discuss the broader lessons that can be drawn for aerodynamic surrogate modeling using coordinate-based neural representations.

Paper organization. The remainder of this paper is organized as follows. Section 2 reviews the most relevant literature on machine-learning surrogate models for computational fluid dynamics. Section 3 presents the CRM-WBPN benchmark and the evaluation protocol, while Section 4 reviews the baseline models provided by the challenge organizers. Section 5 describes the proposed neural-field formulation together with the main components of the fina methodology. Section 6 presents the progressive development of the winning solution through a comprehensive ablation study, evaluates the contribution of each methodological component, and concludes with a discussion of alternative approaches that were explored but ultimately discarded. Section 7 presents the final quantitative and qualitative results on the hidden test set. Finally, Section 8 concludes the paper and outlines several directions for future research.

Table 1: Top-ranked solutions on the final hidden test set of the ONERA CRM Wall Distribution Regression Challenge, as reported on the leaderboard<sup>3</sup>.
<table><tr><td>Rank</td><td>Team / Participant</td><td>Method</td><td>Final Score</td></tr><tr><td>1</td><td>Cenaero (this work)</td><td>Neural field ensembles</td><td>8.8141</td></tr><tr><td>2</td><td>ONERA baseline</td><td>Global MLP</td><td>8.6401</td></tr><tr><td>3</td><td>Air Liquide</td><td>Zonal MLP</td><td>8.4764</td></tr><tr><td>4</td><td>INTA</td><td>VAE + Gaussian process regression</td><td>8.4370</td></tr><tr><td>5</td><td>Centrale Supélec - Transvalor</td><td>Physics informed operator</td><td>8.4254</td></tr></table>

## 2 Related Work

Machine-learning surrogate models have become an active research area for accelerating Computational Fluid Dynamics (CFD) simulations. Their objective is to approximate the mapping between geometric and aerodynamic parameters and the corresponding flow solution while reducing the computational cost by several orders of magnitude. Recent approaches can be broadly categorized into global regression methods, topology-aware neural networks, neural operators, and coordinate-based neural fields.

Early surrogate models relied on classical regression techniques, MLPs, or Convolutional Neural Networks (CNNs) to directly predict aerodynamic quantities from operating conditions. While these approaches can provide accurate predictions for restricted parameter spaces, their ability to represent complex nonlinear flow phenomena on realistic geometries remains limited [2], particularly when high-dimensional output fields must be predicted.

More recently, Graph Neural Networks (GNNs) have emerged as a natural framework for learning CFD solutions on unstructured meshes by explicitly exploiting mesh connectivity. Architectures such as MeshGraphNets demonstrated accurate prediction of transient fluid dynamics and solid mechanics by performing message passing over mesh elements [3]. Several subsequent works have extended graph-based operators to aerodynamic surrogate modeling on complex geometries and irregular meshes [4]. Although these methods effectively exploit mesh topology, they require access to mesh connectivity, which is unavailable in the CRM-WBPN benchmark considered in this work.

In parallel, neural operators have been introduced to directly learn mappings between function spaces rather than finite-dimensional vectors. Fourier Neural Operators (FNO) demonstrated that entire families of PDE solutions can be approximated efficiently through spectral convolutions [5], while subsequent developments such as Geo-FNO extended these ideas to irregular geometries through learned domain deformations [6]. Neural operators have become one of the main paradigms for data-driven PDE solving and surrogate modeling.

Another line of research relies on Implicit Neural Representations (INRs), also known as neural fields, which represent physical quantities as continuous functions of spatial coordinates. Originally developed for representing images, shapes and radiance fields, neural fields have subsequently been extended to PDE surrogate modeling and operator learning [7, 8]. Their continuous and discretization-independent formulation makes them particularly attractive for complex geometries and point-cloud representations. Recent advances, including Fourier feature encoding [9] and sinusoidal representation networks (SIREN) [10], have substantially improved their ability to capture high-frequency signals and localized phenomena.

Neural fields have recently demonstrated excellent performance for aerodynamic surrogate modeling. Catalani et al. [11] proposed an INR framework capable of predicting aerodynamic fields on unstructured meshes while generalizing across unseen geometries. Their results showed that neural fields can outperform state-of-the-art GNNs on several aerodynamic benchmarks while remaining discretization independent. More recently, the MARIO framework further demonstrated the scalability of neural-field surrogates to large industrial aerodynamic datasets through efficient geometric conditioning and multiscale representations [12].

The methodology proposed in this work belongs to the family of conditional neural fields. Unlike previous studies focusing on multiple geometries or complete CFD meshes, we consider the challenge of predicting aerodynamic wall quantities on a fixed but complex aircraft geometry for a limited number of operating conditions and without access to mesh connectivity. The proposed methodology therefore focuses on carefully designing the model formulation and training pipeline, rather than introducing increasingly sophisticated neural architectures.

## 3 Problem Definition and Dataset

## 3.1 Problem Definition

The objective of the challenge is to predict the pressure and friction coefficients on the surface of the NASA CRM configuration under varying aerodynamic conditions. The dataset consists of 468 RANS simulations generated using the Spalart–Allmaras turbulence model. Among the 468 simulations, $n _ { t r } = 3 1 2$ are made available for model development and training, while the remaining $n _ { t e } = 1 4 6$ are retained as a hidden test set (see Figure 2). Predictions on these unseen configurations are evaluated exclusively through the Codabench platform, ensuring an unbiased assessment of model generalization.

Each simulation provides values at $n _ { p } = 2 6 0$ , 774 surface points, defined by their spatial coordinates and surface normals. The flow conditions are parameterized by the Mach number, Angle of Attack (AoA), and stagnation pressure $p _ { i }$ . In the CRM-WBPN database, the Reynolds number is uniquely determined from these variables and therefore does not constitute an additional degree of freedom in the parameter space [1, Section 2.2]. The dataset covers a broad spectrum of aerodynamic regimes, ranging from subsonic to near-stall conditions, and includes transonic cases characterized by complex and highly diverse flow behaviors. The quality of the CFD solutions was assessed through confidence-factor metrics derived from the convergence history of both the residuals and the drag coefficient. Most simulations exhibit a satisfactory level of convergence; however, reduced confidence was observed for a subset of configurations corresponding to high Angles of Attack (AoAs) and low Mach numbers. To account for this variability, each simulation is associated with a confidence weight $w _ { f }$ equal to 1.0 for well-converged cases and 0.5 for lowerconfidence solutions. These weights are subsequently incorporated into the computation of the challenge evaluation metrics (Section 3.2), so reducing the influence of simulations affected by numerical convergence uncertainties. An additional characteristic of the benchmark is that the computational mesh used to generate the CFD solutions is not provided. As a result, surrogate models must be designed independently of mesh connectivity, which strongly influences the choice of machine learning architectures and feature representations.

![](images/03470e70bc4aab89d27538a5e106cad0408d29f6efa86c69ed08ddab9cd045dc.jpg)

![](images/45d896fec01173bb97eb00e20e13a669240cf681bf14a53d1e9597a919c6a568.jpg)

![](images/84e6904180e685de8246ccab4fac5d44b0483a88a317bee394b340e1fb873cdb.jpg)  
Figure 2: Training and hidden testing set repartition among the 468 different simulations of the database.

Formally, the objective is to predict the aerodynamic surface fields:

$$
\mathbf { C } = ( C _ { p } , C _ { f x } , C _ { f y } , C _ { f z } )\tag{1}
$$

as a function of spatial location and flow parameters. Then, the model learns:

$$
\mathbf { C } ( x , y , z ) = f _ { \boldsymbol { \theta } } ( x , y , z , n _ { x } , n _ { y } , n _ { z } , \mathrm { A o A } , M _ { \infty } , p _ { i } )\tag{2}
$$

where $( x , y , z )$ are spatial coordinates, $( n _ { x } , n _ { y } , n _ { z } )$ the surface normal, and $( \mathrm { A o A } , M _ { \infty } , p _ { i } )$ the aerodynamic conditions.

## 3.2 Evaluation Metrics

Model performance is assessed using two complementary metrics: the coefficient of determination $( R ^ { 2 } )$ and the worst-case relative Mean Absolute Error (wrMAE). While the former measures the overall predictive accuracy of the surrogate model across the entire test set, the latter evaluates its robustness by focusing on the most challenging test

configuration. For a given aerodynamic quantity y and its prediction yˆ, the coefficient of determination is defined as

$$
R _ { y } ^ { 2 } = 1 - \frac { \displaystyle \sum _ { b = 0 } ^ { n _ { t e } } \sum _ { i = 0 } ^ { n _ { p } } w _ { f } ^ { b } \left( y _ { b , i } - \hat { y } _ { b , i } \right) ^ { 2 } } { \displaystyle \sum _ { b = 0 } ^ { n _ { t e } } \sum _ { i = 0 } ^ { n _ { p } } w _ { f } ^ { b } \left( y _ { b , i } - \bar { y } \right) ^ { 2 } } ,
$$

with

$$
\bar { y } = \frac { 1 } { n _ { t e } n _ { p } } \sum _ { b = 0 } ^ { n _ { t e } } \sum _ { i = 0 } ^ { n _ { p } } y _ { b , i } ,
$$

where $n _ { t e }$ denotes the number of test simulations, $n _ { p }$ the number of surface points, and $\boldsymbol { w } _ { f } ^ { b }$ the confidence weight associated with simulation b. The metric is computed independently for each target variable, yielding $R _ { C _ { p } } ^ { 2 } , R _ { C f _ { x } } ^ { 2 }$ $R _ { C f _ { y } } ^ { 2 }$ , and $R _ { C f _ { z } } ^ { 2 }$ . The global coefficient of determination is then obtained by averaging the four quantities,

$$
R ^ { 2 } = \frac { R _ { C _ { p } } ^ { 2 } + R _ { C f _ { x } } ^ { 2 } + R _ { C f _ { y } } ^ { 2 } + R _ { C f _ { z } } ^ { 2 } } { 4 } .
$$

To complement this global measure of accuracy, the challenge introduces a worst-case relative Mean Absolute Error (wrMAE). For a given simulation b, the relative mean absolute error is then given by

$$
\mathrm { r M A E } _ { y } ^ { b } = \frac { \displaystyle \sum _ { i = 0 } ^ { n _ { p } } \left| y _ { i , b } - \hat { y } _ { i , b } \right| } { \displaystyle \sum _ { i = 0 } ^ { n _ { p } } \left| y _ { i , b } \right| } .
$$

The corresponding worst-case relative mean absolute error is then defined as

$$
\mathrm { w r M A E } _ { y } = \operatorname* { m a x } _ { b \in [ 1 , n _ { t e } ] } \mathrm { r M A E } _ { y } ^ { b } .
$$

Similarly to the $R ^ { 2 }$ metric, wrMAE is computed independently for each predicted quantity, yielding $\mathrm { w r M A E } _ { C _ { p } }$ wr $\cdot { \mathrm { M A E } _ { C f _ { x } } }$ , w $\cdot { \mathrm { M A E } _ { C f _ { y } } }$ , w $\cdot { \mathrm { M A E } _ { C f _ { z } } }$ , and then averaged over the four aerodynamic variables,

$$
\mathrm { w r M A E } = \frac { \mathrm { w r M A E } _ { C _ { p } } + \mathrm { w r M A E } _ { C f _ { x } } + \mathrm { w r M A E } _ { C f _ { y } } + \mathrm { w r M A E } _ { C f _ { z } } } { 4 } .
$$

The final challenge score combines both metrics according to

$$
\mathrm { S c o r e } = 5 \times \mathrm { R } ^ { 2 } + 5 \times ( 1 - \mathrm { w r M A E } ) .\tag{3}
$$

The maximum achievable score is 10, corresponding to perfect predictions with $R ^ { 2 } = 1$ and $\mathrm { w r M A E } = 0$

Metrics interpretation. The evaluation protocol is designed to jointly assess predictive accuracy and robustness. The $R ^ { 2 }$ metric quantifies the fraction of variance explained by the surrogate model and therefore measures its overall ability to reproduce the aerodynamic fields over the complete test set. Values close to one indicate excellent agreement with the reference CFD solutions, whereas values near or below zero correspond to models that perform no better than a constant predictor.

In contrast, the wrMAE metric focuses exclusively on the most difficult test configuration by considering the largest relative error observed across all simulations. As a result, minimizing wrMAE requires the model to maintain satisfactory performance even in regions of the parameter space associated with complex flow phenomena, such as transonic shocks or near-stall conditions. Optimizing only for global accuracy may therefore be insufficient, since large errors on a small number of challenging configurations can significantly degrade the final score.

From a statistical perspective, the combined score emphasizes models that not only capture the central tendency and variance of the data distribution, as reflected by the $\hat { R ^ { 2 } }$ metric, but also provide robust predictions in the tails of the distribution. The benchmark therefore rewards surrogate models capable of balancing average predictive performance with reliability on the most demanding aerodynamic conditions.

## 3.3 Obstacles

The challenge presents several difficulties that make the development of accurate surrogate models particularly demanding. These obstacles arise from both the physical complexity of the underlying aerodynamic problem and the practical constraints imposed by the competition framework.

A first challenge is the limited amount of training data available relative to the complexity of the target problem. Although each simulation contains aerodynamic quantities at more than $2 . 6 \times 1 0 ^ { 5 }$ surface locations, only 312 CFD configurations are provided for model development. The dataset must therefore capture a high-dimensional mapping between aerodynamic operating conditions and wall distributions using a relatively small number of independent flow realizations. This situation is particularly challenging given the large variability of the flow physics represented in the database.

The diversity of aerodynamic regimes constitutes another major difficulty. The dataset covers Mach numbers ranging from 0.3 to 0.96 and AoA between $- 1 5 ^ { \circ }$ and $1 5 ^ { \circ }$ , spanning subsonic, transonic, and near-stall conditions. As a consequence, the surrogate model must simultaneously learn smooth flow regimes and highly nonlinear phenomena such as shock waves, strong pressure gradients, flow separation, and the onset of stall. Several high-angle-of-attack configurations also exhibit increased drag coefficient fluctuations, indicating emerging flow instabilities and reduced numerical convergence confidence. Accurately capturing such heterogeneous flow behaviors within a single model is inherently challenging.

The geometric complexity of the NASA/Boeing Common Research Model further increases the difficulty of the task. Unlike simplified wing-only benchmarks, the CRM-WBPN configuration includes the wing, fuselage, pylon, and nacelle components, resulting in a complex three-dimensional geometry with intricate local flow interactions. Moreover, the aircraft surface cannot be naturally mapped onto a regular Cartesian grid without introducing significant distortion [13]. This prevents the direct application of conventional CNN architectures that have demonstrated strong performance on structured image-like data.

An additional constraint is the absence of the CFD mesh from the released dataset. Since mesh connectivity information is unavailable, machine-learning approaches based on graph or mesh operators cannot be directly applied without reconstructing an approximate connectivity structure. Surrogate models must therefore rely exclusively on geometric descriptors, surface coordinates, and flow parameters, which naturally favors mesh-independent representations.

The characteristics of the target fields themselves also contribute to the complexity of the problem. Pressure and skin-friction distributions exhibit spatial structures spanning multiple scales, ranging from large smooth regions to highly localized discontinuities associated with shocks and separation phenomena. Such multiscale behavior is known to be difficult for standard neural networks due to their tendency to preferentially learn low-frequency features, a phenomenon commonly referred to as spectral bias.

Finally, the competition framework imposes additional methodological constraints. Participants were limited to twenty submissions on the hidden test set throughout the duration of the challenge. Consequently, model selection and hyperparameter optimization could not rely on repeated leaderboard probing. A possible solution is the construction of a sufficiently representative local validation strategy but this further reduced the amount of data available for training, introducing a trade-off between model development and reliable performance estimation. Furthermore, the evaluation score combines a global accuracy metric $( R ^ { 2 } )$ with a worst-case error criterion (wrMAE), requiring models not only to achieve strong average performance but also to remain robust on the most challenging flow configurations. This dual objective significantly increases the difficulty of the challenge and strongly influences the design of both training objectives and model architectures.

## 4 Reference Baseline Models

To establish reference performance levels for the CRM-WBPN benchmark, Peter et al. [1] evaluated several machinelearning approaches, including both pointwise and global regression strategies. Among the organizer-provided baselines, the best-performing model was a global Multi-Layer Perceptron (Global MLP), also denoted as a Modewise MultiLayer Perceptron in the original publication [1, Section 5.1]. A pointwise regression model was also investigated and is arguably the baseline most closely related to the neural-field formulation proposed in this work. Nevertheless, its performance remained significantly below that of the Global MLP.

The benchmark study considered a total of seven surrogate modeling approaches. A subset of these methods had previously been developed and evaluated by the organizers in [14] on a closely related aerodynamic surrogate-modeling benchmark involving a wing geometry subjected to varying operating conditions. The pointwise methods included a standard MLP, a λ-DNN architecture specifically designed to better couple geometric information with aerodynamic operating conditions, and a decision-tree regressor. The global approaches comprised the Global MLP, a k-Nearest-Neighbor (kNN) interpolation method operating in the input parameter space, an IsoMap-based dimensionality reduction combined with Radial Basis Function (RBF) interpolation, and a reduced-order modeling strategy based on Proper Orthogonal Decomposition (POD) coupled with RBF regression. Together, these baselines cover a broad range of surrogate modeling philosophies, from local pointwise prediction to global field reconstruction through nonlinear manifold learning and reduced-order representations.

## 4.1 Global MLP Baseline

The Global MLP adopts a direct field-regression strategy. Instead of predicting aerodynamic quantities independently at each surface location, the model learns a mapping from the aerodynamic operating conditions to the complete wall distribution of a given target quantity. For each aerodynamic field, a dedicated neural network is trained to approximate the mapping

$$
f _ { \theta } : ( { \cal M } _ { \infty } , { \cal A } o { \cal A } , p _ { i } )  { \bf C } ,\tag{4}
$$

where C denotes the full field containing the values of either $C _ { p } , C _ { f _ { x } } , C _ { f _ { y } } , \mathrm { o r } C _ { f _ { z } }$ at all surface locations. Consequently, four independent neural networks are required to predict the complete set of challenge outputs. Unlike pointwise regressors, which learn local relationships at individual surface locations, the Global MLP directly models the entire aerodynamic field as a high-dimensional output vector. This formulation allows the network to exploit correlations between distant regions of the aircraft surface and implicitly learn global flow structures.

Because the output dimension exceeds $2 . 6 \times 1 0 ^ { 5 }$ values per simulation, direct field prediction leads to an exceptionally large regression problem. Following a dedicated hyperparameter search, Peter et al. [1] selected a Global MLP architecture with hidden-layer dimensions (75, 120, 1226, 16490). Owing to the very large output layer, each network contains approximately $4 \dot { \times } 1 0 ^ { 9 }$ trainable parameters. As four independent networks are required to predict $C _ { p } , C _ { f _ { x } }$ $C _ { f _ { y } }$ , and $C _ { f _ { z } }$ , the overall baseline comprises more than $1 . 7 \times 1 0 ^ { 1 0 }$ parameters, making it several orders of magnitude larger than the neural-field models considered in this work.

## 4.2 Strengths and Limitations

The Global MLP baseline presents several attractive characteristics. First, it directly predicts the target fields without requiring any intermediate representation, dimensionality reduction technique, or geometric preprocessing. The approach therefore remains conceptually simple and straightforward to implement.

Second, by treating the complete aerodynamic field as a single output object, the model can implicitly learn large-scale correlations across the aircraft surface. Such correlations are difficult to capture with purely local regression strategies.

However, this formulation also exhibits several limitations. Most importantly, the geometry itself is not explicitly provided to the network. The model only receives the aerodynamic operating conditions as inputs and must therefore infer the entire wall distribution from a low-dimensional parameter space. Consequently, all geometric information is embedded implicitly within the training data and the learned network weights.

Furthermore, the direct prediction of more than 260, 000 output values leads to a very large regression problem. While the network can successfully capture dominant flow trends, representing localized phenomena such as shock waves, separation regions, and sharp pressure gradients remains challenging. These features often correspond to highly nonlinear and localized variations of the aerodynamic fields, which are difficult to reconstruct accurately through a purely global parameter-to-field mapping.

## 4.3 Performance Analysis

Despite its conceptual simplicity, the Global MLP constitutes a remarkably strong baseline. According to the results reported by Peter et al. [1], it achieved the best performance among the seven baseline models evaluated by the organizers, both in terms of the global accuracy metric $( R ^ { 2 } )$ and the worst-case relative error. The authors attribute this performance, at least partially, to the exceptionally large number of trainable parameters of the model, which may provide sufficient capacity to learn the highly nonlinear relationship between operating conditions and aerodynamic wall distributions.

Nevertheless, the qualitative analysis presented by Peter et al. [1] reveals several limitations of the Global MLP. Although the global error metrics are relatively high, some important flow features remain poorly reconstructed. In particular, the pressure coefficient and skin-friction predictions fail to accurately reproduce the transonic shock observed on the upper wing surface for certain operating conditions. These observations suggest that strong global metrics do not necessarily guarantee an accurate representation of localized aerodynamic phenomena, especially when sharp discontinuities and spatial high-frequency structures are involved. The methodology proposed in this paper, introduced in Section 5, is explicitly designed to address these limitations by improving the reconstruction of small-scale and localized flow structures.

## 5 Methodology

The proposed final solution is based on a coordinate-based neural field representation trained to predict aerodynamic wall quantities directly from geometric coordinates and operating conditions. The approach was specifically designed to address the constraints of the CRM-WBPN benchmark, namely the absence of mesh connectivity information, the large number of surface points, and the presence of multiscale flow phenomena ranging from smooth pressure distributions to localized shock structures. The methodology developed in this work is summarized in Figure 3.

![](images/661137f6179beaec9223ac9cc16da4aece4a1b0a808ef4fe6c407ec1910850cc.jpg)  
Figure 3: Methodology overview.

## 5.1 Neural Field Formulation

Neural fields, also referred to as Implicit Neural Representations (INRs), model a physical quantity as a continuous function parameterized by a neural network. Instead of storing the values of a field on a predefined mesh or grid, the network directly learns a mapping from spatial coordinates to the corresponding field values [15]. This representation has recently emerged as a powerful paradigm for modeling complex signals and geometric objects, due to its ability to represent continuous functions with arbitrary spatial resolution.

The neural field formulation can naturally be extended by conditioning the network on additional parameters describing the physical system. In the context of surrogate modeling, these conditioning variables typically correspond to boundary conditions, material properties, or operating parameters [16]. Such coordinate-based representations have been successfully applied to the approximation of PDE solutions and operator learning problems on complex geometries [7]. They also constitute the foundation of several Physics-Informed Neural Network (PINN) formulations, where the network output is constrained through the governing equations of the underlying physical system [17].

In the present work, let $\mathbf { x } = ( x , y , z )$ denote the spatial coordinates of a point located on the aircraft surface, n = $( n _ { x } , n _ { y } , n _ { z } )$ the corresponding surface normal vector, and $\mathbf { p } = \left( M _ { \infty } , A o A , p _ { i } \right)$ the aerodynamic operating conditions. The objective is to learn the continuous mapping

$$
f _ { \theta } : ( { \mathbf { x } } , { \mathbf { n } } , { \mathbf { p } } ) \longrightarrow \left( C _ { p } ( { \mathbf { x } } ) , C _ { f _ { x } } ( { \mathbf { x } } ) , C _ { f _ { y } } ( { \mathbf { x } } ) , C _ { f _ { z } } ( { \mathbf { x } } ) \right) ,\tag{5}
$$

where θ denotes the trainable parameters of the neural network. The model therefore predicts the pressure coefficient and the three components of the skin-friction coefficient at any location on the aircraft surface.

Motivation. Several characteristics of the challenge data make neural fields particularly well suited to the problem. First, the dataset is provided as an unstructured collection of surface points associated with geometric descriptors, while the underlying CFD mesh connectivity is not available. As a result, methods that explicitly rely on mesh topology or graph connectivity cannot be directly applied. In contrast, neural fields operate directly on point coordinates and geometric features, without requiring any connectivity information. This mesh-independent formulation enables the modeling of physical fields on irregular and geometrically complex configurations, such as the CRM-WBPN geometry considered in this challenge, while remaining agnostic to the discretization.

A second advantage of neural fields lies in their continuous representation of the solution. Although not required by the challenge evaluation protocol, the learned field can in principle be queried at arbitrary spatial locations, independently of the point distribution used during training. More importantly, the coordinate-based formulation provides a flexible framework for representing aerodynamic quantities exhibiting a wide range of spatial scales, from smooth variations over large portions of the aircraft surface to localized flow structures associated with shocks, strong pressure gradients, or separated regions.

Finally, conditioning the neural field on the aerodynamic operating conditions naturally leads to a parametric surrogate model of the flow solution manifold. From this perspective, the approach can be viewed as an instance of operator learning, where the network learns a continuous mapping from operating conditions to aerodynamic fields [8, 18]. The resulting model is able to interpolate continuously within the parameter space and predict pressure and friction distributions for previously unseen flight conditions, which is precisely the objective of the challenge.

## 5.2 Fourier Feature Encoding

A well-known limitation of MLPs is their spectral bias [19], which causes them to preferentially learn low-frequency functions while struggling to represent high-frequency features. This limitation is particularly problematic for aerodynamic applications where shock waves and separation regions generate localized discontinuities and sharp gradients. To alleviate this issue, the spatial coordinates are first transformed through a Fourier Feature Encoding (FFE) inspired by the positional encodings commonly used in neural fields. For an input coordinate vector x, the encoded representation is given by

$$
\gamma ( { \bf x } ) = \left[ \sin ( 2 \pi B { \bf x } ) , \cos ( 2 \pi B { \bf x } ) \right] ,
$$

where $\boldsymbol { B } \in \mathbb { R } ^ { N _ { e } \times 3 }$ is a matrix of random frequencies whose entries are independently sampled from a zero-mean Gaussian distribution with variance $\sigma ^ { 2 }$ . The parameter $\sigma ^ { 2 }$ controls the frequency spectrum represented by the FFE and therefore determines the range of spatial scales that can be captured by the model. The encoded coordinates are subsequently concatenated with the aerodynamic operating conditions and normal before being processed by the neural network. This representation enriches the input space with multiple spatial frequencies and significantly improves the ability of the model to capture localized flow structures. The encoding hyperparameters were selected empirically based on validation experiments. In particular, the variance of the frequency distribution was fixed to $\sigma ^ { 2 } = 1$ , while the encoding dimension was set to $N _ { e } = 1 2 8$

## 5.3 Network Architecture

The surrogate model is implemented as a fully connected MLP that receives as input the Fourier-encoded spatial coordinates (Section 5.2), the surface normal vector, and the aerodynamic operating conditions. The network jointly predicts the four target quantities $( C _ { p } , C f _ { x } , C f _ { y } , C f _ { z } )$ . Adopting a multi-output formulation enables the model to exploit the physical correlations that exist between pressure and skin-friction fields, while avoiding the additional computational cost associated with training separate networks for each quantity.

Normalization. All input and output variables are normalized prior to training. The spatial coordinates are linearly scaled to the interval [−1, 1], while the surface normal vector is already normalized by construction. The Mach number $M _ { \infty }$ and stagnation pressure $p _ { i }$ are scaled to [0, 1], whereas the AoA is mapped to [−1, 1]. Regarding the target variables, the pressure coefficient field is standardized to zero mean and unit variance, while the three skin-friction components are linearly scaled to the interval [−1, 1]. These normalization strategies were selected empirically based on validation experiments and were found to improve both training stability and predictive accuracy. In particular, bringing all variables to comparable numerical ranges facilitates optimization and reduces the risk of gradient imbalance between the different inputs and outputs.

MLP Architecture. The final architecture consists of a deep feed-forward network with five hidden layers, each containing 700 neurons, resulting in approximately $2 . 1 5 \times 1 0 ^ { 6 }$ trainable parameters. A LeakyReLU activation function is applied after each hidden layer to introduce nonlinearity while maintaining stable gradient propagation throughout the network. Several architectures with varying depths and widths were evaluated during the development process. While increasing the network size generally led to modest improvements in validation performance, the gains became marginal once the number of trainable parameters exceeded approximately $2 . 1 5 \times 1 0 ^ { 6 }$ . The selected architecture therefore represents a favorable compromise between model capacity, computational efficiency, and generalization performance.

## 5.4 Loss Function

Preliminary experiments revealed that optimizing a conventional mean squared error objective produced models with strong $R ^ { 2 }$ scores but suboptimal performance with respect to the challenge ranking metric. Since the leaderboard score explicitly depends on the relative prediction error through the wrMAE metric, a better alignment between training and evaluation objectives was required. The final model was therefore trained using a Relative Squared Error (RSE) loss defined as

$$
\mathcal { L } _ { R S E } ( y , \hat { y } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { ( y _ { i } - \hat { y } _ { i } ) ^ { 2 } } { y _ { i } ^ { 2 } + \varepsilon } ,
$$

where $\varepsilon$ is a small numerical constant introduced for stability, $y$ and $\hat { y }$ are the ground truth and prediction respectively. Compared with the standard MSE, this formulation assigns a larger weight to relative prediction accuracy and was found to substantially improve the wrMAE metric, which ultimately resulted in higher leaderboard scores.

## 5.5 Optimization Strategy

Model training was performed using the SOAP optimizer [20], combined with a learning-rate schedule consisting of an initial linear warm-up phase followed by cosine annealing [21]. All experiments were conducted on a single NVIDIA A100 GPU. The validation loss was monitored throughout training, and early stopping was employed to prevent overfitting and reduce unnecessary computation once the validation performance ceased to improve. The strategy used to partition the available data into training and validation subsets is described in Section 5.7. The initial learning rate was set to $1 0 ^ { - 2 }$ and the batch size $\mathbf { t o } 2 \times n _ { p }$ , where $n _ { p }$ denotes the number of surface points per configuration. These hyperparameters were selected empirically based on validation performance. The early-stopping patience was fixed to 20 epochs, meaning that training was terminated after 20 consecutive epochs without improvement in the validation loss.

The SOAP optimizer was retained following comparative experiments against AdamW. While both optimizers achieved similar levels of predictive accuracy, SOAP consistently provided slightly improved validation performance with only a modest increase in computational cost. Unlike AdamW, which relies exclusively on first-order gradient information, SOAP incorporates a second-order preconditioning strategy based on an approximation of the Hessian matrix. This additional curvature information improves the conditioning of the optimization problem and can lead to faster convergence and enhanced optimization accuracy.

## 5.6 Ensemble Learning

Although the single neural-field model achieved strong predictive performance, a noticeable discrepancy remained between validation and hidden-test results, particularly for the worst-case error metric. To improve robustness, multiple independently trained neural fields can be combined through ensemble averaging (see, e.g., [22]). Let $f _ { \theta _ { 1 } } , \dots , f _ { \theta _ { K } }$ denote K independently trained models. The final prediction is then obtained as

$$
\hat { y } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } f _ { \theta _ { k } } ( x ) .\tag{6}
$$

Averaging predictions reduces variance and mitigates individual model errors, leading to improved generalization and lower worst-case prediction errors.

## 5.7 K-Fold Cross-Validation Ensemble

The final competition submission relies on a cross-validation ensemble strategy. The 312 available simulations are partitioned into five folds. For each fold, a dedicated training-validation split is constructed and several neural fields are trained using different random initializations.

The 20 best-performing models across all folds are retained and combined into a final ensemble, corresponding to approximately $7 . 3 \times 1 0 ^ { 7 }$ trainable parameters.

This strategy increases the diversity of the ensemble while allowing every simulation to contribute to model training. The resulting predictor achieved the highest score on the final hidden test set and constitutes the final solution presented in this work.

## 6 Progressive Development and Ablation Study

This section analyzes the successive methodological choices that led to the final winning solution. Starting from a simple neural-field baseline, each modification was introduced to address a specific limitation identified through validation experiments. The objective is not only to justify the final architecture but also to provide practical insights into the design of surrogate models for aerodynamic wall-field prediction.

## 6.1 Initial Neural Field Prototype

The development process began with a simple coordinate-based neural field implemented as a MLP, following the reasoning of the formulation introduced in Section 5.1. Adopting a simple MLP architecture allowed us to establish a lightweight and easily reproducible baseline, facilitating rapid experimentation and the progressive evaluation of subsequent architectural improvements. The initial prototype consisted of a fully connected MLP with five hidden layers of 128 neurons each and LeakyReLU activation functions, resulting in approximately $6 . 7 \times 1 0 ^ { 4 }$ trainable parameters. A single network was trained to jointly predict the four target quantities, as preliminary experiments outperformed the alternative strategy of training one model per field. This behavior suggests that jointly learning the pressure and skin-friction fields enables the network to exploit the physical correlations between these quantities. The model was trained using a weighted MSE loss, with the relative weights between the four outputs selected empirically, and optimized using AdamW. A total of 260 simulations were used for training, while the remaining 52 configurations were reserved for validation.

The performance of this initial prototype is summarized in Table 2. Although the model produced encouraging predictions $( R ^ { 2 } > 0 . 9 0 )$ despite its modest size, it remained substantially less accurate than both the organizer’s point-wise MLP and Global MLP baselines, which comprise approximately $1 . 2 5 \times 1 0 ^ { 5 }$ and $1 . 7 \times 1 0 ^ { 1 0 }$ trainable parameters, respectively. This performance gap initially suggested that increasing the model capacity could improve predictive accuracy. More importantly, qualitative inspection of the predicted fields revealed a limited ability to reproduce localized flow structures, particularly shock waves. This observation highlighted the spectral limitations of conventional MLPs and motivated the introduction of Fourier feature encoding, described in the next subsection.

## 6.2 Fourier Feature Encoding

The initial neural-field prototype demonstrated that a simple coordinate-based MLP was capable of learning the overall structure of the aerodynamic fields. However, visual inspection of the predictions consistently revealed difficulties in reproducing localized flow structures. This limitation was observed despite increasing the network size, suggesting that the bottleneck was not solely related to model capacity but also to the spectral bias mentioned in Section 5.2. Since aerodynamic wall quantities simultaneously exhibit smooth global trends and localized discontinuities, particularly in transonic regimes, enriching the spatial representation appeared more promising than only increasing the number of trainable parameters.

To address the spectral bias, the Cartesian coordinates were replaced by the Fourier Feature Encoding (FFE) introduced in Section 5.2. The same encoding was also applied to the surface normal vectors, but as it yielded no measurable improvement in predictive performance, the original representation was retained.

The introduction of FFE substantially improved both the $R ^ { 2 }$ score and the wRMAE, as summarized in Table 2, resulting in an increase of the overall score from 7.80 to 8.34. The most pronounced qualitative improvements were observed in regions exhibiting localized flow phenomena, including transonic shock waves and abrupt variations in the skin-friction fields, where sharp discontinuities and gradients were reproduced much more accurately. In addition to the ability of FFE to alleviate MLP spectral bias, FFE considerably increases the expressive power of the model while preserving the network architecture and adding only a negligible number of trainable parameters.

## 6.3 Increasing Model Capacity

Following the introduction of FFE, the next objective was to determine whether the model remained capacity-limited. The network width was therefore progressively increased while keeping the overall architecture unchanged. As summarized in Table 2, increasing the model capacity consistently improved predictive performance, raising the overall score from 8.34 to 8.45 and yielding improvements in both the $\dot { R } ^ { 2 }$ and wRMAE metrics for all predicted quantities except wr $\mathrm { M A E } _ { C f _ { y } }$

While larger networks were better able to capture the highly nonlinear mapping between the geometric coordinates, operating conditions, and aerodynamic fields (final single-model architecture containing approximately $2 . 1 5 \times 1 0 ^ { 6 }$ trainable parameters), further increasing the model size resulted in marginal performance gains but substantial increase in the computational cost. This architecture was therefore retained as the best compromise between predictive accuracy and computational efficiency.

## 6.4 Metric-Aligned Loss Function

Despite the improvements obtained through FFE and increased model capacity, the validation analysis revealed an important discrepancy between the optimization objective and the competition evaluation metric. Models trained with a conventional MSE loss achieved excellent $R ^ { 2 }$ values but remained comparatively weak on the wrMAE metric. This behavior can be explained by the fact that the challenge score explicitly incorporates a relative error measure. Large errors on a small number of challenging configurations therefore have a disproportionate impact on the final ranking, even when the global prediction accuracy remains high.

To better align the training objective with the evaluation protocol, the mean squared error was replaced by a Relative Squared Error (RSE) loss as described in Section 5.4. This formulation normalizes the prediction error by the magnitude of the target quantity, encouraging the network to minimize relative rather than absolute errors. It also places the loss contributions of the different target fields on a comparable scale, eliminating the need for manually tuned weighting coefficients.

The impact of this modification was immediate. While the $R ^ { 2 }$ metric remained essentially unchanged (from 0.965 to 0.967), a substantial reduction of the wrMAE was observed decreasing from 0.276 to 0.251. The wrMAE was improved for all the fields except for $C f _ { x }$ which remained unchanged. The overall validation score increased from approximately 8.45 to 8.58, demonstrating that loss-function design can be as important as architectural improvements when the evaluation metric emphasizes robustness.

## 6.5 Ensemble Learning

Although the single neural-field model achieved strong validation performance, a persistent gap remained between the validation and hidden-test scores. This discrepancy was primarily attributable to the wrMAE metric, indicating that a small number of hidden-test predictions were less accurate. To improve the robustness of the predictions, an ensemble learning strategy was therefore introduced, as described in Section 5.6. Multiple neural-field models sharing the same architecture and training dataset, but initialized with different random seeds, were trained independently. The final prediction was obtained by averaging the outputs of the individual models. The underlying hypothesis is that the prediction errors of independently initialized models are only partially correlated. Consequently, configurations that are poorly predicted by one model may be better approximated by another, allowing the averaging process to reduce the influence of individual prediction failures. From a statistical perspective, ensemble averaging reduces the variance of the estimator while preserving its bias, resulting in more robust predictions and improved generalization on previously unseen operating conditions.

A seven-model ensemble, corresponding to approximately $1 . 5 \times 1 0 ^ { 7 }$ trainable parameters, was initially constructed. As reported in Table 2, ensemble averaging improved both the $R ^ { 2 }$ and wrMAE metrics, increasing the hidden-test score from 8.58 to 8.65. Nevertheless, the improvement remained moderate, suggesting that the dominant source of error was no longer the random initialization of the models but rather the limited diversity of the training data. Since all ensemble members were trained on the same train-validation split, they were exposed to the same distribution of aerodynamic configurations and therefore tended to share similar weaknesses. This observation motivated the investigation of a k-fold cross-validation strategy, presented in Section $6 . 6 ,$ to increase the diversity of the ensemble while exploiting the entire training dataset.

## 6.6 K-Fold Cross-Validation Ensemble

A detailed analysis of the hidden-test predictions revealed that the largest errors were concentrated in a small number of configurations characterized by relatively high Mach numbers and large Angles of Attack $( M _ { \infty } = 0 . 8 5 , A o A = 8 . 0 ^ { \circ }$ $p _ { i } = 4 . 0 \times 1 0 ^ { 5 }$ Pa for example). These regions correspond to complex transonic flow conditions where shocks and partial flow separation become increasingly prominent. This observation suggested that part of the remaining error could be attributed to limitations of the training-validation split. Given the relatively small number of available simulations, certain regions of the parameter space may be underrepresented in a single training partition.

To mitigate this effect, a five-fold cross-validation strategy was adopted. The 312 available simulations were partitioned into five different train-validation configurations. For each fold, 8 neural-field models were trained using different random initializations. The best-performing models from all folds were subsequently aggregated into a single ensemble. This approach offers two complementary advantages. First, every simulation contributes to model training in multiple folds, resulting in a more efficient use of the available data. Second, the diversity of the resulting ensemble is significantly increased because it contains models that are exposed to a different subset of training examples.

The final ensemble combines twenty neural-field models corresponding to approximately 73 million trainable parameters. This strategy produced a significant improvement, reducing the wrMAE from 0.241 to 0.210 while simultaneously increasing the $R ^ { 2 }$ score. The resulting leaderboard score reached 8.8141 and ultimately secured first place in the challenge.

## 6.7 Summary of Performance Improvements

Table 2 summarizes the successive improvements obtained throughout the development process. The results highlight that the winning solution emerged from the combination of several complementary ingredients: improved spatial encoding through FFE, tailored model capacity, metric-aligned optimization, ensemble regularization, and crossvalidation-based data utilization.

These results suggest that, for the CRM-WBPN benchmark, improvements in robustness and generalization were ultimately more beneficial than further increases in model complexity. In particular, the most significant gains were obtained through techniques specifically targeting the worst-case error metric, which plays a central role in the challenge evaluation protocol.

Table 2: Progressive improvement of the solution throughout the development process.
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>Initod ol rotd type</td><td rowspan=1 colspan=1>+Fog  rg FFE)</td><td rowspan=1 colspan=1>ncaded cc ace +</td><td rowspan=1 colspan=1>el osə es eaoss+s</td><td rowspan=1 colspan=1>+nne enqnng</td><td rowspan=1 colspan=1>rnso o-os - +</td></tr><tr><td rowspan=1 colspan=1>Score ↑</td><td rowspan=1 colspan=2>7.80</td><td rowspan=1 colspan=1>8.34</td><td rowspan=1 colspan=1>8.45</td><td rowspan=1 colspan=1>8.58</td><td rowspan=1 colspan=1>8.65</td><td rowspan=1 colspan=1>8.81</td></tr><tr><td rowspan=1 colspan=1> $\overline { { R ^ { 2 } \uparrow } }$ </td><td rowspan=1 colspan=2>0.908</td><td rowspan=1 colspan=1>0.943</td><td rowspan=1 colspan=1>0.965</td><td rowspan=1 colspan=1>0.967</td><td rowspan=1 colspan=1>0.970</td><td rowspan=1 colspan=1>0.973</td></tr><tr><td rowspan=1 colspan=1> $R _ { C _ { p } } ^ { 2 }$  $R _ { C f _ { x } } ^ { 2 }$  $R _ { C f _ { y } } ^ { 2 }$  $R _ { C f _ { z } } ^ { 2 }$ </td><td rowspan=1 colspan=2>0.9640.8910.8900.888</td><td rowspan=1 colspan=1>0.9780.9220.9310.942</td><td rowspan=1 colspan=1>0.9790.9560.9600.966</td><td rowspan=1 colspan=1>0.9800.9580.9630.968</td><td rowspan=1 colspan=1>0.9820.9620.9660.971</td><td rowspan=1 colspan=1>0.9830.9660.9690.973</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { w r M A E \downarrow }$ </td><td rowspan=1 colspan=2>0.349</td><td rowspan=1 colspan=1>0.276</td><td rowspan=1 colspan=1>0.276</td><td rowspan=1 colspan=1>0.251</td><td rowspan=1 colspan=1>0.241</td><td rowspan=1 colspan=1>0.210</td></tr><tr><td rowspan=3 colspan=1> $\mathrm { w r M A E } _ { C _ { p } }$  $\mathrm { w r M A E } _ { C f _ { x } }$  $\mathrm { w r M A E } _ { C f _ { y } }$  $\mathrm { w r M A E } _ { C f _ { z } }$ </td><td rowspan=2 colspan=2>0.2860.2360.436</td><td rowspan=1 colspan=1>0.2000.194</td><td rowspan=1 colspan=1>0.1710.179</td><td rowspan=1 colspan=1>0.1620.179</td><td rowspan=1 colspan=1>0.1600.173</td><td rowspan=1 colspan=1>0.1570.170</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>0.393</td><td rowspan=1 colspan=1>0.471</td><td rowspan=1 colspan=1>0.409</td><td rowspan=2 colspan=1>0.3980.233</td><td rowspan=2 colspan=1>0.3150.197</td></tr><tr><td rowspan=1 colspan=2>0.439</td><td rowspan=1 colspan=1>0.317</td><td rowspan=1 colspan=1>0.282</td><td rowspan=1 colspan=1>0.253</td></tr></table>

## 6.8 Alternative Approaches Investigated

Several alternative modeling strategies were explored before arriving at the final approach presented in this paper. Although some of these approaches appeared promising from a theoretical perspective, none ultimately provided a favorable trade-off between predictive performance, implementation complexity, and robustness compared with the final neural-field ensemble. This section summarizes the main directions investigated and the lessons learned from these exploratory experiments.

Mesh reconstruction and graph neural networks. Graph Neural Networks (GNNs) constitute a natural candidate for CFD surrogate modeling, as they can exploit mesh connectivity information to propagate geometric and physical information across neighboring surface elements. Several recent studies have demonstrated the effectiveness of graph-based operators for learning aerodynamic quantities directly on unstructured meshes [23, 24].

However, the CRM-WBPN benchmark [1] does not provide the computational mesh used to generate the CFD solutions. Only point coordinates and surface normals are available. Applying graph-based methods therefore requires reconstructing an approximate connectivity structure from the point cloud. Preliminary investigations were conducted using nearest-neighbor graphs and geometric proximity criteria. While these approaches produced visually plausible connectivity patterns, the resulting graphs remained only rough approximations of the original CFD mesh and introduced additional hyperparameters related to graph construction.

Given the limited number of available simulations and the substantial implementation effort required, graph-based approaches were not pursued further. In contrast, neural fields operate directly on point coordinates and naturally avoid the need for mesh reconstruction.

Parametric zonal decomposition strategies. A natural hypothesis was that the mapping between operating conditions and aerodynamic fields could be simplified by partitioning the parameter space into regions associated with different flow regimes and training a dedicated surrogate model for each region. Such an approach could reduce the complexity of the learning task by allowing each model to specialize in a narrower range of aerodynamic behaviors. Several partitioning strategies based on the AoA were therefore investigated. A first decomposition separated negative and positive incidence configurations (AoA $< - 1 ^ { \circ }$ and $A o A \geq - 1 ^ { \circ } )$ . A second strategy exploited prior knowledge of the CFD database by distinguishing three regions: AoA $< - 4 ^ { \circ } , - 4 ^ { \circ } \leq A o A \leq 5 ^ { \circ }$ , and $\bar { A } O A \bar { > } 5 ^ { \circ }$ . These thresholds were selected after analyzing the standard deviation of the drag coefficient during the last 20% of the CFD iterations, which remains low within the central range but increases markedly at larger positive and negative incidences, indicating the progressive onset of flow instabilities and partial stall conditions.

Although this decomposition reduced the variability of the flow within each subdomain, it also substantially decreased the number of training simulations available for each individual model. Given the already limited size of the dataset, the loss of training data outweighed the potential benefit of specialization, and no consistent improvement was observed on the validation score. Consequently, parametric zonal decomposition strategies were not retained in the final methodology.

Alternative coordinate encodings. The representation of spatial coordinates was identified early as a critical component of the surrogate model. Several coordinate encoding strategies were considered, including direct coordinate inputs and learnable feature transformations. Direct coordinate regression using a standard MLP consistently underperformed compared with FFE. Learnable frequency encoding [25] were also considered as a potential extension of the FFE approach. While theoretically appealing, preliminary experiments did not reveal a clear advantage over fixed FFE. Given their simplicity, computational efficiency, and strong empirical performance, fixed FFE were ultimately retained.

Design insights from exploratory investigations. Several conclusions emerge from these exploratory investigations. First, the absence of mesh connectivity strongly favors mesh-independent representations and limits the practical applicability of graph-based methods. Second, the ability to capture high-frequency spatial features is essential for accurately reproducing transonic flow phenomena, making coordinate encoding a critical design choice. Third, the challenge evaluation metric places a strong emphasis on robustness, which makes metric-aligned loss functions and ensemble strategies particularly effective. Finally, the results suggest that, for this benchmark, careful optimization of relatively simple neural-field models is more beneficial than introducing increasingly sophisticated architectures.

These observations provide useful guidance for future work on the CRM-WBPN dataset and, more broadly, for the design of machine-learning surrogates operating on complex aerodynamic geometries under limited-data conditions.

## 7 Final Results

This section evaluates the performance of the proposed methodology (see Section 5) on the hidden test set of the ONERA CRM Wall Distribution Regression Challenge and compares its results with those of the organizer-provided baseline, which ranked second on the leaderboard.

Table 3 reports the performance of the proposed approach together with the best baseline model introduced by Peter et al. [1].

Table 3: Performance comparison between the proposed methodology and the best organizer-provided baseline on the hidden test set.
<table><tr><td></td><td>Proposed methodology</td><td>Global MLP (Peter et al. [1])</td></tr><tr><td>Score ↑</td><td>8.81</td><td>8.640</td></tr><tr><td> $\overline { { R ^ { 2 } \uparrow } }$ </td><td>0.973</td><td>0.956</td></tr><tr><td> $\overline { { R _ { C _ { p } } ^ { 2 } } }$ </td><td>0.983</td><td>0.972</td></tr><tr><td> $R _ { C f _ { x } } ^ { 2 }$ </td><td>0.966</td><td>0.944</td></tr><tr><td> $R _ { C f _ { y } } ^ { 2 }$ </td><td>0.969</td><td>0.951</td></tr><tr><td> ${ \underline { { R _ { C f _ { z } } ^ { 2 } } } }$ </td><td>0.973</td><td>0.957</td></tr><tr><td> $\mathrm { w r M A E \downarrow }$ </td><td>0.210</td><td>0.228</td></tr><tr><td> $\mathrm { w r M A E } _ { C _ { p } }$ </td><td>0.157</td><td>0.198</td></tr><tr><td> $\mathrm { w r M A E } _ { C f _ { x } }$ </td><td>0.170</td><td>0.163</td></tr><tr><td> $\mathrm { w r M A E } _ { C f _ { y } }$ </td><td>0.315</td><td>0.314</td></tr><tr><td> $\mathrm { w r M A E } _ { C f _ { z } }$ </td><td>0.197</td><td>0.237</td></tr><tr><td> $\# \mathrm { p a r a m e t e r s } \downarrow$ </td><td> $\overline { { 7 . 3 \times 1 0 ^ { 7 } } }$ </td><td> $\overline { { 1 . 7 \times 1 0 ^ { 1 0 } } }$ </td></tr></table>

Performance comparison. The results demonstrate that the proposed methodology consistently outperforms the Global MLP baseline across all evaluation metrics. Improvements are observed not only for the average coefficient of determination but also for the worst-case relative error, indicating that the model achieves a better compromise between global predictive accuracy and robustness on the most challenging aerodynamic configurations. This behavior is particularly important since the competition score explicitly rewards methods capable of maintaining high accuracy on difficult operating conditions.

Computational cost. Despite relying on an ensemble of neural fields, the proposed approach remains computationally tractable. Each individual model contains approximately $2 . 1 5 \times 1 0 ^ { 6 }$ trainable parameters, resulting in roughly $7 . 3 \times 1 0 ^ { 7 }$ parameters for the final model ensemble. This remains several orders of magnitude smaller than the Global MLP baseline, which contains approximately $1 . 7 \times 1 0 ^ { 1 0 }$ trainable parameters while achieving lower predictive performance. The proposed methodology therefore provides a parameter-efficient surrogate model for aerodynamic field prediction.

## 7.1 Qualitative Flow Predictions

Beyond the quantitative evaluation, visual inspection of the predicted flow fields provides additional insight into the behavior of the surrogate model. Figures 4 and 5 compares representative predictions obtained with the proposed methodology against the reference CFD solutions for several operating conditions including high Mach and high pressure regimes.

Overall, the predicted aerodynamic fields remain in close agreement with the CFD reference solutions over the entire aircraft surface. The largest discrepancies are generally confined to localized regions exhibiting strong spatial gradients, such as transonic shocks or areas affected by flow separation.

![](images/0b530d630d7baca7790f6c1bc3b4ead6b342748649f43e9a59293ee130ba6a09.jpg)

![](images/b549c2cf1164b95018321c49da446c5b92428acddac6342139114e00aa4915cf.jpg)

![](images/fd02c77612b808a48c925b86173aab6d4cc4804cc3aa00c11500231ba3d516e2.jpg)

![](images/ae492fcd4ee8387a016f3dd5f7a87925c14f2e81bfefa5fea8a61145c0df1f33.jpg)

![](images/9c3299698d37d094d76ea47dc6d98fa77069f8ed88d5dc5a366a604df7a6383c.jpg)  
(a) Pressure coefficient $C _ { p } .$ From top to bottom: CFD reference, model prediction, and absolute prediction error.

![](images/140794e17106a37630c73b7c47d1e334d43553530fa25f7333465ad3e00028fd.jpg)  
(b) Norm of the friction field $( C f _ { x } , C f _ { y } , C f _ { z } )$ . From top to bottom: CFD reference, model prediction, and absolute prediction error.  
Figure 4: Pressure and friction fields prediction obtained with the proposed methodology for the aerodynamic conditions $\bar { M _ { \infty } } = 0 . 5 , A o A = 0 . 0 ^ { \circ }$ and $p _ { i } = \dot { 4 } . 0 \times 1 0 ^ { 5 } ~ \mathrm { P a }$ .

![](images/3070371b52bec2e08bec933a1625b714ea763e113de6db05503fc788482a0a98.jpg)

![](images/bd6f47b1a02c40c44f035467fee1e606f436ca945602173d37f0e760a246be87.jpg)

![](images/f39d3fb16fedbdd798c606fd5ae2479d51462909af348fee74eb3925eac2ac77.jpg)

![](images/ca91d0e64b7b197faba7842cd3a9cf67901d5574fd0b1481720090231d2a7721.jpg)

![](images/a271dc2bf83e046411a2e986d632a06e9f591f0dc983972bd154498f4cc10d63.jpg)  
(a) Pressure coefficient $C _ { p } .$ From top to bottom: CFD reference, model prediction, and absolute prediction error.

![](images/68d18564dce1f019f9edc1ec33818373739385f93a3d1c7e4ab67feaefe0bc5a.jpg)  
(b) Norm of the friction field $( C f _ { x } , C f _ { y } , C f _ { z } )$ . From top to bottom: CFD reference, model prediction, and absolute prediction error.  
Figure 5: Pressure and friction fields prediction obtained with the proposed methodology for the aerodynamic conditions $\bar { M _ { \infty } } = 0 . 9 6 , A o A = - 2 . 0 ^ { \circ }$ and $p _ { i } \stackrel { . } { = } 2 . 0 \times 1 0 ^ { 5 }$ Pa.

## 8 Conclusion

This paper presented the methodology that achieved first place in the ONERA CRM Wall Distribution Regression Challenge for aerodynamic wall-field prediction on the CRM-WBPN benchmark [1]. Rather than introducing a fundamentally new neural architecture, the objective was to demonstrate how a careful combination of existing machinelearning techniques, guided by the characteristics of the benchmark and a systematic experimental analysis, can substantially improve predictive performance.

The proposed surrogate model formulates aerodynamic field prediction as a conditional neural field operating directly on point coordinates, surface normals, and aerodynamic operating conditions. Starting from a simple multilayer perceptron prototype, the final methodology was progressively developed through the introduction of FFE, an increased model capacity, a RSE loss aligned with the evaluation metric, an ensemble learning strategy, and k-fold cross-validation. The progressive ablation study shows that each of these components contributes to the final performance, while the investigation of several alternative strategies provides additional insight into the design of surrogate models for complex aerodynamic configurations.

On the hidden test set, the final methodology achieved the highest score in the competition, outperforming the strongest organizer-provided baseline across both global accuracy and worst-case prediction metrics. Although the proposed solution relies on an ensemble of neural fields, it remains considerably more parameter-efficient than the Global MLP baseline while providing more accurate predictions of localized flow structures, including transonic shocks and sharp variations in the skin-friction fields. These results demonstrate that coordinate-based neural fields, combined with appropriate training strategies, constitute an interesting framework for aerodynamic surrogate modeling when mesh connectivity is unavailable and the amount of training data is limited.

Perspectives. Several directions could further improve the proposed methodology and broaden its applicability. A first perspective consists in studying the influence of the training database size on the model prediction accuracy in order to better characterize the data requirements of neural-field surrogate models. In addition, active learning strategies could be explored to selectively identify the most informative training samples and maximize performance gains. Evaluating the practical deployment of such surrogates in industrial workflows also deserves further investigation, particularly regarding the trade-off between model size, inference time, and predictive accuracy.

Another important direction is to extend the methodology beyond a single aircraft configuration by considering multiple geometries during training. This naturally leads to the development of foundation models for aerodynamic surrogate modeling, capable of learning generalized geometric and physical representations that could subsequently be adapted or fine-tuned to downstream tasks involving new aircraft configurations for which only sparse CFD data are available.

Future work could also investigate more advanced ensemble strategies and uncertainty quantification techniques in order to improve the reliability of the predictions and provide confidence estimates for engineering applications Likewise, incorporating physics-informed weighting strategies during training, for example by emphasizing wing regions containing transonic shocks or other critical flow features, could further improve the prediction of localized discontinuities that dominate the overall error.

Finally, while the absence of mesh connectivity motivated the coordinate-based formulation adopted in this work, future benchmarks providing access to the computational mesh would enable direct comparisons with graph neural networks and mesh-based neural operators. Similarly, evaluating point-cloud architectures, such as Point Transformer models, would provide further insight into the respective advantages and limitations of coordinate-based, point-cloud, and topology-aware approaches for high-fidelity aerodynamic surrogate modeling.

## Acknowledgments

The authors would like to thank Jacques Peter and Quentin Bennehard (ONERA) for organizing the CRM Wall Distribution Regression Challenge and for simulating and releasing the CRM-WBPN benchmark dataset. Their efforts in designing a realistic and challenging benchmark, together with the public release of the dataset following the competition, provide a valuable resource for the development and evaluation of machine-learning methods for aerodynamic surrogate modeling.

This work is supported by the "ARIAC by DigitalWallonia4.ai" research project (grant agreement No 2010235 – TRAIL institute) and benefited from computational resources made available on Lucia, the Tier-1 supercomputer of the Walloon Region, infrastructure funded by the Walloon Region (grant agreement No 1910247).

## AI Disclosure Statement

Generative AI tools, such as ChatGPT, were used for language editing and improving the clarity of the manuscript. All intellectual content, research design, and data analysis were conducted solely by the authors.

## References

[1] Jacques Peter, Quentin Bennehard, Sébastien Heib, Jean-Luc Hantrais-Gervois, and Frédéric Moëns. Onera’s crm wbpn database for machine learning activities, related regression challenge and first results. Computers & Fluids, 302:106838, 2025.

[2] Saakaar Bhatnagar, Yaser Afshar, Shaowu Pan, Karthik Duraisamy, and Shailendra Kaushik. Prediction of aerodynamic flow fields using convolutional neural networks. Computational Mechanics, 64(2):525–545, 2019.

[3] Tobias Pfaff, Meire Fortunato, Alvaro Sanchez-Gonzalez, and Peter W Battaglia. Learning mesh-based simulation with graph networks. arXiv preprint arXiv:2010.03409, 2020.

[4] Zongyi Li, Nikola Kovachki, Chris Choy, Boyi Li, Jean Kossaifi, Shourya Otta, Mohammad Amin Nabian, Maximilian Stadler, Christian Hundt, Kamyar Azizzadenesheli, et al. Geometry-informed neural operator for large-scale 3d pdes. Advances in Neural Information Processing Systems, 36:35836–35854, 2023.

[5] Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial differential equations. arXiv preprint arXiv:2010.08895, 2020.

[6] Zongyi Li, Daniel Zhengyu Huang, Burigede Liu, and Anima Anandkumar. Fourier neural operator with learned deformations for pdes on general geometries. Journal ofMachine Learning Research, 24(388):1–26, 2023.

[7] Louis Serrano, Lise Le Boudec, Armand Kassaï Koupaï, Thomas X Wang, Yuan Yin, Jean-Noël Vittaut, and Patrick Gallinari. Operator learning with neural fields: Tackling pdes on general geometries. Advances in Neural Information Processing Systems, 36:70581–70611, 2023.

[8] Nikola Kovachki, Zongyi Li, Burigede Liu, Kamyar Azizzadenesheli, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Neural operator: Learning maps between function spaces with applications to pdes. Journal ofMachine Learning Research, 24(89):1–97, 2023.

[9] Matthew Tancik, Pratul Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan Barron, and Ren Ng. Fourier features let networks learn high frequency functions in low dimensional domains. Advances in neural information processing systems, 33:7537–7547, 2020.

[10] Vincent Sitzmann, Julien Martel, Alexander Bergman, David Lindell, and Gordon Wetzstein. Implicit neural representations with periodic activation functions. Advances in neural information processing systems, 33:7462– 7473, 2020.

[11] Giovanni Catalani, Siddhant Agarwal, Xavier Bertrand, Frederic Tost, Michael Bauerheim, and Joseph Morlier. Neural fields for rapid aircraft aerodynamics simulations. Scientific Reports, 14(1):25496, 2024.

[12] Giovanni Catalani, Jean Fesquet, Xavier Bertrand, Frédéric Tost, Michael Bauerheim, and Joseph Morlier. Towards scalable surrogate models based on neural fields for large scale aerodynamic simulations. Computers & Fluids, page 106929, 2025.

[13] David Massegur and Andrea Da Ronch. Graph convolutional multi-mesh autoencoder for steady transonic aircraft aerodynamics. Machine learning: science and technology, 5(2):025006, 2024.

[14] Jacques Peter and M. Lippi. Comparison of data -driven methods for the prediction of flows about a wing. In ECCOMAS 2024, 01 2024.

[15] Amer Essakine, Yanqi Cheng, Chun-Wun Cheng, Lipei Zhang, Zhongying Deng, Lei Zhu, Carola-Bibiane Schönlieb, and Angelica I Aviles-Rivero. Where do we stand with implicit neural representations? a technical and performance survey. arXiv preprint arXiv:2411.03688, 2024.

[16] Yiheng Xie, Towaki Takikawa, Shunsuke Saito, Or Litany, Shiqin Yan, Numair Khan, Federico Tombari, James Tompkin, Vincent Sitzmann, and Srinath Sridhar. Neural fields in visual computing and beyond. In Computer graphicsforum, volume 41, pages 641–676. Wiley Online Library, 2022.

[17] Salvatore Cuomo, Vincenzo Schiano Di Cola, Fabio Giampaolo, Gianluigi Rozza, Maziar Raissi, and Francesco Piccialli. Scientific machine learning through physics–informed neural networks: Where we are and what’s next. Journal ofScientific Computing, 92(3):88, 2022.

[18] Kamyar Azizzadenesheli, Nikola Kovachki, Zongyi Li, Miguel Liu-Schiaffini, Jean Kossaifi, and Anima Anandkumar. Neural operators for accelerating scientific simulations and design. Nature Reviews Physics, 6(5):320–328, 2024.

[19] Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred Hamprecht, Yoshua Bengio, and Aaron Courville. On the spectral bias of neural networks. In International conference on machine learning, pages 5301–5310. PMLR, 2019.

[20] Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira, David Brandfonbrener, Lucas Janson, and Sham M. Kakade. SOAP: Improving and stabilizing shampoo using adam for language modeling. In The Thirteenth International Conference on Learning Representations, 2025.

[21] Ilya Loshchilov and Frank Hutter. SGDR: Stochastic gradient descent with warm restarts. In International Conference on Learning Representations, 2017.

[22] Ury Naftaly, Nathan Intrator, and David Horn. Optimal ensemble averaging of neural networks. Network: Computation in Neural Systems, 8(3):283, 1997.

[23] Filipe De Avila Belbute-Peres, Thomas Economon, and Zico Kolter. Combining differentiable pde solvers and graph neural networks for fluid flow prediction. In international conference on machine learning, pages 2402–2411. PMLR, 2020.

[24] Siye Li, Zhensheng Sun, Yujie Zhu, and Chi Zhang. Physics-constrained and flow-field-message-informed graph neural network for solving unsteady compressible flows. Physics of Fluids, 36(4), 2024.

[25] Yang Li, Si Si, Gang Li, Cho-Jui Hsieh, and Samy Bengio. Learnable fourier features for multi-dimensional spatial positional encoding. Advances in Neural Information Processing Systems, 34:15816–15829, 2021.