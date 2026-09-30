# Explainable Machine Learning for Multilayer Planar Winding Inductance Estimation

Spyros Rigas, Theofilos Papadopoulos, Georgios Alexandridis, and Antonios Antonopoulos

Abstract—Rapid and accurate self-inductance estimation for multilayer rectangle-shaped planar windings is essential for modern high-frequency power converters, yet traditional workflows rely on complex mathematical equations, rigid monomial formulas or unexplainable black-box machine learning (ML) models that degrade severely outside their training domain. This paper introduces an explainable ML framework unifying post-hoc feature attribution (SHAP and permutation importance) with Kolmogorov– Arnold Network-guided symbolic regression via the SR-KAN framework to discover closed-form analytical equations without prior structural assumptions. Evaluated on a new open-source dataset of over 10,000 Finite Element Analysis (FEA) simulations across seven out-ofdistribution (OOD) classes, standard tree-based ensembles exhibit severe extrapolation errors (> 36%), whereas the unconstrained SR-KAN expression achieves a robust OOD relative error of 8.22%. Experimental verification across 55 physical printed circuit board prototypes (up to 8 layers, with inductances from 4.11 µH to 559.27 µH) confirms that the KAN-discovered expression translates effectively to real-world hardware, predicting inductance with a mean absolute relative error of 6.26%. To support reproducible research, the complete FEA simulation dataset and prototype measurements are released open-source.

Index Terms—Inductance, Planar Windings, Machine Learning, XAI, Symbolic Regression, Kolmogorov–Arnold Networks

## I. INTRODUCTION

W <sup>ITH</sup> <sup>the</sup> <sup>continuous</sup> <sup>push</sup> <sup>toward</sup> <sup>high-power-density,</sup> miniaturization, and high-efficiency for modern power electronics, planar magnetics have emerged as a viable alternative to bulky conventional wire-wound components. With printed circuit board (PCB) copper traces, planar windings (PWs) offer highly-repetitive parameters, easy and inexpensive structural reproducibility, and low profiles ideal for highfrequency operations [1]. These attributes have made PWs attractive in power applications, including electric vehicle (EV) powertrains [2], wide-bandgap (WBG) semiconductor high-frequency converters [3], data-center power converters [4], and even gate-driver power supplies [5].

Particularly in resonant topologies, such as LLC, CLLC and Dual Active Bridge (DAB) converters, the precise estimation of magnetic parameters is important, as variations directly affect soft-switching conditions and shift the resonant frequency [6]. To address this, magnetic integration methodologies are frequently deployed to take advantage of leakage and magnetizing inductances within a singular structural layout, shrinking the overall converter footprint [7], [8]. While many integrated structures utilize a ferrite core, giving special consideration to windows or air-gaps, to control flux distributions [9], coreless (air-core) planar implementations can be also used, where linear operation, zero core losses, reduced weight, and cost minimization are prioritized [10].

Regardless of the core configuration, achieving a highly accurate and computationally inexpensive estimation of winding inductance remains an open issue. Traditional design workflows heavily rely on numerical Finite Element Analysis (FEA), which yields high accuracy but demands prohibitive computational time and meshing overhead when executing large-scale geometric optimization loops. Conversely, classical analytical expressions (e.g., Wheeler, Rosa, or standard Monomial equations) offer instant evaluation but exhibit unacceptable accuracy degradations as the winding deviates from the standard square-shape, single layer design. Others have proposed estimation equations based on geometrical parameters, but for radio-frequency (RF) small-dimension designs [11].

To bridge the gap between slow and expensive numerical simulations and rigid analytical formulas, FEA and data-driven results have been introduced to the domain of planar magnet ics. For instance, Machine Learning (ML) models have been applied to optimize the highly non-linear parameter spaces of medium-frequency transformers in solid-state topologies [12], while advanced neural networks have been utilized to map the mutual inductance variations in dynamic wireless power transfer setups [13]. While such approaches can fit complex datasets with near-perfect accuracy, they operate as black boxes that obscure the underlying physical mechanisms.

In engineering, explainability is a practical necessity for understanding the core mechanics of the system, its sensitivity to the parameters, and ensuring safety margins and stability boundaries. For these reasons, our previous work [14] proposed a Multiple Linear Regression (MLR) framework to derive a closed-form monomial equation for specific multilayer arrangements. Although this MLR formulation yielded an inherently explainable expression, purely data-driven nonexplainable models can generally achieve superior estimation accuracy. Furthermore, deriving that analytical formula relied heavily on prior domain knowledge – specifically, assuming that a power-law product would describe the problem well

(b)

(a)

– which limits its generalizability when no prior hint of an explicit symbolic form exists.

To achieve explainability without sacrificing predictive accuracy, this paper proposes a holistic explainable ML framework for planar magnetics that unifies two methodological paradigms: post-hoc attribution and Symbolic Regression (SR). The first relies on post-hoc explainability, where attribution techniques such as SHapley Additive exPlanations (SHAP) [15] and permutation importance [16] are applied to pre-trained black-box models, including tree-based ensembles [17], [18] and support vector regressors (SVRs) [19]. The second paradigm bypasses the black box by searching directly for analytical, closed-form expressions without making prior structural assumptions about the modeled system. To this end, we utilize SR-KAN [20], which leverages the interpretable univariate edge functions of Kolmogorov–Arnold Networks (KANs) [21], and systematically benchmark it against the accepted state-of-the-art in evolutionary symbolic regression, PySR [22], as well as the original MLR methodology of [14].

The proposed framework is deployed on a newly developed, open-source dataset comprising over 10,000 FEA simulations of Multilayer Rectangle-Shaped Planar Windings (MLRPWs), which we release as a core contribution of this work [23]. This dataset captures geometric variations across extensive layer counts and consists of a standard training and testing subset alongside separate, independent edge cases explicitly categorized into distinct classes of out-of-distribution (OOD) configurations. Moreover, to validate the accuracy of the simulated dataset under real-world laboratory tolerances, we manufacture and measure more than 50 physically printed ML-RPW prototypes across diverse layer configurations. Crucially, while winding inductance estimation serves as the workhorse application in this study, the introduced explainable pipeline is entirely domain-agnostic and can be readily generalized to alternative modeling tasks across industrial electronics.

The remainder of this paper is organized as follows: Section II introduces the geometric parameters of MLRPWs and outlines the open-source FEA dataset generation, including specific geometric configurations selected for OOD testing. Section III details the explainable ML methodology, highlighting both core aspects of the proposed framework: post-hoc explainability via SHAP and permutation importance, as well as equation discovery via SR-KAN and other symbolic regression baselines. Section IV presents the benchmark evaluations and post-hoc attributions across diverse ML models and the predictive performance of the derived analytical expressions across both standard and OOD domains. Section V provides experimental laboratory verification against more than 50 physically fabricated MLRPW prototypes, and Section VI concludes the paper.

## II. GEOMETRIC PARAMETERS AND DATASET EXPLANATION

The self-inductance of an air-core winding is dictated by its physical shape and geometric proportions. For MLRPWs, this behavior is defined by a multi-dimensional parameter space containing both coil-level and trace-level variables. As shown in Fig. 1, the complete physical layout is determined by the outer-side lengths $D _ { 1 }$ and $D _ { 2 }$ , the number of turns per layer $N _ { T }$ , the total number of layers $N _ { L } .$ , the copper trace width w, the horizontal inter-turn trace spacing $s ,$ and the vertical distance between consecutive insulating layers O. From these independent variables, the inner-aperture side lengths $d _ { 1 }$ and $d _ { 2 }$ are explicitly calculated as:

$$
d _ { i } = D _ { i } - 2 N _ { T } ( w + s ) + 2 s\tag{1}
$$

where $i \in \{ 1 , 2 \}$ denotes the respective orthogonal axes of the rectangular profile.

Planar windings can span vastly different physical scales depending on their targeted industrial application. While smallscale profiles on the micrometer scale are common in lowpower RF chips, power applications require much larger footprints to satisfy the thermal and the electrical conditions. In this direction, the geometric parameters in this study are chosen based on the IPC-2221 design standard. The trace width w spans from 3 mm to 5 mm, which corresponds to a continuous current-carrying capability of approximately 5 A to 10 A for a $2 0 ~ ^ { \circ } \mathrm { C }$ maximum temperature rise limit. Similarly, the horizontal trace spacing s ranges from 0.1 mm to 0.5 mm, satisfying the 300 V turn-to-turn insulation voltagewithstand requirements for permanent polymer-coated external conductors under IPC class B4. To accommodate real-world magnetic cores (such as standard EE or EI geometries), the inner dimensions $d _ { 1 }$ and $d _ { 2 }$ are restricted to a minimum clearance threshold of 17 mm. The complete boundaries of the core design space are summarized in Table I.

![](images/ecc16913185b850238a116925b3e8878cb68ce686b854d977e57fec1b830e042.jpg)

![](images/9e2d4858c4c8cbfebc070f966599ef97288756f845362fba78e31234c7ea4ea9.jpg)  
Fig. 1. Geometric parameters for a typical MLRPW (4-layered).

It should be noted that, using the geometric features of Table I, the previous study [14] introduced a new variable $O ^ { \prime } \equiv O ^ { N _ { L } - \bar { 1 } }$ , i.e., the distance between layers O, raised to $( N _ { L } - 1 )$ , to account for the undefined state where $N _ { L } = 1$ Furthermore, following the current sheet approximation (CSA) principles established in historical methods [24], the geometric mean perimeters $\bar { D } _ { 1 } = ( D _ { 1 } + d _ { 1 } ) / 2$ and $\bar { D } _ { 2 } = ( D _ { 2 } + d _ { 2 } ) / 2$ were also introduced to track the path of the primary mutual flux linkages. Based on these, we construct a FEA dataset comprising 9 features, namely $D _ { 1 } , \ D _ { 2 } , \ \bar { D } _ { 1 } , \ \bar { D } _ { 2 } , \ w , \ s , \ N _ { T }$ $N _ { L } , O ^ { \prime }$ , as well as the inductance L, as the target variable.

Generating a large-scale dataset via FEA is computationally expensive, often requiring months of numerical solver runtime. To optimize this process, we exploit the geometric symmetry of the problem. Because the spatial orientation of a rectangular winding does not affect its flux path or total self-inductance, a configuration with outer boundaries $\{ D _ { 1 } ^ { * } , D _ { 2 } ^ { * } \}$ behaves identically to one with $\{ D _ { 2 } ^ { * } , D _ { 1 } ^ { * } \}$ . Enforcing the strict condition $D _ { 1 } ~ \leq ~ D _ { 2 }$ eliminates redundant combinations and cuts the required FEA simulation volume exactly in half.

For the generation of the FEA dataset, 3D parametric models within ANSYS Maxwell 3D are utilized. To ensure high numerical accuracy while optimizing computational efficiency, an auto-adaptive meshing scheme is applied across the global region, supplemented by an explicitly defined fine mesh localized within the immediate vicinity of the winding. This strategy yields a highly dense discretization ranging from approximately 700,000 to 1,000,000 tetrahedra per simulation. The iterative solver convergence process runs continuously until the global energy error drops below a strict 1% threshold.

The complete open-source dataset developed for this work [23] is split into two parts: a core dataset containing 10,110 samples and an independent OOD testing dataset consisting of 1,124 samples. While the core dataset can be randomly partitioned into standard training and testing subsets to evaluate the interpolation accuracy of ML models, the OOD dataset is reserved exclusively to evaluate how well the models generalize to data from an entirely different distribution. The samples of the OOD dataset are categorized into seven classes:

TABLE I  
VALUE RANGES FOR DATASET PARAMETERS
<table><tr><td>Parameter</td><td>Value Range</td><td></td><td># of Values</td><td>Comments</td></tr><tr><td> $D _ { 1 }$ </td><td> $7 0 { : } 1 0 { : } 1 6 0 ^ { 1 }$ </td><td>mm</td><td>10</td><td> $D _ { 1 } \leq D _ { 2 }$ </td></tr><tr><td> $D _ { 2 }$ </td><td></td><td>mm</td><td>10</td><td></td></tr><tr><td> $d _ { 1 }$ </td><td></td><td></td><td></td><td>derived</td></tr><tr><td> $d _ { 2 }$ </td><td>∈ [17, 123]</td><td>mm</td><td></td><td>from (1)</td></tr><tr><td> $_ w$ </td><td>3,4,5</td><td>mm</td><td>3</td><td></td></tr><tr><td> $s$ </td><td>0.1, 0.3, 0.5</td><td>mm</td><td>3</td><td></td></tr><tr><td> $N _ { T }$ </td><td>6,8, 10</td><td>turns</td><td>3</td><td></td></tr><tr><td> $N _ { L }$ </td><td>1,2, 3,4</td><td>layers</td><td>4</td><td></td></tr><tr><td>0</td><td>0.5, 1.0, 1.5</td><td>mm</td><td>3</td><td>not defined for  $N _ { L } = 1$ </td></tr></table>

• low-turn extension $( N _ { T } = 4 ) - 1 2 0$ samples (Class 1)

• geometric footprint extension $( D _ { 1 } , D _ { 2 } \geq 1 6 0 ~ \mathrm { m m } ) - 3 6 0$ samples (Class 2)

• larger layer stack $( N _ { L } = 6 ) - 1 7 0$ samples (Class 3)

• combined large footprint and layer configuration $( D _ { 1 } , D _ { 2 } \ge 1 6 0 $ mm and $N _ { L } = 6 ) - 6 0$ samples (Class 4)

• ultra-wide track and spacing variant $( w = s = 1 \ \mathrm { m m } ) \ -$ 180 samples (Class 5)

• extreme copper trace width (w = 7 mm) – 84 samples (Class 6)

• constricted inner aperture $( d _ { 1 } \leq 1 7 \ \mathrm { m m } ) - 1 5 0$ samples (Class 7)

## III. EXPLAINABLE MACHINE LEARNING METHODOLOGY

To estimate the planar winding inductance L from the 9- dimensional geometric feature space detailed in Section II, and to quantify the contribution of each feature to the resulting prediction, we implement an explainable framework structured into two distinct branches. The first branch applies classical ML algorithms to evaluate interpolation accuracy and utilizes post-hoc feature attribution to quantify feature importance. The second branch implements symbolic regression to derive closed-form analytical equations. To evaluate both interpolation fidelity and extrapolation robustness, all models are trained and tested under two partitioning schemes: a standard 80%–20% random split of the core dataset (hereafter core split) and an out-of-distribution (OOD) extrapolation split, where models are trained on the full core dataset and evaluated exclusively on the OOD samples (hereafter OOD split).

## A. Machine Learning and Post-Hoc Explainability

Across both data splits, the ML algorithms we deploy are Random Forest (RF) ensembles, Extreme Gradient Boosting (XGBoost), and Support Vector Regression (SVRs) with radial basis function kernels. In principle, RF constructs an ensemble of independent decision trees and averages their predictions to reduce variance [17], XGBoost sequentially trains weak base learners by optimizing a gradient-based objective function with regularization [18] and SVR maps inputs into a highdimensional space to identify a linear boundary maximizing a specified margin tolerance [19]. Prior to training, all input geometric features x and the target inductance L are uniformly mapped to a logarithmic space via $\begin{array} { r } { \textbf { x } \to \log _ { 1 0 } ( \textbf x + \epsilon ) } \end{array}$ where $\epsilon = 1 0 ^ { - 8 }$ acts as a numerical stabilizer. Alternative transformation strategies standard in tabular ML, such as min-max scaling and standardization, were also evaluated; however, the uniform logarithmic transformation demonstrated superior numerical robustness across these models as well as all subsequent symbolic experiments.

The aforementioned models aim to minimize estimation error for $L ,$ therefore the use of post-hoc attribution methods that identify which geometric features are most meaningful to the predictions is required. To this end, we apply permutation importance, which quantifies global feature dependency by calculating the mean decrease in the coefficient of determination $( \Delta R ^ { 2 } )$ when a given feature vector is randomly shuffled over k iterations $( k = 2 0$ in this study) [16], as well as SHAP, which computes local, additive feature attributions based on cooperative game theory [15]. To consolidate these attribution techniques, we compute a normalized consensus ranking (CR) for each feature. The individual importance scores from permutation importance and SHAP, denoted as $\bar { I } _ { \mathrm { p e r m } }$ and $\bar { I } _ { \mathrm { S H A P } }$ respectively, are first normalized and then averaged across all three models:

$$
C R = \frac { 1 } { 3 } \sum _ { m \in \mathcal { M } } \left( \frac { \bar { I } _ { \mathrm { p e r m } , m } + \bar { I } _ { \mathrm { S H A P } , m } } { 2 } \right)\tag{2}
$$

where $\mathcal { M } = \{ \mathrm { R F } , \mathrm { X G B o o s t } , \mathrm { S V R } \}$ . This aggregated metric unifies the local attributions of SHAP with the global sensitivity of permutation importance into a single score.

## B. Symbolic Regression

Our previous work [14] utilized MLR to fit a linear hyperplane to the log-transformed features of the original dataset, which enforces a rigid power-law monomial structure. While this approach builds upon prior work for the case of MLRPWs [24], enforcing a fixed symbolic form introduces structural bias that limits generalization performance across diverse application cases. To achieve problem-agnostic equation discovery without prior assumptions, this work primarily deploys the SR-KAN framework [20]. SR-KAN addresses the bottlenecks of original Kolmogorov–Arnold Networks (KANs) for symbolic regression by replacing B-splines with computationally efficient radial basis functions parameterizing a Reflectional Switch Activation Function (RSWAF). Furthermore, to overcome the inability of standard additive KAN layers to capture product-based parameter couplings, the architecture explicitly incorporates separable multiplicative subunits [25]. Training is guided by an objective function that couples an $\ell _ { 1 }$ magnitude penalty with row-wise and column-wise entropy regularizations to promote network sparsity. The operational pipeline executes a hierarchical search starting from singlelayer, single-unit configurations up to deeper architectures, systematically pruning inactive edges before regressing the remaining univariate transformations against a symbolic function vocabulary to turn the network into a single analytic expression.

To evaluate the performance achieved by the expression derived by SR-KAN, we benchmark it against both the classical MLR formulation and PySR, a genetic programming framework widely recognized as the literature standard for symbolic regression. PySR explores a problem-agnostic algebraic search space composed of fundamental operators $( \mathrm { e . g . , \ell + , \ell - , \ell \times }$ ÷) via tournament selection and regularized mutations [22]. Rather than optimizing a single isolated expression, both PySR and SR-KAN discover a multi-objective Pareto front of candidate formulas that balance mathematical complexity against predictive accuracy. Regarding accuracy, because the raw inductance values are inherently small, standard mean squared error metrics become numerically insensitive; therefore, after evaluating the candidate expressions in the logtransformed space, predictions are mapped back to the linear domain via a base-10 exponential transformation to compute the mean absolute relative error (E):

$$
\mathcal { E } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left| \frac { L _ { \mathrm { p r e d } , i } - L _ { \mathrm { t a r g e t } , i } } { L _ { \mathrm { t a r g e t } , i } } \right|\tag{3}
$$

where N denotes the number of samples, $L _ { \mathrm { t a r g e t } }$ represents the true inductance from the FEA dataset, and $L _ { \mathrm { p r e d } }$ is the model prediction.

## IV. RESULTS & DISCUSSION

This section presents the experimental results of the explainable framework proposed herein, deployed on the newly introduced MLRPW dataset. To ensure statistical significance against sources of inherent numerical randomness, such as the data partitioning of the core split or the weight initialization seeds of the neural layers within SR-KAN, all relevant experiments are executed across three independent random seeds. Performance metrics are reported as the mean value bounded by the standard error of the mean (±SEM). Throughout these benchmarks, all models are systematically compared against the baseline analytical expression derived in [14]:

$$
\begin{array} { r l } & { L = - 5 . 7 - 0 . 5 9 D _ { 1 } - 0 . 3 8 D _ { 2 } + 1 . 1 8 \bar { D } _ { 1 } + 1 . 0 7 \bar { D } _ { 2 } } \\ & { \qquad - 0 . 1 8 w - 0 . 0 1 s + 1 . 7 9 N _ { T } + 1 . 8 N _ { L } - 0 . 0 0 6 O ^ { \prime } , } \end{array}\tag{4}
$$

where the target inductance and all geometric features are logtransformed, but are written without the explicit functional notation of the logarithm for notational brevity. This convention is maintained for the remainder of the text, mapping the multiplicative power-law monomial directly into a linear expression.

## A. Machine Learning Results

Prior to executing the experiments, the hyperparameters for each regressor are tuned as follows: for the RF model, the ensemble size is configured to 300 independent decision trees to ensure variance reduction, with the maximum tree depth bounded at 20 to restrict individual estimator complexity. The XGBoost model is similarly tuned to 300 sequential base estimators utilizing the histogram-based tree method, which bin continuous features into discrete bins to significantly accelerate training throughput on tabular datasets. The SVR model is implemented using a radial basis function kernel, with the regularization parameter set to $C = 1 0 0$ to penalize margin violations, a tube insensitivity threshold of $\epsilon = 0 . 0 1$ to ignore minor training residuals and a data-dependent scaling parameter to modulate the decision boundary sensitivity. These hyperparameter selections are justified by the dimensionality of the feature space and total number of samples in the simulation dataset, based on common practices in tabular ML [26]. The RF and SVR architectures are implemented via the scikit-learn Python library [27], while XGBoost is deployed using the official xgboost library [18].

The evaluation metrics achieved by these ML models across both data splits are presented in Table II, which also includes the predictive performance of the reference analytical expression defined in Eq. (4). Performance across both the core split and the OOD split is quantified via the mean coefficient of determination $( R ^ { 2 } )$ alongside the mean absolute relative error (E) defined in Eq. (3). To reflect statistical significance, the relative error $\mathcal { E }$ is bounded explicitly by its corresponding standard error of the mean (±SEM) across the independent evaluation seeds for the core split. For the OOD split, the standard error reduces to zero or near-zero because the deterministic SVR formulation is seed independent, while the full-sample training routine over the entire core dataset stabilizes the tree-based ensembles against variance induced by subset partitioning (the core split is deterministic).

TABLE II  
PREDICTIVE ACCURACY COMPARISON OF MACHINE LEARNING MODELS WITHIN THE CORE AND OOD SPLITS
<table><tr><td rowspan="2">Model</td><td colspan="2">Core Split</td><td colspan="2">OOD Split</td></tr><tr><td> $\mathbf { R } ^ { 2 } \mathbf { \Sigma } ( \% )$ </td><td> $\varepsilon \left( \% \right)$ </td><td> ${ \bf R } ^ { 2 } { \bf \Pi } ( \% )$ </td><td>ε (%)</td></tr><tr><td>RF</td><td>99.90</td><td> $1 . 8 7 \pm 0 . 0 2$ </td><td>52.53</td><td> $3 6 . 6 5 \pm 0 . 0 0$ </td></tr><tr><td>XGBoost</td><td>99.94</td><td> $1 . 4 3 \pm 0 . 0 2$ </td><td>51.29</td><td> $3 7 . 4 5 \pm 0 . 0 0$ </td></tr><tr><td>SVR</td><td>99.66</td><td> $3 . 0 8 \pm 0 . 0 5$ </td><td>94.18</td><td> $1 2 . 9 5 \pm 0 . 0 0$ </td></tr><tr><td>Reference MLR [14]</td><td>98.71</td><td> $5 . 8 8 \pm 0 . 1 7$ </td><td>98.10</td><td> $9 . 9 2 \pm 0 . 0 0$ </td></tr></table>

The empirical results compiled in Table II reveal distinct performance trade-offs between internal dataset interpolation and out-of-distribution extrapolation. Within the core split, all models achieve high predictive accuracy, yielding $R ^ { 2 }$ scores above 98%. The tree-based ensembles demonstrate the highest performance, with XGBoost being the leader $( \mathcal { E } = 1 . 4 3 \% )$ followed closely by RF. The non-linear SVR occupies a middle tier, whereas the reference monomial model yields the highest error rate within this group, approaching a mean absolute relative error of nearly 5.88%. This situation flips entirely for the OOD split. In this case, the reference monomial model maintains the lowest overall error profile $( \mathcal { E } = 9 . 9 2 \% )$ followed by the SVR model with an $R ^ { \bar { 2 } }$ of 94.18 and ${ \mathcal { E } } =$ 12.95%. Conversely, the tree-based models experience severe performance degradation; their $R ^ { 2 }$ coefficients plummet to approximately 51%–53%, while their mean absolute relative errors surge past 36%. This failure stems directly from the nature of recursive partitioning trees, which lack the capacity to extrapolate outside their learned feature bounds and are highly prone to overfitting.

## B. Attribution via SHAP and Permutation Importance

To evaluate which features dominate the model decisions that lead to the performances presented in Table II, posthoc feature attribution is applied for all studied ML models. To this end, global feature variations are captured via permutation importance implemented using scikit-learn. Local, additive contributions are evaluated using the shap library [15], employing the specialized TreeExplainer algorithm for the tree-based models [28] and a sample-averaged KernelExplainer approximation for SVR. To balance out the distinct metrics generated by these different techniques, the raw importance scores from each model are normalized so that the feature allocations within each individual method sum up to unity. The resulting relative feature metrics are summarized across models in Table III, while the consolidated consensus ranking computed via Eq. (2) is illustrated in Fig. 2.

NORMALIZED FEATURE IMPORTANCE SCORES ACROSS MACHINE LEARNING MODELS  
TABLE III
<table><tr><td rowspan="2">Feature</td><td colspan="4">Permutation Importance</td><td colspan="2">SHAP</td></tr><tr><td>RF</td><td>XGBoost</td><td>SVR</td><td>RF</td><td>XGBoost</td><td>SVR</td></tr><tr><td> $D _ { 1 }$ </td><td>0.0150</td><td>0.0087</td><td>0.1030</td><td>0.0769</td><td>0.0482</td><td>0.1350</td></tr><tr><td> $D _ { 2 }$ </td><td>0.0004</td><td>0.0028</td><td>0.0071</td><td>0.0044</td><td>0.0258</td><td>0.0333</td></tr><tr><td> $\bar { D } _ { 1 }$ </td><td>0.0823</td><td>0.0609</td><td>0.3152</td><td>0.1135</td><td>0.1326</td><td>0.2459</td></tr><tr><td> $\bar { D } _ { 2 }$ </td><td>0.0482</td><td>0.0234</td><td>0.0109</td><td>0.0881</td><td>0.0679</td><td>0.0407</td></tr><tr><td> $w$ </td><td>0.0164</td><td>0.0210</td><td>0.0027</td><td>0.0483</td><td>0.0733</td><td>0.0220</td></tr><tr><td> $s$ </td><td>0.0002</td><td>0.0004</td><td>0.0001</td><td>0.0031</td><td>0.0066</td><td>0.0014</td></tr><tr><td> $N _ { T }$ </td><td>0.1980</td><td>0.1156</td><td>0.1561</td><td>0.2008</td><td>0.1827</td><td>0.1702</td></tr><tr><td> $N _ { L }$ </td><td>0.2675</td><td>0.7651</td><td>0.3246</td><td>0.1908</td><td>0.4399</td><td>0.2465</td></tr><tr><td> $O ^ { \prime }$ </td><td>0.3719</td><td>0.0021</td><td>0.0802</td><td>0.2741</td><td>0.0229</td><td>0.1051</td></tr></table>

The relative normalized feature attributions compiled in Table III demonstrate a strong consensus across all algorithms regarding parameter dominance. For all models and both attribution techniques, the macro-structural winding parameters – specifically the total number of layers $N _ { L }$ and the number of turns per layer $N _ { T }$ – emerge as the primary predictors. This effect is most pronounced in the XGBoost architecture, where $N _ { L }$ accounts for 76.51% of the global permutation drop and 43.99% of the local SHAP attribution. Conversely, layout metrics such as the horizontal inter-turn trace spacing s and the outer-side length $D _ { 2 }$ consistently yield near-zero importance across all models.

The low individual attribution scores assigned to $D _ { 2 }$ and the copper trace width w do not indicate physical insignificance, but rather reflect the high multicollinearity inherent to Eq. (1). Because the uniform log-transformation preserves these linear dependencies, the models suffer from variance redundancy. Once an algorithm captures the global boundary via the primary outer-side length $D _ { 1 }$ and the turn allocations $( N _ { T } , N _ { L } )$ , the variance explained by $D _ { 2 }$ and w becomes largely redundant, causing their isolated attribution scores to collapse. A similar mechanism is observed between the modified layer distance term $O ^ { \prime }$ and $N _ { L }$ . For the XGBoost model, the attribution metrics for $O ^ { \prime }$ appear practically negligible (0.0021 in permutation and 0.0229 in SHAP); however, this is directly attributed to its structural relation to $N _ { L }$ , which acts as the dominant primary predictor and absorbs the shared variance during sequential gradient boosting.

![](images/24cf94598ffe33ebbad740dbf8e46a82405726c7c1e13158b45795c76dd23f6d.jpg)  
Fig. 2. Consensus Ranking (CR) scores across the 9-dimensional MLRPW geometric space, showing the stacked contribution of mean permutation importance and mean SHAP attribution.

The consolidated feature metrics are illustrated in Fig. 2, which explicitly displays the composition of the final consensus ranking as the superposition of both attribution methods. The visual breakdown confirms that $N _ { L }$ is the heavily dominating structural driver across both local and global attribution methods, followed symmetrically by $N _ { T }$ and $\bar { D } _ { 1 }$ . Conversely, features such as s and $D _ { 2 }$ show almost negligible individual segments, visually underscoring how collinearity suppresses their isolated ranking metrics. Ultimately, the fact that distinct machine learning architectures and completely different attribution techniques converge on this exact parameter hierarchy is highly significant; it uncovers the most influential features of the dataset, which are expected to be prioritized moving forward into the symbolic regression phase.

## C. Refining the MLR Approach

Before proceeding to fully unconstrained symbolic regression, we first revisit the MLR approach introduced in [14], which assumes a monomial expression for L. The previous study assumed that training on a subset of the current, full dataset was sufficient and that the full dataset would not yield significantly different model parameters. However, while the expression corresponding to Eq. (4) achieved a mean absolute relative error of 1.24% in [14], its error increases to $5 . 8 8 \pm 0 . 1 7 \%$ on the full dataset presented here. This nearly five-fold increase in error indicates that generating the complete dataset is highly meaningful; it also indicates that we should re-fit the MLR model to find an updated expression that performs better on the full dataset.

Following the logarithmic transformation and least-squares optimization methodology outlined in [14], the refined monomial expression for the MLRPW inductance is given by:

$$
\begin{array} { c } { { L = - 5 . 5 - 1 . 8 D _ { 1 } + 0 . 1 1 D _ { 2 } + 2 . 2 \bar { D } _ { 1 } + 0 . 6 5 \bar { D } _ { 2 } } } \\ { { { } } } \\ { { - 0 . 0 5 3 w + 0 . 0 1 s + 2 . 0 N _ { T } + 1 . 8 N _ { L } - 0 . 0 0 5 O ^ { \prime } . } } \end{array}\tag{5}
$$

Evaluating this updated expression yields a mean absolute relative error of $\mathcal { E } = 4 . 1 2 \pm 0 . 0 3 \%$ alongside a determination coefficient of $R ^ { 2 } = 9 9 . 5 1 \%$ for the core split. For the OOD split, the refined model achieves an error of $\mathcal { E } = 7 . 9 1 \%$ and an $R ^ { 2 }$ of 98.98%. While this refinement does indeed lead to better performance, it does not alter the qualitative conclusions established in the previous benchmarks: the standard ML models of Section IV-A still exhibit superior interpolation within the core split, whereas the MLR model shows superior robustness when generalizing outside the domain bounds for the OOD split.

To illustrate the performance improvement, the plots comparing the dataset ground-truth inductance $L _ { \mathrm { t r u e } }$ against the model predictions $L _ { \mathrm { p r e d } }$ are presented in Fig. 3. Across both the core and OOD splits, the data points for the refined model of Eq. (5) display a noticeably tighter concentration along the ideal $L _ { \mathrm { p r e d } } = L _ { \mathrm { t r u e } }$ identity line. This reduction in dispersion is especially evident at higher inductance values (above 150 $\mu \mathrm { H }$ in the core split and above 400 $\mu \mathrm { H }$ in the OOD split) where the original expression consistently exhibits a systematic upward bias, overestimating the target inductance.

The sign and magnitude of the coefficients in Eqs. (4) and (5) provide valuable physical insights, as they correspond directly to the power-law exponents of the underlying monomial. In the following analysis, the explicit log notation is reintroduced to avoid confusion between the physical geometric features and their log-transformed counterparts.

In both models, the coefficients for log $N _ { T }$ and log $N _ { L }$ have the highest relative magnitudes (excluding the geometric variables), confirming the primary importance ranking shown in the CR analysis of Fig. 2. Note that the near-zero exponent of the modified layer distance log $\begin{array} { r } { O ^ { \prime } \ \left( \approx \ - 0 . 0 0 5 \right) } \end{array}$ in both expressions is not an indication of physical insignificance, but rather a consequence of its high collinearity with $N _ { L }$ , which suppresses its individual weight.

However, an apparent discrepancy arises when observing the coefficient of log $D _ { 2 } .$ , which flips from −0.38 in Eq. (4) to +0.11 in Eq. (5). While a sign flip typically suggests a contradiction in the predicted physics of the system, this behavior is a mathematical artifact of the strong collinear coupling between $D _ { 2 }$ and $\bar { D } _ { 2 } \equiv ( D _ { 2 } + d _ { 2 } ) / 2$ . Following Eq. (1), $d _ { 2 }$ can be expressed as $d _ { 2 } = D _ { 2 } - 2 \Delta$ , where $\Delta = N _ { T } \left( w + s \right) - s .$ therefore combining these two expressions yields:

$$
\bar { D } _ { 2 } = D _ { 2 } \left( 1 - { \frac { \Delta } { D _ { 2 } } } \right) .\tag{6}
$$

![](images/fd6a2fb851b7909f7090f1eeabfe504b96090ce419d92dacd9388b293d043a9b.jpg)

![](images/fa9104bf84f77fcd2188a924fc714bf42e6d7c918d0ef75f96436a6811157f93.jpg)

![](images/5b71075b12d2e8fbc616c2e99f84290594de311d61fea21c31e08239a10b5b3f.jpg)

![](images/e6de39558799c1ceab35a88a950ea5700ed08c37fe2f53343d90f76c8d1f47cc.jpg)  
Fig. 3. Comparison plots mapping the predicted self-inductance $L _ { \mathsf { p r e d } }$ against the true self-inductance $L _ { \mathrm { t r u e } }$ for the original monomial (left column) and the refined MLR model (right column) evaluated across both the core split (top row) and the OOD split (bottom row). The dashed diagonal represents the ideal zero-error line $( \dot { L } _ { \mathsf { p r e d } } = L _ { \mathsf { t r u e } } )$

As a first-order approximation, one may assume that $\Delta / D _ { 2 }$ is sufficiently small, so that performing a Taylor expansion and keeping the linear terms, i.e., log $\bar { D } _ { 2 } \approx$ log $D _ { 2 } - \Delta / D _ { 2 }$ , is justified. Then, the net effective coefficients for log $D _ { 2 }$ in Eqs. (4) and (5) become $- 0 . 3 8 + 1 . 0 7 \ = \ 0 . 6 9 \ > \ 0$ and $0 . 1 1 + 0 . 6 5 = 0 . 7 6 > 0 .$ respectively, demonstrating that the underlying physics predicted by both expressions is consistent.

## D. Symbolic Regression Results

With the refined MLR model established as our new baseline, we proceed to fully unconstrained symbolic regression using SR-KAN alongside PySR, the current domain standard. For the PySR execution, we select 300 iterations across 50 populations, deploying an extended mathematical vocabulary that includes basic algebraic operators and the power operation. Each run generates a collection of candidate models from which the framework automatically elects the Pareto-optimal expression, balancing low functional complexity against high predictive accuracy.

For the SR-KAN implementation, we enforce an absolute error threshold of 0.08 in the log-transformed space between prediction and target. The base regularization coefficient is set to $\lambda _ { 0 } ~ = ~ 2 \times 1 0 ^ { - 4 }$ , with relative regularization components configured as $\lambda _ { 1 } = 1$ for magnitude regularization, $\lambda _ { 2 } = 2$ for entropy regularization and $\lambda _ { 3 } ~ = ~ 1$ for l<sub>1</sub>-regularization [20]. The results for each framework are summarized in Table IV. Note that, because the OOD split is deterministic, the refined MLR baseline yields no standard error there; however, the stochastic nature of PySR and SR-KAN necessitates seedaveraging across both splits.

As demonstrated in Table IV, all three frameworks achieve high predictive accuracy within the core split, consistently yielding $R ^ { 2 }$ values above 99%. Within this domain, SR-KAN exhibits a slight performance edge with a mean absolute relative error of $3 . 7 7 \pm 0 . 0 4 \% ,$ with variance as low as that of the refined MLR model. Conversely, PySR yields slightly lower accuracy and higher variance compared to both alternative frameworks. The performance divergence becomes far more pronounced when examining the results for the OOD split. Within this regime, the refined MLR baseline maintains the highest performance, followed very closely by SR-KAN. In contrast, PySR exhibits significantly degraded generalization capability, producing a mean error of $1 7 . 3 2 \pm 3 . 1 7 \%$ , which is more than double that of the other two models.

These results demonstrate that SR-KAN not only outperforms PySR in both splits, but also matches or exceeds the performance of the refined MLR baseline. This represents a major milestone: while the MLR framework relies on a rigid, pre-defined monomial assumption rooted in previous literature to constrain its optimization, SR-KAN achieves equivalent generalization without any prior assumptions regarding the underlying functional form. Consequently, this robustness highlights KAN-based symbolic regression as an exceptionally powerful tool for industrial electronics applications where the exact parametric scaling laws are unknown, or impractical to determine.

TABLE IV  
PREDICTIVE ACCURACY COMPARISON OF SYMBOLIC REGRESSION FRAMEWORKS WITHIN THE CORE AND OOD SPLITS
<table><tr><td rowspan="2">Model</td><td colspan="2">Core Split</td><td colspan="2">OOD Split</td></tr><tr><td> ${ \bf R } ^ { 2 } { \bf \Xi } ( \% )$ </td><td>ε (%)</td><td> ${ \bf R } ^ { 2 } { \bf \Xi } ( \% )$ </td><td>ε (%)</td></tr><tr><td>Refined MLR</td><td>99.51</td><td> $4 . 1 2 \pm 0 . 0 3$ </td><td>98.98</td><td> $7 . 9 1 \pm 0 . 0 0$ </td></tr><tr><td>PySR</td><td>99.29</td><td> $4 . 7 4 \pm 0 . 1 7$ </td><td>96.14</td><td> $1 7 . 3 2 \pm 3 . 1 7$ </td></tr><tr><td>SR-KAN</td><td>99.49</td><td> $3 . 7 7 \pm 0 . 0 4$ </td><td>98.42</td><td> $8 . 2 2 \pm 0 . 5 5$ </td></tr></table>

To examine the specific geometric features selected by each framework and verify if they align with the consensus feature rankings established in Section IV-B, we isolate the expressions from the best-performing individual runs on the OOD split. The optimal PySR model achieves a determination coefficient of $R ^ { 2 } = 9 9 . 3 1 \%$ with an error of $\mathcal { E } = 4 . 4 6 \%$ for the core split, alongside OOD metrics of $R ^ { 2 } = 9 7 . 0 2 \%$ and $\mathcal { E } = 1 3 . 8 4 \%$ , yielding the explicit functional form presented in Eq. (7).

$$
L = - 2 . 6 9 + N _ { L } + N _ { T } ^ { 1 . 4 2 } + \frac { D _ { 2 } + N _ { L } + \bar { D } _ { 1 } } { 1 . 1 5 } + \frac { 3 . 7 2 } { w }\tag{7}
$$

Conversely, the best-performing SR-KAN expression achieves a core-split performance of $\bar { R ^ { 2 } } = 9 9 . 4 6 \%$ and $\mathcal { E } = 3 . 8 4 \%$ while maintaining remarkable generalization across the OOD split with $R ^ { 2 } = 9 8 . 7 3 \%$ and $\mathcal { E } = 7 . 3 6 \%$ (even lower than the refined MLR baseline). This optimal analytical expression is provided in Eq. (8).

$$
\begin{array} { r l } & { L = - 2 . 5 8 - 0 . 4 6 \bar { D } _ { 1 } ^ { 2 } + 1 . 2 \bar { D } _ { 1 } + 0 . 7 4 \bar { D } _ { 2 } + 2 . 0 5 N _ { T } } \\ & { \qquad - 0 . 7 7 \sqrt { 1 0 . 3 D _ { 1 } + 1 4 . 6 6 } } \\ & { \qquad + 2 . 1 \sin ( 0 . 9 2 N _ { L } ) - 0 . 0 2 \sin ( 6 . 4 1 N _ { L } ) } \end{array}\tag{8}
$$

An inspection of the discovered equations reveals that both symbolic frameworks select $N _ { L } , N _ { T }$ , and $\bar { D } _ { 1 }$ . These variables exactly mirror three of the top four parameters identified by the consensus ranking in Section IV-B. While the fourth high-ranking parameter, $O ^ { \prime }$ , is omitted by both algorithms, its exclusion follows the exact collinearity mechanism observed in the MLR baseline; its high consensus ranking during posthoc feature attribution stems from its coupling with $N _ { L }$ Additionally, both models selectively retain a sparse subset of minor geometric metrics: $D _ { 2 }$ and w for PySR and ${ \bar { D } _ { 2 } }$ and $D _ { 1 }$ for SR-KAN. This demonstrates another advantage of unconstrained symbolic regression, as lower-importance features (such as s) are systematically pruned, leading to simpler expressions.

To conclude this analysis, we provide in Fig. 4 the relative residuals for the refined MLR, PySR and SR-KAN expressions evaluated exclusively on the OOD split. To isolate exactly where each expression struggles or excels, the individual residuals are color-coded according to the seven distinct OOD classes defined in the end of Section II. A primary observation is the distinct performance degradation of the PySR framework, which exhibits a significantly broader vertical dispersion, with relative errors reaching as high as 70% in magnitude for specific edge cases (Class 3). Conversely, the refined

![](images/d1cf95d799358f60f39f60ad2e18801a0e457903ba69a395f4a13f59a426f8c3.jpg)

![](images/aa78dc1f398502dc0bde3b035f97d16da2a63390a8795c78ef974c3c02e95028.jpg)

![](images/612375427c5154d7a37ebc14b964bd1be0a0fd45b5f9fbbe229ccbb1b3b86c99.jpg)  
Fig. 4. Relative residuals $( L _ { \mathrm { t r u e } } - L _ { \mathsf { p r e d } } ) / L _ { \mathrm { t r u e } }$ plotted against the target self-inductance $L _ { \mathrm { { t r u e } } }$ on the $\mathtt { O O D }$ split for the refined MLR, PySR and SR-KAN models. The individual error points are color-coded according to the seven distinct OOD classes.

MLR and SR-KAN expressions display a far more constricted residual range, bounding the majority of their tracking errors within a much tighter envelope. Furthermore, MLR and SR-KAN demonstrate remarkably congruent error characteristics across the individual geometric classes; both frameworks share very similar distribution topologies for most classes, which are nearly identical in the cases of Class 6 and Class 7. In contrast, the PySR residual profile behaves independently. This cross-framework structural alignment strongly suggests that SR-KAN independently discovered an analytical mapping that mirrors the foundational power-law scaling principles embedded within the MLR formulation, accounting for its stable generalization across the OOD split.

## V. EXPERIMENTAL VALIDATION

To assess the numerical accuracy and physical validity of the parametric FEA models, experimental validation was performed on hardware prototypes. Over 50 distinct MLRPW specimens were fabricated using standard commercial PCB prototyping techniques. The manufactured component portfolio covers a broad geometric design landscape, featuring trace widths up to 7 mm, layer counts extending from single-layer up to an 8-layer stack and a wide experimental self-inductance range spanning from 4.11 µH to 559.27 µH. The physical parameter ranges and boundary metrics of these experimental test specimens are summarized in Table V.

Laboratory measurements were conducted utilizing an HP 4284A high-precision LCR meter, which offers a manufacturer-specified accuracy ranging from ±0.1% for high-impedance windings $t o \ \pm 1 \%$ for low-impedance configurations. To suppress unwanted residual impedance, stray capacitance and induction errors introduced by the test leads, the OPEN and SHORT compensation calibration routines were executed immediately prior to data collection at a test frequency of 100 kHz, matching the harmonic excitation of the FEA simulations.

The tracking correlation between the numerical simulations and physical laboratory findings demonstrates exceptional consistency across the entire dataset. Evaluating all 55 prototype configurations yields a mean relative error of −0.757%. A single prominent tracking anomaly is observed at Sample #36, where the numerical solver underestimates the physical hardware response by −29.86%. This localized deviation is classified as a physical manufacturing defect, likely originating from internal copper layer misalignment or localized prepreg thickness variations during the board pressing stage. Excluding this single anomalous specimen from the validation pool compresses the global mean relative error to a negligible −0.228%, while achieving an overall mean absolute relative error of just 2.39% and a tight standard deviation of 3.10%. These tight error bounds confirm that the synthetic FEA dataset reflects real-world physical behavior, establishing a mathematically sound and highly dependable framework for training and testing the explainable ML models and SR frameworks.

TABLE V  
EXPERIMENTAL PROTOTYPE RANGES AND FEA VALIDATION SUMMARY (56 SAMPLES)
<table><tr><td>Physical Parameter</td><td>Minimum</td><td>Maximum</td><td>Unit</td></tr><tr><td>Outer dimension  $D _ { 1 }$ </td><td>70.0</td><td>210.0</td><td>mm</td></tr><tr><td>Outer dimension  $D _ { 2 }$ </td><td>100.0</td><td>294.0</td><td>mm</td></tr><tr><td>Trace width w</td><td>3.0</td><td>7.0</td><td>mm</td></tr><tr><td>Trace spacing s</td><td>0.1</td><td>2.0</td><td>mm</td></tr><tr><td>Turns per layer  $N _ { T }$ </td><td>4</td><td>12</td><td>turns</td></tr><tr><td>Layer count  $N _ { L }$ </td><td>1</td><td>8</td><td>layers</td></tr><tr><td>Measured  $L _ { \mathrm { m e a s } }$ </td><td>4.11</td><td>559.27</td><td>µH</td></tr></table>

<table><tr><td>Error Metric  $( L _ { \mathrm { s i m } }$  vs. Lmeas)</td><td>All Samples</td><td>Excl. Sample #36</td></tr><tr><td>Mean Relative Error (%)</td><td>-0.757</td><td>-0.228</td></tr><tr><td>Mean Absolute Error (%)</td><td>2.882</td><td>2.392</td></tr><tr><td>Standard Deviation σ (%)</td><td>5.014</td><td>3.104</td></tr></table>

To evaluate the practical utility of the derived closed-form expressions, Eqs. (5), (7), and (8) are compared against the laboratory measurements $L _ { \mathrm { m e a s } } .$ . As illustrated in Fig. 5, all expressions demonstrate strong alignment across the entire experimental range. Evaluating the 55 valid physical specimens (excluding the Sample #36 defect), the refined MLR monomial (5) achieves a mean absolute relative error of $\mathcal { E } \ : = \ : 7 . 1 1 \%$ and a standard deviation of $\sigma = 7 . 3 8 \%$ . The PySR model (7) achieves an accuracy of $\mathcal { E } ~ = ~ 6 . 1 0 \% ~ ( \sigma ~ = ~ 7 . 9 8 \% )$ The unconstrained expression discovered via the SR-KAN framework (8) yields an error of $\mathcal { E } = 5 . 9 8 \% ( \sigma = 8 . 3 3 \% )$ All three analytical expressions maintain reliable prediction bounds across demanding structural edge cases, such as $N _ { L } \in \{ 6 , 8 \}$ , confirming that symbolic regression translates successfully from synthetic FEA simulation environments to real-world power electronic hardware.

![](images/949b0615fda398e2db4f454104e0715d686178ebe0d42c4505114149be67fd47.jpg)  
Fig. 5. Comparison plots mapping the measured self-inductance L<sub>meas</sub> against the estimated self-inductance $L _ { e q u a t i o n s }$ for equations (5), (7), and (8).

## VI. CONCLUSION

This paper introduced a holistic, explainable ML framework for estimating the inductance of MLRPWs across broad geometric domains. By unifying SHAP and permutation importance, as post-hoc feature attribution techniques, with the SR-KAN framework for symbolic regression, the proposed framework achieves high predictive accuracy while yielding transparent, closed-form mathematical equations. Unlike classical analytical models or prior MLR approaches that enforce a rigid power-law monomial structure, SR-KAN enables problemagnostic equation discovery without requiring predefined functional assumptions about the underlying physical system. By removing these pre-enforced mathematical constraints, the framework yields a structurally unbiased equation discovery process that naturally captures complex non-linear parametric couplings.

Benchmarking across both the core and OOD splits revealed critical trade-offs regarding model architecture and extrapolation capability. Standard tree-based algorithms (XGBoost and RF) achieved near-perfect interpolation within the core split $( \mathcal { E } < 1 . 9 \% )$ , but suffered severe performance degradation when evaluated on OOD edge cases $( \mathcal { E } > 3 6 \% )$ . This failure underscores that standard black-box regressors cannot be safely extrapolated for hardware optimization outside their explicit training boundaries. Conversely, the closed-form expression discovered by SR-KAN maintained stable performance, yielding an average OOD error of 8.22% (with an optimal single run reaching 7.36%), matching and occasionally exceeding the 7.91% error of the refined MLR baseline.

Post-hoc feature attribution established a consistent parameter ranking across distinct ML architectures, identifying layer count, turns per layer, and geometric mean perimeters as the primary structural drivers of self-inductance. Furthermore, the symbolic regression models automatically pruned highly collinear or low-impact variables, such as inter-turn trace spacing, producing compact algebraic expressions that mirror CSA principles. While the inter-layer distance term $O ^ { \prime }$ was pruned by SR-KAN (and the PySR baseline) for the specific dataset due to its high correlation to $N _ { L } .$ , its impact is anticipated to be far more dominant in WPT applications, where significantly larger $O ^ { \prime }$ geometry strongly governs magnetic field coupling and spatial field distribution.

The FEA simulation environment was experimentally validated against 55 custom-fabricated PCB prototypes, spanning a wide range of inductances and layer counts. Excluding a single physical manufacturing defect, the FEA solver aligned closely with hardware measurements, exhibiting a mean relative error of −0.23% and a mean absolute relative error of 2.39% with a standard deviation of 3.10%. When evaluated directly to physical windings, the refined MLR monomial and the SR-KAN-discovered equation achieved mean absolute relative errors of 7.11% and 5.98%, respectively. Both formulas maintained tight tracking bounds across demanding physical edge cases, confirming that the equations translate effectively to real-world hardware.

Beyond planar magnetics, the proposed methodology offers a fully domain-agnostic explainable pipeline that can be readily generalized to any data-driven modeling or optimization dataset in power electronics, without being restricted to inductance estimation or planar components. To support further research in data-driven magnetics, the complete dataset of over 10,000 FEA simulations, alongside the physical prototype measurement database, has been made publicly available [23], with detailed guidelines and routines for loading the data hosted in a public GitHub repository [29].

## REFERENCES

[1] H. Wouters and W. Martinez, “Integrated Inductor-Transformers for High-Frequency Converters: An Overview,” IEEE Transactions on Power Electronics, vol. 40, DOI 10.1109/TPEL.2025.3569420, no. 9, pp. 13 157–13 176, 2025.

[2] D. Lee, H.-P. Kieu, J. Kim, S. Choi, S. Kim, and W. Martinez, “A ganbased improved zeta converter with integrated planar transformer for 800-v electric vehicles,” IEEE Transactions on Industrial Electronics, vol. 71, DOI 10.1109/TIE.2024.3390728, no. 12, pp. 15 815–15 825, 2024.

[3] R. Barzegarkhoo, F. Groon, A. Sengupta, and M. Liserre, “Wide voltage range reconfigurable dab converter realized by monolithic bidirectional gan/discrete sic-fets, and planar transformer,” IEEE Transactions on Power Electronics, vol. 40, DOI 10.1109/TPEL.2025.3587501, no. 11, pp. 17 366–17 383, 2025.

[4] A. Nabih, F. Jin, and Q. Li, “Efficient Integrated Transformer-Inductor With High PCB Utilization and Optimized Core,” IEEE Transactions on Industrial Electronics, vol. 71, DOI 10.1109/TIE.2023.3294637, no. 6, pp. 5653–5662, 2024.

[5] Z. Yan, G. Liu, S. Luan, Y. Gao, R. Wang, B. F. KjA¦rsgaard, M. R.<sup>˜</sup> Nielsen, B. Rannestad, H. Zhao, and S. Munk-Nielsen, “Gate driver power supply with low-capacitance-coupling and constant output voltage for medium-voltage sic mosfets,” IEEE Transactions on Power Electronics, vol. 40, DOI 10.1109/TPEL.2025.3538907, no. 6, pp. 8194–8205, 2025.

[6] Z. Zhang, K. Xu, Z. Li, D. Ye, Z. Xu, M. He, and X. Ren, “1-kV Input 1-MHz GaN Stacked Bridge LLC Converters,” IEEE Transactions on Industrial Electronics, vol. 67, DOI 10.1109/TIE.2019.2952806, no. 11, pp. 9227–9237, 2020.

[7] Y. Liu, H. Wu, Z. Ge, and G. Ji, “Magnetic integration for multiple resonant converters,” IEEE Transactions on Industrial Electronics, vol. 70, DOI 10.1109/TIE.2022.3229381, no. 8, pp. 7604–7614, 2023.

[8] Y. Liu, H. Wu, C. Jiang, X. Ren, S. An, and T. Long, “Advances and challenges for magnetic planarization and integration - an overview,” IEEE Open Journal of the Industrial Electronics Society, vol. 7, DOI 10.1109/OJIES.2026.3679790, pp. 747–765, 2026.

[9] Y. Liu, H. Wu, and G. Ji, “Inductance calculation method considering the window effect of planarized magnetic core,” IEEE Transactions on Power Electronics, vol. 38, DOI 10.1109/TPEL.2023.3299984, no. 10, pp. 12 999–13 007, 2023.

[10] G. K. Y. Ho, Y. Fang, and B. M. H. Pong, “A Multiphysics Design and Optimization Method for Air-Core Planar Transformers in High-Frequency LLC Resonant Converters,” IEEE Transactions on Industrial Electronics, vol. 67, DOI 10.1109/TIE.2019.2910023, no. 2, pp. 1605– 1614, 2020.

[11] H. A. Aebischer, “Inductance formula for rectangular planar spiral inductors with rectangular conductor cross section,” Advanced Electromagnetics, vol. 9, DOI 10.7716/aem.v9i1.1346, no. 1, pp. 1–18, 2020.

[12] E. Noh, J. So, J.-H. Park, and S.-H. Lee, “Machine-Learning-Based Optimal Design of a 100 kW, 99% Efficiency, 70 kV Insulated, Leakage-Inductance-Integrated Medium-Frequency Transformer,” IEEE Transactions on Industrial Electronics, DOI 10.1109/TIE.2026.3663785, pp. 1– 11, 2026.

[13] T. Boulanger, V. Cirimele, M. Ricco, and E. Monmasson, “Z-SpecNNet: A Real-Time Embedded NN-Based Parameters Estimation for WPT Systems,” IEEE Transactions on Industrial Electronics, vol. 72, DOI 10.1109/TIE.2024.3515264, no. 7, pp. 7595–7604, 2025.

[14] T. Papadopoulos and A. Antonopoulos, “Inductance estimation for high-power multilayer rectangle planar windings,” IEEE Journal of Emerging and Selected Topics in Industrial Electronics, vol. 6, DOI 10.1109/JESTIE.2025.3564119, no. 3, pp. 1082–1088, 2025.

[15] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in Proceedings of the 31st International Conference on Neural Information Processing Systems, pp. 4768–4777, 2017.

[16] A. Fisher, C. Rudin, and F. Dominici, “All models are wrong, but many

are useful: Learning a variable’s importance by studying an entire class of prediction models simultaneously,” Journal of Machine Learning Research, vol. 20, no. 177, pp. 1–81, 2019. [Online]. Available: http://jmlr.org/papers/v20/18-760.html

[17] L. Breiman, “Random forests,” Machine Learning, vol. 45, DOI 10.1023/A:1010933404324, pp. 5–32, 2001.

[18] T. Chen and C. Guestrin, “Xgboost: A scalable tree boosting system,” in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, DOI 10.1145/2939672.2939785, pp. 785–794, 2016. [Online]. Available: https://doi.org/10.1145/2939672.2939785

[19] H. Drucker, C. J. C. Burges, L. Kaufman, A. Smola, and V. Vapnik, “Support vector regression machines,” in Advances in Neural Information Processing Systems, vol. 9, 1996. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 1996/file/d38901788c533e8286cb6400b40b386d-Paper.pdf

[20] M. A. Buhler and G. Guillen-Gosalbez, “Sr-kan: A kolmogorov–arnold¨ network guided symbolic regression framework,” Computers & Chemical Engineering, vol. 213, DOI 10.1016/j.compchemeng.2026.109721, p. 109721, 2026.

[21] Z. Liu, Y. Wang, S. Vaidya, F. Ruehle, J. Halverson, M. Soljacic, T. Y. Hou, and M. Tegmark, “KAN: Kolmogorov–arnold networks,” in The Thirteenth International Conference on Learning Representations, 2025. [Online]. Available: https://openreview.net/forum?id=Ozo7qJ5vZi

[22] M. Cranmer, “Interpretable machine learning for science with pysr and symbolicregression.jl,” 2023. [Online]. Available: https: //arxiv.org/abs/2305.01582

[23] T. Papadopoulos, A. Antonopoulos, and S. Rigas, “Dataset of Multilayer Rectangle-shaped Planar Windings,” 2026. [Online]. Available: https: //doi.org/10.5281/zenodo.21762502

[24] S. Mohan, M. del Mar Hershenson, S. Boyd, and T. Lee, “Simple accurate expressions for planar spiral inductances,” IEEE Journal of Solid-State Circuits, vol. 34, DOI 10.1109/4.792620, no. 10, pp. 1419– 1424, 1999.

[25] Z. Liu, M. Tegmark, P. Ma, W. Matusik, and Y. Wang, “Kolmogorovarnold networks meet science,” Phys. Rev. X, vol. 15, DOI 10.1103/4t7t-v19l, p. 041051, Dec. 2025. [Online]. Available: https: //link.aps.org/doi/10.1103/4t7t-v19l

[26] A. Tschalzev, S. Marton, S. Ludtke, C. Bartelt, and H. Stuckenschmidt,¨ “A data-centric perspective on evaluating machine learning models for tabular data,” in The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. [Online]. Available: https://openreview.net/forum?id=kWTvdSSH5W

[27] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay, “Scikit-learn: Machine Learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.

[28] S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. Katz, J. Himmelfarb, N. Bansal, and S.-I. Lee, “From local explanations to global understanding with explainable ai for trees,” Nature Machine Intelligence, vol. 2, no. 1, pp. 2522–5839, 2020.

[29] S. Rigas, T. Papadopoulos, G. Alexandridis, and A. Antonopoulos, “MLRPW,” https://github.com/srigas/MLRPW, 2026.