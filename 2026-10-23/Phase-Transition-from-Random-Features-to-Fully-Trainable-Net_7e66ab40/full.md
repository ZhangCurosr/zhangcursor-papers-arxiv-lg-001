# Phase Transition from Random Features to Fully Trainable Networks

Xin He, Huan Lao, Xiliang Lu, and Shijun Zhang<sup>\*</sup>

Abstract. Constructing accurate and computationally tractable approximation spaces is a fundamental task in scientific computing. Partially trainable neural networks bridge fixed random features and fully adaptive representations. This paper quantifies a phase transition pattern in approximation bounds and observes analogous behavior in training errors: a rapid but smooth initial improvement followed by diminishing returns, with a sharper transition under greater initialization mismatch. Under polynomial spectral decay with exponent $\alpha > 1$ and leverage compatibility, we establish a squared $L _ { 2 }$ approximation rate $O ( ( \kappa _ { P } \log ( m / \delta ) / m ) ^ { \alpha } )$ for two-layer networks of width � and trainable proportion $P ,$ where $\kappa _ { P } = ( P / \kappa _ { 1 } ^ { * } + ( 1 - P ) / \kappa _ { 0 } ^ { * } ) ^ { - 1 }$ . The bound holds uniformly over the unit ball of a reproducing kernel Hilbert space (RKHS) with probability at least $1 - \delta .$ The constants $\boldsymbol { \kappa } _ { 0 } ^ { * }$ and $\boldsymbol { \kappa } _ { 1 } ^ { * }$ quantify the fixed and trainable groups’ sampling mismatches relative to the target RKHS. For spherical biased-ReLU features in dimension $d ,$ we verify these assumptions with $\alpha = ( d + 2 ) / ( d - 1 )$ and $\kappa _ { 0 } ^ { * } = 4 \sqrt { 2 } \tau ( d + 1 ) / d$ for initialization scale $\tau \geq 1$ , and $\kappa _ { 1 } ^ { * } = 1$ under ridge leverage sampling. For $d = 5$ and $\tau = 1 0 ,$ , a trainable proportion of approximately 4.1% captures 90% of the reduction in the approximation bound from $P = 0$ to $P = 1$ . A structured three-layer example illustrates how a few trainable first-layer neurons exploit fixed downstream blocks through nonlinear composition. Numerical experiments across diverse settings, including networks with up to 15 layers, exhibit similar phase transitions in training errors.

## Contents

1. Introduction 2   
2. Preliminaries and Problem Setting 6   
3. General Approximation Results 12   
4. Analysis of Spherical Biased-ReLU Features 18   
5. A Structured Three-Layer Illustration 22   
6. Numerical Evidence of Phase Transitions . 24   
7. Discussion and Conclusions 30   
Overview of the Appendices . 31   
A. Additional Technical Notes 31   
B. Proofs of Main Results in Section 3 – General Bounds . 33   
C. Proofs of Main Results in Section 4 – Specific Spherical Setting . 42   
D. Error Bounds for the Structured Three-Layer Example 5.1 54   
E. Complementary Experiments . . 59   
References . 65

## 1. Introduction

The pursuit of accurate, eficient, and stable representations lies at the heart of scientific computing. A key task is to construct suitable approximation spaces. Many classical methods, such as polynomial, Fourier, and wavelet approximation, prescribe a family of functions and approximate the target through their linear combinations. Random feature methods (RFMs) [45, 48] follow a similar principle, using fixed sampled features whose ability to capture the target depends on how well the initialization distribution matches it [4, 44, 52]. Feed-forward neural networks (FNNs) ofer a more adaptive approach: training the hidden parameters allows the feature functions themselves to adapt to the target, yielding strong expressive power [46, 59, 80]. RFMs and fully trainable FNNs thus represent two endpoints, connected by partially trainable networks in which a proportion of hidden parameters are trainable and the rest randomly fixed. This paper investigates the phase transition between these endpoints, addressing the question:

## How does theperformance ofpartially trainable neural networks vary as the trainable proportion increases?

To investigate this question, this paper studies the �-trainable networks, where the same proportion $P \in [ 0 , 1 ]$ of neurons in each hidden layer are trainable and the rest are randomly fixed. Although arbitrary subsets of network parameters can be made trainable, the present formulation assigns trainability at the neuron level. Our approximation analysis shows that the efect of the trainable proportion depends on how well the fixed and trainable neuron groups match the target function class. The corresponding theoretical bounds and numerical experiments together identify a distinct phase transition pattern: A small trainable group can ofset the mismatch between the initialization distribution and the target function class, yielding substantial performance gains at small � followed by diminishing returns, and greater initialization mismatch sharpens the transition.

Building on the approximation results for two-layer �-trainable networks with general activation functions, the analysis of spherical biased-ReLU features provides the explicit rates that quantify this dependence on initialization. Figure 1 illustrates qualitative agreement between the predicted and observed phase transitions across initialization scales $\tau \geq 1$ Within this family, the chosen admissible mismatch constant increases linearly with �. Increasing � generally sharpens phase transitions. More concretely, for $d = 5$ and $\tau = 1 0$ , a trainable proportion of only about 4.1% captures 90% of the total reduction in the approximation bound from $P = 0$ to � = 1.

A structured three-layer example further demonstrates how a small trainable group uses randomly fixed downstream neurons through nonlinear composition enabled by depth.

![](images/96d9548dad399db0208c90158aeb258c8718dba9c9468d01879867d492bfb25e.jpg)  
(a) Threshold for 90% gain.

![](images/cb84a14eb3a9bdd4055d8fa5ea1ef750f66716f871c4852b6fda195c4b551260.jpg)  
(b) Normalized gain.

Figure 1. Phase transitions for two-layer spherical biased-ReLU �-trainable networks with $d = 5$ Red curves are derived from the approximation bound, while blue curves show training results for a fixed RKHS target. Increasing the fixed initialization scale $\tau \geq 1$ raises the mismatch constant from its minimum at $\tau = 1$ . Both theory and training show qualitatively similar transition shapes and sharper transitions at larger �. Quantitative diferences are expected because training also depends on finite sampling, the optimizer, and the training budget. Threshold $P _ { 0 . 1 }$ marks 90% normalized gain. Shaded bands indicate 95% bootstrap intervals. See experimental details in Appendix E.1.  
![](images/43bb22b8070c8df3cc0effb085691682161130ad2c5287428f1a96652c01fc3f.jpg)  
(a) Depth variation from 3 to 7 at width 64.

![](images/54882412880a72fdd7c0775f4b36711353d0de31001b468a5062b70f90ea16b3.jpg)  
(b) Width variation from 32 to 256 at depth 6.  
Figure 2. Representative numerical normalized gains for �-trainable ReLU networks on a surface target function. Gains are normalized by the observed loss reduction between $P = 0$ and $P = 1$ . Both depth and width comparisons show rapid gains at small �, followed by diminishing gains.

Extensive numerical experiments across target functions and architectures exhibit similar phase transition patterns, with representative width and depth comparisons in Figure 2.

## 1.1. Main Results

The main contributions of this work are summarized as follows.

(i) Small-� Phase Transition for Shallow Networks. For two-layer �-trainable networks, the analysis samples the trainable group from an adjustable distribution to quantify how the trainable proportion and initialization mismatch afect approximation bounds.

• General approximation rates. Theorem 3.2 gives a suficient width condition for approximating the RKHS unit ball uniformly to any prescribed tolerance with high probability.

Under polynomial spectral decay with exponent $\alpha > 1$ and leverage compatibility, Theorem 3.6 then gives the approximation rate for width �

$$
O \bigg ( \Big ( \Big ( \frac { P } { \kappa _ { 1 } ^ { * } } + \frac { 1 - P } { \kappa _ { 0 } ^ { * } } \Big ) \frac { m } { \log ( m / \delta ) } \Big ) ^ { - \alpha } \bigg ) , \quad P \in [ 0 , 1 ] ,
$$

with probability at least $1 - \delta .$ , where $\delta \in ( 0 , 1 )$ . The constants $\boldsymbol { \kappa } _ { 1 } ^ { * }$ and $\boldsymbol { \kappa } _ { 0 } ^ { * }$ control how well the sampling distributions of the trainable and randomly fixed groups match the target function class, respectively. Both groups contribute to the approximation bound, and their relative contributions depend on the mismatch ratio $\kappa _ { 0 } ^ { * } / \kappa _ { 1 } ^ { * }$

• Phase transition sharpness. Corollary 3.10 shows that when $\kappa _ { 0 } ^ { * } / \kappa _ { 1 } ^ { * } \gg 1$ , a trainable proportion of order $( \kappa _ { 0 } ^ { * } / \kappa _ { 1 } ^ { * } ) ^ { - 1 }$ sufices to capture any fixed fraction of the reduction between the endpoint bounds. A larger mismatch ratio produces a sharper transition in the bound.

• Spectral and mismatch verification. For spherical biased-ReLU features on $\mathbb { S } ^ { d - 1 }$ , Propositions 4.1 and 4.2 establish the spectral exponent $\alpha = ( d + 2 ) / ( d - 1 )$ and the admissible mismatch constant $\kappa _ { 0 } ^ { * } = 4 \sqrt { 2 } \tau ( d + 1 ) / d$ . With $\kappa _ { 1 } ^ { * } = 1$ from ridge leverage sampling of the trainable group, Theorem 4.3 gives the explicit spherical rate. Increasing the fixed initialization scale $\tau \geq 1$ generally sharpens predicted and observed transitions (Figure 1).

(ii) Similar Transitions in Deep Networks. A deep neural network example and numerical experiments show similar small-� transitions and further examine the roles of depth and randomly fixed neurons.

• Gainsfrom nonlinear composition. Example 5.1 provides explicit approximation upper bounds for a structured three-layer network in which a few trainable first-layer neurons use several randomly fixed downstream neurons. The bounds exhibit a similar small-� phase transition. At selected trainable neuron counts in the first hidden layer, the threelayer construction also gives a smaller error upper bound than its two-layer counterpart, highlighting the gains from nonlinear composition enabled by depth.

• Phase transitions in training performance. Across three targets and diferent widths and depths, most tested configurations in Section 6 recover a substantial fraction of the observed loss reduction between the two endpoints at small �, followed by diminishing gains. The transition strength varies with the target and architecture, with representative depth and width results in Figure 2. Besides, Table 4 shows lower training losses with more randomly fixed neurons at a fixed trainable neuron count per hidden layer, supporting the expressive role for the fixed group.

## 1.2. Related Work

The following discussion reviews approximation and learning theory for RFMs, partially trainable networks, and fully trainable FNNs, as well as studies of pruning and parameter freezing. For each topic, it explains the connections to our work and the diferences in model construction, analysis, or research objectives.

Approximation Theory for Random Feature Methods. RFMs approximate the infinitedimensional feature representation of a kernel using finitely many random features sampled from a prescribed distribution [48,49]. Early analyses focused on approximating the kernel itself, particularly with random Fourier feature analysis [67,68]. Subsequent operator-based work considered general integral feature maps and studied RFMs via a framework of function approximation and statistical learning. In this framework, the required feature number is governed by the spectrum and efective dimension of the associated integral operator [4, 36, 52]. Beyond sampling from the original feature distribution, leverage-based sampling schemes have been developed at both the population and empirical levels to reduce the required number of features [2,4,34,35]. When the feature map is specialized to ReLU neurons, approximation bounds have also been established for two-layer networks with randomly sampled hidden parameters [19, 25].

RFMs also have limitations in approximation capacity. Under standard Gaussian inputs, approximating even a single ReLU neuron can require exponentially many features or exponentially large output coeficients in the input dimension [74]. Approximation lower bounds derived using ridgelet analysis show that two-layer RFMs with bounded hidden parameters can fail to achieve universal approximation when the sampling support omits a nonzero part of the target’s ridgelet spectrum [32].

Our analysis extends the operator framework in [4] from a single sampling distribution to two independent groups of fixed sizes sampled from separate distributions, giving unified approximation bounds over $P \in [ 0 , 1 ]$ . Beyond the sampling support restrictions studied in [32], our mismatch factor quantifies insuficient sampling mass relative to the target RKHS and its efect on the suficient network width. The bounds further show how a small, appropriately chosen trainable group can compensate for the initialization mismatch.

Learning Theory for Partially Trainable Networks. Partially trainable networks have been studied under various constraints on which parameters are optimized. Semi-random feature models combine randomly fixed and trainable components within individual units [27], while other work establishes universal approximation for networks with fixed random weights and trainable biases [71]. Optimization and function-space analyses also provide convergence and complexity results for models with a randomly fixed first hidden layer [8, 9]. Related approximation studies investigate representations with a small number of target-dependent parameters [58, 75], as well as multigrade and layer-wise approximation constructions [77, 78].

Our analysis considers a diferent form of partial trainability: hidden neurons within each layer are divided into trainable and randomly fixed groups, and we study the resulting finite-width approximation properties.

Approximation Theory for Feed-Forward Neural Networks. Classical universal approximation theorems establish the density of FNNs with suitable activations in spaces of continuous functions [12, 24]. Quantitative approximation theory relates approximation accuracy to network size, target regularity, and architectural constraints. For shallow networks, Barron-type and variation spaces provide important frameworks for this analysis [5, 13, 14, 63], with approximation rates and optimality results available under various assumptions on the activation and error norm [38,61,62,64]. More general approximationspace frameworks have also been developed for deep networks [20].

Quantitative bounds for Hölder, Sobolev, and Besov classes have been established using ReLU and related rectified-power networks [21, 37, 40, 55, 59, 60, 69, 73]. Other results address analytic and bandlimited functions [15,43], as well as smooth functionals [66]. Further studies investigate how activation choices and constraints on network size or parameter magnitude afect approximation capabilities [26, 30, 31, 56, 57, 76].

Beyond target regularity, approximation theory also examines the role ofinput geometry [6,7,53] and architectural structure, including convolutional networks, dynamical-systems formulations, Transformers, and structured network models [10,11,33,54,79,81]. Connections with finite element methods and hierarchical basis constructions provide additional perspectives [22, 23].

Our work complements these studies by examining how partial trainability afects approximation within a given architecture and target function class.

Lottery Ticket Hypothesis and Parameter Freezing. The lottery ticket hypothesis states that a randomly initialized dense network contains a smaller subnetwork that, when trained in isolation, can attain performance comparable to the original network [17]. Related studies show that suitably pruned randomly initialized networks may contain subnetworks capable of approximating target networks with little or no weight training [39, 50]. Instead, parameter freezing trains only a selected subset of parameters while holding the remainder fixed, often at their random initializations [51, 72]. Studies of parameter freezing investigate parameter selection rules and freezing ratios, often with an emphasis on practical performance and training eficiency.

Our setting is related to these studies but difers in both construction and objective. A prescribed proportion of hidden neurons is trainable. The remaining hidden parameters stay fixed after initialization, while all neurons are retained in the forward computation. The objective is to study approximation capacity as the trainable proportion varies, rather than to search for subnetworks or design parameter selection rules.

Organization. The rest of the paper is organized as follows. Section 2 introduces the model, approximation task and analytical framework. Section 3 presents the general approximation bound, explicit polynomial rates, and the resulting phase transition phenomenon. Section 4 specializes the analysis to specific spherical biased-ReLU features. Section 5 provides the structured three-layer illustration. Section 6 reports the numerical illustrations, and Section 7 concludes the paper. Supplementary proofs are provided in the appendices.

## 2. Preliminaries and Problem Setting

Notation. Let $\mathbb { N } _ { + } : = \{ 1 , 2 , 3 , . . . \}$ and $\mathbb { N } : = \mathbb { N } _ { + } \cup \{ 0 \}$ . For $n \in \mathbb { N } _ { + }$ , write $[ n ] = \{ 1 , \dots , n \}$ with $[ 0 ] = \emptyset$ . Empty sums are understood to be zero. We denote the set of real numbers by R. Write $A \lesssim B$ and $B \gtrsim A$ when $A \leq c B$ for a constant $c > 0$ . And $A \asymp B$ means $A \lesssim B$ and $B \lesssim A$ . Vectors written as tuples are understood as column vectors. For $\boldsymbol { x } \in \mathbb { R } ^ { d } , \| \boldsymbol { x } \| _ { p }$ denotes the $\ell _ { p } { \ - } { \mathrm { n o r m } }$ for $p \in [ 1 , + \infty )$ . For a measurable function $f : \mathbb { R } ^ { d } \to \mathbb { R } , \| f \| _ { L _ { p } ( d \mu ) }$ denotes the $L _ { p } { \mathrm { - n o r m } }$ with respect to $\mu .$ . Table 5 in the appendix summarizes the main notations for quick reference.

## 2.1. Model Space: �-Trainable Neural Networks

Feed-forward Neural Networks (FNNs). Let $W \in \mathbb { N } _ { + }$ and $L \ge 2$ . We consider a feedforward neural network $f _ { L } \in \mathcal { F } ( W , L ) : \mathbb { R } ^ { d } \to$ R ofwidth � and depth $L$ , with $L - 1$ hidden layers. Set $m _ { 0 } = d$ and $m _ { L } = 1$ . For $\ell \in [ L - 1 ]$ , let $\theta _ { \ell , j } \in \mathbb { R } ^ { m _ { \ell - 1 } + 1 }$ denote the parameter vector of the $j \mathrm { - t h }$ neuron in the ℓ-th hidden layer and $\phi : \mathbb { R } \to \mathbb { R }$ be the activation function. The network is then defined recursively by

$$
\begin{array} { r l } & { f _ { 0 } ( x ) = ( x , 1 ) , } \\ & { f _ { \ell } ( x ) = \bigl ( \phi ( x ; \theta _ { \ell , 1 } ) , \ldots , \phi ( x ; \theta _ { \ell , m _ { \ell } } ) , 1 \bigr ) , \quad \ell \in [ L - 1 ] , } \\ & { f _ { L } ( x ) = \displaystyle \sum _ { j = 1 } ^ { m _ { L - 1 } } \beta _ { j } \phi ( x ; \theta _ { L - 1 , j } ) , } \end{array}\tag{2.1}
$$

where $\beta = ( \beta _ { 1 } , \dots , \beta _ { m _ { L - 1 } } ) \in \mathbb { R } ^ { m _ { L - 1 } }$ denotes the output coeficient vector. The width of the network is defined as $W = \operatorname* { m a x } \{ m _ { 0 } , m _ { 1 } , . . . , m _ { L } \}$ . With a slight abuse of notation, we define the feature induced by the �-th neuron as $\phi ( x ; \theta _ { \ell , j } ) : = \phi ( \langle \theta _ { \ell , j } , f _ { \ell - 1 } ( x ) \rangle )$

�-Trainable Neural Networks. Let $P \in [ 0 , 1 ]$ denote the proportion of trainable neurons in each hidden layer. For every $\ell \in [ L - 1 ]$ , define $m _ { \ell , \mathrm { f r e e } } ( P ) = \lfloor P m _ { \ell } \rfloor$ and $m _ { \ell , \mathrm { f i x } } ( P ) =$ $m _ { \ell } - m _ { \ell , \mathrm { f r e e } } ( P )$ . When the dependence on � is clear, write $m _ { \ell , \mathrm { f r e e } }$ and $m _ { \ell , \mathrm { f i x } }$ without their arguments. The hidden-layer widths $\{ m _ { 1 } , \hdots , m _ { L - 1 } \}$ are fixed when comparing diferent values of �. For indexing, the trainable neurons are listed before the randomly fixed neurons in each hidden layer. Specifically,

$$
\theta _ { \ell , j } \in \mathbb { R } ^ { m _ { \ell - 1 } + 1 } , \ 1 \le j \le m _ { \ell , \mathrm { f r e e } } , \ \mathrm { a n d } \ \theta _ { \ell , j } = \bar { \theta } _ { \ell , j - m _ { \ell , \mathrm { f r e e } } } \sim \rho _ { 0 , \ell } , \ m _ { \ell , \mathrm { f r e e } } + 1 \le j \le m _ { \ell } ,
$$

where $\theta _ { \ell , j }$ denotes the parameter of a trainable neuron while ${ \bar { \theta } } _ { \ell , j - m _ { \ell , \mathrm { f r e e } } }$ denotes the parameter sampled from a prescribed distribution $\rho _ { 0 , \ell }$ . In the two-layer setting, write $\rho _ { 0 } = \rho _ { 0 , 1 }$

With $\phi ( x ; \theta _ { \ell , j } ) : = \phi ( \langle \theta _ { \ell , j } , f _ { \ell - 1 } ^ { ( P ) } ( x ) \rangle )$ , the �-trainable network is defined by

$$
f _ { 0 } ^ { ( P ) } ( x ) = ( x , 1 ) ,
$$

$$
f _ { \ell } ^ { ( P ) } ( x ) = \Big ( \big ( \phi ( x ; \theta _ { \ell , j } ) \big ) _ { j = 1 } ^ { m _ { \ell , \mathrm { f r e e } } } , \big ( \phi ( x ; \bar { \theta } _ { \ell , j } ) \big ) _ { j = 1 } ^ { m _ { \ell , \mathrm { f i x } } } , 1 \Big ) , \quad \ell \in [ L - 1 ] ,
$$

$$
f _ { L } ^ { ( P ) } ( x ) = \sum _ { j = 1 } ^ { m _ { L - 1 , \mathrm { f r e e } } } \beta _ { j } \phi ( x ; \theta _ { L - 1 , j } ) + \sum _ { j = 1 } ^ { m _ { L - 1 , \mathrm { f r } } } \beta _ { m _ { L - 1 , \mathrm { f r e e } } + j } \phi ( x ; \bar { \theta } _ { L - 1 , j } ) ,\tag{2.2}
$$

where the output coeficient vector $\beta = ( \beta _ { 1 } , . . . , \beta _ { m _ { L ^ { - 1 } } } ) = ( \beta _ { \mathrm { f r e e } } , \beta _ { \mathrm { f i x } } )$ is partitioned according to the trainable and fixed hidden neurons. All entries of $\beta$ remain trainable. For a given realization of the fixed hidden parameters, the resulting class of �-trainable neural networks is denoted by $\mathcal { F } _ { P } ( W , L )$ . Figure 3 illustrates the architecture.

![](images/7770c944e02f0f1de6d75fe3f61b91e3a4c1db62542adf21a95afa1a97b0c360.jpg)  
Figure 3. A �-trainable network with hidden layers of equal width �. Each hidden layer contains $\lfloor P W \rfloor$ trainable neurons and $W - \left\lfloor P W \right\rfloor$ fixed neurons. Thus, the network has $O ( P W ^ { 2 } L + W )$ trainable parameters out of $O ( W ^ { 2 } L )$ total parameters.

Notably, there are two limiting cases. At $P = 0$ , all hidden parameters are randomly initialized and held fixed, so only the output coeficients are trained. Thus, $\mathcal { F } _ { 0 } ( W , L )$ corresponds to an RFM class. $\mathrm { A t } P = 1$ , all hidden parameters are trainable, so the �-trainable network class coincides with the fully trainable FNN class, and hence $\mathcal { F } _ { 1 } ( W , L ) = \mathcal { F } ( W , L )$

Although the model (2.2) is defined for general depth �, the main approximation theory concerns two-layer networks in Sections 3 and 4. Section 5 provides a structured three-layer illustration, and Section 6 examines deeper networks numerically.

Remark 2.1 (Neuron-wise Trainability). Partial training can also be defined at the level of individual parameter entries. Here, each hidden neuron is assigned to the trainable or fixed group through its entire incoming parameter vector $\theta _ { \ell , j }$ , including the bias. In deeper networks, the outputs of fixed neurons may still change as earlier trainable layers change.

## 2.2. Approximation Task: Two Complementary Views

For the two-layer analysis, set $m = m _ { 1 }$ and assume that �� is an integer, so that $m _ { \mathrm { f r e e } } = P m$ and $m _ { \mathrm { f i x } } = ( 1 - P ) m$ . Let $\boldsymbol { \chi } \subset \mathbb { R } ^ { d }$ be the input domain and let $f ^ { * } : X \to \mathbb { R }$ be a target function. The approximation task by a two-layer �-trainable network is described from two complementary viewpoints.

Finite-dimensional View. Given input points $\{ x _ { 1 } , \ldots , x _ { n } \} \subset X$ and the corresponding target values $\{ f ^ { * } ( x _ { i } ) \} _ { i = 1 } ^ { n }$ , the task is to choose the output coeficients and trainable hidden parameters. By (2.2), a two-layer �-trainable network $f \in { \mathcal { F } } _ { P } ( W , 2 )$ has the form

$$
f _ { 2 } ^ { ( P ) } ( x ) = \sum _ { j = 1 } ^ { m _ { \mathrm { f r e e } } } \beta _ { j } \phi ( x ; \theta _ { j } ) + \sum _ { j = 1 } ^ { m _ { \mathrm { f i x } } } \beta _ { m _ { \mathrm { f r e e } } + j } \phi ( x ; \bar { \theta } _ { j } ) .
$$

With the randomly sampled hidden parameters held fixed during optimization, the expected best approximation error on these input points is

$$
\mathcal { E } _ { n , m } ( f ^ { * } , \mathcal { F } _ { P } ( W , 2 ) ) = \mathbb { E } _ { \bar { \theta } _ { 1 : m _ { \mathrm { f i x } } } } \Big [ \operatorname* { i n f } _ { \beta , \theta _ { 1 : m _ { \mathrm { f r e e } } } } \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \big ( f ^ { * } ( x _ { i } ) - f _ { 2 } ^ { ( P ) } ( x _ { i } ) \big ) ^ { 2 } \Big ] .\tag{2.3}
$$

When $P = 0$ , only the output coeficients $\beta$ are optimized, which gives a linear least-squares problem. When $m _ { \mathrm { f r e e } } > 0$ , jointly optimizing the output coeficients and trainable hidden parameters generally yields a nonlinear, non-convex problem.

Population View. Let $\mu$ be a probability measure on the input space $\chi .$ . For any target function $f ^ { * } \in L _ { 2 } ( X , d \mu )$ , its squared $L _ { 2 } ( d \mu )$ norm is defined as

$$
\| f ^ { * } \| _ { L _ { 2 } ( d \mu ) } ^ { 2 } = \int _ { \chi } | f ^ { * } ( x ) | ^ { 2 } d \mu ( x ) .
$$

Replacing the average in (2.3) with $\| \cdot \| _ { L _ { 2 } ( d \mu ) } ^ { 2 }$ gives the population approximation problem

$$
\mathcal { E } _ { m } ( f ^ { * } , \mathcal { F } _ { P } ( W , 2 ) ) = \mathbb { E } _ { \bar { \theta } _ { 1 : m _ { \mathrm { f i x } } } } \prod _ { \beta , \theta _ { 1 : m _ { \mathrm { f r e c } } } } \Vert f ^ { * } - f _ { 2 } ^ { ( P ) } \Vert _ { L _ { 2 } ( d \mu ) } ^ { 2 } ] .\tag{2.4}
$$

The finite-dimensional problem (2.3) evaluates approximation on the prescribed input points, while the population problem (2.4) measures approximation in $L _ { 2 } ( d \mu )$ . These formulations define the expected best approximation errors over the fixed initialization, while the subsequent results are stated as high-probability bounds over the sampled parameters.

## 2.3. Target Function Space: The Reproducing Kernel Hilbert Space

Let $\mu$ be the probability measure on the input space $\chi$ and $\rho$ on the parameter space $\Theta$

Assumption 2.2 (Operator Regularity). The feature map � is jointly measurable and belongs to $L _ { 2 } ( X \times \Theta , \mu \otimes \rho )$

Under Assumption 2.2, the feature family $\{ \phi ( \cdot ; \theta ) : \theta \in \Theta \}$ induces the positive semidefinite kernel $k _ { \rho }$ on a $\mu$ full measure subset<sup>1</sup> of $\chi$

$$
k _ { \rho } ( x , x ^ { \prime } ) : = \int _ { \Theta } \phi ( x ; \theta ) \phi ( x ^ { \prime } ; \theta ) d \rho ( \theta ) = \mathbb { E } _ { \theta \sim \rho } [ \phi ( x ; \theta ) \phi ( x ^ { \prime } ; \theta ) ] .\tag{2.5}
$$

Let $\mathcal { H } _ { \rho } \subseteq L _ { 2 } ( X , d \mu )$ denote the reproducing kernel Hilbert space (RKHS) induced by $k _ { \rho }$ We assume that the target $f ^ { * }$ in the main analysis belongs to the RKHS $\mathcal { H } _ { \rho }$

Integral Operator Induced by the Feature Map. To compare finite sampled features with the integral representation oftargets in $\mathcal { H } _ { \rho }$ , define the feature operator $S _ { \rho } : L _ { 2 } ( \Theta , d \rho ) $ $L _ { 2 } ( X , d \mu )$ and its adjoint $S _ { \rho } ^ { * } : L _ { 2 } ( X , d \mu )  L _ { 2 } ( \Theta , d \rho )$ by

$$
( S _ { \rho } a ) ( x ) : = \int _ { \Theta } a ( \theta ) \phi ( x ; \theta ) d \rho ( \theta ) { \mathrm { ~ a n d ~ } } ( S _ { \rho } ^ { * } g ) ( \theta ) = \langle \phi ( \cdot ; \theta ) , g ( \cdot ) \rangle _ { L _ { 2 } ( d \mu ) } .
$$

Here $a \in L _ { 2 } ( \Theta , d \rho )$ is a coeficient function and $g \in L _ { 2 } ( X , d \mu )$ is an arbitrary square integrable function. The integral operator associated with $k _ { \rho }$ is defined as

$$
T _ { \rho } : = S _ { \rho } S _ { \rho } ^ { * } : L _ { 2 } ( X , d \mu )  L _ { 2 } ( X , d \mu ) .
$$

Equivalently, for any $g \in L _ { 2 } ( X , d \mu )$

$$
( T _ { \rho } g ) ( x ) = \int _ { X } k _ { \rho } ( x , x ^ { \prime } ) g ( x ^ { \prime } ) d \mu ( x ^ { \prime } ) = \int _ { \Theta } \phi ( x ; \theta ) \langle \phi ( \cdot ; \theta ) , g ( \cdot ) \rangle _ { L _ { 2 } ( d \mu ) } d \rho ( \theta ) .\tag{2.6}
$$

Therefore, $T _ { \rho } = \mathbb { E } _ { \theta \sim \rho } [ \phi ( \cdot ; \theta ) \otimes _ { L _ { 2 } ( d \mu ) } \phi ( \cdot ; \theta ) ]$ , where � $\otimes _ { L _ { 2 } ( d \mu ) } b : L _ { 2 } ( X , d \mu ) \to L _ { 2 } ( X , d \mu )$ is an operator such that $( a \otimes _ { L _ { 2 } ( d \mu ) } b ) g = \langle b , g \rangle _ { L _ { 2 } ( d \mu ) } a .$

Spectral Framework. Under Assumption 2.2, $S _ { \rho }$ is a Hilbert-Schmidt operator and hence compact. Thus $T _ { \rho } = S _ { \rho } S _ { \rho } ^ { * }$ is self-adjoint, positive semi-definite, trace-class and compact [65]. By the spectral theorem for compact self-adjoint operators [28], the set of positive eigenvalues $\{ \eta _ { k } \} _ { k \ge 1 }$ of $T _ { \rho }$ is finite or countably infinite. These eigenvalues can be arranged in non-increasing order and counted with multiplicity, $\eta _ { 1 } \geq \eta _ { 2 } \geq \cdot \cdot \cdot > 0$ . The corresponding eigenfunctions $\{ e _ { k } \} _ { k \ge 1 }$ form an orthonormal basis of ker $T _ { \rho } ) ^ { \perp }$ . Therefore, the integral operator $T _ { \rho }$ and its positive fractional powers have the following expansions,

$$
T _ { \rho } = \sum _ { k \geq 1 } \eta _ { k } ( e _ { k } \otimes e _ { k } ) , { \mathrm { ~ a n d ~ } } T _ { \rho } ^ { s } = \sum _ { k \geq 1 } \eta _ { k } ^ { s } ( e _ { k } \otimes e _ { k } ) , s > 0 .\tag{2.7}
$$

In the population analysis, RKHS functions that agree �-almost everywhere are identified, and $\mathcal { H } _ { \rho }$ is equipped with the norm defined below. Under this convention, $\mathcal { H } _ { \rho } =$ Range $( T _ { \rho } ^ { 1 / 2 } ) = \mathrm { R a n g e } ( S _ { \rho } )$ . Consequently, for every target $f ^ { * } \in \mathcal { H } _ { \rho }$ , there exists a unique $q _ { f } ^ { * } \in ( \ker T _ { \rho } ) ^ { \perp }$ and a unique $a _ { f } ^ { \ast } \in ( \ker S _ { \rho } ) ^ { \perp }$ such that

$$
f ^ { * } = T _ { \rho } ^ { 1 / 2 } q _ { f } ^ { * } = S _ { \rho } a _ { f } ^ { * } = \int _ { \Theta } a _ { f } ^ { * } ( \theta ) \phi ( \cdot ; \theta ) d \rho ( \theta ) ,\tag{2.8}
$$

These equalities hold in $L _ { 2 } ( X , d \mu )$ , with the norm defined by

$$
\| f ^ { * } \| _ { \mathcal { H } _ { \rho } } = \operatorname* { i n f } _ { a \in L _ { 2 } ( d \rho ) } \| a \| _ { L _ { 2 } ( d \rho ) } = \| q _ { f } ^ { * } \| _ { L _ { 2 } ( d \mu ) } = \| a _ { f } ^ { * } \| _ { L _ { 2 } ( d \rho ) } .
$$

This representation underlies subsequent analysis of the population approximation error.

## 2.4. Distributional Approximation Strategy

Although the trainable parameters can be chosen freely, the direct optimization over these parameters is generally nonlinear and non-convex. To facilitate the approximation analysis, all hidden-layer parameters are constructed through sampling.

Definition 2.3 (Two-group Sampling). Let $\rho$ be the reference probability measure defining the target RKHS. Let $\rho _ { 0 }$ be the prescribed initialization distribution for the fixed neurons and $\rho _ { 1 }$ an adjustable distribution used to construct the trainable hidden parameters. Assume that both distributions are absolutely continuous with respect to $\rho _ { ; }$ , and write

$$
d \rho _ { i } ( \theta ) = \gamma _ { i } ( \theta ) d \rho ( \theta ) , i \in \{ 0 , 1 \} ,
$$

where $\gamma _ { i }$ is a nonnegative measurable Radon-Nikodym density satisfying $\begin{array} { r l } { \int _ { \Theta } \gamma _ { i } ( \theta ) d \rho ( \theta ) = } & { { } } \end{array}$ 1 for each $i \in \{ 0 , 1 \}$ . Given fixed group sizes $m _ { \mathrm { f r e e } }$ and $m _ { \mathrm { f i x } }$ , set $m = m _ { \mathrm { f r e e } } + m _ { \mathrm { f i x } } \geq 1$ and the trainable proportion $P = m _ { \mathrm { f r e e } } / m$ . The hidden parameters are sampled according to

$$
\theta _ { j } \stackrel { \mathrm { i . i . d . } } { \sim } \rho _ { 1 } , j \in [ m _ { \mathrm { f r e e } } ] , \mathrm { a n d } \bar { \theta } _ { j } \stackrel { \mathrm { i . i . d . } } { \sim } \rho _ { 0 } , j \in [ m _ { \mathrm { f i x } } ] .\tag{2.9}
$$

The two groups $\{ \theta _ { j } \} _ { j = 1 } ^ { m _ { \mathrm { f r e e } } }$ and $\left\{ \bar { \theta } _ { j } \right\} _ { j = 1 } ^ { m _ { \mathrm { f i x } } }$ are mutually independent and have fixed sizes. Their associated mixture measure is $\rho _ { P } { \ ' } = P \rho _ { 1 } + ( 1 - P ) \rho _ { 0 }$ , satisfying

$$
d \rho _ { P } ( \theta ) = \gamma _ { P } ( \theta ) d \rho ( \theta ) , \mathrm { { w i t h ~ d e n s i t y } } \gamma _ { P } ( \theta ) = P \gamma _ { 1 } ( \theta ) + ( 1 - P ) \gamma _ { 0 } ( \theta ) .\tag{2.10}
$$

For the approximation analysis, assume that each $\gamma _ { i } , i \in \{ 0 , 1 \}$ is positive for �-almost every � satisfying $\| \phi ( \cdot ; \theta ) \| _ { L _ { 2 } ( d \mu ) } > 0$ . This ensures both sampling distributions retain the nonzero features used to define $T _ { \rho }$ , while allowing zero densities where the feature vanishes.

The two-group sampling in Definition 2.3 difers from i.i.d. sampling directly from $\rho _ { P }$ The former uses fixed group sizes while the latter would randomize the group counts. Both sampling distributions remain fixed within each construction. Every realized construction belongs to the �-trainable network class $\mathcal { F } _ { P } ( W , 2 )$ , so its approximation error provides an upper bound on the best approximation error over this network class. Besides, the fixed initialization distribution $\rho _ { 0 }$ is prescribed, while the choice of the adjustable distribution $\rho _ { 1 }$ is discussed in the next section.

By (2.4), optimizing only the output coeficients and taking expectations over both groups thus yields

$$
\begin{array} { r l } {  { \mathcal { E } _ { m } ( f ^ { * } , \mathcal { F } _ { P } ( W , 2 ) ) } } \\ & { \leq \mathbb { E } _ { \theta _ { 1 : m _ { \mathrm { f r e e } } } , \bar { \theta } _ { 1 : m _ { \mathrm { f i x } } } } [ \operatorname* { i n f } _ { \beta } \Big \| f ^ { * } - \sum _ { j = 1 } ^ { m _ { \mathrm { f r e e } } } \beta _ { j } \phi ( \cdot ; \theta _ { j } ) - \sum _ { j = 1 } ^ { m _ { \mathrm { f r e e } } } \beta _ { m _ { \mathrm { f r e e } } + j } \phi ( \cdot ; \bar { \theta } _ { j } ) \Big \| _ { L _ { 2 } ( d \mu ) } ^ { 2 } ] . } \end{array}\tag{2.11}
$$

Ridge Regression. Conditional on the sampled hidden features, the network output is linear in $\beta .$ . We analyze (2.11) by choosing the output coeficients $\beta$ by population ridge regression. The objective is specified in the next section.

Remark 2.4 (Analytical Construction). The distributional construction is used only for the approximation analysis and does not require the practical training algorithm to optimize explicitly over probability distributions.

## 3. General Approximation Results

This section establishes a general approximation bound for two-layer �-trainable networks with targets in $\mathcal { H } _ { \rho }$ . Under polynomial spectral decay and leverage compatibility assumptions, this bound yields an explicit rate and quantifies the phase transition as $P$ varies. Note that $T _ { \rho } = 0$ is trivial for approximation results, since $\mathcal { H } _ { \rho } = \{ 0 \}$ . Henceforth, assume $T _ { \rho } \neq 0$

## 3.1. Approximation Bounds for General Feature Maps

Following the importance reweighting construction in [4], define $\phi _ { P } ( x ; \theta ) = \gamma _ { P } ^ { - 1 / 2 } ( \theta ) \phi ( x ; \theta )$ when $\| \phi ( \cdot ; \theta ) \| _ { L _ { 2 } ( d \mu ) } > 0$ , where $\gamma _ { P }$ is defined in (2.10). For parameters inducing the zero function in $L _ { 2 } ( d \mu )$ , set $\phi _ { P } ( \cdot ; \theta ) = 0$ . The following lemma shows that this reweighting preserves the kernel and its integral operator.

Lemma 3.1 (Invariance of Kernel and Operator). UnderAssumption 2.2, let the two groups be sampled according to $( 2 . 9 )$ , with $P = m _ { \mathrm { f r e e } } / m$ . The reweightedfeatures $\phi _ { P }$ satisfy the following properties.

(1) Thefollowing empirical kernel is unbiasedfor $( \mu \otimes \mu )$ -almost every $( x , x ^ { \prime } ) \in X \times X$

$$
\hat { k } _ { \rho } ^ { ( P ) } ( x , x ^ { \prime } ) = \frac { 1 } { m } \bigg [ \sum _ { j = 1 } ^ { m _ { \mathrm { f r e e } } } \phi _ { P } ( x ; \theta _ { j } ) \phi _ { P } ( x ^ { \prime } ; \theta _ { j } ) + \sum _ { j = 1 } ^ { m _ { \mathrm { f r e } } } \phi _ { P } ( x ; \bar { \theta } _ { j } ) \phi _ { P } ( x ^ { \prime } ; \bar { \theta } _ { j } ) \bigg ] ,
$$

with $\begin{array} { r } { \mathbb { E } \bigl [ \hat { k } _ { \rho } ^ { ( P ) } ( x , x ^ { \prime } ) \bigr ] = k _ { \rho } ( x , x ^ { \prime } ) = \int _ { \Theta } \phi _ { P } ( x ; \theta ) \phi _ { P } ( x ^ { \prime } ; \theta ) d \rho _ { P } ( \theta ) . } \end{array}$

(2) The integral operator induced by $\phi _ { P }$ is still $T _ { \rho }$ and the original eigenpairs $\{ ( \eta _ { k } , e _ { k } ) \} _ { k \ge 1 }$ remain unchanged, satisfying $\begin{array} { r } { T _ { \rho } = \mathbb { E } _ { \theta \sim \rho _ { P } } \left[ \phi _ { P } ( \cdot ; \theta ) \otimes \phi _ { P } ( \cdot ; \theta ) \right] = \sum _ { k \geq 1 } \eta _ { k } ( e _ { k } \otimes e _ { k } ) } \end{array}$

Lemma 3.1 ensures that the reweighted features preserve $T _ { \rho }$ . Using these sampled features, the output coeficients are chosen by population ridge regression. For any $\lambda > 0$

$$
\hat { \beta } _ { \lambda } = \underset { \beta \in \mathbb { R } ^ { m } } { \arg \operatorname* { m i n } } \left\{ \left\| f ^ { * } - \sum _ { j = 1 } ^ { m _ { \mathrm { f r e e } } } \beta _ { j } \phi _ { P } ( \cdot ; \theta _ { j } ) - \sum _ { j = 1 } ^ { m _ { \mathrm { f r } } } \beta _ { m _ { \mathrm { f r e e } } + j } \phi _ { P } ( \cdot ; \bar { \theta } _ { j } ) \right\| _ { L _ { 2 } ( d \mu ) } ^ { 2 } + \lambda m \left\| \beta \right\| _ { 2 } ^ { 2 } \right\} .\tag{3.1}
$$

For each realization, reweighting rescales the output coeficients and preserves the span of the sampled features. To analyze the approximation error of this ridge construction, the size of each reweighted feature is measured relative to the regularized operator $T _ { \rho } + \lambda I$ Specifically, define the ridge leverage score

$$
s _ { \lambda , P } ( \theta ) = \langle \phi _ { P } ( \cdot ; \theta ) , ( T _ { \rho } + \lambda I ) ^ { - 1 } \phi _ { P } ( \cdot ; \theta ) \rangle _ { L _ { 2 } ( d \mu ) } ,\tag{3.2}
$$

where $\phi _ { P } = \gamma _ { P } ^ { - 1 / 2 } \phi .$ , with the convention that $s _ { \lambda , P } ( \theta ) = 0$ whenever $\| \phi ( \cdot ; \theta ) \| _ { L _ { 2 } ( d \mu ) } = 0$ The maximal ridge leverage value under the sampling distribution $\rho _ { P }$ is

$$
d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) = \operatorname { e s s } _ { \theta \sim \rho _ { P } } s _ { \lambda , P } ( \theta ) .\tag{3.3}
$$

The following theorem uses the regularization scale $\lambda = \varepsilon / 4$ to give a suficient width condition for squared approximation error at most � with confidence level $1 - \delta$

Theorem 3.2 (General Approximation Bounds). Under Assumption 2.2, let the error tolerance $\varepsilon > 0 _ { ; }$ $m _ { \mathrm { f r e e } } , m _ { \mathrm { f i x } } \in \mathbb { N }$ with $m = m _ { \mathrm { f r e e } } + m _ { \mathrm { f i x } } \geq 1$ , and the trainable proportion $P = m _ { \mathrm { f r e e } } / m \in [ 0 , 1 ]$ . Assume that $d _ { \operatorname* { m a x } } ( \gamma _ { P } , \varepsilon / 4 ) < \infty$ . Let the hidden parameters be sampled according to Definition 2.3. For any $\delta \in ( 0 , 1 )$ , if

$$
m \ge \frac { 4 0 } { 9 } d _ { \mathrm { m a x } } ( \gamma _ { P } , \varepsilon / 4 ) \log \frac { 1 4 d _ { \mathrm { m a x } } ( \gamma _ { P } , \varepsilon / 4 ) } { \delta } ,\tag{3.4}
$$

then, with probability at least $1 - \delta _ { \mathrm { { ; } } }$

$$
\operatorname* { s u p } _ { \| f ^ { * } \| _ { \mathcal H _ { \rho } } \leq 1 } \operatorname* { i n f } _ { \| \beta \| _ { 2 } ^ { 2 } \leq \frac { 4 } { m } } \bigg \| f ^ { * } - \sum _ { j = 1 } ^ { m _ { \mathrm { f r e e } } } \beta _ { j } \phi _ { P } ( \cdot ; \theta _ { j } ) - \sum _ { j = 1 } ^ { m _ { \mathrm { f r e e } } } \beta _ { m _ { \mathrm { f r e e } } + j } \phi _ { P } ( \cdot ; \bar { \theta } _ { j } ) \bigg \| _ { L _ { 2 } ( d \mu ) } ^ { 2 } \leq \varepsilon .\tag{3.5}
$$

Theorem 3.2 reveals that a single realization of the sampled hidden parameters supports the general approximation bounds uniformly over the unit ball of $\mathcal { H } _ { \rho }$ , with output coeficients chosen separately for each target function. The constraint $\| \beta \| _ { 2 } ^ { 2 } \leq 4 / m$ controls the output coeficients in the reweighted representation in (3.1).

The width condition (3.4) depends on the sampling scheme through $d _ { \operatorname* { m a x } } ( \gamma _ { P } , \varepsilon / 4 )$ . To further understand what governs the approximation capabilities of �-trainable networks, the next subsection examines how $d _ { \operatorname* { m a x } } ( \gamma _ { P } , \varepsilon / 4 )$ depends on the trainable proportion �, the sampling distributions $\rho _ { 0 }$ and $\rho _ { 1 }$ , and the regularization scale �.

## 3.2. Approximation Rates under Polynomial Spectral Decay

The quantity $d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda )$ can be written as the product of a spectral complexity of $T _ { \rho }$ and a sampling mismatch factor. Specifically, the spectral complexity is measured by the efective dimension, defined for $\lambda > 0$ as

$$
d ( \lambda ) = \int _ { \Theta } s _ { \lambda , P } ( \theta ) d \rho _ { P } ( \theta ) = \operatorname { t r } ( T _ { \rho } ( T _ { \rho } + \lambda I ) ^ { - 1 } ) .\tag{3.6}
$$

Since reweighting preserves $T _ { \rho }$ , the efective dimension depends only on $T _ { \rho }$ and $\lambda ,$ independent of the trainable proportion � and the sampling distributions $\rho _ { 0 }$ and $\rho _ { 1 }$ .

For $d ( \lambda ) > 0$ , define the mismatch factor of the sampling distribution $\rho _ { P }$ by

$$
\kappa _ { \gamma _ { P } } ( \lambda ) = \frac { d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) } { d ( \lambda ) } = \frac { \mathrm { e s s } \operatorname* { s u p } _ { \theta \sim \rho _ { P } } s _ { \lambda , P } ( \theta ) } { \mathbb { E } _ { \theta \sim \rho _ { P } } \left[ s _ { \lambda , P } ( \theta ) \right] } \geq 1 .\tag{3.7}
$$

![](images/ac4da8931bbaeb0f54789bc04d8cab5fb17f310d067c8550578206186647dfb5.jpg)  
(a) Well-balanced sampling with small $\kappa _ { \gamma _ { P } } ( \lambda )$

![](images/70d0bbc5c97cf5db0800f8b707215de81213398f60a4d74108827e108145a42e.jpg)  
(b) Local undersampling with large $\kappa _ { \gamma _ { P } } ( \lambda )$  
Figure 4. Schematic illustration of the mismatch factor $\kappa _ { \gamma _ { P } } ( \lambda )$ . The sampling density closely follows the ridge leverage density in (a), but assigns insuficient mass near the right peak in (b).

The mismatch factor $\kappa _ { \gamma _ { P } } ( \lambda )$ compares the maximal score with its sampling average. Since the suficient width condition (3.4) in Theorem 3.2 is controlled by $d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda )$ , a natural sampling benchmark is a density minimizing this maximal ridge leverage for fixed $T _ { \rho }$ and �. Such a benchmark is provided by the optimal ridge leverage density with respect to $\rho _ { ; }$

$$
\gamma _ { \lambda } ^ { \ast } ( \theta ) = \frac { \gamma _ { P } ( \theta ) s _ { \lambda , P } ( \theta ) } { d ( \lambda ) } = \frac { \langle \phi ( \cdot ; \theta ) , ( T _ { \rho } + \lambda I ) ^ { - 1 } \phi ( \cdot ; \theta ) \rangle _ { L _ { 2 } ( d \mu ) } } { d ( \lambda ) } .\tag{3.8}
$$

By construction, we have $\begin{array} { r } { \int _ { \Theta } \gamma _ { \lambda } ^ { \ast } ( \theta ) d \rho ( \theta ) = 1 , d _ { \mathrm { m a x } } ( \gamma _ { \lambda } ^ { \ast } , \lambda ) = d ( \lambda ) } \end{array}$ , and $\kappa _ { \gamma _ { \lambda } ^ { * } } ( \lambda ) = 1$ . Since $d _ { \operatorname* { m a x } } ( \gamma , \lambda ) \geq d ( \lambda )$ for any admissible density $\gamma ,$ the density $\gamma _ { \lambda } ^ { \ast }$ gives the smallest suficient width prescribed by (3.4) for uniform approximation over the unit ball of the target RKHS $\mathcal { H } _ { \rho }$ . It therefore provides an optimal sampling benchmark adapted to the entire target function class. Moreover, the mismatch factor admits the density representation

$$
\kappa _ { \gamma _ { P } } ( \lambda ) = \cos \operatorname* { s u p } _ { \theta \sim \rho _ { P } } \frac { \gamma _ { \lambda } ^ { * } ( \theta ) } { \gamma _ { P } ( \theta ) } .
$$

Thus, $\kappa _ { \gamma _ { P } } ( \lambda )$ measures the sampling mismatch through the largest local ratio. Values close to 1 indicate that the current sampling density closely matches the ridge leverage benchmark. Large values indicate that some parameter regions receive insuficient sampling mass. Figure 4 illustrates these two cases.

Since the dependence on the trainable proportion � enters through the mixture density $\gamma _ { P } = P \gamma _ { 1 } + ( 1 - P ) \gamma _ { 0 }$ , the following lemma bounds $d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda )$ in terms of � and the mismatch factors of the two component sampling densities.

Lemma 3.3 (Mixture Leverage Bound). Let the mixture density be $\gamma _ { P } = P \gamma _ { 1 } + ( 1 - P ) \gamma _ { 0 }$ with $P \in [ 0 , 1 ]$ . For any $\lambda > 0$ and $\kappa _ { \gamma _ { i } } ( \lambda ) < \infty , i \in \{ 0 , 1 \}$ , we have

$$
d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) \leq \Big ( \frac { P } { \kappa _ { \gamma _ { 1 } } ( \lambda ) } + \frac { 1 - P } { \kappa _ { \gamma _ { 0 } } ( \lambda ) } \Big ) ^ { - 1 } d ( \lambda ) .\tag{3.9}
$$

To derive explicit approximation rates, the next assumption further controls these two factors over the relevant regularization scales.

Assumption 3.4 (Leverage Compatibility). There exist $\lambda _ { 0 } > 0$ and constants $1 \le { \kappa } _ { 1 } ^ { * } \le { \kappa } _ { 0 } ^ { * } <$ such that,for each $0 < \lambda \leq \lambda _ { 0 }$ , the sampling densities satisfy $\kappa _ { \gamma _ { i } } ( \lambda ) \leq \kappa _ { i } ^ { * } f o r i \in \{ 0 , 1 \}$ Here, �0 isprescribed, while $\gamma _ { 1 }$ may depend on �and isfixed in each sampling construction.

Together with the mixture leverage bound in Lemma 3.3, this assumption controls the efect of mixture sampling through the constants $\boldsymbol { \kappa } _ { 1 } ^ { * }$ and $\boldsymbol { \kappa } _ { 0 } ^ { * } .$ . The remaining step is to determine how the efective dimension $d ( \lambda )$ grows as � decreases.

Assumption 3.5 (Polynomial Spectral Decay). There exist constants $c _ { - } , c _ { + } > 0$ and $\alpha > 1$ such that $c _ { - } k ^ { - \alpha } \leq \eta _ { k } \leq c _ { + } k ^ { - \alpha }$ for every $k \geq 1$

This assumption specifies the eigenvalue decay of the integral operator $T _ { \rho }$ and implies $d ( \lambda ) \asymp \lambda ^ { - 1 / \alpha }$ for $0 < \lambda \leq \lambda _ { 0 }$ , as shown in the proof of Theorem 3.6. Together with the mixture leverage bound in Lemma 3.3, Assumptions 3.4 and 3.5 are used to derive explicit approximation rates. Further interpretation of assumptions is discussed in Appendix A.2.

Theorem 3.6 (Polynomial Approximation Rates). Under Assumptions 2.2, 3.4 and 3.5, let $m \in \mathbb { N } _ { + }$ and $P \in [ 0 , 1 ]$ satisfy $P m \in \mathbb { N } ,$ , and set $W = \operatorname* { m a x } \{ d , m , 1 \}$ . For any $\delta \in ( 0 , 1 )$ under the sampling scheme in Definition 2.3, the rate holds with probability at least $1 - \delta _ { : }$

$$
\operatorname* { s u p } _ { \| f \| _ { \mathcal H _ { \rho } } \leq 1 } \operatorname* { i n f } _ { f _ { 2 } ^ { ( P ) } \in \mathcal F _ { P } ( W , 2 ) } \| f - f _ { 2 } ^ { ( P ) } \| _ { L _ { 2 } ( d \mu ) } ^ { 2 } \lesssim \Big ( \Big ( \frac { P } { { \kappa _ { 1 } ^ { * } } } + \frac { 1 - P } { { \kappa _ { 0 } ^ { * } } } \Big ) \frac { 9 m } { 4 0 \log ( 6 3 m / ( 2 0 \delta ) ) } \Big ) ^ { - \alpha } ,\tag{3.10}
$$

where the implicit constant is independent of �, � and �.

The stated rate (3.10) does not require the regularization scale to satisfy $\lambda \leq \lambda _ { 0 }$ . When $\lambda > \lambda _ { 0 }$ , the zero approximant used in the proof gives a squared error bounded by a constant times �. The same rate therefore follows after adjusting the implicit constant, without an additional restriction on $m$ . Accordingly, the adjustable density $\gamma _ { 1 }$ for the trainable group is selected at the scale used in the proof of Theorem 3.6.

By scaling, the result extends to every target $f \in \mathcal { H } _ { \rho }$

$$
\operatorname* { i n f } _ { f _ { 2 } ^ { ( P ) } \in \mathcal { F } _ { P } ( W , 2 ) } \lVert f - f _ { 2 } ^ { ( P ) } \rVert _ { L _ { 2 } ( d \mu ) } ^ { 2 } \lesssim \Big ( \Big ( \frac { P } { \kappa _ { 1 } ^ { * } } + \frac { 1 - P } { \kappa _ { 0 } ^ { * } } \Big ) \frac { 9 m } { 4 0 \log ( 6 3 m / ( 2 0 \delta ) ) } \Big ) ^ { - \alpha } \lVert f \rVert _ { \mathcal { H } _ { \rho } } ^ { 2 } .
$$

The reference measure $\rho$ therefore afects the bound through its kernel spectrum, the associated mismatch constants, and the RKHS norm of the target function.

For a fixed reference measure $\rho _ { ; }$ , Theorem 3.6 provides an explicit approximation rate in terms of the trainable proportion $P ,$ the hidden-layer width $m$ , and the spectral decay exponent $\alpha .$ For fixed $P , \delta$ and mismatch constants, the bound decreases at the rate of $O ( ( m / \log m ) ^ { - \alpha } )$ as � . For fixed � and $\delta ,$ the dependence on $P$ afects the bound through the prefactor $( P / \kappa _ { 1 } ^ { * } + ( 1 - P ) / \kappa _ { 0 } ^ { * } ) ^ { - \alpha }$ , which is non-increasing in $P$ since $\kappa _ { 1 } ^ { * } \leq \kappa _ { 0 } ^ { * } .$ Choosing $\gamma _ { 1 } = \gamma _ { \lambda } ^ { * }$ at each regularization choice gives $\kappa _ { 1 } ^ { * } = 1$ . More generally, when $\kappa _ { 1 } ^ { * } \ll$ $\boldsymbol { \kappa } _ { 0 } ^ { * } ,$ a substantial reduction in the bound can occur even for small $P ,$ as quantified in the subsequent phase transition Corollary 3.10.

Remark 3.7 (Probability Statement When $P = 1 )$ . At $P = 1$ , the fully trainable network class is deterministic, but the proof still constructs the trainable parameters by sampling from $\rho _ { 1 }$ . The failure probability $\delta$ refers to this sampled construction. A successful realization yields an approximation bound over the deterministic network class. The highprobability formulation is retained for consistency across admissible $P \in [ 0 , 1 ]$

Substituting $P = m _ { \mathrm { f r e e } } / m$ into Theorem 3.6 gives an equivalent approximation bound in terms of the numbers of trainable and fixed neurons.

Corollary 3.8 (Rate by Neuron Counts). Under the same assumptions andsampling scheme, let $m _ { \mathrm { f r e e } } , m _ { \mathrm { f i x } } \in \mathbb { N }$ with $m = m _ { \mathrm { f r e e } } + m _ { \mathrm { f i x } } \geq 1$ , and set $P = m _ { \mathrm { f r e e } } / m$ . For $\delta \in ( 0 , 1 )$ , with probability at least $1 - \delta$

$$
\operatorname* { s u p } _ { \| f \| _ { \mathcal H _ { \rho } } \leq 1 } \operatorname* { i n f } _ { f _ { 2 } ^ { ( P ) } \in \mathcal F _ { P } ( W , 2 ) } \| f - f _ { 2 } ^ { ( P ) } \| _ { L _ { 2 } ( d \mu ) } ^ { 2 } \lesssim \Big ( \Big ( \frac { m _ { \mathrm { f r e e } } } { \kappa _ { 1 } ^ { * } } + \frac { m _ { \mathrm { f i x } } } { \kappa _ { 0 } ^ { * } } \Big ) \frac { 9 } { 4 0 \log ( 6 3 m / ( 2 0 \delta ) ) } \Big ) ^ { - \alpha } .
$$

The two neuron groups enter the bound through the weighted counts $m _ { \mathrm { f r e e } } / \kappa _ { 1 } ^ { * }$ and $m _ { \mathrm { f i x } } / \kappa _ { 0 } ^ { * }$ . Their contributions are comparable when $m _ { \mathrm { f r e e } } \asymp m _ { \mathrm { f i x } } ( \kappa _ { 1 } ^ { * } / \kappa _ { 0 } ^ { * } )$ . Thus, when $\begin{array} { r } { \kappa _ { 1 } ^ { * } \ll \kappa _ { 0 } ^ { * } , } \end{array}$ a small trainable group can substantially reduce the bound. The reciprocal mismatch ratio $\kappa _ { 1 } ^ { * } / \kappa _ { 0 } ^ { * }$ determines the scale of the relative trainable count, while the hidden-layer width � still afects the overall approximation accuracy guaranteed by the bound.

Remark 3.9 (Influence of Input Dimension). The input dimension � afects the approximation bounds through the spectral decay exponent � and the mismatch constants $\boldsymbol { \kappa } _ { 0 } ^ { * }$ and $\boldsymbol { \kappa } _ { 1 } ^ { * }$ . Section 4 makes this dependence explicit for the spherical biased-ReLU features.

## 3.3. Small-� Phase Transition

To quantify the efect of � on the approximation rate of two-layer �-trainable networks, fix the hidden-layer width � and the confidence level $1 - \delta$ , and let $\mathcal { E } ( P )$ denote the rate expression on the right-hand side of (3.10). For $\kappa _ { 1 } ^ { * } < \kappa _ { 0 } ^ { * }$ , define the normalized gain by

$$
G ( P ) = \frac { \mathcal { E } ( 0 ) - \mathcal { E } ( P ) } { \mathcal { E } ( 0 ) - \mathcal { E } ( 1 ) } .\tag{3.11}
$$

This quantity $G ( P )$ measures the fraction of the total bound reduction between $P = 0$ and $P = 1$ attained at trainable proportion �.

Corollary 3.10 (Phase Transition Improvement). Under the conditions ofTheorem 3.6, let $r = \kappa _ { 0 } ^ { * } / \kappa _ { 1 } ^ { * } > 1$ . Then the normalized gain satisfies

$$
G ( P ) = \frac { 1 - \left( 1 + ( r - 1 ) P \right) ^ { - \alpha } } { 1 - r ^ { - \alpha } } , f o r P \in [ 0 , 1 ] .\tag{3.12}
$$

![](images/6cd8c13e4d0f909d93acb409a02eeaa6fc265a4917ede485443b2c3264529a0e.jpg)  
Figure 5. Minimum trainable proportion for 90% normalized gain, with $\alpha = 2$ and $\varepsilon = 0 . 1$ . Shaded region above $P _ { \varepsilon } ( r )$ satisfies $G ( P ) \geq 1 - \varepsilon$ . The endpoint at $r = 1$ represents the limit $1 - \varepsilon \operatorname { a s } r \to 1 _ { + }$

Moreover, $G ( 0 ) = 0 , G ( 1 ) = 1 , G ^ { \prime } ( P ) > 0 ,$ , and $G ^ { \prime \prime } ( P ) < 0 .$ . Therefore, increasing the trainable proportion � monotonically reduces the approximation bound, with diminishing marginal improvement.

The normalized gain depends only on the mismatch ratio � and the spectral exponent �. This dependence also quantifies how a small proportion of trainable neurons can capture most of the total improvement. For any tolerance level $ { \varepsilon } \in ( 0 , 1 )$ , define

$$
P _ { \varepsilon } ( r ) = \frac { ( \varepsilon + ( 1 - \varepsilon ) r ^ { - \alpha } ) ^ { - 1 / \alpha } - 1 } { r - 1 } ,
$$

which is the smallest trainable proportion required to achieve the normalized gain of $1 - \varepsilon$

$$
G ( P ) \geq G ( P _ { \varepsilon } ) = 1 - \varepsilon \iff P \geq P _ { \varepsilon } .
$$

For a fixed width �, the corresponding minimum trainable count is $\lceil m P _ { \varepsilon } ( r ) \rceil$ . Furthermore, $P _ { \varepsilon } \le \bigl ( \varepsilon ^ { - 1 / \alpha } - 1 \bigr ) / ( r - 1 )$ . For fixed � and $\alpha ,$ the threshold of $P _ { \varepsilon }$ therefore decreases at the rate $O ( r ^ { - 1 } )$ as � increases. This quantifies the phase transition phenomenon in the bound: when the trainable distribution has a substantially smaller mismatch constant than the fixed initialization distribution $( \boldsymbol { \kappa } _ { 1 } ^ { * } \ll \boldsymbol { \kappa } _ { 0 } ^ { * } )$ , a small trainable proportion can capture most of the total reduction available between $P = 0$ and $P = 1$ . Figure 5 illustrates this threshold for $\alpha = 2$ and $\varepsilon = 0 . 1$ . As � increases, a smaller trainable proportion sufices to attain $9 0 \%$ of the normalized gain. At $r = 1 0 0 , P _ { \varepsilon } \approx 0 . 0 2 1 8$ , giving a proportion threshold of approximately 2.2%.

Remark 3.11 (Phase Transition in the Upper Bound). The normalized gain describes the reduction in the theoretical upper bound. Observed errors also depend on the target function, input sampling, and optimization, and may therefore deviate from this curve. Nevertheless, the subsequent numerical experiments exhibit a similar qualitative phase transition.

## 4. Analysis of Spherical Biased-ReLU Features

Since the preceding results in Section 3 apply to general feature maps under assumptions on sampling compatibility and spectral decay, this section verifies these assumptions for spherical biased-ReLU features.

## 4.1. The Target RKHS and Kernel Spectrum

Target Function Class. Let $d \ge 2$ and consider the spherical input and parameter spaces

$$
\begin{array} { r l r } & { X = \mathbb { S } ^ { d - 1 } = \{ x \in \mathbb { R } ^ { d } , \| x \| _ { 2 } = 1 \} \subset \mathbb { R } ^ { d } , \ \mu = \mathrm { U n i f } ( \mathbb { S } ^ { d - 1 } ) , } & \\ & { \Theta = \mathbb { S } ^ { d } \subset \mathbb { R } ^ { d + 1 } , \ } & { \rho = \mathrm { U n i f } ( \mathbb { S } ^ { d } ) , } \end{array}
$$

where Unif denotes the normalized rotation-invariant surface measure. Throughout this section, $\rho$ serves as the reference measure defining the kernel $k _ { \rho } ,$ , the integral operator $T _ { \rho } ,$ and the associated target RKHS $\mathcal { H } _ { \rho }$

For $\theta = ( \omega , b ) \in \Theta$ , define the biased ReLU feature by

$$
\phi ( x ; \theta ) = \sigma _ { R } ( \omega ^ { \top } x + b ) = ( \theta ^ { \top } \widetilde x ) _ { + } , \mathrm { ~ w h e r e ~ } \sigma _ { R } ( t ) = \operatorname* { m a x } \{ t , 0 \} , \mathrm { ~ a n d ~ } \widetilde x = ( x , 1 ) .
$$

Since $\| \theta \| _ { 2 } ^ { 2 } = 1$ and $\| { \tilde { x } } \| _ { 2 } = { \sqrt { 2 } } .$ , we have $0 \leq \phi ( x ; \theta ) \leq \sqrt { 2 }$ . These bounded features satisfy the integrability Assumption 2.2. Thus, the associated operator $T _ { \rho }$ taking the form of (2.6) is self-adjoint, positive semi-definite, trace-class, and compact on $L _ { 2 } ( \mathbb { S } ^ { d - 1 } , d \mu )$

Scaled Spherical Initialization. To examine the initialization mismatch while keeping the target RKHS fixed, we consider the following family of sampling distributions for hidden neurons. Let $\omega _ { 0 } \sim N ( 0 , I _ { d } )$ and $b _ { 0 } \sim N ( 0 , 1 )$ be independent. For $\tau \geq 1$ , define

$$
\theta = \frac { ( \omega _ { 0 } , \tau b _ { 0 } ) } { \sqrt { \| \omega _ { 0 } \| _ { 2 } ^ { 2 } + \tau ^ { 2 } b _ { 0 } ^ { 2 } } } = ( \omega , b ) \in \Theta = \mathbb { S } ^ { d } ,
$$

and denote its distribution by $\rho _ { 0 } ^ { ( \tau ) }$ . Its density with respect to $\rho$ is

$$
\gamma _ { 0 } ^ { ( \tau ) } ( \theta ) = \frac { d \rho _ { 0 } ^ { ( \tau ) } } { d \rho } ( \theta ) = \frac { 1 } { \tau } \Big ( 1 - b ^ { 2 } + \frac { b ^ { 2 } } { \tau ^ { 2 } } \Big ) ^ { - ( d + 1 ) / 2 } .
$$

In particular, $\rho _ { 0 } ^ { ( 1 ) } = \rho$ . The density satisfies $\gamma _ { 0 } ^ { ( \tau ) } ( \theta ) \geq 1 / \tau$ for all $\theta \in \Theta$ , since $\tau \geq 1$ and $| b | \leq$ 1. The randomly fixed neurons are sampled from $\rho _ { 0 } ^ { ( \tau ) } = \gamma _ { 0 } ^ { ( \tau ) } \rho$ . For each regularization scale $\lambda > 0$ , the trainable parameters are constructed using the optimal ridge leverage distribution introduced in (3.8), that is, $\rho _ { 1 } = \gamma _ { 1 } \rho$ with $\gamma _ { 1 } = \gamma _ { \lambda } ^ { * }$ and $\kappa _ { 1 } ^ { * } = 1$ . The corresponding mixture density is therefore

$$
\gamma _ { P } ( \boldsymbol { \theta } ) = P \gamma _ { 1 } ( \boldsymbol { \theta } ) + ( 1 - P ) \gamma _ { 0 } ^ { ( \tau ) } ( \boldsymbol { \theta } ) .
$$

With the reweighted features defined in Subsection 3.1, Lemma 3.1 ensures that this sampling construction preserves $k _ { \rho }$ and $T _ { \rho }$ . Varying � only changes the initialization distribution while leaving the target space $\mathcal { H } _ { \rho }$ unchanged.

![](images/9e373de5bb97982febd67014cf57f044482947663262183e106751d28eca2f25.jpg)  
Figure 6. Scaled spherical initialization. Increasing � shifts fixed-neuron parameters toward the poles $b = \pm 1$ . The right panel schematically shows how greater mismatch can sharpen the gain at small �.

Figure 6 illustrates the role of $\tau . \mathrm { A t } \tau = 1$ , fixed neurons follow the uniform reference distribution $\rho _ { \cdot }$ . Increasing � concentrates their initialization near $b = \pm 1$ . The right panel schematically illustrates how greater initialization mismatch with the target RKHS can sharpen the normalized gain as � increases. Thus the scaled spherical family provides a concrete setting for analyzing the efect of initialization mismatch, and the framework also accommodates other admissible initialization distributions.

Eigen-decomposition of $T _ { \rho }$ . The spectrum of $T _ { \rho }$ is characterized using spherical harmonics. Following [1, 3], let $\mathbb { Y } _ { k }$ be the space of spherical harmonics of degree � on $\mathbb { S } ^ { d - 1 }$ , and let $\{ Y _ { k , j } \} _ { j = 1 } ^ { N ( d , k ) }$ be a real orthonormal basis of $\mathbb { Y } _ { k }$ in $L _ { 2 } ( \mathbb { S } ^ { d - 1 } , d \mu )$ , where

$$
N ( d , 0 ) = 1 , N ( d , k ) = \frac { 2 k + d - 2 } { k } \binom { k + d - 3 } { d - 2 } , d \geq 2 , k \geq 1 .\tag{4.1}
$$

Every $f \in L _ { 2 } ( \mathbb { S } ^ { d - 1 } , d \mu )$ admits the following expansion, with convergence in $L _ { 2 } ( d \mu )$

$$
f ( x ) = \sum _ { k = 0 } ^ { + \infty } \sum _ { j = 1 } ^ { N ( d , k ) } \left. f , Y _ { k , j } \right. _ { L _ { 2 } ( d \mu ) } Y _ { k , j } ( x ) = \sum _ { k = 0 } ^ { + \infty } N ( d , k ) \int _ { X } f ( y ) P _ { k } ( x ^ { \top } y ) d \mu ( y ) .
$$

Here $P _ { k }$ is the normalized Legendre polynomial ofdegree � in dimension �, with $P _ { k } ( 1 ) = 1$ By the rotation invariance of $\rho$ and results in [3, Section 3.1], the induced kernel $k _ { \rho } ( x , x ^ { \prime } )$ now has the explicit form

$$
k _ { \rho } ( \boldsymbol { x } , \boldsymbol { x } ^ { \prime } ) = \mathbb { E } _ { \theta \sim \rho } [ \phi ( \boldsymbol { x } ; \theta ) \phi ( \boldsymbol { x } ^ { \prime } ; \theta ) ] = h \Big ( \frac { 1 + \boldsymbol { x } ^ { \top } \boldsymbol { x } ^ { \prime } } { 2 } \Big ) ,
$$

where $\begin{array} { r } { h ( t ) = \frac { 1 } { \pi ( d + 1 ) } ( \sqrt { 1 - t ^ { 2 } } + ( \pi - \operatorname { a r c c o s } t ) t ) } \end{array}$ . Thus, $k _ { \rho }$ is a continuous rotation-invariant kernel on $\mathbb { S } ^ { d - 1 }$ , allowing its spectrum to be analyzed through the spherical harmonic decomposition.

Proposition 4.1 (Spectrum and Eigenvalue Decay of ReLU Kernel). Under the spherical setting described above, thefollowing properties hold:

(1) Every spherical harmonic $Y _ { k , j } \in \mathbb { Y } _ { k }$ is an eigenfunction of $\mathrm { \Delta } \mathrm { \ddot { \it T } } _ { \rho }$ , with $T _ { \rho } Y _ { k , j } = \xi _ { k } Y _ { k , j } , j =$ $1 , \ldots , N ( d , k )$ , where the eigenvalue associated with degree � satisfies

$$
\xi _ { k } = \frac { \Gamma ( d / 2 ) } { \sqrt { \pi } \Gamma ( ( d - 1 ) / 2 ) } \int _ { - 1 } ^ { 1 } h \Big ( \frac { 1 + t } { 2 } \Big ) P _ { k } ( t ) ( 1 - t ^ { 2 } ) ^ { ( d - 3 ) / 2 } d t > 0 .\tag{4.2}
$$

Here Γ is the Gammafunction and $\begin{array} { r } { h ( t ) = \frac { 1 } { \pi ( d + 1 ) } ( \sqrt { 1 - t ^ { 2 } } + ( \pi - \operatorname { a r c c o s } t ) t ) . } \end{array}$

(2) Consequently, the operator and kernel admit the decompositions:

$$
T _ { \rho } = \sum _ { k = 0 } ^ { + \infty } { \xi _ { k } } \sum _ { j = 1 } ^ { N ( d , k ) } { Y _ { k , j } } \otimes { Y _ { k , j } } , a n d k _ { \rho } ( x , x ^ { \prime } ) = \sum _ { k = 0 } ^ { + \infty } { \xi _ { k } } \sum _ { j = 1 } ^ { N ( d , k ) } { Y _ { k , j } } ( x ) Y _ { k , j } ( x ^ { \prime } ) .
$$

(3) The degree-wise eigenvalues satisfy $\xi _ { k } \asymp k ^ { - ( d + 2 ) } a s k \to \infty$ . Moreover, the eigenvalues $\{ \eta _ { k } \} _ { k \ge 1 } \ o f T _ { \rho }$ arranged in non-increasing order and counted with multiplicity satisfy $\eta _ { k } \asymp k ^ { - \frac { d + 2 } { d - 1 } }$

Proposition 4.1 verifies the polynomial spectral decay Assumption 3.5 with $\textstyle \alpha = { \frac { d + 2 } { d - 1 } }$ It also characterizes the considered target RKHS as

$$
\mathcal { H } _ { \rho } = \left\{ f = \sum _ { k = 0 } ^ { \infty } \sum _ { j = 1 } ^ { N ( d , k ) } c _ { k , j } Y _ { k , j } : \| f \| _ { \mathcal { H } _ { \rho } } ^ { 2 } = \sum _ { k = 0 } ^ { \infty } \sum _ { j = 1 } ^ { N ( d , k ) } \frac { c _ { k , j } ^ { 2 } } { \xi _ { k } } < \infty \right\} .\tag{4.3}
$$

## 4.2. Initialization Mismatch and Approximation Rates

With the spectral decay established, the remaining step is to verify Assumption 3.4 by bounding the initialization mismatch $\boldsymbol { \kappa } _ { \gamma _ { 0 } ^ { ( \tau ) } } ( \lambda )$ . This provides the compatibility constant needed to specialize the approximation rates.

Quantification of the Initialization Mismatch. For the scaled spherical initialization $\bar { \rho _ { 0 } ^ { ( \tau ) } } = \gamma _ { 0 } ^ { ( \tau ) } \rho$ , the mismatch factor defined in (3.7) is

$$
\kappa _ { \gamma _ { 0 } ^ { ( \tau ) } } ( \lambda ) = \frac { d _ { \operatorname* { m a x } } ( \gamma _ { 0 } ^ { ( \tau ) } , \lambda ) } { d ( \lambda ) } = \frac { 1 } { d ( \lambda ) } \operatorname { e s s } _ { \theta \sim \rho _ { 0 } ^ { ( \tau ) } } \frac { \langle \phi ( \cdot ; \theta ) , ( T _ { \rho } + \lambda I ) ^ { - 1 } \phi ( \cdot ; \theta ) \rangle _ { L _ { 2 } ( d \mu ) } } { \gamma _ { 0 } ^ { ( \tau ) } ( \theta ) } .
$$

The following proposition bounds this factor explicitly in terms of $d$ and $\tau .$

Proposition 4.2 (Mismatch ofScaled Spherical Initialization). Under the spherical biased-ReLU setting,fix $\tau \geq 1$ and let the randomlyfixed neuronsfollow $\rho _ { 0 } ^ { ( \tau ) } = \dot { \gamma _ { 0 } } ^ { ( \tau ) } \rho .$ . Then,for every $\varepsilon > 0$ , there exists $\lambda _ { \varepsilon , \tau } > 0$ such that

$$
\frac { \sqrt { 2 } \tau ( d + 1 ) } { d } - \varepsilon \le \kappa _ { \gamma _ { 0 } ^ { ( \tau ) } } ( \lambda ) \le \frac { 2 \sqrt { 2 } \tau ( d + 1 ) } { d } + \varepsilon , f o r \lambda \in ( 0 , \lambda _ { \varepsilon , \tau } ] .
$$

To absorb the � term in the upper bound, we choose $\kappa _ { 0 } ^ { * } = 4 \sqrt { 2 } \tau ( d + 1 ) / d$ as an admissible compatibility constant for subsequent analysis, valid for all suficiently small $\lambda > 0$

These estimates show that the initialization mismatch is of linear order in �. Choosing the optimal ridge leverage distribution for the trainable neurons gives $\kappa _ { 1 } ^ { * } = 1$ . Now the leverage compatibility Assumption 3.4 is verified.

Substituting these constants and the spectral decay exponent $\alpha = ( d + 2 ) / ( d - 1 )$ into the general polynomial rate yields the following result.

Theorem 4.3 (Rates of Spherical Biased-ReLU Features). Under the spherical biased-ReLU setting,fix $\tau \geq 1$ . Let $m _ { \mathrm { f r e e } } , m _ { \mathrm { f i x } } \in \mathbb { N }$ with $m = m _ { \mathrm { f r e e } } + m _ { \mathrm { f i x } } \geq 1$ , and set $P = m _ { \mathrm { f r e e } } / m$ and $W = \operatorname* { m a x } \{ d , m , 1 \}$ . For any $\delta \in ( 0 , 1 )$ , there exists $\lambda > 0$ such that, when the two groups are sampled according to Definition 2.3, withfixed-group distribution $\rho _ { 0 } ^ { ( \tau ) }$ and trainablegroup distribution $\rho _ { 1 } = \gamma _ { \lambda } ^ { * } \rho$ , thefollowing bound holds with probability at least $1 - \delta .$

$$
\operatorname* { s u p } _ { \| f \| _ { \mathcal H _ { \rho } } \leq 1 } \operatorname* { i n f } _ { f _ { 2 } ^ { ( P ) } \in \mathcal F _ { P } ( W , 2 ) } \| f - f _ { 2 } ^ { ( P ) } \| _ { L _ { 2 } ( d \mu ) } ^ { 2 } \lesssim \Big ( \Big ( P + \frac { ( 1 - P ) d } { 4 \sqrt 2 \tau ( d + 1 ) } \Big ) \frac { 9 m } { 4 0 \log ( 6 3 m / ( 2 0 \delta ) ) } \Big ) ^ { - \frac { d + 2 } { d - 1 } } .
$$

The corresponding admissible mismatch ratio in Corollary 3.10 is currently $r = \kappa _ { 0 } ^ { * } / \kappa _ { 1 } ^ { * } =$ $4 \sqrt { 2 } \tau ( d + 1 ) / d .$ . Increasing � raises this ratio and produces a sharper predicted transition at small �. The rate thus quantifies how the fixed initialization and trainable proportion � afect the approximation accuracy.

Figure 7 illustrates the predicted normalized gain in a representative high-dimensional setting with $d = 2 0 .$ , for which $\alpha = 2 2 / 1 9$ . In particular, the proportion required to attain 90% of the gain decreases from approximately 0.563 at $\tau = 1$ to 0.100 at $\tau = 1 0$

![](images/762e115d7c458e5b09299b5e6045c4d8e8b2d61ab0d46b1fe0db193e8aa8324a.jpg)

![](images/5ee13de6038aa9c58cb36c00660b5ab2d183dc1ecb7759cd78288f58b4aa62a3.jpg)

![](images/3cc6fe41443d37a2f885776968068e3b5f94cc40e3a0cf5c9dfae7e0fad023da.jpg)

![](images/d9a1358036e8b491241f9d4564889e94f4fb70e4ec3ce8d99aad26c683242b36.jpg)  
Figure 7. Normalized gain predicted by the spherical biased-ReLU bound in dimension $d = 2 0 ( \alpha =$ $2 2 / 1 9 )$ for increasing �. Gray dashed guides mark 90% gain and the endpoint $G ( 1 ) = 1$

Besides, Figure 1 in the introduction provides a numerical illustration for a fixed target in the spherical RKHS unit ball with $d = 5$ , using a two-layer ReLU network of width 64. The observed $9 0 \%$ gain threshold decreases overall as � increases, and the gain curve for $\tau = 1 0$ rises more rapidly at small � than that for $\tau = 1$ . These observations are qualitatively consistent with the dependence on initialization predicted by the bound. The target construction and training settings for this $d = 5$ experiment are provided in Appendix E.1.

Mechanism and Scope. Varying the fixed-neuron initialization while keeping the target RKHS unchanged isolates the efect of sampling mismatch. The spherical biased-ReLU setting precisely demonstrates how a small trainable group can ofset the fixed group’s mismatch and substantially reduce the approximation bound, consistent with the general theory in Section 3. The framework can also be applied to other input domains, feature maps, and target spaces, when the corresponding operator properties, spectral decay, and sampling compatibility are verified. Such extensions may require diferent proof techniques. Nevertheless, the experiments in Section 6 show similar phase transitions across diverse settings.

## 5. A Structured Three-Layer Illustration

Although the �-trainable architecture is well-defined at arbitrary depth, shared upstream parameters couple downstream features through nonlinear composition in deeper networks. This complicates a direct deep-layer extension of the probabilistic kernel argument. The following structured three-layer example provides a complementary illustration ofthe phase transition pattern in approximation bounds and the advantage of nonlinear composition.

Example 5.1 (Structured Three-Layer Illustration). Let $\chi = [ - 1 , 1 ] ^ { 3 } , \mu = \mathrm { U n i f } ( [ - 1 , 1 ] ^ { 3 } )$ and consider the target function

$$
f ^ { * } ( x ) = ( x _ { 2 } ^ { 2 } - 1 / 3 ) + \frac { 1 } { 2 } ( x _ { 3 } ^ { 2 } - 1 / 3 ) .
$$

Let $\phi ( x ; \theta ) = \mathrm { R e L U } ( x ; \theta ) = ( \theta ^ { \top } \tilde { x } ) .$ , where $\tilde { x } = ( x , 1 )$ . Consider the three-layer network

$$
f _ { 3 } ( x ) = \beta _ { 0 } + \sum _ { j = 1 } ^ { 1 0 } \sum _ { \ell = 1 } ^ { 2 0 } \beta _ { j , \ell } \omega _ { j , \ell } \big ( \phi ( x ; \theta _ { 2 j - 1 } ) + \phi ( x ; \theta _ { 2 j } ) - b _ { j , \ell } \big ) _ { + } .
$$

Here $\beta _ { 0 }$ is a freely trainable output bias. The 20 first-layer neurons form 10 disjoint pairs, each feeding a separate block of 20 second-layer neurons. Each second-layer neuron receives inputs only from its assigned pair, with the same weight on both incoming edges.

Parameter Configuration. This structured illustration varies the trainable count only in the first hidden layer, rather than using a common trainable proportion across hidden layers. For $m _ { \mathrm { f r e e } } \in \{ 0 , \ldots , 2 0 \}$ , the first $m _ { \mathrm { f r e e } }$ first-layer parameters are trainable in $\mathbb { R } ^ { 4 }$ . The remaining parameters $\theta _ { i } = ( \theta _ { i , 1 } , \theta _ { i , 2 } , \theta _ { i , 3 } , \theta _ { i , 4 } )$ are sampled independently from $\rho ,$ with

$$
\mathbb { P } \big ( ( \theta _ { i , 1 } , \theta _ { i , 2 } , \theta _ { i , 3 } ) = \mathbf { e } _ { 1 } \big ) = \mathbb { P } \big ( ( \theta _ { i , 1 } , \theta _ { i , 2 } , \theta _ { i , 3 } ) = - \mathbf { e } _ { 1 } \big ) = 0 . 4 9 9 9 ,
$$

$$
\mathbb { P } \big ( ( \theta _ { i , 1 } , \theta _ { i , 2 } , \theta _ { i , 3 } ) = \mathbf { e } _ { k } \big ) = \mathbb { P } \big ( ( \theta _ { i , 1 } , \theta _ { i , 2 } , \theta _ { i , 3 } ) = - \mathbf { e } _ { k } \big ) = 0 . 0 0 0 0 5 , k = 2 , 3 ,
$$

where $\mathbf { e } _ { k }$ is the �-th standard basis vector. Each bias $\theta _ { i , 4 } \sim \mathrm { U n i f } [ - 1 , 1 ]$ is independent of the weight vector. The second-layer parameters are independent draws of $\omega _ { j , \ell } \sim \mathrm { U n i f } [ 1 , 2 ]$ and $b _ { j , \ell } \sim \mathrm { U n i f } [ 0 , 1 ]$ , independently of the first-layer initialization. All sampled parameters remain fixed, while the trainable first-layer parameters and output parameters are optimized.

![](images/3b93bc08d100b59de028e9147108005ea21ffe416dde66dff6d6fbd1099549bb.jpg)  
Figure 8. Architectural comparison with 220 hidden neurons and four trainable neurons. In (a), 20 first-layer neurons form 10 pairs, each feeding a block of 20 fixed neurons with tied incoming weights. In (b), the second-layer budget is reallocated to the single hidden layer. Only representative fixed neurons are shown.

Two-layer Comparison. To additionally examine the approximation efect of nonlinear composition alongside the phase transition, consider the two-layer network

$$
f _ { 2 } ( x ) = \beta _ { 0 } ^ { ( 2 ) } + \sum _ { i = 1 } ^ { 2 2 0 } \beta _ { i } ^ { ( 2 ) } \phi ( x ; \theta _ { i } ^ { ( 2 ) } ) ,
$$

with the same total hidden-neuron and trainable-neuron counts. Its first $m _ { \mathrm { f r e e } }$ hidden parameters are trainable. The remaining hidden parameters are also sampled independently from $\rho$ and fixed. All output parameters are optimized. Figure 8 compares these two architectures with four trainable hidden neurons.

Error Analysis and Normalized Gain. The approximation error is defined as

$$
\mathcal { E } ( f ^ { * } , f _ { d } ) = \mathbb { E } \operatorname* { i n f } _ { \theta _ { \mathrm { f r e e } } , \beta } \| f ^ { * } - f _ { d } \| _ { L _ { 2 } ( d \mu ) } ^ { 2 } , d \in \{ 2 , 3 \} ,
$$

where $\beta$ denotes all output parameters, including bias $\beta _ { 0 }$ , and the expectation is over the fixed random parameters. Let $\mathcal { E } ^ { ( d ) } ( m _ { \mathrm { f r e e } } )$ denote its explicit upper bound derived in Proposition D.1. For each architecture, we normalize the reduction in this upper bound relative to the endpoint $m _ { \mathrm { f r e e } } = 2 0 $

$$
G ^ { ( d ) } ( m _ { \mathrm { f r e e } } ) = \frac { \mathcal { E } ^ { ( d ) } ( 0 ) - \mathcal { E } ^ { ( d ) } ( m _ { \mathrm { f r e e } } ) } { \mathcal { E } ^ { ( d ) } ( 0 ) - \mathcal { E } ^ { ( d ) } ( 2 0 ) } , d \in \{ 2 , 3 \} .
$$

Table 1 reports the upper bounds and normalized gains for two or four trainable neurons.

Phase Transition. Table 1 shows that a small trainable group captures most of the normalized reduction in the approximation bounds. With only two trainable hidden neurons, both constructions attain approximately 80% of the reduction measured relative to their respective bounds at $m _ { \mathrm { f r e e } } = 2 0$ . With four trainable neurons, the normalized gains reach 0.9997 and 0.9877, respectively. These results illustrate a rapid reduction in the bounds at small trainable counts, consistent with the phase transition pattern in the two-layer theory. Moreover, with four trainable neurons, the three-layer error upper bound is only 2.06% of its two-layer counterpart under matched neuron budgets.

Table 1. Approximation error upper bounds and normalized gains with two or four trainable hidden neurons. Both baseline bounds equal $1 / 9 .$ Each gain is normalized relative to its architecture’s bound at $m _ { \mathrm { f r e e } } = 2 0 .$
<table><tr><td>Trainable Neurons</td><td>Architecture</td><td>Error Bound</td><td>Gain</td><td>Bound Ratio</td><td rowspan="3"></td></tr><tr><td rowspan="2">2 in the first layer</td><td>Three-layer  $f _ { 3 }$ </td><td> $2 . 2 2 \times 1 0 ^ { - 2 }$ </td><td>0.7998</td><td> $\frac { \mathcal { E } ^ { ( 3 ) } ( 2 ) } { \mathcal { E } ^ { ( 2 ) } ( 2 ) } \approx 0 . 9 5 3 9$ </td></tr><tr><td>Two-layer  $f _ { 2 }$ </td><td> $2 . 3 3 \times 1 0 ^ { - 2 }$ </td><td>0.7902</td><td></td></tr><tr><td rowspan="2">4 in the first layer</td><td>Three-layer  $f _ { 3 }$ </td><td> $2 . 8 2 \times 1 0 ^ { - 5 }$ </td><td>0.9997</td><td></td><td rowspan="2"> $\frac { \mathcal { E } ^ { ( 3 ) } ( 4 ) } { \mathcal { E } ^ { ( 2 ) } ( 4 ) } \approx 0 . 0 2 0 6$ </td></tr><tr><td>Two-layer  $f _ { 2 }$ </td><td> $1 . 3 7 \times 1 0 ^ { - 3 }$ </td><td>0.9877</td><td></td></tr></table>

Role of Depth. The diference in the absolute upper bounds arises from nonlinear composition and feature reuse across layers. Both constructions approximate the quadratic components of the target function by piecewise linear interpolation, but use their trainable neurons diferently. In the three-layer construction, four trainable first-layer neurons can generate $| x _ { 2 } |$ and $\left| x _ { 3 } \right|$ . Each feeds a fixed block of 20 second-layer neurons, providing 20 basis functions for approximating its square. Thus, four trainable neurons adapt 40 downstream basis functions through the fixed second layer. In contrast, the two-layer construction uses its trainable neurons directly to implement interpolation basis functions, while leaving the fixed features unused. Therefore, this example illustrates how a small trainable upstream group can use fixed downstream random features through nonlinear composition to obtain a smaller explicit approximation bound.

The construction and error estimates of this example follow from a direct interpolation argument, detailed in Appendix D. Beyond this structured example, a general theory characterizing phase transitions and quantifying the approximation advantages of nonlinear composition in multilayer �-trainable networks remains to be developed. It is currently unclear whether the probabilistic kernel approach can be extended to address these questions or whether diferent methods are needed.

## 6. Numerical Evidence of Phase Transitions

The preceding sections derive approximation bounds for two-layer �-trainable networks and analyze a structured three-layer construction. This section complements these results by

![](images/40677f7cd28da8f1291dd4d771233658dd1b7a381529d27b0acc9eb4ea2f4364.jpg)  
(a) Target $f _ { 1 } ^ { * }$

![](images/b985c46b474e2c5546a95470e54a956ef31cff02639e11bbc01f72c94de48494.jpg)  
(b) Target $f _ { 2 } ^ { * }$

![](images/edafaab1be5c226a883edf76b2622dc4ae94634159bba4129192835963e9d661.jpg)  
(c) Target $f _ { 3 } ^ { * }$ .  
Figure 9. Target functions in the numerical experiments. Targets $f _ { 1 } ^ { * }$ and $f _ { 2 } ^ { * }$ are shown as curves, while $f _ { 3 } ^ { * }$ is shown as a heatmap, with color indicating the function value.

examining the attained training losses across diferent target functions, widths, and depths. The numerical results show that the small-� phase transition pattern also appears in the training performance under a fixed optimization budget.

## 6.1. Experimental Setting

Target Functions. Three target functions cover diferent oscillation scales and input dimensions, as illustrated in Figure 9.

The first target combines a relatively low-frequency oscillatory term with a Gaussian term and is sampled at 12,000 uniformly spaced points on 4, 4 :

$$
f _ { 1 } ^ { * } ( x ) = x \sin x + e ^ { - 0 . 1 x ^ { 2 } } .
$$

The second target contains multiple frequency components with higher-frequency oscillations and is sampled at 12,000 uniformly spaced points on 2�, 2� :

$$
f _ { 2 } ^ { * } ( x ) = 0 . 1 \sin ( 2 1 x ) + 0 . 2 \cos ( 1 7 x + 0 . 4 ) + 0 . 3 5 \sin ( 1 3 x - 0 . 7 ) + 0 . 5 \cos ( 9 x ) + \sin ( 5 x ) .
$$

The third target extends the experiments to two dimensions. It combines a separable trigonometric term with a radial Gaussian term and is sampled on a $2 5 0 \times 2 5 0$ uniform grid over $[ - 2 , 2 ] ^ { 2 }$ :

$$
f _ { 3 } ^ { * } ( x , y ) = \sin ( \pi x ) \cos ( \pi y ) + e ^ { - ( x ^ { 2 } + y ^ { 2 } ) } .
$$

All main experiments use deterministic uniform grids without shufling. For each target, the training inputs, target values, and sample order are identical across configurations.

Architecture Configurations. The experiments follow the neuron-wise trainability convention in Section 2, considering �-trainable networks with equal hidden-layer width $W \in$ 32, 64, 128, 256 . The depth � includes the output layer, so each network has $L - 1$ hidden layers. In each hidden layer, �� neurons are trainable through their incoming weights and biases, while the remaining hidden parameters stay fixed after initialization. All neurons remain active in the forward computation, and the output layer, including its bias, is fully trainable. The main experiments use ReLU activation.

Table 2. Experimental configuration with ReLU activation.
<table><tr><td>Item</td><td>Value</td><td>Item</td><td>Value</td></tr><tr><td>Batch size</td><td>Full batch</td><td>Optimizer</td><td>Adam</td></tr><tr><td>Training steps</td><td>12,000</td><td>Base LR</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>Gradient clipping</td><td>10.0</td><td>LR schedule</td><td>ReduceLROnPlateau</td></tr><tr><td>Early stopping</td><td>Disabled</td><td>LR factor</td><td>0.5</td></tr><tr><td>Precision</td><td>float32</td><td>LR patience</td><td>400</td></tr><tr><td>Hardware</td><td>Single NVIDIA A100</td><td>LR threshold</td><td>10⁻6</td></tr><tr><td>Repeats</td><td>21</td><td>Minimum LR</td><td> $1 0 ^ { - 6 }$ </td></tr></table>

Experimental Set-up. The following settings are used to compare trainable proportions under shared initialization and a fixed training budget.

• Trainable Proportions. The trainable proportion � is evaluated on the following grid

$$
P \in \mathcal { G } _ { \mathrm { e x p } } = \frac { 1 } { 3 2 } \Big \{ 0 , \ 1 , \ 2 , \ \cdots , 7 , \ 8 , \ 1 0 , \ 1 2 , \ 1 4 , \ 1 6 , \ 2 0 , \ 2 4 , \ 2 8 , \ 3 2 \Big \} ,
$$

with finer spacing at small � to examine the phase transition. Every tested width is a multiple of 32, so �� is an integer at each grid point, without rounding.

• Depth and Width Variation. The depth experiments use � = 64 and $L \in \{ 2 , 3 , 4 , 5 , 7 , 9 ,$ 11, 13, 15 . The width experiments use $L = 6$ and � 32, 64, 128, 256 .

• Shared Initialization. Configurations with the same network architecture and repetition index share the same initial values of all parameters across �. The repetition index starts at 0. The initialization seeds are

$$
\mathrm { S e e d } _ { \mathrm { i n i t } } ( W ) = 4 2 + ( \mathrm { R e p e a t \ I n d e x } ) + 1 0 0 W ,
$$

for width variation, and

$$
\mathrm { S e e d _ { i n i t } } ( L ) = 4 2 + 1 0 0 ( \mathrm { R e p e a t \ I n d e x } ) + ( L - 1 ) ,
$$

for depth variation. Both formulas are independent of �. A dedicated torch.Generator generates initial weights and biases in a fixed order. For all activation functions, weights use Xavier-uniform initialization [18] with gain $5 / 3$ , and biases are sampled uniformly from the same interval as the corresponding layer’s weights.

• Repetitions and Training Budget. Each configuration is trained for 12,000 steps and repeated 21 times with diferent initialization seeds. Table 2 summarizes the settings.

![](images/16281b687389a8ff5d2df95d274203b0a3bda669efed2135be0ade8c45367331.jpg)  
(a) Depth variation loss.

![](images/44ced840924b7fc55e7fc418b922f67f1512a64c3db3703af858c4024b095aac.jpg)  
(b) Depth variation gain.

![](images/af23b25f8bb0dd4615ab5655ad6001435a2c2d54937c98ee549cd27285273a0b.jpg)  
(c) Width variation loss.

![](images/7b7f8e73d51d517f2e30b42271fc231ebb3e794ae3e5471cc43c674f0cc11dbe.jpg)  
(d) Width variation gain.  
Figure 10. Training results for $f _ { 1 } ^ { * }$ with ReLU activation. The top and bottom rows show depth and width variation, respectively. The left column reports the geometric mean of the best training loss across 21 repetitions, and the right column shows the normalized gain.

Evaluation Metrics. Let $\mathcal { L } _ { t , r } ( \boldsymbol { P } )$ denote the training mean-square error at step � in repetition �. For each run, the best training loss within the budget � is $\begin{array} { r } { \mathcal { E } _ { r } ( P ) = \operatorname* { m i n } _ { 0 \leq t \leq T } \mathcal { L } _ { t , r } ( P ) } \end{array}$ The losses are aggregated by averaging their logarithms across $R = 2 1$ repetitions and transforming back to the original scale. The corresponding geometric mean loss and normalized training gain are defined as

$$
\mathcal E ( P ) = \exp \Big ( \frac { 1 } { R } \sum _ { r = 1 } ^ { R } \log \mathcal E _ { r } ( P ) \Big ) = \Big ( \prod _ { r = 1 } ^ { R } \mathcal E _ { r } ( P ) \Big ) ^ { 1 / R } , \mathrm { ~ a n d ~ } G ( P ) = \frac { \mathcal E ( 0 ) - \mathcal E ( P ) } { \mathcal E ( 0 ) - \mathcal E ( 1 ) } .
$$

Thus $G ( P )$ is computed from the aggregated losses and measures the fraction ofthe observed reduction between $P = 0$ and $P = 1$ attained by partial training.

## 6.2. Experimental Results

Figures 10, 11, and 12 report the results for $f _ { 1 } ^ { * } , f _ { 2 } ^ { * }$ , and $f _ { 3 } ^ { * }$ , respectively. For each target, the top and bottom rows show depth and width variation, while the left and right columns show $\mathcal { E } ( P )$ and $G ( P )$ . The horizontal axes and the gain axes use linear scales, while the loss axes use logarithmic scales. Insets enlarge the regions where the gain is close to 1.

![](images/f8212b1af58507bda857a36e5ad29102a0a6ddceff9fc76636c6832e385cf88b.jpg)  
(a) Depth variation loss.

![](images/f3ad7ba4a2b8a8961fdc3c6bf76326c5016384ebbd39a920b4a7c22affc5af18.jpg)  
(b) Depth variation gain.

![](images/0b9fae92dd1e0fe01dcec002e567f949859c902bbada3e6b1abce5aa4ff9790c.jpg)  
(c) Width variation loss.

![](images/4d11982bff603bb3804de8164c5d453ee89dc2a9f18a16ac0d932f7e3b2ce946.jpg)  
(d) Width variation gain.

Figure 11. Training loss and normalized gain for $f _ { 2 } ^ { * }$ . The layout and axis scales follow Figure 10.  
![](images/4698323dce9f6efdd0663d4eee3518462f8152e72c15f79b77f9ecb3e39a8953.jpg)  
(a) Depth variation loss.

![](images/0adbe017b4928b2c4c2f735e68ce179256d84c6a2e114064a76fd46051742666.jpg)  
(b) Depth variation gain.

![](images/eb10793c2f1216dee75b9a8fce4dd4f35b36fc0955163dd37df98c97e12dbaec.jpg)  
(c) Width variation loss.

![](images/37288f8055cbf8877ae534e77b26952d5aa62d0ba61f05b7758b4e32ee1aad0e.jpg)  
(d) Width variation gain.  
Figure 12. Training loss and normalized gain for $f _ { 3 } ^ { * }$ . The layout and axis scales follow Figure 10.

The two metrics provide complementary information. The absolute loss $\mathcal { E } ( P )$ directly measures the attained errors, allowing comparisons across trainable proportions and architectures. The normalized gain $G ( P )$ expresses the loss reduction on a common relative scale across configurations with diferent endpoint losses. A gain close to 1 need not imply a small absolute error. The two metrics should therefore be interpreted together.

Relation to Approximation Theory. Before interpreting the numerical results, we note that the attained training losses depend on optimization, the finite training grids, and the training budget. They provide empirical indicators of approximation performance on these grids, so the comparison with theory concerns qualitative trends as � varies.

Results and Discussion. The figures lead to the following observations.

• Observed Phase Transition. Most configurations exhibit substantial loss reductions at small �, followed by diminishing gains as the normalized gain curves approach a plateau. This pattern is particularly clear for $f _ { 1 } ^ { * }$ and for the deeper networks tested on $f _ { 2 } ^ { * }$ and $f _ { 3 } ^ { * }$ These observations are qualitatively consistent with the small-� phase transition in the theoretical approximation bounds.

• Benefits of Partial Training. Partial training can achieve substantial loss reductions with far fewer trainable hidden parameters. For example, in the width experiments for $f _ { 1 } ^ { * }$ in Figure 10(d), all tested widths recover most of their endpoint improvement at $P = 1 / 3 2$ Thus, obtaining most of the observed improvement does not require training all hidden parameters. This provides a useful trade-of between the number of trainable hidden parameters and the attained loss reduction.

• Task-dependent Deviation. For the high-frequency target $f _ { 2 } ^ { * }$ , the two- and three-layer networks retain losses above $1 0 ^ { - 1 }$ throughout the tested range of � in Figure 11(a), indicating poor fitting even at full training. Under the current settings, the three-layer network’s loss fluctuates with �. Since neither architecture fits the target well under the current settings, their normalized gain curves are omitted from Figure 11(b).

## 6.3. Results with Other Activation Functions

The width and depth experiments are also conducted with GELU and tanh to examine whether similar phase transition patterns occur. Table 3 summarizes representative training losses and normalized gains at $L = 5$ and $W = 6 4$ . Both activations recover over 95% of the endpoint loss reduction at $P = 1 / 8$ for all three targets, consistent with the small-� phase transition behavior observed with ReLU. The complete plots are reported in Appendix E.

## 6.4. Expressive Role of Randomly Fixed Neurons

The width experiments include configurations with the same number of trainable hidden neurons but diferent numbers of randomly fixed neurons. Table 4 compares such configurations for $f _ { 1 } ^ { * }$ at $L = 6$ , with 32 trainable neurons per hidden layer.

Efect of Fixed Neuron Count. As the number of randomly fixed neurons increases, the observed loss decreases progressively, with a reduction of approximately 94% from $W = 3 2$

Table 3. Training losses and normalized gains for GELU and tanh with depth $L = 5$ and width $W = 6 4 \AA$ Losses are geometric means of the best training losses across 21 repetitions.
<table><tr><td>Activation</td><td>Target</td><td>ε(0)</td><td>8(1/32)</td><td>ε(1)</td><td>G(1/32)</td><td> ${ \bf G } ( { \bf 1 } / { \bf 8 } )$ </td></tr><tr><td rowspan="3">GELU</td><td> $f _ { 1 } ^ { * }$ </td><td> $9 . 7 0 \times 1 0 ^ { - 5 }$ </td><td> $2 . 1 3 \times 1 0 ^ { - 6 }$ </td><td> $3 . 3 3 \times 1 0 ^ { - 7 }$ </td><td>0.9814</td><td>0.9927</td></tr><tr><td> $f _ { 2 } ^ { * }$ </td><td> $6 . 8 0 \times 1 0 ^ { - 1 }$ </td><td> $1 . 7 8 \times 1 0 ^ { - 1 }$ </td><td> $7 . 6 9 \times 1 0 ^ { - 4 }$ </td><td>0.7390</td><td>0.9723</td></tr><tr><td> $f _ { 3 } ^ { * }$ </td><td> $1 . 7 2 \times 1 0 ^ { - 1 }$ </td><td> $1 . 5 9 \times 1 0 ^ { - 3 }$ </td><td> $4 . 2 4 \times 1 0 ^ { - 6 }$ </td><td>0.9908</td><td>0.9996</td></tr><tr><td rowspan="3">tanh</td><td> $f _ { 1 } ^ { * }$ </td><td> $3 . 6 3 \times 1 0 ^ { - 3 }$ </td><td> $1 . 9 6 \times 1 0 ^ { - 5 }$ </td><td> $4 . 6 9 \times 1 0 ^ { - 7 }$ </td><td>0.9947</td><td>0.9997</td></tr><tr><td> $f _ { 2 } ^ { * }$ </td><td> $6 . 1 8 \times 1 0 ^ { - 1 }$ </td><td> $3 . 1 7 \times 1 0 ^ { - 1 }$ </td><td> $3 . 3 0 \times 1 0 ^ { - 4 }$ </td><td>0.4867</td><td>0.9552</td></tr><tr><td> $f _ { 3 } ^ { * }$ </td><td> $6 . 3 5 \times 1 0 ^ { - 2 }$ </td><td> $4 . 1 8 \times 1 0 ^ { - 3 }$ </td><td> $1 . 1 1 \times 1 0 ^ { - 5 }$ </td><td>0.9343</td><td>0.9973</td></tr></table>

Table 4. Training losses for $f _ { 1 } ^ { * }$ with ReLU activation at depth 6, keeping 32 trainable neurons per hidden layer while varying the number of fixed neurons. Losses are geometric means of the best training losses across 21 repetitions. Percentage reductions are relative to $W = 3 2 .$
<table><tr><td>Width</td><td>P</td><td>Trainable Per Layer</td><td>Fixed Per Layer</td><td> $\varepsilon ( P )$ </td><td>Loss reduction (%)</td></tr><tr><td>32</td><td>1</td><td>32</td><td>0</td><td> $1 . 5 0 \times 1 0 ^ { - 5 }$ </td><td>0.00</td></tr><tr><td>64</td><td> $1 / 2$ </td><td>32</td><td>32</td><td> $4 . 8 2 \times 1 0 ^ { - 6 }$ </td><td>67.76</td></tr><tr><td>128</td><td> $1 / 4$ </td><td>32</td><td>96</td><td> $1 . 7 5 \times 1 0 ^ { - 6 }$ </td><td>88.33</td></tr><tr><td>256</td><td> $1 / 8$ </td><td>32</td><td>224</td><td> $\mathbf { 9 . 4 9 \times 1 0 ^ { - 7 } }$ </td><td>93.66</td></tr></table>

to $W = 2 5 6 .$ . The number of trainable neurons per hidden layer remains fixed, although the total number of trainable parameters varies with width. These results support the expressive role of the randomly fixed neurons.

## 7. Discussion and Conclusions

This work studied the approximation capacity of partially trainable networks. For two-layer networks, the finite-width bounds quantify how sampling mismatch afects approximation as the trainable proportion varies. Under polynomial spectral decay and leverage compatibility, the rates show that a small trainable group can substantially reduce the bound when its mismatch is much smaller than that of the fixed group. The spherical biased-ReLU analysis establishes the spectral decay and two-sided bounds on the initialization mismatch, while a structured three-layer example illustrates a similar phase transition and shows how nonlinear composition allows trainable neurons to use randomly fixed downstream features to obtain a smaller explicit approximation bound. Numerical experiments show similar phase transition patterns across targets and architectures and further support the expressive role of the randomly fixed neurons.

The approximation bounds rely on a distributional construction of the trainable parameters. The leverage-score distribution provides an efective choice for deriving the bounds, but does not characterize the parameters produced by gradient-based training. Connecting this construction to the evolution of trainable parameters during optimization is an important direction for understanding when the predicted gains can be attained in training.

Besides, a general phase transition theory for deeper �-trainable networks remains to be developed. Changes in upstream trainable parameters also change the inputs to later neurons, including those with fixed parameters. This dependence between layers complicates a direct extension of current two-layer kernel techniques. Further analysis is needed to quantify how such interactions afect approximation as � varies. Deterministic approximation techniques may help characterize more directly how a small trainable group uses the representation provided by the randomly fixed group.

## Overview of the Appendices

• Appendix A summarizes the notation and collects supplementary technical remarks.

• Appendix B proves the general approximation bounds and polynomial rates.

• Appendix C proves the spectral and mismatch bounds for spherical ReLU features.

• Appendix D derives error bounds for the structured two- and three-layer examples.

• Appendix E presents additional experiments that complement the main results.

## A. Additional Technical Notes

## A.1. The Summary of Notations

Table 5 summarizes the main notations used in the paper.

## A.2. Clarifications of the Assumptions

For convenience, we restate all assumptions used in the main text before discussing their roles and necessity.

Assumption 2.2 (Operator Regularity). Thefeature map � isjointly measurable and belongs to $L _ { 2 } ( X \times \Theta , \mu \otimes \rho )$

Clarification. Assumption 2.2 provides the operator regularity needed for the general analysis. Joint measurability and square integrability of $\phi$ make $S _ { \rho }$ a Hilbert–Schmidt operator. Consequently, $T _ { \rho } = S _ { \rho } S _ { \rho } ^ { * }$ is self-adjoint, positive semi-definite, trace-class, and compact. These properties justify the spectral decomposition and ensure that $d ( \lambda )$ is finite for every $\lambda > 0$ . The assumption is general and directly verifiable, for example, when the feature map is bounded on probability spaces.

Table 5. Notation used throughout the paper.
<table><tr><td>Symbols</td><td>Descriptions</td></tr><tr><td> $\chi \ \& \ \Theta$ </td><td>The input domain and the parameter space.</td></tr><tr><td> $\phi ( x ; \theta )$   $\mathcal { F } ( W , L )$ </td><td>Feature map and its reweighted form  $\phi _ { P } ( x ; \theta )$  The neural network class with W width and depth  $L$  in (2.1).</td></tr><tr><td> $\mathcal { F } _ { P } ( W , L )$   $k _ { \rho } ( x , x ^ { \prime } ) \ \& \ T _ { \rho }$ </td><td>The partially trainable counterpart of  $\mathcal { F } ( W , L )$  in (2.2). Kernel induced by  $\phi ( x ; \theta )$  (2.5) and the corresponding integral operator (2.6).</td></tr><tr><td> $\mathcal { H } _ { \rho }$ </td><td>Reproducing Kernel Hilbert Space (RKHS) induced by  $\rho$  in (2.8).</td></tr><tr><td> $\rho$ </td><td>Reference measure defining the target RKHS  $\mathcal { H } _ { \rho }$ </td></tr><tr><td> $\rho _ { 0 }$ </td><td>Prescribed initialization distribution for fixed neurons in Definition 2.3.</td></tr><tr><td> $\rho _ { 1 }$ </td><td>Adjustable sampling distribution for trainable neurons in Definition 2.3.</td></tr><tr><td> $\rho _ { P } \ \& \ \gamma _ { P }$ </td><td>Mixture measure and its density relative to  $\rho$  in (2.10)</td></tr><tr><td> $s _ { \lambda , P } ( \theta )$ </td><td>The ridge leverage score defined in (3.2).</td></tr><tr><td> $d _ { \operatorname* { m a x } } ( \gamma , \lambda )$ </td><td></td></tr><tr><td> $\boldsymbol { \kappa } _ { \gamma _ { P } } ( \lambda ) , \boldsymbol { \kappa } _ { 0 } ^ { * } , \boldsymbol { \kappa } _ { 1 } ^ { * }$ </td><td>The maximal ridge leverage defined in (3.3).</td></tr><tr><td> $P \ \& \ G ( P )$ </td><td>Mismatch factor of  $\rho _ { P } \ ( 3 . 7 ) ;$  Admissible constants  $\boldsymbol { \kappa } _ { 0 } ^ { * }$  for  $\rho _ { 0 }$  , and  $\boldsymbol { \kappa } _ { 1 } ^ { * }$  for  $\rho _ { 1 }$  Trainable proportion P and normalized gain  $G ( P )$  in (3.11).</td></tr><tr><td> $Y _ { k , j } \ : \& \ : \mathbb { Y } _ { k }$ </td><td>The real orthonormal basis  $\{ Y _ { k , j } \}$  of degree k spherical harmonics space  $\mathbb { Y } _ { k }$ </td></tr><tr><td> $\eta$ </td><td></td></tr><tr><td></td><td>Eigenvalues ordered by magnitude (with multiplicity) η in (2.7).</td></tr><tr><td> $\xi$ </td><td>Eigenvalues  $\xi$  indexed by spherical harmonic degree in (4.2).</td></tr><tr><td> $\lesssim \ \& \ \gtrsim$ </td><td> $A \lesssim B$  and  $B \gtrsim A$  denote  $A \leq c B$  for some positive constant  $c > 0 .$ </td></tr><tr><td> $\asymp$ </td><td> $A \asymp B$  means  $c _ { 1 } A \leq B \leq c _ { 2 } A$  for some positive constants  $c _ { 1 } , c _ { 2 } > 0 .$ </td></tr><tr><td> $O ( \cdot ) \ \& \ \tilde { O } ( \cdot )$ </td><td> ${ \tilde { O } } ( \cdot )$ </td></tr><tr><td></td><td> $O ( \cdot )$  is used to hide constant factors, while ignores logarithmic factors.</td></tr><tr><td> $\left\| \cdot \right\| _ { L _ { 2 } ( d \mu ) }$ </td><td> $L _ { 2 }$  norms under input measure and  $\| { \cdot } \| _ { L _ { 2 } ( d \rho ) }$  under parameter measure.</td></tr><tr><td> $a \otimes _ { L _ { 2 } ( d \mu ) } b$ </td><td></td></tr><tr><td></td><td> $( a \otimes _ { L _ { 2 } ( d \mu ) } b ) g = \langle b , g \rangle _ { L _ { 2 } ( d \mu ) } a$  for any  $g \in L _ { 2 } ( d \mu ) .$ </td></tr><tr><td> $A \preccurlyeq B , B \succcurlyeq A$ </td><td>Positive semidefinite order, meaning that  $B - A$  is positive semidefinite.</td></tr></table>

Assumption 3.4 (Leverage Compatibility). There exist $\lambda _ { 0 } > 0$ and constants $1 \le \kappa _ { 1 } ^ { * } \le \kappa _ { 0 } ^ { * } <$ such that,for each $0 < \lambda \leq \lambda _ { 0 } ,$ , the sampling densities satisfy $\kappa _ { \gamma _ { i } } ( \lambda ) \leq \kappa _ { i } ^ { * } f o r i \in \{ 0 , 1 \}$ Here, �0 isprescribed, while $\gamma _ { 1 }$ may depend on �and isfixed in each sampling construction.

Clarification. Assumption 3.4 is a uniform compatibility condition between each candidate sampling distribution density $\gamma _ { i }$ and the optimal ridge leverage density $\gamma _ { \lambda } ^ { \ast }$ . Indeed,

$$
\kappa _ { \gamma _ { i } } ( \lambda ) = \frac { d _ { \operatorname* { m a x } } ( \gamma _ { i } , \lambda ) } { d ( \lambda ) } = \exp \frac { \gamma _ { \lambda } ^ { * } ( \theta ) } { \theta \in \Theta } \frac { \gamma _ { \lambda } ^ { * } ( \theta ) } { \gamma _ { i } ( \theta ) } .
$$

Hence, the assumption requires $\gamma _ { i } ( \theta ) \geq \gamma _ { \lambda } ^ { \ast } ( \theta ) / \kappa _ { i } ^ { \ast }$ for $\rho \cdot$ -almost every $\theta$ at each $\lambda \in ( 0 , \lambda _ { 0 } ]$ Its role is to bound the sampling cost $d _ { \operatorname* { m a x } } ( \gamma , \lambda )$ by a constant multiple of the efective dimension $d ( \lambda )$ in the rate analysis. The ordering $\kappa _ { 1 } ^ { * } \leq \kappa _ { 0 } ^ { * }$ further expresses the advantage of the adjustable distribution at the level of these bounds.

This condition is closely related to the compatibility condition of [52, Assumption 2], which, in our notation, controls the maximal ridge leverage score by $d _ { \operatorname* { m a x } } ( \gamma , \lambda ) \leq C _ { \gamma } \lambda ^ { - C ^ { * } }$ for some constants $C _ { \gamma } > 0$ , and $C ^ { * } \in [ 0 , 1 ]$ . Our assumption likewise controls $d _ { \operatorname* { m a x } } ( \gamma _ { i } , \lambda )$ but requires a stronger constant-factor bound $d _ { \operatorname* { m a x } } ( \gamma _ { i } , \lambda ) \leq \kappa _ { i } ^ { * } d ( \lambda )$ . Under Assumption 3.5, we have $d ( \lambda ) \asymp \lambda ^ { - 1 / \alpha }$ . Hence our condition implies a bound of the former form with $C ^ { * } = 1 / \alpha$ for suficiently small �.

Assumption 3.5 (Polynomial Spectral Decay). There exist constants $c _ { - } , c _ { + } > 0$ and $\alpha > 1$ such that $c _ { - } k ^ { - \alpha } \leq \eta _ { k } \leq c _ { + } k ^ { - \alpha }$ for every $k \geq 1$

Clarification. Assumption 3.5 specifies the spectral scale needed to turn the general approximation bound into a polynomial rate. Its two-sided bounds imply $d ( \lambda ) \asymp \lambda ^ { - 1 / \alpha }$ (Derivation in (B.15)). The upper spectral bound controls the approximation rate, while the lower bound ensures that $\alpha$ describes the actual polynomial scale rather than only an upper envelope. The trace-class property of $T _ { \rho }$ alone does not establish this two-sided decay, but it can be verified for the particular kernel, as in the spherical ReLU example in Proposition 4.1.

## B. Proofs of Main Results in Section 3 – General Bounds

## Outline

• Subsection B.1 proves an operator Bernstein inequality with a variance proxy and uses it to establish the regularized operator concentration for the two independently sampled feature groups.

• Subsection B.2 first verifies kernel and operator invariance, then combines the concentration result with the ridge construction to prove the general approximation theorem. It subsequently proves the mixture leverage bound and combines this bound with sampling compatibility and polynomial spectral decay to derive polynomial approximation rates.

## B.1. Auxiliary Theoretical Preliminaries

To prove the general approximation result in Theorem 3.2, we utilize the following operator Bernstein inequality and the partial order bound on the deviation ${ \widehat T } _ { \rho } - T _ { \rho }$ between empirical and integral operators.

Lemma B.1 (Operator Bernstein Inequality With Variance Proxy). Let $X _ { 1 } , \ldots , X _ { m }$ be independent, centered, self-adjoint Hilbert–Schmidt random operators on a separable Hilbert space. Suppose that $\| X _ { j } \| _ { \mathsf { o p } } \leq U$ almost surelyfor $j \in [ m ]$ and some positive �. Thenfor $\sigma ^ { 2 } > 0$ with $\begin{array} { r } { \sigma ^ { 2 } \geq \| \sum _ { j = 1 } ^ { m } \mathbb { E } [ ( X _ { j } ) ^ { 2 } ] \| _ { \mathrm { o p } } } \end{array}$ and any $t > 0$ satisfying $t ^ { 2 } \ge \sigma ^ { 2 } + U t / 3$ , we have

$$
\mathbb { P } \Big ( \Big \| \sum _ { j = 1 } ^ { m } X _ { j } \Big \| _ { \mathrm { o p } } > t \Big ) \leq 1 4 \frac { \mathrm { t r } ( \sum _ { j = 1 } ^ { m } \mathbb { E } [ ( X _ { j } ) ^ { 2 } ] ) } { \sigma ^ { 2 } } \exp \Big ( - \frac { t ^ { 2 } / 2 } { \sigma ^ { 2 } + U t / 3 } \Big ) .
$$

Proof of Lemma B.1. The result follows from the intermediate estimate (3.9) in the proof of Theorem 3.1 of [42], which gives

$$
\mathbb { P } \Big ( \eta _ { \operatorname* { m a x } } \Big ( \sum _ { j = 1 } ^ { m } X _ { j } \Big ) > t \Big ) \leq \frac { \operatorname { t r } ( \sum _ { j = 1 } ^ { m } \mathbb { E } [ ( X _ { j } ) ^ { 2 } ] ) } { \sigma ^ { 2 } } \exp \Big ( - \frac { t ^ { 2 } / 2 } { \sigma ^ { 2 } + U t / 3 } \Big ) \Big ( 1 + \frac { 6 ( \sigma ^ { 2 } + U t / 3 ) ^ { 2 } } { t ^ { 4 } } \Big ) .
$$

The condition $t ^ { 2 } \ge \sigma ^ { 2 } + U t / 3$ implies the last factor is at most 7. Apply the same estimate to $- \textstyle \sum _ { j } X _ { j }$ and use a union bound to obtain the factor 14. Further with the operator extension of Minsker’s Bernstein inequality [42, Section 3.2], we complete the proof.

Proposition B.2 (Regularized Operator Concentration). Let $\lambda > 0 , t \in ( 0 , 1 )$ and assume that $0 \leq d _ { m a x } ( \gamma _ { P } , \lambda ) < \infty$ . Suppose that $\gamma _ { P } ( \theta ) > 0 f o r \rho \mathrm { - }$ almost every � with $\| \phi ( \cdot ; \theta ) \| _ { L _ { 2 } ( d \mu ) } >$ 0. Let $P = m _ { \mathrm { f r e e } } / m , 1 - P = m _ { \mathrm { f i x } } / m , m = m _ { \mathrm { f r e e } } + m _ { \mathrm { f i x } } \geq 1$ . Let the hidden parameters be sampled according to Definition 2.3. Define

$$
\widehat { T } _ { \rho } = \frac { 1 } { m } \Bigl ( \sum _ { j = 1 } ^ { m _ { \mathrm { f r e e } } } \phi _ { P } \bigl ( \cdot ; \theta _ { j } \bigr ) \otimes \phi _ { P } \bigl ( \cdot ; \theta _ { j } \bigr ) + \sum _ { j = 1 } ^ { m _ { \mathrm { f r e e } } } \phi _ { P } \bigl ( \cdot ; \bar { \theta } _ { j } \bigr ) \otimes \phi _ { P } \bigl ( \cdot ; \bar { \theta } _ { j } \bigr ) \Bigr ) .
$$

Then for any $\delta \in ( 0 , 1 )$ , if

$$
m \ge \operatorname* { m a x } \Bigg \{ d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) \Big ( \frac { 1 } { t ^ { 2 } } + \frac { 1 } { 3 t } \Big ) , 2 d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) \Big ( \frac { 1 } { t ^ { 2 } } + \frac { 1 } { 3 t } \Big ) \log \frac { 1 4 d ( \lambda ) } { \delta } \Bigg \} ,
$$

with probability at least $1 - \delta ,$ , we have

$$
( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } ( \widehat { T } _ { \rho } - T _ { \rho } ) ( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } \approx - t I , e q u i v a l e n t l y , \widehat { T } _ { \rho } + \lambda I \asymp ( 1 - t ) ( T _ { \rho } + \lambda I ) .
$$

Proof ofProposition B.2. We note that if $d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) = 0$ , then $\phi _ { P } ( \cdot ; \theta ) = 0$ for �<sub>�</sub>-almost every �, and $T _ { \rho } = \widehat { T } _ { \rho } = 0$ almost surely. The conclusion holds for every $m \geq 1$ . Thus, below assume $d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) > 0$ . We define the rank-one operator as

$$
A _ { P } ( \theta ) = \left( T _ { \rho } + \lambda I \right) ^ { - 1 / 2 } \bigl [ \phi _ { P } ( \cdot ; \theta ) \otimes _ { L _ { 2 } ( d \mu ) } \phi _ { P } ( \cdot ; \theta ) \bigr ] \left( T _ { \rho } + \lambda I \right) ^ { - 1 / 2 } .
$$

And the two related self-adjoint operators $X _ { j } ^ { \mathrm { f r e e } }$ and $X _ { j } ^ { \mathrm { f i x } }$ are defined as

$$
\begin{array} { l l } { { \displaystyle X _ { j } ^ { \mathrm { f r e e } } = \frac { 1 } { m } \big ( A _ { P } \big ( \theta _ { j } \big ) - \mathbb { E } _ { \theta \sim \rho _ { 1 } } \big [ A _ { P } ( \theta ) \big ] \big ) , ~ j \in [ m _ { \mathrm { f r e e } } ] , } } \\ { { \displaystyle X _ { j } ^ { \mathrm { f i x } } = \frac { 1 } { m } \big ( A _ { P } \big ( \bar { \theta } _ { j } \big ) - \mathbb { E } _ { \bar { \theta } \sim \rho _ { 0 } } \big [ A _ { P } ( \bar { \theta } ) \big ] \big ) , ~ j \in [ m _ { \mathrm { f i x } } ] . } } \end{array}
$$

By construction and the fixed group sizes, we have

$$
\begin{array} { l } { { \displaystyle \sum _ { j = 1 } ^ { m _ { \mathrm { f r e e } } } X _ { j } ^ { \mathrm { f r e e } } + \sum _ { j = 1 } ^ { m _ { \mathrm { f r } } } X _ { j } ^ { \mathrm { f r } } } } \\ { { = \displaystyle \frac { 1 } { m } \Big ( \sum _ { j = 1 } ^ { m _ { \mathrm { f r e e } } } A _ { P } ( \theta _ { j } ) + \sum _ { j = 1 } ^ { m _ { \mathrm { f r e e } } } A _ { P } ( \bar { \theta } _ { j } ) \Big ) - P \mathbb { E } _ { \theta \sim \rho _ { 1 } } [ A _ { P } ( \theta ) ] - ( 1 - P ) \mathbb { E } _ { \theta \sim \rho _ { 0 } } [ A _ { P } ( \theta ) ] } } \\ { { = ( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } \widehat { T } _ { \rho } ( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } - \mathbb { E } _ { \theta \sim \rho _ { P } } \left[ A _ { P } ( \theta ) \right] = ( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } ( \widehat { T } _ { \rho } - T _ { \rho } ) ( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } . } } \end{array}
$$

Therefore, we bound the operator norm of $\begin{array} { r l } { \sum _ { j = 1 } ^ { m _ { \mathrm { f r e e } } } X _ { j } ^ { \mathrm { f r e e } } + \sum _ { j = 1 } ^ { m _ { \mathrm { f i x } } } X _ { j } ^ { \mathrm { f i x } } } \end{array}$ with high probability to get the final result.

First, recall the following notation in (3.6) and (3.3),

$$
\begin{array} { r l } & { d ( \lambda ) = \mathrm { t r } ( T _ { \rho } ( T _ { \rho } + \lambda I ) ^ { - 1 } ) } \\ & { \qquad = \displaystyle \int _ { \Theta } \mathrm { t r } ( ( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } [ \phi _ { P } ( \cdot ; \theta ) \otimes \phi _ { P } ( \cdot ; \theta ) ] ( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } ) d \rho _ { P } ( \theta ) } \\ & { \qquad = \displaystyle \int _ { \Theta } \frac { \langle \phi ( \cdot ; \theta ) , ( T _ { \rho } + \lambda I ) ^ { - 1 } \phi ( \cdot ; \theta ) \rangle } { \gamma _ { P } ( \theta ) } \gamma _ { P } ( \theta ) d \rho ( \theta ) \leq d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) . } \end{array}
$$

For $\rho _ { P }$ -almost every $\theta ,$ , we have

$$
\begin{array} { l } { \displaystyle \| A _ { P } ( \theta ) \| _ { \mathrm { o p } } = \| \frac { 1 } { \gamma _ { P } ( \theta ) } ( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } [ \phi ( \cdot ; \theta ) \otimes _ { L _ { 2 } ( d \mu ) } \phi ( \cdot ; \theta ) ] ( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } \| _ { \mathrm { o p } } } \\ { = \displaystyle \frac { 1 } { \gamma _ { P } ( \theta ) } \| [ ( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } \phi ( \cdot ; \theta ) ] \otimes _ { L _ { 2 } ( d \mu ) } [ ( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } \phi ( \cdot ; \theta ) ] \| _ { \mathrm { o p } } } \\ { = \displaystyle \frac { \| ( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } \phi ( \cdot ; \theta ) \| _ { L _ { 2 } ( d \mu ) } ^ { 2 } } { \gamma _ { P } ( \theta ) } = \frac { \langle \phi ( \cdot ; \theta ) , ( T _ { \rho } + \lambda I ) ^ { - 1 } \phi ( \cdot ; \theta ) \rangle } { \gamma _ { P } ( \theta ) } } \\ { \leq d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) , } \end{array}\tag{B.1}
$$

where $\| \cdot \| _ { \mathrm { o p } }$ denotes the operator norm, which here equals the largest singular value. We use the identity $\| u \otimes _ { L _ { 2 } ( d \mu ) } u \| _ { \mathrm { o p } } = \| u \| _ { L _ { 2 } ( d \mu ) } ^ { 2 }$ for every $u \in L _ { 2 } ( d \mu )$ . By the construction $\left( \phi _ { P } \otimes \phi _ { P } \right)$ is positive) and (B.1), we have

$$
0 \precsim A _ { P } ( \theta ) \preccurlyeq d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) I .\tag{B.2}
$$

Second, by construction, we have $\mathbb { E } _ { \theta \sim \rho _ { 1 } } [ X _ { j } ^ { \mathrm { f r e e } } ] = 0$ and $\mathbb { E } _ { \bar { \theta } \sim \rho _ { 0 } } [ X _ { j } ^ { \mathrm { f i x } } ] = 0$ . Thus these operators are mutually independent, centered and identically distributed within each group, although they are not identically distributed across the two groups.

And for $X _ { j } ^ { \mathrm { f r e e } }$ and $X _ { j } ^ { \mathrm { f i x } }$ , by (B.2), we have

$$
- \frac { 1 } { m } d _ { \mathrm { m a x } } ( \gamma _ { P } , \lambda ) I \prec X _ { j } ^ { \mathrm { f r e e } } , X _ { j } ^ { \mathrm { f i x } } \prec \frac { 1 } { m } d _ { \mathrm { m a x } } ( \gamma _ { P } , \lambda ) I ,
$$

which means $\begin{array} { r } { \| X _ { j } ^ { \mathrm { f r e e } } \| _ { \mathrm { o p } } \leq \frac { d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) } { m } , j \in [ m _ { \mathrm { f r e e } } ] } \end{array}$ and $\begin{array} { r } { \| X _ { j } ^ { \mathrm { f i x } } \| _ { \mathrm { o p } } \leq \frac { d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) } { m } , j \in [ m _ { \mathrm { f i x } } ] } \end{array}$

Third, we need to estimate the operators’ variance $\mathbb { E } [ ( X _ { j } ^ { \mathrm { f r e e } } ) ^ { 2 } ]$ and $\mathbb { E } [ ( X _ { j } ^ { \mathrm { f i x } } ) ^ { 2 } ]$ . For $\mathbb { E } [ ( X _ { j } ^ { \mathrm { f r e e } } ) ^ { 2 } ]$ , we have

$$
\begin{array} { l l l } { \displaystyle \mathbb { E } _ { \rho _ { 1 } } \big [ \big ( X _ { j } ^ { \mathrm { f r e e } } \big ) ^ { 2 } \big ] = \frac { 1 } { m ^ { 2 } } \big ( \mathbb { E } _ { \rho _ { 1 } } \big [ A _ { P } ( \theta ) ^ { 2 } \big ] - \mathbb { E } _ { \rho _ { 1 } } \big [ A _ { P } ( \theta ) \big ] ^ { 2 } \big ) } \\ { \displaystyle \qquad \prec \frac { 1 } { m ^ { 2 } } \mathbb { E } _ { \rho _ { 1 } } \big [ A _ { P } ( \theta ) ^ { 2 } \big ] \preccurlyeq \frac { d _ { \mathrm { m a x } } \big ( \gamma _ { P } , \mathcal { \lambda } \big ) } { m ^ { 2 } } \mathbb { E } _ { \rho _ { 1 } } \big [ A _ { P } ( \theta ) \big ] . } \end{array}
$$

Similarly, the relation $\begin{array} { r } { \mathbb { E } _ { \rho _ { 0 } } [ ( X _ { j } ^ { \mathrm { f i x } } ) ^ { 2 } ] \prec \frac { d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) } { m ^ { 2 } } \mathbb { E } _ { \rho _ { 0 } } [ A _ { P } ( \theta ) ] } \end{array}$ holds for $X _ { j } ^ { \mathrm { f i x } }$ . Thus,

$$
\begin{array} { r l } & { \displaystyle \sum _ { j = 1 } ^ { m _ { \mathrm { f r e e } } } \mathbb { E } _ { \rho _ { 1 } } \big [ ( X _ { j } ^ { \mathrm { f r e e } } ) ^ { 2 } \big ] + \sum _ { j = 1 } ^ { m _ { \mathrm { f r e e } } } \mathbb { E } _ { \rho _ { 0 } } \big [ ( X _ { j } ^ { \mathrm { f r } } ) ^ { 2 } \big ] \prec \frac { d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) } { m } \Big ( \frac { m _ { \mathrm { f r e e } } } { m } \mathbb { E } _ { \rho _ { 1 } } \big [ A _ { P } ( \theta ) \big ] + \frac { m _ { \mathrm { f r e e } } } { m } \mathbb { E } _ { \rho _ { 0 } } \big [ A _ { P } ( \theta ) \big ] \Big ) } \\ & { = \frac { d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) } { m } \mathbb { E } _ { \rho _ { P } } \big [ A _ { P } ( \theta ) \big ] = \frac { d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) } { m } T _ { \rho } ( T _ { \rho } + \lambda I ) ^ { - 1 } . } \end{array}
$$

For notational convenience, denote $\begin{array} { r } { \sum _ { j = 1 } ^ { m _ { \mathrm { f r e e } } } \mathbb { E } _ { \rho _ { 1 } } \big [ ( X _ { j } ^ { \mathrm { f r e e } } ) ^ { 2 } \big ] + \sum _ { j = 1 } ^ { m _ { \mathrm { f r e e } } } \mathbb { E } _ { \rho _ { 0 } } \big [ ( X _ { j } ^ { \mathrm { f i x } } ) ^ { 2 } \big ] \ \mathrm { a s } \sum _ { j = 1 } ^ { m } \mathbb { E } \big [ ( X _ { j } ) ^ { 2 } \big ] } \end{array}$ Therefore, we have

$$
\begin{array} { r l } & { \displaystyle \left\| \displaystyle \sum _ { j = 1 } ^ { m } \mathbb { E } [ ( X _ { j } ) ^ { 2 } ] \right\| _ { \mathrm { o p } } \leq \frac { d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) } { m } \mathrm { , ~ a n d ~ } } \\ & { \displaystyle \operatorname { t r } \Big ( \displaystyle \sum _ { j = 1 } ^ { m } \mathbb { E } [ ( X _ { j } ) ^ { 2 } ] \Big ) \leq \frac { d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) } { m } \operatorname { t r } ( T _ { \rho } ( T _ { \rho } + \lambda I ) ^ { - 1 } ) = \frac { d ( \lambda ) d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) } { m } . } \end{array}
$$

Finally, we apply the operator Bernstein inequality. By Lemma B.1, for � satisfying (B.4) below, we have

$$
\mathbb { P } \left( \left. \sum _ { j = 1 } ^ { m _ { \mathrm { f r e c } } } X _ { j } ^ { \mathrm { f r e e } } + \sum _ { j = 1 } ^ { m _ { \mathrm { f i x } } } X _ { j } ^ { \mathrm { f r } } \right. _ { \mathrm { o p } } > t \right) \leq 1 4 d ( \lambda ) \exp \Big ( - \frac { t ^ { 2 } / 2 } { \big ( d _ { \operatorname* { m a x } } \big ( \gamma _ { P } , \lambda \big ) / m \big ) \big ( 1 + t / 3 \big ) } \Big ) .\tag{B.3}
$$

Lemma B.1 requires that, with $\sigma ^ { 2 } = U = d _ { \mathrm { m a x } } ( \gamma _ { P } , \lambda ) / m$

$$
t ^ { 2 } \geq \sigma ^ { 2 } + { \frac { U t } { 3 } } \Longrightarrow m \geq d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) \Bigl ( { \frac { 1 } { t ^ { 2 } } } + { \frac { 1 } { 3 t } } \Bigr ) .\tag{B.4}
$$

At the same time, for any $\delta \in ( 0 , 1 )$ , with $d ( \lambda ) \leq d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda )$ , to make the upper bound in (B.3) at most �, it sufices that

$$
m \geq 2 d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) \biggl ( \frac { 1 } { t ^ { 2 } } + \frac { 1 } { 3 t } \biggr ) \log \frac { 1 4 d ( \lambda ) } { \delta } .\tag{B.5}
$$

Thus when (B.4) and (B.5) are valid, we have with probability at least $1 - \delta .$

$$
\bigg \| \sum _ { j = 1 } ^ { m _ { \mathrm { f r e e } } } X _ { j } ^ { \mathrm { f r e e } } + \sum _ { j = 1 } ^ { m _ { \mathrm { f i x } } } X _ { j } ^ { \mathrm { f i x } } \bigg \| _ { \mathrm { o p } } \leq t , \ \mathrm { a n d } \ ( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } ( \widehat T _ { \rho } - T _ { \rho } ) ( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } \asymp - t I .
$$

## B.2. Proofs of the Main Results in Section 3

Proof ofLemma 3.1 (Invariance ofKernel and Operator).

For item 1, the changes of measure below use $\gamma _ { P } > 0$ for �-almost every parameter in the nonzero-feature region, with the reweighted feature set to zero elsewhere. By Fubini’s theorem, this zero-feature convention preserves the kernel for $( \mu \otimes \mu )$ -almost every input

pair. All kernel identities below are understood in this sense. Since $d \rho _ { P } = \gamma _ { P } d \rho$ in (2.10) and $\phi _ { P } = \gamma _ { P } ^ { - 1 / 2 } \phi$ on the nonzero-feature region, we have

$$
k _ { \rho } ( x , x ^ { \prime } ) = \int _ { \Theta } \phi ( x ; \theta ) \gamma _ { P } ^ { - 1 } ( \theta ) \phi ( x ^ { \prime } ; \theta ) \gamma _ { P } ( \theta ) d \rho ( \theta ) = \int _ { \Theta } \phi _ { P } ( x ; \theta ) \phi _ { P } ( x ^ { \prime } ; \theta ) d \rho _ { P } ( \theta ) .
$$

For $0 < P < 1$ , linearity of expectation and the fixed group sizes yield

$$
\begin{array} { r l } & { \mathbb { E } [ \hat { k } _ { \rho } ^ { ( P ) } ( x , x ^ { \prime } ) ] = \frac { m _ { \mathrm { f r e e } } } { m } \mathbb { E } _ { \theta \sim \rho _ { 1 } } [ \phi _ { P } ( x ; \theta ) \phi _ { P } ( x ^ { \prime } ; \theta ) ] + \frac { m _ { \mathrm { f i x } } } { m } \mathbb { E } _ { \theta \sim \rho _ { 0 } } [ \phi _ { P } ( x ; \theta ) \phi _ { P } ( x ^ { \prime } ; \theta ) ] } \\ & { \qquad = \displaystyle \int _ { \Theta } \phi _ { P } ( x ; \theta ) \phi _ { P } ( x ^ { \prime } ; \theta ) d \big ( P \rho _ { 1 } + ( 1 - P ) \rho _ { 0 } \big ) ( \theta ) = k _ { \rho } ( x , x ^ { \prime } ) . } \end{array}
$$

For $P = 0$ or $P = 1$ , the same identity follows directly from the single nonempty group.

For item 2, Assumption 2.2 gives

$$
\int _ { \Theta } \left\| \phi _ { P } ( \cdot ; \theta ) \right\| _ { L _ { 2 } ( d \mu ) } ^ { 2 } d \rho _ { P } ( \theta ) = \int _ { \Theta } \left\| \phi ( \cdot ; \theta ) \right\| _ { L _ { 2 } ( d \mu ) } ^ { 2 } d \rho ( \theta ) < \infty .
$$

Thus the integral of the rank-one operators is well defined. By the change of measure,

$$
\int _ { \Theta } \phi _ { P } ( \cdot ; \theta ) \otimes \phi _ { P } ( \cdot ; \theta ) d \rho _ { P } ( \theta ) = \int _ { \Theta } \gamma _ { P } ( \theta ) ^ { - 1 } \big ( \phi ( \cdot ; \theta ) \otimes \phi ( \cdot ; \theta ) \big ) \gamma _ { P } ( \theta ) d \rho ( \theta ) = T _ { \rho } .
$$

Thus the original spectral decomposition $\begin{array} { r } { T _ { \rho } = \sum _ { k \geq 1 } \eta _ { k } { \left( e _ { k } \otimes e _ { k } \right) } } \end{array}$ and its eigenpairs remain unchanged.

Proof of Theorem 3.2 (General Approximation Bounds). The following proof builds on the general strategy of[4, Proposition 1], but requires a nontrivial modification. The original proof considers i.i.d. features drawn from a single distribution, while our sampling scheme consists of two independent groups drawn separately from $\rho _ { 1 }$ and $\rho _ { 0 }$ . We therefore extend the proof to the two-group sampling setting and use the variance-proxy form of Bernstein’s inequality.

Proof Overview. The ridge solution reduces control of both the approximation error and the coeficient norm to a common quadratic form involving the regularized empirical operator in (B.10). The variance-proxy Bernstein inequality provides the operator comparison needed to bound this quadratic form by the target’s squared RKHS norm, yielding (B.13). Since the concentration event does not depend on the target function, the resulting bounds hold uniformly over the RKHS unit ball. Finally, choosing $\lambda = \varepsilon / 4$ in (B.14) gives the prescribed accuracy.

The proof details are demonstrated below.

Ridge Construction. First, by Definition 2.3, we recall necessary notations:

$$
d \rho _ { 1 } ( \theta ) = \gamma _ { 1 } ( \theta ) d \rho ( \theta ) , d \rho _ { 0 } ( \theta ) = \gamma _ { 0 } ( \theta ) d \rho ( \theta ) .
$$

Assuming $P m \in \mathbb { N } ,$ , we have $m _ { \mathrm { f r e e } } = P m , m _ { \mathrm { f i x } } = ( 1 - P ) m , \gamma _ { P } \left( \theta \right) = P \gamma _ { 1 } ( \theta ) + ( 1 - P ) \gamma _ { 0 } ( \theta )$ and $d \rho _ { P } ( \theta ) = \gamma _ { P } ( \theta ) d \rho ( \theta )$ . By (2.8), there exists $a _ { f } ^ { \ast } \in L _ { 2 } ( d \rho )$ such that

$$
f = \int _ { \Theta } a _ { f } ^ { \ast } ( \theta ) \phi ( \cdot ; \theta ) d \rho = \int _ { \Theta } \gamma _ { P } ^ { - 1 / 2 } a _ { f } ^ { \ast } ( \theta ) \phi _ { P } ( \cdot ; \theta ) d \rho _ { P } ( \theta ) ,
$$

with $\| a _ { f } ^ { * } \| _ { L _ { 2 } ( d \rho ) } = \| \gamma _ { P } ^ { - 1 / 2 } a _ { f } ^ { * } \| _ { L _ { 2 } ( d \rho _ { P } ) } = \| f \| _ { \mathcal { H } _ { \rho } }$ . Therefore, it sufices to consider $\| f \| _ { \mathcal { H } _ { \rho } } \leq 1$ and set $a _ { f } = \gamma _ { P } ^ { - 1 / 2 } a _ { f } ^ { * }$ when $\gamma _ { P } > 0$ and $ { \boldsymbol { a } } _ { f } = 0$ when $\gamma _ { P } = 0$ . With the regularization parameter $\lambda > 0$ , now consider the following minimization problem

$$
\operatorname* { m i n } _ { \beta } \left\| \int - \sum _ { j = 1 } ^ { m _ { \mathrm { f r e e } } } \beta _ { j } \phi _ { P } \big ( \cdot ; \theta _ { j } \big ) - \sum _ { j = 1 } ^ { m _ { \mathrm { f r e e } } } \beta _ { m _ { \mathrm { f r e e } } + j } \phi _ { P } \big ( \cdot ; \bar { \theta } _ { j } \big ) \right\| _ { L _ { 2 } ( d \mu ) } ^ { 2 } + \lambda m \| \beta \| _ { 2 } ^ { 2 } ,\tag{B.6}
$$

where $\phi _ { P } ( \cdot ; \theta ) = \gamma _ { P } ( \theta ) ^ { - 1 / 2 } \phi ( \cdot ; \theta ) , \theta _ { j } \stackrel { \mathrm { i . i . d . } } { \sim } \rho _ { 1 } , j \in [ m _ { \mathrm { f r e e } } ]$ , and $\bar { \theta } _ { j } \stackrel { \mathrm { i . i . d . } } { \sim } \rho _ { 0 } , j \in [ m _ { \mathrm { f i x } } ]$

We consider the operators $\Phi _ { \mathrm { f r e e } } : \mathbb { R } ^ { m _ { \mathrm { f r e e } } }  L _ { 2 } ( X , d \mu )$ and $\Phi _ { \mathrm { f i x } } : \mathbb { R } ^ { m _ { \mathrm { f i x } } }  L _ { 2 } ( X , d \mu )$ such that

$$
\Phi _ { \mathrm { f r e e } } \beta _ { \mathrm { f r e e } } = \sum _ { j = 1 } ^ { m _ { \mathrm { f r e e } } } \beta _ { j } \phi _ { P } ( \cdot ; \theta _ { j } ) , \ \Phi _ { \mathrm { f i x } } \beta _ { \mathrm { f i x } } = \sum _ { j = 1 } ^ { m _ { \mathrm { f i x } } } \beta _ { m _ { \mathrm { f r e e } } + j } \phi _ { P } ( \cdot ; \bar { \theta } _ { j } ) .
$$

Further define $\Phi _ { P } : \mathbb { R } ^ { m _ { \mathrm { f r e e } } } \times \mathbb { R } ^ { m _ { \mathrm { f i x } } }  L _ { 2 } ( X , d \mu )$ , with $\Phi _ { P } = [ \Phi _ { \mathrm { f r e e } } \Phi _ { \mathrm { f i x } } ]$ and $\Phi _ { P } \beta = \Phi _ { \mathrm { f r e e } } \beta _ { \mathrm { f r e e } } +$ $\Phi _ { \mathrm { f i x } } \beta _ { \mathrm { f i x } }$ . Therefore, we solve the following regularized least-squares problem

$$
\operatorname* { m i n } _ { \beta } \big \| f - \Phi _ { P } \beta \big \| _ { L _ { 2 } ( d \mu ) } ^ { 2 } + \lambda m ( \| \beta _ { \mathrm { f r e e } } \| _ { 2 } ^ { 2 } + \| \beta _ { \mathrm { f i x } } \| _ { 2 } ^ { 2 } ) ,
$$

whose solution is

$$
( \beta _ { \mathrm { f r e e } } , \beta _ { \mathrm { f i x } } ) = ( \Phi _ { P } ^ { * } \Phi _ { P } + \lambda m I ) ^ { - 1 } \Phi _ { P } ^ { * } f = \frac { 1 } { m } \Phi _ { P } ^ { * } \bigl ( \frac { 1 } { m } \Phi _ { P } \Phi _ { P } ^ { * } + \lambda I \bigr ) ^ { - 1 } f \in \mathbb { R } ^ { m } .\tag{B.7}
$$

We thus denote the empirical integral operator $\widehat { T } _ { \rho } : L _ { 2 } ( X , d \mu ) \to L _ { 2 } ( X , d \mu )$ as

$$
\widehat { T } _ { \rho } = \frac { 1 } { m } \Phi _ { P } \Phi _ { P } ^ { * } = \frac { 1 } { m } \bigg [ \sum _ { j = 1 } ^ { m _ { \mathrm { f r e c } } } \phi _ { P } ( \cdot ; \theta _ { j } ) \otimes \phi _ { P } ( \cdot ; \theta _ { j } ) + \sum _ { j = 1 } ^ { m _ { \mathrm { f r } } } \phi _ { P } ( \cdot ; \bar { \theta } _ { j } ) \otimes \phi _ { P } ( \cdot ; \bar { \theta } _ { j } ) \bigg ] .
$$

By construction, for $\widehat { T } _ { \rho }$ , we have $\mathbb { E } [ \widehat { T } _ { \rho } ] = T _ { \rho }$ from

$$
\begin{array} { l } { \displaystyle \mathbb { E } [ \widehat { T } _ { \rho } ] = \frac { m _ { \mathrm { f r e e } } } { m } \int _ { \Theta } \frac { \gamma _ { 1 } ( \theta ) } { \gamma _ { P } ( \theta ) } \phi ( \cdot ; \theta ) \otimes \phi ( \cdot ; \theta ) d \rho ( \theta ) + \frac { m _ { \mathrm { f r e e } } } { m } \int _ { \Theta } \frac { \gamma _ { 0 } ( \theta ) } { \gamma _ { P } ( \theta ) } \phi ( \cdot ; \theta ) \otimes \phi ( \cdot ; \theta ) d \rho ( \theta ) } \\ { \displaystyle \quad = \int _ { \Theta } \frac { P \gamma _ { 1 } ( \theta ) + ( 1 - P ) \gamma _ { 0 } ( \theta ) } { \gamma _ { P } ( \theta ) } \phi ( \cdot ; \theta ) \otimes \phi ( \cdot ; \theta ) d \rho ( \theta ) } \\ { \displaystyle \quad = \int _ { \Theta } \phi ( \cdot ; \theta ) \otimes \phi ( \cdot ; \theta ) d \rho ( \theta ) = T _ { \rho } , } \end{array}
$$

where we utilize the $m _ { \mathrm { f r e e } } / m = P , m _ { \mathrm { f i x } } / m = 1 - P$ and definition of $\gamma _ { P } ( \theta )$

Error and Coeficient Bounds. Now we consider $\| f - \Phi _ { P } \beta \| _ { L _ { 2 } ( d \mu ) } ^ { 2 }$

$$
\begin{array} { r } { \| f - \Phi _ { P } \beta \| _ { L _ { 2 } ( d \mu ) } ^ { 2 } = \| f - \widehat { T _ { \rho } } ( \widehat { T _ { \rho } } + \lambda I ) ^ { - 1 } f \| _ { L _ { 2 } ( d \mu ) } ^ { 2 } = \| \lambda ( \widehat { T _ { \rho } } + \lambda I ) ^ { - 1 } f \| _ { L _ { 2 } ( d \mu ) } ^ { 2 } } \\ { = \lambda ^ { 2 } \langle f , ( \widehat { T _ { \rho } } + \lambda I ) ^ { - 2 } f \rangle _ { L _ { 2 } ( d \mu ) } \leq \lambda \langle f , ( \widehat { T _ { \rho } } + \lambda I ) ^ { - 1 } f \rangle _ { L _ { 2 } ( d \mu ) } , } \end{array}\tag{B.8}
$$

where the classical partial order between self-adjoint operators indicates $( \widehat { T } _ { \rho } + \lambda I ) ^ { - 2 } \preccurlyeq$ $\lambda ^ { - 1 } ( \widehat { T } _ { \rho } + \lambda I ) ^ { - 1 }$

Then consider �� $( \| \beta _ { \mathrm { f r e e } } \| _ { 2 } ^ { 2 } + \| \beta _ { \mathrm { f i x } } \| _ { 2 } ^ { 2 } )$ , with (B.7), we have

$$
\begin{array} { r l } & { \displaystyle \lambda m ( \| \beta _ { \mathrm { f r e e } } \| _ { 2 } ^ { 2 } + \| \beta _ { \mathrm { f i x } } \| _ { 2 } ^ { 2 } ) } \\ & { = \lambda m \langle \displaystyle \frac { 1 } { m } \Phi _ { P } ^ { * } ( \frac { 1 } { m } \Phi _ { P } \Phi _ { P } ^ { * } + \lambda I ) ^ { - 1 } f , \displaystyle \frac { 1 } { m } \Phi _ { P } ^ { * } ( \frac { 1 } { m } \Phi _ { P } \Phi _ { P } ^ { * } + \lambda I ) ^ { - 1 } f \rangle _ { \mathbb { R } ^ { m } } } \\ & { = \lambda \langle ( \widehat { T _ { \rho } } + \lambda I ) ^ { - 1 } f , \widehat { T _ { \rho } } ( \widehat { T _ { \rho } } + \lambda I ) ^ { - 1 } f \rangle _ { L _ { 2 } ( d \mu ) } \leq \lambda \langle f , ( \widehat { T _ { \rho } } + \lambda I ) ^ { - 1 } f \rangle _ { L _ { 2 } ( d \mu ) } , } \end{array}\tag{B.9}
$$

where we use $( \widehat { T } _ { \rho } + \lambda I ) ^ { - 2 } \widehat { T } _ { \rho } \preccurlyeq ( \widehat { T } _ { \rho } + \lambda I ) ^ { - 1 }$

With (B.6), (B.8) and (B.9), both quantities are bounded by

$$
\begin{array} { r } { \operatorname* { m a x } \{ \| f - \Phi _ { P } \beta \| _ { L _ { 2 } ( d \mu ) } ^ { 2 } , \lambda m ( \| \beta _ { \mathrm { f r e e } } \| _ { 2 } ^ { 2 } + \| \beta _ { \mathrm { f i x } } \| _ { 2 } ^ { 2 } ) \} \le \lambda \langle f , ( \widehat { T } _ { \rho } + \lambda I ) ^ { - 1 } f \rangle _ { L _ { 2 } ( d \mu ) } } \end{array}\tag{B.10}
$$

Uniform Control over the RKHS Unit Ball. To bound $\langle f , ( \widehat { T } _ { \rho } + \lambda I ) ^ { - 1 } f \rangle _ { L _ { 2 } ( d \mu ) }$ , use the operator factorization in [16],

$$
\begin{array} { r l } & { \langle f , ( \widehat { T _ { \rho } } + \lambda I ) ^ { - 1 } f \rangle _ { L _ { 2 } ( d \mu ) } = \langle f , ( T _ { \rho } + \lambda I + \widehat { T _ { \rho } } - T _ { \rho } ) ^ { - 1 } f \rangle _ { L _ { 2 } ( d \mu ) } } \\ & { = \langle f , ( T _ { \rho } + \lambda I + ( T _ { \rho } + \lambda I ) ^ { 1 / 2 } ( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } ( \widehat { T _ { \rho } } - T _ { \rho } ) ( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } ( T _ { \rho } + \lambda I ) ^ { 1 / 2 } ) ^ { - 1 } f \rangle _ { L _ { 2 } ( d \mu ) } } \\ & { = \langle ( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } f , ( I + ( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } ( \widehat { T _ { \rho } } - T _ { \rho } ) ( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } ) ^ { - 1 } ( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } f \rangle _ { L _ { 2 } ( d \mu ) } . } \end{array}
$$

Based on Proposition B.2, for any $\delta \in ( 0 , 1 )$ , we can prove that the deviation part satisfies, for any $t \in ( 0 , 1 )$ , with probability at least $1 - \delta$

$$
( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } ( \widehat { T } _ { \rho } - T _ { \rho } ) ( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } \succcurlyeq - t I .
$$

Then we consider $\textstyle t = { \frac { 3 } { 4 } }$ . Note that the choice $t = 3 / 4$ is made only for notational convenience. Any fixed $t \in ( 0 , 1 )$ leads to the same approximation rate, only with diferent universal constants. We require that

$$
m \geq \operatorname* { m a x } \left\{ \frac { 2 0 } { 9 } d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) , \frac { 4 0 } { 9 } d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) \log \frac { 1 4 d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) } { \delta } \right\}\tag{B.11}
$$

Since $m \in \mathbb { N } _ { + }$ , we can turn (B.11) into

$$
m \ge \frac { 4 0 } { 9 } d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) \log \frac { 1 4 d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) } { \delta } ,\tag{B.12}
$$

which is because when $\begin{array} { r } { d _ { \mathrm { m a x } } \leq \frac { 9 } { 2 0 } , m \geq \frac { 2 0 } { 9 } d _ { \mathrm { m a x } } } \end{array}$ naturally holds while when $\begin{array} { r } { d _ { \mathrm { m a x } } > \frac { 9 } { 2 0 } . } \end{array}$ log $\begin{array} { r } { \frac { 1 4 d _ { \operatorname* { m a x } } } { \delta } > \log ( 1 4 \cdot \frac { 9 } { 2 0 } ) > \frac { 1 } { 2 } } \end{array}$ . Thus when (B.12) is valid, with probability at least $1 - \delta$

$$
( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } ( \widehat { T } _ { \rho } - T _ { \rho } ) ( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } \stackrel { _ { \ast } } { _ { \ast } } - \frac { 3 } { 4 } I .
$$

Therefore,

$$
\begin{array} { r l } & { \langle f , ( \widehat { T } _ { \rho } + \lambda I ) ^ { - 1 } f \rangle _ { L _ { 2 } ( d \mu ) } \leq \langle ( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } f , ( 1 - t ) ^ { - 1 } ( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } f \rangle _ { L _ { 2 } ( d \mu ) } } \\ & { \qquad = ( 1 - t ) ^ { - 1 } \langle f , ( T _ { \rho } + \lambda I ) ^ { - 1 } f \rangle _ { L _ { 2 } ( d \mu ) } } \\ & { \qquad = 4 \langle f , ( T _ { \rho } + \lambda I ) ^ { - 1 } f \rangle _ { L _ { 2 } ( d \mu ) } . } \end{array}\tag{B.13}
$$

We next bound $\langle f , ( T _ { \rho } + \lambda I ) ^ { - 1 } f \rangle _ { L _ { 2 } ( d \mu ) }$ . By Cauchy-Schwarz inequality, we have for any $b \in L _ { 2 } ( X , d \mu )$

$$
\begin{array} { r l } & { \langle b , ( f \otimes _ { L _ { 2 } ( d \mu ) } f ) b \rangle _ { L _ { 2 } ( d \mu ) } = \Big ( \displaystyle \int _ { X } b ( x ) f ( x ) d \mu ( x ) \Big ) ^ { 2 } } \\ & { = \Big ( \displaystyle \int _ { X } \int _ { \Theta } b ( x ) a _ { f } ( \theta ) \phi _ { P } ( x ; \theta ) d \mu ( x ) d \rho _ { P } ( \theta ) \Big ) ^ { 2 } } \\ & { \le \displaystyle \int _ { \Theta } a _ { f } ( \theta ) ^ { 2 } d \rho _ { P } ( \theta ) \cdot \displaystyle \int _ { \Theta } \Big ( \displaystyle \int _ { X } b ( x ) \phi _ { P } ( x ; \theta ) d \mu ( x ) \Big ) ^ { 2 } d \rho _ { P } ( \theta ) } \\ & { \le \| a _ { f } \| _ { L _ { 2 } ( d \rho _ { P } ) } ^ { 2 } \cdot \displaystyle \int _ { \Theta } \langle b , ( \phi _ { P } \otimes _ { L _ { 2 } ( d \mu ) } \phi _ { P } ) b \rangle _ { L _ { 2 } ( d \mu ) } d \rho _ { P } ( \theta ) \le \langle b , T _ { \rho } b \rangle _ { L _ { 2 } ( d \mu ) } , } \end{array}
$$

where $\begin{array} { r } { T _ { \rho } = \int _ { \Theta } \phi _ { P } ( \cdot , \theta ) \otimes _ { L _ { 2 } ( d \mu ) } \phi _ { P } ( \cdot , \theta ) d \rho _ { P } ( \theta ) } \end{array}$ . This means $f \otimes _ { L _ { 2 } ( d \mu ) } f \preccurlyeq T _ { \rho }$ . Using this inequality with $( T _ { \rho } + \lambda I ) ^ { - 1 / 2 }$ gives

$$
[ ( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } f ] \otimes [ ( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } f ] \preccurlyeq [ ( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } ] T _ { \rho } [ ( T _ { \rho } + \lambda I ) ^ { - 1 / 2 } ] \preccurlyeq I .
$$

This yields $\langle f , ( T _ { \rho } + \lambda I ) ^ { - 1 } f \rangle _ { L _ { 2 } ( d \mu ) } \leq 1$ . Consequently, $\langle f , ( \widehat { T } _ { \rho } + \lambda I ) ^ { - 1 } f \rangle _ { L _ { 2 } ( d \mu ) } \leq 4$ . Further with (B.10) and (B.13), we have with probability at least $1 - \delta$

$$
\| f - \Phi _ { P } \beta \| _ { L _ { 2 } ( d \mu ) } ^ { 2 } \le \lambda \langle f , ( \widehat { T } _ { \rho } + \lambda I ) ^ { - 1 } f \rangle _ { L _ { 2 } ( d \mu ) } \le \lambda \cdot \frac { 1 } { 1 - t } = 4 \lambda .\tag{B.14}
$$

The coeficients also satisfy

$$
\lambda m ( \| \beta _ { \mathrm { f r e e } } \| _ { 2 } ^ { 2 } + \| \beta _ { \mathrm { f i x } } \| _ { 2 } ^ { 2 } ) \leq 4 \lambda \Longrightarrow \| \beta _ { \mathrm { f r e e } } \| _ { 2 } ^ { 2 } + \| \beta _ { \mathrm { f i x } } \| _ { 2 } ^ { 2 } = \| \beta \| _ { 2 } ^ { 2 } \leq \frac { 4 } { m } ,
$$

when (B.12) is satisfied. Let $\lambda = \varepsilon / 4$ for any $\varepsilon > 0$ , and we prove the desired result.

Proof ofLemma 3.3 (Mixture Leverage Bound). By the definition of $d _ { \mathrm { m a x } }$ , the following inequalities hold �-almost everywhere.

$$
\langle \phi ( \cdot ; \theta ) , ( T _ { \rho } + \lambda I ) ^ { - 1 } \phi ( \cdot ; \theta ) \rangle _ { L _ { 2 } ( d \mu ) } \leq d _ { \operatorname* { m a x } } ( \gamma _ { i } , \lambda ) \gamma _ { i } ( \theta ) , \mathrm { f o r } i \in \{ 0 , 1 \} .
$$

With $\gamma _ { P } = P \gamma _ { 1 } + ( 1 - P ) \gamma _ { 0 }$ , it follows that

$$
\gamma _ { P } ( \theta ) \geq \Big ( \frac { P } { d _ { \operatorname* { m a x } } ( \gamma _ { 1 } , \lambda ) } + \frac { 1 - P } { d _ { \operatorname* { m a x } } ( \gamma _ { 0 } , \lambda ) } \Big ) \langle \phi ( \cdot ; \theta ) , ( T _ { \rho } + \lambda I ) ^ { - 1 } \phi ( \cdot ; \theta ) \rangle _ { L _ { 2 } ( d \mu ) } .
$$

Dividing by $\gamma _ { P } ( \theta )$ and taking the essential supremum yields

$$
d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) \leq \Big ( \frac { P } { d _ { \operatorname* { m a x } } ( \gamma _ { 1 } , \lambda ) } + \frac { 1 - P } { d _ { \operatorname* { m a x } } ( \gamma _ { 0 } , \lambda ) } \Big ) ^ { - 1 } = \Big ( \frac { P } { \kappa _ { \gamma _ { 1 } } ( \lambda ) } + \frac { 1 - P } { \kappa _ { \gamma _ { 0 } } ( \lambda ) } \Big ) ^ { - 1 } d ( \lambda ) ,
$$

where we use $\kappa _ { \gamma _ { i } } ( \lambda ) = d _ { \operatorname* { m a x } } ( \gamma _ { i } , \lambda ) / d ( \lambda )$

Proof of Theorem 3.6 (Polynomial Approximation Rates). Replacing $\lambda _ { 0 }$ by min $\{ \lambda _ { 0 } , \eta _ { 1 } \}$ preserves Assumption 3.4. Thus, assume without loss of generality that $\lambda _ { 0 } \leq \eta _ { 1 }$

For $0 < \lambda \leq \lambda _ { 0 }$ , since the $\{ \eta _ { j } \}$ are in the descending order, let $J _ { \lambda } = \operatorname* { m a x } \{ j \ge 1 : \eta _ { j } \ge \lambda \}$ Under Assumption 3.5, we have $\eta _ { J _ { \lambda } } \asymp \lambda \asymp J _ { \lambda } ^ { - \alpha }$ , and hence

$$
\begin{array} { l } { { d ( \lambda ) = \displaystyle \sum _ { j \geq 1 } \displaystyle \frac { \eta _ { j } } { \eta _ { j } + \lambda } = \displaystyle \sum _ { 1 \leq j \leq J _ { \lambda } } \displaystyle \frac { \eta _ { j } } { \eta _ { j } + \lambda } + \displaystyle \sum _ { j > J _ { \lambda } } \displaystyle \frac { \eta _ { j } } { \eta _ { j } + \lambda } \lesssim 1 \cdot J _ { \lambda } + \frac 1 \lambda \sum _ { j > J _ { \lambda } } \eta _ { j } } } \\ { { \lesssim J _ { \lambda } + \displaystyle \frac 1 \lambda \sum _ { j > J _ { \lambda } } j ^ { - \alpha } \lesssim J _ { \lambda } + \displaystyle \frac 1 \lambda J _ { \lambda } ^ { 1 - \alpha } \asymp \lambda ^ { - 1 / \alpha } . } } \end{array}\tag{B.15}
$$

The matching lower bound follows from $d ( \lambda ) \geq J _ { \lambda } / 2 \asymp \lambda ^ { - 1 / \alpha }$ . Thus $d ( \lambda ) \asymp \lambda ^ { - 1 / \alpha }$ . For each such �, Assumption 3.4 guarantees a choice of $\gamma _ { 1 }$ for which Lemma 3.3 gives

$$
d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) \leq \Big ( \frac { P } { \kappa _ { \gamma _ { 1 } } ( \lambda ) } + \frac { 1 - P } { \kappa _ { \gamma _ { 0 } } ( \lambda ) } \Big ) ^ { - 1 } d ( \lambda ) \leq \Big ( \frac { P } { \kappa _ { 1 } ^ { * } } + \frac { 1 - P } { \kappa _ { 0 } ^ { * } } \Big ) ^ { - 1 } d ( \lambda ) .
$$

Combining this with (B.15) yields

$$
d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) \lesssim \Big ( \frac { P } { \kappa _ { 1 } ^ { * } } + \frac { 1 - P } { \kappa _ { 0 } ^ { * } } \Big ) ^ { - 1 } \lambda ^ { - 1 / \alpha } ,
$$

where the implicit constant is independent of $P .$ . For the given network width �, a suficient condition for (3.4) is

$$
d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) \leq \frac { 9 m } { 4 0 \log ( 6 3 m / ( 2 0 \delta ) ) } .
$$

Indeed, log $( 6 3 m / ( 2 0 \delta ) ) > 1$ , so this condition implies $d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) \leq 9 m / 4 0$ and hence $\log ( 1 4 d _ { \operatorname* { m a x } } ( \gamma _ { P } , \lambda ) / \delta ) \le \log ( 6 3 m / ( 2 0 \delta ) )$ . The width condition (3.4) holds. It therefore sufices to choose � satisfying

$$
\lambda \gtrsim \Big ( \Big ( \frac { P } { \kappa _ { 1 } ^ { * } } + \frac { 1 - P } { \kappa _ { 0 } ^ { * } } \Big ) \frac { 9 m } { 4 0 \log ( 6 3 m / ( 2 0 \delta ) ) } \Big ) ^ { - \alpha } ,\tag{B.16}
$$

where the implicit constant is chosen suficiently large. Choose � to attain this lower bound, with the implicit constant fixed. If $\lambda \leq \lambda _ { 0 }$ , then (3.5) with $\varepsilon = 4 \lambda$ gives, with probability at least $1 - \delta .$

$$
\operatorname* { s u p } _ { \| f \| _ { \mathcal { H } _ { \rho } \leq 1 } } \operatorname* { i n f } _ { f _ { 2 } ^ { ( P ) } \in \mathcal { F } _ { P } ( W , 2 ) } \| f - f _ { 2 } ^ { ( P ) } \| _ { L _ { 2 } ( d \mu ) } ^ { 2 } \leq 4 \lambda .
$$

If $\lambda > \lambda _ { 0 }$ , the zero approximant instead gives

$$
\operatorname* { i n f } _ { f _ { 2 } ^ { ( P ) } \in \mathcal { F } _ { P } ( W , 2 ) } \lVert f - f _ { 2 } ^ { ( P ) } \rVert _ { L _ { 2 } ( d \mu ) } ^ { 2 } \leq \lVert T _ { \rho } \rVert _ { \mathrm { o p } } \leq \frac { \lVert T _ { \rho } \rVert _ { \mathrm { o p } } } { \lambda _ { 0 } } \lambda ,
$$

where we use

$$
\begin{array} { r } { \Vert f \Vert _ { L _ { 2 } ( d \mu ) } ^ { 2 } = \Vert T _ { \rho } ^ { 1 / 2 } q _ { f } \Vert _ { L _ { 2 } ( d \mu ) } ^ { 2 } = \langle q _ { f } , T _ { \rho } q _ { f } \rangle _ { L _ { 2 } ( d \mu ) } \leq \Vert T _ { \rho } \Vert _ { \mathrm { o p } } \Vert q _ { f } \Vert _ { L _ { 2 } ( d \mu ) } ^ { 2 } \leq \Vert T _ { \rho } \Vert _ { \mathrm { o p } } . } \end{array}
$$

Combining the two cases with (B.16) yields

$$
\operatorname* { s u p } _ { \| f \| _ { \mathcal H _ { \rho } } \leq 1 } \operatorname* { i n f } _ { f _ { 2 } ^ { ( P ) } \in \mathcal F _ { P } ( W , 2 ) } \| f - { f _ { 2 } ^ { ( P ) } } \| _ { L _ { 2 } ( d \mu ) } ^ { 2 } \lesssim \Big ( \Big ( \frac { P } { \kappa _ { 1 } ^ { * } } + \frac { 1 - P } { \kappa _ { 0 } ^ { * } } \Big ) \frac { 9 m } { 4 0 \log ( 6 3 m / ( 2 0 \delta ) ) } \Big ) ^ { - \alpha } .
$$

The implicit constant is independent of �, � and �.

## C. Proofs of Main Results in Section 4 – Specific Spherical Setting

## Outline

• Subsection C.1 establishes the Funk–Hecke formula and harmonic representations of the ridge leverage score and ReLU harmonic coeficients. The Jacobi representation then yields sharp estimates for the degree-wise eigenvalues.

• Subsection C.2 uses the above estimates to prove the kernel spectrum and eigenvalue decay. It then bounds the initialization mismatch, first for the uniform spherical distribution and then for the scaled spherical family. Combining the spectral decay and mismatch bounds gives final explicit spherical approximation rates.

## C.1. Auxiliary Theoretical Preliminaries

Lemma C.1 (The Funk–Hecke Formula). Let $\mu = \mathrm { U n i f } ( \mathbb { S } ^ { d - 1 } )$ be the normalized surface measure on $\mathbb { S } ^ { d - 1 }$ , where $d \ge 2 .$ . Let $\psi : [ - 1 , 1 ] \to \mathbb { R }$ be measurable and satisfy

$$
\int _ { - 1 } ^ { 1 } | \psi ( t ) | ( 1 - t ^ { 2 } ) ^ { ( d - 3 ) / 2 } d t < + \infty .
$$

For every spherical harmonic $Y _ { k , j } \in \mathbb { Y } _ { k }$ and every $x \in \mathbb { S } ^ { d - 1 }$

$$
\int _ { { \mathbb S } ^ { d - 1 } } \psi ( x ^ { \top } y ) Y _ { k , j } ( y ) d \mu ( y ) = \widehat { \psi } _ { k } Y _ { k , j } ( x ) ,
$$

where

$$
\widehat { \psi } _ { k } = \frac { \Gamma ( d / 2 ) } { \sqrt { \pi } \Gamma ( ( d - 1 ) / 2 ) } \int _ { - 1 } ^ { 1 } \psi ( t ) P _ { k } ( t ) ( 1 - t ^ { 2 } ) ^ { ( d - 3 ) / 2 } d t ,
$$

and $P _ { k }$ is the degree-� Legendre polynomial in dimension �, normalized by $P _ { k } ( 1 ) = 1$

Proof ofLemma C.1. This is the Funk–Hecke formula [1, Theorem 2.22], rewritten for the normalized surface measure $d \mu = d S _ { d - 1 } / | \mathbb { S } ^ { d - 1 } |$ , using $\begin{array} { r } { \frac { | \mathbb { S } ^ { d - 2 } | } { | \mathbb { S } ^ { d - 1 } | } = \frac { \Gamma ( { d } / 2 ) } { \sqrt { \pi } \Gamma ( ( d - 1 ) / 2 ) } . } \end{array}$

Lemma C.2 (Harmonic Representation of the Ridge Leverage Score). Let $\theta = ( \omega , b ) =$ $( \sqrt { 1 - b ^ { 2 } } u , b ) \in \mathbb { S } ^ { d }$ , where $u \sim \mathrm { U n i f } ( \mathbb { S } ^ { d - 1 } )$ is independent of �, and � has density

$$
p ( b ) = \frac { \Gamma ( ( d + 1 ) / 2 ) } { \sqrt { \pi } \Gamma ( d / 2 ) } ( 1 - b ^ { 2 } ) ^ { ( d - 2 ) / 2 } , w i t h b \in ( - 1 , 1 ) .\tag{C.1}
$$

Define

$$
\chi _ { k } ( b ) = \frac { \Gamma ( d / 2 ) } { \sqrt { \pi } \Gamma ( ( d - 1 ) / 2 ) } \int _ { - 1 } ^ { 1 } ( \sqrt { 1 - b ^ { 2 } } t + b ) _ { + } P _ { k } ( t ) ( 1 - t ^ { 2 } ) ^ { ( d - 3 ) / 2 } d t .
$$

For $\lambda > 0 ,$ , write $s _ { \lambda } ( \theta ) = \langle \phi ( \cdot ; \theta ) , ( T _ { \rho } + \lambda I ) ^ { - 1 } \phi ( \cdot ; \theta ) \rangle _ { L _ { 2 } ( d \mu ) }$ for the unweighted score. Then, for every $Y _ { k , j } \in \mathbb { Y } _ { k }$ , we have $\langle \phi ( \cdot ; \theta ) , Y _ { k , j } \rangle _ { L _ { 2 } ( d \mu ) } = \chi _ { k } ( b ) Y _ { k , j } ( u )$ . Moreover,

$$
\xi _ { k } = \int _ { - 1 } ^ { 1 } \chi _ { k } ( b ) ^ { 2 } p ( b ) d b , s _ { \lambda } ( \theta ) = \sum _ { k = 0 } ^ { \infty } \frac { N ( d , k ) \chi _ { k } ( b ) ^ { 2 } } { \xi _ { k } + \lambda } , a n d d ( \lambda ) = \sum _ { k = 0 } ^ { \infty } \frac { N ( d , k ) \xi _ { k } } { \xi _ { k } + \lambda } .\tag{C.2}
$$

Proof of Lemma C.2. Rewrite every $\theta = ( \omega , b ) \in \mathbb { S } ^ { d }$ , except on a zero measure set (when $b = \pm 1 )$ , as

$$
\theta = ( \omega , b ) = ( \| \omega \| _ { 2 } \frac { \omega } { \| \omega \| _ { 2 } } , b ) : = ( \sqrt { 1 - b ^ { 2 } } u , b ) ,
$$

where $u = \omega / \| \omega \| _ { 2 } \in \mathbb { S } ^ { d - 1 }$ is independent of � and $u \sim \mathrm { U n i f } ( \mathbb { S } ^ { d - 1 } )$ . By the spherical integral, � has density

$$
p ( b ) = \frac { \Gamma ( ( d + 1 ) / 2 ) } { \sqrt { \pi } \Gamma ( d / 2 ) } ( 1 - b ^ { 2 } ) ^ { ( d - 2 ) / 2 } , \mathrm { ~ w i t h ~ } b \in ( - 1 , 1 ) .
$$

And we can write $\phi ( x ; \theta ) = \sigma _ { R } ( \omega ^ { \top } x + b ) = \sigma _ { R } ( \sqrt { 1 - b ^ { 2 } } u ^ { \top } x + b )$ . Applying the Funk-Hecke formula of Lemma C.1 to the feature map �, we have for any $Y _ { k , j } \in \mathbb { Y } _ { k }$

$$
\langle \phi ( \cdot ; \theta ) , Y _ { k , j } ( \cdot ) \rangle _ { L _ { 2 } ( d \mu ) } = \int _ { \cal X } \sigma _ { R } ( \sqrt { 1 - b ^ { 2 } } u ^ { \top } x + b ) Y _ { k , j } ( x ) d \mu ( x ) = \chi _ { k } ( b ) Y _ { k , j } ( u ) ,\tag{C.3}
$$

with

$$
\chi _ { k } ( b ) = \frac { \Gamma ( d / 2 ) } { \sqrt { \pi } \Gamma ( ( d - 1 ) / 2 ) } \int _ { - 1 } ^ { 1 } \sigma _ { R } ( \sqrt { 1 - b ^ { 2 } } t + b ) P _ { k } ( t ) ( 1 - t ^ { 2 } ) ^ { ( d - 3 ) / 2 } d t .
$$

The independence of � and � and orthonormality give

$$
\begin{array} { l } { { \langle Y _ { \ell , i } , T _ { \rho } Y _ { k , j } \rangle = \displaystyle \int _ { \Theta } \langle Y _ { \ell , i } , \phi ( \cdot ; \theta ) \rangle \langle \phi ( \cdot ; \theta ) , Y _ { k , j } \rangle d \rho ( \theta ) } } \\ { { \displaystyle \qquad = \int _ { - 1 } ^ { 1 } \chi _ { \ell } ( b ) \chi _ { k } ( b ) p ( b ) d b \int _ { \mathbb { S } ^ { d - 1 } } Y _ { \ell , i } ( u ) Y _ { k , j } ( u ) d \mu ( u ) . } } \end{array}
$$

The second integral is zero unless $( \ell , i ) = ( k , j )$ , when it equals one. Completeness therefore gives $T _ { \rho } Y _ { k , j } = \xi _ { k } Y _ { k , j }$ with $\begin{array} { r } { \xi _ { k } = \int _ { - 1 } ^ { 1 } \chi _ { k } ( b ) ^ { 2 } p ( b ) d b } \end{array}$

Further, by the addition formula $\begin{array} { r } { \sum _ { j = 1 } ^ { N ( d , k ) } Y _ { k , j } ( u ) ^ { 2 } = N ( d , k ) } \end{array}$ , the corresponding spectral decomposition of $( T _ { \rho } + \lambda I ) ^ { - 1 }$ similar to item 2 of Proposition 4.1, and (C.3), we obtain

$$
\begin{array} { l } { { s _ { \lambda } ( \theta ) = \langle \phi ( \cdot ; \theta ) , ( T _ { \rho } + \lambda I ) ^ { - 1 } \phi ( \cdot ; \theta ) \rangle _ { L _ { 2 } ( d \mu ) } } } \\ { { \ = \langle \phi ( \cdot ; \theta ) , \displaystyle \sum _ { k = 0 } ^ { \infty } \ \sum _ { j = 1 } ^ { N ( d , k ) } \ \frac { 1 } { \xi _ { k } + \lambda } Y _ { k , j } \otimes Y _ { k , j } \phi ( \cdot ; \theta ) \rangle _ { L _ { 2 } ( d \mu ) } } } \\ { { \ = \displaystyle \sum _ { k = 0 } ^ { \infty } \ \sum _ { j = 1 } ^ { N ( d , k ) } \ \frac { | \langle \phi ( \cdot ; \theta ) , Y _ { k , j } ( \cdot ) \rangle _ { L _ { 2 } ( d \mu ) } | ^ { 2 } } { \xi _ { k } + \lambda } } } \\ { { \ = \displaystyle \sum _ { k = 0 } ^ { \infty } \ \sum _ { j = 1 } ^ { N ( d , k ) } \ \frac { \left( \chi _ { k } ( b ) Y _ { k , j } ( u ) \right) ^ { 2 } } { \xi _ { k } + \lambda } = \displaystyle \sum _ { k = 0 } ^ { \infty } \ \frac { N ( d , k ) \chi _ { k } ( b ) ^ { 2 } } { \xi _ { k } + \lambda } . } } \end{array}\tag{C.4}
$$

With $d ( \lambda ) = \mathbb { E } _ { \theta \sim \rho } [ s _ { \lambda } ( \theta ) ]$ , we also obtain $\begin{array} { r } { d ( \lambda ) = \sum _ { k = 0 } ^ { \infty } \frac { N ( d , k ) \xi _ { k } } { \xi _ { k } + \lambda } } \end{array}$

Lemma C.3 (Jacobi Representation of the ReLU Harmonic Coeficients). Let $\begin{array} { r } { c _ { d } = \frac { d - 3 } { 2 } } \end{array}$ $G _ { d } ( b ) = ( 1 - 2 b ^ { 2 } ) _ { + } ^ { d / 2 } ( 1 - b ^ { 2 } ) ^ { ( 2 - d ) / 2 } f o r | b | < 1 , s e t G _ { d } ( \pm 1 ) = 0$ and let $\begin{array} { r } { C _ { 1 , d } = \frac { 2 ^ { ( d - 2 ) / 2 } \Gamma ( \bar { d } / 2 ) } { \pi } } \end{array}$ Then thefollowing statements holdfor $k \geq 2 .$

(1) $I f | b | \geq 1 / { \sqrt { 2 } } ,$ , then $\chi _ { k } ( b ) = 0$ . While $i f \left| b \right| < 1 / { \sqrt { 2 } } ,$

$$
\chi _ { k } ( b ) = \frac { \Gamma ( d / 2 ) \sqrt { 1 - b ^ { 2 } } } { 4 \sqrt { \pi } \Gamma ( ( d - 1 ) / 2 ) k ( k - 1 ) J _ { k } ^ { ( c _ { d } , c _ { d } ) } ( 1 ) } \Big ( \frac { 1 - 2 b ^ { 2 } } { 1 - b ^ { 2 } } \Big ) ^ { c _ { d } + 2 } J _ { k - 2 } ^ { ( c _ { d } + 2 , c _ { d } + 2 ) } \Big ( - \frac { b } { \sqrt { 1 - b ^ { 2 } } } \Big ) .
$$

(2) There exists a constant $C _ { d } > 0$ such that

$$
\chi _ { k } ( b ) ^ { 2 } \leq C _ { d } G _ { d } ( b ) k ^ { - ( d + 2 ) } , \qquad b \in [ - 1 , 1 ] .
$$

(3) For every compact interval $I \subset ( - 1 / { \sqrt { 2 } } , 1 / { \sqrt { 2 } } )$

$$
\chi _ { k } ( b ) = C _ { 1 , d } \sqrt { G _ { d } ( b ) } k ^ { - ( d + 2 ) / 2 } [ \cos \Psi _ { k } ( b ) + O ( k ^ { - 1 } ) ]\tag{C.5}
$$

uniformlyfor $b \in I ,$ , where

$$
\Psi _ { k } ( b ) = \Big ( k + \frac { d - 2 } { 2 } \Big ) \operatorname { a r c c o s } \Big ( - \frac { b } { \sqrt { 1 - b ^ { 2 } } } \Big ) - \frac { \pi ( d + 2 ) } { 4 } .
$$

Proof of Lemma C.3. We prove the statements item by item.

For item 1: Let $J _ { k } ^ { ( \alpha , \beta ) }$ denote the standard Jacobi polynomial. Since the Legendre polynomial $P _ { k }$ is normalized by $P _ { k } ( 1 ) = 1$ and, by [1, Section 2.6.1], $P _ { k }$ is proportional to the Jacobi polynomial $J _ { k } ^ { ( ( d - 3 ) ) ^ { } / 2 , ( \ddot { d } - 3 ) ^ { } / 2 ) } ( t )$ $P _ { k }$ can be written as

$$
P _ { k } ( t ) = \frac { J _ { k } ^ { ( ( d - 3 ) / 2 , ( d - 3 ) / 2 ) } ( t ) } { J _ { k } ^ { ( ( d - 3 ) / 2 , ( d - 3 ) / 2 ) } ( 1 ) } .
$$

If $b \leq - { \frac { 1 } { \sqrt { 2 } } }$ , the feature $\sigma _ { R } ( \sqrt { 1 - b ^ { 2 } } u ^ { \top } x + b )$ vanishes. And if $\begin{array} { r } { b \ge \frac { 1 } { \sqrt { 2 } } } \end{array}$

$$
\sigma _ { R } ( \sqrt { 1 - b ^ { 2 } } u ^ { \top } x + b ) = b + \sqrt { 1 - b ^ { 2 } } u ^ { \top } x \in \mathbb { Y } _ { 0 } \oplus \mathbb { Y } _ { 1 } .
$$

By the orthogonality of spherical harmonics of diferent degrees and $( \mathrm { C } . 3 ) , \chi _ { k } ( b ) = 0$ for every $k \geq 2$ . Thus when $k \geq 2$ and $\begin{array} { r } { | b | \geq \frac { 1 } { \sqrt { 2 } } } \end{array}$ , the $\chi _ { k } ( b ) = 0$

For $\begin{array} { r } { | b | < \frac { 1 } { \sqrt { 2 } } } \end{array}$ , by the Rodrigues formula for Jacobi Polynomials, let $c _ { d } = ( d - 3 ) / 2$

$$
( 1 - t ^ { 2 } ) ^ { c _ { d } } J _ { k } ^ { ( c _ { d } , c _ { d } ) } ( t ) = \frac { ( - 1 ) ^ { k } } { 2 ^ { k } k ! } \frac { d ^ { k } } { d t ^ { k } } [ ( 1 - t ^ { 2 } ) ^ { c _ { d } + k } ]
$$

after two integrations by parts, we have for $k \geq 2$

$$
\begin{array} { r l } & { \chi _ { k } ( b ) } \\ & { = C _ { \Gamma } \cdot \frac { \sqrt { 1 - b ^ { 2 } } } { J _ { k } ^ { ( c _ { d } , c _ { d } ) } ( 1 ) } \cdot \frac { ( - 1 ) ^ { k } } { 2 ^ { k } k ! } \int _ { - \frac { b } { \sqrt { 1 - b ^ { 2 } } } } ^ { 1 } \bigg ( t + \frac { b } { \sqrt { 1 - b ^ { 2 } } } \bigg ) \frac { d ^ { k } } { d t ^ { k } } [ ( 1 - t ^ { 2 } ) ^ { c _ { d } + k } ] d t } \\ & { = \frac { C _ { \Gamma } \sqrt { 1 - b ^ { 2 } } } { 4 k ( k - 1 ) J _ { k } ^ { ( c _ { d } , c _ { d } ) } ( 1 ) } \cdot \frac { ( - 1 ) ^ { k - 2 } } { 2 ^ { k - 2 } ( k - 2 ) ! } \cdot \frac { d ^ { k - 2 } } { d t ^ { k - 2 } } [ ( 1 - t ^ { 2 } ) ^ { c _ { d } + k } ] \Bigg | _ { t = - \frac { b } { \sqrt { 1 - b ^ { 2 } } } } } \\ & { = \frac { C _ { \Gamma } \sqrt { 1 - b ^ { 2 } } } { 4 k ( k - 1 ) J _ { k } ^ { ( c _ { d } , c _ { d } ) } ( 1 ) } \bigg ( \frac { 1 - 2 b ^ { 2 } } { 1 - b ^ { 2 } } \bigg ) ^ { c _ { d } + 2 } J _ { k - 2 } ^ { ( c _ { d } + 2 , c _ { d } + 2 ) } \bigg ( - \frac { b } { \sqrt { 1 - b ^ { 2 } } } \bigg ) , } \end{array}\tag{C.6}
$$

where $\begin{array} { r } { C _ { \Gamma } = \frac { \Gamma ( d / 2 ) } { \sqrt { \pi } \Gamma ( ( d - 1 ) / 2 ) } } \end{array}$

For item 2: Based on [41, Eq.(2.3.41)], we have for $t \in ( - 1 , 1 )$ ,

$$
\left| \frac { J _ { k } ^ { ( c _ { d } , c _ { d } ) } ( t ) } { \| J _ { k } ^ { ( c _ { d } , c _ { d } ) } \| } \right| \leq C _ { d } \Bigl ( \sqrt { 1 - t } + \frac { 1 } { k } \Bigr ) ^ { - c _ { d } - 1 / 2 } \Bigl ( \sqrt { 1 + t } + \frac { 1 } { k } \Bigr ) ^ { - c _ { d } - 1 / 2 } \leq C _ { d } ( 1 - t ^ { 2 } ) ^ { - c _ { d } / 2 - 1 / 4 } .
$$

Since $\| J _ { k } ^ { ( c _ { d } , c _ { d } ) } \| = O ( k ^ { - 1 / 2 } )$ with the norm taken in $L ^ { 2 } ( [ - 1 , 1 ] , ( 1 - t ^ { 2 } ) ^ { c _ { d } } d t )$ , we have

$$
| J _ { k } ^ { ( c _ { d } , c _ { d } ) } ( t ) | \leq C _ { d } k ^ { - 1 / 2 } ( 1 - t ^ { 2 } ) ^ { - c _ { d } / 2 - 1 / 4 } \mathrm { f o r } t \in ( - 1 , 1 ) .
$$

Therefore, we use

$$
| J _ { k - 2 } ^ { ( c _ { d } + 2 , c _ { d } + 2 ) } ( t ) | \leq C _ { d } ( k - 2 ) ^ { - 1 / 2 } ( 1 - t ^ { 2 } ) ^ { - ( c _ { d } + 2 ) / 2 - 1 / 4 } , k \geq 3 .\tag{C.7}
$$

Moreover, for $J _ { k } ^ { ( c _ { d } , c _ { d } ) } ( 1 )$ , by the definition, we derive

$$
J _ { k } ^ { ( c _ { d } , c _ { d } ) } ( 1 ) = \frac { \Gamma ( c _ { d } + k + 1 ) } { \Gamma ( k + 1 ) \Gamma ( c _ { d } + 1 ) } \sim \frac { 1 } { \Gamma ( c _ { d } + 1 ) } k ^ { c _ { d } } \asymp k ^ { ( d - 3 ) / 2 } .\tag{C.8}
$$

Combining (C.6), (C.7) and (C.8), we obtain, for $k \geq 3$

$$
\begin{array} { l } { \displaystyle | \chi _ { k } ( b ) | } \\ { \le \frac { \Gamma ( d / 2 ) \sqrt { 1 - b ^ { 2 } } } { \sqrt { \pi } \Gamma ( ( d - 1 ) / 2 ) 4 k ( k - 1 ) } \Big ( \frac { 1 - 2 b ^ { 2 } } { 1 - b ^ { 2 } } \Big ) ^ { c _ { d } + 2 } \cdot C _ { d } k ^ { - c _ { d } } ( k - 2 ) ^ { - 1 / 2 } \Big ( \frac { 1 - 2 b ^ { 2 } } { 1 - b ^ { 2 } } \Big ) ^ { - ( c _ { d } + 2 ) / 2 - 1 / 4 } } \\ { \le C _ { d } ( 1 - b ^ { 2 } ) ^ { ( 2 - d ) / 4 } ( 1 - 2 b ^ { 2 } ) ^ { d / 4 } k ^ { - ( d + 2 ) / 2 } . } \end{array}
$$

When $k = 2 , J _ { k - 2 } ^ { ( c _ { d } + 2 , c _ { d } + 2 ) }$ is constant and can be included by enlarging the constant $C _ { d }$ Consequently, for $\begin{array} { r } { | b | < \frac { 1 } { \sqrt { \gamma } } , | \chi _ { k } ( b ) | ^ { 2 } \leq C _ { d } ( 1 - b ^ { 2 } ) ^ { ( 2 - d ) / 2 } ( 1 - 2 b ^ { 2 } ) ^ { d / 2 } k ^ { - ( d + 2 ) } } \end{array}$ . And for $\begin{array} { r } { | b | \geq \frac { 1 } { \sqrt { 2 } } , | \chi _ { k } ( b ) | = 0 } \end{array}$ . We can write (Recall that the final aim is to bound $\kappa _ { \gamma _ { 0 } } ( \lambda )$ via the score $s _ { \lambda } ( \theta )$ in (C.4))

$$
| \chi _ { k } ( b ) | ^ { 2 } \leq C _ { d } G _ { d } ( b ) k ^ { - ( d + 2 ) } \mathrm { ~ f o r ~ } b \in [ - 1 , 1 ] .
$$

This global bound will be used to control endpoint regions where the Darboux expansion below is not uniform.

For item 3: For $\left| b \right| < { \frac { 1 } { \sqrt { 2 } } }$ , let $z _ { b } = \operatorname { a r c c o s } ( - \frac { b } { \sqrt { 1 - b ^ { 2 } } } )$ . Applying the classical Darboux formula for Jacobi polynomials [70, Theorem 8.21.8] to $J _ { k - 2 } ^ { ( c _ { d } + 2 , c _ { d } + 2 ) } ( \cos z _ { b } )$ with $c _ { d } = ( d - 3 ) / 2$ we have

$$
\begin{array} { l } { { J _ { k - 2 } ^ { ( ( d + 1 ) / 2 , ( d + 1 ) / 2 ) } ( \cos z _ { b } ) } } \\ { { = \displaystyle \frac { 1 } { \pi ^ { 1 / 2 } ( k - 2 ) ^ { 1 / 2 } } \Big ( \sin \displaystyle \frac { z _ { b } } { 2 } \cos \displaystyle \frac { z _ { b } } { 2 } \Big ) ^ { - ( d + 2 ) / 2 } \Big [ \cos \Big ( \displaystyle \frac { d - 2 + 2 k } { 2 } z _ { b } - \displaystyle \frac { \pi ( d + 2 ) } { 4 } \Big ) + { \cal O } ( k ^ { - 1 } ) \Big ] } } \\ { { = \displaystyle \frac { 2 ^ { ( d + 2 ) / 2 } } { \pi ^ { 1 / 2 } ( k - 2 ) ^ { 1 / 2 } } \Big ( \displaystyle \frac { 1 - 2 b ^ { 2 } } { 1 - b ^ { 2 } } \Big ) ^ { - ( d + 2 ) / 4 } \Big [ \cos \Big ( \displaystyle \frac { d - 2 + 2 k } { 2 } z _ { b } - \displaystyle \frac { \pi ( d + 2 ) } { 4 } \Big ) + { \cal O } ( k ^ { - 1 } ) \Big ] , } } \end{array}
$$

where we have used

$$
\sin z _ { b } = \sqrt { \frac { 1 - 2 b ^ { 2 } } { 1 - b ^ { 2 } } } , \sin \frac { z _ { b } } { 2 } \cos \frac { z _ { b } } { 2 } = \frac { 1 } { 2 } \sin z _ { b } = \frac { 1 } { 2 } \sqrt { \frac { 1 - 2 b ^ { 2 } } { 1 - b ^ { 2 } } } .
$$

The remainder $O ( k ^ { - 1 } )$ is uniform for � in every compact subset of $( - 1 / \sqrt { 2 } , 1 / \sqrt { 2 } )$ . Further combining above estimate of $J _ { k - 2 } ^ { ( ( d + 1 ) / 2 , ( d + 1 ) / 2 \bar { ) } } ( \cos { z _ { b } } )$ , (C.6) and

$$
J _ { k } ^ { ( c _ { d } , c _ { d } ) } ( 1 ) = \frac { \Gamma ( k + c _ { d } + 1 ) } { \Gamma ( k + 1 ) \Gamma ( c _ { d } + 1 ) } = \frac { k ^ { c _ { d } } } { \Gamma ( c _ { d } + 1 ) } ( 1 + { \cal O } ( k ^ { - 1 } ) ) ,
$$

we obtain

$$
\begin{array} { l } { \displaystyle \chi _ { k } ( b ) = \frac { \Gamma ( d / 2 ) \sqrt { 1 - b ^ { 2 } } } { \sqrt { \pi } \Gamma ( ( d - 1 ) / 2 ) 4 k ( k - 1 ) J _ { k } ^ { ( c _ { d } , c _ { d } ) } ( 1 ) } \Big ( \frac { 1 - 2 b ^ { 2 } } { 1 - b ^ { 2 } } \Big ) ^ { c _ { d } + 2 } J _ { k - 2 } ^ { ( c _ { d } + 2 , c _ { d } + 2 ) } \Big ( - \frac { b } { \sqrt { 1 - b ^ { 2 } } } \Big ) } \\ { \displaystyle \qquad = \frac { \Gamma ( d / 2 ) \sqrt { 1 - b ^ { 2 } } \Gamma ( c _ { d } + 1 ) ( 1 + O ( k ^ { - 1 } ) ) } { \sqrt { \pi } \Gamma ( ( d - 1 ) / 2 ) 4 k ( k - 1 ) k ^ { c _ { d } } } \Big ( \frac { 1 - 2 b ^ { 2 } } { 1 - b ^ { 2 } } \Big ) ^ { c _ { d } + 2 } J _ { k - 2 } ^ { ( c _ { d } + 2 , c _ { d } + 2 ) } \Big ( - \frac { b } { \sqrt { 1 - b ^ { 2 } } } \Big ) } \\ { \displaystyle \qquad = C _ { 1 , d } \sqrt { G _ { d } ( b ) } k ^ { - ( d + 2 ) / 2 } \Big [ \cos \Big ( \frac { d - 2 + 2 k } { 2 } z _ { b } - \frac { \pi ( d + 2 ) } { 4 } \Big ) + O ( k ^ { - 1 } ) \Big ] , } \end{array}
$$

where $\begin{array} { r } { C _ { 1 , d } = \frac { 2 ^ { ( d - 2 ) / 2 } \Gamma ( d / 2 ) } { \pi } } \end{array}$ and $G _ { d } ( b ) = ( 1 - 2 b ^ { 2 } ) _ { + } ^ { d / 2 } ( 1 - b ^ { 2 } ) ^ { ( 2 - d ) / 2 }$ . The asymptotic relation in above formula is again uniform for � in compact subsets of $( - 1 / \sqrt { 2 } , 1 / \sqrt { 2 } )$ . Now we finish the proof. ■

Corollary C.4 (Sharp Spectral Estimation of �). Under the spherical biased-ReLU setting,

$$
\xi _ { k } = \frac { C _ { 1 , d } ^ { 2 } d } { 2 \sqrt { 2 } ( d + 1 ) } k ^ { - ( d + 2 ) } ( 1 + o ( 1 ) ) ,
$$

Proof of Corollary C.4. By Lemma C.2 and the fact that $\chi _ { k } ( b ) = 0$ for $| b | \geq 1 / { \sqrt { 2 } }$ , we have

$$
k ^ { d + 2 } \xi _ { k } = \int _ { - 1 / \sqrt { 2 } } ^ { 1 / \sqrt { 2 } } k ^ { d + 2 } \chi _ { k } ( b ) ^ { 2 } p ( b ) d b .
$$

To estimate this integral, let $z _ { b } = \operatorname { a r c c o s } ( - b / \sqrt { 1 - b ^ { 2 } } )$ . Squaring (C.5) gives

$$
k ^ { d + 2 } \chi _ { k } ( b ) ^ { 2 } = C _ { 1 , d } ^ { 2 } G _ { d } ( b ) \Big [ \cos ^ { 2 } \Big ( ( k + \frac { d - 2 } { 2 } ) z _ { b } - \frac { \pi ( d + 2 ) } { 4 } \Big ) + { \cal O } ( k ^ { - 1 } ) \Big ] ,\tag{C.9}
$$

uniformly for � in compact subsets of $( - 1 / { \sqrt { 2 } } , 1 / { \sqrt { 2 } } )$ . For each fixed $\vert b \vert < 1 / { \sqrt { 2 } } .$ , the remainder $O ( k ^ { - 1 } )$ in (C.9) tends to zero as $k \to \infty$ . Using $\cos ^ { 2 } t = ( 1 + \cos ( 2 t ) ) / 2$ , we rewrite (C.9) as

$$
\begin{array} { l } { { \displaystyle { k ^ { d + 2 } \chi _ { k } ( b ) ^ { 2 } = \frac { C _ { 1 , d } ^ { 2 } } { 2 } G _ { d } ( b ) + \frac { C _ { 1 , d } ^ { 2 } } { 2 } G _ { d } ( b ) \cos \Big ( ( 2 k + d - 2 ) z _ { b } - \frac { \pi ( d + 2 ) } { 2 } \Big ) } } } \\ { { \displaystyle { ~ + C _ { 1 , d } ^ { 2 } G _ { d } ( b ) O ( k ^ { - 1 } ) . } } } \end{array}
$$

By the global bound in item 2 of Lemma C.3 and dominated convergence, the last term contributes $o ( 1 )$ after integration. Hence,

$$
\begin{array} { r l r } {  { k ^ { d + 2 } \xi _ { k } = \frac { C _ { 1 , d } ^ { 2 } } { 2 } \int _ { - 1 / \sqrt { 2 } } ^ { 1 / \sqrt { 2 } } G _ { d } ( b ) p ( b ) d b } } \\ & { } & { \quad + \frac { C _ { 1 , d } ^ { 2 } } { 2 } \int _ { - 1 / \sqrt { 2 } } ^ { 1 / \sqrt { 2 } } G _ { d } ( b ) p ( b ) \cos { \Big ( ( 2 k + d - 2 ) z _ { b } - \frac { \pi ( d + 2 ) } { 2 } \Big ) } d b + o ( 1 ) . } \end{array}
$$

The first integral can be evaluated directly. Using the density in (C.1), a beta-integral calculation gives

$$
\int _ { - 1 / \sqrt { 2 } } ^ { 1 / \sqrt { 2 } } G _ { d } ( b ) p ( b ) d b = \frac { \Gamma ( ( d + 1 ) / 2 ) } { \sqrt { \pi } \Gamma ( d / 2 ) } \int _ { - 1 / \sqrt { 2 } } ^ { 1 / \sqrt { 2 } } ( 1 - 2 b ^ { 2 } ) ^ { d / 2 } d b = \frac { d } { \sqrt { 2 } ( d + 1 ) } .
$$

For the second integral, the change of variables $z = z _ { b }$ , with $\begin{array} { r } { b = - \frac { \cos z } { \sqrt { 1 + \cos ^ { 2 } z } } } \end{array}$ , gives

$$
\begin{array} { l } { { \displaystyle \int _ { - 1 / \sqrt 2 } ^ { 1 / \sqrt 2 } G _ { d } ( b ) p ( b ) \cos \Big ( ( 2 k + d - 2 ) z _ { b } - \frac { \pi ( d + 2 ) } 2 \Big ) d b } } \\ { { \displaystyle = \int _ { 0 } ^ { \pi } \widetilde { G } _ { d } ( z ) \cos \Big ( ( 2 k + d - 2 ) z - \frac { \pi ( d + 2 ) } 2 \Big ) d z , } } \end{array}
$$

where

$$
\widetilde { G } _ { d } ( z ) = \frac { \Gamma ( ( d + 1 ) / 2 ) } { \sqrt { \pi } \Gamma ( d / 2 ) } \frac { \sin ^ { d + 1 } z } { ( 1 + \cos ^ { 2 } z ) ^ { ( d + 3 ) / 2 } } \in L _ { 1 } ( 0 , \pi ) .
$$

The Riemann–Lebesgue lemma therefore implies that this integral tends to zero as $k \to \infty$

Combining these estimates yields

$$
\xi _ { k } = \frac { C _ { 1 , d } ^ { 2 } d } { 2 \sqrt { 2 } ( d + 1 ) } k ^ { - ( d + 2 ) } ( 1 + o ( 1 ) ) ,
$$

which proves the result. In particular, $\xi _ { k } \asymp k ^ { - ( d + 2 ) }$ for large �.

## C.2. Proofs of Main Results in Section 4

## Proof ofProposition 4.1 (Spectrum and Eigenvalue Decay ofReLU Kernel).

Proof Overview. Rotation invariance and the Funk–Hecke formula identify the spherical harmonics as eigenfunctions of $T _ { \rho }$ , while the density of biased-ReLU networks establishes the strict positivity of the eigenvalues (item 1). Completeness of the spherical harmonics then gives the operator decomposition, and Mercer’s theorem yields the corresponding kernel expansion (item 2). Finally, the degree-wise eigenvalue decay follows from Corollary C.4. Combining this decay with the spherical harmonic multiplicities gives the decay rate of the eigenvalues ordered with multiplicity (item 3).

Proof Details. We prove the proposition item by item.

For item 1: The rotation invariance of $k _ { \rho }$ implies that it depends on � and $x ^ { \prime }$ only through their inner product. Write $h : [ - 1 , 1 ] \to$ R such that

$$
k _ { \rho } ( x , x ^ { \prime } ) = h \Big ( \frac { 1 + x ^ { \top } x ^ { \prime } } { 2 } \Big ) .
$$

For any $Y _ { k , j } \in \mathbb { Y } _ { k }$ , the Funk-Hecke formula in Lemma C.1 gives

$$
T _ { \rho } Y _ { k , j } ( x ) = \int _ { \mathbb S ^ { d - 1 } } h \Big ( \frac { 1 + x ^ { \top } y } { 2 } \Big ) Y _ { k , j } ( y ) d \mu ( y ) = \xi _ { k } Y _ { k , j } ( x ) ,
$$

where

$$
\xi _ { k } = \frac { \Gamma ( d / 2 ) } { \sqrt { \pi } \Gamma ( ( d - 1 ) / 2 ) } \int _ { - 1 } ^ { 1 } h \Big ( \frac { 1 + t } { 2 } \Big ) P _ { k } ( t ) ( 1 - t ^ { 2 } ) ^ { ( d - 3 ) / 2 } d t .
$$

Hence every element of $\{ Y _ { k , j } \} _ { j = 1 } ^ { N ( d , k ) }$ is an eigenfunction of $T _ { \rho }$ with eigenvalue $\xi _ { k }$

We next establish the strict positivity of $\xi _ { k }$ . Since $T _ { \rho } Y _ { k , j } = \xi _ { k } Y _ { k , j } , \| Y _ { k , j } \| _ { L _ { 2 } ( d \mu ) } = 1$ and $T _ { \rho } = S _ { \rho } S _ { \rho } ^ { * } .$ , we can also write

$$
\xi _ { k } = \langle Y _ { k , j } , T _ { \rho } Y _ { k , j } \rangle _ { L _ { 2 } ( d \mu ) } = \langle S _ { \rho } ^ { * } Y _ { k , j } , S _ { \rho } ^ { * } Y _ { k , j } \rangle _ { L _ { 2 } ( d \rho ) } = \int _ { \Theta } | \langle ( \theta ^ { \top } \rangle _ { + } , Y _ { k , j } \rangle _ { L _ { 2 } ( d \mu ) } | ^ { 2 } d \rho ( \theta ) \geq 0 .
$$

Suppose $\xi _ { k } = 0$ , then we have

$$
\langle ( \theta ^ { \top } \widetilde { \cdot } ) _ { + } , Y _ { k , j } \rangle _ { L _ { 2 } ( d \mu ) } = 0 , { \mathrm { ~ f o r ~ } } \rho { \mathrm { - a l m o s t ~ e v e r y ~ } } \theta \in \Theta .\tag{C.10}
$$

The map $\theta \mapsto \langle ( \theta ^ { \top } \tilde { \cdot } ) _ { + } , Y _ { k , j } \rangle _ { L _ { 2 } ( d \mu ) }$ is continuous on ${ \mathbb S } ^ { d }$ . Since $\rho$ has full support on ${ \mathbb S } ^ { d }$ this identity holds for every $\theta \in \Theta$

By the universal approximation theorem for non-polynomial activation function [29], with the homogeneity of ReLU, the two-layer network class $\mathcal { F } = \cup _ { W \geq d } \mathcal { F } ( W , 2 )$ with parameters restricted to ${ \mathbb S } ^ { d }$ is dense in $C ( \mathbb { S } ^ { d - 1 } )$ . Consequently, is dense in $L _ { 2 } ( \mathbb { S } ^ { d - 1 } , d \mu )$ since $C ( \mathbb { S } ^ { d - 1 } )$ is dense in $L _ { 2 } ( \mathbb { S } ^ { d - 1 } , d \mu )$ . Further with $\left( \mathrm { C } . 1 0 \right)$ , this means $Y _ { k , j }$ is orthogonal to $\mathcal { F }$ which is dense in $L _ { 2 } ( \mathbb { S } ^ { d - 1 } , d \mu )$ . And hence $Y _ { k , j } = 0$ in $L _ { 2 } ( \mathbb { S } ^ { d - 1 } , d \mu )$ , contradicting $\| Y _ { k , j } \| _ { L _ { 2 } ( d \mu ) } = 1$ . Therefore, $\xi _ { k } > 0$ for every $k \geq 0$

It remains to determine the explicit form of ℎ from the definition of $k _ { \rho }$ . Denote $\begin{array} { r } { u = \frac { \tilde { x } } { \sqrt { 2 } } } \end{array}$ $\begin{array} { r } { v = \frac { \tilde { x } ^ { \prime } } { \sqrt { 2 } } } \end{array}$ and $u ^ { \top } v = t = \cos \varphi _ { 1 }$ where $\varphi _ { 1 } \in [ 0 , \pi ]$ . Since the $\rho$ is rotation-invariant, we can let

$$
u = \mathbf { e } _ { 1 } , \ \mathrm { a n d } \ v = \cos \varphi _ { 1 } \mathbf { e } _ { 1 } + \sin \varphi _ { 1 } \mathbf { e } _ { 2 } .
$$

Projecting any $\boldsymbol { \theta } \in \mathbb { S } ^ { d }$ into the span $\{ u , v \}$ , we write $\left( \theta _ { 1 } , \theta _ { 2 } \right) = r \left( \cos \varphi _ { 2 } , \sin \varphi _ { 2 } \right)$ with $\varphi _ { 2 } \in$ 0, 2� . Thus

$$
\begin{array} { r l } & { k _ { \rho } ( x , x ^ { \prime } ) } \\ & { = \mathbb { E } _ { \theta \sim \rho } [ \phi ( x ; \theta ) \phi ( x ^ { \prime } ; \theta ) ] = 2 \mathbb { E } _ { \theta \sim \rho } [ ( \theta ^ { \top } u ) _ { + } ( \theta ^ { \top } v ) _ { + } ] } \\ & { = 2 \mathbb { E } _ { \theta \sim \rho } [ r ^ { 2 } ( \cos \varphi _ { 2 } ) _ { + } ( \cos ( \varphi _ { 2 } - \varphi _ { 1 } ) ) _ { + } ] = \displaystyle \frac { 2 } { \pi ( d + 1 ) } \int _ { \varphi _ { 1 } - \frac { \pi } { 2 } } ^ { \frac { \pi } { 2 } } \cos \varphi _ { 2 } \cos ( \varphi _ { 2 } - \varphi _ { 1 } ) d \varphi _ { 2 } } \\ & { = \displaystyle \frac { 1 } { \pi ( d + 1 ) } [ \sin \varphi _ { 1 } + ( \pi - \varphi _ { 1 } ) \cos \varphi _ { 1 } ] = \displaystyle \frac { 1 } { \pi ( d + 1 ) } [ \sqrt { 1 - t ^ { 2 } } + ( \pi - \operatorname { a r c c o s } t ) t ] , } \end{array}
$$

where we use $\textstyle \mathbb { E } [ r ^ { 2 } ] = { \frac { 2 } { d + 1 } }$ and $t = \cos \varphi _ { 1 }$ . Further with $\begin{array} { r } { t = \frac { 1 + x ^ { \top } x ^ { \prime } } { 2 } } \end{array}$ , we obtain the stated ℎ. For item 2: Since $\{ Y _ { k , j } \} _ { k , j }$ is a complete orthonormal basis in $L _ { 2 } ( \mathbb { S } ^ { d - 1 } , d \mu )$ and the spherical harmonic spaces form an orthogonal decomposition [1, Theorem 2.38], we have $L _ { 2 } ( \mathbb { S } ^ { d - 1 } , d \mu ) = \bigoplus _ { k = 0 } ^ { \infty } \mathbb { Y } _ { k }$ , where the direct sum is in the $L _ { 2 }$ space sense. For any $f \in$ $L _ { 2 } ( \mathbb { S } ^ { d - 1 } , d \mu )$ , the following series converge in $L _ { 2 } ( d \mu )$

$$
f ( x ) = \sum _ { k = 0 } ^ { + \infty } \sum _ { j = 1 } ^ { N ( d , k ) } \langle f , Y _ { k , j } \rangle _ { L _ { 2 } ( d \mu ) } Y _ { k , j } ( x ) \mathrm { ~ a n d ~ } T _ { \rho } f ( x ) = \sum _ { k = 0 } ^ { + \infty } \xi _ { k } \sum _ { j = 1 } ^ { N ( d , k ) } \langle f , Y _ { k , j } \rangle _ { L _ { 2 } ( d \mu ) } Y _ { k , j } ( x ) .
$$

This indicates that $\begin{array} { r } { T _ { \rho } = \sum _ { k = 0 } ^ { + \infty } \xi _ { k } \sum _ { j = 1 } ^ { N ( d , k ) } Y _ { k , j } \otimes Y _ { k , j } } \end{array}$ . Moreover, since $k _ { \rho }$ is continuous and positive semi-definite on the compact space $\mathbb { S } ^ { d - 1 } \times \mathbb { S } ^ { d - 1 }$ , by Mercer’s theorem, we also have the expansion of $k _ { \rho }$

For item 3: The estimate $\xi _ { k } \asymp k ^ { - ( d + 2 ) }$ follows directly from Corollary C.4. It remains to account for multiplicities and determine the decay of the ordered eigenvalues. Now we

rearrange the eigenvalues in non-increasing order with multiplicity. The dimension formula gives

$$
\begin{array} { l } { { N ( d , k ) } } \\ { { \displaystyle = \frac { 2 k + d - 2 } { k } \binom { k + d - 3 } { d - 2 } = \frac { 2 k + d - 2 } { k } \frac { ( k + d - 3 ) ! } { ( d - 2 ) ! ( k - 1 ) ! } \sim \frac { 2 } { ( d - 2 ) ! } k ^ { d - 2 } \mathrm { a s } k  + \infty . } } \end{array}
$$

Thus, $N ( d , k ) \asymp k ^ { d - 2 }$ . Consequently, the number of eigenvalues, counted with multiplicity, contributed by degrees up to � satisfies $\begin{array} { r } { \sum _ { k = 0 } ^ { K } N ( d , k ) \times K ^ { d - 1 } } \end{array}$

Combining $\xi _ { k } \asymp k ^ { - ( d + 2 ) }$ with $\begin{array} { r } { \sum _ { k = 0 } ^ { K } N ( d , k ) \times K ^ { d - 1 } } \end{array}$ shows that an eigenvalue with ordered index � corresponds to a degree $K \asymp n ^ { 1 / ( d - 1 ) }$ . Hence,

$$
\eta _ { n } = \xi _ { K } \asymp K ^ { - ( d + 2 ) } \asymp n ^ { - \frac { d + 2 } { d - 1 } } , ~ \mathrm { a s } ~ n \to \infty .
$$

This completes the proof.

Before proving Proposition 4.2, we first prove the special case of uniform spherical distribution with $\tau = 1$

Proposition C.5 (Mismatch ofUniform Spherical Initialization). Under the spherical biased ReLU setting, let $\rho _ { 0 } = \mathrm { U n i f } ( \mathbb { S } ^ { d } )$ . For every $\varepsilon > 0 ,$ , there exists $\lambda _ { \varepsilon } > 0$ such that the mismatch associated with the initialization distribution $\rho _ { 0 }$ satisfies

$$
\frac { \sqrt { 2 } ( d + 1 ) } { d } - \varepsilon \le \kappa _ { \gamma _ { 0 } } ( \lambda ) \le \frac { 2 \sqrt { 2 } ( d + 1 ) } { d } + \varepsilon , f o r \lambda \in ( 0 , \lambda _ { \varepsilon } ] .
$$

Proof ofProposition C.5. For the upper bound, by the definition of $\kappa _ { \gamma _ { 0 } } ( \lambda )$ and (C.2),

$$
\kappa _ { \gamma _ { 0 } } ( \lambda ) = \mathop { \mathrm { e s s } } _ { \theta \sim \rho _ { 0 } } \operatorname* { s u p } _ { d ( \lambda ) } \frac { s _ { \lambda } ( \theta ) } { \theta \sim \rho _ { 0 } } = \mathop { \mathrm { e s s } } _ { k = 0 } \operatorname* { s u p } _ { \theta \sim \rho _ { 0 } } \left( \sum _ { k = 0 } ^ { \infty } \frac { N ( d , k ) \chi _ { k } ( b ) ^ { 2 } } { \xi _ { k } + \lambda } \right) \Biggl / \left( \sum _ { k = 0 } ^ { \infty } \frac { N ( d , k ) \xi _ { k } } { \xi _ { k } + \lambda } \right) .
$$

Fix $\varepsilon > 0$ . To bound this ratio, we compare $\chi _ { k } ( b ) ^ { 2 }$ with $\xi _ { k }$ uniformly over $b \in \left[ { - 1 , 1 } \right]$ for all suficiently large �.

We first consider a compact interval inside $( - 1 / \sqrt { 2 } , 1 / \sqrt { 2 } )$ . For $\delta \in ( 0 , 1 / \sqrt { 2 } )$ , let $I _ { \delta } = [ - 1 / \sqrt { 2 } + \delta , 1 / \sqrt { 2 } - \delta ]$ . Squaring (C.5), using $\cos ^ { 2 } \Psi _ { k } ( b ) \leq 1$ , and comparing with Corollary C.4, we obtain

$$
\operatorname* { l i m } _ { k \to \infty } \operatorname* { s u p } _ { b \in I _ { \delta } } \frac { \chi _ { k } ( b ) ^ { 2 } } { G _ { d } ( b ) \xi _ { k } } \leq \frac { C _ { 1 , d } ^ { 2 } } { C _ { 1 , d } ^ { 2 } d / [ 2 \sqrt { 2 } ( d + 1 ) ] } = \frac { 2 \sqrt { 2 } ( d + 1 ) } { d } .
$$

Therefore, there exists $K \geq 2$ such that for any $k \ge K , b \in I _ { \delta }$

$$
\chi _ { k } ( b ) ^ { 2 } \leq \Big ( \frac { 2 \sqrt { 2 } ( d + 1 ) } { d } + \frac { \varepsilon } { 2 } \Big ) G _ { d } ( b ) \xi _ { k } .
$$

We next extend this estimate to the whole interval $[ - 1 , 1 ]$ . The global bound in item 2 of Lemma C.3, together with $\xi _ { k } \asymp k ^ { - ( d + 2 ) }$ and $\xi _ { k } > 0$ , gives

$$
\chi _ { k } ( b ) ^ { 2 } \leq C _ { d } G _ { d } ( b ) \xi _ { k } , k \geq 2 , b \in [ - 1 , 1 ] .
$$

Since $G _ { d } ( b )$ tends to zero as $| b |  1 / \sqrt { 2 }$ and vanishes for $| b | \geq 1 / { \sqrt { 2 } }$ , choose � suficiently small that

$$
C _ { d } G _ { d } ( b ) \leq \frac { \varepsilon } { 2 } , b \in [ - 1 , 1 ] \backslash I _ { \delta } .
$$

For this �, take the corresponding � in the interior estimate. Since $\chi _ { k } ( b ) ^ { 2 } \leq ( \varepsilon / 2 ) \xi _ { k }$ outside $I _ { \delta }$ and $G _ { d } ( b ) \leq 1$ inside it, combining the two regions gives

$$
\chi _ { k } ( b ) ^ { 2 } \leq \Big ( \frac { 2 \sqrt { 2 } ( d + 1 ) } { d } + \frac { \varepsilon } { 2 } \Big ) \xi _ { k } , \ k \geq K , \ b \in [ - 1 , 1 ] .
$$

For this fixed �, continuity of $\chi _ { k }$ on $[ - 1 , 1 ]$ gives

$$
C _ { K } : = \sum _ { k = 0 } ^ { K - 1 } \frac { N ( d , k ) } { \xi _ { k } } \operatorname* { s u p } _ { t \in [ - 1 , 1 ] } \chi _ { k } ( t ) ^ { 2 } < \infty .
$$

Splitting the sum at � and using the uniform bound above, we obtain

$$
\begin{array} { l } { \displaystyle \frac { s _ { \lambda } ( \theta ) } { d ( \lambda ) } = \frac { 1 } { d ( \lambda ) } \Big [ \sum _ { k = 0 } ^ { K - 1 } \frac { N ( d , k ) \chi _ { k } ( b ) ^ { 2 } } { \xi _ { k } + \lambda } + \sum _ { k = K } ^ { \infty } \frac { N ( d , k ) \chi _ { k } ( b ) ^ { 2 } } { \xi _ { k } + \lambda } \Big ] } \\ { \displaystyle \qquad \leq \frac { C _ { K } } { d ( \lambda ) } + \Big ( \frac { 2 \sqrt { 2 } ( d + 1 ) } { d } + \frac { \varepsilon } { 2 } \Big ) \frac { 1 } { d ( \lambda ) } \sum _ { k = K } ^ { \infty } \frac { N ( d , k ) \xi _ { k } } { \xi _ { k } + \lambda } \leq \frac { C _ { K } } { d ( \lambda ) } + \frac { 2 \sqrt { 2 } ( d + 1 ) } { d } + \frac { \varepsilon } { 2 } . } \end{array}
$$

Taking the essential supremum yields

$$
\kappa _ { \gamma _ { 0 } } ( \lambda ) \leq \frac { 2 \sqrt { 2 } ( d + 1 ) } { d } + \frac { \varepsilon } { 2 } + \frac { C _ { K } } { d ( \lambda ) } .
$$

Since $\xi _ { k } > 0$ for every �, monotone convergence gives lim $\partial {  } 0 .$ $\begin{array} { r } { d ( \lambda ) = \sum _ { k = 0 } ^ { \infty } N ( d , k ) = } \end{array}$ . Hence there exists $\lambda _ { \varepsilon } > 0$ such that $C _ { K } / d ( \lambda ) \leq \varepsilon / 2$ whenever $0 < \lambda \leq \lambda _ { \varepsilon }$ . Consequently,

$$
\kappa _ { \gamma _ { 0 } } ( \lambda ) \leq \frac { 2 \sqrt { 2 } ( d + 1 ) } { d } + \varepsilon , 0 < \lambda \leq \lambda _ { \varepsilon } .
$$

For the lower bound, setting $b = 0$ in Lemma C.3 gives $\begin{array} { r } { \Psi _ { k } ( 0 ) = \frac { ( k - 2 ) \pi } { 2 } } \end{array}$ . Combining this with the expansion in (C.5) and Corollary C.4, we have

$$
\frac { \chi _ { k } ( 0 ) ^ { 2 } } { \xi _ { k } } = \frac { 2 \sqrt { 2 } ( d + 1 ) } { d } \cos ^ { 2 } ( \frac { k \pi } { 2 } ) + o ( 1 ) , k  \infty .\tag{C.11}
$$

We note that the cosine squared equals one for even � and zero for odd �. To compare the contributions of even and odd degrees, the dimension formula (4.1) gives

$$
\frac { N ( d , 2 k + 1 ) } { N ( d , 2 k ) } = \frac { 4 k + d } { 4 k + d - 2 } \frac { 2 k + d - 2 } { 2 k + 1 }  1 , k  \infty ,
$$

and Corollary C.4 also gives

$$
\biggl | \frac { \xi _ { 2 k } + \lambda } { \xi _ { 2 k + 1 } + \lambda } - 1 \biggr | = \biggl | \frac { \xi _ { 2 k } - \xi _ { 2 k + 1 } } { \xi _ { 2 k + 1 } + \lambda } \biggr | \leq \biggl | \frac { \xi _ { 2 k } } { \xi _ { 2 k + 1 } } - 1 \biggr |  0 , \ k  \infty ,
$$

uniformly over $\lambda > 0$ . To get the final lower bound, we need to estimate $\begin{array} { r } { \frac { 1 } { d ( \lambda ) } \sum _ { k = 1 } ^ { \infty } \frac { N ( d , 2 k ) \xi _ { 2 k } } { \xi _ { 2 k } + \lambda } } \end{array}$ For fixed $\varepsilon > 0$ , by the preceding estimates, there exists $K \geq 2$ , independent of $\lambda ,$ such that for all $k \geq K$ and $\lambda > 0$

$$
\left| \frac { N ( d , 2 k ) \xi _ { 2 k } } { \xi _ { 2 k } + \lambda } - \frac { N ( d , 2 k + 1 ) \xi _ { 2 k + 1 } } { \xi _ { 2 k + 1 } + \lambda } \right| \leq \varepsilon \frac { N ( d , 2 k ) \xi _ { 2 k } } { \xi _ { 2 k } + \lambda } .
$$

Since $\begin{array} { r } { d ( \lambda ) = \sum _ { k = 0 } ^ { \infty } \frac { N ( d , k ) \xi _ { k } } { \xi _ { k } + \lambda } } \end{array}$ , we have

$$
\begin{array} { l } { \displaystyle \left| \frac { 1 } { d ( \lambda ) } \Big ( 2 \sum _ { k = 1 } ^ { \infty } \frac { N ( d , 2 k ) \xi _ { 2 k } } { \xi _ { 2 k } + \lambda } - d ( \lambda ) \Big ) \right| } \\ { \displaystyle \leq \frac { 1 } { d ( \lambda ) } \sum _ { k = 0 } ^ { 2 K - 1 } \frac { N ( d , k ) \xi _ { k } } { \xi _ { k } + \lambda } + \frac { 1 } { d ( \lambda ) } \sum _ { k = K } ^ { \infty } \left| \frac { N ( d , 2 k ) \xi _ { 2 k } } { \xi _ { 2 k } + \lambda } - \frac { N ( d , 2 k + 1 ) \xi _ { 2 k + 1 } } { \xi _ { 2 k + 1 } + \lambda } \right| } \\ { \displaystyle \leq \frac { \sum _ { k = 0 } ^ { 2 K - 1 } N ( d , k ) } { d ( \lambda ) } + \frac { \varepsilon } { d ( \lambda ) } \sum _ { k = K } ^ { \infty } \frac { N ( d , 2 k ) \xi _ { 2 k } } { \xi _ { 2 k } + \lambda } \leq \frac { \sum _ { k = 0 } ^ { 2 K - 1 } N ( d , k ) } { d ( \lambda ) } + \varepsilon . } \end{array}
$$

Since $d ( \lambda )  \infty$ as $\lambda \to 0 _ { + }$ and $\varepsilon > 0$ is arbitrary, this proves

$$
\operatorname* { l i m } _ { \lambda \to 0 _ { + } } { \frac { 1 } { d ( \lambda ) } } \sum _ { k = 1 } ^ { \infty } { \frac { N ( d , 2 k ) \xi _ { 2 k } } { \xi _ { 2 k } + \lambda } } = { \frac { 1 } { 2 } } .\tag{C.12}
$$

Since $s _ { \lambda }$ is continuous on ${ \mathbb S } ^ { d }$ and $\rho _ { 0 }$ has full support, the essential supremum is at least $s _ { \lambda } ( ( \omega , 0 ) )$ for $\theta = \left( \omega , 0 \right)$ . Consequently, by (C.2), (C.11) and (C.12)

$$
\begin{array} { l } { \displaystyle \operatorname* { l i m i n f } _ { \lambda \to 0 _ { + } } \kappa _ { \gamma _ { 0 } } ( \lambda ) \geq \operatorname* { l i m } _ { \lambda \to 0 _ { + } } \frac { s _ { \lambda } ( ( \omega , 0 ) ) } { d ( \lambda ) } } \\ { = \displaystyle \frac { 2 \sqrt { 2 } ( d + 1 ) } { d } \operatorname* { l i m } _ { \lambda \to 0 _ { + } } \frac { 1 } { d ( \lambda ) } \sum _ { k = 0 } ^ { \infty } \frac { N ( d , k ) \xi _ { k } } { \xi _ { k } + \lambda } \cos ^ { 2 } \left( \frac { k \pi } { 2 } \right) } \\ { = \displaystyle \frac { 2 \sqrt { 2 } ( d + 1 ) } { d } \operatorname* { l i m } _ { \lambda \to 0 _ { + } } \frac { 1 } { d ( \lambda ) } \sum _ { k = 1 } ^ { \infty } \frac { N ( d , 2 k ) \xi _ { 2 k } } { \xi _ { 2 k } + \lambda } = \frac { \sqrt { 2 } ( d + 1 ) } { d } . } \end{array}\tag{C.13}
$$

After decreasing $\lambda _ { \varepsilon }$ if necessary, the upper and lower bounds together give

$$
\frac { \sqrt { 2 } ( d + 1 ) } { d } - \varepsilon \le \kappa _ { \gamma _ { 0 } } ( \lambda ) \le \frac { 2 \sqrt { 2 } ( d + 1 ) } { d } + \varepsilon , 0 < \lambda \le \lambda _ { \varepsilon } .
$$

Proposition C.5 now gives the corresponding result for scaled spherical initialization.

Proof ofProposition 4.2 (Mismatch of Scaled Spherical Initialization). The leverage score associated with $\rho _ { 0 } ^ { ( \tau ) }$ satisfies $\begin{array} { r } { s _ { \lambda , 0 } ^ { ( \tau ) } ( \theta ) = \frac { s _ { \lambda } ( \theta ) } { \gamma _ { 0 } ^ { ( \tau ) } ( \theta ) } } \end{array}$ . Moreover, the corresponding efective dimension is still $d ( \lambda )$ , since

$$
\int _ { \mathbb { S } ^ { d } } s _ { \lambda , 0 } ^ { ( \tau ) } ( \theta ) d \rho _ { 0 } ^ { ( \tau ) } ( \theta ) = \int _ { \mathbb { S } ^ { d } } \frac { s _ { \lambda } ( \theta ) } { \gamma _ { 0 } ^ { ( \tau ) } ( \theta ) } \gamma _ { 0 } ^ { ( \tau ) } ( \theta ) d \rho ( \theta ) = d ( \lambda ) .
$$

Since $\gamma _ { 0 } ^ { ( \tau ) } ( \theta ) \geq 1 / \tau$ and $\gamma _ { 0 } ^ { ( \tau ) } ( ( \omega , 0 ) ) = 1 / \tau$ for $\omega \in \mathbb { S } ^ { d - 1 }$ , we obtain

$$
\tau \frac { s _ { \lambda } ( ( \omega , 0 ) ) } { d ( \lambda ) } \le \kappa _ { \gamma _ { 0 } ^ { ( \tau ) } } ( \lambda ) = \mathsf { e s s u p } \frac { s _ { \lambda } ( \theta ) } { \theta \in \mathbb { S } ^ { d } } \frac { \mathsf { \sigma } s _ { \lambda } ( \theta ) } { \gamma _ { 0 } ^ { ( \tau ) } ( \theta ) d ( \lambda ) } \le \tau \mathsf { e s s u p } \frac { s _ { \lambda } ( \theta ) } { \theta \in \mathbb { S } ^ { d } } = \tau \kappa _ { \gamma _ { 0 } ^ { ( 1 ) } } ( \lambda ) .
$$

Applying Proposition C.5 with $\varepsilon / \tau$ and using the limit at $b = 0$ established in (C.13) gives

$$
\frac { \sqrt { 2 } \tau ( d + 1 ) } { d } - \varepsilon \leq \kappa _ { \gamma _ { 0 } ^ { ( \tau ) } } ( \lambda ) \leq \frac { 2 \sqrt { 2 } \tau ( d + 1 ) } { d } + \varepsilon
$$

for all suficiently small $\lambda > 0$

Proof of Theorem 4.3 (Rates of Spherical Biased-ReLU Features). By Proposition 4.1, the ordered eigenvalues of $T _ { \rho }$ satisfy $\eta _ { k } \asymp k ^ { - \frac { d + 2 } { d - 1 } }$ . Hence, the polynomial eigenvalue decay assumption in Theorem 3.6 holds with exponent $\textstyle \alpha = { \frac { d + 2 } { d - 1 } }$

For the trainable neurons, take the normalized leverage density $\begin{array} { r } { \gamma _ { 1 } ( \theta ) = \frac { s _ { \lambda } ( \theta ) } { d ( \lambda ) } } \end{array}$ which indicates the $\boldsymbol { \kappa } _ { \gamma _ { 1 } } ( \lambda ) = \boldsymbol { \kappa } _ { 1 } ^ { * } = 1$ . For the randomly fixed neurons, apply Proposition 4.2 with $\begin{array} { r } { \varepsilon = \frac { 2 \sqrt { 2 } \tau ( d + 1 ) } { d } } \end{array}$ , and there exists $\bar { \lambda } _ { d , \tau } > 0$ such that

$$
\kappa _ { \gamma _ { 0 } ^ { ( \tau ) } } \left( \lambda \right) \leq \frac { 4 \sqrt { 2 } \tau ( d + 1 ) } { d } , \mathrm { f o r } 0 < \lambda \leq \bar { \lambda } _ { d , \tau } .
$$

Set $W = \operatorname* { m a x } \{ d , m , 1 \}$ . Following (B.16) in the proof of Theorem 3.6, choose

$$
\lambda _ { P , m } = A _ { d } \Bigl ( \Bigl ( P + \frac { ( 1 - P ) d } { 4 \sqrt { 2 } \tau ( d + 1 ) } \Bigr ) \frac { 9 m } { 4 0 \log ( 6 3 m / ( 2 0 \delta ) ) } \Bigr ) ^ { - \frac { d + 2 } { d - 1 } } ,
$$

where $A _ { d } > 0$ depends only on � and is chosen to satisfy (B.16). If $\lambda _ { P , m } \leq \bar { \lambda } _ { d , \tau }$ , Theorem 3.6 directly yields, with probability at least $1 - \delta$

$$
\operatorname* { s u p } _ { \| f \| _ { \mathcal H _ { \rho } } \leq 1 } \operatorname* { i n f } _ { f _ { 2 } ^ { ( P ) } \in \mathcal F _ { P } ( W , 2 ) } \| f - f _ { 2 } ^ { ( P ) } \| _ { L _ { 2 } ( d \mu ) } ^ { 2 } \lesssim \Big ( \Big ( P + \frac { ( 1 - P ) d } { 4 \sqrt 2 \tau ( d + 1 ) } \Big ) \frac { 9 m } { 4 0 \log ( 6 3 m / ( 2 0 \delta ) ) } \Big ) ^ { - \frac { d + 2 } { d - 1 } } .
$$

If $\lambda _ { P , m } \geq \bar { \lambda } _ { d , \tau }$ , we use the zero function $f _ { 2 } ^ { ( P ) } = 0 \mathrm { i n } \mathcal { F } _ { P } ( W , 2 )$ . Since $\| f \| _ { \mathcal { H } _ { \rho } } = \| q _ { f } \| _ { L _ { 2 } ( d \mu ) } \leq$ 1 implies

$$
\begin{array} { r } { \Vert f \Vert _ { L _ { 2 } ( d \mu ) } ^ { 2 } = \Vert T _ { \rho } ^ { 1 / 2 } q _ { f } \Vert _ { L _ { 2 } ( d \mu ) } ^ { 2 } = \langle q _ { f } , T _ { \rho } q _ { f } \rangle _ { L _ { 2 } ( d \mu ) } \leq \Vert T _ { \rho } \Vert _ { \mathrm { o p } } \Vert q _ { f } \Vert _ { L _ { 2 } ( d \mu ) } ^ { 2 } \leq \Vert T _ { \rho } \Vert _ { \mathrm { o p } } , } \end{array}
$$

then we have

$$
\operatorname* { s u p } _ { \| f \| _ { \mathcal H _ { \rho } \leq 1 } f _ { 2 } ^ { ( P ) } \in \mathcal F _ { P } ( W , 2 ) } \| f - f _ { 2 } ^ { ( P ) } \| _ { L _ { 2 } ( d \mu ) } ^ { 2 } \leq \| T _ { \rho } \| _ { \mathrm { o p } } \leq \frac { \| T _ { \rho } \| _ { \mathrm { o p } } } { \bar { \lambda } _ { d , \tau } } \lambda _ { P , m } .
$$

Thus, the same rate holds in both cases, with a constant depending only on the fixed spherical setting and �.

Besides, since $\begin{array} { r } { P + \frac { ( 1 - P ) d } { 4 \sqrt { 2 } \tau ( d + 1 ) } \geq \frac { d } { 4 \sqrt { 2 } \tau ( d + 1 ) } } \end{array}$ , we have

$$
\lambda _ { P , m } \leq A _ { d } \Big ( \frac { d } { 4 \sqrt { 2 } \tau ( d + 1 ) } \frac { 9 m } { 4 0 \log ( 6 3 m / ( 2 0 \delta ) ) } \Big ) ^ { - \frac { d + 2 } { d - 1 } } .
$$

Therefore, $\lambda _ { P , m } \leq \bar { \lambda } _ { d , \imath }$ whenever $\begin{array} { r } { \frac { 9 m } { 4 0 \log ( 6 3 m / ( 2 0 \delta ) ) } \geq \frac { 4 \sqrt { 2 } \tau ( d + 1 ) } { d } \big ( A _ { d } / \bar { \lambda } _ { d , \tau } \big ) ^ { \frac { d - 1 } { d + 2 } } } \end{array}$ , which means that, for fixed $d , \tau$ and $\delta ,$ , suficiently large � leads to the same bound result.

## D. Error Bounds for the Structured Three-Layer Example 5.1

Let $r = m _ { \mathrm { f r e e } } \in \{ 0 , \ldots , 2 0 \}$ be the number of trainable first-layer neurons in the three-layer network and the number of trainable hidden neurons in the two-layer network. The two network architectures have the expression

$$
\begin{array} { l } { { f _ { 3 } ( x ) = \displaystyle { \beta _ { 0 } + \sum _ { j = 1 } ^ { 1 0 } \sum _ { \ell = 1 } ^ { 2 0 } \beta _ { j , \ell } \omega _ { j , \ell } \big ( \phi ( x ; \theta _ { 2 j - 1 } ) + \phi ( x ; \theta _ { 2 j } ) - b _ { j , \ell } \big ) _ { + } , } } } \\ { { \displaystyle f _ { 2 } ( x ) = \beta _ { 0 } ^ { ( 2 ) } + \sum _ { i = 1 } ^ { 2 2 0 } \beta _ { i } ^ { ( 2 ) } \phi ( x ; \theta _ { i } ^ { ( 2 ) } ) . } } \end{array}
$$

Proposition D.1 (Approximation Error Upper Bounds). For the construction in Example 5.1, the approximation errors satisfy

$$
\mathbb { E } \operatorname* { i n f } _ { \theta _ { \mathrm { f r e e } } , \beta } \big \| f ^ { * } - f _ { d } \big \| _ { L _ { 2 } ( d \mu ) } ^ { 2 } \leq \mathcal { E } ^ { ( d ) } ( r ) , \ d \in \{ 2 , 3 \} , \ r \in \{ 0 , 2 , 4 , 2 0 \} ,
$$

where the explicit upper bounds are

$$
{ \mathcal { E } } ^ { ( 3 ) } ( 0 ) = { \frac { 1 } { 9 } } , { \mathcal { E } } ^ { ( 3 ) } ( 2 ) = { \frac { 1 } { 4 5 } } + { \frac { 1 4 4 } { \prod _ { j = 2 1 } ^ { 2 5 } j } } , { \mathcal { E } } ^ { ( 3 ) } ( 4 ) = { \frac { 1 8 0 } { \prod _ { j = 2 1 } ^ { 2 5 } j } } , { \mathcal { E } } ^ { ( 3 ) } ( 2 0 ) = { \frac { 5 8 0 } { \prod _ { j = 1 0 1 } ^ { 1 0 5 } j } } , { \mathcal { E } } ^ { ( 3 ) } ( 2 0 ) = { \frac { 5 8 0 } { \prod _ { j = 1 0 1 } ^ { 1 0 5 } j } } ,
$$

and

$$
\mathscr { E } ^ { ( 2 ) } ( 0 ) = \frac { 1 } { 9 } , \mathscr { E } ^ { ( 2 ) } ( 2 ) = \frac { 1 7 } { 7 2 9 } , \mathscr { E } ^ { ( 2 ) } ( 4 ) = \frac { 1 } { 7 2 9 } , \mathscr { E } ^ { ( 2 ) } ( 2 0 ) = \frac { 1 } { 9 \cdot 1 1 ^ { 4 } } .
$$

Here $\mathcal { E } ^ { ( d ) } ( r )$ denotes the stated upper bound.

Proof ofProposition D.1. Overview. The zero output gives a common baseline upper bound $\mathcal { E } ^ { ( d ) } ( 0 )$ . For the three-layer network, pairs of trainable first-layer neurons produce absolute coordinate values, allowing the fixed second-layer blocks to implement piecewiselinear interpolation with random nodes. The interpolation residuals and moments of the random spacings then yield the expected error bounds $\mathcal { E } ^ { ( 3 ) } ( r )$ . For the two-layer network, the trainable neurons directly implement symmetric piecewise-linear interpolation with equally spaced nodes. Centering the interpolation residuals and integrating their squares then yields the two-layer error bounds for the diferent numbers oftrainable neurons $\mathcal { E } ^ { ( 2 ) } ( r )$

Proof Details. Since the coordinates of � are independent and $\begin{array} { r } { \int _ { - 1 } ^ { 1 } \frac { t ^ { 2 } - 1 / 3 } { 2 } d t = 0 } \end{array}$ with $\textstyle \int _ { - 1 } ^ { 1 } { \frac { ( t ^ { 2 } - 1 / 3 ) ^ { 2 } } { 2 } } d t = { \frac { 4 } { 4 5 } }$ , the target satisfies $\begin{array} { r } { \| f ^ { * } \| _ { L _ { 2 } ( d \mu ) } ^ { 2 } = \frac { 4 } { 4 5 } ( 1 + \frac { 1 } { 4 } ) = \frac { 1 } { 9 } } \end{array}$ . Choosing all output coeficients and the output bias to be zero therefore gives the baseline upper bound $\mathcal { E } ^ { ( d ) } ( 0 )$ for both architectures.

The Three-Layer Upper Bound. Since $\omega _ { j , \ell } > 0$ , the approximation residual is

$$
\begin{array} { l } { f _ { 3 } ( x ) - f ^ { * } ( x ) } \\ { = \beta _ { 0 } + \displaystyle \sum _ { j = 1 } ^ { 1 0 } \displaystyle \sum _ { \ell = 1 } ^ { 2 0 } \beta _ { j , \ell } \omega _ { j , \ell } \left[ \phi ( x ; \theta _ { 2 j - 1 } ) + \phi ( x ; \theta _ { 2 j } ) - b _ { j , \ell } \right] _ { + } - ( x _ { 2 } ^ { 2 } - 1 / 3 ) - \displaystyle \frac 1 2 ( x _ { 3 } ^ { 2 } - 1 / 3 ) . } \end{array}
$$

(i) Construction with Four Trainable Neurons. With four trainable first-layer neurons, choose them as

$$
\theta _ { 1 } = ( { \bf e } _ { 2 } , 0 ) , \theta _ { 2 } = ( - { \bf e } _ { 2 } , 0 ) , \theta _ { 3 } = ( { \bf e } _ { 3 } , 0 ) , \theta _ { 4 } = ( - { \bf e } _ { 3 } , 0 ) .
$$

Their pairwise sums are $| x _ { 2 } |$ and $\left| x _ { 3 } \right|$ . Setting the output coeficients of all other blocks to zero therefore gives

$$
\begin{array} { l } { { f _ { 3 } ( x ) - f ^ { * } ( x ) = \displaystyle \beta _ { 0 } + \sum _ { \ell = 1 } ^ { 2 0 } \beta _ { 1 , \ell } \omega _ { 1 , \ell } ( | x _ { 2 } | - b _ { 1 , \ell } ) _ { + } + \sum _ { \ell = 1 } ^ { 2 0 } \beta _ { 2 , \ell } \omega _ { 2 , \ell } ( | x _ { 3 } | - b _ { 2 , \ell } ) _ { + } } } \\ { { \displaystyle ~ - ~ ( x _ { 2 } ^ { 2 } - 1 / 3 ) - \frac { 1 } { 2 } ( x _ { 3 } ^ { 2 } - 1 / 3 ) . } } \end{array}
$$

Although $x _ { k }$ ranges over $[ - 1 , 1 ]$ , the folded input $\left| x _ { k } \right|$ is uniform on 0, 1 . The interpolation below is therefore performed on 0, 1 with $u = \left| x _ { k } \right|$ , for which $u ^ { 2 } = x _ { k } ^ { 2 }$

For each selected block $j = 1 , 2$ , reorder and relabel its biases and the corresponding weights so that, almost surely,

$$
0 < b _ { j , 1 } < \cdot \cdot \cdot < b _ { j , 2 0 } < 1 , b _ { j , 0 } = 0 , b _ { j , 2 1 } = 1 .
$$

These biases serve as interpolation nodes. For the first block, approximate $u ^ { 2 }$ by a function that is constant with value $b _ { 1 , 1 } ^ { 2 }$ on $[ 0 , b _ { 1 , 1 } ]$ and linearly interpolates $u ^ { 2 }$ on each $\left[ b _ { 1 , \ell } , b _ { 1 , \ell + 1 } \right]$ The linear interpolant has slope $b _ { 1 , \ell } + b _ { 1 , \ell + 1 }$ on each such interval. Those coeficients

$\beta _ { 1 , \ell } \omega _ { 1 , \ell }$ decide the successive slope diferences, with zero slope on the initial interval. Consequently, choose

$$
\begin{array} { r } { \beta _ { 1 , \ell } = \left\{ \begin{array} { l l } { \frac { b _ { 1 , 1 } + b _ { 1 , 2 } } { \omega _ { 1 , 1 } } , } & { \ell = 1 , } \\ { \frac { b _ { 1 , \ell + 1 } - b _ { 1 , \ell - 1 } } { \omega _ { 1 , \ell } } , } & { 2 \leq \ell \leq 2 0 . } \end{array} \right. } \end{array}
$$

Together with the constant term $b _ { 1 , 1 } ^ { 2 }$ , these coeficients give the prescribed interpolation, with residual

$$
\begin{array} { l } { { b _ { 1 , 1 } ^ { 2 } + \displaystyle \sum _ { \ell = 1 } ^ { 2 0 } \beta _ { 1 , \ell } \omega _ { 1 , \ell } ( u - b _ { 1 , \ell } ) _ { + } - u ^ { 2 } } } \\ { { \ } } \\ { { \qquad = \left\{ \begin{array} { l l } { { b _ { 1 , 1 } ^ { 2 } - u ^ { 2 } , } } & { { 0 \leq u \leq b _ { 1 , 1 } , } } \\ { { ( u - b _ { 1 , \ell } ) ( b _ { 1 , \ell + 1 } - u ) , } } & { { 1 \leq \ell \leq 2 0 , \ b _ { 1 , \ell } \leq u \leq b _ { 1 , \ell + 1 } . } } \end{array} \right. } } \end{array}\tag{D.1}
$$

Evaluating this expression at $u = | x _ { 2 } |$ gives the residual for the first target coordinate on 1, 1 . For the second block, apply the same construction to ${ \scriptstyle { \frac { 1 } { 2 } } } u ^ { 2 }$ and evaluate it at $u = \left| x _ { 3 } \right|$ This gives

$$
\begin{array} { r } { \beta _ { 2 , \ell } = \left\{ \begin{array} { l l } { \frac { b _ { 2 , 1 } + b _ { 2 , 2 } } { 2 \omega _ { 2 , 1 } } , } & { \ell = 1 , } \\ { \frac { b _ { 2 , \ell + 1 } - b _ { 2 , \ell - 1 } } { 2 \omega _ { 2 , \ell } } , } & { 2 \leq \ell \leq 2 0 . } \end{array} \right. } \end{array}
$$

Its interpolation residual has the same form with its own biases and a factor of $1 / 2$

(ii) Centering and Error Estimation. Since the target has zero spatial mean, $\int _ { X } f ^ { * } d \mu =$ $0 ,$ center the constructed approximation so that $\int _ { X } f _ { 3 } d \mu = 0$ by choosing

$$
\begin{array} { r l } & { \displaystyle \beta _ { 0 } = b _ { 1 , 1 } ^ { 2 } + \frac 1 2 b _ { 2 , 1 } ^ { 2 } - \left[ b _ { 1 , 1 } ^ { 2 } + \frac 1 2 b _ { 2 , 1 } ^ { 2 } + \sum _ { j = 1 } ^ { 2 } \sum _ { \ell = 1 } ^ { 2 0 } \beta _ { j , \ell } \omega _ { j , \ell } \frac 1 2 \int _ { - 1 } ^ { 1 } ( | u | - b _ { j , \ell } ) _ { + } d u \right] } \\ & { \quad = - \frac 1 2 \sum _ { j = 1 } ^ { 2 } \sum _ { \ell = 1 } ^ { 2 0 } \beta _ { j , \ell } \omega _ { j , \ell } ( 1 - b _ { j , \ell } ) ^ { 2 } , } \end{array}
$$

where the first $b _ { 1 , 1 } ^ { 2 } + b _ { 2 , 1 } ^ { 2 } / 2$ supplies the constant terms of the two interpolants, and the later bracketed expression gives the combined spatial mean. With all coeficients determined, the resulting residual is

$$
\begin{array} { l } { { f _ { 3 } ( x ) - f ^ { * } ( x ) = \displaystyle \sum _ { \ell = 1 } ^ { 2 0 } \beta _ { 1 , \ell } \omega _ { 1 , \ell } \Big [ ( | x _ { 2 } | - b _ { 1 , \ell } ) _ { + } - \frac { ( 1 - b _ { 1 , \ell } ) ^ { 2 } } { 2 } \Big ] - ( x _ { 2 } ^ { 2 } - 1 / 3 ) } } \\ { { \displaystyle \qquad + \sum _ { \ell = 1 } ^ { 2 0 } \beta _ { 2 , \ell } \omega _ { 2 , \ell } \Big [ ( | x _ { 3 } | - b _ { 2 , \ell } ) _ { + } - \frac { ( 1 - b _ { 2 , \ell } ) ^ { 2 } } { 2 } \Big ] - \frac { 1 } { 2 } ( x _ { 3 } ^ { 2 } - 1 / 3 ) . } } \end{array}
$$

Each coordinate error has zero spatial integral, so the cross term vanishes when the squared norm is expanded. Taking the first block as an example, since $x _ { 2 }$ is uniformly distributed

on 1, 1 , we have

$$
\begin{array} { l } { \displaystyle \mathbb { E } \int _ { - 1 } ^ { 1 } \Big ( \sum _ { \ell = 1 } ^ { 2 0 } \beta _ { 1 , \ell } \omega _ { 1 , \ell } \Big [ ( | x _ { 2 } | - b _ { 1 , \ell } ) _ { + } - \frac { ( 1 - b _ { 1 , \ell } ) ^ { 2 } } { 2 } \Big ] - ( x _ { 2 } ^ { 2 } - 1 / 3 ) \Big ) ^ { 2 } \frac { d x _ { 2 } } { 2 } } \\ { \displaystyle = \mathbb { E } \int _ { 0 } ^ { 1 } \Big ( b _ { 1 , 1 } ^ { 2 } + \sum _ { \ell = 1 } ^ { 2 0 } \beta _ { 1 , \ell } \omega _ { 1 , \ell } ( u - b _ { 1 , \ell } ) _ { + } - u ^ { 2 } \Big ) ^ { 2 } d u } \\ { \displaystyle \qquad - \mathbb { E } \Big ( \int _ { 0 } ^ { 1 } \Big [ b _ { 1 , 1 } ^ { 2 } + \sum _ { \ell = 1 } ^ { 2 0 } \beta _ { 1 , \ell } \omega _ { 1 , \ell } ( v - b _ { 1 , \ell } ) _ { + } - v ^ { 2 } \Big ] d v \Big ) ^ { 2 } , } \end{array}
$$

where we use $\begin{array} { r } { ( 1 - b _ { 1 , \ell } ) ^ { 2 } / 2 = \int _ { 0 } ^ { 1 } ( v - b _ { 1 , \ell } ) _ { + } d v , \int _ { 0 } ^ { 1 } v ^ { 2 } d v = 1 / 3 } \end{array}$ and replace $x _ { 2 }$ with �. Use the inequality for any square-integrable function � on 0, 1 ,

$$
\int _ { 0 } ^ { 1 } \Big [ g ( u ) - \int _ { 0 } ^ { 1 } g ( v ) d v \Big ] ^ { 2 } d u = \int _ { 0 } ^ { 1 } g ( u ) ^ { 2 } d u - \Big [ \int _ { 0 } ^ { 1 } g ( u ) d u \Big ] ^ { 2 } \leq \int _ { 0 } ^ { 1 } g ( u ) ^ { 2 } d u ,
$$

we get

$$
\begin{array} { r l } & { \mathbb { E } \displaystyle \int _ { - 1 } ^ { 1 } \Big ( \displaystyle \sum _ { \ell = 1 } ^ { 2 0 } \beta _ { 1 , \ell } \omega _ { 1 , \ell } \Big [ ( | x _ { 2 } | - b _ { 1 , \ell } ) _ { + } - \frac { ( 1 - b _ { 1 , \ell } ) ^ { 2 } } { 2 } \Big ] - ( x _ { 2 } ^ { 2 } - 1 / 3 ) \Big ) ^ { 2 } \frac { d x _ { 2 } } { 2 } } \\ & { \leq \mathbb { E } \displaystyle \int _ { 0 } ^ { 1 } \Big ( b _ { 1 , 1 } ^ { 2 } + \displaystyle \sum _ { \ell = 1 } ^ { 2 0 } \beta _ { 1 , \ell } \omega _ { 1 , \ell } ( u - b _ { 1 , \ell } ) _ { + } - u ^ { 2 } \Big ) ^ { 2 } d u } \end{array}
$$

Therefore, further using the residual results in (D.1), direct integration yields

$$
\begin{array} { l } { \displaystyle \mathbb { E } \| f _ { 3 } - f ^ { * } \| _ { L _ { 2 } ( d \mu ) } ^ { 2 } \leq \mathbb { E } \int _ { 0 } ^ { 1 } \Big ( b _ { 1 , 1 } ^ { 2 } + \sum _ { \ell = 1 } ^ { 2 0 } \beta _ { 1 , \ell } \omega _ { 1 , \ell } \big ( u - b _ { 1 , \ell } \big ) _ { + } - u ^ { 2 } \Big ) ^ { 2 } d u } \\ { \displaystyle \qquad + \mathbb { E } \int _ { 0 } ^ { 1 } \Big ( \frac { 1 } { 2 } b _ { 2 , 1 } ^ { 2 } + \sum _ { \ell = 1 } ^ { 2 0 } \beta _ { 2 , \ell } \omega _ { 2 , \ell } \big ( u - b _ { 2 , \ell } \big ) _ { + } - \frac { 1 } { 2 } u ^ { 2 } \Big ) ^ { 2 } d u } \\ { \displaystyle = \mathbb { E } \Big ( \frac { 8 } { 1 5 } \big ( b _ { 1 , 1 } - b _ { 1 , 0 } \big ) ^ { 5 } + \frac { 1 } { 3 0 } \sum _ { \ell = 1 } ^ { 2 0 } \big ( b _ { 1 , \ell + 1 } - b _ { 1 , \ell } \big ) ^ { 5 } \Big ) } \\ { \displaystyle \qquad + \frac { 1 } { 4 } \mathbb { E } \Big ( \frac { 8 } { 1 5 } \big ( b _ { 2 , 1 } - b _ { 2 , 0 } \big ) ^ { 5 } + \frac { 1 } { 3 0 } \sum _ { \ell = 1 } ^ { 2 0 } \big ( b _ { 2 , \ell + 1 } - b _ { 2 , \ell } \big ) ^ { 5 } \Big ) , } \end{array}
$$

where $b _ { j , 0 } = 0 , j \in \{ 1 , 2 \}$

By the uniform-spacing identity [47], each spacing $b _ { j , \ell + 1 } - b _ { j , \ell } , \ell \in \{ 0 , 1 , \cdot \cdot \cdot , 2 0 \}$ has the density $p ( x ) = 2 0 ( 1 - x ) ^ { 1 9 }$ . Therefore, E $\begin{array} { r } { ( b _ { j , \ell + 1 } - b _ { j , \ell } ) ^ { 5 } = 2 0 \int _ { 0 } ^ { 1 } x ^ { 5 } ( 1 - x ) ^ { 1 9 } d x = } \end{array}$ $\frac { 1 2 0 } { \prod _ { \ell = 2 1 } ^ { 2 5 } \ell }$ . Consequently,

$$
\mathbb { E } \Vert f _ { 3 } - f ^ { * } \Vert _ { L _ { 2 } ( d \mu ) } ^ { 2 } \le \Big ( 1 + \frac { 1 } { 4 } \Big ) \Big ( \frac { 8 } { 1 5 } + \frac { 2 0 } { 3 0 } \Big ) \frac { 1 2 0 } { \prod _ { \ell = 2 1 } ^ { 2 5 } \ell } = \frac { 1 8 0 } { \prod _ { \ell = 2 1 } ^ { 2 5 } \ell } = \mathcal { E } ^ { ( 3 ) } ( 4 ) .
$$

Taking the infimum over the trainable hidden parameters and output coeficients can only decrease the error, thereby proving the upper bound with four trainable neurons.

(iii) Two and Twenty Trainable Neurons. With two trainable first-layer neurons, choose $\theta _ { 1 } = ( \mathbf { e } _ { 2 } , 0 ) , \theta _ { 2 } = ( - \mathbf { e } _ { 2 } , 0 )$ . Their sum is $| x _ { 2 } |$ , so the first second-layer block provides 20 random hinge features for approximating $x _ { 2 } ^ { 2 } - 1 / 3$ . Set the output coeficients of all other blocks to zero. Applying the single-block interpolation estimation derived above and leaving the $x _ { 3 } \cdot$ -component unapproximated gives

$$
\mathbb { E } \operatorname* { i n f } _ { \theta _ { \mathrm { f r e e } } , \beta } \| f _ { 3 } - f ^ { * } \| _ { L _ { 2 } ( d \mu ) } ^ { 2 } \le \frac { 1 } { 4 5 } + \Big ( \frac { 8 } { 1 5 } + \frac { 2 0 } { 3 0 } \Big ) \frac { 1 2 0 } { \prod _ { j = 2 1 } ^ { 2 5 } j } = \frac { 1 } { 4 5 } + \frac { 1 4 4 } { \prod _ { j = 2 1 } ^ { 2 5 } j } = { \mathcal E } ^ { ( 3 ) } ( 2 ) .
$$

With 20 trainable first-layer neurons, assign $^ { 5 }$ pairs to $\left( x _ { 2 } \right) _ { + } + \left( - x _ { 2 } \right)$ and the other 5 pairs to $\left( x _ { 3 } \right) _ { + } + \left( - x _ { 3 } \right) _ { + }$ . Each coordinate then feeds 5 fixed second-layer blocks, providing 100 independent uniform biases. Pooling and ordering these biases gives the same interpolation construction with 100 nodes per coordinate. Repeating the calculation above yields

$$
\mathbb { E } \operatorname* { i n f } _ { \theta _ { \mathrm { f r e c } } , \beta } \big \| f _ { 3 } - f ^ { * } \big \| _ { L _ { 2 } ( d \mu ) } ^ { 2 } \le \Big ( 1 + \frac { 1 } { 4 } \Big ) \Big ( \frac { 8 } { 1 5 } + \frac { 1 0 0 } { 3 0 } \Big ) \frac { 1 2 0 } { \prod _ { \ell = 1 0 1 } ^ { 1 0 5 } \ell } = \frac { 5 8 0 } { \prod _ { \ell = 1 0 1 } ^ { 1 0 5 } \ell } = \mathcal { E } ^ { ( 3 ) } ( 2 0 ) .
$$

All second-layer parameters remain fixed in this construction.

The Two-Layer Upper Bound. We use the following symmetric interpolation construction for approximating $u ^ { 2 }$ . Let $q \in \{ 1 , 2 , \ldots , 5 \}$ , set $m _ { \mathrm { f r e e } } = 4 q$ , and consider the equally spaced nodes $\begin{array} { r } { u _ { i } = - 1 + \frac { 2 i } { 2 q + 1 } , \ 0 \leq i \leq 2 q + 1 } \end{array}$ . The piecewise-linear interpolant of $u ^ { 2 }$ at these nodes is

$$
I _ { q } ( u ) = \frac { 1 } { ( 2 q + 1 ) ^ { 2 } } + \frac { 4 } { 2 q + 1 } \sum _ { r = 1 } ^ { q } \bigg [ \Big ( u - \frac { 2 r - 1 } { 2 q + 1 } \Big ) _ { + } + \Big ( - u - \frac { 2 r - 1 } { 2 q + 1 } \Big ) _ { + } \bigg ] .
$$

Thus, $I _ { q }$ is represented by $q$ symmetric pairs of ReLU features, or $2 q$ neurons in total. On every interval $\left[ u _ { i } , u _ { i + 1 } \right]$ , we have $I _ { q } ( u ) - u ^ { 2 } = ( u - u _ { i } ) ( u _ { i + 1 } - u )$ . The spatial mean of the interpolation residual is

$$
\int _ { - 1 } ^ { 1 } \bigl [ I _ { q } ( u ) - u ^ { 2 } \bigr ] \frac { d u } { 2 } = \frac { 2 q + 1 } { 2 } \int _ { 0 } ^ { 2 / ( 2 q + 1 ) } u \Bigl ( \frac { 2 } { 2 q + 1 } - u \Bigr ) d u = \frac { 2 } { 3 ( 2 q + 1 ) ^ { 2 } } .
$$

Therefore, $\begin{array} { r } { I _ { q } ( u ) - \frac 1 3 - \frac { 2 } { 3 ( 2 q + 1 ) ^ { 2 } } } \end{array}$ has zero spatial mean and approximates $u ^ { 2 } - 1 / 3$ . Moreover,

$$
\int _ { - 1 } ^ { 1 } \left[ I _ { q } ( u ) - \frac { 1 } { 3 } - \frac { 2 } { 3 ( 2 q + 1 ) ^ { 2 } } - ( u ^ { 2 } - 1 / 3 ) \right] ^ { 2 } \frac { d u } { 2 } = \frac { 4 } { 4 5 ( 2 q + 1 ) ^ { 4 } } .
$$

To implement these two coordinate-wise interpolants, assign $q$ symmetric pairs of trainable neurons to the $x _ { 2 }$ -coordinate and another $q$ symmetric pairs to the $x _ { 3 }$ -coordinate. For

$1 \leq r \leq q$ , choose the corresponding hidden parameters as

$$
\theta _ { 2 r - 1 } ^ { ( 2 ) } = \Big ( { \bf e } _ { 2 } , - \frac { 2 r - 1 } { 2 q + 1 } \Big ) , \qquad \theta _ { 2 r } ^ { ( 2 ) } = \Big ( - { \bf e } _ { 2 } , - \frac { 2 r - 1 } { 2 q + 1 } \Big ) ,
$$

$$
\theta _ { 2 q + 2 r - 1 } ^ { ( 2 ) } = \Big ( \mathbf { e } _ { 3 } , - \frac { 2 r - 1 } { 2 q + 1 } \Big ) , \quad \theta _ { 2 q + 2 r } ^ { ( 2 ) } = \Big ( - \mathbf { e } _ { 3 } , - \frac { 2 r - 1 } { 2 q + 1 } \Big ) .
$$

Choose their output coeficients as $\begin{array} { r } { \beta _ { 2 r - 1 } ^ { ( 2 ) } = \beta _ { 2 r } ^ { ( 2 ) } = \frac { 4 } { 2 q + 1 } , \beta _ { 2 q + 2 r - 1 } ^ { ( 2 ) } = \beta _ { 2 q + 2 r } ^ { ( 2 ) } = \frac { 2 } { 2 q + 1 } } \end{array}$ , where the second pair of coeficients includes the factor $1 / \dot { 2 }$ multiplying the �<sub>3</sub>-component of the target. Set the output coeficients of all fixed neurons to zero. Finally, choose the output bias as

$$
\beta _ { 0 } ^ { ( 2 ) } = \Big ( 1 + \frac { 1 } { 2 } \Big ) \Big [ \frac { 1 } { ( 2 q + 1 ) ^ { 2 } } - \frac { 1 } { 3 } - \frac { 2 } { 3 ( 2 q + 1 ) ^ { 2 } } \Big ] = \frac { 1 - ( 2 q + 1 ) ^ { 2 } } { 2 ( 2 q + 1 ) ^ { 2 } } .
$$

For these parameters, the network output is

$$
f _ { 2 } ( x ) = I _ { q } ( x _ { 2 } ) - \frac { 1 } { 3 } - \frac { 2 } { 3 ( 2 q + 1 ) ^ { 2 } } + \frac { 1 } { 2 } \Bigl [ I _ { q } ( x _ { 3 } ) - \frac { 1 } { 3 } - \frac { 2 } { 3 ( 2 q + 1 ) ^ { 2 } } \Bigr ] .
$$

The two coordinate errors have zero spatial mean and depend on independent coordinates, so their cross term vanishes. Consequently,

$$
\| f _ { 2 } - f ^ { * } \| _ { L _ { 2 } ( d \mu ) } ^ { 2 } = \Big ( 1 + \frac { 1 } { 4 } \Big ) \frac { 4 } { 4 5 ( 2 q + 1 ) ^ { 4 } } = \frac { 1 } { 9 ( 2 q + 1 ) ^ { 4 } } .
$$

The infimum over the trainable hidden parameters and output coeficients is no larger than the error of this explicit choice. Since the construction is available for every realization of the fixed random neurons, taking expectation over those parameters gives

$$
\mathbb { E } \operatorname* { i n f } _ { \theta _ { \mathrm { f r e e } } , \beta } \lVert f ^ { * } - f _ { 2 } \rVert _ { L _ { 2 } ( d \mu ) } ^ { 2 } \le \mathcal { E } ^ { ( 2 ) } ( m _ { \mathrm { f r e e } } ) = \frac { 1 } { 9 ( m _ { \mathrm { f r e e } } / 2 + 1 ) ^ { 4 } } .
$$

In particular, $\begin{array} { r } { \mathcal { E } ^ { ( 2 ) } ( 4 ) = \frac { 1 } { 9 \cdot 3 ^ { 4 } } = \frac { 1 } { 7 2 9 } , \mathcal { E } ^ { ( 2 ) } ( 2 0 ) = \frac { 1 } { 9 \cdot 1 1 ^ { 4 } } } \end{array}$

For the case oftwo trainable hidden neurons, use one symmetric pair for the $x _ { 2 } – \mathrm { c o o r d i n a t e } . $ Taking $q = 1$ in the interpolation construction gives the centered approximation error $\textstyle { \frac { 4 } { 4 5 \cdot 3 ^ { 4 } } } = { \frac { 4 } { 3 6 4 5 } }$ for $x _ { 2 } ^ { 2 } - 1 / 3$ . Leaving the $x _ { 3 }$ -component unapproximated therefore yields

$$
\mathbb { E } \operatorname* { i n f } _ { \theta _ { \mathrm { f r e e } } , \beta } \big \| f _ { 2 } - f ^ { * } \big \| _ { L _ { 2 } ( d \mu ) } ^ { 2 } \leq \frac { 4 } { 3 6 4 5 } + \frac { 1 } { 4 5 } = \frac { 1 7 } { 7 2 9 } = \mathcal { E } ^ { ( 2 ) } ( 2 ) .
$$

For $m _ { \mathrm { f r e e } } \in \left\{ 2 , 4 \right\}$ , substitution gives $G ^ { ( 3 ) } ( 2 ) \approx 0 . 7 9 9 8 , G ^ { ( 3 ) } ( 4 ) \approx 0 . 9 9 9 7 , G ^ { ( 2 ) } ( 2 ) \approx$ $0 . 7 9 0 2 , G ^ { ( 2 ) } ( 4 ) \approx 0 . 9 8 7 7$

## E. Complementary Experiments

## E.1. Efect of the Initialization Scale

This experiment examines the dependence of the small-� transition on � for a fixed target in the spherical biased-ReLU RKHS in Section 4.

Target Function. Let $\mu = \operatorname { U n i f } ( \mathbb { S } ^ { 4 } )$ and $\rho = \operatorname { U n i f } ( \mathbb { S } ^ { 5 } )$ . Define

$$
Y _ { 2 } ( x ) = \sqrt { \frac { 3 5 } { 9 2 } } ( 4 x _ { 1 } ^ { 2 } + 2 x _ { 2 } ^ { 2 } + x _ { 3 } ^ { 2 } - 3 x _ { 4 } ^ { 2 } - 4 x _ { 5 } ^ { 2 } ) , \mathrm { ~ a n d ~ } Y _ { 4 } ( x ) = \sqrt { 3 4 6 5 } x _ { 1 } x _ { 2 } x _ { 3 } x _ { 4 } .
$$

These are orthonormal spherical harmonics of degrees 2 and 4. We take the target function

$$
f ^ { * } ( x ) = C _ { \xi } \bigl ( \sqrt { 0 . 9 } Y _ { 2 } ( x ) + \sqrt { 0 . 1 } Y _ { 4 } ( x ) \bigr ) , \mathrm { w i t h } C _ { \xi } = \bigl ( \frac { 0 . 9 } { \xi _ { 2 } } + \frac { 0 . 1 } { \xi _ { 4 } } \bigr ) ^ { - 1 / 2 } ,
$$

where $\xi _ { 2 }$ and $\xi _ { 4 }$ are the corresponding eigenvalues of $T _ { \rho }$ . By the spectral characterization of $\mathcal { H } _ { \rho }$ in (4.3) and the $L _ { 2 } ( d \mu )$ orthonormality of $Y _ { 2 }$ and $Y _ { 4 }$ , we have

$$
\| f ^ { * } \| _ { \mathcal { H } _ { \rho } } ^ { 2 } = \Big ( \frac { 0 . 9 } { \xi _ { 2 } } + \frac { 0 . 1 } { \xi _ { 4 } } \Big ) ^ { - 1 } \Big ( \frac { 0 . 9 } { \xi _ { 2 } } + \frac { 0 . 1 } { \xi _ { 4 } } \Big ) = 1 .
$$

Thus $\| f ^ { * } \| _ { \mathcal { H } _ { \rho } } = 1$ and $f ^ { * } \in \mathcal { H } _ { \rho }$ . The target and reference RKHS remain unchanged as � varies. Training uses $f ^ { * } / C _ { \xi }$ , with both the target and network output rescaled by $C _ { \xi }$ for reporting. This common scaling leaves the normalized gain unchanged.

## Experimental Set-up.

• Network and Parameter Grid. We use a two-layer ReLU network of width 64, with

$$
P \in { \frac { 1 } { 6 4 } } { \Bigl \{ } 0 , \ 1 , \ 2 , \ . . . , \ 1 9 , \ 2 0 , \ 2 4 , \ 3 2 , \ 4 8 , \ 6 4 { \Bigr \} } ,
$$

and �  1, 1.2, 1.5, 2, 2.5, 3, 3.5, 4, 5, 7.5, 8, 9, 10 .

• Initialization Distributions. Fixed neurons are initialized from $\rho _ { 0 } ^ { ( \tau ) }$ , while trainable neurons use a numerical approximation to the ridge leverage distribution in (3.8). We use $\lambda \approx 4 . 5 2 4 4 \times 1 0 ^ { - 4 }$ so that $d ( \lambda ) \approx 6 4 / \log 6 4$ for all $P$ and $\tau .$ In the spherical setting, $s _ { \lambda } ( \theta )$ depends only on the bias $b ,$ , so the bias density is proportional to $p ( b ) s _ { \lambda } ( \theta )$ and then normalized. The leverage score $s _ { \lambda } ( \theta )$ is evaluated using harmonic degrees up to 32, with the omitted tail replaced by the remaining feature energy divided by � as an upper bound. The bias density is evaluated on 4097 grid points. We mix the approximate leverage distribution with the uniform reference distribution $\rho _ { ; }$ , assigning weight $1 0 ^ { - 3 }$ to $\rho$ to ensure positivity. Trainable hidden parameters are unconstrained during training, and all output parameters are also optimized.

• Training and Repetitions. Each configuration is trained on the same 12,000 independent samples from $\mu$ for 12,000 full-batch Adam steps and repeated 21 times. The remaining optimizer settings follow Table 2. Within each repetition, the two independent initialization pools are shared across $P$ and �. The full-training result is reused across �, since no hidden neurons remain fixed at $P = 1$

Metrics. The empirical gain uses the same geometric mean training loss and normalization as in Section 6. The numerical threshold $P _ { 0 . 1 }$ is the smallest tested � satisfying $G ( P ) \ge 0 . 9$ Pointwise $9 5 \%$ intervals are obtained from 500 paired bootstrap resamples across repetitions. For $d = 5$ , the theoretical curves use (3.12) with $\alpha = 7 / 4$ and $r = 2 4 \sqrt { 2 } \tau / 5$

Results and Interpretation. Figure 1 is reproduced below. The numerical threshold $P _ { 0 . 1 }$ decreases overall as � increases. The gain curve for $\tau = 1 0$ also rises more rapidly at small � than that for $\tau = 1$ . These observations are qualitatively consistent with the theoretical trend. The theoretical curves are derived from the approximation upper bound over the RKHS unit ball in Theorem 4.3 with $d = 5$ , whereas the numerical curves describe training performance for one fixed target $f ^ { * }$ in that ball under finite sampling and a fixed training budget. Quantitative diferences between the curves are therefore expected. Nevertheless, the numerical results exhibit the same overall dependence on �, providing empirical support for the theoretical prediction within the scaled spherical family.

![](images/36ce1fd1d6794b7674133779b03ca8263412d28aabb2be0d7c15c1d3a597ea34.jpg)  
(a) Threshold for 90% gain.

![](images/b2f88c7b920c8799495a1984333c41e039d96069b3aa5adcad6b972d90e2b0d1.jpg)  
(b) Normalized gain.  
Figure 1 (reproduced). Theoretical and numerical phase transitions for two-layer spherical biased-ReLU �-trainable networks with $d = 5$ . Red curves are derived from the approximation bound, while blue curves show illustrative training results for a fixed RKHS target. Threshold $P _ { 0 . 1 }$ marks 90% normalized gain. Shaded bands indicate 95% bootstrap intervals.

## E.2. Additional Experiments with Other Activation Functions

This section presents the width and depth experiments with GELU and tanh, using the same targets, architectures, and training settings as in Section 6.1. Figures 13 to 18 below show small-� phase transition patterns similar to those observed with ReLU. For GELU on $f _ { 2 } ^ { * }$ , the two-layer network also shows normalized gains slightly above 1, so we omit this gain curve. Partial training attains a slightly lower loss than full training at some � under the fixed training budget. The corresponding diferences in absolute training loss are only of order $1 0 ^ { - 5 }$ , representing small fluctuations that do not alter the overall phase transition pattern.

![](images/8adf2ab69027df712c1e14588c0397b48fec4d92c0c883e3f7958b647dcef98d.jpg)  
(a) Depth variation loss.

![](images/e7ccc71d1d4decf3992abfd3f9d271c0cf1708dec0e5cb8e58ed70353ce438ec.jpg)  
(b) Depth variation gain.

![](images/b1cd30f0572c37a127e86698e5896b33cdd617d308d02c13a4a27a3453f63314.jpg)  
(c) Width variation loss.

![](images/02b4a798a77bede9fee5d7547b3d7cbe109a8f589824243defc7081ce2b21647.jpg)  
(d) Width variation gain.

Figure 13. Training results for $f _ { 1 } ^ { * }$ with GELU activation.  
![](images/2071b1e37da129f9ea9b917aa629476173d50e63372a721ad2651b671dc27e78.jpg)  
(a) Depth variation loss.

![](images/e28533d9bce6ce436b0e31420d130c7154a858350c4e31ba589f5e2abcf33565.jpg)  
(b) Depth variation gain.

![](images/f848b470c12ea1f533e7e7d078514dc0e2a54041a05877ebef06c5fb23a146bf.jpg)  
(c) Width variation loss.

![](images/ebe493e0934b579e19845c7511fb87ae5e967f0f0e4850b941155af65b05ffc4.jpg)  
(d) Width variation gain.  
Figure 14. Training results for $f _ { 2 } ^ { * }$ with GELU activation.

![](images/743185e593cb5ed475cb1b8c7ea537638a35dedaf46f0d0deb366421b8c9fab4.jpg)  
(a) Depth variation loss.

![](images/c6c17b1d778e9274f541924907191932bcc0fd69bf3c57335a1072c85a3a6292.jpg)  
(b) Depth variation gain.

![](images/68ebf60256897cdcd5fad174d44df0f89b2263b4e221ae9f4d0c570f57d72d50.jpg)  
(c) Width variation loss.

![](images/19b2cd67596e0a0f2f5d8eb4cfff95d7b0b8ae7ba9f7a7f99ed7c94e5195757d.jpg)  
(d) Width variation gain.

Figure 15. Training results for $f _ { 3 } ^ { * }$ with GELU activation.  
![](images/5f062e254447d171ea095e92cd1eb8537fb9d18e7844174e0e401460abe4789d.jpg)  
(a) Depth variation loss.

![](images/98de967fee6108ae618fbeb3c059fefee8a9accce300e32f6b3828ac27432c79.jpg)

![](images/34c5f97648f7c12694ac6fc28f6217c4311b33ff521b8def2703828ff9f9ee82.jpg)  
(c) Width variation loss.

(b) Depth variation gain.  
![](images/ebdd6b95cc4d29a6f5d6914e296f2674b8221e3982ca63ba16ad470e3f4c58da.jpg)  
(d) Width variation gain.  
Figure 16. Training results for $f _ { 1 } ^ { * }$ with tanh activation.

![](images/aa30319b471536f178f643f217ff26cf17ce309e739c0d73c929ea5247bbd425.jpg)  
(a) Depth variation loss.

![](images/c027ecc9ec886f3d283963b49869560637367143ce8f2e0aa82ae83a8ccba83d.jpg)  
(b) Depth variation gain.

![](images/05a6d8de3e0381d8b62ea7dc2cc00b5a32f3a9d3d2311ce28a64fbdd2075b601.jpg)  
(c) Width variation loss.

![](images/a87aea4d5011228ce3b86d9d40c31c4ac6cc8adf1ac7b2c556e27e87bd09624e.jpg)  
(d) Width variation gain.

Figure 17. Training results for $f _ { 2 } ^ { * }$ with tanh activation.  
![](images/5f278681a90190a6078607cedf69a8339d3a2e984c0f1d3835ef3f63d72e56b1.jpg)  
(a) Depth variation loss.

![](images/5117694af38eb91b52783ac19b10bf15523dcc365ddd96d51b02a9078af606a7.jpg)  
(b) Depth variation gain.

![](images/f001ad5c11f2758fa3de2ff073a4d52e347986c30cf4a53cad2f01af9d8b3b4f.jpg)  
(c) Width variation loss.

![](images/79153049ff529d145aead0cf7bad2b099e628246e4180f04e7a096f17f5d2790.jpg)  
(d) Width variation gain.  
Figure 18. Training results for $f _ { 3 } ^ { * }$ with tanh activation.

Acknowledgments. The authors used ChatGPT (OpenAI) for language polishing and assistance in checking potential errors. The authors take full responsibility for the content of this article.

## References

[1] K. Atkinson and W. Han, Spherical harmonics and approximations on the unit sphere: an introduction. Lecture Notes in Mathematics 2044, Springer Heidelberg, 2012

[2] H. Avron, M. Kapralov, C. Musco, C. Musco, A. Velingker, and A. Zandieh, Random Fourier features for kernel ridge regression: Approximation bounds and statistical guarantees. In International Conference on Machine Learning, pp. 253–262, PMLR, 2017

[3] F. Bach, Breaking the curse of dimensionality with convex neural networks. Journal ofMachine Learning Research 18 (2017), no. 19, 1–53

[4] F. Bach, On the equivalence between kernel quadrature rules and random feature expansions. Journal ofMachine Learning Research 18 (2017), no. 21, 1–38

[5] A. R. Barron, Universal approximation bounds for superpositions of a sigmoidal function. IEEE Transactions on Information Theory 39 (1993), no. 3, 930–945

[6] J.-F. Cai, D. Li, J. Sun, and K. Wang, Enhanced expressive power and fast training of neural networks by random projections. CSIAM Transactions on Applied Mathematics 2 (2021), 532– 550

[7] M. Chen, H. Jiang, W. Liao, and T. Zhao, Eficient approximation of deep ReLU networks for functions on low dimensional manifolds. Advances in Neural Information Processing Systems 32 (2019)

[8] Z. Chen, E. Vanden-Eijnden, and J. Bruna, On feature learning in neural networks with globa convergence guarantees. In International Conference on Learning Representations, 2022

[9] Z. Chen, E. Vanden-Eijnden, and J. Bruna, A functional-space mean-field theory of partiallytrained three-layer neural networks. Journal ofMachine Learning Research 27 (2026), no. 52, 1–67

[10] J. Cheng, Q. Li, T. Lin, and Z. Shen, Interpolation, approximation, and controllability of deep neural networks. SIAM Journal on Control and Optimization 63 (2025), no. 1, 625–649

[11] J. Cheng, T. Lin, Z. Shen, and Q. Li, A unified framework for establishing the universal approximation of transformer-type architectures. Advances in Neural Information Processing Systems 38 (2025), 129131–129163

[12] G. Cybenko, Approximation by superpositions ofa sigmoidal function. Mathematics ofControl, Signals and Systems 2 (1989), no. 4, 303–314

[13] R. DeVore, R. D. Nowak, R. Parhi, and J. W. Siegel, Weighted variation spaces and approximation by shallow ReLU networks. Applied and Computational Harmonic Analysis 74 (2025), 101713

[14] W. E, C. Ma, and L. Wu, The Barron space and the flow-induced function spaces for neural network models. Constructive Approximation 55 (2022), 369–406

[15] W. E and Q. Wang, Exponential convergence of the deep neural network approximation for analytic functions. Science China Mathematics 61 (2018), 1733–1740

[16] A. El Alaoui and M. W. Mahoney, Fast randomized kernel ridge regression with statistical guarantees. Advances in Neural Information Processing Systems 28 (2015)

[17] J. Frankle and M. Carbin, The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. In International Conference on Learning Representations, 2019

[18] X. Glorot and Y. Bengio, Understanding the dificulty of training deep feedforward neural networks. In Proceedings ofthe Thirteenth International Conference on Artificial Intelligence and Statistics, pp. 249–256, JMLR Workshop and Conference Proceedings, 2010

[19] L. Gonon, L. Grigoryeva, and J.-P. Ortega, Approximation bounds for random neural networks and reservoir systems. The Annals ofApplied Probability 33 (2023), no. 1, 28–69

[20] R. Gribonval, G. Kutyniok, M. Nielsen, and F. Voigtlaender, Approximation spaces of deep neural networks. Constructive Approximation 55 (2022), 259–367

[21] I. Gühring, G. Kutyniok, and P. Petersen, Error bounds for approximations with deep ReLU neural networks in ��, � norms. Analysis and Applications 18 (2020), no. 05, 803–859

[22] J. He, L. Li, and J. Xu, ReLU deep neural networks from the hierarchical basis perspective. Computers & Mathematics with Applications 120 (2022), 105–114

[23] J. He, L. Li, J. Xu, and C. Zheng, Relu deep neural networks and linear finite elements. Journal ofComputational Mathematics 38 (2020), no. 3, pp. 502–527

[24] K. Hornik, M. Stinchcombe, and H. White, Multilayer feedforward networks are universal approximators. Neural Networks 2 (1989), no. 5, 359–366

[25] D. Hsu, C. H. Sanford, R. Servedio, and E. V. Vlatakis-Gkaragkounis, On the approximation power of two-layer networks of random ReLUs. In Conference on Learning Theory, pp. 2423– 2461, PMLR, 2021

[26] Y. Jiao, Y. Lai, X. Lu, F. Wang, J. Z. Yang, and Y. Yang, Deep neural networks with ReLUsine-exponential activations break curse of dimensionality in approximation on Hölder class. SIAM Journal on Mathematical Analysis 55 (2023), no. 4, 3635–3649

[27] K. Kawaguchi, B. Xie, and L. Song, Deep semi-random features for nonlinear function approximation. In Proceedings ofthe AAAI Conference on Artificial Intelligence, 32, 2018

[28] H. König, Eigenvalues of compact operators with applications to integral operators. Linear Algebra and its Applications 84 (1986), 111–122

[29] M. Leshno, V. Y. Lin, A. Pinkus, and S. Schocken, Multilayer feedforward networks with a nonpolynomial activation function can approximate any function. Neural Networks 6 (1993), no. 6, 861–867

[30] B. Li, Z. Shen, H. Yang, and S. Zhang, Optimal Tradeofs Between Network Size and Parameter Magnitude in Neural Approximation and Minimax Regression. arXiv e-prints (2026), arXiv:2609.25710

[31] B. Li, H. Yang, and S. Zhang, Sobolev Approximation by Fixed-Size Neural Networks with Arbitrary Accuracy. arXiv e-prints (2026), arXiv:2606.16975

[32] M. Li, S. Sonoda, F. Cao, Y. G. Wang, and J. Liang, How powerful are shallow neural networks with bandlimited random weights? In International Conference on Machine Learning, pp. 19960–19981, PMLR, 2023

[33] Q. Li, T. Lin, and Z. Shen, Deep learning via dynamical systems: An approximation perspective. Journal ofthe European Mathematical Society 25 (2023), no. 5, 1671–1709

[34] Z. Li, J.-F. Ton, D. Oglic, and D. Sejdinovic, Towards a unified analysis of random Fourier features. In International Conference on Machine Learning, pp. 3905–3914, PMLR, 2019

[35] F. Liu, X. Huang, Y. Chen, J. Yang, and J. Suykens, Random Fourier features via fast surrogate leverage weighted sampling. In Proceedings ofthe AAAI Conference on Artificial Intelligence, 34, 2020

[36] J. Long, X. Peng, and L. Wu, Optimal rates and saturation for noiseless kernel ridge regression. arXiv preprint arXiv:2402.15718 (2024)

[37] J. Lu, Z. Shen, H. Yang, and S. Zhang, Deep network approximation for smooth functions. SIAM Journal on Mathematical Analysis 53 (2021), no. 5, 5465–5506

[38] L. Ma, J. W. Siegel, and J. Xu, Uniform approximation rates and metric entropy of shallow neural networks. Research in the Mathematical Sciences 9 (2022), no. 3, 46

[39] E. Malach, G. Yehudai, S. Shalev-Schwartz, and O. Shamir, Proving the lottery ticket hypothesis: Pruning is all you need. In International Conference on Machine Learning, pp. 6682–6691, PMLR, 2020

[40] T. Mao, J. W. Siegel, and J. Xu, Approximation rates for shallow $\scriptstyle { \mathrm { R e L U } } ^ { k }$ neural networks on Sobolev spaces via the Radon transform. SIAM Journal on Mathematical Analysis 58 (2026), no. 2, 1171–1186

[41] G. Mastroianni and G. V. Milovanović, Interpolation processes: Basic theory and applications. Springer, 2008

[42] S. Minsker, On some extensions of Bernstein’s inequality for self-adjoint operators. Statistics & Probability Letters 127 (2017), 111–119

[43] H. Montanelli, H. Yang, and Q. Du, Deep ReLU networks overcome the curse of dimensionality for bandlimited functions. Journal ofComputational Mathematics 39 (2021), no. 6, 801–815

[44] N. H. Nelsen and A. M. Stuart, Operator learning using random features: A tool for scientific computing. SIAM Review 66 (2024), no. 3, 535–571

[45] Y.-H. Pao, G.-H. Park, and D. J. Sobajic, Learning and generalization characteristics of the random vector functional-link net. Neurocomputing 6 (1994), no. 2, 163–180

[46] A. Pinkus, Approximation theory of the MLP model in neural networks. Acta Numerica 8 (1999), 143–195

[47] R. Pyke, Spacings. Journal ofthe Royal Statistical Society: Series B (Methodological) 27 (1965), no. 3, 395–436

[48] A. Rahimi and B. Recht, Random features for large-scale kernel machines. Advances in Neural Information Processing Systems 20 (2007)

[49] A. Rahimi and B. Recht, Weighted sums of random kitchen sinks: Replacing minimization with randomization in learning. Advances in Neural Information Processing Systems 21 (2008)

[50] V. Ramanujan, M. Wortsman, A. Kembhavi, A. Farhadi, and M. Rastegari, What’s hidden in a randomly weighted neural network? In 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 11890–11899, IEEE, 2020

[51] A. Rosenfeld and J. K. Tsotsos, Intriguing properties of randomly weighted networks: Generalizing while learning next to nothing. In 2019 16th conference on Computer and Robot Vision (CRV), pp. 9–16, IEEE, 2019

[52] A. Rudi and L. Rosasco, Generalization properties of learning with random features. Advances in Neural Information Processing Systems 30 (2017)

[53] U. Shaham, A. Cloninger, and R. R. Coifman, Provable approximation properties for deep neura networks. Applied and Computational Harmonic Analysis 44 (2018), no. 3, 537–557

[54] G. Shen, Y. Jiao, Y. Lin, and J. Huang, Approximation with CNNs in sobolev space: with applications to classification. In Advances in neural information processing systems, edited by S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, pp. 2876–2888, 35, Curran Associates, Inc., 2022

[55] Z. Shen, H. Yang, and S. Zhang, Deep network approximation characterized by number of neurons. Communications in Computational Physics 28 (2020), no. 5, 1768–1811

[56] Z. Shen, H. Yang, and S. Zhang, Neural network approximation: Three hidden layers are enough. Neural Networks 141 (2021), 160–173

[57] Z. Shen, H. Yang, and S. Zhang, Deep network approximation: Achieving arbitrary accuracy with fixed number of neurons. Journal ofMachine Learning Research 23 (2022), no. 276, 1–60

[58] Z. Shen, H. Yang, and S. Zhang, Deep network approximation in terms of intrinsic parameters. In Proceedings ofthe 39th international conference on machine learning, edited by K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvari, G. Niu, and S. Sabato, pp. 19909–19934, Proceedings of Machine Learning Research 162, PMLR, 2022

[59] Z. Shen, H. Yang, and S. Zhang, Optimal approximation rate of ReLU networks in terms of width and depth. Journal de Mathématiques Pures et Appliquées 157 (2022), 101–135

[60] J. W. Siegel, Optimal approximation rates for deep ReLU neural networks on Sobolev and Besov spaces. Journal ofMachine Learning Research 24 (2023), no. 357, 1–52

[61] J. W. Siegel and J. Xu, Approximation rates for neural networks with general activation functions. Neural Networks 128 (2020), 313–321

[62] J. W. Siegel and J. Xu, High-order approximation rates for shallow neural networks with cosine and ReLU<sup>�</sup> activation functions. Applied and Computational Harmonic Analysis 58 (2022), 1–26

[63] J. W. Siegel and J. Xu, Characterization of the variation spaces corresponding to shallow neural networks. Constructive Approximation 57 (2023), no. 3, 1109–1132

[64] J. W. Siegel and J. Xu, Sharp bounds on the approximation rates, metric entropy, and n-widths of shallow neural networks. Foundations ofComputational Mathematics 24 (2024), no. 2, 481– 537

[65] B. Simon, Trace ideals and their applications. Cambridge University Press, 1979

[66] L. Song, Y. Liu, J. Fan, and D.-X. Zhou, Approximation ofsmooth functionals using deep ReLU networks. Neural Networks 166 (2023), 424–436

[67] B. Sriperumbudur and Z. Szabó, Optimal rates for random Fourier features. Advances in Neural Information Processing Systems 28 (2015)

[68] D. J. Sutherland and J. Schneider, On the error of random Fourier features. In Proceedings ofthe Thirty-First Conference on Uncertainty in Artificial Intelligence, pp. 862–871, UAI’15, AUAI Press, Arlington, Virginia, USA, 2015

[69] T. Suzuki, Adaptivity of deep ReLU network for learning in Besov and mixed smooth Besov spaces: optimal rate and curse of dimensionality. In International Conference on Learning Representations, 2019

[70] G. Szegő, Orthogonal polynomials. American Mathematical Society Colloquium Publications 23, American Mathematical Society, 1939

[71] E. Williams, A. Payeur, A. Ryoo, T. Jiralerspong, M. Perich, L. Mazzucato, and G. Lajoie, Expressivity of neural networks with random weights and learned biases. In International Conference on Learning Representations, pp. 46033–46058, 2025

[72] P. Wimmer, J. Mehnert, and A. Condurache, FreezeNet: Full performance by reduced storage costs. In Asian Conference on Computer Vision, pp. 685–701, Springer, 2021

[73] D. Yarotsky, Error bounds for approximations with deep ReLU networks. Neural Networks 94 (2017), 103–114

[74] G. Yehudai and O. Shamir, On the power and limitations of random features for understanding neural networks. Advances in Neural Information Processing Systems 32 (2019)

[75] S. Zhang, Sharp Approximation Rates for Neural Networks with Afine Latent Parameterizations. arXiv e-prints (2026), arXiv:2608.31157

[76] S. Zhang, J. Lu, and H. Zhao, Deep network approximation: Beyond ReLU to diverse activation functions. Journal ofMachine Learning Research 25 (2024), no. 35, 1–39

[77] S. Zhang, Z. Shen, and Y. Xu, Layer-wise Geometric Approximation Rates for Deep Networks. arXiv e-prints (2026), arXiv:2604.20219

[78] S. Zhang, Z. Shen, and Y. Xu, Multigrade Neural Network Approximation. arXiv e-prints (2026), arXiv:2601.16884

[79] S. Zhang, H. Zhao, Y. Zhong, and H. Zhou, Structured and balanced multicomponent and multilayer neural networks. SIAM Journal on Scientific Computing 47 (2025), no. 5, C1059–C1090

[80] G. Zhong, L.-N. Wang, X. Ling, and J. Dong, An overview on data representation learning: From traditional feature learning to recent deep learning. The Journal of Finance and Data Science 2 (2016), no. 4, 265–278

[81] D.-X. Zhou, Universality of deep convolutional neural networks. Applied and Computational Harmonic Analysis 48 (2020), no. 2, 787–794

## Xin He

School of Mathematics and Statistics, Wuhan University, Bayi Road 299, 430072 Wuhan, China; xin\_he@whu.edu.cn

## Huan Lao

School of Mathematics and Statistics, Wuhan University, Bayi Road 299, 430072 Wuhan, China; laohuan0618@whu.edu.cn

## Xiliang Lu

School of Mathematics and Statistics, Wuhan University, Bayi Road 299, 430072 Wuhan; Computational Science Hubei Key Laboratory, Wuhan University, 430072 Wuhan, China; xllv.math@whu.edu.cn

## Shijun Zhang (corresponding author)

Department of Applied Mathematics, The Hong Kong Polytechnic University, Hung Hom, Kowloon, Hong Kong SAR, China; shijun.zhang@polyu.edu.hk