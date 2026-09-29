# Context-dependent time-series prediction via HyperReservoirs

Kohei Tsuchiyama, Takatomo Mihana, Ryoichi Horisaki, and Andr´e R¨ohm

Department of Information Physics and Computing,

Graduate School of Information Science and Technology,

The University of Tokyo, 7-3-1 Hongo, Bunkyo-ku, Tokyo 113-8656, Japan

(Dated: September 2026)

Time series prediction is a common application of reservoir computing. When the training and testing time series data contains multiple dynamical regimes, because an underlying parameter is changing, or the data in fact consists of multiple distinct systems, simple application of the reservoir computing principle produces high prediction errors. Here, we propose a HyperReservoir as an extended model of reservoir computing especially designed for such cases. The HyperReservoir combines a main reservoir with a smaller context reservoir, where the latter modulates the output weights of the former. This structure resembles the hypernetworks from deep neural network literature. However, in contrast, HyperReservoirs retain the simple training via linear regression of standard reservoir computing. We compare the proposed architecture with a conventional ESN, in which context acts at the input, and a full-matrix Conceptor, in which context modulates the reservoir state space. We evaluate all three models on time-series prediction tasks based on Lorenz and R¨ossler systems, including for varying bifurcation parameters and time sampling scales. We find that the HyperReservoir achieves the lowest mean test error in all three tasks, and particularly outperforms conceptors on data that is sampled from the same attractor but at diferent time scales.

## I. INTRODUCTION

High-dimensional recurrent dynamics provide a useful basis for representing and transforming temporal signals. Reservoir computing makes use of this principle by retaining a high-dimensional dynamical system as a fixed nonlinear substrate while training only a comparatively simple output transformation [1, 2]. This restriction reduces the need for recurrent optimization and is particularly attractive when the underlying dynamics are readily available but dificult or costly to modify, as in many physical implementations [3–6]. However, a largely fixed recurrent substrate also limits how much the computation can be adapted to changing conditions.

Many computational systems must operate across multiple tasks, environmental conditions, or dynamical regimes rather than perform a single fixed computation [7–9]. Reusing a common recurrent substrate across such conditions can allow a single recurrent system to support multiple dynamical or computational regimes [9, 10]. Flexible behavior in biological and artificial recurrent networks likewise relies on shared neural dynamics being recruited diferently depending on task or context [7, 8, 11]. Importantly, similar observations or internal representations can require diferent responses under diferent contexts [11]. Therefore, contextual information is not merely an additional input variable; it specifies how a shared computational substrate should be used for the currently relevant mapping.

A natural question is then how contextual flexibility can be introduced without discarding recurrent structure that remains useful across regimes. In particular, when a common recurrent representation remains informative across contexts, it may be advantageous to preserve that representation and adapt only the mapping from the representation to the required output. This suggests a dis tinction between context-dependent modification of the recurrent representation itself and context-dependent interpretation of an otherwise shared representation. Con text can in principle act at several stages of a recurrent computation. It may be supplied as an external input, as in parameter-aware or multifunctional reservoir computing [10, 12, 13]; it may restrict the accessible region of reservoir state space, as in Conceptors [14]; or it may alter how a shared reservoir representation is mapped to the output, as in reservoir architectures with multiple readouts [15, 16]. Related work has also extended reservoir readouts through nonlinear combinations of reservoir variables [17]. These alternatives impose diferent structural assumptions because they determine which parts of the computation remain shared across dynamical regimes and which are allowed to change with context. The central issue is therefore not simply whether contextual information is available, but which components of the recurrent computation should remain common and where context-dependent degrees of freedom should be introduced.

This distinction becomes especially important when the target regimes cannot be readily distinguished from state-space geometry alone. If a context change generates geometrically distinct attractors, the occupied regions of state space already provide substantial information about the active regime [14, 18]. This becomes more dificult when attractors from diferent contexts are similar, and is particularly restrictive when they share the same underlying attractor geometry. A key example is temporal scaling. Temporal scaling has been studied in recurrent neural dynamics, where similar or approximately invariant trajectories can be traversed at diferent speeds [19, 20].

For example, consider two systems that difer only by a temporal-scale factor,

$$
\dot { \mathbf { s } } = \nu _ { c } \mathbf { f } ( \mathbf { s } ) ,\tag{1}
$$

where $\nu _ { c }$ depends on context. Changing $\nu _ { c }$ changes the rate at which a trajectory advances through the attractor without changing the underlying continuous-time orbit geometry. Consequently, the principal contextual distinction is not which region of state space is occupied, but how rapidly the state evolves through that region. For future-state prediction, similar current states can therefore require diferent outputs depending on the temporal context. In such a setting, the challenge is not necessarily to construct a diferent recurrent representation for every context, but to map a substantially shared representation to diferent context-dependent targets.

Motivated by this distinction, we propose the HyperReservoir, a reservoir-computing architecture for context-dependent decoding of shared recurrent dynamics. The architecture retains a common main reservoir that represents the observed dynamics, while a smaller context reservoir encodes the supplied context and mod ulates the readout applied to the main-reservoir features. Separate reservoir pathways for measurement and regime-parameter sequences have previously been used for multi-regime time-series prediction [21]. Here, the two representations are instead coupled through a bilinear readout, so that the context-reservoir state parameterizes the efective mapping from shared main-reservoir features to the output. This interpretation is closely related to Hypernetworks, which provide a general framework for making selected parameters of a computation depend on contextual information [22]. Importantly, both recurrent subsystems remain fixed, and the complete readout is fitted by linear regression.

To evaluate the proposed readout-level modulation, we compare the HyperReservoir with two established alternatives that introduce the same contextual information at diferent stages of reservoir computation: a conventional context-input ESN, in which context enters through the reservoir input [10, 23], and full-matrix Conceptors, in which context acts through state-space restriction [14]. Together, the three architectures provide input-, state-, and readout-level forms of contextual adaptation.

We evaluate these architectures on three future-state prediction settings: geometrically distinct Lorenz and R¨ossler systems, related R¨ossler regimes, and the same R¨ossler attractor traversed at diferent temporal scales. These settings range from clearly distinct state-space geometries to a case in which the underlying continuoustime attractor geometry is shared while the required finite-time prediction remains context dependent.

Across all three settings, the augmented HyperReservoir achieves lower mean prediction error than the conventional context-input ESN, with larger diferences for the related and shared-attractor regimes. Additional analyses examine the functional use of the supplied context and the contribution of the context-dependent read out structure. Together, the results support contextdependent decoding as a useful mechanism when recurrent features can remain substantially shared across regimes while the required predictive mapping changes with context.

## II. CONTEXTUAL ADAPTATION IN RESERVOIR COMPUTING

We now define the three contextual reservoir architectures compared in this study. They receive the same observed variables and explicit contextual information, but difer in where that context acts on the computation. Figure 1 summarizes the context-input ESN, the full-matrix Conceptor, and the proposed HyperReservoir. In the conventional context-input echo state network (ESN), context modifies the external input to the reservoir. In the Conceptor model, context selects a state-space operator that acts on the reservoir state. In the HyperReservoir, context instead modifies the readout applied to a shared reservoir representation.

## A. Context-input echo state network

We first consider a conventional ESN. Its reservoir state $\mathbf { x } _ { t } \in \mathbb { R } ^ { N }$ evolves according to

$$
\mathbf { x } _ { t + 1 } = \mathbf { x } _ { t } + \alpha \left[ - \mathbf { x } _ { t } + J \phi ( \mathbf { x } _ { t } ) + W _ { \mathrm { i n } } \mathbf { u } _ { t } \right] ,\tag{2}
$$

and the prediction is obtained from the linear readout,

$$
\widehat { \mathbf { y } } _ { t } = W _ { \mathrm { o u t } } \mathbf { x } _ { t } + \mathbf { b } _ { \mathrm { o u t } } .\tag{3}
$$

Here, $J \in \mathbb { R } ^ { N \times N }$ is the recurrent matrix, $W _ { \mathrm { i n } }$ is the input matrix, α is the leak coeficient, and $\phi ( \cdot )$ is an elementwise nonlinearity. The recurrent and input matrices are randomly initialized and remain fixed, whereas the output coeficients are fitted by ridge regression.

For the context-input baseline, $\mathbf { u } _ { t }$ denotes the input available to the reservoir, including both the observed dynamical state and the explicit context signal. Thus, context afects prediction through the reservoir state, while the recurrent matrix and output transformation remain shared across contexts (see Fig. 1a).

## B. Conceptor-based state modulation

A diferent strategy is to introduce context directly at the level of the reservoir state. Conceptors associate each stored pattern or context with a linear operator constructed from the corresponding reservoir-state distribution [14].

In this study, we use the full-matrix form of the Conceptor, so that each context is represented by a matrix $\mathbf { C } _ { k } \in \mathbb { R } ^ { N \times N }$ acting on the complete reservoir state. Unlike a diagonal state-wise scaling, the full-matrix operator can filter arbitrary directions in reservoir state space, including directions defined by correlations among diferent reservoir units.

a  
b  
![](images/6c344be098c2c897978dd67d891822992b719cde67784e8ee623e8711481cc3b.jpg)  
FIG. 1. Three mechanisms for introducing contextual information into a shared reservoir system. (a) In the context-input ESN, context enters through the external input while the recurrent operator and readout remain shared. (b) In the full-matrix Conceptor model, context selects a state-space operator $C _ { k }$ that filters the provisional reservoir state. (c) In the HyperReservoir, an additional context reservoir encodes the supplied context, and the bilinear term $\mathbf { h } _ { t } ^ { \mathrm { H } } \otimes \mathbf { h } _ { t } ^ { \mathrm { R } }$ allows this context representation to modulate the coeficients applied to the shared main-reservoir features. The recurrent matrices remain fixed in all three architectures.

Let $\overline { { \mathbf { x } } } _ { t + 1 }$ denote the provisional state obtained from the same fixed reservoir dynamics as in Eq. (2), with only the observed dynamical variables supplied to the reservoir. For context k, the empirical state-correlation matrix is

$$
R _ { k } = \big \langle \overline { { \mathbf { x } } } _ { t } \overline { { \mathbf { x } } } _ { t } ^ { \mathsf { T } } \big \rangle _ { t \in \mathcal { T } _ { k } } ,\tag{4}
$$

where $\mathcal { T } _ { k }$ contains the training samples associated with that context. The corresponding full-matrix Conceptor

is

$$
C _ { k } = { { R } _ { k } } \left( { { R } _ { k } } + { \gamma } ^ { - 2 } I \right) ^ { - 1 } ,\tag{5}
$$

where $\gamma > 0$ is the aperture parameter.

During prediction, the context selects the corresponding Conceptor and the provisional reservoir state is filtered according to

$$
\mathbf { x } _ { t + 1 } = C _ { k } \overline { { \mathbf { x } } } _ { t + 1 } .\tag{6}
$$

The output is then obtained from the shared readout

$$
\widehat { \mathbf { y } } _ { t } = W _ { \mathrm { o u t } } \mathbf { x } _ { t } + \mathbf { b } _ { \mathrm { o u t } } .\tag{7}
$$

The eigenvalues of $C _ { k }$ lie between zero and one, so the operator acts as a soft state-space filter that retains directions strongly represented by the training states while attenuating weakly represented directions. The aperture γ controls the strength of this restriction. Each $C _ { k }$ is estimated only from training reservoir states, and a single output matrix is fitted to the Conceptor-filtered states. Thus, contextual dependence enters through the state representation while the readout remains shared (see Fig. 1b).

## C. Proposed HyperReservoir for context-dependent readout modulation

We next consider readout-level contextual adaptation. Rather than modifying the main-reservoir state, the HyperReservoir keeps the main recurrent dynamics fixed and allows context to change how its representation is decoded. Related reservoir-computing approaches have used multiple readout modules [15, 16], generalized nonlinear readouts [17], and separate reservoirs for measurement and regime-parameter sequences [21]. The Hyper-Reservoir uses a specific combination of these ideas: observation and context are processed separately, and their interaction enters through a bilinear readout.

The HyperReservoir contains two fixed recurrent subsystems. The main reservoir, with state $\mathbf { h } _ { t } ^ { \mathrm { R } } \in \mathbb { R } ^ { N }$ , is driven by the observed dynamical state:

$$
{ \bf h } _ { t + 1 } ^ { \mathrm { R } } = { \bf h } _ { t } ^ { \mathrm { R } } + \alpha _ { \mathrm { R } } \left[ - { \bf h } _ { t } ^ { \mathrm { R } } + J ^ { \mathrm { R } } \phi ( { \bf h } _ { t } ^ { \mathrm { R } } ) + W _ { \mathrm { i n } } ^ { \mathrm { R } } { \bf u } _ { t } \right] .\tag{8}
$$

The context reservoir, with state $\mathbf { h } _ { t } ^ { \mathrm { H } } \in \mathbb { R } ^ { M }$ , is driven only by the explicit contextual signal $\mathbf { c } _ { t } \mathbf { : }$

$$
{ \bf h } _ { t + 1 } ^ { \mathrm { H } } = { \bf h } _ { t } ^ { \mathrm { H } } + \alpha _ { \mathrm { H } } \left[ - { \bf h } _ { t } ^ { \mathrm { H } } + J ^ { \mathrm { H } } \phi ( { \bf h } _ { t } ^ { \mathrm { H } } ) + W _ { \mathrm { i n } } ^ { \mathrm { H } } { \bf c } _ { t } \right] .\tag{9}
$$

The recurrent matrices $J ^ { \mathrm { R } }$ and $J ^ { \mathrm { H } }$ and their input matrices remain fixed after initialization. The main reservoir provides nonlinear features of the observed dynamics, whereas the smaller context reservoir provides a lowdimensional representation of the supplied context (see Fig. 1c).

Separate reservoir pathways for observed variables and regime information have previously been used for multiregime time-series prediction [21]. In the present experiments, the supplied context is constant within each trial.

We therefore do not assume that recurrent memory in the context pathway is itself essential; the question addressed here is how the resulting context representation acts on the shared computation.

The principal construction considered here is the augmented HyperReservoir, whose readout feature vector is

$$
\psi _ { t } ^ { \mathrm { a u g } } = \left[ \begin{array} { c } { { \mathbf { h } _ { t } ^ { \mathrm { R } } } } \\ { { \mathbf { h } _ { t } ^ { \mathrm { H } } } } \\ { { \mathbf { h } _ { t } ^ { \mathrm { H } } } \otimes { \mathbf { h } _ { t } ^ { \mathrm { R } } } } \end{array} \right] ,\tag{10}
$$

where $\otimes$ denotes the Kronecker product. The corresponding prediction is

$$
\widehat { \mathbf { y } } _ { t } = W ^ { \mathrm { R } } \mathbf { h } _ { t } ^ { \mathrm { R } } + W ^ { \mathrm { H } } \mathbf { h } _ { t } ^ { \mathrm { H } } + W ^ { \mathrm { R H } } \left( \mathbf { h } _ { t } ^ { \mathrm { H } } \otimes \mathbf { h } _ { t } ^ { \mathrm { R } } \right) + \mathbf { b } _ { \mathrm { o u t } } .\tag{11}
$$

All coeficients in this readout are fitted by ridge regression; neither recurrent subsystem is trained by backpropagation.

To expose the context-dependent structure of the readout, we partition $W ^ { \mathrm { R H } }$ into M blocks,

$$
W ^ { \mathrm { R H } } = \left[ B _ { 1 } B _ { 2 } \ \cdot \cdot \cdot B _ { M } \right] ,\tag{12}
$$

where $B _ { m } \in \mathbb { R } ^ { D _ { \mathrm { o u t } } \times N }$ . Equation (11) can then be written as

$$
\widehat { \mathbf { y } } _ { t } = \left[ W ^ { \mathrm { R } } + \sum _ { m = 1 } ^ { M } h _ { t , m } ^ { \mathrm { H } } B _ { m } \right] \mathbf { h } _ { t } ^ { \mathrm { R } } + W ^ { \mathrm { H } } \mathbf { h } _ { t } ^ { \mathrm { H } } + \mathbf { b } _ { \mathrm { o u t } } .\tag{13}
$$

The efective coeficient matrix applied to the mainreservoir features is

$$
W _ { \mathrm { e f f } } \left( \mathbf { h } _ { t } ^ { \mathrm { H } } \right) = W ^ { \mathrm { R } } + \sum _ { m = 1 } ^ { M } { h } _ { t , m } ^ { \mathrm { H } } B _ { m } .\tag{14}
$$

Equation (14) provides the central interpretation of the architecture. The matrix $W ^ { \mathrm { R } }$ defines a contextindependent baseline mapping, while the remaining term provides a context-dependent correction. The efective readout therefore forms an afine family parameterized by the context state, so that shared predictive structure can remain in $W ^ { \mathrm { R } }$ while only the context-dependent part of the mapping changes.

This parameterization is closely related to the Hypernetwork viewpoint, in which one network or representation determines parameters used by another computation [22]. Here, however, the context dependence is restricted to the readout, and all fitted coeficients are obtained in a single ridge-regression problem.

This construction is related to previous reservoir architectures with multiple readouts [15, 16], but does not assign an independently fitted output matrix to each regime. Instead, the efective readouts are coupled through the shared baseline $W ^ { \mathrm { R } }$ , the matrices $\{ B _ { m } \} _ { m = 1 } ^ { \tilde { M } } .$ and the low-dimensional context representation $\mathbf { h } _ { t } ^ { \mathrm { H } }$

The bilinear feature block is also related to generalized nonlinear reservoir readouts [17]. Here, however, the multiplicative terms are specifically cross-interactions between the separately driven context and main-reservoir states. This structure gives the bilinear term the interpretation of a context-dependent correction to the mainreservoir readout rather than a generic nonlinear feature expansion.

Alternative additive and multiplicative readout constructions are compared in Sec. V.

## III. NUMERICAL EXPERIMENTS

We compare the three architectures on three futurestate prediction settings based on Lorenz and R¨ossler dynamics, summarized in Fig. 2. The settings range from geometrically distinct attractors to a shared attractor traversed at diferent temporal scales.

In all experiments, the context is explicitly supplied, remains fixed within each trial, and is available at every time step. All architectures use the same trajectories, data partitions, prediction targets, and evaluation metric.

## A. Geometrically distinct Lorenz and R¨ossler dynamics

The first task combines the Lorenz and R¨ossler systems. The Lorenz dynamics are

$$
\dot { \chi } = \sigma ( \psi - \chi ) ,\tag{15}
$$

$$
\dot { \psi } = \chi ( \rho - \omega ) - \psi ,\tag{16}
$$

$$
\begin{array} { r } { \dot { \boldsymbol { \omega } } = \chi \psi - \beta \boldsymbol { \omega } , } \end{array}\tag{17}
$$

with

$$
\sigma = 1 0 , \qquad \rho = 2 8 , \qquad \beta = \frac { 8 } { 3 } .\tag{18}
$$

The R¨ossler dynamics are

$$
\dot { \chi } = - \psi - \omega ,\tag{19}
$$

$$
\dot { \psi } = \chi + a _ { \mathrm { { R } } } \psi ,\tag{20}
$$

$$
\dot { \omega } = b _ { \mathrm { R } } + \omega ( \chi - c _ { \mathrm { R } } ) ,\tag{21}
$$

with

$$
a _ { \mathrm { R } } = 0 . 2 , \qquad b _ { \mathrm { R } } = 0 . 2 , \qquad c _ { \mathrm { R } } = 5 . 7 .\tag{22}
$$

Each trial contains a trajectory from one of the two systems, whose identity is supplied as a two-dimensional one-hot context vector. The two systems have clearly distinct attractor geometries.

## B. Related R¨ossler regimes

The second task considers two regimes belonging to the same R¨ossler family. The vector field is again given by Eqs. (19)–(21), with $a _ { \mathrm { R } } = b _ { \mathrm { R } } = 0 . 2$ , while

$$
c _ { \mathrm { R } } \in \{ 3 . 5 , 5 . 7 \} .\tag{23}
$$

The two values are represented by a two-dimensional one-hot context vector and are both included in the training, validation, and test sets. Thus, the experiment concerns prediction for known regimes rather than interpolation to an unseen value of $c _ { \mathrm { R } }$

## C. Same attractor under diferent temporal scales

The third setting uses the same R¨ossler system in both contexts but changes its temporal scale. Both regimes use the R¨ossler system with $a _ { \mathrm { R } } = b _ { \mathrm { R } } = 0 . 2$ and $c _ { \mathrm { R } } = 5 . 7 , $ but the complete vector field is multiplied by a contextdependent temporal-scale factor:

$$
\dot { \boldsymbol { \phi } } = \nu _ { k } \mathbf { f } _ { \mathrm { R } } ( \boldsymbol { \phi } ) , \qquad \boldsymbol { \phi } = \left[ \chi \textit { \psi } \boldsymbol { \omega } \right] ^ { \mathsf { T } } .\tag{24}
$$

The two temporal regimes are

$$
\nu _ { \mathrm { s l o w } } = 0 . 5 , \qquad \nu _ { \mathrm { f a s t } } = 1 . 5 .\tag{25}
$$

The supplied context is $( c _ { \mathrm { R } } , \nu _ { k } )$ . Since $c _ { \mathrm { R } } = 5 . 7$ in both regimes, only $\nu _ { k }$ distinguishes the slow and fast cases. For constant $\nu _ { k }$ , scaling the complete vector field amounts to a reparameterization of time: the continuous-time orbit geometry is unchanged, but the trajectory is traversed at a diferent rate. Consequently, the same current state can require diferent future-state predictions over a fixed prediction interval.

## D. Prediction protocol

All dynamical systems are integrated using a fourthorder Runge–Kutta method with an internal integration step of 0.005 and are sampled every 0.05 continuoustime units. Each trajectory contains 1000 recorded samples. The training, validation, and test sets contain 256, 64, and 64 independently initialized trajectories, respectively. The same generated datasets are used for all architectures. Each architecture is evaluated using three independently initialized reservoir realizations, corresponding to seeds $s \in \{ 0 , 1 , 2 \}$ . Additional integration settings, initial-condition distributions, and reproducibility details are provided in the Supplemental Material.

The task is one-step-ahead future-state prediction. At time t, the observed state is

$$
\mathbf { s } _ { t } = \left[ \chi _ { t } \psi _ { t } \omega _ { t } \right] ^ { \mathsf { T } } ,\tag{26}
$$

and the supervised target is

$$
\mathbf { y } _ { t } = \mathbf { s } _ { t + H } , \qquad H = 1 .\tag{27}
$$

Because the sampling interval is 0.05, the corresponding physical prediction horizon is

$$
T _ { \mathrm { p r e d } } = H \Delta t _ { \mathrm { s a m p } } = 0 . 0 5 .\tag{28}
$$

a  
![](images/8885e006a4df570d265aab47410c2395ea91cce17d7d409acbb882e47ebdd726.jpg)

![](images/720f39a6dbfc21b3abdb1e0599e7d99101bfbc52cca622157799ed00a3642ef9.jpg)

![](images/0f7c5d50b06c7feb2c28bca7103f46c528e2f873982e54c30c9942229faecfa4.jpg)

![](images/d034c77581b73e0a57e39b473def06ab6f9622a2b0792872dfd5b1df37a5965f.jpg)  
FIG. 2. Dynamical settings used for contextual future-state prediction. (a) Geometrically distinct R¨ossler and Lorenz attractors. (b) Two R¨ossler regimes with $c _ { R } = 3 . 5 $ and $c _ { R } = 5 . 7 $ . (c) The same R¨ossler attractor traversed at two temporal scales, $\nu _ { \mathrm { s l o w } } = 0 . 5$ and $\nu _ { \mathrm { f a s t } } = 1 . 5$ . Scaling the complete vector field preserves the continuous-time orbit geometry while changing the rate of traversal. Representative traces of the ω component illustrate the resulting temporal diference.

The first 20 samples of each trajectory are discarded as reservoir washout, and samples without a valid future target are excluded from both fitting and evaluation. During evaluation, the observed dynamical state is supplied to the reservoir at every time step; predicted states are not fed back recursively. The reported errors therefore measure one-step prediction along observed trajectories rather than autonomous attractor generation.

All normalization statistics are estimated from the training set and then applied unchanged to the validation and test sets. A single set of component-wise statistics is shared across contextual regimes within each task; in particular, the Lorenz and R¨ossler trajectories are not standardized separately. This avoids introducing regime identity through preprocessing. Further details are provided in the Supplemental Material.

## E. Recurrent-state allocation and parameter selection

The same observations and contextual information are available to all architectures, but the context is routed to diferent computational stages as defined in Sec. II. Because the HyperReservoir contains an additional context reservoir, we control the total number of recurrent state variables across architectures. We fix this total dimen sion to

$$
N _ { \mathrm { t o t } } = 1 2 0 .\tag{29}
$$

The conventional ESN and full-Conceptor model therefore use 120 recurrent states in their reservoir. For the HyperReservoir, the same budget is divided between the main and context reservoirs:

$$
N + M = 1 2 0 .\tag{30}
$$

The context-reservoir dimension is evaluated over

$$
M \in \{ 2 , 5 , 1 0 , 2 0 , 4 0 \} , \qquad N = 1 2 0 - M .\tag{31}
$$

For each dynamical setting, one value of M is selected using the mean validation cNMSE over the three reservoir realizations and is then used for all test evaluations. This procedure gives

$$
( N , M ) = \left\{ \begin{array} { l l } { { ( 1 1 0 , 1 0 ) , } } & { { \mathrm { L o r e n z \mathrm { - } R i s s l e r , } } } \\ { { ( 1 1 5 , 5 ) , } } & { { \mathrm { r e l a t e d \ R i s s l e r , } } } \\ { { ( 1 1 5 , 5 ) , } } & { { \mathrm { s a m e \ a t t r a c t o r . } } } \end{array} \right.\tag{32}
$$

The Conceptor aperture is selected in the same manner, using only validation data. The resulting values are

$$
\gamma ^ { * } = \left\{ \begin{array} { l l } { 4 , } & { \mathrm { L o r e n z - R i s s l e r } , } \\ { 1 , } & { \mathrm { r e l a t e d ~ R i s s l e r } , } \\ { 1 , } & { \mathrm { s a m e ~ a t t r a c t o r } . } \end{array} \right.\tag{33}
$$

The tested aperture values and other selection details are given in the Supplemental Material.

Both M and $\gamma$ are selected from validation data using the mean over the three reservoir realizations; test data are not used for parameter selection.

The ridge coeficient is fixed to

$$
\lambda = 1 0 ^ { - 4 } .\tag{34}
$$

The recurrent spectral radii, input scales, and leak coeficients are listed in the Supplemental Material.

Each Conceptor is constructed exclusively from post washout training reservoir states; validation and test states are not used.

Fixing the number of recurrent state variables does not equalize the number of fitted coeficients, storage requirements, or arithmetic cost. These diferences are considered separately in Sec. V and the Supplemental Material.

## F. Evaluation metric

Prediction accuracy is quantified using componentnormalized mean squared error (cNMSE). For output component d, we define

$$
E _ { d } = \frac { \sum _ { b , t } m _ { b , t } \left( \widehat { y } _ { b , t , d } - y _ { b , t , d } \right) ^ { 2 } } { \operatorname* { m a x } \left[ \sum _ { b , t } m _ { b , t } y _ { b , t , d } ^ { 2 } , \epsilon \right] } ,\tag{35}
$$

where b indexes trajectories, t indexes time, and $^ { m _ { b , t } }$ is the evaluation mask. Specifically, $m _ { b , t } = 1$ for postwashout samples with a valid future-state target and

$m _ { b , t } ~ = ~ 0$ otherwise. The constant $\epsilon = 1 0 ^ { - 4 }$ prevents numerical instability if the target energy in the denominator becomes very small.

The reported error is

$$
\mathrm { c N M S E } = \frac { 1 } { D } \sum _ { d = 1 } ^ { D } E _ { d } , \qquad D = 3 .\tag{36}
$$

Lower cNMSE indicates more accurate prediction.

## G. Results

Fig. 3 shows the results of the numerical simulations across all three tasks. Figure $\mathrm { 3 ( a ) }$ shows the Lorenz– R¨ossler result. All three architectures achieve mean cN-MSE below $4 . 5 \times 1 0 ^ { - 2 }$ , with the ESN and Conceptor giving similar errors. The augmented HyperReservoir gives the lowest error for each of the three reservoir realiza tions.

Figure 3(b) shows the two R¨ossler regimes with $c _ { \mathrm { R } } \in$ {3.5, 5.7}. The separation among the architectures is substantially larger than in panel (a). The HyperReservoir gives the lowest error, while the Conceptor error is approximately one order of magnitude larger than that of the conventional ESN.

Figure $3 ( \mathrm { c } )$ shows the same-attractor setting with different temporal scales. The HyperReservoir again gives the lowest test cNMSE, with a substantially larger separation from both reference models than in the Lorenz– R¨ossler setting.

Taken together, the three tasks show that the augmented HyperReservoir provides a consistent advantage over the conventional context-input ESN and Conceptor. The advantage is weaker on the first task, where the underlying target dynamics are geometrically distinct. This pattern is consistent with the interpretation that context-dependent decoding becomes particularly useful when the representation implemented by the main reservoir can remain shared while the required predictive mapping changes across contexts.

Aggregate prediction errors do not show how these differences arise along individual trajectories. We therefore examine the same-attractor setting in more detail in Sec. IV.

## IV. CONTEXTUAL SEPARATION ON A SHARED ATTRACTOR

We examine the same-attractor setting using local prediction errors, context replacement, and a comparison of the corresponding Conceptor operators. First, we compare the prediction errors under the two temporal regimes. Second, we see how the prediction reacts to changing only the supplied contextual information while keeping the observed trajectory fixed. Third, we compare the Conceptors learned for the two regimes to determine how strongly the temporal distinction is expressed in their reservoir-state geometry.

a  
![](images/4d0ba54d22f63083183a327550eccdcd155ed58c412398e5ac1bbd72e54921af.jpg)

![](images/8691c61a557b7c2d0f7282a5782be6857abf67259b4544ffe8bb3abe4f612036.jpg)

![](images/8e0a4525e8dc80855f7137b4fbc86193de64ce9ada2b0fea76d163d9942903ea.jpg)  
FIG. 3. Future-state prediction for the three dynamical settings introduced in Sec. III. (a) Geometrically distinct Lorenz and R¨ossler systems. (b) Two R¨ossler regimes with $c _ { \mathrm { R } } = 3 . 5$ and $c _ { \mathrm { R } } = 5 . 7 .$ (c) The same R¨ossler attractor traversed at diferent temporal scales, $\nu _ { \mathrm { s l o w } } = 0 . 5$ and $\nu _ { \mathrm { f a s t } } = 1 . 5$ . Test component-normalized mean squared error (cNMSE) is shown for the contextinput ESN, full-matrix Conceptor, and augmented HyperReservoir. Individual markers denote reservoir realizations; horizontal markers and error bars show the mean and sample standard deviation. The vertical axes are logarithmic, and lower cNMSE indicates more accurate prediction.

## A. Finite-time prediction on a shared attractor

Figures $4 ( \mathrm { a } )$ and 4(b) show the one-step prediction error of the first R¨ossler variable,

$$
e _ { \chi , t } = \hat { \chi } _ { t + 1 } - \chi _ { t + 1 } ,\tag{37}
$$

for representative slow and fast test trajectories. The HyperReservoir shows smaller local errors than the contextinput ESN and full-matrix Conceptor in this realization, particularly in the fast regime, consistent with the aggregate errors in Fig. 3.

## B. Correct- and wrong-context prediction

To quantify how prediction depends on the supplied context, we replace the context while keeping the observed test trajectory fixed.

Let

$$
E _ { i j } = \mathrm { c N M S E } \left( \mathrm { t r u e \ c o n t e x t } \ i , \mathrm { s u p p l i e d \ c o n t e x t } \ j \right)\tag{38}
$$

denote the prediction error under true context i and supplied context $j .$ . The diagonal terms $E _ { 1 1 }$ and $E _ { 2 2 }$ correspond to the correct context, whereas $E _ { 1 2 }$ and $E _ { 2 1 }$ correspond to context mismatch.

We summarize the dependence on context by the ratio

$$
R _ { \mathrm { c t x } } = \frac { E _ { 1 1 } + E _ { 2 2 } } { E _ { 1 2 } + E _ { 2 1 } } .\tag{39}
$$

A value close to unity indicates little sensitivity to context replacement, whereas a value substantially below unity indicates that prediction accuracy strongly depends on the supplied context.

For consistency with the main prediction comparison, the Conceptor wrong-context analysis uses the task-level aperture selected by mean validation cNMSE across the three reservoir realizations, $\gamma ^ { * } = 1$

Figure $4 ( \mathrm { c } )$ shows that the HyperReservoir has the smallest correct-to-wrong context error ratio. Its prediction error therefore increases most strongly when the supplied context is replaced. The conventional ESN shows an intermediate response, whereas the Conceptor ratio remains closer to unity. Thus, the low prediction error of the HyperReservoir in this setting is accompanied by a strong dependence on the supplied context.

## C. Similarity of the slow and fast Conceptors

We next examine how strongly the slow and fast regimes are distinguished by the corresponding Concep tor operators. The Conceptor construction represents each context through the second-order correlation structure of its reservoir states. If the slow and fast temporal regimes induce strongly diferent reservoir-state geometries, their corresponding Conceptors should difer accordingly. Therefore, we compare the operators learned for the two contexts.

We use the same validation-selected aperture, $\gamma ^ { * } = 1$ as in the prediction comparison.

Let $C _ { \mathrm { s l o w } }$ and $C _ { \mathrm { f a s t } }$ denote the Conceptors obtained for the two temporal regimes. Validation and test reservoir states are not used to estimate either Conceptor.

Their similarity is quantified using the Frobenius cosine

$$
S _ { C } = \frac { \mathrm { t r } \left( C _ { \mathrm { s l o w } } ^ { \mathsf { T } } C _ { \mathrm { f a s t } } \right) } { \left\| C _ { \mathrm { s l o w } } \right\| _ { F } \left\| C _ { \mathrm { f a s t } } \right\| _ { F } } .\tag{40}
$$

![](images/56303b3733ae59d9969f8f11781b1ab2d5341c853730aff1c3b0e2339abc6364.jpg)  
FIG. 4. Analysis of contextual prediction on a shared attractor. (a,b) One-step prediction errors, $e _ { \boldsymbol { \chi } , t } = \hat { { \chi } } _ { t + 1 } - { { \chi } _ { t + 1 } }$ , for the slow $\nu = 0 . 5$ and fast $\nu = 1 . 5$ regimes. The same representative reservoir realization and evaluation window are used for all models; the vertical scales difer between panels for visibility. (c) Dependence on the supplied context, quantified by the ratio $R _ { \mathrm { c t x } }$ between correct- and wrong-context prediction errors. Lower values indicate a larger increase in error after context replacement. (d) Frobenius cosine similarity between the slow and fast Conceptors for the three reservoir realizations. Values close to unity indicate strong alignment of the two operators.

A value close to unity indicates strong alignment of the two operators in Frobenius space.

Figure $4 ( \mathrm { d } )$ shows that the slow and fast Conceptors remain substantially aligned across reservoir realizations. Across the three reservoir realizations,

$$
S _ { C } = 0 . 8 1 3 \pm 0 . 0 2 3 ,\tag{41}
$$

where the uncertainty denotes the sample standard deviation across the three reservoir realizations. A descriptive comparison of the ordered eigenvalue spectra is provided in Fig. S2 of the Supplemental Material.

The two Conceptors are therefore not identical, but remain strongly aligned. Together with the comparatively small efect of context replacement for the Conceptor model, this indicates that the slow–fast distinction is only weakly expressed in the Conceptor-based statespace representation used here.

We next examine the structure of the HyperReservoir readout in Sec. V.

## V. STRUCTURE OF THE CONTEXT-DEPENDENT READOUT

We compare three readout constructions to determine how the bilinear interaction contributes to the Hyper-Reservoir prediction. All three use the same main and context reservoirs and difer only in the features supplied to the readout.

The three feature constructions are

$$
\begin{array} { r l } & { \psi _ { t } ^ { \mathrm { c o n c a t } } = \left[ \mathbf { h } _ { t } ^ { \mathrm { R } } \right] , } \\ & { \psi _ { t } ^ { \mathrm { s t r i c t } } = \mathbf { h } _ { t } ^ { \mathrm { H } } \otimes \mathbf { h } _ { t } ^ { \mathrm { R } } , } \\ & { \quad \psi _ { t } ^ { \mathrm { a u g } } = \left[ \begin{array} { l } { ~ \mathbf { h } _ { t } ^ { \mathrm { R } } } \\ { ~ \mathbf { h } _ { t } ^ { \mathrm { H } } \otimes \mathbf { h } _ { t } ^ { \mathrm { R } } } \end{array} \right] . } \end{array}\tag{42}
$$

The Concat readout provides direct additive access to both recurrent states but does not allow the coeficients applied to the main-reservoir features to depend on context. The Strict readout retains only the bilinear term and therefore implements a purely contextdependent multiplicative readout without an explicit context-independent main-reservoir mapping. The Augmented readout combines both additive features and the bilinear interaction and is the construction used in the principal comparisons. They are visualized in Fig. 5(a).

Figure 5(b) shows that the Augmented readout gives the lowest mean test cNMSE in all three settings. Both Concat and Strict give higher errors, indicating that neither additive access to the context state alone nor the bilinear interaction alone reproduces the performance of the complete Augmented readout. The diference is particularly large for the related-R¨ossler setting.

The comparison also argues against readout dimension alone as an explanation for the performance diference. Although the Augmented model has the largest readout, the Strict model already contains the NM bilinear feature block and therefore has a comparable number of fitted coeficients, yet its prediction error is substantially larger. Detailed feature dimensions are reported in the Supplemental Material.

We next examine the dependence on the contextreservoir dimension M. Keeping the total number of recurrent states fixed at $N + M = 1 2 0$ , we vary

$$
M \in \{ 2 , 5 , 1 0 , 2 0 , 4 0 \} ,\tag{43}
$$

Increasing M therefore reallocates recurrent states from the main reservoir to the contextual reservoir.

For the related-R¨ossler task shown in Fig. 5(c), increasing M does not produce a monotonic improvement in prediction. The cNMSE first drops, and then starts to increase again.

The validation-selected dimensions were

$$
M ^ { * } = \left\{ \begin{array} { l l } { 1 0 , } & { \mathrm { L o r e n z \mathrm { - } R \ " o s s l e r } , } \\ { 5 , } & { \mathrm { r e l a t e d \ R \ " o s s l e r } , } \\ { 5 , } & { \mathrm { s a m e \ a t t r a c t o r } . } \end{array} \right.\tag{44}
$$

Thus, the validation procedure selects relatively small context reservoirs in all three settings. Increasing the context-reservoir dimension beyond the validationselected value does not systematically improve prediction and can instead reduce accuracy.

This behavior is consistent with the functional division between the two recurrent subsystems. Because N + M is fixed, increasing M enlarges the context representation while reducing the number of states available to the main reservoir. The non-monotonic dependence is therefore consistent with a trade-of between these two state allocations.

Changing M also changes the dimensionality of the Augmented readout. Under $N + M = 1 2 0$ , the number of bilinear features is

$$
N M = M ( 1 2 0 - M ) .\tag{45}
$$

Over the evaluated range, NM and the total number of fitted coeficients increase monotonically with M, as shown in Fig. 5(d). Prediction accuracy does not improve monotonically over the same range. Thus, the observed dependence on M cannot be explained simply by an increase in fitted readout dimension.

Together, the readout comparison and dimension sweep show that increasing context dimension or fitted feature count alone does not account for the prediction improvement of the Augmented HyperReservoir.

## VI. CONCLUSION

We proposed the HyperReservoir, a reservoircomputing architecture for context-dependent decoding of shared recurrent dynamics. A common main reservoir represents the observed dynamics, while a smaller context pathway modulates the readout applied to this representation. In the augmented construction, the efective readout consists of a shared baseline mapping and a context-dependent correction.

We compared the HyperReservoir with two alternatives that introduce the same contextual informa tion at diferent stages of the computation: a conventional context-input ESN and full-matrix Conceptorbased state modulation. The three future-state prediction tasks ranged from geometrically distinct Lorenz and R¨ossler systems, through related R¨ossler regimes, to the same R¨ossler attractor traversed at diferent temporal scales. The augmented HyperReservoir achieved the lowest mean test cNMSE in all three tasks. We observed a modest advantage in the first task and larger advan tages in the latter two, which is consistent with contextdependent decoding being most useful when more of the recurrent representation can remain shared while the required predictive mapping changes across contexts.

The temporal-scale task illustrates a form of multifunctionality complementary to the multistability studied in much of the reservoir-computing literature, where a single trained system supports multiple coexisting attractors [9, 10, 24], including cases in which the corresponding trajectories overlap in state space [24]. The slow and fast regimes share the exact same continuous-time orbit geometry, but their finite-time flow maps difer. Multifunctionality therefore need not require multiple attractors: diferent computations can correspond to diferent temporal evolutions on a common state-space structure. The HyperReservoir realizes the corresponding separation between representation and interpretation: the main reservoir provides nonlinear features reused across regimes, and the context-dependent readout determines how those features contribute to the prediction.

Where context should enter a recurrent system depends on which part of the computation must vary across regimes. In the context-input ESN, context changes the reservoir trajectory through external forcing, and a single readout must decode all contexts. In the Conceptor model, context restricts the reservoir state space and therefore relies on regimes that induce distinguishable state distributions. In the HyperReservoir, the representation is retained and context changes only its decoding. For the geometrically distinct Lorenz and R¨ossler systems, state-level modulation remains efective: the

a  
![](images/3f1ee87659120ee28edc73de5e6132f84fef9b9167b76903093f9ddc8c759192.jpg)

![](images/b51446f5cfa75871bc358fce744eb55281269e42cdf9bb5d4dbfabbe736e727a.jpg)

c  
![](images/1d33406724c03ea788cca0b5009d41174726fde287e80524261855ec461044a3.jpg)

d  
![](images/18aadc4e195714d5979391ef032ef3777f8028a8cc88a602ea0a0f30df38923f.jpg)  
FIG. 5. Structure and dimensionality of the HyperReservoir readout. (a) Three readout constructions. The Concat model uses the main- and context-reservoir states additively, the Strict model uses only their bilinear interaction, and the Augmented model combines both. (b) Test cNMSE for the three readout constructions across the dynamical settings introduced in Sec. III. Individual markers denote reservoir realizations, and error bars show the mean and sample standard deviation. The Lorenz– R¨ossler setting uses $N = 1 1 0 , M = 1 0 .$ , whereas the related-R¨ossler and same-attractor settings use $N = 1 1 5 , M = 5 . \mathrm { ( c ) }$ ) Test cNMSE of the Augmented HyperReservoir for the related-R¨ossler setting as the context-reservoir dimension M is varied while $N + M = 1 2 0$ . The dashed line denotes the corresponding context-input ESN result. (d) Number of fitted output coeficients as M is varied under the same constraint.

Conceptor and the conventional ESN reach nearly identical errors. In the same-attractor setting, the slow and fast Conceptors remain strongly aligned, and replacing the supplied Conceptor produces a comparatively smaller change in prediction than replacing the context in the HyperReservoir. These observations are consistent with the temporal distinction being only weakly expressed in the Conceptor-based state-space representation used here.

The readout comparison shows that neither additive access to the context state nor the bilinear interaction alone reproduces the performance of the augmented construction. The context-dimension sweep further shows that prediction does not improve monotonically with either context dimension or fitted readout size.

Several limitations define the scope of these conclusions. During evaluation, the observed state is supplied at every time step, so the results concern short-horizon prediction rather than autonomous attractor generation. Context is explicitly supplied, and inference of an unknown or switching context is not addressed. The evaluated contexts are also a finite discrete set, so generalization to unseen regimes remains to be tested. Finally, the comparison controls the total number of recurrent state variables, not the number of fitted coeficients, storage requirements, or arithmetic cost.

When substantial recurrent structure can remain shared while its predictive interpretation must change, context-dependent decoding provides a direct mechanism for contextual flexibility. The HyperReservoir implements this principle while retaining training by linear regression.

## SUPPLEMENTARY MATERIAL

Supplemental Material is included after the main text and provides numerical settings and data-generation details, seedwise results, context-dimension and readout comparisons, same-attractor analyses, readout and storage counts, and software environment details.

## ACKNOWLEDGMENTS

This study was supported in part by a Grant-in-Aid for Transformative Research Areas (A) (JP22H05197), a Grant-in-Aid for JSPS Fellows (JP24KJ0868), Grantin-Aid for Exploratory Research (JP25K22227), JST-ALCA-Next (JPMJAN25F1), JST FOREST Program (JPMJFR2448) and SECOM Science and Technology Foundation.

[1] H. Jaeger and H. Haas, Harnessing nonlinearity: Predicting chaotic systems and saving energy in wireless communication, Science 304, 78 (2004).

[2] W. Maass, T. Natschl¨ager, and H. Markram, Real-time computing without stable states: A new framework for neural computation based on perturbations, Neural Computation 14, 2531 (2002).

[3] L. Appeltant, M. C. Soriano, G. Van der Sande, J. Danckaert, S. Massar, J. Dambre, B. Schrauwen, C. R. Mirasso, and I. Fischer, Information processing using a single dynamical node as complex system, Nature Communications 2, 468 (2011).

[4] K. Vandoorne, P. Mechet, T. Van Vaerenbergh, M. Fiers, G. Morthier, D. Verstraeten, B. Schrauwen, J. Dambre, and P. Bienstman, Experimental demonstration of reservoir computing on a silicon photonics chip, Nature Communications 5, 3541 (2014).

[5] K. Nakajima, H. Hauser, T. Li, and R. Pfeifer, Information processing via physical soft body, Scientific Reports 5, 10487 (2015).

[6] T. Kanao, H. Suto, K. Mizushima, H. Goto, T. Tanamoto, and T. Nagasawa, Reservoir computing on spin-torque oscillator array, Physical Review Applied 12, 024052 (2019).

[7] G. R. Yang, M. R. Joglekar, H. F. Song, W. T. Newsome, and X.-J. Wang, Task representations in neural networks trained to perform many cognitive tasks, Nature Neuroscience 22, 297 (2019).

[8] L. N. Driscoll, K. V. Shenoy, and D. Sussillo, Flexible multitask computation in recurrent networks utilizes shared dynamical motifs, Nature Neuroscience 27, 1349 (2024).

## AUTHOR DECLARATIONS

## Conflicts of Interest

The authors have no conflicts to disclose.

Author Contributions Kohei Tsuchiyama: Conceptualization (lead); Methodology (lead); Formal analysis (lead); Investigation (lead); Visualization (lead); Writing - original draft (lead); Writing - review & editing (equal). Takatomo Mihana: Investigation (supporting); Writing - review & editing (supporting). Ryoichi Horisaki: Writing - review & editing (supporting). Andr´e R¨ohm: Conceptualization (supporting); Methodology (supporting); Formal analysis (supporting); Investigation (supporting); Writing - original draft (supporting); Writing - review & editing (equal).

## DATA AVAILABILITY

The code, numerical data, and figure-generation scripts that support the findings of this study are available from the corresponding author upon reasonable request.

[9] A. Flynn, V. A. Tsachouridis, and A. Amann, Multifunctionality in a reservoir computer, Chaos 31, 013125 (2021).

[10] Y. Du, H. Luo, J. Guo, J. Xiao, Y. Yu, and X. Wang, Multifunctional reservoir computing, Physical Review E 111, 035303 (2025).

[11] V. Mante, D. Sussillo, K. V. Shenoy, and W. T. Newsome, Context-dependent computation by recurrent dynamics in prefrontal cortex, Nature 503, 78 (2013).

[12] L.-W. Kong, H.-W. Fan, C. Grebogi, and Y.-C. Lai, Machine learning prediction of critical transition and system collapse, Physical Review Research 3, 013090 (2021).

[13] R. Xiao, L.-W. Kong, Z.-K. Sun, and Y.-C. Lai, Predicting amplitude death with machine learning, Physical Review E 104, 014205 (2021).

[14] H. Jaeger, Using conceptors to manage neural longterm memories for temporal patterns, Journal of Machine Learning Research 18, 1 (2017).

[15] A. Laan and R. Vicente, Echo state networks with multiple read-out modules, bioRxiv , 017558 (2017).

[16] Y. Tanaka and H. Tamukoh, Self-organizing multiple readouts for reservoir computing, IEEE Access 11, 138839 (2023).

[17] A. Ohkubo and M. Inubushi, Reservoir computing with generalized readout based on generalized synchronization, Scientific Reports 14, 30918 (2024).

[18] Z. Lu, B. R. Hunt, and E. Ott, Attractor reconstruction by machine learning, Chaos 28, 061104 (2018).

[19] J. Wang, D. Narain, E. A. Hosseini, and M. Jazayeri, Flexible timing by temporal scaling of cortical responses, Nature Neuroscience 21, 102 (2018).

[20] V. Goudar and D. V. Buonomano, Encoding sensory and motor patterns as time-invariant trajectories in recurrent

[21] S. Zhong, X. Xie, L. Lin, and F. Wang, Genetic algorithm optimized double-reservoir echo state network for multiregime time series prediction, Neurocomputing 238, 191 (2017).

neural networks, eLife 7, e31134 (2018).

[22] D. Ha, A. M. Dai, and Q. V. Le, Hypernetworks, in International Conference on Learning Representations (2017).

[23] H. Jaeger, The “echo state” approach to analysing and training recurrent neural networks—with an erratum note, Tech. Rep. GMD Report 148 (German National Research Center for Information Technology, Bonn, Germany, 2001).

[24] A. Flynn, V. A. Tsachouridis, and A. Amann, Seeing double with a multifunctional reservoir computer, Chaos 33, 113115 (2023).

# Supplemental Material for Context-dependent time-series prediction via HyperReservoirs

Kohei Tsuchiyama, Takatomo Mihana, Ryoichi Horisaki, and Andr´e R¨ohm (Dated: September 29, 2026)

## S1. NUMERICAL SETTINGS AND DATA GENERATION

This section gives the numerical settings used in all experiments. The same generated datasets are used for all architectures, while the random recurrent realization is varied independently over three model seeds. Table S1 summarizes the common settings.

## S1.1. Trajectory generation

All continuous-time systems are integrated with a fourth-order Runge–Kutta scheme using an internal integration step of 0.005. The states are recorded every ten integration steps, giving a sampling interval of 0.05. Each recorded trajectory contains 1000 samples.

Before recording, each system is integrated for 2000 internal steps to reduce dependence on the initial condition sampled. For the R¨ossler systems, the initial conditions are drawn independently as

$$
x _ { 0 } \sim \mathcal { N } ( 0 , 1 ) , \qquad y _ { 0 } \sim \mathcal { N } ( 0 , 1 ) , \qquad z _ { 0 } \sim \mathcal { N } ( 0 . 5 , 0 . 2 ^ { 2 } ) .\tag{S1}
$$

For the Lorenz system,

$$
\begin{array} { r } { x _ { 0 } \sim \mathcal { N } ( 0 , 1 ) , \qquad y _ { 0 } \sim \mathcal { N } ( 0 , 1 ) , \qquad z _ { 0 } \sim \mathcal { N } ( 2 0 , 2 ^ { 2 } ) . } \end{array}\tag{S2}
$$

No observation noise or process noise is added. As a numerical safeguard, each dynamical-state component is trimmed to $[ - 8 0 , 8 0 ]$ during trajectory generation.

The training, validation and test sets contain 256, 64, and 64 independently initialized trajectories, respectively. The same generated datasets are used for all architectures. Each architecture is evaluated using three independently initialized reservoir realizations.

## S1.2. Reservoir initialization

For a reservoir of dimension N, the recurrent matrix is initialized with independent Gaussian entries of standard deviation $N ^ { - 1 / 2 }$ and then rescaled to spectral radius 0.9. The context-reservoir matrix is initialized independently using the same procedure.

Input weights are initialized independently from a zero-mean Gaussian distribution with standard deviation $\sigma _ { \mathrm { i n } } / \sqrt { U }$ , where $U$ is the total number of available observation and context channels. Input channels not used by a particular reservoir are then masked out according to the routing summarized in Table S2. The input scales are

$$
\sigma _ { \mathrm { i n } } ^ { \mathrm { R } } = 0 . 3 , \qquad \sigma _ { \mathrm { i n } } ^ { \mathrm { H } } = 0 . 1 .\tag{S3}
$$

The recurrent and input matrices are initialized as described above, and the bias terms in both reservoir-state updates are set to zero.

The leak coeficient is $\alpha = 0 . 1$ for the context-input ESN and Conceptor reservoir. For the HyperReservoir, the main and context reservoirs use

$$
\alpha ^ { \mathrm { R } } = 0 . 1 , \qquad \alpha ^ { \mathrm { H } } = \frac { 1 } { 6 0 } ,\tag{S4}
$$

respectively.

## S1.3. Context signals

For the Lorenz–R¨ossler task, the context is a twodimensional one-hot vector identifying the active system. For the related-R¨ossler task, a two-dimensional one-hot vector identifies $c _ { \mathrm { R } } = 3 . 5$ or 5.7. For the same-attractor task,

$$
{ \mathbf c } = \left[ c _ { \mathrm { { R } } } \ \nu \right] ^ { \mathsf { T } } , \qquad c _ { \mathrm { { R } } } = 5 . 7 , \qquad \nu \in \{ 0 . 5 , 1 . 5 \} .\tag{S5}
$$

In the HyperReservoir, these contextual channels are supplied only to the context reservoir.

## S1.4. Normalization

For all three tasks, normalization parameters are estimated exclusively from the training split and are subsequently applied without modification to the validation and test sets. For component $d ,$

$$
\widetilde { u } _ { t , d } = \frac { u _ { t , d } - \mu _ { u , d } ^ { \mathrm { t r a i n } } } { \sigma _ { u , d } ^ { \mathrm { t r a i n } } } , \qquad \widetilde { y } _ { t , d } = \frac { y _ { t , d } - \mu _ { y , d } ^ { \mathrm { t r a i n } } } { \sigma _ { y , d } ^ { \mathrm { t r a i n } } } .\tag{S6}
$$

The standard deviations are lower bounded by $1 0 ^ { - 6 }$ . The same transformation is reused for validation and test data without re-estimating any statistics.

For the Lorenz–R¨ossler task, the two dynamical systems are not standardized separately. Instead, a single set of normalization statistics is estimated from the combined training trajectories and applied to both contextual regimes. This avoids introducing system identity through context-dependent preprocessing.

TABLE S1. Common numerical and model settings used in the three contextual prediction tasks. The model seed changes the random recurrent realization while the generated datasets remain fixed.
<table><tr><td>Category</td><td>Quantity</td><td>Value</td></tr><tr><td>Data</td><td>data seed</td><td>0</td></tr><tr><td></td><td>model seeds</td><td>{0, 1, 2}</td></tr><tr><td></td><td>training trajectories</td><td>256</td></tr><tr><td></td><td>validation trajectories</td><td>64</td></tr><tr><td></td><td>test trajectories</td><td>64</td></tr><tr><td></td><td>samples per trajectory</td><td>1000</td></tr><tr><td></td><td>prediction horizon</td><td> $H = 1$ </td></tr><tr><td></td><td>reservoir washout</td><td>20 samples</td></tr><tr><td>Dynamical-system integration integrator</td><td></td><td>fourth-order Runge-Kutta</td></tr><tr><td></td><td>internal integration step</td><td>0.005</td></tr><tr><td></td><td>sampling interval</td><td>0.05</td></tr><tr><td></td><td>dynamical burn-in</td><td>2000 internal steps</td></tr><tr><td>Reservoir</td><td>total recurrent-state budget</td><td> $N _ { \mathrm { t o t } } = 1 2 0$ </td></tr><tr><td></td><td>main-reservoir spectral radius</td><td>0.9</td></tr><tr><td></td><td>context-reservoir spectral radius</td><td>0.9</td></tr><tr><td></td><td>main-reservoir input scale</td><td>0.3</td></tr><tr><td></td><td>context-reservoir input scale</td><td>0.1</td></tr><tr><td></td><td>ESN/Conceptor leak coefficient</td><td> $\alpha = 0 . 1$ </td></tr><tr><td></td><td>HyperReservoir main leak coefficient</td><td> $\alpha ^ { \mathrm { ~ R ~ } } = 0 . 1$ </td></tr><tr><td></td><td>HyperReservoir context leak coefficient nonlinearity</td><td> $\alpha ^ { \mathrm { H } } = 1 / 6 0$ </td></tr><tr><td>Readout</td><td>ridge coefficient</td><td>tanh  $\lambda = 1 0 ^ { - 4 }$ </td></tr><tr><td></td><td>output bias</td><td>fitted and unregularized</td></tr><tr><td>HyperReservoir selection</td><td>context dimensions</td><td> $M \in \{ 2 , 5 , 1 0 , 2 0 , 4 0 \}$ </td></tr><tr><td></td><td>main dimension</td><td> $N = 1 2 0 - M$ </td></tr><tr><td>Conceptor selection</td><td></td><td></td></tr><tr><td></td><td>initial aperture candidates additional Lorenz-Rössler refinement</td><td> $\gamma \in \{ 1 , 3 , 1 0 , 3 0 \}$   $\gamma \in \{ 1 . 5 , 2 , 4 , 5 , 7 \}$ </td></tr></table>

TABLE S2. Architecture-specific routing of the observed dynamical state and context.
<table><tr><td>Model</td><td>Main reservoir</td><td>Context reservoir / selector Context-dependent operation</td><td></td></tr><tr><td>Context-input ESN</td><td>observation + context none</td><td></td><td>input forcing</td></tr><tr><td>full-matrix Conceptor observation only</td><td></td><td>context selects  $C _ { k }$ </td><td>state-space filtering</td></tr><tr><td>HyperReservoir</td><td>observation only</td><td>context only</td><td>readout modulation</td></tr></table>

## S1.5. Readout fitting

All output coeficients are fitted by ridge regression after the washout interval. Let X denote the matrix of valid readout features and Y the corresponding targets. An unregularized constant column is appended to the feature matrix:

$$
X _ { \mathrm { a u g } } = \left[ X \textbf { 1 } \right] .\tag{S7}
$$

The fitted coeficients are

$$
W _ { \mathrm { a u g } } = \left( X _ { \mathrm { a u g } } ^ { \mathsf { T } } X _ { \mathrm { a u g } } + \lambda R \right) ^ { - 1 } X _ { \mathrm { a u g } } ^ { \mathsf { T } } Y .\tag{S8}
$$

Here, R is the identity matrix except that the entry corresponding to the constant feature is zero. Thus, the output bias is not regularized. The ridge coeficient is fixed to $\lambda = 1 0 ^ { - 4 }$ in all principal experiments and is not selected from validation data.

## S1.6. Conceptor estimation

Conceptors are estimated exclusively from postwashout training reservoir states. For context k, the un centered empirical correlation matrix is

$$
R _ { k } = \frac { 1 } { n _ { k } } \sum _ { i \in \mathcal { T } _ { k } } \mathbf { x } _ { i } \mathbf { x } _ { i } ^ { \mathsf { T } } ,\tag{S9}
$$

where $\mathcal { T } _ { k }$ indexes training states in context k. The corresponding full-matrix Conceptor is

$$
C _ { k } = { { R } _ { k } } \left( { { R } _ { k } } + { \gamma } ^ { - 2 } I \right) ^ { - 1 } .\tag{S10}
$$

No validation or test states are used in estimating $R _ { k }$ or $C _ { k }$ . The readout is then trained from the Conceptorfiltered training states.

## S1.7. Validation-based architecture selection

The HyperReservoir context dimension is evaluated over

$$
M \in \{ 2 , 5 , 1 0 , 2 0 , 4 0 \} , \qquad N = 1 2 0 - M .\tag{S11}
$$

The full-matrix Conceptor aperture is initially evaluated over

$$
\gamma \in \{ 1 , 3 , 1 0 , 3 0 \} .\tag{S12}
$$

For the Lorenz–R¨ossler task, the aperture search is additionally refined around the best-performing region of the initial validation sweep using

$$
\gamma \in \{ 1 . 5 , 2 , 4 , 5 , 7 \} .\tag{S13}
$$

For a candidate configuration θ, let $E _ { \mathrm { v a l } } ^ { ( s ) } ( \theta )$ denote its validation cNMSE for reservoir seed $s .$ . One task-level configuration is selected according to

$$
\theta ^ { \star } = \operatorname * { a r g m i n } _ { \theta } \frac { 1 } { 3 } \sum _ { s \in \{ 0 , 1 , 2 \} } E _ { \mathrm { v a l } } ^ { ( s ) } ( \theta ) .\tag{S14}
$$

All numerical architecture and hyperparameter selections use validation cNMSE; test cNMSE is not included in the numerical selection criterion. A separate architecture is therefore not selected from the test performance of each seed.

For the augmented HyperReservoir, the validationselected dimensions are

$$
\begin{array} { r l r } { M _ { \mathrm { d i s t i n c t } } ^ { \star } = 1 0 , } & { { } \quad } & { N _ { \mathrm { d i s t i n c t } } ^ { \star } = 1 1 0 , } \\ { M _ { \mathrm { r e l a t e d } } ^ { \star } = 5 , } & { { } \quad } & { N _ { \mathrm { r e l a t e d } } ^ { \star } = 1 1 5 , } \\ { M _ { \mathrm { s a m e } } ^ { \star } = 5 , } & { { } \quad } & { N _ { \mathrm { s a m e } } ^ { \star } = 1 1 5 . } \end{array}\tag{S15}
$$

For the full-matrix Conceptor model, the final tasklevel apertures selected by mean validation cNMSE across the three reservoir realizations are

$$
\begin{array} { r } { \gamma _ { \mathrm { d i s t i n c t } } ^ { \star } = 4 , } \\ { \gamma _ { \mathrm { r e l a t e d } } ^ { \star } = 1 , } \\ { \gamma _ { \mathrm { s a m e } } ^ { \star } = 1 . } \end{array}\tag{S16}
$$

The same task-level aperture is used for every reservoir realization and in all Conceptor analyses associated with a given task, including the principal prediction comparison and the same-attractor operator analysis.

## S1.8. Evaluation-mask details

The first 20 samples of each trajectory are excluded as reservoir washout. The final H samples are also excluded because they do not have a valid future-state target. The component-normalized mean squared error is evaluated on the remaining samples using targets normalized with training-derived statistics. The numerical stabilization constant in the denominator is $1 0 ^ { - 4 }$ . No validation- or test-set recentering is performed.

The arithmetic mean and sample standard deviation across the three reservoir realizations are reported only as descriptive summaries. Because the number of real izations is small, individual seed values are shown in the principal figures.

## S2. MAIN-COMPARISON RESULTS

Table S3 reports the individual test cNMSE values for all three random reservoir realizations after validationbased architecture selection.

## S3. CONTEXT-RESERVOIR DIMENSION SWEEP

The context dimension is varied while keeping the total recurrent-state budget fixed:

$$
N + M = 1 2 0 .\tag{S17}
$$

The tested context dimensions are

$$
M \in \{ 2 , 5 , 1 0 , 2 0 , 4 0 \} .\tag{S18}
$$

Table S4 gives the test summaries for all three tasks. The architecture used in the comparison is selected from validation performance rather than from the test values in this table. Panel (c) of Fig. 5 in the main manuscript shows the related-R¨ossler sweep, for which the validationselected value is $M ^ { \star } = 5$

The validation-selection procedure chooses $M = 1 0$ for the geometrically distinct Lorenz–R¨ossler task and $M =$ 5 for both the related-R¨ossler and same-attractor tasks. Increasing M does not produce a monotonic improvement in test prediction accuracy. Under the matchedstate constraint, increasing the number of contextreservoir states simultaneously reduces the dimension available to the main reservoir and changes the size of the bilinear readout.

## S4. READOUT-COMPARISON RESULTS

Here, we present details for the three variations of the HyperReservoir: As explained in the main manuscript, there are several ways for combining the features of the main and context reservoir. Here, we chose the following:

$$
\begin{array} { r l r } & { \psi _ { t } ^ { \mathrm { c o n c a t } } = \left[ \mathbf { h } _ { t } ^ { \mathrm { R } } \right] , } & \\ & { \psi _ { t } ^ { \mathrm { s t r i c t } } = \mathbf { h } _ { t } ^ { \mathrm { H } } \otimes \mathbf { h } _ { t } ^ { \mathrm { R } } , } & \\ & { \psi _ { t } ^ { \mathrm { a u g } } = \left[ \begin{array} { l } { \mathbf { h } _ { t } ^ { \mathrm { R } } } \\ { \mathbf { h } _ { t } ^ { \mathrm { H } } } \end{array} \right] . } & \\ & { \psi _ { t } ^ { \mathrm { H } } \otimes \mathbf { h } _ { t } ^ { \mathrm { R } } } \end{array}\tag{S19}
$$

TABLE S3. Seedwise test cNMSE. The final column gives the arithmetic mean and sample standard deviation over the three random reservoir realizations.
<table><tr><td>Task</td><td>Model</td><td>Seed 0</td><td>Seed 1</td><td>Seed 2</td><td> $\mathrm { M e a n } \pm \mathrm { s a m p l e } \mathrm { S D }$ </td></tr><tr><td rowspan="3">Distinct</td><td>ESN</td><td> $4 . 5 3 6 5 \times 1 0 ^ { - 2 }$ </td><td> $4 . 4 2 3 2 \times 1 0 ^ { - 2 }$ </td><td> $4 . 4 4 2 5 \times 1 0 ^ { - 2 }$ </td><td> $( 4 . 4 6 7 4 \pm 0 . 0 6 0 6 ) \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Conceptor</td><td> $4 . 7 6 9 3 \times 1 0 ^ { - 2 }$ </td><td> $4 . 2 9 1 8 \times 1 0 ^ { - 2 }$ </td><td> $4 . 1 8 0 9 \times 1 0 ^ { - 2 }$ </td><td> $( 4 . 4 1 4 0 \pm 0 . 3 1 2 6 ) \times 1 0 ^ { - 2 }$ </td></tr><tr><td>HyperReservoir</td><td> $3 . 2 9 5 0 \times 1 0 ^ { - 2 }$ </td><td> $3 . 9 1 7 6 \times 1 0 ^ { - 2 }$ </td><td> $3 . 1 1 4 6 \times 1 0 ^ { - 2 }$ </td><td> $( 3 . 4 4 2 4 \pm 0 . 4 2 1 3 ) \times 1 0 ^ { - 2 }$ </td></tr><tr><td rowspan="3">Related</td><td>ESN</td><td> $4 . 6 5 9 3 \times 1 0 ^ { - 4 }$ </td><td> $5 . 7 9 5 3 \times 1 0 ^ { - 4 }$ </td><td> $3 . 4 6 9 8 \times 1 0 ^ { - 4 }$ </td><td> $( 4 . 6 4 1 5 \pm 1 . 1 6 2 9 ) \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Conceptor</td><td> $4 . 1 5 2 2 \times 1 0 ^ { - 3 }$ </td><td> $3 . 5 6 6 7 \times 1 0 ^ { - 3 }$ </td><td> $8 . 5 2 5 5 \times 1 0 ^ { - 3 }$ </td><td> $( 5 . 4 1 4 8 \pm 2 . 7 0 9 8 ) \times 1 0 ^ { - 3 }$ </td></tr><tr><td>HyperReservoir</td><td> $8 . 7 6 6 5 \times 1 0 ^ { - 5 }$ </td><td> $7 . 3 3 3 3 \times 1 0 ^ { - 5 }$ </td><td> $7 . 5 4 0 6 \times 1 0 ^ { - 5 }$ </td><td> $( 7 . 8 8 0 1 \pm 0 . 7 7 4 6 ) \times 1 0 ^ { - 5 }$ </td></tr><tr><td rowspan="3">Same attractor Conceptor</td><td>ESN</td><td> $4 . 9 3 5 7 \times 1 0 ^ { - 3 }$ </td><td> $5 . 4 0 1 0 \times 1 0 ^ { - 3 }$ </td><td> $3 . 8 8 9 9 \times 1 0 ^ { - 3 }$ </td><td> $( 4 . 7 4 2 2 \pm 0 . 7 7 3 9 ) \times 1 0 ^ { - 3 }$ </td></tr><tr><td></td><td> $6 . 7 3 2 8 \times 1 0 ^ { - 3 }$ </td><td> $7 . 6 2 5 5 \times 1 0 ^ { - 3 }$ </td><td> $1 . 7 9 3 6 \times 1 0 ^ { - 2 }$ </td><td> $( 1 . 0 7 6 5 \pm 0 . 6 2 2 6 ) \times 1 0 ^ { - 2 }$ </td></tr><tr><td>HyperReservoir</td><td> $6 . 6 3 4 5 \times 1 0 ^ { - 4 }$ </td><td> $5 . 3 1 4 8 \times 1 0 ^ { - 4 }$ </td><td> $7 . 3 7 5 6 \times 1 0 ^ { - 4 }$ </td><td> $( 6 . 4 4 1 6 \pm 1 . 0 4 3 9 ) \times 1 0 ^ { - 4 }$ </td></tr></table>

TABLE S4. Test cNMSE for the augmented HyperReservoir as the context-reservoir dimension M is varied under the fixed recurrent-state budget $N + M = 1 2 0$ . Values are the mean and sample standard deviation across three reservoir seeds.
<table><tr><td>M</td><td>Distinct</td><td>Related</td><td>Same attractor</td></tr><tr><td>2</td><td> $( 3 . 9 7 2 2 \pm 0 . 3 1 0 6 ) \times 1 0 ^ { - 2 }$ </td><td> $( 1 . 8 8 0 1 \pm 0 . 3 0 1 3 ) \times 1 0 ^ { - 4 }$ </td><td> $( 1 . 6 1 6 2 \pm 0 . 8 6 2 7 ) \times 1 0 ^ { - 3 }$ </td></tr><tr><td>5</td><td> $( 3 . 5 6 7 6 \pm 0 . 1 2 2 8 ) \times 1 0 ^ { - 2 }$ </td><td> $( 7 . 8 8 0 1 \pm 0 . 7 7 4 6 ) \times 1 0 ^ { - 5 }$ </td><td> $( 6 . 4 4 1 6 \pm 1 . 0 4 3 9 ) \times 1 0 ^ { - 4 }$ </td></tr><tr><td>10</td><td> $( 3 . 4 4 2 4 \pm 0 . 4 2 1 3 ) \times 1 0 ^ { - 2 }$ </td><td> $( 1 . 1 0 4 0 \pm 0 . 1 3 1 8 ) \times 1 0 ^ { - 4 }$ </td><td> $( 6 . 6 4 3 2 \pm 1 . 4 0 8 4 ) \times 1 0 ^ { - 4 }$ </td></tr><tr><td>20</td><td> $( 3 . 5 9 9 7 \pm 0 . 2 9 8 7 ) \times 1 0 ^ { - 2 }$ </td><td> $( 1 . 4 2 4 3 \pm 0 . 1 5 3 0 ) \times 1 0 ^ { - 4 }$ </td><td> $( 8 . 4 3 4 9 \pm 3 . 8 6 1 6 ) \times 1 0 ^ { - 4 }$ </td></tr><tr><td>40</td><td> $( 3 . 7 9 6 2 \pm 0 . 3 3 2 1 ) \times 1 0 ^ { - 2 }$ </td><td> $( 2 . 5 5 7 1 \pm 1 . 2 0 5 2 ) \times 1 0 ^ { - 4 }$ </td><td> $( 7 . 7 2 5 4 \pm 1 . 5 9 9 3 ) \times 1 0 ^ { - 4 }$ </td></tr></table>

Table S5 gives the corresponding test performance.

The Augmented construction gives the lowest mean cNMSE in all three settings. The Strict construction performs particularly poorly for the related-R¨ossler setting. All three readouts use identical reservoir realizations, data partitions, and fitting conventions within each setting.

## S5. DETAILED SAME-ATTRACTOR ANALYSES

Panels $4 ( \mathrm { a } )$ and 4(b) of the main manuscript show onestep prediction errors. For the displayed χ component, the residual is defined as

$$
e _ { \chi , t } = \hat { \chi } _ { t + 1 } - \chi _ { t + 1 } .
$$

The raw prediction traces of the three models closely overlap on the state scale, so the residual representation is used to expose local diferences among the architectures without changing the underlying prediction task or evaluation data.

The representative reservoir realization is chosen without using test performance. Among the reservoir seeds common to the conventional ESN, full-matrix Conceptor, and HyperReservoir, seeds are ranked separately by validation cNMSE for each architecture, and the seed with the lowest mean validation rank across the three models is used for visualization. The same selected seed is used for all models.

The slow and fast panels use the same fixed display window of 80 samples, beginning at sample 150 of the selected test trajectories. With the sampling interval of 0.05 and prediction horizon $H = 1$ , the displayed futuretarget times are approximately 7.55–11.50. The same window is used for all models. The vertical axes of the slow and fast panels are scaled independently and symmetrically around zero because the absolute error magnitudes difer substantially between the two temporal regimes. This scaling afects only the visualization; all quantitative comparisons in the main text use cNMSE evaluated over the valid test set.

## S5.1. Wrong-context intervention

To directly test whether each architecture functionally uses the supplied context, we perform a counterfactual context intervention in the same-attractor temporal-scale task. For true context i and supplied context j, we define

$$
E _ { i j } = \mathrm { c N M S E } \left( \mathrm { t r u e \ c o n t e x t } i , \mathrm { s u p p l i e d \ c o n t e x t } j \right)\tag{S20}
$$

The diagonal entries correspond to prediction with the correct context, whereas the of-diagonal entries correspond to prediction after replacing the supplied context while keeping the observed trajectory unchanged.

The summary quantity reported in Fig. 4(c) of the main manuscript is

$$
R _ { \mathrm { c t x } } = \frac { E _ { 1 1 } + E _ { 2 2 } } { E _ { 1 2 } + E _ { 2 1 } } ,\tag{S21}
$$

which is equivalently the mean correct-context cNMSE divided by the mean wrong-context cNMSE. A value substantially below unity therefore indicates that replacing the supplied context strongly degrades prediction accuracy.

TABLE S5. Readout comparison using the validation-selected N and M of the augmented HyperReservoir for each task. Values are mean test cNMSE and sample standard deviation across three random reservoir realizations.
<table><tr><td>Readout</td><td>Distinct</td><td>Related</td><td>Same attractor</td></tr><tr><td>Concat</td><td> $( 4 . 4 3 2 0 \pm 0 . 3 3 4 1 ) \times 1 0 ^ { - 2 }$ </td><td> $( 2 . 0 7 0 9 \pm 0 . 3 4 4 4 ) \times 1 0 ^ { - 4 }$ </td><td> $( 2 . 0 7 9 5 \pm 0 . 1 2 1 7 ) \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Strict multiplicative</td><td> $( 5 . 3 5 9 2 \pm 0 . 6 9 4 6 ) \times 1 0 ^ { - 2 }$ </td><td> $( 9 . 1 0 1 1 \pm 2 . 2 0 1 8 ) \times 1 0 ^ { - 2 }$ </td><td> $( 1 . 9 7 8 3 \pm 0 . 9 1 8 6 ) \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Augmented</td><td> $( 3 . 4 4 2 4 \pm 0 . 4 2 1 3 ) \times 1 0 ^ { - 2 }$ </td><td> $( 7 . 8 8 0 1 \pm 0 . 7 7 4 6 ) \times 1 0 ^ { - 5 }$ </td><td> $( 6 . 4 4 1 6 \pm 1 . 0 4 3 9 ) \times 1 0 ^ { - 4 }$ </td></tr></table>

The intervention is implemented according to the context-dependent pathway of each architecture. For the context-input ESN, the context channels in the reservoir input are replaced. For the full-matrix Conceptor, the context selects the corresponding context-specific Conceptor matrix. For the HyperReservoir, the explicit context supplied to the context reservoir is replaced while the main reservoir continues to receive the same observed trajectory. For every true/supplied-context pair, the complete input sequence is passed through a new model forward pass. Thus, whenever context afects a recurrent state, that state is recomputed under the counterfactual context rather than reused from the correct-context trajectory.

Figure S1 shows the individual error terms underlying the summary ratio $R _ { \mathrm { c t x } }$ in Fig. 4(c) of the main manuscript. For both the conventional ESN and the HyperReservoir, replacing the supplied context increases the prediction error by several orders of magnitude relative to the corresponding diagonal entries. This separation is especially pronounced for the HyperReservoir, which combines low correct-context errors with large errors under counterfactual context replacement. By contrast, the full-matrix Conceptor exhibits a substantially smaller separation between its diagonal and of-diagonal errors. These matrices therefore provide the direct errorlevel view underlying the smaller $R _ { \mathrm { c t x } }$ values of the ESN and, most strongly, the HyperReservoir.

## S5.2. Full-matrix Conceptor comparison

Let $C _ { \mathrm { s l o w } }$ and $C _ { \mathrm { f a s t } }$ denote the full-matrix Conceptors estimated from training trajectories for the two temporal regimes using the task-level validation-selected aperture $\gamma ^ { * } = 1$ . In the main manuscript, we report the Frobenius cosine similarity between these operators. As an additional descriptive comparison, Fig. S2 shows their ordered eigenvalue spectra for the representative reservoir realization. The spectra largely overlap; however, eigenvalue similarity alone does not establish operator similarity because the corresponding eigenvectors are not considered.

To examine whether the two Conceptors also difer in their spectral filtering profiles, we now also compare their ordered eigenvalue spectra. The ordered eigenvalue spectra shown in Fig. S2 largely overlap across the two regimes, consistent with the high Frobenius cosine similarity reported in Fig. 4(d) of the main manuscript.

TABLE S6. Readout feature dimensions and fitted output coeficients for the recurrent-state allocations used in the readout ablation. The Conceptor storage entry is not included because it is separate from the shared linear readout.
<table><tr><td rowspan="2">Allocation</td><td rowspan="2">Model</td><td colspan="2">Feature Fitted readout</td></tr><tr><td>dimension</td><td>coefficients</td></tr><tr><td rowspan="3"> $N = 1 1 0$ </td><td>Concat</td><td>120</td><td>363</td></tr><tr><td>M = 10 Strict</td><td>1100</td><td>3303</td></tr><tr><td>Augmented</td><td>1220</td><td>3663</td></tr><tr><td rowspan="3"> $N = 1 1 5 ,$ </td><td>Concat</td><td>120</td><td>363</td></tr><tr><td>M = 5 Strict</td><td>575</td><td>1728</td></tr><tr><td>Augmented</td><td>695</td><td>2088</td></tr></table>

## S6. FITTED READOUT DIMENSIONS AND STORED CONTEXTUAL QUANTITIES

The matched recurrent-state budget controls the dimension of the dynamical state but does not equalize the number of fitted output coeficients or stored contextspecific quantities. This section gives the corresponding counts explicitly.

For $D = 3$ outputs and F nonconstant readout fea tures,

$$
\begin{array} { r } { P _ { \mathrm { o u t } } = D ( F + 1 ) } \\ { = 3 ( F + 1 ) } \end{array}\tag{S22}
$$

(S23)

The conventional ESN and the shared Conceptor readout each have $F = 1 2 0$ , giving 363 fitted coeficients. In addition, the two-context Conceptor model stores two $1 2 0 \times 1 2 0$ matrices, corresponding to 28,800 contextspecific matrix entries. Table S6 gives the readout dimensions used in the HyperReservoir comparisons.

## S6.1. Dependence on context dimension

Under the fixed state budget $N + M = 1 2 0$ , the augmented feature dimension is

$$
F _ { \mathrm { a u g } } ( M ) = 1 2 0 + M ( 1 2 0 - M ) .\tag{S24}
$$

The fitted readout count is therefore

$$
P _ { \mathrm { a u g } } ( M ) = 3 \left[ 1 2 1 + M ( 1 2 0 - M ) \right] .\tag{S25}
$$

b

![](images/b1fa24a18c5abafc4da1850860ce4aafc8f596b31f1f3b914612684ee87ea39d.jpg)

![](images/5430c06faa47bd715e1a3c64f09b67f78c70fc86f7779bfa30d9ff18168a07f6.jpg)

![](images/cd2fdcea005853aef041915ee10a4b8a01bfaef16d95d1b7e2fc5db085651785.jpg)

FIG. S1. Counterfactual context dependence in the same-attractor temporal-scale task. Mean cNMSE matrices for (a) the context-input ESN, (b) the full-matrix Conceptor, and (c) the HyperReservoir. Rows indicate the true temporal-scale context and columns indicate the supplied context, with $\nu \in \{ 0 . 5 , 1 . 5 \}$ . Diagonal entries therefore correspond to the correct context, whereas of-diagonal entries correspond to counterfactual context replacement. Each matrix is averaged over the three reservoir realizations after task-level validation-based model selection. A common color scale is used across the three architectures.  
![](images/ce5630696cdfec1f6b23bc81da25279fcd90bf3857c06db380840a0756f1113b.jpg)  
FIG. S2. Ordered eigenvalue spectra of the slow and fast Conceptors in the same-attractor temporal-scale task. The curves show the spectra of $C _ { \mathrm { s l o w } }$ and $C _ { \mathrm { f a s t } }$ for the representative reservoir realization (seed 1), using the task-level validation-selected aperture $\bar { \gamma } ^ { * } = 1$ . Both Conceptors are estimated from post-washout training reservoir states. This comparison is descriptive only: similarity of the eigenvalue spectra does not by itself imply similarity of the corresponding operators, because their eigenvectors are not considered.

Increasing M therefore has two distinct efects under the matched-state constraint. It reallocates recurrent states from the main reservoir to the context reservoir while simultaneously increasing the bilinear readout dimension over the evaluated range. The M sweep is thus not a simple increase in model size.

The conventional ESN readout scales as $\mathcal { O } ( D N _ { \mathrm { t o t } } )$ . A full-matrix Conceptor additionally requires a dense statespace multiplication of order $\mathcal { O } ( N _ { \mathrm { t o t } } ^ { 2 } )$ when the filter is applied. The augmented HyperReservoir constructs an NM bilinear feature block and therefore incurs outputstage work that scales with NM in addition to the linear readout.

TABLE S7. HyperReservoir feature dimension and fitted readout size across the context-dimension sweep under $N +$ $M = 1 2 0$
<table><tr><td>M N</td><td> $F _ { \mathrm { a u g } }$ </td><td> $\mathrm { \mathit { P } _ { a u g } }$ </td></tr><tr><td>2 118</td><td></td><td>356 1071</td></tr><tr><td>5 115</td><td></td><td>695 2088</td></tr><tr><td>10 110 1220 3663</td><td></td><td></td></tr><tr><td>20 100 2120 6363</td><td></td><td></td></tr><tr><td>40 </td><td>80 3320 9963</td><td></td></tr></table>

## S7. SOFTWARE ENVIRONMENT

All final experiments were executed using Python 3.10.14 and PyTorch 2.7.1 on macOS 14.4 (arm64) with an Apple M3 processor (8 CPU cores, 16 GB memory).