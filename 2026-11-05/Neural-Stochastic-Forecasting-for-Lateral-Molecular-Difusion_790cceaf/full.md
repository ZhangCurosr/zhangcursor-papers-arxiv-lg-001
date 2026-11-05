# Neural Stochastic Forecasting for Lateral Molecular Difusion Simulations

Jacob Ede Levine<sup>1</sup>, Mario Hidalgo-Soria<sup>2</sup>, Chen Yun Wen<sup>2</sup>, Yun Lyna Luo<sup>2\*</sup>, Sai Chandra Kosaraju<sup>2\*</sup>

<sup>1\*</sup>Department of Computer Science, California State Polytechnic University, Pomona, 91768, California, USA.

<sup>2</sup>Department of Biotechnology and Pharmaceutical Sciences, Western University of Health Sciences, Pomona, 91766, California, USA.

\*Corresponding author(s). E-mail(s): luoy@westernu.edu; skosaraju@cpp.edu;

Contributing authors: jlevine@cpp.edu; mhidalgosoria@westernu.edu; chenyun.wen@westernu.edu;

## Abstract

Predicting the long-time difusion of membrane proteins is challenging because molecular dynamics (MD) simulations are computationally expensive, while experimental single-particle tracking is often limited by finite observation times. Here, we introduce a neural stochastic (NS) model for forecasting the collective difusion of membrane proteins from coarse-grained MD trajectories. The model combines a transformer-based representation of protein configurations, recurrent networks operating at multiple temporal scales, and a stochastic latent process with learned memory. A cross-attention decoder then generates correlated protein displacements while accounting for the instantaneous system geometry and periodic boundary conditions. We evaluate the model in two membrane systems: seven interacting PIEZO channels and a mixed system containing seven PIEZO and ten TREK channels. For the seven PIEZO system, the NS model reproduces the anomalous difusion exponent measured in MD simulations α ∼ 1, substantially improving on a previously developed variational autoregressive model that exhibits superdifusive behavior. The model also generates trajectories extending to 40,000 frames and captures the distinct difusion characteristics of PIEZO and TREK populations in the mixed system. Comparisons of time-averaged meansquare displacements, Van Hove displacement distributions, and non-Gaussian parameters show that the model preserves key statistical features of the simulated dynamics. These results establish neural stochastic modeling as a promising

approach for generating long trajectories of multi-protein membrane difusion and extending the statistical analysis of molecular dynamics simulations.

Keywords: Keyword1, Keyword2, Keyword3, Keyword4

## 1 Introduction

Protein membrane difusion is a widely studied phenomenon within the realm of cellular transport [1]. Its biological significance is crucial for understanding vital cellular activities like cell signaling [2], and membrane domain organization [3]. Further, the study of protein membrane difusion has been applied to identify disease mechanisms in the cell [4] and drug targeting [5].

Traditionally single particle tracking (SPT) experiments are implemented to obtain trajectories of individual proteins in the plasma membrane [6]. SPT techniques allow to reconstruct individual membrane protein time series with nano-meter scale localization precision and mili-second temporal resolution [7]. Despite its high spatiotemporal resolution, fluorescence based SPT is generally limited by photobleaching, blinking, and molecular loss from the observation region, resulting in individual trajectories that typically span tenth of seconds [8], although specialized labeling and imaging can extend observation times to minutes [8].

Due to these limitations exhibited by SPT experiments, in silico methods like molecular dynamics (MD) simulations, are commonly employed to investigate protein membrane difusion and other cellular dynamics over longer time scales (minutes). However MD simulations are highly expensive in terms of computational power since they represent solving numerically all the equations of motion for the atoms forming the set of proteins in question.

Recently in [9] we introduced a collection of variational generative models to forecast protein membrane difusion. Showing that a stochastic latent representation in a variational auto-regressive (VAR) architecture was good enough to capture shorttime multi-protein membrane difusion. Nevertheless, these models were not able to replicate/forecast long time dynamics.

In this work we introduce a physics informed neural stochastic model which let us to forecast long time transport of diferent protein membrane systems. Our model is tested in a multi-protein system with 7 PIEZO channels difusing in a bi-lipid membrane [9, 10]. Then our model is implemented to predict the difusion in a mixed multi-protien system where 7 PIEZO channels interact and difuse altogether with 10 TREK channels [11]. We show how the inclusion of a physics informed model within a neural stochastic framework is able to describe accurate long-term difusion regimes and forecast MD trajectories four times more in length compare with the ground truth.

## 2 Methods

The model describes the motion of membrane proteins with a latent stochastic process coupled to neural networks that encode the protein configuration and its history.

The neural networks determine the mean of the latent process and convert each sample of the latent state into displacements of the proteins. The model is trained by matching statistics of physical observables between generated trajectories and coarsegrained molecular dynamics (MD) trajectories. Section 2.1 formulates the model, and Section 2.2 describes the MD datasets. Sections 2.3 to 2.9 describe the training data and the network components, and Sections 2.10 to 2.15 describe the training loss, the optimization, the evaluation, the model selection and the metrics.

## 2.1 Neural stochastic diferential equation

Each MD trajectory is a sequence of T frames, $\{ \mathbf { X } ^ { ( 0 ) } , \ldots , \mathbf { X } ^ { ( T - 1 ) } \}$ , where the superscript t indexes the frame. Each frame $\mathbf { X } ^ { ( t ) } \in \dot { \mathbb { R } } ^ { N \times 3 }$ contains the three-dimensional positions of the $N$ proteins of the system. Row i of a frame, $\mathbf { X } _ { i } ^ { ( t ) } \in \mathbb { R } ^ { 3 }$ , is the position of protein $i = 1 , \ldots , N$ in <sup>˚</sup>A. The position of each protein is defined for each system in Section 2.2.

The model represents the efect of the unrecorded degrees of freedom on the protein motion through a latent stochastic state. This state is propagated forward in time under a learned model of its fluctuations. The recorded positions are treated as a stochastic and, in general, non-Markovian process, in which the future motion of a protein depends on its past trajectory and not only on its current configuration. The MD trajectories are generated by numerical integration of the equations of motion of a coarse-grained force field whose potential couples the proteins to the surrounding lipids. The reduced trajectories retain one position per protein and do not contain the lipid coordinates, the internal coordinates of the proteins or the forces acting on the proteins.

Each trajectory generated by the model, referred to as a rollout, is initialized from a single MD configuration. After the initial configuration, every input to the model is the output of the model at the previous step, during both training and inference. Unlike a next-frame predictor, the model is not conditioned on a window of observed frames.

Each step of a rollout advances the configuration by one frame through five modules applied in sequence. The modules are a transformer encoder, a multirate bank of long short-term memory (LSTM) cells, a latent transition head, a latent stochastic integrator and a cross-transformer decoder. Each module is summarized in the following paragraphs and described in detail in Sections 2.5 to 2.9.

The transformer encoder (Section 2.5) converts the current frame into a geometry vector, which is the input of the LSTM bank and the decoder. The input of the encoder is one vector per protein, which encodes the absolute position of the protein at the initial frame and the most recent displacement of the protein at every later step. A self-attention layer exchanges information among the proteins and produces one feature vector per protein. The geometry vector consists of these feature vectors and the distances between all pairs of proteins.

The multirate LSTM bank (Section 2.6) summarizes the history of the geometry vector over multiple time scales. The bank consists of LSTM cells that are updated at diferent temporal strides, from every frame to once every 300 frames. A cell with stride λ is updated once every λ frames and holds its state between updates.

The latent transition head (Section 2.7) maps the output of the LSTM bank to the mean of a latent state. The latent state has six components per protein, and these components have no prescribed physical meaning. Each component of the mean depends on the LSTM states of all proteins.

The latent stochastic integrator (Section 2.8) advances the latent state by one step of a discrete Ornstein-Uhlenbeck (OU) process. At each step, the latent state relaxes toward the mean produced by the transition head and receives a Gaussian random increment. The process has two learned parameters, a memory coeficient and a noise amplitude. The memory coeficient sets the fraction of the previous deviation from the mean that is retained at each step, and the noise amplitude sets the magnitude of the random increment.

The cross-transformer decoder (Section 2.9) converts the latent state into one displacement per protein. In its attention layer, the attention weights are computed from the geometry vector, and the information exchanged between proteins is taken from their latent states. Through this layer, each protein combines its own latent state with the latent states of the other proteins. For each protein, the decoder outputs the mean and a full 3 × 3 covariance matrix of a Gaussian distribution of the three-dimensional displacement, which includes correlations between the Cartesian components. One displacement is sampled from this distribution for each protein, and the protein position is advanced by this displacement and wrapped into the periodic simulation box.

Temporal correlations over a rollout are represented by the LSTM bank and the latent process. The LSTM bank represents temporal correlations through the mean of the latent state. The latent process represents temporal correlations through its memory coeficient, to an extent that increases with the value of this coeficient. After the initial step, the geometry vector is computed only from the most recent displacement of each protein and the current pair distances. The decoder retains no information between steps and determines only the instantaneous mean and covariance of the displacement.

The model is trained autoregressively, in the same mode used at inference, without teacher forcing. Teacher forcing replaces the predicted frame with the measured frame as the input at each training step [12].

The loss function compares statistics of physical observables computed over a rollout with the same statistics computed over the corresponding MD trajectory. Apart from the initial configuration, the MD trajectory enters training only through this loss. All loss terms except one compare distributions or averages rather than individual frames. The single exception, defined in Section 2.10, is a pointwise comparison of the products of single-frame displacements used to compute the velocity autocorrelation function.

Training rollouts and evaluation rollouts difer only in their length, in the use of dropout and in the arithmetic precision (Sections 2.11 and 2.12). A training rollout spans 2,999 steps, and an evaluation rollout spans 40,000 steps. Dropout [13], which sets randomly selected activations to zero, is applied during training and disabled during evaluation. Training uses mixed-precision arithmetic [14] with the bfloat16 format [15], a 16-bit floating-point format with the exponent range of single precision, whereas evaluation uses single precision.

The evaluation in Section 2.12 compares statistics of evaluation rollouts with those of the MD trajectories at lag times of up to 5,000 frames. Model selection and reporting use this evaluation (Section 2.14).

## 2.2 Molecular dynamics datasets

We use two coarse-grained systems, PIEZO and PIEZO-TREK, summarized in Table 1. Both systems were simulated with LAMMPS [16], and each trajectory is reduced to one position per protein per frame. The PIEZO system contains seven PIEZO channels, and the PIEZO-TREK system contains seven PIEZO channels and ten TREK channels. The same architecture and the same training procedure are applied to both systems. The systems difer in the number of protein species and in the loss weights of Section 2.10.

The species of a protein determines which parameters of the model it shares with other proteins. Each species has its own input projection in the transformer encoder, its own LSTM bank, its own latent dynamics parameters and its own output heads in the decoder (Sections 2.5 to 2.9). Proteins of the same species share these parameters, and the remaining parameters are common to all proteins.

Table 1 Coarse-grained systems. Frames are counted after the removal of duplicated frames described in the text. $L _ { \mathrm { r a w } }$ is the raw in-plane box edge, $\sigma _ { \Delta }$ the displacement scale defined after Equation 1, and $L = L _ { \mathrm { r a w } } / \sigma _ { \Delta }$ the dimensionless normalized box edge of Equation 2.
<table><tr><td></td><td>PIEZO</td><td>PIEZO-TREK</td></tr><tr><td>Proteins N</td><td>7</td><td>17</td></tr><tr><td>Composition</td><td>7 PIEZO</td><td>7 PIEZO, 10 TREK</td></tr><tr><td>Species</td><td>1</td><td>2</td></tr><tr><td>Relabeling symmetry</td><td>all permutations</td><td>within each species</td></tr><tr><td>Replicas used</td><td>2</td><td>2</td></tr><tr><td>Frames, replica 1</td><td>12,578</td><td>11,001</td></tr><tr><td>Frames, replica 2</td><td>7,103</td><td>5,931</td></tr><tr><td>Training frames</td><td colspan="2">first 3,000 frames of replica 1</td></tr><tr><td>MD steps per frame, replica 1</td><td>10,000</td><td>not recorded</td></tr><tr><td>MD steps per frame, replica 2</td><td>10,000</td><td>10,000</td></tr><tr><td>In-plane box edge  $L _ { \mathrm { r a w } }$ </td><td>3440.3686 Å</td><td>3400 Å</td></tr><tr><td>Displacement scale  $\sigma _ { \Delta }$ </td><td>4.018202 Å</td><td>6.298365 Å</td></tr><tr><td>Normalized box L</td><td>856.196</td><td>539.823</td></tr><tr><td>Protein pairs</td><td>21</td><td>136</td></tr></table>

The PIEZO system contains seven copies of a coarse-grained elastic-network model [10] of the mechanosensitive channel PIEZO [17], difusing in a large tension-free bilayer. The model is a three-fold symmetric trimer of 1869 beads built on the cryoelectron microscopy structure of mouse PIEZO2 (PDB 6KG7) [18]. At this resolution, the model does not distinguish PIEZO1 from PIEZO2 [10].

The position of each PIEZO channel is the position of its central marker bead. Each channel carries four marker beads, one at the center and three on the arms. The three arm markers, whose positions fluctuate under constant pressure, are not used. The central markers are unwrapped across the periodic boundary, the initial frames of each replica are removed, and the production segments of each replica are joined into one trajectory with one unwrapped position per protein per frame.

Two independent replicas of the PIEZO system are used. Replica 1 has 12,632 frames, or 12,578 frames after the removal of duplicated frames described below, and replica 2 has 7,103 frames. Frames are recorded every 10,000 MD steps, which corresponds to 0.5 ns per frame with an integration time step of 50 fs.

The seven PIEZO channels of the PIEZO system are identical copies of one protein and are treated as one species. The statistics of the system are invariant under any permutation of the seven protein indices.

Both PIEZO replicas are wrapped at the in-plane box edge of replica 1, 3440.3686 <sup>˚</sup>A. The in-plane box edge of replica 2 is 3439.5151 <sup>˚</sup>A, which difers from that of replica 1 by 0.853 <sup>˚</sup>A, or 0.025 percent. The PIEZO positions are unwrapped before they are wrapped at the common edge. Single-protein steps are not afected by the diference in box edge, and only pair separations across the box boundary are afected.

The PIEZO-TREK system contains seven PIEZO channels and ten TREK channels difusing together in one bilayer with an in-plane edge of 3400 <sup>˚</sup>A. TREK channels are two-pore-domain potassium channels of the TREK subfamily [19]. The system contains 17 proteins and 136 protein pairs.

The PIEZO and TREK channels of the PIEZO-TREK system are treated as two species. The statistics of the system are invariant under permutations of the seven PIEZO indices and under permutations of the ten TREK indices, and they are not invariant under an exchange of a PIEZO channel with a TREK channel.

Each protein of the PIEZO-TREK system is represented by one position per frame. The simulation provides four three-dimensional points for each PIEZO channel and one point for each TREK channel. The position of a PIEZO channel is the first of its four points, and the position of a TREK channel is its single point. These positions are wrapped and scaled with Equation 2 without further processing.

Replica 1 of the PIEZO-TREK system has 11,003 frames, or 11,001 frames after the removal of duplicated frames described below. The frame stride of this replica in MD steps is not recorded, and results for this system are reported in frames.

Replica 2 of the PIEZO-TREK system is converted to the box of replica 1. This replica has 5,931 frames recorded every 10,000 MD steps, after the removal of 71 frames that a restarted run records twice. It was simulated in a box with an in-plane edge of 3296.94 <sup>˚</sup>A and an edge of 600 <sup>˚</sup>A along the membrane normal. Its minimumimage steps are computed in this box, accumulated from its first frame, expressed in the length unit of replica 1 and wrapped into the box of replica 1, without rescaling of the coordinates. This conversion preserves every single-protein displacement exactly and does not preserve the separation of two proteins across the box boundary.

Replica 2 of the PIEZO-TREK system is used only in the evaluation reference of Section 2.12 and in the allowance of Section 2.13. It supplies neither training frames nor initial conditions. The pooling of replica 2 with replica 1 assumes that the two replicas share a frame stride, which is not recorded for replica 1, and treats the diference in box size as part of the variation between replicas.

Frames that a restarted MD run records twice are removed from the trajectories. Replica 1 of the PIEZO system is joined from production segments that overlap at five restart boundaries. For every MD step that is recorded more than once, the last copy is kept, and the replica is reduced from 12,632 to 12,578 frames at a constant stride of 10,000 MD steps. The first 7,102 frames, which include all training frames, are not afected. In replica 1 of the PIEZO-TREK system, the two frames that exactly repeat their predecessor are removed, and one of them lies within the training frames.

The coordinates are wrapped into the periodic simulation box and divided by a fixed displacement scale $\sigma _ { \Delta }$ . The scale is computed from the single-step minimumimage displacements

$$
\Delta \mathbf { X } _ { i } ^ { ( t ) } = \mathbf { X } _ { i } ^ { ( t + 1 ) } - \mathbf { X } _ { i } ^ { ( t ) } - L _ { \mathrm { r a w } } { \mathrm { ~ r o u n d } } \left( { \frac { \mathbf { X } _ { i } ^ { ( t + 1 ) } - \mathbf { X } _ { i } ^ { ( t ) } } { L _ { \mathrm { r a w } } } } \right)\tag{1}
$$

which are evaluated componentwise for every protein and every pair of consecutive frames. Here round(·) denotes rounding to the nearest integer, and $L _ { \mathrm { r a w } }$ is the raw inplane box edge in $\mathrm { \AA } .$ . The subtraction replaces each step by the shortest step between periodic images and removes the jumps of wrapped coordinates by integer multiples of the box edge. The displacement scale $\sigma _ { \Delta }$ , in $\mathrm { { \AA } }$ , is the standard deviation about the mean of all three components of $\Delta \mathbf { X } _ { i } ^ { ( t ) }$ , pooled over all proteins and over the frames specified below for each system.

The normalized coordinates and the normalized box edge are

$$
{ \bf x } _ { i } ^ { ( t ) } = \big ( { \bf X } _ { i } ^ { ( t ) } \mathrm { m o d } L _ { \mathrm { r a w } } \big ) / \sigma _ { \Delta } , \qquad L = L _ { \mathrm { r a w } } / \sigma _ { \Delta } ,\tag{2}
$$

where $\mathbf { x } _ { i } ^ { ( t ) } \in [ 0 , L ) ^ { 3 }$ is the dimensionless wrapped position of protein i and $L$ is the dimensionless box edge used in all subsequent equations. The modulo operation y mod $L _ { \mathrm { r a w } } = y - L _ { \mathrm { r a w } } \lfloor y / L _ { \mathrm { r a w } } \rfloor \in \lbrack 0 , L _ { \mathrm { r a w } } )$ , with ⌊·⌋ the largest integer not greater than its argument, is applied componentwise. The normalized single-step displacements are of order unity.

In both systems, the membrane-normal component is wrapped at the in-plane edge $L _ { \mathrm { r a w } }$ in Equations 1 and 2. The recorded box edge along the membrane normal is 2000 <sup>˚</sup>A for the PIEZO system and 600 <sup>˚</sup>A for replica 2 of the PIEZO-TREK system, and it is not recorded for replica 1 of the PIEZO-TREK system. All membranenormal steps in these trajectories are shorter than $L _ { \mathrm { r a w } } / 2$ , and Equation 1 leaves them unchanged.

The displacement scale is computed once for each system and applied to all replicas of that system. For the PIEZO system, $\sigma _ { \Delta }$ is computed over both replicas pooled, with replica 1 taken before the removal of duplicated frames, and the pool includes the steps of replica 1 across its five overlapping restart boundaries. For the PIEZO-TREK system, $\sigma _ { \Delta }$ is computed over the 11,003 frames of replica 1, and replica 2 is expressed in the same unit. In both systems, $\sigma _ { \Delta }$ is computed over all species jointly rather than per species. Physical units are recovered by multiplication of a coordinate or a displacement by $\sigma _ { \Delta }$

Within the model, positions are wrapped and displacements are computed in the normalized box of edge L. All three normalized coordinates are wrapped at the inplane edge L (Equation 35). Displacements are computed with the minimum-image map [20] m $\mathbf { \partial } ( \mathbf { y } ) = \mathbf { y } - L$ round $. ( \mathbf { y } / L )$ , which is applied componentwise to a diference $\mathbf { y }$ of normalized positions.

Lateral statistics are computed on unwrapped lateral positions $\mathbf { r } _ { i } ^ { ( t ) } \in \mathbb { R } ^ { 2 }$ , which are accumulated from minimum-image steps,

$$
\begin{array} { r } { \mathbf { r } _ { i } ^ { ( t _ { 0 } ) } = \mathbf { x } _ { i , \mathrm { x y } } ^ { ( t _ { 0 } ) } , \qquad \mathbf { r } _ { i } ^ { ( t + 1 ) } = \mathbf { r } _ { i } ^ { ( t ) } + \mathrm { m i } \bigl ( \mathbf { x } _ { i , \mathrm { x y } } ^ { ( t + 1 ) } - \mathbf { x } _ { i , \mathrm { x y } } ^ { ( t ) } \bigr ) , } \end{array}\tag{3}
$$

where the subscript xy selects the two lateral components and $t _ { 0 }$ is the first frame of the trajectory. Equation 3 recovers the continuous lateral path when every singleframe lateral displacement is shorter than $L / 2$ . Every lateral step of the training data satisfies this condition in both systems.

## 2.3 Training data, symmetry augmentation and sampling

Each model is trained on the first 3,000 frames of replica 1 of its system. For the PIEZO system, these frames span 1.5 µs. The training views are generated from these frames by three symmetry operations, namely in-plane rotations and reflections, inplane translations and time reversal. These operations produce equivalent views of the same frames and add no information beyond the MD data.

The dihedral group $D _ { 4 }$ of the square provides eight views of each trajectory. The group consists of the four in-plane rotations by multiples of $9 0 °$ , with and without reflection. The in-plane box of both systems is square and periodic. The rotations and reflections act on the lateral coordinates about the center of the box, the result is wrapped into the box, and the membrane-normal coordinate is unchanged.

Each $D _ { 4 }$ view is shifted by 1,500 rigid in-plane translations, which are exact symmetries of the periodic box. The translations consist of the identity and 1,499 shifts drawn uniformly from $[ 0 , L ) ^ { 2 }$ with a fixed seed, and the lateral coordinates are wrapped after each shift. The same 1,500 shifts are applied to every element of $D _ { 4 }$ . The translations change the absolute positions that the transformer encoder receives at the initial frame (Section 2.5).

Each view is also used with its frames in reversed order. Every statistic of the evaluation in Section 2.12 is even in the displacement and takes the same value on a trajectory and on its time reversal. The cohesion term of the loss (Section 2.10) matches signed increments of pair distances, which change sign under time reversal. With both directions in the training set, this term matches a symmetrized distribution of these increments. A model trained on both directions learns a time-symmetrized transition, whether or not the MD dynamics are reversible.

The training set of each system consists of $8 \times 1 , 5 0 0 \times 2 = 2 4 , 0 0 0$ views of the same 3,000 frames, half in forward order and half in reversed order. Particle relabeling, which is a symmetry within each species (Table 1), is not used for augmentation.

Each training item is one of the 24,000 views, selected uniformly at random, and uses all 3,000 frames of the view as one window. Within a window, frame 0 is the initial condition and frames 1 to 2,999 are the targets of Section 2.10. An epoch draws 960 windows in batches of 64, which gives 15 optimizer steps per epoch.

No validation split is held out. Model selection uses the long-horizon evaluation of Section 2.12.

## 2.4 Notation and network components

Vectors are columns set in bold, and matrices are set in bold capitals. The symbol ⊙ denotes elementwise multiplication, and [· , ·] denotes the concatenation of vectors into a longer vector. The Euclidean norm is denoted ∥ · ∥, and | · | denotes the absolute value of a number or the number of elements of a set. For a vector, mean(·) and $\mathrm { V a r } ( \cdot )$ denote the mean and the variance of its entries. The brackets $\langle \cdot \rangle$ denote an average over the set named in the subscript or in the text. The function $\operatorname { a t a n 2 } ( y _ { 1 } , y _ { 2 } )$ returns the angle in $( - \pi , \pi ]$ whose tangent is $y _ { 1 } / y _ { 2 }$ , with the quadrant determined by the signs of both arguments. Every use of atan2 in the following subsections has a nonnegative first argument, and the resulting angles lie in [0, π]. A softmax over an index set normalizes the exponentials of its arguments to a sum of one over that set.

Upright x, y and z denote the coordinate axes, and z is the membrane normal. Each protein i belongs to a species $s ( i ) , \mathcal { P } _ { s }$ is the set of proteins of species $s ,$ and $s$ is the set of species of a system.

Four elementwise nonlinear functions are used in the model. Each function maps a real number $y$ to a real number and is applied independently to every entry of its argument,

$$
\mathrm { s i g m o i d } ( y ) = \frac { 1 } { 1 + \exp ( - y ) } \in ( 0 , 1 ) , \qquad \mathrm { s o f t p l u s } ( y ) = \ln \bigl ( 1 + \exp ( y ) \bigr ) \in ( 0 , \infty ) ,\tag{4}
$$

$$
\mathrm { S i L U } ( y ) = y \ \mathrm { s i g m o i d } ( y ) , \qquad \mathrm { G E L U } ( y ) = y \Phi ( y ) ,\tag{5}
$$

where $\Phi$ is the cumulative distribution function of the standard normal distribution. The sigmoid maps any real number into the open unit interval and is used for quantities that represent fractions or gates. The softplus is a smooth approximation of max $( 0 , y )$ and is used for quantities that must be strictly positive, such as standard deviations. The SiLU [21, 22] and GELU [21] functions are smooth activation functions that pass large positive arguments almost unchanged and suppress negative arguments. These two functions serve as the activation functions of the model in place of rectified linear units.

Layer normalization [23] rescales a vector to zero mean and unit variance across its components and applies a learned afine transformation,

$$
\operatorname { L N } ( \mathbf { y } ) = \gamma \odot { \frac { \mathbf { y } - \operatorname { m e a n } ( \mathbf { y } ) } { \sqrt { \operatorname { V a r } ( \mathbf { y } ) + 1 0 ^ { - 5 } } } } + \beta ,\tag{6}
$$

in which $\mathbf { y }$ is the input vector, $\gamma$ and $\beta$ are learned vectors of the same length, and the constant $1 0 ^ { - 5 }$ prevents division by zero. Each layer normalization in the model has its own $\gamma$ and $\beta$ and is identified by a subscript.

Every learned weight matrix is denoted W and every learned bias b, each with a superscript that identifies its layer. All parameters without a stated initialization use the default initialization of PyTorch.

Spectral normalization [24] is applied to the first linear layer of the transition head, of the value map of the decoder and of the mean and covariance heads of the decoder. For a weight matrix W with largest singular value $\| \mathbf { W } \| _ { 2 }$ , the spectrally normalized weight is

$$
\mathrm { s n } ( \mathbf { W } ) = \mathbf { W } / \| \mathbf { W } \| _ { 2 } ,\tag{7}
$$

which satisfies $\| \operatorname { s n } ( \mathbf { W } ) \mathbf { y } \| \leq \| \mathbf { y } \|$ for every y.

The divisor $\| \mathbf { W } \| _ { 2 }$ is estimated by power iteration, which repeatedly multiplies a vector by $\mathbf { W } ^ { \top } \mathbf { W }$ and converges to the leading singular vector. The estimate advances by one iteration at each evaluation of the layer during training, which occurs at every rollout step, and is held fixed during evaluation. Only the first linear layer of each path is constrained, and the composite map is not constrained to be non-expansive.

## 2.5 Transformer encoder

One attention head combines value vectors with weights computed from scaled dot products of queries and keys [25]. For matrices Q, K and V whose rows are the query, key and value projections of a set of items, the head computes

$$
{ \mathrm { A t t n } } ( \mathbf { Q } , \mathbf { K } , \mathbf { V } ) = { \mathrm { s o f t m a x } } \left( { \frac { \mathbf { Q } \mathbf { K } ^ { \top } } { \sqrt { d } } } \right) \mathbf { V } ,\tag{8}
$$

where d is the number of components of the query and key of one head and the softmax is taken along each row. The weights that one item places on the other items are non-negative and sum to one, and each output row is a weighted average of value rows.

Several attention heads operate in parallel on separate learned projections, and their outputs are concatenated and mixed by a learned matrix. An attention block adds its output to its input through a residual connection [26] and is followed by a feedforward network applied to each item separately. In an attention layer, the vector that represents one item is called a token.

The transformer encoder is a single encoder layer of the original transformer [25], with layer normalization after each residual connection, applied to the set of protein tokens without positional encoding. In this form, the encoder is the set attention block of the Set Transformer [27], which treats its tokens as an unordered set. The encoder has no mask and no edge list, and every protein attends to every protein, including itself. On the complete graph of proteins, the encoder is a form of graph attention [28], with weights from the scaled dot products of Equation 8 between node features instead of the additive score of the original graph attention network, and without restriction of the softmax to the neighbors of a node.

The attention weights of the encoder do not depend on the distances between proteins. After the initial frame, the node features contain only the most recent step of each protein, and the attention weights depend on this motion and on the species of the proteins but not on their positions. The spatial arrangement of the proteins enters the model only through the pairwise distances, which are appended after the attention block (Equation 13) and are used by the LSTM bank and the decoder.

The node input of protein i at step t is a nine-component vector,

$$
\mathbf { f } _ { i } ^ { ( t ) } = \left\{ \begin{array} { l l } { [ \sin \bigl ( 2 \pi \mathbf { x } _ { i } ^ { ( 0 ) } / L \bigr ) , \cos \bigl ( 2 \pi \mathbf { x } _ { i } ^ { ( 0 ) } / L \bigr ) , \ \mathbf { 0 } _ { 3 } ] , } & { t = 0 , } \\ { \bigl [ \mathbf { 0 } _ { 6 } , \ \mathrm { m i } \bigl ( \mathbf { x } _ { i } ^ { ( t ) } - \mathbf { x } _ { i } ^ { ( t - 1 ) } \bigr ) \bigr ] , } & { t \geq 1 , } \end{array} \right.\tag{9}
$$

in which the sine and cosine act on each of the three components, the three sines precede the three cosines, and ${ \bf 0 } _ { 3 }$ and $\mathbf { 0 } _ { 6 }$ are vectors of three and six zeros. At the initial frame, the node input encodes the absolute position of the protein through a sinecosine embedding, which is continuous across the periodic boundary and diferentiable everywhere. At every later step, the node input contains only the minimum-image step of the protein. Absolute position enters the encoder only at the initial frame, and no gradient passes through the sine and cosine of the positions predicted by the model.

Each node input is projected to a 63-component node feature by an afine map shared by the proteins of one species, and the node features form the rows of the node matrix $\dot { \mathbf { H } } ^ { ( t ) } \in \mathrm { \overline { { \mathbb { R } } } } ^ { N \times 6 3 }$ ，

$$
\mathbf { H } _ { i , : } ^ { ( t ) } = \left( \mathbf { W } _ { s ( i ) } ^ { \mathrm { n o d e } } \mathbf { f } _ { i } ^ { ( t ) } + \mathbf { b } _ { s ( i ) } ^ { \mathrm { n o d e } } \right) ^ { \top } ,\tag{10}
$$

where $\mathbf { H } _ { i . : } ^ { ( t ) }$ is row $i ,$ and $\mathbf { W } _ { s } ^ { \mathrm { n o d e } }$ and $\mathbf { b } _ { s } ^ { \mathrm { n o d e } }$ are the projection parameters of species s. The PIEZO system has one projection, and the PIEZO-TREK system has one projection per species.

The proteins exchange information in one attention block and one feedforward block, each with a residual connection followed by layer normalization,

$$
\begin{array} { r } { \hat { \mathbf { H } } ^ { ( t ) } = \mathrm { L N } _ { \mathrm { G 1 } } \Big ( \mathbf { H } ^ { ( t ) } + \mathrm { M H A } \big ( \mathbf { H } ^ { ( t ) } \big ) \Big ) , } \end{array}\tag{11}
$$

$$
\tilde { \mathbf { H } } ^ { ( t ) } = \mathrm { L N } _ { \mathrm { G 2 } } \Big ( \hat { \mathbf { H } } ^ { ( t ) } + \mathrm { G E L U } \big ( \hat { \mathbf { H } } ^ { ( t ) } \mathbf { W } ^ { \mathrm { G 1 } \top } \big ) \mathbf { W } ^ { \mathrm { G 2 } \top } \Big ) ,\tag{12}
$$

where layer normalization acts on each row. The multi-head attention MHA applies Equation 8 in each of 21 heads, with query, key and value projections of H<sup>(t)</sup> from 63 to 3 components per head $( d = 3 )$ , and concatenates the 21 head outputs and multiplies them by a learned 63 × 63 matrix. The feedforward weights $\mathbf { W } ^ { \mathrm { G 1 } }$ and $\mathbf { W } ^ { \mathrm { G 2 } }$ map 63 components to 126 and back, and biases are omitted from Equations 11 and 12. The transpose of row i of $\tilde { \mathbf { H } } ^ { ( t ) }$ , denoted $\mathbf { p } _ { i } ^ { ( t ) } \in \mathbb { R } ^ { 6 3 }$ , is the protein block of protein i.

The geometry vector combines the protein blocks with the minimum-image distances between all protein pairs. The encoder computes the $N ( N - 1 ) / 2$ threedimensional pair distances

$$
\begin{array} { r l } & { \rho _ { i j } ^ { ( t ) } = \sqrt { \left\| \operatorname* { m i } \bigl ( \mathbf { x } _ { i } ^ { ( t ) } - \mathbf { x } _ { j } ^ { ( t ) } \bigr ) \right\| ^ { 2 } + 1 0 ^ { - 8 } } \nearrow \angle L , \qquad 1 \leq i < j \leq N , } \end{array}\tag{13}
$$

and collects them in the vector $\pmb { \rho } ^ { ( t ) } \in \mathbb { R } ^ { N ( N - 1 ) / 2 }$ . The geometry vector is the concatenation of the protein blocks and this vector,

$$
\mathbf { e } ^ { ( t ) } = \big [ \mathbf { p } _ { 1 } ^ { ( t ) } , \dots , \mathbf { p } _ { N } ^ { ( t ) } , \mathbf { \ } \pmb { \rho } ^ { ( t ) } \big ] \in \mathbb { R } ^ { 6 3 N + N ( N - 1 ) / 2 } .\tag{14}
$$

The pair distances are expressed as fractions of the box edge and include the membrane-normal component. The division by L in Equation 13 sets this scale, and the constant $1 0 ^ { - 8 }$ keeps the square root diferentiable at zero separation. The membranenormal coordinate, which no loss term constrains (Section 2.10), enters the model through these distances and through the steps of Equation 9.

The geometry vector is the complete description of one frame that the LSTM bank and the decoder receive. It has 462 components for the PIEZO system and 1207 components for the PIEZO-TREK system (Table 2). Protein i occupies components $6 3 ( i - 1 ) + 1$ through 63i of $\mathbf { e } ^ { ( t ) }$ , and the $N ( N - 1 ) / 2$ distances occupy the final components. The LSTM bank of Section 2.6 uses this layout to route the features of each protein to the bank of its species.

## 2.6 Multirate LSTM bank

Each cell of the bank is an LSTM cell [29] with a forget gate [30]. For an input $\tilde { \mathbf { e } } ,$ a previous hidden state $\mathbf { h } _ { \mathrm { p r e v } }$ and a previous cell state $\mathbf { c } _ { \mathrm { p r e v } }$ , the cell computes three gates and a candidate content,

$$
\begin{array} { r } { \mathbf { g } ^ { \mathrm { i n } } = \mathrm { s i g m o i d } \big ( \mathbf { W } ^ { \mathrm { i n } } [ \tilde { \mathbf { e } } , \mathbf { h } _ { \mathrm { p r e v } } ] + \mathbf { b } ^ { \mathrm { i n } } \big ) , \qquad \mathbf { g } ^ { \mathrm { f g } } = \mathrm { s i g m o i d } \big ( \mathbf { W } ^ { \mathrm { f g } } [ \tilde { \mathbf { e } } , \mathbf { h } _ { \mathrm { p r e v } } ] + \mathbf { b } ^ { \mathrm { f g } } \big ) , } \end{array}\tag{15}
$$

$$
\begin{array} { r } { \mathbf { g } ^ { \mathrm { o u t } } = \mathrm { s i g m o i d } \big ( \mathbf { W } ^ { \mathrm { o u t } } [ \tilde { \mathbf { e } } , \mathbf { h } _ { \mathrm { p r e v } } ] + \mathbf { b } ^ { \mathrm { o u t } } \big ) , \qquad \tilde { \mathbf { c } } = \operatorname { t a n h } \big ( \mathbf { W } ^ { \mathrm { c a n d } } [ \tilde { \mathbf { e } } , \mathbf { h } _ { \mathrm { p r e v } } ] + \mathbf { b } ^ { \mathrm { c a n d } } \big ) , } \end{array}\tag{16}
$$

and updates its two states,

$$
\mathbf { c } = \mathbf { g } ^ { \mathrm { f g } } \odot \mathbf { c } _ { \mathrm { p r e v } } + \mathbf { g } ^ { \mathrm { i n } } \odot { \tilde { \mathbf { c } } } , \qquad \mathbf { h } = \mathbf { g } ^ { \mathrm { o u t } } \odot { \mathrm { t a n h } } ( \mathbf { c } ) .\tag{17}
$$

The three gates take values between zero and one and act as soft switches. The forget gate $\mathbf { g } ^ { \mathrm { f g } }$ sets the fraction of the previous memory $\mathbf { c } _ { \mathrm { p r e v } }$ that is retained, the input gate $\mathbf { g } ^ { \mathrm { i n } }$ sets the fraction of the candidate content $\tilde { \mathbf { c } }$ that is written, and the output gate ${ \bf g } ^ { \mathrm { o u t } }$ sets the fraction of the memory that is exposed as the hidden state h. The cell is written compactly as $\mathrm { L S T M } ( \tilde { \mathbf { e } } , \mathbf { h } _ { \mathrm { p r e v } } , \mathbf { c } _ { \mathrm { p r e v } } ) \mapsto ( \mathbf { h } , \mathbf { c } )$

The LSTM bank consists of seven single-layer LSTM cells with 64-component states, one for each stride $\lambda \in \{ 1 , 3 , 1 0 , 3 0 , 1 0 0 , 2 0 0 , 3 0 0 \}$ . The cells run at diferent sampling rates, a design related to recurrent networks whose modules operate at separate clock rates [31]. There is one bank per species, and all proteins of a species share it. Every entry of ${ \bf b } ^ { \mathrm { f g } }$ is initialized to two, and each cell starts training with its forget gate open.

The input to the bank of protein i consists of the protein block of that protein and the pairwise distances,

$$
\begin{array} { r } { \tilde { \mathbf { e } } _ { i } ^ { ( t ) } = \mathrm { L N } _ { s ( i ) } \big ( [ \mathbf { p } _ { i } ^ { ( t ) } , \pmb { \rho } ^ { ( t ) } ] \big ) \in \mathbb { R } ^ { 6 3 + N ( N - 1 ) / 2 } . } \end{array}\tag{18}
$$

The pairwise distances are common to all proteins, and the normalization statistics are computed separately for each protein. The normalization $\mathrm { L N } _ { s }$ computes its mean and variance over the vector of each protein, and its learned scale and shift are shared within species s, as are the cell weights.

Cell λ of the bank of species $s ( i )$ is updated only at steps that are multiples of its stride and otherwise keeps its states unchanged,

$$
\begin{array} { r } { \left( \mathbf { h } _ { i , \lambda } ^ { ( t ) } , \mathbf { c } _ { i , \lambda } ^ { ( t ) } \right) = \left\{ \begin{array} { l l } { \mathrm { L S T M } _ { s ( i ) , \lambda } \big ( \tilde { \mathbf { e } } _ { i } ^ { ( t ) } , \mathbf { h } _ { i , \lambda } ^ { ( t - 1 ) } , \mathbf { c } _ { i , \lambda } ^ { ( t - 1 ) } \big ) } & { \mathrm { i f ~ } t \mathrm { ~ m o d ~ } \lambda = 0 , } \\ { \big ( \mathbf { h } _ { i , \lambda } ^ { ( t - 1 ) } , \mathbf { c } _ { i , \lambda } ^ { ( t - 1 ) } \big ) } & { \mathrm { o t h e r w i s e , } } \end{array} \right. } \end{array}\tag{19}
$$

where $\mathbf { h } _ { i , \lambda } ^ { ( t ) }$ and $\mathbf { c } _ { i , \lambda } ^ { ( t ) }$ are the states of protein i in the cell of stride λ, $\mathrm { L S T M } _ { s , \lambda }$ is the cell of Equations 15 to 17 with the weights of species s and stride λ, and both states start from zero, $\mathbf { h } _ { i , \lambda } ^ { ( - 1 ) } = \mathbf { c } _ { i , \lambda } ^ { ( - 1 ) } = \mathbf { 0 }$ . The step index t counts from the first frame of a training window or from the initial condition of a rollout, and every cell is updated at $t = 0$ . The proteins of one species are processed together with the same weights.

The states of a cell are held, not reset, at skipped steps. The cell of stride 1 is updated at every step, and the cell of stride 300 is updated once every 300 steps and applies its forget gate once per 300 frames. After the initial frame, each update of a cell receives the single-frame step of Equation 9 and the current pair distances, and not the displacement accumulated over its stride.

The hidden states of all proteins and strides are concatenated in protein order and normalized,

$$
\boldsymbol { \psi } ^ { ( t ) } = \mathrm { L N } _ { \psi } \big ( \big [ \mathbf { h } _ { 1 , 1 } ^ { ( t ) } , \mathbf { h } _ { 1 , 3 } ^ { ( t ) } , \ldots , \mathbf { h } _ { 1 , 3 0 0 } ^ { ( t ) } , \mathbf { h } _ { 2 , 1 } ^ { ( t ) } , \ldots , \mathbf { h } _ { N , 3 0 0 } ^ { ( t ) } \big ] \big ) \in \mathbb { R } ^ { 4 4 8 N } ,\tag{20}
$$

which gives 3136 components for the PIEZO system and 7616 components for the PIEZO-TREK system.

## 2.7 Latent transition head

The latent state $\zeta ^ { ( t ) } \in \mathbb { R } ^ { 6 N }$ has six components per protein, and components $6 ( i - 1 ) + 1$ through 6i form the slice $\zeta _ { i } ^ { ( t ) } \in \mathbb { R } ^ { 6 }$ of protein i. The slices have no prescribed physical meaning. The decoder uses slice i as the value of protein i (Section 2.9).

The transition head maps the bank readout to the mean $\pmb { \mu } ^ { ( t ) } \in \mathbb { R } ^ { 6 N }$ toward which the latent state relaxes,

$$
\tilde { \pmb { \mu } } ^ { ( t ) } = \mathbf { W } ^ { \mu 2 } \operatorname { S i L U } \big ( \sin \big ( \mathbf { W } ^ { \mu 1 } \big ) \psi ^ { ( t ) } + \mathbf { b } ^ { \mu 1 } \big ) + \mathbf { b } ^ { \mu 2 } ,\tag{21}
$$

$$
\pmb { \mu } ^ { ( t ) } = 1 0 \operatorname { t a n h } ( \tilde { \pmb { \mu } } ^ { ( t ) } / 1 0 ) ,\tag{22}
$$

where $\mathbf { W } ^ { \mu 1 }$ is a spectrally normalized layer of width $2 2 4 N = \operatorname* { m a x } ( 4 4 8 N / 2 , 6 N )$ and $\mathbf { W } ^ { \mu 2 }$ is a linear layer onto the 6N components. The hidden width is 1568 for the PIEZO system and 3808 for the PIEZO-TREK system. The first layer of the head grows with the square of N and contains most of the parameters of the model in both systems (Table 2). The smooth clamp of Equation 22 acts on each component, bounds the mean to the interval (−10, 10) and is close to the identity for components much smaller than 10 in magnitude. The head outputs only the mean, and the noise of the latent process is generated by the stochastic integrator of Section 2.8.

The transition head is fully connected across proteins, and each slice of $\mu ^ { ( t ) }$ depends on the LSTM states of all proteins. The head receives $\psi ^ { ( t ) }$ with the proteins in a fixed order, and the model as a whole is not invariant to relabeling of the proteins within a species. The transformer encoder, the LSTM banks and the decoder treat the proteins of a species identically.

## 2.8 Latent stochastic integrator

The latent dynamics follow the framework of latent stochastic diferential equations, in which observations are readouts of a hidden state that evolves under a drift and a noise parameterized by neural networks [32, 33]. The model is trained by matching rollout statistics (Section 2.10) instead of the variational objectives of [32, 33]. The stochastic diferential equation serves as the interpretation of the discrete update below.

The Ornstein-Uhlenbeck (OU) process [34] is a mean-reverting process that relaxes exponentially toward its mean under white-noise forcing. Observed at a fixed interval, the OU process is a first-order autoregressive process whose coeficient is the exponential of minus the ratio of the interval to the relaxation time [35].

The latent state advances by one step of a discrete OU process toward the mean of Equation 22, and the slice of protein i is updated as

$$
\zeta _ { i } ^ { ( t ) } = \mu _ { i } ^ { ( t ) } + a _ { s ( i ) } \big ( \zeta _ { i } ^ { ( t - 1 ) } - \mu _ { i } ^ { ( t ) } \big ) + \kappa _ { s ( i ) } \xi _ { i } ^ { ( t ) } , \qquad \xi _ { i } ^ { ( t ) } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { 6 } ) ,\tag{23}
$$

where $\mu _ { i } ^ { ( t ) }$ is the slice of protein i of the mean, $a _ { s }$ is the memory coeficient and $\kappa _ { s }$ the noise amplitude of species s, and $\pmb { \xi } _ { i } ^ { ( t ) }$ is a vector of six independent standard normal numbers drawn at every step, with $\mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { 6 } )$ the standard normal distribution in six dimensions. The update is computed in single precision, and during training with mixed precision its result is rounded to bfloat16 between steps. The update starts from $\zeta _ { i } ^ { ( - 1 ) } = \mathbf { 0 }$ , and the first state is $\zeta _ { i } ^ { ( 0 ) } = ( 1 - a _ { s ( i ) } ) { \pmb \mu } _ { i } ^ { ( 0 ) } + \bar { \kappa } _ { s ( i ) } { \pmb \xi } _ { i } ^ { ( 0 ) }$ . No separate initialization of the latent state is used.

The memory coeficient and the noise amplitude are learned and are constrained to their admissible ranges by

$$
a _ { s } = \mathrm { s i g m o i d } ( \tilde { a } _ { s } ) \in ( 0 , 1 ) , \qquad \kappa _ { s } = \mathrm { s o f t p l u s } ( \tilde { \kappa } _ { s } ) \in ( 0 , \infty ) ,\tag{24}
$$

where $\tilde { a } _ { s }$ and ${ \tilde { \kappa } } _ { s }$ are the unconstrained parameters updated by the optimizer. The parameters are initialized to $a _ { s } = 0 . 8$ and $\kappa _ { s } = 5 . 0$ . The PIEZO system has one pair of latent parameters, and the PIEZO-TREK system has one pair per species. Both $\tilde { a } _ { s }$ and ${ \tilde { \kappa } } _ { s }$ belong to a dedicated optimizer group (Section 2.11).

In Equation 23, the latent state relaxes toward the target $\mu _ { i } ^ { ( t ) }$ , retains the fraction $a _ { s ( i ) }$ of its previous deviation from that target and receives a random increment of typical magnitude $\kappa _ { s ( i ) }$ . The coeficient $a _ { s }$ sets the memory of the process, and $\kappa _ { s }$ sets the strength of its random forcing. For $a _ { s }$ close to zero, the latent state follows its mean, and for $a _ { s }$ close to one, the latent state relaxes slowly toward its mean and accumulates its random increments.

For a fixed mean, Equation 23 is the exact one-frame transition of an OU process with a relaxation time of −1/ ln a frames and a stationary standard deviation about the mean of $\kappa _ { s } / ( 1 - a _ { s } ^ { 2 } ) ^ { 1 / 2 }$ . The mean of the model changes from step to step, and these two quantities describe the fluctuations of the latent state about its moving mean, not a stationary distribution.

The memory coeficient and the noise amplitude are parameterized directly, without a relaxation rate or an integration time step. The model learns both parameters of the discrete process at the sampling interval of the data.

## 2.9 Cross-transformer decoder

Cross-attention is an attention operation in which the keys and values come from a diferent source than the queries [25]. The original transformer refers to this operation as encoder-decoder attention. The queries and keys determine which items exchange information, and the values carry the information that is exchanged. A feedforward block, residual connections and layer normalization complete the layer.

The decoder is a cross transformer that maps the latent state to a displacement of each protein. It consists of one attention block followed by one feedforward block and operates on one token per protein and one additional pair token. The queries and keys are computed from the protein blocks of the transformer encoder, with one additional key computed from the pairwise distances, and the values are computed from the latent state. In encoder-decoder attention, the keys and values come from the same source, and in this decoder the two sources are divided between the keys and the values. The attention weights depend on the protein blocks and the pairwise distances and do not depend on the latent state. A protein does not attend to itself, and its own latent slice is added directly to its token.

The protein token of protein i is computed from its protein block, and the pair token is computed from the pairwise distances,

$$
\mathbf { u } _ { i } = \mathbf { W } ^ { \mathrm { u } } \operatorname { L N } _ { \mathrm { u } } \bigl ( \mathbf { p } _ { i } ^ { ( t ) } \bigr ) + \mathbf { b } ^ { \mathrm { u } } , \qquad \mathbf { u } _ { \mathrm { p a i r } } = \mathbf { W } ^ { \mathrm { p a i r } } \operatorname { L N } _ { \mathrm { p a i r } } \bigl ( \boldsymbol \rho ^ { ( t ) } \bigr ) + \mathbf { b } ^ { \mathrm { p a i r } } ,\tag{25}
$$

where both tokens have 256 components, $\mathbf { W } ^ { \mathrm { u } }$ maps 63 components to 256, and W<sup>pair</sup> maps $N ( N - 1 ) / 2$ components to 256. The step index t is omitted from the quantities of the decoder. No positional, index or species embedding is added, and the tokens of proteins of the same species are exchangeable.

Queries and keys are linear projections of the layer-normalized tokens,

$$
\mathbf { q } _ { i } = \mathbf { W } ^ { \mathrm { q } } \mathrm { L N } _ { \mathrm { q } } ( \mathbf { u } _ { i } ) + \mathbf { b } ^ { \mathrm { q } } , \qquad \mathbf { k } _ { j } = \mathbf { W } ^ { \mathrm { k } } \mathrm { L N } _ { \mathrm { k } } ( \mathbf { u } _ { j } ) + \mathbf { b } ^ { \mathrm { k } } , \quad j \in \{ 1 , \dots , N , \mathrm { p a i r } \} ,\tag{26}
$$

each with 256 components split into four heads of 64 components. Only protein tokens issue queries, and the pair token serves only as a key. In each head, the attention

weight that protein i places on token $j$ is

$$
\omega _ { i j } = \mathrm { s o f t m a x } \left( \frac { \mathbf { q } _ { i } \cdot \mathbf { k } _ { j } } { \sqrt { 6 4 } } \right) , \qquad \mathcal { U } _ { i } = \{ 1 , \dots , N , \mathrm { p a i r } \} \setminus \{ i \} ,\tag{27}
$$

where the dot product runs over the 64 components of the head and the softmax runs over the set $\mathcal { U } _ { i }$ , which excludes protein i. The softmax is evaluated in single precision, and dropout at rate 0.252 is applied to the attention weights during training.

After the initial frame, the attention weights among proteins do not depend on the separations between proteins. The protein tokens then contain the recent steps and the species of the proteins and not their positions. The key of the pair token depends on the pairwise distances and changes only the weight assigned to the pair token relative to the other proteins.

The value of protein i is computed from its own latent slice, and the value of the pair token is zero,

$$
\mathbf { v } _ { i } = \mathrm { S i L U } \big ( \mathrm { s n } \big ( \mathbf { W } ^ { \mathrm { v } } \big ) \zeta _ { i } ^ { ( t ) } + \mathbf { b } ^ { \mathrm { v } } \big ) \in \mathbb { R } ^ { 2 5 6 } , \qquad \mathbf { v } _ { \mathrm { p a i r } } = \mathbf { 0 } ,\tag{28}
$$

where $\mathbf { W } ^ { \mathbf { v } }$ is a spectrally normalized map from six components to 256. Attention weight assigned to the pair token transmits no information from the other proteins.

The decoder representation of protein i is formed from the attention output, the residual stream and a feedforward block,

$$
\begin{array} { r } { \mathbf { m } _ { i } = \left[ \sum _ { j \in \mathcal { U } _ { i } } \omega _ { i j } \mathbf { v } _ { j } \right] _ { 4 \mathrm { ~ h e a d s } } , } \end{array}\tag{29}
$$

$$
\mathbf { o } _ { i } = \mathbf { u } _ { i } + \mathbf { v } _ { i } + \mathbf { W } ^ { \mathrm { m } } \mathbf { m } _ { i } + \mathbf { b } ^ { \mathrm { m } } ,\tag{30}
$$

$$
\begin{array} { r } { \hat { \mathbf { o } } _ { i } = \mathrm { L N } _ { \mathrm { o } } \Big ( \mathbf { o } _ { i } + \mathbf { W } ^ { \mathrm { f 2 } } \mathrm { G E L U } \big ( \mathbf { W } ^ { \mathrm { f 1 } } \mathrm { L N } _ { \mathrm { f } } ( \mathbf { o } _ { i } ) + \mathbf { b } ^ { \mathrm { f 1 } } \big ) + \mathbf { b } ^ { \mathrm { f 2 } } \Big ) , } \end{array}\tag{31}
$$

in which each head forms its weighted sum over its own 64 value components and $[ \cdot ] _ { 4 \mathrm { h e a d s } }$ concatenates the four head outputs into 256 components. The residual stream of a token is its running representation, to which each block adds its output. The matrix $\mathbf { W } ^ { \mathrm { m } }$ maps 256 components to 256, and $\mathbf { W } ^ { \mathrm { f 1 } }$ and $\mathbf { W } ^ { \mathrm { f 2 } }$ map 256 components to 512 and back. During training, dropout at rate 0.252 is also applied after the GELU and after $\mathbf { W } ^ { \mathrm { f 2 } }$ , and it is omitted from the equations and inactive during evaluation.

Two output heads act on $\hat { \mathbf { o } } _ { i } ,$ and each head consists of a spectrally normalized linear layer from 256 to 128 components, a SiLU and a final linear layer,

$$
\begin{array} { r } { \boldsymbol { \Delta } \bar { \mathbf { x } } _ { i } = \mathbf { W } _ { s ( i ) } ^ { \mathrm { d r 2 } } \operatorname { S i L U } \bigl ( \mathrm { s n } \bigl ( \mathbf { W } _ { s ( i ) } ^ { \mathrm { d r 1 } } \bigr ) \hat { \mathbf { o } } _ { i } + \mathbf { b } _ { s ( i ) } ^ { \mathrm { d r 1 } } \bigr ) + \mathbf { b } _ { s ( i ) } ^ { \mathrm { d r 2 } } \in \mathbb { R } ^ { 3 } , } \\ { \boldsymbol { \varpi } _ { i } = \mathbf { W } _ { s ( i ) } ^ { \mathrm { c o v 2 } } \operatorname { S i L U } \bigl ( \mathrm { s n } \bigl ( \mathbf { W } _ { s ( i ) } ^ { \mathrm { c o v 1 } } \bigr ) \hat { \mathbf { o } } _ { i } + \mathbf { b } _ { s ( i ) } ^ { \mathrm { c o v 1 } } \bigr ) + \mathbf { b } _ { s ( i ) } ^ { \mathrm { c o v 2 } } \in \mathbb { R } ^ { 6 } , } \end{array}\tag{32}
$$

where $\Delta \bar { \bf x } _ { i }$ is the mean displacement of protein i and $\varpi _ { i } = ( \varpi _ { i , 1 } , \ldots , \varpi _ { i , 6 } )$ contains its six raw covariance parameters. The bias $ { \mathbf { b } } _ { s } ^ { \mathrm { d r } 2 }$ is initialized to zero. The PIEZO system has one set of output heads, and the PIEZO-TREK system has one set per species. The attention block, the feedforward block and the layer normalizations are shared by all species.

The six covariance parameters of protein i are mapped to a lower-triangular Cholesky factor of the covariance of its displacement,

$$
\Lambda _ { i } = \left( \begin{array} { c c c } { { \mathrm { s o f t p l u s } ( \varpi _ { i , 1 } ) + 1 0 ^ { - 3 } } } & { { 0 } } & { { 0 } } \\ { { \varpi _ { i , 4 } } } & { { \mathrm { s o f t p l u s } ( \varpi _ { i , 2 } ) + 1 0 ^ { - 3 } } } & { { 0 } } \\ { { \varpi _ { i , 5 } } } & { { \varpi _ { i , 6 } } } & { { \mathrm { s o f t p l u s } ( \varpi _ { i , 3 } ) + 1 0 ^ { - 3 } } } \end{array} \right) ,\tag{33}
$$

in which the softplus floor of $1 0 ^ { - 3 }$ keeps every diagonal entry strictly positive and the covariance $\pmb { \Lambda } _ { i } \pmb { \Lambda } _ { i } ^ { \top }$ positive definite. The three of-diagonal entries are unconstrained and represent the correlations between the $\mathrm { x , y }$ and z components of the displacement of a single protein. The covariance is block diagonal across proteins.

Displacements are sampled by reparameterization [36, 37], which writes each sample as a deterministic function of the network outputs and of independent standard normal noise,

$$
\Delta { \mathbf { x } _ { i } ^ { ( t ) } } = \Delta \bar { \mathbf { x } } _ { i } + \mathbf { A } _ { i } \varepsilon _ { i } ^ { ( t ) } , \qquad \varepsilon _ { i } ^ { ( t ) } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } _ { 3 } ) , \qquad i = 1 , \dots , N ,\tag{34}
$$

where $\varepsilon _ { i } ^ { ( t ) }$ is a vector of three independent standard normal numbers drawn at every step. Gradients flow through both the mean and the covariance factor. The N displacements advance the wrapped state,

$$
\mathbf { x } _ { i } ^ { ( t + 1 ) } = \big ( \mathbf { x } _ { i } ^ { ( t ) } + \Delta \mathbf { x } _ { i } ^ { ( t ) } \big ) \bmod L ,\tag{35}
$$

applied to all three components. The next geometry vector is computed from $\mathbf { x } ^ { ( t + 1 ) }$ and $\mathbf { x } ^ { ( t ) }$

A training window of 3,000 frames uses frame 0 to build the first geometry vector, and Equations 34 and 35 are then iterated 2,999 times, each iteration starting from the position produced by the previous iteration.

Table 2 Widths of the internal vectors of the model and parameter counts for each system.
<table><tr><td></td><td>PIEZO</td><td>PIEZO-TREK</td></tr><tr><td>Geometry vector  $\mathbf { e } ^ { ( t ) }$ </td><td>462</td><td>1207</td></tr><tr><td>Bank input  $\tilde { \mathbf { e } } _ { i } ^ { ( t ) }$ </td><td>84</td><td>199</td></tr><tr><td>LSTM banks (one per species)</td><td>1</td><td>2</td></tr><tr><td>Bank readout  $\dot { \boldsymbol { \psi } } ^ { ( t ) }$ </td><td>3136</td><td>7616</td></tr><tr><td>Transition head hidden width</td><td>1568</td><td>3808</td></tr><tr><td>Latent state  $\zeta ^ { ( t ) }$ </td><td>42</td><td>102</td></tr><tr><td>Decoder tokens (proteins and pair)</td><td>7+ 1</td><td>17 + 1</td></tr><tr><td>Decoder head sets (one per species)</td><td>1</td><td>2</td></tr><tr><td>Parameters, transformer encoder</td><td>33,075</td><td>33,705</td></tr><tr><td>Parameters, LSTM banks and transition head</td><td>5,259,954</td><td>30,359,842</td></tr><tr><td>Parameters, decoder</td><td>553,265</td><td>649,888</td></tr><tr><td>Parameters, latent dynamics</td><td>2</td><td>4</td></tr><tr><td>Parameters, total</td><td>5,846,296</td><td>31,043,439</td></tr></table>

## 2.10 Training loss

The loss compares statistics of the rollout of each training window with the same statistics of frames 1 to 2,999 of the window, mostly as empirical distributions. For two samples y<sup>mod</sup> and $\mathbf { y } ^ { \mathrm { M D } }$ of equal size $K$ , the first from the rollout and the second from the MD, the distribution-matching function sorts each sample and computes the mean absolute diference of entries of equal rank,

$$
\mathcal { Q } ( \mathbf { y } ^ { \mathrm { m o d } } , \mathbf { y } ^ { \mathrm { M D } } ) = \frac { 1 } { K } \sum _ { n = 1 } ^ { K } \bigl | y _ { ( n ) } ^ { \mathrm { m o d } } - y _ { ( n ) } ^ { \mathrm { M D } } \bigr | ,\tag{36}
$$

where $y _ { ( n ) }$ is the n-th smallest entry of a sample. For samples of equal size, this quantity is the 1-Wasserstein distance between the two empirical distributions [38, arXiv version, Remark 2.28]. The sorting removes the correspondence between individual draws of the two samples. The rollout and MD samples have the same size, and each rollout has the shape of its target.

Single-protein samples are matched separately within each species. For a quantity computed at every window, time origin and protein or protein pair, the sample over the rollouts of a batch is denoted $\{ \cdot \} ^ { \mathrm { m o d } }$ , and the same sample over the corresponding MD target frames is denoted $\{ \cdot \} ^ { \mathrm { M D } }$ . The species-wise match is

$$
\mathcal { Q } _ { \mathrm { s p } } \big ( \{ \cdot \} ^ { \mathrm { m o d } } , \{ \cdot \} ^ { \mathrm { M D } } \big ) = \frac { 1 } { | \mathcal { S } | } \sum _ { s \in \mathcal { S } } \mathcal { Q } \big ( \{ \cdot \} _ { s } ^ { \mathrm { m o d } } , \{ \cdot \} _ { s } ^ { \mathrm { M D } } \big ) ,\tag{37}
$$

where the subscript s selects the entries of the proteins of species s and $| S |$ is the number of species. Within a species, all windows, time origins and proteins are pooled into one sorted sample. The match is invariant to relabeling of the proteins within a species and is sensitive to diferences in mobility between species. The species are weighted equally, independent of their numbers of proteins. For the PIEZO system, which has one species, Equation 37 reduces to the pooled match. Terms based on protein pairs are not split by species and apply Equation 36 to all pairs.

The single-protein lateral observables are computed on the unwrapped lateral positions of Equation 3, accumulated from the first predicted frame for the rollout and from frame 1 for the MD targets. The lateral displacement over a lag of τ frames is $\Delta _ { \tau } \mathbf { r } _ { i } ^ { ( t ) } = \mathbf { r } _ { i } ^ { ( t + \tau ) } - \mathbf { r } _ { i } ^ { ( t ) }$ . A statistic at lag τ uses every time origin t at which it is defined within the 2,999 frames, lags longer than the window are omitted, and an average over a lag set runs over the remaining lags.

Two loss terms constrain the lateral motion of a single protein, the displacement term and the turning-angle term. These two terms and the cohesion term use the lag set T<sub>lag</sub> = {1, 5, 10, 25, 50, 100, 200, 300, 500, 1000, 2000, 3000} in both systems.

The displacement term matches the distribution of displacement magnitudes at each lag,

$$
\mathcal { L } _ { \mathrm { d i s p } } = \left. \mathcal { Q } _ { \mathrm { s p } } \left( \left\{ \sqrt { \| \Delta _ { \tau } \mathbf { r } _ { i } ^ { ( t ) } \| ^ { 2 } + 1 0 ^ { - 6 } } \right\} ^ { \mathrm { m o d } } , \left\{ \sqrt { \| \Delta _ { \tau } \mathbf { r } _ { i } ^ { ( t ) } \| ^ { 2 } + 1 0 ^ { - 6 } } \right\} ^ { \mathrm { M D } } \right) \right. _ { \tau \in \mathcal { T } _ { \mathrm { l a g } } } ,\tag{38}
$$

where the constant $1 0 ^ { - 6 }$ keeps the square root diferentiable at zero displacement. This term is the distributional counterpart of the time-averaged mean squared displacement. Lag 3000 does not fit in the window and is omitted.

The turning-angle term matches the distribution of the unsigned in-plane angle between successive displacements over the same lag [39],

$$
\begin{array} { r l } & { \boldsymbol { \theta } _ { i } ^ { ( t ) } ( \tau ) = \mathrm { a t a n 2 } \Big ( \big | \Delta _ { \tau } { \bf r } _ { i } ^ { ( t ) } \times \Delta _ { \tau } { \bf r } _ { i } ^ { ( t + \tau ) } \big | , ~ \Delta _ { \tau } { \bf r } _ { i } ^ { ( t ) } \cdot \Delta _ { \tau } { \bf r } _ { i } ^ { ( t + \tau ) } \Big ) \in [ 0 , \pi ] , } \\ & { \quad \quad \quad \mathcal { L } _ { \mathrm { a n g } } = \Big \langle \mathcal { Q } _ { \mathrm { s p } } \big ( \{ \boldsymbol { \theta } _ { i } ^ { ( t ) } ( \tau ) \} ^ { \mathrm { m o d } } , \{ \boldsymbol { \theta } _ { i } ^ { ( t ) } ( \tau ) \} ^ { \mathrm { M D } } \big ) \Big \rangle _ { \tau \in \mathcal { T } _ { \mathrm { l a g } } } , } \end{array}\tag{39}
$$

where $\mathbf { y } _ { 1 } \times \mathbf { y } _ { 2 } = y _ { 1 , 1 } y _ { 2 , 2 } - y _ { 1 , 2 } y _ { 2 , 1 }$ is the scalar cross product of two lateral vectors. A lag is used when the window yields at least two angles, and lags 2000 and 3000 are omitted.

Two loss terms constrain the collective structure of the proteins, the pairwise term and the cohesion term. Both terms use the lateral minimum-image pair distances of the wrapped coordinates,

$$
\ell _ { i j } ^ { ( t ) } = \sqrt { \left\| \operatorname* { m i } \bigl ( \mathbf { x } _ { i , \mathrm { x y } } ^ { ( t ) } - \mathbf { x } _ { j , \mathrm { x y } } ^ { ( t ) } \bigr ) \right\| ^ { 2 } + 1 0 ^ { - 8 } } \nearrow I , \qquad 1 \leq i < j \leq N ,\tag{40}
$$

which difer from the distances of Equation 13 only by the omission of the membranenormal component. The pairwise term matches the distribution of these distances, and the cohesion term matches the distribution of their increments $\Delta _ { \tau } \ell _ { i j } ^ { ( t ) } = \ell _ { i j } ^ { ( t + \tau ) } - \ell _ { i j } ^ { ( t ) }$ 2

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { p a i r } } = \mathcal { Q } \big ( \{ \ell _ { i j } ^ { ( t ) } \} ^ { \mathrm { m o d } } , \{ \ell _ { i j } ^ { ( t ) } \} ^ { \mathrm { M D } } \big ) , \qquad \mathcal { L } _ { \mathrm { c o h } } = \Big \langle \mathcal { Q } \big ( \{ \Delta _ { \tau } \ell _ { i j } ^ { ( t ) } \} ^ { \mathrm { m o d } } , \{ \Delta _ { \tau } \ell _ { i j } ^ { ( t ) } \} ^ { \mathrm { M D } } \big ) \Big \rangle _ { \tau \in \mathcal { T } _ { \mathrm { l a g } } } , } \end{array}\tag{41}
$$

with samples taken over windows, time origins and all pairs. Lag 3000 is omitted from the cohesion term.

The remaining terms use three lateral statistics, which also enter the evaluation of Section 2.12. For a set $\mathcal { P }$ of proteins, these statistics are the mean squared displacement, the non-Gaussian parameter [40] and the normalized velocity autocorrelation function (VACF),

$$
\begin{array} { r } { \mathrm { M S D } _ { \mathcal { P } } ( \tau ) = \big \langle \lVert \Delta _ { \tau } \mathbf { r } _ { i } ^ { ( t ) } \rVert ^ { 2 } \big \rangle , } \end{array}\tag{42}
$$

$$
\alpha _ { 2 , \mathcal { P } } ( \tau ) = \frac { \langle \parallel \Delta _ { \tau } { \bf r } _ { i } ^ { ( t ) } \parallel ^ { 4 } \rangle } { 2 \langle \parallel \Delta _ { \tau } { \bf r } _ { i } ^ { ( t ) } \parallel ^ { 2 } \rangle ^ { 2 } } - 1 ,\tag{43}
$$

$$
\mathrm { V A C F } _ { \mathcal P } ( \tau ) = \frac { \big \langle \Delta _ { 1 } \mathbf r _ { i } ^ { ( t ) } \cdot \Delta _ { 1 } \mathbf r _ { i } ^ { ( t + \tau ) } \big \rangle } { \big \langle \lVert \Delta _ { 1 } \mathbf r _ { i } ^ { ( t ) } \rVert ^ { 2 } \big \rangle } ,\tag{44}
$$

where every average runs over the trajectories considered, every time origin t at which the quantity is defined and every protein $i \in \mathcal { P }$ , and the constants of at most $1 0 ^ { - 8 }$ that guard the denominators are omitted. The mean squared displacement is time-averaged [41].

The non-Gaussian parameter, denoted $\alpha _ { 2 }$ by convention and distinct from the exponent $\alpha$ below, uses the factor 2 for two dimensions. It vanishes for a Gaussian distribution of displacements, is positive for tails heavier than Gaussian and is negative for suppressed tails. The VACF uses single-frame displacements as velocities and is normalized by the mean squared single-frame displacement.

Three loss terms constrain the velocity autocorrelation at diferent levels of detail. The first term matches the full distribution of the cosine similarity between two displacements over a window of $\tau ^ { \prime }$ frames separated by $\tau$ frames,

$$
\begin{array} { r l r } & { } & { \chi _ { i } ^ { ( t ) } ( \tau ^ { \prime } , \tau ) = \frac { \Delta _ { \tau ^ { \prime } } { \bf r } _ { i } ^ { ( t ) } \cdot \Delta _ { \tau ^ { \prime } } { \bf r } _ { i } ^ { ( t + \tau ) } } { \sqrt { \| \Delta _ { \tau ^ { \prime } } { \bf r } _ { i } ^ { ( t ) } \| ^ { 2 } + 1 0 ^ { - 8 } } \sqrt { \| \Delta _ { \tau ^ { \prime } } { \bf r } _ { i } ^ { ( t + \tau ) } \| ^ { 2 } + 1 0 ^ { - 8 } } } , } \\ & { } & { \mathcal { L } _ { \mathrm { v q } } = \left. \mathcal { Q } _ { \mathrm { s p } } \big ( \{ \chi _ { i } ^ { ( t ) } ( \tau ^ { \prime } , \tau ) \} ^ { \mathrm { m o d } } , \{ \chi _ { i } ^ { ( t ) } ( \tau ^ { \prime } , \tau ) \} ^ { \mathrm { M D } } \big ) \right. _ { \tau ^ { \prime } \in \mathcal { T } _ { \mathrm { v e l } } , \tau \in \mathcal { T } _ { \mathrm { c o r r } } } , } \end{array}\tag{45}
$$

over the velocity windows $\mathcal { T } _ { \mathrm { v e l } } = \{ 1 , 5 , 1 0 , 2 5 , 5 0 , 1 0 0 , 2 0 0 \}$ and the correlation lags $\mathcal { T } _ { \mathrm { c o r r } } ~ = ~ \{ 1 , 5 , 1 0 , 2 5 , 5 0 , 1 0 0 , 2 0 0 , 3 0 0 , 5 0 0 , 1 0 0 0 , 2 0 0 0 , 3 0 0 0 \}$ . These two sets are the same for both systems and are independent of $\mathcal { T } _ { \mathrm { l a g } }$ . Lag 3000 does not fit in the window, and $7 7$ of the 84 pairs of $\tau ^ { \prime }$ and τ are used.

The second VACF term matches the mean normalized autocorrelation curve as a scalar target at each lag,

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { v c } } = \left. \left( \mathrm { V A C F } _ { \mathrm { a l l } } ^ { \mathrm { m o d } } ( \tau ) - \mathrm { V A C F } _ { \mathrm { a l l } } ^ { \mathrm { M D } } ( \tau ) \right) ^ { 2 } \right. _ { \tau \in \mathcal { T } _ { \mathrm { v c } } } , \qquad \mathcal { T } _ { \mathrm { v c } } = \{ 1 , 5 , 1 0 , 2 5 , 5 0 , 1 0 0 , 2 0 0 \} , } \end{array}\tag{46}
$$

where $\mathrm { V A C F _ { a l l } }$ is Equation 44 with $\mathcal { P }$ the set of all proteins, averaged over the windows of the batch and computed on the rollout or on the MD as indicated by the superscript, and $\mathcal { T } _ { \mathrm { v c } }$ is its set of correlation lags. This term does not distinguish species.

The third VACF term compares the raw products $\Pi _ { i } ^ { ( t ) } ( \tau ) = \bar { \Delta _ { 1 } } \mathbf { r } _ { i } ^ { ( t ) } \cdot \Delta _ { 1 } \mathbf { r } _ { i } ^ { ( t + \tau ) }$ point by point, without sorting,

$$
\begin{array} { r l r } { \mathcal { L } _ { \mathrm { v s } } = \Big \langle \big \langle \big | \Pi _ { i } ^ { ( t ) , \mathrm { m o d } } ( \tau ) - \Pi _ { i } ^ { ( t ) , \mathrm { M D } } ( \tau ) \big | \big \rangle _ { \mathrm { w i n d o w s } , t , i } \Big \rangle _ { \tau \in \mathcal { T } _ { \mathrm { v s } } } , } & { } & { \mathcal { T } _ { \mathrm { v s } } = \{ 1 , 5 , 1 0 , 2 5 , 5 0 , 1 0 0 \} , } \end{array}\tag{47}
$$

where $\Pi _ { i } ^ { ( t ) , \mathrm { m o d } } \mathrm { a n d } \Pi _ { i } ^ { ( t ) , \mathrm { M D } }$ are the products of the rollout and of the MD at the same window, time and protein. This term is the only term that compares the rollout and the MD frame by frame.

A rollout and its target share only the initial frame, and the unsorted term is an absolute diference between two independent realizations rather than a distribution match. For independent realizations, the expectation of the term is smallest when the product of the rollout equals the median of the MD products, and the term favors products with less variability rather than a matching distribution.

Two loss terms target scalar summaries of the difusive regime. The first is the anomalous exponent $\alpha ,$ the slope of the mean squared displacement on logarithmic axes $[ 4 1 , 4 2 ]$ , which distinguishes normal difusion $( \alpha = 1 )$ , subdifusion $( \alpha < 1 )$ and superdifusion $( \alpha > 1 )$ . The exponent is obtained by ordinary least squares in closed form, and the gradient of the loss passes through the fit,

$$
\alpha _ { \mathcal { P } } = \frac { \lvert \mathcal { T } _ { \alpha } \rvert \sum _ { \tau } \ln \tau \ln \mathrm { M S D } _ { \mathcal { P } } ( \tau ) - \sum _ { \tau } \ln \tau \sum _ { \tau } \ln \mathrm { M S D } _ { \mathcal { P } } ( \tau ) } { \lvert \mathcal { T } _ { \alpha } \rvert \sum _ { \tau } ( \ln \tau ) ^ { 2 } - \left( \sum _ { \tau } \ln \tau \right) ^ { 2 } } ,\tag{48}
$$

with every sum over $\tau \in \mathcal { T } _ { \alpha }$ and $| \mathcal { T } _ { \alpha } | = 8$ lags. In the loss, the mean squared displacement is clipped below at $1 0 ^ { - 8 }$ before the logarithm and is averaged over a single window and the proteins of one species, and one exponent is fitted per window and species. The exponent term is

$$
\mathcal { L } _ { \alpha } = \big \langle \big ( \alpha _ { \mathcal { P } _ { s } } ^ { \mathrm { m o d } } - \alpha _ { \mathcal { P } _ { s } } ^ { \mathrm { M D } } \big ) ^ { 2 } \big \rangle _ { \mathrm { w i n d o w s } , s \in \mathcal { S } } ,\tag{49}
$$

where $\alpha _ { \mathcal { P } _ { s } } ^ { \mathrm { m o d } }$ and $\alpha _ { \mathcal { P } _ { s } } ^ { \mathrm { M D } }$ are fitted on the rollout and on the MD target of the same window.

The second scalar term matches the non-Gaussian parameter of Equation 43. The non-Gaussian parameter is pooled over the windows of the batch, the time origins and the proteins of a species and is matched by squared error,

$$
\mathcal { L } _ { \mathrm { n g p } } = \Big \langle \frac { 1 } { | \mathcal { S } | } \sum _ { s \in \mathcal { S } } \big ( \alpha _ { 2 , \mathcal { P } _ { s } } ^ { \mathrm { m o d } } ( \tau ) - \alpha _ { 2 , \mathcal { P } _ { s } } ^ { \mathrm { M D } } ( \tau ) \big ) ^ { 2 } \Big \rangle _ { \tau \in \mathcal { T } _ { \mathrm { n g p } } } , \qquad \mathcal { T } _ { \mathrm { n g p } } = \{ 1 , 5 , 1 0 , 2 5 , 5 0 , 1 0 0 , 2 5 0 \} .\tag{50}
$$

The non-Gaussian parameter and the mean squared displacement in the exponent fit have the same definitions in the loss and in the evaluation of Section 2.12. The exponent is used only in the loss. The loss uses lags that fit within the 2,999 predicted frames of a window, and the evaluation uses lags within rollouts of 40,000 steps.

The total loss is a fixed weighted sum of the nine terms,

$$
\begin{array} { r l } & { \mathcal { L } = w _ { \mathrm { d i s p } } \mathcal { L } _ { \mathrm { d i s p } } + w _ { \mathrm { a n g } } \mathcal { L } _ { \mathrm { a n g } } + w _ { \mathrm { p a i r } } \mathcal { L } _ { \mathrm { p a i r } } + w _ { \mathrm { c o h } } \mathcal { L } _ { \mathrm { c o h } } + w _ { \mathrm { v q } } \mathcal { L } _ { \mathrm { v q } } } \\ & { \qquad + w _ { \mathrm { v c } } \mathcal { L } _ { \mathrm { v c } } + w _ { \mathrm { v s } } \mathcal { L } _ { \mathrm { v s } } + w _ { \alpha } \mathcal { L } _ { \alpha } + w _ { \mathrm { n g p } } \mathcal { L } _ { \mathrm { n g p } } , } \end{array}\tag{51}
$$

with the loss weights w of Table 3. All terms are lateral, and no term constrains the membrane-normal coordinate.

The weights of the PIEZO system were selected by Gaussian-process Bayesian optimization [43] over 30 trials in Optuna [44]. The first 12 trials followed a Sobol design [45], a low-discrepancy sequence that distributes points evenly over the search space. Each trial trained one model on the PIEZO system and scored it with the replica-referenced objective of the training monitor (Section 2.11). The trial with the best score was retained. The search range of each weight was set separately for each term.

The weight search used a configuration of the model and of the training that differs from the final configuration. In the search configuration, the decoder used the latent state as the attention query, the transformer encoder received the sine-cosine embedding of absolute position at every frame, and the latent parameters were initialized to $a _ { s } = 0 . 5$ and $\kappa _ { s } = 0 . 3$ . The search training used a plateau learning-rate schedule, which halves the learning rate when the monitored objective stops improving, and early stopping, which ends training after the objective has not improved for a set number of epochs. It also used a learning rate searched between $1 0 ^ { - 7 }$ and $1 0 ^ { - 4 }$ , two rollouts per window and windows of 3,000 frames cut at any position in augmented views of the first 9,999 frames of replica 1, taken before the removal of duplicated frames. The search loss also included two membrane-normal terms and a latent-autocorrelation term, which the final models do not use. The nine weights of Table 3 are carried over unchanged from the selected trial.

Table 3 Loss weights of Equation 51. The weights are dimensionless multipliers on observables in normalized units and are identical for the five training seeds of a system.
<table><tr><td>Term</td><td>Weight</td><td>PIEZO</td><td>PIEZO-TREK</td></tr><tr><td>Displacement</td><td>wdisp</td><td>1.836</td><td>7.712</td></tr><tr><td>Turning angle</td><td>Wang</td><td>181.67</td><td>858.79</td></tr><tr><td>Pairwise distance</td><td> $w _ { \mathrm { p a i r } }$ </td><td>18.125</td><td>18.125</td></tr><tr><td>Cohesion</td><td> $w _ { \mathrm { c o h } }$ </td><td>1871.69</td><td>583.36</td></tr><tr><td>VACF distribution</td><td> $w _ { \mathbf { v } \mathbf { q } }$ </td><td>12.705</td><td>2.229</td></tr><tr><td>VACF curve</td><td> $w _ { \mathrm { v c } }$ </td><td>166.43</td><td>166.43</td></tr><tr><td>VACF, unsorted</td><td> $w _ { \mathrm { v s } }$ </td><td>4.020</td><td>4.020</td></tr><tr><td>Anomalous exponent</td><td> $w _ { \alpha }$ </td><td>2214.83</td><td>1571.96</td></tr><tr><td>Non-Gaussian parameter</td><td> $w _ { \mathrm { n g p } }$ </td><td>94.555</td><td>179.43</td></tr></table>

The weights of the PIEZO-TREK system are a point of a 24-point Latin-hypercube design [46] around the PIEZO weights. The design multiplies six weights, those of the displacement, turning-angle, cohesion, VACF-distribution, exponent and non-Gaussian terms, by factors between 0.1 and 10 distributed uniformly in their logarithm, and keeps the pairwise, VACF-curve and unsorted VACF weights. The factors of the selected point are 4.20, 4.73, 0.31, 0.18, 0.71 and 1.90, in the order of the six terms listed above.

Five points of the design, points 1, 6, 8, 11 and 19, were trained for the PIEZO-TREK system with training seed 500. Every epoch was scored with the rollouts of Section 2.12 against per-lag replica tolerances taken from the PIEZO system. Point 11, which has the lowest best-epoch score under these tolerances, is used. Seed 500 is also one of the five training seeds of the final models.

## 2.11 Optimization and schedule

The parameters are optimized with Sophia [47], a stochastic second-order optimizer that divides a moving average of the gradient by a moving average of a diagonal estimate of the Hessian and clips the step of each coordinate. The two published variants of Sophia estimate this diagonal with the Hutchinson estimator [48] or with the Gauss-Newton-Bartlett estimator, which samples labels from the predictive distribution of the model.

We use the clipped update of Sophia with a diferent curvature estimate, a moving average of the squared minibatch gradient that is refreshed at every step, without sampled labels and without a refresh interval. For each scalar parameter φ, with $\partial \mathcal { L } / \partial \varphi$ its minibatch gradient after clipping, each step updates two moving averages $\mathcal { R } _ { \varphi }$ and $\mathcal { G } _ { \varphi }$ , both initialized at zero, and then the parameter,

$$
\mathcal { R } _ { \varphi } \gets 0 . 9 9 \mathcal { R } _ { \varphi } + 0 . 0 1 \left( \partial \mathcal { L } / \partial \varphi \right) ^ { 2 } ,\tag{52}
$$

$$
\mathcal { G } _ { \varphi } \gets 0 . 9 6 5 \mathcal { G } _ { \varphi } + 0 . 0 3 5 \partial \mathcal { L } / \partial \varphi ,\tag{53}
$$

$$
\varphi \gets \varphi - \eta \operatorname { s i g n } ( \mathcal { G } _ { \varphi } ) \operatorname* { m i n } \left( \frac { \left| \mathcal { G } _ { \varphi } \right| } { \operatorname* { m a x } \bigl ( 1 . 0 1 7 \mathcal { R } _ { \varphi } , 1 0 ^ { - 1 5 } \bigr ) } , 1 \right) ,\tag{54}
$$

where $\eta$ is the learning rate of the parameter group and 1.017 is the product of the curvature parameter 0.015890 and the batch size 64. The step of every coordinate is at most $\eta ,$ and this bound is reached when the momentum is large compared with the curvature estimate.

The parameters are divided into three groups. The first group receives weight decay and contains only the weights of the plain linear and attention layers, with 0.58 million parameters in the PIEZO model and 0.93 million parameters in the PIEZO-TREK model. Parameters of this group are multiplied by $1 - 0 . 0 1 \eta$ before the update of Equation 54, which is a decoupled weight decay [49] of 0.01. The second group receives no weight decay and contains every other parameter outside the third group, including the recurrent weights, the weights of the spectrally normalized layers, and all biases and normalization parameters. The third group contains the latent dynamics parameters $\tilde { a } _ { s }$ and ${ \tilde { \kappa } } _ { s }$ of Equation 24, receives no weight decay and has a learning rate multiplied by 3162.27.

The learning rate is constant during training, with $\eta \ : = \ : 2 . 2 1 1 7 \times 1 0 ^ { - 5 }$ for the network and 0.06994 per step for the latent dynamics group. The base rate was selected on the PIEZO system from the grid $6 . 9 9 4 \times 1 0 ^ { - 6 }$ , 2.2117 × 10<sup>−5</sup>, $6 . 9 9 4 \times 1 0 ^ { - 5 }$ and $2 . 2 1 1 7 \times 1 0 ^ { - 4 }$ , with the learning rate of the latent dynamics group held at 0.03497 per step. The final models use twice this latent learning rate.

Every model is trained for 25 epochs of 15 steps, 375 optimizer steps in total, on full windows of 3,000 frames from the first step. Five models are trained per system with the training seeds 500 to 504. The seed sets the initial weights, the order of the training windows and the noise of the training rollouts, up to non-deterministic GPU kernels. Table 4 lists these settings.

Gradients are computed through the whole 2,999-step rollout of a window by backpropagation through time [50], which takes the gradient of a recurrent computation through all of its steps. Training uses bfloat16 mixed precision and allows reducedprecision arithmetic in single-precision matrix products. Gradients are clipped to a global norm of 20 [51].

Non-finite values are handled by skipping and stopping rules. A batch with a nonfinite total loss is skipped. The optimizer skips a step with non-finite gradients and stops the run after three such steps in a row or ten in total. Training also stops when skipped batches and epochs with a non-finite monitor value, counted together, reach three.

Models were trained with PyTorch 2.10.0 [52] and PyTorch Lightning 2.6.1 [53] on NVIDIA L40, L40S and RTX A6000 GPUs.

Table 4 Training settings, shared by both systems.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Optimizer</td><td>Sophia with squared-gradient curvature, Equations 52 to 54</td></tr><tr><td>Learning rate η</td><td>2.2117 × 10−5, constant</td></tr><tr><td>Learning rate, latent dynamics</td><td>0.06994, constant</td></tr><tr><td>Weight decay</td><td>0.01, plain linear and attention weights only</td></tr><tr><td>Batch</td><td>64 windows</td></tr><tr><td>Window</td><td>3,000 frames, 1 initial condition and 2,999 targets</td></tr><tr><td>Windows per epoch</td><td>960, giving 15 optimizer steps</td></tr><tr><td>Epochs</td><td>25, giving 375 optimizer steps</td></tr><tr><td>Gradient clipping</td><td>global norm 20</td></tr><tr><td>Dropout</td><td>0.252, decoder attention and feedforward only</td></tr><tr><td>Precision</td><td>bfloat16 mixed precision, latent update in single precision and rounded to bfloat16 between steps</td></tr><tr><td>Training seeds</td><td>500 to 504</td></tr></table>

At the end of every epoch, a training monitor rolls out 128 initial conditions from the reference replicas for 20,000 steps and computes a summary objective. The objective is the mean, over six lateral statistics and, in the PIEZO-TREK system, a seventh statistic that compares the mobilities of the two species, of the rectified excess of the distance between the model and the MD relative to the distance between two MD replicas. For the PIEZO-TREK system, the replica distance is taken from the PIEZO system. The monitor changes neither the gradients nor the learning rate, does not enter the model selection of Section 2.14, and enters training only through the stopping rule above. The weight search of Section 2.10 was scored with an objective of the same form.

## 2.12 Long-horizon evaluation

A single long-horizon evaluation is used for model selection and for reporting. The evaluation measures the agreement between the lateral difusion statistics of long roll outs and those of the reference trajectories. It is applied, separately from training, to the weights at the end of every epoch of every training run. Each rollout starts from one MD configuration and is generated as described in Section 2.1.

For each system, 512 initial conditions are drawn without replacement, with a generator of fixed seed, from the pooled frames of the replicas listed in Table 5. Each drawn frame supplies the positions of all proteins for one initial condition. The same 512 initial conditions are used for every checkpoint of a system, and most of them lie outside the training frames.

Each initial condition is rolled out autoregressively for 40,000 steps under Equations 34 and 35, with dropout disabled and in single precision. For the PIEZO system, 40,000 steps correspond to 20 µs. The rollout length is 3.2 times the frame count of the longest PIEZO replica and 3.6 times that of the longest PIEZO-TREK replica. Both the latent integrator and the decoder are stochastic, and each checkpoint is rolled out under three noise seeds with the initial conditions held fixed, which gives $5 1 2 \times 4 0 , 0 0 0 \times 3 = 6 . 1 4 \times 1 0 ^ { 7 }$ predicted frames per checkpoint.

Table 5 Evaluation settings. Initial conditions are drawn from the listed frames, and the reference pools both replicas of each system. The largest scored lag $\tau _ { \operatorname* { m a x } , s }$ of species s is defined in the text, and the stand-in rollouts and the draws are defined in Section 2.13.
<table><tr><td></td><td>PIEZO</td><td>PIEZO-TREK</td></tr><tr><td>Initial conditions from</td><td>replicas 1 and 2</td><td>replica 1</td></tr><tr><td>Frames available</td><td>19,681</td><td>11,001</td></tr><tr><td>Largest lag τmax,s</td><td>5000</td><td>5000 for both species</td></tr><tr><td>Cells</td><td>204</td><td> $2 0 4 + 2 0 4 = 4 0 8$ </td></tr><tr><td>Stand-in rollouts</td><td>2 × 40</td><td> $2 \times 4 0$ </td></tr><tr><td>Draws</td><td>600</td><td>960</td></tr></table>

The statistics are computed separately for each species s on the lateral coordinates at the lags of the ladder $\mathcal { T } _ { s } = \{ 1 , 1 0 0 , 2 0 0 , \dots , \tau _ { \operatorname* { m a x } , s } \}$ , which contains lag 1 and every multiple of 100 up to $\tau _ { \operatorname* { m a x } , s }$ . Lags 2 to 99 are not scored. At each lag, the evaluation computes the mean squared displacement, the non-Gaussian parameter and the VACF of Equations 42 to 44 with $\mathcal { P } ~ = ~ \mathcal { P } _ { s }$ , and the empirical sample of displacement magnitudes

$$
\begin{array} { r } { \mathcal { V } _ { s } ( \tau ) = \big \{ \| \Delta _ { \tau } \mathbf { r } _ { i } ^ { ( t ) } \| : \mathrm { a l l \ t r a j e c t o r i e s } , t , \ i \in \mathcal { P } _ { s } \ \big \} , } \end{array}\tag{55}
$$

which is the empirical form of the self part of the van Hove correlation function [54]. For the rollouts, all 512 rollouts, all time origins and all proteins of the species are pooled after unwrapping with Equation 3.

A sample with more than 120,000 values is reduced to 120,000 values by a uniform draw with replacement, with a fixed seed for each lag.

The reference statistics are computed from the MD replicas in the same way, replica by replica. The reference value of the mean squared displacement, the non-Gaussian parameter and the VACF is the mean of the two replica values, and the reference van Hove sample is the concatenation of the two replica samples. Replica 1 of each system contains the 3,000 training frames, and the reference includes these frames.

Each statistic $M \in \{ \mathrm { M S D } , \mathrm { N G P } , \mathrm { v H } , \mathrm { V A C F } \}$ of species s at lag $\tau$ forms one cell, and the distance between the model and the reference in that cell is

$$
\delta _ { s , M } ( \tau ) = \left\{ \begin{array} { l l } { \displaystyle \big | \mathrm { l n } \frac { \operatorname* { m a x } \big ( \mathrm { M S D } _ { \mathcal { P } _ { s } } ^ { \mathrm { m o d } } ( \tau ) , 1 0 ^ { - 9 } \big ) } { \operatorname* { m a x } \big ( \mathrm { M S D } _ { \mathcal { P } _ { s } } ^ { \mathrm { r e f } } ( \tau ) , 1 0 ^ { - 9 } \big ) } \big | , } & { M = \mathrm { M S D } , } \\ { \displaystyle \big | \alpha _ { 2 , \mathcal { P } _ { s } } ^ { \mathrm { m o d } } ( \tau ) - \alpha _ { 2 , \mathcal { P } _ { s } } ^ { \mathrm { r e f } } ( \tau ) \big | , } & { M = \mathrm { N G P } , } \\ { \displaystyle \big | \mathrm { V A C F } _ { \mathcal { P } _ { s } } ^ { \mathrm { m o d } } ( \tau ) - \mathrm { V A C F } _ { \mathcal { P } _ { s } } ^ { \mathrm { r e f } } ( \tau ) \big | , } & { M = \mathrm { V A C F } , } \\ { W _ { 1 } \big ( \mathcal { V } _ { s } ^ { \mathrm { m o d } } ( \tau ) , \mathcal { V } _ { s } ^ { \mathrm { r e f } } ( \tau ) \big ) \big / \big ( \mathrm { r m s } \big ( \mathcal { V } _ { s } ^ { \mathrm { r e f } } ( \tau ) \big ) + 1 0 ^ { - 6 } \big ) , } & { M = \mathrm { v H } , } \end{array} \right.\tag{56}
$$

where the superscripts mod and ref denote the rollout statistic and the reference statistic, and rms(·) is the root mean square of a sample. The 1-Wasserstein distance $W _ { 1 }$ between two samples $\mathcal { V }$ and $\mathcal { V } ^ { \prime }$ is evaluated at evenly spaced probability levels,

$$
W _ { 1 } ( y , y ^ { \prime } ) = \frac { 1 } { J } \sum _ { n = 0 } ^ { J - 1 } \Bigl | F _ { y } ^ { - 1 } \Bigl ( \frac { n } { J - 1 } \Bigr ) - F _ { y ^ { \prime } } ^ { - 1 } \Bigl ( \frac { n } { J - 1 } \Bigr ) \Bigr | , \qquad J = \operatorname* { m i n } \bigl ( | \mathcal { V } | , | \mathcal { V } ^ { \prime } | , 1 2 0 , 0 0 0 \bigr ) ,\tag{57}
$$

with $F _ { y } ^ { - 1 }$ the empirical quantile function of ${ \mathcal { V } } ,$ linearly interpolated between order statistics, and |Y| the size of a sample. The two samples difer in size, and the probability levels replace the rank pairing of Equation 36.

For the PIEZO-TREK system, the per-epoch evaluation computes the distances of the model to replica 1 alone, and the score uses the reference pooled over both replicas. For the mean squared displacement, the non-Gaussian parameter and the VACF, the model value of each rollout seed is recovered from its distance to replica 1, the value of replica 1 and the mean model curve, and the distance to the pooled reference is computed from this value. The per-epoch evaluation of this system provides the van Hove distance of the model to replica 1 but not the van Hove sample of the model. The van Hove distance to the pooled reference is replaced by an upper bound from the triangle inequality of $W _ { 1 }$ ，

$$
\delta _ { s , \mathrm { v H } } ( \tau ) \leq \frac { W _ { 1 } \bigl ( \mathcal { V } _ { s } ^ { \mathrm { m o d } } , \mathcal { V } _ { s } ^ { \mathrm { r e p 1 } } \bigr ) + W _ { 1 } \bigl ( \mathcal { V } _ { s } ^ { \mathrm { r e p 1 } } , \mathcal { V } _ { s } ^ { \mathrm { r e f } } \bigr ) } { \mathrm { r m s } \bigl ( \mathcal { V } _ { s } ^ { \mathrm { r e f } } \bigr ) + 1 0 ^ { - 6 } } ,\tag{58}
$$

where $\mathcal { V } _ { s } ^ { \mathrm { r e p 1 } }$ is the van Hove sample of replica 1 alone and the lag argument $\tau$ is omitted on the right. The score of a PIEZO-TREK checkpoint uses a value at least as large as the true van Hove distance, up to the quantile discretization of Equation 57.

## 2.13 Perfect-model allowance and excess score

The allowance of a cell is the distance that a perfect model reaches or exceeds with a probability of 5 percent, given a reference of the same size and layout as the MD reference. A reference built from two finite replicas difers from the long-run statistics of the process that produced it, and a perfect model has a nonzero distance in every cell.

Two trained models of each system stand in for the true process. These models are distinct from the evaluated models and come from separate training runs with other settings. Each stand-in model is rolled out from its own initial conditions for 40,000 steps, with 40 rollouts per model, and its rollouts are divided into two halves. The long-run statistics of one half, computed from all rollouts at full length, serve as the true statistics. The construction is a parametric bootstrap [55], in which replicate references are simulated from fitted models instead of being resampled from the MD.

Each draw builds a reference from the other half in the layout of the MD reference. For each MD replica, the draw takes a distinct rollout and cuts from it a contiguous segment of the length of that replica at a random ofset. The cut reference is assembled in the same way as the MD reference, and its distance to the true statistics, $\delta _ { s , M } ^ { \mathrm { d r a w } } ( \tau )$ computed with Equation 56, measures the deviation introduced by a finite reference alone. Both halves serve in turn as the true statistics, and the draws of the two standin models are pooled, which gives 600 draws for the PIEZO system and 960 draws for the PIEZO-TREK system.

The raw allowance $\tilde { A } _ { s , M } ( \tau )$ is the 95th percentile of $\delta _ { s , M } ^ { \mathrm { d r a w } } ( \tau )$ over the draws. For lags of 100 and above, the raw allowance is smoothed along the ladder. The smoothed allowance at a lag is the value at that lag of a straight line fitted by least squares to ln A<sup>˜</sup> against ln τ over the five ladder points centered on it, with lag 1 excluded and with fewer points at the ends of the ladder, but never fewer than three.

At lag 1, the allowance is the larger of A<sup>˜</sup> and the 95th percentile of the same distance between the MD reference and 400 of its circular block-bootstrap resamples [56, 57]. The bootstrap resamples each MD replica in blocks of 1,000 frames with uniformly placed starts, wraps around the end of the replica and moves whole frames, which keeps all proteins of a frame together. The result is the allowance $A _ { s , M } ( \tau )$

In cells where the distance between the two MD replicas exceeds both the 95th percentile of the corresponding distance between draws and the allowance, the allowance is replaced by the distance between the replicas. With $\Gamma _ { s , M } ( \tau )$ the distance of Equation 56 between the two MD replicas and $\Gamma _ { s , M } ^ { \mathrm { d r a w } } ( \tau )$ the same distance between the two replica cuts of one draw, the guarded allowance is

$$
A _ { s , M } ^ { \mathrm { g } } ( \tau ) = \left\{ \begin{array} { l l } { \Gamma _ { s , M } ( \tau ) } & { \mathrm { i f ~ } \Gamma _ { s , M } ( \tau ) > \mathrm { p c t } _ { \mathrm { g } 5 } \left( \Gamma _ { s , M } ^ { \mathrm { d r a w } } ( \tau ) \right) \mathrm { ~ a n d ~ } \Gamma _ { s , M } ( \tau ) > A _ { s , M } ( \tau ) , } \\ { A _ { s , M } ( \tau ) } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{59}
$$

where $\mathrm { p c t _ { 9 5 } }$ is the 95th percentile over the draws.

For one checkpoint and one rollout seed, the excess score is the mean, over all cells of the system, of the rectified excess of the distance over its allowance,

$$
\mathcal { D } = \frac { 1 } { \left| \mathcal { C } \right| } \sum _ { ( s , M , \tau ) \in \mathcal { C } } \operatorname* { m a x } \left( 0 , \frac { \delta _ { s , M } ( \tau ) } { \operatorname* { m a x } \left( A _ { s , M } ^ { \mathrm { g } } ( \tau ) , 1 0 ^ { - 4 } \right) } - 1 \right) ,\tag{60}
$$

where $\mathcal { C }$ is the set of cells, consisting of the four statistics at every lag of $\mathcal { T } _ { s }$ for every species s, and |C| is its size, 204 for the PIEZO system and 408 for the PIEZO-TREK system. All cells have the same weight. The score of a checkpoint, D<sup>¯</sup>, is the mean of D over its three rollout seeds. A lower score indicates closer agreement with the reference, and $\bar { \mathcal { D } } = 0$ when every cell lies within its allowance. The score of a trained model uses the guarded allowance $A _ { s , M } ^ { \mathrm { g } } ( \tau )$ , which is built from the draws of both stand-in models and widened at lag 1 by the bootstrap.

The threshold $\mathcal { D } _ { 9 5 }$ of a system is the 95th percentile of the excess score that the stand-in models obtain out of sample. Each draw of a stand-in model is scored once with Equation 60, against its own cut reference and against allowances and a guard built from the draws of the other stand-in model alone, without the bootstrap widening at lag 1. No selection over epochs or seeds enters the threshold. A checkpoint is at or below the threshold when $\bar { \mathcal { D } } \leq \mathcal { D } _ { 9 5 }$

## 2.14 Model selection

The model of each training seed is the checkpoint of its best epoch. For each of the five training seeds of a system, the checkpoints of all 25 epochs are evaluated with the protocol of Sections 2.12 and 2.13, and the selected epoch is

$$
\hat { E } = \underset { E \in \{ 0 , \ldots , 2 4 \} } { \arg \operatorname* { m i n } } \ \bar { \mathcal { D } } _ { E } ,\tag{61}
$$

where E is the zero-based epoch index, the checkpoint of epoch E has been trained for $1 5 ( E + 1 )$ optimizer steps, and $\bar { \mathcal { D } } _ { E }$ is its score. This procedure gives five models per system and ten models in total, each at its own best epoch.

The score used to select an epoch and the score reported for the selected model come from the same evaluation. Both use the same 512 initial conditions, the same three rollout seeds and the same reference, which pools both replicas of the system and includes the training frames. No validation loss is used, and the training loss does not enter the selection.

The selection is also repeated with one rollout seed held out. The epoch of each run is chosen from the mean of D over two of its three rollout seeds and scored on the remaining seed, for each of the three choices of the held-out seed. The initial conditions and the reference are the same for all three seeds, and this check holds out rollout noise and not data.

The threshold $\mathcal { D } _ { 9 5 }$ depends on the reference layout and on the stand-in models and does not depend on the trained model.

## 2.15 Metrics

The time-averaged mean squared displacement (TAMSD) of a protein is its squared lateral displacement averaged over all time origins at a given lag time τ. For a single trajectory of $T$ frames with unwrapped lateral positions $\mathbf { r } ( t )$ , the TAMSD is

$$
\overline { { \Delta r ^ { 2 } } } ( \tau ) = \frac { 1 } { T - \tau } \sum _ { t = 0 } ^ { T - \tau - 1 } \Vert \mathbf { r } ( t + \tau ) - \mathbf { r } ( t ) \Vert ^ { 2 } ,\tag{62}
$$

where the squared displacement is $\left( x ( t + \tau ) - x ( t ) \right) ^ { 2 } + \left( y ( t + \tau ) - y ( t ) \right) ^ { 2 }$ , with x and y the lateral coordinates. The TAMSD is the mean squared displacement of Equation 42 for a single protein and a single trajectory.

For normal difusion in two dimensions, the ensemble average of the TAMSD grows linearly with the lag time, $\langle \overline { { \Delta r ^ { 2 } } } ( \tau ) \rangle = 4 D \tau$ , and the difusion coeficient D is obtained from this relation. The anomalous exponent α is obtained by a fit of the TAMSD curves to $\overline { { \Delta r ^ { 2 } } } ( \tau ) \propto \tau ^ { \alpha }$ . The motion is normal difusion for $\alpha = 1$ , subdifusion for $\alpha < 1$ and superdifusion for $\alpha > 1$

The self part of the van Hove function, $G _ { s } ( \boldsymbol { r } , \tau )$ , is the probability density of the lateral displacement of a protein over a lag time τ, expressed as a function of the displacement magnitude r,

$$
G _ { s } ( r , \tau ) = \frac { 1 } { 2 \pi r } \left. \delta \Big ( r - \sqrt { \Delta x ^ { 2 } + \Delta y ^ { 2 } } \Big ) \right. ,\tag{63}
$$

where $\Delta x = x ( t _ { 0 } + \tau ) - x ( t _ { 0 } )$ and $\Delta y = y ( t _ { 0 } + \tau ) - y ( t _ { 0 } )$ , and the brackets denote an average over all time origins $t _ { 0 }$ and all trajectories of the ensemble. The function satisfies the radial normalization $\begin{array} { r } { \int _ { 0 } ^ { \infty } 2 \pi r G _ { s } ( r , \tau ) d r = 1 } \end{array}$ . Equation 55 gives the empirical sample from which this function is estimated. The results compare the van Hove functions of the MD and of the model at several lag times.

For Gaussian displacements, the self part of the van Hove function is

$$
G _ { s } ( \boldsymbol { r } , \tau ) = \frac { 1 } { 2 \pi \sigma ^ { 2 } ( \tau ) } \exp \left( - \frac { r ^ { 2 } } { 2 \sigma ^ { 2 } ( \tau ) } \right) ,\tag{64}
$$

where $\sigma ^ { 2 } ( \tau )$ is the variance of each Cartesian component of the lateral displacement at lag time τ .

The non-Gaussian parameter measures the deviation of the distribution of lateral displacements from a Gaussian distribution,

$$
\alpha _ { 2 } ( \tau ) = \frac { \langle r ^ { 4 } ( \tau ) \rangle } { 2 \langle r ^ { 2 } ( \tau ) \rangle ^ { 2 } } - 1 ,\tag{65}
$$

where $r ^ { 2 } ( \tau ) = \Delta x ^ { 2 } + \Delta y ^ { 2 } , r ^ { 4 } ( \tau ) = ( \Delta x ^ { 2 } + \Delta y ^ { 2 } ) ^ { 2 }$ , and the averages run over all proteins and time origins at lag time τ. This quantity is the non-Gaussian parameter of Equation 43. For Gaussian displacements in two dimensions, $\langle r ^ { 4 } \rangle \ : = \ : \hat { 2 } \langle r ^ { 2 } \rangle ^ { 2 }$ and $\alpha _ { 2 } = 0$ . Positive values correspond to tails heavier than Gaussian, and negative values correspond to tails narrower than Gaussian.

## 3 Results

## 3.1 7 PIEZO

In a previous publication of our lab [9], we reported VAR models succeed in describing short time dynamics of a multi-protein (7 PIEZO) system difusion in the membrane but fails to describe mid/long term dynamics. As mentioned in the methods section for this particular multi-protein system, we improved the forecasting in trajectories, by introducing our new moel architecture described in Methods.

Now, as we can see in Fig. 1 we compare the ground truth trajectories and difusive statistics with those obtained from our predictions using our physics informed model and the best model from the VAR collection published in [9], namely the VAR- SDE model with a per-step latent resampling with non-recurrent decoder. In Fig. 1 a) the whole unwrapped trajectories are displayed, for the MD simulations (ground truth with 10,651 time frames). The same is shown for the NS model prediction Fig. 1 b) and for those of the VAR-SDE model Fig. 1 c). In the case of the NS predictions the trajectories were sampled during an extent of 4 more times compared with the ground truth, e.g. 40,000 time frames. For the case of the VAR-SDE model the trajectories are just shown for 10,000 time frames, since they only attempted to replicate the dynamics of the Ground Truth. As we observe the NS model tracks visually resemble more the Ground Truth’s, Brownian like motion, when compared with those obtained by using the VAR-SDE model.

In $\mathrm { F i g . } \ 1 \ d \ – f )$ we show the TAMSD Eq. 62 and the ensemble TAMSD (ETAMSD) for the respective set of trajectories (Ground Truth, NS Model, VAR-SDE model). Having that for the Ground Truth there are 7 trajectories, for the NS Model 42 tracks, and for the VAR-SDE model 35 trajectories. The mean anomalous exponent α defined by Eq. was obtained by fitting each TAMSD curve and computing the mean value for each case. For the Ground Truth the mean anomalous exponent is $\bar { \alpha } = 1 . 0 4 9 \pm 0 . 0 2 2$ (SEM), for the NS Model we have $\bar { \alpha } = 1 . 0 4 8 \pm 0 . 0 1 2 ~ \mathrm { ( S E M ) }$ , and for the VAR-SDE model we reported $\bar { \alpha } = 1 . 5 6 3 \pm 0 . 0 1 6 ~ \mathrm { ( S E M ) }$ . So we can conclude that the NS model accurately describes the difusion regime in the seven PIEZO system, and predicts the existence of normal difusion over 4 times the actual Ground Truth’s trajectories.

The Van Hove functions Eq.63 (see Fig. 1 g-i) show that the NS model better reproduces the displacement distributions over diferent lag times. Namely compared with the VAR prediction for which the Van Hove distribution exhibits large tails, the corresponding function for the NS model for the lag times analyzed $\Delta t =$ {100, 500, 1000, 2000} remain Gaussian (see Eq. 64), curbing the prescence of considerably large displacements. And further the non-Gaussian parameter Eq. 65 (see Fig. 1 j-l) provides additional evidence that the NS model is able to capture the underlying dynamics. In particular the non Gaussian Parameter in the NS prediction Fig. 1 k) for lag times $\Delta t > 4 0 0 0$ takes negative values implying the presence of suppressed tails in accordance to the respective metric in the MD simulation see Fig. 1 j). Overall, the results in Fig. 1 demonstrate that our NS model provides a substantially improved description of the multi-protein difusion dynamics compared with the previously developed VAR-SDE architecture. In particular, the close agreement between the TAMSD curves of the ground truth and NS model is reflected in their anomalous difusion exponents, with $\bar { \alpha } = 1 . 0 4 9 \pm 0 . 0 2 2$ for the MD simulations and $\bar { \alpha } = 1 . 0 4 8 \pm 0 . 0 1 2$ for the NS predictions. In contrast, the VAR-SDE model exhibits a considerably larger exponent, $\bar { \alpha } = 1 . 5 6 3 \pm 0 . 0 1 6$ , indicating a departure from the nearly Brownian dynamics observed in the MD simulations. Taken together, these results indicate that our model successfully captures multiple aspects of the transport statistics of the seven PIEZO system, extending the predictive time horizon to approximately four times the duration of the ground-truth trajectory while maintaining the observed difusive behavior.

![](images/5f584dc984f7d9874b4311208eb8ef8416c74b476b00c2dad44e761d7a6c899c.jpg)

![](images/c7bece3667093fd2e1fe397097fe9e93640756bb07f56fc37b3c1831e2abb9b6.jpg)

![](images/67841e8263f90e5696a56e4d05d318e5af58dfdf06ca586ca7d973b3d17b80f8.jpg)

![](images/ec00af438e4d74bafefd928c04fdced0caadd21ef83efefef43dfe7b222c36c4.jpg)

![](images/0d776a6707c55813b90590c6d52ebed3ab401a35f929d4d894f782785ebd1d43.jpg)

![](images/7ca54b04cb04d3f873582a6ec0f7916cc80b87336fa451cc2bf9e2a316703da5.jpg)

![](images/dd9d5e4a06940dee283a713221d73cc64810fb94d2baf6d4ae0d82bb3f7336de.jpg)

![](images/a2d16b71b546d14e9912ae3eaaed02ff650c74e81c907add0a566aa179553e2f.jpg)

![](images/59ce9aa4d11269d19d8381965ba86fe1e3209af2e6c1e7427ecf555d49a2bff7.jpg)

![](images/02a5976587aea1e3fec5cc4aacc68a58213d8d2b4255016b81d28f9eb20af5b0.jpg)

![](images/3ef5167f7bf8aa8032e8346542a9c7caa82fc8a947a177808b3b2cd3d3e8ec09.jpg)

![](images/8c3d06bc1ff1ecb223d0e90ccf80c1d5304768a653480c4dd6878dff780f2e64.jpg)  
Fig. 1 Comparison between the ground truth (MD simulations) and predictions of our neural stochastic (NS) model and VAR model SDE [9], for the 7 PIEZO multi-protein system. a) Trajectories of the 7 PIEZO channels (center of mass) obtained from of MD simulations during a total time of 10,651 time frames. b) Predicted trajectories using the NS model of the PIEZO channels during a total time with 40,000 time frames. c) Predicted trajectories obtained by the VAR-SDE model for tracks with 10,000 time frames. d-f) TAMSD and ETAMSD for ground truth trajectories and predictions respectively. For convergence of the statistics, the TAMSD is computed in 10% of the trajectories (cite). g-i) Van Hove function Eq. for selected lag-times $\Delta t = \{ 1 0 0 , \bar { 5 } 0 0 , 1 0 0 0 , 2 0 0 0 \}$ , for MD simulations and predictions. And j-l) Non-Gaussian parameter Eq. for lag times covering the whole trajectories of the ground truth and each prediction respectively.

## 3.2 7 PIEZO and 10 TREK

As we mentioned we also implemented our new NS model for describing and predicting the difusive dynamics of a multi-protein with two diferent species of channels. For the 7 PIEZO and 10 TREK system the MD trajectories and those predicted by our model are shown in Fig. 2, displaying the ground truth’s tracks in ${ \mathrm { F i g . ~ } } 2 \ a )$ for MD simulations with 9,000 time frames. And in Fig. 2 b) we display the predicted trajectories using the NS model for 40,000 time frames. In both cases the 7 PIEZO trajectories are shown in red lines and the 10 TREK tracks are shown in blue lines. As we can see in Fig. 2 Brownian-like dynamics is displayed in both cases, still there are diferences in between diferent protein dynamics. For instance PIEZO channels tend to make shorter displacements when compared to TREK trajectories, this efect is more evident in the MD simulations.

![](images/0a5a58d197bc5c126a485bf98da1e77db2d46d012ce02efb12576420f44ae5b5.jpg)

![](images/33fc8020f4737d94f2e6f5863b0bc0906fa74767f82a8c76e40f46cb8db3c1fb.jpg)  
Fig. 2 Comparison between the ground truth (MD simulations) and predictions of our neural stochastic (NS) model , for the 7 PIEZO and 10 TREK multi-protein system. $a )$ Trajectories of the 7 PIEZO channels (red lines) and 10 TREK channels (blue lines) obtained from of MD simulations during a total time of 9,000 time frames. b) Predicted trajectories using the NS model of the PIEZO channels and TREK during a total time with 40,000 time frames.

Now as we observe in Fig. 3 the TAMSD $\operatorname { E q . }$ 62 analysis shows linear dependence of the TAMSD and ETAMSD with respect to the lag time. Having that in all cases the ETAMSD renders anomalous exponents close to one, indicating Brownian dynamics in the MD simulations and NS predictions. Particularly for the PIEZO channels in the Ground Truth we have $\bar { \alpha } = 0 . 9 4 3 { \pm } 0 . 0 1 6 ~ \mathrm { ( S E M ) }$ . In the case of the TREK channels in the Ground Truth the mean anomalous exponent is $\bar { \alpha } = 0 . 9 9 5 \pm 0 . 0 1 9 ~ \mathrm { ( S E M ) }$ . For the NS predictions the PIEZO channels gave a anomalous exponent of $\bar { \alpha } = 1 . 0 1 5 \pm 0 . 0 1 3$ (SEM) and $\bar { \alpha } = 1 . 0 1 1 \pm 0 . 0 1 1$ (SEM) for the TREK trajectories. As we note the diference in the values of the anomalous exponent between the ground truth and the NS predictions is covered by the SEM in all cases, so we can say that the NS model predicts accurately the difusion regime (normal difusion) even in a diferent specie multi-protein system.

The diference in the spreading of the displacements for PIEZO and TREK as mentioned is present in Fig. 2 was also replicated by the NS model and this can be corroborated by the respective Van Hove function Eq. 63 as we show in Fig.4. For the MD simulations for lag times of $\Delta t = \{ 1 0 0 , 5 0 0 \}$ the Van Hove function exhibits large tails when compared to the Gaussian distribution Eq. (64) and for $\Delta t = \{ 1 0 0 0 , 2 0 0 0 \}$ the distribution of displacements displacements exhibits suppressed tails with respect to the Gaussian distribution. For TREK tracks in the Ground Truth for $\Delta t = \{ 1 0 0 , 5 0 0 , 1 0 0 0 , 2 0 0 0 \}$ the distribution of displacements exhibits large tails. For the NS prediction and for the lag times analyzed in the case of PIEZO proteins the Van Hove functions resembles a Gaussian distribution and the for the TREK channels for lag times $\Delta t = \{ 1 0 0 , 5 0 0 \}$ the distribution of displacements is Gaussian, whereas for lag times $\Delta t = \{ 1 0 0 0 , 2 0 0 0 \}$ the Van Hove function displays large tails. This behavior is corroborated by the evolution of the non Gaussian parameter Eq. (65) as we show in Fig. 5.

![](images/234d80b839c2281dceabceb1d0f6cff2463ec983f2569a7510602a1cabbf571a.jpg)

![](images/64e8ab343b54dac958a7d9c0e3b88648892876c86222e8af8ab3651033cc2536.jpg)

![](images/3b9280035eeb0602f416033f4cde35e8bd4dc098aa4e6917928c380b66fd763b.jpg)

![](images/9745ba3e4a44e6f3280a54f2ee79480cdbf1c752457b3e72a28c62e8f5e6bb41.jpg)  
Fig. 3 TAMSD: comparison between the ground truth (MD simulations) and predictions of our neural stochastic (NS) model , for the 7 PIEZO and 10 TREK multi-protein system. a-b) Ground Truth TAMSD (gray lines) and ETAMD (black line) for the 7 PIEZO channels and the 10 TREK proteins respectively. c-d) TAMSD and ETAMSD for the NS-Model for the 5 rollouts of the 7PIEZO and 10 TREK system, with 35 TAMSD curves for the PIEZO channels see c) and 50 TAMSD curves for TREK channels see d).

As we show in $\mathrm { F i g . ~ 5 ~ } a )$ the fluctuating behavior of the non Gaussian parameter in PIEZO channels for the MD simulations is preserved for lag times $\Delta t > 2 0 0 0$ , but for ∆t > 8000 remains negative, implying the presence of suppressed tails. In the case of TREK trajectories in the Ground Truth for lag times $\Delta t > 2 0 0 0$ the non Gaussian parameter is remains positive indicating large tails in the distribution of displacements. This trend is hold by the non Gaussian parameter in the NS predictions.

Overall, the Van Hove functions and the non Gaussian parameter show that the NS model reproduces the contrasting displacement statistics of PIEZO and TREK channels. In particular, it captures the tendency toward suppressed tails in PIEZO displacement distributions and enhanced tails in TREK distributions, preserving the distinct dynamical behavior of both protein populations over extended prediction times.

## 4 Discussion

In this work, we introduced a neural stochastic model for forecasting the long-time difusion dynamics of membrane proteins from coarse-grained molecular dynamics simulations. A central challenge in this task is that reproducing short-time displacement statistics does not necessarily guarantee accurate long-time dynamics, particularly when predictions are generated autoregressively without further input from the reference trajectories. Our results show that combining a multirate recurrent architecture with a stochastic latent process provides a way to retain temporal correlations over extended rollouts. In particular, the model reproduces the nearly Brownian difusion regime of the seven-PIEZO system, substantially improving upon the previously developed VAR-SDE architecture [9], which exhibits a pronounced departure from the reference dynamics at intermediate and long times.

![](images/f1b97e1592abf0270c5848071b4912d08ef56603b077e742be4c783942e1a718.jpg)

![](images/836808a8a096ff308dc3df01a1b6ff61f0fda3fe469b7407f4839b578224da52.jpg)

![](images/2cc7bc770961f54a2e3f04234c85dc747d402aa02f1389eb3ec8adc22185fc0a.jpg)

![](images/f738a82e3f74c49d49b5b32f712a6ce354db17bf4ba9620e6290aa8432507274.jpg)  
Fig. 4 Van Hove function: comparison between the ground truth (MD simulations) and predictions of our neural stochastic (NS) model , for the 7 PIEZO and 10 TREK multi-protein system. a-b) Ground Truth Van Hove function for the 7 PIEZO channels and the 10 TREK proteins respectively. c-d) Van Hove function for the NS-Model for the 5 rollouts of the 7PIEZO and 10 TREK system, for the PIEZO channels see c) and for TREK channels see d).

An important aspect of the proposed model is its ability to describe systems containing multiple interacting protein populations. In the mixed PIEZO-TREK system, the model reproduces the distinct mobilities of the two protein species, despite their simultaneous difusion within the same membrane environment. The diferences observed in their displacement distributions are also reflected in the Van Hove functions and non Gaussian parameters. This suggests that the model captures aspects of the collective stochastic dynamics that cannot be described solely by matching the mean squared displacement. The use of species-specific parameters, together with a cross-attention decoder that allows information to be exchanged between proteins, provides a flexible framework for representing heterogeneous multi-protein systems.

Beyond reproducing the transport statistics of the reference simulations, the ability to generate trajectories substantially longer than those available from MD simulations ofers a potential computational advantage. The model generates 40,000 frame trajectories, extending beyond the duration of the reference trajectories while preserving several key statistical properties of the underlying dynamics. Such long rollouts could be useful for investigating transport observables that require extensive trajectory sampling, including displacement distributions, temporal correlations, and long-time difusion behavior. Nevertheless, the generated trajectories should be regarded as statistical surrogates of the simulated dynamics rather than exact reconstructions of individual molecular trajectories or replacements for the underlying molecular dynamics.

Several limitations should also be considered. First, the model is trained and evaluated on a limited set of coarse-grained membrane systems, and its ability to generalize to diferent protein compositions, membrane properties, and simulation conditions remains to be established. Second, although the architecture incorporates periodic boundary conditions, pairwise geometric information, and stochastic latent dynamics, it does not explicitly enforce the underlying molecular force field or membrane energetics. Its predictions therefore depend on the statistical information captured during training. Future work could investigate the inclusion of additional physical constraints and the evaluation of the model across a broader range of membrane systems. These developments would help establish the extent to which neural stochastic models can provide reliable and computationally eficient tools for studying long-time membrane-protein dynamics.

![](images/3a711a5d66071b3f14ba07c5e10efd5788b534f5644b98cef17323bd559b98c4.jpg)

![](images/a823846a7e1af5b3be1bc52cb4b57f3856977793bc36b078e57f9cd4dc10da02.jpg)

![](images/1451cd72923752e90deb520b76103bb31004f1a539940343eebabe35c91a3689.jpg)

![](images/3d27d495325c028e4bd361e0e6dc290d66bcb384e4ae69a76fcc4261ac54c7a1.jpg)  
Fig. 5 Non Gaussian parameter: comparison between the ground truth (MD simulations) and predictions of our neural stochastic (NS) model , for the 7 PIEZO and 10 TREK multi-protein system. a-b) Ground Truth TAMSD for the 7 PIEZO channels and the 10 TREK proteins respectively. c-d) Non Gaussian parameter for the NS-Model employing 5 rollouts for the 7PIEZO and 10 TREK system, for the PIEZO channels see c) and for TREK channels see d).

## Supplementary information.

Acknowledgments. This work used resources available through the National Research Platform (NRP) at the University of California, San Diego. NRP has been developed, and is supported in part, by funding from National Science Foundation, from awards 1730158, 1540112, 1541349, 1826967, 2112167, 2100237, and 2120019, as well as additional funding from community partners.

## Declarations

## References

[1] Javanainen, M., Hammaren, H., Monticelli, L., Jeon, J.-H., Miettinen, M.S., Martinez-Seara, H., Metzler, R., Vattulainen, I.: Anomalous and normal difusion of proteins and lipids in crowded lipid membranes. Faraday discussions 161, 397–417 (2012)

[2] Cho, W., Stahelin, R.V.: Membrane-protein interactions in cell signaling and membrane traficking. Annu. Rev. Biophys. Biomol. Struct. 34(1), 119–151 (2005)

[3] Owen, D.M., Williamson, D., Rentero, C., Gaus, K.: Quantitative microscopy: protein dynamics and membrane organisation. Trafic 10(8), 962–971 (2009)

[4] Alenghat, F.J., Golan, D.E.: Membrane protein dynamics and functional implications in mammalian cells. In: Current Topics in Membranes vol. 72, pp. 89–120. Elsevier, ??? (2013)

[5] International Transporter Consortium: Membrane transporters in drug develop ment. Nature reviews Drug discovery 9(3), 215–236 (2010)

[6] Saxton, M.J., Jacobson, K.: Single-particle tracking: applications to membrane dynamics. Annual review of biophysics and biomolecular structure 26(1), 373–399 (1997)

[7] Barkai, E., Garini, Y., Metzler, R.: Strange kinetics of single molecules in living cells. Physics today 65(8), 29–35 (2012)

[8] Ly, A.T., Freites, J.A., Bertaccini, G.A., Evans, E.L., Dickinson, G.D., Tobias, D.J., Pathak, M.M.: Single-particle tracking reveals heterogeneous piezo1 difusion. Biophysical Journal 124(24), 4457–4470 (2025)

[9] Thai, P.C., Soria, M.H., Levine, J.E., Wen, C.Y., Luo, Y.L., Kosaraju, S.C.: Variational generative modeling for forecasting protein difusion from molecular dynamics. In: Proceedings of the 17th ACM International Conference on Bioinformatics, Computational Biology and Health Informatics, pp. 1–6 (2026)

[10] Wen, C.Y., Luo, Y.L., Madsen, J.J.: Optimizing a coarse-grained model for largescale membrane protein simulation. Biophysical Reports 5(4) (2025)

[11] Glogowska, E., Arhatte, M., Chatelain, F.C., Lesage, F., Xu, A., Grashof, C., Discher, D.E., Patel, A., Honore, E.: Piezo1 and piezo2 foster mechanical gating of k2p channels. Cell reports 37(9) (2021)

[12] Williams, R.J., Zipser, D.: A learning algorithm for continually running fully recurrent neural networks. Neural Computation 1(2), 270–280 (1989) https://doi. org/10.1162/neco.1989.1.2.270

[13] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R.: Dropout: A simple way to prevent neural networks from overfitting. Journal of Machine Learning Research 15(56), 1929–1958 (2014)

[14] Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E., Garcia, D., Ginsburg, B., Houston, M., Kuchaiev, O., Venkatesh, G., Wu, H.: Mixed precision

training. In: International Conference on Learning Representations (ICLR 2018) (2018)

[15] Kalamkar, D., Mudigere, D., Mellempudi, N., Das, D., Banerjee, K., Avancha, S., Vooturi, D.T., Jammalamadaka, N., Huang, J., Yuen, H., Yang, J., Park, J., Heinecke, A., Georganas, E., Srinivasan, S., Kundu, A., Smelyanskiy, M., Kaul, B., Dubey, P.: A study of BFLOAT16 for deep learning training. arXiv preprint arXiv:1905.12322 (2019) https://doi.org/10.48550/arXiv.1905.12322

[16] Thompson, A.P., Aktulga, H.M., Berger, R., Bolintineanu, D.S., Brown, W.M., Crozier, P.S., in ’t Veld, P.J., Kohlmeyer, A., Moore, S.G., Nguyen, T.D., Shan, R., Stevens, M.J., Tranchida, J., Trott, C., Plimpton, S.J.: LAMMPS: a flexible simulation tool for particle-based materials modeling at the atomic, meso, and continuum scales. Computer Physics Communications 271, 108171 (2022) https: //doi.org/10.1016/j.cpc.2021.108171

[17] Coste, B., Mathur, J., Schmidt, M., Earley, T.J., Ranade, S., Petrus, M.J., Dubin, A.E., Patapoutian, A.: Piezo1 and Piezo2 are essential components of distinct mechanically activated cation channels. Science 330(6000), 55–60 (2010) https: //doi.org/10.1126/science.1193270

[18] Wang, L., Zhou, H., Zhang, M., Liu, W., Deng, T., Zhao, Q., Li, Y., Lei, J., Li, X., Xiao, B.: Structure and mechanogating of the mammalian tactile channel PIEZO2. Nature 573(7773), 225–229 (2019) https://doi.org/10.1038/ s41586-019-1505-8

[19] Goldstein, S.A.N., Bayliss, D.A., Kim, D., Lesage, F., Plant, L.D., Rajan, S.: International Union of Pharmacology. LV. Nomenclature and molecular relationships of two-P potassium channels. Pharmacological Reviews 57(4), 527–540 (2005) https://doi.org/10.1124/pr.57.4.12

[20] Allen, M.P., Tildesley, D.J.: Computer Simulation of Liquids, 2nd edn. Oxford University Press, Oxford (2017). https://doi.org/10.1093/oso/9780198803195. 001.0001

[21] Hendrycks, D., Gimpel, K.: Gaussian error linear units (GELUs). arXiv preprint arXiv:1606.08415 (2016) https://doi.org/10.48550/arXiv.1606.08415

[22] Elfwing, S., Uchibe, E., Doya, K.: Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. Neural Networks 107, 3–11 (2018) https://doi.org/10.1016/j.neunet.2017.12.012

[23] Ba, J.L., Kiros, J.R., Hinton, G.E.: Layer normalization. arXiv preprint arXiv:1607.06450 (2016) https://doi.org/10.48550/arXiv.1607.06450

[24] Miyato, T., Kataoka, T., Koyama, M., Yoshida, Y.: Spectral normalization for generative adversarial networks. In: International Conference on Learning

[25] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. In: Advances in Neural Information Processing Systems, vol. 30, pp. 5998–6008 (2017)

[26] He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778 (2016). https://doi.org/10.1109/CVPR.2016.90

[27] Lee, J., Lee, Y., Kim, J., Kosiorek, A.R., Choi, S., Teh, Y.W.: Set Transformer: A framework for attention-based permutation-invariant neural networks. In: Proceedings of the 36th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 97, pp. 3744–3753 (2019)

[28] Veliˇckovi´c, P., Cucurull, G., Casanova, A., Romero, A., Li\`o, P., Bengio, Y.: Graph attention networks. In: International Conference on Learning Representations (ICLR 2018) (2018). https://doi.org/10.48550/arXiv.1710.10903

[29] Hochreiter, S., Schmidhuber, J.: Long short-term memory. Neural Computation 9(8), 1735–1780 (1997) https://doi.org/10.1162/neco.1997.9.8.1735

[30] Gers, F.A., Schmidhuber, J., Cummins, F.: Learning to forget: Continual prediction with LSTM. Neural Computation 12(10), 2451–2471 (2000) https://doi.org/ 10.1162/089976600300015015

[31] Koutn´ık, J., Gref, K., Gomez, F., Schmidhuber, J.: A clockwork RNN. In: Proceedings of the 31st International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 32, pp. 1863–1871 (2014)

[32] Tzen, B., Raginsky, M.: Neural stochastic diferential equations: Deep latent Gaussian models in the difusion limit. arXiv preprint arXiv:1905.09883 (2019) https://doi.org/10.48550/arXiv.1905.09883

[33] Li, X., Wong, T.-K.L., Chen, R.T.Q., Duvenaud, D.: Scalable gradients for stochastic diferential equations. In: Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics. Proceedings of Machine Learning Research, vol. 108, pp. 3870–3882 (2020)

[34] Uhlenbeck, G.E., Ornstein, L.S.: On the theory of the Brownian motion. Physical Review 36(5), 823–841 (1930) https://doi.org/10.1103/PhysRev.36.823

[35] Gillespie, D.T.: Exact numerical simulation of the Ornstein-Uhlenbeck process and its integral. Physical Review E 54(2), 2084–2091 (1996) https://doi.org/10. 1103/PhysRevE.54.2084

[36] Kingma, D.P., Welling, M.: Auto-encoding variational Bayes. arXiv preprint

[37] Rezende, D.J., Mohamed, S., Wierstra, D.: Stochastic backpropagation and approximate inference in deep generative models. In: Proceedings of the 31st International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 32, pp. 1278–1286 (2014)

[38] Peyr´e, G., Cuturi, M.: Computational optimal transport with applications to data sciences. Foundations and Trends in Machine Learning 11(5-6), 355–607 (2019) https://doi.org/10.1561/2200000073 1803.00567

[39] Burov, S., Tabei, S.M.A., Huynh, T., Murrell, M.P., Philipson, L.H., Rice, S.A., Gardel, M.L., Scherer, N.F., Dinner, A.R.: Distribution of directional change as a signature of complex dynamics. Proceedings of the National Academy of Sciences 110(49), 19689–19694 (2013) https://doi.org/10.1073/pnas.1319473110

[40] Rahman, A.: Correlations in the motion of atoms in liquid argon. Physical Review 136(2A), 405–411 (1964) https://doi.org/10.1103/PhysRev.136.A405

[41] Metzler, R., Jeon, J.-H., Cherstvy, A.G., Barkai, E.: Anomalous difusion models and their properties: non-stationarity, non-ergodicity, and ageing at the centenary of single particle tracking. Physical Chemistry Chemical Physics 16(44), 24128– 24164 (2014) https://doi.org/10.1039/C4CP03465A

[42] Metzler, R., Klafter, J.: The random walk’s guide to anomalous difusion: a fractional dynamics approach. Physics Reports 339(1), 1–77 (2000) https://doi.org/ 10.1016/S0370-1573(00)00070-3

[43] Snoek, J., Larochelle, H., Adams, R.P.: Practical Bayesian optimization of machine learning algorithms. In: Advances in Neural Information Processing Systems, vol. 25, pp. 2951–2959 (2012)

[44] Akiba, T., Sano, S., Yanase, T., Ohta, T., Koyama, M.: Optuna: A nextgeneration hyperparameter optimization framework. In: Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pp. 2623–2631 (2019). https://doi.org/10.1145/3292500.3330701

[45] Sobol’, I.M.: On the distribution of points in a cube and the approximate evaluation of integrals. USSR Computational Mathematics and Mathematical Physics 7(4), 86–112 (1967) https://doi.org/10.1016/0041-5553(67)90144-9

[46] McKay, M.D., Beckman, R.J., Conover, W.J.: Comparison of three methods for selecting values of input variables in the analysis of output from a computer code. Technometrics 21(2), 239–245 (1979) https://doi.org/10.1080/00401706. 1979.10489755

[47] Liu, H., Li, Z., Hall, D., Liang, P., Ma, T.: Sophia: A scalable stochastic secondorder optimizer for language model pre-training. In: International Conference on Learning Representations (ICLR 2024), pp. 1621–1650 (2024)

[48] Hutchinson, M.F.: A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines. Communications in Statistics - Simulation and Computation 18(3), 1059–1076 (1989) https://doi.org/10.1080/03610918908812806

[49] Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: International Conference on Learning Representations (ICLR 2019) (2019)

[50] Werbos, P.J.: Backpropagation through time: what it does and how to do it. Proceedings of the IEEE 78(10), 1550–1560 (1990) https://doi.org/10.1109/5. 58337

[51] Pascanu, R., Mikolov, T., Bengio, Y.: On the dificulty of training recurrent neural networks. In: Proceedings of the 30th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 28, pp. 1310–1318 (2013)

[52] Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., K¨opf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., Chintala, S.: PyTorch: An imperative style, high-performance deep learning library. In: Advances in Neural Information Processing Systems, vol. 32 (2019)

[53] Falcon, W., The PyTorch Lightning team: PyTorch Lightning. Zenodo (2019). https://doi.org/10.5281/zenodo.3828935

[54] Van Hove, L.: Correlations in space and time and Born approximation scattering in systems of interacting particles. Physical Review 95(1), 249–262 (1954) https: //doi.org/10.1103/PhysRev.95.249

[55] Efron, B., Tibshirani, R.J.: An Introduction to the Bootstrap. Chapman & Hall, New York (1993)

[56] K¨unsch, H.R.: The jackknife and the bootstrap for general stationary observations. The Annals of Statistics 17(3), 1217–1241 (1989) https://doi.org/10.1214/ aos/1176347265

[57] Politis, D.N., Romano, J.P.: A circular block-resampling procedure for stationary data. In: LePage, R., Billard, L. (eds.) Exploring the Limits of Bootstrap, pp. 263–270. Wiley, New York (1992)