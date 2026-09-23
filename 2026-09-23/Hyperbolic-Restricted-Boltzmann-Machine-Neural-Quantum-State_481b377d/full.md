# Hyperbolic Restricted Boltzmann Machine Neural Quantum State

H. L. Dao<sup>∗</sup>

September 23, 2026

## Abstract

We construct the first type of non-Euclidean non-autoregressive neural quantum state (NQS) in the form of the hyperbolic Restricted Boltzmann Machine (HRBM), which is studied in the variational Monte-Carlo (VMC) setting of the Quantum Sherrington-Kirkpatrick (QSK) model whose ground state exhibits volume-law entanglement. Across a 512-fold increase in the Hilbert space dimension corresponding to a system size increase from N = 14 to N = 24, HRBM NQS robustly outperforms its Euclidean version, the RBM ${ \mathrm { N Q S } } ,$ in terms of better ground state energy optimization as well as lower R´enyi-2 $S _ { 2 }$ and von Neumann $S _ { v N }$ absolute entanglement entropy reconstruction errors. More importantly, for all QSK system sizes, HRBM NQS demonstrates a superior expressivity in faithfully reproducing the entire entanglement spectrum of the QSK model from the top eigenvalues down to the tail end across 15 orders of magnitude, while RBM NQS consistently overestimates the sub-dominant modes. This work furnishes a proof-of-concept demonstrating that hyperbolic non-autoregressive NQS ansatz¨e, thanks to the exponential volume of the hyperbolic geometry underlying their constructions, might be more natural at representing volume-law quantum systems than conventional Euclidean NQS. Furthermore, an interesting byproduct of this work is the polynomial scaling result of RBM-type NQS ansatz¨e in the QSK volume-law system as the Hilbert space increases exponentially.

## Contents

1 Introduction 1   
2 Hyperbolic RBM construction 2   
3 QSK VMC Experiments 4   
3.1 VMC experiment settings 4   
3.2 Scaling study results . 5   
3.3 Full entanglement spectrum results . 8   
4 Concluding remarks 10   
A Appendix 11   
A.1 Hyperbolic Lorentzian mathematical operations . 11   
A.2 QSK models: Exact energy, $S _ { 2 }$ and S entropies . 13   
A.3 Scaling results at diferent QSK realizations 14   
A.4 Full entanglement spectrum at N = 14 to N = 22 . 17   
A.5 Efects of spatial constraint hyperparameter L<sub>max</sub> 22

References

## 1 Introduction

Within quantum many-body systems, those possessing ground states that exhibit volume-law entanglement property such as quantum spin glasses, SYK and random matrix models (among others), are particularly interesting but dificult to study due to the exponential size of their Hilbert spaces. With the advent of neural quantum states (NQS) [3] -[12] that are capable of eficiently representing many diferent types of quantum many-body systems, these volume-law quantum systems represent a new challenging frontier. Many recent works, such as [1], [2], discussed and addressed the question of whether NQS can efectively model the ground states of these volume-law system. While [1] hypothesized that NQS (feedforward type) required an exponential scaling in order to represent volume-law systems such as the fermionic SYK model, the work [2] refuted this hypothesis by furnishing computational evidences showing that NQS did not require an exponential scaling to represent the volume-law SYK-like model if a suitable representation was chosen to take into account the fermionic sign structure of the fermionic states. In particular, the work [2] considered two volume-law quantum systems, one being the SYK-type disordered fermionic system and the other being the quantum Sherrington-Kirkpatric (QSK) model [18], [19] and showed that a suitably chosen representation of NQS only required a polynomial scaling to represent the volume-law ground states of these systems.

Motivated by these recent developments, and by the results of our own works [14], [15], [16] in which the first few types of non-Euclidean NQS, in the form of hyperbolic Poincar´e and Lorentz recurrent architectures, were constructed and applied in the transverse field Ising model (TFIM) and Heisenberg spin systems, we are interested in constructing new types of hyperbolic NQS for applications in volume-law quantum systems. Specifically, in volume-law quantum systems, the ground state requires an exponentially large Hilbert space capacity to represent dense entangling correlations, so the natural question that arises is, would hyperbolic NQS, similar to those constructed recently in the works [14], [15], [16], be a more natural representation compared to Euclidean NQS, based on the fact that the hyperbolic geometry underlying the construction of hyperbolic NQS is known to have an exponential volume? In this work, we attempt to address this question by presenting a proof-of-concept construction for a new type of hyperbolic NQS, the hyperbolic Restricted Boltzmann Machine (HRBM) NQS, which is benchmarked against the regular or Euclidean RBM NQS in a VMC [13] scaling study of the QSK model with system sizes N ranging from 14 to 24, where exact diagonalization is possible. The hyperbolic NQS constructed in this work is the first hyperbolic non-autoregressive NQS introduced, since the ones constructed in [14], [15], [16] are all autoregressive NQS based on recurrent architectures (including the simple recurrent neural network (RNN) and gated recurrent unit (GRU) architectures).

This paper is organized as follows. In Section 2, we describe the mathematical construction of the hyperbolic RBM using the Lorentz model of hyperbolic space. In Section 3, we briefly describe the QSK model before describing in detail the VMC experiment settings (section 3.1) and finally presenting the main results of this work, which includes a scaling study benchmarking the performance of the newly constructed HRBM NQS against that of the RBM NQS (see section 3.2), as well as a full entanglement spectrum study in section 3.3. Section 4 summarizes the paper. In the Appendix, Section A.1 includes the definitions of all relevant mathematical operations for the Lorentz model of hyperbolic space, Section A.2 lists the exact ground state energies and $S _ { 2 } / S _ { v N }$ entropies of all QSK models considered in this work. These values serve as the exact references against which the performances of both HRBM and RBM NQS ansatzes are benchmarked. Section A.3 includes the scaling study results for RBM and HRBM NQS at diferent realizations of QSK (corresponding to diferent random seeds). Section A.4 includes the full entanglement spectrum plots corresponding to the exact wavefunction and the RBM/HRBM NQS at diferent QSK model sizes N. Finally, Section A.5 includes the results of a comparison study at N = 24 in which diferent HRBM ansatz¨e with diferent hyperparameter $L _ { m a x }$ are studied to choose the optimally performing ansatz.

The Python codes to create and run the HRBM training as well as the trained model weights of the NQS constructed in this work are available at: https://github.com/lorrespz/qsk hrbm. In another departure from our earlier works on hyperbolic NQS which use pytorch and a standalone training pipeline, the HRBM NQS is implemented with jax and we make use of NetKet [20] NQS training framework to perform the training.

## 2 Hyperbolic RBM construction

Before describing the construction of the hyperbolic RBM, we first recall the definition of the RBM.

An RBM consists of two layers of stochastic binary units, the N visible units $v _ { i } ~ ( i = 1 , \ldots , N )$ represent the observable data (such as physical spins in a quantum spin systems) and the M hidden units $h _ { j } \ ( j = 1 , \ldots , M )$ are the latent variables that capture correlations and features among the visible units. The network is restricted in the sense that there are no connections between units in the same layer (in other words, no intra-layer interactions). For a joint state configuration $( v , h )$ , the network assigns an energy:

$$
E ( v , h ) = - \sum _ { i } a _ { i } v _ { i } - \sum _ { j } b _ { j } h _ { j } - \sum _ { i , j } v _ { i } W _ { i j } h _ { j }\tag{1}
$$

where $a _ { i }$ and $b _ { j }$ are bias vectors for the visible and hidden layers, $W _ { i j }$ is the weight matrix connecting visible unit i to hidden unit $j .$ The joint probability of a configuration is governed by the classical Boltzmann distribution:

$$
P ( v , h ) = \frac { 1 } { Z } e ^ { - E ( v , h ) }\tag{2}
$$

where $\begin{array} { r } { Z = \sum _ { v , h } e ^ { - E ( v , h ) } } \end{array}$ is the partition function.

To represent the spin-1/2 systems, the visible layer can be taken to be the physical spin variables $( { \vec { x } } \ =$ $\sigma _ { 1 } ^ { z } , \dots , \sigma _ { N } ^ { z } )$ , so that RBM NQS is defined as follows:

$$
\Psi _ { \mathrm { R B M } } ( { \vec { x } } ) = \sum _ { \{ h _ { j } \} } \exp \left( \sum _ { i } a _ { i } x _ { i } + \sum _ { j } b _ { j } h _ { j } + \sum _ { i j } W _ { i j } h _ { j } x _ { i } \right)\tag{3}
$$

which, after tracing out the hidden units, becomes

$$
\ln \Psi _ { \mathrm { R B M } } ( \vec { x } ) = \sum _ { i = 1 } ^ { N } a _ { i } x _ { i } + \sum _ { j = 1 } ^ { M } \ln \left[ 2 \cosh \left( b _ { j } + \sum _ { i = 1 } ^ { N } x _ { i } W _ { i j } \right) \right]\tag{4}
$$

Note that the vectors $\vec { a } , \vec { b }$ and weight matrix W can be either real (for a real NQS with just the amplitude part) or complex (for a complex NQS with both the amplitude and phase factor).

To construct the hyperbolic RBM, in the case of a real NQS, the hidden weight vector $\vec { b }$ and the input vector ⃗x are mapped to a Lorentz hyperboloid via the exponential map (given in $\mathrm { E q . 1 5 ) }$ while the visible weight vector ⃗a is kept Euclidean, alongside with the weight matrix W. The conversion of $b _ { j }$ to a hyperbolic vector allows for a more expressive and flexible representation that is made possible by the exponential volume of the Lorentz hyperboloid.

$$
\ln \Psi _ { \mathrm { H R B M } } ( \vec { x } ) = \sum _ { i = 1 } ^ { N } a _ { i } x _ { i } + \sum _ { j = 1 } ^ { M } \ln \left\{ 2 \cosh \left[ \log _ { \Omega _ { \mathcal { L } } } \left( \exp _ { \mathbf { 0 } _ { \mathcal { L } } } ( b _ { j } ) \oplus _ { \mathcal { L } } \sum _ { i = 1 } ^ { N } W _ { i j } \otimes _ { \mathcal { L } } \exp _ { \mathbf { 0 } _ { \mathcal { L } } } ( x _ { i } ) \right) \right] \right\}\tag{5}
$$

In Eq.7 above, $\oplus _ { \mathcal { L } } , \otimes _ { \mathcal { L } }$ are the Lorentz addition and multiplication operations (Eq.19 and Eq.21), while $\mathrm { e x p } _ { \mathbf { 0 } _ { \mathcal { L } } } ,$ $\log _ { \mathbf { 0 } _ { \mathcal { L } } }$ are the exponential and logarithmic mappings between the Lorentz hyperboloid and its Euclidean tangent space. These hyperbolic Lorentz mathematical operations are defined in the Appendix A.1. This construction is similar to the hyperbolic recurrent NQS architectures of [14]-[16], in the sense that only the bias vectors are embedded in hyperbolic space (either in the Poincar´e or Lorentz model). The hyperbolic NQS constructions of [14]-[16], are in turns based on the hyperbolic neural networks originally introduced in the work [17] in the context of natural language processing (NLP).

For a complex NQS, a dual hyperbolic embedding is needed to take into account the real and imaginary components of the bias vector $\begin{array} { r } { \vec { b } . } \end{array}$

$$
\ln \Psi _ { \mathrm { H R B M } } ( \vec { x } ) = \sum _ { i = 1 } ^ { N } a _ { i } x _ { i } + \sum _ { j = 1 } ^ { M } \ln \left[ 2 \cosh \left( \theta _ { j , \mathrm { r e } } ( \vec { x } ) + i \theta _ { j , \mathrm { i m } } ( \vec { x } ) \right) \right]\tag{6}
$$

where, for ${ \vec { b } } = { \vec { b } } _ { \mathrm { r e } } + i { \vec { b } } _ { \mathrm { i m } }$ and $W = W _ { \mathrm { r e } } + i W _ { \mathrm { i m } } ,$

$$
\begin{array} { r l r } { \vec { \theta } _ { \mathrm { r e } } } & { = } & { \log _ { \mathbf { 0 } _ { \mathcal { L } } } \left[ \exp _ { \mathbf { 0 } _ { \mathcal { L } } } \left( \vec { b } _ { \mathrm { r e } } \right) \oplus _ { \mathcal { L } } \left( W _ { \mathrm { r e } } \otimes _ { \mathcal { L } } \exp _ { \mathbf { 0 } _ { \mathcal { L } } } ( \vec { x } ) \right) \right] } \\ { \vec { \theta } _ { \mathrm { i m } } } & { = } & { \log _ { \mathbf { 0 } _ { \mathcal { L } } } \left[ \exp _ { \mathbf { 0 } _ { \mathcal { L } } } \left( \vec { b } _ { \mathrm { i m } } \right) \oplus _ { \mathcal { L } } \left( W _ { \mathrm { i m } } \otimes _ { \mathcal { L } } \exp _ { \mathbf { 0 } _ { \mathcal { L } } } ( \vec { x } ) \right) \right] } \end{array}\tag{7}
$$

Given the defining equations of the RBM NQS $\left( \mathrm { E q . 4 } \right)$ and the HRBM NQS (Eq.7), we note that the two NQS architectures have the exact same number of learnable parameters, the complex-valued bias vectors $\vec { a } , \vec { b }$ and the weight matrix W. The major diference between HRBM and RBM is the underlying geometry of the manifold on which mathematic operations such as matrix multiplication, bias addition and non-linear activation are carried out. As shown from our previous hyperbolic NQS constructions [14], [15], [16], it is this exact geometry that is largely responsible for the diference in the expressive capacity of hyperbolic and Euclidean NQS ansatz¨e, since hyperbolic space with its exponential volume growth is capable of better representing hierarchical, tree-like systems, as opposed to Euclidean space with its polynomial volume growth. On the other hand, to ensure numerical stability during the network training process, [15] introduces the spatial constraint hyperparameter, $L _ { m a x } { } ^ { 1 }$ , for Lorentz-type hyperbolic NQS ansatz¨e, which will be used for the HRBM constructed in this work. It must be noted that $L _ { m a x }$ is not a trainable parameter but a hyperparameter (much like the learning rate) that should be chosen carefully to ensure optimal performances of the HRBM NQS.

## 3 QSK VMC Experiments

In this section, we describe the VMC experiment results involving the HRBM and RBM NQS applied to the QSK system, which describes a quantum spin glass model with random long-range interactions [18], [19]. The Hamiltonian is given by

$$
H = - \frac { 1 } { \sqrt { N } } \sum _ { i < j } J _ { i j } \sigma _ { i } ^ { z } \sigma _ { j } ^ { z } - \Gamma \sum _ { i } \sigma _ { i } ^ { x }\tag{8}
$$

where $\sigma _ { x , z }$ are the Pauli $x , z$ matrices, $J _ { i j }$ is the random spin glass coupling and Γ is the transverse magnetic field along the x-axis. Note that the random all-to-all couplings $J _ { i j }$ are normally distributed numbers with zero mean and they are responsible for the volume-law entanglement property in the ground state of the QSK model. At a fixed system size N, choosing a diferent random initialization seed for the model leads to a diferent set of $J _ { i j }$ coeficients, hence a diferent realization of the QSK model at the same size N.

## 3.1 VMC experiment settings

The VMC experiments in this work are carried out for six diferent system sizes $N = 1 4 , 1 6 , 1 8 , 2 0 , 2 2 , 2 4$ , spaning an increase of $2 ^ { 1 0 } = 5 1 2$ folds in the Hilbert size dimension, using two types of NQS ansatzes: HRBM and RBM (see Table 1 for the exact number of parameters). For each N, six diferent QSK random seeds are used corresponding to the six diferent sets of $J _ { i j }$ couplings. For each fixed QSK seed, five diferent NQS seeds are used corresponding to the diferent random initial configurations of NQS trainings. In total, for each $N ,$ 60 VMC experiment runs are performed corresponding to the two types of NQS, each using five diferent seeds in six diferent realizations of the QSK model. The exact ground state energy and $S _ { 2 } / S _ { v N }$ entropies of all QSK models considered are listed in Table 4. It is interesting to note that the number of RBM/HRBM parameters

<table><tr><td>N</td><td>RBM HRBM</td></tr><tr><td>14 224</td><td>224  $\overline { { ( L _ { m a x } = 1 0 ) } }$ </td></tr><tr><td>16 288</td><td>288  $( L _ { m a x } = 1 0 )$ </td></tr><tr><td>18 360</td><td>360  $( L _ { m a x } = 1 0 )$ </td></tr><tr><td>20 440</td><td>440  $( L _ { m a x } = 1 0 )$ </td></tr><tr><td>22 528 528</td><td> $( L _ { m a x } = 1 0 )$ </td></tr><tr><td>24 624</td><td>624  $( L _ { m a x } = 2 0 )$ </td></tr></table>

Table 1: This table lists the diferent QSK system size N considered and the corresponding number of parameters for the RBM/HRBM NQS ansatzes. For a general α (the ratio of the hidden to visible units), the number of parameters for an RBM at QSK size N is given by $N + \alpha N + \alpha N ^ { 2 }$ . When $\alpha = 1$ , the number of parameters is simply $( 2 + N ) N$ . Note that both RBM and HRBM ansatzes use $\alpha = 1$ , and they have the exact same number of parameters at each N. As mentioned in the previous section, HRBM uses the additional hyperparameter $L _ { m a x }$ that places a spatial constraint on the vectors in hyperbolic space to ensure numerical stability. For $N = 1 4$ to $N = 2 2$ , an $L _ { m a x } = 1 0$ sufices to guarantee an optimal performance of HRBM, while for $N = 2 4$ , a larger $L _ { m a x } = 2 0$ is needed to ensure HRBM optimal performance.

(listed in Table 1) are very small compared to the dimensions of the QSK Hilbert space (ranging from $2 ^ { 1 4 }$ to $2 ^ { 2 4 } )$ , but as the results show later, this very modest range of NQS parameters is suficient to achieve the desired relative energy error that is smaller than $1 0 ^ { - 3 }$ . For both RBM and HRBM NQS, the total number of training epochs is set to be 450, with an early stopping mechanism built in to stop the training if an early convergence is reached or if the mean energy fails to improve within 100 epochs. Similary, the learning rate, initially set at 0.05 is adjusted by a factor of 0.5 whenever the training runs into a plateau for 40 epochs. In all VMC experiments, we use the NetKet sampler MetropolisLocal with the number of samples being 1008 for $1 4 \leq N \leq 2 0$ and 2016 for $2 2 \leq N \leq 2 4$ . As mentioned in the previous section and in [14], [15], [16], hyperbolic neural networks require a spatial constraint hyperparameter in order to perform stably. For all the HRBM NQS ansatzes in this work, from $N = 1 4$ to $N = 2 2$ , an $L _ { m a x } = 1 0$ sufices to guarantee an optimal performance of HRBM, while for $N = 2 4$ , a larger $L _ { m a x } = 2 0$ is needed to ensure HRBM optimal performance. This is because as N expands, quantum state vectors extend further toward the boundary of the hyperbolic manifold, so increasing $L _ { \mathrm { m a x } }$ at $N = 2 4$ expands the accessible hyperbolic domain while keeping numerical overflow bounded. The choice of which $L _ { m a x }$ to use for $N = 2 4$ is made by checking the performances of diferent HRBM ansatzes at diferent $L _ { m a x }$ values in the range [8,10,12,15,20,25] and choosing the best-performing one (see Tables 5 and 6 as well as Fig.14 in Section $\mathrm { A . 5 } ) ^ { 2 }$

Interestingly, despite the more complex construction of HRBM which requires hyperbolic mathematical operations (such as matrix exponentiation and Lorentzian additions $\oplus _ { \mathcal { L } }$ among others), thanks to the new jax implementation, there is no significant diference in the training time and computational overhead of HRBM compared to RBM, in direct constrast to our previous implementations that utilize Tensorflow and pytorch. For small system sizes up at $N = 2 0$ , the computations can be done on a modern machine (with around 8-16GB of RAM) with CPU alone in less than an hour, while for system sizes N = 22, 24, GPU computing is required.

## 3.2 Scaling study results

To benchmark HRBM NQS versus standard RBM NQS across system sizes $N ,$ we look at the various metrics that evaluate energy accuracy, state fidelity, and entanglement reconstruction such as the relative energy error $\epsilon ( N ) = \langle | E _ { \mathrm { N Q S } } - E _ { \mathrm { E D } } | / | E _ { \mathrm { E D } } | \rangle$ , energy variance $\sigma _ { E } ^ { 2 } ( N )$ , R´enyi- $\cdot 2 \ \langle S _ { 2 } \rangle$ and von Neumann $S _ { v N }$ entropies<sup>3</sup>, and absolute $S _ { 2 } ~ ( S _ { v N } )$ entropy error $| S _ { 2 } ^ { \mathrm { e r r } } | = | S _ { 2 } ^ { \mathrm { N Q S } } - \bar { S } _ { 2 } ^ { \mathrm { E x a c t } } | ( S _ { v N } ^ { \mathrm { e r r } } = | S _ { v N } ^ { \mathrm { N Q S } } - S _ { v N } ^ { \mathrm { E x a c t } } | )$

Our main results are shown in Tables 2, 3 and graphically in Fig.1.
<table><tr><td>N</td><td>NQS</td><td>Mean relative energy error</td><td>Mean variance</td></tr><tr><td>14</td><td>HRBM</td><td> $\overline { { 2 . 0 5 \times 1 0 ^ { - 4 } } }$ </td><td>0.022248</td></tr><tr><td rowspan="3"></td><td></td><td>(0.000030)</td><td>(0.003114)</td></tr><tr><td>RBM</td><td> $3 . 3 6 \times 1 0 ^ { - 4 }$ </td><td>0.031814</td></tr><tr><td></td><td> $( 0 . 0 0 0 0 7 7 )$ </td><td>(0.005671)</td></tr><tr><td rowspan="3">16</td><td>HRBM</td><td> $\overline { { 2 . 4 9 \times 1 0 ^ { - 4 } } }$ </td><td>0.029062</td></tr><tr><td></td><td>(0.000019)</td><td>(0.001046)</td></tr><tr><td>RBM</td><td> $3 . 4 3 \times 1 0 ^ { - 4 }$  (0.000039)</td><td>0.039285 (0.003938)</td></tr><tr><td rowspan="3">18</td><td>HRBM</td><td> $2 . 5 5 \times 1 0 ^ { - 4 }$ </td><td>0.036046</td></tr><tr><td></td><td>(0.000014)</td><td>(0.002325)</td></tr><tr><td>RBM</td><td> $3 . 6 7 \times 1 0 ^ { - 4 }$ </td><td>0.048461</td></tr><tr><td rowspan="3">20</td><td>HRBM</td><td>(0.000045)  $\overline { { 3 . 0 0 \times 1 0 ^ { - 4 } } }$ </td><td>(0.005313) 0.046668</td></tr><tr><td></td><td> $\left( 0 . 0 0 0 0 2 8 \right)$ </td><td>(0.004544)</td></tr><tr><td>RBM</td><td> $5 . 3 4 \times 1 0 ^ { - 4 }$ </td><td>0.075392</td></tr><tr><td rowspan="5">22</td><td>HRBM</td><td>(0.000080)</td><td>(0.010029)</td></tr><tr><td></td><td> $\overline { { 2 . 9 4 \times 1 0 ^ { - 4 } } }$ </td><td>0.051025</td></tr><tr><td>RBM</td><td> $\left( 0 . 0 0 0 0 1 3 \right)$ </td><td>(0.002307)</td></tr><tr><td></td><td> $4 . 5 8 \times 1 0 ^ { - 4 }$ </td><td>0.072427</td></tr><tr><td>HRBM</td><td>(0.000023)</td><td>(0.003932)</td></tr><tr><td rowspan="4">24</td><td></td><td> $2 . 7 8 \times 1 0 ^ { - 4 }$ </td><td>0.052951</td></tr><tr><td rowspan="3">RBM</td><td> $\left( 0 . 0 0 0 0 1 9 \right)$ </td><td>(0.003229)</td></tr><tr><td></td><td></td></tr><tr><td> $3 . 1 6 \times 1 0 ^ { - 4 }$   $\left( 0 . 0 0 0 0 1 1 \right)$ </td><td>0.057231 (0.001482)</td></tr></table>

Table 2: The mean relative energy error and mean variance of HRBM & RBM NQS, obtained by averaging over all NQS and QSK seeds for diferent QSK sizes N. For each NQS, the mean value is recorded in the first line followed by the standard error (in brackets) in the second line.

<table><tr><td>N</td><td>NQS</td><td>Mean  $S _ { v N }$ </td><td>Mean  $S _ { 2 }$ </td><td>Mean |Ser | </td><td>Mean |Serr</td></tr><tr><td>14</td><td>Exact</td><td>0.570010</td><td>0.275898</td><td></td><td></td></tr><tr><td rowspan="4"></td><td>HRBM</td><td>(0.039986) 0.565579</td><td>(0.027441) 0.274576</td><td></td><td>0.001552</td></tr><tr><td></td><td></td><td></td><td>0.004431</td><td></td></tr><tr><td></td><td>(0.040151)</td><td>(0.027650)</td><td>(0.001549)</td><td>(0.000946)</td></tr><tr><td>RBM</td><td>0.564904</td><td>0.273258</td><td>0.005134</td><td>0.002686</td></tr><tr><td>16</td><td>Exact</td><td>(0.040847) 0.602501</td><td>(0.027738) 0.293109</td><td>(0.001373)</td><td>(0.000752)</td></tr><tr><td rowspan="4"></td><td></td><td>(0.043697)</td><td>(0.034675)</td><td></td><td></td></tr><tr><td>HRBM</td><td>0.597331</td><td>0.290445</td><td>0.005349</td><td>0.003338</td></tr><tr><td></td><td>(0.042822)</td><td>(0.033513)</td><td>(0.001840)</td><td>(0.002035)</td></tr><tr><td>RBM</td><td>0.597042</td><td>0.290364</td><td>0.005459</td><td>0.002745</td></tr><tr><td>18</td><td>Exact</td><td>(0.043476) 0.685606</td><td>(0.034450) 0.334836</td><td>(0.000703)</td><td>(0.000371)</td></tr><tr><td></td><td>HRBM</td><td>(0.032044) 0.679772</td><td>(0.035352) 0.332130</td><td>0.005834</td><td>0.002857</td></tr><tr><td></td><td>RBM</td><td>(0.031794) 0.679921</td><td>(0.034433) 0.331672</td><td>(0.000617) 0.005685</td><td>(0.000914) 0.003177</td></tr><tr><td>20</td><td>Exact</td><td>(0.032087) 0.776061</td><td>(0.035286) 0.378016</td><td>(0.000822)</td><td>(0.000580)</td></tr><tr><td></td><td>HRBM</td><td>(0.050514) 0.769457</td><td>(0.038485) 0.375597</td><td>0.006604</td><td>0.002567</td></tr><tr><td>22</td><td>RBM</td><td>(0.050001) 0.767773 (0.051167)</td><td>(0.038126) 0.373303</td><td>(0.001182) 0.008288</td><td>(0.000884) 0.004712</td></tr><tr><td></td><td>Exact</td><td>0.872736 (0.039558)</td><td>(0.039078) 0.421106</td><td>(0.001496)</td><td>(0.001109)</td></tr><tr><td></td><td>HRBM</td><td>0.867364</td><td>(0.029092) 0.420541</td><td>0.005372</td><td>0.001134</td></tr><tr><td></td><td></td><td>(0.039512)</td><td>(0.029421)</td><td>(0.000381)</td><td>(0.000241)</td></tr><tr><td></td><td>RBM</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>0.415225</td><td></td><td></td></tr><tr><td></td><td></td><td>0.863283</td><td></td><td>0.009453</td><td>0.005880</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>(0.039009)</td><td>(0.028383)</td><td>(0.001105)</td><td>(0.000950)</td></tr><tr><td>24</td><td>Exact</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>0.958293</td><td>0.494072</td><td></td><td></td></tr><tr><td></td><td></td><td>(0.056262)</td><td>(0.053739)</td><td></td><td></td></tr><tr><td></td><td>HRBM</td><td>0.951050</td><td>0.491859</td><td>0.007243</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>0.002222</td></tr><tr><td></td><td></td><td>(0.056044)</td><td>(0.053728)</td><td>(0.000950)</td><td>(0.000393)</td></tr><tr><td></td><td>RBM</td><td>0.950577</td><td>0.489406</td><td>0.007716</td><td>0.004665</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>(0.056263)</td><td>(0.053833)</td><td>(0.000385)</td><td>(0.000159)</td></tr></table>

Table 3: This table lists the mean R´enyi-2 entropy $S _ { 2 } .$ , the von Neumann entropy $S _ { v N }$ as well as the absolute entropy errors $\vert S _ { v N } ^ { \mathrm { e r r } } \vert , \vert S _ { 2 } ^ { \mathrm { e r r } } \vert$ of HRBM & RBM NQS, obtained by averaging over all NQS and QSK seeds for diferent QSK sizes N. The exact values are the mean over QSK seeds of the exact S<sub>2</sub> and $S _ { v N }$ from the exact wavefunction obtained by ED. For each NQS, the mean value is recorded in the first line followed by the standard error (in brackets) in the second line. For the exact wavefunction, there is no $| S _ { 2 } ^ { \mathrm { e r r } } |$ , |S<sup>err</sup><sub>vN</sub>| (where the corresponding entries are filled with $\mathrm { ~ a ~ } ^ { \cdot } - \ ' )$

![](images/bbf69c9150fc1c537773c11fde3871474a9248f81a35cab74d0483581d274ec5.jpg)  
Figure 1: The performance of HRBM NQS versus RBM NQS in the QSK system with increasing sizes from $N = 1 4$ to $N = 2 4$ . From left to right, top to bottom, the various plots show the scaling behaviors of diferent metrics for RBM and HRBM. Top row, first subfigure from left: Relative energy error scaling (ϵ vs N): Log-linear plot showing $\langle | E _ { \mathrm { N Q S } } - E _ { \mathrm { E D } } | / | E _ { \mathrm { E D } } | \rangle$ . This demonstrates how variational expressivity scales with Hilbert space dimension $2 ^ { N }$ . While RBM exhibits a non-monotonic optimization trajectory with a spike at $N = 2 0$ , HRBM displays a smooth, monotonic optimization trajectory across all QSK system sizes. Top row, second subfigure from left: R´enyi-2 $\langle S _ { 2 } \rangle$ entanglement entropy vs $N :$ Linear plot comparing disorder-averaged $\mathrm { N Q S } \ S _ { 2 }$ entropy against the exact $S _ { 2 }$ obtained by ED. This shows how each ansatz tracks physical volume-law growth. Top row, third subfigure from left: von-Neumann $\langle S _ { \mathrm { v N } } \rangle$ entanglement entropy vs $N ;$ : Linear plot comparing disorderaveraged NQS entropies against the exact $S _ { v N }$ obtained by ED. This shows how each ansatz tracks physical volume-law growth. Bottom row, first subfigure from left: Energy variance scaling $( \sigma _ { E } ^ { 2 } \ \mathrm { v s } \ N )$ : Linear plot of $\langle \langle H ^ { 2 } \rangle - \langle H \rangle ^ { \bar { 2 } } \rangle$ . This evaluates how closely each NQS architecture approaches the true exact Hamiltonian ground state wavefunction (where $\sigma _ { E } ^ { 2 } = 0 )$ as $N$ grows. Bottom row, second (third) subfigures from left: Absolute $S _ { 2 }$ $( S _ { v N } )$ entropy error $| S _ { 2 } ^ { \mathrm { N Q S } } - S _ { \mathrm { E x a c t } } | \ : \left( | S _ { v N } ^ { \mathrm { N Q S } } - S _ { \mathrm { E x a c t } } | \right)$ : Log-linear plot. This highlights the exact entropy gap divergence as a function of system size. The error bars capture both the disorder variation stemming from the variance between diferent physical realizations of the random couplings $J _ { i j }$ across the QSK Hamiltonian seeds, as well as the NQS optimization stochastics originating from the variance in final network convergence due to the random weight initializations across the diferent NQS seeds. The scaling results at individual QSK seeds corresponding to diferent QSK model realizations are included in the Appendix, section A.3.

Across all system sizes $( N = 1 4  2 4 )$ , the scaling benchmark demonstrates a clear separation between energy optimization, $S _ { 2 }$ and $S _ { v N }$ entanglement entropies, and fine-grained quantum state reconstruction. While both NQS architectures track macroscopic entanglement trends, HRBM systematically suppresses error growth as the Hilbert space dimension expands from $2 ^ { 1 4 }$ to $2 ^ { 2 4 }$ . In particular, the following observations can be made regarding the key metrics.

• In terms of the ground state energy optimization (ϵ & $\sigma _ { E } ^ { 2 } )$ :

Relative energy error ϵ: HRBM maintains systematic superiority across all system sizes. Standard Euclidean RBM sufers from non-monotonic optimization spikes, starting from $3 . 3 6 \times 1 0 ^ { - 4 } \mathrm { a t } N = 1 4$ and peaking at $5 . 3 4 \times 1 0 ^ { - 4 }$ at $N = 2 0$ . On the other hand, HRBM displays smooth, bounded behavior $( \varepsilon \leq 3 . 0 \times 1 0 ^ { - 4 } )$ . At $N = 2 4$ , HRBM reaches $\varepsilon = 2 . 7 8 \times 1 0 ^ { - 4 }$ compared to RBM’s $3 . 1 6 \times 1 0 ^ { - 4 }$

– Energy variance $\sigma _ { E } ^ { 2 }$ : HRBM exhibits lower local energy variance across the entire scaling range. While RBM variance fluctuates non-monotonically (peaking at $\sigma ^ { 2 } \approx 0 . 0 7 5$ at $N = 2 0 )$ , HRBM scales smoothly from ≈ 0.022 (N = 14) to ≈ 0.053 (N = 24), demonstrating greater physical stability against disorder.

• In terms of the macroscopic entanglement entropies $( S _ { 2 } \ \& \ S _ { v N } )$ : Both HRBM and RBM display nearperfect overlap with the exact ground state across all system sizes, with HRBM visibly maintaining a closer overlap with the exact state at $1 8 \leq N \leq 2 4$ . Despite this, it must be noted that the $S _ { 2 }$ and $S _ { v N }$ scalar entropies are heavily weighted by the few largest eigenvalues $\left( \lambda _ { i } > 1 0 ^ { - 4 } \right)$ , so the macroscopic plots of $S _ { 2 } \& S _ { v N }$ tend to mask underlying NQS architectural diferences in representing high-rank correlations.

• In terms of the absolute entanglement error scaling $\left( \vert S _ { 2 } ^ { \mathrm { e r r } } \vert ~ \& ~ \vert S _ { v N } ^ { \mathrm { e r r } } \vert \right)$ : HRBM achieves lower absolute reconstruction errors. $\mathrm { A t } \ N = 2 4$ , HRBM reduces the R´enyi-2 entropy error $( | S _ { 2 } - S _ { 2 , \mathrm { e x a c t } } | ) \mathrm { t o } \approx 2 . 2 { \times } 1 0 ^ { - 3 }$ compared to ≈ $4 . 7 \times 1 0 ^ { - 3 }$ for standard RBM.

Addtionally, in terms of the individual QSK disorder realizations (see Figs.3-8 in Section A.3), we note the following trends.

• Standard RBM frequently encounters local optimization traps in specific disorder realizations at intermediate system sizes $( \mathrm { e . g . , } N = 2 0 $ in seeds 1234 and 6666). HRBM suppresses these severe error spikes. This observation might indicate that the primary failure mode of standard RBM on disordered spin glasses is getting stuck in high-energy local minima at intermediate system sizes while hyperbolic embedding stabilizes training and prevents these spikes, making it a more reliable ansatz when scaling into regimes where exact ground states are unavailable.

• In instances like Seed 5678, where the exact ground state entanglement non-monotonically spikes at $N = 1 6$ and dips at $N = 1 8$ , HRBM captures these physics features without losing numerical accuracy.

• Seed 9999 represents an outlier where RBM achieves slightly lower relative energy error than HRBM at $N = 2 4$ . However, aggregated across all disorder realizations, HRBM dominates.

## 3.3 Full entanglement spectrum results

In the previous section, we analyze scalar metrics like energy and integrated entropies, but these metrics do not provide the full picture as macroscopic scalar entropies like $S _ { 2 }$ and $S _ { v N }$ mask critical model deficiencies given that they are overwhelmingly dominated by the top few eigenvalues of the reduced density matrix $\rho$ (where $\lambda _ { i } > 1 0 ^ { - 4 } )$ , so a model can achieve respectable relative energy errors while failing to capture the true quantum state structure. On the other hand, the full entanglement spectrum $( \lambda _ { i } )$ - in particular the spectral tail decay - is much more informative in terms of identifying the true expressivity of an NQS model. As such, in this section, we discuss the full entanglement spectrum $\lambda _ { i }$ results of the VMC experiments done in order to comprehensively compare the performance of HRBM versus RBM NQS.

The plot showing the full entanglement spectrum for $N = 2 4$ is included in Fig.2, while those for $N = 1 4$ to $N = 2 2$ are included in $\mathrm { F i g . 9 - F i g . 1 3 }$ in the Appendix section A.4.

![](images/ed667c7c93fda6ad648e9f6c822d83b20820090cb5100394dc853511c502e3e0.jpg)  
Figure 2: Full entanglement spectrum of RBM and HRBM NQS in QSK VMC setting with N = 24 at diferent QSK model random seeds. In each subplot, the RBM and HRBM lines denote the mean value obtained by averaging over diferent VMC runs involving diferent NQS random seeds, with shading indicating one standard deviation.

Based on Figs.2-13, the following observations can be made:

• With respect to the top eigenvalues (where $1 0 ^ { - 6 } < \lambda _ { i } < 1 0 ^ { 0 } )$ : Both RBM and HRBM match the exact GS curve (black) almost perfectly across all system sizes and disorder seeds. Because $S _ { 2 }$ and $S _ { v N }$ are heavily weighted by these largest eigenvalues $( \lambda _ { i } ^ { 2 }$ for $S _ { 2 }$ and $- \lambda _ { i }$ log $\lambda _ { i }$ for $\boldsymbol { S _ { v N } } )$ , both models appear to coincide with the exact result on macro-level plots.

• With respect to the sub-dominant tail (where eigenvalues $\lambda _ { i } < 1 0 ^ { - 8 } )$ , the true representation bottleneck of Euclidean space is revealed.

– RBM (the red dashed line) systematically overestimates the tail eigenvalues across virtually every QSK seed and size. The RBM line visibly drifts above the exact curve, creating a “fat tail” that accumulates spurious weight in high-index Schmidt modes. Because flat Euclidean space expands polynomially, standard RBM lacks the geometric capacity to enforce sharp eigenmode truncation during VMC optimization, which probably leads to the over-parameterization of sub-dominant modes.

HRBM (the blue dash-dotted line) faithfully tracks the steep decay of the exact ground state curve (in black) down to the numerical noise floor $( \sim 1 0 ^ { - 1 5 } )$ , showing almost no upward drift for most of the QSK disorder realizations<sup>4</sup>, thanks to hyperbolic geometry’s exponential volume matching the

hierarchical decay of the entanglement spectrum.

• With respect to the scaling property with system size N: As N increases from 14 to 24 (where the number of eigenmodes grows to beyond 4000), RBM’s spectral deviation in the tail becomes progressively wider, while HRBM stays tightly bound to the exact curve. This shows that HRBM ansatz genuinely captures the underlying quantum state topology.

Given the results of this proof-of-concept work, it would be interesting to scale the QSK system up beyond $N = 2 4$ and carry out the same study done in this work. However, at $N = 2 6$ , exact diagonalization becomes much harder than at $N = 2 4$ , given the doubling of the Hilbert space dimension, so for $N \geq 2 6$ , alternative computational methods are required to provide a reliable benchmark to test the performances of the NQS. Also, with our current computing resources, while it is not feasible to carry out the computations for the $N = 2 6$ case, it is natural to wonder whether, at $N = 2 6$ , HRBM would continue to outperform RBM, given the trends shown in Fig.1, where RBM displays a non-monotonic optimization trajectory with an error spike at $N = 2 0$ whereas HRBM displays a monotonic trajectory. Could this mean that at $N = 2 6$ , RBM would outperform HRBM in terms of relative energy error, energy variance and reconstruction errors? To answer this question (speculatively), we rely on the following observations. First, the RBM non-monotonicity is characteristic of optimization instability in flat geometry: As N scales, Euclidean landscapes develop complex local minima traps where optimization success varies unpredictably depending on system size and disorder realization. On the other hand, the smooth scaling and strict monotonicity of HRBM indicates that hyperbolic geometry provides a consistent, predictable representation capacity that does not sufer from geometric optimization bottlenecks as Hilbert space expands. Second, and more crucially, even when RBM approaches HRBM in the scalar metrics like energy and $S _ { 2 }$ or $S _ { v N }$ entropies at $N = 2 4$ , it still fails to capture non-local quantum correlations, maintaining an unphysical ‘fat tail’ (where $\lambda _ { i } < 1 0 ^ { - 8 } )$ in its entanglement spectrum for all QSK realizations at all sizes, in direct constrast to HRBM’s ability to reconstruct the entire spectrum down to $\lambda _ { i } \sim 1 0 ^ { - 1 5 }$ . This means that even at $N = 2 6$ or beyond, HRBM would likely emerge as the better NQS variant to represent the QSK volumelaw ground state wavefunction, given its capacity to faithfully match the hierarchical entanglement spectrum decay.

## 4 Concluding remarks

In this work, we introduce the hyperbolic RBM (HRBM) NQS and show that it is capable of outperforming the Euclidean RBM NQS in the volume-law QSK system representing quantum spin glasses. In a scaling study where both architectures were evaluated under identical polynomial parameter budgets $( O ( \mathrm { p o l y } ( N ) ) )$ , we track the performances of the RBM and HRBM NQS across an expansion of $2 ^ { 1 0 } = 5 1 2$ folds in the Hilbert space dimension for QSK models whose size N ranges from $N = 1 4$ to $N = 2 4$ . These small sizes are chosen because exact diagonalization is feasible computationally and provides an exact benchmark against which to compare the performances of the RBM and HRBM NQS.

In terms of both the reachable ground state energy and entanglement entropies (including the $S _ { 2 }$ and von-Neumann $S _ { v N } )$ , HRBM outperforms RBM across all QSK size $\bar { N }$ as shown in Fig.1. While standard Euclidean RBM displays a non-monotonic optimization trajectory as N scales, HRBM achieves asymptotic convergence across both energy and entropy metrics. Even more interestingly, HRBM also demonstrates its ability to faithfully capture the full entanglement spectrum of the QSK model across 15 orders of magnitudes across all system sizes, while RBM visibly overestimates the spectrum for sub-dominant modes (see Figs.2-13). This is very likely due to the fundamental diferences in the geometry underlying the two types of NQS. Since Euclidean latent space grows only polynomially, standard Euclidean RBM lacks the geometric capacity to enforce sharp, high-order entanglement mode decay. On the other hand, the fact that quantum entanglement spectra decay exponentially $( \lambda _ { i } \sim e ^ { - \alpha i }$ for some proportionality constant α) means that the exponential volume expansion of hyperbolic space naturally mirrors this hierarchical entanglement eigenvalue decomposition, allowing HRBM to represent thousands of sub-dominant modes without parameter saturation or spectral leakage. This is arguably the strongest supporting evidence for our hypothesis that hyperbolic NQS (including the HRBM considered in this work as well as other types of yet-to-be-constructed hyperbolic NQS ansatzes) might be a genuinely better match than Euclidean NQS for volume-law quantum systems thanks to their superior expressivity. Furthermore, we note that an interesting byproduct of the results of this work is the numerical proof showing that in order to maintain a relative energy error of less than $1 0 ^ { - 3 } .$ , only polynomial scaling of the RBM/HRBM NQS is required<sup>5</sup> when the Hilbert space dimension of the volume-law QSK system expands exponentially from $2 ^ { 1 4 }$ to $2 ^ { 2 4 }$ states.

Given the results of this proof-of-concept work that successfully demonstrate the potential of non-autoregressive hyperbolic RBM NQS to achieve high physical fidelity for volume-law states without requiring exponential parameter expansion, it would be instructive to construct new types of hyperbolic NQS such as hyperbolic transformer NQS or hyperbolic graph neural network NQS and study these new constructions in the context of volume-law quantum systems. Another promising future direction might involve combining the HRBM NQS constructed in this work with a Slater determinant to tackle the highly challenging volume-law fermionic SYK model. More concretely, in fermionic systems, the Slater determinant enforces the required anti-symmetry under particle exchange, while the HRBM network acts as an eficient non-local correlation backflow factor. Ideally, pairing the two should leverage the geometric eficiency of hyperbolic space to capture strong, dense electron correlations that standard Euclidean neural-network backflow factors struggle to represent without exponential parameters. We hope to return to these in future works.

## A Appendix

## A.1 Hyperbolic Lorentzian mathematical operations

The n-dimensional Lorentz hyperboloid $\mathbb { H } ^ { n }$ with constant negative curvature −k is given as

$$
\mathbb { H } ^ { n } \equiv \left\{ \mathbf { x } \in \mathbb { R } ^ { n + 1 } : \langle \mathbf { x } , \mathbf { x } \rangle _ { \mathcal { L } } = - k , x _ { 0 } > 0 \right\}\tag{10}
$$

where $\langle \mathbf { x } , \mathbf { y } \rangle _ { \mathcal { L } }$ , the Lorentzian scalar product for $( n + 1 )$ -dimensional vectors $\mathbf { x } , \mathbf { y } \in \mathbb { R } ^ { n + 1 }$ , is defined as

$$
\langle \mathbf { x } , \mathbf { y } \rangle _ { \mathcal { L } } \equiv - x _ { 0 } y _ { 0 } + \sum _ { i = 1 } ^ { n } x _ { i } y _ { i } .\tag{11}
$$

Note that in the Lorentz model of hyperbolic space, the origin is the point $\mathbf { 0 } _ { \mathcal { L } } = ( 1 , 0 , 0 , \dots , 0 )$ . Throughout this work, we fix $k = 1$ . All equations that follow has $k = 1$

The distance function between two points $\mathbf { x } , \mathbf { y } \in \mathbb { H } ^ { n }$ is

$$
d _ { \mathbb { H } } ( \mathbf { x } , \mathbf { y } ) = \operatorname { a r c o s h } \left( - \langle \mathbf { x } , \mathbf { y } \rangle _ { \mathcal { L } } \right) ,\tag{12}
$$

while the Lorentzian norm of a vector v is

$$
\| \mathbf { v } \| _ { \mathcal { L } } = \sqrt { \langle \mathbf { v } , \mathbf { v } \rangle _ { \mathcal { L } } } .\tag{13}
$$

The tangent space at x is the n-dimensional Euclidean vector space approximating $\mathbb { H } ^ { n }$ around x:

$$
\mathcal { T } _ { x } \mathbb { H } ^ { n } \equiv \left\{ \mathbf { x } \in \mathbb { R } ^ { n + 1 } : \langle \mathbf { v } , \mathbf { x } \rangle _ { \mathcal { L } } = 0 \right\}\tag{14}
$$

Mappings between the Lorentz hyperboloid and its tangent space are done using the exponential $\exp _ { \mathbf { x } } ( \mathbf { v } )$ and logarithmic $\log _ { \mathbf { x } } ( \mathbf { y } )$ maps.

$$
\mathcal { T } _ { \mathbf { x } } \mathbb { H } ^ { n } \to \mathbb { H } ^ { n } : \qquad \exp _ { \mathbf { x } } ( \mathbf { v } ) = \cosh \left( | | \mathbf { v } | | _ { \mathcal { L } } \right) \mathbf { x } + \sinh \left( | | \mathbf { v } | | _ { \mathcal { L } } \right) \frac { \mathbf { v } } { | | \mathbf { v } | | _ { \mathcal { L } } }\tag{15}
$$

$$
\mathbb { H } ^ { n } \to { \mathcal { T } } _ { \mathbf { x } } \mathbb { H } ^ { n } : \qquad \log _ { \mathbf { x } } ( \mathbf { y } ) = d _ { \mathbb { H } } ( \mathbf { x } , \mathbf { y } ) { \frac { \mathbf { y } + \langle \mathbf { x } , \mathbf { y } \rangle _ { \mathcal { L } } \mathbf { x } } { | | \mathbf { y } + \langle \mathbf { x } , \mathbf { y } \rangle _ { \mathcal { L } } \mathbf { x } | | _ { \mathcal { L } } } }\tag{16}
$$

The parallel transport operation that maps a point $\mathbf { z } \in \mathcal { T } _ { \mathbf { x } } \mathbb { H } ^ { n }$ to a point in $\mathcal { T } _ { \mathbf { y } } \mathbb { H } ^ { n }$ is defined as

$$
P _ { \mathbf { x }  \mathbf { y } } ( \mathbf { z } ) = \mathbf { z } + \frac { \langle \mathbf { y } , \mathbf { z } \rangle _ { \mathcal { L } } } { 1 - \langle ( \mathbf { x } , \mathbf { y } ) \rangle _ { \mathcal { L } } } ( \mathbf { x } + \mathbf { y } )\tag{17}
$$

When $\mathbf { x } = \mathbf { 0 } _ { \mathcal { L } }$

$$
P _ { \mathbf { 0 } _ { \mathcal { L } }  \mathbf { y } } ( \mathbf { z } ) = \mathbf { z } + \frac { \langle \mathbf { y } , \mathbf { z } \rangle _ { \mathcal { L } } } { 1 - \langle ( \mathbf { 0 } _ { \mathcal { L } } , \mathbf { y } ) \rangle _ { \mathcal { L } } } ( \mathbf { 0 } _ { \mathcal { L } } + \mathbf { y } )\tag{18}
$$

The various Lorentz mathematical operations are defined in terms of the exponential/logarithm mappings at $\mathbf { x } = \mathbf { 0 } _ { \mathcal { L } }$ in (15), (16) and parallel transport operations given in (18) as follows.

• For $\mathbf { x } , \mathbf { y } \in \mathbb { H } ^ { n }$ , the Lorentz addition $\oplus _ { \mathcal { L } }$ is defined as:

Lorentz addition :

$$
\mathbf { x } \oplus _ { \mathcal { L } } \mathbf { y } = \exp _ { \mathbf { x } } [ P _ { \mathbf { 0 } _ { \mathcal { L } }  \mathbf { x } } ( \log _ { \mathbf { 0 } _ { \mathcal { L } } } ( \mathbf { y } ) ) ]\tag{19}
$$

• The scalar multiplication $\odot { c }$ between a point $\mathbf { x } \in \mathbb { H } ^ { n }$ and $r \in \mathbb { R }$ is

$$
\mathrm { L o r e n t z ~ s c a l a r ~ m u l t i p l i c a t i o n : } \qquad r \odot _ { \mathcal { L } } \mathbf { x } = \exp _ { \mathbf { 0 } _ { \mathcal { L } } } \left[ r \log _ { \mathbf { 0 } _ { \mathcal { L } } } ( \mathbf { x } ) \right]\tag{20}
$$

• The matrix multiplication $\otimes _ { \mathcal { L } }$ between a point $\mathbf { x } \in \mathbb { H } ^ { n }$ and a matrix $M \in \mathbb { R } ^ { n } \times \mathbb { R } ^ { n }$ is

$$
M \otimes _ { \mathcal { L } } \mathbf { x } = \exp _ { \mathbf { 0 } _ { \mathcal { L } } } \left[ M \log _ { \mathbf { 0 } _ { \mathcal { L } } } ( \mathbf { x } ) \right]\tag{21}
$$

• The nonlinear activation $f ^ { \otimes _ { C } } ( \mathbf { x } )$ where $x \in \mathbb { H } ^ { n }$ is

$$
f ^ { { \otimes } _ { \mathcal { L } } } = \exp _ { \mathbf { 0 } _ { \mathcal { L } } } \left( f ( \log _ { \mathbf { 0 } _ { \mathcal { L } } } ( x ) ) \right)\tag{22}
$$

## A.2 QSK models: Exact energy, $S _ { 2 }$ and $S _ { v N }$ entropies

<table><tr><td> $\overline { { \mathrm { ~ N ~ } } }$ </td><td>QSK seed</td><td>Exact Energy</td><td>Exact  $\overline { { S _ { 2 } } }$ </td><td>Exact  $\overline { { S _ { v N } } }$ </td></tr><tr><td>14</td><td>1234</td><td>-14.764654</td><td>0.238397</td><td>0.523197</td></tr><tr><td></td><td>4444</td><td>-14.762866</td><td>0.241584</td><td>0.527381</td></tr><tr><td></td><td>5678</td><td>-14.784242</td><td>0.249586</td><td>0.541275</td></tr><tr><td></td><td>6666</td><td>-14.884761</td><td>0.300660</td><td>0.612231</td></tr><tr><td></td><td>9012</td><td>-14.777944</td><td>0.223110</td><td>0.469309</td></tr><tr><td>=</td><td>9999</td><td>-15.419419</td><td>0.402053</td><td>0.746667</td></tr><tr><td>16</td><td>1234</td><td>-16.992467</td><td>0.245180</td><td>0.553498</td></tr><tr><td></td><td>4444</td><td>-16.946620</td><td>0.225920</td><td>0.529474</td></tr><tr><td></td><td>5678</td><td>-17.101175</td><td>0.406380</td><td>0.703491</td></tr><tr><td></td><td>6666</td><td>-17.079820</td><td>0.257172</td><td>0.544865</td></tr><tr><td></td><td>9012</td><td>-16.823153</td><td>0.227077</td><td>0.513519</td></tr><tr><td>-</td><td>9999</td><td>-17.128633</td><td>0.396925</td><td>0.770161</td></tr><tr><td>18</td><td>1234</td><td>-18.989854</td><td>0.254455</td><td>0.593687</td></tr><tr><td></td><td>4444</td><td>-19.142463</td><td>0.361591</td><td>0.718623</td></tr><tr><td></td><td>5678</td><td>-18.995777</td><td>0.283652</td><td>0.655164</td></tr><tr><td></td><td>6666</td><td>-19.192267</td><td>0.324922</td><td>0.665972</td></tr><tr><td></td><td>9012</td><td>-18.976817</td><td>0.289690</td><td>0.656387</td></tr><tr><td>=</td><td>9999</td><td>-19.729642</td><td>0.494707</td><td>0.823804</td></tr><tr><td>20</td><td>1234</td><td>-21.166380</td><td>0.319844</td><td>0.714197</td></tr><tr><td></td><td>4444</td><td>-21.069415</td><td>0.282516</td><td>0.652132</td></tr><tr><td></td><td>5678</td><td>-21.245930</td><td>0.317694</td><td>0.672997</td></tr><tr><td></td><td>6666</td><td>-21.450500</td><td>0.487732</td><td>0.909258</td></tr><tr><td></td><td>9012</td><td>-21.316428</td><td>0.356613</td><td>0.762007</td></tr><tr><td></td><td>9999</td><td>-21.842648</td><td>0.503695</td><td>0.945772</td></tr><tr><td>22</td><td>1234</td><td>-23.306757</td><td>0.340331</td><td>0.750125</td></tr><tr><td></td><td>4444</td><td>-23.524750</td><td>0.542537</td><td>1.042305</td></tr><tr><td></td><td>5678</td><td>-23.227276</td><td>0.389616</td><td>0.840151</td></tr><tr><td></td><td>6666</td><td>-23.375017</td><td>0.374663</td><td>0.831556</td></tr><tr><td></td><td>9012</td><td>-23.425041</td><td>0.427985</td><td>0.881539</td></tr><tr><td>=</td><td>9999</td><td>-23.806345</td><td>0.451503</td><td>0.890738</td></tr><tr><td>24</td><td>1234</td><td>-25.424855</td><td>0.457565</td><td>0.929380</td></tr><tr><td></td><td>4444</td><td>-25.582844</td><td>0.473011</td><td>0.955019</td></tr><tr><td></td><td>5678</td><td>-25.342593</td><td>0.305626</td><td>0.710423</td></tr><tr><td></td><td>6666</td><td>-25.910862</td><td>0.539969</td><td>1.021905</td></tr><tr><td></td><td>9012</td><td>-26.044812</td><td>0.710787</td><td>1.117530</td></tr><tr><td></td><td>9999</td><td>-25.562186</td><td>0.477472</td><td>1.015500</td></tr></table>

Table 4: Exact ground state energy (obtained by exact diagonalization) and exact $S _ { 2 }$ , $S _ { v N }$ entropies of the QSK models considered.

## A.3 Scaling results at diferent QSK realizations

![](images/fe3684c7a173474ac4dd14383fc3cc590638ea22edb2f83bcf14690ba441b95c.jpg)  
Figure 3: Scaling benchmark (QSK seed=1234) for RBM and HRBM in the QSK setting as the system size N increases from $N = 1 4$ to N = 22. The error bars are obtained by averaging over diferent NQS random seeds.

![](images/4a909987e24dd3eab6882a39d1911d7b342ece5519b4317f7dd8b3a8010bbc13.jpg)

![](images/783781106eff22fbcc5e23620932f82205f0ae526dd6d54a6222bd99b33b4a9a.jpg)

![](images/417b96b68c80b0660e743b116fdcddee9422b67f8e18c2342b1e30d0ca79bcf3.jpg)

![](images/4eca6049a3a0393651f27b81e37760cd0b604d6178e418ea219610849190ac3c.jpg)

![](images/c90c3d9d338a92ae4a94a67cb9137df3bf81900523dece70fae5692fbdda9987.jpg)

![](images/f87d581e07827976657b2d13c580af308027243ed3f9880c464c31341e5e902e.jpg)  
Figure 4: Scaling benchmark (QSK seed=4444) for RBM and HRBM in the QSK setting as the system size N increases from $N = 1 4$ to $N = 2 2$ . The error bars are obtained by averaging over diferent NQS random seeds.

![](images/20f4508a908077f4704ba7dc527aad783f4d433837413877008096cb3439028f.jpg)

![](images/6ac3a9b1701b4313a0c1dc64ba3309d981a8b9ef96bc5d491177dcaa086bb7d4.jpg)

![](images/9f358ceb90735bb0c6f8e151d54bac4f754d2354e4d1f0fd67e4a57d622f4a02.jpg)

![](images/ffc9841d2fbc4d84f0099f4f874a9f54398275e0a1b2b0977f91f7e04d6d1b28.jpg)

![](images/f7343ec96bbd9cfccf09b6658045cbc43948db2e9c225e97d5df66f1bce4d1e6.jpg)

![](images/2c980b57906f8e65a256bdbec3020db6578219881aee268bb822d31b03db803f.jpg)  
Figure 5: Scaling benchmark (QSK seed=5678) for RBM and HRBM in the QSK setting as the system size N increases from $N = 1 4$ to $N = 2 2$ . The error bars are obtained by averaging over diferent NQS random seeds.

![](images/7c8e91e7d5e1c0cefc06a453c572b2924bf1ae12a5002b69c9f66c5f8782bf8e.jpg)

![](images/dbc3348e91781fd72996a32ca50c5d3d670ae29badf17d2d3f022b95e7926512.jpg)

![](images/afea6bc0d1d8cc2bb96cd15c79fca12c91ce237bdfaba69bc21466c3beaa9985.jpg)

![](images/482061eb347c998e77c20d1feb28a05ff3a059c3a22ab2b56ddff198a951d76d.jpg)

![](images/1dd4a45c50c7604b912fff925c9d70bcace3484aa7e503df72bf8a9c998158de.jpg)

![](images/2457ff762403dfae2639b484b80191d4f064253111cf142745b7d5ba54f03468.jpg)  
Figure 6: Scaling benchmark (QSK seed=6666) for RBM and HRBM in the QSK setting as the system size N increases from $N = 1 4$ to $N = 2 2$ . The error bars are obtained by averaging over diferent NQS random seeds.

![](images/df2b20eda0b46449cb612039895e4c1bc8dce85d218e7b242d6888c8fea5eb6a.jpg)

![](images/40d55d42708b53a8224228c0e2258329ee463916b627c7de5c2daf15f15c450a.jpg)

![](images/275454d78becb7a86658144098f4232e850574b4ef78c389eb7a4639977bb941.jpg)

![](images/af8191de9f027f355ab0372260426dc81839675c7b10c2e08bc291cc59a4d192.jpg)

![](images/f8d282191bcf55b3f3d4a70c3ca3ab645101e17f679a83c734f39d76ea353453.jpg)

![](images/8ffc37c27eb19ed1ccb6ea5112fef6f739e53e09c29fe57ab9c7ba10b8c76971.jpg)  
Figure 7: Scaling benchmark (QSK seed=9012) for RBM and HRBM in the QSK setting as the system size N increases from $N = 1 4$ to $N = 2 2$ . The error bars are obtained by averaging over diferent NQS random seeds.

![](images/cdf304c42ec47fc3256ca9905dd18c4bde68ee831e777fecb5b7e5ab6de306ea.jpg)

![](images/8ca49468160f860bfb4cff0622348b729b7e7626cff9722ac168ee12decf4328.jpg)

![](images/5724874190f9b0c9db366dc0dd61fd2bde6e2fab82d4adecf5c295dde2511936.jpg)

![](images/bb29cfffdf1d7ba40c9a3f8d76a979b15fd97916ef7d187e14aa8990b373930d.jpg)

![](images/c170b49f7852d7a41c233060e60fbd4ce3f458815c00b81dbb5215ec1ba117b8.jpg)

![](images/1b71d1c7a2ace011f8d5b0ae5fdb1e5f65df750bcb977173cf1bc94406c2fa4e.jpg)  
Figure 8: Scaling benchmark (QSK seed=9999) for RBM and HRBM in the QSK setting as the system size N increases from $N = 1 4$ to $N = 2 2$ . The error bars are obtained by averaging over diferent NQS random seeds.

## A.4 Full entanglement spectrum at $N = 1 4$ to $N = 2 2$

![](images/52c39301fbe8ff9e5ca64d11ca2d9905f0a74d4d95863cc4d48c63f015258cf2.jpg)  
Figure 9: Full entanglement spectrum of RBM and HRBM NQS in QSK VMC setting with N = 14 at diferent QSK model random seeds. In each subplot, the RBM and HRBM lines denote the mean value obtained by averaging over diferent VMC runs involving diferent NQS random seeds, with shading indicating one standard deviation.

![](images/a9d40c21a69723f8c6e834b0217e92e04c9288e2ed4e263930298c9d71a4e73c.jpg)

![](images/1037bba7b52f521018d2e882ee88434f30af6c9c39fb23e8145c57dd349a4494.jpg)

![](images/30514207fcd95e200737e8322e81dda789b232b9f95593f87c986b60039b12c4.jpg)

![](images/41d0a9ec9217802f727d863e932372715ae4555d1bf3b853551c6027bf2e0de1.jpg)

![](images/534e4fc0f2c591896b8053e1bbd04f07e6c41123b342fe367b38cd20e81b7952.jpg)

![](images/8a57ee8e5b090aabd5e0478142421b7455af3c121b7415c23130b7b13b281fdc.jpg)  
Figure 10: Full entanglement spectrum of RBM and HRBM NQS in QSK VMC setting with $N = 1 6$ at diferent QSK model random seeds. In each subplot, the RBM and HRBM lines denote the mean value obtained by averaging over diferent VMC runs involving diferent NQS random seeds, with shading indicating one standard deviation.

![](images/19b6cc1f1e966b158ce7e278733b4dcb0f82730b3ece3535f7dddd1c1cff66bc.jpg)  
Figure 11: Full entanglement spectrum of RBM and HRBM NQS in QSK VMC setting with $N = 1 8$ at diferent QSK model random seeds. In each subplot, the RBM and HRBM lines denote the mean value obtained by averaging over diferent VMC runs involving diferent NQS random seeds, with shading indicating one standard deviation.

![](images/2e93ebc190733fc861a0a68d8de581524ca7085aad55e185e4d9c1ad76b03e8b.jpg)  
Figure 12: Full entanglement spectrum of RBM and HRBM NQS in QSK VMC setting with N = 20 at diferent QSK model random seeds. In each subplot, the RBM and HRBM lines denote the mean value obtained by averaging over diferent VMC runs involving diferent NQS random seeds, with shading indicating one standard deviation.

![](images/07bb77a0a9966c373aadf142669734b297973d7ef348cf3a4f75bdab93e041a2.jpg)

![](images/98817774146023196119f08b9fe70d1e6a8158d5eeaff70bd85e1b627d1e6556.jpg)

![](images/4274840a4c515c15abeada49264bdf760198bba57cca20c93e38c6c10be42780.jpg)

![](images/308d06689b4b3500ea47977741f848bc6506fecb7c010fac082ff0c802a97e3a.jpg)

![](images/ef4546e199f6854f88503b18761413af55f84e4ae711f1a9e57b5db87dc4accf.jpg)

![](images/24ddf7781fce8bca4f14a708ea0c2e418a4e12bc56979fb705b765de562f4426.jpg)  
Figure 13: Full entanglement spectrum of RBM and HRBM NQS in QSK VMC setting with $N = 2 2$ at diferent QSK model random seeds. In each subplot, the RBM and HRBM lines denote the mean value obtained by averaging over diferent VMC runs involving diferent NQS random seeds, with shading indicating one standard deviation.

![](images/9fe395100e7c111a401cad512a549f24bb950b62153b8ed37ef24beef676dc5a.jpg)

## A.5 Efects of spatial constraint hyperparameter $L _ { m a x }$

Metrics Aggregated Across QSK (N = 24) Disorder Realizations (Mean ±1σ)

Figure 14: Efects of the spatial constraint hyperparameter $L _ { m a x }$ on the performance of HRBM at $N = 2 4$ For $N \leq 2 2$ , choosing $L _ { m a x } = 1 0$ sufices to ensure that HRBM outperform RBM but for $N = 2 4$ , choosing $L _ { m a x } = 1 0$ leads to an underperformance of HRBM compared to RBM. Enlarging $L _ { m a x }$ to 20 leads to a significant improve in HRBM’s performance.
<table><tr><td>NQS model</td><td> $\overline { { \epsilon ( \times 1 0 ^ { - 4 } ) } }$ </td><td> $\sigma ( \epsilon )$ </td><td>Mean variance</td><td>σ(variance)</td></tr><tr><td>RBM</td><td>3.16</td><td>0.000028</td><td>0.057231</td><td>0.003630</td></tr><tr><td>HRBM  $( L _ { m a x } = 8 )$ </td><td>3.75</td><td>0.000102</td><td>0.073887</td><td>0.022625</td></tr><tr><td>HRBM  $( L _ { m a x } = 1 0 )$ </td><td>3.37</td><td>0.000085</td><td>0.064724</td><td>0.017896</td></tr><tr><td>HRBM  $( L _ { m a x } = 1 2 )$ </td><td>3.17</td><td>0.000077</td><td>0.059689</td><td>0.014402</td></tr><tr><td>HRBM  $( L _ { m a x } = 1 5 )$ </td><td>2.91</td><td>0.000039</td><td>0.055376</td><td>0.007503</td></tr><tr><td>HRBM  $( L _ { m a x } = 2 0 )$ </td><td>2.78</td><td>0.000045</td><td>0.052951</td><td>0.007910</td></tr><tr><td>HRBM  $( L _ { m a x } = 2 5 )$ </td><td>2.78</td><td>0.000046</td><td>0.052918</td><td>0.007946</td></tr></table>

Table 5: This table lists the mean relative energy error ϵ and its associated standard deviation $\sigma ( \epsilon )$ , as well as mean variance and its associated standard deviation of HRBM NQS at diferent $L _ { m a x }$ and RBM NQS. These values are averaged over diferent QSK realizations (as well as NQS initialization seeds).

<table><tr><td>NQS</td><td> $\overline { { S _ { 2 } } }$ </td><td> $\overline { { \sigma ( S _ { 2 } ) } }$ </td><td> $\overline { { S _ { v N } } }$ </td><td> $\overline { { \sigma ( S _ { v N } ) } }$ </td><td> $\overline { { | S _ { 2 } ^ { \mathrm { e r r } } | } }$ </td><td> $\overline { { \sigma ( | S _ { 2 } ^ { \mathrm { e r r } } | ) } }$ </td><td> $\mathrm { ~ \textmu ~ } { \overline { { | S _ { v N } ^ { \mathrm { e r r } } | } } }$ </td><td> $\overline { { \sigma ( | S _ { v N } ^ { \mathrm { e r r } } | ) } }$ </td></tr><tr><td>RBM</td><td>0.489406</td><td>0.131863</td><td>0.950577</td><td>0.137814</td><td>0.004665</td><td>0.000390</td><td>0.007716</td><td>0.000944</td></tr><tr><td>HRBM  $( L _ { m a x } = 8 )$ </td><td>0.487151</td><td>0.128701</td><td>0.947000</td><td>0.135904</td><td>0.006924</td><td>0.003861</td><td>0.011293</td><td>0.003956</td></tr><tr><td>HRBM  $( L _ { m a x } = 1 0 )$ </td><td>0.489364</td><td>0.127650</td><td>0.949835</td><td>0.136158</td><td>0.004711</td><td>0.004537</td><td>0.008458</td><td>0.001999</td></tr><tr><td>HRBM  $( L _ { m a x } = 1 2 )$ </td><td>0.489676</td><td>0.127353</td><td>0.950345</td><td>0.136300</td><td>0.004418</td><td>0.005466</td><td>0.007948</td><td>0.002745</td></tr><tr><td>HRBM  $( L _ { m a x } = 1 5 )$ </td><td>0.490842</td><td>0.129606</td><td>0.950702</td><td>0.136795</td><td>0.003239</td><td>0.002747</td><td>0.007591</td><td>0.002387</td></tr><tr><td>HRBM  $( L _ { m a x } = 2 0 )$ </td><td>0.491859</td><td>0.131606</td><td>0.951050</td><td>0.137279</td><td>0.002222</td><td>0.000961</td><td>0.007243</td><td>0.002328</td></tr><tr><td>HRBM  $( L _ { m a x } = 2 5 )$ </td><td>0.491898</td><td>0.131683</td><td>0.951059</td><td>0.137293</td><td>0.002183</td><td>0.000958</td><td>0.007233</td><td>0.002331</td></tr></table>

Table 6: This table lists the mean R´enyi-2 $S _ { 2 }$ and von-Neumann $S _ { v N }$ entropies and their associated standard deviations, as well as the absolute entropy errors $( | S _ { 2 } ^ { \mathrm { e r r } } | = | S _ { 2 } ^ { N Q S } - \bar { S } _ { 2 } ^ { \mathrm { e x a c t } } | , \ : | \ : \hat { S } _ { v N } ^ { \mathrm { e r r } } | = S _ { v N } ^ { N Q S } - \bar { S } _ { v N } ^ { \mathrm { e x a c t } } | )$ and their associated standard deviations of HRBM NQS at diferent $L _ { m a x }$ and RBM NQS. These values are averaged over diferent QSK realizations (as well as NQS initialization seeds).

## References

[1] G. Passetti, D. Hofmann, P. Neitemeier, L. Grunwald, M. A. Sentef, and D. M. Kennes, Can Neural Quantum States Learn Volume-Law Ground States?, Physical Review Letters 131, 036502 (2023).

[2] Z. Denis, A. Sinibaldi, and G. Carleo, Comment on ’Can Neural Quantum States Learn Volume-Law Ground States, arXiv:2309.11534v2 [quant-ph]

[3] G. Carleo and M. Troyer, Solving the Quantum Many-Body Problem with Artificial Neural Networks, Science 355, 602 (2017), arXiv:1606.02318 [cond-mat.dis-nn]

[4] L. Huang and L. Wang, Accelerate Monte Carlo Simulations with Restricted Boltzmann Machines, arXiv:1610.02746v2.

[5] Z. Cai and J. Liu, Approximating quantum many-body wave-functions using artificial neural networks,Phys. Rev. B 97, 035116 (2018), arXiv:1704.05148 [cond-mat.str-el]

[6] H. Saito and M. Kato, Machine learning technique to find quantum many-body ground states of bosons on a lattice, J. Phys. Soc. Jpn. 87, 014001 (2018), arXiv:1709.05468 [cond-mat.dis-nn]

[7] X. Liang, Wen-Yuan Liu, Pei-Ze Lin, Guang-Can Guo, Yong-Sheng Zhang, and Lixin He, Solving frustrated quantum many-particle models with convolutional neural networks, arXiv: 1807.09422v2

[8] C. Roth, A. Szab´o, and A. H. MacDonald, High-accuracy variational Monte Carlo for frustrated magnets with deep neural networks, Phys. Rev. B 108, 054410 (2023), arXiv:2211.07749v2 [cond-mat.str-el]

[9] M. Hibat-Allah, M. Ganahl, L. E. Hayward, R. G. Melko, and J. Carrasquill, Recurrent neural network wave functions, Physical Review Research 2, 023358 (2020).

[10] M. Hibat-Allah, E. Merali, G. Torlai, R. G. Melko and J. Carrasquilla, Recurrent neural network wave functions for Rydberg atom arrays on kagome lattice, arXiv:2405.20384v1 [cond-mat.quant-gas]

[11] K. Sprague and S. Czischek, Variational Monte Carlo with Large Patched Transformers, Commun Phys 7, 90 (2024), arXiv:2306.03921 [quant-ph]

[12] H. Lange, G. Bornet, G. Emperauger, C. Chen, T. Lahaye, S. Kienle, A. Browaeys, A. Bohrdt, Transformer neural networks and quantum simulators: a hybrid approach for simulating strongly correlated systems, Quantum 9, 1675 (2025), arXiv:2406.00091 [cond-mat.dis-nn]

[13] F. Becca and S. Sorella, Quantum Monte Carlo Approaches for Correlated Systems (Cambridge University Press, 2017)

[14] H. L. Dao, Hyperbolic recurrent neural network as the first type of non-Euclidean neural quantum state ansatz, Eur. Phys. J. Plus 141:199, arXiv:2505.22083 [quant-ph, cond-mat.dis-nn, cs.LG, physics.comp-ph] (2026)

[15] H. L. Dao, New non-Euclidean neural quantum states from hyperbolic Lorentz recurrent architectures, arXiv:2604.2337 [quant-ph, cs.LG, cond-mat.dis-nn]

[16] H. L. Dao, Two-dimensional Hyperbolic RNN Neural Quantum State, arXiv:2606.25600 [quant-ph]

[17] O.-E. Ganea, G. Becigneul, and T. Hofmann, Hyperbolic Neural Networks, Advances in Neural Information Processing Systems 31, pages 5345–5355. Curran Associates, Inc. arXiv: 1805.09112 [cs.LG]

[18] D. Sherrington and S. Kirkpatrick, Solvable Model of a Spin-Glass, Physical Review Letters 35, 1792 (1975).

[19] P. M. Schindler, T. Guaita, T. Shi, E. Demler, and J. I. Cirac, A Variational Ansatz for the Ground State of the Quantum Sherrington-Kirkpatrick Model, arXiv:2204.02923v2

[20] F. Vicentini, D. Hofmann, A. Szabo, D. Wu, C. Roth, C. Giuliani, G. Pescia, J. Nys, V. Vargas-Calderon, N. Astrakhantsev, and G. Carleo, NetKet 3: Machine Learning Toolbox for Many-Body Quantum Systems, SciPost Phys. Codebases, 7 (2022).