# An Eficient Quantum Circuit for Flow Model Execution Using Quantum Neural Networks

Rui Che<sup>1\*</sup> and Ludvig af Klinteberg<sup>1</sup>

<sup>1\*</sup>Department of Business and Mathematics, M¨alardalen University, Universitetsplan 1, V¨aster˚as, 721 23, Sweden.

\*Corresponding author(s). E-mail(s): rui.che@mdu.se; Contributing authors: ludvig.af.klinteberg@mdu.se;

## Abstract

Flow models generate trajectories from an initial distribution to a target distribution by solving an ordinary diferential equation defined by a velocity field. Flow matching learns this velocity field by modeling the transport dynamics between the two distributions. Wavefunction flow establishes a formal connection between flow models and quantum dynamics by introducing a continuity Hamiltonian, which drives the Schr¨odinger evolution of quantum states. In this paper, we investigate accurate and eficient quantum simulation of the wavefunction flow, thereby realizing the eficient implementation of flow models on quantum computers. We first leverage a quantum read-only memory (QROM)-based phase kickback framework for the wavefunction flow simulation, generating probability densities that closely match those produced by the corresponding conventional flow model. To address the high circuit-resource cost, we further incorporate a trained quantum neural network (QNN) into the phase kickback framework, replacing QROM for data encoding. Numerical experiments demonstrate that our proposed method implements flow models on quantum computers more eficiently, since it maintains the accuracy of wavefunction flow simulation compared with the QROM-based framework, and significantly reduces the circuit resources.

Keywords: Flow models, Wavefunction flow, Quantum read-only memory, Quantum neural networks

## 1 Introduction

Generative models (Bond-Taylor et al. 2021) are a class of machine learning algorithms that learn unknown data distributions and generate new samples accordingly. The most widely used generative models include Generative Adversarial Networks (GANs) (Goodfellow et al. 2014), Variational Autoencoders (VAEs) (Kingma and Welling 2013), Autoregressive models (Van Den Oord et al. 2016), and flow models (Gat et al. 2024; Lipman et al. 2022). Flow models aim at generating complex target distributions from simple source distributions by learning velocity fields. The learned velocity fields govern the transportation of distributions. Flow matching (Lipman et al. 2022) provides an eficient method of learning velocity fields by approximating trajectories between source and target distributions that have been predefined, which is an ordinary diferential equation (ODE)-based model.

Recently, the connection between flow models and quantum dynamics has been developed (Layden et al. 2025). By introducing a continuity Hamiltonian ${ \hat { H } } .$ , the implementation of flow models on quantum computers can be regarded as the Schr¨odinger evolution under this continuity Hamiltonian. The resulting quantum dynamics is defined as wavefunction flow. Wavefunction flow can be realized through the simulation of the Hamiltonian evolution operator $e ^ { - i t { \hat { H } } }$ . By discretizing the evolution time and using the Suzuki-Trotter formula, it can be expressed as a product of exponentiation operators $e ^ { - i \Delta t \hat { H } _ { j } }$ corresponding to all Hamiltonian terms $\hat { H } _ { j }$ (Chen et al. 2022; Low et al. 2023). Here, ∆t denotes the discretized time step interval. Each operator can be regarded as a quantum oracle and is implemented sequentially in the quantum circuit (Kornell and Selinger 2023).

Phase kickback (Cleve et al. 1998) is a classical quantum technique for the implementation of the evolution operator $e ^ { - i \Delta t H _ { j } }$ . It encodes the eigenvalues of $\hat { H } _ { j }$ into ancilla qubits, then leverages rotation gates to generate the desired phase $e ^ { - i \Delta t \hat { H } _ { j } }$ Finally, the adjoint eigenvalue encoding oracle is employed to uncompute the ancilla qubits to the initial states. In the wavefunction flow framework, the continuity Hamiltonian $\hat { H } _ { j }$ is constructed by the time-dependent potential term $V _ { t }$ and the kinetic term K. These two terms can be transformed into diagonal operators $D _ { V _ { t } }$ and $D _ { K }$ by selecting the appropriate basis (Layden et al. 2025). To implement such evolution operators $\bar { e } ^ { - i \Delta t D _ { V _ { t } } ^ { - } }$ and $e ^ { - i \Delta t D _ { K } }$ using phase kickback frameworks, one of the key steps is to encode all diagonal entries of $D _ { V _ { t } }$ and $D _ { K }$ into the quantum register.

Quantum read-only memory (QROM) (Babbush et al. 2018; Lee et al. 2021) is a widely used quantum technique for encoding classical data in the quantum circuit, where each data item corresponds to an initial basis state. We utilize QROMbased phase kickback frameworks to implement the evolution operators $e ^ { - i \Delta t \dot { D } _ { V _ { t } } }$ and $e ^ { - i \Delta t \bar { D _ { K } } }$ for the wavefunction flow simulation, and verify the correspondence between the flow model and the wavefunction flow simulation. However, the QROM circuit is constructed by a series of multi-controlled X gates (Zindorf and Bose 2025), and the decomposition of a multi-controlled X with n control qubits without ancilla qubits requires $\Theta ( 2 ^ { n } )$ CNOT gates (Barenco et al. 1995). Here, Θ denotes “of order exactly 2<sup>n</sup>” (Knuth 1976). Hence, QROM faces exponentially increasing computational complexity, which leads to a high circuit resource cost for the wavefunction flow simulation.

To solve this problem, we introduce a specialized quantum neural network (QNN) circuit to replace QROM in the phase kickback framework for the wavefunction flow simulation. The QNN is a type of parameterized quantum circuit (PQC)-based variational quantum algorithm (Abbas et al. 2021; Schuld et al. 2014). It combines features of neural networks and quantum circuits. The main areas of QNN application include classification problems (Mahmud et al. 2024), generative learning (Tian et al. 2023), optimization (Amosy et al. 2024), chemistry (Smaldone et al. 2025), and financial predictions (Paquet and Soleymani 2022). Through appropriate and suficient training, our QNN-based phase kickback framework can maintain accuracy compared to that of the QROM-based framework. In addition, with this replacement, the number of required CNOT gates can be reduced to Θ(n). QNN has been applied to Hamiltonian simulation using the variational quantum simulator (VQS) (Li and Benjamin 2017; Heya et al. 2023). The VQS trains the QNN to learn the target quantum state from the given initial quantum state. However, each such QNN is designed for a single initial quantum state, and can generate only the corresponding target quantum state. For a diagonal Hamiltonian operator in the wavefunction flow model, where each diagonal entry corresponds to a diferent initial state, VQS is unable to generate all the desired target states by training the QNN. Therefore, this approach is not capable of realizing the evolution generated by the entire Hamiltonian. In contrast, our QNN-based framework achieves the whole evolution eficiently.

In this paper, inspired by the theory of wavefunction flow (Layden et al. 2025), we investigate an eficient simulation framework for wavefunction flow, thereby realizing the implementation of flow models on quantum computers eficiently. Building on the QROM-based phase kickback framework, we develop a QNN circuit to replace the QROM for data encoding in the wavefunction flow simulation. The new approach significantly reduces the cost of circuit resources while maintaining accuracy compared with the QROM-based framework. The contributions of the paper are as follows.

• We leverage a QROM-based phase kickback technique to simulate wavefunction flow, thereby verifying the wavefunction flow theory and demonstrating the feasibility of implementing flow models on quantum computers.

• We propose a QNN circuit to replace QROM in the phase kickback framework for data encoding in the wavefunction flow simulation. Instead of employing multicontrolled X gates to encode the exact binary bits in the ancilla qubits through QROM, our QNN circuit learns the mapping from the states of the address qubits to the corresponding probabilities of the ancilla qubits through training parameterized rotation gates and controlled rotation gates. This method avoids the use of highlevel controlled gates and the high circuit resource cost. We also extend the model to encode time-dependent potential operators by utilizing a time oracle inside the QNN circuit. Compared with the QROM-based framework, our proposed QNNbased phase kickback framework maintains accuracy in wavefunction flow simulation while significantly reducing circuit resources, enabling the implementation of flow models on quantum computers more eficiently.

• We combine flow matching with wavefunction flow simulation. The conventional flow model employs the velocity field learned from flow matching to generate the final distribution. In our work, we learn the potential using flow matching, and its gradient defines the conservative velocity field. Then we incorporate this velocity field into our wavefunction flow simulation. The final distribution generated by the wavefunction flow simulation successfully reconstructs the target distribution and matches the final distribution generated by the conventional flow model. This demonstrates the feasibility of employing wavefunction flow simulation for implementing flow models on quantum computers.

The remainder of this paper is organized as follows. Section 2 provides a review of the flow model and flow matching, wavefunction flow, Hamiltonian simulation, quantum data encoding via QROM, and QNN. Section 3 introduces the eficient wavefunction flow simulation through the QROM-based phase kickback framework and our proposed QNN-based phase kickback framework. Section 4 presents the experimental results and the discussion. Section 5 provides a brief conclusion.

## 2 Background

## 2.1 Flow model and flow matching

The flow model is defined by a time-dependent d-dimensional velocity field $\boldsymbol { v } ( \boldsymbol { x } , t )$ ， which generates the flow of samples through the ODE

$$
\frac { d } { d t } x _ { t } = v ( x , t ) , \ 0 \leq t \leq 1 .\tag{1}
$$

Given samples from the d-dimensional initial distribution $x _ { 0 } ~ \sim ~ p _ { 0 }$ , new samples $x _ { 1 }$ can be generated that follow the target d-dimensional distribution $p _ { 1 }$ by solving Equation (1).

Flow matching (Lipman et al. 2022) aims to learn a parameterized velocity field $v _ { \theta } ( x , t )$ that approximates the exact velocity field $v ( x , t )$ , which generates a probability path from an initial distribution $p _ { 0 }$ to a target distribution $p _ { 1 }$ . This is done in two steps. First, a d-dimensional probability path $( p _ { t } ) _ { 0 \leq t \leq 1 }$ connecting $p _ { 0 }$ and $p _ { 1 }$ is defined. For samples $x _ { 0 } \sim p _ { 0 }$ and $x _ { 1 } \sim p _ { 1 }$ , the flow trajectory can be viewed as the interpolation over $t \in [ 0 , 1 ]$ (Lipman et al. 2022; Liu et al. 2022)

$$
x _ { t } = ( 1 - t ) x _ { 0 } + t x _ { 1 } .\tag{2}
$$

The conditional probability path $p _ { t } ( x | x _ { 0 } , x _ { 1 } )$ can be defined conditioned on a sample pair $( x _ { 0 } , x _ { 1 } )$ , resulting in the probability path that follows the expression (Lipman et al. 2022; Feng et al. 2025)

$$
p _ { t } ( x ) = \int \int p _ { t } ( x | x _ { 0 } , x _ { 1 } ) p _ { 0 } ( x _ { 0 } ) p _ { 1 } ( x _ { 1 } ) d x _ { 0 } d x _ { 1 } ,\tag{3}
$$

which describes the transport of probability from $x _ { 0 }$ to $x _ { 1 }$ .

In the second step, a parameterized velocity field $v _ { \theta } ( x , t )$ that can generate $p _ { t }$ is learned through neural networks. It has been demonstrated that there exists a target velocity field $v _ { p } ( x , t )$ that is capable of generating the probability path $p _ { t }$ (Lipman et al. 2022; Feng et al. 2025). This produces the flow matching objective

$$
\mathcal { L } ( \theta ) _ { \mathrm { { F M } } } = \mathbb { E } _ { t , x _ { 0 } \sim p _ { 0 } , x _ { 1 } \sim p _ { 1 } } \left[ \left. v _ { \theta } ( x , t ) - v _ { p } ( x , t ) \right. ^ { 2 } \right] .\tag{4}
$$

However, this objective is dificult to implement, as the velocity field $v _ { p } ( x , t )$ is complicated and dificult to evaluate. Therefore, the conditional velocity field can be employed to simplify the training process (Lipman et al. 2022, 2024). For the interpolation trajectory in Equation (2), the conditional velocity field can be defined as the derivative of $x _ { t }$ with respect to time t

$$
v ( x , t | x _ { 0 } , x _ { 1 } ) = { \frac { d x _ { t } } { d t } } = x _ { 1 } - x _ { 0 } .\tag{5}
$$

Then, the conditional flow matching objective is defined as

$$
\mathcal { L } ( \theta ) _ { \mathrm { C F M } } = \mathbb { E } _ { t , x _ { 0 } \sim p _ { 0 } , x _ { 1 } \sim p _ { 1 } } \left[ \| v _ { \theta } ( x _ { t } , t ) - v ( x , t | x _ { 0 } , x _ { 1 } ) \| ^ { 2 } \right] .\tag{6}
$$

A neural network with parameters $\theta$ is trained to learn $v _ { \theta } ( x _ { t } , t )$ by minimizing the objective. After training, the trajectory can be generated using the Euler update rule

$$
x _ { t + \Delta t } = x _ { t } + v _ { \theta } ( x _ { t } , t ) \Delta t ,\tag{7}
$$

or a higher-order Runge-Kutta method, such as RK45. (Dormand and Prince 1980).

## 2.2 Wavefunction flow

This section reviews the wavefunction flow theory proposed in (Layden et al. 2025), including the connection between flow models and quantum dynamics, the construction of the continuity Hamiltonian, and the definition of the wavefunction flow.

Define $v _ { t } ( x ) : = v ( x , t )$ . For an infinitesimal time step dt, by defining the position transformation $x _ { t + d t } = x _ { t } + v _ { t } ( x ) d t$ , and by employing the change of variable formula, one can describe the change of probability density through its Jacobian. This determines that the probability path $p _ { t }$ in Equation (3), which is generated by the velocity field $v _ { t } ( x )$ , satisfies the continuity equation

$$
\frac { \partial } { \partial t } p _ { t } = - \nabla \cdot \left( v _ { t } ( x ) p _ { t } \right) .\tag{8}
$$

The connection between flow models and quantum dynamics has been established by the definition of an appropriate continuity Hamiltonian. Given the probability path

$p _ { t }$ and its wavefunction $\Psi _ { t } = \sqrt { p _ { t } } ;$ , by derivation

$$
i \frac { \partial } { \partial t } \Psi _ { t } = \frac { i } { 2 \sqrt { p _ { t } } } \frac { \partial } { \partial t } p _ { t } = - \frac { i } { 2 \Psi _ { t } } \nabla \cdot \left( \boldsymbol { v } _ { t } ( \boldsymbol { x } ) \Psi _ { t } ^ { 2 } \right) ,\tag{9}
$$

and by applying the product rule, we obtain

$$
\begin{array} { r l r } & { } & { i \displaystyle { \frac { \partial } { \partial t } } \Psi _ { t } = - \frac { i } { 2 \Psi _ { t } } \left( \Psi _ { t } \nabla \cdot ( \boldsymbol { v } _ { t } ( \boldsymbol { x } ) \Psi _ { t } ) + ( \boldsymbol { v } _ { t } ( \boldsymbol { x } ) \Psi _ { t } ) \cdot \nabla \Psi _ { t } \right) } \\ & { } & { ~ = \displaystyle { \frac { 1 } { 2 } \left( ( - i \nabla ) \cdot ( \boldsymbol { v } _ { t } ( \boldsymbol { x } ) \Psi _ { t } ) + \boldsymbol { v } _ { t } ( \boldsymbol { x } ) \cdot ( - i \nabla ) \Psi _ { t } \right) } . } \end{array}\tag{10}
$$

Thus, if $p _ { t }$ follows the continuity equation (8) under the velocity field $v _ { t } ( x )$ , the wavefunction $\Psi _ { t }$ follows the Schr¨odinger equation

$$
i \frac { \partial } { \partial t } \Psi _ { t } = \hat { H } _ { t } \Psi _ { t } ,\tag{11}
$$

under time-dependent d-dimensional continuity Hamiltonian $\hat { H } _ { t }$ defined as

$$
\hat { H } _ { t } = \frac { 1 } { 2 } \left( \boldsymbol { \hat { p } } \cdot \boldsymbol { v } _ { t } ( \boldsymbol { x } ) + \boldsymbol { v } _ { t } ( \boldsymbol { x } ) \cdot \boldsymbol { \hat { p } } \right) .\tag{12}
$$

Here, ${ \hat { p } } = - i \nabla$ denotes the d-dimensional momentum operator.

When the velocity field is conservative, i.e., $v _ { t } ~ = ~ \nabla V _ { t }$ , where $V _ { t }$ is a potential function, the corresponding time-dependent continuity Hamiltonian can be written in commutator form

$$
\hat { H } _ { t } = i [ \hat { K } , V _ { t } ] .\tag{13}
$$

Here, $[ \hat { K } , V _ { t } ] = \hat { K } V _ { t } - V _ { t } \hat { K }$ , and $\begin{array} { r } { \hat { K } = \frac { 1 } { 2 } \hat { p } \cdot \hat { p } \stackrel { \cdot } { p } = - \frac { 1 } { 2 } \nabla ^ { 2 } } \end{array}$ denotes the kinetic energy operator. In addition, $V _ { t }$ represents the potential function.

Therefore, the continuity Hamiltonians defined in Equations (12) and (13) establish the equivalence between the continuity equation and the corresponding Schr¨odinger equation. This equivalence formulates the close connection between flow models and quantum dynamics. Given the initial distribution $p _ { 0 }$ and the corresponding wavefunction $\Psi _ { 0 } = \sqrt { p _ { 0 } }$ , define the entire evolution time as T. The final distribution $p _ { T }$ generated by the flow model under the velocity field $v _ { t }$ has the following relationship with the final wavefunction $\Psi _ { T }$ evolved through the Schr¨odinger equation driven by the $v _ { t } .$ -based continuity Hamiltonian

$$
p _ { T } = \Psi _ { T } ^ { 2 } .\tag{14}
$$

$\mathrm { B y }$ constructing the continuity Hamiltonian, the quantum dynamics it generates is defined as a wavefunction flow.

## 2.3 Hamiltonian simulation

To implement the evolution of the wavefunction flow on a quantum computer, the corresponding continuity Hamiltonian must be eficiently simulated. This can be achieved

by Hamiltonian simulation, which implements quantum dynamics generated by the Schr¨odinger Equation (11). For the time-dependent continuity Hamiltonian $\hat { H } _ { t } .$ , the solution of the Schr¨odinger equation (Childs et al. 2021) is

$$
\Psi _ { T } = \mathcal { T } \exp \left( - i \int _ { 0 } ^ { T } \hat { H } _ { \tau } d \tau \right) \Psi _ { 0 } ,\tag{15}
$$

Here $\tau$ denotes the time-ordered operator (Childs et al. 2021), which preserves the Hamiltonians in the correct order according to the time evolution, since Hamiltonians at diferent times do not commute. This determines the quantum evolution from the initial state $\Psi _ { 0 }$ to the final state $\Psi _ { T }$ . Therefore, the central objective of the Hamiltonian simulation is to simulate the time evolution operator

$$
U ( T , 0 ) = \mathcal { T } \mathrm { e x p } \left( - i \int _ { 0 } ^ { T } \hat { H } _ { \tau } d \tau \right) .\tag{16}
$$

The main methods for simulating the operator include the Trotter product formula (Low et al. 2023; Watson and Watkins 2025), the linear combination of unitaries (LCU) (Berry et al. 2015), and Qubitization (Low and Chuang 2019). Among all approaches, the Trotter product formula is the most straightforward to implement. Therefore, it is widely used in various scenarios.

To approximate the operator $U ( T , 0 )$ in Equation (16) using the Trotter product formula, the evolution time $T$ is generally divided into r time steps with the interval $\Delta t = T / r$ . Then $U ( T , 0 )$ is expressed as

$$
U ( T , 0 ) = U \left( T , t _ { r - 1 } \right) U \left( t _ { r - 1 } , t _ { r - 2 } \right) \cdot \cdot \cdot U \left( t _ { 1 } , 0 \right) ,
$$

where $t _ { n + 1 } - t _ { n } = \Delta t$ . Each decomposition operator ${ \cal U } ( t _ { n + 1 } , t _ { n } )$ is defined as

$$
U ( t _ { n + 1 } , t _ { n } ) = \mathcal { T } \exp \left( - i \int _ { t _ { n } } ^ { t _ { n + 1 } } \hat { H } _ { \tau } d \tau \right) ,\tag{17}
$$

and can be approximated by $W ( t _ { n } )$

$$
U ( t _ { n + 1 } , t _ { n } ) \approx W ( t _ { n } ) = e ^ { - i \hat { H } _ { t _ { n } } \Delta t } .\tag{18}
$$

Given the 4-step Trotter product formula for commutators (Chen et al. 2022)

$$
e ^ { x A } e ^ { x B } e ^ { - x A } e ^ { - x B } = e ^ { x ^ { 2 } [ A , B ] } + O ( x ^ { 3 } ) ,\tag{19}
$$

consider a Hamiltonian of the commutator form $H _ { t _ { n } } = i [ A , B ]$ . By choosing $x = { \sqrt { \Delta t } }$ we can construct the corresponding approximation term $W ( t _ { n } )$ , which generates

$$
\begin{array} { r l } & { W ( t _ { n } ) = e ^ { - i \cdot i [ A , B ] \Delta t } = e ^ { \Delta t [ A , B ] } } \\ & { \qquad = e ^ { i \sqrt { \Delta t } A } e ^ { i \sqrt { \Delta t } B } e ^ { - i \sqrt { \Delta t } A } e ^ { - i \sqrt { \Delta t } B } + O \left( \Delta t ^ { 3 / 2 } \right) . } \end{array}\tag{20}
$$

The 8-step product formula (Childs and Wiebe 2013) is a symmetric form of the 4-step Trotter product formula. By choosing $x = \sqrt { \frac { \Delta t } { 2 } }$ , the formula is given by

$$
\begin{array} { r l } & { W ( t _ { n } ) } \\ & { \ = e ^ { i \sqrt { \frac { \Delta t } { 2 } } A } e ^ { i \sqrt { \frac { \Delta t } { 2 } } B } e ^ { - i \sqrt { \frac { \Delta t } { 2 } } A } e ^ { - i \sqrt { \frac { \Delta t } { 2 } } B } e ^ { - i \sqrt { \frac { \Delta t } { 2 } } A } e ^ { - i \sqrt { \frac { \Delta t } { 2 } } B } e ^ { i \sqrt { \frac { \Delta t } { 2 } } A } e ^ { i \sqrt { \frac { \Delta t } { 2 } } A } e ^ { i \sqrt { \frac { \Delta t } { 2 } } B } + { \cal O } \left( \Delta t ^ { 2 } \right) . } \end{array}\tag{21}
$$

## 2.4 Data encoding via QROM

The question of how to load classical data into quantum computers is a critical problem in quantum algorithms. In Hamiltonian simulation, encoding the data items corresponding to the Hamiltonian operator into the quantum register is the first step.

For example, consider a Hamiltonian that is a diagonal operator; all the diagonal entries must be encoded into the quantum register. Quantum read-only memory (QROM) is a prominent technique used to encode such classical data entries into the quantum register (Babbush et al. 2018); see also (Hann et al. 2021, Sec. VI). Specifically, for an n-qubit quantum circuit system, there are $N = 2 ^ { n }$ basis states in total. QROM is able to encode N classical data entries in the quantum register, and each data entry corresponds to a basis state.

In the QROM oracle, each data item is represented as an m-bit binary string. For the i-th data entry $d _ { i }$ , the corresponding binary string is expressed as

$$
d _ { i } = ( d _ { i , 1 } , \cdot \cdot \cdot , d _ { i , m } ) ,\tag{22}
$$

where $d _ { i , j } \in \{ 0 , 1 \}$ . Furthermore, m ancilla qubits are used to encode these binary bits in the circuit.

For example, when $n = 2$ and $m = 2$ , Figure 1 illustrates the corresponding QROM circuit. The first two qubits serve as address qubits associated with the indices of the encoded data entries. The last two qubits are ancilla qubits initialized as 0 , which play the role of the data register. For each address $x _ { i } \in \{ 0 , 1 \} ^ { 2 }$ , a controlled operator $O _ { i }$ is implemented through multi-controlled X gates, whose control values correspond to the address state $\left| x _ { i } \right.$ . The specific implementation form of $O _ { i }$ depends on the binary string of the i-th data entry $d _ { i }$ . If $d _ { i , j } = 1$ , a multi-controlled X will be implemented in the j-th ancilla qubit, transforming its state from 0 to 1 . If $d _ { i , j } = 0$ , no X gate is applied, and the state of j-th ancilla qubit remains 0 . Hence, each controlled oracle selects a state of the address qubits and encodes the binary representation of the data entry corresponding to the same address index into the quantum circuit.

![](images/67f3db267bead183960bc730a4921344163bb8526134c39b54c8a56e2cfac4be.jpg)  
Fig. 1 QROM circuit for $n = 2$ and $m = 2 ,$ where add denotes the address qubit $^ { \mathrm { ~ s , ~ } }$ and anc denotes the ancilla qubits. Multi-controlled operators $O _ { i }$ are implemented on ancilla qubits and controlled by address qubits by iterating over all $2 ^ { 2 } = 4$ address states. Black and white control nodes represent controls on |0⟩ and $| 1 \rangle$ , respectively, which select the diferent address states. Therefore, the controlled states of four $O _ { i }$ operators are |00⟩, |01⟩, |10⟩ and |11⟩, respectively

Therefore, the operator $O _ { i }$ performs the transformation

$$
| x _ { i } \rangle | 0 \rangle \stackrel { \otimes m } { \longrightarrow } | x _ { i } \rangle | d _ { i , 1 } \rangle | d _ { i , 2 } \rangle \cdot \cdot \cdot | d _ { i , m } \rangle ,\tag{23}
$$

for all $i \in \{ 1 , 2 , \ldots , N \}$ . This implies that all the classical data entries are encoded into the quantum circuit via QROM.

QROM enables data encoding without preparing them in superposition (Crum et al. 2025). It has been applied in quantum cryptography (Gidney and Eker˚a 2021), chemistry (Lee et al. 2021), and some other fields to encode classical data into the quantum circuit.

## 2.5 Quantum neural networks

Quantum neural networks (QNNs) have been widely used in quantum machine learning. They combine the capabilities of both quantum computing and neural networks. QNNs are frameworks based on parameterized quantum circuits (PQCs) (Benedetti et al. 2019), also known as variational quantum circuits, and optimized by classical gradient-based methods (Abbas et al. 2021).

The workflow of a QNN is summarized as follows. First, for the n-qubit QNN circuit whose initial state is $| 0 \rangle ^ { \otimes n }$ , the input data is encoded into the quantum circuit by preparing the corresponding quantum state $| \psi _ { \mathrm { i n } } \rangle$ . Then, the prepared state passes through several layers consisting of rotation gates parameterized with $\pmb \theta = ( \theta _ { 1 } , \dots , \theta _ { m } )$ and CNOT gates (Nielsen and Chuang 2010). The latter generate entanglement between qubits. The evolved state is measured by an observable, such as Pauli-Z. Then, the output of the QNN is defined as the expectation value of the observable (Zhao and Wang 2021)

$$
\hat { y } = \langle \psi _ { \mathrm { o u t } } | M | \psi _ { \mathrm { o u t } } \rangle ,\tag{24}
$$

where M denotes the measurement observable, and $| \psi _ { \mathrm { o u t } } \rangle$ represents the final quantum state generated by the QNN

$$
\begin{array} { r } { | \psi _ { \mathrm { o u t } } \rangle = U ( \pmb \theta ) | 0 \rangle ^ { \otimes n } . } \end{array}\tag{25}
$$

The gradient of the output $\hat { y }$ with respect to $\theta _ { i }$ can be obtained by the parameter shift rule (Schuld et al. 2019). Once the gradients are obtained, we can construct a suitable loss function to measure the distance between the predicted output and the exact output. Then the learning objective is to minimize the loss function. This process can be completed outside the quantum circuit using classical gradient-based methods, such as the Adam optimizer (Kingma and Ba 2014).

Advanced architectures of QNN include quantum convolutional neural networks (Cong et al. 2019), quantum recurrent neural networks (Siemaszko et al. 2023), quantum graph neural networks (Verdon et al. 2019), and quantum autoencoders (Romero et al. 2017). Recently, the emergence of QNNs in a wide range of applications (Innan et al. 2024; Mahmud et al. 2024; Amosy et al. 2024) has helped quantum machine learning to develop rapidly in diferent fields.

## 3 Methodology

In this section, first, a phase kickback framework via QROM for wavefunction flow simulation is presented (see Section 3.1), followed by the proposed QNN-based phase kickback framework for more eficient simulation (see Section 3.2).

## 3.1 QROM-based phase kickback framework for wavefunction flow simulation

To implement the wavefunction flow simulation under the continuity Hamiltonian $\hat { H } _ { t }$ defined in Equation (13), both the spatial domain and the evolution time need to be discretized. In addition, we adopt a periodic spatial domain, which can be represented as a d-dimensional torus, to match the Fourier spatial discretization.

## 3.1.1 Spatial and temporal discretization

In this subsection, we adopt the spatial and temporal discretization methods proposed in (Layden et al. 2025).

Since the kinetic operator is discretized through the quantum Fourier transform (QFT) (Nielsen and Chuang 2010), to match the periodic characteristic of the Fourier basis, and to avoid the spatial boundary discontinuity, the spatial domain is chosen as a torus. Consider a d-dimensional torus $\mathbb { T } ^ { d } .$ where each spatial dimension is defined on the domain $[ - L , L ]$ . Let N be a positive integer power of 2, the domain $[ - L , L ]$ is first discretized into an N-point grid

$$
\mathbb { X } _ { N } = \frac { 2 L } { N } \{ - N / 2 , - N / 2 + 1 , \dots , N / 2 - 1 \} \subset \mathbb { T } ^ { 1 } .\tag{26}
$$

Thus, the d-dimensional grid $\mathbb { X } _ { N } ^ { d }$ consists of $N ^ { d }$ spatial grid points, and each grid point $x = ( x _ { 1 } , \ldots , x _ { d } ) \in \mathbb { X } _ { N } ^ { d }$ corresponds to a basis state by assigning an integer index to each coordinate $x _ { i }$

$$
k _ { i } = { \frac { ( x _ { i } + L ) N } { 2 L } } \in \{ 0 , 1 , \ldots , N - 1 \} .\tag{27}
$$

Here $k _ { i }$ corresponds to the one-hot vector $| k _ { i } \rangle \in \mathbb { C } ^ { N } , { \mathrm { i . e . , } } | k _ { i } \rangle = ( 0 , \ldots , 0 , 1 , 0 , \cdots , 0 ) ^ { T }$ where the $k _ { i } .$ -th component is 1 and other components are 0. Therefore, the basis state corresponding to x is given by

$$
| x \rangle = | k _ { 1 } \rangle \otimes | k _ { 2 } \rangle \otimes \dots \otimes | k _ { d } \rangle ,\tag{28}
$$

where $\otimes$ denotes the tensor product of the vectors.

For the continuity Hamiltonian associated with the wavefunction flow, which is defined in Equation (13), a discrete continuity Hamiltonian is defined for simulation

$$
H _ { t } = i [ K , D _ { V _ { t } } ] .\tag{29}
$$

Here, $D _ { V _ { t } }$ denotes the $N ^ { d } \times N ^ { d }$ diagonal matrix containing discretized values of the potential function $V _ { t }$ in Equation (13)

$$
D _ { V _ { t } } = \sum _ { x \in \mathbb { X } _ { N } ^ { d } } V _ { t } ( x ) \vert x \rangle \langle x \vert .\tag{30}
$$

Each diagonal entry in $D _ { V _ { t } }$ corresponds to the value of $V _ { t }$ at the associated spatial grid point $\boldsymbol { x } \in \mathbb { X } _ { N } ^ { d }$ , which is given by $D _ { V _ { t } } | x \rangle = V _ { t } ( x ) | x \rangle$

The matrix K denotes the discretization of the kinetic operator $\hat { K }$ in Equation (13). For one spatial dimension, $\begin{array} { r } { \hat { K } = - \frac { 1 } { 2 } \frac { \partial ^ { 2 } } { \partial x ^ { 2 } } } \end{array}$ . Since we have the second-order derivative of $e ^ { i k x }$

$$
\frac { \partial ^ { 2 } } { \partial x ^ { 2 } } e ^ { i k x } = - k ^ { 2 } e ^ { i k x } ,\tag{31}
$$

thus,

$$
\hat { K } e ^ { i k x } = - \frac { 1 } { 2 } \frac { \partial ^ { 2 } } { \partial x ^ { 2 } } e ^ { i k x } = \frac { 1 } { 2 } k ^ { 2 } e ^ { i k x } .\tag{32}
$$

Hence, $e ^ { i k x }$ is the eigenfunction of $\hat { K }$ with eigenvalue ${ \scriptstyle { \frac { 1 } { 2 } } } k ^ { 2 }$ . Therefore, the kinetic operator is diagonal in the Fourier basis. After spatial discretization, define

$$
k _ { j } = \frac { \pi } { L } \left( j - \frac { N } { 2 } \right) , \ j = 0 , \ldots , N - 1 ,\tag{33}
$$

where $j - \frac { N } { 2 }$ represents the centered Fourier basis index. The diagonal representation of the discretized kinetic operator K in the Fourier basis for one spatial dimension can be defined as

$$
D _ { K } = \left( \frac { \pi } { L } \right) ^ { 2 } \sum _ { j = 0 } ^ { N - 1 } \left( j - \frac { N } { 2 } \right) ^ { 2 } | j \rangle \langle j | .\tag{34}
$$

Therefore, $K$ can be expressed using $D _ { K }$ via QFT across all d spatial dimensions, which is defined as

$$
K = \frac { 1 } { 2 } ( S F D _ { K } F ^ { \dagger } S ^ { \dagger } ) ^ { \oplus d } .\tag{35}
$$

The operator F represents the $\mathrm { Q F T }$ transformation

$$
F = \frac { 1 } { \sqrt { N } } \sum _ { j , \ell = 0 } ^ { N - 1 } \exp \left( \frac { i 2 \pi j \ell } { N } \right) | j \rangle \langle \ell | .\tag{36}
$$

This enables us to encode a state $| x \rangle$ by superposition of computational basis $| j \rangle$ , with diferent relative phases and the same amplitude $\scriptstyle { \frac { 1 } { \sqrt { N } } }$ (Ruiz-Perez and Garcia-Escartin 2017)

$$
F | x \rangle = \frac { 1 } { \sqrt { N } } \sum _ { j = 0 } ^ { N - 1 } e ^ { i \frac { 2 \pi x j } { N } } | j \rangle .\tag{37}
$$

The phase shift operator $S$ is defined as

$$
{ \cal { S } } = \sum _ { j = 0 } ^ { N - 1 } ( - 1 ) ^ { j } \ : | j \rangle \langle j | ,\tag{38}
$$

which employ a phase shift value $( - 1 ) ^ { j }$ to each basis state $| j \rangle$ . The operator $M ^ { \oplus d }$ denotes the Kronecker sum, which transforms the one-dimensional matrix M into a d-dimensional operator

$$
M ^ { \oplus d } = ( M \otimes I ^ { \otimes d - 1 } ) + ( I \otimes M \otimes I ^ { \otimes d - 2 } ) + \cdot \cdot \cdot + ( I ^ { \otimes d - 1 } \otimes M ) .\tag{39}
$$

Each term involves applying M to one dimension and applying the identity operator to other dimensions.

The temporal discretization divides the entire evolution time T into multiple time steps with the interval $\Delta t$ . For an interval $[ t , t _ { 1 } ]$ with $0 \leq t < t _ { 1 } \leq T$ , the 8-step Trotter product formula in Equation (21) can be employed for the simulation of the operator $W ( t )$ to approximate $U ( t _ { 1 } , t )$ defined in Equation (18). This yields

$$
\begin{array} { r } { W ( t ) = e ^ { i \beta D _ { V _ { t } } } e ^ { i \alpha K } e ^ { - i \beta D _ { V _ { t } } } e ^ { - i \alpha K } e ^ { - i \beta D _ { V _ { t } } } e ^ { - i \alpha K } e ^ { i \beta D _ { V _ { t } } } e ^ { i \alpha K } , } \end{array}\tag{40}
$$

where

$$
\alpha = \frac { 2 L } { \pi N } \sqrt { \frac { \Delta t } { d } } , \qquad \beta = \frac { \pi N } { 4 L } \sqrt { d \Delta t } .\tag{41}
$$

The parameters satisfy $\begin{array} { r } { \alpha \beta = \frac { \Delta t } { 2 } } \end{array}$ according to Equation (21), and N denotes the number of grids in each dimension after spatial discretization.

The corresponding temporal discretization error $| | U ( t _ { 1 } , t ) - W ( t ) | |$ satisfies

$$
\| U ( t _ { 1 } , t ) - W ( t ) \| = \mathcal { O } \left( \Delta t ^ { 2 } \right) ,\tag{42}
$$

where $\| \cdot \|$ denotes the spectral norm (Yoshida and Miyato 2017). The spectral norm of a matrix A is defined as

$$
\| A \| = \operatorname* { m a x } _ { \xi \neq 0 } { \frac { \| A \xi \| _ { 2 } } { \| \xi \| _ { 2 } } } ,\tag{43}
$$

where $\xi$ is a non-zero vector that A can act on.

The 4-step Trotter product formula for commutators defined in Equation (20) can also be utilized for the approximation, which reduces the implementation of $W ( t )$ from 8 exponential terms to 4. This realizes a more eficient implementation, which generates

$$
W ( t ) = e ^ { i \beta D _ { V _ { t } } } e ^ { i \alpha K } e ^ { - i \beta D _ { V _ { t } } } e ^ { - i \alpha K } ,\tag{44}
$$

where α and $\beta$ satisfy $\alpha \beta = \Delta t$ according to Equation (20). Thus, we can employ

$$
\alpha = \frac { 2 L } { \pi N } \sqrt { \frac { \Delta t } { d } } , \qquad \beta = \frac { \pi N } { 2 L } \sqrt { d \Delta t } .\tag{45}
$$

The corresponding temporal discretization error $| | U ( t _ { 1 } , t ) - W ( t ) | |$ satisfies

$$
\begin{array} { r } { | | U ( t _ { 1 } , t ) - W ( t ) | | = \mathcal { O } \left( \Delta t ^ { 3 / 2 } \right) . } \end{array}\tag{46}
$$

## 3.1.2 Phase kickback framework-based simulation

For the temporal discretization interval $[ t , t _ { 1 } ]$ , each exponential term in the corresponding approximation operator $W ( t )$ defined in Equation (44) can be considered as an oracle. These oracles are implemented sequentially in the quantum circuit, as shown in Figure 2. Here, $n _ { 1 } , \ldots , n _ { d }$ denote the number of qubits used to represent each spatial dimension, and all of these $n = n _ { 1 } + \cdot \cdot \cdot + n _ { d }$ qubits are defined as address qubits. For the $j { \mathrm { - t h } }$ spatial dimension, $N _ { j } = 2 ^ { n _ { j } }$ basis states, i.e., address states, are provided, and each address state corresponds to a discretized spatial grid point as described in Equation (26). In addition, m denotes the number of ancilla qubits used for data encoding.

![](images/3e13de766265562dfe1d5ff37e4582583526c63e11ed45a0b3021b2e34863c3c.jpg)  
Fig. 2 Quantum simulation of $W ( t )$ using the 4-step Trotter product formula, where $\operatorname { a d d } _ { j }$ denotes the address qubits corresponding to the j-th spatial dimension, and anc denotes the ancilla qubits

Each oracle is suggested to be executed using the phase kickback framework (Layden et al. 2025; Cleve et al. 1998). Since $D _ { V _ { t } }$ is diagonal, we incorporate QROM into the phase kickback framework to encode the diagonal entries of $D _ { V _ { t } }$ when implementing the oracle $e ^ { i \beta D _ { V _ { t } } }$ . Figure $\mathrm { 3 ( a ) }$ presents the details of the phase kickback simulation of $\bar { e } ^ { i \beta D _ { V _ { t } } }$ , where the $\mathrm { Q R O M } ( D _ { V _ { t } } )$ oracle is used to encode all diagonal entries of $D _ { V _ { t } }$ using the structure shown in Figure 1.

For the $2 ^ { n } \times 2 ^ { n }$ diagonal matrix $D _ { V _ { t } }$ defined in Equation (30), the phase kickback framework is applied over all n address qubits across all spatial dimensions. Each diagonal entry corresponds to a basis state $| k \rangle = | k _ { 1 } \rangle \otimes \dotsb \otimes | k _ { d } \rangle$ , which is defined in Equation (28). For the l-th min-max normalized diagonal entry $V _ { t } ^ { l } \in [ 0 , 1 ]$ in $D _ { V _ { t } }$ we define its m-bit approximation as $0 . b ( t , l , 1 ) \cdots b ( t , l , m )$ . Let $| s _ { t } ^ { l } \rangle$ denote states $| b ( t , l , 1 ) \rangle \cdot \cdot \cdot | b ( t , l , m ) \rangle$ , the QROM oracle encodes the approximation

![](images/18b716e9954d17be7500e068c771ce0da0e980100e1475a04ed1ed278fb97404.jpg)  
(a) The simulation of $e ^ { i \beta D _ { V _ { t } } }$

![](images/811cfd5b6753ad4460b40b7fa7dda74acf5fb0c2db9a4b335a362c99bef68e8d.jpg)  
(b) The simulation of $e ^ { i \alpha K }$ in the j-th dimension  
Fig. 3 The QROM-based phase kickback framework for the simulation of $\cdot _ { e ^ { i \beta D } V _ { t } }$ , and the framework for the simulation of $e ^ { i \alpha K }$ in the j-th spatial dimension, where add<sub>j</sub> denotes the address qubits corresponding to the j-th spatial dimension, anc denotes the i-th ancilla qubit, and $\gamma = \alpha / 2$

$$
\tilde { V } _ { t } ^ { l } = \sum _ { k = 1 } ^ { m } b ( t , l , k ) 2 ^ { - k } ,\tag{47}
$$

into the ancilla qubits conditioned on the corresponding address state l . This realizes the state transformation

$$
| l \rangle | 0 \rangle ^ { \otimes m } \xrightarrow { \mathrm { \ Q R O M } } | l \rangle | s _ { t } ^ { l } \rangle .\tag{48}
$$

Then, an $\mathrm { R Z } ( \beta / 2 ^ { k } )$ gate is applied to each ancilla qubit k (see Figure 3), jointly encoding the exponential $e ^ { i \beta \tilde { V } _ { t } ^ { l } }$ , which realizes

$$
| l \rangle | | s _ { t } ^ { l } \rangle \xrightarrow { \mathrm { R Z } } e ^ { i \beta \tilde { V } _ { t } ^ { l } } | l \rangle | s _ { t } ^ { l } \rangle .\tag{49}
$$

Finally, the adjoint of the QROM oracle is employed to uncompute the ancilla qubits to the initial states, which is expressed as

$$
e ^ { i \beta \tilde { V } _ { t } ^ { l } } | l \rangle | s _ { t } ^ { l } \rangle \xrightarrow { \mathrm { Q R O M } ^ { \dagger } } e ^ { i \beta \tilde { V } _ { t } ^ { l } } | l \rangle | 0 \rangle ^ { \otimes m } .\tag{50}
$$

Thus, the phase kickback framework realizes the transformation over all basis states

$$
\sum _ { k } | k \rangle | 0 \rangle ^ { \otimes m } \xrightarrow { \mathrm { \ p h a s e ~ k i c k { b a c k } } } e ^ { i \beta D _ { V _ { t } } } \sum _ { k } | k \rangle | 0 \rangle ^ { \otimes m } .\tag{51}
$$

According to Equation (35), for the $N ^ { d } \times N ^ { d }$ kinetic operator $K ,$ , to simulate the evolution operator $e ^ { i \alpha K }$ , the phase kickback framework corresponding to $D _ { K }$ is implemented independently to each spatial dimension. The implementation of $e ^ { i \alpha K }$ corresponding to the j-th spatial dimension is shown in Figure 3(b), where $\gamma = \alpha / 2$

Therefore, the simulation among all spatial dimensions performs

$$
\sum _ { k _ { 1 } , \ldots , k _ { d } } | k _ { 1 } \rangle \cdot \cdot \cdot | k _ { d } \rangle | 0 \rangle ^ { \otimes m } \xrightarrow { \mathrm { \ p h a s e ~ k i c k b a c k } } \sum _ { k _ { 1 } , \ldots , k _ { d } } e ^ { i \alpha K } | k _ { 1 } \rangle \cdot \cdot \cdot | k _ { d } \rangle | 0 \rangle ^ { \otimes m } .\tag{52}
$$

## 3.2 QNN-based phase kickback framework for wavefunction flow simulation

Although QROM is widely used in data encoding applications, for n address qubits, it requires $\Theta ( 2 ^ { n } )$ CNOT gates (Barenco et al. 1995) for implementation, facing exponentially increasing computational complexity. To address this problem, we propose a QNN circuit for wavefunction flow simulation, which replaces QROM in the phase kickback framework with a trained QNN for data encoding. The proposed QNN mimics the structure of QROM by employing controlled rotation gates instead of multi-controlled X gates, which reduces the number of CNOT gates to $\Theta ( n )$ . The structure of the QNN-based phase kickback framework to implement the potential evolution operator is illustrated in Figure 4(a), and the framework for implementing the kinetic evolution operator in the j-th spatial dimension is shown in Figure 4(b).

![](images/9b2e1c642948f20683f15903d6addc9cb387e71b04c27a7d48d84bbe09316c6e.jpg)  
(a) The simulation of $e ^ { i \beta D _ { V _ { t } } }$

![](images/03442649184e3ae102f479e2e21ab276856435556d55a2e1be44698eb243c3c5.jpg)  
(b) The simulation of $e ^ { i \alpha K }$ in the j-th dimension  
Fig. 4 The QNN-based phase kickback framework for the simulation of $e ^ { i \beta D _ { V _ { t } } }$ , and the framework for the simulation of $e ^ { i \alpha { \tilde { K } } }$ corresponding to the j-th spatial dimension, where $\gamma = \alpha / 2$

We first propose a time-independent QNN circuit for the phase kickback framework of the kinetic evolution operator $e ^ { i \alpha K }$ , which does not change over time. Then we introduce a QNN circuit for the phase kickback framework of the time-dependent potential evolution operator $e ^ { i \beta D _ { V _ { t } } }$ by integrating a time oracle into the time-independent QNN circuit.

## 3.2.1 Time-independent QNN circuit for kinetic evolution operators

Consider the simulation of the kinetic evolution operator $e ^ { i \alpha K }$ corresponding to one spatial dimension; QROM encodes the l-th diagonal entry $D _ { K } ^ { l }$ of $D _ { K }$ by loading its binary fractional representation $0 . b ( l , 1 ) \cdots b ( l , m )$ into the m ancilla qubits as basis states. For the q-th ancilla qubit, the resulting state is $| b ( l , q ) \rangle$ after the implementation of the QROM oracle. To replace QROM, we propose a QNN circuit, which learns the map from the address state l to the corresponding m ancilla states $| b ( l , 1 ) \rangle \cdots | b ( l , m ) \rangle$ . The QNN circuit encodes $0 . b ( l , 1 ) \cdots b ( l , m )$ into the m ancilla qubits by loading the predicted probabilities $\hat { b } ( l , 1 ) , \dots , \hat { b } ( l , m )$ through minimizing the approximation error. Specifically, the trained QNN performs the map

$$
\begin{array} { r } { | l \rangle | 0 \rangle ^ { \otimes m } \xrightarrow { \mathrm { \tiny ~ Q N N } } | l \rangle | \hat { \Psi } ( l ) \rangle . } \end{array}\tag{53}
$$

Here $\hat { b } ( l , k ) = P \Big ( k \mathrm { - t h ~ a n c i l l a ~ q u b i t = 1 } \Big | | \hat { \Psi } ( l ) \rangle \Big )$ denotes the predicted probability of k-th ancilla qubit that is measured in state 1 given the state $| \hat { \Psi } ( l ) \rangle$ generated by the trained QNN.

The training of the QNN can be viewed as a binary classification problem in machine learning for each ancilla qubit. Therefore, the loss function is defined using binary cross-entropy loss

$$
L _ { K } = - \sum _ { i = 1 } ^ { N } \sum _ { h = 1 } ^ { m } \Big [ b ( i , h ) \log \hat { b } ( i , h ) + \big ( 1 - b ( i , h ) \big ) \log \big ( 1 - \hat { b } ( i , h ) \big ) \Big ] ,\tag{54}
$$

where N denotes the number of diagonal entries in the matrix $D _ { K }$ . After training, the predicted state $| \hat { \Psi } ( l ) \rangle$ is employed to encode the approximation of $D _ { K } ^ { l }$ into the ancilla qubits by the trained QNN circuit.

The time-independent QNN in each spatial dimension has repeated layers implemented in sequence, which is shown in Figure 5. Each layer contains Rot oracles and controlled-Rot oracles, which is illustrated in Figure 6.

$$
\begin{array}{c} { \begin{array} { r l } & { { \mathrm { a d d } } _ { j } : | k _ { j } \rangle \cdot \neq \overbrace { \int _ { \begin{array} { c } { 0 } \\ { \mathrm { ~ a s } } \end{array} } \left[ \mathrm { Q N N } ( D _ { K _ { j } } ) \right] } ^ { n _ { j } } = \overbrace { \begin{array} { c } { { \mathrm { a d d } } _ { j } : | k _ { j } \rangle \cdot \neq { \sqrt { \left[ { \mathrm { L a y e r } } \right] } } } \\ { \left[ \mathrm { ~ a s c } : ~ | 0 \rangle \cdot \neq { \sqrt { \left[ { \mathrm { L a y e r } } \right] } } \right]} \end{array} } ^ { { \mathrm { a d d } } _ { j } : | k _ { j } \rangle \cdot \overbrace { \left[ { \mathrm { L a y e r } } \right] } ^ { n _ { j } } - \overbrace { \left[ { \mathrm { L a y e r } } \right] } ^ { \left[ { \mathrm { L a y e r } } \right] } - \overbrace { \left[ { \mathrm { L a y e r } } \right] } ^ { \left[ { \mathrm { L a y e r } } \right] } - \overbrace { \left[ { \mathrm { L a y e r } } \right] } ^ { \left[ { \mathrm { L a y e r } } \right] } } \end{array} }  } \\ & { { \mathrm { ~ a s c } } : ~ | 0 \rangle \cdot \neq \overbrace { \left[ { \mathrm { ~ L a y e r } } \right] } ^ { \left[ \mathrm { L a } _ { j } \right] } = \overbrace { { \left[ { \mathrm { ~ L a y e r } } \right] } } ^ { \left[ \mathrm { ~ L a y e r } \right] } . } \end{array} 
$$

Fig. 5 The QNN circuit in the phase kickback framework for the simulation of $e ^ { i \alpha K }$ corresponding to the $j \cdot$ -th spatial dimension  
![](images/bed0212ae7e71243f355ebe327f72c790a2f39e6a6c2ae76ef7241fc16c24ce0.jpg)  
Fig. 6 The QNN layer circuit in the phase kickback framework for the simulation of $e ^ { i \alpha K }$ corresponding to the j-th spatial dimension, where add $\ l _ { j , i }$ denotes the i-th address qubit corresponding to the j-th spatial dimension

Inside each layer, m Rot oracles $\operatorname { R o t } ( \theta _ { 0 , 1 } ) , \cdot \cdot \cdot , \operatorname { R o t } ( \theta _ { 0 , m } )$ are applied to the ancilla qubits in the beginning. Then, a controlled Rot oracle is applied between each address qubit and ancilla qubit. The address qubits are controls, and ancilla qubits serve as targets. Each Rot oracle consists of three rotation gates, and each controlled-Rot gate contains three controlled rotation gates, as shown in Figure 7.

![](images/cc844a3af09d7bb9a009565a4ee469c9747ae74ae3f602de89e38a2b1dc06bdf.jpg)  
(a) The structure of a Rot oracle

![](images/f7aa86bd09fba907752e320ce9132230644d46ec2703f4b2162396e24a0b35c9.jpg)  
(b) The structure of a controlled-Rot oracle  
Fig. 7 Circuits of Rot and controlled-Rot oracles, $\phi ,$ ω and ψ denote three angle parameters in each oracle

## 3.2.2 Time-dependent QNN circuit for potential evolution operators

QROM encodes diagonal entries of the $N ^ { d } \times N ^ { d }$ diagonal matrix $D _ { V _ { t } }$ at each time t, respectively, where $N ^ { d } = 2 ^ { n }$ denotes the number of address states provided by all n address qubits of d dimensions jointly. We propose a time-dependent QNN circuit that can encode the diagonal operator $D _ { V _ { t } }$ at time t into the phase kickback framework by incorporating a time register oracle. For a time step h and the corresponding time $t _ { h }$ , given the l-th diagonal entry $D _ { V _ { t _ { h } } } ^ { l }$ and its binary fractional representation $0 . 6 ( t _ { h } , l , 1 ) \cdot \cdot \cdot b ( t _ { h } , l , m )$ , the QNN learns the map from the address state l to probabilities $b ( t _ { h } , l , 1 ) , \cdot \cdot \cdot , b ( t _ { h } , l , m )$ . This generates

$$
| l \rangle | 0 \rangle ^ { \otimes m } \xrightarrow { \mathrm { \tiny ~ Q N N } } | l \rangle | \hat { \Psi } ( t _ { h } , l ) \rangle .\tag{55}
$$

Let $\hat { b } ( t _ { h } , l , k ) = P \Big ( k \mathrm { - t h ~ a n c i l l a ~ q u b i t } = 1 \Big | | \hat { \Psi } ( t _ { h } , l ) \rangle \Big )$ denote the predicted probability of k-th ancilla qubit measured in state 1 for the state $| \hat { \Psi } ( t _ { h } , l ) \rangle$ generated by the trained QNN. To learn the encoding of all diagonal entries over all time steps, the loss function for QNN training is defined as

$$
{ \cal L } _ { \cal F } = - \sum _ { k = 1 } ^ { N _ { t } } \sum _ { i = 1 } ^ { N ^ { d } } \sum _ { j = 1 } ^ { m } \left[ b ( t _ { k } , i , j ) \log \hat { b } ( t _ { k } , i , j ) + \left( 1 - b ( t _ { k } , i , j ) \right) \log \left( 1 - \hat { b } ( t _ { k } , i , j ) \right) \right] ,\tag{56}
$$

where $N _ { t }$ denotes the number of discretized time intervals. After training, the predicted state $| \hat { \Psi } ( t _ { h } , l ) \rangle$ is employed to encode the approximation of $D _ { V _ { t _ { h } } } ^ { l }$ into the ancilla qubits by the trained QNN circuit.

For the time step h, the time-dependent QNN circuit operates on the address qubits across all spatial dimensions together with the ancilla qubits. The repeated layers are applied in sequence between a time register oracle and its adjoint oracle, which is shown in Figure 8.

![](images/f9efd2b87fe031f0d56a6f4d7819b64a39411216a615668e507a70f82e75eb19.jpg)  
Fig. 8 The time-dependent QNN circuit in the phase kickback framework for the simulation of $e ^ { i \beta D _ { V _ { t _ { h } } } }$

In each layer, m Rot oracles $\operatorname { R o t } ( \theta _ { 0 , 1 } ^ { h } ) , \cdot \cdot \cdot , \operatorname { R o t } ( \theta _ { 0 , m } ^ { h } )$ operate on the m ancilla qubits. A controlled-Rot oracle is applied between each address qubit and the ancilla qubit. Figure 9 illustrates the structure of the layer.

![](images/ecfe48ee12272bdf796f6f303380da54eb61ed4d92d6cc1e952dd1638215b80a.jpg)  
Fig. 9 The QNN layer circuit in the phase kickback framework for the implementation of $e ^ { i \beta D _ { V _ { t _ { h } } } }$

The Time oracle contains X gates applied to selected ancilla qubits according to the binary representation of the time step h. For example, when $h = 0$ , the states of the ancilla qubits are set as 0 ⊗<sup>m</sup>. Hence, no X gate operates on the ancilla qubits. When $h = 1$ , the ancilla qubits are prepared in the states $| 0 \rangle ^ { \otimes ( m - 1 ) } | 1 \rangle$ . Thus, an X gate is applied to the last ancilla qubit in oracle Time(1). Examples of the Time oracle are shown in Figure 10.

## 3.2.3 Resource analysis of QROM and QNN circuits

To compare the circuit resources required for QROM and our proposed QNN simulation directly, we analyze the number of CNOT gates required in QROM and QNN circuits with the same number of address qubits and ancilla qubits.

For a QROM circuit with n address qubits and m ancilla qubits, whose structure is the same as shown in Figure 1, the number of data entries to be encoded in the quantum register is $2 ^ { n }$ , corresponding to all address states. In addition, each data entry corresponds to m binary bits that are encoded in ancilla qubits. Therefore, the number of multi-controlled X gates in the QROM circuit is $2 ^ { n } m$

![](images/e2ae85970b70373df433188f2d260b08f488aeea0470400ac597142a30cd9aa2.jpg)  
(a) The structure of Time(1)

![](images/965b574bfc82449b8074f1aa96154ed2531e5fadcf4b41e55abe0c444e1927d5.jpg)  
(b) The structure of Time(3)  
Fig. 10 Examples of the Time oracles

According to the multi-controlled X decomposition method without ancilla qubits (Barenco et al. 1995), a multi-controlled X gate with n control qubits consists of $3 \cdot 2 ^ { n - 1 } - 4 \mathrm { C N O T }$ gates. Thus, the number of CNOT gates required for the QROM circuit is $( 3 \cdot 2 ^ { n - 1 } - 4 ) \cdot 2 ^ { n } \cdot m = ( 3 \times 2 ^ { 2 n - 1 } - 2 ^ { n + 2 } ) m$ . Therefore, for fixed m and l, the number of CNOT gates scales as

$$
N _ { \mathrm { Q R O M } } = \Theta \left( 4 ^ { n } \right) ,\tag{57}
$$

resulting in exponential growth with the increase of $n .$

For our QNN circuit containing l layers with the same number of address qubits and ancilla qubits, whose layer structure is shown in Figure 6, mn controlled-Rot oracles are required in each layer. Since each controlled-Rot oracle consists of 3 controlled rotation gates, and each controlled rotation gate requires 2 CNOT gates for decomposition (Barenco et al. 1995), the total number of CNOT gates in each QNN layer is 6mn. Thus, the number of CNOT gates required for the QNN circuit is 6mnl. Therefore, for fixed m and l, the number of CNOT gates scales as

$$
N _ { \mathrm { Q N N } } = \Theta \left( n \right) .\tag{58}
$$

This shows that the number of CNOT gates grows linearly as n increases in the QNN circuit.

From Equations (57) and (58), it is clear that our proposed QNN circuit requires significantly fewer circuit resources than the QROM circuit with a large value of n. This is also numerically evaluated in Subsection 4.4.2.

## 4 Results and discussion

## 4.1 Reference method

Given the initial distribution, to evaluate the performance of our approach, the final probability distributions generated by the wavefunction simulation are compared with the reference final distribution obtained from the corresponding conventional flow model, i.e., the solution of the continuity equation defined in Equation (8).

Given the initial distribution $p _ { 0 } ( x )$ and the corresponding wavefunction $\Psi _ { 0 } ( x ) =$ $\sqrt { p _ { 0 } ( x ) }$ , for the evolution time $T ,$ , the final wavefunction $\Psi _ { T } ( x )$ generated by the wavefunction flow simulation produces the final probability distribution through $p _ { T } ( x ) = | \Psi _ { T } ( x ) | ^ { 2 }$ . Here, $x = ( x _ { 1 } , \ldots , x _ { d } ) \in \mathbb { X } _ { N } ^ { d }$ denotes a d-dimensional grid point.

The reference final distribution is generated through the numerical solution of the continuity equation

$$
\frac { \partial } { \partial t } p _ { t } ( x ) + \boldsymbol { \nabla } \cdot \left[ \boldsymbol { v } _ { t } ( \boldsymbol { x } ) p _ { t } ( \boldsymbol { x } ) \right] = 0 ,\tag{59}
$$

where the velocity field is conservative and is generated by the potential function $v _ { t } ( x ) = \nabla V _ { t } ( x )$

If the dimension of $v _ { t }$ is 1, the continuity equation becomes

$$
\frac { \partial } { \partial t } p _ { t } ( x ) + \frac { \partial } { \partial x } ( v _ { t } ( x ) p _ { t } ( x ) ) = 0 .\tag{60}
$$

If $v _ { t }$ is multi-dimensional, the continuity equation can be written as

$$
\frac { \partial } { \partial t } p _ { t } ( x ) + \sum _ { i = 1 } ^ { d } \frac { \partial } { \partial x _ { i } } \left( v _ { t , i } ( x ) p _ { t } ( x ) \right) = 0 .\tag{61}
$$

Starting from the initial distribution $p _ { 0 } ( x )$ , the spatial derivative terms in the continuity function are obtained using the Fourier spectral method, and the corresponding equation is then solved from $t = 0$ to $t = T$ using the RK45 method. This generates the reference final distribution $p _ { T } ^ { \mathrm { r e f } } ( x )$

## 4.2 Experimental setup

## 4.2.1 Simulation environment and settings

All quantum simulation-based experiments were conducted using Python 3.12. The quantum software stack comprised Xanadu’s Pennylane 0.45.1 and pennylanelightning 0.45.0. Quantum circuits were executed using the lightning.gpu backend on an NVIDIA Tesla T4 GPU.

The RK45 solver used to generate reference solutions of the continuity equation in Equation (8) is provided by SciPy, and both relative and absolute tolerances are 1e-12.

The one-dimensional and two-dimensional wavefunction flow simulations were implemented in our experiments. For the one-dimensional wavefunction flow simulation circuit, we set the number of address qubits to 4, which is capable of representing $2 ^ { 4 } = 1 6$ discretized spatial points in total. In addition, 10 ancilla qubits were used to encode the data entries of potential and kinetic operators into the quantum circuit. We also employed 3 address qubits for each spatial dimension when implementing the two-dimensional wavefunction flow simulation, corresponding to $2 ^ { 3 } = 8$ discretized spatial grid points per dimension. The number of ancilla qubits is 12.

In our experiments, the uniform distribution and a periodic exponential distribution given by

$$
p ( x ) = e ^ { - K \sin ^ { 2 } ( \pi x / L ) }\tag{62}
$$

were chosen as the initial distribution $p _ { 0 } ( x )$ for the wavefunction flow simulation and reference solutions. In the periodic exponential distribution, K is a constant, and we set $K = 1$ in our experiments. Both distributions are normalized so that their total probability is 1, and the corresponding initial density is represented as $\rho _ { 0 } ( x )$

## 4.2.2 Evaluation metrics

Let $x ^ { ( i ) }$ denote the i-th discretized spatial point in the wavefunction flow simulation, and let $x _ { \mathrm { r e f } } ^ { ( j ) }$ denote the j-th discretized spatial point in the reference method. Let $p _ { T } ( x ^ { ( i ) } )$ represent the final probability at $x ^ { ( i ) }$ generated by the wavefunction flow simulation, and let $p _ { T } ^ { \mathrm { r e f } } ( x _ { \mathrm { r e f } } ^ { ( j ) } )$ denote the reference final probability at $x _ { \mathrm { r e f } } ^ { ( j ) }$ . Denote

$$
\rho _ { T } ( x ^ { ( i ) } ) = p _ { T } ( x ^ { ( i ) } ) / ( \Delta x ) ^ { d }
$$

as the corresponding wavefunction flow simulation-generated final probability density, and denote

$$
\rho _ { T } ^ { \mathrm { r e f } } ( x _ { \mathrm { r e f } } ^ { ( j ) } ) = p _ { T } ^ { \mathrm { r e f } } ( x _ { \mathrm { r e f } } ^ { ( j ) } ) / ( \Delta x _ { \mathrm { r e f } } ) ^ { d }
$$

as the corresponding reference final probability density. Here, $\Delta x$ is the spatial discretization interval in each dimension for wavefunction flow simulation, and $\Delta x _ { \mathrm { r e f } }$ denotes the spatial discretization interval employed in the reference method. Let N denote the number of discretized spatial grid points in the wavefunction flow simulation. To implement the evaluation, the reference final probability density is linearly interpolated onto the spatial grids of the wavefunction flow simulation. Let the interpolated reference final probability density at $\boldsymbol { x } ^ { ( i ) }$ be represented by $\rho _ { T } ^ { \mathrm { r e f } } ( x ^ { ( i ) } )$ .

To evaluate the accuracy of the wavefunction flow simulation, we employ four error metrics: the root mean square error (RMSE), relative maximum error (RME), KL divergence $( D _ { \mathrm { K L } } )$ , and Wasserstein-1 distance $( W _ { 1 } )$

• Root mean square error (RMSE)

$$
\mathrm { R M S E } = \sqrt { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( \rho _ { T } ( x ^ { ( i ) } ) - \rho _ { T } ^ { \mathrm { r e f } } ( x ^ { ( i ) } ) \right) ^ { 2 } } .\tag{63}
$$

• Relative maximum error (RME), which measures the maximum diference between probability densities $\rho _ { T } ( x )$ and $\rho _ { T } ^ { \mathrm { r e f } } ( x )$ , normalized by the maximum value of the reference final probability density $\rho _ { T } ^ { \mathrm { r e f } } ( x )$

$$
\mathrm { R M E } = \frac { \operatorname* { m a x } _ { i } \left| \rho _ { T } ( x ^ { ( i ) } ) - \rho _ { T } ^ { \mathrm { r e f } } ( x ^ { ( i ) } ) \right| } { \operatorname* { m a x } _ { i } \rho _ { T } ^ { \mathrm { r e f } } ( x ^ { ( i ) } ) } .\tag{64}
$$

• Kullback-Leibler (KL) divergence, which measures the distance between the normalized final density $\tilde { \rho } _ { T } ( x )$ and the normalized reference final probability density $\widetilde { \rho } _ { T } ^ { \mathrm { r e f } } ( x )$

$$
D _ { \mathrm { K L } } = \sum _ { i = 1 } ^ { N } \tilde { \rho } _ { T } ^ { \mathrm { r e f } } ( x ^ { ( i ) } ) \ln \frac { \tilde { \rho } _ { T } ^ { \mathrm { r e f } } ( x ^ { ( i ) } ) } { \tilde { \rho } _ { T } ( x ^ { ( i ) } ) } .\tag{65}
$$

where

$$
\tilde { \rho } _ { T } ( x ^ { ( i ) } ) = \frac { \rho _ { T } ( x ^ { ( i ) } ) } { \sum _ { j = 1 } ^ { N } \rho _ { T } ( x ^ { ( j ) } ) } , \quad \tilde { \rho } _ { T } ^ { \mathrm { r e f } } ( x ^ { ( i ) } ) = \frac { \rho _ { T } ^ { \mathrm { r e f } } ( x ^ { ( i ) } ) } { \sum _ { j = 1 } ^ { N } \rho _ { T } ^ { \mathrm { r e f } } ( x ^ { ( j ) } ) }
$$

• Wasserstein-1 distance, which measures the minimum cost of transporting the normalized final density $\tilde { \rho } _ { T } ( x )$ to $\widetilde { \rho } _ { T } ^ { \mathrm { r e f } } ( x )$

$$
W _ { 1 } = \operatorname * { m i n } _ { \gamma _ { i j } \geq 0 } \sum _ { i = 1 } ^ { N } \sum _ { j = 1 } ^ { N } \gamma _ { i j } | x ^ { ( i ) } - x ^ { ( j ) } | .\tag{66}
$$

Here

$$
\sum _ { j = 1 } ^ { N } \gamma _ { i j } = \tilde { \rho } _ { T } ( x ^ { ( i ) } ) , \sum _ { i = 1 } ^ { N } \gamma _ { i j } = \tilde { \rho } _ { T } ^ { \mathrm { r e f } } ( x ^ { ( j ) } ) ,
$$

and $\gamma _ { i j }$ denotes the amount of probability density moving from $\boldsymbol { x } ^ { ( i ) }$ to $x ^ { ( j ) }$

Smaller values of these error metrics indicate higher consistency between the final probability densities generated by the wavefunction flow simulation and the reference final probability densities, and demonstrate higher accuracy of the wavefunction flow simulation.

## 4.3 Training analysis of the QNN circuit

To implement the wavefunction flow simulation using a trained QNN-based phase kickback framework, the QNN circuits shown in Figures 5 and 8 must be trained first. The values of parameters in QNN circuits were initialized from a normal distribution $\mathcal { N } ( 0 , 1 )$ . The evolution time $T$ was set to 0.02 s, which was divided into 200 time steps. We chose a one-dimensional potential function

$$
V ( x , t ) = \cos ( \pi x / L + 0 . 7 5 t ) .\tag{67}
$$

We analyzed the convergence of the training process and compared the performance of diferent numbers of QNN layers and diferent loss functions.

## 4.3.1 Depth analysis

We conducted an experiment for the convergence analysis of the QNN training process with diferent numbers of QNN layers. For the simulation of the potential evolution operator $e ^ { i \beta D _ { V _ { t } } }$ , the QNN circuit shown in Figure 8 was employed for each time step. To simulate the kinetic evolution operator $e ^ { i \alpha K }$ , the QNN circuit shown in Figure 5 was implemented independently to each spatial dimension.

For the training of QNN circuits associated with the potential and kinetic evolution, the loss functions defined in Equations (54) and (56) were used, respectively. The Adam optimizer with a learning rate of 0.3 was employed in the training process. We recorded the training losses for the QNN circuits of both potential and kinetic evolution with diferent numbers of layers. The loss convergence curves are shown in

![](images/05cc93776fd22996a1e3dcf5cde48604ba35b8d5aef00869631f51ec153e0a39.jpg)

![](images/e4d1a51ddf01bf53853737b4076d24fa8ccb0610a8d90a6452ea859154544e9f.jpg)  
Fig. 11 The training losses for the potential and kinetic evolution QNN circuits with diferent numbers of layers l

Table 1 Final training loss values under diferent depths of QNN circuits
<table><tr><td>Number of layers</td><td>QNN circuit for potential evolution</td><td>QNN circuit for kinetic evolution</td></tr><tr><td>1</td><td>0.1712</td><td>0.1499</td></tr><tr><td>2</td><td>0.0421</td><td>0.0279</td></tr><tr><td>3</td><td>0.0039</td><td>0.0073</td></tr><tr><td>4</td><td>0.0001</td><td>0.0001</td></tr></table>

Figure 11. The final loss values for potential and kinetic evolution QNN circuits with diferent depths are shown in Table 1.

Figure 11 shows that the training losses for both the QNN circuits for potential and kinetic evolution decrease as the training step increases, demonstrating the stable training process and the convergence behavior. For both QNN circuits, as the number of layers increases, the convergence of loss becomes faster, and the final training loss becomes lower. These results indicate that a deeper QNN circuit is able to improve the performance of approximating QROM in the phase kickback frameworks.

In addition, as shown in Table 1, the final training losses of two QNN circuits for l = 1 are 0.1712 and 0.1499, respectively. When l increases to 4, the final training losses are both 0.0001. This indicates that the training loss becomes very small when the QNN depth reaches 4. In addition, as l increases, the final training loss decreases rapidly, demonstrating a rapid convergence of the loss with respect to l.

## 4.3.2 Loss function comparison

To explore the influence of diferent loss functions for QNN training, we compared three loss functions in the training process

• Loss function 1: Binary cross-entropy loss defined in Equations (54) and (56).

• Loss function 2: Mean squared error between the exact values of diagonal entries and the values generated by predicted probabilities obtained from the QNN circuit

$$
L _ { K } = \sum _ { i = 1 } ^ { N } \left( D _ { K } ^ { i } - \sum _ { h = 1 } ^ { m } \hat { b } ( i , h ) 2 ^ { - h } \right) ^ { 2 } ,
$$

$$
L _ { P } = \sum _ { k = 1 } ^ { N _ { t } } \sum _ { i = 1 } ^ { N } \left( D _ { V _ { t _ { h } } } ^ { i } - \sum _ { j = 1 } ^ { m } \hat { b } ( t _ { k } , i , j ) 2 ^ { - j } \right) ^ { 2 } .
$$

• Loss function 3: Fidelity-based loss between the normalized vectors based on exact potential and kinetic diagonal entries and those predicted by the QNN circuit

$$
L _ { K } = \frac { 1 } { d } \sum _ { j = 1 } ^ { d } \left( 1 - \left| \left. \psi _ { K } ^ { \mathrm { e x a c t } } \Big | \psi _ { K } ^ { \mathrm { p r e d } } \right. \right| ^ { 2 } \right) ,
$$

$$
L _ { P } = \frac { 1 } { N _ { t } } \sum _ { j = 1 } ^ { N _ { t } } \left( 1 - \left| \left. \psi _ { V _ { j } } ^ { \mathrm { e x a c t } } \Big | \psi _ { V _ { j } } ^ { \mathrm { p r e d } } \right. \right| ^ { 2 } \right) ,
$$

where $\psi _ { K } ^ { \mathrm { e x a c t } }$ and $\psi _ { V _ { j } } ^ { \mathrm { e x a c t } }$ denote the normalized vectors consisting of the exact diagonal entries of $D _ { K }$ and $D _ { V _ { t _ { i } } }$ , respectively. And $\psi _ { K } ^ { \mathrm { p r e d } }$ and $\psi _ { V _ { i } } ^ { \mathrm { p r e d } }$ denote the normalized vectors consisting of the predicted diagonal entries of $D _ { K } ^ { \check { } }$ and $D _ { V _ { t _ { j } } }$ generated by the QNN circuit.

For the simulation of potential and kinetic operators, the Adam optimizer with a learning rate of 0.3 is adopted, and the number of QNN layers is 2.

![](images/7f95b393e42553b441fff4f246239c80d5e07b6d1b0297f96e270946fb432521.jpg)

![](images/359aff1dd64313a1c8cf68ee87b5965d3343da1728e361122bedcc6cddda3061.jpg)  
Fig. 12 The training losses for the potential and kinetic evolution QNN circuits with diferent loss functions

Figure 12 shows the training loss curves for both QNN circuits using three loss functions. For both QNN circuits of potential and kinetic evolution, the training losses of all three loss functions converge steadily, and their final values reach approximately 0. Meanwhile, the convergence speed of Loss function 1 is lower than those of Loss functions 2 and 3. This indicates that Loss functions 2 and 3 outperform Loss function 1 for the training process of both QNN circuits according to the training speed.

However, when comparing the binary representations of the exact potential diagonal entries of $D _ { V _ { t _ { 0 } } }$ with the corresponding probabilities learned by QNN under diferent loss functions, diferent performances are exhibited. The results are shown in Figure 13. The mean distances and standard deviations across all probabilities associated with all diagonal entries of $D _ { V _ { t _ { 0 } } }$ under diferent loss functions are summarized in Table 2.

![](images/5d081d3b10bc0a80a23572713b77aa7fb5b4a2de86efb04e8abb15ed19e881b0.jpg)  
Fig. 13 The comparison between the binary representations of the exact potential diagonal entries of $D _ { V _ { t _ { 0 } } }$ and the corresponding probabilities learned by the QNN circuit under diferent loss functions. The y-axis represents the states of the input, where each state corresponds to a diagonal entry. The x-axis represents the ancilla qubit indices, where the i-th qubit index corresponds to the i-th binary bit of the diagonal entry

Table 2 The mean distances and standard deviations across all probabilities under diferent loss functions
<table><tr><td>Loss function</td><td>Mean distance</td><td>Standard deviation</td></tr><tr><td>Loss function 1</td><td>0.0001</td><td>0.0001</td></tr><tr><td>Loss function 2</td><td>0.3726</td><td>0.3645</td></tr><tr><td>Loss function 3</td><td>0.4027</td><td>0.3286</td></tr></table>

It is observed from the results that the predicted probabilities generated by the QNN circuit using Loss function 1 successfully approximate the exact binary bits of the diagonal entries, which are 0 and 1. However, the predicted probabilities generated by the QNN circuit using Loss functions 2 and 3 cannot match the corresponding binary representation, since massive probabilities stay far from 0 and 1. These indicate that, compared with the other two loss functions, the output states of the QNN circuits using the binary cross-entropy loss are precise and pure, which reduce entanglement significantly. In addition, as shown in Table 2, the mean distance and standard deviation under Loss function 1 are both 0.0001, which are significantly lower than those obtained under Loss functions 2 and 3. These results demonstrate that the binary cross-entropy loss outperforms the other two loss functions in the training of potential and kinetic evolution QNN circuits according to the binary bit approximation.

## 4.4 Comparison between QNN and QROM-based phase kickback frameworks for wavefunction flow simulation

To evaluate the performance of wavefunction flow simulation through our QNN-based phase kickback framework, we first compared its final probability densities with those obtained by the QROM-based phase kickback framework. Then we compared the CNOT gate cost of QROM and QNN circuits, as well as the corresponding phase kickback frameworks with diferent address qubits. The decomposition tools in PennyLane are used for counting the number of CNOT gates.

## 4.4.1 Accuracy comparison

We first conducted a comprehensive study of the accuracy performance of the wavefunction flow simulation using our trained QNN-based and QROM-based phase kickback frameworks under fixed potentials. To this end, given the same initial distribution, we evaluated the final probability densities generated by the two frameworks compared to the reference final densities, i.e., the results generated by the corresponding conventional flow model.

We set $L = 1$ , and the domain of each spatial dimension is $[ - L , L ]$ . We chose four periodic time-dependent potentials, two of which are one-dimensional,

$$
V _ { 1 } ( x , t ) = 3 \sin ( \pi x / L + 0 . 7 5 t ) ,
$$

$$
V _ { 2 } ( x , t ) = 3 \cos ( \pi x / L + 0 . 7 5 t ) .
$$

The other two potentials are two-dimensional,

$$
V _ { 3 } ( x , y , t ) = \sin \left( \frac { \pi } { L } ( x + y ) + 0 . 5 t \right) ,
$$

$$
V _ { 4 } ( x , y , t ) = \cos \left( \frac { \pi } { L } ( x + y ) + 0 . 5 t \right) .
$$

All these potentials are normalized to [0, 1] using the min-max normalization.

The QROM-based phase kickback frameworks for the simulation of potential and kinetic evolution operators are shown in Figures $\mathrm { 3 ( a ) }$ and 3(b), respectively. The evolution time T was set to 0.02 s and discretized into $N _ { t } = 2 0 0$ time steps. Thus, each time interval is $\Delta t = 1 e - 4$ . We recorded the final probability densities generated by the wavefunction flow simulation using two frameworks. Then we compared them with the corresponding reference final probability densities, which were obtained by solving Equations (60) and (61) using the RK45 method with the same values of T and $N _ { t }$ as those used in the wavefunction flow simulation. For the one-dimensional case, the number of discretized spatial grid points in the reference method is twice that employed in the wavefunction flow simulation. For the two-dimensional case, the number of grid points for each dimension in the reference method is the same as that in the wavefunction flow simulation. The comparison results with diferent potentials and initial distributions are shown in Figures 14, 15, 16, and 17. Tables 3 and 4 summarize the diferent error metrics between the final probability densities generated by the wavefunction flow simulation using QNN and QROM-based phase kickback frameworks, and the corresponding reference final probability densities.

![](images/7e80e9595f37375220bcfbb479532ee4de2527ec904be9cba1d1b3be8c2ebd8c.jpg)

![](images/72d108306a83c86a22b69b87eac4a4b1f720e90f0573d0cfd1f86a6facb9fcfb.jpg)

Fig. 14 Comparison of final probability densities generated by wavefunction flow simulation using QNN- and QROM-based phase kickback frameworks, and reference final densities under V<sub>1</sub>(x, t)  
![](images/8d6d4d93547fb9ba7e68651b6aea0cbbd6a19207944f10e023586fe3104eda52.jpg)

![](images/626c242437cf9df26b257592e8c0980886f3e73128c9fe0d238b2a02d904e557.jpg)  
Fig. 15 Comparison of final probability densities generated by wavefunction flow simulation using QNN and QROM-based phase kickback frameworks, and reference final densities under V (x, t)

We first evaluate the performance of the wavefunction flow simulation by comparing its results with those obtained from the flow model. As shown in Figures 14 and 15, the final probability densities produced by the wavefunction flow simulation using both QNN and QROM-based phase kickback frameworks under one-dimensional potentials closely match the corresponding reference final densities under uniform and periodic exponential initial distributions. This is because their curves almost overlap and the values at each discrete spatial point nearly coincide. In addition, Figures 16 and 17 exhibit consistent densities generated by wavefunction flow simulation using two frameworks and the reference method under two-dimensional potentials.

![](images/966be2fb3518e7deeaece3ef35372d48718ed1fb9e4c5aa00e5fd1d51fc3745a.jpg)

![](images/4578792b0c831416b43ea9561cb560588fca11c088a18fc1a09109dcef47118e.jpg)

![](images/f7df1b27a3deabc786ec68ff7680db5bf548c9af9d9efb363c1b6a94963a2fec.jpg)

![](images/4d92e5b5658104e453b22dd167c6a1b3de7e2532203f136927a5e63bb46d8cce.jpg)  
(a) Uniform initial distribution

![](images/10ecd31b703ca36339b71ce9c9918208c1d13181a06ad314a483a9c4f2cee7df.jpg)

![](images/75ee6b5052d54ca225d5ffb078e77c70cc854c9d1844a084095cb36094fa87e7.jpg)

![](images/5e04c94e8aeb15439d96a5834b9e5a103ed779a931baa414475ad2f1f1e2b781.jpg)  
(b) Periodic exponential initial distribution  
Fig. 16 Comparison of final probability densities generated by wavefunction flow simulation using QNN- and QROM-based phase kickback frameworks, and reference final densities under $V _ { 3 } ( x , y , t )$

Table 3 Errors between final probability densities generated by wavefunction flow simulation using QNN- and QROM-based phase kickback frameworks, and reference final probability densities under the uniform initial distribution
<table><tr><td colspan="5"></td><td colspan="4">QROM-based phase kickback</td></tr><tr><td>V</td><td>RMSE</td><td>RME</td><td> $D _ { \mathrm { K L } }$ </td><td> $W _ { 1 }$ </td><td>RMSE</td><td>RME</td><td> $D _ { \mathrm { K L } }$ </td><td> $W _ { 1 }$ </td></tr><tr><td>V1</td><td>5.965e-3</td><td>0.011</td><td>6.475e-5</td><td>1.119e-3</td><td>6.141e-3</td><td>0.012</td><td>6.677e-5</td><td>9.107e-4</td></tr><tr><td>V2</td><td>6.424e-3</td><td>0.014</td><td>6.443e-5</td><td>9.353e-4</td><td>6.381e-3</td><td>0.014</td><td>6.386e-5</td><td>9.191e-4</td></tr><tr><td>V3</td><td>1.580e-4</td><td>0.013</td><td>6.028e-5</td><td>7.397e-3</td><td>1.550e-4</td><td>0.011</td><td>5.306e-5</td><td>7.202e-3</td></tr><tr><td>V4</td><td>1.572e-4</td><td>0.011</td><td>5.548e-5</td><td>6.819e-3</td><td>1.568e-4</td><td>0.011</td><td>5.499e-5</td><td>6.818e-3</td></tr></table>

According to Tables 3 and 4, the RMSE values corresponding to the one-dimensional potentials are on the order of $1 0 ^ { - 3 }$ , while those for the two-dimensional potentials are on the order of $1 0 ^ { - 4 }$ . These results indicate very small density errors between the wavefunction flow simulation and the reference method. The RME measures the maximum diference of all discretized spatial points between the wavefunction flow simulation and the reference method. All RME values are not greater than 0.022, which shows that local density errors at all spatial points are well controlled.

We take small KL divergence values to indicate reliable density matching, and all the values in the two tables are below $2 \times 1 0 ^ { - 4 }$ . In addition, all the values of $W _ { 1 }$ in the two tables are below 0.02. These also indicate the close agreement between the densities. All these results demonstrate that the phase kickback framework-based wavefunction flow simulation generates the expected evolution of the wavefunction flow theory and matches the results of the flow model, which is obtained by solving the corresponding continuity equation.

![](images/7dd1622e5db7bc4a03a06af0f9d22249e1a3d10f94462773d270144cd8d420de.jpg)

![](images/21c13582d788fdf4d41f27251f94393d78aebdb3b65f471b4cd72ab139773cdc.jpg)

![](images/daa7177cd28ece06b8a2b0d719db8c400b60d76272d8dbea094cf4a74cb6157e.jpg)  
(a) Uniform initial distribution

![](images/c0622c31c85a4bf71ccedeb1fa915286633918ac8ec2aaa2bdcebdd495d19864.jpg)

![](images/cc05ce3cae9ce68049bf35d30ffcf14b755fac2e50f16a66d1bd6f731478644e.jpg)

![](images/f9f7f9a5eb8e5cfcd7a41181609a257e74da918f831f03dc08b2535b1cf60d39.jpg)

![](images/882a69a380463bbfa6cbe7db496c43b4c95c3b9dab6c0c6e04e29b58a452418d.jpg)  
(b) Periodic exponential initial distribution

![](images/6a2ff5716632f42aeb1a680dda6e7cb37a35b8eafc9e9793413afb4745bf25c2.jpg)  
Fig. 17 Comparison of final probability densities generated by wavefunction flow simulation using QNN- and QROM-based phase kickback frameworks, and reference final densities under $V _ { 4 } ( x , y , t )$

Table 4 Errors between final probability densities generated by wavefunction flow simulation using QNN- and QROM-based phase kickback frameworks, and reference final densities under the periodic exponential initial distribution
<table><tr><td rowspan="2">V</td><td colspan="4">QNN-based phase kickback</td><td colspan="4">QROM-based phase kickback</td></tr><tr><td>RMSE</td><td>RME</td><td> $D _ { \mathrm { K L } }$ </td><td> $W _ { 1 }$ </td><td>RMSE</td><td>RME</td><td> $D _ { \mathrm { K L } }$ </td><td> $W _ { 1 }$ </td></tr><tr><td>V1</td><td>6.339e-3</td><td>0.012</td><td>7.010e-5</td><td>8.843e-4</td><td>6.234e-3</td><td>0.013</td><td>6.724e-5</td><td>7.669e-4</td></tr><tr><td>V2</td><td>7.154e-3</td><td>0.013</td><td>5.382e-5</td><td>1.444e-3</td><td>7.852e-3</td><td>0.013</td><td>6.704e-5</td><td>1.382e-3</td></tr><tr><td>V3</td><td>2.344e-4</td><td>0.013</td><td>9.852e-5</td><td>7.870e-3</td><td>2.346e-4</td><td>0.013</td><td>9.826e-5</td><td>7.870e-3</td></tr><tr><td>V4</td><td>3.465e-4</td><td>0.021</td><td>1.691e-4</td><td>1.051e-2</td><td>3.525e-4</td><td>0.022</td><td>1.720e-4</td><td>1.036e-2</td></tr></table>

Then we compare the accuracy of wavefunction flow using QNN- and QROM-based wavefunction flow simulation. According to Figures 14, 15, 16, and 17, for all four potentials, the final probability densities produced by the wavefunction flow simulation through the QNN-based phase kickback framework closely match those obtained using the QROM-based phase kickback framework. In Table 3, for $V _ { 1 } ,$ the RMSE, RME, KL divergence, and Wasserstein-1 distance between the final probability density generated by the QNN-based phase kickback framework and the reference final probability density under uniform initial distribution are 5.965e-3, 0.011, 6.475e-5, and 1.119e-3, respectively. These values are very close to those generated by the QROM-based phase kickback framework, which are 6.141e-3, 0.012, 6.677e-5, and 9.107e-4. Similar behaviors are observed for V<sub>2</sub>, V<sub>3</sub>, and $V _ { 4 }$ under both initial distributions. All error metrics corresponding to wavefunction flow simulation using the QNN-based phase kickback framework and the QROM-based phase kickback framework show only slight diferences. These results demonstrate that the wavefunction flow simulation through the QNN-based phase kickback framework achieves accuracy comparable to that of the simulation using the QROM-based phase kickback framework.

## 4.4.2 Circuit resource comparison

We measured the number of CNOT gates used in (i) QNN and QROM circuits, (ii) QNN-based phase kickback and QROM-based phase kickback frameworks for the simulation of $e ^ { - i \beta D _ { V _ { t } } }$ corresponding to the potential function $V _ { 1 } ( x , t )$ with diferent numbers of address qubits. The number of ancilla qubits was set to 6. Figure 18 presents the trends of the CNOT gate counts with diferent numbers of address qubits. Detailed statistics for QNN and QROM circuits, as well as the corresponding phase kickback frameworks, are shown in Table 5.

![](images/17b220dea22de90002a11439fc7b75757568566b6030043e101d008260db9c33.jpg)  
Fig. 18 The number of CNOT gates in QNN and QROM circuits obtained from decomposition tools in PennyLane for the simulation of $e ^ { - i \beta \bar { D } _ { V _ { t } } }$ corresponding to V<sub>1</sub>(x, t)

Table 5 The number of CNOT gates in QNN and QROM circuits, and the corresponding phase kickback frameworks from decomposition tools in PennyLane
<table><tr><td></td><td colspan="2">QNN</td><td colspan="2">QROM</td></tr><tr><td>n</td><td>QNN circuit</td><td>Phase kickback framework</td><td>QROM circuit</td><td>Phase kickback framework</td></tr><tr><td>2</td><td>192</td><td>384</td><td>24</td><td>48</td></tr><tr><td>3</td><td>432</td><td>864</td><td>576</td><td>1152</td></tr><tr><td>4</td><td>960</td><td>1920</td><td>6080</td><td>12160</td></tr><tr><td>5</td><td>1680</td><td>3360</td><td>39424</td><td>78848</td></tr></table>

As shown in Figure 18 and Table $5 ,$ when $n = 2 .$ the numbers of CNOT gates in the QNN and QROM circuits to simulate $e ^ { - i \beta D _ { V _ { t } } }$ are 192 and 24, respectively. As n increases from 2 to $5 ,$ the number of CNOT gates in the QNN circuit increases to 1680. However, the number of CNOT gates in the QROM circuit increases exponentially to 39424, which is significantly higher than that of the QNN circuit.

In addition, the number of CNOT gates required for the QNN-based phase kickback framework is twice that of the single QNN circuit. The same applies to the QROM-based phase kickback framework, since the phase kickback framework includes a QROM or QNN circuit together with its adjoint.

Overall, these results demonstrate that as we introduce more address qubits to improve the precision of the wavefunction flow simulation, the QROM-based framework requires more circuit resources than the QNN-based framework. Therefore, our QNN-based framework provides a more eficient method for the implementation of flow models on quantum computers.

## 4.5 Error analysis

We investigate the errors in the phase kickback framework-based wavefunction flow simulation. The primary sources of quantum simulation errors include the approximation errors of the values encoded in the phase kickback framework, spatial discretization errors, and temporal discretization errors.

## 4.5.1 Phase kickback encoding error

To simulate the potential evolution operator $e ^ { - i \beta V _ { t } }$ using the phase kickback framework, each diagonal entry $V _ { j } ~ \in ~ [ 0 , 1 ]$ in $D _ { V _ { t } }$ must first be encoded in the ancilla register. For a phase kickback framework with m ancilla qubits, the corresponding approximation value $\tilde { V _ { j } }$ is encoded in the quantum circuit

$$
\tilde { V } _ { j } = \sum _ { k = 1 } ^ { m } b _ { k } 2 ^ { - k } ,\tag{68}
$$

where $0 . b _ { 1 } \cdots b _ { m }$ denotes the binary fractional representation of $V _ { j }$ .

Similarly, to simulate the kinetic evolution operator $e ^ { - i \alpha D _ { K } }$ , for any diagonal entry $K _ { j } \in [ 0 , 1 ]$ in the operator $D _ { K }$ , whose binary fractional representation is $0 . k _ { 1 } \cdots k _ { m } .$ the approximation value $\tilde { K _ { j } }$ is encoded in the circuit

$$
\tilde { K } _ { j } = \sum _ { i = 1 } ^ { m } k _ { i } 2 ^ { - i } .\tag{69}
$$

When m changes, the approximation errors $E _ { V } = | \tilde { V _ { j } } - V _ { j } |$ and $E _ { K } = | \tilde { K _ { j } } - K _ { j } |$ are expected to follow

$$
E _ { V } , E _ { K } \sim 2 ^ { - m } .\tag{70}
$$

We set the evolution time $T$ to 0.1 s, and the number of time steps to 100. The potential function was chosen to be

$$
V ( x , t ) = \sin ( \pi x / L + 0 . 7 5 t ) ,\tag{71}
$$

which is normalized to [0, 1] using the min-max normalization. The uniform distribution was chosen as the initial distribution. We evaluated the errors between the final densities generated by the wavefunction flow simulation and the reference final densities with diferent values of m. The results are shown in Figure 19.

![](images/4ead468648b46fe092c9829815ad8dab277e32d2fd1c804eadc14e0d1cca92e0.jpg)  
Fig. 19 Errors between the final probability densities generated by wavefunction flow simulation and the reference final probability densities with diferent numbers of ancilla qubits m

Increasing the number of ancilla qubits m leads to an increase in the number of binary fractional representation bits of the diagonal entries. According to Equation (70), the corresponding approximation error is expected to decrease. As shown in Figure 19, when the number of ancilla qubits increases from 6 to 14, the final density error decreases from 4.7e-02 to a converged value of 4.4e-03. This indicates that increasing the number of ancilla qubits improves the approximation precision of the phase kickback data encoding and enhances the accuracy of the wavefunction flow simulation.

## 4.5.2 Spatial discretization error

To perform the wavefunction flow simulation corresponding to the continuous Hamiltonian $\hat { H } _ { t }$ defined in the spatial domain $[ - L , L ]$ , the method described in Subsection 3.1.1 was employed for spatial discretization. The domain of each spatial dimension was first discretized into N points $\begin{array} { r } { \frac { 2 L } { N } \{ - N / 2 , - N / 2 + 1 , \dots , N / 2 - 1 \} } \end{array}$ . The kinetic operator corresponding to each spatial dimension was then discretized into N diagonal entries, and the potential operator was discretized into $N ^ { d }$ diagonal entries across all d dimensions. These diagonal entries are encoded in the phase kickback framework. In the wavefunction flow simulation circuit, spatial discretization depends on the number of address qubits. When the number of address qubits for each spatial dimension is n, the number of discretized points for each dimension is $N = 2 ^ { n }$ . Each discretized point corresponds to a basis state, into which the associated discretized potential and kinetic entries are encoded.

We set the evolution time $T$ to 0.02 s, and the number of time steps $N _ { t }$ to 200. The one-dimensional potential function was given by Equation (71), and the initial distribution was the uniform distribution. The number of address qubits changed from $2$ to $^ { 8 , }$ corresponding to $N = \left\lceil 4 , 8 , 1 6 , 3 2 , 6 4 , 1 2 8 , 2 5 6 \right\rceil$ discretized spatial points and discretization intervals $\Delta x = [ \dot { \frac { 2 } { 4 } } , \frac { 2 } { 8 } , \frac { 2 } { 1 6 } , \frac { 2 } { 3 2 } , \frac { 2 } { 6 4 } , \frac { 2 } { 1 2 8 } , \frac { 2 } { 2 5 6 } ]$

![](images/6ead22ed724edd66ee37568d8699503e95bc8a6c0fe91371bba9364242da37d9.jpg)  
Fig. 20 Errors between the final probability densities generated by wavefunction flow simulation and reference final probability densities with diferent ∆x

Figure 20 shows the RMSE between the final probability densities generated by the wavefunction flow simulation and the reference final probability densities with diferent discretization intervals. When the number of address qubits n increases from 2 to $5 ,$ the spatial discretization interval $\Delta x = 2 L / 2 ^ { n }$ decreases from 0.5 to 0.0625, while the RMSE decreases approximately exponentially from 4.670e-03 to 4.093e-05. This is consistent with the expected rapid decrease of the error, since we adopt Fourier spatial discretization and a smooth periodic initial distribution. As n continues to increase from 5 to $^ { 8 , }$ corresponding to a decrease in $\Delta x$ from 0.0625 to 0.0078125, the RMSE converges to a value of 1.075e-05. A further decrease of the spatial discretization interval will not lead to a lower error, since the temporal discretization error becomes dominant. These results demonstrate that finer discretized spatial intervals improve the accuracy of the wavefunction flow simulation.

## 4.5.3 Temporal discretization error

For the evolution time $T$ in the wavefunction flow simulation, the method described in Subsection 3.1.1 was employed for temporal discretization. The total time was divided into $N _ { t }$ time intervals $\left[ 0 , \Delta t , 2 \Delta t , \ldots , ( N _ { t } - 1 ) \Delta t \right]$ , where $\Delta t = T / N _ { t }$ denotes the length of each interval. At each discrete time point, the diagonal operator $D _ { V _ { t } }$ varies, leading to a diferent quantum simulation of $e ^ { - i \beta D _ { V _ { t } } }$ using the Trotter product formula. The diagonal operator $D _ { K }$ remains unchanged with time, making the simulation of $e ^ { - i \alpha D _ { K } }$ identical at diferent time steps.

We set the evolution time $T$ to 0.1 s, and the number of time steps $N _ { t }$ varied in $N _ { t } \in \{ 5 0 , 1 0 0 , 2 0 0 , 2 5 0 , 4 0 0 , 5 0 0$ , 800, 1000 , leading to diferent values of the discretized temporal interval $\Delta t = T / N _ { t }$ . The initial distribution was chosen to be the uniform distribution, and the potential function was defined as

$$
V ( x , t ) = ( 1 + 0 . 5 \sin { ( 0 . 5 t ) } ) \left( \sin ( \pi x / L ) \right) ,
$$

which is normalized to [0, 1] using the min-max normalization.

![](images/9da3b6f9328cc673e9c73356ccf97ab37061ad338c1decbd6d9a71cb7359d388.jpg)  
Fig. 21 Errors between the final probability densities generated by the wavefunction flow simulation and reference final probability densities with diferent $N _ { t }$ under diferent numbers of address qubits

Figure 21 exhibits the RMSE between the final probability densities generated by the wavefunction flow simulation and the reference final probability densities with diferent discretized temporal intervals $\Delta t$ under diferent discretized spatial points. We chose the number of address qubits to be 4, 5 and 6 for comparison, corresponding to the number of discretized spatial points 16, 32 and 64, respectively, leading to a decrease in $\Delta x$ . As $N _ { t }$ increases and the corresponding $\Delta t$ decreases, the final density errors under all three values of $\Delta x$ decrease. This shows that smaller temporal discretization intervals improve the precision of the wavefunction flow simulation.

However, for $n = 4 ,$ corresponding to $\Delta x = 0 . 1 2 5$ , the temporal discretization error converges to about 1.0e-03. For $n = 5 ,$ whose corresponding $\Delta x$ is 0.0625, the temporal discretization error converges to about 1.5e-04. This value is much lower than that of $\Delta x = 0 . 1 2 5$ . This is more significant when $n = 6$ , where $\Delta x$ decreases further to 0.03125, the temporal discretization error decreases linearly with increasing $N _ { t }$ and does not show convergence until $N _ { t } = 1 0 0 0$ . This is consistent with a straight line scaling as $\begin{array} { r } { O \left( \frac { 1 } { N _ { t } } \right) } \end{array}$ . These results indicate that smaller $\Delta x$ influence the temporal discretization errors less significantly. However, we observe that the decreasing trend does not match the expected temporal discretization error with a scaling $\begin{array} { r } { O \left( \frac { 1 } { N _ { t } ^ { 3 / 2 } } \right) } \end{array}$ as analyzed in Equation (46), this is because the influence of a combination of other sources of error.

## 4.6 Evaluation of wavefunction flow simulation combined with flow matching

In Subsection 4.4, the potentials employed in the wavefunction flow simulation are fixed. We now consider the potential $V _ { t }$ learned through the flow matching process.

Given the initial and target distributions, the flow matching learns a velocity field that governs the transport of particles from the initial distribution to the target distribution. We adopted a two-dimensional distribution transport experiment. The initial distribution $p _ { 0 }$ is uniform in the domain $[ - 2 , 2 ] \times [ - 2 , 2 ]$ , and the target distribution $p _ { T }$ consists of four Gaussian balls centered at (1.0, 0), (-1.0, 0), (0, 1.0) and $( 0 , - 1 . 0 )$ ， each with standard deviation 0.2. The left part of Figure 22 shows the initial and target particle plots.

Instead of directly learning the velocity field, we used neural networks to learn the potential $V _ { t }$ using flow matching. The gradient of $V _ { t }$ defines the velocity field that transports particles from $p _ { 0 }$ to $p _ { T }$ . The total evolution time $T$ was set to $2 \ \mathrm { s } ,$ and the number of time steps for particle transport was 2000. Then $V _ { t }$ was learned using a fully connected neural network with layer widths of 5-128-64-32-16-1, and the Tanh activation function was employed in each hidden layer. To satisfy the periodic requirement of the spatial domain, each spatial variable was encoded using sine and cosine functions. Then they were combined with the time variable $t ,$ resulting in a 5- dimensional input. The velocity field generated by the learned $V _ { t }$ was then employed to guide particle transportation starting from $p _ { 0 }$ in the flow model. The right part of Figure 22 presents the final particle plot transported by the velocity field produced by the learned $V _ { t } ,$ together with the target particle plot.

![](images/f83caa43bef86a0c972b33f14707c612df228503a81ece0c9326c26c7af97511.jpg)

![](images/63299987eb54256209792916959ef0bed7e4c87d09873ed15f422e4abba4c842.jpg)  
Fig. 22 The left figure shows the initial and the target particle plots. The right figure shows the final particle plot obtained by the flow model, in which the particles are transported under the velocity field generated by the learned potential $V _ { t }$ through flow matching, together with the exact target particle plot

As shown in Figure 22, the flow model successfully transports particles from the initial distribution to the target distribution consisting of four Gaussian balls using the learned potential through flow matching. Although there are a few scattered particles outside each Gaussian ball, they can be regarded as the error generated by flow matching learning.

The $V _ { t }$ learned by flow matching was then incorporated into the wavefunction flow simulation as the potential operator in the evolution of $e ^ { - i \beta D _ { V _ { t } } }$ . The total time $T$ was set to ${ \mathrm { ~ 2 ~ s , ~ } }$ and the number of time steps $N _ { t }$ was 200. The number of address qubits for each spatial dimension was set to $^ { 3 , }$ generating 8 spatial points in each dimension. The evolution was implemented following the process of Figure 2 with the uniform initial distribution.

![](images/cebd1ac1602493e74b171fa0d1371250f41a00c969b2fe0bccdd1d74621eec53.jpg)  
Fig. 23 Comparison of the final probability density obtained by wavefunction flow simulation and the target distribution. The first figure represents the initial probability density, the second figure denotes the final probability density obtained by the wavefunction flow simulation, the third figure shows the the corresponding reference final density, and the last figure shows the target probability density consisting of four Gaussian balls

Figure 23 shows the initial density, the final probability density obtained by the wavefunction flow simulation, the reference final density generated by the flow model, and the target probability density. Both the final density obtained by the wavefunction flow simulation and the reference final density have four peak regions, which match the target distribution consisting of four Gaussian balls. This indicates that the wavefunction flow simulation, driven by the velocity field generated by $V _ { t }$ learned from the flow matching, successfully generates the target density, and produces results consistent with those obtained from the flow model. Although the probability density value of each peak in the final probability density is lower than those of the reference final density and the target probability density, the similarity between the probabil ity densities demonstrates that flow matching can be successfully integrated with the wavefunction flow simulation.

Table 6 Comparison of final probability density errors between wavefunction flow simulation and the reference method
<table><tr><td>Method</td><td>RMSE</td><td>Wasserstein-1 distance</td><td>KL divergence</td></tr><tr><td>Wavefunction flow simulation</td><td>2.248e-2</td><td>8.338e-2</td><td>2.935e-1</td></tr><tr><td>Reference method</td><td>3.291e-2</td><td>9.213e-2</td><td>2.492e+0</td></tr></table>

Table 6 summarizes the RMSE, Wasserstein-1 distance, and KL divergence between the final probability density generated by the wavefunction flow simulation and the target probability density, as well as those between the corresponding reference final probability density and the target probability density. As shown in Table 6, these values for the wavefunction flow simulation are 2.248e-2, 8.338e-2, and 2.935e-1, which are close to those generated by the reference method. These results demonstrate that the wavefunction flow simulation is comparable to the reference method, i.e. the conventional flow model, in generating the target probability density driven by the velocity field learned from flow matching.

## 5 Conclusion

We employed a QROM-based phase kickback framework for wavefunction flow simulation to verify the wavefunction flow theory, thereby realizing the implementation of flow models on quantum computers. The experimental results show that this framework generates final distributions that are highly consistent with those obtained from the corresponding conventional flow model. We also proposed a QNN circuit to replace QROM for more eficient wavefunction flow simulation. Experiments show that our QNN-based phase kickback framework maintains the accuracy of the wavefunction flow simulation compared with the QROM-based phase kickback framework while reducing the circuit resources significantly. These results demonstrate that our approach provides a more eficient framework for implementing flow models on quantum computers. Additionally, we incorporate the potential function, which is learned by flow matching, into the wavefunction flow simulation. The results reveal that the final distribution generated by the wavefunction flow simulation successfully reproduces the target distribution and matches those obtained from conventional flow models.

Despite promising results, one major constraint of our model is that for a timedependent potential function, the values of all discretized time steps should be included in the training process of the time-dependent QNN circuit, resulting in considerable training time. Additionally, to encode multi-dimensional potential and kinetic operators in the circuit using QNN frameworks, more than eight QNN layers are required to maintain high precision. This leads to high circuit resources. Further research could explore a more eficient QNN circuit for replacing QROM and encoding data in quantum circuits. Furthermore, we aim to develop better strategies for training the QNN circuits, which could lead to a more eficient and accurate implementation of the wavefunction flow.

Acknowledgements. We would like to express sincere gratitude to Masood Aryapoor from M¨alardalen University for his help in formulating the theoretical framework, insightful discussions, and guidance in preparing the manuscript.

## Statements and Declarations

• Funding This research was partially funded by the initiative AI@MDU.

• Competing interests The authors declare no competing interests.

• Ethics approval and consent to participate Not applicable.

• Consent for publication Not applicable.

• Data availability No datasets were generated or analyzed during the current study.

• Materials availability Not applicable.

• Code availability The code used in this study will be made available upon reasonable request.

• Author contribution All authors conceived the research idea. Rui Che conducted the experiments and wrote the main manuscript text. Ludvig af Klinteberg provided guidance and feedback during the research process and manuscript preparation. All authors reviewed the manuscript.

## References

Amosy, O., Danzig, T., Lev, O., Porat, E., Chechik, G., Makmal, A.: Iterationfree quantum approximate optimization algorithm using neural networks. Quantum Machine Intelligence 6(2), 38 (2024) https://doi.org/10.1007/s42484-024-00159-y

Abbas, A., Sutter, D., Zoufal, C., Lucchi, A., Figalli, A., Woerner, S.: The power of quantum neural networks. Nature computational science 1(6), 403–409 (2021) https://doi.org/10.1038/s43588-021-00084-1

Barenco, A., Bennett, C.H., Cleve, R., DiVincenzo, D.P., Margolus, N., Shor, P., Sleator, T., Smolin, J.A., Weinfurter, H.: Elementary gates for quantum computation. Physical review A 52(5), 3457 (1995) https://doi.org/10.1103/PhysRevA.52. 3457

Berry, D.W., Childs, A.M., Cleve, R., Kothari, R., Somma, R.D.: Simulating hamiltonian dynamics with a truncated taylor series. Physical review letters 114(9), 090502 (2015) https://doi.org/10.1103/PhysRevLett.114.090502

Babbush, R., Gidney, C., Berry, D.W., Wiebe, N., McClean, J., Paler, A., Fowler, A., Neven, H.: Encoding electronic spectra in quantum circuits with linear t complexity. Physical Review X 8(4), 041015 (2018) https://doi.org/10.1103/PhysRevX.8. 041015

Benedetti, M., Lloyd, E., Sack, S., Fiorentini, M.: Parameterized quantum circuits as machine learning models. Quantum science and technology 4(4), 043001 (2019) https://doi.org/10.1088/2058-9565/ab4eb5

Bond-Taylor, S., Leach, A., Long, Y., Willcocks, C.G.: Deep generative modelling: A comparative review of vaes, gans, normalizing flows, energy-based and autoregressive models. IEEE transactions on pattern analysis and machine intelligence 44(11), 7327–7347 (2021) https://doi.org/10.1109/TPAMI.2021.3116668

Chen, Y.-A., Childs, A.M., Hafezi, M., Jiang, Z., Kim, H., Xu, Y.: Eficient product formulas for commutators and applications to quantum simulation. Physical Review Research 4(1), 013191 (2022) https://doi.org/10.1103/PhysRevResearch.4.013191

Cong, I., Choi, S., Lukin, M.D.: Quantum convolutional neural networks. Nature Physics 15(12), 1273–1278 (2019) https://doi.org/10.1038/s41567-019-0648-8

Cleve, R., Ekert, A., Macchiavello, C., Mosca, M.: Quantum algorithms revisited. Proceedings of the Royal Society of London. Series A: Mathematical, Physical and Engineering Sciences 454(1969), 339–354 (1998) https://doi.org/10.1098/rspa.1998. 0164

Crum, N.A., Sunny, L., Ronagh, P., Laflamme, R., Balu, R., Siopsis, G.: Stochastic security as a performance metric for quantum-enhanced generative ai. Quantum Machine Intelligence 7(1), 25 (2025) https://doi.org/10.1007/s42484-025-00256-6

Childs, A.M., Su, Y., Tran, M.C., Wiebe, N., Zhu, S.: Theory of trotter error with commutator scaling. Physical Review X 11(1), 011020 (2021) https://doi.org/10. 1103/PhysRevX.11.011020

Childs, A.M., Wiebe, N.: Product formulas for exponentials of commutators. Journal of Mathematical Physics 54(6) (2013) https://doi.org/10.1063/1.4811386

Dormand, J.R., Prince, P.J.: A family of embedded runge-kutta formulae. Journal of computational and applied mathematics 6(1), 19–26 (1980) https://doi.org/10. 1016/0771-050X(80)90013-3

Feng, R., Yu, C., Deng, W., Hu, P., Wu, T.: On the guidance of flow matching. arXiv preprint arXiv:2502.02150 (2025) https://doi.org/10.48550/arXiv.2502.02150

Gidney, C., Eker˚a, M.: How to factor 2048 bit rsa integers in 8 hours using 20 million noisy qubits. Quantum 5, 433 (2021) https://doi.org/10.22331/q-2021-04-15-433

Goodfellow, I.J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial nets. Advances in neural information processing systems 27 (2014)

Gat, I., Remez, T., Shaul, N., Kreuk, F., Chen, R.T., Synnaeve, G., Adi, Y., Lipman, Y.: Discrete flow matching. Advances in Neural Information Processing Systems 37, 133345–133385 (2024) https://doi.org/10.48550/arXiv.2407.15595

Hann, C.T., Lee, G., Girvin, S., Jiang, L.: Resilience of quantum random access memory to generic noise. Prx Quantum 2(2), 020311 (2021) https://doi.org/10.1103/ PRXQuantum.2.020311

Heya, K., Nakanishi, K.M., Mitarai, K., Yan, Z., Zuo, K., Suzuki, Y., Sugiyama, T., Tamate, S., Tabuchi, Y., Fujii, K., et al.: Subspace variational quantum simulator. Physical Review Research 5(2), 023078 (2023) https://doi.org/10.1103/ PhysRevResearch.5.023078

Innan, N., Sawaika, A., Dhor, A., Dutta, S., Thota, S., Gokal, H., Patel, N., Khan,

M.A.-Z., Theodonis, I., Bennai, M.: Financial fraud detection using quantum graph neural networks. Quantum Machine Intelligence 6(1), 7 (2024) https://doi.org/10. 1007/s42484-024-00143-6

Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014) https://doi.org/10.48550/arXiv.1412.6980

Knuth, D.E.: Big omicron and big omega and big theta. ACM Sigact News 8(2), 18–24 (1976) https://doi.org/10.1145/1008328.1008329

Kornell, A., Selinger, P.: Some improvements to product formula circuits for hamiltonian simulation. arXiv preprint arXiv:2310.12256 (2023) https://doi.org/10.48550/ arXiv.2310.12256

Kingma, D.P., Welling, M.: Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 (2013)

Li, Y., Benjamin, S.C.: Eficient variational quantum simulator incorporating active error minimization. Physical Review X 7(2), 021050 (2017) https://doi.org/10.1103/ PhysRevX.7.021050

Lee, J., Berry, D.W., Gidney, C., Huggins, W.J., McClean, J.R., Wiebe, N., Babbush, R.: Even more eficient quantum computations of chemistry through tensor hypercontraction. PRX quantum 2(3), 030305 (2021) https://doi.org/10.1103/ PRXQuantum.2.030305

Low, G.H., Chuang, I.L.: Hamiltonian simulation by qubitization. Quantum 3, 163 (2019) https://doi.org/10.22331/q-2019-07-12-163

Lipman, Y., Chen, R.T., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling. arXiv preprint arXiv:2210.02747 (2022) https://doi.org/10.48550/ arXiv.2210.02747

Liu, X., Gong, C., Liu, Q.: Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003 (2022) https://doi.org/10. 48550/arXiv.2209.03003

Lipman, Y., Havasi, M., Holderrieth, P., Shaul, N., Le, M., Karrer, B., Chen, R.T., Lopez-Paz, D., Ben-Hamu, H., Gat, I.: Flow matching guide and code. arXiv preprint arXiv:2412.06264 (2024) https://doi.org/10.48550/arXiv.2412.06264

Layden, D., Sweke, R., Havl´ıˇcek, V., Chowdhury, A., Neklyudov, K.: Wavefunction flows: Eficient quantum simulation of continuous flow models. arXiv preprint arXiv:2510.08462 (2025) https://doi.org/10.48550/arXiv.2510.08462

Low, G.H., Su, Y., Tong, Y., Tran, M.C.: Complexity of implementing trotter steps. PRX Quantum 4(2), 020323 (2023) https://doi.org/10.1103/PRXQuantum. 4.020323

Mahmud, J., Mashtura, R., Fattah, S.A., Saquib, M.: Quantum convolutional neural networks with interaction layers for classification of classical data. Quantum Machine Intelligence 6(1), 11 (2024) https://doi.org/10.1007/s42484-024-00145-4

Nielsen, M.A., Chuang, I.L.: Quantum Computation and Quantum Information. Cambridge university press, Cambridge, UK (2010)

Paquet, E., Soleymani, F.: Quantumleap: Hybrid quantum neural network for financial predictions. Expert Systems with Applications 195, 116583 (2022) https://doi.org/ 10.1016/j.eswa.2022.116583

Romero, J., Olson, J.P., Aspuru-Guzik, A.: Quantum autoencoders for eficient compression of quantum data. Quantum Science and Technology 2(4), 045001 (2017) https://doi.org/10.1088/2058-9565/aa8072

Ruiz-Perez, L., Garcia-Escartin, J.C.: Quantum arithmetic with the quantum fourier transform. Quantum Information Processing 16(6), 152 (2017) https://doi.org/10. 1007/s11128-017-1603-1

Schuld, M., Bergholm, V., Gogolin, C., Izaac, J., Killoran, N.: Evaluating analytic gradients on quantum hardware. Physical Review A 99(3), 032331 (2019) https: //doi.org/10.1103/PhysRevA.99.032331

Siemaszko, M., Buraczewski, A., Le Saux, B., Stobi´nska, M.: Rapid training of quantum recurrent neural networks. Quantum Machine Intelligence 5(2), 31 (2023) https://doi.org/10.1007/s42484-023-00117-0

Smaldone, A.M., Shee, Y., Kyro, G.W., Xu, C., Vu, N.P., Dutta, R., Farag, M.H., Galda, A., Kumar, S., Kyoseva, E., et al.: Quantum machine learning in drug discovery: Applications in academia and pharmaceutical industries. Chemical Reviews 125(12), 5436–5460 (2025) https://doi.org/10.48550/arXiv.2409.15645

Schuld, M., Sinayskiy, I., Petruccione, F.: The quest for a quantum neural network. Quantum Information Processing 13(11), 2567–2586 (2014) https://doi.org/ 10.1007/s11128-014-0809-8

Tian, J., Sun, X., Du, Y., Zhao, S., Liu, Q., Zhang, K., Yi, W., Huang, W., Wang, C., Wu, X., et al.: Recent advances for quantum neural networks in generative learning. IEEE Transactions on Pattern Analysis and Machine Intelligence 45(10), 12321–12340 (2023) https://doi.org/10.1109/TPAMI.2023.3272029

Van Den Oord, A., Kalchbrenner, N., Kavukcuoglu, K.: Pixel recurrent neural networks. In: International Conference on Machine Learning, pp. 1747–1756 (2016). https://doi.org/10.48550/arXiv.1601.06759 . PMLR

Verdon, G., McCourt, T., Luzhnica, E., Singh, V., Leichenauer, S., Hidary, J.: Quantum graph neural networks. arXiv preprint arXiv:1909.12264 (2019) https://doi.

org/10.48550/arXiv.1909.12264

Watson, J.D., Watkins, J.: Exponentially reduced circuit depths using trotter error mitigation. PRX Quantum 6(3), 030325 (2025) https://doi.org/10.1103/kw39-yxq5

Yoshida, Y., Miyato, T.: Spectral norm regularization for improving the generalizability of deep learning. arXiv preprint arXiv:1705.10941 (2017) https://doi.org/10. 48550/arXiv.1705.10941

Zindorf, B., Bose, S.: Eficient implementation of multicontrolled quantum gates. Physical Review Applied 24(4), 044030 (2025) https://doi.org/10.1103/8blx-nfcr

Zhao, R., Wang, S.: A review of quantum neural networks: methods, models, dilemma. arXiv preprint arXiv:2109.01840 (2021) https://doi.org/10.48550/arXiv.2109.01840