# Normalizing Flows for Classical Simulation of Quantum Circuits in Phase Space

Edward Jiang Quantum Information Theory Group ICFO - Institute of Photonic Sciences Barcelona Institute of Science and Technology ye-duo.jiang@icfo.eu

Timothy Heightman Quantum Information Theory Group ICFO - Institute of Photonic Sciences Barcelona Institute of Science and Technology

Marcin Płodzien´ Instytut Fizyki Teoretycznej Wydział Fizyki, Astronomii i Informatyki Stosowanej Uniwersytet Jagiellonski´

## Abstract

We present a novel framework to learn a representation of quantum circuits based on the phase space formulation of many-body systems. Given that N-qubit states can be represented as probability distributions on a smooth 2N-dimensional manifold, we show how unitary gates can be parameterized as normalizing flows between these distributions, opening a new avenue for quantum circuit simulation using generative models. We evaluate our method on three common families of quantum circuits, and show that its accuracy exceeds that of current state-of-the-art neural network methods for large system sizes.

## 1 Introduction

The problem of classical simulation of quantum many-body systems is pertinent to both the existence of quantum advantage, and to inference and control of quantum hardware [1, 2]. Naive methods for classical simulation of quantum systems include exact state-vector methods and Feynman-style path integral schemes [3–5], which are simple to implement yet have an algorithmic complexity that grows exponentially in the size of the system. Other techniques exploit the structure in certain classes of quantum circuits, such as Clifford-ness [6] or low tree-rank [7, 8], which lead to efficient classical algorithms but are not applicable to other circuits. For condensed matter systems in 1D or 2D, tensor network approaches comprise the vast majority of state-of-the-art algorithms [9, 10], under the assumption that the amount of entanglement in the system is bounded by the area of any bipartition. Recently, a new class of algorithms called Pauli propagation [11, 12] has also been shown to approximately simulate quantum circuits in low magic or noisy settings.

Machine-learning techniques have also been applied to the simulation of quantum systems. Neural quantum states (NQS) and its variations [13–17] offer compact representations and were shown to be able to evolve nontrivial many-body states. More recently, generative models have been used to learn representations of quantum states in the probabilistic formulation of quantum dynamics [18–20]. However, the accuracy of all of these methods typically degrades with system size and evolution time.

In this paper, we present a novel method to learn the output state of a quantum circuit based on the phase space formalism of many-body qubit systems [21]. Each intermediate state is represented as a probability distribution, and unitary gates are represented as normalizing flow transformations that map from one distribution to the next. Unlike previous methods, ours is based on a continuous input space rather than a discrete domain [13–15, 18], which avoids having to use continuous relaxations or hand-crafted embeddings. In addition, our architecture is naturally composable due to the sequential nature of normalizing flows, mirroring the structure of the quantum circuit. This is in contrast with previous methods which retrain a fixed architecture variationally after each gate [16–18]. We demonstrate our method on three common families of quantum circuits, and show that the classical fidelity of our model exceeds that of two neural network baselines [14, 18] as the size and depth of the circuit grows.

## 2 Methods

In this section we describe our framework to learn a representation of the states in a given quantum circuit. We give a brief overview of the phase space of multi-qubit systems before describing how quantum circuits can be modeled using normalizing flows.

## 2.1 Phase space representation

In the phase-space formalism, N-qubit states are represented by probability distributions over the manifold $( S ^ { 2 } ) ^ { \mathbf { \hat { N } } }$ , and unitary maps between states are given by a convolution with an integral kernel (or Feynman-like propagators). Although many different phase space representations exist [21], the clear choice here is the $Q$ function due to its positivity, and thus its interpretation as a probability distribution. For an N-qubit state |ψ⟩, the $Q$ function of |ψ⟩ is given by

$$
Q _ { | \psi \rangle } ( \Omega ) = \frac { 1 } { ( 2 \pi ) ^ { N } } | \langle \Omega | \psi \rangle | ^ { 2 }\tag{1}
$$

where Ω is shorthand for $( \theta _ { 1 } , \varphi _ { 1 } , \theta _ { 2 } , \varphi _ { 2 } , \cdot \cdot \cdot , \theta _ { N } , \varphi _ { N } )$ specifying a point on $( S ^ { 2 } ) ^ { N }$ , and the spin coherent states |Ω⟩ are defined as [22–24]

$$
| \Omega \rangle = \bigotimes _ { i = 1 } ^ { N } | \Omega _ { i } \rangle = \bigotimes _ { i = 1 } ^ { N } \left( \cos \frac { \theta _ { i } } { 2 } | 0 \rangle + e ^ { i \varphi _ { i } } \sin \frac { \theta _ { i } } { 2 } | 1 \rangle \right)\tag{2}
$$

One can show that the Q function is a valid probability distribution over $( S ^ { 2 } ) ^ { N }$ under the usual spherical measure [21] . Examples of $Q$ functions for some selected states are shown in App. A, see also [21].

Now suppose we have a unitary operator $U .$ . Since U is linear, there exists some transition kernel $K _ { U } ( \Omega ^ { \prime } , \bar { \Omega } )$ ) [21] such that, for any two states $| \psi \rangle , | \psi ^ { \prime } \rangle$ with $| \psi ^ { \prime } \rangle = U | \psi \rangle$ , we have

$$
Q _ { | \psi ^ { \prime } \rangle } ( \Omega ^ { \prime } ) = \int _ { ( S ^ { 2 } ) ^ { N } } K _ { U } ( \Omega ^ { \prime } , \Omega ) Q _ { | \psi \rangle } ( \Omega ) d \Omega\tag{3}
$$

so that unitary maps in the Hilbert space correspond to convolution with the integral kernels $K _ { U }$ Explicitly, given $\bar { U }$ , the kernel $K _ { U }$ has the form

$$
K _ { U } ( \Omega ^ { \prime } , \Omega ) = \frac { 1 } { ( 2 \pi ) ^ { N } } \mathrm { T r } \left[ U ^ { \dagger } | \Omega ^ { \prime } \rangle \langle \Omega ^ { \prime } | U \Delta ( \Omega ) \right] \qquad \Delta ( \Omega ) = \bigotimes _ { i = 1 } ^ { N } ( 3 | \Omega _ { i } \rangle \langle \Omega _ { i } | - I )\tag{4}
$$

Importantly, if U only acts on a small subset S of qubits, the integral (3) need only be performed over the coordinates $\Omega _ { i }$ with $i \in S$ (since the non-support qubits can be integrated out analytically), so the integral can be computed efficiently in this case. Examples of transition kernels for some common unitaries U are shown in App. B.

## 2.2 Quantum circuits as normalizing flows

A quantum circuit is specified by an initial state $| \psi _ { 0 } \rangle$ along with an ordered collection of gates $U _ { 1 } , { \bar { U } } _ { 2 } , \cdots , U _ { K }$ . The state after the k-th gate is given by

$$
| \psi _ { k } \rangle = U _ { k } U _ { k - 1 } \cdot \cdot \cdot U _ { 2 } U _ { 1 } | \psi _ { 0 } \rangle\tag{5}
$$

From the previous section, each intermediate state $| \psi _ { k } \rangle$ can be represented as a probability distribution over $( S ^ { 2 } ) ^ { \mathbf { \lfloor \delta \delta \delta \eta } }$ via Eq. (1), and mapping between states $| \psi _ { k - 1 } \rangle$ and $\left| \psi _ { k } \right.$ is accomplished by substituting the corresponding gate $U _ { k }$ into Eq. (3). However, $K _ { U _ { k } }$ may in general contain negative values, which renders the naive efficient sampling scheme from the distribution $Q _ { \left| \psi _ { k } \right. } ( \Omega ^ { \prime } )$ given $Q _ { | \psi _ { k - 1 } \rangle } ( \Omega )$ inapplicable [20, 25]. Our solution is therefore to represent the Q function of $| \psi _ { k } \rangle$ as a normalizing flow using the Q function of $| \psi _ { k - 1 } \rangle$ as the initial distribution. More specifically, suppose that $f _ { k } : ( S ^ { 2 } ) ^ { \breve { N } }  ( \breve { S } ^ { 2 } ) ^ { N }$ is a flow transformation corresponding to $U _ { k }$ . Then our goal is to find $f _ { k }$ such that the $Q$ function of $| \psi _ { k } \rangle$ is well-approximated:

![](images/a9a20074c1713bf468e70ef9b43790ee51c437cbf084d050af3c8033a9f7253e.jpg)  
Figure 1: Our proposed framework for learning the output state of a quantum circuit. In the unitary picture (top half), the state of a quantum circuit is represented by the state vectors $| \psi _ { k } \rangle$ , and maps from one state to the next are given by unitary operators $U _ { k }$ . In the phase-space picture (bottom half), the state of a quantum circuit is represented by its $Q$ function, and maps from one state to the next are induced by flow transformations $f _ { k }$ of phase space, which we parameterize by $\alpha _ { k }$ . Here, the distributions are shown over R for illustrative purposes, but in reality are over $( S ^ { 2 } ) ^ { \tilde { N } }$

$$
Q _ { | \psi _ { k } \rangle } ( \Omega ) \approx \tilde { Q } _ { | \psi _ { k } \rangle } ( \Omega ) = Q _ { | \psi _ { k - 1 } \rangle } \left( f _ { k } ^ { - 1 } ( \Omega ) \right) \left| \operatorname* { d e t } \frac { \partial f _ { k } ^ { - 1 } } { \partial \Omega } \right|\tag{6}
$$

with the right-hand-side being the usual change-of-variables formula for the probability density [26]. This process is composable: to approximate the Q function for a state $| \psi _ { k } \rangle$ , we can pushforward the $Q$ function for the initial state $| \psi _ { 0 } \rangle$ through each of the flows $f _ { 1 } , f _ { 2 } , \cdots , f _ { k }$ in order.

We emphasize that the $Q$ function is a distribution over an 2N-dimensional manifold, and thus the normalizing flows can be implemented efficiently. The problem of the exponential growth of the Hilbert space is now a problem of model expressivity [13, 14].

The initial state $| \psi _ { 0 } \rangle$ is usually some fiducial state, which can be taken to be $| 0 0 \cdots 0 \rangle$ . This has the simple product $\dot { Q }$ function given by (see also App. A):

$$
Q _ { | \psi _ { 0 } \rangle } ( \Omega ) = \prod _ { i = 1 } ^ { N } Q _ { | 0 \rangle } ( \Omega _ { i } ) = \frac { 1 } { ( 4 \pi ) ^ { N } } \prod _ { i = 1 } ^ { N } ( 1 + \cos \theta _ { i } )\tag{7}
$$

where, importantly, sampling and computing likelihoods are both efficient and exact. We parameterize each flow $f _ { k }$ with trainable parameters $\alpha _ { k }$ , and train to minimize the KL divergence between the $Q$ functions obtained via the transition kernel (3) and the ones given by the trainable flows:

$$
\mathcal { L } ( \alpha ) = \sum _ { k = 1 } ^ { K } \mathcal { D } ( \int K _ { U _ { k } } ( \Omega ^ { \prime } , \Omega ) \tilde { Q } _ { | \psi _ { k - 1 } \rangle } ( \Omega ) d \Omega  \tilde { Q } _ { | \psi _ { k } \rangle } ( \Omega ; \alpha _ { k } ) )\tag{8}
$$

where $\tilde { Q } _ { \vert \psi _ { k } \rangle }$ is the mode $[ \mathbf { \bar { s } }$ approximation of $Q _ { | \psi _ { k } \rangle }$ , and $\tilde { Q } _ { | \psi _ { 0 } \rangle } = Q _ { | \psi _ { 0 } \rangle }$ . Assuming each gate $U _ { k }$ acts on a small number of qubits, the KL divergences can be estimated efficiently using Monte Carlo sampling; the k-th term requires samples from the k − 1-th Q function, which can be obtained efficiently due to the composability of normalizing flows. Thus training uses only the learned Q functions, and the exact $Q$ functions are used only to evaluate the trained model.

![](images/7ef20f10831de3e095f56c41ef1bc5b719780a81a2977be78971f8326260c979.jpg)  
(a) 1D cluster state preparation

![](images/d8075c719ce0ed0049a4b2028be0b25a32dc27a3c9f914bd87e6363a42cb750c.jpg)  
(b) GHZ state preparation  
Figure 2: Classical fidelity between the exact Q representation of the final state and the output distribution of the normalizing flow model, as a function of the total number of qubits. For comparison, we show the fidelities obtained in [14] and [18].

![](images/4a3148a051b038f0be1aa0d256b1040f219710cee8c0b380d554fb65ebc1b8c3.jpg)  
Figure 3: Classical fidelity between the exact $Q$ representation of each intermediate state and the learned distribution of the normalizing flow model, as a function of the gate number for a 10-qubit VQE circuit for TFIM ground state preparation. For comparison, we show the fidelities obtained in [18].

In practice, we pre-train each of the flows sequentially from $k = 1$ to $k = K$ for better stability in the early stages of training, and then fine-tune all flows jointly on Eq. 8. Our framework is illustrated in Fig. 1.

## 2.3 Related work

Normalizing flows have also been applied to quantum systems directly. Closest to our work is Q-Flow [27], which likewise represents the Husimi Q function with a normalizing flow, but for open continuous-variable systems: the phase space is $\mathbb { R } ^ { \tilde { 2 } N }$ , and the flow is evolved in continuous time under the PDE that the Lindblad dynamics induces for Q. Flows have also been used as neural quantum states for continuous-variable many-body systems [28], and flow matching on spin Wigner functions has been used to learn distributions over multi-qubit states [29]. Our setting differs in three ways. First, we treat qubits, whose phase space is the compact manifold $( S ^ { 2 } ) ^ { N }$ . Second, the dynamics is a sequence of discrete gates, whose phase-space action is a convolution rather than a differential operator. Third, instead of re-fitting or evolving a single flow, we assign one flow per gate and compose them.

## 3 Experiments

To evaluate our framework, we train normalizing flow models for three families of quantum circuits: 1D cluster state preparation, GHZ state preparation, and ground state preparation of a 1D transversefield Ising model (TFIM) at the critical point. These are the same families of circuits that were used for evaluation in [18].

For our normalizing flow architecture, we use a continuous normalizing flow [30,31] over the manifold $( S ^ { 2 } ) ^ { N }$ for each unitary operator, using a simple multi-layer perceptron (MLP) to parameterize the vector field. We train on the loss function in equation (8) with the Adam optimizer [32] with learning rate 0.001 until convergence.

Nothing in our method uses the Clifford structure of the GHZ and cluster circuits. They were chosen because exact Q functions are available at large N, and to allow direct comparison with [18]. In contrast, the VQE circuit contains arbitrary-angle rotations and is not Clifford.

We evaluate the trained model by computing the classical fidelity between the Q function of the true state $Q _ { | \psi _ { K } \rangle }$ and the final predicted distribution $\tilde { Q }$ :

$$
F \left( Q _ { | \psi _ { K } \rangle } , \tilde { Q } \right) = \int _ { ( S ^ { 2 } ) ^ { N } } \sqrt { Q _ { | \psi _ { K } \rangle } ( \Omega ) \tilde { Q } ( \Omega ) } d \Omega\tag{9}
$$

since it is efficient to estimate for large system sizes (whereas computing the quantum fidelity is intractable). We compare against the results in [18], which uses a transformer architecture to learn the IC-POVM probability vector of a state instead of the Q function, and evaluates it using the discrete version of the classical fidelity (9). We pull the metrics directly from [18]. For the 1D cluster state and GHZ state, we also benchmark against neural quantum states (NQS) [14], which uses the TDVP algorithm to find optimal parameters that solve the dynamics. In this case, we run the training routine from scratch, using an MLP to parameterize the real and imaginary parts of the wavefunction and the stochastic reconfiguration algorithm for TDVP time-stepping.

The results are shown in Figures 2 and 3. At the largest sizes, our model reaches $F \approx 0 . 9 6$ at N = 64 for the cluster state, compared with 0.91 (IC-POVM) and 0.84 (NQS) at $N = 6 0$ , and F ≈ 0.965 for the GHZ state, compared with 0.95 (IC-POVM) and 0.96 (NQS). For the VQE circuit, we see that the fidelity decreases roughly linearly by $\sim 3 \times 1 0 ^ { - 4 }$ per gate, reaching 0.985 compared with 0.965 for the transformer.

## 4 Discussion and limitations

Computational cost. Each gate has its own flow, so the number of parameters grows linearly with the number of gates K. Training gate k requires evaluating $\tilde { Q } _ { k - 1 }$ at new points, which means inverting the $k - 1$ preceding flows, so the total training cost grows as $\mathcal { O } ( K ^ { 2 } )$ . In contrast, the baseline neural methods retrain a fixed-size model after each gate, so their total training cost grows as $\mathcal O ( K )$ As a result, Figures 2 and 3 compare accuracy at a given system size and are not compute-matched.

Negativity. The model is necessary in the first place because the transition kernels $K _ { U _ { k } }$ generally contain negative values. Despite this, convolution with the kernel can be computed efficiently assuming that $U _ { k }$ only has support on a small number of qubits (say 2), since the unsupported coordinates can be integrated out analytically. On the other hand, simulating the whole circuit as a single unitary gate would require integrating over all 2N coordinates. Thus the sign problem manifests as approximation error of the learned model, which compounds exponentially with the circuit depth (see Figure 3).

State dependence. For a single-qubit gate U acting on qubit i, the Q function transforms exactly via the rotation $R _ { U }$ that U induces on the i-th Bloch sphere. Hence $f _ { k } = R _ { U _ { k } }$ is an exact flow that is independent of the input, including for non-Clifford gates such as T. For entangling gates however, no such flow exists in general. Each learned $f _ { k }$ is therefore an approximate transport map from $\tilde { Q } _ { k - 1 }$ to its convolution with $K _ { U _ { k } }$ . For a different input state $\hat { Q } _ { k - 1 }$ , the transport map does not carry over and would need to be retrained.

## 5 Conclusion and outlook

In this contribution, we introduced a novel framework to learn the output state of a quantum circuit based on normalizing flows in the phase space representation, and showed numerically that the method is able to scale well to large system sizes.

Natural next benchmarks are Trotterized dynamics $e ^ { - i H t }$ of local Hamiltonians, in regimes that challenge tensor networks and Pauli propagation, as well as noisy and open-system evolution [33]. Since local observables can be estimated from flow samples, the composed flow could also serve as a variational ansatz: one would optimize the gate parameters of a circuit using energies estimated through the flows.

## 6 Acknowledgements

The authors acknowledge support from Quside Technologies SL for providing GPU resources used in the experiments.

EJ acknowledges support from the predoctoral program AGAUR-FI ajuts (2025 FI-1 00298) Joan Oró, which is backed by the Secretariat of Universities and Research of the Department of Research and Universities of the Generalitat of Catalonia, as well as the European Social Plus Fund.

TH acknowledges support from the Government of Spain (Severo Ochoa CEX2019-000910-S, Quantum in Spain, FUNQIP and European Union NextGenerationEU PRTR-C17.I1), the European Union (PASQuanS2.1, 101113690 and Quantera Veriqtas), Fundació Cellex, Fundació Mir-Puig, Generalitat de Catalunya (CERCA program), the ERC AdG CERQUTE and the AXA Chair in Quantum Information Science.

## References

[1] Hsin-Yuan Huang, Soonwon Choi, Jarrod R. McClean, and John Preskill. Vast world of quantum advantage. Phys. Rev. X, 16:030501, 2026.

[2] Michael A Nielsen and Isaac L Chuang. Quantum computation and quantum information. Cambridge university press, 2010.

[3] Mark D Penney, Dax Enshan Koh, and Robert W Spekkens. Quantum circuit dynamics via path integrals: Is there a classical action for discrete-time paths? New Journal of Physics, 19(7):073006, 2017.

[4] Christopher M Dawson, Henry L Haselgrove, Andrew P Hines, Duncan Mortimer, Michael A Nielsen, and Tobias J Osborne. Quantum computing and polynomial equations over the finite field z\_2. arXiv preprint quant-ph/0408129, 2004.

[5] B Rudiak-Gould. The sum-over-histories formulation of quantum computing; 2006. arXiv Preprint arXiv: Quant-Ph/0607151.

[6] Daniel Gottesman. The heisenberg representation of quantum computers. arXiv preprint quant-ph/9807006, 1998.

[7] Igor L Markov and Yaoyun Shi. Simulating quantum computation by contracting tensor networks. SIAM Journal on Computing, 38(3):963–981, 2008.

[8] Nadav Yoran. Efficiently contractable quantum circuits cannot produce much entanglement. arXiv preprint arXiv:0802.1156, 2008.

[9] Guifré Vidal. Efficient classical simulation of slightly entangled quantum computations. Physical review letters, 91(14):147902, 2003.

[10] Guifré Vidal. Class of quantum many-body states that can be efficiently simulated. Physical review letters, 101(11):110501, 2008.

[11] Manuel S. Rudolph, Tyson Jones, Yanting Teng, Armando Angrisani, and Zoë Holmes. Pauli propagation: A computational framework for simulating quantum systems. PRX Quantum, 7:032001, 2026.

[12] Manuel S Rudolph, Enrico Fontana, Zoë Holmes, and Lukasz Cincio. Classical surrogate simulation of quantum systems with lowesa. arXiv preprint arXiv:2308.09109, 2023.

[13] Juan Carrasquilla, Giacomo Torlai, Roger G Melko, and Leandro Aolita. Reconstructing quantum states with generative models. Nature Machine Intelligence, 1(3):155–161, 2019.

[14] Giuseppe Carleo and Matthias Troyer. Solving the quantum many-body problem with artificial neural networks. Science, 355(6325):602–606, 2017.

[15] Or Sharir, Yoav Levine, Noam Wies, Giuseppe Carleo, and Amnon Shashua. Deep autoregressive models for the efficient variational simulation of many-body quantum systems. Physical review letters, 124(2):020503, 2020.

[16] Kaelan Donatella, Zakari Denis, Alexandre Le Boité, and Cristiano Ciuti. Dynamics with autoregressive neural quantum states: Application to critical quench dynamics. Physical Review A, 108(2):022210, 2023.

[17] Irene López Gutiérrez and Christian B Mendl. Real time evolution with neural-network quantum states. Quantum, 6:627, 2022.

[18] Juan Carrasquilla, Di Luo, Felipe Pérez, Ashley Milsted, Bryan K Clark, Maksims Volkovs, and Leandro Aolita. Probabilistic simulation of quantum circuits using a deep-learning architecture. Physical Review A, 104(3):032610, 2021.

[19] EO Kiktenko, AO Malyshev, AS Mastiukova, VI Man’ko, AK Fedorov, and D Chrusci´ nski. Probability´ representation of quantum dynamics using pseudostochastic maps. Physical Review A, 101(5):052320, 2020.

[20] John van de Wetering. Quantum theory is a quasi-stochastic process theory. Electronic Proceedings in Theoretical Computer Science, 266:179–196, February 2018.

[21] Timothy Heightman, Edward Jiang, Ruth Mora-Soto, Maciej Lewenstein, and Marcin Płodzien. Quantum´ machine learning in multi-qubit phase-space part i: Foundations. arXiv preprint arXiv:2507.12117, 2025.

[22] Askold M Perelomov. Coherent states for arbitrary lie group. Communications in Mathematical Physics, 26(3):222–236, 1972.

[23] J Michael Radcliffe. Some properties of coherent spin states. Journal of Physics A: General Physics, 4(3):313–323, 1971.

[24] Wei-Min Zhang, Robert Gilmore, et al. Coherent states: Theory and some applications. Reviews of Modern Physics, 62(4):867, 1990.

[25] D Chrusci ´ nski, VI Man’ko, Giuseppe Marmo, and Franco Ventriglia. On pseudo-stochastic matrices and ´ pseudo-positive maps. Physica Scripta, 90(11):115202, 2015.

[26] Danilo Rezende and Shakir Mohamed. Variational inference with normalizing flows. In International conference on machine learning, pages 1530–1538. PMLR, 2015.

[27] Owen M. Dugan, Peter Y. Lu, Rumen Dangovski, Di Luo, and Marin Soljaciˇ c. Q-flow: Generative´ modeling for differential equations of open quantum dynamics with normalizing flows. In ICML, volume 202 of PMLR, pages 8879–8901, 2023.

[28] Scott Lawrence, Arlee Shelby, and Yukari Yamauchi. Quantum states from normalizing flows. arXiv:2406.02451, 2024.

[29] Jaehoon Hahm, Tak Hur, Joonseok Lee, and Daniel K. Park. Generative modeling of quantum distribution with functional flow matching. In QTML (extended abstract), 2024. arXiv:2607.00301.

[30] Will Grathwohl, Ricky TQ Chen, Jesse Bettencourt, Ilya Sutskever, and David Duvenaud. Ffjord: Freeform continuous dynamics for scalable reversible generative models. arXiv preprint arXiv:1810.01367, 2018.

[31] Luca Falorsi. Continuous normalizing flows on manifolds. arXiv preprint arXiv:2104.14959, 2021.

[32] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

[33] Goran Lindblad. On the generators of quantum dynamical semigroups. Communications in mathematical physics, 48(2):119–130, 1976.

## A Q Functions of Selected States

Recall that the Q function is defined as

$$
Q _ { | \psi \rangle } ( \Omega ) = \frac { 1 } { ( 2 \pi ) ^ { N } } | \langle \Omega | \psi \rangle | ^ { 2 }\tag{10}
$$

where $\pmb { \Omega } = ( \theta _ { 1 } , \varphi _ { 1 } , \theta _ { 2 } , \varphi _ { 2 } , \cdots , \theta _ { N } , \varphi _ { N } )$ , and the coherent state |Ω⟩ is given by

$$
| \Omega \rangle = \left( \cos { \frac { \theta _ { 1 } } { 2 } } | 0 \rangle + e ^ { i \varphi _ { 1 } } \sin { \frac { \theta _ { 1 } } { 2 } } | 1 \rangle \right) \otimes \cdots \otimes \left( \cos { \frac { \theta _ { N } } { 2 } } | 0 \rangle + e ^ { i \varphi _ { N } } \sin { \frac { \theta _ { N } } { 2 } } | 1 \rangle \right)\tag{11}
$$

In Table 1 we show the $Q$ functions for the eigenstates of the single-qubit Pauli operators, the four 2-qubit Bell states, and the 3-qubit GHZ and W states.

Table 1: Q functions for selected states
<table><tr><td># qubits</td><td>State</td><td>Q function</td></tr><tr><td>1</td><td>|0)</td><td> $\textstyle { \frac { 1 } { 4 \pi } } ( 1 + \cos \theta )$ </td></tr><tr><td>1</td><td>|1)</td><td> $\textstyle { \frac { 1 } { 4 \pi } } \left( 1 - \cos \theta \right)$ </td></tr><tr><td>1</td><td>|+)</td><td> ${ \begin{array} { r l } { { \frac { 1 } { 4 \pi } } \left( 1 + \sin \theta \cos \varphi \right) { \mathrm { ~ } } } \end{array} }$ </td></tr><tr><td>1</td><td>1-)</td><td> ${ \begin{array} { r l } { { \frac { 1 } { 4 \pi } } } & { { } ( 1 - \sin \theta \cos \varphi ) } \end{array} }$ </td></tr><tr><td>1</td><td>|+i)</td><td> ${ \begin{array} { l } { { \frac { 1 } { 4 \pi } } \left( 1 + \sin \theta \sin \varphi \right) } \end{array} }$ </td></tr><tr><td>1</td><td>|−i)</td><td> ${ \begin{array} { l } { { \frac { 1 } { 4 \pi } } \left( 1 - \sin \theta \sin \varphi \right) } \end{array} }$ </td></tr><tr><td>2</td><td>|Φ+)</td><td> $\begin{array} { r } { \frac { 1 } { ( 4 \pi ) ^ { 2 } } \left( 1 + \cos \theta _ { 1 } \cos \theta _ { 2 } + \sin \theta _ { 1 } \sin \theta _ { 2 } \cos \left( \varphi _ { 1 } + \varphi _ { 2 } \right) \right) } \end{array}$ </td></tr><tr><td>2</td><td>|Φ−&gt;</td><td> $\begin{array} { r } { \frac { 1 } { ( 4 \pi ) ^ { 2 } } \left( 1 + \cos \theta _ { 1 } \cos \theta _ { 2 } - \sin \theta _ { 1 } \sin \theta _ { 2 } \cos \left( \varphi _ { 1 } + \varphi _ { 2 } \right) \right) } \end{array}$ </td></tr><tr><td>2</td><td>|Ψ+&gt;</td><td> $\begin{array} { r } { \frac { 1 } { ( 4 \pi ) ^ { 2 } } \left( 1 - \cos \theta _ { 1 } \cos \theta _ { 2 } + \sin \theta _ { 1 } \sin \theta _ { 2 } \cos \left( \varphi _ { 1 } + \varphi _ { 2 } \right) \right) } \end{array}$ </td></tr><tr><td>2</td><td>|亚〉</td><td> $\begin{array} { r } { \frac { 1 } { ( 4 \pi ) ^ { 2 } } \left( 1 - \cos \theta _ { 1 } \cos \theta _ { 2 } - \sin \theta _ { 1 } \sin \theta _ { 2 } \cos \left( \varphi _ { 1 } + \varphi _ { 2 } \right) \right) } \end{array}$ </td></tr><tr><td>3</td><td>|GHZ|</td><td> $\begin{array} { c } { { \displaystyle \frac { 1 } { ( 4 \pi ) ^ { 3 } } ( 1 + \cos \theta _ { 1 } \cos \theta _ { 2 } + \cos \theta _ { 1 } \cos \theta _ { 3 } + \cos \theta _ { 2 } \cos \theta _ { 3 } } } \\ { { + \sin \theta _ { 1 } \sin \theta _ { 2 } \sin \theta _ { 3 } \cos ( \varphi _ { 1 } + \varphi _ { 2 } + \varphi _ { 3 } ) ) } } \end{array}$ </td></tr><tr><td></td><td></td><td> ${ \frac { 1 } { 3 ( 4 \pi ) ^ { 3 } } } { \Big ( } ( 1 + \cos \theta _ { 1 } ) ( 1 - \cos \theta _ { 2 } \cos \theta _ { 3 } + 2 \sin \theta _ { 2 } \sin \theta _ { 3 } \cos ( \varphi _ { 2 } - \varphi _ { 3 } ) )$ </td></tr><tr><td>3</td><td>|W&gt;</td><td> $+ \left( 1 + \cos \theta _ { 2 } \right) ( 1 - \cos \theta _ { 1 } \cos \theta _ { 3 } + 2 \sin \theta _ { 1 } \sin \theta _ { 3 } \cos ( \varphi _ { 1 } - \varphi _ { 3 } ) )$ </td></tr></table>

## B Transition Kernels of Selected Unitaries

Recall that the transition kernel $K _ { U } ( \Omega ^ { \prime } , \Omega )$ of a unitary U is defined as

$$
Q _ { | \psi ^ { \prime } \rangle } ( \Omega ^ { \prime } ) = \int _ { ( S ^ { 2 } ) ^ { N } } K _ { U } ( \Omega ^ { \prime } , \Omega ) Q _ { | \psi \rangle } ( \Omega ) d \Omega\tag{12}
$$

for all states $| \psi \rangle , | \psi ^ { \prime } \rangle$ such that $| \psi ^ { \prime } \rangle = U | \psi \rangle$ . In Table 2 we show the kernels for the Pauli $X , Y , Z$ and H gates, as well as an arbitrary Z rotation and the two-qubit controlled Z gate. Altogether these operators form a universal set of gates.

For brevity, we define

$$
X ( \Omega ) = X ( \theta , \varphi ) = \sin \theta \cos \varphi\tag{13}
$$

$$
Y ( \Omega ) = Y ( \theta , \varphi ) = \sin \theta \sin \varphi
$$

$$
Z ( \Omega ) = Z ( \theta , \varphi ) = \cos \theta\tag{14}
$$

(15)

Table 2: Transition kernels for common unitary operations
<table><tr><td>Unitary U</td><td>Kernel  $K _ { U } ( \Omega ^ { \prime } , \Omega )$ </td></tr><tr><td>X</td><td> $\begin{array} { r } { \frac { 1 } { 4 \pi } \left( 1 + 3 X ( \Omega ^ { \prime } ) X ( \Omega ) - 3 Y ( \Omega ^ { \prime } ) Y ( \Omega ) - 3 Z ( \Omega ^ { \prime } ) Z ( \Omega ) \right) } \end{array}$ </td></tr><tr><td>Y</td><td> $\begin{array} { r } { \frac { 1 } { 4 \pi } \left( 1 - 3 X ( \Omega ^ { \prime } ) X ( \Omega ) + 3 Y ( \Omega ^ { \prime } ) Y ( \Omega ) - 3 Z ( \Omega ^ { \prime } ) Z ( \Omega ) \right) } \end{array}$ </td></tr><tr><td>Z</td><td> $\begin{array} { r } { \frac { 1 } { 4 \pi } \left( 1 - 3 X ( \Omega ^ { \prime } ) X ( \Omega ) - 3 Y ( \Omega ^ { \prime } ) Y ( \Omega ) + 3 Z ( \Omega ^ { \prime } ) Z ( \Omega ) \right) } \end{array}$ </td></tr><tr><td>H</td><td> $\begin{array} { r } { \frac { 1 } { 4 \pi } \big ( 1 + 3 X ( \Omega ^ { \prime } ) Z ( \Omega ) - 3 Y ( \Omega ^ { \prime } ) Y ( \Omega ) + 3 Z ( \Omega ^ { \prime } ) X ( \Omega ) \big ) } \end{array}$ </td></tr><tr><td>Rz(α)</td><td> $\begin{array} { r } { \frac { 1 } { 4 \pi } \left( 1 + 3 X ( \theta ^ { \prime } , \varphi ^ { \prime } - \alpha ) X ( \Omega ) + 3 Y ( \theta ^ { \prime } , \varphi ^ { \prime } - \alpha ) Y ( \Omega ) + 3 Z ( \Omega ^ { \prime } ) Z ( \Omega ) \right) } \end{array}$ </td></tr><tr><td></td><td> $\begin{array} { r l } { C Z \quad } & { \frac { 1 } { ( 4 \pi ) ^ { 2 } } \Big [ \big ( 1 + 3 Z ( \Omega _ { 1 } ^ { \prime } ) Z ( \Omega _ { 1 } ) \big ) \big ( 1 + 3 Z ( \Omega _ { 2 } ^ { \prime } ) Z ( \Omega _ { 2 } ) \big ) } \\ & { \qquad + 3 \big ( 3 Z ( \Omega _ { 1 } ) + Z ( \Omega _ { 1 } ^ { \prime } ) \big ) \big ( X ( \Omega _ { 2 } ^ { \prime } ) X ( \Omega _ { 2 } ) + Y ( \Omega _ { 2 } ^ { \prime } ) Y ( \Omega _ { 2 } ) \big ) } \\ & { \qquad + 3 \big ( X ( \Omega _ { 1 } ^ { \prime } ) X ( \Omega _ { 1 } ) + Y ( \Omega _ { 1 } ^ { \prime } ) Y ( \Omega _ { 1 } ) \big ) \big ( 3 Z ( \Omega _ { 2 } ) + Z ( \Omega _ { 2 } ^ { \prime } ) \big ) } \\ & { \qquad + 9 \big ( Y ( \Omega _ { 1 } ^ { \prime } ) X ( \Omega _ { 1 } ) - X ( \Omega _ { 1 } ^ { \prime } ) Y ( \Omega _ { 1 } ) \big ) \big ( Y ( \Omega _ { 2 } ^ { \prime } ) X ( \Omega _ { 2 } ) - X ( \Omega _ { 2 } ^ { \prime } ) Y ( \Omega _ { 2 } ) \big ) \Big ] } \end{array}$ </td></tr></table>