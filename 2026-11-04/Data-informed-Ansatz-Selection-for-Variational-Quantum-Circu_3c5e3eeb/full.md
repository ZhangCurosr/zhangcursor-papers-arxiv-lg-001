# Data informed Ansatz Selection for Variational Quantum Circuits

Manav Seksaria<sup>1</sup>, Advith Desu<sup>3</sup>, and Anil Prabhakar<sup>1,2</sup>

<sup>1</sup>Center for Quantum Information, Communication and Computing, Indian Institute of Technology Madras, Chennai, India

<sup>2</sup>Department of Electrical Engineering, Indian Institute of Technology Madras, Chennai, India

<sup>3</sup>Engineering Systems and Design, Singapore University of Technology and Design, Singapore November 4, 2026

Choosing a variational quantum circuit (VQC) usually means searching over ansätze and depths, and training each candidate. We give three rules that fix these choices from the data and the gate set alone, before any training. First, a circuit without entangling gates has a rank-one Fourier coeficient tensor on every subset of qubits. A singular value decomposition of a dataset’s Fourier coeficients therefore indicates whether entanglement is needed, which qubits it connects, and a lower bound on the error of omitting it. Second, the rank of the quantum Fisher information matrix grows by p<sub>0</sub> per layer, after a first-layer deficit d<sub>0</sub>, until it reaches a ceiling set by the dynamical Lie algebra. This law fixes the depth. Third, reading out all 2<sup>N</sup> −1 Pauli-Z products from one batch of shots resolves half of the state’s directions, the most any single-basis readout can. A single-qubit readout resolves an exponentially small fraction. For the Madelon dataset, the rules prescribe 7 blocks of 3 qubits at depth 3: a 239-parameter classifier with 78.5% test accuracy, found without a grid search. Ablations confirm three predicted efects: shallower circuits, smaller blocks and dropping reuploading all lower accuracy.

## 1 Introduction

Designing a variational quantum circuit (VQC) for a learning task starts today with a search over candidate ansätze and depths. The search is run by brute force or by a learning-based method such as quantum architecture search, reinforcement learning, or Bayesian optimisation [1]. Either way, the search costs a lot of computation before any model is trained.

Circuits can be judged without training by analytic measures such as expressibility, entangling capability [2], and efective-dimension [3 These describe what a given circuit can do. They do not say which circuit to choose, or at what depth. This work proposes analytical rules that fix several of these choices without running or training any circuit.

We focus on three considerations, each tied to a distinct stage of how a parameterised circuit converts inputs into classifications (Fig. 1).

![](images/f9ad0614fad7e0267dcdb47a0009d0992150fe7b6f93f8c22cc48659590be2e8.jpg)  
Figure 1: One consideration applies at each stage of the circuit’s information path: representability at the encoding, QFIM rank at the state, extraction at the readout.

Representability. Bowles, Ahmed, and Schuld [4] benchmark a dozen quantum classifiers on six binary classification tasks. The variant with entangling gates removed often matches or beats its parent, and one cannot tell which is better without training both. The structure of the target labels decides whether a circuit needs entangling gates. We propose a diagnostic that runs before training. From a labelled dataset, it decides whether a product ansatz can represent the target and, if not, which qubits must be entangled (Sec. 2).

QFIM rank. The second consideration is how many of a circuit’s parameters move the state in independent directions. The rank of the quantum Fisher information matrix (QFIM) counts them. We show that the rank grows linearly with depth until it reaches a ceiling set by the dynamical Lie algebra (DLA) of the gate set (Sec. 3).

Extraction. The readout fixes how much of the state a VQC can use. The extraction ratio ρ measures this (Sec. 4). Reading out all $2 ^ { N } - 1$ Pauli-Z products gives the largest $\rho$ of any single-basis readout, and one batch of computational-basis shots provides them all.

## 2 Representability

A circuit without entangling gates can only compute functions whose Fourier coeficients factorise. This section proves that constraint and turns it into a test that decides, from the data alone, whether entanglement is needed.

## 2.1 Motivation

Entangling gates couple pairs of qubits and are expensive, but they reach functions that no arrangement of single-qubit gates can produce. For a given learning task, the design question is whether they are needed. A VQC without entangling gates is a product ansatz.

Our result builds on Schuld, Sweke, and Meyer [5], who showed that a VQC computes a trigonometric polynomial whose frequencies are fixed by the data encoding. Their analysis says which Fourier modes a circuit can reach. Casas and Cervera-Lierta [6] note that without entanglement the circuit computes a product of one-dimensional Fourier series. We make this exact as a rank-one condition on the coeficients of every subset of qubits. Together, the two results give a test, run on a labelled dataset, of whether entanglement is needed. Wiedmann et al. [7] also compare the Fourier transform of a dataset with the spectrum of candidate circuits to rank architectures. They do not test for entanglement.

## 2.2 Setup and notation

We consider an N-qubit VQC acting on the initial state $| 0 \rangle ^ { \otimes N }$ , which we write |0⟩. The input $x = ( x _ { 0 } , x _ { 1 } , \ldots , x _ { N - 1 } ) \in \mathbb { R } ^ { N }$ is encoded one coordinate per qubit: $x _ { q }$ enters qubit $q \in$ $[ N ] ~ = ~ \{ 0 , 1 , \ldots , N - 1 \}$ through the rotation $R _ { y } ( x _ { q } ) ~ = ~ \exp ( - i x _ { q } Y / 2 )$ The circuit has L layers. Each layer applies this encoding to every qubit, then variational single-qubit rotations and, optionally, entangling gates. We write $| \psi ( x , \pmb \theta ) \rangle$ for the resulting state, with θ the vector of all variational angles.

The circuit’s output is the expectation value of an observable M. We take M to be diagonal in a product basis. Local rotations can be absorbed into the circuit, so this basis can be the computational one, and M is a weighted sum of Pauli-Z products. We label each product by a binary string $\alpha = ( \alpha _ { 0 } , \ldots , \alpha _ { N - 1 } ) \in \{ 0 , 1 \} ^ { N }$ 2 where $\alpha _ { q }$ = 1 marks the qubits on which Z acts and $\alpha _ { q } = 0$ the qubits on which the identity acts:

$$
M = \sum _ { \alpha \in \{ 0 , 1 \} ^ { N } } w _ { \alpha } Z ^ { \alpha } , \qquad Z ^ { \alpha } = \bigotimes _ { q = 0 } ^ { N - 1 } Z _ { q } ^ { \alpha _ { q } } ,\tag{1}
$$

with real weights $w _ { \alpha } \in \mathbb { R }$ . Here $Z _ { q }$ and $I _ { q }$ are the Pauli-Z and identity operators on qubit q, with $Z _ { q } ^ { 0 } = I _ { q }$ and $Z _ { q } ^ { 1 } = Z _ { q }$

Writing $\langle Z ^ { \acute { \alpha } } ( x ) \rangle ~ = ~ \langle \psi ( x , \pmb \theta ) | Z ^ { \alpha } | \psi ( x , \pmb \theta ) \rangle$ for the expectation of $Z ^ { \alpha }$ , the function computed by the circuit is

$$
f ( x ) = \sum _ { \alpha \in \{ 0 , 1 \} ^ { N } } w _ { \alpha } \langle Z ^ { \alpha } ( x ) \rangle .\tag{2}
$$

Schuld, Sweke, and Meyer [5] showed that any function of the form of Eq. (2) is a trigonometric polynomial,

$$
f ( x ) = \sum _ { \ell \in \Omega } c ( \ell ) e ^ { i \ell \cdot x } , \qquad \ell \cdot x = \sum _ { q = 0 } ^ { N - 1 } \ell _ { q } x _ { q } ,\tag{3}
$$

where $c ( \ell ) \in \mathbb { C }$ is the Fourier coeficient of the mode at frequency $\ell = ( \ell _ { 0 } , \dots , \ell _ { N - 1 } )$ . The sum runs over the frequency support $\Omega \subseteq \mathbb { Z } ^ { N }$ , the frequencies at which the circuit can have non-zero coeficients. The spectrum of the encoding generator fixes Ω [5]. For the $R _ { y }$ encoding used here, $\Omega \ \subseteq \ \{ - L , \cdot \cdot \cdot , L \} ^ { N }$ The support of a multi-index $\boldsymbol { \ell } \in \mathbb { Z } ^ { N }$ is the set of qubits on which it is non-zero,

$$
\operatorname { s u p p } ( \ell ) = \{ q : \ell _ { q } \neq 0 \} .\tag{4}
$$

## 2.3 Product factorisation

We now restrict to circuits with no entangling gates (product ansätze) and write $\mathcal { F } _ { \mathrm { p r o d } }$ for the family of functions they compute. With only single-qubit gates acting on |0⟩, the state is a product at every stage of the circuit,

$$
\vert \psi ( x , \pmb \theta ) \rangle = \bigotimes _ { q = 0 } ^ { N - 1 } \vert \psi _ { q } ( x _ { q } , \pmb \theta _ { q } ) \rangle ,\tag{5}
$$

where $\theta _ { q }$ holds the variational angles on qubit $q ,$ and each factor depends only on the local coordinate $x _ { q } .$

The factorisation passes to the readout. Since $Z ^ { \alpha }$ is a tensor product of single-qubit operators, and the state is a tensor product of single-qubit states,

$$
\langle Z ^ { \alpha } ( x ) \rangle = \prod _ { q = 0 } ^ { N - 1 } f _ { q } ( x _ { q } ) ^ { \alpha _ { q } } .\tag{6}
$$

Here, $\begin{array} { r l r } { f _ { q } ( x _ { q } ) } & { { } = } & { \langle \psi _ { q } ( x _ { q } , \theta _ { q } ) | Z | \psi _ { q } ( x _ { q } , \theta _ { q } ) \rangle } \end{array}$ ∈ $[ - 1 , 1 ]$ is the single-qubit Pauli-Z expectation. The convention $f _ { q } ^ { 0 } = 1$ mirrors $\begin{array} { r l r } { Z _ { q } ^ { 0 } } & { { } = } & { I \colon } \end{array}$ qubits with $\alpha _ { q } \ \mathrm { ~  ~ { ~ \alpha ~ } ~ } = \ 0$ contribute a factor of one. Substituting Eq. (6) into Eq. (2) gives the explicit form of ${ \mathcal { F } } _ { \mathrm { p r o d } }$

$$
\mathcal { F } _ { \mathrm { p r o d } } = \sum _ { \alpha \in \{ 0 , 1 \} ^ { N } } w _ { \alpha } \prod _ { q = 0 } ^ { N - 1 } f _ { q } ( x _ { q } ) ^ { \alpha _ { q } } .\tag{7}
$$

## 2.4 Rank-one constraint

Applied to a single qubit, the expansion of Eq. (3) gives

$$
f _ { q } ( x _ { q } ) = \sum _ { \ell _ { q } } c _ { q } ( \ell _ { q } ) e ^ { i \ell _ { q } x _ { q } } ,\tag{8}
$$

with coeficients $c _ { q } ( \ell _ { q } ) \in \mathrm { ~ \mathbb { C } ~ }$ supported on $\{ - L , \ldots , L \}$ These coeficients form the per-qubit Fourier vector $c _ { q } .$ . Substituting into Eq. (7) and matching Fourier modes shows that the Fourier coeficients of f supported on any fixed subset $S \subseteq [ N ]$ are products of per-qubit coeficients $c _ { q } ( \ell _ { q } )$ , up to a scalar that depends only on S. For any $S \subseteq [ N ]$ with $| S | \ge 2$ , collect the coeficients $\{ c ( \ell ) : \mathrm { s u p p } ( \ell ) = S \}$ into the Fourier coeficient tensor on S,

$$
C _ { ( \ell _ { q } ) _ { q \in S } } ^ { ( S ) } = c ( \ell ) , \qquad \ell _ { q } \in \mathbb { Z } \setminus \{ 0 \} \mathrm { ~ f o r ~ } q \in S ,\tag{9}
$$

an order-|S| array with each axis of length 2L (indexed by $\{ \pm 1 , \ldots , \pm L \} )$ for an $R _ { y }$ encoding at depth L.

Theorem 1 (Rank-one constraint). For any $f \in$ $\mathcal { F } _ { \mathrm { p r o d } }$ and any $S \subseteq [ N ]$ with $| S | \ge 2$ , the Fourier coeficient tensor satisfies rank $( C ^ { ( S ) } ) \le 1$

Proof. Each term $\begin{array} { r } { w _ { \alpha } \prod _ { q \in \mathrm { s u p p } ( \alpha ) } f _ { q } ( x _ { q } ) } \end{array}$ of Eq. (7) is a product of univariate Fourier series in disjoint variables, so its coeficient at ℓ is $\begin{array} { r } { w _ { \alpha } \prod _ { q \in \mathrm { s u p p } ( \alpha ) } c _ { q } ( \ell _ { q } ) } \end{array}$ [6]. It contributes to supp $\mathbf { \chi } \left( \ell \right) = \mathbf { \chi } \mathbf { \chi } S$ only if $\operatorname { s u p p } ( \alpha ) \quad \supseteq \quad S$ , with the other qubits entering through their constant terms $c _ { q ^ { \prime } } ( 0 )$ . Summing over α gives $C ^ { ( S ) } = W _ { S } \otimes _ { q \in S } { \dot { c } } _ { q } | _ { \neq 0 }$ for a scalar $W _ { S }$ , a single outer product of complex vectors, so its rank over C is at most one. □

Here rank is taken over $\mathbb { C } .$ since the coeficients are complex. $\mathrm { F o r } \ | S | = 2$ it is the matrix rank of $C ^ { ( S ) }$ . For $| S | \ge 3$ , it is the tensor rank, the smallest number of outer products of complex vectors that sum to $C ^ { ( S ) }$

Let $f ^ { * }$ be the target, with Fourier coeficients $c ^ { * } ( \ell )$ and tensors $C ^ { * ( S ) }$

Corollary 1 (Unreachability witness). If rank $( C ^ { * ( S ) } ) ~ \ge ~ 2$ for some S with $| S | \ge 2$ then no product ansatz at any depth represents $f ^ { * }$ exactly: the qubits in S must be entangled.

A generic trigonometric polynomial fails the rank-one constraint on at least one subset S. For example, cos $( x _ { 0 } + x _ { 1 } ) = \cos x _ { 0 } \cos x _ { 1 } -$ sin x<sub>0</sub> sin $x _ { 1 }$ has Fourier coeficient matrix (in the real basis) diag(1, −1), of rank two. No product ansatz, at any depth, can produce cos $( x _ { 0 } + x _ { 1 } )$

When $f ^ { \ast } \notin \mathcal { F } _ { \mathrm { p r o d } }$ , the same constraint gives a lower bound on the mean squared error (MSE) of any product ansatz. The circuit output is 2π-periodic in every coordinate. We measure error in the squared $L ^ { 2 }$ norm on the torus $\mathbb { T } ^ { N } =$ $( \mathbb { R } / 2 \pi \mathbb { Z } ) ^ { N }$ with normalised measure, in which $\{ e ^ { i \ell \cdot x } \} _ { \ell \in \mathbb { Z } ^ { N } }$ is orthonormal. For a target $f ^ { * }$ with Fourier coeficient tensors $C ^ { * ( S ) }$ , let

$$
\rho _ { 1 } ( C ^ { \ast ( S ) } ) : = \operatorname* { m i n } _ { R : \operatorname { r a n k } ( R ) \leq 1 } \| C ^ { \ast ( S ) } - R \| _ { F } ^ { 2 }\tag{10}
$$

be its best rank-one approximation error. For $| S | \quad = \quad 2 .$ , the best rank-one approximation is the leading term of the singular value decomposition [8]. Hence,

$$
\rho _ { 1 } ( C ^ { * ( S ) } ) = \sum _ { k \geq 2 } \sigma _ { k } ( C ^ { * ( S ) } ) ^ { 2 } ,\tag{11}
$$

with $\sigma _ { 1 } \geq \sigma _ { 2 } \geq \cdots$ the singular values of $C ^ { * ( S ) }$ . For $| S | \ge 3 .$ , the same tail over any mode

unfolding is a lower bound, since the mode-n unfolding of a rank-one tensor is a rank-one matrix [8].

Proposition 1 (MSE lower bound). For any target $f ^ { * } \in L ^ { 2 } ( \mathbb { T } ^ { N } )$ and any $f \in \mathcal { F } _ { \mathrm { p r o d } } $

$$
\Vert f ^ { * } - f \Vert _ { L ^ { 2 } } ^ { 2 } \geq \sum _ { S \subseteq [ N ] : | S | \geq 2 } \rho _ { 1 } ( C ^ { * ( S ) } ) .\tag{12}
$$

Proof. By Parseval’s identity, $\| f ^ { * } - f \| _ { L ^ { 2 } } ^ { 2 } =$ $\begin{array} { r l } { \sum _ { \ell } | c ^ { * } ( \ell ) - c ( \ell ) | ^ { 2 } } \end{array}$ . We group the terms by the support $S$ of ℓ. Terms with $| S | \le 1$ are non-negative. For $\vert S \vert \ge 2$ the group equals $\| C ^ { * ( S ) } - C ^ { ( S ) } \| _ { F } ^ { 2 }$ , and Theorem 1 makes $C ^ { ( S ) }$ rank-one, so it is at least $\rho _ { 1 } ( C ^ { * ( S ) } )$ . Summing over S gives the bound. □

The bound is computed from the target alone. It is not tight in general: one per-qubit Fourier vector is shared by every subset that contains the qubit, so the exact problem is a coupled low-rank approximation of the kind studied in collective matrix factorisation [9].

## 2.5 Pre-training diagnostic

Corollary 1 and Proposition 1 together give a procedure for choosing between a product ansatz and an entangling ansatz before any training.

1. Fourier transform the data. From labelled samples of $f ^ { * }$ , estimate the coeficients $c ^ { * } ( \ell )$ on the encoding’s frequency support Ω, and group them into the tensors $C ^ { * ( S ) }$ with $| S | \geq$ 2.

2. Measure the distance to rank one. For each $C ^ { * ( S ) }$ , compute $\rho _ { 1 } ( C ^ { * ( S ) } )$ , its squared distance to the nearest rank-one tensor. For $| S | \ = \ 2 .$ , this is the sum of the squared singular values after the largest, Eq. (11). For $\vert S \vert ~ \ge ~ 3$ , the same sum over a mode unfolding gives a lower bound.

3. Choose the architecture. If $\rho _ { 1 } ( C ^ { * ( S ) } ) = 0$ for every $S ,$ entangling gates are not necessary. Otherwise, the supports $S$ with $\rho _ { 1 } \mathopen { } \mathclose \bgroup \left( C ^ { * ( S ) } \aftergroup \egroup \right) >$ 0 identify which qubits must be entangled, and $\textstyle \sum _ { S } \rho _ { 1 } ( C ^ { * ( S ) } )$ is a lower bound on the MSE of leaving them unentangled.

The diagnostic needs only a discrete Fourier transform of the dataset and singular value decompositions. Its verdict does not depend on which entangling gate is later chosen.

## 3 QFIM rank

Two parameters can move the state along the same tangent direction, so the parameter count overstates the number of independent directions. Larocca et al. [10] relate the number of independent directions to the dynamical Lie algebra (DLA) of the ansatz. This section counts them with the QFIM rank. The count grows linearly with depth, at a rate $p _ { 0 }$ set by the gate layout, until it reaches a ceiling $C _ { \mathrm { r a n k } }$ set by the DLA g and the initial state |0⟩. The QFIM is the metric on the map from parameters to states. Its $( i , j )$ entry measures how much the output state changes when parameters i and j vary together.

Definition 1 (QFIM). For a state $| \psi ( \pmb \theta ) \rangle$ ， the quantum Fisher information matrix F has entries

$$
F _ { i j } ( { \pmb \theta } ) : = 4 \operatorname { R e } \big \langle \partial _ { i } \psi \big | ( I - | \psi \rangle \langle \psi | ) \big | \partial _ { j } \psi \big \rangle .\tag{13}
$$

Its rank is the number of independent directions in which the parameters can move the state.

Definition 2 $\mathrm { ( D L A ) }$ . The dynamical Lie algebra g of an ansatz is the real Lie algebra generated under nested commutators $b y$ the operators $i W ^ { j } H _ { k } W ^ { - j } , \ j \ \in \ \mathbb { Z }$ , where $H _ { k }$ are the Pauli generators of the layer’s rotations and W is the layer’s entangler, the product of its CX gates. We write $H \in { \mathfrak { g } }$ for $i H \in { \mathfrak { g } }$

Throughout this section, U is a depth-L circuit, $\left| \psi \right. = U \left| 0 \right.$ is its output state, and $F ( L )$ is its QFIM. Its trainable rotations are $e ^ { - i \theta _ { \mu } H _ { \mu } / 2 }$ , each generated by a Pauli string $H _ { \mu } ,$ , a tensor product of $I , X , Y$ and Z. The QFIM rank is bounded above by $C _ { \mathrm { r a n k } } = \dim { \mathfrak { g } } - \dim { \mathrm { S t a b } } _ { \mathfrak { g } } ( | 0 \rangle )$ . The stabiliser ${ \mathrm { S t a b } } _ { \mathfrak { g } } ( | 0 \rangle ) = \{ H \in \mathfrak { g } : H | 0 \rangle \propto | 0 \rangle \}$ is the subalgebra of generators that fix the initial state up to a phase. Sec. 3.1 derives this ceiling. One layer does not in general reach $C _ { \mathrm { r a n k } }$ . For a hardware-eficient circuit, Haug, Bharti, and Kim [11] find numerically that the rank climbs linearly with depth until it saturates. We take the rate to be $p _ { 0 }$ per layer, where $p _ { 0 }$ is the number of independent rotation angles in one layer. Some generators of the first layer do not point in independent directions. We count them in a first-layer deficit $d _ { 0 } : = p _ { 0 } - \operatorname* { m a x } _ { \theta } \operatorname { r a n k } F ( 1 )$ 2 taken where one layer stays below the ceiling. We observe two sources of this degeneracy:

• A generator $H _ { k }$ may already fix the initial state up to a phase $\left( H _ { k } \left| 0 \right. \ \propto \ \left| 0 \right. \right)$ . Its derivative is then parallel to |ψ⟩ and the projector $\left( I - | \psi \rangle \langle \psi | \right)$ removes it. The HVA chain shows this: its N−1 commuting $R z z$ generators all fix |0⟩, so at $N = 2$ the lone $Z _ { 0 } Z _ { 1 }$ contributes $d _ { 0 } = 1$

• A layer may have more on-site generators than the local tangent has room for. N-Local shows this: three rotations per qubit on a 2-dimensional Bloch tangent give a deficit of one per qubit, so at $N \ = \ 1$ the three rotations on |0⟩ contribute $d _ { 0 } = 1$

The rank law is therefore

$$
\operatorname { r a n k } F ( L ) \leq \operatorname* { m i n } \left( p _ { 0 } L - d _ { 0 } , C _ { \mathrm { r a n k } } \right) .\tag{14}
$$

Table 1 lists the ingredients of the law for 9 families of common ansätze.

## 3.1 Orbit-stabiliser ceiling

Every parameter derivative of the output state |ψ⟩ lies in ${ \mathfrak { g } } \left| \psi \right.$ , whose dimension is the same at every point of the orbit. The QFIM rank is therefore bounded by the dimension of the orbit of |0⟩ at every depth. By the orbit-stabiliser theorem, the orbit of the state under $e ^ { { \mathfrak { g } } }$ has dimension dim g minus the dimension of the subalgebra that fixes it [12], which gives

$$
C _ { \mathrm { r a n k } } = \dim ( { \mathfrak { g } } ) - \dim { ( { \mathrm { S t a b } } _ { \mathfrak { g } } ( | 0 \rangle ) ) } .\tag{15}
$$

As an example, take the hardware-eficient ansatz (HEA) on $ { \boldsymbol { N } } \ = \ 3$ qubits with a CX chain. Single-qubit rotations span $\mathfrak { s u } ( 2 )$ on each site and the CX chain couples all qubits, so the DLA is the full su(8) with dim ${ \mathfrak { g } } = 4 ^ { 3 } - 1 = 6 3$ . The stabiliser of |000⟩ inside su(8) has dimension 63− $\cdot 2 ( 2 ^ { 3 } – 1 )$ = 49. Its complement is the 14-dimensional tangent to the space of pure states, $\mathbb { C P } ^ { 7 }$ , so $C _ { \mathrm { r a n k } } = 1 4 .$ Each layer adds $p _ { 0 } = 2 N = 6$ parameters, and there is no first-layer deficit $( d _ { 0 } \ = \ 0 )$ The rank grows as 6L until the ceiling, reached at the rank-saturation depth $L _ { \mathrm { r a n k } } ^ { \ast } ~ = ~ \lceil 1 4 / 6 \rceil ~ =$ $3 . \mathrm { A t } N = 3 .$ , layers beyond $L \ = \ 3$ add parameters but no new directions. When the qubits split into disconnected CX components, the state factorises, and the ceiling adds across components. A connected component of $q _ { b }$ qubits with su(2)-spanning gates contributes $2 ( 2 ^ { q _ { b } } - 1 )$ , the real dimension of its projective pure-state manifold. Haug, Bharti, and Kim [11] bound the rank by $2 ^ { N + 1 } - 2$ for any ansatz, and Larocca et al. [10] by the orbit dimension, which is at most dim g. Eq. (15) gives this dimension in closed form. For the HVA chain at $N = 4$ , where ${ \mathfrak { g } } \cong { \mathfrak { s o } } ( 8 )$ [13], the three bounds are 30, 28 and 19, and Eq. (14) gives the measured rank 4 at $L = 1$

## 3.2 Rank growth rate

We now show why the rank grows by $p _ { 0 }$ per layer: each new angle adds a direction that the earlier layers did not reach. The derivative of the output state with respect to $\theta _ { \mu }$ is

$$
\partial _ { \mu } \left| \psi \right. = - { \textstyle \frac { i } { 2 } } U \hat { H } _ { \mu } \left| 0 \right. ,\tag{16}
$$

where $\hat { H } _ { \mu }$ is $H _ { \mu }$ conjugated by the gates before it [10]. Since U is unitary, Eq. (16) shows that the QFIM is the Gram matrix of the vectors $( I -$ |0⟩ $\langle 0 | ) \hat { H } _ { \mu } \left| 0 \right.$ , and its rank is the dimension of their real span $V _ { L }$ . The bound in Eq. (14) holds for every θ, and the rank is maximal for almost every θ, so a single parameter point that reaches the bound fixes the generic rank (Appendix A, Propositions 3 and 4).

The count is easiest to see at $\theta _ { \mu } ~ = ~ \pi / 2$ Conjugating a Pauli string P by a rotation $R =$ $e ^ { - i \theta \mathbf { \breve { G } } / 2 }$ gives

$$
\begin{array}{c} \begin{array} { r } { R ^ { \dagger } P R = \left\{ { P } , \qquad [ P , G ] = 0 , \right.} \\ { P \cos \theta + i G P \sin \theta , \quad \{ P , G \} = 0 . } \end{array}   \end{array}\tag{17}
$$

At $\theta = \pi / 2$ the second line of Eq. (17) is the single string iGP. A CX also maps one string to one string, so every $\hat { H } _ { \mu }$ is a single Pauli string. On |0⟩, a Pauli string flips the qubits where it has X or Y and leaves the others alone, so each $\hat { H } _ { \mu } \left| 0 \right.$ is one basis state |x⟩, up to a factor ±1 or ±i. Counting directions then becomes counting bit strings: the rank at this point is the number of distinct |x⟩ and $i \left| x \right.$ with $x \neq 0$

This picture shows why each layer adds new directions. A CX copies an X on its control onto its target, so a generator conjugated by more layers becomes a longer string and flips more qubits. In EficientSU2 at $N \ = \ 3$ , the first layer flips one qubit at a time, giving states such as |100⟩. The $R _ { y }$ on $q _ { 0 }$ in the second layer, conjugated by the first layer, becomes $Z _ { 0 } Y _ { 1 } Y _ { 2 }$ and flips two qubits, giving |011⟩, a state no first-layer angle reaches.

![](images/ad7a57a3560fb6b235569cd74008a6d46153a4ab93846ab23bea26c9f43a75d3.jpg)  
Figure 2: The rank grows at slope $p _ { 0 }$ and turns a sharp corner at the orbit-stabiliser ceiling. Markers: measured QFIM rank of HEA at $N = 5$ . Red line: the predictor min $( p _ { 0 } L - d _ { 0 } , C _ { \mathrm { r a n k } } )$

A special point such as $\pi / 2$ can only lose directions, so this count is a lower bound on the generic rank. The exact ranks of Table 1 are computed at random angles instead, $\theta _ { \mu } =$ 4 arctan $t _ { \mu }$ with $t _ { \mu }$ drawn uniformly from $\mathbb { Z } \cap$ $( 1 , 1 0 ^ { 3 } ]$ (seed 42), so that every sine and cosine is rational and the rank is exact. The $\pi / 2$ counting here is illustrative.

At every $( N , L )$ we have measured, the rate holds for generic θ: HEA, EficientSU2 and N-Local at $N \leq 1 2$ and every depth up to the ceiling, including the $s = 3$ block of Sec. 6. An example for HEA is shown in Fig. 2 at $N = 5 \mathrm { : }$ the measured rank follows a straight line of slope $p _ { 0 }$ and turns a sharp corner at $C _ { \mathrm { r a n k } }$ . We treat the rate as we would a full-rank Jacobian in a neural network. When it fails, it fails for every $\theta ,$ so the cause lies in the circuit, and Sec. 3.3 shows how to find it from the gate list.

## 3.3 Frozen expectations

The rate can fail because some Pauli expectation $\langle P \rangle$ stays at zero for every θ, which confines the reachable states to a smaller set. We find such P by following the initial state forward through the gates. The initial state is the average of all $2 ^ { N }$ Pauli-Z products of Eq. (1), |0⟩ ⟨0| = $2 ^ { - N } \textstyle \sum _ { \alpha } Z ^ { \alpha }$ , so the expectation of any Pauli string $P$ in the output state is

$$
\langle P \rangle = 2 ^ { - N } \sum _ { \alpha } \mathrm { t r } \big ( P U Z ^ { \alpha } U ^ { \dagger } \big ) .\tag{18}
$$

By Eq. (17), a rotation leaves a Pauli string unchanged or mixes it with one partner, while a CX maps one string to one string. We call the Pauli strings that $U Z ^ { \alpha } U ^ { \dagger }$ can contain the strings reachable from $Z ^ { \alpha }$ . If no Pauli-Z product reaches $P ,$ every term in Eq. (18) vanishes and $\langle P \rangle$ is frozen at zero for every θ. Each frozen expectation whose diferential is independent on the tangent space of the orbit of |0⟩ removes one direction from the ceiling, so

$$
\operatorname { r a n k } F ( L ) \leq C _ { \operatorname { r a n k } } - \kappa ( L ) ,\tag{19}
$$

where the codimension $\kappa ( L )$ counts the independent frozen expectations at depth L. For the $\mathfrak { s u } ( 2 ^ { N } )$ families in Table 1, the first $R _ { y } R _ { z }$ already turns each $Z _ { q }$ into $X _ { q } , ~ Y _ { q }$ and $Z _ { q } ,$ so every Pauli string is reachable after one layer, nothing is frozen, and Eq. (19) changes nothing.

In the HVA chain each layer extends the strings reachable from the end operators $Z _ { 0 }$ and $Z _ { N - 1 }$ , which fix |0⟩, by one site, so strings more than $L$ sites from the ends stay frozen. Appendix B follows them and finds $4 ( N - L )$ frozen expectations, which lower the ceiling to $C _ { \mathrm { r a n k } } - 4 ( N - L )$ once the two reachable sets overlap at $2 L > N$ . With Eq. (14) this gives

$$
\begin{array} { r } { F ( L ) = \left\{ \begin{array} { l l } { p _ { 0 } L - d _ { 0 } , } & { 2 L \le N , } \\ { C _ { \mathrm { r a n k } } - 4 ( N - L ) , } & { N < 2 L \le 2 N , } \\ { C _ { \mathrm { r a n k } } , } & { L > N . } \end{array} \right. } \end{array}\tag{20}
$$

The same steps apply to any ansatz: list the strings reachable from the Pauli-Z products, count the strings that none of them reach, and check which of their diferentials are independent.

## 3.4 Validation

We test the rank law against exact ranks of nine common ansatz families (Table 1). The HVA chain has $R _ { x }$ on every site, so after exchanging X and Z its generators are those of the so(2N) entry of Wiersema et al. [13]. Their transverse-field Ising entry has no field on the first site and DLA so $( 2 N - 1 )$ ). The XY chain has DLA so(2N) with or without the end fields.

Brickwork couples even pairs in one sublayer and odd pairs in the next, so we take one layer to be an even sublayer followed by an odd one, which connects all qubits. Each of its two end qubits then adds only two directions per layer, so $p _ { 0 } = 4 N - 4$ (worked through in Appendix C). Two small-N points miss the law, Brickwork at $N = 4 , L = 2$ and the XY ring at $N = 4 , L = 1$ 2 and Appendix D explains both.

## 4 Extraction

The usual single-qubit Pauli-Z readout resolves an exponentially shrinking fraction of the state. Reading out all Pauli-Z products resolves half of it at every N, the most any single-basis readout can. The extraction ratio $\rho$ is this fraction: the share of an ansatz’s reachable state directions that a readout resolves. Throughout, we restrict to pure states. Noisy circuits are outside the scope of this work.

An N-qubit pure state $| \psi \rangle \in \mathbb { C } ^ { 2 ^ { N } }$ has $2 \cdot 2 ^ { N }$ real parameters. Normalisation $\| \psi \| = 1$ and the global phase symmetry $| \psi \rangle \sim e ^ { i \varphi } | \psi \rangle$ each remove one, leaving

$$
\dim { \mathcal { M } } = 2 ( 2 ^ { N } - 1 ) ,\tag{21}
$$

the dimension of the projective pure-state manifold M. Eq. (21) is the largest number of independent real parameters any readout could resolve from a pure N-qubit state. The orbit-stabiliser ceiling $C _ { \mathrm { r a n k } }$ of Sec. 3 caps the dimension a given ansatz actually reaches, with $C _ { \mathrm { r a n k } } \ \leq$ dim M. A readout is a collection of Hermitian observables $\{ O _ { 1 } , \ldots , O _ { n _ { \mathrm { o b s } } } \}$ whose expectation values form the output of the VQC. We take $n _ { \mathrm { o b s } }$ to be the number of linearly independent operators in the set, so that $\operatorname { s p a n } _ { \mathbb { R } } \{ O _ { 1 } , \dots , O _ { n _ { \mathrm { o b s } } } \}$ has dimension $n _ { \mathrm { o b s } }$ in the space of Hermitian matrices.

Definition 3 (Extraction ratio). For a readout $\{ O _ { 1 } , \ldots , O _ { n _ { \mathrm { o b s } } } \}$ acting on an ansatz with orbit-stabiliser ceiling $C _ { \mathrm { r a n k } }$

$$
\rho = { \frac { n _ { \mathrm { o b s } } } { \operatorname* { m i n } ( C _ { \mathrm { r a n k } } , \dim { \mathcal { M } } ) } } .\tag{22}
$$

We call a readout single-basis when a fixed orthonormal basis of $\mathbb { C } ^ { 2 ^ { \nu ^ { \star } } }$ diagonalises all of its observables simultaneously. The expectation values of such a readout follow from the outcome statistics of one measurement basis, with no further post-rotations. A readout that additionally measures $X _ { q }$ on some qubit is not single-basis, since $X _ { q }$ and $Z _ { q }$ are not simultaneously diagonalisable. Both readouts below are single-basis.

## 4.1 Single-qubit readout

A common readout in variational quantum learning measures one observable per qubit, $\{ Z _ { 0 } , Z _ { 1 } , . . . , Z _ { N - 1 } \}$ , giving $n _ { \mathrm { o b s } } ~ = ~ N .$ On a fully expressive ansatz $( C _ { \mathrm { r a n k } } \ = \ \dim { \mathcal M } )$ , the extraction ratio is then

$$
\rho = \frac { { \cal N } } { 2 ( 2 ^ { \cal N } - 1 ) } .\tag{23}
$$

Eq. (23) decays exponentially in N. At $N = 3$ it is $3 / 1 4 \approx 2 1 \%$ , at $N = 5$ it is $5 / 6 2 \approx 8 \%$ and at $N = 1 0$ it is already below 0.5%. The manifold grows exponentially with N, while $n _ { \mathrm { o b s } }$ grows only linearly.

## 4.2 Pauli-Z products

Consider instead the readout of all non-trivial Pauli-Z products $Z ^ { \alpha }$ of Eq. (1), one for every non-zero $\alpha \in \{ 0 , 1 \} ^ { N }$ . There are $2 ^ { N } - 1$ such operators, and they are linearly independent as Hermitian matrices, so $n _ { \mathrm { o b s } } ~ = ~ 2 ^ { N } ~ - ~ 1$ All $2 ^ { N } - 1$ Pauli-Z products follow from one batch of computational-basis shots. With $p _ { \mathbf { b } } = | \left. \mathbf { b } | \psi \right. | ^ { 2 }$ and $\begin{array} { r } { \alpha \cdot \mathbf { b } = \sum _ { q } \alpha _ { q } b _ { q } } \end{array}$ 2

$$
\langle Z ^ { \alpha } \rangle \ : = \ : \ : \sum _ { \mathbf { b } \in \{ 0 , 1 \} ^ { N } } ( - 1 ) ^ { \alpha \cdot \mathbf { b } } p _ { \mathbf { b } } .\tag{24}
$$

The matrix H with entries $H _ { \alpha , \mathbf { b } } = ( - 1 ) ^ { \alpha \cdot \mathbf { b } }$ is the Walsh-Hadamard matrix without its first row, so its rows are orthogonal and it has rank $2 ^ { N } - 1$

Proposition 2 (Maximal single-basis extraction). On an ansatz saturating the pure-state manifold, the readout of all Pauli-Z products achieves

$$
\rho = \frac { 2 ^ { N } - 1 } { 2 ( 2 ^ { N } - 1 ) } = \frac { 1 } { 2 } ,\tag{25}
$$

independent of N. Moreover, no single-basis readout achieves $\rho > 1 / 2$

The upper bound is a counting argument. A single orthonormal basis of $\mathbb { C } ^ { 2 ^ { N } }$ has $2 ^ { N }$ outcomes and hence at most $2 ^ { N } - 1$ independent probabilities, against the $2 ( 2 ^ { N } \mathrm { ~ ~ - ~ } 1 )$ real parameters of a pure state [16]. Every single-basis expectation value is a linear functional of these probabilities, so $n _ { \mathrm { o b s } } \leq 2 ^ { N } - 1$ The factor of two between this count and dim $\mathcal { M } = 2 ( 2 ^ { N } - 1 )$ gives $\rho ~ \leq ~ 1 / 2$ , and Eq. (25) shows that all

Table 1: The rank law taken with equality, rank $F ( L ) = \mathrm { m i n } ( p _ { 0 } L - d _ { 0 } , C _ { \mathrm { r a n k } } - \kappa ( L ) )$ , with $C _ { \mathrm { r a n k } } = \dim { \mathfrak { g } } -$ dim $\operatorname { S t a b } _ { \mathfrak { g } } ( | 0 \rangle )$ ) and the codimension $\kappa ( L )$ of Sec. 3.3 (0 for $L > N )$ , predicts the measured rank exactly for every family at $\mathbf { \bar { \Gamma } } N \in \{ 3 , \ldots , 6 \}$ up to $L = 1 2 ,$ , except at the two small-N exceptions explained in Appendix D. Appendix C works through the Brickwork row step by step.
<table><tr><td>Ansatz</td><td>Gate set</td><td>Coupling</td><td>DLA g</td><td>dim g</td><td>po</td><td> $d _ { 0 } ~ \mathrm { ~ \textit ~ { ~ C ~ } ~ } _ { \mathrm { r a n k } }$ </td><td></td><td>κ</td></tr><tr><td>HEA [14]</td><td> $R _ { y } R _ { z }$ </td><td>chain</td><td> $\mathfrak { s u } ( 2 ^ { N } )$ </td><td> $4 ^ { N } { - } 1$ </td><td>2N</td><td></td><td> $0 \ : \ : 2 ( 2 ^ { N } - 1 )$ </td><td>0</td></tr><tr><td>EfficientSU2 [15]</td><td> $R _ { y } R _ { z }$ </td><td>complete</td><td> $\mathfrak { s u } ( 2 ^ { N } )$ </td><td> $4 ^ { N } { - } 1$ </td><td>2N</td><td></td><td> $0 \ : \ : 2 ( 2 ^ { N } - 1 )$ </td><td>0</td></tr><tr><td>Product</td><td> $R _ { y } R _ { z }$ </td><td>none</td><td> $\mathfrak { s u } ( 2 ) ^ { N }$ </td><td>3N</td><td>2N</td><td>0 2N</td><td></td><td>0</td></tr><tr><td>N-Local [15]</td><td> $R _ { y } R _ { z } R _ { x }$ </td><td>ring</td><td> $\mathfrak { s u } ( 2 ^ { N } )$ </td><td> $4 ^ { N } { - } 1$ </td><td>3N</td><td></td><td> $\begin{array} { l l } { N } & { 2 ( 2 ^ { N } - 1 ) } \end{array}$ </td><td>0</td></tr><tr><td>HVA chain [13]</td><td> $R _ { Z Z } , R _ { x }$ </td><td>chain</td><td> ${ \mathfrak { s o } } ( 2 N )$ </td><td>N(2N-1)</td><td>2N-1</td><td></td><td> $N - 1 N ^ { 2 } { + } N { - } 1$ </td><td>4(N-L)</td></tr><tr><td>HVA ring [13]</td><td> $R _ { Z Z } , R _ { x }$ </td><td>ring</td><td> ${ \mathfrak { s o } } ( 2 N ) ^ { \oplus 2 }$ </td><td>2N(2N−1)</td><td>2N</td><td>N</td><td> $2 N ^ { 2 } { - } 2 N { + } 1$ </td><td>0</td></tr><tr><td>XY chain [13]</td><td> $R _ { X X } , R _ { Y Y } , R _ { z }$ </td><td>chain</td><td> ${ \mathfrak { s o } } ( 2 N )$ </td><td>N(2N−1)</td><td>3N-2</td><td>0</td><td> $N ( N { - } 1 )$ </td><td>0</td></tr><tr><td>XY ring [13]</td><td> $R _ { X X } , R _ { Y Y } , R _ { z }$ </td><td>ring</td><td> ${ \mathfrak { s o } } ( 2 N ) ^ { \oplus 2 }$ </td><td>2N(2N−1)</td><td>3N</td><td>0</td><td> $N ( N { - } 1 )$ </td><td>0</td></tr><tr><td>Brickwork [2]</td><td> $R _ { y } R _ { z }$ </td><td>even/odd</td><td> $\mathfrak { s u } ( 2 ^ { N } )$ </td><td> $4 ^ { N } { - } 1$ </td><td>4N-4</td><td>N−{3,4}</td><td> $2 ( 2 ^ { N } { - } 1 )$ </td><td>0</td></tr></table>

Pauli-Z products attain it. Any $2 ^ { N } - 1$ linearly independent diagonal observables attain it too, so the Pauli-Z products are the natural choice, not the only one. The missing half is the relative phases: one basis resolves the $\bar { 2 } ^ { N } - 1$ independent moduli | $\left. \mathbf { b } | \psi \right. |$ but not the $2 ^ { N } - 1$ independent phases.

## 5 Depth and block size

This section combines the QFIM rank law of Sec. 3 with a gradient-variance bound to fix the depth L and the block size s. The rank ceiling fixes L as a function of s. The gradient-variance bound (Sec. 5.2) gives an upper bound on s. At the parameter budget for the Madelon dataset, the two leave one admissible pair (s, L).

## 5.1 Depth

Consider an HEA block of s qubits: su(2)-spanning rotations on each qubit and a connected CX chain, so $p _ { 0 } ~ = ~ 2 s$ and $d _ { 0 } \ = \ 0$ Eq. (14) gives the depth at which the rank saturates,

$$
\begin{array} { r } { L _ { \mathrm { r a n k } } ^ { * } ( s ) = \left\lceil \frac { C _ { \mathrm { r a n k } } + d _ { 0 } } { p _ { 0 } } \right\rceil = \left\lceil \frac { 2 ( 2 ^ { s } - 1 ) } { 2 s } \right\rceil = \left\lceil \frac { 2 ^ { s } - 1 } { s } \right\rceil . } \end{array}\tag{26}
$$

We prescribe $L = L _ { \mathrm { r a n k } } ^ { * } ( s )$ , the smallest depth at which the block has all $C _ { \mathrm { r a n k } }$ directions. $\mathrm { A t } \ s = 3$ this gives $L = 3$ , and at $s = 5$ it gives $L = 7$

## 5.2 Block size

Eq. (26) fixes L at a given block size s. The gradient magnitude bounds the block size itself from above. For a block of s qubits whose layers form a 2-design, with input |0⟩ and a Pauli-Z-product observable, the Lie-algebraic loss variance [17] and the gradient variance of McClean et al. [18] (equivalently Fontana et al. [19]) evaluate to

$$
{ \mathrm { V a r } } [ C ] = { \frac { 1 } { 2 ^ { s } + 1 } } , \qquad { \mathrm { V a r } } [ \partial _ { \theta } C ] \approx 2 ^ { - ( s + 1 ) } ,\tag{27}
$$

both decaying as $2 ^ { - s }$ because a Pauli-Z product has Tr $O ^ { 2 } \ = \ 2 ^ { s }$ At the depth $L \ = \ 3$ used here, the block is not yet a 2-design for $s \_ { } > { }$ 4, but the variances we measure over 1500 random parameter draws are within a factor 1.7 of Eq. (27) for $s = \ 2 , \ldots , 5$ and halve per added qubit. For a monolithic N-qubit circuit the same formulas give $2 ^ { - N }$ decay, the barren-plateau regime [18, 17]. Splitting the circuit into K independent blocks of size s removes the dependence on N. Each block’s variance is fixed by s and does not depend on K, while the total ceiling $C _ { \mathrm { r a n k } } = K \cdot 2 ( 2 ^ { s } - 1 )$ grows linearly in K. Combining the prescribed depth with the gradient cap,

$$
L = L _ { \mathrm { r a n k } } ^ { * } ( s ) , \qquad s \leq s _ { \mathrm { g r a d } } ( \epsilon _ { g } ) ,\tag{28}
$$

where $s _ { \mathrm { g r a d } } ( \epsilon _ { g } )$ is the largest s whose 2-design gradient variance $\mathrm { V a r } [ \partial _ { \theta } C ]$ of Eq. (27) is at least the trainability threshold ϵ<sub>g</sub>. For $\epsilon _ { g } ~ = ~ 1 0 ^ { - 2 }$ $s _ { \mathrm { g r a d } } = 5$ , and for $\epsilon _ { g } = 1 0 ^ { - 3 } , s _ { \mathrm { g r a d } } = 8$ . The cap binds at $s = 3$ only for $\epsilon _ { g } \in ( 2 . 9 5 \times 1 0 ^ { - 2 } , 5 . 6 4 \times$ $1 0 ^ { - 2 } ]$ . For the parameter budget on the Madelon dataset, we take $s = 3$ and $L = 3$ . Then Eq. (27) gives $\mathrm { V a r } [ C ] = 1 / 9$ and $\mathrm { V a r } [ \partial _ { \theta } C ] = 0 . 0 5 6$ (easily trainable).

## 6 Block classifier

The three considerations of Secs. 2, 3 and 4 fix every hyperparameter of the block classifier.

## 6.1 Hyperparameters

Block size $s = 3 .$ . The gradient cap of Eq. (28) binds at $s = 3$ for a trainability threshold $\epsilon _ { g } = { }$ $5 \times 1 0 ^ { - 2 }$ , which we adopt. At the conventional barren-plateau threshold $\epsilon _ { g } ~ = ~ 1 0 ^ { - 2 }$ it allows $s \leq 5$ , so $s = 3$ is then one of several admissible sizes. Sec. 7.2 compares $s = 2 , 3 , 4$ at the same parameter count.

Depth L = 3. Eq. (26) gives $L = \lceil 7 / 3 \rceil = 3 .$

Readout: Pauli-Z products. By Proposition $2 ,$ measuring all $2 ^ { s } \mathrm { ~ - ~ } 1 \mathrm { ~ = ~ } 7$ non-trivial Pauli-Z products per block attains $\rho = 1 / 2$ , the optimum for a single-basis readout.

Number of blocks $K = 7 :$ set by the feature count. The K blocks must cover the 20 selected features, so $K = \lceil 2 0 / 3 \rceil = 7 .$ . The one spare slot wraps around to the first feature. The parameter count is then an output: 3s parameters per layer, $L = 3$ layers and $2 ^ { s } - 1 = 7$ readout weights per block, plus one bias, give $7 \cdot ( 2 7 + 7 ) + 1 = 2 3 9$ The total ceiling is $K \cdot C _ { \mathrm { r a n k } } ( s ) = 9 8$ directions.

Per-layer reuploading. Each feature $x _ { q }$ is encoded again as $R _ { y } ( w _ { \ell q } x _ { q } )$ at every layer ℓ, with trainable $w _ { \ell q }$ . Reuploading widens the Fourier spectrum the circuit can represent [5]. The ablation in Sec. 7 measures a 4.5-point accuracy cost without it.

## 6.2 The circuit

Each of the K blocks repeats the layer of Fig. 3 L times. One layer consists of s data-encoding rotations with trainable scale, s variational $R _ { y }$ , s variational $R _ { z }$ , and a CX chain, for 3s parameters per layer. Only 2s of them count towards the QFIM rank. The encoding $R _ { y } ( w _ { \ell q } x _ { q } )$ and the variational $R _ { y } ( \theta _ { \ell q } )$ on the same qubit share an axis and merge into one rotation, so at a fixed input the tangent of $w _ { \ell q }$ is $x _ { q }$ times the tangent of $\theta _ { \ell q }$ . The rank law therefore uses $p _ { 0 } = 2 s$ , while the parameter count uses 3s. The full classifier

![](images/a38e8d961c950804aa38cffef534f8b78e4232928fe9d43dd90d4568aec38924.jpg)  
Figure 3: One layer of a 3-qubit block. Left: data encoding with trainable scale $w _ { \ell q }$ . Middle: variational $R _ { y } R _ { z }$ . Right: CX chain. The layer repeats L times with fresh encoding weights at each repetition (reuploading).

runs K such blocks in parallel on disjoint groups of features (Fig. 4). The $2 ^ { s } - 1$ Pauli-Z products of each block are concatenated into a feature vector f of length $K ( 2 ^ { s } - 1 )$ , and a linear readout w · f + b gives the classification logit.

## 7 Experiments

## 7.1 Dataset and training

We use the Madelon dataset [20] from the NIPS 2003 Feature Selection Challenge: 2600 labelled samples and 500 features, of which 5 are informative, 15 are redundant linear combinations of them, and 480 are noise. Points are drawn from 32 clusters at the vertices of a five-dimensional hypercube, each cluster with a random class label. Linear models fail: logistic regression reaches 61.5% and a single-neuron MLP is random. We keep the 20 features with the highest F-score against the label. The retained set contains all 5 informative and all 15 redundant features. We use an 80/20 stratified train/test split. We train all models with AdamW and cosine learning-rate decay (initial learning rate 0.005), batch size 64, for 200 epochs. We run each configuration with five random seeds (42, 123, 456, 789, 1337). Table 2 reports the mean and standard deviation over seeds.

## 7.2 Ablations

The baseline is the prescribed block classifier: $K = 7$ blocks of $s \ = \ 3$ qubits, depth $L \ = \ 3 .$ Pauli-Z-product readout, per-layer reuploading. It reaches a mean accuracy of 78.5% with 239 parameters. Each ablation changes a single design element. Three of the expected efects in

![](images/73855e3085ecdbdf2425cf51df57b7eefb0c9831eaf9e507da223556fd4e77b2.jpg)  
Figure 4: The block classifier: K blocks of s qubits run in parallel on disjoint feature groups. The $2 ^ { s } - 1$ Pauli-Z products of each block are concatenated into f and fed to a linear readout w · f + b that gives the logit.

Table 2 are absent on the Madelon dataset, and the paragraphs below say which.

Depth. Below $L _ { \mathrm { r a n k } } ^ { * }$ the accuracy rises with depth, as the rank law expects. Past $L _ { \mathrm { r a n k } } ^ { * }$ it does not plateau: $L \ = \ 5$ gains 3.5 points and $L \ = \ 8$ gains 3.7, so almost all of the further gain lies between $L \ = \ 3$ and $L \ = \ 5$ The rank law predicts no new directions past $L _ { \mathrm { r a n k } } ^ { * } .$ Overparameterisation past rank saturation is known to ease optimisation [10], which is one candidate explanation. We have not tested it.

Block size. At a matched budget of ∼ 240 parameters, $s \ = \ 3$ beats $s \ = \ 2$ by 4.4 points, because an s = 2 block produces only 3 features. The expected penalty at $s = 4$ (loss variance 1/17 against 1/9, Eq. (27)) does not appear, which agrees with the gradient cap at $\epsilon _ { g } = 1 0 ^ { - 2 }$ that allows $s \leq 5$ (Sec. 5.2). Halving the loss variance costs nothing measurable on this dataset.

Readout. Dropping the Pauli-Z products and keeping only the single-qubit $\langle Z _ { q } \rangle$ leaves the mean accuracy unchanged, although $\rho$ falls from $1 / 2$ to 3/14 (Eq. (22)). The extraction ratio counts the state directions a readout can resolve, not the directions the labels depend on. On this target, the three single-qubit expectations per block hold as much usable information as the seven products.

Reuploading. With the data encoded once instead of at every layer, accuracy drops by 4.5 points, the largest efect in the table.

## 8 Conclusion

We propose three rules to fix a VQC architecture before training. A product ansatz has a rank-one Fourier coeficient tensor on every subset of qubits (Theorem 1). The singular values of a dataset’s Fourier coeficients therefore decide whether entanglement is needed and bound the error of leaving it out (Proposition 1). The QFIM rank grows as $p _ { 0 } L$ − d<sub>0</sub> until the ceiling $C _ { \mathrm { r a n k } }$ $\left( \mathrm { E q . ~ } \left( 1 4 \right) \right)$ , which fixes the depth. Reading out all Pauli-Z products gives $\rho = 1 / 2$ , the most any single-basis readout can (Proposition 2).

Ablations of the prescribed classifier confirm three predicted efects: shallower circuits, smaller blocks and dropping reuploading all lower accuracy. Deeper circuits still gain, because reuploading adds input frequencies that the rank law does not count.

We recognise a few open problems. The rules assume pure states, and noise changes the counting behind $C _ { \mathrm { r a n k } }$ and $\rho .$ Mid-circuit measurements change both the state and the readout. A rank law that includes reuploading would also predict the gain past the saturation depth.

## Acknowledgements

We would like to thank the Mphasis F1 Foundation for their financial support.

## Author contributions

[Contributions of M.S., A.D., and A.P.] Large language models (Claude, Anthropic) and Grammarly were used for grammar checking and language editing, bibliographic search, checking code and calculations. The authors checked all results and are fully responsible for the work.

## References

[1] Darya Martyniuk, Johannes Jung, and Adrian Paschke. “Quantum architecture search: a survey”. In 2024 IEEE

Table 2: Ablations, grouped by the consideration each one stresses. Three of the expected efects are absent. Accuracy is the mean and standard deviation over 5 seeds. ∆ is the diference of means from the baseline in percentage points.
<table><tr><td>Ablation</td><td>Params</td><td> $\operatorname { A c c } \ ( \% )$ </td><td></td><td>∆ Expected from the consideration</td></tr><tr><td>Baseline  $( K = 7 , s = 3 , L = 3 ,$  reuploading, Pauli-Z products)</td><td>239</td><td> ${ \bf 7 8 . 5 \pm 1 . 8 }$ </td><td> $\mathrm { n / a }$ </td><td> $\mathrm { n / a }$ </td></tr><tr><td colspan="5">QFIM rank: depth sweep around  $L _ { \mathrm { r a n k } } ^ { * } ( 3 ) = 3$ </td></tr><tr><td> $L = 1$  (rank at 43% of ceiling)</td><td>113</td><td> $7 4 . 6 \pm 0 . 9 - 3 . 9$ </td><td></td><td>Lower: block rank 6 vs. ceiling 14</td></tr><tr><td> $L = 2 ~ ( \mathrm { r a n k ~ a t ~ 8 6 \% } )$ </td><td>176</td><td> $7 5 . 6 \pm 1 . 4 - 2 . 8$ </td><td></td><td>Lower: block rank 12 vs. 14 No gain: no new QFIM directions</td></tr><tr><td>L = 5 (rank saturated, 126 extra parameters)</td><td>365</td><td> $8 1 . 9 \pm 1 . 0 + 3 . 5$ </td><td></td><td>past  $L _ { \mathrm { r a n k } } ^ { * }$  (not observed)</td></tr><tr><td> $L = 8$  (rank saturated, 315 extra parameters)</td><td>554</td><td> $8 2 . 2 \pm 1 . 7 \ \ \ \ \ + 3 . 7$ </td><td></td><td>Same (not observed)</td></tr><tr><td colspan="5">QFIM rank: block size at a matched budget of ∼ 240 parameters</td></tr><tr><td> $s = 2 , K = 1 6$ </td><td>241</td><td> $7 4 . 1 \pm 2 . 2$ </td><td></td><td>-4.4 Lower: 3 features per block; 2.3× more blocks needed</td></tr><tr><td> $s = 4 , K = 4$ </td><td>253</td><td> $7 8 . 5 \pm 1 . 1$ </td><td>0.0</td><td>Lower: loss variance 1/17 vs.  $1 / 9 ,$  Eq. (27) (not observed)</td></tr><tr><td colspan="5">Extraction</td></tr><tr><td>Single-qubit  $\langle Z _ { q } \rangle$  only</td><td>211</td><td> $7 8 . 5 \pm 0 . 8$ </td><td></td><td>0.0 Lower: ρ drops from  $1 / 2$  to 3/14, Eq. (22) (not observed)</td></tr><tr><td colspan="5">Reuploading No reuploading (encode once)</td></tr><tr><td></td><td>176</td><td> $7 4 . 0 \pm 1 . 3$ </td><td></td><td>-4.5 Lower: Fourier spectrum limited to  $\{ - 1 , 0 , 1 \} ^ { s }$  per feature</td></tr></table>

International Conference on Quantum Computing and Engineering (QCE). Pages 1695–1706. (2024).

[2] Sukin Sim, Peter D. Johnson, and Alán Aspuru-Guzik. “Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms”. Advanced Quantum Technologies 2, 1900070 (2019).

[3] Amira Abbas, David Sutter, Christa Zoufal, Aurélien Lucchi, Alessio Figalli, and Stefan Woerner. “The power of quantum neural networks”. Nature Computational Science 1, 403–409 (2021).

[4] Joseph Bowles, Shahnawaz Ahmed, and Maria Schuld. “Better than classical? The subtle art of benchmarking quantum machine learning models” (2024). arXiv:2403.07059.

[5] Maria Schuld, Ryan Sweke, and Johannes Jakob Meyer. “Efect of data encoding on the expressive power of variational quantum machine learning

models”. Physical Review A 103, 032430 (2021).

[6] Berta Casas and Alba Cervera-Lierta. “Multidimensional Fourier series with quantum circuits”. Physical Review A 107, 062612 (2023).

[7] Marco Wiedmann, Maniraman Periyasamy, and Daniel D. Scherer. “Fourier analysis of variational quantum circuits for supervised learning”. In 2025 IEEE International Conference on Quantum Computing and Engineering (QCE). Pages 1785–1795. (2025).

[8] Tamara G. Kolda and Brett W. Bader. “Tensor decompositions and applications”. SIAM Review 51, 455–500 (2009).

[9] Ajit P. Singh and Geofrey J. Gordon. “Relational learning via collective matrix factorization”. In Proceedings of the 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. Pages 650–658. (2008).

[10] Martín Larocca, Nathan Ju, Diego García-Martín, Patrick J. Coles, and Marco

Cerezo. “Theory of overparametrization in quantum neural networks”. Nature Computational Science 3, 542–551 (2023).

[11] Tobias Haug, Kishor Bharti, and M. S. Kim. “Capacity and quantum geometry of parametrized quantum circuits”. PRX Quantum 2, 040309 (2021).

[12] Francesca Albertini and Domenico D’Alessandro. “Notions of controllability for bilinear multilevel quantum systems”. IEEE Transactions on Automatic Control 48, 1399–1403 (2003).

[13] Roeland Wiersema, Efekan Kökcü, Alexander F. Kemper, and Bojko N. Bakalov. “Classification of dynamical Lie algebras of 2-local spin systems on linear, circular and fully connected topologies”. npj Quantum Information 10, 110 (2024).

[14] Abhinav Kandala, Antonio Mezzacapo, Kristan Temme, Maika Takita, Markus Brink, Jerry M. Chow, and Jay M. Gambetta. “Hardware-eficient variational quantum eigensolver for small molecules and quantum magnets”. Nature 549, 242–246 (2017).

[15] IBM Quantum. “Circuit library: N-local circuits” (2026). Accessed 2 November 2026.

[16] Steven T. Flammia, Andrew Silberfarb, and Carlton M. Caves. “Minimal informationally complete measurements for pure states”. Foundations of Physics 35, 1985–2006 (2005).

[17] Martín Larocca, Supanut Thanasilp, Samson Wang, Kunal Sharma, Jacob Biamonte, Patrick J. Coles, Lukasz Cincio, Jarrod R. McClean, Zoë Holmes, and M. Cerezo. “Barren plateaus in variational quantum computing”. Nature Reviews Physics 7, 174–189 (2025).

[18] Jarrod R. McClean, Sergio Boixo, Vadim N. Smelyanskiy, Ryan Babbush, and Hartmut Neven. “Barren plateaus in quantum neural network training landscapes”. Nature Communications 9, 4812 (2018).

[19] Enrico Fontana, Dylan Herman, Shouvanik Chakrabarti, Niraj Kumar, Romina Yalovetzky, Jamie Heredge, Shree Hari Sureshbabu, and Marco Pistoia. “Characterizing barren plateaus in quantum

ansätze with the adjoint representation”. Nature Communications 15, 7171 (2024).

[20] Isabelle Guyon. “Madelon”. UCI Machine Learning Repository (2004).

[21] Boris S. Mityagin. “The zero set of a real analytic function”. Mathematical Notes 107, 529–530 (2020).

## A Rank bounds

Proposition 3 (upper bound). For every θ and every $L \geq 1$ , dim $V _ { L } \le \mathrm { m i n } ( p _ { 0 } L - d _ { 0 } , C _ { \mathrm { r a n k } } )$

Proof. Every $\hat { H } _ { \mu }$ lies in g, so $V _ { L }$ lies in the tangent space of the orbit of |0⟩, whose dimension is $C _ { \mathrm { r a n k } }$ by Eq. (15). $\mathrm { B y }$ the definition of $d _ { 0 }$ the first layer gives at most $p _ { 0 } - d _ { 0 }$ directions, and every further layer adds at most one direction per independent rotation, which is $p _ { 0 }$ . Together these give the bound. □

Proposition 4 (one point sufices). The rank of $F ( L )$ takes its maximum value at almost every θ. If one point reaches the bound of Proposition 3, equality holds at almost every point.

Proof. Let r be the maximum rank. An $r \ \times \ r$ minor that is non-zero at one point is real-analytic in $\pmb \theta$ and not identically zero, so it vanishes only on a set of measure zero [21], and of that set the rank is r. □

## B HVA chain

This appendix derives the middle case of $\operatorname { E q . } \left( 2 0 \right)$ ， where the rank grows by only 4 per layer. A layer applies $R _ { Z Z }$ to every bond $( j , j + 1 )$ and then $R _ { x }$ to every qubit (Fig. 5). This slowdown comes from the end operators $Z _ { 0 }$ and $Z _ { N - 1 }$ , which fix |0⟩. After L layers, $Z _ { 0 }$ conjugated by the circuit acts only on the first L qubits. Similarly for $Z _ { N - 1 }$ , which acts the same way on the last L qubits.

Reachable strings: Conjugating $Z _ { 0 }$ gate by gate with Eq. (17) adds one string at each step:

$R _ { x }$ on $q _ { 0 }$ mixes $Z _ { 0 }$ with $S _ { 1 } = Y _ { 0 }$

$R _ { Z Z } \ \mathrm { o n } \ ( 0 , 1 )$ mixes $Y _ { 0 }$ with $S _ { 2 } = X _ { 0 } Z _ { 1 }$

$R _ { x }$ on $q _ { 1 }$ mixes $X _ { 0 } Z _ { 1 }$ with $S _ { 3 } = X _ { 0 } Y _ { 1 }$

![](images/782d8085b6a8f521e5db366f4ea2739df569b7e221c9512d1bb2e775a24d5a30.jpg)  
Figure 5: Two layers of the HVA chain at $N = 4$ . Each layer applies $R _ { Z Z }$ on every bond and then $R _ { x }$ on every qubit, so conjugation moves $Z _ { 0 }$ by one qubit per layer.

Continuing along the chain gives the 2N strings

$$
S _ { 2 j } = X _ { 0 } \cdot \cdot \cdot X _ { j - 1 } Z _ { j } , \ S _ { 2 j + 1 } = X _ { 0 } \cdot \cdot \cdot X _ { j - 1 } Y _ { j } ,\tag{29}
$$

with $j = 0 , \ldots , N - 1$ . Each gate mixes only neighbours in this list. One layer therefore moves $Z _ { 0 }$ two places down the list, and after L layers it reaches only $S _ { 0 } , \ldots , S _ { 2 L - 1 }$

Frozen values: Each gate maps the strings of Eq. (29) among themselves, so no string outside the list ever enters it. $Z _ { 0 }$ is the only Pauli-Z product in the list, so by Eq. (18), at every θ,

$$
\left. S _ { a } \right. = 0 , \qquad a \geq 2 L .\tag{30}
$$

The same holds from $Z _ { N - 1 }$ , so Eq. (30) and its mirror freeze $4 ( N - L )$ values.

## C Worked example

This appendix applies the rank law step by step to Brickwork at $N = 5 .$ , as an illustrative example. Fig. 6 shows two of its layers. Each layer applies $R _ { y } R _ { z }$ to every qubit, then CX on the pairs (0, 1) and (2, 3), then $R _ { y } R _ { z }$ again, then CX on (1, 2) and (3, 4), which makes 20 angles per layer.

Rate p : A layer has 20 angles but adds only 16 directions. Two rules decide which rotation gates add a direction:

Rule 1. Rotation gates on a qubit between two CXs add at most three directions.

Rule 2. $R _ { z }$ on a CX control and $R _ { x }$ on a CX target commute with the CX, so they add no new direction.

Per layer, this gives:

• 4 directions for each middle qubit q<sub>1</sub>, q<sub>2</sub> and $q _ { 3 } .$ . A CX acts on each of them in both sublayers, so between consecutive CXs it has two efective rotation gates, due to Rule 1.

• 2 directions for the end qubit $q _ { 0 }$ . Only the CX on (0, 1) acts on it, so between consecutive CXs it has four rotation gates. By Rule 1 they combine into one $R _ { z } R _ { y } R _ { z }$ ， and by Rule 2 the last R adds no new direction.

• Similarly, 2 directions for $q _ { 4 }$ , with $R _ { x }$ in place of $R _ { z }$ .

One layer therefore adds $p _ { 0 } = 4 + 4 + 4 + 2 + 2 = 1 6$ directions.

Deficit $d _ { 0 } \colon$ In the first layer the CX gates on (1, 2) and (3, 4) come last. As fixed gates they do not change the rank, and before them the qubits form three separate groups: q<sub>0</sub>q<sub>1</sub>, q<sub>2</sub>q<sub>3</sub> and $q _ { 4 }$ on its own. A pair of qubits has at most $2 ( 2 ^ { 2 } - 1 ) = 6$ directions and a single qubit 2, so the first layer reaches at most $6 + 6 + 2 = 1 4$ directions, and the exact rank is 14. This gives $d _ { 0 } = 2$ . For any N the first layer splits the same way, into pairs plus one lone qubit when N is odd, which gives

$$
d _ { 0 } = \left\{ \begin{array} { l l } { { N - 3 , } } & { { N \mathrm { ~ o d d } , } } \\ { { N - 4 , } } & { { N \mathrm { ~ e v e n } . } } \end{array} \right.\tag{31}
$$

Eq. (31) is the $d _ { 0 }$ entry of Table 1.

The ceiling follows from the DLA. The rotations span su(2) on every qubit and the CX gates connect all five, so ${ \mathfrak { g } } ~ = ~ { \mathfrak { s u } } ( 3 2 )$ , of dimension $4 ^ { 5 } - 1 = 1 0 2 3$ . A generator fixes |0⟩ when its first column is zero below the top entry, which takes $2 ( 2 ^ { 5 } - 1 ) = 6 2$ real conditions, so $C _ { \mathrm { r a n k } } = 6 2$ . Nothing is frozen either, since the first $R _ { y } R _ { z }$ already turns each $Z _ { q }$ into $X _ { q } , Y _ { q }$ and $Z _ { q } .$

Putting the three numbers into Eq. (14) gives

$$
\operatorname { r a n k } F ( L ) = \operatorname* { m i n } ( 1 6 L - 2 , 6 2 ) ,\tag{32}
$$

so the rank should grow as 14, 30, 46 and reach the ceiling at $L _ { \mathrm { r a n k } } ^ { * } = 4$ . The exact ranks are 14, 30, 46, 62, 62 and 62 for $L = 1 , \dots , 6 ,$ as Eq. (32) predicts.

## D Small-N exceptions

Two measured points miss the law of Table 1. In both, part of the circuit reaches fewer states than its angles count.

XY ring, $N = 4 , L = 1$ . Its $R _ { X X }$ and $R _ { Y Y }$ gates have 8 angles but reach only 6 directions, because 6 independent expectations stay frozen at zero (Sec. 3.3). The four R<sub>z</sub> gates add at most four more, so the rank is 10, not 12. At $N = 5$ and 6 every angle of these gates adds a direction, and the law holds.

![](images/7a065480f4adb19b21a142c9f94be7dc34505d8848c24ede3184db1ff5f1c864.jpg)  
Figure 6: Two Brickwork layers at $N = 5 ,$ , each an even sublayer followed by an odd one. The end qubits q<sub>0</sub> and $q _ { 4 }$ meet one CX per layer, so their four rotations between two CXs (dashed) form one $\mathfrak { s u } ( 2 )$ and add two directions per layer, not four.

Brickwork, $N = 4 , L = 2$ . Two layers have six CX gates. The last one acts after every rotation gate, so it does not change the rank. Of the other five, only one acts across the cut between $\{ q _ { 0 } , q _ { 1 } \}$ and $\{ q _ { 2 } , q _ { 3 } \}$ , and one CX at most doubles the Schmidt rank across a cut. The state therefore has Schmidt rank at most 2 across the cut. A 4×4 matrix of rank 2 has $2 ( 4 + 4 - 2 ) = 1 2$ complex parameters, and removing norm and phase leaves 22 real ones, so the rank is 22, not 24. At $L = 3$ a second CX acts across the cut, and the rank reaches $C _ { \mathrm { r a n k } } = 3 0$