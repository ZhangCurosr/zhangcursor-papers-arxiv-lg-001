# Optimal Low-Rank Quantum State Tomography with Bounded-Sample Joint Measurements

Ashwin Nayak<sup>∗</sup>

Xingyu Zhou<sup>†</sup>

## Abstract

We determine the optimal sample complexity of low-rank quantum state tomography when each measurement may act jointly on at most t samples. For suficiently small $\varepsilon ,$ estimating an unknown state on $\mathbb { C } ^ { d }$ of rank at most r to trace norm error ε with constant success probability requires, and is achievable with,

$$
\Theta \left( \frac { d r } { \varepsilon ^ { 2 } } \operatorname* { m a x } \left\{ 1 , \frac { r } { \sqrt { t } } \right\} \right)
$$

samples. The lower bound allows the protocol to choose each joint measurement adaptively using all previous classical outcomes; the matching upper bound is nonadaptive. Thus joint measurements on at most t samples improve the complexity of algorithms making single-sample measurements by at most a factor ${ \sqrt { t } } .$ Further, measuring order $\bar { r } ^ { 2 }$ samples jointly is necessary and suficient to attain the unrestricted collective rate.

For the lower bound, we vary the support of a state with fixed uniform spectrum and bound the Fisher information trace of every joint measurement on t samples. The adaptive Fisher chain rule and the van Trees inequality then give the trace norm lower bound. For the upper bound, we construct and analyze a nonadaptive tomography protocol based on a Gaussian joint measurement. An explicit second moment identity and a conditional Gaussian law outside the state’s support give a rank-dependent error analysis, yielding the matching rate.

## Contents

1 Introduction 3   
2 Proof overview 7   
2.1 Lower bound 7   
2.2 Upper bound 13   
3 Preliminaries 17   
3.1 States, matrix spaces, and measurements 17   
3.2 Tensor permutations and Gaussian matrices 18   
3.3 Classical Fisher information 19   
3.4 Quantum Fisher information 20   
3.5 Adaptive measurement protocols 21   
From Fisher information bounds to the adaptive lower bound 22   
4.1 The hard family and its local coordinates 22   
4.2 Fisher information trace for joint measurements . 24   
4.3 Adaptive accumulation across measurement rounds 35   
4.4 From a Fisher information bound to expected trace norm loss 37   
4.5 Completion of the lower bound 40   
5 Rank-sensitive tomography 45   
5.1 The protocol 46   
5.2 Properties of the joint measurement 47   
5.3 Analysis of the estimator . 58   
6 Discussion 62   
7 AI disclosure 63   
8 Acknowledgements 63   
References 63   
A Statistical formalism 65   
A.1 Dominated models and finite-dimensional POVMs 65   
A.2 Adaptive transcript domination 66   
A.3 Likelihood regularity 67   
A.4 The van Trees inequality 68   
B Measurable projection onto low-rank states 70

## 1 Introduction

Quantum state tomography reconstructs a classical description of an unknown state $\rho ,$ or more precisely of an approximation to $\rho ,$ from measurements on independent samples of the state. A state on $\mathbb { C } ^ { d }$ is represented by a positive semidefinite matrix of trace one. For a q-qubit system, $d = 2 ^ { q }$ so an unrestricted state has order $d ^ { 2 }$ real parameters. Intuitively, we would therefore expect a tomography algorithm to require of the order of $d ^ { 2 }$ samples for a suficiently accurate approximation. Perhaps surprisingly, the optimal sample complexity of the problem was established only a decade ago, by O’Donnell and Wright [OW16] and Haah, Harrow, Ji, ${ \mathrm { W u } } ,$ and Yu $[ \mathrm { H H J ^ { + } 1 7 } ]$ . These works proved that $\Theta ( d ^ { 2 } / \varepsilon ^ { 2 } )$ samples of $\rho$ are necessary and suficient to learn the state to within $\varepsilon$ in trace distance.

Pure quantum states, central targets of state preparation, have rank $r = 1$ even as the ambient dimension d grows exponentially with the number of qubits. They can be specified with only order d real parameters, and an early result due to Hayashi [Hay98] shows how they may be learnt with correspondingly fewer samples, namely $O ( d )$ samples. Simpler and more eficient algorithms have been discovered since (see, e.g., Ref. [GKKT20]). A natural question is whether mixed states with larger but bounded rank can similarly be learnt with fewer than $\Theta ( d ^ { 2 } )$ samples. In addition to being of theoretical interest, such states are also observed in practice when preparation noise leaves most of the spectral weight on a few eigenvectors. For example, Gross, Liu, Flammia, Becker, and Eisert [GLF<sup>+</sup>10] discuss a reconstructed state from an experiment with eight ions, with $d = 2 5 6$ and 99% of its spectral weight on just 11 eigenvectors. Motivated by these considerations, low rank tomography and its variants have been studied extensively. The optimal sample complexity for low rank tomography has again been shown to match the number of parameters required to specify the state. More precisely, $\Theta ( d r / \varepsilon ^ { 2 } )$ samples of a state $\rho$ with rank at most r are necessary and suficient for reconstructing an ε approximation in trace distance [OW16, HHJ<sup>+</sup>17, SSW25].

In this work, we return to the problem of learning states with bounded rank. While the sampleoptimal algorithms reduce the number of samples needed to the intrinsic dimension of the state, they do so by employing a joint measurement on all the samples used. Joint measurements are, in a sense, unavoidable: algorithms that measure every sample separately in a nonadaptive fashion require $\Omega ( d r ^ { 2 } / \varepsilon ^ { 2 } )$ samples [FGLE12, HHJ<sup>+</sup>17, LN25]. This sample complexity is also known to be optimal (see Ref. [KRT17] with details provided by Ref. $[ \mathrm { H H J ^ { + } 1 7 }$ , Sec. II.A] and Ref. [LN25, Sec. B.2] for one algorithm; and Ref. [GKKT20] for a second algorithm). Thus restricting the measurements to one sample at a time can cost a factor of order r in sample complexity.

The above two extremes of measurement strategies also demand diferent experimental resources. A protocol with joint measurements may require many samples to be stored and processed coherently at once, whereas a protocol measuring individual samples requires no coherent storage or manipulation across samples. Joint measurements of a large number of samples may not be feasible from a practical point of view. Moreover, we may only have access to one or at best a few quantum devices that prepare the state, and the storage of states over time may be dificult. Even assuming we have access to a suitable system that can be prepared with suficiently many samples, joint measurements may be computationally more demanding in time complexity. Finally, only a limited set of measurements may be available in an experimental set-up. This motivates the consideration of learning models that incorporate such limitations, and especially models in which a bounded number of samples are measured jointly. Measurements in which each sample is measured separately have been called by several diferent names in the literature: single-copy, unentangled, incoherent, or independent. Similarly, joint measurements have also been called entangled, coherent, or with quantum memory.

Joint measurements on a bounded number of samples interpolate between the extremes: each measurement acts on at most t fresh samples, after which only classical information is retained by the algorithm. A further degree of freedom that is available with such measurements is adaptivity — the ability to tune measurements based on the outcomes of the previous measurements. The parameter t thus limits joint quantum processing while allowing classical adaptivity across measurements.

Until recently, little was known about sample complexity of state tomography with adaptive bounded-sample joint measurements. Lowe and Nayak [LN25] (first presented at QIP 2022) lifted the $\Omega ( d r ^ { 2 } / \varepsilon ^ { 2 } )$ bound to adaptive single-sample measurements, when the measurements are restricted to a fixed set such as the set of eficiently implementable measurements (i.e., those with polynomial-size circuits). Chen, Huang, Li, Liu, and Sellke [CHL<sup>+</sup>23] subsequently proved an $\Omega ( d ^ { 3 } / \varepsilon ^ { 2 } )$ unconditional lower bound for any adaptive single-sample measurement strategy (with a finite number of outcomes). They also demonstrated the power of adaptivity when approximating the state with respect to infidelity. Both results apply when no guarantee on the rank of the state is given.

Chen, Li, and Liu [CLL24] established an almost $\sqrt { t }$ factor improvement over single-sample tomography with adaptive t-sample joint measurements for states of arbitrary rank (i.e., without a promise on the rank), for $t \leq$ min $\{ d ^ { 2 } , ( \sqrt { d } / \varepsilon ) ^ { 0 . 2 } \}$ . Their lower bound assumes $t \le \varepsilon ^ { - 0 . 1 }$ , suficiently small $\varepsilon ,$ and suficiently large d. The upper bound was subsequently improved by Pelecanos, Spilecki, and Wright [PSW26], matching the lower bound due to Chen, Li, and Liu within this restricted range. For low-rank states, however, the optimal sample complexity for adaptive strategies remained open, even in the case of single-sample measurements. It was not clear whether the techniques from prior works could be strengthened to derive optimal rank-sensitive bounds.

We determine the entire rank-dependent interpolation between single-sample and the fully joint measurement strategies, up to universal constants. The lower bound allows for arbitrary classical adaptivity, while the upper bound is achieved by a matching nonadaptive algorithm.

Theorem 1.1 (Adaptive tomography lower bound). There are universal constants $c , \varepsilon _ { 0 } > 0$ such that the following holds. Let $d \geq 2 , 1 \leq r \leq d ,$ and $t \geq 1$ be integers, and let $0 < \varepsilon \le \varepsilon _ { 0 }$ . Suppose an adaptive protocol uses a total of n samples of an arbitrary unknown state $\rho \ o n \ \mathbb { C } ^ { d }$ of rank at most $r ,$ makes joint measurements on at most t samples at a time, and outputs a state $\widehat { \rho }$ such that

$$
\mathbb { P } _ { \rho } \left[ \left. \widehat { \rho } - \rho \right. _ { 1 } \leq \varepsilon \right] \geq \frac { 2 } { 3 } .
$$

Then

$$
n \geq c { \frac { d r } { \varepsilon ^ { 2 } } } \operatorname* { m a x } \left\{ 1 , { \frac { r } { \sqrt { t } } } \right\} .\tag{1}
$$

Formally, the theorem applies to the following model of adaptive protocols. The protocol proceeds in rounds. In each round it chooses a joint positive operator-valued measure (POVM) on at most t fresh samples as a function of private randomness and all earlier classical outcomes. After the measurement, it retains only classical information. The number of samples measured may vary between rounds, and we allow adaptive stopping, provided that the bound of t on the number of samples measured jointly holds in every round. We allow any countably generated measurable outcome space, including finite, countable, and Euclidean outcome spaces. The complete model is given in Section 3.5. A nonadaptive protocol is the special case in which the sequence of measurements is fixed in advance. In particular, they are independent of the measurement outcomes observed during an execution of the protocol.

We present a nonadaptive protocol with the following guarantee.

Theorem 1.2 (Nonadaptive tomography upper bound). There is a universal constant $C > 0$ such that the following holds. Let $d \geq 2 , 1 \leq r \leq d$ , and $t \geq 1$ be integers, and let $0 < \varepsilon \le 1$ . There is a nonadaptive protocol using joint measurements on at most t samples at a time which, for every state $\rho$ on $\mathbb { C } ^ { d }$ of rank at most $r ,$ outputs a state $\widehat { \rho }$ satisfying

$$
\mathbb { P } _ { \rho } \left[ \rVert \widehat { \rho } - \rho \rVert _ { 1 } \leq \varepsilon \right] \geq \frac { 2 } { 3 }
$$

and uses at most

$$
C \frac { d r } { \varepsilon ^ { 2 } } \operatorname* { m a x } \left. 1 , \frac { r } { \sqrt { t } } \right.\tag{2}
$$

samples.

Together, the two theorems characterize the sample complexity, up to universal constants, for all suficiently small ε. Equivalently, the optimal rate is

$$
\left\{ \begin{array} { l l } { \Theta \left( d r ^ { 2 } / ( \varepsilon ^ { 2 } \sqrt { t } ) \right) , } & { 1 \le t \le r ^ { 2 } , \qquad \mathrm { a n d } } \\ { \Theta \left( d r / \varepsilon ^ { 2 } \right) , } & { t \ge r ^ { 2 } . } \end{array} \right.
$$

In particular, for $t = 1$ , arbitrary classical adaptivity does not improve the optimal nonadaptive single-sample rate. Increasing the number of samples that may be measured jointly to t improves this rate by a factor of order $\sqrt { t }$ until the optimal complexity of $O ( d r / \varepsilon ^ { 2 } )$ is reached. Measuring order $r ^ { 2 }$ samples jointly is therefore necessary and suficient to attain the optimal sample complexity for unrestricted algorithms.

To our knowledge, this is the first matching rank-dependent characterization for arbitrary adaptive joint measurements on at most t samples. $\mathrm { A t } ~ t = 1$ , the lower bound resolves the rankdependent adaptive question posed by Chen, Huang, Li, Liu, and Sellke $[ \mathrm { C H L ^ { + } 2 3 } ]$ and addresses the question of rank dependence in adaptive tomography raised by Lowe and Nayak [LN25]. For intermediate r and t, the two bounds resolve the rank-dependent interpolation left open by Chen, Li, and Liu [CLL24]. At $r = d ,$ our upper bound recovers the known optimal tradeof due to Pelecanos, Spilecki, and Wright [PSW26], while our lower bound removes the earlier restriction relating t and ε in Ref. [CLL24].

The proof of the lower bound in Theorem 1.1 controls the Fisher information trace of every joint POVM on t samples of a state with fixed uniform spectrum and varying support. A conditional score chain rule extends the bound to arbitrary adaptive transcripts. The van Trees inequality, also known as the Bayesian Cram´er–Rao inequality, then converts it into a trace norm lower bound by localizing the support parameter in operator norm.

The algorithm we present averages independent unbiased matrix estimates obtained from a Gaussian joint measurement. An explicit second moment identity controls the error on the support of $\rho$ and between the support and its orthogonal complement in Frobenius norm. Outside the support, we exploit the measurement’s Gaussian structure to bound the error in operator norm. Projecting the average onto the set of density matrices of rank at most r then gives the output of the algorithm.

Both proofs circumvent the use of representation theory; we provide an in-depth overview in Section 2.

## Related work

Table 1 summarizes previous and concurrent sample complexity bounds by rank r and the number t of samples that may be measured jointly. In our adaptive model, for states of rank at most r and suficiently small $\varepsilon ,$ the results in Theorems 1.1 and 1.2 give the optimal sample complexity up to universal constants throughout the displayed regimes of r and t.

We briefly elaborate on the related work below.

Table 1: Previous and concurrent bounds for tomography with constant success probability and suficiently small trace norm error ε. U and L denote upper and lower bounds; a dash means no separate bound is listed here. The rightmost column “Arbitrary measurements” (without a restriction on t) pertains to measurements which may act jointly on all available samples.
<table><tr><td rowspan="2">Rank r</td><td colspan="3">Maximum number t of samples measured jointly</td></tr><tr><td>t = 1</td><td> $1 < t \le r ^ { 2 }$ </td><td>Arbitrary measurements</td></tr><tr><td> $1 \leq r \ll d$ </td><td> $\mathrm { U } \colon O ( d r ^ { 2 } / \varepsilon ^ { 2 } )$   $[ \mathrm { K R T 1 7 , \dot { G } K K T 2 0 } ] \ \#$   $\bar { \mathrm { L } } \colon \Omega ( d r ^ { 2 } / \varepsilon ^ { 2 } ) \ [ \mathrm { L N 2 } \bar { 5 } ] ^ { * }$ </td><td>U:— L: —</td><td> $\mathrm { U } \colon O ( d r / \varepsilon ^ { 2 } )$  [0W16]  $\bar { \mathrm { L } } \colon \Omega ( d \bar { r } / \varepsilon ^ { 2 } )$ </td></tr><tr><td> $r = d$  (no rank promise)</td><td> $\mathrm { U } \colon O ( d ^ { 3 } / \varepsilon ^ { 2 } )$   $[ \mathrm { K R T 1 7 } , \mathrm { G K K T 2 0 } ] \ \#$   $\mathrm { \dot { L } } \colon \Omega ( d ^ { 3 } / \varepsilon ^ { 2 } ) ~ [ \mathrm { C H L } ^ { \dot { + } } 2 3 ] ^ { \dagger } ,$   $\mathrm { [ K L M R 2 6 ] } ^ { \ S }$ </td><td> $\mathrm { U } \colon O ( d ^ { 3 } / ( \varepsilon ^ { 2 } { \sqrt { t } } ) )$   $\mathrm { \Delta [ P S W 2 6 , \Phi { P S T W 2 6 } ] }$   $\dot { \mathrm { L } } \colon \Omega ( d ^ { 3 } / ( \varepsilon ^ { 2 } \sqrt { t } ) ) \ [ \mathrm { C L L 2 4 } ] ^ { \ddag } ,$   $\mathrm { [ K L M R 2 6 ] } ^ { \ S }$ </td><td>[SSW25]  $\mathrm { U } \colon O ( d ^ { 2 } / \varepsilon ^ { 2 } )$  [OW16]  $\bar { \mathrm { L } } \colon \Omega ( d ^ { 2 } / \varepsilon ^ { 2 } )$  [SSW25]</td></tr></table>

<sup>#</sup> See the introduction for the details of the algorithm in Ref. [KRT17].  
∗ [LN25] assumes nonadaptive measurements, or adaptive but eficient measurements.  
<sup>†</sup> [CHL<sup>+</sup>23] assumes measurements with finitely many outcomes.  
<sup>‡</sup> [CLL24] assumes $t < \varepsilon ^ { - 0 . 1 }$  
<sup>§</sup> [KLMR26] was developed independently and concurrently with this work.

Low-rank tomography. Compressed sensing methods reconstruct low-rank states from estimates of a small subset of Pauli expectation values [GLF<sup>+</sup>10, FGLE12]. Other approaches include spectral thresholding [BGK15], recovery from random rank-one measurements [KRT17], and projected least squares [GKKT20]. For single-sample measurements, a tomography algorithm may be derived from the work of Kueng, Rauhut, and Terstiege [KRT17]; see Refs. [HHJ<sup>+</sup>17, LN25]. The algorithm uses $O ( d r ^ { 2 } / \varepsilon ^ { 2 } )$ samples and has constant success probability. In the same setting, Gut¸˘a, Kahn, Kueng, and Tropp [GKKT20] achieve the same upper bound via projected least squares. Lowe and Nayak [LN25] proved an $\dot { \Omega } ( d r ^ { 2 } / \varepsilon ^ { 2 } )$ lower bound for nonadaptive single-sample measurements with finitely many outcomes, for states of exact rank r with $1 \leq r \leq d / 3$ and $0 < \varepsilon < 1 / 8$ , and extended this lower bound to adaptive strategies with eficient measurements. They asked how rank dependence could be incorporated into unconditional lower bounds.

For unrestricted joint measurements, O’Donnell and Wright [OW16] obtained an $O ( d r / \varepsilon ^ { 2 } )$ upper bound. In independent work, Haah, Harrow, Ji, Wu, and Yu [HHJ<sup>+</sup>17] obtained an upper bound within a logarithmic factor of the same expression. Haah et al. also proved a lower bound matching this rate up to a logarithmic factor.

Scharnhorst, Spilecki, and Wright [SSW25] subsequently closed the logarithmic gap by proving the lower bound of $\Omega ( d r / \varepsilon ^ { 2 } )$ , for $d > 1$ and suficiently small ε.

Adaptivity and bounded-sample joint measurements. Without a rank promise, Chen, Huang, Li, Liu, and Sellke [CHL<sup>+</sup>23] proved that adaptive single-sample measurements with finitely many outcomes require $\Omega ( \bar { d } ^ { 3 } / \varepsilon ^ { 2 } )$ samples for tomography in trace norm, for suficiently small ε and suficiently large d. They explicitly left the rank-dependent lower bound open.

For joint measurements on at most t samples, Chen, Li, and Liu [CLL24] gave an upper bound $\widetilde { \cal O } ( d ^ { 3 } / ( \sqrt { t } \varepsilon ^ { 2 } ) )$ for $t \leq \operatorname* { m i n } \{ d ^ { 2 } , ( \sqrt { d } / \varepsilon ) ^ { 0 . 2 } \}$ with constant success probability. They also proved a lower bound of $\Omega ( d ^ { 3 } / ( \sqrt { t } \varepsilon ^ { 2 } ) )$ against adaptive protocols when ε is suficiently small, d is suficiently large, and $t \le \varepsilon ^ { - 0 . 1 }$ . Their open questions include the complexity of rank-dependent tomography, even for $t = 1$

Pelecanos, Spilecki, and Wright [PSW26] subsequently gave the upper bound

$$
O \left( \operatorname* { m a x } \left\{ \frac { d ^ { 3 } } { \sqrt { t } \varepsilon ^ { 2 } } , \frac { d ^ { 2 } } { \varepsilon ^ { 2 } } \right\} \right)
$$

using independent applications of the debiased Keyl algorithm. Pelecanos, Spilecki, Tang, and Wright [PSTW26] obtained the same bound through their reduction from mixed state tomography to pure state tomography, while also guaranteeing time-eficient measurement and other desirable features. Both analyses use unbiased estimators and second moment bounds.

In independent and concurrent work, Keskin, Luo, Majid, and Radzihovsky [KLMR26] have reported the optimal lower bound for adaptive tomography without a rank promise, for every $t \geq 1$ and suficiently small ε. Their bound matches the $r = d$ case of the lower bound we derive in Theorem 1.1.

Statistical methods. Gill and Levit [GL95] give a general treatment of the van Trees inequality and its statistical applications. Butucea, $\mathrm { G u } \mathrm { { \Sigma } } \breve { \mathrm { { a } } } .$ , and Kypraios [BGK15] used support rotations, Fisher information, and van Trees to prove asymptotic Frobenius risk lower bounds for a fixed Pauli measurement design. Gill and Massar [GM00] established Fisher information tradeofs for multiparameter quantum estimation, and Zhou and Chen [ZC26] proved a rank-dependent Fisher information bound for support rotations. In classical estimation under communication constraints, Barnes, Han, and Ozg¨ur [<sup>¨</sup> BHO20<sup>¨</sup> ] combined a conditional Fisher information chain rule with van Trees for sequential protocols.

## Organization

The remainder of the paper is organized as follows. We give a proof overview in Section 2, collect notation and the adaptive measurement model in Section 3, prove the lower and upper bounds in Sections 4 and 5, and conclude with further questions in Section 6.

## 2 Proof overview

## 2.1 Lower bound

The proof is organized around the van Trees inequality, also known as the Bayesian Cram´er–Rao inequality. We first describe the argument for rank $r \leq d / 2 ;$ ; a depolarizing channel reduction at the end handles larger ranks.

## 2.1.1 The statistical core: the van Trees inequality

Consider a p-dimensional parameter θ, drawn from a probability distribution with smooth density $\pi _ { \ i }$ , and let $\mathcal { T } _ { \mathrm { t r } } ( \pmb { \theta } )$ be the Fisher information matrix of the complete measurement transcript. For now, one may think of $p = \Theta ( d r )$ : in the hard family below, these parameters describe spectrumpreserving rotations of an r-dimensional support subspace of $\mathbb { C } ^ { d }$ . Under suitable regularity and boundary conditions, the van Trees inequality states that every estimator T satisfies

$$
\mathbb { E } \left\| T - \pmb { \theta } \right\| _ { 2 } ^ { 2 } \geq \frac { p ^ { 2 } } { \mathbb { E } _ { \pmb { \theta } } \operatorname { T r } \left( \mathbb { Z } _ { \mathrm { t r } } ( \pmb { \theta } ) \right) + I ( \pi ) } ,
$$

where $I ( \pi )$ is the Fisher information of the density π. Thus, if the denominator is small, the expected squared estimation error on the left side must be large. The version used here is stated in Lemma 4.8 and proved in Section A.4.

For the hard family below, the parameter is a matrix $X \in \mathbb { C } ^ { ( d - r ) \times r }$ satisfying $\| X \| _ { \mathrm { o p } } < a$ We construct π by smoothly truncating a Gaussian distribution so that X is supported on this operator norm ball, and show that $I ( \pi ) = O ( d ^ { 2 } r / a ^ { 2 } )$ . If an estimator T also satisfies $\| T \| _ { \mathrm { o p } } \leq a _ { \mathrm { i } }$ then standard matrix norm inequalities give $\Vert T - X \Vert _ { 1 } \geq \Vert T - X \Vert _ { \mathrm { F } } ^ { 2 } / ( 2 a )$ , converting the squared Frobenius loss in van Trees into trace norm loss. The construction of π and this conversion to expected trace norm loss are given in Lemmas 4.9 and 4.10.

On the other hand, confidence amplification and a simple postprocessing step turn a successful tomographic estimator into an estimator of X with operator norm at most a and small expected trace norm loss. Consequently, a suficiently large right side in the van Trees inequality rules out such a tomographic estimator. The main task is therefore to bound the transcript term in the denominator from above. The required postprocessing is carried out in Section 4.5.1.

## 2.1.2 Fisher information along an adaptive decision tree

An adaptive protocol chooses each measurement according to the outcomes observed so far. We represent this process as a decision tree: each node $h$ contains the preceding outcomes together with any private randomness used by the protocol, and the outgoing edges represent the possible outcomes of the next measurement. Once a node h is reached at round $i ,$ the protocol has selected a fixed POVM acting jointly on $t _ { i } ( h ) \leq t$ fresh samples. Therefore, a Fisher information bound that holds for every fixed POVM applies to the measurement selected at every node. A complete transcript corresponds to a root-to-leaf path through the tree, and the unknown state determines the distribution over these paths.

It remains to combine the bounds along such a path. Let $H _ { i }$ be the random history reached before round i, and let $\mathcal { T } _ { i } ( h ; X )$ be the Fisher information matrix of the fixed POVM selected at history h when the parameter is X. We write $\mathbb { E } _ { X }$ for expectation over the transcript generated at parameter X. Even with full adaptivity, Fisher information satisfies the chain rule

$$
\mathrm { T r } \big ( \mathcal { T } _ { \mathrm { t r } } ( X ) \big ) = \mathbb { E } _ { X } \left[ \sum _ { i } \mathrm { T r } \big ( \mathcal { T } _ { i } ( H _ { i } ; X ) \big ) \right] .
$$

We prove this identity in Lemma 4.6. The expectation averages over the random path generated at parameter X, while the sum contains the Fisher information of the fixed POVM used at each node on that path. The main technical step, addressed in the next subsection, is to bound the Fisher information of the joint measurement selected at an arbitrary node. Once this nodewise bound is available, applying it inside the sum and using the pathwise sample bound $\textstyle \sum _ { i } t _ { i } ( H _ { i } ) \leq n$ controls the Fisher information of the complete transcript.

## 2.1.3 The hard family and the information at one node

The hard states that the protocol must distinguish all have rank r and are maximally mixed on their supports. We keep their eigenvalues fixed and vary only the support. Fix an r-dimensional reference subspace $S \subseteq \mathbb { C } ^ { d }$ , and consider linear maps $X : S  S ^ { \bot }$ with small operator norm. Such a map specifies the nearby r-dimensional subspace

$$
\{ { \pmb u } + X { \pmb u } : { \pmb u } \in S \} .
$$

When $X = 0$ , this subspace is $S ;$ small $\| X \| _ { \mathrm { o p } }$ therefore describes a small rotation of the support away from S. Let $\Pi _ { X }$ be the orthogonal projector onto this subspace and set $\rho _ { X } : = \Pi _ { X } / r$ . The real and imaginary parts of X give $p : = 2 r ( d - r )$ local parameters. At $X = 0$ , a direction $\dot { H } \in \mathbb { C } ^ { ( d - r ) \times r }$ changes the state by

$$
\mathrm { D } _ { H } \rho _ { 0 } = \frac { 1 } { r } \left( \begin{array} { c c } { { 0 } } & { { H ^ { \dag } } } \\ { { H } } & { { 0 } } \end{array} \right) .
$$

For X and Y in this small operator norm neighborhood, closeness of the corresponding states implies closeness of their support parameters:

$$
\| X - Y \| _ { 1 } \leq 2 r \left\| \rho _ { X } - \rho _ { Y } \right\| _ { 1 } .
$$

Thus, accurate tomography of this family would give an accurate estimator of X. The construction and the metric comparison are proved in Section 4.1.

The central estimate is that every joint POVM M on t samples satisfies

$$
\mathrm { T r } \left( \boldsymbol { \mathcal { T } } _ { \mathsf { M } } ^ { ( t ) } ( \boldsymbol { X } ) \right) = O \left( d t \operatorname* { m i n } \left\{ 1 , \frac { \sqrt { t } } { r } \right\} \right) ,\tag{3}
$$

for every X. Here the Fisher matrix uses the $2 r ( d - r )$ real support coordinates. At a node where the protocol jointly measures $t _ { i } ( h ) \leq t$ samples, we apply the same estimate with $t _ { i } ( h )$ in place of t. This estimate is proved in Proposition 4.2.

We begin the proof of Equation (3) with the simpler single-sample case. Write the POVM density $M _ { z }$ , with respect to a reference measure $\nu ,$ in blocks relative to $S \oplus S ^ { \perp }$ :

$$
M _ { z } = \left( \begin{array} { c c } { { A _ { z } } } & { { B _ { z } ^ { \dagger } } } \\ { { B _ { z } } } & { { C _ { z } } } \end{array} \right) \succeq 0 .
$$

Here $A _ { z }$ acts on $S , C _ { z }$ acts on $S ^ { \perp }$ , and $B _ { z }$ maps S to $S ^ { \perp }$ . Let $q _ { X } ( z ) : = \operatorname { T r } ( M _ { z } \rho _ { X } )$ denote the likelihood density of the measurement outcome at parameter X. At the reference point $X = 0$ $\rho _ { 0 } = \mathrm { d i a g } ( \mathrm { I } _ { r } / r , 0 )$ , so

$$
q _ { 0 } ( z ) = { \frac { \operatorname { T r } ( A _ { z } ) } { r } } .\tag{4}
$$

For a support rotation direction $H \in \mathbb { C } ^ { ( d - r ) \times r }$ , the state derivative displayed above gives

$$
\mathrm { D } _ { H } q _ { 0 } \ l ( z ) = \mathrm { T r } \big ( M _ { z } \mathrm { D } _ { H } \rho _ { 0 } \big ) = \frac { 2 } { r } \mathrm { R e } \left( \mathrm { T r } ( B _ { z } ^ { \dagger } H ) \right) .
$$

Thus $A _ { z }$ determines the outcome probability, while $B _ { z }$ determines its first-order response to a support rotation. Summing the squared derivatives over an orthonormal basis of $\mathbb { C } ^ { ( d - r ) \times \bar { r } }$ , regarded as a real inner product space, gives

$$
\| \nabla q _ { 0 } ( z ) \| _ { 2 } ^ { 2 } = \frac { 4 } { r ^ { 2 } } \left\| B _ { z } \right\| _ { \mathrm { F } } ^ { 2 } .\tag{5}
$$

Substituting Equations (4) and (5) into the Fisher information trace formula yields

$$
\mathrm { T r } ( \mathcal { T } _ { \mathsf { M } } ^ { ( 1 ) } ( 0 ) ) = \frac { 4 } { r } \int \frac { \| B _ { z } \| _ { \mathrm { F } } ^ { 2 } } { \mathrm { T r } ( A _ { z } ) } \ \mathrm { d } \nu ( z ) .
$$

$\mathrm { I f } \ \mathrm { T r } ( A _ { z } ) = 0$ , positivity also forces $B _ { z } = 0$ , and the quotient is taken to be zero.

It remains to bound the quotient. Positivity of $M _ { z }$ implies

$$
\Vert B _ { z } \Vert _ { \mathrm { F } } ^ { 2 } \leq \mathrm { T r } ( A _ { z } ) \mathrm { T r } ( C _ { z } ) .
$$

Finally, POVM normalization gives $\begin{array} { r } { \int C _ { z } \ \mathrm { d } \nu ( z ) = \mathrm { I } _ { d - r } . } \end{array}$ and hence

$$
\mathrm { T r } ( \mathcal { Z } _ { \mathsf { M } } ^ { ( 1 ) } ( 0 ) ) \leq \frac { 4 } { r } \int \mathrm { T r } ( C _ { z } ) \ \mathrm { d } \nu ( z ) \leq \frac { 4 d } { r } .
$$

The complete proof of the single-sample bound appears in Section 4.2.1. Both this calculation and the calculation for joint measurements below are carried out at $X = 0$ . At the end of this subsection, we explain how a change of basis transfers the resulting bounds to arbitrary $X$

We now consider a joint POVM on $t > 1$ samples, again writing $M _ { z }$ and ν for its operator density and reference measure. Consider an infinitesimal perturbation that tilts a unit vector $h \in S$ toward a fixed direction in $S ^ { \perp }$ . Diferentiating the product state in this direction gives a sum of t terms, according to which tensor factor is rotated. Squaring this sum in the Fisher information formula produces terms involving either the same position or two distinct positions. The latter are controlled by the components of the other tensor factors in the direction h. For each fixed position of the rotated tensor factor, the relevant sum of projections onto h on the other $t - 1$ factors is

$$
N _ { \pmb { h } } ^ { ( t - 1 ) } : = \sum _ { \ell = 1 } ^ { t - 1 } \mathbf { I } _ { S } ^ { \otimes ( \ell - 1 ) } \otimes | \pmb { h } \rangle \langle \pmb { h } | \otimes \mathbf { I } _ { S } ^ { \otimes ( t - 1 - \ell ) } .
$$

The goal is to repeat the single-sample bound on $\| B _ { z } \| _ { \mathrm { F } } ^ { 2 } / \mathrm { T r } ( A _ { z } )$ . With t samples, the analogous numerator also contains the sum over possible excitation positions. Let $f _ { 1 } , \ldots , f _ { d - r }$ be an orthonormal basis of $S ^ { \perp }$ . For $i \in [ d - r ]$ , let $C _ { z , i }$ be the POVM block on the subspace spanned by tensors with exactly one factor in direction $\pmb { f } _ { i }$ and all remaining factors in S. It plays the role of $C _ { z }$ in the single-sample calculation. If a common operator $T _ { t - 1 , r }$ dominates $N _ { h } ^ { ( t - 1 ) }$ for every unit $h .$ , then placing one copy of $\mathrm { I } + T _ { t - 1 , r }$ at each of the t possible excitation positions gives an operator $Q _ { t , r } ^ { ( i ) }$ on this block. The Cauchy–Schwarz argument for a positive semidefinite block matrix cancels the likelihood denominator and bounds the remaining quotient by $\mathrm { T r } ( C _ { z , i } Q _ { t , r } ^ { ( i ) } )$ ; see Equation (52). For each fixed $i ,$ this bound controls rotations from any unit direction $h \in S$ toward $\mathbf { \Delta } f _ { i }$

We choose $T _ { t - 1 , r }$ , and hence ${ Q } _ { t , r } ^ { ( i ) }$ , independently of the outcome z. POVM normalization then gives

$$
\int \mathrm { T r } ( C _ { z , i } Q _ { t , r } ^ { ( i ) } ) \ \mathrm { d } \nu ( z ) = \mathrm { T r } ( Q _ { t , r } ^ { ( i ) } ) .
$$

Thus the trace of this common upper bound controls the Fisher information. It therefore sufices to find a direction-independent bound on $N _ { h } ^ { ( t - 1 ) }$ with small normalized trace, meaning the trace divided by $r ^ { t - 1 }$ . The immediate bound $N _ { h } ^ { ( t - 1 ) } \preceq ( t - 1 ) \mathrm { I }$ has normalized trace $t - 1$ , which is too large. For $1 \leq k \leq t - 1$ , define

$$
\binom { N _ { h } ^ { ( t - 1 ) } } { k } : = \frac { 1 } { k ! } \prod _ { j = 0 } ^ { k - 1 } \left( N _ { h } ^ { ( t - 1 ) } - j \mathrm { I } _ { S ^ { \otimes ( t - 1 ) } } \right) .
$$

Because the summands defining $N _ { h } ^ { ( t - 1 ) }$ are commuting projectors,

$$
\binom { N _ { h } ^ { ( t - 1 ) } } { k } = \sum _ { \stackrel { J \subseteq [ t - 1 ] } { | J | = k } } \prod _ { \ell \in J } \left( \mathrm { I } _ { S } ^ { \otimes ( \ell - 1 ) } \otimes | h \rangle \langle h | \otimes \mathrm { I } _ { S } ^ { \otimes ( t - 1 - \ell ) } \right) .
$$

For each $J ,$ the product on the right projects the selected k factors onto $\pmb { h } ^ { \otimes k }$ . This vector belongs to the symmetric subspace of those factors, so the product is bounded by the corresponding orthogonal projector $\Pi _ { \mathrm { s y m } , J }$ . Therefore,

$$
\binom { N _ { h } ^ { ( t - 1 ) } } { k } \preceq \Omega _ { t - 1 , k } : = \sum _ { \substack { J \subseteq [ t - 1 ] } } \Pi _ { \mathrm { s y m } , J } .
$$

The operator $\Omega _ { t - 1 , k }$ is independent of h. Since $N _ { h } ^ { ( t - 1 ) }$ is a sum of $t - 1$ commuting projectors, its eigenvalues belong to $\{ 0 , 1 , \ldots , t - 1 \}$ . Using the scalar inequality

$$
q \leq k + k { \binom { q } { k } } ^ { 1 / k } , \qquad q \in \{ 0 , 1 , \dots , t - 1 \} ,
$$

and the bound above, we obtain

$$
\begin{array} { r } { T _ { t - 1 , r } : = k \mathrm { I } _ { S ^ { \otimes ( t - 1 ) } } + k \Omega _ { t - 1 , k } ^ { 1 / k } , \qquad N _ { h } ^ { ( t - 1 ) } \preceq k \mathrm { I } _ { S ^ { \otimes ( t - 1 ) } } + k \binom { N _ { h } ^ { ( t - 1 ) } } { k } ^ { 1 / k } \preceq T _ { t - 1 , r } . } \end{array}
$$

Since the symmetric subspace of $S ^ { \otimes k }$ has dimension $\binom { r + k - 1 } { k }$ , standard binomial estimates with $k : = \lceil \sqrt { t - 1 } \rceil$ give

$$
{ \frac { \mathrm { T r } ( T _ { t - 1 , r } ) } { r ^ { t - 1 } } } = O \left( k + { \frac { t - 1 } { k } } + { \frac { t - 1 } { r } } \right) = O \left( { \sqrt { t } } + { \frac { t } { r } } \right) .
$$

Applying this estimate at each excitation position bounds the trace of $Q _ { t , r } ^ { ( i ) }$ . The block positivity bound and POVM normalization then give

$$
\mathrm { T r } ( \mathcal { Z } _ { \mathsf { M } } ^ { ( t ) } ( 0 ) ) \leq \frac { 4 } { r ^ { t } } \sum _ { i = 1 } ^ { d - r } \int \mathrm { T r } ( C _ { z , i } Q _ { t , r } ^ { ( i ) } ) \ \mathrm { d } \nu ( z ) = \frac { 4 } { r ^ { t } } \sum _ { i = 1 } ^ { d - r } \mathrm { T r } ( Q _ { t , r } ^ { ( i ) } ) = O \left( \frac { d t } { r } \left( \sqrt { t } + \frac { t } { r } \right) \right) .
$$

For $t \le r ^ { 2 }$ , we have $t / r \leq \sqrt { t }$ , so the resulting bound is $O ( d t { \sqrt { t } } / r )$ . For $t > r ^ { 2 }$ , the standard classical– quantum Fisher inequality and additivity of SLD quantum Fisher information give $\mathrm { T r } ( \mathcal { T } _ { \mathsf { M } } ^ { ( t ) } ( 0 ) ) \leq 8 d t$

We have now bounded the Fisher information trace at $X = 0$ for every joint POVM. For an arbitrary X, we reduce to this case by a unitary change of basis and show that the corresponding map on perturbations does not increase their Frobenius norm. We construct a unitary $U _ { X }$ satisfying $U _ { X } ^ { \dagger } \rho _ { X } U _ { X } = \rho _ { 0 }$ . Under this change of basis, every perturbation H at X becomes a perturbation $\mathcal { L } _ { X } ( H )$ at 0, where

$$
\begin{array} { r } { U _ { X } ^ { \dagger } ( { \mathrm { D } } _ { H } \rho _ { X } ) U _ { X } = { \mathrm { D } } _ { \mathcal { L } _ { X } ( H ) } \rho _ { 0 } , \qquad \| \mathcal { L } _ { X } ( H ) \| _ { \mathrm { F } } \leq \| H \| _ { \mathrm { F } } . } \end{array}
$$

If $\widetilde { \mathsf { M } }$ is obtained by conjugating M with $U _ { X } ^ { \otimes t }$ , then

$$
\mathcal { T } _ { \mathsf { M } } ^ { ( t ) } ( X ) = \mathcal { L } _ { X } ^ { \dagger } \mathcal { T } _ { \widetilde { \mathsf { M } } } ^ { ( t ) } ( 0 ) \mathcal { L } _ { X } .
$$

Since $\mathcal { L } _ { X }$ is a contraction,

$$
\mathrm { T r } ( \mathcal { T } _ { \mathsf { M } } ^ { ( t ) } ( X ) ) \leq \mathrm { T r } ( \mathbb { Z } _ { \widetilde { \mathsf { M } } } ^ { ( t ) } ( 0 ) ) .
$$

Applying the bounds proved at $X = 0$ to $\widetilde { \mathsf { M } }$ proves Equation (3). The complete proof appears in Section 4.2.2.

## 2.1.4 Putting the bounds together

Since $t _ { i } ( H _ { i } ) \leq t$ in each round and $\begin{array} { r } { \sum _ { i } t _ { i } ( H _ { i } ) \leq n } \end{array}$ on every transcript, the nodewise estimate and the adaptive Fisher chain rule give

$$
\mathrm { T r } \big ( \mathcal { T } _ { \mathrm { t r } } ( X ) \big ) = O \left( d n \operatorname* { m i n } \left\{ 1 , \frac { \sqrt { t } } { r } \right\} \right) .
$$

This transcript bound is stated formally in Corollary 4.7.

The support rotation parameter has real dimension $p = 2 r ( d - r )$ , and the smoothly truncated Gaussian density described above satisfies $I ( \pi ) = O ( d ^ { 2 } r / a ^ { 2 } )$ . If n is below a suficiently small universal constant multiple of

$$
\frac { d r } { a ^ { 2 } } \operatorname* { m a x } \left\{ 1 , \frac { r } { \sqrt { t } } \right\} ,
$$

then both terms in the van Trees denominator are $O ( d ^ { 2 } r / a ^ { 2 } )$ . The van Trees inequality gives expected squared Frobenius loss $\Omega ( a ^ { 2 } r )$ , and operator norm localization converts this to expected trace norm loss $\Omega ( a r )$

By contrast, the accuracy guarantee for an actual tomography algorithm allows failure with constant probability: it requires trace norm error at most ε with probability at least $2 / 3$ . For arbitrary matrix outputs, the error on failure can be arbitrarily large, so this guarantee alone does not control the expected loss. Let $\delta \in ( 0 , 1 )$ be a target failure probabili $\mathrm { t y , }$ to be chosen as a small constant. Confidence amplification (Lemma 4.11) reduces the failure probability to at most $\delta ,$ with trace norm error at most 3ε, using ${ \cal O } ( 1 + \log ( 1 / \delta ) )$ independent runs on fresh samples.

We then postprocess the amplified output into an estimator $\widehat { X }$ whose operator norm is at most a on every run. This preserves $O ( r \varepsilon )$ trace norm accuracy on the successful runs, as proved in Lemma 4.12. On the remaining failure event, both X and $\widehat { X }$ have operator norm at most $^ { a , }$ so $\left\| { \widehat { X } } - X \right\| _ { 1 } \leq 2 a r$ . Splitting the expectation over these two events gives expected trace norm loss $O ( r \varepsilon ) + \bar { O } ( \delta a r )$ . Taking a to be a suficiently large constant multiple of $\varepsilon ,$ and then taking $\delta$ to be a suficiently small constant, makes this upper bound incompatible with the lower bound $\Omega ( a r )$ above. For this fixed $\delta ,$ the repetitions increase the sample count only by a constant factor. This gives the claimed lower bound

$$
n = \Omega \left( \frac { d r } { \varepsilon ^ { 2 } } \operatorname* { m a x } \left\{ 1 , \frac { r } { \sqrt { t } } \right\} \right)
$$

when $r \leq d / 2 ;$ see Proposition 4.13.

When $r > d / 2$ , the support rotation family has $2 r ( d - r )$ real parameters, which can be too few when r is close to d. We first fix an r-dimensional subspace $V \subseteq \mathbb { C } ^ { d }$ and embed into $V$ the hard family for ambient dimension r and rank $k : = \lfloor r / 2 \rfloor$ . Every embedded input state $\sigma$ then has support contained in the same fixed $V .$ . Let Π<sub>V</sub> be the orthogonal projector onto $V .$ . Mixing each embedded input equally with the maximally mixed state $\Pi _ { V } / r$ gives $\rho _ { \sigma } : = ( \sigma + \Pi _ { V } / r ) / 2$ , whose support is exactly V and whose rank is exactly r. Any algorithm that works for all rank-r states must therefore work for this lifted family. The trace norm distance between any two lifted states is exactly half the distance between their corresponding inputs.

The channel acts independently on each sample, so an adaptive protocol for the lifted family can be simulated on the original family using the same number of samples in each round. Its estimate can be converted back with at most twice the error. Thus the lower bound for dimension $r$ and rank k transfers to the rank-r problem in dimension d. Since $k = \Theta ( r )$ and $r = \Theta ( d )$ , this gives the claimed rate. The reduction is proved in Lemma 4.14.

## 2.2 Upper bound

The upper bound protocol repeatedly applies a Gaussian joint measurement, averages its Hermitian matrix estimates, and projects the average onto the set of density matrices of rank at most r. We first define the measurement and state the complete estimation procedure. We then reduce its analysis to three blocks of the averaged error, explain the measurement properties that control these blocks, and show why the construction has those properties.

## 2.2.1 The joint measurement and estimation procedure

We first define $\mathsf { M } _ { t } , \mathrm { a }$ joint measurement on t samples. For $1 \leq \ell \leq \operatorname* { m i n } \{ d , t \}$ , let $\gamma _ { d , \ell }$ be the standard complex Gaussian law on $\mathbb { C } ^ { d \times \ell }$ , and define

$$
\Gamma _ { \ell , t } : = \int ( G G ^ { \dagger } ) ^ { \otimes t } \mathrm { d } \gamma _ { d , \ell } ( G ) .
$$

Let $\Pi _ { \ell , t }$ be the orthogonal projector onto the support of $\Gamma _ { \ell , t }$ , put $\Pi _ { 0 , t } = \boldsymbol { 0 }$ , and set $\Delta _ { \ell , t } : = \Pi _ { \ell , t } - \Pi _ { \ell - 1 , t }$ For the outcome $J = { \boldsymbol { \ell } } .$ , define

$$
\mathsf { M } _ { t } ( \ell , \mathrm { d } G ) : = \Delta _ { \ell , t } ( \Gamma _ { \ell , t } ^ { + } ) ^ { 1 / 2 } ( G G ^ { \dagger } ) ^ { \otimes t } ( \Gamma _ { \ell , t } ^ { + } ) ^ { 1 / 2 } \Delta _ { \ell , t } ~ \mathrm { d } \gamma _ { d , \ell } ( G ) ,
$$

where $\Gamma _ { \ell , t } ^ { + }$ denotes the Moore–Penrose pseudoinverse of $\Gamma _ { \ell , t }$ . The Gaussian measure $\gamma _ { d , \ell }$ is a reference measure. The Born rule determines the outcome density with respect to this measure. The support projectors $\Pi _ { \ell , t }$ are nested, their ranges span the entire tensor product space, and integrating $\mathsf { M } _ { t } ( \ell , \mathrm { d } G )$ over G gives $\Delta _ { \ell , t } ;$ hence these densities form a POVM. These claims are proved in Lemmas 5.2 and 5.3.

The protocol uses $\mathsf { M } _ { s }$ to jointly measure s samples at a time, where

$$
s : = \operatorname* { m i n } \{ t , r ^ { 2 } \} .
$$

For a fixed total number of samples, increasing the number of samples measured jointly improves the asymptotic error bound only until this number reaches $r ^ { 2 }$ , which explains this choice. For a suficiently large universal constant $C _ { 0 }$ , set

$$
B : = \left\lceil C _ { 0 } \frac { d r } { s \varepsilon ^ { 2 } } \left( 1 + \frac { r } { \sqrt { s } } \right) \right\rceil , \qquad N : = B s .
$$

Here B is the number of measurement rounds and N is the total number of samples. Put $\ell _ { \mathrm { m a x } } : =$ min $\{ d , s \}$ . The measurement $\mathsf { M } _ { s }$ has outcome $( J , G )$ , where $1 \le J \le \ell _ { \mathrm { m a x } }$ and $G \in \mathbb { C } ^ { d \times J }$ . From this outcome, we form the Hermitian matrix

$$
Y : = \frac { G G ^ { \dagger } - J \mathrm { I } _ { d } } { s } .
$$

We apply the same measurement independently B times, using s fresh samples each time. If $Y _ { 1 } , \dots , Y _ { B }$ are the resulting matrices, set

$$
\overline { { Y } } : = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } Y _ { b } .
$$

The output is a density matrix nearest in Frobenius norm to ${ \overline { { Y } } } .$ , subject to having rank at most r:

$$
\widehat { \rho } \in \mathop { \mathrm { a r g } } _ { \sigma \in \mathcal { D } _ { r } ( \mathbb { C } ^ { d } ) } \left\| \overline { { Y } } - \sigma \right\| _ { \mathrm { F } } .
$$

This is the complete protocol. The same joint measurement is used in every repetition, so the protocol is nonadaptive.

## 2.2.2 Reducing the error to three blocks

Fix a state $\rho$ of rank at most $r ,$ choose an r-dimensional subspace containing its support, let Π be the orthogonal projector onto this subspace, and put $\Pi ^ { \perp } : = \Pi _ { d } - \Pi$ and $E : = { \overline { { Y } } } - \rho .$ . The projection bound in Lemma 5.9 gives

$$
\left. \widehat { \rho } - \rho \right. _ { 1 } ^ { 2 } = O \left( r \left( \left. \Pi E \Pi \right. _ { \mathrm { F } } ^ { 2 } + \left. \Pi ^ { \perp } E \Pi \right. _ { \mathrm { F } } ^ { 2 } + r \left. \Pi ^ { \perp } E \Pi ^ { \perp } \right. _ { \mathrm { o p } } ^ { 2 } \right) \right) .
$$

Thus it sufices to control ΠEΠ and $\Pi ^ { \perp } E \Pi$ in Frobenius norm and $\Pi ^ { \perp } E \Pi ^ { \perp }$ in operator norm. The projector Π is used only in this analysis and is not known to the protocol.

## 2.2.3 Properties needed from one joint measurement

The blocks involving Im(Π) are controlled by $\operatorname { V a r } _ { \rho } \bigl ( \operatorname { T r } ( H Y ) \bigr )$ , where H ranges over suitable Hermitian test matrices. Suppose rank $( \rho ) \leq r ,$ , and let $F$ be the swap operator between the two factors of $\mathbb { C } ^ { d } \otimes \mathbb { C } ^ { d }$ . The measurement defined above gives an exactly unbiased estimator with an explicit second moment identity, and the expected number of columns of G is small:

$$
\mathbb { E } _ { \rho } [ Y ] = \rho , \qquad \mathbb { E } _ { \rho } [ Y \otimes Y ] = \frac { s - 1 } { s } \rho ^ { \otimes 2 } + \frac { 1 } { s } ( \rho \otimes  { \mathrm { I } _ { d } } +  { \mathrm { I } _ { d } } \otimes \rho ) F + \frac { \mathbb { E } _ { \rho } [ J ] } { s ^ { 2 } } F , \qquad \mathbb { E } _ { \rho } [ J ] = O ( \sqrt { s } ) .\tag{6}
$$

For every Hermitian matrix H, multiplying the second moment identity by $H \otimes H$ , taking the trace, and using unbiasedness gives

$$
\mathrm { V a r } _ { \rho } \big ( \mathrm { T r } ( H Y ) \big ) = - \frac { 1 } { s } \big ( \mathrm { T r } ( H \rho ) \big ) ^ { 2 } + \frac { 2 } { s } \mathrm { T r } ( \rho H ^ { 2 } ) + { \frac { \mathbb { E } _ { \rho } [ J ] } { s ^ { 2 } } } \| H \| _ { \mathrm { F } } ^ { 2 } .
$$

Dropping the nonpositive first term and using independence for the averaged error $E = { \overline { { Y } } } - \rho$ give

$$
\mathrm { V a r } _ { \rho } \big ( \mathrm { T r } ( H Y ) \big ) \leq \frac { 2 } { s } \mathrm { T r } ( \rho H ^ { 2 } ) + \frac { \mathbb { E } _ { \rho } [ J ] } { s ^ { 2 } } \| H \| _ { \mathrm { F } } ^ { 2 } , \qquad \mathbb { E } _ { \rho } \big [ \mathrm { T r } ( H E ) ^ { 2 } \big ] \leq \frac { 2 } { N } \mathrm { T r } ( \rho H ^ { 2 } ) + \frac { \mathbb { E } _ { \rho } [ J ] } { N s } \| H \| _ { \mathrm { F } } ^ { 2 } .\tag{7}
$$

The second inequality in Equation (7) controls the Frobenius norms of ΠEΠ and $\Pi ^ { \perp } E \Pi$ . For a fixed total number of samples $N .$ , its first term on the right is independent of s. Since $\mathbb { E } _ { \rho } [ J ] = O ( \sqrt { s } )$ the second term is $O ( \| H \| _ { \mathrm { F } } ^ { 2 } / ( N \sqrt { s } ) )$ . This factor of $s ^ { - 1 / 2 }$ is the gain from measuring s samples jointly. The moment identities and the variance bound are proved in Section 5.2.3; the estimate on $\mathbb { E } _ { \rho } [ J ]$ is proved in Lemma 5.5.

The $\Pi ^ { \perp } E \Pi ^ { \perp }$ block requires a diferent property: the projection bound asks for control in operator norm, which the above variance estimate does not provide. The additional property is also part of Proposition 5.1: for every ℓ with $\mathbb { P } _ { \rho } [ J = \ell ] > 0$ , conditional on $J = \ell ,$ the columns of $\Pi ^ { \perp } G$ remain independent standard complex Gaussian vectors in Im $( \Pi ^ { \perp } )$ . If $r = d ,$ then $\Pi ^ { \perp } = 0$ and there is no complementary block. Otherwise, for independent outcomes $( J _ { 1 } , G _ { 1 } ) , \dots , ( J _ { B } , G _ { B } )$ , put $K : = \textstyle \sum _ { b } J _ { b }$ and concatenate the coordinate matrices of $\Pi ^ { \perp } G _ { b } .$ , in an orthonormal basis of $\mathrm { I m } ( \Pi ^ { \perp } )$ into $\bar { Z } \in \mathbb { C } ^ { ( d - r ) \times K }$ . Conditional on $J _ { 1 } , \ldots , J _ { B }$ , the matrix Z is standard complex Gaussian. In these coordinates,

$$
\Pi ^ { \perp } E \Pi ^ { \perp } = \frac { 1 } { N } ( Z Z ^ { \dagger } - K \mathrm { I } _ { d - r } ) ,
$$

so controlling this block reduces to bounding the deviation $Z Z ^ { \dagger } - K  { \mathrm { I } _ { d - r } }$ in operator norm. The required bound is Lemma 5.10. The resulting bounds in Frobenius norm and operator norm for these blocks are proved in Lemma 5.11.

## 2.2.4 Why the joint measurement has these properties

We first verify that $\mathsf { M } _ { s }$ is a POVM. Write $\displaystyle \mathcal { V } _ { \ell , s }$ for the span of all $L ^ { \otimes s }$ with $L \subseteq \mathbb { C } ^ { d }$ and dim $L \leq \ell ,$ with $\mathcal { V } _ { 0 , s } = \{ 0 \}$ . Positivity, continuity, and the full support of the reference Gaussian measure show that $\Gamma _ { \ell , s }$ has support $\nu _ { \ell , s }$ . These subspaces are nested and span the tensor product space, so the diferences $\Delta _ { \ell , s }$ are projectors that sum to the identity. The pseudoinverse factors ensure that integrating $\mathsf { M } _ { s } ( \ell , \mathrm { d } G )$ over $G$ gives $\Delta _ { \ell , s } ,$ , establishing the validity of the measurement; see Lemmas 5.2 and 5.3.

To bound $\mathbb { E } _ { \rho } [ J ]$ , we use the distribution of J obtained from this normalization. The Born rule gives

$$
\begin{array} { r } { \mathbb { P } _ { \rho } [ J = \ell ] = \operatorname { T r } ( \Delta _ { \ell , s } \rho ^ { \otimes s } ) , \qquad 1 \leq \ell \leq \ell _ { \operatorname* { m a x } } , } \end{array}
$$

and hence, for $1 \leq q \leq \ell _ { \mathrm { m a x } }$

$$
\begin{array} { r } { { \mathbb P } _ { \rho } [ J \ge q ] = { \mathrm { T r } } \big ( \big ( \mathrm { I } _ { d } ^ { \otimes s } - \Pi _ { q - 1 , s } \big ) \rho ^ { \otimes s } \big ) . } \end{array}
$$

For each q-element subset $T \subseteq [ s ]$ , let $A _ { T }$ be the orthogonal projector that antisymmetrizes the tensor factors in $T .$ . We bound the operator in this tail probability by a sum of these projectors:

$$
\begin{array} { l l } { \mathcal { V } _ { q - 1 , s } = \displaystyle \bigcap _ { T \subseteq [ s ] } \ker ( A _ { T } ) , \qquad } & { \mathrm { I } _ { d } ^ { \otimes s } - \Pi _ { q - 1 , s } \preceq \displaystyle \sum _ { T \subseteq [ s ] } A _ { T } . } \\ { \qquad \quad | T | = q } & \end{array}
$$

Both statements are proved in Lemmas $5 . 4$ and 5.5.

Let $A _ { q }$ be the antisymmetrizer on $( \mathbb { C } ^ { d } ) ^ { \otimes q }$ , and let $\lambda _ { 1 } , \ldots , \lambda _ { d }$ be the eigenvalues of $\rho .$ Each of the $\textstyle { \binom { s } { q } }$ projectors has the same expectation $\operatorname { T r } ( A _ { q } \rho ^ { \otimes q } )$ in the product state $\rho ^ { \otimes s }$ , so

$$
\mathbb { P } _ { \rho } [ J \geq q ] \leq \binom { s } { q } \operatorname { T r } ( A _ { q } \rho ^ { \otimes q } ) = \binom { s } { q } \sum _ { 1 \leq i _ { 1 } < \cdots < i _ { q } \leq d } \lambda _ { i _ { 1 } } \cdot \cdot \cdot \lambda _ { i _ { q } } \leq \frac { \binom { s } { q } } { q ! } \left( \sum _ { i = 1 } ^ { d } \lambda _ { i } \right) ^ { q } = \frac { \binom { s } { q } } { q ! } .
$$

The bound

$$
\frac { { \binom { s } { q } } } { q ! } \leq \left( \frac { \mathrm { e } ^ { 2 } s } { q ^ { 2 } } \right) ^ { q }
$$

shows that the tail decreases geometrically once q is a suficiently large constant multiple of ${ \sqrt { s } } .$ Summing the tail probabilities gives $\mathbb { E } _ { \rho } [ J ] = O ( \sqrt { s } )$ . The full argument is in Lemma 5.5.

We next compute the first and second moments of Y . The main tool is a permutation expansion of Gaussian tensor moments. Let ${ \mathfrak { S } } _ { s }$ be the permutations of $[ s ]$ , let $U _ { \pi }$ permute the tensor factors according to $\pi ,$ and let $c ( \pi )$ count its cycles, including fixed points. The complex form of Isserlis’s theorem [Iss18] gives

$$
\Gamma _ { \ell , s } = \sum _ { \pi \in \mathfrak { S } _ { s } } \ell ^ { c ( \pi ) } U _ { \pi } .
$$

This identity and the resulting commutation properties are proved in Lemma 5.6.

For the contribution from $J = \ell ,$ the Born rule leads to Gaussian integrals containing $( G G ^ { \dagger } ) ^ { \otimes s }$ from the POVM and one or two factors $( G G ^ { \dagger } - \ell \mathbf { I } _ { d } ) / s$ from the estimator. For the first moment, label the measured tensor factors by $1 , \ldots , s$ and place the estimator matrix on an auxiliary output tensor factor $\mathbb { C } _ { a } ^ { d }$ . For $i \in [ s ]$ , let $F _ { a i }$ be the swap operator between this auxiliary factor and input factor i. Writing $\mathrm { I } _ { a }$ for the identity on this auxiliary factor and setting aside the scalar $1 / s$ , the required Gaussian identity is

$$
\int ( G G ^ { \dagger } - \ell \mathrm { I } _ { d } ) _ { a } \otimes ( G G ^ { \dagger } ) ^ { \otimes s } ~ \mathrm { d } \gamma _ { d , \ell } ( G ) = \sum _ { i = 1 } ^ { s } F _ { a i } ( \mathrm { I } _ { a } \otimes \Gamma _ { \ell , s } ) .
$$

To prove this equality, apply the permutation expansion to the Gaussian integral of $( G G ^ { \dagger } ) ^ { \otimes ( s + 1 ) }$ with factors labelled $a , 1 , \ldots , s .$ . Subtracting $\ell  { \mathrm { I } _ { d } }$ in factor a cancels exactly the permutations that fix a. Every remaining permutation is obtained by inserting a into a cycle of a permutation of [s], immediately before one of the s input labels. Summing over these labels gives the right side. For the second moment, apply the same argument to $( G G ^ { \dagger } ) ^ { \otimes ( s + 2 ) }$ , with auxiliary factors a and b. Centering both factors cancels the permutations that fix either one. The remaining permutations insert a and b before distinct input labels, consecutively before the same input label in either order, or together in a two-cycle.

We now use these Gaussian identities in the Born rule calculation. In each layer, the pseudoinverse square roots cancel $\Gamma _ { \ell , s } ,$ , leaving $\Delta _ { \ell , s } \rho ^ { \otimes s }$ . This uses the support and commutation properties in Lemmas 5.2 and 5.6. Summing over the layers gives

$$
\sum _ { \ell = 1 } ^ { \ell _ { \mathrm { { m a x } } } } \Delta _ { \ell , s } \rho ^ { \otimes s } = \rho ^ { \otimes s } .
$$

The first moment calculation therefore gives $\mathbb { E } _ { \rho } [ Y ] = \rho$ . With two output factors, the three types of permutations above contribute, respectively,

$$
{ \frac { s - 1 } { s } } \rho ^ { \otimes 2 } , \qquad { \frac { 1 } { s } } ( \rho \otimes \mathrm { I } _ { d } + \mathrm { I } _ { d } \otimes \rho ) F , \qquad { \frac { \mathbb { E } _ { \rho } [ J ] } { s ^ { 2 } } } F .
$$

Their sum is the second moment in Equation (6). The full calculation is given in Lemmas 5.7 and 5.8 and in the proof of Proposition 5.1.

Finally, we verify the conditional Gaussian law used to control $\Pi ^ { \perp } E \Pi ^ { \perp }$ . Fix $J = \ell$ with positive probability and recall that $\Pi \rho = \rho$ . The support and commutation properties above imply that the Born rule density of $G ,$ , relative to $\gamma _ { d , \ell }$ , depends only on ΠG. Under this reference measure, ΠG and $\Pi ^ { \perp } G$ are independent. The outcome density therefore changes only the distribution of ΠG. Conditional on $J = { \boldsymbol { \ell } } .$ , the matrix $\Pi ^ { \perp } G$ remains independent of $\Pi G ,$ with independent standard complex Gaussian columns in Im $( \Pi ^ { \perp } )$ . This proves the property needed for the complementary block; see the proof of Proposition 5.1.

## 2.2.5 Putting the bounds together

The variance bound in Equation (7), together with $\mathbb { E } _ { \rho } [ J ] = O ( \sqrt { s } )$ , gives

$$
\mathbb { E } _ { \rho } \left[ \left. \Pi E \Pi \right. _ { \mathrm { F } } ^ { 2 } + 2 \left. \Pi ^ { \perp } E \Pi \right. _ { \mathrm { F } } ^ { 2 } \right] = O \left( \frac d N + \frac { d r } { N \sqrt { s } } \right) .
$$

Applying the operator norm covariance bound to the conditional distribution of the concatenated matrix $Z$ gives

$$
\mathbb { E } _ { \rho } \left. \Pi ^ { \perp } E \Pi ^ { \perp } \right. _ { \mathrm { o p } } ^ { 2 } = O \left( \frac { d } { N \sqrt { s } } + \frac { d ^ { 2 } } { N ^ { 2 } } \right) .
$$

When $r = d , \mathbb { E } _ { \rho } \left\| \Pi ^ { \perp } E \Pi ^ { \perp } \right\| _ { \mathrm { o p } } ^ { 2 } = 0$ . These estimates are proved in Lemma 5.11.

Substituting them into the projection bound gives

$$
\mathbb { E } _ { \rho } \left\| \widehat { \rho } - \rho \right\| _ { 1 } ^ { 2 } = O \left( \frac { d r } { N } + \frac { d r ^ { 2 } } { N \sqrt { s } } + \frac { d ^ { 2 } r ^ { 2 } } { N ^ { 2 } } \right) .
$$

This bound also explains the choice $s = \operatorname* { m i n } \{ t , r ^ { 2 } \}$ . For fixed $N _ { ; }$ , the sum of the first two terms reaches order $d r / N$ at $s = r ^ { 2 } ;$ increasing s further changes this sum by at most a constant factor.

Using larger joint measurements would not improve the resulting asymptotic rate. With the stated choice of B and suficiently large $C _ { 0 }$ , Markov’s inequality gives trace norm error at most ε with probability at least $2 / 3$ , using the claimed number of samples. The same joint POVM is used in every round, so the protocol is nonadaptive. The complete argument is in Section 5.3.

## 3 Preliminaries

We collect notation for matrices, measurements, tensor products, and Gaussian matrices, and then recall the classical and quantum Fisher information facts used in the lower bound. The adaptive Fisher information chain rule and the van Trees inequality are introduced later, where they first enter the proof.

## 3.1 States, matrix spaces, and measurements

Throughout, all Hilbert spaces are finite dimensional and complex. For a positive integer n, write $[ n ] : = \{ 1 , \dots , n \}$ . We write $\mathrm { i } : = \sqrt { - 1 }$ for the imaginary unit and e for Euler’s number. For $\mathbf { \boldsymbol { u } } , \mathbf { \boldsymbol { v } } \in \mathcal { H }$ we write $\langle { \pmb u } , { \pmb v } \rangle$ for their inner product and use the convention $\langle \pmb { u } , \pmb { v } \rangle = \pmb { u } ^ { \dag } \pmb { v }$ . We use bold lowercase symbols for vectors and ordinary lowercase symbols for their scalar coordinates. For a linear map $A : \mathcal { H } _ { 1 } \to \mathcal { H } _ { 2 }$ , we write $A ^ { \dagger }$ for its adjoint. Its kernel, image, and rank are

$$
\ker ( A ) : = \{ \pmb { u } \in \mathscr { H } _ { 1 } : A \pmb { u } = 0 \} , \qquad \operatorname { I m } ( A ) : = \{ A \pmb { u } : \pmb { u } \in \mathscr { H } _ { 1 } \} , \qquad \operatorname { r a n k } ( A ) : = \dim \operatorname { I m } ( A ) .
$$

When $\mathcal { H } _ { 1 } = \mathcal { H } _ { 2 }$ , we write $\operatorname { T r } ( A )$ for its trace. For a subspace $u \subseteq \mathcal { H }$ , let $\mathcal { U } ^ { \perp }$ denote its orthogonal complement and let $\Pi _ { \mathcal { U } }$ denote the orthogonal projector onto U. Thus

$$
\mathrm { I m } ( \Pi _ { \mathcal { U } } ) = \mathcal { U } , \qquad \Pi _ { \mathcal { U } } ^ { \dagger } = \Pi _ { \mathcal { U } } , \qquad \Pi _ { \mathcal { U } } ^ { 2 } = \Pi _ { \mathcal { U } } .
$$

We generally use the letter Π for projectors; subscripts specify the subspace or parameter when needed. The notation $A \succeq 0$ means that A is positive semidefinite. For Hermitian operators A and $B ,$ , we write $A \preceq B$ if $B - A \succeq 0$ . A density matrix, or a quantum state, on a Hilbert space H is an operator $\rho \succeq 0$ with $\operatorname { T r } ( \rho ) = 1$ . We write $\mathcal { D } ( \mathcal { H } )$ for the set of density matrices on ${ \mathcal { H } } ,$ and $\mathrm { I } _ { \mathcal { H } }$ for the identity operator on H. In particular, $\mathrm { I } _ { d }$ denotes the identity on $\mathbb { C } ^ { d }$ . For a positive integer $r \leq \dim \mathcal { H }$ , write

$$
\begin{array} { r } { \mathcal D _ { r } ( \mathcal H ) : = \{ \rho \in \mathcal D ( \mathcal H ) : \mathrm { r a n k } ( \rho ) \leq r \} . } \end{array}
$$

For a positive semidefinite operator $A ,$ its support is the subspace supp $( A ) : = \operatorname { I m } ( A ) = \ker ( A ) ^ { \perp }$ For a projector Π on $\mathcal { H } .$ , write $\Pi ^ { \perp } : = \operatorname { I } _ { \mathcal { H } } - \Pi$ for the projector onto $\mathrm { I m } ( \Pi ) ^ { \perp }$

For a matrix H, its Frobenius norm is

$$
\Vert H \Vert _ { \mathrm { F } } : = { \sqrt { \mathrm { T r } ( H ^ { \dagger } H ) } } .
$$

Its trace norm, nuclear norm, or Schatten-1 norm, is

$$
\begin{array} { r } { \| H \| _ { 1 } : = \operatorname { T r } \left( \sqrt { H ^ { \dagger } H } \right) . } \end{array}
$$

We also write $| H | : = { \sqrt { H ^ { \dagger } H } }$ for its absolute value. For a vector ${ \mathbf { } } ^ { \mathbf { } } \mathbf { \Delta } ^ { \mathbf { } } \mathbf { u } ,$ write $\| \boldsymbol { \mathbf { \mathit { u } } } \| : = \sqrt { \langle \boldsymbol { \mathbf { \mathit { u } } } , \boldsymbol { \mathbf { u } } \rangle }$ . The operator norm of a linear map A is

$$
\| A \| _ { \mathrm { o p } } : = \operatorname* { s u p } _ { \| \pmb { u } \| = 1 } \| A \pmb { u } \| .
$$

For a complex number $z , \operatorname { R e } ( z )$ denotes its real part. For positive integers $m , r .$ , we regard the complex matrix space $\mathbb { C } ^ { m \times r }$ as a real Euclidean space with inner product

$$
\langle H , K \rangle _ { \mathbb { R } } : = \operatorname { R e } \left( \operatorname { T r } ( H ^ { \dagger } K ) \right) .\tag{8}
$$

For $1 \leq i \leq m$ and $1 \leq j \leq r$ , let $E _ { i j } \in \mathbb { C } ^ { m \times r }$ denote the matrix whose $( i , j )$ entry is one and whose other entries are zero. Then

$$
\{ E _ { i j } , \mathrm { i } E _ { i j } : 1 \le i \le m , \ 1 \le j \le r \}\tag{9}
$$

is an orthonormal basis of $\mathbb { C } ^ { m \times r }$ regarded as a real inner product space, with the inner product defined in Equation (8). The basis matrices $E _ { i j }$ and $\mathrm { i } E _ { i j }$ correspond to the real and imaginary coordinates of the $( i , j )$ entry, respectively.

For a diferentiable real-valued function f on this matrix space, write $X _ { i j } = x _ { i j } + \mathrm { i } y _ { i j }$ , with $x _ { i j } , y _ { i j } \in \mathbb { R }$ . Its gradient consists of the partial derivatives with respect to these 2mr real coordinates. We also write this gradient as a matrix whose $( i , j )$ entry is $\partial f / \partial x _ { i j } + \mathrm { i } \partial f / \partial y _ { i j }$ . The Euclidean norm $\| \nabla f ( X ) \| _ { 2 }$ equals the Frobenius norm of this matrix.

Let $( \boldsymbol { \mathbb { Z } } , \boldsymbol { \mathcal { Z } } )$ be a measurable outcome space, where Z is the set of possible outcomes and Z is its sigma-algebra of measurable events. A positive operator-valued measure (POVM) on $( Z , { \mathcal { Z } } )$ is a countably additive map M from $\mathcal { Z }$ to positive semidefinite operators on $\mathcal { H } .$ , normalized by

$$
\mathsf { M } ( \mathsf { Z } ) = \mathrm { I } _ { \mathcal { H } } .
$$

When a state $\rho$ is measured by M, the probability of observing an outcome in a measurable event $\mathsf { E } \in \mathcal { Z }$ is

$$
\mathbb { P } _ { \rho } ^ { \mathsf { M } } ( \mathsf { E } ) : = \operatorname { T r } \big ( \rho \mathsf { M } ( \mathsf { E } ) \big ) .\tag{10}
$$

## 3.2 Tensor permutations and Gaussian matrices

For an operator L on H and $\ell \in [ n ]$ , we write

$$
L ^ { [ \ell ] } : = \Gamma _ { \mathcal { H } } ^ { \otimes ( \ell - 1 ) } \otimes L \otimes \Gamma _ { \mathcal { H } } ^ { \otimes ( n - \ell ) }
$$

for the operator on $\mathcal { H } ^ { \otimes n }$ that acts as L on the ℓ-th tensor factor and as the identity on every other factor. The ambient tensor power will always be clear from context.

Let ${ \mathfrak { S } } _ { n }$ be the symmetric group on [n]. For $\pi \in { \mathfrak { S } } _ { n } ,$ let $U _ { \pi }$ denote the operator on $\mathcal { H } ^ { \otimes n }$ defined by

$$
U _ { \pi } ( { \pmb v } _ { 1 } \otimes \cdots \otimes { \pmb v } _ { n } ) : = { \pmb v } _ { \pi ^ { - 1 } ( 1 ) } \otimes \cdots \otimes { \pmb v } _ { \pi ^ { - 1 } ( n ) } .
$$

This convention gives $U _ { \pi } U _ { \tau } = U _ { \pi \tau }$ . The swap operator on two tensor factors is denoted by $F ;$ thus $F ( \pmb { u } \otimes \pmb { v } ) = \pmb { v } \otimes \pmb { u }$ and

$$
\mathrm { T r } \bigl ( F ( A \otimes B ) \bigr ) = \mathrm { T r } ( A B )\tag{11}
$$

for operators A, B on the same space.

For an operator K on n tensor factors and $T \subseteq [ n ] , \operatorname { T r } _ { T } ( K )$ denotes the partial trace over the factors in $T .$ , defined by linearity and

$$
\mathrm { T r } _ { T } ( A _ { 1 } \otimes \cdot \cdot \cdot \otimes A _ { n } ) : = \left( \prod _ { i \in T } \mathrm { T r } ( A _ { i } ) \right) \bigotimes _ { i \in [ n ] \backslash T } A _ { i } ,
$$

where $A _ { i }$ acts on the i-th factor and the remaining factors retain their original order.

For a positive semidefinite operator A, we write $A ^ { + }$ for its Moore–Penrose pseudoinverse. Thus $( A ^ { + } ) ^ { 1 / 2 }$ acts as $A ^ { - 1 / 2 }$ on the support of A and as zero on its kernel.

A standard complex Gaussian scalar is $z = x + \mathrm { i } y$ , where x and y are independent real Gaussian random variables with mean zero and variance $1 / 2$ . A standard complex Gaussian matrix has independent entries with this distribution. We write $\gamma _ { d , \ell }$ for the probability measure of a standard complex Gaussian matrix in $\mathbb { C } ^ { d \times \ell }$

## 3.3 Classical Fisher information

We first recall the classical definition. Let $\Theta \subseteq \mathbb { R } ^ { p }$ be an open parameter set, write $\pmb \theta : = ( \theta _ { 1 } , \ldots , \theta _ { p } )$ and let $\{ \mathbb { P } _ { \pmb { \theta } } : \pmb { \theta } \in \Theta \}$ be a family of probability measures on $( Z , { \mathcal { Z } } )$ with likelihood densities $q _ { \pmb { \theta } }$ relative to a parameter-independent measure ν:

$$
\mathbb { P } _ { \pmb { \theta } } ( \mathsf { E } ) = \int _ { \mathsf { E } } q _ { \pmb { \theta } } ( z ) \ \mathrm { d } \nu ( z )\tag{12}
$$

for every $\mathsf { E } \in { \mathcal { Z } }$ . Assume that $\theta \mapsto q _ { \theta } ( z )$ is diferentiable for ν-almost every z. The measure-theoretic conventions and regularity conditions used for continuous outcomes are collected in Section A. When $\nu$ is counting measure on a discrete outcome space, the integrals below become sums and $q _ { \theta }$ is the probability mass function. The likelihood gradient is

$$
\nabla q _ { \pmb { \theta } } ( z ) : = \big ( \partial _ { 1 } q _ { \pmb { \theta } } ( z ) , \dots , \partial _ { p } q _ { \pmb { \theta } } ( z ) \big ) ^ { \top } .
$$

For a direction $v \in \mathbb { R } ^ { p }$ , we use the notation

$$
\mathrm { D } _ { \pmb { v } } q _ { \pmb { \theta } } ( z ) : = \left. \frac { \mathrm { d } } { \mathrm { d } s } q _ { \pmb { \theta } + s \pmb { v } } ( z ) \right| _ { s = 0 } = \pmb { v } ^ { \top } \nabla q _ { \pmb { \theta } } ( z ) .
$$

The Fisher information matrix is defined as

$$
\mathcal { T } ( \pmb \theta ) : = \int _ { \mathsf { Z } } \frac { \nabla q _ { \pmb \theta } ( z ) \nabla q _ { \pmb \theta } ( z ) ^ { \mathsf { T } } } { q _ { \pmb \theta } ( z ) } \mathrm { d } \nu ( z ) .\tag{13}
$$

The integrand in Equation (13) is defined to be zero when $q _ { \pmb { \theta } } ( z ) = 0$

On the set where $q _ { \pmb { \theta } } ( z ) > 0$ , the score vector is

$$
\mathbf { \delta } _ { \pmb { s } _ { \pmb { \theta } } } ( z ) : = \nabla \log q _ { \pmb { \theta } } ( z ) = \frac { \nabla q _ { \pmb { \theta } } ( z ) } { q _ { \pmb { \theta } } ( z ) } .
$$

Consequently,

$$
\mathcal { T } ( \pmb { \theta } ) = \int _ { \mathsf { Z } } \pmb { s } _ { \pmb { \theta } } ( z ) \pmb { s } _ { \pmb { \theta } } ( z ) ^ { \mathsf { T } } q _ { \pmb { \theta } } ( z ) \ \mathrm { d } \nu ( z ) ,
$$

with the score set to zero on the zero-likelihood set. Equivalently, the Fisher information in a direction v is

$$
\boldsymbol { v } ^ { \top } \mathcal { T } ( \boldsymbol { \theta } ) \boldsymbol { v } = \int _ { \boldsymbol { Z } } \frac { ( \mathrm { D } \boldsymbol { v } q _ { \boldsymbol { \theta } } ( \boldsymbol { z } ) ) ^ { 2 } } { q _ { \boldsymbol { \theta } } ( \boldsymbol { z } ) } \mathrm { d } \nu ( \boldsymbol { z } ) .
$$

The quantity used in our lower bound is the Fisher information trace Tr $\left( { \mathcal { T } } ( \theta ) \right)$ . It is given by

$$
\mathrm { T r } ( \mathcal { Z } ( \pmb { \theta } ) ) = \int _ { \mathbb { Z } } \frac { \| \nabla q _ { \pmb { \theta } } ( z ) \| _ { 2 } ^ { 2 } } { q _ { \pmb { \theta } } ( z ) } ~ \mathrm { d } \nu ( z ) = \sum _ { j = 1 } ^ { p } \int _ { \mathbb { Z } } \frac { ( \partial _ { j } q _ { \pmb { \theta } } ( z ) ) ^ { 2 } } { q _ { \pmb { \theta } } ( z ) } ~ \mathrm { d } \nu ( z ) .\tag{14}
$$

Here $\left\| \cdot \right\| _ { 2 }$ is the Euclidean norm on $\mathbb { R } ^ { p }$

## 3.4 Quantum Fisher information

We first apply the classical definition to the outcome distribution of a quantum measurement. To diferentiate likelihoods at a fixed outcome, we write all outcome laws relative to one measure that does not depend on the state parameter. In finite dimensions, every POVM provides such a measure automatically. Let d := dim H, let M be a POVM on (Z, Z), and define

$$
\nu ( \mathsf E ) : = \frac { 1 } { d } \mathrm { T r } \big ( \mathsf { M } ( \mathsf E ) \big ) .\tag{15}
$$

Finite dimensionality guarantees a positive semidefinite operator density $M _ { z }$ such that

$$
\mathsf { M } ( \mathsf { E } ) = \int _ { \mathsf { E } } M _ { z } \ \mathrm { d } \nu ( z ) , \qquad \int _ { \mathsf { Z } } M _ { z } \ \mathrm { d } \nu ( z ) = \mathrm { I } _ { \mathcal { H } } .\tag{16}
$$

This dominated POVM formulation is standard in quantum statistics; see, for example, [BNG00].   
Its short proof is included in Section A.1.

Let $\theta \mapsto \rho _ { \theta }$ be a diferentiable family of states, and let $\mathbb { P } _ { \theta } : = \mathbb { P } _ { \rho _ { \theta } } ^ { \mathsf { M } }$ be its outcome law. Combining Equations (10) and (16) gives

$$
\mathbb { P } _ { \pmb { \theta } } ( \mathsf { E } ) = \int _ { \mathsf { E } } \mathrm { T r } ( M _ { z } \rho _ { \pmb { \theta } } ) \ \mathrm { d } \nu ( z ) .
$$

Therefore the likelihood density is

$$
q _ { \pmb \theta } ( z ) : = \mathrm { T r } ( M _ { z } \rho _ { \pmb \theta } ) .\tag{17}
$$

For $v \in \mathbb { R } ^ { p }$ , write

$$
\mathrm { D } _ { v } \rho _ { \theta } : = \left. \frac { \mathrm { d } } { \mathrm { d } s } \rho _ { \theta + s v } \right. _ { s = 0 } .
$$

Since the right side of Equation (17) is linear in the state, its directional derivative is

$$
\mathrm { D } _ { v } q _ { \pmb { \theta } } ( z ) = \mathrm { T r } ( M _ { z } \mathrm { D } _ { v } \rho _ { \pmb { \theta } } ) .\tag{18}
$$

We denote the resulting classical Fisher information matrix by ${ \mathcal { T } } _ { \mathsf { M } } ( \theta )$ . Although the formulas use the particular dominating measure ν and density $M _ { z }$ , this Fisher matrix depends only on the outcome distributions generated by M and $\rho _ { \pmb { \theta } }$ . The subscript M denotes the measurement that induces this classical Fisher information matrix.

The quantum Fisher information matrix is instead associated directly with the family $\theta \mapsto \rho _ { \theta }$ and does not depend on a choice of measurement. For $j \in [ p ]$ , let $e _ { j }$ denote the $j { \mathrm { - t h } }$ standard basis vector of $\mathbb { R } ^ { p }$ , and write $\partial _ { j } \rho _ { \theta } : = \mathrm { D } _ { e _ { j } } \rho _ { \theta }$ . A symmetric logarithmic derivative for the j-th parameter is a Hermitian operator $L _ { j }$ satisfying

$$
\partial _ { j } \rho _ { \pmb \theta } = \frac { 1 } { 2 } \big ( L _ { j } \rho _ { \pmb \theta } + \rho _ { \pmb \theta } L _ { j } \big ) .\tag{19}
$$

In finite dimensions, such an operator exists for a diferentiable family of states. When $\rho _ { \pmb { \theta } }$ is not full rank, $L _ { j }$ need not be unique, but the matrix defined below is independent of the choice. The quantum Fisher information matrix is

$$
\big ( \mathbb { Z } _ { \mathrm { Q } } ( \pmb { \theta } ) \big ) _ { j k } : = \frac { 1 } { 2 } \operatorname { T r } \big ( \rho _ { \pmb { \theta } } \big ( L _ { j } L _ { k } + L _ { k } L _ { j } \big ) \big ) .\tag{20}
$$

This is a real symmetric positive semidefinite matrix. The subscript Q distinguishes this measurementindependent quantum quantity from the measurement-induced classical Fisher matrix $\mathcal { T } _ { \mathsf { M } }$

The quantum Fisher information matrix bounds from above the classical Fisher information obtainable from any parameter-independent POVM:

$$
{ \mathcal { T } } _ { \mathsf { M } } ( \pmb { \theta } ) \preceq { \mathcal { T } } _ { \mathrm { Q } } ( \pmb { \theta } ) .\tag{21}
$$

See [BC94, GM00] for the corresponding classical–quantum Fisher information comparison.

We will also use additivity under tensor products. Fix a positive integer $t ,$ and denote the quantum Fisher information matrix of the product family $\rho _ { \pmb { \theta } } ^ { \otimes t }$ by $\mathcal { T } _ { \mathrm { Q } } ^ { ( t ) } ( \pmb { \theta } )$ , where the superscript (t) denotes the number of samples. A symmetric logarithmic derivative for the j-th parameter of this family is

$$
L _ { j } ^ { ( t ) } : = \sum _ { \ell = 1 } ^ { t } L _ { j } ^ { [ \ell ] } .
$$

Indeed, diferentiating the product state and applying Equation (19) in each tensor factor gives

$$
\partial _ { j } \rho _ { \pmb { \theta } } ^ { \otimes t } = \frac { 1 } { 2 } \left( L _ { j } ^ { ( t ) } \rho _ { \pmb { \theta } } ^ { \otimes t } + \rho _ { \pmb { \theta } } ^ { \otimes t } L _ { j } ^ { ( t ) } \right) .
$$

Taking the trace in Equation (19) and using $\operatorname { T r } ( \rho _ { \boldsymbol { \theta } } ) = 1$ gives

$$
\operatorname { T r } ( \rho _ { \theta } L _ { j } ) = \operatorname { T r } ( \partial _ { j } \rho _ { \theta } ) = \partial _ { j } \operatorname { T r } ( \rho _ { \theta } ) = 0 .
$$

Consequently, for distinct $\ell , \ell ^ { \prime } \in [ t ]$

$$
\mathrm { T r } ( \rho _ { \theta } ^ { \otimes t } L _ { j } ^ { [ \ell ] } L _ { k } ^ { [ \ell ^ { \prime } ] } ) = \mathrm { T r } ( \rho _ { \theta } L _ { j } ) \mathrm { T r } ( \rho _ { \theta } L _ { k } ) = 0 .
$$

When we apply Equation (20) to the product family using ${ L } _ { j } ^ { ( t ) }$ and $L _ { k } ^ { ( t ) }$ , the cross terms between distinct tensor factors vanish, while the t terms with $\ell = \ell ^ { \prime }$ each give the corresponding entry of the single-sample quantum Fisher information matrix. Consequently,

$$
\begin{array} { r } { \mathcal { T } _ { \mathrm { Q } } ^ { ( t ) } ( \pmb { \theta } ) = t \mathcal { T } _ { \mathrm { Q } } ( \pmb { \theta } ) . } \end{array}\tag{22}
$$

For a joint POVM M on t samples, denote the induced classical Fisher information matrix by $\mathcal { T } _ { \mathsf { M } } ^ { ( t ) } ( \pmb { \theta } )$ the same superscript again denotes the number of samples measured jointly. Then

$$
\mathrm { T r } \big ( \mathcal { T } _ { \mathsf { M } } ^ { ( t ) } ( \pmb { \theta } ) \big ) \leq t \mathrm { T r } \big ( \mathcal { T } _ { \mathsf { Q } } ( \pmb { \theta } ) \big ) .\tag{23}
$$

## 3.5 Adaptive measurement protocols

Fix a positive integer t. An adaptive protocol using joint measurements on at most t samples receives independent samples of an unknown state. In round $i ,$ it chooses an integer $t _ { i } \in \{ 1 , \ldots , t \}$ and a joint POVM on $t _ { i }$ samples as measurable functions of a private random seed and all earlier classical outcomes. It applies this POVM to $t _ { i }$ fresh samples and retains only the classical outcome. The samples measured in that round are destroyed, and no quantum memory persists between rounds. The measurable outcome space $( Z _ { i } , \mathcal { Z } _ { i } )$ in each round is assumed to be countably generated: there is a countable collection of measurable events that generates $\mathcal { Z } _ { i }$ . This includes finite and countable outcome spaces, as well as Euclidean spaces with their Borel sigma-algebras. A protocol may halt early; its sample complexity is the maximum, over every seed and every transcript, of the total number of samples used. Equivalently, after halting we may pad the protocol with zero-sample, one-outcome rounds. The final estimate is a measurable function of the private seed and the observed outcomes.

The error in Theorems 1.1 and 1.2 is measured in the Schatten 1 norm. If trace distance is defined with the conventional factor $1 / 2$ , only the universal constants change.

The restriction to classical memory between rounds is part of the theorem: with persistent quantum memory, successive rounds could combine into a larger coherent measurement and the number of fresh samples measured in each round would no longer describe the experiment analyzed here.

## 4 From Fisher information bounds to the adaptive lower bound

This section proves Theorem 1.1. We first construct a local parameterization of a hard family of states. We next bound the Fisher information obtainable by jointly measuring t samples at a time, beginning with the simpler single-sample case. We then accumulate these bounds from each round along an adaptive transcript, convert the resulting Fisher information bound into a lower bound on expected trace norm loss using the van Trees inequality, and finish with a depolarizing channel reduction for the large-rank regime.

## 4.1 The hard family and its local coordinates

The hard states all have rank r and are maximally mixed on their supports. We keep the eigenvalues fixed and vary only the support. We parameterize these support rotations by a matrix X, with $X = 0$ corresponding to the reference support. The probability density introduced later is supported on matrices $X \in \mathbb { C } ^ { ( \hat { d } - r ) \times r }$ with small operator norm, so only nearby supports enter the lower bound argument.

Let $d \geq 2$ and $1 \leq r <$ d be integers, and set $m : = d - r$ . Fix an orthogonal decomposition

$$
\mathbb { C } ^ { d } = S \oplus S ^ { \perp } , \qquad \mathrm { d i m } S = r , \qquad \mathrm { d i m } S ^ { \perp } = m .
$$

Choose orthonormal bases of $S$ and $S ^ { \perp }$ , and use them to write vectors in these subspaces as elements of $\mathbb { C } ^ { r }$ and $\mathbb { C } ^ { m }$ , respectively. Let $\Pi _ { 0 }$ be the orthogonal projector onto S. With respect to the resulting orthonormal basis of $\mathbb { C } ^ { d }$ , our reference state has the block representation

$$
\rho _ { 0 } : = \frac { 1 } { r } \Pi _ { 0 } = \frac { 1 } { r } \left( \begin{array} { c c } { { \mathrm { I } _ { r } } } & { { 0 } } \\ { { 0 } } & { { 0 _ { m } } } \end{array} \right) .\tag{24}
$$

We use this family directly when $r \leq d / 2 \AA$ ; the case $r > d / 2$ is reduced to this regime in Section 4.5.3. For $\ b X \in \mathbb { C } ^ { m \times r }$ , define a linear map $V _ { X } : \mathbb { C } ^ { r } \to \mathbb { C } ^ { d }$ by

$$
V _ { X } : = \binom { \mathrm { I } _ { r } } { X } ( \mathrm { I } _ { r } + X ^ { \dagger } X ) ^ { - 1 / 2 } .\tag{25}
$$

The normalization on the right makes $V _ { X }$ an isometry. Consequently,

$$
\Pi _ { X } : = V _ { X } V _ { X } ^ { \dagger } , \qquad \rho _ { X } : = \frac { 1 } { r } \Pi _ { X }\tag{26}
$$

are respectively an orthogonal projector and a density matrix of rank r. The image of $\Pi _ { X }$ is the r-dimensional subspace

$$
\operatorname { I m } ( \Pi _ { X } ) = \{ ( \pmb { \mathscr { u } } , X \pmb { \mathscr { u } } ) : \pmb { \mathscr { u } } \in \mathbb { C } ^ { r } \} .
$$

At $X = 0$ , this subspace is S. Matrices X with small operator norm parameterize the r-dimensional subspaces near $S ,$ obtained by tilting vectors in $S$ toward $S ^ { \perp }$ . The columns of $V _ { X }$ form an orthonormal basis of the corresponding support.

We next compute the tangent to this family at $X = 0$ . Substituting Equation (25) into the definition $\Pi _ { X } = V _ { X } V _ { X } ^ { \dagger }$ in Equation (26) gives

$$
\Pi _ { X } = \binom { \operatorname { I } _ { r } } { X } \left( \operatorname { I } _ { r } + X ^ { \dagger } X \right) ^ { - 1 } \left( \operatorname { I } _ { r } \quad X ^ { \dagger } \right) .
$$

For a direction $H \in \mathbb { C } ^ { m \times r }$ , substitute $X = s H$ . The derivative at $s = 0$ of $( \mathrm { I } _ { r } + s ^ { 2 } H ^ { \dagger } H ) ^ { - 1 }$ is zero, and hence

$$
{ \mit \mathrm { D } } _ { H } { \mit \Pi } _ { 0 } : = \left. \frac { \mathrm { d } } { \mathrm { d } s } { \mit \Pi } _ { s H } \right| _ { s = 0 } = \left( { \bf 0 } _ { H } \right) \left( { \mathrm { I } } _ { r } \quad 0 \right) + \left( { \mathrm { I } } _ { r } \right) \left( 0 \quad H ^ { \dagger } \right) = \left( { \bf 0 } _ { H } \quad H ^ { \dagger } \right) .
$$

Since $\rho _ { X } = \Pi _ { X } / r$ , the corresponding state derivative is

$$
\mathrm { D } _ { H } \rho _ { 0 } : = \left. { \frac { \mathrm { d } } { \mathrm { d } s } } \rho _ { s H } \right. _ { s = 0 } = { \frac { 1 } { r } } \mathrm { D } _ { H } \Pi _ { 0 } = { \frac { 1 } { r } } \left( { \begin{array} { c c } { 0 } & { H ^ { \dagger } } \\ { H } & { 0 } \end{array} } \right) .\tag{27}
$$

The real and imaginary parts of the mr entries of X therefore give 2mr real tangent coordinates at the reference state.

The block representation also gives the metric comparison used when we later recover the parameter from a tomographic estimate. The lower left block of $\Pi _ { X }$ is $X ( \operatorname { I } _ { r } + X ^ { \dagger } X ) ^ { - 1 }$ . For X near zero, this block changes at essentially the same rate as X. The following lemma makes this precise: if $\Pi _ { X }$ and $\Pi _ { Y }$ are close in trace norm, then so are X and $Y$

Lemma 4.1 (Trace norm inverse Lipschitz bound). $I f \left\| X \right\| _ { \mathrm { o p } } , \left\| Y \right\| _ { \mathrm { o p } } \leq 1 / 4$ , then

$$
\left. X - Y \right. _ { 1 } \leq 2 \left. \Pi _ { X } - \Pi _ { Y } \right. _ { 1 } = 2 r \left. \rho _ { X } - \rho _ { Y } \right. _ { 1 } .\tag{28}
$$

Proof. Put $R _ { X } : = ( \mathrm { I } _ { r } + X ^ { \dagger } X ) ^ { - 1 }$ . The lower left block of $\Pi _ { X }$ $X R _ { X }$ , and

$$
X R _ { X } - Y R _ { Y } = ( X - Y ) R _ { X } + Y ( R _ { X } - R _ { Y } ) .
$$

Since $\| X \| _ { \mathrm { o p } } \leq 1 / 4$

$$
\mathrm { I } _ { r } \preceq \mathrm { I } _ { r } + X ^ { \dagger } X \preceq { \frac { 1 7 } { 1 6 } } \mathrm { I } _ { r } , \qquad { \frac { 1 6 } { 1 7 } } \mathrm { I } _ { r } \preceq R _ { X } \preceq \mathrm { I } _ { r } .
$$

Therefore,

$$
\| ( X - Y ) R _ { X } \| _ { 1 } \geq { \frac { 1 6 } { 1 7 } } \| X - Y \| _ { 1 } .
$$

Moreover,

$$
R _ { X } - R _ { Y } = R _ { X } ( R _ { Y } ^ { - 1 } - R _ { X } ^ { - 1 } ) R _ { Y } = R _ { X } ( Y ^ { \dagger } Y - X ^ { \dagger } X ) R _ { Y } ,
$$

and $\| R _ { X } \| _ { \mathrm { o p } } , \| R _ { Y } \| _ { \mathrm { o p } } \leq 1$ . Using the standard trace norm inequality $\left\| A B C \right\| _ { 1 } \leq \left\| A \right\| _ { \mathrm { o p } } \left\| B \right\| _ { 1 } \left\| C \right\| _ { \mathrm { o p } } ,$ we obtain

$$
\left\| Y ( R _ { X } - R _ { Y } ) \right\| _ { 1 } \leq \left\| Y \right\| _ { \mathrm { o p } } \left\| Y ^ { \dagger } Y - X ^ { \dagger } X \right\| _ { 1 } .
$$

Since

$$
Y ^ { \dagger } Y - X ^ { \dagger } X = Y ^ { \dagger } ( Y - X ) + ( Y - X ) ^ { \dagger } X ,
$$

applying the matrix norm inequality again gives

$$
\| Y ( { R _ { X } } - { R _ { Y } } ) \| _ { 1 } \leq \| Y \| _ { \mathrm { o p } } \left( \| X \| _ { \mathrm { o p } } + \| Y \| _ { \mathrm { o p } } \right) \| X - Y \| _ { 1 } \leq \frac { 1 } { 8 } \| X - Y \| _ { 1 } .
$$

The reverse triangle inequality now gives

$$
\begin{array} { l } { \displaystyle { \left\| X R _ { X } - Y R _ { Y } \right\| _ { 1 } \ge \left\| \big ( X - Y \big ) R _ { X } \right\| _ { 1 } - \left\| Y \big ( R _ { X } - R _ { Y } \big ) \right\| _ { 1 } } } \\ { \displaystyle \ge \left( \frac { 1 6 } { 1 7 } - \frac { 1 } { 8 } \right) \| X - Y \| _ { 1 } \ge \frac { 1 } { 2 } \left\| X - Y \right\| _ { 1 } . } \end{array}
$$

Taking a block compression cannot increase the trace norm, which proves the first inequality in Equation (28); the equality follows from $\rho _ { X } = \Pi _ { X } / r$ □

## 4.2 Fisher information trace for joint measurements

Recall that $m = d - r$ . The real and imaginary parts of $\ b X \in \mathbb { C } ^ { m \times r }$ give the 2mr real coordinates introduced in Section 4.1. For a joint POVM M on t samples, let $\mathcal { I } _ { \mathsf { M } } ^ { ( t ) } ( X )$ denote the classical Fisher information matrix of its outcome distribution in these coordinates. The following proposition bounds its trace for every measurement and every parameter X. This is the main technical input to the adaptive lower bound.

Proposition 4.2 (Fisher information trace for a joint measurement on t samples). There is a universal constant $C > 0$ such that, for every positive integer $t ,$ every $\ b X \in \mathbb { C } ^ { m \times r }$ , and every joint POVM M on t samples,

$$
\mathrm { T r } \big ( \mathcal { T } _ { \mathsf { M } } ^ { ( t ) } ( X ) \big ) \leq C d t \operatorname* { m i n } \left\{ 1 , \frac { \sqrt { t } } { r } \right\} .\tag{29}
$$

We first prove the bound at the reference point $X = 0 .$ , beginning with the single-sample case and then treating joint measurements on $t > 1$ samples. We finally transfer the resulting bound to every parameter X.

## 4.2.1 Single-sample measurements

Let M be an arbitrary POVM on $\mathbb { C } ^ { d }$ with measurable outcome space $( \boldsymbol { \ Z } , \boldsymbol { \mathcal { Z } } )$ . Let $\nu$ and $M _ { z }$ be its parameter-independent dominating measure and operator density from Equations (15) and (16). Relative to $S \oplus S ^ { \perp }$ , write

$$
M _ { z } = \left( \begin{array} { c c } { { A _ { z } } } & { { B _ { z } ^ { \dagger } } } \\ { { B _ { z } } } & { { C _ { z } } } \end{array} \right) \succeq 0 ,\tag{30}
$$

where $A _ { z } : S  S , B _ { z } : S  S ^ { \perp }$ , and $C _ { z } : S ^ { \perp } \to S ^ { \perp }$ . Taking the three blocks of $\begin{array} { r } { \int M _ { z } \ \mathrm { d } \nu ( z ) = \mathrm { I } _ { d } } \end{array}$ gives

$$
\int _ { \mathsf { Z } } A _ { z } \ \mathrm { d } \nu ( z ) = \mathrm { I } _ { r } , \qquad \int _ { \mathsf { Z } } B _ { z } \ \mathrm { d } \nu ( z ) = 0 , \qquad \int _ { \mathsf { Z } } C _ { z } \ \mathrm { d } \nu ( z ) = \mathrm { I } _ { m } .\tag{31}
$$

For each X, denote the likelihood density of the outcome of M on $\rho _ { X }$ by

$$
q _ { X } ( z ) : = \operatorname { T r } ( M _ { z } \rho _ { X } ) .
$$

In other words, the probability of an outcome in a measurable event $B \in { \mathcal { Z } }$ is $\begin{array} { r } { \int _ { B } q _ { X } ( z ) \ \mathrm { d } \nu ( z ) } \end{array}$ . At the reference point $X = 0$ , Equations (24) and (30) give

$$
{ q _ { 0 } } ( z ) = { \frac { \mathrm { T r } ( A _ { z } ) } { r } } .\tag{32}
$$

For $H \in \mathbb { C } ^ { m \times r }$ , Equations (18) and (27) give

$$
\mathrm { D } _ { H } q _ { 0 } ( z ) = \mathrm { T r } \big ( M _ { z } \mathrm { D } _ { H } \rho _ { 0 } \big ) = \frac { 1 } { r } \left( \mathrm { T r } \big ( B _ { z } ^ { \dagger } H \big ) + \mathrm { T r } \big ( B _ { z } H ^ { \dagger } \big ) \right) = \frac { 2 } { r } \mathrm { R e } \left( \mathrm { T r } \big ( B _ { z } ^ { \dagger } H \big ) \right) .\tag{33}
$$

We now calculate the numerator $\| \nabla q _ { 0 } ( z ) \| _ { 2 } ^ { 2 }$ in the Fisher information trace formula Equation (14). Here $\nabla q _ { 0 } ( z ) \in \mathbb { R } ^ { 2 m r }$ denotes the gradient of $X \mapsto q _ { X } ( z )$ at $X = 0$ in the coordinates associated with the basis in Equation (9). For $1 \leq i \leq m$ and $1 \leq j \leq r$ , we have $\mathrm { T r } ( B _ { z } ^ { \dagger } E _ { i j } ) = \overline { { ( B _ { z } ) _ { i j } } }$ . Substituting Equation (33) in each coordinate gives

$$
\begin{array} { l } { \displaystyle | | \nabla q _ { 0 } ( z ) | | _ { 2 } ^ { 2 } = \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { r } \left( ( \mathsf { D } _ { E _ { i j } } q _ { 0 } ( z ) ) ^ { 2 } + ( \mathsf { D } _ { \mathrm { i } E _ { i j } } q _ { 0 } ( z ) ) ^ { 2 } \right) } \\ { \displaystyle = \frac { 4 } { r ^ { 2 } } \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { r } \left( \mathrm { R e } \left( \overline { { ( B _ { z } ) _ { i j } } } \right) ^ { 2 } + \mathrm { R e } \left( \mathrm { i } \overline { { ( B _ { z } ) _ { i j } } } \right) ^ { 2 } \right) } \\ { \displaystyle = \frac { 4 } { r ^ { 2 } } \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { r } \left. ( B _ { z } ) _ { i j } \right. ^ { 2 } = \frac { 4 } { r ^ { 2 } } \left. B _ { z } \right. _ { \mathrm { F } } ^ { 2 } . } \end{array}\tag{34}
$$

Let $\mathcal { I } _ { \mathsf { M } } ^ { ( 1 ) } ( 0 )$ denote the measurement Fisher information matrix of the family $X \mapsto q _ { X }$ at $X = 0$ in the real coordinates $\{ E _ { i j } , \mathrm { i } E _ { i j } \} _ { i , j }$ . The superscript (1) indicates that M is applied to one sample of the state. Substituting Equations (32) and (34) into the Fisher information trace formula Equation (14) yields the exact identity

$$
\mathrm { T r } ( \mathcal { T } _ { \mathsf { M } } ^ { ( 1 ) } ( 0 ) ) = \frac { 4 } { r } \int _ { \mathsf { Z } } \frac { \| B _ { z } \| _ { \mathrm { F } } ^ { 2 } } { \mathrm { T r } ( A _ { z } ) } \ \mathrm { d } \nu ( z ) .\tag{35}
$$

$\mathrm { I f } \ \mathrm { T r } ( A _ { z } ) = 0$ , then $A _ { z } = 0$ , and positivity of $M _ { z }$ forces $B _ { z } = 0$ . Hence the likelihood and its derivatives vanish, and the quotient above is defined to be zero.

We next bound this quotient using the positivity of $M _ { z }$ . Applying the Cauchy–Schwarz inequality for the positive semidefinite form induced by $M _ { z }$ to the j-th coordinate vector of S and the i-th coordinate vector of $S ^ { \perp }$ gives

$$
| ( B _ { z } ) _ { i j } | ^ { 2 } \leq ( A _ { z } ) _ { j j } ( C _ { z } ) _ { i i } , \qquad i \in [ m ] , \ j \in [ r ] .
$$

Summing over these coordinates yields

$$
\Vert B _ { z } \Vert _ { \mathrm { F } } ^ { 2 } = \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { r } | ( B _ { z } ) _ { i j } | ^ { 2 } \leq \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { r } ( A _ { z } ) _ { j j } ( C _ { z } ) _ { i i } = \operatorname { T r } ( A _ { z } ) \operatorname { T r } ( C _ { z } ) .
$$

When $\mathrm { T r } ( A _ { z } ) > 0$ , it gives

$$
\frac { \| B _ { z } \| _ { \mathrm { F } } ^ { 2 } } { \mathrm { T r } ( A _ { z } ) } \leq \mathrm { T r } ( C _ { z } ) .
$$

Substituting this inequality into Equation (35) and using Equation (31), we conclude that

$$
\begin{array} { l } { \displaystyle \operatorname { T r } ( \mathcal { Z } _ { \mathsf { M } } ^ { ( 1 ) } ( 0 ) ) \leq \frac { 4 } { r } \int _ { \mathbb { Z } } \operatorname { T r } ( C _ { z } ) \ \mathrm { d } \nu ( z ) } \\ { \displaystyle = \frac { 4 } { r } \operatorname { T r } \left( \int _ { \mathbb { Z } } C _ { z } \ \mathrm { d } \nu ( z ) \right) } \\ { \displaystyle = \frac { 4 } { r } \operatorname { T r } ( \mathrm { I } _ { m } ) \leq \frac { 4 d } { r } . } \end{array}\tag{36}
$$

## 4.2.2 Joint measurements on multiple samples

For $t > 1$ , diferentiating $\rho _ { X } ^ { \otimes }$ produces a sum of t terms, one for each tensor factor. The Fisher information formula squares this sum, so we must control the cross terms between diferent positions. We reduce these cross terms to an operator that counts how many of the remaining $t - 1$ factors lie in the direction of a unit vector $h \in S$ , and then construct an upper bound independent of h with small trace. This gives the $O ( d t \sqrt { t } / r )$ bound when $t \leq r ^ { 2 }$ . When $t > r ^ { 2 }$ , the additive quantum Fisher information bound gives the $O ( d t )$ estimate.

Fix an integer $t \geq 2$ . Let $e _ { 1 } , \ldots , e _ { r }$ and $\boldsymbol { f } _ { 1 } , \ldots , \boldsymbol { f } _ { m }$ denote the orthonormal basis vectors of S and $S ^ { \perp }$ , respectively. We use the same notation $E _ { i j }$ for the operator $| f _ { i } \rangle \langle e _ { j } |$ on $\mathbb { C } ^ { d }$ , which maps $S$ to $S ^ { \perp }$ and vanishes on $S ^ { \perp }$ . Recall from Section 4.1 that $\Pi _ { 0 }$ is the projector onto S. The zero-excitation subspace and the reference state on t samples are

$$
\mathcal { H } _ { 0 } : = S ^ { \otimes t } , \qquad \rho _ { 0 } ^ { \otimes t } = \frac { 1 } { r ^ { t } } \Pi _ { 0 } ^ { \otimes t } .
$$

For each $1 \leq i \leq m$ , define the corresponding one-excitation subspace

$$
\mathcal { H } _ { 1 , i } : = \bigoplus _ { \ell = 1 } ^ { t } S ^ { \otimes ( \ell - 1 ) } \otimes \mathrm { s p a n } \{ \pmb { f } _ { i } \} \otimes S ^ { \otimes ( t - \ell ) } ,
$$

and let $\Pi _ { 1 , i }$ be the orthogonal projector onto $\mathcal { H } _ { 1 , i }$ . The summand indexed by ℓ consists of the tensors whose ℓ-th factor lies in span $\{ f _ { i } \}$ and whose other factors lie in S.

For $1 \leq i \leq m$ and $1 \le j \le r$ , define the collective raising operator

$$
R _ { i , j } : = \sum _ { \ell = 1 } ^ { t } \Pi _ { 0 } ^ { \otimes ( \ell - 1 ) } \otimes E _ { i j } \otimes \Pi _ { 0 } ^ { \otimes ( t - \ell ) } .
$$

This operator maps $\mathcal { H } _ { 0 }$ into $\mathcal { H } _ { 1 , i }$ and vanishes on $\mathcal { H } _ { 0 } ^ { \perp }$ . Its adjoint $R _ { i , j } ^ { \dagger }$ vanishes on $\mathcal { H } _ { 1 , i } ^ { \bot }$ . For a direction $H \in \mathbb { C } ^ { m \times r }$ , the product rule gives

$$
\left. { \frac { \mathrm { d } } { \mathrm { d } \lambda } } \rho _ { \lambda H } ^ { \otimes t } \right| _ { \lambda = 0 } = \sum _ { \ell = 1 } ^ { t } \rho _ { 0 } ^ { \otimes ( \ell - 1 ) } \otimes \mathrm { D } _ { H } \rho _ { 0 } \otimes \rho _ { 0 } ^ { \otimes ( t - \ell ) } .
$$

Using $\rho _ { 0 } = \Pi _ { 0 } / r$ and Equation (27), we obtain

$$
\begin{array} { r } { \left. \frac { \mathrm { d } } { \mathrm { d } \lambda } \rho _ { \lambda E _ { i j } } ^ { \otimes t } \right| _ { \lambda = 0 } = \frac { 1 } { r ^ { t } } ( R _ { i , j } + R _ { i , j } ^ { \dag } ) , } \\ { \left. \frac { \mathrm { d } } { \mathrm { d } \lambda } \rho _ { \lambda \mathrm { i } E _ { i j } } ^ { \otimes t } \right| _ { \lambda = 0 } = \frac { \mathrm { i } } { r ^ { t } } ( R _ { i , j } - R _ { i , j } ^ { \dag } ) . } \end{array}\tag{37}
$$

Thus the tangent directions above connect the zero-excitation subspace only to the one-excitation subspaces.

Let M be an arbitrary POVM on $( \mathbb { C } ^ { d } ) ^ { \otimes t }$ with outcome space $( Z , { \mathcal { Z } } )$ . Let ν and $M _ { z }$ be its parameter-independent dominating measure and operator density from Equations (15) and (16). For $1 \leq i \leq m$ , define the blocks

$$
A _ { z } : = \Pi _ { 0 } ^ { \otimes t } M _ { z } \Pi _ { 0 } ^ { \otimes t } , \qquad B _ { z , i } : = \Pi _ { 1 , i } M _ { z } \Pi _ { 0 } ^ { \otimes t } , \qquad C _ { z , i } : = \Pi _ { 1 , i } M _ { z } \Pi _ { 1 , i } .\tag{38}
$$

Thus $A _ { z }$ acts on $\mathcal { H } _ { 0 } , ~ B _ { z , i }$ maps $\mathcal { H } _ { 0 }$ to $\mathcal { H } _ { 1 , i }$ , and $C _ { z , i }$ acts on $\mathcal { H } _ { 1 , i }$ . Multiplying the identity $\textstyle \int _ { Z } M _ { z } \ \mathrm { d } \nu ( z ) = \operatorname { I } _ { d } ^ { \otimes t }$ on the left and right by the corresponding projectors gives

$$
\int _ { \mathbb { Z } } A _ { z } { \mathrm { ~ } } \mathrm { d } \nu ( z ) = \Pi _ { 0 } ^ { \otimes t } , \qquad \int _ { \mathbb { Z } } B _ { z , i } { \mathrm { ~ } } \mathrm { d } \nu ( z ) = 0 , \qquad \int _ { \mathbb { Z } } C _ { z , i } { \mathrm { ~ } } \mathrm { d } \nu ( z ) = \Pi _ { 1 , i } .\tag{39}
$$

For each X, define the likelihood density

$$
q _ { X } ^ { ( t ) } ( z ) : = \mathrm { T r } \big ( M _ { z } \rho _ { X } ^ { \otimes t } \big ) .
$$

At the reference point, $\rho _ { 0 } ^ { \otimes t } = \Pi _ { 0 } ^ { \otimes t } / r ^ { t }$ , so

$$
{ q _ { 0 } ^ { ( t ) } ( z ) = { \frac { \mathrm { T r } ( A _ { z } ) } { r ^ { t } } } . }\tag{40}
$$

Substituting Equations (18) and (37) in each coordinate gives

$$
\begin{array} { l } { \displaystyle \left\| \nabla q _ { 0 } ^ { ( t ) } ( z ) \right\| _ { 2 } ^ { 2 } = \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { r } \left( ( \mathrm { D } _ { E _ { i j } } q _ { 0 } ^ { ( t ) } ( z ) ) ^ { 2 } + ( \mathrm { D } _ { \mathrm { i } E _ { i j } } q _ { 0 } ^ { ( t ) } ( z ) ) ^ { 2 } \right) } \\ { \displaystyle \qquad = \frac { 4 } { r ^ { 2 t } } \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { r } \left( \mathrm { R e } \left( \mathrm { T r } ( B _ { z , i } ^ { \dagger } R _ { i , j } ) \right) ^ { 2 } + \mathrm { R e } \left( \mathrm { i } \mathrm { T r } ( B _ { z , i } ^ { \dagger } R _ { i , j } ) \right) ^ { 2 } \right) } \\ { \displaystyle \qquad = \frac { 4 } { r ^ { 2 t } } \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { r } \left| \mathrm { T r } ( B _ { z , i } ^ { \dagger } R _ { i , j } ) \right| ^ { 2 } . } \end{array}\tag{41}
$$

Let $\mathcal { I } _ { \mathsf { M } } ^ { ( t ) } ( 0 )$ denote the measurement Fisher information matrix induced by M at $X = 0$ in these coordinates. Substituting Equations (40) and (41) into Equation (14) yields the exact identity

$$
\mathrm { T r } ( \mathcal { T } _ { \mathsf { M } } ^ { ( t ) } ( 0 ) ) = \frac { 4 } { r ^ { t } } \int _ { \mathsf { Z } } \frac { \sum _ { i = 1 } ^ { m } \sum _ { j = 1 } ^ { r } \left| \mathrm { T r } ( B _ { z , i } ^ { \dagger } R _ { i , j } ) \right| ^ { 2 } } { \mathrm { T r } ( A _ { z } ) } ~ \mathrm { d } \nu ( z ) .\tag{42}
$$

If $\mathrm { T r } ( A _ { z } ) = 0$ , then $A _ { z } = 0$ , and positivity of the restriction of $M _ { z }$ to $\mathcal { H } _ { 0 } \oplus \mathcal { H } _ { 1 , i }$ forces $B _ { z , i } = 0$ for every i. Hence the likelihood and its derivatives vanish, and the quotient above is defined to be zero.

For a unit vector $\begin{array} { r } { \pmb { h } = \sum _ { j = 1 } ^ { r } h _ { j } \pmb { e } _ { j } \in S } \end{array}$ , define

$$
R _ { i , h } : = \sum _ { j = 1 } ^ { r } \overline { { h _ { j } } } R _ { i , j } .
$$

For fixed z and $i ,$ the sum over the r coordinate directions can be written as the largest value over a unit direction $h \in S$

$$
\sum _ { j = 1 } ^ { r } \left| \mathrm { T r } ( B _ { z , i } ^ { \dagger } R _ { i , j } ) \right| ^ { 2 } = \operatorname* { s u p } _ { \| h \| = 1 } \left| \sum _ { j = 1 } ^ { r } \overline { { h _ { j } } } \mathrm { T r } ( B _ { z , i } ^ { \dagger } R _ { i , j } ) \right| ^ { 2 } = \operatorname* { s u p } _ { \stackrel { h \in S } { \| h \| = 1 } } \left| \mathrm { T r } ( B _ { z , i } ^ { \dagger } R _ { i , h } ) \right| ^ { 2 } .\tag{43}
$$

The first equality follows from Cauchy–Schwarz and its equality condition. The second uses the definition of $R _ { i , h }$

To bound this supremum, we factor $M _ { z }$ through its positive semidefinite square root and apply the Frobenius Cauchy–Schwarz inequality. For $1 \leq i \leq m$ and ν-almost every $z ,$ define

$$
Y _ { z , 0 } : = \sqrt { M _ { z } } \Pi _ { 0 } ^ { \otimes t } , \qquad Y _ { z , 1 , i } : = \sqrt { M _ { z } } \Pi _ { 1 , i } .
$$

The block definitions in Equation (38) give

$$
A _ { z } = Y _ { z , 0 } ^ { \dagger } Y _ { z , 0 } , \qquad B _ { z , i } = Y _ { z , 1 , i } ^ { \dagger } Y _ { z , 0 } , \qquad C _ { z , i } = Y _ { z , 1 , i } ^ { \dagger } Y _ { z , 1 , i } .
$$

For every unit vector $h \in S$ , Cauchy–Schwarz and cyclicity of the trace give

$$
\begin{array} { r l } & { \left| \mathrm { T r } ( B _ { z , i } ^ { \dagger } R _ { i , h } ) \right| ^ { 2 } = \left| \mathrm { T r } ( Y _ { z , 0 } ^ { \dagger } Y _ { z , 1 , i } R _ { i , h } ) \right| ^ { 2 } } \\ & { \qquad \leq \| Y _ { z , 0 } \| _ { \mathrm { F } } ^ { 2 } \left\| Y _ { z , 1 , i } R _ { i , h } \right\| _ { \mathrm { F } } ^ { 2 } } \\ & { \qquad = \mathrm { T r } ( A _ { z } ) \mathrm { T r } ( C _ { z , i } R _ { i , h } R _ { i , h } ^ { \dagger } ) . } \end{array}\tag{44}
$$

Since $C _ { z , i } \succeq 0$ , any positive semidefinite operator $Q$ on $\mathcal { H } _ { 1 , i }$ satisfying $R _ { i , h } R _ { i , h } ^ { \dagger } \preceq Q$ for every unit h bounds the right side by $\mathrm { T r } ( A _ { z } ) \mathrm { T r } ( C _ { z , i } Q )$ . It therefore sufices to construct such a directionindependent upper bound. To construct this upper bound, we decompose the one-excitation space according to the position of the excitation. This reduces the problem to an operator that counts how many of the remaining tensor factors lie in the direction h.

Reduction to occupation number operators. For $1 \leq \ell \leq t$ , use a hat to indicate the omitted tensor factor and let

$$
K _ { \widehat { \ell } } : = \bigotimes _ { q \in [ t ] \backslash \{ \ell \} } S _ { q } ,
$$

where $S _ { q }$ denotes the copy of S in the q-th tensor factor and the factors retain their original order. The map that inserts $\mathbf { \Delta } f _ { i }$ in position ℓ is an isometry from $\kappa _ { \widehat { \ell } }$ onto the ℓ-th summand of $\mathcal { H } _ { 1 , i } ,$ so

$$
\mathcal { H } _ { 1 , i } \cong \bigoplus _ { \ell = 1 } ^ { t } \mathcal { K } _ { \widehat { \ell } } .
$$

In the rest of the proof, we use these insertion isometries to write operators on $\mathcal { H } _ { 1 , i }$ as block operators on this direct sum.

For each unit vector $h \in S$ , let $\Pi _ { h } : = | h \rangle \langle h |$ be the orthogonal projector onto span $\{ h \}$ . For $\ell \in [ t ]$ and $q \in [ t ] \setminus \{ \ell \}$ , define the operator on $\kappa _ { \widehat { \ell } }$

$$
\Pi _ { h , q } ^ { ( \widehat { \ell } ) } : = \bigotimes _ { p \in [ t ] \backslash \{ \ell \} } \left\{ \Pi _ { h } , \begin{array} { r } { p = q , } \\ { \operatorname { I } _ { S _ { p } } , } \end{array} \right.
$$

For each $\ell \in [ t ]$ , define the occupation number operator on $\kappa _ { \widehat { \ell } } ,$ the tensor product of the remaining t − 1 factors, by

$$
N _ { h , \widehat { \ell } } ^ { ( t - 1 ) } : = \sum _ { q \in [ t ] \setminus \{ \ell \} } \Pi _ { h , q } ^ { ( \widehat { \ell } ) } .\tag{45}
$$

Extend h to an orthonormal basis $\pmb { u } _ { 1 } = \pmb { h } , \pmb { u } _ { 2 } , \dots , \pmb { u } _ { r }$ of S. The vectors

$$
\pmb { v } _ { j } : = \bigotimes _ { q \in [ t ] \backslash \{ \ell \} } \pmb { u } _ { j _ { q } } , \qquad j _ { q } \in [ r ] ,
$$

form an orthonormal basis of $\kappa _ { \widehat { \ell } } .$ Each projector $\Pi _ { h , q } ^ { ( \widehat { \ell } ) }$ leaves $v _ { j }$ unchanged when ${ j _ { q } } = 1$ and sends it to zero otherwise. Consequently,

$$
N _ { h , \widehat { \ell } } ^ { ( t - 1 ) } v _ { j } = | \{ q \in [ t ] \setminus \{ \ell \} : j _ { q } = 1 \} | v _ { j } .
$$

Thus $N _ { h , \widehat { \ell } } ^ { ( t - 1 ) }$ is diagonal in this basis. Its eigenvalue on $v _ { j }$ counts the tensor factors ${ \pmb u } _ { j _ { q } }$ equal to h.

Lemma 4.3 (Reduction to occupation number operators). For every $1 \leq i \leq m$ and every unit vector $h \in S$ 2

$$
R _ { i , h } R _ { i , h } ^ { \dagger } \preceq \bigoplus _ { \ell = 1 } ^ { t } \left( \operatorname { I } _ { \mathcal { K } _ { \widehat { \ell } } } + N _ { h , \widehat { \ell } } ^ { ( t - 1 ) } \right) .\tag{46}
$$

Proof. Fix i and h as in the statement. Using the preceding direct sum representation, write $\pmb { \xi } = ( \pmb { \xi } _ { 1 } , \dots , \pmb { \xi } _ { t } ) \in \mathcal { H } _ { 1 , i } .$ , where $\pmb { \xi } _ { \ell } \in \mathcal { K } _ { \widehat { \ell } } .$ Thus $\xi _ { \ell }$ represents the component obtained by inserting $\mathbf { \Delta } f _ { i }$ in position ℓ. Let $J _ { \ell , h } : K _ { \widehat { \ell } } \to \mathcal { H } _ { 0 }$ be the isometry that instead inserts h in position ℓ. The q-th term of $R _ { i , h } ^ { \dagger }$ applies $| h \rangle \langle f _ { i } |$ to tensor factor q. On the vector obtained from $\xi _ { \ell }$ by inserting $\mathbf { \Delta } f _ { i }$ in position $\ell ,$ every tensor factor other than ℓ lies in S. Thus, if $q \neq \ell .$ , this term vanishes because $\mathbf { \Delta } f _ { i } \perp S$ . The term with $q = \ell$ instead replaces $\mathbf { \Delta } f _ { i }$ by h. Therefore,

$$
R _ { i , h } ^ { \dagger } \pmb { \xi } = \sum _ { \ell = 1 } ^ { t } J _ { \ell , h } \pmb { \xi } _ { \ell } .
$$

On $\mathcal { H } _ { 0 }$ , the operator $J _ { \ell , h } J _ { \ell , h } ^ { \dagger } = \Pi _ { h } ^ { [ \ell ] }$ is the orthogonal projector onto the range of $J _ { \ell , h }$ . For distinct $\ell , q \in [ t ]$ , the projectors $\Pi _ { h } ^ { [ \ell ] }$ and $\Pi _ { h } ^ { [ q ] }$ commute, so

$$
\langle J _ { \ell , h } \xi _ { \ell } , J _ { q , h } \xi _ { q } \rangle = \langle \Pi _ { h } ^ { [ q ] } J _ { \ell , h } \xi _ { \ell } , \Pi _ { h } ^ { [ \ell ] } J _ { q , h } \xi _ { q } \rangle = \langle J _ { \ell , h } \Pi _ { h , q } ^ { ( \widehat { \ell } ) } \xi _ { \ell } , J _ { q , h } \Pi _ { h , \ell } ^ { ( \widehat { q } ) } \xi _ { q } \rangle .
$$

Since the insertion maps are isometries, Cauchy–Schwarz and $2 x y \le x ^ { 2 } + y ^ { 2 }$ give

$$
\begin{array} { r } { 2 \operatorname { R e } \langle J _ { \ell , h } \xi _ { \ell } , J _ { q , h } \xi _ { q } \rangle \leq 2 \left\| \Pi _ { h , q } ^ { ( \widehat { \ell } ) } \xi _ { \ell } \right\| \left\| \Pi _ { h , \ell } ^ { ( \widehat { q } ) } \xi _ { q } \right\| \leq \langle \xi _ { \ell } , \Pi _ { h , q } ^ { ( \widehat { \ell } ) } \xi _ { \ell } \rangle + \langle \xi _ { q } , \Pi _ { h , \ell } ^ { ( \widehat { q } ) } \xi _ { q } \rangle . } \end{array}
$$

We now bound the quadratic form of $R _ { i , h } R _ { i , h } ^ { \dagger }$ on $\xi { : }$

$$
\begin{array} { r l } { \langle \pmb { \xi } , R _ { i , h } R _ { i , h } ^ { \dagger } \pmb { \xi } \rangle = \left\| R _ { i , h } ^ { \dagger } \pmb { \xi } \right\| ^ { 2 } } & { = \displaystyle \sum _ { \ell = 1 } ^ { t } \| \pmb { \xi } _ { \ell } \| ^ { 2 } + 2 \sum _ { \ell < q } \mathrm { R e } \langle J _ { \ell , h } \pmb { \xi } _ { \ell } , J _ { q , h } \pmb { \xi } _ { q } \rangle } \\ & { \leq \displaystyle \sum _ { \ell = 1 } ^ { t } \left( \| \pmb { \xi } _ { \ell } \| ^ { 2 } + \sum _ { q \neq \ell } \langle \pmb { \xi } _ { \ell } , \Pi _ { h , q } ^ { ( \widehat { \ell } ) } \pmb { \xi } _ { \ell } \rangle \right) } \\ & { = \displaystyle \sum _ { \ell = 1 } ^ { t } \langle \pmb { \xi } _ { \ell } , \big ( \mathrm { I } _ { \mathcal { K } _ { \widehat { \ell } } } + N _ { h , \widehat { \ell } } ^ { ( { t - 1 } ) } \big ) \pmb { \xi } _ { \ell } \rangle . } \end{array}
$$

Here the inequality follows by summing the cross term bound over all pairs $\ell < q$ and regrouping the terms by excitation position. For each fixed $\ell ,$ every $q \neq \ell$ appears once. Since this holds for every $\xi \in \mathcal { H } _ { 1 , i }$ , it proves Equation (46). □

A uniform bound on occupation number operators. Each operator in Equation (45) has the same form on a tensor product of $t - 1$ copies of S. We state the required direction-independent bound for a general number n of factors. For every positive integer n and every unit vector $h \in S$ define

$$
N _ { h } ^ { ( n ) } : = \sum _ { \ell = 1 } ^ { n } \Pi _ { h } ^ { [ \ell ] } = \sum _ { \ell = 1 } ^ { n } \Gamma _ { S } ^ { \otimes ( \ell - 1 ) } \otimes \Pi _ { h } \otimes \Gamma _ { S } ^ { \otimes ( n - \ell ) } .
$$

We now construct a positive semidefinite operator $T _ { n , r } ,$ independent of $^ { h , }$ such that $N _ { h } ^ { ( n ) } \preceq T _ { n , \prime }$ for every unit $h \in S$ , while its normalized trace remains small.

Lemma 4.4 (Uniform bound on occupation number operators). There is a universal constant $C > 0$ such that, for all positive integers $n , r$ and every r-dimensional complex Hilbert space $S ,$ there exists a positive semidefinite operator $T _ { n , r }$ on $S ^ { \otimes n }$ such that

$$
N _ { h } ^ { ( n ) } \preceq T _ { n , r } \qquad f o r \ e v e r y \ u n i t \ v e c t o r \ h \in S ,\tag{47}
$$

and

$$
{ \frac { \operatorname { T r } ( T _ { n , r } ) } { r ^ { n } } } \leq C \left( { \sqrt { n } } + { \frac { n } { r } } \right) .\tag{48}
$$

Proof. We obtain an upper bound independent of h by bounding a suitable polynomial in $N _ { h } ^ { ( n ) }$ above by a sum of projectors onto symmetric subspaces. Fix an integer $1 \leq k \leq n$ . For a subset $J \subseteq [ n ]$ with $| J | = k$ , let $\mathfrak { S } _ { J }$ be the set of permutations of J. For each $\pi \in { \mathfrak { S } } _ { J }$ , let $U _ { \pi }$ permute the corresponding tensor factors while fixing the others. The orthogonal projector onto the subspace invariant under these permutations is

$$
\Pi _ { \mathrm { s y m } , J } : = \frac { 1 } { k ! } \sum _ { \pi \in \mathfrak { S } _ { J } } U _ { \pi } .
$$

Set

$$
\Omega _ { n , k } : = \sum _ { { J \subseteq [ n ] \atop | J | = k } } \Pi _ { \mathrm { s y m } , J } .
$$

The operator $\Omega _ { n , k }$ is positive semidefinite and does not depend on $\mathbf { \delta } _ { h } .$

Since the projectors $\Pi _ { h } ^ { [ \ell ] }$ commute,

$$
\binom { N _ { h } ^ { ( n ) } } { k } : = \frac { 1 } { k ! } \prod _ { s = 0 } ^ { k - 1 } ( N _ { h } ^ { ( n ) } - s \mathrm { I } _ { S } ^ { \otimes n } ) = \sum _ { J \subseteq [ n ] } \prod _ { \ell \in J } \Pi _ { h } ^ { [ \ell ] } .
$$

In particular, this binomial coeficient operator is positive semidefinite. For a fixed $J ,$ , the product on the right projects onto the tensors whose factors in J all lie in span $\{ h \}$ . Such tensors are invariant under every permutation of those factors, so

$$
\prod _ { \ell \in J } \Pi _ { h } ^ { [ \ell ] } \preceq \Pi _ { \mathrm { s y m } , J } .
$$

Summing this inequality over J gives

$$
\binom { N _ { h } ^ { ( n ) } } { k } \preceq \Omega _ { n , k } .
$$

We next recover a bound on $N _ { h } ^ { ( n ) }$ from the bound on $\binom { N _ { h } ^ { ( n ) } } { k }$ . Fix h and extend it to an orthonormal basis of $S .$ . Each vector $b _ { 1 } \otimes \cdots \otimes b _ { n }$ in the associated tensor product basis is an eigenvector of $N _ { h } ^ { ( n ) }$ , with eigenvalue equal to the number of indices $\ell \in [ n ]$ for which $b _ { \ell } = h$ . Thus every eigenvalue belongs to $\{ 0 , 1 , \ldots , n \}$ . For every such eigenvalue $x _ { i }$

$$
x \leq k + k { \binom { x } { k } } ^ { 1 / k } .
$$

When $x < k$ , the first term on the right already bounds $x ;$ when $x \geq k$ , the inequality follows from $\textstyle { \binom { x } { k } } \geq ( x / k ) ^ { k }$ . Applying it to every eigenvalue of $N _ { h } ^ { ( n ) }$ gives

$$
N _ { \pmb { h } } ^ { ( n ) } \preceq k \mathrm { I } _ { S } ^ { \otimes n } + k \left( \binom { N _ { \pmb { h } } ^ { ( n ) } } { k } \right) ^ { 1 / k } .
$$

For every $\pi \in { \mathfrak { S } } _ { n }$ , conjugation permutes the summands:

$$
U _ { \pi } N _ { h } ^ { ( n ) } U _ { \pi } ^ { \dagger } = \sum _ { \ell = 1 } ^ { n } \Pi _ { h } ^ { [ \pi ( \ell ) ] } = N _ { h } ^ { ( n ) } .
$$

Thus $N _ { h } ^ { ( n ) }$ commutes with every tensor permutation $U _ { \pi }$ and hence with $\Omega _ { n , k }$ . Taking positive semidefinite k-th roots in a common eigenbasis, the binomial coeficient operator inequality above gives

$$
N _ { h } ^ { \left( n \right) } \preceq k \mathrm { I } _ { S } ^ { \otimes n } + k \Omega _ { n , k } ^ { 1 / k } .\tag{49}
$$

Here $\Omega _ { n , k } ^ { 1 / k }$ denotes the positive semidefinite k-th root of $\Omega _ { n , k }$

It remains to choose k so that the operator on the right has small trace. Since the symmetric subspace of $S ^ { \otimes k }$ has dimension $\binom { r + k - 1 } { k }$ , each $\Pi _ { \mathrm { s y m } , J }$ has trace $\binom { r + k - 1 } { k } r ^ { n - k }$ . Therefore

$$
{ \frac { \operatorname { T r } ( \Omega _ { n , k } ) } { r ^ { n } } } = { \binom { n } { k } } { \frac { ( r + k - 1 ) } { r ^ { k } } } .
$$

Concavity of $x \mapsto x ^ { 1 / k }$ , Jensen’s inequality, and the standard binomial estimate give, for a universal constant $C ^ { \prime } > 0$

$$
\frac { \mathrm { T r } ( \Omega _ { n , k } ^ { 1 / k } ) } { r ^ { n } } \leq \left( \frac { \mathrm { T r } ( \Omega _ { n , k } ) } { r ^ { n } } \right) ^ { 1 / k } \leq \frac { \mathrm { e } n } { k } \frac { \mathrm { e } ( r + k - 1 ) } { k r } \leq C ^ { \prime } \left( \frac { n } { k ^ { 2 } } + \frac { n } { k r } \right) .
$$

Here the second inequality uses ${ \cal ( } _ { k } ^ { u } ) \le ( \mathrm { e } u / k ) ^ { k }$ for integers u $\geq k \geq 1$ . Choose $k : = \lceil { \sqrt { n } } \rceil$ and define

$$
\begin{array} { r } { T _ { n , r } : = k \mathrm { I } _ { S } ^ { \otimes n } + k \Omega _ { n , k } ^ { 1 / k } . } \end{array}
$$

The inequality in Equation (47) follows from Equation (49), while

$$
{ \frac { \operatorname { T r } ( T _ { n , r } ) } { r ^ { n } } } \leq k + C ^ { \prime } \left( { \frac { n } { k } } + { \frac { n } { r } } \right) \leq C \left( { \sqrt { n } } + { \frac { n } { r } } \right) .
$$

This proves Equation (48).

A uniform bound on the one-excitation subspace. We apply the occupation number bound to each summand $\kappa _ { \widehat { \ell } }$ in the direct sum representation of $\mathcal { H } _ { 1 , i }$ . This produces a direction-independent upper bound on $R _ { i , h } R _ { i , h } ^ { \dagger }$ for use in Equation (44).

Lemma 4.5 (Uniform bound on the one-excitation subspace). There is a universal constant $C > 0$ such that, for every $1 \leq i \leq m$ , there is a positive semidefinite operator ${ Q } _ { t , r } ^ { ( i ) }$ on $\mathcal { H } _ { 1 , i }$ such that

$$
R _ { i , h } R _ { i , h } ^ { \dagger } \preceq Q _ { t , r } ^ { ( i ) } f o r e v e r y u n i t v e c t o r h \in S ,\tag{50}
$$

and

$$
\mathrm { T r } ( Q _ { t , r } ^ { ( i ) } ) \leq C t r ^ { t - 1 } \left( \sqrt { t } + \frac { t } { r } \right) .\tag{51}
$$

Proof. Fix $1 \leq i \leq m$ . For each $\ell \in [ t ]$ , let $T _ { t - 1 , r } ^ { ( \widehat { \ell } ) }$ be the operator on $\kappa _ { \widehat { \ell } }$ obtained by placing the tensor factors of $T _ { t - 1 , r }$ from Lemma 4.4 in the positions $[ t ] \setminus \{ \ell \}$ , in their original order. Define

$$
Q _ { t , r } ^ { ( i ) } : = \bigoplus _ { \ell = 1 } ^ { t } \left( \operatorname { I } _ { \mathcal { K } _ { \ell } } + T _ { t - 1 , r } ^ { ( \widehat { \ell } ) } \right) .
$$

This operator does not depend on h. Under the same placement of tensor factors, $N _ { h } ^ { ( t - 1 ) }$ becomes $N _ { h , \widehat { \ell } } ^ { ( t - 1 ) }$ . Therefore, Lemma 4.4 gives

$$
N _ { h , \widehat { \ell } } ^ { ( t - 1 ) } \preceq T _ { t - 1 , r } ^ { ( \widehat { \ell } ) } \qquad \mathrm { f o r ~ e v e r y ~ } \ell \in [ t ] \mathrm { ~ a n d ~ e v e r y ~ u n i t ~ } h \in S .
$$

Combining these inequalities with Lemma 4.3 proves Equation (50). Finally, by Lemma 4.4,

$$
\mathrm { T r } ( Q _ { t , r } ^ { ( i ) } ) = t \left( r ^ { t - 1 } + \mathrm { T r } ( T _ { t - 1 , r } ) \right) \leq C t r ^ { t - 1 } \left( \sqrt { t } + \frac { t } { r } \right) ,
$$

which proves Equation (51).

Completion of the Fisher bound for joint measurements. For each $1 \leq i \leq m$ , we now use the direction-independent operator ${ Q } _ { t , r } ^ { ( i ) }$ to bound the numerator in Equation (42). Since $C _ { z , i } \succeq 0$ combining Equations (44) and (50) and taking the supremum over h as in Equation (43) gives

$$
\sum _ { j = 1 } ^ { r } \left| \mathrm { T r } ( B _ { z , i } ^ { \dagger } R _ { i , j } ) \right| ^ { 2 } \leq \mathrm { T r } ( A _ { z } ) \mathrm { T r } ( C _ { z , i } Q _ { t , r } ^ { ( i ) } ) .\tag{52}
$$

Substituting Equation (52) into Equation (42) and using the POVM normalization in Equation (39) and the trace bound Equation (51), we obtain

$$
\begin{array} { r l } & { \mathrm { T r } ( \mathcal { T } _ { \mathsf { M } } ^ { ( \mathsf { t } ) } ( 0 ) ) \leq \displaystyle \frac { 4 } { r ^ { t } } \displaystyle \sum _ { i = 1 } ^ { m } \int _ { \mathbb { Z } } \mathrm { T r } \big ( C _ { z , i } Q _ { t , r } ^ { ( i ) } \big ) \mathrm { d } \nu ( z ) } \\ & { \qquad = \displaystyle \frac { 4 } { r ^ { t } } \displaystyle \sum _ { i = 1 } ^ { m } \mathrm { T r } \left( \left( \int _ { \mathbb { Z } } C _ { z , i } \mathrm { d } \nu ( z ) \right) Q _ { t , r } ^ { ( i ) } \right) } \\ & { \qquad = \displaystyle \frac { 4 } { r ^ { t } } \displaystyle \sum _ { i = 1 } ^ { m } \mathrm { T r } ( Q _ { t , r } ^ { ( i ) } ) } \\ & { \qquad \leq \displaystyle \frac { 4 } { r ^ { t } } \displaystyle \sum _ { i = 1 } ^ { m } C t r ^ { t - 1 } \left( \sqrt { t } + \displaystyle \frac { t } { r } \right) } \\ & { \qquad \leq 4 C \displaystyle \frac { d t } { r } \left( \sqrt { t } + \displaystyle \frac { t } { r } \right) . } \end{array}\tag{53}
$$

For $t > r ^ { 2 }$ , we instead use the general quantum Fisher information bound from the preliminaries, which gives the required linear bound in this regime. Recall that a symmetric logarithmic derivative for a tangent direction H is a Hermitian operator $L _ { H }$ satisfying $\mathrm { D } _ { H } \rho _ { 0 } = ( \rho _ { 0 } L _ { H } + L _ { H } \rho _ { 0 } ) / 2$ . At $X = 0$ , take the symmetric logarithmic derivatives for the real and imaginary tangent directions $E _ { i j }$ and $\mathrm { i } E _ { i j }$ to be

$$
\begin{array} { r } { L _ { i , j } ^ { \mathrm { R } } : = 2 ( E _ { i j } + E _ { i j } ^ { \dagger } ) , \qquad L _ { i , j } ^ { \mathrm { I } } : = 2 \mathrm { i } ( E _ { i j } - E _ { i j } ^ { \dagger } ) . } \end{array}
$$

Indeed, substituting $\rho _ { 0 } = \Pi _ { 0 } / r$ gives

$$
\frac 1 2 \big ( \rho _ { 0 } L _ { i , j } ^ { \mathrm { R } } + L _ { i , j } ^ { \mathrm { R } } \rho _ { 0 } \big ) = \mathrm { D } _ { E _ { i j } } \rho _ { 0 } , \qquad \frac 1 2 \big ( \rho _ { 0 } L _ { i , j } ^ { \mathrm { I } } + L _ { i , j } ^ { \mathrm { I } } \rho _ { 0 } \big ) = \mathrm { D } _ { \mathrm { i } E _ { i j } } \rho _ { 0 } ,
$$

as required by Equation (19). Also,

$$
\begin{array} { r } { ( L _ { i , j } ^ { \mathrm { R } } ) ^ { 2 } = ( L _ { i , j } ^ { \mathrm { I } } ) ^ { 2 } = 4 ( E _ { i j } ^ { \dagger } E _ { i j } + E _ { i j } E _ { i j } ^ { \dagger } ) . } \end{array}
$$

Since $\rho _ { 0 } = \Pi _ { 0 } / r$ is supported on $S ,$ only $E _ { i j } ^ { \dagger } E _ { i j } = | e _ { j } \rangle \langle e _ { j } |$ contributes to the trace. Thus each of the 2mr coordinates contributes

$$
\mathrm { T r } \big ( \rho _ { 0 } ( L _ { i , j } ^ { \mathrm { R } } ) ^ { 2 } \big ) = \mathrm { T r } \big ( \rho _ { 0 } ( L _ { i , j } ^ { \mathrm { I } } ) ^ { 2 } \big ) = \frac { 4 } { r }
$$

to the trace of the single-sample quantum Fisher information matrix. Hence

$$
\operatorname { T r } ( \mathbb { Z } _ { \mathrm { Q } } ( 0 ) ) = 8 m \leq 8 d .
$$

The classical–quantum Fisher inequality and tensor product additivity in Equation (23) therefore give

$$
\operatorname { T r } ( \mathcal { T } _ { \mathsf { M } } ^ { ( t ) } ( 0 ) ) \leq 8 d t .\tag{54}
$$

Combining Equation (53) for $t \leq r ^ { 2 }$ with Equation (54) for $t > r ^ { 2 }$ , there is a universal constant $C ^ { \prime } > 0$ such that

$$
\mathrm { T r } ( \mathcal { T } _ { \mathsf { M } } ^ { ( t ) } ( 0 ) ) \leq C ^ { \prime } d t \operatorname* { m i n } \left\{ 1 , \frac { \sqrt { t } } { r } \right\} ,\tag{55}
$$

for every joint POVM M on t samples.

Transfer to an arbitrary parameter. The preceding bounds apply only at $X = 0$ . To prove the bound in Proposition 4.2 at an arbitrary X for every $t \geq 1$ , we construct a unitary $U _ { X }$ such that

$$
U _ { X } ^ { \dagger } \rho _ { X } U _ { X } = \rho _ { 0 } .
$$

We show that conjugation sends a perturbation H at X to a perturbation $\mathcal { L } _ { X } ( H )$ at 0, where

$$
\| { \mathcal { L } } _ { X } ( H ) \| _ { \mathrm { F } } \leq \| H \| _ { \mathrm { F } } .
$$

We then use these facts to bound the Fisher information trace at X by that of the conjugated measurement at 0.

Recall the isometry $V _ { X }$ from Equation (25), whose range is the support of $\rho _ { X }$ . To construct $U _ { X }$ define

$$
W _ { X } : = \left( \begin{array} { c } { { - X ^ { \dagger } } } \\ { { \mathrm { I } _ { m } } } \end{array} \right) ( \mathrm { I } _ { m } + X X ^ { \dagger } ) ^ { - 1 / 2 } .
$$

Direct multiplication gives

$$
W _ { X } ^ { \dagger } W _ { X } = \mathrm { I } _ { m } , \qquad V _ { X } ^ { \dagger } W _ { X } = 0 .
$$

Thus

$$
U _ { X } : = { \Big ( } V _ { X } \quad W _ { X } { \Big ) }
$$

is unitary. Its first r columns span the support of $\rho _ { X }$ , and therefore $U _ { X } ^ { \dagger } \rho _ { X } U _ { X } = \rho _ { 0 }$

To compare the likelihood derivatives at X and 0, for each $H \in \mathbb { C } ^ { m \times r }$ , we seek a direction $\mathcal { L } _ { X } ( H )$ satisfying

$$
U _ { X } ^ { \dagger } ( \mathrm { D } _ { H } \rho _ { X } ) U _ { X } = \mathrm { D } _ { \mathcal { L } _ { X } ( H ) } \rho _ { 0 } .
$$

Since $\rho _ { X } = \Pi _ { X } / r$ , Equation (27) shows that the desired identity is equivalent to

$$
U _ { X } ^ { \dagger } ( { \mathrm { D } } _ { H } \Pi _ { X } ) U _ { X } = \left( \begin{array} { c c } { { 0 } } & { { { \mathcal L } _ { X } ( H ) ^ { \dagger } } } \\ { { { \mathcal L } _ { X } ( H ) } } & { { 0 } } \end{array} \right) .
$$

This dictates the construction of $\mathcal { L } _ { X } ( H )$ : we show that the diagonal blocks vanish and take the lower left block to be $\mathcal { L } _ { X } ( H )$ . Since $\Pi _ { X + s H }$ is a projector for every $s ,$ diferentiating $\Pi _ { X + s H } ^ { 2 } = \Pi _ { X + s H }$ at $s = 0$ gives

$$
( \mathrm { D } _ { H } \Pi _ { X } ) \Pi _ { X } + \Pi _ { X } ( \mathrm { D } _ { H } \Pi _ { X } ) = \mathrm { D } _ { H } \Pi _ { X } .
$$

Multiplying this identity on the left and right by $\Pi _ { X }$ , and separately by $\operatorname { I } _ { d } - \Pi _ { X }$ , gives

$$
2 \Pi _ { X } \big ( \mathrm { D } _ { H } \Pi _ { X } \big ) \Pi _ { X } = \Pi _ { X } \big ( \mathrm { D } _ { H } \Pi _ { X } \big ) \Pi _ { X } , \qquad 0 = \big ( \mathrm { I } _ { d } - \Pi _ { X } \big ) \big ( \mathrm { D } _ { H } \Pi _ { X } \big ) \big ( \mathrm { I } _ { d } - \Pi _ { X } \big ) .
$$

The columns of $V _ { X }$ and $W _ { X }$ span the ranges of $\Pi _ { X }$ and $\operatorname { I } _ { d } - \Pi _ { X }$ , respectively, so the diagonal blocks vanish:

$$
U _ { X } ^ { \dagger } ( { \mathrm { D } } _ { H } \Pi _ { X } ) U _ { X } = \left( \begin{array} { c c } { { 0 } } & { { V _ { X } ^ { \dagger } ( { \mathrm { D } } _ { H } \Pi _ { X } ) W _ { X } } } \\ { { W _ { X } ^ { \dagger } ( { \mathrm { D } } _ { H } \Pi _ { X } ) V _ { X } } } & { { 0 } } \end{array} \right) .
$$

It remains to compute the lower left block. The definition of $W _ { X }$ gives

$$
W _ { X } ^ { \dagger } \left( { \small \mathrm { \large { I } } } _ { X } \right) = ( \mathrm { I } _ { m } + X X ^ { \dagger } ) ^ { - 1 / 2 } \left( - X \quad \mathrm { I } _ { m } \right) \left( { \small \mathrm { \large { I } } } _ { X } \right) = 0 .
$$

Using this identity together with $V _ { X } ^ { \dagger } V _ { X } = \operatorname { I } _ { r }$ and $W _ { X } ^ { \dagger } V _ { X } = 0$ , diferentiating $\Pi _ { X } = V _ { X } V _ { X } ^ { \dagger }$ and the explicit formula for $V _ { X }$ gives

$$
\begin{array} { l } { { W _ { X } ^ { \dagger } ( \mathrm { D } _ { H } \Pi _ { X } ) V _ { X } = W _ { X } ^ { \dagger } \left( ( \mathrm { D } _ { H } V _ { X } ) V _ { X } ^ { \dagger } + V _ { X } ( \mathrm { D } _ { H } V _ { X } ) ^ { \dagger } \right) V _ { X } } } \\ { { \ \qquad = W _ { X } ^ { \dagger } ( \mathrm { D } _ { H } V _ { X } ) } } \\ { { \ \qquad = W _ { X } ^ { \dagger } \left[ \left( { \bf 0 } _ { H } \right) \left( \mathrm { I } _ { r } + X ^ { \dagger } X \right) ^ { - 1 / 2 } + \left( \begin{array} { l } { { \mathrm { I } _ { r } } } \\ { X } \end{array} \right) \mathrm { D } _ { H } \left( \left( \mathrm { I } _ { r } + X ^ { \dagger } X \right) ^ { - 1 / 2 } \right) \right] } } \\ { { \ \qquad = W _ { X } ^ { \dagger } \left( \begin{array} { l } { { 0 } } \\ { H } \end{array} \right) \left( \mathrm { I } _ { r } + X ^ { \dagger } X \right) ^ { - 1 / 2 } } } \\ { { \ \qquad = \left( \mathrm { I } _ { m } + X X ^ { \dagger } \right) ^ { - 1 / 2 } H \left( \mathrm { I } _ { r } + X ^ { \dagger } X \right) ^ { - 1 / 2 } . } } \end{array}
$$

Define the real-linear map

$$
\mathscr { L } _ { X } ( H ) : = ( \mathrm { I } _ { m } + X X ^ { \dagger } ) ^ { - 1 / 2 } H ( \mathrm { I } _ { r } + X ^ { \dagger } X ) ^ { - 1 / 2 } .
$$

Both matrices multiplying H have operator norm at most one, so

$$
\| { \mathcal { L } } _ { X } ( H ) \| _ { \mathrm { F } } \leq \| H \| _ { \mathrm { F } } .
$$

Combining these identities gives

$$
{ \cal U } _ { X } ^ { \dagger } ( \mathrm { D } _ { H } \rho _ { X } ) { \cal U } _ { X } = { \frac { 1 } { r } } \left( \begin{array} { c c } { { 0 } } & { { { \mathcal L } _ { X } ( H ) ^ { \dagger } } } \\ { { { \mathcal L } _ { X } ( H ) } } & { { 0 } } \end{array} \right) = \mathrm { D } _ { { \mathcal L } _ { X } ( H ) } \rho _ { 0 } .
$$

Let $\widetilde { \mathsf { M } }$ be the conjugated POVM with density

$$
\widetilde { M } _ { z } : = ( U _ { X } ^ { \otimes t } ) ^ { \dagger } M _ { z } U _ { X } ^ { \otimes t } ,
$$

and let

$$
\widetilde { q } _ { K } ^ { ( t ) } ( z ) : = \mathrm { T r } \left( \widetilde { M } _ { z } \rho _ { K } ^ { \otimes t } \right)
$$

be its likelihood for $K \in \mathbb { C } ^ { m \times r }$ , with $U _ { X }$ held fixed as K varies. The preceding state identities give, for every H,

$$
q _ { X } ^ { ( t ) } ( z ) = \tilde { q } _ { 0 } ^ { ( t ) } ( z ) , \qquad \mathrm { D } _ { H } q _ { X } ^ { ( t ) } ( z ) = \mathrm { D } _ { \mathcal { L } _ { X } ( H ) } \tilde { q } _ { 0 } ^ { ( t ) } ( z ) .
$$

The derivative identity says that a direction H at X changes the likelihood in the same way as the direction $\mathcal { L } _ { X } ( H )$ at 0. By the definitions of the gradient and the adjoint, this gives $\nabla q _ { X } ^ { ( t ) } ( z ) = \mathcal { L } _ { X } ^ { \dagger } \nabla \tilde { q } _ { 0 } ^ { ( t ) } ( z )$ in real coordinates, where $\mathcal { L } _ { X } ^ { \dagger }$ is the adjoint with respect to the real Frobenius inner product. Since $q _ { X } ^ { ( t ) } ( z ) = \widetilde { q } _ { 0 } ^ { ( t ) } ( z )$ , the definition of Fisher information gives

$$
\mathcal { Z } _ { \mathsf { M } } ^ { ( t ) } ( X ) = \int \frac { \mathcal { L } _ { X } ^ { \dagger } \nabla \tilde { q } _ { 0 } ^ { ( t ) } ( z ) \nabla \tilde { q } _ { 0 } ^ { ( t ) } ( z ) ^ { \mathsf { T } } \mathcal { L } _ { X } } { \tilde { q } _ { 0 } ^ { ( t ) } ( z ) } ~ \mathrm { d } \nu ( z ) = \mathcal { L } _ { X } ^ { \dagger } \mathcal { Z } _ { \widetilde { \mathsf { M } } } ^ { ( t ) } ( 0 ) \mathcal { L } _ { X } .
$$

Since $\mathcal { L } _ { X }$ is a contraction, $\mathcal { L } _ { X } \mathcal { L } _ { X } ^ { \dag } \preceq \mathrm { I 2 } m r$ , and hence

$$
\mathrm { T r } ( \mathcal { T } _ { \mathsf { M } } ^ { ( t ) } ( X ) ) = \mathrm { T r } ( \mathcal { T } _ { \widetilde { \mathsf { M } } } ^ { ( t ) } ( 0 ) \mathcal { L } _ { X } \mathcal { L } _ { X } ^ { \dagger } ) \leq \mathrm { T r } ( \mathcal { T } _ { \widetilde { \mathsf { M } } } ^ { ( t ) } ( 0 ) ) .
$$

Proof of Proposition 4.2. At $X = 0$ , the case $t = 1$ is Equation (36), and the case $t \geq 2$ is Equation (55). For an arbitrary X, the trace inequality above bounds $\operatorname { T r } ( \mathcal { T } _ { \mathsf { M } } ^ { ( t ) } ( X ) )$ by the corresponding quantity for the conjugated measurement at 0. Applying the $X = 0$ bounds proves Equation (29).

## 4.3 Adaptive accumulation across measurement rounds

Proposition 4.2 controls the joint measurement selected in one round. To bound the full transcript of an adaptive protocol, represent the protocol as a decision tree: a complete transcript is a root-to-leaf path, and at the node corresponding to history h before round i, the protocol has selected a POVM acting jointly on $t _ { i } ( h ) \leq t$ fresh samples. We use the standard conditional score chain rule for Fisher information along this path and then sum the resulting round-by-round bounds using the pathwise sample bound.

Fix a protocol with pathwise sample bound N. Every nontrivial round uses at least one sample, so there are at most N such rounds on any execution. We give every transcript a fixed length by appending dummy rounds after the protocol halts until it has N rounds. A dummy round uses no samples and contributes no Fisher information, so this does not change the protocol. Let U be its private seed, let $Z _ { i }$ be the outcome in round i, and write $Z _ { < i } : = ( Z _ { 1 } , \ldots , Z _ { i - 1 } )$ . At history $h = ( U , Z _ { < i } )$ , the protocol selects an integer $t _ { i } ( h ) \in \{ 0 , 1 , \ldots , t \}$ and a joint POVM on $t _ { i } ( h )$ samples. Let $p _ { i , X } ( \cdot \mid h )$ denote its conditional outcome density relative to a parameter-independent reference measure $\nu _ { i } ( \cdot \mid h )$ when the input state is $\rho _ { X }$

For finite outcomes, the transcript likelihood is obtained by multiplying the conditional probabilities. The same factorization holds for the continuous outcome spaces in our model: Section A constructs jointly measurable conditional densities with respect to a parameter-independent transcript measure and verifies the likelihood regularity used below. Thus

$$
q _ { N , X } ( u , z _ { 1 } , \ldots , z _ { N } ) : = \prod _ { i = 1 } ^ { N } p _ { i , X } ( z _ { i } \mid u , z _ { < i } ) .\tag{56}
$$

Under this transcript law, $U , Z _ { 1 } , \dots , Z _ { N }$ are the random seed and outcomes. We write $\mathbb { E } _ { X }$ for expectation when the unknown state is $\rho _ { X }$

Let $\mathcal { T } _ { i } ( h ; X )$ be the Fisher information matrix contributed by the round i measurement selected at history $h ,$ evaluated at $X ,$ and let $\mathcal { T } _ { \mathrm { t r } } ^ { ( N ) } ( X )$ be the Fisher information matrix of the full transcript. Both are expressed in the same 2mr real support coordinates, so their matrices can be added. Here the subscript tr stands for transcript, while the superscript $( N )$ denotes the number of padded rounds.

Lemma 4.6 (Adaptive Fisher chain rule). For every $X$

$$
\mathcal { T } _ { \mathrm { t r } } ^ { ( N ) } ( X ) = \sum _ { i = 1 } ^ { N } \mathbb { E } _ { X } [ \mathcal { T } _ { i } ( U , Z _ { < i } ; X ) ] .\tag{57}
$$

Proof. Let $H _ { i } : = ( U , Z _ { < i } )$ be the history before round i. With $\nabla _ { X }$ denoting the gradient in the 2mr real coordinates of $X$ , define the round i score by

$$
S _ { i } ( X ) : = \nabla _ { X } \log p _ { i , X } ( Z _ { i } \mid H _ { i } ) .
$$

$\mathrm { A s }$ in Section 3.3, the score is set to zero when the conditional likelihood vanishes. The likelihood factorization and the regularity established in Lemma A.2 give

$$
\nabla _ { X } \log q _ { N , X } = \sum _ { i = 1 } ^ { N } S _ { i } ( X ) ,
$$

almost surely under the transcript law. By definition, the Fisher matrix of the transcript is therefore

$$
\begin{array} { l } { \displaystyle \mathcal { Z } _ { \mathrm { t r } } ^ { ( N ) } ( X ) = \mathbb { E } _ { X } \left[ \left( \sum _ { i = 1 } ^ { N } S _ { i } ( X ) \right) \left( \sum _ { j = 1 } ^ { N } S _ { j } ( X ) \right) ^ { \top } \right] } \\ { \displaystyle \qquad = \sum _ { i = 1 } ^ { N } \mathbb { E } _ { X } [ S _ { i } ( X ) S _ { i } ( X ) ^ { \top } ] + \sum _ { i < j } \mathbb { E } _ { X } \left[ S _ { i } ( X ) S _ { j } ( X ) ^ { \top } + S _ { j } ( X ) S _ { i } ( X ) ^ { \top } \right] . } \end{array}
$$

We show that the second sum vanishes. Each roundwise score has conditional mean zero given the preceding history. Indeed,

$$
\begin{array} { r l } {  { \mathbb { E } _ { X } [ S _ { i } ( X ) \mid H _ { i } ] = \int _ { \mathbb { Z } _ { i } } p _ { i , X } ( z \mid H _ { i } ) \nabla _ { X } \log p _ { i , X } ( z \mid H _ { i } ) \nu _ { i } ( \mathrm { d } z \mid H _ { i } ) } } \\ & { = \int _ { \mathbb { Z } _ { i } } \nabla _ { X } p _ { i , X } ( z \mid H _ { i } ) \nu _ { i } ( \mathrm { d } z \mid H _ { i } ) } \\ & { = \nabla _ { X } \int _ { \mathbb { Z } _ { i } } p _ { i , X } ( z \mid H _ { i } ) \nu _ { i } ( \mathrm { d } z \mid H _ { i } ) = \nabla _ { X } 1 = 0 . } \end{array}
$$

The reference measure is parameter independent at each fixed history, and the interchange of diferentiation and integration is justified by Lemma A.2. Therefore, if $i < j$ , then $H _ { j }$ contains the outcome of round i, and hence $S _ { i } ( X )$ is determined by $H _ { j }$ . It follows that

$$
\mathbb { E } _ { X } [ S _ { i } ( X ) S _ { j } ( X ) ^ { \mathsf { T } } ] = \mathbb { E } _ { X } \left[ S _ { i } ( X ) \mathbb { E } _ { X } [ S _ { j } ( X ) ^ { \mathsf { T } } \mid H _ { j } ] \right] = 0 .
$$

The transpose cross term vanishes in the same way. For the diagonal terms, the definition of conditional Fisher information and the law of total expectation give

$$
\begin{array} { r } { \mathbb { E } _ { X } [ S _ { i } ( X ) S _ { i } ( X ) ^ { \mathsf { T } } ] = \mathbb { E } _ { X } \left[ \mathbb { E } _ { X } [ S _ { i } ( X ) S _ { i } ( X ) ^ { \mathsf { T } } \mid H _ { i } ] \right] = \mathbb { E } _ { X } [ \mathbb { Z } _ { i } ( H _ { i } ; X ) ] . } \end{array}
$$

Substituting these identities into the expansion above proves Equation (57). The seed contributes no term because its law is parameter independent. □

Corollary 4.7 (Adaptive Fisher information bound). Consider the family $\{ \rho _ { X } \}$ of rank-r states defined in Section $4 . 1$ . Let an adaptive protocol jointly measure at most t fresh samples in each round and use at most N samples along every root-to-leaf path. Then the Fisher information matrix of its transcript satisfies, for every $X$

$$
\mathrm { T r } \big ( \mathcal { T } _ { \mathrm { t r } } ^ { ( N ) } ( X ) \big ) \leq C d N \operatorname* { m i n } \left\{ 1 , \frac { \sqrt { t } } { r } \right\} .\tag{58}
$$

Proof. If $t _ { i } ( h ) = 0$ , the conditional Fisher information is zero. Otherwise, Proposition 4.2 gives

$$
\operatorname { T r } ( \mathcal { T } _ { i } ( h ; X ) ) \leq C d t _ { i } ( h ) \operatorname* { m i n } \left\{ 1 , { \frac { \sqrt { t _ { i } ( h ) } } { r } } \right\} \leq C d t _ { i } ( h ) \operatorname* { m i n } \left\{ 1 , { \frac { \sqrt { t } } { r } } \right\} .
$$

Taking traces in Equation (57) and using the pathwise sample bound yields

$$
\mathrm { T r } \bigl ( \mathcal { T } _ { \mathrm { t r } } ^ { ( N ) } ( X ) \bigr ) \leq C d \operatorname* { m i n } \left\{ 1 , \frac { \sqrt { t } } { r } \right\} \mathbb { E } _ { X } \left[ \sum _ { i = 1 } ^ { N } t _ { i } ( U , Z _ { < i } ) \right] \leq C d N \operatorname* { m i n } \left\{ 1 , \frac { \sqrt { t } } { r } \right\} .
$$

## 4.4 From a Fisher information bound to expected trace norm loss

The adaptive Fisher information bound controls how much the transcript can reveal about the support parameter X. We now convert it into a lower bound on expected trace norm loss. We apply the van Trees inequality using a smooth probability density π supported on parameters satisfying $\| X \| _ { \mathrm { o p } } < a$ . Its denominator contains the Fisher information contributed by the measurements and $I ( \pi )$ . Restricting the estimator $T$ to satisfy $\| T \| _ { \mathrm { o p } } \leq a$ then converts the resulting Frobenius error bound into a trace norm bound. The postprocessing in Section 4.5.1 will produce exactly such an estimator from the output of a tomography protocol.

## 4.4.1 The van Trees inequality

The van Trees inequality formalizes a simple principle: accurate estimation requires suficient information. The denominator in Equation (60) below contains the Fisher information contributed by the measurement transcript and $I ( \pi )$ , the Fisher information of the probability density π. If their sum is small, then the expected squared estimation error on the left side must be large.

Let $\Theta \subseteq \mathbb { R } ^ { p }$ be open, and draw θ from a smooth, compactly supported probability density π on $\Theta .$ Conditional on $\theta ,$ let the observation have likelihood $q _ { \theta }$ in a regular dominated model. Let $\mathcal { T } ( \pmb \theta )$ denote the Fisher information matrix of this likelihood family. The precise regularity conditions are stated in Section A.4; Lemma A.2 verifies them for the adaptive transcript model used here.

We write $C _ { c } ^ { 1 } ( \Theta ; \mathbb { R } )$ for the real-valued functions on Θ that are continuously diferentiable and vanish outside a compact subset of $\Theta .$ . The superscript 1 means that the first partial derivatives exist and are continuous, the subscript c means compact support, and $( \Theta ; \mathbb { R } )$ gives the domain and codomain. To obtain a nonnegative probability density that vanishes near the boundary and whose Fisher information is easy to control, we begin with a function $\psi \in C _ { c } ^ { 1 } ( \Theta ; \mathbb { R } )$ , normalize it in $L _ { 2 }$ and define

$$
\pi ( \pmb \theta ) : = \psi ( \pmb \theta ) ^ { 2 } , \qquad \| \psi \| _ { 2 } ^ { 2 } : = \int _ { \Theta } | \psi ( \pmb \theta ) | ^ { 2 } ~ \mathrm d \pmb \theta = 1 .
$$

Then π integrates to one and vanishes near the boundary of Θ. We define its Fisher information by

$$
I ( \boldsymbol { \pi } ) : = 4 \int _ { \Theta } \| \nabla \boldsymbol { \psi } ( \pmb { \theta } ) \| _ { 2 } ^ { 2 } \mathrm { d } \pmb { \theta } .\tag{59}
$$

This agrees with the usual Fisher information of the density π. Indeed, wherever $\pi ( \pmb \theta ) > 0$

$$
\| \nabla \log \pi ( \pmb \theta ) \| _ { 2 } ^ { 2 } \pi ( \pmb \theta ) = \frac { \| \nabla \pi ( \pmb \theta ) \| _ { 2 } ^ { 2 } } { \pi ( \pmb \theta ) } = \frac { \| 2 \psi ( \pmb \theta ) \nabla \psi ( \pmb \theta ) \| _ { 2 } ^ { 2 } } { \psi ( \pmb \theta ) ^ { 2 } } = 4 \| \nabla \psi ( \pmb \theta ) \| _ { 2 } ^ { 2 } .
$$

When $\pi ( \pmb \theta ) = 0$ , the logarithm is undefined, whereas the expression in Equation (59) remains well defined.

Lemma 4.8 (Van Trees inequality). Assume the preceding likelihood regularity, and let $\pi ( \pmb { \theta } ) : = \psi ( \pmb { \theta } ) ^ { 2 }$ 2 where $\psi \in C _ { c } ^ { 1 } ( \Theta ; \mathbb { R } )$ and $\| \psi \| _ { 2 } = 1$ . Then every bounded measurable estimator T taking values in $\mathbb { R } ^ { p }$ satisfies

$$
\mathbb { E } \left\| T - \pmb { \theta } \right\| _ { 2 } ^ { 2 } \geq \frac { p ^ { 2 } } { \mathbb { E } _ { \pmb { \theta } } \operatorname { T r } \left( \mathbb { Z } ( \pmb { \theta } ) \right) + I ( \pi ) } .\tag{60}
$$

The expectation first draws $\theta \sim \pi$ and then draws the observation from $q \theta$

The proof is given in Section A.4.

## 4.4.2 A smooth probability density for the support parameter

Recall that $m = d - r ,$ , and put $p : = 2 m r$ . We will apply van Trees on $\mathbb { C } ^ { m \times r }$ with a smooth density whose Fisher information is bounded by a universal constant times $p d / a ^ { 2 }$ . We choose the density to vanish unless $\begin{array} { r } { \| X \| _ { \mathrm { o p } } < a ; } \end{array}$ together with the same constraint on the estimator, this will give the trace norm conversion in the next subsection.

Lemma 4.9 (Smooth probability density for rectangular parameters). There is a universal constant $C > 0$ with the following property. For every $d \geq 2 , 1 \leq r < d _ { }$ , and $a > 0$ , put $m : = d - r$ and $p : = 2 m r$ . There is a nonnegative, infinitely diferentiable function ψ on the real Euclidean space $\mathbb { C } ^ { m \times r }$ , normalized by $\begin{array} { r } { \int { \psi ( X ) ^ { 2 } } \mathrm { d } X = 1 } \end{array}$ , that vanishes outside a compact subset of $\{ X : \| X \| _ { \mathrm { o p } } < a \}$ The probability density $\pi ( X ) : = \psi ( X ) ^ { 2 }$ satisfies

$$
I ( \pi ) \leq C { \frac { p d } { a ^ { 2 } } } .\tag{61}
$$

Proof. We begin with a Gaussian density whose variance is chosen so that most of its probability mass lies in the region $\| X \| _ { \mathrm { o p } } \leq a / 2$ . We then multiply its square root by a smooth cutof that equals one on this region and vanishes before $\| X \| _ { \mathrm { o p } }$ reaches a. The cutof will have gradient norm $O ( 1 / a )$ , which keeps its contribution to $I ( \pi )$ under control.

Regard $\mathbb { C } ^ { m \times r }$ as the real Euclidean space from Equation (8). Let $G \in \mathbb { C } ^ { m \times r }$ have entries $G _ { i j } = \xi _ { i j } + \mathrm { i } \eta _ { i j }$ , for $1 \leq i \leq$ m and $1 \le j \le r$ , where all the real random variables $\xi _ { i j }$ and $\eta _ { i j }$ are independent and distributed as ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ . Thus, with respect to the $p =$ 2mr real coordinates, the density of G is

$$
g _ { \sigma } ( X ) = { \frac { 1 } { ( 2 \pi \sigma ^ { 2 } ) ^ { p / 2 } } } \exp \left( - { \frac { \| X \| _ { \mathrm { F } } ^ { 2 } } { 2 \sigma ^ { 2 } } } \right) .
$$

We first show that a constant fraction of the probability mass of G lies where $\| G \| _ { \mathrm { o p } } \leq a / 2$ . Write $G = A + \mathrm { i } B$ , where A and B are independent real Gaussian matrices whose entries have variance $\sigma ^ { 2 }$

Applying the standard operator norm bound for real Gaussian matrices $( \mathrm { s e e } , \mathrm { e . g . }$ , Vershynin $\mathrm { [ V e r 2 6 , }$ Theorem 4.6.1]), followed by a union bound, gives a universal constant $C > 0$ such that

$$
\mathbb { P } \Big [ \| G \| _ { \mathrm { o p } } > C \sigma ( \sqrt { m } + \sqrt { r } ) \Big ] \leq \frac { 1 } { 4 } .
$$

Since ${ \sqrt { m } } + { \sqrt { r } } \leq { \sqrt { 2 d } }$ , choosing $\sigma = c a / \sqrt { d }$ for a suficiently small universal constant $c > 0$ gives

$$
\mathbb { P } \Big [ \| G \| _ { \mathrm { o p } } \leq a / 2 \Big ] \geq \frac { 3 } { 4 } .
$$

We next construct the smooth cutof. Let $h : [ 0 , \infty )  [ 0 , 1 ]$ equal one on $[ 0 , 9 / 1 6 ]$ , vanish on $[ 1 1 / 1 6 , \infty )$ , and be linear between these intervals. Let $\varphi$ be a nonnegative infinitely diferentiable function on $\mathbb { C } ^ { m \times r }$ , supported where $\| H \| _ { \mathrm { F } } \leq a / 1 6$ , and normalized so that $\begin{array} { r } { \int \varphi ( H ) \ \mathrm { d } H = 1 } \end{array}$ . Average the cutof over these small perturbations by setting

$$
\chi ( X ) : = \int h \left( \frac { \| X - H \| _ { \mathrm { o p } } } { a } \right) \varphi ( H ) \mathrm { d } H = \int h \left( \frac { \| Y \| _ { \mathrm { o p } } } { a } \right) \varphi ( X - Y ) \mathrm { d } Y .
$$

The first expression shows that $\chi ( X )$ is a weighted average of the cutof at matrices close to $X$ The second shows that $\chi$ is infinitely diferentiable: diferentiating with respect to X diferentiates $\varphi ( X - Y )$ , not the operator norm. By construction, $0 \leq \chi \leq 1$

If $\| X \| _ { \mathrm { o p } } \leq a / 2$ and $\varphi ( H ) \neq 0$ , then

$$
\left. { \boldsymbol { X } } - { \boldsymbol { H } } \right. _ { \mathrm { o p } } \leq \left. { \boldsymbol { X } } \right. _ { \mathrm { o p } } + \left. { \boldsymbol { H } } \right. _ { \mathrm { F } } \leq { \frac { 9 a } { 1 6 } } ,
$$

so $\chi ( X ) = 1$ . Similarly, if $\| X \| _ { \mathrm { o p } } \geq 3 a / 4 ,$ then $\| X - H \| _ { \mathrm { o p } } \geq 1 1 a / 1 6$ whenever $\varphi ( H ) \neq 0$ , so $\chi ( X ) = 0$ . Finally, the operator norm is 1-Lipschitz with respect to the Frobenius norm, and $h$ is 8-Lipschitz. Averaging its translates therefore gives

$$
| \chi ( X ) - \chi ( Y ) | \leq \frac { 8 } { a } \| X - Y \| _ { \mathrm { F } } , \qquad \| \nabla \chi ( X ) \| _ { 2 } \leq \frac { 8 } { a } .
$$

Define

$$
\widetilde { \psi } ( X ) : = \sqrt { g _ { \sigma } ( X ) } \chi ( X ) , \qquad Z : = \left\| \widetilde { \psi } \right\| _ { 2 } ^ { 2 } .
$$

Because $\chi = 1$ when $\| X \| _ { \mathrm { o p } } \leq a / 2$

$$
Z = \int g _ { \sigma } ( X ) \chi ( X ) ^ { 2 } ~ \mathrm { d } X \geq \mathbb { P } \Big [ \| G \| _ { \mathrm { o p } } \leq a / 2 \Big ] \geq \frac { 3 } { 4 } .
$$

The Gaussian factor satisfies

$$
\nabla \sqrt { g _ { \sigma } ( X ) } = - \frac { X } { 2 \sigma ^ { 2 } } \sqrt { g _ { \sigma } ( X ) } , \qquad \int \| \nabla \sqrt { g _ { \sigma } } \| _ { 2 } ^ { 2 } = \frac { \mathbb { E } \| G \| _ { \mathrm { F } } ^ { 2 } } { 4 \sigma ^ { 4 } } = \frac { p } { 4 \sigma ^ { 2 } } .
$$

The product rule now gives

$$
\nabla  { \widetilde { \psi } } = \chi \nabla \sqrt { g _ { \sigma } } + \sqrt { g _ { \sigma } } \nabla \chi ,
$$

and the inequality $\left\| \pmb { u } + \pmb { v } \right\| _ { 2 } ^ { 2 } \leq 2 \left\| \pmb { u } \right\| _ { 2 } ^ { 2 } + 2 \left\| \pmb { v } \right\| _ { 2 } ^ { 2 }$ gives

$$
\begin{array} { r l } & { \displaystyle \int \left\| \nabla \widetilde { \psi } \right\| _ { 2 } ^ { 2 } \leq 2 \int \chi ^ { 2 } \| \nabla \sqrt { g _ { \sigma } } \| _ { 2 } ^ { 2 } + 2 \int g _ { \sigma } \| \nabla \chi \| _ { 2 } ^ { 2 } } \\ & { \qquad \leq 2 \displaystyle \int \| \nabla \sqrt { g _ { \sigma } } \| _ { 2 } ^ { 2 } + \frac { C } { a ^ { 2 } } \int g _ { \sigma } } \\ & { \qquad = \displaystyle \frac { p } { 2 \sigma ^ { 2 } } + \frac { C } { a ^ { 2 } } } \\ & { \qquad = \displaystyle \frac { p d } { 2 c ^ { 2 } a ^ { 2 } } + \frac { C } { a ^ { 2 } } \leq C \displaystyle \frac { p d } { a ^ { 2 } } . } \end{array}
$$

The second inequality uses $0 \leq \chi \leq 1$ and $\| \nabla \chi \| _ { 2 } \leq 8 / a$ . The two equalities use the Gaussian integral above, $\textstyle \int g _ { \sigma } = 1$ , and $\sigma = c a / \sqrt { d } ;$ the last inequality uses $p d \geq 1$ and that c is a universal constant. Normalize by setting $\psi : = \widetilde \psi / \sqrt { Z }$ . The function ψ is nonnegative and infinitely diferentiable, has norm one in $L _ { 2 }$ , and vanishes whenever $\| X \| _ { \mathrm { o p } } \geq 3 a / 4$ . Its support is therefore a compact subset of $\{ X : \| X \| _ { \mathrm { o p } } < a \}$ . Since $Z \ge 3 / 4$

$$
4 \int \left\| \nabla \psi \right\| _ { 2 } ^ { 2 } = \frac { 4 } { Z } \int \left\| \nabla \tilde { \psi } \right\| _ { 2 } ^ { 2 } \leq C \frac { p d } { a ^ { 2 } } .
$$

The density $\pi : = \psi ^ { 2 }$ proves the lemma.

## 4.4.3 From Frobenius loss to trace norm loss

The van Trees inequality gives a lower bound on squared Frobenius error, whereas the theorem concerns trace norm loss. If the parameter and the estimator both have operator norm at most a, then their diference has operator norm at most 2a, which converts one loss into the other.

Lemma 4.10 (Trace norm consequence of van Trees). Let $X \sim \pi$ take values in $\mathbb { C } ^ { m \times r }$ , viewed as a real Euclidean space of dimension $p : = 2 m r$ , and suppose the density and likelihood satisfy the assumptions of Lemma 4.8. Assume that π vanishes unless $\| X \| _ { \mathrm { o p } } < a$ . Then every measurable estimator $T \in \mathbb { C } ^ { m \times r }$ satisfying $\| T \| _ { \mathrm { o p } } \leq a$ almost surely obeys

$$
\mathbb { E } \left\| T - X \right\| _ { 1 } \geq \frac { p ^ { 2 } } { 2 a \left( \mathbb { E } _ { X \sim \pi } \operatorname { T r } ( \mathcal { I } ( X ) ) + I ( \pi ) \right) } .\tag{62}
$$

Here the expectation is over both $X \sim \pi$ and the corresponding observation.

Proof. Under the real inner product from Equation (8), the Euclidean norm in Lemma 4.8 is the Frobenius norm. For every realization of X and the observation, the singular values of $T - X$ give

$$
\begin{array} { r } { \left\| T - X \right\| _ { \mathrm { F } } ^ { 2 } \leq \left\| T - X \right\| _ { \mathrm { o p } } \left\| T - X \right\| _ { 1 } \leq 2 a \left\| T - X \right\| _ { 1 } . } \end{array}
$$

Taking expectations and applying Lemma 4.8 proves the claim.

## 4.5 Completion of the lower bound

It remains to connect the expected loss bound from Lemma 4.10 to the high-probability guarantee in the theorem. We first reduce the failure probability by repeating the protocol. We then postprocess the tomographic output into a support parameter satisfying the operator norm constraint used to convert Frobenius loss into trace norm loss. Combining the resulting upper and lower bounds on its expected loss proves the theorem when the rank is at most half the dimension. A depolarizing channel reduction handles the remaining ranks.

## 4.5.1 From high-probability tomography to a support estimator

The tomography guarantee allows a failure event of probability $1 / 3 ,$ on which the output may be arbitrarily inaccurate. Before passing to expected loss, we reduce this probability to a small constant δ. We use the standard confidence amplification rule that selects a candidate with the smallest majority radius; see Hsu and Sabato [HS16, Sec. 3.2, Proposition 8 and Algorithm 2]. We include the short argument for completeness.

Lemma 4.11 (Confidence amplification). Let $\eta \geq 0$ , and suppose an estimator in a metric space is within η of its target with probability at least $2 / 3$ . For every $0 < \delta < 1$ , there is an odd integer $K = K ( \delta )$ such that K independent repetitions can be postprocessed, without further observations, into an estimator within 3η of the target with probability at least $1 - \delta$

Proof. Choose an odd K so that more than half of the repetitions are successful with probability at least $1 - \delta$ . Hoefding’s inequality shows that $K = O ( 1 + \log ( 1 / \delta ) )$ sufices. For outputs $T _ { 1 } , \dots , T _ { K } .$ let $R _ { j }$ be the $( K + 1 ) / 2 \mathrm { { \mathrm { - } } t h }$ smallest value among the distances from $T _ { j }$ to the K outputs. Select an output with minimum $R _ { j }$ , breaking ties by index.

Suppose that more than half of the outputs are within η of the target. Every such output is within 2η of every other successful output, and hence has $R _ { j } \leq 2 \eta$ . The selected output therefore has more than half of the outputs within distance $2 \eta$ . This set intersects the successful majority, so the selected output is within 3η of the target. □

For $1 \leq r \leq d / 2$ and $a > 0$ , write

$$
\mathcal { X } _ { a } : = \left\{ X \in \mathbb { C } ^ { m \times r } : \left\| X \right\| _ { \mathrm { o p } } \leq a \right\} .
$$

Recall that $X \mapsto \rho _ { X }$ denotes the rank-r family from Section 4.1. The next lemma converts an arbitrary matrix that accurately estimates $\rho _ { X }$ into an estimate of X belonging to $\mathcal { X } _ { a }$

Lemma 4.12 (Postprocessing to a bounded support estimator). Let $d \geq 2$ and $1 \leq r \leq d / 2$ be integers, let $0 < a \le 1 / 4$ and $\eta > 0$ , and let $0 < \delta < 1$ . Suppose a measurable d × d matrix-valued estimator $\overline { { \rho } }$ satisfies

$$
\mathbb { P } _ { \rho _ { X } } [ \left. \overline { { \rho } } - \rho _ { X } \right. _ { 1 } \leq 3 \eta ] \geq 1 - \delta
$$

for every $X \in \mathcal { X } _ { a }$ . Then there is a measurable function of ${ \overline { { \rho } } } ,$ denoted $\widehat { X }$ , that takes values in $\mathcal { X } _ { a }$ and satisfies, for every $X \in \mathcal { X } _ { a }$ ，

$$
\mathbb { E } _ { X } \left\| \widehat { X } - X \right\| _ { 1 } \leq 1 4 r \eta + 2 \delta a r .\tag{63}
$$

Proof. The set $\mathcal { X } _ { a }$ is compact, and the map $X \mapsto \rho _ { X }$ is continuous. For each $Y \in \mathcal { X } _ { a }$ , the set

$$
\{ X \in \mathcal { X } _ { a } : \| \rho _ { X } - \rho _ { Y } \| _ { 1 } < \eta \}
$$

is open in $\mathcal { X } _ { a } .$ , and these sets cover $\mathcal { X } _ { a }$ as $Y$ varies. By compactness, finitely many of them, centered at parameters $Y _ { 1 } , \dots , Y _ { M } \in \mathcal { X } _ { a }$ , already cover $\mathcal { X } _ { a }$ . Put $\mathcal { C } : = \{ Y _ { 1 } , \ldots , Y _ { M } \}$ . Then, for every $X \in \mathcal { X } _ { a }$ some $Y \in { \mathcal { C } }$ satisfies

$$
\| \rho _ { Y } - \rho _ { X } \| _ { 1 } < \eta .\tag{64}
$$

Given ${ \overline { { \rho } } } ,$ choose ${ \widehat { X } } \in { \mathcal { C } }$ minimizing $\left\| { \overline { { \rho } } } - \rho _ { \widehat { X } } \right\| _ { 1 }$ , with ties broken according to a fixed ordering of $\mathcal { C } .$ This is a measurable postprocessing and guarantees $\left. \hat { X } \right. _ { \mathrm { o p } } \leq a$

Fix the true parameter $X$ , and choose Y as in Equation (64). On the event $\| \overline { { \rho } } - \rho _ { X } \| _ { 1 } \leq 3 \eta$ 2 the minimizing property of $\widehat { X }$ gives

$$
\left. \overline { { \rho } } - \rho _ { \widehat { X } } \right. _ { 1 } \leq \left. \overline { { \rho } } - \rho _ { Y } \right. _ { 1 } \leq 4 \eta .
$$

Hence

$$
\left\| \rho _ { \widehat { X } } - \rho _ { X } \right\| _ { 1 } \leq \left\| \rho _ { \widehat { X } } - \overline { { \rho } } \right\| _ { 1 } + \left\| \overline { { \rho } } - \rho _ { X } \right\| _ { 1 } \leq 7 \eta ,
$$

and the inverse bound in Equation (28) gives

$$
\left. \widehat { X } - X \right. _ { 1 } \leq 1 4 r \eta .
$$

On the complementary event, both parameters belong to $\mathcal { X } _ { a }$ . Since ${ \widehat { X } } - X$ has at most r nonzero singular values,

$$
\left\| \widehat { X } - X \right\| _ { 1 } \leq r \left\| \widehat { X } - X \right\| _ { \mathrm { o p } } \leq 2 a r .
$$

Averaging over the success and failure events gives

$$
\begin{array} { r } { \mathbb { E } _ { X } \left\| \widehat { X } - X \right\| _ { 1 } \leq 1 4 r \eta + 2 a r \mathbb { P } _ { \rho _ { X } } [ \left\| \overline { { \rho } } - \rho _ { X } \right\| _ { 1 } > 3 \eta ] \leq 1 4 r \eta + 2 \delta a r , } \end{array}
$$

which proves Equation (63).

## 4.5.2 The lower bound when the rank is at most half the dimension

We now apply van Trees to the estimator produced above. Successful tomography gives it expected trace norm loss much smaller than ar, once a is chosen as a suficiently large constant multiple of the target accuracy and $\delta$ is suficiently small. If the protocol uses too few samples, however, the Fisher information bound forces expected loss of order $a r$ . Comparing these conclusions gives the lower bound on sample complexity.

Proposition 4.13 (Lower bound when the rank is at most half the dimension). There are universal constants $c _ { 0 } , \eta _ { 0 } > 0$ with the following property. Let $d \geq 2 , 1 \leq r \leq d / 2$ , and $t \geq 1$ be integers, and let $0 < \eta \leq \eta _ { 0 }$ . Consider an adaptive protocol that jointly measures at most t fresh samples in each round and whose output $\widehat { \rho }$ may be an arbitrary $d \times d$ matrix. Suppose that, for every state $\rho$ on $\mathbb { C } ^ { d }$ of rank exactly $^ { r , }$

$$
\mathbb { P } _ { \rho } [ \| \widehat { \rho } - \rho \| _ { 1 } \leq \eta ] \geq \frac { 2 } { 3 } .
$$

If the protocol uses at most n samples on every execution, then

$$
n \geq c _ { 0 } { \frac { d r } { \eta ^ { 2 } } } \operatorname* { m a x } \left\{ 1 , { \frac { r } { \sqrt { t } } } \right\} .\tag{65}
$$

Proof. Let $L > 1$ and $0 < \delta < 1$ be universal constants to be chosen below, and set $a : = L \eta$ . We will choose $\eta _ { 0 }$ so that $a \leq 1 / 4$ . Repeat the protocol independently $K = K ( \delta )$ times and apply Lemma 4.11 in trace norm. The amplified protocol uses at most $N : = K n$ samples on every execution, still jointly measures at most t samples in each round, and outputs a matrix $\overline { { \rho } }$ satisfying

$$
\mathbb { P } _ { \rho } [ \left. \overline { { \rho } } - \rho \right. _ { 1 } \leq 3 \eta ] \geq 1 - \delta
$$

for every state $\rho$ of rank exactly r. Applying Lemma 4.12 to the family $\{ \rho _ { X } : X \in \mathcal { X } _ { a } \}$ produces a measurable estimator $\widehat { X } \in \mathcal { X } _ { a }$ such that, for every $X \in \mathcal { X } _ { a }$ ，

$$
\mathbb { E } _ { X } \left\| \widehat { X } - X \right\| _ { 1 } \leq \left( \frac { 1 4 } { L } + 2 \delta \right) a r .\tag{66}
$$

We next derive the incompatible lower bound. Put $p : = 2 m r$ , the real dimension of the parameter X. Apply Lemma 4.9 and draw $X$ from the resulting density π. This density is supported where $\| X \| _ { \mathrm { o p } } < a$ and satisfies

$$
I ( \pi ) \leq C { \frac { p d } { a ^ { 2 } } } .\tag{67}
$$

From now on, an unqualified expectation first draws $X \sim \pi$ and then draws the complete amplified transcript. Since Equation (66) holds for every X in the support of $\pi _ { \ i }$ it gives the same upper bound under this joint expectation.

We apply the van Trees inequality to $\widehat { X }$ , viewed as a measurable function of the complete transcript of the amplified protocol. By construction, $\left\| \hat { X } \right\| _ { \mathrm { o p } } \leq a$ . By Corollary 4.7,

$$
\mathbb { E } _ { X \sim \pi } \operatorname { T r } \big ( \mathcal { T } _ { \mathrm { t r } } ^ { ( N ) } ( X ) \big ) \le C d N \operatorname* { m i n } \left\{ 1 , \frac { \sqrt { t } } { r } \right\} .\tag{68}
$$

Therefore, Lemma 4.10 and Equations (67) and (68) give

$$
\mathbb { E } \left\| \widehat { X } - X \right\| _ { 1 } \geq \frac { p ^ { 2 } } { 2 a \left( C d N \operatorname* { m i n } \left\{ 1 , \frac { \sqrt { t } } { r } \right\} + C p d / a ^ { 2 } \right) } .\tag{69}
$$

Suppose, toward a contradiction, that

$$
N \leq \frac { d r } { a ^ { 2 } } \operatorname* { m a x } \left\{ 1 , \frac { r } { \sqrt { t } } \right\} .\tag{70}
$$

The two factors involving t cancel:

$$
N \operatorname* { m i n } \left\{ 1 , { \frac { \sqrt { t } } { r } } \right\} \leq { \frac { d r } { a ^ { 2 } } } \operatorname* { m a x } \left\{ 1 , { \frac { r } { \sqrt { t } } } \right\} \operatorname* { m i n } \left\{ 1 , { \frac { \sqrt { t } } { r } } \right\} = { \frac { d r } { a ^ { 2 } } } .
$$

Since $r \leq d / 2$ , we have $m \geq d / 2$ and hence $p = 2 m r \geq d r$ . Both terms in the denominator of Equation (69) are consequently at most a universal constant times $p d / a ^ { 2 }$ . It follows that, for a universal constant $c _ { 1 } > 0$

$$
\mathbb { E } \left\| \widehat { X } - X \right\| _ { 1 } \geq c _ { 1 } \frac { p a } { d } = 2 c _ { 1 } a r \frac { m } { d } \geq c _ { 1 } a r .\tag{71}
$$

Choose L large enough and then δ small enough that $1 4 / L + 2 \delta < c _ { 1 }$ The upper bound Equation (66) then contradicts Equation (71). Thus Equation (70) is false. Since $N = K n$ and $a = L \eta$

$$
n > \frac { 1 } { K L ^ { 2 } } \frac { d r } { \eta ^ { 2 } } \operatorname* { m a x } \left\{ 1 , \frac { r } { \sqrt { t } } \right\} .
$$

Once L and $\delta$ are fixed, so is $K$ . Absorbing this fixed factor into $c _ { 0 }$ , and choosing $\eta _ { 0 } \leq 1 / ( 4 L )$ , proves the proposition. □

## 4.5.3 Rank lifting by depolarization

It remains to reduce ranks larger than half the ambient dimension to the regime of the preceding proposition. The following channel embeds the input state and mixes it with the maximally mixed state on the same r-dimensional subspace. The added component makes every output have rank exactly $^ { r , }$ but it cancels when two outputs are subtracted, so every trace norm distance is scaled by exactly one half. Applying the adjoint channel to each POVM then simulates any adaptive measurement protocol for the lifted states using samples of the original states.

Lemma 4.14 (Depolarizing rank lift). Let $1 \leq r \leq$ d and $t \geq 1$ be integers, let $W : \mathbb { C } ^ { r } \to \mathbb { C } ^ { d }$ be an isometry, and put $\tau : = \mathrm { I } _ { r } / r$ . Define the channel

$$
\Phi ( A ) : = \frac { 1 } { 2 } W A W ^ { \dagger } + \frac { 1 } { 2 } \mathrm { T r } ( A ) W \tau W ^ { \dagger } .\tag{72}
$$

For every state $\sigma$ on $\mathbb { C } ^ { r }$ , the state $\Phi ( \sigma )$ has rank exactly r. For every pair of states $\sigma , \sigma ^ { \prime }$

$$
\| \Phi ( \sigma ) - \Phi ( \sigma ^ { \prime } ) \| _ { 1 } = \frac { 1 } { 2 } \left\| \sigma - \sigma ^ { \prime } \right\| _ { 1 } .\tag{73}
$$

Moreover, an adaptive protocol for the lifted states that jointly measures at most t fresh samples in each round induces a protocol for the original states with identical transcript laws, the same round sizes, and the same sample count on every execution.

If the lifted protocol outputs a $d \times d$ matrix $B ,$ , the induced protocol may output

$$
\begin{array} { r } { \mathcal { R } ( B ) : = 2 W ^ { \dagger } B W - \tau . } \end{array}
$$

For every state $\sigma _ { \mathrm { { ; } } }$ , this postprocessing satisfies

$$
\begin{array} { r } { \| \mathcal { R } ( B ) - \sigma \| _ { 1 } \leq 2 \| B - \Phi ( \sigma ) \| _ { 1 } . } \end{array}\tag{74}
$$

Proof. The map $\Phi$ is completely positive and trace preserving, and hence is a quantum channel. For a state $\sigma ,$

$$
\Phi ( \sigma ) = W \left( \frac { \sigma + \tau } { 2 } \right) W ^ { \dagger } .
$$

The operator $( \sigma + \tau ) / 2$ is positive definite on $\mathbb { C } ^ { r }$ , so $W ( \sigma + \tau ) W ^ { \dagger } / 2$ has rank r. Also,

$$
\Phi ( \sigma ) - \Phi ( \sigma ^ { \prime } ) = \frac { 1 } { 2 } W ( \sigma - \sigma ^ { \prime } ) W ^ { \dag } .
$$

Isometric embedding preserves the nonzero singular values, which proves Equation (73).

The output postprocessing obeys the exact identity

$$
\mathscr { R } ( B ) - \sigma = 2 W ^ { \dagger } \bigl ( B - \Phi ( \sigma ) \bigr ) W .
$$

Since $\left. W ^ { \dagger } A W \right. _ { 1 } \leq \left. A \right. _ { 1 }$ , this proves Equation (74). The map $\mathcal { R }$ is continuous and hence is a measurable postprocessing.

It remains to simulate the adaptive measurements. Suppose that, after a history h in round $i ,$ the lifted protocol chooses a POVM ${ \mathsf { M } } ( \cdot \mid h )$ on $t _ { i } ( h ) \leq t$ samples. For every measurable outcome event $E ,$ define

$$
\mathsf { M } ^ { \prime } ( E \mid h ) : = \left( \Phi ^ { \otimes t _ { i } ( h ) } \right) ^ { \dagger } \left( \mathsf { M } ( E \mid h ) \right) .
$$

For each $k \in \{ 1 , \ldots , t \}$ , on the measurable set of histories where $t _ { i } ( h ) = k$ , this construction applies the fixed linear map $( \Phi ^ { \otimes k } ) ^ { \dagger }$ <sup>†</sup> to the original measurable POVM rule. Hence the transformed rule is also measurable. The adjoint of $\Phi ^ { \otimes t _ { i } ( h ) }$ is completely positive and unital, and hence $M ^ { \prime } ( \cdot \mid h )$ is a POVM. By the definition of the adjoint,

$$
\mathrm { T r } \left( \mathbb { M } ^ { \prime } ( E \mid h ) \sigma ^ { \otimes t _ { i } ( h ) } \right) = \mathrm { T r } \left( \mathbb { M } ( E \mid h ) \Phi ( \sigma ) ^ { \otimes t _ { i } ( h ) } \right) .
$$

Thus the conditional outcome law agrees at every history. Induction over the rounds gives identical full transcript laws. The simulated protocol uses the same $t _ { i } ( h )$ at every history, so neither the allowed joint measurement size nor the total sample count on any execution changes. □

Proof of Theorem 1.1. Let $c _ { 0 } , \eta _ { 0 }$ be the constants from Proposition 4.13, and set $\varepsilon _ { 0 } : = \eta _ { 0 } / 2$ . First suppose $r \leq d / 2$ . Applying Proposition 4.13 to the states of rank $^ { r , }$ with accuracy $\eta = \varepsilon ,$ gives

$$
n \geq c _ { 0 } \frac { d r } { \varepsilon ^ { 2 } } \operatorname* { m a x } \left\{ 1 , \frac { r } { \sqrt { t } } \right\} .
$$

Now suppose $r > d / 2 ,$ so $r \geq 2$ , and set $k : = \lfloor r / 2 \rfloor$ . Fix an isometry $W : \mathbb { C } ^ { r } \to \mathbb { C } ^ { d }$ , independently of the input state, and apply Lemma 4.14. For every state σ on $\mathbb { C } ^ { r }$ of rank exactly k, the state $W \sigma W ^ { \dagger }$ has support contained in the same fixed r-dimensional subspace Im(W), and $\Phi ( \sigma )$ has support exactly Im(W), hence rank exactly $^ r .$ We may therefore simulate the assumed protocol on $\Phi ( \sigma )$ and postprocess its output $\widehat { \rho }$ as $\widehat { \sigma } : = \mathcal { R } ( \widehat { \rho } )$ . By Equation (74),

$$
\mathbb { P } _ { \sigma } [ \| \widehat { \sigma } - \sigma \| _ { 1 } \leq 2 \varepsilon ] \geq \mathbb { P } _ { \Phi ( \sigma ) } [ \| \widehat { \rho } - \Phi ( \sigma ) \| _ { 1 } \leq \varepsilon ] \geq \frac { 2 } { 3 } .
$$

The induced protocol jointly measures at most t samples in each round and uses at most n samples on every execution. Its output need not be a state, which is allowed in Proposition 4.13. Applying that proposition in ambient dimension r, at rank k and accuracy 2ε, gives

$$
n \geq \frac { c _ { 0 } } { 4 } \frac { r k } { \varepsilon ^ { 2 } } \operatorname* { m a x } \left\{ 1 , \frac { k } { \sqrt { t } } \right\} .\tag{75}
$$

For $r \geq 2$ , we have $k \geq r / 3$ . Since $r > d / 2$

$$
r k \geq { \frac { d r } { 6 } } , \qquad \operatorname* { m a x } \left\{ 1 , { \frac { k } { \sqrt { t } } } \right\} \geq { \frac { 1 } { 3 } } \operatorname* { m a x } \left\{ 1 , { \frac { r } { \sqrt { t } } } \right\} .
$$

Substituting these inequalities into Equation (75) yields

$$
n \geq \frac { c _ { 0 } } { 7 2 } \frac { d r } { \varepsilon ^ { 2 } } \operatorname* { m a x } \left\{ 1 , \frac { r } { \sqrt { t } } \right\} .
$$

Taking $c : = c _ { 0 } / 7 2$ proves Equation (1) in both rank regimes.

## 5 Rank-sensitive tomography

This section proves Theorem 1.2 by constructing and analyzing a nonadaptive tomography protocol based on a Gaussian joint measurement. We begin by defining $\mathsf { M } _ { t }$ , a joint measurement on t samples. In Section 5.1, we give the complete nonadaptive protocol and state the properties of one measurement needed for its analysis. We prove these properties in Section 5.2 and analyze the resulting estimator in Section 5.3.

For each positive integer t, put $\ell _ { \mathrm { m a x } } : = \operatorname* { m i n } \{ d , t \}$ . For $1 \leq \ell \leq \ell _ { \mathrm { m a x } }$ , let $\gamma _ { d , \ell }$ be the standard complex Gaussian law on $\mathbb { C } ^ { d \times \ell }$ , and define

$$
\Gamma _ { \ell , t } : = \int _ { \mathbb { C } ^ { d \times \ell } } ( G G ^ { \dagger } ) ^ { \otimes t } \ \mathrm { d } \gamma _ { d , \ell } ( G ) .\tag{76}
$$

Let $\Pi _ { \ell , t }$ be the orthogonal projector onto the support of $\Gamma _ { \ell , t } ,$ , set $\Pi _ { 0 , t } : = 0$ , and define

$$
\Delta _ { \ell , t } : = \Pi _ { \ell , t } - \Pi _ { \ell - 1 , t } , \qquad 1 \leq \ell \leq \ell _ { \operatorname* { m a x } } .\tag{77}
$$

We prove in Lemma 5.2 that the support projectors $\Pi _ { \ell , t }$ are nested, so every $\Delta _ { \ell , t }$ is a projector. The outcome space consists of pairs $( J , G )$ , where $1 \le J \le \ell _ { \mathrm { m a x } }$ and $G \in \mathbb { C } ^ { d \times J }$ . For $J = { \boldsymbol { \ell } } .$ , define the operator-valued density

$$
\mathsf { M } _ { t } ( \ell , \mathrm { d } G ) : = \Delta _ { \ell , t } ( \Gamma _ { \ell , t } ^ { + } ) ^ { 1 / 2 } ( G G ^ { \dagger } ) ^ { \otimes t } ( \Gamma _ { \ell , t } ^ { + } ) ^ { 1 / 2 } \Delta _ { \ell , t } ~ \mathrm { d } \gamma _ { d , \ell } ( G ) ,\tag{78}
$$

where $\Gamma _ { \ell , t } ^ { + }$ denotes the Moore–Penrose pseudoinverse of $\Gamma _ { \ell , t }$ . The Gaussian measure in this definition is a reference measure. The Born rule determines the outcome density with respect to this measure.

The decomposition and the pseudoinverse factors are chosen so that integrating $\mathsf { M } _ { t } ( \ell , \mathrm { d } G )$ over G gives $\Delta _ { \ell , t }$ . These operators sum to the identity, so the densities $\mathsf { M } _ { t } ( \ell , \mathrm { d } G )$ , for $\ell \in [ \ell _ { \mathrm { m a x } } ]$ and $G \in \mathbb { C } ^ { d \times \ell }$ , form a POVM. We prove this and establish the properties of the POVM in Section 5.2.

## 5.1 The protocol

Recall that the rank of the unknown state is at most r. For a fixed total number of samples, increasing the number t of samples measured jointly improves the resulting asymptotic rate only until t reaches $r ^ { 2 }$ . We therefore set

$$
s : = \operatorname* { m i n } \{ t , r ^ { 2 } \} , \qquad B : = \left\lceil C _ { 0 } \frac { d r } { s \varepsilon ^ { 2 } } \left( 1 + \frac { r } { \sqrt { s } } \right) \right\rceil , \qquad N : = B s ,\tag{79}
$$

where $C _ { 0 }$ is a suficiently large universal constant. We apply $\mathsf { M } _ { s }$ independently B times, using s fresh samples each time. Suppose the b-th outcome is the pair $( J _ { b } , G _ { b } )$ ; here $1 \leq J _ { b } \leq \operatorname* { m i n } \{ d , s \}$ and $G _ { b } \in \mathbb { C } ^ { d \times J _ { b } }$ . Form

$$
Y _ { b } : = \frac { G _ { b } G _ { b } ^ { \dagger } - J _ { b } \mathrm { I } _ { d } } { s } , \qquad \overline { { Y } } : = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } Y _ { b } .\tag{80}
$$

The matrices $Y _ { b }$ are Hermitian, but need not be positive semidefinite or have trace one. Output a density matrix nearest in Frobenius norm to ${ \overline { { Y } } } .$ subject to having rank at most r:

$$
\widehat { \rho } \in \mathop { \mathrm { a r g } } _ { \mathcal { D } _ { r } ( \mathbb { C } ^ { d } ) } \left. \overline { { Y } } - \sigma \right. _ { \mathrm { F } } .\tag{81}
$$

Such a minimizer exists because $\mathcal { D } _ { r } ( \mathbb { C } ^ { d } )$ is compact, and a measurable choice is provided by Lemma B.1. This is the complete protocol. It is nonadaptive; it uses the same joint measurement in every repetition.

We prove the correctness of the protocol in Section 5.3 using the following guarantee for one application of the joint measurement.

Proposition 5.1 (Properties of one joint measurement). For every positive integer t, the densities in Equation (78) form a $P O V M \mathsf { M } _ { t }$ on $( \mathbb { C } ^ { d } ) ^ { \otimes t }$ whose outcome is a pair $( J , G )$ , where $1 \leq J \leq \operatorname* { m i n } \{ d , t \}$ and $G \in \mathbb { C } ^ { d \times J }$ . Associate with this outcome the Hermitian matrix

$$
Y : = \frac { G G ^ { \dagger } - J  { \mathrm { I } _ { d } } } { t } .\tag{82}
$$

For every state $\rho \in \mathcal { D } ( \mathbb { C } ^ { d } )$ , this matrix is exactly unbiased:

$$
\mathbb { E } _ { \rho } [ Y ] = \rho .\tag{83}
$$

Its exact uncentered tensor second moment is

$$
\mathbb { E } _ { \rho } [ Y \otimes Y ] = \frac { t - 1 } { t } \rho ^ { \otimes 2 } + \frac { 1 } { t } ( \rho \otimes  { \mathrm { I } _ { d } } +  { \mathrm { I } _ { d } } \otimes \rho ) F + \frac { \mathbb { E } _ { \rho } [ J ] } { t ^ { 2 } } F ,\tag{84}
$$

where F swaps the two factors $o f \mathbb { C } ^ { d } \otimes \mathbb { C } ^ { d }$ $H \rho$ has rank at most $r ,$ then $\mathbb { E } _ { \rho } [ J ] = O ( \operatorname* { m i n } \{ r , { \sqrt { t } } \} )$ Consequently, for every Hermitian matrix H,

$$
\operatorname { V a r } _ { \rho } \bigl ( \mathrm { T r } ( H Y ) \bigr ) \leq \frac { 2 } { t } \operatorname { T r } ( \rho H ^ { 2 } ) + \frac { \mathbb { E } _ { \rho } [ J ] } { t ^ { 2 } } \left\| H \right\| _ { \mathrm { F } } ^ { 2 } .\tag{85}
$$

Finally, let Π be any orthogonal projector satisfying $\Pi \rho = \rho$ . For every ℓ with $\mathbb { P } _ { \rho } [ J = \ell ] > 0$ conditional on $J = { \boldsymbol { \ell } } _ { : }$ the columns of $\Pi ^ { \bot } G _ { ; }$ , viewed in Im $( \Pi ^ { \perp } )$ , are independent standard complex Gaussian vectors and are independent of ΠG.

## 5.2 Properties of the joint measurement

We now prove Proposition 5.1. We first characterize the support of $\Gamma _ { \ell , t }$ , which is used to define $\mathsf { M } _ { t }$ then bound the mean of J, derive the Gaussian moment identities, and prove the remaining claims.

## 5.2.1 Support of the Gaussian moment operators

To verify that the densities in Equation (78) form a POVM, we first show that the projectors $\Pi _ { \ell , t }$ are nested and that their ranges span the entire tensor product space. It then follows that the diferences $\Delta _ { \ell , t }$ are projectors that sum to the identity. For $1 \leq \ell \leq d ,$ define

$$
\mathcal { V } _ { \ell , t } : = \operatorname { s p a n } \left( \bigcup _ { \stackrel { L \subseteq \mathbb { C } ^ { d } } { \dim L \leq \ell } } L ^ { \otimes t } \right) , \qquad \mathcal { V } _ { 0 , t } : = \{ 0 \} .\tag{86}
$$

Lemma 5.2 (Support of the Gaussian moment operators). For every $1 \leq \ell \leq \ell _ { \mathrm { m a x } }$ , the support $o f \Gamma _ { \ell , t } \ i s \ \mathcal { V } _ { \ell , t }$ . The orthogonal projector $\Pi _ { \ell , t }$ onto $\nu _ { \ell , t }$ commutes with every tensor permutation $U _ { \pi }$ $\pi \in { \mathfrak { S } } _ { t }$ , and with $A ^ { \otimes t }$ for every operator A on $\mathbb { C } ^ { d }$ . Moreover,

$$
0 = \Pi _ { 0 , t } \preceq \Pi _ { 1 , t } \preceq \cdots \preceq \Pi _ { \ell _ { \operatorname* { m a x } } , t } = \mathrm { I } _ { d } ^ { \otimes t } .\tag{87}
$$

Proof. For every $G \in \mathbb { C } ^ { d \times \ell }$ , dim $\operatorname { 1 } ( \operatorname { I m } ( G ) ) \leq \ell ,$ so

$$
\operatorname { I m } ( ( G G ^ { \dag } ) ^ { \otimes t } ) \subseteq \operatorname { I m } ( G ) ^ { \otimes t } \subseteq \mathcal { V } _ { \ell , t } ,
$$

and hence

$$
\operatorname { s u p p } ( \Gamma _ { \ell , t } ) \subseteq \mathcal { V } _ { \ell , t } .\tag{88}
$$

Conversely, suppose $\pmb { v } \in \ker ( \Gamma _ { \ell , t } )$ . Then

$$
0 = \langle { \pmb v } , \Gamma _ { \ell , t } { \pmb v } \rangle = \int \langle { \pmb v } , ( G G ^ { \dagger } ) ^ { \otimes t } { \pmb v } \rangle ~ \mathrm { d } \gamma _ { d , \ell } ( G ) .
$$

The integrand is nonnegative and continuous. Since the Gaussian measure has full support,

$$
\langle { \pmb v } , ( G G ^ { \dagger } ) ^ { \otimes t } { \pmb v } \rangle = 0 \qquad \mathrm { f o r ~ e v e r y ~ } G \in { \mathbb C } ^ { d \times \ell } .
$$

Given a subspace $L \subseteq \mathbb { C } ^ { d }$ with dim $L \ \leq \ \ell .$ , choose G with image L such that $G G ^ { \dagger }$ is positive definite on $L .$ Then $( G G ^ { \dagger } ) ^ { \otimes t }$ is positive definite on $L ^ { \otimes t }$ , so the preceding equality implies $\pm \perp L ^ { \otimes t }$ This holds for every such $L ,$ and therefore $\pmb { v } \in \mathbb { \gamma } _ { \ell , t } ^ { \perp }$ . Together with Equation (88), this proves $\mathrm { s u p p } ( \Gamma _ { \ell , t } ) = \mathcal { V } _ { \ell , t }$

Every tensor permutation $U _ { \pi } , \pi \in \mathfrak { S } _ { t } .$ , preserves each subspace $L ^ { \otimes t }$ , and hence $\nu _ { \ell , t }$ . Also,

$$
A ^ { \otimes t } L ^ { \otimes t } \subseteq ( A L ) ^ { \otimes t } \subseteq \mathcal { V } _ { \ell , t } .
$$

If $\pmb { v } \in \mathcal { V } _ { \ell , t }$ and $\pmb { w } \in \gamma _ { \ell , t } ^ { \perp }$ , the same inclusion with $A ^ { \dagger }$ in place of A gives

$$
\langle { \pmb v } , A ^ { \otimes t } { \pmb w } \rangle = \langle ( A ^ { \dagger } ) ^ { \otimes t } { \pmb v } , { \pmb w } \rangle = 0 .
$$

Thus both $\nu _ { \ell , t }$ and its orthogonal complement are invariant under $A ^ { \otimes t }$ , so $\Pi _ { \ell , t }$ commutes with $A ^ { \otimes t }$ The tensor permutations $U _ { \pi }$ are unitary and preserve $\nu _ { \ell , t } ,$ so the projector $\Pi _ { \ell , t }$ commutes with every $U _ { \pi }$ as well.

Finally, the subspaces $\nu _ { \ell , t }$ are nested and $\mathcal { V } _ { \ell _ { \mathrm { m a x } } , t } = ( \mathbb { C } ^ { d } ) ^ { \otimes t }$ . The latter is immediate when $d \leq t ;$ when $t < d ,$ every tensor product basis vector lies in $L ^ { \otimes t }$ for the span L of its t tensor factors. This proves Equation (87). □

By Equation (87), the operators $\Delta _ { \ell , t }$ in Equation (77) are pairwise orthogonal projectors and sum to $\Gamma _ { d } ^ { \otimes t }$ . We now use this decomposition to establish the validity of the measurement.

Lemma 5.3 (Validity of the joint measurement). The densities in Equation (78) form a POVM on $( \mathbb { C } ^ { d } ) ^ { \otimes t }$

Proof. The operator-valued density is positive semidefinite, and for every $1 \leq \ell \leq \ell _ { \mathrm { m a x } }$

$$
\int \mathsf { M } _ { t } ( \ell , \mathrm { d } G ) = \Delta _ { \ell , t } ( \Gamma _ { \ell , t } ^ { + } ) ^ { 1 / 2 } \Gamma _ { \ell , t } ( \Gamma _ { \ell , t } ^ { + } ) ^ { 1 / 2 } \Delta _ { \ell , t } = \Delta _ { \ell , t } \Pi _ { \ell , t } \Delta _ { \ell , t } = \Delta _ { \ell , t } ,
$$

$$
\sum _ { \ell = 1 } ^ { \ell _ { \mathrm { m a x } } } \int \mathsf { M } _ { t } ( \ell , \mathrm { d } G ) = \sum _ { \ell = 1 } ^ { \ell _ { \mathrm { m a x } } } \Delta _ { \ell , t } = \mathrm { I } _ { d } ^ { \otimes t } .
$$

Integrating over G also gives the distribution of $J { \mathrm { : } }$

$$
\mathbb { P } _ { \rho } [ J = \ell ] = \operatorname { T r } \left( \rho ^ { \otimes t } \int \mathsf { M } _ { t } ( \ell , \mathrm { d } G ) \right) = \operatorname { T r } ( \Delta _ { \ell , t } \rho ^ { \otimes t } ) .\tag{89}
$$

## 5.2.2 Bounding the mean of J

The expected value of J helps bound the second moment of the estimator in Equation (84), so we bound $\mathbb { E } _ { \rho } [ J ]$ next. We use antisymmetrizers to bound the tail of the distribution of J and show that J cannot exceed the rank of the state. For $1 \leq q \leq \ell _ { \mathrm { m a x } }$ , Equation (89) gives

$$
\begin{array} { r } { \mathbb { P } _ { \rho } [ J \geq q ] = \operatorname { T r } \big ( ( \operatorname { I } _ { d } ^ { \otimes t } - \Pi _ { q - 1 , t } ) \rho ^ { \otimes t } \big ) . } \end{array}
$$

We bound $\Gamma _ { d } ^ { \otimes t } - \Pi _ { q - 1 , t }$ by a sum of antisymmetrizers. For $S \subseteq [ t ]$ with $| S | = q$ , let ${ \mathfrak { S } } _ { S }$ be the group of permutations of the positions in S, acting trivially outside S. Define

$$
A _ { S } : = \frac { 1 } { q ! } \sum _ { \pi \in { \mathfrak { S } } _ { S } } \operatorname { s g n } ( \pi ) U _ { \pi } .
$$

This is the orthogonal projector onto tensors that are antisymmetric in the positions in $S .$

Lemma 5.4 (Characterization by antisymmetrizers). For $1 \leq q \leq$ min $\{ d , t \}$

$$
\mathcal { V } _ { q - 1 , t } = \bigcap _ { S \subseteq [ t ] } \ker ( A _ { S } ) .\tag{90}
$$

Proof. For every subspace $L \subseteq \mathbb { C } ^ { d }$ with dim $L \leq q - 1$ , any q vectors in L are linearly dependent, so their antisymmetrization is zero. Therefore,

$$
A _ { S } L ^ { \otimes t } = \{ 0 \} \qquad \mathrm { f o r ~ e v e r y } ~ S \subseteq [ t ] \mathrm { ~ w i t h ~ } | S | = q .
$$

Hence $\begin{array} { r } { \mathcal { V } _ { q - 1 , t } \subseteq \bigcap _ { | S | = q } \ker ( A _ { S } ) } \end{array}$ . Since each $A _ { S }$ is a projector,

$$
\left( \bigcap _ { S \subseteq [ t ] } \ker ( A _ { S } ) \right) ^ { \perp } = \sum _ { S \subseteq [ t ] \atop | S | = q } \ker ( A _ { S } ) ^ { \perp } = \sum _ { S \subseteq [ t ] \atop | S | = q } \operatorname { I m } ( A _ { S } ) .
$$

Thus, taking orthogonal complements, it remains to show

$$
\mathcal { V } _ { q - 1 , t } ^ { \perp } \subseteq \sum _ { S \subseteq [ t ] \atop | S | = q } \operatorname { I m } ( A _ { S } ) .\tag{91}
$$

Let $v \in \mathcal { V } _ { q - 1 , t } ^ { \perp }$ . Let $X = [ { \pmb x } _ { 1 } ~ \cdots ~ { \pmb x } _ { t } ] = ( x _ { i j } )$ be a $d \times t$ matrix of indeterminates, and work in the polynomial ring $R : = \mathbb { C } [ x _ { i j } : i \in [ d ] , j \in [ t ] ]$ . Associate with v the polynomial

$$
F _ { \pmb { v } } ( X ) : = \langle \pmb { v } , \pmb { x } _ { 1 } \otimes \cdot \cdot \cdot \otimes \pmb { x } _ { t } \rangle .
$$

This polynomial is homogeneous of degree one in each column of X. For any choice of the columns in $\mathbb { C } ^ { d }$ , we have

$$
\mathrm { r a n k } ( X ) \leq q - 1 \quad \Longrightarrow \quad x _ { 1 } \otimes \cdots \otimes x _ { t } \in \mathcal { V } _ { q - 1 , t } \quad \Longrightarrow \quad F _ { v } ( X ) = 0 .
$$

For $T \subseteq [ d ]$ and $S \subseteq [ t ]$ with $| T | = | S | = q$ , let $D _ { T , S } ( X )$ be the determinant of the submatrix with rows $T$ and columns S, both in increasing order. Let $I _ { q - 1 } \subset R$ be the ideal generated by these determinants: its elements are sums of the $D _ { T , S }$ multiplied by arbitrary polynomials in R. Its common zero set is

$$
\begin{array} { r l } & { V ( I _ { q - 1 } ) : = \{ M \in \mathbb { C } ^ { d \times t } : p ( M ) = 0 \mathrm { ~ f o r ~ e v e r y ~ } p \in I _ { q - 1 } \} } \\ & { \quad \quad \quad = \{ M \in \mathbb { C } ^ { d \times t } : \operatorname { r a n k } ( M ) \leq q - 1 \} , } \end{array}
$$

since a matrix has rank at most $q - 1$ if and only if all its $q \times q$ minors vanish. This set of matrices is the determinantal variety, while $I _ { q - 1 }$ is the corresponding determinantal ideal of polynomials. The preceding vanishing statement says that $F _ { v }$ vanishes on $V ( I _ { q - 1 } )$ . Since C is algebraically closed, Hilbert’s Nullstellensatz [ZS60, Ch. VII, Theorem 14] gives

$$
F _ { v } ^ { j } \in I _ { q - 1 } \qquad { \mathrm { f o r ~ s o m e ~ i n t e g e r ~ } } j \geq 1 .
$$

The ideal $I _ { q - 1 }$ is prime [BV88, Theorem 2.10 and Remark 2.12]: it is a proper ideal, and whenever a product of two polynomials belongs to it, at least one of the factors belongs to it. Applying this property repeatedly to $F _ { v } ^ { j }$ shows that $F _ { v } \in I _ { q - 1 }$

We can therefore write $F _ { v }$ as a sum of the minors $D _ { T , S }$ times polynomial coeficients. Decompose each coeficient into parts homogeneous in each column, and retain only the part of the sum with degree one in every column. The minor $D _ { T , S }$ already has degree one in each column in S and degree zero in every other column. Thus only the coeficient part with degree zero in the columns in S and degree one in each column outside S contributes. Denoting this part by $H _ { T , S }$ , we obtain

$$
F _ { v } = \sum _ { \stackrel { S \subseteq [ t ] , \ | S | = q } { T \subseteq [ d ] , \ | T | = q } } D _ { T , S } H _ { T , S } .\tag{92}
$$

In particular, $H _ { T , S }$ is independent of the columns in $S .$ . Permuting these columns multiplies $D _ { T , S }$ by the sign of the permutation and leaves $H _ { T , S }$ unchanged. Hence every product $D _ { T , S } H _ { T , S }$ is alternating in the columns indexed by S.

Every polynomial homogeneous of degree one in each column has a unique representation as $F _ { w }$ its coeficients are the complex conjugates of the coordinates of w in the standard tensor product basis. Let $_ { w _ { T , S } }$ be the tensor satisfying

$$
\langle { \pmb w } _ { T , S } , { \pmb x } _ { 1 } \otimes \cdot \cdot \cdot \otimes { \pmb x } _ { t } \rangle = D _ { T , S } ( X ) H _ { T , S } ( X ) .
$$

For $\pi \in { \mathfrak { S } } _ { S }$ , unitarity of $U _ { \pi }$ and alternation give

$$
\begin{array} { r l } & { \langle U _ { \pi } { \pmb w } _ { T , S } , { \pmb x } _ { 1 } \otimes \dots \otimes { \pmb x } _ { t } \rangle = \langle { \pmb w } _ { T , S } , { \pmb x } _ { \pi ( 1 ) } \otimes \dots \otimes { \pmb x } _ { \pi ( t ) } \rangle } \\ & { \qquad = \mathrm { s g n } ( \pi ) \langle { \pmb w } _ { T , S } , { \pmb x } _ { 1 } \otimes \dots \otimes { \pmb x } _ { t } \rangle . } \end{array}
$$

Since simple tensors span $( \mathbb { C } ^ { d } ) ^ { \otimes t }$ and the sign is real, $U _ { \pi } { \pmb w } _ { T , S } = \mathrm { s g n } ( \pi ) { \pmb w } _ { T , S }$ for every $\pi \in { \mathfrak { S } } _ { S }$ Averaging with the signs in $A _ { S }$ gives ${ A _ { S } } { w _ { T , S } } = { { w _ { T , S } } }$ , so ${ \pmb w } _ { T , S } \in \mathrm { I m } ( A _ { S } )$ . Thus Equation (92) and uniqueness of the associated tensor imply $\begin{array} { r } { \pmb { v } = \sum _ { S , T } \pmb { w } _ { T , S } \in \sum _ { | S | = q } \operatorname { I m } ( A _ { S } ) } \end{array}$ , proving Equation (91) and the lemma. □

Lemma 5.5 (Mean of J). For every state $\rho \in \mathcal { D } ( \mathbb { C } ^ { d } )$ and every integer $q \geq 1$

$$
\mathbb { P } _ { \rho } [ J \ge q ] \le \operatorname* { m i n } \left\{ 1 , \frac { \binom { t } { q } } { q ! } \right\} ,\tag{93}
$$

where the probability is zero for $q >$ min $\{ d , t \}$ . Consequently,

$$
\mathbb { E } _ { \rho } [ J ] = O ( { \sqrt { t } } ) .\tag{94}
$$

If rank $( \rho ) \leq r$ , then every value of J with positive probability is at most $r .$

Proof. Fix $1 \leq q \leq$ min $\{ d , t \}$ . By the preceding lemma, the intersection of the kernels of the order-q antisymmetrizers is $\gamma _ { q - 1 , t }$ . Define

$$
\ A _ { t , q } : = \sum _ { \stackrel { S \subseteq [ t ] } { | S | = q } } A _ { S } .
$$

Conjugating $\mathbf { \mathcal { A } } _ { t , q }$ by a tensor permutation $U _ { \pi }$ permutes the summands, so $\mathbf { \mathcal { A } } _ { t , q }$ commutes with every $U _ { \pi } , \pi \in \mathfrak { S } _ { t } .$ . Since each $A _ { S }$ is a linear combination of such permutations, $\mathcal { A } _ { t , q }$ commutes with every $A _ { S }$ . Therefore,

$$
A _ { t , q } ^ { 2 } = \sum _ { S } A _ { S } A _ { t , q } = \sum _ { S } A _ { S } A _ { t , q } A _ { S } = \sum _ { S , T } A _ { S } A _ { T } A _ { S } \succeq \sum _ { S } A _ { S } = { \mathcal A } _ { t , q } .\tag{95}
$$

Here $A _ { S } A _ { T } A _ { S } = ( A _ { T } A _ { S } ) ^ { \dagger } ( A _ { T } A _ { S } ) \succeq 0$ , and the term with $T = S$ equals $A _ { S }$ . Moreover,

$$
\ker ( \mathcal A _ { t , q } ) = \bigcap _ { | S | = q } \ker ( A _ { S } ) = \mathcal V _ { q - 1 , t } .
$$

Since $\mathbf { \mathcal { A } } _ { t , q } \succeq 0$ , Equation (95) implies that each of its nonzero eigenvalues is at least one. Hence

$$
\mathcal { A } _ { t , q } \succeq \Gamma _ { d } ^ { \otimes t } - \Pi _ { q - 1 , t } .\tag{96}
$$

Since

$$
\sum _ { \ell = q } ^ { \ell _ { \mathrm { m a x } } } \Delta _ { \ell , t } = \mathrm { I } _ { d } ^ { \otimes t } - \Pi _ { q - 1 , t } ,
$$

the tail of J is controlled by Equation (96). Define the antisymmetrizer on $( \mathbb { C } ^ { d } ) ^ { \otimes q }$ by

$$
A _ { q } : = \frac { 1 } { q ! } \sum _ { \pi \in { \mathfrak { S } } _ { q } } \operatorname { s g n } ( \pi ) U _ { \pi } .
$$

For every $S \subseteq [ t ]$ of size $q ,$

$$
\operatorname { T r } ( A _ { S } \rho ^ { \otimes t } ) = \operatorname { T r } ( A _ { q } \rho ^ { \otimes q } ) .
$$

Therefore, Equations (89) and (96) give

$$
\begin{array} { r l } {  { \mathbb { P } _ { \rho } [ J \geq q ] = \sum _ { \ell = q } ^ { \ell _ { \operatorname* { m a x } } } \operatorname { T r } ( \Delta _ { \ell , t } \rho ^ { \otimes t } ) } } \\ & { = \operatorname { T r } ( ( \mathrm { I } _ { d } ^ { \otimes t } - \Pi _ { q - 1 , t } ) \rho ^ { \otimes t } ) } \\ & { \leq \operatorname { T r } ( \mathcal { A } _ { t , q } \rho ^ { \otimes t } ) = \binom { t } { q } \operatorname { T r } ( \varLambda _ { q } \rho ^ { \otimes q } ) . } \end{array}\tag{97}
$$

Let $\lambda _ { 1 } , \ldots , \lambda _ { d }$ be the eigenvalues of $\rho .$ Since $A _ { q }$ projects onto the totally antisymmetric subspace, diagonalizing $\rho$ gives an eigenbasis for the restriction of $\rho ^ { \otimes q }$ to this subspace, indexed by $1 \leq i _ { 1 } <$ $\cdots < i _ { q } \leq d _ { \because }$ , with corresponding eigenvalue $\lambda _ { i _ { 1 } } \cdots \lambda _ { i _ { q } }$ . Hence

$$
\mathrm { T r } ( A _ { q } \rho ^ { \otimes q } ) = \sum _ { 1 \leq i _ { 1 } < \cdots < i _ { q } \leq d } \lambda _ { i _ { 1 } } \cdot \cdot \cdot \cdot \lambda _ { i _ { q } } \leq \frac { 1 } { q ! } \left( \sum _ { i = 1 } ^ { d } \lambda _ { i } \right) ^ { q } = \frac { 1 } { q ! } .
$$

The inequality holds because, in the expansion of $( \sum _ { i } \lambda _ { i } ) ^ { q }$ , every product with $q$ distinct indices appears q! times, and all remaining terms are nonnegative. Together with Equation (97), this proves Equation (93).

The standard estimates ${ \bf \Pi } ( { } _ { q } ^ { t } ) \le ( \mathrm { e } t / q ) ^ { q }$ and $q ! \geq ( q / \mathrm { e } ) ^ { q }$ give

$$
\frac { { \binom { t } { q } } } { q ! } \leq \left( \frac { \mathrm { e } ^ { 2 } t } { q ^ { 2 } } \right) ^ { q } .
$$

For $q \geq 2 \mathrm { e } \sqrt { t }$ , the right side is at most $4 ^ { - q }$ . The tail sum identity $\begin{array} { r } { \mathbb { E } _ { \rho } [ J ] = \sum _ { q \geq 1 } \mathbb { P } _ { \rho } [ J \geq q ] } \end{array}$ therefore gives

$$
\mathbb { E } _ { \rho } [ J ] \leq \left\lceil 2 \mathrm { e } \sqrt { t } \right\rceil + \sum _ { q \geq \lceil 2 \mathrm { e } \sqrt { t } \rceil } 4 ^ { - q } = O ( \sqrt { t } ) ,
$$

proving Equation (94).

Finally, let L be the support of a state $\rho$ of rank at most r. If $r < \ell _ { \mathrm { m a x } }$ , then

$$
\mathrm { s u p p } ( \rho ^ { \otimes t } ) \subseteq L ^ { \otimes t } \subseteq \mathcal { V } _ { r , t } , \qquad \mathrm { a n d } \qquad \Delta _ { \ell , t } \Pi _ { r , t } = 0 \quad \mathrm { w h e n ~ } \ell > r .
$$

So Equation (89) gives $\mathbb { P } _ { \rho } [ J = \ell ] = 0$ for $\ell > r$ . If $r \geq \ell _ { \mathrm { m a x } }$ , this follows directly from $J \le \ell _ { \mathrm { m a x } } \le$ r. □

## 5.2.3 The outcome law and its moments

It remains to prove the claims about the moments and conditional distribution in Proposition 5.1. We first derive the Gaussian moment identities used to evaluate the first and second moments. We then apply the Born rule to obtain the contribution from each event $J = { \boldsymbol { \ell } } .$ , sum these contributions over $\ell ,$ and finally prove the claimed conditional law of $\Pi ^ { \perp } G$

The Gaussian moment identity. For $\pi \in \mathfrak { S } _ { t }$ , let $c ( \pi )$ be the number of cycles of $\pi ,$ including fixed points.

Lemma 5.6 (Gaussian moment identity). For every $1 \leq \ell \leq \ell _ { \mathrm { m a x } }$ ,

$$
\Gamma _ { \ell , t } = \sum _ { \pi \in \mathfrak { S } _ { t } } \ell ^ { c ( \pi ) } U _ { \pi } .\tag{98}
$$

The operators $( \Gamma _ { \ell , t } : 1 \le \ell \le \ell _ { \mathrm { m a x } } )$ commute pairwise. Each $\Gamma _ { \ell , t }$ also commutes with every tensor permutation $U _ { \pi } , \pi \in { \mathfrak { S } } _ { t }$ , and with $A ^ { \otimes t }$ for every operator A on $\mathbb { C } ^ { d }$

Proof. We compute an arbitrary matrix entry of $\Gamma _ { \ell , t }$ . Let $e _ { 1 } , \ldots , e _ { d }$ be the standard basis of $\mathbb { C } ^ { d } .$ For $\mathbf { i } = ( i _ { 1 } , \dots , i _ { t } )$ and ${ \bf j } = ( j _ { 1 } , \dots , j _ { t } )$ in $[ d ] ^ { t }$ , put

$$
\begin{array} { r l r l } & { e _ { \mathbf { i } } : = e _ { i _ { 1 } } \otimes \cdot \cdot \cdot \otimes e _ { i _ { t } } , } & & { e _ { \mathbf { j } } : = e _ { j _ { 1 } } \otimes \cdot \cdot \cdot \otimes e _ { j _ { t } } . } \end{array}
$$

The entries of G satisfy

$$
\begin{array} { r } { \mathbb { E } [ G _ { i \alpha } G _ { j \beta } ] = 0 , \qquad \mathbb { E } [ G _ { i \alpha } \overline { { G _ { j \beta } } } ] = \delta _ { i j } \delta _ { \alpha \beta } . } \end{array}
$$

These identities hold for all $i , j \in [ d ]$ and $\alpha , \beta \in [ \ell ]$ . Expanding the matrix entry and applying the complex form of Isserlis’s Gaussian pairing formula [Iss18] gives the following calculation. The formula applies to the jointly Gaussian entries even when some indices coincide. Pairings between two unconjugated entries or two conjugated entries vanish, so every surviving pairing matches the t unconjugated factors with the t conjugated factors. The matching is given by a permutation in ${ \mathfrak { S } } _ { t }$

$$
\begin{array} { r l } { \langle e _ { \parallel } , \Gamma _ { i } ( e _ { \parallel } ^ { * } ) \rangle = } & { \displaystyle \left[ \prod _ { u = 1 } ^ { t } ( G G ^ { ( t ) } ) _ { u _ { \parallel } , } \right] } \\ { = } & { \displaystyle \sum _ { \alpha _ { 1 } = \alpha \in \mathbb { R } ^ { * } [ \ell ] } \mathbb { E } \left[ \prod _ { u = 1 } ^ { t } G _ { i , \alpha _ { 1 } } G _ { i , \alpha _ { 2 } } ^ { - } \right] } \\ { = } & { \displaystyle \sum _ { \alpha _ { 1 } = \alpha \in \mathbb { R } ^ { * } [ \ell ] } \sum _ { u = 1 } \prod _ { u = 1 } ^ { T } \mathbb { E } \left[ G _ { i \nu _ { \alpha _ { 1 } } , \alpha _ { 1 } } \overline { { G _ { i , \alpha _ { 1 } } \alpha _ { \alpha _ { 1 } } } } \right] } \\ { = } & { \displaystyle \sum _ { \alpha _ { 1 } = \alpha \in \mathbb { R } ^ { * } [ \ell ] } \sum _ { u \in \mathbb { R } ^ { * } [ \ell ] } \mathbb { E } \left[ G _ { i \nu _ { \alpha _ { 1 } } , \alpha _ { 1 } } \overline { { G _ { i , \alpha _ { 1 } } \alpha _ { 1 } } } \right] } \\ { = } & { \displaystyle \sum _ { \alpha _ { 1 } = \alpha \in \mathbb { R } ^ { * } [ \ell ] } \sum _ { u \in \mathbb { R } ^ { * } [ \ell ] } \sum _ { u > i \nu _ { \alpha _ { 1 } } = 1 } ^ { T } \hat { \sigma } _ { \alpha _ { 1 } , \alpha _ { 2 } ( \alpha _ { 1 } ) } } \\ { = } & { \displaystyle \sum _ { \alpha \in \mathbb { R } ^ { * } [ \ell ] } \sum _ { u = 1 } ^ { T } G _ { i \nu _ { \alpha _ { 1 } } , \alpha _ { 1 } } \langle \sum _ { u = 1 } ^ { T } G _ { i \nu _ { \alpha _ { 1 } } , \alpha _ { 1 } } \rangle } \end{array} \quad \mathrm { ( L s c r l i s ~ t h e o r e m ) }\tag{99}
$$

since the constraints $\alpha _ { u } = \alpha _ { \pi ( u ) }$ make the column indices constant on each cycle of π, giving $\ell ^ { c ( \pi ) }$ choices. Our convention for tensor permutations gives

$$
\langle e _ { \mathbf { i } } , U _ { \pi } e _ { \mathbf { j } } \rangle = \prod _ { u = 1 } ^ { t } \delta _ { i _ { u } , j _ { \pi ^ { - 1 } ( u ) } } .
$$

Since inversion preserves the number of cycles and permutes ${ \mathfrak { S } } _ { t }$

$$
\langle e _ { \mathbf { i } } , \left( \sum _ { \pi \in \mathfrak { S } _ { t } } \ell ^ { c ( \pi ) } U _ { \pi } \right) e _ { \mathbf { j } } \rangle = \sum _ { \pi \in \mathfrak { S } _ { t } } \ell ^ { c ( \pi ) } \prod _ { u = 1 } ^ { t } \delta _ { i _ { u } , j _ { \pi ^ { - 1 } ( u ) } } = \sum _ { \pi \in \mathfrak { S } _ { t } } \ell ^ { c ( \pi ) } \prod _ { u = 1 } ^ { t } \delta _ { i _ { u } , j _ { \pi ( u ) } } .
$$

Comparing this with Equation (99) for every $\mathbf { i } , \mathbf { j } \in [ d ] ^ { t }$ proves Equation (98).

We next use this identity to prove the commutation claims. For every $\tau \in \mathfrak { S } _ { t }$ , conjugation preserves cycle type and gives

$$
U _ { \tau } \Gamma _ { \ell , t } U _ { \tau } ^ { \dagger } = \sum _ { \pi \in \mathfrak { S } _ { t } } \ell ^ { c ( \pi ) } U _ { \tau \pi \tau ^ { - 1 } } = \Gamma _ { \ell , t } .
$$

Thus $\Gamma _ { \ell , t }$ commutes with every $U _ { \tau }$ . Because every $\Gamma _ { \ell ^ { \prime } , t }$ is a linear combination of the $U _ { \tau }$ , the operators $\Gamma _ { \ell , t }$ commute pairwise for diferent values of ℓ. Finally,

$$
U _ { \pi } A ^ { \otimes t } = A ^ { \otimes t } U _ { \pi } \quad \Longrightarrow \quad \Gamma _ { \ell , t } A ^ { \otimes t } = A ^ { \otimes t } \Gamma _ { \ell , t } .
$$

Centered Gaussian moments. For a fixed value $J = \ell ,$ the contribution to the first moment of the estimator involves Gaussian integrals of

$$
\frac { 1 } { t } ( G G ^ { \dagger } - \ell \mathrm { I } _ { d } ) \otimes ( G G ^ { \dagger } ) ^ { \otimes t } .
$$

To separate the estimator matrix from the operators acting on the t measured samples, we temporarily work on the labelled tensor product space

$$
\mathbb { C } _ { a } ^ { d } \otimes \mathbb { C } _ { 1 } ^ { d } \otimes \cdot \cdot \cdot \otimes \mathbb { C } _ { t } ^ { d } ,
$$

where the subscripts label the tensor factors. The factors $1 , \ldots , t$ correspond to the measured samples, while $a$ is an auxiliary factor carrying the matrix output of the estimator. Thus, if B is an output matrix and $A , C$ act on the t input factors, then

$$
B \operatorname { T r } ( A C ) = \operatorname { T r } _ { [ t ] } { \big ( } ( \operatorname { I } _ { a } \otimes A ) ( B _ { a } \otimes C ) { \big ) } .
$$

Here $B _ { a }$ means that B acts on factor $^ { a , }$ and $\operatorname { T r } _ { [ t ] }$ traces out factors $1 , \ldots , t .$ After factoring out $1 / t$ taking $B = G G ^ { \dagger } - \ell \mathrm { I } _ { d }$ and $C = ( G G ^ { \dagger } ) ^ { \otimes t }$ gives the first Gaussian integral below. For the second moment, we use two auxiliary “output” tensor factors a and $b ,$ one for each occurrence of the estimator matrix in its tensor second moment. Write $F _ { u v }$ for the operator that swaps tensor factors u and v and acts as the identity on the remaining factors, and $\mathrm { I } _ { a b }$ for the identity on the two output factors.

Lemma 5.7 (Centered Gaussian moments). Fix $1 \leq \ell \leq \ell _ { \mathrm { m a x } }$ . Then

$$
\int ( G G ^ { \dagger } - \ell \mathrm { I } _ { d } ) _ { a } \otimes ( G G ^ { \dagger } ) ^ { \otimes t } ~ \mathrm { d } \gamma _ { d , \ell } ( G ) = \sum _ { i = 1 } ^ { t } F _ { a i } ( \mathrm { I } _ { a } \otimes \Gamma _ { \ell , t } ) .\tag{100}
$$

With two output factors,

$$
\begin{array} { l } { { \displaystyle \int ( G G ^ { \dagger } - \ell \mathbf { I } _ { d } ) _ { a } \otimes ( G G ^ { \dagger } - \ell \mathbf { I } _ { d } ) _ { b } \otimes ( G G ^ { \dagger } ) ^ { \otimes t } \mathrm { d } \gamma _ { d , \ell } ( G ) } } \\ { { \displaystyle \quad = \sum _ { \stackrel { j , j \in [ t ] } { i \neq j } } F _ { a i } F _ { b j } ( \mathrm { I } _ { a b } \otimes \Gamma _ { \ell , t } ) + \sum _ { i = 1 } ^ { t } \bigl ( F _ { a b } F _ { b i } + F _ { a b } F _ { a i } \bigr ) ( \mathrm { I } _ { a b } \otimes \Gamma _ { \ell , t } ) + \ell F _ { a b } ( \mathrm { I } _ { a b } \otimes \Gamma _ { \ell , t } ) . } } \end{array}\tag{101}
$$

When $t = 1$ , the first sum in Equation (101) is empty and equals zero.

Proof. For $\pi \in \mathfrak { S } _ { t }$ , let $\widetilde { \pi }$ be its extension that fixes a, and write ${ \mathfrak { S } } _ { \{ a \} \cup [ t ] }$ for the permutations of the labels $a , 1 , \ldots , t$ . Applying the Gaussian moment identity in Equation (98) with the output position included gives

$$
\begin{array} { r l } & { \int ( G G ^ { 1 } - \delta _ { d \delta } ) _ { \alpha } \otimes ( G G ^ { 1 } ) ^ { \otimes d } \dag \mathcal { H } _ { \alpha , \delta } ( G ) } \\ & { \quad = \int ( G G ^ { 1 } ) _ { \alpha } \otimes ( G G ^ { 1 } ) ^ { \otimes d } \dag \Pi _ { \alpha , \delta } ( G ) - \bar { H } _ { \alpha } \otimes \int ( G G ^ { 1 } ) ^ { \otimes d } \dag \mathcal { H } _ { \alpha , \delta } ( G ) } \\ & { \quad = \begin{array} { r } { G ^ { \mathrm { e v } } G ^ { ( e ) } G _ { \sigma } - \sum _ { w \in \mathfrak { s } } e ^ { \zeta ( w ) + 1 } ( I _ { \alpha } \otimes \bar { U } _ { x } ) } \\ { \quad _ { \sigma \in \mathfrak { s } \in \mathfrak { s } _ { \alpha , ( \delta \delta ) } ( q ) } } \\ { \quad = \displaystyle \sum _ { \sigma \in \mathfrak { s } _ { \alpha , ( \delta ) } ( q ) } e ^ { \zeta ( w ) } \bar { U } _ { \sigma } } \end{array} } \\ & { \quad \quad = \begin{array} { r } { \sum _ { \sigma \in \mathfrak { s } } e ^ { \zeta ( w ) } \bar { U } _ { \sigma } } \\ { \quad _ { \sigma \in \mathfrak { s } } \sum _ { w \in \mathfrak { s } } e ^ { \zeta ( w ) } \bar { U } _ { \sigma } ( \mathfrak { L } _ { w } \otimes \mathfrak { L } _ { w } ) } \end{array} } \\ & { \quad = \begin{array} { r } { \sum _ { i = 1 } ^ { n } \sum _ { w \in \mathfrak { s } } e ^ { \zeta ( w ) } E _ { \sigma u } ( \mathfrak { L } _ { w } \otimes \mathfrak { L } _ { w } ) } \\ { \quad _ { \sigma \in \mathfrak { s } } \sum _ { \ell \in \mathfrak { s } } e ^ { \zeta ( w ) } \bar { \mathfrak { L } } _ { \sigma } ( \mathfrak { L } _ { w } \otimes \mathfrak { L } _ { w } ) } \end{array} } \\ &  \quad \quad = \begin{array} { r }  \sum _ { i = 1 } ^ { n } E _ { \sigma } ( \mathrm { L } _ { w } \otimes \Gamma \end{array} \end{array}
$$

where the third equality cancels exactly the permutations that fix a: each is the extension $\widetilde { \pi }$ of a unique $\pi \in \mathfrak { S } _ { t } .$ , with $c ( \widetilde { \pi } ) = c ( \pi ) + 1$ and $U _ { \widetilde { \pi } } = \mathrm { I } _ { a } \otimes U _ { \pi }$ . For the fourth equality, if $\sigma ( a ) \neq a ,$ , set $i = \sigma ( a )$ . Then $( a i ) \sigma$ fixes $a , \mathrm { g i v i n g }$ the unique representation $\sigma = ( a i ) \widetilde { \pi }$ . This inserts a immediately before i in its cycle of $\pi ,$ so $c ( \sigma ) = c ( \pi )$ and $U _ { \sigma } = F _ { a i } ( \mathrm { I } _ { a } \otimes U _ { \pi } )$ . This proves Equation (100).

With two outputs, expanding both centered factors and applying the same Gaussian moment identity gives

$$
\begin{array} { r l } & { \int ( G G ^ { \dagger } - \delta _ { I , i } ) _ { \alpha } \otimes ( G G ^ { \dagger } - \delta _ { I , i } ) _ { \beta } \otimes ( G G ^ { \dagger } ) ^ { \otimes \dagger } \mathrm { d } \gamma _ { \alpha , \beta } ( G ) } \\ & { \quad = \int ( G G ^ { \dagger } ) _ { \beta } \otimes ( G G ^ { \dagger } ) _ { \beta } \otimes ( G G ^ { \dagger } ) ^ { \otimes \dagger } \mathrm { d } \gamma _ { \alpha , \beta } ( G ) } \\ & { \quad \quad - \ : \ell \int _ { 1 } \otimes ( G G ^ { \dagger } ) _ { \beta } \otimes ( G G ^ { \dagger } ) ^ { \otimes \dagger } \mathrm { d } \gamma _ { \alpha , \beta } ( G ) - \ : \ell \int ( G G ^ { \dagger } ) _ { \alpha } \otimes \mathrm { I } _ { \beta } \otimes ( G G ^ { \dagger } ) ^ { \otimes \dagger } \mathrm { d } \gamma _ { \alpha , \beta } ( G ) } \\ & { \quad \quad + \ : \ell ^ { 2 } \mathrm { I } _ { \alpha \otimes } \otimes \int ( G G ^ { \dagger } ) ^ { \otimes \dagger } \mathrm { d } \gamma _ { \alpha , \beta } ( G ) } \\ & { \quad = \quad \displaystyle \sum _ { \sigma \in \mathbb { S } _ { \alpha , \beta + 0 } \setminus \{ 0 \} \atop \sigma ( | \alpha | = 0 ) } \ : E ^ { \langle \sigma | \alpha \rangle } U _ { \sigma } - \ : \sum _ { \sigma \in \mathbb { S } _ { \alpha , \beta + 0 } \atop \sigma ( | \alpha | = 0 ) } \ : \ell ^ { \langle \alpha \rangle } U _ { \sigma } + \sum _ { \sigma \in \mathbb { S } _ { \alpha , \beta } \backslash \{ i \} } \ : \ell ^ { \langle \sigma \rangle } U _ { \sigma } } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \sigma ( | \alpha | = \sigma ) \ : \mathrm { d } \sigma \ : \mathrm { I } _ { \alpha } } \\ &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \sigma ( | \ \end{array}
$$

Deleting a and b from the cycles of a permutation σ in the remaining sum leaves a unique permutation $\pi \in \mathfrak { S } _ { t }$ . To recover $\sigma ,$ there are four possibilities: insert them before two distinct input labels $i \neq j ;$ insert them consecutively before one input label i, in either order; or place them together in the two-cycle (a b). Here inserting a before i replaces an arrow $k  i$ in a cycle of π by $k  a  i$

Writing πe for the extension of π that fixes a and b, these possibilities are listed below.

<table><tr><td>configuration</td><td>σ</td><td> $c ( \sigma )$ </td><td> $U _ { \sigma }$ </td></tr><tr><td>distinct positions  $i \neq j$ </td><td> $\overline { { ( a \ i ) ( b \ j ) \tilde { \pi } } }$ </td><td> $c ( \pi )$ </td><td> $\overline { { { F _ { a i } F _ { b j } ( \mathrm { I } _ { a b } \otimes U _ { \pi } ) } } }$ </td></tr><tr><td> $a  b  i$ </td><td> $( a \ b ) ( b \ i ) { \widetilde { \pi } }$ </td><td> $c ( \pi )$ </td><td> $F _ { a b } F _ { b i } ( \mathrm { I } _ { a b } \otimes U _ { \pi } )$ </td></tr><tr><td> $b  a  i$ </td><td> $( a \ b ) ( a \ i ) \widetilde \pi$ </td><td> $c ( \pi )$ </td><td> $F _ { a b } F _ { a i } ( \mathrm { I } _ { a b } \otimes U _ { \pi } )$ </td></tr><tr><td>isolated two-cycle  $( a \ b )$ </td><td> $( a \ b ) \widetilde { \pi }$ </td><td> $c ( \pi ) + 1$ </td><td> $F _ { a b } ( \mathrm { I } _ { a b } \otimes U _ { \pi } )$ </td></tr></table>

Summing over $\pi \in \mathfrak { S } _ { t }$ in each row gives

$$
\begin{array} { r l } & { \displaystyle \sum _ { \sigma \in \mathfrak { S } _ { \{ a , b \} \cup [ t ] } } \ell ^ { c ( \sigma ) } U _ { \sigma } } \\ & { \quad = \displaystyle \sum _ { \substack { i , j \in [ t ] } } F _ { a i } F _ { b j } ( \mathrm { I } _ { a b } \otimes \Gamma _ { \ell , t } ) + \displaystyle \sum _ { i = 1 } ^ { t } ( F _ { a b } F _ { b i } + F _ { a b } F _ { a i } ) ( \mathrm { I } _ { a b } \otimes \Gamma _ { \ell , t } ) + \ell F _ { a b } ( \mathrm { I } _ { a b } \otimes \Gamma _ { \ell , t } ) , } \\ & { \quad \quad = \displaystyle \sum _ { \substack { i , j \in [ t ] } } F _ { a i } F _ { b j } ( \mathrm { I } _ { a b } \otimes \Gamma _ { \ell , t } ) + \displaystyle \sum _ { i = 1 } ^ { t } ( F _ { a b } F _ { b i } + F _ { a b } F _ { a i } ) ( \mathrm { I } _ { a b } \otimes \Gamma _ { \ell , t } ) + \ell F _ { a b } ( \mathrm { I } _ { a b } \otimes \Gamma _ { \ell , t } ) , } \end{array}
$$

which proves Equation (101).

When $t = 1$ , there are no distinct input labels $i \neq j$ . The only permutations of $\{ a , b , 1 \}$ that fix neither a nor b are $( a \ b \ 1 )$ , (a 1 b), and $( a \ b )$ . Since $\ell _ { \mathrm { m a x } } = 1$ and $\Gamma _ { 1 , 1 } = \mathrm { I } _ { d } ,$ their contributions give

$$
\int ( G G ^ { \dagger } - \mathrm { I } _ { d } ) _ { a } \otimes ( G G ^ { \dagger } - \mathrm { I } _ { d } ) _ { b } \otimes ( G G ^ { \dagger } ) _ { 1 } { \mathrm { ~ d } } \gamma _ { d , 1 } ( G ) = F _ { a b } F _ { b 1 } + F _ { a b } F _ { a 1 } + F _ { a b } .
$$

Thus only the last two terms on the right side of Equation (101) remain.

The contribution from a fixed value of J. We now use the preceding Gaussian identities to compute the contribution of the event $J = \ell$ to the moments of the estimator. Fix a state $\rho .$ For $1 \leq \ell \leq \ell _ { \mathrm { m a x } }$ , define

$$
\begin{array} { r } { \omega _ { \ell , t } ( { \rho } ) : = \Delta _ { \ell , t } { \rho } ^ { \otimes t } , \qquad \Theta _ { \ell , { \rho } } : = ( \Gamma _ { \ell , t } ^ { + } ) ^ { 1 / 2 } \omega _ { \ell , t } ( { \rho } ) ( \Gamma _ { \ell , t } ^ { + } ) ^ { 1 / 2 } . } \end{array}
$$

By Lemma 5.2, $\Delta _ { \ell , t }$ commutes with $\rho ^ { \otimes t }$ , so $\omega _ { \ell , t } ( \rho )$ is the unnormalized positive semidefinite component of $\rho ^ { \otimes t }$ on $\mathrm { I m } ( \Delta _ { \ell , t } )$ . Its trace is

$$
\begin{array} { r } { \operatorname { T r } \big ( \omega _ { \ell , t } ( \rho ) \big ) = \operatorname { T r } \big ( \Delta _ { \ell , t } \rho ^ { \otimes t } \big ) = \mathbb { P } _ { \rho } [ J = \ell ] , } \end{array}
$$

by Equation (89). The operator $\Theta _ { \ell , \rho }$ incorporates the pseudoinverse factors from the POVM so that the density of outcomes with $J = { \boldsymbol { \ell } } ,$ relative to $\gamma _ { d , \ell } ,$ , is $\operatorname { T r } ( \Theta _ { \ell , \rho } ( G G ^ { \dagger } ) ^ { \otimes t } )$ ).

Lemma 5.8 (Contribution from a fixed value of $J )$ . Fix a state ρ and $1 \leq \ell \leq \ell _ { \mathrm { m a x } }$ . On the event $J = { \boldsymbol { \ell } } ,$ we have $Y = ( G G ^ { \dagger } - \ell \mathrm { I } _ { d } ) / t . \ I f \ \mathbf { 1 } _ { \{ J = \ell \} }$ denotes the indicator of this event, then

$$
\mathbb { E } _ { \rho } \left[ Y \mathbf { 1 } _ { \{ J = \ell \} } \right] = \operatorname { T r } _ { [ t ] \backslash \{ 1 \} } \left( \omega _ { \ell , t } ( \rho ) \right)\tag{102}
$$

and

$$
\begin{array} { l } { { \displaystyle \mathbb { E } _ { \rho } \left[ ( Y \otimes Y ) \mathbf { 1 } _ { \{ J = \ell \} } \right] = \frac { t - 1 } { t } \operatorname { T r } _ { [ t ] \setminus \{ 1 , 2 \} } \left( \omega _ { \ell , t } ( \rho ) \right) } } \\ { ~ + \frac { 1 } { t } \left( \operatorname { T r } _ { [ t ] \setminus \{ 1 \} } \left( \omega _ { \ell , t } ( \rho ) \right) \otimes \mathrm { I } _ { d } + \mathrm { I } _ { d } \otimes \operatorname { T r } _ { [ t ] \setminus \{ 1 \} } \left( \omega _ { \ell , t } ( \rho ) \right) \right) F } \\ { ~ + \frac { \ell } { t ^ { 2 } } \mathbb { P } _ { \rho } [ J = \ell ] F , } \end{array}\tag{103}
$$

where F is the swap operator on $\mathbb { C } ^ { d } \otimes \mathbb { C } ^ { d }$

Proof. The outcome law on the event $J = \ell$ has density

$$
\begin{array} { r l } & { p _ { \ell , \rho } ( G ) : = \operatorname { T r } \left( \rho ^ { \otimes t } \Delta _ { \ell , t } ( \Gamma _ { \ell , t } ^ { + } ) ^ { 1 / 2 } ( G G ^ { \dagger } ) ^ { \otimes t } ( \Gamma _ { \ell , t } ^ { + } ) ^ { 1 / 2 } \Delta _ { \ell , t } \right) } \\ & { \qquad = \operatorname { T r } \left( \omega _ { \ell , t } ( \rho ) ( \Gamma _ { \ell , t } ^ { + } ) ^ { 1 / 2 } ( G G ^ { \dagger } ) ^ { \otimes t } ( \Gamma _ { \ell , t } ^ { + } ) ^ { 1 / 2 } \right) } \\ & { \qquad = \operatorname { T r } \left( \Theta _ { \ell , \rho } ( G G ^ { \dagger } ) ^ { \otimes t } \right) } \end{array}\tag{104}
$$

relative to $\gamma _ { d , \ell }$ . This density has total mass $\mathbb { P } _ { \rho } [ J = \ell ]$ ; dividing by this probability, when positive, gives the conditional density of $G$

Combining Equations (100) and (104), we rewrite the contribution to the first moment as

$$
\mathbb { E } _ { \rho } \left[ Y \mathbf { 1 } _ { \{ J = \ell \} } \right] = \frac { 1 } { t } \sum _ { i = 1 } ^ { t } \mathrm { T r } _ { [ t ] } \left( (  { \mathrm { I } _ { a } } \otimes \boldsymbol { \Theta } _ { \ell , \rho } ) F _ { a i } (  { \mathrm { I } _ { a } } \otimes \boldsymbol { \Gamma } _ { \ell , t } ) \right) .
$$

To simplify each summand, cyclicity on the traced-out input factors gives

$$
\mathrm { T r } _ { [ t ] } \left( \left(  { \mathrm { I } _ { a } } \otimes \boldsymbol { \Theta } _ { \ell , \rho } \right) \boldsymbol { F } _ { a i } (  { \mathrm { I } _ { a } } \otimes  { \Gamma } _ { \ell , t } ) \right) = \mathrm { T r } _ { [ t ] } \left( \boldsymbol { F } _ { a i } (  { \mathrm { I } _ { a } } \otimes  { \Gamma } _ { \ell , t } \boldsymbol { \Theta } _ { \ell , \rho } ) \right) .
$$

Thus we need to evaluate $\Gamma _ { \ell , t } \Theta _ { \ell , \rho }$ . By Lemma 5.6, $\Gamma _ { \ell , t }$ commutes with $\rho ^ { \otimes t }$ and, when $\ell > 1$ , with $\Gamma _ { \ell - 1 , t }$ , hence with its support projector $\Pi _ { \ell - 1 , t }$ . The case $\ell = 1$ is immediate because $\Pi _ { 0 , t } = 0$ . It also commutes with its own support projector $\Pi _ { \ell , t }$ . Thus $\Gamma _ { \ell , t }$ and $( \Gamma _ { \ell , t } ^ { + } ) ^ { 1 / 2 }$ commute with $\Delta _ { \ell , t }$ and $\omega _ { \ell , t } ( \rho )$ . Moreover, $\Delta _ { \ell , t } \preceq \Pi _ { \ell , t }$ . Hence

$$
\Gamma _ { \ell , t } \Theta _ { \ell , \rho } = \Gamma _ { \ell , t } \Gamma _ { \ell , t } ^ { + } \omega _ { \ell , t } ( \rho ) = \Pi _ { \ell , t } \omega _ { \ell , t } ( \rho ) = \omega _ { \ell , t } ( \rho ) .
$$

Consequently, each summand is

$$
\mathrm { T r } _ { [ t ] } \left( F _ { a i } ( \mathrm { I } _ { a } \otimes \omega _ { \ell , t } ( \rho ) ) \right) .
$$

The swap $F _ { a i }$ moves the i-th input factor to the output factor a before the input factors are traced out. Therefore,

$$
\operatorname { T r } _ { [ t ] } \big ( F _ { a i } ( \mathrm { I } _ { a } \otimes \omega _ { \ell , t } ( \rho ) ) \big ) = \operatorname { T r } _ { [ t ] \setminus \{ i \} } \big ( \omega _ { \ell , t } ( \rho ) \big ) = \operatorname { T r } _ { [ t ] \setminus \{ 1 \} } \big ( \omega _ { \ell , t } ( \rho ) \big ) ,
$$

where the last equality follows from permutation invariance. All t summands are therefore equal, proving Equation (102).

For the second moment, substituting Equation (101) and following the same simplification gives

$$
\begin{array} { l } { \displaystyle \mathbb { E } _ { \rho } \left[ ( Y \otimes Y ) \mathbf { 1 } _ { \{ J = \ell \} } \right] = \frac { 1 } { t ^ { 2 } } \sum _ { i , j \in [ t ] } \mathrm { T r } _ { [ t ] } \left( F _ { a i } F _ { b j } ( \mathrm { I } _ { a b } \otimes \omega _ { \ell , t } ( \rho ) ) \right) } \\ { \displaystyle \qquad + \frac { 1 } { t ^ { 2 } } \sum _ { i = 1 } ^ { t } \mathrm { T r } _ { [ t ] } \left( ( F _ { a b } F _ { b i } + F _ { a b } F _ { a i } ) ( \mathrm { I } _ { a b } \otimes \omega _ { \ell , t } ( \rho ) ) \right) } \\ { \displaystyle \qquad + \frac { \ell } { t ^ { 2 } } \mathrm { T r } _ { [ t ] } \left( F _ { a b } ( \mathrm { I } _ { a b } \otimes \omega _ { \ell , t } ( \rho ) ) \right) . } \end{array}
$$

For $t \geq 2$ , the three kinds of terms evaluate as follows:

$$
\begin{array} { r l } & { \mathrm { T r } _ { [ t ] } \left( F _ { a i } F _ { b j } ( \mathrm { I } _ { a b } \otimes \omega _ { \ell , t } ( \rho ) ) \right) = \mathrm { T r } _ { [ t ] \setminus \{ 1 , 2 \} } \left( \omega _ { \ell , t } ( \rho ) \right) \quad ( i \neq j ) , } \\ & { \mathrm { T r } _ { [ t ] } \left( \left( F _ { a b } F _ { b i } + F _ { a b } F _ { a i } \right) \left( \mathrm { I } _ { a b } \otimes \omega _ { \ell , t } ( \rho ) \right) \right) = \Big ( \mathrm { T r } _ { [ t ] \setminus \{ 1 \} } \left( \omega _ { \ell , t } ( \rho ) \right) \otimes \mathrm { I } _ { d } } \\ & { \qquad \quad + \mathrm { I } _ { d } \otimes \mathrm { T r } _ { [ t ] \setminus \{ 1 \} } \left( \omega _ { \ell , t } ( \rho ) \right) \Big ) F _ { a b } , } \\ & { \mathrm { T r } _ { [ t ] } \left( F _ { a b } ( \mathrm { I } _ { a b } \otimes \omega _ { \ell , t } ( \rho ) ) \right) = \mathrm { T r } \big ( \omega _ { \ell , t } ( \rho ) \big ) F _ { a b } = \mathbb { P } _ { \rho } [ J = \ell ] F _ { a b } . } \end{array}
$$

The first sum contains $t ( t - 1 )$ ordered pairs $i \neq j$ , giving the coeficient $t ( t - 1 ) / t ^ { 2 } = ( t - 1 ) / t$ When $t = 1$ , this coeficient is zero and the sum is empty, so the first term is omitted. The remaining terms are evaluated in the same way, so Equation (103) holds for all $t \geq 1$ □

Proof of Proposition 5.1. By Lemma 5.3, the densities in Equation (78) form a POVM. Fix a state $\rho .$ We first sum the contributions from Lemma 5.8 over ℓ. The projectors $\Delta _ { \ell , t }$ sum to the identity, so

$$
\sum _ { \ell = 1 } ^ { \ell _ { \mathrm { m a x } } } \omega _ { \ell , t } ( \rho ) = \rho ^ { \otimes t } .
$$

Taking one- and two-factor marginals of this identity gives

$$
\sum _ { \ell = 1 } ^ { \ell _ { \mathrm { m a x } } } \mathrm { T r } _ { [ t ] \setminus \{ 1 \} } \left( \omega _ { \ell , t } ( \rho ) \right) = \rho , \qquad \sum _ { \ell = 1 } ^ { \ell _ { \mathrm { m a x } } } \mathrm { T r } _ { [ t ] \setminus \{ 1 , 2 \} } \left( \omega _ { \ell , t } ( \rho ) \right) = \rho ^ { \otimes 2 } \quad \mathrm { w h e n ~ } t \geq 2 .
$$

Summing Equations (102) and (103) over ℓ gives

$$
\mathbb { E } _ { \rho } [ Y ] = \sum _ { \ell = 1 } ^ { \ell _ { \mathrm { { m a x } } } } \operatorname { T r } _ { [ t ] \backslash \{ 1 \} } \left( \omega _ { \ell , t } ( \rho ) \right) = \rho ,
$$

$$
\mathbb { E } _ { \rho } [ Y \otimes Y ] = \frac { t - 1 } { t } \rho ^ { \otimes 2 } + \frac { 1 } { t } ( \rho \otimes \mathrm { I } _ { d } + \mathrm { I } _ { d } \otimes \rho ) F + \frac { 1 } { t ^ { 2 } } \left( \sum _ { \ell = 1 } ^ { \ell _ { \operatorname* { m a x } } } \ell \operatorname { T r } ( \omega _ { \ell , t } ( \rho ) ) \right) F .
$$

By Equation (89),

$$
\sum _ { \ell = 1 } ^ { \ell _ { \mathrm { m a x } } } \ell \operatorname { T r } ( \omega _ { \ell , t } ( \rho ) ) = \sum _ { \ell = 1 } ^ { \ell _ { \mathrm { m a x } } } \ell \mathbb { P } _ { \rho } [ J = \ell ] = \mathbb { E } _ { \rho } [ J ] .
$$

Multiplying the second moment identity by $H \otimes H$ , taking the trace, and using Equation (11) gives

$$
\mathbb { E } _ { \rho } \big [ \mathrm { T r } ( H Y ) ^ { 2 } \big ] = \frac { t - 1 } { t } \big ( \mathrm { T r } ( H \rho ) \big ) ^ { 2 } + \frac { 2 } { t } \mathrm { T r } ( \rho H ^ { 2 } ) + { \frac { \mathbb { E } _ { \rho } [ J ] } { t ^ { 2 } } } \| H \| _ { \mathrm { F } } ^ { 2 } ,
$$

$$
\begin{array} { r l } & { \mathrm { V a r } _ { \rho } \big ( \mathrm { T r } ( H Y ) \big ) = \mathbb { E } _ { \rho } \big [ \mathrm { T r } ( H Y ) ^ { 2 } \big ] - \big ( \mathrm { T r } ( H \rho ) \big ) ^ { 2 } } \\ & { \quad \quad \quad = - \displaystyle \frac { 1 } { t } \big ( \mathrm { T r } ( H \rho ) \big ) ^ { 2 } + \frac { 2 } { t } \mathrm { T r } ( \rho H ^ { 2 } ) + \frac { \mathbb { E } _ { \rho } [ J ] } { t ^ { 2 } } \left\| H \right\| _ { \mathrm { F } } ^ { 2 } } \\ & { \quad \quad \quad \leq \displaystyle \frac { 2 } { t } \mathrm { T r } ( \rho H ^ { 2 } ) + \frac { \mathbb { E } _ { \rho } [ J ] } { t ^ { 2 } } \left\| H \right\| _ { \mathrm { F } } ^ { 2 } . } \end{array}
$$

If $\rho$ has rank at most r, Lemma 5.5 gives $\mathbb { E } _ { \rho } [ J ] = O ( \operatorname* { m i n } \{ r , { \sqrt { t } } \} )$

It remains to prove the assertion about $\Pi ^ { \perp } G$ . Let Π be any orthogonal projector satisfying $\Pi \rho = \rho .$ . For a fixed ℓ with $\mathbb { P } _ { \rho } [ J = \ell ] > 0$ , we compare the Gaussian reference law $\gamma _ { d , \ell }$ with the outcome law of G conditional on $J = { \boldsymbol { \ell } } .$ . We show that the density of this outcome law relative to $\gamma _ { d , \ell }$ depends only on ΠG.

By Lemmas 5.2 and 5.6, both $\Delta _ { \ell , t }$ and $( \Gamma _ { \ell , t } ^ { + } ) ^ { 1 / 2 }$ commute with $\Pi ^ { \otimes t }$ . Therefore,

$$
\rho ^ { \otimes t } = \Pi ^ { \otimes t } \rho ^ { \otimes t } \Pi ^ { \otimes t } \quad \Longrightarrow \quad \omega _ { \ell , t } ( \rho ) = \Pi ^ { \otimes t } \omega _ { \ell , t } ( \rho ) \Pi ^ { \otimes t } \quad \Longrightarrow \quad \Theta _ { \ell , \rho } = \Pi ^ { \otimes t } \Theta _ { \ell , \rho } \Pi ^ { \otimes t } .
$$

The density in Equation (104) satisfies

$$
p _ { \ell , \rho } ( G ) = \mathrm { T r } \left( \Theta _ { \ell , \rho } ( \Pi G G ^ { \dagger } \Pi ) ^ { \otimes t } \right) = p _ { \ell , \rho } ( \Pi G ) .
$$

Put $G _ { \parallel } : = \Pi G$ and $G _ { \bot } : = \Pi ^ { \bot } G$

Let $\gamma _ { \Pi , \ell }$ and $\gamma _ { \Pi ^ { \perp } , \ell }$ denote the standard complex Gaussian laws on matrices whose columns lie in Im(Π) and Im(Π<sup>⊥</sup>), respectively. Under $\gamma _ { d , \ell } ,$ the two components are independent, and their joint law factors as

$$
\mathrm { d } \gamma _ { d , \ell } ( G ) = \mathrm { d } \gamma _ { \Pi , \ell } ( G _ { \parallel } ) \mathrm { d } \gamma _ { \Pi ^ { \perp } , \ell } ( G _ { \perp } ) .
$$

Therefore, the conditional distribution of $( G _ { \parallel } , G _ { \perp } )$ given $J = \ell$ is

$$
\frac { p _ { \ell , \rho } ( G _ { \parallel } ) } { \mathbb { P } _ { \rho } [ J = \ell ] } \ \mathrm { d } \gamma _ { \Pi , \ell } ( G _ { \parallel } ) \ \mathrm { d } \gamma _ { \Pi ^ { \perp } , \ell } ( G _ { \perp } ) .
$$

The factor involving $p _ { \ell , \rho }$ depends only on $G _ { \parallel }$ . Thus, under the outcome law conditional on $J = \ell ,$ $G _ { \perp } = \Pi ^ { \perp } G$ is independent of $G _ { \parallel } = \Pi G$ and has the same law $\gamma _ { \Pi ^ { \perp } , \ell }$ as under the Gaussian reference measure. Equivalently, its columns are independent standard complex Gaussian vectors in $\mathrm { I m } ( \Pi ^ { \perp } )$ . This completes the proof. □

## 5.3 Analysis of the estimator

With Proposition 5.1 established, we now analyze the estimator defined in Equations (80) and (81). Choose an r-dimensional subspace containing the support of the unknown state $\rho ,$ and let Π be the orthogonal projector onto this subspace. We write the error $E : = \overline { { Y } } - \rho$ in the four blocks determined by the direct sum decomposition $\mathbb { C } ^ { d } = \operatorname { I m } ( \Pi ) \oplus \operatorname { I m } ( \Pi ^ { \perp } )$ . The scalar variance bound in Equation (85) controls the supported and of-diagonal blocks in Frobenius norm. The conditional law of $\Pi ^ { \perp } G$ controls the block $\Pi ^ { \perp } E \Pi ^ { \perp }$ in operator norm. The following projection bound combines these three estimates. The protocol does not need to know Π; this decomposition is used only in the analysis.

## 5.3.1 Projection onto low-rank states

Lemma 5.9 (Rank-constrained Frobenius projection). Let $\rho \in \mathcal { D } _ { r } ( \mathbb { C } ^ { d } )$ , let $M \in \mathbb { C } ^ { d \times d }$ be Hermitian, and put $E : = M - \rho$ . Let

$$
{ \widehat { \rho } } \in \arg \operatorname* { m i n } _ { \sigma \in { \mathcal { D } } _ { r } ( \mathbb { C } ^ { d } ) } \left\| M - \sigma \right\| _ { \mathrm { F } } .
$$

If Π is a rank-r orthogonal projector satisfying $\Pi \rho = \rho ,$ then

$$
\begin{array} { r } { \left\| \widehat { \rho } - \rho \right\| _ { 1 } \leq 2 \sqrt { 2 r } \left( \left\| \Pi E \Pi \right\| _ { \mathrm { F } } ^ { 2 } + 2 \left\| \Pi ^ { \perp } E \Pi \right\| _ { \mathrm { F } } ^ { 2 } + 2 r \left\| \Pi ^ { \perp } E \Pi ^ { \perp } \right\| _ { \mathrm { o p } } ^ { 2 } \right) ^ { 1 / 2 } . } \end{array}\tag{105}
$$

Proof. Put $D : = { \widehat { \rho } } - \rho$ . Since $\rho$ is feasible in the minimization that defines ${ \widehat { \rho } } ,$

$$
\begin{array} { r } { \| E - D \| _ { \mathrm { F } } ^ { 2 } \leq \| E \| _ { \mathrm { F } } ^ { 2 } . } \end{array}
$$

Expanding the left-hand side gives

$$
\| D \| _ { \mathrm { F } } ^ { 2 } \leq 2 \mathrm { T r } ( E D ) .\tag{106}
$$

Decomposing the inner product according to $\mathrm { I } _ { d } = \Pi + \Pi ^ { \perp }$ , and using the property that E and D are Hermitian, gives

$$
\begin{array} { r l } & { \mathrm { T r } ( E D ) = \mathrm { T r } ( \boldsymbol { \Pi } E \boldsymbol { \Pi } D \boldsymbol { \Pi } ) + 2 \mathrm { R e } \left( \mathrm { T r } ( \boldsymbol { \Pi } E \boldsymbol { \Pi } ^ { \perp } D \boldsymbol { \Pi } ) \right) + \mathrm { T r } ( \boldsymbol { \Pi } ^ { \perp } E \boldsymbol { \Pi } ^ { \perp } D \boldsymbol { \Pi } ^ { \perp } ) } \\ & { \qquad \leq \left. \boldsymbol { \Pi } E \boldsymbol { \Pi } \right. _ { \mathrm { F } } \left. \boldsymbol { \Pi } D \boldsymbol { \Pi } \right. _ { \mathrm { F } } + 2 \left. \boldsymbol { \Pi } ^ { \perp } E \boldsymbol { \Pi } \right. _ { \mathrm { F } } \left. \boldsymbol { \Pi } ^ { \perp } D \boldsymbol { \Pi } \right. _ { \mathrm { F } } + \left. \boldsymbol { \Pi } ^ { \perp } E \boldsymbol { \Pi } ^ { \perp } \right. _ { \mathrm { o p } } \left. \boldsymbol { \Pi } ^ { \perp } D \boldsymbol { \Pi } ^ { \perp } \right. _ { 1 } . } \end{array}\tag{107}
$$

Both $\rho$ and $\widehat { \rho }$ have rank at most $^ { r , }$ so

$$
\mathrm { r a n k } ( D ) \leq \mathrm { r a n k } ( \rho ) + \mathrm { r a n k } ( \hat { \rho } ) \leq 2 r , \qquad \left\| \Pi ^ { \perp } D \Pi ^ { \perp } \right\| _ { 1 } \leq \sqrt { 2 r } \left\| \Pi ^ { \perp } D \Pi ^ { \perp } \right\| _ { \mathrm { F } } .\tag{108}
$$

The four blocks are orthogonal in the Frobenius (Hilbert–Schmidt) inner product, so

$$
\left\| D \right\| _ { \mathrm { F } } ^ { 2 } = \left\| \Pi D \Pi \right\| _ { \mathrm { F } } ^ { 2 } + 2 \left\| \Pi ^ { \perp } D \Pi \right\| _ { \mathrm { F } } ^ { 2 } + \left\| \Pi ^ { \perp } D \Pi ^ { \perp } \right\| _ { \mathrm { F } } ^ { 2 } .
$$

Using Equation (108) in the right-hand side of Equation (107) and applying Cauchy–Schwarz to the three products gives

$$
\begin{array} { r l } & { \mathrm { T r } ( E D ) \leq \left( \left. \Pi E \Pi \right. _ { \mathrm { F } } ^ { 2 } + 2 \left. \Pi ^ { \perp } E \Pi \right. _ { \mathrm { F } } ^ { 2 } + 2 r \left. \Pi ^ { \perp } E \Pi ^ { \perp } \right. _ { \mathrm { o p } } ^ { 2 } \right) ^ { 1 / 2 } } \\ & { \qquad \cdot \left( \left. \Pi D \Pi \right. _ { \mathrm { F } } ^ { 2 } + 2 \left. \Pi ^ { \perp } D \Pi \right. _ { \mathrm { F } } ^ { 2 } + \left. \Pi ^ { \perp } D \Pi ^ { \perp } \right. _ { \mathrm { F } } ^ { 2 } \right) ^ { 1 / 2 } } \\ & { = \left( \left. \Pi E \Pi \right. _ { \mathrm { F } } ^ { 2 } + 2 \left. \Pi ^ { \perp } E \Pi \right. _ { \mathrm { F } } ^ { 2 } + 2 r \left. \Pi ^ { \perp } E \Pi ^ { \perp } \right. _ { \mathrm { o p } } ^ { 2 } \right) ^ { 1 / 2 } \left. D \right. _ { \mathrm { F } } . } \end{array}
$$

If $D = 0$ , the result is immediate. Otherwise, combining this estimate with Equation (106) and dividing by $\| D \| _ { \mathrm { F } }$ gives

$$
\begin{array} { r } { \left\| D \right\| _ { \mathrm { F } } \leq 2 \left( \left\| \Pi E \Pi \right\| _ { \mathrm { F } } ^ { 2 } + 2 \left\| \Pi ^ { \perp } E \Pi \right\| _ { \mathrm { F } } ^ { 2 } + 2 r \left\| \Pi ^ { \perp } E \Pi ^ { \perp } \right\| _ { \mathrm { o p } } ^ { 2 } \right) ^ { 1 / 2 } . } \end{array}
$$

Finally, $\| D \| _ { 1 } \leq \sqrt { 2 r } \| D \| _ { \mathrm { F } }$ , which proves Equation (105).

## 5.3.2 Bounds for the four blocks

We next bound the operator norm of the block $\Pi ^ { \perp } E \Pi ^ { \perp }$ . For a standard complex Gaussian matrix $\ b { Z } \in \mathbb { C } ^ { m \times k }$ , we have E $[ Z Z ^ { \dagger } ] = k  { \mathrm { I } _ { m } }$ , so $Z Z ^ { \dagger } - k \mathbf { I } _ { m }$ in Equation (109) below is the deviation of $Z Z ^ { \dagger }$ from its expectation. The following bound on the expected deviation follows as in the proof of Theorem 4.6.1 and Remark 4.6.2 (in Sec. 4.6) of the text by Vershynin [Ver26]. (The results are stated for real subgaussian matrices, but can be adapted to the complex case in a straightforward manner.)

Lemma 5.10. Let m, k be positive integers, and let $Z \in \mathbb { C } ^ { m \times k }$ be a standard complex Gaussian matrix. Then

$$
\mathbb { E } \left\| Z Z ^ { \dagger } - k \mathrm { I } _ { m } \right\| _ { \mathrm { o p } } ^ { 2 } = O ( m k + m ^ { 2 } ) .\tag{109}
$$

We now apply Equation (85). It gives the first two bounds below. For the third, we use the conditional law from Proposition 5.1: the columns of $\Pi ^ { \perp } G$ remain independent standard complex Gaussian vectors in Im(Π<sup>⊥</sup>) after conditioning on $J .$

Lemma 5.11 (Error in the four blocks). Consider the measurement and averaging step of Section 5.1 with arbitrary positive integers $B , s _ { i }$ , and let $\rho \in { \mathcal { D } } _ { r } ( \mathbb { C } ^ { d } )$ . Let Π be a rank-r orthogonal projector satisfying $\Pi \rho = \rho$ . Put $m : = d - r _ { \ast }$ , and apply the joint measurement $\mathsf { M } _ { s }$ independently B times, each time on s fresh samples with joint state $\rho ^ { \otimes s }$ . For $b \in [ B ]$ , write $( J _ { b } , G _ { b } )$ for the b-th outcome and $Y _ { b } : = ( G _ { b } G _ { b } ^ { \dagger } - J _ { b } \mathrm { I } _ { d } ) / s$ . Set

$$
\overline { { Y } } : = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } Y _ { b } , \qquad E : = \overline { { Y } } - \rho , \qquad N : = B s .
$$

We have

$$
\mathbb { E } _ { \rho } \left\| \Pi E \Pi \right\| _ { \mathrm { F } } ^ { 2 } \le \frac { 2 r } { N } + \frac { r ^ { 2 } \mathbb { E } _ { \rho } [ J _ { 1 } ] } { N s } ,\tag{110}
$$

$$
\mathbb { E } _ { \rho } \left. \Pi ^ { \perp } E \Pi \right. _ { \mathrm { F } } ^ { 2 } \leq \frac { m } { N } + \frac { m r \mathbb { E } _ { \rho } [ J _ { 1 } ] } { N s } ,\tag{111}
$$

$$
\mathbb { E } _ { \rho } \left\| \Pi ^ { \perp } E \Pi ^ { \perp } \right\| _ { \mathrm { o p } } ^ { 2 } = O \left( \frac { m \mathbb { E } _ { \rho } [ J _ { 1 } ] } { N s } + \frac { m ^ { 2 } } { N ^ { 2 } } \right) .\tag{112}
$$

Proof. For every Hermitian matrix H, unbiasedness of $Y _ { 1 }$ , independence of the B outcomes, and Equation (85) give

$$
\mathbb { E } _ { \rho } [ \mathrm { T r } ( H E ) ^ { 2 } ] = \frac { 1 } { B } \mathrm { V a r } _ { \rho } ( \mathrm { T r } ( H Y _ { 1 } ) ) \leq \frac { 1 } { B } \left( \frac { 2 } { s } \mathrm { T r } ( \rho H ^ { 2 } ) + \frac { \mathbb { E } _ { \rho } [ J _ { 1 } ] } { s ^ { 2 } } \| H \| _ { \mathrm { F } } ^ { 2 } \right) .\tag{113}
$$

To sum this bound over the block on Im(Π), choose an orthonormal basis $\{ e _ { j } \} _ { j = 1 } ^ { r }$ of this subspace. Let $H _ { 1 } , \ldots , H _ { r ^ { 2 } }$ be the Frobenius orthonormal basis of the Hermitian operators supported on Im(Π) consisting of

$$
| e _ { j } \rangle \langle e _ { j } | \quad ( j \in [ r ] ) , \qquad \frac { | e _ { j } \rangle \langle e _ { k } | + | e _ { k } \rangle \langle e _ { j } | } { \sqrt { 2 } } , \quad \frac { \mathrm { i } ( | e _ { j } \rangle \langle e _ { k } | - | e _ { k } \rangle \langle e _ { j } | ) } { \sqrt { 2 } } \quad ( 1 \leq j < k \leq r ) .
$$

For each pair $j < k$ , the squares of the two of-diagonal basis matrices sum to $| e _ { j } \rangle \langle e _ { j } | + | e _ { k } \rangle \langle e _ { k } |$ Hence

$$
\sum _ { u = 1 } ^ { r ^ { 2 } } H _ { u } ^ { 2 } = \sum _ { j = 1 } ^ { r } | e _ { j } \rangle \langle e _ { j } | + \sum _ { 1 \leq j < k \leq r } ( | e _ { j } \rangle \langle e _ { j } | + | e _ { k } \rangle \langle e _ { k } | ) = r \Pi .
$$

This identity and Parseval’s identity give

$$
\| \mathrm { I L H I } \| _ { \mathrm { F } } ^ { 2 } = \sum _ { u } \mathrm { T r } ( H _ { u } E ) ^ { 2 } , \qquad \sum _ { u } \mathrm { T r } ( \rho H _ { u } ^ { 2 } ) = r , \qquad \sum _ { u } \| H _ { u } \| _ { \mathrm { F } } ^ { 2 } = r ^ { 2 } .
$$

Summing Equation (113) over u gives

$$
\mathbb { E } _ { \rho } \left\| \Pi E \Pi \right\| _ { \mathrm { F } } ^ { 2 } \leq \frac { 1 } { B } \left( \frac { 2 r } { s } + \frac { r ^ { 2 } \mathbb { E } _ { \rho } [ J _ { 1 } ] } { s ^ { 2 } } \right) = \frac { 2 r } { N } + \frac { r ^ { 2 } \mathbb { E } _ { \rho } [ J _ { 1 } ] } { N s } ,
$$

which is Equation (110).

If $m = 0$ , the remaining two blocks vanish, so their bounds are immediate. Assume from now on that $m \geq 1$ . For the two of-diagonal blocks, also choose an orthonormal basis $\{ f _ { i } \} _ { i = 1 } ^ { m }$ of $\mathrm { I m } ( \Pi ^ { \perp } )$ For $i \in [ m ]$ and $j \in [ r ]$ , define

$$
H _ { i j } ^ { \mathrm { R } } : = \frac { | f _ { i } \rangle \langle e _ { j } | + | e _ { j } \rangle \langle f _ { i } | } { \sqrt { 2 } } , \qquad H _ { i j } ^ { \mathrm { I } } : = \frac { \mathrm { i } ( | f _ { i } \rangle \langle e _ { j } | - | e _ { j } \rangle \langle f _ { i } | ) } { \sqrt { 2 } } .
$$

A direct calculation gives

$$
\sum _ { i , j } \left( \mathrm { T r } ( H _ { i j } ^ { \mathrm { R } } E ) ^ { 2 } + \mathrm { T r } ( H _ { i j } ^ { \mathrm { I } } E ) ^ { 2 } \right) = 2 \left. \Pi ^ { \perp } E \Pi \right. _ { \mathrm { F } } ^ { 2 } ,
$$

$$
\sum _ { i , j } \left( ( H _ { i j } ^ { \mathrm { R } } ) ^ { 2 } + ( H _ { i j } ^ { \mathrm { I } } ) ^ { 2 } \right) = r \Pi ^ { \perp } + m \Pi .
$$

Moreover, the $2 m r$ matrices $H _ { i j } ^ { \mathrm { R } } , H _ { i j } ^ { \mathrm { I } }$ all have Frobenius norm one. Since $\Pi \rho = \rho ,$ summing Equation (113) over them therefore gives

$$
2 \mathbb { E } _ { \rho } \left. \Pi ^ { \perp } E \Pi \right. _ { \mathrm { F } } ^ { 2 } \leq \frac { 1 } { B } \left( \frac { 2 m } { s } + \frac { 2 m r \mathbb { E } _ { \rho } [ J _ { 1 } ] } { s ^ { 2 } } \right) ,
$$

which is Equation (111).

It remains to control the block $\Pi ^ { \perp } E \Pi ^ { \perp }$ . Let $G _ { \perp , b } \in \mathbb { C } ^ { m \times J _ { b } }$ be the coordinate matrix of $\Pi ^ { \perp } G _ { b }$ in a fixed orthonormal basis of $\mathrm { I m } ( \Pi ^ { \perp } )$ . By Proposition 5.1 and independence of the B measurements, conditional on $J _ { 1 } , \ldots , J _ { B }$ , the matrices $G _ { \bot , 1 } , \dots , G _ { \bot , B }$ are independent standard complex Gaussian matrices. Set

$$
K : = \sum _ { b = 1 } ^ { B } J _ { b } , \qquad Z : = \left[ G _ { \perp , 1 } \quad \cdots \quad G _ { \perp , B } \right] \in \mathbb { C } ^ { m \times K } .
$$

Then

$Z \mid ( J _ { 1 } , \ldots , J _ { B } )$ is a standard complex Gaussian matrix, $Z Z ^ { \dagger } = \sum _ { b = 1 } ^ { B } G _ { \perp , b } G _ { \perp , b } ^ { \dagger } .$

Since $\Pi ^ { \perp } \rho \Pi ^ { \perp } = 0$ and $N = B s$ , in the chosen basis of $\mathrm { I m } ( \Pi ^ { \perp } )$

$$
\Pi ^ { \perp } E \Pi ^ { \perp } = \frac { 1 } { B s } \sum _ { b = 1 } ^ { B } \left( { G _ { \perp , b } G _ { \perp , b } ^ { \dagger } - J _ { b } \mathrm { I } _ { m } } \right) = \frac { 1 } { N } ( Z Z ^ { \dagger } - K \mathrm { I } _ { m } ) .
$$

By Lemma 5.10,

$$
\mathbb { E } _ { \rho } \left[ \left\| \Pi ^ { \perp } E \Pi ^ { \perp } \right\| _ { \mathrm { o p } } ^ { 2 } \mid J _ { 1 } , \ldots , J _ { B } \right] = O \left( { \frac { m K + m ^ { 2 } } { N ^ { 2 } } } \right) .
$$

Taking expectation over $J _ { 1 } , \ldots , J _ { B }$ , and using $\mathbb { E } _ { \rho } [ K ] = B \mathbb { E } _ { \rho } [ J _ { 1 } ]$ and $N = B s$ , gives

$$
\mathbb { E } _ { \rho } \left\| \Pi ^ { \perp } E \Pi ^ { \perp } \right\| _ { \mathrm { o p } } ^ { 2 } = O \left( \frac { m B \mathbb { E } _ { \rho } [ J _ { 1 } ] + m ^ { 2 } } { N ^ { 2 } } \right) = O \left( \frac { m \mathbb { E } _ { \rho } [ J _ { 1 } ] } { N s } + \frac { m ^ { 2 } } { N ^ { 2 } } \right) ,
$$

proving Equation (112).

## 5.3.3 Completion of the upper bound

Proof of Theorem 1.2. Consider the protocol in Equations (79) to (81). Fix $\rho \in { \mathcal { D } } _ { r } ( \mathbb { C } ^ { d } )$ , choose a rank-r orthogonal projector Π satisfying $\Pi \rho = \rho ,$ and let $E : = \overline { { Y } } - \rho$ . By Proposition 5.1, $\mathbb { E } _ { \rho } [ J _ { 1 } ] = O ( \sqrt { s } )$ , and the choice in Equation (79) gives $s \leq r ^ { 2 }$ . Combining the three bounds in Lemma 5.11 gives

$$
\mathbb { E } _ { \rho } \left[ \left. \Pi E \Pi \right. _ { \mathrm { F } } ^ { 2 } + 2 \left. \Pi ^ { \perp } E \Pi \right. _ { \mathrm { F } } ^ { 2 } + 2 r \left. \Pi ^ { \perp } E \Pi ^ { \perp } \right. _ { \mathrm { o p } } ^ { 2 } \right] = O \left( \frac { d } { N } + \frac { d r } { N \sqrt { s } } + \frac { r d ^ { 2 } } { N ^ { 2 } } \right) .
$$

Squaring Equation (105) and taking expectations therefore gives

$$
\mathbb { E } _ { \rho } \left\| \widehat { \rho } - \rho \right\| _ { 1 } ^ { 2 } = O \left( \frac { d r } { N } + \frac { d r ^ { 2 } } { N \sqrt { s } } + \frac { d ^ { 2 } r ^ { 2 } } { N ^ { 2 } } \right) .\tag{114}
$$

This bound explains the choice $s = \operatorname* { m i n } \{ t , r ^ { 2 } \}$ . For fixed $N .$ , the sum of the first two terms reaches order $d r / N$ at $s = r ^ { 2 } ;$ ; increasing s further changes this sum by at most a constant factor. Jointly measuring more samples at a time would not improve the resulting asymptotic rate.

By Equation (79),

$$
\frac { d r } { N } + \frac { d r ^ { 2 } } { N \sqrt { s } } \leq \frac { \varepsilon ^ { 2 } } { C _ { 0 } } , \qquad \frac { d ^ { 2 } r ^ { 2 } } { N ^ { 2 } } \leq \frac { \varepsilon ^ { 4 } } { C _ { 0 } ^ { 2 } } \leq \frac { \varepsilon ^ { 2 } } { C _ { 0 } ^ { 2 } } .
$$

Thus, for suficiently large $C _ { 0 }$ , Equation (114) and Markov’s inequality give

$$
\mathbb { P } _ { \rho } \left[ \left. \widehat { \rho } - \rho \right. _ { 1 } > \varepsilon \right] \leq \frac { 1 } { 3 } .
$$

By the definition of B in Equation (79),

$$
N < C _ { 0 } \frac { d r } { \varepsilon ^ { 2 } } \left( 1 + \frac { r } { \sqrt { s } } \right) + s .
$$

Since $s \leq r ^ { 2 } \leq d r / \varepsilon ^ { 2 }$ and $r / \sqrt { s } = \operatorname* { m a x } \{ 1 , r / \sqrt { t } \}$ , this gives

$$
N = O \left( \frac { d r } { \varepsilon ^ { 2 } } \operatorname* { m a x } \left\{ 1 , \frac { r } { \sqrt { t } } \right\} \right) .
$$

This proves Theorem 1.2.

## 6 Discussion

The matching upper and lower bounds show the role rank plays in the performance of joint measurements in algorithms for state tomography. Relative to single-sample tomography, jointly measuring t samples improves the optimal sample complexity by a factor of order $\sqrt { t }$ until t reaches order $r ^ { 2 }$ . At that point, the sample complexity matches that of unrestricted collective measurements. The upper bound is achieved by a nonadaptive protocol, so classical adaptivity ofers no further improvement beyond constant factors.

Two ingredients of the lower bound may be useful for future work on quantum estimation. The first is our bound on the Fisher information trace for the support rotation family, which holds for every joint measurement on t samples. The second is the use of the conditional score chain rule to extend such bounds to adaptive protocols. This allows the analysis of adaptivity to build on a bound for each joint measurement.

The upper bound controls the error in the supported and of-diagonal blocks of the estimator in Frobenius norm, and the error in the block on the orthogonal complement of the support of the unknown state in operator norm. The first two bounds follow from the estimator’s second moment identity; the third uses the conditional Gaussian law outside the state’s support. The expected number of columns of the matrix outcome G appears in both the Frobenius and operator norm error bounds. The rank-constrained projection bound combines these estimates to give the final trace norm guarantee. This approach may be useful in other variants of tomography.

In retrospect, the Gaussian protocol appears related to several previous mixed state tomography algorithms. Clarifying the precise relationship between these protocols is a direction for future work.

Our protocol uses joint measurements with continuous outcomes. A natural direction is to construct explicit versions with finitely many outcomes that preserve the optimal sample complexity, and to determine how many outcomes are needed.

Our results concern protocols that retain only classical information between measurement rounds. With persistent quantum memory, diferent rounds can combine into a larger coherent measurement, so the largest number of fresh samples measured in one round no longer captures the available quantum resource. Understanding tradeofs that simultaneously constrain the number of samples measured in each round, quantum memory, sample complexity, and computational eficiency remains a natural direction.

## 7 AI disclosure

We used GPT-5.5 and GPT-5.6 Sol for mathematical assistance in this work. Their main mathematical contributions concerned two parts of the proofs. For the lower bound, the models provided substantial assistance with the proofs of Lemmas 4.3 and 4.4. These lemmas are crucial to our Fisher information bound for joint measurements on t > 1 samples and to the resulting dependence of the sample complexity lower bound on t.

For the upper bound, we used the models to reformulate earlier tomography algorithms and their analyses without using representation theory. Our key additional contribution is a rank-dependent error analysis of the resulting Gaussian formulation, presented in Section 5.

We also used these models for discussions of general mathematical questions, literature searches, and editorial revisions. The authors independently verified all mathematical arguments and references and take full responsibility for the content of the paper.

## 8 Acknowledgements

The authors thank Rain Ziming Yang and Jack Spalding-Jamieson for helpful discussions. Ashwin Nayak’s research is supported in part by NSERC grants RGPIN-2023-03731 and ALLRP-578455-2022. Xingyu Zhou is supported by NSERC Grant RGPIN-2024-06493.

## References

[AB06] Charalambos D. Aliprantis and Kim C. Border. Infinite Dimensional Analysis: A Hitchhiker’s Guide. Springer Berlin, Heidelberg, third edition, 2006.

[BC94] Samuel L. Braunstein and Carlton M. Caves. Statistical distance and the geometry of quantum states. Physical Review Letters, 72(22):3439–3443, May 1994.

[BGK15] Cristina Butucea, M˘ad˘alin Gut¸˘a, and Theodore Kypraios. Spectral thresholding quantum tomography for low rank states. New Journal of Physics, 17(11):113050, November 2015.

[BHO20] <sup>¨</sup> Leighton Pate Barnes, Yanjun Han, and Ayfer Ozg¨ur. Lower bounds for learning <sup>¨</sup> distributions under communication constraints via Fisher information. Journal of Machine Learning Research, 21(236):1–30, 2020.

[BNG00] O E Barndorf-Nielsen and R D Gill. Fisher information in quantum statistics. Journal of Physics A: Mathematical and General, 33(24):4481–4490, June 2000.

[BV88] Winfried Bruns and Udo Vetter. Determinantal Rings, volume 1327 of Lecture Notes in Mathematics. Springer Berlin Heidelberg, 1988.

[CHL<sup>+</sup>23] Sitan Chen, Brice Huang, Jerry Li, Allen Liu, and Mark Sellke. When does adaptivity help for quantum state learning? In 2023 IEEE 64th Annual Symposium on Foundations of Computer Science (FOCS), pages 391–404. IEEE, November 2023.

[CLL24] Sitan Chen, Jerry Li, and Allen Liu. An optimal tradeof between entanglement and copy complexity for state tomography. In Proceedings of the 56th Annual ACM Symposium on Theory of Computing, STOC ’24, pages 1331–1342. ACM, June 2024.

[FGLE12] Steven T Flammia, David Gross, Yi-Kai Liu, and Jens Eisert. Quantum tomography via compressed sensing: error bounds, sample complexity and eficient estimators. New Journal of Physics, 14(9):095022, September 2012.

[GKKT20] M Gut¸˘a, J Kahn, R Kueng, and J A Tropp. Fast state tomography with optimal error bounds. Journal of Physics A: Mathematical and Theoretical, 53(20):204001, April 2020.

[GL95] Richard D. Gill and Boris Y. Levit. Applications of the van Trees inequality: A Bayesian Cram´er–Rao bound. Bernoulli, 1(1/2):59–79, 1995.

[GLF<sup>+</sup>10] David Gross, Yi-Kai Liu, Steven T. Flammia, Stephen Becker, and Jens Eisert. Quantum state tomography via compressed sensing. Physical Review Letters, 105(15):150401, October 2010.

[GM00] Richard D. Gill and Serge Massar. State estimation for large ensembles. Physical Review A, 61(4):042312, March 2000.

[Hay98] Masahito Hayashi. Asymptotic estimation theory for a finite-dimensional pure state model. Journal of Physics A: Mathematical and General, 31(20):4633, May 1998.

[HHJ<sup>+</sup>17] Jeongwan Haah, Aram W. Harrow, Zhengfeng Ji, Xiaodi Wu, and Nengkun Yu. Sampleoptimal tomography of quantum states. IEEE Transactions on Information Theory, 63(9):5628–5641, 2017.

[HS16] Daniel Hsu and Sivan Sabato. Loss minimization and parameter estimation with heavy tails. Journal of Machine Learning Research, 17(18):1–40, April 2016.

[Iss18] L. Isserlis. On a formula for the product-moment coeficient of any order of a normal frequency distribution in any number of variables. Biometrika, 12(1-2):134–139, November 1918.

[KLMR26] Ufuk Keskin, Jason Luo, Mahbod Majid, and Matthew Radzihovsky. Tight lower bounds for state tomography with limited entanglement, 2026. arXiv:2609.05718.

[KRT17] Richard Kueng, Holger Rauhut, and Ulrich Terstiege. Low rank matrix recovery from rank one measurements. Applied and Computational Harmonic Analysis, 42(1):88–116, January 2017.

[LN25] Angus Lowe and Ashwin Nayak. Lower bounds for learning quantum states with singlecopy measurements. ACM Transactions on Computation Theory, 17(1):7:1–7:42, March 2025.

[OW16] Ryan O’Donnell and John Wright. Eficient quantum tomography. In Proceedings of the forty-eighth annual ACM symposium on Theory of Computing, STOC ’16, pages 899–912. ACM, June 2016.

[PSTW26] Angelos Pelecanos, Jack Spilecki, Ewin Tang, and John Wright. Mixed state tomography reduces to pure state tomography. In 67th IEEE Annual Symposium on Foundations of Computer Science (FOCS), 2026. To appear.

[PSW26] Angelos Pelecanos, Jack Spilecki, and John Wright. The debiased Keyl’s algorithm: A new unbiased estimator for full state tomography. In Proceedings of the 58th Annual ACM Symposium on Theory of Computing, STOC ’26, pages 1266–1277. ACM, June 2026.

[SSW25] Thilo Scharnhorst, Jack Spilecki, and John Wright. Optimal lower bounds for quantum state tomography, 2025. arXiv:2510.07699.

[Ver26] Roman Vershynin. High-Dimensional Probability: An Introduction with Applications in Data Science. Cambridge Series in Statistical and Probabilistic Mathematics. Cambridge University Press, Cambridge, UK, 2 edition, 2026.

[ZC26] Sisi Zhou and Senrui Chen. Randomized measurements for multiparameter quantum metrology. PRX Quantum, 7(1):010314, January 2026.

[ZS60] Oscar Zariski and Pierre Samuel. Commutative Algebra, volume II. D. Van Nostrand Company, 1960.

## A Statistical formalism

This appendix collects the standard statistical formalism used in the main proof. We first justify the density notation for continuous-outcome POVMs and adaptive transcripts on the general outcome spaces allowed by our model. We then verify the likelihood regularity used by the Fisher information argument and give the van Trees proof. In particular, all reference measures below are independent of the unknown state, so parameter derivatives and Fisher information are computed within one fixed dominated model.

## A.1 Dominated models and finite-dimensional POVMs

Let $\{ \mathbb { P } _ { \theta } : \pmb { \theta } \in \Theta \}$ be probability measures on $( Z , { \mathcal { Z } } )$ . A measure ν dominates this family if $\mathbb { P } _ { \theta } \ll \nu$ for every parameter, meaning that for every $\mathsf { E } \in \mathcal { Z }$

$$
\nu ( \mathsf { E } ) = 0 \quad \Longrightarrow \quad \mathbb { P } _ { \pmb { \theta } } ( \mathsf { E } ) = 0 .
$$

We assume $\nu$ is σ-finite, meaning that Z is a countable union of measurable sets of finite ν-measure. This holds for all the reference probability measures constructed below. The Radon–Nikodym theorem then gives likelihood densities

$$
q _ { \pmb \theta } ( z ) : = \frac { \mathrm { d } \mathbb P _ { \pmb \theta } } { \mathrm { d } \nu } ( z ) ,
$$

which satisfy Equation (12). An event $N \in { \mathcal { Z } }$ is ν-null if $\nu ( N ) = 0$ , and a property holds ν-almost everywhere if it fails only on such an event. Because the model is dominated, every ν-null event also has probability zero under every $\mathbb { P } _ { \theta }$

For a POVM M on $\mathbb { C } ^ { d }$ , the measure ν in Equation (15) is a probability measure:

$$
\nu ( \mathsf Z ) = d ^ { - 1 } \operatorname { T r } \bigl ( \mathsf { M } ( \mathsf Z ) \bigr ) = d ^ { - 1 } \operatorname { T r } ( \mathsf { I } _ { d } ) = 1 .
$$

It dominates every state-induced outcome law. Indeed, if $\nu ( \mathsf { E } ) = 0$ , then the positive semidefinite operator ${ \mathsf { M } } ( { \mathsf { E } } )$ has trace zero and is therefore the zero operator. Hence $\mathrm { T r } ( \rho \mathsf { M } ( \mathsf { E } ) ) = 0$ for every state $\rho .$ Each scalar measure of a matrix entry of M is consequently absolutely continuous with respect to ν. Applying the scalar Radon–Nikodym theorem entrywise gives a measurable operator-valued function $z \mapsto M _ { z }$ satisfying Equation (16). Positivity holds almost everywhere because, for a countable dense set of vectors u, the scalar measure $\mathsf { E } \mapsto \pmb { u } ^ { \dag } \mathsf { M } ( \mathsf { E } ) \pmb { u }$ is positive. Taking the trace in Equation (16) gives

$$
\int _ { \mathsf { E } } \mathrm { T r } ( M _ { z } ) ~ \mathrm { d } \nu ( z ) = \mathrm { T r } \big ( \mathsf { M } ( \mathsf { E } ) \big ) = d \nu ( \mathsf { E } )
$$

for every measurable event E, so $\mathrm { T r } ( M _ { z } ) = d$ ν-almost everywhere. This is the standard tracedominated POVM representation used in quantum statistics [BNG00].

Finally, if a nonnegative likelihood is diferentiable at an interior parameter value where it vanishes, its gradient vanishes there. Thus the score and the Fisher integrands may be set to zero on a zero-likelihood set, as done in Section 3.3.

## A.2 Adaptive transcript domination

Fix a protocol with pathwise sample bound N, and let $( \mathsf { U } , \Sigma \mathsf { u } , \kappa )$ be the probability space of its private random seed. We index the protocol by N rounds, padding it with zero-sample rounds after it halts. Write

$$
\mathsf { H } _ { i } : = \mathsf { U } \times \mathsf { Z } _ { 1 } \times \cdot \cdot \cdot \times \mathsf { Z } _ { i - 1 }
$$

for the history space before round i. At $h \in { \mathsf { H } } _ { i }$ , let $t _ { i } ( h ) \in \{ 0 , 1 , \ldots , t \}$ be the number of samples measured in round $i ,$ and let ${ \mathsf { M } } _ { i } ( \cdot \mid h )$ be the selected POVM on $( \mathbb { C } ^ { d } ) ^ { \otimes t _ { i } ( h ) }$ . The protocol definition requires $t _ { i }$ and the matrix entries of ${ \mathsf { M } } _ { i } ( \mathsf { E } \mid h )$ , on each set of histories where $t _ { i } ( h )$ is fixed, to be measurable in h for each $\mathsf { E } \in \mathcal { Z } _ { i }$ . For tuples, write $z _ { < i } : = ( z _ { 1 } , \dots , z _ { i - 1 } )$

Lemma A.1 (Measurable domination of adaptive measurements). Define the probability kernel

$$
\nu _ { i } ( \mathsf { E } \mid h ) : = d ^ { - t _ { i } ( h ) } \operatorname { T r } \big ( \mathsf { M } _ { i } ( \mathsf { E } \mid h ) \big ) .\tag{115}
$$

For each $t ^ { \prime } \in \{ 0 , \ldots , t \}$ , on the history slice $t _ { i } ( h ) = t ^ { \prime }$ , there is a jointly measurable positive semidefinite operator density $M _ { i } ( z \mid h )$ on $( \mathbb { C } ^ { d } ) ^ { \otimes t ^ { \prime } }$ such that

$$
\mathsf { M } _ { i } ( \mathsf { E } \mid h ) = \int _ { \mathsf { E } } M _ { i } ( z \mid h ) \nu _ { i } ( \mathrm { d } z \mid h ) , \qquad \mathrm { T r } ( M _ { i } ( z \mid h ) ) = d ^ { t ^ { \prime } } ,\tag{116}
$$

for every such history h, every $\mathsf { E } \in \mathcal { Z } _ { i }$ , and every $z \in Z _ { i }$ . Consequently, the parameter-independent measure

$$
\lambda _ { N } ( \mathrm { d } u \ \mathrm { d } z _ { 1 } \cdot \cdot \cdot \mathrm { d } z _ { N } ) : = \kappa ( \mathrm { d } u ) \prod _ { i = 1 } ^ { N } \nu _ { i } ( \mathrm { d } z _ { i } \mid u , z _ { < i } )\tag{117}
$$

dominates the transcript law for every input state.

Proof. The expression in Equation (115) has total mass one. If it assigns zero mass to E, then the positive semidefinite operator ${ \mathsf { M } } _ { i } ( \mathsf { E } \mid h )$ has trace zero and is itself zero. Thus every scalar measure of a matrix entry of the POVM kernel is absolutely continuous with respect to $\nu _ { i } ( \cdot \mid h )$

It remains to choose the Radon–Nikodym derivatives jointly measurably in the history and outcome. Restrict to histories with $t _ { i } ( h ) = t ^ { \prime }$ , choose a basis of $( \mathbb { C } ^ { d } ) ^ { \otimes t ^ { \prime } }$ , and let $\mu _ { a b } ( \mathsf { E } \mid h )$ be the scalar measure associated with the $( a , b ) \ – \mathrm { t h }$ entry of ${ \mathsf { M } } _ { i } ( \mathsf { E } \mid h )$ . Because $\mathcal { Z } _ { i }$ is countably generated,

there is an increasing sequence $( \mathcal { P } _ { n } ) _ { n \geq 1 }$ of finite measurable partitions whose cells generate $\mathcal { Z } _ { i }$ Write $\mathbf { 1 } _ { C }$ for the indicator of a cell $C ,$ equal to one on $C$ and zero outside $C ,$ and set

$$
f _ { a b , n } ( h , z ) : = \sum _ { C \in \mathcal { P } _ { n } } \mathbf { 1 } _ { C } ( z ) \frac { \mu _ { a b } ( C \mid h ) } { \nu _ { i } ( C \mid h ) } , \qquad 0 / 0 : = 0 .
$$

These step functions are jointly measurable in $( h , z )$ . For each fixed history $h ,$ the functions $f _ { a b , n } ( h , \cdot )$ are conditional expectations of the entrywise Radon–Nikodym derivative under $\nu _ { i } ( \cdot \mid h )$ , with respect to the finite partitions $\mathcal { P } _ { n }$ . These increasing partitions generate $\mathcal { Z } _ { i } .$ , so martingale convergence gives a limit that is finite almost everywhere and equal to the Radon–Nikodym derivative. On the jointly measurable set where every real and imaginary part has a finite limit, assemble these limits entrywise into $M _ { i } ( z \mid h )$ ; set $M _ { i } ( z \mid h ) = \operatorname { I } _ { d ^ { t ^ { \prime } } }$ elsewhere.

For every fixed history, positivity follows by intersecting the full-measure sets on which u<sup>†</sup> $M _ { i } ( z \mid$ $h ) { \boldsymbol u } \ge 0$ over a countable dense set of vectors u. The operator density identity and Equation (115) give

$$
\int _ { \mathsf { E } } \mathrm { T r } ( M _ { i } ( z \mid h ) ) \nu _ { i } ( \mathrm { d } z \mid h ) = \mathrm { T r } \big ( \mathsf { M } _ { i } ( \mathsf { E } \mid h ) \big ) = d ^ { t ^ { \prime } } \nu _ { i } ( \mathsf { E } \mid h )
$$

for every measurable event E, so $\operatorname { T r } ( M _ { i } ( z \mid h ) ) = d ^ { t ^ { \prime } }$ almost everywhere. The set on which the entrywise limits fail to exist finitely, positivity fails, or this trace identity fails is jointly measurable, and each of its sections is $\nu _ { i } ( \cdot \mid h ) { \cdot } \mathrm { n u l l }$ . Replacing the density by the identity operator on this exceptional set leaves every kernel integral unchanged and makes the density finite and positive semidefinite with the required trace everywhere.

Perform this construction on each of the finitely many measurable slices $t _ { i } ( h ) = t ^ { \prime }$ , including the scalar $t ^ { \prime } = 0$ slice. The Ionescu–Tulcea construction then gives $\lambda _ { N }$ . Iterating the conditional Born rule proves the asserted domination. □

For the support rotation family, the conditional likelihood density at history $h$ is

$$
p _ { i , X } ( z \mid h ) : = \mathrm { T r } \left( M _ { i } ( z \mid h ) \rho _ { X } ^ { \otimes t _ { i } ( h ) } \right) .\tag{118}
$$

Iterating these conditional densities gives the transcript likelihood in Equation (56) with respect to $\lambda _ { N }$

## A.3 Likelihood regularity

In this subsection, $C ^ { 1 }$ means continuously diferentiable: the derivative exists and varies continuously in the norm under discussion. For a measure $\mu , L _ { 1 } ( \mu )$ is the space of integrable scalar functions, with norm

$$
\| f \| _ { L _ { 1 } ( \mu ) } : = \int | f | \mathrm { d } \mu .
$$

Thus this function space $L _ { 1 }$ norm is distinct from the matrix trace norm, even though both use the index 1.

Lemma A.2 (Likelihood regularity). Let m, r be positive integers, let $\boldsymbol { \Theta } \subseteq \mathbb { C } ^ { m \times r }$ be open when the matrix space is regarded as real, and let $X \mapsto \rho _ { X }$ , for $X \in \Theta$ , be a family of states on $\mathbb { C } ^ { d }$ that is $C ^ { 1 }$ in trace norm. For each history $h ,$ the conditional likelihood in Equation (118) is pointwise $C ^ { 1 }$ for $\nu _ { i } ( \cdot \mid h )$ -almost every outcome and is $C ^ { 1 }$ as a map into $L _ { 1 } ( \nu _ { i } ( \cdot \mid h ) )$ . The full likelihood in Equation (56) has the corresponding properties $\lambda _ { N }$ -almost everywhere and in $L _ { 1 } ( \lambda _ { N } )$ . All derivatives are jointly measurable; conditional derivatives integrate to zero under $\nu _ { i } ( \cdot \mid h )$ , and full likelihood derivatives integrate to zero under $\lambda _ { N }$ . For a direction $H \in \mathbb { C } ^ { m \times r }$ 9

$$
\mathrm { D } _ { H } p _ { i , X } ( z \mid h ) = \mathrm { T r } \left( M _ { i } ( z \mid h ) \mathrm { D } _ { H } ( \rho _ { X } ^ { \otimes t _ { i } ( h ) } ) \right) .
$$

Proof. The construction in Section $\mathrm { A . 2 }$ gives positive semidefinite densities with

$$
\| M _ { i } ( z \mid h ) \| _ { \mathrm { o p } } \leq \mathrm { T r } ( M _ { i } ( z \mid h ) ) = d ^ { t _ { i } ( h ) } \leq d ^ { t } .
$$

Fix a closed ball K contained in Θ. Since $X \mapsto \rho _ { X } ^ { \otimes k }$ is $C ^ { 1 }$ for each $k \in \{ 0 , \ldots , t \}$ , compactness of K gives a finite constant $C _ { K }$ such that

$$
\begin{array} { r } { \left\| \mathrm { D } _ { H } ( \rho _ { X } ^ { \otimes k } ) \right\| _ { 1 } \le C _ { K } \left\| H \right\| _ { \mathrm { F } } \qquad ( X \in K , \ 0 \le k \le t ) . } \end{array}
$$

For fixed $( h , z )$ , diferentiating the trace pairing in Equation (118) gives the stated derivative formula and pointwise $C ^ { 1 }$ regularity. Trace/operator norm duality then gives, uniformly in $h , z$ and $X \in K$

$$
0 \leq p _ { i , X } ( z \mid h ) \leq d ^ { t } , \qquad \left| \mathrm { D } _ { H } p _ { i , X } ( z \mid h ) \right| \leq C _ { K } d ^ { t } \left\| H \right\| _ { \mathrm { F } } .
$$

The transcript likelihood $q _ { N , X }$ is a product of N such conditional densities, so it is pointwise $C ^ { 1 }$ , and the product rule gives

$$
0 \leq q _ { N , X } \leq d ^ { t N } , \qquad | \mathrm { D } _ { H } q _ { N , X } | \leq N C _ { K } d ^ { t N } \| H \| _ { \mathrm { F } } \quad ( X \in K ) .
$$

All derivatives are jointly measurable by their trace formulas and the finite product rule. Since $\nu _ { i } ( \cdot \mid h )$ and $\lambda _ { N }$ are probability measures, these local bounds are integrable. Dominated convergence applied to coordinate diference quotients and to diferences of coordinate derivatives proves $C ^ { 1 }$ regularity in the corresponding $L _ { 1 }$ spaces. It also permits diferentiation under the integrals. The conditional and full likelihood derivatives therefore integrate to zero, since each likelihood has integral one. □

## A.4 The van Trees inequality

For completeness, we prove the van Trees inequality stated in Lemma 4.8. This is the standard argument; see also [GL95]. Let $q _ { \theta }$ be likelihood densities with respect to a parameter-independent measure $\nu .$ We assume that, for ν-almost every observation, the likelihood is pointwise $C ^ { 1 }$ in the parameter, and that its coordinate derivatives are jointly measurable, agree with the $L _ { 1 } ( \nu )$ derivatives, and integrate to zero. These are precisely the properties established for our transcript likelihoods in Lemma $\mathrm { { A . 2 } }$

Proof of Lemma $4 . 8 .$ If the denominator in Equation (60) is infinite, the claim is immediate. We may therefore assume it is finite. The expected squared error is finite because $T$ is bounded and $\pi$ has compact support. Write

$$
f ( \pmb \theta , y ) : = \pi ( \pmb \theta ) q _ { \pmb \theta } ( y )
$$

for the joint density of the parameter and observation. All derivatives below are with respect to θ. At a zero of a diferentiable nonnegative function, its gradient vanishes. We therefore set the quotients involving $q _ { \pmb { \theta } } , \ \pi$ , or $f$ to zero wherever their denominators vanish.

We first bound the integral of $\| \nabla f \| _ { 2 } ^ { 2 } / f$ . The product rule gives $\nabla f = q _ { \pmb { \theta } } \nabla \pi + \pi \nabla q _ { \pmb { \theta } }$ , and hence

$$
\frac { \| \nabla f \| _ { 2 } ^ { 2 } } { f } = q _ { \theta } \frac { \| \nabla \pi \| _ { 2 } ^ { 2 } } { \pi } + \pi \frac { \| \nabla q _ { \theta } \| _ { 2 } ^ { 2 } } { q _ { \theta } } + 2 \langle \nabla \pi , \nabla q _ { \theta } \rangle .
$$

This identity also holds on the zero set of $f$ under the convention above. Since $\pi = \psi ^ { 2 }$ , we have $\nabla \pi = 2 \psi \nabla \psi$ , so

$$
\int _ { \Theta } \frac { \| \nabla \pi \| _ { 2 } ^ { 2 } } { \pi } ~ \mathrm { d } \theta = 4 \int _ { \{ \theta \in \Theta \colon \psi ( \theta ) \neq 0 \} } \| \nabla \psi \| _ { 2 } ^ { 2 } ~ \mathrm { d } \theta \leq 4 \int _ { \Theta } \| \nabla \psi \| _ { 2 } ^ { 2 } ~ \mathrm { d } \theta = I ( \pi ) < \infty .
$$

For each fixed $\theta ,$ the likelihood derivatives belong to $L _ { 1 } ( \nu )$ and integrate to zero by the stated regularity assumptions. Thus the cross term is integrable in $y .$ and

$$
\int \langle \nabla \pi ( \theta ) , \nabla q _ { \theta } ( y ) \rangle ~ \mathrm { d } \nu ( y ) = \langle \nabla \pi ( \theta ) , \int \nabla q _ { \theta } ( y ) ~ \mathrm { d } \nu ( y ) \rangle = 0 .
$$

Integrating the expansion first over y and then over $\theta ,$ using $\textstyle \int q _ { \pmb { \theta } } ( y ) \ \mathrm { d } \nu ( y ) = 1$ and the definition of the Fisher information matrix, therefore gives

$$
\begin{array} { r l } & { J : = \displaystyle \int _ { \Theta } \int \frac { \| \nabla f \| _ { 2 } ^ { 2 } } { f } ~ \mathrm { d } \nu ( y ) ~ \mathrm { d } \theta = \displaystyle \int _ { \Theta } \frac { \| \nabla \pi \| _ { 2 } ^ { 2 } } { \pi } ~ \mathrm { d } \theta + \displaystyle \int _ { \Theta } \pi ( \theta ) \left( \displaystyle \int \frac { \| \nabla q _ { \theta } ( y ) \| _ { 2 } ^ { 2 } } { q _ { \theta } ( y ) } ~ \mathrm { d } \nu ( y ) \right) \mathrm { d } \theta } \\ & { \qquad = \displaystyle \int _ { \Theta } \frac { \| \nabla \pi \| _ { 2 } ^ { 2 } } { \pi } ~ \mathrm { d } \theta + \mathbb { E } _ { \theta } \operatorname { T r } ( \boldsymbol { \mathcal { Z } } ( \theta ) ) } \\ & { \qquad \leq \displaystyle I ( \pi ) + \mathbb { E } _ { \theta } \operatorname { T r } ( \boldsymbol { \mathcal { Z } } ( \theta ) ) < \infty . } \end{array}
$$

This bound also gives the absolute integrability needed for Fubini’s theorem. For each $j \in [ p ]$ ， Cauchy–Schwarz gives

$$
\begin{array} { r l } & { \displaystyle \int _ { \Theta } \int \ d | ( T _ { j } ( y ) - \theta _ { j } ) \partial _ { j } f ( \theta , y ) | \ \mathrm { d } \nu ( y ) \ \mathrm { d } \theta \leq \left( \mathbb { E } ( T _ { j } - \theta _ { j } ) ^ { 2 } \right) ^ { 1 / 2 } \left( \int _ { \Theta } \int \frac { ( \partial _ { j } f ) ^ { 2 } } { f } \ \mathrm { d } \nu ( y ) \ \mathrm { d } \theta \right) ^ { 1 / 2 } } \\ & { \qquad \leq \left( \mathbb { E } ( T _ { j } - \theta _ { j } ) ^ { 2 } \right) ^ { 1 / 2 } \sqrt { J } < \infty . } \end{array}
$$

We now integrate by parts in $\theta _ { j }$ . The boundary term vanishes because $\pi ,$ and hence $f ,$ is compactly supported inside $\Theta .$ . The estimator $T _ { j } ( y )$ does not depend on $\theta ,$ so $\partial _ { j } ( T _ { j } ( y ) - \theta _ { j } ) = - 1$ Fubini’s theorem lets us put the parameter integral first, and integration by parts then gives

$$
\begin{array} { r l } { \displaystyle \int _ { \Theta } \int ( T _ { j } ( y ) - \theta _ { j } ) \partial _ { j } f ( \pmb \theta , y ) \mathrm { d } \nu ( y ) \mathrm { d } \pmb \theta = \int \displaystyle \int _ { \Theta } \left( T _ { j } ( y ) - \theta _ { j } \right) \partial _ { j } f ( \pmb \theta , y ) \mathrm { d } \pmb \theta \mathrm { d } \nu ( y ) } & { } \\ { \displaystyle } & { = - \int \int _ { \Theta } \partial _ { j } ( T _ { j } ( y ) - \theta _ { j } ) f ( \pmb \theta , y ) \mathrm { d } \pmb \theta \mathrm { d } \nu ( y ) } \\ { \displaystyle } & { = \int \int _ { \Theta } f ( \pmb \theta , y ) \mathrm { d } \pmb \theta \mathrm { d } \nu ( y ) } \\ { \displaystyle } & { = \int _ { \Theta } \pi ( \pmb \theta ) \left( \int q _ { \theta } ( y ) \mathrm { d } \nu ( y ) \right) \mathrm { d } \pmb \theta = 1 . } \end{array}
$$

Summing over $j \in [ p ]$ yields

$$
p = \int _ { \Theta } \int \langle T ( y ) - \pmb { \theta } , \nabla f ( \pmb { \theta } , y ) \rangle ~ \mathrm { d } \nu ( y ) ~ \mathrm { d } \pmb { \theta } .
$$

Finally, insert $\sqrt { f }$ into the identity for $p \colon$

$$
p = \int _ { \Theta } \int \langle T ( y ) - \theta , \nabla f \rangle { \mathrm { ~ d } } \nu ( y ) { \mathrm { ~ d } } \theta = \int _ { \Theta } \int \langle ( T ( y ) - \theta ) { \sqrt { f } } , \nabla f / { \sqrt { f } } \rangle { \mathrm { ~ d } } \nu ( y ) { \mathrm { ~ d } } \theta .
$$

Cauchy–Schwarz for these two vector-valued functions gives

$$
\begin{array} { l } { p ^ { 2 } \leq \left( \displaystyle \int _ { \Theta } \int \| T ( y ) - \pmb \theta \| _ { 2 } ^ { 2 } f \mathrm { d } \nu ( y ) \mathrm { d } \pmb \theta \right) \left( \displaystyle \int _ { \Theta } \int \frac { \| \nabla f \| _ { 2 } ^ { 2 } } { f } \mathrm { d } \nu ( y ) \mathrm { d } \theta \right) } \\ { = \mathbb { E } \| T - \pmb \theta \| _ { 2 } ^ { 2 } J . } \end{array}\tag{119}
$$

Substituting the bound for J and rearranging proves

$$
\mathbb { E } \left\| T - \pmb { \theta } \right\| _ { 2 } ^ { 2 } \geq \frac { p ^ { 2 } } { \mathbb { E } _ { \pmb { \theta } } \operatorname { T r } \left( \mathbb { Z } ( \pmb { \theta } ) \right) + I ( \pi ) } ,
$$

as claimed.

## B Measurable projection onto low-rank states

The upper bound estimator chooses a state nearest in Frobenius norm to a Hermitian matrix. The following standard consequence of the measurable maximum theorem [AB06, Theorem 18.19] ensures that ties can be resolved measurably.

Lemma B.1 (Measurable nearest state selection). For every $1 \leq r \leq d ,$ there is a Borel map from the Hermitian matrices on $\mathbb { C } ^ { d }$ to $\mathcal { D } _ { r } ( \mathbb { C } ^ { d } )$ that assigns to each Hermitian matrix M a minimizer of $\| M - \sigma \| _ { \mathrm { F } }$ over $\sigma \in { \mathcal { D } } _ { r } ( \mathbb { C } ^ { d } )$

Proof. Apply the measurable maximum theorem to the constant compact correspondence $M \mapsto$ $\mathcal { D } _ { r } ( \mathbb { C } ^ { d } )$ and the continuous objective $( M , \sigma ) \mapsto - \| M - \sigma \| _ { \mathrm { F } }$ . The theorem gives a Borel measurable selection of minimizers. □