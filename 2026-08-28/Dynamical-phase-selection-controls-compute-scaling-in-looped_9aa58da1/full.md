# Dynamical phase selection controls compute scaling in looped transformers

Gunn Kim<sup>1,</sup> <sup>∗</sup>

<sup>1</sup>Department of Physics, Sejong University, Seoul 05006, Republic of Korea (Dated: August 28, 2026)

A looped transformer performs inference by iterating a weight-tied map, making its computation a dynamical process whose cost is set by the resulting inference dynamics. Here we show that networks with identical architecture and objective, trained to identical accuracy, nevertheless realize distinct dynamical phases depending strongly on initialization, and that the bifurcation defining each phase determines how test-time compute scales. The phases are distinguished by their bifurcation mechanisms, including a saddle-node fold and a Neimark–Sacker-type transition to bounded nonstationary motion. In the fold phase, a one-dimensional normal-form reduction predicts both the relaxation-time and spectral-gap amplitudes from local derivatives of the trained map, yielding the parameter-free relation $\tau ( \varepsilon ) [ 1 - \lambda _ { \operatorname* { m a x } } ( - \varepsilon ) ]  \pi ,$ . Composed with a regular distribution of problem dificulty, the same critical slowing down produces the workload-level tail $P ( \tau > N ) \sim N ^ { - \bar { 2 } }$ . In the Neimark–Sacker phase, the fold scaling law disappears rather than merely changing its prefactor. Thus, test-time compute is not determined by architecture alone. It is governed by the dynamical phase of the solution found by training.

Introduction.—A looped transformer applies one weight-tied block recurrently to arbitrary depth [1–3], an architecture now scaled into language models that convert test-time iteration into reasoning performance [4–6]. Stripping the model to its recurrence makes inference explicit,

$$
z _ { t + 1 } = F _ { \theta } ( z _ { t } ; h ) ,\tag{1}
$$

with the iteration count an operational relaxation time. This raises a basic question: what determines how long the computation takes? Two aspects have received attention: what training makes the network compute, and whether the recurrence remains stable, often through spectral conditions on its linearization [6–9]. Neither fixes the cost of inference, which is set by the fixed-point structure the network relaxes toward and the bifurcation through which a decision is made.

That the linear condition is not the whole story is known: a spectral bound $\rho ( { \overline { { A } } } ) < 1$ on the linear part does not certify convergence of the full map, and a minority of randomly initialized instances circulate on bounded orbits without settling [8]. Our networks satisfy $\rho ( \overline { { A } } ) = 0 . 9 0 < 1$ by construction and nonetheless have a leading multiplier of the full Jacobian that reaches or crosses unity at a finite field. We show that these behaviors are not failures but distinct dynamical phases selected by training, and that the phase determines the scaling law of inference time.

Setup.—We iterate a weight-tied pre-norm attentionplus-MLP block on a latent state $z \in \mathbb { R } ^ { d }$ with one input token carrying the evidence $h ,$ residual scale $\rho ,$ and latent decay γ. The decay permits isolated fixed points: without it, normalization leaves fewer efective degrees of freedom than the number of fixed-point conditions, making an isolated solution nongeneric [10]. Networks are trained end to end with a stationarity penalty to decide sign(h) from evidence withdrawn mid-episode in a fraction µ of trials, so the decision must be maintained without support. The objective specifies neither a fixed point nor a bifurcation nor an iteration count. We sweep $d = 1 6 , K = 4 8 , \mu \in \{ 0 . 2 0 , \ldots , 0 . 8 0 \}$ , and three initializations. All fifteen networks reach held-out accuracy 1.000, with final losses between $1 . 9 \times 1 0 ^ { - 5 }$ and $3 . 6 \times 1 0 ^ { - 4 }$ at the level of task performance, they are indistinguish able.

Phases of inference.—We classify each network by the fate of the fixed point of the disfavored decision as h increases, tracking the leading nontrivial multiplier after removing the exact symmetry mode. We locate a real unit-multiplier crossing by solving

$$
F _ { \theta } ( z ; h ) = z , \qquad ( J - \mathbb { I } ) v = 0 , \qquad v ^ { \top } v = 1 ,
$$

and identify a network as Neimark–Sacker only when no fold is found by the extended system, the fixed-point branch persists, and a complex-conjugate pair crosses the unit circle [10]. Three principal phases emerge. In the fold phase, a real multiplier reaches unity as the fixed point collides with a saddle and both disappear. In the Neimark–Sacker phase, a complex-conjugate pair crosses the unit circle, destabilizing the fixed point while leaving bounded nonstationary dynamics. In the rigid phase, the bistable branch persists throughout the explored field range. Monostable and non-generic cases occur as additional outcomes [10].

Figure 1(a) summarizes the phase assignments across memory pressure and initialization. The phase is strongly initialization-dependent but only weakly dependent on the objective. The phase difers across seeds at every value of $\mu ,$ whereas varying $\mu$ at fixed initialization changes the classification only twice across the sweep. At fixed $\mu = 0 . 5 0$ , ten independent initializations yield five folds, one Neimark–Sacker, one rigid, and three monostable networks, all with accuracy 1.000 [10]. Checkpoints further show that the phase becomes identifiable early in training: eight of ten networks are already in their final phase by step 250; one additional network subsequently changes locally from a non-generic crossing to Neimark– Sacker, while none switches directly between fold and Neimark–Sacker, despite a further 4–24-fold reduction in loss after the phase is established [10]. Thus, task performance does not resolve the dynamical distinction: training finds multiple solutions with essentially identical objective values but qualitatively diferent inference dynamics.

What symmetry constrains.—LayerNorm is invariant under $z  z + c { \bf 1 }$ , so a perturbation along 1 survives normalization only through the residual decay, giving exactly

$$
J { \bf 1 } = ( 1 - \gamma ) ^ { 2 } { \bf 1 }\tag{2}
$$

at every state and field [10]. This symmetry mode carries no information about the solution and is therefore removed before the remaining spectrum is analyzed; otherwise $( 1 - \gamma ) ^ { 2 }$ can appear among the leading multipliers away from the fold. The Jacobians of the nonlinearities are symmetric, so antisymmetry, and hence rotation, enters through compositions of the learned weight matrices. Looped recurrences are known to exhibit substantial nonnormality and rotation even at random initialization [8]; here the decomposition shows that their magnitude and direction vary across initializations of the same architecture.

These constraints narrow the bifurcations observed in our sweep. Because the map is real, multipliers are real or occur in conjugate pairs. No architectural symmetry protects a pitchfork, and we find no transcritical crossings; period doubling is also absent, with the closest approach satisfying $| \lambda + 1 | = 1 . 7 2 1 4$ The observed codimension-one bifurcations are therefore folds and Neimark–Sacker crossings, apart from occasional non-generic crossings [10]. This is a classification of the models explored, not an exclusion theorem.

The decisive quantity is not the spectrum at the resting attractor but its evolution with the control field. Restingstate rotation does not separate the phases: the antisymmetric fraction of the Jacobian overlaps between the fold and Neimark–Sacker networks, spanning [0.067, 0.140] and [0.098, 0.190], respectively [10]. Likewise, a small imaginary part at rest does not preclude a fold, because a complex pair can collide on the real axis as h increases. We therefore classify the phase from the continued Jacobian spectrum rather than from any scalar descriptor at rest. What matters is not how much rotation the map carries, but whether it carries rotation in the critical direction.

Inside the fold phase, an efective theory with no $f i t -$ ted parameters.—At a generic fold, one multiplier reaches unity while the remaining multipliers stay inside the unit circle. The center manifold theorem for maps [11, 12] therefore reduces the dynamics to

$$
u _ { t + 1 } = u _ { t } + g \varepsilon + b u _ { t } ^ { 2 } , \qquad \varepsilon = h - h _ { s n } ,
$$

where $u = l ^ { \top } ( z - z _ { s n } )$ and

$$
\begin{array} { r } { g = l ^ { \top } \partial _ { h } F _ { \theta } , \qquad b = \frac { 1 } { 2 } l ^ { \top } \partial ^ { 2 } F _ { \theta } [ r , r ] , } \end{array}\tag{3}
$$

with $l ^ { \top } r = 1$ . Both coeficients are determined directly from the trained map; choosing the orientation such that $b > 0$ gives $g > 0$ for every fold network [10].

For $\varepsilon > 0 ;$ the continuum limit gives $\dot { u } = g \varepsilon + b u ^ { 2 }$ the normal form of Type-I intermittency [13]. Passage an equal distance ε above the fold and critical slowing on the stable branch an equal distance below it give, respectively,

$$
\tau ( \varepsilon ) = \frac { \pi / 2 } { \sqrt { g b } } \varepsilon ^ { - 1 / 2 } , \qquad 1 - \lambda _ { \mathrm { m a x } } ( - \varepsilon ) = 2 \sqrt { g b } \varepsilon ^ { 1 / 2 } .\tag{4}
$$

The factor $\pi / 2 ,$ , rather than $\pi ,$ reflects the one-sided passage from the fold state; the finite-threshold correction is below $0 . 1 5 \%$ at $\varepsilon = 1 0 ^ { - 6 } \ [ 1 0 ]$ . Thus the exponents and amplitudes are fixed by the same two local derivatives. Evaluating the two relations at equal distances on opposite sides of the fold eliminates $g$ and b:

$$
\Pi ( \varepsilon ) \equiv \tau ( \varepsilon ) \big [ 1 - \lambda _ { \mathrm { m a x } } ( - \varepsilon ) \big ] = \pi + O ( \varepsilon ^ { 1 / 2 } ) .\tag{5}
$$

Unlike the exponents, which follow directly from the normal form, this amplitude relation provides a parameterfree test of the reduction itself. A map could reproduce the exponents while failing this amplitude relation. Across seven fold-phase networks spanning $d = 1 6 { - } 2 0$ ， $K = 2 4$ and 48, both protocols, and $h _ { s n } = 0 . 0 2 1 \ – 0 . 2 1 5$ all predictions agree with the measurements to within a few percent [10]; Fig. 2 and Table I summarize the tests.

The cost of computation follows the phase.—A budget is spent across a population of problems. For a dificulty density $p$ that is smooth and nonzero at the fold, inverting the first relation in Eq. (4) gives

$$
p ( \tau ) \sim 2 A ^ { 2 } p ( 0 ) \tau ^ { - 3 } , \qquad A = \frac { \pi / 2 } { \sqrt { g b } } ,
$$

TABLE I. Parameter-free tests over seven fold-phase networks spanning $d = 1 6 { - } 2 0$ $K = 2 4$ and 48, both protocols, and $h _ { s n } ~ = ~ 0 . 0 2 1 – 0 . 2 1 5 .$ Uncertainties are standard deviations across networks [10].
<table><tr><td>quantity</td><td>predicted</td><td>measured</td></tr><tr><td>branch exponent</td><td> $1 / 2$ </td><td> $0 . 4 9 9 9 \pm 0 . 0 0 1 4$ </td></tr><tr><td>slowing exponent</td><td> $1 / 2$ </td><td> $0 . 5 0 0 4 \pm 0 . 0 0 1 9$ </td></tr><tr><td>slowing amplitude  $/ 2 { \sqrt { g b } }$ </td><td>1</td><td> $1 . 0 0 1 6 \pm 0 . 0 0 8 2$ </td></tr><tr><td>passage exponent</td><td> $- 1 / 2$ </td><td> $- 0 . 4 9 9 7 \pm 0 . 0 0 9 2$ </td></tr><tr><td>passage prefactor  $/ \frac { \pi / 2 } { \sqrt { g b } }$ </td><td>1</td><td> $1 . 0 0 6 3 \pm 0 . 0 2 5 5$ </td></tr><tr><td>Ⅱ at  $\varepsilon = 1 0 ^ { - 6 }$ </td><td> $\pi = 3 . 1 4 1 6$ </td><td> $3 . 1 4 6 \pm 0 . 0 1 7$ </td></tr><tr><td>tail exponent, least squares</td><td>-2</td><td> $- 2 . 1 3 0 \pm 0 . 1 9 6$ </td></tr><tr><td>tail exponent, Hill</td><td>2</td><td> $2 . 1 0 8 \pm 0 . 1 8 3$ </td></tr></table>

![](images/3b5a1115c9faf7c5df9fa4cdd4845493d8f7246c01c4a0694ccdaeb014e273b0.jpg)  
FIG. 1. (a) Phase across memory pressure and initialization, fifteen networks. (b) |Im $\lambda _ { \operatorname* { m a x } } |$ along the branch, with the symmetry mode removed; fold networks are solid. (c) $\lambda _ { \mathrm { m a x } }$ in the complex plane for one fold and one Neimark–Sacker network.

and hence

$$
P ( \tau > N ) \sim N ^ { - 2 } .\tag{6}
$$

The universality is that of the fold normal form combined with the regularity of the workload density, not of the network itself. For the uniform ensembles used here, $\varepsilon \sim$ $\mathcal { U } ( 0 , \varepsilon _ { \mathrm { m a x } } )$ and $p ( 0 ) = 1 / \varepsilon _ { \mathrm { m a x } } ,$ fixing the amplitude as

$$
P ( \tau > N ) = \left( \frac { N ^ { * } } { N } \right) ^ { 2 } , \qquad N ^ { * } = A \varepsilon _ { \mathrm { m a x } } ^ { - 1 / 2 } .
$$

Across $8 . 4 \times 1 0 ^ { 4 }$ problems in seven networks, with no censoring, the distributions collapse onto $( N / N ^ { * } ) ^ { - 2 }$ with unit amplitude. At $N = 1 0 N ^ { * }$ , the predicted unsolved fraction $\mathrm { \bar { 1 0 } ^ { - 2 } }$ is measured as 0.0070–0.0110 [Fig. 2(c)].

For this decision task, $P ( \tau > N )$ is the residual fraction of problems unsolved at budget N and therefore the budget-induced error if an unresolved trajectory is counted as an error. The tail also implies

$$
\langle \tau | \tau > N \rangle \sim { 2 N } ,
$$

so the expected remaining compute remains of the same order as that already spent. Moreover, $\langle \tau \rangle$ is finite while $\left. \tau ^ { 2 } \right.$ diverges logarithmically with the cutof, placing the workload at the boundary between finite and divergent compute fluctuations: rare hard instances dominate the variance.

In the Neimark–Sacker phase, the fold law disappears.—Changing phase does not merely change the prefactor of the fold law; it removes the underlying fixedpoint-annihilation mechanism. In the Neimark–Sacker network, the disfavored fixed point is not destroyed: a complex-conjugate pair crosses the unit circle at $h _ {  { N }  { S } } =$ 0.0386 with multiplier $0 . 9 8 8 6 + 0 . 1 5 0 4 i$ , an angular increment of 0.151 radians per iteration, or about 42 iterations per turn. Past the crossing the fixed point remains present but unstable, and trajectories launched from the branch state remain bounded and nonstationary over the integration interval [10]. Scanning 200 fields above h<sub>NS</sub>, the 35 non-escaping fields form one contiguous low-field interval ending at $h = 0 . 0 4 9 7$ , followed by a single transition into the escaping regime [Fig. 3]. Escape therefore begins at a field 29% above $h _ { N S }$ , well separated from the crossing, and we assign no fold-like power-law exponent to this phase. The fixed point persists, and escape is controlled by the subsequent loss of the bounded nonstationary state rather than by passage through a saddle-node bottleneck. Thus the fold-derived compute-time law has no direct analogue in this network. We analyze escape only for this phase; the rigid and monostable networks have no fold at which the same slowing law can arise. Predictability of cost is itself a property of the dynami cal phase.

Discussion.—That a trained network organizes inference around a bifurcation is not surprising; attractor models of decision making have been built this way for decades [14–17], and networks trained on a single task can realize it through distinct mechanisms whose selection depends on initialization [18–21]. What is new here is that these alternatives are distinct bifurcation phases, and that the selected phase determines the computational time scale. This sharpens rather than contradicts the observation that trained recurrent networks can share dynamical topologies [22]: such sharing can hold within a phase even when the phase itself is not fixed by the task. It also extends the implicit bias identified for looped linear transformers with normalization [23]: optimization can favor a dynamical class, but that class need not be determined by the task.

From a physics perspective, optimization acts as a quench that selects among dynamically inequivalent solutions that are indistinguishable at the level of task performance, with symmetry constraining the available structure rather than selecting the phase. Once selected, the phase admits a remarkably sharp description. In the fold phase, a nonlinear map produced by gradient descent, whose objective contains no bifurcation or compute-time constraint, is quantitatively captured by the saddle-node normal form: both critical exponents and amplitudes are determined by local derivatives of the trained map, including the parameter-free relation of Eq. (5). This is normal-form universality, arising from the structural stability of the fold under smooth coordinate changes, rather than fluctuation-driven universality of a thermodynamic transition. There is no diverging correlation length or thermodynamic limit; the exponent $1 / 2$ follows directly from the local Taylor expansion.

(a)  
![](images/e4b7c2b9325759f09d5a7f2f8f21d29d97a03a55c91795da2d76a4fae905561e.jpg)

(b)  
![](images/42c3fe3ade3a6bb57aac5d36f595136f811a85f0f9b047c8a8b8e53cbcf11917.jpg)

![](images/134d2c85ff96d8aef8d42670e096875c1f3d8765b87c93a01e4364ad6f4cb044.jpg)  
FIG. 2. Seven fold-phase networks, each rescaled by its predicted amplitude; dashed lines carry no free parameter. (a) Node (filled) and saddle (open) branches; inset, sign retained. (b) Critical slowing down; inset, Π(ε) of Eq. (5). (c) Complementary cumulative distributions of $\tau .$

(a)  
![](images/82f9e5ee85796a5c75d930bfecbdd49342687a6584c2a548deaed3fbc7060dac.jpg)

(b)  
![](images/19ecdde537515cb856fcf6f9b3699fa4e214108b9b439ce5bcf789d2c239ccb0.jpg)  
FIG. 3. (a) Escape time against distance to the critical field for a fold network (blue; dashed, Eq. (4)) and a Neimark– Sacker network (orange); crosses mark fields with no escape within $2 . 5 \times 1 0 ^ { 5 }$ iterations. (b) The Neimark–Sacker scan on a linear field axis. The non-escaping fields form a contiguous low-field interval and escape begins at a single onset near $h = 0 . 0 4 9 9$ , well above the crossing $h _ { N S } = 0 . 0 3 8 6$ loss of fixed-point stability and escape are distinct events.

The fold normal form further converts single-problem critical slowing down into an ensemble theory of compute. Combining $\tau \sim \varepsilon ^ { - 1 / 2 }$ with a workload density regular at the bottleneck gives $P ( \tau > N ) \sim N ^ { - 2 }$ , placing the workload at the boundary of finite compute fluctuations. The ingredients are the local normal form and the regularity of the dificulty distribution, not the details of the neural network. The same composition should therefore apply to other iterative computations whose instances approach a bottleneck with varying distance to criticality. By contrast, the Neimark–Sacker phase lacks a fixed-point annihilation and, in our data, no analogous single-parameter critical compute-time scaling emerges. Predictability of test-time compute is therefore itself a property of the dynamical phase.

This perspective also clarifies the relation to stability analyses. Previous work asks whether the recurrence settles and shows that a spectral bound on the linear part does not by itself answer that question [7, 8]; we ask what the full nonlinear map does as it approaches a bifurcation, and how the resulting relaxation time scales. The non-settling trajectories reported at random initialization [8] are consistent with the nonstationary dynamics of our Neimark–Sacker phase, but our results show that such dynamical alternatives can persist after training and depend strongly on initialization. This also reconciles the saturating exponential observed in test-time curves [7], associated with the spectral radius of the full Jacobian [8], with the power-law workload tail found here: exponential relaxation describes the dynamics of a fixed problem, whereas the power law arises from the distribution of relaxation times across problems. The two describe diferent levels of the computation and are

therefore not in conflict.

Several predictions follow that can be tested on existing looped models using per-instance iteration counts [10]. If a workload samples a fold bottleneck for which the leading real multiplier approaches unity and the dificulty density is regular there, the iteration counts should develop the $\dot { N } ^ { - 2 }$ tail even if mean performance retains an approximately exponential dependence on compute. More generally, if the dificulty density behaves as $p ( \varepsilon ) \sim \varepsilon ^ { - a }$ near the bottleneck, the same composition gives $P ( \tau > N ) \sim N ^ { - 2 ( 1 - a ) }$ , or a positive tail index $\theta = 2 ( 1 - a )$ when $P ( \tau > N ) \sim N ^ { - \theta } ;$ ; thus the measured tail probes the distribution of distances to criticality. Architectures with $\rho ( { \overline { { A } } } ) = 1$ by construction [7] provide a natural setting for this test, although marginal linear residual dynamics does not by itself imply a fold. Finally, within our bistable ensemble, every network whose lead ing nontrivial multiplier was exactly real at the resting attractor belonged to the fold phase, although not every fold was real at rest. Since the $N ^ { - 2 }$ law was observed only in the fold phase, the resting spectrum is a candidate diagnostic whose scope must be tested beyond the present ensemble.

Conclusion.—Training determines what a looped transformer computes, while spectral conditions on its linearized recurrence constrain the stability of its residual dynamics. Neither, however, determines how inference unfolds. That is set by the dynamical phase selected by training: networks with the same architecture, objective, and task performance can relax through qualitatively diferent bifurcations, with the choice depending strongly on initialization and becoming identifiable when the memory–attractor structure forms. Symmetry constrains the available phases but does not select among them. In the fold phase, the resulting dynamics obey a parameter-free critical scaling that extends from singleproblem relaxation to a universal workload tail, whereas the Neimark–Sacker phase has no analogous computetime law. Test-time compute is therefore not determined by the architecture alone; it is a dynamical property of the solution that training finds.

Data availability.—The data and code that support the findings of this study are available from the corresponding author upon reasonable request.

∗ Corresponding author: gunnkim@sejong.ac.kr

[1] M. Dehghani, S. Gouws, O. Vinyals, J. Uszkoreit, and

L. Kaiser, in Proc. ICLR (2019), arXiv:1807.03819.

[2] A. Giannou, S. Rajput, J.-y. Sohn, K. Lee, J. D. Lee, and D. Papailiopoulos, in Proc. ICML (2023), p. 11398.

[3] N. Saunshi, N. Dikkala, Z. Li, S. Kumar, and S. J. Reddi, in Proc. ICLR (2025), arXiv:2502.17416.

[4] J. Geiping et al., in Adv. Neural Inf. Process. Syst. 38, 41340 (2025), arXiv:2502.05171.

[5] A. Jeddi, M. Ciccone, and B. Taati, in Proc. ICLR (2026), arXiv:2602.11451.

[6] S. Movahedi, V. Milovanovi´c, S. L. Feigin, A. Theus, T. Hofmann, V. Boeva, T. K. Rusch, and A. Orvieto, arXiv:2606.18206 (2026).

[7] H. Prairie, Z. Novack, T. Berg-Kirkpatrick, and D. Y. Fu, arXiv:2604.12946 (2026).

[8] M. M. M. Buehlmaier, arXiv:2607.10681 (2026).

[9] A. Labovich, arXiv:2604.15259 (2026).

[10] See Supplemental Material for architecture and training, the phase classification protocol and checkpoint statistics, symmetry constraints on the linearization, fold location and saddle continuation, the normal-form reduction and its parameter-free tests, ensemble construction and tail estimation, the Neimark–Sacker escape scan, the monostable and non-generic cases, the one-step forecast, finite-noise scaling, and a protocol for measuring the tail exponent on a large looped model, which includes Refs. [11, 12, 24, 25].

[11] J. Guckenheimer and P. Holmes, Nonlinear Oscillations, Dynamical Systems, and Bifurcations of Vector Fields (Springer, New York, 1983).

[12] Y. A. Kuznetsov, Elements of Applied Bifurcation Theory, 3rd ed. (Springer, New York, 2004).

[13] Y. Pomeau and P. Manneville, Commun. Math. Phys. 74, 189 (1980).

[14] K.-F. Wong and X.-J. Wang, J. Neurosci. 26, 1314 (2006).

[15] X.-J. Wang, Neuron 60, 215 (2008).

[16] V. Mante, D. Sussillo, K. V. Shenoy, and W. T. Newsome, Nature (London) 503, 78 (2013).

[17] D. Sussillo and O. Barak, Neural Comput. 25, 626 (2013).

[18] E. Ghazizadeh and S. Ching, PLoS Comput. Biol. 17, e1009366 (2021).

[19] C. Jarne, Cogn. Neurodyn. 17, 257 (2023).

[20] C. Jarne and R. Laje, J. Comput. Neurosci. 51, 407 (2023).

[21] E. Turner, K. Dabholkar, and O. Barak, in Adv. Neural Inf. Process. Syst. 34 (2021), arXiv:2111.09356.

[22] N. Maheswaranathan, A. Williams, M. Golub, S. Ganguli, and D. Sussillo, in Adv. Neural Inf. Process. Syst. 32 (2019).

[23] L. Wu, C. Zhang, and Y. Cao, arXiv:2606.00605 (2026).

[24] D. Hathcock and J. P. Sethna, Phys. Rev. Research 3, 013156 (2021).

[25] N. Berglund and B. Gentz, Noise-Induced Phenomena in Slow-Fast Dynamical Systems (Springer, London, 2006).