# A THEORY OF NOISY LINEAR RECURRENT NEURAL NETWORK LEARNING DYNAMICS

John J. Vastola, Kanaka Rajan

Department of Neurobiology, Harvard Medical School

Kempner Institute for the Study of Natural and Artificial Intelligence, Harvard University john vastola, kanaka rajan @hms.harvard.edu

## ABSTRACT

Biological neural networks are noisy: their dynamics involve difficult-to-predict fluctuations which are likely due not just to chaos, but also to a variety of low-level biophysical processes. Despite the omnipresence of intrinsic noise, its theoretical consequences are incompletely understood, especially in the context of learning. For example, does noise meaningfully shape representation learning, or is it better viewed as a nuisance factor whose presence or absence changes little about what solution is found and when? Here, we argue that intrinsic noise can significantly modulate representation learning in recurrent neural networks (RNNs), since it can strongly break solution-space degeneracies and in some cases control whether a non-degenerate solution exists at all. We show that this is true for an analytically tractable model of noisy RNN learning with a linear activation function and white noise inputs. We find that, in the so-called rich regime (where weights are initialized with small values), noise strongly modulates the directions in which the read-in, readout, and recurrent weights align. Moreover, we find that noise produces a kind of phase transition: for sufficiently high noise, a non-degenerate solution is not even optimal if parameter scales are limited (e.g., via regularization). These results extend recent analytic work on RNN representation learning and show that noise can act as a fundamental determinant of representations.

## 1 INTRODUCTION

Recurrent neural networks (RNNs) are extraordinarily successful models of the brain (Vogels et al., 2005; Mante et al., 2013; Rajan et al., 2016; Vyas et al., 2020), and the past two decades of theory and experiment have produced a working understanding of what they learn and how. Their loss landscapes are generally more complex than those of feedforward networks, in part because recurrence can produce exploding and vanishing gradients (Bengio et al., 1994; Pascanu et al., 2013); their solution space is generally highly degenerate (Huang et al., 2025); and they can be biased towards learning low-rank solutions (Schuessler et al., 2020). But a major feature of RNNs that remains poorly understood is the significance of intrinsic noise.

Noise is omnipresent in biological neural networks: spike trains and firing rate traces throughout the nervous system involve difficult-to-predict fluctuations (Faisal et al., 2008), which may be a manifestation of chaotic behavior (Sompolinsky et al., 1988; Crisanti & Sompolinsky, 2018) or low-level stochasticity associated with chemical reactions. Although from one perspective noise is a nuisance, from another it provides pressure for networks to find more robust solutions. Noise is routinely used in RNN training (Yang et al., 2019; Driscoll et al., 2024), in part to improve generalization, and there is evidence that noise can drive networks into flatter loss landscape minima (Lim et al., 2021).

From a representation-learning perspective, it may be helpful to view noise as having two potential consequences. First, noise can provide geometric constraints on solutions: weights must be oriented in certain directions (i.e., the less noisy directions) and not others, and their overall scales must be chosen to minimize noise. Second, the goal of minimizing noise may compete with that of achieving good task performance; if there is too much noise, one expects that the system may be driven to a trivial solution at which task performance is poor, but there is little or no noise.

Although RNN learning dynamics are difficult to study, an interesting toy model of them has emerged in recent work: linear RNNs, especially linear RNNs trained to perform leaky integration of a white noise stimulus (Proca et al., 2025; Bordelon et al., 2025). As Bordelon et al. (2025) show, when network parameters are initially small, learning has two phases: an alignment phase, in which parameter movement concentrates along a specific (initialization-dependent) direction of state space, followed by a dynamics phase, in which an eigenvalue ‘escapes’ from the bulk in order to solve the task.

In this work, we consider an extremely similar theoretical setup, whose primary difference is the presence of (generally anisotropic) intrinsic noise. We show that their theory must be substantially modified even if there is only a small amount of noise, since noise affects both the alignment and dynamics phases, and even whether a nontrivial solution exists at all. Throughout, we focus on characterizing learning in the rich regime, as opposed to the lazy regime, in part because the rich regime is thought to be more representative of learning in performant RNNs.

Our main contributions are as follows.

• We develop an analytically tractable theory of representation learning in linear RNNs with intrinsic noise. We show that noise breaks degeneracies of the original objective, which yields preferred state space directions and read-in/readout scales.

• We characterize how intrinsic noise modifies rich-regime learning dynamics. During the alignment phase, anisotropic noise changes the growth rates of different state-space modes, which biases learning toward less noisy directions. Noise also breaks the symmetry between the read-in and readout weights, and in the absence of regularization, drives the <sub>input weight scale to infinity and the output weight scale to zero.</sub>5 σ = 1 σ = 3 σ = 5 τ σ = σ<sub>crit</sub> c/c /τ σ c<sub>op</sub>2 2 2 2 2 2 2 2 2 2 2

• We show that sufficiently strong noise can produce phase transitions beyond which2<sup>σ = 0.5</sup> <sup>σ = 1</sup> <sup>σ = 3</sup> <sup>σ = 5</sup> <sup>τ σ =</sup> <sup>σcrit</sup> <sup>c/c</sup>∗<sup>/τ</sup> <sup>σ copt</sup>t only a trivial solution is learned. This is true both when all parameters are trainable and  <sup>crit</sup>  <sub>opt</sub> σ κ N<sub>opt</sub> = 0N N<sub>opt</sub> σ κ N<sub>opt</sub> = 0o <sub>when some parameters are frozen.</sub>c/c /τ σ c<sub>opt</sub>N Nopt σ<sub>crit</sub> κ N    2

## 2 MATHEMATICAL FORMULATIONN N σ2 κ N = 0pt σ<sup>2</sup> κ Nopt = 0<sup>λ</sup>i <sup>s</sup>i <sup>W</sup> |<sup>u</sup>|<sup>/</sup>|<sup>v</sup>| <sup>s</sup>i <sup>= 0.5</sup> <sup>s</sup>i <sup>= 1</sup> <sup>s</sup>i u / v s = 0.5 s = 1 s = 3 s =J Jnoise c u v c/cλ+ s W u / v s = 0.5 s = 1 s/<sup>c</sup>2 J c σ<sup>2</sup> =

![](images/f6128c67636dd40e8fa20d8a5fe1bedc4c8d84cde76e0adca7f2b71135d0607b.jpg)

![](images/7a05cfdc66eb6c3f2178ddc7499b5d0e43072a6fdf44e55ab0a4ce7f1213c773.jpg)

![](images/d590c3be6d7a4582573c8f5a79af8d88fb8b43037abbdb4013484e39ded50720.jpg)

![](images/dfa25c440c1db37ae66e2e9818295a9f10f2f273ed36f89a64b3dac2c425b5f8.jpg)  
<sup>σ</sup> o<sup>p</sup> <sub>e</sub><sup>r</sup> | t<sup>)</sup>Figure 1: Theoretical setting. (a) We study a noisy linear RNN which receives a scalar white noise fi<sup>l</sup>signal x(t) as input and produces an output $y ( t )$ <sup>y</sup>. The dynamics of individual $r _ { i }$ are noisy (bottom). u <sub>(</sub>t(b) On average, the network filters its input. (c) Output variance is input-independent and eventually <sup>σ x</sup>saturates (assuming the network is stable). (d) We assume the network is trained to optimize an <sup>.</sup> objective with three terms: a filter-matching term that incentivizes matching its average behavior to that of a target filter $f _ { * } ( t )$ = <sup>s</sup>, a noise reduction term, and a regularization term.

<sup>0</sup>Our setup largely mirrors that of Bordelon et al. (2025). Throughout this paper, we study a linear RNN with $\dot { N } \ \geq \ 1$ <sup>=</sup>neurons whose instantaneous state is characterized by a vector $r : = { }$ $( r _ { 1 } , . . . , r _ { N } ) ^ { T } \in \mathbb { R } ^ { N }$ σ. We assume dynamics

$$
{ \dot { r } } = A r + x ( t ) u + G \eta ( t ) = - \gamma r + W r + x ( t ) u + G \eta ( t )\tag{1}
$$

where $x ( t )$ is a time-dependent input signal, $\mathbf { \boldsymbol { u } } \in \mathbb { R } ^ { N }$ are the read-in weights, $\gamma > 0$ is the coefficient of the leak term, and $\mathbf { \bar { \boldsymbol { W } } } \in \mathbb { R } ^ { \hat { N } \times N }$ are the recurrent weights (Fig. 1a). To ease notation, we sometimes work with the combination $\pmb { A } : = - \gamma \pmb { I } + \pmb { W }$ . The remaining term instantiates intrinsic noise: $G \in \mathbb { R } ^ { N \times N }$ controls its amplitude, and $\pmb { \eta } ( t ) \in \mathbb { R } ^ { N }$ is a vector of Gaussian white noise processes. We are most interested in understanding how the structure of G shapes learning.

Because Eq. 1 is linear in r and involves only state-independent additive noise, we can immediately write down the distribution of $\mathbf { } _ { \pmb { r } ( t ) }$ conditional on any initial condition, although we will assume $\mathbf { \boldsymbol { r } } ( 0 ) = \mathbf { 0 }$ for simplicity. We find that $\mathbf { } r ( t )$ is normally distributed with

$$
p ( \boldsymbol { r } , t | \boldsymbol { r } _ { 0 } = \mathbf { 0 } , 0 ) = \mathcal { N } \left( \int _ { 0 } ^ { t } e ^ { A t ^ { \prime } } \boldsymbol { u } \boldsymbol { x } ( t - t ^ { \prime } ) d t ^ { \prime } , \int _ { 0 } ^ { t } e ^ { A t ^ { \prime } } S e ^ { A ^ { \prime } t ^ { \prime } } d t ^ { \prime } \right)\tag{2}
$$

where $S : = G G ^ { T }$ is, up to a factor of 2, the diffusion tensor associated with G (which we take to be positive definite). On average, the network linearly filters its input (Fig. 1b). But because this integration is noisy, S-dependent noise accumulates until (provided the eigenvalues of A all have negative real parts) it saturates (Fig. 1c).

## 2.1 OBJECTIVE FUNCTION

Following Bordelon et al. (2025), assume a linear readout determined by readout weights $\mathbf { \pmb { v } } \in \mathbb { R } ^ { N _ { \pmb { \mathrm { \cdot } } } }$

$$
y ( t ) : = v ^ { T } r ( t )\tag{3}
$$

This readout defines a filter $f ( t ) : = e ^ { - \gamma t } v ^ { T } e ^ { W t } $ u since, on average,

$$
\mathbb { E } _ { \boldsymbol { r } } [ \boldsymbol { y } ( t ) ] = \int _ { 0 } ^ { t } \boldsymbol { v } ^ { T } e ^ { \boldsymbol { A } t ^ { \prime } } \boldsymbol { u } \boldsymbol { x } ( t - t ^ { \prime } ) d t ^ { \prime } = \int _ { 0 } ^ { t } f ( t ^ { \prime } ) \boldsymbol { x } ( t - t ^ { \prime } ) d t ^ { \prime } .\tag{4}
$$

Assume we train some combination of ${ \mathbf { } } u , v ,$ , and $W$ so that $y ( t )$ matches a target output

$$
y _ { * } ( t ) : = \int _ { 0 } ^ { t } e ^ { - c _ { * } t ^ { \prime } } x ( t - t ^ { \prime } ) d t ^ { \prime } = \int _ { 0 } ^ { t } f _ { * } ( t ^ { \prime } ) x ( t - t ^ { \prime } ) d t ^ { \prime }\tag{5}
$$

which linearly filters its input $x ( t )$ using the target filter $f _ { * } ( t ) : = e ^ { - c _ { * } t }$ . Here, $c _ { * } > 0$ controls the time scale on which input is leaky-integrated. Assume we do this using the supervised objective

$$
J : = \operatorname* { l i m } _ { T \to \infty } \frac { 1 } { T } \int _ { 0 } ^ { T } \frac { 1 } { 2 } \mathbb { E } _ { x , r } \{  [ y ( t ) - y _ { * } ( t ) ] ^ { 2 } \} d t + \frac { \kappa } { 2 } \| \pmb { u } \| ^ { 2 } + \frac { \kappa } { 2 } \| \pmb { v } \| ^ { 2 } ,\tag{6}
$$

which demands that the network output $y ( t )$ agrees with $y _ { * } ( t )$ on average. Note that we have also included a regularization term whose strength is determined by $\kappa \geq 0 .$ . This objective can be simplified if the input has white noise statistics (Proca et al., 2025; Bordelon et al., 2025), i.e., if $\mathbb { E } _ { x } [ x ( t ) ] = 0$ and $\dot { \mathbb E } _ { x } [ x ( t ) x ( t - t ^ { \prime } ) ] = \delta ( t - t ^ { \prime } )$ . We find (see Appendix A)

$$
\boxed { J ( u , v , W ) = \int _ { 0 } ^ { \infty } \frac { 1 } { 2 } \left[ f ( t ) - f _ { * } ( t ) \right] ^ { 2 } + \frac { e ^ { - 2 \gamma t } } { 2 } v ^ { T } e ^ { W t } S e ^ { W ^ { T } t } v ~ d t + \frac { \kappa } { 2 } \| u \| ^ { 2 } + \frac { \kappa } { 2 } \| v \| ^ { 2 } } .\tag{7}
$$

Note that J has three terms: (i) a filter-matching term $J _ { \mathrm { f l l t e r } } ,$ , (ii) a noise term $J _ { \mathrm { n o i s e } }$ , and (iii) a regularization term $J _ { \mathrm { r e g } }$ (Fig. 1d). Bordelon et al. (2025) only considered the filter-matching term; we will see that the other two terms strongly modulate learning dynamics.

## 2.2 LEARNING DYNAMICS

RNNs are normally trained via backpropagation through time (Werbos, 1990), FORCE (Sussillo & Abbott, 2009), or real-time recurrent learning (Williams & Zipser, 1989; Irie et al., 2024). One virtue of linear RNNs, at least for theory, is that gradients of J with respect to u, v, and W can be computed exactly, so we can ignore the extra structure associated with those learning rules. Assuming parameters are updated via gradient descent, in the gradient-flow limit in which the learning rate is taken infinitesimally small, we recover ordinary differential equations (ODEs) (see Appendix A)

$$
\begin{array} { l } { \displaystyle \frac { d } { d \tau } u = \int _ { 0 } ^ { \infty } [ f _ { * } ( t ) - f ( t ) ] e ^ { - \gamma t } e ^ { W ^ { T } t } v \ d t - \kappa u } \\ { \displaystyle \frac { d } { d \tau } v = \int _ { 0 } ^ { \infty } [ f _ { * } ( t ) - f ( t ) ] e ^ { - \gamma t } e ^ { W t } u - e ^ { - 2 \gamma t } e ^ { W t } S e ^ { W ^ { T } t } v \ d t - \kappa v } \\ { \displaystyle \frac { d } { d \tau } W = \int _ { 0 } ^ { \infty } [ f _ { * } ( t ) - f ( t ) ] e ^ { - \gamma t } \int _ { 0 } ^ { t } e ^ { W ^ { T } t ^ { \prime } } v u ^ { T } e ^ { W ^ { T } ( t - t ^ { \prime } ) } \ d t ^ { \prime } d t } \\ { \displaystyle \qquad - \int _ { 0 } ^ { \infty } e ^ { - 2 \gamma t } \int _ { 0 } ^ { t } e ^ { W ^ { T } t ^ { \prime } } v v ^ { T } e ^ { W t } S e ^ { W ^ { T } ( t - t ^ { \prime } ) } \ d t ^ { \prime } d t } \end{array}\tag{8}
$$

parameterized by a real-valued ‘learning time’ $\tau \in [ 0 , \infty )$ . These dynamics are generically highly nonlinear and high-dimensional, although we will see shortly that they simplify substantially if the3 σ<sup>2</sup> = 5 τ σ<sup>2</sup> = σ<sup>2</sup> c/c /τ σ<sup>2</sup> c <sub>initial weights are assumed to be small.</sub> ∗

## <sup>3</sup> <sup>NOISE</sup> <sup>PRODUCES</sup> <sup>REPRESENTATIONAL</sup> <sup>PRESSURE</sup>J c σ<sup>2</sup> = 0 σ<sup>2</sup> = 0.5

![](images/09b5f9b8d8e4bc41e46a4d9abda9643bb3e0e613af9ce5eac19f4c2d93211a08.jpg)

![](images/c3ca86abf2c2a19029e111b2c36ee70cb38d84c410b160761d343ebdffb6a752.jpg)

![](images/7f6b988e05f87d09220da5cd3e24c86251814c3239a3b82dbd06a2a4fc87725a.jpg)

![](images/094b73e783a4f559ec48f3b58d45b7f7d2b736d3e02e00425983a3ea5da9aa9b.jpg)  
Figure 2: Noise produces representational pressure. (a) Two representations with the same filter but different readout vectors differ in noise level if S is anisotropic. Pictured is a schematic 2D example with a high-noise direction (horizontal) and a low-noise direction (vertical). One can reduce noise by rotating v (white arrow). (b) The noise term $J _ { \mathrm { n o i s e } }$ in the schematic example as a function of v orientation. $J _ { \mathrm { f i l t e r } }$ is unaffected as long as the filter remains the same. (c) $J _ { \mathrm { n o i s e } }$ and $J _ { \mathrm { f i l t e r } }$ as a function of α (see main text); a zero readout scale is preferred given noise. (d) $J _ { \mathrm { f i l t e r } }$ and $J _ { \mathrm { n o i s e } }$ vs c (see main text) generally have different optima, and their competition can change where the minimum of J is, or whether it has a finite minimum at all.

One elemental observation is that noise produces representational pressure: whereas the noiseless system has many representations compatible with good performance, the noisy system must minimize the corrupting influence of noise, which can greatly reduce the number of good solutions.

The deterministic solution is highly degenerate. In order for $f ( t ) = e ^ { - \gamma t } v ^ { T } e ^ { W t } u$ to match $f _ { * } ( t ) = e ^ { - c _ { * } t }$ , assuming W is diagonalizable, it must be true that $( \mathrm { i } ) \gamma - c _ { * }$ is an eigenvalue of $W ;$ (ii) u has a component along one of the corresponding $r i g h t$ eigenvectors; (iii) v has a component along one of the corresponding $l e f t$ eigenvectors; and $( \mathrm { i } \mathrm { v } ) \ : u ^ { T } \ : v = 1$ . But this yields a solution space with huge parameter degeneracies, including invariance to arbitrary invertible linear transformations. In particular, if $\pmb { T } \in \mathbb { R } ^ { \breve { N } \times N }$ is an invertible linear transformation, then $f ( t )$ remains invariant to the mapping with ${ \pmb v }  { \pmb T } ^ { T } { \pmb v } , { \pmb u }  { \pmb T } ^ { - 1 } { \pmb u }$ , and $W \to T ^ { - 1 } W T$ , since

$$
{ v ^ { T } } { e ^ { W t } } { u } \stackrel { T } { \longrightarrow } { v ^ { T } } { T / T ^ { - 1 } } { e ^ { W t } } { T / T ^ { - 1 } } { u } = { v ^ { T } } { e ^ { W t } } { u } \ : .\tag{9}
$$

Interesting special cases include when T is a rotation or a change in read-in/readout scale. Meanwhile, the noise term in our objective does not generally respect those degeneracies:

$$
\int _ { 0 } ^ { \infty } e ^ { - 2 \gamma t } v ^ { T } e ^ { W t } S e ^ { W ^ { T } t } v d t \xrightarrow { T } \int _ { 0 } ^ { \infty } e ^ { - 2 \gamma t } v ^ { T } e ^ { W t } ( T S T ^ { T } ) e ^ { W ^ { T } t } v d t .\tag{10}
$$

We discuss various interesting consequences of this simple observation below.

Noise picks out preferred state-space directions. If multiple state-space directions could solve the task (i.e., if W has multiple correct eigenvalues), the presence of noise causes less noisy directions to be preferred (Fig. 2a, b). For example, if $\boldsymbol { W } = ( \bar { \gamma } - c _ { * } ) \boldsymbol { I }$ (i.e., all eigenvalues are correct) and $\pmb { v } ^ { T } \pmb { u } = 1$ (so that $\begin{array} { r } { f ( t ) = f _ { * } ( t ) , } \end{array}$ , then the noise term is minimized when

$$
{ v _ { * } } = \frac { { S ^ { - 1 } } u } { { { u ^ { T } } { S ^ { - 1 } } u } } \ : ,\tag{11}
$$

i.e., when state-space directions are read out according to how noisy they are, with $^ { v _ { * } }$ aligning more with less noisy directions. Essentially this solution is found when only the readout v is trained, i $. \mathrm { e } . ,$ when a ‘reservoir computing’ approach (Lukoseviˇ cius & Jaeger, 2009) is used (see Appendix B).ˇ

Noise produces preferred read-in and readout scales. A trivial way to reduce the noise term is to take the scale of the readout weights to zero. This is consistent with the above general observation; if ${ \pmb T } = \alpha { \pmb I }$ , the filter-matching term remains unchanged but the noise term picks up a factor of<sub>t</sub> $\alpha ^ { 2 }$ , and the overall term is minimized when we take α 0 (Fig. 2c). This implies, given no regularization<sub>c</sub><sup>o</sup> $( \kappa = 0 )$ , that it is optimal to take the readout scale to zero and the read-in scale to infinity.<sub>σ</sub><sup>2</sup>

Noise minimization can compete with filter-matching accuracy. The filter-matching term<sup>c</sup> $J _ { \mathrm { f i l t e r } }$ and the noise term $J _ { \mathrm { n o i s e } }$ generally do not share optima, so their relative size can determine where<sub>t</sub> = the fixed points of learning dynamics are, or whether there is any finite fixed point at all. Fig. 2d<sup>2 c</sup> s<sup>i</sup> depicts an example we will consider later in Sec. 7: the balance of the two terms determines whether= <sup>3</sup> the recurrent weight scale c converges to a finite value near $c _ { * }$ or diverges to infinity.<sub>σ</sub><sup>2</sup> 0 <sub>i</sub>

## <sup>4</sup> <sup>NOISE</sup> <sup>DETERMINES</sup> <sup>WEIGHT</sup> <sup>ALIGNMENT</sup> <sup>IN</sup> <sup>RICH</sup> <sup>REGIME</sup>J c σ<sup>2</sup> = 0 σ<sup>2</sup> = 0.5 σ<sup>2</sup> = 1 σ<sup>2</sup> = 3 σ<sup>2</sup> = 5 τ σ<sup>2</sup> = σ<sup>2</sup>       σ<sup>2</sup> = 0.5 σ<sup>2</sup> = 1 σ<sup>2</sup> = 3 σ<sup>2</sup> = 5 τ σ<sup>2</sup> = σ<sup>2</sup> c/c /τ σ<sup>2</sup> c<sup>N</sup> <sup>N</sup>opt <sup>σ2</sup>crit <sup>κ N</sup>opt <sup>= 0</sup>J c σ<sup>2</sup> = 0 σ<sup>2</sup> = 0.5 σ<sup>2</sup> = 1 σ<sup>2</sup> =<sup>= N</sup> u s<sup>i</sup>

![](images/75151cfdffc7235b63ce797de3db1ba4b4810ee68f1f74cc25d345af38da8dda.jpg)

![](images/c5466c459e36d3cdfe95adcb0bccfc1080b7e8b4524396775d1999381ea99d37.jpg)

![](images/89937f6d6b67fba259b29c6640de0942a6ba75b776433c1f135feb8a9da39abe.jpg)

![](images/12d920f976ad0343f2f04f6033851b5deecd2a4ab88251ef1339b2aac6a92bc0.jpg)  
<sub>c</sub>Figure 3: Noise determines alignment. (a) The dominant contribution to $u _ { i } ( \tau )$ and $v _ { i } ( \tau )$ grows exponentially quickly at a rate ${ \lambda } _ { i } ^ { + }$ <sup>J</sup>that decreases as the corresponding $_ { s }$ eigenvalue $( \mathrm { i . e . }$ ., noise level) $s _ { i }$ increases, so higher-noise modes grow more slowly. (b) Although dynamics involves both growing and decaying contributions, after enough learning time has elapsed, mode amplitudes (solid lines) are well-approximated by only their growing contribution (dashed lines). (c) The dot product between u and v, and the least noisy eigenvector of $S ,$ increases during this phase. W aligns in the same direction and becomes approximately rank-one. (d) The discrepancy between the scales of u and v grows with learning, and is higher in directions with larger noise levels.

In this section, we begin our analysis of this model’s learning dynamics (Eq. 8). We focus on the socalled rich regime, in which weights are initialized to small values and low-rank structure emerges. Our analysis here follows the spirit of Bordelon et al. (2025) but differs in its details.

## 4.1 DYNAMICS GIVEN SMALL INITIAL WEIGHTS

Assume u, v, and W are initialized with small values (say, of order $\epsilon > 0 )$ . To leading order in $\epsilon ,$ gradient descent learning dynamics become (see Appendix A)

$$
\frac { d } { d \tau } u \approx \frac { v } { c _ { * } + \gamma } - \kappa u { \frac { d } { d \tau } v \approx \frac { u } { c _ { * } + \gamma } - \frac { S v } { 2 \gamma } - \kappa v } { \frac { d } { d \tau } W \approx \frac { v u ^ { T } } { ( c _ { * } + \gamma ) ^ { 2 } } - \frac { v v ^ { T } S } { 4 \gamma ^ { 2 } } } .\tag{12}
$$

In other words, u and v tend to evolve according to a certain linear system, while $W$ depends only on input from u and v. This is a major improvement, since Eq. 8 is in general a highly nonlinear system of ODEs. Note that the presence of $_ { s }$ yields dynamics different from those studied by Bordelon et al. (2025); S breaks the symmetry between u and v dynamics, since components of v along high-noise eigenvectors of S are penalized while the analogous components of u are not.

## 4.2 WEIGHTS ROTATE AWAY FROM HIGHLY NOISY STATE SPACE DIRECTIONS

The alignment-phase ODEs (Eq. 12) simplify if we use the eigenvectors of S as a basis. Letting $u _ { i }$ and $v _ { i }$ denote the components of u and v along an eigenvector whose eigenvalue is $s _ { i }$ , we have

$$
\frac { d } { d \tau } u _ { i } = - \kappa u _ { i } + \frac { v _ { i } } { c _ { * } + \gamma }
$$

$$
\frac { d } { d \tau } v _ { i } = \frac { u _ { i } } { c _ { * } + \gamma } - \left( \kappa + \frac { s _ { i } } { 2 \gamma } \right) v _ { i } \ ,\tag{13}
$$

i.e., components along different eigenvectors evolve independently. The $2 \times 2$ matrix that governs these dynamics has two eigenvalues (see Appendix C):

$$
\lambda _ { i } ^ { \pm } : = - \left( \kappa + \frac { s _ { i } } { 4 \gamma } \right) \pm \sqrt { \frac { s _ { i } ^ { 2 } } { 1 6 \gamma ^ { 2 } } + \frac { 1 } { ( c _ { * } + \gamma ) ^ { 2 } } } .\tag{14}
$$

If the regularization strength κ is too large, both eigenvalues are negative, and $u _ { i }$ and $v _ { i }$ collapse to a trivial zero solution. Otherwise, ${ \lambda } _ { i } ^ { + }$ is positive and $\lambda _ { i } ^ { - }$ is negative, which means that the ${ \lambda } _ { i } ^ { + }$ component of the solution tends to dominate after an initial transient. More quantitatively,

$$
\begin{array} { l } { { \displaystyle u _ { i } ( \tau ) \approx \frac { 1 } { \lambda _ { i } ^ { + } - \lambda _ { i } ^ { - } } \left[ \frac { v _ { i } ( 0 ) } { c _ { * } + \gamma } - u _ { i } ( 0 ) ( \kappa + \lambda _ { i } ^ { - } ) \right] e ^ { \lambda _ { i } ^ { + } \tau } } } \\ { { \displaystyle v _ { i } ( \tau ) \approx \frac { ( c _ { * } + \gamma ) ( \kappa + \lambda ^ { + } ) } { \lambda _ { i } ^ { + } - \lambda _ { i } ^ { - } } \left[ \frac { v _ { i } ( 0 ) } { c _ { * } + \gamma } - u _ { i } ( 0 ) ( \kappa + \lambda _ { i } ^ { - } ) \right] e ^ { \lambda _ { i } ^ { + } \tau } . } } \end{array}\tag{15}
$$

One interesting feature of this result is that the growth rate ${ \lambda } _ { i } ^ { + }$ is a decreasing function of s<sub>i</sub> (Fig. 3a), which implies that components of u and v in noisier directions grow more slowly (Fig. 3b). Because small differences are exponentially amplified, to good approximation u and v align along the least noisy eigenvectors of S (Fig. 3c). In particular,

$$
\begin{array} { l } { { \displaystyle { \pmb u } ( \tau ) \approx u ( \tau ) { \pmb P } \left[ \frac { [ { \pmb u } ( 0 ) + { \pmb v } ( 0 ) ] } { 2 } + \frac { \left( c _ { * } + \gamma \right) } { 2 } \left( \frac { s _ { \mathrm { m i n } } } { 4 \gamma } + \Delta \right) { \pmb u } ( 0 ) \right] } } \\ { { \displaystyle { \pmb v } ( \tau ) \approx v ( \tau ) { \pmb P } \left[ \frac { [ { \pmb u } ( 0 ) + { \pmb v } ( 0 ) ] } { 2 } + \frac { \left( c _ { * } + \gamma \right) } { 2 } \left( \frac { s _ { \mathrm { m i n } } } { 4 \gamma } + \Delta \right) { \pmb u } ( 0 ) \right] } } \end{array}\tag{16}
$$

where $_ { r }$ is a projection matrix that projects onto the subspace with the smallest eigenvalue $s _ { \mathrm { m i n } }$ (which may be one-dimensional), and where $\begin{array} { r } { \Delta : = \sqrt { \frac { s _ { \mathrm { m i n } } ^ { 2 } } { 1 6 \gamma ^ { 2 } } + \frac { 1 } { ( c _ { * } + \gamma ) ^ { 2 } } - \frac { 1 } { c _ { * } + \gamma } } } \end{array}$ . Note that Eq. 12 implies that $W$ aligns in the same direction as u and v $( \mathrm { F i g . ~ } 3 \mathrm { c ) }$ . Another interesting feature of this result is that $u _ { i } ( \tau )$ generally does not equal $v _ { i } ( \tau )$ even if they are initialized to the same value— noise breaks the ‘balancing’ condition that can be assumed in the noise-free case (Domine et al., 2025; Bordelon et al., 2025). For large $s _ { i } .$ , this discrepancy in scale can be quite large (Fig. 3d).

## 4.3 THE OUTCOME OF ALIGNMENT

To very good approximation, we can assume u and v align and that $W$ is a rank-one matrix supported on the same vector. Letting $\mathbf { \delta } \mathbf { u } _ { 0 }$ denote that (normalized) vector,

$$
\begin{array} { r } { \left| \begin{array} { l l l l l l } { \boldsymbol { u } ( \tau ) \approx \boldsymbol { u } ( \tau ) \boldsymbol { u } _ { 0 } } & { } & { } & { } & { \boldsymbol { v } ( \tau ) \approx \boldsymbol { v } ( \tau ) \boldsymbol { u } _ { 0 } } & { } & { } & { \boldsymbol { W } ( \tau ) \approx [ \gamma - c ( \tau ) ] \boldsymbol { u } _ { 0 } \boldsymbol { u } _ { 0 } ^ { T } } \end{array} \right| } \end{array}\tag{17}
$$

for some training-time-dependent coefficients $u ( \tau ) , v ( \tau )$ , and $c ( \tau )$ . Because gradient-flow dynamics does not change parameter directions under this ansatz, to characterize the next phase it suffices to analyze the dynamics of the scalars $u ( \tau ) , v ( \tau )$ , and $c ( \tau )$ . In the next few sections, we will use $\sigma ^ { 2 }$ to denote the noise level associated with the alignment direction (i.e., to denote the smallest eigenvalue of S).

## 5 RICH REGIME LEARNING DYNAMICS

Alignment reduces an $N ( N + 2 )$ -dimensional problem to a three-dimensional one. Post-alignment learning dynamics correspond to gradient descent on an effective objective (see Appendix D)

$$
J = \frac { 1 } { 4 c ( \tau ) } \left[ u ( \tau ) v ( \tau ) - \frac { 2 c ( \tau ) } { c _ { * } + c ( \tau ) } \right] ^ { 2 } + \left( \frac { 1 } { 4 c _ { * } } - \frac { c ( \tau ) } { [ c _ { * } + c ( \tau ) ] ^ { 2 } } \right) + \frac { \sigma ^ { 2 } v ( \tau ) ^ { 2 } } { 4 c ( \tau ) } + \frac { \kappa } { 2 } [ u ( \tau ) ^ { 2 } + v ( \tau ) ^ { 2 } ]\tag{18}
$$

where we will refer to $\sigma ^ { 2 } > 0$ as the “noise level”. Learning dynamics in this regime follow

$$
\frac { d } { d \tau } u ( \tau ) = \frac { v ( \tau ) } { c _ { * } + c ( \tau ) } - \frac { u ( \tau ) v ( \tau ) ^ { 2 } } { 2 c ( \tau ) } - \kappa u ( \tau )
$$

$$
\frac { d } { d \tau } v ( \tau ) = \frac { u ( \tau ) } { c _ { * } + c ( \tau ) } - \frac { v ( \tau ) [ \sigma ^ { 2 } + u ( \tau ) ^ { 2 } ] } { 2 c ( \tau ) } - \kappa v ( \tau )\tag{19}
$$

$$
\frac { d } { d \tau } c ( \tau ) = \frac { v ( \tau ) ^ { 2 } [ u ( \tau ) ^ { 2 } + \sigma ^ { 2 } ] } { 4 c ( \tau ) ^ { 2 } } - \frac { u ( \tau ) v ( \tau ) } { [ c _ { * } + c ( \tau ) ] ^ { 2 } } .
$$

![](images/e0f4525054a0bb7be286bf18a71d78527659e0776cffd07f8fb68d15a24bfba5.jpg)

![](images/6a3dc68d4a692eb5390f7b548cf6dc68fbcac1eec7e674c2ae295deebe72d802.jpg)

![](images/aa8ae9b9d02880f7d37db9f1433fd2528c767411ef54f1cfd584ab76ea5542e6.jpg)  
Figure 4: Rich regime learning dynamics. For plotting, we take $c _ { * } = 1 , \sigma ^ { 2 } = 1$ , and $\boldsymbol { \kappa } = 0 . \left( \mathbf { a } \right)$ In the absence of regularization, learning dynamics drive c toward its target, $c _ { * }$ , and the amplitude $N : =$ uv toward 1. (b) Unlike the noise-free $( \sigma ^ { 2 } = 0 )$ dynamics, noisy dynamics produce asymmetric trajectories for u and v, even if they are initialized to the same value. Asymptotically, uv $\approx 1$ , u continues to increase, and v continues to decrease. (c) The filter-matching term and noise term are both driven to zero, although in this case the latter occurs on a slower time scale.

In the remainder of this section, we will assume no regularization $( \kappa = 0 )$ . Let $N : = u v$ denote the filter amplitude, so that $f ( t ) = N e ^ { - c t }$ . J is obviously minimized when $N = 1$ and $c = c _ { * }$ , since these choices set the first two terms to zero. Learning dynamics drive N and c toward those values (Fig. 4a).

A key difference between the noisy and noise-free $( \sigma ^ { 2 } \ = \ 0 )$ dynamics is that noise breaks the symmetry between u and v. In the noise-free case, if u and v are initialized to the same value—i.e., if one makes a balancing assumption (Domine et al., 2025; Bordelon et al., $2 0 2 5 ) \_ u = v$ holds throughout training. In the noisy case, if there is no regularization, u continues to grow while v continues to shrink (Fig. 4b). In fact, Eq. 19 implies that

$$
\frac { d } { d \tau } [ u ( \tau ) ^ { 2 } - v ( \tau ) ^ { 2 } ] = \frac { \sigma ^ { 2 } } { c ( \tau ) } v ( \tau ) ^ { 2 } > 0 ,\tag{20}
$$

which means $u  \infty$ and $v  0 .$ This makes sense, since taking $N = u v = 1$ and $v  0$ is precisely the combination that minimizes both the filter-matching term and the noise term (Fig. 4c).

## 6 A NOISE-INDUCED PHASE TRANSITION IN RICH REGIME DYNAMICS σ

![](images/f4024f25d046f4db27047dea0a41db77bb1c54d234671b2fd1322bb6acd4ed90.jpg)

![](images/91cd35fe435a4756d58e7bafa5ae7485b817e25aa9b09c011dc88d896bf3b64d.jpg)

![](images/c8e692b1fe6c9f32da72e6644277343c46f73d2151412a68ee937a416c6e8919.jpg)

![](images/ed5070ab8f356fa6e1116f62680c7377688a57548f3e80763240c19eca1f2c85.jpg)  
σ<sup>2</sup> Figure 5: Excessive noise causes amplitude collapse. For plotting, we take $c _ { * } = 1$ and $\kappa = 0 . 2$ (a) Below a critical noise level $\sigma _ { \mathrm { c r i t } } ^ { 2 }$ <sup>1</sup>, learning dynamics drives the filter amplitude $N : = u v$ toward a nontrivial fixed point. Above that noise level, N is driven to zero. (b) The optimal amplitude $N _ { \mathrm { o p t } }$ as a function of noise level. For $\sigma ^ { 2 } \geq \sigma _ { \mathrm { c r i t } } ^ { 2 }$ , the trivial solution with $f ( t ) = 0$ is optimal. (c) Below $\sigma _ { \mathrm { c r i t } } ^ { 2 } , J$ 0is minimized at a specific c value. Above it, J is insensitive to c since ${ \bf { \bar { N } } } = 0 .$ (d) The phase boundary $\sigma _ { \mathrm { c r i t } } ^ { 2 } ( \kappa )$ <sup>=</sup>. The noise threshold separating trivial and nontrivial solutions depends on <sup>σ</sup>the regularization strength; it is higher when κ is small, and $\sigma _ { \mathrm { c r i t } } ^ { 2 } \to \infty \mathrm { a s } \kappa \to 0$

σ<sup>2</sup> The behavior described in the previous section, aside from the asymmetry between u and v dynam-<sup>c</sup>ics, is not substantially different from what Bordelon et al. (2025) observed in the noise-free case. However, the noisy case becomes significantly different when the regularization strength κ is taken to be nonzero. Without regularization, we saw that the system minimizes noise by taking $u  \infty$ and $v  0 ;$ however, if high u values are penalized, the system can become frustrated. If the noise level is sufficiently high, there is a phase transition, and a trivial solution—which minimizes noise at the cost of a collapse in filter-matching performance—becomes optimal.

Below a critical noise level $\sigma _ { \mathrm { c r i t } } ^ { 2 }$ (which depends on the regularization strength $\kappa )$ , the filter amplitude $N : = u v$ approaches a nonzero fixed point $N _ { \mathrm { o p t } }$ . When $\kappa = 0$ , this equals one, but it decreases as κ increases, since the scales of u and v are penalized by the regularization term. Beyond the critical noise level, dynamics drive N to zero (Fig. 5a), and moreover J is minimized at $\dot { N _ { \mathrm { o p t } } } = 0 ( \mathrm { F i g . } 5 \mathrm { b } )$

One can show algebraically (see Appendix E) that, after minimizing J over u and $v ,$ it becomes

$$
J ( c ) = { \frac { 1 } { 4 c _ { * } } } - c \left[ { \frac { 1 } { c _ { * } + c } } - \kappa \sqrt { 1 + { \frac { \sigma ^ { 2 } } { 2 \kappa c } } } \right] _ { + } ^ { 2 } .\tag{21}
$$

where $[ \ldots ] _ { + }$ denotes rectification (i.e., a ReLU). This simplified objective only has a nontrivial solution if the bracketed quantity is greater than zero; otherwise, J becomes independent of c (Fig. 5c).

We can identify the corresponding phase boundary $\sigma _ { \mathrm { c r i t } } ^ { 2 } ( \kappa )$ by determining the noise level at which the bracketed quantity and its derivative equal zero. We find (see Appendix E) that

$$
\boxed { \sigma _ { \mathrm { c r i t } } ^ { 2 } ( \kappa ) = \frac { [ 1 - q ( \kappa ) ^ { 2 } ] ^ { 2 } } { q ( \kappa ) } }\tag{22}
$$

for $\kappa < 1 / c _ { * }$ , and $\sigma _ { \mathrm { c r i t } } ^ { 2 } = 0$ for $\kappa \geq 1 / c _ { * }$ . Here,

$$
q ( \kappa ) : = \frac { 2 } { \sqrt { 3 } } \sinh \left[ \frac { 1 } { 3 } \mathrm { a r s i n h } ( 3 \sqrt { 3 } \kappa c _ { * } ) \right] \qquad \mathrm { a n d } \qquad \mathrm { a r s i n h } ( x ) : = \log ( x + \sqrt { x ^ { 2 } + 1 } ) .\tag{23}
$$

The phase boundary is plotted in Fig. 5d. It has a few features worth emphasizing. First, since $q ( \kappa )  0$ as $\kappa  \mathrm { 0 , } \sigma _ { \mathrm { c r i t } } ^ { 2 }$ diverges to infinity in this limit; this is consistent with the observation that there is no phase transition without regularization. Second, for small $\kappa ,$ we have the comparatively simple expressions

$$
q ( \kappa ) \approx 2 c _ { * } \kappa
$$

$$
\sigma _ { \mathrm { c r i t } } ^ { 2 } ( \kappa ) \approx \frac { 1 } { 2 c _ { * } \kappa }\tag{24}
$$

which qualitatively match Fig. 5d. In summary, if there is too much noise, it is almost as if the system’s learning dynamics seek an ‘escape route’: parameters are pushed into trivial regimes (here, the filter amplitude N is taken to zero) in order to minimize noise at the expense of task accuracy.

## 7 A SECOND PHASE TRANSITION WHEN SCALE CHANGES ARE IMPOSSIBLE σ

![](images/6bb2c287421808ba4d1cea2789e59531d3da0ffbbf468dc7da6877be3058bcfb.jpg)

b <sub>2 =</sub>  
![](images/dfe37d8e1848454999287a69d59cea695b8b70a3f58a1c059ecb1db8b8f699f5.jpg)

![](images/01fdad6be44f403b6530db93d344f758eac1288aadcbb2d3adddc435f1cc49fb.jpg)

![](images/36112297d4546e12f934ee3b30ac35368f22d67d27ae1edf498fdeb0cef47f35.jpg)  
Figure 6: Excessive noise produces an infinitely fast filter. $c _ { * } = 1$ <sup>2</sup> assumed throughout. (a) When 1the read-in and readout scales are fixed, J is a function of c only, and for low enough noise it has a single attractor basin at $c = c _ { \mathrm { o p t } }$ . As $\sigma ^ { 2 }$ <sup>=</sup>increases, this basin becomes increasingly shallow. (b) For ${ \bar { \sigma ^ { 2 } } } \geq \sigma _ { \mathrm { c r i t } } ^ { 2 } = 3 .$ σ , J is a decreasing function of c with no finite minimum. (c) Below the noise .<sup>5</sup>threshold, dynamics drive c toward the unique stable fixed point; above it, c grows without bound. (d) $c _ { \mathrm { o p t } }$ diverges like $1 / ( \sigma _ { \mathrm { c r i t } } ^ { 2 } - \sigma ^ { 2 } )$ just below the noise threshold.

We just saw how, for sufficiently large intrinsic noise $\sigma ^ { 2 } > 0 ;$ <sup>0</sup>, the system may converge to a trivial solution with zero amplitude, i.e., $u v = 0$ <sup>=</sup>. One may wonder whether this behavior is peculiar to our

setup—for example, whether it depends on trainable read-in and readout weights. Interestingly, if u and v are frozen, a different type of noise-induced phase transition is possible.

Consider the $u = v = 1$ specialization of the simplified objective (Eq. 18):

$$
J = \frac { 1 } { 4 c ( \tau ) } \left[ 1 - \frac { 2 c ( \tau ) } { c _ { * } + c ( \tau ) } \right] ^ { 2 } + \frac { 1 } { 4 c _ { * } } - \frac { c ( \tau ) } { [ c _ { * } + c ( \tau ) ] ^ { 2 } } + \frac { \sigma ^ { 2 } } { 4 c ( \tau ) } = \frac { 1 + \sigma ^ { 2 } } { 4 c ( \tau ) } - \frac { 1 } { c _ { * } + c ( \tau ) } + \frac { 1 } { 4 c _ { * } }\tag{25}
$$

where we have removed the regularization term because it is constant. The corresponding learning dynamics are (see Appendix F)

$$
\frac { d } { d \tau } c = \frac { [ \sqrt { 1 + \sigma ^ { 2 } } ( c _ { * } + c ) + 2 c ] } { 4 c ^ { 2 } ( c _ { * } + c ) ^ { 2 } } [ \sqrt { 1 + \sigma ^ { 2 } } ( c _ { * } + c ) - 2 c ]\tag{26}
$$

For sufficiently small noise, these dynamics have a stable positive fixed point at

$$
c _ { \mathrm { o p t } } = \frac { \sqrt { 1 + \sigma ^ { 2 } } c _ { * } } { 2 - \sqrt { 1 + \sigma ^ { 2 } } } .\tag{27}
$$

As one might expect, $c _ { \mathrm { o p t } }$ equals the target rate $c _ { * }$ when there is no noise. However, as $\sigma ^ { 2 }$ increases, the corresponding basin of J becomes increasingly shallow (Fig. 6a). Pushing the noise level higher than $\sigma _ { \mathrm { c r i t } } ^ { 2 } \overset { \cdot } { = } 3$ causes this basin to disappear (Fig. 6b), and for J to be minimized at $c = \infty$ . Indeed, Eq. 26 has $d c / d \tau > 0$ for all c if ${ \bar { \sigma ^ { 2 } } } \geq 3 ,$ , so learning drives c toward infinity (Fig. 6c). Just below $\sigma _ { \mathrm { c r i t } } ^ { 2 } ,$ , the optimal c value diverges as $1 / ( \sigma _ { \mathrm { c r i t } } ^ { 2 } - \sigma ^ { 2 } )$ (Fig. 6d). We conclude that, although the precise representational ‘escape route’ may vary, the system may be driven toward trivial solutions that minimize noise when $\sigma ^ { 2 }$ is excessively large.

## 8 DISCUSSION

We developed a theory of noisy linear RNN learning dynamics and mathematically showed that noise can decisively shape representation learning. Consistent with Bordelon et al. (2025), in the ‘rich’ regime, where parameters are initialized to small values, a low-rank task (here, matching a single leaky integration time scale) yields an approximately low-rank solution. But our findings differ from those of Bordelon et al. (2025) in three important ways. First, noise strongly modulates the initial alignment phase, and u, v, and W preferentially align along the least noisy eigenvector (or eigenvectors) of S, rather than simply in the average direction of the initial u and v; this is interesting in part because it makes the network’s solution depend much less on its initial condition.

Second, noise produces asymmetry between the read-in and readout weights, and in the second phase u tends to become large while v tends to become small. This is true even if u and v are initialized to identical values because noise breaks the balancing property that holds in the noise-free case. Finally, noise produces multiple types of phase transitions, because minimizing noise can compete with achieving good task performance. If u and v are trainable, we find that excessive noise drives the network toward a solution whose filter amplitude N := uv equals zero; interestingly, this only occurs if regularization constrains the scales of u and v. If only W is trainable, excessive noise drives the system to $f ( t ) = 0$ in a different way, by making the filter’s frequency infinitely fast.

In other machine learning settings, noise is known to shape what types of representations systems learn. Noise in optimization due to using stochastic rather than full-batch gradient descent tends to produce convergence to broader loss landscape minima that generalize better (Smith & Le, 2018; Smith et al., 2020). Noise in diffusion generative modeling may similarly improve generalization, albeit potentially through a different mechanism (Vastola, 2025). Our work suggests that noise shapes RNN representation learning, but does not necessarily produce better solutions—only different ones. Here, noise reduces solution-space degeneracy by constraining the direction of weight alignment and the relative scales of the read-in and readout weights. If there is too much noise, it i clear that noise has deleterious effects, since it drives the system toward trivial solutions.

Limitations and future work. For tractability’s sake, we made a large number of simplifying assumptions: we assumed a whitened scalar input; a linear network; a simple filter-matching task involving learning one time scale; and near-zero parameter initializations that yield low-rank learning dynamics. Although the work we build on makes similar assumptions, it may be possible to relax some of them. For example, it may be possible to study nonlinear networks by linearizing them around a fixed point (either in state space or parameter space), or by exploiting dynamical mean-field theory machinery (Bordelon & Pehlevan, 2023). One class of tasks which is more realistic, but whose learning dynamics may remain tractable, is target pursuit (Vastola & Rajan, 2026).

## ACKNOWLEDGMENTS

KR was funded by the NIH (RF1DA056403, U01NS136507), James S. McDonnell Foundation (220020466), Simons Foundation (Pilot Extension-00003332-02), McKnight Endowment Fund, CI-FAR Azrieli Global Scholar Program, NSF (2046583), a Harvard Medical School Neurobiology Lefler Small Grant Award, and a Harvard Medical School Dean’s Innovation Award. This work has been made possible in part by a gift from the Chan Zuckerberg Initiative Foundation to establish the Kempner Institute for the Study of Natural and Artificial Intelligence at Harvard University.

## AI USE STATEMENT

In this work, we used generative AI tools for brainstorming, literature search, to assist with checking the mathematical correctness of our results, and to assist with code generation. The authors have checked all text and results and take full responsibility for them.

## REFERENCES

Y. Bengio, P. Simard, and P. Frasconi. Learning long-term dependencies with gradient descent is difficult. IEEE Transactions on Neural Networks, 5(2):157–166, 1994. doi: 10.1109/72.279181.

Blake Bordelon and Cengiz Pehlevan. Self-consistent dynamical field theory of kernel evolution in wide neural networks\*. Journal of Statistical Mechanics: Theory and Experiment, 2023(11): 114009, November 2023. doi: 10.1088/1742-5468/ad01b0. URL https://doi.org/10. 1088/1742-5468/ad01b0.

Blake Bordelon, Jordan Cotler, Cengiz Pehlevan, and Jacob A. Zavatone-Veth. Dynamics of learning to integrate in linear recurrent neural networks. arXiv e-prints, art. arXiv:2503.18754, March 2025. doi: 10.48550/arXiv.2503.18754.

A. Crisanti and H. Sompolinsky. Path integral approach to random neural networks. Phys. Rev. E, 98:062120, Dec 2018. doi: 10.1103/PhysRevE.98.062120. URL https://link.aps.org/ doi/10.1103/PhysRevE.98.062120.

Clementine Domine, Nicolas Anguita, Alexandra M Proca, Lukas Braun, Daniel Kunin, Pedro Mediano, and Andrew Saxe. From Lazy to Rich: Exact Learning Dynamics in Deep Linear Networks. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 102485–102536, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ fde1a69a5b6e554b2f1f727197d2651d-Paper-Conference.pdf.

Laura N. Driscoll, Krishna Shenoy, and David Sussillo. Flexible multitask computation in recurrent networks utilizes shared dynamical motifs. Nature Neuroscience, 27(7):1349–1363, July 2024. ISSN 1546-1726. doi: 10.1038/s41593-024-01668-6. URL https://doi.org/10.1038/ s41593-024-01668-6.

A. Aldo Faisal, Luc P. J. Selen, and Daniel M. Wolpert. Noise in the nervous system. Nature Reviews Neuroscience, 9(4):292–303, April 2008. ISSN 1471-0048. doi: 10.1038/nrn2258. URL https://doi.org/10.1038/nrn2258.

Ann Huang, Satpreet Harcharan Singh, Flavio Martinelli, and Kanaka Rajan. Measuring and Controlling Solution Degeneracy across Task-Trained Recurrent Neural Networks. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, Main Conference, pp. 105259–105304. Curran Associates, Inc., 2025. doi: 10.52202/ 085713-3517. URL https://proceedings.neurips.cc/paper\_files/paper/ 2025/file/979d5a0ecc2cc3653928c5b793fc416b-Paper-Conference.pdf.

Kazuki Irie, Anand Gopalakrishnan, and Jurgen Schmidhuber. Exploring the Promise and Limits of¨ Real-Time Recurrent Learning. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun (eds.), International Conference on Learning Representations, volume 2024, pp. 27575– 27598, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/ file/74aec30590e07dbe2e29879f9df14fb2-Paper-Conference.pdf.

Soon Hoe Lim, N. Benjamin Erichson, Liam Hodgkinson, and Michael Mahoney. Noisy Recurrent Neural Networks. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P. S. Liang, and J. Wortman Vaughan (eds.), Advances in Neural Information Processing Systems, volume 34, pp. 5124–5137. Curran Associates, Inc., 2021. URL https://proceedings.neurips.cc/paper\_ files/paper/2021/file/29301521774ff3cbd26652b2d5c95996-Paper.pdf.

Mantas Lukoseviˇ cius and Herbert Jaeger. Reservoir computing approaches to recurrent neuralˇ network training. Computer Science Review, 3(3):127–149, 2009. ISSN 1574-0137. doi: https://doi.org/10.1016/j.cosrev.2009.03.005. URL https://www.sciencedirect.com/ science/article/pii/S1574013709000173.

Valerio Mante, David Sussillo, Krishna V. Shenoy, and William T. Newsome. Context-dependent computation by recurrent dynamics in prefrontal cortex. Nature, 503(7474):78–84, November 2013. ISSN 1476-4687. doi: 10.1038/nature12742. URL https://doi.org/10.1038/ nature12742.

Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio. On the difficulty of training recurrent neural networks. In Sanjoy Dasgupta and David McAllester (eds.), Proceedings ofthe 30th International Conference on Machine Learning, volume 28 of Proceedings of Machine Learning Research, pp. 1310–1318, Atlanta, Georgia, USA, 17–19 Jun 2013. PMLR. URL https://proceedings. mlr.press/v28/pascanu13.html.

Alexandra Maria Proca, Clementine Carla Juliette Domin´ e, Murray Shanahan, and Pedro A. M.´ Mediano. Learning dynamics in linear recurrent neural networks. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 49860–49902. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/proca25a.html.

Kanaka Rajan, Christopher D. Harvey, and David W. Tank. Recurrent Network Models of Sequence Generation and Memory. Neuron, 90(1):128–142, April 2016. ISSN 0896-6273. doi: 10.1016/j. neuron.2016.02.009. URL https://doi.org/10.1016/j.neuron.2016.02.009.

Friedrich Schuessler, Francesca Mastrogiuseppe, Alexis Dubreuil, Srdjan Ostojic, and Omri Barak. The interplay between randomness and structure during learning in RNNs. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 13352–13362. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper\_files/paper/2020/ file/9ac1382fd8fc4b631594aa135d16ad75-Paper.pdf.

Samuel Smith, Erich Elsen, and Soham De. On the generalization benefit of noise in stochastic gradient descent. In Hal Daume III and Aarti Singh (eds.),´ Proceedings ofthe 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 9058–9067. PMLR, 13–18 Jul 2020. URL https://proceedings.mlr.press/v119/ smith20a.html.

Samuel L. Smith and Quoc V. Le. A bayesian perspective on generalization and stochastic gradient descent. In International Conference on Learning Representations, 2018. URL https:// openreview.net/forum?id=BJij4yg0Z.

H. Sompolinsky, A. Crisanti, and H. J. Sommers. Chaos in random neural networks. Phys. Rev. Lett., 61:259–262, Jul 1988. doi: 10.1103/PhysRevLett.61.259. URL https://link.aps. org/doi/10.1103/PhysRevLett.61.259.

David Sussillo and L.F. Abbott. Generating Coherent Patterns of Activity from Chaotic Neural Networks. Neuron, 63(4):544–557, August 2009. ISSN 0896-6273. doi: 10.1016/j.neuron.2009. 07.018. URL https://doi.org/10.1016/j.neuron.2009.07.018.

John Vastola. Generalization through variance: how noise shapes inductive biases in diffusion models. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 34922–34960, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ 564ccf9de3a2a959d3d338889bad9d88-Paper-Conference.pdf.

John J. Vastola and Kanaka Rajan. Solvable models of learning to pursue a moving target. Reinforcement Learning Journal, 7, 2026.

Tim P. Vogels, Kanaka Rajan, and L.F. Abbott. Neural network dynamics. Annual Review of Neuroscience, 28(Volume 28, 2005):357–376, 2005. ISSN 1545-4126. doi: https://doi.org/ 10.1146/annurev.neuro.28.061604.135637. URL https://www.annualreviews.org/ content/journals/10.1146/annurev.neuro.28.061604.135637.

Saurabh Vyas, Matthew D. Golub, David Sussillo, and Krishna V. Shenoy. Computation Through Neural Population Dynamics. Annual Review of Neuroscience, 43 (Volume 43, 2020):249–275, 2020. ISSN 1545-4126. doi: https://doi.org/10.1146/ annurev-neuro-092619-094115. URL https://www.annualreviews.org/content/ journals/10.1146/annurev-neuro-092619-094115.

P.J. Werbos. Backpropagation through time: what it does and how to do it. Proceedings of the IEEE, 78(10):1550–1560, 1990. doi: 10.1109/5.58337.

Ronald J. Williams and David Zipser. A learning algorithm for continually running fully recurrent neural networks. Neural Computation, 1(2):270–280, 1989. doi: 10.1162/neco.1989.1.2.270.

Guangyu Robert Yang, Madhura R. Joglekar, H. Francis Song, William T. Newsome, and Xiao-Jing Wang. Task representations in neural networks trained to perform many cognitive tasks. Nature Neuroscience, 22(2):297–306, February 2019. ISSN 1546-1726. doi: 10.1038/ s41593-018-0310-2. URL https://doi.org/10.1038/s41593-018-0310-2.

## A OBJECTIVE FUNCTION AND DERIVATION OF LEARNING DYNAMICS

In this appendix, we simplify our original objective function and derive expressions for the associated gradient descent learning dynamics (in the gradient-flow limit that yields ODEs rather than a discrete-time recurrence).

## A.1 SETUP

Recall from Sec. 2 that the objective function has the form

$$
\boxed { J : = \operatorname* { l i m } _ { T \to \infty } \frac { 1 } { T } \int _ { 0 } ^ { T } \frac { 1 } { 2 } \mathbb { E } _ { x , r } \left\{ \left[ y ( t ) - y _ { * } ( t ) \right] ^ { 2 } \right\} d t + \frac { \kappa } { 2 } \| u \| ^ { 2 } + \frac { \kappa } { 2 } \| v \| ^ { 2 } }\tag{28}
$$

where $y ( t ) = { \pmb v } ^ { T } { \pmb r } ( t )$ is the RNN’s output and

$$
y _ { * } ( t ) : = \int _ { 0 } ^ { t } e ^ { - c _ { * } t ^ { \prime } } x ( t - t ^ { \prime } ) d t ^ { \prime } = \int _ { 0 } ^ { t } f _ { * } ( t ^ { \prime } ) x ( t - t ^ { \prime } ) d t ^ { \prime }\tag{29}
$$

is the target output. If we assume zero initial activity $( \mathrm { i . e . , } r ( 0 ) = \mathbf { 0 } )$ , then as we mention in Sec. 2,

$$
p ( \boldsymbol { r } , t | \boldsymbol { r } _ { 0 } = \mathbf { 0 } , 0 ) = \mathcal { N } \left( \int _ { 0 } ^ { t } e ^ { A t ^ { \prime } } \boldsymbol { u } \boldsymbol { x } ( t - t ^ { \prime } ) d t ^ { \prime } , \int _ { 0 } ^ { t } e ^ { A t ^ { \prime } } S e ^ { A ^ { T } t ^ { \prime } } d t ^ { \prime } \right) .\tag{30}
$$

This implies that the output $y ( t )$ has statistics

$$
\mathbb { E } _ { \boldsymbol { r } } [ \boldsymbol { y } ( t ) ] = \int _ { 0 } ^ { t } \boldsymbol { v } ^ { T } e ^ { \boldsymbol { A } t ^ { \prime } } \boldsymbol { u } ~ \boldsymbol { x } ( t - t ^ { \prime } ) ~ d t ^ { \prime } = \int _ { 0 } ^ { t } \boldsymbol { f } ( t ^ { \prime } ) ~ \boldsymbol { x } ( t - t ^ { \prime } ) ~ d t ^ { \prime }\tag{31}
$$

$$
\operatorname { v a r } _ { r } [ y ( t ) ] = { \pmb v } ^ { T } \pmb { \Sigma } { \pmb v }
$$

where $f ( t ) : = e ^ { - \gamma t } v ^ { T } e ^ { W t } u$ is the learned filter and the covariance $\pmb { \Sigma }$ is defined as

$$
\Sigma : = \int _ { 0 } ^ { t } e ^ { A t ^ { \prime } } S e ^ { A ^ { T } t ^ { \prime } } d t ^ { \prime } .\tag{32}
$$

Also recall that the input signal $x ( t )$ is assumed to have white noise statistics, i.e., $\mathbb { E } _ { x } [ x ( t ) ] = 0$ and $\mathbb { E } _ { x } [ x ( t ) x ( t - t ^ { \prime } ) ] = \bar { \delta } ( t - \overline { { t } } ^ { \prime } )$

## A.2 SIMPLIFYING THE OBJECTIVE

Decomposing into a bias and variance term. First, average over r:

$$
\begin{array} { l }  \displaystyle { J = \operatorname* { l i m } _ { T \to \infty } \frac { 1 } { T } \int _ { 0 } ^ { T } \frac { 1 } { 2 } \mathbb { E } _ { x } \{ [ \mathbb { E } _ { r } [ y ( t ) ] - y _ { * } ( t ) ] ^ { 2 } + \operatorname { v a r } _ { r } [ y ( t ) ] \} d t + \frac { \kappa } { 2 } \| u \| ^ { 2 } + \frac { \kappa } { 2 } \| v \| ^ { 2 } } \\ { \displaystyle { \operatorname rightharpoonup } \operatorname* { l i m } _ { T \to \infty } \frac { 1 } { T } \int _ { 0 } ^ { T } \frac { 1 } { 2 } \mathbb { E } _ { x } \{ [ \mathbb { E } _ { r } [ y ( t ) ] - y _ { * } ( t ) ] ^ { 2 } \} + \frac { 1 } { 2 } \mathrm { v a r } _ { r } [ y ( t ) ] d t + \frac { \kappa } { 2 } \| u \| ^ { 2 } + \frac { \kappa } { 2 } \| v \| ^ { 2 } } \\ { \displaystyle { \operatorname rightharpoonup } \operatorname* { l i m } _ { T \to \infty } \frac { 1 } { T } \int _ { 0 } ^ { T } \frac { 1 } { 2 } \mathbb { E } _ { x } \{ [ \mathbb { E } _ { r } [ y ( t ) ] - y _ { * } ( t ) ] ^ { 2 } \} + \frac { 1 } { 2 } v ^ { T } \Sigma ( t ) v d t + \frac { \kappa } { 2 } \| u \| ^ { 2 } + \frac { \kappa } { 2 } \| v \| ^ { 2 } } \end{array}\tag{33}
$$

where the second equality uses the fact that variance in $y ( t )$ is input-independent.

Simplifying the mean term. The second term is input-independent; it represents noise due to recurrent dynamics. The first term can be written in terms of filters:

$$
\begin{array} { l } { { \displaystyle = \operatorname* { l i m } _ { T \to \infty } \displaystyle \frac { 1 } { T } \int _ { 0 } ^ { T } \frac { 1 } { 2 } \mathbb { E } _ { x } \bigg \{ \int _ { 0 } ^ { t } \int _ { 0 } ^ { t } \big [ f ( s ) - f _ { * } ( s ) \big ] [ f ( s ^ { \prime } ) - f _ { * } ( s ^ { \prime } ) \big ] x ( t - s ) x ( t - s ^ { \prime } ) \bigg \} \ d s d s ^ { \prime } d t } } \\ { { \displaystyle = \operatorname* { l i m } _ { T \to \infty } \displaystyle \frac { 1 } { T } \int _ { 0 } ^ { T } \frac { 1 } { 2 } \int _ { 0 } ^ { t } \int _ { 0 } ^ { t } \big [ f ( s ) - f _ { * } ( s ) \big ] [ f ( s ^ { \prime } ) - f _ { * } ( s ^ { \prime } ) \big ] \mathbb { E } _ { x } [ x ( t - s ) x ( t - s ^ { \prime } ) ] \ d s d s ^ { \prime } d t } } \\ { { \displaystyle = \operatorname* { l i m } _ { T \to \infty } \displaystyle \frac { 1 } { T } \int _ { 0 } ^ { T } \frac { 1 } { 2 } \int _ { 0 } ^ { t } \int _ { 0 } ^ { t } \big [ f ( s ) - f _ { * } ( s ) \big ] [ f ( s ^ { \prime } ) - f _ { * } ( s ^ { \prime } ) \big ] \delta ( s - s ^ { \prime } ) \ d s d s ^ { \prime } d t } } \\ { { \displaystyle = \operatorname* { l i m } _ { T \to \infty } \displaystyle \frac { 1 } { T } \int _ { 0 } ^ { T } \frac { 1 } { 2 } \int _ { 0 } ^ { t } [ f ( s ) - f _ { * } ( s ) ] ^ { 2 } d s d t . } } \end{array}\tag{34}
$$

This can be simplified with a clever rearrangement. Note,

$$
\begin{array} { l } { \displaystyle \operatorname* { l i m } _ { T \to \infty } \frac { 1 } { 2 T } \int _ { 0 } ^ { T } \int _ { 0 } ^ { t } [ f ( s ) - f _ { * } ( s ) ] ^ { 2 } d s d t = \displaystyle \operatorname* { l i m } _ { T \to \infty } \frac { 1 } { 2 T } \int _ { 0 } ^ { T } \int _ { s } ^ { T } [ f ( s ) - f _ { * } ( s ) ] ^ { 2 } d t d s } \\ { \displaystyle \ = \operatorname* { l i m } _ { T \to \infty } \frac { 1 } { 2 } \int _ { 0 } ^ { T } \left( 1 - \frac { s } { T } \right) [ f ( s ) - f _ { * } ( s ) ] ^ { 2 } d s . } \end{array}\tag{35}
$$

If the eigenvalues of A have negative real parts (i.e., if dynamics is stable), then we can drop the $s / T$ term, since it vanishes in the $T \to \infty$ limit. We conclude that

$$
J _ { \mathrm { f i l t e r } } = { \frac { 1 } { 2 } } \int _ { 0 } ^ { \infty } [ f ( t ) - f _ { * } ( t ) ] ^ { 2 } d t .\tag{36}
$$

Simplifying the variance term. The noise term can be rewritten as

$$
\begin{array} { r l } & { \underset { T  \infty } { \operatorname* { l i m } } \frac { 1 } { 2 T } \int _ { 0 } ^ { T } v ^ { T } \Sigma ( t ) v \ d t } \\ & { = v ^ { T } \{ \underset { T  \infty } { \operatorname* { l i m } } \frac { 1 } { 2 T } \int _ { 0 } ^ { T } \int _ { 0 } ^ { t } e ^ { A s } S e ^ { A ^ { T } s } \ d s d t \} v } \\ & { = v ^ { T } \{ \underset { T  \infty } { \operatorname* { l i m } } \frac { 1 } { 2 T } \int _ { 0 } ^ { T } \int _ { s } ^ { T } e ^ { A s } S e ^ { A ^ { T } s } \ d t d s \} v } \\ & { = v ^ { T } \{ \underset { T  \infty } { \operatorname* { l i m } } \frac { 1 } { 2 } \int _ { 0 } ^ { T } ( 1 - \frac { s } { T } ) e ^ { A s } S e ^ { A ^ { T } s } d s \} v . } \end{array}\tag{37}
$$

Once again, if dynamics are stable, we can drop the $s / T$ term in the infinite T limit. We obtain

$$
J _ { \mathrm { n o i s e } } = \frac { 1 } { 2 } \int _ { 0 } ^ { \infty } { \pmb v } ^ { T } e ^ { { \pmb A } t } { \pmb S } e ^ { { \pmb A } ^ { T } t } { \pmb v } d t .\tag{38}
$$

Total objective. Adding all of the terms, we obtain the simplified objective

$$
\boxed { J ( u , v , W ) = \int _ { 0 } ^ { \infty } \frac { 1 } { 2 } \left[ f ( t ) - f _ { * } ( t ) \right] ^ { 2 } + \frac { e ^ { - 2 \gamma t } } { 2 } v ^ { T } e ^ { W t } S e ^ { W ^ { T } t } v ~ d t + \frac { \kappa } { 2 } \| u \| ^ { 2 } + \frac { \kappa } { 2 } \| v \| ^ { 2 } } .\tag{39}
$$

## A.3 USEFUL DERIVATIVES

A few derivatives are necessary to compute the gradients of J. First, the derivative of the matrix exponential with respect to W has components

$$
{ \frac { \partial } { \partial W _ { i j } } } e ^ { W t } = \int _ { 0 } ^ { t } d t ^ { \prime } e ^ { W t ^ { \prime } } e _ { i } e _ { j } ^ { T } e ^ { W ( t - t ^ { \prime } ) } .\tag{40}
$$

Similarly, the derivative of its transpose is

$$
{ \frac { \partial } { \partial W _ { i j } } } e ^ { W ^ { T } t } = \int _ { 0 } ^ { t } d t ^ { \prime } e ^ { W ^ { T } t ^ { \prime } } e _ { j } e _ { i } ^ { T } e ^ { W ^ { T } ( t - t ^ { \prime } ) } .\tag{41}
$$

The derivative of the learned filter $f ( t )$ with respect to W has components

$$
\frac { \partial } { \partial W _ { i j } } f ( t ) = e ^ { - \gamma t } \int _ { 0 } ^ { t } d t ^ { \prime } \left[ e ^ { W ^ { T } t ^ { \prime } } v \right] _ { i } \left[ e ^ { W ( t - t ^ { \prime } ) } u \right] _ { j } = e ^ { - \gamma t } \int _ { 0 } ^ { t } d t ^ { \prime } e ^ { W ^ { T } t ^ { \prime } } v u ^ { T } e ^ { W ^ { T } ( t - t ^ { \prime } ) }\tag{42}
$$

Finally, the derivative of the variance term with respect to W has components

$$
\begin{array} { l } { { \displaystyle \frac { \partial } { \partial W _ { i j } } { \boldsymbol { v } } ^ { T } e ^ { W t } { \boldsymbol { S } } e ^ { W ^ { T } t } { \boldsymbol { v } } } \ ~ } \\ { { \displaystyle = \int _ { 0 } ^ { t } d t ^ { \prime } ~ } { \boldsymbol { v } } ^ { T } e ^ { W t ^ { \prime } } e _ { i } e _ { j } ^ { T } e ^ { W ( t - t ^ { \prime } ) } { \boldsymbol { S } } e ^ { W ^ { T } t } { \boldsymbol { v } } + \int _ { 0 } ^ { t } d t ^ { \prime } ~ } { \boldsymbol { v } } ^ { T } e ^ { W t } { \boldsymbol { S } } e ^ { W ^ { T } t ^ { \prime } } e _ { j } e _ { i } ^ { T } e ^ { W ^ { T } ( t - t ^ { \prime } ) } { \boldsymbol { v } }  \\ { { \displaystyle = \int _ { 0 } ^ { t } d t ^ { \prime } ~ e ^ { W ^ { T } t ^ { \prime } } { \boldsymbol { v } } { \boldsymbol { v } } ^ { T } e ^ { W t } { \boldsymbol { S } } e ^ { W ^ { T } ( t - t ^ { \prime } ) } + \int _ { 0 } ^ { t } d t ^ { \prime } ~ e ^ { W ^ { T } ( t - t ^ { \prime } ) } { \boldsymbol { v } } { \boldsymbol { v } } ^ { T } e ^ { W t } { \boldsymbol { S } } e ^ { W ^ { T } t ^ { \prime } } } . } \end{array}\tag{43}
$$

## A.4 DERIVATION OF LEARNING DYNAMICS

Using the aforementioned derivatives, we can immediately compute the relevant gradients of $J { \boldsymbol { : } }$

$$
\begin{array} { r l } & { \displaystyle \frac { \partial J } { \partial u } = \int _ { 0 } ^ { \infty } \left[ J ( \ell ) - f _ { * } ( \ell ) \right] \frac { \partial J ( \ell ) } { \partial u } d t + \kappa u } \\ & { \quad = \int _ { 0 } ^ { \infty } \left[ J ( \ell ) - f _ { * } ( \ell ) \right] e ^ { - \gamma _ { \ell } } \epsilon ^ { W ^ { T } t } v ^ { W ^ { T } t } v d t + \kappa u } \\ & { \displaystyle \frac { \partial J } { \partial v } = \int _ { 0 } ^ { \infty } \left[ J ( \ell ) - f _ { * } ( \ell ) \right] \frac { \partial J ( \ell ) } { \partial v } + e ^ { - 2 \gamma _ { \ell } } e ^ { W ^ { T } t } S e ^ { W ^ { T } t } v d t + \kappa v } \\ & { \quad = \int _ { 0 } ^ { \infty } \left[ J ( \ell ) - f _ { * } ( \ell ) \right] e ^ { - \gamma _ { \ell } } \epsilon ^ { W ^ { T } t } \epsilon ^ { W ^ { T } t } u + \epsilon ^ { - 2 \gamma _ { \ell } } \epsilon ^ { W ^ { T } } S e ^ { W ^ { T } t } v d t + \kappa v } \\ & { \displaystyle \frac { \partial J } { \partial W } = \int _ { 0 } ^ { \infty } \left[ J ( \ell ) - f _ { * } ( \ell ) \right] e ^ { - \gamma _ { \ell } } \int _ { 0 } ^ { 1 } e ^ { W ^ { T } t } v ^ { W ^ { T } t } v ^ { W ^ { T } ( t - \ell ^ { \prime } ) } d t ^ { \prime } d t } \\ &  \quad \quad + \int _ { 0 } ^ { \infty } \frac { e ^ { - 2 \gamma _ { \ell } } } { 2 } \int _ { 0 } ^ { t } e ^ { W ^ { T } t } v ^ { W ^ { T } t } v ^ { W ^ { T } t } S e ^ { W ^ { T } ( t - \ell ^ { \prime } ) } d t ^ { \prime } d t + \int _ { 0 } ^ { \infty } \frac { e ^ { - 2 \gamma _ { \ell } } } { 2 } \int _ { 0 } ^ { t } e ^ { W ^ { T } ( t - t ^ { \prime } ) } v v ^ { W ^ { T } t } S e  \end{array}\tag{44}
$$

This result immediately implies that our gradient descent learning dynamics have the form

$$
\begin{array} { r l } & { \frac { d } { d \tau } u = \displaystyle \int _ { 0 } ^ { \infty } \left[ f _ { \star } ( t ) - f ( t ) \right] e ^ { - \gamma t } e ^ { W ^ { T } t } v ~ d t - \kappa u } \\ & { \frac { d } { d \tau } v = \displaystyle \int _ { 0 } ^ { \infty } \left[ f _ { \star } ( t ) - f ( t ) \right] e ^ { - \gamma t } e ^ { W t } u - e ^ { - 2 \gamma t } e ^ { W t } S e ^ { W ^ { T } t } v ~ d t - \kappa v } \\ & { \frac { d } { d \tau } W = \displaystyle \int _ { 0 } ^ { \infty } \left[ f _ { \star } ( t ) - f ( t ) \right] e ^ { - \gamma t } \int _ { 0 } ^ { t } e ^ { W ^ { T } t ^ { \prime } } v u ^ { T } e ^ { W ^ { T } ( t - t ^ { \prime } ) } ~ d t ^ { \prime } d t } \\ & { \phantom { \frac { d } { d \tau } } - \displaystyle \int _ { 0 } ^ { \infty } \frac { e ^ { - 2 \gamma t } } { 2 } \displaystyle \int _ { 0 } ^ { t } e ^ { W ^ { T } t ^ { \prime } } v v ^ { T } e ^ { W ^ { T } t } S e ^ { W ^ { T } ( t - t ^ { \prime } ) } ~ d t ^ { \prime } d t - \displaystyle \int _ { 0 } ^ { \infty } \frac { e ^ { - 2 \gamma t } } { 2 } \displaystyle \int _ { 0 } ^ { t } e ^ { W ^ { T } ( t - t ^ { \prime } ) } v v v ^ { T } e ^ { W t } S e ^ { W ^ { T } t ^ { \prime } } ~ d t ^ { \prime } d t . } \end{array}\tag{45}
$$

By a change of variables $t ^ { \prime } \to t - t ^ { \prime } .$ , one can show that the two noise terms in the $W$ dynamics are identical. This allows the final result to be written in a slightly more compact form:

$$
\begin{array} { l } { \displaystyle \frac { d } { d \tau } u = \int _ { 0 } ^ { \infty } [ f _ { * } ( t ) - f ( t ) ] e ^ { - \gamma t } e ^ { W ^ { T } t } v \ d t - \kappa u } \\ { \displaystyle \frac { d } { d \tau } v = \int _ { 0 } ^ { \infty } [ f _ { * } ( t ) - f ( t ) ] e ^ { - \gamma t } e ^ { W t } u - e ^ { - 2 \gamma t } e ^ { W t } S e ^ { W ^ { T } t } v \ d t - \kappa v } \\ { \displaystyle \frac { d } { d \tau } W = \int _ { 0 } ^ { \infty } [ f _ { * } ( t ) - f ( t ) ] e ^ { - \gamma t } \int _ { 0 } ^ { t } e ^ { W ^ { T } t ^ { \prime } } v u ^ { T } e ^ { W ^ { T } ( t - t ^ { \prime } ) } \ d t ^ { \prime } d t } \\ { \displaystyle \qquad - \int _ { 0 } ^ { \infty } e ^ { - 2 \gamma t } \int _ { 0 } ^ { t } e ^ { W ^ { T } t ^ { \prime } } v v ^ { T } e ^ { W t } S e ^ { W ^ { T } ( t - t ^ { \prime } ) } \ d t ^ { \prime } d t . } \end{array}\tag{46}
$$

## A.5 LEARNING DYNAMICS GIVEN SMALL INITIAL WEIGHTS

We can expand these learning dynamics to leading order in ${ \mathbf { } } u , v ,$ , and $W$ (supposing they are all of order $\epsilon > 0 , \mathrm { s a y } )$ . Doing so, we find that

$$
\begin{array} { l } { \displaystyle \frac { d } { d \tau } \boldsymbol { u } \approx \int _ { 0 } ^ { \infty } f _ { * } ( t ) e ^ { - \gamma t } \boldsymbol { v } d t - \kappa \boldsymbol { u } } \\ { \displaystyle \frac { d } { d \tau } \boldsymbol { v } \approx \int _ { 0 } ^ { \infty } f _ { * } ( t ) e ^ { - \gamma t } \boldsymbol { u } - e ^ { - 2 \gamma t } \boldsymbol { S } \boldsymbol { v } d t - \kappa \boldsymbol { v } } \\ { \displaystyle \frac { d } { d \tau } \boldsymbol { W } \approx \int _ { 0 } ^ { \infty } f _ { * } ( t ) e ^ { - \gamma t } \int _ { 0 } ^ { t } \boldsymbol { v } \boldsymbol { u } ^ { T } d t - \int _ { 0 } ^ { \infty } e ^ { - 2 \gamma t } \int _ { 0 } ^ { t } \boldsymbol { v } \boldsymbol { v } ^ { T } \boldsymbol { S } d t ^ { \prime } d t . } \end{array}\tag{47}
$$

We can be more explicit using the fact that $f _ { * } ( t ) = e ^ { - c _ { * } t }$ . Simplifying yields

$$
\begin{array} { l } { { \displaystyle { \frac { d } { d \tau } } u \approx \frac { 1 } { c _ { * } + \gamma } v - \kappa u } } \\ { { \displaystyle { \frac { d } { d \tau } } v \approx \frac { 1 } { c _ { * } + \gamma } u - \frac { 1 } { 2 \gamma } S v - \kappa v } } \\ { { \displaystyle { \frac { d } { d \tau } } W \approx \int _ { 0 } ^ { \infty } f _ { * } ( t ) e ^ { - \gamma t } \int _ { 0 } ^ { t } v u ^ { T } d t ^ { \prime } d t - \int _ { 0 } ^ { \infty } e ^ { - 2 \gamma t } \int _ { 0 } ^ { t } v v ^ { T } S d t ^ { \prime } d t } } \\ { { \displaystyle \approx \frac { 1 } { ( c _ { * } + \gamma ) ^ { 2 } } v u ^ { T } - \frac { 1 } { 4 \gamma ^ { 2 } } v v ^ { T } S . } } \end{array}\tag{48}
$$

## B RESERVOIR COMPUTING CASE

In this appendix, we work out the learning dynamics of the reservoir computing model in which only the readout v is trained. Its primary virtue is that it illustrates, even when learning dynamics are linear, how noise produces representational pressure. For simplicity, we ignore regularization and set $\kappa = 0$

Recall from Appendix A that v has dynamics

$$
\begin{array} { r l } { \displaystyle \frac { d } { d \tau } v = \int _ { 0 } ^ { \infty } [ f _ { * } ( t ) - f ( t ) ] e ^ { - \gamma t } e ^ { W t } u - e ^ { - 2 \gamma t } e ^ { W t } S e ^ { W ^ { T } t } v \ d t } & { } \\ { = \int _ { 0 } ^ { \infty } e ^ { - ( c _ { * } + \gamma ) t } e ^ { W t } u - e ^ { - 2 \gamma t } e ^ { W t } u u ^ { T } e ^ { W ^ { T } t } v - e ^ { - 2 \gamma t } e ^ { W t } S e ^ { W ^ { T } t } v \ d t } & { } \\ { = \left[ \int _ { 0 } ^ { \infty } e ^ { - ( c _ { * } + \gamma ) t } e ^ { W t } d t \right] u - \left[ \int _ { 0 } ^ { \infty } e ^ { - 2 \gamma t } e ^ { W t } ( u u ^ { T } + S ) e ^ { W ^ { T } t } d t \right] v . } \end{array}\tag{49}
$$

These dynamics are linear, and the matrix that controls which directions of v are learned first is

$$
M : = \int _ { 0 } ^ { \infty } e ^ { - 2 \gamma t } e ^ { W t } ( { \pmb u } { \pmb u } ^ { T } + { \pmb S } ) e ^ { W ^ { T } t } d t .\tag{50}
$$

The fixed point of these dynamics is

$$
\boxed  \begin{array} { r l } & { \boxed { v _ { * } = \left( \int _ { 0 } ^ { \infty } e ^ { - 2 \gamma t } e ^ { W t } ( u u ^ { T } + S ) e ^ { W ^ { T } t } d t \right) ^ { - 1 } \left( \int _ { 0 } ^ { \infty } e ^ { - ( c _ { * } + \gamma ) t } e ^ { W t } d t \right) u } \end{array} } \Biggr . \Biggr \vert _ { \mathbf { \Sigma } } .\tag{51}
$$

Informative simplification. Assume for simplicity that $_ { s }$ is diagonal and $\boldsymbol { W } = ( \gamma - c _ { * } ) \boldsymbol { I } ,$ , i.e., that all state-space directions are compatible with the target filter. In this case,

$$
\begin{array} { c } { { \displaystyle \int _ { 0 } ^ { \infty } e ^ { - ( c _ { * } + \gamma ) t } e ^ { W t } d t = \frac { 1 } { 2 c _ { * } } } } \\ { { \displaystyle \int _ { 0 } ^ { \infty } e ^ { - 2 \gamma t } e ^ { W t } ( u u ^ { T } + S ) e ^ { W ^ { T } t } d t = \frac { 1 } { 2 c _ { * } } ( u u ^ { T } + S ) } } \end{array}\tag{52}
$$

so the optimal readout is

$$
\boldsymbol { v } _ { * } = ( \boldsymbol { u } \boldsymbol { u } ^ { T } + \boldsymbol { S } ) ^ { - 1 } \boldsymbol { u } = \left( \boldsymbol { S } ^ { - 1 } - \frac { \boldsymbol { S } ^ { - 1 } \boldsymbol { u } \boldsymbol { u } ^ { T } \boldsymbol { S } ^ { - 1 } } { 1 + \boldsymbol { u } ^ { T } \boldsymbol { S } ^ { - 1 } \boldsymbol { u } } \right) \boldsymbol { u } = \frac { \boldsymbol { S } ^ { - 1 } \boldsymbol { u } } { 1 + \boldsymbol { u } ^ { T } \boldsymbol { S } ^ { - 1 } \boldsymbol { u } }\tag{53}
$$

For a large read-in vector ${ \mathbf { } } ^ { \mathbf { } } \mathbf { \Delta } ^ { \mathbf { } } \mathbf { u } ,$ this approximately equals

$$
{ v _ { * } } \approx \frac { S ^ { - 1 } u } { { u ^ { T } S ^ { - 1 } u } }\tag{54}
$$

which weighs each input direction by how noisy it is. This prescription also yields the correct optimal filter, since

$$
{ \pmb u } ^ { T } { \pmb v } _ { \ast } \approx \frac { { \pmb u } ^ { T } S ^ { - 1 } { \pmb u } } { { \pmb u } ^ { T } S ^ { - 1 } { \pmb u } } = 1 ,\tag{55}
$$

which implies

$$
\begin{array} { r } { f ( t ) = v _ { * } ^ { T } e ^ { A t } u = e ^ { - c _ { * } t } v _ { * } ^ { T } u \approx e ^ { - c _ { * } t } . } \end{array}\tag{56}
$$

This approximate solution is intuitive since it is precisely the choice of v that, for this choice of W and u, minimizes the quadratic noise term subject to the constraint that ${ \pmb u } ^ { T } { \pmb v } = 1$

## C ALIGNMENT DYNAMICS

In this appendix, we solve the dynamics associated with the alignment phase (see Sec. 4). Using the small-parameter gradient descent updates from Appendix A, we have

$$
\frac { d } { d \tau } u \approx \frac { v } { c _ { * } + \gamma } - \kappa u { \frac { d } { d \tau } v \approx \frac { u } { c _ { * } + \gamma } - \frac { S v } { 2 \gamma } - \kappa v } { \frac { d } { d \tau } W \approx \frac { v u ^ { T } } { ( c _ { * } + \gamma ) ^ { 2 } } - \frac { v v ^ { T } S } { 4 \gamma ^ { 2 } } } .\tag{57}
$$

Because u and v dynamics form a closed system, we will begin by studying u and v only. Once we determine how u and v change with time, we will use them to determine how W changes.

## C.1 SOLVING READ-IN AND READOUT DYNAMICS

Naively, u and v together determine a 2N-dimensional system of ODEs, since

$$
\begin{array} { l } { \displaystyle \frac { d } { d \tau } \pmb { u } = - \kappa \pmb { u } + \frac { \pmb { v } } { c _ { * } + \gamma } } \\ { \displaystyle \frac { d } { d \tau } \pmb { v } = \frac { \pmb { u } } { c _ { * } + \gamma } - \left( \kappa + \frac { \pmb { S } } { 2 \gamma } \right) \pmb { v } . } \end{array}\tag{58}
$$

Note that, to ease notation, we will write equalities instead of .

This system substantially simplifies if we work in a basis of eigenvectors of S. Along the i-th such eigenvector, if $s _ { i } > 0$ denotes the corresponding eigenvalue,

$$
\begin{array} { l } { \displaystyle \frac { d } { d \tau } u _ { i } = - \kappa u _ { i } + \frac { v _ { i } } { c _ { * } + \gamma } } \\ { \displaystyle \frac { d } { d \tau } v _ { i } = \frac { u _ { i } } { c _ { * } + \gamma } - \left( \kappa + \frac { s _ { i } } { 2 \gamma } \right) v _ { i } . } \end{array}\tag{59}
$$

Relevant eigenvalues and eigenvectors. This is just a two-dimensional system governed by the $2 \times 2$ matrix

$$
\begin{array} { r } { M : = \left( \begin{array} { c c } { - \kappa } & { \frac { 1 } { c _ { * } + \gamma } } \\ { \frac { 1 } { c _ { * } + \gamma } } & { - \left( \kappa + \frac { s _ { i } } { 2 \gamma } \right) } \end{array} \right) } \end{array}\tag{60}
$$

whose characteristic equation is

$$
\left( \lambda + \kappa \right) \left( \lambda + \kappa + \frac { s _ { i } } { 2 \gamma } \right) - \frac { 1 } { ( c _ { * } + \gamma ) ^ { 2 } } = 0 .\tag{61}
$$

Equivalently,

$$
\lambda ^ { 2 } + 2 \left[ \kappa + \frac { s _ { i } } { 4 \gamma } \right] \lambda + \left[ \kappa ^ { 2 } + \frac { \kappa s _ { i } } { 2 \gamma } - \frac { 1 } { ( c _ { * } + \gamma ) ^ { 2 } } \right] = 0 .\tag{62}
$$

The solutions of this equation (i.e., the eigenvalues of M) are

$$
\begin{array} { l } { { \lambda _ { i } ^ { \pm } = - \left( \kappa + \displaystyle \frac { s _ { i } } { 4 \gamma } \right) \pm \sqrt { \left( \kappa + \displaystyle \frac { s _ { i } } { 4 \gamma } \right) ^ { 2 } + \displaystyle \frac { 1 } { ( c _ { * } + \gamma ) ^ { 2 } } - \kappa ^ { 2 } - \displaystyle \frac { \kappa s _ { i } } { 2 \gamma } } } } \\ { { = - \left( \kappa + \displaystyle \frac { s _ { i } } { 4 \gamma } \right) \pm \sqrt { \displaystyle \frac { s _ { i } ^ { 2 } } { 1 6 \gamma ^ { 2 } } + \displaystyle \frac { 1 } { ( c _ { * } + \gamma ) ^ { 2 } } } . } } \end{array}\tag{63}
$$

Note that $\lambda _ { i } ^ { - }$ is always negative, and so corresponds to decay. The other root may be positive or negative depending on the size of the regularization strength $\kappa \geq 0$ . Beyond a critical κ value

$$
\kappa _ { \mathrm { c r i t } } : = \sqrt { \frac { s _ { \mathrm { m i n } } ^ { 2 } } { 1 6 \gamma ^ { 2 } } + \frac { 1 } { ( c _ { * } + \gamma ) ^ { 2 } } } - \frac { s _ { \mathrm { m i n } } } { 4 \gamma } ,\tag{64}
$$

where $s _ { \mathrm { m i n } }$ is the smallest of the $s _ { i } .$ , all modes decay and one obtains only the trivial zero solution for all $u _ { i }$ and $v _ { i }$ . For $\kappa < \kappa _ { \mathrm { c r i t } }$ , at least some modes survive.

Assume in what follows that $\kappa < \kappa _ { \mathrm { c r i t } }$ . Then at least some ${ \lambda } _ { i } ^ { + }$ will be positive, which means that $u _ { i }$ and $v _ { i }$ along those directions have components which grow with time. In particular, since the eigenvectors of M are

$$
\begin{array} { r } { \pmb q ^ { \pm } : = \left( \begin{array} { c } { 1 } \\ { ( c _ { * } + \gamma ) ( \kappa + \lambda ^ { \pm } ) , } \end{array} \right) } \end{array}\tag{65}
$$

the solution $z ( \tau ) : = ( u _ { i } ( \tau ) , v _ { i } ( \tau ) ) ^ { T }$ equals

$$
z = e ^ { M \tau } z _ { 0 } = e ^ { M \tau } ( c _ { + } q ^ { + } + c _ { - } q ^ { - } ) = c _ { + } q ^ { + } e ^ { \lambda _ { i } ^ { + } \tau } + c _ { - } q ^ { - } e ^ { \lambda _ { - } ^ { i } \tau }\tag{66}
$$

for some coefficients $c _ { + }$ and $c _ { - }$ .

Enforcing initial conditions. We can determine $c _ { + }$ and $c _ { - }$ by choosing them so that $z ( 0 ) =$ $( u _ { i } ( 0 ) , v _ { i } \overline { { ( 0 ) } } ) ^ { T }$ . This requires enforcing

$$
\begin{array} { l } { { u _ { i } ( 0 ) = c _ { + } + c _ { - } } } \\ { { v _ { i } ( 0 ) = c _ { + } ( c _ { * } + \gamma ) ( \kappa + \lambda ^ { + } ) + c _ { - } ( c _ { * } + \gamma ) ( \kappa + \lambda ^ { - } ) . } } \end{array}\tag{67}
$$

The first condition implies $c _ { - } = u _ { i } ( 0 ) - c _ { + }$ . The second implies

$$
\begin{array} { l } { \displaystyle \frac { v _ { i } ( 0 ) } { c _ { * } + \gamma } = c _ { + } ( \kappa + \lambda ^ { + } ) + c _ { - } ( \kappa + \lambda ^ { - } ) } \\ { \displaystyle = c _ { + } ( \kappa + \lambda ^ { + } ) + [ u _ { i } ( 0 ) - c _ { + } ] ( \kappa + \lambda ^ { - } ) } \\ { \displaystyle = u _ { i } ( 0 ) ( \kappa + \lambda ^ { - } ) + c _ { + } ( \lambda ^ { + } - \lambda ^ { - } ) } \end{array}\tag{68}
$$

so

$$
\left| ~ c _ { + } = \frac { 1 } { \lambda _ { i } ^ { + } - \lambda _ { i } ^ { - } } \left[ \frac { v _ { i } ( 0 ) } { c _ { * } + \gamma } - u _ { i } ( 0 ) ( \kappa + \lambda _ { i } ^ { - } ) \right] \right| .\tag{69}
$$

Because the mode which grows as opposed to shrinks will dominate after enough time has passed, to good approximation we have

$$
\begin{array} { l } { \displaystyle { u _ { i } ( \tau ) \approx e ^ { \lambda _ { i } ^ { + } \tau } c _ { + } = \frac { 1 } { \lambda _ { i } ^ { + } - \lambda _ { i } ^ { - } } \left[ \frac { v _ { i } ( 0 ) } { c _ { * } + \gamma } - u _ { i } ( 0 ) ( \kappa + \lambda _ { i } ^ { - } ) \right] e ^ { \lambda _ { i } ^ { + } \tau } } } \\ { \displaystyle { v _ { i } ( \tau ) \approx e ^ { \lambda _ { i } ^ { + } \tau } c _ { + } ( c _ { * } + \gamma ) ( \kappa + \lambda ^ { + } ) = \frac { ( c _ { * } + \gamma ) ( \kappa + \lambda ^ { + } ) } { \lambda _ { i } ^ { + } - \lambda _ { i } ^ { - } } \left[ \frac { v _ { i } ( 0 ) } { c _ { * } + \gamma } - u _ { i } ( 0 ) ( \kappa + \lambda _ { i } ^ { - } ) \right] e ^ { \lambda _ { i } ^ { + } \tau } } } \end{array}\tag{70}
$$

for moderate τ.

Note that ${ \lambda } _ { i } ^ { + }$ is a decreasing function of $s _ { i }$ . This implies that u and v will tend to get amplified more quickly along less noisy directions. After enough time has passed, the coefficients along the least noisy direction will tend to dwarf the others, so to good approximation we have

$$
\begin{array} { r } { \pmb { u } ( \tau ) \approx \frac { \pmb { u } ( \tau ) } { 2 } \pmb { P } \left[ \frac { \pmb { v } ( 0 ) } { c _ { * } + \gamma } - \pmb { u } ( 0 ) ( \kappa + \lambda _ { \mathrm { m i n } } ^ { - } ) \right] } \\ { \pmb { v } ( \tau ) \approx \frac { \pmb { v } ( \tau ) } { 2 } \pmb { P } \left[ \frac { \pmb { v } ( 0 ) } { c _ { * } + \gamma } - \pmb { u } ( 0 ) ( \kappa + \lambda _ { \mathrm { m i n } } ^ { - } ) \right] } \end{array}\tag{71}
$$

after the alignment phase, where $\lambda _ { \operatorname* { m i n } } ^ { - }$ is the negative eigenvalue associated with the least noisy direction $( \mathrm { i . e . }$ , the smallest $s _ { i }$ , which we can call $s _ { \mathrm { m i n } } )$ . Here, $_ { P }$ is a matrix that projects into the least noisy subspace (which may be of arbitrary dimensionality) and $u ( \tau )$ and $v ( \tau )$ are mode amplitudes whose precise form is given above. Equivalently, ${ \pmb u } ( \tau )$ and $v ( \tau )$ align in the least noisy state-space direction.

Define the quantity

$$
\Delta : = \sqrt { \frac { s _ { \mathrm { m i n } } ^ { 2 } } { 1 6 \gamma ^ { 2 } } + \frac { 1 } { ( c _ { * } + \gamma ) ^ { 2 } } } - \frac { 1 } { c _ { * } + \gamma } \ : ,\tag{72}
$$

which is an increasing function of $s _ { \mathrm { m i n } } . \Delta$ quantifies the gap between the noise-free solution and the noisy solution. In terms of it, we can write our answer as

$$
\begin{array} { r l } & { u ( \tau ) \approx u ( \tau ) P \left[ \frac { \left[ u ( 0 ) + v ( 0 ) \right] } { 2 } + \frac { ( c _ { * } + \gamma ) } { 2 } \left( \frac { s _ { \mathrm { m i n } } } { 4 \gamma } + \Delta ( s _ { \mathrm { m i n } } ) \right) u ( 0 ) \right] } \\ & { v ( \tau ) \approx v ( \tau ) P \left[ \frac { \left[ u ( 0 ) + v ( 0 ) \right] } { 2 } + \frac { ( c _ { * } + \gamma ) } { 2 } \left( \frac { s _ { \mathrm { m i n } } } { 4 \gamma } + \Delta ( s _ { \mathrm { m i n } } ) \right) u ( 0 ) \right] . } \end{array}\tag{73}
$$

## C.2 W DYNAMICS

Given u and v dynamics, W dynamics are extremely simple. Because u and v are, to good approximation, collinear, we can write

$$
{ \pmb u } ( \tau ) \approx { \pmb u } ( \tau ) { \pmb u } _ { 0 }
$$

$$
\pmb { v } ( \tau ) \approx v ( \tau ) \pmb { u } _ { 0 }\tag{74}
$$

for some vector $\mathbf { \delta } \mathbf { u } _ { 0 }$ (which is an eigenvector of $_ { s }$ with eigenvalue $\sigma ^ { 2 } )$ and scalar amplitudes $u ( \tau )$ and $v ( \tau )$ . Eq. 57 then implies that

$$
\frac { d } { d \tau } { \cal W } \approx \left[ \frac { u ( \tau ) v ( \tau ) } { ( c _ { * } + \gamma ) ^ { 2 } } - \frac { \sigma ^ { 2 } v ( \tau ) ^ { 2 } } { 4 \gamma ^ { 2 } } \right] u _ { 0 } u _ { 0 } ^ { T } ,\tag{75}
$$

or equivalently that

$$
\begin{array} { r } { \pmb { W } ( \tau ) \approx [ \gamma - c ( \tau ) ] \pmb { u } _ { 0 } \pmb { u } _ { 0 } ^ { T } } \end{array}\tag{76}
$$

since these updates are all in the same direction. $c ( \tau )$ is a scalar which determines the overall amplitude of $W$ . In other words, $W$ becomes a rank-one matrix aligned in the same direction as u and v.

## D DERIVATION OF LOW-DIMENSIONAL EFFECTIVE DYNAMICS

In this appendix, we derive the effective learning dynamics associated with the post-alignment phase. Recall from Appendix C and Sec. 4 that the outcome of alignment has

<table><tr><td> ${ \pmb u } ( \tau ) \approx { \pmb u } ( \tau ) { \pmb u } _ { 0 }$   $\pmb { v } ( \tau ) \approx v ( \tau ) \pmb { u } _ { 0 }$ </td></tr></table>

(77)

for some (assumed normalized) $\mathbf { \delta } \mathbf { u } _ { 0 }$ which is an eigenvector of S with $S u _ { 0 } = \sigma ^ { 2 } u _ { 0 }$ . As we found earlier, this ought to be the least noisy state-space direction.

We will call this the alignment ansatz. We want to show that dynamics are closed under this ansatz, in the sense that it suffices to track the time evolution of the scalars $u , v ,$ and c. We will also show that gradient descent dynamics on the full objective become equivalent to gradient descent dynamics on a simpler effective objective.

Note that, under the alignment ansatz,

$$
\begin{array} { r } { f ( t ) = { \pmb v } ^ { T } e ^ { { \pmb A } t } { \pmb u } = u ( \tau ) v ( \tau ) e ^ { - c ( \tau ) t } . } \end{array}\tag{78}
$$

## D.1 DYNAMICS ARE CLOSED UNDER THE ALIGNMENT ANSATZ

u dynamics. General u dynamics have the form

$$
\frac { d } { d \tau } \pmb { u } = \int _ { 0 } ^ { \infty } [ f _ { * } ( t ) - f ( t ) ] e ^ { - \gamma t } e ^ { \pmb { W } ^ { T } t } \pmb { v } d t - \kappa \pmb { u } .\tag{79}
$$

Under our ansatz, this simplifies to

$$
\begin{array} { l } { \displaystyle \frac { d } { d \tau } u ( \tau ) u _ { 0 } = \left[ \int _ { 0 } ^ { \infty } [ f _ { * } ( t ) - f ( t ) ] e ^ { - c ( \tau ) t } v ( \tau ) d t - \kappa u ( \tau ) \right] u _ { 0 } } \\ { \displaystyle \qquad = \left[ \int _ { 0 } ^ { \infty } [ e ^ { - c _ { * } t } - u ( \tau ) v ( \tau ) e ^ { - c ( \tau ) t } ] e ^ { - c ( \tau ) t } v ( \tau ) d t - \kappa u ( \tau ) \right] u _ { 0 } } \\ { \displaystyle \qquad = \left[ \frac { v ( \tau ) } { c _ { * } + c ( \tau ) } - \frac { u ( \tau ) v ( \tau ) ^ { 2 } } { 2 c ( \tau ) } - \kappa u ( \tau ) \right] u _ { 0 } . } \end{array}\tag{80}
$$

We indeed find that only the scalar $u ( \tau )$ changes, as desired.

v dynamics. General v dynamics have the form

$$
{ \frac { d } { d \tau } } v = \int _ { 0 } ^ { \infty } [ f _ { * } ( t ) - f ( t ) ] e ^ { - \gamma t } e ^ { W t } u - e ^ { - 2 \gamma t } e ^ { W t } S e ^ { W ^ { T } t } v ~ d t - \kappa v ~ .\tag{81}
$$

Under the ansatz,

$$
\begin{array} { l } { \displaystyle \frac { d } { d \tau } v ( \tau ) u _ { 0 } = \left[ \int _ { 0 } ^ { \infty } [ f _ { * } ( t ) - f ( t ) ] u ( \tau ) e ^ { - c ( \tau ) t } - \sigma ^ { 2 } v ( \tau ) e ^ { - 2 c ( \tau ) t } d t - \kappa v ( \tau ) \right] u _ { 0 } } \\ { \displaystyle = \left[ \int _ { 0 } ^ { \infty } [ e ^ { - c _ { * } t } - u ( \tau ) v ( \tau ) e ^ { - c ( \tau ) t } ] u ( \tau ) e ^ { - c ( \tau ) t } - \sigma ^ { 2 } v ( \tau ) e ^ { - 2 c ( \tau ) t } d t - \kappa v ( \tau ) \right] u _ { 0 } } \\ { \displaystyle = \left[ \frac { u ( \tau ) } { c _ { * } + c ( \tau ) } - \frac { [ u ( \tau ) ^ { 2 } + \sigma ^ { 2 } ] v ( \tau ) } { 2 c ( \tau ) } - \kappa v ( \tau ) \right] u _ { 0 } } \end{array}\tag{82}
$$

where $\sigma ^ { 2 } : = \pmb { u } _ { 0 } ^ { T } S \pmb { u } _ { 0 }$ is the noise level in the alignment direction.

W dynamics. General W dynamics have the form

$$
\begin{array} { r l r } {  { \frac { d } { d \tau } \boldsymbol { W } = \int _ { 0 } ^ { \infty } [ f _ { * } ( t ) - f ( t ) ] e ^ { - \gamma t } \int _ { 0 } ^ { t } e ^ { \boldsymbol { W } ^ { T } t ^ { \prime } } \boldsymbol { v } \boldsymbol { u } ^ { T } e ^ { \boldsymbol { W } ^ { T } ( t - t ^ { \prime } ) } \ d t ^ { \prime } d t } } \\ & { } & { - \int _ { 0 } ^ { \infty } e ^ { - 2 \gamma t } \int _ { 0 } ^ { t } \ e ^ { \boldsymbol { W } ^ { T } t ^ { \prime } } \boldsymbol { v } \boldsymbol { v } ^ { T } e ^ { \boldsymbol { W } t } \boldsymbol { S } e ^ { \boldsymbol { W } ^ { T } ( t - t ^ { \prime } ) } \ d t ^ { \prime } d t . } \end{array}\tag{83}
$$

Under the ansatz,

$$
\begin{array} { c } { { \displaystyle - \frac { d } { d \tau } c ( \tau ) { \boldsymbol u } _ { 0 } { \boldsymbol u } _ { 0 } ^ { T } = u ( \tau ) { \boldsymbol v } ( \tau ) \left[ \int _ { 0 } ^ { \infty } [ f _ { * } ( t ) - f ( t ) ] \ t e ^ { - c ( \tau ) t } d t \right] { \boldsymbol u } _ { 0 } { \boldsymbol u } _ { 0 } ^ { T } } } \\ { { \displaystyle - \sigma ^ { 2 } { \boldsymbol v } ( \tau ) ^ { 2 } \int _ { 0 } ^ { \infty } t e ^ { - 2 c ( \tau ) t } d t { \boldsymbol u } _ { 0 } { \boldsymbol u } _ { 0 } ^ { T } \ . } } \end{array}\tag{84}
$$

Simplifying further,

$$
\begin{array} { c } { { \displaystyle - \frac { d } { d \tau } c ( \tau ) { \boldsymbol u } _ { 0 } { \boldsymbol u } _ { 0 } ^ { T } = { \boldsymbol u } ( \tau ) { \boldsymbol v } ( \tau ) \left[ \frac { 1 } { [ c _ { * } + c ( \tau ) ] ^ { 2 } } - \frac { { \boldsymbol u } ( \tau ) { \boldsymbol v } ( \tau ) } { 4 c ( \tau ) ^ { 2 } } \right] { \boldsymbol u } _ { 0 } { \boldsymbol u } _ { 0 } ^ { T } } } \\ { { \displaystyle - \frac { \sigma ^ { 2 } v ( \tau ) ^ { 2 } } { 4 c ( \tau ) ^ { 2 } } { \boldsymbol u } _ { 0 } { \boldsymbol u } _ { 0 } ^ { T } ~ , } } \end{array}\tag{85}
$$

or

$$
\frac { d } { d \tau } c ( \tau ) { \mathbf { } u _ { 0 } } { \mathbf { } u _ { 0 } ^ { T } } = \left( \frac { v ( \tau ) ^ { 2 } [ u ( \tau ) ^ { 2 } + \sigma ^ { 2 } ] } { 4 c ( \tau ) ^ { 2 } } - \frac { u ( \tau ) v ( \tau ) } { [ c _ { * } + c ( \tau ) ] ^ { 2 } } \right) { \mathbf { } u _ { 0 } } { \mathbf { } u _ { 0 } ^ { T } } \mathrm { ~ . ~ }\tag{86}
$$

In all cases, we indeed discover that the dynamics are closed under this ansatz. In summary, because the direction $\mathbf { \delta } \mathbf { u } _ { 0 }$ never changes under gradient descent dynamics, it suffices to study the reduced system

$$
\begin{array} { l } { { \displaystyle { \frac { d } { d \tau } } u ( \tau ) = { \frac { v ( \tau ) } { c _ { * } + c ( \tau ) } } - { \frac { u ( \tau ) v ( \tau ) ^ { 2 } } { 2 c ( \tau ) } } - \kappa u ( \tau ) } } \\ { { \displaystyle { \frac { d } { d \tau } } v ( \tau ) = { \frac { u ( \tau ) } { c _ { * } + c ( \tau ) } } - { \frac { v ( \tau ) [ \sigma ^ { 2 } + u ( \tau ) ^ { 2 } ] } { 2 c ( \tau ) } } - \kappa v ( \tau ) } } \\ { { \displaystyle { \frac { d } { d \tau } } c ( \tau ) = { \frac { v ( \tau ) ^ { 2 } [ u ( \tau ) ^ { 2 } + \sigma ^ { 2 } ] } { 4 c ( \tau ) ^ { 2 } } } - { \frac { u ( \tau ) v ( \tau ) } { [ c _ { * } + c ( \tau ) ] ^ { 2 } } } . } } \end{array}\tag{87}
$$

## D.2 REDUCED DYNAMICS ARE GRADIENT DESCENT ON AN EFFECTIVE OBJECTIVE

We can make an additional theoretically interesting point about this ansatz: not only is it dynamically closed, but the above ODEs are equivalent to gradient descent on an effective objective function. Fortunately, this objective is exactly what one obtains by naively substituting the ansatz into the original objective.

Recall that the full objective equals

$$
\small J ( u , v , W ) = \int _ { 0 } ^ { \infty } \frac { 1 } { 2 } \left[ f ( t ) - f _ { * } ( t ) \right] ^ { 2 } + \frac { e ^ { - 2 \gamma t } } { 2 } v ^ { T } e ^ { W t } S e ^ { W ^ { T } t } v ~ d t + \frac { \kappa } { 2 } \| u \| ^ { 2 } + \frac { \kappa } { 2 } \| v \| ^ { 2 } .\tag{88}
$$

Substituting our ansatz, it becomes

$$
\begin{array} { c l } { { } } & { { \displaystyle \frac { 1 } { 2 } \int _ { 0 } ^ { \infty } \Big \{ \left[ u v e ^ { - c t } - e ^ { - c _ { * } t } \right] ^ { 2 } + \sigma ^ { 2 } v ^ { 2 } e ^ { - 2 c t } \Big \} d t + \frac { \kappa } { 2 } ( u ^ { 2 } + v ^ { 2 } ) } } \\ { { } } & { { = \displaystyle \frac { 1 } { 2 } \int _ { 0 } ^ { \infty } \Big \{ u ^ { 2 } v ^ { 2 } e ^ { - 2 c t } - 2 u v e ^ { - ( c _ { * } + c ) t } + e ^ { - 2 c _ { * } t } + \sigma ^ { 2 } v ^ { 2 } e ^ { - 2 c t } \Big \} d t + \frac { \kappa } { 2 } ( u ^ { 2 } + v ^ { 2 } ) \Big \} d t } } \\ { { } } & { { = \displaystyle \frac { 1 } { 2 } \left\{ u ^ { 2 } v ^ { 2 } \frac { 1 } { 2 c } - 2 u v \frac { 1 } { c _ { * } + c } + \frac { 1 } { 2 c _ { * } } + \sigma ^ { 2 } v ^ { 2 } \frac { 1 } { 2 c } \right\} + \frac { \kappa } { 2 } ( u ^ { 2 } + v ^ { 2 } ) . } } \end{array}\tag{89}
$$

Simplifying, we obtain

$$
\begin{array} { l } { \displaystyle { J = \frac { u ( \tau ) ^ { 2 } v ( \tau ) ^ { 2 } } { 4 c ( \tau ) } - \frac { u ( \tau ) v ( \tau ) } { c _ { * } + c ( \tau ) } + \frac { 1 } { 4 c _ { * } } + \frac { \sigma ^ { 2 } v ( \tau ) ^ { 2 } } { 4 c ( \tau ) } + \frac { \kappa } { 2 } ( u ^ { 2 } + v ^ { 2 } ) \eqno ( 9 0 ) } } \\ { \displaystyle { \phantom { \frac { J } { J } } = \frac { 1 } { 4 c ( \tau ) } \left[ u ( \tau ) v ( \tau ) - \frac { 2 c ( \tau ) } { c _ { * } + c ( \tau ) } \right] ^ { 2 } + \left( \frac { 1 } { 4 c _ { * } } - \frac { c ( \tau ) } { [ c _ { * } + c ( \tau ) ] ^ { 2 } } \right) + \frac { \sigma ^ { 2 } v ( \tau ) ^ { 2 } } { 4 c ( \tau ) } + \frac { \kappa } { 2 } [ u ( \tau ) ^ { 2 } + v ( \tau ) ^ { 2 } ] . } } \end{array}
$$

It is straightforward to verify that the above ODEs directly correspond to gradient descent on this objective, which establishes our claim.

In this appendix, we characterize the noise-induced phase transition associated with amplitude collapse (see Sec. 6). The objective J has the form

$$
J = \frac { 1 } { 4 c } \left( u v - \frac { 2 c } { c _ { * } + c } \right) ^ { 2 } + \left( \frac { 1 } { 4 c _ { * } } - \frac { c } { ( c _ { * } + c ) ^ { 2 } } \right) + \frac { \sigma ^ { 2 } v ^ { 2 } } { 4 c } + \frac { \kappa } { 2 } ( u ^ { 2 } + v ^ { 2 } ) .\tag{91}
$$

This amounts to determining the conditions under which J has a nontrivial optimum.

## E.1 MINIMIZING J WITH RESPECT TO ITS ARGUMENTS

Let us begin by trying to optimize J with respect to its arguments—although we will find it useful to work in terms of slightly different variables. Define the filter amplitude $N : = u v$ and the scale ratio $r : = u / v$ . Since $\mathbf { \bar { \boldsymbol { v } } } ^ { 2 } = N / r$ and ${ u ^ { 2 } } = N r$ , we can write J as

$$
J ( r , N , c ) = \frac { 1 } { 4 c } \left( N - \frac { 2 c } { c _ { * } + c } \right) ^ { 2 } + \left( \frac { 1 } { 4 c _ { * } } - \frac { c } { ( c _ { * } + c ) ^ { 2 } } \right) + \frac { \sigma ^ { 2 } N } { 4 c r } + \frac { \kappa } { 2 } N \left( \frac { 1 } { r } + r \right) .\tag{92}
$$

Minimizing with respect to r. The derivative of J with respect to r equals

$$
\frac { \partial J } { \partial r } = \frac { \kappa } { 2 } N - \frac { \kappa } { 2 } N \left( 1 + \frac { \sigma ^ { 2 } } { 2 \kappa c } \right) \frac { 1 } { r ^ { 2 } } = 0\tag{93}
$$

which implies that

$$
r ^ { 2 } = 1 + \frac { \sigma ^ { 2 } } { 2 \kappa c }\tag{94}
$$

at the minimum. Note that it depends on c but not on N. In a slight abuse of notation, below we will use r to denote the value of the scale ratio at the minimum. Without loss of generality, choose the positive solution so that

$$
\left| \begin{array} { l } { \displaystyle r = \sqrt { 1 + \frac { \sigma ^ { 2 } } { 2 \kappa c } } } \end{array} \right| .\tag{95}
$$

Since r is defined as the ratio of u and v, this immediately implies that u and v have the same sign at a nontrivial minimum. Relatedly, it must be true that $N \geq 0$ , since $N = u ^ { 2 } / r \geq 0$

Minimizing with respect to N. The derivative of J with respect to N equals

$$
\begin{array} { c } { { \displaystyle \frac { \partial J } { \partial N } = \frac { 1 } { 2 c } \left( N - \frac { 2 c } { c _ { * } + c } \right) + \frac { \kappa } { 2 } \left( 1 + \frac { \sigma ^ { 2 } } { 2 \kappa c } \right) \frac 1 r + \frac { \kappa } { 2 } r } } \\ { { = \displaystyle \frac { 1 } { 2 c } \left( N - \frac { 2 c } { c _ { * } + c } \right) + \kappa r . } } \end{array}\tag{96}
$$

Setting the left-hand side equal to zero, it appears true that

$$
N = \frac { 2 c } { c _ { * } + c } - 2 \kappa r c\tag{97}
$$

at the minimum. But since our r minimization implied that $N \geq 0$ , we actually have

$$
N = 2 c \left[ \frac { 1 } { c _ { * } + c } - \kappa r \right] _ { + }\tag{98}
$$

where $[ \cdots ] _ { + }$ denotes rectification (i.e., a ReLU). That is, if the bracketed quantity is negative or zero, J is minimized when $N = 0$ . This is a sign of amplitude collapse—if $N = 0$ is optimal, it is optimal to take $f ( t ) = 0$

If we define the quantity

$$
\boxed { h ( c ) : = \frac { 1 } { c _ { * } + c } - \kappa r } ,\tag{99}
$$

we can write the optimal N as

$$
\boxed { N = 2 c [ h ( c ) ] _ { + } } .\tag{100}
$$

Rewriting the objective in terms of N. In terms of N, the objective can be written at its minimum as

$$
\begin{array} { l } { \displaystyle { J ( c ) = c \left( \frac { 1 } { c _ { * } + c } - [ h ( c ) ] _ { + } \right) ^ { 2 } + \frac { 1 } { 4 c _ { * } } - \frac { c } { ( c _ { * } + c ) ^ { 2 } } + 2 \kappa c [ h ( c ) ] _ { + } r ( c ) } } \\ { \displaystyle { ~ = \frac { 1 } { 4 c _ { * } } + c [ h ( c ) ] _ { + } \left( - \frac { 2 } { c _ { * } + c } + [ h ( c ) ] _ { + } \right) + 2 \kappa c [ h ( c ) ] _ { + } r ( c ) } } \\ { \displaystyle { ~ = \frac { 1 } { 4 c _ { * } } + c [ h ( c ) ] _ { + } \left( 2 \kappa r ( c ) - \frac { 2 } { c _ { * } + c } + [ h ( c ) ] _ { + } \right) . } } \end{array}\tag{101}
$$

Either $h ( c ) \geq 0$ , in which case the bracketed quantity equals $- h ( c )$ , or $h ( c ) < 0$ , in which case the $\left[ h ( c ) \right] _ { + }$ prefactor makes the entire quantity zero. We conclude that

$$
\boxed { J ( c ) = \frac { 1 } { 4 c _ { * } } - c [ h ( c ) ] _ { + } ^ { 2 } }\tag{102}
$$

at the minimum. This immediately tells us that if $h < 0$ for all $c \geq 0 ,$ , there is a trivial solution with $N = 0$ and J independent of c (see Fig. 5c).

## E.2 DETERMINING THE PHASE BOUNDARY

While it appears prohibitively difficult to derive the exact c that minimizes J (since this requires solving a high-order polynomial equation), we can instead determine where there is a transition from a nontrivial solution to a trivial one.

Phase boundary conditions. Eq. 102 implies that a nontrivial solution exists if and only $\mathrm { i f } h ( c ) >$ 0 for some $c > 0$ , since this yields $N > 0$ and a nontrivial value $c > 0$ at the minimum. On the other hand, if $h ( c ) \leq 0$ for all $c > 0$ , then $N = 0$ and the trivial solution is optimal. The phase boundary therefore occurs when the maximum of $h ( c )$ just touches zero. Hence, the phase boundary is characterized by the two conditions

$$
\begin{array} { r } { \boxed { \begin{array} { c c c c } { h ( c ) = 0 } & { } & { } & { h ^ { \prime } ( c ) = 0 } \end{array} } . } \end{array}\tag{103}
$$

Explicitly, we want

$$
\begin{array} { c } { { \displaystyle \frac { 1 } { c _ { * } + c } = \kappa \sqrt { 1 + \frac { \sigma ^ { 2 } } { 2 \kappa c } } } } \\ { { \displaystyle \frac { 1 } { ( c _ { * } + c ) ^ { 2 } } = \kappa \frac { \frac { \sigma ^ { 2 } } { 2 \kappa c ^ { 2 } } } { 2 \sqrt { 1 + \frac { \sigma ^ { 2 } } { 2 \kappa c } } } . } } \end{array}\tag{104}
$$

Relatedly, a nontrivial solution only exists when $c _ { * } \kappa < 1$ , since for $c _ { * } \kappa \geq 1$

$$
h ( c ) = \frac { 1 } { c _ { * } + c } - \kappa r \leq \frac { 1 } { c _ { * } + c } - \kappa < \frac { 1 } { c _ { * } } - \kappa \leq 0\tag{105}
$$

for all $c > 0$ . In this regime, because $h ( c )$ is always nonpositive, $\sigma _ { \mathrm { c r i t } } ^ { 2 } = 0 .$

Some algebra. It will be helpful to write $\sigma ^ { 2 }$ and κ separately in terms of c. Divide the second equation by the first to obtain

$$
{ \frac { 1 } { c _ { * } + c } } = { \frac { \sigma ^ { 2 } } { 4 \kappa c ^ { 2 } r ^ { 2 } } } .\tag{106}
$$

Rearranging,

$$
r ^ { 2 } = 1 + \frac { \sigma ^ { 2 } } { 2 \kappa c } = \frac { \sigma ^ { 2 } } { 4 \kappa c ^ { 2 } } ( c _ { * } + c ) \implies 1 + \frac { \sigma ^ { 2 } } { 4 \kappa c } = \frac { \sigma ^ { 2 } } { 4 \kappa c ^ { 2 } } c _ { * } \ ,\tag{107}
$$

so

$$
\frac { \sigma ^ { 2 } } { 2 \kappa c } = \frac { 2 } { c _ { * } / c - 1 } .\tag{108}
$$

This implies that r can be written in terms of c as

$$
r ( c ) ^ { 2 } = 1 + { \frac { 2 } { c _ { * } / c - 1 } } = { \frac { c _ { * } / c + 1 } { c _ { * } / c - 1 } } = { \frac { c _ { * } + c } { c _ { * } - c } } .\tag{109}
$$

Using this result, the first condition can be rewritten as

$$
\kappa ^ { 2 } = \frac { 1 } { ( c _ { * } + c ) ^ { 2 } r ^ { 2 } } = \frac { c _ { * } - c } { ( c _ { * } + c ) ^ { 3 } } .\tag{110}
$$

Let $x : = c / c _ { * }$ denote the non-dimensionalized learned rate. In terms of $x ,$ our two conditions read

$$
\left| \begin{array} { l l l l l } { \displaystyle { \kappa ^ { 2 } = \frac { 1 } { c _ { * } ^ { 2 } } \frac { 1 - x } { ( 1 + x ) ^ { 3 } } } } & { } & { } & { } & { } & { \sigma ^ { 2 } = \frac { 4 c _ { * } \kappa x ^ { 2 } } { 1 - x } } \end{array} \right| .\tag{111}
$$

Since both $\sigma ^ { 2 }$ and κ can be written in terms of the optimal value of c (or equivalently, x), it suffices to determine this value. We can then write $\sigma ^ { 2 }$ in terms of κ, and hence determine the phase boundary.

Solving a cubic equation. Note that the first condition expresses a cubic equation (in terms of $c _ { * }$ and κ) whose solution is x:

$$
c _ { * } ^ { 2 } \kappa ^ { 2 } ( 1 + x ) ^ { 3 } = 1 - x .\tag{112}
$$

Note that it only has a valid solution if $0 \leq x < 1$

Generic cubic equations are annoying to solve, but they can be made tractable with a clever substitution. Consider the inspired choice

$$
q : = { \sqrt { \frac { 1 - x } { 1 + x } } }\tag{113}
$$

which implies that

$$
\boxed { x = \frac { 1 - q ^ { 2 } } { 1 + q ^ { 2 } } } .\tag{114}
$$

Relatedly,

$$
1 + x = { \frac { 2 } { 1 + q ^ { 2 } } }
$$

$$
1 - x = \frac { 2 q ^ { 2 } } { 1 + q ^ { 2 } } .\tag{115}
$$

In terms of this new variable q, our cubic equation becomes

$$
c _ { * } ^ { 2 } \kappa ^ { 2 } = \frac { q ^ { 2 } } { 4 } ( 1 + q ^ { 2 } ) ^ { 2 } .\tag{116}
$$

Since $q \geq 0$ by definition, we can take the square root of both sides to find that

$$
2 c _ { * } \kappa = q ( 1 + q ^ { 2 } ) ,\tag{117}
$$

or equivalently that

$$
\begin{array} { r } { \boxed { q ^ { 3 } + q - 2 \kappa c _ { * } = 0 } \ . } \end{array}\tag{118}
$$

This is a depressed cubic equation. Its discriminant equals

$$
\Delta = - 4 - 2 7 ( - 2 \kappa c _ { * } ) ^ { 2 } < 0 ,\tag{119}
$$

so it has one real root and two complex conjugate roots. Because a valid solution must be real, we are interested in the real root.

Using a standard formula for this root, we find that it equals

$$
\boxed { q ( \kappa ) = \frac { 2 } { \sqrt { 3 } } \sinh \left[ \frac { 1 } { 3 } \mathrm { a r s i n h } ( 3 \sqrt { 3 } \kappa c _ { * } ) \right] }\tag{120}
$$

where

$$
\operatorname { a r s i n h } ( x ) : = \log ( x + { \sqrt { x ^ { 2 } + 1 } } ) .\tag{121}
$$

Determining the critical noise level. Combining this result with our earlier formula for $\sigma ^ { 2 }$ in terms of x,

$$
\sigma ^ { 2 } = \frac { 4 c _ { * } \kappa x ^ { 2 } } { 1 - x } = 4 c _ { * } \kappa \frac { ( 1 - q ^ { 2 } ) ^ { 2 } } { 2 q ^ { 2 } ( 1 + q ^ { 2 } ) } .\tag{122}
$$

Since $2 c _ { * } \kappa = q ( 1 + q ^ { 2 } )$ , this becomes

$$
\boxed { \sigma _ { \mathrm { c r i t } } ^ { 2 } = \frac { ( 1 - q ( \kappa ) ^ { 2 } ) ^ { 2 } } { q ( \kappa ) } } .\tag{123}
$$

$$
\begin{array} { c } { { q = \displaystyle \frac { 2 } { \sqrt { 3 } } \sinh \left[ \frac { 1 } { 3 } \mathrm { a r s i n h } ( 3 \sqrt { 3 } \kappa c _ { * } ) \right] } } \\ { { \sigma ^ { 2 } = \displaystyle \frac { ( 1 - q ^ { 2 } ) ^ { 2 } } { q } . } } \end{array}\tag{124}
$$

The final phase boundary result. Summarizing, at the boundary, provided $0 < \kappa c _ { * } \leq 1$

## F CHARACTERIZATION OF FREQUENCY-RUNAWAY PHASE TRANSITION

In this appendix, we characterize the ‘frequency-runaway’ phase transition discussed in Sec. 7 that occurs when u and v are frozen (with $u = v = 1 )$ but c is trainable.

If u and v are frozen after alignment, the reduced objective becomes

$$
J = \frac { 1 + \sigma ^ { 2 } } { 4 c ( \tau ) } - \frac { 1 } { c _ { * } + c ( \tau ) } + \frac { 1 } { 4 c _ { * } }\tag{125}
$$

up to an additive constant. (Because it is constant, we have removed the regularization term). The negative gradient of J equals

$$
\begin{array} { l } { \displaystyle - \frac { \partial J } { \partial c } = \frac { ( 1 + \sigma ^ { 2 } ) } { 4 c ^ { 2 } } - \frac { 1 } { ( c _ { * } + c ) ^ { 2 } } } \\ { = \frac { ( 1 + \sigma ^ { 2 } ) ( c _ { * } + c ) ^ { 2 } - 4 c ^ { 2 } } { 4 c ^ { 2 } ( c _ { * } + c ) ^ { 2 } } } \\ { = \frac { [ \sqrt { 1 + \sigma ^ { 2 } } ( c _ { * } + c ) + 2 c ] [ \sqrt { 1 + \sigma ^ { 2 } } ( c _ { * } + c ) - 2 c ] } { 4 c ^ { 2 } ( c _ { * } + c ) ^ { 2 } } , } \end{array}\tag{126}
$$

so c has dynamics

$$
{ \frac { d } { d \tau } } c = { \frac { [ \sqrt { 1 + \sigma ^ { 2 } } ( c _ { * } + c ) + 2 c ] } { 4 c ^ { 2 } ( c _ { * } + c ) ^ { 2 } } } [ \sqrt { 1 + \sigma ^ { 2 } } ( c _ { * } + c ) - 2 c ] .\tag{127}
$$

The fixed point of c dynamics can only depend on the second factor in the numerator, since the first factor is always positive for $c \geq 0$ and $c _ { * } > 0$ . The second factor equals zero at

$$
\boxed { c _ { \mathrm { o p t } } = \frac { \sqrt { 1 + \sigma ^ { 2 } } c _ { * } } { 2 - \sqrt { 1 + \sigma ^ { 2 } } } } ,\tag{128}
$$

so when $c _ { \mathrm { o p t } }$ is well-defined, it is the fixed point of c dynamics.

Importantly, this quantity blows up when its denominator equals zero, which occurs at the critical noise level $\dot { \sigma } _ { \mathrm { c r i t } } ^ { 2 } : = 3 $ . Just below that noise level,

$$
\boxed { c _ { \mathrm { o p t } } \approx \frac { 8 c _ { * } } { \sigma _ { \mathrm { c r i t } } ^ { 2 } - \sigma ^ { 2 } } } ,\tag{129}
$$

so it diverges as $1 / ( \sigma _ { \mathrm { c r i t } } ^ { 2 } - \sigma ^ { 2 } )$ . For $\sigma ^ { 2 } \geq \sigma _ { \mathrm { c r i t } } ^ { 2 }$ , the second factor equals

$$
\sqrt { 1 + \sigma ^ { 2 } } ( c _ { * } + c ) - 2 c = \sqrt { 1 + \sigma ^ { 2 } } c _ { * } + ( \sqrt { 1 + \sigma ^ { 2 } } - 2 ) c ,\tag{130}
$$

which is always positive. This means that, beyond the critical noise level, dynamics drive c to infinity. Hence, beyond the threshold, J is a decreasing function of c and has no finite minimum.