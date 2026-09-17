# Deep learning emergent spacetime from fermionic spectral functions in holography

Koji Hashimoto<sup>1</sup>,<sup>∗</sup> Hyun-Sik Jeong<sup>2,3</sup>,<sup>†</sup> Keun-Young Kim<sup>4,5</sup>,<sup>‡</sup> Daichi Takeda<sup>6</sup>,<sup>§</sup> and Kwan Yun<sup>1,4¶</sup>

<sup>1</sup>Department of Physics, Kyoto University, Kyoto 606-8502, Japan

<sup>2</sup>Asia Pacific Center for Theoretical Physics, Pohang 37673, Korea

<sup>3</sup>Department of Physics, Pohang University of Science and Technology, Pohang 37673, Korea

<sup>4</sup>Department of Physics and Photon Science, Gwangju Institute of

Science and Technology, 123 Cheomdan-gwagiro, Gwangju 61005, Korea

<sup>5</sup>Research Center for Photon Science Technology, Gwangju Institute of Science and Technology,

123 Cheomdan-gwagiro, Gwangju 61005, Korea and

<sup>6</sup>iTHEMS, RIKEN, Wako, Saitama 351-0198, Japan

We present a physics-informed machine learning framework based on Neural Ordinary Diferential Equations that solves the holographic inverse problem: reconstructing the bulk spacetime and gauge field of a charged AdS black hole directly from boundary fermionic spectral functions. Encoding the UV asymptotics, horizon regularity, and zero temperature extremality as hard constraints in the neural network architecture, our framework reliably reconstructs the extremal Reissner-Nordstr¨om AdS geometry across three quantum critical regimes set by the U(1) probe charge—non-Fermi liquid, marginal Fermi liquid (strange metal), and Fermi-liquid-like states—and can jointly infer the probe charge itself to sub-percent accuracy. Relaxing the near-AdS boundary constraint uncovers a geometrical degeneracy: bulk profiles that difer throughout the radial direction but share the same near-horizon $A d S _ { 2 } \times \mathbb { R } ^ { 2 }$ data reproduce identical spectral functions near the Fermi surface. This isospectral non-uniqueness is precisely the bulk degeneracy expected on general holographic grounds at zero temperature, and its spontaneous emergence across independent training runs shows that the network isolates the IR CFT universality rather than overfitting a single UV completion.

## I. INTRODUCTION

One of the central problems in quantum condensed matter physics concerns the critical phenomena governing zero-temperature quantum phase transitions. At a quantum critical point (QCP), the divergence of the correlation length may give rise to scale invariance and emergent conformal symmetry [1]. However, exotic phases near a QCP—such as the strange metal regime in high-T superconductors and heavy-fermion materials—depart fundamentally from Landau’s Fermi liquid theory [2– 4], exhibiting anomalous thermodynamic and transport properties. Understanding these non-Fermi liquid states requires analytical tools capable of handling strong correlations at finite density beyond weak-coupling field theories.

To address this challenge, holographic condensed matter theory (AdS/CMT) models certain classes of non-Fermi liquids and their associated quantum critical states by introducing probe Dirac fermions into the background of a charged AdS black hole [5–9]. This approach builds on the anti-de Sitter/conformal field theory (AdS/CFT) duality [10–12], which maps strongly coupled quantum many-body systems onto classical gravitational systems in higher dimension [13–15]. In this setup, the charged black hole serves as the dual description of a strongly correlated ground state at finite density, and the scattering of Dirac fermion of the black hole horizon determines the boundary retarded Green’s function $G _ { R } ( \omega , k )$ and the associated spectral function—a direct theoretical analog to angle-resolved photoemission spectroscopy (ARPES) measurements.

This holographic approach has ofered novel insights into the low-energy dynamics of strongly interacting fermionic quantum critical state. Exploring diferent regions of the parameter space reveals both Fermi liquidlike [7] and non-Fermi-liquid behaviors [6, 7, 16], showing that the infrared (IR) low-energy regime of these non-Fermi liquids is governed by a nontrivial quantum fixed point. Consequently, charged AdS black holes thus provide a robust and versatile tool for probing fermionic quantum criticality at finite density.

Historically, holographic modeling has operated in the forward direction: given a bulk gravity action, one solves the classical bulk equations of motion for a specified metric and matter fields, and integrates the probe field equations to compute boundary observables—a well-posed, essentially algorithmic procedure. The inverse problem is fundamentally diferent in character. Because the bulk fields are unknown continuous radial functions and the most boundary observables depend on them implicitly, through the non-linear solution of the bulk equation, there is often no closed-form procedure for inverting boundary data into a bulk profile. Recovering a holographic bulk configuration from a set of boundary measurements is therefore a non-trivial inverse problem, which may admit multiple, physically distinct bulk solu tions consistent with the same boundary data.

Recently, physics-informed machine learning [17, 18] has emerged as a powerful tool to bridge this gap in the context of holography (AdS/CFT) [19, 20]. By identifying deep neural networks [21, 22] as continuous renormalization group flows, bulk geometries can be reconstructed from diverse boundary quantum data from strongly interacting QCD [23–27] and condensed matter systems [28– 33]. In particular, the Neural Ordinary Diferential Equation (Neural ODE) [34–37] architecture allows neural networks to parameterize continuous bulk functions while preserving the exact diferential equations of motion during optimization.

In this work, we present a Neural ODE-based inverse framework that reconstructs bulk spacetime metrics and gauge fields directly from the boundary fermionic spectral function, $\mathcal { A } ( \omega , \boldsymbol { k } ) : = 1 / \pi \operatorname { I m } G _ { R } ( \omega , \boldsymbol { k } )$ The physical motivation for this approach stems from the anomalous transport and spectral properties of strange metals, where low-energy excitations near the Fermi surface deviate sharply from standard Fermi-liquid quasiparticles [2, 38]. Such behavior points to an underlying quantum critical point at low energy scales [39–42]—a phenomenon also prominently observed in heavy-fermion systems near a quantum phase transition [43]. While these features are historically modeled using phenomenological frameworks like the marginal Fermi liquid [38], holography ofers an alternative geometric description.

To test whether bulk geometry can be reverseengineered from these boundary quantum critical features, we apply our architecture to the zero-temperature $( T = 0 )$ AdS background using non-Fermi liquid spectral data [5–8]. We demonstrate that our deep learning model successfully recovers the emergent $A d \bar { S _ { 2 } } \times \mathbb { R } ^ { 2 }$ quantum critical region from boundary fermionic input data.

In Section II, we review the holographic Dirac equations for probe fermions in planar black hole spacetimes and establish the flow equations for computing retarded Green’s functions. In Section III, we present the low-energy quantum critical behaviors of fermionic spectral functions generated from the extremal Reissner-Nordstr¨om AdS background, defining the three representative benchmark datasets. Section IV details our physics-informed Neural ODE framework, including the hard-constrained neural network ansatz and the twostage hybrid optimization strategy. In Section V, we demonstrate the successful reconstruction of the extremal Reissner-Nordstr¨om AdS spacetime and the joint determination of the probe fermion charge. In Section VI, we relax the near-boundary slope constraint, uncover an isospectral geometrical degeneracy among distinct bulk geometries sharing the same near-horizon $\bar { \boldsymbol { A } } d S _ { 2 } \times \mathbb { R } ^ { 2 }$ spacetime. We conclude with a summary in Section VII.

## II. HOLOGRAPHIC DIRAC EQUATIONS

We consider a planar four-dimensional black brane background with the general metric ansatz:

$$
\mathrm { d } s ^ { 2 } = \frac { 1 } { z ^ { 2 } } \left[ - f ( z ) \mathrm { d } t ^ { 2 } + \frac { \mathrm { d } z ^ { 2 } } { f ( z ) } + h ( z ) \mathrm { d } \vec { x } _ { i } ^ { 2 } \right] ,\tag{1}
$$

where $z$ is the radial bulk coordinate, with the AdS boundary at $z  0$ and the event horizon at $z _ { h } = 1$

We introduce a probe Dirac fermion field Ψ propagating in the background (1), with mass m and $U ( 1 )$ charge q—the latter identified with the charge of the dual boundary fermionic operator—governed by the curvedspacetime Dirac equation

$$
\left( \Gamma ^ { M } D _ { M } - m \right) \Psi = 0 ,\tag{2}
$$

where the covariant derivative ${ \cal D } _ { M } = \partial _ { M } + { \textstyle { \frac { 1 } { 4 } } } \omega _ { A B , M } \Gamma ^ { A B } -$ $i q A _ { M }$ incorporates the spin connection ω and the background $U ( 1 )$ gauge potential $A _ { M } = \left( A _ { t } ( z ) , 0 , 0 , 0 \right)$ Here, M denotes a bulk spacetime index, and A, B denote tangent-space indices. Gamma matrices can be expressed in the tangent space as $\Gamma ^ { M } \ = \ \Gamma ^ { A } e _ { A } ^ { M }$ with the inverse vielbein $e _ { A } ^ { \mathcal { M } }$ . We refer the reader to $[ 6 , 9 ]$ for a detailed review of the Dirac equations in holography.

By taking the Fourier transform along the boundary coordinates, we decompose the spinor as

$$
\Psi = \left( - g g ^ { z z } \right) ^ { - 1 / 4 } e ^ { i \left( k x - \omega t \right) } \left( { \phi } _ { + } ( z ) \right) ,\tag{3}
$$

where, without loss of generality, spatial rotational symmetry in $\operatorname { E q . } ( 1 )$ allows us to set $k _ { i } = ( k , 0 )$ . Notably, this rescaling ansatz eliminates the spin connection terms.

To analyze the bulk Dirac equations, it is convenient to use the Gamma matrices $\Gamma ^ { A }$ as

$$
\Gamma ^ { z } = \left( \begin{array} { c c } { { - \mathbb { 1 } _ { 2 } } } & { { 0 } } \\ { { 0 } } & { { \mathbb { 1 } _ { 2 } } } \end{array} \right) , \quad \quad \Gamma ^ { \mu } = \left( \begin{array} { c c } { { 0 } } & { { \gamma ^ { \mu } } } \\ { { \gamma ^ { \mu } } } & { { 0 } } \end{array} \right) ,\tag{4}
$$

where it can be further expressed with Pauli matrices $\sigma _ { i }$ as $\gamma ^ { t } = i \sigma _ { y } , \gamma ^ { x } = \sigma _ { x }$ , and $\gamma ^ { y } = \sigma _ { z }$

Substituting the spinor decomposition (3) into the Dirac equation (2) yields a coupled system of first-order radial equations for $y _ { \pm }$ and $z _ { \mp }$ ，

$$
\begin{array} { l } { { \sqrt { f ( z ) h ( z ) } \left( \partial _ { z } \pm \displaystyle \frac { m } { z \sqrt { f ( z ) } } \right) y _ { \pm } = \pm i \left( k - u \right) z _ { \mp } , } } \\ { { \sqrt { f ( z ) h ( z ) } \left( \partial _ { z } \mp \displaystyle \frac { m } { z \sqrt { f ( z ) } } \right) z _ { \mp } = \mp i \left( k + u \right) y _ { \pm } , } } \end{array}\tag{5}
$$

where

$$
\phi _ { \pm } : = \binom { y _ { \pm } } { z _ { \pm } } \ , \quad u : = \sqrt { \frac { h ( z ) } { f ( z ) } } \left( \omega + q A _ { t } ( z ) \right) .\tag{6}
$$

A convenient change of variables, defined by the ratios

$$
\xi _ { + } = i { \frac { y _ { - } } { z _ { + } } } , \qquad \xi _ { - } = - i { \frac { z _ { - } } { y _ { + } } } ,\tag{7}
$$

decouples the system (5) into two independent, nonlinear (Riccati-type) flow equations,

$$
\begin{array} { l } { \displaystyle \sqrt { f ( z ) } \partial _ { z } \xi _ { \pm } - \frac { 2 m } { z } \xi _ { \pm } \mp \left[ \frac { k } { \sqrt { h ( z ) } } \mp \frac { \omega + q A _ { t } ( z ) } { \sqrt { f ( z ) } } \right] } \\ { \displaystyle \pm \left[ \frac { k } { \sqrt { h ( z ) } } \pm \frac { \omega + q A _ { t } ( z ) } { \sqrt { f ( z ) } } \right] \xi _ { \pm } ^ { 2 } = 0 . } \end{array}\tag{8}
$$

These flow equations can be integrated from the horizon to the AdS boundary to compute the retarded Green’s function $G _ { R } ( \omega , k )$ via holographic renormalization [5–9]

$$
G _ { R } = \operatorname * { l i m } _ { z  0 } z ^ { - 2 m } ( \begin{array} { c c } { { \xi _ { + } ( z ) } } & { { 0 } } \\ { { 0 } } & { { \xi _ { - } ( z ) } } \end{array} ) : = ( \begin{array} { c c } { { G _ { + } ( \omega , k ) } } & { { 0 } } \\ { { 0 } } & { { G _ { - } ( \omega , k ) } } \end{array} ) .\tag{9}
$$

Notably, the two diagonal components are related by reversing the momentum sign [6, 9]

$$
G _ { - } ( \omega , \boldsymbol { k } ) = G _ { + } ( \omega , - \boldsymbol { k } ) ,\tag{10}
$$

so that $G _ { - } ( \omega , \boldsymbol { k } )$ alone already encodes the information carried by both components. The fermionic spectral function is given by the imaginary part of the retarded Green’s function,

$$
\mathcal { A } ( \omega , \boldsymbol { k } ) : = \frac { 1 } { \pi } \mathrm { I m } G _ { - } ( \omega , \boldsymbol { k } ) .\tag{11}
$$

To solve the flow equation (8), an appropriate boundary condition must be imposed at the horizon. For $\omega \neq 0 .$ demanding infalling boundary conditions yields

$$
\xi _ { \pm } { \left( z = 1 \right) } = i .\tag{12}
$$

At zero temperature $( T = 0 )$ , however, the point $\omega = 0$ requires special care: the extremal horizon produces a double zero in $f ( z )$ , causing the standard infalling condition to break down. For the extremal Reissner-Nordstr¨om AdS background, regularity at the horizon instead selects the zero-frequency condition as

$$
\xi _ { \pm } | _ { z = 1 , \omega = 0 } = \frac { m - \sqrt { k ^ { 2 } + m ^ { 2 } - \frac { ( q \mu ) ^ { 2 } } { 6 } - i \epsilon } } { \displaystyle \frac { q \mu } { \sqrt { 6 } } \pm k } , \quad \epsilon  0 ^ { + } .\tag{13}
$$

Furthermore, near the double zero of $f ( z )$ at the extremal horizon, $f \approx f _ { 0 } ( 1 - z ) ^ { 2 }$ , the Dirac field exhibits irregular singular behavior as $\phi _ { \pm } \mathopen { } \mathclose \bgroup \left( z \aftergroup \egroup \right) \approx e ^ { \frac { i \omega } { f _ { 0 } \left( 1 - z \right) } }$ . This singularity contrasts sharply with the finite-temperature case $( T \neq 0 )$ , where the near-horizon behavior is characterized by a power-law branch point, $\left( 1 - z \right) ^ { \frac { - i \omega } { 4 \pi T } }$

## III. QUANTUM CRITICAL BEHAVIOR OF SPECTRAL FUNCTIONS

To prepare the input fermionic spectral data for our neural network framework, we employ the low-energy

boundary spectral functions generated from the extremal Reissner-Nordstr¨om AdS background, whose exact profiles correspond to

$$
\begin{array} { c } { { f ( z ) = 1 - \left( 1 + \mu ^ { 2 } \right) z ^ { 3 } + \mu ^ { 2 } z ^ { 4 } , \quad h ( z ) = 1 , } } \\ { { { \cal A } _ { t } ( z ) = \mu ( 1 - z ) , } } \end{array}\tag{14}
$$

where $\mu$ is the chemical potential, and Hawking temperature is given by $T = ( \bar { 3 } - \mu ^ { 2 } ) / ( 4 \pi )$ , thus the extremal limit $( T = 0 )$ can be achieved by $\mu = { \sqrt { 3 } }$ . In this limit the extremal geometry of (14) becomes $A d S _ { 2 } \times \mathbb { R } ^ { 2 }$ spacetime.

In this manuscript, we adopt the theoretical setup of Faulkner et al. [8], in which a massless probe fermion with varying $U ( \bar { 1 } )$ charge q models non-Fermi liquid and strange metal transport. A brief review of this setup is as follows.

Near the Fermi surface where $\omega  0$ and $k  k _ { F }$ , the retarded Green’s function $G _ { R } ( \omega , k )$ exhibits a characteristic low-energy quantum critical form governed by the emergent $A d \bar { S _ { 2 } } \times \bar { \mathbb { R } } ^ { 2 }$ near-horizon geometry:

$$
G _ { R } ( \omega , k ) = \frac { h _ { 1 } } { k - k _ { F } - \frac { \omega } { v _ { F } } - \Sigma ( \omega , k ) } ,\tag{15}
$$

where $k _ { F }$ is the Fermi momentum, v<sub>F</sub> the Fermi velocity, $h _ { 1 }$ a numerical constant, and $\Sigma ( \omega , \boldsymbol { k } )$ represents the selfenergy for excitations near the Fermi surface, originating from the coupling to the IR CFT, i.e., $\Sigma ( \omega , \boldsymbol { k } ) \approx \mathcal { G } _ { \boldsymbol { k } } ( \omega )$ where $\mathcal { G } _ { k } ( \omega )$ is the IR retarded Green’s function, which can be computed analytically [8, 16].

The low-frequency scaling behavior of the self-energy near the Fermi momentum is controlled by the IR conformal dimension (or scaling exponent $\nu _ { k } )$ , given by

$$
\Sigma ( \omega , k ) \approx c ( k _ { F } ) \omega ^ { 2 \nu _ { k _ { F } } } , \quad \nu _ { k } : = \sqrt { m ^ { 2 } L _ { 2 } ^ { 2 } + k ^ { 2 } \frac { L _ { 2 } ^ { 2 } } { L _ { x } ^ { 2 } } - q ^ { 2 } e _ { d } ^ { 2 } } ,\tag{16}
$$

where $e _ { d } ^ { 2 } : = L _ { 2 } ^ { 4 } A _ { t } ^ { \prime } ( 1 ) ^ { 2 }$ . Here, $c ( k )$ is a complex analytic function of $k ,$ while $L _ { 2 }$ and $L _ { x }$ denote the $\mathrm { { A d S } _ { 2 } }$ and $\mathbb { R } ^ { 2 }$ radii, respectively. $\nu _ { k _ { F } }$ serves as the dynamical critical exponent determining the lifetime and dispersion of lowenergy excitations near $k _ { F }$

Depending on the magnitude of $\nu _ { k _ { F } } ,$ the spectral functions of massless fermions exhibit three representative quantum critical behaviors [8], which serve as distinct benchmark datasets for our machine learning training:

• Data A $\left( \nu _ { k _ { F } } < 1 / 2 \right)$ : Non-Fermi Liquid For $q \ = \ 1$ , the Fermi momentum is $k _ { F } ~ = ~ 0 . 5 3 \mu .$ yielding $\nu _ { k _ { F } } = 0 . 2 4 \ : \left( < 1 / 2 \right)$ . In this regime, the selfenergy $\Sigma ( \bar { \omega } , k ) \approx \omega ^ { 2 \nu _ { k _ { F } } }$ dominates over the linear analytic term $\omega / v _ { F }$ in the denominator of Eq. (15). While a dispersion relation is still present, it becomes distinctly non-linear, $\omega _ { * } ( k ) \approx \left( k - k _ { F } \right) ^ { \frac { 1 } { 2 \nu _ { k _ { F } } } }$ (with $1 / 2 \nu _ { k _ { F } } ~ \approx ~ 2 . 0 8 )$ Because the decay width $\left( \Gamma \propto \mathrm { I m } \Sigma \right)$ scales with the same power as the real part and remains comparable to the excitation energy, the spectral feature forms a broad, incoherent peak rather than a sharp excitation. The quasiparticle residue vanishes at the Fermi surface, representing a non-Fermi liquid devoid of long-lived quasiparticle excitations.

![](images/90ff5df8efba2288df614233ee3562e3a907987456d3156148a813ca37ef9d77.jpg)

![](images/a3a64d108f4ecfab82b05222ab7324dd0b251f036466594abcf9cce78fbddab2.jpg)

![](images/c7dd9299afa85d4a7a5a7ed4763777e50ad3333fd280f470bbee59bbf0450b95.jpg)  
Figure 1. The fermionic spectral function $\begin{array} { r } { A ( \omega , \boldsymbol { k } ) = \frac { 1 } { \pi } \mathrm { I m } G _ { - } ( \omega , \boldsymbol { k } ) } \end{array}$ of the extremal Reissner-Nordstr¨om AdS black hole for three representative probe fermion charges: Data $\mathrm { ~ A ~ } \left( q \right) = 1$ , left), Data B $( q = 1 . 5 6$ , middle), and Data $\mathrm { ~ C ~ } ( q = 2 , \mathrm { { \ r i g h t } } )$ . (a) Data A $\left( \nu _ { k _ { F } } = 0 . 2 4 < 1 / 2 \right)$ displays a broad, incoherent peak with a non-linear dispersion ω∗ $\dot { ( k ) } \approx ( k - k _ { F } ) ^ { 2 . 0 \dot { 8 } }$ and a vanishing quasiparticle residue, representing a non-Fermi liquid without long-lived excitations. (b) Data B $\left( \nu _ { k _ { F } } ~ = ~ 1 / 2 \right)$ realizes the marginal Fermi liquid regime with a quasi-linear dispersion $\begin{array} { r } { \omega _ { * } ( k ) \approx \frac { k - k _ { F } } { \ln | k - k _ { F } | } } \end{array}$ and a linear scattering rate $\Gamma \propto \omega$ , forming a moderately broad spectral ridge characteristic of strange metal phenomenology. (c) Data $\mathrm { ~ C ~ } ( \nu _ { k _ { F } } = 0 . 7 3 > 1 / 2 )$ exhibits an asymptotically linear dispersion $\omega _ { * } ( k ) \approx v _ { F } ( k - k _ { F } )$ with a vanishing relative width $\Gamma / \omega _ { \ast }  0$ near $k _ { F }$ , producing a sharp, long-lived quasiparticle peak with a non-zero residue.

• Data B $\left( \nu _ { k _ { F } } = 1 / 2 \right)$ : Marginal Fermi Liquid For $q ~ = ~ 1 . 5 6$ , the Fermi surface lies at $\begin{array} { r l } { k _ { F } } & { { } = } \end{array}$ $0 . 9 5 2 \mu$ , corresponding precisely to the critical value $\nu _ { k _ { F } } = 1 / 2$ . In this case, the linear frequency term $\omega / v _ { F }$ and the IR self-energy share the same lowfrequency scaling, giving rise to a characteristic logarithmic self-energy:

$$
\mathrm { R e } \Sigma ( \omega ) \approx \omega \log \omega , \quad \mathrm { I m } \Sigma ( \omega ) \approx \omega .\tag{17}
$$

The real part produces a quasi-linear dispersion $\begin{array} { r } { \omega _ { * } ( k ) ~ \approx ~ \frac { - k - k _ { F } } { \ln | k - k _ { F } | } } \end{array}$ , ensuring a clear spectral ridge trajectory. However, this leads to a single-particle scattering rate Γ that depends linearly on frequency, causing the quasiparticle residue to vanish logarithmically as $k  k _ { F }$ . This directly realizes the marginal Fermi liquid state [38] characteristic of strange metal phenomenology and ARPES observations on cuprates [44]. Consequently, the spectral peak exhibits an intermediate broadness: it is neither a sharp, long-lived quasiparticle nor a featureless continuum, but a well-defined yet broad spectral ridge.

• Data C $( \nu _ { k _ { F } } > 1 / 2 )$ : Fermi-Liquid-Like For $q = 2$ , one obtains $k _ { F } = 1 . 3 1 5 \mu$ with a conformal dimension $\nu _ { k _ { F } } = 0 . 7 3 ~ \left( > 1 / 2 \right)$ . In this case, the analytic linear term $\omega / v _ { F }$ dominates the real part of the denominator of (15), restoring an asymptotically linear dispersion $\omega _ { * } ( k ) \approx v _ { F } ( k - k _ { F } )$ as in a Fermi liquid. Nevertheless, the scattering rate scales diferently from that of a Fermi liquid. The relative spectral width scales as $\Gamma ( k ) / | \omega _ { * } ( \bar { k } ) |$ ∣ ≈ $| k - k _ { F } | ^ { 2 \nu _ { k _ { F } } - 1 } \to 0$ as $k  k _ { F }$ . This suppression of the scattering rate sharpens the peak dramatically, giving rise to long-lived, sharp quasiparticle-like excitations with a non-zero residue at the Fermi surface.

We display the fermionic spectral functions (11) of the extremal Reissner-Nordstr¨om black hole for $q = 1 , 1 . 5 6$ and 2 in Fig. 1, which reproduce the results in Ref. [8] and clearly demonstrate the physical descriptions above.

In our machine learning framework, these three distinct spectral regimes—ranging from an incoherent non-Fermi liquid peak to a sharp Fermi-liquid-like quasiparticle excitation—serve as three separate boundary input datasets on which the inverse model is trained independently.

## IV. NEURAL ODE FRAMEWORK

To address the holographic inverse problem, we construct a Neural Ordinary Diferential Equation (Neural ODE) framework [34–37] that directly reconstructs the unknown continuous bulk profile $\{ f ( z ) , h ( z ) , A _ { t } ( z ) \}$ from boundary fermionic spectral data. While the forward holographic calculation determines boundary observables from a specified bulk background, our framework optimizes neural network representations of the bulk metric and gauge fields by embedding the exact radial Dirac flow equation $\operatorname { E q . } ( 8 )$ directly into the network’s learning process.

Hard-Constrained Neural Network Ansatz. Instead of learning unconstrained bulk functions, we incorporate the required physical boundary conditions (AdS boundary at $z  0 )$ , zero temperature $\left( f ^ { \prime } ( 1 ) = 0 \right)$ , and black brane conditions $( f ( 1 ) = 0 )$ directly into the network architecture through hard constraints:

$$
\begin{array} { r l } & { f _ { \theta } \big ( z \big ) = \big ( 1 - z \big ) ^ { 2 } \left[ 1 + \big ( a + 2 \big ) z + z ^ { 2 } D _ { f } \big ( z ; \theta _ { f } \big ) \right] , } \\ & { h _ { \theta } \big ( z \big ) = 1 + a z + z ^ { 2 } D _ { h } \big ( z ; \theta _ { h } \big ) , } \\ & { A _ { t , \theta } \big ( z \big ) = \mu \big ( 1 - z \big ) + z \big ( 1 - z \big ) D _ { A } \big ( z ; \theta _ { A } \big ) , } \end{array}\tag{18}
$$

where $D _ { f } ( z ; \theta _ { f } ) , D _ { h } ( z ; \theta _ { h } )$ , and $D _ { A } { \big ( } z ; \theta _ { A } { \big ) }$ are three independent scalar Multi-Layer Perceptrons (MLPs) parameterized by trainable weights $\dot { \theta } \stackrel { - } { = } \{ \theta _ { f } , \theta _ { h } , \theta _ { A } \}$ . Each MLP consists of 3 hidden layers with 20 neurons per layer and utilizes the Softplus activation function.

By construction, Eq. (18) guarantees the UV AdS boundary conditions $f _ { \theta } ( 0 ) = h _ { \theta } ( 0 ) = 1$ , as well as the IR horizon regularity $f _ { \theta } ( 1 ) = 0$ and $A _ { t , \theta } ( 1 ) = 0$ . The double zero $( 1 - z ) ^ { 2 }$ in $f _ { \theta } ( z )$ strictly enforces zero Hawking temperature $( T = 0 )$ Furthermore, for computational convenience, we also enforce the near-boundary derivative conditions $f _ { \theta } ^ { \prime } ( 0 ) = h _ { \theta } ^ { \prime } ( 0 ) = a .$ , where a is a common asymptotic slope parameter, which vanishes for Reissner-Nordstr¨om AdS background (14).

Numerical Integration and Optimization Strategy. For each boundary data point $( \omega _ { i } , k _ { i } )$ sampled on a regular grid from the spectral functions $\operatorname { E q . } ( 1 1 )$ (or Fig. 1), the dynamical spinor ratio $\xi _ { - } ( z )$ is integrated along the radial direction using the Dirac flow equation (8). The training data are discretized on a uniform $1 2 \times 5$ grid in $( \omega / \mu \times k / \mu )$ , giving $ { N _ { \mathrm { d a t a } } } ~ = ~ 6 0$ points per dataset. We choose a small frequency window, $\omega / \mu \in \ [ - 0 . 4 5 , 0 . 4 5 ]$ , for all three datasets so as to cover the IR Green’s function, while the momentum range is centered on the respective Fermi momentum: $k / \mu \in [ 0 . 1 3 , 0 . 9 3 ] , [ 0 . 5 5 , 1 . 3 5 ]$ , and [0.93, 1.73] for Data A, B, and C, respectively.

The integration proceeds from the near-horizon IR cutof $z _ { h } = 1 - \epsilon _ { h }$ to the near-boundary UV cutof $z _ { b } = \epsilon _ { b }$ with $\epsilon _ { h } = 5 \times 1 0 ^ { - 3 }$ and $\epsilon _ { b } = 1 0 ^ { - 4 }$ . We employ the Tsitouras fifth-order Runge-Kutta method (Tsit5) with an ODE error tolerance of $1 0 ^ { - 6 }$ . The deep learning-induced spectral function is obtained directly from the boundary value:

$$
\mathcal { A } _ { \theta } ( \omega _ { i } , k _ { i } ) = \frac { 1 } { \pi } \mathrm { I m } \xi _ { - } ( z _ { b } ) .\tag{19}
$$

The network parameters θ are trained by minimizing the unweighted mean-squared error (MSE) loss between the $\boldsymbol { \mathcal { A } } _ { \boldsymbol { \theta } } ( \omega _ { i } , \boldsymbol { k } _ { i } )$ and the boundary training data $\mathcal { A } _ { i } ^ { \mathrm { d a t a } }$

$$
\mathcal { L } _ { \mathrm { d a t a } } ( \theta ) = \frac { 1 } { N _ { \mathrm { d a t a } } } \sum _ { i = 1 } ^ { N _ { \mathrm { d a t a } } } \left[ A _ { \theta } ( \omega _ { i } , k _ { i } ) - A _ { i } ^ { \mathrm { d a t a } } \right] ^ { 2 } .\tag{20}
$$

Because all physical endpoint and horizon conditions are built identically into the ansatz (18), no additional penalty terms or regularization loss components are required.

To achieve both fast global exploration and highprecision convergence, we implement a two-stage hybrid optimization strategy: (I) Adam [45]: We first train using the Adam optimizer with a learning rate $\eta _ { \mathrm { A d a m } } = 1 0 ^ { - 3 }$ until the loss drops to $\mathcal { O } ( 1 0 ^ { - 1 } ) ~ \mathrm { o r } ~ \mathcal { O } ( 1 0 ^ { 0 } )$ (II) L-BFGS [46]: Optimization is then transitioned to the L-BFGS algorithm $( \eta _ { \mathrm { L - B F G S } } = 0 . 1$ , max iter = 5, tolerance $\mathtt { g r a d } = 1 0 ^ { - 5 }$ , tolerance change $= 1 0 ^ { - 5 } )$ with the strong-Wolfe line search.

Training is terminated when the standard deviation of the loss over the most recent 10 epochs falls below Std $( { \mathcal { L } } _ { n - 9 } , \ldots , { \mathcal { L } } _ { n } ) < 1 0 ^ { - 9 }$

## V. DEEP LEARNING EMERGENT SPACETIME FROM FERMIONIC SPECTRAL FUNCTIONS

Reconstruction of the extremal RN AdS spacetime. We now present the bulk spacetime reconstruction using the quantum critical fermionic spectral data introduced in Fig. 1. We begin with the baseline configuration: fixing the common asymptotic derivative parameter to $a = 0$ in the neural network ansatz (18) while setting the probe fermion charge q to its exact values $( q = 1 , 1 . 5 6 ,$ and 2 for Data A, B, and C, respectively).

The trained bulk metric components $\{ f ( z ) , h ( z ) \}$ and gauge potential profile $A _ { t } ( z )$ are displayed in Fig. 2. Across all three probe charge regimes, the trained neural profiles match the target analytical extremal Reissner-Nordstr¨om (RN) AdS black hole geometry (14) with chemical potential $\mu = \sqrt { 3 }$ The optimization achieves final L-BFGS loss values of $4 . 6 5 \times 1 0 ^ { - 5 } , 1 . 7 8 \times 1 0 ^ { - 5 }$ , and $8 . 7 2 \times 1 0 ^ { - 3 }$ for Data A, Data B, and Data C, respectively; the comparatively larger residual for Data C reflects the added numerical dificulty of resolving its sharp, longlived quasiparticle peak on the same $1 2 \times 5$ discretization grid used for all three datasets. This high reconstruction fidelity is further demonstrated by the close agreement between the trained spectral functions $\boldsymbol { \mathcal { A } } _ { \boldsymbol { \theta } } \left( \omega _ { i } , \boldsymbol { k } _ { i } \right)$ and the target input data $\mathcal { A } _ { i } ^ { \mathrm { d a t a } }$ , as shown in the lower panels of Fig. 2.

Reconstruction of bulk fields and fermion charge. Next, we investigate whether a $\iota U ( 1 )$ fermion charge q can also be identified simultaneously alongside the bulk fields from the given spectral functions. We demonstrate this joint optimization using Data B, corresponding to the marginal Fermi liquid (strange metal) case. Retaining the fixed derivative condition $a = 0 ,$ , we promote the probe fermion charge q to an independent trainable parameter optimized in addition to the network weights θ of bulk fields.

As shown in Fig. 3, the learned bulk fields again converge to the exact extremal RN AdS profile. Concurrently, the probe charge converges to

$$
q _ { \mathrm { t r a i n e d } } = 1 . 5 6 1 0 ,\tag{21}
$$

which exhibits agreement with the reference true value $q = 1 . 5 6$ used to generate Data B, yielding a relative error of only 0.06%. The optimization achieves a final L-BFGS loss of $7 . 6 6 \times 1 0 ^ { - 6 }$ . This result demonstrates that boundary fermionic spectral data encode suficient physical information to reconstruct not only the continuous bulk spacetime and gauge potential but also the intrinsic charge of the probe fermionic operator.

![](images/aa7c64634a47ae54230704bd3074b1dcc76a422c6f48d90a40dba21ef093bb76.jpg)

![](images/4edfe943e68c500e40fc71cc6b02e67428f74197fec621b3a4e13df83e7ea58b.jpg)

![](images/d7645321b546b3329ce145c7d0c06601a382811fdeef23ce86b636bd3a4f349f.jpg)

![](images/7ab4ee6d22ff0c6d89be6a64d659af2cd7a66080a1627e49c17e3b113d5cbb54.jpg)

![](images/02e6b47ef233786cd4560cffb90013e6fd49ba1301d2092e6d26dee3a84720f6.jpg)

![](images/2a5fa3bd91dc0b8457f72f3af0beedb4d5aa3576ffa65dd4278028c667b4d993.jpg)

![](images/14f2cf0bf192835b1002cb110708c83420e63097933f5870532549bb6f35d7be.jpg)

![](images/b2a186cfd6f6dca707514bb8aba151ef19a0b62c4e3ffd3eaa268857d0806dd1.jpg)

![](images/5307bcb0401da2943c002cfed8e6d043e7c0bdba3d154c0c0ba98f9a2bb2b17d.jpg)

Figure 2. Reconstructed bulk profiles $\{ f ( z ) , h ( z ) , A _ { t } ( z ) \}$ obtained from the quantum critical fermionic spectral data shown in Fig. 1: Data A $( q = 1$ , left), Data B $( q = 1 . 5 6 ,$ , middle), and Data $\mathrm { ~ C ~ } ( q = 2 , \mathrm { r i g h t } )$ . Upper panels: The trained neural network profiles (colored curves) closely match the exact extremal Reissner-Nordstr¨om AdS black hole solutions (14) with $\mu = { \sqrt { 3 } }$ (dashed lines). Lower panels: The corresponding reconstructed fermionic spectral functions $\boldsymbol { \mathcal { A } } _ { \boldsymbol { \theta } } \left( \omega _ { i } , \boldsymbol { k } _ { i } \right)$ (solid lines) compared against the input spectral function $\mathcal { A } _ { i } ^ { \mathrm { d a t a } }$ (dashed lines). Across all three datasets, the reconstructed bulk profiles accurately reproduce even the out-of-sample test data, represented by the orange and blue curves.  
![](images/395cdfd198e0b4b855985ceaf2f1ceabd6f605b8e4370d27694c4d81c7ad3e73.jpg)

![](images/435e99f1865fa96cd77dcf1f5f815494e0157565ef9518eb0d6b512729222e3f.jpg)

![](images/4d57d7bc078a5807e7fad674ce4b202351e70fb7c1e5bfc3a6e67414385e7365.jpg)

![](images/fc6e3ca1665d5f91faf174977aea14fa1e2ca2a26f2debc0b27bb5417b3b4e9d.jpg)  
Figure 3. Simultaneous reconstruction of bulk profiles and probe fermion charge q trained on Data B (marginal Fermi liquid) Left: Reconstructed bulk metric and gauge fields compared against the target extremal Reissner-Nordstr¨om AdS background. Middle: Optimization trajectory of the trainable charge q, converging to $q _ { \mathrm { t r a i n e d } } = 1 . 5 6 1 0$ (target $q = 1 . 5 6 )$ . Right: Evolution of the training loss during the hybrid Adam and L-BFGS optimization, reaching a final loss of $\mathcal { L } = 7 . 6 6 \times 1 \mathrm { \dot { 0 } ^ { - 6 } }$

## VI. ISOSPECTRAL BULK SPACETIME AND IR UNIVERSALITY

Relaxing the near-boundary slope and degenerate spacetime. We investigate the role of the nearboundary derivative constraint $f ^ { \prime } ( 0 ) ~ = ~ h ^ { \prime } ( 0 ) ~ = ~ a$ imposed in the preceding reconstructions. To examine whether the boundary spectral data uniquely specifies the full bulk metric profile, we train the model on Data B $( q = 1 . 5 6 )$ with the bulk fermion charge q fixed, but promote the common asymptotic slope a in Eq. (18) to an independent trainable scalar parameter.

Upon optimization, the slope parameter converges to

$$
a _ { \mathrm { t r a i n e d } } \approx 0 . 1 6 1 3 ,\tag{22}
$$

with the corresponding bulk profiles displayed in Fig. 4.

Dh(z)  
![](images/bd311ccca4e172c855bdf0ed58cf6f302cdb9ec00571170755fd6caa5c9a986d.jpg)

![](images/dde9387ec685b71ec21c22351c78c9081185e14155287b72aa82fcb3e41ba096.jpg)

![](images/44235c9d1122454414d9d8572878ca7a7289ca0cc2ddbb2e2c52e36810473e2c.jpg)

![](images/c853036b2af6d7a2ff4e2406bf8ae594ff39eeebfba361845cc0a38a2a681b87.jpg)

Figure 4. The trained results using Data B $( q = 1 . 5 6 )$ when relaxing the near-boundary derivative condition $a = f ^ { \prime } ( 0 ) = h ^ { \prime } ( 0 )$ Left: The trained bulk metric functions f(z), h(z) and gauge potential $A _ { t } ( z )$ (green solid lines) compared with the exact extremal Reissner-Nordstr¨om AdS background (black dashed lines). Middle: Evolution of the trainable slope parameter a during training, converging to $a _ { \mathrm { t r a i n e d } } \approx 0 . 1 6 1 3 .$ . Right: Convergence of the training loss ${ \mathcal { L } } _ { \mathrm { d a t a } }$ as a function of epoch, reaching a final loss ≈ $4 . 9 8 \times 1 0 ^ { - 5 }$  
![](images/bed18b199fe0788cb34a1cd468357378aaad37f024a0ed4efbd7e032d0a0decc.jpg)

![](images/2ff9b84c46dfd276780260b6aaecc909987f257e9c09572b1388ddb3e37ce955.jpg)

![](images/2773fe14d7016cff9dbbb3e2997f484b096d48d8a3567a2f1c8aed91f1274c6e.jpg)

![](images/b354f3a8edcedab554fb6043f901ae2ef8e7288eee2667384b6e0d7e414cca1e.jpg)

![](images/caf11f0502a4ae4543a985e401a1f50e36282fc59dce9235a3a0b9a4a74c0ce6.jpg)

DA(z)  
![](images/f066d4ebd86cf95353d06d79a3b5a202d4c0814943153c038c5e488a3cdd4fc9.jpg)  
Figure 5. The trained bulk profiles (top) and the functions $D _ { f , h , A } ( z )$ (bottom) obtained with the Neural ODE at $a = 0 .$ . The solid lines (green, blue, and purple) represent three diferent trained solutions, while the black dashed lines denote the exact extremal Reissner-Nordstr¨om AdS solution. The three solutions difer substantially from the Reissner-Nordstr¨om AdS solution, closely reproducing the same spectral function data, with residual losses as $\mathcal { L } _ { \mathrm { d a t a } } = 1 . 6 5 \times 1 0 ^ { - 5 }$ $5 . 9 2 \times { { 1 0 } ^ { - 4 } }$ , and $5 . 0 5 \times 1 0 ^ { - 4 }$ respectively.

Despite departing visibly from the exact extremal RN AdS solution (which corresponds to $a = 0 )$ , these alternative profiles reproduce the input Data B spectrum with a remarkably low final loss of $\mathcal { L } _ { \mathrm { d a t a } } = 4 . 9 8 \times 1 0 ^ { - 5 }$ . This finding demonstrates an intriguing geometrical “degen-$\mathrm { e r a c y } ^ { , \bar { 1 } }$ : distinct bulk geometries across the intermediate radial regime can generate the same boundary fermionic spectral functions. This non-uniqueness is reflected directly in the optimization itself: independent training runs converge to diferent values of a, each giving rise to a diferent, yet equally consistent, degenerate bulk profile.

Revisiting the case of $a = 0 .$ . Notably, the constraint $a = 0$ alone does not always guarantee RN AdS spacetime: whereas the baseline reconstructions of Sec. V (Fig. 2) reliably converge to the exact RN AdS profile from generic random initializations of the network weights, we find that specific, atypical initial conditions—such as oscilla tory initial bulk profiles—can occasionally generate the training toward alternative solutions that are genuinely distinct from the RN AdS geometry, yet reproduce the same boundary spectral data (Fig. 5). Complementary results obtained with an interpretable machine-learning model, in place of the Neural ODE, are provided in $\mathrm { A p - }$ pendix A.

Nonetheless, among the family of solutions consistent with $a = 0 .$ , generic training predominantly converges to the exact RN AdS geometry rather than to one of the alternative degenerate profiles. This suggests that the RN metric constitutes a smoother, more numerically favorable attractor in the Neural ODE optimization landscape.

This degeneracy can be understood from the structure of the low-energy fermionic excitations. Although the full bulk metrics difer from the AdS boundary to the horizon, their near-horizon geometries flow to the same $A d S _ { 2 } \times \mathbb { R } ^ { 2 }$ quantum critical region. Introducing the nearhorizon radial coordinate $\zeta : = 1 - z$ and expanding the metric ansatz (18) as $\zeta \to 0$ brings Eq. (1) to the canonical near-horizon form:

$$
\mathrm { d } s ^ { 2 } = - \frac { \zeta ^ { 2 } } { L _ { 2 } ^ { 2 } } \mathrm { d } t ^ { 2 } + \frac { L _ { 2 } ^ { 2 } } { \zeta ^ { 2 } } \mathrm { d } \zeta ^ { 2 } + L _ { x } ^ { 2 } \mathrm { d } \vec { x } _ { i } ^ { 2 } ,\tag{23}
$$

with the efective $A d S _ { 2 }$ radius $L _ { 2 }$ and the $\mathbb { R } ^ { 2 }$ scale $L _ { x }$ fixed by both the slope a and the horizon values of the trained networks as

$$
L _ { 2 } ^ { 2 } = \frac { 1 } { 3 + a + D _ { f } \big ( 1 ; \theta _ { f } \big ) } , \quad L _ { x } ^ { 2 } = 1 + a + D _ { h } \big ( 1 ; \theta _ { h } \big ) .\tag{24}
$$

For a massless probe fermion in the zero temperature limit, the low-energy scaling exponent $\nu _ { k }$ and the IR conformal Green’s function $\mathcal { G } _ { k } ( \omega )$ that govern the near-Fermi-surface behavior of $G _ { R } ( \omega , k )$ depend on the bulk metric only through the two near-horizon combinations $L _ { 2 } ^ { 2 } / L _ { x } ^ { 2 }$ and $e _ { d } ^ { 2 } \colon$ Eq. (16) [8, 16]. Consequently, any two bulk profiles sharing the same AdS boundary asymptotics and the same near-horizon values of $L _ { 2 } ^ { 2 } / \bar { L } _ { x } ^ { 2 }$ and $\overline { { e _ { d } ^ { 2 } } }$ are guaranteed to produce identical low-energy spectral functions near the Fermi surface, regardless of how they difer at intermediate bulk radius profile.

We verify this mechanism directly on our trained geometries: the solution of Fig. 4 has $L _ { x } ^ { 2 } / L _ { 2 } ^ { 2 } = 5$ .842 and $1 / e _ { d } ^ { 2 } \ = \ 1 1 . 7 5 ,$ while the three degenerate solutions of Fig. 5 have $L _ { x } ^ { 2 } / L _ { 2 } ^ { 2 } = 5 . 9 7 8 , 6 . 1 6 2 , 5 . 8 5 7$ and $1 / e _ { d } ^ { 2 } = 1 2 . 1 4$ 11.79, 12.32, respectively — all consistent, within numerical error, with the exact extremal RN AdS values $L _ { x } ^ { 2 } / L _ { 2 } ^ { 2 } = 6$ and $1 / e _ { d } ^ { 2 } = 1 2$ . This confirms that it is the near-horizon data alone, and not the full radial profile, that controls the boundary spectral function near the Fermi surface: the boundary data fixes the local IR quantum critical universality class but leaves the UVcompleted bulk spacetime that flows to it undetermined.

We emphasize an important caveat regarding this interpretation. The metric ansatz Eq. (18) is not required to satisfy the Einstein-Maxwell equations of motion for any specific bulk gravitational action; its functional form is fixed only by the AdS boundary conditions, extremal horizon regularity, and the fit to the boundary fermionic spectral data through the probe Dirac flow equation Eq. (8). The isospectral degeneracy we observe is therefore a degeneracy of bulk metrics as sources of the probefermion response function, not a statement that these metrics are degenerate on-shell solutions of a fixed bulk equation of motion sourced by a definite gravitational action. Whether this degeneracy survives once the metric is required to self-consistently solve the Einstein-Maxwell equations with back-reaction from the fermion (or a bulk matter sector) is a distinct and interesting question that we leave for future work.

This isospectral non-identifiability is precisely the bulk-metric degeneracy expected on general holographic grounds at zero temperature, and it also constitutes a nontrivial success from a machine-learning standpoint: independent training runs, with no built-in knowledge of this expected universality, spontaneously converge onto distinct bulk profiles that nonetheless share the same near-horizon data — indicating that the deep neural network isolates the physical IR fixed point rather than merely overfitting a single UV completion.

## VII. CONCLUSIONS

In this work, we developed a physics-informed deep learning architecture based on Neural Ordinary Diferential Equations (Neural ODEs) to tackle the holographic inverse problem for fermionic observables. By parameterizing unknown bulk fields with continuous deep neural networks and embedding the exact bulk Dirac equations into the backpropagation path, our framework can construct the emergent bulk spacetime and gauge potentials directly from boundary fermionic spectral functions.

Applying this framework to zero-temperature $( T = 0 )$ extremal Reissner-Nordstr¨om AdS spacetime, we demonstrated successful reconstruction of the bulk geometry and gauge potential across three distinct quantum critical spectral regimes at fixed $U ( 1 )$ probe fermion charge q: a non-Fermi liquid without quasiparticle excitations $( q = 1 )$ , a marginal Fermi liquid characteristic of strangemetal transport $( q \ = \ 1 . 5 6 )$ , and a sharp Fermi-liquidlike quasiparticle state $( q \ = \ 2 )$ . All three reconstructions achieved high fidelity, confirming that the framework is versatile and robust against quantitatively different boundary excitations. In Appendix B, we further reconstruct a three-dimensional (BTZ) black hole background, showing that our deep learning framework is not restricted to a particular bulk dimensionality.

Furthermore, we expanded the scope of the inversion by simultaneously extracting the bulk geometry and a matter-sector parameter: promoting the probe Dirac charge q to a trainable parameter alongside the network weights of bulk fields, using the marginal Fermi liquid data, the optimization converged to $q _ { \mathrm { t r a i n e d } } ~ = ~ 1 . 5 6 1 0$ (a 0.06% relative deviation from the target $q \ = \ 1 . 5 6 )$ , with a final loss of $\mathcal { L } _ { \mathrm { f i n a l } } = 7 . 6 6 \times 1 0 ^ { - 6 }$ . This shows that the boundary fermionic spectral function encodes enough physical information to disentangle the background gravitational geometry from the parameters of the matter fields propagating on it.

We also examined the uniqueness of the holographic reconstruction by relaxing the near AdS boundary constraint of bulk spacetime, and found a geometrical degeneracy: bulk metric profiles that difer substantially throughout the radial bulk can reproduce the same boundary spectral data whenever they share the same near-horizon $A d S _ { 2 } \times \mathbb { R } ^ { 2 }$ data. This is holographically expected, since for massless probe fermions the lowenergy exponent $\nu _ { k _ { F } }$ and self-energy $\Sigma ( \omega , \boldsymbol { k } )$ depend only on the near-horizon geometry, not on the precise UVcomplete bulk profile. From a machine-learning standpoint, the spontaneous emergence of such degenerate solutions across independent training runs is itself evidence that the network isolates this IR universality class, subject to the caveats on probe-limit validity and finite-grid resolution discussed in Sec. VI.

Altogether, our results establish a purely data-driven route from boundary fermionic spectra—comparable to angle-resolved photoemission spectroscopy (ARPES) experiments—to an emergent higher-dimensional holographic gravitational dual. A natural next step is to relax the zero-temperature assumption: at finite $\bar { T }$ the Fermi surface is smeared out, so that the retarded Green’s function (i.e., spectral functions) no longer develops a sharp pole at a well-defined momentum, and a recent study [33] has already begun exploring neural-network reconstruction of charged AdS spacetimes directly from finitetemperature fermionic spectral functions. Extending our Neural ODE inverse framework to non-extremal black holes, and ultimately to real ARPES datasets from high-$T _ { c }$ cuprates and heavy-fermion materials, is a promising direction for future work.

More broadly, our results speak to a central question facing the emerging program of data-driven holographic condensed matter theory: to what extent can a bulk gravitational dual be inferred directly from boundary data, without assuming a specific bulk action in advance? Physics-informed machine learning provides the technical means to pose this inverse problem in a controlled way, but our results show that any such program may also confront the intrinsic non-uniqueness of the reconstruction — here traced to IR $A d S _ { 2 } \times \mathbb { R } ^ { 2 }$ universality — rather than treating a single fitted geometry as the unique gravitational dual. Identifying which features of an inferred bulk geometry are physically robust, and which are artifacts of this residual degeneracy, will be essential as these methods are applied to genuine experimental spectra.

## ACKNOWLEDGMENTS

We would like to thank Yongjun Ahn, Johanna Erdmenger, Ki-Seok Kim, Ren´e Meyer, and Yunseok Seo for valuable discussions and correspondence. HSJ was supported by an appointment to the JRG Program at the APCTP through the Science and Technology Promotion Fund and Lottery Fund of the Korean Government. HSJ was also supported by the Korean Local Governments – Gyeongsangbuk-do Province and Pohang City. This work was supported by the Basic Science Research Program through the National Research Foundation of Korea (NRF) funded by the Ministry of Science, ICT & Future Planning (NRF-2021R1A2C1006791) and the AIbased GIST Research Scientist Project grant funded by the GIST in 2025. This work was also supported by Creation of the Quantum Information Science R&D Ecosystem (Grant No. 2022M3H3A106307411) through the National Research Foundation of Korea (NRF) funded by the Korean government (Ministry of Science and ICT). The work of K. H. was supported in part by JSPS KAK-ENHI Grant No. JP22H05111 and JP22H05115. The work of DT was supported by RIKEN Special Postdoctoral Researchers Program. All authors contributed equally to this paper and should be considered as co-first authors.

## Appendix A: Interpretable machine learning of the bulk spacetime

In this paper we have used the Neural ODE method to produce the bulk metric and the gauge field profiles. In general, once neural network methods are used, the interpretability of the solution profiles is lost, since the functional form of any neural network solution is made of multiply nested structure of activation functions, except for the case of Kolmogorov-Arnold networks [47]. This loss of interpretability is a major reason of failure of fur ther reconstruction of the bulk action from the obtained reconstructed bulk profiles of the gravity and gauge fields. Thus here in this appendix we study a diferent method of machine learning with interpretable profiles.

To this end, the simplest method is to consider bulk profiles expanded in physically interpretable bases,

$$
D _ { c } ( z ) \equiv \sum _ { n = 0 } ^ { K } \theta _ { n } ^ { ( c ) } P _ { n } ( 2 z - 1 ) , \qquad c \in \{ f , h , A \}\tag{A1}
$$

where $P _ { n }$ is the Legendre function, $\theta _ { n } ^ { ( c ) }$ is a real trainable coeficient, and $D _ { c } ( z )$ is the unknown function of (18). We put $a = 0$ in this appendix. This Legendre expansion could be thought of physically as a higher derivative expansion of the bulk action, because the Legendre equation is a sort of mixture of the Taylor expansion and Fourier expansion thus its label n can allow the physical interpretation of the radial momentum. $P _ { n } ( 2 z - 1 )$ is an eigenfunction of the operator $- ( d / d z ) [ z ( 1 - \dot { z } ) ( d / \dot { d z } ) ]$ with the eigenvalue $n ( n { + } 1 )$ , and it forms an orthonormal basis. The first $P _ { 0 }$ is a constant, thus accommodates the RN solution

$$
D _ { f } ^ { \mathrm { R N } } = 3 , \quad D _ { h } ^ { \mathrm { R N } } = 0 , \quad D _ { A } ^ { \mathrm { R N } } = 0 .\tag{A2}
$$

This Legendre expansion may help physical interpretation of the bulk profile when the machine learning is applied not just to reproduce the RN solution but with some actual material data. Together with some regularization term which suppresses large n contributions, such as $\mathcal { L } _ { \mathrm { L e g e n d r e } } = n ^ { 2 }$ , the obtained bulk profile incorporates consistency with the low energy expansion, meaning that the bulk action is a low energy efective gravity — no need for strange higher derivative terms.

![](images/94eee70130db892a5da9b9b9fbd938afacf1d8f330a77526dbb4d0607978e1f1.jpg)  
Figure 6. The epoch evolution of the training for the $K = 2$ case.

In this appendix, we just study the reproduction scheme of the RN solution, to look concretely at the degeneracy problem. (We do not use the loss L<sub>Legendre</sub> mentioned above.) Let us train the functions $( \mathrm { A 1 } )$ with the Data B: $q \ = \ 1 . 5 6 .$ , and the data points consist of the imaginary part of the Green’s function evaluated at the direct product of $\omega \in \{ - 0 . 0 8 , - 0 . 0 4 , 0 . 0 4 , 0 . 0 8 \}$ and k ∈ {0.65, 0.85, 1.05, 1.25}. So the data just consists of 16 points. This number is important in judging whether we expect any degeneracy in the bulk profiles after the training, because for $K \leq 5$ the number of parameters 3K is smaller than the number of data points, while for $K \geq 6$ the number of parameters is larger. This means that for the training with $K \ge 6$ we generically expect that we will not reproduce the RN solution, resulting in degenerate bulk profiles.<sup>1</sup> However, as we have discussed in the main text, we have actually encountered a degeneracy due to the fact that the data only depends on the bulk metric profile of the IR end. This means that even in this Legendre interpretable machine learning with $0 < K \leq 5$ we expect degenerate bulk solutions.

The loss function is arranged as

$$
\begin{array} { r l r } {  { \mathcal { L } _ { \mathrm { t o t a l } } \equiv \mathcal { L } _ { \mathrm { d a t a } } + 1 0 ^ { 6 } \mathcal { L } _ { \mathrm { d o m } } , } } & { \quad \mathrm { ( A 3 ) } } \\ & { \mathcal { L } _ { \mathrm { d a t a } } \equiv \frac { N ^ { - 1 } \sum _ { i = 1 } ^ { N } [ \mathcal { A } _ { \theta } ( \omega _ { i } , k _ { i } ) - \mathcal { A } _ { i } ^ { \mathrm { d a t a } } ] ^ { 2 } } { \operatorname* { m a x } \{ 1 6 ^ { - 1 } \sum _ { i = 1 } ^ { 1 6 } ( \mathcal { A } _ { i } ^ { \mathrm { d a t a } } ) ^ { 2 } , 1 0 ^ { - 4 } \} } , } \\ & { \mathcal { L } _ { \mathrm { d o m } } \equiv \frac { 1 } { 2 5 7 } \sum _ { j = 1 } ^ { 2 5 7 } \big [ \mathrm { R e L U } ( 5 \times 1 0 ^ { - 3 } - ( 1 + 2 z _ { j } + z _ { j } ^ { 2 } D _ { f } ( z _ { j } ) ) ) ^ { 2 } } \\ & { } & { \quad + \mathrm { R e L U } ( 5 \times 1 0 ^ { - 3 } - h ( z _ { j } ) ) ^ { 2 } \big ] . } \end{array}
$$

The first term is for the data fitting, while the second term is to make sure that the functions $f ( z )$ and $h ( z )$ are positive, and we picked up evaluation points $z _ { j } \ \equiv$ $( j - 1 ) / 2 5 6$ (with $j = 1 , 2 , \cdots , 2 5 7 )$

The training history for the various initial conditions for the parameters θ in the case $K = 2$ is shown in Fig. 6. All trainings attained lowered loss values of $\mathcal { O } ( 1 0 ^ { - 8 } )$

The trained bulk profiles are shown in ${ \mathrm { F i g } } .$ 7. It provides bulk configurations looking quite diferent from the RN solution, keeping the Green function data intact. This means the degeneracy, which has been expected from the observation that the data actually looks at the bulk profile only at the IR end (which is the extremal black hole horizon).

In fact, as seen in the bottom right panel of Fig. 7, all the trained profiles go to $D _ { f } = 3$ at the horizon $z = 1$ This is consistent with the fact that the data is solely made out of the IR $\mathrm { { A d S } _ { 2 } }$ geometry with (24).

This same degeneracy is also found, albeit less frequently, in the Neural ODE framework of Sec. VI: standard training at $a = 0$ reliably converges to the exact RN AdS solution (Sec. V), but atypical initial conditions occasionally drive the optimization to alternative solutions. Three such solutions, shown in Fig. 5, difer substantially both from the exact RN AdS geometry and from one another, yet reproduce the same boundary spectral data, with final losses of $\mathcal { L } _ { \mathrm { d a t a } } = 1 . 6 5 \times 1 0 ^ { - 5 } , 5 . 9 \dot { 2 } \times 1 0 ^ { - 4 }$ , and $5 . 0 5 \times 1 0 ^ { - 4 }$ , respectively.

![](images/60701311db001bbbf176c570b5aab696703d20809b86d999e6ade9852d4633d3.jpg)  
Figure 7. The obtained bulk profiles (top) and the functions $D _ { c } ( z )$ (bottom) for the case $K = 2$ , with various initial values of θ for the training. Although the trained bulk profiles (top) look the same as the RN solution, they are indeed diferent, as obviously shown in the bottom figures shown in $D _ { c } ( z )$ variables.

In actual situations with experimental datasets for the fermion Green’s functions, it is better to use large K Legendre profiles but with the loss function $\mathcal { L } _ { \mathrm { L e g e n d r e } } = \bar { n ^ { 2 } }$ to suppress the large radial momenta in the bulk. This will naturally accommodate both the expressibility of the profile functions and the resolution of the degeneracy to obtain a physical bulk configuration which allows low energy efective action in the bulk. The goal of the reconstruction program is not just reproducing the bulk metric profiles but goes to pin down the bulk action itself, which should be derivative expanded to make sure the low energy efective description of the holographic gravitational picture.

We end with some comments on the interpretable neural networks which have been developed in [19] and subsequent papers. There the bulk is directly interpreted as a neural network, and the neural network weights are the bulk metrics. This method actually allows the direct interpretation of the neural network weights, but it sufers from the same degeneracy problem: when the number of data points is small, the bulk metric has a lot of degeneracy. In particular, to make the bulk propagation numerically trusted, one needs to introduce a lot of bulk radial points on each of which the metric function provides the weights and thus the number of network parameters gets huge, resulting in the degeneracy. For example in [48] the Runge-Kutta layers were introduced to make the bulk integration stable, with a large number of layers. This method works only when the number of data points is comparably large. In realistic cases with experimental data at nonzero temperature, we are short of the data points, meaning that the direct interpretable neural network such as the one in [19] may not be suitable to use.

## Appendix B: Emergent spacetime from analytic fermionic spectral functions

As another benchmark of the bulk spacetime reconstruction from boundary fermionic spectral function, we consider three-dimensional black hole geometry, especially Ba˜nados-Teitelboim-Zanelli (BTZ) black holes [49, 50]. Since the retarded Green’s function for fermions can be computed analytically in a BTZ black hole background [51], we utilize this analytic fermionic spectral function as input data for bulk reconstruction within our Neural ODE framework. This serves as a complementary approach to our numerical analysis presented in the main text.

## 1. Analytic fermionic Green’s functions

The BTZ metric is given by

$$
\mathrm { d } s ^ { 2 } = - \left( r ^ { 2 } - 1 \right) \mathrm { d } t ^ { 2 } + \frac { \mathrm { d } r ^ { 2 } } { r ^ { 2 } - 1 } + r ^ { 2 } \mathrm { d } x ^ { 2 } ,\tag{B1}
$$

where the horizon is at $r \ = \ 1$ , and Hawking temperature reads $T = 1 / ( 2 \pi )$ . Following [51], it is convenient to introduce

$$
r = \cosh \rho ,\tag{B2}
$$

for which the metric (B1) becomes

$$
\begin{array} { r } { \begin{array} { r l } & { \mathrm { d } s ^ { 2 } = - \sinh ^ { 2 } \rho \mathrm { d } t ^ { 2 } + \cosh ^ { 2 } \rho \mathrm { d } x ^ { 2 } + \mathrm { d } \rho ^ { 2 } . } \end{array} } \end{array}\tag{B3}
$$

In this frame, the Fourier decomposition takes the form

$$
\Psi = e ^ { - i \omega t + i k x } \psi ( \rho ) .\tag{B4}
$$

Starting from the Dirac equation discussed in Sec. II, its radial part takes the form

$$
\begin{array} { l } { \displaystyle \left[ \Gamma ^ { \varrho } \left( \partial _ { \rho } + \frac { 1 } { 2 } \left( \frac { \cosh \rho } { \sinh \rho } + \frac { \sinh \rho } { \cosh \rho } \right) \right) \right. } \\ { \displaystyle \left. + i \left( \frac { \Gamma ^ { \underline { { x } } } k } { \cosh \rho } - \frac { \Gamma ^ { \underline { { t } } } \omega } { \sinh \rho } \right) - m \right] \psi = 0 , } \end{array}\tag{B5}
$$

where

$$
\Gamma ^ { \underline { { \rho } } } = \sigma _ { 3 } , \Gamma ^ { \underline { { t } } } = i \sigma _ { 2 } , \Gamma ^ { \underline { { x } } } = \sigma _ { 1 } , \boldsymbol { \psi } ^ { \mathrm { T } } = \left( \psi _ { + } , \psi _ { - } \right) .\tag{B6}
$$

To reduce Eq. (B5) to a hypergeometric system, we define

$$
\psi _ { \pm } = \sqrt { \frac { \cosh \rho \pm \sinh \rho } { \cosh \rho \sinh \rho } } \left( \chi _ { 1 } \pm \chi _ { 2 } \right) , \quad u = \operatorname { t a n h } ^ { 2 } \rho .\tag{B7}
$$

The compact radial coordinate u places the horizon at $u = 0$ and the AdS boundary at $u = 1$ . The two first-

order equations then become

$$
\begin{array} { r l r } {  { 2 ( 1 - u ) \sqrt { u } \ \partial _ { u } \chi _ { 1 } - i ( \frac { \omega } { \sqrt { u } } + k \sqrt { u } ) \chi _ { 1 } } } \\ & { } & \\ & { } & { = [ m - \frac { 1 } { 2 } + i ( \omega + k ) ] \chi _ { 2 } , } \\ & { } & \\ & { } & { 2 ( 1 - u ) \sqrt { u } \ \partial _ { u } \chi _ { 2 } + i ( \frac { \omega } { \sqrt { u } } + k \sqrt { u } ) \chi _ { 2 } } \\ & { } & \\ & { } & { = [ m - \frac { 1 } { 2 } - i ( \omega + k ) ] \chi _ { 1 } . } \end{array}\tag{B8}
$$

These Dirac equations can be solved analytically with the hypergeometric function ${ } _ { 2 } F _ { 1 } ( { \mathfrak { a } } , { \mathfrak { b } } ; { \mathfrak { c } } ; u )$ for the infalling solution as follows:

$$
\begin{array} { l } { { \displaystyle \chi _ { 1 } \big ( u \big ) = \frac { { \mathfrak a } - { \mathfrak c } } { \mathfrak c } u ^ { \alpha + \frac 1 2 } \big ( 1 - u \big ) ^ { \beta } { } _ { 2 } F _ { 1 } \big ( { \mathfrak a } , { \mathfrak b } + 1 ; { \mathfrak c } + 1 ; u \big ) , \ } } \\ { { \ } } \\ { { \displaystyle \chi _ { 2 } \big ( u \big ) = u ^ { \alpha } \big ( 1 - u \big ) ^ { \beta } { } _ { 2 } F _ { 1 } \big ( { \mathfrak a } , { \mathfrak b } ; { \mathfrak c } ; u \big ) , \ } } \end{array}\tag{B9}
$$

where

$$
\begin{array} { c } { { \alpha = \displaystyle - \frac { i \omega } { 2 } \ : , \qquad \beta = \displaystyle - \frac { 1 } { 4 } + \frac { m } { 2 } \ : , } } \\ { { { } } } \\ { { { \bf { a } } = \displaystyle \frac { 1 } { 2 } \ : \left( m + \frac { 1 } { 2 } \right) - \displaystyle \frac { i } { 2 } ( \omega - k ) \ : , } } \\ { { { } } } \\ { { { \sf { b } } = \displaystyle \frac { 1 } { 2 } \ : \left( m - \displaystyle \frac { 1 } { 2 } \right) - \displaystyle \frac { i } { 2 } ( \omega + k ) \ : , \qquad { \bf { c } } = \displaystyle \frac { 1 } { 2 } - i \omega \ : . } } \end{array}\tag{B10}
$$

Plugging (B9) into (B7), one can find the near AdS boundary behaviors as

$$
\begin{array} { c } { { \psi _ { + } \approx A ( 1 - u ) ^ { \frac { 1 } { 2 } - \frac { m } { 2 } } + B ( 1 - u ) ^ { 1 + \frac { m } { 2 } } , } } \\ { { \psi _ { - } \approx C ( 1 - u ) ^ { 1 - \frac { m } { 2 } } + D ( 1 - u ) ^ { \frac { 1 } { 2 } + \frac { m } { 2 } } . } } \end{array}\tag{B11}
$$

For $m \geq 0$ , A is the source and D is the response in the standard quantization. With the overall sign chosen consistently with the positive spectral-density convention in Eq. (11), the retarded Green’s function is

$$
G _ { R } ( \omega , k ) = - i \frac { D } { A } = i \frac { \Gamma \left( \frac { 1 } { 2 } - m \right) } { \Gamma \left( \frac { 1 } { 2 } + m \right) } \mathcal { F } _ { L } ( \omega , k ) \mathcal { F } _ { R } ( \omega , k ) ,\tag{B12}
$$

where

$$
\begin{array} { r l } & { { \mathcal F } _ { L } ( \omega , k ) = \displaystyle \frac { \Gamma \left( \frac { 1 } { 4 } + \frac { m } { 2 } - i ( \omega - k ) \right) } { \Gamma \left( \frac { 3 } { 4 } - \frac { m } { 2 } - i ( \omega - k ) \right) } \mathrm { , } } \\ & { { \mathcal F } _ { R } ( \omega , k ) = \displaystyle \frac { \Gamma \left( \frac { 3 } { 4 } + \frac { m } { 2 } - i ( \omega + k ) \right) } { \Gamma \left( \frac { 1 } { 4 } - \frac { m } { 2 } - i ( \omega + k ) \right) } \mathrm { . } } \end{array}\tag{B13}
$$

We display the fermionic spectral function (B12) for massless fermions in Fig. 8.

## 2. Neural ODE reconstruction

Next, using the Neural ODE framework, we reconstruct the BTZ geometry from the analytic spectral

ω

![](images/4def27e39a650444428666d98ec2558c31ec2d9157893888b8876f90e73e5917.jpg)

![](images/69e179db151a1e01511c60abaaa1cb3adda96d9929a53277549dce481773245d.jpg)  
Figure 8. The analytic fermionic spectral function of the BTZ black hole for $m = 0 .$ The left panel displays the density plot in the (ω, k) plane. The right panel shows ω-dependent slices at fixed momenta.

function given in Fig. 8. To align with the inverseproblem framework used in the main text, we rewrite the Dirac equation in terms of the new compact coordinate $u = 1 - z ^ { \hat { 2 } }$ , where the metric ansatz (1) becomes

$$
\mathrm { d } s ^ { 2 } = - { \frac { f ( u ) } { 1 - u } } \mathrm { d } t ^ { 2 } + { \frac { \mathrm { d } u ^ { 2 } } { 4 f ( u ) ( 1 - u ) ^ { 2 } } } + { \frac { h ( u ) } { 1 - u } } \mathrm { d } x ^ { 2 } .\tag{B14}
$$

Here, the horizon and AdS boundary are located at $u =$ 0 and $u = 1$ , respectively. The black hole horizon and asymptotic AdS conditions are

$$
f ( 0 ) = 0 , \qquad f ( 1 ) = 1 , \qquad h ( 1 ) = 1 .\tag{B15}
$$

In this new coordinate system, the exact BTZ solution (B1) becomes

$$
f ( u ) = u , \qquad h ( u ) = 1 .\tag{B16}
$$

Substituting Eq. (B14) into the Dirac equation and using the $\chi _ { 1 , 2 }$ basis defined in $\operatorname { E q . }$ . (B7), one can find

$$
\begin{array} { r l } & { 4 ( 1 - u ) \chi _ { 1 } ^ { \prime } ( u ) + \left[ \mathcal { P } ( u ) - i \mathcal { Q } ( u ) \right] \chi _ { 1 } ( u ) } \\ & { \qquad = \left[ \mathcal { R } ( u ) + i S ( u ) \right] \chi _ { 2 } ( u ) , } \\ & { 4 ( 1 - u ) \chi _ { 2 } ^ { \prime } ( u ) + \left[ \mathcal { P } ( u ) + i \mathcal { Q } ( u ) \right] \chi _ { 2 } ( u ) } \\ & { \qquad = \left[ \mathcal { R } ( u ) - i S ( u ) \right] \chi _ { 1 } ( u ) , } \end{array}\tag{B17}
$$

where

$$
\begin{array} { l } { \displaystyle \mathcal { P } ( u ) = ( 1 - u ) \left[ - \frac { 1 } { u } + \frac { f ^ { \prime } ( u ) } { f ( u ) } + \frac { h ^ { \prime } ( u ) } { h ( u ) } \right] , } \\ { \displaystyle \mathcal { Q } ( u ) = 2 \left[ \frac { \omega } { f ( u ) } + \frac { k \sqrt { u } } { \sqrt { f ( u ) } \sqrt { h ( u ) } } \right] , } \\ { \displaystyle \mathcal { R } ( u ) = - \frac { 1 } { \sqrt { u } } + \frac { 2 m } { \sqrt { f ( u ) } } , } \\ { \displaystyle \mathcal { S } ( u ) = 2 \left[ \frac { \omega \sqrt { u } } { f ( u ) } + \frac { k } { \sqrt { f ( u ) } \sqrt { h ( u ) } } \right] . } \end{array}\tag{B18}
$$

The infalling behavior at the horizon and the leading

fallof near the AdS boundary are made explicit via

$$
\begin{array} { l } { { \chi _ { 1 } ( u ) = u ^ { - \frac { i \omega } { 2 f ^ { \prime } ( 0 ) } } \sqrt { u } ( 1 - u ) ^ { - \frac { 1 } { 4 } + \frac { m } { 2 } } X _ { 1 } ( u ) , } } \\ { { \chi _ { 2 } ( u ) = u ^ { - \frac { i \omega } { 2 f ^ { \prime } ( 0 ) } } \bigl ( 1 - u \bigr ) ^ { - \frac { 1 } { 4 } + \frac { m } { 2 } } X _ { 2 } ( u ) . } } \end{array}\tag{B19}
$$

The functions $X _ { 1 }$ and $X _ { 2 }$ are regular at the horizon. Near the AdS boundary $u = 1$ , they are expanded as

$$
\begin{array} { l } { { \displaystyle X _ { 1 } ( u ) \sim \sum _ { n = 0 } a _ { n } ( 1 - u ) ^ { n } + ( 1 - u ) ^ { \frac { 1 } { 2 } - m } \sum _ { n = 0 } b _ { n } ( 1 - u ) ^ { n } , } } \\ { { \displaystyle X _ { 2 } ( u ) \sim \sum _ { n = 0 } c _ { n } ( 1 - u ) ^ { n } + ( 1 - u ) ^ { \frac { 1 } { 2 } - m } \sum _ { n = 0 } d _ { n } ( 1 - u ) ^ { n } . } } \end{array}\tag{B20}
$$

For each pair $( \omega _ { i } , k _ { i } )$ , we solve the coupled equations (B17) for $X _ { 1 }$ and $X _ { 2 }$ from the near-horizon region to the near-boundary region. The numerical solutions are then fitted to Eqs. (B20) to determine $a _ { 0 } , b _ { 0 } , c _ { 0 }$ , and $d _ { 0 }$ The Green’s function can be expressed in terms of these coeficients as

$$
G _ { R } = \left\{ \begin{array} { l l } { - 2 i \displaystyle \frac { b _ { 0 } + d _ { 0 } } { a _ { 0 } - c _ { 0 } } , } & { m < 0 , } \\ { - \displaystyle \frac { i } { 2 } \displaystyle \frac { a _ { 0 } - c _ { 0 } } { b _ { 0 } + d _ { 0 } } , } & { m \ge 0 . } \end{array} \right.\tag{B21}
$$

We use the same neural-network architecture, ODE solver, Adam–L-BFGS optimization procedure as in Sec. IV. The two changes specific to the BTZ reconstruction are the hard constraints

$$
f _ { \theta } ( u ) = u D _ { f } ( u ; \theta _ { f } ) , \quad h _ { \theta } ( u ) = D _ { h } ( u ; \theta _ { h } ) ,\tag{B22}
$$

and the inclusion of the AdS boundary conditions (B15) in the loss function for computational convenience:

$$
\mathcal { L } _ { \mathrm { c o n d } } = \left[ f _ { \theta } ( 1 ) - 1 \right] ^ { 2 } + \left[ h _ { \theta } ( 1 ) - 1 \right] ^ { 2 } .\tag{B23}
$$

The total loss is therefore

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { d a t a } } + \mathcal { L } _ { \mathrm { c o n d } } , } \end{array}\tag{B24}
$$

where ${ \mathcal { L } } _ { \mathrm { d a t a } }$ has the same form as Eq. (20).

![](images/12e0caa033c8156193dbb34a4ccef05f1d36489c39090124d9454fb9eed5f56e.jpg)

![](images/3100f42a2942adee95a703205c75b464d5034352ecc87eb3b06886f357fdf3db.jpg)  
Figure 9. Neural ODE reconstruction of the BTZ black hole spacetime from the massless fermionic spectral data $\left( { \mathrm { F i g } } . \ 8 \right)$ . The trained metric functions are in good agreement with the reference functions of the exact BTZ background (B16).

The analytic massless fermionic spectral function shown in $\mathrm { F i g . ~ 8 }$ is sampled on an approximately regular grid, following the same digitization procedure used for the spectral data in the main text. The reconstruction displayed below uses the window $0 < \omega < 1$ and $0 < k < 1$

As shown in Fig. 9, the trained functions are consistent with the analytic BTZ spacetime (B16). The optimization achieves a final loss of $8 . 0 3 \times 1 0 ^ { - 4 }$ The successful recovery of both the blackening function $f ( u )$ and the spatial metric component $h ( u )$ from the boundary fermionic spectrum provides an additional benchmark for the black hole spacetime reconstruction from the boundary fermionic data.

[1] S. Sachdev, Quantum Phase Transitions. Cambridge University Press, 4, 2011, 10.1017/cbo9780511973765.

[2] P. W. Anderson, “luttinger-liquid” behavior of the normal metallic state of the 2d hubbard model, Physical Review Letters 64 (1990) 1839–1841.

[3] C. M. Varma, Phenomenology of the normal state of cu-o high-temperature superconductors, Physical Review Letters 63 (1989) 1996–1999.

[4] C. Varma, Z. Nussinov and W. van Saarloos, Singular or non-fermi liquids, Physics Reports 361 (2002) 267–417.

[5] S.-S. Lee, A Non-Fermi Liquid from a Charged Black Hole: A Critical Fermi Ball, Phys.Rev. D79 (2009) 086006, [0809.3402].

[6] H. Liu, J. McGreevy and D. Vegh, Non-Fermi liquids from holography, Phys. Rev. D83 (2011) 065029, [0903.2477].

[7] M. Cubrovic, J. Zaanen and K. Schalm, String Theory, Quantum Phase Transitions and the Emergent Fermi-Liquid, Science 325 (2009) 439–444, [0904.1993].

[8] T. Faulkner, N. Iqbal, H. Liu, J. McGreevy and D. Vegh, Strange metal transport realized by gauge/gravity duality, Science 329 (2010) 1043–1047.

[9] N. Iqbal, H. Liu and M. Mezei, Lectures on holographic non-Fermi liquids and quantum phase transitions, in Theoretical Advanced Study Institute in Elementary Particle Physics: String theory and its Applications: From meV to the Planck Scale, pp. 707–816, 10, 2011. 1110.3814. DOI.

[10] J. M. Maldacena, The Large N limit of superconformal field theories and supergravity, Adv.Theor.Math.Phys. 2 (1998) 231–252, [hep-th/9711200].

[11] S. S. Gubser, I. R. Klebanov and A. M. Polyakov,

Gauge theory correlators from non-critical string theory, Phys. Lett. B428 (1998) 105–114, [hep-th/9802109].

[12] E. Witten, Anti-de Sitter space and holography, Adv. Theor. Math. Phys. 2 (1998) 253–291, [hep-th/9802150].

[13] J. Zaanen, Y.-W. Sun, Y. Liu and K. Schalm, Holographic Duality in Condensed Matter Physics. Cambridge Univ. Press, 2015.

[14] M. Ammon and J. Erdmenger, Gauge/gravity duality. Cambridge Univ. Pr., Cambridge, UK, 2015.

[15] S. A. Hartnoll, A. Lucas and S. Sachdev, Holographic Quantum Matter. MIT Press, 2018.

[16] T. Faulkner, H. Liu, J. McGreevy and D. Vegh, Emergent quantum criticality, Fermi surfaces, and AdS(2), Phys.Rev. D83 (2011) 125002, [0907.2694].

[17] G. E. Karniadakis, I. G. Kevrekidis, L. Lu, P. Perdikaris, S. Wang and L. Yang, Physics-informed machine learning, Nature Reviews Physics 3 (May, 2021) 422–440.

[18] G. Carleo, Machine learning and the physical sciences, Reviews of Modern Physics 91 (2019) .

[19] K. Hashimoto, S. Sugishita, A. Tanaka and A. Tomiya, Deep learning and the AdS/CFT correspondence, Phys. Rev. D 98 (2018) 046019, [1802.08313].

[20] A. Tanaka, A. Tomiya and K. Hashimoto, Deep Learning and Physics. Springer Singapore, 2021, 10.1007/978-981-33-6108-9.

[21] Y. LeCun, Y. Bengio and G. Hinton, Deep learning, Nature 521 (May, 2015) 436–444.

[22] J. Schmidhuber, Deep learning in neural networks: An overview, Neural Networks 61 (2015) 85–117.

[23] T. Akutagawa, K. Hashimoto and T. Sumimoto, Deep

Learning and AdS/QCD, Phys. Rev. D 102 (2020) 026020, [2005.02636].

[24] K. Hashimoto, H.-Y. Hu and Y.-Z. You, Neural ordinary diferential equation and holographic quantum chromodynamics, Mach. Learn. Sci. Tech. 2 (2021) 035011, [2006.00712].

[25] X. Chen and M. Huang, Machine learning holographic black hole from lattice QCD equation of state, Phys. Rev. D 109 (2024) L051902, [2401.06417].

[26] Y.-K. Yan, S.-F. Wu, X.-H. Ge and Y. Tian, Deep learning black hole metrics from shear viscosity, Phys. Rev. D 102 (4, 2020) 101902, [2004.12112].

[27] H.-S. Jeong, H. Kim, K.-Y. Kim, G. Yun, H. Yu and K. Yun, AdS/Deep-Learning made easy II: neural network-based approaches to holography and inverse problems, 2511.22522.

[28] K. Li, Y. Ling, P. Liu and M.-H. Wu, Learning the black hole metric from holographic conductivity, Phys. Rev. D 107 (2023) 066021, [2209.05203].

[29] S. Kim, K. K. Kim and Y. Seo, Phase diagram from nonlinear interaction between superconducting order and density: toward data-based holographic superconductor, JHEP 02 (2025) 077, [2410.06523].

[30] B. Ahn, H.-S. Jeong, K.-Y. Kim and K. Yun, Deep learning bulk spacetime from boundary optical conductivity, JHEP 03 (2024) 141, [2401.00939].

[31] B. Ahn, H.-S. Jeong, C.-W. Ji, K.-Y. Kim and K. Yun, Deep learning-based holography for T-linear resistivity, Phys. Rev. D 112 (2025) 126008, [2502.10245].

[32] B. Ahn, H.-S. Jeong, K.-Y. Kim and K. Yun, Holographic reconstruction of black hole spacetime: machine learning and entanglement entropy, JHEP 01 (2025) 025, [2406.07395].

[33] H.-Z. Xiao, Z.-Z. He, Z.-Y. Xian and S.-F. Wu, Holographic Learning from Fermionic Spectra: Application to Strange Metal Phenomenology, 2607.02861.

[34] R. T. Q. Chen, Y. Rubanova, J. Bettencourt and D. Duvenaud, Neural Ordinary Diferential Equations, 1806.07366.

[35] E. Dupont, A. Doucet and Y. W. Teh, Augmented neural odes, in Advances in Neural Information Processing Systems (H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alch´e-Buc, E. Fox and R. Garnett, eds.), vol. 32, Curran Associates, Inc., 2019.

[36] S. Massaroli, M. Poli, J. Park, A. Yamashita and H. Asama, Dissecting neural odes, in Advances in Neural Information Processing Systems (H. Larochelle, M. Ranzato, R. Hadsell, M. Balcan and H. Lin, eds.), vol. 33, pp. 3952–3963, Curran Associates, Inc., 2020.

[37] H. Yan, J. Du, V. Y. F. Tan and J. Feng, On robustness of neural ordinary diferential equations, ArXiv abs/1910.05513 (2019) .

[38] C. M. Varma, P. B. Littlewood, S. Schmitt-Rink,

E. Abrahams and A. E. Ruckenstein, Phenomenology of the normal state of Cu-O high-temperature superconductors, Phys. Rev. Lett. 63 (1989) 1996–1999.

[39] G. Aeppli, T. E. Mason, S. M. Hayden, H. A. Mook and J. Kulda, Nearly singular magnetic fluctuations in the normal state of a high-t<sub>c</sub> cuprate superconductor, Science 278 (Nov., 1997) 1432–1435.

[40] T. Valla, A. V. Fedorov, P. D. Johnson, B. O. Wells, S. L. Hulbert, Q. Li et al., Evidence for quantum critical behavior in the optimally doped cuprate bi<sub>2</sub>sr<sub>2</sub>cacu<sub>2</sub> o<sub>8</sub>+<sub>δ</sub>, Science 285 (1999) 2110–2113.

[41] J. L. Tallon, J. W. Loram, G. V. M. Williams, J. R. Cooper, I. R. Fisher, J. D. Johnson et al., Critical doping in overdoped high-tc superconductors - a quantum critical point?, phys. stat. sol. (b) 215 (1999) 531, [cond-mat/9911157].

[42] D. v. d. Marel, H. J. A. Molegraaf, J. Zaanen, Z. Nussinov, F. Carbone, A. Damascelli et al., Quantum critical behaviour in a high-tc superconductor, Nature 425 (2003) 271–274.

[43] P. Gegenwart, Q. Si and F. Steglich, Quantum criticality in heavy-fermion metals, Nature Physics 4 (Mar., 2008) 186–197.

[44] E. Abrahams and C. M. Varma, What angle-resolved photoemission experiments tell about the microscopic theory for high-temperature superconductors, Proceedings of the National Academy of Sciences 97 (May, 2000) 5714–5716.

[45] D. P. Kingma and J. Ba, Adam: A Method for Stochastic Optimization, in International Conference on Learning Representations, 12, 2014. 1412.6980.

[46] D. C. Liu and J. Nocedal, On the limited memory BFGS method for large scale optimization, Math. Programming 45 (1989) 503–528.

[47] Z. Liu, Y. Wang, S. Vaidya, F. Ruehle, J. Halverson, M. Soljacic et al., Kan: Kolmogorov–arnold networks, in International conference on learning representations, vol. 2025, pp. 70367–70413, 2025.

[48] K. Hashimoto, K. Matsuo, M. Murata, G. Ogiwara and D. Takeda, Machine-learning emergent spacetime from linear response in future tabletop quantum gravity experiments, Machine Learning: Science and Technology 6 (2025) 015030.

[49] M. Banados, C. Teitelboim and J. Zanelli, The Black hole in three-dimensional space-time, Phys. Rev. Lett. 69 (1992) 1849–1851, [hep-th/9204099].

[50] M. Banados, M. Henneaux, C. Teitelboim and J. Zanelli, Geometry of the (2+1) black hole, Phys. Rev. D 48 (1993) 1506–1525, [gr-qc/9302012].

[51] N. Iqbal and H. Liu, Real-time response in AdS/CFT with application to spinors, Fortsch. Phys. 57 (2009) 367–384, [0903.2596].