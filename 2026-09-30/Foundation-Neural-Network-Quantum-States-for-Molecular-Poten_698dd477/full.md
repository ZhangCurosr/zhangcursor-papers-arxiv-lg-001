# Foundation Neural-Network Quantum States for Molecular Potential Energy Surfaces in Second Quantization

Lizhong Fu<sup>1</sup>, Jianan Wei<sup>2,3</sup>, Wenguan Wang<sup>2,3,\*</sup>, Honghui Shang<sup>1,\*</sup>

<sup>1</sup>State Key Laboratory of Precision and Intelligent Chemistry, University of Science and Technology of China

<sup>2</sup>State Key Lab of Brain-Machine Intelligence, Zhejiang University

<sup>3</sup>College of Artificial Intelligence, Zhejiang University

Corresponding authors

## Abstract

Second-quantized neural-network quantum states have achieved accurate molecular energies, but extending them across molecular geometries requires a shared representation of the geometry-dependent wavefunction coeficients. We introduce geometry-conditioned foundation neural-network quantum states for molecular electronic structure in second quantization. A single autoregressive model learns a family of ground states from sparse anchor geometries and provides wavefunctions at untrained geometries without further optimization. Orbital alignment matches orbital identities and transports their phases, establishing an aligned orbital basis across geometries. Frozen energies reach chemical accuracy at every untrained query geometry for N<sub>2</sub>, CO, and H<sub>4</sub>. On additional molecular paths, the energy-trained wavefunctions yield dipoles, quadrupoles, and natural occupations without property labels. Across three paired N training seeds, orbital alignment lowers the mean absolute energy error over all untrained query geometries from 34–37 mHa to 0.049–0.085 mHa. At approximately 1 mHa mean absolute error, frozen evaluation reduces the per-geometry cost by 986× relative to independent optimization, yielding an estimated 25.8× end-to-end GPU-cost reduction on a 161-point $\mathrm { N _ { 2 } }$ grid.

Keywords: Neural-network quantum states, potential energy surfaces, second quantization, variational Monte Carlo

## 1 Introduction

Molecular potential energy surfaces (PESs) connect electronic structure to vibrations, conformational changes, and chemical reactions. Neural-network quantum states (NQSs) provide an expressive ansatz for many-body wavefunctions [1, 2]. For molecules in second quantization, NQSs parameterize the wavefunction coeficients in an occupation basis and achieve accurate electronic energies through ab initio variational optimization [3–8]. Constructing a PES with independently optimized NQSs, however, repeats this optimization at each geometry, leaving the shared electronic structure along the molecular path largely unused [9].

Sharing an NQS across a family of Hamiltonians ofers a route to amortizing repeated wavefunction optimization. Foundation neural-network quantum states (FNQSs) establish this principle for quantum spin systems by conditioning a shared model on Hamiltonian couplings [10]. Extending this framework to molecular electronic structure in second quantization requires a compact conditioning representation for the molecular Hamiltonian family. In this work, we introduce a geometry-conditioned FNQS that combines molecular geometry with orbital descriptors to provide global and orbital-specific conditioning. Jointly trained at sparse anchor geometries along a PES, the model provides wavefunctions at untrained geometries with frozen parameters, from which energies and electronic observables are evaluated.

Sharing a second-quantized wavefunction model across molecular geometries also requires a consistent orbital basis [11, 12]. Independently generated molecular orbitals can change order and sign across geometries. The same occupation string can then refer to diferent orbital identities or phase conventions, introducing artificial discontinuities in the wavefunction coeficients even when the physical state varies smoothly. We align orbital identities and phases along the path to establish a consistent basis for the shared model. Ablation studies show that this orbital consistency enables accurate predictions at untrained geometries.

Our contributions are:

1. FNQSs for molecular second quantization. To our knowledge, this is the first variationally trained, geometry-conditioned NQS for molecular electronic structure in second quantization that predicts wavefunctions at untrained geometries with frozen parameters. Frozen predictions reach chemical accuracy at every untrained query geometry for $\mathrm { N } _ { \mathrm { 2 } } , \mathrm { C O }$ , and $\mathrm { H } _ { 4 }$ . Energy-trained wavefunctions also yield dipoles, quadrupoles, and natural occupations without propertyspecific training.

2. Orbital consistency for accurate parameter sharing. We combine orbital matching and phase alignment with geometry and per-orbital conditioning. Across three paired $\mathrm { N _ { 2 } }$ seeds, alignment reduces the mean absolute errors over all non-anchor queries from 34–37 to 0.049–0.085 mHa, demonstrating that orbital consistency enables accurate frozen transfer.

3. Amortized computation without query-time optimization. On a 161-point $\mathrm { N _ { 2 } }$ grid, our method yields estimated end-to-end GPU-cost reductions of 25.8× over independent optimization and 2.53× over shared pretraining with fine-tuning, using fixed budgets with MAEs near 1 mHa. The comparison includes initial training and energy evaluation; a cost model separates this initial investment from the cost of additional geometries.

## 2 Related work

Shared real-space wavefunctions and potential energy surfaces. PESNet trains a single real-space wavefunction across molecular geometries, reducing repeated variational optimization [13]. PlaNet adds an energy surrogate trained on intermediate VMC estimates, enabling energy queries without Monte Carlo sampling at inference [14]. Both methods use real-space formulations, parameterizing electronic wavefunctions over continuous electron coordinates [15–17]. Related work extends reuse across molecules through transferable orbital constructions and pretrained wavefunctions [18–20]. We instead adopt a second-quantized autoregressive representation, which facilitates comparison with conventional quantum-chemical methods and evaluation of electronic observables, while also enabling direct sampling under particle-number and symmetry constraints [4, 5, 21].

Foundation neural-network quantum states. FNQSs condition a shared spin wavefunction on Hamiltonian couplings [10]. Their embedding concatenates O(1) global couplings to each configuration patch, or pairs patches of $O ( N )$ couplings with the corresponding configuration patches, where N is the number of spins. The authors identify second-quantized fermions and molecular systems as directions for extension. Molecular two-electron integrals carry four orbital indices and number $O ( M ^ { 4 } )$ for M orbitals, so their full tensor does not directly fit either input construction. Molecular conditioning therefore calls for an appropriate encoding of the Hamiltonian family, alongside orbital alignment to maintain a consistent meaning for the learned coeficients. We use geometry and aligned per-orbital descriptors for this encoding.

Wavefunction interpolation. Eigenvector continuation (EC) interpolates wavefunctions by solving the query Hamiltonian in the span of anchor states [22, 23]. Molecular extensions use a consistent orbital basis to transfer correlated states across geometries and calculate energies and other electronic properties [12]. Applying EC to sampled NQSs requires estimating cross-state overlaps and Hamiltonian matrix elements. Near-linear dependence among anchor states can amplify sampling errors in the resulting generalized eigenvalue problem and produce spurious low-energy solutions; overlap-spectrum truncation and other noise-aware regularization methods address this sensitivity [24, 25]. We instead learn a geometry-conditioned wavefunction whose frozen-query evaluation requires neither cross-state matrix elements nor a generalized eigenvalue solve.

## 3 Method

Problem setup. Fix a molecule, an atomic-orbital basis set, and spin populations $( N _ { \alpha } , N _ { \beta } )$ in M retained spatial orbitals. The electronic Hamiltonian at nuclear geometry R is

$$
\begin{array} { l } { { \displaystyle { \cal H } ( { \bf R } ) = E _ { \mathrm { c } } ( { \bf R } ) + \sum _ { p q } h _ { p q } ( { \bf R } ) a _ { p } ^ { \dagger } a _ { q } } \ ~ } \\ { { \displaystyle ~ + \frac { 1 } { 2 } \sum _ { p q r s } v _ { p q r s } ( { \bf R } ) a _ { p } ^ { \dagger } a _ { q } ^ { \dagger } a _ { s } a _ { r } } , } \end{array}\tag{1}
$$

where $p , q , r ,$ s label spin orbitals, $h _ { p q }$ and $v _ { p q r s } = \langle p q | r s \rangle$ are the one- and two-electron integrals in physicists’ notation and $E _ { \mathrm { c } }$ collects nuclear repulsion and frozen-core terms. A state is a coeficient function $\psi ( \mathbf { n } ; \mathbf { R } )$ over occupation strings $\mathbf { n } = \left( n _ { 1 } , \ldots , n _ { M } \right)$ with $n _ { j } \in \{ 0 , \alpha , \beta , \alpha \beta \}$ . The task is to build a shared model for the ground-state wavefunction of a family $\{ H ( \mathbf { R } ) \}$ along a molecular path.

Overview. Fig. 1 summarizes the construction. We first align orbital bases across geometries (Section 3.1), then condition an autoregressive wavefunction on geometry and aligned orbital descriptors (Section 3.2). Joint variational optimization at sparse anchors learns shared parameters, which remain frozen when evaluating energies and electronic observables at new geometries (Section 3.3).

## 3.1 Orbital alignment across geometries

The choice of orbital basis afects the expressivity of a restricted variational ansatz [26]. The occupation basis is built from RHF orbitals. In this basis, molecular wavefunctions often concentrate much of their probability mass on a relatively small subset of configurations [27]. Batched autoregressive sampling (BAS) exploits this concentration by aggregating repeated configurations [4].

Sharing coeficients across geometries requires a consistent convention for these orbitals. Sorting orbitals by energy at each geometry does not ensure this consistency: independent RHF calculations can return orbitals with arbitrary signs, and energy crossings can change their order. A sign flip of spin-orbital p multiplies $\psi ( \mathbf { n } ; \mathbf { R } ) \ \mathrm { b y \ - 1 }$ if p is occupied, and an orbital permutation relabels configurations and transforms their coeficients with the corresponding fermionic parity. These changes in the orbital basis can make the coeficients discontinuous even when the physical state varies smoothly, complicating learning with a continuous geometry-conditioned model.

![](images/1c5c64ee805565f6a2697fb3bc670a3594722bb8677498766cf89231b682977e.jpg)

b Geometr<sub>y</sub>-cond itioned autore<sub>g</sub> ressive state  
![](images/0b1a6ee66d4206b8f3ffd42e04d78b84c530f5f8f946c8f361f3f954d599b440.jpg)  
Figure 1: Geometry-conditioned FNQSs. (a) Starting from sparse anchor geometries, restricted Hartree–Fock (RHF) orbitals are matched and phase-aligned to construct consistent occupation bases and molecular Hamiltonians. Joint variational Monte Carlo (VMC) training produces a shared wavefunction model. At new geometries, the frozen model yields energies and the one-particle reduced density matrix (1-RDM). (b) Occupation prefixes pass through causal Transformer blocks, with geometry and orbital descriptors supplied through feature-wise linear modulation (FiLM), to produce normalized occupation probabilities and hence the wavefunction amplitude. A separate multilayer perceptron (MLP) maps the full occupation string and geometry to a phase. Amplitude and phase combine into the wavefunction.

Learning from eigenvectors requires accounting for their nonunique representation: SignNet and BasisNet address sign and degenerate-basis ambiguities in spectral graph learning [28], while transferable neural orbitals have incorporated orbital-sign equivariance [18]. In an occupation-basis model, orbital transformations act on the wavefunction coeficients themselves. We handle this dependence by aligning orbital identities and phases before sharing the autoregressive model across geometries.

For orbital coeficient matrices $C _ { A } , C _ { B }$ at neighboring geometries, the overlap is $O _ { A B } \ =$ $C _ { A } ^ { \dagger } S _ { A B } C _ { B }$ , where $S _ { A B }$ contains the overlaps between their atomic-orbital (AO) bases. Within compatible occupied/virtual, frozen-core, and symmetry classes, maximum-overlap assignment [29] selects the permutation $\begin{array} { r } { \pi ^ { * } = \arg \operatorname* { m a x } _ { \pi } \sum _ { j } \left| ( O _ { A B } ) _ { j , \pi ( j ) } \right| } \end{array}$ . The target orbitals are reordered by $\pi ^ { * }$ and phase-aligned to make matched overlaps real and nonnegative. Where exact degeneracies leave symmetry labels ambiguous, orbitals are first resolved with respect to a fixed physical symmetry operation, as in the $\mathrm { N H _ { 3 } }$ construction detailed in Appendix A. One- and two-electron integrals are transformed into the aligned orbital basis, in which orbital descriptors are also evaluated.

## 3.2 Geometry-conditioned autoregressive wavefunctions

Using aligned orbitals with coeficient matrix C(R), the model represents the state as

$$
\begin{array} { l } { { \displaystyle { \big \vert \Psi _ { \theta } \mathbf { ( R ) } \big \rangle } = \sum _ { \mathbf { n } \in \Omega } \psi _ { \theta } \mathbf { ( n ; R ) } \vert \mathbf { n ; } C \mathbf { ( R ) } \rangle } , \ ~ } \\  { \displaystyle { \big \vert \Psi _ { \theta } \mathbf { ( n ; R ) } - e ^ { i \phi _ { \theta } \mathbf { ( n ; R ) } } \prod _ { j = 1 } ^ { M } \sqrt { p _ { \theta } ( n _ { j } \mid \mathbf { n } _ { < j } ; \mathbf { R } ) } . } } \end{array}\tag{2}
$$

Here Ω contains occupation strings with the prescribed spin populations and, when imposed, total spatial irreducible representation (irrep), and $\mathbf { n } _ { < j }$ denotes the occupations preceding orbital $j$ . Before each conditional is normalized, a mask excludes choices that cannot be completed to a configuration in Ω, using an exact backward reachability table when spatial symmetry is imposed. The resulting conditionals define a normalized state and permit direct autoregressive sampling [4, 21, 30, 31]. The amplitude network uses L causal Transformer blocks, each combining attention and a feed-forward network (FFN), to predict each occupation from its prefix [8, 32]. The shifted token and position embeddings in Fig. 1(b) are $h _ { j } ^ { ( 0 ) } = e _ { \mathrm { t o k } } ( n _ { j - 1 } ) + e _ { \mathrm { p o s } } ( j )$ , where $n _ { 0 }$ is the beginning-of-sequence (BOS) token.

For a fixed molecule, basis set, and electronic sector, geometry parameterizes the Hamiltonian family along the chosen path. We therefore use geometry as a compact global condition, supplemented by aligned orbital descriptors that provide orbital-specific electronic information at each position in the occupation sequence. Geometry enters through a global condition ${ \bf c } ( { \bf R } )$ , such as a bond length or a normalized path coordinate, and a vector d(R) with one orbital descriptor per aligned orbital. Each component is the aligned Fock diagonal relative to the mean over active occupied orbitals, scaled by one hartree (Ha): $d _ { j } ( \mathbf { R } ) = \big ( F _ { j j } ^ { \mathrm { a l i g n e d } } ( \mathbf { R } ) - \overline { { F } } _ { \mathrm { o c c } } ( \mathbf { R } ) \big ) / ( 1 \mathrm { H a } )$ , where F<sup>aligned</sup> is the Fock matrix in the aligned orbital basis and $\overline { { F } } _ { \mathrm { o c c } }$ is that mean. Under a signed permutation, its diagonal entries equal the canonical orbital energies.

After each Transformer block, a FiLM adapter [33] modulates the hidden features using both inputs. For block output $u ^ { ( \ell ) } = \mathrm { T r a n s f o r m e r } _ { \ell } \big ( \hat { h } ^ { ( \ell - 1 ) } \big )$ , the FiLM update is

$$
\begin{array} { r } { h _ { j } ^ { ( \ell ) } = \big ( 1 + \gamma _ { \ell } ^ { c } ( \mathbf { c } ) + \gamma _ { \ell } ^ { d } ( d _ { j } ) \big ) \odot u _ { j } ^ { ( \ell ) } + \beta _ { \ell } ^ { c } ( \mathbf { c } ) + \beta _ { \ell } ^ { d } ( d _ { j } ) . } \end{array}\tag{3}
$$

The $\gamma$ and $\beta$ maps produce feature-wise scales and shifts, and ⊙ denotes element-wise multiplication. The geometry maps are afine, while the descriptor maps are linear. All four maps are initialized to zero, so each adapter initially acts as the identity. After the final block, layer normalization (LayerNorm) [34] and a linear projection produce logits for the masked softmax, yielding $p _ { \theta } ( n _ { j } \mid$ $\mathbf { n } _ { < j } ; \mathbf { R } )$ (Fig. 1(b)). Implementation and optimizer settings are given in Appendix B.

A separate phase MLP with rectified linear unit (ReLU) activations takes the concatenation of c(R) and the full occupation string encoded as $\mathbf { s } ( \mathbf { n } ) \in \{ - 1 , 1 \} ^ { 2 M }$ , with $s _ { p } = 2 b _ { p } - 1$ for spinorbital occupation $b _ { p } \in \{ 0 , 1 \}$ . It predicts the angle $\phi _ { \theta } ( \mathbf { n } ; \mathbf { R } )$ , whose factor $e ^ { i \phi _ { \theta } }$ combines with the autoregressive amplitude in Equation (2). Orbital descriptors enter only the amplitude branch.

## 3.3 Joint variational training and frozen inference

The model is trained using a first-principles VMC method. The geometry-dependent molecular Hamiltonians supply the variational objective directly, without labeled energies or properties. At each optimization update, BAS propagates counts through successive occupation prefixes, drawing configurations at each anchor geometry from the corresponding wavefunction distribution [4]. A joint energy estimator constructed from these samples [27, 35] is minimized. Let $S _ { k }$ contain the distinct configurations sampled at anchor k, $\psi _ { S _ { k } }$ their coeficient vector, and $H _ { S _ { k } S _ { k } }$ the corresponding Hamiltonian submatrix. The joint objective is

$$
\begin{array} { r l r } {  { E _ { S _ { k } } ( \theta ; { \bf R } _ { k } ) = \frac { \psi _ { S _ { k } } ^ { \dagger } H _ { S _ { k } S _ { k } } ( { \bf R } _ { k } ) \psi _ { S _ { k } } } { \psi _ { S _ { k } } ^ { \dagger } \psi _ { S _ { k } } } , } } \\ & { } & { \mathcal { L } _ { S } ( \theta ) = \sum _ { k } w _ { k } E _ { S _ { k } } ( \theta ; { \bf R } _ { k } ) , ~ } \end{array}\tag{4}
$$

where $w _ { k }$ are the prescribed anchor weights. All anchors contribute to every joint update, extending the shared-training infrastructure of Wu et al. [9]. Appendix B gives the explicit gradient and sampling implementation. The energy of the complete network state is evaluated separately using the frozen-query procedure below.

Frozen queries and electronic observables. At a query geometry ${ \mathbf { R } } _ { q }$ , orbital preparation supplies the aligned Hamiltonian and the geometry and orbital inputs $\left( \mathrm { F i g . \ 1 ( a ) } \right)$ . The network parameters remain frozen while fresh configurations are drawn from $| \psi _ { \theta } ( \mathbf { n } ; \mathbf { R } _ { q } ) | ^ { 2 }$ . For each sampled configuration, the local energy includes all determinants connected by the Hamiltonian:

$$
\begin{array} { r } { E _ { \mathrm { l o c } , \theta } ( { \bf n } ; { \bf R } ) = \displaystyle \sum _ { { \bf n } ^ { \prime } } H _ { { \bf n } { \bf n } ^ { \prime } } ( { \bf R } ) \frac { \psi _ { \theta } ( { \bf n } ^ { \prime } ; { \bf R } ) } { \psi _ { \theta } ( { \bf n } ; { \bf R } ) } , } \\ { E _ { \theta } ( { \bf R } ) = \mathbb { E } _ { | \psi _ { \theta } | ^ { 2 } } \mathrm { R e } E _ { \mathrm { l o c } , \theta } . \quad \quad } \end{array}\tag{5}
$$

The expectation is taken over the normalized network distribution at the query geometry. It is a variational upper bound to the ground-state energy in the chosen orbital space and sector; its Monte Carlo estimate carries sampling uncertainty, which is reported.

Replacing H by another operator gives its expectation in the same wavefunction. This yields the spin-summed 1-RDM, $\begin{array} { r } { \gamma _ { i j } = \sum _ { \sigma } \langle a _ { i \sigma } ^ { \dagger } a _ { j \sigma } \rangle } \end{array}$ , where $i , j$ label spatial orbitals. Its eigenvalues are the natural occupations. Molecular dipoles are obtained from the density matrix, including the frozen-core and nuclear contributions.

## 4 Experiments

Systems and evaluation. The energy benchmarks comprise $\mathrm { N _ { 2 } }$ and CO bond scans in STO-3G [36] and $\mathrm { H } _ { 4 }$ rectangular deformation in cc-pVDZ [37]. Property benchmarks cover $\mathrm { N H _ { 3 } }$ inversion in 6-31G [38], together with $\mathrm { { C O } _ { 2 } }$ symmetric stretching and $\mathrm { C _ { 2 } H _ { 4 } }$ torsion in STO-3G. For each system, a model is trained jointly at sparse anchors and evaluated with frozen parameters at intervening query geometries. Appendix C specifies the paths, geometry splits, and retained electronic spaces; Appendix B gives the network and optimization settings.

Full configuration interaction (FCI) calculations provide the reference values in the same basis and frozen-core space, following the validation protocol for second-quantized neural states [3, 4, 8]. For an evaluation set $Q ,$ , mean absolute error (MAE) is $\begin{array} { r } { \vert Q \vert ^ { - 1 } \sum _ { q \in Q } \vert E _ { \theta } ( \mathbf { R } _ { q } ) - E _ { \mathrm { F C I } } ( \mathbf { R } _ { q } ) \vert } \end{array}$ ; we also report the maximum absolute error over $Q$ and use 1.6 mHa as the chemical-accuracy threshold. Anchor and query errors are kept separate. The main energy benchmarks report statistics over all non-anchor queries. Neural energies use Equation (5), evaluated using three independent sampling replicates per geometry. Sampling error bars give one standard error of the mean. Property errors use the corresponding FCI observables at the same queries.

![](images/6e63ae4aea49f4c6c7c61cc5ec81e4b2370ad69d8758d02e68e228e3c2e25eec.jpg)

![](images/1dbadc692b0aed802346f518de6833a3021a0eac05d1d645cd3e6cd863decf3c.jpg)

![](images/9ffb151a71fb1e8837785179ee2d2eb321ca1a488cffdfe76ac23ce9969285bc.jpg)  
Figure 2: Potential energy surfaces. (a) $\mathrm { N } _ { 2 } ,$ , (b) CO, and (c) $\mathrm { H } _ { 4 } .$ . Each panel combines NQS, RHF, and coupled-cluster singles and doubles (CCSD) relative energies (upper) and NQS signed errors against FCI (lower), with the 1.6 mHa band below. Energies are relative to each method’s path origin. CCSD uses a restricted reference. Squares mark anchors, dots queries, and bars one sampling standard error.

## 4.1 Frozen energy predictions across molecular paths

Sparse-anchor training yields accurate energy profiles at untrained geometries. In Fig. 2, a single frozen model per system resolves the energy profiles of $\mathrm { N _ { 2 } }$ , CO, and $\mathrm { H } _ { 4 } .$ with every anchor and query within 1.6 mHa of FCI. $\mathrm { N _ { 2 } }$ and CO query MAEs of 0.049 and 0.253 mHa show that anchor accuracy carries across both homonuclear and heteronuclear bond scans without further optimization.

$\mathrm { H } _ { 4 }$ tests transfer on a finer energy scale: its rise toward the square geometry spans only a few mHa. The model resolves this shallow multireference profile in the larger cc-pVDZ orbital space with a maximum query error of 0.120 mHa. Thus, the shared wavefunction captures both broad bond-stretching curves and small energy diferences along deformation. Appendix D.1 gives the full anchor and query statistics.

## 4.2 Orbital alignment enables accurate transfer

Fig. 3 compares the errors with unaligned and aligned orbitals along the $\mathrm { N _ { 2 } }$ scan across three paired training seeds. Within each pair, only the orbital basis changes; the architecture, optimizer, and sampling budget are identical. The aligned models predict every untrained geometry within chemical accuracy (query MAE 0.049–0.085 mHa across seeds). The unaligned models fit most anchors but fail between them, with query MAEs of 34–37 mHa and errors above 100 mHa adjacent to anchors; some unaligned runs also miss chemical accuracy at an anchor. The largest failures occur where canonical orbital ordering changes (upper plot in Fig. 3). Alignment removes these artificial discontinuities in the wavefunction coeficients, allowing accuracy at the anchors to extend to the intervening geometries.

Orbital 4 Orbitals 5, 6 Orbital 7  
![](images/2af1f01ea8a412213def6015a408cf14ade0558c195b379b8e1995a6f8966715.jpg)

![](images/7566b51ce722308b98c3aee9fc1ff705db662ba0d2a70940b534c2705d91e4fd.jpg)  
Figure 3: Orbital alignment. Alignment enables transfer between $\mathrm { N _ { 2 } }$ training anchors. The upper plot tracks aligned RHF orbital energies; the degenerate pair $5 / 6$ is drawn as one curve. Arrows label changes in the canonical indices of the indicated continuous branches. Both plots share the N–N distance axis; vertical bands mark the two orbital-ordering exchanges (0.92–0.94 and 1.18–1.20 Å). The lower plot compares absolute energy errors over three paired seeds: lines show means and colored ribbons the minimum–maximum range. Squares mark anchors, dots queries, and the dotted line 1.6 mHa. Architecture, optimization, and sampling budgets are matched within each pair.

Within the aligned orbital basis, conditioning ablations show smaller gains from FiLM and orbital descriptors (Appendix D.3).

## 4.3 Electronic observables from the shared wavefunction

The model supplies a complete electronic wavefunction, giving access to observables from the same state used to evaluate energy. Fig. 4 probes charge displacement, spatial charge distribution, and frontier occupations. Each observable is computed from the one-particle reduced density matrix of the frozen, energy-trained state, without property labels or additional fitting.

The $\mathrm { N H _ { 3 } }$ dipole follows charge redistribution during inversion with query MAE 0.0058 debye (D), comparable to CCSD. Symmetric $\mathrm { C O _ { 2 } }$ has no permanent dipole; its quadrupole instead reveals the changing charge distribution as the bonds lengthen. The model follows this variation with query MAE $0 . 0 4 0 3 ~ e a _ { 0 } ^ { 2 } .$ , below the CCSD error. Ethylene probes the growing multireference character during torsion: its frontier occupations approach one another toward the twisted geometry, a change that a restricted single determinant cannot describe. Their query MAE, averaged over both displayed occupations, is 0.0009, also below the CCSD error. These complementary observables show that the shared state retains useful electronic-structure information beyond its energy. The lower panels resolve signed errors; Appendix D.2 reports energies from the same checkpoints and RHF/CCSD comparisons, and Appendix C defines the observables.

![](images/e4d03d44679e9a8d125761f88f2c74728ca2cfcaec528a81a6a1f610b66fed43.jpg)

![](images/1d4d20f65cd3cc35d8e4e6558f397502f782cbf7e1f8bc8640f619f499d9de89.jpg)

![](images/f01b815cb720c49fa8dc3cd242a32c1bc31337ce3a346c8de289b3ea4daa1231.jpg)  
Figure 4: Electronic properties. (a) $\mathrm { N H _ { 3 } }$ dipole, (b) $\mathrm { C O _ { 2 } }$ quadrupole, and (c) ethylene natural occupations $6 / 7$ Each panel combines the observable (upper) and its NQS-minus-FCI error (lower). Upper curves show NQS estimates connected by solid lines, RHF (dashed), and CCSD (dash-dotted). Circles mark queries, squares anchors, and bars one sampling standard error.

## 4.4 Computational eficiency

Sharing a wavefunction model reduces the work required at each new geometry. We quantify this benefit on the $\mathrm { N _ { 2 } }$ bond scan by comparing four workflows: independent optimization at every geometry, shared pretraining followed by per-geometry fine-tuning [9], sequential warm-start optimization from the preceding geometry, and joint training of our geometry-conditioned FNQS followed by frozen evaluation. The shared-pretraining baseline uses the same NQS backbone and aligned orbitals but omits geometry conditioning. It minimizes the average loss over the training anchors before its weights are fine-tuned independently at each query.

The cost comparison uses separate fixed-budget checkpoints from the main accuracy benchmarks, targeting an MAE of approximately 1 mHa; Appendix D.4 reports the budgets, measured errors, and timing details. Following PESNet and PlaNet [13, 14], we include both initial training and subsequent queries in the total A100 GPU-hour cost. For each workflow m, the estimated cost of G geometries takes the linear form

$$
\widehat { C } _ { m } ( G ) = a _ { m } + G b _ { m } .\tag{6}
$$

The intercept $a _ { m }$ represents the fixed cost, and the slope $b _ { m }$ the average cost of each additional geometry. Independent optimization has zero intercept: each geometry requires training from scratch and an energy estimate. Shared pretraining contributes a one-time cost, followed by fine-tuning and an energy estimate at each geometry. Sequential warm starts require training from scratch at the first geometry, followed by fine-tuning and an energy estimate at each subsequent geometry. Our geometry-conditioned FNQS incurs a one-time joint-training cost, after which each geometry requires only an energy estimate with frozen parameters (Fig. 5).

![](images/b9b215a6f67ddebd3b678ce9711108b349b76c4c641e22c5c007f182630ef87c.jpg)

![](images/c4097fdc9698dee1bbe5e7263ce276c16e85de614477710061816565d779cea4.jpg)  
Figure 5: Computational cost. The $\mathrm { N _ { 2 } }$ workflows use fixed training budgets with MAEs near 1 mHa. (a) Average cost per additional geometry, including optimization where applicable and energy evaluation. The warm-start range spans the measured means at 41- and 161-point spacings. (b) Estimated total costs, including initial training, on logarithmic axes. Dots and vertical guides mark frozen evaluation on the 41- and 161-point grids. Independent and shared-adaptation costs are estimated from eight geometries selected by stratified sampling; warm-start dashed and solid curves use the 41- and 161-point spacings, respectively. Each curve holds its measured average per-geometry cost fixed as the query count varies.

Frozen evaluation reduces the cost of each additional geometry by approximately 986× relative to independent optimization, 77× relative to shared pretraining with fine-tuning, and 41–43× relative to sequential warm starts. On the evaluated 161-point $\mathrm { N _ { 2 } }$ grid, the total cost is 1.644 GPU-hours, giving estimated end-to-end GPU-cost reductions of 25.8×, 2.53×, and 1.18×, respectively. Denser queries spread the initial training investment over more geometries, bringing the overall speedup closer to the per-geometry ratio. Sequential warm starts impose dependencies along the path; the other workflows permit parallel computation across geometries, including multi-geometry training for the shared and conditioned models. Appendix D.4 reports the budgets, measured MAEs, and cost breakdowns.

## 5 Discussion and conclusion

We have established geometry-conditioned FNQSs for molecular electronic structure in second quantization. Joint variational training turns sparse anchor calculations into a shared representation of molecular ground states, supplying energies and electronic observables at untrained geometries with frozen parameters. Orbital alignment establishes an aligned orbital basis across geometries, enabling the shared network to learn the physical variation along a molecular path.

The energy-trained states recover charge redistribution and changes in natural occupations without property-specific fitting. At fixed training budgets, denser queries amortize joint training, while per-geometry optimization and evaluation costs determine the asymptotic speedup. This work establishes an eficient route to computing molecular potential energy surfaces with neural-network quantum states in an occupation basis.

Limitations and outlook. Our benchmarks cover one-dimensional molecular paths with FCI references in specified finite orbital spaces; basis-set errors remain. Extending the method to larger active spaces will require eficient sampling and Hamiltonian evaluation as the number of relevant configurations grows. In multidimensional geometry domains, orbital matching can depend on the transport path, making consistency around closed loops an additional challenge near orbital crossings. A further direction is to extend orbital correspondence across diferent active spaces and, ultimately, across molecules. Together with architectures that accommodate varying orbital spaces and electron counts, such correspondences could support pretraining and wavefunction reuse across a broader range of chemical systems.

## Reproducibility statement

Appendices A, B, and C specify orbital alignment and query attachment, network and sampling settings, and molecular paths and electronic sectors, respectively.

## AI use statement

The authors used ChatGPT and Cursor (with Claude Fable 5.1 and Grok 4.7) to assist with methodological development, experimental design, method implementation and debugging, numerical data processing, and interpretation of results. The tools also assisted with literature search, manuscript editing, and figure preparation. The authors reviewed all AI-assisted work, including the text, code, experimental design, results, and references. The authors take responsibility for the final content of this submission.

## References

[1] Giuseppe Carleo and Matthias Troyer. Solving the quantum many-body problem with artificial neural networks. Science, 355:602–606, 2017. doi: 10.1126/science.aag2302. URL https: //arxiv.org/abs/1606.02318.

[2] Jan Hermann, James Spencer, Kenny Choo, Antonio Mezzacapo, W. M. C. Foulkes, David Pfau, Giuseppe Carleo, and Frank Noé. Ab initio quantum chemistry with neural-network wavefunctions. Nature Reviews Chemistry, 7:692–709, 2023. doi: 10.1038/s41570-023-00516-8. URL https://arxiv.org/abs/2208.12590.

[3] Kenny Choo, Antonio Mezzacapo, and Giuseppe Carleo. Fermionic neural-network states for ab-initio electronic structure. Nature Communications, 11:2368, 2020. doi: 10.1038/ s41467-020-15724-9. URL https://www.nature.com/articles/s41467-020-15724-9.

[4] Thomas D. Barrett, Aleksei Malyshev, and A. I. Lvovsky. Autoregressive neural-network wavefunctions for ab initio quantum chemistry. Nature Machine Intelligence, 4:351–358, 2022. doi: 10. 1038/s42256-022-00461-z. URL https://www.nature.com/articles/s42256-022-00461-z.

[5] Tianchen Zhao, James Stokes, and Shravan Veerapaneni. Scalable neural quantum states architecture for quantum chemistry. Machine Learning: Science and Technology, 4:025034, 2023. doi: 10.1088/2632-2153/acdb2f. URL https://arxiv.org/abs/2208.05637.

[6] An-Jun Liu and Bryan K. Clark. Neural network backflow for ab initio quantum chemistry. Physical Review B, 110:115137, 2024. doi: 10.1103/PhysRevB.110.115137. URL https: //arxiv.org/abs/2403.03286.

[7] Bowen Kan, Yingqi Tian, Yangjun Wu, Yunquan Zhang, and Honghui Shang. Bridging the gap between transformer-based neural networks and tensor networks for quantum chemistry. Journal of Chemical Theory and Computation, 21(7):3426–3439, 2025. doi: 10.1021/acs.jctc.4c01703. URL https://pubs.acs.org/doi/10.1021/acs.jctc.4c01703.

[8] Honghui Shang, Chu Guo, Yangjun Wu, Zhenyu Li, and Jinlong Yang. Solving the manyelectron Schrödinger equation with a transformer-based framework. Nature Communications, 16:8464, 2025. doi: 10.1038/s41467-025-63219-2. URL https://www.nature.com/articles/ s41467-025-63219-2.

[9] Yangjun Wu, Wanlu Cao, Jiacheng Zhao, and Honghui Shang. Fast and scalable neural network quantum states method for molecular potential energy surfaces. IEEE Transactions on Parallel and Distributed Systems, 36(7):1431–1443, 2025. doi: 10.1109/TPDS.2025.3568360. URL https://ieeexplore.ieee.org/document/11000098.

[10] Riccardo Rende, Luciano Loris Viteritti, Federico Becca, Antonello Scardicchio, Alessandro Laio, and Giuseppe Carleo. Foundation neural-networks quantum states as a unified ansatz for multiple Hamiltonians. Nature Communications, 16:7213, 2025. doi: 10.1038/s41467-025-62098-x. URL https://arxiv.org/abs/2502.09488.

[11] Moritz Bensberg and Markus Reiher. Corresponding active orbital spaces along chemical reaction paths. The Journal of Physical Chemistry Letters, 14:2112–2118, 2023. doi: 10.1021/ acs.jpclett.2c03905. URL https://arxiv.org/abs/2212.12883.

[12] Yannic Rath and George H. Booth. Interpolating numerically exact many-body wave functions for accelerated molecular dynamics. Nature Communications, 16:2005, 2025. doi: 10.1038/ s41467-025-57134-9. URL https://www.nature.com/articles/s41467-025-57134-9.

[13] Nicholas Gao and Stephan Günnemann. Ab-initio potential energy surfaces by pairing GNNs with neural wave functions. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=apv504XsysP.

[14] Nicholas Gao and Stephan Günnemann. Sampling-free inference for ab-initio potential energy surface networks. In International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/2205.14962.

[15] David Pfau, James S. Spencer, Alexander G. de G. Matthews, and W. M. C. Foulkes. Ab initio solution of the many-electron Schrödinger equation with deep neural networks. Physical Review Research, 2:033429, 2020. doi: 10.1103/PhysRevResearch.2.033429. URL https: //arxiv.org/abs/1909.02487.

[16] Jan Hermann, Zeno Schätzle, and Frank Noé. Deep-neural-network solution of the electronic Schrödinger equation. Nature Chemistry, 12:891–897, 2020. doi: 10.1038/s41557-020-0544-y. URL https://www.nature.com/articles/s41557-020-0544-y.

[17] Ingrid von Glehn, James S. Spencer, and David Pfau. A self-attention ansatz for ab-initio quantum chemistry. In International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/2211.13672.

[18] Michael Scherbela, Leon Gerard, and Philipp Grohs. Towards a transferable fermionic neural wavefunction for molecules. Nature Communications, 15:120, 2024. doi: 10.1038/ s41467-023-44216-9. URL https://www.nature.com/articles/s41467-023-44216-9.

[19] Nicholas Gao and Stephan Günnemann. Generalizing neural wave functions. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of PMLR, pages 10708–10726, 2023. URL https://proceedings.mlr.press/v202/gao23c.html.

[20] Adam Foster, Zeno Schätzle, P. Bernát Szabó, Lixue Cheng, Jonas Köhler, Gino Cassella, Nicholas Gao, Jiawei Li, Frank Noé, and Jan Hermann. An ab initio foundation model of wavefunctions that accurately describes chemical bond breaking. Nature Communications, 17:9976, 2026. doi: 10.1038/s41467-026-76604-2. URL https://www.nature.com/articles/ s41467-026-76604-2.

[21] Aleksei Malyshev, Juan Miguel Arrazola, and A. I. Lvovsky. Autoregressive neural quantum states with quantum number symmetries. arXiv:2310.04166, 2023. URL https://arxiv.org/ abs/2310.04166.

[22] Dillon Frame, Rongzheng He, Ilse Ipsen, Daniel Lee, Dean Lee, and Ermal Rrapaj. Eigenvector continuation with subspace learning. Physical Review Letters, 121:032501, 2018. doi: 10.1103/ PhysRevLett.121.032501. URL https://arxiv.org/abs/1711.07090.

[23] Carlos Mejuto-Zaera and Alexander F. Kemper. Quantum eigenvector continuation for chemistry applications. Electronic Structure, 5:045007, 2023. doi: 10.1088/2516-1075/ad018f. URL https://iopscience.iop.org/article/10.1088/2516-1075/ad018f.

[24] Ethan N. Epperly, Lin Lin, and Yuji Nakatsukasa. A theory of quantum subspace diagonalization. SIAM Journal on Matrix Analysis and Applications, 43(3):1263–1290, 2022. doi: 10.1137/ 21M145954X. URL https://arxiv.org/abs/2110.07492.

[25] Caleb Hicks and Dean Lee. Trimmed sampling algorithm for the noisy generalized eigenvalue problem. Physical Review Research, 5:L022001, 2023. doi: 10.1103/PhysRevResearch.5.L022001. URL https://arxiv.org/abs/2209.02083.

[26] Javier Robledo Moreno, Jefrey Cohn, Dries Sels, and Mario Motta. Enhancing the expressivity of variational neural, and hardware-eficient quantum states through orbital rotations. arXiv:2302.11588, 2023. URL https://arxiv.org/abs/2302.11588.

[27] Aleksei Malyshev, Markus Schmitt, and A. I. Lvovsky. Neural quantum states and peaked molecular wave functions: Curse or blessing? arXiv:2408.07625, 2024. URL https://arxiv. org/abs/2408.07625.

[28] Derek Lim, Joshua Robinson, Lingxiao Zhao, Tess Smidt, Suvrit Sra, Haggai Maron, and Stefanie Jegelka. Sign and basis invariant networks for spectral graph representation learning. In International Conference on Learning Representations, 2023. URL https://arxiv.org/ abs/2202.13013.

[29] Harold W. Kuhn. The Hungarian method for the assignment problem. Naval Research Logistics Quarterly, 2:83–97, 1955. doi: 10.1002/nav.3800020109.

[30] Or Sharir, Yoav Levine, Noam Wies, Giuseppe Carleo, and Amnon Shashua. Deep autoregressive models for the eficient variational simulation of many-body quantum systems. Physical Review Letters, 124:020503, 2020. doi: 10.1103/PhysRevLett.124.020503. URL https://arxiv.org/ abs/1902.04057.

[31] Mohamed Hibat-Allah, Martin Ganahl, Lauren E. Hayward, Roger G. Melko, and Juan Carrasquilla. Recurrent neural network wave functions. Physical Review Research, 2:023358, 2020. doi: 10.1103/PhysRevResearch.2.023358. URL https://arxiv.org/abs/2002.02973.

[32] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017. URL https://papers.neurips.cc/paper\_ files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html.

[33] Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron Courville. FiLM: Visual reasoning with a general conditioning layer. In Proceedings of the AAAI Conference on Artificial Intelligence, 2018. URL https://arxiv.org/abs/1709.07871.

[34] Jimmy Lei Ba, Jamie Ryan Kiros, and Geofrey E. Hinton. Layer normalization. arXiv:1607.06450, 2016. URL https://arxiv.org/abs/1607.06450.

[35] Yangjun Wu, Chu Guo, Yi Fan, Pengyu Zhou, and Honghui Shang. NNQS-Transformer: an eficient and scalable neural network quantum states approach for ab initio quantum chemistry. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–13, 2023. doi: 10.1145/3581784.3607061. URL https://arxiv.org/abs/2306.16705.

[36] Warren J. Hehre, Robert F. Stewart, and John A. Pople. Self-consistent molecular-orbital methods. i. use of gaussian expansions of slater-type atomic orbitals. The Journal of Chemical Physics, 51:2657–2664, 1969. doi: 10.1063/1.1672392.

[37] Thom H. Dunning, Jr. Gaussian basis sets for use in correlated molecular calculations. i. the atoms boron through neon and hydrogen. The Journal of Chemical Physics, 90:1007–1023, 1989. doi: 10.1063/1.456153.

[38] Warren J. Hehre, Robert Ditchfield, and John A. Pople. Self-consistent molecular orbital methods. xii. further extensions of gaussian-type basis sets for use in molecular orbital studies of organic molecules. The Journal of Chemical Physics, 56:2257–2261, 1972. doi: 10.1063/1. 1677527.

[39] Qiming Sun, Xing Zhang, Samragni Banerjee, Peng Bao, Marc Barbry, et al. Recent developments in the PySCF program package. The Journal of Chemical Physics, 153:024109, 2020. doi: 10.1063/5.0006074. URL https://arxiv.org/abs/2002.12531.

[40] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems, volume 32, 2019. URL https://arxiv.org/abs/1912.01703.

[41] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://arxiv.org/abs/1711.05101.

[42] George D. Purvis, III and Rodney J. Bartlett. A full coupled-cluster singles and doubles model: The inclusion of disconnected triples. The Journal of Chemical Physics, 76:1910–1918, 1982. doi: 10.1063/1.443164.

## SUMMARY OF THE APPENDIX

The appendix is organized as follows:

• Appendix A describes orbital transformations, alignment protocols, and query attachment.

• Appendix B specifies the network, optimizer, sampling, and energy-evaluation settings.

• Appendix C defines the molecular paths, retained orbital spaces, and electronic observables.

• Appendix D reports detailed energy and property results, per-seed ablations, and computational costs.

## A Orbital alignment and query attachment

## A.1 Orbital transformations

At fixed geometry, $\begin{array} { r } { \widetilde { a } _ { p } ^ { \dagger } = \sum _ { q } a _ { q } ^ { \dagger } U _ { q p } } \end{array}$ induces the exterior-power representation of U on determinants. In a basis with separately ordered alpha and beta determinants, its matrix element between occupied sets is

$$
\mathcal { U } _ { I , J } = \operatorname* { d e t } U _ { I _ { \alpha } , J _ { \alpha } } \operatorname* { d e t } U _ { I _ { \beta } , J _ { \beta } } .\tag{7}
$$

Conversion to interleaved spin-orbital order includes the fermionic ordering signs. Signed permutations therefore relabel determinants with the corresponding parity; general rotations mix them through these minors. For any retained-space unitary, U is unitary and

$$
( \mathcal { U } ^ { \dagger } \psi ) ^ { \dagger } ( \mathcal { U } ^ { \dagger } H \mathcal { U } ) ( \mathcal { U } ^ { \dagger } \psi ) = \psi ^ { \dagger } H \psi .\tag{8}
$$

Rotations are confined to the retained orbital space; frozen-core and excluded orbitals define separate boundaries. Across diferent geometries, the AO functions themselves move, so cross-AO overlaps enter the matching step.

## A.2 Alignment protocols

The $\mathrm { N _ { 2 } / C O }$ construction transports the orbital bases at training anchors from a fixed root, then aligns each query directly to its nearest aligned training anchor. The $\mathrm { H _ { 4 } }$ endpoint RHF solution is continued from the lower canonical branch in the common $\mathrm { C } _ { 2 h }$ subgroup.

Orbital preparation also uses query geometries as additional RHF support.

For $\mathrm { N H _ { 3 } , }$ , orbital preparation resolves parity under a fixed physical reflection, $z \mapsto - z$ in the laboratory frame. The reflection matrix is diagonalized within exactly degenerate Fock blocks (energy tolerance $1 0 ^ { - 7 }$ Ha), keeping occupied/virtual and frozen-core boundaries separate. This step rotates degenerate orbitals into a common parity convention before maximum-overlap matching and sign alignment. The orbital bases at the five anchors are transported from the planar root, and each query is attached to its nearest anchor.

## B Optimization and sampling details

## B.1 Network and optimizer settings

Electronic-structure calculations use $\mathrm { P y S C F }$ [39], and neural-network models are implemented in PyTorch [40].

Production models use four amplitude blocks, four heads, and four phase-network layers of width 512 with rectified linear unit activations. The amplitude width is 32 except for ethylene, which uses width 64. Occupations are decoded in reverse orbital order; descriptor and irrep arrays follow that same order. The phase branch concatenates the geometry condition c(R) with the signed spin-occupation vector $\mathbf { s } ( \mathbf { n } )$ . Diatomics use bond lengths in angstroms as geometry features; the other molecular paths rescale their coordinate to [−1, 1] using fixed domain bounds. Spin populations are fixed in all runs; the $\mathrm { N _ { 2 } }$ model uses no spatial-irrep mask, while the other five systems impose the prescribed total irrep. Dropout is zero. The FiLM maps in Equation (3) have zero initial weights and biases, with no bias in the descriptor map. The output head applies a final LayerNorm followed by the linear projection. AdamW [41] uses a base learning rate 10<sup>−3</sup>, betas (0.9, 0.99), epsilon $1 0 ^ { - 9 }$ , and zero weight decay. Computation uses float32 model parameters.

The displayed models use seed 11 and 20,000 updates. $\mathrm { N _ { 2 } }$ uses constant learning rate $1 0 ^ { - 3 }$ . CO, $\mathrm { H } _ { 4 } , \mathrm { N H } _ { 3 }$ , and ethylene use cosine decay from $1 0 ^ { - 3 }$ to 10<sup>−5</sup> without warmup. $\mathrm { C O _ { 2 } }$ uses $3 \times 1 0 ^ { - 4 }$ through update $6 , 0 0 0 , 1 0 ^ { - 4 }$ through 15,000, and $3 \times 1 0 ^ { - 5 }$ through 20,000. Training uses two A100 GPUs except for ethylene (four A100s) and $\mathrm { N H _ { 3 } } .$ , which uses one V100 through update 15,000 and three A100s thereafter. All final models use FiLM after each of the four Transformer blocks. The orbital-alignment and conditioning ablations use the corresponding $\mathrm { N _ { 2 } / C O }$ schedules and seeds 11, 23, and 37.

Table 1: Training budgets and per-geometry distinct-configuration targets.
<table><tr><td>Study</td><td>Steps</td><td>Geometries/update</td><td>Distinct configurations</td></tr><tr><td> $\mathrm { N _ { 2 } }$ </td><td>20,000</td><td>5</td><td>6,000–50,000</td></tr><tr><td>CO</td><td>20,000</td><td>5</td><td>600-2,400</td></tr><tr><td> $\mathrm { H _ { 4 } / c c { \mathrm { - p V D Z } } }$ </td><td>20,000</td><td>4</td><td>2,000-4,000</td></tr><tr><td> $\mathrm { N H _ { 3 } }$ </td><td>20,000</td><td>5</td><td>6,000-20,000</td></tr><tr><td>Ethylene</td><td>20,000</td><td>7</td><td>6,000-20,000</td></tr><tr><td> $\mathrm { C O _ { 2 } }$ </td><td>20,000</td><td>5</td><td>6,000-20,000</td></tr></table>

## B.2 Sampled-subspace training and full-state evaluation

The native Hamiltonian backend constructs in-set single- and double-excitation connections. Network probabilities, normalized over $S _ { k }$ , weight its energy and gradient; BAS multiplicities determine set membership. Holding $S _ { k }$ fixed during diferentiation gives

$$
\nabla _ { \boldsymbol { \theta } } E _ { S _ { k } } = 2 \operatorname { R e } \sum _ { \boldsymbol { x } \in S _ { k } } \frac { \vert \psi _ { \boldsymbol { \theta } } ( \boldsymbol { x } ) \vert ^ { 2 } } { \sum _ { \boldsymbol { y } \in S _ { k } } \vert \psi _ { \boldsymbol { \theta } } ( \boldsymbol { y } ) \vert ^ { 2 } } \left[ \frac { ( H _ { S _ { k } S _ { k } } \psi _ { S _ { k } } ) _ { \boldsymbol { x } } } { \psi _ { \boldsymbol { \theta } } ( \boldsymbol { x } ) } - E _ { S _ { k } } \right] \nabla _ { \boldsymbol { \theta } } \log \psi _ { \boldsymbol { \theta } } ( \boldsymbol { x } ) ^ { * } .\tag{9}
$$

The sampled set is refreshed between updates. Related sampled-subspace objectives are described by Malyshev et al. [27], Wu et al. [35].

For complete-state reporting, BAS multiplicities $c _ { x }$ retain the frequency of every distinct sampled configuration. The energy estimate is

$$
{ \widehat { E } } = { \frac { \sum _ { x } c _ { x } \operatorname { R e } E _ { \mathrm { l o c } } ( x ) } { \sum _ { x } c _ { x } } } .\tag{10}
$$

The local-energy sum includes connected states outside $S ,$ with their amplitudes evaluated by the same frozen network. Three independent runs at each geometry give sampling replicates. The standard error of the mean is the larger of the within-replicate propagated error, $\sqrt { \textstyle \sum _ { r = 1 } ^ { 3 } \sigma _ { r } ^ { 2 } } / 3$ and the between-replicate sample standard deviation divided by ${ \sqrt { 3 } } .$ Relative-energy uncertainties include the point and reference-origin variances in quadrature.

## C Molecular paths and property conventions

Table 2: Molecular systems. Electron/orbital counts refer to the retained space.
<table><tr><td>System</td><td>Basis</td><td>e−/orbitals</td><td>Coordinate</td><td>Anchors</td></tr><tr><td> $\mathrm { N _ { 2 } }$ </td><td>STO-3G</td><td>14/10</td><td>bond length</td><td>5</td></tr><tr><td>CO</td><td>STO-3G</td><td>14/10</td><td>bond length</td><td>5</td></tr><tr><td> $\mathrm { H _ { 4 } }$  rectangle</td><td> $\mathrm { c c - p V D Z }$ </td><td> $4 / 2 0$ </td><td>rectangle angle</td><td>4</td></tr><tr><td> $\mathrm { N H _ { 3 } }$ </td><td>6-31G, frozen core</td><td>8/14</td><td>umbrella height</td><td>5</td></tr><tr><td> $\mathrm { C O _ { 2 } }$ </td><td>STO-3G, frozen core</td><td>16/12</td><td>C-O distance</td><td>5</td></tr><tr><td> $\mathrm { C _ { 2 } H _ { 4 } }$ </td><td>STO-3G, frozen core</td><td>12/12</td><td>torsion angle</td><td>7</td></tr></table>

Diatomics and $\mathbf { H } _ { 4 }$ . $\mathrm { N _ { 2 } }$ spans 0.60–1.40 Å and CO spans 0.80–1.60 Å, both at 0.02 Å reporting spacing and with five anchors at 0.20 Å spacing. Each grid contains five anchors and 36 non-anchor queries. $\mathrm { H _ { 4 } }$ has coordinates $( \pm x , \pm y , 0 )$ with $x = R \cos ( \theta / 2 ) , y = R \sin ( \theta / 2 )$ , and $R = 3 . 2 8 4 3 a _ { 0 }$ Its 25-point grid spans $8 4 ^ { \circ } - 9 0 ^ { \circ }$ in $0 . 2 5 ^ { \circ }$ steps, with anchors $8 4 ^ { \circ } , 8 6 ^ { \circ } , 8 8 ^ { \circ } , 9 0 ^ { \circ }$ , and 21 non-anchor queries. The $\mathrm { C } _ { 2 h } / A _ { g }$ convention continues the RHF density from $8 9 . 7 5 ^ { \circ }$ to the square endpoint.

$\mathbf { N H } _ { 3 }$ . The coordinate q is the nitrogen height above the $\mathrm { H _ { 3 } }$ plane at fixed N–H distance 1.02 Å. The five anchors are $q = 0 , 0 . 1 5 , 0 . 3 0 , 0 . 4 5 , 0 . 6 0 \mathrm { ~ \AA ~ }$ , within a uniform 25-point reporting grid. A common vertical $\mathrm { C } _ { s }$ reflection is used at every geometry, including the planar endpoint. N 1s is frozen, leaving eight electrons in 14 orbitals. The dipole axis points from the $\mathrm { H _ { 3 } }$ plane toward positive N height.

$\mathbf { C O } _ { 2 }$ . The molecule is linear with carbon at the origin and oxygen atoms at $z \ = \ \pm r$ , with $r \in [ 1 . 0 5 , 1 . 3 5 ] \mathrm { ~ \AA ~ }$ . Five anchors at 0.075 Å spacing define the aligned orbital bases; the 25-point reporting grid has 0.0125 $\mathrm { \AA }$ spacing. Matching preserves $\mathrm { D } _ { 2 h }$ irreps and the core, occupied, and virtual orbital classes. Freezing the C and O 1s cores leaves 16 electrons in 12 STO-3G orbitals. The reported traceless axial quadrupole is $\begin{array} { r } { Q _ { z z } = \frac { 1 } { 2 } \sum _ { a } q _ { a } ( 3 z _ { a } ^ { 2 } - r _ { a } ^ { 2 } ) } \end{array}$ , including electronic, frozen-core, and nuclear contributions, in $e a _ { 0 } ^ { 2 }$

Ethylene. The fixed planar structure has C=C distance 1.339 Å, C–H distance 1.087 ${ \mathrm { \AA } } ,$ and $\mathrm { C - C - }$ H angle 121.3<sup>◦</sup>. Symmetric fragment rotations by $\pm \theta / 2$ define $\theta \in [ 0 , 9 0 ^ { \circ } ]$ . Seven Chebyshev–Lobatto anchors are 0, 6.028857, 22.5, 45, 67.5, 83.971143, 90<sup>◦</sup>. The reporting grid comprises 33 uniformly spaced angles and the two additional anchors at 6.028857<sup>◦</sup> and $8 3 . 9 7 1 1 4 3 ^ { \circ }$ , giving 35 points and 28 queries. All seven anchors share the orbital labels and phase conventions of the initial five-anchor basis. Both carbon 1s cores are frozen, leaving 12 electrons in 12 orbitals in a common $\mathrm { C _ { 2 } }$ subgroup.

The displayed frontier occupations are the sixth and seventh eigenvalues of the active spin-summed 1-RDM, sorted in descending order; their $\pi / \pi ^ { * }$ character is checked against the reference natural orbitals.

Electronic observables. The spin-summed 1-RDM is Hermitized before diagonalization, and its trace equals the retained electron count. Natural occupations are sorted in descending order. Dipoles include electronic, frozen-core, and nuclear contributions with a fixed origin convention. Property MAEs use all non-anchor reporting points; ethylene metrics average occupations 6 and 7. RHF and CCSD [42] use the same geometries, retained spaces, origins, and operators as NQS and FCI. CCSD properties are obtained by contracting the relevant operators with the unrelaxed spin-summed one-particle density matrix from converged amplitude and Λ equations. Occupied and virtual orbitals are separately canonicalized while retaining the RHF determinant. All reporting geometries have converged solutions.

## D Supporting results

## D.1 Frozen energy accuracy

Table 3 reports the complete anchor and query errors behind Fig. 2. Table 4 and Fig. 6 give the energies of the same checkpoints used for the three property panels. Every query in Table 3 is within 1.6 mHa of FCI.

Table 3: Frozen molecular predictions, in mHa relative to FCI. Anchor and query metrics use the same final checkpoint.
<table><tr><td>System</td><td>Molecular path Anchors /</td><td>queries</td><td>Anchor MAE</td><td>Query MAE</td><td>Query max.</td></tr><tr><td> $\mathrm { N _ { 2 } }$ </td><td>bond stretch</td><td>5 / 36</td><td>0.036</td><td>0.049</td><td>0.095</td></tr><tr><td>CO</td><td>bond stretch</td><td>5 / 36</td><td>0.197</td><td>0.253</td><td>0.585</td></tr><tr><td> $\mathrm { H _ { 4 } }$ </td><td>rectangle</td><td>4/ 21</td><td>0.088</td><td>0.076</td><td>0.120</td></tr></table>

## D.2 Molecular-property benchmarks

Table 5 quantifies the property comparisons in Fig. 4. Each metric uses all non-anchor reporting points; occupation errors average over both displayed eigenvalues.

Table 4: Energy errors for the three molecular-property benchmarks in Fig. 4, in mHa relative to FCI. Anchor and query metrics use the same final checkpoints as the property comparisons; queries include all non-anchor grid points.
<table><tr><td>System</td><td>Molecular path</td><td>Anchors / queries</td><td>Anchor MAE</td><td>Query MAE</td><td>Query max.</td></tr><tr><td> $\mathrm { C _ { 2 } H _ { 4 } }$ </td><td>torsion</td><td>7/28</td><td>0.919</td><td>0.946</td><td>1.233</td></tr><tr><td> $\mathrm { N H _ { 3 } }$ </td><td>inversion</td><td>5 / 20</td><td>0.872</td><td>1.036</td><td>1.456</td></tr><tr><td> $\mathrm { C O _ { 2 } }$ </td><td>symmetric stretch</td><td>5 / 20</td><td>0.591</td><td>0.535</td><td>0.895</td></tr></table>

(a)  
![](images/8466ef8d125bb931b9583a3dd3e1f097b52ced6c428e4325ebd02a1fdc9873a4.jpg)

![](images/d3e0f1d148659573f966d55bc2c22ac7b1cb9d3d9e40855aa1ecb0127fe6c718.jpg)  
(b)

![](images/aa36899392b42b3339d98b05c72869de4ed4e2f42087c9ba9066950c5ca2a5d0.jpg)  
(c)

![](images/44200748e520f5f25713d2f6fcc90a941680f8a67f671fc1542d7d1fd2fe34a2.jpg)  
N height q (Å)

![](images/6f84483cc5ee6234b8afa2a4b924b494bc98f217b0946163ea50e4576a621897.jpg)  
C–O distance (Å)

![](images/48b03811240c6c5071ddaf58146e9bab02101b591826c71365d7bbefe011699b.jpg)  
Figure 6: Frozen energy predictions. (a) $\mathrm { N H _ { 3 } }$ inversion, (b) $\mathrm { C O _ { 2 } }$ symmetric stretching, and (c) ethylene torsion. Each panel combines frozen NQS energies relative to each path origin (upper) and the corresponding signed complete-Hamiltonian errors against FCI (lower), with the ±1.6 mHa band. Hollow squares mark anchors; dots mark queries. Error bars show one sampling standard error. The corresponding property panels appear in Fig. 4.

Table 5: Query property MAEs against FCI. All methods use identical geometries, retained spaces, and observable conventions.
<table><tr><td>Observable</td><td>RHF</td><td>CCSD</td><td>NQS</td></tr><tr><td>NH3 dipole (D)</td><td>0.06057</td><td>0.00575</td><td>0.0058</td></tr><tr><td> $\mathrm { C O _ { 2 } }$  quadrupole (ea)</td><td>1.07714</td><td>0.06971</td><td>0.0403</td></tr><tr><td> $\mathrm { C _ { 2 } H _ { 4 } }$  occupations</td><td>0.27589</td><td>0.00485</td><td>0.0009</td></tr></table>

![](images/ee1ecaefd25db83aad30d3bf8b94889574a65fcfca754d047381fc839864c2d6.jpg)  
Figure 7: Conditioning ablations. All variants use the aligned CO orbital basis. Open circles show query MAEs for three seeds and filled diamonds their means. The geometry condition c and orbital descriptors d enter either the input embeddings or post-block FiLM adapters. The backbone, phase network, training budget, and evaluation geometries are matched.

## D.3 Per-seed component ablations

Table 6 gives the individual runs underlying Figs. 3 and 7. Alignment improves $\mathrm { N _ { 2 } }$ query accuracy for every paired seed. All four CO conditioning variants reach chemical accuracy; their diferences are small compared with the orbital-alignment efect. The Transformer and phase-network backbones, sampler settings, and optimization budgets are matched within each comparison, with seeds 11, 23, and 37.

Conditioning architecture in the aligned orbital basis. In the aligned orbital basis, Fig. 7 compares four ways to condition the amplitude. The c input baseline adds an afine embedding of the geometry condition c (the bond length for CO) to the token and position embeddings; c+d input additionally supplies a linear projection of the orbital descriptor $d _ { j }$ . Their FiLM counterparts apply these signals after each Transformer block, as in Equation (3). Final 20,000-update checkpoints are evaluated on eight CO query geometries over three seeds.

FiLM lowers the mean query MAE from 0.427 to 0.303 mHa with geometry alone. Descriptors have a smaller efect at the input, but combining them with FiLM gives the lowest mean MAE, 0.284 mHa. These diferences are comparable to seed variation and smaller than the alignment efect. We use c + d FiLM in the final model.

Table 6: Per-seed component ablations (MAE in mHa). $\mathrm { N _ { 2 } }$ uses all 36 query geometries; CO uses the eight query geometries. All runs use 20,000 updates with post-block FiLM where applicable; $\mathrm { N _ { 2 } }$ uses a fixed learning rate of $\bar { 1 0 ^ { - 3 } }$ and CO uses cosine decay. Each cell reports the error of a fixed final checkpoint.
<table><tr><td>Model</td><td>Seed 11</td><td>Seed 23</td><td>Seed  $3 7$ </td></tr><tr><td> $\mathrm { N _ { 2 } }$  unaligned</td><td>36.806</td><td>33.609</td><td>35.027</td></tr><tr><td> $\mathrm { N _ { 2 } }$  aligned</td><td>0.049</td><td>0.085</td><td>0.053</td></tr><tr><td>CO c input</td><td>0.292</td><td>0.414</td><td>0.575</td></tr><tr><td> $\mathrm { C O } \textbf { c } \mathrm { F i L M }$ </td><td>0.328</td><td>0.333</td><td>0.247</td></tr><tr><td> $\mathrm { C O } { \bf c } + { \bf d }$  input</td><td>0.364</td><td>0.438</td><td>0.443</td></tr><tr><td> $\mathrm { C O \ } \mathbf { c } + \mathbf { d } \ \mathrm { F i L M }$ </td><td>0.277</td><td>0.194</td><td>0.382</td></tr></table>

Table 7: Fixed-budget $\mathrm { N _ { 2 } }$ comparison. MAEs use the common eight geometries. Costs include training and final energy evaluation; independent and shared costs are estimated from the sampled geometries. The two warm-start MAEs correspond to the 41- and 161-point grids, respectively.
<table><tr><td>Workflow</td><td>MAE (mHa)</td><td>41 points (GPU-h)</td><td>161 points (GPU-h)</td></tr><tr><td></td><td>Updates</td><td>1.027 10.785</td><td>42.351</td></tr><tr><td>Independent Shared + fine-tune</td><td>4,000  $2 , 0 0 0 + 2 0 0 / \mathrm { p o i n t }$ </td><td>1.064 1.682</td><td>4.162</td></tr><tr><td>Sequential warm-start</td><td> $3 , 0 0 0 + 1 0 0 / \mathrm { n e x t }$  point 1.173 / 0.988</td><td>0.664</td><td>1.948</td></tr><tr><td>Joint  $+$  frozen</td><td>4,000</td><td>0.793 1.612</td><td>1.644</td></tr></table>

## D.4 Computational cost protocol and estimates

We compare four fixed-budget workflows on the $\mathrm { N _ { 2 } }$ bond scan. Table 7 gives the budgets, MAEs on the same eight geometries, and total cost estimates. All energies use three independent full-Hamiltonian evaluation replicates. The cost checkpoints are separate from the 20,000-update models used in the main energy results.

Training budgets. Independent optimization starts each geometry from random initialization and uses AdamW for 4,000 updates, with 100 updates of linear warmup to $3 \times 1 0 ^ { - 4 }$ and cosine decay to $3 \times 1 0 ^ { - 5 }$ at update 4,000. Joint training uses five anchors and a constant learning rate of $1 0 ^ { - 3 }$ for 4,000 updates. Shared pretraining removes geometry conditioning and minimizes the equally weighted mean of the five anchor losses for 2,000 updates at $3 \times 1 0 ^ { - 4 }$ . Each query then receives 200 fine-tuning updates with a fresh optimizer. Sequential warm starts optimize the first geometry for 3,000 updates, then transfer model weights to the next geometry with a fresh optimizer for 100 updates. The first-point schedule uses cosine decay from $1 0 ^ { - 3 }$ toward $1 0 ^ { - 5 }$ over 20,000 updates, without warmup. Shared fine-tuning and subsequent warm-start points use 200-update linear warmup toward $1 0 ^ { - 4 }$ and cosine decay toward $1 0 ^ { - 5 }$ over 10,000 updates. All workflows use the same aligned orbital basis and full-Hamiltonian evaluation protocol.

Hardware and accounting. Independent optimization and adaptation use one A100; joint training and shared pretraining use two. GPU-hours sum training, process initialization, checkpoint loading, and final full-Hamiltonian evaluation across allocated GPUs; CPU orbital and integral preparation is outside this GPU-cost accounting. Shared pretraining and joint training cost 0.835 and 1.601 GPU-hours, respectively. The frozen-evaluation slope in Fig. 5 is 0.961 GPU-seconds per geometry, measured over the 161-point grid with three sampling replicates per geometry.

Table 8: Absolute full-Hamiltonian energy errors (mHa) at the eight common $\mathrm { N _ { 2 } }$ comparison geometries. Warm-start columns correspond to the two complete sequential paths.
<table><tr><td>R (Å)</td><td>Independent</td><td>Shared + fine-tune</td><td>Warm (41)</td><td>Warm (161)</td><td>Joint + frozen</td></tr><tr><td>0.60</td><td>0.284</td><td>0.458</td><td>0.365</td><td>0.365</td><td>0.327</td></tr><tr><td>0.76</td><td>0.663</td><td>0.473</td><td>0.559</td><td>0.458</td><td>0.527</td></tr><tr><td>0.84</td><td>1.054</td><td>0.601</td><td>0.705</td><td>0.566</td><td>0.573</td></tr><tr><td>0.96</td><td>1.097</td><td>0.728</td><td>0.898</td><td>0.787</td><td>0.723</td></tr><tr><td>1.00</td><td>0.709</td><td>0.790</td><td>0.998</td><td>0.872</td><td>0.788</td></tr><tr><td>1.18</td><td>1.868</td><td>1.477</td><td>1.666</td><td>1.392</td><td>1.088</td></tr><tr><td>1.28</td><td>1.755</td><td>1.864</td><td>2.019</td><td>1.669</td><td>1.138</td></tr><tr><td>1.32</td><td>0.782</td><td>2.123</td><td>2.176</td><td>1.795</td><td>1.176</td></tr></table>

Timing geometries and accuracy. Independent optimization and shared fine-tuning are timed at the same eight geometries selected by stratified sampling (Table 8); per-geometry costs are weighted by the fraction of the grid represented by each stratum. All four workflows report MAE on these common points. Sequential warm starts traverse the 41- and 161-point grids in ascending bond length, with full-grid MAEs of 1.182 and 0.993 mHa; frozen evaluation gives 0.823 and 0.822 mHa, respectively.

Query-count scaling. For each workflow, Equation (6) holds its measured average per-geometry cost fixed. For warm starts, the intercept is the first-point training and evaluation cost minus one subsequent-point cost, so that G = 1 recovers the first-point cost exactly. The two warm-start curves use the mean costs measured at their respective grid spacings. Writing the joint-training cost as $a _ { \mathrm { f r o z e n } }$ and the per-geometry evaluation cost as $b _ { \mathrm { f r o z e n } }$ , the estimated speedup over joint training with frozen evaluation is

$$
S _ { m } ( G ) = \frac { a _ { m } + G b _ { m } } { a _ { \mathrm { f r o z e n } } + G b _ { \mathrm { f r o z e n } } } , \qquad \operatorname * { l i m } _ { G \to \infty } S _ { m } ( G ) = \frac { b _ { m } } { b _ { \mathrm { f r o z e n } } } .\tag{11}
$$

The slope ratios are 986 for independent optimization, 77.4 for shared fine-tuning, and 40.8–43.0 for sequential warm starts. The 41- and 161-point grids have been evaluated for joint and sequential workflows; curves at other query counts are cost-model projections.