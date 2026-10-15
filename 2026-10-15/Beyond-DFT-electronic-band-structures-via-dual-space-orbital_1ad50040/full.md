# Beyond DFT electronic band structures via dual-space orbital learning

Minhyuk Kang<sup>\*,1,3</sup>, Beom Seok Kang<sup>\*,2</sup>, Nikolai Argatof<sup>1,4</sup>, Miguel Nouman<sup>2</sup>, Danish Khan<sup>1</sup>, William A. Goddard III<sup>2,†</sup>, and Anima Anandkumar<sup>1,†</sup>

<sup>1</sup>Division of Engineering and Applied Science, California Institute of Technology <sup>2</sup>Division of Chemistry and Chemical Engineering, California Institute of Technology <sup>3</sup>Department of Chemical Engineering, Imperial College London <sup>4</sup>Department of Mathematics, ETH Zürich Equal Contribution

<sup>†</sup>Corresponding author(s). E-mail(s): wag@caltech.edu; anima@caltech.edu

## Abstract

Accurate electronic band structures are essential for predicting material properties, yet highfidelity calculations remain computationally demanding. Here we introduce O-kRystal, a dual-space orbital-learning framework that uses a lower-cost Perdew–Burke–Ernzerhof (PBE) calculation both as an energy baseline and as a source of physically structured orbital information, combining band energies within real- and reciprocal-space orbital matrices to predict higher-level corrections to individual electronic states as a function of crystal momentum k and spin. Using a ±5 eV window around the band-edge midpoint, O-kRystal predicts Heyd–Scuseria–Ernzerhof (HSE06)-level band energies with a mean absolute error of 0.171 ± 0.009 eV across nearly 6,000 three-dimensional materials, spanning insulating/metallic and magnetic/non-magnetic electronic structures. On a separately held-out benchmark of six strongly correlated materials, it reduces the corresponding PBE band-energy MAE by 68.7%, while resolving orbital-dependent changes in the relative placement of transition-metal d and ligand p states. Even with supervision only at high-symmetry points, the model maintains comparable accuracy at unseen path-interior k points and reconstructs HSE06 densities of states from dense Brillouin-zone predictions. Finally, training O-kRystal for G<sub>0</sub>W<sub>0</sub> quasiparticle energies of two-dimensional materials demonstrates applicability across levels of theory, establishing orbital learning as a general strategy for higher-fidelity k-resolved electronic-structure prediction beyond density functional theory (DFT) calculations.

## 1 Introduction

Electronic band structure provides a momentum (k)-resolved description of the allowed electronic states of a crystalline material [1, 2]. While scalar quantities such as the band gap compress this electronic structure into individual descriptors, the full dispersion retains the energies, curvature, crossings, and spin dependence of individual states. These features govern carrier dynamics and are central to electronic and thermal transport, where the relative energies and shapes of the bands strongly influence material response [3–6]. Consequently, accurate band dispersions provide microscopic design information for screening and engineering materials for photovoltaic, electronic, and thermoelectric devices [7, 8]. Momentum-resolved electronic structure is also particularly important in correlated, topological, and other quantum materials, where changes in the dispersion can define qualitatively diferent electronic phases, including metallic, insulating, and topological states, and can be probed experimentally by angle-resolved photoemission spectroscopy [9–12]. Predicting the full band structure therefore preserves significantly more of the underlying electronic-state information than learning a predefined set of scalar properties.

First-principles electronic-structure methods provide a systematic framework for calculating band structures, but improved descriptions of electronic states generally come at greater computational cost [13, 14]. Density-functional theory (DFT) with semilocal exchange–correlation functionals, such as Perdew–Burke–Ernzerhof (PBE), is eficient for large-scale materials calculations [15], but typically underestimates band gaps because of well-known limitations associated with delocalization error and the absence of the exchange-correlation derivative discontinuity [16–18]. Hybrid functionals such as Heyd–Scuseria–Ernzerhof (HSE06) incorporate screened nonlocal exact exchange, reducing delocalization errors and systematically improving band gaps and electronic state energies [19–21]. Many-body approaches, such as the Green’s function based GW approximation, use the electronic self-energy to obtain quasiparticle addition and removal energies that are more directly connected to spectroscopic observables [22, 23]. Both approaches are substantially more expensive than semilocal DFT: conventional hybrid-functional and $G _ { 0 } W _ { 0 }$ implementations can exhibit asymptotic scaling of approximately $\mathcal { O } ( N ^ { 4 } )$ with system size, compared with $\mathcal { O } ( N ^ { 3 } )$ for Kohn–Sham DFT. Additionally, their cost can also grow more steeply with the number of sampled k-points, making many calculations prohibitively expensive and sometimes infeasible [24, 25]. This cost–accuracy trade-of motivates methods that retain the electronic information already available from an inexpensive baseline calculation while eficiently predicting the corresponding target-theory band structure. Importantly, both HSE06 and GW remain closely connected to a lower-level electronic-structure starting point [20, 23]. HSE06 operates within a generalized Kohn–Sham framework, while GW calculations are typically constructed from mean-field orbitals and eigenvalues. This makes the orbital information obtained from an inexpensive DFT calculation a natural, physically structured representation for learning corrections to higher-level electronic structure.

Machine learning has increasingly been used to accelerate electronic-structure prediction, from scalar material properties to complete state-resolved descriptions [26]. Structure-based machinelearning models can eficiently predict quantities such as band gaps or densities of states directly from crystal structures [27–29], while direct structure-to-band approaches predict the electronic dispersion without an intermediate electronic-structure calculation [30]. At a more advanced level, equivariant neural networks can learn electronic Hamiltonians from atomic structure and thereby recover band structures through a subsequent diagonalization [31–35]. Matrix-supervised Hamiltonian-learning approaches require target-theory labels in a specified basis. Learning bandenergy corrections instead uses eigenvalue targets without requiring target-theory matrix elements. Recently, neural operators have alternatively targeted the Kohn–Sham potential-to-density map to accelerate self-consistent calculations [36].

An intermediate strategy is to retain the electronic structure from a lower-cost calculation and learn its correction to a higher level of theory. This is particularly attractive because state-resolved band and quasiparticle energies are increasingly available in electronic-structure databases and are more directly connected to spectroscopic observables than basis-dependent target-theory Hamiltonians [10, 11, 23, 37–39]. Existing band-structure learning approaches, however, remain restricted along diferent conditions. Bandformer directly predicts DFT-level band dispersions from crystal structure on a large materials dataset, but is demonstrated for non-magnetic materials using a fixed six-band representation near the Fermi level [30]. Adhikari et al. instead learned PBE-to-HSE06 band-energy corrections for 337 bulk semiconducting and insulating compounds, using states from one sampled k-point per material [40]. Knøsgaard and Thygesen learned PBE-to- $G _ { 0 } W _ { 0 }$ quasiparticle corrections for 286 non-magnetic two-dimensional semiconductors over a fixed manifold of eight valence and four conduction bands [41].

These studies leave an important gap: higher-fidelity electronic band-structure prediction has not yet been demonstrated at scale across chemically diverse materials while simultaneously covering metals and insulators, explicitly resolving spin, and accommodating diferent numbers of bands across materials. To our knowledge, no previous framework combines these capabilities within a single state-resolved electronic band-structure learning approach.

Orbital learning ofers a natural way to use the information already contained in a lower-cost electronic-structure calculation. Rather than predicting target-theory properties from atomic structure alone, OrbNet and its subsequent variants use orbital-based quantities from inexpensive quantum-mechanical calculations to generate input features, matrices representing the electronicstructure (Fock, density, overlap, core-Hamiltonian), which have demonstrated high accuracy and data eficiency across molecular systems [42–45]. These developments show that low-cost orbital information can provide a highly informative and data-eficient representation for learning higher-fidelity quantum-mechanical properties.

Extending orbital learning from molecules to crystalline materials requires an explicit treatment of crystal momentum k, which labels Bloch states in a periodic solid [1, 2]. In a localized atomic-orbital basis, electronic operators can be represented either by real-space matrix elements connecting orbitals across lattice translations or by their reciprocal-space Bloch transforms at each k. Real-space emphasizes localized orbital interactions, whereas reciprocal-space exposes their momentum dependence and the resulting electronic dispersion [46]. We combine both views in dual-space orbital learning, allowing the model to jointly encode local electronic interactions and k-resolved band information.

We introduce O-kRystal, a crystal-momentum (k)-resolved orbital-learning framework for higherlevel electronic band structures from a lower-cost PBE calculation. The PBE calculation provides the baseline band structure through its band energies, while its real- and reciprocalspace orbital matrices ofer complementary representations of the underlying electronic state. For the primary PBE-to-HSE06 task, the model is trained on the diference between matched HSE06 and PBE band energies at high-symmetry k-points. O-kRystal learns how the energy of each electronic state changes from PBE baseline across band index, crystal momentum, and spin, and adds this predicted correction to the corresponding PBE energy to obtain the target-level band structure. Because the model predicts electronic states individually, it can accommodate diferent numbers of valence and conduction bands across materials rather than requiring a fixed-size band representation.

We evaluate the framework on nearly 6,000 chemically diverse three-dimensional materials spanning insulating, metallic, and spin-polarized electronic structures [37]. For PBE-to-HSE06 prediction, O-kRystal achieves a band-energy MAE of $0 . 1 7 1 { \pm } 0 . 0 0 9 \ \mathrm { e V }$ for electronic states within ±5 eV of the band-edge midpoint. Even though training uses only high-symmetry k-points, the model generalizes well to unseen k-points throughout the Brillouin zone, enabling accurate reconstruction of HSE06 densities of states and derived electronic properties. O-kRystal also recovers the target-theory electronic structure even when the PBE baseline gives the wrong qualitative picture, including materials that are metallic at the PBE level but insulating at HSE06, while retaining systems that remain metallic. On six separately held-out strongly correlated materials, the model reduces the band-energy MAE over densely sampled Brillouinzone states within the 5 eV window from 1.532 eV with PBE to $0 . 4 7 9 \ \mathrm { e V }$ , a 68.7% reduction. For the canonical Mott insulator AFM-II MnO, orbital-projected spectra further reveal distinct corrections to Mn 3d and O 2p state energies, demonstrating behavior beyond a uniform shift of the band edges. The predicted band structures further improve the predictions of experimentally validated downstream properties, including fundamental band gaps and efective masses. Finally, when trained separately on 286 two-dimensional semiconductors, the same framework achieves a quasiparticle-energy MAE of $\mathrm { 0 . 1 2 6 \pm 0 . 0 1 7 ~ e V }$ for $\mathrm { P B E - t o } { - } G _ { 0 } W _ { 0 }$ prediction [38, 39]. Together, these results establish O-kRystal, to our knowledge, as the first broadly applicable framework for beyond-DFT electronic band-structure prediction across chemically and electronically diverse crystalline materials, opening a route toward scalable high-fidelity electronic-structure prediction across materials and levels of theory.

## 2 Results

## a Beyond DFT band correction

![](images/29eb04ff0ea01187066ab5e394aa8298082bd867c4227982337eca4b4c558d8d.jpg)

![](images/97fbea7a1148e7361e631d47c396991abed34d7e138f16d52d05d8be6f6d8b25.jpg)

![](images/e5b6ee5a95cf548b0e37b0512a1c0f34a239b2b5fbf184eb37d75da20a732878.jpg)

![](images/455d2c46499f0170e3a162c6998e25db1d961a3dd92479eac987a3e62e7b90e4.jpg)  
"Figure 1: Dual-space orbital learning and representative applications. ${ \mathbf { a } } ,$ Starting from <sub></sub> <sup>\$</sup>a crystal structure, a DFT calculation supplies baseline band energies and orbital matrices, \$<sup></sup> "/including the Hamiltonian H, overlap S, density P, and spin-density matrices $\mathbf { P } _ { s p i n } ^ { ( \sigma ) } .$ , represented 6 <sub>+</sub>in real (R)- and reciprocal (k)-space. The dual-space orbital representations are processed by a <sup>'</sup>periodic E(3)-equivariant graph neural network to predict state-dependent energy corrections. Target-theory (e.g., HSE06 or $G _ { 0 } W _ { 0 } )$ band energies are reconstructed as $\hat { \varepsilon } _ { n \mathbf { k } \sigma } ^ { \mathrm { t a r g e t } } = \varepsilon _ { n \mathbf { k } \sigma } ^ { \mathrm { b a s e } } + \widehat { \Delta \varepsilon } _ { n \mathbf { k } \sigma }$ $\mathbf { b } ,$ <sub> /0"0%/ %</sub>6	<sub></sub>     6	      6	  #/,)10Band dispersion and density of states for the displayed spin-down channel of $\mathrm { G a O _ { 2 } }$ %, comparing PBE, HSE06, and O-kRystal. For the three-dimensional $\mathrm { P B E - t o - H S E 0 6 }$ task, supervision is restricted to high-symmetry points. The model generalizes to unseen path-interior points, while the density of states is reconstructed from predictions on a dense Brillouin-zone mesh. $\mathbf { c } ,$ Spin-resolved Mn 3d and O 2p projected densities of states for $\mathrm { M n O }$ with AFM-II ordering along [111]. The O-kRystal reconstruction combines predicted HSE06 state energies with fixed PBE Mulliken projection weights, illustrating orbital-dependent changes in state-energy placement while retaining the baseline orbital character. Resolving the relative placement of transitionmetal d and ligand $p$ states is particularly important in strongly correlated oxides, where these energy scales are central to Mott and charge-transfer electronic behavior. $\mathbf { d } ,$ , Absolute band-gap errors relative to experiment for ten independently evaluated bulk materials from the mstar60 benchmark [47], compared across PBE, HSE06, and O-kRystal

## 2.1 The O-kRystal Framework

O-kRystal extends orbital learning to periodic electronic structures by formulating target-theory band prediction as a correction to a lower-cost baseline calculation (Fig. 1a). Given baseline band energies $\varepsilon _ { n \mathbf { k } \sigma } ^ { \mathrm { b a s e } }$ , the target energy is reconstructed as

$$
{ \widehat { \varepsilon } } _ { n \mathbf { k } \sigma } ^ { \mathrm { t a r g e t } } = \varepsilon _ { n \mathbf { k } \sigma } ^ { \mathrm { b a s e } } + { \widehat { \Delta } } \varepsilon _ { n \mathbf { k } \sigma } ,\tag{1}
$$

where $n ,$ k, and $\sigma$ denote the band index, crystal momentum, and spin channel, respectively, and the hat denotes a model prediction. In this work, PBE serves as the principal baseline, while HSE06 and $G _ { 0 } W _ { 0 }$ define the target theories [15, 19, 20, 22]. This ∆-learning formulation preserves the dispersion already captured by the baseline calculation and focuses the learning problem on the state- and crystal-momentum-dependent correction between electronic-structure methods [48].

Orbital representation Beyond the baseline eigenvalues, the underlying PBE calculation provides orbital-resolved information about the converged electronic state. O-kRystal represents the baseline electronic state through the Kohn–Sham Hamiltonian H, overlap matrix S, density matrix P, and spin-density matrix $\mathbf { P } _ { \mathrm { s p i n } } ^ { ( \sigma ) } = \mathbf { P } _ { \sigma } - \mathbf { P } _ { \bar { \sigma } }$ , where σ denotes the spin channel and σ¯ the opposite spin channel. On-site matrix blocks describe the electronic environment associated with individual atoms, while of-diagonal blocks encode interactions between atomic orbitals on neighboring sites (Fig. 1b). Overall, O-kRystal uses the lower-cost calculation not only as an energy baseline, but also as a physically structured representation of the material-specific electronic state.

Dual-space periodic representation For periodic systems, the same orbital-based quantities can be represented in real- and reciprocal-space. The real-space representation exposes localized and sparse atom-centered interactions, whereas the reciprocal-space representation naturally provides an explicit k-resolved description of the corresponding electronic structure (Fig. 1b). O-kRystal processes these complementary representations with a periodic E(3)-equivariant graph neural network. Real-space on-site blocks define atomic features and pairwise blocks enter equivariant message passing, while Bloch-phase information and reciprocal-space orbital features introduce explicit dependence on the queried crystal momentum. This formulation enables evaluation at crystal momenta that were not used as supervised training targets. Detailed definitions of the orbital representation and model architecture are provided in Methods 4.2 and 4.3.

## 2.2 Target-theory band structure prediction

Dataset and evaluation We first evaluate O-kRystal on PBE-to-HSE06 band-structure prediction using 6,900 band-structure entries from 5,988 parent materials, spanning broad elemental and electronic-structure diversity, including insulating and metallic as well as magnetic and nonmagnetic materials (Fig. 2a) [37]. The primary benchmark evaluates individual band energies at the supervised high-symmetry k-points over progressively wider energy windows around the band-edge midpoint. Dataset construction, splitting, and evaluation protocols are described in Methods 4.1.

State-resolved prediction accuracy O-kRystal substantially reduces the state-resolved error of the PBE baseline across both occupied and unoccupied bands (Fig. 2b). Evaluated over 575,487 state pairs within $| E - E _ { \mathrm { m i d } } | < 5$ eV from the representative model shown in Fig. 2b, the state-weighted MAE decreases from 0.84 eV for PBE to 0.19 eV with O-kRystal. Across three separately trained models using distinct data splits, O-kRystal achieves mean entry-balanced band-energy MAEs of $0 . 1 6 9 \pm 0 . 0 0 8 ~ \mathrm { e V } , 0 . 1 7 1 \pm 0 . 0 0 9 ~ \mathrm { e V } , 0 . 1 7 4 \pm 0 . 0 1 0 ~ \mathrm { e V }$ , and $0 . 1 8 8 \pm 0 . 0 1 2$ eV within $| E - E _ { \mathrm { m i d } } | < 2 , 5$ , 10, and 20 eV, respectively (Fig. 2c). The relatively small increase in error as the evaluation window expands indicates that the learned correction is not confined to band-edge energies, but remains efective over a broad portion of the electronic spectrum.

![](images/f04ecc5a76b5da4f996cd9ef2f1d95268d3619ef6b560c2c58937a61b3398c77.jpg)

![](images/bf58c4cdfb72423f8fc3365c9b3d6e643c9b0be2e145ec988bdad0861c5fac28.jpg)

![](images/a37c9f3041950504af69f295658b211821bdddcaf48c4a6ae7d277762f26d9b1.jpg)

![](images/5c0a30b7b960dd9fd57b4e24e94e4f9c2d73577401419d1961dfb637dad2d646.jpg)  
d

![](images/18b96edcf32139422711059127499a5567d8756a086b57ce7eff8662b1ff0b77.jpg)  
Figure 2: Large-scale prediction of HSE06 band structures with O-kRystal. ${ \mathbf { a } } ,$ Composition of the three-dimensional materials dataset containing 5,988 chemically diverse parent materials [37]. Element counts indicate the number of materials containing each element. Horizontal bars summarize $\mathrm { P B E - t o - H S E 0 6 }$ electronic-character classes and spin polarization. $\mathbf { b } ,$ PBE and O-kRystal band energies compared with HSE06 for 575,487 matched test-state pairs within the PBE-defined 5 eV energy window for a representative held-out test split. Energies are independently referenced to the band-edge midpoint of each spectrum. The dashed line denotes parity, and the reported MAEs are state weighted. $\mathbf { c } ,$ Entry-balanced band-energy MAEs for uncorrected PBE, scissor correction, Linear+KRR, Geometry GNN, and O-kRystal over energy windows of 2, 5, 10, and $2 0 ~ \mathrm { e V }$ . ∆ identifies the neural models trained directly on state-resolved PBE-to-HSE06 energy corrections. $\mathbf { d } ,$ Learning curves for O-kRystal as a function of training-set size, shown separately for insulating and metallic HSE06 entries within the 5 and 20 eV windows. Values and error bars in c and d denote the mean and sample standard deviation across three independently trained models with parent-grouped splits. Band-energy evaluations in b–d use the supervised high-symmetry k-points.

Comparison with correction baselines A common approximation to target-level band structures is to apply a simple correction to the baseline PBE energies. Therefore, we compare

O-kRystal with multiple PBE-to-HSE06 correction strategies, including a rigid scissor correction, a combined Linear+KRR model, and a geometry-based graph neural network trained using the same material splits and prediction targets (Methods 4.5; Fig. 2c). The uncorrected PBE MAE increases from 0.624 eV within 2 eV of the band-edge midpoint to 1.115 eV within 20 eV. The scissor and Linear+KRR corrections reduce the error, with Linear+KRR reaching MAEs of 0.29– 0.33 eV. Incorporating crystal geometry through the neural baseline further lowers these errors to 0.18–0.21 eV. O-kRystal achieves the lowest errors across all four windows, reaching 0.17–0.19 eV. Relative to the matched Geometry GNN, the reduction increases from 7.3% within 2 eV to 9.8% within 20 eV. These results show that the geometry-based model supplied with the PBE band baseline captures a substantial component of the PBE-to-HSE06 correction on the broad test distribution, while explicit orbital information provides a consistent additional improvement across the electronic spectrum. Learning-curve ablations further show that real-space orbital features improve upon the geometry-based model, with explicit reciprocal-space orbital matrices providing an additional advantage that becomes clearer with increasing training-set size and window range (Supplementary Fig. S1).

Learning across electronic regimes Prediction accuracy improves systematically as the amount of training data increases across both insulating and metallic systems (Fig. 2d). Within $| E - E _ { \mathrm { m i d } } | < 5 ~ \mathrm { e V }$ , the MAE for insulating materials decreases from $0 . 2 2 3 \pm 0 . 0 0 1$ eV with 0.5k training entries to $\mathrm { 0 . 1 5 1 \pm 0 . 0 0 8 ~ e V }$ with the full 5.5k training set, while the corresponding metallic error decreases from 0.278 ± 0.015 eV to $0 . 2 0 7 \pm 0 . 0 2 2$ eV. A similar trend is observed over the broader $| E - E _ { \mathrm { m i d } } | < 2 0 ~ \mathrm { e V }$ window, where the insulating and metallic errors decrease from $0 . 2 5 4 \pm 0 . 0 0 5$ eV and $0 . 3 1 1 \pm 0 . 0 1 1$ eV to $0 . 1 6 9 \pm 0 . 0 1 1$ eV and $\mathrm { 0 . 2 2 3 \pm 0 . 0 2 3 \ e V }$ , respectively. Metallic systems remain consistently more challenging across all training-set sizes, reflecting the greater complexity of partially occupied and near-degenerate electronic states. Nevertheless, both regimes continue to improve with additional training data, with no clear performance plateau over the range examined.

Overall, these results establish accurate PBE-to-HSE06 band correction across diverse materials, electronic regimes, and broad energy ranges at the sampled high-symmetry points. We next examine whether the learned k dependence extends beyond these reciprocal-space positions to unseen points along band paths and subsequently to of-path points throughout the Brillouin zone.

## 2.3 Generalization across the Brillouin Zone

Accurate prediction at high-symmetry points alone is insuficient for reconstructing the full electronic dispersion, as many electronic properties depend on states between and away from conventional high-symmetry paths. Thus, we examine whether the k dependence learned by O-kRystal extends to reciprocal-space positions that are not used as training targets. We evaluate reciprocal-space generalization in two settings: unseen points along the high-symmetry paths and of-path points throughout the three-dimensional Brillouin zone (Fig. 3).

Prediction at unseen path-interior points O-kRystal is evaluated at previously unseen k-points between the high-symmetry points used during training (Fig. 3a). Across 693 bandstructure entries, the median band-energy MAE remains essentially unchanged, from 0.122 eV at the supervised high-symmetry points to 0.121 eV at unseen path-interior points. This negligible degradation shows that the learned correction is not restricted to the discrete reciprocal-space positions encountered during supervision and motivates evaluation beyond the high-symmetry paths. A detailed analysis of the dependence on reciprocal-space distance, relative path position, and segment length is provided in Supplementary Fig. S2

![](images/5d7add327b15881cc568fad4c47630f9dfd6663787aa648d67f793b137b2510e.jpg)

![](images/9216bc270b4450a82648078c683837b1e56b181b285b85177781c9e43b325cfc.jpg)

![](images/5d09519bd8d774af6ba2db710563a59b928edac803c4ebc29652b1011eb3bcba.jpg)  
Figure 3: Generalization between and beyond high-symmetry k-points. a, Comparison of high-symmetry-point and path-interior band-energy MAEs for 693 held-out band-structure entries within the 5 eV window. Each point represents one entry, including separate spin channels where applicable. Median MAEs are 0.122 eV and 0.121 eV, respectively, and the dashed line denotes parity. b, Distribution of normalized density-of-states (DOS) reconstruction errors for 356 insulating and 225 metallic parent materials. Spectra are reconstructed from dense of-path predictions and compared with HSE06 over $| E - E _ { \mathrm { m i d } } | < 5$ eV using the normalized $L _ { 1 }$ metric defined in Methods 4.6. Dashed horizontal markers indicate the PBE and O-kRystal medians. $\mathrm { G a O _ { 2 } }$ and WC identify the examples displayed in Figs. 1b and $3 \mathrm { c } ,$ respectively. c, Band structure and density of states of WC, comparing HSE06, O-kRystal, and PBE. Marked high-symmetry points show the sampling convention used for supervision. Intermediate band-path points and the dense mesh used for DOS reconstruction are evaluated without target-theory information at inference.

Of-path prediction across the Brillouin zone We next consider generalization at of-path k-points throughout the Brillouin zone. Because the available reference data do not provide consistently matched PBE and HSE06 band states at arbitrary of-path k-points, we evaluate whether predictions across a dense reciprocal-space grid collectively reconstruct the HSE06 density of states (DOS), which depends on electronic states sampled throughout the Brillouin zone (Fig. 3b). The reconstruction error is quantified using area-normalized $L _ { 1 }$ diference from the HSE06 DOS over $E _ { \mathrm { m i d } } \pm 5 ~ \mathrm { e V }$ , as defined in Methods 4.6. Across 356 insulating and 225 metallic materials, O-kRystal achieves median DOS errors of 0.224 and 0.210, respectively, reduced from 0.874 and 0.457 for the corresponding PBE electronic structures.

The representative $\mathrm { G a O _ { 2 } }$ and WC examples, selected near the median O-kRystal DOS error of the respective electronic classes, illustrate both forms of reciprocal-space generalization (Figs. 1b and 3c). At unseen path-interior points, O-kRystal follows the HSE06 band dispersion substantially more closely than PBE, while dense of-path predictions reproduce the principal features of the corresponding HSE06 DOS. Together, these results show that the learned state corrections remain accurate between the supervised high-symmetry points and support spectral reconstruction from predictions at previously unseen of-path locations throughout the Brillouin zone.

## 2.4 Electronic-character recovery and strongly correlated materials

A central question is whether O-kRystal learns material- and state-specific changes in electronic structure rather than primarily reproducing the systematic tendency of HSE06 to increase band separations relative to PBE. We examine this through electronic-character recovery, a separately held-out benchmark of six strongly correlated materials, and orbital-resolved reconstruction of transition-metal 3d and oxygen 2p states.

Recovery of electronic character A qualitative discrepancy occurs when PBE and HSE06 disagree on whether a material is metallic or insulating. Because HSE06 often increases the separation between occupied and unoccupied states relative to PBE, improved agreement could arise from a generic tendency to open gaps rather than from a material-specific correction. Thus, we classify the test entries according to their PBE-to-HSE06 electronic-character change as insulator-to-insulator, metal-to-insulator, and metal-to-metal systems (Fig. 4a). Across the three trained models, O-kRystal recovers the corresponding HSE06 electronic character with accuracies of $9 9 . 8 \pm 0 . 3 \%$ 2 $8 6 . 2 \pm 6 . 0 \%$ , and $7 9 . 6 \pm 1 . 7 \%$ , respectively. The ability to recover metal-to-insulator cases while preserving metal-to-metal systems indicates that the learned correction is material dependent rather than a uniform tendency to open band gaps.

Strongly correlated material evaluation A separate set of six strongly correlated materials was excluded from training and reserved a priori for evaluation because these correlation-sensitive systems are well established as challenging for semilocal density-functional approximations [49– 62]. For each material, the represented magnetic and electronic state was additionally checked against the corresponding literature-supported phase (Methods 4.7; Fig. 4b).

The six strongly correlated materials are evaluated on dense Brillouin-zone meshes using the representative model. Within the 5 eV window, the material-balanced band-energy MAE decreases from 1.532 eV for PBE to 0.479 eV with O-kRystal, corresponding to a 68.7% reduction. The matched Geometry GNN reaches 0.692 eV, giving O-kRystal a further 30.8% reduction in error and indicating that explicit orbital information provides additional predictive value in this correlation-sensitive regime.

The improvement also extends to the orbital-resolved electronic structure (Fig. 4c). Within the ±5 eV PDOS window, the material-balanced mean normalized $L _ { 1 }$ error for transition-metal 3d states decreases from 1.451 with PBE to 0.709 with O-kRystal, while Geometry GNN reaches 0.989. The corresponding oxygen $2 p$ error decreases from 1.036 for PBE to 0.631 for O-kRystal, compared with 0.753 for Geometry GNN. These values correspond to reductions relative to PBE of 51.1% and 39.1% for the 3d and $2 p$ channels, respectively, with further reductions of 28.3% and 16.2% relative to the geometry-based model. O-kRystal outperforms Geometry GNN for the transition-metal 3d channel in all six materials, whereas the oxygen 2p improvement is more material dependent. Because the model PDOS is reconstructed using fixed PBE Mulliken projection weights transported to the predicted state energies, these results quantify improvements in orbital-resolved state-energy placement rather than prediction of target-theory wavefunctions or changes in hybridization.

PBE-to-HSE06 electronic-character change  
![](images/7b6463250696cd71820c1e78311482bb584b982b7b258c243aaf57f552a9db3a.jpg)  
c Orbital-dependent corrections  
b

![](images/645df3db44f7a56db681d849b61cbf6452444db84b29a3d8ac1580b40b1a2588.jpg)  
PBE E HSE06 O-kRystalBE HSE06 O-kRGeometry GNN (Δ) O-kRystal (Δ)

![](images/3da11f42f793f6823aa300dc404f8ab2ca14f99199b561dc117d6dfae98f373a.jpg)

![](images/9c2a4d789b095096d87669b1be980d9e0761dae1b40906ed4967049bc94ad90d.jpg)

![](images/379b385a536747163fdd90419dbc70bb90c17403ad614b397e5941f6b33ef713.jpg)  
<sup>4</sup> <sup>0</sup> <sup>4 2.8</sup> <sup>0</sup> <sup>2.8</sup>  Figure 4: Electronic-character recovery and orbital-dependent corrections in strongly correlated materials. ${ \mathbf { a } } ,$ <sup>PDOS</sup> <sup>(states</sup> <sup>eV−1)PDOS</sup> <sup>(states eV−1)</sup>Recovery of HSE06 electronic character for insulator-to-insulator, metal-to-insulator, and metal-to-metal systems. Bars show the fraction for which O-kRystal predicts the correct HSE06 metallic or insulating character across three independently trained models. Adjacent fractions report the number of correct predictions and total entries pooled across the corresponding test evaluations. b, Distributions of band-energy MAE for PBE, Geometry GNN, and O-kRystal in the main held-out test cohort of 693 entries and a separate benchmark of six literature-validated strongly correlated materials [49–62]. The correlated-material comparison is evaluated on dense Brillouin-zone meshes within the 5 eV window. Horizontal markers denote means. The annotated 9.2% and 30.8% reductions compare O-kRystal with Geometry GNN within the respective cohorts. $\mathbf { c } ,$ Reconstruction errors relative to HSE06 for transition-metal 3d and oxygen 2p projected densities of states (PDOS) across six correlated materials, with spectra reconstructed using fixed PBE Mulliken projection weights to isolate diferences in state-energy placement. Each row corresponds to one material, with PBE, Geometry GNN, and O-kRystal evaluated separately. The schematic summarizes the relative energy alignment of metal-centered transition-metal 3d and ligand-derived oxygen $2 p$ states, which is particularly relevant to correlation-sensitive electronic structure. The MnO spectra in Figure 1 illustrate the corresponding orbital-dependent energy redistribution.

Orbital-dependent electronic structure of MnO For a closer examination, Figure 1c shows antiferromagnetic rocksalt MnO with AFM-II ordering along [111] [57–59]. The reconstructed spectrum shows distinct corrections across the Mn 3d and O 2p states. Occupied and unoccupied Mn 3d features undergo diferent energy shifts across the two spin channels, while the O 2p spectral distribution is displaced toward the corresponding HSE06 energy range. The correction therefore modifies the relative placement of metal- and ligand-derived states rather than acting only through a uniform change in the fundamental gap. Such $d { - } p$ energetics are particularly relevant in correlated oxides, where the relative placement of transition-metal d and ligand p states is central to Mott and charge-transfer-sensitive electronic structure. These orbital-resolved corrections could therefore support screening of strongly correlated materials in which doping, pressure, or strain reorganizes transition-metal d and ligand p states in ways relevant to emergent superconducting phases.

## 2.5 Derived electronic properties

![](images/de0539dbb647c07afe7524941c133be7b95ce35bf3860b8a2068877a0a63be2b.jpg)

b Gap-informed effective masses  
![](images/55fc763c92ee1231a04c30fd05a05616e3d905ac3c32ade3d18340d33d50ea6e.jpg)  
Figure 5: Band gaps and gap-informed efective-mass estimates. a, Fundamental band gaps derived from PBE and O-kRystal band energies compared with HSE06 reference gaps across the test set. The dashed line denotes parity, and the corresponding MAE and $R ^ { 2 }$ values are reported for both methods. b, Material-balanced mean absolute percentage errors (MAPE) relative to experiment for 31 directional efective masses across ten bulk materials from mstar60 benchmark [47]. Results are grouped into III–V semiconductors, chalcogenides, and other semiconductors. O-kRystal\* denotes a calibrated gap-informed reconstruction in which the corresponding PBE efective mass for the same carrier branch, band extremum, and crystallographic direction provides the baseline curvature information and is rescaled using the predicted-to-PBE fundamental-gap ratio. Reconstruction parameters are fitted to HSE06 masses from the other material families in a leave-one-family-out protocol.

The predicted state energies also enable evaluation of derived electronic properties. Fundamental band gaps are obtained directly from the predicted valence- and conduction-band edges, while carrier efective masses are estimated using a gap-dependent reconstruction motivated by the Kane model.

Fundamental band gaps Across the test set, O-kRystal systematically improves the fundamental band gaps relative to PBE (Fig. 5a). The PBE gaps exhibit an MAE of 0.926 eV relative to HSE06 and an $R ^ { 2 }$ of 0.70. In comparison, gaps derived from the O-kRystal-predicted band energies reduce the MAE to 0.203 eV and increase $R ^ { 2 }$ to 0.98. The band gap is recovered directly from the predicted valence- and conduction-band energies rather than from a separate scalar output head. Its improved accuracy therefore shows that the state-resolved predictions remain consistent at the band edges.

Comparison with experimental band gaps We next compare with experimental measurements using the mstar60 benchmark [47], using the ten-material bulk cohort defined in Methods 4.7 (Figs. 1d and 5b). PBE generally underestimates the experimental gaps, whereas HSE06 provides closer agreement. O-kRystal follows these material-dependent HSE06 corrections, improving agreement with experiment where HSE06 does while retaining similar deviations where HSE06 itself remains inaccurate.

Gap-informed efective masses Because the available HSE06 data do not densely resolve the local band curvature around the extrema, O-kRystal is not evaluated as a direct efectivemass predictor. Instead, its predicted HSE06 gaps are propagated through a gap-dependent reconstruction motivated by the Kane model (Methods 4.7) to estimate 31 directional efective masses across the ten-material benchmark (Fig. 5b). Using leave-one-family-out calibration, the reconstructed masses reduce the overall experimental MAPE from 32.05% for PBE to 27.05%, compared with 19.87% for HSE06. The improvement is consistent across all three material families, indicating that more accurate band-edge energetics can improve downstream efective-mass estimates even when the local curvature itself is not directly supervised.

## 2.6 Extension to $G _ { 0 } W _ { 0 }$ quasiparticle energies

![](images/ed5d3b34b1a6481c87f506a39b6aa1259201ae06e98e40f39b05acdf488c50a5.jpg)

![](images/dbba3bbd22dce49317848a9b2f15af411dbbda4efef5ff7424d5767d14857ce6.jpg)  
Figure 6: Extension to $G _ { 0 } W _ { 0 }$ quasiparticle energies. a, Five-fold cross-validated quasiparticle-state and band-gap MAEs for O-kRystal on 286 two-dimensional semiconductors from the C2DB-derived $G _ { 0 } W _ { 0 }$ benchmark [38, 39, 41]. Separately trained models use either GPAW or independently calculated FHI-aims PBE eigenvalues [63, 64], while the $G _ { 0 } W _ { 0 }$ labels, sampled k points, and orbital-matrix inputs are held fixed. Each outer split uses three folds for training, one for validation and one for testing. Bars and error bars denote means and sample standard deviations across the five test folds. b, Absolute diferences between FHI-aims and GPAW PBE eigenvalues for the eight highest valence (VB1–VB8) and four lowest conduction (CB1–CB4) states. Circles indicate the median discrepancy, boxes the interquartile range, and whiskers the 10th–90th percentile range. Shading highlights the frontier bands VB1 and CB1.

The preceding results focus on PBE-to-HSE06 corrections in three-dimensional periodic materials. We finally test whether the same orbital-learning framework extends to a physically distinct electronic-structure setting by considering $G _ { 0 } W _ { 0 }$ quasiparticle energies. Whereas HSE06 remains a generalized Kohn-Sham description, $G _ { 0 } W _ { 0 }$ introduces an energy-dependent electronic selfenergy to approximate electron addition and removal energies, providing a more direct connection to photoemission observables [22, 23]. We evaluate O-kRystal on a C2DB-derived quasiparticle benchmark comprising 286 two-dimensional semiconductors using the cross-validation protocol described in Methods 4.8 [38, 39, 41].

Beyond DFT band structure prediction Using five-fold material-level cross-validation with a separate validation fold for checkpoint selection, O-kRystal achieves a quasiparticle-state MAE of $0 . 1 2 6 \pm 0 . 0 1 7$ eV across the twelve available bands and a band-gap MAE of $0 . 1 8 5 \pm 0 . 0 3 5$ eV when using the GPAW PBE baseline (Fig. 6a; Supplementary Table S9). These results extend the same orbital-learning formulation from three-dimensional PBE-to-HSE06 band reconstruction to two-dimensional $G _ { 0 } W _ { 0 }$ quasiparticle energies without modifying the underlying learning architecture.

The established C2DB $G _ { 0 } W _ { 0 }$ benchmark also enables comparison with the state-specific approach of Knøsgaard and Thygesen, which used electronic state fingerprints constructed from orbital projections and operator matrix elements together with gradient-boosted regression [41]. Their model achieved a band-gap MAE of 0.18 eV on the selected test split and state-energy MAE of $\mathrm { 0 . 1 3 \pm 0 . 0 2 ~ e V }$ with repeated random splits. O-kRystal achieves comparable accuracy under a separate train/validation/test fold protocol, while retaining a unified orbital-learning architecture across HSE06 and $G _ { 0 } W _ { 0 }$ targets.

Baseline consistency in ∆-learning Because the reference $G _ { 0 } W _ { 0 }$ calculations were performed on top of GPAW PBE electronic structures, we further examine how consistency between the baseline and target calculations afects the learned quasiparticle correction (Fig. 6a). Using the corresponding GPAW PBE baseline, O-kRystal achieves an all-band MAE of $\mathrm { 0 . 1 2 6 \pm 0 . 0 1 7 ~ e V }$ whereas replacing it with independently calculated FHI-aims PBE bands increases the error to $0 . 1 6 8 \pm 0 . 0 1 0$ eV. The band-gap MAE also increases, from $0 . 1 8 5 \pm 0 . 0 3 5$ eV with the GPAW baseline to $\mathrm { 0 . 2 0 6 \pm 0 . 0 2 7 ~ e V }$ with the FHI-aims baseline [63, 64].

The state-resolved comparison helps explain this diference (Fig. 6b). The two PBE calculations agree most closely for the band-edge energies surrounding the band gap, whereas their discrepancy increases for states farther from the band edges, particularly for the highest-energy conduction bands. This pattern is consistent with the larger sensitivity of full state-resolved quasiparticle prediction to baseline inconsistency, while the band-gap error reflects only the frontier states. Diferences between the two PBE calculations may arise from several implementation-specific factors, including the basis representation and treatment of the ionic cores. More generally, these results show that state-resolved ∆-learning benefits from consistency between the baseline electronic structure and the starting point used to construct the corresponding target-theory quasiparticle calculation.

## 3 Discussion

O-kRystal uses lower-cost electronic structure as both the baseline for ∆-learning and a representation of the underlying electronic state. While geometry and PBE band information capture much of the systematic PBE-to-HSE06 correction, orbital features provide additional information for material- and state-specific deviations. This advantage becomes more pronounced in the separately held-out correlated-material benchmark and is particularly consistent for transition-metal 3d states, suggesting that explicit electronic information is especially valuable when localized, correlation-sensitive states dominate the low-energy spectrum. Representation comparisons likewise show that the full dual-space model achieves progressively lower errors than the geometry-based and real-space alternatives as the training set and evaluated energy range increase (Supplementary Fig. S1).

The reciprocal-space results further show that the learned correction generalizes beyond the high-symmetry k-points used for supervision. Accuracy remains essentially unchanged at unseen path-interior points, while dense Brillouin-zone evaluation enables reconstruction of target-level spectral quantities away from conventional band paths. This combination of state-resolved prediction and reciprocal-space generalization allows O-kRystal to recover material-specific changes in electronic dispersion without requiring target-theory calculations at each queried k-point.

These capabilities provide a natural route for electronic-structure screening. O-kRystal recovers both metal-to-insulator changes and systems that remain metallic, while the correlated-material analysis resolves orbital-dependent changes in the relative placement of transition-metal d and ligand p states. Such predictions could identify materials for which DFT provides an inadequate low-energy electronic structure and prioritize candidates for more computationally demanding hybrid-functional or many-body calculations, including systems in which doping, pressure, or strain reorganizes metal–ligand states in ways relevant to superconducting phases and Cooperpair formation. Extension to $G _ { 0 } W _ { 0 }$ quasiparticle energies further shows that the state-resolved learning framework is not specific to HSE06, while the sensitivity to the PBE starting point emphasizes that the baseline electronic states should remain consistent with those from which the target correction is defined.

Several limitations define the present scope. O-kRystal still requires a converged baseline electronic-structure calculation and therefore occupies a diferent cost-accuracy regime from structure-only surrogate models. This dependence could be substantially reduced by emerging approaches that learn the potential-to-density map and avoid repeated orbital diagonalizations during self-consistency [36]. Coupling such accelerated baseline solvers with orbital learning target-theory correction could extend the present framework to larger systems while retaining a physically grounded electronic starting point. O-kRystal also predicts target-theory state energies rather than wavefunctions, so orbital-projected reconstructions retain projection information from the baseline calculation and cannot capture target-induced changes in hybridization or wavefunction character. A natural next step is therefore to move from state-energy correction toward learning richer electronic objects, including electron densities [65], compact target-theory Hamiltonians in Wannier representations [66, 67] and, ultimately, many-body Green’s functions or self-energies that provide access to quasiparticle spectra, spectral functions, and broader ground- and excited-state observables [68]. In conclusion, these directions suggest a path from accelerating the baseline quantum calculation to learning increasingly complete representations of the target electronic theory.

## 4 Methods

## 4.1 Dataset construction and electronic-structure data

Three-dimensional PBE–HSE06 dataset The primary dataset is derived from paired PBE and HSE06 electronic-structure calculations for chemically diverse three-dimensional crystalline materials [37]. After data cleaning and exclusion of materials reserved for downstream benchmarks, the final benchmark subset contains 6,900 band-structure entries corresponding to 5,988 parent materials. Spin-polarized materials are represented by separate spin-channel entries. Parent materials are removed when the absolute diference between the PBE and HSE06 magnetic moments exceeds 0.5 $\mu _ { \mathrm { B } }$ , reducing inconsistencies between the baseline and target electronic states.

The final benchmark dataset is partitioned at the parent-material level into approximately 8:1:1 training, validation, and test sets, such that repeated entries and corresponding spin channels remain within the same partition. Complete dataset accounting and seed-specific split sizes are reported in Supplementary Section S1 and Supplementary Table S1. Materials used for the correlated-material and experimental downstream benchmarks are excluded before construction of these splits.

State alignment and learning targets PBE and HSE06 states are paired according to their stored band ordering relative to the band edges. The common valence and conduction manifolds are retained, and each PBE and HSE06 spectrum is independently centered at the midpoint between the pathwise maximum of the highest valence band and minimum of the lowest conduction band before constructing the state-resolved correction

$$
\begin{array} { r } { \Delta \varepsilon _ { n \mathbf { k } \sigma } = \varepsilon _ { n \mathbf { k } \sigma } ^ { \mathrm { H S E 0 6 } } - \varepsilon _ { n \mathbf { k } \sigma } ^ { \mathrm { P B E } } . } \end{array}\tag{2}
$$

Throughout the band-energy analysis, $E _ { \mathrm { m i d } }$ denotes this operational band-edge-energy reference rather than a separately determined thermodynamic chemical potential.

For the primary three-dimensional task, supervision is restricted to the stored high-symmetry k-points. Interior points along the corresponding paths are reserved for evaluation of reciprocalspace generalization. Band-energy errors are evaluated within PBE-side energy windows of 2, 5, 10, and 20 eV around the band-edge midpoint.

Electronic-structure data The baseline orbital quantities are obtained from atom-centered electronic-structure calculations in the FHI-aims representation [64] and include the Hamiltonian, overlap, density, and spin-density matrices described below. Orbital preprocessing and the final input representation are summarized in Supplementary Section S2.

## 4.2 Orbital representation

Orbital-resolved input features For each baseline PBE calculation, $_ \mathrm { O - k R y }$ stal uses orbitalresolved quantities derived from an atom-centered basis representation. The input operator set comprises the Hamiltonian H, overlap S, density matrix P, and spin-density matrix $\mathbf { P } _ { \mathrm { s p i n } } .$ These quantities are organized into atom-centered on-site and pairwise blocks and provide a representation of the converged baseline electronic state beyond the PBE eigenvalues alone.

For an atom-centered nonorthogonal basis, the generalized eigenvalue problem at each crystal momentum k is written as

$$
\mathbf { H } ( \mathbf { k } ) \mathbf { C } ( \mathbf { k } ) = \mathbf { S } ( \mathbf { k } ) \mathbf { C } ( \mathbf { k } ) \boldsymbol { \varepsilon } ( \mathbf { k } ) ,\tag{3}
$$

where $\mathbf { C } ( \mathbf { k } )$ contains the generalized eigenvectors and $\varepsilon ( { \bf k } )$ the corresponding PBE eigenvalues. The density matrix is constructed from the occupied eigenstates and subsequently represented in the atomic-orbital basis. Additional orbital-representation and preprocessing details are provided in Supplementary Section S2.

Real-space representation For atomic orbitals $\mu$ and $\nu$ centered on atoms $i$ and $j ,$ respectively, the real-space matrix block associated with lattice translation R is

$$
O _ { i j } ^ { \mu \nu } ( { \mathbf { R } } ) = \left. \phi _ { i } ^ { \mu } ( \mathbf { 0 } ) \middle | \hat { O } \middle | \phi _ { j } ^ { \nu } ( { \mathbf { R } } ) \right. ,\tag{4}
$$

where $\hat { O } \in \{ \hat { H } , \hat { S } , \hat { P } , \hat { P } _ { \mathrm { s p i n } } \}$ . On-site blocks are retained separately, while pairwise blocks encode interactions between orbitals on neighboring periodic images. The neural representation uses atom-centered orbital blocks padded to a maximum dimension of 16 per atom. The real-space representation is organized as a periodic directed atom-image graph. Each retained pair edge carries the corresponding orbital matrix blocks, while on-site blocks are presented separately. The preprocessed pair lists contain up to 32 electronically selected pair blocks per central atom.

The reciprocal-space blocks are associated with the same retained atom-image pairs as the real-space representation, with separate k-dependent on-site blocks.

Reciprocal-space representation The corresponding reciprocal-space representation is related to the real-space matrix elements through the Bloch transform,

$$
O _ { i j } ^ { \mu \nu } ( { \bf k } ) = \sum _ { \bf R } e ^ { i { \bf k } \cdot { \bf R } } O _ { i j } ^ { \mu \nu } ( { \bf R } ) .\tag{5}
$$

In the final model, the real components of H(k), S(k), P(k), and $\mathbf { P } _ { \mathrm { s p i n } } ( \mathbf { k } )$ are used as reciprocalspace orbital features.

Bloch-phase conditioning k dependence is additionally introduced on each periodic pair through the full atom-to-image displacement,

$$
\mathbf { r } _ { i j \mathbf { R } } = \mathbf { r } _ { j } - \mathbf { r } _ { i } + \mathbf { R } ,\tag{6}
$$

using

$$
\left[ \sin ( \mathbf { k } \cdot \mathbf { r } _ { i j \mathbf { R } } ) , \cos ( \mathbf { k } \cdot \mathbf { r } _ { i j \mathbf { R } } ) \right] .\tag{7}
$$

These phase factors condition the local representation on the queried crystal momentum. The full O-kRystal representation combines the real-space orbital blocks, Bloch-phase conditioning, and explicit reciprocal-space orbital features. A corresponding real-space + phase ablation removes the explicit reciprocal-space matrices retaining the remaining inputs unchanged.

## 4.3 O-kRystal architecture

Periodic equivariant message passing O-kRystal is built on a periodic E(3)-equivariant graph neural network that processes the orbital blocks associated with atoms and periodic atom pairs. On-site orbital blocks initialize atom-centered features, while pairwise blocks contribute to equivariant message passing through periodic relative displacements and spherical-harmonic geometric features. Atomic species are incorporated through learned element embeddings, and the resulting atom-wise representations are pooled into a material- and k-dependent latent representation.

The final model comprises two jointly trained branches: a real-space + Bloch-phase afine branch and a dual-space residual branch that additionally receives explicit reciprocal-space orbital features. The afine branch inherits four equivariant interaction layers and four decoding layers from the model defaults, whereas the residual branch explicitly uses two interaction and two decoding layers. Real-space orbital features and Bloch-phase information enter the afine branch, while explicit reciprocal-space orbital features provide an additional k-resolved residual contribution. Architectural and training hyperparameters are summarized in Supplementary Section S2 and Supplementary Table S2

For the representation ablation, we additionally train a Real+Bloch-phase variant that retains the real-space orbital blocks and Bloch-phase conditioning while omitting the explicit reciprocalspace operator matrices. All other training and evaluation settings are matched to the full model (Supplementary Fig. S1).

Band-query representation Rather than predicting a fixed-dimensional band vector, OkRystal represents each electronic state through a signed band query defined relative to the band edges. Valence states are assigned negative indices and conduction states positive indices, allowing a variable number of valence and conduction bands to be queried across materials.

Each band query is combined with five deterministic descriptors derived from the corresponding PBE band over the retained high-symmetry-point set: the mean energy, standard deviation, minimum energy, maximum energy, and band width. Using the PBE band-edge reference

$$
E _ { \mathrm { m i d } } ^ { \mathrm { P B E } } = \frac { 1 } { 2 } \left[ \underset { \mathbf { k } } { \operatorname* { m a x } } \varepsilon _ { \mathrm { V B 1 , \mathbf { k } } } ^ { \mathrm { P B E } } + \underset { \mathbf { k } } { \operatorname* { m i n } } \varepsilon _ { \mathrm { C B 1 , \mathbf { k } } } ^ { \mathrm { P B E } } \right] ,
$$

the descriptor vector for band n is

$$
\mathbf { f } _ { n } = \left[ \frac { \mu _ { n } - E _ { \mathrm { m i d } } ^ { \mathrm { P B E } } } { 1 0 } , \frac { \sigma _ { n } } { 5 } , \frac { \varepsilon _ { n , \mathrm { m i n } } ^ { \mathrm { P B E } } - E _ { \mathrm { m i d } } ^ { \mathrm { P B E } } } { 1 0 } , \frac { \varepsilon _ { n , \mathrm { m a x } } ^ { \mathrm { P B E } } - E _ { \mathrm { m i d } } ^ { \mathrm { P B E } } } { 1 0 } , \frac { \varepsilon _ { n , \mathrm { m a x } } ^ { \mathrm { P B E } } - \varepsilon _ { n , \mathrm { m i n } } ^ { \mathrm { P B E } } } { 5 } \right] ,
$$

with all input energies in $\mathrm { e V }$ . These descriptors are supplied to the afine decoder with the signed-band embedding.

The descriptors are computed over valid high-symmetry points when available. If no retained high-symmetry point is present for a sample, all valid retained k-points are used.

Afine band correction and dual-space residual The state-resolved correction is decomposed into an afine band contribution and a k-dependent residual. For band n, the afine contribution is written as

$$
\begin{array} { r } { \widehat { \Delta \varepsilon } _ { n \mathbf { k } } ^ { \mathrm { a f f } } = c _ { n } + b _ { n } \left( \varepsilon _ { n \mathbf { k } } ^ { \mathrm { P B E } } - \overline { { \varepsilon } } _ { n } ^ { \mathrm { P B E } } \right) , } \end{array}\tag{8}
$$

where $c _ { n }$ and $b _ { n }$ are predicted from the pooled material representation and band-query features, and $\overline { { \varepsilon } } _ { n } ^ { \mathrm { P B E } }$ denotes the mean PBE band energy over the available high-symmetry points. This term captures band-dependent shifts and changes in dispersion relative to PBE.

A second branch predicts a pointwise residual $r _ { n \mathbf { k } }$ from the combined real- and reciprocal-space orbital representation. To avoid redundancy with the afine contribution, the residual is projected at the supervised high-symmetry points to remove its mean and its component linear in the PBE dispersion, yielding

$$
r _ { n \mathbf k } ^ { \perp } = r _ { n \mathbf k } - \overline { { r } } _ { n } - a _ { n } \left( \varepsilon _ { n \mathbf k } ^ { \mathrm { P B E } } - \overline { { \varepsilon } } _ { n } ^ { \mathrm { P B E } } \right) ,\tag{9}
$$

where $\overline { { r } } _ { n }$ is the mean residual over the supervised high-symmetry points and $a _ { n }$ is the leastsquares coeficient of the residual along the centered PBE dispersion. The final correction is then

$$
\widehat { \Delta \varepsilon } _ { n \mathbf { k } } = \widehat { \Delta \varepsilon } _ { n \mathbf { k } } ^ { \mathrm { a f f } } + r _ { n \mathbf { k } } ^ { \perp } .\tag{10}
$$

By construction, the afine branch captures band-dependent shifts and changes correlated with the PBE dispersion, while the orthogonalized residual branch represents the remaining k-dependent correction beyond these components.

The target-theory energy is reconstructed as

$$
\widehat { \varepsilon } _ { n \mathbf { k } } ^ { \mathrm { t a r g e t } } = \varepsilon _ { n \mathbf { k } } ^ { \mathrm { P B E } } + \widehat { \Delta \varepsilon } _ { n \mathbf { k } } .\tag{11}
$$

Both branches are optimized jointly. The model predicts state energies only and does not predict target-theory wavefunctions or orbital-projection coeficients.

## 4.4 Learning objective and optimization

State-resolved ∆-learning O-kRystal is trained to predict the state-resolved correction between the baseline and target electronic-structure methods,

$$
\begin{array} { r } { \Delta \varepsilon _ { n \mathbf { k } \sigma } = \varepsilon _ { n \mathbf { k } \sigma } ^ { \mathrm { t a r g e t } } - \varepsilon _ { n \mathbf { k } \sigma } ^ { \mathrm { P B E } } . } \end{array}\tag{12}
$$

For the primary three-dimensional task, the target theory is HSE06. PBE and HSE06 states are paired by their stored band ordering relative to the band edges, and unmatched or padded states are excluded from the loss.

Training objective Let $m , n ,$ k, and σ index the band-structure entry, band, crystal momentum, and spin channel, respectively. For each valid PBE–target state pair, we define the prediction residual as $e _ { m n \mathbf { k } \sigma } = \hat { \Delta { \varepsilon } } _ { m n \mathbf { k } \sigma } - \Delta { \varepsilon } _ { m n \mathbf { k } \sigma }$ and penalize it using a SmoothL1 loss with transition parameter $\kappa = 0 . 1 \ \mathrm { e V }$

To emphasize states near the PBE reference energy while retaining supervision over the broader electronic spectrum, each valid state is assigned the weight

$$
w _ { m n \mathbf { k } \sigma } = \exp \left[ - \left( \frac { \varepsilon _ { m n \mathbf { k } \sigma } ^ { \mathrm { P B E } } - E _ { \mathrm { m i d } } ^ { \mathrm { P B E } } } { E _ { w } } \right) ^ { 2 } \right] , \qquad E _ { w } = 1 0 ~ \mathrm { e V } .\tag{13}
$$

The state-energy loss is the normalized weighted SmoothL1 error,

$$
\mathcal { L } _ { \mathrm { s t a t e } } = \frac { \displaystyle \sum _ { m , n , \mathbf { k } , \sigma ) \in \mathcal { V } } w _ { m n \mathbf { k } \sigma } \rho _ { \kappa } ( e _ { m n \mathbf { k } \sigma } ) } { \displaystyle \sum _ { ( m , n , \mathbf { k } , \sigma ) \in \mathcal { V } } w _ { m n \mathbf { k } \sigma } } ,\tag{14}
$$

where V denotes the set of valid matched states in the training batch and $\rho _ { \kappa }$ denotes the SmoothL1 penalty.

To penalize systematic bandwise shifts, we additionally apply an ofset loss to the mean signed residual of each band over the supervised high-symmetry points,

$$
\mathcal { L } _ { \mathrm { o f f s e t } } = \frac { 1 } { | \mathcal { B } | } \sum _ { ( m , n , \sigma ) \in \mathcal { B } } \rho _ { \kappa } \left( \frac { 1 } { N _ { m n \sigma } ^ { \mathrm { H S } } } \sum _ { \mathbf { k } \in \mathrm { H S } } e _ { m n \mathbf { k } \sigma } \right) ,\tag{15}
$$

where $\boldsymbol { B }$ is the set of valid entry–band–spin combinations and $N _ { m n \sigma } ^ { \mathrm { H S } }$ is the number of supervised high-symmetry points for each combination.

An indirect-gap loss additionally penalizes errors in the reconstructed fundamental gap,

$$
\mathcal { L } _ { \mathrm { g a p } } = \frac { 1 } { \vert \mathcal { G } \vert } \sum _ { m \in \mathcal { G } } \rho _ { \kappa } \Bigl ( \widehat { E } _ { g , m } - E _ { g , m } ^ { \mathrm { t a r g e t } } \Bigr ) ,\tag{16}
$$

where $\widehat { E } _ { g , m }$ and $E _ { g , m } ^ { \mathrm { t a r g e t } }$ are the predicted and reference fundamental gaps obtained from the corresponding valence- and conduction-band extrema, and $\mathcal { G }$ contains entries for which both extrema are valid.

The final training objective is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { s t a t e } } + \mathcal { L } _ { \mathrm { o f f s e t } } + 0 . 2 \mathcal { L } _ { \mathrm { g a p } } .\tag{17}
$$

All other auxiliary objectives, including direct-gap, dispersion, curvature, ranking, and efectivemass terms, have zero weight in the production model. The state-energy correction targets are expressed directly in eV without standardization. Additional training settings are summarized in Supplementary Section S2 and Supplementary Table S2.

Optimization and checkpoint selection Models are optimized with AdamW using an initial learning rate of $5 \times 1 0 ^ { - 4 }$ , weight decay of 0.005, and momentum parameters $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 5 )$ Training is performed for 100 epochs with a 20-epoch linear warm-up followed by cosine learningrate decay to $1 0 ^ { - 5 }$ . Training uses dynamic batching with between one and twelve entries per batch, subject to a maximum of 700 atoms. Additional pair- and k-pair padding limits are imposed to control memory usage during training.

Three independently trained models are used for the principal three-dimensional results, with seeds 42, 43, and 44. Within each seed, the learning-curve subsets are nested while the validation and test populations remain fixed across training-set sizes. The model checkpoint used for evaluation is selected by the lowest unweighted mean validation MAE across the 2, 5, 10, and 20 eV energy windows.

## 4.5 Baseline models

Scissor and Linear+KRR baselines We compare O-kRystal against progressively more expressive corrections to the PBE band energies. The scissor baseline applies separate global mean corrections to valence- and conduction-side states, estimated from the training set. The Linear+KRR baseline uses six PBE-derived state and material descriptors in a linear model followed by RBF kernel-ridge regression of the remaining residual. For the primary comparison, both baselines are fitted using only the high-symmetry-point states of the corresponding training partition and evaluated on the matched high-symmetry-point test subset using the same PBEside reference and energy-window conventions as O-kRystal. Implementation details are provided in Supplementary Section S2 and Supplementary Table S3.

Geometry GNN baseline To compare against a geometry-based correction supplied with the PBE band baseline, we train a graph neural network using the same seed-specific data splits and state-resolved prediction targets. The model receives atomic species, periodic pair distances, Bloch-phase features, and the PBE band energies and descriptors used by the band decoder, but no orbital matrices. Unlike the electronically preselected pair graph used by the orbital models, the Geometry GNN constructs its periodic graph directly from atomic positions, retaining up to 32 nearest atom-image neighbors per central atom by distance. Its periodic message-passing network therefore learns the PBE-to-HSE06 correction from geometric information together with the baseline band structure. The Geometry GNN inputs and their distinction from the orbital model are summarized in Supplementary Section S2 and Supplementary Table S3.

Real-space + Bloch-phase ablation To quantify the contribution of explicit reciprocalspace orbital information, we compare the full dual-space O-kRystal representation with a Real+Bloch-phase variant that retains the real-space orbital blocks, atomic structure, PBE band information, band-query representation, and Bloch-phase conditioning while omitting the explicit reciprocal-space orbital matrices. Both representations are trained using the same optimization settings, loss function, checkpoint-selection criterion, seed-specific data splits, and training-set sizes. The comparison is evaluated across approximately 0.5k, 2.0k, 3.5k, and 5.5k training entries for seeds 42, 43, and 44. Results are reported in Supplementary Section S3 and Supplementary Figure S1.

## 4.6 Evaluation

Band-energy errors State-resolved prediction errors are evaluated as the mean absolute error between predicted and target-theory band energies over valid matched states. For a band-structure entry m and energy window W

$$
\mathrm { M A E } _ { m } ( W ) = \frac { 1 } { | V _ { m } ( W ) | } \sum _ { ( n , \mathbf { k } ) \in V _ { m } ( W ) } \left| \widehat { \varepsilon } _ { m n \mathbf { k } } - \varepsilon _ { m n \mathbf { k } } ^ { \mathrm { t a r g e t } } \right| ,\tag{18}
$$

where $V _ { m } ( W )$ denotes the set of valid states satisfying the corresponding PBE-side energy-window criterion

$$
| E ^ { \mathrm { P B E } } - E _ { \mathrm { m i d } } ^ { \mathrm { P B E } } | < W .
$$

Entries for which $V _ { m } ( W )$ is empty are excluded from the corresponding window average for all methods. Unless otherwise stated, reported three-dimensional band-energy errors are averaged over band-structure entries, with spin channels treated as separate entries. For the main HSE06 benchmark, $W \in \{ 2 , 5 , 1 0 , 2 0 \} \ \mathrm { e V }$

Evaluation checkpoint The primary benchmark in Fig. 2 reports results across three separately trained models using distinct parent-level data splits. Unless otherwise stated, the detailed reciprocal-space and downstream analyses in Figs. 3, 4, and 5 use the provided seed-42 best\_avg\_win checkpoint.

Reciprocal-space generalization For the three-dimensional HSE06 task, O-kRystal is supervised only at the stored high-symmetry k-points. Reciprocal-space generalization is evaluated on fixed query sets containing previously unseen path-interior points, using the same state-matching and PBE-side energy-window conventions as the primary band-energy evaluation. Individual predictions are therefore conditioned on the supplied k-point set rather than interpreted as query-set-independent single-point predictions. Path-interior diagnostics, including the diference between interior-point and high-symmetry-point errors and the dependence on reciprocal-space distance, relative segment position, and segment length, are reported in Supplementary Section S4 and Supplementary Fig. S2. Representative full-path predictions are shown in Supplementary Fig. S3.

Density of states reconstruction To assess predictions away from conventional highsymmetry paths, O-kRystal is evaluated on dense, material-specific regular k-point meshes throughout the Brillouin zone. Predicted state energies are used to reconstruct the density of states using tetrahedron integration followed by Gaussian broadening with $\sigma = 0 . 1 0 \ \mathrm { e V }$ on a common energy grid from −10 to 10 $\mathrm { e V }$

The DOS reconstruction error within $W = 5$ eV of the band-edge midpoint is quantified using the normalized $L _ { 1 }$ diference

$$
D _ { m } = \frac { \int _ { - W } ^ { W } \left| g _ { m } ( E ) - g _ { m } ^ { \mathrm { H S E 0 6 } } ( E ) \right| d E } { \int _ { - W } ^ { W } g _ { m } ^ { \mathrm { H S E 0 6 } } ( E ) d E } ,\tag{19}
$$

where $g _ { m } ( E )$ and $g _ { m } ^ { \mathrm { H S E 0 6 } } ( E )$ denote the predicted and reference densities of states, respectively. Spin-resolved contributions are combined at the parent-material level before evaluating the DOS error. This metric evaluates aggregate spectral agreement over the energy window rather than pointwise correspondence of individual of-path eigenstates.

Electronic-character classification Electronic character is determined from the indirect band gap of each band-structure entry. An entry is classified as insulating when the corresponding physical gap is positive and metallic otherwise; the predefined odd-electron metallicity convention is applied to unpolarized entries. Spin channels are classified separately, while a parent material is metallic if any constituent spin channel is metallic. We categorize the baseline-to-target change as insulator-to-insulator, metal-to-insulator, metal-to-metal, or insulator-to-metal, and compare the O-kRystal classification with the corresponding HSE06 reference.

## 4.7 Downstream benchmarks

Correlated-material benchmark and band-energy evaluation A separate literaturecurated benchmark is used to evaluate O-kRystal in materials with localized and correlationsensitive electronic structure. Six parent materials are retained after requiring consistency between the represented magnetic and electronic state and the corresponding literature-supported phase. The cohort comprises the correlated ferromagnetic metal $\mathrm { C r O _ { 2 } \ [ 5 1 , 5 2 ] }$ , low-spin $\mathrm { L a C o O _ { 3 } }$ [53, 54], correlated metallic $\mathrm { L a N i O _ { 3 } }$ [55, 56], AFM-II MnO and NiO [57–60], and ferromagnetic Mott-insulating $\mathrm { Y T i O _ { 3 } } \left[ 6 1 , 6 2 \right]$ . All six materials are excluded before construction of the primary training, validation, and test splits.

Band-energy errors are evaluated using the representative seed-42 model on dense, materialspecific Brillouin-zone meshes. For material $m ,$ , the mean absolute error within an energy window W is

$$
\mathrm { M A E } _ { m } ( W ) = \frac { 1 } { | \mathcal { V } _ { m } ( W ) | } \sum _ { ( n , \mathbf { k } , \sigma ) \in \mathcal { V } _ { m } ( W ) } \left| \hat { \varepsilon } _ { m n \mathbf { k } \sigma } - \varepsilon _ { m n \mathbf { k } \sigma } ^ { \mathrm { H S E 0 6 } } \right| ,\tag{20}
$$

where $\nu _ { m } ( W )$ contains the retained states satisfying

$$
\left| \varepsilon _ { m n \mathbf { k } \sigma } ^ { \mathrm { P B E } } - E _ { \mathrm { m i d } , m } ^ { \mathrm { P B E } } \right| \leq W .\tag{21}
$$

The main-text comparison uses $W = 5 \mathrm { e V }$ , while $W = 2 0 \ \mathrm { e V }$ is used as a wider-window evaluation. No reciprocal-space quadrature weights are applied to the band-energy MAE. For spin-polarized materials, retained states from both spin channels are pooled before computing the material-level error. Aggregate values are material balanced by assigning equal weight to each of the six parent materials.

Orbital-resolved projected density of states Orbital-resolved electronic structure is evaluated for the transition-metal 3d and oxygen $2 p$ channels. To isolate diferences in stateenergy placement, all projected densities of states are constructed using the same Mulliken orbital weights obtained from the baseline PBE wavefunctions. For a set of state energies $\varepsilon _ { m } ^ { X }$ nkσ from method X,

$$
g _ { \alpha , m } ^ { X } ( E ) = \sum _ { n \mathbf { k } \sigma } w _ { \mathbf { k } } p _ { \alpha , m n \mathbf { k } \sigma } ^ { \mathrm { P B E } } \delta \Bigl ( E - \varepsilon _ { m n \mathbf { k } \sigma } ^ { X } \Bigr ) ,\tag{22}
$$

where α denotes the orbital channel, $p _ { \alpha , m n \mathbf { k } \sigma } ^ { \mathrm { P B E } }$ is the corresponding PBE Mulliken projection weight, and $w _ { \mathbf { k } }$ is the reciprocal-space integration weight. For PBE, Geometry GNN, O-kRystal, and HSE06, only the state energies entering the reconstruction difer. The PDOS is evaluated on dense reciprocal-space meshes using linear tetrahedron integration and subsequently broadened with a Gaussian of width $\sigma = 0 . 1 0 \ \mathrm { e V }$ . This construction therefore compares orbital-resolved energy distributions at fixed orbital character and does not evaluate target-theory changes in wavefunctions or hybridization.

Agreement with the HSE06 energy distribution is quantified using the normalized $L _ { 1 }$ error

$$
D _ { \alpha , m } = \frac { \displaystyle \int _ { - 5 \mathrm { e V } } ^ { 5 \mathrm { e V } } \left. g _ { \alpha , m } ^ { \mathrm { m e t h o d } } ( E ) - g _ { \alpha , m } ^ { \mathrm { H S E 0 6 } } ( E ) \right. d E } { \displaystyle \int _ { - 5 \mathrm { e V } } ^ { 5 \mathrm { e V } } \left. g _ { \alpha , m } ^ { \mathrm { H S E 0 6 } } ( E ) \right. d E } ,\tag{23}
$$

where energies are referenced to $E _ { \mathrm { m i d } }$ . The metric is dimensionless, and aggregate values are obtained by averaging the six material-level errors with equal weight.

Experimental band-gap benchmark Experimental comparisons use the mstar60 benchmark [47]. The primary bulk cohort contains ten materials for which the required PBE, HSE06, and experimental quantities are available and whose constituent elements are represented in the three-dimensional training set. Monolayer systems are excluded from the bulk comparison, Pb chalcogenides are excluded from the primary benchmark because Pb is absent from the training chemistry, and black phosphorus is excluded because the required PBE efective masses are unavailable. The benchmark composition and inclusion criteria are summarized in Supplementary Section S6 and Supplementary Table S7.

Gap-informed efective-mass reconstruction Efective masses are not predicted directly from O-kRystal band curvature. Instead, for observation $i ,$ the carrier efective mass is predicted through the reconstruction

$$
\begin{array} { r } { \hat { m } _ { i } ^ { * } = m _ { i , \mathrm { P B E } } ^ { * } r _ { i } ^ { \beta _ { i } } , } \end{array}\tag{24}
$$

where $\hat { m } _ { i } ^ { * }$ is the predicted carrier efective mass, $m _ { i , \mathrm { P B E } } ^ { * }$ is the PBE efective mass for the same carrier branch, band extremum, and crystallographic direction as observation $i , r _ { i }$ is a regularized ratio between the O-kRystal-predicted and PBE fundamental band gaps and $\beta _ { i }$ is a mass-dependent exponent whose parameters are calibrated without the held-out material family.

For this downstream analysis, the PBE and predicted HSE06 fundamental gaps are evaluated over the complete stored band path as the conduction-band minimum minus the valence-band maximum. The reconstruction is evaluated for 31 directional efective-mass observations across ten materials, grouped into III–V semiconductors, chalcogenides, and other semiconductors. To assess transfer across chemical families, each family is held out in turn while the reconstruction parameters are calibrated using HSE06 efective masses from the remaining families. For evaluation, the absolute percentage errors are first averaged over all retained efective-mass observations belonging to each material, and the resulting material-level errors are averaged equally across materials. Reported family and overall MAPE values are therefore material balanced. Experimental masses are excluded from calibration and are used only for final evaluation. The exact regularization, calibration procedure, family-specific parameters, benchmark composition, and all 31 efective-mass observations are provided in Supplementary Section S6 and Supplementary Tables S7 and S8.

## 4.8 $G _ { 0 } W _ { 0 }$ quasiparticle benchmark

Two-dimensional quasiparticle dataset We evaluate the same orbital learning framework on a two-dimensional $G _ { 0 } W _ { 0 }$ quasiparticle benchmark derived from C2DB [38, 39, 41]. The final cohort contains 286 nonmagnetic semiconductor records with quasiparticle energies for the eight highest valence states and four lowest conduction states. The target quantities are $G _ { 0 } W _ { 0 }$ quasiparticle energies constructed from GPAW PBE starting points. We compare two PBE eigenvalue baselines: the corresponding GPAW spectrum and an independently calculated FHI-aims spectrum [63, 64]. The $G _ { 0 } W _ { 0 }$ labels, k-point sets, and real- and reciprocal-space orbital inputs are held fixed between the two variants. Changing the PBE eigenvalues consequently changes the ∆-learning targets. Separate models are trained from scratch for the two baselines using otherwise matched settings.

The $G _ { 0 } W _ { 0 }$ task is trained separately from the three-dimensional HSE06 model while retaining the same underlying orbital learning architecture and state-resolved ∆-learning formulation. Unlike the primary three-dimensional task, the $G _ { 0 } W _ { 0 }$ training queries include the available high-symmetry points together with up to nine evenly spaced stored interior points per path segment. Five-fold cross-validation is performed with separate training, validation, and test folds. For each outer fold, three source folds are used for training, one fold is used for validation, and the remaining fold is evaluated once as held-out test data. Additional cohort and cross-validation details are provided in Supplementary Section S7 and Supplementary Table S9.

Quasiparticle learning and evaluation For each matched state,

$$
\begin{array} { r } { \Delta \varepsilon _ { n \mathbf { k } } ^ { G W } = \varepsilon _ { n \mathbf { k } } ^ { G _ { 0 } W _ { 0 } } - \varepsilon _ { n \mathbf { k } } ^ { \mathrm { P B E } } , } \end{array}\tag{25}
$$

and the quasiparticle energy is reconstructed by adding the predicted correction to the corresponding PBE baseline. State matching follows the fixed valence- and conduction-band ordering used in the benchmark.

Performance is reported as the mean absolute error over the twelve available quasiparticle states and as the error in the resulting fundamental band gap. Results are averaged across the five folds, with uncertainties reported as the sample standard deviation. To assess sensitivity to the baseline electronic structure, the same $G _ { 0 } W _ { 0 }$ target is learned separately from GPAW and FHI-aims PBE inputs [63, 64]. State-resolved diferences between the two PBE baselines are additionally evaluated after independent energy alignment to characterize where baseline inconsistency is largest across the valence and conduction manifolds. State-resolved errors across the valence and conduction manifolds are shown in Supplementary Fig. S4, with fold-resolved state and band-gap errors reported in Supplementary Table S9.

Data Availability All datasets used in this study will be made available upon publication.

Code Availability All code used in this study will be made available upon publication.

Acknowledgments A.A. acknowledges support in part from the Bren Endowed Chair, the Ofice of Naval Research through MURI grant N00014-18-1-2624, DARPA ExpMath (HR0011), and the AI2050 Senior Fellow program at Schmidt Sciences. W.A.G. acknowledges support from the Liquid Sunlight Alliance, which is supported by the U.S. Department of Energy, Ofice of Science, Ofice of Basic Energy Sciences, Fuels from Sunlight Hub under Award Number DE-SC0021266, and from the Hong Kong Quantum AI Lab, AIR@InnoHK of the Hong Kong. M.K. acknowledges support through the Caltech Summer Undergraduate Research Fellowship. B.S.K. acknowledges support from the Pritzker AI+Science Fund, DARPA Biological Technologies (HR0011), and the Eddleman Graduate Fellowship. D.K. acknowledges support from the Pritzker AI+Science Fund and DARPA Biological Technologies (HR0011). Authors acknowledge Maurice Hanisch, Abdullah Maruf, and Sergei Kliavinek for helpful discussions. Authors also acknowledge Akhil Sugathan Nair for providing the HSE06 dataset used in the work, and Jens Jørgen Mortensen and Kristian Sommer Thygesen for providing the C2DB $G _ { 0 } W _ { 0 }$ dataset. This work used the Delta system at the National Center for Supercomputing Applications through allocation DMR160114 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296.

We acknowledge the use of large language models (LLMs) for initial brainstorming, assistance with handling public datasets, development of analysis pipelines around publicly available packages, and support with figure design, visualization, and figure-generation workflows. LLMs were not used in the development of the core model. All implementations, analyses, figures, and reported results were independently checked and validated by the authors.

## References

[1] Neil W. Ashcroft and N. David Mermin. Solid State Physics. Holt, Rinehart and Winston, New York, 1976.

[2] John C Slater and George F Koster. Simplified lcao method for the periodic potential problem. Physical review, 94(6):1498, 1954.

[3] Jiawei Zhou and Gang Chen. Material insights on electronic transport of charge and heat from first principles. Nature Reviews Physics, 8:291–305, 2026.

[4] Lucy D Whalley, Jarvist M Frost, Benjamin J Morgan, and Aron Walsh. Impact of nonparabolic electronic band structure on the optical and transport properties of photovoltaic materials. Physical review B, 99(8):085207, 2019.

[5] Yanzhong Pei, Xiaoya Shi, Aaron LaLonde, Heng Wang, Lidong Chen, and G. Jefrey Snyder. Convergence of electronic bands for high performance bulk thermoelectrics. Nature, 473:66–69, 2011.

[6] Junsoo Park, Maxwell Dylla, Yi Xia, Max Wood, G. Jefrey Snyder, and Anubhav Jain. When band convergence is not beneficial for thermoelectrics. Nature Communications, 12:3425, 2021.

[7] Liping Yu and Alex Zunger. Identification of potential photovoltaic absorbers based on first-principles spectroscopic screening of materials. Physical Review Letters, 108:068701, 2012.

[8] Prashun Gorai, Vladan Stevanović, and Eric S. Toberer. Computationally guided discovery of thermoelectric materials. Nature Reviews Materials, 2:17053, 2017.

[9] Haijun Zhang, Chao-Xing Liu, Xiao-Liang Qi, Xi Dai, Zhong Fang, and Shou-Cheng Zhang. Topological insulators in bi2se3, bi2te3 and sb2te3 with a single dirac cone on the surface. Nature physics, 5(6):438–442, 2009.

[10] Haifeng Yang, Aiji Liang, Chaofan Chen, Chaofan Zhang, Niels B. M. Schroeter, et al. Visualizing electronic structures of quantum materials by angle-resolved photoemission spectroscopy. Nature Reviews Materials, 3:341–353, 2018.

[11] Baiqing Lv, Tian Qian, and Hong Ding. Angle-resolved photoemission spectroscopy and its application to topological materials. Nature Reviews Physics, 1:609–626, 2019.

[12] R. Patrick Xian, Vincent Stimper, Marios Zacharias, Maciej Dendzik, Shuo Dong, Samuel Beaulieu, Bernhard Schölkopf, Martin Wolf, Laurenz Rettig, Christian Carbogno, Stefan Bauer, and Ralph Ernstorfer. A machine learning route between band mapping and band structure. Nature Computational Science, 3:101–114, 2023.

[13] Pierre Hohenberg and Walter Kohn. Inhomogeneous electron gas. Physical Review, 136(3B):B864–B871, 1964.

[14] Walter Kohn and Lu Jeu Sham. Self-consistent equations including exchange and correlation efects. Physical Review, 140(4A):A1133–A1138, 1965.

[15] John P. Perdew, Kieron Burke, and Matthias Ernzerhof. Generalized gradient approximation made simple. Physical Review Letters, 77(18):3865–3868, 1996.

[16] John P Perdew and Mel Levy. Physical content of the exact kohn-sham orbital energies: band gaps and derivative discontinuities. Physical Review Letters, 51(20):1884, 1983.

[17] Paula Mori-Sánchez, Aron J. Cohen, and Weitao Yang. Localization and delocalization errors in density functional theory and implications for band-gap prediction. Physical Review Letters, 100(14):146401, 2008.

[18] Jason M Crowley, Jamil Tahir-Kheli, and William A Goddard III. Resolution of the band gap prediction problem for materials design. The journal of physical chemistry letters, 7(7):1198–1203, 2016.

[19] Jochen Heyd, Gustavo E. Scuseria, and Matthias Ernzerhof. Hybrid functionals based on a screened coulomb potential. The Journal of Chemical Physics, 118(18):8207–8215, 2003.

[20] Aliaksandr V. Krukau, Oleg A. Vydrov, Artur F. Izmaylov, and Gustavo E. Scuseria. Influence of the exchange screening parameter on the performance of screened hybrid functionals. The Journal of Chemical Physics, 125(22):224106, 2006.

[21] Mohan Liu, Abhijith Gopakumar, Vinay Ishwar Hegde, Jiangang He, and Chris Wolverton. High-throughput hybrid-functional dft calculations of bandgaps and formation energies and multifidelity learning with uncertainty quantification. Physical Review Materials, 8(4):043803, 2024.

[22] Lars Hedin. New method for calculating the one-particle green’s function with application to the electron-gas problem. Physical Review, 139(3A):A796–A823, 1965.

[23] Dorothea Golze, Marc Dvorak, and Patrick Rinke. The gw compendium: A practical guide to theoretical photoemission spectroscopy. Frontiers in chemistry, 7:377, 2019.

[24] Matthias Redies, Gregor Michalicek, Juba Bouaziz, Christian Terboven, Matthias S. Müller, Stefan Blügel, and Daniel Wortmann. Fast all-electron hybrid functionals and their application to rare-earth iron garnets. Frontiers in Materials, 9:851458, 2022.

[25] Max Großmann, Malte Grunert, and Erich Runge. A robust, simple, and eficient convergence workflow for GW calculations. npj Computational Materials, 10:135, 2024.

[26] Zechen Tang, Hao Chen, Yibo Li, et al. Deep-learning electronic structure calculations. Nature Computational Science, 5:1133–1146, 2025.

[27] Tian Xie and Jefrey C. Grossman. Crystal graph convolutional neural networks for an accurate and interpretable prediction of material properties. Physical Review Letters, 120(14):145301, 2018.

[28] Kamal Choudhary and Brian DeCost. Atomistic line graph neural network for improved materials property predictions. npj Computational Materials, 7(1):185, 2021.

[29] Ya Zhuo, Aria Mansouri Tehrani, and Jakoah Brgoch. Predicting the band gaps of inorganic solids by machine learning. The Journal of Physical Chemistry Letters, 9(7):1668–1673, 2018.

[30] Weiyi Gong, Tao Sun, Hexin Bai, Jeng-Yuan Tsai, Haibin Ling, and Qimin Yan. Graph transformer networks for accurate band structure prediction: An end-to-end approach, 2024.

[31] He Li, Zun Wang, Nianlong Zou, Meng Ye, Runzhang Xu, Xiaoxun Gong, Wenhui Duan, and Yong Xu. Deep-learning density functional theory hamiltonian for eficient ab initio electronic-structure calculation. Nature Computational Science, 2:367–377, 2022.

[32] Xiaoxun Gong, He Li, Nianlong Zou, Runzhang Xu, Wenhui Duan, and Yong Xu. General framework for e (3)-equivariant neural network representation of density functional theory hamiltonian. Nature Communications, 14(1):2848, 2023.

[33] He Li, Zechen Tang, Xiaoxun Gong, Nianlong Zou, Wenhui Duan, and Yong Xu. Deeplearning electronic-structure calculation of magnetic superstructures. Nature Computational Science, 3:321–327, 2023.

[34] Zechen Tang, He Li, Peize Lin, Xiaoxun Gong, Gan Jin, Lixin He, Hong Jiang, Xinguo Ren, Wenhui Duan, and Yong Xu. A deep equivariant neural network approach for eficient hybrid density functional calculations. Nature Communications, 15(1):8815, 2024.

[35] Xiaoxun Gong, Steven G. Louie, Wenhui Duan, and Yong Xu. Generalizing deep learning electronic structure calculation to the plane-wave basis. Nature Computational Science, 4:752–760, 2024.

[36] Danish Khan, Maurice D. Hanisch, Nikolai Argatof, Evan Xie, Sandeep Sharma, and Anima Anandkumar. Learning the kohn–sham map with neural operators for quasi-linear scaling density functional theory, 2026.

[37] Akhil S Nair, Lucas Foppa, and Matthias Schefler. Materials database from all-electron hybrid functional dft calculations. Scientific Data, 12(1):1518, 2025.

[38] S. Haastrup, M. Strange, M. Pandey, T. Deilmann, P. S. Schmidt, N. F. Hinsche, M. N. Gjerding, D. Torelli, P. M. Larsen, A. C. Riis-Jensen, J. Gath, K. W. Jacobsen, J. J. Mortensen, T. Olsen, and K. S. Thygesen. The computational 2d materials database: highthroughput modeling and discovery of atomically thin crystals. 2D Materials, 5(4):042002, 2018.

[39] Morten N. Gjerding, Alireza Taghizadeh, Anders Rasmussen, Shabbir Ali, Federico Bertoldo, Thorsten Deilmann, Uriel P. Holguin, N. R. Knøsgaard, Matthias Kruse, Anders H. Larsen, Simone Manti, Thomas G. Pedersen, Troels Skovhus, Mikkel K. Svendsen, Jens Jørgen Mortensen, Thomas Olsen, and Kristian S. Thygesen. Recent progress of the computational 2d materials database (c2db). 2D Materials, 8(4):044002, 2021.

[40] Santosh Adhikari, Jacob Clary, Ravishankar Sundararaman, Charles Musgrave, Derek Vigil-Fowler, and Christopher Sutton. Accurate prediction of hse06 band structures for a diverse set of materials using δ-learning. Chemistry of Materials, 35(20):8397–8405, 2023.

[41] Nick R. Knøsgaard and Kristian S. Thygesen. Representing individual electronic states for machine learning GW band structures of 2d materials. Nature Communications, 13:468, 2022.

[42] Zhuoran Qiao, Matthew Welborn, Animashree Anandkumar, Frederick R Manby, and Thomas F Miller. Orbnet: Deep learning for quantum chemistry using symmetry-adapted atomic-orbital features. The Journal of chemical physics, 153(12), 2020.

[43] Anders S Christensen, Sai Krishna Sirumalla, Zhuoran Qiao, Michael B O’Connor, Daniel GA Smith, Feizhi Ding, Peter J Bygrave, Animashree Anandkumar, Matthew Welborn, Frederick R Manby, et al. Orbnet denali: A machine learning potential for biological and organic chemistry with semi-empirical cost and dft accuracy. The Journal of Chemical Physics, 155(20):204103, 2021.

[44] Zhuoran Qiao, Anders S Christensen, Matthew Welborn, Frederick R Manby, Anima Anandkumar, and Thomas F Miller III. Informing geometric deep learning with electronic interactions to accelerate quantum chemistry. Proceedings of the National Academy of Sciences, 119(31):e2205221119, 2022.

[45] Beom Seok Kang, Vignesh C. Bhethanabotla, Amin Tavakoli, Maurice D. Hanisch, William A. Goddard III, and Anima Anandkumar. Orbitall: A unified quantum mechanical representation deep learning framework for all molecular systems, 2025.

[46] Jue-Yi Qi, Xin-Yi Liu, Chuan-Nan Li, Jinshan Li, and Xie Zhang. ARPES-inspired reciprocal-space crystal property predictor. PRX Intelligence, 1:013007, 2026.

[47] Magdalena Laurien and Oleg Rubel. Benchmarking exchange-correlation potentials with the mstar60 dataset: Importance of the nonlocal exchange potential for efective mass calculations in semiconductors. Phys. Rev. B, 106:045204, Jul 2022.

[48] Raghunathan Ramakrishnan, Pavlo O Dral, Matthias Rupp, and O Anatole Von Lilienfeld. Big data meets quantum chemistry approximations: the δ-machine learning approach. Journal of chemical theory and computation, 11(5):2087–2096, 2015.

[49] Vladimir I. Anisimov, Jan Zaanen, and Ole K. Andersen. Band theory and mott insulators: Hubbard u instead of stoner i. Physical Review B, 44(3):943–954, 1991.

[50] Burak Himmetoglu, Andrea Floris, Stefano de Gironcoli, and Matteo Cococcioni. Hubbardcorrected dft energy functionals: The lda+u description of correlated systems. International Journal of Quantum Chemistry, 114(1):14–49, 2014.

[51] T. Tsujioka, T. Mizokawa, J. Okamoto, A. Fujimori, M. Nohara, H. Takagi, K. Yamaura, and M. Takano. Hubbard splitting and electron correlation in the ferromagnetic metal CrO2. Physical Review B, 56:R15509–R15512, 1997.

[52] M. A. Korotin, V. I. Anisimov, D. I. Khomskii, and G. A. Sawatzky. CrO2: A self-doped double exchange ferromagnet. Physical Review Letters, 80:4305–4308, 1998.

[53] T. Saitoh, T. Mizokawa, A. Fujimori, M. Abbate, Y. Takeda, and M. Takano. Electronic structure and temperature-induced paramagnetism in LaCoO3. Physical Review B, 55:4257– 4266, 1997.

[54] M. W. Haverkort, Z. Hu, J. C. Cezar, T. Burnus, H. Hartmann, M. Reuther, C. Zobel, T. Lorenz, A. Tanaka, N. B. Brookes, H. H. Hsieh, H.-J. Lin, C. T. Chen, and L. H. Tjeng. Spin state transition in LaCoO3 studied using soft x-ray absorption spectroscopy and magnetic circular dichroism. Physical Review Letters, 97:176405, 2006.

[55] Daniel G. Ouellette, SungBin Lee, Junwoo Son, Susanne Stemmer, Leon Balents, Andrew J. Millis, and S. James Allen. Optical conductivity of LaNiO3: Coherent transport and correlation-driven mass enhancement. Physical Review B, 82:165112, 2010.

[56] Victoria L. Karner, Aris Chatzichristos, David L. Cortie, Martin H. Dehn, Oleksandr Foyevtsov, Kateryna Foyevtsova, Derek Fujimoto, Robert F. Kiefl, C. D. Philip Levy, Ruohong Li, Ryan M. L. McFadden, Gerald D. Morris, Matthew R. Pearson, Monika Stachura, John O. Ticknor, Georg Cristiani, Gennady Logvenov, Friederike Wrobel, Bernhard Keimer, Junjie Zhang, John F. Mitchell, and W. Andrew MacFarlane. Local metallic and structural properties of the strongly correlated metal LaNiO<sub>3</sub> using <sup>8</sup>Li β–NMR. Phys. Rev. B, 100:165109, Oct 2019.

[57] W. L. Roth. Magnetic structures of MnO, FeO, CoO, and NiO. Physical Review, 110:1333– 1341, 1958.

[58] S. Klotz, K. Komatsu, A. Polian, S. Machida, A. Sano-Furukawa, J.-P. Itie, and T. Hattori. Crystal structure and magnetism of MnO under pressure. Physical Review B, 101:064105, 2020.

[59] Peitao Liu, Cesare Franchini, Martijn Marsman, and Georg Kresse. Assessing modeldielectric-dependent hybrid functionals on the antiferromagnetic transition-metal monoxides MnO, FeO, CoO, and NiO. Journal of Physics: Condensed Matter, 32:015502, 2020.

[60] J. Kunes, V. I. Anisimov, S. L. Skornyakov, A. V. Lukoyanov, and D. Vollhardt. NiO: Correlated band structure of a charge-transfer insulator. Physical Review Letters, 99:156404, 2007.

[61] K. Morikawa, T. Mizokawa, A. Fujimori, Y. Taguchi, and Y. Tokura. Photoemission spectral weight distribution in $\mathrm { Y } _ { 1 - x } \mathrm { C a } _ { x } \mathrm { T i O } _ { 3 }$ . Physical Review B, 54:8446–8451, 1996.

[62] F. Iga, M. Tsubota, M. Sawada, H. Huang, S. Kura, M. Takemura, K. Yaji, M. Nagira, A. Kimura, T. Jo, T. Takabatake, H. Namatame, and M. Taniguchi. Determination of the orbital polarization in YTiO3 by using soft x-ray linear dichroism. Physical Review Letters, 93:257207, 2004.

[63] Jens Jørgen Mortensen, Ask Hjorth Larsen, Mikael Kuisma, Aleksei V. Ivanov, Alireza Taghizadeh, Andrew Peterson, Anubhab Haldar, Asmus Ougaard Dohn, Christian Schäfer, Elvar Örn Jónsson, et al. GPAW: An open python package for electronic structure calculations. The Journal of Chemical Physics, 160(9):092503, 2024.

[64] Volker Blum, Ralf Gehrke, Felix Hanke, Paula Havu, Ville Havu, Xinguo Ren, Karsten Reuter, and Matthias Schefler. Ab initio molecular simulations with numeric atom-centered orbitals. Computer Physics Communications, 180(11):2175–2196, 2009.

[65] Zekun Lou, Alan M. Lewis, and Mariana Rossi. Long-range machine learning of electron density for twisted bilayer moiré materials. PRX Intelligence, 1(1):013015, 2026.

[66] Yubo Qi, Weiyi Gong, and Qimin Yan. Bridging deep learning force fields and electronic structures with a physics-informed approach. npj Computational Materials, 11:177, 2025.

[67] Alexander C. Tyner. Towards a universal model for spin-orbit coupled wannier hamiltonians, 2026.

[68] Christian Venturella, Jiachen Li, Christopher Hillenbrand, Ximena Leyva Peralta, Jessica Liu, and Tianyu Zhu. Unified deep learning framework for many-body quantum chemistry via green’s functions. Nature Computational Science, 5:502–513, 2025.

[69] Evan O. Kane. Band structure of indium antimonide. Journal of Physics and Chemistry of Solids, 1(4):249–261, 1957.

[70] Fei-Hung Lin and Chia-Jyi Liu. An algorithm of calculating transport parameters of thermoelectric materials using single kane band model with riemann integral methods. Scientific Reports, 12:7056, 2022.

# Supplementary Information

# Beyond DFT electronic band structures via dual-space orbital learning

## S1 Dataset construction and splits

The three-dimensional benchmark is derived from the paired PBE–HSE06 all-electron dataset [15, 19, 20, 37]. We distinguish parent materials from band-structure entries: an unpolarized parent contributes one entry, whereas a collinear spin-polarized parent contributes separate spin-up and spin-down entries. All entries associated with the same parent are retained within the same data partition.

The paired source dataset contains 6,576 parent materials and 8,040 band-structure entries. To reduce cases in which the baseline and target calculations represent diferent magnetic states, a complete parent is removed when the absolute diference between its PBE and HSE06 magnetic moments exceeds $0 . 5 \mu _ { \mathrm { B } }$ . This leaves 6,045 parents and 6,978 entries, including 933 spin-polarized parents represented by 1,866 explicit spin-channel entries. Materials reserved for the correlated-material and experimental downstream benchmarks are then excluded, yielding a final benchmark dataset of 5,988 parents and 6,900 entries.

The final benchmark dataset is partitioned at the parent-material level into approximately 8:1:1 training, validation, and test sets. Corresponding spin channels and repeated entries remain grouped within the same partition. Three independent grouped splits are used for the model/split replicates reported in the main benchmark. Exact parent and entry counts are given in Supplementary Table S1.

Table S1: Dataset composition and grouped-split balance. Parent-material and band-structureentry counts are exact. Atoms and k-point counts are reported as entry-weighted means with interquartile ranges shown beneath in brackets. $k _ { \mathrm { H S } }$ and $k _ { \mathrm { f u l l } }$ denote the stored high-symmetry and full-path k-point counts, respectively. Electronic-character fractions are parent weighted. All retained datasets contain 59 chemical elements.
<table><tr><td>Stage seed</td><td>/ Partition Parents Entries</td><td></td><td></td><td>Atoms mean [IQR]</td><td>kHS mean [IQR]</td><td> $k _ { \mathrm { f u l l } }$  mean [IQR]</td><td> $I  I \quad M  I \quad M  M$ </td><td>Parent fraction (%)</td><td></td></tr><tr><td>Dataset construction</td><td></td><td></td><td></td><td></td><td>13.1</td><td>184.0</td><td></td><td></td><td></td></tr><tr><td>Paired source</td><td>All</td><td>6576</td><td>8040</td><td>26.6 [9-32] 27.6</td><td>[12-15] 13.1</td><td>[131-220] 181.1</td><td>50.8</td><td>7.2</td><td>42.0</td></tr><tr><td>After magnetic- state cleaning</td><td>All</td><td>6045</td><td>6978</td><td>[9-34]</td><td>[12-15]</td><td>[130-216]</td><td>54.6</td><td>6.3</td><td>39.1</td></tr><tr><td>Final benchmark dataset</td><td>All</td><td>5988</td><td>6900</td><td>27.8 [10-34]</td><td>13.1 [12-15]</td><td>180.6 [129-216]</td><td>54.5</td><td>6.2</td><td>39.3</td></tr><tr><td colspan="10">Grouped train/validation/test splits</td></tr><tr><td rowspan="2">Seed 42</td><td>Train</td><td>4786</td><td>5513</td><td>27.8 [10-34]</td><td>13.1 [12-15]</td><td>180.7 [129-216]</td><td>54.5</td><td>6.2</td><td>39.3</td></tr><tr><td>Val</td><td>600</td><td>694</td><td>27.7 [10-33]</td><td>13.0 [12-15]</td><td>177.0 [126-212]</td><td>54.7</td><td>6.3</td><td>39.0</td></tr><tr><td rowspan="3">Seed 43</td><td>Test</td><td>602</td><td>693</td><td>27.9 [10-32]</td><td>13.2 [12-15]</td><td>182.5 [130-217]</td><td>54.7</td><td>6.0</td><td>39.4</td></tr><tr><td>Train</td><td>4786</td><td>5520</td><td>27.8 [10-34]</td><td>13.1 [12-15]</td><td>180.1 [129-215]</td><td>54.5</td><td>6.2</td><td>39.4</td></tr><tr><td>Val</td><td>598</td><td>684</td><td>27.4 [9-34] 27.8</td><td>13.1 [12-15] 13.2</td><td>182.8 [131-218] 182.0</td><td>54.7</td><td>6.7</td><td>38.6</td></tr><tr><td rowspan="3">Seed 44</td><td>Test</td><td>604</td><td>696</td><td>[10-34] 27.7</td><td>[12-15] 13.1</td><td>[129-216] 180.5</td><td>54.6</td><td>6.1</td><td>39.2</td></tr><tr><td>Train</td><td>4787</td><td>5518</td><td>[10-34] 27.1</td><td>[12-15] 13.0</td><td>[129-216] 183.2</td><td>54.5</td><td>6.2</td><td>39.3</td></tr><tr><td>Val</td><td>606</td><td>706</td><td>[9-32] 29.2</td><td>[11-14] 13.2</td><td>[130-216]</td><td>54.3</td><td>6.4</td><td>39.3</td></tr><tr><td>Test</td><td></td><td>595</td><td>676</td><td>[10-32]</td><td>[12-15]</td><td>178.2 [129-212]</td><td>54.8</td><td>5.9</td><td>39.3</td></tr></table>

## S2 Orbital representation, architecture, and training

Orbital representation For each PBE calculation, O-kRystal uses atom-centered blocks of the converged Kohn–Sham Hamiltonian H, overlap matrix S, density matrix P, and spin-density matrix $\mathbf { P } _ { \mathrm { s p i n } } ^ { ( \sigma ) } = \mathbf { P } _ { \sigma } - \mathbf { P } _ { \bar { \sigma } }$

For non-spin-polarized entries, the spin-density channel is zero. The input uses a trimmed-valence atomic-orbital representation with one-center and pair blocks padded to a maximum dimension of 16. Real-space orbital blocks describe local electronic interactions, while the corresponding reciprocal-space matrices provide explicit k-resolved information. The production model uses the real components of H(k), S(k), P(k), and ${ \bf P } _ { \mathrm { s p i n } } ( { \bf k } )$ with imaginary reciprocal channels disabled.

Crystal-momentum dependence is additionally introduced through Bloch-phase features on each periodic pair, $[ \sin ( \mathbf { k } \cdot \mathbf { r } _ { i j \mathbf { R } } ) , \cos ( \mathbf { k } \cdot \mathbf { r } _ { i j \mathbf { R } } ) ] , ~ \mathbf { r } _ { i j \mathbf { R } } = \mathbf { r } _ { j } - \mathbf { r } _ { i } + \mathbf { R }$ . The reciprocal orbital matrices are obtained directly from the electronic-structure exports and are not reconstructed from the truncated graph.

Model architecture O-kRystal consists of two jointly trained periodic equivariant branches. The afine branch uses real-space orbital information and Bloch-phase conditioning to predict a banddependent correction to the PBE dispersion, while the dual-space residual branch additionally uses explicit reciprocal-space orbital matrices to predict a k-dependent residual. In the final configuration, the afine branch inherits four interaction and four decoding layers from the model defaults, whereas the residual branch explicitly uses two interaction and two decoding layers. The residual is projected on the high-symmetry reference set to remove its constant and PBE-linear components before being added to the afine correction.

Electronic states are addressed through signed band queries relative to the band edges, with negative indices for valence bands and positive indices for conduction bands. The afine decoder additionally receives five deterministic PBE band descriptors: the band-edge-referenced band mean, standard deviation, minimum, maximum, and band width, as defined in the main Methods. These descriptors are computed over the valid high-symmetry-point subset and use fixed energy scales of 10 eV for the mean, minimum, and maximum terms and 5 eV for the standard deviation and width. The signed-band representation allows the model to accommodate variable numbers of valence and conduction states rather than requiring a fixed-dimensional band output.

Training The primary target is the state-resolved PBE-to-HSE06 correction. Training uses only the stored high-symmetry k-points. The primary state objective is an energy-weighted SmoothL1 loss with transition parameter $\beta = 0 . 1$ eV. An additional band-ofset term penalizes the SmoothL1 error of the mean signed residual for each material and band over the supervised high-symmetry points, and an indirect-gap term penalizes errors in the reconstructed fundamental gap. The final objective is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { s t a t e } } + \mathcal { L } _ { \mathrm { o f f s e t } } + 0 . 2 \mathcal { L } _ { \mathrm { g a p } } .
$$

All other auxiliary objectives, including direct-gap, dispersion, curvature, ranking, and efective-mass terms, have zero weight in the production model.

State weights decay with the PBE energy relative to the band-edge reference using a characteristic scale of 10 eV. Models are trained for 100 epochs with AdamW, a 20-epoch linear warm-up and cosine learning-rate decay. The reported checkpoints are selected by the lowest unweighted mean validation MAE over the 2, 5, 10, and 20 eV energy windows. Exact architectural and optimization settings are summarized in Supplementary Table S2.

Table S2: Architecture and training settings for the three-dimensional PBE-to-HSE06 O-kRystal model.
<table><tr><td>Category</td><td>Setting</td><td>Value</td></tr><tr><td>Model</td><td>Trainable parameters Branches</td><td>8,940,709 Real+phase affine; dual-space residual</td></tr><tr><td></td><td>Interaction / decoder layers</td><td>affine 4 / 4; residual 2 / 2</td></tr><tr><td></td><td>Representation Orbital operators Reciprocal representation</td><td>H, S, P, Pspin Real components of orbital matrices</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td>AO block dimension</td><td>16</td></tr><tr><td></td><td>Radial basis functions</td><td>16</td></tr><tr><td></td><td>Attention</td><td>4 heads; dimension 64 per head</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Decoder</td><td>Material/k latent dimension</td><td>64</td></tr><tr><td></td><td>Signed-band embedding dimension 32</td><td></td></tr><tr><td></td><td>Band descriptors</td><td>mean, s.d., minimum, maximum, width</td></tr><tr><td></td><td>Residual projection</td><td>HS constant and PBE-affine components removed</td></tr><tr><td>Training</td><td>Target</td><td> $E ^ { \mathrm { H S E 0 6 } } - E ^ { \mathrm { P B E } }$ </td></tr><tr><td></td><td>Supervised k-points</td><td>High-symmetry points only</td></tr><tr><td></td><td>Loss</td><td>SmoothL1 state + offset + 0.2× indirect-gap loss</td></tr><tr><td></td><td>Optimizer</td><td>AdamW, 5 × 10−4 initial learning rate</td></tr><tr><td></td><td>Weight decay / betas</td><td>0.005 / (0.9, 0.95)</td></tr><tr><td></td><td>Schedule</td><td>20-epoch warm-up; cosine decay; 100 epochs</td></tr><tr><td></td><td>Precision</td><td>bfloat16 autocast</td></tr><tr><td></td><td>Checkpoint selection</td><td>Mean validation MAE over 2, 5, 10 and 20 eV windows</td></tr></table>

Comparison models and representation ablation The scissor and Linear+KRR baselines operate directly on the PBE band energies and derived state descriptors and contain no learned geometric or orbital representation. The scissor model fits separate global mean PBE-to-HSE06 corrections for valence- and conduction-side states using the training set only. Linear+KRR uses six standardized PBE-derived descriptors in a ridge-linear model followed by RBF kernel-ridge regression of the remaining residual.

The Geometry GNN provides a neural control in which the orbital matrices are removed while retaining the PBE band baseline and explicit crystal-momentum conditioning. It uses atomic-number embeddings, periodic geometric pair features, and Bloch-phase information in a six-layer gated message-passing network with a PBE-conditioned afine band decoder. The Real+Bloch-phase ablation instead retains the real-space orbital representation of O-kRystal together with Bloch-phase conditioning, while omitting the explicit reciprocal-space orbital matrices.

Algorithm 1: O-kRystal training and evaluation procedure   
Input: Parent-grouped $\overline { { \mathcal { D } _ { \mathrm { t r a i n } } , \mathcal { D } _ { \mathrm { v a l } } } } .$ , and $\overline { { \mathcal { D } _ { \mathrm { t e s t } } } }$   
Output: Selected model $f _ { \theta ^ { * } }$ and test predictions   
Notation: ${ \mathcal { O } } ^ { R }$ and $\mathcal { O } ^ { k }$ are real- and reciprocal-space orbital operators; $\phi$ denotes Bloch phases;   
$q$ is the signed-band query; and d denotes PBE descriptors.;   
1. Construct orbital features and targets;   
for $i \in \mathcal { D } _ { \mathrm { t r a i n } }$ do   
$\mathbf { T } _ { i }  ( \mathcal { O } _ { i } ^ { R } , \mathcal { O } _ { i } ^ { k } , \phi _ { i } , q _ { i } , \mathbf { d } _ { i } ) ;$   
$\Delta _ { i }  \dot { \mathbf { E } } _ { i } ^ { \mathrm { H S E 0 6 } } - \dot { \mathbf { E } } _ { i } ^ { \mathrm { P B E } } ;$   
end   
for $i \in \mathcal { D } _ { \mathrm { v a l } } \cup \mathcal { D } _ { \mathrm { t e s t } }$ do   
$\mathbf { T } _ { i }  ( \mathcal { O } _ { i } ^ { R } , \mathcal { O } _ { i } ^ { k } , \phi _ { i } , q _ { i } , \mathbf { d } _ { i } ) ;$   
end   
2. Train and select model;   
$\{ \theta _ { e } \} \gets \mathrm { T r a i n } ( f _ { \theta } , \mathcal { D } _ { \mathrm { t r a i n } } ) ;$   
${ \mathcal { W } } \gets \{ 2 , 5 , 1 0 , 2 0 \} \mathrm { e V } ;$   
$\theta ^ { * } \gets \arg \operatorname* { m i n } _ { \theta _ { e } } \frac { 1 } { | \mathcal { W } | } \sum _ { W \in \mathcal { W } }$ $\theta ^ { * }  \arg \operatorname* { m i n } _ { \theta _ { * } } \frac { 1 } { | \mathcal { W } | } \sum \mathrm { M A E } _ { W } ( \mathcal { D } _ { \mathrm { v a l } } ; \theta _ { e } )$   
3. Predict test bands;   
$\widehat { \Delta } _ { i } \gets f _ { \theta ^ { * } } ( \mathbf { T } _ { i } ) , i \in \mathcal { D } _ { \mathrm { t e s t } } ;$   
$\widehat { \mathbf { E } } _ { i } ^ { \mathrm { H S E 0 6 } } \gets \mathbf { E } _ { i } ^ { \mathrm { P B E } } + \widehat { \Delta } _ { i } ;$

Table S3: Definitions and implementation details for comparison models and representation controls. For the neural learning-curve comparison, Geometry GNN, Real+Bloch phase, and OkRystal use matched seed-specific training subsets, the same final training objective and optimization schedule, raw best\_avg\_win checkpoints, and the same high-symmetry-point evaluation protocol. The three neural models therefore difer primarily in their electronic representation and architecture.
<table><tr><td>Model</td><td>Component</td><td>Setting</td></tr><tr><td>Scissor</td><td>Inputs</td><td>PBE band energies and stored occupied/unoccupied assign- ment</td></tr><tr><td rowspan="2">Linear+KRR</td><td>Fit</td><td>Separate global mean PBE-to-HSE06 corrections for occu- pied and unoccupied training states; no validation or test fitting</td></tr><tr><td>Inputs</td><td>Relative PBE energy, occupation indicator, band width, band standard deviation, log(1 + nearest-band spacing), and  $\log ( N _ { \mathrm { a t o m s } } )$ </td></tr><tr><td rowspan="5"></td><td>Linear stage</td><td>Training-set standardization followed by ridge regression with  $\lambda _ { \mathrm { r i d g e } } = 1 0 ^ { - 3 } N$  , where N is the number of sampled</td></tr><tr><td>Kernel stage</td><td>training states RBF kernel-ridge regression of the linear-model residual with  $\gamma = 1 / 6$  and regularized system  $K ^ { \mathsf { T } } K + 1 0 ^ { - 2 } N K _ { c c } + 1 0 ^ { - 6 } I$ </td></tr><tr><td>Sampling</td><td>Up to 768 high-symmetry states per training entry; up to 768 kernel centres sampled without replacement from the pooled training states</td></tr><tr><td>Geometry GNN Inputs</td><td>Atomic species, periodic geometry, Bloch-phase features, PBE band energies, signed-band queries, and five PBE band descriptors; no orbital matrices</td></tr><tr><td>Architecture</td><td>Six gated periodic message-passing layers with hidden di- mension 256; adaptive nearest-32 periodic graph initialized at 12 bohr and expanded to at most 32 bohr; 16 Gaussian radial functions over 0–32 bohr</td></tr><tr><td rowspan="4"></td><td>Decoder</td><td>PBE-conditioned affine band decoder with 32-dimensional signed-band embedding and 64-dimensional band latent</td></tr><tr><td>Parameters</td><td>5,291,356</td></tr><tr><td>Real+Bloch phase Inputs</td><td>Real-space  $H , S , P ,$  and  $P _ { \mathrm { s p i n } }$  orbital blocks, periodic geome- try, Bloch-phase features, PBE band energies and descriptors, and signed-band queries</td></tr><tr><td>Architecture</td><td>Two interaction and two decoding layers with the PBE- conditioned affine band decoder; explicit reciprocal-space orbital matrices and reciprocal residual branch omitted</td></tr><tr><td>Training</td><td>Parameters</td><td>3,238,604 SmoothL1  $( \beta ~ = ~ 0 . 1 )$  state loss with normalized energy , high-symmetry offset loss (weight</td></tr><tr><td>Optimization</td><td></td><td>weighting  $( E _ { w } = 1 0 ~ \mathrm { e V } )$  1), and indirect-gap loss (weight 0.2) AdamW, learning rate  $5 ~ \times ~ 1 0 ^ { - 4 }$  , weight decay 0.005,</td></tr><tr><td>Checkpoint / evaluation</td><td></td><td> $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 5 )$  ; 20-epoch linear warm-up followed by cosine decay; 100 epochs Lowest unweighted mean validation MAE over the 2, 5, 10,</td></tr></table>

## S3 Contribution of reciprocal-space orbital information

To examine the contribution of orbital representation, we compare the full dual-space O-kRystal with two controls across matched training-set sizes (Supplementary Fig. S1). The Geometry GNN retains the PBE band baseline, band descriptors, periodic geometry, and Bloch-phase conditioning but removes the orbital matrices, whereas the Real+phase model retains the real-space orbital blocks and Bloch-phase features while omitting the explicit reciprocal-space orbital matrices. The Geometry GNN uses a distance-selected periodic graph, whereas Real+phase and O-kRystal share the same electronically preselected real-space orbital graph. Their comparison therefore evaluates the complete dual-space architecture against a real-space orbital alternative that omits the explicit reciprocal-space matrices.

The three representations perform similarly in the smallest-data regime, but their learning curves separate as the training set increases. Real-space orbital information improves upon the geometrybased representation, while the full dual-space model achieves the lowest error at the largest training size in both energy windows. The separation is more pronounced over the 20 eV window, where accurate correction requires representing a broader range of electronic states. These results show that the complete dual-space model achieves the lowest error in the larger-data regime, with the separation becoming clearer over the broader energy window.

![](images/b2dcdf584f14fc0cb7ed093a855d29f2dc0009c7eeed6d4906b41e8bb268eeb8.jpg)

![](images/4e39ef3a4478df8462039205d3493d6400fd451fed87429712ede541abca28a5.jpg)  
Figure S1: Learning-curve comparison of geometric, real-space orbital, and dual-space orbital representations. Test-set band-energy MAE as a function of the number of training entries for states within a, 5 eV and b, 20 eV of the PBE reference energy. The Geometry GNN uses periodic crystal geometry, Bloch-phase conditioning, and the PBE band baseline and descriptors, but no orbital matrices. Real+phase additionally uses real-space orbital blocks with Bloch-phase conditioning, while O-kRystal further incorporates explicit reciprocal-space orbital matrices. All models are evaluated on the corresponding fixed test sets using matched training-data subsets. Points and error bars denote the mean and sample standard deviation across seeds 42–44. The representations perform similarly in the low-data regime, whereas the full dual-space model shows a progressively clearer advantage with increasing training-set size, particularly across the broader 20 eV energy window.

## S4 Reciprocal-space generalization

O-kRystal is trained only on stored high-symmetry k-points for the three-dimensional PBE-to-HSE06 task. Generalization is evaluated at unseen points along the corresponding high-symmetry paths using the same state-matching and energy-window conventions as the primary benchmark. Because predictions are conditioned on the supplied k-point set, path-based evaluations use the complete query set for each path.

Supplementary Fig. S2 examines whether the error at unseen path-interior points depends systematically on reciprocal-space location. The distribution of the diference between interior-point and high-symmetry-point MAE is centered near zero, consistent with the nearly unchanged aggregate error reported in the main text. For insulating systems, the interior-point error increases gradually with reciprocal distance from the nearest supervised high-symmetry point, while remaining nearly independent of normalized position along the path segment. A similar increase is observed for longer high-symmetry-to-high-symmetry segments. Metallic systems exhibit larger fluctuations, with the highest-error bins dominated by a small number of dificult materials, primarily $\mathrm { T i _ { 2 } P }$ and $\mathrm { Z r _ { 2 } P }$ , rather than by a uniform trend across the metallic subset.

Representative full-path predictions are shown in Supplementary Fig. S3.

![](images/4a324521bce3cbaa4ada9b3c0f296ebd3d00d4d76419977ce8d269c7b4f6c9bc.jpg)

![](images/976ad49e5aa87b70f4a4eabfa90936117484efada5a255b211a0d7172ed49c0f.jpg)

![](images/39e90f66f859292098363caf7460f1aa8946f91e52e3f155681bfeaa71b39164.jpg)

![](images/42537e8aca231999b7f29a47c39888059f0b166ed6241bf87b0c04f9ceb67cb3.jpg)  
Figure S2: Dependence of unseen-point error on reciprocal-space position. Diagnostics for path-interior states within $\vert E - E _ { \mathrm { m i d } } \vert < 5 \ \mathrm { e V }$ using the seed-42 raw-weight checkpoint. a, Distribution of the per-entry diference between interior-point and high-symmetry-point MAE. b, Band-energy MAE as a function of reciprocal distance to the nearest high-symmetry point. c, Band-energy MAE as a function of normalized position within a path segment. d, Band-energy MAE as a function of segment length. Panels b–d pool valid states across k-point rows and are therefore state-weighted rather than material-balanced. The elevated metallic bins are concentrated in a small number of high-error materials and should not be interpreted as a universal monotonic dependence on reciprocalspace distance or segment length.

![](images/b4aa0a79bcd3df0ab08a4d298239fbe5446f5d9ff217f7f1d284b7883b4f3f6b.jpg)  
Figure S3: Representative full-path band-structure predictions. PBE, HSE06, and O-kRystal band energies are shown along identical stored k-paths for representative materials spanning insulating, metallic, and spin-polarized electronic structures. The gallery includes $\mathrm { G a O _ { 2 } }$ and WC, whose dense Brillouin-zone DOS reconstructions are shown in Figs. 1b and $s \mathrm { c } ,$ respectively, together with AFM-II MnO, whose orbital-resolved spectrum is shown in Fig. 1c. Additional examples illustrate performance across diferent PBE-to-HSE06 electronic-character changes and spin configurations. Values in parentheses denote the O-kRystal band-energy MAE for each displayed entry within $\vert E - E _ { \mathrm { m i d } } \vert < 5 \mathrm { e V }$

## S5 Strongly correlated materials

Benchmark construction The correlated-material benchmark is constructed independently of model performance. Candidate materials are retained only when the represented magnetic and electronic state is consistent with the corresponding literature-supported phase and when primary literature supports classification as a strongly correlated, Mott-like, charge-transfer, or correlatedmetal system. Applying these criteria yields six materials: $\mathrm { C r O _ { 2 } , L a C o O _ { 3 } , L a N i O _ { 3 } , A F M - I I ~ M n O _ { 3 } }$ AFM-II NiO, and $\mathrm { Y T i O _ { 3 } \ [ 5 1 - 6 2 ] }$ . All six materials are excluded before construction of the primary training, validation, and test splits. Candidate systems are omitted when the available calculation does not provide a defensible representation of the literature-supported phase, including cases with incompatible magnetic ordering, insuficient cell size for the reported antiferromagnetic state, or an HSE06 solution inconsistent with the established electronic character.

The quantitative analysis uses the representative seed-42 model evaluated on dense, material-specific Brillouin-zone meshes. Band-energy errors are computed over states selected by inclusive PBE-centered energy windows. No reciprocal-space quadrature weights are applied to the band-energy MAE. For spin-polarized systems, retained states from both spin channels are pooled within each parent material before calculating its MAE, after which the six parent materials are weighted equally. The primary comparison uses the 5 eV window, while the 20 eV window provides a broader-energy-range evaluation.

Table S4: Correlated-material band-energy benchmark. Band-energy mean absolute errors (MAEs) are evaluated using the representative seed-42 model on dense Brillouin-zone meshes. States are selected using inclusive PBE-centered energy windows of 5 and 20 eV. For spin-polarized systems, retained states are pooled across spin channels before calculating the parent-material MAE. The final row reports the equal-material mean across the six materials. All values are in eV.
<table><tr><td></td><td colspan="3">≤ 5 eV</td><td colspan="3">≤ 20 eV</td></tr><tr><td>Material</td><td>PBE</td><td>Geometry GNN O-kRystal PBE Geometry GNN O-kRystal</td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathrm { C r O _ { 2 } }$ </td><td>0.974</td><td>0.348</td><td>0.279</td><td>1.259</td><td>0.440</td><td>0.332</td></tr><tr><td> $\mathrm { L a C o O _ { 3 } }$ </td><td>1.741</td><td>0.997</td><td>0.591</td><td>1.600</td><td>0.956</td><td>0.516</td></tr><tr><td> $\mathrm { L a N i O _ { 3 } }$ </td><td>1.576</td><td>0.462</td><td>0.367</td><td>1.469</td><td>0.364</td><td>0.368</td></tr><tr><td> $\mathrm { Y T i O _ { 3 } }$ </td><td>1.036</td><td>0.198</td><td>0.102</td><td>1.201</td><td>0.177</td><td>0.121</td></tr><tr><td> $\mathrm { M n O }$ </td><td>1.678</td><td>0.692</td><td>0.605</td><td>1.537</td><td>0.500</td><td>0.489</td></tr><tr><td>NiO</td><td>2.186</td><td>1.455</td><td>0.930</td><td>1.766</td><td>1.046</td><td>0.632</td></tr><tr><td>Material-balanced mean 1.532</td><td></td><td>0.692</td><td>0.479</td><td>1.472</td><td>0.581</td><td>0.410</td></tr></table>

Within the 5 eV window, O-kRystal reduces the equal-material mean band-energy MAE from 1.532 eV for PBE and 0.692 eV for Geometry GNN to 0.479 eV. The same aggregate ordering is retained over the wider 20 eV window, where the corresponding MAEs are 1.472, 0.581, and 0.410 eV, respectively. At the material level, O-kRystal gives the lowest 5 eV error for all six systems. At 20 eV, Geometry GNN and O-kRystal perform similarly for $\mathrm { L a N i O _ { 3 } }$ , while O-kRystal retains the lower aggregate error.

Fundamental-gap recovery Fundamental gaps are evaluated over the dense Brillouin-zone predictions using the stored valence–conduction band boundary. For each spin channel, the gap is defined as the minimum conduction-band energy minus the maximum valence-band energy. For spin-polarized materials, the material-level gap is the minimum over spin channels. Negative gaps are classified as metallic and positive gaps as insulating.

Across the six materials, the fundamental-gap MAE decreases from 1.953 eV for PBE to 1.769 eV for Geometry GNN and 1.537 eV for O-kRystal. The HSE06 metallic or insulating character is recovered for four of six materials with PBE and five of six with both neural correction models. For O-kRystal, $\mathrm { C r O _ { 2 } }$ is the remaining classification failure, with the metallic HSE06 reference predicted as insulating.

Table S5: Fundamental-gap recovery for the correlated-material benchmark. Predicted fundamental gaps are compared with the corresponding HSE06 values. Values in parentheses denote the absolute gap error relative to HSE06. Negative gaps indicate metallic electronic character and positive gaps indicate insulating character. All values are in eV.
<table><tr><td>Material</td><td>HSE06</td><td>PBE</td><td>Geometry GNN</td><td>O-kRystal</td></tr><tr><td> $\mathrm { C r O _ { 2 } }$ </td><td></td><td>-0.809 -1.162 (0.353)</td><td>+1.052 (1.861)</td><td>+1.692 (2.501)</td></tr><tr><td> $\mathrm { L a C o O _ { 3 } }$ </td><td></td><td>+2.669 −0.084 (2.753)</td><td>+0.735 (1.934)</td><td>+1.852 (0.818)</td></tr><tr><td> $\mathrm { L a N i O _ { 3 } }$ </td><td></td><td>-2.159 -1.311 (0.849)</td><td>-0.744 (1.416)</td><td>-0.175 (1.985)</td></tr><tr><td> $\mathrm { Y T i O _ { 3 } }$ </td><td></td><td>+1.852 −0.364 (2.216)</td><td>+0.340 (1.513)</td><td>+0.536 (1.316)</td></tr><tr><td> $\mathrm { M n O }$ </td><td></td><td>+2.864 +0.803 (2.061)</td><td>+1.835 (1.029)</td><td>+2.611 (0.253)</td></tr><tr><td>NiO</td><td></td><td>+4.458 +0.973 (3.485)</td><td>+1.595 (2.863)</td><td>+2.109 (2.349)</td></tr><tr><td>Gap MAE</td><td></td><td>1.953</td><td>1.769</td><td>1.537</td></tr><tr><td>Correct electronic character</td><td></td><td>4/6</td><td>5/6</td><td>5/6</td></tr></table>

Orbital-resolved projected density of states Orbital-resolved spectral reconstruction is evaluated for transition-metal 3d and oxygen 2p channels across all six correlated materials. To isolate diferences in state-energy placement, the PBE, Geometry GNN, O-kRystal, and HSE06 projected densities of states are constructed using the same Mulliken orbital-projection weights obtained from the baseline PBE wavefunctions. Only the state energies entering the reconstruction difer between methods. Projected densities of states are evaluated on dense reciprocal-space meshes using linear tetrahedron integration followed by Gaussian broadening with $\sigma = 0 . 1 0 \ \mathrm { e V }$

Agreement with HSE06 is quantified using the normalized $L _ { 1 }$ error defined in Methods. The integration window is centered at $E _ { \mathrm { m i d } }$ and extends from −5 to +5 eV. Because the orbital weights are fixed across all spectra, the comparison isolates diferences in orbital-resolved state-energy placement rather than changes in target-theory wavefunctions or hybridization.

Table S6: Orbital-resolved PDOS reconstruction errors for the correlated-material benchmark. Normalized $L _ { 1 }$ errors are evaluated relative to HSE06 within $E _ { \mathrm { m i d } } \pm 5 ~ \mathrm { e V }$ . All spectra use identical PBE Mulliken projection weights, such that only the state energies difer between PBE, Geometry GNN, O-kRystal, and HSE06. The final row reports the equal-material mean across the six materials. The metric is dimensionless.
<table><tr><td></td><td colspan="3">Transition-metal 3d</td><td colspan="3">Oxygen 2p</td></tr><tr><td>Material</td><td>PBE</td><td>Geometry GNN O-kRystal PBE</td><td></td><td></td><td>Geometry GNN O-kRystal</td><td></td></tr><tr><td> $\mathrm { { C r O } _ { 2 } }$ </td><td>1.055</td><td>0.466</td><td>0.456</td><td>0.642</td><td>0.323</td><td>0.453</td></tr><tr><td> $\mathrm { L a C o O _ { 3 } }$ </td><td>1.181</td><td>0.839</td><td>0.571</td><td>0.648</td><td>0.734</td><td>0.621</td></tr><tr><td> $\mathrm { L a N i O _ { 3 } }$ </td><td>0.931</td><td>0.862</td><td>0.634</td><td>0.801</td><td>0.671</td><td>0.726</td></tr><tr><td> $\mathrm { Y T i O _ { 3 } }$ </td><td>1.477</td><td>0.780</td><td>0.450</td><td>1.309</td><td>0.396</td><td>0.255</td></tr><tr><td> $\mathrm { M n O }$ </td><td>1.938</td><td>1.163</td><td>0.715</td><td>1.378</td><td>1.109</td><td>0.676</td></tr><tr><td>NiO</td><td>2.123</td><td>1.827</td><td>1.430</td><td>1.440</td><td>1.286</td><td>1.056</td></tr><tr><td>Equal-material mean 1.451</td><td></td><td>0.989</td><td>0.709</td><td>1.036</td><td>0.753</td><td>0.631</td></tr></table>

For transition-metal 3d states, O-kRystal lowers the normalized PDOS error relative to Geometry GNN for all six materials and reduces the equal-material mean from 0.989 to 0.709. For oxygen 2p

states, the aggregate error decreases from 0.753 to 0.631, although the material-resolved improvement is less uniform. In particular, Geometry GNN gives lower oxygen 2p errors for $\mathrm { C r O _ { 2 } }$ and $\mathrm { L a N i O _ { 3 } }$ whereas O-kRystal performs better for the remaining four materials. The stronger and more consistent improvement in the transition-metal 3d channel motivates the orbital-resolved analysis discussed in the main text.

Direct-source MnO case study The MnO spectrum displayed in Fig. 1c is generated from a separately retained direct-source evaluation and is not included in the rebuilt six-material PDOS aggregate reported in Table S6. The calculation represents rocksalt MnO in the AFM-II magnetic configuration along [111] using the corresponding four-atom rhombohedral cell [57, 58]. The PBE and HSE06 calculations use the same resolved $1 2 \times 1 2 \times 1 2$ reciprocal-space mesh. The orbital-resolved reconstruction follows the same fixed-projection convention used in the six-material benchmark: PBE Mulliken weights are retained while the state energies are replaced by the corresponding O-kRystal predicted HSE06 values. The direct-source spectrum is used as an illustrative case study of the redistribution of Mn 3d and O 2p states and is kept numerically separate from the rebuilt benchmark.

## S6 Experimental gap and efective-mass benchmark

The experimental benchmark is derived from the mstar60 dataset [47] and restricted to bulk materials for which the required experimental, PBE, and HSE06 quantities are available. The final cohort contains 31 directional efective-mass observations across ten materials, grouped into III–V semiconductors (GaAs, GaN, InP), chalcogenides (CdS, CdTe, bulk 2H-MoS<sub>2</sub>, bulk 2H-WS<sub>2</sub>), and other semiconductors (Si, SiC, BN). Monolayers are excluded from the bulk comparison, Pb chalcogenides are treated as out of domain because Pb is absent from the training chemistry, and black phosphorus is excluded because the required PBE efective masses are unavailable. One GaAs observation with a non-positive PBE mass and one CdTe observation with an unreliable HSE06 reference value are also excluded.

Because the available HSE06 supervision does not densely resolve the local curvature around band extrema, efective masses are not predicted directly from O-kRystal band dispersions. Instead, we use a gap-informed reconstruction motivated by the Kane model, which relates changes in band-edge efective mass to the electronic band gap through interband coupling [69, 70]. Specifically, the baseline PBE efective mass is rescaled according to the ratio between the O-kRystal-predicted and PBE fundamental gaps. The resulting relation is a calibrated generalization of this gap–mass dependence rather than a direct implementation of the Kane Hamiltonian.

For this analysis, the PBE and predicted HSE06 fundamental gaps are evaluated over the complete stored band path as the conduction-band minimum minus the valence-band maximum. Reconstruction parameters are calibrated separately for each held-out material family using HSE06 efective masses from the remaining families. Experimental efective masses are used only for final evaluation and do not enter the calibration.

For observation i belonging to held-out family $f ,$ the reconstruction uses

$$
\beta _ { \mathrm { r a w } , i } = 1 - \delta _ { f } m _ { i } ^ { \mathrm { P B E } } ,\tag{S1}
$$

where $m _ { i } ^ { \mathrm { P B E } }$ is expressed in units of the free-electron mass.

The exponent is regularized as

$$
\begin{array} { r } { \beta _ { i } = \left\{ \begin{array} { l l } { \operatorname* { m i n } ( \beta _ { \mathrm { r a w } , i } , 1 ) , } & { \beta _ { \mathrm { r a w } , i } \geq 0 , } \\ { - \beta _ { \mathrm { f l o o r } , f } \operatorname { t a n h } \left[ \frac { - \beta _ { \mathrm { r a w } , i } } { \operatorname* { m a x } ( \beta _ { \mathrm { f l o o r } , f } , 1 0 ^ { - 3 } ) } \right] , } & { \beta _ { \mathrm { r a w } , i } < 0 . } \end{array} \right. } \end{array}\tag{S2}
$$

The gap ratio is

$$
r _ { i } = \operatorname* { m i n } \left[ 1 0 ^ { 3 } , \frac { \operatorname* { m a x } ( E _ { g , i } ^ { O - k R y s t a l } , 1 0 ^ { - 3 } \mathrm { e V } ) } { \operatorname* { m a x } ( E _ { g , i } ^ { \mathrm { P B E } } , 1 0 ^ { - 3 } \mathrm { e V } ) } \right] ,\tag{S3}
$$

and the reconstructed efective mass is

$$
m _ { i } ^ { O - k R y s t a l ^ { * } } = m _ { i } ^ { \mathrm { P B E } } r _ { i } ^ { \beta _ { i } } .\tag{S4}
$$

The parameters $( \delta _ { f } , \beta _ { \mathrm { f l o o r } , f } )$ are selected by exhaustive grid search to minimize the material-balanced mean absolute logarithmic error relative to HSE06 efective masses in the calibration families. The search uses $\delta = 0 . 1 , 0 . 2 , \ldots , 1 1 . 6$ and $\beta _ { \mathrm { f l o o r } } = 0 . 0 5 , 0 . 1 0 , \ldots , 1 . 0 0$ . The resulting parameters are (4.4, 0.50) for the III–V holdout, (4.0, 0.25) for the chalcogenide holdout, and (2.9, 0.50) for the other-semiconductor holdout.

Supplementary Table S7 summarizes the benchmark cohort, and Supplementary Table S8 reports all 31 efective-mass observations used in the primary analysis.

Table S7: Efective-mass benchmark cohort. The primary benchmark contains 31 directional efective-mass observations across ten bulk materials.
<table><tr><td>Family</td><td>Materials</td><td>Materials Observations</td></tr><tr><td>III-V</td><td>GaAs, GaN, InP</td><td>13</td></tr><tr><td>Chalcogenides</td><td>CdS, CdTe, 2H-MoS2, 2H-WS2</td><td>9</td></tr><tr><td>Other semiconductors Si, SiC, BN</td><td></td><td>9</td></tr><tr><td>Total</td><td></td><td>31</td></tr></table>

Table S8: Efective-mass observations used in the primary bulk benchmark. All masses are reported in units of the free-electron mass. O-kRystal<sup>∗</sup> denotes the family-held-out gap-informed reconstruction obtained from the PBE efective mass and the fundamental gap predicted by the raw seed-42 O-kRystal checkpoint. It is not a direct efective-mass output of the model.
<table><tr><td>Material</td><td>Family</td><td>Mass type</td><td>k point</td><td>Direction</td><td>Experiment</td><td>PBE</td><td>HSE06</td><td>O-kRystal*</td></tr><tr><td>GaAs</td><td>III-V</td><td> $m _ { n }$ </td><td>Γ</td><td></td><td>0.066</td><td>0.028</td><td>0.062</td><td>0.050</td></tr><tr><td>GaAs</td><td>III-V</td><td> $m _ { n , \perp }$ </td><td> $X _ { 6 }$ </td><td></td><td>0.230</td><td>0.233</td><td>0.223</td><td>0.229</td></tr><tr><td>GaAs</td><td>III-V</td><td> $m _ { n , \perp }$ </td><td> $L _ { 6 }$ </td><td></td><td>0.075</td><td>0.102</td><td>0.110</td><td>0.148</td></tr><tr><td>GaAs</td><td>III-V</td><td> $m _ { n , \parallel }$ </td><td> $L _ { 6 }$ </td><td></td><td>1.900</td><td>1.660</td><td>1.660</td><td>1.188</td></tr><tr><td>GaAs</td><td>III-V</td><td> $m _ { p , \mathrm { h h } }$ </td><td>Γ</td><td>[100]</td><td>0.395</td><td>0.324</td><td>0.309</td><td>0.257</td></tr><tr><td>GaAs</td><td>III-V</td><td> $m _ { p , \mathrm { l h } }$ </td><td>Γ</td><td>[100]</td><td>0.090</td><td>0.034</td><td>0.077</td><td>0.060</td></tr><tr><td>GaAs</td><td>III-V</td><td> $m _ { p , \mathrm { s o } }$ </td><td>Γ</td><td></td><td>0.160</td><td>0.107</td><td>0.157</td><td>0.152</td></tr><tr><td>GaN</td><td>III-V</td><td> $m _ { n , \parallel }$ </td><td>Γ</td><td></td><td>0.200</td><td>0.158</td><td>0.183</td><td>0.171</td></tr><tr><td>GaN</td><td>III-V</td><td> $m _ { n , \perp }$ </td><td>Γ</td><td></td><td>0.200</td><td>0.175</td><td>0.202</td><td>0.186</td></tr><tr><td>InP</td><td>III-V</td><td> $m _ { n }$ </td><td>Γ</td><td></td><td>0.079</td><td>0.054</td><td>0.086</td><td>0.075</td></tr><tr><td>InP</td><td>III-V</td><td> $m _ { p , \mathrm { h h } }$ </td><td>Γ</td><td>[100]</td><td>0.565</td><td>0.469</td><td>0.421</td><td>0.379</td></tr><tr><td>InP</td><td>III-V</td><td> $m _ { p , \mathrm { l h } }$ </td><td>Γ</td><td>[100]</td><td>0.120</td><td>0.073</td><td>0.114</td><td>0.098</td></tr><tr><td>InP</td><td>III-V</td><td> $m _ { p , \mathrm { s o } }$ </td><td>Γ</td><td></td><td>0.210</td><td>0.142</td><td>0.189</td><td>0.167</td></tr><tr><td>CdS</td><td>Chalcogenides</td><td> $m _ { n , \perp }$ </td><td>Γ</td><td>A exciton</td><td>0.192</td><td>0.133</td><td>0.174</td><td>0.162</td></tr><tr><td>CdS</td><td>Chalcogenides</td><td> $m _ { n , \parallel }$ </td><td>Γ</td><td>A exciton</td><td>0.168</td><td>0.123</td><td>0.159</td><td>0.152</td></tr><tr><td>CdS</td><td>Chalcogenides</td><td> $m _ { p , \perp }$ </td><td>Γ</td><td>A exciton</td><td>0.675</td><td>0.245</td><td>0.299</td><td>0.247</td></tr><tr><td>CdS</td><td>Chalcogenides</td><td> $m _ { p , \parallel }$ </td><td>Γ</td><td>A exciton</td><td>5.000</td><td>2.300</td><td>1.730</td><td>2.072</td></tr><tr><td>CdTe</td><td>Chalcogenides</td><td> $m _ { p , \mathrm { l h } }$ </td><td>Γ</td><td>[100]</td><td>0.130</td><td>0.058</td><td>0.104</td><td>0.073</td></tr><tr><td>CdTe</td><td>Chalcogenides</td><td> $m _ { p , \mathrm { h h } }$ </td><td>Γ</td><td>[100]</td><td>0.720</td><td>0.431</td><td>0.403</td><td>0.400</td></tr><tr><td> ${ \mathrm { M o S } } _ { 2 }$ </td><td>Chalcogenides</td><td> $m _ { p }$ </td><td>ī</td><td>Γ-K</td><td>0.670</td><td>0.775</td><td>0.649</td><td>0.709</td></tr><tr><td> $\mathrm { W S _ { 2 } }$ </td><td>Chalcogenides</td><td> $m _ { p }$ </td><td>K</td><td>ř−K</td><td>0.350</td><td>0.354</td><td>0.300</td><td>0.322</td></tr><tr><td> $\mathrm { W S _ { 2 } }$ </td><td>Chalcogenides</td><td> $m _ { p , \mathrm { V B - 1 } }$ </td><td>K</td><td>Γ-K</td><td>0.430</td><td>0.495</td><td>0.431</td><td>0.447</td></tr><tr><td>Si</td><td>Others</td><td> $m _ { n , \perp }$ </td><td>CBM</td><td></td><td>0.191</td><td>0.193</td><td>0.186</td><td>0.243</td></tr><tr><td>Si</td><td>Others</td><td> $m _ { n , \parallel }$ </td><td>CBM</td><td></td><td>0.916</td><td>0.943</td><td>0.914</td><td>0.727</td></tr><tr><td>Si</td><td>Others</td><td> $m _ { p , \mathrm { h h } }$ </td><td>Γ</td><td>[100]</td><td>0.460</td><td>0.267</td><td>0.259</td><td>0.300</td></tr><tr><td>Si</td><td>Others</td><td> $m _ { p , \mathrm { l h } }$ </td><td>Γ</td><td>[100]</td><td>0.171</td><td>0.193</td><td>0.189</td><td>0.243</td></tr><tr><td>Si</td><td>Others</td><td> $m _ { p , \mathrm { s o } }$ </td><td>Γ</td><td></td><td>0.262</td><td>0.230</td><td>0.224</td><td>0.274</td></tr><tr><td>SiC</td><td>Others</td><td> $m _ { n , \parallel }$ </td><td>X</td><td></td><td>0.662</td><td>0.652</td><td>0.613</td><td>0.573</td></tr><tr><td>SiC</td><td>Others</td><td> $m _ { n , \perp }$ </td><td>X</td><td></td><td>0.244</td><td>0.225</td><td>0.218</td><td>0.247</td></tr><tr><td>SiC</td><td>Others</td><td> $m _ { p }$ </td><td>Γ</td><td>[100]</td><td>0.450</td><td>0.603</td><td>0.523</td><td>0.533</td></tr><tr><td>BN</td><td>Others</td><td> $m _ { p }$ </td><td>K</td><td>Γ−K</td><td>0.490</td><td>0.971</td><td>0.805</td><td>0.873</td></tr></table>

## S7 $G _ { 0 } W _ { 0 }$ quasiparticle benchmark

The two-dimensional quasiparticle benchmark contains 286 nonmagnetic semiconductor records derived from C2DB, with eight valence states (VB8–VB1) and four conduction states (CB1–CB4) per material [38, 39, 41]. The target is the one-shot $G _ { 0 } W _ { 0 }$ quasiparticle spectrum obtained from a GPAW PBE starting point [22, 23, 63]. Training and validation use the available high-symmetry vertices together with up to nine evenly spaced stored interior points per high-symmetry path segment. The $G _ { 0 } W _ { 0 }$ task is trained separately from the three-dimensional HSE06 model while retaining the same state-resolved correction framework.

Five-fold material-level cross-validation is used with separate training, validation, and test folds. For each outer fold, three source folds are used for training, one fold is used for validation, and the remaining fold is evaluated as held-out test data. Performance is reported as the MAE over the twelve quasiparticle states and the corresponding band-gap MAE. The GPAW- and FHI-aims-baseline variants use identical $G _ { 0 } W _ { 0 }$ labels, k-point sets, and real- and reciprocal-space orbital inputs [63, 64]. Only the baseline PBE eigenvalues, and consequently the $G _ { 0 } W _ { 0 } – \mathrm { P B E }$ correction targets, difer. The two variants are trained independently from scratch using otherwise matched model, sampling, and optimization settings. Supplementary Fig. S4 reports the state-resolved errors across the valence and conduction manifolds, and Supplementary Table S9 gives the fold-resolved results.

![](images/249cea9a5da729ab0844699e79d7bc9cfcede817ea40670d2cfc695be27420cf.jpg)  
Figure S4: State-resolved $G _ { 0 } W _ { 0 }$ quasiparticle prediction errors. Mean absolute error for individual valence (VB8–VB1) and conduction (CB1–CB4) states across the five material-held-out folds. The two variants difer only in the PBE eigenvalue baseline used to define the state-resolved correction target. The $G _ { 0 } W _ { 0 }$ labels, k-point sets, and orbital inputs are otherwise identical.

Table S9: Five-fold $G _ { 0 } W _ { 0 }$ quasiparticle benchmark. State and band-gap MAEs are reported in eV for independently trained models. For each outer fold, three source folds train the model, one separate fold selects the checkpoint by validation MAE, and the remaining fold is evaluated once as test data.
<table><tr><td>Fold</td><td>Train Validation Test GPAW state [63]</td><td></td><td></td><td></td><td>GPAW</td><td>gap FHI-aims state [64]</td><td>FHI-aims gap</td></tr><tr><td>0</td><td>171</td><td>57</td><td>58</td><td>0.1197</td><td>0.2132</td><td>0.1541</td><td>0.2374</td></tr><tr><td>1</td><td>172</td><td>57</td><td>57</td><td>0.1380</td><td>0.2173</td><td>0.1811</td><td>0.2229</td></tr><tr><td>2</td><td>172</td><td>57</td><td>57</td><td>0.1449</td><td>0.1995</td><td>0.1728</td><td>0.2018</td></tr><tr><td>3</td><td>172</td><td>57</td><td>57</td><td>0.1237</td><td>0.1575</td><td>0.1645</td><td>0.2021</td></tr><tr><td>4</td><td>171</td><td>58</td><td>57</td><td>0.1021</td><td>0.1396</td><td>0.1649</td><td>0.1654</td></tr><tr><td>Mean ± s.d.</td><td></td><td></td><td></td><td>0.1257 ± 0.0167 0.1854 ± 0.0349</td><td></td><td>0.1675 ± 0.0101 0.2059 ± 0.0272</td><td></td></tr></table>

## S8 Computational and reproducibility details

Training of the three-dimensional models used dynamic batches of 1–12 band-structure entries, subject to a maximum of 700 atoms per batch and implementation-level padding limits for real-space and k-resolved pairs. Mixed-precision computation used bfloat16 autocast. Training the final O-kRystal model requires approximately 6 min per epoch on a single NVIDIA H200 GPU, corresponding to approximately 10 GPU-hours over the 100-epoch training schedule. Reported three-dimensional evaluations use the raw best\_avg\_win checkpoint weights, selected by the lowest unweighted mean validation MAE over the 2, 5, 10, and 20 eV energy windows.

The seed-42 model contains 8,940,709 trainable parameters. The corresponding checkpoint, model configuration, parent-grouped split manifest, and seed-specific UID lists are retained with the reproducibility artifacts. Source code, model configurations, evaluation scripts, split manifests, and numerical figure-source data will be released with the public repository. The trained checkpoint will be distributed separately as part of the reproducibility package.

Available computational accounting for 6,576 source materials totals 88,051 CPU core-hours for PBE and 1,600,661 CPU core-hours for HSE06, corresponding to an aggregate HSE06-to-PBE computational-cost ratio of 18.18. These values represent cumulative CPU resource usage rather than wall-clock time and are not used to infer an end-to-end speedup.