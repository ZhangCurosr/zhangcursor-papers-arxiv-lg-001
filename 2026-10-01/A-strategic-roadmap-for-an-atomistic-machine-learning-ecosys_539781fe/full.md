# A strategic roadmap for an atomistic machine-learning ecosystem

J¨org Behler ,<sup>1,</sup> <sup>2</sup> Michele Ceriotti ,<sup>3,</sup> <sup>∗</sup> Cecilia Clementi ,<sup>4</sup> G´abor Cs´anyi ,<sup>5,</sup> <sup>6</sup> Alin-Marin Elena ,<sup>7</sup> Aditi Krishnapriyan ,<sup>8,</sup> <sup>9</sup> Joseph W. Abbott ,<sup>3</sup> Fabio Afinito ,<sup>10</sup> Albert P. Bart´ok ,<sup>11,</sup> <sup>12</sup> Ilyes Batatia ,<sup>5</sup> Filippo Bigi ,<sup>3,</sup> <sup>13</sup> Florian N. Br¨unig ,<sup>14</sup> Yannick Calvino Alonso ,<sup>15</sup> Giuseppe Carleo ,<sup>16</sup> Aur´elie Champagne ,<sup>17,</sup> <sup>18</sup> Stefan Chmiela ,<sup>19,</sup> <sup>20</sup> Marc L. Descoteaux ,<sup>21</sup> Ralf Drautz ,<sup>22</sup> Alexandra Farcas ,<sup>23,</sup> <sup>24</sup> Meng Gao ,<sup>13</sup> Rohit Goswami ,<sup>3</sup> Michael F. Herbst ,<sup>25</sup> Christian Holm ,<sup>26</sup> James R. Kermode ,<sup>12</sup> Alexander L. M. Knoll ,<sup>1,</sup> <sup>2</sup> Tobias Kreiman ,<sup>9</sup> Hoang-Thien Luu ,<sup>27</sup> Yury Lysogorskiy ,<sup>22</sup> Mihai-Cosmin Marinica ,<sup>28</sup> Rocco Meli ,<sup>29</sup> Klaus-Robert M¨uller ,<sup>19,</sup> <sup>20,</sup> <sup>30,</sup> <sup>31</sup> Frank No´e ,<sup>32,</sup> <sup>33</sup> Mohamadhosein Nosratjoo ,<sup>34</sup> Simon Olsson ,<sup>35</sup> Christoph Ortner ,<sup>36</sup> Aldo S. Pasos-Trejo ,<sup>4</sup> Anyang Peng ,<sup>37</sup> Eric Qu ,<sup>9</sup> Andrea Rizzi ,<sup>38</sup> Mariana Rossi ,<sup>39,</sup> <sup>40</sup> Bassem Sboui ,<sup>17,</sup> <sup>18</sup> Gregor N. C. Simm ,<sup>41</sup> Alexandre Tkatchenko ,<sup>14</sup> Jacopo Venturin ,<sup>4</sup> O. Anatole von Lilienfeld ,<sup>42,</sup> <sup>43</sup> William C. Witt ,<sup>21</sup> Brandon M. Wood ,<sup>13</sup> Tigany Zarrouk ,<sup>44</sup> and Fabian Zills <sup>26</sup>

<sup>1</sup>Lehrstuhl f¨ur Theoretische Chemie II, Ruhr-Universit¨at Bochum, 44780 Bochum, Germany   
<sup>2</sup>Research Center Chemical Sciences and Sustainability,   
Research Alliance Ruhr, 44780 Bochum, Germany   
<sup>3</sup>Laboratory of Computational Science and Modeling, Institut des Mat´eriaux,   
Ecole Polytechnique F´ed´erale de Lausanne, 1015 Lausanne, Switzerland<sup>´</sup>   
<sup>4</sup>Department of Physics, Freie Universit¨at Berlin, 14195 Berlin, Germany   
<sup>5</sup>Engineering Laboratory, University of Cambridge, Trumpington St, Cambridge, UK   
<sup>6</sup>Max Planck Institute for Polymer Research, Ackermannweg 10, Mainz, Germany   
<sup>7</sup>Scientific Computing Department, Science and Technology Facilities Council,   
Daresbury Laboratory, Keckwick Lane, Daresbury WA4 4AD, UK   
<sup>8</sup>Lawrence Berkeley National Laboratory (LBNL)   
<sup>9</sup>University of California, Berkeley   
<sup>10</sup>CINECA, Consorzio Interuniversitario, Bologna, Italy   
<sup>11</sup>Department of Physics, University of Warwick, Coventry, CV4 7AL, UK   
<sup>12</sup>Warwick Centre for Predictive Modelling, School of Engineering,   
University of Warwick, Coventry, CV4 7AL, UK   
<sup>13</sup>FAIR, Meta, San Francisco, CA, USA   
<sup>14</sup>Department of Physics and Materials Science, University of Luxembourg, L-1511 Luxembourg   
<sup>15</sup>Laboratory for Computational Molecular Design, Institute of Chemical Sciences and Engineering,   
Ecole Polytechnique F´ed´erale de Lausanne, 1015 Lausanne, Switzerland<sup>´</sup>   
<sup>16</sup>Laboratory of Computational Quantum Science, Institut de Physique,   
Ecole Polytechnique F´ed´erale de Lausanne, 1015 Lausanne, Switzerland<sup>´</sup>   
<sup>17</sup>Universit´e de Bordeaux, CNRS, Bordeaux INP, ICMCB, F-33600 Pessac, France   
<sup>18</sup>R´eseau sur le Stockage Electrochimique de l’Energie (RS2E),   
CNRS FR 3459, Cedex 1, F-80039 Amiens, France   
<sup>19</sup>Machine Learning Group, Technische Universit¨at Berlin, 10587 Berlin, Germany   
<sup>20</sup>Berlin Institute for the Foundations of Learning and Data – BIFOLD, 10587 Berlin, Germany   
<sup>21</sup>John A. Paulson School of Engineering and Applied Sciences, Harvard University, Cambridge, MA, USA   
<sup>22</sup>Interdisciplinary Centre for Advanced Materials Simulation (ICAMS),   
Ruhr-University Bochum, 44780 Bochum, Germany   
<sup>23</sup>Department of Physics and Chemistry, Technical University of Cluj-Napoca, 400641 Cluj-Napoca, Romania   
<sup>24</sup>National Institute for Research and Development of Isotopic and   
Molecular Technologies, 67-103 Donat, 400293 Cluj-Napoca, Romania   
<sup>25</sup>Mathematics for Materials Modelling, Institute of Mathematics & Institute of Materials,   
Ecole Polytechnique F´ed´erale de Lausanne, 1015 Lausanne, Switzerland<sup>´</sup>   
<sup>26</sup>Institute for Computational Physics, University of Stuttgart, Stuttgart, Germany   
<sup>27</sup>Institute of Metallurgy, Technical University of Clausthal, Clausthal-Zellerfeld, Germany   
<sup>28</sup>Universit´e Paris-Saclay, CEA, Service de recherche en Corrosion et   
Comportement des Mat´eriaux, SRMP, 91191, Gif-sur-Yvette, France   
<sup>29</sup>Swiss National Supercomputing Centre, ETH Z¨urich, Lugano, Switzerland   
<sup>30</sup>Max Planck Institute for Informatics, Stuhlsatzenhausweg, 66123 Saarbr¨ucken, Germany   
<sup>31</sup>Department of Artificial Intelligence, Korea University, Anam-dong, Seongbuk-gu, 02841 Seoul, Korea   
<sup>32</sup>Microsoft Research AI for Science, 10178 Berlin, Germany   
<sup>33</sup>Department of Mathematics, Freie Universit¨at Berlin, 14195 Berlin, Germany   
<sup>34</sup>Department of Chemistry, The University of Manchester, Manchester, UK   
<sup>35</sup>Department of Computer Science and Engineering,   
Chalmers University of Technology and University of Gothenburg, SE-41296 Gothenburg, Sweden   
<sup>36</sup>Department of Mathematics, University of British Columbia, Vancouver, Canada   
<sup>37</sup>AI for Science Institute, Beijing, China

<sup>38</sup>Achira Inc., San Francisco, California, USA

<sup>39</sup>MPI for the Structure and Dynamics of Matter, Hamburg, Germany

<sup>40</sup>Yusuf Hamied Department of Chemistry, Cambridge University, Cambridge, UK

<sup>41</sup>Microsoft Research, AI for Science, Cambridge CB1 2FB, UK <sup>42</sup>Department of Chemistry, Department of Materials Science & Engineering, and Department of Physics, University of Toronto, Toronto, Ontario, Canada <sup>43</sup>Vector Institute for Artificial Intelligence, Toronto, Ontario, Canada <sup>44</sup>Department of Chemistry and Materials Science, Aalto University, 02150 Espoo, Finland (Dated: October 1, 2026)

Data-driven machine learning (ML) techniques have become an essential tool in many domains of science. Their application to atomistic simulations of matter is particularly widespread and impact ful. This success is due largely to the existence of a well-developed and established physics-based modeling framework, ranging from first-principles electronic-structure calculations to molecular dynamics and statistical sampling, into which ML was integrated naturally to reshape long-standing trade-ofs between accuracy, eficiency, and scale. Nevertheless, this integration raises both conceptual and practical challenges, from choosing between data-centric and physics-based modeling approaches to adapting established software stacks to modern hardware accelerators and ML libraries. As the field evolves rapidly, fueled in part by widespread enthusiasm but also by tangible impact, it seems appropriate to take a moment to consider the current state of the art and open challenges, and reflect on what can be done to better coordinate eforts across the community. With this goal in mind, several members of this community met in Lausanne in January 2026 at CECAM to discuss algorithms, models, software and hardware infrastructure, and the most promising scien tific applications that have become possible thanks to the use of artificial intelligence in atomic-scale simulations. This strategic roadmap paper summarizes the outcomes of these discussions, suggesting some long-term goals, and some concrete actions, to establish a healthy, sustainable and impactful atomistic ML ecosystem.

## I. WHAT IS ATOMISTIC MACHINE LEARNING

The simulation of matter at the atomic scale is one of the most established and impactful applications of computing in science and engineering. Decades of work on electronic-structure theory, statistical sampling, and molecular dynamics (MD) have produced a mature framework that, in many cases, can not only explain but also efectively complement or even anticipate experimental findings [49, 62, 128, 261]. Yet, the cost of accurate first-principles calculations limits what can actually be simulated to much smaller systems and shorter timescales than what is needed to address the most consequential questions in chemistry, biology, and materials science. In the last two decades, machine learning (ML) has emerged as the most successful strategy to lift these limitations, and the speed at which it is doing so is striking enough to deserve a coordinated reflection on where the field is heading.

This roadmap builds on the discussions of a workshop held at CECAM in Lausanne in January 2026[241], where developers and users of atomistic machine-learning methods came together to discuss algorithms, models, soft ware and hardware infrastructure, as well as applications. Our scope is intentionally focused. We consider only atomistic machine learning in the sense of particlebased modeling of matter, with the goal of predicting the structure and properties of matter by computer simulation from fundamental physical laws (meaning without experimental input), on the basis of quantum-chemical calculations, mainly density functional theory (DFT) but occasionally also higher-accuracy methods. We do not cover sequence-based generative models for biology in the style of AlphaFold [3], cheminformatics, or analysis pipelines that postprocess simulation outputs.

Atomistic ML is an application of ML in which three features are especially prominent. First, the underlying “source of truth” or physical framework is well understood, meaning that large amounts of training data can be generated almost routinely using electronic-structure calculations. Second, sharing data is harder than it should be: small diferences in pseudopotentials, exchange–correlation functionals, basis sets, or convergence settings, which can all be perfectly justified on a perproject basis, often lead to numerically incompatible datasets. Third, the usual tension between interpolation and extrapolation is not a side issue but a central methodological question.

Reference datasets for a specific system of interest are used to construct an eficient machine-learning interatomic potential (MLIP), also termed a machine-learning force field (MLFF)—the two expressions originate from the materials-science and biochemical-modeling communities [203], respectively, and are used interchangeably for ML models; we use the former in this roadmap. MLIPs are then used to scale up simulations both in size and time, to gain insights inaccessible to electronicstructure calculations, which requires a reliable description clearly beyond the geometries covered in the training set. Additionally, models are often expected to describe chemistries that are not present in their training sets, i.e., potentials should be applicable to systems with notably diferent structures or chemical composition. Unless there is reliable extrapolation beyond the training domain, the model is not useful.

![](images/1d9f48ca1c2e8797f8621cde0d6274e01604205891f2978ba777a5333404c7b1.jpg)  
FIG. 1. Accuracy and timescales accessible for traditional approaches and atomistic machine learning.

The community now finds itself with a powerful new set of tools in a fast-evolving but uneven landscape. Methods, datasets, and software stacks are proliferating; new models appear almost weekly, and benchmarks struggle to keep up. We argue that this is precisely the moment to coordinate. The aim of this roadmap is to summarize the state of the art and the most pressing open problems, and to identify some community actions that we believe would have a strong impact on the longterm health of the field. Sec. II surveys the main use cases of atomistic ML, with an emphasis on interatomic potentials while also taking a deliberate look at other atomic properties and electronic-structure surrogates. Sec. III discusses approaches to reach longer time and length scales: coarse-grained models, enhanced sampling, generative samplers, and learned integrators. Sec. IV discusses current models, datasets, and representative applications. Sec. V addresses the assessment of model and data quality, including benchmarks, challenges, and uncertainty quantification. Sec. VI examines software and hardware considerations. Sec. VII closes with some concrete recommendations.

## II. USE CASES FOR ATOMISTIC MACHINE LEARNING

By far the most mature application of atomistic machine learning is the parameterization of interatomic potentials, and it is what most of this roadmap concentrates on. The reach of atomistic ML, however, is much broader, and several adjacent problems are evolving fast enough to be discussed alongside interatomic potentials rather than in isolation. We sketch the landscape here, leaving a more critical discussion of the state of the art

to Sec. IV.

## A. Machine learning interatomic potentials and force fields

Interatomic potentials are multidimensional functions that map a configuration of atoms (and, where applicable, the vectors defining a periodic unit cell) to its potential energy and the associated forces and stresses. Their construction has been a central problem of computational physics and chemistry since their inception, motivated by the wish to reach the accuracy of electronic-structure calculations for molecular dynamics and geometry optimiza tion applications but without the cost of representing the electrons explicitly. The first ML approaches that were transferable across configurations of a given system were introduced for adsorbed molecules in the mid-1990s [41]. Broad adoption at that time was still hindered by the restriction of early MLIPs to only a few degrees of freedom. The modern field can be traced back to the introduction of atom-centered geometry representations exploiting locality [30], which enabled the construction of potentials for very large systems. The earliest examples for material systems were high-dimensional neural-network potentials (HDNNPs) [30] based on atom-centered symmetry functions [27], and shortly afterwards Gaussian Approximation Potentials (GAP) based on spherical harmonic representations and kernel regression [18, 19]—well before the deep-learning boom in other fields of science. Kernel and neural models were also shown to be capable of learning quantum energies and related properties across chemical compound space, with Coulomb-matrix representations being applied to QM7/QM9-type molecular data [221, 280, 294], initiating a rapid development that quickly reduced errors relative to DFT below chemical accuracy and, with ∆-learning, brought predictions close to chemical accuracy relative to higher levels of theory [281]. MLIPs have evolved significantly since then, going through successive generations [28, 78, 79, 336], improving the range of applications, the complexity of the underlying model, and the understanding of relationships between structural representations and architectures [43, 232, 237, 272]. From those foundations, the field has expanded into a dense ecosystem of diverse architectures and software stacks that we discuss in Sec. IV.

Two regimes have emerged over the past few years. On one side, models trained for a specific system or chemistry can reach excellent accuracy, rivaling electronic structure calculations with modest training sets (on the order of a few thousand structures). This regime is relevant whenever a study requires very high quantitative fidelity, high eficiency, or both, and is well served by both shallow and deep architectures. On the other side, increasingly large and chemically diverse datasets have enabled the training of general-purpose potentials, sometimes referred to as “universal” or “foundation” models, that aim to cover a substantial fraction of the periodic table with a single set of learned parameters [15, 20, 22, 24, 61, 76, 106, 139, 165, 170, 179, 198, 201, 209, 214, 216, 220, 235, 256, 284, 357– 361, 365, 367, 372, 373, 379]. The terminology is not without controversy. “Foundation” is borrowed from natural language processing, where it denotes models pre-trained on broad data and then adapted to many downstream tasks. In this sense the label fits: generalpurpose potentials are frequently fine-tuned, often with little data, to reach the accuracy required for a given system. The same need for fine-tuning, however, shows that “universal” is aspirational: current models are better described as broadly transferable energy and force predictors, with emerging extensions to derived properties. It is also worth keeping in mind that for molecular systems even early MLIPs boasted higher transferability than empirical potentials [307, 315], and that it is not dificult to find quantitative and qualitative failure modes for most general-purpose models [184]. The change of the past few years is one of scale and acces sibility: a researcher starting on a new system can now obtain a reasonable first structural model with little or no in-house training—without the parameterization effort of a classical force field, and at a fraction of the cost of a first-principles simulation—which has simplified the workflow of computational studies and has also made the fast screening of large numbers of molecules or materials practical, boosting data-driven materials discovery [26].

The two regimes are not in opposition. Generalpurpose models can serve as initial guesses to be specialized through fine-tuning [22, 178, 209, 279], distillation [6], or retraining on a focused dataset. Specialized models continue to set the standard of accuracy for individual systems and, increasingly, also serve as cheap inference targets for production simulations. In many cases, however, fine-tuning is superfluous, given that the zero-shot error of universal models against their reference electronic-structure model when performing sophisticated simulation tasks is often lower than the error of the reference against experiments [216]. This also means that further improvements depend as much on the quality of the training data as on advances in ML itself. This roadmap proposes that the community think about training data, software interfaces, and benchmarks in a way that reflects the continuum from specialized to generally applicable models.

## B. What are MLIPs good for

MLIPs can be used in all applications for which an empirical potential or an ab initio ground-state potential calculation could be used, but with higher accuracy than the former, and greater speed than the latter [28, 79, 151]. The most impactful use case is probably that of MD— used to compute both equilibrium thermodynamic averages [64] and time-dependent behavior [380]—but similar acceleration can be exploited for high-throughput campaigns to look for locally stable geometries [220] as well as to look for transition states [342]. For this latter task, which probes the regions of the potential energy surface for which it is hardest to collect training structures (and for which electronic structure calculations are least reliable), it might be advisable not to rely entirely on MLIPs, but to use them to accelerate transition-state search algorithms such as the dimer method, or the nudged elastic band family of methods [136, 155]. Pairing MLIP based optimization and ab initio refinement can cut by up to 90% the cost of the search of both local minima and saddle points by reducing the number of ab initio force evaluations [127, 288], while still delivering nominal first-principles accuracy.

## C. Other atomic and electronic properties

Many practical questions require properties beyond energies and forces. Multipole moments, polarizabilities, nuclear magnetic resonance (NMR) chemical shifts, x-ray photoelectron spectra (XPS) core-electron binding energies, infrared and Raman intensities, and other simple response properties have been targets of ML approaches for at least a decade [16, 51, 88, 125, 142, 166, 204, 222, 293, 301, 338], and are now mature enough to be routinely included as auxiliary outputs of MLIP architectures with little overhead. Such properties allow for not only the validation of generated structural models by comparison with experimental data but also the direct inference of atomistic structures that agree with the experiment using inverse methods [208, 369, 370].

A more fundamental development is the prediction of electronic-structure outputs, such as the electron density, the density of states, the single-particle Hamiltonian, or the density matrix in a chosen basis [31, 45, 52, 59, 98, 126, 159, 180, 196, 238, 274, 308, 317, 352, 374, 378]. These approaches are advancing quickly and represent a qualitatively diferent proposition: rather than replacing the electronic-structure potential energy surface with a surrogate, one replaces (parts of) the electronic-structure calculation itself, in a way that preserves access to all derived properties. This approach promises to reduce the computational cost of certain bottlenecks of the ab initio calculation or even reduce the scaling with system size. Most of these models regress Hamiltonian or densitymatrix elements in a fixed atomic-orbital basis, inheriting its conventions rather than learning basis-independent observables directly, a choice that binds the target to an arbitrary representation and limits the opportunity for ML to drive model order reduction. Intermediate approaches that combine ML with semi-empirical or tight binding Hamiltonians retain an explicit electronic structure at low cost [99, 255, 348, 383]. For these electronicstructure surrogates, the data, software, and validation challenges are correspondingly larger than for MLIPs; we return to them in Sec. VI. We expect this area to grow into a major component of the ecosystem in the coming years, and to enable a step change in the achievable accuracy of derived simulations.

## III. LONGER TIME AND LENGTH SCALES

## A. Coarse-graining

Atomistic MLIPs hold the promise of near-quantum accuracy at a fraction of first-principles cost, greatly extending the time and length scales accessible to simulations (cf. Fig. 1), yet they remain more expensive than the classical force fields used in biomolecular simulations, where reaching beyond microsecond timescales is often necessary. Coarse-grained (CG) models introduce a second level of dimensional reduction: instead of replacing electronic structure by an atomistic potential, they emulate an atomistic simulation by an even lower-resolution model. A coarse-graining map projects atomistic coordinates onto CG coordinates, e.g., the positions of backbone and side-chain CG particles (or “beads”) for an amino acid chain, or a small collection of beads representing the functional groups of molecules in the condensed phase. The formally exact target is the free energy surface, or equivalently the potential of mean force, whose Boltzmann distribution equals the marginal distribution obtained by mapping the atomistic equilibrium ensemble to the CG variables [148, 152, 240, 311]. CG models accelerate simulations by reducing the number of particles that need to be treated for a given system size, and by allowing for longer time steps, since fast molecular motions such as the vibration of covalent bonds are averaged out.

Unlike atomistic MLIPs, machine-learned coarsegrained (MLCG) potentials are not generally trained on directly available pointwise energy labels. Systematic bottom-up approaches instead optimize CG force fields using variational force matching or relative-entropy minimization [148, 240, 311]. The exact CG potential is generally many-body, even when the underlying atomistic force field contains mostly few-body terms. This is where pre-ML CG force fields were limited in accuracy and transferability [72], and where ML makes an important diference: flexible neural, kernel, graph neural network and cluster-expansion representations can approximate many-body CG free-energy surfaces and thereby produce bottom-up force fields that better reproduce atomistic equilibrium distributions [90, 91, 145, 154, 182, 343, 344, 347].

Bottom-up matching is not the only route to a CG model and is arguably not the most established one. Indeed, a large body of widely used CG force fields is parameterized top-down, by tuning interactions to reproduce experimental or macroscopic observables—partition free energies, densities, interfacial tensions—rather than the statistics of a finer simulation [264, 319]. The two philosophies have complementary failure modes: bottomup models are systematically tied to, and limited by, the accuracy of their atomistic or even quantum-mechanical reference data, whereas top-down models can match target observables while misrepresenting the underlying ensemble. For ML, this suggests a direction that neither tradition has fully taken advantage of: hybrid objectives in which a force-matched or relative-entropy base model is corrected, regularized, or fine-tuned against experimental targets [202, 234]. Such hybrid training is also one of the few principled ways to escape the “quantum ceiling” discussed in Sec. V D, since the correction is anchored to experiment rather than to a reference method that may itself be wrong.

Recent MLCG work has moved from system-specific models toward transferable biomolecular force fields. Early neural and Gaussian-process CG models showed that force-matched ML potentials can reproduce finegrained equilibrium distributions and folding free-energy landscapes for small molecular systems [145, 154, 344]. Protein-scale MLCG models now preserve important thermodynamic features of atomistic simulations while accelerating sampling by several orders of magnitude [213]. The newest transferable protein MLCG models show that, with suficiently diverse atomistic training data and expressive architectures, a single bottom-up force field can generalize across sequences and reproduce folded structures, intermediates, intrinsically disordered ensembles, folding-upon-binding, and mutation-induced stability changes [60].

Together, these results point toward foundation-style CG simulators for proteins, polymers, membranes, and molecular assemblies—but realizing this ambition depends not only on expressive architectures and diverse atomistic training data [60] but also on community agreement on standard mappings, shared CG datasets, and CG-appropriate benchmarks, the same coordination actions recommended for atomistic data in Sec. VII.

Several features set the evaluation of CG models apart from the atomistic case discussed in Sec. V. There are no pointwise energy labels to fit against; the natural reference quantities are distributional (CG-resolution radial distribution functions, free-energy surfaces, folding land scapes) or dynamical, and the matched forces are noisy gradients of a potential of mean force rather than deterministic targets. Uncertainty quantification for CG freeenergy surfaces is correspondingly less developed than for atomistic MLIPs. CG also inherits, in sharper form, the data-compatibility problem that we stress for electronicstructure settings (Sec. IV C): data generated under different mappings or at diferent state points cannot be naively assembled, because the underlying CG potential is itself mapping- and state-point-dependent [89].

Important limitations remain. Transferability across thermodynamic conditions and to non-protein systems is an active research objective [339], and the CG mapping itself fixes what physics is retained.

The mapping—the number and location of the retained degrees of freedom—need not be fixed by hand, and could itself be learned. Most current CG work assumes a chemically motivated mapping and learns only the corresponding free energy surface. Yet the mapping determines what physics is representable, and the retained coordinates do not need to coincide with atomistic or molecular positions: they could be a center of mass, a center of charge, or any other collective coordinate. Learning the optimal map for a desired number of reduced degrees of freedom, under an explicit accuracy or informationpreservation criterion, is therefore a well-posed problem in its own right. Data-driven approaches to select a mapping—autoencoder [345] and graph-based [350] constructions, or variational criteria that optimize the mapping jointly with the model [101, 122]—are a natural way to address it, and the interaction between mapping choice, achievable accuracy for equilibrium properties, and dynamical correctness remains largely unexplored [364].

Thermodynamic consistency, even when achieved, does not deliver correct dynamics. Integrating out fast degrees of freedom removes friction and memory that the retained coordinates experienced implicitly, so a CG model that reproduces the atomistic equilibrium distribution will, in general, reproduce neither its difusion constants nor its transition rates [245]. Recovering physically meaningful timescales requires treating lost degrees of freedom explicitly, for example, through generalized Langevin or Mori–Zwanzig formulations with learned memory kernels [175], or by learning the propagator rather than the potential. The latter connects coarsegraining directly to the learned stochastic integrators and implicit-transfer-operator models of Sec. III D [306]: where a CG potential plus a thermostat cannot be expected to produce the correct kinetics, a learned transition density at CG resolution may be obtained, but at the cost of giving up the explicit energy function. Whether thermodynamically consistent CG potentials and kinetically faithful CG propagators can be unified in a single model is one of the central open problems.

CG simulations need to be further processed: their configurations are typically interpreted, analyzed, or re-coupled to finer descriptions, all of which require reconstructing atomistic detail from CG coordinates. This backmapping step is itself a conditional generative problem and can be addressed with the same flowand difusion-based machinery used for equilibrium sam pling [313, 346, 363], making decoding a natural companion to the encoding implied by the CG mapping. Treating mapping, potential, sampler, and backmapping as parts of one learnable pipeline—rather than as separate, manually bridged stages—is a promising direction that current MLCG eforts only partially realize.

## B. Enhanced sampling with ML potentials

ML potentials accelerate energy and force evaluations compared with quantum chemistry methods, but still suffer from the sampling problem. Many target observables are ensemble or path quantities—free-energy diferences, phase equilibria, nucleation events, reaction rates, conformational change and binding or folding populations— and direct trajectories can remain trapped in metastable basins and lead to intractable computational cost for direct MD simulation even when each force evaluation only takes milliseconds of wall-clock time [250]. A key opportunity is therefore to combine MLIPs with established enhanced-sampling estimators such as umbrella sampling [330], replica exchange [320], metadynamics [186], thermodynamic integration [173], free-energy perturbation [384], nonequilibrium switching [73, 150], and related methods [135].

Metadynamics, and its evolution On-the-fly Probabil ity Enhanced Sampling (OPES) [146], have been recently combined with active learning to build reactive neural network potentials and to obtain free-energy profiles for solution and catalytic chemistry, including urea decomposition in water and several Haber–Bosch-relevant surface reactions [47, 262, 331, 333, 362], and enhanced sampling with MLIPs has been applied at scale to electrified catalytic interfaces [295]. In condensed-phase chemistry, umbrella sampling has been combined with a neural network potential for studying Strecker synthesis [329], alchemically equipped MLIPs have been used with rigorous free-energy perturbation protocols to compute solvation free energies with high accuracy [224], and implicit-solvent ML potentials have been combined with FEP/BAR-style path reweighting to accelerate solvationfree-energy estimation [290]. In materials science, MLIPs have been used in an on-the-fly Bayesian adaptive biasing force method for computing formation free energies in metastable basins [187, 376, 377], and descriptor den sity of states (D-DOS) approaches use the internal representations of MLIPs as collective variables for modelagnostic free-energy estimation [323].

Similar ideas are relevant at coarse resolution: machine-learned coarse-grained potentials define diferentiable thermodynamic models that can be combined with free-energy perturbation to compute free energy differences such as mutation-based changes in folding free energies [60]. These examples suggest that future benchmarks should evaluate not only force and energy errors, but also the stability of biased simulations, uncertainty under extrapolative sampling, and the accuracy of the final thermodynamic or kinetic observable.

## C. Generative samplers: Boltzmann generators and emulators

Generative ML aims to bypass the timescale problem by drawing independent samples directly from a target distribution [250]. One class of generative samplers are Boltzmann generators which combine generative models and statistical mechanical reweighting or Monte-Carlo acceptance to sample from a given equilibrium distribution of the form exp{−u(x)} [239]. This approach was first demonstrated to reproduce equilib rium ensembles of small condensed matter systems and a small protein using normalizing flows [239]; later work extended the approach to other classes of systems, including Lennard-Jones clusters [177] and crystals [299, 355], studied its temperature dependence [80, 225], and introduced flow-matching or difusion models and annealinginspired training [156, 195, 340]. This versatile class of models has also been used in simulations to accelerate the convergence of free energy estimators [298, 354] or as proposals in more traditional Markov chain Monte Carlo settings [108].

A limitation of Boltzmann generators is that they effectively generate the entire molecular structure in a single, global Monte Carlo move, and the requirement to match the proposal and target density with high precision practically limits their application to a few thousand degrees of freedom. Alternatively, Boltzmann emulators such as AlphaFlow [153] or BioEmu [195] are difusion or flow models trained directly to generate ensembles using approximate equilibrium data from MD simulations or other sources. These models generally cannot be reweighted to a prescribed microscopic target density, so the task of learning an MLIP that defines the microscopic energy and the task of eficiently sampling from that energy remain separate. Recent approaches have attempted to bridge this gap by learning energy functions and samplers that are mutually consistent, or by deriving difusion-model scores from a learned energy under constraints that match the difusion-model distribution to the energy-induced distribution [12, 82, 268].

## D. Force-free integrators

A more recent line of work folds force prediction and time integration into a single learned evolution map [10, 36, 39, 81, 289, 306, 327]. By predicting finitetime updates directly, these models aim to overcome the small-step restriction imposed by high-frequency modes, making longer simulation times and larger molecular systems more accessible than with conventional MD. Conceptually, they learn a surrogate for the flow map of the dynamics, amortizing the efect of many small integration steps into one learned update. This approach changes the nature of the approximation problem: by absorbing time integration into the learned evolution map, it shifts the control of integration errors from the numerical solver into the model itself, making long-time accuracy, stabil ity, conservation laws, and transferability central open questions.

For time steps beyond tens of femtoseconds, time integration becomes stochastic as the conditional transition density $p _ { \tau } ( x _ { t + \tau } \mid \ x _ { t } )$ must be sampled in order to represent molecular thermodynamics and kinetics. This leads to generative approaches for time propagation [10, 82, 174, 306]. Such models go beyond pure thermodynamic quantities as enabled by Boltzmann generators and emulators, and instead aim at preserving kinetic properties, such as transition rates, time-correlation functions and kinetic experimental observables, while significantly enhancing sampling beyond direct MD simulation.

![](images/a2c497fa02dde84219edf1b4dad2232707cb137e3e3d04e9b74bbd04b89476da.jpg)  
FIG. 2. A schematic overview of the diferent elements determining the nature and the performance of an atomistic ML model.

## IV. STATE OF THE ART

The evolution of ML models, in this field as in others, is driven by the need to maximize their performance— in terms of the sometimes conflicting goals of accuracy, speed, and breadth of applicability to classes of materials and simulation use cases. There are many diferent possible ways to look at the ingredients that make up a model, but for the purpose of this roadmap we find it useful to discuss the model nature in terms of the physics that it is designed to incorporate, the mathematical architecture, and the type of data that are used (Fig. 2). These three axes are obviously coupled (for instance, a symmetryconstrained architecture will ensure that a model makes predictions consistent with physical symmetries) but not in a trivial way (an unconstrained model can be symmetric to a very high degree if trained with data augmentation).

## A. Physics

Surrogate models of microscopic interactions must be able to describe the underlying physics, which serves at the same time as a consistency constraint and as inspiration for the mathematical structure of the model architecture. First, physical considerations imply that the structure-property mapping must obey some general constraints, such as invariance or equivariance with respect to rigid translations and rotations, as well as some general properties such as smoothness with respect to atomic displacements, relationships between diferent quantities (e.g., forces being the negative gradient of the energy with respect to positions, which ensures that they behave as a conservative vector field) or (with the notable exception of electrostatics, polarization, and dispersion [169]) nearsightedness of interactions [273]. More generally, the construction of an atomistic model benefits from being aware of the specific physical processes it should describe: Coulomb interactions, charge transfer, magnetism, quantum delocalization, electronic excitation, all imply a physical form of the interatomic interactions that can be translated into an appropriate parameterization of the model (such as including magnetic moments on atoms to describe magnetic interactions). Note that it is not necessary for the model to take the physical form too literally. For example, electrostatics can be modeled by learning atomic charges explicitly [118, 176], but also by building descriptors that can capture the appropriate asymptotic behavior of Coulomb interactions [131, 132], by learning atomic charges indirectly using only energy and forces as targets [63], or simply by having charge-like atomic features that can be used to transfer information at long distance using a reciprocal-space electrostaticslike kernel [181, 292].

## B. Model architectures

The physical considerations above are ultimately reflected in the choice of model architecture. For example, to balance exploitation of locality with the ability to incorporate intermediate and long-range interactions, models have used architectures based on atomcentered descriptors [18, 27, 232], that describe the relative position of atoms within a set cutof. Message passing [121, 307, 309] extends the efective receptive field of a per-atom prediction beyond the explicit cutof [25, 237]. As discussed above, genuine long-range information transfer can be achieved by manipulating structural descriptors in reciprocal space, explicitly or implicitly reproducing the functional form of Coulomb interactions. It can also be achieved, in a less transparently physics-motivated manner, using all-to-all transformers [183, 277], linear scaling attention [103] and virtual nodes [57].

Architectures can also be distinguished by the complexity and depth of the functional form they are based on. Shallow models retain a clear raison d’ˆetre. Linear, non-linear and kernel approaches remain competitive and important for small data, fast inference, and distillation, especially when working on a well-defined chemical or configurational space [19, 67, 68, 78, 252, 310]. Modern MLIP architectures, especially when designed to cover large portions of chemical space, are dominated by graph neural networks acting on a neighbor list, with element embeddings that scale gracefully to chemically diverse training data. While most current architectures employ a fixed computational depth, recent implicit architectures introduce adaptive depth, allowing the amount of computation to vary between atomic configurations [212]. Within the GNN family, a clear trend has been the integration of equivariance with respect to rotations and inversions. Models that propagate equivariant features, typically via tensor products on the irreducible represen tations of SO(3) and SO(2), have become very popular in the past few years [21, 23, 25, 104, 107, 198, 231, 257]. High body-order representations [87, 124, 237, 353] and equivariant message passing belong to a family that includes the Atomic Cluster Expansion [74, 87, 92] and its neural relatives [21, 23, 43] that establishes formal completeness of local and semi-local representations [43, 87, 92]. At the same time, irreducible representations are not required for universality—invariant constructions built from inner products are already universal approximators of O(3)-vectorial [341] and general equivariant [83] functions, and similar arguments apply to local message-passing architectures [115]—so the value of equivariant, high-body-order architectures lies in inductive bias and data eficiency rather than in expressiveness in the limit.

At the other end of the spectrum, the need to incorporate symmetries into the structure of the model is increasingly being challenged: unconstrained models that learn approximate equivariance from data, transformerbased potentials that relax the explicit graph locality assumption, and hybrid forms that mix the two have shown that strict symmetry adaptation, while elegant and dataeficient, is not always a hard requirement at the scale of modern training sets [183, 216, 271, 276, 277, 284, 335]. Recent work on plain transformers trained directly on Cartesian coordinates, with no graph and no built-in symmetry, has shown that locality and approximate equivariance can be recovered from data alone, and that empirical neural scaling laws of the kind familiar from language and vision apply also to atomistic models [96, 183]. The trade-of between bias and capacity is one of the more interesting open questions of current model design, and it intersects directly with the discus sion of how to test models (Sec. V): emergent physical constraints, when they appear, may behave diferently from hard-coded ones in out-of-distribution regimes, and require their own validation.

Another architectural choice that clearly reflects the trade-of between exact compliance with physical constraints and a data-centric philosophy is that between conservative and direct force prediction. Conservative forces, obtained as gradients of an energy model, guarantee energy conservation in symplectic integrators, an essential property for thermodynamic sampling and traditionally considered as a quality measure of MD simulations [68]. Direct-force models avoid a gradient computation by predicting forces independently from energies [38, 115, 235], resulting in two- to three-fold faster inference at the cost of energy conservation. Where energy conservation remains desirable in the final model, the speed of direct force prediction can still be exploited: to reduce training cost during a pre-training stage [38, 107], or to accelerate molecular dynamics through multipletime-stepping schemes [334] that use direct forces for the inner steps and conservative forces for the outer ones [38]. The overall suitability of this design choice depends on the application, and several recent models ofer both modes [214, 216, 284].

## C. Data

The growth of training datasets has been one of the most visible trends of the past five years. Materials-focused datasets such as the OQMD [171] and Materials-Project-derived corpora [141, 149], OMat24 [15], MPtrj [76], Alexandria [302], Open-LAM [260], MAD [217], MatterSim [361], MDR [178], GNoME-derived sets [220], SMAX [44] and others have pushed the count of structures with first-principles labels into the hundreds of millions. Molecular datasets such as QM7 [221, 294], QM9 [280], ANI-1x [315], QM7-X [140], MD17/22 [68, 69], SPICE [93], Aquamarine [218], and the GEMS [337], OMol25 [194], OPoly26 [193], QCML [110], and QCell [158] datasets play an analogous role for organic chemistry and biomolecular fragments. Even where dataset sizes look comparable across eforts, the underlying first-principles settings are not. A dataset built with a particular pseudopotential, exchange–correlation functional, dispersion correction, plane-wave cutof and other supporting grid densities, k-point mesh, and smearing is internally consistent but not necessarily compatible with another dataset built with diferent but equally reasonable choices. Magnetism, spin polarization, and Hubbard +U values are particularly problematic [349]: the same system can yield qualitatively diferent reference forces depending on whether collinear, non-collinear, or no spin treatment is used. The accuracy of DFT itself varies dramatically across the periodic table and across bonding regimes, in a way that is well known to electronic-structure practitioners but poorly accounted for in many ML pipelines.

These compatibility problems bias models in ways that are hard to detect downstream: a dataset that is internally consistent will validate well against itself even if it is systematically wrong, and the resulting model will inherit those errors silently. Foundation potentials inherit the same constraint: a model trained on a corpus built with a given functional and pseudopotential set is, in efect, a foundation model for that level of theory, and porting it to a diferent reference (a diferent functional, an all-electron treatment, or a higheraccuracy method) requires either explicit fine-tuning on data at the new level [22, 164] or multi-fidelity strategies [24, 86, 114, 281, 314, 316, 321, 373]. Such multifidelity strategies explicitly model the ofsets between methods and can lead, for example, to a joint global potential energy surface based on data from DFT and correlated wavefunction methods [86]. In principle such approaches enable the training of ML models from a “dataset of opportunity” [100] amalgamated from existing databases of heterogeneous electronic-structure calculations. However, while these methods can benefit from data of diferent sources [170, 200, 312, 357] (potentially even without imposing an accuracy order across sources [100]) they still require the data sources themselves to be internally consistent. As such, it remains key that individual datasets are properly labeled with standardized metadata (functional, pseudopotential version, force-drift checks, sum-of-forces tests, k-point convergence) and verified through automated workflow validation.

Moreover, as all multi-fidelity approaches require a ground-truth anchor point, to which all lower fidelities are related by the employed statistical model, community-curated reference subsets at higher levels of theory are essential.

Recent additions to the high-accuracy electronicstructure toolbox are also driven by AI innovations. Neural density-functional theory keeps the Kohn–Sham variational structure, but replaces part of the handcrafted exchange–correlation approximation by a learned functional. For instance, it was shown that neural exchange– correlation functionals trained with fractional-charge and fractional-spin constraints can reduce major delocalization and static-correlation errors [172]. More recently, Skala, a neural exchange–correlation functional trained on high-accuracy wavefunction data, achieved broad improvements on molecular thermochemistry benchmarks while retaining the O(N<sup>3</sup>) scaling and semi-local cost profile of Kohn–Sham DFT [207]. If these methods become robust across chemical regimes, they will become valuable data generators for the next generation of MLIPs: much more accurate than standard density functionals, but cheap enough to generate larger and more diverse reference datasets than with CCSD(T).

A very high-end AI-enabled electronic structure method is variational Monte Carlo (VMC) with neuralnetwork wavefunctions [55, 138, 266]. In this approach, energies and observables are evaluated as stochastic expectation values over configurations sampled from the neural wavefunction, while its parameters are optimized variationally [55]. Two complementary formulations have been developed. In first quantization, antisymmetric neural wavefunctions defined directly in continuous electronic coordinates, including PauliNet, FermiNet, PsiFormer and neural backflow/message-passing constructions, have reached very high accuracy for atoms, molecules, and extended interacting-electron systems [123, 138, 263, 266]. In second quantization, neural wavefunctions instead parameterize the amplitudes of electronic configurations in an orbital basis, providing a complementary route to ab initio molecular Hamiltonians and strongly correlated electronic structure [14, 55, 70].

The two representations have diferent computational bottlenecks and ofer complementary advantages depending on the system and basis. Recent architectures and implementations, including PsiFormer and Forward-Laplacian-based approaches, have improved accuracy and scaling [123, 197]. Geometry-dependent and transferable neural wavefunctions create opportunities to amortize calculations over potential-energy surfaces rather than isolated geometries, and multi-state formulations extend this direction to excited states [97, 102, 112, 113, 265, 283, 300]. Neural-wavefunction approaches are also being extended beyond the Born–Oppenheimer approximation to coupled electron–nuclear quantum systems, as well as to real-time many-electron dynamics [205, 249]. Despite a polynomial system-size scaling [266, 297], a very large prefactor makes these calculations very expensive, limiting routine applications to a few heavy atoms, and requiring efective core potentials for metals [199, 275]. Nevertheless, neural VMC is becoming a plausible source of high-end reference and fine-tuning data for strongly correlated chemistry, complementing DFT, coupled-cluster, multireference quantum chemistry, and emerging neural-DFT functionals.

A persistent split runs through the data landscape: materials-oriented datasets and molecule-oriented datasets have developed largely independently, and the energy and structural resolution that practitioners consider acceptable on each side difers significantly. The electronic-structure approximations themselves diverge: plane-wave pseudopotential DFT for periodic systems, hybrid functionals and post-Hartree–Fock methods for molecules. Training a single model across both worlds is increasingly common, but the incompatibilities in the underlying data are usually papered over rather than resolved. Recent dataset building eforts emphasize that simply scaling up existing datasets following the same logic as high-throughput searches is not necessarily the most eficient approach, and have demonstrated that smaller datasets (with fewer than 1M structures) can be competitive as long as they are carefully selected to be representative of all classes of the materials and molecules domain, across many energy scales, and target highly converged, internally consistent, and/or higher-level-oftheory electronic-structure settings [160, 217]. This is one of the areas where coordinated community efort would have the highest leverage (Sec. VII): coordination would allow groups to generate datasets that are at the same time diverse, well-curated, and high-accuracy, and larger than can be produced by individual groups.

## D. What is now possible

The applications enabled by current MLIPs are too numerous to do justice to in this section, and we list only a few representative directions [166]. Since MLIPs for complex systems were first developed in the field of materials [29, 30], large-scale and long-time simulations of amorphous and disordered systems, includ ing glasses, liquid–solid interfaces, and complex defects such as grain boundaries and dislocations, are now routine [5, 17, 32, 56, 278, 318, 376]. Solid-state electrolytes, an area where MLIP-driven MD has reduced the cost of computing ionic conductivities by orders of magnitude [46, 77, 119], and heterogeneous catalysis un der realistic temperature, coverage, and pressure condi tions [251], are similar success stories. High-throughput discovery campaigns now use general-purpose potentials to scan thousands to millions of candidates for stabil ity and for properties such as ionic conductivity [220]. On the molecular side, water, as a key system for many questions in chemistry, has been a center of attention for a long time [228, 252, 324]. The controversy over liquid-water structure across exchange–correlation func tionals [120] is now suficiently well resolved that simula tions can target experimentally accessible quantities with quantitative confidence [223], in some cases at essentially coupled-cluster accuracy [75]. Matter under extreme pressure and temperature, long the preserve of shock experiments and diamond-anvil cells, has likewise come within reach of large-scale simulation: MLIPs now make it possible to follow pressure-induced phase transitions, melting curves, and the plastic and shear response of ma terials and minerals at conditions relevant to planetary interiors [44, 65, 236]. Single-molecule chemistry, includ ing reactive pathways and excited-state dynamics, has been pushed forward by both kernel models [67, 68] and neural surrogates [335, 351, 352]. Molecular materials and biomolecular systems remain harder, requiring efi cient architectures and implementations [157]: still, clas sical force fields are competitive in terms of cost, and the timescales of interest are often beyond the reach of MLIP driven MD without using the acceleration strategies dis cussed in Sec. III or hybrid machine learning/molecular mechanics schemes [226, 291].

Several frontiers remain genuinely dificult. As discussed above, long-range electrostatics and dispersion were introduced in MLIPs long ago [11, 142, 227], but are not yet handled in a unified way; explicit chargeequilibration schemes [118, 176, 286], latent long-range messages, and Ewald-like augmentations all coexist, and the suitability of a method is often determined by the physics of the system [111, 176]. Self-consistent treatments are beginning to emerge [13, 20], but the design space is large and remains to be explored. There is not even consensus on when an explicit long-range treatment matters, with some works finding very little practical impact on several bulk properties [371]. Excited states, multiple spin states [95], non-collinear magnetism [88, 287] and systems with strong static correlation push the limits of both the reference data and the model architectures. Furthermore, the indirect benefits of MLIPs are not fully exploited yet: for instance, even though MLIPs make it possible (and in a certain sense necessary) to simulate explicitly the quantum mechanical nature of light nuclei such as hydrogen, the sampling algorithms needed to model quantum nuclear efects [215], although they have been combined with ML [233, 368], are not yet broadly available inside end-to-end modeling tools.

## V. ASSESSING THE QUALITY OF MODELS AND DATA

Constructing an MLIP should never be the final target of a study. It is a step in a longer toolchain that ends in the calculation of a thermodynamic property, a dynamical observable, or a structural prediction, sometimes based on combined information from millions of evaluations. A model that scores well on per-configuration energy or force RMSE of a fixed test set may still produce unphysical trajectories, fail to reproduce melting points or radial distribution functions, or break down in long simulations. Conversely, a model with an apparently mediocre RMSE may be perfectly adequate for the actual question being asked, because the errors average out or cancel between configurations. A further dificulty is that many properties, such as free energies, reaction rates, or transport coeficients, have no reference value to test against: unlike energies, forces, and stresses, they cannot be computed at the reference level of electronic structure theory at acceptable cost, and they emerge from the potential only together with a model of the dynamics and a finite compute budget, so a disagreement with experiments may stem from the potential, the dynamics, or simply too short a simulation. Assessing the quality of MLIPs, and of the data they are trained on, is therefore more complicated than simply validating a regression model, and the community is still developing the tools to perform this task.

## A. Beyond test-set error

The first issue is that the test-set RMSE is necessary but rarely suficient as a quality measure. The distribution of configurations sampled by a thermalized molecular dynamics simulation is not the distribution of the training set; transferability away from equilibrium, smoothness of the potential energy surface, and stability under long integration are all properties that the training loss or the test set error do not directly measure [270]. At the very least, the distribution of errors should always be reported, as well as learning curves as a function of training-set size that provide information on model capacity [204, 294]. The community has accumulated a body of practical experience showing that “beyond RMSE” tests or downstream tasks, including stability checks under high-temperature or near-singular geometries, computation of static observables (radial distribution functions, mean square displacements, elastic constants, defect formation energies), dynamical observables (difusion coeficients, melting temperatures, vibrational spectra), and macroscopic thermodynamic targets, are essential complements to the standard error metrics [147, 229] (cf. Fig. 3). These are also the tests that bring out the relevance of distributions of errors rather than their averages: small parts of a system, such as a reactive site, a defect, a transition state, or an interface, can dominate the physics while contributing negligibly to a global RMSE.

![](images/2bef545695a623e190087e67229e991453957cff5600ed3a31f631c7996ecd9a.jpg)  
FIG. 3. Thorough evaluation of MLIPs requires assessment beyond static error metrics on energies and forces.

A complementary observation is that benchmarks on raw labels are only as good as the labels themselves. Reference energies, forces, and stresses inherit the system atic errors of the underlying first-principles method, so an “error of 1 meV/atom” against DFT does not translate directly into an error of 1 meV/atom against experiment or against a higher-level quantum-chemistry reference. Benchmarks should therefore report the spread across reasonable DFT settings alongside the headline error, annotate data with DFT uncertainty estimates (see Sec. V D), and, where a higher level of theory is available, quote the model error against that reference as well; only then does the reported fidelity reflect the model rather than an arbitrary choice of reference.

## B. Benchmarks: useful, but not as goals in themselves

Benchmarks have been a force for good in atomistic ML. They have improved accessibility, lowered entry barriers, and produced visible prediction accuracy gains across the field [53]. Established platforms such as Matbench Discovery [285], MLIP Arena [66], ML-PEG [162], LAMBench [259], the Fairchem leaderboard [194] and others now ofer standardized comparisons across a substantial subset of the model landscape with various degrees of diversity across the validation sets. The accumulated evidence, however, also suggests that benchmarks become problematic when they turn from a measurement tool into the actual goal of method development, e.g., a pure leaderboard rather than a tool of understanding. Once a benchmark becomes a target for funding, publication, or career visibility, its incentive structure pushes models towards designs that score well on it, sometimes at the cost of performance on out-ofdistribution problems that matter more in practice (see, e.g., the Clever Hans efect [163, 188], in which models predict correctly but for the wrong reason). A subtler problem is that benchmarks erode as the field’s data grows: the more widely a benchmark is adopted, the more likely its test structures are to migrate, usually inadvertently, into the training sets of later models, a form of data leakage that silently inflates apparent accuracy. This failure mode is well documented in other ML communities, most prominently in the contamination of large-language-model evaluations [296], and atomistic ML is not immune. It is a further reason for benchmarks to carry an explicit life-cycle, with periodic refreshing or retirement.

A handful of design principles can keep benchmarks useful and honest:

• they should include physically meaningful diagnostics alongside raw-label errors

• they should be clear about which settings of the reference method were used and why

• they should make a best-efort estimate of the intrinsic noise and uncertainty in the benchmark labels

• they should explicitly discuss the limitations of the reference method used to generate the targets

• they should be neutral with respect to method development, in the sense that the benchmark maintainers and the model developers should not be the same group

• they should also provide a standardized, unbiased assessment of the computational cost

• they should have an explicit life-cycle, including a way to retire benchmarks that have outlived their usefulness

When a benchmark targets an emergent observable (a phase boundary, a reaction free energy, . . . ) rather than a raw microscopic label, there should be strict ac counting of the simulation protocol and costs, preferably over a range of diferent systems, to gauge both transferability and eficiency. Benchmarks should also be openly available to all stakeholders—including experimentalists, engineers, and domain scientists beyond the ML community—so that their scope and relevance can be collectively shaped. A portfolio of complementary tests is preferable to a single number that ranks all models: it is less convenient, but it rewards diversity of approach and makes it harder to game any one test.

## C. Community one-time challenges as a complement to benchmarks

A diferent incentive structure emerges from community one-time challenges. Inspired by CASP in protein structure prediction [366] and the CSP blind test [144] in crystal structure prediction, a challenge periodically asks the community to predict outcomes for problems that are blind to the participants. Challenges are hard to organize and resource-intensive, particularly when they require the acquisition of new experimental measurements as reference, but they have several attractive properties that benchmarks lack: by design, the test set cannot be in the training set (although data leakage can still take place when the new data points are similar); the reference is generated independently, often after the predictions are submitted; and the visibility benefits accrue to the participants who genuinely solve a problem of interest, rather than those who optimize against a fixed leaderboard. They are, moreover, the natural way to assess properties for which no afordable computational reference exists, since an independent experimental measurement can supply the ultimate grounding in physical reality that a calculation cannot. We see one-time challenges as a complement to, not a replacement for, benchmarks: each addresses a diferent question. Indeed, blind challenges often serve to point to areas where current benchmarks fail to be predictive. A particularly attractive variant has the winner of one challenge organize the next, with the constraint that they cannot themselves participate, naturally rotating responsibility while keeping continuity. A challenge with the described characteristics would enable fair evaluation of methodologies while at the same time encouraging a healthy competitive environment among the current participants and inspiring contributions from new actors. Initiatives in this direction are only starting to emerge in the community, e.g., the LAM Crystal Philately competition [260].

## D. Uncertainty and validity

Whether through benchmarks or challenges, end users ultimately need to know not only “which model is most accurate or eficient or both on average”, but “what is the expected accuracy of a given model for my system, and how do I know it has not silently broken down”. Built in uncertainty quantification is a key part of the answer. Ensemble approaches [305], deep evidential regression [7], last-layer prediction rigidity [35], methods to partition uncertainty into basis/representation, local/data-density, and model-form contributions and various calibration schemes are all available with varying overhead [167, 332]; what is missing is an established practice in which uncer tainty estimates accompany every published model, are themselves validated, and are reported through interfaces standard enough for downstream tools to consume. Calibration of the uncertainty is itself an open methodological question, but the community can and should converge on a baseline practice now and refine it later. Considera tion of uncertainties due to model mis-specification [322] as well as approaches based on ensuring internal consis tency of models [274] should complement reliance on calibration to hold-out validation sets. The accuracy of the underlying first-principles method must enter the same conversation: a 5 meV/atom uncertainty in the model might be viewed as meaningless if the reference DFT it self is uncertain at the level of 50 meV/atom for the property of interest, and even for the same functional, results depend on numerical choices such as the pseudopotential [192], although errors of MLIPs and DFT are often of a diferent nature, with the latter being more systematic and therefore leading to better cancellation of errors for derived properties. At the same time, the high flexibility of MLIPs can lead to essentially unusable models for practical simulations beyond some level of pointwise error. Ultimately, one should aim for a holistic error metric that provides the error against experiments. This is not without challenges, since experimental data are abundant for macroscopic observables but scarce for microscopic ones, and are themselves subject to errors; moreover, the error with respect to experiment depends on the accuracy of the MLIP, the accuracy of the reference method, and, for properties extracted from simulation, the dynamics and the amount of sampling; these contributions must be disentangled to know what to improve.

In this regard, the ongoing developments towards the practical estimation of DFT model and numerical errors are worth highlighting. This includes the development of probabilistic DFT models, such as the BEEF family of models [71, 230] and uncertainty-aware functional distributions (UAFD) [134], which replace a single choice of DFT parameters by calibrated distributions, encoding experimental uncertainty. In combination with diferentiable DFT codes [137, 161, 211, 375] these parameter distributions can now be systematically and eficiently propagated to arbitrary quantities of interest such as DFToptimized structures or forces [303]. Using MLIPs to in expensively reproduce calibrated DFT ensembles might provide afordable end-to-end uncertainty quantification that reports directly on the experimental error, rather than on the error relative to a first-principles baseline of unknown accuracy [168]. Furthermore, recent mathematical advances in the estimation of the plane-wave basis set error [54] enable the eficient estimation of basis set errors in DFT forces and other quantities of inter est [303]. As these approaches continue to mature, considering such error estimates during MLIP training becomes viable and ofers hope to systematically quantify the efect of changing DFT parameters (basis set cutof energy, pseudopotential, Hubbard-U parameter, etc.) on MLIP predictions in the future.

Finally, since MLIPs are trained on electronicstructure calculations rather than on experimental measurements [185], their accuracy against experiment is ultimately bounded by that of the reference method: even a model with zero generalization error reproduces the self-interaction and delocalization errors of the functional it was trained on. Crossing this “quantum ceiling” requires training data from higher levels of theory, such as coupled-cluster calculations, whose cost precludes the routine generation of large datasets. Multi-fidelity and ∆-learning strategies [204] (Sec. IV C) and hybrid training against experimental targets (Sec. III A) are the most promising routes to bridge this gap.

## VI. A CHALLENGE OF SOFTWARE AND HARDWARE

The advances of past years have been made possible by an underlying revolution in software and hardware that the atomistic community largely did not drive. The dominant tooling, PyTorch [258] and JAX [50] above all, was designed for deep-learning workloads in industry; the dominant accelerators are GPUs and increasingly specialized artificial intelligence (AI) hardware whose roadmaps are set by a small number of vendors. This has been an enormous boost, and it has strongly influenced the evolu tion of the computational ecosystem (software, hardware, infrastructure). It also creates fragility: the lower layers of the stack evolve at a pace that the typical scientific code base cannot follow without continuous investment, and key low-level components (from vendors) are not always open source.

Historically, atomistic simulations have relied on classical force fields implemented in C++ or Fortran MD engines designed around spatial domain decomposition, whereas modern ML architectures are developed predominantly within Python-centric frameworks such as PyTorch and JAX. Bridging the two requires the model’s structural representations and neighbor-list routines to execute natively on accelerators and within domain-decomposed MD codes [256], which is particularly demanding for equivariant message-passing networks, where tensor products on irreducible representations must be evaluated on the fly [326]. Modern implementations can also execute automated activelearning loops directly on distributed HPC systems, flagging extrapolative configurations during long production runs [210, 244]. Assembling these closed loops, however, still demands expert knowledge across all involved domains, and dedicated workflow frameworks aim to reduce the barrier to entry by packaging recurring operations as reusable, modular building blocks that can be shared across groups and projects [109, 117, 143, 219, 381]. Looking further ahead, agentic interfaces may orchestrate computational workflows directly [267] and coordinate entire parts of the research process [382], which could make high-throughput active-learning pipelines accessible to non-specialists.

![](images/8c356d6872f2a0567a67f473980b1605db2a8f5c1eb75fa09d289b174b8055ea.jpg)  
FIG. 4. The atomistic-ML software and hardware stack, organized as layers of increasing abstraction.

## A. Anatomy of the stack

A useful way to organize the discussion is to think of the atomistic-ML stack as a small number of layers with very diferent characteristics (Fig. 4). At the bottom sit the hardware vendors and the linear algebra, tensor-operation, and communication libraries (BLAS/LAPACK [8, 40, 84, 85, 190], FFT [105], NCCL [248]/RCCL [4], kernel libraries for sparse and equivariant operations). Above them, ML frameworks (such as PyTorch, JAX, TensorFlow [1], MLX [133], Lux.jl [254], Reactant.jl [282] and a long tail of compilers and intermediate representations such as XLA [253], Triton [328], TorchInductor [9]) provide automatic differentiation, kernel scheduling, and a programming surface. On top of these, atomistic-ML libraries provide higher-level building blocks: descriptors, equivariant operations, neighbor lists, dispersion corrections, Ewald summation and particle–mesh routines, and uncertainty estimation. Applications then build on these to define training pipelines, fine-tuning workflows, and inference engines. Finally, simulation drivers, such as LAMMPS [269], GROMACS [2], OpenMM [94], ASE [189] (and a plethora of other workflows built on top of it), i-PI [58], PLUMED [48], NVALCHEMI [246, 247], and the various electronic structure packages, embed the inference engines and orchestrate molecular dynamics, geometry optimization, enhanced sampling, data labeling and analysis.

The trouble is that this stack is currently fragmented in ways that hurt productivity and reproducibility. Each architecture tends to ship with its own training code, its own dataset format, its own neighbor-list implementation, and its own bindings to a subset of simulation drivers. Re-implementations of a LAMMPS or OpenMM interface, with subtle diferences, are a source of repeated eforts and are often poorly structured to leverage typical ML performance strategies such as batching that are strong drivers of hardware and algorithmic development. Low-level operations that should be shared (equivariant tensor products, neighbor-list builders, dispersion corrections, Ewald summation) often are not, and the few cases where reusable libraries have emerged (e3nn [116], sphericart [37], cuEquivariance, OpenEquivariance [33], FlashTP [191], vesin [34], torch-pme [206], matscipy [130], metatensor [34]) have not always resulted in standardized adoption.

The complexity and heterogeneity of the stack increase even further if one considers eforts at the intersection between ML and electronic-structure theory: the large memory footprint of calculations, the diversity of underlying formalisms (plane waves, Gaussian orbitals, numerical atomic orbitals, etc.) and the need for high levels of parallelism make it even harder to conceive a modular design and the definition of interfaces with ML libraries. That said, there are recent examples of autodiferentiable quantum chemistry codes [137, 161, 211, 303, 375] relying on similar software frameworks as the traditional atomistic-ML stack, which are promising for such eforts.

## B. Reasons for monolithic codes, and reasons against

There are reasons why monolithic codes have thrived. Tight integration of the components allows aggressive optimization, simpler governance with clear responsibilities, and a single design philosophy. For graduate students and small groups, a self-contained package gives a clear publishable artifact and avoids the overhead of ne gotiating interfaces with collaborators. Modular ecosystems have their own pathologies: dependency hell, version pinning that breaks downstream tools, the need for coordination across diferent components, the difusion of responsibility when a bug spans two libraries, and the dificulty of tracking citations and giving credit for low-level building blocks. We do not argue for a single architecture, nor for a forced consolidation. We argue that the field has reached a degree of maturity where the costs of fragmentation now outweigh the benefits of unhindered independent development, and where modest investment in shared interfaces would unlock disproportionate gains. This balance is shifting further as AI coding agents mature. As the generation and maintenance of bindings, glue code, and interface boilerplate— including chasing breaking changes in fast-moving upstream libraries—becomes increasingly automated, the implementation efort that once justified folding everything into a single tightly integrated codebase is becom ing a commodity, lowering the barrier to assembling systems from independent, interoperable components.

A useful guiding principle, articulated during the work shop, is to separate algorithms, reference implementations, and performant implementations. The community should agree on a small set of algorithmic primitives that matter (e.g., segmented sparse operations, equivariant tensor products, Ewald-like long-range routines, neighbor-list construction, shape-flexible loss functions). These should have clean reference implementations that are primarily easy to read and modify, and one or more performant implementations, possibly closed-source and vendor-supplied, that are interchangeable behind the same interface. This pattern has worked well in numerical linear algebra and signal processing for decades: there is no fundamental reason it should not work here.

We remark that modern software frameworks and programming languages such as JAX, Julia or PyTorch can in fact blur the distinction between a reference and a performant implementation, in the sense that implementations employing those languages can remain hackable, while still featuring production-grade eficiency [129, 137, 206, 304]. As a result, core computational routines typically remain accessible and not hidden away in low-level kernels written in a separate language.

This is crucial for mathematical research, which requires both the flexibility to explore ideas on wellcontrolled toy problems and the gradual upscaling of algorithms towards the full atomistic modeling setting. In this niche the Julia programming language has grown considerably in popularity and has played a role in the development of new sampling algorithms [42], of the MD engine Molly [129], of algorithms for ACE model training in ACEpotentials.jl [356], and of the aforementioned DFT error estimation strategies [303] in the Density-Functional ToolKit (DFTK) [137]. For such eforts the key advantage of Julia is that custom algorithms (including novel bottom-level linear algebra routines) can be written in a high-level language with only minimal performance impact while still fully integrating with other flagship features such as GPU acceleration or diferen tiability. This makes individual Julia tools competitive for many research tasks, but their integration with standard simulation ecosystems—a prerequisite for wider adoption—remains rudimentary.

## C. Interoperability and isomorphic interfaces

Where standardization has emerged in our field, it has done so organically. The ASE Calculator interface, with its simple positions–species–cell to energy–forces–stress contract, has become the de facto common surface for MLIPs over the past few years, despite never having been formally proposed as a standard, and despite the limitations associated with the Python framework. Useful precedents include LAMMPS’s ML-IAP plugin, OpenKIM [325] for classical force fields, the interfaces of the JuliaMolSim community [242] employed by most Julia-based atomistic simulation software, as well as the metatomic interface that abstracts the MD engine away from the model [34]. Looking forward, an explicit specification of an isomorphic calculator interface, languageagnostic and implementable in Python, Julia, C, C++, Fortran, Rust, or any future language, would be a modest investment with a large potential return for the community. The same logic applies to dataset formats: the field is converging on a small set of options (extended XYZ, HDF5, LMDB, Zarr, Parquet) and would benefit far more from agreement on a common ontology of fields than on a single binary representation. A naming convention that distinguishes “Hirshfeld charges” from “Mulliken charges”, and that allows for new fields to be added without breaking old code, is the kind of work that pays for itself in the medium term.

Plugin-style architectures are a natural complement: a lean core, with heavyweight or domain-specific functionality in optional modules that can evolve and be deprecated independently. ASE itself is moving in this direction. A language-agnostic calculator specification would naturally come with bindings in multiple languages and a stable C ABI that allows production codes (often written in Fortran or C++) to call into ML inference en gines without depending on the entire Python ecosystem. Compiling models to language- and architectureagnostic formats such as the Open Neural Network Exchange (ONNX) or StableHLO on MLIR (Multi-Level Intermediate Representation) has the potential to enable further interoperability. Yet modularity must be designed with a global view of the software stack: plugin boundaries should be chosen so as not to introduce performance bottlenecks, and architectural decisions at the interface level should be informed by end-to-end performance considerations rather than local convenience. By making interface code cheap to write, AI coding agents also raise the stakes for proper design and rigorous testing, which too often are second-class citizens in scientific software development.

## D. Hardware: leverage and lock-in

The atomistic-ML community’s relationship with hardware vendors is asymmetric. Most modern training and inference happens on NVIDIA GPUs, with AMD, Intel, Apple, Tenstorrent, and various TPU and TPU-like accelerators in supporting roles. The de facto centrality of NVIDIA reflects the maturity of CUDA and the opensource ecosystem built around it; this also creates risks. The community has limited leverage to demand support for our specific operations from large vendors, but it has more leverage than it currently uses, and acting as a coordinated voice rather than as a collection of individual groups would help. Concretely, agreeing on a small set of operations that we collectively care about and engaging with vendors and HPC centers around those operations is far more efective than each architecture group negotiating its own kernel support. Vendor-optimized operations can also smooth the transition to new hardware, something that research groups with limited resources struggle to do. The field seems to be moving in this direction, and vendors are engaged in a constructive fashion. The fact that the community has also been actively developing fully open, hardware-agnostic domain libraries mitigates the risk of vendor lock-in.

Performance comparisons across architectures, hardware, and software stacks are notoriously hard. The same algorithm can vary by an order of magnitude in throughput depending on implementation details; reasonable benchmarks must specify hardware, drivers, compilers, batching strategy, and which parts of the pipeline are timed. Whole-MD throughput, including the simulation driver, is the most application-relevant metric and the one most prone to confounding factors. The community would benefit from a standardized harness, ideally maintained by an entity that does not also develop a competing model, that runs end-to-end MD through put tests on a fixed set of hardware. A comprehensive definition of benchmarking protocols, ontologies and a publicly available database of results would be beneficial for the community and the whole HPC ecosystem, including technology providers and funding agencies. In this respect, the HPC community is the natural partner for the scientific community to move forward.

## VII. STRATEGY FOR THE COMMUNITY

The previous sections have surveyed where the field stands and what the key open problems are. We close with a small number of concrete actions that, in our view, would most improve the long-term health of the atomistic-ML ecosystem. None of them is technically novel; their value is in being adopted as community practice rather than left to individual goodwill.

a. Coordinate, do not consolidate. The field benefits from a diversity of architectures, training pipelines, and software stacks. The right level of action is interoperability, not unification, also because the field is still rapidly evolving. It is simply too early to stop significant developments by narrowing down the ecosystem. In practice this means agreeing on a small set of language-agnostic interfaces (a calculator API, a dataset ontology, modelcard metadata) and on a small set of low-level primitives that should be shared across architectures. Several of these already exist (ASE [189], metatensor and its derivatives [34], e3nn [116], sphericart [37], vesin [34], torch-pme [206], NVALCHEMI, the JuliaMolSim interfaces); the action is to formalize their APIs in a collab orative fashion, document them, define their governance and resource their maintenance.

b. Datasets: depth, breadth, and consistency. The community should prioritize the production of higherfidelity reference data, especially for systems and properties where DFT is uncertain (strong correlation, transition metals, magnetism, non-covalent interactions, condensed-phase nuclear quantum efects). Sharing should be supported by metadata standards that record functional, pseudopotential version, k-point grid, spin treatment, and basic sanity-check quantities such as force drift; multi-fidelity strategies should be encouraged where they ofer the most leverage. To make this enforce able rather than aspirational, the community should des ignate a small, public set of reference structures—a few hundred spanning the main bonding regimes—that every contributed dataset recomputes and reports at its own level of theory; the resulting fingerprints make electronicstructure settings comparable across datasets and expose silent incompatibilities before models are trained on them. Computing centers and data hosts can play a much larger role than they currently do, both as longterm repositories and as providers of CI infrastructure that runs sanity checks on contributed datasets, even though long-term support might still rely on individual engagement more than on institutional mechanisms.

c. Benchmarks and challenges. Existing benchmarks should be embraced as part of the ecosystem, but supplemented with diverse, application-driven tests that go beyond label RMSE; with explicit reporting of the underlying reference settings; and with mechanisms to retire benchmarks that have outlived their purpose. Standardized assessment of training and inference cost would provide complementary information to validation accuracy, and drive development to minimize the energetic and environmental impact of the field. In parallel, we propose a regular (e.g., biennial) series of blind challenges paired with a community conference, hosted by CECAM and run by a steering committee that rotates on the winner-organizes-the-next principle of Sec. V C and is resourced by sponsoring institutions. The same committee would act as custodian for the standards called for throughout this roadmap—the calculator API, the dataset ontology, model-card metadata, the shared lowlevel primitives, and the recommended UQ practices— ratifying a versioned release of each at every meeting and naming a maintainer of record, so that the recurring call to “agree as a community” resolves to a named venue and an explicit decision procedure.

d. Controlled model problems. Important theoretical questions highlighted throughout this roadmap—such as when a model extrapolates, whether emergent constraints behave like hard-coded ones, and how representations and architectures limit attainable accuracy—can be sharpened, and sometimes settled, through carefully chosen model problems: reduced settings that isolate specific mechanisms, such as a known long-range tail, an isolated defect, or an analytically tractable energy landscape, in situations where the ground truth is known by construction. Combined with the theoretical tools developed around them, such problems can deepen our understanding of the fundamental capabilities and limitations of ML models, or provide a rigorous basis for substantiating or falsifying empirical claims.

e. Uncertainty quantification as default, not feature. Every model intended for production use should ship with calibrated uncertainty estimates, validated on heldout data, and exposed through a standard interface of per-atom uncertainties and hooks for propagation to derived properties. Uncertainty must be reported alongside the accuracy of the reference method. A small set of community-recommended UQ approaches, with reference implementations, would lower the barrier to adoption.

f. Reward software and infrastructure work. Sustainable shared infrastructure cannot be built on volunteer efort alone. Career paths for research software engineers, with progression comparable to academic ranks, are essential, and several countries have started to provide them; the community should reinforce that trend by giving software contributions visible academic credit. Citation conventions for low-level libraries (CITATION.cf, archival DOIs, explicit acknowledgment in publications) should be enforced as a matter of practice.

g. Engage traditional codes and vendors as partners. The boundary between “traditional” simulation codes and modern ML-driven workflows is increasingly artificial. Electronic structure codes, sampling drivers, and ML inference engines need to interoperate at the level of in-memory data, not text files. Joint workshops, shared CI infrastructure, and joint hires with HPC centers and vendor research groups would accelerate this convergence. Vendor lock-in is a real risk; the antidote is not to refuse vendor support, but to ensure that vendorsupplied implementations sit behind community-defined interfaces that admit alternatives.

[1] Abadi, M., A. Agarwal, P. Barham, E. Brevdo, Z. Chen, C. Citro, G. S. Corrado, A. Davis, J. Dean, M. Devin, S. Ghemawat, I. Goodfellow, A. Harp, G. Irving, M. Isard, Y. Jia, R. Jozefowicz, L. Kaiser, M. Kudlur, J. Levenberg, D. Man´e, R. Monga, S. Moore, D. Murray, C. Olah, M. Schuster, J. Shlens, B. Steiner, I. Sutskever, K. Talwar, P. Tucker, V. Vanhoucke, V. Vasudevan, F. Vi´egas, O. Vinyals, P. Warden, M. Wattenberg, M. Wicke, Y. Yu, and X. Zheng (2015), “Tensor-Flow: Large-scale machine learning on heterogeneous

h. Stewardship. Finally, this roadmap should be maintained as a living document rather than a snapshot: the steering committee proposed above should publish a versioned update of these recommendations after each meeting, tracking which actions have been delivered and which remain open. We invite the broader community to make use of the online and ofline spaces that already exist.[243]

The atomistic machine-learning ecosystem has outgrown its early, exploratory phase. Its scientific impact is now suficient to justify the modest amount of coordination this roadmap recommends, and the cost of inaction, in duplicated efort, lock-in, and lost scientific opportunity, is rising. The opportunity is there. It will not stay open indefinitely.

## AUTHOR CONTRIBUTIONS STATEMENT

JB, MC, CC, GC, A-ME and AK organized the CE-CAM meeting and/or coordinated the on-site discussion, prepared a first draft consolidating the minutes of such discussion and finalized the manuscript. MC coordinated the manuscript preparation and handled the editorial correspondence. All other authors have read the manuscript, provided comments and suggested edits, and agree with the strategic vision laid out in the roadmap. AI was used to summarize workshop notes, and to prepare an early draft of the roadmap based on an extended outline.

## ACKNOWLEDGMENTS

We thank CECAM for hosting the workshop that originated this roadmap, as well as Psi-k and Achira for sponsoring it. We are grateful to all participants for the vig orous and generous discussion that shaped its contents, and the broader atomistic machine-learning community whose contributions, online and ofline, kept informing the analysis after the workshop ended.

systems,” Software available from tensorflow.org.

[2] Abraham, M. J., T. Murtola, R. Schulz, S. P´all, J. C. Smith, B. Hess, and E. Lindahl (2015), SoftwareX 1, 19.

[3] Abramson, J., J. Adler, J. Dunger, R. Evans, T. Green, A. Pritzel, O. Ronneberger, L. Willmore, A. J. Ballard, J. Bambrick, S. W. Bodenstein, D. A. Evans, C.-C. Hung, M. O’Neill, D. Reiman, K. Tunyasuvunakool, Z. Wu, A. Zemgulyt˙e, E. Arvaniti, C. Beat-<sup>ˇ</sup> tie, O. Bertolli, A. Bridgland, A. Cherepanov, M. Con-

greve, A. I. Cowen-Rivers, A. Cowie, M. Figurnov, F. B. Fuchs, H. Gladman, R. Jain, Y. A. Khan, C. M. R. Low, K. Perlin, A. Potapenko, P. Savy, S. Singh, A. Stecula, A. Thillaisundaram, C. Tong, S. Yakneen, E. D. Zhong, M. Zielinski, A. Z´ıdek, V. Bapst, P. Kohli, M. Jader-<sup>ˇ</sup> berg, D. Hassabis, and J. M. Jumper (2024), Nature 630 (8016), 493.

[4] Advanced Micro Devices, Inc., (2026), “ROCm Systems,” .

[5] Allera, A., T. D. Swinburne, A. M. Goryaeva, B. Bienvenu, F. Ribeiro, M. Perez, M.-C. Marinica, and D. Rodney (2025), Nat. Comm. 16, 8367.

[6] Amin, I., S. Raja, and A. Krishnapriyan (2025), in International Conference on Learning Representations, Vol. 2025, edited by Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu, pp. 71153–71175.

[7] Amini, A., W. Schwarting, A. Soleimany, and D. Rus (2020), “Deep evidential regression,” arXiv:1910.02600 [cs.LG].

[8] Angerson, E., Z. Bai, J. Dongarra, A. Greenbaum, A. McKenney, J. Du Croz, S. Hammarling, J. Demmel, C. Bischof, and D. Sorensen (1990), in Supercomputing ’90:Proceedings of the 1990 ACM/IEEE Conference on Supercomputing, pp. 2–11.

[9] Ansel, J., E. Yang, H. He, N. Gimelshein, A. Jain, M. Voznesensky, B. Bao, P. Bell, D. Berard, E. Burovski, G. Chauhan, A. Chourdia, W. Constable, A. Desmaison, Z. DeVito, E. Ellison, W. Feng, J. Gong, M. Gschwind, B. Hirsh, S. Huang, K. Kalambarkar, L. Kirsch, M. Lazos, Y. Liang, J. Liang, Y. Lu, C. Luk, B. Maher, Y. Pan, C. Puhrsch, M. Reso, M. Saroufim, M. Y. Siraichi, H. Suk, S. Zhang, M. Suo, P. Tillet, X. Zhao, E. Wang, K. Zhou, R. Zou, X. Wang, A. Mathews, W. Wen, G. Chanan, P. Wu, and S. Chintala (2024), in Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, ASPLOS ’24, Vol. 2 (Association for Computing Machinery, New York, NY, USA) pp. 929–947.

[10] Antoniadis, P., B. Pavesi, S. Olsson, and O. Winther (2026), “Protein language model embeddings improve generalization of implicit transfer operators,” arXiv:2602.11216.

[11] Artrith, N., T. Morawietz, and J. Behler (2011), Phys. Rev. B 83, 153101.

[12] Arts, M., V. G. Satorras, C.-W. Huang, D. Zuegner, M. Federici, C. Clementi, F. No´e, R. Pinsler, and R. van den Berg (2023), J. Chem. Theory Comput. 19 (18), 6151.

[13] Baldwin, W. J., I. Batatia, M. Vondr´ak, J. T. Margraf, and G. Cs´anyi (2026), “Design space of self–consistent electrostatic machine learning interatomic potentials,” arXiv:2603.14700 [physics.chem-ph].

[14] Barrett, T. D., A. Malyshev, and A. I. Lvovsky (2022), Nature Machine Intelligence 4, 351.

[15] Barroso-Luque, L., M. Shuaibi, X. Fu, B. M. Wood, M. Dzamba, M. Gao, A. Rizvi, C. L. Zitnick, and Z. W. Ulissi (2026), “Open materials 2024 (omat24) inorganic materials dataset and models,” arXiv:2410.12771 [condmat.mtrl-sci].

[16] Bart´ok, A. P., S. De, C. Poelking, N. Bernstein, J. R. Kermode, G. Cs´anyi, and M. Ceriotti (2017), Science Advances 3 (12), e1701816.

[17] Bart´ok, A. P., J. Kermode, N. Bernstein, and G. Cs´anyi (2018), Phys. Rev. X 8 (4), 041048.

[18] Bart´ok, A. P., R. Kondor, and G. Cs´anyi (2013), Physical Review B 87 (18), 184115.

[19] Bart´ok, A. P., M. C. Payne, R. Kondor, and G. Cs´anyi (2010), Physical Review Letters 104 (13), 136403.

[20] Batatia, I., W. J. Baldwin, D. Kuryla, J. Hart, E. Kasoar, A. M. Elena, H. Moore, M. J. Gawkowski, B. X. Shi, V. Kapil, P. Kourtis, I.-B. Magd˘au, and G. Cs´anyi (2026), “Mace-polar-1: A polarisable electrostatic foundation model for molecular chemistry,” arXiv:2602.19411 [physics.chem-ph].

[21] Batatia, I., S. Batzner, D. P. Kov´acs, A. Musaelian, G. N. C. Simm, R. Drautz, C. Ortner, B. Kozinsky, and G. Cs´anyi (2025), Nature Machine Intelligence 7 (1), 56. [22] Batatia, I., P. Benner, Y. Chiang, A. M. Elena, D. P. Kov´acs, J. Riebesell, X. R. Advincula, M. Asta, M. Avaylon, W. J. Baldwin, F. Berger, N. Bernstein, A. Bhowmik, F. Bigi, S. M. Blau, V. C˘arare, M. Ceriotti, S. Chong, J. P. Darby, S. De, F. Della Pia, V. L. Deringer, R. Elijoˇsius, Z. El-Machachi, E. Fako, F. Falcioni, A. C. Ferrari, J. L. A. Gardner, M. J. Gawkowski, A. Genreith-Schriever, J. George, R. E. A. Goodall, J. Grandel, C. P. Grey, P. Grigorev, S. Han, W. Handley, H. H. Heenen, K. Hermansson, C. H. Ho, S. Hofmann, C. Holm, J. Jaafar, K. S. Jakob, H. Jung, V. Kapil, A. D. Kaplan, N. Karimitari, J. R. Kermode, P. Kourtis, N. Kroupa, J. Kullgren, M. C. Kuner, D. Kuryla, G. Liepuoniute, C. Lin, J. T. Margraf, I.- B. Magd˘au, A. Michaelides, J. H. Moore, A. A. Naik, S. P. Niblett, S. W. Norwood, N. O’Neill, C. Ortner, K. A. Persson, K. Reuter, A. S. Rosen, L. A. M. Rosset, L. L. Schaaf, C. Schran, B. X. Shi, E. Sivonxay, T. K. Stenczel, C. Sutton, V. Svahn, T. D. Swinburne, J. Tilly, C. Van Der Oord, S. Vargas, E. Varga-Umbrich, T. Vegge, M. Vondr´ak, Y. Wang, W. C. Witt, T. Wolf, F. Zills, and G. Cs´anyi (2025), The Journal of Chemical Physics 163 (18), 184110.

[23] Batatia, I., D. P. Kovacs, G. Simm, C. Ortner, and G. Csanyi (2022), Advances in Neural Information Processing Systems 35, 11423.

[24] Batatia, I., C. Lin, J. Hart, E. Kasoar, A. M. Elena, S. W. Norwood, T. Wolf, and G. Cs´anyi (2025), “Cross learning between electronic structure theories for unifying molecular, surface, and inorganic crystal foundation force fields,” arXiv:2510.25380 [physics.chem-ph].

[25] Batzner, S., A. Musaelian, L. Sun, M. Geiger, J. P. Mailoa, M. Kornbluth, N. Molinari, T. E. Smidt, and B. Kozinsky (2022), Nat Commun 13 (1), 2453.

[26] Bauer, S., P. Benner, T. Bereau, V. Blum, M. Boley, C. Carbogno, C. R. A. Catlow, G. Dehm, S. Eibl, R. Ernstorfer, A. Fekete, L. Foppa, P. Fratzl, C. Freysoldt, B. Gault, L. M. Ghiringhelli, S. K. Giri, A. Gladyshev, P. Goyal, J. Hattrick-Simpers, L. Kabalan, P. Karpov, M. S. Khorrami, C. T. Koch, S. Kokott, T. Kosch, I. Kowalec, K. Kremer, A. Leitherer, Y. Li, C. H. Liebscher, A. J. Logsdail, Z. Lu, F. Luong, A. Marek, F. Merz, J. R. Mianroodi, J. Neugebauer, Z. Pei, T. A. R. Purcell, D. Raabe, M. Rampp, M. Rossi, J.-M. Rost, J. Saal, U. Saalmann, K. N. Sasidhar, A. Saxena, L. Sbail\`o, M. Scheidgen, M. Schloz, D. F. Schmidt, S. Teshuva, A. Trunschke, Y. Wei, G. Weikum, R. P. Xian, Y. Yao, J. Yin, M. Zhao, and M. Schefler (2024), Modelling and Sim-

ulation in Materials Science and Engineering 32 (6), 063301.

[27] Behler, J. (2011), The Journal of Chemical Physics 134 (7), 074106.

[28] Behler, J. (2021), Chemical Reviews 121 (16), 10037.

[29] Behler, J., R. Martoˇn´ak, D. Donadio, and M. Parrinello (2008), Phys. Rev. Lett. 100, 185501.

[30] Behler, J., and M. Parrinello (2007), Physical Review Letters 98 (14), 146401.

[31] Ben Mahmoud, C., A. Anelli, G. Cs´anyi, and M. Ceriotti (2020), Phys. Rev. B 102 (23), 235130.

[32] Bernstein, N., B. Bhattarai, G. Cs´anyi, D. A. Drabold, S. R. Elliott, and V. L. Deringer (2019), Angew. Chem. Int. Ed. 58 (21), 7057.

[33] Bharadwaj, V., A. Glover, A. Bulu¸c, and J. Demmel (2025), 2025 Proceedings of the Conference on Applied and Computational Discrete Algorithms (ACDA) , 32.

[34] Bigi, F., J. W. Abbott, P. Loche, A. Mazitov, D. Tisi, M. F. Langer, A. Goscinski, P. Pegolo, S. Chong, R. Goswami, P. Febrer, S. Chorna, M. Kellner, M. Ceriotti, and G. Fraux (2026), The Journal of Chemical Physics 164 (6), 064113.

[35] Bigi, F., S. Chong, M. Ceriotti, and F. Grasselli (2024), Mach. Learn.: Sci. Technol. 5 (4), 045018.

[36] Bigi, F., S. Chong, A. Kristiadi, and M. Ceriotti (2025), in The Thirty-Ninth Annual Conference on Neural Information Processing Systems.

[37] Bigi, F., G. Fraux, N. J. Browning, and M. Ceriotti (2023), The Journal of Chemical Physics 159 (6), 064802.

[38] Bigi, F., M. F. Langer, and M. Ceriotti (2025), in Proceedings of the 42nd International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 267, edited by A. Singh, M. Fazel, D. Hsu, S. Lacoste-Julien, F. Berkenkamp, T. Maharaj, K. Wagstaf, and J. Zhu (PMLR) pp. 4384–4414.

[39] Bigi, F., J. Spies, and M. Ceriotti (2026), Phys. Rev. Lett. 136, 237301.

[40] Blackford, L. S., J. Demmel, J. Dongarra, I. Duf, S. Hammarling, G. Henry, M. Heroux, L. Kaufman, A. Lumsdaine, A. Petitet, R. Pozo, K. Remington, and R. C. Whaley (2002), ACM Trans. Math. Softw. 28 (2), 135.

[41] Blank, T. B., S. D. Brown, A. W. Calhoun, and D. J. Doren (1995), The Journal of Chemical Physics 103 (10), 4129.

[42] Blassel, N., and G. Stoltz (2024), Journal of Statistical Physics 191, 10.1007/s10955-024-03230-x.

[43] Bochkarev, A., Y. Lysogorskiy, and R. Drautz (2024), Physical Review X 14, 021036.

[44] Bochkarev, A., Y. Lysogorskiy, A. Subramanyam, R. Drautz, and D. Perez (2026), “Exploring the extremes: Atomic basis for multi-elemental materials science under complex thermodynamic conditions,” .

[45] Bogojeski, M., L. Vogt-Maranto, M. E. Tuckerman, K.- R. M¨uller, and K. Burke (2020), Nature communications 11 (1), 5223.

[46] B¨ohm, J., and A. Champagne (2026), Advanced Intelligent Systems 8 (6), e202501382.

[47] Bonati, L., D. Polino, C. Pizzolitto, P. Biasi, R. Eckert, S. Reitmeier, R. Schl¨ogl, and M. Parrinello (2023), Proceedings of the National Academy of Sciences 120 (50), e2313023120.

[48] Bonomi, M., G. Bussi, C. Camilloni, G. A. Tribello, P. Ban´aˇs, A. Barducci, M. Bernetti, P. G. Bolhuis, S. Bottaro, D. Branduardi, R. Capelli, P. Carloni, M. Ceriotti, A. Cesari, H. Chen, W. Chen, F. Colizzi, S. De, M. De La Pierre, D. Donadio, V. Drobot, B. Ensing, A. L. Ferguson, M. Filizola, J. S. Fraser, H. Fu, P. Gasparotto, F. L. Gervasio, F. Giberti, A. Gil-Ley, T. Giorgino, G. T. Heller, G. M. Hocky, M. Iannuzzi, M. Invernizzi, K. E. Jelfs, A. Jussupow, E. Kirilin, A. Laio, V. Limongelli, K. Lindorf-Larsen, T. L¨ohr, F. Marinelli, L. Martin-Samos, M. Masetti, R. Meyer, A. Michaelides, C. Molteni, T. Morishita, M. Nava, C. Paissoni, E. Papaleo, M. Parrinello, J. Pfaendtner, P. Piaggi, G. Piccini, A. Pietropaolo, F. Pietrucci, S. Pipolo, D. Provasi, D. Quigley, P. Raiteri, S. Raniolo, J. Rydzewski, M. Salvalaglio, G. C. Sosso, V. Spiwok, J. Sponer, D. W. H. Swenson, P. Tiwary, O. Vals-<sup>ˇ</sup> son, M. Vendruscolo, G. A. Voth, A. White, and The PLUMED consortium (2019), Nature Methods 16 (8), 670.

[49] Bowman, G. R., E. R. Bolin, K. M. Hart, B. C. Maguire, and S. Marqusee (2015), Proc. Natl. Acad. Sci. USA 112 (9), 2734.

[50] Bradbury, J., R. Frostig, P. Hawkins, M. J. Johnson, Y. Katariya, C. Leary, D. Maclaurin, G. Necula, A. Paszke, J. VanderPlas, S. Wanderman-Milne, and Q. Zhang (2018), “JAX: composable transformations of python+numpy programs,” http://github. com/jax-ml/jax.

[51] Briling, K. R., Y. Calvino Alonso, A. Fabrizio, and C. Corminboeuf (2024), J. Chem. Theory Comput. 20 (3), 1108.

[52] Brockherde, F., L. Vogt, L. Li, M. E. Tuckerman, K. Burke, and K. R. M¨uller (2017), Nature Communications 8 (1), 872.

[53] Butler, K. T., K. Choudhary, G. Csanyi, A. M. Ganose, S. V. Kalinin, and D. Morgan (2024), npj Computational Materials 10 (1), 231.

[54] Canc\`es, E., G. Dusson, G. Kemlin, and A. Levitt (2022), SIAM Journal on Scientific Computing 44, B1312.

[55] Carleo, G., and M. Troyer (2017), Science 355 (6325), 602.

[56] Caro, M. A., A. Aarva, V. L. Deringer, G. Cs´anyi, and T. Laurila (2018), Chem. Mater. 30 (21), 7446.

[57] Caruso, A., J. Venturin, L. Giambagli, E. Rolando, Z. El-Machachi, F. No´e, and C. Clementi (2026), Nat. Commun. 17 (1), 10.1038/s41467-026-69715-3.

[58] Ceriotti, M., J. More, and D. E. Manolopoulos (2014), Computer Physics Communications 185 (3), 1019.

[59] Chandrasekaran, A., D. Kamal, R. Batra, C. Kim, L. Chen, and R. Ramprasad (2019), npj Comput Mater 5 (1), 22.

[60] Charron, N. E., K. Bonneau, A. S. Pasos-Trejo, A. Guljas, Y. Chen, F. Musil, J. Venturin, D. Gusew, I. Zaporozhets, A. Kr¨amer, C. Templeton, A. Kelkar, A. E. P. Durumeric, S. Olsson, A. P´erez, M. Majewski, B. E. Husic, A. Patel, G. De Fabritiis, F. No´e, and C. Clementi (2025), Nat. Chem. 17 (8), 1284.

[61] Chen, C., and S. P. Ong (2022), Nature Computational Science 2 (11), 718.

[62] Chen, H., G. Hautier, A. Jain, C. Moore, B. Kang, R. Doe, L. Wu, Y. Zhu, Y. Tang, and G. Ceder (2012), Chem. Mater. 24 (11), 2009.

[63] Cheng, B. (2025), npj Computational Materials 11 (1), 10.1038/s41524-025-01577-7.

[64] Cheng, B., E. A. Engel, J. Behler, C. Dellago, and M. Ceriotti (2019), Proceedings of the National Academy of Sciences of the United States of America 116 (4), 1110.

[65] Cheng, B., G. Mazzola, C. J. Pickard, and M. Ceriotti (2020), Nature 585 (7824), 217.

[66] Chiang, Y., T. Kreiman, C. Zhang, M. C. Kuner, E. Weaver, I. Amin, H. Park, Y. Lim, J. Kim, D. Chrzan, A. Walsh, S. M. Blau, M. Asta, and A. S. Krishnapriyan (2025), “Mlip arena: Advancing fairness and transparency in machine learning interatomic potentials via an open, accessible benchmark platform,” arXiv:2509.20630 [physics.chem-ph].

[67] Chmiela, S., H. E. Sauceda, K.-R. M¨uller, and A. Tkatchenko (2018), Nat Commun 9 (1), 3887.

[68] Chmiela, S., A. Tkatchenko, H. E. Sauceda, I. Poltavsky, K. T. Sch¨utt, and K.-R. M¨uller (2017), Sci. Adv. 3 (5), e1603015.

[69] Chmiela, S., V. Vassilev-Galindo, O. T. Unke, A. Kabylda, H. E. Sauceda, A. Tkatchenko, and K.- R. M¨uller (2023), Science Advances 9 (2), eadf0873.

[70] Choo, K., A. Mezzacapo, and G. Carleo (2020), Nature Communications 11, 2368.

[71] Christensen, R., T. Bligaard, and K. W. Jacobsen (2020), in Uncertainty Quantification in Multiscale Materials Modeling, Elsevier Series in Mechanics of Advanced Materials, edited by Y. Wang and D. L. Mc-Dowell (Woodhead Publishing) pp. 77–91.

[72] Clementi, C. (2008), Curr. Opin. Struct. Biol. 18 (1), 10.

[73] Crooks, G. E. (1999), Physical Review E 60 (3), 2721.

[74] Darby, J. P., D. P. Kov´acs, I. Batatia, M. A. Caro, G. L. W. Hart, C. Ortner, and G. Cs´anyi (2023), Phys. Rev. Lett. 131, 028001.

[75] Daru, J., H. Forbert, J. Behler, and D. Marx (2022), Phys. Rev. Lett. 129, 226001.

[76] Deng, B., P. Zhong, K. Jun, J. Riebesell, K. Han, C. J. Bartel, and G. Ceder (2023), Nature Machine Intelligence 5 (9), 1031.

[77] Deng, Z., C. Chen, X.-G. Li, and S. P. Ong (2019), npj Computational Materials 5 (1), 75.

[78] Deringer, V. L., A. P. Bart´ok, N. Bernstein, D. M. Wilkins, M. Ceriotti, and G. Cs´anyi (2021), Chem. Rev. 121 (16), 10073.

[79] Deringer, V. L., M. A. Caro, and G. Cs´anyi (2019), Advanced Materials 31 (46), 1902765.

[80] Dibak, M., L. Klein, A. Kr¨amer, and F. No´e (2022), Physical Review Research 4 (4), 10.1103/physrevresearch.4.l042005.

[81] Diez, J. V., M. Schreiner, and S. Olsson (2026), Science Advances 12 (15), 10.1126/sciadv.aed2333.

[82] Diez, J. V., M. J. Schreiner, O. Engkvist, and S. Olsson (2025), in The Thirteenth International Conference on Learning Representations.

[83] Domina, M., F. Bigi, P. Pegolo, and M. Ceriotti (2025), The Journal of Chemical Physics 163 (16), 164114.

[84] Dongarra, J. J., J. Du Croz, S. Hammarling, and I. S. Duf (1990), ACM Trans. Math. Softw. 16 (1), 1.

[85] Dongarra, J. J., J. Du Croz, S. Hammarling, and R. J. Hanson (1988), ACM Trans. Math. Softw. 14 (1), 1.

[86] Dral, P. O., A. Owens, A. Dral, and G. Cs´anyi (2020), The Journal of Chemical Physics 152,

10.1063/5.0006498.

[87] Drautz, R. (2019), Physical Review B 99 (1), 014104.

[88] Drautz, R. (2020), Phys. Rev. B 102 (2), 024104.

[89] Dunn, N. J. H., T. T. Foley, and W. G. Noid (2016), Accounts of Chemical Research 49 (12), 2832.

[90] Durumeric, A. E. P., N. E. Charron, C. Templeton, F. Musil, K. Bonneau, A. S. Pasos-Trejo, Y. Chen, A. Kelkar, F. No´e, and C. Clementi (2023), Curr. Opin. Struct. Biol. 79, 102533.

[91] Durumeric, A. E. P., Y. Chen, A. S. Pasos-Trejo, F. No´e, and C. Clementi (2026), Nat. Commun. 17 (1), 10.1038/s41467-026-70818-0.

[92] Dusson, G., M. Bachmayr, G. Cs´anyi, R. Drautz, S. Etter, C. van der Oord, and C. Ortner (2022), Journal of Computational Physics 454, 110946.

[93] Eastman, P., P. K. Behara, D. L. Dotson, R. Galvelis, J. E. Herr, J. T. Horton, Y. Mao, J. D. Chodera, B. P. Pritchard, Y. Wang, G. De Fabritiis, and T. E. Markland (2023), Sci Data 10 (1), 11.

[94] Eastman, P., R. Galvelis, R. P. Pel´aez, C. R. Abreu, S. E. Farr, E. Gallicchio, A. Gorenko, M. M. Henry, F. Hu, J. Huang, et al. (2023), The Journal of Physical Chemistry B 128 (1), 109.

[95] Eckhof, M., and J. Behler (2021), npj Comp. Mater. 7, 170.

[96] Eissler, M., T. Korjakow, S. Ganscha, O. T. Unke, K.-R. M¨uller, and S. Gugler (2026), The Journal of Chemical Physics 164 (9), 10.1063/5.0295035.

[97] Entwistle, M. T., Z. Sch¨atzle, P. A. Erdman, J. Hermann, and F. No´e (2023), Nature Communications 14, 274.

[98] Fabrizio, A., K. R. Briling, D. D. Girardier, and C. Corminboeuf (2020), J. Chem. Phys. 153 (20), 10.1063/5.0033326.

[99] Fedik, N., B. Nebgen, N. Lubbers, K. Barros, M. Kulichenko, Y. W. Li, R. Zubatyuk, R. Messerly, O. Isayev, and S. Tretiak (2023), The Journal of Chemical Physics 159 (11), 110901.

[100] Fisher, K. E., M. F. Herbst, and Y. M. Marzouk (2024), The Journal of Chemical Physics 161, 014114.

[101] Foley, T. T., M. S. Shell, and W. G. Noid (2015), The Journal of Chemical Physics 143 (24), 243104.

[102] Foster, A., Z. Sch¨atzle, P. B. Szab´o, L. Cheng, J. K¨ohler, G. Cassella, N. Gao, J. Li, F. No´e, and J. Hermann (2026), Nat. Commun. 17, 9976.

[103] Frank, J. T., S. Chmiela, K.-R. M¨uller, and O. T. Unke (2026), Nat. Mach. Intell. 8 (3), 388.

[104] Frank, J. T., O. T. Unke, K.-R. M¨uller, and S. Chmiela (2024), Nature Communications 15 (1), 6539.

[105] Frigo, M. (2004), SIGPLAN Not. 39 (4), 642.

[106] Fu, X., B. M. Wood, L. Barroso-Luque, D. S. Levine, M. Gao, M. Dzamba, and C. L. Zitnick (2025), in Proceedings of the 42nd International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 267, edited by A. Singh, M. Fazel, D. Hsu, S. Lacoste-Julien, F. Berkenkamp, T. Maharaj, K. Wagstaf, and J. Zhu (PMLR) pp. 17875–17893.

[107] Fu, X., B. M. Wood, L. Barroso-Luque, D. S. Levine, M. Gao, M. Dzamba, and C. L. Zitnick (2025), “Learning smooth and expressive interatomic potentials for physical property prediction,” arXiv:2502.12147 [physics.comp-ph].

[108] Gabri´e, M., G. M. Rotskof, and E. Vanden-Eijnden (2022), Proceedings of the National Academy of Sci-

ences 119 (10), e2109420119.

[109] Ganose, A. M., H. Sahasrabuddhe, M. Asta, K. Beck, T. Biswas, A. Bonkowski, J. Bustamante, X. Chen, Y. Chiang, D. C. Chrzan, J. Clary, O. A. Cohen, C. Ertural, M. C. Gallant, J. George, S. Gerits, R. E. A. Goodall, R. D. Guha, G. Hautier, M. Horton, T. J. Inizan, A. D. Kaplan, R. S. Kingsbury, M. C. Kuner, B. Li, X. Linn, M. J. McDermott, R. S. Mohanakrishnan, A. N. Naik, J. B. Neaton, S. M. Parmar, K. A. Persson, G. Petretto, T. A. R. Purcell, F. Ricci, B. Rich, J. Riebesell, G.-M. Rignanese, A. S. Rosen, M. Schefler, J. Schmidt, J.-X. Shen, A. Sobolev, R. Sundararaman, C. Tezak, V. Trinquet, J. B. Varley, D. Vigil-Fowler, D. Wang, D. Waroquiers, M. Wen, H. Yang, H. Zheng, J. Zheng, Z. Zhu, and A. Jain (2025), Digital Discovery 4 (7), 1944.

[110] Ganscha, S., O. T. Unke, D. Ahlin, H. Maennel, S. Kashubin, and K.-R. M¨uller (2025), Scientific data 12 (1), 406.

[111] Gao, A., and R. C. Remsing (2022), Nat Commun 13 (1), 1572.

[112] Gao, N., and S. G¨unnemann (2022), in International Conference on Learning Representations.

[113] Gao, N., and S. G¨unnemann (2023), in Proceedings of the 40th International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 202 (PMLR) pp. 10708–10726.

[114] Gardner, J. L. A., H. Schulz, J. Helie, L. Sun, and G. N. C. Simm (2025), “Understanding multifidelity training of machine-learned force-fields,” arXiv:2506.14963 [physics].

[115] Gasteiger, J., F. Becker, and S. G¨unnemann (2021), in Advances in Neural Information Processing Systems, Vol. 34, edited by M. Ranzato, A. Beygelzimer, Y. Dauphin, P. Liang, and J. W. Vaughan (Curran Associates, Inc.) pp. 6790–6802.

[116] Geiger, M., and T. Smidt (2022), “e3nn: Euclidean neural networks,” arXiv:2207.09453 [cs.LG].

[117] Gelˇzinyt˙e, E., S. Wengert, T. K. Stenczel, H. H. Heenen, K. Reuter, G. Cs´anyi, and N. Bernstein (2023), The Journal of Chemical Physics 159 (12), 124801.

[118] Ghasemi, S. A., A. Hofstetter, S. Saha, and S. Goedecker (2015), Physical Review B 92 (4), 045131.

[119] Gigli, L., D. Tisi, F. Grasselli, and M. Ceriotti (2024), Chem. Mater. 36 (3), 1482.

[120] Gillan, M. J., D. Alf\`e, and A. Michaelides (2016), The Journal of Chemical Physics 144 (13), 130901.

[121] Gilmer, J., S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl (2017), in Proceedings of the 34th International Conference on Machine Learning - Volume 70, ICML’17 (JMLR.org) pp. 1263–1272.

[122] Giulini, M., R. Menichetti, M. S. Shell, and R. Potestio (2020), Journal of Chemical Theory and Computation 16 (11), 6795.

[123] von Glehn, I., J. S. Spencer, and D. Pfau (2023), in International Conference on Learning Representations.

[124] Glielmo, A., C. Zeni, and A. De Vita (2018), Physical Review B 97 (18), 184307, 1801.04823.

[125] Golze, D., M. Hirvensalo, P. Hern´andez-Le´on, A. Aarva, J. Etula, T. Susi, P. Rinke, T. Laurila, and M. A. Caro (2022), Chemistry of Materials 34 (14), 6240.

[126] Gong, X., H. Li, N. Zou, R. Xu, W. Duan, and Y. Xu (2023), Nature Communications 14 (1), 2848.

[127] Goswami, R., M. Masterov, S. Kamath, A. Pena-Torres, and H. J´onsson (2025), Journal of Chemical Theory and Computation 21 (16), 7935.

[128] Greeley, J., T. F. Jaramillo, J. Bonde, I. Chorkendorf, and J. K. Nørskov (2006), Nature Materials 5, 909.

[129] Greener, J. G. (2024), Chemical Science 15, 4897.

[130] Grigorev, P., L. Fr´erot, F. Birks, A. Gola, J. Golebiowski, J. Grießer, J. L. H¨ormann, A. Klemenz, G. Moras, W. G. N¨ohring, J. A. Oldenstaedt, P. Patel, T. Reichenbach, T. Rocke, L. Shenoy, M. Walter, S. Wengert, L. Zhang, J. R. Kermode, and L. Pastewka (2024), J. Open Source Softw. 9 (93), 5668.

[131] Grisafi, A., and M. Ceriotti (2019), J. Chem. Phys. 151 (20), 204105.

[132] Grisafi, A., J. Nigam, and M. Ceriotti (2021), Chem. Sci. 12 (6), 2078.

[133] Hannun, A., J. Digani, A. Katharopoulos, and R. Collobert (2023), “MLX: Eficient and flexible machine learning on apple silicon,” .

[134] Hansen, T., J. J. Mortensen, T. Bligaard, and K. W. Jacobsen (2025), Phys. Rev. B 112 (7), 075412.

[135] H´enin, J., T. Leli\`evre, M. R. Shirts, O. Valsson, and L. Delemotte (2022), Living Journal of Computational Molecular Science 4 (1), 1583.

[136] Henkelman, G., and H. J´onsson (1999), The Journal of Chemical Physics 111 (15), 7010.

[137] Herbst, M. F., A. Levitt, and E. Canc\`es (2021), Proceedings of the JuliaCon Conference 3 (26), 69.

[138] Hermann, J., Z. Sch¨atzle, and F. No´e (2020), Nature Chemistry 12, 891.

[139] Ho, C. H., C. van der Oord, J. P. Darby, T. Keane, R. L. Benson, C. R. Espinoza, R. Kulkarni, E. Spinu, M. Papanikolaou, R. Tomsett, R. M. Forrest, J. J. Bean, G. Cs´anyi, and C. Ortner (2026), “Equivariant manybody message passing interatomic potentials for magnetic materials,” arXiv:2604.08143 [cond-mat.mtrl-sci].

[140] Hoja, J., L. Medrano Sandonas, B. G. Ernst, A. Vazquez-Mayagoitia, R. A. DiStasio Jr., and A. Tkatchenko (2021), Sci Data 8 (1), 43.

[141] Horton, M. K., P. Huck, R. X. Yang, J. M. Munro, S. Dwaraknath, A. M. Ganose, R. S. Kingsbury, M. Wen, J. X. Shen, T. S. Mathis, A. D. Kaplan, K. Berket, J. Riebesell, J. George, A. S. Rosen, E. W. C. Spotte-Smith, M. J. McDermott, O. A. Cohen, A. Dunn, M. C. Kuner, G.-M. Rignanese, G. Petretto, D. Waroquiers, S. M. Grifin, J. B. Neaton, D. C. Chrzan, M. Asta, G. Hautier, S. Cholia, G. Ceder, S. P. Ong, A. Jain, and K. A. Persson (2025), Nature Materials 24 (10), 1522.

[142] Houlding, S., S. Y. Liem, and P. L. A. Popelier (2007), Int. J. Quantum Chem. 107, 2817.

[143] Huber, S. P., S. Zoupanos, M. Uhrin, L. Talirz, L. Kahle, R. H¨auselmann, D. Gresch, T. M¨uller, A. V. Yakutovich, C. W. Andersen, F. F. Ramirez, C. S. Adorf, F. Gargiulo, S. Kumbhar, E. Passaro, C. Johnston, A. Merkys, A. Cepellotti, N. Mounet, N. Marzari, B. Kozinsky, and G. Pizzi (2020), Scientific Data 7 (1), 300.

[144] Hunnisett, L. M., J. Nyman, N. Francia, N. S. Abraham, C. S. Adjiman, S. Aitipamula, T. Alkhidir, M. Almehairbi, A. Anelli, D. M. Anstine, J. E. Anthony, J. E. Arnold, F. Bahrami, M. A. Bellucci, R. M. Bhardwaj, I. Bier, J. A. Bis, A. D. Boese, D. H. Bowskill, J. Bramley, J. G. Brandenburg, D. E. Braun, P. W. V.

R. Couch, R. Cuadrado, T. Darden, G. M. Day, H. Dietrich, Y. Ding, A. DiPasquale, B. Dhokale, B. P. van Eijck, M. R. J. Elsegood, D. Firaha, W. Fu,

Butler, J. Cadden, S. Carino, E. J. Chan, C. Chang, B. Cheng, S. M. Clarke, S. J. Coles, R. I. Cooper,

Z. Liu, Z.-P. Liu, J. W. Lubach, N. Marom, A. A. Maryewski, H. Matsui, A. Mattei, R. A. Mayo, J. W. Melkumov, S. Mohamed, Z. Momenzadeh Abardeh,

[145] Husic, B. E., N. E. Charron, D. Lemm, J. Wang, A. P´erez, M. Majewski, A. Kr¨amer, Y. Chen, S. Olsson, G. de Fabritiis, F. No´e, and C. Clementi (2020), J. Chem. Phys. 153 (19), 194101.

[146] Invernizzi, M., and M. Parrinello (2020), The Journal of Physical Chemistry Letters 11 (7), 2731–2736.

[147] Isamura, B. K., O. Aten, M. Nosratjoo, and P. L. A. Popelier (2026), Communications Chemistry 9 (1), 138.

[148] Izvekov, S., and G. A. Voth (2005), The Journal of Physical Chemistry B 109 (7), 2469.

[149] Jain, A., S. P. Ong, G. Hautier, W. Chen, W. D. Richards, S. Dacek, S. Cholia, D. Gunter, D. Skinner, G. Ceder, and K. A. Persson (2013), APL Materials 1 (1), 011002.

[150] Jarzynski, C. (1997), Physical Review Letters 78 (14), 2690.

[151] Jia, W., H. Wang, M. Chen, D. Lu, L. Lin, R. Car, E. Weinan, and L. Zhang (2020), SC20: International Conference for High Performance Computing, Networking, Storage and Analysis , 1.

[152] Jin, J., A. J. Pak, A. E. P. Durumeric, T. D. Loose, and G. A. Voth (2022), Journal of Chemical Theory and Computation 18 (10), 5759.

[153] Jing, B., B. Berger, and T. Jaakkola (2024), in Proceedings of the 41st International Conference on Machine Learning, ICML’24 (JMLR.org).

[154] John, S. T., and G. Cs´anyi (2017), The Journal of Physical Chemistry B 121 (48), 10934.

[155] J´onsson, H., G. Mills, and K. W. Jacobsen (1998), in Classical and Quantum Dynamics in Condensed Phase

Simulations, edited by B. J. Berne, G. Ciccotti, and D. F. Coker (World Scientific) pp. 385–404.

[156] Joshi, C. K., X. Fu, Y.-L. Liao, V. Gharakhanyan, B. K. Miller, A. Sriram, and Z. W. Ulissi (2025), in Fortysecond International Conference on Machine Learning.

[157] Kabylda, A., J. T. Frank, S. Su´arez-Dou, A. Khabibrakhmanov, L. Medrano Sandonas, O. T. Unke, S. Chmiela, K.-R. M¨uller, and A. Tkatchenko (2025), J. Am. Chem. Soc. 147 (37), 33723.

[158] Kabylda, A., S. Su´arez-Dou, N. Davoine, F. N. Br¨unig, and A. Tkatchenko (2026), AI Sci. 2 (2), 025003.

[159] Kaniselvan, M., B. K. Miller, M. Gao, J. Nam, and D. S. Levine (2026), in The Fourteenth International Conference on Learning Representations.

[160] Kaplan, A. D., R. Liu, J. Qi, T. W. Ko, B. Deng, J. Riebesell, G. Ceder, K. A. Persson, and S. P. Ong (2025), “A foundational potential energy surface dataset for materials,” .

[161] Kasim, M. F., S. Lehtola, and S. M. Vinko (2022), The Journal of Chemical Physics 156, 084801.

[162] Kasoar, E., J. Hart, I. Batatia, A. Elena, and G. Cs´anyi (2025), “ml-peg,” .

[163] Kaufmann, J., J. Dippel, L. Ruf, W. Samek, K.-R. M¨uller, and G. Montavon (2025), Nature Machine Intelligence 7 (3), 412.

[164] Kaur, H., F. Della Pia, I. Batatia, X. R. Advincula, B. X. Shi, J. Lan, G. Cs´anyi, A. Michaelides, and V. Kapil (2025), Faraday Discussions 256, 120.

[165] Kavanagh, S. R., C. W. Tan, M. Wang, M. L. Descoteaux, G. de Miranda Nascimento, U. Unneberg, L. Zichi, F. Libbi, N. Rivano, A. Glover, V. Bharadwaj, A. Johansson, W. C. Witt, A. Musaelian, and B. Kozinsky (2026), “Fast and accurate equivariant foundation models for atomistic simulation,” arXiv:2607.28461 [physics.comp-ph].

[166] Keith, J. A., V. Vassilev-Galindo, B. Cheng, S. Chmiela, M. Gastegger, K.-R. Muller, and A. Tkatchenko (2021), Chemical reviews 121 (16), 9816.

[167] Kellner, M., and M. Ceriotti (2024), Mach. Learn.: Sci. Technol. 5 (3), 035006.

[168] Kellner, M., T. Hansen, T. Bligaard, K. W. Jacobsen, and M. Ceriotti (2026), “Errors that matter: Uncertainty-aware universal machine-learning potentials calibrated on experiments,” .

[169] Khabibrakhmanov, A., M. Gori, C. M¨uller, and A. Tkatchenko (2025), Journal of the American Chemical Society 147 (44), 40763.

[170] Kim, J., J. You, Y. Park, Y. Lim, Y. Kang, J. Kim, H. Jeon, S. Ju, D. Hong, S. Y. Lee, S. Choi, Y. Kim, J. W. Lee, and S. Han (2026), Nature Communications 17 (1), 3432.

[171] Kirklin, S., J. E. Saal, B. Meredig, A. Thompson, J. W. Doak, M. Aykol, S. R¨uhl, and C. Wolverton (2015), npj Computational Materials 1 (1), 15010.

[172] Kirkpatrick, J., B. McMorrow, D. H. P. Turban, A. L. Gaunt, J. S. Spencer, A. G. D. G. Matthews, A. Obika, L. Thiry, M. Fortunato, D. Pfau, L. R. Castellanos, S. Petersen, A. W. R. Nelson, P. Kohli, P. Mori-S´anchez, D. Hassabis, and A. J. Cohen (2021), Science 374 (6573), 1385.

[173] Kirkwood, J. G. (1935), The Journal of Chemical Physics 3 (5), 300.

[174] Klein, L., A. Y. K. Foong, T. E. Fjelde, B. K. Mlodozeniec, M. Brockschmidt, S. Nowozin, F. No´e, and

R. Tomioka (2023), Advances in Neural Information Processing Systems 36, 52863.

[175] Klippenstein, V., M. Tripathy, G. Jung, F. Schmid, and N. F. A. van der Vegt (2021), The Journal of Physical Chemistry B 125 (19), 4931.

[176] Ko, T. W., J. A. Finkler, S. Goedecker, and J. Behler (2021), Nat Commun 12 (1), 398.

[177] K¨ohler, J., L. Klein, and F. Noe (2020), in Proceedings of the 37th International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 119, edited by H. D. III and A. Singh (PMLR) pp. 5361–5370.

[178] Koker, T., A. Gangan, M. Kotak, J. Marian, and T. Smidt (2026), “Pft: Phonon fine-tuning for machine learned interatomic potentials,” arXiv:2601.07742 [cond-mat.mtrl-sci].

[179] Koker, T., M. Kotak, and T. Smidt (2025), “Training a foundation model for materials on a budget,” arXiv:2508.16067 [physics.comp-ph].

[180] Kong, S., F. Ricci, D. Guevarra, J. B. Neaton, C. P. Gomes, and J. M. Gregoire (2022), Nature Communications 13 (1), 949.

[181] Kosmala, A., J. Gasteiger, N. Gao, and S. G¨unnemann (2023), in Proceedings of the 40th International Conference on Machine Learning, ICML’23, Vol. 202 (JMLR.org) pp. 17544–17563.

[182] Kr¨amer, A., A. E. P. Durumeric, N. E. Charron, Y. Chen, C. Clementi, and F. No´e (2023), J. Phys. Chem. Lett. 14 (17), 3970.

[183] Kreiman, T., Y. Bai, F. Atieh, E. Weaver, E. Qu, and A. S. Krishnapriyan (2025), “Transformers discover molecular structure without graph priors,” arXiv:2510.02259 [cs.LG].

[184] Kreiman, T., and A. S. Krishnapriyan (2026), Digital Discovery 5 (1), 415.

[185] Kulichenko, M., J. S. Smith, B. Nebgen, Y. W. Li, N. Fedik, A. I. Boldyrev, N. Lubbers, K. Barros, and S. Tretiak (2021), The Journal of Physical Chemistry Letters 12 (26), 6227.

[186] Laio, A., and M. Parrinello (2002), Proceedings of the National Academy of Sciences 99 (20), 12562.

[187] Lapointe, C., A. Zhong, T. D. Swinburne, F. Bruneval, M. Ath\`enes, and M.-C. Marinica (2025), Phys. Rev. Mater. 9, 093801.

[188] Lapuschkin, S., S. W¨aldchen, A. Binder, G. Montavon, W. Samek, and K.-R. M¨uller (2019), Nature communications 10 (1), 1096.

[189] Larsen, A., J. Mortensen, J. Blomqvist, I. Castelli, R. Christensen, M. Dulak, J. Friis, M. Groves, B. Hammer, C. Hargus, E. Hermes, P. Jennings, P. Jensen, J. Kermode, J. Kitchin, E. Kolsbjerg, J. Kubal, K. Kaasbjerg, S. Lysgaard, J. Maronsson, T. Maxson, T. Olsen, L. Pastewka, A. Peterson, C. Rostgaard, J. Schiøtz, O. Sch¨utt, M. Strange, K. Thygesen, T. Vegge, L. Vilhelmsen, M. Walter, Z. Zeng, and K. W. Jacobsen (2017), J. Phys. Condens. Matter 29 (27), 273002.

[190] Lawson, C. L., R. J. Hanson, D. R. Kincaid, and F. T. Krogh (1979), ACM Trans. Math. Softw. 5 (3), 308.

[191] Lee, S. Y., H. Kim, Y. Park, D. Jeong, S. Han, Y. Park, and J. W. Lee (2025), in Proceedings of the 42nd International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 267, edited by A. Singh, M. Fazel, D. Hsu, S. Lacoste-Julien,

F. Berkenkamp, T. Maharaj, K. Wagstaf, and J. Zhu (PMLR) pp. 33143–33156.

[192] Lejaeghere, K., G. Bihlmayer, T. Bjorkman, P. Blaha, S. Blugel, V. Blum, D. Caliste, I. E. Castelli, S. J. Clark, A. Dal Corso, S. de Gironcoli, T. Deutsch, J. K. Dewhurst, I. Di Marco, C. Draxl, M. Du ak, O. Eriksson, J. A. Flores-Livas, K. F. Garrity, L. Genovese, P. Giannozzi, M. Giantomassi, S. Goedecker, X. Gonze, O. Granas, E. K. U. Gross, A. Gulans, F. Gygi, D. R. Hamann, P. J. Hasnip, N. A. W. Holzwarth, D. Iu an, D. B. Jochym, F. Jollet, D. Jones, G. Kresse, K. Koepernik, E. Kucukbenli, Y. O. Kvashnin, I. L. M. Locht, S. Lubeck, M. Marsman, N. Marzari, U. Nitzsche, L. Nordstrom, T. Ozaki, L. Paulatto, C. J. Pickard, W. Poelmans, M. I. J. Probert, K. Refson, M. Richter, G.-M. Rignanese, S. Saha, M. Schefler, M. Schlipf, K. Schwarz, S. Sharma, F. Tavazza, P. Thunstrom, A. Tkatchenko, M. Torrent, D. Vanderbilt, M. J. van Setten, V. Van Speybroeck, J. M. Wills, J. R. Yates, G.-X. Zhang, and S. Cottenier (2016), Science 351 (6280), aad3000.

[193] Levine, D. S., N. Liesen, L. Chua, J. Difenderfer, H. Ingolfsson, M. P. Kroonblawd, N. Kumar, A. Maiti, S. S. Mohottalalage, M. Shuaibi, B. V. Essen, B. M. Wood, C. L. Zitnick, S. M. Blau, and E. R. Antoniuk (2026), “The open polymers 2026 (opoly26) dataset and evaluations,” arXiv:2512.23117 [physics.chem-ph].

[194] Levine, D. S., M. Shuaibi, E. W. C. Spotte-Smith, M. G. Taylor, M. R. Hasyim, K. Michel, I. Batatia, G. Cs´anyi, M. Dzamba, P. Eastman, N. C. Frey, X. Fu, V. Gharakhanyan, A. S. Krishnapriyan, J. A. Rackers, S. Raja, A. Rizvi, A. S. Rosen, Z. Ulissi, S. Vargas, C. L. Zitnick, S. M. Blau, and B. M. Wood (2026), “The open molecules 2025 (omol25) dataset, evaluations, and models,” arXiv:2505.08762 [physics.chem-ph].

[195] Lewis, S., T. Hempel, J. Jim´enez-Luna, M. Gastegger, Y. Xie, A. Y. K. Foong, V. G. Satorras, O. Abdin, B. S. Veeling, I. Zaporozhets, Y. Chen, S. Yang, A. E. Foster, A. Schneuing, J. Nigam, F. Barbero, V. Stimper, A. Campbell, J. Yim, M. Lienen, Y. Shi, S. Zheng, H. Schulz, U. Munir, R. Sordillo, R. Tomioka, C. Clementi, and F. No´e (2025), Science 389 (6761), 10.1126/science.adv9817.

[196] Li, H., Z. Wang, N. Zou, M. Ye, R. Xu, X. Gong, W. Duan, and Y. Xu (2022), Nature Computational Science 2 (6), 367.

[197] Li, R., H. Ye, D. Jiang, X. Wen, C. Wang, Z. Li, X. Li, D. He, J. Chen, W. Ren, and L. Wang (2024), Nature Machine Intelligence 6, 209.

[198] Li, T., W. Li, A. Peng, J. Xue, L. Zhang, D. Zhang, and H. Wang (2026), “Dpa4: Pushing the accuracycost frontier of interatomic potentials with emfa so(2) convolution,” arXiv:2606.02419 [physics.chem-ph].

[199] Li, X., C. Fan, W. Ren, and J. Chen (2022), Physical Review Research 4 (1), 10.1103/physrevresearch.4.013021.

[200] Liang, T., K. Xu, E. Lindgren, Z. Chen, R. Zhao, J. Liu, E. Berger, B. Tang, B. Zhang, Y. Wang, K. Song, P. Ying, N. Xu, H. Dong, S. Chen, P. Erhart, Z. Fan, T. Ala-Nissila, and J. Xu (2026), Nature Computational Science 6 (7), 789.

[201] Liao, Y.-L., A. J. Hofman, S. C. Shen, A. Duval, S. W. Norwood, and T. Smidt (2026), “Equiformerv3: Scaling eficient, expressive, and general se(3)-equivariant graph

attention transformers,” arXiv:2604.09130 [cs.LG].

[202] Liebl, K., and G. A. Voth (2025), Journal of Chemical Theory and Computation 21 (9), 4846.

[203] Lifson, S., and A. Warshel (1968), The Journal of Chemical Physics 49 (11), 5116.

[204] von Lilienfeld, O. A., K.-R. M¨uller, and A. Tkatchenko (2020), Nature Reviews Chemistry 4 (7), 347.

[205] Linteau, D., S. Moroni, G. Carleo, and M. Holzmann (2026), Physical Review Research 8, L022007.

[206] Loche, P., K. K. Huguenin-Dumittan, M. Honarmand, Q. Xu, E. Rumiantsev, W. B. How, M. F. Langer, and M. Ceriotti (2025), The Journal of Chemical Physics 162 (14), 142501.

[207] Luise, G., C.-W. Huang, T. Vogels, D. P. Kooi, S. Ehlert, S. Lanius, K. J. H. Giesbertz, A. Karton, D. Gunceler, M. Stanley, W. P. Bruinsma, L. Huang, X. Wei, J. Garrido Torres, A. Katbashev, B. M´at´e, S.- O. Kaba, R. Sordillo, Y. Chen, D. B. Williams-Young, C. M. Bishop, J. Hermann, R. van den Berg, and P. Gori-Giorgi (2025), “Accurate and scalable exchangecorrelation with deep learning,”

[208] Luong, H. T., A. Front, T. Zarrouk, H. Jiang, K. Meinander, V. Durairaj, S. Mousavihashemi, G. Pauls, C. Guizani, V. Siipola, K. S. Kontturi, T. Tammelin, M. A. Caro, and T. Laurila (2026), Chemistry of Materials 38 (8), 4245.

[209] Lysogorskiy, Y., A. Bochkarev, and R. Drautz (2026), npj Computational Materials 12 (1), 114.

[210] Lysogorskiy, Y., A. Bochkarev, M. Mrovec, and R. Drautz (2023), Physical Review Materials 7 (4), 043801.

[211] M. Casares, P. A., J. S. Baker, M. Medvidovi´c, R. d. Reis, and J. M. Arrazola (2024), The Journal of Chemical Physics 160 (6), 062501.

[212] Maeß, J., L. Werner, J. T. Frank, W. Ripken, M. Michajlow, J. Futterer, K.-R. M¨uller, and S. Chmiela (2026), “Implicit machine learning force fields accelerate molecular dynamics simulations,” .

[213] Majewski, M., A. P´erez, P. Th¨olke, S. Doerr, N. E. Charron, T. Giorgino, B. E. Husic, C. Clementi, F. No´e, and G. De Fabritiis (2023), Nature Communications 14, 5739.

[214] Malosso, C., F. Bigi, P. Pegolo, J. W. Abbott, P. Loche, M. Rossi, M. Ceriotti, and A. Mazitov (2026), arXiv preprint arXiv:2603.02089 10.48550/arXiv.2603.02089.

[215] Markland, T. E., and M. Ceriotti (2018), Nature Reviews Chemistry 2 (3), 0109.

[216] Mazitov, A., F. Bigi, M. Kellner, P. Pegolo, D. Tisi, G. Fraux, S. Pozdnyakov, P. Loche, and M. Ceriotti (2025), Nat Commun 16 (1), 10653.

[217] Mazitov, A., S. Chorna, G. Fraux, M. Bercx, G. Pizzi, S. De, and M. Ceriotti (2025), Sci Data 12 (1), 1857.

[218] Medrano Sandonas, L., D. Van Rompaey, A. Fallani, M. Hilfiker, D. Hahn, L. Perez-Benito, J. Verhoeven, G. Tresadern, J. Kurt Wegner, H. Ceulemans, et al. (2024), Scientific Data 11 (1), 742.

[219] Menon, S., Y. Lysogorskiy, A. L. M. Knoll, N. Leimeroth, M. Poul, M. Qamar, J. Janssen, M. Mrovec, J. Rohrer, K. Albe, J. Behler, R. Drautz, and J. Neugebauer (2024), npj Computational Materials 10 (1), 261.

[220] Merchant, A., S. Batzner, S. S. Schoenholz, M. Aykol, G. Cheon, and E. D. Cubuk (2023), Nature 624 (7990), 80.

[221] Montavon, G., K. Hansen, S. Fazli, M. Rupp, F. Biegler, A. Ziehe, A. Tkatchenko, A. V. Lilienfeld, and K.- R. M¨uller (2012), in Advances in Neural Information Processing Systems 25 , edited by F. Pereira, C. J. C. Burges, L. Bottou, and K. Q. Weinberger (Curran Associates, Inc.) pp. 440–448.

[222] Montavon, G., M. Rupp, V. Gobre, A. Vazquez-Mayagoitia, K. Hansen, A. Tkatchenko, K.-R. M¨uller, and O. Anatole von Lilienfeld (2013), New Journal of Physics 15 (9), 095003.

[223] Montero De Hijes, P., C. Dellago, R. Jinnouchi, and G. Kresse (2024), The Journal of Chemical Physics 161 (13), 131102.

[224] Moore, J. H., D. J. Cole, and G. Cs´anyi (2026), Journal of the American Chemical Society 148 (5), 4928.

[225] Moqvist, S., W. Chen, M. Schreiner, F. N¨uske, and S. Olsson (2025), Journal of Chemical Theory and Computation 21 (5), 2535.

[226] Morado, J. a., K. Zinovjev, L. O. Hedges, D. J. Cole, and J. Michel (2025), Journal of Chemical Theory and Computation 21 (22), 11805, https://pubs.acs.org/jctcce/articlepdf/21/22/11805/41282566/ct5c01464.pdf.

[227] Morawietz, T., and J. Behler (2013), The Journal of Physical Chemistry A 117 (32), 7356–7366.

[228] Morawietz, T., A. Singraber, C. Dellago, and J. Behler (2016), PNAS 113, 8368.

[229] Morrow, J. D., J. L. A. Gardner, and V. L. Deringer (2023), The Journal of Chemical Physics 158 (12), 10.1063/5.0139611.

[230] Mortensen, J. J., K. Kaasbjerg, S. L. Frederiksen, J. K. Nørskov, J. P. Sethna, and K. W. Jacobsen (2005), Physical Review Letters 95, 10.1103/physrevlett.95.216401.

[231] Musaelian, A., S. Batzner, A. Johansson, L. Sun, C. J. Owen, M. Kornbluth, and B. Kozinsky (2023), Nat Commun 14 (1), 579.

[232] Musil, F., A. Grisafi, A. P. Bart´ok, C. Ortner, G. Cs´anyi, and M. Ceriotti (2021), Chem. Rev. 121 (16), 9759.

[233] Musil, F., I. Zaporozhets, F. No´e, C. Clementi, and V. Kapil (2022), J. Chem. Phys. 157 (18), 10.1063/5.0120386.

[234] Navarro, C., M. Majewski, and G. De Fabritiis (2023), Journal of Chemical Theory and Computation 19 (21), 7518.

[235] Neumann, M., J. Gin, B. Rhodes, S. Bennett, Z. Li, H. Choubisa, A. Hussey, and J. Godwin (2024), “Orb: A fast, scalable neural network potential,” arXiv:2410.22570 [cond-mat.mtrl-sci].

[236] Nguyen-Cong, K., J. T. Willman, S. G. Moore, A. B. Belonoshko, R. Gayatri, E. Weinberg, M. A. Wood, A. P. Thompson, and I. I. Oleynik (2021), in Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, SC ’21 (ACM) p. 1–12.

[237] Nigam, J., S. Pozdnyakov, G. Fraux, and M. Ceriotti (2022), J. Chem. Phys. 156 (20), 204115.

[238] Nigam, J., M. J. Willatt, and M. Ceriotti (2022), J. Chem. Phys. 156 (1), 014115.

[239] No´e, F., S. Olsson, J. K¨ohler, and H. Wu (2019), Science 365 (6457), eaaw1147.

[240] Noid, W. G., J.-W. Chu, G. S. Ayton, V. Krishna, S. Izvekov, G. A. Voth, A. Das, and H. C. Ander-

sen (2008), The Journal of Chemical Physics 128 (24), 244114.

[241] CECAM (Centre Europ´een de Calcul Atomique et Mol´eculaire) is a European organization that hosts workshops in the area of atomic-scale simulations of matter. The web page of the workshop can be consulted at https://www.cecam.org/workshop-details/ a-roadmap-for-an-atomistic-machine-learning-sof

[242] For the AtomsCalculator interface of the JuliaMol-Sim community see https://juliamolsim.github.io/ AtomsCalculators.jl/stable/interface.

[243] An online community space is available at https:// ml4atoms.org/slack.

[244] Novikov, I. S., K. Gubaev, E. V. Podryabinkin, and A. V. Shapeev (2021), Machine Learning: Science and Technology 2 (2), 025002.

[245] N¨uske, F., L. Boninsegna, and C. Clementi (2019), J. Chem. Phys. 151 (4), 10.1063/1.5100131.

[246] NVIDIA Corporation, (2025), “NVIDIA ALCHEMI Toolkit: A developer toolkit for accelerating ai in chemistry and material science,” https://github.com/ NVIDIA/nvalchemi-toolkit.

[247] NVIDIA Corporation, (2025), “NVIDIA ALCHEMI Toolkit-Ops: Optimized batch kernels to accelerate computational chemistry and material science workflows,” https://github.com/NVIDIA/ nvalchemi-toolkit-ops.

[248] NVIDIA Corporation, (2026), “NVIDIA Collective Communications Library (NCCL),” .

[249] Nys, J., G. Pescia, A. Sinibaldi, and G. Carleo (2024), Nature Communications 15, 9404.

[251] Omranpour, A., J. Elsner, K. N. Lausch, and J. Behler (2025), ACS Catalysis 15, 1616.

[250] Olsson, S. (2026), Current Opinion in Structural Biology 96, 103213.

[252] Omranpour, A., P. M. D. Hijes, J. Behler, and C. Dellago (2024), J. Chem. Phys. 160, 170901.

[253] OpenXLA Contributors, (2023), “XLA: A machine learning compiler for GPUs, CPUs, and ML accelerators,” https://github.com/openxla/xla.

[254] Pal, A. (2023), “Lux: Explicit Parameterization of Deep Neural Networks in Julia,” .

[255] Panosetti, C., A. Engelmann, L. Nemec, K. Reuter, and J. T. Margraf (2020), Journal of Chemical Theory and Computation 16 (4), 2181.

[256] Park, Y., J. Kim, S. Hwang, and S. Han (2024), Journal of Chemical Theory and Computation 20 (11), 4857, https://pubs.acs.org/jctcce/articlepdf/20/11/4857/1779290/ct4c00190.pdf.

[257] Passaro, S., and C. L. Zitnick (2023), in Proceedings of the 40th International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 202, edited by A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett (PMLR) pp. 27420– 27438.

[258] Paszke, A., S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. K¨opf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala (2019), “Pytorch: an imperative style, high-performance deep learning library,” in Proceedings of the 33rd International Conference on Neural Information Processing Systems (Curran Associates Inc., Red Hook, NY, USA).

[259] Peng, A., C. Cai, M. Guo, D. Zhang, C. Zhang, W. Jiang, Y. Wang, A. Loew, C. Wu, W. E, L. Zhang, and H. Wang (2026), npj Computational Materials 12 (1), 62.

[260] Peng, A., X. Liu, M.-Y. Guo, L. Zhang, and H. Wang (2025), Machine Learning: Science and Technology 6 (2), 020701.

ware-ecosystem-1376.[261] Peng, F., Y. Sun, C. J. Pickard, R. J. Needs, Q. Wu, and Y. Ma (2017), Phys. Rev. Lett. 119, 107001.

[262] Perego, S., and L. Bonati (2024), npj Computational Materials 10, 291.

[263] Pescia, G., J. Nys, J. Kim, A. Lovato, and G. Carleo (2024), Physical Review B 110, 035108.

[264] Peter, C., and K. Kremer (2009), Soft Matter 5 (22), 4357.

[265] Pfau, D., S. Axelrod, H. Sutterud, I. von Glehn, and J. S. Spencer (2024), Science 385 (6711), eadn0137.

[266] Pfau, D., J. S. Spencer, A. G. D. G. Matthews, and W. M. C. Foulkes (2020), Physical Review Research 2, 033429.

[267] Pham, T. D., A. Tanikanti, and M. Ke¸celi (2026), Communications Chemistry 9 (1), 33.

[268] Plainer, M., H. Wu, L. Klein, S. G¨unnemann, and F. Noe (2025), in Advances in Neural Information Processing Systems, Vol. 38, edited by D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (Curran Associates, Inc.) pp. 27728–27773.

[269] Plimpton, S. (1995), Journal of Computational Physics 117 (1), 1.

[270] Poltavsky, I., M. Puleva, A. Charkin-Gorbulin, G. Fonseca, I. Batatia, N. J. Browning, S. Chmiela, M. Cui, J. Thorben Frank, S. Heinen, B. Huang, S. K¨aser, A. Kabylda, D. Khan, C. M¨uller, A. J. A. Price, K. Riedmiller, K. T¨opfer, T. Wai Ko, M. Meuwly, M. Rupp, G. Cs´anyi, O. A. von Lilienfeld, J. T. Margraf, K.-R. M¨uller, and A. Tkatchenko (2025), Chem. Sci. 16 (8), 3738.

[271] Pozdnyakov, S., and M. Ceriotti (2023), Advances in Neural Information Processing Systems 36, 79469.

[272] Pozdnyakov, S. N., M. J. Willatt, A. P. Bart´ok, C. Ortner, G. Cs´anyi, and M. Ceriotti (2020), Physical Review Letters 125, 166001.

[273] Prodan, E., and W. Kohn (2005), Proceedings of the National Academy of Sciences 102 (33), 11635.

[274] Qian, C., V. Vitartas, J. R. Kermode, and R. J. Maurer (2026), Npj Comput. Mater. 12 (1), 169.

[275] Qian, Y., W. Fu, W. Ren, and J. Chen (2022), The Journal of Chemical Physics 157 (16), 164104.

[276] Qu, E., and A. S. Krishnapriyan (2024), Advances in Neural Information Processing Systems 37, 139030.

[277] Qu, E., B. M. Wood, A. S. Krishnapriyan, and Z. W. Ulissi (2026), “A recipe for scalable attention-based mlips: unlocking long-range accuracy with all-to-all node attention,” arXiv:2603.06567 [cs.LG].

[278] Quaranta, V., M. Hellstr¨om, and J. Behler (2017), J. Phys. Chem. Lett. 8, 1476.

[279] Radova, M., W. G. Stark, C. S. Allen, R. J. Maurer, and A. P. Bart´ok (2025), npj Computational Materials 11 (1), 237.

[280] Ramakrishnan, R., P. O. Dral, M. Rupp, and O. A. Von Lilienfeld (2014), Scientific Data 1, 140022.

[281] Ramakrishnan, R., P. O. Dral, M. Rupp, and O. A. Von Lilienfeld (2015), Journal of Chemical Theory and Computation 11 (5), 2087, 1503.04987v1.

[282] Reactant.jl Contributors, (2024), “Reactant.jl: Optimize Julia functions with MLIR and XLA,” https: //github.com/EnzymeAD/Reactant.jl.

[283] Rende, R., L. L. Viteritti, F. Becca, A. Scardicchio, A. Laio, and G. Carleo (2025), Nature Communications 16, 7213.

[284] Rhodes, B., S. Vandenhaute, V. Simkus, J. Gin, J. God-<sup>ˇ</sup> win, T. Duignan, and M. Neumann (2025), “Orb-v3: atomistic simulation at scale,” arXiv:2504.06231 [condmat.mtrl-sci].

[285] Riebesell, J., R. E. A. Goodall, P. Benner, Y. Chiang, B. Deng, G. Ceder, M. Asta, A. A. Lee, A. Jain, and K. A. Persson (2025), Nature Machine Intelligence 7 (6), 836.

[286] Rinaldi, M., A. Bochkarev, Y. Lysogorskiy, and R. Drautz (2025), Physical Review Materials 9 (3), 033802.

[287] Rinaldi, M., M. Mrovec, A. Bochkarev, Y. Lysogorskiy, and R. Drautz (2024), npj Computational Materials 10 (1), 12.

[288] Garijo del R´ıo, E., J. J. Mortensen, and K. W. Jacobsen (2019), Physical Review B 100 (10), 10.1103/physrevb.100.104103.

[289] Ripken, W., M. Plainer, G. Lied, T. Frank, O. T. Unke, S. Chmiela, F. No´e, and K. R. M¨uller (2026), arXiv preprint arXiv:2601.22123 10.48550/arXiv.2601.22123.

[290] R¨ocken, S., A. F. Burnet, and J. Zavadlav (2024), The Journal of Chemical Physics 161 (23), 234101.

[291] Rufa, D. A., H. E. Bruce Macdonald, J. Fass, M. Wieder, P. B. Grinaway, A. E. Roitberg, O. Isayev, and J. D. Chodera (2020), BioRxiv , 2020.

[292] Rumiantsev, E., M. F. Langer, T.-E. Sodjargal, M. Ceriotti, and P. Loche (2026), Transactions on Machine Learning Research .

[293] Rupp, M., R. Ramakrishnan, and O. A. von Lilienfeld (2015), J. Phys. Chem. Lett. 6 (16), 3309.

[294] Rupp, M., A. Tkatchenko, K.-R. M¨uller, and O. A. von Lilienfeld (2012), Physical Review Letters 108 (5), 058301.

[295] Sahoo, S. J., M. Maraschin, J. B. Varley, D. S. Levine, Z. Ulissi, C. L. Zitnick, W. Takemura, J. A. Gauthier, N. Govindarajan, and M. Shuaibi (2026), “Insights into co dimerization at electrified cu interfaces from largescale machine learning simulations,” arXiv:2509.17862 [cond-mat.mtrl-sci].

[296] Sainz, O., J. Campos, I. Garc´ıa-Ferrero, J. Etxaniz, O. L. de Lacalle, and E. Agirre (2023), Findings of the Association for Computational Linguistics: EMNLP 2023 , 10776.

[297] Sch¨atzle, Z., P. B. Szab´o, M. Mezera, J. Hermann, and F. No´e (2023), The Journal of Chemical Physics 159 (9), 094108.

[298] Schebek, M., J. He, E. Hofmann, Y. Du, F. No´e, and J. Rogal (2026), The Journal of Chemical Physics 164 (18), 10.1063/5.0320214.

[299] Schebek, M., F. No´e, and J. Rogal (2026), Nature Communications 17 (1), 10.1038/s41467-026-73900-9.

[300] Scherbela, M., R. Reisenhofer, L. Gerard, P. Marquetand, and P. Grohs (2022), Nature Computational Science 2, 331.

[301] Schienbein, P. (2023), J. Chem. Theory Comp. 19, 705.

[302] Schmidt, J., N. Hofmann, H.-C. Wang, P. Borlido, P. J. Carri¸co, T. F. Cerqueira, S. Botti, and M. A. Marques (2023), Advanced Materials 35 (22), 2210788.

[303] Schmitz, N. F., B. Ploumhans, and M. F. Herbst (2025), npj Computational Materials 12, 6.

[304] Schoenholz, S. S., and E. D. Cubuk (2021), J. Stat. Mech. 2021 (12), 124016.

[305] Schran, C., K. Brezina, and O. Marsalek (2020), J. Chem. Phys. 153, 104105.

[306] Schreiner, M., O. Winther, and S. Olsson (2023), Advances in Neural Information Processing Systems 36, 36449.

[307] Sch¨utt, K. T., F. Arbabzadah, S. Chmiela, K. R. M¨uller, and A. Tkatchenko (2017), Nature Communications 8, 13890.

[308] Sch¨utt, K. T., M. Gastegger, A. Tkatchenko, K.-R. M¨uller, and R. J. Maurer (2019), Nat Commun 10 (1), 5024.

[309] Sch¨utt, K. T., H. E. Sauceda, P.-J. Kindermans, A. Tkatchenko, and K.-R. M¨uller (2018), The Journal of Chemical Physics 148 (24), 241722.

[310] Shapeev, A. V. (2016), Multiscale Modeling & Simulation 14 (3), 1153.

[311] Shell, M. S. (2008), The Journal of Chemical Physics 129 (14), 144108.

[312] Shiota, T., K. Ishihara, T. M. Do, T. Mori, and W. Mizukami (2024), “Taming multi-domain, -fidelity data: Towards foundation models for atomistic scale simulations,” arXiv:2412.13088 [physics.chem-ph].

[313] Shmilovich, K., M. Stiefenhofer, N. E. Charron, and M. Hofmann (2022), The Journal of Physical Chemistry A 126 (48), 9124.

[314] Shoghi, N., A. Kolluru, J. R. Kitchin, Z. W. Ulissi, C. L. Zitnick, and B. M. Wood (2024), “From molecules to materials: Pre-training large generalizable models for atomic property prediction,” arXiv:2310.16802 [cs.LG].

[315] Smith, J. S., O. Isayev, and A. E. Roitberg (2017), Chemical Science 8 (4), 3192.

[316] Smith, J. S., R. Zubatyuk, B. Nebgen, N. Lubbers, K. Barros, A. E. Roitberg, O. Isayev, and S. Tretiak (2020), Scientific Data 7 (1), 134.

[317] Snyder, J. C., M. Rupp, K. Hansen, K.-R. M¨uller, and K. Burke (2012), Physical Review Letters 108 (25), 253002.

[318] Sosso, G. C., G. Miceli, S. Caravati, J. Behler, and M. Bernasconi (2012), Phys. Rev. B 85, 174103.

[319] Souza, P. C. T., R. Alessandri, J. Barnoud, S. Thallmair, I. Faustino, F. Gr¨unewald, I. Patmanidis, H. Abdizadeh, B. M. H. Bruininks, T. A. Wassenaar, P. C. Kroon, J. Melcr, V. Nieto, V. Corradi, H. M. Khan, J. Doma´nski, M. Javanainen, H. Martinez-Seara, N. Reuter, R. B. Best, I. Vattulainen, L. Monticelli, X. Periole, D. P. Tieleman, A. H. de Vries, and S. J. Marrink (2021), Nature Methods 18 (4), 382.

[320] Sugita, Y., and Y. Okamoto (1999), Chemical Physics Letters 314 (1–2), 141.

[321] Sun, G., and P. Sautet (2019), J. Chem. Theory Comput. 15 (10), 5614.

[322] Swinburne, T., and D. Perez (2025), Mach. Learn. Sci. Technol. 6 (1), 015008.

[323] Swinburne, T. D., C. Lapointe, and M.-C. Marinica (2026), Nat. Comm. 17, 248.

[324] Symons, B. C. B., and P. L. A. Popelier (2022), Journal of Chemical Theory and Computation 18 (9), 5577, https://pubs.acs.org/jctcce/articlepdf/18/9/5577/3868663/ct2c00311.pdf.

[325] Tadmor, E. B., R. S. Elliott, J. P. Sethna, R. E. Miller, and C. A. Becker (2011), Jom 63 (7), 17.

[326] Tan, C. W., M. L. Descoteaux, M. Kotak, G. de Miranda Nascimento, S. R. Kavanagh, L. Zichi, M. Wang, A. Saluja, Y. R. Hu, T. Smidt, A. Johansson, W. C. Witt, B. Kozinsky, and A. Musaelian (2025), “High-performance training and inference for deep equivariant interatomic potentials,” arXiv:2504.16068 [physics.comp-ph].

[327] Thiemann, F. L., T. Resch¨utzegger, M. Esposito, T. Taddese, J. D. Olarte-Plata, and F. Martelli (2026), Nature Machine Intelligence 8 (5), 764.

[328] Tillet, P., H. T. Kung, and D. Cox (2019), in Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, MAPL 2019 (Association for Computing Machinery, New York, NY, USA) pp. 10–19.

[329] Tokita, A. M., T. Devergne, A. M. Saitta, and J. Behler (2025), The Journal of Chemical Physics 162 (17), 174120.

[330] Torrie, G. M., and J. P. Valleau (1977), Journal of Computational Physics 23 (2), 187.

[331] Tosello Gardini, A., U. Raucci, and M. Parrinello (2025), Nature Communications 16, 2475.

[332] Tran, K., W. Neiswanger, J. Yoon, Q. Zhang, E. Xing, and Z. W. Ulissi (2020), Mach. Learn.: Sci. Technol. 1 (2), 025006.

[333] Tripathi, S., L. Bonati, S. Perego, and M. Parrinello (2024), ACS Catalysis 14 (7), 4944.

[334] Tuckerman, M., B. J. Berne, and G. J. Martyna (1992), The Journal of Chemical Physics 97 (3), 1990.

[335] Unke, O. T., S. Chmiela, M. Gastegger, K. T. Sch¨utt, H. E. Sauceda, and K.-R. M¨uller (2021), Nature communications 12 (1), 7273.

[336] Unke, O. T., S. Chmiela, H. E. Sauceda, M. Gastegger, I. Poltavsky, K. T. Schutt, A. Tkatchenko, and K.-R. M¨uller (2021), Chem. Rev. 121 (16), 10142.

[337] Unke, O. T., M. St¨ohr, S. Ganscha, T. Unterthiner, H. Maennel, S. Kashubin, D. Ahlin, M. Gastegger, L. Medrano Sandonas, J. T. Berryman, A. Tkatchenko, and K.-R. M¨uller (2024), Sci. Adv. 10 (14), eadn4397.

[338] Veit, M., D. M. Wilkins, Y. Yang, R. A. DiStasio, and M. Ceriotti (2020), J. Chem. Phys. 153 (2), 024113.

[339] Venturin, J., and C. Clementi (2026), arXiv:2606.14111 10.48550/arXiv.2606.14111.

[340] Viguera Diez, J., S. Romeo Atance, O. Engkvist, and S. Olsson (2024), Machine Learning: Science and Technology 5 (2), 025010.

[341] Villar, S., D. W. Hogg, K. Storey-Fisher, W. Yao, and B. Blum-Smith (2021), Advances in Neural Information Processing Systems 34, 28848.

[342] Wander, B., M. Shuaibi, J. R. Kitchin, Z. W. Ulissi, and C. L. Zitnick (2025), ACS Catalysis 15 (7), 5283.

[343] Wang, J., S. Chmiela, K.-R. M¨uller, F. No´e, and C. Clementi (2020), J. Chem. Phys. 152 (19), 10.1063/5.0007276.

[344] Wang, J., S. Olsson, C. Wehmeyer, A. P´erez, N. E. Charron, G. de Fabritiis, F. No´e, and C. Clementi (2019), ACS Central Science 5 (5), 755.

[345] Wang, W., and R. G´omez-Bombarelli (2019), npj Computational Materials 5, 125.

[346] Wang, W., M. Xu, C. Cai, B. K. Miller, T. Smidt, Y. Wang, J. Tang, and R. G´omez-Bombarelli (2022), “Generative coarse-graining of molecular conforma-

tions,” arXiv:2201.12176 [cs.LG].

[347] Wang, Y., G. Cs´anyi, and C. Ortner (2025), “Many-body coarse-grained molecular dynamics with the atomic cluster expansion,” arXiv:2502.04661 [physics.chem-ph].

[348] Wang, Z., S. Ye, H. Wang, J. He, Q. Huang, and S. Chang (2021), npj Computational Materials 7 (1), 11.

[349] Warford, T., F. L. Thiemann, and G. Cs´anyi (2026), Machine Learning: Science and Technology 7 (3), 035033.

[350] Webb, M. A., J.-Y. Delannoy, and J. J. de Pablo (2019), Journal of Chemical Theory and Computation 15 (2), 1199.

[351] Westermayr, J., and P. Marquetand (2020), Machine Learning: Science and Technology 1 (4), 043001.

[352] Westermayr, J., and R. J. Maurer (2021), Chem. Sci. 12 (32), 10755.

[353] Willatt, M. J., F. Musil, and M. Ceriotti (2019), Journal of Chemical Physics 150 (15), 154110.

[354] Wirnsberger, P., A. J. Ballard, G. Papamakarios, S. Abercrombie, S. Racani\`ere, A. Pritzel, D. Jimenez Rezende, and C. Blundell (2020), The Journal of Chemical Physics 153 (14), 10.1063/5.0018903.

[355] Wirnsberger, P., G. Papamakarios, B. Ibarz, S. Racani\`ere, A. J. Ballard, A. Pritzel, and C. Blundell (2022), Machine Learning: Science and Technology 3 (2), 025009.

[356] Witt, W. C., C. van der Oord, E. Gelˇzinyt˙e, T. J¨arvinen, A. Ross, J. P. Darby, C. H. Ho, W. J. Baldwin, M. Sachs, J. Kermode, N. Bernstein, G. Cs´anyi, and C. Ortner (2023), The Journal of Chemical Physics 159, 10.1063/5.0158783.

[357] Wood, B., M. Dzamba, X. Fu, M. Gao, M. Shuaibi, L. Barroso-Luque, K. Abdelmaqsoud, V. Gharakhanyan, J. Kitchin, D. Levine, K. Michel, A. Sriram, T. Cohen, A. Das, S. Sahoo, A. Rizvi, Z. Ulissi, and L. Zitnick (2025), in Advances in Neural Information Processing Systems, Vol. 38, edited by D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (Curran Associates, Inc.) pp. 129391–129427.

[358] Xu, Z., W. Xie, and P. Hu (2026), “Edge cluster expansion with radial rotary attention for interatomic potentials,” arXiv:2607.10664 [stat.ML].

[359] Xu, Z., W. Xie, and P. Hu (2026), “Spectral/spatial tensor atomic cluster expansion with universal embeddings in cartesian space,” arXiv:2509.14961 [stat.ML].

[360] Yan, K., M. Bohde, A. Kryvenko, Z. Xiang, K. Zhao, S. Zhu, S. Kolachina, D. Sarıt¨urk, J. Xie, R. Arroyave, X. Qian, X. Qian, and S. Ji (2025), “A materials foundation model via hybrid invariant-equivariant architectures,” arXiv:2503.05771 [cs.LG].

[361] Yang, H., C. Hu, Y. Zhou, X. Liu, Y. Shi, J. Li, G. Li, Z. Chen, S. Chen, C. Zeni, M. Horton, R. Pinsler, A. Fowler, D. Z¨ugner, T. Xie, J. Smith, L. Sun, Q. Wang, L. Kong, C. Liu, H. Hao, and Z. Lu (2024), “Mattersim: A deep learning atomistic model across elements, temperatures and pressures,” arXiv:2405.04967 [cond-mat.mtrl-sci].

[362] Yang, M., L. Bonati, D. Polino, and M. Parrinello (2022), Catalysis Today 387, 143.

[363] Yang, S., and R. G´omez-Bombarelli (2023), “Chemically transferable generative backmapping of coarse-

grained proteins,” arXiv:2303.01569 [cs.LG].

[364] Yang, W., C. Templeton, D. Rosenberger, A. Bittracher, F. N¨uske, F. No´e, and C. Clementi (2023), ACS Cent. Sci. 9 (2), 186.

[365] Yin, B., J. Wang, W. Du, P. Wang, P. Ying, H. Jia, Z. Zhang, Y. Du, C. P. Gomes, C. Duan, G. Henkelman, and H. Xiao (2025), “Alphanet: Scaling up local-frame-based atomistic interatomic potential,” arXiv:2501.07155 [cs.LG].

[366] Yuan, R., J. Zhang, A. Kryshtafovych, R. D. Schaefer, J. Zhou, Q. Cong, and N. V. Grishin (2026), Proteins: Structure, Function, and Bioinformatics 94 (1), 86, https://onlinelibrary.wiley.com/doi/pdf/10.1002/prot.7

[367] yzchen08, (2025), “eqnorm: Generalized machine learning potential,” https://github.com/yzchen08/eqnorm.

[368] Zaporozhets, I., F. Musil, V. Kapil, and C. Clementi (2024), J. Chem. Phys. 161 (13), 10.1063/5.0226764.

[369] Zarrouk, T., and M. A. Caro (2025), “Molecular augmented dynamics: Generating experimentally consistent atomistic structures by design,” arXiv:2508.17132 [cond-mat].

[370] Zarrouk, T., R. Ibragimova, A. P. Bart´ok, and M. A. Caro (2024), Journal of the American Chemical Society 146 (21), 14645.

[371] Zaverkin, V., M. Ferraz, F. Alesiani, and M. Niepert (2026), Journal of Chemical Theory and Computation 22 (4), 2074.

[372] Zhang, D., X. Liu, X. Zhang, C. Zhang, C. Cai, H. Bi, Y. Du, X. Qin, A. Peng, J. Huang, B. Li, Y. Shan, J. Zeng, Y. Zhang, S. Liu, Y. Li, J. Chang, X. Wang, S. Zhou, J. Liu, X. Luo, Z. Wang, W. Jiang, J. Wu, Y. Yang, J. Yang, M. Yang, F.-Q. Gong, L. Zhang, M. Shi, F.-Z. Dai, D. M. York, S. Liu, T. Zhu, Z. Zhong, J. Lv, J. Cheng, W. Jia, M. Chen, G. Ke, W. E, L. Zhang, and H. Wang (2024), npj Computational Materials 10 (1), 293.

[373] Zhang, D., A. Peng, C. Cai, W. Li, Y. Zhou, J. Zeng, M. Guo, C. Zhang, B. Li, H. Jiang, T. Zhu, W. Jia, L. Zhang, and H. Wang (2025), “A graph neural network for the era of large atomistic models,” .

[374] Zhang, L., B. Onat, G. Dusson, A. McSloy, G. Anand, R. J. Maurer, C. Ortner, and J. R. Kermode (2022), npj Comput Mater 8 (1), 158.

[375] Zhang, X., and G. K.-L. Chan (2022), The Journal of Chemical Physics 157 (20), 10.1063/5.0118200.

[376] Zhong, A., C. Lapointe, A. M. Goryaeva, K. Arakawa, M. Ath\`enes, and M.-C. Marinica (2025), PRX Energy 4 (1), 013008.

1.[377] Zhong, A., C. Lapointe, A. M. Goryaeva, J. Baima, M. Ath\`enes, and M.-C. Marinica (2023), Phys. Rev. Mater. 7 (2), 023802.

[378] Zhong, Y., H. Yu, M. Su, X. Gong, and H. Xiang (2023), npj Computational Materials 9 (1), 182.

[379] Zhou, Y., S. Hu, X. Zhang, H. Wang, G. Tan, and W. Jia (2026), “Matris: Toward reliable and eficient pretrained machine learning interatomic potentials,” arXiv:2603.02002 [cs.LG].

[380] Zhou, Y., W. Zhang, E. Ma, and V. L. Deringer (2023), Nat Electron 6 (10), 746.

[381] Zills, F., M. R. Sch¨afer, N. Segreto, J. K¨astner, C. Holm, and S. Tovey (2024), The Journal of Physical Chemistry B 128 (15), 3662.

[382] Zou, Y., A. H. Cheng, A. Aldossary, J. Bai, S. X. Leong, J. A. Campos-Gonzalez-Angulo, C. Choi, C. T. Ser, G. Tom, A. Wang, Z. Zhang, I. Yakavets, H. Hao, C. Crebolder, V. Bernales, and A. Aspuru-Guzik (2025), Matter 8 (7), 102263.

[383] Zubatiuk, T., B. Nebgen, N. Lubbers, J. S. Smith, R. Zubatyuk, G. Zhou, C. Koh, K. Barros, O. Isayev, and S. Tretiak (2021), The Journal of Chemical Physics 154 (24), 244108.

[384] Zwanzig, R. W. (1954), The Journal of Chemical Physics 22 (8), 1420.