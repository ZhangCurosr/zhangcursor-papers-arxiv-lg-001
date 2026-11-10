# AI-Driven Discovery of Entanglement Witness in Spin Hamiltonians

Jiwei Tian,<sup>1,</sup> <sup>∗</sup> Mouyang Cheng,<sup>2,</sup> <sup>3,</sup> <sup>†</sup> and Mingda Li<sup>2,</sup> <sup>4,</sup> <sup>‡</sup>

<sup>1</sup>School of Physics, Peking University, Beijing 100084, China

<sup>2</sup>Center for Computational Science and Engineering, MIT, Cambridge, MA 02139, USA

<sup>3</sup>Department of Materials Science and Engineering, MIT, Cambridge, MA 02139, USA

<sup>4</sup>Department of Nuclear Science and Engineering, MIT, Cambridge, MA 02139, USA

Entanglement is a crucial resource in quantum science and engineering, but detecting it in many-body states from experimentally accessible observables remains challenging. Here we introduce EntNet, an AI-driven framework to discover compact combinations of spin observables that accurately distinguish entangled from fully separable many-body mixed states in one-dimensional spin-1/2 chains governed by short-range Hamiltonians. We construct local-unitary-invariant representations of measurable Pauli correlators up to four-body order, sparsely select 16 candidate invariant feature channels, and learn their spatial structure using a Transformer architecture with rotary positional embeddings. Using only expectation values of these low-order observables, EntNet achieves > 98% recall on certified entangled states with near-zero false-positive rates on separable states. Remarkably, it exhibits transferability across system sizes and distinct Hamiltonian families, including frustrated and disordered spin models, with minimal loss of accuracy. Beyond detection, EntNet provides physical insight into entanglement detection by revealing a selective spatial organization of pair correlations, while its finitetemperature decision boundary tracks the excitation gap of Jordan–Wigner quasiparticles. These results establish a transferable and physically interpretable framework for data-driven discovery of entanglement witnesses from local correlation functions.

## I. INTRODUCTION

Quantum entanglement is one of the defining features of quantum mechanics in many-body systems. Beyond its fundamental significance, entanglement serves as a key resource in quantum communication, quantum computation, and quantum cryptography protocols [1–5]. It is also deeply connected to the structure and complexity of quantum many-body states. In particular, entanglement entropy and the entanglement spectrum for pure states characterize entanglement between subsystems, revealing information that is not readily captured by conventional local observables. Their characteristic behavior can provide signatures of quantum criticality, symmetry breaking, and topological order, making entanglement a powerful tool for characterizing strongly correlated quantum matter [6–8]. The scaling and structure of entanglement also strongly constrain the eficiency of tensor-network representations, such as matrix-product states (MPS) and projected entangled-pair states (PEPS), and help explain the success of numerical methods such as the density-matrix renormalization group (DMRG) [9–11]. Reliable characterization of entanglement is therefore important both for understanding quantum matter and for developing eficient methods to represent and manipulate many-body quantum states.

However, detecting entanglement in generic many-body mixed states is computationally challenging. In fact, even in a simplified case, such as the weak-membership problem for bipartite separability, the computational task is still NPhard [12]. A variety of computable entanglement criteria and witnesses have consequently been developed, including the positive-partial-transpose (PPT) criterion and negativity, realignment and cross-norm criteria (CCNR) [1, 13], spinsqueezing inequalities [14], covariance-matrix criteria [15], and the quantum-Fisher-information-based criteria [16, 17]. Recent advances have connected quantum Fisher information to dynamical susceptibilities and demonstrated entanglement witnessing using inelastic neutron scattering [17, 18]. Related protocols have also been proposed for resonant inelastic x-ray scattering (RIXS) to probe electronic and spin-orbital entanglement [19, 20]. A three-spin chirality operator has also been proposed to characterize topological phases of Kitaev quantum spin liquids [21]. Despite these advances, existing approaches often require density-matrix information or rely on predefined witness constructions and specific experimentally accessible observables, potentially overlooking broader entanglement signatures. This motivates the search for generalizable entanglement signatures that can be learned from diverse low-order local spin correlations without reconstructing the full quantum state, which is generally inaccessible in condensed matter systems.

Recent advances in artificial intelligience (AI) and machine learning (ML) provide a complementary, data-driven route to identifying entanglement from experimentally or numerically accessible information [22, 23]. ML models have been used, for example, to infer logarithmic negativity or entanglement entropy from a restricted set of observables [24, 25], or to classify entanglement within structured families such as few-body GHZ- and W-type states [26, 27]. Two key challenges nevertheless remain. First, extending such approaches to spatially extended many-body systems requires capturing correlations across diferent length scales while remaining transferable across system sizes and Hamiltonians. Second, practical entanglement detection calls for methods that operate directly on experimentally accessible observables without reconstructing the full density matrix or assuming a narrowly prescribed family of target states. Addressing these limitations in extended many-body systems demands a scalable and transferable ML framework capable of resolving entanglement across diverse states, system sizes, and Hamiltonians.

In this work, we develop EntNet, an ML framework for detecting entanglement in finite-temperature mixed states of extended one-dimensional spin-1/2 systems using only loworder spin correlation functions. A central consideration is that entanglement is invariant under local unitary (LU) transformations of individual spins, whereas the measured spin correlations generally depend on the choice of local basis. Learning this invariance directly from data is dificult, particularly when a large number of potentially redundant correlators are involved. To address these challenges, we construct LU-invariant representations of two-, three-, and four-body correlators using singular values and tensor contractions, and sparsely distill high-dimensional candidate invariant channels into only 16 features. Their spatial organization is subsequently learned by a Transformer architecture with rotary positional embeddings (RoPE) [28, 29]. Trained using 35,000 synthetically generated $N = 1 6$ spin-chain states, comprising certified entangled and fully separable examples, EntNet achieves > 98% detection accuracy on the evaluated benchmarks, while maintaining near-zero false-positive rates on separable states. Without retraining, the model transfers from $N ~ = ~ 1 6$ to � = 32 �� chains and generalizes across the tested ���, frustrated $J _ { 1 } \cdot$ $J _ { 2 } .$ , and random local spin Hamiltonians, with minimal loss of accuracy. Beyond the high detection accuracy, the distilled representation from EntNet provides physical insight into the entanglement structure encoded in low-order observables. For example, pair correlations at lattice-site separations $r = 1$ and $r = 4$ emerge as particularly important, in contrast to the negligible contributions from $r = 2$ and $r = 3$ pair features. At finite temperature, EntNet further resolves a narrow crossover between entangled and separable regimes: in the saturated XX chain, its lower decision boundary quantitatively tracks the thermal excitation scale set by the quasiparticle gap. These results establish EntNet as a scalable AI-driven framework for discovering compact, transferable, and physically interpretable entanglement witnesses from low-order observables in generic many-body mixed states.

## II. METHOD

EntNet detects many-body entanglement from low-order Pauli correlators without requiring full quantum-state tomography. We first construct labeled spin-chain states and extract LU-invariant features from local spin correlators. Sparse feature selection then reduces the candidate observables to a compact set of 16 channels, which form site-resolved inputs to a Transformer that captures their spatial organization. Below, we describe the construction of training labels and datasets, the invariant representation, and the architecture of EntNet, which are also illustrated in Fig. 1.

## A. Entanglement and computable certificates

For a system of � spins, a pure state is fully separable if it is a product state $\left| \psi \right. = \bigotimes _ { i = 1 } ^ { N } \left| \phi _ { i } \right.$ . A mixed state is fully separable if it can be written as a probabilistic mixture of product states,

$$
\rho _ { \mathrm { S e p } } = \sum _ { \mu } p _ { \mu } \bigotimes _ { i = 1 } ^ { N } \rho _ { i } ^ { ( \mu ) } , \qquad p _ { \mu } \geq 0 , \quad \sum _ { \mu } p _ { \mu } = 1 .\tag{1}
$$

States that do not admit such a decomposition are entangled [1]. Rather than attempting the generally dificult separability test for the full many-body state, we search for entanglement in small subsystems. For each subset � of $k = 2 , 3$ , or 4 sites, we construct its reduced density matrix $\rho _ { S } = \operatorname { T r } _ { \bar { S } } \rho$ , where $\rho$ describes the full �-spin system and $\mathrm { T r } _ { \bar { S } }$ traces out the remaining $N { - } k$ spins, leaving the quantum state of the selected subsystem �. Because every marginal of a fully separable state must itself be separable, finding an entangled marginal is suficient to certify entanglement of the full �-site state.

For each reduced density matrix $\rho _ { S } ,$ , we test every nontrivial bipartition $A | B$ of the selected sites using two complementary separability criteria: PPT and CCNR [1, 13]. For PPT, we partially transpose the density matrix with respect to subsystem �, denoted by $\rho ^ { T _ { A } }$ , whose matrix elements in a product basis satisfy $( \rho _ { S } ^ { T _ { A } } ) _ { i \alpha , j \beta } ~ = ~ ( \rho _ { S } ) _ { j \alpha , i \beta } ,$ For CCNR, we realign the density matrix across the same bipartition, denoted by $\mathcal { R } _ { A | B } ( { \rho } )$ , by rearranging its matrix elements as $[ \mathcal { R } _ { A | B } ( \rho _ { S } ) ] _ { i j , \alpha \beta } = ( \rho _ { S } ) _ { i \alpha , j \beta }$ , and compute the trace norm, i.e. the sum of its singular values. Here, �, � and $\alpha , \beta$ index basis states of subsystems � and �, respectively. For separable states, which are convex mixtures of product states, partial transposition preserves the positivity of each productstate contribution, while realignment yields a trace norm no greater than unity. Therefore, any state separable across $A | B$ must satisfy

$$
\lambda _ { \operatorname* { m i n } } \Bigl ( \rho _ { S } ^ { T _ { A } } \Bigr ) \geq 0 , \qquad \Bigl \| \mathcal { R } _ { A | B } ( \rho _ { S } ) \Bigr \| _ { 1 } \leq 1 .\tag{2}
$$

Here, $\lambda _ { \operatorname* { m i n } }$ is the smallest eigenvalue and $\lVert \cdot \rVert _ { 1 }$ is the trace norm. Thus, either a negative eigenvalue of $\rho _ { S } ^ { T _ { A } }$ or a realignment norm exceeding unity rigorously certifies entanglement across that bipartition [30–33].

We use these suficient criteria to assign entanglement labels [22, 27], testing subsystems in increasing size. A state is labeled $\mathrm { E } _ { 2 }$ if any two-site marginal violates PPT; otherwise, a PPT or CCNR violation in a three-site marginal assigns $\mathrm { E } _ { 3 }$ and, if no lower-order certificate is found, a violation in a foursite marginal assigns $\mathrm { E } _ { 4 }$ . These labels indicate the smallest subsystem size at which our chosen criteria certify entanglement. All $\mathrm { E } _ { 2 }$ $\operatorname { E } _ { 3 } .$ , and $\mathrm { E } _ { 4 }$ states could enter the dataset as positive (entangled) examples. Negative examples are separable states generated by construction from product states and their convex mixtures, as defined in Eq. (1) (Fig. 1b). States with no detected certificate are labeled “unknown” and excluded from EntNet training and testing to avoid ambiguous supervision [34]. Further details are given in Appendix A.

## B. Dataset preparation

The core training dataset consists of 35,000 synthetically generated states of $N = 1 6 ~ \mathrm { s p i n } { - 1 / 2 }$ systems, equally divided between separable and certified entangled states. To evaluate the model on physically motivated quantum states, we primarily consider the finite-temperature Gibbs states of the anisotropic XY spin chain with periodic boundary condition during evaluation, as shown in Fig. 1a,

$$
H _ { \mathrm { X Y } } = - \sum _ { i = 1 } ^ { N } { J \left[ \frac { 1 + \gamma } { 2 } X _ { i } X _ { i + 1 } + \frac { 1 - \gamma } { 2 } Y _ { i } Y _ { i + 1 } \right] } - h \sum _ { i = 1 } ^ { N } { Z _ { i } } ,\tag{3}
$$

where $i + N \equiv i .$ The corresponding two-, three-, and four-body spin correlators can be evaluated using the Jordan–Wigner solution [35–37]. The negative class of the dataset is generated inside the fully separable set, including product states and convex mixtures of independently oriented product states according to Eq. (1). Candidate entangled states are first generated from random MPS [10, 11], then verified from $\mathrm { E } _ { 2 } , \mathrm { E } _ { 3 }$ and $\mathrm { E } _ { 4 }$ criteria. We also test the transferability of EntNet to other 1D spin Hamiltonians, including open XXZ model, frustrated $J _ { 1 } \cdot$ $J _ { 2 }$ model and random local spin Hamiltonians. Further details on the target Hamiltonians and numerical protocols are shown in Appendix A.

Additionally, we note that unless otherwise stated, the default training set includes only $\mathrm { E } _ { 2 }$ states as entangled examples, while $\mathrm { E } _ { 3 }$ and $\mathrm { E } _ { 4 }$ states are deliberately held out for testing. This setup allows us to assess whether EntNet can generalize beyond the entanglement structures explicitly represented during training and recognize more complex entangled states from their low-order correlation signatures.

## C. LU-invariant representation

The measured expectation values for Pauli spin correlators constitute the raw physical input to EntNet. Because entanglement is invariant under local basis rotations [1, 5], the density matrices $\rho$ and $U \rho U ^ { \dagger }$ , with $U = \bigotimes _ { i } U _ { i }$ , should receive the same ML label even though their raw Pauli correlators generally difer. Rather than learning this symmetry through data augmentation, EntNet imposes LU invariance directly through its feature input.

For sites $i _ { 1 } < \cdots < i _ { q }$ , we define the �-body Pauli correlation tensor

$$
T _ { \alpha _ { 1 } \cdots \alpha _ { q } } ^ { ( q ) } = \mathrm { T r } \left[ \rho \sigma _ { i _ { 1 } } ^ { \alpha _ { 1 } } \cdot \cdot \cdot \sigma _ { i _ { q } } ^ { \alpha _ { q } } \right] ,\tag{4}
$$

where $\alpha _ { j } \in \{ x , y , z \}$ specifies the Pauli component at site $i _ { j } ,$ with the one- through four-body sectors denoted $T ^ { ( 1 ) } { - } T ^ { ( 4 ) }$ Under a local unitary transformation, each Pauli index transforms through an orthogonal rotation $R _ { i } \ \in \ S \mathrm { O } ( 3 )$ . Accordingly, after reshaping a correlation tensor across a bipartition $A | B ;$

$$
T _ { A | B } ^ { ( q ) } \mapsto R _ { A } T _ { A | B } ^ { ( q ) } R _ { B } ^ { \top } ,\tag{5}
$$

where $R _ { A }$ and $R _ { B }$ are Kronecker products of the corresponding local rotations. Because left- and right- multiplication by orthogonal matrices leaves singular values unchanged, the singular values of these tensor unfoldings provide exact LUinvariant features. EntNet uses these singular values, together with additional LU-invariant tensor norms and contractions, as its input representation, with further construction details given in Appendix A, B.

## D. Transformer architecture and feature distillation

As illustrated in Fig. 1c, EntNet organizes the LU-invariant correlation features into a sequence along the spin chain. At each lattice site $i ,$ the invariant channels are concatenated with geometry masks accounting for boundary efects. The resulting vector $\boldsymbol { x } ^ { i }$ is projected into a shared embedding $\pmb { h } ^ { i }$ , giving the token sequence

$$
[ \mathrm { C L S } ] , \thinspace h ^ { 1 } , \dots , \thinspace h ^ { N } .
$$

A four-layer Transformer [28] then exchanges information between correlation patterns at diferent lattice positions through self-attention. The final CLS representation at the first token summarizes the sequence and is mapped to predict whether the probed quantum state is entangled or separable.

To encode how these local correlation patterns are arranged in space, we incorporate rotary position embedding (RoPE) [29] into the attention mechanism. As is shown in Fig. 1c, for each two-dimensional query-key component,

$$
\begin{array} { r } { Q _ { i , \ell } ^ { \prime } = R ( i \omega _ { \ell } ) Q _ { i , \ell } , \qquad K _ { j , \ell } ^ { \prime } = R ( j \omega _ { \ell } ) K _ { j , \ell } , } \end{array}\tag{6}
$$

such that

$$
\begin{array} { r } { ( \boldsymbol { Q } _ { i , \ell } ^ { \prime } ) ^ { \top } \boldsymbol { K } _ { j , \ell } ^ { \prime } = \boldsymbol { Q } _ { i , \ell } ^ { \top } R [ ( j - i ) \omega _ { \ell } ] \boldsymbol { K } _ { j , \ell } . } \end{array}\tag{7}
$$

where $R ( \phi )$ is the two-dimensional rotation matrix

$$
R ( \phi ) = \left( \begin{array} { c c } { { \cos \phi } } & { { - \sin \phi } } \\ { { \sin \phi } } & { { \cos \phi } } \end{array} \right) ,
$$

and $\omega _ { \ell }$ is the rotation frequency associated with component ℓ. The attention between two tokens therefore explicitly depends on their relative separation $j - i .$ This is particularly useful for length extrapolation, where a relation learned between correlation patterns can be reused at the same separation in longer spin chains.

To identify which physical observables are most informative for entanglement detection, we place a learnable gate on each candidate LU-invariant channel, following the stochastic gating approach of Louizos et al. [38]. Each gate acts on an entire observable channel and is shared across lattice positions. The gates are optimized jointly with EntNet while being encouraged to suppress channels that contribute little to classification. Their learned activation strengths therefore provide a direct measure of the relative importance of the diferent Pauli-correlation features used by the model. We then retain the top $K \ = \ 1 6$ channels for subsequent analysis, yielding a lightweight distilled EntNet with ∼480,000 model parameters from an initial set of 117 candidate correlator channels, corresponding to an 86% reduction in feature dimensionality. Unlike generic learned latent features, these channels are explicitly defined LU-invariant functions of spin correlators, allowing the selected inputs to retain a physical interpretation. However, this reduction in feature dimensionality does not necessarily imply an equivalent reduction in the number of Pauli measurements, since constructing one invariant may require multiple correlators. Finally, after training, EntNet classifies states as entangled or separable from the sigmoid output using a fixed decision threshold of $p = 0 . 5$ Further architectural and evaluation details are provided in Appendix C.

## III. RESULTS

## A. Learned compact representation of spatial correlation structure

We first determine how many physical channels are required for reliable entanglement detection. Here, � denotes the number of LU-invariant feature channels shared across lattice sites. As shown in Fig. $2 { \bf b } , K = 1 6$ gives the best overall performance on the XY model, with all metrics approaching unity. With fewer channels, the distilled representation does not retain sufficient information for robust classification, most notably for the held-out $\mathrm { E } _ { > 2 }$ states used to test extrapolation beyond the $\mathrm { E } _ { 2 }$ entanglement class. Increasing � beyond 16 provides no systematic improvement and eventually degrades performance, including a sharp loss of true-negative rate for the largest feature sets. These results indicate substantial redundancy among the candidate physical observables and motivate the use of the distilled 16-channel representation throughout the remainder of this work.

Having fixed the representation channels at $K \ = \ 1 6 .$ , we next examine which correlation features are retained by the distilled model. The 16-channel representation contains five two-site features, including three of them at separation $r = 1$ and two at $r = 4 .$ , together with contiguous three-site $( T ^ { ( 3 ) } )$ and four-site $( T ^ { ( 4 ) } )$ features (Fig. 2a). Surprisingly, no explicit $r = 2 \ \mathrm { o r } \ r = 3$ pair channel is retained. Here, $r = | i - j |$ denotes the separation between two lattice sites; the absence of a selected channel at a given � does not imply that the corresponding spin correlations vanish. As shown in Fig. 2b, the first $r = 3$ and $r = 2$ pair features enter only beyond the $K = 1 6$ cutof. Notably, this contrast in the contributions from diferent separations � is obtained from training on a 16-site spin Hamiltonian, and whether it persists in the thermodynamic limit remains to be established. The appearance of the $r = 2$ channel further coincides with a pronounced degradation in the true-negative rate, suggesting that these intermediate-range pair features provide little additional discriminative information. These results indicate that entanglement detection does not simply favor correlations according to spatial proximity. The $r = 1$ pair features naturally capture the dominant nearestneighbor correlations generated by local interactions, whereas the survival of $r \ = \ 4$ features suggests that more extended two-site correlations provide complementary information not already contained in the local pair sector. The absence of explicit $r \ = \ 2$ and $r \ = \ 3$ pair channels reflects the learned compact representation, rather than the physical irrelevance of intermediate-range correlations. Indeed, adding training examples of entangled states dominated by second-neighbor $( r = 2 )$ correlations improves the $J _ { 1 } { - } J _ { 2 } \operatorname { E } _ { 2 }$ recall from 0.865 to 1.000 without changing the selected 16 channels, suggesting that the retained features encode relevant spatial information that appropriate training examples help the classifier exploit (Section III.B, Appendix D). Feature-ablation tests further confirm the complementary roles of these channels: higherorder features are essential for detecting higher-order entanglement, while additional four-body features cannot fully replace the selected longer-range pair correlations (Appendix D).

## B. Generalizability of EntNet

With the optimized 16-channel LU-invariant representation, EntNet reaches essentially perfect performance on the $N = 1 6$ XY benchmark, reaching 100% AUROC (Fig. 2d). Moreover, EntNet achieves a true-negative rate (TNR) of 0.997 on an independent benchmark of 2,000 fully separable states with nontrivial classical correlations (Fig. 2g), demonstrating its ability to distinguish entanglement from classical correlations. A more stringent test is whether the same model generalizes beyond the conditions seen during training. We therefore test extrapolation to diferent Hamiltonian families, higher-order certificates, and larger system sizes. Unless otherwise stated, all results below directly use the identical EntNet model trained only on $N = 1 6$ separable states and $\mathrm { E } _ { 2 }$ states.

We first examine transfer across diferent Hamiltonian families. On the XXZ, frustrated $J _ { 1 } { - } J _ { 2 }$ , and random-local-Hamiltonian benchmarks, EntNet achieves near-perfect AU-ROC and accurately identifies higher-order certified $\mathrm { E } _ { 3 }$ and $\mathrm { E } _ { 4 }$ states $( \mathrm { F i g } . 2 \mathrm { c } )$ . The main performance degradation occurs in the $J _ { 1 } { - } J _ { 2 } \operatorname { E } _ { 2 }$ sector, where recall decreases to 0.86 while AU-ROC remains 1.0000. This indicates a limitation associated with the default decision threshold of $p = 0 . 5$ , rather than a failure to correctly rank entangled states relative to separable controls. Further analysis indicates that this deficit arises from insuficient coverage of relevant spatial correlation patterns in the training distribution, rather than missing information in the selected features. To verify this, we augment the training set with independently generated states exhibiting strong entanglement between sites separated by two lattice spacings $( r \ : = \ : 2 )$ Without changing the selected 16 channels or including any $J _ { 1 } { - } J _ { 2 }$ states in training, the $\mathrm { E } _ { 2 }$ recall on the $J _ { 1 } { - } J _ { 2 }$ benchmark is restored to 1.0000 (Appendix D).

We further evaluate EntNet on two independent $N = 3 2$ benchmarks, comprising finite-temperature XY-chain states and synthetically constructed states with diverse entanglement structures. Both benchmarks include states whose entanglement can only be certified from three- or four-site reduced density matrices. Without retraining or feature reselection, EntNet maintains near-perfect ranking and high recall across these diferent entanglement classes [Fig. 2(e,f)]. These results indicate that the learned representation remains efective for larger systems and extends beyond the specific quantum states and entanglement certificates used in training. This size transfer is consistent with the use of RoPE, which encodes relative spatial positions without restricting the model to the training sequence length. Self-attention allows the model to combine correlation information from sites at diferent separations, while RoPE makes their relative displacement explicit in the attention weights. The same positional relationships can therefore be transfered to longer chains without introducing new position-specific parameters.

![](images/38db129b7a2c5b23016227a7312e90108a8f538db88f848b138f012da128b697.jpg)  
FIG. 1. Overview of the correlator-based entanglement detection framework. a. Schematic illustration of an XY spin chain under transverse field, where the low-order Pauli correlators are measured and their expectation values are used to construct local-unitary (LU) invariant features. b. Product states and their mixtures provide constructively separable samples, while locally noised random matrix-product states provide candidate entangled states. A hierarchy of two- to four-site PPT/CCNR tests assigns entanglement certificates (E2–E4) or leaves states uncertified. c. The selected K LU-invariant channels and geometry masks form site tokens, which are tokenized and then processed by a four-block Transformer with rotary positional encoding (RoPE). The final CLS representation is used for classification.

Together, these results demonstrate that EntNet learns transferable correlation signatures associated with many-body entanglement, rather than merely fitting the training distribution, with the learned representation generalizing across Hamiltonian families, system sizes, and entanglement certificate levels.

## C. Role of LU invariance in the input representation

We next examine the role of enforcing LU symmetry directly in the input representation of EntNet. We compare three treatments of the same physical observables: raw Pauli correlators with no LU constraint, raw correlators trained with a soft consistency loss under random local basis rotations, and the exact LU-invariant representation used by the 16-channel EntNet. The training data and Transformer architecture are otherwise kept as similar as possible, with implementation details provided in Appendix C.

The diference becomes pronounced once we deploy the trained EntNet variants and apply them under transferability tests in Fig. 2d-g. The model based on raw correlators generalizes poorly and can exhibit substantial false-positive rates, while the soft LU constraint improves performance but remains less stable across benchmarks. By contrast, the exact LU-invariant representation used in this work preserves

Selected 16 LU-invariant channels

![](images/2901fa59ec3d741f0156be5610811869c3b57b9fa196e622def38d234cc52874.jpg)

c Transfer to unseen Hamiltonians  
![](images/24104ead4668ee4080136315c195b962302a5e029faadf4d85ddf0ada313644e.jpg)

Pair-channel ranking and performance vs. K  
![](images/89338e5a145b6f6029eb7749029f90d03da5874de2e34f18285b9dd46fb1fc2c.jpg)

![](images/0144774c2f4c2953129c6999b788d68a450163556dadc4014d16761f2d80596e.jpg)

![](images/d48f601f30100514119c3577a8556edb937ccd9ffd17c4c50427fa93b6785089.jpg)

![](images/3c40486320fa4a4a9623700f1d5faee4d69b9d4ba03f9c86048ebc6e8c6833cb.jpg)

![](images/e163e97887915983ad2413477113133e3d446621d14d263ba8e0985a63c1313a.jpg)

![](images/504efe00dc1955537ce99f3d18ee22d7103fe778da04f88022ab4b56d8a805c1.jpg)  
FIG. 2. Sparse LU-invariant feature selection and generalization of EntNet. a. The 16 selected local-unitary (LU) invariant channels, ordered by their consensus ranking across five independent feature selectors. The consensus score $S = 1 1 8 - { \bar { r } } , \quad$ where ¯� is the mean rank among 117 candidate channels, quantifies the preference for each feature. b. Distribution of two-site channels in the consensus ranking (upper) and classification performance as a function of the number of selected channels � (lower). Stars mark the first occurrence of each pair separation, and the vertical dashed line indicates $K = 1 6 .$ . Classifiers in the lower panel are independently trained and evaluated on $N = 1 6 \mathrm { X Y }$ -chain states. $\mathbf { c } _ { \bullet }$ Generalization to XXZ, frustrated $J _ { 1 } { - } J _ { 2 }$ , and random-local-Hamiltonian systems, none of which are included in training. The horizontal dashed line indicates unity. d–f. Comparison of models using raw correlators, softly constrained LU representations, and exactly LU-invariant features on d. $N = 1 6 \mathrm { X Y }$ -chain states, e. $N = 3 2$ XY-chain states, and f. independently constructed $N = 3 2$ states with diverse entanglement structures. g. True-negative rates on the separable subsets of these benchmarks and an additional independent set of 2,000 constructively separable states. All classifiers are trained only on synthetically generated � = 16 separable and $\mathrm { E } _ { 2 }$ states. Here $E _ { > 2 }$ includes $\mathrm { E } _ { 3 }$ and $\mathrm { E } _ { 4 }$ states. Recall and true-negative rate (TNR) are evaluated at the fixed threshold $p = 0 . 5$

both strong recall of certified entangled states and high truenegative rates for constructively separable states across system size and certificate order. These results highlight the efectiveness of incorporating known physical symmetries into the representation, providing a robust foundation for entanglement detection.

## D. Finite-temperature entanglement and learned physical boundaries

We finally test whether the learned classifier behaves consistently along continuous finite-temperature trajectories, where the underlying physics can be understood analytically. For a fixed $N ~ = ~ 1 6$ XY Hamiltonian with the thermal state $\rho ( T ) = e ^ { - H / T } / Z ,$ , writing $\beta = 1 / T$ and $d = 2 ^ { N }$ , at the hightemperature limit the thermal state can be expanded as

$$
\rho ( T ) = \frac { I } { d } - \frac { 1 } { d T } \left( H - \frac { \mathrm { T r } H } { d } I \right) + O ( T ^ { - 2 } ) .\tag{8}
$$

Therefore,

$$
\operatorname* { l i m } _ { T \to \infty } \rho ( T ) = \frac { I } { 2 ^ { N } } = \bigotimes _ { i = 1 } ^ { N } \frac { I _ { i } } { 2 } ,\tag{9}
$$

which is an exactly separable state. Thus, regardless of the lowtemperature behavior, every thermal trajectory must eventually approach the same separable infinite-temperature limit.

This trend is reproduced directly by the EntNet predictions on the XY Hamiltonian in Eq.(3). Along six representative thermal trajectories, the classifier output $p = \sigma ( z )$ approaches zero at high temperature and converges toward the common infinite-temperature limit (Fig. 3a). In particular, for $\gamma = 0$ and $h = 1 . 2$ , EntNet exhibits a reentrant classification pattern with increasing temperature, transitioning from separable to entangled and back to separable predictions. To resolve the underlying physical evolution of these same trajectories, we independently evaluate their PPT/CCNR entanglement certificates (Fig. 3b). The states evolve through $\mathrm { E } _ { 2 } , \mathrm { E } _ { 3 } / \mathrm { E } _ { 4 }$ , and eventually uncertified regimes as thermal mixing increases. This certificate-resolved analysis provides a direct physical interpretation of the continuous changes seen in the EntNet score, showing that its predictions track the evolving entanglement structure of the thermal states.

Moreover, Fig. 3c shows the classifier output as a heatmap over the full (ℎ, �) plane in the XX limit, $\gamma = 0$ , where the temperature-dependence of entanglement is non-monotonic. For $h > 1$ , the zero-temperature ground state is fully polarized and therefore separable, and EntNet correctly assigns this region $p < 0 . 5$ . Upon heating, however, thermally populated excited states introduce nontrivial correlations into the Gibbs state, causing the classifier to enter a model-positive region with $p > 0 . 5$ At even higher temperature, thermal mixing suppresses these correlations and drives the state back toward the maximally mixed separable limit. The resulting evolution represents a thermally induced reentrant classifier response and is consistent with known finite-temperature entanglement behavior in cyclic XX chains [39].

The lower decision boundary of EntNet admits a simple physical interpretation in terms of thermally activated Jordan– Wigner fermions. In the Jordan–Wigner representation [35, 36], the $\gamma = 0$ quasiparticle dispersion is (setting � = 1)

$$
\epsilon _ { k } = 2 ( h - \cos k ) \simeq 2 ( h - 1 ) + k ^ { 2 } , \qquad h > 1 , \quad k \simeq 0 .\tag{10}
$$

The excitation gap is therefore $\Delta = 2 ( h - 1 )$ . At low temperature, the density of thermally excited Jordan-Wigner fermions scales as

$$
n _ { \mathrm { t h } } \propto \int d k e ^ { - \epsilon _ { k } / T } \propto \sqrt { T } e ^ { - 2 ( h - 1 ) / T } .\tag{11}
$$

In the dilute-excitation regime, local spin correlators acquire corrections controlled by this thermally activated quasiparticle density. It is therefore natural to expect the lower classifier boundary to occur when these excitation-induced correlations reach an approximately fixed sensitivity scale $C _ { * }$ . This gives

$$
\sqrt { T _ { * } } \exp \left[ - \frac { 2 ( h - 1 ) } { T _ { * } } \right] = C _ { * } ,\tag{12}
$$

where $T _ { * } ( h )$ denotes the lower $p = 0 . 5$ decision boundary. As shown in $\mathrm { F i g . 3 d . }$ , this activated form closely follows the refined classifier boundary over $1 . 0 5 \leq h \leq 1 . 5 0$ . An unweighted fit in log �<sub>∗</sub> gives $C _ { * } ^ { ^ { - } } = 1 . 0 4 1 \times 1 0 ^ { - 3 } , R _ { \log _ { 1 0 } T _ { * } } ^ { 2 } = 0 . 9 9 9 3$ , and $\mathrm { R M S E } ( \log _ { 1 0 } T _ { * } ) = 0 . 0 0 6 8$ . The close agreement is consistent with the sensitivity of EntNet’s classifier to the excitationgap scale that controls thermally populated Jordan–Wigner fermions. EntNet therefore learns a decision boundary across the $( h , T )$ plane that quantitatively follows the quasiparticle activation law. Importantly, this field-dependent decision boundary of EntNet should not be interpreted as a rigorous separability boundary. In the XX limit $( \gamma = 0 )$ , for example, the exchange Hamiltonian commutes with the total � magnetization, $\begin{array} { r } { [ H _ { \mathrm { X X } } ( 0 ) , \sum _ { i } Z _ { i } ] = 0 } \end{array}$ . At any fixed $T > 0 ,$ Gibbs states at diferent finite fields are related by invertible local filtering, $\rho _ { h } ( T ) \propto F _ { h } \rho _ { 0 } ( T ) F _ { h }$ , where $F _ { h } = \bigotimes _ { i } \exp [ h Z _ { i } / ( 2 T ) ]$ Since invertible local transformation preserves full separability, the exact separability property at fixed positive temperature is independent of ℎ [40, 41]. The field-dependent lower contour therefore characterizes EntNet’s sensitivity to thermally activated correlation signatures rather than an exact onset of entanglement. The resulting prediction map connects three distinct regimes: an initially separable polarized ground state, an intermediate-temperature region where thermally excited correlations produce a positive classifier response, and the high-temperature separable limit. The reentrant structure and thermally activated scaling above show that EntNet captures finite-temperature signatures of many-body entanglement directly from low-order observables. These results suggest that the continuous prediction score of EntNet carries qualitative information about the evolving entanglement structure, beyond its role as a binary classification indicator.

## IV. DISCUSSION

We have developed EntNet to identify signatures of manybody entanglement from the collective spatial organization of low-order spin correlations. Rather than relying on a single or a few predefined observables and entanglement witnesses, as done in conventional scattering probes, EntNet combines information from diverse spin correlations across diferent orders and spatial separations, uncovering complementary signatures of many-body entanglement. With only 16 LU-invariant feature channels, EntNet detects states certified by two- to foursite entanglement criteria and generalizes across system sizes and Hamiltonian families with strong performance. At finite temperature, EntNet further exhibits a reentrant classification pattern, with its lower decision boundary quantitatively following the quasiparticle activation scale set by the excitation gap of Jordan-Wigner fermions. These results establish that a compact set of low-order observables can encode transferable, physically-informed signatures of many-body entanglement across correlation order, spatial scale, system size, and Hamiltonian family.

Several limitations define the next steps for EntNet. First, the present framework is developed for one-dimensional spin chains, and extending the representation to two-dimensional and three-dimensional lattices and more complex geometries will be important for broader many-body applications. Second, EntNet currently focuses on spin degrees of freedom; extending the same symmetry-aware strategy to fermionic systems would test whether a similar strategy can remain valid beyond spin Hamiltonians. Third, although the final representation is compact, each LU-invariant channel is constructed from multiple underlying Pauli correlators, making measurement eficiency an important consideration for actual experimental implementation. Programmable quantum simulators, such as superconducting-qubit arrays and trapped-ion systems, ofer a promising route, where two-, three-, and four-site Pauli correlators can be estimated through repeated state preparation, local basis rotations, and projective measurements [42, 43]. For quantum materials, scattering techniques ofer complementary possibilities: neutron scattering probes two-spin correlations [44], polarized neutron scattering can access threespin chiral correlations under suitable conditions [45], and resonant inelastic X-ray scattering (RIXS) is sensitive to selected higher-order correlations [46, 47]. However, connecting these scattering observables to multiple local equal-time correlators required by EntNet will require further development. Reducing the required measurement set while preserving classification accuracy will be essential for translating EntNet to

![](images/35a84528f6aaf0a71e19733de0addd21ddf9e27d05c54655e1ed839856b0d127.jpg)

![](images/a3a064cdfae512caa9a2da1a09df2dbce3389e1da0ca9ed36462137e3d2246d1.jpg)

![](images/3c031ccbfe0cdf97abbbb092c318ba26509a3f223a2700a1dcc5220a5b3b23f7.jpg)

![](images/64b3165c851d8d65b477789241b1e7ea90967f94c8fff7f940b0d198d1566528.jpg)  
FIG. 3. Thermal response and activated reentry in EntNet predictions. a. EntNet predictions $p = \sigma ( z )$ along six finite-temperature trajectories of the $N = 1 6 \mathrm { X Y }$ chain, using the original trained ensemble. The dashed line marks the classification threshold $p = 0 . 5 ,$ and diamonds indicate the common infinite-temperature limit. b. Entanglement certificates along the same trajectories, independently determined from PPT and CCNR tests on two- to four-site reduced density matrices. “Unknown” indicates the absence of an entanglement certificate, rather than separability, while the infinite-temperature state is exactly separable. c. EntNet predictions in the $( h , T )$ plane of the XX chain $( \gamma = 0 )$ , using an ensemble trained with additional states exhibiting strong distance-two entanglement. For $h / J > 1$ , thermal excitations drive the initially separable ground state into a region classified as entangled, before the predicted probability decreases again at higher temperatures. Black contours mark $p = 0 . 5$ , while gray bands indicate variations across independently trained models. The temperature axis contains a break separating the low- and high-temperature regions. d. Refined lower boundary $T ^ { * } ( h )$ of the model-positive region and its fit to the thermally activated form $\sqrt { T ^ { * } } \exp [ - 2 ( h - \bar { 1 } ) / T ^ { * } ] = C ^ { * }$ , with energies expressed in units of �. The fit uses 19 boundary points over $1 . 0 5 \leq h / J \leq 1 . 5 0 ,$ yielding $C ^ { * } = 1 . 0 4 1 \times 1 0 ^ { - 3 }$ and $R _ { \log _ { 1 0 } T ^ { * } } ^ { 2 } = 0 . 9 9 9 3$ . This boundary characterizes the learned classifier and should not be interpreted as a rigorous entanglement threshold or a thermodynamic phase transition.

these experimental settings.

Nevertheless, the EntNet framework opens a route toward scalable entanglement detection in many-body systems where full state tomography is impractical and only restricted sets of correlation measurements are experimentally accessible [7, 13, 48]. Conventional entanglement witnesses provide rigorous certification when an appropriate inequality is available, whereas EntNet ofers a complementary way to screen and characterize many-body states from a compact set of low-order correlation functions [7]. This is especially useful for scalable characterization of engineered entanglement. Potential applications include quantum simulators [49], superconductingqubit platforms [50], and strongly correlated quantum materials [51, 52], where many-body entanglement is routinely generated, manipulated, or inferred from restricted measurements. Such models could help map regions of detectable entanglement in phase diagrams, prioritize experimentally prepared states for stronger certification, and guide the selection of informative correlation measurements. More broadly, this approach suggests a paradigm shift from applying predefined entanglement witnesses toward discovering informative combinations of experimentally accessible correlations, potentially guiding the development of more eficient and transferable entanglement detection protocols.

[1] R. Horodecki, P. Horodecki, M. Horodecki, and K. Horodecki, Quantum entanglement, Rev. Mod. Phys. 81, 865 (2009).

[2] C. H. Bennett and D. P. DiVincenzo, Quantum information and computation, Nature 404, 247 (2000).

[3] W. D¨ur and H. J. Briegel, Entanglement purification and quantum error correction, Reports on Progress in Physics 70, 1381 (2007).

[4] N. Gisin, G. Ribordy, W. Tittel, and H. Zbinden, Quantum cryptography, Reviews of Modern Physics 74, 145 (2002).

[5] M. B. Plenio and S. Virmani, An introduction to entanglement measures, Quantum Information and Computation 7, 1 (2007).

[6] L. Amico, R. Fazio, A. Osterloh, and V. Vedral, Entanglement in many-body systems, Rev. Mod. Phys. 80, 517 (2008).

[7] N. Friis, G. Vitagliano, M. Malik, and M. Huber, Entanglement certification from theory to experiment, Nature Reviews Physics 1, 72 (2019).

[8] N. Laflorencie, Quantum entanglement in condensed matter systems, Physics Reports 646, 1 (2016).

[9] S. R. White, Density matrix formulation for quantum renormalization groups, Phys. Rev. Lett. 69, 2863 (1992).

[10] U. Schollwock, The density-matrix renormalization group in the¨ age of matrix product states, Ann. Phys. 326, 96 (2011).

[11] J. I. Cirac, D. Perez-Garc´ ´ıa, N. Schuch, and F. Verstraete, Matrix product states and projected entangled pair states: Concepts, symmetries, theorems, Rev. Mod. Phys. 93, 045003 (2021).

[12] L. Gurvits, Classical deterministic complexity of Edmonds problem and quantum entanglement, in Proceedings of the Thirty-Fifth Annual ACM Symposium on Theory of Computing (Association for Computing Machinery, 2003) pp. 10–19.

[13] O. G¨uhne and G. Toth, Entanglement detection, Phys. Rep.´ 474, 1 (2009).

[14] G. Toth, C. Knapp, O. G¨uhne, and H. J. Briegel, Optimal spin´ squeezing inequalities detect bound entanglement in spin models, Physical review letters 99, 250405 (2007).

[15] O. G¨uhne, P. Hyllus, O. Gittsovich, and J. Eisert, Covariance matrices and the separability problem, Physical review letters 99, 130504 (2007).

[16] P. Hyllus, W. Laskowski, R. Krischek, C. Schwemmer, W. Wieczorek, H. Weinfurter, L. Pezze, and A. Smerzi, Fisher informa-\` tion and multiparticle entanglement, Phys. Rev. A 85, 022321 (2012).

[17] P. Hauke, M. Heyl, L. Tagliacozzo, and P. Zoller, Measuring multipartite entanglement through dynamic susceptibilities, Nature Physics 12, 778 (2016).

[18] A. Scheie, P. Laurell, A. M. Samarakoon, B. Lake, S. E. Nagler, G. E. Granroth, S. Okamoto, G. Alvarez, and D. A. Tennant,

Witnessing entanglement in quantum magnets using neutron scattering, Physical Review B 103, 224434 (2021).

[19] T. Liu, L. Xu, J. Liu, and Y. Wang, Entanglement witness for indistinguishable electrons using solid-state spectroscopy, Physical Review X 15, 011056 (2025).

[20] Z. Shen, S. Ding, Z. Zhao, F. A. Evangelista, and Y. Wang, Witnessing spin-orbital entanglement using resonant inelastic x-ray scattering, Physical Review Letters 136, 116504 (2026).

[21] K. Hwang, A. Go, J. H. Seong, T. Shibauchi, and E.-G. Moon, Identification of a kitaev quantum spin liquid by magnetic field angle dependence, Nature communications 13, 323 (2022).

[22] S. Lu, S. Huang, K. Li, J. Li, J. Chen, D. Lu, Z. Ji, Y. Shen, D. Zhou, and B. Zeng, Separability-entanglement classifier via machine learning, Phys. Rev. A 98, 012315 (2018).

[23] Y. Chen, Y. Pan, G. Zhang, and S. Cheng, Detecting quantum entanglement with unsupervised learning, Quantum Sci. Technol. 7, 015005 (2022).

[24] J. Gray, L. Banchi, A. Bayat, and S. Bose, Machine-learningassisted many-body entanglement measurement, Phys. Rev. Lett. 121, 150503 (2018).

[25] R. Berkovits, Extracting many-particle entanglement entropy from observables using supervised machine learning, Phys. Rev. B 98, 241411(R) (2018).

[26] C. Chen, C. Ren, H. Lin, and H. Lu, Entanglement structure detection via machine learning, Quantum Sci. Technol. 6, 035017 (2021).

[27] J. Urena, A. Sojo, J. Bermejo-Vega, and D. Manzano, Entangle-˜ ment detection with classical deep neural networks, Sci. Rep. 14, 18109 (2024).

[28] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, Attention is all you need, in Advances in Neural Information Processing Systems, Vol. 30 (2017).

[29] J. Su, M. Ahmed, Y. Lu, S. Pan, B. Wen, and Y. Liu, RoFormer: Enhanced transformer with rotary position embedding, Neurocomputing 568, 127063 (2024).

[30] A. Peres, Separability criterion for density matrices, Phys. Rev. Lett. 77, 1413 (1996).

[31] M. Horodecki, P. Horodecki, and R. Horodecki, Separability of mixed states: Necessary and suficient conditions, Phys. Lett. A 223, 1 (1996).

[32] O. Rudolph, Further results on the cross norm criterion for separability, Quantum Inf. Process. 4, 219 (2005).

[33] K. Chen and L.-A. Wu, A matrix realignment method for recognizing entanglement, Quantum Inf. Comput. 3, 193 (2003).

[34] J. Paw lowski and M. Krawczyk, Identification of quantum en-

tanglement with siamese convolutional neural networks and semisupervised learning, Physical Review Applied 22, 014068 (2024).

[35] E. Lieb, T. Schultz, and D. Mattis, Two soluble models of an antiferromagnetic chain, Ann. Phys. 16, 407 (1961).

[36] E. Barouch, B. M. McCoy, and M. Dresden, Statistical mechanics of the XY model. i, Phys. Rev. A 2, 1075 (1970).

[37] E. Barouch and B. M. McCoy, Statistical mechanics of the XY model. ii. spin-correlation functions, Phys. Rev. A 3, 786 (1971).

[38] C. Louizos, M. Welling, and D. P. Kingma, Learning sparse neural networks through $L _ { 0 }$ regularization, in International Conference on Learning Representations (2018).

[39] N. Canosa and R. Rossignoli, Entanglement between distant qubits in cyclic XX chains, Phys. Rev. A 75, 032350 (2007).

[40] R. Rossignoli and N. Canosa, Global thermal entanglement in �-qubit systems, Physical Review A 72, 012335 (2005).

[41] N. Canosa and R. Rossignoli, Global entanglement in � � � chains, Physical Review A 73, 022347 (2006).

[42] A. H. Karamlou, I. T. Rosen, S. E. Muschinske, et al., Probing entanglement in a 2d hard-core bose–hubbard lattice, Nature 629, 561 (2024).

[43] T. Brydges, A. Elben, P. Jurcevic, B. Vermersch, C. Maier, B. P. Lanyon, P. Zoller, R. Blatt, and C. F. Roos, Probing Renyi´ entanglement entropy via randomized measurements, Science 364, 260 (2019).

[44] A. Scheie, P. Laurell, B. Lake, S. E. Nagler, M. B. Stone, J.-S. Caux, and D. A. Tennant, Quantum wake dynamics in heisenberg antiferromagnetic chains, Nature Communications 13, 5796 (2022).

[45] S. Grigoriev, S. Metelev, S. Maleyev, A. Okorokov, P. Boni,¨ R. Georgii, D. Lamago, H. Eckerlebe, and K. Pranzas, Critical two-and three-spin correlations in eus: An investigation with polarized neutrons, Physical Review B 72, 214423 (2005).

[46] J. van den Brink, The theory of indirect resonant inelastic x-ray scattering on magnons, EPL (Europhysics Letters) 80, 47003 (2007).

[47] L. Braicovich, L. Ament, V. Bisogni, F. Forte, C. Aruta, G. Balestrino, N. Brookes, G. De Luca, P. Medaglia, F. M. Granozio, et al., Dispersion of magnetic excitations in the cuprate la 2 cuo 4 and cacuo 2 compounds measured using resonant x-ray

scattering, Physical review letters 102, 167401 (2009).

[48] A. Elben, S. T. Flammia, H.-Y. Huang, R. Kueng, J. Preskill, B. Vermersch, and P. Zoller, The randomized measurement toolbox, Nature Reviews Physics 5, 9 (2023).

[49] R. Islam, R. Ma, P. M. Preiss, M. Eric Tai, A. Lukin, M. Rispoli, and M. Greiner, Measuring entanglement entropy in a quantum many-body system, Nature 528, 77 (2015).

[50] A. Gold, J. Paquette, A. Stockklauser, M. J. Reagor, M. S. Alam, A. Bestwick, N. Didier, A. Nersisyan, F. Oruc, A. Razavi, et al., Entanglement across separate silicon dies in a modular superconducting qubit device, npj Quantum Information 7, 142 (2021).

[51] A. Scheie, E. Ghioldi, J. Xing, J. Paddison, N. Sherman, M. Dupont, L. Sanjeewa, S. Lee, A. Woods, D. Abernathy, et al., Proximate spin liquid and fractionalization in the triangular antiferromagnet kybse2, Nature Physics 20, 74 (2024).

[52] J. Hales, U. Bajpai, T. Liu, D. R. Baykusheva, M. Li, M. Mitrano, and Y. Wang, Witnessing light-driven entanglement using time-resolved resonant inelastic x-ray scattering, Nature Communications 14, 3512 (2023).

[53] J. Kurmann, H. Thomas, and G. M¨uller, Antiferromagnetic longrange order in the anisotropic quantum spin chain, Physica A 112, 235 (1982).

[54] F. Verstraete, J. J. Garc´ıa-Ripoll, and J. I. Cirac, Matrix product density operators: Simulation of finite-temperature and dissipative systems, Phys. Rev. Lett. 93, 207204 (2004).

[55] A. E. Feiguin and S. R. White, Finite-temperature density matrix renormalization using an enlarged hilbert space, Phys. Rev. B 72, 220401(R) (2005).

[56] J. Haegeman, C. Lubich, I. Oseledets, B. Vandereycken, and F. Verstraete, Unifying time evolution and optimization with matrix product states, Phys. Rev. B 94, 165116 (2016).

[57] S. Sugiura and A. Shimizu, Canonical thermal pure quantum state, Phys. Rev. Lett. 111, 010401 (2013).

[58] J. Jaklic and P. Prelovˇ sek, Lanczos method for the calculation ofˇ finite-temperature quantities in correlated systems, Phys. Rev. B 49, 5065 (1994).

[59] I. Loshchilov and F. Hutter, Decoupled weight decay regularization, in International Conference on Learning Representations (2019).

## APPENDIX A: LABEL CONSTRUCTION AND BENCHMARK SCOPE

## 1. Certificate hierarchy

Every negative example is constructive. A separable density operator is generated as

$$
\rho _ { \mathrm { S e p } } = \sum _ { \mu = 1 } ^ { M } w _ { \mu } \bigotimes _ { i = 1 } ^ { N } \rho _ { i } ^ { ( \mu ) } , \qquad w _ { \mu } \geq 0 , \quad \sum _ { \mu } w _ { \mu } = 1 .\tag{13}
$$

Consequently, a negative label does not mean merely “not detected by a witness.” For a bipartition $A | { \bar { A } } .$ , a state is separable across that cut when it is a convex mixture of products $\rho _ { A } \otimes \rho _ { \bar { A } }$ . In a product basis $\{ \vert i \rangle _ { A } \vert \alpha \rangle _ { \bar { A } } \}$ , partial transposition on � is defined by

$$
\langle i \alpha | \rho ^ { T _ { A } } | j \beta \rangle = \langle j \alpha | \rho | i \beta \rangle .\tag{14}
$$

Separable states across the cut have $\rho ^ { T _ { A } } \succeq 0 [ 3 0 $ , 31]; a negative eigenvalue therefore certifies entanglement. The converse need not hold in generic multipartite dimensions. For the complementary computable cross-norm or realignment (CCNR) test, write $\rho _ { i \alpha , j \beta }$ for matrix elements of a bipartite state and rearrange them as

$$
\begin{array} { r } { [ \mathcal { R } ( \rho ) ] _ { i j , \alpha \beta } = \rho _ { i \alpha , j \beta } . } \end{array}\tag{15}
$$

Separable states obey $\begin{array} { r } { \| \mathcal { R } ( \rho ) \| _ { 1 } \leq 1 ; } \end{array}$ a larger trace norm is a suficient entanglement certificate [32, 33]. We test the inequivalent bipartitions of each reduced subsystem using fixed numerical tolerances. Positive labels are assigned by scanning these reduced states through a fixed hierarchy. In the datasets used here, $\mathrm { E } _ { 2 }$ denotes a state first certified by a negative partial transpose of a two-site marginal. $\mathrm { E } _ { 3 }$ and $\mathrm { E } _ { 4 }$ denote states for which no lower-order certificate is found but a three- or four-site marginal is certified by the partial-transpose or CCNR criterion. A state that is not certified by this hierarchy is marked “unknown” and excluded from all binary metrics. Thus the classifier is trained against certified positives and constructive negatives; it is never rewarded for treating an uncertified state as separable.

Finite-temperature states are mixed because of the Gibbs distribution, and the synthetic training families also contain purification- and mixture-induced mixedness. This physical mixedness is part of the task. By contrast, the input features reported here are exact expectation values. Finite sampling, readout errors, and imperfect state preparation have not yet been included, so the present results should not be described as experimentally noise robust.

## 2. Datasets and split roles

TABLE I. Principal datasets used for training, feature selection, and evaluation. Unknown labels are excluded from binary metrics and never treated as separable.
<table><tr><td>Dataset</td><td>Size</td><td>Composition</td><td>Role</td></tr><tr><td>Primary training</td><td>35,000</td><td> $1 7 , 5 0 0 \mathrm { S e p } + 1 7 , 5 0 0 \mathrm { E } _ { 2 }$ </td><td>classifier training</td></tr><tr><td>Feature-selection data</td><td>2,812</td><td>1,406 Sep/Ent pairs</td><td>channel ranking</td></tr><tr><td>Distance-two support</td><td>1,400</td><td>700 Sep/E2 pairs</td><td>training control</td></tr><tr><td> $N = 1 6 \mathrm { X Y }$ </td><td>912</td><td>432 Sep, 432 Ent, 48 unknown evaluation</td><td></td></tr><tr><td> $N = 3 2 \mathrm { X Y }$ </td><td>500</td><td>20 Sep, 432 Ent, 48 unknown</td><td>size transfer</td></tr><tr><td> $N = 3 2$  synthetic states</td><td>2,000</td><td>400 Sep, 1,600 Ent</td><td>transfer</td></tr><tr><td>Independent separable states 2,000</td><td></td><td>constructive mixtures</td><td>false-positive control</td></tr></table>

TABLE II. Composition of the independent $N = 1 6$ Gibbs-state benchmarks from diferent Hamiltonian families. Adaptive temperature refinement targets regions near the disappearance of the two-site entanglement certificate. No artificial noise is added; unknown states are excluded from performance metrics.
<table><tr><td>Family</td><td>Total  $\mathrm { S e p }$ </td><td> $\mathrm { E } _ { 2 }$   $\mathrm { E } _ { 3 }$   $\mathrm { E } _ { 4 }$ </td><td>Unknown</td></tr><tr><td>XXZ</td><td>343</td><td>79 103 52 30</td><td>79</td></tr><tr><td> $J _ { 1 } { - } J _ { 2 }$ </td><td>463</td><td>30 155 80 30</td><td>168</td></tr><tr><td>Random local H</td><td>626</td><td>39 238 87 40</td><td>222</td></tr></table>

The datasets serve distinct purposes. The channel ranking is obtained from the primary training distribution together with two auxiliary collections of matched separable and entangled states. After this ranking is frozen, the final 16-channel EntNet classifiers and the other top-� classifiers are initialized and trained independently using only the 35,000 primary training states. A separate control supplements this training distribution with states exhibiting strong distance-two correlations, without changing the selected channels. Parent states, matched pairs, and their local-unitary copies remain within a single data split. No evaluation benchmark is used for feature selection or classifier training.

The positive-label provenance is important for interpreting generalization across certificate levels. Both the primary training distribution and the additional distance-two training distribution contain only $\mathrm { E } _ { 2 }$ positive labels. Neither training route contains $\mathrm { E } _ { 3 }$ or $\mathrm { E } _ { 4 }$ labels, and no higher-order-certified state from any evaluation benchmark is used for training or model selection.

## 3. Training-set generation

Primary training distribution.— The full parent dataset contains 50,000 � = 16 states, split 70:15:15 within each label. Only its 35,000 training states enter the primary classifier-training route. Its 25,000 constructive negative examples comprise 5,000 product states, 17,500 convex mixtures of 2–32 product states, and 2,500 classical GHZ mixtures. More explicitly, every negative has a known decomposition

$$
\rho _ { \mathrm { S e p } } = \sum _ { \mu = 1 } ^ { K } w _ { \mu } \bigotimes _ { i = 1 } ^ { N } \frac { I + r _ { \mu i } \cdot \sigma } { 2 } .\tag{16}
$$

Local Bloch vectors have isotropic directions and a mixture of pure and Bloch-ball radii. The product family has $K = 1$ , whereas the general family mixes 2–32 product components with sampled weights. The classical GHZ mixture is not a coherent GHZ state; it is the explicitly separable state

$$
\rho _ { \mathrm { c G H Z } } = \frac { 1 } { 2 } \bigotimes _ { i } \rho ( { + } { \pmb n } _ { i } ) + \frac { 1 } { 2 } \bigotimes _ { i } \rho ( - { \pmb n } _ { i } ) , \qquad \rho ( { \pmb n } ) = \frac { I + { \pmb n } \cdot { \pmb \sigma } } { 2 } ,\tag{17}
$$

with independently sampled local axes. Finally, independent Haar-distributed single-qubit rotations are applied at all sites. These operations preserve the displayed convex product decomposition, so the negative label is known constructively rather than inferred from failure of an entanglement criterion.

The 25,000 positive parent states are random matrix product states (MPSs) with requested bond dimensions between 2 and $^ { 1 6 , }$ followed by local dephasing, depolarizing, or amplitude-damping noise. Noise strengths range from easily certified states to states near certificate boundaries, which are located separately for each parent and noise channel. Every accepted positive retains a two-site negative-partial-transpose (NPT) marginal at separation at most two. Thus all positive examples used for the primary classifier training are $\mathrm { E } _ { 2 }$ states.

Auxiliary feature-selection data.— Two independent collections of matched separable and entangled states are used only to rank candidate features. The first contains pairs constructed from product states and finite-depth entangling circuits, or from classical versus coherent GHZ and cluster blocks. The second pairs exactly separable ground states with entangled ground states generated from random-local Hamiltonians of the form in Eq. (26). Each collection contains 1,000 parent pairs, split by pair identity before feature selection. Their training partitions contain 706 and 700 pairs, respectively, giving the 2,812 states listed in Table I. Neither auxiliary collection enters the final compact-classifier training. The random-local thermal benchmark uses independent Hamiltonian realizations, although its Hamiltonian family is also represented in the auxiliary feature-selection data. Its transfer result is therefore zero-shot for the final classifier, but the entire feature-selection pipeline is not completely unfamiliar with that Hamiltonian family.

Distance-two training support.— To test the efect of missing spatial patterns in the training distribution, we construct an independent collection of matched states from the $N = 1 6$ open-boundary Interleaved-XXZ Hamiltonian in Eq. (22). This model is designed to exhibit strong correlations between sites separated by two lattice spacings. For each sampled Hamiltonian, two-site density-matrix renormalization group (DMRG) produces an $\mathrm { E } _ { 2 }$ ground state whose most negative two-site partial-transpose eigenvalue occurs at separation two. The matched negative is an exactly polarized product ground state, established using the bondwise saturation bound

$$
h _ { \mathrm { s a t } } = J _ { 2 } ( \Delta _ { 2 } + 1 ) + \epsilon J _ { 1 } ( \Delta _ { 1 } + 1 ) ,\tag{18}
$$

with $| h | = h _ { \mathrm { s a t } } + \delta h$ and $\delta h > 0 .$ . The full collection contains 1,000 $\mathrm { E } _ { 2 }$ states and 1,000 constructively separable states. Parent-wise 70:15:15 splitting yields 1,400 training, 300 validation, and 300 test states, without sharing parents across splits. The DMRG states satisfy an energy-variance-per-site check below $1 0 ^ { - 8 }$ . To characterize their spatial structure, we define

$$
R _ { 2 1 } = \frac { \langle s _ { 2 } [ T ^ { ( 2 ) } ( i , i + 2 ) ] \rangle _ { i } } { \langle s _ { 2 } [ T ^ { ( 2 ) } ( i , i + 1 ) ] \rangle _ { i } + 1 0 ^ { - 9 } } ,\tag{19}
$$

where the averages run over valid lattice anchors. Among the entangled states, 94.7% satisfy $R _ { 2 1 } > 3 \AA$ , confirming that their distance-two correlations dominate the corresponding nearest-neighbor correlations. The $J _ { 1 } { - } J _ { 2 }$ evaluation states and classifier predictions are not used to construct or filter this additional dataset. The separable partners supplement, rather than replace, the more diverse separable families in the primary training distribution.

## 4. Hamiltonians and XY parameter-space structure

The XY Hamiltonian is given in Eq. (3). The open-chain transfer benchmarks use $S _ { i } ^ { \alpha } = \sigma _ { i } ^ { \alpha } / 2$ and the Hamiltonians

$$
H _ { \mathrm { X X Z } } = J \sum _ { i = 1 } ^ { N - 1 } ( S _ { i } ^ { x } S _ { i + 1 } ^ { x } + S _ { i } ^ { y } S _ { i + 1 } ^ { y } + \Delta S _ { i } ^ { z } S _ { i + 1 } ^ { z } ) - h \sum _ { i = 1 } ^ { N } S _ { i } ^ { z } ,\tag{20}
$$

$$
H _ { J _ { 1 } J _ { 2 } } = J _ { 1 } \sum _ { i = 1 } ^ { N - 1 } { \cal S } _ { i } \cdot { \cal S } _ { i + 1 } + J _ { 2 } \sum _ { i = 1 } ^ { N - 2 } { \cal S } _ { i } \cdot { \cal S } _ { i + 2 } - h \sum _ { i = 1 } ^ { N } { \cal S } _ { i } ^ { z } .\tag{21}
$$

The additional distance-two training states are generated from a distinct Interleaved-XXZ family,

$$
\begin{array} { r l } & { H _ { \mathrm { I X X Z } } = J _ { 2 } \displaystyle \sum _ { i = 1 } ^ { N - 2 } \left( S _ { i } ^ { x } S _ { i + 2 } ^ { x } + S _ { i } ^ { y } S _ { i + 2 } ^ { y } + \Delta _ { 2 } S _ { i } ^ { z } S _ { i + 2 } ^ { z } \right) } \\ & { \quad \quad \quad \quad + \epsilon J _ { 1 } \displaystyle \sum _ { i = 1 } ^ { N - 1 } \left( S _ { i } ^ { x } S _ { i + 1 } ^ { x } + S _ { i } ^ { y } S _ { i + 1 } ^ { y } + \Delta _ { 1 } S _ { i } ^ { z } S _ { i + 1 } ^ { z } \right) - h \displaystyle \sum _ { i = 1 } ^ { N } S _ { i } ^ { z } . } \end{array}\tag{22}
$$

At small $\epsilon ,$ its distance-two bonds form two weakly coupled even and odd XXZ subchains. These equations specify the signs and operator normalizations; the use of Pauli operators in the $\mathrm { X Y }$ model versus spin operators in the open-chain models matters when comparing field scales. The XY benchmarks use periodic boundary conditions and the parameter region $0 \leq \gamma \leq 1 , 0 \leq h \leq 1 . 5$ . The single-particle dispersion is

$$
\varepsilon _ { k } = 2 \sqrt { ( h - \cos k ) ^ { 2 } + \gamma ^ { 2 } \sin ^ { 2 } k } ,\tag{23}
$$

which reveals the gap-closing lines and the following zero-temperature structure [35–37].

For $0 < \gamma \leq 1$ and $0 \leq h < 1$ , the thermodynamic ground state belongs to the $\mathbb { Z } _ { 2 }$ -ordered ferromagnetic phase. On a finite periodic chain, Hamiltonian eigenstates may be chosen with definite parity; away from exact degeneracy, such a state does not itself select either

broken-symmetry orientation. The thermodynamic Ising transition lies at $h = 1$ for $\gamma > 0$ . The line $\gamma = 0$ gives the number-conserving XX model, which is gapless for $0 \leq h < 1$ and fully polarized for $h > 1$

The circle

$$
\gamma ^ { 2 } + h ^ { 2 } = 1\tag{24}
$$

is not a phase boundary. Instead, it is the factorization locus within the ordered region [53], where the ground space admits two symmetrybreaking product states

$$
| \Phi _ { \pm } \rangle = \bigotimes _ { i = 1 } ^ { N } \left( \cos \phi | 0 \rangle _ { i } \pm \sin \phi | 1 \rangle _ { i } \right) , \qquad \phi = \frac { 1 } { 2 } \operatorname { a r c c o s } \left( \frac { h } { 1 + \gamma } \right) .\tag{25}
$$

Their local Bloch vectors are (± sin 2�, 0, cos 2�). Definite-parity ground states can instead be formed from even and odd superpositions of these product branches. This finite-size symmetry issue does not change the existence of the exact separable branches.

For $h > 1$ , the anisotropic XY chain is in the field-polarized paramagnetic phase. However, this region should not be identified entirely with an exactly saturated product eigenstate. At finite field and $\gamma > 0$ , the anisotropic exchange contains pair-flip terms, so the ground state remains weakly dressed and approaches full polarization only as $h  \infty$ . The exact saturated state $| 0 \rangle ^ { \otimes N }$ used as a benchmark anchor occurs on the XX line $\gamma = 0$ with $h > 1$ , where the magnetization-conserving exchange annihilates the all-up state. For $T > 0 ,$ the evaluated object is the Gibbs mixed state $\rho _ { T } = e ^ { - H _ { \mathrm { X Y } } / T } / \mathrm { T r } e ^ { - H _ { \mathrm { X Y } } / T }$ rather than an individual eigenstate. Phase labels therefore describe the zero-temperature spectrum underlying the sampled thermal crossover.

The open XXZ benchmark uses Eq. (20) with Latin-hypercube coverage $J \in \{ - 1 , + 1 \} , - 2 . 5 \leq \Delta \leq 3 . 0 ,$ , and $- 3 . 5 \leq h \leq 3 . 5$ , supplemented by structured anchors including $J = 0$ product controls. The open $J _ { 1 } { - } J _ { 2 }$ benchmark uses Eq. (21) with $J _ { 1 } \ = \ 1 , \ - 0 . 5 \le J _ { 2 } \ \le \ 1 . 5 .$ , and $- 4 \leq h \leq 4$ . Positive $J _ { 2 }$ frustrates the antiferromagnetic nearest-neighbor exchange. The empirical condition $J _ { 2 } / J _ { 1 } \ge 0 . 9$ isolates a dificult strongly frustrated subset for analysis; it is not presented as a phase boundary. In both families, large-field fully polarized states are included as constructively separable controls whenever a bondwise positive-semidefinite bound establishes that they are ground states.

The random-local Hamiltonian family uses an open chain of Pauli spins,

$$
H _ { \mathrm { R L } } = - \sum _ { i = 1 } ^ { N } f _ { i } \ : n _ { i } \cdot \sigma _ { i } + \sum _ { i = 1 } ^ { N - 1 } \sum _ { a , b = x , { \bf y } , z } \left[ - c _ { i } n _ { i } ^ { a } n _ { i + 1 } ^ { b } + \lambda A _ { i } ^ { a b } \right] \sigma _ { i } ^ { a } \sigma _ { i + 1 } ^ { b } .\tag{26}
$$

Here each $\pmb { n } _ { i }$ is a unit vector, while $A _ { i }$ is a real bond matrix orthogonal to $\pmb { n } _ { i } \pmb { n } _ { i + 1 } ^ { \top }$ and normalized to unit Frobenius norm. The sampled fields are suficiently strong that the commuting $\lambda = 0$ parent has a proved product ground state. Independent, weakly homogeneous, and smoothly varying site profiles broaden the local axes and couplings, while nonzero � introduces interactions that do not commute with th local fields. The exact profile distributions and sampling seeds are recorded in the generation code. The evaluation states are independent of the random-local ground-state pairs used in feature selection and do not enter final classifier training.

For the distance-two support family, we fix $J _ { 1 } = J _ { 2 } = 1$ and vary the anisotropies, field, and weak nearest-neighbor coupling. Its dominant distance-two bonds connect the even and odd sublattices into two XXZ backbones, whereas the �-scaled nearest-neighbor term weakly couples them. Entangled fields are sampled symmetrically from fixed fractions up to 0.78 of the saturation bound in Eq. (18); separable partners are placed beyond that bound. This family is deliberately distinct from the isotropic frustrated $J _ { 1 } { - } J _ { 2 }$ evaluation model and provides additional distance-two correlation patterns without reproducing the evaluated Hamiltonians.

## 5. Evaluation-set generation

XYbenchmarks.— The � = 16 XY benchmark consists of 480 finite-temperature Gibbs states sampled across the parameter region, including neighborhoods of the critical line, factorization circle, and low-temperature limit, together with 432 analytically constructed separable states. Of these separable states, 20 are product-state anchors covering the factorization circle and saturated XX line, and the remaining 412 are independently constructed factorized states equally divided between the two branches in Eq. (25). Exact free-fermion methods provide the correlators. All energies and temperatures are expressed in units of the XY exchange coeficient. Two-site marginals and contiguous three and four-site marginals are screened in hierarchical order. The final benchmark contains 432 separable states, 313 $\mathrm { E } _ { 2 }$ states, 119 higherorder-certified states, and 48 unknown states. The unknown states are retained for diagnostic purposes but excluded from binary evaluation metrics.

The $N = 3 2 \mathrm { X Y }$ benchmark uses the same 480 Hamiltonian parameter points and 20 analytic product anchors as the corresponding $N = 1 6$ sample, but recomputes all observables and entanglement certificates at the larger system size. It contains 20 separable states, 312 $\mathrm { E } _ { 2 }$ states, $8 0 \mathrm { E } _ { 3 }$ states, 40 $\mathrm { E } _ { 4 }$ states, and 48 unknown states. It therefore tests application of the frozen network to longer chains rather than reusing the $N = 1 6$ feature vectors.

Synthetic $N = 3 2$ benchmark and separable controls.— The synthetic benchmark contains 2,000 independently constructed states, including 400 constructively separable states and 1,600 entangled states distributed across the three certificate levels. No dense $2 ^ { 3 2 }$ density matrix is formed. Each embedded-block entangled state takes the form

$$
\rho = U _ { B } \rho _ { B } U _ { B } ^ { \dagger } \otimes \bigotimes _ { i \notin { B } } \rho _ { i } , \qquad | B | \in \{ 2 , 3 , 4 \} ,\tag{27}
$$

where $B$ is a randomly placed contiguous block, $U _ { B }$ is a tensor product of independently sampled single-qubit Haar rotations, and the remaining sites form a random product background. Strict negatives are generated as convex mixtures of 32-site product states. Thus the required moments

and reduced density matrices can be evaluated from the small embedded blocks and product backgrounds without constructing a full-chain density matrix.

Higher-order entangled blocks are generated from GHZ, graph, and shallow-circuit states, with noise applied before certificate screening. Three elementary noise maps are used:

$$
\begin{array} { r l r } {  { \mathcal { W } _ { p } ( \rho ) = ( 1 - p ) \rho + p I / 2 ^ { | B | } , } } \\ { \textstyle \mathcal { D } _ { p } ^ { \otimes | B | } ( \rho ) = \sum _ { a } K _ { a } \rho K _ { a } ^ { \dagger } , } & { \{ K _ { a } \} = \{ \sqrt { 1 - p } I , \sqrt { p } | 0 \rangle \langle 0 | , \sqrt { p } | 1 \rangle \langle 1 | \} , } \\ { \textstyle \mathcal { A } _ { p } ^ { \otimes | B | } ( \rho ) = \sum _ { a } A _ { a } \rho A _ { a } ^ { \dagger } , } & { \boldsymbol { A } _ { 0 } = ( 1 \sum _ { 0 } \sum _ { \sqrt { 1 - p } } ^ { 0 } ) , } & { \boldsymbol { A } _ { 1 } = ( 0 \begin{array} { c c } { 0 } & { \sqrt { p } } \\ { 0 } & { 0 } \end{array} ) . } \end{array}\tag{28}
$$

Here the same $p$ is applied independently to every qubit in the embedded block. This denotes a tensor product of local quantum channels acting on a possibly entangled density operator, not a factorization of the density operator itself. The $\mathrm { E } _ { 3 }$ sources include a white-noised three-qubit GHZ state, a weakly white-noised GHZ state mixed toward a product endpoint, and GHZ-derived states subjected to white noise, dephasing, or amplitude damping. The $\mathrm { E } _ { 4 }$ sources include white-noised four-qubit star-graph states, dephased and weakly white-noised four-qubit GHZ states, and shallow random circuits. The $\mathrm { E } _ { 2 }$ sources include embedded Bell pairs, depth-1–3 random two-qubit circuits, and entangled two-site Heisenberg Gibbs states. The resulting benchmark contains 400 states of each of the separable and $\mathrm { E } _ { 2 }$ classes, together with 600 $\mathrm { E } _ { 3 }$ and 600 $\mathrm { E } _ { 4 }$ states.

Every candidate is screened in the fixed order $\mathrm { E } _ { 2 }  \mathrm { E } _ { 3 }  \mathrm { E } _ { 4 }$ . A two-site NPT certificate takes priority. An $\mathrm { E } _ { 3 }$ candidate must have no certified pair but must contain a certified contiguous triple. An $\mathrm { E } _ { 4 }$ candidate must have no lower-order certificate but must contain a certified contiguous quadruple. Fixed positive safety margins exclude numerical boundary cases. Candidates that fail the certificate or margin check are not admitted to the benchmark. Separately, 2,000 additional constructively separable 32-site states are generated independently to assess false positives.

Cross-Hamiltonian benchmarks.— For the XXZ, $J _ { 1 } { - } J _ { 2 }$ , and random-local Hamiltonian families, Gibbs states are sampled over temperature and the sampling is refined near the disappearance of two-site entanglement certificates. This procedure captures states whose entanglement is detectable only in higher-order marginals. Their class compositions are reported in Table II. Each family contains at least ten independent parent Hamiltonians within each higher-order certificate class. No artificial noise channel is applied to these Gibbs states, and uncertified states remain unknown rather than becoming negative examples. The random-local thermal trajectories are independent of the random-local ground-state pairs used for feature selection. Hamiltonian parameters are sampled independently of classifier predictions; certificate transitions guide only temperature refinement. Evaluation states do not enter training or checkpoint selection. Classifier weights, normalization, and the decision threshold remain frozen throughout evaluation, with temperature grids and random seeds retained in the generation code.

## 6. Finite-temperature correlators and convergence checks

The XXZ Gibbs states are computed through matrix-product-state purification [54, 55]. At $\beta = 0 ,$ , each physical spin is maximally entangled with an ancilla,

$$
| I \rangle = \bigotimes _ { i = 1 } ^ { N } \frac { | 0 \rangle _ { p _ { i } } | 0 \rangle _ { a _ { i } } + | 1 \rangle _ { p _ { i } } | 1 \rangle _ { a _ { i } } } { \sqrt { 2 } } .\tag{29}
$$

Imaginary-time evolution acts only on the physical degrees of freedom,

$$
| \Psi _ { \beta } \rangle = e ^ { - \beta H / 2 } | I \rangle , \qquad \rho _ { \beta } = \frac { { \mathrm { T r } } _ { a } | \Psi _ { \beta } \rangle \langle \Psi _ { \beta } | } { \langle \Psi _ { \beta } | \Psi _ { \beta } \rangle } .\tag{30}
$$

Temperatures are traversed from high to low using two-site imaginary-time time-dependent variational principle (TDVP) evolution with open boundaries [56]. The required reduced density matrices are contracted directly from the purified MPS. Independent bond-dimension and time-step resolutions must agree on the certificate class, and the relevant reduced states and certificate margins must satisfy the convergence tolerances recorded in the implementation. Disagreement leaves a state uncertified.

The frustrated $J _ { 1 } { - } J _ { 2 }$ chain is evaluated using canonical thermal typicality [57], because purification becomes more expensive in the parameter regime considered. For a random phase vector |�⟩ and $\beta = 1 / T$ , Lanczos–Krylov propagation [58] approximates

$$
| \phi _ { r } ( \beta ) \rangle = e ^ { - \beta H / 2 } | r \rangle .\tag{31}
$$

For each required support �, the reduced density matrix is estimated directly as

$$
\widehat { \rho } _ { S } ( \beta ) = \frac { \sum _ { r = 1 } ^ { R } \mathrm { T r } _ { \bar { S } } | \phi _ { r } ( \beta ) \rangle \langle \phi _ { r } ( \beta ) | } { \sum _ { r = 1 } ^ { R } \langle \phi _ { r } ( \beta ) | \phi _ { r } ( \beta ) \rangle } .\tag{32}
$$

Independent random-vector estimates and their pooled estimate must assign the same certificate order. We additionally require the certificate margin to exceed five times its block-estimated sampling uncertainty; otherwise the state is marked unknown.

For the random-local family, zero-temperature states are obtained by two-site DMRG and checked using energy variance. The initial finite-temperature scan uses the reduced-density-matrix-first thermal-typicality procedure above, while additional thermal trajectories use two independently resolved MPS purifications. Certification requires the corresponding stochastic or two-resolution agreement checks. Failed

candidates remain unknown, not negative. The open boundaries and Pauli-operator normalization follow Eq. (26); no Hamiltonian coeficient or temperature is supplied to the classifier.

Finally, the Pauli tensors supplied to the invariant pipeline are evaluated from the corresponding reduced states,

$$
T _ { a _ { 1 } \cdots a _ { q } } ^ { ( q ) } ( i _ { 1 } , \ldots , i _ { q } ) = \mathrm { T r } \bigl [ \rho _ { i _ { 1 } \cdots i _ { q } } \sigma _ { a _ { 1 } } \otimes \cdots \otimes \sigma _ { a _ { q } } \bigr ] .\tag{33}
$$

Thus EntNet does not receive MPS tensors, Lanczos vectors, energies, or Hamiltonian parameters. It receives the same low-order local-unitaryinvariant correlator features used for all other datasets.

## APPENDIX B: INVARIANT FEATURE CONSTRUCTION

## 7. Why unfolding spectra are LU invariant

For a $q \cdot$ site Pauli correlation tensor, independent on-site unitary transformations induce orthogonal rotations $R _ { i } \in \mathrm { S O } ( 3 )$ . After grouping tensor indices into a bipartition $A | B ,$ the unfolded matrix transforms as

$$
T _ { A | B } ^ { \prime } = R _ { A } T _ { A | B } R _ { B } ^ { \mathsf { T } } , \qquad R _ { A } = \bigotimes _ { i \in A } R _ { i } , \quad R _ { B } = \bigotimes _ { j \in B } R _ { j } .\tag{34}
$$

Both $R _ { A }$ and $R _ { B }$ are orthogonal. Consequently, $( T _ { A | B } ^ { \prime } ) ^ { \intercal } T _ { A | B } ^ { \prime }$ is orthogonally similar to $T _ { A | B } ^ { \mathsf { T } } T _ { A | B } ,$ and its singular values are unchanged. Pair features use both the raw two-body tensor $T ^ { ( 2 ) }$ and the connected covariance

$$
C _ { a b } ^ { ( 2 ) } ( i , j ) = T _ { a b } ^ { ( 2 ) } ( i , j ) - m _ { i } ^ { a } m _ { j } ^ { b } , \qquad m _ { i } ^ { a } = \mathrm { T r } ( \rho \sigma _ { i } ^ { a } ) .\tag{35}
$$

The subtraction removes the disconnected product of one-site Bloch vectors. Both terms transform covariantly, so $C ^ { ( 2 ) } ( i , j ) \mapsto R _ { i } C ^ { ( 2 ) } ( i , j ) R _ { i } ^ { \intercal }$ and its singular values are also exact local-unitary (LU) invariants rather than approximate invariants learned through data augmentation.

The measurement support includes pairs at separations $r = 1 , \ldots , 4$ , but only contiguous three- and four-site windows that fit inside the chain. It does not wrap at the feature boundary, even when the XY Hamiltonian itself has periodic boundary conditions. The final 16-channel representation uses pair separations $r = 1$ and $r = 4 ,$ together with contiguous higher-order windows. Its 16 scalar invariants are computed from Pauli correlators; they are not 16 independently measured observables. The actual experimental measurement cost depends on how the underlying correlators are grouped and how measurement shots are allocated.

## 8. Candidate dictionary and selected channels

Let $s _ { k } \left[ X \right]$ denote the �th singular value of a matrix $X ,$ ordered nonincreasingly. The complete translation-tied candidate dictionary is defined as follows. The one-body block contains the Bloch-vector norm $\| m _ { i } \| _ { 2 }$ . For each pair distance $r = 1 , \ldots ,$ 4 and each $X \in \{ T ^ { ( 2 ) } , C ^ { ( 2 ) } \}$ the two-body block contains

$$
\left\{ s _ { 1 } \left[ \boldsymbol { X } \right] , s _ { 2 } \left[ \boldsymbol { X } \right] , s _ { 3 } \left[ \boldsymbol { X } \right] , \operatorname* { d e t } \boldsymbol { X } , \pmb { m } _ { i } ^ { \top } \boldsymbol { X } \pmb { m } _ { j } , \| \boldsymbol { X } \pmb { m } _ { j } \| _ { 2 } , \| \boldsymbol { X } ^ { \top } \pmb { m } _ { i } \| _ { 2 } \right\} .\tag{36}
$$

This yields 14 pair invariants per distance and 56 in total. For a contiguous triple, we use both the raw tensor $T ^ { ( 3 ) }$ and the connected third cumulant

$$
\begin{array} { c } { { K _ { a b c } ^ { ( 3 ) } ( i , j , k ) = T _ { a b c } ^ { ( 3 ) } ( i , j , k ) - m _ { i } ^ { a } T _ { b c } ^ { ( 2 ) } ( j , k ) - m _ { j } ^ { b } T _ { a c } ^ { ( 2 ) } ( i , k ) } } \\ { { { } } } \\ { { - m _ { k } ^ { c } T _ { a b } ^ { ( 2 ) } ( i , j ) + 2 m _ { i } ^ { a } m _ { j } ^ { b } m _ { k } ^ { c } . } } \end{array}\tag{37}
$$

For each tensor, the candidate dictionary contains its Frobenius norm and the three singular values of each of its three 1|2 unfoldings, yielding 20 three-body invariants. Finally, the raw contiguous four-body tensor contributes its Frobenius norm, the three singular values of each of four $1 | 3$ unfoldings, and the nine singular values of each of the 12|34, 13|24, and 14|23 unfoldings, giving 40 four-body invariants. No connected four-body tensor is included in this candidate pool. Together with the one-body norm, these quantities define 117 candidate physical channels. Geometry masks are deterministic inputs rather than selectable observables. Table III lists the selected 16 channels in consensus order.

The selected representation contains five pair, four three-body, and seven four-body channels. Its pair channels occur only at separations $r = 1$ and $r = 4 ,$ with no explicit pair channel at $r = 2 \mathrm { o r } r = 3 $ . Five deterministic geometry masks are supplied separately from the physical channels.

TABLE III. The 16 selected LU-invariant channels, ordered by consensus rank. For $T ^ { ( 3 ) }$ , mode � denotes the $a | ( b c )$ unfolding. Four-body partitions refer to the site indices within a contiguous block.
<table><tr><td>Rank</td><td>Order</td><td>Support</td><td>Tensor / unfolding</td><td>Invariant</td></tr><tr><td>1</td><td> $T ^ { ( 3 ) }$ </td><td>contiguous</td><td>raw, mode 0</td><td>S3</td></tr><tr><td>2</td><td> $T ^ { ( 2 ) }$ </td><td> $r = 1$ </td><td>raw</td><td>S2</td></tr><tr><td>3</td><td> $T ^ { ( 4 ) }$ </td><td>contiguous</td><td>raw, 14|23</td><td>S3</td></tr><tr><td>4</td><td> $T ^ { ( 3 ) }$ </td><td>contiguous</td><td>raw, mode 1</td><td>S3</td></tr><tr><td>5</td><td> $T ^ { ( 4 ) }$ </td><td>contiguous</td><td>raw, 13|24</td><td>S2</td></tr><tr><td>6</td><td> $T ^ { ( 4 ) }$ </td><td>contiguous</td><td>raw, 13|24</td><td> $s _ { 3 }$ </td></tr><tr><td>7</td><td> $C ^ { ( 2 ) }$ </td><td> $r = 1$ </td><td>connected</td><td> $\| ( C ^ { ( 2 ) } ) ^ { \top } m _ { i } \| _ { 2 }$ </td></tr><tr><td>8</td><td> $T ^ { ( 4 ) }$ </td><td>contiguous</td><td>raw, 2|134</td><td>S2</td></tr><tr><td>9</td><td> $T ^ { ( 3 ) }$ </td><td>contiguous</td><td>raw, mode 1</td><td>S2</td></tr><tr><td>10</td><td> $T ^ { ( 4 ) }$ </td><td>contiguous</td><td>raw, 14|23</td><td>S2</td></tr><tr><td>11</td><td> $T ^ { ( 2 ) }$ </td><td> $r = 4$ </td><td>raw</td><td>S2</td></tr><tr><td>12</td><td> $T ^ { ( 4 ) }$ </td><td>contiguous</td><td>raw, 2|134</td><td>S3</td></tr><tr><td>13</td><td> $T ^ { ( 4 ) }$ </td><td>contiguous</td><td>raw, 12|34</td><td>S2</td></tr><tr><td>14</td><td> $C ^ { ( 2 ) }$ </td><td> $r = 4$ </td><td>connected</td><td>S2</td></tr><tr><td>15</td><td> $T ^ { ( 2 ) }$ </td><td>r = 1</td><td>raw</td><td>S3</td></tr><tr><td>16</td><td> $T ^ { ( 3 ) }$ </td><td>contiguous</td><td>raw, mode 0</td><td>S2</td></tr></table>

## APPENDIX C: SELECTION, TRAINING, AND EVALUATION PROTOCOL

## 9. Sparse channel selection

Feature selection is label supervised but translation tied: each of the 117 physical channel types has one scalar gate shared across every lattice anchor. Geometry masks are never gated. For channel $^ { c , }$ the hard-concrete relaxation [38] is

$$
\begin{array} { l } { { s _ { c } = \mathrm { s i g m o i d } \left[ { \displaystyle \frac { \log u - \log ( 1 - u ) + \log \alpha _ { c } } { \beta _ { \mathrm { h c } } } } \right] , \qquad u \sim U ( 0 , 1 ) , } } \\ { { g _ { c } = \mathrm { c l i p } _ { [ 0 , 1 ] } \left[ s _ { c } ( \zeta - \gamma ) + \gamma \right] , } } \end{array}\tag{38}
$$

with $\beta _ { \mathrm { h c } } = 2 / 3 , \gamma = - 0 . 1$ , and $\zeta = 1 . 1$ . Its analytic nonzero probability is

$$
\begin{array} { r } { \pi _ { c } = \mathrm { s i g m o i d } [ \log \alpha _ { c } - \beta _ { \mathrm { h c } } \log ( - \gamma / \zeta ) ] , \qquad \widehat { K } = \displaystyle \sum _ { c = 1 } ^ { N _ { \mathrm { t o t a l } } } \pi _ { c } . } \end{array}\tag{39}
$$

During selector training, the normalized channel values are multiplied by stochastic gates $g _ { c }$ . The objective is

$$
\mathcal { L } _ { \mathrm { s e l } } = \mathcal { L } _ { \mathrm { B C E } } + \eta _ { e } \left[ 0 . 0 1 \frac { \widehat { K } } { N _ { \mathrm { t o t a l } } } + 2 0 \left( \frac { \widehat { K } - N _ { \mathrm { t a r g e t } } } { N _ { \mathrm { t o t a l } } } \right) ^ { 2 } \right] , \qquad \eta _ { e } = \mathrm { m i n } ( 1 , e / 2 5 ) ,\tag{40}
$$

where the 25-epoch ramp prevents the sparsity penalty from suppressing channels before the classifier has learned a useful signal. Here $N _ { \mathrm { t o t a l } } = 1 1 7$ and $N _ { \mathrm { t a r g e t } } = \bar { 3 2 }$ for the reported selector. The soft target $N _ { \mathrm { t a r g e t } }$ stabilizes a complete ordering; it is not the final input width. After warm-up, checkpoints are selected by minimizing validation binary cross entropy (BCE) plus $0 . 0 2 | \widehat { K } - N _ { \mathrm { t a r g e t } } | / N _ { \mathrm { t a r g e t } }$

Five independent selectors are trained on the feature-selection data described in Appendix A. Thus the frozen channel ordering is not obtained solely from the primary classifier-training distribution. Within each run, channels are ranked according to their learned log $\alpha _ { c }$ . The consensus order is determined first by mean rank across the five selectors, then by mean log $\alpha _ { c }$ , and finally by the fixed channel index. Truncating this ranking gives nested channel budgets. After the ranking is frozen, all selector gates, selector parameters, and classifier weights used during feature selection are discarded. Each compact classifier is initialized and trained independently on the selected physical channels. EntNet uses the first 16 channels in the consensus order, together with all geometry masks. No $N = 3 2$ evaluation example enters gate learning, ranking, normalization, checkpoint selection, or early stopping.

As a provenance diagnostic, repeating the five-selector procedure using only the primary training and validation states produces a top-16 list containing nine of the original 16 channels. This alternative ordering is not substituted into any reported classifier. All results use the same frozen channel identities and ordering.

## 10. EntNet Transformer architecture

At anchor �, the 16 selected invariant values are concatenated with five deterministic availability masks: four for pair separations $r = 1 , \ldots , 4$ and one for contiguous three-site support. With open feature boundaries, a contiguous four-site support is valid at exactly the same anchors as an $r = 3$ pair, so the corresponding pair mask also specifies four-site validity. Invalid physical entries are zero padded before token assembly. A separate binary sequence mask $V _ { b j } \in \{ 0 , 1 \}$ distinguishes real tokens from batch padding when chains of diferent lengths are processed together; the classification token is always valid. These masks contain only geometry information, not entanglement labels.

The resulting tensor $X ^ { ( 0 ) } \in \mathbb { R } ^ { B \times N \times \bar { 2 } 1 }$ is standardized channelwise using training-split statistics and passed through a shared $2 1  1 2 8$ afine projection. A learned classification (CLS) token is prepended, giving a Transformer input of shape $\left[ B , N + 1 , \hat { 1 2 8 } \right]$ . Each of the four Transformer layers contains four attention heads, each with width $d _ { h } = 3 2$ . Rotary positional encoding (RoPE) [29] acts on paired coordinates of queries and keys,

$$
\widetilde { q } _ { p , \ell } = R ( p \omega _ { \ell } ) q _ { p , \ell } , \qquad \widetilde { k } _ { p , \ell } = R ( p \omega _ { \ell } ) k _ { p , \ell } , \qquad \omega _ { \ell } = 1 0 , 0 0 0 ^ { - 2 \ell / d _ { h } } ,\tag{41}
$$

where

$$
R ( \theta ) = { \binom { \cos \theta } { \sin \theta } } \quad \cos \theta \quad .\tag{42}
$$

Values are not rotated. Orthogonality gives

$$
\begin{array} { r } { \widetilde { \pmb q } _ { i , \ell } ^ { \top } \widetilde { \pmb k } _ { j , \ell } = \pmb q _ { i , \ell } ^ { \top } R [ ( j - i ) \omega _ { \ell } ] \pmb { k } _ { j , \ell } , } \end{array}\tag{43}
$$

so the query–key phase depends on relative displacement without introducing a learned absolute-position vector into the physical features. For one attention head, the masked attention operation is

$$
\mathrm { A t t n } ( \cal Q , K , V ) = \mathrm { s o f t m a x } \left( { \frac { { \widetilde Q } { \widetilde K } ^ { \mathsf { T } } } { \sqrt { 3 2 } } } + M _ { \mathrm { a t t n } } \right) V ,\tag{44}
$$

where $( M _ { \mathrm { a t t n } } ) _ { b q k } = 0$ for valid keys and −∞ for padded keys in the mathematical idealization. The implementation uses the smallest finite attention-logit value in place of −∞ before softmax; the valid CLS key ensures that every softmax remains defined. Attention weights and block outputs at padded query positions are explicitly reset to zero. Each pre-normalized residual block is therefore

$$
\begin{array} { r l r } {  { Y ^ { ( \ell ) } = X ^ { ( \ell ) } + \mathbf { M H A } [ \mathbf { L N } ( X ^ { ( \ell ) } ) ] , } } \\ { \quad } & { { X ^ { ( \ell + 1 ) } = Y ^ { ( \ell ) } + \mathrm { F F N } [ \mathbf { L N } ( Y ^ { ( \ell ) } ) ] , } } \end{array}\tag{45}
$$

with a 128 → 192 → 128 GELU feed-forward network and dropout 0.1. After four blocks, only the CLS representation is read. A final LayerNorm and $1 2 8  1 2 8  1$ GELU classification head produce the logit � and probability $p = { \mathrm { s i g m o i d } } ( z )$ . Each compact model contains 483,969 trainable parameters. The architecture allows each anchor to compare its local invariant features with patterns elsewhere in the chain while keeping the physical input exactly LU invariant.

Each compact checkpoint minimizes unweighted BCE-with-logits using AdamW [59]. Three independently initialized classifiers use the same optimizer settings, validation-BCE checkpoint rule, and early-stopping procedure. Selector runs use separate seeds and a sparsity-penalty warm-up; their checkpoints are discarded after the consensus ranking is frozen. Learning rates, scheduler settings, and random seeds are recorded in the training configuration.

## 11. Raw-correlator LU controls

We compare EntNet with two Transformers trained on exactly the same 35,000 primary training states and the same parent-safe training, validation, and test partitions. Their tokens contain 147 Cartesian Pauli-correlation components: three one-body components, four sets of nine two-body components at separations $r = 1 , \ldots , 4 , 2 7$ contiguous three-body components, and 81 contiguous four-body components. Together with the same five deterministic geometry masks, this gives an input width of 152. All physical components required to construct the selected EntNet invariants are present in the raw representation. Unavailable boundary components are zero padded, and their masks are retained. Channel means and standard deviations are computed only from valid anchors in the primary training split, with masks left unstandardized.

Both controls use the same four-layer, four-head, pre-normalized RoPE Transformer architecture, hidden width 128, feed-forward width 192, CLS pooling, optimizer, early-stopping criterion, and three random seeds as EntNet. They also use the same two-layer $1 2 8 \to 1 2 8 \to 1$ classification head. Only the input projection width difers; the raw-input model has more trainable parameters because it receives a larger set of physical channels.

The first control is trained using canonical raw correlators and unweighted binary cross entropy. The second imposes a soft LU-consistency constraint: the same network processes a canonical input and a second input obtained by independently rotating the local basis at every site using Haar-random single-qubit unitaries. These rotations act on every index of the one- through four-body Pauli tensors, while geometry masks remain unchanged. Its loss is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { S i a m e s e } } = \frac { 1 } { 2 } \big [ \mathcal { L } _ { \mathrm { B C E } } ( z , y ) + \mathcal { L } _ { \mathrm { B C E } } ( z ^ { \prime } , y ) \big ] + \lambda _ { \mathrm { L U } } \big | \sigma ( z ) - \sigma ( z ^ { \prime } ) \big | , \qquad \lambda _ { \mathrm { L U } } = 0 . 5 , } \end{array}\tag{46}
$$

averaged over each minibatch. This encourages basis consistency but does not make the raw representation exactly invariant. Checkpoints for both controls are selected solely by validation BCE on canonical inputs. Evaluation benchmarks are used only after training, with the same three-seed probability averaging as EntNet. No test-dependent checkpoint selection is performed.

The raw-versus-soft comparison holds the input information and architecture fixed, testing the combined efect of rotation-augmented supervision and the soft consistency penalty; it does not separately isolate these two contributions. The comparison with EntNet instead tests an analytically invariant and compressed representation against the full raw tensor representation. Although the latter contains the information required to construct the selected invariants, the representation dimensionality and inductive bias also difer. Therefore, any observed generalization gap cannot be attributed exclusively to LU invariance. Per-seed metrics and individual predictions are retained in the experiment records; the soft-control recall on the synthetic $N = 3 2$ benchmark varies substantially across its three seeds.

TABLE IV. Comparison of input representations using correlations through fourth order. All models are trained on the same primary $N = 1 6$ training distribution. Predictions are averaged as probabilities across three independently initialized classifiers. Recall and false-positive rate (FPR) use the fixed threshold $p = 0 . 5$ . For the independent separable benchmark, only FPR is defined.
<table><tr><td>Model</td><td>Benchmark</td><td>AUROC</td><td> $\mathrm { E } _ { 2 }$  recall</td><td> $\mathrm { E } _ { > 2 }$  recall</td><td>FPR</td></tr><tr><td>Raw correlators,</td><td> $T _ { 1 - 4 } ~ N = 1 6 \mathrm { X Y }$ </td><td>0.6944</td><td>0.8914</td><td>1.0000</td><td>1.0000</td></tr><tr><td rowspan="5"></td><td> $N = 3 2 \mathrm { X Y }$ </td><td>0.3931</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>N = 32 synthetic</td><td>0.1212</td><td>0</td><td>0</td><td>0.0200 0.0600</td></tr><tr><td>Independent Sep</td><td></td><td></td><td></td><td></td></tr><tr><td> $N = 1 6 \mathrm { X Y }$ </td><td>0.5975</td><td>0.6326</td><td>0.8739</td><td>1.0000</td></tr><tr><td> $N = 3 2 \mathrm { X Y }$ </td><td>0.6072</td><td>0.7468 0.4075</td><td>0.9500 0.5033</td><td>1.0000 0.3325</td></tr><tr><td rowspan="5">EntNet</td><td>N = 32 synthetic Independent Sep</td><td>0.6856</td><td></td><td></td><td>0.2885</td></tr><tr><td>N = 16 XY</td><td></td><td></td><td></td><td></td></tr><tr><td> $N = 3 2 \mathrm { X Y }$ </td><td>1.0000 1.0000</td><td>1.0000 1.0000</td><td>1.0000 1.0000</td><td>0 0</td></tr><tr><td>N = 32 synthetic</td><td>0.9998</td><td>0.9875</td><td>0.9833</td><td>0</td></tr><tr><td>Independent Sep</td><td></td><td></td><td></td><td>0.0030</td></tr></table>

On the synthetic $N = 3 2$ benchmark, the soft constraint increases AUROC from 0.1212 to 0.6856 relative to the matched raw-correlator model, but also increases the false-positive rate from 0.0200 to 0.3325. Its false-positive rate on the independent separable benchmark rises from 0.0600 to 0.2885. The three-seed $\mathrm { E } _ { 2 }$ recall for the soft-control model has a mean of 0.4133 and a sample standard deviation of 0.3400, indicating substantial variation across initializations. EntNet achieves the strongest combined discrimination and separable-state rejection in this comparison. This advantage is consistent with the benefits of exact LU invariance, but cannot isolate them from the efects of feature compression and representation design.

## 12. Ensemble evaluation and benchmark metrics

For retained checkpoint $s ,$ let $z _ { s } ( \pmb { x } )$ denote the output logit and $p _ { s } ( { \pmb x } ) = \sigma [ z _ { s } ( { \pmb x } ) ]$ its predicted probability. The three-checkpoint ensemble averages probabilities and converts the result back to a logit for visualization:

$$
\bar { p } ( { \pmb x } ) = \frac { 1 } { 3 } \sum _ { s = 1 } ^ { 3 } p _ { s } ( { \pmb x } ) , \qquad z ( { \pmb x } ) = \log \frac { \bar { p } ( { \pmb x } ) } { 1 - \bar { p } ( { \pmb x } ) } .\tag{47}
$$

Let $\mathcal { E } = \mathrm { E } _ { 2 } \cup \mathrm { E } _ { 3 } \cup \mathrm { E } _ { 4 }$ denote the certified entangled set and $s$ the constructively separable set. At the fixed threshold $\bar { p } = 0 . 5 ,$ , class-specific recall and false-positive rate are defined by

$$
\begin{array} { c } { { \mathrm { R e c a l l } _ { E _ { q } } = \displaystyle \frac { 1 } { | E _ { q } | } \sum _ { x \in E _ { q } } \mathbf { 1 } [ z ( { \pmb x } ) \geq 0 ] , } } \\ { { \mathrm { F P R } = \displaystyle \frac { 1 } { | S | } \sum _ { x \in S } \mathbf { 1 } [ z ( { \pmb x } ) \geq 0 ] . } } \end{array}\tag{48}
$$

The area under the receiver operating characteristic curve (AUROC) is the threshold-independent probability that a randomly selected certified entangled state scores above a randomly selected separable state, assigning half weight to ties:

$$
\begin{array} { r l } { \displaystyle \mathrm { A U R O C } = \frac { 1 } { | \mathcal E | | S | } \sum _ { x \in \mathcal { E } } \sum _ { y \in S } \big [ \mathbf { 1 } \{ s ( x ) > s ( y ) \} } & { } \\ { \displaystyle + \frac 1 2 \mathbf { 1 } \{ s ( x ) = s ( y ) \} \big ] . } \end{array}\tag{49}
$$

For the ensemble results reported here, $s = \bar { p }$ is the arithmetic mean of three predicted probabilities, not an average of logits. Unknown states are excluded from these metrics but retained in diagnostic score plots. The decision threshold is fixed before evaluation.

To evaluate diferent feature budgets, we freeze the 117-channel consensus ranking and train a new classifier independently for each nested top-� prefix, with $K = 4 , 8 , \ldots , 3 2$ . Every budget uses the same primary training and validation partitions, three seeds, architecture, optimizer, and validation-BCE checkpoint rule; only the number of selected physical channels changes. All five geometry masks remain. Evaluation uses the $N = 1 6 \mathrm { X Y }$ benchmark containing 432 constructively separable states, $3 1 3 ~ \mathrm { E } _ { 2 }$ states, 119 higher-order-certified states, and 48 unknown states. Unknown states are excluded from all performance metrics. Table V reports the three-seed probability-averaged ensemble results at th fixed threshold $\bar { p } = 0 . 5$

The reference 16-channel model preceded completion of the full budget scan, but the $N = 1 6 \mathrm { X Y }$ results were considered when adopting $K = 1 6$ . Therefore, this scan is a retrospective comparison rather than an independent prospective test of an entirely prespecified channel count. It was not used to rerank candidate channels or select individual checkpoints, and the cross-Hamiltonian and $N = 3 2$ benchmarks remained outside the budget comparison. The input width $K + 5$ counts numerical model coordinates, not independent Pauli measurement settings, because multiple spectral invariants can be computed from the same correlation tensor.

TABLE V. Channel-budget comparison on the $N = 1 6 \mathrm { X Y }$ benchmark. Classifiers use nested prefixes of the same frozen channel ranking and are independently trained on the primary training distribution. Values are computed from three-seed probability-averaged predictions. FPR refers to 432 constructively separable controls.
<table><tr><td>K</td><td>Width</td><td>AUROC</td><td> $\mathrm { E } _ { 2 }$  recall</td><td> $\mathrm { E } _ { > 2 }$  recall</td><td>FPR</td></tr><tr><td>4</td><td>9</td><td>0.6667</td><td>0.6709</td><td>0.1513</td><td>0</td></tr><tr><td>8</td><td>13</td><td>0.6968</td><td>0.7764</td><td>0.1597</td><td>0</td></tr><tr><td>12</td><td>17</td><td>0.8981</td><td>0.7316</td><td>0.7731</td><td>0</td></tr><tr><td>16</td><td>21</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>0</td></tr><tr><td>20</td><td>25</td><td>0.9606</td><td>0.9201</td><td>0.9496</td><td>0</td></tr><tr><td>24</td><td>29</td><td>0.9954</td><td>0.9521</td><td>0.9328</td><td>0</td></tr><tr><td>28</td><td>33</td><td>0.7905</td><td>0.9042</td><td>0.8655</td><td>1.0000</td></tr><tr><td>32</td><td>37</td><td>0.7245</td><td>0.9105</td><td>0.8992</td><td>1.0000</td></tr></table>

For this benchmark, the independently trained $K = 1 6$ ensemble reproduces the four aggregate performance metrics of the established EntNet ensemble, although the checkpoints and individual predicted probabilities difer. Performance does not improve monotonically with �: the $K = 2 8$ and $K = 3 2$ models classify all 432 constructively separable controls as entangled at the fixed threshold, even though their AUROC remains above chance. Additional selected features therefore do not guarantee better generalization or decision-threshold behavior. These retrospective results support the present compact representation but do not establish a universally optimal channel count.

TABLE VI. Performance of the frozen EntNet ensemble on the $N = 1 6$ Gibbs-state benchmarks from diferent Hamiltonian families. Counts in parentheses denote ${ \mathrm { S e p } } / { \mathrm { E } _ { 2 } } / { \mathrm { E } _ { 3 } } / { \mathrm { E } _ { 4 } } .$ . Unknown states are excluded from all metrics. Recall and FPR are evaluated at the fixed threshold $p = 0 . 5 .$
<table><tr><td>Family  $( \mathrm { S e p } / \mathrm { E } _ { 2 } / \mathrm { E } _ { 3 } / \mathrm { E } _ { 4 } )$ </td><td>AUROC</td><td>FPR</td><td> $\mathrm { E } _ { 2 }$ </td><td> $\mathrm { E } _ { 3 }$ </td><td> $\mathrm { E } _ { 4 }$ </td></tr><tr><td>XXZ (79/103/52/30)</td><td>1.0000</td><td>0</td><td>0.9903</td><td>1.0000</td><td>1.0000</td></tr><tr><td> $J _ { 1 ^ { - } } J _ { 2 } ( 3 0 / 1 5 5 / 8 0 / 3 0 )$ </td><td>1.0000</td><td>0</td><td>0.8645</td><td>1.0000</td><td>1.0000</td></tr><tr><td>Random local H (39/238/87/40)</td><td>1.0000</td><td>0</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr></table>

The system-size transfer tests apply the same $N = 1 6 \mathrm { - t r a i n e d }$ network to $N = 3 2$ token sequences without updating its parameters. The variable-length architecture accepts the longer chains directly. Table IV reports the model’s performance on both the $N = 3 2 \mathrm { X Y }$ states and the independently constructed synthetic states. A separate field-of-view intervention isolating the contribution of sites 17–32 has not been verified for this checkpoint; consequently, the present results establish successful evaluation at larger � but do not quantify how strongly the network uses information from the additional sites.

## APPENDIX D: CONTROLLED INTERVENTIONS

## 13. Diagnosing missing spatial correlation patterns

The lower $\mathrm { E } _ { 2 }$ recall on the $J _ { 1 } { - } J _ { 2 }$ benchmark reflects a mismatch between the original training distribution and the spatial correlations encountered in this Hamiltonian family. In the regime $J _ { 1 } \to 0 ,$ , the even and odd sites form nearly independent chains, making correlations at separation $r = 2$ particularly important. Although EntNet contains no explicit $r = 2$ pair channel, its contiguous three- and four-body features probe correlations spanning this distance. The observed deficit could therefore arise either from information missing in the selected representation or from insuficient exposure to these patterns during training.

To distinguish these explanations, we augment the training distribution with independently constructed states exhibiting strong distance-two entanglement, while keeping the same 16 input channels. This restores positive predictions for all $1 5 5 \mathrm { E } _ { 2 }$ states in the $J _ { 1 } { - } J _ { 2 }$ benchmark, without introducing any $J _ { 1 } { - } J _ { 2 }$ evaluation state into training. The recovery demonstrates that the selected representation contains enough information to recognize these states and identifies insuficient training coverage, rather than the absence of a dedicated $r = 2$ pair observable, as the main source of the original deficit.

To examine how EntNet combines correlations of diferent orders, we replace either the three-/four-body feature block or the pair-feature block with its training-set mean, leaving the remaining inputs unchanged. As shown in Fig. $4 ( \mathrm { c - f } )$ , replacing higher-order features degrades predictions even for E<sub>2</sub> states, although their labels are defined by two-site NPT certificates. Conversely, replacing pair features also changes predictions for states whose entanglement is certified only at higher order. Thus the classifier does not simply assign distinct input blocks to the E<sub>2</sub>, $\mathrm { E } _ { 3 } ,$ and $\mathrm { E } _ { 4 }$ certificate levels.

![](images/e651bbe81155fd485d3fec3c5dffca8f4ee372818b327b7eef8a9d240b08ebe8.jpg)

![](images/14e2c4fc01e0bdbecfa9573a91343a3e926ecc8d4395b829e375ab295edd3dcf.jpg)  
Separable (n=30) E>2 (n=110) Full input E2 (n=155) Unknown (n=168) Block replaced

![](images/5db0b49248e8d2deaaf9ba974774764abf02356a5f7499647fc3391694c5cd35.jpg)

![](images/24222b86a9f18294a50c268fa91f22dfb3cc459b7ca7da54da191887ca950f88.jpg)

![](images/ec597b8c4f7418f5ff527ee34c16978ffd52e6e965416bef97e81c170bd94d22.jpg)

![](images/c65d51f2ab4fcf410dcba0333c3cdac3a62e507d7fe1af7be4c2aad0c0d4057c.jpg)  
FIG. 4. Complementary contributions of pair and higher-order correlations. (a) $\mathrm { E } _ { 4 }$ recall on the synthetic $N = 3 2$ benchmark using features through third order, after adding two selected four-body channels, and with the complete 16-channel representation. (b) Changes in exact-state AUROC and independent separable-state true-negative rate (TNR) after removing the selected $r = 4$ pair channels or replacing them with additional four-body invariants. (c)–(f) Classifier-score distributions for $N = 1 6 \ J _ { 1 ^ { - } } J _ { 2 }$ Gibbs states before and after replacing selected feature blocks with their training-set means. Three- and four-body channels are replaced in (c,e), while pair channels are replaced in (d,f). Panels (c,d) use the original training distribution; panels (e,f) use the same distribution supplemented with independently generated states exhibiting strong distance-two correlations. Within each state class, the original and modified inputs are shown on the left and right, respectively. Circles indicate medians, vertical segments show interquartile ranges, and the dashed line marks $p = 0 . 5$ . Because modified inputs need not correspond to physical density matrices, these interventions measure the model’s dependence on feature blocks rather than causal contributions to entanglement.

Instead, EntNet uses the combined spatial and tensor structure of pair, three-body, and four-body correlations. Higher-order features provide information about how a pair-correlation pattern should be interpreted, while pair features supply complementary spatial information about higher-order structures. This behavior is consistent with learning joint signatures of entanglement rather than separately approximating the two-, three-, and four-site tests used to generate labels. Because replacing a feature block by its mean may produce a feature vector that does not correspond to any physical density matrix, these interventions diagnose model dependence and should not be interpreted as physically realizable changes to a quantum state.

Two further controls test whether the selected channels merely provide interchangeable versions of the same signal. Removing four-body features substantially reduces the recognition of $\mathrm { E } _ { 4 }$ states. Removing the selected $r = 4$ pair channels also degrades performance, and replacing them with additional four-body invariants does not recover the loss [Fig. 4(a,b)]. These observations indicate that the selected representation benefits from complementary correlation orders and spatial length scales.

## 14. $J _ { 1 } { - } J _ { 2 }$ feature-block interventions

The aggregate higher-order-certified counts conceal some dependence on certificate order. Replacing pair features leaves all 80 $\mathrm { E } _ { 3 }$ states above threshold in both models. For $\mathrm { E } _ { 4 }$ states, the number remaining above threshold changes from 30 to 29 with the original training distribution and from 30 to 26 with augmented training. In the latter case, the median $\mathrm { E } _ { 4 }$ probability decreases from 0.999999 to 0.671. The augmented model is trained on the 35,000 primary training states and 1,400 additional distance-two support states, with no $J _ { 1 } { - } J _ { 2 }$ or $N = 3 2$ evaluation states. The complete per-state predictions underlying Fig. 4 are retained with the figure data.

TABLE VII. Numbers of $J _ { 1 } { - } J _ { 2 }$ states classified as entangled $( p ~ \ge ~ 0 . 5 )$ before and after replacing a feature block by the corresponding training-set means. The higher-order-certified class contains 80 E and 30 $\mathrm { E } _ { 4 }$ states. “Original” denotes training on the primary distribution alone; “Augmented” adds 1,400 independent states with strong distance-two correlation patterns, without changing the selected 16 channels.
<table><tr><td>Training</td><td>Class</td><td>n Full  $\mathrm { N o } T ^ { ( 3 ) } / T ^ { ( 4 ) }$ </td><td>No pair</td></tr><tr><td rowspan="5">Original</td><td> $\mathrm { S e p }$ </td><td>30 0</td><td>30 0</td></tr><tr><td> $\mathrm { E } _ { 2 }$ </td><td>155 134</td><td>81 153</td></tr><tr><td> $\mathrm { E } _ { > 2 }$ </td><td>110110</td><td>110 109</td></tr><tr><td>Unknown 168 168</td><td></td><td>168 84</td></tr><tr><td> $\mathrm { S e p }$ </td><td>30 0</td><td>30 0</td></tr><tr><td rowspan="4">Augmented</td><td> $\mathrm { E } _ { 2 }$ </td><td>155 155</td><td>83 155</td></tr><tr><td> $\mathrm { E } _ { > 2 }$ </td><td>110 110</td><td>110 106</td></tr><tr><td>Unknown 168 168</td><td></td><td>168</td></tr><tr><td></td><td></td><td>79</td></tr></table>

## 15. Distance-two training-support control

Here $r _ { \mathrm { c e r t } } = 2$ denotes a state whose most negative two-site partial-transpose eigenvalue occurs at separation two. The ratio $R _ { 2 1 }$ , defined in Eq. $( 1 9 ) ,$ , compares spatially averaged second singular values of pair-correlation matrices at separations two and one. With the original 16 selected channels frozen, augmenting the training distribution with the independently generated distance-two states restores positive predictions for all $1 5 5 \mathrm { E } _ { 2 }$ states in the expanded $J _ { 1 } { - } J _ { 2 }$ evaluation set. No state from that evaluation set enters the augmented training distribution. Since the selected pair channels occur only at $r = 1$ and $r = 4$ , while contiguous $T ^ { ( 3 ) }$ and $T ^ { ( 4 ) }$ windows span sites separated by two lattice spacings, this recovery shows that the existing representation can recognize the relevant spatial pattern once the classifier encounters suitable example during training. It does not require introducing an additional input channel.

For the exact-state ablation in Fig. 4(b), the AUROC is evaluated on a 500-state subset of the $N = 1 6$ XY benchmark, comprising 20 constructively separable states, $3 1 3 \mathrm { E } _ { 2 }$ states, 119 higher-order-certified states, and 48 unknown states. The unknown states are excluded from AUROC. The independent 2,000-state separable benchmark is used separately to evaluate the true-negative rate. These controls isolate the efects of specific selected feature groups without altering the classifier-training distribution.