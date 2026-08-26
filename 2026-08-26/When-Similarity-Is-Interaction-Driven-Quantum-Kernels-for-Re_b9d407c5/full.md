# When Similarity Is Interaction-Driven: Quantum Kernels for Regime-Sensitive Learning

Hanqiu Peng

Jianlong Lu

Ying Chen<sup>∗</sup>

Centre for Quantitative Finance, Department of Mathematics Risk Management Institute, National University of Singapore, Singapore

## Abstract

Similarity in many decision systems is governed not by distance alone but by interactions among variables. In fraud and anomaly detection, small local perturbations can cross interaction-sensitive decision boundaries while leaving ambient distance almost unchanged. Motivated by this setting, we introduce a thin-slab interaction model and an interactiondriven quantum kernel constructed from entangled Pauli-string feature maps. The feature map explicitly encodes sparse high-order block interactions. We show that the resulting fidelity kernel is positive semidefinite, admits an exact block-factorized formulation, and induces a geometry sensitive to changes in interaction regime. Across balanced and imbalanced synthetic experiments spanning third-, fourth-, sixth-, and eighth-order interactions, the proposed kernel consistently outperforms linear, radial basis function, Laplacian, and polynomial kernels, as well as an engineered-interaction linear baseline supplied with the planted block products. On real fraud-detection benchmarks, it achieves the highest mean accuracy and F1 on Credit Card Fraud Detection and ranks second on IEEE-CIS Fraud Detection. These findings show that quantum-kernel performance depends on alignment between feature-map geometry and the underlying predictive structure, rather than on Hilbert-space dimension alone. Because the prescribed block-factorized kernel can also be evaluated exactly on a classical computer, the results establish predictive and representational value rather than computational quantum speedup.

Keywords: quantum kernels; inductive bias; high-order interaction learning; thin-slab geometry; fraud detection; kernel methods

## 1 Introduction

In many fraud and anomaly-detection problems, predictive similarity is governed by combinations of contextual variables rather than by distance in the original feature space alone. Two transactions can be close in amount, location, and timing yet belong to diferent risk regimes because of a small set of interacting conditions. This form of contextual anomalousness is consistent with the broader anomaly-detection literature [Chandola et al., 2009, Hilal et al., 2022]. Fraud detection also involves class imbalance, distribution shift, evolving adversarial behavior, and heterogeneous transaction contexts [Dal Pozzolo et al., 2015a, Lucas and Jurgovsky, 2020]. These features motivate similarity measures that respond to sparse high-order interactions rather than only to coordinate-wise proximity.

Quantum kernels provide a circuit-defined approach to constructing similarity. Rather than measuring proximity only in the original feature space, quantum feature maps can encode structured high-order interactions directly into entangled quantum states, inducing a geometry naturally sensitive to interaction-driven behavior. Motivated by fraud and anomaly detection, this study examines whether interaction-aligned quantum kernels can provide a more informative representation of hidden behavioral regimes than conventional distance-based kernels. Our central hypothesis is simple: when predictive structure is governed by interactionsensitive decision boundaries rather than distance alone, kernels aligned with those interactions can substantially improve the representation of risk and regime similarity.

Kernel methods derive their predictive power from the geometry induced by their similarity functions. Linear kernels encode coordinate-wise relationships, while radial basis function (RBF) and Laplacian kernels define similarity through input-space distance. Polynomial kernels can represent high-order interactions, but generic polynomial kernels do not specifically privilege the sparse block products relevant here. These inductive biases are efective in many learning problems, but can become poorly aligned when predictive regimes are separated by sparse interaction boundaries rather than large geometric distances. In such settings, nearby observations may belong to entirely diferent behavioral regimes, whilst distant observations may share the same underlying interaction structure. The central challenge is therefore not simply nonlinear representation, but the construction of a similarity geometry aligned with the true predictive mechanism.

Recent work in both classical and quantum machine learning has increasingly highlighted the importance of inductive-bias alignment [Havlíček et al., 2019, Schuld and Killoran, 2019, Kübler et al., 2021, Huang et al., 2021, Caro et al., 2022]. High-order interactions are particularly dificult because the relevant predictive signal is often sparse and structurally localized. This challenge is well recognized in classical interaction learning, where detecting and selecting interaction terms becomes increasingly dificult as the number of candidate feature combinations grows [Sorokina et al., 2008, Bien et al., 2013]. Polynomial kernels can theoretically represent such interactions, but the relevant terms are embedded within combinatorially large and largely unstructured feature expansions. Under finite samples, learning may fail to exploit the relevant terms efectively because they are embedded amongst many irrelevant monomials, making sparse interaction regimes dificult to isolate through generic kernel constructions. Related models such as factorization machines and neural factorization machines address this issue by explicitly parameterizing or learning feature interactions in sparse predictive settings [Rendle, 2010, He and Chua, 2017]. This motivates the search for feature maps whose induced geometry is explicitly aligned with interaction-driven predictive structure.

To investigate this question systematically, we introduce a thin-slab interaction framework that separates interaction-driven predictive structure from generic distance-based similarity. Within this framework, an entangled Pauli-string feature map encodes fixed block-wise highorder products. The resulting fidelity kernel is positive semidefinite, admits an exact blockfactorized formulation, and depends explicitly on coordinate-level phase diferences and blockproduct discrepancies. Because the block partition is fixed and known, the implementation evaluates the block states separately and avoids constructing a statevector spanning all input features. This is a block-factorized quantum-circuit representation of the prescribed interaction geometry, although the same known-block structure also admits eficient classical evaluation. The objective is therefore feature-map alignment rather than representation dimension alone, consistent with analyses that relate quantum-kernel performance to the target function class and kernel spectrum [Kübler et al., 2021, Huang et al., 2021, Caro et al., 2022].

Our work is related to the growing literature on quantum kernel methods and quantumenhanced feature spaces Havlíček et al. [2019], Schuld and Killoran [2019]. Prior studies have investigated generalization properties Caro et al. [2022], efective dimension Abbas et al. [2021], concentration phenomena Thanasilp et al. [2024], and the importance of data structure in quantum machine learning Huang et al. [2021], Kübler et al. [2021]. Rather than pursuing a universal quantum-advantage claim, our work focuses on a more specific and practically relevant question:

under what predictive structures can quantum-induced similarity geometry provide predictive benefits over conventional kernels? In particular, we argue that interaction-driven decision boundaries provide a natural setting in which quantum feature maps may ofer substantial representational benefits.

Empirically, we adopt a three-stage evaluation design. First, controlled synthetic experiments test the interaction mechanism at third and fourth order. Second, a higher-order scaling study tests whether its predictive efectiveness persists as input dimension and interaction order increase. Third, two fraud-detection benchmarks assess applicability beyond the controlled synthetic setting using data from the Fraud Dataset Benchmark (FDB) Grover et al. [2022].

Across the synthetic settings, from the controlled third- and fourth-order tasks to sixth- and eighth-order scaling, the proposed quantum kernel consistently outperforms standard linear, RBF, Laplacian, and polynomial kernels. It also outperforms InterFea, a ridge-regularized linear classifier trained on the original features augmented with the planted block products, in every evaluated configuration—by accuracy in balanced tasks and F1 in imbalanced tasks. This shows that the fidelity geometry adds value beyond direct linear access to those products. In a more interaction-sensitive balanced setting, the quantum kernel raises accuracy from approximately 0.50–0.52 for standard kernels to over 0.75, a relative improvement of more than 45%. In imbalanced settings, it raises F1 from approximately 0.20–0.28 to above 0.53, more than doubling performance in several regimes.

In the scaling study, the quantum kernel achieves the highest mean accuracy and F1 in both the 60-dimensional sixth-order and 80-dimensional eighth-order settings, with (Acc., F1) = (0.706, 0.412) and (0.695, 0.395), respectively. Relative to InterFea, the paired mean F1 gaps are 0.032 and 0.075. Together with the controlled third- and fourth-order results, these findings establish a persistent advantage across all evaluated orders, with the largest noise-free imbalanced F1 gap at eighth order.

On the real fraud-detection datasets, the method achieves the strongest F1 on Credit Card Fraud Detection, improving mean F1 from 0.348 for the best competing classical kernel to 0.472, corresponding to a relative improvement of approximately 36%. It also remains competitive on IEEE-CIS Fraud Detection, outperforming most conventional kernels and ranking second to the Laplacian kernel. Together, the controlled synthetic results demonstrate a quantuminduced predictive advantage across progressively more demanding interaction regimes, while the real-data results show that the method remains competitive beyond the controlled synthetic setting.

The main contributions of this paper are as follows:

• We introduce a thin-slab interaction model as a controlled abstraction of interactionsensitive decision boundaries, where Euclidean proximity becomes misaligned with predictive similarity.

• We propose an interaction-driven quantum kernel based on entangled Pauli-string feature maps that explicitly encode sparse high-order interactions.

• We show theoretically that the resulting fidelity kernel is positive semidefinite, admits an exact block-factorized formulation, and induces a geometry naturally sensitive to interaction-driven behavior.

• We quantify the resources required by the block-factorized implementation, including reusable qubits, logical gate counts, circuit depth, statevector storage, and kernel-matrix memory.

• We provide systematic numerical evidence across multiple interaction orders, class-balance regimes, and scaling settings, showing consistent predictive gains over conventional distancebased kernels and an explicit known-block interaction baseline.

• We evaluate the method on two real fraud-detection benchmarks, where it achieves strong or competitive performance beyond the controlled synthetic setting.

• More broadly, we show that the efectiveness of quantum kernels depends critically on inductive-bias alignment between feature-map geometry and the underlying predictive structure, highlighting the importance of interaction-driven representations in quantum machine learning.

Scope of the contribution. This paper builds on quantum kernels, inductive-bias analysis, and classical high-order interaction learning. Existing work has established that quantum feature maps can induce rich similarity geometries and that their practical value depends on feature-map design rather than Hilbert-space dimension alone. The contribution here is a controlled setting in which the required alignment can be stated and tested directly: the thin-slab construction makes Euclidean proximity a poor proxy for label similarity, and the proposed feature map is designed to match that structure rather than to maximize expressivity. The paper therefore demonstrates a quantum-induced predictive advantage in the evaluated regimes, not universal or computational quantum advantage. Because the disjoint-block kernel studied here is exactly classically evaluable, the advantage established is representational and statistical rather than computational.

The remainder of this paper is structured as follows. Section 2 reviews related work on quantum kernels, inductive bias, and high-order interaction learning. Section 3 introduces the thin-slab interaction model, and Section 4 develops the proposed interaction-driven quantum kernel and its theoretical properties. Section 5 describes the experimental protocol, and Section 6 reports empirical evaluations on synthetic interaction-learning tasks and real frauddetection benchmarks. Section 7 discusses implications, limitations, and directions for future research, and Section 8 concludes.

## 2 Related Work

## 2.1 Quantum kernels and inductive bias

Quantum kernel methods map classical data into quantum states and define similarity through state overlap [Havlíček et al., 2019, Schuld and Killoran, 2019]. Early work focused primarily on the representational richness of quantum feature spaces and the possibility of quantumenhanced learning through high-dimensional embeddings. Subsequent work has shown, however, that practical performance depends strongly on feature-map design, data structure, and the geometry induced by the resulting kernel, rather than on Hilbert-space dimension alone. Related studies have investigated efective dimension [Abbas et al., 2021], high-dimensional implementation on noisy quantum processors [Peters et al., 2021], exponential concentration of quantum kernels [Thanasilp et al., 2024], and trainable or optimized quantum-kernel constructions [Pellow-Jarman et al., 2024, Xu et al., 2024].

Our work follows this feature-map-design perspective. Rather than treating quantum kernels as generic high-dimensional embeddings, we investigate whether quantum feature maps can provide advantages when explicitly aligned with interaction-sensitive predictive structure. In particular, we focus on sparse high-order interaction regimes in which Euclidean proximity becomes a poor proxy for predictive similarity. The proposed Pauli-string feature map is therefore designed not for maximal expressivity, but for structured interaction encoding and interaction-aware similarity geometry. This is consistent with recent work on quantum featuremap optimization and kernel-target alignment, where the goal is to design quantum kernels whose induced similarity geometry better matches the supervised learning task [Pellow-Jarman et al., 2024, Xu et al., 2024].

## 2.2 High-order interactions and similarity geometry

High-order interactions are central to many learning problems, including anomaly detection, recommendation systems, genomics, and financial risk modelling. However, such interactions are dificult to identify when embedded implicitly within large nonlinear feature spaces. Linear models fail when predictive structure contains little first-order signal, whilst distance-based kernels smooth according to the geometry of the original features, which may be unrelated to hidden interaction regimes. Polynomial kernels can theoretically represent high-order interactions, but their feature spaces contain all monomials up to a given degree. The number of such monomials grows combinatorially with dimension and interaction order, making the relevant sparse terms dificult to isolate within an unstructured feature space under finite samples.

Classical machine learning has long studied ways to represent and select interaction structure more eficiently. Functional ANOVA and ANOVA kernels provide decompositions of functions or kernels into lower- and higher-order interaction components [Wahba, 1990, Stitson et al., 1999]. Sparse and hierarchical interaction models aim to select informative interactions while controlling model complexity [Bien et al., 2013, Lim and Hastie, 2015]. Tree-based interactiondetection methods provide another route for identifying non-additive feature efects [Sorokina et al., 2008]. In sparse predictive settings, factorization machines and neural factorization machines explicitly parameterize feature interactions without constructing a full unstructured polynomial expansion [Rendle, 2010, He and Chua, 2017].

Our work is conceptually related to this direction, but difers in its use of quantum-induced similarity geometry. Instead of explicitly enumerating interaction features, the proposed quantum kernel encodes structured block-wise interactions directly into entangled feature-map phases. This creates a similarity measure that depends explicitly on hidden interaction regimes rather than Euclidean proximity in the original feature space.

## 2.3 Fraud detection and interaction-sensitive learning

Fraud detection and anomaly monitoring provide natural settings for interaction-sensitive learning because predictive regimes are often determined by subtle behavioural combinations rather than large feature deviations. Fraudulent transactions are frequently designed to remain locally similar to legitimate activity, making Euclidean distance a poor indicator of predictive similarity. Small contextual changes, such as device identity, timing, behavioural history, or transaction sequence, may abruptly alter the underlying risk regime despite minimal geometric change in the raw feature space. These properties make fraud detection particularly challenging under severe class imbalance, heterogeneous behavioural patterns, and evolving adversarial dynamics.

Beyond static transaction-level classification, several studies emphasize behavioural and sequential structure in credit-card fraud detection. For example, sequence-based fraud-detection models formulate the task using transaction histories and recurrent neural networks, showing that temporal context and cardholder behaviour can improve detection beyond isolated transaction features [Jurgovsky et al., 2018]. Other work has developed realistic modelling strategies for credit-card fraud under delayed labels, concept drift, and non-stationary transaction distributions [Dal Pozzolo et al., 2018]. These studies support the view that fraud risk is often context-dependent and behaviour-sensitive, rather than determined by marginal feature deviations alone.

To support reproducible evaluation across fraud and abuse-detection tasks, Grover et al. [2022] introduced the Fraud Dataset Benchmark (FDB), which provides standardized datasets, train-test splits, and evaluation protocols. In this work, we use the Credit Card Fraud Detection and IEEE-CIS Fraud Detection benchmarks within the FDB framework. Importantly, these experiments are not intended to claim that real fraud labels follow the synthetic thin-slab interaction mechanism exactly. Instead, they test whether interaction-aligned quantum kernels remain efective when the underlying predictive structure is not controlled to match the proposed

inductive bias.

## 3 High-Order Interaction Learning under Thin-Slab Geometry

We now define the controlled learning problem used to test whether an interaction-aware quantum kernel can outperform standard classical kernels. The construction is intentionally simple: the label is generated by planted block-wise high-order interactions, whereas the sampling distribution places many observations close to the coordinate hyperplanes where those interaction signs can flip. This creates a mismatch between Euclidean proximity and label similarity.

## 3.1 Block-structured high-order interaction signal

Let d denote the input dimension, let B be the number of blocks, and let k be the number of coordinates in each block. These quantities satisfy

$$
d = B k .\tag{1}
$$

For an input vector $x \in [ - 1 , 1 ] ^ { d }$ , we use B disjoint blocks of size $k ,$ indexed by $b \in \{ 1 , \ldots , B \}$ :

$$
\begin{array} { r } { \boldsymbol { x } = \big ( \boldsymbol { x } ^ { ( 1 ) } , \boldsymbol { x } ^ { ( 2 ) } , \ldots , \boldsymbol { x } ^ { ( B ) } \big ) , \qquad \boldsymbol { x } ^ { ( b ) } = ( x _ { b , 1 } , \ldots , x _ { b , k } ) \in [ - 1 , 1 ] ^ { k } . } \end{array}\tag{2}
$$

For each block $b \in \{ 1 , \ldots , B \}$ , define a block-level regime bit by the sign of the k-way product

$$
r _ { b } ( x ) = { \mathrm { s i g n } } \left( \prod _ { j = 1 } ^ { k } x _ { b , j } \right) \in \{ + 1 , - 1 \} .\tag{3}
$$

Let $w _ { b } > 0$ denote the fixed weight of block $b ,$ and let $\tau$ denote a decision threshold. The binary label is generated by the weighted majority vote

$$
y ( x ) = \mathrm { s i g n } \biggl ( \sum _ { b = 1 } ^ { B } w _ { b } r _ { b } ( x ) - \tau \biggr ) .\tag{4}
$$

In balanced experiments we set $\tau = 0$ with equal block weights. In imbalanced experiments we keep equal block weights but use a stricter positive-class threshold, so that $y = + 1$ only when the block-regime score is suficiently large. In the reported imbalanced runs we use $\tau = 3$ for both $( d , k , B ) = ( 9 , 3 , 3 )$ and $( 2 0 , 4 , 5 )$ ; thus, in the $B = 3$ case the positive class occurs only when all three block-regime bits are positive, while in the $B = 5$ case it occurs when at least four of the five block-regime bits are positive. If the argument of the sign function is exactly zero, we assign $y = + 1$ ; this convention has negligible efect under the continuous sampling scheme.

This model is explicitly interaction-driven. The relevant information in block b is not contained in any single coordinate $x _ { b , j }$ , nor in pairwise distances between samples. Instead, the signal is encoded in the sign of the planted k-way monomial

$$
m _ { b } ( \boldsymbol { x } ) = \prod _ { j = 1 } ^ { k } x _ { b , j } .\tag{5}
$$

For $k > 2$ , this produces a target structure that is invisible to linear models and not directly represented by polynomial kernels whose degree is below $k$

## 3.2 Thin-slab distribution

We define the thin-slab distribution explicitly. Let $\epsilon \in ( 0 , 1 )$ denote the slab thickness. For each block $b ,$ select one active coordinate

$$
j _ { b } ^ { \star } \sim \operatorname { U n i f } \{ 1 , \dots , k \} ,\tag{6}
$$

and then draw

$$
\begin{array} { r } { x _ { b , j _ { b } ^ { \star } } \sim \mathrm { U n i f } ( [ - 1 , 1 ] ) , \qquad x _ { b , j } = \epsilon \xi _ { b , j } , \quad j \neq j _ { b } ^ { \star } , \qquad \xi _ { b , j } \overset { \mathrm { i . i . d . } } { \sim } \mathrm { U n i f } ( [ - 1 , 1 ] ) . } \end{array}\tag{7}
$$

Here, Unif denotes the uniform distribution, and the auxiliary variables $\xi _ { b , j }$ are independently and identically distributed. When ϵ is small, most coordinates in each block are close to zero, so each sample lies near several coordinate hyperplanes. The support is a union of thin slabs rather than a full-dimensional uniform cloud.

This sampling scheme amplifies the diference between input-space distance and interaction geometry. Since most inactive coordinates are of order ϵ, a sign flip in one inactive coordinate can reverse the sign of the block product while changing the Euclidean norm by only $O ( \epsilon )$ Thus, the label can change sharply across a very small Euclidean perturbation. This is exactly the regime in which a local distance-based kernel may assign high similarity to samples whose block-regime bits difer.

Proposition 1 (Small Euclidean perturbations can flip block regimes). Fix a block b and suppose x, $x ^ { \prime } \in [ - 1 , 1 ] ^ { d }$ difer only in one coordinate ℓ of that block. Assume

$$
x _ { b , \ell } = \delta , \qquad x _ { b , \ell } ^ { \prime } = - \delta , \qquad \delta > 0 ,\tag{8}
$$

and $x _ { a , j } = x _ { a , j } ^ { \prime }$ for all $( a , j ) \neq ( b , \ell )$ . If all other coordinates in block b are nonzero, then

$$
\begin{array} { r } { r _ { b } ( x ^ { \prime } ) = - r _ { b } ( x ) , } \end{array}\tag{9}
$$

while

$$
\| x - x ^ { \prime } \| _ { 2 } = 2 \delta .\tag{10}
$$

In particular, under the thin-slab distribution in (7), such regime flips can occur at distance $O ( \epsilon )$

Proof. The two samples difer only by replacing ${ \boldsymbol { x } } _ { b , \ell }$ with $- { \boldsymbol { x } } _ { b , \ell }$ . Therefore the block product $\textstyle \prod _ { j = 1 } ^ { k } x _ { b , j }$ changes sign while all other coordinates remain unchanged. Since the samples difer in exactly one coordinate by 2δ, their Euclidean distance is 2δ. □

Proposition 1 explains why the problem is boundary-sensitive. The boundary of a block regime is the union of coordinate hyperplanes

$$
\bigcup _ { j = 1 } ^ { k } \{ x ^ { ( b ) } \in [ - 1 , 1 ] ^ { k } : x _ { b , j } = 0 \} .\tag{11}
$$

The thin-slab distribution concentrates many samples near these hyperplanes, making Euclidean smoothness a poor default inductive bias.

## 3.3 Why generic kernels can be misaligned

For two inputs x and $z ,$ the linear kernel,

$$
k _ { \mathrm { l i n } } ( x , z ) = x ^ { \top } z ,\tag{12}
$$

is inadequate because the target depends on block-wise products rather than first-order coordinates. Distance-based kernels such as

$$
k _ { \mathrm { R B F } } ( x , z ) = \exp ( - \gamma \| x - z \| _ { 2 } ^ { 2 } ) , \qquad k _ { \mathrm { L a p } } ( x , z ) = \exp ( - \gamma \| x - z \| _ { 1 } ) ,\tag{13}
$$

are more expressive but still define similarity through input-space distance. Proposition 1 shows that nearby samples can have diferent block-regime bits, so these kernels can smooth across interaction boundaries.

Here, the positive parameter $\gamma$ controls the distance scale of the RBF and Laplacian kernels and is tuned separately for each kernel family.

Polynomial kernels provide a more serious baseline because they can represent feature interactions:

$$
k _ { \mathrm { p o l y } , p } ( x , z ) = ( \gamma x ^ { \top } z + c ) ^ { p } .\tag{14}
$$

Here, $p$ is the polynomial degree, $\gamma > 0$ is a scale parameter, and $c \geq 0$ is an ofset; all three quantities are specified or tuned as part of the corresponding baseline. If $p < k$ , the planted degree-k terms are absent. When $c \neq 0$ and $p \geq k$ , the kernel contains monomials of degrees up to $p ,$ including the relevant degree-k terms, but embeds them within a much larger expansion. The number of monomials of total degree at most $p$ in d variables is

$$
{ \binom { d + p } { p } } .\tag{15}
$$

When $c = 0 ,$ the kernel is homogeneous and contains only monomials of total degree exactly $p ;$ in that case, the planted terms occur directly only when $p = k$ . For the inhomogeneous expansion, only $B = d / k$ of the monomials correspond exactly to the planted block products. For example, when $d = 2 0$ and $p = 4$ , the polynomial feature space contains $( _ { 4 } ^ { 2 4 } ) = 1 0 \AA , 6 2 6$ monomials up to degree four, while only five are the planted block products. This does not mean that polynomial kernels are theoretically unable to represent the target interaction. Rather, under finite samples the relevant block-product terms are embedded in a large unstructured monomial space, so the sparse signal can be dificult to isolate amongst many irrelevant interaction terms. In our experiments, polynomial kernels of several degrees are evaluated under the same validation protocol as the other baselines, but higher-degree kernels remain sensitive to scaling and conditioning, especially under thin-slab sampling where many coordinates have small magnitude. Thus, polynomial kernels are useful but imperfect baselines for sparse k-way interaction learning.

This motivates InterFea, an engineered-interaction linear baseline. We augment the raw feature vector with the planted block products,

$$
\Phi _ { \mathrm { I n t e r F e a } } ( x ) = \left( x _ { 1 } , \dots , x _ { d } , \prod _ { j = 1 } ^ { k } x _ { 1 , j } , \dots , \prod _ { j = 1 } ^ { k } x _ { B , j } \right) ,\tag{16}
$$

which induces the linear kernel

$$
k _ { \mathrm { I n t e r F e a } } ( x , z ) = \Phi _ { \mathrm { I n t e r F e a } } ( x ) ^ { \top } \Phi _ { \mathrm { I n t e r F e a } } ( z ) .\tag{17}
$$

In the experiments, the augmented features are standardized using training-set statistics, and an $L _ { \mathrm { 2 ^ { - r e g u l a r i z e d } } }$ linear least-squares classifier is fitted directly in this engineered feature space using scikit-learn’s RidgeClassifier. Binary labels are encoded as $\{ - 1 , + 1 \}$ , and predictions are determined by the sign of the fitted linear score. This primal implementation avoids explicitly constructing the corresponding Gram matrix while retaining the kernel interpretation in (17). InterFea therefore provides a strong control for determining whether the performance gain comes from encoding the relevant interaction features or from the specifically quantum fidelity geometry.

## 4 Quantum Feature Map and Kernel Construction

## 4.1 Design hypothesis

The proposed feature map is based on the following hypothesis: when the target function depends on sparse high-order block interactions, a kernel that explicitly compares samples through these interactions should provide a better-aligned inductive bias than generic kernels that distribute capacity over many irrelevant directions. In the synthetic model, the relevant interaction variables are the block products $\begin{array} { r } { m _ { b } ( x ) = \prod _ { j = 1 } ^ { k } x _ { b , j } } \end{array}$ . The feature map is therefore designed so that diferences in $m _ { b } ( x )$ directly afect the quantum phase and, consequently, the fidelity kernel. This design does not imply that the resulting kernel is universally superior. Rather, it predicts an advantage only when the encoded block-product structure is aligned with the data-generating mechanism.

## 4.2 Block-wise Pauli-string feature map

For the mathematical construction, let $d = B k$ and associate one qubit with each input coordinate. For block $b ,$ let $Z _ { b , j }$ denote the Pauli-Z operator acting on the qubit corresponding to coordinate $x _ { b , j }$ , let $\alpha _ { b }$ denote the interaction-phase scale for that block, and let $\beta$ denote the shared one-body phase scale. Let H denote the Hadamard gate. We define the d-qubit feature state

$$
| \psi ( x ) \rangle = \bigotimes _ { b = 1 } ^ { B } | \psi ^ { ( b ) } ( x ^ { ( b ) } ) \rangle ,\tag{18}
$$

where the k-qubit block state is

$$
| \psi ^ { ( b ) } ( x ^ { ( b ) } ) \rangle = \exp \left( i \alpha _ { b } \left( \prod _ { j = 1 } ^ { k } x _ { b , j } \right) Z _ { b , 1 } \cdot \cdot \cdot Z _ { b , k } \right) \left[ \prod _ { j = 1 } ^ { k } \exp ( i \beta x _ { b , j } Z _ { b , j } ) \right] H ^ { \otimes k } | 0 \rangle ^ { \otimes k } .\tag{19}
$$

Equivalently, the full feature map may be written as

$$
| \psi ( x ) \rangle = \left[ \prod _ { b = 1 } ^ { B } \exp \left( i \alpha _ { b } \left( \prod _ { j = 1 } ^ { k } x _ { b , j } \right) Z _ { b , 1 } \cdot \cdot \cdot Z _ { b , k } \right) \right] \left[ \prod _ { i = 1 } ^ { d } \exp ( i \beta x _ { i } Z _ { i } ) \right] H ^ { \otimes d } | 0 \rangle ^ { \otimes d } .\tag{20}
$$

All phase gates commute because they are diagonal in the computational basis. The one-body phases preserve coordinate-level information, while the block-wise Pauli-string phase encodes the k-way product within each block. In the reported experiments, a common interaction scale $\alpha _ { b } = \alpha$ is shared across all blocks and tuned jointly with $\beta .$ Although the full feature state is written mathematically as a d-qubit state, its factorization across blocks allows the B block states to be evaluated separately. In simulator-based settings, the same k wires are reused; the $( d , k ) = ( 8 0 , 8 )$ scaling setting instead evaluates the exactly equivalent closed-form state amplitudes. Numerical-backend details are given in Appendix A.

The associated quantum kernel is the fidelity kernel

$$
k _ { Q } ( x , x ^ { \prime } ) = \vert \langle \psi ( x ) \vert \psi ( x ^ { \prime } ) \rangle \vert ^ { 2 } .\tag{21}
$$

Proposition 2 (Positive semidefiniteness). The function k<sub>Q</sub> in (21) is a positive semidefinite kernel.

Proof. For any samples $x _ { 1 } , \ldots , x _ { N }$ and coeficients $c _ { 1 } , \ldots , c _ { N }$ 2

$$
\sum _ { a , b = 1 } ^ { N } c _ { a } ^ { * } c _ { b } k _ { Q } ( x _ { a } , x _ { b } ) = \sum _ { a , b = 1 } ^ { N } c _ { a } ^ { * } c _ { b } | \langle \psi ( x _ { a } ) | \psi ( x _ { b } ) \rangle | ^ { 2 }\tag{22}
$$

$$
= \left\| \sum _ { a = 1 } ^ { N } c _ { a } | \psi ( x _ { a } ) \rangle \langle \psi ( x _ { a } ) | \right\| _ { \mathrm { H S } } ^ { 2 } \geq 0 .\tag{23}
$$

Here, ∥ · ∥<sub>HS</sub> denotes the Hilbert–Schmidt norm.

## 4.3 Closed-form block kernel

Since the block circuit is diagonal after the Hadamard layer, the block state has expansion

$$
| \psi ^ { ( b ) } ( x ^ { ( b ) } ) \rangle = 2 ^ { - k / 2 } \sum _ { s \in \{ \pm 1 \} ^ { k } } \exp \left[ i \beta \sum _ { j = 1 } ^ { k } x _ { b , j } s _ { j } + i \alpha _ { b } \left( \prod _ { j = 1 } ^ { k } x _ { b , j } \right) \prod _ { j = 1 } ^ { k } s _ { j } \right] | s \rangle ,\tag{24}
$$

where $s _ { j } \in \{ \pm 1 \}$ denotes the $Z -$ -eigenvalue of qubit $j$ . Therefore,

$$
\langle \psi ^ { ( b ) } ( x ^ { ( b ) } ) | \psi ^ { ( b ) } ( x ^ { \prime ( b ) } ) \rangle = 2 ^ { - k } \sum _ { s \in \{ \pm 1 \} ^ { k } } \exp \left[ i \beta \sum _ { j = 1 } ^ { k } ( x _ { b , j } ^ { \prime } - x _ { b , j } ) s _ { j } + i \alpha _ { b } \{ m _ { b } ( x ^ { \prime } ) - m _ { b } ( x ) \} \prod _ { j = 1 } ^ { k } s _ { j } \right] .\tag{25}
$$

Let

$$
a _ { j } = \beta ( x _ { b , j } ^ { \prime } - x _ { b , j } ) , \qquad c _ { b } = \alpha _ { b } ( m _ { b } ( x ^ { \prime } ) - m _ { b } ( x ) ) .\tag{26}
$$

Using the parity identity separating the two sectors of $\Pi _ { j } s _ { j }$ , we obtain

$$
\langle \psi ^ { ( b ) } ( x ^ { ( b ) } ) | \psi ^ { ( b ) } ( x ^ { \prime ( b ) } ) \rangle = \cos ( c _ { b } ) \prod _ { j = 1 } ^ { k } \cos ( a _ { j } ) + i ^ { k + 1 } \sin ( c _ { b } ) \prod _ { j = 1 } ^ { k } \sin ( a _ { j } ) .\tag{27}
$$

Hence the block contribution to the fidelity kernel is

$$
k _ { Q } ^ { ( b ) } ( x ^ { ( b ) } , x ^ { \prime ( b ) } ) = \left| \cos ( c _ { b } ) \prod _ { j = 1 } ^ { k } \cos ( a _ { j } ) + i ^ { k + 1 } \sin ( c _ { b } ) \prod _ { j = 1 } ^ { k } \sin ( a _ { j } ) \right| ^ { 2 } .\tag{28}
$$

This expression shows explicitly that the kernel depends not only on coordinate-wise diferences, but also on block-product discrepancies $m _ { b } ( { \boldsymbol { x } } ^ { \prime } ) - m _ { b } ( { \boldsymbol { x } } )$

Because the full state factorizes over blocks,

$$
\langle \psi ( x ) | \psi ( x ^ { \prime } ) \rangle = \prod _ { b = 1 } ^ { B } \langle \psi ^ { ( b ) } ( x ^ { ( b ) } ) | \psi ^ { ( b ) } ( x ^ { \prime ( b ) } ) \rangle ,\tag{29}
$$

and therefore

$$
k _ { Q } ( x , x ^ { \prime } ) = \prod _ { b = 1 } ^ { B } k _ { Q } ^ { ( b ) } ( x ^ { ( b ) } , x ^ { \prime ( b ) } ) .\tag{30}
$$

## 4.4 Representation structure and circuit resources

The resource comparison below follows the implementations used in the synthetic and realdata experiments. All reported settings use a fixed partition into $B = d / k$ blocks; they do not enumerate all degree-k subsets. For IEEE-CIS, the 60 selected numerical features are ordered by their training-set correlation scores before forming ten blocks of six. Accordingly, the classical InterFea baseline in $\operatorname { E q }$ . 16 appends exactly B block products to the original data. Its output dimension is $d + B$ , and its construction cost is $O ( B k ) = O ( d )$ per observation.

For the simulator-based experiments, the PennyLane implementation also exploits the same block factorization. It creates a default.qubit device with only k wires and prepares the B block states separately, so the peak simulated qubit requirement is $k ,$ rather than d. The $( d , k ) = ( 8 0 , 8 )$ setting uses the exact closed-form numerical backend described in Appendix $\operatorname { A } ;$ the resource counts below nevertheless describe the quantum circuit represented by the same feature map. For one block, that circuit applies k Hadamard gates, optionally k local $R _ { Z }$ gates when $\beta \neq 0$ , and one k-qubit Pauli-Z rotation, where $R _ { Z }$ denotes a single-qubit rotation about the Z axis. A standard controlled-NOT (CNOT) ladder decomposition of the Pauli rotation uses $2 ( k - 1 )$ logical CNOTs and one further $R _ { Z }$ gate. Let $N _ { H } , \ N _ { R _ { Z } }$ , and $N _ { \mathrm { C N O T } }$ denote the total numbers of the corresponding logical gates across all B block preparations for one observation. Then

$$
N _ { H } = B k = d , \qquad N _ { R _ { Z } } = B + B k { \bf 1 } _ { \{ \beta \neq 0 \} } , \qquad N _ { \mathrm { C N O T } } = 2 B ( k - 1 ) .\tag{31}
$$

Here, $\mathbf { 1 } _ { \{ \beta \neq 0 \} }$ is the indicator that the one-body phase is active. With the ladder executed serially within a block, the logical depth of one block-state preparation is 2k for $\beta = 0$ and $2 k + 1$ for $\beta \neq 0$ , before hardware routing and basis-gate translation.

Table 1: Logical circuit resources per observation. Depth refers to one block circuit under a CNOT-ladder decomposition.
<table><tr><td>Dataset/setting</td><td>B Peak qubits</td><td>InterFea dim.</td><td>H</td><td> $R _ { Z } \colon \beta = 0 ~ ( \neq 0 )$ </td><td>CNOTs</td><td>Block depth:  $\beta = 0 ~ ( \neq 0 )$ </td></tr><tr><td>Synthetic:  $d = 9 , \ k = 3$ </td><td>3</td><td>3</td><td>12 9</td><td>3 (12)</td><td>12</td><td>6 (7)</td></tr><tr><td>Synthetic:  $d = 2 0 , \ k = 4$ </td><td>5</td><td>4</td><td>25 20</td><td>5 (25)</td><td>30</td><td>8 (9)</td></tr><tr><td>Synthetic scaling:  $d = 6 0 , \ k = 6$ </td><td>10</td><td>6</td><td>70 60</td><td>10 (70)</td><td>100</td><td>12 (13)</td></tr><tr><td>Synthetic scaling:  $d = 8 0 , \ k = 8$ </td><td>10</td><td>8</td><td>90 80</td><td>10 (90)</td><td>140</td><td>16 (17)</td></tr><tr><td>Credit Card:  $d = 2 8 , \ k = 4$ </td><td>7</td><td>4</td><td>35 28</td><td>7 (35)</td><td>42</td><td>8 (9)</td></tr><tr><td>IEEE-CIS:  $d = 6 0 , \ k = 6$ </td><td>10</td><td>6</td><td>70 60</td><td>10 (70)</td><td>100</td><td>12 (13)</td></tr></table>

In the higher-order scaling study, the block count is fixed at $B = 1 0$ . Increasing k from six to eight therefore raises the peak circuit width from six to eight reusable qubits and the logical CNOT count per observation from 100 to 140, while the engineered classical representation grows from 70 to 90 coordinates. The feature map thus provides a direct circuit realization of the prescribed interaction order without enumerating all candidate degree-k subsets. Under the known fixed partition, however, both the circuit representation and the engineered classical representation remain eficiently evaluable; the comparison is statistical rather than an asymptotic runtime separation.

For statevector-based evaluation, each observation is represented and cached by B block statevectors. The statevector storage therefore scales as $O ( B 2 ^ { k } )$ per observation, while one pairwise kernel entry costs $O ( B 2 ^ { k } )$ when computed from the cached block states. Because the block overlap also admits the closed form in Eq. 27, the same kernel entry can alternatively be evaluated in $O ( B k )$ time.

The selected quantum configurations for both real-data benchmarks have $\beta \neq 0$ . Consequently, the per-observation logical gate counts are 28 Hadamard gates, 35 $R _ { Z }$ rotations, and 42 CNOTs for Credit Card Fraud, and 60 Hadamard gates, 70 $R _ { Z }$ rotations, and 100 CNOTs for IEEE-CIS. Since the blocks are evaluated separately, the peak simulated device sizes remain four and six qubits, respectively.

At the dataset level, let $n _ { \mathrm { t r a i n } }$ and $n _ { \mathrm { t e s t } }$ denote the numbers of training and test observations. One complete construction produces

$$
B ( n _ { \mathrm { t r a i n } } + n _ { \mathrm { t e s t } } )\tag{32}
$$

block statevectors. Since each statevector contains $2 ^ { k }$ complex amplitudes, their temporary storage scales as $O \Big ( B 2 ^ { k } ( n _ { \mathrm { t r a i n } } + n _ { \mathrm { t e s t } } ) \Big )$ . The dense training and test–training kernel matrices require

$$
O \Big ( n _ { \mathrm { t r a i n } } ^ { 2 } + n _ { \mathrm { t e s t } } n _ { \mathrm { t r a i n } } \Big )\tag{33}
$$

storage.

For one complete quantum-kernel construction using the selected hyperparameters, the Credit Card Fraud experiment contains 834 training and 56,962 test observations. Its temporary complex block-state arrays require approximately 98.8 MiB, while the corresponding float64 training and test–training kernel matrices together require approximately 367.8 MiB. The IEEE-CIS experiment contains 6000 training and 29,527 test observations, requiring approximately 346.9 MiB for the temporary complex block-state arrays and 1,626.3 MiB for the corresponding float64 kernel matrices. These figures refer to the post-selection construction used for test evaluation. The complete test partitions are not used during model selection; hyperparameter tuning uses only the training-derived subtraining and validation partitions described below.

Although the kernel is defined through fidelities between states prepared by a Pauli-string quantum feature map, its block-factorized structure permits exact classical evaluation in the present setting. The reported gate counts therefore characterize a pre-transpilation logical circuit representation; they are not hardware-executed gate counts and should not be interpreted as evidence of quantum computational speedup.

## 5 Experimental Design

## 5.1 Synthetic settings

We evaluate two representative block-structured configurations:

• Setting 1: $d = 9 , k = 3 , B = 3 ;$

• Setting 2: $d = 2 0 , k = 4 , B = 5 .$

For both settings, we vary slab thickness $\epsilon \in \{ 0 . 0 5 , 0 . 1 5 \}$ and training-label flip probability $p _ { \mathrm { f l i p } } \in \{ 0 . 0 0 , 0 . 1 0 \}$ . Clean labels are generated from the block-regime rule, and label flips are applied to the training labels only. The test labels remain clean, so $p _ { \mathrm { f i i p } }$ measures robustness to noisy supervision rather than contamination of the evaluation target.

We study both balanced and imbalanced label constructions. In the balanced case, labels are defined by the block-score majority rule with threshold $\tau = 0$ . Since the block-regime bits are symmetric under the synthetic sampling scheme and the number of blocks is odd in both settings, the clean positive-class proportion is theoretically 50%.

In the imbalanced case, labels are generated by the threshold rule in (4) with $\tau = 3$ , making the positive class rarer. For $( d , k , B ) = ( 9 , 3 , 3 )$ , this threshold requires all three block-regime bits to be positive, and hence the clean positive-class proportion is

$$
\mathrm { P r } ( y = + 1 ) = 2 ^ { - 3 } = 1 2 . 5 \% .
$$

For $( d , k , B ) = ( 2 0 , 4 , 5 )$ , the threshold requires at least four of the five block-regime bits to be positive, giving

$$
\mathrm { P r } ( y = + 1 ) = \frac { \binom { 5 } { 4 } + \binom { 5 } { 5 } } { 2 ^ { 5 } } = 1 8 . 7 5 \% .
$$

These proportions refer to the clean labels. Label flips are applied only to the training split, before the subtrain/validation split, while the test labels remain clean. Therefore, when the clean positive-class proportion is $q$ and the flip probability is $p _ { \mathrm { f i i p } }$ , the expected noisy positiveclass proportion in the training and validation portions is

$$
q _ { \mathrm { n o i s y } } = q ( 1 - p _ { \mathrm { H i p } } ) + ( 1 - q ) p _ { \mathrm { H i p } } .
$$

For $p _ { \mathrm { f l i p } } = 0 . 1 0$ , this gives an expected positive-class proportion of 50% in the balanced setting, 20% in the imbalanced $B = 3$ setting, and 25% in the imbalanced $B = 5$ setting.

For every synthetic run, the available training set contains 2000 observations. Of these, 1600 are used for subtraining and 400 for validation during model selection; the selected model is subsequently refitted on all 2000 observations. Evaluation uses an independent test set of

500 observations. Each configuration is evaluated over ten random seeds. For each setting and each random seed, hyperparameters are retuned from scratch on the validation split. Balanced experiments are selected by validation accuracy and reported by test accuracy. Imbalanced experiments use class-balanced sample weights during training, are selected by validation F1, and are reported by test F1.

## 5.1.1 Higher-order scaling protocol

The higher-order scaling study evaluates whether the interaction-aligned quantum geometry remains efective as dimension and interaction order increase. It considers two imbalanced synthetic settings,

$$
( d , k , B ) = ( 6 0 , 6 , 1 0 ) \qquad \mathrm { a n d } \qquad ( d , k , B ) = ( 8 0 , 8 , 1 0 ) .\tag{34}
$$

Holding $B = 1 0$ fixed isolates the increase in the order of each planted interaction from an increase in the number of planted blocks. Both settings use $\epsilon = 0 . 1 5 , p _ { \mathrm { f i p } } = 0$ , and the same threshold $\tau = 3$ . Because the block score must exceed three, at least seven of the ten regime bits must be positive; hence the clean positive-class probability is

$$
\operatorname* { P r } ( y = + 1 ) = 2 ^ { - 1 0 } \sum _ { j = 7 } ^ { 1 0 } { \binom { 1 0 } { j } } = { \frac { 1 7 6 } { 1 0 2 4 } } \approx 1 7 . 1 9 \%\tag{35}
$$

Each scaling setting uses 2000 training observations, including 400 validation observations for model selection, and an independent test set of 500 observations. Results are reported as mean ± standard deviation over ten random seeds. All methods use class-balanced sample weights, are selected by validation F1, and are refitted on the full training sample before test evaluation. To match the other imbalanced synthetic experiments, we report test accuracy and F1, with F1 treated as the primary metric because accuracy can be inflated by majority-class predictions.

The baseline set contains linear, RBF, and Laplacian kernels; polynomial kernels of degrees 2, 4, 6, and 8; and InterFea, which is supplied with the ten planted block products. All method-specific hyperparameters are retuned for the higher-dimensional settings under the common validation protocol described below. The scale-aware search design is documented in Appendix B.

## 5.2 Baselines and model selection

We compare the quantum kernel against the following baselines under the same validation protocol:

• linear kernel;

• RBF kernel;

• Laplacian kernel;

• polynomial kernels with degrees 2, 4, 6, and 8;

• engineered-interaction baseline, denoted InterFea.

The InterFea baseline uses the engineered feature map in (16) and the regularized linear classifier described above. Its regularization parameter is selected on the validation subset. This deliberately strong baseline is given direct access to the planted interaction variables.

All kernel methods use kernel ridge regression with precomputed Gram matrices. For every setting and random seed, kernel and regularization hyperparameters are selected independently on the validation subset. For each polynomial degree, γ, c, and the kernel-ridge regularization parameter are tuned jointly. For the quantum kernel, $\alpha , \beta ,$ , and the kernel-ridge regularization parameter are selected under the same validation protocol. The final selected model is then refitted using the full available training set.

## 5.3 Real-data benchmarks and evaluation protocol

We also consider two fraud-detection benchmarks: the Credit Card Fraud Detection dataset and the IEEE-CIS Fraud Detection dataset, both used through the Fraud Dataset Benchmark (FDB) [Grover et al., 2022]. Both are highly imbalanced binary classification tasks. We use the complete test partition returned by FDB for each benchmark and report its positive-class proportion in the corresponding result subsection. This is particularly important because resampling is applied only to the training subset, while the test distribution is never resampled.

For both real-data benchmarks, we report accuracy and F1, with validation F1 used as the tuning objective because the positive class is rare and accuracy alone can be misleading. All realdata experiments follow a split-first evaluation protocol. The train/test split is performed before any resampling, standardization, feature selection, or block construction. The validation subset is drawn only from the training portion. Standardization parameters, feature-selection scores, and correlation-based groupings used to define blocks are computed only from the training portion and then applied to validation and test data. Resampling is applied exclusively to the model-fitting subset. The test distribution is never resampled, so the reported metrics reflect performance under the original class imbalance.

## 6 Results

## 6.1 Balanced synthetic setting

Tables 2 and 3 report balanced synthetic results as mean ± standard deviation over ten random seeds. In the smaller setting $( d , k , B ) = ( 9 , 3 , 3 )$ , the quantum kernel achieves the best mean accuracy in all four balanced configurations, although the engineered-interaction baseline InterFea remains very close. This indicates that once the relevant block products are exposed directly to a classical model, much of the performance gain can already be recovered.

In the larger setting $( d , k , B ) = ( 2 0 , 4 , 5 )$ , the pattern becomes clearer. Standard classical kernels achieve mean accuracies near the 0.5 random-classification baseline, whereas both the quantum kernel and InterFea achieve substantially higher accuracy. The quantum kernel is best in all four balanced $d = 2 0$ settings, with modest but consistent gains over InterFea. This supports the interpretation that interaction-aware geometry is the main driver of performance.

Table 2: Balanced synthetic results for $( d , k , B ) = ( 9 , 3 , 3 )$ , reported as mean test accuracy ± standard deviation over ten seeds. Columns correspond to $( \epsilon , p _ { \mathrm { f l i p } } )$
<table><tr><td>Model</td><td>(0.05, 0.00)</td><td>(0.05, 0.10)</td><td>(0.15, 0.00)</td><td>(0.15,0.10)</td></tr><tr><td>Linear</td><td> $0 . 5 0 6 \pm 0 . 0 2 5$ </td><td> $0 . 4 9 4 \pm 0 . 0 2 0$ </td><td> $0 . 5 0 2 \pm 0 . 0 2 3$ </td><td> $0 . 5 0 4 \pm 0 . 0 1 9$ </td></tr><tr><td>RBF</td><td> $0 . 4 9 4 \pm 0 . 0 2 7$ </td><td> $0 . 4 9 7 \pm 0 . 0 3 2$ </td><td> $0 . 6 0 8 \pm 0 . 0 2 0$ </td><td> $0 . 5 9 3 \pm 0 . 0 3 8$ </td></tr><tr><td>Laplacian</td><td> $0 . 5 0 8 \pm 0 . 0 3 1$ </td><td> $0 . 5 1 0 \pm 0 . 0 2 4$ </td><td> $0 . 5 4 5 \pm 0 . 0 3 5$ </td><td> $0 . 5 3 8 \pm 0 . 0 2 6$ </td></tr><tr><td>Poly2</td><td> $0 . 4 9 0 \pm 0 . 0 1 6$ </td><td> $0 . 5 0 4 \pm 0 . 0 1 9$ </td><td> $0 . 4 9 0 \pm 0 . 0 1 6$ </td><td> $0 . 5 0 4 \pm 0 . 0 1 1$ </td></tr><tr><td>Poly4</td><td> $0 . 6 7 3 \pm 0 . 0 2 1$ </td><td> $0 . 6 4 1 \pm 0 . 0 2 7$ </td><td> $0 . 6 7 2 \pm 0 . 0 1 8$ </td><td> $0 . 6 4 4 \pm 0 . 0 2 3$ </td></tr><tr><td>Poly6</td><td> $0 . 5 9 3 \pm 0 . 0 2 1$ </td><td> $0 . 5 6 3 \pm 0 . 0 2 7$ </td><td> $0 . 6 3 9 \pm 0 . 0 1 8$ </td><td> $0 . 6 1 9 \pm 0 . 0 2 9$ </td></tr><tr><td>Poly8</td><td> $0 . 5 6 2 \pm 0 . 0 3 8$ </td><td> $0 . 5 4 9 \pm 0 . 0 2 8$ </td><td> $0 . 5 6 6 \pm 0 . 0 4 3$ </td><td> $0 . 5 5 0 \pm 0 . 0 3 8$ </td></tr><tr><td>InterFea</td><td> $0 . 7 6 5 \pm 0 . 0 2 5$ </td><td> $0 . 7 6 0 \pm 0 . 0 2 3$ </td><td> $0 . 7 6 7 \pm 0 . 0 2 2$ </td><td> $0 . 7 6 3 \pm 0 . 0 2 5$ </td></tr><tr><td>Quantum</td><td> $\mathbf { 0 . 7 7 2 \pm 0 . 0 3 4 }$ </td><td> $\mathbf { 0 . 7 6 6 \pm 0 . 0 3 8 }$ </td><td> $\mathbf { 0 . 7 7 2 \pm 0 . 0 3 5 }$ </td><td> $\mathbf { 0 . 7 6 9 \pm 0 . 0 2 7 }$ </td></tr></table>

Table 3: Balanced synthetic results for $( d , k , B ) = ( 2 0 , 4 , 5 )$ , reported as mean test accuracy ± standard deviation over ten seeds. Columns correspond to $( \epsilon , p _ { \mathrm { f l i p } } )$
<table><tr><td>Model</td><td> $( 0 . 0 5 , 0 . 0 0 )$ </td><td>(0.05, 0.10)</td><td>(0.15,0.00)</td><td>(0.15, 0.10)</td></tr><tr><td>Linear</td><td> $0 . 5 2 0 \pm 0 . 0 1 9$ </td><td> $0 . 5 0 9 \pm 0 . 0 1 8$ </td><td> $0 . 5 1 5 \pm 0 . 0 1 5$ </td><td> $0 . 5 1 3 \pm 0 . 0 1 8$ </td></tr><tr><td>RBF</td><td> $0 . 5 0 4 \pm 0 . 0 2 3$ </td><td> $0 . 5 0 4 \pm 0 . 0 0 7$ </td><td> $0 . 5 1 2 \pm 0 . 0 2 0$ </td><td> $0 . 5 0 2 \pm 0 . 0 2 2$ </td></tr><tr><td>Laplacian</td><td> $0 . 5 0 2 \pm 0 . 0 1 1$ </td><td> $0 . 5 1 0 \pm 0 . 0 1 9$ </td><td> $0 . 5 0 8 \pm 0 . 0 1 2$ </td><td> $0 . 5 0 5 \pm 0 . 0 1 5$ </td></tr><tr><td>Poly2</td><td> $0 . 5 1 2 \pm 0 . 0 2 4$ </td><td> $0 . 5 1 4 \pm 0 . 0 1 9$ </td><td> $0 . 5 0 7 \pm 0 . 0 1 8$ </td><td> $0 . 5 0 6 \pm 0 . 0 2 2$ </td></tr><tr><td>Poly4</td><td> $0 . 5 1 6 \pm 0 . 0 2 0$ </td><td> $0 . 4 9 6 \pm 0 . 0 0 5$ </td><td> $0 . 5 0 2 \pm 0 . 0 1 4$ </td><td> $0 . 4 8 7 \pm 0 . 0 2 5$ </td></tr><tr><td>Poly6</td><td> $0 . 4 9 2 \pm 0 . 0 2 9$ </td><td> $0 . 5 0 1 \pm 0 . 0 1 7$ </td><td> $0 . 4 9 9 \pm 0 . 0 2 0$ </td><td> $0 . 4 8 8 \pm 0 . 0 2 6$ </td></tr><tr><td>Poly8</td><td> $0 . 5 0 7 \pm 0 . 0 2 2$ </td><td> $0 . 4 9 4 \pm 0 . 0 2 1$ </td><td> $0 . 5 1 4 \pm 0 . 0 2 2$ </td><td> $0 . 4 9 5 \pm 0 . 0 2 1$ </td></tr><tr><td>InterFea</td><td> $0 . 7 3 3 \pm 0 . 0 2 9$ </td><td> $0 . 7 2 2 \pm 0 . 0 2 2$ </td><td> $0 . 7 3 4 \pm 0 . 0 2 7$ </td><td> $0 . 7 1 0 \pm 0 . 0 2 2$ </td></tr><tr><td>Quantum</td><td> $\mathbf { 0 . 7 3 9 \pm 0 . 0 2 3 }$ </td><td> $\mathbf { 0 . 7 4 2 \pm 0 . 0 2 2 }$ </td><td> $\mathbf { 0 . 7 5 3 \pm 0 . 0 1 6 }$ </td><td> $\mathbf { 0 . 7 3 2 \pm 0 . 0 1 7 }$ </td></tr></table>

## 6.1.1 Learning-curve analysis

To examine sample eficiency, we conduct an additional learning-curve analysis for the balanced $( d , k , B ) = ( 2 0 , 4 , 5 )$ setting with $\epsilon = 0 . 1 5$ and $p _ { \mathrm { f l i p } } = 0$ . The available training sample size is varied over $N \in \{ 2 5 0 , 5 0 0 , 1 0 0 0 , 2 0 0 0 \}$ . For each value of N, 20% of the available observations are used for validation and hyperparameter selection, after which the selected model is refitted using all N observations. Within each seed, the training sets are nested and evaluation uses the same fixed test set of 500 observations. The analysis follows the same ten-seed protocol as the corresponding synthetic experiment.

Figure 1 shows that the quantum kernel has the highest mean accuracy at every investigated sample size. Its mean test accuracy increases from 0.718 at N = 250 to 0.753 at $N = 2 0 0 0$ , while its variability decreases as more training data become available. The engineered-interaction baseline also improves steadily, from 0.666 to 0.734, and therefore narrows the gap with the quantum kernel at the largest sample size. Linear, RBF, Laplacian, and polynomial kernels remain near the 0.5 random-classification baseline throughout the investigated range.

![](images/9e821fb64a26a317b5bf2970f9b0f16e1b1a571cdab9e09ee914011684395436.jpg)  
Figure 1: Learning curves for the balanced $( d , k ) = ( 2 0 , 4 )$ setting. Lines and bands show mean test accuracy and ±1 standard deviation.

The degree comparison in Figure 1(b) shows that polynomial kernels of degrees 2, 4, 6, and 8 remain near the 0.5 random-classification baseline as the training sample size increases. Although the higher-degree polynomial kernels provide a richer generic interaction space, increasing either polynomial degree or sample size alone does not isolate the sparse planted interaction signal over the investigated range. By contrast, both structure-aligned representations improve with additional data, and the quantum kernel maintains the highest mean accuracy at every sample size. This pattern supports the interpretation that alignment with the planted interaction structure, rather than generic nonlinear capacity, improves sample eficiency.

## 6.2 Imbalanced synthetic setting

For the imbalanced synthetic experiments, we use test F1 as the main metric, with model selection based on validation F1. Tables 4 and 5 summarize the results. In both settings, the quantum kernel achieves the best mean F1 in all four configurations, while InterFea remains the closest competitor. This reinforces the conclusion that the gain is tied to interaction-aware similarity rather than generic nonlinear modelling.

Table 4: Imbalanced synthetic results for $( d , k , B ) = ( 9 , 3 , 3 ) \quad$ : mean test accuracy and $\mathrm { F 1 \pm }$ standard deviation over ten seeds. Columns correspond to $( \epsilon , p _ { \mathrm { f i i p } } )$ . Bold and underlined values denote the best and second-best results.
<table><tr><td rowspan="2">Model</td><td colspan="2">(0.05, 0.00)</td><td colspan="2">(0.05,0.10)</td><td colspan="2">(0.15,0.00)</td><td colspan="2">(0.15,0.10)</td></tr><tr><td>Acc.</td><td>F1</td><td>Acc.</td><td>F1</td><td>Acc.</td><td>F1</td><td> $\operatorname { A c c . }$ </td><td>F1</td></tr><tr><td>Linear</td><td>0.506 ± 0.012</td><td> $0 . 2 0 0 \pm 0 . 0 2 7$ </td><td> $0 . 4 9 7 \pm 0 . 0 2 5$ </td><td> $0 . 1 9 4 \pm 0 . 0 3 6$ </td><td> $0 . 4 9 9 \pm 0 . 0 1 4$ </td><td> $0 . 1 9 7 \pm 0 . 0 3 3$ </td><td>0.493 ± 0.021</td><td>0.191 ± 0.040</td></tr><tr><td>RBF</td><td> $0 . 5 8 8 \pm 0 . 0 3 2$ </td><td> $0 . 1 7 9 \pm 0 . 0 4 1$ </td><td> $0 . 5 6 8 \pm 0 . 0 1 9$ </td><td> $0 . 1 9 1 \pm 0 . 0 3 2$ </td><td> $0 . 7 4 8 \pm 0 . 0 5 0$ </td><td> $0 . 2 6 8 \pm 0 . 0 6 5$ </td><td> $0 . 6 0 8 \pm 0 . 0 7 9$ </td><td> $0 . 2 1 5 \pm 0 . 0 3 9$ </td></tr><tr><td>Laplacian</td><td> $\mathbf { 0 . 8 3 4 \pm 0 . 0 2 4 }$ </td><td> $0 . 1 0 1 \pm 0 . 0 5 4$ </td><td> $\mathbf { 0 . 7 7 3 \pm 0 . 0 1 8 }$ </td><td> $0 . 1 6 1 \pm 0 . 0 2 9$ </td><td> $\mathbf { 0 . 8 4 8 \pm 0 . 0 2 1 }$ </td><td> $0 . 1 0 6 \pm 0 . 0 4 0$ </td><td> $\mathbf { 0 . 7 9 1 \pm 0 . 0 1 6 }$ </td><td> $0 . 1 9 2 \pm 0 . 0 3 1$ </td></tr><tr><td>Poly2</td><td> $0 . 5 3 9 \pm 0 . 0 2 8$ </td><td> $0 . 1 8 0 \pm 0 . 0 3 7$ </td><td> $0 . 5 2 2 \pm 0 . 0 1 5$ </td><td> $0 . 1 8 0 \pm 0 . 0 2 6$ </td><td> $0 . 5 3 3 \pm 0 . 0 2 4$ </td><td> $0 . 1 8 1 \pm 0 . 0 3 9$ </td><td> $0 . 5 2 2 \pm 0 . 0 2 1$ </td><td> $0 . 1 8 3 \pm 0 . 0 3 1$ </td></tr><tr><td>Poly4</td><td> $0 . 7 3 9 \pm 0 . 0 3 0$ </td><td> $0 . 3 2 0 \pm 0 . 0 8 2$ </td><td> $0 . 6 6 0 \pm 0 . 0 2 1$ </td><td> $0 . 2 9 0 \pm 0 . 0 3 9$ </td><td> $0 . 7 5 2 \pm 0 . 0 2 6$ </td><td> $0 . 3 5 1 \pm 0 . 0 6 6$ </td><td> $0 . 6 5 4 \pm 0 . 0 4 1$ </td><td> $0 . 2 7 7 \pm 0 . 0 5 1$ </td></tr><tr><td>Poly6</td><td> $0 . 6 7 8 \pm 0 . 0 5 3$ </td><td> $0 . 2 3 7 \pm 0 . 0 4 0$ </td><td> $0 . 6 3 7 \pm 0 . 0 2 8$ </td><td> $0 . 2 4 2 \pm 0 . 0 4 9$ </td><td> $0 . 7 3 5 \pm 0 . 0 3 0$ </td><td> $0 . 2 9 1 \pm 0 . 0 5 6$ </td><td> $0 . 6 6 2 \pm 0 . 0 4 5$ </td><td> $0 . 2 6 3 \pm 0 . 0 6 0$ </td></tr><tr><td>Poly8</td><td> $0 . 6 2 7 \pm 0 . 0 5 2$ </td><td> $0 . 2 2 5 \pm 0 . 0 4 4$ </td><td> $0 . 6 0 9 \pm 0 . 0 5 3$ </td><td> $0 . 2 0 4 \pm 0 . 0 4 8$ </td><td> $0 . 6 7 6 \pm 0 . 0 3 2$ </td><td> $0 . 2 4 5 \pm 0 . 0 4 6$ </td><td> $0 . 5 9 6 \pm 0 . 0 1 0$ </td><td> $0 . 2 1 9 \pm 0 . 0 2 6$ </td></tr><tr><td>InterFea</td><td> $0 . 8 0 1 \pm 0 . 0 3 9$ </td><td> $\underline { { 0 . 4 8 7 \pm 0 . 0 6 7 } }$ </td><td> $0 . 7 1 9 \pm 0 . 0 4 7$ </td><td> $\underline { { 0 . 4 3 8 \pm 0 . 0 7 7 } }$ </td><td> $0 . 8 0 3 \pm 0 . 0 3 6$ </td><td> $\underline { { 0 . 4 9 0 \pm 0 . 0 6 5 } }$ </td><td> $0 . 7 2 2 \pm 0 . 0 4 9$ </td><td> $\underline { { 0 . 4 4 2 } } \pm 0 . 0 7 9$ </td></tr><tr><td>Quantum</td><td> $\underline { { 0 . 8 0 6 \pm 0 . 0 3 5 } }$ </td><td> $\mathbf { 0 . 4 9 4 } \pm \mathbf { 0 . 0 7 0 }$ </td><td> $\underline { { 0 . 7 2 5 \pm 0 . 0 4 0 } }$ </td><td> $\mathbf { 0 . 4 4 6 \pm 0 . 0 6 6 }$ </td><td> $\underline { { 0 . 8 2 7 \pm 0 . 0 3 7 } }$ </td><td> $\mathbf { 0 . 5 4 4 \pm 0 . 0 8 7 }$ </td><td> $\underline { { 0 . 7 3 0 \pm 0 . 0 4 2 } }$ </td><td> $\mathbf { 0 . 4 4 9 \pm 0 . 0 6 7 }$ </td></tr></table>

Table 5: Imbalanced synthetic results for $( d , k , B ) = ( 2 0 , 4 , 5 )$ : mean test accuracy and $\mathrm { F 1 \pm }$ standard deviation over ten seeds. Columns correspond to $( \epsilon , p _ { \mathrm { f i i p } } )$ . Bold and underlined values denote the best and second-best results.
<table><tr><td rowspan="2">Model</td><td colspan="2">(0.05, 0.00)</td><td colspan="2">(0.05,0.10)</td><td colspan="2">(0.15,0.00)</td><td colspan="2">(0.15,0.10)</td></tr><tr><td>Acc.</td><td>F1</td><td>Acc.</td><td>F1</td><td>Acc.</td><td>F1</td><td>Acc.</td><td>F1</td></tr><tr><td>Linear</td><td> $0 . 5 0 2 \pm 0 . 0 3 8$ </td><td> $0 . 2 8 1 \pm 0 . 0 4 4$ </td><td> $0 . 5 1 4 \pm 0 . 0 2 6$ </td><td> $0 . 2 9 6 \pm 0 . 0 3 2$ </td><td> $0 . 5 0 6 \pm 0 . 0 3 6$ </td><td> $0 . 2 8 4 \pm 0 . 0 5 0$ </td><td> $0 . 5 1 1 \pm 0 . 0 2 8$ </td><td> $0 . 2 8 9 \pm 0 . 0 4 1$ </td></tr><tr><td>RBF</td><td> $0 . 6 4 6 \pm 0 . 0 2 3$ </td><td> $0 . 2 0 7 \pm 0 . 0 3 3$ </td><td> $0 . 6 1 9 \pm 0 . 0 1 2$ </td><td> $0 . 2 4 2 \pm 0 . 0 2 3$ </td><td> $0 . 6 7 8 \pm 0 . 0 3 0$ </td><td> $0 . 1 9 3 \pm 0 . 0 4 0$ </td><td> $0 . 6 7 2 \pm 0 . 0 3 0$ </td><td> $0 . 2 2 1 \pm 0 . 0 4 7$ </td></tr><tr><td>Laplacian</td><td> $0 . 7 2 8 \pm 0 . 0 2 3$ </td><td> $0 . 1 8 8 \pm 0 . 0 2 4$ </td><td> $0 . 6 9 1 \pm 0 . 0 1 9$ </td><td> $0 . 2 1 1 \pm 0 . 0 5 9$ </td><td> $0 . 7 3 5 \pm 0 . 0 1 8$ </td><td> $0 . 1 8 3 \pm 0 . 0 5 8$ </td><td> $0 . 7 0 0 \pm 0 . 0 1 6$ </td><td> $0 . 2 0 5 \pm 0 . 0 4 8$ </td></tr><tr><td>Poly2</td><td> $0 . 5 7 3 \pm 0 . 0 1 9$ </td><td> $0 . 2 7 3 \pm 0 . 0 3 4$ </td><td> $0 . 5 4 2 \pm 0 . 0 1 9$ </td><td> $0 . 2 6 7 \pm 0 . 0 1 9$ </td><td> $0 . 5 6 8 \pm 0 . 0 2 6$ </td><td> $0 . 2 5 8 \pm 0 . 0 3 1$ </td><td> $0 . 5 4 7 \pm 0 . 0 0 6$ </td><td> $0 . 2 7 1 \pm 0 . 0 1 9$ </td></tr><tr><td>Poly4</td><td> $0 . 5 9 2 \pm 0 . 0 2 6$ </td><td> $0 . 2 3 7 \pm 0 . 0 3 9$ </td><td> $0 . 5 8 1 \pm 0 . 0 3 4$ </td><td> $0 . 2 3 3 \pm 0 . 0 2 9$ </td><td> $0 . 6 1 8 \pm 0 . 0 2 1$ </td><td> $0 . 2 2 1 \pm 0 . 0 4 1$ </td><td> $0 . 5 8 6 \pm 0 . 0 2 8$ </td><td> $0 . 2 2 9 \pm 0 . 0 3 5$ </td></tr><tr><td>Poly6</td><td> $0 . 5 8 7 \pm 0 . 0 3 6$ </td><td> $0 . 2 4 9 \pm 0 . 0 1 3$ </td><td> $0 . 5 9 1 \pm 0 . 0 3 4$ </td><td> $0 . 2 3 1 \pm 0 . 0 1 8$ </td><td> $0 . 6 4 9 \pm 0 . 0 3 3$ </td><td> $0 . 1 9 7 \pm 0 . 0 4 0$ </td><td> $0 . 5 9 7 \pm 0 . 0 2 8$ </td><td> $0 . 2 1 5 \pm 0 . 0 5 4$ </td></tr><tr><td>Poly8</td><td> $0 . 6 0 2 \pm 0 . 0 5 2$ </td><td> $0 . 2 4 1 \pm 0 . 0 3 2$ </td><td> $0 . 5 8 2 \pm 0 . 0 4 6$ </td><td> $0 . 2 5 2 \pm 0 . 0 3 7$ </td><td> $0 . 6 4 0 \pm 0 . 0 2 5$ </td><td> $0 . 2 1 2 \pm 0 . 0 5 1$ </td><td> $0 . 6 2 9 \pm 0 . 0 1 1$ </td><td> $0 . 2 2 5 \pm 0 . 0 5 4$ </td></tr><tr><td>InterFea</td><td> $\underline { { 0 . 7 5 6 \pm 0 . 0 1 2 } }$ </td><td> $\underline { { 0 . 5 0 8 \pm 0 . 0 3 0 } }$ </td><td> $\underline { { 0 . 7 1 4 \pm 0 . 0 1 8 } }$ </td><td> $\underline { { 0 . 4 9 3 \pm 0 . 0 2 5 } }$ </td><td> $\underline { { 0 . 7 5 3 \pm 0 . 0 1 7 } }$ </td><td> $\underline { { 0 . 5 0 5 \pm 0 . 0 3 6 } }$ </td><td> $\underline { { 0 . 7 1 0 \pm 0 . 0 2 2 } }$ </td><td> $\underline { { 0 . 4 9 1 \pm 0 . 0 2 6 } }$ </td></tr><tr><td>Quantum</td><td> $\mathbf { 0 . 7 6 2 \pm 0 . 0 1 6 }$ </td><td> $\mathbf { 0 . 5 2 5 \pm 0 . 0 3 4 }$ </td><td> $\mathbf { 0 . 7 2 7 \pm 0 . 0 2 0 }$ </td><td> $\mathbf { 0 . 5 1 4 \pm 0 . 0 3 1 }$ </td><td> $\mathbf { 0 . 7 7 0 \pm 0 . 0 1 6 }$ </td><td> $\mathbf { 0 . 5 3 4 \pm 0 . 0 2 9 }$ </td><td> $\mathbf { 0 . 7 3 2 \pm 0 . 0 1 7 }$ </td><td> $\mathbf { 0 . 5 1 6 \pm 0 . 0 2 5 }$ </td></tr></table>

The training-label flip experiments test resilience to noisy supervision. At $p _ { \mathrm { f l i p } } = 0 . 1 0 $ , the quantum kernel retains the highest mean accuracy in every balanced configuration and the highest mean F1 in every imbalanced configuration. Balanced accuracy changes only slightly, from a decrease of 0.021 to an increase of 0.003, whereas imbalanced F1 decreases by 0.011– 0.095, reflecting the greater sensitivity of minority-class learning to label corruption. These results demonstrate comparative resilience to 10% training-label noise within the evaluated settings, rather than general robustness to arbitrary or quantum-device noise.

## 6.2.1 Scaling to higher interaction order and dimension

Table 6 presents the higher-order scaling results. The quantum kernel achieves the highest mean accuracy and F1 in both settings, so its advantage is not confined to either overall classification performance or minority-class recovery. At $( d , k , B ) = ( 6 0 , 6 , 1 0 )$ , it obtains an accuracy of 0.706 and an F1 of 0.412. The corresponding InterFea values are 0.672 and 0.379, while the strongest generic classical results across the two metrics are an accuracy of 0.651 and an F1 of 0.247. At $( d , k , B ) = ( 8 0 , 8 , 1 0 )$ , the quantum kernel retains an accuracy of 0.695 and an F1 of 0.395, compared with 0.631 and 0.320 for InterFea; the strongest generic classical results are 0.661 in accuracy and 0.256 in F1. Relative to the strongest generic result for each metric, the quantum kernel improves mean accuracy by approximately 8% and 5%, and mean F1 by approximately 67% and 54%, at $k = 6$ and $k = 8 ,$ respectively.

Table 6: Higher-order imbalanced scaling results at $\epsilon = 0 . 1 5$ and $p _ { \mathrm { f l i p } } = 0 \colon$ mean test accuracy and $\mathrm { F 1 \pm }$ standard deviation over ten seeds. Bold and underlined values denote the best and second-best results within each column.
<table><tr><td rowspan="2">Model</td><td colspan="2"> $( d , k , B ) = ( 6 0 , 6 , 1 0 )$ </td><td colspan="2"> $( d , k , B ) = ( 8 0 , 8 , 1 0 )$ </td></tr><tr><td> $\operatorname { A c c . }$ </td><td>F1</td><td>Acc.</td><td>F1</td></tr><tr><td>Linear</td><td> $0 . 4 9 4 \pm 0 . 0 2 4$ </td><td> $0 . 2 4 7 \pm 0 . 0 1 7$ </td><td> $0 . 5 0 8 \pm 0 . 0 2 0$ </td><td> $0 . 2 5 2 \pm 0 . 0 4 3$ </td></tr><tr><td>RBF</td><td> $0 . 5 9 7 \pm 0 . 0 4 6$ </td><td> $0 . 2 2 5 \pm 0 . 0 3 4$ </td><td> $0 . 5 8 0 \pm 0 . 0 5 8$ </td><td> $0 . 2 3 7 \pm 0 . 0 4 7$ </td></tr><tr><td>Laplacian</td><td> $0 . 6 5 1 \pm 0 . 0 5 4$ </td><td> $0 . 2 0 0 \pm 0 . 0 4 3$ </td><td> $\underline { { 0 . 6 6 1 \pm 0 . 0 5 8 } }$ </td><td> $0 . 2 2 4 \pm 0 . 0 5 4$ </td></tr><tr><td>Poly2</td><td> $0 . 5 7 4 \pm 0 . 0 3 6$ </td><td> $0 . 2 4 1 \pm 0 . 0 3 7$ </td><td> $0 . 5 5 1 \pm 0 . 0 6 7$ </td><td> $0 . 2 5 6 \pm 0 . 0 4 6$ </td></tr><tr><td>Poly4</td><td> $0 . 6 0 3 \pm 0 . 0 7 9$ </td><td> $0 . 2 2 1 \pm 0 . 0 4 8$ </td><td> $0 . 5 5 8 \pm 0 . 0 5 4$ </td><td> $0 . 2 5 2 \pm 0 . 0 4 6$ </td></tr><tr><td>Poly6</td><td> $0 . 6 0 7 \pm 0 . 0 5 5$ </td><td> $0 . 2 3 1 \pm 0 . 0 4 1$ </td><td> $0 . 5 8 4 \pm 0 . 0 5 4$ </td><td> $0 . 2 4 2 \pm 0 . 0 4 5$ </td></tr><tr><td>Poly8</td><td> $0 . 6 1 2 \pm 0 . 0 5 4$ </td><td> $0 . 2 2 3 \pm 0 . 0 4 1$ </td><td> $0 . 5 8 1 \pm 0 . 0 4 0$ </td><td> $0 . 2 3 6 \pm 0 . 0 3 8$ </td></tr><tr><td>InterFea</td><td> $\underline { { 0 . 6 7 2 } } \pm 0 . 0 1 9$ </td><td> $\underline { { 0 . 3 7 9 } } \pm 0 . 0 4 8$ </td><td> $0 . 6 3 1 \pm 0 . 0 2 6$ </td><td> $0 . 3 2 0 \pm 0 . 0 5 4$ </td></tr><tr><td> $\mathrm { Q u a n t u m }$ </td><td> $\mathbf { 0 . 7 0 6 \pm 0 . 0 1 7 }$ </td><td> $\mathbf { 0 . 4 1 2 \pm 0 . 0 5 6 }$ </td><td> $\mathbf { 0 . 6 9 5 \pm 0 . 0 2 0 }$ </td><td> $\mathbf { 0 . 3 9 5 \pm 0 . 0 4 3 }$ </td></tr></table>

The agreement between the two metrics is important: the quantum kernel ranks first in accuracy and F1 in both scaling settings. Nevertheless, F1 remains the primary selection and interpretation metric because the task is imbalanced. The Laplacian kernel illustrates this distinction: it attains relatively high accuracies of 0.651 and 0.661 while producing F1 scores of only 0.200 and 0.224. Accuracy is therefore reported alongside F1 for consistency and to show that the quantum model’s improved minority-class recovery is not obtained at the expense of overall classification performance.

Because all methods use the same random seeds, the comparison with InterFea can be evaluated pairwise. The quantum-minus-InterFea mean F1 diference is 0.032 at $( d , k ) =$ (60, 6), with a 95% paired t confidence interval of [0.011, 0.053] $( p = 0 . 0 0 7$ , two-sided). At $( d , k ) \ = \ ( 8 0 , 8 )$ , the diference is 0.075, with an interval of [0.032, 0.119] $( p = 0 . 0 0 4 )$ . The fidelity kernel therefore adds predictive value beyond exposing the planted products to a linear classifier, with a larger gap at $k = 8$

Increasing polynomial degree does not close the gap. The degree-matched Poly6 and Poly8 kernels achieve mean F1 scores of only 0.231 and 0.236 in their respective settings. These results indicate that matching the nominal polynomial degree is insuficient: the generic polynomial kernels do not isolate the ten prescribed block interactions, whereas the proposed feature map concentrates its similarity geometry on those interactions. This comparison supports an inductive-bias explanation rather than an explicit-enumeration or computational-complexity claim.

Together with the controlled third- and fourth-order results, the scaling study establishes a persistent advantage across $k \in \{ 3 , 4 , 6 , 8 \}$ . The quantum kernel outperforms InterFea in every balanced and imbalanced synthetic configuration. At $\epsilon = 0 . 1 5$ and $p _ { \mathrm { f l i p } } = 0$ , the mean F1 gaps are approximately 0.054, 0.029, 0.032, and 0.075, respectively. The gap is positive at every order and largest at $k = 8 .$ . Thus, the contribution of the Pauli-string fidelity geometry is not order-specific and remains efective as input dimension and interaction order increase.

## 6.3 Credit Card Fraud Detection benchmark

We first evaluate the models on the Credit Card Fraud Detection dataset from the Fraud Dataset Benchmark, originally associated with European cardholder transactions [Dal Pozzolo et al., 2015b, Grover et al., 2022]. The original dataset contains 284,807 credit-card transactions over a two-day period, with 492 fraudulent transactions and 284,315 non-fraudulent transactions, corresponding to an overall positive-class fraud rate of approximately 0.173%. The original features are anonymized for confidentiality; in our experiment, we use the 28 numerical input features and treat fraud detection as a highly imbalanced binary classification problem.

Following the chronological split used by the Fraud Dataset Benchmark, transactions are ordered by the Time variable. The earliest 80% are assigned to training and the latest 20% to testing. This produces an initial training set of 227,845 observations and a fixed test set of 56,962 observations. The training portion contains 417 fraudulent transactions, while the test set contains 75 fraudulent and 56,887 non-fraudulent transactions, corresponding to a test fraud rate of approximately 0.132%.

Resampling is applied only after the chronological train–test split. In each run, all 417 fraudulent training observations are retained and an equal number of non-fraudulent training observations are randomly sampled, producing a balanced model-fitting set of 834 observations. A validation fraction of 0.2 is used for hyperparameter selection, giving 667 subtraining and 167 validation observations. After model selection, the selected model is refitted on all 834 resampled training observations and evaluated on the complete chronological test set. The resulting feature dimension is d = 28, and the quantum kernel uses a block structure with $k = 4$ and $B = 7$

The chronological train–test division is fixed across all runs. Ten seeds vary the sampled nonfraudulent training observations and the internal validation partition. The reported standard deviations therefore measure sensitivity to training-side resampling and validation selection rather than variation across diferent test sets. Since the positive class remains extremely rare in the test distribution, all models are tuned using validation F1.

The results in Table 7 show that the quantum kernel achieves the highest mean accuracy and F1 among the evaluated methods, while Poly2 is the second-best method on both metrics. The quantum kernel improves mean F1 from 0.348 for Poly2 to 0.472, corresponding to an absolute improvement of 0.124 and a relative improvement of approximately 36%. It also achieves a higher F1 than Poly2 in each of the ten runs. Evaluation on the complete chronological test partition avoids an additional test-set subsampling step and preserves the original temporal test distribution.

Table 7: Credit Card Fraud results on the complete chronological FDB test partition: mean test accuracy and F1 ± standard deviation over ten training-resampling seeds. Bold and underlined values denote the best and second-best results.
<table><tr><td>Model</td><td>Acc.</td><td>F1</td></tr><tr><td>Linear</td><td> $0 . 9 8 3 \pm 0 . 0 0 7$ </td><td> $0 . 1 2 4 \pm 0 . 0 5 0$ </td></tr><tr><td>RBF</td><td> $0 . 9 9 5 \pm 0 . 0 0 1$ </td><td> $0 . 3 3 5 \pm 0 . 0 8 5$ </td></tr><tr><td>Laplacian</td><td> $0 . 9 9 4 \pm 0 . 0 0 0$ </td><td> $0 . 2 7 5 \pm 0 . 0 1 3$ </td></tr><tr><td>Poly2</td><td> $0 . 9 9 6 \pm 0 . 0 0 0$ </td><td> $\underline { { 0 . 3 4 8 \pm 0 . 0 2 1 } }$ </td></tr><tr><td>Poly4</td><td> $0 . 9 8 3 \pm 0 . 0 0 2$ </td><td> $0 . 1 1 9 \pm 0 . 0 1 5$ </td></tr><tr><td>Poly6</td><td> $0 . 9 7 7 \pm 0 . 0 1 1$ </td><td> $0 . 1 0 5 \pm 0 . 0 3 5$ </td></tr><tr><td>Poly8</td><td> $0 . 9 6 8 \pm 0 . 0 1 1$ </td><td> $0 . 0 7 3 \pm 0 . 0 2 2$ </td></tr><tr><td>InterFea</td><td> $0 . 9 9 1 \pm 0 . 0 0 2$ </td><td> $0 . 1 9 7 \pm 0 . 0 3 4$ </td></tr><tr><td>Quantum</td><td> $\mathbf { 0 . 9 9 8 \pm 0 . 0 0 1 }$ </td><td> $\mathbf { 0 . 4 7 2 \pm 0 . 0 7 5 }$ </td></tr></table>

## 6.4 IEEE-CIS Fraud Detection benchmark

We next evaluate the models on the IEEE-CIS Fraud Detection dataset from the Fraud Dataset Benchmark [IEEE Computational Intelligence Society, 2019, Grover et al., 2022]. The original IEEE-CIS dataset is a large-scale transaction-fraud benchmark released through the IEEE-CIS Fraud Detection Kaggle competition and provided by Vesta Corporation. The FDB version uses a standardized chronological train–test split and has a training positive-class ratio of approximately 3.50%, indicating a highly imbalanced classification task.

Our experiment uses the numeric-only FDB version of IEEE-CIS. We select 60 numerical features using correlation-based feature selection and use a block structure with $d = 6 0 , k = 6$ and $B = 1 0$ for the quantum kernel. The feature blocks are constructed by correlation-sorted grouping. We use the complete train and test partitions returned by the FDB loader, without further subsampling of the test set. These partitions contain 561,013 training observations and 29,527 test observations. The test set contains 1,169 fraudulent and 28,358 non-fraudulent transactions, corresponding to a positive-class fraud rate of approximately 3.96%.

Resampling is applied only to the training partition. In each run, up to 3000 fraud samples and the same number of non-fraud samples are used, producing a balanced training subset of size 6000. A validation fraction of 0.2 is used within this resampled training set for hyperparameter selection, giving 4800 subtraining and 1200 validation observations. After model selection, the selected model is refitted on all 6000 training observations and evaluated on the fixed 29,527- observation FDB test set. As in the Credit Card Fraud Detection experiment, all models are tuned using validation F1 and evaluated under the original class imbalance. The experiment is repeated over ten random seeds, and we report mean ± standard deviation.

The results are shown in Table 8. The Laplacian kernel achieves the strongest mean accuracy and F1, with values of 0.831 and 0.267, respectively. The quantum kernel obtains the second-best result on both metrics, with accuracy 0.821 and F1 0.239. Evaluation on the complete FDB test partition preserves the original class imbalance and avoids an additional test-set subsampling step. These results suggest that the proposed quantum feature map remains competitive on this large real tabular fraud benchmark, although its inductive bias appears less closely aligned with this dataset than with the controlled synthetic tasks.

Table 8: IEEE-CIS results on the complete FDB test partition: mean test accuracy and $\mathrm { F 1 \pm }$ standard deviation over ten seeds. Bold and underlined values denote the best and second-best results.
<table><tr><td>Model</td><td>Acc.</td><td>F1</td></tr><tr><td>Linear</td><td> $0 . 8 0 4 \pm 0 . 0 0 5$ </td><td> $0 . 2 1 6 \pm 0 . 0 0 4$ </td></tr><tr><td>RBF</td><td> $0 . 8 1 2 \pm 0 . 0 0 7$ </td><td> $0 . 2 3 1 \pm 0 . 0 0 8$ </td></tr><tr><td>Laplacian</td><td> $\mathbf { 0 . 8 3 1 \pm 0 . 0 0 8 }$ </td><td> $\mathbf { 0 . 2 6 7 \pm 0 . 0 0 7 }$ </td></tr><tr><td>Poly2</td><td> $0 . 8 0 6 \pm 0 . 0 0 8$ </td><td> $0 . 2 2 4 \pm 0 . 0 0 8$ </td></tr><tr><td>Poly4</td><td> $0 . 8 0 9 \pm 0 . 0 0 7$ </td><td> $0 . 2 2 7 \pm 0 . 0 0 4$ </td></tr><tr><td>Poly6</td><td> $0 . 8 1 5 \pm 0 . 0 1 1$ </td><td> $0 . 2 2 9 \pm 0 . 0 0 9$ </td></tr><tr><td>Poly8</td><td> $0 . 8 1 2 \pm 0 . 0 1 0$ </td><td> $0 . 2 2 5 \pm 0 . 0 0 7$ </td></tr><tr><td>InterFea</td><td> $0 . 8 0 3 \pm 0 . 0 0 5$ </td><td> $0 . 2 1 7 \pm 0 . 0 0 6$ </td></tr><tr><td>Quantum</td><td> $0 . 8 2 1 \pm 0 . 0 0 6$ </td><td> $0 . 2 3 9 \pm 0 . 0 0 8$ </td></tr></table>

## 7 Discussion

## 7.1 Mechanism of improvement

The synthetic results indicate that improvement comes primarily from alignment with the planted interaction structure rather than generic nonlinearity. Distance-based kernels remain weak where Euclidean closeness does not imply regime agreement, and degree-matched Poly6 and Poly8 remain substantially below the proposed kernel. InterFea is consistently the strongest classical competitor, confirming that direct access to the block products explains much of the gain. Yet the quantum kernel also outperforms InterFea in every synthetic configuration across $k \in \{ 3 , 4 , 6 , 8 \}$ . The product-of-fidelities geometry therefore adds predictive value beyond a linear representation of the planted variables.

## 7.2 Quantum-specific contribution

Across the staged synthetic design, the Pauli-string fidelity kernel outperforms both generic kernels and InterFea at the evaluated third-, fourth-, sixth-, and eighth-order interactions. In the sixth- and eighth-order scaling settings, it retains mean accuracy near 0.70 and mean F1 near 0.40, ranking first on both metrics. Degree-matched polynomial kernels remain near 0.23 in F1, while InterFea falls from 0.379 to 0.320. These results provide direct scaling evidence that the quantum-induced similarity remains efective as input dimension and interaction order increase.

The quantum-specific contribution consists of two linked elements. The first is a blockfactorized circuit-native realization in which a k-qubit Pauli-Z string directly encodes each planted k-way product. The second is the nonlinear similarity produced by state fidelity and multiplicative composition across blocks. The persistent advantage over generic kernels shows the value of this quantum-induced inductive bias, while the advantage over InterFea shows that the observed improvement is not exhausted by placing the planted products in a classical linear feature vector. The scaling study therefore provides evidence for a predictively useful quantum-induced representation in the statistical and geometric sense.

This quantum-induced predictive advantage is distinct from computational quantum advantage. Under the known disjoint block partition used in these experiments, the fidelity kernel admits an exact classical formula and can be evaluated eficiently. The evidence therefore concerns predictive performance, interaction-aware geometry, and persistence across the investigated scaling settings rather than classical intractability. Demonstrating computational quantum advantage would require a non-factorizing or otherwise classically hard feature map and an explicit complexity separation, neither of which is assumed here.

## 7.3 Real-data interpretation

The real-data experiments add nuance. On Credit Card Fraud, the quantum kernel achieves the highest mean accuracy and F1 on the fixed chronological test set. On IEEE-CIS, the Laplacian kernel performs best, while the quantum kernel is second. This suggests that the usefulness of the quantum feature map depends on how closely the true predictive structure resembles the interaction geometry encoded by the circuit. Real fraud labels are not generated by the synthetic block-product rule, and their structure may involve temporal, categorical, behavioral, or graph-based efects absent from our numerical feature representation. Thus, the real-data results should be interpreted as evidence of competitiveness beyond the controlled synthetic setting, not as universal dominance.

## 7.4 Limitations

The main limitation of the controlled synthetic study is that the data-generating mechanism is deliberately aligned with the proposed feature map. In real data, the relevant interactions and block structure are not known in advance, and the IEEE-CIS result shows that the kernel is not uniformly dominant. Although the complete chronological Credit Card Fraud test partition contains 56,962 observations, it includes only 75 fraudulent transactions. Across both real-data benchmarks, the repetitions share fixed test partitions and vary the training-side resampling and internal validation split. Their standard deviations therefore quantify sensitivity to model fitting and selection, but not uncertainty across alternative temporal test periods. In addition, the reported experiments use exact numerical simulation; they do not quantify finite-shot estimation error, device noise, routing overhead, or calibration efects.

The scaling study demonstrates predictive persistence up to 80 dimensions and eighth-order interactions. The block-factorized construction extends to larger d and k: direct statevector storage costs $O ( B 2 ^ { k } )$ per observation, while the exact overlap in Eq. 27 permits O(Bk)-time evaluation of each kernel entry. Larger values of d and k are therefore accessible without enumerating all degree-k interactions, although dense kernel storage remains quadratic in sample size. Whether the predictive advantage persists at those scales requires further evaluation.

Future work should prioritize evaluation at larger dimensions and interaction orders, datadriven block discovery, trainable or overlapping Pauli-string feature maps, finite-shot and devicenoise analysis, and hardware implementation. Establishing computational quantum advantage would additionally require a non-factorizing or otherwise classically hard feature map, together with explicit complexity and runtime comparisons against strong classical interaction-learning and kernel-approximation methods.

## 8 Conclusion

This study examined quantum-kernel learning through the lens of inductive-bias alignment. We introduced a thin-slab interaction model in which labels are generated by sparse blockwise k-way product signs, creating a controlled setting where Euclidean distance and low-order moments are misaligned with the target structure. We then designed a block-factorized quantum feature map whose Pauli-string phases encode the relevant products directly, yielding a fidelity kernel with an explicit interaction-aware geometry.

The experiments show that quantum-kernel performance depends on feature-map–target alignment. From the controlled third- and fourth-order settings to the 60-dimensional sixthorder and 80-dimensional eighth-order settings, the proposed kernel consistently outperforms generic classical kernels and InterFea. Its advantage over InterFea is positive across all evaluated orders and largest at eighth order, showing that the product-of-fidelities geometry contributes beyond the planted products alone. On the fraud benchmarks, the method achieves the strongest F1 on Credit Card Fraud and ranks second on IEEE-CIS.

These findings establish a quantum-induced predictive advantage in the interaction-driven regimes studied here: high-order Pauli-string phases provide a block-factorized circuit-native representation whose induced fidelity geometry remains efective across the investigated increases in interaction order, while the evaluated generic distance and polynomial kernels do not isolate the relevant sparse interactions. The advantage arises from the quantum featuremap construction and induced similarity geometry, but the present factorized kernel is also exactly classically evaluable. The results therefore demonstrate predictive and representational value that persists across the investigated scaling settings, rather than universal superiority or computational quantum advantage.

Data availability. The synthetic data are generated programmatically. The real benchmark datasets used in this study are publicly available from their respective sources and through the Fraud Dataset Benchmark.

Code availability. Code is available from the authors upon reasonable request.

Author contributions. All authors contributed to the study design, analysis, and writing of the manuscript.

## A Numerical kernel implementation

The mathematical feature map, closed-form block overlap, circuit resources, and experimental settings are presented in the main text. This appendix records only the additional numerical details needed to connect that construction to the reported implementation.

## A.1 Numerical backends

The reference circuit is implemented with PennyLane’s default.qubit simulator using k reusable wires and the gate sequence defined in Sec. 4. The $d = 9$ and $d = 2 0$ synthetic settings, the real-data experiments, and the $( d , k , B ) = ( 6 0 , 6 , 1 0 )$ scaling setting generate their block states through this simulator.

For the $( d , k , B ) = ( 8 0 , 8 , 1 0 )$ experiment only, the block states are evaluated in NumPy from the exact amplitudes in Eq. (24). This change avoids the repeated simulator decomposition of a broadcast eight-qubit Pauli rotation. It changes only the numerical evaluation path, not the feature map or kernel definition. The NumPy statevectors were compared directly with the corresponding PennyLane circuit outputs and were required to agree to a maximum absolute error below $1 0 ^ { - 1 0 }$

## A.2 Batched Gram-matrix construction

For a dataset $\boldsymbol { X } \in \mathbb { R } ^ { n \times d } ,$ , the implementation reshapes the inputs into $B = d / k$ blocks. Let $S _ { b } ( X ) \in \mathbb { C } ^ { n \times 2 ^ { k } }$ contain the block-b statevectors as rows. For two datasets X and Z, the block Gram matrix is evaluated by batched matrix multiplication,

$$
K _ { b } ( X , Z ) = \left| S _ { b } ( X ) S _ { b } ( Z ) ^ { \dagger } \right| ^ { \circ 2 } ,\tag{36}
$$

where $| \cdot | ^ { \circ 2 }$ denotes elementwise squared modulus. The full kernel matrix is accumulated as

$$
K _ { Q } ( X , Z ) = K _ { 1 } ( X , Z ) \circ \cdots \circ K _ { B } ( X , Z ) ,\tag{37}
$$

with ◦ denoting the Hadamard product. The diagonal of the training Gram matrix is set to one to enforce exact unit self-fidelity. No kernel centering or positive-semidefinite correction is applied.

## B Model selection and search design

For each imbalanced scaling run, every method uses the same training–validation partition and the same validation-F1 selection procedure. Each model searches its own method-appropriate hyperparameter grid, the configuration with the highest validation F1 is selected, and that configuration is then refitted on the complete training set before a single evaluation on the independent test set. Thus, “the same validation-F1 procedure” refers to the data partition, selection criterion, and refitting protocol; it does not mean that diferent model families use identical hyperparameters.

The scaling study uses broad multi-scale grids designed for the higher-dimensional settings. Classical kernel bandwidths are adjusted for the change in ambient dimension, ensuring that the $d = 6 0$ and $d = 8 0$ models are evaluated at appropriate distance scales. The quantum interaction-phase range is adjusted for the rapid decrease in the characteristic magnitude of a thin-slab block product as k increases. Regularization and the remaining kernel parameters are tuned jointly over ranges covering multiple scales. These choices give both model families search ranges appropriate to each problem scale. The exact numerical grids are retained in the accompanying experiment code rather than repeated here.

## References

Varun Chandola, Arindam Banerjee, and Vipin Kumar. Anomaly detection: A survey. ACM Computing Surveys, 41(3):1–58, 2009.

Waleed Hilal, S. Andrew Gadsden, and John Yawney. Financial fraud: A review of anomaly detection techniques and recent advances. Expert Systems with Applications, 193:116429, 2022. doi: 10.1016/j.eswa.2021.116429.

Andrea Dal Pozzolo, Giacomo Boracchi, Olivier Caelen, Cesare Alippi, and Gianluca Bontempi. Credit card fraud detection and concept-drift adaptation with delayed supervised information. In Proceedings of the International Joint Conference on Neural Networks, pages 1–8. IEEE, 2015a.

Yvan Lucas and Johannes Jurgovsky. Credit card fraud detection using machine learning: A survey. arXiv preprint arXiv:2010.06479, 2020.

Vojtěch Havlíček, Antonio D. Córcoles, Kristan Temme, Aram W. Harrow, Abhinav Kandala, Jerry M. Chow, and Jay M. Gambetta. Supervised learning with quantum-enhanced feature spaces. Nature, 567(7747):209–212, 2019. doi: 10.1038/s41586-019-0980-2.

Maria Schuld and Nathan Killoran. Quantum machine learning in feature Hilbert spaces. Physical Review Letters, 122(4):040504, 2019. doi: 10.1103/PhysRevLett.122.040504.

Jonas M. Kübler, Simon Buchholz, and Bernhard Schölkopf. The inductive bias of quantum kernels. In Advances in Neural Information Processing Systems, volume 34, pages 12661– 12673, 2021.

Hsin-Yuan Huang, Michael Broughton, Masoud Mohseni, Ryan Babbush, Sergio Boixo, Hartmut Neven, and Jarrod R. McClean. Power of data in quantum machine learning. Nature Communications, 12(1):2631, 2021. doi: 10.1038/s41467-021-22539-9.

Matthias C. Caro, Hsin-Yuan Huang, M. Cerezo, Kunal Sharma, Andrew Sornborger, Lukasz Cincio, and Patrick J. Coles. Generalization in quantum machine learning from few training data. Nature Communications, 13(1):4919, 2022. doi: 10.1038/s41467-022-32550-3.

Daria Sorokina, Rich Caruana, and Mirek Riedewald. Detecting statistical interactions with additive groves of trees. In Proceedings of the 25th International Conference on Machine Learning, pages 1000–1007, 2008.

Jacob Bien, Jonathan Taylor, and Robert Tibshirani. A lasso for hierarchical interactions. The Annals of Statistics, 41(3):1111–1141, 2013.

Stefen Rendle. Factorization machines. In Proceedings of the 2010 IEEE International Conference on Data Mining, pages 995–1000. IEEE, 2010.

Xiangnan He and Tat-Seng Chua. Neural factorization machines for sparse predictive analytics. In Proceedings of the 40th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 355–364, 2017.

Amira Abbas, David Sutter, Christa Zoufal, Aurelien Lucchi, Alessio Figalli, and Stefan Woerner. The power of quantum neural networks. Nature Computational Science, 1(6):403–409, 2021. doi: 10.1038/s43588-021-00084-1.

Supanut Thanasilp, Samson Wang, M. Cerezo, and Zoë Holmes. Exponential concentration in quantum kernel methods. Nature Communications, 15:5200, 2024. doi: 10.1038/ s41467-024-49287-w.

Prince Grover, Julia Xu, Justin Tittelfitz, Anqi Cheng, Zheng Li, Jakub Zablocki, Jianbo Liu, and Hao Zhou. Fraud dataset benchmark and applications. arXiv preprint arXiv:2208.14417, 2022.

Evan Peters, João Caldeira, Alan Ho, Stefan Leichenauer, Masoud Mohseni, Hartmut Neven, Panagiotis Spentzouris, Doug Strain, and Gabriel N. Perdue. Machine learning of high dimensional data on a noisy quantum processor. npj Quantum Information, 7(1):161, 2021. doi: 10.1038/s41534-021-00498-9.

Rowan Pellow-Jarman, Anban Pillay, Ilya Sinayskiy, and Francesco Petruccione. Hybrid genetic optimization for quantum feature map design. Quantum Machine Intelligence, 6(2):45, 2024. doi: 10.1007/s42484-024-00177-w.

Li Xu, Xiao-yu Zhang, Ming Li, and Shu-qian Shen. Quantum classifiers with a trainable kernel. Physical Review Applied, 21(5):054056, 2024. doi: 10.1103/PhysRevApplied.21.054056.

Grace Wahba. Spline Models for Observational Data. SIAM, Philadelphia, 1990.

Mark O. Stitson, Jason Weston, Alex Gammerman, Vladimir Vovk, Vladimir Vapnik, Léon Bottou, Bernhard Schölkopf, and Alexander Smola. Support vector regression with anova decomposition kernels. In Advances in Kernel Methods: Support Vector Learning, pages 285– 292. MIT Press, 1999.

Michael Lim and Trevor Hastie. Learning interactions via hierarchical group-lasso regularization. Journal of Computational and Graphical Statistics, 24(3):627–654, 2015. doi: 10.1080/10618600.2014.938812.

Johannes Jurgovsky, Michael Granitzer, Konstantin Ziegler, Sylvie Calabretto, Pierre-Edouard Portier, Liyun He-Guelton, and Olivier Caelen. Sequence classification for credit-card fraud detection. Expert Systems with Applications, 100:234–245, 2018. doi: 10.1016/j.eswa.2018.01. 037.

Andrea Dal Pozzolo, Giacomo Boracchi, Olivier Caelen, Cesare Alippi, and Gianluca Bontempi. Credit card fraud detection: A realistic modeling and a novel learning strategy. IEEE Transactions on Neural Networks and Learning Systems, 29(8):3784–3797, 2018. doi: 10.1109/TNNLS.2017.2736643.

Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi. Calibrating probability with undersampling for unbalanced classification. In 2015 IEEE Symposium Series on Computational Intelligence, pages 159–166. IEEE, 2015b. doi: 10.1109/SSCI.2015.33.

IEEE Computational Intelligence Society. IEEE-CIS fraud detection, 2019. Kaggle competition dataset.