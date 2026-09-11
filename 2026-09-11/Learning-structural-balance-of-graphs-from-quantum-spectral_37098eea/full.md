# Learning structural balance of graphs from quantum spectral features

Stefano Scali<sup>1,</sup> <sup>∗</sup> and Oleksandr Kyriienko<sup>2</sup>

<sup>1</sup>Department of Physics and Astronomy, University of Exeter, Exeter EX4 4QL, UK

<sup>2</sup>School of Mathematical and Physical Sciences, University of Shefield, Shefield S10 2TN, UK

We develop a quantum approach to spectral feature extraction from the density of states (DOS) of a problem-dependent Hamiltonian, and apply it to machine learning on signed graphs. We propose to embed a signed graph as an Ising model instance with positive and negative interactions, and use the standardized moments of the Ising DOS as features for learning. We show that these moments count signed closed walks, are switching-invariant, and are size-free by construction. As a benchmark, we target learning the frustration index, an NP-hard measure of structural balance that can be labeled exactly at moderate size. At zero field, the models can be sampled classically, allowing the quantum extraction procedure to be certified against exact ground truth. We propose DOS-QPE, a phase estimation on a purified maximally mixed probe, which samples the spectral density with orders of magnitude fewer shots than Hadamard test-based trace sampling and feeds the resulting features directly into classically trained models. On 1.4  10<sup>5</sup> labeled graphs the exact DOS determines the frustration index, and five moments recover it with a mean error of 0.4, well below one sign flip. Beyond zero field, the underlying trace-estimation problem is DQC1-complete, providing access to spectral features for which no eficient classical sampling method is known. Our work opens routes towards quantum applications in social network balance analysis, spin-glass studies, correlation clustering, and protein-interaction networks.

## I. INTRODUCTION

Machine learning models for graphs rely on informative features, and spectral features are among the most efective. The spectra of adjacency and Laplacian operators summarize connectivity in a permutation-invariant way and underpin graph kernels and spectral embeddings [1]. Spectral convolutions built on the graph Laplacian power graph neural networks [2, 3], leading modularity eigenvectors enable community detection [4, 5], and combinatorial Laplacian spectra encode Betti numbers in topological data analysis [6–8]. A recurring object is the density of states (DOS), a global, basis-invariant spectral summary that quantum protocols can estimate directly without diagonalization [7, 9].

One direction that has received much less attention is that of signed graphs. Here every edge carries a positive or negative label, describing systems built on antagonistic pairwise relations: alliances and rivalries in social networks [10–12], correlation structures of financial assets [13], and quenched couplings of spin glasses [14–16]. A signed graph G is balanced when every cycle contains an even number of negative edges, and the frustration index L(G) measures the distance from balance: the minimum number of edge signs one has to flip to reach it. Computing L is NP-hard [15, 17], since it generalizes MAXCUT, which one recovers when all edges are negative. Integer programming solves moderate instances exactly [18], and signed-graph problems have recently drawn attention from the quantum side [19–22]. However, spectral feature sets designed for signed graphs remain scarce [23], even though the sign structure is precisely where quantities like structural balance reside.

Here, we develop a DOS-based quantum feature extraction and learning approach for signed graphs [Fig. 1]. The signed graph is embedded as an Ising Hamiltonian, closely related to Hamiltonian-based encodings for other graph problems [24, 25]. The relevant features are represented by the standardized moments of the Hamiltonian DOS. This construction matches the symmetry of the problem. The moments count signed closed walks through an exact combinatorial identity and are invariant under switching, the gauge transformation that preserves the frustration index. We benchmark the developed approach on the frustration index itself. Although NP-hard in the worst case, it can be labeled exactly at moderate sizes, allowing every prediction to be scored against ground truth. Across a dataset of 1.4 × 10<sup>5</sup> exactly labeled graphs, the exact DOS determines L, with no two graphs sharing a density of states while difering in frustration index. A regressor using five moments recovers L with a mean absolute error below 0.4 sign flips at n = 12 vertices across seventeen classes, while edge and triangle counting incurs twice the error on the same splits. The standardized moments are size-free, allowing the model to transfer across graph sizes and from synthetic ensembles to real-world signed networks.

We highlight that the required features can be extracted natively on quantum hardware, providing access to the DOS without diagonalization. To this end, we simulate two quantum feature extraction routes. The first is a near-term approach that samples the trace of time evolution through Hadamard tests [7]. The second, introduced here as DOS-QPE, targets early fault-tolerant devices and applies quantum phase estimation (QPE) to a purified maximally mixed probe, so that each shot samples directly from the spectral density. Ref. [9] formalizes the corresponding estimation theory and extends the probe beyond maximal mixing. We show that DOS-QPE reaches the noiseless-feature ceiling with orders of magnitude fewer shots than trace sampling and allows its output to be used directly by classically trained models without retraining. At zero field, the pipeline can also be simulated classically. We prove that the DOS is samplable with one edge-list pass per draw (Proposition 1), allowing feature generation at arbitrary graph size while certifying the quantum routes against exact ground truth. For general dynamics, estimation of the normalized trace of the evolution is complete for the one-clean-qubit class (DQC1) [26, 27] and is therefore believed to be classically intractable, while the same feature extraction remains available on quantum hardware. We also show that the choice of encoding is essential: the line-graph (spin-ice) alternative is switching-blind, with a spectrum that carries no sign information (Appendix A).

![](images/21a5cdc19b087674724dd44a6285955737c9a6e110b3764ed11041b14f132c6c.jpg)  
Figure 1. The DOS feature pipeline for signed graphs. A signed graph (positive edges solid, negative edges dashed) is encoded in its native Ising Hamiltonian, $\operatorname { E q . }$ (6). The density of states of H is sampled by either arm of the pipeline: classically, by exact enumeration or Monte Carlo over uniform vertex 2-colorings (Proposition 1), or on quantum hardware, by trace sampling or by the DOS-QPE circuit of Fig. 5. Five spectral moments of the sampled density form the feature vector $\phi ( { \mathcal { G } } )$ of Eq. (3), which is switching-invariant and size-free by construction and feeds a classically trained model; we benchmark on the frustration index $L ( { \mathcal { G } } )$ , for which exact labels are available.

## II. SPECTRAL FEATURES FROM THE DENSITY OF STATES

The density of states (DOS) of a Hamiltonian H acting on a Hilbert space of dimension $D _ { : }$ with spectral decomposition $\begin{array} { r } { H = \sum _ { i = 1 } ^ { D } \omega _ { i } \vert i \rangle \langle i \vert } \end{array}$ , is $\begin{array} { r } { S ( \omega ) = D ^ { - 1 } \sum _ { i = 1 } ^ { D } \delta ( \omega - } \end{array}$ $\omega _ { i } )$ , normalized here to represent a probability density. Its Fourier transform is the normalized trace of the time evolution,

$$
s ( t ) = \frac { 1 } { D } \mathrm { t r } U ( t ) = \int d \omega S ( \omega ) e ^ { - i \omega t } , \qquad U ( t ) = e ^ { - i H t } ,\tag{1}
$$

so estimating $s ( t )$ over time and inverting the transform recovers $S ( \omega )$ This forms the basis of the near-term pipeline described in Sec. VII.

We summarize the DOS through its moments, distinguishing two families. The energy moments are the raw spectral averages

$$
M _ { k } = \mathrm { t r } \big ( H ^ { k } \big ) / D ,\tag{2}
$$

which carry the combinatorial content. For a Hamiltonian supported on a graph, they expand into counts of closed walks (Sec. V A). For learning features, we instead map the spectrum onto the unit interval by an afine rescaling ω 7→ x and use the central moments $\begin{array} { r } { \mu _ { k } \ = \ \sum _ { i } p _ { i } ( x _ { i } - \bar { x } ) ^ { k } } \end{array}$ of the resulting discrete distribution $\{ ( x _ { i } , p _ { i } ) \}$ , where $p _ { i }$ is the DOS weight of level i and $\bar { x } = \textstyle \sum _ { i } p _ { i } x _ { i }$ . We define the standardized moments as $\gamma _ { k } = \mu _ { k } / \mu _ { 2 } ^ { k / 2 }$ , of which $\gamma _ { 3 }$ and $\gamma _ { 4 }$ are the skewness and kurtosis. Given a graph $\mathcal { G }$ and its Hamiltonian encoding, our feature vector is

$$
\phi ( \mathcal { G } ) = \left( \sqrt { \mu _ { 2 } } , \gamma _ { 3 } , \gamma _ { 4 } , \gamma _ { 5 } , \gamma _ { 6 } \right) ,\tag{3}
$$

providing five numbers per graph that capture the width of the rescaled spectrum and its four leading shape parameters. The standardized moments are dimensionless, while rescaling the spectral support makes the full feature set size-free by construction and comparable across graphs of diferent sizes.

The pipeline, summarized in Fig. 1, has four stages: (i) encode the graph in a Hamiltonian whose spectrum carries the structure of interest; (ii) estimate the DOS, classically, by enumeration or, when the encoding permits it, by Monte Carlo (Sec. V B), or on quantum hardware, by trace sampling or direct spectral sampling; (iii) compress the density into the five moments of Eq. (3); (iv) feed them to a standard, classically trained model. Every stage is problem-agnostic except the first: the encoding decides which invariances the features inherit and which structure they can see. The remainder of the paper instantiates this pipeline for signed graphs, using an Ising Hamiltonian whose spectral features inherit the invariance required by the quantities of interest.

## III. FRUSTRATION INDEX

Next, we recall some core concepts from the theory of signed networks. The frustration index, also known as the line index of balance, is a fundamental measure of structural balance in signed graphs [10]. A signed graph is a graph $\mathcal { G } = ( \mathcal { V } , \mathcal { E } , \sigma )$ with $n = | \nu |$ vertices, where $\sigma : \mathcal { E }  \{ + 1 , - 1 \}$ assigns a sign $\sigma _ { e }$ to each edge $e ,$ indicating a positive (friendly) or negative (hostile) relationship. For an edge $e = \{ u , v \}$ , we also write $\sigma _ { u v }$ . The frustration index measures the “distance” of a graph from balance, i.e., from the condition that every cycle contains an even number of negative edges.

Formally, the frustration index $L ( { \mathcal { G } } )$ of the signed graph G is defined as the minimum number of edges whose sign reversal results in a balanced graph,

$$
L ( \mathcal G ) = \operatorname* { m i n } _ { X \subseteq \mathcal { E } } \left( | X | : \mathcal { G } ^ { * } = ( \mathcal { V } , \mathcal { E } , \sigma _ { X } ^ { * } ) \mathrm { ~ i s ~ b a l a n c e d } \right) ,\tag{4}
$$

where $\sigma _ { X } ^ { * }$ agrees with σ on ${ \mathcal { E } } \setminus X$ and flips the sign of every edge in X. Deleting the edges of X instead of flipping them yields the same minimum [17], so $L ( { \mathcal { G } } )$ equally counts the fewest edge deletions that restore balance.

Equivalently, the frustration index can be formulated in terms of vertex 2-colorings as

$$
L ( \mathcal { G } ) = \operatorname* { m i n } _ { f : \mathcal { V } \to \{ 0 , 1 \} } \left| \left\{ \{ u , v \} \in \mathcal { E } : \sigma _ { u v } \neq ( - 1 ) ^ { f ( u ) \oplus f ( v ) } \right\} \right| ,\tag{5}
$$

where we call an edge frustrated by the coloring f when it violates the condition $\sigma _ { u v } = ( - \dot { 1 } ) ^ { f ( u ) \oplus f ( v ) }$ , and satisfied otherwise. This formulation highlights the connection to cut problems: for an all-negative signed graph, Eq. (5) reduces to MAXCUT, and the decision version of the problem is NP-complete [15, 17]. By contrast, deciding balance itself $( L = 0 )$ is solvable in linear time, illustrating the sharp diference between detecting perfect balance and determining the degree of frustration. Equation (5) is also the formulation we use to label our datasets exactly via a standard integer linear program (Appendix B) [18, 28, 29].

## IV. ISING GRAPH EMBEDDING

We map the frustration index problem onto a spinglass Ising Hamiltonian with quenched ±1 couplings supported on the edges of G [14, 16],

$$
H = \sum _ { \{ i , j \} \in \mathcal E } J _ { i j } Z _ { i } Z _ { j } ,\tag{6}
$$

where $J _ { i j } = - \sigma _ { i j }$ and $Z _ { i }$ are Pauli-Z operators. Thus, a positive (friendly) edge gives a ferromagnetic coupling $J _ { i j } = - 1$ , while a negative (hostile) edge gives an antiferromagnetic coupling $J _ { i j } = + 1$ . By construction, $J _ { i j } = 0$ for non-neighboring vertices.

The frustration index is exactly encoded in the groundstate energy of this Hamiltonian. For a computationalbasis state |s⟩, with $s \in \{ \pm 1 \} ^ { | \nu | }$ denoting the corresponding Z eigenvalues and interpreted as a vertex 2-coloring, each satisfied edge contributes −1 and each frustrated edge +1. Hence,

$$
E ( s ) = 2 f _ { \mathcal { G } } ( s ) - | \mathcal { E } | , \qquad E _ { 0 } = 2 L ( \mathcal { G } ) - | \mathcal { E } | ,\tag{7}
$$

where $f _ { \mathcal { G } } ( s )$ counts the edges frustrated by the coloring s. Estimating $L ( { \mathcal { G } } )$ is therefore equivalent to determining the ground-state energy, motivating us to bypass optimization and instead extract information from spectral statistics.

Switching invariance. A switching transformation flips the signs of all edges across a cut $( S , \mathcal { V } \backslash S )$ . It preserves the frustration index and partitions signed graphs into switching-equivalence classes, with G balanced if it is switching-equivalent to the all-positive graph [30]. On the Hamiltonian side, switching by S is implemented by conjugation with the unitary $\begin{array} { r } { U _ { S } = \prod _ { i \in S } X _ { i } } \end{array}$ , with $X _ { i }$ the Pauli-X operator on vertex i: since $\mathbf { \bar { \it X } } _ { i } Z _ { i } \boldsymbol { X } _ { i } = - Z _ { i }$ , the conjugation flips exactly the couplings with one endpoint in $S$

$$
H _ { \sigma ^ { \prime } } = U _ { S } H _ { \sigma } U _ { S } ^ { \dagger } ,\tag{8}
$$

where $H _ { \sigma }$ denotes the Hamiltonian of Eq. (6) built on the sign function σ, and $\sigma ^ { \prime }$ is the switched assignment. The spectrum of H, and hence the DOS and all of its moments, is therefore a switching-class invariant, matching the invariance of $L ( { \mathcal { G } } )$ . The DOS features thus discard the gauge redundancy associated with the choice of representative within a switching class, while retaining sign information that is absent from the unsigned graph spectrum. In particular, a balanced graph has the same DOS as its unsigned counterpart.

We stress that not every graph embedding preserves the relevant sign structure. A natural alternative encodes the signed graph in its line graph, with edges as sites and sign products on adjacent pairs, in the spirit of spinice constructions. We prove in Appendix A that this route is spectrally sign-blind: the signed line graph is switching-equivalent to the line graph of the underlying unsigned graph, so its spectrum carries no information about frustration. By contrast, the direct encoding of Eq. (6) retains the switching-class information on which balance quantities depend, and is therefore the encoding we adopt.

## V. MOMENTS OF THE ISING DENSITY OFSTATES

We first consider the Hamiltonian of Eq. (6) without external fields, referring to this case as zero field. Its spectrum is supported on integers $E \in \{ - | \mathcal { E } | , \dots , | \mathcal { E } | \}$ of fixed parity; a transverse field is introduced later in Sec. V B. The afine rescaling of Sec. II then becomes $x = ( E + | \mathcal { E } | ) / ( 2 | \mathcal { E } | )$ . Since tr $H = 0$ , the rescaled spec trum has mean $\bar { x } = 1 / 2$ and carries no graph-dependent information, and the two moment families are related by $\mu _ { k } = M _ { k } / ( 2 | \mathcal { E } | ) ^ { k }$ for $k \geq 2$

The first feature contains only the edge-count information. As shown below, $M _ { 2 } = | \mathcal { E } |$ , and hence $\sqrt { \mu _ { 2 } } =$ $1 / ( 2 \sqrt { | \mathcal { E } | } )$ exactly. We retain this feature because the classifier requires the edge count, while noting that the moment feature set therefore already contains |E|, which is relevant to the feature-set comparisons below. The remaining four standardized moments $\gamma _ { 3 } , \ldots , \gamma _ { 6 }$ capture the shape of the DOS and form the size-free part of the representation.

## A. Why moments encode frustration

The energy moments of the DOS admit an exact combinatorial expansion in terms of the sign structure of the graph. Expanding the k-th power of Eq. (6) in Eq. (2), with $D = 2 ^ { \vert \nu \vert }$ and $\begin{array} { r } { H = - \sum _ { \{ i , j \} \in \mathcal { E } } \sigma _ { i j } Z _ { i } Z _ { j } } \end{array}$ , the uniform average over computational basis states annihilates every term in which any vertex appears an odd number of times. The surviving terms are ordered k-tuples of edges whose multiset has even degree at every vertex and can therefore be decomposed into closed circuits,

$$
M _ { k } = ( - 1 ) ^ { k } \sum _ { \stackrel { ( e _ { 1 } , \ldots , e _ { k } ) \in { \mathcal { E } } ^ { k } } { \mathrm { e v e n ~ v e r t e x ~ d e g r e e s } } } \prod _ { a = 1 } ^ { k } \sigma _ { e _ { a } } .\tag{9}
$$

The sign product $\Pi _ { a } \sigma _ { e _ { a } }$ of any closed circuit is switching-invariant, consistent with the spectral invariance derived above. The lowest moments are

$$
M _ { 1 } = 0 , \qquad M _ { 2 } = | { \mathcal { E } } | ,\tag{10}
$$

$$
M _ { 3 } = - 6 \left( t _ { + } - t _ { - } \right) = 1 2 t _ { - } - 6 t ,\tag{11}
$$

where $t _ { \pm }$ is the number of balanced/unbalanced triangles and $t = t _ { + } + t _ { - } .$ The third moment, equivalently the skewness of the DOS, counts triangle imbalance di rectly: unbalanced triangles shift the DOS towards positive skew $\mathrm { [ F i g . ~ 2 ] }$ , connecting our lowest odd feature to triangle-based balance measures [22]. Even moments are dominated by sign-independent pairings of edges, such as the $O ( | \mathcal { E } | ^ { 2 } )$ contribution to $M _ { 4 } .$ , with sign-sensitive corrections entering through short signed circuits (squares in $M _ { 4 }$ , pentagons in $M _ { 5 }$ , and so on). Odd moments, by contrast, are purely sign-sensitive. This explains the empirical feature hierarchy observed below: standardized odd moments (skewness, fifth moment) carry the frustration signal, while even moments encode the cycle structure against which it must be normalized. Higher moments probe progressively longer signed circuits, so that the collection $\{ M _ { k } \}$ interpolates between local triangle counts and the global cycle-space information that ultimately determines $L ( { \mathcal { G } } )$ through Eq. (7).

## B. Classical samplability at zero field

For fixed k, Eq. (9) can be evaluated classically in time polynomial in |E|, so the low-order moments used by our classifier are themselves eficiently computable. At zero field, however, a stronger result holds: the full DOS can be sampled directly.

Proposition 1 (Monte-Carlo samplability). Let G be a signed graph with $\begin{array} { l } { \displaystyle { m ~ = ~ | \mathcal { E } | } } \end{array}$ edges and let $p ( E )$ be the normalized DOS of the Hamiltonian in $E q .$ . (6). Then $p ( E )$ is the distribution of the random variable $\begin{array} { r } { E ( s ) \ = \ - \sum _ { \{ u , v \} \in \mathcal { E } } \sigma _ { u v } s _ { u } s _ { v } } \end{array}$ under s drawn uniformly from $\{ \pm 1 \} ^ { | \nu | }$ . Hence, independent samples from the DOS can be generated classically at $O ( m )$ cost per draw, for any graph size.

![](images/b6339bedbb7ecb82f41e8ffddaed8a66cbcdc253ded9c7c0bf6e796b0cce897a.jpg)

![](images/c4ea9eb8cce32b9a11b5e0696d8429cff5890d54c4bba482014adc0f7e37bfe9.jpg)  
Figure 2. (a) Exact DOS of three signed graphs with $n = 1 0$ vertices and $| \mathcal { E } | \ = \ 3 9$ edges, with frustration index $L \ =$ $0 , 5 , 1 0 ,$ Frustration drives the spectrum toward symmetry: the skewness rises from $\gamma _ { 3 } = - 1 . 9 0$ $\mathrm { t o \ - } 0 . 6 4$ to 0.15 and the kurtosis falls from $\gamma _ { 4 } ~ = ~ 8 . 4 2$ to 4.16 to 2.74, moving towards the Gaussian value of 3. The lower edge sits at $E _ { 0 } = - 3 9 , - 2 9 , - 1 9$ , which is $2 L - | \mathcal { E } |$ in each case, as Eq. (7) requires. (b) DOS skewness $\gamma _ { 3 }$ against the frustration index across the $n = 1 0$ ensemble $( 2 \times 1 0 ^ { 4 }$ graphs; gray: individual graphs, markers: median and interquartile range). The trend reflects $\operatorname { E q } .$ . (11), while the spread shows why a single moment is insuficient, motivating the combined features of Eq. (3).

Proof. The Hamiltonian H is diagonal in the computational basis with diagonal entries $E ( s )$ , and the DOS assigns every basis state the same weight $2 ^ { - | \nu | }$ , which is the uniform distribution over colorings. □

Proposition 1 enables a quantum-inspired version of the zero-field pipeline. It scales feature generation to arbitrary graph size, since classical Monte Carlo reproduces the entire zero-field spectral density, rather than only its low moments, at a measured cost near 1 ns per edge per draw. It also gives the quantum pipelines of Sec. VII an exactly certifiable target, a possibility rarely available on classically hard problems. A similar expansion shows where classical computability ends. A transverse field, $\begin{array} { r } { H ( h _ { x } ) = H + h _ { x } \sum _ { i \in \mathcal { V } } X _ { i } } \end{array}$ , keeps every fixed-order moment a closed-form polynomial in h , $h _ { x } ,$ since expanding tr $H ( h _ { x } ) ^ { k }$ in Pauli words annihilates every word carrying an odd number of X factors on any site, with $M _ { 3 }$ in particular exactly independent of $h _ { x }$ . However, the field breaks the diagonal structure on which Proposition 1 rests, so that no classical sampler is known for the density itself or for spectral functionals beyond fixed-order moments, while the hardware routes of Sec. VII remain available.

## VI. LEARNING THE FRUSTRATION INDEX

Datasets. For each $n = 6 , \ldots , 1 2$ , we generate a pool of $2 \times 1 0 ^ { 4 }$ signed graphs from a density-swept Erd˝os– R´enyi $G ( n , p )$ ensemble with $p \in [ 0 . 2 5 , 0 . 7 5 ]$ We use two sign generators: independent random signs, and a near-balanced generator, constructed by randomly flipping signs in a balanced graph followed by a random switching, to populate the otherwise rare low- and high-$L$ classes. An integer linear program labels every graph exactly [Eq. (5)], in milliseconds at $n = 1 2$ . The binding constraint on size is therefore not the labeling but the $2 ^ { | \nu | }$ enumeration behind the exact DOS, which is the constraint that Proposition 1 removes.

Deduplication. Graphs sharing the exact energy histogram are indistinguishable to any method based on the DOS. These include isomorphic and switching-equivalent graphs, the latter produced deliberately by our nearbalanced generator, as well as accidental cospectral pairs. Splitting such graphs across training and test sets would introduce leakage, so we group the pool by exact histogram and retain one representative per group. The reduction is substantial at small sizes but negligible at large ones: the $2 \times 1 0 ^ { 4 }$ graphs reduce to 654 distinct spectra at $n = 6$ , 11 402 at $n = 8$ , and 19 862 at $n = 1 2$ All numbers below are computed on the deduplicated sets, as mean and standard deviation over five stratified $7 0 / 3 0$ splits. We cap each class at 1500 representatives and drop classes with fewer than 50, leaving 4 classes at $n = 6$ and 17 classes $( L = 0 , \ldots , 1 6 )$ at $n = 1 2$ . Further details are given in Appendix B.

An empirical ceiling. Grouping by exact histogram also measures how much the representation can possibly deliver. Across all $1 . 4 \times 1 0 ^ { 5 }$ labeled graphs, every group of graphs sharing a DOS also shares a single value of $L \colon$ no counterexample occurs at any size. On this corpus the exact density of states therefore determines the frustration index, and a Bayes-optimal predictor given the exact DOS would recover L without error. The errors reported below therefore arise from compressing the density into five features and from the learner, rather than from the full DOS representation.

Models and baselines. Using the features of Eq. (3), we train a multinomial logistic regression that treats the frustration index as a class, together with linear and random-forest regressors that treat it as an ordinal quantity [31]. Regression performance is measured by the mean absolute error (MAE) of the rounded prediction. We compare against two groups of baselines. The first consists of counting features available directly from the edge list: edge count, negative-edge count, triangle count, and signed-triangle sum, which underlie degreeand triangle-based balance indices. The second consists of spectral balance measures from the signed-graph literature: $\begin{array} { r } { \beta _ { A } = \sum _ { i } ( \lambda _ { i } ( \Sigma ) - \lambda _ { i } ( G ) ) ^ { 2 } , \gamma _ { A } = \bar { \lambda _ { 1 } } ( G ) - \bar { \lambda _ { 1 } } ( \Sigma ) } \end{array}$ , and the smallest signed-Laplacian eigenvalue [23]. Here, Σ and $G$ are the adjacency matrices of the signed graph and its underlying unsigned graph, respectively, with eigenvalues ordered as $\lambda _ { 1 } \geq \cdots \geq \lambda _ { n }$

(a)  
![](images/8593541168ba6667f379c89a1771d742bab55833acdeeacf8283889e3233bbd7.jpg)

![](images/baebcb479f15faa9320d2f5b99fc8b3782006e4a33623c4515cd24303ce8e800.jpg)  
Figure 3. Estimation quality against graph size for the feature sets of Sec. VI, on the deduplicated evaluation sets. Markers and error bars are the mean and standard deviation over five stratified $7 0 / 3 0$ splits. (a) Multiclass accuracy of the multinomial logistic regression; the number of classes grows from 4 at $n = 6$ to 17 at $n = 1 2$ . The dotted line is the exact-DOS ceiling: every group of graphs sharing a density of states also shares a single $L ,$ so a predictor with access to the full density would be exactly right, and the gap below it is the cost of compressing that density into five numbers. (b) Mean absolute error (MAE) of the rounded random-forest regression. Exactclass accuracy necessarily degrades as the label range grows, while the ordinal error stays below 0.4 sign flips throughout.

Results. Estimation quality against graph size is shown in Fig. 3. At $n = 6$ , the five DOS moments alone drive a linear classifier to $0 . 9 0 8 \pm 0 . 0 1 1$ accuracy, while adding the counting features reaches $0 . 9 7 3 \pm 0 . 0 1 2$ . As n grows, the number of classes more than quadruples, and the moments-only accuracy falls to $0 . 5 5 0 \pm 0 . 0 0 6$ at $n = 1 2$ compared with $0 . 4 3 3 \pm 0 . 0 0 8$ for the counting features on the same splits. The ordinal view remains sharper: the rounded regression MAE is 0.388 ± 0.007 sign flips at $n = 1 2$ with moments alone and $0 . 3 4 0 \pm 0 . 0 0 3$ with all features, while the counting features give $0 . 7 5 9 \pm 0 . 0 1 0$

The DOS moments outperform the counting features at every size, by roughly 0.12 in accuracy and a factor of two in MAE at $n = 1 2$ , showing that the DOS captures signed circuits missed by edge and triangle counts. Classical spectral balance summaries [23] outperform the DOS moments at most sizes $( 0 . 6 1 3 { \pm } 0 . 0 0 3 \mathrm { v s } 0 . 5 8 1 { \pm } 0 . 0 0 6$ for the random forest at $n = 1 2 )$ , showing that frustration leaves a signature in the signed spectrum under diferent spectral summaries, without privileging our particular choice. The two feature families are complementary, and their union $( ^ { 6 } \mathrm { a l l } ^ { \dag } )$ gives the best performance throughout, reaching $0 . 6 8 8 \pm 0 . 0 0 7$ accuracy at $n = 1 2$ This remains below the exact-DOS ceiling of unit accuracy, indicating that a more informative summary of the same density is possible.

Scope: estimation, not optimization. At the sizes studied here, exact answers are cheap. Twenty restarts of a textbook greedy 1-flip descent over vertex 2-colorings recover the exact frustration index on at least 99.8% of the graphs and always provide a certified upper bound. The search cost is comparable to the $2 ^ { | \nu | }$ enumeration required for our features at $n = 6$ and more than an order of magnitude lower at $n = 1 2 \ ( \mathrm { T a b l e } \ \mathrm { I } )$ . The purpose of the spectral representation is therefore not to compete with direct optimization at small sizes. Its value lies in providing five switching-invariant, size-free features that any learning model can consume and that can be extracted on quantum hardware.

The empirical ceiling stated above makes this precise. The exact density of states determines L on every one of the $1 . 4 \times 1 0 ^ { 5 }$ labeled graphs, and five moments of that density recover it far above the reach of edge and triangle counting. That is a statement about what the spectrum of $\operatorname { E q . }$ (6) encodes, rather than how eficiently it can be computed. A quantity defined through an optimization over $\cdot _ { 2 } | \nu |$ colorings leaves a strong signature in a handful of low-order spectral statistics, just as the third moment directly counts signed triangles. This makes the frustration index amenable to estimation from a measured density of states, with the quantum device performing Hamiltonian simulation rather than combinatorial optimization.

Size transfer. The standardized moments of Eq. (3) are size-free (Sec. II), allowing models trained at small n to be applied directly at larger sizes. Training on $n \leq 1 1$ and testing on $n = 1 2$ , the moments-plus-counts model retains $0 . 5 6 6 \pm 0 . 0 0 7$ accuracy and the full feature set $0 . 6 2 1 \pm 0 . 0 0 5$ , whereas the spectral-balance summaries, which are not size-normalized, fall to $0 . 2 1 5 \pm 0 . 0 0 1$ Within this range, the combined feature sets transfer best. Moments alone are less accurate than the counting features $( 0 . 3 5 0 \pm 0 . 0 0 1 \ \mathrm { v s } \ 0 . 4 1 0 \pm 0 . 0 0 6 )$ , while still outperforming them in ordinal error $( 0 . 5 8 8 \pm 0 . 0 0 1$ vs $0 . 7 7 3 { \pm } 0 . 0 0 2 \ \mathrm { M A E } )$ . The behavior changes sharply when the test graphs lie entirely outside the training-size range.

Real networks. As an out-of-distribution test we apply models trained only on the deduplicated synthetic pools $( n \leq 1 2 )$ to two classic signed social networks [32].

<table><tr><td>n</td><td>mean  $\left| L _ { h } - L \right|$ </td><td>exact</td><td> $t ( \mathrm { f e a t u r e s } )$ </td><td>t(search)</td><td>MAE</td></tr><tr><td>6</td><td>0.000</td><td>100.0%</td><td> $4 \mu \mathrm { s }$ </td><td> $6 \mu \mathrm { s }$ </td><td>0.123</td></tr><tr><td>8</td><td>0.002</td><td>99.8%</td><td> $2 7 \mu \mathrm { s }$ </td><td> $1 1 \mu \mathrm { s }$ </td><td>0.147</td></tr><tr><td>10</td><td>0.000</td><td>100.0%</td><td> $5 5 \mu \mathrm { s }$ </td><td> $1 0 \mu \mathrm { s }$ </td><td>0.260</td></tr><tr><td>12</td><td>0.002</td><td>99.8%</td><td> $3 7 1 \mu \mathrm { s }$ </td><td> $1 8 \mu \mathrm { s }$ </td><td>0.388</td></tr></table>

Table I. Multi-restart local search against the learned estimator, on the deduplicated evaluation sets (500 graphs per size, median times, single thread). $L _ { h }$ is the frustration index returned by the search; t(features) is the exact-DOS enumeration the estimator requires; t(search) is the entire heuristic. The last column repeats the estimator’s momentsonly rounded-regression mean absolute error (MAE) from Fig. 3(b).

![](images/05b7bd8f787f881cb71b51e3785e8c0dca6723a92642529010288d0b6635a42a.jpg)  
Figure 4. The Gahuku-Gama alliance network [11] visualized as a signed graph of $n = 1 6$ tribes with 29 positive (rova, solid) and 29 negative (hina, dashed) edges corresponding to alliances and antagonistic relations, respectively. Node color marks the two alliance blocs, and the frustration index is $L ( { \mathcal { G } } ) = 7$

The first one is the Gahuku-Gama alliance network of the New Guinea highlands $( n = 1 6 , \vert \mathcal { E } \vert = 5 8 ,$ , 29 negative) [11], shown in Fig. 4. The second one is the Sampson monastery network $( n = 1 8 )$ [12], which we discuss below. For Gahuku-Gama the exact frustration index, recomputed with our ILP, is $L = 7$ , in agreement with Ref. [17]. Reading only the five moments, and trained on graphs four vertices smaller, the multinomial logistic regression puts $0 . 5 7 4 { \pm } 0 . 0 1 6$ of its probability on the true $L = 7 .$ , and the linear ordinal regressor predicts 7.09.

The feature families behave diferently under this ex trapolation. Adding the raw counting features, despite improving estimation throughout the training-size range, causes the same classifier to assign 0.978 probability to $L = 9$ and essentially none to the correct class. Edge, negative-edge, and triangle counts grow directly with graph size, so a model relying on them encounters feature values outside its training range and becomes confidently incorrect. By contrast, the normalized DOS features retain their interpretation across graph sizes and transfer successfully even though they are not the most accurate representation in distribution.

The Sampson network [12] probes the opposite boundary and delimits the training envelope. Symmetrized by the sign of the net pair rating it has $| \mathcal { E } | = 1 1 0$ and $L = 2 9$ far outside the training label range $L  \leq 1 6$ , forcing the estimator to extrapolate. The classifier saturates at its highest available class, while the random-forest regressor returns a frustration density $L / | \mathcal { E } |$ of 0.119 against the true 0.264, with no indication that the input lies outside the training distribution. The practical lesson concerns training pool design: the synthetic ensemble must cover the density and frustration regime of the target graphs. The size-normalized features make this easier to achieve across diferent graph sizes, since appropriate training data can be generated at any convenient n.

Sampson’s ratings are directed, so the frustration index itself depends on how the directed ratings are converted into a signed undirected graph. We therefore examine several conventions rather than assigning a unique value. Symmetrizing by the sign of the net pair rating gives $L = 2 9$ over 110 edges; keeping zero-net pairs as positive gives $L = 3 6$ over 126 edges; calling an edge negative whenever either direction is negative gives $L = 3 5 ;$ and restricting to the 48 reciprocated pairs gives $L = 8$ The corresponding frustration densities are 0.264, 0.286, 0.278, and 0.167. The normalized quantity is therefore more comparable across conventions than L itself, motivating frustration density as a natural target for realworld networks.

## VII. QUANTUM PIPELINES

The pipeline of Sec. VI used exact spectra. Proposition 1 guarantees that at zero field these spectra are also cheap classically. This makes the frustration index a rare benchmark for quantum protocols: the hardware pipeline can be certified end to end against exact ground truth. Here we perform that certification and ask which quantum route returns the DOS most cheaply, and how much sampling it takes before a downstream classifier stops noticing the diference. We simulate both routes end to end and sample from the exact outcome statistics of each protocol, so that shot noise, the dominant error source, enters exactly.

NISQ route: trace sampling. On noisy intermediatescale quantum (NISQ) hardware, the normalized trace $s ( t ) = \mathrm { t r } U ( t ) / 2 ^ { | \nu | }$ is estimated by Hadamard tests on a maximally mixed input, with Re s and Im s each obtained from a finite number of binary shots [7]. Because the zero-field spectrum is integer, sampling on the uniform time grid $t _ { k } = 2 \pi k / K$ with $ K = 2 \vert \mathcal { E } \vert + 2$ points sufices for the inverse discrete Fourier transform to recover the DOS exactly in the infinite-shot limit; at finite shots the reconstructed DOS is clipped at zero and renormalized before computing Eq. (3).

Early fault-tolerant route: DOS-QPE. We introduce a circuit that samples the DOS directly, which we name DOS-QPE and draw in Fig. 5: textbook quantum phase estimation run not on an eigenstate but on the maximally mixed state $\rho ~ = ~ 1 / 2 ^ { | \nu | }$ , prepared unitarily as half of a maximally entangled state (one Bell pair per system qubit, the purifying register left idle).

An M-qubit phase register applies the controlled powers $U ^ { 2 ^ { k } }$ of $U \ = \ e ^ { - i H \tau }$ , with $\tau = 2 \pi / ( 2 | \mathcal { E } | + 2 )$ , and is read out after an inverse quantum Fourier transform. This τ maps the integer spectrum onto the distinct eigenphases $\varphi _ { j } = ( E _ { j } + | \mathcal { E } | + 1 ) / ( 2 | \mathcal { E } | + 2 )$ , after the trivial relabeling of register outcomes that absorbs the sign and ofset of the raw phase $- E _ { j } \tau / ( 2 \pi )$ mod 1. The choice essentially maximizes τ without aliasing: the spectrum spans $2 | \mathcal { E } | + 1$ integer values, so any period of 2|E| or less would wrap the band edges onto each other; a period of $2 | \mathcal { E } | + 2$ , matching the time grid of the trace route, leaves every eigenvalue with its own phase and one empty slot, and any finer τ would waste register resolution on phases the spectrum never visits. Because the probe weighs all eigenstates equally, the register outcomes $y \in \mathsf { \Gamma } \forall , \mathsf { \Gamma } , \mathsf { \Gamma } . \mathrm { ~ . ~ . ~ } , 2 ^ { \check { M } } \mathrm { ~ - ~ } 1 \}$ are distributed according to the spectrum convolved with the M-bit QPE kernel, $\begin{array} { r } { P ( y ) = \sum _ { i } 2 ^ { - | \mathcal { V } | } F _ { M } ( y ; \varphi _ { j } ) } \end{array}$ , with $F _ { M } ( y ; \varphi ) =$ |sin $( \pi 2 ^ { M } \Delta ) / ( 2 ^ { M }$ sin $\pi \Delta ) | ^ { 2 }$ and $\Delta = \varphi - y / 2 ^ { M }$ Each shot is one flat-weight draw from the DOS, and the features of Eq. (3) are the empirical moments of the sampled phases. Ref. [9] develops the extension beyond maximally mixed probes, where arbitrary purification unitaries select general spectral weights, together with a formalized post-processing and estimation pipeline.

![](images/cc2968a714c25f4c9daa742b27bf23b9aaadd7fc4ddc1029686b82cdc0ffb18b.jpg)  
Figure 5. The DOS-QPE circuit. The lower two registers hold  qubits each. A layer of Hadamards followed by a transversal cnot prepares one Bell pair per system qubit, so that discarding the purifying register, which takes no further gates and is never measured, leaves the system in $\rho = \mathbb { 1 } / 2 ^ { | \nu | }$ Phase estimation then proceeds as usual: the M-qubit register controls the powers $U ^ { 2 ^ { k } }$ of $U = e ^ { - i H \tau }$ with $\tau = 2 \pi / ( 2 | \mathcal { E } | + 2 )$ and an inverse quantum Fourier transform is followed by measurement. Because the probe weighs every eigenstate equally, each outcome $_ y$ is one flat-weight draw from the spectral density, and the empirical moments of the sampled phases are the features of Eq. (3). The register size M sets the resolution of the sampled density and, through Table $\operatorname { I I } ,$ the highest moment order that survives the kernel.

Results. Both routes are run on the same deduplicated graphs the classical estimator was evaluated on, over a full grid of shot budgets and register sizes, with five stratified splits per configuration. Figure 6(b) shows the $n = 8$ comparison. The two parameters play distinct roles: the shot budget governs how quickly each curve rises by controlling the sampling variance, while the register size determines where it saturates by fixing the kernel bias that cannot be removed by increasing the number of shots.

(i) Shot eficiency. DOS-QPE with an M = 10 register reaches $0 . 7 6 4 \pm 0 . 0 \dot { 1 } 6 \mathrm { a t } 1 0 ^ { 4 }$ shots. NISQ trace sampling remains at $0 . 6 1 2 \pm 0 . 0 0 8$ at the same budget, reaches only $0 . 7 4 5 \pm 0 . 0 1 9 \mathrm { a t } 1 0 ^ { 6 }$ shots, and does not attain the QPE performance within the range studied. Matching the DOS-QPE result therefore requires more than two orders of magnitude more measurements for the nearterm route.

(ii) The register sets the plateau. At $n = 8 .$ the exactmoment ceiling is $0 . 7 8 6 \pm 0 . 0 1 2 .$ obtained by feeding the same classifier noiseless moments. This is the best performance attainable by any sampler of Eq. (3) and is distinct from the exact-DOS ceiling of Sec. VI, which no fivenumber compression reaches. A 6-bit register saturates at $0 . 7 2 7 \pm 0 . 0 1 2$ and stops improving beyond $1 0 ^ { 5 }$ shots, leaving an irreducible deficit of about 0.06. Increasing the register to $M = 8$ raises the plateau to $0 . 7 7 7 \pm 0 . 0 1 1$ ， while $M = 1 0$ and $M \ : = \ : 1 2$ reach $0 . 7 8 2 \pm 0 . 0 1 1$ and $0 . 7 8 5 \pm 0 . 0 1 2$ , respectively, indistinguishable from the ceiling. The same pattern holds at $n = 1 2$ , where $M = 1 0$ and $M = 1 2$ both converge to 0.559, against a ceiling of $0 . 5 5 9 \pm 0 . 0 0 8$ . This mirrors Table II: the plateau deficit is set by the moment bias imposed by the kernel, so improving beyond it requires more register qubits rather than more shots.

(a)  
![](images/180ed931db415916357f989a2b6d8268b17b9ba2d9f2085405fabffb7069a67b.jpg)

(b)  
![](images/ed6c7be4f99b7f9794021493751b8eb87b2f55b2a112c9a15e17dbd90c2ed6f7.jpg)  
Figure 6. (a) DOS reconstruction for an $n = 8 , \vert \mathcal { E } \vert = 1 4 ,$ $L = 4$ graph at $1 0 ^ { 3 }$ shots, aggregated on the physical energy grid: exact spectrum (gray), NISQ trace sampling (orange), DOS-QPE with $M = 8$ register bits (blue). The NISQ reconstruction develops spurious tail weight, from clipping the noisy inverse Fourier transform, which biases the highorder moments we use as features. (b) Classifier accuracy at $n = 8$ against shot budget, for models trained and tested on quantum-sampled features. Markers and error bars are the mean and standard deviation over five stratified splits of the deduplicated set; the band is the exact-moment ceiling with its own spread. Two controls act independently: the shot budget sets how fast each curve rises, while the register size M sets the plateau it rises to. A 6-bit register saturates at $0 . 7 2 7 \pm 0 . 0 1 2$ , short of the $0 . 7 8 6 \pm 0 . 0 1 2$ ceiling, whereas $M \geq 1 0$ reaches the ceiling and stays there. NISQ trace sampling has not caught up even at $\mathrm { 1 0 ^ { \circ } }$ shots.

(iii) Train classically, deploy quantumly. Training on exact moments and testing on quantum-sampled features represents the realistic deployment mode, since training data come from classical simulation. This transfer succeeds only when the kernel bias is small enough that the classical and quantum feature distributions remain suficiently aligned. $\mathrm { A t } \ n = 8$ transfer accuracy tracks the self-trained value almost exactly for $M = 1 2$ $( 0 . 7 8 9 \pm 0 . 0 1 4$ against $0 . 7 8 4 \pm 0 . 0 1 6 )$ and for $M = 1 0$ $( 0 . 7 6 8 \pm 0 . 0 1 5$ against $0 . 7 8 1 \pm 0 . 0 1 2 )$ , but falls away at $M = 8 \ ( 0 . 5 9 7 \pm 0 . 0 1 2$ against $0 . 7 7 4 { \scriptstyle \pm 0 . 0 0 9 } )$ and collapses at $M = 6 \ ( 0 . 4 2 2 { \pm } 0 . 0 1 6$ against $0 . 7 2 5 { \pm } 0 . 0 1 1 )$ . The NISQ route reaches $0 . 6 3 8 \pm 0 . 0 1 4$ only at $1 0 ^ { 6 }$ shots, because clipping the noisy inverse transform distorts the feature distribution rather than merely broadening it. A classically trained model can therefore be deployed on quantum hardware using sampled features, but only above a register threshold, which for the five-moment feature set is $M \gtrsim 1 0$

Register resolution sets the usable moment order. The register outcomes follow the DOS convolved with the QPE kernel. As this kernel has $1 / \Delta ^ { 2 }$ tails, it places weight far from each eigenvalue. High-order standardized moments are particularly sensitive to this leakage because they increasingly weight the tails. The resulting error is a bias rather than a variance: it does not vanish with more shots, but only with increasing register size. Table II compares both samplers against the exact moments on 300 graphs at $n = 1 0$ , each using $1 0 ^ { 5 }$ shots.

Classical Monte Carlo, which draws from the true den sity, maintains small relative errors across all moment orders, with the remaining error due entirely to sampling variance. DOS-QPE matches this accuracy through the fourth moment at $M \ = \ 1 0$ and through the sixth at $M = 1 2$ but then degrades rapidly with moment order. By order 20, the estimate is no longer useful for any register considered here. Each additional pair of register qubits reduces the sixth-moment bias by a factor of three to six, while increasing the number of controlled evolutions by a factor of four.

The pipeline of $\operatorname { E q . }$ (3) therefore requires $M \gtrsim 1 0$ , set by the highest retained moment rather than by any need to resolve individual spectral levels. This requirement remains mild because the feature set is low-order. Retain ing higher moments would require progressively larger registers, which we do not pursue here. At the sizes studied, the additional accuracy from such moments is driven by the lower spectral edge, whose relative weight decreases exponentially with graph size, making this ad vantage increasingly dificult for any sampling-based approach to retain. Since cumulants add under convolution and the QPE kernel is known in closed form, its contribution could in principle be subtracted rather than resolved, further relaxing the register requirement. We leave this possibility to future work.

<table><tr><td>moment order</td><td>classical MC</td><td> $M = 8$ </td><td> $M = 1 0$ </td><td> $M = 1 2$ </td></tr><tr><td>3</td><td>0.033</td><td>0.059</td><td>0.034</td><td>0.032</td></tr><tr><td>4</td><td>0.003</td><td>0.066</td><td>0.011</td><td>0.005</td></tr><tr><td>6</td><td>0.007</td><td>0.310</td><td>0.052</td><td>0.017</td></tr><tr><td>12</td><td>0.022</td><td>7.3</td><td>1.2</td><td>0.24</td></tr><tr><td>20</td><td>0.034</td><td>231</td><td>39.8</td><td>5.1</td></tr></table>

Table II. Median relative error $| \hat { \gamma } _ { k } - \gamma _ { k } | / | \gamma _ { k } |$ of standardized moments estimated from $\mathrm { 1 0 ^ { 5 } }$ draws, against the exact values, for 300 graphs at $n = 1 0$ . Classical Monte Carlo draws from the true density and its error is pure variance. DOS-QPE draws from the kernel-convolved density, so its error contains a bias that no shot budget removes.

## VIII. DISCUSSION

Our results are average-case statements about estimation over natural ensembles and two real networks, complementary to the worst-case intractability of exact computation. Typical signed graphs carry substantial spectral information about their frustration index, and the exact DOS determines it across our entire dataset. The five-moment representation remains below this ceiling, while the complementary performance of the spectralbalance summaries of Ref. [23] suggests that the same density admits more informative compressions. These could include spectral functionals beyond moments, such as tail quantiles or low-lying gaps, which DOS-QPE can access with the same underlying circuit. On the learning side, the real-network tests also establish a requirement on training-pool design: the ensemble must cover the density and frustration regime of the target graphs. The size-normalized features make this easier to satisfy because training can be performed at a convenient graph size.

Interestingly, related approaches to physics-native feature extraction have recently proved useful in other learning settings. Directly measured optical spectra can serve as learning features, with the emission spectrum of a single nonlinear mode enabling classical recognition of squeezed quantum states [33], while quantum Fourier features extracted from quantum-encoded flow fields enable the detection of vortical structures [34]. For graph learning, photonic positional embeddings generated by light propagation on synthetic frequency lattices can augment graph neural networks [35], while lattices of polariton condensates can physically encode relational and topological information for subsequent classical learning [36]. Together with the present results, these examples point to a broader role for physics-native and spectral features as representations that can enhance machine learning.

On the hardware side, the register requirement M ≳ 10 is set by kernel bias in the higher-order moments and could be relaxed by the kernel-subtraction postprocessing outlined in Sec. VII. The zero-field benchmark is classically samplable (Proposition 1), which enables exact certification of the quantum extraction routes and provides a controlled boundary between classically accessible and more general dynamics. Beyond zero field, normalized-trace estimation for general dynamics is DQC1-complete [26, 27]. Fixed-order moments remain classically computable even under a transverse field, but spectral functionals beyond them, beginning with the sampled density itself, have no known eficient classical sampler; DOS-QPE provides access to these while retaining the sample-complexity guarantees of Ref. [9].

Another motivation is the native realization of Ising models in neutral atom arrays with Rydberg-mediated interactions. These systems provide a promising hardware route for implementing the Ising dynamics required by both pipelines, with programmable Ising-type Hamiltonians available through analog evolution and gate-

based control [37, 38].

On the application side, the same framework extends naturally beyond balance analysis and social networks. Signed interactions arise in spin glasses [39], regulatory networks [40], and protein-interaction networks [41], and related signed-graph problems include correlation clustering [42]. Across these settings, DOS-based features can provide a compact, switching-invariant summary of global structure that can be extracted on quantum hardware through Hamiltonian simulation.

Finally, the lemma in Appendix A adds a broader design principle: spectral methods for signed structures are not encoding-agnostic. The line-graph route, natural from a spin ice perspective, provably discards all sign information, whereas the direct encoding of Eq. (6) preserves the switching-class information on which balance quantities depend. The efectiveness of the resulting representation therefore comes from matching the Hamiltonian symmetry to the invariances of the learning problem. DOS-based spectral features, already useful for topology and community structure [5, 7, 8], extend naturally to signed graphs and provide a route from quantum spectral estimation to classical learning without requiring the quantum device itself to solve the underlying combinatorial optimization problem.

## IX. CONCLUSIONS

We developed a quantum approach to spectral feature extraction from the density of states of a problem dependent Hamiltonian and applied it to signed-graph learning. Signed graphs are embedded as Ising Hamiltonians whose energy moments count signed closed walks and yield switching-invariant spectral features. Using the frustration index as a benchmark, we showed that the exact DOS determines the target across $1 . 4 \times 1 0 ^ { 5 }$ labeled graphs, while five moments recover it with a mean error of 0.4 sign flips. We introduced DOS-QPE, which samples the spectral density with orders of magnitude fewer shots than Hadamard-test trace sampling and supports direct use of classically trained models. At zero field, classical DOS sampling enables exact certification. Beyond zero field, the underlying trace-estimation problem is DQC1-complete, providing access to spectral features for which no eficient classical sampler is known.

## ACKNOWLEDGMENTS

The authors would like to thank Shaheen Acheche and Louis-Paul Henry from Pasqal for useful discussions. O. K. acknowledges the support from UK EPSRC award under the Agreement No. EP/Z53318X/1 (QCi3 Hub).

## Appendix A: Line graphs cannot hear frustration

Given a signed graph $\mathcal { G } = ( \nu , \mathcal { E } , \sigma )$ with underlying unsigned graph $G ,$ define its signed line graph $\Lambda ( \stackrel { . } { \mathcal { G } } )$ as the graph whose vertices are the edges E, with $e \sim f$ whenever e and f share an endpoint, and with edge signs given by the product $\sigma _ { e f } = \sigma _ { e } \sigma _ { f }$ . This is the natural “spin-$\operatorname { i c e } ^ { \mathfrak { N } }$ encoding in which frustrated plaquettes of $\mathcal { G }$ would be represented by interactions among edge variables.

Lemma 1. $\Lambda ( \mathcal G )$ is switching-equivalent to $\Lambda ( G )$ , the allpositive line graph of the underlying graph: with $W =$ diag $( \sigma _ { e } ) _ { e \in \mathcal E }$

$$
A _ { \Lambda ( \mathcal { G } ) } = W A _ { \Lambda ( G ) } W .\tag{A1}
$$

Consequently, the adjacency and signed-Laplacian spectra of $\Lambda ( \mathcal G )$ , and the spectrum of any Ising Hamiltonian of the form of Eq. (6) built on $\Lambda ( \mathcal G )$ , are independent of the sign function σ. No spectral functional of $\Lambda ( \mathcal G )$ can determine $L ( { \mathcal { G } } )$

Proof. Entrywise, $[ A _ { \Lambda ( \mathcal { G } ) } ] _ { e f } = \sigma _ { e } \sigma _ { f } [ A _ { \Lambda ( G ) } ] _ { e f }$ by the definition of the sign product, which is Eq. (A1). Since W is diagonal with entries ±1, it is orthogonal, so the adjacency spectra coincide. The signed Laplacian (the degree matrix minus the adjacency) conjugates the same way because the degree matrix is diagonal. For the Ising Hamiltonian, conjugation by $\begin{array} { r } { U = \prod _ { e : \sigma _ { e } = - 1 } X _ { e } } \end{array}$ maps $H _ { \Lambda ( { \mathcal { G } } ) }$ to $H _ { \Lambda ( G ) }$ , cf. the switching argument of the main text. Finally, the frustration index is not constant on sign configurations of a fixed underlying graph (the all-positive and all-negative triangles have $L = 0$ and $L = 1$ but identical Λ spectra), so no function of the $\Lambda ( \mathcal G )$ spectrum can compute it. □

The lemma explains a dead end that, to our knowledge, has not been recorded: any pipeline that extracts spectral statistics (moments, DOS, gaps, spectral measures of balance) from this line-graph encoding returns exactly the same answer for every signing of a given graph, and cannot even separate balance from maximal frustration. The information loss occurs at the embedding stage, before any classification method is applied. The statement concerns the product-sign line graph defined above; sign conventions based on bidirected incidences [30] define different objects and are not covered by (nor needed for) our argument. We verified Eq. (A1) as an exact integer identity, along with the coincidence of adjacency and Ising spectra and the variation of L across signings, on 320 random signed graphs with n = 5–7 (code accompanies the paper).

[1] Nils M. Kriege, Fredrik D. Johansson, and Christopher Morris. A survey on graph kernels. Applied Network Science, 5:6, 2020. doi:10.1007/s41109-019-0195-3.

[2] Micha¨el Deferrard, Xavier Bresson, and Pierre Van-

## Appendix B: Datasets, labeling, and models

Exact labeling. We label every graph by the standard ILP formulation of Eq. (5) [18]: binary color variables $x _ { v } ,$ frustration indicators y , constraints $y _ { e } \ge \pm ( x _ { u } - x _ { v } )$ for positive edges and $y _ { e } \geq x _ { u } + x _ { v } - 1 , y _ { e } \geq 1 - x _ { u } - x _ { v }$ for negative edges, objective min $\sum _ { \boldsymbol { e } } y _ { e }$ , with $x _ { 1 } ~ = ~ 0$ breaking the global flip symmetry; solved with HiGHS via JuMP [28, 29]. Against brute-force enumeration on 80 graphs $( n = 4 \mathrm { ~ t o ~ } 7 )$ the ILP agrees exactly and runs about 360 times faster already at $n = 7$ . We checked every returned flip set by applying it and confirming the resulting graph has no unbalanced cycle, and every instance in the corpus closed to proven optimality, so no label in this paper rests on a truncated search.

Pools. Per size $n \in \{ 6 , \ldots , 1 2 \} \colon 2 \times 1 0 ^ { 4 }$ graphs made unique at generation time by hashing the edge list and sign vector, generators as in Sec. VI, seeds and generator parameters stored with the data. That hash removes literal repeats only; the far stronger deduplication by exact energy histogram described in Sec. VI, which is what the evaluation uses, is applied afterwards and reduces these pools to between 654 and 19 862 distinct spectra. The zero-field spectrum is computed directly as the $2 ^ { n }$ coloring energies of Eq. (7), without constructing any operator.

Models. Multinomial logistic regression (no regularization, standardized inputs), random-forest classifier and regressor (300 trees), and linear regression [31]. Every reported number is the mean and standard deviation over five stratified 70/30 splits of the deduplicated set, with a per-class cap of 1500 and a minimum class size of 50 applied after deduplication. Reported MAEs score rounded regression outputs against the integer label.

Quantum simulations. Both pipelines are sampled from exact outcome statistics: binomial shot noise on $\operatorname { R e } s ( t _ { k } )$ , Im $s ( t _ { k } )$ for the NISQ route, and multinomial sampling of the kernel-convolved register distribution for DOS-QPE. Graphs are the deduplicated representatives of Sec. VI, stratified by L with a cap of 400 per class, giving 622, 2883 and 6283 graphs at $n = 6 ,$ , 8 and 12. The grid is rectangular: every shot budget in $\{ 1 0 ^ { 2 } , \ldots , 1 0 ^ { 6 } \}$ for NISQ, and every pair of $M \in \{ \breve { 6 } , 8 , 1 0 , \breve { 1 } 2 \}$ with the same budgets for DOS-QPE. The kernel-convolved distribution is built once per graph and register size, then sampled at each budget. Simulator moments converge to the exact features in the large-shot limit (deviations ${ \sim } 3 \times 1 0 ^ { - 2 }$ in $\gamma _ { 6 }$ at $1 0 ^ { 7 }$ shots).

[3] Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations, 2017.

[4] M. E. J. Newman. Modularity and community structure in networks. Proceedings of the National Academy of Sciences, 103(23):8577–8582, 2006. doi:10.1073/pnas.0601602103.

[5] Chukwudubem Umeano, Stefano Scali, and Oleksandr Kyriienko. Quantum community detection via deterministic elimination. Physical Review A, 112(5):052422, 2025. doi:10.1103/s7cm-c9qy.

[6] Gunnar Carlsson. Topology and data. Bulletin of the American Mathematical Society, 46(2):255–308, 2009. doi:10.1090/S0273-0979-09-01249-X.

[7] Stefano Scali, Chukwudubem Umeano, and Oleksandr Kyriienko. Quantum topological data analysis via the estimation of the density of states. Physical Review A, 110(4):042616, October 2024. ISSN 2469-9934. doi:10.1103/physreva.110.042616. URL http://dx. doi.org/10.1103/PhysRevA.110.042616.

[8] Stefano Scali, Chukwudubem Umeano, and Oleksandr Kyriienko. The topology of data hides in quantum thermal states. APL Quantum, 1(3):036106, 2024. doi:10.1063/5.0209201.

[9] Stefano Scali, Josh Kirsopp, Antonio M´arquez Romero, and Micha l Krompiec. Purified phase estimation samples spectra eficiently, 2025. URL https://arxiv.org/abs/ 2510.14744.

[10] Frank Harary. On the notion of balance of a signed graph. Michigan Mathematical Journal, 2(2):143–146, 1953. doi:10.1307/mmj/1028989917.

[11] Kenneth E. Read. Cultures of the Central Highlands, New Guinea. Southwestern Journal of Anthropology, 10 (1):1–43, 1954. doi:10.1086/soutjanth.10.1.3629074.

[12] Samuel F. Sampson. A novitiate in a period of change: An experimental and case study of social relationships. PhD thesis, Cornell University, 1968.

[13] Shivam Sharma, Supreeth Mysore Venkatesh, and Pushkin Kachroo. Toward quantum utility in finance: A robust data-driven algorithm for asset clustering. In Fr´ed´eric Barbaresco and Fran¸cois Gerin, editors, Quantum Engineering Sciences and Technologies for Industry and Services, pages 14–22, Cham, 2026. Springer Nature Switzerland. ISBN 978-3-032-13855-2.

[14] P.W. Anderson. Localisation theory and the Cu– Mn problem: Spin glasses. Materials Research Bulletin, 5(8):549–554, August 1970. ISSN 0025-5408. doi:10.1016/0025-5408(70)90096-6. URL http://dx. doi.org/10.1016/0025-5408(70)90096-6.

[15] Francisco Barahona. On the computational complexity of Ising spin glass models. Journal of Physics A: Mathematical and General, 15(10):3241–3253, 1982. doi:10.1088/0305-4470/15/10/028.

[16] Ada Altieri and Marco Baity-Jesi. An introduction to the theory of spin glasses. In Tapash Chakraborty, editor, Encyclopedia of Condensed Matter Physics, pages 361–370. Elsevier, 2 edition, 2024. ISBN 9780323914086. doi:10.1016/B978-0-323-90800-9.00249-3.

[17] Samin Aref and Mark C Wilson. Balance and frustration in signed networks. Journal of Complex Networks, 7(2): 163–189, 2019. doi:10.1093/comnet/cny015.

[18] Samin Aref, Andrew J. Mason, and Mark C. Wilson. A modeling and computational study of the frustration index in signed networks. Networks, 75(1):95–110, 2020.

doi:10.1002/net.21907.

[19] Etsuo Segawa and Yusuke Yoshie. Quantum search of matching on signed graphs. Quantum Information Processing, 20(5):182, 2021. doi:10.1007/s11128-021-03089-x.

[20] Kuo-Chin Chen, Simon Apers, and Min-Hsiu Hsieh. (quantum) complexity of testing signed graph clusterability. In 19th Conference on the Theory of Quantum Computation, Communication and Cryptography (TQC 2024), volume 310 of Leibniz International Proceedings in Informatics (LIPIcs), pages 8:1–8:16. Schloss Dagstuhl – Leibniz-Zentrum f¨ur Informatik, 2024. doi:10.4230/LIPIcs.TQC.2024.8.

[21] Massimiliano Incudini, Casper Gyurik, Riccardo Molteni, and Vedran Dunjko. Testing the presence of balanced and bipartite components in a sparse graph is QMA<sub>1</sub>-hard, 2024. URL https://arxiv.org/abs/2412.14932.

[22] Steven Kordonowy, Bibhas Adhikari, and Hannes Leipold. A perfectly distributable quantum-classical algorithm for estimating triangular balance in a signed edge stream, 2026. URL https://arxiv.org/abs/2603. 16029.

[23] K. Shahul Hameed, K. Biju, and Zoran Stani´c. Spectral measures of balance for signed graphs. Australasian Journal of Combinatorics, 93(1):198–210, 2025. URL https://ajc.maths.uq.edu.au/v93.p198.

[24] Itay Hen and A. P. Young. Solving the graphisomorphism problem with a quantum annealer. Physical Review A, 86:042310, 2012. doi:10.1103/PhysRevA.86.042310.

[25] Frank Gaitan and Lane Clark. Graph isomorphism and adiabatic quantum computing. Physical Review A, 89: 022342, 2014. doi:10.1103/PhysRevA.89.022342.

[26] E. Knill and R. Laflamme. Power of one bit of quantum information. Physical Review Letters, 81(25):5672–5675, 1998. doi:10.1103/PhysRevLett.81.5672.

[27] Peter W. Shor and Stephen P. Jordan. Estimating Jones polynomials is a complete problem for one clean qubit. Quantum Information and Computation, 8(8–9): 681–714, 2008. doi:10.26421/QIC8.8-9-1.

[28] Miles Lubin, Oscar Dowson, Joaquim Dias Garcia, Joey Huchette, Benoˆıt Legat, and Juan Pablo Vielma. JuMP 1.0: recent improvements to a modeling language for mathematical optimization. Mathematical Programming Computation, 15:581–589, 2023. doi:10.1007/s12532-023-00239-3.

[29] Qi Huangfu and J. A. J. Hall. Parallelizing the dual revised simplex method. Mathematical Programming Computation, 10(1):119–142, 2018. doi:10.1007/s12532-017-0130-5.

[30] Thomas Zaslavsky. Signed graphs. Discrete Applied Mathematics, 4(1):47–74, 1982. doi:10.1016/0166-218X(82)90033-6.

[31] Fabian Pedregosa et al. Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12: 2825–2830, 2011.

[32] J´erˆome Kunegis. KONECT: the Koblenz network collection. In Proceedings of the 22nd International Conference on World Wide Web (Companion), pages 1343– 1350, 2013. doi:10.1145/2487788.2488173.

[33] Wouter Verstraelen, Stanis law Swierczewski, An-<sup>´</sup> drzej Opala, Andrew Haky, Matteo Gadani, Huawen Xu, Oleksandr Kyriienko, Micha l Matuszewski, Alberto Bramati, and Timothy C. H. Liew. Spec-

troscopy on a single nonlinear mode recognizes quantum states. ACS Photonics, 13(9):2397–2407, 2026. doi:10.1021/acsphotonics.5c02750.

[34] Chelsea A. Williams, Annie E. Paine, Antonio A. Gentile, Daniel Berger, and Oleksandr Kyriienko. Vortex detection from quantum data. Physical Review A, 112 (6):062409, 2025. doi:10.1103/mn3x-8ygh.

[35] Yuan Wang and Oleksandr Kyriienko. Photonicsenhanced graph convolutional networks, 2025. URL https://arxiv.org/abs/2512.15549.

[36] Yuan Wang, Stefano Scali, and Oleksandr Kyriienko. Polaritonic machine learning for graph-based data analysis. Physical Review E, 114(2):025305, 2026. doi:10.1103/fgy1-n4vp.

[37] Lo¨ıc Henriet, Lucas Beguin, Adrien Signoles, Thierry Lahaye, Antoine Browaeys, Georges-Olivier Reymond, and Christophe Jurczak. Quantum computing with neutral atoms. Quantum, 4:327, 2020. doi:10.22331/q-2020-09-21-327.

[38] A. Andrea Gentile, Shaheen Acheche, Atiyo Ghosh, Oleksandr Kyriienko, and Louis-Paul Henry. Hardwareinformed applications, application-informed hardware. IEEE Nanotechnology Magazine, 19(5):33–43, 2025.

doi:10.1109/MNANO.2025.3585999.

[39] S. F. Edwards and P. W. Anderson. Theory of spin glasses. Journal of Physics F: Metal Physics, 5(5):965– 974, 1975. doi:10.1088/0305-4608/5/5/017.

[40] Elisabeth Remy and Paul Ruet. From minimal signed circuits to the dynamics of Boolean regulatory networks. Bioinformatics, 24(16):i220–i226, 2008. doi:10.1093/bioinformatics/btn287.

[41] Livia Perfetto, Leonardo Briganti, Alberto Calderone, Andrea Cerquone Perpetuini, Marta Iannuccelli, Francesca Langone, Luana Licata, Milica Marinkovic, Anna Mattioni, Theodora Pavlidou, Daniele Peluso, Lucia Lisa Petrilli, Stefano Pirr\`o, Daniela Posca, Elena Santonico, Alessandra Silvestri, Filomena Spada, Luisa Castagnoli, and Gianni Cesareni. SIGNOR: a database of causal relationships between biological entities. Nucleic Acids Research, 44(D1):D548–D554, 01 2016. ISSN 0305-1048. doi:10.1093/nar/gkv1048.

[42] Nikhil Bansal, Avrim Blum, and Shuchi Chawla. Correlation clustering. Machine Learning, 56(1–3):89–113, 2004. doi:10.1023/B:MACH.0000033116.57574.95.