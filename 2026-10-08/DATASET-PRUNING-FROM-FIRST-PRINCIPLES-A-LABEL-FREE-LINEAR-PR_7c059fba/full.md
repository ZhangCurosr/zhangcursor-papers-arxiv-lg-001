# DATASET PRUNING FROM FIRST PRINCIPLES: A LABEL-FREE LINEAR PROGRAMMING APPROACH

Rodrigo Schuller & Francisco Ganacim Instituto de Matematica Pura e Aplicada (IMPA)´ Rio de Janeiro, Brazil rodrigo.loro,ganacim @impa.br

## ABSTRACT

Dataset pruning reduces a large training set to a representative subset while preserving model performance. Existing geometry-based methods typically assume that nearby points in embedding space share similar properties. Rather than imposing this assumption, we derive geometric selection criteria by reformulating unbiased subset selection as a variance minimization problem. Unbiasedness ensures that unweighted subset averages recover full-dataset averages in expectation, including losses and gradients at fixed model parameters. Specifically, we characterize a family of unbiased subset selection algorithms as a high-dimensional polytope. In this context, minimizing the expected sampling variance is a linear objective. Differences in sampling variance, averaged over rigid motions, admit closed-form pairwise expressions. Because the polytope has high dimension, directly applying standard linear programming is impractical. We instead use these expressions to construct an efficient vertex walk that optimizes an approximation of the variance objective while preserving unbiasedness, yielding a method that requires neither labels nor model training during selection. Across CIFAR-10, MNIST, and CelebA benchmarks, our method matches or exceeds uniform sampling in mean test accuracy at every evaluated budget and outperforms competing geometric methods in several settings, particularly at small selection budgets. Beyond dataset pruning, the same framework reduces stochastic-gradient variance by increasing diversity within mini-batches while keeping the batch size unchanged.

## 1 INTRODUCTION

Training large deep learning models on abundant data raises a basic question: given a fixed budget, which samples should we train on? The budget may be compute, as in self-supervised pretraining, where scaling laws favor larger models and datasets (Kaplan et al., 2020; Hoffmann et al., 2022) at substantial computational cost (Strubell et al., 2019), and where a well-chosen subset can outperform a larger random one (Sorscher et al., 2022). It may also be annotation, as in remote sensing, where unlabeled archives are abundant but expert labels are expensive (Tuia et al., 2011). Dataset pruning, also known as coreset selection, chooses a small, representative subset of the data. Addressing both cases requires selection that is label-free, so that annotation can be deferred until after selection, and training-free, so that selection does not itself consume the compute it is meant to save. We study methods that satisfy both requirements.

Without labels or training signals, selection must rely on the geometry of the embedding space alone. Existing geometry-based methods typically assume that nearby points in embedding space share similar properties (Moser et al., 2026). This assumption is a heuristic rather than a derived principle: it treats nearby points as redundant, which biases the distribution of the selected subset and can push accuracy below uniform random sampling (Guo et al., 2022). We drop this assumption and instead derive a geometric selection criterion from the principle of no preferred orientation.

We seek a subset whose unweighted averages are unbiased, minimum-variance estimates of fulldataset averages of an unknown downstream function, such as a loss or gradient. Since we know nothing about how downstream functions are aligned with the data, we assume no preferred orientation. Under this assumption, we average the mean squared estimation error over all rigid motions of the function, or equivalently of the data (Figure 1, left). For unbiased selections, this averaged error is exactly the expected sampling variance. On toroidal and spherical embedding spaces, this averaging yields closed-form expressions previously studied in the context of blue-noise sampling for computer graphics (Pilleboue et al., 2015). We then use these formulas to build a novel unbiased coreset selection method.

![](images/a854143a35ccbfc63a03414add135c43f4998ab36cb0e795117d9cb687575604.jpg)  
Figure 1: Left: orientation averaging. The data and the selected subset stay fixed while the function rotates, which is equivalent to rotating the data. The three orientations are illustrative; the estimation error is averaged over all of them. Right: spatial distributions of points selected from a vortex-shaped candidate distribution over Australia, shown in an equal-area projection. The top and middle rows pool 100 and 2,000 selected points from 10,000 and 500 runs, respectively, giving one million selected points per panel; the bottom row shows the first 2,000-point run. KS denotes the pooled two-dimensional Kolmogorov–Smirnov discrepancy against a vortex reference sample (lower is better; not a p-value).

These two geometries cover most embedding spaces used in practice. Many encoders are trained with cosine similarity and output $\ell _ { 2 } \cdot$ -normalized features, so their embeddings lie on the unit sphere; this is the case for contrastive image encoders (Chen et al., 2020; Wang & Isola, 2020), sentence embeddings for language (Reimers & Gurevych, 2019), and multimodal models such as CLIP (Radford et al., 2021). Any other embedding distribution with bounded support, or with rapidly decaying tails such as the approximately Gaussian latent codes of a variational autoencoder (Kingma & Welling, 2014), can be enclosed in a sufficiently large box. Identifying opposite faces of this box places the data on a flat torus while leaving the pairwise distances between data points unchanged.

Our method rests on one reformulation: we treat a selection algorithm as a probability distribution over subsets of the training set (Section 3). Unbiased algorithms that select k elements then form a convex polytope, and the orientation-averaged estimation error is a linear function over it, so unbiased minimum-variance selection is a linear program. The program has one coordinate per subset, far too many for standard solvers, so we instead develop an efficient vertex walk that decreases a variance surrogate while staying on the polytope, and hence unbiased.

We evaluate on CIFAR-10 and MNIST within the widely used DeepCore benchmarking framework, whose comparative study found uniform random selection to remain a strong baseline that competing methods did not consistently outperform (Guo et al., 2022; Moser et al., 2026). In our experiments, our method matches or exceeds uniform selection in mean accuracy at every evaluated budget and outperforms the compared geometric methods at small budgets on CIFAR-10 and MNIST (Section 6). Beyond dataset pruning, the same framework applies to mini-batch construction for stochastic gradient descent, where optimized batches show lower mean gradient variance than random batches at the same batch size (Section 6.2).

## 1.1 CONTRIBUTIONS

1. Formulation. We derive unbiased selection criteria from the principle of no preferred orientation and obtain closed-form pairwise variance expressions on the sphere and the torus (Section 4).

2. Algorithm. We give a label-free, training-free vertex walk that lowers a variance surrogate while preserving unbiasedness (Section 3).

3. Experiments. On CIFAR-10, MNIST, and CelebA, our method matches or exceeds uniform sampling at every budget and outperforms the compared geometric methods at small budgets on CIFAR-10 and MNIST (Section 6).

4. Mini-batches. The same framework reduces gradient variance in mini-batch construction at fixed batch size (Section 6.2).

## 2 RELATED WORK

We refer to Moser et al. (2026) for a recent taxonomy that organizes coreset selection into training-free, training-oriented, and label-free approaches. The DeepCore benchmark (Guo et al., 2022) compares many of these methods on CIFAR-10 and ImageNet and finds that uniform random selection remains a strong baseline that many sophisticated methods fail to beat consistently. Uniform sampling is unbiased; we keep this property by construction and search for lower variance within it.

Training-oriented methods score points with signals collected during training: gradient norms early in training (Paul et al., 2021), gradient matching (Mirzasoleiman et al., 2020; Killamsetty et al., 2021a), and bilevel formulations (Killamsetty et al., 2021b). These methods can reach higher accuracy than training-free selection, but they require labels and at least one training pass over the full dataset, which is precisely the cost that training-free and label-free selection is designed to avoid.

Closest to our setting are selection rules that operate directly on an embedding of the data, without labels or training runs. Herding (Welling, 2009; Chen et al., 2010) greedily selects points so that the subset mean tracks the full-data mean; k-center greedy (Sener & Savarese, 2018) minimizes a covering radius over the feature space; and Sorscher et al. (2022) prune by distance to self-supervised cluster prototypes. Related geometric criteria additionally exploit labels or training signals: moderate coresets keep points at intermediate distances from class centers (Xia et al., 2023), and coveragecentric selection stratifies the budget over difficulty scores to preserve coverage (Zheng et al., 2023). All of these methods encode some version of the heuristic that nearby points are redundant, and all of them bias the distribution of the selected subset, which can drive accuracy below uniform sampling, particularly at small budgets and under class imbalance (Guo et al., 2022; Zheng et al., 2023). In contrast, we derive the geometric intuition from first principles and search only among selections that are provably unbiased.

Unbiasedness is a classical requirement in the subset-sampling literature. Sensitivity-based coresets attach importance weights to the selected points so that the weighted subset loss is an unbiased estimator of the full loss (Feldman & Langberg, 2011; Bachem et al., 2017), and determinantal point processes yield Monte Carlo estimators with faster-than-standard convergence rates (Bardenet & Hardy, 2020). Kernel thinning (Dwivedi & Mackey, 2024) recursively splits the data into balanced blocks using a kernel-dependent randomized procedure, then selects a block with small maximum mean discrepancy and refines it greedily. The last two steps can bias the selection. We make the opposite choice: optimize the partition, then return a uniformly chosen block. Unlike the weighted estimators above, our selections are unweighted, and unbiasedness enters as a linear constraint that turns the set of selection algorithms into a convex polytope.

Mini-batch construction is a second setting where unbiased subset averages matter, since stochastic gradient descent relies on uniformly sampled batches being unbiased. Importance sampling draws examples non-uniformly and reweights their gradients to stay unbiased (Needell et al., 2014; Zhao & Zhang, 2015; Katharopoulos & Fleuret, 2018), at the cost of per-example gradient information. Stratified sampling draws each batch proportionally from clusters of similar examples (Zhao & Zhang, 2014), and repulsive point processes make similar examples less likely to share a batch, whether determinantal (Zhang et al., 2017) or Poisson disk (Zhang et al., 2019), at the cost of a batch gradient that is, in general, unbiased only for a re-weighted objective. Example ordering is a related lever: random reshuffling improves on sampling with replacement (Mishchenko et al., 2020), and GraB uses herding on stale gradients to order examples so that partial sums of gradients track the full gradient (Lu et al., 2022). Our batches are the blocks of an unbiased partition recomputed at each epoch from the embedding geometry alone. Like stratified and repulsive sampling, this needs no gradient information, but the partition minimizes our variance surrogate rather than a fixed clustering or a repulsion heuristic, and a uniformly chosen batch has an unbiased gradient for the original objective at fixed model parameters. Like reshuffling, every example is used once per epoch.

Finally, our closed forms are related to the variance analysis of sampling patterns in computer graphics, where the error of Monte Carlo integration under randomly transformed point sets admits spectral expressions on the torus and on the sphere (Pilleboue et al., 2015; Oztireli, 2016; Singh<sup>¨</sup> et al., 2019). The Poisson disk batches of Zhang et al. (2019) come from the same machinery: the variance of the stochastic gradient is expressed through the first- and second-order product densities of a point process over the data. These works analyze continuous point processes; we transport the machinery to the discrete setting of selecting subsets of a fixed dataset, average the estimation error over orientations of the unknown function, and couple the resulting closed forms with the polytope characterization of unbiased selections to obtain a practical algorithm that is unbiased by construction.

## 3 PROBLEM FORMULATION

## 3.1 DATASET PRUNING

Let $\mathcal { T } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ be a training dataset, consisting of N independent samples from the joint probability distribution $P ( X , Y )$ . We define $x _ { i } \in { \mathcal { X } }$ as the inputs and $y _ { i } \in \mathcal { V }$ as the ground-truth labels. Dataset pruning aims to find a subset $s \subset \tau$ such that a model $M _ { \theta _ { S } } \in \mathcal { M }$ trained on and with parameters $\theta _ { S }$ achieves a good generalization error (Moser et al., 2026):

$$
\begin{array} { r } { S ^ { * } : = \underset { S \subset \mathcal { T } , \frac { | S | } { \left\lceil \mathcal { T } \right\rceil } \leq 1 - \alpha } { \arg \operatorname* { m i n } } \mathbb { E } _ { ( x , y ) \sim P } \big [ \mathcal { L } ( x , y ; M _ { \theta s } ) \big ] , } \end{array}
$$

where $S ^ { * }$ is the optimal subset, $\alpha ~ \in ~ ( 0 , 1 )$ is known as the pruning ratio, and $\mathcal { L }$ is the loss function. In practical applications, a subset $s \subset \tau$ is considered good if $| S | \ll | T |$ and $\mathbb { E } _ { ( x , y ) \sim P } \big [ \mathcal { L } ( x , y ; M _ { \theta s } ) \big ] \approx \mathbb { E } _ { ( x , y ) \sim P } \big [ \mathcal { L } ( x , y ; M _ { \theta _ { T } } ) \big ]$ , that is, if the model trained on the subset generalizes nearly as well as the model trained on the full dataset.

## 3.2 UNBIASED SELECTION

We want selection algorithms that are unbiased, meaning that subset averages equal full-dataset averages in expectation. Under this constraint, the mean squared error of any quantity we estimate from the subset, whether a loss, a gradient, or an accuracy, reduces to its variance, which is therefore the quantity to minimize. To make this precise, we model a selection algorithm $p$ as a discrete probability measure on the subsets of $\tau$ . The set of all such measures is $\mathcal { P } : = \{ q \in [ 0 , 1 ] ^ { 2 ^ { | T | } }$ $\textstyle \sum _ { S \subseteq T } q _ { S } = 1 \}$ , a probability simplex with $2 ^ { | \mathcal { T } | }$ coordinates. If $p \in \mathcal P$ , then $p _ { S }$ is the probability that p assigns to subset $S$ . Unbiasedness is then a condition on $p \mathrm { : }$

Definition 1. A selection algorithm $p \in \mathcal P$ is unbiased if for every function $f : X \to V$ from an arbitrary set $X \supset \tau$ to a vector space $\dot { V }$ over R,

$$
\mathbb { E } _ { S \sim p } \left[ { \frac { 1 } { | S | } } \sum _ { x \in S } f ( x ) \right] = { \frac { 1 } { | T | } } \sum _ { t \in T } f ( t ) .
$$

Two restricted classes of unbiased algorithms will be central to what follows.

Definition 2. An unbiased selection algorithm $p ^ { ( k ) } \in \mathcal { P }$ selects k elements if $\forall S \subseteq \mathcal { T } , \ p _ { S } ^ { ( k ) } >$ $0 \implies | S | = k$ . The set of all such algorithms is denoted by $\mathcal { U } _ { k }$

Definition 3. An algorithm $p ^ { ( k ) } \in \mathcal { U } _ { k }$ is a uniform k-selection algorithm, or simply a k-selection, if $\forall S \subseteq T , p _ { S } ^ { ( k ) } \in \left\{ 0 , k / \left| T \right| \right\}$

It follows that a k-selection exists only if k divides  , and that every k-selection amounts to breaking $\tau$ into $| \mathcal { T } | / k$ blocks of size k and selecting one of them uniformly at random.

## 4 FROM ORIENTATION AVERAGES TO PRACTICAL OBJECTIVES

This section shows how averaging over all orientations yields a practical objective. Since nothing is known about how the downstream function is aligned with the data, no orientation is preferred, and the variance is averaged over all orientations of the function, or equivalently of the data (Figure 1, left). On the sphere, a change of orientation is a rigid motion; on the flat torus, it is a rotation followed by a translation, with points wrapped back onto the torus. Either way the domain is mapped to itself, so the average never involves values of the function on unrelated regions outside it. Averaging over these transformations turns differences in variance between selection algorithms into sums over pairs of selected points (Theorems A.1 and A.2, proved in the Appendix), and these pairwise expressions motivate surrogate objectives that depend only on the geometry of the selection.

Let $f \in L ^ { 2 } ( \mathcal { X } \to \mathbb { R } )$ be a function whose full-dataset average is to be estimated by subset averages. For an unbiased selection algorithm $p \in \mathcal P$ , the variance coincides with the mean squared error and is defined by

$$
V _ { p } : = \mathbb { E } _ { S \sim p } \left[ \left( \frac { 1 } { | T | } \sum _ { t \in \mathcal { T } } f ( t ) - \frac { 1 } { | S | } \sum _ { x \in S } f ( x ) \right) ^ { 2 } \right] .
$$

After a change of orientation $g$ of the points, the variance becomes

$$
V _ { p } ( g ) : = \mathbb { E } _ { S \sim p } \Bigg [ \bigg ( \frac { 1 } { | T | } \sum _ { t \in T } ( f \circ g ) ( t ) - \frac { 1 } { | S | } \sum _ { x \in S } ( f \circ g ) ( x ) \bigg ) ^ { 2 } \Bigg ] .
$$

By Theorems A.1 and A.2, for two unbiased selection algorithms $p , q \in \mathcal { P }$ , averaging the difference $\dot { V _ { p } } ( g ) - V _ { q } ( g )$ over the uniform distribution $\mathcal { G }$ of orientations on the torus or the sphere gives

$$
\mathbb { E } _ { g \sim \mathcal { G } } \big [ V _ { p } ( g ) - V _ { q } ( g ) \big ] = \mathbb { E } _ { S \sim p } \bigg [ \frac { 1 } { | S | ^ { 2 } } \sum _ { x , y \in S } v _ { \mathcal { X } } \big ( h _ { \mathcal { X } } ( x , y ) \big ) \bigg ] - \mathbb { E } _ { S \sim q } \bigg [ \frac { 1 } { | S | ^ { 2 } } \sum _ { x , y \in S } v _ { \mathcal { X } } \big ( h _ { \mathcal { X } } ( x , y ) \big ) \bigg ] ,
$$

in which

$$
v _ { \mathbb { T } ^ { d } } ( r ) : = \sum _ { j \in \mathbb { Z } ^ { d } } \left| \hat { f } ( j ) \right| ^ { 2 } \frac { \Gamma ( d / 2 ) } { ( \pi r \| j \| _ { 2 } ) ^ { d / 2 - 1 } } J _ { d / 2 - 1 } ( 2 \pi r \| j \| _ { 2 } ) , ~ h _ { \mathbb { T } ^ { d } } ( x , y ) : = \| x - y \| _ { 2 }\tag{1}
$$

$$
v _ { S ^ { d } } ( t ) : = \sum _ { \ell \in \mathbb { N } } \| f _ { \ell } \| _ { L ^ { 2 } ( \sigma ) } ^ { 2 } P _ { \ell } ^ { d + 1 } ( t ) , h _ { S ^ { d } } ( x , y ) : = x \cdot y .\tag{2}
$$

Here $\hat { f } ( j )$ is the $j \cdot$ -th Fourier coefficient of $f , J _ { \alpha }$ is the Bessel function of the first kind, $\sigma$ is the normalized surface measure on $S ^ { d }$ , with $\sigma ( S ^ { d } ) = 1 , f _ { \ell }$ is the degree-ℓ spherical harmonic component of $f ,$ and $P _ { \ell } ^ { d + 1 }$ is the generalized Legendre polynomial of dimension $d + 1$ and degree ℓ.

In both cases, the orientation-averaged variance difference is a linear combination of oscillating functions of the pairwise geometry (Figure 2). The spectral weights $| \hat { f } ( j ) | ^ { 2 }$ and $\| f _ { \ell } \| _ { L ^ { 2 } ( \sigma ) } ^ { 2 }$ depend on the unknown function $f ,$ but the oscillating factors they multiply depend only on the distance between pairs of selected points on the torus, or on their inner product on the sphere. The envelopes of these factors identify regions of low correlation: large separations on the torus, and inner products near zero, corresponding to approximately perpendicular points, on the sphere. Placing pairs in these regions keeps every term of the combination small regardless of $f ,$ , which motivates a variance surrogate built from a geometry-only pairwise score $\tilde { v } _ { \mathcal { X } } ;$

$$
\widetilde { V } _ { \mathcal { X } } ( \boldsymbol { p } ) : = \mathbb { E } _ { S \sim \boldsymbol { p } } \left[ \frac { 1 } { \left| S \right| ^ { 2 } } \sum _ { \boldsymbol { x } , \boldsymbol { y } \in S } \widetilde { \boldsymbol { v } } _ { \mathcal { X } } ( \boldsymbol { x } , \boldsymbol { y } ) \right] .\tag{3}
$$

![](images/f091d4371da3fe5886b364b7ee610b082d70ae89bca469e1230356d7c4369cd6.jpg)  
Figure 2: Left: the normalized Bessel factor as a function of the pairwise separation r. Right: the normalized Gegenbauer polynomial $P _ { \ell } ^ { d + 1 } ( t )$ as a function of the inner product t, both for $d = 5$ Colors identify the frequency norm $\Vert j \Vert _ { 2 }$ (left) or the degree ℓ (right), from 1 to $7 ;$ the constant mode is omitted. Dotted red curves show the envelope of each family, and arrows mark the regions of low correlation that motivate the surrogate.

The difference $\widetilde { V } _ { \mathcal { X } } ( p ) \ : - \ : \widetilde { V } _ { \mathcal { X } } ( q )$ then stands in for the orientation-averaged variance difference $\mathbb { E } _ { g \sim \mathcal { G } } [ V _ { p } ( g ) - V _ { q } ( \bar { g } ) ]$ . The experiments use the inverse-distance score $\begin{array} { r } { \tilde { v } _ { \mathcal { X } } ( x , y ) \ = \ \frac { 1 } { \| x - y \| + 0 . 0 1 } } \end{array}$ which favors large separations, following the torus motivation, although exact upper bounds could also be used. On the sphere, scores that favor distant points also favor approximately perpendicular pairs when pairwise inner products are nonnegative. We observed this positive-inner-product regime for almost all sampled pairs of IMDb embeddings.

## 5 OUR METHOD

We partition $\tau$ into $m = \left| \mathcal { T } \right| / k$ blocks $S _ { 1 } , \ldots , S _ { m } ,$ each containing k points.<sup>1</sup> Selecting one of these blocks uniformly at random defines a k-selection p (see Definition 3), which corresponds to a vertex of $\mathcal { U } _ { k }$ . Swapping two points from different blocks preserves the partition structure and therefore defines another k-selection. Algorithm 1 exploits this property by repeatedly proposing swaps between blocks and accepting them only when they decrease the variance surrogate from Section 4:

$$
\widetilde V _ { \mathcal { X } } ( p ) = \frac { 1 } { m k ^ { 2 } } \sum _ { i = 1 } ^ { m } \sum _ { x , y \in S _ { i } } \tilde { v } _ { \mathcal { X } } ( x , y ) .\tag{4}
$$

Lemma A.3 shows that each k-selection of the training set corresponds to a vertex of the polytope of unbiased selection algorithms with selection budget k. It also shows that the expected sum of pairwise kernel evaluations is a linear objective in the subset-selection probabilities. Thus, our swap-based method moves among extreme points of the feasible polytope, each of which is optimal for some linear objective.

As a faster alternative to iterative swaps, we encode each embedding as an $N _ { b } – \mathbf { b i t }$ vector whose scaled Hamming distances approximate the original $\ell _ { 1 }$ distances. We maintain per-bit counts for each bucket, allowing us to compute a point’s total Hamming distance to its members without storing pairwise distances. Using these distances as a surrogate, we greedily assign points to distant buckets while ensuring that each bucket ends with exactly k points. Selecting one completed bucket uniformly at random then preserves unbiasedness and yields a partition-induced vertex of $\mathcal { U } _ { k }$ . Excluding encoding, assignment costs $O ( m N _ { b } | \mathcal { T } | )$ ), which is linear in dataset size for a fixed selection fraction and number of bits. This greedy alternative gives up some fidelity to our derived variance surrogate, and may perform fewer comparisons, in exchange for faster execution (see Section 6).

Algorithm 1 Swap-based partition optimization   
Require: Dataset $\tau ,$ subset size k with $k \mid \mid \tau \mid$ , variance surrogate $\tilde { v } _ { \mathcal { X } }$ , number of iterations L   
Output: A subset of k points   
1: $\bar { m }  | \mathcal { T } | / k$   
2: Precompute $V _ { x , y } \gets \tilde { v } _ { \mathcal { X } } ( x , y )$ for all x, $y \in \tau$   
3: Partition into blocks $\dot { S } _ { 1 } , \dots , S _ { m }$ , each stored as a list of k points   
4: for $\ell = 1 , \ldots , L$ do   
5: Sample $i , j$ uniformly without replacement from $\{ 1 , \ldots , m \}$   
6: Sample $r , .$ s independently and uniformly from $\{ 1 , \ldots , k \}$   
7: $\begin{array} { r } { \Delta \gets - \sum _ { q = 1 } ^ { k } \big ( V _ { S _ { i } [ r ] , S _ { i } [ q ] } + V _ { S _ { j } [ s ] , S _ { j } [ q ] } \big ) } \end{array}$   
8: Swap $S _ { i } [ r ]$ and $S _ { j } [ s ]$   
9: $\begin{array} { r } { \Delta \gets \Delta + \sum _ { q = 1 } ^ { k } \big ( V _ { S _ { i } [ r ] , S _ { i } [ q ] } + V _ { S _ { j } [ s ] , S _ { j } [ q ] } \big ) } \end{array}$   
10: if $\Delta \geq 0$ then   
11: Swap $S _ { i } [ r ]$ and $S _ { j } [ s ]$ back   
12: end if   
13: end for   
14: Sample i uniformly from $\{ 1 , \ldots , m \}$   
15: return $S _ { i }$

## 6 EXPERIMENTS

## 6.1 DATASET PRUNING

We evaluate our method on CIFAR-10, MNIST, CelebA (Liu et al., 2015), and IMDb (Maas et al., 2011) at selection fractions from 0.005 to 0.3 of the training set. We compare against three trainingfree, label-free baselines: uniform sampling, herding (Welling, 2009; Chen et al., 2010), and k-center greedy (Sener & Savarese, 2018). We use the DeepCore framework (Guo et al., 2022) without class-balanced selection quotas, since enforcing them requires labels. Herding uses a corrected implementation, since the original DeepCore implementation contains a bug. Tables report mean test accuracy in percent with 95% Wald confidence-interval half-widths; boldface marks the highest mean in each column, not a statistically significant difference.

On CIFAR-10 and MNIST, the embedding-based methods use features from a ResNet18 trained for 10 epochs on the full labeled training set, following DeepCore’s protocol for benchmark comparability. A freshly initialized ResNet18 is then trained on the selected subset for 200 epochs. Training uses stochastic gradient descent (SGD) with a batch size of 128 and a learning rate of 0.1. Each setting uses 30 jobs. Each job comprises five selection-and-training runs; we average their best-epoch test accuracies within each job and report means and confidence intervals across jobs.

CelebA and IMDb use pretrained embeddings. On CelebA, we use the latent codes of a variational autoencoder (VAE) (Kingma & Welling, 2014), whose regularized, approximately Gaussian embedding fits the toroidal setting. On IMDb, a film-review dataset, we use Qwen3-Embedding-8B embeddings (Zhang et al., 2025) to evaluate the spherical and label-free setting. Among one million uniformly sampled pairs of distinct IMDb training embeddings, all inner products were positive and over 99% exceeded 0.2 (Figure 3). Thus, increasing pairwise separation favors movement toward orthogonality in the observed regime. Test accuracy is that of a multilayer perceptron (MLP) trained on the selected subset, averaged over 100 independent runs per setting on CelebA and 200 on IMDb. Additional experimental details are given in Appendix B.

On all four datasets we use the surrogate score $\tilde { v } _ { \mathcal { X } } ( x , y ) = 1 / ( \| x - y \| _ { 2 } + 0 . 0 1 )$ and run Algorithm 1 for 65,536k proposed swaps, reduced to 2,048k only at fraction 0.3 on CIFAR-10 and MNIST. The fast variant, reported as Ours (fast), replaces Euclidean distances by Hamming distances between 64kbit binary representations of the embeddings. Tables 1, 2, and 3 show that our method is particularly competitive at small budgets on CIFAR-10 and MNIST, whereas k-center greedy achieves the highest mean accuracy at larger budgets on these datasets. On CelebA and IMDb, our method has the highest mean accuracy at every budget except the smallest, where herding leads. Unlike the other methods, ours matches or exceeds uniform sampling in mean accuracy at every evaluated budget.

Table 1: Test accuracy (%) on CIFAR-10 and MNIST.
<table><tr><td colspan="6">CIFAR-10</td></tr><tr><td>Method / selected fraction</td><td>0.005</td><td>0.01</td><td>0.05</td><td>0.1</td><td>0.3</td></tr><tr><td>Herding</td><td> $2 4 . 6 5 \pm 0 . 3 0$ </td><td> $3 0 . 6 3 \pm 0 . 3 9$ </td><td> $5 6 . 6 6 \pm 0 . 4 8$ </td><td> $7 4 . 7 6 \pm 0 . 3 4$ </td><td> $9 0 . 6 1 \pm 0 . 0 7$ </td></tr><tr><td>Ours</td><td> $3 1 . 3 0 \pm 0 . 2 5$ </td><td> ${ \bf 3 8 . 3 0 \pm 0 . 2 8 }$ </td><td> ${ \bf 6 3 . 8 4 \pm 0 . 3 2 }$ </td><td> $\mathbf { 7 6 . 9 1 } \pm 0 . 2 6$ </td><td> $9 0 . 5 9 \pm 0 . 0 6$ </td></tr><tr><td>Ours (fast)</td><td> ${ \bf 3 1 . 3 0 \pm 0 . 2 8 }$ </td><td> $3 8 . 2 0 \pm 0 . 2 8$ </td><td> $6 3 . 6 3 \pm 0 . 4 1$ </td><td> $7 6 . 4 8 \pm 0 . 2 1$ </td><td> $9 0 . 6 7 \pm 0 . 0 5$ </td></tr><tr><td>Uniform</td><td> $3 0 . 6 3 \pm 0 . 5 4$ </td><td> $3 8 . 2 0 \pm 0 . 5 1$ </td><td> $6 3 . 2 2 \pm 0 . 6 9$ </td><td> $7 6 . 3 5 \pm 0 . 3 6$ </td><td> $9 0 . 4 9 \pm 0 . 1 1$ </td></tr><tr><td>k-center greedy</td><td> $2 8 . 9 5 \pm 0 . 2 7$ </td><td> $3 4 . 9 3 \pm 0 . 2 3$ </td><td> $6 0 . 1 9 \pm 0 . 5 1$ </td><td> $7 5 . 7 3 \pm 0 . 3 5$ </td><td> ${ \bf 9 1 . 0 3 \pm 0 . 0 7 }$ </td></tr><tr><td colspan="6">MNIST</td></tr><tr><td>Herding</td><td> $6 4 . 2 5 \pm 1 . 1 3$ </td><td> $8 2 . 8 3 \pm 0 . 6 7$ </td><td> $9 8 . 2 2 \pm 0 . 0 5$ </td><td> $9 9 . 1 7 \pm 0 . 0 2$ </td><td> $9 9 . 5 6 \pm 0 . 0 1$ </td></tr><tr><td>Ours</td><td> $\mathbf { 8 8 . 4 4 \pm 0 . 4 0 }$ </td><td> $9 2 . 8 3 \pm 0 . 2 5$ </td><td> $9 8 . 3 4 \pm 0 . 0 5$ </td><td> $9 9 . 1 3 \pm 0 . 0 2$ </td><td> $9 9 . 5 3 \pm 0 . 0 1$ </td></tr><tr><td>Ours (fast)</td><td> $8 7 . 8 7 \pm 0 . 4 5$ </td><td> $9 2 . 5 7 \pm 0 . 2 1$ </td><td> $9 8 . 2 6 \pm 0 . 0 4$ </td><td> $9 9 . 1 2 \pm 0 . 0 1$ </td><td> $9 9 . 5 3 \pm 0 . 0 1$ </td></tr><tr><td>Uniform</td><td> $8 6 . 3 9 \pm 0 . 7 8$ </td><td> $9 2 . 4 1 \pm 0 . 3 5$ </td><td> $9 8 . 2 0 \pm 0 . 0 7$ </td><td> $9 9 . 0 9 \pm 0 . 0 3$ </td><td> $9 9 . 5 3 \pm 0 . 0 1$ </td></tr><tr><td>k-center greedy</td><td> $6 8 . 9 3 \pm 1 . 2 8$ </td><td> $8 6 . 2 1 \pm 0 . 5 1$ </td><td> ${ \bf 9 8 . 9 6 \pm 0 . 0 2 }$ </td><td> ${ \pm 0 . 4 9 \pm 0 . 0 1 }$ </td><td> ${ \pm 0 . 6 5 \pm 0 . 0 0 }$ </td></tr></table>

Table 2: CelebA mean test attribute accuracy (%).

<table><tr><td>Method / selected fraction</td><td>0.005</td><td>0.01</td><td></td><td></td><td>0.1</td><td>0.3</td></tr><tr><td>Herding</td><td> ${ \pm 0 . 8 9 \pm 0 . 0 4 }$ </td><td> $8 5 . 0 4 \pm 0 . 0 2$ </td><td> $8 8 . 5 6 \pm 0 . 0 1$ </td><td></td><td> $8 9 . 5 2 \pm 0 . 0 1$ </td><td> $9 0 . 6 2 \pm 0 . 0 0$ </td></tr><tr><td>Ours</td><td> $8 2 . 6 5 \pm 0 . 0 7$ </td><td> ${ \pm 0 . 0 8 \pm 0 . 0 3 }$ </td><td> $\mathbf { 8 8 . 6 4 \pm 0 . 0 1 }$ </td><td></td><td> $\mathbf { 8 9 . 6 0 \pm 0 . 0 1 }$ </td><td> ${ \bf 9 0 . 6 6 \pm 0 . 0 1 }$ </td></tr><tr><td>Ours (fast)</td><td> $8 2 . 6 7 \pm 0 . 0 6$ </td><td> $8 5 . 0 2 \pm 0 . 0 3$ </td><td> $8 8 . 6 3 \pm 0 . 0 1$ </td><td></td><td> $8 9 . 5 8 \pm 0 . 0 1$ </td><td> $9 0 . 6 4 \pm 0 . 0 1$ </td></tr><tr><td>Uniform</td><td> $8 2 . 5 7 \pm 0 . 0 7$ </td><td> $8 4 . 9 5 \pm 0 . 0 3$ </td><td> $8 8 . 5 8 \pm 0 . 0 2$ </td><td></td><td> $8 9 . 5 7 \pm 0 . 0 1$ </td><td> $9 0 . 6 5 \pm 0 . 0 1$ </td></tr><tr><td>k-center greedy</td><td> $8 0 . 1 8 \pm 0 . 0 3$ </td><td> $8 1 . 8 0 \pm 0 . 0 7$ </td><td> $8 8 . 2 7 \pm 0 . 0 1$ </td><td></td><td> $8 9 . 4 6 \pm 0 . 0 1$ </td><td> $9 0 . 6 2 \pm 0 . 0 0$ </td></tr></table>

Table 3: IMDb test accuracy (%).
<table><tr><td>Method / fraction</td><td>0.005</td><td>0.01</td><td>0.05</td></tr><tr><td>Herding</td><td> $9 2 . 7 3 \pm 0 . 0 1$ </td><td> $9 3 . 4 6 \pm 0 . 0 1$ </td><td> $9 4 . 4 7 \pm 0 . 0 2$ </td></tr><tr><td>Ours</td><td> $9 2 . 4 4 \pm 0 . 0 9$ </td><td> $9 3 . 5 3 \pm 0 . 0 5$ </td><td> $9 4 . 5 0 \pm 0 . 0 2$ </td></tr><tr><td>Ours (fast)</td><td> $9 2 . 5 2 \pm 0 . 0 9$ </td><td> $9 3 . 5 4 \pm 0 . 0 6$ </td><td> ${ \bf 9 4 . 5 3 \pm 0 . 0 2 }$ </td></tr><tr><td>Uniform</td><td> $9 2 . 0 3 \pm 0 . 1 2$ </td><td> $9 3 . 3 3 \pm 0 . 0 6$ </td><td> $9 4 . 4 5 \pm 0 . 0 2$ </td></tr><tr><td>kCenterGreedy</td><td> $8 4 . 7 6 \pm 0 . 6 1$ </td><td> $9 1 . 0 3 \pm 0 . 0 8$ </td><td> $9 4 . 2 9 \pm 0 . 0 2$ </td></tr></table>

![](images/cfbba2153394b72194940050f91de64ddd0ee97872c6a8180f9800f8795e189b.jpg)  
Figure 3: IMDb inner products.  
Figure 4 shows the recorded runtimes for 4096-dimensional embeddings. The largest run of our method, with $2 ^ { 2 2 }$ embeddings, takes 153.1 seconds, whereas both baselines complete at most $2 ^ { 1 8 }$ embeddings within the 300-second limit of the sweep.

## 6.2 MINI-BATCH VARIANCE REDUCTION

We evaluate mini-batch construction on CelebA’s 40 attributes using ResNet-18 trained with AdamW (initial learning rate $1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 } )$ and cosine annealing $( T _ { \mathrm { m a x } } = 3 0 , \eta _ { \mathrm { m i n } } = 0 )$ . At each epoch, we use the previous epoch’s 512-dimensional pooled features, extracted in evaluation mode, as the selection features for batch optimization. Before each reported epoch, we compare random and optimized partitions of the same samples at fixed model parameters and batch size 128. For each scheme, we estimate variance as the mean squared Euclidean distance of the mini-batch loss gradients from their mean. We report means 95% Wald confidence-interval half-widths across 256 paired runs. Table 4 shows lower mean gradient variance for optimized batches than for random batches at every evaluated epoch.

## 7 LIMITATIONS AND FUTURE WORK

The linear objective of Lemma A.3 has one variable per subset, so Algorithm 1 settles for a local search over uniform k-selections. It accepts only improving swaps, uses a fixed iteration budget with no optimality guarantee, and precomputes all pairwise scores at quadratic cost. Faster or more accurate optimizers, of which the fast variant is a first step, are directions for future work.

The inverse-distance surrogate is motivated by the envelopes of the exact kernels in Theorems A.1 and A.2, but decreasing it does not guarantee a reduction in the true variance. When a single spectral component dominates, the sign of its geometric factor matters: the Bessel factor alternates sign with separation, and even-degree sphere factors favor perpendicular rather than antipodal pairs, so pushing pairs apart can increase the variance. Estimating the dominant components of $f$ from a few labels, and using the exact kernels, could close this gap.

![](images/9f6cfe968e931d9c5fac04ac1c6397cb01cd0918d884a318b7172807023951b8.jpg)  
Figure 4: Recorded runtime versus input size (linear axes). We select $n / 1 2 8$ points from standardnormal, 4096-dimensional float32 embeddings on an NVIDIA GeForce RTX 5090. We use binary representations with 64k bits.

Table 4: Gradient variance of random and optimized mini-batches at fixed batch size.
<table><tr><td>Epoch</td><td>Random variance</td><td>Optimized variance</td><td>Reduction (%)</td></tr><tr><td>2</td><td> $0 . 2 0 3 9 7 \pm 0 . 0 0 9 0 1$ </td><td> $0 . 1 7 3 6 7 \pm 0 . 0 0 7 5 1$ </td><td>14.85</td></tr><tr><td>9</td><td> $0 . 2 3 5 4 4 \pm 0 . 0 0 8 2 7$ </td><td> $0 . 2 0 5 3 1 \pm 0 . 0 0 6 4 4$ </td><td>12.80</td></tr><tr><td>16</td><td> $0 . 2 4 0 4 2 \pm 0 . 0 0 8 3 9$ </td><td> $0 . 2 1 8 2 9 \pm 0 . 0 0 6 5 1$ </td><td>9.20</td></tr><tr><td>23</td><td> $0 . 1 0 8 3 2 \pm 0 . 0 0 4 7 5$ </td><td> $0 . 1 0 3 6 5 \pm 0 . 0 0 4 0 7$ </td><td>4.32</td></tr><tr><td>30</td><td> $0 . 0 0 8 0 0 \pm 0 . 0 0 0 1 8$ </td><td> $0 . 0 0 7 9 4 \pm 0 . 0 0 0 1 7$ </td><td>0.78</td></tr></table>

When the data lie on a submanifold of the domain, as images do in an embedding space, most motions carry them off it. The orientation average then evaluates the downstream function, such as the loss of an image classifier, at points that are not images, and the closed forms are no longer informative about the actual variance. To address this, we could use a submersion that maps the data onto a lower-dimensional sphere or torus that they fill, so that no motion carries them off it.

Unbiasedness is a design choice. For particular functions, a biased selection can have lower estimation error than any unbiased one, for instance a smooth $f$ estimated from the evenly spread points of k-center greedy. The price is a selected subset whose distribution differs from that of the data, as in the k-center greedy column of Figure 1 (right). This is undesirable when the subset must represent the population, for instance under class imbalance.

Future work. Beyond the directions above, the Haar average extends to other homogeneous spaces, such as hyperbolic and projective embeddings and products of spheres, where harmonic analysis yields analogous pairwise kernels. Weighted selections enlarge the polytope and connect the framework to sensitivity-based coresets. On the application side, the annotation-budget setting that motivates label-free selection, such as remote sensing, remains to be tested.

## 8 CONCLUSION

Label-free, training-free dataset pruning has relied on the heuristic that nearby points are redundant, which biases the selected subset. This paper replaces the heuristic with a principle. If no orientation of the downstream function relative to the data is preferred, the objective is the sampling variance averaged over all orientations, and the search space is the polytope of unbiased selection algorithms. On the torus and the sphere, the averaged variance has closed pairwise forms in which the unknown function enters only through spectral weights, and the envelopes of the geometric factors give a surrogate that a vertex walk optimizes without labels or training. The resulting selections keep the unbiasedness of uniform sampling and match or exceed its mean accuracy at every budget on CIFAR-10, MNIST, CelebA, and IMDb with the largest gains at small budgets. The same partitions lower gradient variance in mini-batch construction at fixed batch size.

## AI USE STATEMENT

In this work, we did not use generative AI tools for any task with required disclosure: generating synthetic datasets, developing theoretical models or conceptual frameworks, formulating mathematical claims, providing critical ingredients for proofs, writing proofs, proposing or refining hypotheses, designing or giving feedback on the research methodology or experiments, implementing methods, translation, cleaning or reformatting datasets, qualitative or thematic data analysis, and interpreting results.

We used generative AI tools for the following tasks with recommended disclosure. In writing, they corrected grammar and spelling and gave feedback on the quality of the text. They helped identify relevant literature. They helped create and format the figures and plots. Most of the code in the project was written by hand; generative AI assisted in optimizing CUDA kernels and in orchestrating (but not design) tests.

We have reviewed all AI-assisted work. Every AI-suggested text edit was reviewed by the authors before being accepted; every AI-suggested reference was read and verified against its original source before citation; and the AI-assisted CUDA kernels and test orchestration were verified and tested for correctness by the authors. We take responsibility for the final content of this work, including text, claims, or artifacts produced with the aid of generative AI.

## REPRODUCIBILITY STATEMENT

All theoretical results are stated with their assumptions and proved in full in the Appendix: Theorems A.1 and A.2 give the closed-form variance differences on the torus and the sphere, and Lemma A.3 establishes the polytope and linear-objective structure the method relies on. The definitions of unbiased selection and of k-selections are in Section 3, and Algorithm 1 specifies the selection procedure completely, with the surrogate score used in all experiments given in Section 4. The experimental setup in Section 6 lists the datasets, selection fractions, baselines, training framework and hyperparameters, numbers of runs and seeds, and the reporting conventions for accuracies and confidence intervals. The remaining dataset-pruning settings and the CelebA autoencoder details are given in Appendices B.1 and B.2, and the mini-batch experiments are described in Section 6.2.

## ACKNOWLEDGMENTS

The first author is supported by a scholarship from the Brazilian National Council for Scientific and Technological Development (CNPq). The second author is supported by the Institute for Pure and Applied Mathematics (IMPA).

## REFERENCES

Olivier Bachem, Mario Lucic, and Andreas Krause. Practical coreset constructions for machine learning, 2017.

Remi Bardenet and Adrien Hardy. Monte carlo with determinantal point processes. ´ The Annals of Applied Probability, 30(1), 2020.

Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

Yutian Chen, Max Welling, and Alexander J. Smola. Super-samples from kernel herding. In Proceedings ofthe 26th Conference on Uncertainty in Artificial Intelligence (UAI), 2010.

Raaz Dwivedi and Lester Mackey. Kernel thinning. Journal ofMachine Learning Research, 25(152): 1–77, 2024.

Dan Feldman and Michael Langberg. A unified framework for approximating and clustering data. In Proceedings ofthe 43rd Annual ACM Symposium on Theory ofComputing (STOC), 2011.

H Groemer. Geometric applications of Fourier series and spherical harmonics. ENCYCLOPEDIA OF MATHEMATICS AND ITS APPLICATIONS.

Chengcheng Guo, Bo Zhao, and Yanbing Bai. DeepCore: A Comprehensive Library for Coreset Selection in Deep Learning, 2022.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre. Training compute-optimal large language models, 2022. URL https://arxiv.org/abs/ 2203.15556.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models, 2020. URL https://arxiv.org/abs/2001.08361.

Angelos Katharopoulos and Franc¸ois Fleuret. Not all samples are created equal: Deep learning with importance sampling. In Proceedings ofthe 35th International Conference on Machine Learning, volume 80 of Proceedings ofMachine Learning Research, pp. 2525–2534, 2018.

Krishnateja Killamsetty, Durga Sivasubramanian, Ganesh Ramakrishnan, Abir De, and Rishabh Iyer. GRAD-MATCH: Gradient matching based data subset selection for efficient deep model training. In Proceedings ofthe 38th International Conference on Machine Learning (ICML), 2021a.

Krishnateja Killamsetty, Durga Sivasubramanian, Ganesh Ramakrishnan, and Rishabh Iyer. GLIS-TER: Generalization based data subset selection for efficient and robust learning. In Proceedings ofthe AAAI Conference on Artificial Intelligence, 2021b.

Diederik P. Kingma and Max Welling. Auto-encoding variational Bayes. In International Conference on Learning Representations (ICLR), 2014.

Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), pp. 3730–3738, 2015. URL https://openaccess.thecvf.com/content\_iccv\_2015/ html/Liu\_Deep\_Learning\_Face\_ICCV\_2015\_paper.html.

Yucheng Lu, Wentao Guo, and Christopher De Sa. GraB: Finding provably better data permutations than random reshuffling. In Advances in Neural Information Processing Systems 35, 2022.

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. Learning word vectors for sentiment analysis. In Proceedings ofthe 49th Annual Meeting ofthe Associationfor Computational Linguistics: Human Language Technologies, pp. 142–150, 2011. URL https://aclanthology.org/P11-1015/.

Baharan Mirzasoleiman, Jeff Bilmes, and Jure Leskovec. Coresets for data-efficient training of machine learning models. In Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

Konstantin Mishchenko, Ahmed Khaled, and Peter Richtarik. Random reshuffling: Simple analysis´ with vast improvements. In Advances in Neural Information Processing Systems 33, 2020.

Brian B. Moser, Arundhati S. Shanbhag, Stanislav Frolov, Federico Raue, Joachim Folz, and Andreas Dengel. A Coreset Selection of Coreset Selection Literature: Introduction and Recent Advances, 2026.

Deanna Needell, Nathan Srebro, and Rachel Ward. Stochastic gradient descent, weighted sampling, and the randomized Kaczmarz algorithm. In Advances in Neural Information Processing Systems 27, 2014.

A. Cengiz Oztireli. Integration with stochastic point processes. <sup>¨</sup> ACM Transactions on Graphics, 35 (5), 2016.

Mansheej Paul, Surya Ganguli, and Gintare Karolina Dziugaite. Deep learning on a data diet: Finding important examples early in training. In Advances in Neural Information Processing Systems 34 (NeurIPS), 2021.

Adrien Pilleboue, Gurprit Singh, David Coeurjolly, Michael Kazhdan, and Victor Ostromoukhov. Variance analysis for Monte Carlo integration. ACM Transactions on Graphics, 34(4), 2015.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In Proceedings ofthe 38th International Conference on Machine Learning (ICML), 2021.

Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence embeddings using siamese BERTnetworks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 2019.

Ozan Sener and Silvio Savarese. Active learning for convolutional neural networks: A core-set approach. In International Conference on Learning Representations (ICLR), 2018.

Gurprit Singh, Cengiz Oztireli, Abdalla G. M. Ahmed, David Coeurjolly, Kartic Subr, Oliver Deussen,<sup>¨</sup> Victor Ostromoukhov, Ravi Ramamoorthi, and Wojciech Jarosz. Analysis of sample correlations for Monte Carlo rendering. Computer Graphics Forum, 38(2):473–491, 2019.

Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, and Ari S. Morcos. Beyond neural scaling laws: Beating power law scaling via data pruning. In Advances in Neural Information Processing Systems 35 (NeurIPS), 2022.

Emma Strubell, Ananya Ganesh, and Andrew McCallum. Energy and policy considerations for deep learning in NLP. In Proceedings ofthe 57th Annual Meeting ofthe Associationfor Computational Linguistics (ACL), 2019.

Devis Tuia, Michele Volpi, Loris Copa, Mikhail Kanevski, and Jordi Munoz-Mar˜ ´ı. A survey of active learning algorithms for supervised remote sensing image classification. IEEE Journal ofSelected Topics in Signal Processing, 5(3):606–617, 2011.

Tongzhou Wang and Phillip Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

Max Welling. Herding dynamical weights to learn. In Proceedings of the 26th International Conference on Machine Learning (ICML), 2009.

Xiaobo Xia, Jiale Liu, Jun Yu, Xu Shen, Bo Han, and Tongliang Liu. Moderate coreset: A universal method of data selection for real-world data-efficient deep learning. In International Conference on Learning Representations (ICLR), 2023.

Cheng Zhang, Hedvig Kjellstrom, and Stephan Mandt. Determinantal point processes for mini-batch¨ diversification. In Proceedings of the 33rd Conference on Uncertainty in Artificial Intelligence (UAI), 2017.

Cheng Zhang, Cengiz Oztireli, Stephan Mandt, and Giampiero Salvi. Active mini-batch sampling <sup>¨</sup> using repulsive point processes. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 33, pp. 5741–5748, 2019.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 Embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025. URL https://arxiv.org/abs/2506.05176.

Peilin Zhao and Tong Zhang. Accelerating minibatch stochastic gradient descent using stratified sampling, 2014.

Peilin Zhao and Tong Zhang. Stochastic optimization with importance sampling for regularized loss minimization. In Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pp. 1–9, 2015.

Haizhong Zheng, Rui Liu, Fan Lai, and Atul Prakash. Coverage-centric coreset selection for high pruning rates. In International Conference on Learning Representations (ICLR), 2023.

## A APPENDIX

## A.1 THEOREMS

Theorem A.1 (Torus). Let $\mathcal { T } \subset \mathbb { R } ^ { d } / \mathbb { Z } ^ { d } = \mathbb { T } ^ { d }$ be a finite set of independent samples from $\mu ,$ and let $f \in L ^ { 2 } (  { \mathbb { T } } ^ { d } \to  { \mathbb { R } } )$ . Identify torus points with their canonical representatives in $[ 0 , 1 ) ^ { d } ,$ , and let $w : \mathbb { R } ^ { d }  \mathbf { \bar { \Phi } } [ 0 , 1 ) ^ { d }$ denote coordinatewise wrapping modulo one. For a translation $\mathbf { \chi } _ { t } \in  { \mathbb { T } } ^ { d }$ and a rotation $R \in S O ( d )$ , define

$$
g _ { t , R } ( x ) : = w ( R x + t ) .
$$

Thus, rotation of the fixed representatives is followed by translation and periodic wrapping. The distances below are Euclidean distances between these representatives. Write $g = g _ { t , R }$ . For an unbiased selection algorithm $p \in \mathcal P$ and a nonempty subset ${ \dot { S } } \subseteq { \mathcal { T } }$ , define

$$
V ( g , S ) : = \bigg [ \int _ { \mathbb { T } ^ { d } } \bigl ( f \circ g \bigr ) ( x ) \ \mathrm { d } \mu ( x ) - \frac { 1 } { | S | } \sum _ { x \in S } \bigl ( f \circ g \bigr ) ( x ) \bigg ] ^ { 2 } , \ a n d
$$

$$
V _ { p } ( g ) : = \mathbb { E } _ { S \sim p } [ V ( g , S ) ] = \sum _ { \emptyset \neq S \subseteq \mathcal { T } } p _ { S } V ( g , S ) .
$$

itfollows that $V ( I , S )$ is the realized sampled variance of $S \subseteq { \mathcal { T } }$ , and thus $V _ { p } ( I )$ is the mean ofthe realized sampled variances of the unbiased selection algorithm p (see Definition 1).

If $p , q \in \mathcal { P }$ are unbiased selection algorithms and $\mathcal { G }$ is the distribution of ${ { g } _ { t , R } }$ induced by independently sampling t and R from the normalized Haar measures on $\mathbb { T } ^ { d }$ and $S O ( d )$ , respectively, then

$$
\begin{array} { r l } { \displaystyle \mathbb { E } _ { g \sim \mathcal { G } } \left[ V _ { p } ( g ) - V _ { q } ( g ) \right] = \mathbb { E } _ { S \sim p } \left[ \frac { 1 } { \left| S \right| ^ { 2 } } \sum _ { x , y \in S } v _ { \mathbb { T } ^ { d } } \big ( \left\| x - y \right\| _ { 2 } \big ) \right] } & { } \\ { \displaystyle - \mathbb { E } _ { S \sim q } \left[ \frac { 1 } { \left| S \right| ^ { 2 } } \sum _ { x , y \in S } v _ { \mathbb { T } ^ { d } } \big ( \left\| x - y \right\| _ { 2 } \big ) \right] } \end{array}
$$

in which

$$
v _ { \mathbb { T } ^ { d } } ( r ) = \sum _ { j \in \mathbb { Z } ^ { d } } \left| \hat { f } ( j ) \right| ^ { 2 } \frac { \Gamma ( d / 2 ) } { ( \pi \left\| j \right\| _ { 2 } r ) ^ { d / 2 - 1 } } J _ { d / 2 - 1 } ( 2 \pi \left\| j \right\| _ { 2 } r )
$$

and $J _ { \alpha }$ is the Besselfunction ofthefirst kind. The normalized Besselfactor is defined by its continuous extension, with value one when $\| j \| _ { 2 } r = 0$

Proof. Expanding the square yields

$$
\mathbb { E } _ { g \sim \mathcal { G } } \left[ V ( g , S ) \right] = \mathbb { E } _ { g \sim \mathcal { G } } \left[ \left[ \int _ { \mathbb { T } ^ { d } } ( f \circ g ) ( x ) \ \mathrm { d } \mu ( x ) - \frac { 1 } { | S | } \sum _ { a \in S } ( f \circ g ) ( a ) \right] ^ { 2 } \right]\tag{5}
$$

$$
= \mathbb { E } _ { g \sim \mathcal { G } } \left[ c _ { S } ( g ) \right] + \frac { 1 } { \left| S \right| ^ { 2 } } \mathbb { E } _ { g \sim \mathcal { G } } \left[ \left[ \sum _ { a \in S } ( f \circ g ) ( a ) \right] ^ { 2 } \right] .\tag{6}
$$

Note that for almost every $^ { g , }$

$$
\begin{array} { r l } {  { \mathbb { E } _ { S \sim p } [ c _ { S } ( g ) ] = ( \int _ { \mathbb { T } ^ { d } } ( f \circ g ) ( x ) \ \mathrm { d } \mu ( x ) ) ^ { 2 } } } \\ & { \phantom { = \ } - 2 \int _ { \mathbb { T } ^ { d } } ( f \circ g ) ( x ) \ \mathrm { d } \mu ( x ) \sum _ { \emptyset \ne S \subset T } \frac { p _ { S } } { | S | } \sum _ { a \in S } ( f \circ g ) ( a ) } \end{array}
$$

It follows from our definition of unbiased selection algorithm that

$$
\begin{array} { l } { \displaystyle \mathbb { E } _ { S \sim p } [ c _ { S } ( g ) ] = \left( \int _ { \mathbb { T } ^ { d } } ( f \circ g ) ( x ) \ \mathrm { d } \mu ( x ) \right) ^ { 2 } } \\ { \displaystyle \qquad - 2 \int _ { \mathbb { T } ^ { d } } ( f \circ g ) ( x ) \ \mathrm { d } \mu ( x ) \frac { 1 } { | \mathcal { T } | } \sum _ { z \in \mathcal { T } } ( f \circ g ) ( z ) , } \end{array}
$$

which is independent on the algorithm p and thus

$$
\mathbb { E } _ { S \sim p } [ c _ { S } ( g ) ] = \mathbb { E } _ { S \sim q } [ c _ { S } ( g ) ] .
$$

One can conclude that both terms cancel out and that we can concentrate on the second element of Equation 6. Let us define $\mathcal { R }$ as the uniform distribution over rotation matrices in $S O ( d )$ . Write $\widetilde f = f \circ$ w for the periodic extension of $f .$ Then, it follows that

$$
{ \frac { 1 } { \left| S \right| ^ { 2 } } } \mathbb { E } _ { g \sim \mathcal { G } } \left[ \left[ \sum _ { a \in S } ( f \circ g ) ( a ) \right] ^ { 2 } \right] = { \frac { 1 } { \left| S \right| ^ { 2 } } } \sum _ { a , b \in S } \mathbb { E } _ { g \sim \mathcal { G } } \left[ ( f \circ g ) ( a ) ( f \circ g ) ( b ) \right] ,
$$

in which

$$
\mathbb { E } _ { g \sim \mathcal { G } } \left[ ( f \circ g ) ( a ) ( f \circ g ) ( b ) \right] = \mathbb { E } _ { R \sim \mathcal { R } } \left[ \int _ { [ 0 , 1 ) ^ { d } } \widetilde { f } ( R a + t ) \widetilde { f } ( R b + t ) \ \mathrm { d } t \right] .
$$

Using the substitution $y = R a + t ,$ it follows that $\mathrm { d } y = \mathrm { d } t$ . The translated domain $R a + [ 0 , 1 ) ^ { d }$ can be replaced by $[ 0 , 1 ) ^ { d }$ because $\widetilde { f } ( y ) \widetilde { f } ( R ( b - a ) + y )$ is $\mathbb { Z } ^ { d }$ -periodic, and we can proceed by applying Parseval’s theorem

$$
\begin{array} { r l } { \mathbb { E } _ { R \sim \mathcal { R } } \left[ \displaystyle \int _ { [ 0 , 1 ) ^ { d } } \widetilde { f } ( R a + t ) \widetilde { f } ( R b + t ) \mathrm { ~ d t } \right] } & { = \mathbb { E } _ { R \sim \mathcal { R } } \left[ \displaystyle \int _ { [ 0 , 1 ) ^ { d } } \widetilde { f } ( y ) \widetilde { f } ( R ( b - a ) + y ) \mathrm { ~ d y } \right] } \\ & { = \mathbb { E } _ { R \sim \mathcal { R } } \left[ \displaystyle \sum _ { j \in \mathcal { R } ^ { d } } \mathcal { F } [ f ] ( j ) \overline { { \mathcal { F } [ \widetilde { f } ( \cdot + R ( b - a ) ) ] ( j ) } } \right] } \\ & { = \mathbb { E } _ { R \sim \mathcal { R } } \left[ \displaystyle \sum _ { j \in \mathcal { R } ^ { d } } \Big | \widehat { f } ( j ) \Big | ^ { 2 } e ^ { - 2 \pi i j \cdot R ( b - a ) } \right] } \\ & { = \displaystyle \sum _ { j \in \mathcal { R } ^ { d } } \left[ \widehat { f } ( j ) \right] ^ { 2 } \mathbb { E } _ { R \sim \mathcal { R } } \left[ e ^ { - 2 \pi i j \cdot R ( b - a ) } \right] . } \end{array}
$$

Here $\begin{array} { r } { { \hat { f } } ( j ) = \int _ { [ 0 , 1 ) ^ { d } } f ( y ) e ^ { - 2 \pi i j \cdot y } } \end{array}$ dy; the interchange of expectation and summation follows from $\begin{array} { r } { \sum _ { j } \left| \hat { f } ( j ) \right| ^ { 2 } < \infty } \end{array}$

For $d = 1$ , the formula simplifies to

$$
\sum _ { j \in \mathbb { Z } } \left. \hat { f } ( j ) \right. ^ { 2 } \mathbb { E } _ { R \sim \mathcal { R } } \left[ e ^ { - 2 \pi i j R ( b - a ) } \right] = \sum _ { j \in \mathbb { Z } } \left. \hat { f } ( j ) \right. ^ { 2 } \cos \left( 2 \pi j \left\| b - a \right\| _ { 2 } \right) .
$$

Indeed, $S O ( 1 ) = \{ 1 \}$ and $\left| { \hat { f } } ( - j ) \right| ^ { 2 } = \left| { \hat { f } } ( j ) \right| ^ { 2 }$ since $f$ is real.

Otherwise, $| S ^ { d - 2 } |$ is defined and it follows that (with $\alpha = d / 2 - 1 )$

$$
\begin{array} { r l } { \mathbb { E } _ { R \sim \mathbb { R } } \left[ e ^ { - 2 \pi \hat { \nu } \hat { \nu } \cdot R ( \hat { \nu } - \alpha ) } \right] = \frac { 1 } { \left[ S ^ { \alpha - 1 } \right] } \int _ { S ^ { \alpha - 1 } } e ^ { - 2 \pi \hat { \nu } \cdot ( | b - \alpha | ) } 2 \mathrm { d } x } \\ & { = \frac { 1 } { \left[ S ^ { \alpha - 1 } \right] } \int _ { 0 } ^ { \pi } e ^ { - 2 \pi \hat { \nu } \cdot | b | \cdot | \alpha - \alpha | ) } | \hat { \nu } | ^ { \alpha - \alpha } | \hat { \nu } | ^ { \alpha - 2 } \mathrm { d } \hat { \nu } } \\ & { = \frac { \frac { \Gamma ( \hat { \nu } \cdot ( \hat { \nu } - \hat { \nu } ) / \alpha ) } { 2 ( \hat { \nu } + \hat { \nu } / 2 ) } } { \frac { \Gamma ( \hat { \nu } + \hat { \nu } / 2 ) } { 2 ( \hat { \nu } / 2 ) } } \int _ { 0 } ^ { \pi } e ^ { - 2 \pi \hat { \nu } \cdot | | b | \cdot | \alpha - \hat { \nu } | } \sin ( \theta ) ^ { \hat { \nu } | ^ { \alpha - 2 } } \mathrm { d } \theta } \\ & { = \frac { \Gamma ( \hat { \nu } / 2 ) } { \Gamma ( \hat { \nu } / 2 ) \sqrt { \pi } } \int _ { 0 } ^ { \pi } e ^ { - 2 \pi \hat { \nu } \cdot | | b | \cdot | \alpha - \hat { \nu } | } \frac { 1 } { \alpha } \mathrm { d } \hat { \nu } ( \theta ) ^ { \hat { \nu } | ^ { \alpha - 2 } } \mathrm { d } \theta } \\ &  = \frac { \Gamma ( \hat { \nu } + 1 ) } { \Gamma ( \hat { \nu } / 2 ) \sqrt { \pi } } \int _ { 0 } ^ { \pi } e ^ { - 2 \pi \hat { \nu } \cdot | | b | \cdot | \alpha - \hat { \nu } | } \sin ( \theta ) ^ { \hat { \nu } | ^ { \alpha - 2 } } \mathrm { d } \hat { \nu }  \end{array}
$$

Theorem A.2 (Sphere). Let $\mathcal { T } \subset S ^ { d }$ be afinite set ofindependent samplesfrom $\mu , g : S ^ { d } \to S ^ { d } a$ rigid motion and $\dot { \boldsymbol { f } } \in L ^ { 2 } ( S ^ { d } \to \mathbb { R } )$ . Let σ be the normalized surface measure on $S ^ { d }$ , with $\sigma ( S ^ { d } ) = 1$ and let $f _ { \ell }$ be the degree-ℓ spherical harmonic component off. Its squared norm is

$$
\| f _ { \ell } \| _ { L ^ { 2 } ( \sigma ) } ^ { 2 } : = \int _ { S ^ { d } } \left| f _ { \ell } ( x ) \right| ^ { 2 } \mathrm { d } \sigma ( x ) .
$$

For an unbiased selection algorithm $p \in \mathcal P$ and a nonempty subset $S \subseteq { \mathcal { T } }$ , define

$$
\begin{array} { c l } { \displaystyle V ( \boldsymbol { g } , \boldsymbol { S } ) : = \left[ \int _ { S ^ { d } } ( \boldsymbol { f } \circ \boldsymbol { g } ) ( \boldsymbol { x } ) ~ \mathrm { d } \mu ( \boldsymbol { x } ) - \frac { 1 } { | \boldsymbol { S } | } \sum _ { x \in \boldsymbol { S } } ( \boldsymbol { f } \circ \boldsymbol { g } ) ( \boldsymbol { x } ) \right] ^ { 2 } , a n d } \\ { \displaystyle V _ { p } ( \boldsymbol { g } ) : = \mathbb { E } _ { \boldsymbol { S } \sim p } [ V ( \boldsymbol { g } , \boldsymbol { S } ) ] = \sum _ { \boldsymbol { \vartheta } \neq \boldsymbol { S } \subseteq \boldsymbol { \mathcal { T } } } p _ { \boldsymbol { S } } V ( \boldsymbol { g } , \boldsymbol { S } ) . } \end{array}
$$

itfollows that $V ( I , S )$ is the realized sampled variance of $S \subseteq { \mathcal { T } }$ , and thus $V _ { p } ( I )$ is the mean ofthe realized sampled variances ofthe unbiased selection algorithm p (see Definition 1).

$H p , q \in { \mathcal { P } }$ are unbiased selection algorithms and $\mathcal { G }$ is the uniform distribution over the isometries in $S ^ { d }$ , then

$$
\begin{array} { r l r } {  { \mathbb { E } _ { g \sim \mathcal { G } } [ V _ { p } ( g ) - V _ { q } ( g ) ] = \mathbb { E } _ { S \sim p } [ \frac { 1 } { | S | ^ { 2 } } \sum _ { x , y \in S } v _ { S ^ { d } } ( x \cdot y ) ] } } \\ & { } & { - \mathbb { E } _ { S \sim q } [ \frac { 1 } { | S | ^ { 2 } } \sum _ { x , y \in S } v _ { S ^ { d } } ( x \cdot y ) ] } \end{array}
$$

in which

$$
v _ { S ^ { d } } ( t ) : = \sum _ { \ell \in \mathbb { N } } \| f _ { \ell } \| _ { L ^ { 2 } ( \sigma ) } ^ { 2 } P _ { \ell } ^ { d + 1 } ( t ) ,
$$

and $P _ { \ell } ^ { d + 1 }$ is the generalized Legendre polynomial of dimension $d + 1$ and degree $\ell ,$ that can alternatively be defined for $d \geq 2$ as:

$$
P _ { \ell } ^ { d + 1 } ( t ) = \frac { C _ { \ell } ^ { ( \lambda ) } ( t ) } { C _ { \ell } ^ { ( \lambda ) } ( 1 ) }
$$

in which $C _ { \ell } ^ { ( \lambda ) }$ are the Gegenbauer polynomials, and $\lambda = ( d - 1 ) / 2 .$ . For $d = 1$ , we use the limiting definition

$$
P _ { \ell } ^ { 2 } ( t ) = \cos ( \ell \operatorname { a r c c o s } t ) , \qquad t \in [ - 1 , 1 ] .
$$

Proof. Expanding the square yields

$$
\begin{array} { r l } { \mathbb { E } _ { g \sim \mathcal { G } } \left[ V ( g , S ) \right] = \mathbb { E } _ { g \sim \mathcal { G } } \left[ \left[ \displaystyle \int _ { S ^ { d } } ( f \circ g ) ( x ) \ \mathrm { d } \mu ( x ) - \frac { 1 } { | S | } \displaystyle \sum _ { a \in S } ( f \circ g ) ( a ) \right] ^ { 2 } \right] } & { } \\ { = \mathbb { E } _ { g \sim \mathcal { G } } \left[ c _ { S } ( g ) \right] + \frac { 1 } { | S | ^ { 2 } } \mathbb { E } _ { g \sim \mathcal { G } } \left[ \left[ \displaystyle \sum _ { a \in S } ( f \circ g ) ( a ) \right] ^ { 2 } \right] . } & { } \end{array}\tag{7}
$$

(8)

As shown in Theorem A.1, unbiasedness gives

$$
\mathbb { E } _ { S \sim p } [ c _ { S } ( g ) ] = \mathbb { E } _ { S \sim q } [ c _ { S } ( g ) ]
$$

for almost every $g .$ Thus these terms cancel out in $V _ { p } ( g ) - V _ { q } ( g )$ , and we can focus on the second element of Equation $^ { 8 . }$

$$
{ \frac { 1 } { \left| S \right| ^ { 2 } } } \mathbb { E } _ { g \sim { \mathcal { G } } } \left[ \left[ \sum _ { a \in S } ( f \circ g ) ( a ) \right] ^ { 2 } \right] = { \frac { 1 } { \left| S \right| ^ { 2 } } } \sum _ { a , b \in S } \mathbb { E } _ { Q \sim { \mathcal { G } } } \left[ f ( Q a ) f ( Q b ) \right] .\tag{9}
$$

Let $R$ be any rotation such that $R a = a$ . Since all isometries on the sphere are orthogonal transformations, we will denote $g$ with the letter Q:

$$
\begin{array} { r l } & { \alpha : = \mathbb { E } _ { Q \sim \mathcal { G } } \left[ f ( Q a ) f ( Q b ) \right] } \\ & { \quad = \mathbb { E } _ { Q \sim \mathcal { G } } \left[ f ( Q a ) f ( Q R b ) \right] } \\ & { \quad = \mathbb { E } _ { Q \sim \mathcal { G } } \left[ \displaystyle \frac { 1 } { | C _ { Q a } | } \int _ { x \in C _ { Q a } } f ( Q a ) f ( x ) ~ \mathrm { d } x \right] , } \end{array}
$$

in which $C _ { y } = \{ \left| y \right| x : x \in S ^ { d } \land x \cdot y / \left| y \right| = a \cdot b \}$ $\mathbf { A } \mathbf { t } \ \boldsymbol { a } \cdot \boldsymbol { b } = 1$ , each normalized slice integral below is interpreted as evaluation at $y ;$ at $a \cdot b = - 1$ , it is interpreted as evaluation $\mathrm { a t } - y$ . Note that now the expected value only depends on $Q a$ rather than $Q .$ , and it follows that

$$
\begin{array} { l } { { \displaystyle \alpha = \frac { 1 } { | S ^ { d } | } \int _ { y \in S ^ { d } } \frac { 1 } { | C _ { y } | } \int _ { x \in C _ { y } } f ( y ) f ( x ) \mathrm { ~ d } x \mathrm { ~ d } y } } \\ { { \displaystyle ~ = \frac { 1 } { | S ^ { d } | } \int _ { y \in S ^ { d } } f ( y ) \frac { 1 } { | C _ { y } | } \int _ { x \in C _ { y } } f ( x ) \mathrm { ~ d } x \mathrm { ~ d } y } } \\ { { \displaystyle ~ = \frac { 1 } { | S ^ { d } | } \int _ { y \in S ^ { d } } f ( y ) \overline { { h ( y ) } } \mathrm { ~ d } y } } \\ { { \displaystyle ~ = \langle f , h \rangle _ { L ^ { 2 } ( \sigma ) } . } } \end{array}
$$

Let $A _ { \ell }$ denote the space of degree-ℓ spherical harmonics on $S ^ { d }$ , identified with their homogeneous harmonic extensions to $\mathbb { R } ^ { d + 1 }$

By expressing f as the direct sum

$$
f = \widehat { \bigoplus } _ { \ell \in \mathbb { N } } f _ { \ell } , f _ { \ell } \in A _ { \ell } ,
$$

it follows that

$$
h ( y ) = \frac { 1 } { | C _ { y } | } \int _ { x \in C _ { y } } f ( x ) \ \mathrm { d } x\tag{10}
$$

$$
= { \frac { 1 } { | C _ { y } | } } \int _ { x \in C _ { y } } \sum _ { \ell \in \mathbb { N } } f _ { \ell } ( x ) \ \mathrm { d } x\tag{11}
$$

$$
= \sum _ { \ell \in \mathbb { N } } { \frac { 1 } { | C _ { y } | } } \int _ { x \in C _ { y } } f _ { \ell } ( x ) \ \mathrm { d } x .\tag{12}
$$

We want to show that

$$
h _ { \ell } ( y ) : = \frac { 1 } { | C _ { y } | } \int _ { x \in C _ { y } } f _ { \ell } ( x ) ~ \mathrm { d } x
$$

is in A<sub>ℓ</sub>.

1. $h _ { \ell } ( \lambda y ) = \lambda ^ { \ell } h _ { \ell } ( y )$

For $\lambda \neq 0 ,$

$$
\begin{array} { l } { \displaystyle h _ { \ell } \bigl ( \lambda y \bigr ) = \frac { 1 } { \bigl | C _ { \lambda y } \bigr | } \int _ { x \in C _ { \lambda y } } f _ { \ell } \bigl ( x \bigr ) \mathrm { d } x } \\ { \displaystyle = \frac { 1 } { \bigl | \lambda \bigr | ^ { d - 1 } | C _ { y } | } \int _ { x \in C _ { y } } f _ { \ell } \bigl ( \lambda x \bigr ) \bigl | \lambda \bigr | ^ { d - 1 } \mathrm { d } x } \\ { \displaystyle = \lambda ^ { \ell } h _ { \ell } \bigl ( y \bigr ) . } \end{array}
$$

2. $\Delta h _ { \ell } = 0$

This is a consequence of the Funk–Hecke theorem Groemer, since $\forall y \in S ^ { d } , h _ { \ell } ( y ) =$ $\beta _ { \ell } ( a \cdot b ) \ f _ { \ell } ( y )$ for some function $\beta _ { \ell } \colon [ - 1 , 1 ] \to$ R that will be characterized latter. The previous item can be used to extend the equality to $\mathbb { R } ^ { d + 1 }$ , thus proving that $h _ { \ell } ( y ) \in A _ { \ell }$

$$
\therefore h = \widehat { \bigoplus } _ { \ell \in \mathbb { N } } h _ { \ell } , h _ { \ell } \in A _ { \ell }
$$

Then, we can use the orthogonality of the spherical harmonics and the Funk–Hecke theorem to conclude that

$$
\begin{array} { l } { \displaystyle \alpha = \sum _ { \ell \in \mathbb { N } } \langle f _ { \ell } , h _ { \ell } \rangle _ { L ^ { 2 } ( \sigma ) } } \\ { \displaystyle \ = \sum _ { \ell \in \mathbb { N } } \| f _ { \ell } \| _ { L ^ { 2 } ( \sigma ) } ^ { 2 } \beta _ { \ell } ( a \cdot b ) } \\ { \displaystyle \ = \sum _ { \ell \in \mathbb { N } } \| f _ { \ell } \| _ { L ^ { 2 } ( \sigma ) } ^ { 2 } P _ { \ell } ^ { d + 1 } ( a \cdot b ) , } \end{array}
$$

in which $P _ { \ell } ^ { d + 1 }$ is the generalized Legendre polynomial of dimension $d + 1$ and degree ℓ. □

A selection algorithm $p \in \mathcal P$ is unbiased in the sense of Definition 1 if and only if

$$
p _ { \emptyset } = 0 , \qquad \sum _ { \stackrel { S \subseteq T } { x \in S } } \frac { p _ { S } } { | S | } = \frac { 1 } { | T | } \quad \mathrm { f o r e v e r y } x \in \mathcal { T } .
$$

Indeed, the empty subset must have zero probability for the subset average to be defined. Expanding the expectation over nonempty subsets gives

$$
\mathbb { E } _ { S \sim p } \left[ \frac { 1 } { | S | } \sum _ { x \in S } f ( x ) \right] = \sum _ { x \in \mathcal { T } } f ( x ) \sum _ { \stackrel { S \subseteq \mathcal { T } } { x \in S } } \frac { p _ { S } } { | S | } .
$$

The stated conditions therefore imply unbiasedness; conversely, applying unbiasedness to the indicator function of each individual point gives these conditions. For subsets of fixed size $k \geq 1$ , they reduce to equal inclusion probabilities:

$$
\operatorname* { P r } _ { S \sim p ^ { ( k ) } } ( x \in S ) = \sum _ { \stackrel { S \subseteq \mathcal { T } } { x \in S } } p _ { S } ^ { ( k ) } = \frac { k } { | { \mathcal { T } } | } \quad { \mathrm { f o r ~ e v e r y ~ } } x \in { \mathcal { T } } .
$$

These linear constraints yield the polytope characterization below.

Lemma A.3. Let $\binom { \mathcal { T } } { k } = \left\{ S _ { 1 } , \cdot \cdot \cdot , S _ { \binom { \lvert \mathcal { T } \rvert } { k } } \right\}$ be an enumeration of the subsets of  with size $k .$ Note that any unbiased selection algorithm ofsize k can be uniquely identified as

$$
\begin{array} { c } { p ^ { ( k ) } \in [ 0 , 1 ] ^ { \binom { | T | } { k } } , i n w h i c h } \\ { p _ { i } ^ { ( k ) } = p _ { S _ { i } } ^ { ( k ) } . } \end{array}
$$

Similarly, any lossfunction $\mathcal { L } : \binom { \mathcal { T } } { k } $ R can be uniquely identified as

$$
\begin{array} { c } { \mathcal { L } \in \mathbb { R } ^ { { ( \mathbf { \Gamma } _ { k } ^ { \top } ) } } , i n w h i c h } \\ { \mathcal { L } _ { i } = \mathcal { L } ( S _ { i } ) . } \end{array}
$$

It follows from the definition of expectation that

$$
\begin{array} { r } { \mathbb { E } _ { S \sim p ^ { ( k ) } } [ \mathcal { L } ( S ) ] = \Big \langle p ^ { ( k ) } , \mathcal { L } \Big \rangle _ { \mathbb { R } ^ { \binom { | T | } { k } } } = p ^ { ( k ) } \cdot \mathcal { L } . } \end{array}
$$

Thus,

$$
p ^ { * } = \underset { p ^ { ( k ) } \in \mathcal { U } _ { k } } { \arg \operatorname* { m i n } } \mathbb { E } _ { S \sim p ^ { ( k ) } } [ \mathcal { L } ( S ) ] ,\tag{13}
$$

is a linear programming problem. If k divides $| \tau |$ and $p ^ { ( k ) }$ is a uniform k-selection, then $p ^ { ( k ) }$ is a vertex of the convex polytope $\mathcal { U } _ { k }$

Proof. Enumerate $\mathcal { T } = \{ t _ { 1 } , \ldots , t _ { | T | } \}$ . Let

$$
B : = \Big [ c ( S _ { 1 } ) \quad c ( S _ { 2 } ) \quad \cdots \quad c \left( S _ { \binom { \lvert T \rvert } { k } } \right) \Big ] \in \mathbb { R } ^ { \lvert T \rvert \times \binom { \lvert T \rvert } { k } } ,
$$

in which $c ( S _ { j } )$ are the binary column vectors

$$
c ( S _ { j } ) _ { i } : = \left\{ \begin{array} { l l } { { 1 \mathrm { i f } t _ { i } \in S _ { j } } } \\ { { 0 \mathrm { i f } t _ { i } \notin S _ { j } } } \end{array} \right. .
$$

Using this definition, we can rewrite Equation 13 as the following optimization problem:

Given $\mathcal { L } \in \mathbb { R } ^ { \binom { | T | } { k } } )$ , minimize

$$
\boldsymbol { p } ^ { ( k ) } \cdot \boldsymbol { \mathcal { L } } ,
$$

in which $p ^ { ( k ) } \in \mathbb { R } ^ { \binom { | T | } { k } } )$ is subject to being a probability measure on $\textstyle { \binom { \mathcal { T } } { k } }$ :

$$
\forall 1 \leq i \leq { \binom { | T | } { k } } , 0 \leq p _ { i } ^ { ( k ) } = p _ { S _ { i } } ^ { ( k ) }\tag{14}
$$

$$
\sum _ { i = 1 } ^ { \left( \stackrel { \left| \mathcal { T } \right| } { k } \right) } p _ { i } ^ { \left( k \right) } = 1\tag{15}
$$

and being unbiased, i.e.

$$
\forall t \in T , \sum _ { 1 \leq i \leq { \binom { | T | } { k } } } { \frac { p _ { i } ^ { ( k ) } } { k } } = { \frac { 1 } { | T | } } .\tag{16}
$$

From Equation 16 and non-negativity, it follows immediately that

$$
\forall i , p _ { i } ^ { ( k ) } \leq \frac { k } { | T | } .
$$

Thus, we can rewrite Equations 14-16 as follows:

$$
0 \leq p _ { i } ^ { ( k ) } \leq \frac { k } { | T | }
$$

$$
\begin{array} { r } { \mathbf { 1 } ^ { \top } p ^ { ( k ) } = 1 } \end{array}\tag{17}
$$

(18)

$$
B p ^ { ( k ) } = \frac { k } { | T | } \mathbf { 1 }\tag{19}
$$

which indeed is a convex polytope $\mathcal { U } _ { k } \subset \mathbb { R } ^ { \binom { | T | } { k } }$ . Since in real-world applications the number of dimensions would surpass the number of atoms in the universe, traditional linear programming techniques are not applicable. However, we can find vertices of $\mathcal { U } _ { k }$ by solving the surrogate problem below.

For the vertex claim, assume $k \mid \mid \tau \mid$ . For $v \in \mathbb { R } ^ { ( \mathcal { T } | } )$ , maximize

$$
\| \boldsymbol { v } \| _ { 2 } ^ { 2 }
$$

Subject to $v \in \mathcal V$ , a convex polytope defined by the following two equations

$$
0 \leq v _ { i } \leq \frac { k } { | T | }\tag{20}
$$

$$
\mathbf { 1 } ^ { \mathsf { T } } v = 1\tag{21}
$$

The problem has at least one solution, since $\nu \neq \emptyset$ is a compact set and $\| \boldsymbol { v } \| _ { 2 } ^ { 2 }$ is continuous. We will show that if $v ^ { * }$ is a solution to the problem, then $v _ { i } ^ { * } \in \left\{ 0 , \frac { k } { | T | } \right\}$ . Suppose by hypothesis that u maximizes $\lVert u \rVert _ { 2 } ^ { 2 }$ , but

$$
\exists i _ { 0 } , 0 < u _ { i _ { 0 } } < { \frac { k } { | T | } } .
$$

Since $k \mid \mid \tau \mid$ and $\mathbf { 1 } ^ { \mathsf { T } } u = 1$ , it follows that

$$
\exists i _ { 1 } \neq i _ { 0 } , 0 < u _ { i _ { 1 } } < { \frac { k } { | T | } } .
$$

This implies that u is not a solution, since

$$
u _ { i } ^ { \prime } = \left\{ \begin{array} { l l } { u _ { i } } & { \mathrm { i f ~ } i \neq i _ { 0 } \land i \neq i _ { 1 } } \\ { \operatorname* { m i n } \left\{ \frac { k } { | T | } , u _ { i _ { 0 } } + u _ { i _ { 1 } } \right\} } & { \mathrm { i f ~ } i = i _ { 0 } } \\ { \operatorname* { m a x } \left\{ u _ { i _ { 0 } } + u _ { i _ { 1 } } - \frac { k } { | T | } , 0 \right\} } & { \mathrm { i f ~ } i = i _ { 1 } } \end{array} \right.
$$

satisfies the Equations 20 and 21, and $\| u ^ { \prime } \| _ { 2 } ^ { 2 } > \| u \| _ { 2 } ^ { 2 }$ . Thus, there are ${ \binom { \binom { | T | } { k } } { \frac { | T | } { k } } }$ solutions $v ^ { \ast } \in \mathcal { V }$ that maximize the distance to the origin, which includes all uniform k-selections (see Definition 3). Note that among these solutions, only the uniform k-selections belong to $\mathcal { U } _ { k }$

Since squared Euclidean norm is strictly convex, these maximizers are vertices of $\mathcal { U } _ { k }$ □

## B ADDITIONAL EXPERIMENTAL DETAILS

## B.1 DATASET-PRUNING SETTINGS

CIFAR-10 and MNIST. We use DeepCore (Guo et al., 2022) with ResNet18 and the standard training and test splits. All results in Table 1 use selection without class-balanced quotas. For the embedding-based methods, a randomly initialized ResNet18 is trained on the full labeled training set for 10 epochs with cross-entropy loss. Selection uses its penultimate-layer features, extracted in evaluation mode after the final epoch. Thus, the selection rules require neither labels nor additional training given the embeddings, but producing these embeddings involves supervised training. Evaluation uses a separate, freshly initialized ResNet18 trained on the selected subset for 200 epochs.

Preprocessing and optimization. CIFAR-10 images are normalized channelwise with means (0.4914, 0.4822, 0.4465) and standard deviations (0.2470, 0.2435, 0.2616); MNIST uses mean 0.1307 and standard deviation 0.3081. Downstream CIFAR-10 training additionally uses random crops with four-pixel reflection padding and random horizontal flips. Embedding training, feature extraction, and test evaluation use no augmentation; MNIST uses no augmentation in either training stage. Both training stages use SGD with batch size 128, momentum 0.9, Nesterov acceleration, weight decay $5 \times 1 0 ^ { - 4 }$ , and cross-entropy loss. Embedding training uses a constant learning rate of 0.1; downstream training uses an initial learning rate of 0.1 with cosine annealing to $1 0 ^ { - 4 }$ over 200 epochs, updated after each mini-batch.

## B.2 CELEBA EXPERIMENTAL DETAILS

VAE embeddings. We use an ADM-style residual and self-attention VAE pretrained on aligned CelebA images. RGB images are converted to floating-point values in [0, 1] and resized bilinearly to $1 2 8 \times 1 2 8$ . The model has approximately 130 million parameters and a 512-dimensional Gaussian latent space. Its residual blocks use GroupNorm and SiLU, with four-head self-attention at spatial resolutions $3 2 \times 3 2$ and below; the decoder ends with a sigmoid. The VAE is trained for 140 epochs with batch size 32, learning rate $1 0 ^ { - 4 }$ , and zero weight decay. The objective combines a Gaussian reconstruction likelihood with standard deviation 0.1, KL regularization with a free-bits threshold of 0.5, and a VGG perceptual loss weighted by 0.5. Selection and downstream classification use the deterministic 512-dimensional posterior means, extracted with the VAE in evaluation mode, rather than sampled latent vectors.

Downstream classification. The selection pool contains 40.5k embedding vectors; selection fractions are relative to this pool, and the selected vectors are used for downstream training. Evaluation uses a separate 20.2k-vector test split. The classifier is an MLP with widths $5 1 2 \to 2 5 6 { \overset { \cdot } { \to } } 1 2 8 \to 2 0$ Each hidden linear layer is followed by LayerNorm, ReLU, and dropout with probability 0.2; the output layer produces 20 logits. We train for 20 epochs with Adam, a constant learning rate of $1 0 ^ { - 3 }$ batch size 256, and binary cross-entropy with logits.

The targets are Male, Young, Attractive, Smiling, Heavy Makeup, Wearing Lipstick, Blond Hair, Black Hair, Brown Hair, Gray Hair, Bald, Bangs, Eyeglasses, Goatee, Mustache, No Beard, Wearing

Hat, Wearing Earrings, Pale Skin, and Chubby. Labels are mapped from 1, 1 to 0, 1 , and predictions use a sigmoid threshold of 0.5. Mean test attribute accuracy averages the per-attribute test accuracies over these 20 targets. Each run contributes its highest mean test attribute accuracy over the 20 training epochs, and the table averages these values over 100 runs per setting.

## B.3 IMDB EXPERIMENTAL DETAILS

Data and embeddings. We use the standard labeled IMDb splits: 25,000 training reviews and 25,000 test reviews, each containing equal numbers of positive and negative examples. The unlabeled split is not used. Reviews are embedded with the pretrained Qwen3-Embedding-8B model (Zhang et al., 2025), using the original review text without an added task-specific instruction and a truncation limit of 8,192 tokens. The resulting 4,096-dimensional, unit- $\cdot \ell _ { 2 } .$ -norm embeddings are stored in float32 and used unchanged for selection and downstream classification, with no additional normalization, dimensionality reduction, or augmentation. Selection operates only on the training split, without class-balanced quotas or additional embedding training.

Classifier and optimization. For each selected subset, we train a freshly initialized MLP with widths $4 0 9 6 \to 1 2 8 \to 1 2 8 \to 2$ , ReLU activations after both hidden layers, and no dropout or normalization layers. The output consists of two sentiment logits, trained with cross-entropy loss. Training runs for 200 epochs using SGD with batch size 128, momentum 0.9, Nesterov acceleration, and weight decay $5 \times \bar { 1 0 } ^ { - 4 }$ . The learning rate starts at 0.1 and follows cosine annealing to 10<sup>−4</sup> over 200 epochs, updated after each mini-batch. Training examples are shuffled each epoch.

Epoch selection and reporting. We evaluate accuracy on the full test split after every epoch. Each run contributes its highest test accuracy over the 200 training epochs; no separate validation split is used for epoch selection. The main IMDb comparison uses 200 selection-and-training runs per method–fraction setting. Table 3 reports their mean accuracy and 95% Wald confidence-interval halfwidth, computed as 1.96s/ 200, where s is the sample standard deviation of the per-run best-epoch accuracies.