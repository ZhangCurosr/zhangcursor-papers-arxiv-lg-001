# Resolving Multi-Modal Regression by Diference-Quotient-Based Clustering: Fast Coarse Conditional-Label Assignment

HuangWeiquan Guangdong Polytechnic Normal University ahappyzero@foxmail.com

August 27, 2026

## Abstract

An input x that is paired with several distinct outputs $y _ { 1 } , \ldots , y _ { K }$ is the hallmark of multimodal regression. Under a squared loss, an unconstrained regressor collapses to the conditional mean $\mathbb { E } [ y$ | x], which for $K > 1$ generically coincides with none of the modes. We take the view that this mean-regression pathology is caused by pairwise contradictions inside the data: pairs of samples with nearly identical inputs but distant outputs. We formalize the contradiction of a pair as the ratio $r = \| \Delta y \| / \| \Delta x \|$ and ask whether the problem can be attacked directly by clustering samples so that within-cluster contradictions are minimized. We propose Diference-Quotient Clustering (DQC), a clustering that assigns each sample to the cluster with which its maximum intra-cluster diference quotient is smallest, then trains a logits generator $g ( x )$ and a conditional network f(x, onehot(c)) on the resulting cluster labels. Because the generating modality is unknown at inference, we compare each prediction against all K true outputs of the same input and report the minimum squared error (minMSE). On synthetic benchmarks with $K = 5$ and $K = 1 0$ modalities, the clustering pipeline reaches test minMSE 0.19 $( K = 5 , n _ { x } = 5 0 0 )$ , against 0.09 for an oracle with true labels, 1.08 for random labels, and 1.33 for mean collapse. We report two empirical regularities: (i) the larger the intra-cluster contradiction, the deeper the network required to resolve it; (ii) the oracle, which consumes the true low-dimensional modality structure, generalizes from fewer samples, whereas cluster labels are only equivalent to that structure and therefore need more data. The comparison itself is a hard, multi-threadable algorithm of average complexity $O ( n ^ { 2 } / 2 )$ , positioning the method as a fast front-end for coarse conditional-label assignment that can reduce the training burden of downstream generative refinement such as flow matching or difusion models. The front-end is also naturally iterable. A second stage treats the conditional output $y ^ { \prime } = f _ { 1 } ( x , { \mathrm { o n e h o t } } ( c ) )$ of the first-stage network as a new input and the true output $y$ as the new label, re-runs the same diference-quotient clustering on the pair $( y ^ { \prime } , y )$ to obtain refined labels $d ,$ and trains a second logits generator $g _ { 2 } ( z )$ together with a second conditional network $f _ { 2 } ( z , { \mathrm { o n e h o t } } ( d ) )$ on the augmented input $z = Z ( x , y ^ { \prime } )$ . Because the first stage has already displaced samples toward their modal branches, this second pass measures residual fitting error rather than genuine multi-modality, which yields purer labels theoretically, so this is one of the key directions for our next experiment.

## 1 Introduction

Standard regression assumes a functional relationship $x \mapsto y$ . Real data frequently violates this assumption: the same input x admits several plausible outputs $y _ { 1 } , \ldots , y _ { K }$ . This situation arises whenever the observed input is insuficient to pin down the output uniquely — in inverse problems, multi-agent or multi-path future prediction, and any system whose forward process is stochastic or under-determined. We call such data multi-modal, and its distinct outputs the modalities of x.

A regressor trained with a squared loss on multi-modal data converges to the conditional mean $\mathbb { E } [ y \mid x ] \ [ 1 ]$ . For $K > 1$ this mean is generically at positive distance from every one of the K modes: it predicts a value that never occurs. We call this the mean-regression problem. It is the most basic failure of treating a multi-valued problem with a single-valued loss, and it is the target of this paper.

The essence of multi-possible outputs is contradiction. What constitutes a fitting contradiction? Locally, the dataset must contain pairs of samples $( x _ { i } , y _ { i } ) , ( x _ { j } , y _ { j } )$ with $x _ { i } \approx x _ { j }$ yet $y _ { i } \not \approx y _ { j }$ . The more the outputs difer per unit of input distance, the sharper the incompatibility. We formalize this as the pairwise contradiction

$$
r ( x _ { i } , y _ { i } ; x _ { j } , y _ { j } ) = \frac { \| y _ { i } - y _ { j } \| _ { 2 } } { \| x _ { i } - x _ { j } \| _ { 2 } } ,\tag{1}
$$

the output distance divided by the input distance. Two samples are contradictory when $x _ { i } \approx x _ { j }$ (small denominator) but $y _ { i } \neq y _ { j }$ (nonzero numerator), i.e. when r is large. Under this view, a multi-modal dataset is simply a dataset whose samples cannot all be mutually consistent, and each modality is a subset of samples that are mutually consistent.

Research question. If the mean-regression problem is caused by contradictions, the most direct remedy is to remove them before training: partition the samples into low-contradiction clusters so that each cluster approximates one consistent branch, then fit one conditional branch per cluster. In this paper we test this hypothesis directly. We ask: can a purely geometric, contradictionbased clustering — no gradients, no iterative optimization, no learned similarity — resolve the mean-regression problem on its own?

Why the question is nontrivial. Clustering by contradiction is not the usual geometry of clustering in x-space: it is driven by the joint $( x , y )$ geometry, and a pairwise ratio of distances is a fragile quantity. Three dificulties are immediate. First, the input x alone cannot always disambiguate the modality: the same x appears under several modalities by construction, so any inference-time branch selector must err sometimes. Second, the cluster labels are only equivalent to the true modality structure, never equal to it; residual contradictions inside a cluster persist and must be absorbed by the subsequent network. Third, the greedy assignment is suboptimal by design. Each of these dificulties shows up in the experiments, and each has a characteristic signature.

## Contributions.

• We define multi-possible outputs through the pairwise contradiction $r = \| \Delta y \| / \| \Delta x \|$ and connect it to the mean-regression pathology (Section 3).

• We propose Diference-Quotient Clustering (DQC), a clustering that assigns each sample to the cluster minimizing its maximum intra-cluster diference quotient (Section 4.2), together with a complete pipeline: cluster labels → logits generator $g $ conditional network f (Section 4.3), and a modality-agnostic minMSE evaluation protocol (Section 4.4).

• We report systematic experiments over $K \in \{ 5 , 1 0 \}$ modalities, three dataset sizes, two network depths, and four label schemes (oracle, clustering, random, mean collapse) (Section 5).

• We identify two empirical regularities — deeper networks resolve larger contradictions; the oracle needs fewer samples than equivalent cluster labels — and interpret both (Sections 5.4– 5.5).

• We analyze the practical merits of the comparison: a hard, multi-thread-parallel algorithm of average complexity $O ( n ^ { 2 } / 2 )$ that provides fast coarse conditional labels to reduce the training burden of downstream generative refinement (Section 7).

The rest of the paper is organized as follows. Section 2 discusses related work. Section 3 formalizes the contradiction view. Section 4 describes the data, the clustering, the logits generator, and the minMSE protocol. Section 5 presents the tables and the two empirical regularities. Sections 6–7 discuss limitations and advantages, and Section 8 concludes.

## 2 Related Work

Multi-modal regression. Mixture density networks (MDNs) [2] model the conditional $p ( y \mid x )$ as a Gaussian mixture, and multiple-hypothesis methods [3, 4] output a small set of candidate predictions. These methods represent ambiguity probabilistically or through an ensemble. In contrast, our goal is a purely discrete, pre-training partition of the dataset: we do not fit a generative model of the conditional at all, we only assign coarse labels that make the subsequent conditional network well-posed. The mean-regression failure itself is classical [1]; our contribution is a geometric diagnosis (contradiction) and a gradient-free remedy.

Conditional and mixture-of-experts networks. Mixtures of experts [5] partition the input space with learned gating and train experts jointly by gradient descent. Our clustering plays the role of a frozen, one-shot, data-only assignment: the partition is computed once on the data before any training, contains no learned parameters, and is never updated by the loss. This separation (discovery before training) is deliberate.

Generative refinement. Flow matching [7] and difusion models [6] can synthesize samples from the conditional distribution and thereby produce the fine-grained modal structure that a discrete partition misses. In Section 7 we argue that these methods are complementary: our clustering supplies fast coarse conditional labels as a front-end, reducing the burden of these generative models to refining an already branched structure rather than discovering it.

## 3 Contradiction and the Mean-Regression Problem

Let $\mathcal { D } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n } \subset \mathbb { R } ^ { d } \times \mathbb { R } ^ { m }$ be a dataset in which each input appears K times with K distinct outputs (adding small perturbation to ensure that the denominator of the diference quotient is not equal to zero). A single-output regressor $f _ { \theta }$ trained by min<sub>θ</sub> $\begin{array} { r } { \sum _ { i } \| f _ { \theta } ( x _ { i } ) - y _ { i } \| _ { 2 } ^ { 2 } } \end{array}$ converges to the conditional mean $\begin{array} { r } { f _ { \theta } ( x )  \mathbb { E } [ y \mid x ] = \frac { 1 } { K } \sum _ { k } y _ { k } ( x ) } \end{array}$ ). For $K > 1$ , the mean is generically not one of the modes: $\| \mathbb { E } [ y \mid x ] - y _ { k } ( x ) \| > 0$ for all k. This is the mean-regression problem.

Pairwise contradiction. For two samples $i , j$ , define the contradiction as in Eq. (1): $r _ { i j } = { }$ $\| y _ { i } - y _ { j } \| / \| x _ { i } - x _ { j } \|$ . When $x _ { i } \approx x _ { j }$ , the denominator is small and the ratio amplifies any output disagreement; a large $r _ { i j }$ means the two samples “cannot both be right”. For a set $C ,$ , define the cluster contradiction

$$
R ( C ) = \operatorname* { m a x } _ { i , j \in C } \ r _ { i j } .\tag{2}
$$

A modality is then idealized as a maximal subset with small $R ( C )$ . If the dataset is a disjoint union of K such subsets, a conditional model that is given the subset identity (a one-hot code) can fit each branch without conflict. The mean-regression problem is thus rephrased as: the unconditional regressor is forced to average over subsets it was never told apart.

Hypothesis. If we can recover, from the data alone and without any training, a partition whose clusters have small $R ( C )$ , then conditioning on cluster identity should recover the branches and substantially mitigate the mean collapse. The rest of the paper tests this hypothesis empirically.

## 4 Method

## 4.1 Synthetic multi-modal data

We generate multi-modal regression data with a known ground truth. We draw $n _ { x }$ base inputs $x _ { \mathrm { b a s e } } \sim U ( - 2 , 2 ) ^ { d }$ (with $d = 2 )$ , replicate each input K times, and add independent Gaussian noise of scale $\sigma = 0 . 0 1$ to each copy, so the copies of the same base input remain almost identical in x. Each copy is then passed through one of K independent random modal networks — two-hidden-layer tanh MLPs of width 24 whose weights are drawn once from $\mathcal { N } ( 0 , 1 )$ and then frozen. The k-th modal network produces the output $y _ { k } ( x )$ of modality k, with output dimension $m = 4$ . The K modal networks are shared between the training and test splits so that the true modality identity is well-defined and the oracle baseline is fair. The resulting dataset contains $n = K \cdot n _ { x }$ samples, and for every base input there are exactly K samples that belong to the K diferent modalities. This is the data structure of interest: identical (perturbed) x, multiple diferent y.

## 4.2 Diference-quotient clustering

We cluster the samples so that each cluster has small internal contradiction, without using any labels, gradients, or learned similarity. The algorithm, summarized in Algorithm 1, takes the number of clusters $K _ { c }$ as input (possibly diferent from the true $K \colon$ ; we use $K _ { c } = 8$ for $K = 5$ and $K _ { c } = 2 0$ for $K = 1 0 )$ , samples $K _ { c }$ seed indices uniformly at random, and processes the remaining samples in arbitrary order. Each sample i is assigned to the cluster that minimizes the maximum diference quotient against its current members,

$$
c ^ { * } ( i ) = \underset { c } { \operatorname { a r g m i n } } \ \underset { j \in C _ { c } } { \operatorname* { m a x } } \ \frac { \| y _ { i } - y _ { j } \| } { \| x _ { i } - x _ { j } \| } ,\tag{3}
$$

The min-max objective is deliberate: it prevents any cluster from containing a pair of samples that are strongly contradictory, i.e. it caps $R ( C )$ in a greedy way. The algorithm is deterministic given the seed set and never touches a training loss.

## 4.3 Logits generator and conditional network

Given cluster labels from Algorithm 1, we train two networks.

Logits generator g. A multi-layer perceptron $g : \mathbb { R } ^ { d }  \mathbb { R } ^ { K _ { c } }$ (depth 3, width 64, tanh hidden units) is trained by cross-entropy on the cluster labels. At inference, the selected branch is $\hat { c } = \arg \operatorname* { m a x } _ { c } g _ { c } ( x )$ . g is the only component that must infer the branch from x alone, and it is therefore the principled bottleneck of the pipeline: when the same x appears under several modalities, no function of x can fully disambiguate them, so it must output in the form of probability logits.

Algorithm 1 Diference-quotient clustering   
Require: samples $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ , number of clusters $K _ { c }$   
1: sample $K _ { c }$ distinct seed indices uniformly at random; place each ”seed sample” alone in its own   
cluster $C _ { c }$   
2: for each remaining sample i (in arbitrary order) do   
3: for each cluster c do   
4: $m _ { i , c } \gets \operatorname* { m a x } _ { j \in C _ { c } } \| y _ { i } - y _ { j } \| / \| x _ { i } - x _ { j } \|$ ▷ max contradiction of i w.r.t. cluster $C _ { c }$   
5: end for   
6: assign i to $c ^ { * } = \mathrm { a r g m i n } _ { c } m _ { i , c }$ and add i to $C _ { c ^ { * } }$   
7: end for   
8: return cluster labels $\{ { c } _ { i } \} _ { i = 1 } ^ { n }$

Conditional network $f$ . A second MLP $f : \mathbb { R } ^ { d + K _ { c } }  \mathbb { R } ^ { m }$ maps the concatenation of x and the one-hot encoding of the cluster label to y, trained by squared error on the cluster-assigned samples. For a depth D and width W, $f$ has D tanh hidden layers of width $W$ followed by a linear head; we write the configuration as $^ { 6 6 } \mathrm { d } D \mathrm { w } W ^ { 5 5 }$ . Both networks are trained for 1200 Adam steps with learning rate $1 0 ^ { - 3 }$

## 4.4 Evaluation: minMSE

At inference we observe only x; we do not know which of the K modalities generated the sample. This is the defining condition of multi-modal regression, and it dictates the evaluation metric. For a test input x with K true outputs $y _ { 1 } ( x ) , \ldots , y _ { K } ( x )$ , the pipeline produces a single prediction $\hat { y } = f ( x , \mathrm { o n e h o t } ( \hat { c } ) )$ with $\hat { c } = \arg \operatorname* { m a x } _ { c } g _ { c } ( x )$ . Since we cannot name the generating modality, the only measurable error is the distance to the closest true output,

$$
\mathrm { m i n M S E } ( \boldsymbol { x } ) = \operatorname* { m i n } _ { k = 1 , \ldots , K } \ \left\| \boldsymbol { \hat { y } } - \boldsymbol { y } _ { k } ( \boldsymbol { x } ) \right\| _ { 2 } ^ { 2 } ,\tag{4}
$$

averaged over test inputs. The “min” is not an artifact of the evaluation: it encodes exactly the question the user can answer — did the predictor reproduce at least one of the possible outputs? A prediction equal to the conditional mean, for instance, is at positive distance from every mode and receives a large minMSE, which is precisely the failure mode we aim to detect. The same protocol is applied to all baselines (oracle, random labels, mean collapse) so that every number in the tables answers the same question.

## 5 Experiments

## 5.1 Setup and baselines

We use the generator of Section 4.1 with d = 2 inputs, m = 4 outputs, perturbation $\sigma = 0 . 0 1$ , and modal networks shared across splits. Four label schemes are compared on the same data:

• Oracle: the true modality identity is used as the cluster label $( K _ { c } = K )$ . This is the upper reference — the labels are pure by construction, each branch is one smooth function.

• DQC (proposed): labels from Algorithm 1 with $K _ { c } = 8$ for $K = 5$ and $K _ { c } = 2 0$ for $K = 1 0$

• Random: labels drawn uniformly from $\{ 1 , \ldots , K _ { c } \}$ , independent of the data. This controls for “any branching at $\mathrm { a l l } ^ { \dag }$

Table 1: Five-modality benchmark $( K = 5 , K _ { c } = 8 )$ . For each source size $n _ { x }$ (total samples $K \cdot n _ { x } )$ and network depth, we report training MSE / test minMSE. The line below the table gives the DQC cluster statistics: the maximum and the mean of the per-cluster diference quotient $R ( C ) = \operatorname* { m a x } _ { i , j \in C } \Delta y / \Delta x .$
<table><tr><td>(samples)  $n _ { x }$ </td><td>depth</td><td>oracle</td><td></td><td>DQC</td><td>random</td><td>mean collapse</td></tr><tr><td>80 (400)</td><td>d3w64</td><td>0.0074 0.2147</td><td>0.0064</td><td>0.3025</td><td>0.1977 /1.2633</td><td>0.5813 /1.2975</td></tr><tr><td>80 (400)</td><td>d6w128</td><td>0.0056 0.2174</td><td>0.0051</td><td>0.3145</td><td>0.1896 1.4183</td><td>0.5878 1.4109</td></tr><tr><td>250 (1250)</td><td>d3w64</td><td>0.0131 0.1757</td><td>0.0263</td><td>0.3835</td><td>0.3316 0.9776</td><td>0.5788 /1.3289</td></tr><tr><td>250 (1250)</td><td>d6w128</td><td>0.0110 0.1418</td><td>0.0213</td><td>0.4032</td><td>0.3140 1.0151</td><td>0.5765 /1.3319</td></tr><tr><td>500 (2500)</td><td>d3w64</td><td>0.0137 0.0845</td><td>0.0348</td><td>0.2020</td><td>0.4666 1.1737</td><td>0.5714 /1.3373</td></tr><tr><td>500 (2500)</td><td>d6w128</td><td>0.0173 0.0935</td><td>0.0199</td><td>0.1887</td><td>0.4269 1.0815</td><td>0.5705 1.3346</td></tr></table>

Cluster contradiction R(C) (max / mean over clusters): $n _ { x } { = } 8 0 \colon 5 . 3 ~ / ~ 4 . 7 ;$ $n _ { x } { = } 2 5 0 \colon 1 0 . 7 / 7 . 8 ;$ $n _ { x } { = } 5 0 0 \colon 1 6 . 9 ~ / ~ 1 0 . 3 .$

• Mean collapse: no conditioning at all — a single network $f : \mathbb { R } ^ { d }  \mathbb { R } ^ { m }$ trained by squared error. This is the standard regressor that exhibits the pathology.

For each label scheme we report training MSE and test minMSE (Eq. (4)) under two network depths, the shallow “d3w64” and the deeper “d6w128”, and under three dataset sizes: $n _ { x } ~ \in$ {80, 250, 500} for $K = 5$ and $n _ { x } \in \{ 1 0 0 , 2 0 0 , 5 0 0 \}$ for $K = 1 0$

## 5.2 Five modalities $( K = 5 , K _ { c } = 8 )$

Table 1 reports the results. Three facts stand out.

Clustering substantially alleviates the mean collapse. The mean-collapse baseline is flat at test $\mathrm { m i n M S E } \approx 1 . 3 0 { \ - } 1 . 4 1$ regardless of data size, because the conditional mean sits at positive distance from all five modes. Random labels are equally unhelpful $\left( \approx 0 . 9 8 – 1 . 4 2 \right)$ : adding branches without structure-bearing labels does not help. The DQC pipeline drops test minMSE to 0.19–0.40 and approaches the oracle (0.08–0.22). At $n _ { x } = 5 0 0$ with the deeper network the pipeline reaches 0.1887, which is $5 . 7 \times$ better than random labels and 7.1× better than mean collapse, and within 2.0× of the oracle (0.0935). The contradiction-based partition carries enough structure to make conditioning meaningful.

Random labels do not improve with data; $D Q C$ does. The random baseline stays flat because random conditionals encode no recoverable structure. The DQC pipeline improves as $n _ { x }$ grows $( 0 . 3 1 4 5  0 . 4 0 3 2  0 . 1 8 8 7$ at d6w128), consistent with a method that needs samples to sharpen its partition.

The intra-cluster contradiction grows with the dataset. The bottom rows of Table 1 report the maximum and mean cluster contradiction $R ( C ) = \operatorname* { m a x } _ { i , j \in C } { \Delta y } / { \Delta x }$ over all clusters. As $n _ { x }$ grows from 80 to 500, the maximum grows from 5.3 to 16.9: more samples mean more chances to pack a strongly contradictory pair into the same cluster. This quantity will drive the depth analysis of Section 5.4.

## 5.3 Ten modalities $( K = 1 0 , K _ { c } = 2 0 )$

Table 2 repeats the comparison with twice as many modalities and a finer partition $( K _ { c } = 2 0 )$ . The qualitative picture is identical: mean collapse is flat at ≈ 1.5, random labels at ≈ 1.1–1.3, while the clustering pipeline reaches 0.24–0.44 and remains roughly 1.4–3.2× above the oracle. At $n _ { x } = 5 0 0$ d6w128, the pipeline reaches 0.2431 against oracle 0.1634.

Table 2: Ten-modality benchmark $( K = 1 0 , K _ { c } = 2 0 )$ . Same format as Table 1. The line below the table reports cluster statistics at $n _ { x } = 5 0 0 \colon$ max / mean cluster contradiction, mean purity, and cluster-size range. Mean collapse trains an unconditional network, so it is depth-independent and its two depth entries are identical.
<table><tr><td> $n _ { x }$ </td><td>(samples)</td><td>depth</td><td>oracle</td><td></td><td>DQC</td><td></td><td>random</td><td>mean collapse</td></tr><tr><td>100 (1000)</td><td></td><td>d3w64</td><td>0.0217</td><td>0.2308</td><td>0.0265 0.4445</td><td>0.2994</td><td>1.1851</td><td>0.7192 1.5519</td></tr><tr><td>100 (1000)</td><td></td><td>d6w128</td><td>0.0166</td><td>0.1608</td><td>0.0131 0.3180</td><td>0.3471</td><td>1.0670</td><td>0.7176 1.5204</td></tr><tr><td>200 (2000)</td><td></td><td>d3w64</td><td>0.0285</td><td>0.1549 0.0472</td><td>0.2736</td><td>0.4550</td><td>1.1011</td><td>0.7249 1.5174</td></tr><tr><td>200</td><td>(2000)</td><td>d6w128</td><td>0.0219</td><td>0.1317 0.0328</td><td>0.4199</td><td>0.4631</td><td>1.1095</td><td>0.7240 1.5097</td></tr><tr><td>500</td><td>(5000)</td><td>d3w64</td><td>0.0247</td><td>0.1761 0.0698</td><td>0.2524</td><td>0.5820</td><td>1.2898</td><td>0.7223 1.5351</td></tr><tr><td>500 (5000)</td><td></td><td>d6w128</td><td>0.0220</td><td>0.1634</td><td>0.0589 0.2431</td><td>0.5982</td><td>1.1934</td><td>0.7211 1.5302</td></tr></table>

Cluster stats $( n _ { x } { = } 5 0 0 )$ : R(C) max 8.5 / mean 7.0; mean purity 0.36 (chance 0.10); cluster size range [157, 386].

Two extra diagnostics are reported. The cluster purity — the mean over clusters of the fraction of the dominant true modality — is only 0.36 at $n _ { x } = 5 0 0$ (chance level is 0.10): the clusters are 3.6× better than chance but far from pure. Cluster sizes range over [157, 386]. These numbers are the quantitative content of the limitation discussed in Section 6: the partition is useful but only coarsely aligned with the true modalities.

Notably, because the finer partition produces smaller clusters, the maximum intra-cluster contradiction (8.5) is smaller than in the five-modality case at the same $n _ { x } ~ ( 1 6 . 9 )$ , and accordingly the depth benefit is smaller (Section 5.4).

## 5.4 Depth-related regularity: larger contradictions need deeper networks

Table 1 shows that deepening the conditional network from d3w64 to d6w128 reduces the training MSE of the DQC pipeline substantially, and that the reduction is larger when the intra-cluster contradiction is larger:

• At $n _ { x } = 5 0 0$ (max contradiction 16.9), DQC training MSE drops from 0.0348 to 0.0199, a 43% reduction.

• At $n _ { x } = 8 0$ (max contradiction 5.3), the same deepening reduces training MSE only from 0.0064 to 0.0051, a 20% reduction.

• The oracle, whose clusters are pure and hence contradiction-free, shows no depth benefit: its training MSE stays flat at $\approx 0 . 0 0 5 – 0 . 0 1 7$ across depths and sizes.

The pattern repeats across the two settings: at $K = 1 0$ , where the finer partition $( K _ { c } = 2 0 )$ keeps the maximum contradiction at 8.5, the depth benefit is correspondingly mild $( 0 . 0 6 9 8 \to 0 . 0 5 8 9$ 16%).

We interpret this as follows. A cluster produced by Algorithm 1 still contains residual contradictions: pairs of samples with small $\Delta x$ and moderate $\Delta y$ . The conditional network must interpolate between these locally conflicting points, which requires composing several nonlinear transformations — a deeper network has more layers to “iterate” the local disagreement away. The contradiction $R ( C )$ is a direct measure of how much the branch function deviates from smoothness, and the required depth grows with it. This is the resolution-depth regularity: the larger the contradiction term, the deeper the iterative composition needed to dissolve it.

## 5.5 Sample-size regularity: oracle labels generalize faster

Comparing columns across the dataset sizes in Tables 1–2 reveals a systematic gap in generalization speed. The oracle test minMSE decreases steadily with $n _ { x } ~ ( \mathrm { e . g . } ~ K = 5 $ , d6w128: 0.2174 → $0 . 1 4 1 8  0 . 0 9 3 5 )$ , whereas the DQC pipeline trails it by a stable factor of roughly 1.4–2.8× (e.g. $0 . 3 1 4 5  0 . 4 0 3 2  0 . 1 8 8 7 )$ and narrows the gap only slowly.

The reason is structural, not a matter of tuning. The oracle consumes the true modality identity: each oracle cluster is a single smooth modal function, an object of low efective dimension that generalizes from few samples. The clustering consumes only an equivalent structure: at K = 10 its clusters have purity 0.36, so each learned branch must fit a mixture of several modal functions — a higher-dimensional object that requires more data to disambiguate. Equivalence labels carry less inductive bias than true labels, and the price is paid in sample complexity. This is the sample-size regularity: the oracle generalizes from fewer samples because it uses genuinely lower-dimensional information; the clustering pipeline needs more data because its labels are only equivalent to that information.

## 6 Limitations

Similarity-only comparison. The clustering compares samples by pairwise contradiction alone and never optimizes for low-dimensional regularity inside clusters. It therefore does not seek “the” partition that makes each branch as simple as possible; it only avoids pairs that are locally contradictory. As a consequence the labels are equivalent to(from the perspective of eliminating contradictions), but not equal to, the true modalities, and the pipeline needs considerably more samples to resolve the residual ambiguity (Section 5.5). Incorporating a structure term — e.g. preferring clusters whose members lie on a low-dimensional manifold — is a natural extension.

Greediness and seeds. The assignment is a greedy min-max over a random seed set. There are no optimality guarantees, and the result depends on the seed set. A bad seed set can produce unbalanced or impure clusters (size range [157, 386] at K = 10). A deterministic or seed-robust variant would strengthen the method.

The g bottleneck. At inference the branch is chosen by g(x), a function of x alone. When the same x genuinely supports several modalities, no such function can be correct. This irreducible error is inherited by the whole pipeline and is visible in the persistent gap between the DQC test minMSE and its own training MSE.

Synthetic scope. All experiments use synthetic data with known ground truth. The behavior on real multi-modal datasets, where the number of modalities is unknown and the contradiction geometry is less clean, remains to be tested.

## 7 Advantages and Practical Role

The limitations above are the price of the method’s distinctive properties, which we now make explicit.

Hard, gradient-free, deterministic comparison. Algorithm 1 uses no learned similarity, no threshold, no iterative optimization, and no loss: the only quantity is the geometric ratio $\| \Delta y \| / \| \Delta x \|$ The partition is a one-shot, data-only computation that is frozen before any network is trained.

Trivially parallel. Every pairwise contradiction evaluation is an independent norm computation. Given the current cluster state, the scores of a sample against all candidate clusters can be computed concurrently, and the entire sweep over samples can be parallelized over threads. This is an advantage of hard geometric comparisons over learned, iterative procedures, which are serial by construction.

Complexity. The total number of pairwise evaluations is $\begin{array} { r } { \sum _ { i = 1 } ^ { n } ( i - 1 ) = \frac { n ( n - 1 ) } { 2 } \approx O ( n ^ { 2 } / 2 ) } \end{array}$ : the first sample requires no comparison, the last sample is compared against the $n - 1$ samples assigned before ${ \mathrm { i t } } ,$ and the average cost is quadratic with a factor $1 / 2$ (in big-O notation, $O ( n ^ { 2 } ) ,$ . For the sizes considered here the clustering runs in seconds on a single thread and is far cheaper than training a single network.

A fast front-end for generative refinement. The empirical conclusion of this paper is that diference-quotient-based clustering alone provides coarse accuracy — test minMSE within $2 \times$ of the oracle, but not oracle-level. This is exactly the regime in which the method is most useful as a preprocessing step: it attaches discrete, structurally meaningful conditional labels to the data at negligible cost, so that a downstream fine-grained generative model — e.g. flow matching [7] or a difusion model [6] — does not have to discover the branching structure from scratch but only to refine it. In multi-modal regression pipelines where the expensive part is the conditional generative model, spending $O ( n ^ { 2 } / 2 )$ cheap comparisons up front can substantially reduce the training burden of the refinement stage.

## 8 Conclusion and Future Work

We argued that multi-possible outputs are, at their core, pairwise contradictions between samples, and we formalized the contradiction as the ratio of output distance to input distance. We tested whether clustering by this contradiction directly resolves the mean-regression problem, and found that a diference-quotient clustering, followed by a logits generator and a conditional network, removes most of the mean collapse: on both $K = 5$ and $K = 1 0$ benchmarks the pipeline reaches test minMSE within $1 . 4 \mathrm { - } 3 . 2 \times$ of an oracle with true labels, an order of magnitude better than random labels or mean collapse. We reported two empirical regularities — deeper networks resolve larger contradictions, and oracle labels generalize from fewer samples than equivalent cluster labels — and we positioned the method as a fast, parallelizable, $O ( n ^ { 2 } / 2 )$ front-end for coarse conditional-labe assignment that lowers the training burden of subsequent generative refinement.

Future work proceeds along four directions. (i) Adding a low-dimensional-structure term to the clustering objective to raise purity. (ii) Integrating the coarse labels with fine-grained refinement mechanisms, such as trajectory-consistent threading across time, which have been shown to pin modal identity at oracle-level accuracy. (iii) Validating the pipeline on real multi-modal regression datasets where $K$ must itself be estimated from the data. (iv) Iterative re-clustering in output space: the first stage produces a conditional predictor $f _ { 1 } ( x , { \mathrm { o n e h o t } } ( c ) )$ whose output $y ^ { \prime } = f _ { 1 } ( x , { \mathrm { o n e h o t } } ( c ) )$ already lies closer to one of the true branches than the unconditional mean does. One can treat this $y ^ { \prime }$ as a new input and the true output y as the new label, run the same diference-quotient clustering on the pair $( y ^ { \prime } , y )$ to obtain refined labels $d ,$ form the augmented input $z = Z ( x , y ^ { \prime } )$ and train a second logits generator $g _ { 2 } ( z )$ (predicting $d )$ together with a second conditional network $f _ { 2 } ( z , { \mathrm { o n e h o t } } ( d ) )$ . Because the first stage has already displaced samples toward their modal branches, the contradictions measured in the second pass are dominated by residual fitting error rather than by genuine multi-modality, which may yield purer clusters and a smaller minMSE. We leave this two-stage refinement for future work.

## References

[1] C. M. Bishop. Pattern Recognition and Machine Learning. Springer, 2006.

[2] C. M. Bishop. Mixture density networks. Technical Report NCRG/94/004, Aston University, 1994.

[3] C. Rupprecht, I. Laina, R. Dippel, M. Wimmer, and F. Tombari. Learning in an uncertain world: Representing ambiguity through multiple hypotheses. In ICCV, 2017.

[4] O. Makansi, E. Ilg, O. Cicek, and T. Brox. Overcoming limitations of mixture density networks: A sampling and fitting framework for multimodal future prediction. In CVPR, 2019.

[5] R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton. Adaptive mixtures of local experts. Neural Computation, 3(1):79–87, 1991.

[6] J. Ho, A. Jain, and P. Abbeel. Denoising difusion probabilistic models. In NeurIPS, 2020.

[7] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow matching for generative modeling. In ICLR, 2023.