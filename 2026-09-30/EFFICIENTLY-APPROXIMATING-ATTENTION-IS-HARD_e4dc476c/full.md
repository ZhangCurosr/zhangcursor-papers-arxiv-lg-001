# EFFICIENTLY APPROXIMATING ATTENTION IS HARD

Lukas Haverbeck <sup>♠∗</sup> Carmen Amo Alonso <sup>♣</sup> Andres Felipe Posada-Moreno <sup>♠</sup> Sebastian Trimpe <sup>♠</sup> Marco Pavone <sup>♣</sup>

♠ Institute for Data Science in Mechanical Engineering, RWTH Aachen University

♣ Autonomous Systems Lab, Stanford University

## ABSTRACT

Softmax attention is ubiquitous in modern machine learning, but its quadratic scaling with sequence length makes it costly. To reduce this cost, attention is often approximated with fast algorithms, which incur error but can still perform well in practice and on some inputs. At the same time, the growing diversity of attention applications makes approximation guarantees that do not depend on particular input structure a compelling target. For such uniform guarantees over all inputs, known runtime lower bounds rule out fast algorithms for near-exact attention, but leave open the practically important regime: is there an efficient algorithm with even a modest uniform approximation guarantee? We answer this question negatively. Under standard complexity-theoretic assumptions, no truly subquadratic algorithm can approximate attention with any nontrivial additive or relative guarantee uniformly over all inputs. This impossibility holds in the mildest parameter regime for which known algorithms do not already achieve strong approximation guarantees in near-linear time, and extends to practically relevant relaxations: even after polynomial preprocessing of the KV cache, no efficient algorithm can obtain a nontrivial uniform approximation guarantee, or identify a small set of keys receiving substantial attention under sparsity. Overall, our results settle the computational limits of uniform attention approximation.

## 1 INTRODUCTION

The Transformer architecture has become the de facto standard for the most challenging sequence modeling problems across various domains (Vaswani et al., 2017; Devlin et al., 2019; Dosovitskiy et al., 2021; Gulati et al., 2020; Rives et al., 2021; Zhou et al., 2021), owing largely to its ability to model complex relationships between different points in a sequence. The key architectural component enabling this is softmax attention, which transforms a sequence of vectors based on pairwise comparisons. While attention makes Transformers powerful in practice, its standard implementation scales quadratically with the sequence length, making it the principal bottleneck of Transformer inference and often prohibitively expensive on long sequences (Beltagy et al., 2020; Dao, 2024).

To address this problem, various methods aim to approximate attention with fast algorithms, which can work well empirically on typical inputs (Kitaev et al., 2020; Xiong et al., 2021; Tay et al., 2020; Roy et al., 2021) or admit formal guarantees under structural assumptions on the inputs (Vyas et al., 2020; Chen et al., 2021; Han et al., 2024). In practice, however, Transformers are deployed across increasingly diverse applications and input distributions, making it difficult to test approximation quality exhaustively or to verify that data-dependent assumptions are satisfied. That makes algorithms particularly attractive if they can provide approximation guarantees uniformly over all inputs.

Remarkably strong approximation is already possible with uniform guarantees over sufficiently lowdimensional inputs with sufficiently small norms. In this regime, existing algorithms run in nearlinear time, and guarantee error vanishing at polynomial or even super-polynomial rates in the sequence length (Alman & Song, 2023; Schroder & Mackey, 2026). Outside this regime, however,¨ known lower bounds rule out such strong guarantees. More precisely, the best known lower bounds show that truly subquadratic algorithms, for some sequences of length $n ,$ must incur error $n ^ { - C }$ where $C \gg 0$ is a large constant (Alman & Song, 2023). While these results rule out extreme accuracy with fast algorithms, they leave a substantial gap at the approximation accuracy most relevant in practice. In particular, existing lower bounds do not even rule out a linear-time algorithm that uniformly guarantees error at most, say, $n ^ { - 1 0 0 }$ . For all practical purposes, such high accuracy is effectively indistinguishable from exact computation, while empirical evidence suggests that Transformers remain robust even under much cruder approximations (Liu et al., 2024; Lin et al., 2022; Michel et al., 2019). This raises the question whether attention can be approximated efficiently once one tolerates a modest amount of error. We therefore ask:

## Can attention be approximated efficiently with even a coarse uniform error guarantee?

We show that the answer is no: such an algorithm does not exist. Concretely, we strengthen the proof of Alman & Song (2023) to show that, under standard complexity-theoretic assumptions, no algorithm running in truly subquadratic<sup>1</sup> time can give any nontrivial approximation guarantee uniformly over all inputs. Our result holds for both additive and relative error notions. Moreover, it applies under minimal assumptions on the data dimension and norms: even the slightest relaxation of these assumptions enters the setting in which existing algorithms supply near-linear runtime and vanishingly small error. Consequently, our new lower bound establishes a sharp transition of uniform attention approximation from possible at high accuracy to impossible at any accuracy.

Additionally, we study whether preprocessing or sparse attention structure can accelerate subsequent attention computations with uniform guarantees. Such approaches are common in practice, where one is often willing to incur the quadratic cost of initially processing the input, as well as additional preprocessing overhead, in exchange for faster inference on subsequent queries (Li et al., 2024; Tang et al., 2024; Chen et al., 2025; Liu et al., 2025). We show that, even in this relaxed setting, uniform approximation remains hard. Under the same mild assumptions as before, it is impossible to preprocess the input in polynomial time and subsequently approximate attention in truly sublinear time per token while guaranteeing either a nontrivial approximation of the attention output or, under significant sparsity, retrieval of a small set of keys capturing a constant fraction of the attention.

Contributions. We provide new lower bounds that settle the limits of uniform attention approximation, closing the gap left by prior work and extending them to autoregressive decoding with preprocessing and sparsity. Beyond the regimes where uniform high-precision approximation is already efficiently possible, we show that approximate attention must exploit data-dependent structure.

## 2 RELATED WORK

Fast approximate attention. The quadratic cost of the standard attention implementation has motivated extensive work on faster approximations. For instance, existing approaches exploit sparsity (Child et al., 2019; Zaheer et al., 2020; Tay et al., 2020; Kitaev et al., 2020; Roy et al., 2021), lowrank structure (Wang et al., 2020; Xiong et al., 2021; Chen et al., 2021), or other data-dependent properties of the attention matrix (Zandieh et al., 2023; Han et al., 2024; Aliakbarpour et al., 2026) for efficient attention approximation. These methods can substantially reduce computation while preserving model quality, but they demonstrate performance often only empirically for typical inputs or give guarantees that depend on structural properties of the input. In particular, most approaches do not provide uniform guarantees over all inputs even though such guarantees are appealing given the use of attention across domains as varied as language, vision, speech, biology, and time-series forecasting (Devlin et al., 2019; Dosovitskiy et al., 2021; Gulati et al., 2020; Rives et al., 2021; Zhou et al., 2021). This raises Question (A): is the dependence on input structure merely a limitation of existing algorithms orfundamentally unavoidable in practice?

Uniform approximation guarantees. Another line of work gives approximation guarantees uniformly over a broad class of inputs, typically satisfying a fixed norm bound without requiring additional instance-specific structure. For instance, Choromanski et al. (2021) prove uniform convergence of a random-feature approximation when the query and key norms are bounded, although the number of features depends strongly on the radius and desired accuracy. Alman & Song (2023) obtain a near-linear time algorithm with approximation error decaying at a rate of $n ^ { - c }$ in the sequence length n for a configurable constant $c > 0$ when the entries of the key and query vectors have sufficiently small magnitude. More recently, Schroder & Mackey (2026) achieve even super-¨ polynomial error decay in the sequence length when keys and queries have sufficiently small norms. While these algorithms give approximation guarantees uniformly over all inputs, they have in common that they only result in fast algorithms for sufficiently low-dimensional inputs with sufficiently small norms. This raises Question (B): are these restricted parameter regimes merely a limitation ofexisting algorithms or are they the broadest possiblefor uniform approximation guarantees?

Runtime lower bounds. Existing runtime lower bounds provide evidence that approximating attention is hard, but do not conclusively answer Questions (A) or (B). Early work by Duman Keles et al. (2023) establishes that near-exact attention computation requires near-quadratic runtime when the input dimension and norms are sufficiently large. More directly, Alman & Song (2023) show that, outside the parameter regime in which their algorithm guarantees arbitrary polynomial error decay, there exists some constant $C \gg 0$ such that no truly subquadratic algorithm can approximate attention with error at most $n ^ { - C }$ uniformly over all input sequences of length n. These results concern high-precision approximation, since Alman & Song (2023) quantify $\bar { C }$ only existentially, while Duman Keles et al. (2023) only rule out fast algorithms with even faster-decaying additive error. More recent lower bounds by Gupta et al. (2026, Thm. 5.4) rule out constant-error approximation, but only for inputs with extremely large norms, while their complementary results in more realistic regimes (Gupta et al., 2026, Thm. 5.7) again require extreme precision. In contrast, our new lower bound rules out any nontrivial additive or relative error guarantee, thereby resolving Questions (A) and (B). At a technical level, our results strengthen the proof of Alman & Song (2023) and therefore build on the lower bounds of Rubinstein (2018) for the approximate closest pair problem.

Approximation with preprocessing and sparsity. In practice, the runtime of Transformers on long sequences is often not dominated by the processing of the initial input but instead by the cost of subsequent queries during generation, since the former can be parallelized across the sequence whereas the latter cannot. Because of that, significant effort has gone into designing algorithms that speed up generation, often by preprocessing the sequence into some data structure that subsequently exploits sparsity to attend only over a small subset of the original input (Li et al., 2024; Tang et al., 2024; Chen et al., 2025; Liu et al., 2025). These methods incur the full quadratic cost of attention before generation, as well as the added cost of preprocessing, in exchange for faster generation. Prior lower bounds leave open whether relaxing the computational problem in this way enables fast attention approximation with uniform guarantees during generation. We address this gap by extending our runtime lower bounds to the relaxed problem settings of approximate attention with preprocessing and retrieval under sparsity.

## 3 NONTRIVIAL GUARANTEES ARE IMPOSSIBLE UNDER MILD ASSUMPTIONS

In this section, we ask whether a fast algorithm can approximate attention uniformly over all inputs. After formally defining the problem in Section 3.1, we explain in Section 3.2 why existing lower bounds do not resolve this question. Our main result in Section 3.3 then strengthens the lower-bound argument of Alman & Song (2023) to rule out any nontrivial additive or relative approximation guarantee in truly subquadratic time.

## 3.1 PROBLEM FORMULATION

We consider the standard softmax attention operation over a sequence of input vectors. Attention allows each vector to selectively aggregate information from all other vectors in the sequence. To determine what information to aggregate, each element exposes a query, a key, and a value. Each query is compared with all keys, and the resulting scores determine how the corresponding values are combined. Formally, this operation is defined as follows.

Definition 3.1 (Attention). For matrices $\pmb { Q } \in \mathbb { R } ^ { m \times d }$ and K, $V \in \mathbb { R } ^ { n \times d } ,$ , let $\pmb { A } = \exp ( \pmb { Q } \pmb { K } ^ { \top } / d )$ and $D = \mathrm { d i a g } ( A \mathbf { 1 } _ { n } )$ , where exp is applied entrywise. Define attention as

$$
{ \mathrm { A t t } } ( Q , K , V ) : = D ^ { - 1 } A V \in \mathbb { R } ^ { m \times d } .
$$

Remark 3.2. For better comparability of our results with prior work, we follow Alman & Song (2023, Def. 1.1) in normalizing $Q K ^ { \top }$ by d. In practice, normalization often uses $\sqrt { d }$ or a tunable parameter, instead. The precise normalization is inconsequential for our results, however, because it can always be absorbed into the keys and queries by suitable scaling.

The straightforward implementation of attention computes the full attention matrix A, comparing each of the m queries to each of the n keys before aggregating the values. In the usual setting where $m = n$ , this results in $\mathcal { O } ( n ^ { 2 } )$ operations. At the same time, both the input and output only have $\mathcal { O } ( n d )$ entries and the embedding dimension d is often much smaller than the sequence length $n .$ Consequently, one may hope for an algorithm that avoids explicitly forming all $n ^ { 2 }$ comparisons and computes $\mathrm { A i t } ( Q , K , { \dot { V } } )$ , at least approximately, in much less than $\mathcal { O } ( n ^ { 2 } )$ time. We formalize this problem below for different uniform approximation guarantees such an algorithm could give.

## Problem 1: Approximating attention

Fix parameters $n , d \in \mathbb { N }$ and $B \geq 0$ . Given matrices $Q , K \in [ - B , B ] ^ { n \times d }$ and $V \in [ 0 , 1 ] ^ { n \times d }$ the task is to compute a matrix $\pmb { T } \in \mathbb { R } ^ { n \times d }$ . For this output, the problem

$$
\mathsf { A b s A t t } ( n , d , B , \eta _ { \mathrm { a b s } } )
$$

$$
\begin{array} { r } { \| \mathbf { \boldsymbol { T } } - \mathrm { A t t } ( \mathbf { \boldsymbol { Q } } , \boldsymbol { K } , \boldsymbol { V } ) \| _ { \ell _ { \infty } } \leq \eta _ { \mathrm { a b s } } , } \end{array}
$$

$$
\mathsf { R e l A t t } _ { p } ( n , d , B , \eta _ { \mathrm { r e l } } )
$$

$$
\begin{array} { r } { \| { \boldsymbol { T } } - \mathrm { A t t } ( { \boldsymbol { Q } } , { \boldsymbol { K } } , { \boldsymbol { V } } ) \| _ { \mathcal { S } _ { p } } \leq \eta _ { \mathrm { r e l } } \| \mathrm { A t t } ( { \boldsymbol { Q } } , { \boldsymbol { K } } , { \boldsymbol { V } } ) \| _ { \mathcal { S } _ { p } } . } \end{array}
$$

We allow an algorithm to approximate attention with either an additive or a relative error guarantee. For additive error, we follow prior work in measuring error entrywise in $\ell _ { \infty }$ norm and restrict the values to [0, 1], since otherwise the error could be inflated arbitrarily by scaling the values. For relative error, we instead allow any Schatten norm ${ \mathcal { S } } _ { p } ,$ covering a broad range of natural relative and spectral guarantees not addressed by previous lower bounds. Under these conventions, additive error $\bar { 1 / 2 }$ and relative error 1 are trivial to achieve with no dependence on the input. A useful algorithm should therefore guarantee additive error below $1 / 2$ or relative error below 1.

Our goal is to show that an algorithm with such nontrivial approximation guarantees cannot run in truly less than $\mathcal { O } ( n ^ { 2 } )$ time, under the mildest possible assumptions on d and B. Concretely, we target $d = \Theta ( \mathrm { l o g } \dot { n } )$ and $B = \Theta ( \sqrt { \log n } )$ , since making either parameter asymptotically smaller yields known algorithms with near-linear runtime and with vanishingly small error (Alman & Song, 2023; Schroder & Mackey, 2026). To rule out efficient algorithms at this boundary, we work under¨ the strong exponential time hypothesis (SETH), which is a standard assumption in fine-grained complexity theory. In particular, essentially all prior lower bounds for attention are conditional on SETH (Alman & Song, 2023; Duman Keles et al., 2023; Gupta et al., 2026).

Hypothesis 3.3 (SETH, Impagliazzo & Paturi (2001)). For every $\varepsilon > 0$ , there is an integer $k \geq 3$ such that k-SAT on formulas with n variables cannot be solved in $O ( 2 ^ { ( 1 - \varepsilon ) n } )$ time, not even by a randomized algorithm with bounded error probability.

Table 1: Comparison of hardness results for approximating attention. All results establish $n ^ { 2 - o ( 1 ) }$ runtime lower bounds under SETH, but apply to different settings (head dimensions d and entry bounds B) and rule out different error guarantees. For comparability, all parameters are standardized to exp $( Q K ^ { \top } / d )$ normalization. “Error scale” reports the error threshold below which hardness is established. For the exact attention output $\pmb { Y } \in \overset { \bullet } { \mathbb { R } } ^ { n \times d }$ and an approximation $\pmb { T } \in \mathbb { R } ^ { n \times d }$ , additive error means $\| \mathbfcal { T } - \mathbf { Y } \| _ { \ell _ { \infty } } ;$ ; entrywise-relative error means the smallest η such that $| T _ { i j } - Y _ { i j } | \leq \eta | Y _ { i j } |$ for all $i , j ;$ and norm-relative error means $\| \pmb { T } - \pmb { Y } \| _ { \mathcal { S } _ { p } } / \| \pmb { Y } \| _ { \mathcal { S } _ { p } }$ for any Schatten norm $S _ { p \cdot } \operatorname { A }$ dditive errors are comparable because all constructions use value entries in [0, 1]. All previous results either apply only under extreme temperature or rule out only extremely high-precision approximation. We establish hardness in the mildest setting and for the coarsest error scales, ruling out every nontrivial approximation guarantee under all three error notions.
<table><tr><td rowspan="2">Result</td><td colspan="2">Setting</td><td colspan="3">Error scale</td></tr><tr><td> $d$ </td><td>B</td><td>Additive</td><td>Entrywise relative</td><td>Norm relative</td></tr><tr><td>Gupta et al. (2026) Thm. 5.4</td><td> $2 ^ { \Theta ( \log ^ { * } n ) }$ </td><td> $n ^ { \Theta ( 1 ) }$ </td><td>Θ(1)</td><td>Θ(1)</td><td></td></tr><tr><td>Gupta et al. (2026) Thm. 5.7</td><td> $\Theta ( \log n )$ </td><td> $\Theta ( { \sqrt { \log n } } )$ </td><td> $n ^ { - \Theta ( 1 ) }$ </td><td> $n ^ { - { \dot { \Theta } } ( 1 ) }$ </td><td></td></tr><tr><td>Duman Keles et al. (2023)</td><td>ω(log n)</td><td> $\Theta ( d ^ { 3 / 2 } )$ </td><td> $n ^ { - \Omega ( d ) }$ </td><td>1</td><td></td></tr><tr><td>Alman &amp; Song (2023)</td><td>Θ(log n)</td><td> $\Theta ( { \sqrt { \log n } } )$ </td><td> $n ^ { - \Theta ( 1 ) }$ </td><td> $n ^ { - \Theta ( 1 ) }$ </td><td></td></tr><tr><td>Theorem 3.7 (ours)</td><td>Θ(log n)</td><td> $\Theta ( { \sqrt { \log n } } )$ </td><td>112</td><td>1</td><td>1</td></tr></table>

## 3.2 EXISTING LOWER BOUNDS ONLY RULE OUT HIGH-PRECISION APPROXIMATION

Attention can already be efficiently approximated to extremely high accuracy when the data dimension and norms are sufficiently small. Specifically, when either $d = \mathcal { O } ( \log n )$ and $B = o ( \sqrt { \log n } )$ or $d \ = \ o ( \log n )$ and $B \ = \ \mathcal { O } ( \sqrt { \log n } )$ , there are algorithms with near-linear runtime solving $\mathsf { A b s A t t } ( n , d , B , \eta _ { \mathrm { a b s } } )$ for any constant error guarantee $\eta _ { \mathrm { a b s } } > 0$ . Concretely, for any constant $c > 0$ Alman & Song (2023) achieve error vanishing as $n ^ { - c }$ with the sequence length in $n ^ { 1 + o ( 1 ) }$ time, and Schroder & Mackey (2026) even achieve error vanishing as¨ $n ^ { - } { } ^ { \stackrel { - } { - } \omega ( 1 ) }$ in $n ^ { \bar { 1 } + o ( 1 ) }$ time. Although asymptotic, these guarantees amount effectively to exact attention computation at the sequence lengths for which approximate attention is practically relevant. Consequently, the mildest regime for which we do not already know how to solve Problem 1 efficiently is at the boundary where $d = \Theta ( \log n )$ and $B = \Theta ( \sqrt { \log n } )$

At this boundary, existing runtime lower bounds show that efficient algorithms can no longer guarantee extreme accuracy. Concretely, Alman & Song (2023) show that some polynomial error decay cannot be attained in truly subquadratic time unless SETH fails.

Theorem 3.4 (Alman & Song (2023, Thm. 4.6)). Fix $q > 0$ . Assuming ${ \mathsf { S E T H } } ,$ there exist constants $D , B , C > 0$ such that $\mathsf { A b s A t t } ( n , D$ log $n , B { \sqrt { \log n } } , n ^ { - C } )$ cannot be solved in $\mathcal { O } ( n ^ { 2 - q } )$ time.

While this result rules out the extreme accuracy possible for small d and $B ,$ it leaves open whether some of this accuracy can be exchanged for fast attention approximation in broader regimes. In particular, Theorem 3.4 does not preclude a linear-time algorithm for AbsAtt(n, log $n , { \sqrt { \log n } } , 0 . 4 9 )$

As summarized in Table 1, other lower bounds do not resolve this question either. Closing the gap would require ruling out coarse approximation when $d \ : = \ : \Theta ( \log n )$ and $B \ = \ \Theta ( { \sqrt { \log n } } )$ Duman Keles et al. (2023) rule out nontrivial entrywise-relative approximation, but only for superpolynomially small additive error and under more restrictive assumptions on d and $B .$ Likewise, Gupta et al. (2026, Thm. 5.7) applies to $d = \Theta ( \log n )$ and $B = \Theta ( \sqrt { \log n } )$ , but again only shows that some polynomial decay is impossible, while Gupta et al. (2026, Thm. 5.4) rules out constanterror approximation only in the practically unrealistic setting where query and key entries grow polynomially with n.

Consequently, existing lower bounds leave open whether truly subquadratic algorithms can provide uniform approximation guarantees beyond the parameter regime of known algorithms with near-linear runtime and near-exact approximation. Closing this gap requires ruling out the entire nontrivial range of additive and relative errors already at $\bar { d } \bar { = } \Theta ( \bar { \log n } )$ and $B = \Theta ( { \overline { { \sqrt { \log n } } } } )$ .

## 3.3 NONTRIVIAL GUARANTEES ARE IMPOSSIBLE UNDER THE MILDEST ASSUMPTIONS

We show that, once $d = \Theta ( \log n )$ and $B = \Theta ( \sqrt { \log n } )$ , no truly subquadratic algorithm can solve either $\mathsf { A b s A t t } ( n , d , B , \eta _ { \mathrm { a b s } } )$ for any $\eta _ { \mathrm { a b s } } < 1 / 2$ or $\mathsf { R e l A t t } _ { p } ( n , d , B , \eta _ { \mathrm { r e l } } )$ for any $\eta _ { \mathrm { r e l } } < 1$ , unless SETH fails. To that end, we strengthen the proof of Alman & Song (2023), which draws on the connection between computing attention and finding close vectors.

Definition 3.5 (Closest pair problem). For $n , d \in \mathbb { N }$ and $\varepsilon > 0$ , the problem ${ \mathsf { C P } } ( n , d , \varepsilon )$ is the following. Given sets $\{ a _ { 1 } , \dotsc , a _ { n } \} , \{ b _ { 1 } , \dotsc , b _ { n } \} \subseteq \{ 0 , 1 \} ^ { d } , f n d i ^ { \star } , j ^ { \star } \in [ n ]$ such that

$$
\| a _ { i ^ { \star } } - b _ { j ^ { \star } } \| _ { 0 } \leq ( 1 + \varepsilon ) \operatorname* { m i n } _ { i , j \in [ n ] } \| a _ { i } - b _ { j } \| _ { 0 } .
$$

The central idea underlying the proof of Alman & Song (2023) is that the attention matrix can be repurposed to encode pairwise distances between the two sets of vectors, while the subsequent value aggregation is used to recover a close pair. Hence, a fast algorithm for computing attention yields an essentially equally fast algorithm for finding an approximate closest pair. Finding such a pair, however among two sets of n binary vectors in dimension $d = \mathcal { O } ( \log n )$ cannot be done much more efficiently than checking all $n ^ { 2 }$ possible pairs unless SETH fails, as Rubinstein (2018) proves.

Proposition 3.6 (Rubinstein (2018, Thm. 1.1)). Fix $q > 0 .$ . Assuming SETH, there exist $C > 0$ and $\varepsilon \in ( 0 , 1 )$ such that ${ \mathsf { C P } } ( n , C \log n , \varepsilon )$ cannot be solved in $\mathcal { O } ( n ^ { 2 - q } )$ time.

A truly subquadratic time algorithm for attention therefore cannot exist under SETH. When only a fast approximate algorithm for attention is available, however, the particular construction of Alman & Song (2023) recovers such a pair only when the approximation error is extremely small. We therefore strengthen their construction to find a close pair reliably even when only a very coarse approximation of attention is available. Put together, any truly subquadratic algorithm for attention with a nontrivial approximation guarantee, already for $d = \Theta ( \log n )$ and $\bar { B \ = \ } \Theta ( \sqrt { \log n } )$ , then implies a truly subquadratic algorithm for the closest pair problem, which would contradict SETH via Proposition 3.6. The following theorem formalizes this result. We sketch the main idea here and give the full proof in Appendix A.

Theorem 3.7. Fix $\eta _ { \mathrm { a b s } } \in [ 0 , \frac { 1 } { 2 } ) , \eta _ { \mathrm { r e l } } \in [ 0 , 1 )$ , and $q > 0 .$ . Assuming SETH, there exist $C _ { d } , C _ { B } > 0$ such that neither $\mathsf { A b s A t t } ( n , D , B , \eta _ { \mathrm { a b s } } )$ nor $\mathsf { R e l A t t } _ { p } ( n , D , B , \eta _ { \mathrm { r e l } } )$ can be solved in $\mathcal { O } ( n ^ { 2 - q } )$ time, for $D : = C _ { d } \log n , B : = C _ { B } \sqrt { \log n } ,$ , and any $p \in [ 1 , \infty ]$

Proofsketch. Given two sets of binary vectors as an input to the closest pair problem, we construct from them query and key matrices so that the full $n \times n$ attention matrix of all key–query comparisons already contains the information needed to find close pairs, with the largest attention scores corresponding to closest vectors. The difficulty is that an attention algorithm does not return these scores but only their normalized aggregation against the values, which can obscure the contribution of any individual close pair. Our main idea is to turn this normalization itself into a test whether a close pair with distance below a given threshold exists. To achieve this, we add dummy keys and augment the genuine queries and keys so that close pairs score above the dummies, while far pairs score below them. We then choose the values to measure the total attention paid to genuine keys. By scaling the logits appropriately, a single close genuine key can be made to outweigh all dummy keys whereas the combined contribution of all far genuine keys remains negligible, all while keeping the query and key entries bounded by $B = \mathcal { O } ( \sqrt { \log n } )$ . The resulting attention output then becomes an almost-binary indicator of whether a pair below distance t exists, allowing even any fixed nontrivial additive or relative approximation to distinguish the two cases. Repeating this test over logarithmically many thresholds recovers an approximate closest pair. Thus, a truly subquadratic algorithm for approximating attention would yield a truly subquadratic algorithm for the closest pair problem, contradicting Proposition 3.6 unless SETH fails. □

Theorem 3.7 therefore settles the boundary of efficient attention approximation with uniform guarantees. At $d = \Theta ( \log n )$ and $B \ : = \ : \Theta ( \dot { \sqrt { \log n } } )$ , even the weakest nontrivial guarantees require essentially quadratic work under standard complexity-theoretic assumptions. At the same time, relaxing either parameter to $d = o ( \log n )$ or $\bar { B } \ : = \ : \dot { o } ( \sqrt { \log n } )$ admits algorithms with near-linear runtime and near-exact approximation (Alman & Song, 2023; Schroder & Mackey, 2026). Our re- ¨ sults therefore identify a sharp transition of efficient attention approximation from possible at high accuracy to impossible at any nontrivial accuracy.

## 4 APPROXIMATION REMAINS HARD AFTER KV CACHE PREPROCESSING

Transformer inference in practice is typically optimized for autoregressive generation, where the model extends the original input sequence one token at a time. This differs from prefill, where the initial input is processed all at once. Because autoregressive generation is inherently sequential, it is much harder to parallelize than prefill, making attention during generation often much more costly than during prefill. This has motivated substantial work on accelerating attention specifically during generation through preprocessing of the keys and values produced during prefill, collectively known as the KV cache. In particular, such methods may spend additional computation during or after prefill to compress, index, or otherwise organize the KV cache to reduce the cost of subsequent attention queries (Li et al., 2024; Tang et al., 2024; Chen et al., 2025; Liu et al., 2025).

Allowing algorithms to preprocess the KV cache before approximating attention could, in principle, make uniform approximation substantially easier. Yet existing runtime lower bounds do not address this tradeoff between preprocessing and subsequent query time, despite its practical relevance. We therefore ask whether KV cache preprocessing enables fast attention approximation with nontrivial uniform guarantees that are impossible without it. To answer this question, Section 4.1 adapts the computational problem of approximate attention to autoregressive generation after KV cache preprocessing. Section 4.2 then shows that, even with arbitrary polynomial-time preprocessing and under the same mild assumptions as before, nontrivial uniform guarantees remain impossible in every regime not already covered by fast high-accuracy approximation algorithms.

## 4.1 PROBLEM FORMULATION

To model approximate attention during autoregressive generation, we separate the approximation problem into two stages. In the preprocessing stage, the algorithm receives the KV cache and constructs from it an arbitrary data structure. Only afterward is a single query revealed, and the algorithm must compute its attention output using the preprocessed cache.

```latex
Problem 2: Approximating attention with preprocessing
Fix parameters $n , d \in \mathbb { N }$ and $B \geq 0 .$ . Given matrices $K \in [ - B , B ] ^ { n \times d }$ and $V \in [ 0 , 1 ] ^ { n \times d } ;$
preprocess them. Then, given a query $Q \in [ - B , B ] ^ { 1 \times d }$ , the task is to compute a matrix
$\dot { \boldsymbol { T } } \in \mathbb { R } ^ { 1 \times d }$ . For this output, the problem
$\mathsf { A b s P r e A t t } ( n , d , B , \eta _ { \mathrm { a b s } } )$ requires $\lVert \mathbf { \boldsymbol { T } } - \mathrm { A t t } ( \mathbf { \boldsymbol { Q } } , \boldsymbol { K } , \boldsymbol { V } ) \rVert _ { \ell _ { \infty } } \leq \eta _ { \mathrm { a b s } } ,$ and
${ \mathsf { R e l P r e A t t } } ( n , d , B , \eta _ { \mathrm { r e l } } )$ requires $\begin{array} { r } { \| T - \mathrm { A t t } ( Q , K , V ) \| _ { \mathcal { S } _ { 2 } } \leq \eta _ { \mathrm { r e l } } \| \mathrm { A t t } ( Q , K , V ) \| _ { \mathcal { S } _ { 2 } } . } \end{array}$
```

The naive exact implementation for Problem 2 simply stores the KV cache and subsequently scans the entire cache for each query, costing $\mathcal { O } ( n d )$ preprocessing time and $\mathcal { O } ( n d )$ decoding time. Furthermore, as before, additive error $1 / \hat { 2 }$ and relative error 1 are trivial to guarantee without preprocessing. The relevant question is therefore whether nontrivial uniform approximation guarantees can be achieved in truly sublinear decoding time $\mathcal { O } ( n ^ { 1 - q } )$ after potentially substantial preprocessing.

## 4.2 NONTRIVIAL GUARANTEES ARE IMPOSSIBLE UNDER THE MILDEST ASSUMPTIONS

To rule out fast approximation even after substantial preprocessing, we again exploit the connection between attention and the closest pair problem from Definition 3.5. In fact, Rubinstein (2018, Cor. 1.3) already shows that finding an approximate closest pair still requires checking essentially all pairs even when one of the two vector sets is revealed in advance and may be preprocessed in polynomial time. Incidentally, the reduction underlying Theorem 3.7 already mirrors this structure: one vector set determines $\kappa$ and $V ,$ while the other is used only to form Q. The same reduction therefore naturally extends to Problem 2. We defer the proof details to Appendix A and state only the conclusion here. Under the same mild assumptions as before, arbitrary polynomial-time preprocessing does not enable efficient decoding with nontrivial uniform approximation guarantees.

Theorem 4.1. Fix $\eta _ { \mathrm { a b s } } ~ \in ~ [ 0 , \frac { 1 } { 2 } ) , ~ \eta _ { \mathrm { r e l } } ~ \in ~ [ 0 , 1 )$ , and $k , q \ > \ 0 .$ . Assuming SETH, there exist $C _ { d } , C _ { B } ~ > ~ 0$ such that neither $\mathsf { A b s P r e A t t } ( n , D , B , \eta _ { \mathrm { a b s } } )$ nor $\mathsf { R e l P r e A t t } ( n , D , B , \eta _ { \mathrm { r e l } } )$ can be solved with $O ( n ^ { k } )$ preprocessing time and ${ \dot { \mathcal { O } } } ( n ^ { 1 - q } )$ decoding time, for $\dot { D } : = C _ { d }$ log n and $B : = C _ { B } \sqrt { \log n } .$

## 5 APPROXIMATION REMAINS HARD FOR SPARSE ATTENTION

Attention often exhibits sparse structure in practice, with a small number of tokens receiving most of the attention for any particular query. Many practical methods therefore exploit this structure to accelerate attention during generation by identifying the relevant tokens for each query and attending only over them (Li et al., 2024; Tang et al., 2024; Chen et al., 2025; Liu et al., 2025). Sparse attention methods therefore pursue an objective not captured by Problem 1 and 2: rather than approximating the full attention output, they aim to retain a small set of tokens receiving most of the attention. This can preserve accurate performance at substantially reduced cost if those tokens can be identified efficiently.

Identifying the important tokens could, in principle, be easier than approximating the full attention output, and neither prior work nor our preceding lower bounds rule out this possibility. This section therefore asks whether such retrieval is possible with uniform guarantees over all inputs. To that end, Section 5.1 formalizes retrieval under sparsity as a computational problem. Section 5.2 then shows that, even after arbitrary polynomial-time preprocessing of the $\operatorname { K V }$ cache and under the same mild assumptions as before, no fast algorithm can identify a small set of tokens receiving substantial attention for a given query, even when its attention distribution is highly concentrated.

## 5.1 PROBLEM FORMULATION

A sparse attention method maps a query to a small subset of relevant keys and subsequently evaluates attention only over this subset. To preserve the relevant information from the full computation, the selected keys should collectively receive a substantial fraction of the total attention. We therefore allow an algorithm to preprocess the keys into an arbitrary data structure and, once a query is revealed, ask it to return a small subset capturing a constant fraction of its attention. This objective is meaningful only when the attention distribution itself exhibits sparse structure. If all keys are identical, for example, attention is uniform and no small subset can receive much of the attention. We therefore restrict the problem to queries for which the attention distribution is actually sparse by promising that some key already receives a constant fraction of the attention. Returning any small set containing such a key then suffices.

Problem 3: Sparse attention with preprocessing   
Fix parameters n, $d \in \mathbb { N } , B \geq 0 ,$ and $\alpha \in ( 0 , 1 )$ . Given a matrix $K \in [ - B , B ] ^ { n \times d }$ , preprocess   
it. Then, given a query $\pmb q \in [ - B , B ] ^ { d }$ , the problem Sparse $\mathsf { A t t } ( n , d , B , \alpha )$ asks for a subset   
$\mathbb { I } \subseteq [ n ]$ of the keys receiving attention   
$\begin{array} { r } { \sum _ { i \in \mathbb { I } } \mathrm { s o f t m a x } _ { i } ( K \pmb q / d ) \ge \alpha } \end{array}$ promised that softmax<sub>i</sub>(Kq/d) ≥ α for some $i \in [ n ]$

To solve this problem, a trivial algorithm could compute all attention scores and return a singleton containing the promised key, or it could return all keys, which always captures the full attention. Both approaches require $\mathcal { O } ( n )$ time, however, and therefore provide no speedup over simply computing attention over all keys. The relevant question is therefore whether Problem 3 can be solved in truly sublinear time $\scriptstyle { \mathcal { O } } ( n ^ { 1 - q } )$ after preprocessing. In particular, we count the size of the set toward the runtime, requiring it to carry truly fewer than ${ \mathcal { O } } ( n )$ keys.

## 5.2 EFFICIENTLY RETAINING TOKENS RECEIVING SUBSTANTIAL ATTENTION IS IMPOSSIBLE

We show that, under the same mild assumptions as before, no truly sublinear algorithm can retain any constant fraction of the attention after arbitrary polynomial-time preprocessing. While we still exploit the connection between attention and the closest pair problem from Definition 3.5, the reduction of Alman & Song (2023) underlying Theorems 3.7 and 4.1 does not extend directly to sparse attention. Its construction encodes close pairs through attention scores that are large relative to those of far pairs. Yet, a closest pair instance may contain many approximately closest pairs at similar distances. On such inputs, approximating the full attention output can reveal some such pair even when the attention distribution is diffuse, violating the promise of Problem 3. To obtain hardness in the presence of sparsity, we therefore modify the reduction so that one approximately closest candidate dominates all others. We achieve this by randomly perturbing the encoded keys in a way that isolates one of the approximately closest keys to receive almost all of the attention. Any subset capturing a constant fraction of the attention must then contain this key and thereby reveal an approximate closest pair. We state the resulting lower bound below and sketch the reduction, deferring the full proof to Appendix A.

Theorem 5.1. Fix any $\alpha \in ( 0 , 1 )$ and k, $q > 0 .$ Assuming SETH, there exist $C _ { d } , C _ { B } > 0$ such that $\mathsf { S p a r s e A t t } ( n , D , B , \alpha )$ cannot be solved with $\mathcal { O } ( n ^ { k } )$ preprocessing time and $O ( n ^ { 1 - q } )$ decoding time, for $D : = C _ { d } \log _ { i }$ n and $B : = C _ { B } \sqrt { \log n }$

Proofsketch. Suppose there were an algorithm for Problem 3 with polynomial preprocessing time and truly sublinear decoding time. We use it to construct a bounded-error randomized algorithm for approximate nearest-neighbor search after polynomial preprocessing, contradicting SETH under the lower bound of Rubinstein (2018). As in Theorem 3.7, we encode the preprocessed vectors as keys and the later query vector as an attention query, so that nearby vectors receive much larger scores than distant ones. The difficulty is that there may be many close keys, resulting in a diffuse attention distribution, whereas a sparse attention algorithm requires the attention distribution to be concentrated. Our main idea is to randomly perturb the key matrix so that one close key dominates all others with high probability. Concretely, we create a small number of copies of the key matrix and randomly shift each key by an exponentially distributed amount in a fixed direction. We truncate these shifts so that all entries remain bounded by $B = \mathcal { O } ( \sqrt { \log n } )$ and far vectors continue to receive negligible attention. We show that, with high probability, at least one perturbed copy contains a key corresponding to an approximate closest neighbor of the query while receiving at least $\textstyle 1 - { \frac { 1 } { n } } { \dot { > } }$ $1 - \alpha$ attention. For that copy, any subset capturing an α-fraction of the attention must contain this key, thereby revealing an approximate closest neighbor. Running the hypothetical sparse attention algorithm on all copies still yields a truly sublinear bounded-error randomized algorithm for the closest-pair search problem and thus contradicting SETH. □

This demonstrates that retrieval of relevant tokens under sparsity is fundamentally hard to guarantee uniformly over all inputs: in the mildest regime beyond the reach of highly accurate attention approximation with near-linear runtime, no algorithm can uniformly and efficiently identify a small set of tokens receiving substantial attention under future queries, even when attention is highly concentrated and even when allowing arbitrary polynomial-time preprocessing of the KV cache. Consequently, efficient sparse attention must exploit additional structure of the inputs beyond sparsity itself.

## 6 CONCLUSION

Motivated by the computational cost of Transformers, we study whether attention can be approxi mated efficiently with uniform guarantees over all inputs. Under standard complexity-theoretic assumptions, we identify that the known tractable regimes mark the limit of what is possible: outside the regimes of existing algorithms with near-linear runtime and vanishingly small error, approximating attention with a nontrivial additive or relative guarantee uniformly over all inputs requires essentially quadratic time. Interestingly, this barrier persists during autoregressive generation after arbitrary polynomial-time preprocessing of the KV cache, and then even for retrieval under sparsity. While our results make substantial improvements in algorithms with uniform worst-case guarantees unlikely, they do not rule out algorithms that perform well empirically or obtain strong guarantees under structural assumptions on the inputs. In fact, our lower bounds substantiate the need for such assumptions: beyond the already tractable regimes, exploiting input structure is genuinely necessary to obtain meaningful guarantees for fast approximate attention.

## ACKNOWLEDGMENTS

Funded by the European Union. This work has received funding from the European High Performance Computing Joint Undertaking (JU) and from the German Federal Ministry of Research, Technology and Space (BMFTR), the Ministry of Culture and Science of North Rhine-Westphalia (MKW NRW) and the Hessian Ministry of Science and Research, Arts and Culture (HMWK) under grant agreement No 101250682. LH was supported by an RWTH Research Ambassador Scholarship. CAA is supported by a Schmidt Science Fellowship.

## AI USE STATEMENT

In the preparation of this paper, large language models were used to polish the authors’ original writing. All scientific content was written and reviewed by the authors.

## REFERENCES

Maryam Aliakbarpour, Vladimir Braverman, Junze Yin, and Haochen Zhang. Support basis: Fast attention beyond bounded entries. In Proceedings of The 29th International Conference on Artificial Intelligence and Statistics, volume 300 of Proceedings of Machine Learning Research, pp. 325–333. PMLR, 2026. URL https://proceedings.mlr.press/v300/ aliakbarpour26a.html.

Josh Alman and Zhao Song. Fast attention requires bounded entries. In Advances in Neural Information Processing Systems, volume 36, pp. 63117–63135, 2023. doi: 10.52202/075280-2755.

Iz Beltagy, Matthew E Peters, and Arman Cohan. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150, 2020. URL https://arxiv.org/abs/2004.05150.

Beidi Chen, Tri Dao, Eric Winsor, Zhao Song, Atri Rudra, and Christopher Re. Scatterbrain: Uni-´ fying sparse and low-rank attention. In Advances in Neural Information Processing Systems, volume 34, pp. 17413–17426, 2021. URL https://proceedings.neurips.cc/paper/ 2021/hash/9185f3ec501c674c7c788464a36e7fb3-Abstract.html.

Zhuoming Chen, Ranajoy Sadhukhan, Zihao Ye, Yang Zhou, Jianyu Zhang, Niklas Nolte, Yuandong Tian, Matthijs Douze, Leon Bottou, Zhihao Jia, and Beidi Chen. MagicPIG: LSH sampling for efficient LLM generation. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ 6d50d824ae819d5a961c1d8edc15e833-Paper-Conference.pdf.

Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. Generating long sequences with Sparse Transformers. arXiv preprint arXiv:1904.10509, 2019. URL https://arxiv.org/abs/ 1904.10509.

Krzysztof Marcin Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarl´ os, Peter Hawkins, Jared Quincy Davis, Afroz Mohiuddin, Łukasz Kaiser,´ David Benjamin Belanger, Lucy J. Colwell, and Adrian Weller. Rethinking attention with Performers. In International Conference on Learning Representations, 2021. URL https: //openreview.net/forum?id=Ua6zuk0WRH.

Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/file/ 98ed250b203d1ac6b24bbcf263e3d4a7-Paper-Conference.pdf.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 4171–4186, 2019. doi: 10.18653/v1/N19-1423.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021. URL https: //openreview.net/forum?id=YicbFdNTTy.

Feyza Duman Keles, Pruthuvi Mahesakya Wijewardena, and Chinmay Hegde. On the computational complexity of self-attention. In Proceedings of The 34th International Conference on Algorith mic Learning Theory, volume 201 of Proceedings of Machine Learning Research, pp. 597–619. PMLR, 2023. URL https://proceedings.mlr.press/v201/duman-keles23a. html.

Anmol Gulati, James Qin, Chung-Cheng Chiu, Niki Parmar, Yu Zhang, Jiahui Yu, Wei Han, Shibo Wang, Zhengdong Zhang, Yonghui Wu, and Ruoming Pang. Conformer: Convolution-augmented transformer for speech recognition. In Interspeech 2020, pp. 5036–5040, 2020. doi: 10.21437/ Interspeech.2020-3015.

Shreya Gupta, Boyang Huang, Barna Saha, Yinzhan Xu, and Christopher Ye. Subquadratic algorithms and hardness for attention with any temperature. In International Conference on Learning Representations, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ 8a01099096c85890b1d1aff3c6b4ea56-Paper-Conference.pdf.

Insu Han, Rajesh Jayaram, Amin Karbasi, Vahab Mirrokni, David Woodruff, and Amir Zandieh. HyperAttention: Long-context attention in near-linear time. In International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/file/ ab5aa940590399350401c57cdf52ce78-Paper-Conference.pdf.

Russell Impagliazzo and Ramamohan Paturi. On the complexity of k-SAT. Journal of Computer and System Sciences, 62(2):367–375, 2001. doi: 10.1006/jcss.2000.1727.

Nikita Kitaev, Łukasz Kaiser, and Anselm Levskaya. Reformer: The efficient transformer. In International Conference on Learning Representations, 2020. URL https://openreview. net/forum?id=rkgNKkHtvB.

Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. SnapKV: LLM knows what you are looking for before generation. In Advances in Neural Information Processing Systems, volume 37, pp. 22947–22970, 2024. doi: 10.52202/079017-0722.

Yang Lin, Tianyu Zhang, Peiqin Sun, Zheng Li, and Shuchang Zhou. FQ-ViT: Post-training quantization for fully quantized vision transformer. In Proceedings of the Thirty-First International Joint Conference on Artificial Intelligence, pp. 1173–1179, 2022. doi: 10.24963/ijcai.2022/164.

Di Liu, Meng Chen, Baotong Lu, Huiqiang Jiang, Zhenhua Han, Qianxi Zhang, Qi Chen, Chengruidong Zhang, Bailu Ding, Kai Zhang, Chen Chen, Fan Yang, Yuqing Yang, and Lili Qiu. RetrievalAttention: Accelerating long-context LLM inference via vector retrieval. In Advances in Neural Information Processing Systems, volume 38, pp. 54358–54385, 2025. doi: 10.52202/085713-1816.

Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. KIVI: A tuning-free asymmetric 2bit quantization for KV cache. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 32332–32344. PMLR, 2024. URL https: //proceedings.mlr.press/v235/liu24bz.html.

Paul Michel, Omer Levy, and Graham Neubig. Are sixteen heads really better than one? In Advances in Neural Information Processing Systems, volume 32, pp. 14014–14024, 2019. URL https://proceedings.neurips.cc/paper\_files/paper/2019/ hash/2c601ad9d2ff9bc8b282670cdd54f69f-Abstract.html.

Alexander Rives, Joshua Meier, Tom Sercu, Siddharth Goyal, Zeming Lin, Jason Liu, Demi Guo, Myle Ott, C. Lawrence Zitnick, Jerry Ma, and Rob Fergus. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. Proceedings of the National Academy ofSciences, 118(15):e2016239118, 2021. doi: 10.1073/pnas.2016239118.

Aurko Roy, Mohammad Saffar, Ashish Vaswani, and David Grangier. Efficient content-based sparse attention with Routing Transformers. Transactions ofthe Associationfor Computational Linguistics, 9:53–68, 2021. doi: 10.1162/tacl a 00353.

Aviad Rubinstein. Hardness of approximate nearest neighbor search. In Proceedings of the 50th Annual ACM SIGACT Symposium on Theory ofComputing, pp. 1260–1268, 2018. doi: 10.1145/ 3188745.3188916.

Tobias Schroder and Lester Mackey. WildCat: Near-linear attention in theory and practice. In¨ Proceedings of the 43rd International Conference on Machine Learning, 2026. URL https: //openreview.net/forum?id=lfqyLp4hZm.

Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, and Song Han. QUEST: Query-aware sparsity for efficient long-context LLM inference. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 47901–47911. PMLR, 2024. URL https://proceedings.mlr.press/ v235/tang24l.html.

Yi Tay, Dara Bahri, Liu Yang, Donald Metzler, and Da-Cheng Juan. Sparse Sinkhorn attention. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pp. 9438–9447. PMLR, 2020. URL https: //proceedings.mlr.press/v119/tay20a.html.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, pp. 5998–6008, 2017. URL https://proceedings.neurips.cc/paper\_files/paper/2017/hash/ 3f5ee243547dee91fbd053c1c4a845aa-Abstract.html.

Apoorv Vyas, Angelos Katharopoulos, and Franc¸ois Fleuret. Fast transformers with clustered attention. In Advances in Neural Information Processing Systems, volume 33, pp. 21665–21674, 2020. URL https://proceedings.neurips.cc/paper/2020/ hash/f6a8dd1c954c8506aadc764cc32b895e-Abstract.html.

Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, and Hao Ma. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768, 2020. URL https://arxiv.org/ abs/2006.04768.

Yunyang Xiong, Zhanpeng Zeng, Rudrasis Chakraborty, Mingxing Tan, Glenn Fung, Yin Li, and Vikas Singh. Nystromformer: A Nystr ¨ om-based algorithm for approximating self-attention. ¨ Proceedings of the AAAI Conference on Artificial Intelligence, 35(16):14138–14148, 2021. doi: 10.1609/aaai.v35i16.17664.

Manzil Zaheer, Guru Guruganesh, Kumar Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontan˜on, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, and Amr Ahmed. Big Bird:´ Transformers for longer sequences. In Advances in Neural Information Processing Systems, volume 33, pp. 17283–17297, 2020. URL https://proceedings.neurips.cc/paper/ 2020/hash/c8512d142a2d849725f31a9a7a361ab9-Abstract.html.

Amir Zandieh, Insu Han, Majid Daliri, and Amin Karbasi. KDEformer: Accelerating transformers via kernel density estimation. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pp. 40605–40623. PMLR, 2023. URL https://proceedings.mlr.press/v202/zandieh23a.html.

Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. Proceedings ofthe AAAI Conference on Artificial Intelligence, 35(12):11106–11115, 2021. doi: 10.1609/aaai. v35i12.17325.

## A PROOFS

This appendix contains the formal proofs of Theorems 3.7, 4.1 and 5.1. Section A.1 presents a variant of the approximate closest pair problem, to which all our results reduce, and connects it to the lower bounds of Rubinstein (2018). Section A.2 shows how even a coarse approximation of attention can be used to identify approximate closest pairs, which Section A.3 uses to prove Theorems 3.7 and 4.1. Similarly, Section A.4 shows how sparse attention retrieval can be used to identify approximate closest pairs, which Section A.5 uses to prove Theorem 5.1.

## A.1 SOURCE PROBLEM

We prove all our results by showing that a fast algorithm for attention with a given approximation guarantee would ultimately give a fast algorithm for the approximate closest pairs problem below, which Rubinstein (2018) proves is impossible unless SETH fails.

Definition 3.5 (Closest pair problem). For $n , d \in \mathbb { N }$ and $\varepsilon > 0$ , the problem ${ \mathsf { C P } } ( n , d , \varepsilon )$ is the following. Given sets $\{ a _ { 1 } , \dotsc , a _ { n } \} , \{ b _ { 1 } , \dotsc , b _ { n } \} \subseteq \{ 0 , 1 \} ^ { d } , f n d i ^ { \star } , j ^ { \star } \in [ n ]$ such that

$$
\| a _ { i ^ { \star } } - b _ { j ^ { \star } } \| _ { 0 } \leq ( 1 + \varepsilon ) \operatorname* { m i n } _ { i , j \in [ n ] } \| a _ { i } - b _ { j } \| _ { 0 } .
$$

Proposition 3.6 (Rubinstein (2018, Thm. 1.1)). Fix $q > 0 .$ . Assuming SETH, there exist $C > 0$ and $\varepsilon \in ( 0 , 1 )$ such that ${ \mathsf { C P } } ( n , C \log n , \varepsilon )$ cannot be solved in $\mathcal { O } ( n ^ { 2 - q } )$ time.

Remark A.1. Although Proposition 3.6, as stated, only concerns deterministic algorithms it also holds against bounded-error randomized algorithms under randomized SETH. Indeed, both the standard reduction from k-SAT to Orthogonal Vectors and the subsequent reduction of Rubinstein (2018) to CP are deterministic. Consequently, a bounded-error randomized algorithm for CP with truly subquadratic runtime would contradict SETH as stated in Hypothesis 3.3.

For convenience, we define the decision variant of this problem with preprocessing. We also follow Alman & Song (2023) in working with inputs of balanced Hamming weights, that is, vectors in dimension 2d that have Hamming weight d. To that end, define, for even d $\in \mathbb { N } .$ , the slice $\mathbb { M } _ { d } : =$ $\{ v \in \{ 0 , 1 \} ^ { d } \mid \| v \| _ { 0 } = d / 2 \}$ of the Boolean cube with balanced Hamming weight.

Definition A.2 (Preprocessed Gap Closest Pair). For $n , m , d \ \in \ \mathbb { N }$ and $\varepsilon \ > \ 0$ , the problem F ${ } ^ { \mathsf { o } } \mathsf { r e C P } ( n , m , d , \varepsilon )$ is the following. Given a threshold $t \in [ 0 , d ]$ and set $\mathbb { B } = \left\{ b _ { 1 } , \ldots , b _ { n } \right\} \subseteq \mathbb { M } _ { d }$ preprocess them. Then, given vectors ${ \pmb a } _ { 1 } , \ldots , { \pmb a } _ { m } \in \mathbb { M } _ { d }$ , decide if

$$
\operatorname* { m i n } _ { i \in [ m ] , ~ j \in [ n ] } \| a _ { i } - b _ { j } \| _ { 0 } \leq t \qquad o r \qquad \operatorname* { m i n } _ { i \in [ m ] , ~ j \in [ n ] } \| a _ { i } - b _ { j } \| _ { 0 } > ( 1 + \varepsilon ) t ,
$$

promised that one ofthe two holds.

Solving this problem much more efficiently than checking all mn possible pairs remains hard even for bounded-error randomized algorithm with polynomial preprocessing and under the restriction to balanced inputs, as the following proposition shows. The proof effectively combines the arguments behind Rubinstein (2018, Cor. 1.3) and Alman & Song (2023, Rem. 4.4) together with a standard boosting trick for randomization.

Proposition A.3 (Balanced gap version of Rubinstein (2018, Cor. 1.3)). Fix $k , q \ > \ 0 .$ . Suppose that SETH holds. Then, there exist $C , \varepsilon > 0$ such that $\mathsf { P r e C P } ( n , m ( n ) , d , \varepsilon )$ cannot be solved, not even by a bounded-error randomized algorithm, with $\mathcal { O } ( n ^ { k } )$ preprocessing time and $\mathcal { O } ( m ( n ) n ^ { 1 - q } )$ query time,for $d : = C$ log n and anyfunction $m : \mathbb { N }  \mathbb { N }$ with $m ( n ) \in [ 2 n ]$

Proof. Assume w.l.o.g. that $k > 1$ and $q \leq 1 / 4$ . Let $\gamma : = 1 / ( 2 k )$ and $\bar { q } : = q \gamma / 2$ . By Proposition 3.6, there exist $C > 0$ and $\varepsilon \in ( 0 , 1 )$ such that ${ \mathsf { C P } } ( n , \textstyle { \frac { \gamma C } { 2 } } \log n , \varepsilon )$ cannot be solved in $\mathcal { O } ( n ^ { 2 - \bar { q } } )$ time. Suppose, for contradiction, that $\mathsf { P r e C P } ( n , m ( n ) , d , \varepsilon )$ can be solved by a bounded-error randomized algorithm, for some function $m : \mathbb { N }  \mathbb { N }$ with $m ( n ) \in [ 2 n ]$ , with $\mathcal { O } ( n ^ { k } )$ preprocessing time and $\mathcal { O } ( m ( n ) n ^ { 1 - q } )$ query time. We construct an algorithm for $\mathsf { C P } ( n , \gamma d / 2 , \varepsilon )$ ) running in $\mathcal { O } ( n ^ { 2 - \bar { q } } )$ time, contradicting Proposition 3.6. To that end, let sets $\mathbb { A } = \{ \pmb { a } _ { 1 } , \dots , \pmb { a } _ { n } \} \subseteq \{ 0 , 1 \} ^ { \gamma d / 2 }$ and $\mathbb { B } = \{ b _ { 1 } , \ldots , b _ { n } \} \subseteq \{ 0 , 1 \} ^ { \gamma d / 2 }$ be inputs to ${ \mathsf { C P } } ( n , \textstyle { \frac { \gamma C } { 2 } } \log n , \varepsilon )$ . We replace them with the balanced sets $\widehat { \mathbb { A } } : = \{ \widehat { \pmb { a } } _ { 1 } , \dots , \widehat { \pmb { a } } _ { n } \} \subseteq \mathbb { M } _ { \gamma d }$ and $\widehat { \mathbb { B } } : = \{ \widehat { b } _ { 1 } , . . . , \widehat { b } _ { n } \} \subseteq \mathbb { M } _ { \gamma d }$ , where $\widehat { \pmb x } : = ( \pmb x , \mathbf 1 - \pmb x ) \in$ $\mathbb { M } _ { \gamma d } ,$ so that $\begin{array} { r } { \| \pmb { a } - \pmb { b } \| _ { 0 } = \frac 1 2 \| \widehat { \pmb { a } } - \widehat { \pmb { b } } \| _ { 0 } } \end{array}$

We now devise the algorithm. Let $r : = n ^ { \gamma }$ and $s : = m ( r )$ so that $\gamma d = C \log r$ . Partition $\widehat { \mathbb { B } }$ into sets $\widehat { \mathbb { B } } _ { 1 } , \ldots , \widehat { \mathbb { B } } _ { \lceil n / r \rceil }$ , each of size at most $r _ { \ast }$ Then, for each threshold $t \in \{ 0 , \ldots , \gamma d \}$ in increasing order, do the following. Apply the assumed preprocessing algorithm for PreCP independently $L : =$ $C _ { L }$ log n times to every set $\widehat { \mathbb { B } } _ { j }$ separately, padding it to r separate vectors by duplication when necessary. Additionally, partition $\widehat { \mathbb { A } }$ into sets $\widehat { \mathbb { A } } _ { 1 } , \ldots , \widehat { \mathbb { A } } _ { \lceil n / s \rceil }$ , each of size at most s. On each preprocessed subset ${ \widehat { \mathbb { B } } } _ { j } .$ , independently run the corresponding $L$ decoding algorithms on each query bulk $\widehat { \mathbb { A } } _ { i } .$ , again padding them to $s = m ( r )$ vectors when necessary, and use their majority answer. That is, accept if at least half of the independent invocations of the decoding algorithm accept.

Choosing $C _ { L }$ as a sufficiently large constant guarantees that, on every promised input, each majority answer is incorrect with probability at most $n ^ { - 4 }$ . Overall, we compute at most $\mathcal { O } ( ( \gamma d +$ $\ \bar { 1 } ) ( \dot { n } / r ) ( n / s ) ) = \mathcal { O } ( n ^ { 2 } \log n )$ such answers. Hence, by a union bound, with probability at least

$$
1 - { \mathcal { O } } ( n ^ { 2 } \log n ) n ^ { - 4 } \geq { \frac { 2 } { 3 } } ,
$$

all majority answers on promised inputs are correct simultaneously. Condition on that event and consider the smallest t for which the majority answer accepts on some ${ \widehat { \mathbb { B } } } _ { j }$ and $\widehat { \mathbb { A } } _ { i }$ . Then,

$$
\operatorname* { m i n } _ { \widehat { a } \in \widehat { \mathbb { A } } _ { i } , \widehat { b } \in \widehat { \mathbb { B } } _ { j } } \| \widehat { \pmb { a } } - \widehat { { b } } \| _ { 0 } \leq ( 1 + \varepsilon ) t \leq ( 1 + \varepsilon ) \operatorname* { m i n } _ { \widehat { \pmb { a } } \in \widehat { \mathbb { A } } , \widehat { { \pmb { b } } } \in \widehat { \mathbb { B } } } \| \widehat { \pmb { a } } - \widehat { { b } } \| _ { 0 } ,
$$

where the first inequality uses that min $\hat { \pmb { a } } \in \widehat { \mathbb { A } } _ { i } , \widehat { \pmb { b } } \in \widehat { \mathbb { B } } _ { i } \ \lVert \widehat { \pmb { a } } - \widehat { \pmb { b } } \rVert _ { 0 } > ( 1 + \varepsilon ) t$ would force the decoding algorithm to reject by the problem definition, and the second inequality uses that the algorithm must accept once $\begin{array} { r } { t = \operatorname* { m i n } _ { \widehat { \pmb { a } } \in \widehat { \mathbb { A } } . \widehat { \pmb { b } } \in \widehat { \mathbb { B } } } \| \widehat { \pmb { a } } - \widehat { \pmb { b } } \| _ { 0 } } \end{array}$ . Consequently, computing arg $\operatorname* { m i n } _ { \widehat { \pmb { a } } \in \widehat { \mathbb { A } } _ { i } , \widehat { \pmb { b } } \in \widehat { \mathbb { B } } _ { j } }$ gives a pair $( a , b )$ for the original input with

$$
| | a - b | | _ { 0 } = \frac { 1 } { 2 } | | \widehat { a } - \widehat { b } | | _ { 0 } \leq \frac { 1 } { 2 } ( 1 + \varepsilon ) \operatorname* { m i n } _ { \widehat { a ^ { \prime } } \in \widehat { \mathbb { A } } , \widehat { b ^ { \prime } } \in \widehat { \mathbb { B } } } | | \widehat { a } ^ { \prime } - \widehat { b ^ { \prime } } | | _ { 0 } = ( 1 + \varepsilon ) \operatorname* { m i n } _ { a ^ { \prime } \in \widehat { \mathbb { A } } , b ^ { \prime } \in \mathbb { B } } | | a ^ { \prime } - b ^ { \prime } | | _ { 0 }
$$

satisfying the output condition of CP. Since we only need to search the small subsets $\widehat { \mathbb { A } } _ { i }$ and ${ \widehat { \mathbb { B } } } _ { j }$ , such a pair can be computed by brute force in $\mathcal { O } ( r s d ) \leq \mathcal { O } ( r ^ { 2 } d ) \leq \mathcal { O } ( n \log n )$ time.

Consequently, we obtain an algorithm with bounded error for $\mathsf { C P } ( n , \gamma d / 2 , \varepsilon )$ . For each threshold, the preprocessing takes $\mathcal { O } ( L ^ { \frac { r ^ { k } n } { r } } ) \leq \mathcal { O } ( n ^ { 3 / 2 } \log n )$ time and the decoding takes $\begin{array} { r } { \mathcal { O } ( L ^ { \frac { n } { r } \frac { n } { s } s r ^ { 1 - q } } ) \leq } \end{array}$ $\mathcal { O } ( n ^ { 2 - q \gamma } \log n )$ time. Repeating for all $\gamma d + 1$ thresholds adds a factor ${ \mathcal { O } } ( d ) = { \mathcal { O } } ( \log { \dot { n } } )$ . Thus, the algorithm runs in $\mathcal { O } ( n ^ { 2 - q \gamma / 2 } )$ time, contradicting Proposition 3.6 together with Remark A.1. □

We also use the fact that the problem is easy to solve when the decision threshold t is small. To prove this, we apply the argument of Alman & Song (2023)[Lem. B.1, Case 1] to our particular problem formulation to give a brute-force algorithm searching the immediate vicinity of every preprocessed vector, resulting in $\mathcal { O } ( m n ^ { 1 - q } )$ runtime when the search space is sufficiently small.

Lemma A.4. Fix $C _ { d } ~ > ~ 0$ and $q ~ \in ~ ( 0 , 1 )$ . There exists a constant $c \in \mathsf { \Gamma } ( 0 , C _ { d } / 2 )$ such that $\mathsf { P r e C P } ( n , m , C _ { d } \log n , \varepsilon )$ can be solved with $\mathcal { O } ( n d )$ preprocessing time and $\dot { \mathcal { O } } ( m n ^ { 1 - \dot { q } } )$ decoding time whenever the input threshold t satisfies $t \leq c$ log n.

Proof. Let $d : = C _ { d } \log n$ . Choose any $c \in ( 0 , C _ { d } / 2 )$ such that c log $\begin{array} { r } { \left( \frac { e C _ { d } } { c } \right) < 1 - q } \end{array}$ . Such a choice is always possible since lim $\begin{array} { r } { \_ } { c \oslash c \log \left( \frac { e C _ { d } } { c } \right) \ = 0 } \end{array}$ . Now, let $\mathbb { B } : = \{ b _ { 1 } , \ldots , b _ { n } \} \subseteq \mathbb { M } _ { d }$ be the input set to $\mathsf { P r e C P } ( n , m , d , \varepsilon )$ and ${ \pmb a } _ { 1 } , \dots , { \pmb a } _ { m } \in \mathbb { M } _ { d }$ the query revealed after preprocessing. We give a brute-force algorithm for deciding this input, following the Hamming-ball enumeration of Alman $\&$ Song (2023, Lem. B.1, Case 1). Preprocess B by storing it in a trie. This takes $\mathcal { O } ( n d )$ time. Then, enumerate, for all $i \in [ m ]$ separately, all vectors $\pmb { b } \in \overline { { \{ 0 , 1 \} ^ { d } } }$ satisfying $\| \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \| \mathbf { } \mathbf { } \mathbf { } \mathbf { } \| \mathbf { } \mathbf { } \mathbf { } \| \mathbf { } \mathbf { } \| \mathbf { } \big (  \mathbf { } \mathbf { } \cdot \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \big ) \| \big (  \mathbf { } \mathbf { } \big ) \| \big (  \mathbf { } \mathrm { ~ \cdot ~ } \mathbf { } \mathbf { } \big ) \| \big (  \mathbf { } \big ) \| \big (  \mathbf { } \big ) \| \big (  \mathbf { } \big ) \| \big (  \mathbf { } \big ) \| \big (  \mathbf { } \big )$ and accept if any such b belongs to B. This correctly decides the instance: if some $\mathbf { a } _ { i }$ has a close partner in $\mathbb { B } ,$ exhaustive enumeration finds $\mathbf { i t } ,$ and if no $\mathbf { a } _ { i }$ has such a partner, no partner is found. Since $t \leq c \log n < d / 2$ , the decoding time is

$$
\begin{array} { r l } { \mathcal { O } \left( m d \displaystyle \sum _ { k = 0 } ^ { \lfloor t \rfloor } { \binom { d } { k } } \right) \leq \mathcal { O } \left( m d \left( \displaystyle \frac { e d } { t } \right) ^ { t } \right) \leq \mathcal { O } \left( m \log n \left( \displaystyle \frac { e C _ { d } \log n } { c \log n } \right) ^ { c \log n } \right) } & { } \\ { \leq \mathcal { O } \left( m n ^ { c \log ( e C _ { d } / c ) + o ( 1 ) } \right) \leq \mathcal { O } \left( m n ^ { 1 - q } \right) . } \end{array}
$$

## A.2 FINDING CLOSE PAIRS FROM A COARSE ATTENTION APPROXIMATION

Following the general strategy of Alman & Song (2023), we encode a closest pair instance as an attention instance so that approximating the attention output reveals an approximate closest pair. Deviating from their construction, however, we ensure that recovering such a pair does not require a high-precision attention approximation. To achieve this, we shift the attention scores relative to a collection of dummy keys so that the contribution of a close pair dominates the output, while the combined contribution of all far pairs remains negligible. Importantly, the keys and values depend only on one of the two closest pair sets, whereas the queries depend only on the other. The construction is therefore compatible with preprocessing the keys and values before the query vectors are revealed. The following lemma formalizes these properties.

Lemma A.5. $F i x \varepsilon , c , C _ { d } > 0$ . There exists $C _ { B } > 0$ such that thefollowing holdsfor all sufficiently large $n \in \mathbb { N }$ and $N : = 2 n , d : = 2 C _ { d } \log n , D : = 4 C _ { d } \log N$ , and $\mathbf { \bar { \Sigma } } B : = C _ { B } \sqrt { \log N }$

Choose $t \geq$ c log n with $d \geq ( 1 + \varepsilon ) t ,$ , a set of vectors $\mathbb { B } = \left\{ b _ { 1 } , \ldots , b _ { n } \right\} \subseteq \mathbb { M } _ { d } ,$ and a sequence $\pmb { a } _ { 1 } , \ldots , \pmb { a } _ { m } \in \mathbb { M } _ { d } .$ One can construct matrices ${ \pmb { K } } \in [ - B , \dot { B ] } ^ { \tilde { N } \times D } a n d \mathbf { \bar { V } } \in [ 0 , 1 ] ^ { N \times D }$ independent of $\pmb { a } _ { 1 } , \ldots , \pmb { a } _ { m }$ and in $\mathcal { O } ( n D )$ time, and another matrix $\dot { \boldsymbol { Q } } \in [ - B , B ] ^ { m \times \bar { D } }$ in $\mathcal { O } ( m D )$ time such that $\mathrm { A t t } ( Q , K , V ) = y e _ { 1 } ^ { \top }$ for some $\pmb { y } \in \mathbb { R } ^ { m }$ satisfying, for every $i \in [ m ]$

$$
y _ { i } \geq \frac { n } { n + 1 } \qquad i f \quad \operatorname* { m i n } _ { j \in [ n ] } \| { \pmb a } _ { i } - { \pmb b } _ { j } \| _ { 0 } \leq t , \quad a n d
$$

$$
y _ { i } \leq \frac { 1 } { n + 1 } \qquad i f \quad \operatorname* { m i n } _ { j \in [ n ] } \| \pmb { a } _ { i } - \pmb { b } _ { j } \| _ { 0 } \geq ( 1 + \varepsilon ) t .
$$

Proof. Define

$$
\rho : = \sqrt { D / ( 2 d ) } , \qquad \beta : = \frac { 1 2 d \log n } { \varepsilon t } , \qquad \mathrm { a n d } \qquad \mu : = \frac { 1 } { 2 } - \left( \frac { 1 } { 2 } + \frac { \varepsilon } { 3 } \right) \frac { t } { d }
$$

and note that

$$
0 \leq \frac { \varepsilon } { 6 ( 1 + \varepsilon ) } \leq \mu \leq \frac { 1 } { 2 }
$$

since $t / d \leq 1 / ( 1 + \varepsilon )$ by assumption.

We construct the key and value matrices as

$$
\begin{array} { r } { \boldsymbol { K } : = \rho \sqrt { \beta } \left[ \begin{array} { c c c c c c c } { \boldsymbol { b } _ { 1 } ^ { \top } } & { - \boldsymbol { \mu } \mathbf { 1 } _ { d } ^ { \top } } & { \mathbf { 0 } _ { D - 2 d } ^ { \top } } \\ { \vdots } & { \vdots } & { \vdots } \\ { \boldsymbol { b } _ { n } ^ { \top } } & { - \boldsymbol { \mu } \mathbf { 1 } _ { d } ^ { \top } } & { \mathbf { 0 } _ { D - 2 d } ^ { \top } } \\ { \mathbf { 0 } _ { d } ^ { \top } } & { \mathbf { 0 } _ { d } ^ { \top } } & { \mathbf { 0 } _ { D - 2 d } ^ { \top } } \\ { \vdots } & { \vdots } & { \vdots } \\ { \mathbf { 0 } _ { d } ^ { \top } } & { \mathbf { 0 } _ { d } ^ { \top } } & { \mathbf { 0 } _ { D - 2 d } ^ { \top } } \end{array} \right] \in \mathbb { R } ^ { N \times D } \qquad \mathrm { a n d } \qquad \boldsymbol { V } : = \left[ \begin{array} { c c c c c c } { 1 } & { 0 } & { \cdots } & { 0 } \\ { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { 1 } & { 0 } & { \cdots } & { 0 } \\ { 0 } & { 0 } & { \cdots } & { 0 } \\ { \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { 0 } & { 0 } & { \cdots } & { 0 } \end{array} \right] \in \mathbb { R } ^ { N \times D } } \end{array}
$$

and the query matrix as

$$
\pmb { Q } : = \rho \sqrt { \beta } \left[ \begin{array} { c c c } { \pmb { a } _ { 1 } ^ { \top } } & { \mathbf { 1 } _ { d } ^ { \top } } & { \mathbf { 0 } _ { D - 2 d } ^ { \top } } \\ { \vdots } & { \vdots } & { \vdots } \\ { \pmb { a } _ { m } ^ { \top } } & { \mathbf { 1 } _ { d } ^ { \top } } & { \mathbf { 0 } _ { D - 2 d } ^ { \top } } \end{array} \right] \in \mathbb { R } ^ { m \times D } .
$$

Next, we choose $C _ { B }$ so that the constructed matrices satisfy the required entry bounds. Since $t \geq$ c log $n , d = 2 C _ { d }$ log n, and $\rho ^ { 2 } = \log ( 2 n ) / \log n \leq 2 ,$

$$
\rho ^ { 2 } \beta = \frac { \log N } { \log n } \frac { 1 2 d \log n } { \varepsilon t } \leq \frac { 2 4 C _ { d } \log N } { \varepsilon c }
$$

Hence, there exists a constant $C _ { B } \ > \ 0$ , depending only on $C _ { d } , \varepsilon .$ , and $c ,$ such that $\rho \sqrt { \beta } \ \leq$ $C _ { B } \sqrt { \log N }$ . Since $0 \leq \mu \leq 1 / 2$ , it follows that $\| Q \| _ { \ell _ { \infty } } \leq \dot { B }$ and $\| \kappa \| _ { \ell _ { \infty } } \leq B$

We now show that the attention output separates close from far pairs as claimed. To that end, let $Y : = \mathrm { A t t } ( Q , K , V )$ . For $i \in [ m ]$ and $j \in [ n ]$ , the logit between the i-th query and the j-th non-dummy key is

$$
\begin{array} { l } { \displaystyle \frac { \langle Q _ { i } , K _ { j } \rangle } { D } = \frac { \rho ^ { 2 } \beta } { D } \left( \langle a _ { i } , b _ { j } \rangle - \mu d \right) = \frac { \beta } { 2 d } \left( \langle a _ { i } , b _ { j } \rangle - \mu d \right) } \\ { \displaystyle \quad = \frac { \beta } { 2 d } \left( \frac { \| a _ { i } \| _ { 2 } ^ { 2 } + \| b _ { j } \| _ { 2 } ^ { 2 } - \| a _ { i } - b _ { j } \| _ { 2 } ^ { 2 } } { 2 } - \mu d \right) } \\ { \displaystyle \quad = \frac { \beta } { 2 d } \left( \frac { d - \| a _ { i } - b _ { j } \| _ { 0 } } { 2 } - \mu d \right) } \\ { \displaystyle \quad = \frac { \beta } { 4 } \left( 1 - \frac { \| a _ { i } - b _ { j } \| _ { 0 } } { d } \right) - \frac { \beta \mu } { 2 } , } \end{array}\tag{1}
$$

where the third equality uses that $a _ { i }$ and $b _ { j }$ both have Hamming weight $d / 2 .$ For $i \in [ m ]$ , let

$$
s _ { i } : = \sum _ { j = 1 } ^ { n } \exp \left( \frac { \langle Q _ { i } , K _ { j } \rangle } { D } \right)\tag{2}
$$

be the total score assigned to non-dummy keys. By the definition of $V ,$ the i-th attention output is

$$
Y _ { i } = \frac { s _ { i } } { s _ { i } + n } e _ { 1 } ^ { \top } ,\tag{3}
$$

since the $n$ dummy keys are 0 so that each of them has logit 0 and contributes score 1 and only the first column of $V$ is nonzero. Now, fix $i \in \ [ m ]$ and distinguish the following two cases from the claim. Suppose first that mi $\mathrm { n } _ { j \in [ n ] } \| a _ { i } - b _ { j } \| _ { 0 } \leq t$ . Then, Equation (1) and the definition of $\mu$ imply

$$
\frac { \langle Q _ { i } , K _ { j } \rangle } { D } \ge \frac { \beta } { 4 } \left( 1 - \frac { t } { d } \right) - \frac { \beta \mu } { 2 } = \frac { \beta \varepsilon t } { 6 d } = 2 \log ( n )
$$

for some $j \in [ n ]$ . Consequently, $s _ { i } \geq n ^ { 2 }$ and Equation (3) gives $Y _ { i , 1 } \geq n / ( n + 1 )$ . Now, suppose instead that mi $1 _ { j \in [ n ] } \| a _ { i } - b _ { j } \| _ { 0 } \geq ( 1 + \varepsilon ) t$ . Then, Equation (1) and the definition of $\mu$ imply

$$
{ \frac { \langle Q _ { i } , K _ { j } \rangle } { D } } \leq { \frac { \beta } { 4 } } \left( 1 - { \frac { ( 1 + \varepsilon ) t } { d } } \right) - { \frac { \beta \mu } { 2 } } = - { \frac { \beta \varepsilon t } { 1 2 d } } = - \log n
$$

for all $j \in [ n ]$ . Consequently, $s _ { i } \leq 1$ , and Equation (3) gives $Y _ { i , 1 } \leq 1 / ( n + 1 )$ for all $i \in [ m ]$

Finally, K and $V$ can be constructed in $\mathcal { O } ( N D )$ time and $Q$ in $\mathcal { O } ( m D )$ time by copying and padding the respective inputs. That proves the claim. □

## A.3 PROOFS OF THEOREMS 3.7 AND 4.1

Theorems 3.7 and 4.1 can now be proved by using Lemma A.5 to encode an input to the closest pair problem from Definition A.2 as an attention input whose output gives an almost-binary indicator of whether an approximate close pair exists. Consequently, a fast algorithm with even a coarse approximation would give a fast algorithm for deciding the closest pair problem, which would contradict Proposition A.3. While the proof is essentially the same for both theorems, the underlying problem formulations differ slightly. For compactness, we therefore formulate below a generalized problem formulation including both Problem 1 and 2 as special cases. We then prove that this generalized problem cannot be solved much more efficiently than by the trivial algorithm for attention, from which Theorems 3.7 and 4.1 follow immediately.

## Problem 4: Approximating attention with preprocessing (batch variant)

Fix parameters $n , m , d \in \mathbb { N }$ and $B \geq 0$ . Given matrices $K \in [ - B , B ] ^ { n \times d }$ and $V \in [ 0 , 1 ] ^ { n \times d } ;$ preprocess them. Then, given a query $Q \in [ - B , B ] ^ { m \times d }$ , the task is to compute a matrix $\mathbf { \bar { \boldsymbol { T } } } \in \mathbb { R } ^ { m \times d }$ . For this output and the exact target $Y : = \bar { \mathrm { A t t } } ( Q , K , V )$ , the problem

$$
\begin{array} { r l r l } & { \mathsf { A b s P r e B a t c h A t t } ( n , m , d , B , \eta _ { \mathrm { a b s } } ) } & & { \mathsf { r e q u i r e s } } & & { \| T - \pmb { Y } \| _ { \ell _ { \infty } } \le \eta _ { \mathrm { a b s } } , \quad \mathrm { ~ a n d } } \\ & { \mathsf { R e l P r e B a t c h A t t } _ { p } ( n , m , d , B , \eta _ { \mathrm { r e l } } ) } & & { \mathsf { r e q u i r e s } } & & { \| T - \pmb { Y } \| _ { S _ { p } } \le \eta _ { \mathrm { r e l } } \| \pmb { Y } \| _ { S _ { p } } . } \end{array}
$$

Proposition A.6. Fix $\eta _ { \mathrm { a b s } } ~ \in ~ [ 0 , 1 / 2 ) , ~ \eta _ { \mathrm { r e l } } ~ \in ~ [ 0 , 1 )$ and $k , q \ > \ 0 .$ Suppose that SETH holds. Then, there exist $C _ { d } , C _ { B } , \bar { \varepsilon } ~ > ~ \bar { 0 }$ such that neither AbsPreBatch $\mathsf { A t t } ( n , m ( n ) , d , B , \eta _ { \mathrm { a b s } } )$ nor RelPreBatc $\mathsf { \Lambda } \mathsf { A t t } _ { p } ( n , m ( n ) , d , B , \eta _ { \mathrm { r e l } } )$ can be solved with $\mathcal { O } ( n ^ { k } )$ preprocessing time and $O ( m ( n ) n ^ { 1 - q } )$ query time, $f o r d : = C$ log $n , B : = C _ { B } \sqrt { \log n } ,$ , any $p \in [ 1 , \infty ]$ , and any function $m : \mathbb { N } \overset { \cdot } {  } \mathbb { N }$ with $m ( n ) \in [ 2 n ]$

Proof. Define shorthand notation $N : = 2 n$ and $m : = m ( n )$ . Assume w.l.o.g. that $k > 1 , q \in$ $( 0 , 1 / 2 )$ , and that n is sufficiently large. We choose $C _ { d } , \varepsilon > 0$ as the constant from Proposition ${ \mathrm { A } } . 3$ so that $\mathsf { P r e C P } ( n , m , d , \varepsilon )$ cannot be solved with $\mathcal { O } ( n ^ { k } )$ preprocessing time and $\mathcal { O } ( m \bar { n } ^ { 1 - q } )$ query time for $\begin{array} { r } { d : = { \frac { C _ { d } } { 2 } } } \end{array}$ log n. Suppose, for contradiction, that either $\mathsf { A b s P r e B a t c h A t t } ( n , m , D , B , \eta _ { \mathrm { a b s } } )$ or RelPreBatc $\mathsf { \Omega } \cap \mathsf { A t t } _ { p } ( n , m , D , B , \eta _ { \mathrm { r e l } } )$ can be solved with $O ( n ^ { k } )$ preprocessing time and $\mathcal { O } ( m n ^ { 1 - q } )$ decoding time.

We construct an algorithm for $\mathsf { P r e C P } ( n , m ( n ) , d , \varepsilon )$ with $\mathcal { O } ( n ^ { k } )$ preprocessing time and $\mathcal { O } ( m n ^ { 1 - q } )$ query time, contradicting Proposition ${ \mathbf { A } } . 3 .$ To that end, let a threshold $t \in [ 0 , d ]$ and a set B = $\left\{ b _ { 1 } , \ldots , b _ { n } \right\} \subseteq \mathbb { M } _ { d }$ be an input to $\mathsf { P r e C P } ( n , m , d , \varepsilon )$ , together with queries ${ \pmb a } _ { 1 } , \ldots , { \pmb a } _ { m } \in \mathbb { M } _ { d }$ revealed after preprocessing. Applying Lemma A.4 to $C _ { d }$ and q gives a constant $c \in ( 0 , C _ { d } / 2 )$ so that we can decide the $\mathsf { P r e C P }$ instance with $\ O ( n d ) \leq \ O ( n ^ { k } )$ preprocessing time and $\mathcal { O } ( m n ^ { 1 - q } )$ decoding time if t ≤ c log n. We may there assume that $t > c \log n$ . We may further assume that $( 1 + \varepsilon ) t \leq d .$ Indeed, i $\mathrm { ~ f ~ } ( 1 + \varepsilon ) t > d ,$ , then every conceivable pair $( \breve { a } , b ) \in \{ 0 , \dot { 1 } \} ^ { d } \times { \mathbb { B } }$ has Hamming distance at most $d < ( 1 + \varepsilon ) t ,$ so the problem is trivial to decide.

We construct a KV cache $( K , V )$ and a query Q such that the PreCP instance can be decided from an attention approximation after KV cache preprocessing. After seeing only t and B, we apply Lemma $\mathsf { A } . 5$ with parameters ε, c and choose t as the threshold and B as the set. This supplies the constant $C _ { B } > 0$ and allows us to construct, during preprocessing and in $\mathcal { O } ( N D ) = \mathcal { O } ( \bar { n } \bar { \log { n } } ) \leq$ $\mathcal { O } ( n ^ { k } )$ time, matrices $K \in \mathsf { \Gamma } [ - B , B ] ^ { N \times D }$ and $\breve { V } \in \bar { [ 0 , 1 ] } ^ { N \times \breve { D } }$ , which we preprocess with the hypothetical preprocessing algorithm for attention in $\mathcal { O } ( \bar { n } ^ { k } )$ time.

After that, let the vectors $\pmb { a } _ { 1 } , \ldots , \pmb { a } _ { m }$ be revealed. Now, Lemma A.5 allows us to construct, in $\mathcal { O } ( m D ) ~ = ~ \mathcal { O } ( m \log n ) ~ \leq ~ \mathcal { O } ( m n ^ { 1 - q } )$ time, a matrix $\boldsymbol { Q } \in [ - B , B ] ^ { m \times D }$ such that $\mathbf { \nabla } \mathbf { Y } : =$ $\mathrm { A t t } ( Q , K , V ) = y e _ { 1 } ^ { \top }$ for some $\pmb { y } \in \mathbb { R } ^ { m }$ with

$$
y _ { i } \geq { \frac { n } { n + 1 } } \qquad { \mathrm { i f ~ } } \operatorname* { m i n } _ { j \in [ n ] } \| \pmb { a } _ { i } - \pmb { b } _ { j } \| _ { 0 } \leq t\tag{4}
$$

and

$$
y _ { i } \leq { \frac { 1 } { n + 1 } } \qquad { \mathrm { i f ~ } } \operatorname* { m i n } _ { j \in [ n ] } \| a _ { i } - b _ { j } \| _ { 0 } \geq ( 1 + \varepsilon ) t\tag{5}
$$

for all $i \in [ m ]$ . We use that to decide the PreCP instance. To that end, run the assumed approximate attention decoding algorithm on $Q$ and let $\pmb { T } \in \mathbb { R } ^ { m \times D }$ be its output.

Suppose first that the algorithm guarantees $\| T - Y \| _ { \ell _ { \infty } } \leq \eta _ { \mathrm { a b s } }$ . If min $\mathbf { \Phi } _ { i \in [ m ] , j \in [ n ] } \| \mathbf { \pmb { a } } _ { i } - \pmb { b } _ { j } \| _ { 0 } \leq t$ then

$$
T _ { i , 1 } \geq Y _ { i , 1 } - \eta _ { \mathrm { a b s } } \stackrel { ( 4 ) } { \geq } \frac { n } { n + 1 } - \eta _ { \mathrm { a b s } } > \frac { 1 } { 2 }
$$

for some $i \in [ m ]$ ]. If, instead, mi $1 _ { i \in [ m ] , j \in [ n ] } \| { \pmb a } _ { i } - { \pmb b } _ { j } \| _ { 0 } \geq ( 1 + \varepsilon ) t$ , then

$$
T _ { i , 1 } \leq Y _ { i , 1 } + \eta _ { \mathrm { a b s } } \leq \frac { ( 5 ) } { n + 1 } + \eta _ { \mathrm { a b s } } < \frac { 1 } { 2 }
$$

for all $i \in [ m ]$ . Thus, we can decide the PreCP instance by checking if any entry of T exceeds $1 / 2 .$ Suppose now that, instead, the algorithm guarantees $\| T - Y \| _ { S _ { p } } \leq \eta _ { \mathrm { r e l } } \| \boldsymbol { Y } \| _ { S _ { p } }$ . Since $\mathbf { Y } = y e _ { 1 } ^ { \top }$ $\begin{array} { r } { \| \pmb { Y } \| _ { \pmb { S } _ { p } } = \| \pmb { y } \| _ { 2 } \leq \sqrt { m _ { \cdot } } \operatorname { I f } \operatorname* { m i n } _ { i \in [ m ] , j \in [ n ] } \| \pmb { a } _ { i } - \pmb { b } _ { j } \| _ { 0 } \leq t , } \end{array}$ then

$$
\| T \| s _ { p } \geq \| Y \| _ { s _ { p } } - \| T - Y \| s _ { p } \geq ( 1 - \eta _ { \mathrm { r e l } } ) \| Y \| _ { s _ { p } } = ( 1 - \eta _ { \mathrm { r e l } } ) \| y \| _ { 2 } \overset { ( 4 ) } { \geq } ( 1 - \eta _ { \mathrm { r e l } } ) \frac { n } { n + 1 } .
$$

for some $i \in [ m ] , { \mathrm { I f } } .$ instead, $\begin{array} { r } { \operatorname* { m i n } _ { i \in [ m ] , j \in [ n ] } \| { \pmb a } _ { i } - { \pmb b } _ { j } \| _ { 0 } \geq ( 1 + \varepsilon ) t } \end{array}$ , then

$$
\| T \| _ { S _ { p } } \leq \| Y \| _ { S _ { p } } + \| T - Y \| _ { S _ { p } } \leq ( 1 + \eta _ { \mathrm { r e l } } ) \| Y \| _ { S _ { p } } = ( 1 + \eta _ { \mathrm { r e l } } ) \| y \| _ { 2 } \overset { ( 5 ) } { \leq } ( 1 + \eta _ { \mathrm { r e l } } ) \frac { \sqrt { m } } { n + 1 } .
$$

Thus, we can decide the PreCP instance by checking if $\begin{array} { r } { \| \pmb { T } \| _ { \mathcal { S } _ { p } } > ( 1 - \eta _ { \mathrm { r e l } } ) \frac { n } { n + 1 } } \end{array}$ . Computing $\| \pmb { T } \| _ { \mathcal { S } _ { p } }$ takes $\mathcal { O } ( m D ^ { 2 } ) = \mathcal { O } ( m \log ^ { 2 } n ) \leq \mathcal { O } ( m n ^ { 1 - q } )$ time. Either guarantee gives an algorithm for deciding the PreCP instance from T in $\mathcal { O } ( D ) = \mathcal { O } ( \log n ) \leq \mathcal { O } ( n ^ { 1 - q } )$ time. Computing ${ \mathbf { } } T ,$ , by assumption, only requires $O ( n ^ { k } )$ preprocessing time and $\mathcal { O } ( m n ^ { 1 - q } )$ decoding time. That contradicts Proposition A.3 and thereby proves the claim. □

Theorem 3.7. Fix $\eta _ { \mathrm { a b s } } \in [ 0 , \frac { 1 } { 2 } ) , \eta _ { \mathrm { r e l } } \in [ 0 , 1 )$ , and $q > 0 .$ . Assuming SETH, there exist $C _ { d } , C _ { B } > 0$ such that neither Abs $\mathfrak { t t } ( n , D , B , \eta _ { \mathrm { a b s } } )$ nor $\mathsf { R e l A t t } _ { p } ( n , D , B , \eta _ { \mathrm { r e l } } )$ can be solved in $\mathcal { O } ( n ^ { 2 - q } )$ time, for $D : = C _ { d } \log n , B : = C _ { B } \sqrt { \log n } ,$ , and any $p \in [ 1 , \infty ] .$

Proof. This follows directly from Proposition A.6 with $m ( n ) : = n$

Theorem 4.1. Fix $\eta _ { \mathrm { a b s } } ~ \in ~ [ 0 , \frac { 1 } { 2 } ) , ~ \eta _ { \mathrm { r e l } } ~ \in ~ [ 0 , 1 )$ , and $k , q \ > \ 0$ . Assuming SETH, there exist $C _ { d } , C _ { B } ~ > ~ 0$ such that neither $\mathsf { A b s P r e A t t } ( n , D , B , \eta _ { \mathrm { a b s } } )$ nor Rel $\mathsf { r e A t t } ( n , D , B , \eta _ { \mathrm { r e l } } )$ can be solved with $O ( n ^ { k } )$ preprocessing time and ${ \dot { \mathcal { O } } } ( n ^ { 1 - q } )$ decoding time, for $\dot { D } : = C _ { d }$ log n and $B : = C _ { B } \sqrt { \log n } .$

Proof. This follows directly from Proposition A.6 with $m ( n ) : = 1$

## A.4 FINDING CLOSE PAIRS FROM A SPARSE ATTENTION APPROXIMATION

We next show how a sparse attention algorithm can be used to identify approximate closest pairs. As before, we encode distances between a query vector and the preprocessed set as attention scores, with closer vectors receiving larger scores. We would then like to recover an approximate closest pair from any small set of keys carrying a constant fraction of the attention mass. For this argument to apply, however, the resulting attention instance must satisfy the sparsity promise of Problem 3. But a closest pair instance may contain many approximately closest vectors, causing the corresponding keys to receive similar scores and the attention mass to be spread among them. We overcome this by randomly perturbing the scores so that, with sufficiently high probability, the score corresponding to a close pair is amplified enough to dominate the attention. Exponential random variables provide exactly this effect. At the same time, the random perturbation must not allow a far pair to dominate by chance. We therefore truncate the exponential variables, limiting how much any score can be amplified while still allowing a close pair to separate from the rest. The following lemma shows that these truncated random variables indeed still create the required separation. Lemma A.8 then translates this separation into an attention instance in which, with high probability, an approximate closest pair carries almost all of the attention mass and can therefore be identified by a retrieval algorithm for sparse attention.

Lemma A.7. Fix $L > 0 , n \geq 2 ,$ , and $z _ { 1 } , \ldots , z _ { n } \in \mathbb { R } .$ . Sample exponential random variables $E _ { 1 } , \dots , E _ { n } \overset { \mathrm { i i d } } { \sim } \mathrm { E x p } ( \theta )$ of rate $\theta > 0$ and clip them to $\widehat { E } _ { j } : = \operatorname* { m i n } \{ E _ { j } , L \}$ . Then, for any $g > 0$

$$
\operatorname* { P r } \left[ \exists j ^ { \star } \in [ n ] : \forall j ^ { \star } \neq j \in [ n ] : ( s _ { j ^ { * } } + \widehat { E } _ { j ^ { \star } } ) - ( s _ { j } + \widehat { E } _ { j } ) \geq g \right] \geq e ^ { - \theta g } - n e ^ { - \theta L } .
$$

Proof. Define the scores $\widehat { Z } _ { j } : = s _ { j } + \widehat { E } _ { j }$ and their corresponding values before clipping $Z _ { j } : =$ $s _ { j } + E _ { j }$ . Fix an arbitrary $j ^ { \star } \in [ n ]$ and condition on $E _ { - j ^ { \star } }$ , that is, all $E _ { j }$ except $E _ { j } ,$ ⋆ . Among these, let

$$
M _ { j ^ { \star } } : = \operatorname* { m a x } _ { \substack { j \in [ n ] , j \neq j ^ { \star } } } Z _ { j }
$$

be the maximum score. Then,

$$
\begin{array} { r l } & { \operatorname* { P r } \big [ Z _ { j ^ { \star } } \geq M _ { j ^ { \star } } + g \mid E _ { - j ^ { \star } } \big ] = e ^ { - \theta \operatorname* { m a x } \{ 0 , M _ { j ^ { \star } } + g - s _ { j ^ { \star } } \} } } \\ & { \qquad \geq e ^ { - \theta g } e ^ { - \theta \operatorname* { m a x } \{ 0 , M _ { j ^ { \star } } - s _ { j ^ { \star } } \} } = e ^ { - \theta g } \operatorname* { P r } \big [ Z _ { j ^ { \star } } \geq M _ { j ^ { \star } } \mid E _ { - j ^ { \star } } \big ] . } \end{array}
$$

Taking expectations over $E _ { - j ^ { \star } }$

$$
\mathrm { P r } \left[ Z _ { j ^ { \star } } \geq M _ { j ^ { \star } } + g \right] \geq e ^ { - \theta g } \mathrm { P r } \left[ Z _ { j ^ { \star } } \geq M _ { j ^ { \star } } \right] .
$$

Summing over all possible choices of $j ^ { \star }$ and using the fact that the continuous random variables $s _ { j } + E _ { j }$ have a unique maximum almost surely,

$$
\begin{array} { r l } & { \operatorname* { P r } \Big [ \exists j ^ { \star } \in [ n ] : \forall j ^ { \star } \neq j \in [ n ] : Z _ { j ^ { \star } } - Z _ { j } \geq g \Big ] = \displaystyle \sum _ { j ^ { \star } \in [ n ] } \operatorname* { P r } \big [ Z _ { j ^ { \star } } \geq M _ { j ^ { \star } } + g \big ] } \\ & { \qquad \geq e ^ { - \theta g } \displaystyle \sum _ { j ^ { \star } \in [ n ] } \operatorname* { P r } \big [ Z _ { j ^ { \star } } \geq M _ { j ^ { \star } } \big ] \geq e ^ { - \theta g } . } \end{array}
$$

Finally, it may happen that some of the exponential random variables are clipped. Therefore, working on the even that no clipping occurs and using the union bound,

$$
\begin{array} { r l } & { \operatorname* { P r } \left[ \exists j ^ { \star } \in [ n ] : \forall j ^ { \star } \neq j \in [ n ] : \widehat { Z } _ { j ^ { \star } } - \widehat { Z } _ { j } \geq g \right] } \\ & { \geq \operatorname* { P r } \left[ \exists j ^ { \star } \in [ n ] : \forall j ^ { \star } \neq j \in [ n ] : Z _ { j ^ { \star } } - Z _ { j } \geq g \right] - \operatorname* { P r } \left[ \exists j \in [ n ] : E _ { j } > L \right] } \\ & { \geq e ^ { - \theta g } - n e ^ { - \theta L } . } \end{array}
$$

Lemma A.8. Fix $\varepsilon , c , C _ { d } > 0$ and $\gamma \in ( 0 , 1 )$ . There exists $C _ { B } > 0$ such that the following holds for all sufficiently large $n \in  { \mathbb { N } } , d : = C _ { d }$ log n, and $B : = C _ { B } \sqrt { \log n } .$

Choose $t \geq c \log n$ with $d \ge ( 1 + \varepsilon ) t , a$ set of vectors $\mathbb { B } = \left\{ b _ { 1 } , \ldots , b _ { n } \right\} \subseteq \mathbb { M } _ { d } ,$ , and a query $\pmb { a } \in \mathbb { M } _ { d } .$ One can sample matrices $\dot { \pmb { K } } \in [ - B , \dot { B } ] ^ { n \times 2 d }$ independent of a and in $\mathcal { O } ( n d )$ time, and construct a vector $q \in [ - B , B ] ^ { 2 d }$ in $\mathcal O ( d )$ time such that

$$
\operatorname* { P r } \left[ \exists j ^ { \star } \in [ n ] : \| a - b _ { j ^ { \star } } \| _ { 0 } \leq ( 1 + \varepsilon ) t { \mathrm { ~ } } a n d { \mathrm { ~ } } \operatorname { s o f t m a x } _ { j ^ { \star } } \left( K q / ( 2 d ) \right) > 1 - { \frac { 1 } { n } } \right] \geq { \frac { 1 } { 2 } } n ^ { - \gamma }
$$

if there exists $b ^ { \star } \in \mathbb { B }$ with $\| { \pmb { a } } - { \pmb { b } } ^ { \star } \| \leq t .$

Proof. Define

$$
\theta : = \frac { \gamma } { 2 } , \qquad R : = \frac { 3 } { \theta } , \qquad \mathrm { a n d } \qquad \beta : = \frac { 4 d ( R + 1 ) \log n } { \varepsilon t } .
$$

We first sample clipped exponential random variables

$$
\widehat { E } _ { j } : = \operatorname* { m i n } \{ E _ { j } , R \log n \} , \qquad \mathrm { w h e r e } \quad E _ { 1 } , \ldots , E _ { n } \overset { \mathrm { i i d } } { \sim } \mathrm { E x p } ( \theta ) .
$$

We then construct the key matrix and query vector as

$$
K : = \sqrt { \beta } \left[ \begin{array} { l l } { b _ { 1 } ^ { \top } } & { \frac { 2 \widehat { E } _ { 1 } } { \beta } \mathbf { 1 } _ { d } ^ { \top } } \\ { \vdots } & { \vdots } \\ { b _ { n } ^ { \top } } & { \frac { 2 \widehat { E } _ { n } } { \beta } \mathbf { 1 } _ { d } ^ { \top } } \end{array} \right] \in \mathbb { R } ^ { n \times 2 d } \qquad \mathrm { a n d } \qquad q : = \sqrt { \beta } \left[ a ^ { \top } \quad \mathbf { 1 } _ { d } ^ { \top } \right] \in \mathbb { R } ^ { 2 d } .
$$

Next, choose $C _ { B }$ so that the constructed matrices satisfy the required entry bounds. Since $d =$ $C _ { d } \log { n }$ and $t \geq c \log n$

$$
\beta = \frac { 4 d ( R + 1 ) \log n } { \varepsilon t } \leq \frac { 4 C _ { d } ( R + 1 ) \log n } { \varepsilon c } .
$$

Moreover, since $t \leq d / ( 1 + \varepsilon )$ and ${ \widehat { E } } _ { j } \leq R$ log n,

$$
0 \leq \frac { 2 \widehat { E } _ { j } } { \beta } \leq \frac { R \varepsilon } { 2 ( R + 1 ) ( 1 + \varepsilon ) } < \frac 1 2 .
$$

Consequently, there is a constant $C _ { B }$ depending only on $C _ { d } , \varepsilon , c ,$ and R such that $\| \kappa \| _ { \ell _ { \infty } } \leq$ $C _ { B } { \sqrt { \log n } }$ and $\| \pmb { q } \| _ { \ell _ { \infty } } \le C _ { B } \sqrt { \log n }$

By construction, the logit between the query and key $j \in [ n ]$ is

$$
\begin{array} { l } { { \displaystyle \widehat { Z } _ { j } : = \frac { \boldsymbol { K } _ { j } ^ { \top } \boldsymbol { q } } { 2 d } = \frac { \beta } { 2 d } \Big ( \langle \boldsymbol { a } , \boldsymbol { b } _ { j } \rangle + \frac { 2 d \widehat { E } _ { j } } { \beta } \Big ) = \frac { \beta } { 2 d } \langle \boldsymbol { a } , \boldsymbol { b } _ { j } \rangle + \widehat { E } _ { j } } } \\ { { \displaystyle \quad = \frac { \beta } { 4 d } \big ( \| \boldsymbol { a } \| _ { 2 } ^ { 2 } + \| \boldsymbol { b } _ { j } \| _ { 2 } ^ { 2 } - \| \boldsymbol { a } - \boldsymbol { b } _ { j } \| _ { 0 } \big ) + \widehat { E } _ { j } = \frac { \beta } { 4 d } \big ( d - \| \boldsymbol { a } - \boldsymbol { b } _ { j } \| _ { 0 } \big ) + \widehat { E } _ { j } } } \\ { { \displaystyle \quad = \frac { \beta } { 4 } \Big ( 1 - \frac { \| \boldsymbol { a } - \boldsymbol { b } _ { j } \| _ { 0 } } { d } \Big ) + \widehat { E } _ { j } , } } \end{array}\tag{6}
$$

where we use that $\| \pmb { a } \| _ { 0 } = \| \pmb { b } \| _ { 0 } = d / 2$ . Suppose now there is some close partner for a at index $i \in [ n ]$ with $\| \pmb { a } - \pmb { b } _ { i } \| _ { 0 } \leq t .$ . Then, for all far vectors at index $j \in [ n ]$ with $\| \pmb { a } - \pmb { b } _ { j } \| _ { 0 } > ( 1 + \varepsilon ) t$

$$
\begin{array} { r l } & { \widehat { Z } _ { i } - \widehat { Z } _ { j } \overset { ( 6 ) } { = } \frac { \beta } { 4 d } \big ( \| { \pmb a } - { \pmb b } _ { j } \| _ { 0 } - \| { \pmb a } - { \pmb b } _ { i } \| _ { 0 } \big ) + \widehat { E } _ { i } - \widehat { E } _ { j } } \\ & { \qquad \ge \frac { \beta \varepsilon t } { 4 d } + \widehat { E } _ { i } - \widehat { E } _ { j } } \\ & { \qquad \ge ( R + 1 ) \log n - R \log n > 0 , } \end{array}
$$

so logits of close partners are separated from those of far partners. In particular,

$$
j ^ { \star } : = \underset { j \in [ n ] } { \arg \operatorname* { m a x } } \widehat Z _ { j } \in \big \{ j \in [ n ] : \| a - b _ { j } \| _ { 0 } \leq ( 1 + \varepsilon ) t \big \} .\tag{7}
$$

Now, compare this maximal logit at position $j ^ { \star }$ against the other logits. Applying Lemma $\mathrm { A . 7 }$ with $\theta = \gamma / 2 , L = R \log n$ , and $g = 2$ log n gives

$$
\operatorname* { P r } \left[ \forall j ^ { \star } \neq j \in [ n ] : { \widehat { Z } } _ { j ^ { \star } } - { \widehat { Z } } _ { j } \geq 2 \log n \right] \geq n ^ { - \gamma } - n ^ { - 2 } \geq { \frac { 1 } { 2 } } n ^ { - \gamma }
$$

for all sufficiently large n. On the same event,

$$
1 - \mathrm { s o f t m a x } _ { j ^ { \ast } } ( q K ^ { \top } / ( 2 d ) ) = \frac { \sum _ { j ^ { \ast } \neq j \in [ n ] } e ^ { \widehat Z _ { j } } } { e ^ { \widehat Z _ { j } } + \sum _ { j ^ { \ast } \neq j \in [ n ] } e ^ { \widehat Z _ { j } } } \leq \frac { ( n - 1 ) e ^ { \widehat Z _ { j } - 2 \log n } } { e ^ { \widehat Z _ { j } } } = \frac { n - 1 } { n ^ { 2 } } \leq \frac { 1 } { n } ,
$$

which proves the claim.

## A.5 PROOF OF THEOREM 5.1

We can now prove Theorem 5.1 by applying Lemma A.8 to the closest pair problem from Definition $\mathsf { A } . 2 .$ . The construction identifies an approximate closest pair by letting its key receive almost all the attention for a given query, with some probability decreasing polynomially in the input size $n .$ To ensure a large success probability, we construct and preprocess sufficiently many independent copies, which increases preprocessing only polynomially and can be chosen so that the total decoding time remains truly sublinear. On a successful copy, every valid sparse-attention output must contain the dominant key.

Theorem 5.1. Fix any $\alpha \in ( 0 , 1 )$ and k, $q > 0 .$ . Assuming SETH, there exist $C _ { d } , C _ { B } > 0$ such that SparseAtt(n, D, B, α) cannot be solved with $\mathcal { O } ( n ^ { k } )$ preprocessing time and $\mathcal { O } ( n ^ { 1 - q } )$ decoding time, for $D : = C _ { d } \log$ n and $B : = C _ { B } \sqrt { \log n } .$

Proof. Assume w.l.o.g. that $k > 1$ and $q \in ( 0 , 1 / 2 )$ . We choose $C _ { d } , \varepsilon > 0$ as the constant from Proposition A.3 so that $\mathsf { P r e C P } ( n , 1 , d , \varepsilon )$ cannot be solved with ${ \mathcal { O } } ( n ^ { k + 1 } )$ preprocessing time and $\mathcal { O } ( \bar { n } ^ { 1 - q / 2 } )$ decoding time for d $\begin{array} { r } { \ell : = \frac { C _ { d } } { 2 } } \end{array}$ log n. Suppose, for contradiction, that $\mathsf { S p a r s e A t t } ( n , D , B , \alpha )$ can be solved with $O ( n ^ { k } )$ preprocessing time and $\mathcal { O } ( n ^ { 1 - q } )$ decoding time.

We construct an algorithm for $\mathsf { P r e C P } ( n , 1 , d , \varepsilon )$ with $\mathcal { O } ( n ^ { k + 1 } )$ preprocessing time and $\mathcal { O } ( n ^ { 1 - q / 2 } )$ query time, contradicting Proposition $_ { \mathrm { A } . 3 }$ . To that end, let a threshold $t \in [ 0 , d ]$ and a set $\mathbb { B } =$ $\{ b _ { 1 } , \cdot \cdot \cdot , b _ { n } \} \subseteq \mathbb { M } _ { d }$ be an input to $\mathsf { P r e C P } ( n , 1 , d , \varepsilon )$ , together with a query $\pmb { a } \in \mathbb { M } _ { d }$ revealed after preprocessing. Applying Lemma A.4 to $C _ { d }$ and q gives a constant $c \in ( 0 , C _ { d } / 2 )$ so that we can decide the PreCP instance with $\ O ( n d ) \leq \ O ( n ^ { k } )$ preprocessing time and $\mathcal { O } ( n ^ { 1 - q } )$ decoding time $\arg t \leq c \log n$ . We may there assume that $t > c \log n$ . We may further assume that $( 1 + \varepsilon ) t \leq d .$ Indeed, if $( 1 + \varepsilon ) t > \dot { d }$ , then every conceivable pair $( a , b ) \in \{ \bar { 0 } , 1 \} ^ { d } \times \mathbb { B }$ has Hamming distance at most $d < ( 1 + \varepsilon ) t$ , so the problem is trivial to decide.

We show that the PreCP instance can be decided by finding a small set of keys in K receiving at least an α-fraction of the attention for the query q. After seeing only t and B, we apply Lemma A.8 with parameters $\varepsilon , c , C _ { d }$ , and $\gamma = q / 4 .$ . This supplies the constant $C _ { B } \ > \ 0$ and allows us to sample, during preprocessing and in $\ddot { \mathcal { O } } ( T n d ) \ \leq \ \dot { \mathcal { O } } ( n ^ { k + 1 } )$ time, $T : = C _ { T } n ^ { \gamma }$ independent key matrices ${ \cal K } ^ { ( 1 ) } , \ldots , { \cal K } ^ { ( T ) } \in \bar { [ - B , B ] } ^ { n \times D }$ , which we preprocess with the hypothetical preprocessing algorithm for sparse attention in $\mathcal { O } ( \bar { T } n ^ { k } ) \leq \mathcal { O } ( n ^ { k + 1 } )$ time.

After that, let the vector $\textbf { \em a } \in \mathbb { M } _ { d }$ be revealed. Now, Lemma A.8 allows us to construct, in $\mathcal { O } ( T d ) \leq \mathcal { O } ( n ^ { 1 - q } )$ time, the queries $\pmb q ^ { ( 1 ) } , \dots , \pmb q ^ { ( T ) } \in [ - B , B ] ^ { D }$ corresponding to the key matrices and guaranteeing that

$$
\operatorname* { P r } \left[ \exists j ^ { \star } \in [ n ] : \| a - b _ { j ^ { \star } } \| _ { 0 } \leq ( 1 + \varepsilon ) t \mathrm { ~ a n d ~ } \mathrm { s o f t m a x } _ { j ^ { \star } } \big ( K ^ { ( \ell ) } q ^ { ( \ell ) } / D \big ) > 1 - \frac { 1 } { n } \right] \geq \frac { 1 } { 2 } n ^ { - \gamma }\tag{8}
$$

for each copy $\ell \in [ T ]$ if $\begin{array} { r } { \operatorname* { m i n } _ { j \in [ n ] } \| \pmb { a } - \pmb { b } _ { j } \| _ { 0 } \le t . } \end{array}$ . Again for each copy, we run the hypothetical decoding algorithm for sparse attention, resulting in explicit sets $\mathbb { I } ^ { ( 1 ) } , \dots , \mathbb { I } ^ { ( T ) } \subseteq [ n ]$ . Each invocation runs in $\mathcal { O } ( n ^ { 1 - q } )$ time by assumption, so, in particular, $| \mathbb { I } ^ { ( \ell ) } | = \mathcal { O } ( n ^ { 1 - q } )$ for each $\ell \in [ T ]$ . For each copy $\ell \in [ T ]$ and each index $i \in \mathbb { I } ^ { ( \ell ) }$ , we check if $\| \pmb { a } - \pmb { b } _ { i } \| _ { 0 } \leq ( 1 + \varepsilon ) t$ . If we find such an index $i ,$ we accept the $\mathsf { P r e C P }$ instance. Otherwise, reject the instance.

The algorithm never falsely accepts an instance. If, however, min $_ { \cdot j \in [ n ] } \| a - b _ { j } \| _ { 0 } \leq t$ , then it follows from (8) that only with probability

$$
\left( 1 - \frac { 1 } { 2 } n ^ { - \gamma } \right) ^ { { C _ { T } } n ^ { \gamma } } \leq e ^ { - { C _ { T } } / 2 }
$$

no copy contains an index receiving more than $\textstyle 1 - { \frac { 1 } { n } }$ attention. Conversely, we can choose $C _ { T }$ as a sufficiently large absolute constant so that, with probability at least $2 / 3$ , some copy $\ell \in [ T ]$ has an index $i \in [ n ]$ with

$$
\operatorname { s o f t m a x } _ { i } \left( K ^ { ( \ell ) } \pmb { q } ^ { ( \ell ) } / D \right) > 1 - \frac { 1 } { n } .
$$

On such a copy, assuming n is sufficiently large, $\mathbb { I } ^ { ( \ell ) }$ can only receive at least α attention if it includes the index i. Consequently, the hypothetical decoding algorithm for sparse attention guarantees $i \in$ $\mathbb { I } ^ { ( \ell ) }$ , so our algorithm for PreCP accepts.

Put together, we obtain a bounded-error algorithm for Pre $\mathsf { T } \mathsf { P } ( n , 1 , d , \varepsilon )$ The algorithm uses $O ( n ^ { k + 1 } )$ time during preprocessing and $\mathcal { O } ( n ^ { \gamma } n ^ { 1 - q } d ) \leq \mathcal { O } ( n ^ { 1 - q / 2 } )$ time during decoding. That contradicts Proposition A.3 and thereby proves the claim. □