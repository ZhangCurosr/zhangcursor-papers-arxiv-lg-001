# Poisoning Atacks on the PGM-index

Atsuki Sato   
a\_sato@hal.t.u-tokyo.ac.jp   
Graduate School of Information   
Science and Technology, The   
University of Tokyo   
Tokyo, Japan   
Martin Aumüller   
maau@itu.dk   
Algorithms Group, IT University of   
Copenhagen   
Copenhagen, Denmark   
Yusuke Matsui   
matsui@hal.t.u-tokyo.ac.jp   
Graduate School of Information   
Science and Technology, The   
University of Tokyo   
Tokyo, Japan

## ABSTRACT

The PGM-index (Ferragina and Vinciguerra, VLDB’20) is one of the most practical learned indexes, owing to its theoretical elegance and consistently strong empirical performance. It is built on optimal piecewise linear approximations (PLAs) that minimize the number of segments. In this paper, we ask how sensitive this optimal PLA itself is to poisoning attacks. We propose PGM-attack, an eficient poisoning attack that sequentially inserts adversarial keys to inflate the resulting number of segments, and we develop a method for deriving theoretical upper bounds on the number of segments attainable under arbitrary insertions. Our experiments show that poisoning only 10% of the keys allows PGM-attack to increase the segment count by up to 120×. On every evaluated instance, our instance-dependent upper bound is at most 1.92× the segment count attained by PGM-attack, certifying that PGM-attack achieves at least 52% of the optimum. This increase in the number of segments enlarges the PGM-index by up to 120×. Moreover, the attack also transfers to other learned indexes, substantially inflating the index size of PLA-based ones in particular. Our results reveal that, despite the optimality of its PLAs, the PGM-index has an intrinsic vulnerability rooted in its optimization objective, motivating robustness-aware objective design for future learned indexes.

## 1 INTRODUCTION

In recent years, learned indexes [23] have attracted considerable attention as space-eficient, high-performance alternatives to traditional index structures such as B+-trees. The PGM-index [16] is a leading example. It recursively constructs optimal piecewise linear approximations (PLAs) of the cumulative distribution function (CDF), where an optimal PLA is one that uses the minimum number of segments. This optimality allows the PGM-index to provide worst-case theoretical guarantees while achieving a favorable trade of between memory usage and query time. These properties make the PGM-index an appealing choice for large-scale settings where both practical eficiency and worst-case guarantees matter, and it has begun to be adopted in high-performance database systems and search engines [19, 28].

As learned indexes become increasingly practical, understanding their robustness becomes correspondingly important. By quantify ing how much an index’s memory eficiency and query performance can deteriorate after the insertion of a small amount of data, we can assess its stability and deploy it with greater confidence. For tradi tional indexes, such efects are often straightforward to characterize: for example, a B+-tree requires Θ(�) space and supports queries in Θ(log �) time, making the impact of a small increase in the number of keys readily predictable. For learned indexes, however, the impact is far less clear because their performance can depend sensitively on the data distribution. Motivated by this concern, recent studies have investigated attacks against several learned indexes, including poisoning attacks against mean-squared-error (MSE) minimizing linear models and Recursive Model Indexes (RMIs) [22, 23, 34], as well as algorithmic-complexity attacks against ALEX [10, 35, 36].

![](images/57f2c548f6423f2324e3253c9186b8790a86f4a9ff7d832b7267b124c07c1e93.jpg)  
<sup>Fi</sup>gure <sup>1</sup>: <sup>O</sup>ur <sup>b</sup>o<sup>tt</sup>om-up approac<sup>h</sup> progresses <sup>f</sup>rom maximum-error maximization (P1), through covered-key minimization (P2), to segment maximization (P3). The P3 <sub>p</sub>anel illustrates how our PGM-attack can efectivel<sub>y</sub> de<sub>g</sub>rade a PLA: before <sub>p</sub>oisonin<sub>g,</sub> a sin<sub>g</sub>le se<sub>g</sub>ment sufices to approximate the ranks of all keys within error � = 1, whereas insertin<sub>g</sub> onl<sub>y</sub> two <sub>p</sub>oison ke<sub>y</sub>s increases the re<sub>q</sub>uired <sub>num</sub>b<sub>er</sub> <sub>o</sub>f <sub>segmen</sub>t<sub>s</sub> t<sub>o</sub> th<sub>ree.</sub>

We note that, although the PGM-index enjoys several theoretical guarantees, its robustness against poisoning (or, more generally, against data insertions) remains largely uncharacterized, which is an important concern in practical deployments. For example, the PGM-index has a worst-case guarantee of provably superior performance to conventional B+-trees [16], but this guarantee does not imply robustness against poisoning: the strength of the PGM-index lies in the fact that it typically performs far better than B+-trees, and if a small number of poison keys can drive its performance toward this worst case, this strength is lost. Moreover, in terms of sensitivity to attacks, i.e., the discrepancy between the performance before and after an attack, the PGM-index may even be more sensitive than a conventional B+-tree. Such sensitivity poses a serious problem when deploying the PGM-index in real systems: database administrators are often drawn to the PGM-index for its high performance, yet they would have to worry that its benefits might be largely eliminated by only a small number of key insertions.

In this paper, we propose PGM-attack, a poisoning attack against PLA-based learned indexes, including the PGM-index. PGM-attack substantially increases the number of segments in the optimal PLA, which forms the bottom-level PLA of the PGM-index. The key-rank plots at the bottom of Figure 1 show an illustrative example; insert ing only two poison keys increases the number of segments from one to three. Because these bottom-level segments dominate the model size of the PGM-index, increasing their number reduces compression eficiency and can degrade query performance through more cache misses and larger upper-level structures.

Analyzing poisoning attacks against PLAs is challenging for several reasons. First, this problem is dificult to analyze theoretically because even the basic building block of PLA construction (maximum-error-minimizing linear regression) does not admit a simple closed-form solution. This is clearly distinct from the MSEminimizing regression considered in prior work [22, 34], which admits a closed-form solution. Second, inserting a poison key shifts the ranks of all subsequent keys and can change the boundaries and sizes of later segments, causing cascading changes throughout the segmentation. Prior attacks against RMIs [22] do not need to account for such global efects because they assume that each linear regression model covers a fixed number of keys. Finally, simple strategies such as inserting a consecutive run of keys, as used in algorithmic-complexity attacks against ALEX [35, 36], are inefective at increasing the total number of PLA segments. Such keys can typically be absorbed into a single PLA segment and therefore have little efect on the segment count.

To address these challenges, we adopt a bottom-up approach that proceeds from simpler subproblems to our final objective, as illustrated in Figure 1. We first analyze P1, the problem of maximiz ing the maximum error of linear regression. The resulting insights allow us to solve P2, which minimizes the number of legitimate keys covered by a single segment. Finally, by repeatedly solving P2, we address P3, which maximizes the total number of segments in the PLA, thereby constructing PGM-attack. We also propose algorithms for deriving upper bounds on the number of segments attainable under arbitrary key insertions. Our experiments show that poisoning only 10% of the keys can increase the number of segments by up to 120×. These results reveal that, despite the optimality of its PLAs, the PGM-index has an intrinsic vulnerability rooted in its optimization objective. This motivates the design of future learned indexes with more robustness-aware objectives.

Contributions. Our contributions are summarized as follows:

• We formulate three new poisoning problems on the CDF: P1, maximizing the maximum error in linear regression; P2, mini mizing the number of covered legitimate keys in linear regression; and P3, maximizing the number of segments in the PLA.

• For P1, we characterize optimal poison sets (Section 3) and show experimentally that appropriately selected consecutive-integer poison keys are almost always optimal (Section 6.2).

• For P2, we prove a reduction to P1 and develop a near-lineartime algorithm (Section 4). We experimentally show that this algorithm generates near-optimal consecutive-integer poison keys (Section 6.3).

• For P3, we propose PGM-attack, which repeatedly generates candidate poison sets for P2 and sequentially selects among them (Section 5.2). Injecting 10% poison keys increases the segment count and PGM-index size by up to 120×. The attack also transfers to other learned indexes, particularly inflating the size of PLA-based ones (Sections 6.4 and 6.5).

• For P3, we also propose algorithms for upper-bounding the number of segments attainable under arbitrary key insertions (Sections 5.3 and 5.4). On all evaluated instances, the resulting upper bound is at most 1.92× the segment count obtained by PGM-attack, certifying that the attack achieves at least 52% of the optimum on these instances (Section 6.4).

## 2 PRELIMINARIES

We consider the indexing problem, which requires building a data structure that stores a multiset X and supports operations such as member (given an element �, is � ∈ X?), predecessor (given an element �, output the largest � ≤ � such that � ∈ X), and insert (add � to X). A simple but eficient way to implement such a data structure is to store the elements of X in a sorted array �. Then, all operations boil down to eficiently finding the rank of an element �, i.e., the number of elements in X that are smaller than � [16].

The idea of a learned index as popularized by Kraska et al. [23] is that the index employs a model that learns the rank function based on the data. In other words, the model learns the cumulative distribution function (CDF) ofthe given dataset. Given an element �, the model predicts the rank of � and then uses a second verification step, for example using binary search, to compute the exact answer.

## 2<sub>.</sub>1 PLA Al<sub>gor</sub>ith<sub>m</sub>

Piecewise linear approximation (PLA) is a central building block of learned indexes: it underlies not only the PGM-index [16] but also a broad range of designs [18, 21], including recent state-ofthe-art learned indexes [8, 27]. A PLA algorithm approximates the rank function by a sequence of linear segments: given a maximum allowed error parameter �, it repeatedly extends the current segment as far as possible while keeping the maximum error at most �, and then starts a new segment. The PLA construction problem (formally defined in Section 5) thus builds on the segment extension problem (Section 4), which in turn builds on the problem of minimizing the maximum error of a single segment (Section 3).

PLA algorithms come in optimal variants that minimize the number of segments [31] and faster variants that construct near-optimal approximations [12]. Since the PGM-index adopts an optimal PLA algorithm, we focus on this setting. Accordingly, although the PGMindex serves as our primary application, our study targets the PLA algorithm itself rather than any PGM-specific implementation.

## 2.2 Im<sub>p</sub>ortance of the PLA Se<sub>g</sub>ment Count

Although the PGM-index has a recursive multi-level structure, its space and query-time complexity are dominated by the bottom-level PLA over the CDF [16]: for a dataset of � keys with optimal segment count $m _ { \mathrm { o p t } }$ , the index requires $O ( m _ { \mathrm { o p t } } )$ space and $O ( \log m _ { \mathrm { o p t } } { + } \log \varepsilon )$ query time. Attacks that increase $m _ { \mathrm { o p t } }$ therefore directly degrade the index’s practical performance. Our theoretical analysis focuses on this fundamental PLA component, and our experiments (Section 6) validate its end-to-end impact on the PGM-index and other learned indexes.

## 2<sub>.</sub>3 Th<sub>rea</sub>t M<sub>o</sub>d<sub>e</sub>l

Following prior work [22, 34], we adopt a white-box setting in which the attacker knows all legitimate keys and can inject poison keys before the index is constructed; for the problems studied in Sections 4 and 5, the attacker also knows the hyperparameter �. This is the standard starting point in attack studies [22, 34]: it isolates the inherent sensitivity of the PLA construction objective from deployment-specific restrictions and provides a basis for weakerknowledge attacks. For example, a black-box attacker could first estimate the data distribution or � from limited observations and then apply a white-box attack to the resulting surrogate [36]. We also follow prior work [22, 34] by restricting poison keys to lie strictly between the minimum and maximum legitimate keys. This restriction conservatively limits the attacker’s capabilities by excluding poison keys outside the legitimate data range, which could be easily detected and removed as outliers. We show that even under this restriction, the attacker can substantially increase the number of segments.

## 3 MAXIMUM-ERROR MAXIMIZATION

## 3.1 Problem Settin<sub>g</sub>

We refer to the keys present in the original (pre-attack) training data as legitimate keys and denote the set of legitimate keys by K, with $n : = | \mathcal { K } |$ . We use X to denote a general key set that may include both legitimate and poison keys, and let $N : = | X |$ . For a positive integer �, we write $[ m ] : = \{ 1 , 2 , . . . , m \}$

We begin by defining linear regression on CDFs.

Definition 1 (Linear Regression on CDFs (Maximum Error)). $\mathit { L e t X } = \left\{ x _ { 1 } , x _ { 2 } , \ldots , x _ { N } \right\}$ be a multiset ofnatural numb<sub>ers suc</sub>h th<sub>a</sub>t $x _ { 1 } \le x _ { 2 } \le \cdots \le x _ { N }$ <sub>.</sub> F<sub>or eac</sub>h $i \in [ N ]$ , define the rank of �<sub>�</sub> as $r _ { i } : = i .$ The maximum error of linear regression on CDFs for X is defined by

$$
E ( X ) : = \operatorname* { m i n } _ { w , b } \operatorname* { m a x } _ { i \in [ N ] } | w x _ { i } + b - r _ { i } | .\tag{1}
$$

Our definition difers from prior work [22, 34] only in the choice of loss function: instead of minimizing the MSE [34, Eq. 1–2],

$$
\operatorname* { m i n } _ { w , b } \frac { 1 } { N } \sum _ { i \in [ N ] } { ( w x _ { i } + b - r _ { i } ) ^ { 2 } } ,\tag{2}
$$

we minimize the maximum error, which enables a more direct connection to PLA algorithms. Computing �(X) can be framed as a three-dimensional linear programming problem [6] and solved in $O ( N )$ time via Megiddo’s algorithm [30]; unlike MSE, no simple closed-form solution is known.

We now define the corresponding poisoning problem (again identical to the MSE-based formulation [22, 34] except for the objective).

Definition 2 (Maximum-Error Maximization Problem). Let $\mathcal { K } = \{ k _ { 1 } , k _ { 2 } , . . . , k _ { n } \} \subset \mathbb { N }$ b<sub>e �</sub> di<sub>s</sub>ti<sub>nc</sub>t l<sub>eg</sub>iti<sub>ma</sub>t<sub>e</sub> k<sub>eys suc</sub>h th<sub>a</sub>t $k _ { 1 } < k _ { 2 } < \cdots < k _ { n } ,$ <sub>an</sub>d l<sub>e</sub>t $\lambda \in \mathbb { N } .$ . The maximum-error maximization problem is to choose up to � integersfrom $\{ k _ { 1 } , k _ { 1 } +$ $1 , \ldots , k _ { n } \} \backslash \mathcal { K }$ <sub>so as</sub> t<sub>o max</sub>i<sub>m</sub>i<sub>ze</sub> th<sub>e max</sub>i<sub>mum error:</sub>

$$
\operatorname { a r g m a x } _ { \mathcal { P } \mathrm { ~ s . t . ~ } | \mathcal { P } | \leq \lambda , \mathcal { P } \subseteq \{ k _ { 1 } , k _ { 1 } + 1 , \ldots , k _ { n } \} \backslash \mathcal { K } } E ^ { ( \mathcal { K } \cup \mathcal { P } ) . }\tag{3}
$$

Connection to the PGM-index. This problem is fundamental to attacking the PGM-index. The PLA algorithm extends each segment as far as possible while keeping the maximum error within �. Therefore, increasing $m _ { \mathrm { o p t } }$ amounts to forcing the maximum error to exceed � earlier, thereby shortening segments. We revisit this connection in Sections 4 and 5.

Dificulty of the Problem. This problem involves several intrinsic challenges. Rank shifting: As in the MSE setting, inserting a poison key increases the rank of every subsequent key and poison by one. This global interaction makes the optimal choice of poison keys highly non-trivial. Restricted integer choices: Integers already used as legitimate keys or previously selected poison keys cannot be reused. This discrete feasibility constraint complicates analytical reasoning. No closed-form solution: Unlike ordinary least squares under MSE, which admits a closed-form solution, the maximum-error regression problem has no known closed-form characterization. This significantly complicates theoretical analysis.

Illustrative Example. Figure 2 illustrates an example of a maximum-error poisoning attack on $\mathcal { K } = \{ 1 , 1 5 , 1 7 , 2 1 , 3 2 , 3 7 , 3 9 \}$ As in the MSE setting, inserting a poison key shifts the ranks of all subsequent keys and poisons by one. For example, comparing Figures 2a and 2b, the insertion of a poison does not afect the ranks of keys less than the poison, while it increases the rank of each key greater than the poison by one. Figure 2d plots $E ( { \mathcal { K } } \cup \{ p \} )$ for each candidate poison �. Integers already present in K are excluded from the candidate set. In each interval, the poison that maximizes the maximum error is adjacent to a legitimate key. This structural property is formally established later in Corollary 1.

Duplicate-Allowed Setting. We also introduce a relaxed setting in which duplicate values are allowed. In this variant, poison keys may coincide with legitimate keys, and the same integer may be selected multiple times as poison. The optimal value in this relaxed setting provides an upper bound for the original (duplicateforbidden) problem.

## 3.2 The Structure of O<sub>p</sub>timal Attacks

We prove that there exists an optimal solution to the maximumerror poisoning problem (Definition 2) in which every poison key is adjacent to a legitimate key, either directly or through a chain of adjacent poison keys. Formally, we have the following theorem.

$$
\begin{array} { r l r } & { \mathrm { T H E O R E M 1 . ~ } T h e r e e x i s t s a n o p t i m a l s o l u t i o n \mathcal { P } ^ { * } t o t h e m a x i m u m \cdot } & \\ & { e r r o r \label [ ] { o i s o n i n g ~ p r o b l e m } ( D e f i n i t i o n  { 2 } ) s a t i s f y i n g } & \\ & { \forall p \in \mathcal { P } ^ { * } , \exists k \in \mathcal { K } , \{ i \in \mathbb { N } \mid \operatorname* { m i n } ( p , k ) < i < \operatorname* { m a x } ( p , k ) \} \subset \mathcal { P } ^ { * } . } & \\ & { ( 4 ) | } & \end{array}
$$

To prove Theorem 1, we first establish the following lemma, which shows that there exists at least one direction (either left

![](images/fee1873b941be514a044df4eac8f69f44e5d432f8b4d9a64efa4daeba653d34b.jpg)  
(a) Regression Before Poisoning.

![](images/d8d22f129ba1e3cf1df494fb9668e79186a75299e4c60c46366a2be5b659d139.jpg)  
(b) Optimal 1-point Poisoning.

![](images/dfdba93ed2031fb903609170a0e18077b5c52a2a245ddfbdd38e93941d00b289.jpg)

![](images/8c9c93254acb4101e345b3d06f700b2f8912a69a4c57a9a127c23024a92b0817.jpg)  
(c) Optimal 2-point Poisoning.  
(d) Loss under 1-point poisoning.

Fi<sub>gure</sub> 2<sub>:</sub> P<sub>o</sub>i<sub>son</sub>i<sub>ng a</sub>tt<sub>ac</sub>k <sub>on</sub> $\mathcal { K } = \{ 1 , 1 5 , 1 7 , 2 1 , 3 2 , 3 7 , 3 9 \}$ under the maximum-error objective. By inserting poison points, the <sub>ran</sub>k <sub>o</sub>f <sub>eac</sub>h k<sub>ey</sub> <sub>grea</sub>t<sub>er</sub> th<sub>an</sub> <sub>a</sub> <sub>po</sub>i<sub>son</sub> i<sub>ncreases.</sub> Th<sub>e</sub> li<sub>near</sub> <sub>regress</sub>i<sub>on</sub> <sub>mo</sub>d<sub>e</sub>l i<sub>s</sub> th<sub>en</sub> fitt<sub>e</sub>d <sub>on</sub> th<sub>e</sub> <sub>po</sub>i<sub>sone</sub>d d<sub>a</sub>t<sub>ase</sub>t<sub>,</sub> l<sub>ea</sub>di<sub>ng</sub> t<sub>o</sub> a change in the maximum error. Note that the optimal 2-point attack {33, 34} does not contain the optimal 1-point attack.

![](images/80e2bf501fbcd74b748dc3c37fb5133ffd7920ab0ba7bff1a54113c9d301f824.jpg)  
Fi<sub>gure</sub> 3<sub>:</sub> Ill<sub>us</sub>t<sub>ra</sub>ti<sub>on</sub> <sub>o</sub>f th<sub>e</sub> <sub>proo</sub>f <sub>o</sub>f Th<sub>eorem</sub> 1<sub>.</sub> B<sub>y</sub> L<sub>emma</sub> 1<sub>,</sub> <sub>s</sub>hifti<sub>ng</sub> <sub>an</sub> i<sub>so</sub>l<sub>a</sub>t<sub>e</sub>d <sub>po</sub>i<sub>son</sub> bl<sub>oc</sub>k <sub>re</sub>d<sub>uces</sub> th<sub>e</sub> <sub>num</sub>b<sub>er</sub> <sub>o</sub>f i<sub>so-</sub> l<sub>a</sub>t<sub>e</sub>d bl<sub>oc</sub>k<sub>s w</sub>ith<sub>ou</sub>t d<sub>ecreas</sub>i<sub>ng</sub> th<sub>e max</sub>i<sub>mum error.</sub>

or right) in which shifting a contiguous block of keys does not decrease the maximum error.

Lemma 1. Let $\mathcal { X } = \{ x _ { 1 } , x _ { 2 } , \ldots , x _ { N } \}$ b<sub>e a mu</sub>lti<sub>se</sub>t <sub>w</sub>ith $x _ { 1 } \leq x _ { 2 } \leq$ · · · $\leq x _ { N }$ <sub>,</sub> <sub>an</sub>d l<sub>e</sub>t $1 < l \leq r < N .$ . Define

$$
\mathcal { A } : = \{ x _ { i } \} _ { i = 1 } ^ { l - 1 } \uplus \{ x _ { i } \} _ { i = r + 1 } ^ { N } , \quad \mathcal { B } ( \delta ) : = \{ x _ { i } + \delta \} _ { i = l } ^ { r } ,\tag{5}
$$

$f o r \delta \in \left[ x _ { l - 1 } - x _ { l } , x _ { r + 1 } - x _ { r } \right] \mathrm { , }$ <sub>,</sub> <sub>an</sub>d l<sub>e</sub>t $\mathcal { B } : = \mathcal { B } ( 0 )$ . Then, for any $\delta _ { - } \in \big [ x _ { l - 1 } - x _ { l } , 0 \big ] a n d \delta _ { + } \in \big [ 0 , x _ { r + 1 } - x _ { r } \big ] ,$

$$
E ( \mathcal { A } \uplus \mathcal { B } ) \leq \operatorname* { m a x } ( E ( \mathcal { A } \uplus \mathcal { B } ( \delta _ { - } ) ) , E ( \mathcal { A } \uplus \mathcal { B } ( \delta _ { + } ) ) ) ,\tag{6}
$$

where ⊎ denotes multiset addition.

Proof Sketch of Lemma 1. Fix $\varepsilon \geq 0$ and consider the feasible sets of line parameters. By reinterpreting the block shift in a suitably transformed parameter space, we can show that

$$
E \big ( \mathcal { A } \mathbin { \uplus } \mathcal { B } ( \delta _ { - } ) \big ) \leq \varepsilon \wedge E \big ( \mathcal { A } \mathbin { \uplus } \mathcal { B } ( \delta _ { + } ) \big ) \leq \varepsilon \Rightarrow E \big ( \mathcal { A } \mathbin { \uplus } \mathcal { B } \big ) \leq \varepsilon , \ ( \mathcal { A } \mathbin { \uplus } \mathcal { B } ) \leq \varepsilon , \ ( \mathcal { A } \mathbin { \uplus } \mathcal { B } ) .\tag{7}
$$

and � = max (�(A ⊎ B(�<sub>−</sub>)), �(A ⊎ $\mathcal { B } ( \delta _ { + } ) ) )$ yields the claim. □

Using Lemma 1, we can prove Theorem 1.

Proof Sketch of Theorem 1. Consider an optimal solution with the fewest isolated poison blocks. By Lemma 1, any isolated block can be shifted to a legitimate key or another poison block without decreasing the error (see Figure 3). This contradicts the minimality of the optimal solution, stating that such an optimal solution has no isolated poison blocks. □

Theorem 1 implies that, to find an optimal solution, it sufices to consider poison sets in which every poison key is adjacent to a legitimate key, either directly or via a chain of adjacent poison keys. The number of such candidates is at most $\binom { 2 n - 3 + \lambda } { \lambda }$ ; without this structural result, the number of possible poison sets is roughly $\binom { k _ { n } - k _ { 1 } } { \lambda }$ . Thus, Theorem 1 reduces the search space from depending on the key-universe size to depending on �, substantially reducing the cost of enumeration.

Corollary: Single-Point Attack. From Theorem 1, we obtain the following corollary for the single-point attack:

Corollary 1. For the single-point attack $( \lambda = 1 ) ,$ th<sub>ere</sub> <sub>ex</sub>i<sub>s</sub>t<sub>s</sub> an optima<sup>l</sup> so<sup>l</sup>ution w<sup>h</sup>ere t<sup>h</sup>e poison is <sup>d</sup>irect<sup>l</sup>y a<sup>d</sup>jacent to a le<sub>g</sub>itimate ke<sub>y</sub>. Formall<sub>y</sub>, there exists an optimal P<sup>∗</sup> satisf<sub>y</sub>in<sub>g</sub>

$$
p ^ { * } \in \{ k + 1 \mid k \in \mathcal { K } \} \cup \{ k - 1 \mid k \in \mathcal { K } \} .\tag{8}
$$

By this corollary, we can find an optimal single-point attack by checking all integers adjacent to legitimate keys. Since there are $O ( n )$ such candidates and each evaluation takes $O ( n )$ time, we can compute the optimal single-point poison in $O ( n ^ { 2 } )$ time.

Note, however, that greedily repeating the optimal single-point attack does not necessarily yield a globally optimal solution. For example, in Figure 2, the optimal two-point attack is $\mathcal { P } = \{ 3 3 , 3 4 \}$ (see Figure 2c). Neither ${ \mathcal { P } } = \{ 3 3 \}$ nor $\mathcal { P } ~ = ~ \{ 3 4 \}$ is an optimal single-point attack, so the greedy repetition of optimal single-point attacks fails to recover the optimal two-point attack. Note also that our theorems assert existence of an optimal adjacent solution, not that every optimum is adjacent; e.g., in Figure 2, {19} is also optimal for the single-point attack.

Duplicate-Allowed Setting. We now present a structural theorem for the duplicate-allowed setting.

Theorem 2 (Structure of an Optimal Attack in the Duplicate-Allowed Setting). In the duplicate-allowed setting, ever<sub>y</sub> o<sub>p</sub>tima<sup>l</sup> so<sup>l</sup>ution ${ \mathcal { P } } ^ { * }$ <sup>t</sup>o <sup>th</sup>e max<sup>i</sup>mum-error po<sup>i</sup>son<sup>i</sup>ng pro<sup>b</sup>- lem uses the full budget, $i . e . , | \mathcal { P } ^ { * } | = \lambda$ <sub>.</sub> M<sub>oreover,</sub> th<sub>ere ex</sub>i<sub>s</sub>t<sub>s an</sub> <sub>op</sub>ti<sub>ma</sub>l <sub>so</sub>l<sub>u</sub>ti<sub>on</sub> $\mathcal { P } ^ { * }$ <sub>suc</sub>h th<sub>a</sub>t <sub>a</sub>ll <sub>po</sub>i<sub>son</sub> k<sub>eys</sub> t<sub>a</sub>k<sub>e</sub> th<sub>e</sub> <sub>same</sub> <sub>va</sub>l<sub>ue,</sub> and this value is a le<sub>g</sub>itimate ke<sub>y</sub> in K.

Proof Sketch of Theorem 2. Adding a poison key strictly increases the maximum error, so every optimal solution uses the full budget. By Lemma 1, all poison keys can be moved onto legitimate keys. Finally, poison mass at distinct legitimate keys can be merged at one of them without decreasing the error, yielding an optimal solution in which all poison keys equal a single legitimate key. □

This theorem suggests that concentrating poisons at a single location is a promising strategy. Motivated by this theorem, in the next section we define a class of poison sets in which poisons are concentrated within a single contiguous region, and we propose an eficient algorithm for finding optimal solutions within this class.

This result provides a clear contrast to prior observations for the MSE setting. Prior work conjectured that, in the duplicateallowed MSE setting, an optimal attack can always be realized by concentrating poisons on at most three legitimate keys: the two extremes $( k _ { 1 }$ and $k _ { n } )$ and one internal key [34, Conj. 1]. By contrast, our theorem shows that under the maximum-error objective, there always exists an optimal solution that concentrates all poisons on a single legitimate key. This highlights a structural diference between the MSE and maximum-error objectives.

## 3<sub>.</sub>3 M<sub>e</sub>th<sub>o</sub>d<sub>s</sub> f<sub>o</sub>r Obt<sub>a</sub>inin<sub>g</sub> P<sub>o</sub>i<sub>so</sub>n S<sub>o</sub>l<sub>u</sub>ti<sub>o</sub>n<sub>s</sub>

Based on the theoretical results above, we propose four poisoning methods: three for the setting without duplicate keys (Optimal, Greedy, and Consecutive) and one for the setting that allows duplicates (Duplicate-Opt). The main takeaway is that the Consecutive method ofers the best overall trade-of: it is faster than the Greedy method and, as we show later in Section 6.2, typically attains the global optimum in the vast majority of cases.

Optimal Method. By Theorem 1, we can restrict the candidate optimal solutions to $O \left( \binom { 2 n - 3 + \lambda } { \lambda } \right)$ possibilities. Since evaluating each candidate takes $O ( n + \lambda )$ time, the Optimal method yields an exact optimal solution in $O \left( \binom { 2 n - 3 + \lambda } { \lambda } \left( n + \lambda \right) \right)$ time.

Greedy Method. The Greedy method repeatedly applies the optimal single-point attack. At each iteration, we compute the optimal single poison for the current set $\mathcal { K } \cup \mathcal { P }$ and add it to P.

The total time complexity is $O ( \lambda ( n + \lambda ) ^ { 2 } )$ . In each iteration, there are $O ( n + \lambda )$ candidates for the optimal single-point attack, and evaluating each candidate takes $O ( n + \lambda )$ time. This process is repeated O(�) times.

Consecutive Method. Motivated by Theorem 2, we consider solutions restricted to consecutive poisons, defined as follows.

Definition 3. A poison set P is called consecutive ifthere exist i<sub>n</sub>t<sub>egers</sub> �<sub>, � w</sub>ith $k _ { 1 } < l \leq r < k _ { n }$ <sub>suc</sub>h th<sub>a</sub>t

$$
\mathcal { P } = \{ l , l + 1 , . . . , r \} \backslash \mathcal { K } \quad \land \quad | \mathcal { P } | = \lambda .\tag{9}
$$

The Consecutive method searches for an optimal solution under this restriction. The following theorem allows us to further reduce the candidate space.

Theorem 3. There exists an optimal consecutive solution whose left or right boundary is adjacent to a legitimate key. Formally, t<sup>h</sup>ere exists an o<sub>p</sub>tima<sup>l</sup> consecutive <sub>p</sub>oison set $\mathcal { P } _ { \mathrm { c o n } } ^ { * }$ <sub>suc</sub>h th<sub>a</sub>t

$$
\operatorname* { m i n } ( \mathcal { P } _ { \mathrm { c o n } } ^ { * } ) - 1 \in \mathcal { K } \mathrm { ~ \vee ~ } \operatorname* { m a x } ( \mathcal { P } _ { \mathrm { c o n } } ^ { * } ) + 1 \in \mathcal { K } .\tag{10}
$$

Proof of Theorem 3. By Lemma 1, any consecutive poison block not adjacent to a legitimate key can be shifted without decreasing the maximum error until it becomes adjacent to one, yielding the claim. □

Using Theorem 3, Consecutive runs in $O ( n ( n + \lambda ) )$ time; there are $O ( n )$ candidate intervals to examine, and evaluating each candidate takes $O ( n + \lambda )$ time.

Duplicate-Opt Method. The Duplicate-Opt method computes the optimal solution in the duplicate-allowed setting. By Theorem 2, it sufices to try � candidates, one for each $k \in \mathcal K$ , where all � poisons are placed at �. Evaluating each candidate takes $O ( n + \lambda )$ time. Hence, the total time complexity is $O ( n ( n + \lambda ) )$ , and this yields the optimal solution in the duplicate-allowed setting.

## 4 COVERED-LEGITIMATE-KEYS MINIMIZATION

This section bridges the attack studied in the previous section, namely maximizing maximum error for a fixed set of covered keys, to the problem that is directly relevant to the PGM-index.

## 4.1 Problem Settin<sub>g</sub>

We consider the following setting. Given a maximum allowed error parameter �, a linear regression model is built so that it covers as many keys as possible while keeping the maximum error at most �.

Definition 4 (Linear Regression on CDFs (Number of Covered Keys)). Let $X = \{ x _ { 1 } , x _ { 2 } , \ldots , x _ { N } \}$ be a multiset of <sub>na</sub>t<sub>ura</sub>l <sub>num</sub>b<sub>ers w</sub>ith $x _ { 1 } \le x _ { 2 } \le \cdots \le x _ { N } .$ <sub>.</sub> F<sub>or eac</sub>h $i \in [ N ]$ define the rank of �<sub>�</sub> as $r _ { i } : = i .$ Given $\varepsilon \ \geq \ 0$ <sub>,</sub> th<sub>e</sub> <sub>max</sub>i<sub>mum</sub> number ofcovered keys oflinear regression on CDFs for X under the maximum allowed error parameter � is defined by

$$
S _ { \varepsilon } ( X ) : = \operatorname* { m a x } _ { w , b } \operatorname* { m a x } \left\{ i \in [ N ] \left| \operatorname* { m a x } _ { j \in [ i ] } | w x _ { j } + b - r _ { j } | \leq \varepsilon \right\} . \right.\tag{11}
$$

This problem can be solved in $O ( S _ { \varepsilon } ( X ) )$ time using the algorithm of [31]. The PGM-index construction repeats this procedure to obtain a PLA with as few segments as possible.

We next define the poisoning variant of CDF linear regression under the number-of-covered-keys objective.

Definition 5 (Covered-Legitimate-Keys Minimization Problem). Let $\mathcal { K } = \{ k _ { 1 } , k _ { 2 } , \ldots , k _ { n } \} \subset$ N b<sub>e �</sub> di<sub>s</sub>ti<sub>nc</sub>t l<sub>eg</sub>iti<sub>-</sub> <sub>ma</sub>t<sub>e</sub> k<sub>eys suc</sub>h th<sub>a</sub>t $k _ { 1 } < k _ { 2 } < \cdots < k _ { n }$ <sub>,</sub> l<sub>e</sub>t $\varepsilon \geq 0$ <sub>, an</sub>d l<sub>e</sub>t $\lambda \in \mathbb { N } .$ <sup>F</sup>or a po<sup>i</sup>son se<sup>t</sup> $\mathcal { P } \subseteq \{ k _ { 1 } , k _ { 1 } + 1 , \dots , k _ { n } \} \mid$ K with $| \mathcal { P } | \leq \lambda ,$ <sub>,</sub> l<sub>e</sub>t $\mathcal { X } : = \mathcal { K } \cup \mathcal { P } = \{ x _ { 1 } , x _ { 2 } , . . . , x _ { N } \}$ be the sorted key set. The number of covered legitimate keys is defined as

$$
C _ { \varepsilon } ( \mathcal { K } , \mathcal { P } ) : = \left| \mathcal { K } \cap \left\{ x _ { 1 } , x _ { 2 } , . . . , x _ { S _ { \varepsilon } ( \chi ) } \right\} \right| .\tag{12}
$$

The covered-legitimate-keys minimization problem is to find a poison set minimizing the number ofcovered legitimate keys:

$$
\operatorname * { a r g m i n } _ { \mathcal { P } _ { \mathrm { \ s . t . } } \mid \mathcal { P } \mid \leq \lambda , \mathcal { P } \subseteq \{ k _ { 1 } , k _ { 1 } + 1 , \ldots , k _ { n } \} \setminus \mathcal { K } } C _ { \varepsilon } ( \mathcal { K } , \mathcal { P } ) .\tag{13}
$$

The key-rank plots in the P2 block of Figure 1 illustrate this problem: before poisoning, a segment covers 7 legitimate keys, but after inserting a single poison key, it covers only 5.

Since, on the model-construction side, one cannot distinguish legitimate keys from poisoned keys, it fits a model that maximizes $S _ { \varepsilon } ( \mathcal { K } \cup \mathcal { P } )$ . The attacker, however, chooses P to minimize $C _ { \varepsilon } ( \mathcal { K } , \mathcal { P } )$ This objective captures the attacker’s goal of minimizing the efective coverage size of a segment, i.e., the number of legitimate keys covered. Repeatedly solving this problem increases the number of segments in the resulting PGM-index (detailed in Section 5).

![](images/41fbdbf48030e8711faf8e384e9b939337a2a6a93a235ae90af225d0ed11c141.jpg)  
Fi<sub>gure</sub> 4<sub>:</sub> Di<sub>scre</sub>t<sub>e-</sub>i<sub>n</sub>t<sub>ercep</sub>t linear re<sub>g</sub>ression. For each <sub>can</sub>did<sub>a</sub>t<sub>e</sub> i<sub>n</sub>t<sub>ercep</sub>t<sub>, ex</sub>t<sub>en</sub>d th<sub>e segmen</sub>t <sub>w</sub>hil<sub>e</sub> f<sub>eas</sub>ibl<sub>e</sub> <sub>an</sub>d <sub>se</sub>l<sub>ec</sub>t th<sub>e</sub> l<sub>onges</sub>t <sub>one.</sub>

![](images/64619a0c57e01c28aeacd3e8e730a23435b7c73c739a8f8b5ac917a36d11cacc.jpg)  
Fi<sub>g</sub>ure 5: DI-Consecutive. With <sub>precompu</sub>t<sub>e</sub>d f<sub>eas</sub>ibl<sub>e</sub> slo<sub>p</sub>e intervals<sub>,</sub> each <sub>p</sub>oison <sub>se</sub>t <sub>can</sub>did<sub>a</sub>t<sub>e can</sub> b<sub>e eva</sub>l<sub>u-</sub> <sub>a</sub>t<sub>e</sub>d i<sub>n</sub> $O ( | \bar { \mathcal { I } } |$ log �) time.

## 4<sub>.</sub>2 R<sub>e</sub>d<sub>uc</sub>i<sub>ng</sub> C<sub>overe</sub>d<sub>-</sub>K<sub>eys</sub> Mi<sub>n</sub>i<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> t<sub>o</sub> M<sub>a</sub>xim<sub>u</sub>m-Err<sub>o</sub>r M<sub>a</sub>ximiz<sub>a</sub>ti<sub>o</sub>n

We now relate the maximum-error maximization problem (Definition 2) to the covered-legitimate-keys minimization problem (Definition 5) through the following theorem.

Theorem 4. Let $\mathcal { P } _ { \mathrm { c o v } } ^ { * }$ b<sub>e an op</sub>ti<sub>ma</sub>l <sub>so</sub>l<sub>u</sub>ti<sub>on</sub> t<sub>o</sub> th<sub>e covere</sub>d<sub>-</sub> legitimate-keys minimization problem (Definition 5), and let $C ^ { * }$ denote the corresponding number of covered legitimate keys. If $C ^ { * } < n ,$ l<sub>e</sub>t $\mathcal { P } _ { \mathrm { e r r } } ^ { * }$ b<sub>e an op</sub>ti<sub>ma</sub>l <sub>so</sub>l<sub>u</sub>ti<sub>on</sub> t<sub>o</sub> th<sub>e max</sub>i<sub>mum-error</sub> maximization problem (Definition 2) for the key set $\mathcal { K } _ { \le C ^ { * } + 1 } : =$ $\{ k _ { 1 } , k _ { 2 } , \dotsc , k _ { C ^ { * } + 1 } \}$ <sub>.</sub> Th<sub>en,</sub> $\mathcal { P } _ { \mathrm { e r r } } ^ { * }$ i<sub>s a</sub>l<sub>so an op</sub>ti<sub>ma</sub>l <sub>so</sub>l<sub>u</sub>ti<sub>on</sub> t<sub>o</sub> th<sub>e</sub> <sub>covere</sub>d<sub>-</sub>l<sub>eg</sub>iti<sub>ma</sub>t<sub>e-</sub>k<sub>eys</sub> <sub>m</sub>i<sub>n</sub>i<sub>m</sub>i<sub>za</sub>ti<sub>on</sub> <sub>pro</sub>bl<sub>em.</sub>

Proof of Theorem 4. From the definition of $\mathcal { P } _ { \mathrm { c o v } } ^ { * }$ and $\mathcal { P } _ { \mathrm { e r r } } ^ { * }$ , we have $E ( { \mathcal { K } } _ { \leq C ^ { * } + 1 } \cup { \mathcal { P } } _ { \mathrm { e r r } } ^ { * } ) \geq E ( { \mathcal { K } } _ { \leq C ^ { * } + 1 } \cup { \mathcal { P } } _ { \mathrm { c o v } } ^ { * } ) > \varepsilon .$ . Therefore, we have $C _ { \varepsilon } ( \mathcal { K } , \mathcal { P } _ { \mathrm { e r r } } ^ { * } ) \leq C ^ { * } ;$ thus, $\mathcal { P } _ { \mathrm { e r r } } ^ { * }$ is optimal for Definition 5. □

This theorem shows that, to find an optimal solution for Definition 5, it sufices to search for optimal poison sets for Definition 2 over prefixes of the legitimate keys. Thus, we can leverage the insights from Section 3 to design eficient poisoning methods.

## 4<sub>.</sub>3 M<sub>e</sub>th<sub>o</sub>d<sub>s</sub> f<sub>o</sub>r Obt<sub>a</sub>inin<sub>g</sub> P<sub>o</sub>i<sub>so</sub>n S<sub>o</sub>l<sub>u</sub>ti<sub>o</sub>n<sub>s</sub>

Based on Theorem 4, we propose three methods for constructing poison sets. Let $c : = C _ { \varepsilon } ( \mathcal { K } , \varnothing )$ denote the number of covered legitimate keys without poisoning. The main takeaway is that, while Consecutive is efective, it takes quadratic time in $c ; \mathrm { D I } \cdot$ Consecutive reduces this to $O ( c \log c )$ while achieving comparable performance in practice (as shown in Section 6.3).

Optimal Method. By Theorems 1 and 4, an optimal solution can be obtained by enumerating $O \left( { \binom { 2 c - 3 + \lambda } { \lambda } } \right)$ candidates, as in Section 3. Since each candidate is evaluated in $O ( c + \lambda )$ time, the total running time is $O \left( \binom { 2 c - 3 + \lambda } { \lambda } \left( c + \lambda \right) \right)$

Consecutive Method. Motivated by Consecutive proposed in Section 3, we enumerate all consecutive poison sets and select the one minimizing $C _ { \varepsilon } ( \mathcal { K } , \mathcal { P } )$ . By Theorems 3 and 4, it sufices to consider $O ( c )$ candidates whose left or right boundary is adjacent to a legitimate key. Evaluating each candidate takes $O ( c + \lambda )$ time, giving a total running time of $O \left( c ( c + \lambda ) \right)$ ).

Discrete-Intercept Consecutive Method (DI-Consecutive). Because � can be large (indeed, prior work [15, 27] has shown that � scales as $\Theta ( \varepsilon ^ { 2 } ) )$ ), the quadratic-time Consecutive method can be impractical, especially when it must be applied repeatedly. We therefore propose DI-Consecutive, which approximates $C _ { \varepsilon } ( \mathcal { K } , \mathcal { P } )$ using an extension of SwingFilter [12]. Whereas SwingFilter fixes the intercept at $k _ { 1 }$ to 0, our extension considers a predefined set I of candidate intercepts (Figure 4). After precomputation, sparse tables evaluate each consecutive candidate in $O ( | \boldsymbol { \mathcal { I } } | \log c )$ time (Figure 5), making the total running time $O ( | \boldsymbol { \mathcal { I } } | c \log c )$ . As we show later in Section 6.3, approximately 40 intercept candidates sufice to closely match Consecutive in our experiments. Further details are provided in Appendix B.

## 5 POISONING TO MAXIMIZE $m _ { \mathrm { o p t } }$

In this section, we propose a poisoning attack that increases the number of segments in an optimal PLA over a CDF, which is used as the bottom-level PLA of the PGM-index. We call our attack PGM-attack. PGM-attack repeatedly applies the attack from the previous section, which minimizes the number of covered keys, thereby increasing the number of segments in the optimal PLA. Note that PGM-attack targets not the PGM-index specifically, but rather the PLA construction algorithm that underlies a wide range of learned indexes [8, 18, 21, 27].

## 5.1 Problem Settin<sub>g</sub>

We first formalize the optimal PLA on CDFs and define $m _ { \mathrm { o p t } } ( \boldsymbol { X } , \varepsilon )$ i.e., the minimum number of segments.

Definition 6 (PLA on CDFs). Let $\mathcal { X } = \{ x _ { 1 } , x _ { 2 } , \ldots , x _ { N } \}$ b<sub>e a</sub> multiset ofnatural numbers such that $x _ { 1 } \le x _ { 2 } \le \cdots \le x _ { N }$ . For <sub>eac</sub>h $i \in [ N ]$ , define the rank by $r _ { i } : = i ,$ <sub>an</sub>d l<sub>e</sub>t $\varepsilon \geq 0 .$ A segment is a triple $( s , e , f )$ <sub>w</sub>h<sub>ere</sub> $1 \leq s \leq e \leq N$ <sub>an</sub>d $f$ is a linearfunction such that ma $\mathrm { x } _ { i \in \{ s , . . . , e \} } | f ( x _ { i } ) - r _ { i } | \leq \varepsilon .$ A PLA with � segments for X is a sequence of segments ${ \bigl ( } ( s _ { 1 } , e _ { 1 } , f _ { 1 } ) , \ldots , ( s _ { m } , e _ { m } , f _ { m } ) { \bigr ) }$ <sub>suc</sub>h th<sub>a</sub>t $s _ { 1 } ~ = ~ 1 , e _ { m } ~ = ~ N _ { \mathrm { { ; } } }$ <sub>,</sub> <sub>an</sub>d $s _ { j + 1 } = e _ { j } + 1$ for all $j \in [ m - 1 ]$ The objective is to minimize the number of segments �. The minimum number of segments is denoted by $m _ { \mathrm { o p t } } ( \boldsymbol { X } , \varepsilon )$

This problem can be solved in O (�) time using the algorithm of [16, 31]. We next define the poisoning problem of maximizing $m _ { \mathrm { o p t } }$ for the PGM-index.

Definition 7 (Segment Maximization Problem). Let   
$\mathcal { K } = \{ k _ { 1 } , k _ { 2 } , \ldots , k _ { n } \} \subset \mathbb { N }$ b<sub>e �</sub> di<sub>s</sub>ti<sub>nc</sub>t l<sub>eg</sub>iti<sub>ma</sub>t<sub>e</sub> k<sub>eys suc</sub>h   
th<sub>a</sub>t $k _ { 1 } < k _ { 2 } < \cdots < k _ { n }$ <sub>,</sub> l<sub>e</sub>t $\varepsilon \geq 0 ,$ , and let � ∈ N.   
The segment maximization problem is to find a poison set maxi  
mizing the number of segments:

$$
\operatorname * { a r g m a x } _ { \mathcal { P } \mathrm { ~ s . t . ~ } | \mathcal { P } | \leq \lambda , \mathcal { P } \subseteq \{ k _ { 1 } , k _ { 1 } + 1 , \ldots , k _ { n } \} \backslash \mathcal { K } } m _ { \mathrm { o p t } } ( \mathcal { K } \cup \mathcal { P } , \varepsilon ) .\tag{14}
$$

Dificulty of the Problem. This problem is challenging for two main reasons. Downstream dependence: The placement of poison keys in the early part of the key space afects the entire sequence of subsequent segments. Thus, the contribution of each poison key cannot be evaluated independently, which makes the problem highly non-local. Computational constraints: In typical deployments, the key set size � is often on the order of $1 0 ^ { 8 }$ , and the poison budget � is often on the order of $1 0 ^ { 6 }$ . Moreover, depending on the choice of $\varepsilon ,$ the number of keys covered by a single segment can also be large, $\mathrm { e . g . }$ , on the order of $1 0 ^ { 4 }$ or more. Let � denote this average number. Naive dynamic programming or direct use of heavy subroutines such as Consecutive (Section 4.3) incurs a total computational cost of Ω(���) and is therefore practically infeasible.

Algorithm 1 PGM-attack   
Input: Legitimate key set $\mathcal { K } ,$ total poison budget �   
Output: Poison key set $\mathcal { P }$   
1: $n \gets | \mathcal { K } |$   
2: $\theta _ { 0 }  \mathrm { g e t T h e t a 0 } ( \mathcal { K } )$ ⊲ initial value of �   
3: $s \gets 1$ ⊲ segment start (index in K)   
4: $\mathcal { P }  \emptyset$   
<sub>5:</sub> while $s \leq n$ d<sub>o</sub>   
6: $\theta  \mathrm { g e t T h e t a } ( \theta _ { 0 } , ( s - 1 ) / n , | \mathcal { P } | / \lambda )$ ⊲ Equation (15)   
7: $C ^ { * } \gets C _ { \varepsilon } ( \mathcal { K } [ s , s + 1 , . . . ] , \emptyset )$ ⊲ Equation (12)   
8: $\lambda ^ { * }  0 , \mathcal { P } _ { \mathrm { s e g } } ^ { * }  \varnothing$   
9: $\mathcal { K } _ { \mathrm { s e g } }  \mathcal { K } [ s , s + 1 , \dotsc , s + C ^ { * } - 1 ]$   
10: f<sub>or</sub> $\lambda _ { \mathrm { c a n d } } \in \Lambda$ d<sub>o</sub>   
11: $\mathbf { i f } \ | \mathcal { P } | + \lambda _ { \mathrm { c a n d } } > \lambda$ th<sub>en</sub>   
12: continue   
13: $\mathcal { P } _ { \mathrm { c a n d } }  \mathrm { D I - C o n s g c U T I V E } ( \mathcal { K } _ { \mathrm { s e g } } , \lambda _ { \mathrm { c a n d } } )$ ⊲ Section 4.3   
14: $C _ { \mathrm { c a n d } }  C _ { \varepsilon } ( { \mathcal { K } } _ { \mathrm { s e g } } , { \mathcal { P } } _ { \mathrm { c a n d } } )$   
15: if $\dot { \cdot } { \cal C } _ { \mathrm { c a n d } } + \theta \lambda _ { \mathrm { c a n d } } < C ^ { * } + \theta \lambda ^ { * }$ th<sub>en</sub>   
16: $\mathcal { P } _ { \mathrm { s e g } } ^ { * }  \mathcal { P } _ { \mathrm { c a n d } } , ~ C ^ { * }  C _ { \mathrm { c a n d } } , ~ \lambda ^ { * }  \lambda _ { \mathrm { c a n d } }$   
17: $\mathcal { P }  \mathcal { P } \cup \mathcal { P } _ { \mathrm { s e g } } ^ { * } , s  s + C ^ { * }$   
18: return $\mathcal { P }$

## 5.2 PGM-attack

Algorithm overview. To address downstream dependencies and computational constraints, we adopt a sequential heuristic that fixes segments from left to right. At each step, the algorithm selects poison keys for the current segment $( \mathrm { i . e . , }$ the segment starting from the �-th legitimate key), and then advances � to the beginning of the next segment. Specifically, we prepare a candidate set Λ for the number of poisons to be inserted into the current segment. For each $\lambda \in \Lambda ,$ we use DI-Consecutive (Algorithm 2) to eficiently generate a poison set. Among these candidates, we select the one that minimizes the cost. The full procedure is given in Algorithm 1.

Cost function. Let $C _ { \mathrm { c a n d } }$ denote the number of legitimate keys covered after inserting � poison keys. We define the cost as $C _ { \mathrm { c a n d } } + \theta \lambda _ { \mathrm { c a n d } }$ , where � is a trade-of parameter justified as follows. Suppose that each poison key reduces the number of covered legitimate keys by $\hat { \theta }$ in expectation. Assigning $\lambda _ { \mathrm { c a n d } }$ poison keys to the current segment has two efects: it leaves $C _ { \mathrm { c a n d } }$ legitimate keys covered by the current segment, while consuming $\lambda _ { \mathrm { c a n d } }$ poison keys that could otherwise reduce the coverage of subsequent segments by $\hat { \theta } \lambda _ { \mathrm { c a n d } }$ keys in expectation. Thus, $C _ { \mathrm { c a n d } } + { \hat { \theta } } \lambda _ { \mathrm { c a n d } }$ represents the expected number of covered legitimate keys that this candidate fails to remove from coverage. Minimizing it therefore minimizes the expected number oflegitimate keys covered per segment, equivalently increasing the expected number of segments.

Tuning �. Following this intuition, we empirically initialize � as the average reduction in covered legitimate keys per poison. We sample 100 random segments, construct poison sets for each $\lambda \in \Lambda ,$ and set $\theta _ { 0 }$ to the average reduction per poison.

We then adapt � through a multiplicative update based on the imbalance between the progress over legitimate keys and the consumption of the poison budget. Formally, we define

$$
\theta = \theta _ { 0 } e ^ { \mu ( r _ { \lambda } - r _ { n } ) } , \quad r _ { n } : = \frac { s - 1 } { n } , \quad r _ { \lambda } : = \frac { | \mathcal { P } | } { \lambda } .\tag{15}
$$

The parameter $\mu > 0$ controls the strength of the correction. When $r _ { \lambda } > r _ { n } ,$ , the poison budget is consumed faster than the algorithm progresses through the legitimate keys, so � increases and discourages larger values of �. Conversely, when $r _ { \lambda } < r _ { n } , \theta$ decreases and favors larger values of �. Thus, � acts as an adaptive penalty coeficient that balances the two resources throughout the process. Based on the results of the ablation analysis detailed in Appendix D.1.3, we set $\mu = 1 0 0$ by default.

Choice of Λ. To reduce computation, we restrict the number of poisons assigned to each segment to a small candidate set Λ. $_ { \mathrm { A s } }$ shown in Section 6.3, the reduction in covered legitimate keys typically saturates around $\lambda = 2 \varepsilon + 1$ , beyond which additional poisons provide little or no benefit. This is because, under a mild sparsity assumption on the keys, placing 2�+1 poison keys adjacent to the first legitimate key is suficient to reduce the number of covered legitimate keys to one. Motivated by this observation, we set $\Lambda : = \{ \lfloor t ( 2 \varepsilon + 1 ) / 9 \rfloor | t = 0 , 1 , . . . , 9 \}$

## 5.3 Instance-A<sub>g</sub>nostic U<sub>pp</sub>er Bound on $m _ { \mathrm { o p t } }$

In this section, we derive an upper bound on $m _ { \mathrm { o p t } }$ . This upper bound provides quantitative insight for both attackers and defenders. From the attacker’s perspective, this upper bound indicates how much further $m _ { \mathrm { o p t } }$ could potentially be increased beyond the current solution. From the defender’s perspective, it can be used as a measure of robustness against data insertions, including both poisoning and benign insertions.

We first present the following theorem, which provides an upper bound on $m _ { \mathrm { o p t } }$ after poisoning using only minimal information about the instance $\mathcal { K } .$

Theorem 5. When $\varepsilon \ge 1 / 2 , f o r a n y \mathcal { K }$ <sub>an</sub>d ${ \mathcal { P } } ,$

$$
m _ { \mathrm { o p t } } ( { \mathcal { K } } \cup { \mathcal { P } } , \varepsilon ) \leq m _ { \mathrm { o p t } } ( { \mathcal { K } } , \varepsilon ) + | { \mathcal { P } } | .\tag{16}
$$

Proof Sketch of Theorem 5. Given an optimal PLA for $\mathcal { K } ,$ inserting a single key � requires at most one additional segment: add a singleton segment if $\dot { p }$ lies between two segments, or split the segment containing � otherwise. Applying this argument repeatedly proves Equation (16). □

In standard PGM-index settings, $\varepsilon \ge 1$ (see [16, Def. 1]), so the assumption $\varepsilon \ge 1 / 2$ in Theorem 5 always holds. Interestingly, when $\varepsilon < 1 / 2 ,$ , Theorem 5 can fail. For example, with $\varepsilon = 0$ and $\mathcal { K } = \{ 1 , 3 , 5 , 7 \} , m _ { \mathrm { o p t } } ( \mathcal { K } , 0 ) = 1$ . However, with $\mathcal { P } = \{ 4 \}$ , we have $m _ { \mathrm { o p t } } ( \mathcal { K } \cup \mathcal { P } , 0 ) = 3$ , which is greater than $\begin{array} { r } { m _ { \mathrm { o p t } } ( \mathcal { K } , 0 ) + \vert \mathcal { P } \vert = 2 . } \end{array}$

We further show that there exists K for which the bound in Theorem 5 is tight.

Theorem 6. For any $\varepsilon \geq 0$ <sub>an</sub>d $\lambda \in \mathbb { N } ,$

$$
\exists \mathcal { K } , \mathcal { P } \ s . t . \ | \mathcal { P } | = \lambda \ \wedge \ m _ { \mathrm { o p t } } ( \mathcal { K } \cup \mathcal { P } , \varepsilon ) = m _ { \mathrm { o p t } } ( \mathcal { K } , \varepsilon ) + \lambda .\tag{17}
$$

Proof Sketch of Theorem 6. The proof is constructive. Let K consist of $\lambda + 1$ suficiently long blocks of consecutive integers separated by suficiently large gaps. Then each segment covers one block, so $m _ { \mathrm { o p t } } ( \mathcal { K } , \varepsilon ) = \lambda + 1$ . Let P be a size-� poison set with one poison at each gap midpoint. Each inserted poison key creates one additional segment, and thus $m _ { \mathrm { o p t } } ( \mathcal { K } \cup \mathcal { P } , \varepsilon ) = 2 \lambda + 1$ □

Theorem 6 shows that the upper bound in Theorem 5 is tight as an instance-agnostic bound, and hence cannot be improved without using instance-specific information.

## 5.4 Instance-De<sub>p</sub>endent U<sub>pp</sub>er Bound on $m _ { \mathrm { o p t } }$

By exploiting the structure ofK, we derive a tighter upper bound on �<sub>opt</sub> than the instance-agnostic bound in Section 5.3. We relax the problem (Definition 7) to obtain an eficiently computable bound with theoretical guarantees. Our approach consists of three steps: relaxing the problem by partitioning K into blocks, deriving perblock upper bounds, and globally allocating the poison budget. We briefly describe each step below; full details are provided in Appendix C.

Relaxation. We weaken the PLA construction by partitioning K into blocks and fixing the slope within each block in advance. We partition K using an ��-PLA with � ≥ 1 and fix each block’s slope to that of the corresponding segment. We evaluate the upper bound for $\alpha \in \{ 1 . 0 , 1 . 2 , 1 . 4 , 1 . 6 , 1 . 8 , 2 . 0 \}$ and take the minimum. We also strengthen the attacker by allowing duplicate poisons.

Per-block upper bound. Fixing the slope within each block allows us to derive a closed-form lower bound on the number of poisons required for a segment to span any pair of keys. We use these pairwise costs as edge weights in a DAG, where each path represents a segmentation and its weight lower-bounds the required poison budget. The edge weights satisfy the Monge property [2], allowing eficient shortest-path computation via LARSCH [25]. Combined with Lagrangian relaxation [3], this yields an upper bound on the number of segments achievable under a given poison budget. The bound is parameterized by a Lagrange parameter $\nu > 0 ;$ we evaluate it for each � ∈ {1, 2, 3, 4, 5, 10, 20, 40, 80, 160} and take the minimum.

Allocating poisons across blocks. Each per-block bound is concave in the allocated budget, so the global allocation reduces to a concave knapsack problem. We solve it by greedily assigning each poison to the block with the largest marginal gain [14].

## 6 EXPERIMENTS

This section evaluates our poisoning attacks and their impact on index performance. We first study attacks on CDF linear regression under the maximum-error loss (Section 6.2) and attacks that minimize the number of covered legitimate keys (Section 6.3). We then evaluate the efect of PGM-attack on the number of PLA segments (Section 6.4) and the performance of various indexes (Section 6.5).

![](images/616aea3116d760c4748e17cd557bb326f73e8e285cd8d1e233dc63d568bddcbb.jpg)  
Fi<sub>gure</sub> 6<sub>:</sub> M<sub>ean max</sub>i<sub>mum error across</sub> 100 <sub>see</sub>d<sub>s.</sub> Whil<sub>e</sub> Ra<sub>n-</sub> <sub>dom</sub> b<sub>are</sub>l<sub>y c</sub>h<sub>anges</sub> th<sub>e max</sub>i<sub>mum error, our me</sub>th<sub>o</sub>d<sub>s</sub> i<sub>n-</sub> <sub>crease</sub> it <sub>su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>ll<sub>y.</sub>

## 6.1 Ex<sub>p</sub>erimental Setu<sub>p</sub>

We implemented all methods in C++ and ran experiments on a Linux machine with an Intel Core i9-11900H CPU @ 2.50 GHz and 64 GB of RAM. We compiled the code with GCC 9.4.0 and the -O3 flag. We used eight threads for poison generation to reduce wall-clock time. In contrast, index construction and query-time measurements were performed using a single thread for reproducibility.

Datasets. We use standard real-world benchmarks for learned indexes together with synthetic datasets. From SOSD [29], we use Amzn, Osmc, and Face; from the ALEX study [10], we use YCSB, Longitudes, and Longlat. We scale the real-valued Longitudes and Longlat keys to $[ 0 , 2 ^ { 6 3 } ]$ and round them to integers. Amzn and Osmc contain 800M keys, while the other real-world datasets contain 200M keys each. For the synthetic datasets Uniform, Normal, and Lognormal, we draw 200M keys i.i.d., scale them to $[ 0 , 2 ^ { 6 3 } ]$ , and round them to integers. All datasets are duplicate-free.

## 6.2 Maximum-Error Maximization

We evaluate the maximum-error poisoning attacks introduced in Section 3. For each dataset in Section 6.1, we construct a legitimate key set by extracting � consecutive keys, reflecting that each PLA segment covers a contiguous key range. We choose the starting position uniformly at random and repeat the experiment with 100 seeds. We fix $n = 1 0 0$ and vary the poison budget as $\lambda \in \{ 2 , 4 , 6 , \dots , 2 0 \}$

Methods. We evaluate Greedy, Consecutive, and Duplicate-Opt from Section 3. We compare them with two baselines: Random, which uniformly samples � keys from the allowed integers, and Random-Adjacent, which uniformly samples from integers adjacent to legitimate keys, as motivated by Theorem 1.

Results. Figure 6 shows the maximum error averaged over 100 seeds. While Random has little efect, Greedy, Consecutive, and Duplicate-Opt substantially increase the maximum error. Their curves nearly overlap, and the increase is approximately linear in �, with a slope of about 0.4 across all datasets. Random-Adjacent sometimes outperforms Random, but remains consistently less efective than these three methods.

![](images/990e209653c2e9c8b11b4197ee68780e9bc55f5d62e0beb148400e6cf2f3081c.jpg)  
Fi<sub>g</sub>ure 7: Mean number of covered le<sub>g</sub>itimate ke<sub>y</sub>s � versus � f<sub>or</sub> $\varepsilon = 6 4 .$ <sub>.</sub> Consecutive <sub>a</sub>nd DI-Consecutive <sub>ac</sub>hi<sub>eve</sub> l<sub>owe</sub>r � th<sub>an</sub> th<sub>e ran</sub>d<sub>om</sub> b<sub>ase</sub>li<sub>nes.</sub>

Consecutive is optimal in almost all cases. Let $E _ { \mathrm { C } }$ and $E _ { \mathrm { D } }$ denote the maximum errors achieved by Consecutive and Duplicate-Opt, respectively. Among all 9,000 cases (9 datasets, 10 values of �, and 100 seeds), we observe $E _ { \mathrm { D } } = E _ { \mathrm { C } }$ in 8,994 cases. In the remaining six cases, the gap satisfies $E _ { \mathrm { D } } - E _ { \mathrm { C } } \leq 0 . 8 7$ . Because $E _ { \mathrm { D } }$ upperbounds the optimum of the original problem, these results show that Consecutive attains the optimum in the vast majority of cases and remains close to it in all remaining cases.

We next compare Consecutive with Greedy. Let $E _ { \mathrm { G } }$ denote the maximum error achieved by Greedy. We have $E _ { \mathrm { C } } \geq E _ { \mathrm { G } }$ in all 9,000 cases, with strict inequality in 3,105 cases and a maximum gap of 4.4. Moreover, Consecutive is substantially faster: it runs in O(�(�+�)) time, whereas Greedy requires $O ( \lambda ( n + \lambda ) ^ { 2 } )$ time. Thus, Consecutive is both more efective and eficient than Greedy.

Key takeaway. Consecutive eficiently finds a globally optimal maximum-error attack in almost all cases.

## 6<sub>.</sub>3 C<sub>overe</sub>d<sub>-</sub>L<sub>eg</sub>iti<sub>ma</sub>t<sub>e-</sub>K<sub>eys</sub> Mi<sub>n</sub>i<sub>m</sub>i<sub>za</sub>ti<sub>on</sub>

We evaluate the covered-legitimate-keys minimization attack introduced in Section 4. For each dataset and seed, we construct the legitimate key set K (see Definition 5) by selecting a random position and taking the sufix starting from that position. Thus, the number of keys covered before poisoning varies across datasets and seeds. We set $\varepsilon = 6 4 _ { \mathrm { : } }$ vary $\lambda \in \{ 1 6 i \mid i = 1 , . . . , 1 2 \}$ , and use 20 random seeds. For DI-Consecutive, we use $\mathcal { T } = \{ t \varepsilon / 2 0 \mid t = - 2 0 , - 1 9 , . . . , 2 0 \}$

Methods. We evaluate Consecutive and DI-Consecutive from Section 4. We also include Random and Random-Adjacent, defined as in Section 6.2, as baselines.

Results. Figure 7 shows the mean number of covered legitimate keys before and after poisoning. Random has little efect, and Random-Adjacent achieves only a modest reduction. By contrast, Consecutive and DI-Consecutive substantially reduce the covered-key count and achieve nearly identical results.

When $\lambda \ \leq \ 2 \varepsilon .$ , the number of covered keys decreases almost linearly with �. In contrast, when $\lambda > 2 \varepsilon ,$ , the number of covered keys converges to nearly one and does not decrease further. This

![](images/aa3fc3320ef22aa60c2ca2dc3822911f21a178103ffac185cf9881132ee329ae.jpg)  
Fi<sub>gure</sub> 8<sub>:</sub> P<sub>o</sub>i<sub>son</sub> <sub>genera</sub>ti<sub>on</sub> ti<sub>me.</sub> DI<sub>-</sub>Co<sub>n</sub>sec<sub>u</sub>ti<sub>v</sub>e i<sub>s</sub> <sub>over</sub> 1,000× faster than Consecutive.

T<sub>a</sub>bl<sub>e</sub> 1<sub>:</sub> $m _ { \mathrm { o p t } }$ b<sub>e</sub>f<sub>o</sub>r<sub>e</sub>/<sub>a</sub>ft<sub>e</sub>r <sub>po</sub>i<sub>so</sub>nin<sub>g, a</sub>nd in<sub>s</sub>t<sub>a</sub>n<sub>ce</sub>-d<sub>epe</sub>nd<sub>e</sub>nt u<sub>pp</sub>er <sup>b</sup>oun<sup>d</sup>s $( \varepsilon = 1 2 8 )$ . Original reports $m _ { \mathrm { o p t } }$ b<sub>e</sub>f<sub>ore</sub> <sub>po</sub>i<sub>son-</sub> in<sub>g.</sub> PGM-attack in<sub>c</sub>r<sub>eases</sub> $m _ { \mathrm { o p t } }$ by up to 120× under 10% poisoning. The instance-specific upper bound is at most 1.92× th<sub>e</sub> $m _ { \mathrm { o p t } }$ achieved b<sub>y</sub> PGM-attack.

<table><tr><td colspan="2"></td><td colspan="2">λ = 0.01 n</td><td colspan="2"> $\lambda = 0 . 1 n$ </td></tr><tr><td>Dataset</td><td>Original</td><td>PGM-ATTACK</td><td>Instance UB</td><td>PGM-ATTACK</td><td>Instance UB</td></tr><tr><td>Amzn</td><td>268K</td><td>335K (1.25×)</td><td>619K (2.31×)</td><td>670K (2.50×)</td><td>1.23M (4.60×)</td></tr><tr><td>Osmc</td><td>668K</td><td>723K (1.08×)</td><td>1.23M (1.85×)</td><td>1M (1.51×)</td><td>1.93M (2.88×)</td></tr><tr><td>Face</td><td>256K</td><td>283K (1.10×)</td><td>444K (1.73×)</td><td>374K (1.46×)</td><td>668K (2.60×)</td></tr><tr><td>YCSB</td><td>663</td><td>10.3K (15.5×)</td><td>17.6K (26.6×)</td><td>79.8K (120×)</td><td>130K (196×)</td></tr><tr><td>Longitudes</td><td>13.7K</td><td>22.3K (1.63×)</td><td>41.5K (3.02×)</td><td>92.4K (6.73×)</td><td>154K (11.2×)</td></tr><tr><td>Longlat</td><td>58.1K</td><td>67.7K (1.16×)</td><td>122K (2.11×)</td><td>138K (2.37×)</td><td>245K (4.22×)</td></tr><tr><td>Uniform</td><td>3.42K</td><td>12.9K (3.78×)</td><td>23.7K (6.91×)</td><td>83.7K (24.5×)</td><td>136K (39.8×)</td></tr><tr><td>Normal</td><td>3.5K</td><td>13K (3.70×)</td><td>23.7K (6.78×)</td><td>83.5K (23.8×)</td><td>136K (38.9×)</td></tr><tr><td>Lognormal</td><td>3.49K</td><td>12.9K (3.71×)</td><td>23.7K (6.81×)</td><td>83.4K (23.9×)</td><td>136K (39.1×)</td></tr></table>

observation motivates our choice of Λ in Section 5.2, where we use 10 evenly spaced integers from 0 to $2 \varepsilon + 1$

Figure 8 compares the poison generation times of Consecutive and DI-Consecutive with $\lambda = \varepsilon .$ The points show the means over seeds, and the shaded regions indicate the 5th–95th percentiles. DI-Consecutive is consistently much faster than Consecutive. The speedup becomes particularly pronounced when � is large or when the number of legitimate keys before poisoning is large, reaching over 1,000× in the largest cases.

Key takeaway. DI-Consecutive achieves nearly the same reduction in covered legitimate keys as Consecutive, while being over three orders of magnitude faster in the largest cases.

## 6<sub>.</sub>4 PLA S<sub>egmen</sub>t M<sub>ax</sub>i<sub>m</sub>i<sub>za</sub>ti<sub>on</sub>

We evaluate how much PGM-attack increases $m _ { \mathrm { o p t } } ,$ , along with the instance-dependent upper bounds (Instance UB) from Section 5. We use all keys in each dataset: Amzn and Osmc contain 800M keys; the others, 200M.

T<sub>a</sub>bl<sub>e</sub> 2<sub>:</sub> R<sub>esu</sub>lt<sub>s</sub> f<sub>or</sub> $m _ { \mathrm { o p t } }$ after <sub>p</sub>oisonin<sub>g</sub> at diferent <sub>�</sub> values (� = 0.1�). Larger � tends to yield a higher increase ratio.
<table><tr><td>Dataset</td><td> $\varepsilon = 1 6$ </td><td> $\varepsilon = 3 2$ </td><td> $\varepsilon = 6 4$ </td><td> $\varepsilon = 1 2 8$ </td></tr><tr><td>Amzn</td><td>12.1M (1.52×)</td><td>4.79M (1.94×)</td><td>1.73M (2.17×)</td><td>670K (2.50×)</td></tr><tr><td>Osmc</td><td>9.18M (1.49×)</td><td>4.3M (1.49×)</td><td>2.06M (1.50×)</td><td>1M (1.51×)</td></tr><tr><td>Face</td><td>3.08M (1.45×)</td><td>1.53M (1.45×)</td><td>760K (1.45×)</td><td>374K (1.46×)</td></tr><tr><td>YCSB</td><td>810K (11.6×)</td><td>361K (14.4×)</td><td>169K (24.2×)</td><td>79.8K (120×)</td></tr><tr><td>Longitudes</td><td>831K (5.05×)</td><td>380K (6.07×)</td><td>185K (6.66×)</td><td>92.4K (6.73×)</td></tr><tr><td>Longlat</td><td>1.1M (2.44×)</td><td>537K (2.45×)</td><td>271K (2.42×)</td><td>138K (2.37×)</td></tr><tr><td>Uniform</td><td>930K (4.57×)</td><td>391K (7.38×)</td><td>177K (13.2×)</td><td>83.7K (24.5×)</td></tr><tr><td>Normal</td><td>929K (4.58×)</td><td>391K (7.39×)</td><td>177K (13.1×)</td><td>83.5K (23.8×)</td></tr><tr><td>Lognormal</td><td>929K (4.58×)</td><td>391K (7.39×)</td><td>177K (13.1×)</td><td>83.4K (23.9×)</td></tr></table>

Table 3: Poison generation time (hours). $\varepsilon = 1 2 8 , \lambda = 0 . 1 n .$
<table><tr><td>Dataset</td><td>Time (h)</td></tr><tr><td>Amzn</td><td>12.3</td></tr><tr><td>Osmc</td><td>5.0</td></tr><tr><td>Face</td><td>1.0</td></tr></table>

<table><tr><td>Dataset</td><td>Time (h)</td></tr><tr><td>YCSB</td><td>14.7</td></tr><tr><td>Longitudes</td><td>5.3</td></tr><tr><td>Longlat</td><td>2.6</td></tr></table>

<table><tr><td>Dataset</td><td>Time (h)</td></tr><tr><td>Uniform</td><td>7.3</td></tr><tr><td>Normal</td><td>7.1</td></tr><tr><td>Lognormal</td><td>7.2</td></tr></table>

<sup>P</sup>o<sup>i</sup>son<sup>i</sup>ng <sup>I</sup>mpac<sup>t</sup> on $m _ { \mathrm { o p t } }$ . Table 1 reports $m _ { \mathrm { o p t } }$ before and after PGM-attack poisoning for $\varepsilon = 1 2 8 ,$ , with parenthesized values indicating the factor relative to pre-poisoning $m _ { \mathrm { o p t } } .$ . PGM-attack can increase $m _ { \mathrm { o p t } }$ by large factors. On YCSB, PGM-attack yields a 15.5× increase in $m _ { \mathrm { o p t } }$ at 1% poisoning $( \lambda = 0 . 0 1 n )$ and up to 120× at 10% poisoning $( \lambda = 0 . 1 n )$ . Even on datasets other than YCSB, PGM-attack increases $m _ { \mathrm { o p t } }$ by factors ranging from 1.46× to 24.5× with 10% poisoning. In contrast, Random and Random-Adjacent yield increases of at most 1.30× on the other datasets and 2.42× even on YCSB (see Appendix D). These results show PGM-attack increases $m _ { \mathrm { o p t } }$ far more efectively than naive random insertions.

Upper bounds. Table 1 also reports the Instance UB. Across all instances, the Instance UB is at most 1.92× the segment count PGMattack achieves, certifying that PGM-attack achieves at least $1 / 1 . 9 2 \approx 5 2 \%$ of the optimum on every instance. The remaining gap may reflect suboptimality of the attack, looseness of the bound, or both. The Instance UB is more than an order of magnitude tighter than the instance-agnostic upper bound; for example, on Amzn, the instance-agnostic upper bound is 80M, whereas the Instance UB is only 1.23M. These results demonstrate that exploiting instancespecific structure enables substantially tighter upper bounds.

Dependence on �. Table 2 reports the poisoning impact for vary ing � with $\lambda = 0 . 1 n ;$ parenthesized values indicate increases relative to the pre-poisoning $m _ { \mathrm { o p t } }$ . Larger � generally yields a larger increase ratio, as can be explained as follows. Let <sup>ˆ</sup>� denote the average persegment reduction in legitimate keys caused by one poison, and let $m _ { \mathrm { o p t } }$ and $m _ { \mathrm { o p t } } ^ { \prime }$ denote the segment counts before and after poisoning, respectively. Since each segment receives on average $\lambda / m _ { \mathrm { o p t } } ^ { \prime }$ poisons while covered legitimate keys drop from $n / m _ { \mathrm { o p t } }$ to $n / m _ { \mathrm { o p t } } ^ { \prime } ,$ we get $\hat { \theta } \lambda / m _ { \mathrm { o p t } } ^ { \prime } \approx n / m _ { \mathrm { o p t } } - n / m _ { \mathrm { o p t } } ^ { \prime } , \mathrm { i . e . , } m _ { \mathrm { o p t } } ^ { \prime } / m _ { \mathrm { o p t } } \approx 1 + \hat { \theta } \lambda / n . \mathrm { A s }$ shown in Figure 7, covered legitimate keys decrease approximately linearly to 1 at $\lambda \approx 2 \varepsilon ,$ so $\hat { \theta } \approx c / ( 2 \varepsilon )$ , where � is the pre-poisoning covered keys per segment. Since prior work [15, 27] suggests � scales superlinearly with � $( { \mathrm { e . g . , ~ } } c = \Theta ( \varepsilon ^ { 2 } )$ under certain distributional assumptions), $m _ { \mathrm { o p t } } ^ { \prime } / m _ { \mathrm { o p t } }$ increases with �. This implies a trade-of: larger � enables the superlinear compression that makes the PGM-index attractive, but also increases poisoning sensitivity.

Poison Generation Time. Table 3 reports poison-key generation time for $\varepsilon = 1 2 8 , \lambda = 0 . 1 n .$ . Across datasets, generation takes 1.0– 14.7 hours. Although non-negligible, this cost remains practical for an ofline attacker, since the poison keys can be prepared in advance and inserted before index construction; it is therefore a one-time cost incurred per target instance. Generation time tends to grow larger on datasets where poisoning was more efective, e.g., YCSB. This is because PGM-attack processes keys left to right, and larger per-segment shortening allows the next segment to start earlier, forcing repeated recomputation over already-processed regions.

Key takeaways. PGM-attack markedly increases $m _ { \mathrm { o p t } } ,$ especially when � is large or the legitimate-key instance has a small $m _ { \mathrm { o p t } }$ . Our upper bound certifies PGM-attack achieves at least 52% of the optimum on all instances.

## 6.5 Im<sub>p</sub>act on Index Performance

We evaluate the impact of poison keys generated by PGM-attack on various index structures. We use all keys in each dataset. We report only the results on real-world datasets in the main paper; the complete results are provided in Appendix D.

Poisoning Methods. In addition to our PGM-attack, we evaluate Random, which samples poison keys uniformly at random. This comparison isolates the efect of deliberately selecting poison keys from that of merely increasing the number of keys.

Indexes. In addition to the PGM-index, we evaluate the following index structures: FT (FITing-Tree) [18], an early representative PLA-based learned index; PGM++ [27], a state-of-the-art PLAbased learned index extending the PGM-index; RMI [24], the first learned index and a representative non-PLA-based approach; and B+-tree [5], a representative classical index structure.

Evaluation Metrics. We evaluate index size and query time. Index size is defined as the memory footprint ofthe index structure. Query time is the average time required to look up a uniformly sampled key in the sorted array. We measure query time end to end; for example, for the PGM-index, it includes both the evaluation of the hierarchical PLA models and the last-mile binary search around the final predicted position. For each method, we execute ten runs of 10<sup>6</sup> queries and report the median average query time per query.

Impact on the PGM-index. Table 4 reports the index size and query time under $\varepsilon = 1 2 8$ and a 10% poison budget $( \lambda = 0 . 1 n )$ together with their ratios to the legitimate case.

For the PGM-index, PGM-attack significantly increases the index size. Because its memory footprint is dominated by the bottomlevel PLA, whose size is approximately proportional to $m _ { \mathrm { o p t } } ,$ , the index grows by up to 120×. This extreme increase occurs on YCSB, whose legitimate index is exceptionally compact; on the other realworld datasets, the increase remains 1.46–6.73×. These increases are substantially larger than those caused by random poisoning, which increases the index size by only 1.98× on YCSB and 1.00–1.02× on the other datasets. This demonstrates that PGM-attack efectively selects keys that degrade the PGM-index. Lookup time also consistently increases, by up to 1.22×; this increase is larger than the increase caused by random poisoning, which is at most 1.03×. This increase is likely attributable to the enlarged bottom-level PLA, which may cause more cache misses.

Table 4: Im<sub>p</sub>act of PGM-attack on index size and <sub>q</sub>uer<sub>y</sub> time for � = 128 and � = 0.1�. PGM-attack increases the PGMindex size by up to 120× and query time by up to 1.22×.
<table><tr><td>Dataset Index</td><td></td><td colspan="2">Index Size [MB]</td><td colspan="2">Query Time [µs]</td></tr><tr><td></td><td></td><td>Random PGM-ATTACK</td><td></td><td></td><td>Random PGM-ATTACK</td></tr><tr><td>Amzn</td><td>PGM</td><td>4.2 (1.01×)</td><td>10.5 (2.50×)</td><td>0.81 (1.01×)</td><td>0.96 (1.20×)</td></tr><tr><td></td><td>FT</td><td>33.3 (1.01×)</td><td>58.9 (1.78×)</td><td>0.85 (1.01×)</td><td>0.95 (1.13×)</td></tr><tr><td></td><td>PGM++</td><td>4.2 (1.01×)</td><td>10.5 (2.50×)</td><td>0.74 (0.98×)</td><td>0.86 (1.15×)</td></tr><tr><td></td><td>RMI</td><td>0.19 (1.00×)</td><td>0.19 (1.00×)</td><td>0.84 (1.00×)</td><td>0.98 (1.18×)</td></tr><tr><td></td><td>B+-tree</td><td>62.3 (1.10×)</td><td>62.3 (1.10×)</td><td>0.97 (1.01×)</td><td>0.95 (0.99×)</td></tr><tr><td>Osmc</td><td>PGM</td><td>10.5 (1.01×)</td><td>15.7 (1.51×)</td><td>0.92 (1.01×)</td><td>1.03 (1.12×)</td></tr><tr><td></td><td>FT</td><td>61.2 (1.01×)</td><td>75.8 (1.25×)</td><td>0.93 (1.00×)</td><td>0.94 (1.02×)</td></tr><tr><td></td><td>PGM++</td><td>2.6 (2.02×)</td><td>1.4 (1.13×)</td><td>0.97 (0.95×)</td><td>1.02 (1.00×)</td></tr><tr><td></td><td>RMI</td><td>0.19 (1.00×)</td><td>0.19 (1.00×)</td><td>1.45 (0.97×)</td><td>1.51 (1.02×)</td></tr><tr><td></td><td>B+-tree</td><td>62.3 (1.10×)</td><td>62.3 (1.10×)</td><td>0.96 (1.04×)</td><td>0.93 (1.01×)</td></tr><tr><td>Face</td><td>PGM</td><td>4.0 (1.00×)</td><td>5.9 (1.46×)</td><td>0.76 (1.01×)</td><td>0.78 (1.02×)</td></tr><tr><td></td><td>FT</td><td>23.4 (1.00×)</td><td>29.3 (1.25×)</td><td>0.82 (0.99×)</td><td>0.85 (1.02×)</td></tr><tr><td></td><td>PGM++</td><td>1.9 (1.00×)</td><td>5.9 (3.12×)</td><td>0.73 (0.96×)</td><td>0.81 (1.06×)</td></tr><tr><td></td><td>RMI</td><td>0.19 (1.00×)</td><td>0.19 (1.00×)</td><td>1.20 (0.89×)</td><td>1.35 (1.00×)</td></tr><tr><td></td><td>B+-tree</td><td>15.6 (1.10×)</td><td>15.6 (1.10×)</td><td>0.80 (1.01×)</td><td>0.80 (1.01×)</td></tr><tr><td>YCSB</td><td>PGM</td><td>0.021 (1.98×)</td><td>1.2 (120×)</td><td>0.67 (1.03×)</td><td>0.79 (1.22×)</td></tr><tr><td></td><td>FT</td><td>0.33 (1.27×)</td><td>4.7 (18.5×)</td><td>0.65 (1.03×)</td><td>0.72 (1.15×)</td></tr><tr><td></td><td></td><td>PGM++ 0.001 (12.8×)</td><td>1.7 (35770×)</td><td>0.68 (1.22×)</td><td>0.73 (1.30×)</td></tr><tr><td></td><td>RMI</td><td>0.19 (1.00×)</td><td>0.19 (1.00×)</td><td>0.54 (1.00×)</td><td>0.72 (1.33×)</td></tr><tr><td></td><td>B+-tree</td><td>15.6 (1.10×)</td><td>15.6 (1.10×)</td><td>0.82 (1.03×)</td><td>0.80 (1.00×)</td></tr><tr><td>Longit</td><td>PGM</td><td>0.22 (1.01×)</td><td>1.4 (6.73×)</td><td>0.76 (0.99×)</td><td>0.85 (1.10×)</td></tr><tr><td></td><td>FT</td><td>1.2 (1.03×)</td><td>4.9 (4.03×)</td><td>0.67 (0.99×)</td><td>0.72 (1.06×)</td></tr><tr><td></td><td>PGM++</td><td>0.22 (1.92×)</td><td>1.7 (14.7×)</td><td>0.73 (0.98×)</td><td>0.73 (0.98×)</td></tr><tr><td></td><td>RMI</td><td>0.19 (1.00×)</td><td>0.19 (1.00×)</td><td>0.78 (1.00×)</td><td>0.85 (1.09×)</td></tr><tr><td></td><td>B+-tree</td><td>15.6 (1.10×)</td><td>15.6 (1.10×)</td><td>0.81 (1.02×)</td><td>0.80 (1.01×)</td></tr><tr><td>Longlat PGM</td><td></td><td>0.93 (1.02×)</td><td>2.2 (2.37×)</td><td>0.85 (1.01×)</td><td>0.85 (1.01×)</td></tr><tr><td></td><td>FT</td><td>5.1 (1.02×)</td><td>8.7 (1.74×)</td><td>0.72 (1.00×)</td><td>0.73 (1.01×)</td></tr><tr><td></td><td>PGM++</td><td>0.28 (1.76×)</td><td>0.34 (2.13×)</td><td>0.82 (0.87×)</td><td>0.83 (0.89×)</td></tr><tr><td></td><td>RMI</td><td>0.19 (1.00×)</td><td>0.19 (1.00×)</td><td>1.21 (0.99×)</td><td>1.27 (1.03×)</td></tr><tr><td></td><td>B+-tree</td><td>15.6 (1.10×)</td><td>15.6 (1.10×)</td><td>0.81 (1.02×)</td><td>0.80 (1.01×)</td></tr></table>

Impact on Other Indexes. Table 4 also shows that PGM-attack transfers to other PLA-based learned indexes. The index size increases by 1.25–18.5× for FITing-Tree and by 1.13–35,770× for PGM++. The extreme increase for PGM++ occurs on YCSB, where the legitimate index is exceptionally compact before poisoning. Thus, by maximizing the number of PLA segments, PGM-attack also degrades the memory eficiency of other PLA-based indexes.

The RMI has a fixed model size, so its index size remains 0.19 MB under poisoning. Its query time nevertheless increases by up to 1.33×, because keys that are dificult for PLAs to approximate also increase RMI prediction errors, enlarging the final search ranges.

In contrast, the B+-tree is largely unafected by poisoning: its size increases by only about 1.1×, and its query time remains nearly unchanged. This behavior is consistent with its Θ(�) space complexity and Θ(log �) query-time complexity, which make its performance relatively insensitive to small changes in the key distribution. Nevertheless, learned indexes can remain substantially more spaceeficient in absolute terms; for example, on Amzn, the PGM-index uses 5.9× less memory than the B+-tree. These results highlight a trade-of between the eficiency of learned indexes and the predictable robustness of the B+-tree. Our analysis helps make the behavior of learned indexes more tractable by characterizing and bounding their degradation under poisoning.

Key takeaways. PGM-attack substantially increases the PGMindex’s memory footprint and moderately increases its query time. It also transfers to other learned indexes, particularly PLA-based ones, whereas classical indexes remain largely unafected. These results demonstrate that our attack exploits the distribution sensitivity distinguishing learned indexes from traditional indexes.

## 7 RELATED WORK

## 7<sub>.</sub>1 L<sub>earne</sub>d I<sub>n</sub>d<sub>exes</sub>

Learned indexes enhance or replace classical index structures, such as the B+-tree, with machine-learned models to improve memory eficiency and/or query throughput [23]. The core idea is to approximate the cumulative distribution function (CDF) of the data with a learned regression model. Since the seminal work of Kraska et al. [23], learned indexes have been extended in various directions, including multi-dimensional data, string keys, dynamic workloads, and theoretical analyses; we refer the reader to a recent survey [4] for a comprehensive overview.

A particularly important and practical line ofresearch is based on PLA. Its explicit error control and high compressibility make PLA well suited for learned indexes. Representative examples include FITing-Tree [18], which replaces B+-tree leaves with linear models, and RadixSpline [21], which combines a PLA with a radix table for fast and memory-eficient static indexing. More recently, faster PLA construction methods such as GreedyPLA and SwingFilter have been studied, trading slight losses in memory eficiency and query performance for shorter construction times [32]. Among PLAbased indexes, the PGM-index [15, 16] is one of the most mature and practically relevant, attracting substantial research attention and adoption in real-world systems. For example, Manticore Search, an open-source search engine, uses the PGM-index for secondary indexes [28], and Infinity, an open-source AI-native database, uses PGM-based structures for filtering tasks [19].

## 7<sub>.</sub>2 Att<sub>ac</sub>k<sub>s on</sub> L<sub>earne</sub>d I<sub>n</sub>d<sub>exes</sub>

Although learned indexes achieve strong performance on typical workloads, recent studies have shown that they can be vulnerable to adversarial manipulation. Prior work has investigated attacks that degrade the performance of learned Bloom filters [11, 13, 33], learned sketches [20], learned cardinality estimators [26, 37], and learned index advisors [38, 39].

Attacks and robustness issues concerning learned indexes in the narrower sense (i.e., learned B-tree) have also received considerable attention [22, 34–36]. Kornaropoulos et al. [22] study poisoning attacks against linear regression models trained on CDFs and extend their attack to RMI. A related but distinct line of work studies algorithmic complexity attacks (ACAs) [35, 36] against ALEX [10]. Unlike our setting, in which poison keys are injected before index construction, these attacks exploit weaknesses in ALEX’s online insertion algorithms to degrade its performance. We compare against these attacks in Appendix D and find that our attack consistently causes greater performance degradation.

## 8 DISCUSSION

Designing robust learned indexes. Developing learned indexes that perform well on benign data while remaining resilient to poi soning attacks, or more generally to unfavorable data insertions, is an important direction for future work. Our results show that inserting only a small number of keys can substantially reduce the compression benefit of PGM-index, which is built on optimal PLAs. This finding suggests that worst-case performance guarantees alone are insuficient to guarantee robustness: achieving robustness requires rethinking the objective function used to construct the index. A seemingly simple defense is to pre-insert the poisoning keys identified by our method as virtual points. However, this would merely incur the attack-induced increase in index size in advance, sacrificing the benign-data eficiency that the defense is intended to preserve. Possible directions include relaxing the maximum-error constraint or using more flexible models in hard-to-approximate regions, as well as separating suspicious or hard-to-approximate keys from the main index structure.

A key challenge is that efective poisoning keys are dificult to distinguish from legitimate ones. Theorem 1 guarantees the existence of an optimal poison set consisting of consecutive blocks adjacent to legitimate keys. Therefore, classical robust regression techniques [7, 17], which suppress the influence of outliers, cannot be directly applied to regression over the CDF, as also discussed in prior work [22, 34]. Developing efective defenses will therefore likely require accounting for these problem-specific characteristics.

Improvingpoisoning attacks andupperbounds. The Instance UB is up to 1.92× the segment count achieved by PGM-attack. This gap may stem from suboptimality of the attack, looseness of the bound, or both. The attack’s suboptimality arises mainly from its greedy strategy, which selects poisoning keys segment by segment from left to right. The bound’s looseness has two sources: the relaxation and the per-block upper bound. In both cases, dynamic programming could achieve better results, but a naive implementation incurs quadratic time complexity, which is a major bottleneck.

Transferability ofPGM-attack. As shown in Table 4, the poison ing keys generated by PGM-attack remain largely efective across a diverse range of learned indexes. We attribute this transferability to the fact that PGM-attack targets not an implementation-specific property of the PGM-index, but the PLA, a fundamental approximation component, which is used in many learned indexes. Many learned-index architectures share the same underlying principle of partitioning the key space into local regions and approximating each region with a simple model. The poisoning keys generated by PGM-attack deliberately induce many local regions that are hard to approximate, thereby inflating the number of models or the size of auxiliary structures even in indexes that adopt diferent model classes or partitioning strategies.

Attacks without knowledge ofthe PLA parameter. Although our threat model is white-box and assumes that the attacker knows the error parameter �, we also evaluate the attack under the more realistic scenario in which this knowledge is unavailable. Specifically, we generate poison keys assuming $\varepsilon _ { \mathrm { a t t a c k } }$ and evaluate them against a PLA built with $\varepsilon _ { \mathrm { t r u e } } ,$ , which may difer from $\varepsilon _ { \mathrm { a t t a c k } }$ . The attack is most efective when $\varepsilon _ { \mathrm { a t t a c k } } = \varepsilon _ { \mathrm { t r u e } } ,$ , but it retains substantial efectiveness even under mismatch; complete results are provided in Appendix D. A possible explanation is that PGM-attack creates local irregularities that remain dificult to approximate with linear models across diferent PLA parameters. These results show that our white-box attack also functions as a gray-box attack, where the attacker knows the legitimate key set but not the PLA parameter. This finding provides a basis for developing attacks under more restrictive knowledge assumptions.

Beyond a security threat model. Although we formulate our study in terms of a poisoning adversary, its significance extends beyond security. Our attack can also be viewed as a worst-case analysis of data insertions into PLA-based indexes, identifying inputs that sharply increase the segment count, index size, and query time. The generated poisoning keys can therefore serve as stress tests for learned indexes and help expose performance failure modes that should be addressed in the design of robust data structures.

## 9 CONCLUSION

We presented the first systematic study of poisoning attacks against optimal PLA construction. Specifically, we investigated three problems: maximizing the maximum error in CDF linear regression, minimizing the number of legitimate keys covered under a maximumerror constraint, and maximizing the number of segments in an optimal PLA. For each problem, we developed both theoretical results and practical algorithms. Experiments on real-world and synthetic datasets showed that our attacks substantially degrade the memory eficiency of the PGM-index and other learned indexes. These results reveal that optimal segment minimization does not guarantee robustness to adversarial insertions, motivating the development of robustness-aware learned indexes.

## REFERENCES

[1] 2019. A Benchmark for Machine-generated Data Management. https://github. com/BrownBigData/MgBench.

[2] Alok Aggarwal, Maria Klawe, Shlomo Moran, Peter Shor, and Robert Wilber. 1986. Geometric applications of a matrix searching algorithm. In Proceedings of the second annual symposium on Computational geometry. 285–292.

[3] Alok Aggarwal, Baruch Schieber, and Takashi Tokuyama. 1993. Finding a mini mum weight K-link path in graphs with Monge property and applications. In Proceedings of the ninth annual symposium on Computational geometry. 189–197.

[4] Abdullah Al-Mamun, Hao Wu, Qiyang He, Jianguo Wang, and Walid G. Aref. 2025. A Survey of Learned Indexes for the Multi-dimensional Space. ACM Comput. Surv. 58 (2025), 37.

[5] Rudolf Bayer and Edward M. McCreight. 1972. Organization and maintenance of large ordered indexes. Acta Inf. 1 (1972), 17.

[6] Stephen Boyd and Lieven Vandenberghe. 2004. Convex optimization. Cambridge University Press.

[7] Danny Z Chen and Haitao Wang. 2013. Approximating points by a piecewise linear function. Algorithmica 66, 3 (2013), 682–713.

[8] Leying Chen and Shimin Chen. 2026. LINE: A Learned Index with Group-Enhanced Leaves and Cache-Optimized Inner Tree. Proc. ACM Manag. Data 4 (2026), 27.

[9] Andrew Crotty. 2026. MgBench. https://github.com/andrewcrotty/mgbench. GitHub repository. Accessed: 2026-03-10.

[10] Jialin Ding, Umar Farooq Minhas, Jia Yu, Chi Wang, Jaeyoung Do, Yinan Li, Hantian Zhang, Badrish Chandramouli, Johannes Gehrke, Donald Kossmann, David Lomet, and Tim Kraska. 2020. ALEX: An Updatable Adaptive Learned Index. In Proceedings ofthe ACM SIGMOD International Conference on Management of Data. Association for Computing Machinery, New York, NY, USA, 969–984.

[11] Fangming Dong, Pinghui Wang, Rundong Li, Xueyao Cui, Junzhou Zhao, Jing Tao, Chen Zhang, and Xiaohong Guan. 2025. Poisoning Attacks and Defenses to Learned Bloom Filters for Malicious URL Detection. IEEE Transactions on Dependable and Secure Computing 22 (2025), 3275–3288.

[12] Hazem Elmeleegy, Ahmed Elmagarmid, Emmanuel Cecchet, Walid G Aref, and Willy Zwaenepoel. 2009. Online piece-wise linear approximation of numerical streams with precision guarantees. (2009).

[13] Harman Singh Farwah, Gagandeep Singh, and Cheng Tan. 2024. Exploiting Time Channel Vulnerability of Learned Bloom Filters. In ICLR Tiny Papers Track. OpenReview.net, Vienna, Austria, 5.

[14] Awi Federgruen and Henri Groenevelt. 1986. The greedy procedure for resource allocation problems: Necessary and suficient conditions for optimality. Operations research 34, 6 (1986), 909–918.

[15] Paolo Ferragina, Fabrizio Lillo, and Giorgio Vinciguerra. 2020. Why Are Learned Indexes So Efective?. In Proceedings ofthe International Conference on Machine Learning. PMLR, Virtual, 3123–3132.

[16] Paolo Ferragina and Giorgio Vinciguerra. 2020. The PGM-index: a fully-dynamic compressed learned index with provable worst-case bounds. Proc. VLDB Endow. 13 (2020), 14.

[17] Martin A Fischler and Robert C Bolles. 1981. Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography. Commun. ACM 24 (1981), 381–395.

[18] Alex Galakatos, Michael Markovitch, Carsten Binnig, Rodrigo Fonseca, and Tim Kraska. 2019. FITing-Tree: A Data-aware Index Structure. In Proceedings ofthe ACM SIGMOD International Conference on Management ofData. ACM, Amsterdam, The Netherlands, 1189–1206.

[19] InfiniFlow. 2024. AI-native database Infinity 0.1.0 is released. https://medium. com/@infiniflowai/31a51e4ecc83 Accessed: 2026-06-16.

[20] Xuyang Jing, Xiaojun Cheng, Zheng Yan, and Xian Li. 2022. Deceiving Learningbased Sketches to Cause Inaccurate Frequency Estimation. In IEEE International Conference on Trust, Security and Privacy in Computing and Communications. IEEE, Wuhan, China, 209–216.

[21] Andreas Kipf, Ryan Marcus, Alexander van Renen, Mihail Stoian, Alfons Kemper, Tim Kraska, and Thomas Neumann. 2020. RadixSpline: a single-pass learned index. In International Workshop on Exploiting Artificial Intelligence Techniques for Data Management. Association for Computing Machinery, New York, NY, USA, 5.

[22] Evgenios M Kornaropoulos, Silei Ren, and Roberto Tamassia. 2022. The price of tailoring the index to your data: Poisoning attacks on learned index structures. In Proceedings of the ACM SIGMOD International Conference on Management of Data. ACM, Philadelphia, PA, USA, 1331–1344.

[23] Tim Kraska, Alex Beutel, Ed H. Chi, Jefrey Dean, and Neoklis Polyzotis. 2018. The Case for Learned Index Structures. In Proceedings of the ACM SIGMOD International Conference on Management of Data. Association for Computing Machinery, New York, NY, USA, 489–504.

[24] Ani Kristo, Kapil Vaidya, Ugur Çetintemel, Sanchit Misra, and Tim Kraska. 2020. The Case for a Learned Sorting Algorithm. In Proceedings ofthe ACM SIGMOD International Conference on Management ofData.

[25] Lawrence L Larmore and Baruch Schieber. 1991. On-line dynamic programming with applications to the prediction of RNA secondary structure. Journal of Algorithms 12, 3 (1991), 490–515.

[26] Yingze Li, Xianglong Liu, Dong Wang, Zixuan Wang, Hongzhi Wang, Kaixing Zhang, and Yiming Guan. 2025. Algorithmic Complexity Attacks on All Learned Cardinality Estimators: A Data-centric Approach. arXiv preprint arXiv:2507.07438 (2025).

[27] Qiyu Liu, Siyuan Han, Yanlin Qi, Jingshu Peng, Jin Li, Longlong Lin, and Lei Chen. 2025. Why Are Learned Indexes So Efective but Sometimes Inefective? Proc. VLDB Endow. 18 (2025), 13

[28] Manticore Software Ltd. 2025. Introduction | Manticore Search Manual. https: //manual.manticoresearch.com/Introduction Accessed: 2026-06-16.

[29] Ryan Marcus, Andreas Kipf, Alexander van Renen, Mihail Stoian, Sanchit Misra, Alfons Kemper, Thomas Neumann, and Tim Kraska. 2020. Benchmarking Learned Indexes. Proc. VLDB Endow. 14 (2020), 1–13.

[30] Nimrod Megiddo. 1984. Linear programming in linear time when the dimension is fixed. Journal ofthe ACM (JACM) 31, 1 (1984), 114–127.

[31] Joseph O’Rourke. 1981. An on-line algorithm for fitting straight lines between data ranges. Commun. ACM 24, 9 (1981), 574–578.

[32] Jiayong Qin, Xianyu Zhu, Qiyu Liu, Guangyi Zhang, Zhigang Cai, Jianwei Liao, Sha Hu, Jingshu Peng, Yingxia Shao, and Lei Chen. 2025. Piecewise linear approximation in learned index structures: Theoretical and empirical analysis. arXiv preprint arXiv:2506.20139 (2025).

[33] Pedro Reviriego, José Alberto Hernández, Zhenwei Dai, and Anshumali Shri vastava. 2021. Learned bloom filters in adversarial environments: A malicious URL detection use-case. In IEEE International Conference on High Performance Switching and Routing. IEEE, Paris, France, 1–6.

[34] Atsuki Sato, Martin Aumüller, and Yusuke Matsui. 2026. Mathematical Foundations of Poisoning Attacks on Linear Regression over Cumulative Distribution Functions. In Proceedin s ofthe ACM SIGMOD International Conference on Management ofData. ACM, Bengaluru, India, 25.

[35] Roei Schuster, Jin Peng Zhou, Thorsten Eisenhofer, Paul Grubbs, and Nicolas Papernot. 2025. Learned-Database Systems Security. Transactions on Machine Learning Research 2025 (2025), 23.

[36] Rui Yang, Evgenios M. Kornaropoulos, and Yue Cheng. 2023. Algorithmic Complexity Attacks on Dynamic Learned Indexes. Proc. VLDB Endow. 17 (2023), 780–793.

[37] Jintao Zhang, Chao Zhang, Guoliang Li, and Chengliang Chai. 2024. PACE: Poisoning attacks on learned cardinality estimation. Proceedings ofthe ACM on Management of Data 2 (2024), 1–27.

[38] Yihang Zheng, Chen Lin, Xian Lyu, Xuanhe Zhou, Guoliang Li, and Tianqing Wang. 2024. Robustness of Updatable Learning-based Index Advisors against Poisoning Attack. Proceedings ofthe ACM on Management ofData 2 (2024), 1–26.

[39] Wei Zhou, Chen Lin, Xuanhe Zhou, Guoliang Li, and Tianqing Wang. 2024. TRAP: Tailored Robustness Assessment for Index Advisors via Adversarial Per turbation. In IEEE International Conference on Data Engineering. IEEE, Utrecht, The Netherlands, 42–55.

![](images/5ce2e8f6a757bee02bbf5cfc35a3643e416ed2533726ee093bf1799ae4102903.jpg)  
Fi<sub>gure</sub> 9<sub>:</sub> Ill<sub>us</sub>t<sub>ra</sub>ti<sub>on o</sub>f th<sub>e proo</sub>f <sub>o</sub>f L<sub>emma</sub> 1<sub>.</sub> L<sub>e</sub>t $\begin{array} { r l } { \mathcal { B } _ { - } } & { { } : = } \end{array}$ ${ \mathcal B } ( \delta _ { - } ) , \ { \mathcal B } \ : = \ { \mathcal B } ( 0 )$ <sub>, an</sub>d $\begin{array} { r } { \mathcal { B } _ { + } \ : = \ \mathcal { B } ( \delta _ { + } ) } \end{array}$ <sub>.</sub> I<sub>n</sub> th<sub>e</sub> t<sub>rans</sub>f<sub>orme</sub>d $( 1 / w , b / w )$ <sub>space,</sub> <sub>mov</sub>i<sub>ng</sub> th<sub>e</sub> bl<sub>oc</sub>k <sub>a</sub>l<sub>ong</sub> th<sub>e</sub> k<sub>ey</sub> <sub>ax</sub>i<sub>s,</sub> $\mathbf { i . e . , }$ from B to B to $\mathcal { B } _ { + s }$ <sub>, correspon</sub>d<sub>s</sub> t<sub>o</sub> t<sub>rans</sub>l<sub>a</sub>ti<sub>ng</sub> it<sub>s</sub> f<sub>eas</sub>ibl<sub>e</sub> re<sub>g</sub>ion alon<sub>g</sub> the $b / w$ di<sub>rec</sub>ti<sub>on.</sub>

## A PROOFS

Here, we provide the proofs we omitted in the main text.

## A<sub>.</sub>1 P<sub>roo</sub>f <sub>o</sub>f L<sub>emma</sub> 1

Proof of Lemma 1. The proof proceeds in three main steps: (1) defining the feasible region in a transformed parameter space, (2) deriving the translation of this region when the block is shifted, and (3) utilizing the Minkowski diference and convexity to establish the final error bound. Figure 9 illustrates the main idea.

1. Feasible Region and Parameter Transformation. We fix the error threshold $\varepsilon \geq 0 .$ . For a given multiset $X = \{ x _ { 1 } , x _ { 2 } , \dots ,$ �<sub>�</sub>} with $x _ { 1 } \le x _ { 2 } \le \cdots \le x _ { N }$ , we define the feasible region $\mathcal { F } ( \boldsymbol { \chi } )$ in the (�, �) plane as:

$$
\mathcal F ( X ) : = \{ ( w , b ) \in \mathbb R ^ { 2 } \ | \ \forall x _ { i } \in X , \ | w x _ { i } + b - r _ { i } | \leq \varepsilon \} ,\tag{18}
$$

where $r _ { i }$ is the rank of $x _ { i }$ in $\chi , \mathrm { i . e . , } r _ { i } : = i .$ Since each inequality |� $\begin{array} { r } { c _ { i } + b - r _ { i } \vert \le \varepsilon } \end{array}$ defines a convex region bounded by two parallel lines, their intersection F(X) is also a convex region.

Now, we apply the transformation $( u , v ) = ( 1 / w , b / w )$ . Since both keys �<sub>�</sub> and ranks $r _ { i }$ are non-decreasing, the optimal slope � is strictly positive. Therefore, the transformation $( u , v ) = ( 1 / w , b / w )$ is a bijection. In this transformed space, the inequality $| w x _ { i } + b - r _ { i } | \leq$ � is equivalent to:

$$
( r _ { i } - \varepsilon ) u - v \leq x _ { i } \leq ( r _ { i } + \varepsilon ) u - v .\tag{19}
$$

Let ${ \mathcal { F } } ^ { \prime } ( \chi )$ denote the feasible region in the (�, �) space. Being defined by a set of linear inequalities, ${ \mathcal { F } } ^ { \prime } ( \chi )$ is also a convex polygon.

2. Shift in the Transformed Space. The condition $\delta \in [ x _ { l - 1 } -$ $x _ { l } , x _ { r + 1 } - x _ { r } ]$ ensures that the rank $r _ { i }$ for each key $x _ { i }$ remains constant. For the shifted block $\mathcal { B } ( \delta ) = \{ x _ { i } + \delta \} _ { i = l } ^ { r } :$ , the feasibility condition becomes:

$$
( r _ { i } - \varepsilon ) u - v \leq x _ { i } + \delta \leq ( r _ { i } + \varepsilon ) u - v\tag{20}
$$

$$
\iff ( r _ { i } - \varepsilon ) u - ( v + \delta ) \leq x _ { i } \leq ( r _ { i } + \varepsilon ) u - ( v + \delta ) .\tag{21}
$$

This matches the condition for the original block $\mathcal { B } ( 0 )$ evaluated at the point $( u , v + \delta )$ . Therefore, the feasible region for the shifted block is simply the region for the original block shifted by −� along the �-axis:

$$
\mathcal { F } ^ { \prime } ( \mathcal { B } ( \delta ) ) = \mathcal { F } ^ { \prime } ( \mathcal { B } ( 0 ) ) + d ( \delta )\tag{22}
$$

where $\pmb { d } ( \delta ) : = ( 0 , - \delta )$

![](images/9930fadfb9781ae9e7ee7a70fa7886127eb89cd98692f17139752e8b255c78f2.jpg)  
Fi<sub>g</sub>ure 10: Ste<sub>p</sub> 1 of the <sub>p</sub>roof of Theorem 2: showin<sub>g</sub> $| { \mathcal { P } } ^ { * } | = \lambda .$ Re<sub>g</sub>ardless of how the three su<sub>pp</sub>ortin<sub>g</sub> <sub>p</sub>oints are arran<sub>g</sub>ed<sub>,</sub> <sub>we</sub> <sub>can</sub> <sub>a</sub>l<sub>ways</sub> <sub>a</sub>dd <sub>a</sub> <sub>su</sub>it<sub>a</sub>bl<sub>e</sub> <sub>po</sub>i<sub>son</sub> t<sub>o</sub> <sub>s</sub>t<sub>r</sub>i<sub>c</sub>tl<sub>y</sub> i<sub>ncrease</sub> th<sub>e</sub> <sub>max</sub>i<sub>mum error.</sub>

3. Overlap and Convexity. Let $\mathcal { G } : = \mathcal { F } ^ { \prime } ( \mathcal { A } )$ and $\mathcal { H } : = \mathcal { F } ^ { \prime } ( \mathcal { B } ( 0 ) )$ Both $\mathcal { G }$ and H are convex sets. The combined dataset $\mathcal { A } \not \Theta \mathcal { B } ( \delta )$ has an error of at most � if and only if $\mathcal { G } \cap \left( \mathcal { H } + d ( \delta ) \right) \neq \emptyset .$

Using the Minkowski diference $\mathcal { D } : = \mathcal { G } - \mathcal { H } = \{ \pmb { g } - \pmb { h } \ | \ \pmb { g } \in$ ${ \mathcal { G } } , h \in { \mathcal { H } } \}$ , the intersection condition ${ \mathcal { G } } \cap ( { \mathcal { H } } + d ( \delta ) ) \neq ($ ∅ is equivalent to $\pmb { d } ( \delta ) \in \mathcal { D }$ . Because $\mathcal { G }$ and H are convex, their Minkowski diference D is also convex.

Now, let $\varepsilon = \operatorname* { m a x } ( E ( \mathcal { A } \uplus \mathcal { B } ( \delta _ { - } ) ) , E ( \mathcal { A } \uplus \mathcal { B } ( \delta _ { + } ) ) )$ . By assumption, both extreme shifts achieve an error of at most $\varepsilon ,$ which implies:

$$
\begin{array} { r } { \pmb { d } ( \delta _ { - } ) = ( 0 , - \delta _ { - } ) \in \mathcal { D } \quad \mathrm { a n d } \quad \pmb { d } ( \delta _ { + } ) = ( 0 , - \delta _ { + } ) \in \mathcal { D } . } \end{array}\tag{23}
$$

Since $\delta _ { - } \leq 0 \leq \delta _ { + }$ , the shift $\delta = 0$ is a convex combination of �<sub>−</sub> and $\delta _ { + } .$ . Consequently, the zero vector $\pmb { d } ( 0 ) = ( 0 , 0 )$ is a convex combination of $\pmb { d } ( \delta _ { - } )$ and $\pmb { d } ( \delta _ { + } )$ . By the convexity of D, it follows that $\pmb { d } ( 0 ) \in \mathcal { D }$

This implies ${ \mathcal { G } } \cap { \mathcal { H } } \neq \emptyset .$ meaning there exists a valid parameter pair $( u , v )$ for the unshifted dataset. Thus, $E \left( \mathcal { A } \uplus \mathcal { B } ( 0 ) \right) \leq \varepsilon .$ which completes the proof. □

## A<sub>.</sub>2 P<sub>roo</sub>f <sub>o</sub>f Th<sub>eorem</sub> 1

Proof of Theorem 1. We prove the claim by contradiction. Assume that no optimal solution satisfies the stated property. That is, in every optimal solution, there exists at least one poison key that does not belong to any block adjacent to a legitimate key.

Among all optimal solutions, let $\mathcal { P } ^ { * }$ be one minimizing the number of isolated poison blocks, where an isolated poison block is a maximal consecutive-integer subset of poisons whose endpoints are not adjacent to any legitimate key. Formally, an isolated poison block is a set of poison keys $\{ a , a + 1 , \ldots , b \} \subseteq { \mathcal { P } } ^ { * }$ such that $a - 1 \not \in { \mathcal { K } } \cup { \mathcal { P } } ^ { * }$ and $b + 1 \not \in { \mathcal { K } } \cup { \mathcal { P } } ^ { * }$

By assumption, $\mathcal { P } ^ { * }$ contains at least one such block. Let B be one of them, and let $\mathcal { A } : = ( \mathcal { K } \cup \mathcal { P } ^ { * } )$ \ B. By Lemma 1, shifting B either left or right does not decrease the maximum-error value, while reducing the number of isolated poison blocks (see Figure 3). This contradicts the choice of $\mathcal { P } ^ { * }$ , completing the proof. □

## A<sub>.</sub>3 P<sub>roo</sub>f <sub>o</sub>f Th<sub>eorem</sub> 2

Next, we provide the proof of Theorem 2.

Proof of Theorem 2. The proof proceeds in three main steps: (1) showing that any optimal solution ${ \mathcal { P } } ^ { * }$ uses the full budget $| { \mathcal { P } } ^ { * } | =$ �, (2) showing that there exists an optimal $\mathcal { P } ^ { * }$ with $\operatorname { S u p p } ( { \mathcal { P } } ^ { * } ) \subseteq { \mathcal { K } }$ and (3) showing that there exists an optimal $\mathcal { P } ^ { * }$ supported on a single integer $k \in \mathcal { K }$ (all poisons at the same value).

1. Full Budget at Optimality. It sufices to show that for any multiset P with $| \mathcal { P } | < \lambda$ , there exists a key value � such that adding one copy of $\mathcal { P }$ to $\mathcal { P }$ strictly increases the maximum-error value $E ( \mathcal { K } \uplus \mathcal { S } )$ . Such a $\boldsymbol { p }$ can be identified by examining the supporting points (see Figure 10).

![](images/708c8788c4958cedc4a683efc3d33045a69cfa631d10ae35004678b40c66e193.jpg)  
Fi<sub>g</sub>ure 11: Ste<sub>p</sub> 2 of the <sub>p</sub>roof of Theorem 2: showin<sub>g</sub> Supp $( { \mathcal { P } } ^ { * } ) \subseteq { \mathcal { K } } .$ <sub>.</sub> B<sub>y</sub> <sub>s</sub>hifti<sub>ng</sub> <sub>a</sub> bl<sub>oc</sub>k i<sub>n</sub> $\operatorname { S u p p } ( { \mathcal { P } } ^ { * } )$ \ K left or <sub>r</sub>i<sub>g</sub>ht<sub>,</sub> <sub>we</sub> <sub>can</sub> <sub>re</sub>d<sub>uce</sub> $| \mathrm { S u p p } ( { \mathcal P ^ { * } } ) \setminus \mathcal { K } |$ <sub>w</sub>ith<sub>ou</sub>t d<sub>ecreas</sub>i<sub>ng</sub> th<sub>e</sub> objective.

Let P be a multiset with $| { \mathcal { P } } | < \lambda ,$ , let $\varepsilon = E ( \mathcal { K } \not \in \mathcal { S } )$ , and consider an optimal maximum-error regression line for K ⊎ P. It is known that there exist three points �<sub>1</sub>, �<sub>2</sub>, �<sub>3</sub> ∈ K ⊎ P whose errors attain ±� with alternating signs. If the signs are $- \varepsilon , + \varepsilon , - \varepsilon$ (left panel of Figure 10), choose $\boldsymbol { p } = \boldsymbol { x } _ { 1 }$ . If the signs are $+ \varepsilon , - \varepsilon ,$ +� (right panel), choose $\boldsymbol { p } = x _ { 3 }$ . In either case, no single line can regress the four points $( { \mathrm { i . e . , } } x _ { 1 } , x _ { 2 } , x _ { 3 }$ , and �) with maximum error $\varepsilon ,$ hence the optimum maximum error must strictly increase. Therefore, whenever $| { \mathcal { P } } | < \lambda ,$ , we can strictly increase $E ( \mathcal { K } \uplus \mathcal { S } )$ by adding one poison, which implies $| \mathcal { P } ^ { * } | = \lambda$

2. Support on Legitimate Keys. We prove the existence of an optimal solution supported only on legitimate keys by a contradiction argument similar to that of Theorem 1. Assume that every optimal poison set contains at least one value that is not a legitimate key. Among optimal solutions, choose ${ \mathcal { P } } ^ { * }$ that minimizes $| \mathrm { S u p p } ( { \mathcal P ^ { * } } ) \ \backslash \ \mathcal { K } |$ . Then there exists a contiguous block of support values in Supp $( { \mathcal { P } } ^ { * } ) \backslash { \mathcal { K } } .$ By shifting this block left or right (within the range where ranks do not change), we can apply the block-moving argument to obtain another optimal solution whose maximum error does not decrease, while $| \mathrm { S u p p } ( { \mathcal P ^ { * } } ) \backslash { \mathcal K } |$ decreases (see Figure 11). This contradicts the minimality of |Supp $( \mathcal { P } ^ { * } ) \backslash \mathcal { K } |$ Hence, there exists an optimal solution with Supp $( { \mathcal { P } } ^ { * } ) \subseteq { \mathcal { K } }$

3. Concentration on a Single Key. Finally, we show that any optimal poison set whose support is contained in K can be transformed into another optimal poison set supported on a single key value.

For $\begin{array} { r } { \pmb { d } \in \mathbb { Z } _ { \geq 0 } ^ { n } , } \end{array}$ let $Q _ { \mathcal K } ( d )$ denote the multiset that contains $d _ { i }$ copies of $k _ { i }$ for each �. Consider any � with $d _ { i } > 0$ and $d _ { j } > 0$ for some $i < j .$ . Define $\mathbf { \delta } _ { d . }$ <sub>−</sub> by moving all copies at $k _ { j }$ to $k _ { i }$ , and define $\pmb { d } _ { + }$ by moving all copies at $k _ { i }$ to $k _ { j } ,$ , that is,

$$
\begin{array} { c c c } { d = [ d _ { 1 } , \ldots , d _ { i - 1 } , ~ } & { ~ d _ { i } , \ldots , ~ } & { ~ d _ { j } , d _ { j + 1 } , \ldots , d _ { n } ] , } \end{array}\tag{24}
$$

$$
\pmb { d } _ { - } = [ d _ { 1 } , \varrho . . . , d _ { i - 1 } , \quad d _ { i } + d _ { j } , \varrho . . . , \qquad 0 , d _ { j + 1 } , \varrho . . . , d _ { n } ] ,\tag{25}
$$

$$
\begin{array} { c c c } { \pmb { d _ { + } } = [ d _ { 1 } , \ldots , d _ { i - 1 } , } & { } & { 0 , \ldots , } & { d _ { i } + d _ { j } , d _ { j + 1 } , \ldots , d _ { n } ] . } \end{array}\tag{26}
$$

We claim that for any $\varepsilon \geq 0 ,$

$$
E ( { \mathcal { K } } \uplus Q _ { { \mathcal { K } } } ( d _ { - } ) ) \leq \varepsilon \ \wedge \ E ( { \mathcal { K } } \uplus Q _ { { \mathcal { K } } } ( d _ { + } ) ) \leq \varepsilon\tag{27}
$$

$$
\implies E ( \mathcal { K } \uplus Q _ { \mathcal { K } } ( d ) ) \leq \varepsilon .\tag{28}
$$

![](images/2051311e0057eb6f09138b92dfd0eef8e6c3d7e1030650b04f90016c32a0bb7a.jpg)  
Fi<sub>gure</sub> 12<sub>:</sub> St<sub>ep</sub> 3 <sub>o</sub>f th<sub>e proo</sub>f <sub>o</sub>f Th<sub>eorem</sub> 2<sub>.</sub> Th<sub>e</sub> fi<sub>gure</sub> ill<sub>us-</sub> t<sub>ra</sub>t<sub>es</sub> th<sub>e case w</sub>h<sub>ere</sub> $d _ { i } = 2$ <sub>an</sub>d $d _ { j } = 3 .$ Pro<sub>p</sub>ert<sub>y</sub> 1 shows that th<sub>e error o</sub>f $f$ <sub>w</sub>ith <sub>respec</sub>t t<sub>o</sub> th<sub>e</sub> bl<sub>ac</sub>k<sub>, ye</sub>ll<sub>ow, an</sub>d <sub>green</sub> <sub>p</sub>oints is at most <sub>�</sub>. Pro<sub>p</sub>ert<sub>y</sub> 2 shows that the error for the orange <sub>p</sub>oints is at most �.

To see this, let $f _ { - } ( x )$ and $f _ { + } ( x )$ be regression lines achieving maximum error at most � for K ⊎ $Q _ { \mathcal K } ( d _ { - } )$ and K ⊎ $Q _ { \mathcal { K } } ( d _ { + } )$ , respectively. Define the convex combination

$$
f ( x ) : = \frac { d _ { i } \cdot f _ { + } ( x ) + d _ { j } \cdot f _ { - } ( x ) } { d _ { i } + d _ { j } } .\tag{29}
$$

Then, we can show that $f ( x )$ attains maximum error at most � on ${ \mathcal { K } } \uplus Q _ { { \mathcal { K } } } ( d )$ , which establishes Equation (28). To see this, it is suficient to show the following two properties (each of which can be proved using elementary methods):

• Property 1: If a point $( x , y )$ has an error of at most � under both $f _ { - }$ and $f _ { + }$ , then its error under � is also at most �.

• Property 2: If the points $( x , y )$ and $( x , y + d _ { i } + d _ { j } )$ have errors of at most � under $f _ { - }$ and $f _ { + }$ , respectively, then the error of the point $( x , y + d _ { i } )$ under � is at most �.

We provide Figure 12 to aid understanding. K ⊎ $Q _ { \mathcal { K } } ( d _ { - } )$ corresponds to the black, purple, and green point sets; K ⊎ $Q _ { \mathcal { K } } ( d _ { + } )$ corresponds to the black, yellow, and blue point sets; and K ⊎ $Q _ { \mathcal K } ( d )$ corresponds to the black, yellow, orange, and green point sets. By Property 1, it follows that the error of the black, yellow, and green point sets under � is at most �. By Property 2, it follows that the error of the orange point set under � is at most �. Thus, via Properties 1 and $^ { 2 , }$ it is shown that the error of K ⊎ $Q _ { \mathcal K } ( d )$ under $f$ is at most �.

Now, setting $\varepsilon = \operatorname* { m a x } \bigl ( E ( \mathcal { K } \uplus Q _ { \mathcal { K } } ( d _ { - } ) ) , E ( \mathcal { K } \uplus Q _ { \mathcal { K } } ( d _ { + } ) ) \bigr )$ in Equation (28) yields immediately

$$
E ( { \mathcal { K } } \uplus Q _ { { \mathcal { K } } } ( d ) ) \leq \operatorname* { m a x } \left( E ( { \mathcal { K } } \uplus Q _ { { \mathcal { K } } } ( d _ { - } ) ) , E ( { \mathcal { K } } \uplus Q _ { { \mathcal { K } } } ( d _ { + } ) ) \right) .\tag{30}
$$

Therefore, whenever poisons are distributed across multiple key values, we can merge their mass to one endpoint without decreasing the maximum error. By repeatedly applying this merging operation, we obtain an optimal solution whose poisons are supported on a single integer $k \in \mathcal K$ □

## A<sub>.</sub>4 P<sub>roo</sub>f <sub>o</sub>f Th<sub>eorem</sub> 5

Proof of Theorem 5. We first prove the case where one poison is added to $\mathcal { K } ;$ that is, we show that adding a single point � to K satisfies $m _ { \mathrm { o p t } } ( { \mathcal { K } } \cup \{ p \} , \varepsilon ) \leq m _ { \mathrm { o p t } } ( { \mathcal { K } } , \varepsilon ) + 1$ . Now, let $s =$ ${ \bigl ( } ( s _ { 1 } , e _ { 1 } , f _ { 1 } ) , ( s _ { 2 } , e _ { 2 } , f _ { 2 } ) , \ldots , ( s _ { m } , e _ { m } , f _ { m } ) { \bigr ) }$ be a PLA for K achieving $m : = m _ { \mathrm { o p t } } ( \mathcal { K } , \varepsilon )$ . The proof proceeds by considering the following two cases based on the position of�: Case 1: � lies between two ad jacent segments in S, and Case 2: � lies inside an existing segment in S.

![](images/79cf5a0dd92b412d6bb24acf52d65c3382e39a2282a094dac8b1986a75531733.jpg)  
Figure 13: Proof of Theorem 5. (1) Case 1 (left): when $\mathcal { P }$ li<sub>es</sub> between two adjacent segments, we add a new segment covering only �. (2) Case 2 (right): when � lies inside an existing se<sub>g</sub>ment<sub>,</sub> we s<sub>p</sub>lit the se<sub>g</sub>ment containin<sub>g</sub> <sub>�</sub> into two.

Case 1: More precisely, we consider the case where there exists some $j \in [ m - 1 ]$ such that $x _ { e _ { j } } < x _ { j } < x _ { s _ { j + 1 } }$ . In this case, we can construct a PLA for K ∪ {�} with � + 1 segments as follows (refer to the left side of Figure 13): We retain the first � segments (those preceding �) as they are, increase the function values of the (� + 1)-th through �-th segments by one, and insert a new segment dedicated to covering �. It is clear that the approximation error remains at most � for all keys in K and the poisoned point �.

Case 2: More precisely, we consider the case where there exists some � ∈ [�] such that $x _ { s _ { j } } < x _ { j } < x _ { e _ { j } }$ . In this case, a PLA with � + 1 segments can be constructed as follows (refer to the right side of Figure 13): We keep the first � − 1 segments unchanged and increase the function values of the $( j + 1 )$ -th through �-th segments by one. We then split the �-th segment into two parts at $x _ { p } ,$ , and increase the function value of the second part by one. Under the assumption that $\varepsilon \ \geq \ 1 / 2 ,$ , the point � can always be covered by at least one of these two segments within an error of �. Thus, the approximation error for all keys and the poisoned point is guaranteed to be at most �.

From Case 1 and Case 2, it follows that $m _ { \mathrm { o p t } } ( \mathcal { K } \cup \{ p \} , \varepsilon ) \ \leq$ $m _ { \mathrm { o p t } } ( \mathcal { K } , \varepsilon ) + 1$ . We now complete the proof for the general case using induction on $| { \mathcal { P } } |$ . Assume that $m _ { \mathrm { o p t } } ( { \mathcal { K } } \cup { \mathcal { P } } , \varepsilon ) \leq m _ { \mathrm { o p t } } ( { \mathcal { K } } , \varepsilon ) + k$ holds for $| { \mathcal { P } } | = k .$ For $| \mathcal { P } | = k + 1$ , let $\mathcal { \dot { P } } = \mathcal { P } ^ { \prime } \cup \{ p \}$ where $| { \mathcal { P } } ^ { \prime } | = k$ Then, we have:

$$
m _ { \mathrm { o p t } } ( { \mathcal K } \cup { \mathcal P } , \varepsilon ) = m _ { \mathrm { o p t } } ( ( { \mathcal K } \cup { \mathcal P } ^ { \prime } ) \cup \{ p \} , \varepsilon )\tag{31}
$$

$$
\leq m _ { \mathrm { o p t } } ( \mathcal { K } \cup \mathcal { P } ^ { \prime } , \varepsilon ) + 1\tag{32}
$$

$$
\leq m _ { \mathrm { o p t } } ( \mathcal { K } , \varepsilon ) + k + 1 ,\tag{33}
$$

where the first inequality follows from the result for a single-point insertion, and the second inequality follows from the inductive hypothesis. This concludes the proof for any $\mathcal { P } .$ □

## A<sub>.</sub>5 P<sub>roo</sub>f <sub>o</sub>f Th<sub>eorem</sub> 6

Proof of Theorem 6. The proof is constructive. Let $b : = 4 \varepsilon + 3$ and $B : = ( 2 \varepsilon + 1 ) ( 2 \varepsilon + 2 ) + 2 .$ . We define a block as a set of � consecutive integers. We construct K by placing these blocks with a gap of 2� between each pair of adjacent blocks. The poisoning set $\mathcal { P }$ is then placed at the center of these gaps (that is, $| \mathcal { P } | = \lambda )$ Precisely, we have

![](images/129d21ba7fb6db6e22d16149c096124b3441e7d94b0533bcf9b8dd40d1646f69.jpg)  
Figure 14: An example of the tight upper bound (Theorem 6), <sub>w</sub>h<sub>ere</sub> $\lambda = 2$ <sub>an</sub>d $\varepsilon = 0 .$ . Before poisoning (left), $m _ { \mathrm { o p t } } ( \mathcal { K } , \varepsilon ) = 3$ and after poisoning (right), $m _ { \mathrm { o p t } } ( \mathcal { K } \cup \mathcal { P } , \varepsilon ) = 5 .$

Algorithm 2 DI-Consecutive   
Input: Legitimate key set K, poison budget �   
Output: Poison set $\mathcal { P } ^ { * }$   
1: Precompute slope intervals I and build sparse tables   
2: $C ^ { \ast } \gets C _ { \varepsilon } ( \mathcal { K } , \emptyset ) , \mathcal { P } ^ { \ast } \gets \emptyset$   
3: for each candidate consecutive poison set P with $| { \mathcal { P } } | = \lambda$ d<sub>o</sub>   
4: $C _ { \mathcal { P } } \gets C _ { \varepsilon } ( \mathcal { K } , \mathcal { P } )$ ⊲ time O (|I| log �) via sparse tables   
5: $\mathbf { i f } \ C _ { \mathcal { P } } < C ^ { * }$ th<sub>en</sub>   
6: $C ^ { * } \gets C _ { \mathcal { P } } , \mathcal { P } ^ { * } \gets \mathcal { P }$   
7: return ${ \mathcal { P } } ^ { * }$

$$
\mathcal { K } = \{ 2 i B + j \ | \ i \in \ \{ 0 , 1 , . . . , \lambda \} , j \in \ \{ 0 , 1 , . . . , b - 1 \} \} ,\tag{34}
$$

$$
\mathcal { P } = \{ ( 2 i + 1 ) B + ( 2 \varepsilon + 1 ) \mid i \in \{ 0 , 1 , \ldots , \lambda - 1 \} \} .\tag{35}
$$

In this configuration, we have $m _ { \mathrm { o p t } } ( \mathcal { K } , \varepsilon ) = \lambda + 1$ . This is because an optimal PLA covers each block with exactly one segment, and no single segment can cover more than one block. This property is guaranteed by the fact that � and � are chosen to be suficiently large.

Furthermore, it can be shown that $m _ { \mathrm { o p t } } ( \mathcal { K } \cup \mathcal { P } , \varepsilon ) = 2 \lambda + 1$ . This can be verified by greedily determining the segments from left to right:

• The 1-st segment covers the first block of K.

• For $i \in \left\{ 1 , 2 , \ldots , \lambda \right\}$ , the 2�-th segment covers the �-th poisoned point in P and the first 2� + 1 elements of the (� + 1)-th block.

• Fo $\mathfrak { r } i \in \{ 1 , 2 , \dotsc , \lambda \}$ , the (2�+1)-th segment covers the remaining 2� + 2 elements of the (� + 1)-th block.

Thus, in this specific case, the equality in Equation (17) is satisfied, completing the proof. □

## B DETAILS ON DI-CONSECUTIVE

In this section, we provide the technical details omitted from the main text regarding DI-Consecutive (Discrete-Intercept Consecutive method) proposed in Section 4.3. The overall procedure is summarized in Algorithm 2.

Main Idea. The core objective of DI-Consecutive is to optimize the evaluation of each candidate of consecutive poisons, $\mathrm { i . e . , }$ to compute the number of covered legitimate keys. We reduce the computational complexity from the conventional O (�), where � is the number of covered legitimate keys without poisoning, to $O ( | \boldsymbol { \mathcal { I } } | \log c )$ , where $| { \cal T } |$ is the number of candidate intercepts. In typical settings where $c \geq 1 0 ^ { 4 }$ and $| \mathcal { I } | \approx 4 1$ , this improvement is substantial.

![](images/cda5a289373b2288a7fda904e1aacad54bfa81c805baf5e117f300aab0ba6c5e.jpg)  
Fi<sub>g</sub>ure 15: An exam<sub>p</sub>le of DI-Consecutive evaluatin<sub>g</sub> a candidate <sub>p</sub>oison se<sub>q</sub>uence (starting from $k _ { 3 } + 1$ <sub>w</sub>ith $\lambda = 2 )$ . For a s<sub>p</sub>ecific interce<sub>p</sub>t $b ,$ s<sup>l</sup>ope ranges <sup>f</sup>or pre-po<sup>i</sup>son $( k _ { 2 } , k _ { 3 } )$ <sub>an</sub>d p<sup>ost-</sup>p<sup>oison</sup> $( k _ { 5 } , k _ { 6 } )$ ke<sub>y</sub>s are retrieved in $O ( 1 )$ <sub>v</sub>i<sub>a sparse</sub> t<sub>a</sub>bl<sub>es.</sub> The slo<sub>p</sub>e ran<sub>g</sub>e for the <sub>p</sub>oisons is also com<sub>p</sub>uted in $O ( 1 )$ <sup>b</sup><sub>y</sub> findin<sub>g</sub> the ran<sub>g</sub>e coverin<sub>g</sub> $k _ { 3 } + 1$ <sub>an</sub>d th<sub>e</sub> l<sub>arges</sub>t <sub>po</sub>i<sub>son.</sub>

Figure 15 illustrates the intuition behind our approach. The figure shows the evaluation of a candidate consecutive poisons starting from $k _ { 3 } + 1$ . DI-Consecutive independently computes the range of slopes that can cover: (1) legitimate keys preceding the poisons, (2) legitimate keys following the poisons, and (3) the consecutive integer sequence containing the poisons. The candidate poisons and keys are coverable if and only if these three slope ranges have a non-empty intersection.

## B.1 Precom<sub>p</sub>utation

For each candidate intercept $b \in { \mathcal { I } }$ , we perform a precomputation step with a time complexity of O(� log �). Consequently, the total precomputation complexity is O(|I|� log �). Below, we describe the precomputation step for a particular �.

First, we compute $\underline { { u } } _ { i } , \overline { { u } } _ { i } , \underline { { v } } _ { i } , \overline { { v } } _ { i }$ for $i \in \{ 2 , 3 , . . . , c \}$ as follows:

$$
\underline { { u } } _ { i } : = \frac { ( i - \varepsilon ) - ( 1 + b ) } { k _ { i } - k _ { 1 } } , \quad \overline { { u } } _ { i } : = \frac { ( i + \varepsilon ) - ( 1 + b ) } { k _ { i } - k _ { 1 } } ,\tag{36}
$$

$$
\underline { { v } } _ { i } : = \frac { ( i + \lambda - \varepsilon ) - ( 1 + b ) } { k _ { i } - k _ { 1 } } , \quad \overline { { v } } _ { i } : = \frac { ( i + \lambda + \varepsilon ) - ( 1 + b ) } { k _ { i } - k _ { 1 } } .\tag{37}
$$

The interval $[ \underline { { u } } _ { i } , \overline { { u } } _ { i } ]$ represents the slope range required to cover $k _ { i }$ when it precedes the poisons. For example, in Figure 15, the slope of the line connecting $( k _ { 1 } , 1 + b )$ and $( k _ { 3 } , 3 + \varepsilon )$ corresponds to ${ \overline { { u } } } _ { 3 } .$ . Similarly, $[ \underline { { \upsilon } } _ { i } , \overline { { \upsilon } } _ { i } ]$ represents the slope range required to cover the consecutive integer sequence containing the poisons when $k _ { i }$ follows the poisons. For instance, in Figure 15, the slope of the line connecting $( k _ { 1 } , 1 + b )$ and $( k _ { 6 } , 6 + \lambda + \varepsilon )$ corresponds to ${ \overline { { v } } } _ { 6 }$ . These values can be computed in $O ( c )$ time.

Furthermore, we construct sparse tables over these four arrays to support O (1) Range Maximum Queries for � and �, and $O ( 1 )$ Range Minimum Queries for � and �. This construction requires O (� log �) time.

## B<sub>.</sub>2 E<sub>va</sub>l<sub>ua</sub>ti<sub>o</sub>n

The evaluation phase computes, for a given candidate consecutive poison set $\mathcal { P } _ { : }$ , the maximum number of legitimate keys that can be covered.

To achieve a time complexity of $O ( | \boldsymbol { \mathcal { I } } | \log c )$ for each candidate consecutive poison set, we design the evaluation so that, for each intercept $b \in { \mathcal { I } }$ , the number of coverable legitimate keys can be computed in $O ( \log c )$ time. This $O ( \log c )$ computation is realized via binary search combined with an $O ( 1 )$ feasibility check, which determines whether, for the given poison set, intercept, and keys, there exists a slope under which they are coverable.

Below, we first describe the Maximal Coverage Search, which computes in O (log �) time the number of legitimate keys coverable for a fixed poison set and intercept. We then explain the Coverability Verification, which determines in $O ( 1 )$ time whether the given poison set, intercept, and keys are feasible.

Maximal Coverage Search. First, for a given candidate consecutive poison set ${ \mathcal { P } } ,$ we find the smallest legitimate key greater than the largest poison in the set, and denote its index by $j .$ For instance, in Figure $1 5 , j = 5 . \mathrm { A }$ straightforward method would inspect the integers not occupied by legitimate keys in increasing order. In the worst case, this takes $O ( c + \lambda )$ time. To reduce this cost, we precompute the number of unused integers between each pair of consecutive legitimate keys, as well as their prefix sums. Using this information, we can perform a binary search to determine where the largest poison falls relative to the gaps between legitimate keys, thereby identifying the first legitimate key after the poisons in O (log �) time.

Next, for each intercept $b \in { \mathcal { I } } .$ , we perform a binary search to find the maximum $r \in \{ j , \ldots , c \}$ such that $\{ k _ { 1 } , . . . , k _ { r } \} \cup \mathcal { P }$ are coverable. $\mathrm { A t }$ each step, we invoke the Coverability Verification (described below) to determine whether $\{ k _ { 1 } , \ldots , k _ { r } \} \cup \mathcal { P }$ are coverable. This yields the maximum feasible � for the given intercept in $O ( \log c )$ time.

After evaluating all $b \in { \mathcal { I } }$ , we select the intercept that yields the largest �, which corresponds to the maximum number of covered legitimate keys for the given poison set.

Coverability Verification. The binary search requires verifying whether the set $\{ k _ { 1 } , \ldots , k _ { r } \} \cup \mathcal { P }$ is coverable for a given intercept �. We determine the existence of a valid slope in $O ( 1 )$ time. Such a slope exists if and only if the following three ranges have a nonempty intersection: (1) the pre-poison range, (2) the post-poison range, and (3) the poison region range.

(1) The pre-poison range is the slope range covering legitimate keys preceding the poisons $( \mathrm { i . e . , }$ , from $k _ { 2 }$ to $k _ { i } )$ . This is computed as

$$
\left[ \operatorname* { m a x } _ { 2 \leq l \leq i } { \underline { { u } } _ { l } } , \operatorname* { m i n } _ { 2 \leq l \leq i } { \overline { { u } } _ { l } } \right] .\tag{38}
$$

We can obtain this in $O ( 1 )$ time using a sparse table constructed during the precomputation step.

(2) The post-poison range is the slope range covering legitimate keys following the poisons (i.e., from $k _ { j }$ to $k _ { r } ) .$ . This is computed as

$$
\left[ \operatorname* { m a x } _ { j \le l \le r } { \underline { { v } } _ { l } } , \operatorname* { m i n } _ { j \le l \le r } \overline { { v } } _ { l } \right] .\tag{39}
$$

We can obtain this in $O ( 1 )$ time using the precomputed sparse table.

![](images/e5248a5c06ffae43d8cab094b5114771ad8360124e5ca91e8207f9bf1ab37263.jpg)  
Fi<sub>g</sub>ure 16: Overview of the instance-de<sub>p</sub>endent u<sub>pp</sub>er-bound al<sub>g</sub>orithm: <sub>p</sub>artition K into blocks with fixed slo<sub>p</sub>es<sub>,</sub> com-<sub>pu</sub>t<sub>e</sub> <sub>per-</sub>bl<sub>oc</sub>k <sub>upper</sub> b<sub>oun</sub>d<sub>s,</sub> <sub>an</sub>d <sub>a</sub>ll<sub>oca</sub>t<sub>e</sub> th<sub>e</sub> <sub>po</sub>i<sub>son</sub> b<sub>u</sub>d<sub>ge</sub>t <sub>across</sub> bl<sub>oc</sub>k<sub>s</sub> t<sub>o max</sub>i<sub>m</sub>i<sub>ze</sub> th<sub>e</sub>i<sub>r sum.</sub>

(3) The poison region range is the slope range covering the consecutive integers including the poisons $( { \mathrm { i . e . } }$ , the consecutive integers from $k _ { i } + 1$ to the largest poison). This slope range is obtained by intersecting the slope range that covers $k _ { i } + 1$ with the slope range that covers the largest poison. This is because, by linearity, if the error at both endpoints is at most �, then the error at every point between them is also guaranteed to be at most �.

## C DETAILS ON THE INSTANCE-DEPENDENT UPPER BOUND ALGORITHM

In this section, we provide the details omitted in the main text regarding the instance-dependent upper-bound algorithm proposed in Section 5.4. Figure 16 illustrates the pipeline: partitioning into blocks, upper-bounding the number of segments in each block, and then allocating the total poison budget across blocks.

## C<sub>.</sub>1 Di<sub>v</sub>idin<sub>g</sub> int<sub>o</sub> Bl<sub>oc</sub>k<sub>s</sub>

We partition K into blocks using an ��-PLA with $\alpha \geq 1$ and fix the slope within each block to that of the corresponding PLA segment. This restricts the PLA construction by requiring each block to use a predetermined slope and by treating the blocks independently. We also strengthen the attacker by allowing duplicate poisons. Since these modifications weaken the index constructor and strengthen the attacker, the maximum number of segments in the resulting relaxed problem upper-bounds that in the original problem.

In our implementation, we evaluate the upper bound for $\alpha \in$ {1.0, 1.2, 1.4, 1.6, 1.8, 2.0} and take the minimum. Since each value of � yields a valid upper bound, taking their minimum preserves the upper-bound guarantee.

## C.2 Per-Block U<sub>pp</sub>er Bound

We next derive an upper bound on the number of segments in a single block. Let $\{ k _ { 1 } , k _ { 2 } , \ldots , k _ { n } \}$ be the keys in the block, and let � be its fixed slope.

First, we prove the following lemma concerning the minimum number of poisons required to separate two keys.

Lemma 2. Let $\{ k _ { 1 } , k _ { 2 } , \ldots , k _ { n } \}$ b<sub>e</sub> th<sub>e</sub> k<sub>eys</sub> i<sub>n</sub> th<sub>e</sub> bl<sub>oc</sub>k<sub>,</sub> <sub>an</sub>d l<sub>e</sub>t � be its fixed slope. Define $A _ { l } : = l - w k _ { l } f o r l \in [ n ]$ <sub>.</sub> Th<sub>en,</sub> th<sub>e</sub> <sub>a</sub>tt<sub>ac</sub>k<sub>er</sub> <sub>requ</sub>i<sub>res</sub> <sub>a</sub>t l<sub>eas</sub>t $\lceil c _ { i , j } \rceil$ <sub>p</sub>oisons to ma<sup>k</sup>e t<sup>h</sup>e se<sub>g</sub>ment

![](images/0ec05f64fbaa02b67e8e994183abf7c1f5f26f16301ae99eaf6718619c727054.jpg)  
Fi<sub>g</sub>ure 17: Exam<sub>p</sub>le of a <sub>p</sub>ath in the DAG for u<sub>pp</sub>er-boundin<sub>g</sub> th<sub>e</sub> <sub>num</sub>b<sub>er</sub> <sub>o</sub>f <sub>segmen</sub>t<sub>s</sub> i<sub>n</sub> <sub>a</sub> bl<sub>oc</sub>k<sub>.</sub> A <sub>pa</sub>th <sub>represen</sub>t<sub>s</sub> <sub>a</sub> <sub>seg-</sub> mentation<sub>;</sub> its len<sub>g</sub>th <sub>g</sub>ives the number of se<sub>g</sub>ments<sub>,</sub> and its t<sub>o</sub>t<sub>a</sub>l <sub>we</sub>i<sub>g</sub>ht l<sub>ower-</sub>b<sub>oun</sub>d<sub>s</sub> th<sub>e requ</sub>i<sub>re</sub>d <sub>num</sub>b<sub>er o</sub>f <sub>po</sub>i<sub>sons.</sub>

<sup>startin</sup>g <sup>at</sup> $k _ { i }$ end before $k _ { j } ,$ <sub>,</sub> <sub>w</sub>h<sub>ere</sub>

$$
c _ { i , j } : = \operatorname* { m a x } \left( 0 , 2 \varepsilon - \left( \operatorname* { m a x } _ { i \leq l \leq j } A _ { l } - \operatorname* { m i n } _ { i \leq l \leq j } A _ { l } \right) \right) .\tag{40}
$$

Proof of Lemma 2. Under a fixed slope �, let � be the optimal intercept for the set of keys $\{ k _ { i } , . . . , k _ { j } \}$ . The prediction error of the position for each key $k _ { l } ~ ( i ~ \leq ~ l ~ \leq ~ j )$ is expressed as $| ( w k _ { l } + b ) - l | = | b - ( l - w k _ { l } ) | = | b - A _ { l } |$ . Therefore, the optimal intercept that minimizes the maximum error in this segment is $\begin{array} { r } { b ^ { * } = \frac { 1 } { 2 } \big ( \operatorname* { m a x } _ { i \leq l \leq j } A _ { l } + \operatorname* { m i n } _ { i \leq l \leq j } A _ { l } \big ) } \end{array}$ , and the initial maximum error $E _ { \mathrm { i n i t } }$ before poisoning is given by:

$$
E _ { \mathrm { i n i t } } = { \frac { 1 } { 2 } } \left( \operatorname* { m a x } _ { i \leq l \leq j } A _ { l } - \operatorname* { m i n } _ { i \leq l \leq j } A _ { l } \right)\tag{41}
$$

Moreover, we can show that each duplicate poison can increase the maximum error by at most $1 / 2 .$ This holds because the linear regression can limit the increase in the maximum error to $1 / 2$ by increasing the intercept � by 1/2 for each added poison.

Therefore, at least

$$
\left\lceil \operatorname* { m a x } \left( 0 , \frac { \varepsilon - E _ { \mathrm { i n i t } } } { 1 / 2 } \right) \right\rceil = \lceil c _ { i , j } \rceil\tag{42}
$$

poisons are needed to make the error exceed �.

We now formulate the per-block problem as a path problem on a DAG. The DAG contains nodes $1 , \ldots , n + 1$ , where node $n + 1$ is a sentinel representing the end of the block. For each $1 \leq i < j \leq n ,$ we add an edge from � to � with weight $c _ { i , j }$ . We also add an edge of weight zero from each node � ∈ [�] to the sentinel node $n + 1 ;$ that is, define $c _ { i , n + 1 } = 0$ for all $i \in [ n ]$

As illustrated in Figure 17, each path from node 1 to node $n + 1$ represents a segmentation of the block. The starting node of each edge corresponds to the starting position of a segment, so the number ofedges in the path equals the number ofsegments. By Lemma 2, the weight $c _ { i , j }$ lower-bounds the number of poisons required to force the segment starting at $k _ { i }$ to end before $k _ { j }$ . The final edge has weight zero because the last segment naturally terminates at the end of the block. Consequently, the total weight of a path lowerbounds the number of poisons required to realize the corresponding segmentation.

Next, to reduce the computational complexity of finding shortest paths in the $\mathrm { { D A G , } }$ we prove the following lemma regarding the Monge property of $c _ { i , j }$

Lemma 3. The cost function $c _ { i , j }$ satisfies the Monge property, i.e., for any $1 \leq i < i ^ { \prime } < j < j ^ { \prime } \leq n$

$$
c _ { i , j } + c _ { i ^ { \prime } , j ^ { \prime } } \leq c _ { i , j ^ { \prime } } + c _ { i ^ { \prime } , j } .\tag{43}
$$

Proof of Lemma 3. Let $\begin{array} { r } { D ( i , j ) : = \operatorname* { m a x } _ { i \leq l \leq j } A _ { l } - \operatorname* { m i n } _ { i \leq l \leq j } A _ { l } . } \end{array}$ For any $i < i ^ { \prime } < j < j ^ { \prime }$ , consider the intervals $I _ { 1 } = \left[ i , j \right]$ and $I _ { 2 } = $ $[ i ^ { \prime } , j ^ { \prime } ]$ . Their union is $I _ { 1 } \cup I _ { 2 } = [ i , j ^ { \prime } ]$ and their intersection is $I _ { 1 } \cap I _ { 2 } =$ [�<sup>′</sup>, �]. By the basic properties of the range over intervals, the sum of the ranges of the union and intersection is upper-bounded by the sum of the ranges of the original intervals:

$$
D ( i , j ^ { \prime } ) + D ( i ^ { \prime } , j ) \leq D ( i , j ) + D ( i ^ { \prime } , j ^ { \prime } ) .\tag{44}
$$

Furthermore, the range is monotonically non-decreasing with respect to interval inclusion, meaning $D ( i , j ^ { \prime } ) \geq \operatorname* { m a x } ( D ( i , j ) , D ( i ^ { \prime } , j ^ { \prime } ) )$ . Since $c _ { p , q } = \operatorname* { m a x } ( 0 , 2 \varepsilon - D ( p , q ) )$ , these two properties together imply $c _ { i , j } + c _ { i ^ { \prime } , j ^ { \prime } } \leq c _ { i , j ^ { \prime } } + c _ { i ^ { \prime } , j }$ , concluding the proof. □

We finally derive an upper bound on the number of segments under a poison budget. For a path � from node 1 to node $n + 1$ , let |�| denote its number of edges and let

$$
C ( P ) : = \sum _ { ( i , j ) \in P } c _ { i , j }\tag{45}
$$

denote its total weight. For any parameter $\nu > 0 ,$ define

$$
D ( \nu ) : = \operatorname* { m i n } _ { P } \left( C ( P ) - \nu | P | \right)\tag{46}
$$

Equivalently, $D ( \nu )$ is the shortest-path distance from node 1 to node $n + 1$ after subtracting � from every edge weight in the DAG.

For any path � realizable with at most � poisons, we have $C ( P ) \leq$ �. By the definition of �(�),

$$
D ( \nu ) \leq C ( P ) - \nu | P | \leq \lambda - \nu | P | .\tag{47}
$$

Therefore, for any path � realizable with at most � poisons,

$$
| P | \leq \frac { \lambda - D ( \nu ) } { \nu } .\tag{48}
$$

Thus, Equation (48) gives an upper bound on the number of segments achievable within the block.

Subtracting the same constant � from every edge preserves the Monge property. Hence, �(�) can be computed in $O ( n )$ time using the LARSCH algorithm [2, 25]. In our experiments, we evaluate the bound for $\nu \in N : = \{ 1 , 2 , 3 , 4 , 5 , 1 0 , 2 0 , 4 0 , 8 0 , 1 6 0 \}$ and take the minimum. For a block � and an allocated budget �, we denote the resulting bound by $\mathrm { U B } _ { b } ( \lambda )$ , defined as follows:

$$
\mathrm { U B } _ { b } ( \lambda ) : = \operatorname* { m i n } _ { \nu \in N } \frac { \lambda - D _ { b } ( \nu ) } { \nu } ,\tag{49}
$$

where $D _ { b } ( \nu )$ is the shortest-path distance from node 1 to node $n + 1$ after subtracting � from every edge weight in the DAG for block �.

## C<sub>.</sub>3 All<sub>oca</sub>ti<sub>o</sub>n A<sub>c</sub>r<sub>oss</sub> Bl<sub>oc</sub>k<sub>s</sub>

Suppose that K is divided into � blocks. Let $\lambda _ { b }$ be the number of poisons allocated to block �. Since the total poison budget is �, the allocations satisfy

$$
\sum _ { b = 1 } ^ { B } \lambda _ { b } \leq \lambda .\tag{50}
$$

Using the per-block bound in Equation (49), the global upper bound is obtained by solving

$$
\operatorname* { m a x } _ { \lambda _ { 1 } , \ldots , \lambda _ { B } \in \mathbb { Z } _ { \geq 0 } } \sum _ { b = 1 } ^ { B } \operatorname { U B } _ { b } ( \lambda _ { b } ) .\tag{51}
$$

Each $\mathrm { U B } _ { b } ( \lambda _ { b } )$ is the pointwise minimum of afine functions of $\lambda _ { b }$ and is therefore concave. Consequently, its marginal gain

$$
\mathrm { U B } _ { b } \big ( \lambda _ { b } + 1 \big ) - \mathrm { U B } _ { b } \big ( \lambda _ { b } \big )\tag{52}
$$

is non-increasing in $\lambda _ { b }$ . The allocation problem in Equation (51) can therefore be solved optimally by repeatedly assigning one poison to the block with the largest current marginal gain [14]. The sum of the resulting per-block bounds gives the instance-dependent upper bound for the selected block partition.

## D ADDITIONAL EXPERIMENTS

## D<sub>.</sub>1 Additi<sub>o</sub>n<sub>a</sub>l R<sub>esu</sub>lt<sub>s o</sub>n PLA S<sub>eg</sub>m<sub>e</sub>nt M<sub>ax</sub>i<sub>m</sub>i<sub>za</sub>ti<sub>on</sub>

Here, we present a comprehensive set ofresults on the post-poisoning $m _ { \mathrm { o p t } }$ and the upper bounds we derive, which were omitted from the main text. As attack methods, in addition to PGM-attack, we also evaluate Random and Random-Adjacent, the same baselines used in Section 6.2. As upper bounds, we evaluate the instanceagnostic upper bound (Agnostic UB; Section 5.3) and the instancedependent upper bound (Instance UB; Section 5.4). We use � ∈ {16, 32, 64, 128}. The numbers in parentheses indicate the factor relative to $m _ { \mathrm { o p t } }$ for the legitimate keys.

D.1.1 Complete Results on Poisoning and Upper Bounds. Table 5 shows the results. Across all rows of Table 5, PGM-attack attains a higher post-poisoning $m _ { \mathrm { o p t } }$ than both Random and Random-Adjacent. On the YCSB dataset, PGM-attack increases $m _ { \mathrm { o p t } }$ by 15.5× with 1% poisoning and by up to 120× with 10% poisoning. This pronounced efect stems from the fact that YCSB is nearly uniform, allowing a PLA with very few segments to be constructed before poisoning. By adding only a small number of poison keys, our attack drastically increases the required number of segments. Random poisoning also has a relatively large impact on YCSB: at a 10% poisoning rate, Random and Random-Adjacent increase $m _ { \mathrm { o p t } }$ by 1.99× and 2.42×, respectively. Nevertheless, the increase caused by PGM-attack is substantially larger. The impact of our attack is also significant on datasets other than YCSB. At a 10% poisoning rate, it increases $m _ { \mathrm { o p t } }$ by factors ranging from 1.46× to 24.5×, substantially exceeding the increases caused by the random baselines, which range from 1.00× to 1.30×.

Regarding the upper bounds, Instance UB is consistently tighter than Agnostic UB and is at most 1.92× the attained $m _ { \mathrm { o p t } }$ . This provides an instance-wise certificate that PGM-attack achieves at least 52% of the optimum on all evaluated instances. It also shows that incorporating instance-specific information yields a tighter bound than the closed-form Agnostic UB. These results support the usefulness of our upper bounds for understanding the maximum achievable impact ofpoisoning attacks and, more broadly, for characterizing how much $m _ { \mathrm { o p t } }$ can increase under arbitrary insertions.

T<sub>a</sub>bl<sub>e</sub> 5<sub>:</sub> R<sub>esu</sub>lt<sub>s</sub> f<sub>or</sub> $m _ { \mathrm { o p t } }$ <sub>a</sub>ft<sub>er</sub> <sub>po</sub>i<sub>son</sub>i<sub>ng.</sub> A<sub>cross</sub> <sub>a</sub>ll d<sub>a</sub>t<sub>ase</sub>t<sub>s</sub> <sub>an</sub>d <sub>se</sub>tti<sub>ngs</sub> <sub>o</sub>f <sub>�</sub> <sub>an</sub>d $\lambda ,$ our PGM-attack causes a substantiall<sub>y</sub> lar<sub>g</sub>er increase than randoml<sub>y</sub> insertin<sub>g</sub> <sub>p</sub>oison ke<sub>y</sub>s.
<table><tr><td colspan="3"></td><td colspan="5">λ = 0.01 n</td><td colspan="5">λ = 0.1 n</td></tr><tr><td></td><td></td><td></td><td colspan="3">Poisoning</td><td colspan="2">Upper Bound</td><td colspan="3">Poisoning</td><td colspan="2">Upper Bound</td></tr><tr><td>ε Dataset</td><td></td><td>Original</td><td>PGM-attack</td><td>Random</td><td>Random-Adj.</td><td>Agnostic UB</td><td>Instance UB</td><td>PGM-attack</td><td>Random</td><td>Random-Adj.</td><td>Agnostic UB</td><td>Instance UB</td></tr><tr><td>16</td><td>Amzn</td><td>7.94M</td><td>8.98M (1.13×)</td><td>7.95M (1.00×)</td><td>8.06M (1.01×)</td><td>15.9M (2.01×)</td><td>14.5M (1.82×)</td><td>12.1M (1.52×)</td><td>8.01M (1.01×)</td><td>9.12M (1.15×)</td><td>87.9M (11.1×)</td><td>21.2M (2.67×)</td></tr><tr><td></td><td>Osmc</td><td>6.18M</td><td>6.76M (1.10×)</td><td>6.19M (1.00×)</td><td>6.25M (1.01×)</td><td>14.2M (2.30×)</td><td>11.2M (1.81×)</td><td>9.18M (1.49×)</td><td>6.28M (1.02×)</td><td>6.92M (1.12×)</td><td>86.2M (14.0×)</td><td>17.4M (2.82×)</td></tr><tr><td>Face</td><td></td><td>2.12M</td><td>2.33M (1.10×)</td><td>2.12M (1.00×)</td><td>2.14M (1.01×)</td><td>4.12M (1.94×)</td><td>3.64M (1.71×)</td><td>3.08M (1.45×)</td><td>2.14M (1.01×)</td><td>2.34M (1.10×)</td><td>22.1M (10.4×)</td><td>5.36M (2.53×)</td></tr><tr><td></td><td>YCSB</td><td>70.1K</td><td>176K (2.51×)</td><td>72.5K (1.03×)</td><td>74.1K (1.06×)</td><td>2.07M (29.5×)</td><td>352K (5.02×)</td><td>810K (11.6×)</td><td>94.8K (1.35×)</td><td>111K (1.58×)</td><td>20.1M (286×)</td><td>1.3M (18.6×)</td></tr><tr><td></td><td>Longitudes</td><td>164K</td><td>262K (1.59×)</td><td>166K (1.01×)</td><td>169K (1.03×)</td><td>2.16M (13.2×)</td><td>514K (3.13×)</td><td>831K (5.05×)</td><td>180K (1.10×)</td><td>208K (1.26×)</td><td>20.2M (123×)</td><td>1.46M (8.86×)</td></tr><tr><td></td><td>Longlat</td><td>452K</td><td>543K (1.20×)</td><td>456K (1.01×)</td><td>459K (1.01×)</td><td>2.45M (5.42×)</td><td>992K (2.19×)</td><td>1.1M (2.44×)</td><td>473K (1.05×)</td><td>516K (1.14×)</td><td>20.5M (45.2×)</td><td>2.01M (4.45×)</td></tr><tr><td></td><td>Uniform</td><td>203K</td><td>320K (1.57×)</td><td>209K (1.03×)</td><td>209K (1.03×)</td><td>2.2M (10.8×)</td><td>615K (3.02×)</td><td>930K (4.57×)</td><td>262K (1.29×)</td><td>264K (1.30×)</td><td>20.2M (99.3×)</td><td>1.64M (8.04×)</td></tr><tr><td></td><td>Normal</td><td>203K</td><td>320K (1.57×)</td><td>205K (1.01×)</td><td>209K (1.03×)</td><td>2.2M (10.8×)</td><td>614K (3.02×)</td><td>929K (4.58×)</td><td>224K (1.10×)</td><td>264K (1.30×)</td><td>20.2M (99.5×)</td><td>1.64M (8.06×)</td></tr><tr><td></td><td>Lognormal</td><td>203K</td><td>320K (1.57×)</td><td>205K (1.01×)</td><td>209K (1.03×)</td><td>2.2M (10.8×)</td><td>614K (3.02×)</td><td>929K (4.58×)</td><td>224K (1.10×)</td><td>264K (1.30×)</td><td>20.2M (99.5×)</td><td>1.64M (8.06×)</td></tr><tr><td>32 Amzn</td><td></td><td>2.46M</td><td>2.97M (1.21×)</td><td>2.47M (1.00×)</td><td>2.52M (1.02×)</td><td>10.5M (4.25×)</td><td>5.15M (2.09×)</td><td>4.79M (1.94×)</td><td>2.49M (1.01×)</td><td>3.03M (1.23×)</td><td>82.5M (33.5×)</td><td>9.19M (3.73×)</td></tr><tr><td>Osmc</td><td></td><td>2.89M</td><td>3.14M (1.09×)</td><td>2.89M (1.00×)</td><td>2.92M (1.01×)</td><td>10.9M (3.77×)</td><td>5.31M (1.84×)</td><td>4.3M (1.49×)</td><td>2.92M (1.01×)</td><td>3.22M (1.11×)</td><td>82.9M (28.7×)</td><td>8.26M (2.86×)</td></tr><tr><td>Face</td><td></td><td>1.06M</td><td>1.16M (1.10×)</td><td>1.06M (1.00×)</td><td>1.07M (1.01×)</td><td>3.06M (2.90×)</td><td>1.82M (1.72×)</td><td>1.53M (1.45×)</td><td>1.06M (1.01×)</td><td>1.16M (1.10×)</td><td>21.1M (20.0×)</td><td>2.69M (2.55×)</td></tr><tr><td>YCSB</td><td></td><td>25.1K</td><td>63.9K (2.54×)</td><td>25.5K (1.02×)</td><td>25.9K (1.03×)</td><td>2.03M (80.6×)</td><td>113K (4.51×)</td><td>361K (14.4×)</td><td>29.3K (1.17×)</td><td>32.8K (1.31×)</td><td>20M (797×)</td><td>563K (22.4×)</td></tr><tr><td></td><td>Longitudes</td><td>62.6K</td><td>102K (1.63×)</td><td>62.9K (1.00×)</td><td>63.7K (1.02×)</td><td>2.06M (32.9×)</td><td>203K (3.25×)</td><td>380K (6.07×)</td><td>66.1K (1.06×)</td><td>74.3K (1.19×)</td><td>20.1M (320×)</td><td>653K (10.4×)</td></tr><tr><td></td><td>Longlat</td><td>219K</td><td>259K (1.18×)</td><td>221K (1.01×)</td><td>222K (1.01×)</td><td>2.22M (10.1×)</td><td>475K (2.17×)</td><td>537K (2.45×)</td><td>226K (1.03×)</td><td>244K (1.11×)</td><td>20.2M (92.2×)</td><td>969K (4.42×)</td></tr><tr><td></td><td>Uniform</td><td>53K</td><td>103K (1.94×)</td><td>54.5K (1.03×)</td><td>54.6K (1.03×)</td><td>2.05M (38.7×)</td><td>209K (3.95×)</td><td>391K (7.38×)</td><td>68.4K (1.29×)</td><td>69.1K (1.30×)</td><td>20.1M (378×)</td><td>670K (12.6×)</td></tr><tr><td></td><td>Normal</td><td>52.9K</td><td>103K (1.94×)</td><td>53.4K (1.01×)</td><td>54.5K (1.03×)</td><td>2.05M (38.8×)</td><td>209K (3.95×)</td><td>391K (7.39×)</td><td>58.2K (1.10×)</td><td>68.8K (1.30×)</td><td>20.1M (379×)</td><td>670K (12.7×)</td></tr><tr><td></td><td>Lognormal</td><td>52.9K</td><td>103K (1.94×)</td><td>53.5K (1.01×)</td><td>54.5K (1.03×)</td><td>2.05M (38.8×)</td><td>209K (3.95×)</td><td>391K (7.39×)</td><td>58.2K (1.10×)</td><td>68.8K (1.30×)</td><td>20.1M (379×)</td><td>670K (12.7×)</td></tr><tr><td>64 Amzn</td><td></td><td>797K</td><td>962K (1.21×)</td><td>798K (1.00×)</td><td>810K (1.02×)</td><td>8.8M (11.0×)</td><td>1.73M (2.17×)</td><td>1.73M (2.17×)</td><td>804K (1.01×)</td><td>929K (1.16×)</td><td>80.8M (101×)</td><td>3.24M (4.06×)</td></tr><tr><td>Osmc</td><td></td><td>1.38M</td><td>1.49M (1.08×)</td><td>1.38M (1.00×)</td><td>1.39M (1.01×)</td><td>9.38M (6.82×)</td><td>2.55M (1.85×)</td><td>2.06M (1.50×)</td><td>1.39M (1.01×)</td><td>1.52M (1.11×)</td><td>81.4M (59.2×)</td><td>3.97M (2.88×)</td></tr><tr><td>Face</td><td></td><td>523K</td><td>577K (1.10×)</td><td>523K (1.00×)</td><td>528K (1.01×)</td><td>2.52M (4.82×)</td><td>905K (1.73×)</td><td>760K (1.45×)</td><td>524K (1.00×)</td><td>575K (1.10×)</td><td>20.5M (39.2×)</td><td>1.35M (2.57×)</td></tr><tr><td>YCSB</td><td></td><td>6.95K</td><td>25.6K (3.68×)</td><td>7.12K (1.02×)</td><td>7.36K (1.06×)</td><td>2.01M (289×)</td><td>46.6K (6.71×)</td><td>169K (24.2×)</td><td>8.73K (1.26×)</td><td>10.3K (1.49×)</td><td>20M (2877×)</td><td>272K (39.1×)</td></tr><tr><td></td><td>Longitudes</td><td>27.8K</td><td>45.6K (1.64×)</td><td>27.9K (1.00×)</td><td>28.2K (1.01×)</td><td>2.03M (72.9×)</td><td>87K (3.13×)</td><td>185K (6.66×)</td><td>28.7K (1.03×)</td><td>31.7K (1.14×)</td><td>20M (720×)</td><td>312K (11.2×)</td></tr><tr><td>Longlat</td><td></td><td>112K</td><td>131K (1.17×)</td><td>113K (1.01×)</td><td>113K (1.01×)</td><td>2.11M (18.9×)</td><td>238K (2.12×)</td><td>271K (2.42×)</td><td>114K (1.02×)</td><td>123K (1.10×)</td><td>20.1M (180×)</td><td>483K (4.32×)</td></tr><tr><td>Uniform</td><td></td><td>13.5K</td><td>34.9K (2.60×)</td><td>13.9K (1.03×)</td><td>13.8K (1.03×)</td><td>2.01M (150×)</td><td>69K (5.12×)</td><td>177K (13.2×)</td><td>17.5K (1.30×)</td><td>17.6K (1.30×)</td><td>20M (1487×)</td><td>294K (21.8×)</td></tr><tr><td>Normal</td><td></td><td>13.5K</td><td>34.9K (2.59×)</td><td>13.6K (1.01×)</td><td>13.9K (1.03×)</td><td>2.01M (149×)</td><td>69K (5.11×)</td><td>177K (13.1×)</td><td>14.8K (1.10×)</td><td>17.5K (1.30×)</td><td>20M (1484×)</td><td>294K (21.8×)</td></tr><tr><td></td><td>Lognormal</td><td>13.5K</td><td>34.9K (2.59×)</td><td>13.6K (1.01×)</td><td>13.9K (1.03×)</td><td>2.01M (149×)</td><td>69K (5.11×)</td><td>177K (13.1×)</td><td>14.8K (1.10×)</td><td>17.6K (1.30×)</td><td>20M (1483×)</td><td>294K (21.8×)</td></tr><tr><td>128 Amzn</td><td></td><td>268K</td><td></td><td>268K (1.00×)</td><td>273K (1.02×)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Osmc</td><td></td><td>668K</td><td>335K (1.25×) 723K (1.08×)</td><td>669K (1.00×)</td><td>674K (1.01×)</td><td>8.27M (30.9×) 8.67M (13.0×)</td><td>619K (2.31×)</td><td>670K (2.50×) 1M (1.51×)</td><td>269K (1.01×) 671K (1.01×)</td><td>315K (1.17×) 738K (1.10×)</td><td>80.3M (300×) 80.7M (121×)</td><td>1.23M (4.60×) 1.93M (2.88×)</td></tr><tr><td>Face</td><td></td><td>256K</td><td>283K (1.10×)</td><td>256K (1.00×)</td><td>259K (1.01×)</td><td>2.26M (8.80×)</td><td>1.23M (1.85×) 444K (1.73×)</td><td>374K (1.46×)</td><td>257K (1.00×)</td><td>282K (1.10×)</td><td>20.3M (79.0×)</td></table>

D.1.2 Robustness to Unknown Index Parameters. We further evaluate whether PGM-attack remains efective when the attacker does not know the true PGM-index parameter. Specifically, we generate poison keys using an assumed value $\varepsilon _ { \mathrm { a t t a c k } }$ and construct the index using the actual value $\varepsilon _ { \mathrm { t r u e } }$ . Table 6 reports the resulting $m _ { \mathrm { o p t } }$

As expected, the attack is generally most efective when �<sub>atack</sub> = � . Nevertheless, PGM-attack remains efective even under parameter mismatch. For example, on Amzn with $\varepsilon _ { \mathrm { t r u e } } = 1 2 8$ , the attack increases $m _ { \mathrm { o p t } }$ by 2.50× when the attacker knows the true parameter and sets $\varepsilon _ { \mathrm { a t t a c k } } = 1 2 8$ . When the attacker instead assumes $\varepsilon _ { \mathrm { a t t a c k } } = 3 2$ , the increase decreases to 1.36×. Even so, this remains larger than the increases caused by random and random-adjacent poisoning, which are 1.01× and 1.17×, respectively (Table 5). Similarly, on YCSB with $\varepsilon _ { \mathrm { t r u e } } = 1 2 8 _ { \mathrm { : } }$ , mismatched attacks still increase $m _ { \mathrm { o p t } }$ by at least 22.7×, far exceeding the 1.99× and 2.42× increases caused by random and random-adjacent poisoning.

These results suggest that poison keys generated to target a PLA with one value of� partially transfer to PLAs constructed with other values. Thus, exact knowledge of � is not necessary to generate efective poison keys. In this sense, our white-box attack already remains efective in a gray-box setting where the attacker knows the data but not the value of �. Our method therefore provides a useful foundation for developing more sophisticated attacks under gray-box and black-box settings.

D.1.3 Ablation Study on �. To study the efect of $\mu ,$ we report in Figure 18 the results averaged over five seeds on $n = 1 M$ keys. For � ∈ {16, 32, 64, 128}, we plot the factor by which $m _ { \mathrm { o p t } }$ increases relative to legitimate keys. Very small or very large � can reduce the amplification: small $\mu$ under-penalizes poison usage and exhausts budget unevenly, while large � overreacts to budget imbalance and behaves unstably. Empirically, $\mu \approx 1 0 0$ works robustly; dynamic � improves over fixing $\theta = \theta _ { 0 } ( \mu = 0 )$ by up to 20×. These results show that dynamically updating � yields higher measured $m _ { \mathrm { o p t } }$ than fixing $\theta = \theta _ { 0 }$

D.1.4 Poison Generation Time. Table 7 reports the poison generation time for all combinations of � and �. Across all datasets and parameter settings, poison generation takes between 0.6 and 14.7 hours. Although this computational cost is non-negligible, it remains practical for an ofline attacker because the poison keys can be generated in advance and subsequently inserted into the target index.

T<sub>a</sub>bl<sub>e</sub> 6<sub>:</sub> $m _ { \mathrm { o p t } }$ <sub>a</sub>ft<sub>e</sub>r PGM-attack <sub>w</sub>h<sub>e</sub>n th<sub>e</sub> PGM-ind<sub>e</sub>x i<sub>s co</sub>n-<sub>s</sub>t<sub>ruc</sub>t<sub>e</sub>d <sub>w</sub>ith $\varepsilon _ { \mathrm { t r u e } } ,$ but <sub>p</sub>oisons are <sub>g</sub>enerated assumin<sub>g</sub> that th<sub>e</sub> PGM-ind<sub>e</sub>x <sub>pa</sub>r<sub>a</sub>m<sub>e</sub>t<sub>e</sub>r i<sub>s</sub> $\varepsilon _ { \mathrm { a t t a c k } } . \lambda = 0 . 1 n .$
<table><tr><td rowspan="2">Amzn</td><td rowspan="2"> $\varepsilon _ { \mathrm { t r u e } } = 1 6$ </td><td rowspan="2"> $\varepsilon _ { \mathrm { t r u e } } = 3 2$ </td><td rowspan="2"> $\varepsilon _ { \mathrm { t r u e } } = 6 4$ </td><td rowspan="2"> $\varepsilon _ { \mathrm { t r u e } } = 1 2 8$ </td></tr><tr><td></td></tr><tr><td> $\varepsilon _ { \mathrm { a t t a c k } } = 1 6$ </td><td> $\mathbf { 1 2 . 1 M \left( 1 . 5 2 \times \right) }$ </td><td> $3 . 1 9 M \left( 1 . 3 0 \times \right)$ </td><td> $1 . 0 4 \mathrm { M } \left( 1 . 3 1 \times \right)$ </td><td> $3 8 6 \mathrm { K } \left( 1 . 4 4 \times \right)$ </td></tr><tr><td> $\varepsilon _ { \mathrm { a t t a c k } } = 3 2$ </td><td> $1 0 . 2 \mathrm { M } \stackrel { \cdot } { ( 1 . 2 8 \times ) }$ </td><td> $\mathbf { 4 . 7 9 M \left( 1 . 9 4 \times \right) }$ </td><td> $1 . 0 5 \mathrm { M } \dot { ( 1 . 3 2 \times ) }$ </td><td> $3 6 5 \mathrm { K } \left( 1 . 3 6 \times \right)$ </td></tr><tr><td> $\varepsilon _ { \mathrm { a t t a c k } } = 6 4$ </td><td> $9 . 2 \mathrm { M } \left( 1 . 1 6 \times \right)$ </td><td> $3 . 5 5 \mathrm { M } \dot { ( 1 . 4 4 \times ) }$ </td><td> $\mathbf { 1 . 7 3 M } \left( 2 . 1 7 \times \right)$ </td><td> $3 9 5 \mathrm { K } \left( 1 . 4 7 \times \right)$ </td></tr><tr><td> $\varepsilon _ { \mathrm { a t t a c k } } = 1 2 8$ </td><td> $8 . 5 2 \mathrm { M } \stackrel { . } { ( 1 . 0 7 \times ) }$ </td><td> $3 . 0 3 M \overset { \cdot } { ( } 1 . 2 3 \times \overset { \cdot } { ) }$ </td><td> $1 . 3 3 \mathrm { M } \dot { ( } 1 . 6 7 \times \dot { ) }$ </td><td>670K (2.50×)</td></tr><tr><td>Osmc</td><td> $\varepsilon _ { \mathrm { t r u e } } = 1 6$ </td><td> $\varepsilon _ { \mathrm { t r u e } } = 3 2$ </td><td> $\varepsilon _ { \mathrm { t r u e } } = 6 4$ </td><td> $\varepsilon _ { \mathrm { t r u e } } = 1 2 8$ </td></tr><tr><td> $\varepsilon _ { \mathrm { a t t a c k } } = 1 6$ </td><td> $9 . 1 8 \mathbf { M } \left( 1 . 4 9 \times \right)$ </td><td> $3 . 4 7 \mathrm { M } \left( 1 . 2 0 \times \right)$ </td><td> $1 . 6 \mathrm { M } \left( 1 . 1 7 \times \right)$ </td><td>780K (1.17×)</td></tr><tr><td> $\varepsilon _ { \mathrm { a t t a c k } } = 3 2$ </td><td> $7 . 4 8 \mathrm { M } \dot { ( 1 . 2 1 \times ) }$ </td><td> $\mathbf { 4 . 3 M \left( 1 . 4 9 \times \right) }$ </td><td> $1 . 6 6 \mathrm { M } \left( 1 . 2 1 \times \right)$ </td><td>773K (1.16×)</td></tr><tr><td> $\varepsilon _ { \mathrm { a t t a c k } } = 6 4$ </td><td> $6 . 8 4 \mathrm { M } \stackrel { . } { ( } 1 . 1 1 \times \stackrel { . } { ) }$ </td><td> $3 . 5 3 M \overset { \cdot } { ( } 1 . 2 2 \times \overset { \cdot } { ) }$ </td><td> $\mathbf { 2 . 0 6 M } \left( \mathbf { \bar { 1 . 5 0 } } \times \right)$ </td><td>809K (1.21×)</td></tr><tr><td> $\varepsilon _ { \mathrm { a t t a c k } } = 1 2 8$ </td><td> $6 . 5 3 \mathrm { M } \left( 1 . 0 6 \times \right)$ </td><td> $3 . 2 2 \mathrm { M } \left( 1 . 1 1 \times \right)$ </td><td> $1 . 6 9 \mathrm { M } \stackrel { \cdot } { ( 1 . 2 3 \times ) }$ </td><td>1M (1.51×)</td></tr><tr><td>Face</td><td> $\varepsilon _ { \mathrm { t r u e } } = 1 6$ </td><td> $\varepsilon _ { \mathrm { t r u e } } = 3 2$ </td><td> $\varepsilon _ { \mathrm { t r u e } } = 6 4$ </td><td> $\varepsilon _ { \mathrm { t r u e } } = 1 2 8$ </td></tr><tr><td> $\varepsilon _ { \mathrm { a t t a c k } } = 1 6$ </td><td> $\mathbf { 3 . 0 8 M \left( 1 . 4 5 \times \right)} $ </td><td> $1 . 1 9 \mathrm { M } \left( 1 . 1 3 \times \right)$ </td><td> $5 8 1 \mathrm { K } \left( 1 . 1 1 \times \right)$ </td><td>286K (1.11×)</td></tr><tr><td> $\varepsilon _ { \mathrm { a t t a c k } } = 3 2$ </td><td>2.59M (1.22×) 1.53M (1.45×)</td><td></td><td> $5 9 4 \mathrm { K } \left( 1 . 1 4 \times \right)$ </td><td>286K (1.12×)</td></tr><tr><td> $\varepsilon _ { \mathrm { a t t a c k } } = 6 4$   $\varepsilon _ { \mathrm { a t t a c k } } = 1 2 8$ </td><td> $2 . 4 \mathrm { M } \left( 1 . 1 3 \times \right)$ </td><td> $1 . 3 \mathrm { M } \mathrm { \dot { ( } } 1 . 2 3 \times \mathrm { \dot { ) } }$ </td><td>760K (1.45×)</td><td>293K (1.15×)</td></tr><tr><td>YCSB</td><td> $2 . 2 9 \mathrm { M } \left( 1 . 0 8 \times \right)$ </td><td> $1 . 2 \mathrm { M } \left( 1 . 1 3 \times \right)$ </td><td> $6 4 6 \mathrm { K } \left( 1 . 2 3 \times \right)$ </td><td>374K (1.46×)</td></tr><tr><td></td><td> $\varepsilon _ { \mathrm { t r u e } } = 1 6$ </td><td> $\varepsilon _ { \mathrm { t r u e } } = 3 2$ </td><td> $\varepsilon _ { \mathrm { t r u e } } = 6 4$ </td><td> $\varepsilon _ { \mathrm { t r u e } } = 1 2 8$ </td></tr><tr><td> $\varepsilon _ { \mathrm { a t t a c k } } = 1 6$ </td><td> $\mathbf { 8 1 0 K } \left( \mathbf { 1 1 . 6 } \times \right)$ </td><td> $1 4 4 \mathrm { K } \left( 5 . 7 4 \times \right)$ </td><td> $4 7 . 6 \mathrm { K } \left( 6 . 8 5 \times \right)$ </td><td>15.1K (22.7×)</td></tr><tr><td> $\varepsilon _ { \mathrm { a t t a c k } } = 3 2$ </td><td> $6 2 2 \mathrm { K } \dot { ( 8 . 8 7 \times ) }$ </td><td> $\mathbf { 3 6 1 K } \left( \mathbf { \bar { 1 4 . 4 } } \times \right)$ </td><td> $6 4 . 7 \mathrm { K } \left( 9 . 3 0 \times \right)$   $\mathbf { 1 6 9 K } \left( \mathbf { \dot { 2 4 . 2 } } \times \right)$ </td><td>24.1K (36.3×)</td></tr><tr><td> $\varepsilon _ { \mathrm { a t t a c k } } = 6 4$   $\varepsilon _ { \mathrm { a t t a c k } } = 1 2 8$ </td><td> $3 2 8 \mathrm { K } \left( 4 . 6 7 \times \right)$   $2 0 6 \mathrm { K } \left( 2 . 9 3 \times \right)$ </td><td> $2 6 4 \mathrm { K } \dot { ( 1 0 . 5 \times ) }$   $1 6 2 \mathrm { K } \left( 6 . 4 4 \times \right)$ </td><td> $1 0 3 \mathrm { K } \dot { ( 1 4 . 8 \times ) }$ </td><td>35.6K (53.7×) 79.8K (120×)</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathrm { L o n g i t u d e s }$ </td><td> $\varepsilon _ { \mathrm { t r u e } } = 1 6$ </td><td> $\varepsilon _ { \mathrm { t r u e } } = 3 2$ </td><td> $\varepsilon _ { \mathrm { t r u e } } = 6 4$ </td><td> $\varepsilon _ { \mathrm { t r u e } } = 1 2 8$ </td></tr><tr><td> $\varepsilon _ { \mathrm { a t t a c k } } = 1 6$ </td><td> $\mathbf { 8 3 1 K } \left( \mathbf { 5 . 0 5 } \times \right)$ </td><td> $1 7 4 \mathrm { K } \left( 2 . 7 7 \times \right)$ </td><td> $6 6 . 7 \mathrm { K } \left( 2 . 4 0 \times \right)$ </td><td>33.1K (2.41×)</td></tr><tr><td> $\varepsilon _ { \mathrm { a t t a c k } } = 3 2$ </td><td> $4 9 0 \mathrm { K } \stackrel { \cdot } { ( 2 . 9 8 \times ) }$ </td><td> $\mathbf { 3 8 0 K } \left( \mathbf { 6 . 0 7 } \times \right)$ </td><td> $7 9 . 7 \mathrm { K } \left( 2 . 8 6 \times \right)$ </td><td>29.5K (2.15×)</td></tr><tr><td> $\varepsilon _ { \mathrm { a t t a c k } } = 6 4$   $\varepsilon _ { \mathrm { a t t a c k } } = 1 2 8$ </td><td> $3 2 4 \mathrm { K } \left( 1 . 9 7 \times \right)$   $2 4 5 \mathrm { K } \left( 1 . 4 9 \times \right)$ </td><td> $2 2 0 \mathrm { K } \dot { ( 3 . 5 1 \times ) }$   $1 4 2 \mathrm { K } \left( 2 . 2 6 \times \right)$ </td><td> $\mathbf { 1 8 5 K } \left( 6 . 6 6 \times \right)$  106K (3.82×)</td><td>39.6K (2.89×) 92.4K (6.73×)</td></tr><tr><td>Longlat</td><td> $\varepsilon _ { \mathrm { t r u e } } = 1 6$ </td><td></td><td></td><td></td></tr><tr><td></td><td></td><td> $\varepsilon _ { \mathrm { t r u e } } = 3 2$ </td><td> $\varepsilon _ { \mathrm { t r u e } } = 6 4$ </td><td> $\varepsilon _ { \mathrm { t r u e } } = 1 2 8$ </td></tr><tr><td> $\varepsilon _ { \mathrm { a t t a c k } } = 1 6$   $\varepsilon _ { \mathrm { a t t a c k } } = 3 2$ </td><td> $\mathbf { 1 . 1 M \left( 2 . 4 4 \times \right) }$ </td><td> $3 2 3 \mathrm { K } \left( 1 . 4 7 \times \right)$ </td><td> $1 4 8 \mathrm { K } \left( 1 . 3 2 \times \right)$   $1 6 4 \mathrm { K } \left( 1 . 4 6 \times \right)$ </td><td> $7 6 . 1 \mathrm { K } \left( 1 . 3 1 \times \right)$   $7 4 . 1 \mathrm { K } \left( 1 . 2 7 \times \right)$ </td></tr><tr><td> $\varepsilon _ { \mathrm { a t t a c k } } = 6 4$ </td><td> $7 6 5 \mathrm { K } \dot { ( 1 . 6 9 \times ) }$   $6 1 1 \mathrm { K } \left( 1 . 3 5 \times \right)$ </td><td> $\mathbf { 5 3 7 K } \left( \mathbf { 2 . 4 5 \times } \right)$   $3 7 5 \mathrm { K } \dot { ( 1 . 7 1 \times ) }$ </td><td> $\mathbf { 2 7 1 K } \left( \mathbf { \dot { 2 } . 4 2 \times } \right)$ </td><td>84.2K (1.45×)</td></tr><tr><td> $\varepsilon _ { \mathrm { a t t a c k } } = 1 2 8$ </td><td> $5 3 4 \mathrm { K } \left( 1 . 1 8 \times \right)$ </td><td> $2 9 8 \mathrm { K } \left( 1 . 3 6 \times \right)$ </td><td> $1 9 0 \mathrm { K } \left( 1 . 7 0 \times \right)$ </td><td>138K (2.37×)</td></tr><tr><td>Uniform</td><td> $\varepsilon _ { \mathrm { t r u e } } = 1 6$ </td><td> $\varepsilon _ { \mathrm { t r u e } } = 3 2$ </td><td> $\varepsilon _ { \mathrm { t r u e } } = 6 4$ </td><td> $\varepsilon _ { \mathrm { t r u e } } = 1 2 8$ </td></tr><tr><td> $\varepsilon _ { \mathrm { a t t a c k } } = 1 6$ </td><td> $\mathbf { 9 3 0 K } \left( 4 . 5 7 \times \right)$ </td><td> $1 9 3 \mathrm { K } \left( 3 . 6 3 \times \right)$ </td><td> $6 9 \mathrm { K } \left( 5 . 1 2 \times \right)$ </td><td> $2 8 . 1 \mathrm { K } \left( 8 . 2 1 \times \right)$ </td></tr><tr><td> $\varepsilon _ { \mathrm { a t t a c k } } = 3 2$ </td><td> $5 8 8 \mathrm { { K } \setminus ( 2 . 8 9 \times ) }$ </td><td> $\mathbf { 3 9 1 K } \left( 7 . 3 8 \times \right)$ </td><td> $7 9 . 8 \mathrm { K } \left( \dot { 5 } . 9 3 \times \right)$ </td><td>27.3K (7.99×)</td></tr><tr><td> $\varepsilon _ { \mathrm { a t t a c k } } = 6 4$ </td><td> $4 0 3 \mathrm { K } \left( 1 . 9 8 \times \right)$ </td><td> $2 4 3 \mathrm { K } \dot { ( 4 . 5 9 \times ) }$ </td><td> $\mathbf { 1 7 7 K } \left( \mathbf { 1 3 . 2 } \times \right)$ </td><td>36.7K (10.7×)</td></tr><tr><td> $\varepsilon _ { \mathrm { a t t a c k } } = 1 2 8$ </td><td> $3 0 3 \mathrm { K } \left( 1 . 4 9 \times \right)$ </td><td> $1 5 4 \mathrm { K } \left( 2 . 9 1 \times \right)$ </td><td> $1 1 1 \mathrm { K } \left( 8 . 2 3 \times \right)$ </td><td>83.7K (24.5×)</td></tr><tr><td>Normal</td><td> $\varepsilon _ { \mathrm { t r u e } } = 1 6$ </td><td> $\varepsilon _ { \mathrm { t r u e } } = 3 2$ </td><td> $\varepsilon _ { \mathrm { t r u e } } = 6 4$ </td><td> $\varepsilon _ { \mathrm { t r u e } } = 1 2 8$ </td></tr><tr><td> $\varepsilon _ { \mathrm { a t t a c k } } = 1 6$ </td><td> $\mathbf { 9 2 9 K } \left( 4 . 5 8 \times \right)$ </td><td> $1 9 2 \mathrm { K } \left( 3 . 6 4 \times \right)$ </td><td> $6 9 \mathrm { K } \left( 5 . 1 2 \times \right)$ </td><td>28.3K (8.06×)</td></tr><tr><td>εattack = 32</td><td> $5 8 7 \mathrm { K } \stackrel { \cdot } { ( 2 . 8 9 \times ) }$ </td><td> $\mathbf { 3 9 1 K } \left( \mathbf { \dot { 7 } . 3 9 } \times \right)$ </td><td> $7 9 . 7 \mathrm { K } \left( 5 . 9 1 \times \right)$ </td><td>27.2K (7.76×)</td></tr><tr><td> $\varepsilon _ { \mathrm { a t t a c k } } = 6 4 $ </td><td> $4 0 3 \mathrm { K } \left( 1 . 9 8 \times \right)$ </td><td> $2 4 3 \mathrm { K } \left( 4 . 6 0 \times \right)$ </td><td> $\mathbf { 1 7 7 K } \left( \mathbf { \dot { 1 3 . 1 } } \times \right)$ </td><td>36.7K (10.5×)</td></tr><tr><td> $\varepsilon _ { \mathrm { a t t a c k } } = 1 2 8$ </td><td> $3 0 2 \mathrm { K } \left( 1 . 4 9 \times \right)$ </td><td> $1 5 3 \mathrm { K } \left( 2 . 9 0 \times \right)$ </td><td> $1 1 0 \mathrm { K } \dot { ( } 8 . 1 8 \times \dot { ) }$ </td><td>83.5K (23.8×)</td></tr><tr><td>Lognormal</td><td> $\varepsilon _ { \mathrm { t r u e } } = 1 6$ </td><td> $\varepsilon _ { \mathrm { t r u e } } = 3 2$ </td><td> $\varepsilon _ { \mathrm { t r u e } } = 6 4$ </td><td> $\varepsilon _ { \mathrm { t r u e } } = 1 2 8$ </td></tr><tr><td rowspan="2"> $\varepsilon _ { \mathrm { a t t a c k } } = 1 6$ </td><td> $\mathbf { 9 2 9 K } \left( 4 . 5 8 \times \right)$ </td><td> $1 9 2 \mathrm { K } \left( 3 . 6 3 \times \right)$ </td><td> $6 9 \mathrm { K } \left( 5 . 1 2 \times \right)$ </td><td> $2 8 . 3 \mathrm { K } \left( 8 . 1 0 \times \right)$ </td></tr><tr><td> $5 8 7 \mathrm { K } \stackrel { \cdot } { ( 2 . 8 9 \times ) }$ </td><td> $\mathbf { 3 9 1 K } \left( \mathbf { \dot { 7 } . 3 9 } \times \right)$ </td><td> $7 9 . 7 \mathrm { K } \left( \mathrm { \bar { 5 } } . 9 1 \times \right)$ </td><td>27.2K (7.81×)</td></tr><tr><td> $\varepsilon _ { \mathrm { a t t a c k } } = 3 2$   $\varepsilon _ { \mathrm { a t t a c k } } = 6 4$ </td><td> $4 0 3 \mathrm { K } \left( 1 . 9 8 \times \right)$ </td><td> $2 4 3 \mathrm { K } \dot { ( 4 . 6 0 \times ) }$ </td><td>177K (13.1×)</td><td>36.7K (10.5×)</td></tr><tr><td></td><td> $3 0 2 \mathrm { K } \left( 1 . 4 9 \times \right)$ </td><td></td><td></td><td></td></tr><tr><td> $\varepsilon _ { \mathrm { a t t a c k } } = 1 2 8$ </td><td></td><td>153K (2.90×)</td><td> $1 1 0 \mathrm { K } \dot { ( } 8 . 1 6 \times \dot { ) }$ </td><td>83.4K (23.9×)</td></tr></table>

The generation time tends to increase as either � or � increases. The main reason is that these settings allow the selected poison keys to shorten the current segment more substantially. PGM-attack processes the key sequence from left to right. At each step, it generates poison candidates for the segment starting at the current segment-start position, selects poison keys, and recomputes the resulting segmentation. If the selected poisons shorten the segment only slightly, the next segment starts relatively far to the right, and the algorithm soon moves to a mostly new region of the dataset. In contrast, when the shortening is large, the next segment starts much earlier than before poisoning. Consequently, the algorithm must again generate and evaluate poison candidates for a segment that substantially overlaps a previously processed region. This repeated recomputation over overlapping regions increases the total generation time. Larger � permits greater reductions in segment length, while a larger poison budget � increases the cumulative shortening caused by the selected poisons; both lead to more repeated processing.

![](images/e2cbbb8cee0c9d075264afc3418b714776c7028a66c289ed90d2bb2616bab98b.jpg)  
Fi<sub>g</sub>ure 18: Efect of <sub>�</sub> on $m _ { \mathrm { o p t } }$ for � = 100,000.

A secondary reason for the increase with � is that the segments before poisoning tend to contain more keys. For a segment containing � legitimate keys, DI-Consecutive requires O (� log �) time to generate a poison candidate. Thus, as � increases and the unpoisoned segment length � becomes larger, the cost of processing each candidate segment also grows superlinearly in �.

## D<sub>.</sub>2 Additi<sub>ona</sub>l R<sub>esu</sub>lt<sub>s on</sub> I<sub>n</sub>d<sub>ex</sub> P<sub>er</sub>f<sub>ormance</sub>

We further evaluate the practical impact of PGM-attack on index size and query time. Because this extended evaluation covers more datasets, indexes, and poisoning methods than the evaluation in Section 6.5, we present the results as figures rather than tables for readability. Figures 19 and 20 report the index size and query time, respectively, for � = 128 and 10% poisoning (� = 0.1�). For each PGM-attack bar, we additionally display the ratio relative to the corresponding original, unpoisoned index above the bar.

Poisoning Methods. In addition to our proposed PGM-attack, we evaluate two random baselines, Random and Random-Adj. Random selects poison keys uniformly at random, without replacement, from the unoccupied integers within the legitimate key range. Motivated by Theorem 1, Random-Adj. instead samples uniformly from unoccupied integers adjacent to legitimate keys. We also evaluate the poisoning attack proposed for RMIs [22], which we refer to as RMI-attack. Although RMI-attack was specifically designed to maximize the squared prediction error of RMI models, we apply the poison keys generated by every method to every evaluated index to examine their transferability across index architectures.

We further evaluate two methods proposed as algorithmic complexity attacks (ACAs) against ALEX. The first is a space ACA targeting ALEX data nodes, which we denote by ALEX-DN-ACA [36]. The second, denoted by SZEGP, is a space ACA that induces repeated structural expansion in ALEX through adversarial insertions [35]. Both methods were originally designed to increase the memory consumption of an already constructed ALEX by dynamically inserting adversarially selected keys. This difers from our poisoning setting, in which the attacker injects poison keys into the dataset before index construction. To ensure a consistent comparison, we evaluate all methods under our poisoning setting: each index is bulk-loaded from the dataset containing the generated poison keys, rather than being attacked through online insertions after construction.

![](images/07669de09005188fe31ff85efd84f19a55f4a16526312e92780eba3682d44c33.jpg)  
Figure 19: Index sizes before and after 10% poisoning (� = 0.1�) for � = 128. Numbers above the PGM-attack bars indicate ratios relative to the corres<sub>p</sub>ondin<sub>g</sub> ori<sub>g</sub>inal indexes.

![](images/9c9ce3399aa8818bb4d644d76b1762c258382e72943623b04644302a52696faf.jpg)  
Figure 20: Query times before and after 10% poisoning (� = 0.1�) for � = 128. Numbers above the PGM-attack bars indicate ratios relative to the corres<sub>p</sub>ondin<sub>g</sub> ori<sub>g</sub>inal indexes.

Table 7: Poison generation time (hours).
<table><tr><td></td><td colspan="4"> $\lambda = 0 . 0 1 n$ </td><td colspan="4"> $\lambda = 0 . 1 n$ </td></tr><tr><td>Dataset</td><td> $\varepsilon = 1 6$ </td><td> $\varepsilon = 3 2$ </td><td> $\varepsilon = 6 4$ </td><td> $\varepsilon = 1 2 8$ </td><td> $\varepsilon = 1 6$ </td><td> $\varepsilon = 3 2$ </td><td> $\varepsilon = 6 4$ </td><td> $\varepsilon = 1 2 8$ </td></tr><tr><td>Amzn</td><td>3.1</td><td>2.9</td><td>4.3</td><td>4.7</td><td>4.9</td><td>5.8</td><td>8.7</td><td>12.3</td></tr><tr><td>Osmc</td><td>2.9</td><td>2.5</td><td>2.6</td><td>2.3</td><td>4.9</td><td>4.8</td><td>4.8</td><td>5.0</td></tr><tr><td>Face</td><td>0.9</td><td>0.8</td><td>0.8</td><td>0.6</td><td>1.3</td><td>1.2</td><td>1.0</td><td>1.0</td></tr><tr><td>YCSB</td><td>1.5</td><td>1.2</td><td>2.0</td><td>3.3</td><td>3.5</td><td>4.2</td><td>6.0</td><td>14.7</td></tr><tr><td>Longitudes</td><td>1.0</td><td>0.9</td><td>1.1</td><td>1.1</td><td>3.0</td><td>4.3</td><td>5.0</td><td>5.3</td></tr><tr><td>Longlat</td><td>0.7</td><td>0.7</td><td>0.7</td><td>0.6</td><td>2.2</td><td>2.7</td><td>2.7</td><td>2.6</td></tr><tr><td>Uniform</td><td>1.0</td><td>1.2</td><td>1.6</td><td>1.9</td><td>2.3</td><td>3.7</td><td>5.3</td><td>7.3</td></tr><tr><td>Normal</td><td>1.0</td><td>1.2</td><td>1.6</td><td>1.8</td><td>2.3</td><td>3.9</td><td>5.5</td><td>7.1</td></tr><tr><td>Lognormal</td><td>1.0</td><td>1.2</td><td>1.6</td><td>1.8</td><td>2.3</td><td>3.6</td><td>5.3</td><td>7.2</td></tr></table>

Efect on PLA-Based Learned Indexes. PGM-attack is not only efective against PGM-index, but also efective against all four other PLA-based learned indexes. The largest multiplicative increases tend to occur when the original index is particularly small, that is, when the index benefits strongly from the regularity and lin ear approximability of the original key distribution. In such cases, poisoning removes much of this advantage by creating many local regions that cannot be accurately represented using a small number of linear segments.

This efect is most pronounced on YCSB, although PGM-attack produces greater performance degradation than the other poisoning methods in nearly all cases on the remaining real-world and synthetic datasets. On YCSB, relative to the original index, PGMattack increases the index size by up to 18.5× for FITing-Tree, 3.03× for RadixSpline, 37.5× for PLEX, and 35,770× for PGM++. The exceptionally large ratio for PGM++ partly reflects its extremely small original index size on YCSB. Across all other real-world and synthetic datasets, PGM-attack increases the sizes of all PLAbased learned indexes by factors ranging from 1.13× to 130×. In nearly every case, it produces a larger increase than any competing poisoning method.

These results suggest that PGM-attack exploits the underlying PLA approximation problem rather than an implementationspecific property of the PGM-index. Its poison keys therefore transfer to learned indexes that employ diferent PLA construction algorithms, recursive organizations, or auxiliary search structures.

The increase in query time is less pronounced than the increase in index size. Nevertheless, in most cases, PGM-attack causes the largest query-time degradation among the evaluated poisoning methods, increasing query time by up to 1.30×. This slowdown is likely attributable, at least in part, to the larger structures induced by the increased number of segments, which may lead to additional cache misses and traversal overhead.

Efect on Other Learned Indexes. PGM-attack can also substantially increase both the index size and query time of ALEX. In particular, it increases the ALEX index size by 98.6× on YCSB and by 129× on Uniform. These large increases can be explained by the fact that ALEX’s bulk-loading algorithm adaptively determines its index structure based on model accuracy. During construction, ALEX evaluates the model accuracy at each node and decides whether to retain the keys in a single data node or partition them among multiple child nodes. Lower model accuracy can therefore lead to a more highly partitioned structure and a larger overall index.

Because ALEX uses linear models to navigate its nodes, the poison keys generated by PGM-attack likely also increase the prediction errors of these models. These results demonstrate that the transferability of PGM-attack extends beyond the family of PLA-based learned indexes.

Comparison with RMI-attack. Notably, PGM-attack is at least as efective as RMI-attack, even on RMI, for which the latter was specifically designed. On RMI, PGM-attack increases query time by up to 1.40×, whereas RMI-attack increases it by at most 1.04×.

One possible explanation is the diference between the objectives optimized by the two attacks. RMI-attack selects poison keys to maximize the squared prediction error of the RMI models. However, an RMI answers a query by performing a last-mile search, such as linear or exponential search, around the predicted position. Squared error may therefore be imperfectly aligned with the actual queryprocessing cost: a small number of very large errors can dominate the objective without proportionally increasing the average lastmile search cost. In contrast, PGM-attack targets local worst-case approximation error and places poison keys in regions where they substantially disrupt linear approximation. This objective may be more closely aligned with the last-mile search cost incurred by learned indexes.

The optimization procedure used by RMI-attack may also limit its efectiveness. In particular, it assumes that leafmodels are responsible for equal numbers of keys and employs a hill-climbing procedure initialized from a uniform poison allocation. Consequently, the resulting poison set may remain far from the optimum for a particular key distribution or RMI configuration.

Comparison with Attacks on ALEX.. A similar pattern arises for ALEX. Under our evaluation setting, PGM-attack causes greater degradation than the attacks proposed specifically for ALEX [35, 36]. This diference is largely attributable to the threat models for which these ACA methods were designed. They exploit algorithmic behavior triggered by online insertions into an already constructed ALEX, such as ineficient resizing, splitting, or structural expansion, rather than constructing a poisoned dataset from which ALEX is subsequently bulk-loaded.

When the keys generated by these attacks are instead included during bulk loading, ALEX can jointly optimize its initial structure over both legitimate and adversarial keys, avoiding many of the transient ineficiencies that arise during online updates. Bulk load ing is therefore generally a more favorable setting for the index than adversarial online insertion. The efectiveness of PGM-attack even in this setting suggests that it does not merely exploit an inefficient update procedure. Rather, it reveals a broader vulnerability in the model-based construction objective itself: carefully placed keys can make the data intrinsically dificult to represent using the simple local models on which the index relies.

Transferability across Learned Indexes. Finally, PGM-attack exhibits substantially greater transferability than the prior indexspecific attacks. The efects of RMI-attack, ALEX-DN-ACA, and SZEGP are generally limited outside the architectures for which they were designed. By contrast, PGM-attack consistently degrades a broad range of learned indexes.

We attribute this transferability to a design principle shared by many learned indexes: they partition the key space into regions and represent each region using a small, simple model, often a linear model. PGM-attack concentrates poison keys at locations where they substantially increase local approximation dificulty and worst case prediction error. Because these quantities are closely related to the number of required models, the complexity of the index structure, and the cost of last-mile search, the resulting poison keys can remain efective across substantially diferent learned-index architectures.

## E DUPLICATE-ALLOWED POISONING

In this section, we discuss poisoning in the duplicate-allowed setting. We first describe how duplicates are handled in the actual PGM-index implementation. We then discuss how to generate poisons under this setting. Finally, we experimentally evaluate PGMattack in this setting.

## E<sub>.</sub>1 PGM-ind<sub>e</sub>x <sub>w</sub>ith D<sub>up</sub>li<sub>ca</sub>t<sub>es</sub>

When given a sorted integer array containing duplicate keys, the PGM-index does not construct the PLA directly from the original key-rank pairs. Instead, it constructs the PLA over virtual points introduced to handle duplicates. Specifically, it first adds $( x _ { 1 } , 1 )$ Then, for each position $i \in \{ 2 , 3 , . . . , n \}$ , if �<sub>�</sub> difers from the previous key, it adds $( x _ { i } , i )$ . If �<sub>�</sub> is a duplicate, it does not add that point directly; instead, only when $x _ { i }$ is at the end of a duplicate run and there is a gap before the next distinct key, it adds $( x _ { i } + 1 , i )$ Algorithm 3 presents the pseudocode, and Figure 21 illustrates the resulting virtual points. The PLA is then constructed with error at most � on these virtual points.

Algorithm 3 PLA construction in PGM-index with duplicate keys   
Re<sub>q</sub>uire:   
Nondecreasing keys array $X = ( x _ { 1 } , x _ { 2 } , \ldots , x _ { n } ) ;$   
Error parameter �.   
Ensure: A PLA built for X with error at most �.   
1: PLA ← initializePLA(�)   
2: $\operatorname { P L A . a d d P o i n t } ( x _ { 1 } , 1 )$   
3: <sup>f</sup>or $i \gets 2$ t<sub>o �</sub> d<sub>o</sub>   
4: $\mathbf { i f } \ x _ { i } = x _ { i - 1 }$ th<sub>en</sub>   
5: if $i < n \land x _ { i } + 1 < x _ { i + 1 }$ th<sub>en</sub>   
6: PLA.addPoint $( x _ { i } + 1 , i )$   
7: <sub>e</sub>l<sub>se</sub>   
8: $\operatorname { P L A . a d d P o i n t } ( x _ { i } , i )$   
9: return PLA

![](images/3c05e80eaed875b93196be6df69ffa517c40ad5a4e6053aee8f72040acfffc49.jpg)  
Fi<sub>gu</sub>r<sub>e</sub> 21: PLA <sub>co</sub>n<sub>s</sub>tr<sub>uc</sub>ti<sub>o</sub>n <sub>w</sub>ith d<sub>up</sub>li<sub>ca</sub>t<sub>e</sub> k<sub>eys</sub> in th<sub>e</sub> PGMi<sub>n</sub>d<sub>ex.</sub> I<sub>ns</sub>t<sub>ea</sub>d <sub>o</sub>f di<sub>rec</sub>tl<sub>y cons</sub>t<sub>ruc</sub>ti<sub>ng</sub> th<sub>e</sub> PLA f<sub>rom</sub> th<sub>e</sub> ori<sub>g</sub>inal ke<sub>y</sub>-rank <sub>p</sub>airs<sub>,</sub> the PGM-index constructs the PLA <sub>over a se</sub>t <sub>o</sub>f <sub>v</sub>i<sub>r</sub>t<sub>ua</sub>l <sub>po</sub>i<sub>n</sub>t<sub>s</sub> i<sub>n</sub>t<sub>ro</sub>d<sub>uce</sub>d t<sub>o</sub> h<sub>an</sub>dl<sub>e</sub> d<sub>up</sub>li<sub>ca</sub>t<sub>es.</sub>

This keeps the construction feasible even when a key occurs more than 2� + 1 times; if the raw data points are used directly, PLA construction becomes infeasible in this case (see Figure 21). At the same time, the resulting PLA can still answer lower-bound and related queries correctly. In particular, $( x _ { i } + 1 , i )$ captures the information immediately after the duplicate run of value $x _ { i } .$ As a result, even with duplicate keys, the PGM-index can properly construct the approximate position and correction range used internally, allowing its final search procedure to return the correct result for lower-bound-style queries.

## E<sub>.</sub>2 P<sub>o</sub>i<sub>son</sub> G<sub>enera</sub>ti<sub>on w</sub>ith D<sub>up</sub>li<sub>ca</sub>t<sub>es</sub>

At a high level, our algorithm for generating poisons in the duplicateallowed setting is the same as the algorithm for the duplicateforbidden setting (Algorithm 1). The only diference is the function DI-Consecutive $( \mathcal { K } _ { \mathrm { s e g } } , \lambda _ { \mathrm { c a n d } } )$ in Algorithm 1, namely, the function that generates � poisons for a given segment $\mathcal { K } _ { \mathrm { s e g } }$

In the duplicate-forbidden setting, we used the discrete-intercept algorithm to eficiently evaluate candidate poison sets consisting of consecutive integers. In the duplicate-allowed setting, we again use the discrete-intercept algorithm, but now we evaluate poison sets consisting of a single integer. This design is motivated by Theorem 2, which shows that, in the duplicate-allowed maximumerror maximization problem, an optimal poison set can consist of a single legitimate key. Motivated by this result, we eficiently evaluate such singleton candidates using the discrete-intercept algorithm and select the poison set that minimizes the number of covered legitimate keys.

T<sub>a</sub>bl<sub>e</sub> 8<sub>:</sub> R<sub>esu</sub>lt<sub>s</sub> f<sub>or</sub> $m _ { \mathrm { o p t } }$ after du<sub>p</sub>licate-allowed <sub>p</sub>oisonin<sub>g</sub>. PGM-attack increases $m _ { \mathrm { o p t } }$ relative to legitimate ke<sub>y</sub>s (b<sub>y</sub> u<sub>p</sub> to 4.68× at � = 0.01� and up to 36.1× at � = 0.1�), whereas Random and Random-Adj. barely change $m _ { \mathrm { o p t } }$ (by at most 1.24× and 1.30×, respectively). The amplification factor for PGM-attack tends to grow with �.
<table><tr><td></td><td></td><td colspan="4"></td><td colspan="3"> $\lambda = 0 . 1 n$ </td></tr><tr><td>ε</td><td>Dataset</td><td>Original</td><td>PGM-ATTACK</td><td>RANDOM</td><td>RANDOM-ADJ.</td><td>PGM-ATTACK</td><td>RANDOM</td><td>RANDOM-ADJ.</td></tr><tr><td>16</td><td>Weblogs</td><td>291K</td><td>340K (1.17×)</td><td>292K (1.00×)</td><td>295K (1.01×)</td><td>583K (2.00×)</td><td>297K (1.02×)</td><td>331K (1.14×)</td></tr><tr><td></td><td>IoT</td><td>178K</td><td>225K (1.27×)</td><td>179K (1.01×)</td><td>181K (1.02×)</td><td>526K (2.96×)</td><td>186K (1.05×)</td><td>206K (1.16×)</td></tr><tr><td></td><td>Wiki</td><td>228K</td><td>334K (1.46×)</td><td>230K (1.01×)</td><td>233K (1.02×)</td><td>937K (4.11×)</td><td>245K (1.07×)</td><td>280K (1.23×)</td></tr><tr><td></td><td>Zipf</td><td>83.4K</td><td>151K (1.81×)</td><td>85.4K (1.02×)</td><td>85.5K (1.03×)</td><td>668K (8.01×)</td><td>104K (1.24×)</td><td>105K (1.26×)</td></tr><tr><td>32</td><td>Weblogs</td><td>122K</td><td>145K (1.19×)</td><td>123K (1.00×)</td><td>124K (1.01×)</td><td>267K (2.18×)</td><td>124K (1.01×)</td><td>139K (1.14×)</td></tr><tr><td></td><td>IoT</td><td>78.3K</td><td>101K (1.29×)</td><td>78.5K (1.00×)</td><td>79.3K (1.01×)</td><td>253K (3.23×)</td><td>80.6K (1.03×)</td><td>89.6K (1.14×)</td></tr><tr><td></td><td>Wiki</td><td>85.3K</td><td>130K (1.53×)</td><td>85.8K (1.01×)</td><td>86.8K (1.02×)</td><td>414K (4.86×)</td><td>89.4K (1.05×)</td><td>101K (1.18×)</td></tr><tr><td></td><td>Zipf</td><td>24.9K</td><td>58.1K (2.33×)</td><td>25.5K (1.02×)</td><td>25.6K (1.03×)</td><td>326K (13.1×)</td><td>30.2K (1.21×)</td><td>31.8K (1.28×)</td></tr><tr><td>64</td><td>Weblogs</td><td>49.3K</td><td>59.8K (1.21×)</td><td>49.4K (1.00×)</td><td>50.1K (1.01×)</td><td>119K (2.41×)</td><td>49.8K (1.01×)</td><td>56.2K (1.14×)</td></tr><tr><td></td><td>IoT</td><td>36.3K</td><td>47.2K (1.30×)</td><td>36.4K (1.00×)</td><td>36.8K (1.01×)</td><td>123K (3.39×)</td><td>36.9K (1.02×)</td><td>41K (1.13×)</td></tr><tr><td></td><td>Wiki</td><td>37.5K</td><td>56.9K (1.52×)</td><td>37.7K (1.01×)</td><td>38K (1.01×)</td><td>194K (5.16×)</td><td>38.7K (1.03×)</td><td>42.4K (1.13×)</td></tr><tr><td></td><td>Zipf</td><td>7.24K</td><td>23.7K (3.27×)</td><td>7.34K (1.01×)</td><td>7.44K (1.03×)</td><td>160K (22.1×)</td><td>8.58K (1.19×)</td><td>9.32K (1.29×)</td></tr><tr><td>128</td><td>Weblogs</td><td>20K</td><td>25K (1.25×)</td><td>20K (1.00×)</td><td>20.3K (1.01×)</td><td>53.4K (2.67×)</td><td>20.2K (1.01×)</td><td>22.7K (1.14×)</td></tr><tr><td></td><td>IoT</td><td>16.3K</td><td>22.2K (1.37×)</td><td>16.3K (1.00×)</td><td>16.5K (1.02×)</td><td>60.4K (3.72×)</td><td>16.5K (1.01×)</td><td>18.5K (1.14×)</td></tr><tr><td></td><td>Wiki</td><td>19.2K</td><td>28K (1.46×)</td><td>19.3K (1.01×)</td><td>19.4K (1.01×)</td><td>94.7K (4.93×)</td><td>19.5K (1.02×)</td><td>21.1K (1.10×)</td></tr><tr><td></td><td>Zipf</td><td>2.18K</td><td>10.2K (4.68×)</td><td>2.2K (1.01×)</td><td>2.25K (1.03×)</td><td>78.7K (36.1×)</td><td>2.52K (1.16×)</td><td>2.83K (1.30×)</td></tr></table>

Note that, when selecting this single integer and evaluating the poison set, we use virtual points rather than raw data points. Specifically, for each key appearing among the virtual points introduced in Algorithm 3, we form a poison set consisting of � copies of that key and evaluate it. As in Appendix B, each candidate can be evaluated in O (|I| log �) time, where � is the number of legitimate keys currently coverable by one segment. Since the number of candidates is O (�), the total time complexity is O (|I |� log �). By using virtual points in this way, the poison-generation algorithm partially reflects the behavior of the PGM-index PLA construction algorithm.

## E.3 Ex<sub>p</sub>erimental E<sub>v</sub>aluation

Here, we evaluate how much PGM-attack can increase the number of segments when duplicates are allowed.

Datasets. To evaluate the efect of PGM-attack in the duplicateallowed setting, we use three real-world datasets containing dupli cates and one synthetic dataset with duplicates:

• From SOSD [29], we use the Wiki dataset. It contains 200.0M keys in total, of which 90.0M are unique.

• Following the PGM-index paper [16] and subsequent work, we use the Weblogs and IoT datasets. However, the MgBench dataset [1] used in that line of work is no longer publicly avail able. We therefore use the mgbench dataset released by Andrew Crotty [9]. In this benchmark, bench2 corresponds to Weblogs and bench3 to IoT. They contain 75.7M and 109.0M keys, respectively, and the numbers of unique keys are 18.7M and 108.9M.

• As a synthetic dataset, we generate a Zipf dataset. It contains 200.0M keys in total, of which 34.8M are unique.

Methods. We compare PGM-attack with Random and Random-Adj. Note, however, that these random baselines difer slightly from those in the duplicate-forbidden setting, although the underlying policy is analogous.

For Random, in the duplicate-forbidden setting we sampled uniformly from the allowed integers, i.e., integers in the key range $[ k _ { 1 } , k _ { n } ]$ that do not belong to the legitimate keys. In the duplicateallowed setting, by contrast, every integer in the key range is allowed, so we simply sample � integers uniformly at random from that key range.

For Random-Adj., in the duplicate-forbidden setting we sampled uniformly from integers adjacent to legitimate keys. In the duplicateallowed setting, since integers already contained in the legitimate keys are also allowed, we instead sample � integers uniformly at random, with replacement, from the legitimate keys.

Results. Table 8 reports how much PGM-attack increases the number of segments in the duplicate-allowed setting.

For every dataset, PGM-attack increases $m _ { \mathrm { o p t } }$ relative to the legitimate-key baseline: by up to 4.68× at 1% poisoning $( \lambda = 0 . 0 1 n )$ and up to 36.1× at 10% poisoning $( \lambda = 0 . 1 n )$ . In contrast, Random and Random-Adj. barely change �<sub>opt</sub>, by at most 1.24× and 1.30×, respectively. These results show that, even when duplicates are allowed, PGM-attack yields larger increases in �<sub>opt</sub> than random insertions.

We also observe, as in the duplicate-forbidden setting, that the amplification factor of PGM-attack tends to grow with �. For example, when $\lambda = 0 . 1 n ,$ , the amplification factor is at most 8.01× for $\varepsilon = 1 6 ,$ , but rises to 36.1× for � = 128. As discussed in Section 6.4, this trend can be understood intuitively in the same way as in the duplicate-forbidden setting.