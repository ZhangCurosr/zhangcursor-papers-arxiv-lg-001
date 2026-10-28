# On the VC Dimension of Autoregressive Linear Threshold Functions

Ilan Doron-Arad MIT

Idan Mehalel The Hebrew University

Elchanan Mossel

MIT

## Abstract

Linear threshold functions (LTFs) are a canonical class of Boolean predictors. In the autoregressive setting, the same LTF reads the most recent d bits, generates the next bit, and returns the bit produced after T steps. While the one-step class has VC dimension $d ,$ for general T the counting upper bound is $O ( d ^ { 2 } )$ . We prove that the VC dimension is $\Theta ( d ^ { 2 } )$ for every $T \geq d ^ { 2 }$ , resolving the quadratic gap for deterministic autoregressive LTFs. We then adapt the same technique to show an exponential separation between deterministic and stochastic autoregressive models: replacing the sign of LTFs by logistic sampling can increase the VC dimension of the final majority prediction class from quadratic to exponential in d.

## 1 Introduction

Understanding recursive rules is central in mathematics, connecting number theory, probability, and dynamical systems. Known examples include the 3n + 1 problem [1] and random Fibonacci sequences [2], illustrating how an elementary update rule yields notoriously dificult mathematical questions.

In recent years, understanding recursive formulas has become important in large language models (LLMs). As an abstraction, an autoregressive model repeatedly applies a fixed next-token generator to the current sequence and appends the generated token. Intermediate tokens can serve as computational steps [3, 4], increasing the computational power of autoregressive models [5, 6]. To study the statistical cost of this process, Joshi et al. [7] introduced a learning framework in which a fixed generator is applied repeatedly for T steps and the last token is the answer. Training reveals either only the prompt string and the answer bit (end-to-end supervision (e2e)), or reveals all intermediate tokens (chain-of-thought supervision (CoT)). A fundamental question is to understand the statistical complexity of CoT and e2e supervision.

Perhaps the most fundamental test case for autoregressive learning from the work of Joshi et al. [7] onward has been the class of linear threshold functions (LTFs). In the studied learning problem, an LTF outputs the sign of a weighted sum over the Boolean hypercube. In our autoregressive model, the same LTF reads the most recent d bits at every step. Its weights and threshold never change during generation. Formally, for integers $d , T \geq 1$ and $w \in \mathbb { Q } ^ { d }$ , define a autoregressive linear threshold function (LTF) $f _ { w } ^ { T } : \{ 0 , 1 \} ^ { d } \to \{ 0 , 1 \}$ of dimension d with T autoregressive steps as follows. Given $x = ( x _ { 1 - d } , \ldots , x _ { 0 } )$ , set $z _ { t } = x _ { t }$ for $1 - d \leq t \leq 0$ and

$$
z _ { t } = \mathbf { 1 } \left\{ \sum _ { \ell = 1 } ^ { d } w _ { \ell } z _ { t - \ell } \geq 1 \right\} \quad ( 1 \leq t \leq T ) .\tag{1}
$$

Here 1{A} is the indicator function, equal to 1 when the Boolean expression A is true and 0 otherwise. Let $f _ { w } ^ { T } ( x ) = z _ { T }$ and let $\mathcal { F } _ { d , T } = \{ f _ { w } ^ { T } : w \in \mathbb { Q } ^ { d } \}$ be the class of all autoregressive LTFs with rational weights. Standard LTFs are obtained from this class by setting T = 1.

For this class, a large gap remained in the VC dimension, which measures how many inputs can receive arbitrary labels and controls sample complexity in distribution-free learning [10]. Formally, a set $A \subseteq \{ 0 , 1 \} ^ { d }$ is shattered by $\mathcal { F } _ { d , T }$ if, for every map $\lambda : A  \{ 0 , 1 \}$ , there exists $f \in \mathcal { F } _ { d , T }$ with $f ( x ) = \lambda ( x )$ for all $x \in A$ The VC dimension is the maximum size of a set shattered by $\mathcal { F } _ { d , T }$ . For $T = 1$ , it is well known that VCdim $( \mathcal { F } _ { d , 1 } ) = d$ from [11], even though the number of LTFs over the boolean cube is $2 ^ { \Theta ( d ^ { 2 } ) }$ [12], and the VC dimension of a class may grow logarithmically with its size. Since every LTF uniquely determines its autoregressive trajectory, the counting bound $\mathrm { V C d i m } ( \mathcal { F } _ { d , T } ) = O ( d ^ { 2 } )$ holds for every T. As for lower bounds, [14] proved that $\mathrm { V C d i m } ( \mathcal { F } _ { d , T } ) = \Omega ( d )$ for suficiently large T. However, whether the VC dimension is of order $d , d ^ { 2 }$ , or anything in between remained open, and was asked explicitly in [7, 13, 14]. Thus, it remained unclear how autoregressive steps afect the VC dimension of LTFs which directly implies the sample complexity with e2e supervision. In contrast, it is known that with CoT supervision the sample complexity depends linearly on d. The upper bound was proved in [13, 15], and the lower bound in [14].

Our results. Our first main result is that, for $T \geq d ^ { 2 }$ , the VC dimension of autoregressive LTFs is $\Omega ( d ^ { 2 } )$ , substantially larger than the Θ(d) VC dimension of the base (non-autoregressive) LTF class.

Theorem 1. For all integers d, $T \geq 1$ with $T \geq d ^ { 2 } , \mathrm { V C d i m } ( \mathcal { F } _ { d , T } ) \geq d ^ { 2 } / 3 0 2 4 ^ { 2 }$

The weights used in the proof are rational numbers of polynomial bit length. Thus, a single LTF, reused without changing its parameters, can increase VC dimension from linear to quadratic. In particular, at fixed accuracy and confidence, e2e learning requires $\Theta ( d ^ { 2 } )$ examples, whereas the analogue CoT-supervision bound is $\Theta ( d )$ . This resolves the question of statistical separation left open in [7].

Implication for stochastic autoregressive learning. Our second result concerns stochastic autoregressive learning, replacing deterministic generation by a distribution of the next token at each step. We consider this as the canonical study case for the stochastic setting [17]. Let $\begin{array} { r } { \sigma ( u ) = \frac { 1 } { 1 + e ^ { - u } } } \end{array}$ be the sigmoid (or logistic) function. Here, we replace the threshold in (1) by an afine logistic probability: for $w \in \mathbb { R } ^ { d } , b \in \mathbb { R }$ , and a prompt $x = ( x _ { 1 - d } , \ldots , x _ { 0 } )$ , initialize $Z _ { t } = x _ { t }$ for $1 - d \leq t \leq 0$ and sample the next token $Z _ { t }$ by the following distribution:

$$
\Pr ( Z _ { t } = 1 \mid Z _ { 1 - d } , \ldots , Z _ { t - 1 } ) = \sigma \left( b + \sum _ { \ell = 1 } ^ { d } w _ { \ell } Z _ { t - \ell } \right) , \qquad ( 1 \leq t \leq T ) .\tag{2}
$$

Define the final probability by $q _ { w , b } ^ { ( T ) } ( x ) = \operatorname* { P r } ( Z _ { T } = 1 \mid x )$ and the majority class by

$$
\mathcal { H } _ { d , T } = \left\{ x \mapsto \mathbf { 1 } \left\{ q _ { w , b } ^ { ( T ) } ( x ) > 1 / 2 \right\} : w \in \mathbb { R } ^ { d } , \ b \in \mathbb { R } \right\} .\tag{3}
$$

Thus, $\mathcal { H } _ { d , T }$ consists of the deterministic majority predictions of the stochastic generators. We prove that even in this simple stochastic setting shattering exponential many points is possible, though with small margin.

Theorem 2. There are constants $c , C > 0$ and $d _ { 0 }$ such that for every $d \geq d _ { 0 }$ there is an integer $T \leq 2 ^ { 2 ^ { C d } }$ for which VCdim $( \mathcal { H } _ { d , T } ) \geq 2 ^ { c d }$

Recall that the pseudo-dimension of a real valued function class $\mathcal { Q }$ is the largest n for which there exist inputs $x _ { 1 } , \ldots , x _ { n }$ and fixed thresholds $r _ { 1 } , \ldots , r _ { n }$ such that, for every labeling $\lambda \in \{ 0 , 1 \} ^ { n }$ , some $q \in \mathcal { Q }$ satisfies ${ \bf 1 } \{ q ( x _ { i } ) > r _ { i } \} = \lambda _ { i }$ for all $i \in [ n ]$ . In particular, Theorem 2 shows that the class $\{ q _ { w , b } ^ { ( T ) } : w \in \mathbb { R } ^ { d } , \ b \in \mathbb { R } \}$ has pseudo-dimension $2 ^ { \Omega ( d ) }$ for some $T ,$ , even with all thresholds fixed at $r _ { i } = 1 / 2$ , in stark contrast to the $O ( d ^ { \bar { 2 } } )$ bound in the deterministic setting. Previous work show qualitative diferences between deterministic and stochastic autoregressive learning [14, 17], but the separating constructions relied on specially designed next-token distributions. In contrast, our exponential separation holds even for the canonical logistic model, where each next token probability is a monotone function of an afine linear score.

Further related work. Joshi et al. [7] also establish a computational separation between e2e and CoT supervision: autoregressive LTFs are eficiently learnable with CoT supervision, whereas e2e learning is hard under cryptographic assumptions. Magen, Joshi, and Vardi [19] show that a single autoregressive threshold neuron can simulate threshold circuits with polynomial overhead, after a fixed encoding of the inputs. Joshi et al. [16] study learning from multiple thinkers and show that CoT supervision from even a small number of thinkers can make learning computationally hard. The autoregressive learning framework has also been extended to online prediction [14] and stochastic next-token generation [17]. Li [15] obtains optimal realizable PAC bounds for learning complete CoT trajectories over multiclass alphabets, independently of trajectory length.

## 1.1 Overview of the techniques

## Theorem 1: deterministic generators

Our goal is to shatter $\Theta ( d ^ { 2 } )$ points by a class of autoregressive LTFs. We arrange the labels we wish to realize by this class as an m × m binary matrix λ, where $m = \Theta ( d )$ . Each input will specify one row i and one entry s in that row. We encode row i as the intege

$$
\theta _ { i } = \sum _ { s = 1 } ^ { m } 2 ^ { s - 1 } \lambda _ { i , s } .
$$

Thus, m numbers, one number for each row, contain the information on all $m ^ { 2 }$ labels. We place these numbers in selected weights and construct fixed inputs whose final outputs are the requested binary digits. Encoding labels in weights and extracting their digits is a familiar approach in neural network VC lower bounds [18]. Here, however, the challenge is to perform the entire extraction with a single fixed LTF, while storing the current state of the recursion in only $O ( m ) = O ( d )$ previously generated bits (the memory). We accomplish this by first constructing a recursive formula shattering suficiently many points with several LTFs per round. Then, we explain how to reduce this formula to a single LTF, and finally, explain how to choose the input points independent of the labels.

Shattering with several LTFs. We recover the entries of a row using several LTFs, allowing first $O ( m )$ LTF computations in each autoregressive round t. The recursion keeps four groups of bits: row bits used to select the row i corresponding to the number $\theta _ { i } ;$ One test bit computing a linear threshold test; output bits storing the digits already recovered, and pointer bits selecting the next digit to be discovered. Starting with zero output, we recover the digits from largest to smallest. To recover digit s, the test checks whether $\theta _ { i }$ minus the value represented by the current output is at least $2 ^ { s - 1 }$ . Its answer is exactly the label $\lambda _ { i , s }$ The corresponding output bit is then updated and the pointer moves to the next digit; the other output bits and the selected row remain unchanged. Every operation is an LTF computation, including the conditional update of an output bit. After $O ( m )$ rounds, all entries of the selected row have been recovered and remain stored.

Implementing each round with one LTF. We encode the entire state of round t of the above recursion in a block $E ^ { t }$ of $O ( m )$ bits. Each block contains a fixed initial 1 marking the beginning of a block, designated positions for the four groups of bits (row, test, output, pointer), and zeros elsewhere. The same LTF generates the next block $E ^ { t + 1 }$ based on the current block $E ^ { t }$ , one bit at a time. Within each new block $E ^ { t + 1 }$ , the test bit is generated before the output bits, allowing the output bits to depend on both the new test bit and the previous block $E ^ { t }$ . To implement all updates with the same LTF, we carefully choose the positions of the bits within each block. Using modular arithmetic, we ensure that coeficients intended for one update do not interfere with others. Each test, output, and pointer bit is stored in two consecutive positions. The LTF computes the bit at the first position, while at the second bit, pairs of opposite weights cancel the contributions of earlier bits, allowing the LTF to copy the bit just generated. Consequently, one fixed LTF transforms $E ^ { t }$ into $E ^ { t + 1 }$ using $O ( m )$ generated bits per round. The full procedure requires $\Theta ( m )$ blocks and therefore takes $\Theta ( m ^ { 2 } ) = \Theta ( d ^ { 2 } )$ autoregressive steps.

Choosing the inputs. Recovering a row is not yet enough: all $m ^ { 2 }$ inputs must be fixed before the labels are chosen, and every requested entry must be returned at the same time $T .$ For each row, concatenate enough initial zeros with its first two blocks $E ^ { 0 } , E ^ { 1 }$ , both of which are independent of the labels. Choose the input for entry s to be a window of d consecutive bits in this string, shifted by an amount $r _ { s }$ . We choose the shift $r _ { s }$ so that time $T$ falls at the output position storing entry s of a later block. Once the required digit is extracted, the output bits remain fixed across blocks, so this works for every suficiently large $T .$ . The resulting $m ^ { 2 }$ inputs realize the arbitrary matrix λ under one common weight vector, proving the quadratic lower bound. As a sanity check, considering the $O ( d ^ { 2 } )$ upper bound, note that our construction indeed cannot shatter more than $O ( d ^ { 2 } )$ inputs, since selecting among m rows and storing their m recovered digits requires $\Theta ( m )$ bits in the d-bit history, limiting m to $O ( d )$

## Theorem 2: stochastic generators

Unlike the deterministic construction, we encode exponentially many labels in a single real number $\alpha \in ( 0 , 1 )$ using its base-4 digits. Each m-bit input selects one of $N = 2 ^ { m }$ digits. To recover the required bit, we repeatedly sample a Bernoulli bit whose odds are proportional to $4 ^ { j } \alpha$ , where j is the selected digit’s index. These odds are implemented by a logistic gate with an afine score. The key observation is that, although the sampling probability is monotone, the probability that the number of sampled ones is 0 or 1 modulo 4 oscillates around $1 / 2$ . With suficiently many samples, the sign of this diference reveals the selected digit, and hence its label. We implement this experiment with a single logistic generator by adapting the deterministic block construction, using $O ( m )$ memory to shatter $2 ^ { m }$ inputs.

Organization. Section 2 gives the recursion using several LTFs, Section 3 implements it with one LTF, and Section 4 chooses the inputs and proves Theorem 1. Sections 5–5.2 give the stochastic digit reader, implement it with one logistic generator, and prove Theorem 2.

## 2 Shattering with Several LTFs

In this section, we shatter $\Theta ( m ^ { 2 } ) = \Theta ( d ^ { 2 } )$ inputs using $O ( m )$ LTFs updated in each round t. Then, in the next section we show how to reduce them to a single more involved LTF. Fix $m \geq 2$ and an arbitrary labeling $\lambda \in \{ 0 , 1 \} ^ { m \times m }$ . Write $[ m ] = \{ 1 , \dots , m \}$ . Define the encoding of row i by

$$
\theta _ { i } = \sum _ { j = 1 } ^ { m } 2 ^ { j - 1 } \lambda _ { i , j } \qquad ( i \in [ m ] ) .
$$

Let $\theta = ( \theta _ { 1 } , \ldots , \theta _ { m } )$ be the encoding of λ. We will compute the binary digits of $\theta _ { i }$ from the largest to the smallest using a recursive formula. For the moment, we allow several simple LTF computations in each round t. Specifically, for each selected row $i \in [ m ]$ and every $t \geq 0$ , we define $3 m + 2$ bits $f _ { \theta , i } ^ { t } \in \{ 0 , 1 \} ^ { 3 m + 2 }$ interpreted as follows, by the displayed order.

$f _ { \theta , i , \mathrm { r o w } , j } ^ { t }$ for $j \in [ m ]$ . These row bits select the encoded row.

$f _ { \theta , i , \mathrm { t e s t } } ^ { t }$ . This test bit compares the encoded row with the current output and pointer values.

$f _ { \theta , i , \mathrm { o u t } , j } ^ { t }$ for $j \in [ m ]$ . These output bits store the recovered digits of the selected row.

$f _ { \theta , i , \mathrm { p t r } , j } ^ { t }$ for $j \in [ m + 1 ]$ ]. These pointer bits will specify the next digit.

We now define $f _ { \theta , i } ^ { t } \in \{ 0 , 1 \} ^ { 3 m + 2 }$ formally. For each $i \in [ m ]$ , initialize the values at $t = 0$ by

$$
f _ { \theta , i , \mathrm { r o w } , j } ^ { 0 } = \mathbf { 1 } \{ j = i \} , \quad f _ { \theta , i , \mathrm { t e s t } } ^ { 0 } = f _ { \theta , i , \mathrm { p t r } , m + 1 } ^ { 0 } = 1 , \quad f _ { \theta , i , \mathrm { o u t } , j } ^ { 0 } = f _ { \theta , i , \mathrm { p t r } , j } ^ { 0 } = 0 \quad ( j \in [ m ] ) .\tag{4}
$$

For each $i \in [ m ]$ and $t \geq 0$ , define $f _ { \theta , i } ^ { t + 1 }$ from $f _ { \theta , i } ^ { t }$ by

$$
f _ { \theta , i , \mathrm { r o w } , j } ^ { t + 1 } = f _ { \theta , i , \mathrm { r o w } , j } ^ { t }
$$

$$
( j \in [ m ] ) ,\tag{5}
$$

$$
f _ { \theta , i , \mathrm { t e s t } } ^ { t + 1 } = { \bf 1 } \left\{ \sum _ { j = 1 } ^ { m } \theta _ { j } f _ { \theta , i , \mathrm { r o w } , j } ^ { t } - \sum _ { j = 1 } ^ { m } 2 ^ { j - 1 } \big ( f _ { \theta , i , \mathrm { o u t } , j } ^ { t } + f _ { \theta , i , \mathrm { p t r } , j } ^ { t } \big ) \geq 0 \right\} ,\tag{6}
$$

$$
f _ { \theta , i , \mathrm { o u t } , j } ^ { t + 1 } = { \bf 1 } \left\{ 2 f _ { \theta , i , \mathrm { o u t } , j } ^ { t } + f _ { \theta , i , \mathrm { p t r } , j } ^ { t } + f _ { \theta , i , \mathrm { t e s t } } ^ { t + 1 } \geq 2 \right\}
$$

$$
( j \in [ m ] ) ,\tag{7}
$$

$$
f _ { \theta , i , \mathrm { p t r } , j } ^ { t + 1 } = f _ { \theta , i , \mathrm { p t r } , j + 1 } ^ { t }\tag{8}
$$

$$
( j \in [ m ] ) ,
$$

$$
f _ { \theta , i , \mathrm { p t r } , m + 1 } ^ { t + 1 } = 0 .\tag{9}
$$

The test bit $f _ { \theta , i , \mathrm { t e s t } } ^ { t + 1 }$ is evaluated before the output bits $f _ { \theta , i , \mathrm { o u t } , j } ^ { t + 1 } ;$ the output bits can then be evaluated in parallel. Note that the above equations can all be implemented as LTFs. In particular, copying a bit and producing the constant zero are LTF operations. Thus, these equations use only LTF computations. An illustration is given in Figure 1. The next lemma shows that this recursion recovers every entry of the arbitrary labeling λ.

![](images/7c5e61101c2093138b36915602e0108f7364a2c0862e5ab6adc7a33bf4e48bed.jpg)  
Figure 1: One round of the recursion for $m = 5 ,$ , selected row $i = 2 ,$ , and encoding $\theta _ { 2 } = 2 1$ , from $t = 3$ to $t + 1 = 4$ . Within each region (Row, Test, Output, Pointer), coordinates are ordered by increasing $j .$ The row bits remain unchanged. The current output bits encode 16, and the pointer selects position 3, whose weight is $2 ^ { 3 - 1 } = 4$ . Since $2 1 - 1 6 \geq 4$ , the new test bit is 1, so the third output bit is set to 1, while all other output bits remain unchanged. Finally, the pointer shifts from position 3 to position 2, selecting the next digit to test.

Lemma 1. For every labeling $\lambda \in \{ 0 , 1 \} ^ { m \times m }$ with encoding $\theta ,$ we have $f _ { \theta , i , \mathrm { o u t } , j } ^ { t } = \lambda _ { i , j }$ for all $i , j \in [ m ]$ and $t \geq m + 1$ . Moreover, for every labeling $\lambda ^ { \prime } \in \{ 0 , 1 \} ^ { m \times m }$ with encoding $\theta ^ { \prime }$ , we have $f _ { \theta , i } ^ { t } = f _ { \theta ^ { \prime } , i } ^ { t }$ for all $i \in [ m ]$ and $t \in \{ 0 , 1 \}$

Proof. Fix $i \in [ m ]$ . We prove the two assertions separately starting from the second. By (4) and (5), $f _ { \theta , i , \mathrm { r o w } , j } ^ { t } = \mathbf { 1 } \{ j = i \}$ for all $t \geq 0$ and $j \in [ m ]$ . Equations (6)–(9) give

$$
\begin{array} { r l r l } & { f _ { \theta , i , \mathrm { t e s t } } ^ { 1 } = \mathbf { 1 } \{ \theta _ { i } \geq 0 \} = 1 , } & & { } \\ & { f _ { \theta , i , \mathrm { o u t } , j } ^ { 1 } = 0 } & & { ( j \in [ m ] ) , } \\ & { f _ { \theta , i , \mathrm { p t r } , j } ^ { 1 } = \mathbf { 1 } \{ j = m \} } & & { ( j \in [ m + 1 ] ) . } \end{array}\tag{10}
$$

Thus, all coordinates of $f _ { \theta , i } ^ { 0 }$ and $f _ { \theta , i } ^ { 1 }$ are determined by i alone, independently of λ.

We now prove the first claim. First, (4) and (8)–(9) give, by induction on $t ,$

$$
f _ { \theta , i , \mathrm { p t r } , j } ^ { t } = { \bf 1 } \{ t = m + 1 - j \} \qquad ( j \in [ m + 1 ] , \ t \geq 0 ) .\tag{11}
$$

Indeed, the formula holds at $t = 0 ;$ for $j \in [ m ] , f _ { \theta , i , \mathrm { p t r } , j } ^ { t + 1 } = f _ { \theta , i , \mathrm { p t r } , j + 1 } ^ { t } = \mathbf { 1 } \{ t = m - j \}$ , while $f _ { \theta , i , \mathrm { p t r } , m + 1 } ^ { t } = 0$ for $t \geq 1$ . We prove by descending induction on $k = m , \ldots , 1$ that, for $t = m - k + 2$ it holds that

$$
f _ { \theta , i , \mathrm { o u t } , j } ^ { t - 1 } = \left\{ \begin{array} { l l } { \lambda _ { i , j } , } & { j > k , } \\ { 0 , } & { j \le k , } \end{array} \right.\tag{12}
$$

and that round t satisfies $f _ { \theta , i , \mathrm { o u t } , k } ^ { t } = \lambda _ { i , k }$ and $f _ { \theta , i , \mathrm { o u t } , j } ^ { t } = f _ { \theta , i , \mathrm { o u t } , j } ^ { t - 1 }$ for every $j \neq k$

Base case $( k = m )$ . Here $t = 2 .$ , and (10) gives $f _ { \theta , i , \mathrm { o u t } , j } ^ { 1 } = 0$ for every $j \in [ m ]$ , as required by (12).

Induction step. Assume (12) for k. By (11), the only nonzero pointer bit at time t − 1 is at position k. Hence,

$$
\sum _ { j = 1 } ^ { m } 2 ^ { j - 1 } \big ( f _ { \theta , i , \mathrm { o u t } , j } ^ { t - 1 } + f _ { \theta , i , \mathrm { p t r } , j } ^ { t - 1 } \big ) = 2 ^ { k - 1 } + \sum _ { j = k + 1 } ^ { m } 2 ^ { j - 1 } \lambda _ { i , j } .
$$

Substituting into (6) gives

$$
f _ { \theta , i , \mathrm { t e s t } } ^ { t } = \mathbf { 1 } \left\{ 2 ^ { k - 1 } ( \lambda _ { i , k } - 1 ) + \sum _ { j = 1 } ^ { k - 1 } 2 ^ { j - 1 } \lambda _ { i , j } \geq 0 \right\} = \lambda _ { i , k } ,
$$

where the last equality follows since $\textstyle \sum _ { j = 1 } ^ { k - 1 } 2 ^ { j - 1 } \lambda _ { i , j } < 2 ^ { k - 1 }$ . Therefore, by (7),

$$
f _ { \theta , i , \mathrm { o u t } , k } ^ { t } = \mathbf { 1 } \{ 1 + f _ { \theta , i , \mathrm { t e s t } } ^ { t } \geq 2 \} = \lambda _ { i , k } .
$$

For every $j \neq k$ , (11) gives $f _ { \theta , i , \mathrm { p t r } , j } ^ { t - 1 } = 0$ , so

$$
f _ { \theta , i , \mathrm { o u t } , j } ^ { t } = \mathbf { 1 } \{ 2 f _ { \theta , i , \mathrm { o u t } , j } ^ { t - 1 } + f _ { \theta , i , \mathrm { t e s t } } ^ { t } \geq 2 \} = f _ { \theta , i , \mathrm { o u t } , j } ^ { t - 1 } .
$$

For $k > 1$ , this establishes (12) for $k - 1$ , completing the induction. At time $m + 1$ , all output bits equal their labels. By (11), $f _ { \theta , i , \mathrm { p t r } , j } ^ { t } = \mathbf { 1 } \{ t = m + 1 - j \} = 0$ for all $j \in [ m + 1 ]$ and $t \geq m + 1$ . Thus, (7) gives $f _ { \theta , i , \mathrm { o u t } , j } ^ { t + 1 } = f _ { \theta , i , \mathrm { o u t } , j } ^ { t }$ for all $j \in [ m ]$ and $t \geq m + 1$ . Hence, $f _ { \theta , i , \mathrm { o u t } , j } ^ { t } = \lambda _ { i , j }$ for every $t \geq m + 1$ , proving the lemma. □

## 3 Implementing the Recursion with One LTF

The recursive formula described in the previous section updates several bits per round, whereas (1) produces one bit per step. To reduce the recursive formula to one LTF, in this section we encode each vector of bits $f _ { \theta , i } ^ { t }$ as a block of p bits, where

$$
p = 3 7 8 ( m + 2 ) .\tag{13}
$$

We choose the coeficients of a single LTF so that the current state, encoded in a block $E _ { \theta , i } ^ { t }$ (defined below), produces $E _ { \theta , i } ^ { t + 1 }$ as its next block. We now define the blocks. For the entire section, fix $\lambda \in \{ 0 , 1 \} ^ { m \times m }$ and its encoding θ. Number the positions of each block by $0 , \ldots , p - 1$ , and define positions within the block by

$$
\begin{array} { r l r l } & { \pi ( \mathrm { t e s t } ) = p / 6 + 3 , } \\ & { \pi ( \mathrm { r o w } , j ) = p / 6 + 6 3 j + 6 } & & { ( j \in [ m ] ) , } \\ & { \pi ( \mathrm { o u t } , j ) = p / 6 + 6 3 j + 1 5 \quad } & & { ( j \in [ m ] ) , } \\ & { \pi ( \mathrm { p t r } , j ) = p / 6 + 6 3 j + 4 5 + 6 ( j \mathrm { m o d } 2 ) } & & { ( j \in [ m + 1 ] ) . } \end{array}\tag{14}
$$

These positions are distinct and lie in $( p / 6 , p / 3 )$ ; the largest is at most $p / 3 - 1 2$ . Each row bit will be stored once at $\pi ( \mathrm { r o w } , j )$ , while each test, output, and pointer bit will be stored twice, at $\pi ( b )$ and $\pi ( b ) + 1$ , as specified in (15). The bit at position 0 is fixed to 1. Now, formally, define the block $E _ { \theta , i } ^ { t } \in \{ 0 , 1 \} ^ { p }$ using the above positions by

![](images/185b69bf6f840485ace818a890daf4ea2c0121e2f7ddce4fc4607d706519452c.jpg)  
Figure 2: Transition from $E ^ { 3 }$ to $E ^ { 4 }$ for $m = 5 , i = 2 .$ , and $\theta _ { 2 } = 2 1$ . The black 1 is the first bit of each block, fixed to 1 by the definition of the encoding in (15). The same LTF generates all bits of $E ^ { 4 }$ from left to right. The row bits remain unchanged. The output bits initially represent 16, and the pointer selects $j = 3 ,$ , with weight $2 ^ { 3 - 1 } = 4$ . Since $2 1 - 1 6 \geq 4 ,$ the new test bit is 1. As the lower arrow indicates, this new test bit sets the third output bit to 1, so the output bits now represent $1 6 + 4 = 2 0$ . The pointer changes to $j = 2$ . Test, output, and pointer bits are each stored twice in adjacent positions. Unmarked positions are zero; spacing is schematic.

$$
\begin{array} { r l } & { E _ { \theta , i , 0 } ^ { t } = 1 , } \\ & { E _ { \theta , i , \pi ( \mathrm { r o w } , j ) } ^ { t } = f _ { \theta , i , \mathrm { r o w } , j } ^ { t } \qquad ( j \in [ m ] ) , } \\ & { E _ { \theta , i , \pi ( \mathrm { t e s t } ) } ^ { t } = E _ { \theta , i , \pi ( \mathrm { t e s t } ) + 1 } ^ { t } = f _ { \theta , i , \mathrm { t e s t } } ^ { t } , } \\ & { E _ { \theta , i , \pi ( \mathrm { o u t } , j ) } ^ { t } = E _ { \theta , i , \pi ( \mathrm { o u t } , j ) + 1 } ^ { t } = f _ { \theta , i , \mathrm { o u t } , j } ^ { t } \qquad ( j \in [ m ] ) , } \\ & { E _ { \theta , i , \pi ( \mathrm { p t r } , j ) } ^ { t } = E _ { \theta , i , \pi ( \mathrm { p t r } , j ) + 1 } ^ { t } = f _ { \theta , i , \mathrm { p t r } , j } ^ { t } \qquad ( j \in [ m + 1 ] ) , } \end{array}\tag{15a}
$$

and set all other positions to zero. In particular, every encoded block ends in zero. We will use the single recurrence

$$
z _ { n } = { \bf 1 } \{ - B + \sum _ { \ell = 1 } ^ { p + 1 } a _ { \ell } z _ { n - \ell } \geq 0 \} .\tag{16}
$$

For $d \geq p + 1$ , this is (1) with $w _ { \ell } = a _ { \ell } / B$ for $\ell \leq p + 1$ and $w _ { \ell } = 0$ otherwise. The next lemma shows that there are coeficients $B , ( a _ { \ell } ) _ { \ell \in [ p + 1 ] }$ that can faithfully update the blocks as defined above. For $i \in [ m ]$ and $t \geq 0$ , the history $( 0 , E _ { \theta , i } ^ { t } )$ denotes the initial values $z _ { - p - 1 } = 0$ and $z _ { - p + r } = E _ { \theta , i , r } ^ { t }$ for $0 \leq r < p$

Lemma 2. For every $\theta \in \{ 0 , \ldots , 2 ^ { m } - 1 \} ^ { m }$ , there are integers $B > 0$ and $a _ { 1 } , \dotsc , a _ { p + 1 }$ such that, for every $i \in [ m ]$ and $t \geq 0$ , the history $( 0 , E _ { \theta , i } ^ { t } )$ generates $E _ { \theta , i } ^ { t + 1 }$ as its next p bits under (16).

Proof. Set $B = 2 ^ { m + 5 } ( m + 2 )$ . We first define an auxiliary weight vector $a ^ { ( 0 ) }$ to be used for defining the final weight vector a. Define integer coeficients $a _ { 1 } ^ { ( 0 ) } , \dots , a _ { p + 1 } ^ { ( 0 ) }$ as follows, for every $j \in [ m ]$ :

$$
a _ { p + \pi ( \mathrm { t e s t } ) - \pi ( \mathrm { r o w } , j ) } ^ { ( 0 ) } = 4 \theta _ { j } ,\tag{17a}
$$

$$
a _ { p + \pi ( \mathrm { t e s t } ) - \pi ( \mathrm { o u t } , j ) } ^ { ( 0 ) } = - 2 ^ { j + 1 } ,\tag{17b}
$$

$$
a _ { p + \pi ( \mathrm { t e s t } ) - \pi ( \mathrm { p t r } , j ) } ^ { ( 0 ) } = - 2 ^ { j + 1 } ,\tag{17c}
$$

$$
a _ { \pi ( \mathrm { o u t } , j ) - \pi ( \mathrm { t e s t } ) } ^ { ( 0 ) } = 1 ,\tag{17d}
$$

$$
a _ { p + \pi ( \mathrm { o u t } , j ) - \pi ( \mathrm { p t r } , j ) } ^ { ( 0 ) } = 1 ,\tag{17e}
$$

$$
a _ { p + \pi ( \mathrm { p t r } , j ) - \pi ( \mathrm { p t r } , j + 1 ) } ^ { ( 0 ) } = 4 \qquad ( j \in [ m ] ) .\tag{17f}
$$

For every $j \in [ m ]$ , also set

$$
\begin{array} { r l } & { a _ { p + \pi ( \mathrm { t e s t } ) - \pi ( \mathrm { o u t } , j ) + 1 } ^ { ( 0 ) } = 2 ^ { j + 1 } , } \\ & { a _ { p + \pi ( \mathrm { t e s t } ) - \pi ( \mathrm { p t r } , j ) + 1 } ^ { ( 0 ) } = 2 ^ { j + 1 } , } \\ & { a _ { \pi ( \mathrm { o u t } , j ) - \pi ( \mathrm { t e s t } ) + 1 } ^ { ( 0 ) } = - 1 , } \\ & { a _ { p + \pi ( \mathrm { o u t } , j ) - \pi ( \mathrm { p t r } , j ) + 1 } ^ { ( 0 ) } = - 1 , } \\ & { a _ { p + \pi ( \mathrm { p t r } , j ) - \pi ( \mathrm { p t r } , j + 1 ) + 1 } ^ { ( 0 ) } = - 4 . } \end{array}\tag{18}
$$

Set all other coeficients $a _ { \ell } ^ { ( 0 ) }$ for $\ell \in [ p + 1 ]$ not defined in (17) and (18) to zero. We show that the above assignment is consistent.

Claim 1. For any $\ell \in [ p + 1 ]$ , if (17)–(18) assign both $a _ { \ell } ^ { ( 0 ) } = q$ and $a _ { \ell } ^ { ( 0 ) } = q ^ { \prime }$ , then $q = q ^ { \prime }$

Proof. By (14), all defined indices lie in $( 0 , p / 6 ) \cup ( 5 p / 6 , p )$ , and

$$
( \pi ( { \mathrm { t e s t } } ) , \pi ( { \mathrm { r o w } } , j ) , \pi ( { \mathrm { o u t } } , j ) , \pi ( { \mathrm { p t r } } , j ) ) \equiv ( 3 , 6 , 1 5 , 4 5 + 6 ( j { \mathrm { m o d } } 2 ) ) { \pmod { 6 } } .\tag{19}
$$

For $R = \{ 3 , 6 , 1 5 , 4 5 , 5 1 \}$ , the 20 diferences $u - v$ with $u , v \in R$ and u ̸= v modulo 63 are precisely $\{ 3 , 6 , \ldots , 6 0 \}$ Thus, for $u , v , u ^ { \prime } , v ^ { \prime } \in R$ with u $\neq v$ and $u ^ { \prime } \ne v ^ { \prime }$ it holds that

$$
u - v \equiv u ^ { \prime } - v ^ { \prime } { \pmod { 6 3 } } \implies ( u , v ) = ( u ^ { \prime } , v ^ { \prime } ) .\tag{20}
$$

Since 63 | p, the indices in the six lines of (17) have residues modulo 63 belonging, respectively, to {60}, {51}, {15, 21}, {12}, {27, 33}, and {6, 57}. These sets are pairwise disjoint, so indices from diferent lines cannot coincide. Within (17a)–(17d), equality of indices implies $j = j ^ { \prime }$ by (14). Within (17e) and (17f), the assigned values are 1 and 4, respectively, even when indices repeat:

$$
\pi ( { \mathrm { o u t } } , j ) - \pi ( { \mathrm { p t r } } , j ) = - 3 0 - 6 ( j { \mathrm { ~ m o d } } 2 ) ,\tag{21a}
$$

$$
\pi ( { \mathrm { p t r } } , j ) - \pi ( { \mathrm { p t r } } , j + 1 ) = - 6 9 + 1 2 ( j { \mathrm { ~ m o d ~ } } 2 ) .\tag{21b}
$$

Finally, all indices in (17) are 0 modulo 3. Each equality in (18) is obtained from (17b)–(17f) by adding 1 to the index and negating the value. These indices are 1 modulo 3, so they do not coincide with those in (17) and cannot assign conflicting values either.

We now define the vector of coeficients a based on $a ^ { ( 0 ) }$ . Throughout the remainder of this proof, b ranges over test, $( \mathrm { o u t } , j )$ for $j \in [ m ]$ , and $( \mathrm { p t r } , j )$ for $j \in [ m + 1 ]$ . Start with $a _ { \ell } = a _ { \ell } ^ { ( 0 ) }$ , and for the following coordinates update

$$
\begin{array} { r l } & { a _ { 1 } = 2 , } \\ & { a _ { p - 1 } = 1 - B , } \\ & { a _ { p } = B + 1 , } \\ & { a _ { p + 1 } = - B - 1 , } \\ & { a _ { \pi ( \mathrm { t e s t } ) } = B + 1 , } \\ & { a _ { \pi ( \mathrm { o u t } , j ) } = B - 2 \quad \quad ( j \in [ m ] ) , } \\ & { a _ { \pi ( \mathrm { p t r } , j ) } = B - 3 \quad \quad ( j \in [ m + 1 ] ) , } \\ & { a _ { \pi ( b + 1 } = B - 1 . } \end{array}\tag{22}
$$

and

$$
a _ { \ell } = - B ^ { 2 } \qquad ( p / 3 < \ell \leq 2 p / 3 ) .\tag{23}
$$

These assignments are disjoint from each other and from $( 1 7 ) \mathrm { - } ( 1 8 )$ , by their index ranges and residues modulo 3. Fix $i \in [ m ]$ and $t \geq 0$ . Let

$$
\widehat { E } = ( 0 , E _ { \theta , i } ^ { t } , E _ { \theta , i } ^ { t + 1 } ) ,
$$

indexed by $- p - 1 , \dotsc , p - 1$ , and define the set of positions specified in (15) by

$$
\Pi = \{ 0 \} \cup \{ \pi ( \mathrm { r o w } , j ) : j \in [ m ] \} \cup \bigcup _ { b } \{ \pi ( b ) , \pi ( b ) + 1 \} ,
$$

where b ranges over test, $( \mathrm { o u t } , j )$ for $j \in [ m ]$ , and $( \mathrm { p t r } , j )$ for $j \in [ m + 1 ]$ . Recall that all other positions are zero. For $0 \leq r < p _ { ; }$ , define

$$
\sigma _ { r } = - B + \sum _ { \ell = 1 } ^ { p + 1 } a _ { \ell } \widehat { E } _ { r - \ell } .\tag{24}
$$

Claim 2. $\widehat { E } _ { r } = \mathbf { 1 } \{ \sigma _ { r } \geq 0 \}$ for every $r \in \Pi$

Proof. We first compute the contribution of the coeficients $a _ { \ell } ^ { ( 0 ) }$ to $\sigma _ { r }$ in (24). For $0 \leq r < p ,$ set

$$
\sigma _ { r } ^ { ( 0 ) } = \sum _ { \ell = 1 } ^ { p + 1 } a _ { \ell } ^ { ( 0 ) } \widehat E _ { r - \ell } .\tag{25}
$$

For the index calculations below, pad the vector $a ^ { ( 0 ) }$ with additional zero entries by setting $a _ { \ell } ^ { ( 0 ) } = 0$ for $\ell \in \mathbb { Z } \setminus [ p + 1 ]$ . Equations (21a)–(21b), (20), and (17e)–(17f) give

$$
\begin{array} { r } { a _ { p + \pi ( \mathrm { o u t } , j ) - \pi ( \mathrm { p t r } , k ) } ^ { ( 0 ) } = \mathbf { 1 } \{ k = j \} \qquad ( j \in [ m ] , k \in [ m + 1 ] ) , } \end{array}\tag{26a}
$$

$$
a _ { p + \pi ( \mathrm { p t r } , j ) - \pi ( \mathrm { p t r } , k ) } ^ { ( 0 ) } = 4 \mathbf { 1 } \{ k = j + 1 \} \qquad ( j , k \in [ m + 1 ] ) .\tag{26b}
$$

Also, (15) gives

$$
\widehat { E } _ { r - \ell } = 0 \qquad ( r \in \Pi , \ r \equiv 0 \quad ( \mathrm { m o d \ 3 } ) , \ \ell \in [ p + 1 ] , \ \ell \equiv 1 \quad ( \mathrm { m o d \ 3 } ) ) .\tag{27}
$$

Thus, (18) contributes zero at all positions $\pi ( \mathrm { t e s t } ) , \pi ( \mathrm { r o w } , j ) , \pi ( \mathrm { o u t } , j )$ , and $\pi ( \mathrm { p t r } , j )$ . Substituting (15) into (25), using (20), gives the following identities: (17a)–(17c) give

$$
\sigma _ { \pi ( \mathrm { t e s t } ) } ^ { ( 0 ) } = 4 \sum _ { j = 1 } ^ { m } \theta _ { j } f _ { \theta , i , \mathrm { r o w } , j } ^ { t } - 4 \sum _ { j = 1 } ^ { m } 2 ^ { j - 1 } ( f _ { \theta , i , \mathrm { o u t } , j } ^ { t } + f _ { \theta , i , \mathrm { p t r } , j } ^ { t } ) ;\tag{28}
$$

(17d) and (26a) give

$$
\sigma _ { \pi ( \mathrm { o u t } , j ) } ^ { ( 0 ) } = f _ { \theta , i , \mathrm { t e s t } } ^ { t + 1 } + f _ { \theta , i , \mathrm { p t r } , j } ^ { t } \qquad ( j \in [ m ] ) ;
$$

and (26b) gives

$$
\sigma _ { \pi ( { \mathrm { p t r } } , j ) } ^ { ( 0 ) } = 4 \sum _ { k = 1 } ^ { m + 1 } \mathbf { 1 } \{ k = j + 1 \} f _ { \theta , i , { \mathrm { p t r } } , k } ^ { t } \qquad ( j \in [ m + 1 ] ) .\tag{29}
$$

In particular, $\sigma _ { \pi ( \mathrm { p t r } , m + 1 ) } ^ { ( 0 ) } = 0$

We now show that the contribution of $a ^ { ( 0 ) }$ to $\sigma _ { \pi ( b ) + 1 }$ is zero for every coordinate b stored at both $\pi ( b )$ and $\pi ( b ) + 1$ in (15). For every equality $a _ { \ell } ^ { ( 0 ) } = q$ in (17b)–(17f), (15) and (20) give

$$
\widehat { E } _ { \pi ( b ) + 1 - \ell } = \widehat { E } _ { \pi ( b ) - \ell } .
$$

Since $a _ { \ell + 1 } ^ { ( 0 ) } = - a _ { \ell } ^ { ( 0 ) }$ by (18) it holds that

$$
a _ { \ell } ^ { ( 0 ) } \widehat { E } _ { \pi ( b ) + 1 - \ell } + a _ { \ell + 1 } ^ { ( 0 ) } \widehat { E } _ { \pi ( b ) - \ell } = 0 .
$$

Equation (17a) contributes zero at $\pi ( b ) + 1$ by (15a) and (20). At $r = 0 ,$ , each index ℓ in (17)–(18) satisfies (−ℓ) mod $p \in ( 0 , p / 6 ) \cup ( 5 p / 6 , p )$ , so $\widehat { E } _ { - \ell } = 0$ by (15). At $r = \pi ( \mathrm { r o w } , j )$ , (19)–(20) and (27) give zero for every term. Therefore,

$$
\sigma _ { 0 } ^ { ( 0 ) } = \sigma _ { \pi ( \mathrm { r o w } , j ) } ^ { ( 0 ) } = \sigma _ { \pi ( b ) + 1 } ^ { ( 0 ) } = 0 \qquad ( j \in [ m ] ) .\tag{30}
$$

We next consider (22)–(23). By (14) and (15), for $r \in \Pi$ it holds that

$$
{ \widehat E } _ { r - \pi ( b ) - e } = \mathbf { 1 } \{ r = \pi ( b ) + e \} \qquad ( e \in \{ 0 , 1 \} ) ,
$$

$$
\widehat E _ { r - \ell } = 0 \qquad ( p / 3 < \ell \leq 2 p / 3 ) .\tag{31}
$$

Substituting (28)–(29), (30), and (31) into (24) gives

$$
\sigma _ { 0 } = 1 ,\tag{32a}
$$

$$
\sigma _ { \pi ( \mathrm { r o w } , j ) } = ( B + 1 ) f _ { \theta , i , \mathrm { r o w } , j } ^ { t } - B
$$

$$
( j \in [ m ] ) ,
$$

$$
\sigma _ { \pi ( \mathrm { t e s t } ) } = 1 + 2 f _ { \theta , i , \mathrm { t e s t } } ^ { t } + \sigma _ { \pi ( \mathrm { t e s t } ) } ^ { ( 0 ) } ,\tag{32b}
$$

$$
\sigma _ { \pi ( \mathrm { o u t } , j ) } = 2 f _ { \theta , i , \mathrm { o u t } , j } ^ { t } + f _ { \theta , i , \mathrm { p t r } , j } ^ { t } + f _ { \theta , i , \mathrm { t e s t } } ^ { t + 1 } - 2
$$

$$
( j \in [ m ] ) ,
$$

$$
\sigma _ { \pi ( \mathrm { p t r } , j ) } = 2 f _ { \theta , i , \mathrm { p t r } , j } ^ { t } + 4 f _ { \theta , i , \mathrm { p t r } , j + 1 } ^ { t } - 3
$$

$$
( j \in [ m ] ) ,
$$

$$
\begin{array} { l } { \sigma _ { \pi ( \mathrm { p t r } , m + 1 ) } = 2 f _ { \theta , i , \mathrm { p t r } , m + 1 } ^ { t } - 3 < 0 , } \\ { \sigma _ { \pi ( b ) + 1 } = 2 f _ { \theta , i , b } ^ { t + 1 } - 1 . } \end{array}\tag{32c}
$$

By (28) we have $\sigma _ { \pi ( \mathrm { t e s t } ) } ^ { ( 0 ) } \in 4 \mathbb { Z }$ . Since $1 + 2 f _ { \theta , i , \mathrm { t e s t } } ^ { t } \in \{ 1 , 3 \}$ , (32b) and (6) imply

$$
\mathbf { 1 } \{ \sigma _ { \pi ( { \mathrm { t e s t } } ) } \geq 0 \} = \mathbf { 1 } \{ \sigma _ { \pi ( { \mathrm { t e s t } } ) } ^ { ( 0 ) } \geq 0 \} = f _ { \theta , i , { \mathrm { t e s t } } } ^ { t + 1 } .
$$

Observe that for any bits $u , v$ it holds that $\ 1 \{ 2 u + 4 v - 3 \geq 0 \} = v$ . Thus, (32a)–(32c) give ${ \bf 1 } \{ \sigma _ { 0 } \geq 0 \} = 1$ $\mathbf { 1 } \{ \sigma _ { \pi ( \mathrm { r o w } , j ) } \geq 0 \} = f _ { \theta , i , \mathrm { r o w } , j } ^ { t + 1 }$ , and, for every test, output, or pointer coordinate b we have

$$
\begin{array} { r } { \mathbf { 1 } \{ \sigma _ { \pi ( b ) } \geq 0 \} = \mathbf { 1 } \{ \sigma _ { \pi ( b ) + 1 } \geq 0 \} = f _ { \theta , i , b } ^ { t + 1 } . } \end{array}
$$

Hence, by (15) it holds that $\begin{array} { r } { \mathbf { 1 } \{ \sigma _ { r } \geq 0 \} = \widehat { E } _ { \eta } } \end{array}$ for every $r \in \Pi$ . This proves the claim.

We show that the scores at all remaining positions are negative.

Claim 3. $\sigma _ { r } < 0$ for every $0 \leq r < p$ with $r \not \in \Pi$

Proof. By (17)–(18), for each $j \in [ m ]$ the assigned absolute values are $4 \theta _ { j }$ , four times $2 ^ { j + 1 }$ , four times 1, and twice 4. By Claim 1, repeated indices define the same coeficient, rather than adding their assigned values.

Summing the absolute values of all assignments therefore gives an upper bound:

$$
\begin{array} { l } { { \displaystyle \sum _ { \ell = 1 } ^ { p + 1 } | a _ { \ell } ^ { ( 0 ) } | \le \sum _ { j = 1 } ^ { m } \left( 4 \theta _ { j } + 4 \cdot 2 ^ { j + 1 } + 1 2 \right) } } \\ { { \displaystyle \qquad = 4 \sum _ { j = 1 } ^ { m } \theta _ { j } + 1 6 ( 2 ^ { m } - 1 ) + 1 2 m } } \\ { { \displaystyle \qquad < ( 7 m + 1 6 ) 2 ^ { m } < B / 2 . } } \end{array}
$$

and

$$
\sum _ { \ell = 1 } ^ { p + 1 } \operatorname* { m a x } \{ a _ { \ell } , 0 \} < ( 4 m + 6 ) ( B + 1 ) + B / 2 < B ^ { 2 } .
$$

If $p / 3 < r \leq 2 p / 3$ , then $a _ { r } = - B ^ { 2 }$ and $\widehat { E } _ { 0 } = 1$ 1, so

$$
\sigma _ { r } \leq - B - B ^ { 2 } + \sum _ { \ell = 1 } ^ { p + 1 } \operatorname* { m a x } \{ a _ { \ell } , 0 \} < 0 .
$$

Otherwise,

$$
{ \widehat E } _ { r - p } = { \widehat E } _ { r - \pi ( b ) } = { \widehat E } _ { r - \pi ( b ) - 1 } = 0 .
$$

For the last two equalities, a nonzero entry would require $r = \pi ( b ) + e \in \Pi$ , or $r \in ( p / 3 , 2 p / 3 )$ ; a negative index instead corresponds to a block position greater than $2 p / 3$ , whose entry is zero. Thus,

$$
\sigma _ { r } \le - B + \sum _ { \ell = 1 } ^ { p + 1 } | a _ { \ell } ^ { ( 0 ) } | + 2 < - B + B / 2 + 2 < 0 .
$$

This proves the claim.

By Claims 2 and 3 it holds that $\widehat { E } _ { r } = \mathbf { 1 } \{ \sigma _ { r } \geq 0 \}$ for every $0 \leq r < p .$ . Since $- p - 1 \leq r - \ell < r$ for every $1 \leq \ell \leq p + 1$ , induction on r shows that (16), initialized with $( 0 , E _ { \theta , i } ^ { t } )$ , generates exactly $E _ { \theta , i } ^ { t + 1 }$ . The coeficients depend only on m and θ. □

## 4 Choosing the Inputs

We now construct $m ^ { 2 }$ fixed input strings and prove that, for every assignment of labels to these inputs, a single autoregressive LTF realizes all the labels. For each fixed encoding θ, write $E _ { i } ^ { t } : = E _ { \theta , i } ^ { t }$ . By Lemma 1, the blocks $E _ { i } ^ { 0 } , E _ { i } ^ { 1 }$ depend only on i, not on the labels. We construct inputs $x ^ { i , s }$ by selecting d consecutive bits from this string which is independent of the labels, with initial zero padding as specified below. We choose $r _ { s } \in \{ 0 , \ldots , p - 1 \}$ so that $T + r _ { s } \equiv 1 + \pi ( \mathrm { o u t } , s )$ (mod p). Thus, the bit generated at time T is at position π(out, s) of an encoded block. Assume $d \geq p + 1$ and $T \ge ( m + 3 ) p$ . For each $i \in [ m ]$ , define the string $E _ { i } = ( ( E _ { i } ) _ { n } ) _ { n = 1 - d } ^ { p }$ by

$$
( E _ { i } ) _ { n } = \left\{ { \begin{array} { l l } { 0 , } & { 1 - d \leq n \leq - p , } \\ { ( E _ { i } ^ { 0 } ) _ { n + p - 1 } , } & { 1 - p \leq n \leq 0 , } \\ { ( E _ { i } ^ { 1 } ) _ { n - 1 } , } & { 1 \leq n \leq p . } \end{array} } \right.\tag{33}
$$

Equivalently, $E _ { i } = 0 ^ { d - p } E _ { i } ^ { 0 } E _ { i } ^ { 1 }$ , where $0 ^ { d - p }$ denotes $d - p$ zeros. For each column $s \in [ m ]$ , set

$$
r _ { s } = ( \pi ( \mathrm { o u t } , s ) - T + 1 ) \bmod p ,\tag{34}
$$

and define

$$
x _ { n } ^ { i , s } = ( E _ { i } ) _ { r _ { s } + n } \qquad ( 1 - d \leq n \leq 0 , \ i , s \in [ m ] ) .\tag{35}
$$

All indices lie between $1 - d$ and $p - 1$ , so these $m ^ { 2 }$ prompts are well defined and independent of the labels.

![](images/279981b3e3c76236e6147bce7c269b5bb5c86886e8d0c166b59a40cc10ad85d2.jpg)  
Figure 3: Choosing inputs for a fixed row i. The three rows correspond to columns $s _ { 1 } , s _ { 2 } , s _ { 3 } ;$ each repeats the same string $0 ^ { d - p } E _ { i } ^ { 0 } E _ { i } ^ { 1 }$ on the left. Each blue frame selects d consecutive bits, forming $x ^ { i , s }$ . The short arrows show the shift $r _ { s }$ from the start of the string to the start of each input. The long arrows represent $T$ generated bits. The blue cell is at position $\pi ( \mathrm { o u t } , s )$ in the later block $E _ { i } ^ { t }$ and equals $\lambda _ { i , s }$ . Since $E _ { i } ^ { 0 }$ and $E _ { i } ^ { 1 }$ are independent of the labels, so are the selected inputs. Shown for $T \textrm { a }$ multiple of $p ;$ spacing is schematic.

Lemma 3. For $d \geq p + 1$ and $T \geq ( m + 3 ) p $ , the $m ^ { 2 }$ prompts in (35) are distinct and shattered by $\mathcal { F } _ { d , T }$

Proof. Fix $\lambda \in \{ 0 , 1 \} ^ { m \times m }$ and its encoding θ. Take $B , a _ { 1 } , \dotsc , a _ { p + 1 }$ from Lemma 2, and define

$$
\begin{array} { r } { w _ { \ell } = \left\{ \begin{array} { l l } { a _ { \ell } / B , } & { 1 \le \ell \le p + 1 , } \\ { 0 , } & { p + 1 < \ell \le d . } \end{array} \right. } \end{array}\tag{36}
$$

For each $i \in [ m ]$ , define $z _ { n } ^ { ( i ) }$ for $n \geq 1 - d$ by

$$
( z _ { 1 - d } ^ { ( i ) } , \ldots , z _ { 0 } ^ { ( i ) } ) = 0 ^ { d - p } E _ { i } ^ { 0 } ,\tag{37}
$$

$$
z _ { n } ^ { ( i ) } = \mathbf { 1 } \{ \sum _ { \ell = 1 } ^ { d } w _ { \ell } z _ { n - \ell } ^ { ( i ) } \geq 1 \} \qquad ( n \geq 1 ) .\tag{38}
$$

We prove by induction on $t \geq 0$ that

$$
( z _ { ( t - 1 ) p } ^ { ( i ) } , \cdot \cdot \cdot , z _ { t p } ^ { ( i ) } ) = ( 0 , E _ { i } ^ { t } ) .\tag{39}
$$

For $t = 0 ,$ , this follows from (37) and $d \geq p + 1$ . Assume (39) holds for t. By (36), Lemma 2, and $( E _ { i } ^ { t } ) _ { p - 1 } = 0$ from (15), we have

$$
( z _ { t p } ^ { ( i ) } , \ldots , z _ { ( t + 1 ) p } ^ { ( i ) } ) = ( ( E _ { i } ^ { t } ) _ { p - 1 } , E _ { i } ^ { t + 1 } ) = ( 0 , E _ { i } ^ { t + 1 } ) .
$$

This proves (39). By (33), (37), and (39) at $t = 1$

$$
z _ { n } ^ { ( i ) } = ( E _ { i } ) _ { n } \qquad ( 1 - d \leq n \leq p ) .\tag{40}
$$

Fix $i , s \in [ m ]$ . Define

$$
z _ { n } ^ { ( i , s ) } = x _ { n } ^ { i , s } \qquad ( 1 - d \leq n \leq 0 ) ,
$$

$$
z _ { n } ^ { ( i , s ) } = \mathbf { 1 } \left\{ \sum _ { \ell = 1 } ^ { d } w _ { \ell } z _ { n - \ell } ^ { ( i , s ) } \geq 1 \right\} \qquad ( 1 \leq n \leq T ) .\tag{41}
$$

Since $0 \leq r _ { s } < p _ { ; }$ equations (35) and (40) give

$$
z _ { n } ^ { ( i , s ) } = x _ { n } ^ { i , s } = ( E _ { i } ) _ { r _ { s } + n } = z _ { r _ { s } + n } ^ { ( i ) } \qquad ( 1 - d \leq n \leq 0 ) .
$$

For $1 \leq n \leq T$ , assume this equality holds at every index from $1 - d \mathrm { t o } \ n - 1$ . Since $1 - d \leq n - \ell < n$ for

$\ell \in [ d ]$ and $r _ { s } + n \ge 1$ , (41) and (38) give

$$
z _ { n } ^ { ( i , s ) } = \mathbf { 1 } \{ \sum _ { \ell = 1 } ^ { d } w _ { \ell } z _ { r _ { s } + n - \ell } ^ { ( i ) } \geq 1 \} = z _ { r _ { s } + n } ^ { ( i ) } .
$$

Induction therefore yields

$$
f _ { w } ^ { T } ( x ^ { i , s } ) = z _ { T } ^ { ( i , s ) } = z _ { T + r _ { s } } ^ { ( i ) } .\tag{42}
$$

By (34), $t : = 1 + ( T + r _ { s } - 1 - \pi ( \mathrm { o u t } , s ) ) / p$ is an integer satisfying

$$
\begin{array} { r } { T + r _ { s } = ( t - 1 ) p + 1 + \pi ( \mathrm { o u t } , s ) , } \end{array}\tag{43}
$$

$$
t - 1 = { \frac { T + r _ { s } - 1 - \pi ( { \mathrm { o u t } } , s ) } { p } } \geq { \frac { T - p } { p } } \geq m + 2 .
$$

The inequalities use $r _ { s } \ge 0 , \pi ( \mathrm { o u t } , s ) \le p - 1$ , and $T \geq ( m + 3 ) p$ . Thus (42), (43), (39), (15), and Lemma 1 imply

$$
f _ { w } ^ { T } ( x ^ { i , s } ) = ( E _ { i } ^ { t } ) _ { \pi ( \mathrm { o u t } , s ) } = f _ { \theta , i , \mathrm { o u t } , s } ^ { t } = \lambda _ { i , s } .\tag{44}
$$

The vector w in (36) is common to all i, s. By Lemma 1 and $( 3 3 )  { - } ( 3 5 )$ , the inputs do not depend on λ. It remains only to show that all of these inputs are distinct. Assume for a contradiction that $x ^ { i , s } = x ^ { i ^ { \prime } , s ^ { \prime } }$ for distinct index pairs $( i , s ) \neq ( i ^ { \prime } , s ^ { \prime } )$ ; then, choose $\lambda _ { i , s } = 0$ and $\lambda _ { i ^ { \prime } , s ^ { \prime } } = 1$ . Equation (44) then gives the contradiction

$$
0 = f _ { w } ^ { T } ( x ^ { i , s } ) = f _ { w } ^ { T } ( x ^ { i ^ { \prime } , s ^ { \prime } } ) = 1 .
$$

Hence, the $m ^ { 2 }$ inputs are distinct, and (44) shows that they are shattered by $\mathcal { F } _ { d , T }$

We now combine the preceding construction with the choice of m to prove Theorem 1.

Proof of Theorem 1. Set

$$
m = \operatorname* { m a x } \left\{ 1 , \left\lfloor { \frac { d - 1 } { 3 7 8 } } \right\rfloor - 2 \right\} .\tag{45}
$$

If $m \geq 2 ,$ , equations (13) and (45) give

$$
\begin{array} { c } { { p = 3 7 8 ( m + 2 ) \leq d - 1 , } } \\ { { ( m + 3 ) p < p ^ { 2 } < d ^ { 2 } \leq T . } } \end{array}
$$

Lemma 3 therefore gives VCdim $\phantom { } _ { 1 } ( \mathcal { F } _ { d , T } ) \geq m ^ { 2 }$ . If m = 1, (1) gives

$$
f _ { \bf 0 } ^ { T } ( 1 , \ldots , 1 ) = 0 , \qquad f _ { ( 1 , 0 , \ldots , 0 ) } ^ { T } ( 1 , \ldots , 1 ) = 1 ,\tag{46}
$$

so VCdim $( \mathcal { F } _ { d , T } ) \ge 1 = m ^ { 2 }$ in this case as well.

Equation (45) implies $d \leq 1 5 1 2$ when $m = 1$ , and $d \leq 3 7 8 ( m + 3 ) \leq 1 5 1$ 2m when $m \geq 2$ . Consequently,

$$
\mathrm { V C d i m } ( \mathcal { F } _ { d , T } ) \ge m ^ { 2 } \ge \frac { d ^ { 2 } } { 1 5 1 2 ^ { 2 } } \ge \frac { d ^ { 2 } } { 3 0 2 4 ^ { 2 } } .
$$

For $m \geq 2 .$ , the coeficient definitions (17)–(18), (22), and (23) give $\begin{array} { r } { \operatorname* { m a x } _ { 1 \leq \ell \leq p + 1 } | a _ { \ell } | \leq B ^ { 2 } } \end{array}$ and

$$
\log _ { 2 } B = m + 5 + \log _ { 2 } ( m + 2 ) = O ( m + \log m ) .
$$

Thus, each weight in (36) has a numerator and denominator of $O ( m + \log m )$ bits. Since $m = O ( d )$ , the weights have polynomial bit length in d. For $m = 1$ , the two vectors in (46) have entries in {0, 1}. □

## 5 Stochastic Digit Extraction

We prove Theorem 2 in three steps. First, we show how repeated independent random bits can recover any of $2 ^ { m }$ labels encoded in a single real number. Next, we extend Lemma 2 to implement this procedure with one logistic generator, while controlling the simulation errors. Finally, we construct $2 ^ { m }$ fixed, prompts

independent of the labels whose final output probabilities realize every labeling at a common horizon. Fix m $\geq 2$ and put $N = 2 ^ { m }$ . For arbitrary labels $\lambda _ { 0 } , \ldots , \lambda _ { N - 1 } \in \{ 0 , 1 \}$ , encode

$$
\alpha = 4 ^ { - 2 N } + \sum _ { j = 0 } ^ { N - 1 } 2 ( 1 - \lambda _ { j } ) 4 ^ { - ( j + 1 ) } .\tag{47}
$$

Then $0 < \alpha < 1$ . For $x = ( x _ { 0 } , \ldots , x _ { m - 1 } ) \in \{ 0 , 1 \} ^ { m }$ , define its address

$$
j ( x ) = \sum _ { r = 0 } ^ { m - 1 } 2 ^ { r } x _ { r } .\tag{48}
$$

Set $K = 4 ^ { 2 N }$ and $\beta = \log ( 2 \pi \alpha / K )$ , where log is the natural logarithm. The following odds give an afine logistic sampling probability:

$$
a _ { x } = \frac { 2 \pi \alpha } { K } 4 ^ { j ( x ) } ,\tag{49}
$$

$$
p _ { x } = \frac { a _ { x } } { 1 + a _ { x } } = \sigma \big ( \beta + ( \log 4 ) j ( x ) \big ) .\tag{50}
$$

For each fixed $x \in \{ 0 , 1 \} ^ { m }$ , let $f _ { \mathrm { t e s t } } ^ { t } \in \{ 0 , 1 \} , t \geq 1$ , be independent Bernoulli(p<sub>x</sub>) random variables and define

$$
S _ { t } = \sum _ { s = 1 } ^ { t } f _ { \mathrm { t e s t } } ^ { s } ,\tag{51}
$$

$$
Y _ { t } = \mathbf { 1 } \{ S _ { t } { \mathrm { ~ m o d ~ } } 4 \in \{ 0 , 1 \} \} .\tag{52}
$$

Set $\begin{array} { r } { \delta _ { N } = \frac { 1 } { 4 } e ^ { - 2 \cdot 4 ^ { N } } } \end{array}$ . The next lemma shows that the probability of the final output being 1 lies above or below $1 / 2$ , according to the selected label, with a gap of at least $\delta _ { N }$

Lemma 4. For every $x \in \{ 0 , 1 \} ^ { m }$ it holds that $( 2 \lambda _ { j ( x ) } - 1 ) ( \operatorname* { P r } ( Y _ { K } = 1 ) - 1 / 2 ) \geq \delta _ { N }$

Proof. In (47), the terms preceding digit j become integers after multiplication by $4 ^ { j }$ . Since $\textstyle \sum _ { h \geq 2 } 2 \cdot 4 ^ { - h } = 1 / 6$ for $0 \leq j < N$ , we have

$$
0 \leq 4 ^ { j } \alpha - \lfloor 4 ^ { j } \alpha \rfloor - \frac { 1 - \lambda _ { j } } { 2 } \leq \frac { 1 } { 6 } + 4 ^ { - N - 1 } < \frac { 1 } { 4 } .\tag{53}
$$

Thus, $2 \pi 4 ^ { j } \alpha - \pi / 4$ is within $\pi / 4$ of 0 when $\lambda _ { j } = 1$ and of π when $\lambda _ { j } = 0$ , modulo 2π. Hence,

$$
( 2 \lambda _ { j } - 1 ) \cos ( 2 \pi 4 ^ { j } \alpha - \pi / 4 ) \geq 1 / \sqrt { 2 } .\tag{54}
$$

Let $\mathrm { i } ^ { 2 } = - 1$ . For integer s, the real part $\mathrm { R e } ( ( 1 - \mathrm { i } ) \mathrm { i } ^ { s } )$ is 1 on residues 0, 1 and −1 on residues 2, 3. Independence gives

$$
\begin{array} { r } { \mathrm { P r } ( Y _ { K } = 1 ) - \frac { 1 } { 2 } = \frac { 1 } { 2 } \mathrm { R e } \bigl ( ( 1 - \mathrm { i } ) ( 1 - p _ { x } + \mathrm { i } p _ { x } ) ^ { K } \bigr ) } \end{array}
$$

$$
= \frac { 1 } { \sqrt { 2 } } \left( \frac { \sqrt { 1 + a _ { x } ^ { 2 } } } { 1 + a _ { x } } \right) ^ { K } \cos ( K \arctan a _ { x } - \pi / 4 ) .\tag{55}
$$

For $a \geq 0$ it holds that $0 \leq a -$ arctan $a \leq a ^ { 3 } / 3$ . Thus, since $K a _ { x } < 2 \cdot 4 ^ { N }$ by (49) we have

$$
0 \leq 2 \pi \alpha \mathrm { { 4 } } ^ { j ( x ) } - K \arctan { a _ { x } } \leq K { a _ { x } ^ { 3 } / 3 } < { \frac { 8 } { 3 } } \mathrm { { 4 } } ^ { - N } < { \frac { 1 } { 8 } } .\tag{56}
$$

The cosine function is 1-Lipschitz, so (54) gives

$$
( 2 \lambda _ { j ( x ) } - 1 ) \cos ( K \arctan a _ { x } - \pi / 4 ) \geq 1 / \sqrt { 2 } - 1 / 8 > 1 / 2 .\tag{57}
$$

Also,

$$
\left( \frac { \sqrt { 1 + a _ { x } ^ { 2 } } } { 1 + a _ { x } } \right) ^ { K } \geq ( 1 + a _ { x } ) ^ { - K } \geq e ^ { - K a _ { x } } \geq e ^ { - 2 \cdot 4 ^ { N } } .\tag{58}
$$

Equations (55), (57), and (58) give the proof.

## 5.1 Reusing the Deterministic Simulation

In this section, we first describe a recursion that generates the random bits from the previous section and tracks their sum modulo 4 using a constant number of threshold gates per round. We then implement this recursion with a single logistic generator, analogous to our techniques in the deterministic setting.

For fixed α and $x \in \{ 0 , 1 \} ^ { m }$ , we represent the recursion at round t by a random vector $f ^ { t } \in \{ 0 , 1 \} ^ { m + 7 }$ whose coordinates have the following roles. The m coordinates $( \mathrm { i n } , j )$ store the input bits $x _ { j }$ , and the coordinate test stores the newly sampled bit. The two coordinates $( { \mathrm { s t a t e } } , i ) , i \in \{ 1 , 2 \}$ , encode the number of sampled ones modulo 4. The auxiliary and output coordinates update the counter: if the sampled bit is $0 ,$ the state $( u , v )$ remains unchanged; if it is 1, the state becomes $( v , 1 - u )$ . These updates are computed using linear threshold functions. We suppress $\alpha _ { \mathrm { { : } } }$ x in the notation. Initially, no bits have been sampled, and the counter value 0 is represented by (0, 1). Thus, all coordinates of $f ^ { 0 }$ are zero except

$$
f _ { \mathrm { s t a t e , 2 } } ^ { 0 } = 1 .\tag{59}
$$

Fix $G \geq 1$ . At round $t \geq 1$ , compute these scores in order, immediately generating the corresponding bit by (61)–(62):

$$
g _ { \mathrm { t e s t } } ^ { t } = \beta + ( \log 4 ) \sum _ { j = 0 } ^ { m - 1 } 2 ^ { j } f _ { \mathrm { i n } , j } ^ { t - 1 } ,\tag{60a}
$$

$$
g _ { \mathrm { a u x } , 1 } ^ { t } = G ( f _ { \mathrm { s t a t e , 2 } } ^ { t - 1 } + f _ { \mathrm { t e s t } } ^ { t } - 3 / 2 ) ,\tag{60b}
$$

$$
g _ { \mathrm { a u x , 2 } } ^ { t } = G ( f _ { \mathrm { t e s t } } ^ { t } - f _ { \mathrm { s t a t e , 1 } } ^ { t - 1 } - 1 / 2 ) ,\tag{60c}
$$

$$
g _ { \mathrm { o u t } , 1 } ^ { t } = G ( f _ { \mathrm { s t a t e } , 1 } ^ { t - 1 } - f _ { \mathrm { t e s t } } ^ { t } + 2 f _ { \mathrm { a u x } , 1 } ^ { t } - 1 / 2 ) ,\tag{60d}
$$

$$
g _ { \mathrm { o u t , 2 } } ^ { t } = G ( f _ { \mathrm { s t a t e , 2 } } ^ { t - 1 } - f _ { \mathrm { t e s t } } ^ { t } + 2 f _ { \mathrm { a u x , 2 } } ^ { t } - 1 / 2 ) ,\tag{60e}
$$

$$
g _ { \mathrm { s t a t e } , i } ^ { t } = G ( f _ { \mathrm { o u t } , i } ^ { t } - 1 / 2 ) .\tag{60f}
$$

Let $\mathcal { V } = \{ \mathrm { t e s t } \} \cup \{ ( \mathrm { a u x } , i ) , ( \mathrm { o u t } , i ) , ( \mathrm { s t a t e } , i ) : i \in \{ 1 , 2 \} \}$ be the seven non-address coordinates. At each round $t ,$ independently sample $f _ { \mathrm { t e s t } } ^ { t } = 1$ with probability $\sigma ( g _ { \mathrm { t e s t } } ^ { t } )$ . Every other non-address bit $f _ { v } ^ { t }$ for $v \in \mathcal { V } \backslash$ {test} is determined by its score: it equals 1 if $g _ { v } ^ { t } \geq 0 ,$ , and 0 otherwise. The address bits remain unchanged. Formally,

$$
f _ { \mathrm { t e s t } } ^ { t } \sim \mathrm { B e r n o u l l i } ( \sigma ( g _ { \mathrm { t e s t } } ^ { t } ) ) ,\tag{61}
$$

$$
f _ { v } ^ { t } = { \bf 1 } \{ g _ { v } ^ { t } \geq 0 \} ,\tag{62}
$$

$$
f _ { \mathrm { i n } , j } ^ { t } = f _ { \mathrm { i n } , j } ^ { t - 1 } .\tag{63}
$$

Since the address bits remain equal to $x _ { j } , ( 6 0 \mathrm { a } )$ and (50) give $\sigma ( g _ { \mathrm { t e s t } } ^ { t } ) = \sigma \big ( \beta + ( \log 4 ) j ( x ) \big ) = p _ { x }$ . Thus, the bits $f _ { \mathrm { t e s t } } ^ { t }$ are independent Bernoul $\operatorname { l i } ( p _ { x } )$ random variables, as required.

Let $( u _ { t } , v _ { t } ) = ( f _ { \mathrm { s t a t e , 1 } } ^ { t } , f _ { \mathrm { s t a t e , 2 } } ^ { t } )$ denote the counter state after round t. Equations (60b)–(60f) leave this state unchanged when $f _ { \mathrm { t e s t } } ^ { t } = 0$ , and change $( u , v )$ to $( v , 1 - u )$ when $f _ { \mathrm { t e s t } } ^ { t } = 1$ . Starting from (0, 1), each sampled 1 advances the state along the cycle $( 0 , 1 )  ( 1 , 1 )  ( 1 , 0 )  ( 0 , 0 )  ( 0 , 1 )$ . Thus, the second coordinate equals 1 exactly when the number of sampled ones is 0 or 1 modulo 4. Consequently, by (52)

$$
f _ { \mathrm { o u t , 2 } } ^ { t } = f _ { \mathrm { s t a t e , 2 } } ^ { t } = Y _ { t } .\tag{64}
$$

## Block Construction

We now define the blocks and first the positions within each block, analogous to the deterministic case. Set $p = 6 1 4 4 ( m + 1 )$ . For the indices $i , j$ above, define

$$
\begin{array} { r l } & { ~ \pi ( { \mathrm { t e s t } } ) = p / 6 + 4 , } \\ & { ~ \pi ( { \mathrm { a u x } } , i ) = p / 6 + 4 \cdot 2 ^ { i } , } \\ & { ~ \pi ( { \mathrm { o u t } } , i ) = p / 6 + 1 6 \cdot 2 ^ { i } , } \\ & { ~ \pi ( { \mathrm { s t a t e } } , i ) = p / 6 + 6 4 \cdot 2 ^ { i } , } \\ & { ~ \pi ( \mathrm { i n } , j ) = p / 6 + 1 0 2 4 j + 5 1 2 . } \end{array}\tag{65}
$$

These are distinct multiples of 4 in $( p / 6 , p / 3 )$ . As in (15), define $E ^ { t } \in \{ 0 , 1 \} ^ { p }$ , indexed by $0 , \ldots , p - 1$ , by setting, for $v \in \mathcal V$ and $e \in \{ 0 , 1 \}$

$$
\begin{array} { c } { E _ { 0 } ^ { t } = 1 , } \\ { E _ { \pi ( \mathrm { i n } , j ) } ^ { t } = f _ { \mathrm { i n } , j } ^ { t } , } \\ { E _ { \pi ( v ) + e } ^ { t } = f _ { v } ^ { t } , } \end{array}\tag{66}
$$

and all other entries to zero. Write $E _ { x } ^ { 0 }$ for the initial block; it is independent of the labels by (59).

Lemma 5. For every $\alpha \in ( 0 , 1 )$ and $G \geq 1$ , there exist $w \in \mathbb { R } ^ { p + 1 }$ and $b \in \mathbb { R }$ such that, for every $x \in \{ 0 , 1 \} ^ { m }$ ， the process (2) initialized at $( 0 , E _ { x } ^ { 0 } )$ generates blocks $\widehat { E } ^ { 1 } , \ldots , \widehat { E } ^ { K }$ satisfying

$$
\begin{array} { r } { \left| \operatorname* { P r } ( \widehat { E } _ { \pi ( \mathrm { o u t } , 2 ) } ^ { K } = 1 ) - \operatorname* { P r } ( Y _ { K } = 1 ) \right| \leq p K e ^ { - G / 2 } . } \end{array}
$$

Proof. The proof is an adaptation of Lemma 2 for the stochastic setting. For $v \in \mathcal V .$ , let $b _ { v }$ be the constant term and $c _ { v h u }$ the coeficient of $f _ { u } ^ { t - h }$ in (60), where $h \in \{ 0 , 1 \}$ and u ranges over all coordinates. For every nonzero $c _ { v h u } .$ let

$$
a _ { h p + \pi ( v ) - \pi ( u ) } ^ { ( 0 ) } = c _ { v h u } .\tag{67}
$$

For every $c _ { v h u } \neq 0$ with $u \in \mathcal V$ , also define

$$
a _ { h p + \pi ( v ) - \pi ( u ) + 1 } ^ { ( 0 ) } = - c _ { v h u } .\tag{68}
$$

All other entries of $a ^ { ( 0 ) } \in \mathbb { R } ^ { p + 1 }$ are zero. By (65), all coeficient indices defined in (67)–(68) belong to $( 0 , p / 6 ) \cup ( 5 p / 6 , p )$ . The nonzero ordered diferences of $\{ 2 ^ { i } : 2 \leq i \leq 9 \}$ are distinct modulo 1024: their absolute values are at most 508, and $2 ^ { a } - 2 ^ { b } = 2 ^ { b } ( 2 ^ { a - b } - \mathrm { \bar { 1 } } )$ identifies $a > b .$ . By (65)–(68), every $\ell \in [ p + 1 ]$ with $a _ { \ell } ^ { ( 0 ) } \neq 0$ satisfies $0 < \ell < p / 6$ or $5 p / 6 < \ell < p$ . Set

$$
B = 1 6 \left( 1 + 3 G + \sum _ { \ell = 1 } ^ { p + 1 } | a _ { \ell } ^ { ( 0 ) } | + \sum _ { v \in \mathcal { V } } | b _ { v } | \right) .\tag{69}
$$

We now define the coeficients $a _ { 1 } , \dotsc , a _ { p + 1 }$ for the stochastic construction. Initially, set $a _ { \ell } = a _ { \ell } ^ { ( 0 ) }$ for every $\ell \in [ p + 1 ]$ , where $a ^ { ( 0 ) }$ was defined in (67)–(68). Then overwrite the following coeficients:

$$
\begin{array} { r } { \begin{array} { c } { a _ { 1 } = 2 G , } \\ { a _ { p - 1 } = - ( B + G ) , } \\ { a _ { p } = B + G , } \\ { a _ { p + 1 } = - ( B + G ) , } \\ { a _ { \pi ( v ) } = B + b _ { v } , } \\ { a _ { \pi ( v ) + 1 } = B - G . } \end{array} } \end{array}\tag{70}
$$

Finally, set $a _ { \ell } = - B ^ { 2 } \mathrm { f o r }$ every integer $p / 3 < \ell \leq 2 p / 3$ , as in the deterministic construction (23). These indices do not overlap any coeficients assigned above, so this completes the definition of the vector a for the stochastic setting.

Fix $t \in [ K ]$ . To analyze how the logistic generator produces $E ^ { t }$ from $E ^ { t - 1 }$ we do the following. For each position $0 \leq r < p ,$ define its logistic score by index $\widehat { E } = ( 0 , E ^ { t - 1 } , E ^ { t } ) \mathrm { ~ b y ~ } - p - 1 , \dotsc , p - 1$ , and define, for $0 \leq r < p ;$

$$
s _ { r } = - B + \sum _ { \ell = 1 } ^ { p + 1 } a _ { \ell } \widehat { E } _ { r - \ell } .\tag{71}
$$

By (66), for every coordinate u and $h \in \{ 0 , 1 \}$ ,

$$
\begin{array} { r l } & { \qquad \widehat { E } _ { \pi ( u ) - h p } = f _ { u } ^ { t - h } , } \\ & { \widehat { E } _ { \pi ( u ) - h p - 1 } = 0 , } \\ & { \widehat { E } _ { \pi ( u ) - h p + 1 } = \left\{ { f } _ { u } ^ { t - h } , \ : \ : u \in \mathcal { V } , \right. } \\ & { \qquad \left. u = ( \mathrm { i n } , j ) . \right. } \end{array}\tag{72}
$$

By the uniqueness of coeficient indices established above, (67)–(68) and (72) give, for every $v \in \mathcal { V } .$

$$
\sum _ { \ell = 1 } ^ { p + 1 } a _ { \ell } ^ { ( 0 ) } \widehat { E } _ { \pi ( v ) - \ell } = \sum _ { h , u } c _ { v h u } f _ { u } ^ { t - h } = g _ { v } ^ { t } - b _ { v } ,\tag{73}
$$

$$
\sum _ { \ell = 1 } ^ { p + 1 } a _ { \ell } ^ { ( 0 ) } \widehat { E } _ { \pi ( v ) + 1 - \ell } = \sum _ { h , u \in \mathcal { V } } c _ { v h u } \big ( f _ { u } ^ { t - h } - f _ { u } ^ { t - h } \big ) = 0 .\tag{74}
$$

The same index uniqueness and the absence of nonzero block entries outside the positions in (66) give

$$
\sum _ { \ell = 1 } ^ { p + 1 } a _ { \ell } ^ { ( 0 ) } \widehat { E } _ { \pi ( \mathrm { i n } , j ) - \ell } = 0 ,\tag{75}
$$

$$
\sum _ { \ell = 1 } ^ { p + 1 } a _ { \ell } ^ { ( 0 ) } \widehat { E } _ { - \ell } = 0 .
$$

For $r \in \{ 0 , \pi ( \mathrm { i n } , j ) , \pi ( v ) , \pi ( v ) + 1 \}$ , where $v \in \mathcal V$ and $0 \leq j < m$ , the block encoding gives

$$
\begin{array} { r l } { \widehat { E } _ { r - \pi ( v ) - e } = \mathbf { 1 } \{ r = \pi ( v ) + e \} \qquad } & { ( e \in \{ 0 , 1 \} ) , } \\ { \widehat { E } _ { r - \ell } = 0 \qquad } & { ( p / 3 < \ell \leq 2 p / 3 ) , } \end{array}
$$

$$
\widehat { E } _ { r - 1 } = \left\{ \begin{array} { l l } { f _ { v } ^ { t } , } & { r = \pi ( v ) + 1 , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{76}
$$

$$
\widehat { E } _ { r - p } - \widehat { E } _ { r - p + 1 } - \widehat { E } _ { r - p - 1 } = \left\{ \begin{array} { l l } { 1 , } & { r = 0 , } \\ { f _ { \mathrm { i n } , j } ^ { t - 1 } , } & { r = \pi ( \mathrm { i n } , j ) , } \\ { 0 , } & { r = \pi ( v ) \mathrm { o r } \pi ( v ) + 1 . } \end{array} \right.
$$

Substituting (73)–(76) and the coeficients in (70) into (71) gives

$$
\begin{array} { c } { { s _ { \pi ( v ) } = g _ { v } ^ { t } , } } \\ { { \nonumber } } \\ { { s _ { \pi ( v ) + 1 } = G ( 2 f _ { v } ^ { t } - 1 ) , } } \\ { { \nonumber } } \\ { { s _ { \pi ( \mathrm { i n } , j ) } = ( B + G ) f _ { \mathrm { i n } , j } ^ { t - 1 } - B , } } \\ { { \nonumber } } \\ { { s _ { 0 } = G . } } \end{array}\tag{77}
$$

Let

$$
\Pi = \{ 0 \} \cup \{ \pi ( \mathrm { i n } , j ) : 0 \leq j < m \} \cup \bigcup _ { v \in \mathcal { V } } \{ \pi ( v ) , \pi ( v ) + 1 \} .
$$

We prove that $s _ { r } < - G$ for every $r \in \{ 0 , \ldots , p - 1 \} \setminus$ Π.

By (69) and (70),

$$
\sum _ { \ell = 1 } ^ { p + 1 } \operatorname* { m a x } \{ a _ { \ell } , 0 \} \leq 1 5 B + 3 G + \sum _ { \ell = 1 } ^ { p + 1 } | a _ { \ell } ^ { ( 0 ) } | + \sum _ { v \in \mathcal { V } } | b _ { v } | < B ^ { 2 } .
$$

If $p / 3 < r \leq 2 p / 3$ , then $a _ { r } = - B ^ { 2 }$ and $\widehat { E } _ { 0 } = 1$ , so (71) gives

$$
s _ { r } \leq - B - B ^ { 2 } + \sum _ { \ell = 1 } ^ { p + 1 } \operatorname* { m a x } \{ a _ { \ell } , 0 \} < - B < - G .
$$

Now suppose $r \not \in$ Π and $r \not \in ( p / 3 , 2 p / 3 ]$ . By (66), every nonzero position of $\widehat { E }$ belongs to $( \Pi - p ) \cup \Pi$ Since $\pi ( v ) \in ( p / 6 , p / 3 - 1 )$ for every $v \in \mathcal { V } .$ it follows that

$$
\widehat { E } _ { r - p } = 0 , \widehat { E } _ { r - \pi ( v ) } = \widehat { E } _ { r - \pi ( v ) - 1 } = 0 ( v \in \mathcal { V } ) .
$$

Hence, the terms with coeficients $a _ { p } , a _ { \pi ( v ) }$ , and $a _ { \pi ( v ) + 1 }$ vanish. The coeficients $a _ { p - 1 } , \ : a _ { p + 1 }$ , and those in

$( p / 3 , 2 p / 3 ]$ are nonpositive. Since $a _ { 1 } = 2 G$ and all bits belong to {0, 1}, (71) gives

$$
s _ { r } \leq - B + \sum _ { \ell = 1 } ^ { p + 1 } | a _ { \ell } ^ { ( 0 ) } | + 2 G < - G ,
$$

where the last inequality follows from (69). Thus,

$$
s _ { r } \leq - B + \sum _ { \ell = 1 } ^ { p + 1 } | a _ { \ell } ^ { ( 0 ) } | + 2 G < - G .\tag{78}
$$

Set $w = a$ and $b = - B$ in (2). For every $t \in [ K ]$ and $0 \leq r < p ,$ let $U _ { t , r }$ be independent uniform random variables on [0, 1]. Define the ideal samples by $f _ { \mathrm { t e s t } } ^ { t } = \mathbf { 1 } \{ U _ { t , \pi ( \mathrm { t e s t } ) } \leq p _ { x } \}$ , and the remaining ideal bits by (62)–(63). Initialize $( Z _ { - p } , \ldots , Z _ { 0 } ) = ( 0 , E _ { x } ^ { 0 } )$ ). For $n = ( t - 1 ) p + 1 + r ,$ generate

$$
Z _ { n } = \mathbf { 1 } \{ U _ { t , r } \leq \sigma \left( - B + \sum _ { \ell = 1 } ^ { p + 1 } a _ { \ell } Z _ { n - \ell } \right) \} ,
$$

and define $\widehat { E } _ { r } ^ { t } = Z _ { n }$

Let $A _ { t , r }$ be the event that $\widehat { E } _ { j } ^ { s } = E _ { j } ^ { s }$ at every position $( s , j )$ preceding (t, r) in lexicographic order. On $A _ { t , r }$ , the score of $Z _ { n }$ equals $s _ { r }$ in (71). If $r = \pi ( \mathrm { t e s t } )$ , then $s _ { r } = g _ { \mathrm { t e s t } } ^ { t }$ , so $\widehat { E } _ { r } ^ { t } = E _ { r } ^ { t }$ by the shared variable $U _ { t , r }$ At every other position, (77) and (78) give $( 2 E _ { r } ^ { t } - 1 ) s _ { r } \geq G / 2$ . Since $U _ { t , r }$ is independent of all preceding variables, $\mathrm { P r } ( \widehat { E } _ { r } ^ { t } \neq E _ { r } ^ { t } \mid A _ { t , r } ) \leq e ^ { - G / 2 }$ . The events $A _ { t , r } \cap \{ \widehat { E } _ { r } ^ { t } \neq E _ { r } ^ { t } \}$ are disjoint and cover every possible first disagreement. Hence,

$$
\operatorname* { P r } ( \exists t \in [ K ] : { \widehat { E } } ^ { t } \neq E ^ { t } ) \leq p K e ^ { - G / 2 } .
$$

Finally, $E _ { \pi ( \mathrm { o u t } , 2 ) } ^ { K } = Y _ { K }$ by (64) and (66), so the claimed probability bound follows.

## 5.2 Choosing the Stochastic Inputs

For each $x \in \{ 0 , 1 \} ^ { m }$ , choose

$$
\begin{array} { r } { \widehat { x } = ( 0 , E _ { x } ^ { 0 } ) \in \{ 0 , 1 \} ^ { p + 1 } . } \end{array}\tag{79}
$$

The prompts are distinct and independent of the labels by (59) and (66). The first generated bit has block position 0, so the output of round K occurs at step

$$
T _ { m } = ( K - 1 ) p + 1 + \pi ( \mathrm { o u t , 2 } ) < K p .\tag{80}
$$

Proof of Theorem 2. Fix $m \geq 2$ and put $N = 2 ^ { m }$ . For arbitrary labels Given arbitrary labels $\lambda _ { 0 } , \dots , \lambda _ { N - 1 } \in$ {0, 1}, choose α by (47) and apply Lemma 5 with

$$
G = 2 \log ( 2 p K / \delta _ { N } ) \geq 1 .\tag{81}
$$

By (64) and (80), the lemma’s weights and bias satisfy

$$
\begin{array} { r } { \left| q _ { w , b } ^ { ( T _ { m } ) } ( \widehat { x } ) - \operatorname* { P r } ( Y _ { K } = 1 ) \right| \le p K e ^ { - G / 2 } = \delta _ { N } / 2 . } \end{array}\tag{82}
$$

Lemma 4 gives, for every x,

$$
\begin{array} { r } { ( 2 \lambda _ { j ( x ) } - 1 ) \left( q _ { w , b } ^ { ( T _ { m } ) } ( \widehat { x } ) - \frac { 1 } { 2 } \right) \geq \delta _ { N } / 2 > 0 . } \end{array}\tag{83}
$$

Thus $\mathcal { H } _ { p + 1 , T _ { m } }$ shatters the $2 ^ { m }$ prompts. For suficiently large $d ,$ choose the largest $m \geq 2$ with

$$
p + 1 = 6 1 4 4 ( m + 1 ) + 1 \leq d .\tag{84}
$$

Prepend zeros to the prompts and set additional weights to zero; the output probabilities are unchanged. Since $m = \Theta ( d )$

$$
\mathrm { V C d i m } ( \mathcal { H } _ { d , T _ { m } } ) \geq 2 ^ { m } = 2 ^ { \Omega ( d ) } ,\tag{85}
$$

$$
T _ { m } < 6 1 4 4 ( m + 1 ) 4 ^ { 2 \cdot 2 ^ { m } } \leq 2 ^ { 2 ^ { O ( d ) } } .\tag{86}
$$

The horizon depends only on $d .$

## Acknowledgments

Idan Mehalel is supported by the European Research Council (ERC) under the European Union’s Horizon 2022 research and innovation program (grant agreement No. 101041711), the Israel Science Foundation (grant number 2258/19), and the Simons Foundation (as part of the Collaboration on the Mathematical and Scientific Foundations of Deep Learning). Ilan Doron-Arad is supported by grant NSF DMS-2031883 and Vannevar Bush Faculty Fellowship ONR-N00014-20-1-2826 (PI Mossel). Elchanan Mossel is partially supported by NSF DMS-2031883, Vannevar Bush Faculty Fellowship ONR-N00014-20-1-2826, MURI N000142412742, and a Simons Investigator Award.

AI assistance. GPT Astra found the main proof technique. Iterative work with GPT Astra simplified the construction and its presentation. The authors take full responsibility for the paper.

## References

[1] Jefrey C. Lagarias. The 3x + 1 problem and its generalizations. The American Mathematical Monthly, 92(1):3–23, 1985. doi:10.2307/2322189.

[2] Divakar Viswanath. Random Fibonacci sequences and the number 1.13198824 . . .. Mathematics of Computation, 69(231):1131–1155, 2000. doi:10.1090/S0025-5718-99-01145-X.

[3] Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, Charles Sutton, and Augustus Odena. Show your work: Scratchpads for intermediate computation with language models. arXiv:2112.00114, 2021.

[4] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, volume 35, pages 24824–24837, 2022. arXiv:2201.11903.

[5] Eran Malach. Auto-regressive next-token predictors are universal learners. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 34417–34431, 2024. https://proceedings.mlr.press/v235/malach24a.html.

[6] William Merrill and Ashish Sabharwal. The expressive power of transformers with chain of thought. In International Conference on Learning Representations, 2024. https://openreview.net/forum?id=CDmerQ37Zs.

[7] Nirmit Joshi, Gal Vardi, Adam Block, Surbhi Goel, Zhiyuan Li, Theodor Misiakiewicz, and Nathan Srebro. A theory of learning with autoregressive chain of thought. In Proceedings of the 38th Conference on Learning Theory, volume 291 of Proceedings of Machine Learning Research, pages 3161–3212, 2025. https://proceedings.mlr.press/v291/joshi25a.html.

[8] Frank Rosenblatt. The perceptron: A probabilistic model for information storage and organization in the brain. Psychological Review, 65(6):386–408, 1958. doi:10.1037/h0042519.

[9] Itay Hubara, Matthieu Courbariaux, Daniel Soudry, Ran El-Yaniv, and Yoshua Bengio. Binarized neural networks. In Advances in Neural Information Processing Systems, volume 29, 2016. https://papers.nips.cc/paper/6573-binarized-neural-networks.

[10] Anselm Blumer, Andrzej Ehrenfeucht, David Haussler, and Manfred K. Warmuth. Learnability and the Vapnik–Chervonenkis dimension. Journal of the ACM, 36(4):929–965, 1989. doi:10.1145/76359.76371.

[11] Thomas M. Cover. Geometrical and statistical properties of systems of linear inequalities with applications in pattern recognition. IEEE Transactions on Electronic Computers, EC-14(3):326–334, 1965. doi:10.1109/PGEC.1965.264137.

[12] Yu. A. Zuev. Asymptotics of the logarithm of the number of Boolean threshold functions. Doklady Akademii Nauk SSSR, 306(3):528–530, 1989. English translation in Soviet Mathematics Doklady, 39:512–513. https://www.mathnet.ru/eng/dan7111.

[13] Steve Hanneke, Idan Mehalel, and Shay Moran. Sample complexity of autoregressive reasoning: Chain-of-thought vs. end-to-end. arXiv:2604.12013, 2026.

[14] Ilan Doron-Arad, Idan Mehalel, and Elchanan Mossel. A theory of online learning with autoregressive chain-of-thought reasoning. arXiv:2605.06819, 2026.

[15] Zhiyuan Li. The optimal sample complexity of learning autoregressive chain-of-thought. arXiv:2607.07423, 2026.

[16] Nirmit Joshi, Roey Magen, Nathan Srebro, Nikolaos Tsilivis, and Gal Vardi. Learning to think from multiple thinkers. arXiv:2604.24737, 2026.

[17] Ilan Doron-Arad, Idan Mehalel, and Elchanan Mossel. Stochastic autoregressive learning. arXiv:2608.07224, 2026.

[18] Peter L. Bartlett, Nick Harvey, Christopher Liaw, and Abbas Mehrabian. Nearly-tight VC-dimension and pseudodimension bounds for piecewise linear neural networks. Journal of Machine Learning Research, 20(63):1–17, 2019. https://jmlr.org/papers/v20/17-612.html.

[19] Roey Magen, Nirmit Joshi, and Gal Vardi. The surprising expressivity of a single neuron with autoregressive chain-of-thought. arXiv:2610.17283, 2026.