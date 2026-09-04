# The Head Complexity of Boolean Functions in Single-Layer Atention

Rajmohan Rajaraman Ravi Sundaram<sup>∗</sup> Amanuel Tesfaye<sup>∗</sup>

## Abstract

What can a single layer of self-attention compute? We study head complexity: the minimum number of attention heads required to compute a function in a one-layer attention-only model. We establish an exact hierarchy under this measure: � heads compute �-bit parity but cannot compute (� + 1)-bit parity.

The lower bound is unconditional in the two resources a transformer might otherwise exploit; it holds at unbounded embedding dimension and unbounded numerical precision. The proof rests on an alternating-sum obstruction: after clearing the softmax denominators, every monomial in the resulting decision polynomial omits at least one of the � + 1 input bits, forcing its correlation with parity to vanish. The same obstruction yields lower bounds for related tasks, including the well-studied multi-hop induction-head task.

We also establish compactness bounds for embedding dimension and numerical precision. Specifically, a compactness theorem shows that any function computable at all can be computed with embedding dimension and precision bounded by the discrete data of the task, namely, head count, alphabet size, and length. Thus, potentially unbounded dimension or precision provably cannot substitute for heads. Finally, we derive nearly matching universal bounds for general binary functions: $2 ^ { n }$ heads sufice to compute every �-bit binary function, with one head per monomial in its multilinear expansion, while a counting argument shows almost all such functions require $\Omega ( 2 ^ { n } / n ^ { 2 } )$ heads. This lower bound matches the upper bound to within a poly(�) factor, even when dimension and precision are unbounded. Together, these results characterize head requirements for Boolean computation in this model.

## 1 Introduction

To understand what a transformer can compute, we analyze it not as an empirical artifact but as a formal model of computation. Half a century ago, the pioneers of circuit complexity asked how many logic gates a function requires, and the structural limits they proved for bounded-depth circuits became bedrock for the field. For a single attention layer, the natural analogue of the gate is the attention head, and the natural question is its head complexity: the minimum number of heads needed to compute a target function. This lens grew out of the two-head study [TKRS26], where two narrow heads solve the Endpoint Selection Problem (ESP) that a single head of unbounded width cannot; this motivated head count as a resource for studying single-layer expressivity. The present paper extends that observation to an exact hierarchy across head counts and to worst-case bounds for Boolean functions.

Why isolate a single attention layer? We isolate a single attention layer in order to study head count separately from depth and feed-forward computation. This restricted setting permits exact upper and lower bounds. Settling the head complexity has two additional payofs. It is a statement about cost — heads carry parameters and quadratic attention compute, so a lower bound on heads lower-bounds model size within this attention-only setting. And it grounds mechanistic interpretability, whose unit of analysis is the individual head. We therefore take the number of heads as the resource being measured.

## 1.1 Summary of Results

We measure a self-attention layer by its number of heads. Our main results are the following.

• An exact head hierarchy, calibrated by parity. We prove that � heads cannot compute (�+1)-bit parity, and give a matching construction in which � heads compute �-bit parity. Hence �-bit parity needs exactly � heads, and the head-count classes form a strictly increasing chain ${ \mathcal { F } } _ { 1 } \subsetneq { \mathcal { F } } _ { 2 } \subsetneq \cdot \cdot$ · that separates at every single head. The lower bound is unconditional: it holds at unbounded embedding dimension and unbounded numerical precision.

• One obstruction, many tasks. The lower bound is an elementary alternating-sum obstruction, and it is reusable: the same obstruction gives the exact (�+1)-head requirement for �-ESP, which is a generalization of the ESP task introduced in [TKRS26] and an unconditional (�+1)-head lower bound for the (�+1)-hop induction-head task [SHT24b]. The hierarchy is thus intrinsic to one-layer attention, not a peculiarity of parity.

• Dimension and precision can be bounded. A compactness theorem shows that any computable function has a realization whose embedding dimension and numerical precision are bounded in terms of the discrete data of the task: head count, alphabet size, and sequence length. The two potentially unbounded resources provably cannot substitute for heads.

• How many heads the hardest functions need. With $2 ^ { n }$ heads a single layer computes every �-bit Boolean function (one additive head per monomial of its multilinear expansion); a counting argument shows that almost all functions require $\Omega ( 2 ^ { n } / n ^ { 2 } )$ heads. The two bounds meet to within a poly(�) factor, even with unbounded embedding dimension and numerical precision.

Together, these results show that head count strictly orders expressivity in the one-layer attentiononly model, while unbounded dimension and precision do not eliminate the need for additional heads. Throughout, our upper bounds use the restricted additive embedding and our lower bounds the general joint one, the strongest pairing possible for these two regimes, since it pins both at once. One further result appears in the appendix: any joint �-head computation can be simulated additively with $O ( n ^ { k } )$ heads, giving polynomial overhead in � for every fixed �.

## 1.2 Closest Prior Work

The nearest parity lower bounds are due to Kozachinskiy, Steifer, and Wałęga [KSW26] and Hsu [Hsu26]. The former proves the qualitative statement that a fixed one-layer softmax transformer cannot compute parity, even with an MLP, using sensitivity and first-order-logic arguments; the latter proves the quantitative bound $h p \geq n / 4$ for one-layer softmax attention with a degree-� rational read-out, which for a linear read-out gives $h = \Omega ( n )$ heads. Both establish that parity is hard for one layer, but neither gives an exact head threshold. Our lower bound rules out � heads for (�+1)-bit parity while our construction attains �-bit parity with � heads, so the hierarchy separates at every single head; the compactness theorem then shows that allowing unbounded dimension or precision does not enlarge the class of computable functions. A full discussion of related work appears in Section 7.

## 1.3 Overview of Techniques

Our results rest on four main ideas.

The obstruction (Section 3.1). We first argue that clearing the strictly positive softmax denominators that appear in transformer computations turns the decision of a �-head transformer into the sign of a single polynomial $F _ { x }$ in the efective head variables. Each monomial of $F _ { x }$ is a product of � factors, and each factor reads only one input position, so a monomial depends on at most � of the $k + 1$ parity bits: some bit is always missing. Summing $F _ { x }$ against the parity sign then factors through $\begin{array} { r } { \sum _ { b \in \{ 0 , 1 \} } ( - 1 ) ^ { b } = 0 } \end{array}$ , giving $\begin{array} { r } { \sum _ { x } ( - 1 ) ^ { f ( x ) } F _ { x } = 0 ; } \end{array}$ this is impossible if every signed summand is nonnegative and positive for odd-parity inputs, as computing parity would require. The argument invokes nothing about the embedding, which is why it survives at unbounded dimension and precision, and nothing about parity beyond “each token sees one $\mathrm { b i t } ^ { \dprime } ,$ which is why it transfers to �-ESP and (�+1)-hop through a suitable encoding.

The interpolation (Section 3.2). The matching construction runs the rational-function viewpoint in reverse. The heads are arranged so that, on an input of Hamming weight �, head � has denominator $c + j$ and numerator gap afine in �, hence contributes a term $( \alpha _ { j } + \beta _ { j } c ) / ( c + j )$ to the decision statistic. A partial-fraction choice of the coeficients makes the �-term sum alternate in sign on the $k + 1$ Hamming levels $c = 0 , \ldots , k ,$ exactly matching parity. The �-ESP upper bound also uses the same Hamming-weight interpolation technique.

Compactness (Section 4). The layer is a bilinear machine: the function it computes reads its parameters only through one rectangular matrix of inner products, pairing the query and read-out vectors against the token embeddings. The required embedding dimension is at most one greater than the rank of that matrix. For precision, Renegar-type estimates [Ren92] first produce a bounded rational point in exponentiated-score coordinates; a quantitative stability argument transfers this bound to rational scores and hence to the underlying model matrices.

Counting (Section 5.2). The entire input–output behavior of a �-head transformer is the sign vector of the $2 ^ { n }$ polynomials $\{ F _ { x } \}$ evaluated at a single point in $O ( k n )$ variables. Warren’s bound on the number of sign patterns of low-degree polynomials caps the number of realizable functions at $2 ^ { O ( k n ^ { 2 } ) }$ ; comparing this with the $2 ^ { 2 ^ { n } }$ Boolean functions forces $k = \Omega ( 2 ^ { n } / n ^ { 2 } )$ for almost all of them. The nearly matching universal upper bound is a direct construction: one additive head per monomial of the target’s multilinear expansion.

## 2 Preliminaries

We study a one-layer, attention-only transformer that reads a Boolean input and predicts one output bit. We state the standard two-logit model and immediately reduce its decision to a single scalar. For some arguments we use an afine scalar threshold; Theorem 8 shows that this changes dimension by at most one coordinate but never changes the number of heads.

## 2.1 Inputs and embeddings

Fix � and present $x = ( x _ { 1 } , \ldots , x _ { n } ) \in \{ 0 , 1 \} ^ { n }$ as the sequence $x _ { 1 } x _ { 2 } \cdots x _ { n } ?$ , where ? is a fixed query token. Write $v _ { b , i } : = \Phi ( b , i ) \in \mathbb { R } ^ { d }$ for $b \in \{ 0 , 1 \}$ and $i \in [ n ]$ , and $\upsilon _ { \ ? } : = \Phi ( ? ) \in \mathbb { R } ^ { d }$ . Let $X = X ( x ) \in \mathbb { R } ^ { ( n + 1 ) \times d }$ have rows $\mathscr { V } _ { x _ { 1 } , 1 } , \ldots , \mathscr { V } _ { x _ { n } , n } , \mathscr { V } _ { ? }$ . In the general joint regime the vectors $v _ { b , i }$ are unrestricted; the additive regime requires $v _ { b , i } = \tau _ { b } + \pi _ { i }$ . Thus every additive model is a joint model.

## 2.2 The model

Definition 1 (One-layer �-attention-only transformer, arg-max read-out). Fix an embedding dimension �, a value width $d _ { \mathrm { o u t } }$ , and � heads. Head � has $A _ { j } \in \mathbb { R } ^ { d \times d }$ and $V _ { j } \in \mathbb { R } ^ { d \times d _ { \mathrm { o u t } } }$ ; the two output logits share $W _ { O } \in \mathbb { R } ^ { 2 \times d _ { \mathrm { o u t } } }$ with rows $w _ { 0 } , w _ { 1 }$ . At the query position,

$$
\mathrm { M H A } ( x ) : = \sum _ { j = 1 } ^ { k } \mathrm { S o f t m a x } ( v _ { \gamma } A _ { j } X ^ { \top } ) X V _ { j } , \qquad Z _ { s } ( x ) : = \langle w _ { s } , \mathrm { M H A } ( x ) \rangle \quad ( s \in \{ 0 , 1 \} ) .
$$

The model sets out $\mathbf { \Sigma } ( x ) : = \arg \operatorname* { m a x } _ { s \in \{ 0 , 1 \} } Z _ { s } ( x )$ , resolving ties toward 0; equivalently, ou $\mathbf { \boldsymbol { \cdot } } ( \mathbf { \boldsymbol { x } } ) = 1$ exactly when $Z _ { 1 } ( x ) > Z _ { 0 } ( x )$ . It computes $f : \{ 0 , 1 \} ^ { n }  \{ 0 , 1 \}$ when $\operatorname { o u t } ( x ) = f ( x )$ for every �.

## 2.3 Scalar normal form

Only the diference between the two logits matters. For head $j ,$ set

$$
\begin{array} { r l r l } & { \delta _ { j } : = V _ { j } ( w _ { 0 } - w _ { 1 } ) ^ { \top } , } & & { r _ { b , i } ^ { j } : = e ^ { v _ { \uparrow } A _ { j } v _ { b , i } ^ { \top } } , } & & { u _ { b , i } ^ { j } : = v _ { b , i } \cdot \delta _ { j } , } \\ & { r _ { ? } ^ { j } : = e ^ { v _ { \uparrow } A _ { j } v _ { \uparrow } ^ { \top } } > 0 , } & & { u _ { ? } ^ { j } : = v _ { ? } \cdot \delta _ { j } , } \\ & { d _ { x } ^ { j } : = r _ { ? } ^ { j } + \displaystyle \sum _ { i = 1 } ^ { n } r _ { x _ { i } , i } ^ { j } > 0 , } & & { s _ { x } ^ { j } : = r _ { ? } ^ { j } u _ { ? } ^ { j } + \displaystyle \sum _ { i = 1 } ^ { n } r _ { x _ { i } , i } ^ { j } u _ { x _ { i } , i } ^ { j } . } \end{array}
$$

Writing $a _ { j } , b _ { j }$ for the two columns of the folded map $Y _ { j } : = V _ { j } W _ { O } ^ { \top }$ , the vector numerator and scalar bridge satisfy

$$
n _ { x } ^ { j } : = r _ { 2 } ^ { j } v _ { 2 } + \displaystyle \sum _ { i = 1 } ^ { n } r _ { x _ { i } , i } ^ { j } v _ { x _ { i } , i } , \qquad \delta _ { j } = a _ { j } - b _ { j } , \qquad s _ { x } ^ { j } = n _ { x } ^ { j } \cdot ( a _ { j } - b _ { j } ) , \qquad \frac { n _ { x } ^ { j } } { d _ { x } ^ { j } } \cdot ( a _ { j } - b _ { j } ) = \frac { s _ { x } ^ { j } } { d _ { x } ^ { j } } .
$$

Thus the entire Boolean decision is governed by the output-0 gap

$$
\Delta ( x ) : = Z _ { 0 } ( x ) - Z _ { 1 } ( x ) = \sum _ { j = 1 } ^ { k } { \frac { s _ { x } ^ { j } } { d _ { x } ^ { j } } } , \qquad \mathrm { o u t } ( x ) = 1 \iff \Delta ( x ) < 0 .\tag{1}
$$

Hence the transformer computes � exactly when $f ( x ) = 1 \iff \Delta ( x ) < 0$ for every input. This ratioof-sums form is the only reduction used in the main arguments; a proof clears denominators only when needed.

Definition 2 (Afine scalar read-out). An afine scalar read-out uses the attention core of Definition 1 and outputs

$$
\mathrm { n e t } ( x ) = \mathbf { 1 } \{ \langle \omega , \mathrm { M H A } ( x ) \rangle - \theta > 0 \} , \qquad \omega \in \mathbb { R } ^ { d _ { \mathrm { o u t } } } , \ \theta \in \mathbb { R } .
$$

For constructions and counting we occasionally use this convention. As stated and proved in Theorem 8, it computes exactly the same Boolean functions at every head count as the two-logit model above.

## 3 The Exact Parity Hierarchy

The resource we track is the number of attention heads. This section shows that it strictly, and exactly, orders the power of the one-layer model of Definition 1: for every �, one more head lets the network compute strictly more Boolean functions, and (�+1)-bit parity marks the boundary. We prove the lower bound first (Section 3.1), then the matching upper bound (Section 3.2).

Head-count classes. Let $\mathcal { F } _ { k } = \mathcal { F } _ { k } ^ { \mathrm { j o i n t } }$ denote the Boolean functions, of any arity, computable by at most � heads in the joint model, with all other resources unbounded; let $\mathcal { F } _ { k } ^ { \mathrm { a d d } }$ denote the additive counterpart. Padding with zero-valued heads gives $\mathcal { F } _ { k } \subseteq \mathcal { F } _ { k + 1 }$ . Writing $\mathrm { P A R } _ { m }$ for �-bit parity, the two results below show that every inclusion is strict:

$$
\mathrm { P A R } _ { k + 1 } \in \mathcal { F } _ { k + 1 } \ \backslash \ \mathcal { F } _ { k } .
$$

## 3.1 The alternating-sum obstruction: � heads cannot compute (�+1)-bit parity

The (�+1)-parity problem takes input $x = ( x _ { 1 } , \dots , x _ { k + 1 } ) \in \{ 0 , 1 \} ^ { k + 1 }$ , presented as the (�+2)-token string $x _ { 1 } x _ { 2 } \cdot \cdot \cdot x _ { k + 1 } ?$ , and asks for the bit $\bigoplus _ { 1 \leq i \leq k + 1 } x _ { i }$

We specialize the computation criterion (1) of Section 2 to this task. Recall from there that, since the query may attend to itself, head � contributes the numerator $\begin{array} { r } { n _ { x } ^ { j } = r _ { ? } ^ { j } v _ { ? } + \sum _ { 1 \leq i \leq k + 1 } r _ { x _ { i } , i } ^ { j } v _ { x _ { i } , i } } \end{array}$ and the denominator $\begin{array} { r } { d _ { x } ^ { j } \ = \ r _ { ? } ^ { j } + \sum _ { 1 \leq i \leq k + 1 } r _ { x _ { i } , i } ^ { j } \ > \ 0 } \end{array}$ , where $v _ { b , i } ~ = ~ \Phi ( b , i ) , ~ v _ { ? } ~ = ~ \Phi ( ? ) , ~ r _ { b , i } ^ { j } ~ = ~ e ^ { v _ { ? } A _ { j } v _ { b , i } ^ { \top } } ~ > ~ 0 _ { ! }$ , and $r _ { \gamma } ^ { j } = e ^ { v _ { ? } A _ { j } v _ { ? } ^ { \top } } > 0$ is the query’s (input-independent) self-weight; $a _ { j } , b _ { j }$ are the two columns of the folded read-out $Y _ { j } = V _ { j } W _ { O } ^ { \top }$ (for outputs 0 and 1). By Equation 1, a one-layer �-head transformer solves (�+1)-parity if

$$
( - 1 ) ^ { \sum _ { 1 \leq i \leq k + 1 } x _ { i } } \cdot \sum _ { j = 1 } ^ { k } \frac { n _ { x } ^ { j } } { d _ { x } ^ { j } } \cdot ( a _ { j } - b _ { j } ) \ \geq \ 0 \qquad \mathrm { f o r ~ e v e r y ~ } x \in \{ 0 , 1 \} ^ { k + 1 } ,
$$

with strict inequality whenever the parity is 1.

Theorem 1. There is no choice of vectors $v _ { ? } , v _ { 0 , i } , v _ { 1 , i } , a _ { j } , b _ { j }$ , and positive reals $r _ { 0 , i } ^ { j } , r _ { 1 , i } ^ { j } , r _ { ? } ^ { j } , f o r 1 \leq i \leq k + 1$ ， $1 \leq j \leq k _ { \mathrm { { i } } }$ , such that for every $x \in \{ 0 , 1 \} ^ { k + 1 }$ the following holds:

$$
( - 1 ) ^ { \sum _ { 1 \leq i \leq k + 1 } x _ { i } } \cdot \sum _ { j = 1 } ^ { k } { \frac { n _ { x } ^ { j } } { d _ { x } ^ { j } } } \cdot ( a _ { j } - b _ { j } ) \ \left\{ \begin{array} { l l } { > 0 } & { \ i f \sum _ { 1 \leq i \leq k + 1 } x _ { i } \ i s \ o d d \ ( t a r g e t \ b i t \ = 1 ) , } \\ { \geq 0 } & { \ i f \sum _ { 1 \leq i \leq k + 1 } x _ { i } \ i s \ e v e n \ ( t a r g e t \ b i t \ = 0 ) . } \end{array} \right.
$$

Proof. Recall that $\sigma _ { x } = ( - 1 ) ^ { \sum _ { i = 1 } ^ { k + 1 } x _ { i } }$

Step 1: Clear the denominators. Set

$$
s _ { x } ^ { j } : = n _ { x } ^ { j } \cdot ( a _ { j } - b _ { j } ) = r _ { ? } ^ { j } \big ( v _ { ? } \cdot ( a _ { j } - b _ { j } ) \big ) + \sum _ { i = 1 } ^ { k + 1 } r _ { x _ { i } , i } ^ { j } \big ( v _ { x _ { i } , i } \cdot ( a _ { j } - b _ { j } ) \big ) .
$$

Each of $s _ { x } ^ { j }$ and $d _ { x } ^ { j }$ is therefore a sum of terms depending on at most one input bit: the query self-terms are constant, while the �th summands depend only on $x _ { i } .$ . Define $\textstyle \Delta ( x ) : = \sum _ { j = 1 } ^ { k } s _ { x } ^ { j } / d _ { x } ^ { j }$

The inequality at � reads $\sigma _ { x } \Delta ( x ) \ge 0$ , with strict inequality whenever $\sigma _ { x } = - 1$ . Since each $d _ { x } ^ { j } > 0$ multiplying through by the strictly positive quantity $\textstyle \prod _ { j = 1 } ^ { k } d _ { x } ^ { j }$ preserves sign:

$$
\mathrm { s i g n } ( \Delta ( x ) ) = \mathrm { s i g n } ( F _ { x } ) , \qquad F _ { x } : = \prod _ { j = 1 } ^ { k } d _ { x } ^ { j } \Delta ( x ) = \sum _ { j = 1 } ^ { k } s _ { x } ^ { j } \prod _ { j ^ { \prime } \neq j } d _ { x } ^ { j ^ { \prime } } .
$$

The theorem is therefore equivalent to the assertion: no choice of parameters makes $\sigma _ { x } F _ { x } \geq 0$ hold for every $x \in \{ 0 , 1 \} ^ { k + 1 }$ , with strict inequality whenever $\sigma _ { x } = - 1$

Step 2: Alternating-sum identity.

Claim 1. For every choice of the parameters, $\begin{array} { r } { \sum _ { x \in \{ 0 , 1 \} ^ { k + 1 } } \sigma _ { x } F _ { x } = 0 } \end{array}$

Proof of claim. Expand $F _ { x }$ as a sum of monomials. Each term in the defining sum has the form $s _ { x } ^ { j } \prod _ { j ^ { \prime } \neq j } d _ { x } ^ { j ^ { \prime } }$ and we further expand each factor using

$$
\begin{array} { l } { { \displaystyle s _ { x } ^ { j } = r _ { ? } ^ { j } \big ( \boldsymbol { v } _ { ? } \cdot ( \boldsymbol { a } _ { j } - \boldsymbol { b } _ { j } ) \big ) + \sum _ { i = 1 } ^ { k + 1 } r _ { x _ { i } , i } ^ { j } \big ( \boldsymbol { v } _ { x _ { i } , i } \cdot ( \boldsymbol { a } _ { j } - \boldsymbol { b } _ { j } ) \big ) } , }  \\ { { \displaystyle d _ { x } ^ { j ^ { \prime } } = r _ { ? } ^ { j ^ { \prime } } + \sum _ { i = 1 } ^ { k + 1 } r _ { x _ { i } , i } ^ { j ^ { \prime } } } . } \end{array}
$$

A typical monomial � is a product of exactly � factors—one from $s _ { x } ^ { j }$ and one from each of the $k - 1$ remaining denominators. Each factor is either a query constant $( r _ { ? } ^ { j } ( { v } _ { ? } \cdot ( a _ { j } - b _ { j } ) )$ or $r _ { ? } ^ { j ^ { \prime } } )$ or a single-position term $( r _ { x _ { i } , i } ^ { j } ( v _ { x _ { i } , i } \cdot ( a _ { j } - b _ { j } ) ) \mathrm { o r } r _ { x _ { i } , i } ^ { j ^ { \prime } } )$ . Hence � is a product of � scalars, each depending on at most one binary coordinate, so � depends on at most � of the $k + 1$ coordinates $x _ { 1 } , \ldots , x _ { k + 1 }$ . Since $k < k + 1$ , there exists at least one index $i ^ { * } \in \left\{ 1 , \ldots , k + 1 \right\}$ } on which � does not depend. Splitting the sum over � by isolating the free coordinate $x _ { i ^ { * } }$

$$
\sum _ { \stackrel { x \in \{ 0 , 1 \} ^ { k + 1 } } { l \not = i ^ { * } } } \sigma _ { x } \cdot m = \left( \sum _ { \stackrel { x _ { l } \in \{ 0 , 1 \} } { l \not = i ^ { * } } } ( - 1 ) ^ { \sum _ { l \not = i ^ { * } } x _ { l } } m \right) \cdot \underbrace { \sum _ { \stackrel { x _ { i ^ { * } } \in \{ 0 , 1 \} } { = 0 } } ( - 1 ) ^ { x _ { i ^ { * } } } \ = \ 0 . } _ { = 0 }
$$

Since every monomial of $F _ { x }$ vanishes individually under the alternating sum, so does: $\begin{array} { r } { \sum _ { x } \sigma _ { x } F _ { x } = 0 } \end{array}$ □

Step 3: Contradiction. Suppose, toward a contradiction, that the required inequalities hold simultaneously: for every $x \in \{ 0 , 1 \} ^ { k + 1 } , \sigma _ { x } F _ { x } \ \geq \ 0 _ { }$ , with strict inequality whenever $\sigma _ { x } = - 1$ (equivalently, whenever the target bit is 1). Since parity is balanced on $\{ 0 , 1 \} ^ { k + 1 }$ , there are $2 ^ { k } > 0$ inputs with odd Hamming weight, so at least one term $\sigma _ { x } F _ { x }$ is strictly positive; every other term is nonnegative by hypothesis. Hence

$$
\sum _ { x \in \{ 0 , 1 \} ^ { k + 1 } } \sigma _ { x } F _ { x } > 0 ,
$$

contradicting the identity $\begin{array} { r } { \sum _ { x } \sigma _ { x } F _ { x } = 0 } \end{array}$ established in Step 2. Therefore no choice of parameters can satisfy all $2 ^ { k + 1 }$ inequalities, completing the proof.

Remark 1. The same alternating-sum argument establishes a more general parity obstruction. Consider any system of $2 ^ { P }$ inequalities indexed by $i = ( i _ { 1 } , \dotsc , i _ { P } ) \in \{ 0 , 1 \} ^ { P }$ , with required signs $( - 1 ) ^ { i _ { 1 } + \dots + i _ { P } }$ , of the form

$$
( - 1 ) ^ { i _ { 1 } + \cdots + i _ { P } } \cdot \sum _ { j = 1 } ^ { H } \frac { s _ { i } ^ { j } } { d _ { i } ^ { j } } \ > \ 0 ,
$$

in which each $s _ { i } ^ { j }$ and each $d _ { i } ^ { j } > 0$ is a sum of real-valued summands, each depending on at most one of the binary indices. Whenever $P > H _ { : }$ , the system is infeasible: defining $\begin{array} { r } { F _ { i } = \sum _ { j } s _ { i } ^ { j } \prod _ { j ^ { \prime } \neq j } d _ { i } ^ { j ^ { \prime } } } \end{array}$ , each monomial in $F _ { i }$ is a product of � such summands and therefore involves at most $H < P$ of the binary indices, leaving at least one free index to kill the alternating sum, and the same contradiction in Step 3 closes the argument. The condition $P > H$ is tight: at $P = H _ { : }$ , a single monomial of $F _ { i }$ may collectively involve all � binary indices.

In the present setting, $P = k + 1$ binary indices (the input bits) and $H = k$ heads, so $P > H$ by exactly one and the theorem follows.

Beyond parity. The lower bound extends to other problems that meet the hypotheses of Remark 1: � heads cannot solve the �-ESP endpoint-selection task of [TKRS26], which �+1 heads solve exactly (Sections B and B.3), nor the (�+1)-hop induction-heads task of [SHT24b] (Section C).

## 3.2 Matching upper bound: � heads compute �-bit parity

We now match the lower bound. For �-bit parity the input is the (�+1)-token string $x _ { 1 } x _ { 2 } \cdots x _ { k } ?$ with target $\textstyle \bigoplus _ { i = 1 } ^ { k } x _ { i } ;$ by the criterion (1) of Section 2, to prove the upper bound, it sufices to construct parameters such that

$$
\begin{array} { r l r } { ( - 1 ) ^ { \sum _ { i = 1 } ^ { k } x _ { i } } } & { \underbrace { \displaystyle \sum _ { j = 1 } ^ { k } \frac { n _ { x } ^ { j } } { d _ { x } ^ { j } } \cdot ( a _ { j } - b _ { j } ) } _ { = \Delta ( x ) = \displaystyle \sum _ { j = 1 } ^ { k } s _ { x } ^ { j } / d _ { x } ^ { j } } } & { > 0 \qquad \mathrm { f o r ~ e v e r y ~ } x \in \{ 0 , 1 \} ^ { k } . } \end{array}
$$

The lower bound (Theorem $1 , k - 1$ heads against � bits) rules out $k - 1$ heads; we show � sufice, even in embedding dimension $d = k + 3$

Theorem 2. For every $k \geq 1$ there is a choice of embedding Φ, matrix $W _ { O }$ , and attention and value matrices $A _ { j } , V _ { j }$ such that the one-layer �-head attention-only transformer above solves �-bit parity.

Proof sketch. The proof has three main ideas. First, make each head’s denominator equal to $c + j$ where $\textstyle c = \sum _ { i = 1 } ^ { k } x _ { i }$ is the Hamming weight of the input. Second, use the value matrices $V _ { j }$ to make each head’s numerator gap equal to an afine function $\alpha _ { j } + \beta _ { j } c$ . Third, choose the coeficients so that the sum $\begin{array} { r } { \Delta ( c ) = \sum _ { j = 1 } ^ { k } ( \alpha _ { j } + \beta _ { j } c ) / ( c + j ) } \end{array}$ has sign $( - 1 ) ^ { c }$ for every $c = 0 , \ldots , k$ . Since the required signed gap is $( - 1 ) ^ { c }$ this solves parity.

The supporting partial-fractions lemma and the full construction are deferred to Section A.

Remark 2. The head count is tight. The matching lower bound rules out $k - 1$ heads for �-parity, and the construction attains � using an additive token-plus-position embedding. The value/output gap gives diferent weights to token 0 and token 1, so each head has an afine numerator in the total Hamming weight �. Summing � afine-over-linear terms gives a rational function whose sign alternates on all $k + 1$ Hamming levels $0 , \ldots , k$

## 4 Compactness: Dimension and Precision Are Bounded by the Discrete Data

The lower bounds of this paper charge the model for heads. A natural question to ask is whether for a given number of heads, the capabilities of the transformer can greatly increase with the power ofered by the other two continuous resources: say, an enormous embedding dimension $d ,$ or unbounded arithmetic precision. This section rules that out. We show that if a function is computable by the one-layer �-attention-only model at all, it is computable with both resources bounded by the discrete data of the problem: the head count �, the number of output labels �, the number of token values $q = | \Sigma |$ , and the sequence length �. The key idea behind this compactness result is a single observation: the model is a bilinear (inner-product) machine, and the function it computes reads its parameters of one rectangular matrix of inner products.

The argument has two independent parts. First, the computation factors through a finite cross-Gram matrix, whose rank bounds the required dimension. Second, correctness is described by finitely many strict polynomial inequalities; quantitative rational approximation and stability then bound the bit-length of a realizing model.

We state the results for the model of Definition 1 generalized to a value alphabet $\Sigma ( q = | \Sigma | )$ and � output labels (output matrix $W _ { O } \in \mathbb { R } ^ { L \times d _ { \mathrm { o u t } } }$ , folded $Y _ { j } = V _ { j } W _ { O } ^ { \top } \in \mathbb { R } ^ { d \times L }$ with columns $y _ { j } ^ { ( 1 ) } , \ldots , y _ { j } ^ { ( L ) }$ ; in the binary case, $q = L = 2 , N = n + 1$ , and the columns are indexed by labels 0 and 1). Write $\xi _ { j } = v _ { ? } A _ { j }$ for the efective query, $p _ { t } ^ { j } = \xi _ { j } \cdot v _ { t }$ for the score of token type � (an occurring (value, position) pair, with embedding $v _ { t } )$ , and $w _ { t } ^ { j , l } = v _ { t } \cdot y _ { i } ^ { ( l ) }$ for its value-projection to label �.

For a rational number $z = a / b$ in lowest terms, let its bit-length be the maximum of the binary lengths of � and �. The numerical precision of a rational model realization is the maximum bit-length of an entry of its embedding table and weight matrices $\Phi , \{ A _ { j } , V _ { j } \} _ { j \in [ k ] } , W _ { O }$

## 4.1 The computation factors through a cross-Gram matrix

Lemma 1 (Factoring lemma). Fix a reference label ★ and set

$$
\mathcal { A } = \{ \xi _ { j } \} _ { j \in [ k ] } \cup \{ y _ { j } ^ { ( l ) } - y _ { j } ^ { ( \star ) } \} _ { j \in [ k ] , l \neq \star } , \qquad \mathcal { B } = \{ v _ { t } : t a n o c c u r r i n g t o k e n t y p e \} ,
$$

so $| \mathcal { R } | = k L$ and $| \mathcal { B } | \leq q N$ . The function computed depends on the parameters $\Phi , \{ A _ { j } \} , \{ V _ { j } \} , W _ { O }$ only through the cross-inner-product matrix

$$
M \in \mathbb { R } ^ { \mathcal { A } \times \mathcal { B } } , \qquad M _ { a , t } = a \cdot v _ { t } .
$$

No token–token product $\boldsymbol { v } _ { t } \cdot \boldsymbol { v } _ { t ^ { \prime } }$ , and no product within ${ \mathcal { A } } ,$ ever enters.

Proof. Only logit diferences decide the prediction. Each is built from the softmax weights and the weighted value sums over the token types occurring in the input; the former are functions of the scores $p _ { t } ^ { j } = \xi _ { j } \cdot v _ { t }$ alone, the latter of $w _ { t } ^ { j , l } - w _ { t } ^ { j , \star } = ( y _ { j } ^ { ( l ) } - y _ { j } ^ { ( \star ) } ) \cdot v _ { t }$ . Both families are exactly the entries of �. □

Remark 3 (Why the model is “bilinear”). The lemma says the geometry the function can see is one-sided: it reads how each parameter vector $a \in \mathcal { A }$ pairs against each token embedding $\boldsymbol { v } _ { t } \in \mathcal { B }$ , but never how the $v _ { t }$ sit relative to one another, nor how the � do. Dimension and precision are properties of that ambient geometry; the function only reads a single rectangular slice of it whose size depends on other model parameters; hence unbounded dimension or precision does not add more to the expressivity of the transformer.

## 4.2 Dimension: if realizable, realizable in dimension rank $( M ) + 1$

Theorem 3 (Dimension compactness). Every realization has $d \geq { \mathrm { r a n k } } ( M )$ , and any prescribed cross-Gram matrix � is realizable in dimension $r + 1 _ { : }$ , where $r = { \mathrm { r a n k } } ( M )$ . Hence the least embedding dimension computing � satisfies

$$
\operatorname { r a n k } ( M ^ { \star } ) \ \leq \ d ^ { \star } ( f ) \ \leq \ \operatorname { r a n k } ( M ^ { \star } ) + 1 , \qquad M ^ { \star } \in \ \operatorname { a r g m i n } \ \operatorname { r a n k } ( M ) ,
$$

and in particular $d ^ { \star } ( f ) \leq \operatorname* { m i n } \{ k L , q N \} + 1 ;$ for binary output, $d ^ { \star } \le \operatorname* { m i n } \{ 2 k , q N \} + 1$

Proof deferred to Section A.

Remark 4. For the tasks studied in the paper (binary $L = 2 , q = 2 , N = \Theta ( k ) )$ Theorem 3 yields $d ^ { \star } \leq$ min $( 2 k , 2 N ) + 1 = O ( k )$ , matching the linear bound on the dimension used by all constructions in the paper.

## 4.3 Precision: if realizable, realizable with bounded-bit rational model parameters

To expose polynomial constraints, introduce the auxiliary variables $r _ { t } ^ { j } = e ^ { p _ { t } ^ { j } } > 0$ and $w _ { t } ^ { j , l } = v _ { t } \cdot y _ { j } ^ { ( l ) }$ . Clearing the strictly positive softmax denominators turns each pairwise decision $\mathrm { ^ { * } l o g i t } _ { f ( x ) } ( x ) > \mathrm { l o g i t } _ { l ^ { \prime } } ( x ) ^ { * }$ into a strict polynomial inequality $P _ { x , l ^ { \prime } } ( r , w ) > 0$ of degree at most $k + 1$ in $n _ { \mathrm { v a r } } : = O ( k L q N )$ variables, with integer coeficients of bit-length $O ( k \log ( N + 1 ) )$ . A rational-point bound first controls $( r , w )$ . We then quantify the distance of this rational point from every decision boundary, approximate each log-score $p _ { t } ^ { j } = \log r _ { t } ^ { j }$ by a rational number within that distance, and realize the resulting rational score and projection tables by rational embeddings and weight matrices.

Theorem 4 (Precision compactness). If � is computable at all, then it is computable by a one-layer �-head attention-only transformer whose embedding table and weight matrices are rational and have bit-length

$$
\widehat { \beta } \le { \cal O } \big ( k \log ( N + 1 ) \big ) \cdot ( k + 1 ) ^ { { \cal O } ( n _ { \mathrm { v a r } } ) } = 2 ^ { { \cal O } ( n _ { \mathrm { v a r } } \log ( k + 1 ) ) } , \qquad n _ { \mathrm { v a r } } = { \cal O } ( k L q N ) .
$$

The realization may be taken in embedding dimension at most $q N + 1$

Proofdeferred to Section A.

Thus, every computable function has a realization whose embedding dimension and numerical precision are both bounded in terms of the discrete parameters of the task.

Remark 5 (Additive embedding: $q N  q + N )$ . For functions that are additively realizable $( \Phi ( b , i ) = \tau _ { b } + \pi _ { i } )$ the dimension bound sharpens to $d ^ { \star } \le \operatorname* { m i n } \{ k L , q + N \} + 1$ . The standard-parameter precision theorem above continues to apply with the general count $n _ { \mathrm { v a r } } = O ( k L q N )$ ; obtaining the sharper count $O ( k L ( q + N ) )$ while preserving additivity would require a separate generator-level semialgebraic argument, which we do not claim here. Indeed each cross-Gram entry then splits, $M _ { a , ( b , i ) } = a \cdot \tau _ { b } + a \cdot \pi _ { i } .$ so � factors through only the � value-vectors $\{ \tau _ { b } \}$ and the � position-vectors $\left\{ \pi _ { i } \right\}$ ; the rank reconstruction of Theorem 3, applied to the $k L \times ( q + N )$ matrix over $\{ \tau _ { b } \} \cup \{ \pi _ { i } \}$ , returns the value- and position-vectors separately, so setting $\Phi ^ { \prime } ( b , i ) = \tau _ { b } ^ { \prime } + \pi _ { i } ^ { \prime }$ preserves additivity for free.

## 5 Universal Bounds for General Binary Functions

## 5.1 Universal Upper Bound: the Monomial Vote

With enough heads a single attention layer computes an arbitrary Boolean function, and it does so even under the restricted additive embedding $\Phi ( b , i ) = \tau _ { b } + \pi _ { i } ,$ , using one head per monomial of the target’s multilinear expansion, in embedding dimension $O ( n )$ and with $O ( n )$ -bit parameters. This is the construction we use to bracket the hierarchy from above. Throughout, � is the input arity: $f : \{ 0 , 1 \} ^ { n }  \{ 0 , 1 \}$ , presented as in Definition 1 with an additive Φ.

The multilinear expansion. Every $f : \{ 0 , 1 \} ^ { n }  \{ 0 , 1 \}$ agrees on the cube with its unique multilinear (Möbius) expansion

$$
f ( x ) = \sum _ { S \subseteq [ n ] } \alpha _ { S } \prod _ { i \in S } x _ { i } , \qquad \alpha _ { S } = \sum _ { T \subseteq S } ( - 1 ) ^ { | S | - | T | } f ( \mathbf { 1 } _ { T } ) \in \mathbb { Z } ,
$$

where $\mathbf { 1 } _ { T } \in \{ 0 , 1 \} ^ { n }$ is the indicator of $T \subseteq [ n ]$ . The coeficients are integers with $| \alpha _ { S } | \le 2 ^ { | S | }$ , so $\textstyle \sum _ { S } | \alpha _ { S } | \leq$ $\begin{array} { r } { \sum _ { S } 2 ^ { | S | } = 3 ^ { n } } \end{array}$ . Crucially, the monomials $\textstyle \prod _ { i \in S } x _ { i }$ are monotone (positive-literal) terms — “every bit of � is on” — and a monotone term is exactly the primitive a single additive head detects sharply, using the query token itself as its reference (so no auxiliary token is added to the input).

Lemma 2 (Monotone-term head). For every $S \subseteq [ n ]$ and every $R > 0$ there is a single additive head, of embedding dimension $O ( n )$ and �(�)-bit parameters, whose scalar query read-out $A _ { S } ( x )$ satisfies

$$
\begin{array} { r } { \left| A _ { S } ( x ) - \prod _ { i \in S } x _ { i } \right| \ \le \ n e ^ { - R / 2 } \qquad f o r e \nu e r y x \in \{ 0 , 1 \} ^ { n } . } \end{array}
$$

Proof deferred to Section A.

Theorem 5 (Monomial vote). Every $f : \{ 0 , 1 \} ^ { n }  \{ 0 , 1 \}$ is computed by an additive transformer with

$$
k \leq 2 ^ { n } h e a d s , d = O ( n ) , O ( n ) - b i t p a r a m e t e r s .
$$

Proof deferred to Section A.

## 5.2 A Universal Lower Bound: Almost All Functions Need $\Omega ( 2 ^ { n } / n ^ { 2 } )$ Heads

The monomial vote (Theorem 5) computes any $f : \{ 0 , 1 \} ^ { n }  \{ 0 , 1 \}$ with at most $2 ^ { n }$ heads. We now show this is essentially unavoidable: a $1 - o ( 1 )$ fraction of all Boolean functions require $\Omega ( 2 ^ { n } / n ^ { 2 } )$ heads, so the two bounds meet to within a poly(�) factor. The argument is a counting one (in the tradition of Shannon’s bound that almost all Boolean functions require large circuits [Sha49]), and, crucially, it holds for the general joint embedding, even with unbounded embedding dimension and unbounded numerical precision. Neither of those two analog resources can substitute for heads.

## 5.2.1 Each input’s decision is one polynomial sign in $O ( k n )$ variables

Fix a �-head transformer with a joint embedding and use the afine scalar convention of Definition 2. For head �, put $c _ { j } : = V _ { j } \omega \in \mathbb { R } ^ { d }$ and define

$$
r _ { b , i } ^ { j } : = e ^ { v _ { j } A _ { j } v _ { b , i } ^ { \top } } > 0 , \qquad u _ { b , i } ^ { j } : = v _ { b , i } \cdot c _ { j } , \qquad r _ { ? } ^ { j } : = e ^ { v _ { ? } A _ { j } v _ { ? } ^ { \top } } > 0 , \qquad u _ { ? } ^ { j } : = v _ { ? } \cdot c _ { j } .
$$

Then

$$
d _ { x } ^ { j } : = r _ { ? } ^ { j } + \sum _ { i = 1 } ^ { n } r _ { x _ { i } , i } ^ { j } > 0 , \qquad s _ { x } ^ { j } : = r _ { ? } ^ { j } u _ { ? } ^ { j } + \sum _ { i = 1 } ^ { n } r _ { x _ { i } , i } ^ { j } u _ { x _ { i } , i } ^ { j } ,
$$

and the decision statistic is

$$
G ( x ) : = \sum _ { j = 1 } ^ { k } \frac { s _ { x } ^ { j } } { d _ { x } ^ { j } } - \theta , \qquad \mathrm { n e t } ( x ) = 1 \{ G ( x ) > 0 \} .
$$

Lemma 3 (Dimension- and precision-free reduction). The function computed by a �-head afine-scalar transformer depends on its parameters $( \Phi , \{ A _ { j } , V _ { j } \} , \omega , \theta )$ only through the at most $4 k n + 2 k$ reals

$$
\{ r _ { b , i } ^ { j } , u _ { b , i } ^ { j } \} _ { b \in \{ 0 , 1 \} , i \in [ n ] , j \in [ k ] } \cup \{ r _ { ? } ^ { j } , u _ { ? } ^ { j } \} _ { j \in [ k ] } ,
$$

together with �. This list is independent ofembedding dimension, value $w i d t h ,$ and any precision bound.

Proof. The displayed formulas reconstruct every $s _ { x } ^ { j } , d _ { x } ^ { j }$ , hence $G ( x )$ , from the listed scalars. The triples $( b , i , j )$ contribute 2�� weights and 2�� scalar values, and the query contributes two more scalars per head. The ambient dimensions occur only inside the defining products and disappear after this reduction. □

Collect these numbers into a single parameter vector $y \in \mathbb { R } ^ { v } , v = O ( k n )$ (at most 4�� + 2� weights and scalar values, plus �). Because every $d _ { x } ^ { j } > 0 ;$ , multiplying $G ( x ) > 0$ through by the positive product $\textstyle \prod _ { j } d _ { x } ^ { j }$ is sign-preserving, so

$$
\mathrm { n e t } ( x ) = 1 \{ P _ { x } ( y ) > 0 \} , \qquad P _ { x } ( y ) = \sum _ { j = 1 } ^ { k } s _ { x } ^ { j } \biggl | \underset { j ^ { \prime } \neq j } { d } \biggr | _ { x } ^ { j ^ { \prime } } - \theta \prod _ { j = 1 } ^ { k } d _ { x } ^ { j } .\tag{2}
$$

Each $d _ { x } ^ { j }$ is linear in � and each $s _ { x } ^ { j }$ is quadratic (a sum of products $r \cdot u )$ , so a term $s _ { x } ^ { j } \prod _ { j ^ { \prime } \neq j } d _ { x } ^ { j ^ { \prime } }$ has degree $2 + ( k - 1 ) = k + 1$ in �, and $\textstyle \theta \prod _ { j } d _ { x } ^ { j }$ has degree $k + 1 ;$ hence $\deg _ { y } P _ { x } \leq k + 1$ . Two points make this a genuine relaxation, which is all the counting needs. First, we impose no constraint on � beyond $r _ { b , i } ^ { j } > 0$ (in particular, we do not require the additive tie of a token-plus-position embedding), so the argument covers the joint model (and additive as a special case). Second, letting � range over all of R<sup>�</sup> can only enlarge the set of achievable decision patterns, and $v = O ( k n )$ is independent of� and of any precision bound.

## 5.2.2 A small polynomial system has few sign patterns

A transformer’s entire input–output behavior is the vector $( \mathrm { n e t } ( x ) ) _ { x \in \{ 0 , 1 \} ^ { n } }$ , read of by (2) from the signs of the $2 ^ { n }$ polynomials $P _ { x }$ at the single point �. How many such sign vectors are possible as � ranges over $\mathbb { R } ^ { v _ { ? } }$ This is bounded by a classical result of Warren [War68], refining the Milnor–Thom bound [Mil64, Tho65] on the number of connected components of a real algebraic set.

Theorem 6 (Warren’s sign-pattern bound). Let $P _ { 1 } , \ldots , P _ { m }$ be real polynomials in � real variables, each of degree at most �, with � $\geq v \geq 1$ . The number of distinct nonzero sign vectors

$$
\bigl ( \operatorname { s g n } P _ { 1 } ( y ) , \ldots , \operatorname { s g n } P _ { m } ( y ) \bigr ) \in \{ - 1 , + 1 \} ^ { m } , \qquad y \in \mathbb { R } ^ { v } ,
$$

that actually occur is at most $( C D m / v ) ^ { v }$ for an absolute constant �. The bound depends only on $m , D , v ,$ , never on the coeficients; in particular, it presupposes no bound on precision.

## 5.2.3 Counting the realizable functions

Theorem 7 (Universal lower bound). The number ofBoolean functions $f : \{ 0 , 1 \} ^ { n }  \{ 0 , 1 \}$ computable by a one-layer �-attention-only transformer (with a joint embedding, at any dimension and any precision) is at most $2 ^ { O ( k n ^ { 2 } ) }$ . Consequently a $1 - o ( 1 )$ fraction of all $2 ^ { 2 ^ { n } }$ Boolean functions require

$$
k \ = \ \Omega \Big ( \frac { 2 ^ { n } } { n ^ { 2 } } \Big ) h e a d s ,
$$

even with unbounded embedding dimension and unbounded precision.

Proof. Before applying Warren’s bound, we may assume that every decision is strict. Given a realization, replace � by $\theta + \varepsilon ,$ , where

$$
0 < \varepsilon < \operatorname* { m i n } _ { x : f ( x ) = 1 } G ( x )
$$

when $f ^ { - 1 } ( 1 ) \neq \emptyset$ , and take any $\varepsilon > 0$ otherwise. The finitely many positive margins remain positive, while every input previously satisfying $G ( x ) \leq 0$ becomes strictly negative. Thus the computed function is unchanged and $P _ { x } ( y ) \neq 0$ for every input �.

Apply Theorem 6 to the system $\{ P _ { x } \} _ { x \in \{ 0 , 1 \} ^ { n } }$ of (2): here $m = 2 ^ { n }$ , the degree is $D \leq k + 1$ , and the number of variables is $v = O ( k n )$ . We split into two cases. If $m < v _ { ; }$ , then the trivial bound by the total number of Boolean functions gives

$$
\# \{ { \mathrm { r e a l i z a b l e } } f \} \leq 2 ^ { m } \leq 2 ^ { O ( k n ^ { 2 } ) } ,
$$

where the second inequality follows from $m < v = O ( k n )$ . We may therefore assume that $m \geq v$ , in which case Theorem 6 applies. Since net(�) is determined by sgn $P _ { x } ( y )$ , two transformers computing diferent functions must realize diferent sign vectors, and hence

$$
\begin{array} { r } { \# \{ \mathrm { r e a l i z a b l e } f \} \ \leq \ \left( \frac { C ( k + 1 ) 2 ^ { n } } { v } \right) ^ { v } \ = \ 2 ^ { O ( v n ) } \ = \ 2 ^ { O ( k n ^ { 2 } ) } , } \end{array}
$$

using $v = O ( k n )$ and, since $\begin{array} { r } { k + 1 \leq 2 v \leq 2 m , \log \frac { C ( k + 1 ) 2 ^ { n } } { v } \leq n + O ( 1 ) = O ( n ) } \end{array}$ . Thus, in both cases, the number of realizable Boolean functions is at most $2 ^ { O ( k n ^ { 2 } ) }$ . The bound holds for all $\boldsymbol { y } \in \mathbb { R } ^ { \boldsymbol { v } }$ , hence even when precision and embedding dimension are unbounded, and (by the relaxation above) the joint embedding. Write the bound as $2 ^ { c k n ^ { 2 } }$ . To realize even a $2 ^ { - o ( 2 ^ { n } ) }$ fraction of the $2 ^ { 2 ^ { n } }$ functions one needs $2 ^ { c k n ^ { 2 } } \geq 2 ^ { 2 ^ { n } / 2 }$ , i.e. $k \geq 2 ^ { n } / ( 2 c n ^ { 2 } ) = \Omega ( 2 ^ { n } / n ^ { 2 } )$ . Contrapositively, if $\textstyle k < { \frac { 1 } { 2 c } } 2 ^ { n } / n ^ { 2 }$ then at most $2 ^ { 2 ^ { n } / 2 }$ functions are realizable (a $2 ^ { - \Omega ( 2 ^ { n } ) }$ fraction of the total), so a $1 - o ( 1 )$ fraction of all � need $\Omega ( 2 ^ { n } / n ^ { 2 } )$ heads. □

Remark 6 (The bounds meet; head count is the sole resource). The additive upper bound (Theorem $5 , \leq 2 ^ { n }$ heads) and this joint lower bound $\textstyle ( \Omega ( 2 ^ { n } / n ^ { 2 } )$ heads) bracket the head-complexity of a generic Boolean function to within poly(�). Because the upper bound is additive and the lower bound is joint, and additive $\subseteq { \mathrm { j o i n t } } ,$ , the two meet for both embedding regimes at once. Neither the embedding dimension (which the constructions keep at $O ( n ) )$ nor the precision $( O ( n )$ bits) is ever the bottleneck, since Lemma 3 and Theorem 6 make the lower bound blind to both. The one poly(�) gap that remains — closing $\Omega ( 2 ^ { n } / n ^ { 2 } )$ up to the trivial $2 ^ { n }$ from below — is a genuine open problem: a single layer has no composition, so heads cannot reuse sub-computations the way a Lupanov-type circuit does.

## 6 Conclusion

We have made the number of attention heads a precise complexity measure for a single self-attention layer. Three results support the claim: heads form an exact, per-head hierarchy pinned by parity (Theorems 1 and 2); the two continuous resources, embedding dimension and numerical precision, provably collapse to the discrete data of the task (Theorems 3 and $4 ) ;$ and the head complexity of a generic �-bit function is $2 ^ { n } / \mathrm { p o l y } ( n )$ (Theorem 7). What the measure still hides are two explicit-function questions, one within a layer and one across layers, each asking for a named function that provably outruns a bounded budget of heads. Parity is inexpensive in this model, motivating the search for explicit functions with large head complexity.

An explicit hard function for one layer. Our universal lower bound is a counting argument, in the tradition of Shannon’s circuit-size bound [Sha49]: almost every � has head complexity $2 ^ { n } / \mathrm { p o l y } ( n )$ , yet the proof produces no particular witness. It remains open to exhibit an explicit, natural family with exponential head complexity, thereby complementing the counting lower bound with a concrete witness. A concrete first step is quantitative: our lower and upper bounds pin the worst case only to within $\mathsf { a  p o l y } ( n )$ factor $( 2 ^ { n } / n ^ { 2 }$ versus $2 ^ { n } )$ , and closing that gap would fix the exact worst-case head complexity of a single layer.

An explicit function that forces depth. The hierarchy of this paper varies one resource, heads, at a fixed depth of one; the sharper question varies depth, asking for the analogue of parity across layers. Is there an explicit family $f _ { n }$ computable by an attention network of �(1) (or $O ( \log n ) )$ layers with poly(�) heads per layer, yet requiring $2 ^ { \Omega ( n ) }$ heads in any single layer? Single-layer attention ofers no obvious mechanism for composing or reusing intermediate results, and Theorem 1 gives one exact manifestation of this limitation. Such a separation would clarify when additional depth can reduce head requirements. Unlike the counting bound above, it would name the obstruction.

The two problems are one demand in diferent arenas: a specific function that provably breaks a fixed head budget. We have shown the budget is real and exactly ordered; naming the functions that break it is the natural next target.

## 7 Related Work

Attention heads as a resource. Multi-head attention was introduced to let diferent heads attend to diferent relations in paralle $[ \mathrm { V S P ^ { + } 1 7 } ]$ , and the QK/OV view of attention-only transformers gives a useful lens for studying such heads as separate computational units $\left[ \mathrm { E N O ^ { + } 2 1 } \right]$ . Mechanistic work on induction heads further suggests that individual heads implement recognizable algorithmic subroutines $\left[ \mathrm { O E N ^ { + } 2 2 } \right]$ Prior theoretical work has studied limitations of pure attention with depth [DCL21], memorization capacity as a function of the number of heads [MLT24], and approximation-theoretic phase transitions in head count for one-layer transformers with feed-forward blocks $[ \mathrm { Y J B ^ { + } 2 6 } ]$ . Closest in spirit to our work is the two-head separation for ESP [TKRS26], which shows that two narrow heads can solve tasks that one arbitrarily wide head cannot. Our results extend this head-count viewpoint from the 1-vs.-2 case to an exact $k \mathrm { - v s . } \ – ( k + 1 )$ hierarchy.

Parity and formal-language limitations. Parity is a classical obstruction for shallow models and loworder representations [MP69, PS94]. In transformers, parity and related formal languages have been used as canonical tests of expressivity and learnability $\mathrm { [ B A G 2 0 , S M W ^ { + } 2 4 ] }$ . Hahn proved length-asymptotic limitations of self-attention on parity-like languages [Hah20], while Chiang and Cholak showed that additional depth and feed-forward computation can overcome such limitations [CC22]. Similarly, transformers with feed-forward computation can learn counting-style shortcuts to automata, including parity-like behavior, by first aggregating counts and then post-processing them $\left[ \mathrm { L A G } ^ { + } 2 3 \right]$ . These results are consistent with our setting: our hierarchy is for a self-attention layer, where the number of heads is the computational resource being measured.

The closest prior parity lower bounds are those of Kozachinskiy, Steifer, and Wałęga [KSW26] and Hsu [Hsu26], discussed in Section 1.2: both establish that parity is hard for one layer, but neither gives an exact head threshold. The novelty of our result is not the qualitative hardness of parity, but the exact unit-tight head hierarchy and its matching upper bound.

One-layer lower-bound techniques. Several lower bounds for one-layer transformers use communication complexity, precision restrictions, or dimension/parameter tradeofs. Sanford, Hsu, and Telgarsky prove representational lower bounds for transformers via communication complexity [SHT23] and show that one-layer transformers need ℎ�� = Ω(�) resources for induction-head tasks when head count, dimension, and precision are all counted [SHT24a]. Peng, Narayanan, and Papadimitriou prove limitations for composition and graph-style reasoning under related resource assumptions [PNP24]. The authors in [YKD<sup>+</sup>24] study when transformers can count and identify dimension/width requirements for extracting counts from softmax ratios. Kozachinskiy et al. [KUJ<sup>+</sup>25] introduce the split-VC-dimension method for infinite-precision one-layer lower bounds; they separately propose Strassen attention as an alternative attention mechanism that solves the tasks used to establish those lower bounds. These approaches are powerful for size, precision, depth, or approximation tradeofs, but they do not yield an unconditional finite statement of the form “� heads fail, regardless of dimension and precision.”

Depth and parallelism. A complementary line of work studies depth as a computational resource. Saturated and log-precision transformers are connected to constant-depth threshold circuits [MSS22, MS23], logarithmic depth increases expressive power [SHT24b, MS25], and pointer-chasing and composition tasks yield further limitations for shallow transformer computation [PNP24]. From a learning perspective, Ye, Fu, Jia, and Sharan prove that an �-layer Disentangled Transformer can implement graph connectivity up to diameter 3<sup>�</sup>, while the training distribution determines whether gradient descent learns this algorithm or a degree-based shortcut [YFJS26].

Universality, positional encodings, and scope. Universality and Turing-completeness results for transformers rely on resources outside our exact hierarchy setting, such as additional layers, feed-forward computation, approximation rather than exact finite computation, or unbounded decoding time [YBR<sup>+</sup>20, KS24, PBM21]. Strobl, Angluin, and Frank show that concise one-layer transformers can evaluate functions when the function table is supplied in context [SAF25]; our arbitrary-function construction instead encodes the function in the weights and measures the number of heads required. Finally, work on fast attention and subquadratic alternatives studies computational eficiency rather than expressivity [AS23, AY25].

## A Deferred Proofs

This appendix collects the proofs deferred from the main text, in order of appearance.

## A.1 Equivalence of the two read-outs

Theorem 8 (The arg-max and afine scalar read-outs are equivalent on Boolean tasks). Fix an attention core producing $\mathrm { M H A } ( x ) \in \mathbb { R } ^ { d _ { \mathrm { o u t } } }$

1. Every two-logit arg-max read-out has an afine scalar read-out on the same core computing the same function; take $\omega = w _ { 1 } - w _ { 0 }$ and $\theta = 0$

2. Every afine scalar read-out has an arg-max realization computing the same function after adding one embedding coordinate and one value coordinate, but no head.

Consequently the two read-outs compute exactly the same Boolean functions at every head count.

Proof of Theorem 8. Part (1). By definition,

$$
\begin{array} { r l } & { \mathrm { o u t } ( x ) = 1 \iff Z ( x ) _ { 1 } > Z ( x ) _ { 0 } } \\ & { \qquad \iff \left. w _ { 1 } , \mathrm { M H A } ( x ) \right. > \left. w _ { 0 } , \mathrm { M H A } ( x ) \right. } \\ & { \qquad \iff \left. w _ { 1 } - w _ { 0 } , \mathrm { M H A } ( x ) \right. > 0 . } \end{array}
$$

Taking $\omega : = w _ { 1 } - w _ { 0 }$ and $\theta : = 0$ gives

$$
\mathrm { n e t } ( x ) = \mathbf { 1 } \{ \langle \omega , \mathrm { M H A } ( x ) \rangle - 0 > 0 \} = \mathrm { o u t } ( x ) \qquad \mathrm { f o r ~ e v e r y ~ } x .
$$

No coordinate is added and the core is untouched, so the LTF model computes the same function.

Part (2). We give the context one coordinate that equals the same constant on every input, and then let the arg-max read-out use that coordinate as a bias. Mark augmented objects with a superscript +.

Step 1 (embedding). Append a coordinate $d + 1$ to every token embedding, set to $1 \colon ( v _ { b , i } ) _ { d + 1 } = 1$ for all $b , i$ and $\Phi ( ? ) _ { d + 1 } = 1$ ; leave coordinates $1 , \ldots , d$ unchanged. Then every token’s augmented embedding has last coordinate 1.

Step 2 (attention and values). Extend each attention matrix to $A _ { j } ^ { + } \in \mathbb { R } ^ { ( d + 1 ) \times ( d + 1 ) }$ , equal to $A _ { j }$ on the top-left $d \times d$ block and 0 in the new row and column. Since the query and token embeddings agree with the originals on the first � coordinates and the new row/column of $A _ { j } ^ { + }$ is zero, every score is unchanged,

$$
\Phi ^ { + } ( ? ) A _ { j } ^ { + } ( X _ { i } ^ { + } ) ^ { \top } = \Phi ( ? ) A _ { j } X _ { i } ^ { \top } ,
$$

hence so are all softmax weights. Extend each value matrix to $V _ { j } ^ { + } \in \mathbb { R } ^ { ( d + 1 ) \times ( d _ { \mathrm { o u t } } + 1 ) }$ , equal to $V _ { j }$ on the top-left $d \times d _ { \mathrm { o u t } }$ block, and mapping the new input coordinate $d + 1$ to the new value coordinate $d _ { \mathrm { o u t } } + 1 ;$ thus $( X _ { i } ^ { + } V _ { j } ^ { + } ) _ { d _ { \mathrm { o u t } } + 1 } = ( X _ { i } ^ { + } ) _ { d + 1 } = 1 \mathrm { a n d } ( X _ { i } ^ { + } V _ { j } ^ { + } ) _ { 1 : d _ { \mathrm { o u t } } } = X _ { i } V _ { j }$

Step 3 (a constant coordinate). For each head the new context coordinate is a softmax average of 1’s, hence equal to 1; summing the � heads,

$$
\mathrm { M H A } ^ { + } ( x ) _ { d _ { \mathrm { o u t } } + 1 } = \sum _ { j = 1 } ^ { k } 1 = k = : C \quad \mathrm { f o r ~ e v e r y ~ } x , \qquad \mathrm { M H A } ^ { + } ( x ) _ { 1 : d _ { \mathrm { o u t } } } = \mathrm { M H A } ( x ) ,
$$

so $\mathrm { M H A ^ { + } } ( x ) = \left( \mathrm { M H A } ( x ) , C \right)$ with $C = k$ a fixed nonzero constant.

Step 4 (read-out). Take the output matrix $W _ { O } ^ { + } \in \mathbb { R } ^ { 2 \times ( d _ { \mathrm { o u t } } + 1 ) }$ with rows $w _ { 0 } ^ { + } = \mathbf { 0 }$ and $w _ { 1 } ^ { + } = \big ( \omega , - \theta / C \big )$ Then for every �,

$$
\begin{array} { r } { Z ^ { + } ( x ) _ { 1 } - Z ^ { + } ( x ) _ { 0 } = \langle w _ { 1 } ^ { + } , \mathrm { M H A } ^ { + } ( x ) \rangle = \langle \omega , \mathrm { M H A } ( x ) \rangle + \Big ( - \frac { \theta } { C } \Big ) \cdot C = \langle \omega , \mathrm { M H A } ( x ) \rangle - \theta . } \end{array}
$$

Hence ou ${ \mathrm { \Sigma } } ^ { + } ( x ) = 1 \ \Longleftrightarrow \ \langle \omega , \mathrm { M H A } ( x ) \rangle - \theta > 0 \ \Longleftrightarrow \ \mathrm { n e t } ( x ) = 1$ . The simulation used exactly one extra value coordinate (and one extra embedding coordinate feeding it), completing the proof. □

Value width. In the arg-max form, the Boolean decision depends on $V _ { j }$ and $W _ { O }$ only through $V _ { j } ( w _ { 0 } -$ $w _ { 1 } ) ^ { \top } \in \mathbb { R } ^ { d }$ . For any fixed $w _ { 0 } \neq w _ { 1 }$ , the map $V _ { j } \mapsto V _ { j } ( w _ { 0 } - w _ { 1 } ) ^ { \top }$ is onto $\mathbb { R } ^ { d }$ for every $d _ { \mathrm { o u t } } \geq 1$ . Thus value width does not change the class of Boolean functions computed at a fixed head count.

Remark 7 (Where the threshold comes from). Part (1) always yields threshold $\theta = 0 { : }$ each logit $\langle w _ { s } , \mathrm { M H A } ( x ) \rangle$ is homogeneous in $\mathrm { M H A } ( x )$ , so the two logits are equal at ${ \mathrm { M H A } } ( x ) = 0$ and the arg-max boundary passes through the origin. A nonzero threshold is an afine, not a homogeneous, condition, and the constant coordinate installed in Part (2) is exactly what lets a homogeneous read-out express it. When the core already carries such a coordinate (for instance an anchor token whose fixed value every head averages in), no augmentation is needed, and the two read-outs coincide with no overhead.

## A.2 The parity upper bound

Lemma 4 (Alternating partial fractions). Let

$$
D ( t ) = \prod _ { j = 1 } ^ { k } ( t + j ) , \qquad P ( t ) = ( - 1 ) ^ { k } \prod _ { m = 0 } ^ { k - 1 } \left( t - m - \frac { 1 } { 2 } \right) .
$$

There exist real numbers $\alpha _ { j } , \beta _ { j }$ such that

$$
{ \frac { P ( t ) } { D ( t ) } } = \sum _ { j = 1 } ^ { k } { \frac { \alpha _ { j } + \beta _ { j } t } { t + j } } .
$$

Moreover, $D ( c ) > 0$ and sign $P ( c ) = ( - 1 ) ^ { c }$ for every $c \in \{ 0 , \ldots , k \}$

Proof. Since deg $P = \deg D$ and the roots $- 1 , \ldots , - k$ of � are simple,

$$
{ \frac { P ( t ) } { D ( t ) } } = C + \sum _ { j = 1 } ^ { k } { \frac { \lambda _ { j } } { t + j } }
$$

for some $\lambda _ { 1 } , \dots , \lambda _ { k } \in \mathbb { R }$ , where $C = ( - 1 ) ^ { k }$ is the ratio of the leading coeficients. Absorb the constant term as

$$
C = \sum _ { j = 1 } ^ { k } \frac { C } { k } \frac { t + j } { t + j } ,
$$

and set $\alpha _ { j } = \lambda _ { j } + C j / k$ and $\beta _ { j } = C / k$ . This gives the claimed form.

For the sign claim, $D ( c ) > 0 \mathrm { o n } \left\{ 0 , \ldots , k \right\}$ . The roots of � are half-integers, so no factor of $P ( c )$ vanishes at an integer. Exactly $k - c$ factors are negative, so

$$
\operatorname { s i g n } P ( c ) = ( - 1 ) ^ { k } ( - 1 ) ^ { k - c } = ( - 1 ) ^ { c } .
$$

Proof of Theorem 2. Write $c \ = \ \textstyle \sum _ { i = 1 } ^ { k } x _ { i }$ for the Hamming weight, whose parity sign is $( - 1 ) ^ { c } .$ , with $c \in$ $\left\{ 0 , \ldots , k \right\}$ . We construct the output gap as a rational function $\Delta ( c )$ with sign $( - 1 ) ^ { c }$ on these $k + 1$ values.

Embedding and scores. Set $d = k + 3$ and use the standard basis $e _ { 1 } , \ldots , e _ { k + 3 } { \mathrm { : } }$ the first two coordinates encode the token value and the remaining $k + 1$ coordinates the positions:

$$
\Phi ( 0 , i ) = e _ { 1 } + e _ { i + 2 } , \Phi ( 1 , i ) = e _ { 2 } + e _ { i + 2 } \quad ( 1 \leq i \leq k ) , \qquad v _ { 7 } : = \Phi ( ? ) = e _ { k + 3 } .
$$

Specify each $A _ { j }$ through its efective query $\xi _ { j } : = v _ { ? } A _ { j }$ . For $j = 1 , \ldots , k ,$ , set

$$
\xi _ { j } = \log ( j / k ) e _ { 1 } + \log ( 1 + j / k ) e _ { 2 } - M e _ { k + 3 } ,\tag{3}
$$

with $M \to \infty$ so that the query slot is masked. Since $\xi _ { j }$ has no positional coordinates, every bit position scores $\log ( j / k )$ on token 0 and $\log ( 1 + j / k )$ on token 1, and the query position scores $- M .$ Hence,

ignoring the vanishing query self-weight for the moment, the scalar denominator is

$$
d _ { x } ^ { j } = \left( k - c \right) \frac { j } { k } + c \left( 1 + \frac { j } { k } \right) = c + j .
$$

Values.

By Lemma 4, fix $\alpha _ { j } , \beta _ { j }$ with $\begin{array} { r } { P ( t ) / D ( t ) = \sum _ { j = 1 } ^ { k } ( \alpha _ { j } + \beta _ { j } t ) / ( t + j ) } \end{array}$ , and define

$$
\eta _ { 0 , j } : = \frac { \alpha _ { j } } { j } , \qquad \eta _ { 1 , j } : = \frac { \beta _ { j } + \alpha _ { j } / k } { 1 + j / k } , \qquad \delta _ { j } : = \eta _ { 0 , j } e _ { 1 } + \eta _ { 1 , j } e _ { 2 } .
$$

To realize this scalar readout in the model, take $d _ { \mathrm { o u t } } = 2$ and $W _ { O } = I _ { 2 } ,$ , and choose $V _ { j }$ so that $V _ { j } e _ { 1 } = \delta _ { j } / 2$ and $V _ { j } e _ { 2 } = - \delta _ { j } / 2$ . Then $V _ { j } \big ( w _ { 0 } - w _ { 1 } \big ) ^ { \top } = \delta _ { j }$ , exactly as in Section 2. Since $\delta _ { j }$ has no positional coordinates,

$$
\begin{array} { r } { \boldsymbol { u } _ { 0 , i } ^ { j } = \Phi ( 0 , i ) \cdot \delta _ { j } = \eta _ { 0 , j } , \qquad \boldsymbol { u } _ { 1 , i } ^ { j } = \Phi ( 1 , i ) \cdot \delta _ { j } = \eta _ { 1 , j } , \qquad \boldsymbol { u } _ { 2 } ^ { j } = \boldsymbol { v } _ { ? } \cdot \delta _ { j } = 0 . } \end{array}
$$

Hence the scalar numerator is

$$
s _ { x } ^ { j } = ( k - c ) \frac { j } { k } \eta _ { 0 , j } + c \left( 1 + \frac { j } { k } \right) \eta _ { 1 , j } = \alpha _ { j } + \beta _ { j } c .
$$

Total gap. Summing over the heads,

$$
\Delta ( x ) = \Delta ( c ) = \sum _ { j = 1 } ^ { k } { \frac { s _ { x } ^ { j } } { d _ { x } ^ { j } } } = \sum _ { j = 1 } ^ { k } { \frac { \alpha _ { j } + \beta _ { j } c } { c + j } } = { \frac { P ( c ) } { D ( c ) } } ,
$$

and by Lemma 4 its sign is $( - 1 ) ^ { c } = ( - 1 ) ^ { \sum _ { i = 1 } ^ { k } x _ { i } }$ for every $c \in \{ 0 , \ldots , k \}$ , as required.

Finally. The computation above masked the query slot exactly $( M \to \infty ) ;$ ; a suficiently large finite � sufices. Indeed, the query contributes $\varepsilon = e ^ { - M }$ to each denominator and, since $u _ { ? } ^ { j } = v _ { ? } \cdot \delta _ { j } = 0$ , nothing to any numerator gap, so the finite-� gap is $\begin{array} { r } { \sum _ { j = 1 } ^ { k } ( \alpha _ { j } + \beta _ { j } c ) / ( c + j + \varepsilon ) \to P ( c ) / \dot { D ( c ) } } \end{array}$ as $\varepsilon \to 0$ . Since the limit is nonzero for each of the finitely many $c \in \{ 0 , \ldots , k \}$ , all $2 ^ { k }$ strict inequalities hold once � is large enough. Therefore the transformer outputs the parity bit. □

## A.3 Compactness

Proof of the upper bound in Theorem 3. Factor $M = P \Sigma Q ^ { \intercal }$ (thin SVD, $\Sigma \succ 0$ of size $r \times r , r = \mathrm { r a n k } ( M ) )$ , and set

$$
a ^ { \prime } : = \Sigma ^ { 1 / 2 } P _ { a , \cdot } \in \mathbb { R } ^ { r } ( a \in A ) , \qquad v _ { t } ^ { \prime } : = \Sigma ^ { 1 / 2 } Q _ { t , \cdot } \in \mathbb { R } ^ { r } ( t \in B ) ,
$$

so that $a ^ { \prime } \cdot v _ { t } ^ { \prime } = M _ { a , t }$ for all $a , t ;$ in particular $\boldsymbol { v } _ { \ u { \gamma } } ^ { \prime } \in \mathbb { R } ^ { r }$ is the reconstructed query vector, possibly 0.

Work now in dimension $r + 1$ . Append one new coordinate:

$$
\begin{array} { r } { v _ { t } : = ( v _ { t } ^ { \prime } , 0 ) \in \mathbb { R } ^ { r + 1 } \quad ( t \in B , \ t \neq ? ) , \qquad \Phi ( ? ) : = ( v _ { ? } ^ { \prime } , 1 ) \in \mathbb { R } ^ { r + 1 } . } \end{array}
$$

For each $j ,$ define $A _ { j } \in \mathbb { R } ^ { ( r + 1 ) \times ( r + 1 ) }$ by: its top � rows are 0, and its last row is $( \xi _ { j } ^ { \prime } , 0 )$ , where $\xi _ { j } ^ { \prime } : = a ^ { \prime }$ is the reconstructed vector for $a = \xi _ { j } \in A$ . Then, writing $\Phi ( ? ) = ( v _ { ? } ^ { \prime } , 1 )$ ,

$$
\Phi ( ? ) A _ { j } \ = \ { \boldsymbol v } _ { ? } ^ { \prime } \cdot ( \mathrm { t o p } r \ \mathrm { r o w s \ o f } \ A _ { j } ) \ + \ 1 \cdot ( \xi _ { j } ^ { \prime } , 0 ) \ = \ ( \xi _ { j } ^ { \prime } , 0 ) ,
$$

regardless of $v _ { \gamma } ^ { \prime } -$ the top rows being zero exactly cancels any dependence on the old query coordinates. So set $\xi _ { j } : = ( \xi _ { j } ^ { \prime } , 0 ) \in \mathbb { R } ^ { r + 1 } ;$ ; we have engineered $\Phi ( ? ) A _ { j } = \xi _ { j }$ exactly, as the model requires.

It remains to check $\xi _ { j }$ still reproduces every entry of � it is responsible for. For an input token $t \neq ? { \mathrm { : } }$ $\boldsymbol { \xi } _ { j } \cdot \boldsymbol { v } _ { t } = ( \xi _ { j } ^ { \prime } , 0 ) \cdot ( v _ { t } ^ { \prime } , 0 ) = \boldsymbol { \xi } _ { j } ^ { \prime } \cdot v _ { t } ^ { \prime } = M _ { \xi _ { j } , t }$ , unchanged. For the self-score, $t = ? : \xi _ { j } \cdot \Phi ( ? ) = ( \xi _ { i } ^ { \prime } , 0 ) \cdot ( v _ { ? } ^ { \prime } , 1 ) =$ $\boldsymbol { \xi } _ { j } ^ { \prime } \cdot \boldsymbol { v } _ { ? } ^ { \prime } = \boldsymbol { M } _ { \xi _ { j } , ? }$

The remaining objects $- \Phi ( ? ) { \mathrm { : } }$ ’s use as a reference point for value-projections $w _ { t } ^ { j , l } = v _ { t } \cdot y _ { j } ^ { ( l ) }$ , and the read-out vectors $y _ { j } ^ { ( l ) }$ themselves — are reconstructed exactly as before (padded with a trailing $0 )$ , so all other entries of � are reproduced unchanged. The reassembled transformer therefore has dimension $r + 1$ and computes the same function. □

ProofofTheorem 4. Under the tie-breaking rule a realizer may satisfy some pairwise comparisons with equality, so we first make every comparison strict. Fix a realizer and, for each label �, shift head ${ 1 \mathord { \ : } } s$ value-projections to label � by a constant, $w _ { t } ^ { 1 , l }  w _ { t } ^ { 1 , l } + \delta _ { l }$ , for every occurring token type �, the query token included. Because the shift is uniform over tokens, it adds $\delta _ { l } d _ { x } ^ { 1 } / d _ { x } ^ { 1 } = \delta _ { l }$ to $\log \mathrm { i t } _ { l } ( x )$ on every input �. Ordering the $\delta _ { l }$ by the tie-breaking priority (the preferred label receiving the larger shift) and taking them small enough — there are finitely many inputs and labels — turns every tie into a strict inequality in favor of its original winner while preserving every previously strict comparison. Hence the strict system consisting of the sign conditions $\{ \sigma _ { x , l ^ { \prime } } P _ { x , l ^ { \prime } } > 0 \}$ over the $\leq q ^ { N } ( L - 1 )$ pairs $( x , l ^ { \prime } )$ , together with the positivity constraints $\{ r _ { t } ^ { j } > 0 \}$ (degree-1 inequalities, required so that any solution represents exponentiated attention weights), is nonempty — the shifted realizer satisfies all of them — and its solution set is open, hence full-dimensional and basic semialgebraic; any of its points computes $f$ under the tiebreaking semantics. By the standard point-bit-length bounds for such sets [Ren92, BPR06], a nonempty full-dimensional set cut out by degree-≤ $\delta$ polynomials in $n _ { \mathrm { v a r } }$ variables with ≤ �-bit integer coeficients contains a rational point of bit-length $\leq \tau \delta ^ { O ( n _ { \mathrm { v a r } } ) }$ . Here $\delta = k + 1$ and $\tau = O ( k \log ( N + 1 ) )$ , giving $\beta \le { \cal O } \bigl ( k \log ( N + 1 ) \bigr ) ( k + 1 ) ^ { { \cal O } ( n _ { \mathrm { v a r } } ) } = 2 ^ { { \cal O } ( n _ { \mathrm { v a r } } \log ( k + 1 ) ) }$

Let $z ^ { \star } = ( r ^ { \star } , w ^ { \star } ) \in \mathbb { Q } ^ { n _ { \mathrm { v a r } } }$ be this $\beta \mathrm { \cdot }$ -bit rational point. We first quantify how accurately its log-scores must be rounded. For any strict signed decision polynomial $Q = \sigma _ { x , l ^ { \prime } } P _ { x , l ^ { \prime } }$ , clearing the coordinate denominators in $Q ( z ^ { \star } ) > 0$ shows that

$$
Q ( z ^ { \star } ) ~ \ge ~ 2 ^ { - \beta \delta n _ { \mathrm { v a r } } } .
$$

Indeed, a common denominator divides the product of the $n _ { \mathrm { v a r } }$ coordinate denominators raised to the degree $\delta ,$ and the resulting numerator is a positive integer. On the unit $\ell _ { \infty }$ -box around $z ^ { \star }$ , every coordinate has magnitude at most $R : = 2 ^ { \beta + 1 }$ . Since $Q$ has degree at most $\delta ,$ coeficient magnitude at most $2 ^ { \tau }$ , and at most $\binom { n _ { \mathrm { v a r } } + \delta } { \delta }$ monomials, its $\ell _ { 1 }$ -gradient is bounded there by

$$
\Lambda : = n _ { \mathrm { v a r } } \delta \binom { n _ { \mathrm { v a r } } + \delta } { \delta } 2 ^ { \tau } R ^ { \delta - 1 } .
$$

The mean-value theorem therefore shows that every signed decision polynomial remains positive throughout the $\ell _ { \infty }$ -ball of radius

$$
\varepsilon : = \operatorname* { m i n } \biggl \{ 1 , \frac { 2 ^ { - \beta \delta n _ { \mathrm { v a r } } - 1 } } { \Lambda } \biggr \}
$$

around $z ^ { \star }$ . In particular,

$$
\log _ { 2 } ( \varepsilon ^ { - 1 } ) = O ( \beta \delta n _ { \mathrm { v a r } } + \tau + \delta \log ( n _ { \mathrm { v a r } } + \delta ) ) .
$$

For each $r _ { t } ^ { j , \star }$ , put $p _ { t } ^ { j , \star } = \log r _ { t } ^ { j , \star }$ and choose a dyadic rational $\widehat { p } _ { t } ^ { j }$ satisfying

$$
\left| \widehat { p } _ { t } ^ { j } - p _ { t } ^ { j , \star } \right| \leq 2 ^ { - \beta - 2 } \varepsilon .
$$

Because a positive �-bit rational lies in $[ 2 ^ { - \beta } , 2 ^ { \beta } ]$ , we have $| p _ { t } ^ { j , \star } | \leq \beta \log 2$ . The mean-value theorem for the exponential gives

$$
\left| e ^ { \widehat { p } _ { t } ^ { j } } - r _ { t } ^ { j , \star } \right| \ \leq \ \varepsilon .
$$

Moreover, the dyadic approximants can be chosen with bit-length

$$
\widehat { \beta } \ = \ O ( \beta \delta n _ { \mathrm { v a r } } + \tau + \delta \log ( n _ { \mathrm { v a r } } + \delta ) ) \ = \ O \big ( k \log ( N + 1 ) \big ) ( k + 1 ) ^ { O ( n _ { \mathrm { v a r } } ) } \ = \ 2 ^ { O ( n _ { \mathrm { v a r } } \log ( k + 1 ) ) } .
$$

Thus $( e ^ { \widehat { p } } , w ^ { \star } )$ lies in the same realizing cell as $( r ^ { \star } , w ^ { \star } )$ and computes � with rational score and valueprojection tables of the claimed bit-length.

Finally, realize these tables by rational model parameters. Enumerate the $T : = | B | \leq q N$ occurring token types, including the query, and work in dimension $T + 1 ,$ . Assign each token type � its standard basis vector $\boldsymbol { v } _ { t } = \boldsymbol { e } _ { t }$ . For each head �, put the entries $\widehat { p } _ { t } ^ { j }$ in the query row of $A _ { j }$ , so that $\Phi ( ? ) A _ { j } { v } _ { t } ^ { \top } = \widehat { p } _ { t } ^ { j }$ . For each label �, take $y _ { j } ^ { ( l ) }$ to have coordinates $( w _ { t } ^ { j , l } ) ^ { \star }$ on these basis vectors and zero in the extra coordinate. Taking $d _ { \mathrm { o u t } } = L$ $W _ { O } = I _ { L }$ , and $V _ { j } = Y _ { j }$ realizes these read-out vectors exactly. All entries of $\Phi , \{ A _ { j } , V _ { j } \} , W _ { O }$ are rational of bit-length at most $\beta ,$ and the resulting transformer has dimension at most $q N + 1$ and computes $f .$

## A.4 The additive monomial vote

ProofofLemma 2. Use the query token itself as the head’s reference: by Definition 1 the query attends to itself, so set its self-score to give $r _ { ? } = e ^ { - R / 2 }$ , read the query’s value as 1, and read every bit token’s value as 0 (additively realizable: the value-projection’s position part is free, so put weight 1 on the query position and 0 on the bit positions, with bit-slope 0). Choose the efective query so that the exponentiated bit scores are

$$
r _ { x _ { i } , i } = e ^ { - R x _ { i } } ~ ( i \in S ) , \qquad r _ { x _ { i } , i } = e ^ { - 2 R - R x _ { i } } \leq e ^ { - 2 R } ~ ( i \not \in S ) .
$$

This is additive: the bit-slope $\Delta = \xi \cdot ( \tau _ { 1 } - \tau _ { 0 } ) = - R$ is a single head constant (position-independent), while the position biases $h ( i ) = \xi \cdot \pi _ { i }$ , free in � through the $\pi _ { i }$ , are 0 on � and −2� of � (so of-� tokens score at most $e ^ { - 2 R }$ regardless of their bit). Because only the query carries value 1, the read-out is the query’s self-attention weight,

$$
A _ { S } ( x ) = { \frac { e ^ { - R / 2 } } { e ^ { - R / 2 } + \sum _ { i \in S } e ^ { - R x _ { i } } + \sum _ { i \notin S } e ^ { - 2 R - R x _ { i } } } } .
$$

If $x _ { i } = 1$ for all $i \in S$ (the term holds), the middle sum is $| S | e ^ { - R }$ and $A _ { S } ( x ) = 1 - O ( n e ^ { - R / 2 } )$ . If some $i \in S$ has $x _ { i } = 0$ (the term fails), that token contributes $e ^ { 0 } = 1$ to the denominator, so $A _ { S } ( x ) \leq e ^ { - R / 2 }$ . Either way $A _ { S } ( x )$ is within $n e ^ { - R / 2 }$ of $\textstyle \prod _ { i \in S } x _ { i }$ . The construction adds no token to the input; it uses $d = O ( n )$ , and all scores are $O ( R )$ □

Proof of Theorem 5. Take one monotone-term head (Lemma 2) per nonempty � with $\alpha _ { S } \neq 0$ , all sharing a common steepness �, and read them out in the afine scalar form of Definition 2 with head weights $\alpha _ { S }$ and threshold $\theta = { \textstyle \frac { 1 } { 2 } } - \alpha _ { \emptyset }$ (the empty term $\textstyle \prod _ { i \in { \mathcal { D } } } x _ { i } \equiv 1$ is the constant $\alpha _ { \emptyset }$ , folded into �). If no such head is needed, use one zero-valued head; since $n \geq 1$ , the head count remains at most $2 ^ { n }$ . The decision statistic is then

$$
F ( x ) \ = \ \Bigl ( \alpha _ { \mathcal { O } } + \sum _ { S \neq \emptyset } \alpha _ { S } A _ { S } ( x ) \Bigr ) - \frac { 1 } { 2 } \ = \ \Bigl ( \underbrace { \sum _ { S } \alpha _ { S } \prod _ { i \in S } x _ { i } } _ { = f ( x ) } - \frac { 1 } { 2 } \Bigr ) \ + \ \sum _ { S \neq \emptyset } \alpha _ { S } \bigl ( A _ { S } ( x ) - \prod _ { i \in S } x _ { i } \bigr ) .
$$

The error term is at most $\begin{array} { r } { n e ^ { - R / 2 } \sum _ { S } | \alpha _ { S } | \le n 3 ^ { n } e ^ { - R / 2 } } \end{array}$ (Lemma 2), so choosing $R = O ( n )$ (concretely any $R > 2 { \bigl ( } n \ln 3 + \ln ( 2 n ) { \bigr ) } )$ makes it strictly less than ${ \frac { 1 } { 2 } } .$ . Then sign $F ( x ) = \mathrm { s i g n } { \bigl ( } f ( x ) - { \textstyle { \frac { 1 } { 2 } } } { \bigr ) }$ , so net $( x ) = 1 \{ F ( x ) >$ $0 \} = f ( x )$ for all �. The read-out weights $\alpha _ { S }$ are �-bit integers and the scores are $O ( R ) = O ( n )$ , so every parameter is $O ( n ) { \cdot } \mathrm { l o i t } $ □

## B The �-ESP Problem

## B.1 Problem Definition

This task is the natural �-bit extension of the Endpoint Selection Problem (ESP) of [TKRS26]: ESP uses one selector bit, whereas �-ESP selects between the two endpoints using the parity of � selector bits.

Fix a finite endpoint vocabulary U with $| \mathcal { U } | \ge 2 ,$ disjoint from the selector symbols $\{ 0 , 1 \}$ and the query symbol ?. The �-ESP problem takes the (�+3)-token input

$$
x _ { 1 } x _ { 2 } \cdots x _ { k } \ p \ q \ ? , \qquad x _ { 1 } , \ldots , x _ { k } \in \{ 0 , 1 \} , \quad p , q \in { \mathcal U } , \quad p \neq q ,
$$

and outputs � if $\textstyle \bigoplus _ { i = 1 } ^ { k } x _ { i } = 0$ and � otherwise.

Remark 8 (Relation to parity). Fix distinct $u _ { 0 } , u _ { 1 } \in \mathcal { U }$ and restrict to

$$
( p , q ) = \big ( u _ { x _ { k + 1 } } , u _ { 1 - x _ { k + 1 } } \big ) .
$$

The required output is $u _ { \oplus _ { i = 1 } ^ { k + 1 } x _ { i } } ;$ hence, under the relabeling $u _ { b }  b ,$ , this restriction is exactly (� + 1)-bit parity.

## B.2 Lower Bound for �-ESP

For the lower bound, fix distinct $u _ { 0 } , u _ { 1 } \in \mathcal { U }$ , naming them so that �<sub>0</sub> has priority over $u _ { 1 }$ under the model’s fixed tie-break, and restrict to $( p , q ) \in \{ ( u _ { 0 } , u _ { 1 } ) , ( u _ { 1 } , u _ { 0 } ) \}$ . Introduce a bit $x _ { k + 1 }$ by $p = u _ { x _ { k + 1 } } , q = u _ { 1 - x _ { k + 1 } }$ , and write $x _ { k + 2 } = 1 - x _ { k + 1 }$ . On this restriction the input is

$$
x _ { 1 } x _ { 2 } \cdot \cdot \cdot x _ { k } u _ { x _ { k + 1 } } u _ { x _ { k + 2 } } ? , \qquad x _ { k + 2 } = 1 - x _ { k + 1 } ,
$$

and the target is $u _ { x }$ with $x = \bigoplus _ { 1 \leq i \leq k + 1 } x _ { i }$ . Thus any transformer solving arbitrary-vocabulary �-ESP must solve this restricted family.

For $b \in \{ 0 , 1 \}$ , use the binary index � to parametrize the two-symbol restriction by writing

$$
\begin{array} { r } { v _ { b , i } : = \left\{ \Phi ( b , i ) , \quad 1 \leq i \leq k , \quad \quad \quad r _ { b , i } ^ { j } : = \exp ( v _ { \top } A _ { j } v _ { b , i } ^ { \top } ) , \quad \quad v _ { ? } : = \Phi ( ? ) . \right. } \end{array}
$$

Specializing the criterion (1) of Section 2, now with the per-head sums running over all $k { + 2 }$ token positions and the query’s self-term, head � has numerator $\begin{array} { r } { n _ { x } ^ { j } = r _ { ? } ^ { j } v _ { ? } + \sum _ { 1 \leq i \leq k + 2 } r _ { x _ { i } , i } ^ { j } v _ { x _ { i } , i } } \end{array}$ and denominator $\begin{array} { r } { d _ { x } ^ { j } = r _ { ? } ^ { j } + \sum _ { 1 \leq i \leq k + 2 } r _ { x _ { i } , i } ^ { j } > 0 _ { ` } } \end{array}$ , where $r _ { \gamma } ^ { j } = e ^ { v _ { ? } A _ { j } v _ { ? } ^ { \top } } > 0$ is the query’s (input-independent) self-weight and

$a _ { j } , b _ { j }$ are the columns of $Y _ { j } = V _ { j } W _ { O } ^ { \top }$ reading of $u _ { 0 } , u _ { 1 }$ . Here $Y _ { j }$ may contain columns for every output symbol in $\mathcal { U } ;$ only the two columns for $u _ { 0 } , u _ { 1 }$ enter the necessary pairwise comparison, through $a _ { j } - b _ { j }$

Any transformer solving arbitrary-vocabulary �-ESP must, on this restriction, satisfy for every $x =$ $( x _ { 1 } , \ldots , x _ { k + 1 } , x _ { k + 2 } )$ with $x _ { k + 2 } = 1 - x _ { k + 1 }$

$$
( - 1 ) ^ { \sum _ { 1 \leq i \leq k + 1 } x _ { i } } \cdot \sum _ { j = 1 } ^ { k } { \frac { n _ { x } ^ { j } } { d _ { x } ^ { j } } } \cdot ( a _ { j } - b _ { j } ) \ \left\{ \begin{array} { l l } { > 0 } & { { \mathrm { i f } } \sum _ { 1 \leq i \leq k + 1 } x _ { i } { \mathrm { ~ i s ~ o d d } } ( \mathrm { t a r g e t } u _ { 1 } ) , } \\ { \geq 0 } & { { \mathrm { i f } } \sum _ { 1 \leq i \leq k + 1 } x _ { i } { \mathrm { ~ i s ~ e v e n } } ( \mathrm { t a r g e t } u _ { 0 } ) . } \end{array} \right.\tag{4}
$$

(As with parity, a tie between the two output logits resolves toward $u _ { 0 }$ per Definition 1’s tie-break, so a 0-target input only needs the displayed gap to be nonnegative, not positive.) The one structural feature the proof uses is that positions $k { + 1 }$ and $k { + 2 }$ both hold functions of the single bit $x _ { k + 1 }$

Theorem 9. For everyfinite U with $| \mathcal { U } | \geq 2 .$ , no one-layer �-head attention-only transformer solves �-ESP over U. Indeed, after fixing distinct $u _ { 0 } , u _ { 1 } \in \mathcal { U }$ and restricting to the two ordered pairs above, the following stronger infeasibility statement holds.

There is no choice ofvectors $v _ { ? } , v _ { b , i } , a _ { j } , b _ { j } ,$ , and positive reals $r _ { b , i } ^ { j } , r _ { ? } ^ { j } , f o r b \in \{ 0 , 1 \} , 1 \leq i \leq k + 2 , 1 \leq j \leq k ,$ such that for every $x = \left( x _ { 1 } , \ldots , x _ { k + 1 } , x _ { k + 2 } \right)$ with $x _ { k + 2 } = 1 - x _ { k + 1 }$ , the following holds:

$$
( - 1 ) ^ { \sum _ { 1 \leq i \leq k + 1 } x _ { i } } \cdot \sum _ { j = 1 } ^ { k } { \frac { n _ { x } ^ { j } } { d _ { x } ^ { j } } } \cdot ( a _ { j } - b _ { j } ) \ \left\{ { \begin{array} { l l } { > 0 } & { i f \sum _ { 1 \leq i \leq k + 1 } x _ { i } \ i s \ o d d , } \\ { \geq 0 } & { i f \sum _ { 1 \leq i \leq k + 1 } x _ { i } \ i s \ e \nu e n . } \end{array} } \right.
$$

Proof. Write $\sigma _ { x } = ( - 1 ) ^ { \sum _ { 1 \leq i \leq k + 1 } x _ { i } } \in \{ \pm 1 \}$ for the required sign at �, and let · denote the Euclidean inner product on $\mathbb { R } ^ { d }$

Step 1: Clear the denominators. Set

$$
s _ { x } ^ { j } : = n _ { x } ^ { j } \cdot ( a _ { j } - b _ { j } ) = r _ { ? } ^ { j } \big ( v _ { ? } \cdot ( a _ { j } - b _ { j } ) \big ) + \sum _ { i = 1 } ^ { k + 2 } r _ { x _ { i } , i } ^ { j } \big ( v _ { x _ { i } , i } \cdot ( a _ { j } - b _ { j } ) \big ) .
$$

In both $s _ { x } ^ { j }$ and $d _ { x } ^ { j }$ , the query term is constant, positions $1 , \ldots , k$ depend respectively on $x _ { 1 } , \ldots , x _ { k }$ , and positions $k + 1 , k + 2$ both depend only on $x _ { k + 1 }$ . Thus every summand depends on at most one of the $k + 1$ free bits. Define $\begin{array} { r } { \Delta _ { 0 1 } ( x ) : = \sum _ { j = 1 } ^ { k } s _ { x } ^ { j } / d _ { x } ^ { j } } \end{array}$

The inequality at � reads $\sigma _ { x } \Delta _ { 0 1 } ( x ) \geq 0$ , with strict inequality whenever $\sigma _ { x } = - 1$ . Since each $d _ { x } ^ { j } > 0$ multiplying through by the strictly positive quantity $\textstyle \prod _ { j = 1 } ^ { k } { \bar { d } } _ { x } ^ { j }$ preserves sign:

$$
\begin{array} { l } { \displaystyle \mathrm { s i g n } ( \Delta _ { 0 1 } ( x ) ) = \mathrm { s i g n } ( F _ { x } ) , } \\ { F _ { x } : = \displaystyle \prod _ { j = 1 } ^ { k } d _ { x } ^ { j } \Delta _ { 0 1 } ( x ) = \sum _ { j = 1 } ^ { k } s _ { x } ^ { j } \prod _ { j ^ { \prime } \neq j } d _ { x } ^ { j ^ { \prime } } . } \end{array}
$$

The theorem is therefore equivalent to the assertion: no choice of parameters makes $\sigma _ { x } F _ { x } \geq 0$ hold for every $x \in \{ 0 , 1 \} ^ { k + 1 }$ , with strict inequality whenever $\sigma _ { x } = - 1$

Step 2: Alternating-sum identity.

Claim 2. For every choice of the parameters, $\begin{array} { r } { \sum _ { x \in \{ 0 , 1 \} ^ { k + 1 } } \sigma _ { x } F _ { x } = 0 } \end{array}$

Proofofclaim. Expand $F _ { x }$ as a sum of monomials. Each term in the defining sum has the form $\begin{array} { r } { s _ { x } ^ { j } \prod _ { j ^ { \prime } \neq j } d _ { x } ^ { j ^ { \prime } } } \end{array}$ and we further expand each factor using

$$
\begin{array} { l } { { \displaystyle s _ { x } ^ { j } = r _ { ? } ^ { j } \big ( \boldsymbol { v } _ { ? } \cdot ( \boldsymbol { a } _ { j } - \boldsymbol { b } _ { j } ) \big ) + \sum _ { i = 1 } ^ { k + 2 } r _ { x _ { i } , i } ^ { j } \big ( \boldsymbol { v } _ { x _ { i } , i } \cdot ( \boldsymbol { a } _ { j } - \boldsymbol { b } _ { j } ) \big ) } , }  \\ { { \displaystyle d _ { x } ^ { j ^ { \prime } } = r _ { ? } ^ { j ^ { \prime } } + \sum _ { i = 1 } ^ { k + 2 } r _ { x _ { i } , i } ^ { j ^ { \prime } } } . } \end{array}
$$

Letting $x _ { k + 2 } = 1 - x _ { k + 1 }$ , a typical monomial � is a product of exactly � factors—one drawn from $s _ { x } ^ { j }$ and one from each of the $k - 1$ remaining denominators. Each factor depends on at most one free bit, so � depends on at most � of the $k + 1$ coordinates. Therefore some $i ^ { * } \in \left\{ 1 , \ldots , k + 1 \right\}$ is absent. Splitting the sum over � by isolating $x _ { i ^ { * } }$

$$
\sum _ { \stackrel { x \in \{ 0 , 1 \} ^ { k + 1 } } { l \not = i ^ { * } } } \sigma _ { x } \cdot m = \left( \sum _ { \stackrel { x _ { l } \in \{ 0 , 1 \} } { l \not = i ^ { * } } } ( - 1 ) ^ { \sum _ { l \not = i ^ { * } } x _ { l } } m \right) \cdot \underbrace { \sum _ { \stackrel { x _ { i ^ { * } } \in \{ 0 , 1 \} } { = 0 } } ( - 1 ) ^ { x _ { i ^ { * } } } \ = \ 0 . } _ { = 0 }
$$

Since every monomial of $F _ { x }$ vanishes individually under the alternating sum, so does the total: $\begin{array} { r } { \sum _ { x } \sigma _ { x } F _ { x } = } \end{array}$ 0. □

Step 3: Contradiction. Suppose, toward a contradiction, that the required inequalities hold simultaneously: for every $x \in \{ 0 , 1 \} ^ { k + 1 }$ (recall $x _ { k + 2 } = 1 - x _ { k + 1 }$ is determined, so the free coordinates are $x _ { 1 } , \ldots , x _ { k + 1 } )$

$$
\sigma _ { x } F _ { x } \ \ge \ 0 ,
$$

with strict inequality whenever $\sigma _ { x } = - 1$ (odd weight). Since the free coordinates range over all of $\{ 0 , 1 \} ^ { k + 1 }$ exactly $2 ^ { k } > 0$ of them have odd weight, so at least one term is strictly positive while the rest are nonnegative by hypothesis. Hence

$$
\sum _ { x \in \{ 0 , 1 \} ^ { k + 1 } } \sigma _ { x } F _ { x } > 0 ,
$$

contradicting the identity $\begin{array} { r } { \sum _ { x } \sigma _ { x } F _ { x } = 0 } \end{array}$ of Step 2. Therefore no choice of parameters satisfies all $2 ^ { k + 1 }$ inequalities, completing the proof.

Since any solver for arbitrary-vocabulary �-ESP would also solve the fixed two-symbol restriction, the same lower bound holds for every finite endpoint vocabulary U with $| \mathcal { U } | \ge 2$

## B.3 Upper Bound for �-ESP

Fix a finite endpoint vocabulary U with $m : = | \mathcal { U } | \ge 2 .$ , treated as disjoint from the bit-role tokens $\{ 0 , 1 , ? \}$ The input for �-ESP is the (�+3)-token string

$$
x _ { 1 } x _ { 2 } \cdot \cdot \cdot x _ { k } \ p \ q \ ? , \qquad x _ { i } \in \{ 0 , 1 \} , \quad p , q \in { \mathcal U } , \quad p \neq q ,
$$

and the target is $p \operatorname { i f } \bigoplus _ { i = 1 } ^ { k } x _ { i } = 0$ and � otherwise. For a one-layer �-head transformer, fold $V _ { j } W _ { O } ^ { \top }$ into the $d { \times } m$ matrix $Y _ { j }$ with columns $y _ { j } ^ { ( u ) } , u \in \mathcal { U }$ . On an admissible input, write $\upsilon _ { ? } = \Phi ( ? , k { + } 3 )$ and $r _ { ? } ^ { j } = e ^ { v { \gamma } A _ { j } v _ { ? } ^ { \top } } > 0$ Head � has

$$
n _ { x } ^ { j } = r _ { ? } ^ { j } v _ { ? } + \sum _ { i = 1 } ^ { k } r _ { x _ { i } , i } ^ { j } v _ { x _ { i } , i } + r _ { p , k + 1 } ^ { j } v _ { p , k + 1 } + r _ { q , k + 2 } ^ { j } v _ { q , k + 2 } ,
$$

$$
d _ { x } ^ { j } = r _ { ? } ^ { j } + \sum _ { i = 1 } ^ { k } r _ { x _ { i } , i } ^ { j } + r _ { p , k + 1 } ^ { j } + r _ { q , k + 2 } ^ { j } > 0 .
$$

Thus, for the supplied pair $\ p , q .$

$$
Z _ { \phi } ( x ) - Z _ { q } ( x ) = \sum _ { j } \frac { n _ { x } ^ { j } } { d _ { x } ^ { j } } \cdot \big ( y _ { j } ^ { ( p ) } - y _ { j } ^ { ( q ) } \big ) .
$$

The construction first takes the idealized limit $r _ { ? } ^ { j } \ \to \ 0$ and then restores a suficiently small positive self-weight.

It sufices to make the �-versus-� logit gap have sign $( - 1 ) ^ { \sum _ { i = 1 } ^ { k } x _ { i } }$ while keeping every noncandidate logit below the winning candidate; the construction below makes all noncandidate logits zero and both candidate logits positive.

The lower bound shows $H = k$ heads do not sufice: restricting to two vocabulary symbols and both endpoint orders exposes $k + 1$ binary coordinates, so the alternating-sum obstruction forces $H \geq k + 1$ . We show this minimum is achieved. In particular, we present embedding $\Phi ,$ matrix $W _ { O }$ , and attention and value matrices $\boldsymbol { A } _ { j } ^ { \ \prime } \boldsymbol { s }$ and $V _ { j } ^ { , }$ s such that �-ESP is solvable using $k + 1$ heads.

Theorem 10. For everyfinite U with $| \mathcal { U } | \ge 2$ and every $k \geq 1$ , there is an additive one-layer (�+1)-head attention-only transformer that solves �-ESP over U, in embedding dimension $d = k + | \mathcal { U } | + 5$

Factorization. Write

$$
c : = \# \{ i : 1 \leq i \leq k , \ x _ { i } = 1 \} , \quad \quad \sigma _ { x } : = ( - 1 ) ^ { c } .
$$

The first � bits determine the required sign by rational interpolation in �. The two endpoint positions play a separate role: their values copy the identities of the supplied candidates $\boldsymbol { p }$ and � into their own output logits, while their unequal attention masses create the per-head �-versus-� residue. Because those masses depend only on the two fixed positions, they add input-independent mass to each denominator and do not interfere with the interpolation in �.

Embedding, attention, and value matrices. Fix a bijection $\iota : \mathcal { U } \to [ m ]$ and write $g _ { u } : = e _ { 2 + \iota ( u ) }$ for the vocabulary-identity coordinate of�. Set $d = k + m + 5$ . The first two coordinates encode the bit tokens, the next � coordinates encode endpoint identities, and the final $k + 3$ coordinates encode positions:

$$
\Phi _ { t } ( b ) = e _ { b + 1 } \quad ( b \in \{ 0 , 1 \} ) , \qquad \Phi _ { t } ( u ) = g _ { u } \quad ( u \in \mathcal { U } ) , \qquad \Phi _ { t } ( \ ? ) = 0 ,
$$

$$
\Phi _ { p } ( i ) = e _ { m + 2 + i } \quad ( 1 \leq i \leq k + 3 ) , \qquad \Phi ( t , i ) = \Phi _ { t } ( t ) + \Phi _ { p } ( i ) .
$$

With $\upsilon _ { ? } = \Phi ( ? , k + 3 )$ as above, specify each $A _ { j }$ through its efective query $\xi _ { j } : = v _ { ? } A _ { j } \in \mathbb { R } ^ { d }$

$$
\xi _ { j } = a _ { 0 } ^ { j } e _ { 1 } + a _ { 1 } ^ { j } e _ { 2 } + \rho _ { k + 1 } ^ { j } e _ { m + k + 3 } + \rho _ { k + 2 } ^ { j } e _ { m + k + 4 } - M e _ { m + k + 5 } .\tag{5}
$$

with per-head constants $a _ { 0 } ^ { j } , a _ { 1 } ^ { j } , \rho _ { k + 1 } ^ { j }$ , and $\rho _ { k + 2 } ^ { j }$ specified below and $M  \infty$ so that the query slot is masked. The scores $\rho _ { k + 1 } ^ { j } , \rho _ { k + 2 } ^ { j }$ are indexed by the two paired positions: they multiply the positional coordinates

$e _ { m + k + 3 } , e _ { m + k + 4 } ,$ so each applies to its position regardless of which token occupies it.

Note that $\xi _ { j }$ can be implemented by setting row $m + k + 5 \operatorname { o f } A _ { j }$ in columns $1 , 2 , m + k + 3 , m + k + 4 , m + k + 5$ to $a _ { 0 } ^ { j } , a _ { 1 } ^ { j } , \rho _ { k + 1 } ^ { j } , \stackrel { \textstyle , j } { \rho _ { k + 2 } ^ { j } } , - M .$ , respectively, and setting all other entries to 0.

Per-head scores. Using (5), we obtain the following scores for each input element:

$$
\begin{array} { r l }  { \mathrm { p o s ~ } i \in \left\{ 1 , . . . , k \right\} \left( { \mathrm { b i t ~ } } x _ { i } \right) : } & { \xi _ { j } \cdot \left( e _ { x _ { i } + 1 } + e _ { m + 2 + i } \right) = a _ { x _ { i } } ^ { j } , } \\ { { \mathrm { p o s ~ } } k + 1 \left( p \right) : } & { \xi _ { j } \cdot \left( g _ { \mathcal { P } } + e _ { m + k + 3 } \right) = \rho _ { k + 1 } ^ { j } , } \\ { { \mathrm { p o s ~ } } k + 2 \left( q \right) : } & { \xi _ { j } \cdot \left( g _ { q } + e _ { m + k + 4 } \right) = \rho _ { k + 2 } ^ { j } , } \\ { { \mathrm { p o s ~ } } k + 3 \left( ? \right) : } & { - M . } \end{array}
$$

Thus $r _ { x _ { i } , i } ^ { j } = e ^ { a _ { x _ { i } } ^ { j } }$ for $1 \leq i \leq k , r _ { p , k + 1 } ^ { j } = e ^ { \rho _ { k + 1 } ^ { j } }$ , and $r _ { q , k + 2 } ^ { j } = e ^ { \rho _ { k + 2 } ^ { j } }$ , independently of the identities of $\dot { \boldsymbol { p } }$ and $q .$ The value and output matrices. Set $V _ { j } = I _ { d }$ for every head and let the row of $W _ { O }$ corresponding to $u \in \mathcal { U }$ be $g _ { u } ^ { \top }$ . Equivalently, the folded output column is $y _ { j } ^ { ( u ) } = g _ { u }$ . For the supplied pair $\ P , q ,$ , the relevant gap direction is

$$
y _ { j } ^ { ( p ) } - y _ { j } ^ { ( q ) } = g _ { \dot { p } } - g _ { q } .
$$

Every bit and position coordinate is orthogonal to $g _ { P } - g _ { q }$ , while

$$
g _ { \cal P } \cdot ( g _ { \cal P } - g _ { \cal q } ) = 1 , \qquad g _ { \cal q } \cdot ( g _ { \cal P } - g _ { \cal q } ) = - 1 .
$$

Hence the first endpoint position contributes positively to the �-versus-� gap and the second contributes negatively. Moreover, every $u \not \in \{ p , q \}$ receives logit 0, while the $\mathcal { P } ^ { - }$ and �-logits are sums of positive attention weights and are therefore positive.

The gap as a rational function of �.

In the idealized limit $r _ { ? } ^ { j } = 0 _ { \vdots }$ , set

$$
\begin{array} { r } { s _ { x , p , q } ^ { j } : = n _ { x } ^ { j } \cdot ( g _ { p } - g _ { q } ) . } \end{array}
$$

Then the �-versus-� output gap is $\begin{array} { r } { Z _ { p } ( x ) - Z _ { q } ( x ) = \sum _ { j } s _ { x , p , q } ^ { j } / d _ { x } ^ { j } . } \end{array}$

Then, summing over the first � “free” bits (of which say � are 1 and $k - c$ are 0),

$$
\begin{array} { r l } & { d _ { x } ^ { j } \ = \ \underbrace { \left( k - c \right) e ^ { a _ { 0 } ^ { j } } + c e ^ { a _ { 1 } ^ { j } } } _ { \mathrm { f r e e ~ b i t s } } + \underbrace { e ^ { \rho _ { k + 1 } ^ { j } } + e ^ { \rho _ { k + 2 } ^ { j } } } _ { \mathrm { p a i r e d } } \ = \ P _ { j } + Q _ { j } c , } \\ & { P _ { j } = k e ^ { a _ { 0 } ^ { j } } + e ^ { \rho _ { k + 1 } ^ { j } } + e ^ { \rho _ { k + 2 } ^ { j } } , \qquad Q _ { j } = e ^ { a _ { 1 } ^ { j } } - e ^ { a _ { 0 } ^ { j } } , } \end{array}
$$

a linear function of �.

The first � bits contribute nothing to $s _ { x , p , q } ^ { j } .$ , and the two endpoint positions give

$$
s _ { x , p , q } ^ { j } = e ^ { \rho _ { k + 1 } ^ { j } } - e ^ { \rho _ { k + 2 } ^ { j } } = \operatorname { R } _ { j } .
$$

Therefore

$$
\frac { s _ { x , p , q } ^ { j } } { d _ { x } ^ { j } } = \frac { R _ { j } } { { P _ { j } } + { Q _ { j } } c } ,
$$

and

$$
Z _ { \underline { { { p } } } } ( x ) - Z _ { q } ( x ) = \sum _ { j = 1 } ^ { k + 1 } { \frac { R _ { j } } { P _ { j } + Q _ { j } c } } = \preceq \Delta ( c ) .
$$

Below we choose $R _ { j } , P _ { j } , Q _ { j }$ so that sign $\Delta ( c ) = ( - 1 ) ^ { c }$ for every $c \in \{ 0 , 1 , \ldots , k \}$ ; then � wins for even parity and � wins for odd parity.

Realizability and interpolation. Two lemmas reduce the design of Δ to a Cauchy interpolation.

Lemma 5 (Realizability). For any reals $R , P > | R |$ , and $Q > 0$ , there exist reals $a _ { 0 } , a _ { 1 } , \rho _ { k + 1 }$ , and $\rho _ { k + 2 }$ realizing $( R , P , Q )$ for one head.

Proof. Indeed, set $\begin{array} { r } { S = \frac { 1 } { 2 } ( P + | R | ) \in ( | R | , P ) , e ^ { \rho k + 1 } = \frac { 1 } { 2 } ( S + R ) , e ^ { \rho k + 2 } = \frac { 1 } { 2 } ( S - R ) , e ^ { a _ { 0 } } = ( P - S ) / k , e ^ { a _ { 1 } } = Q + e ^ { a _ { 0 } } \log ( 1 + R ) . } \end{array}$ ; all quantities are positive, and the defining equations hold. □

Lemma 6 (Cauchy interpolation). For distinct reals $p _ { 1 } , . . . , p _ { n }$ and distinct reals $q _ { 0 } , \ldots , q _ { n - 1 }$ with $q _ { i } \neq p _ { j }$ for any target $y \in \mathbb { R } ^ { n }$ , there exists a unique � such that for all $\textstyle 0 \leq i < n , \sum _ { j } \lambda _ { j } / ( q _ { i } - p _ { j } ) = y _ { i }$

Proof. The $n \times n$ matrix $\left[ 1 / ( q _ { i } - p _ { j } ) \right]$ is a Cauchy matrix and is hence invertible. Therefore, the system of equations $\textstyle \sum _ { j } \lambda _ { j } / ( q _ { i } - p _ { j } ) = y _ { i }$ over variables $\lambda _ { j } , 1 \le j \le n ,$ has a unique solution. □

We are now ready to complete the proof of the theorem. Set $p _ { j } = - j \left( j = 1 , \ldots , k + 1 \right.$ , strictly negative) and the $q _ { c } = c , c = 0 , 1 , \ldots , k$ (nonnegative, hence $q _ { c } \neq p _ { j } )$ . By Lemma 6 (with $n = k + 1 )$ there is a unique $\widetilde { \lambda } \neq 0$ with $\textstyle \sum _ { j } \widetilde { \lambda } _ { j } / ( c + j ) = ( - 1 ) ^ { c }$ at every node. Put

$$
\varepsilon ^ { * } = \operatorname* { m i n } _ { \stackrel { { 1 \leq j \leq k + 1 } } { \widetilde \lambda _ { j } \neq 0 } } \frac { j } { | \widetilde \lambda _ { j } | } > 0 ,
$$

well defined since $\widetilde { \lambda } \neq 0 ;$ coordinates with $\widetilde { \lambda } _ { j } = 0$ simply get $R _ { j } = 0$ , which poses no realizability problem below. Fix $\varepsilon \in ( 0 , \varepsilon ^ { * } )$ and for each head set $Q _ { j } = 1 , P _ { j } = j , R _ { j } = \varepsilon \widetilde { \lambda } _ { j }$ . Then $P _ { j } = j > \varepsilon | \widetilde { \lambda _ { j } } | = | R _ { j } |$ and $Q _ { j } > 0 _ { : }$ so Lemma 5 supplies head parameters $a _ { 0 } ^ { j } , a _ { 1 } ^ { j } , \rho _ { k + 1 } ^ { j } , \rho _ { k + 2 } ^ { j }$ realizing $( R _ { j } , P _ { j } , Q _ { j } )$ . Since $p _ { j } = - j = - P _ { j } / Q _ { j }$ , the denominator of the �th summand of $\Delta ( c )$ is $P _ { j } + Q _ { j } c = c - p _ { j }$ , hence

$$
\Delta ( c ) = \sum _ { j = 1 } ^ { k + 1 } { \frac { R _ { j } } { c - p _ { j } } } = \varepsilon \sum _ { j = 1 } ^ { k + 1 } { \frac { \widetilde { \lambda } _ { j } } { c - p _ { j } } } = \varepsilon ( - 1 ) ^ { c } .
$$

Thus sign $\Delta ( c ) = ( - 1 ) ^ { c }$ on $c \in \{ 0 , \ldots , k \}$ , so $Z _ { p } > Z _ { q }$ when the parity is even and $Z _ { q } > Z _ { p }$ when it is odd. Since all noncandidate logits are 0 and both candidate logits are positive, the argmax is the required member of $\{ p , q \}$ . This holds uniformly for every distinct $p , q \in { \mathcal { U } } ,$ proving the theorem. □

Remark 9. The head count is tight. The matching lower bound rules out � heads for $k { \mathrm { - E S P } } ,$ , and the construction attains $k + 1$

The vocabulary-identity channel lets the same fixed parameters copy arbitrary supplied endpoints $\hbar , q$ into their own logits; it changes only the embedding dimension, not the interpolation. The �-bit weight � still consumes all $k + 1$ heads through the degree count:

over the common denominator $\textstyle \prod _ { j } ( P _ { j } + Q _ { j } c )$ , the numerator $\begin{array} { r l } { \sum _ { j } R _ { j } \prod _ { j ^ { \prime } \neq j } ( P _ { j ^ { \prime } } + Q _ { j ^ { \prime } } c ) } \end{array}$ has degree $\leq k .$ and realizing � interior sign changes of Δ across $\left\{ 0 , \ldots , k \right\}$ requires � real roots, forcing $H - 1 \geq k$

Finally, the query slot need not be masked with an infinite score. For finite �, let $\eta = e ^ { - M }$

Since the query embedding has zero projection onto $g _ { P } - g _ { q }$ , it contributes nothing to the �-versus-� numerator and contributes � to each denominator. The candidate gap is therefore

$$
\sum _ { j = 1 } ^ { k + 1 } \frac { R _ { j } } { P _ { j } + Q _ { j } c + \eta } ,
$$

which converges as $\eta  0$ to

$$
\varepsilon ( - 1 ) ^ { c } .
$$

Since there are only finitely many cases, a suficiently large finite � preserves all output signs. Thus the construction is realized with finite parameters.

## C � heads cannot solve (� + 1)-hop induction heads

We now use the same alternating-sum obstruction to show that a one-layer �-head attention-only transformer cannot solve the (� + 1)-hop induction heads task. The idea is a reduction: we encode an arbitrary $( k + 1 )$ -bit parity instance as a $( k + 1 )$ -hop instance, in such a way that the query token is fixed and every input token depends on at most one parity bit. A hop solver, restricted to these inputs, would then solve (� + 1)-parity with � heads, which the obstruction forbids.

The �-hop task. Recall the �-hop induction map of Sanford, Hsu, and Telgarsky. For a string $X =$ $X _ { 1 } \cdots X _ { N }$ over a finite alphabet, let find $\mathsf { l } _ { X } ^ { 1 } ( i ) = \operatorname* { m a x } ( \{ 0 \} \cup \{ j \leq i : X _ { j - 1 } = X _ { i } \} )$ (the position immediately after the rightmost earlier occurrence of the token $X _ { i } , \mathrm { o r } \ 0$ if none), and iterate $\mathrm { f i n d } _ { X } ^ { m } = \mathrm { f i n d } _ { X } ^ { 1 } \circ \mathrm { f i n d } _ { X } ^ { m - 1 }$ The �-hop task is to output, at the final (query) position $i ,$ the token $X _ { \mathrm { f i n d } _ { X } ^ { m } ( i ) }$

The encoding. Over the five-symbol alphabet $\{ a , b , 0 , 1 , ? \}$ , map a parity input $x = ( x _ { 0 } , x _ { 1 } , \ldots , x _ { k } ) \in$ $\{ 0 , 1 \} ^ { k + 1 }$ to the length-(4�+3) string

$$
E ( x ) \ = \ y _ { 0 } y _ { 1 } \cdot \cdot \cdot y _ { k } ? ,
$$

where each block �<sub>�</sub> is defined by

$$
y _ { 0 } = a x _ { 0 } b ( 1 - x _ { 0 } ) , \qquad y _ { i } = { \left\{ \begin{array} { l l } { a x _ { i } b \bigl ( 1 - x _ { i } \bigr ) } & { 1 \leq i < k , i { \mathrm { ~ e v e n } } , } \\ { 0 a 1 b { \mathrm { ~ i f ~ } } x _ { i } = 0 , { \mathrm { ~ e l s e ~ } } 0 b 1 a } & { 1 \leq i < k , i { \mathrm { ~ o d d } } , } \end{array} \right. }
$$

and the final block is

$$
y _ { k } = { \left\{ \begin{array} { l l } { ? x _ { k } } & { k { \mathrm { ~ e v e n } } , } \\ { ? a { \mathrm { ~ i f ~ } } x _ { k } = 0 , { \mathrm { ~ e l s e ~ ? ~ } } b } & { k { \mathrm { ~ o d d } } , } \end{array} \right. }
$$

followed by a single trailing query token ?.

## The encoding computes parity.

Lemma 7. For every $x \in \{ 0 , 1 \} ^ { k + 1 }$ , the (�+1)-hop output $o f E ( x )$ at the trailing query is the token $\textstyle \bigoplus _ { i = 0 } ^ { k } x _ { i } \in$ {0, 1}.

The chase realizes an XOR accumulator. Hop 1 leaves the query ? and reads $x _ { k }$ through $y _ { k }$ , landing in a two-valued “state” carried by the pair $\{ a , b \}$ (or {0, 1}). Hops $2 , \ldots , k$ read $x _ { k - 1 } , \ldots , x _ { 1 }$ through $y _ { k - 1 } , \ldots , y _ { 1 } ;$ the alternation between the $\{ a , b \}$ blocks (even levels) and the {0, 1} blocks (odd levels) is arranged so that each hop lands on the successor token encoding the running parity $x _ { i } \oplus \cdots \oplus x _ { k }$ . After hop � the state (an � or a �) encodes $x _ { 1 } \oplus \cdots \oplus x _ { k }$ . Hop �+1 resolves this state through $y _ { 0 } = a x _ { 0 } b \left( 1 - x _ { 0 } \right)$ , which sends $a \mapsto x _ { 0 }$ and $b \mapsto 1 - x _ { 0 } .$ , so the emitted token is

$$
x _ { 0 } \oplus ( x _ { 1 } \oplus \cdot \cdot \cdot \oplus x _ { k } ) \ = \ \bigoplus _ { i = 0 } ^ { k } x _ { i } .
$$

(The block $y _ { 0 }$ is exactly an “even” block for level 0; freezing it to � 0 � 1 would hardcode $x _ { 0 } = 0$ and lose one parity bit, reducing the reduction to only � free bits.)

Structural properties of the encoding. Two facts hold by construction and are all we need:

(P1) The query embedding is fixed. Let $N = 4 k + 3 .$ , write $v _ { t , p } : = \Phi ( t , p )$ , and set $v _ { \mathrm { q r y } } : = \Phi ( ? , N )$ . For a head $j ,$ the score to position $\boldsymbol { p }$ is $v _ { \mathrm { q r y } } A _ { j } v _ { t _ { p } , p } ^ { \top } .$ so its exponential $r _ { p } ^ { j }$ depends on the input only through the token $t _ { p }$ at that fixed position.

(P2) Every position depends on at most one bit. In each block the fixed symbols are constant and every variable slot is a function of just the bit indexing its block. Hence both $r _ { p } ^ { j }$ and any fixed scalar projection of $v _ { t _ { p } , p }$ depend on at most one of $x _ { 0 } , \ldots , x _ { k }$

Theorem 11. There is no one-layer �-head attention-only transformer that solves the (�+1)-hop induction heads task.

Proof. Suppose such a transformer exists. Run it on the inputs $E ( x ) , x \in \{ 0 , 1 \} ^ { k + 1 }$ . Solving the task requires emitting the correct output token at the query; by Lemma 7 that token is $\textstyle \bigoplus _ { i = 0 } ^ { k } x _ { i } \in \{ 0 , 1 \}$ . Write $y _ { j } ^ { ( \hat { t } ) }$ for the column of $Y _ { j } = V _ { j } W _ { O } ^ { \top }$ indexed by output token �, set $a _ { j } : = y _ { j } ^ { ( 0 ) }$ and $b _ { j } : = y _ { j } ^ { ( 1 ) }$ , and, for the token $t _ { p }$ at position ${ \boldsymbol { p } } ,$ let

$$
r _ { \ L { p } } ^ { j } : = e ^ { v _ { \scriptscriptstyle { \mathrm { q r y } } } A _ { j } v _ { t _ { \ L { p } } , \ L { p } } ^ { \top } } > 0 , \qquad d _ { x } ^ { j } : = \sum _ { \ L { p } = 1 } ^ { N } r _ { \ L { p } } ^ { j } > 0 , \qquad n _ { x } ^ { j } : = \sum _ { \ L { p } = 1 } ^ { N } r _ { \ L { p } } ^ { j } v _ { t _ { \ L { p } } , \ L { p } } ,
$$

where the sums run over all positions of $E ( x )$

Correctness on the whole family requires, for every �, that the logit of the correct symbol among {0, 1} at least match the logit of the other symbol and, under Definition 1’s tie-break toward $^ { 0 , }$ strictly exceed it whenever the correct symbol is 1. Writing

$$
\Delta _ { 0 1 } ( x ) : = Z _ { 0 } ( x ) - Z _ { 1 } ( x ) = \sum _ { j = 1 } ^ { k } \frac { n _ { x } ^ { j } } { d _ { x } ^ { j } } \cdot ( a _ { j } - b _ { j } )
$$

this reads

$$
\begin{array} { r } { ( - 1 ) ^ { \sum _ { i = 0 } ^ { k } x _ { i } } \Delta _ { 0 1 } ( x ) \begin{array} { l } { \left\{ > 0 \quad \mathrm { i f } \bigoplus _ { i = 0 } ^ { k } x _ { i } = 1 , \right. } \\ { \left. \geq 0 \quad \mathrm { i f } \bigoplus _ { i = 0 } ^ { k } x _ { i } = 0 . \right.} \end{array}  } \end{array}
$$

By (P1)–(P2), each summand of $s _ { x } ^ { j } : = n _ { x } ^ { j } \cdot ( a _ { j } - b _ { j } )$ and of $d _ { x } ^ { j }$ depends on at most one of the $k + 1$ bits $x _ { 0 } , \ldots , x _ { k }$ , and $d _ { x } ^ { j } > 0$ . This is precisely the hypothesis of the parity obstruction (Remark 1) with $P = k + 1$ binary indices and $H = k$ heads.

Explicitly, multiplying the inequality at � by $\Pi _ { j } d _ { x } ^ { j } > 0$ gives $\sigma _ { x } F _ { x } \ge 0$ for every �, with strict inequality whenever $\sigma _ { x } = - 1$ , where

$$
\sigma _ { x } : = ( - 1 ) ^ { \sum _ { i = 0 } ^ { k } x _ { i } } , \qquad F _ { x } : = \prod _ { j = 1 } ^ { k } d _ { x } ^ { j } \Delta _ { 0 1 } ( x ) = \sum _ { j = 1 } ^ { k } s _ { x } ^ { j } \prod _ { j ^ { \prime } \neq j } d _ { x } ^ { j ^ { \prime } } .
$$

Each monomial of $F _ { x }$ is a product of � summands, each depending on at most one bit by $\mathrm { ( P 1 ) \mathrm { - ( P 2 ) } }$ , so it depends on at most $k < k + 1$ bits. Summing that monomial against $\sigma _ { x }$ cancels over a missing coordinate, giving $\begin{array} { r } { \sum _ { x } \sigma _ { x } F _ { x } = 0 } \end{array}$ . But the $2 ^ { k }$ odd-weight inputs each contribute a strictly positive term and every other term is nonnegative, so $\begin{array} { r } { \sum _ { x } \sigma _ { x } F _ { x } > 0 } \end{array}$ , a contradiction.

Remark 10. The bound is one hop short of tight in the following sense. The reduction shows (�+1)-hop needs at least �+1 heads, matching the pattern that �-hop induction requires at least � heads in a one-layer attention-only transformer. This complements the constructive side, where such tasks are solved with a number of heads exponential in the number of hops; closing the gap between the linear lower bound and the exponential upper bound is left open. The reduction also uses only the necessary condition that the correct {0, 1} symbol outrank the other, so it applies to any model that solves the task, regardless of how the remaining alphabet logits behave.

## D Reducing Joint Embeddings to Additive Ones

Since additive embeddings are a restriction of joint embeddings, $\mathcal { F } _ { k } ^ { \mathrm { a d d } } \subseteq \mathcal { F } _ { k } ^ { \mathrm { j o i n t } }$ . Conversely, we show that every joint �-head computation admits an additive simulation using $O ( n ^ { k } )$ heads.

## D.1 A joint �-head net is a degree-� polynomial threshold function

Use the afine scalar convention of Definition 2. For head �, let $c _ { j } : = V _ { j } \omega$ , set $u _ { b , i } ^ { j } : = v _ { b , i } \cdot c _ { j }$ and $u _ { ? } ^ { j } : = v _ { ? } \cdot c _ { j }$ and retain the attention weights � and denominators $d _ { x } ^ { j }$ of the scalar normal form. Then

$$
s _ { x } ^ { j } : = r _ { ? } ^ { j } u _ { ? } ^ { j } + \sum _ { i = 1 } ^ { n } r _ { x _ { i } , i } ^ { j } u _ { x _ { i } , i } ^ { j } , \qquad \mathrm { n e t } ( x ) = 1 \left\{ \sum _ { j = 1 } ^ { k } \frac { s _ { x } ^ { j } } { d _ { x } ^ { j } } - \theta > 0 \right\} .
$$

Lemma 8 (PTF form). For a joint �-head transformer, since every $d _ { x } ^ { j } > 0$ , multiplying the decision inequality by $\textstyle \prod _ { j } d _ { x } ^ { j }$ is sign-preserving, so net $( x ) = 1 \{ P _ { x } > 0 \}$ with

$$
P _ { x } = \sum _ { j = 1 } ^ { k } s _ { x } ^ { j } { \prod _ { j ^ { \prime } \neq j } d _ { x } ^ { j ^ { \prime } } } \ - \ \theta \prod _ { j = 1 } ^ { k } d _ { x } ^ { j } .
$$

As a function of the input �, each $s _ { x } ^ { j }$ and $d _ { x } ^ { j }$ is afine, so $P _ { x }$ has degree $\mathrel { \mathop : } \leq k$ in �: each summand contains one afine numerator and $k - 1$ afine denominators, while the threshold term contains � afine denominators. Reducing modulo $x _ { i } ^ { 2 } = x _ { i }$ gives a multilinear polynomial

$$
\tilde { P } ( x ) = \sum _ { | S | \le k } \alpha _ { S } \prod _ { i \in S } x _ { i }
$$

that agrees with $P _ { x }$ on $\{ 0 , 1 \} ^ { n }$ (so $1 \{ \tilde { P } > 0 \} = 1 \{ P _ { x } > 0 \} = f$ there), with at most $\begin{array} { r } { \big ( _ { \leq k } ^ { n } \big ) = \sum _ { r = 0 } ^ { k } \binom { n } { r } } \end{array}$ monomials.

This is the containment $\mathcal { F } _ { k } \subseteq \{ \mathrm { d e g r e e } - k$ polynomial threshold functions}. Note the two diferent degree counts in play: $P _ { x }$ has degree $\leq k$ in the input � (used here), whereas viewed as a polynomial in the head parameters it has degree $\leq k + 1$ (the form used for the counting lower bound, Lemma 3).

## D.2 The monomial vote, run on �

The additive model detects a monotone term $\textstyle \prod _ { i \in S } x _ { i }$ sharply (Lemma 2). The reduction simply re-votes the joint net over the monotone-monomial basis of its own degree-� threshold polynomial.

Theorem 12 (Joint ⇒ additive). Suppose $f : \{ 0 , 1 \} ^ { n }  \{ 0 , 1 \}$ is computed by a joint �-head transformer. Then $f$ is computed by an additive transformer with

$$
\begin{array} { r } { k ^ { \prime } \ \leq \ \operatorname* { m i n } \Bigl \{ 2 ^ { n } , \ \binom { n } { \leq k } \Bigr \} \ = O ( n ^ { k } ) \ h e a d s , \qquad d ^ { \prime } = O ( n ) . } \end{array}
$$

Proof. Because the Boolean cube is finite, we may increase � by a suficiently small amount without changing any prediction; this makes every decision strict. Let $\begin{array} { r } { \tilde { P } = \sum _ { | S | \leq k } \alpha _ { S } \prod _ { i \in S } x _ { i } } \end{array}$ be the multilinear threshold polynomial of Lemma $^ { 8 , }$ and take one monotone-term head (Lemma 2) per nonempty � with $\alpha _ { S } \neq 0 ,$ common steepness $R ,$ read-out weight $\alpha _ { S }$ , and threshold folding in $\alpha _ { \emptyset } ;$ this is exactly the monomial vote of Theorem 5, but applied to $\tilde { P }$ rather than to the Möbius expansion of ${ \bf \nabla } \cdot f .$ Consequently, $\begin{array} { r } { \gamma _ { P } : = \operatorname* { m i n } _ { x } | \tilde { P } ( x ) | > 0 } \end{array}$ The additive read-out is $\begin{array} { r } { F ( x ) = \sum _ { S } \alpha _ { S } A _ { S } ( x ) } \end{array}$ , and by Lemma 2

$$
\big | F ( x ) - \tilde { P } ( x ) \big | \ \le \ n { e ^ { - R / 2 } } \sum _ { S } | \alpha _ { S } | .
$$

Since $\tilde { P }$ has finitely many coeficients, the right-hand side above tends to 0 as $R \to \infty ;$ choose a finite � for which it is smaller than $\gamma _ { P } .$ . Then sign $F = \mathrm { s i g n } \tilde { P } ,$ so $1 \{ F > 0 \} = f$ . Each head has dimension $O ( n )$ by Lemma 2, and there are at most $\textstyle { \binom { n } { \leq k } }$ nonempty sets � with $\alpha _ { S } \neq 0$ . Therefore, for functions of arity �,

$$
\mathcal { F } _ { k } ^ { \mathrm { j o i n t } } \subseteq \mathcal { F } _ { \operatorname* { m i n } \{ 2 ^ { n } , \binom { n } { \leq k } \} } ^ { \mathrm { a d d } } \subseteq \mathcal { F } _ { O ( n ^ { k } ) } ^ { \mathrm { a d d } } .
$$

## Acknowledgements

Generative AI tools were used to improve the clarity and presentation of the writing and proofs. The authors reviewed and verified all resulting content and take full responsibility for the manuscript.

## References

[AS23] Josh Alman and Zhao Song. Fast attention requires bounded entries. In Advances in Neural Information Processing Systems (NeurIPS), 2023. arXiv:2302.13214. 13

[AY25] Josh Alman and Hantao Yu. Fundamental limitations on subquadratic alternatives to transformers. In International Conference on Learning Representations (ICLR), 2025. arXiv: 2410.04271. 13

[BAG20] Satwik Bhattamishra, Kabir Ahuja, and Navin Goyal. On the ability and limitations of transformers to recognize formal languages. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020. arXiv:2009.11264, doi:10.18653/v1/2020.emnlp-main.576. 12

[BPR06] Saugata Basu, Richard Pollack, and Marie-Françoise Roy. Algorithms in Real Algebraic Geometry, volume 10 of Algorithms and Computation in Mathematics. Springer, 2 edition, 2006. 17

[CC22] David Chiang and Peter Cholak. Overcoming a theoretical limitation of self-attention. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (ACL), 2022. arXiv:2202.12172, doi:10.18653/v1/2022.acl-long.527. 12

[DCL21] Yihe Dong, Jean-Baptiste Cordonnier, and Andreas Loukas. Attention is not all you need: Pure attention loses rank doubly exponentially with depth. In Proceedings ofthe 38th International Conference on Machine Learning (ICML), 2021. arXiv:2103.03404. 12

[ENO<sup>+</sup>21] Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam Mc-Candlish, and Chris Olah. A mathematical framework for transformer circuits. Transformer Circuits Thread, 2021. URL: https://transformer-circuits.pub/2021/ framework/index.html. 12

[Hah20] Michael Hahn. Theoretical limitations of self-attention in neural sequence models. Transactions ofthe Association for Computational Linguistics, 8:156–171, 2020. arXiv:1906.06755, doi:10.1162/tacl\_a\_00306. 12

[Hsu26] Daniel Hsu. Lower bounds for one-layer transformers that compute parity, 2026. arXiv: 2605.12171. 2, 13

[KS24] Tokio Kajitsuka and Issei Sato. Are transformers with one layer self-attention using low-rank weight matrices universal approximators? In International Conference on Learning Representations (ICLR), 2024. arXiv:2307.14023. 13

[KSW26] Alexander Kozachinskiy, Tomasz Steifer, and Przemysław Wałęga. Parity, sensitivity, and transformers, 2026. arXiv:2602.05896. 2, 13

[KUJ<sup>+</sup>25] Alexander Kozachinskiy, Felipe Urrutia, Hector Jimenez, Tomasz Steifer, Germán Pizarro, Matías Fuentes, Francisco Meza, Cristian B. Calderon, and Cristóbal Rojas. Strassen attention, split VC dimension and compositionality in transformers. In Advances in Neural Information Processing Systems (NeurIPS), 2025. arXiv:2501.19215. 13

[LAG<sup>+</sup>23] Bingbin Liu, Jordan T. Ash, Surbhi Goel, Akshay Krishnamurthy, and Cyril Zhang. Transformers learn shortcuts to automata. In International Conference on Learning Representations (ICLR), 2023. arXiv:2210.10749. 12

[Mil64] John Milnor. On the Betti numbers of real varieties. Proceedings of the American Mathematical Society, 15(2):275–280, 1964. 10

[MLT24] Sadegh Mahdavi, Renjie Liao, and Christos Thrampoulidis. Memorization capacity of multihead attention in transformers. In International Conference on Learning Representations (ICLR), 2024. arXiv:2306.02010. 12

[MP69] Marvin Minsky and Seymour Papert. Perceptrons: An Introduction to Computational Geometry. MIT Press, 1969. 12

[MS23] William Merrill and Ashish Sabharwal. The parallelism tradeof: Limitations of log-precision transformers. Transactions of the Association for Computational Linguistics, 11:531–545, 2023. arXiv:2207.00729. 13

[MS25] William Merrill and Ashish Sabharwal. A little depth goes a long way: The expressive power of log-depth transformers. In Advances in Neural Information Processing Systems (NeurIPS), 2025. arXiv:2503.03961. 13

[MSS22] William Merrill, Ashish Sabharwal, and Noah A. Smith. Saturated transformers are constantdepth threshold circuits. Transactions of the Association for Computational Linguistics, 10:843– 856, 2022. arXiv:2106.16213, doi:10.1162/tacl\_a\_00493. 13

[OEN<sup>+</sup>22] Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Scott Johnston, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris Olah. In-context learning and induction heads. Transformer Circuits Thread, 2022. arXiv:2209.11895. 12

[PBM21] Jorge Pérez, Pablo Barceló, and Javier Marinkovic. Attention is Turing-complete. Journal ofMachine Learning Research, 22(75):1–35, 2021. URL: https://jmlr.org/papers/ v22/20-302.html. 13

[PNP24] Binghui Peng, Srini Narayanan, and Christos Papadimitriou. On limitations of the transformer architecture. In First Conference on Language Modeling (COLM), 2024. arXiv:2402.08164. 13

[PS94] Ramamohan Paturi and Michael E. Saks. Approximating threshold circuits by rational functions. Information and Computation, 112(2):257–272, 1994. doi:10.1006/inco.1994.1059. 12

[Ren92] James Renegar. On the computational complexity and geometry of the first-order theory of the reals. Part I. Journal ofSymbolic Computation, 13(3):255–299, 1992. 2, 17

[SAF25] Lena Strobl, Dana Angluin, and Robert Frank. Concise one-layer transformers can do function evaluation (sometimes), 2025. arXiv:2503.22076. 13

[Sha49] Claude E. Shannon. The synthesis of two-terminal switching circuits. The Bell System Technical Journal, 28(1):59–98, 1949. 9, 12

[SHT23] Clayton Sanford, Daniel Hsu, and Matus Telgarsky. Representational strengths and limitations of transformers. In Advances in Neural Information Processing Systems (NeurIPS), 2023. arXiv: 2306.02896. 13

[SHT24a] Clayton Sanford, Daniel Hsu, and Matus Telgarsky. One-layer transformers fail to solve the induction heads task, 2024. arXiv:2408.14332. 13

[SHT24b] Clayton Sanford, Daniel Hsu, and Matus Telgarsky. Transformers, parallel computation, and logarithmic depth. In Proceedings of the 41st International Conference on Machine Learning (ICML), 2024. arXiv:2402.09268. 1, 6, 13

[SMW<sup>+</sup>24] Lena Strobl, William Merrill, Gail Weiss, David Chiang, and Dana Angluin. What formal languages can transformers express? A survey. Transactions ofthe Association for Computational Linguistics, 12:543–561, 2024. arXiv:2311.00208, doi:10.1162/tacl\_a\_00663. 12

[Tho65] René Thom. Sur l’homologie des variétés algébriques réelles. In Stewart S. Cairns, editor, Diferential and Combinatorial Topology (A Symposium in Honor of Marston Morse), pages 255–265. Princeton University Press, 1965. 10

[TKRS26] Amanuel Tesfaye, Zeno Kujawa, Rajmohan Rajaraman, and Ravi Sundaram. Two (narrow) heads are better than (an arbitrarily wide) one. In International Conference on Learning Representations (ICLR), 2026. URL: https://openreview.net/pdf?id=RRmPbbZsvl. 1, 6, 12, 19

[VSP<sup>+</sup>17] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems (NeurIPS), 2017. arXiv:1706.03762. 12

[War68] Hugh E. Warren. Lower bounds for approximation by nonlinear manifolds. Transactions of the American Mathematical Society, 133(1):167–178, 1968. 10

[YBR<sup>+</sup>20] Chulhee Yun, Srinadh Bhojanapalli, Ankit Singh Rawat, Sashank J. Reddi, and Sanjiv Kumar. Are transformers universal approximators of sequence-to-sequence functions? In International Conference on Learning Representations (ICLR), 2020. arXiv:1912.10077. 13

[YFJS26] Qilin Ye, Deqing Fu, Robin Jia, and Vatsal Sharan. Transformers provably learn algorithmic solutions for graph connectivity, but only with the right data. In International Conference on Machine Learning (ICML), 2026. arXiv:2510.19753. 13

[YJB<sup>+</sup>26] Penghao Yu, Haotian Jiang, Zeyu Bao, Ruoxi Yu, and Qianxiao Li. The efect of attention head count on transformer approximation. In International Conference on Learning Representations (ICLR), 2026. arXiv:2510.06662. 12

[YKD<sup>+</sup>24] Gilad Yehudai, Haim Kaplan, Guy Dar, Royi Rassin, Asma Ghandeharioun, Mor Geva, and Amir Globerson. When can transformers count to n?, 2024. arXiv:2407.15160. 13