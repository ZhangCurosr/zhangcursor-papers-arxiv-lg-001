# A BOREL CONCEPT CLASS OF VC DIMENSION ONE WITH ANON-PAC CONSISTENT LEARNER IN ZFC

MATEUS JESUS DE ARRUDA CAMPOS, GABRIEL FERNANDES, AND VINICIUS DE OLIVEIRA RODRIGUES

Abstract. The fundamental theorem of statistical learning states that, under suitable measurability assumptions, finite Vapnik–Chervonenkis (VC) dimension guarantees that every proper consistent learning rule is probably approximately correct (PAC). Blumer, Ehrenfeucht, Haussler, and Warmuth showed, assuming the Continuum Hypothesis, that the “well-behavedness” condition of the concept class cannot be omitted: they constructed a concept class of Borel sets of VC dimension one admitting a consistent learning rule that is not PAC. We show that the Continuum Hypothesis is unnecessary. Working in Zermelo–Fraenkel set theory with the Axiom of Choice (ZFC) alone, we construct a concept class of Borel sets on [0, 1] of VC dimension one and a proper consistent learning rule that is not PAC. More precisely, for a suitable Borel probability measure and target concept, the rule has true risk one at every sample size on a set of samples of outer probability one. Consequently, finite VC dimension and Borel measurability of the individual concepts do not sufice to guarantee that every proper consistent learning rule is PAC. The result shows, with no need of extra set-theoretical assumptions, that the additional regularity assumption in the fundamental theorem cannot in general be omitted.

## 1. Introduction

The fundamental theorem of statistical learning is a cornerstone in the theory of machine learning. Under suitable measurability assumptions, it establishes a connection between the learnability of a concept class and its Vapnik–Chervonenkis (VC) dimension. Following [2], under the “well-behavedness” condition, finite VC dimension of a Borel class of concepts implies that any consistent learning rule is PAC for the given class. Building on [3], they show that the hypothesis of well-behavedness cannot be removed if one assumes an extra set-theoretic hypothesis, namely, the Continuum Hypothesis (CH). Under CH, they construct an example of a Borel concept class of VC dimension one and a consistent learning rule that is not PAC.

Their example left open whether the extra set-theoretic hypothesis is necessary. Pestov [5] subsequently studied the same issue, attaining partial results that assumed Martin’s Axiom, an additional set-theoretic hypothesis which is weaker than CH and is also independent of ZFC.

In this paper, we settle this question by removing the need for the Continuum Hypothesis or other additional set-theoretic hypotheses: we construct, in the standard Zermelo–Fraenkel set theory with the Axiom of Choice (ZFC), a concept class of Borel sets of VC dimension one and a consistent learning rule that is not PAC.

Theorem 1.1. There exists (in ZFC alone) a concept class C of Borel sets on the interval [0, 1] with VC dimension one and a consistent learning rule for C which is not PAC.

More precisely, we construct a Borel probability measure and a Borel class of concepts such that, for a suitable target concept, the rule has true risk one at every sample size on a set of samples of outer probability 1.

This result shows, without the need of extra set-theoretic assumptions, that the fundamental theorem of statistical learning from [2] cannot be extended to all concept classes of Borel sets merely by dropping the well-behavedness hypothesis. The theorem demonstrates that the passage from finite VC dimension to consistent PAC learnability is not purely combinatorial. It delineates a mathematical boundary in the foundations of statistical learning by separating what follows from VC dimension alone from what requires measure-theoretic regularity.

## 2. Preliminaries

A Polish space is a separable completely metrizable topological space, such as $\mathbb { R } ^ { n }$ . A concept is a subset of $X$ , and a concept class C is a collection of concepts.

Definition 2.1 ([6]). Let X be a Polish space and let C be a concept class on $X$ . For a finite set $F \subseteq X$ , the trace of $\mathcal { C }$ on F is $\mathcal { C } \uparrow F = \{ H \cap F : H \in \mathcal { C } \}$

A finite set $F$ is shattered by C if ${ \mathcal { C } } \cap F = { \mathcal { P } } ( F )$ . The VC dimension of $\mathcal { C }$ is defined as

$\operatorname { V C d i m } ( { \mathcal { C } } ) = \operatorname* { s u p } \left\{ | F | : F \subseteq X \right.$ is finite and shattered by $c \}$

If the supremum is infinite, we write ${ \mathrm { V C d i m } } ( { \mathcal { C } } ) = \infty$

Blumer, Ehrenfeucht, Haussler, and Warmuth showed, under suitable measurability assumptions, that the VC dimension characterizes distribution-free binary PAC learnability [2, Theorems 2.1(i) and A2.2]. Their characterization is the basis of what is now known as the fundamental theorem of statistical learning.

If X denotes a topological space, then $B ( X )$ denotes its Borel $\sigma { \mathrm { - a l g e b r a } } .$ that is, the σ-algebra generated by the open sets of X. A Borel probability measure on X is a probability measure $\mu$ defined on $B ( X )$ . If $\mu$ is a Borel probability measure on $X$ , we continue writing $\mu$ for the unique extension to its completion. We say that a concept class $\mathcal { C }$ is Borel if every concept in $\mathcal { C }$ is a Borel set.

For $m \geq 1$ , let $\mu ^ { \otimes m }$ denote the product probability measure on $( X ^ { m } , B ( X ^ { m } ) )$ Assuming that X is Polish, $B ( X ^ { m } ) = B ( X ) ^ { \otimes m }$ . An element of $X ^ { m }$ is called a sample of size $m _ { \colon }$ or an m-sample. A labeled m-sample is an element of $( X \times \{ 0 , 1 \} ) ^ { m }$

Given $A \subseteq X$ , we denote by ${ \bf 1 } _ { A }$ its characteristic function. For a concept $C \in { \mathcal { C } }$ and $s = ( x _ { 1 } , \ldots , x _ { n } ) \in X ^ { n }$ , the C-labeled sample generated by s is

$$
S _ { C } ( s ) = \Big ( ( x _ { 1 } , \mathbf { 1 } _ { C } ( x _ { 1 } ) ) , \ . \ . . , ( x _ { n } , \mathbf { 1 } _ { C } ( x _ { n } ) ) \Big ) .
$$

We work in the realizable setting: an unknown target concept $C \in { \mathcal { C } }$ labels each $x \in X$ by $\mathbf { 1 } _ { C } ( x )$ Given a finite C-labeled sample, the learner aims to output a hypothesis $H \in { \mathcal { C } }$ with small error relative to $C$

Formally, a learning rule for ${ \mathcal { C } } .$ , or a learner for $\mathcal { C }$ , is a sequence of maps

$$
L _ { n } : ( X \times \{ 0 , 1 \} ) ^ { n } \longrightarrow { \mathcal { C } } \qquad ( n \geq 1 ) .
$$

Thus, all our learners are proper. The learner is said to be consistent if for every $C \in { \mathcal { C } } $ for every $n \geq 1$ and every sample $s \in X ^ { n }$ ,

$$
x _ { i } \in L _ { n } ( S _ { C } ( s ) ) \quad \iff \quad x _ { i } \in C \qquad ( i = 1 , \dots , n ) .
$$

Definition 2.2 (True risk). Let X be a Polish space, let C be a Borel concept class on X, and let $L = ( L _ { n } ) _ { n \geq 1 }$ be a learning rule for C. The true risk of L with respect to $C \in { \mathcal { C } }$ a Borel probability measure $\mu$ on $X$ , and $n \geq 1$ is the function $r _ { C , \mu , n } : X ^ { n } \to [ 0 , 1 ]$ defined by

$$
r _ { C , \mu , n } ( s ) = \mu { \Big ( } L _ { n } ( S _ { C } ( s ) ) \triangle C { \Big ) } ,
$$

where $s \in X ^ { n }$ and $\triangle$ denotes the symmetric diference of sets. For each $\varepsilon > 0 , n \geq 1$ $C \in { \mathcal { C } }$ , and Borel probability measure $\mu$ on X, the bad-sample event at accuracy ε is

$$
B _ { C , \mu , n } ( \varepsilon ) = \{ s \in X ^ { n } : r _ { C , \mu , n } ( s ) > \varepsilon \} .
$$

Thus, $r _ { C , \mu , n } ( s )$ is the probability that $L _ { n } ( S _ { C } ( s ) )$ disagrees with the target $C$ on an independent point $x \sim \mu$

Definition 2.3. Let X be a Polish space, let C be a Borel concept class on X, and let $L = ( L _ { n } ) _ { n \geq 1 }$ be a learning rule for C. The rule L is probably approximately correct $( P A C )$ for C if, for every $\varepsilon , \delta > 0$ , there exists $N \geq 1$ such that, for every $n \geq N$ , every $C \in { \mathcal { C } }$ , and every Borel probability measure µ on X,

$$
( \mu ^ { \otimes n } ) ^ { * } \left( B _ { C , \mu , n } ( \varepsilon ) \right) \leq \delta ,
$$

where $( \mu ^ { \otimes n } ) ^ { * }$ denotes the outer measure induced by $\mu ^ { \otimes n }$

We say that C is PAC learnable if there exists a PAC learning rule for C.

We say that C is consistently PAC learnable if every consistent learning rule for C is PAC.

We now state the parts of the fundamental theorem of statistical learning relevant to our discussion.

Theorem 2.4 ([2, Theorem 2.1]). Let X be a Polish space and let C be a Borel concept class.

(i) $I f { \mathrm { V C d i m } } ( { \mathcal { C } } ) = \infty$ and C is non-trivial, then C is not PAC learnable.

(ii) $I f \operatorname { V C d i m } ( { \mathcal { C } } ) < \infty$ and C is well-behaved, then C is consistently PAC learnable.

The well-behavedness assumption is a measurability condition on certain sets of samples; see [2, p. 953] for its definition. It holds, in particular, for universally separable Borel classes [2, Lemma A1.1, p. 954]. It is not needed to prove that if VCdim $( { \mathcal { C } } ) = \infty$ then C is not PAC learnable. However, it is needed in their proof that VCdim(C) < ∞ implies that C is consistently PAC learnable. In the next section, we show that the well-behavedness assumption cannot be removed.

## 3. Proof of Theorem 1.1

The classical CH construction of Durst and Dudley [3, Proposition 2.2], as modified by Blumer et al. [2, Appendix A1, p. 953], uses CH to well-order the whole domain so that every proper initial segment is countable and hence Borel. To obtain a ZFC example with Borel concepts, we replace those initial segments by an increasing chain of Borel null sets.

Let $I = [ 0 , 1 ]$ with its usual topology, let λ be Lebesgue measure on I, and let

$$
{ \mathcal { N } } = \{ A \subseteq I : \lambda ^ { * } ( A ) = 0 \}
$$

be its null ideal. Identifying cardinals with initial ordinals, set

$$
\mathrm { a d d } ( { \mathcal { N } } ) = \operatorname* { m i n } \left\{ | A | : A \subseteq { \mathcal { N } } , \bigcup { \mathcal { A } } \neq { \mathcal { N } } \right\} .
$$

The cardinal add(N) is known as the additivity of the null ideal. It is well known that $\aleph _ { 1 } \leq \mathrm { a d d } ( { \mathcal { N } } ) \leq { \mathfrak { c } } .$ , where c is the cardinality of the continuum. For background on add(N ), see [1].

In the construction below, we repeatedly use the fact that if $\mu$ is a Borel probability measure on I and $A \subseteq I$ satisfies $\mu ^ { * } ( A ) = c ,$ then there exists a Borel set $B \supseteq A$ such that $\mu ( B ) = c$

Lemma 3.1. There exist an increasing family $\left. C _ { \alpha } : \alpha < \mathrm { a d d } ( \mathcal { N } ) \right.$ of Borel subsets of I and a Borel probability measure $\mu$ on I such that:

(i) $\mu ( C _ { \alpha } ) = 0$ for every $\alpha < \mathrm { a d d } ( \mathcal { N } )$ , and

(ii) $\mu ^ { * } \left( \cup _ { \alpha < \mathrm { a d d } ( \mathcal { N } ) } C _ { \alpha } \right) = 1 .$

Proof. Choose $\langle A _ { \xi } : \xi < \mathrm { a d d } ( { \mathcal { N } } ) \rangle \subseteq { \mathcal { N } }$ such that $\bigcup _ { \xi < \mathrm { a d d } ( \mathcal { N } ) } A _ { \xi } \notin \mathcal { N }$ . Such a family exists

by the definition of $\operatorname { a d d } ( { \mathcal { N } } )$

Let λ be the Lebesgue measure on I. By transfinite recursion, choose a family $\left. C _ { \alpha } : \alpha < \mathrm { a d d } ( \mathcal { N } ) \right.$ of Borel λ-null sets such that, for every $\alpha < \mathrm { a d d } ( \mathcal { N } )$

$$
A _ { \alpha } \cup \bigcup _ { \beta < \alpha } C _ { \beta } \subseteq C _ { \alpha } .
$$

At stage $\alpha ,$ the set on the left is a union of fewer than add $( \mathcal { N } )$ members of ${ \mathcal { N } } .$ so it is null by the definition of add(N ) and has a Borel null superset. Thus the recursion is possible and the family $\left. C _ { \alpha } \right.$ is increasing. Put

$$
Y = \bigcup _ { \alpha < \mathrm { a d d } ( \mathcal { N } ) } C _ { \alpha } \qquad \mathrm { a n d } \qquad c = \lambda ^ { * } ( Y ) .
$$

Notice that $c > 0$ as

$$
c = \lambda ^ { * } ( Y ) \geq \lambda ^ { * } \left( \bigcup _ { \alpha < \mathrm { a d d } ( \mathcal { N } ) } A _ { \alpha } \right) > 0 .
$$

Let G be a Borel set such that $Y \subseteq G$ and $\lambda ( G ) = c .$ . Define a Borel probability measure $\mu$ on I by

$$
\mu ( B ) = { \frac { \lambda ( B \cap G ) } { c } } \qquad ( B \in B ( I ) ) .
$$

Then

$$
\mu ( C _ { \alpha } ) = \frac { \lambda ( C _ { \alpha } \cap G ) } { c } = \frac { \lambda ( C _ { \alpha } ) } { c } = 0 .
$$

Moreover, if $B \supseteq Y$ is Borel, then

$$
Y \subseteq B \cap G \subseteq G ,
$$

and hence

$$
c = \lambda ^ { * } ( Y ) \leq \lambda ( B \cap G ) \leq \lambda ( G ) = c .
$$

Thus $\mu ( B ) = 1$ for every Borel superset B of $Y$ , and therefore $\mu ^ { * } ( Y ) = 1$

We shall also use the following form of Tonelli’s theorem. For a reference, see $[ 4 ,$ 252P].

Theorem 3.2 (Tonelli’s theorem). Let $\mu$ be a Borel probability measure on $I ,$ let $m \geq 1$ , and let $f : I \times I ^ { m }  [ 0 , \infty ]$ be Borel measurable. Then the map

$$
x \longmapsto \int _ { I ^ { m } } f ( x , z ) d \mu ^ { \otimes m } ( z )
$$

is Borel measurable and

$$
\int _ { I ^ { m + 1 } } f d \mu ^ { \otimes ( m + 1 ) } = \int _ { I } \left( \int _ { I ^ { m } } f ( x , z ) d \mu ^ { \otimes m } ( z ) \right) d \mu ( x ) .
$$

The following is well-known, but we include a proof for completeness.

Lemma 3.3. Let $\mu$ be a Borel probability measure on $I ,$ and let $Y \subseteq I$ satisfy $\mu ^ { * } ( Y ) = 1$ For every $n \geq 1$ 7

$$
( \mu ^ { \otimes n } ) ^ { * } ( Y ^ { n } ) = 1 .
$$

Proof. Proceed by induction on $n$ . The assertion for $n = 1$ is trivial by hypothesis. Suppose it holds for $n ,$ and let $B \subseteq I ^ { n + 1 }$ be a Borel set containing $Y ^ { n + 1 }$ . For every $x \in Y$ , the Borel section

$$
B _ { x } = \{ z \in I ^ { n } : ( x ) \cap z \in B \}
$$

contains $Y ^ { n }$ , so $\mu ^ { \otimes n } ( B _ { x } ) = 1$ by the induction hypothesis. Set $g ( x ) = \mu ^ { \otimes n } ( B _ { x } )$ . By Theorem 3.2, g is Borel measurable, so

$$
E = \{ x \in I : g ( x ) = 1 \}
$$

is Borel, contains $Y ,$ and therefore has $\mu -$ -measure one. Applying Theorem 3.2 to $\mathbf { 1 } _ { B }$ gives

$$
1 \geq \mu ^ { \otimes ( n + 1 ) } ( B ) = \int _ { I } g ( x ) d \mu ( x ) \geq \int _ { E } g ( x ) d \mu ( x ) = \mu ( E ) = 1 .
$$

Thus $\mu ^ { \otimes ( n + 1 ) } ( B ) = 1$ . Taking the infimum over Borel supersets B proves the claim. □

Now we are ready to prove the main result of this paper.

Proof of Theorem 1.1. Fix an increasing family $\left. C _ { \alpha } : \alpha < \mathrm { a d d } ( \mathcal { N } ) \right.$ of Borel subsets of I and a Borel probability measure $\mu$ on I as in Lemma 3.1. Let

$$
Y = \bigcup _ { \alpha < \mathrm { a d d } ( \mathcal { N } ) } C _ { \alpha } \qquad \mathrm { a n d } \qquad \mathcal { C } = \{ \emptyset , I \} \cup \{ C _ { \alpha } : \alpha < \mathrm { a d d } ( \mathcal { N } ) \} .
$$

The class C is a Borel concept class that forms a chain containing both ∅ and I. Hence it shatters every singleton and no two-point set, so ${ \mathrm { V C d i m } } ( { \mathcal { C } } ) = 1$

For a labeled sample $\sigma = \left( ( x _ { 1 } , b _ { 1 } ) , \dots , ( x _ { n } , b _ { n } ) \right)$ , let

$$
P ( \sigma ) = \{ x _ { i } : b _ { i } = 1 \} .
$$

For every nonempty finite $P \subseteq Y$ , set

$$
\rho ( P ) = \operatorname* { m i n } \{ \alpha < \mathrm { a d d } ( N ) : P \subseteq C _ { \alpha } \} .
$$

The set on the right is nonempty because $P$ is finite, the family is increasing, and its union is $Y$ . Define

$$
L _ { n } ( \sigma ) = \left\{ \begin{array} { l l } { \varnothing , } & { P ( \sigma ) = \varnothing , } \\ { C _ { \rho ( P ( \sigma ) ) } , } & { \varnothing \neq P ( \sigma ) \subseteq Y , } \\ { I , } & { P ( \sigma ) \notin Y . } \end{array} \right.
$$

Claim 3.3.1. The learning rule $L = ( L _ { n } ) _ { n \geq 1 }$ is consistent for C.

Proof of the claim. Fix $n \geq 1$ , a target $C \in { \mathcal { C } } .$ , and a sample $s \in I ^ { n }$ . We verify that $L _ { n } ( S _ { C } ( s ) )$ agrees with $C$ on every point of s.

If $C = \varnothing$ , then every label is zero. Thus $P ( S _ { C } ( s ) ) = \emptyset$ , and the learner returns $\varnothing .$ which agrees with all the labels.

Suppose that $C = C _ { \beta }$ for some $\beta < \mathrm { a d d } ( \mathcal { N } )$ . Every positively labeled sample point belongs to $C _ { \beta }$ . If there are no such points, $L _ { n } ( S _ { C } ( s ) ) = \emptyset$ and hence agrees with the sample. Otherwise,

$$
\varnothing \neq P ( S _ { C } ( s ) ) \subseteq C _ { \beta } \subseteq Y ,
$$

so $\rho ( P ( S _ { C } ( s ) ) ) \le \beta$ . Consequently,

$$
P ( S _ { C } ( s ) ) \subseteq C _ { \rho ( P ( S _ { C } ( s ) ) ) } \subseteq C _ { \beta } .
$$

The first inclusion shows that the output contains every positively labeled sample point. The second shows that it contains no negatively labeled sample point, since every such point lies outside the target $C _ { \beta }$

Finally, suppose that $C = I$ . Then every sample point has label one, so $P ( S _ { C } ( s ) )$ is the set of all points occurring in s. If this set is contained in $Y$ , the learner returns $C _ { \rho ( P ( S _ { C } ( s ) ) ) }$ , which contains all the sample points. Otherwise it returns $I ,$ which also contains all of them. Thus the output agrees with every label in this case as well. □

Now we show that the rule is not PAC. Consider $C = I$ and the Borel probability measure $\mu .$ Every $C _ { \alpha }$ is $\mu { \mathrm { - } } \mathrm { n u l l }$ , and the learning rule returns I if and only if at least one sample point lies outside Y. If $s \in Y ^ { n }$ , then $L _ { n } ( S _ { I } ( s ) ) = C _ { \alpha }$ for some $\alpha < \mathrm { a d d } ( \mathcal { N } )$ and hence $r _ { I , \mu , n } ( s ) = 1$ because $\mu ( C _ { \alpha } ) = 0$ . Otherwise, $L _ { n } ( S _ { I } ( s ) ) = I$ , and hence $r _ { I , \mu , n } ( s ) = 0$ . Therefore,

$$
r _ { I , \mu , n } ( s ) = \mu { \Big ( } L _ { n } ( S _ { I } ( s ) ) \triangle I { \Big ) } = \mathbf { 1 } _ { Y ^ { n } } ( s ) .
$$

For every $0 < \varepsilon < 1$ , it follows that $B _ { I , \mu , n } ( \varepsilon ) = \{ s : r _ { I , \mu , n } ( s ) > \varepsilon \} = Y ^ { n }$ , and Lemma 3.3   
gives, for every $n \geq 1$ 7

$$
( \mu ^ { \otimes n } ) ^ { * } ( B _ { I , \mu , n } ( \varepsilon ) ) = ( \mu ^ { \otimes n } ) ^ { * } ( Y ^ { n } ) = 1 .
$$

Taking, for instance, $\varepsilon = \delta = 1 / 2$ contradicts Definition 2.3. Thus, the rule is not PAC, and the proof of Theorem 1.1 is complete. □

## 4. Conclusion

Theorem 1.1 shows that the passage from finite VC dimension to consistent PAC learnability is not purely combinatorial. Some measure-theoretic regularity is needed to connect the trace bounds supplied by finite VC dimension to probabilities of the bad-sample events. The assumption that each individual concept is Borel does not sufice. We note that in our construction, we do not guarantee that the class is not PAC learnable: the theorem exhibits a consistent learning rule that fails to be PAC, but does not rule out the existence of another PAC learning rule for the same class.

Earlier counterexamples relied on additional set-theoretic assumptions, most notably the Continuum Hypothesis, and therefore showed only that such a failure is consistent with ZFC. Our construction removes this dependence by producing a counterexample in ZFC alone. Consequently, ZFC refutes the assertion that every Borel concept class of finite VC dimension is consistently PAC learnable.

## Acknowledgments

The authors gratefully acknowledge financial support from the São Paulo Research Foundation (FAPESP) through grants 2025/07302-0 (Vinicius de Oliveira Rodrigues) and 2025/09425-1 (Gabriel Fernandes).

## Statements and declarations

Competing interests. The authors declare no competing interests.

Data availability. Data sharing is not applicable to this article because no datasets were generated or analyzed.

Use of generative artificial intelligence. OpenAI’s GPT-5.6 Sol, accessed through Codex, was used for brainstorming and testing ideas, producing first drafts of proofs, and assisting with bibliographic searches, proofreading, LaTeX editing, and manuscript revision. All AI-generated output and suggested references were reviewed, hand-edited, and revised by the authors, who take full responsibility for the final manuscript.

## References

1. A. Blass, Combinatorial cardinal characteristics of the continuum, Handbook of Set Theory (M. Foreman and A. Kanamori, eds.), Springer, Dordrecht, 2010, doi:10.1007/978-1-4020-5764-9\_7, pp. 395–489.

2. A. Blumer, A. Ehrenfeucht, D. Haussler, and M. K. Warmuth, Learnability and the Vapnik–Chervonenkis dimension, J. Assoc. Comput. Mach. 36 (1989), no. 4, 929–965, doi:10.1145/76359.76371.

3. M. Durst and R. M. Dudley, Empirical processes, Vapnik–Chervonenkis classes and Poisson processes, Probab. Math. Statist. 1 (1980), no. 2, 109–115, Available from Probability and Mathematical Statistics.

4. D. H. Fremlin, Measure theory. volume 2: Broad foundations, Torres Fremlin, Colchester, 2001, Available from the author’s chapter PDF.

5. V. Pestov, PAC learnability versus VC dimension: a footnote to a basic result of statistical learning, 2011 International Joint Conference on Neural Networks, IEEE, 2011, doi:10.1109/IJCNN.2011.6033352, pp. 1141–1145.

6. V. N. Vapnik and A. Ya. Chervonenkis, On the uniform convergence of relative frequencies of events to their probabilities, Theory Probab. Appl. 16 (1971), no. 2, 264–280, doi:10.1137/1116025. INSTITUTO DE MATEMÁTICA ESTATÍSTICA E CIÊNCIA DA COMPUTACÃO UNIVERSIDADE DE SÃC

nstituto de atem tica, stat stica e i ncia da omputaç o, niversidade de o Paulo<sub>,</sub> Rua do Matão 1010<sub>,</sub> 05508-090 São Paulo<sub>,</sub> SP<sub>,</sub> Brazil

Instituto de Ciências Matemáticas e de Computa<sub>ç</sub>ão<sub>,</sub> Universidade de São Paulo<sub>,</sub> Avenida Trabalhador São-carlense 400<sub>,</sub> 13566-590 São Carlos<sub>,</sub> SP<sub>,</sub> Brazil Email address: fernandes@icmc.usp.br

Instituto de Matemática<sub>,</sub> Estatística e Ciência da Computa<sub>ç</sub>ão<sub>,</sub> Universidade de São Paulo<sub>,</sub> Rua do Matão 1010<sub>,</sub> 05508-090 São Paulo<sub>,</sub> SP<sub>,</sub> Brazil Email address: vinior@ime.usp.br