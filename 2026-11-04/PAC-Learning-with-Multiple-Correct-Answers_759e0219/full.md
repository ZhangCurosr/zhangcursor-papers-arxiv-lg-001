# PAC Learning with Multiple Correct Answers

Steve Hanneke<sup>∗</sup> Farnam Mansouri<sup>†</sup> Alireza F. Pour<sup>‡</sup> Amirreza Shaeiri<sup>§</sup> Kun Wang<sup>¶</sup>

November 2026

## Abstract

In multiclass PAC learning, a prediction is correct only if it equals the label of the example, and learnability is characterized by the Natarajan dimension for finite label spaces and by the DS dimension in general. Many tasks, however, admit several correct answers: a theorem has many valid proofs, and any one of them will do. We study PAC learning when each example comes with a set of at most ℓ correct labels and a prediction is correct if it belongs to this set. Training may reveal the full set, only whether a proposed label is correct (bandit feedback), or one correct label chosen by a demonstrator (demonstrations).

Our main finding is that demonstrations can be much harder than full sets, and even than bandit feedback, both in which classes can be learned and in how many examples are needed. For finite label spaces, a generalized Natarajan dimension characterizes learnability from full sets and from bandit feedback, in both the realizable and the agnostic setting. From demonstrations, however, a class is realizably learnable if and only if the intersections of at most ℓ + 1 of its hypotheses are learnable from full sets. Already with three labels, this condition can fail for a class learnable from a single full-set example or even a single bandit answer. When the condition holds, we show that $\widetilde { O } ( \varepsilon ^ { - \ell } )$ demonstrations sufice, and the exponent ℓ cannot be improved in general, even for classes learnable from $\widetilde { O } ( 1 / \varepsilon )$ full sets or bandit answers.

For countable label spaces and $\ell \geq 2 .$ , finiteness of the DS dimension, extended to set-valued hypotheses, is neither necessary nor suficient for learning from full sets. Instead, our packed dimension, which asks for many disjoint groups of at most ℓ+1 hypotheses with no common label, characterizes full-set learnability, and demonstrations still reduce to intersections. Agnostic learning from demonstrations, already for finite label spaces, is not determined by the dimensions of the intersection class that characterize realizable learning. Under a mild nondegeneracy assumption, we characterize it by the DS dimension of an induced multiclass class, together with a local compatibility condition.

Finally, we show that the uniform bound ℓ on the number of correct labels is essential. Without this bound, it is impossible to characterize learnability by any dimension determined by finite subclasses.

Keywords: PAC Learning, Multi-Label Learning, Multiple Correct Answers, Learning from Demonstrations, Bandit Feedback, Natarajan Dimension, DS Dimension, One-Inclusion Graph, Sample Compression Scheme

## Contents

1 Introduction 4   
1.1 Multi-Label PAC Learning 4   
1.2 Main Results 6   
1.3 Organization 12   
2 Related Work 12   
3 Technical Overview 14   
3.1 Full Sets and Finite Label Spaces 15   
3.2 One Demonstrated Label 17   
3.3 Countable Label Spaces 19   
3.4 The Exponent ℓ Is Necessary 21   
3.5 Agnostic Learning from One Correct Label . 22   
3.6 Impossibility Results 23   
4 Conclusion 24   
AI Statement 25   
5 Preliminaries 25   
5.1 Multi-Label Hypotheses 26   
5.2 Learning Models 26   
5.3 Projections, Intersections, and One-Inclusion Hypergraphs 28   
5.4 Dimensions 29   
5.5 Plurality Votes 33   
6 Examples 33   
7 Full-Set Learning with Finite Labels 36   
7.1 A Transductive Mistake Bound 37   
7.2 Finite Covers 40   
7.3 Lower Bounds . 41   
7.4 Realizable Full-Set Learning 43   
7.5 Agnostic Full-Set Learning 45   
7.6 Proof of Theorem 16 47   
8 Bandit Learning with Finite Labels 48   
8.1 Transductive Bounds 48   
8.2 Inductive Bounds 49   
8.3 Confidence Amplification 52   
8.4 Proof of Theorem 29 55   
9 A Reduction between Full-Set and One-Correct-Label Feedback 55   
10 One-Correct-Label Learning with Finite Labels 61   
11 Worst-Case One-Correct-Label Sample Complexity 63   
12 Full-Set Learning with Countably Many Labels 68   
12.1 The Lower Bound . 69   
12.2 A Weak Learner from the Packed Dimension 69   
12.3 Compression and the Realizable Upper Bound 72   
12.4 Agnostic Full-Set Learning 75   
13 One-Correct-Label Learning with Countably Many Labels 77   
13.1 Short Descriptions of Partial Intersections 78   
13.2 Message Families of Single Hypotheses 79   
13.3 Proof of the Conversion Theorem 80   
13.4 The Feedback Equivalence and the Packed Rate 81   
13.5 Joint DS and Disjoint-Neighbor Bounds . 82   
14 Agnostic Learning from One Correct Label 82   
14.1 Nondegenerate Classes and the Characterization 83   
14.2 The Degenerate Obstruction 85   
14.3 Regions and Local Alphabets 87   
14.4 Necessity . 88   
14.5 Learning on the Pure Region . 92   
14.6 Learning on the Mixed Region 93   
14.7 Proof of the Characterization 97   
15 Limits of Dimension Characterizations 99   
15.1 Scale-Free Impossibility . 100   
15.2 Scale-Sensitive Impossibility 103   
References 106   
A Full-Set Learnability and the Joint DS Dimension Are Incomparable 112   
A.1 Joint DS Dimension One Does Not Imply Full-Set Learnability 112   
A.2 A Full-Set Learnable Class with Infinite Joint DS Dimension 115   
B Comparing Packed and DS-Type Dimensions of Intersection Classes 117   
B.1 Two Extraction Lemmas 118   
B.2 List Reduction 120   
B.3 Proof of the Comparison Theorem 126   
B.4 Proof of the Common-Label Lemm 129   
B.5 Proof of the Bounded-List Lemma 138   
C Connection with Cost-Sensitive PAC Learning 141   
C.1 Encoding the Learning Problem 142   
C.2 Conflicts and Cube Dimensions 143   
C.3 Quantitative Bounds 144   
D Hypergraph Coloring with Forbidden Crossings 146   
E Standard Borel Instance Spaces 149   
E.1 Assumptions and Statement 149   
E.2 Choosing the Pool Rules Measurably 150   
E.3 Finite Rules, Compression, and Prediction Families 151   
E.4 Randomization and the Remaining Arguments 153   
E.5 Agnostic Learning from Demonstrations 153

## 1 Introduction

A central question in statistical learning theory is which classes can be learned from finitely many examples [VC74]. The most prominent framework that makes this question precise is the probably approximately correct $( P A C )$ model of [Val84]. When every example has exactly one correct answer, learnability in this model is now well understood: binary classes are learnable exactly when their VC dimension is finite [VC71, BEHW89], and a line of work beginning with [Nat89] culminated in a characterization of multiclass learnability by the DS dimension [DSS14, BCD<sup>+</sup>22]. It is natural to ask what remains of this picture when an example may have several correct answers.

Many learning tasks have this form. A theorem may have many valid proofs, and a prover succeeds if it produces any one of them. Similarly, a specification can be met by many diferent programs, and a chess position may have several winning moves. In such tasks, a target assigns to every input x a set $h ^ { \star } ( x )$ of correct labels, and a prediction is correct if it belongs to this set. We ask which classes can be learned under diferent forms of feedback, and in particular when each training example reveals only one of its correct labels.

Training data of this kind, called demonstrations, are what practice usually provides. A proof written by a mathematician, or a program written by a programmer, certifies one correct answer and says nothing about the others. Supervised fine-tuning trains on exactly such data, one correct output per prompt, a setting recently studied formally by [JLB<sup>+</sup>26]. On the other hand, the existing theory of autoregressive learning [Mal24, JVB<sup>+</sup>25, HMM26] measures correctness against a single target output. As [HMM26] points out, this misses tasks such as proving a theorem, where every correct output should count. Producing one of many valid outputs is also the theme of language generation in the limit [KM24].

This paper develops a PAC theory for tasks with several correct outputs. The learner may see the whole set of correct labels, one correct label, or only whether a label it proposes is correct, as when a checker verifies a proposed proof. Three features significantly distinguish this theory from the multiclass one. First, learning from demonstrations is governed not by the class itself but by the intersections of at most ℓ + 1 of its hypotheses, where ℓ bounds the number of correct labels. Second, for infinite label spaces, the natural extension of the DS dimension no longer characterizes learnability, and we introduce a new dimension that does. Third, realizable and agnostic learnability, which coincide in the multiclass theory [DMY16, HKLM22, HMS25], come apart for demonstrations, and for nondegenerate classes, we characterize agnostic learnability by the DS dimension of an induced multiclass class, together with local compatibility.

## 1.1 Multi-Label PAC Learning

Let X be an instance space and Y a countable label space.<sup>1</sup> A multi-label hypothesis h is defined on a set $\operatorname { s u p p } ( h ) \subseteq \mathcal X$ and assigns to every $x \in \operatorname { s u p p } ( h )$ a nonempty set $h ( x ) \subseteq \mathcal { V }$ of correct labels, of size at most a fixed integer ℓ. The learner outputs a single label (unlike in multi-label classification [ZZ14]), and it succeeds if this label belongs to the correct set.

As for partial concept classes [AHHM21], h may be undefined outside supp(h), and the data are then drawn from supp(h). Partial hypotheses arise even from total classes, because the intersection of two hypotheses is undefined where their correct-label sets are disjoint. A class H is a set of multi-label hypotheses. The learner outputs a predictor $g : \mathcal { X }  \mathcal { Y }$ , whose error with respect to a target $h ^ { \star } \in { \mathcal { H } }$ and a distribution P on $\operatorname { s u p p } ( h ^ { \star } )$ is

$$
L _ { P , h ^ { \star } } ( g ) = \operatorname* { P r } _ { x \sim P } \bigl [ g ( x ) \notin h ^ { \star } ( x ) \bigr ] .
$$

During training, the learner receives independent inputs $x _ { 1 } , \ldots , x _ { m } \sim P$ together with one of three kinds of feedback:

– full-set feedback: the sets $h ^ { \star } ( x _ { 1 } ) , \ldots , h ^ { \star } ( x _ { m } )$ ;

– bandit feedback: for each $x _ { i } .$ , the learner proposes a label $u _ { i } .$ , possibly chosen adaptively, and observes whether $u _ { i } \in h ^ { \star } ( x _ { i } )$ ;

– one-correct-label feedback, or demonstrations: for each $x _ { i }$ , a label $y _ { i } \in h ^ { \star } ( x _ { i } )$ drawn independently from an unknown conditional distribution given $x _ { i } ,$ the demonstrator.

The class H is PAC learnable under a feedback model if, for every $\varepsilon , \delta \in ( 0 , 1 )$ , there are a learner and a finite sample size m such that, for every target, every distribution and, under one-correct-label feedback, every demonstrator, the learner returns from m examples a predictor with error at most ε with probability at least $1 - \delta$ . We write $m _ { \mathcal { H } } ^ { \mathrm { f u l l } } ( \varepsilon , \delta ) , m _ { \mathcal { H } } ^ { \mathrm { b a n d } } ( \varepsilon , \delta )$ and $m _ { \mathcal { H } } ^ { \mathrm { o n e } } ( \varepsilon , \delta )$ for the least such m under the three feedback models. When $\ell = 1$ , this is multiclass PAC learning.

In the agnostic setting, examples $( x , A )$ are drawn from an arbitrary distribution D on inputs and label sets, and a prediction is correct if it belongs to the sampled set A. A hypothesis predicts a whole set, and, as with the zero-one loss in the multiclass case, we count it as wrong when this set difers from the sampled one. Thus the learner competes with the benchmark

$$
L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } ) = \operatorname* { i n f } _ { h \in \mathcal { H } } \operatorname* { P r } _ { ( x , A ) \sim \mathcal { D } } [ h ( x ) \neq A ] .
$$

This benchmark is natural: for each $h \in \mathcal H$ , a predictor that outputs a label of $h ( x )$ has error at most $\operatorname* { P r } [ h ( x ) \neq A ]$ . For $\ell = 1$ , it is the usual multiclass benchmark. Under each feedback model, agnostic learning asks for error at most $L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } ) + \varepsilon$ , where demonstrations are now labels drawn from the sampled sets, and we write $m _ { \mathcal { H } } ^ { \mathrm { f u l l , a g } } ( \varepsilon , \delta )$ for the sample complexity under full-set feedback. The main question of this work is the following.

<table><tr><td>Setting</td><td>Learnable if and only if Sample complexity</td><td></td></tr><tr><td>finite , full set</td><td> $d _ { \mathrm { N } } ^ { \Omega } ( \mathcal { H } ) < \infty$ </td><td> $\begin{array} { r } { \widetilde { O } _ { \ell } \big ( ( d + \log \frac { 1 } { \delta } ) / \varepsilon \big ) } \end{array}$ </td></tr><tr><td>finite , bandit</td><td> $d _ { \mathrm { N } } ^ { \Omega } ( \mathcal { H } ) < \infty$ </td><td> $\begin{array} { r } { \widetilde { O } _ { \ell } \big ( k d \varepsilon ^ { - 1 } \log \frac { 1 } { \delta } \big ) } \end{array}$ </td></tr><tr><td>finite , one correct label</td><td> $d _ { \mathrm { N } } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) ) < \infty$ </td><td> $\begin{array} { r } { \widetilde { O } _ { \ell } \big ( d \varepsilon ^ { - \ell } \log \frac { 1 } { \delta } \big ) } \end{array}$ </td></tr><tr><td>countable  ${ \mathcal { V } } ,$  full set</td><td> $d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathcal { H } ) < \infty$ </td><td> $\begin{array} { r } { \widetilde { O } _ { \ell , \gamma } \big ( ( p + \log \frac { 1 } { \delta } ) / \varepsilon \big ) } \end{array}$ </td></tr><tr><td>countable  ${ \mathcal { V } } ,$  one correct label</td><td> $d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) ) < \infty$ </td><td> $\widetilde { O } _ { \ell , \gamma } \big ( p \varepsilon ^ { - \ell } \log \frac { 1 } { \delta } \big )$ </td></tr><tr><td>agnostic, one correct label</td><td colspan="2"> $d _ { \mathrm { D S } } ^ { \Omega } ( \widetilde { \mathcal { H } } ) < \infty$  and local compatibility (nondegenerate H)</td></tr><tr><td>no uniform bound l</td><td colspan="2">no dimension with finite witnesses</td></tr></table>

Table 1: Summary of results. Here d and p denote the dimensions in the second column, $k = | \mathcal { V } | , \mathcal { T } _ { \ell + 1 } ( \mathcal { H } )$ is the class of intersections of at most $\ell + 1$ hypotheses of H (defined before Theorem B), $\widetilde { \mathcal { H } }$ is the collapsed class (defined before Theorem F), and $\widetilde O$ hides logarithmic factors. The bounds are for the realizable setting; the bandit bound is from Theorem 29. Under full-set feedback, and under bandit feedback with finite label spaces, realizable and agnostic learnability coincide.

## 1.2 Main Results

The answer depends on the size of the label space, the available feedback, and whether the number of correct labels is uniformly bounded. Table 1 summarizes our results, and the rest of this section states them.

## Finite Label Spaces

Our first result extends the classical characterization of multiclass learnability by [Nat89] and [BDCBHL95] to several correct labels, under both full-set and bandit feedback. We use generalizations of the Natarajan and DS dimensions to set-valued hypotheses. The generalized Natarajan dimension requires, at each input, a family of correct-label sets with empty intersection, such that every combination of choices across inputs is realized by the class. Formally, distinct inputs $x _ { 1 } , \ldots , x _ { n }$ are product-shattered by H if there are finite families $\mathcal { A } _ { 1 } , \ldots , \mathcal { A } _ { n }$ of label sets such that the sets in each $\mathbf { \mathcal { A } } _ { i }$ have no common label and, for every choice of $A _ { 1 } \in { \mathcal { A } } _ { 1 } , \ldots , A _ { n } \in { \mathcal { A } } _ { n }$ , some $h \in \mathcal H$ satisfies $h ( x _ { i } ) = A _ { i }$ for all i. The generalized Natarajan dimension $d _ { \mathrm { N } } ^ { \cap } ( \mathcal { H } )$ is the largest size of a product-shattered sequence. When $\ell = 1$ a family of singletons with no common label must contain two diferent labels, so $d _ { \mathrm { N } } ^ { \bigcap }$ recovers the Natarajan dimension.

Two examples illustrate the definition. Unions of at most r intervals on the line, where both labels 0 and 1 are correct at the endpoints of the intervals, form a class with $d _ { \mathrm { N } } ^ { \cap } = 2 r$ matching the VC dimension of unions of r intervals (Example 12). In nearest-site classification on the line, a hypothesis is given by k sites $s _ { 1 } < \cdots < s _ { k }$ , and the correct labels at an input are the indices of its nearest sites, so two labels are correct at each midpoint between adjacent

sites; this class has $d _ { \mathrm { N } } ^ { \cap } = k - 1$ (Example 13).

The joint DS dimension $d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { H } )$ uses a weaker, local condition. A pattern on $x _ { 1 } , \ldots , x _ { n }$ is a tuple $( h ( x _ { 1 } ) , \ldots , h ( x _ { n } ) )$ for some $h \in \mathcal H$ defined on these inputs. A DS witness is a finite nonempty set V of patterns with the following property: for each $v \in V$ and each $x _ { i } ,$ the patterns of V that agree with v on all other inputs have no common label at $x _ { i }$ . In other words, revealing the correct-label sets at all inputs but one never leaves a label that is correct at the hidden input for all consistent patterns. The dimension $d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { H } )$ is the largest n for which there exist distinct $x _ { 1 } , \ldots , x _ { n }$ that admit a DS witness. When $\ell = 1$ , it reduces to the usual DS dimension. Section 5.4 gives the precise definitions of these and the other dimensions below.

Theorem A (Full-set and bandit feedback). Let Y be finite, and let H be a class whose correct-label sets have size at most ℓ. The following are equivalent.

$$
- \ d _ { \mathrm { N } } ^ { \cap } ( \mathcal { H } ) < \infty .
$$

$$
- d _ { \mathrm { D S } } ^ { \Omega } ( \mathcal { H } ) < \infty .
$$

– H is PAC learnable from full sets, in the realizable $o r ,$ equivalently, in the agnostic setting.

– H is PAC learnable with bandit feedback, in the realizable $o r ,$ equivalently, in the agnostic setting.

Moreover, if $d = d _ { \mathrm { N } } ^ { \cap } ( \mathcal { H } ) < \infty$ and $k = | \mathcal { D } |$ , then

$$
m _ { \mathcal { H } } ^ { \mathrm { f u l l } } ( \varepsilon , \delta ) = O _ { \ell } \Big ( \frac { 1 } { \varepsilon } \Big ( d \log \frac { ( d + 1 ) ( k + 1 ) } { \varepsilon } + \log \frac { 2 } { \delta } \Big ) \Big ) .
$$

For $d \geq 2$ and fixed failure probability, the dependence on $d / \varepsilon$ is optimal up to logarithmic factors (Proposition 23). Under bandit feedback our rates carry an extra factor $k ,$ since a uniform query tests a given label only with probability $1 / k$ . The qualitative part of Theorem A for realizable full-set learning can also be derived from [BBCB<sup>+</sup>25] (Appendix C); our direct construction adds a uniform bound with only logarithmic dependence on k for fixed ℓ.

The common starting point of our algorithms is a transductive rule that knows a pool of n inputs in advance and makes at most $O _ { \ell } ( d \log ( n k ) )$ ) mistakes. Its mistake bound yields a realizable full-set learner and a finite cover which is used for agnostic and bandit learning. Section 3 explains the counting argument behind the rule, and the proofs are given in Sections 7 and 8 (Theorems 16 and 29).

The next result shows that learning from a single correct label is governed by a diferent class. Let $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ be the class of all intersections $h _ { 1 } \cap \cdots \cap h _ { r }$ of $r \leq \ell + 1$ hypotheses of $\mathcal { H } .$ restricted to any set of inputs on which the intersection is nonempty. If H is learnable from demonstrations, then $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ is learnable from full sets. Indeed, a label in the intersection is a valid demonstration for every $h _ { i }$ , so the demonstration learner, fed with such labels, has small error for every $h _ { i }$ simultaneously and hence, by a union bound, for their intersection. The converse is the main technical result of this paper.

Theorem B (One correct label). Let Y be finite with $k = | \mathcal { D } |$ , and let H be a class whose correct-label sets have size at most ℓ. The following are equivalent.

– H is PAC learnable from one correct label.

$- ~ { \mathcal { T } } _ { \ell + 1 } ( { \mathcal { H } } )$ is PAC learnable from full sets.

$$
- \ d _ { \mathrm { N } } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) ) < \infty .
$$

$$
- \ d _ { \mathrm { D S } } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) ) < \infty .
$$

Moreover, if $d = d _ { \mathrm { N } } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) ) < \infty$ , then

$$
m _ { \mathcal { H } } ^ { \mathrm { o n e } } ( \varepsilon , \delta ) = O _ { \ell } \Big ( \frac { d + 1 } { \varepsilon ^ { \ell } } \log ^ { 2 } \Big ( \frac { ( d + 2 ) ( k + 1 ) } { \varepsilon } \Big ) \log \frac { 2 } { \delta } \Big ) .
$$

The intersection class cannot be replaced by H itself, even for a class learnable from a single full-set example. In Example 15, every hypothesis but one accepts the label 1 everywhere, together with a label in {2, 3} that depends on the input, and the remaining hypothesis accepts exactly 2 and 3 everywhere. The dimensions $d _ { \mathrm { N } } ^ { \cap }$ and $d _ { \mathrm { D S } } ^ { \cap }$ of this class equal one, and a single full-set example, or a single bandit answer, tells the two kinds of targets apart, after which a constant prediction is always correct. A demonstrator who never shows the label 1, however, reveals nothing about the kind of the target, and the class is not learnable from one correct label.

The number ℓ + 1 comes from a simple fact: if sets of size at most ℓ have no common label, then at most $\ell + 1$ of them already have no common label (Lemma 8). Deeper intersections therefore add nothing for learnability: if $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ is learnable from full sets, then so is $\mathcal { T } _ { R } ( \mathcal { H } )$ for every fixed R (Corollary 58). The dependence on $\varepsilon ^ { - \ell }$ is also necessary in the worst case, as Theorem C shows. The proof of Theorem B is in Sections 9 and 10 (Theorem 41).

Theorem C (The exponent ℓ is necessary). For every $\ell \geq 2$ and $d \geq 1$ , there is a class H with label space [ℓ] such that $d \leq d _ { \mathrm { N } } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) ) \leq d + ( \ell + 1 ) ^ { \ell }$ and

$$
m _ { \mathcal H } ^ { \mathrm { o n e } } ( \varepsilon , 1 / 8 ) > \frac { d } { 4 ( 3 2 ( \ell + 1 ) \varepsilon ) ^ { \ell } } \qquad f o r \ e v e r y \ 0 < \varepsilon \leq 1 / 2 5 6 .
$$

Since $\mathcal { H } \subseteq \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } )$ , this class has $d _ { \mathrm { N } } ^ { \cap } ( \mathcal { H } ) \leq d + ( \ell + 1 ) ^ { \ell }$ , so by Theorem A it is learnable from full sets with $\widetilde { O } _ { \ell } ( ( d + \log ( 1 / \delta ) ) / \varepsilon )$ examples. Thus one-correct-label feedback can cost a factor of order $\varepsilon ^ { - ( \ell - 1 ) }$ compared with full-set feedback. This is a statement about the worst class of a given dimension: some classes with the same dimension are learnable from demonstrations with $O ( ( d + \log ( 1 / \delta ) ) / \varepsilon )$ examples (Remark 44). The proof is in Section 11 (Theorem 43).

## Countable Label Spaces

When Y is infinite, finiteness of the generalized Natarajan dimension no longer sufices, already for $\ell = 1 \ \mathrm { [ B C D ^ { + } 2 2 ] }$ . One might expect the joint DS dimension to take over, as in the multiclass case. We show that it does not once $\ell \geq 2$ , and we introduce a dimension that characterizes learnability. The dificulty is that, with sets of size two or more, an empty intersection may force very few errors: in a DS witness, if all but one of the candidates that remain at the hidden input contain the label 1, predicting 1 is wrong for only one of them. Our dimension therefore asks for many disjoint small conflicts at every input, each of which forces an error whatever the learner predicts.

Fix $0 < \gamma < 1 / ( \ell + 1 )$ . Consider a finite nonempty list of patterns on distinct inputs $x _ { 1 } , \ldots , x _ { n } ,$ possibly with repetitions. A conflict block at $x _ { i }$ is a set of at most $\ell + 1$ entries of the list that agree on all inputs other than $x _ { i }$ and have no common label at $x _ { i } ,$ so every label predicted at $x _ { i }$ is wrong for at least one entry of the block. The packed dimension $d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathcal { H } )$ is the largest n for which some list has, at every $x _ { i }$ , pairwise disjoint conflict blocks that together contain at least $\gamma$ times as many entries as the list.<sup>2</sup> Such a list forces errors on at least a ${ \bf \nabla } \cdot { \bf \gamma } \gamma / ( \ell + 1 )$ fraction of its entries at every input.

Theorem D (Full-set feedback). Let Y be countable, and let H be a class whose correct-label sets have size at most ℓ. The following are equivalent.

$$
- d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathcal { H } ) < \infty .
$$

– H is PAC learnable from full sets.

– H is agnostically PAC learnable from full sets.

Moreover, if $p = d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathcal { H } ) < \infty$ and $B = p + 1 + \log ( 2 / \delta )$ , then

$$
m _ { \mathcal { H } } ^ { \mathrm { f u l l } } ( \varepsilon , \delta ) = O _ { \ell , \gamma } \Big ( \frac { B } { \varepsilon } \log ^ { 2 } \frac { B } { \varepsilon } \Big ) ,
$$

$$
m _ { \mathcal { H } } ^ { \mathrm { f u l l , a g } } ( \varepsilon , \delta ) = O _ { \ell , \gamma } \Big ( \frac { B } { \varepsilon ^ { 2 } } \log ^ { 2 } \frac { B } { \varepsilon } \Big ) .
$$

In contrast, finiteness of the joint DS dimension is neither necessary nor suficient: for every $\ell \geq 2$ , there are two total classes with at most two correct labels per input such that one has $d _ { \mathrm { D S } } ^ { \Omega } ( \mathcal { H } ) = 1$ and is not learnable from full sets, and the other has $d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { H } ) = \infty$ and is learnable from full sets.

For the upper bounds, the packed dimension yields a weak learner, which we boost into a sample compression scheme. The proof is in Section 12 (Theorem 45), and the two separating classes are in Appendix A (Theorems 90 and 91).

For learning from demonstrations, the reduction to the intersection class holds for countable label spaces as well. Together with the preceding full-set characterization, this gives a characterization in terms of the packed dimension of $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ . The next theorem also gives suficient conditions in terms of $d _ { \mathrm { D S } } ^ { \cap }$ and of the disjoint-neighbor dimension $d _ { \mathrm { p a i r } }$ , which strengthens the DS condition: for each pattern v of the witness and each $x _ { i } .$ , some pattern of the witness that agrees with v on all other inputs must have a correct-label set at $x _ { i }$ that is disjoint from that of v.

Theorem E (One correct label). Let Y be countable, and let H be a class whose correct-label sets have size at most ℓ. The following are equivalent.

– H is PAC learnable from one correct label.

$- ~ { \mathcal { T } } _ { \ell + 1 } ( { \mathcal { H } } )$ is PAC learnable from full sets.

$$
- \ d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) ) < \infty .
$$

Moreover, $i f p = d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) ) < \infty$ , then

$$
m _ { \mathcal { H } } ^ { \mathrm { o n e } } ( \varepsilon , \delta ) = O _ { \ell , \gamma } \Big ( \frac { p + 1 } { \varepsilon ^ { \ell } } \log ^ { 2 } \frac { 2 } { \varepsilon } \log \frac { 2 } { \delta } \Big ) .
$$

Finally, if $D = d _ { \mathrm { D S } } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) )$ or $D = d _ { \mathrm { p a i r } } ( \mathscr { T } _ { \ell + 1 } ( \mathcal { H } ) )$ is finite, then $p = O _ { \ell , \gamma } ( D \log ( D + 2 ) )$ 2 so H is PAC learnable from one correct label.

The reduction converts every full-set learner for $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ with small constant error into a demonstration learner whose sample size is linear in that of the full-set learner (Corollary 57). The packed dimension supplies such a learner with $O _ { \ell , \gamma } ( p + 1 )$ examples, which is why the bound is linear in $p + 1$ . The proof is in Section 13 and Appendix B. Whether finiteness of $d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { T } _ { \ell + 1 } ( \mathcal { H } ) )$ is also necessary is left open (Section 4).

## Agnostic Learning from One Correct Label

Under full-set feedback, and under bandit feedback with finite label spaces, realizable and agnostic learnability coincide (Theorems A and D), as in the multiclass case [DMY16, HKLM22, HMS25]. Under one-correct-label feedback they do not. A demonstration reveals only one label of the sampled set, so diferent distributions of correct-label sets may produce the same demonstrations and yet call for diferent predictions. In particular, every distribution of demonstrations is also produced by singleton sets, namely the demonstrated labels themselves, and the learner cannot rule this out.

Our characterization uses two conditions on H and an associated multiclass class. We call H nondegenerate if it contains two hypotheses that are defined at every input, have singleton values, and disagree at every input. These hypotheses play the role of the two constant hypotheses in binary classification; for example, every class containing $x \mapsto \{ 1 \}$ and x 7→ {2} is nondegenerate.

We call H locally compatible if, at each input x, every hypothesis with at least two labels at x contains the label of every hypothesis with a single label at x. Therefore, if a learner predicts only labels that occur as singleton values at x, its prediction is accepted by every hypothesis with at least two correct labels there.

This motivates defining the collapsed class $\widetilde { \mathcal { H } }$ , which replaces each singleton $\{ a \}$ with the label a and each set of at least two labels with one new label ⋄, while undefined values stay undefined. It is a multiclass class over $\mathcal { V } \cup \{ \circ \}$ , so $d _ { \mathrm { D S } } ^ { \cap } ( \widetilde { \mathcal { H } } )$ is its usual DS dimension.

Theorem F (Agnostic learning from one correct label). Let Y be countable, and let H be a class whose correct-label sets have size at most ℓ. Assume that X is countable, or that X is standard Borel and H satisfies the regularity assumptions of Appendix E.

– If H is nondegenerate, then H is agnostically PAC learnable from one correct label if and only if $d _ { \mathrm { D S } } ^ { \Omega } ( \widetilde { \mathcal { H } } ) < \infty$ and H is locally compatible.

– Nondegeneracy cannot be dropped. For $\ell = 2$ , there is a locally compatible class H on a single input with three labels such that $d _ { \mathrm { D S } } ^ { \Omega } ( \widetilde { \mathcal { H } } ) = 0$ and the four dimensions $d _ { \mathrm { N } } ^ { \cap } , ~ d _ { \mathrm { D S } } ^ { \cap } ,$ $d _ { \mathrm { p a i r } }$ and $d _ { \mathrm { p a c k } , \gamma } ^ { \cap }$ of $\mathcal { T } _ { 3 } ( \mathcal { H } )$ all equal one, but every learner, at every sample size, has expected error at least $L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } ) + 1 / 9$ under some distribution and demonstrator. In particular, H is not agnostically PAC learnable from one correct label.

The class in the second part consists of the hypotheses with correct-label sets {1, 2}, {1, 3} and {2, 3} at its only input, and three distributions of correct-label sets give uniform demonstrations but have diferent best predictions. The class of the three hypotheses $x \mapsto \{ j \}$ on one input has an intersection class with the same four dimensions and is agnostically learnable (Remark 68), so these dimensions do not determine agnostic learnability.

For the upper bound in the first part, the learner treats separately the inputs where every defined hypothesis value is a singleton, on which an agnostic multiclass learner sufices, and the remaining inputs, on which it predicts only labels that occur as singleton values. Conversely, local compatibility is necessary for every class, and the DS condition follows from a multiclass lower bound and an indistinguishability argument. Section 3 explains both directions. The proofs are in Section 14 (Theorems 64 and 67).

## Limits of Dimensions

All the results above assume a uniform bound ℓ on the number of correct labels, and our last result shows that this assumption cannot simply be dropped. Say that a parameter D of classes has finite witnesses if $\mathsf { D } ( \mathcal { H } )$ is the supremum of D over the finite subclasses of $\mathcal { H } ;$ all the dimensions above have this property.

Theorem G (Impossibility of scale-free and scale-sensitive dimensions). Consider countable classes of total hypotheses over countable instance and label spaces.

(a) Scale-free. Under full-set or one-correct-label feedback, with correct-label sets that are finite but not uniformly bounded in size, no parameter with finite witnesses is finite exactly on the PAC learnable classes, and no parameter D taking integer values on all learnable classes admits a finite-valued Ψ that bounds the sample complexity of every learnable class H by $\Psi ( \mathsf { D } ( \mathcal { H } ) , \varepsilon , \delta )$

(b) Scale-sensitive. Under one-correct-label feedback, with correct-label sets that may be infinite, consider families $( \mathsf { D } _ { \alpha } ) _ { \alpha }$ of relabeling-invariant parameters with finite witnesses. No such family is finite at all scales exactly on the learnable classes. Nor does any such family taking integer values on all learnable classes admit a finite-valued Ψ and a scale map $\varepsilon \mapsto \alpha ( \varepsilon )$ that bound the sample complexity of every learnable class H by $\Psi ( \operatorname { D } _ { \alpha ( \varepsilon ) } ( \mathcal { H } ) , \varepsilon , \delta )$

The proofs are in Section 15 (Theorems 86 and 88). Related limitations of dimensions were found by [LBD24] for learning classes of distributions.

## Hypergraph Coloring

The counting argument behind the transductive rule of Theorem A is not specific to learning. Consider an r-uniform hypergraph whose edges are covered by m complete r-partite hypergraphs with forbidden crossings, meaning that for any two of them, some part of one is disjoint from some part of the other. The argument shows that such a hypergraph has chromatic number exp $( O _ { r } ( \log ^ { 2 } ( m + 1 ) ) )$ ) (Appendix D). This extends the quasipolynomial bound of [MV09], for partitions of the edges of a graph into complete bipartite graphs, to r-uniform hypergraphs and to covers with forbidden crossings. For partitions of graphs, [Fox26] gives a sharper leading constant in the exponent.

## 1.3 Organization

Section 2 discusses related work, Section 3 gives an overview of the proofs, and Section 4 concludes with open problems. The technical part begins with the formal definitions in Section 5 and four examples in Section 6. Theorem A is proved in Sections 7 and 8, Theorem B in Sections 9 and 10, Theorem C in Section 11, Theorem D in Section 12, Theorem E in Section 13, Theorem F in Section 14, and Theorem G in Section 15. The appendices contain the separating examples for Theorem D (Appendix A), the comparison of dimensions for Theorem E (Appendix B), the hypergraph coloring bound (Appendix D), and general instance spaces (Appendix E).

## 2 Related Work

Multiclass PAC learning. When every example has exactly one correct answer (ℓ = 1), learning from full sets or from demonstrations is exactly multiclass PAC learning. For finite label spaces, learnability is characterized by the Natarajan dimension [Nat89, BDCBHL95]. Infinite label spaces behave diferently: Daniely et al. [DSBDSS11] showed that empirical risk minimizers for the same class can have very diferent sample complexities, and Daniely and Shalev-Shwartz [DSS14] introduced the DS dimension and proved that its finiteness is necessary for learnability. Brukhim et al. [BCD<sup>+</sup>22] proved that it is also suficient, and that finite Natarajan dimension does not sufice. Their learner combines the one-inclusion graph [HLW94, RBR06, DSS14] with list learning and sample compression. List learning, in which the learner outputs a short list of labels, was also characterized in the PAC setting by Charikar and Pabbaraju [CP23]; it is dual to our problem, in which the target, not the learner, provides several labels. Recently, building on an optimal Sauer lemma over k-ary alphabets [HMMS26a], Pabbaraju [Pab26a] determined the optimal sample complexity of multiclass and list learning. Our results show which parts of this picture survive when an input has up to ℓ correct labels. For finite label spaces, the generalized Natarajan dimension still characterizes learnability from full sets (Theorem A). For countable label spaces, however, the joint DS dimension no longer does so once $\ell \geq 2$ , and the packed dimension takes its place (Theorem D).

Partial concept classes. Alon et al. [AHHM21] developed the theory of PAC learning for partial concept classes, whose hypotheses may be undefined on part of the domain. We use this framework even for total classes H, because the intersection class $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ of Theorems B and E may be partial: an intersection of correct-label sets may be empty.

Online learning with several correct answers. Raman, Subedi and Tewari [RST24] study a model in which the learner predicts one label and then observes the whole set of correct labels, an online counterpart of full-set feedback. They show that deterministic and randomized online learnability difer, and they characterize both by dimensions in the realizable setting. Pour, Mansouri and Ben-David [PMBD26] study online learning with multiple correct answers under three feedback models. For each model, they determine the optimal realizable mistake bound in terms of a combinatorial dimension, describe the possible agnostic regret rates, and derive batch sample complexity bounds from the same dimensions. The agnostic obstruction in the second part of Theorem F adapts their indistinguishability argument. Shao et al. [SFLT26] study an online model in which the learner observes one correct label per round, and Kosoy [Kos26] studies one in which a prediction may consist of several labels. We work directly in the PAC setting and characterize both realizable and agnostic learnability, from full sets and from demonstrations for finite and countable label spaces, and from bandit feedback for finite ones. For agnostic learning from demonstrations, the characterization holds for nondegenerate classes.

Learning from correct demonstrations. Closest to our work is the statistical study of learning to answer from correct demonstrations by Joshi et al. [JLB<sup>+</sup>26]. For each training prompt, the learner observes one correct answer, and an unknown reward model from a known class specifies which answers are correct. For finite reward classes, their sample complexity is logarithmic in the class size and inverse-linear in accuracy, up to logarithmic factors, even for adaptive correct demonstrations. They also show that maximizing demonstration likelihood can fail. Theorems B and E characterize realizable learnability from i.i.d. demonstrations for possibly infinite classes with at most ℓ correct labels per input: such a class is learnable exactly when its intersection class $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ is learnable from full sets. The dependence on the accuracy can then be far worse than inverse-linear: the classes of Theorem C are infinite and have finite $d _ { \mathrm { N } } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathscr { H } ) )$ , yet their sample complexity is of order at least $\varepsilon ^ { - \ell }$

Partial-label learning. Partial-label, or superset, learning [CST11, LD14] reverses the role of the label set: each example comes with a set of candidate labels of which only one is correct. In our model every label in the set is correct, and under one-correct-label feedback the learner observes a single element of this set.

Agnostic one-inclusion graphs. The packed dimension gives a combinatorial certificate of dificulty in the weighted realizable one-inclusion prediction game, using conflict blocks of at most ℓ + 1 hypotheses. The analogous agnostic full-set game allows arbitrary assignments of correct-label sets and subtracts their minimum exact-set disagreement with H, just as the agnostic one-inclusion framework of [ADD<sup>+</sup>24] subtracts Hamming-distance credits. Our packed witnesses consist of realizable assignments, for which these credits are zero.

Bandit feedback. Under bandit feedback, the learner proposes a label and observes only whether it is correct. For multiclass classification, this feedback has been studied in the online setting [KSST08, DH13, Lon20, RRS<sup>+</sup>24, ECK<sup>+</sup>24b, FHMM24, HSW25a, HSW25b] and in the statistical setting [ECK<sup>+</sup>24a, HSZ25, HSW25b, HMMS26b]. With several correct labels, a positive answer reveals one correct label and a negative answer excludes one label. For finite label spaces, this feedback still sufices: bandit learnability is characterized by the same dimension as full-set learnability (Theorem A). Our bandit learners make independent uniform queries during training, as in the batch models of Daniely et al. [DSBDSS11] and Hanneke et al. [HMMS26b].

Limits of dimensions. Ben-David et al. [BDHM<sup>+</sup>19] showed that EMX (estimating the maximum) learnability can be independent of ZFC set theory, which rules out a characterization of EMX learnability by a dimension of finite character. Hanneke and Yang [HY23] established independence from ZFC for identifying near-optimal arms in structured bandits. For classes of distributions, Lechner and Ben-David [LBD24] established limitations of dimensions as characterizations of learnability, and they showed that no scale-sensitive dimension characterizes the learnability of such classes [LBD26]. Theorem G is a result of the same kind for multi-label PAC learning. When correct-label sets are finite but not uniformly bounded, no parameter with finite witnesses (a requirement in the spirit of finite character) is finite exactly on the learnable classes. This holds under full-set and under one-correct-label feedback, and here the obstruction is combinatorial rather than set-theoretic.

## 3 Technical Overview

Most of the dificulty in our setting comes from one mismatch: the learner has to output only one of several correct labels, while the data may reveal all of them, only one of them, or only whether a guess is correct. For each upper bound we start from the natural first approach, show where this mismatch breaks it, and then describe what works. For the lower bounds and obstructions we describe the hard classes directly. Formal definitions are in Section 5.

Section 3.1 treats full-set feedback over a finite label space through a potential argument, and Section 3.2 reduces learning from one demonstrated label to full-set learning of the intersection class. We then extend both results to countable label spaces in Section 3.3, and show in Section 3.4 that the factor $\varepsilon ^ { - \ell }$ in Theorem B cannot be avoided. We end with agnostic learning from demonstrations in Section 3.5, and with classes that have no uniform bound on the number of correct labels in Section 3.6, where we show that no parameter with finite witnesses characterizes learnability.

## 3.1 Full Sets and Finite Label Spaces

When $k = | \mathcal { D } |$ is finite, a natural first idea is to view H as a multiclass class whose labels are the correct-label sets, and to learn the whole set $h ^ { \star } ( x )$ by applying the characterization based on the Natarajan dimension of this class. This can be much harder than predicting one correct label. For example, on an infinite $\mathcal { X } _ { : }$ , let H contain every h with $h ( x ) \in \{ \{ 1 , 2 \} , \{ 1 , 3 \} \}$ } for all x. As a multiclass class, H has infinite Natarajan dimension, yet the constant predictor 1 is always correct without requiring any training samples.

A dimension for our problem should therefore count only choices of sets with no common label. This is what the generalized Natarajan dimension $d _ { \mathrm { N } } ^ { \cap } ( \mathcal { H } )$ introduced in Section 1.2 does: product-shattering uses only families of sets with empty intersection. Product-shattered inputs force errors under full-set feedback, and a DS witness forces errors under each of our feedback models; these lower bounds complete the characterization (Propositions 22 and 23). The rest of this subsection describes the upper bounds.

Counting realized products. Our main tool is a transductive rule that receives a pool $S = ( x _ { 1 } , \ldots , x _ { n } ) $ of inputs. The positions of $S$ are then presented one at a time, in any order. At each step, the rule predicts a label and then observes the correct-label set. We show that it makes $O _ { \ell } ( d \log ( n k ) )$ mistakes, where $d = d _ { \mathrm { N } } ^ { \Omega } ( \mathcal { H } )$

For binary classes, where $d _ { \mathrm { N } } ^ { \cap }$ is the VC dimension, such a rule can be obtained by counting shattered sets. Let the version space W be the set of patterns on $S$ that agree with the labels seen so far, let $\Phi ( W )$ be the number of sets of positions shattered by $W$ , the empty set included, and let $W _ { i , \{ y \} }$ consist of the patterns of W with label y at position i.

Every set shattered by $W _ { i , \{ 0 \} }$ or by $W _ { i , \{ 1 \} }$ omits i and is shattered by W. If a set is shattered by both, then adding i to it gives a set shattered by $W$ that neither half shatters. Thus every set counted twice in the sum Φ $( W _ { i , \{ 0 \} } ) + \Phi ( W _ { i , \{ 1 \} } )$ is paired with a set counted zero times, and hence the sum is at most $\Phi ( W )$ . So at most one label y has $\Phi ( W _ { i , \{ y \} } ) > \Phi ( W ) / 2$ and predicting such a label, or either label if none exists, at least halves $\Phi ( W )$ on every mistake. Since only sets of at most d positions can be shattered, initially $\Phi ( W ) \leq ( n + 1 ) ^ { d }$ and the rule makes at most $d \log _ { 2 } ( n + 1 )$ mistakes.

We extend this rule to correct-label sets. The version space W is now a set of tuples on $S ;$ initially it consists of the tuples $( h ( x _ { 1 } ) , \ldots , h ( x _ { n } ) )$ for all $h \in \mathcal H$ . A product realized by $W$ specifies $\mathrm { a }$ set $T$ of positions and, at each $i \in T$ , a minimal family of sets with empty intersection (no proper subfamily has empty intersection), such that every combination of one set from each family occurs in $W$ at the positions of $T .$ . For binary classes, these products are exactly the shattered sets.

Let $\Phi ( W )$ now be the number of products realized by W, including the empty product with $T = \emptyset$ . Each of them product-shatters the inputs at the positions of $T _ { i }$ so $| T | \leq d .$ A minimal family has at most $\ell + 1$ members (Lemma 8), and as each member has size at most $\ell ,$ there are at most $k ^ { O _ { \ell } ( 1 ) }$ such families. Hence, log $\Phi ( W ) = O _ { \ell } ( d \log ( n k ) )$ .

For a set A, write $W _ { i , A }$ for the tuples of W with value A at position i. The corresponding inequality for set-valued hypotheses (Lemma 17) is that, for every minimal family $\{ A _ { 1 } , \ldots , A _ { r } \}$ with empty intersection,

$$
\sum _ { j = 1 } ^ { r } \Phi ( W _ { i , A _ { j } } ) \leq ( r - 1 ) \Phi ( W ) .
$$

Indeed, no product realized by some $W _ { i , A _ { j } }$ uses position $i ,$ so a product of $W$ that uses position i is counted zero times on the left. A product counted r times is realized by every $W _ { i , A _ { j } }$ , and adding the family $\{ A _ { 1 } , \ldots , A _ { r } \}$ at position i turns it into a product of W that uses position i. Thus every product counted r times is paired with a distinct product counted zero times, and every other product is counted at most $r - 1$ times.

Since $r \leq \ell + 1$ , this inequality implies that the sets A with $\begin{array} { r } { \Phi ( W _ { i , A } ) > \frac { \ell } { \ell + 1 } \Phi ( W ) } \end{array}$ have a common label. Indeed, otherwise, by Lemma 8, some minimal subfamily of $r \leq \ell + 1$ of them has empty intersection, and the left side exceeds $\begin{array} { r } { r \cdot \frac { \ell } { \ell + 1 } \Phi ( W ) \geq ( r - 1 ) \Phi ( W ) } \end{array}$ . The rule predicts such a common label, or any label if there are no such sets. After a mistake, the revealed set A does not contain the prediction, so it is not one of these sets, and replacing W by $W _ { i , A }$ multiplies $\Phi ( W )$ by at most $\ell / ( \ell + 1 )$ . The tuple of the target always remains in W, so $\Phi ( W ) \geq 1$ throughout. Hence the rule makes $O _ { \ell } ( d \log ( n k ) )$ mistakes on every pool and in every order of presentation (Theorem 19). Standard leave-one-out arguments turn this mistake bound into the realizable learners (Theorems 24 and 25).

Finite covers. The rule changes its version space only after a mistake, so its run against a target is determined by a short list: the at most $O _ { \ell } ( d \log ( n k ) )$ positions where it makes a mistake, together with the sets revealed there. For each such list, we simulate the rule with these answers and record its predictions, with the prediction at each listed position replaced by a label of the revealed set. These vectors form a finite cover $F _ { S }$ of $\mathcal { H }$ on $S { \mathrm { : } }$ every $h \in \mathcal H$ has a member that is correct at every input of S where $h$ is defined. This cover can be built from the inputs $S = ( x _ { 1 } , \ldots , x _ { n } ) $ alone, without any correct-label sets, and log $| F _ { S } | = O _ { \ell } ( d \log ^ { 2 } ( n k ) )$ (Lemma 21). Running exponential weights over its members on the training inputs and the test input, in a random order, gives the agnostic full-set learner (Theorem 27).

Bandit feedback. On a known pool, the cover gives a bandit learner by elimination: predict the label recommended by the largest number of surviving members of $F _ { S }$ , and after a negative answer remove every member that recommended it. Each mistake removes at least a $1 / k$ fraction of the survivors, since the predicted label has the most votes, and a member that is correct for the target is never removed (Proposition 30). In the batch setting, the dificulty is that $F _ { S }$ depends on the test input, which is not known when the training queries are made. We overcome this by querying an independent uniformly random label at each training input and recording the answer. Once the test input arrives, the learner constructs the cover of the combined pool and processes the recorded queries and answers in a uniformly random order. A query reveals that an expert is wrong if the queried label is the one recommended by that expert and the answer is negative. Thus a wrong expert is detected with probability $1 / k$ , while a correct expert is never detected as wrong. In the realizable case, the learner eliminates the detected experts. In the agnostic case, multiplying the indicator of this event by k gives an unbiased estimate of the expert’s loss, which is used in exponential weights. A leave-one-out argument then gives the expected-risk bounds (Section 8).

## 3.2 One Demonstrated Label

The easy direction of Theorem B, from demonstrations for H to full sets for $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ , was explained in Section 1.2 (Lemma 37). We now describe the converse, which Section 9 proves for every countable label space.

A natural approach is to choose a hypothesis consistent with the demonstrations and predict one of its labels, as in realizable PAC learning. Consistency alone does not sufice, since observing one correct label does not determine the others. For instance, if $h ^ { \star } ( x ) = \{ 1 , 2 \}$ and $h ( x ) = \{ 2 , 3 \}$ at every input, a demonstrator who always shows the label 2 makes h consistent with every sample, while the label 3 of h is always wrong. Rather than choosing a single consistent hypothesis, we retain a collection of candidates and seek predictions with small error against all of them. By Lemma 8, if the correct-label sets of a collection of candidates have empty intersection at some input, then this is already witnessed by at most $\ell + 1$ candidates. Thus it sufices to control agreement among groups of $\ell + 1$ candidates, which explains the role of $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$

The learner works on the pool ${ \cal S } = ( x _ { 1 } , \dots , x _ { N } )$ that consists of the $N - 1$ training inputs and the test input, and its candidates are the hypotheses of H defined on all of S. It draws a random set J of $\lfloor N / 2 \rfloor$ training positions, uses only the demonstrations at these positions to select the candidates, and predicts on the remaining positions $U = [ N ] \backslash J _ { : }$ , which include the test position. We say that $\ell + 1$ candidates conflict at a position if their sets have empty intersection there. The argument has two steps. First, if the learner retains a set of candidates in which any $\ell + 1$ members conflict only on a small part of $U$ , then there is a randomized prediction rule with small average error on $U$ against every retained candidate. Second, the demonstrations on J sufice to retain such a set of candidates that contains the target with high probability.

Agreement of small groups sufices. Suppose that the learner has a collection of candidates that contains the target and in which any $\ell + 1$ members of the collection conflict on at most a $\theta$ fraction of the positions of U. We show that there is a randomized prediction rule whose average error on $U$ is at most $\theta ^ { 1 / \ell }$ against every candidate in the collection.

By the minimax theorem, it sufices to show that for every distribution $\mu$ on the candidates, there are labels with small expected error against $h \sim \mu$ . Against $\mu ,$ at each position $i \in U$

the best label at $x _ { i }$ is wrong with probability

$$
\nu _ { i } = 1 - \operatorname* { m a x } _ { y \in \mathcal { V } } \operatorname* { P r } _ { h \sim \mu } [ y \in h ( x _ { i } ) ] ,
$$

so every label y satisfies $\operatorname* { P r } _ { h \sim \mu } [ y \neq h ( x _ { i } ) ] \geq \nu _ { i }$ . Draw $h _ { 1 } , \ldots , h _ { \ell + 1 }$ independently from $\mu$ and condition on $h _ { 1 } ( x _ { i } )$ , which has at most ℓ labels. Assign each of these labels to a diferent candidate among $h _ { 2 } , \ldots , h _ { \ell + 1 }$ (some may stay unassigned). Each assigned candidate excludes its label with probability at least $\nu _ { i } .$ , independently of the others, and if all of them do, the $\ell + 1$ sets have empty intersection at $x _ { i }$ . Hence $h _ { 1 } , \ldots , h _ { \ell + 1 }$ conflict at $x _ { i }$ with probability at least $\nu _ { i } ^ { \ell } .$ Since any $\ell + 1$ candidates conflict on at most a $\theta$ fraction of $U$ , averaging over $i \in U$ gives

$$
\frac { 1 } { | U | } \sum _ { i \in U } \nu _ { i } ^ { \ell } \le \theta .
$$

By Jensen’s inequality, the average of the $\nu _ { i }$ over $U$ is at most $\theta ^ { 1 / \ell }$ . Thus, to obtain average error at most $\varepsilon ,$ we should bound the fraction of conflicts by $\varepsilon ^ { \ell }$ . This is the source of the $\varepsilon ^ { - \ell }$ dependence in the sample complexity bound.

Making conflicts rare. The demonstrations show agreement only on $J { \boldsymbol { : } }$ all candidates consistent with them share the demonstrated label at every position of J, but their correctlabel sets need not intersect on $U$ . To control these conflicts, we use a random-split argument with a union bound. However, there may be infinitely many candidates, so we apply it to finitely many label vectors instead. For every candidate h defined on $S _ { \mathrm { { ; } } }$ , the learner uses a family $F _ { h , S }$ of at most $B _ { N }$ label vectors on $S _ { \mathrm { { ; } } }$ , determined by h and $S$ alone. The families are chosen so that we have the following intersection property: for any $\ell + 1$ candidates $h _ { 1 } , \ldots , h _ { \ell + 1 }$ , some $f \in \bigcap _ { j = 1 } ^ { \ell + 1 } F _ { h _ { j } , S }$ predicts a label in $\textstyle \bigcap _ { j = 1 } ^ { \ell + \bar { 1 } } h _ { j } ( x _ { i } )$ whenever this intersection is nonempty (Definition 38). For a threshold α fixed below, the learner retains a candidate h if

(i) h accepts every demonstrated label in $J ,$ and

(ii) every $f \in F _ { h , S }$ that is correct for h on all of J is wrong for h on at most an α fraction of $S .$

Both conditions depend only on the demonstrations and the candidate $h ,$ not on the unobserved correct-label sets of the target.

By exchangeability, we may analyze the expected loss by choosing J uniformly among the $\lfloor N / 2$ ⌋-subsets of [N], then choosing the test position uniformly from $U = [ N ] \backslash J$ . We first show that the target is retained with high probability. It satisfies (i) because its demonstrated labels are correct. For (ii), a fixed $f$ that is wrong for $h ^ { \star }$ on more than an α fraction of the pool is correct on all of the random set J with probability at most $e ^ { - \alpha \lfloor N / 2 \rfloor }$ . Since $F _ { h ^ { \star } , S }$ is determined independently of $J ,$ a union bound over the $B _ { N }$ members of $F _ { h ^ { \star } , S }$ shows that the target is retained with probability at least $1 - 1 / N$ once we take α of order (log $B _ { N } { + } \log N ) / N$

We next show that the retained candidates rarely conflict. Any $\ell + 1$ of them all accept the demonstrated labels, so their sets intersect at every position of J, and the intersection property gives a vector $f$ in all of their families that is correct for all of them on J. By (ii), $f$ makes at most $\alpha N$ mistakes against each candidate on the entire pool. At every position where the candidates conflict, $f$ must be wrong for at least one of them. Thus there are at most $( \ell + 1 ) \alpha N$ conflicting positions, and since $| U | \ge N / 2$ , the fraction of conflicts on $U$ is at most $2 ( \ell + 1 ) \alpha$

The resulting bound. With $\theta = O ( \ell \alpha )$ , the minimax step gives randomized predictions on U whose average error against every retained candidate, and in particular against the target, is $O ( ( \ell \alpha ) ^ { 1 / \ell } )$ . By exchangeability, the test position is a uniform position of $U _ { : }$ , so this average error, plus the probability $1 / N$ of discarding the target, bounds the expected test error (Theorem 39). Thus, with this choice of $\alpha , N$ of order $\varepsilon ^ { - \ell } ( \log B _ { N } + \log N )$ gives expected error ε. Repeating the learner on independent blocks and taking a most probable label of the average prediction distribution gives a high-probability guarantee with an additional $O _ { \ell } ( \log ( 2 / \delta ) )$ factor in the sample size (Lemma 40).

For a finite label space, a single family serves every candidate: the finite cover of $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ on $S$ from Section 3.1 contains, for any $\ell + 1$ candidates, a vector that is correct for their intersection wherever it is defined. Hence log $B _ { N } = O _ { \ell } ( d \log ^ { 2 } ( N k ) )$ with $d = d _ { \mathrm { N } } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) )$ which gives Theorem B (Section 10).

## 3.3 Countable Label Spaces

When Y is infinite, the bounds above are vacuous because they grow with k. For $\ell = 1$ learnability is characterized by the DS dimension $\mathrm { [ B C D ^ { + } 2 2 ] }$ , so the joint DS dimension is the natural candidate for larger ℓ. For $\ell \geq 2$ , however, it fails in both directions (Theorem D and Appendix A).

The packed dimension of Section 1.2 asks for many disjoint conflicts. A conflict block at $x _ { i }$ is a group of at most $\ell + 1$ entries of a list of patterns that agree outside $x _ { i }$ and have no common label at $x _ { i } . \mathrm { ~ A ~ } \gamma \ – p a c k e d$ witness on inputs $x _ { 1 } , \ldots , x _ { n }$ is a list of patterns, with repetitions that act as weights, such that at every $x _ { i }$ disjoint conflict blocks cover at least a $\gamma$ fraction of the list. A predictor that sees the other coordinates of the hidden pattern makes an error on at least one entry of each block, and hence on at least a $\gamma / ( \ell + 1 )$ fraction of the list.

A weak learner. The packed dimension yields a weak learner through a one-inclusion argument. Fix a pool of n inputs, a finite set V of patterns on it, and a distribution $\mu$ on V. For each position i, a predictor that sees the correct-label sets at all other inputs knows the hyperedge of the hidden pattern, and its best choice is the label contained in the largest µ-mass of patterns of that hyperedge; let $b _ { i }$ be the resulting error probability. Lemma 49 shows that $b _ { 1 } + \cdot \cdot \cdot + b _ { n } \leq p + n \gamma$ , where $p = d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathcal { H } )$

To see this, let $\mu$ be rational, so that it is a list of patterns with repetitions. In each hyperedge at position $i ,$ greedily remove minimal groups of entries whose sets at $x _ { i }$ have empty intersection. Each removed group is a conflict block, and the remaining entries share a label at $x _ { i } .$ , so the removed blocks cover at least the mass $b _ { i }$ that the best label misses (Lemma 48). If more than $p$ positions had $b _ { i } > \gamma _ { ; }$ their blocks would form a packed witness on more than $p$ distinct inputs, which is impossible. So at most $p$ positions contribute at most 1 each, the others contribute at most $\gamma$ each, and the sum is at most $p + n \gamma$

Thus every distribution on patterns has a response whose total error over the n positions is at most $p + n \gamma$ . By the minimax theorem, and by compactness when there are infinitely many patterns, a single randomized transductive rule achieves this bound on every pattern. A leave-one-out argument then gives a full-set learner with expected error at most $\gamma + p / ( m + 1 )$ from m examples, and since $\gamma < 1 / ( \ell + 1 )$ , this is below $1 / ( \ell + 1 )$ for $m = O _ { \ell , \gamma } ( p + 1 )$ (Lemma 50).

Boosting and compression. We boost the weak learner with multiplicative weights over the sample positions and turn the result into a compression scheme. The threshold $1 / ( \ell + 1 )$ plays the role of $1 / 2$ in binary boosting: if a distribution on Y puts less than $1 / ( \ell + 1 )$ of its mass outside a set of at most ℓ labels, then each of its most probable labels lies in that set (Lemma 11). In each round, some multiset of $M = O _ { \ell , \gamma } ( p + 1 )$ sample examples gives the weak learner error below $1 / ( \ell + 1 )$ under the current weights. After ${ \cal O } ( \log N )$ rounds, the average of the weak predictions puts less than $1 / ( \ell + 1 )$ of its mass on incorrect labels at every sample input, so its most probable label is correct on the whole sample (Lemma 51). Listing the multisets of all rounds gives a sample compression scheme of size $O ( ( p + 1 ) \log ( 2 N ) )$ , and the realizable bound follows from the usual generalization argument for compression schemes. This follows the approach of [FS97, MY16, DMY16, CEH<sup>+</sup>26] (Section 12).

For agnostic learning, the learner decodes every short list of sample examples and outputs a decoded predictor of smallest empirical error. For any $h \in \mathcal H$ , the examples on which h equals the observed set form a realizable subsample, so some decoded predictor has empirical error at most the empirical set disagreement of $h ,$ and the standard concentration argument for compression schemes finishes the proof (Lemma 53).

Prediction families from compression. To learn from one demonstrated label we need prediction families with the intersection property. For infinite Y, a single finite cover need not exist, even when one example identifies the target. For instance, suppose our class consists of all constant hypotheses $x \mapsto \{ y \} , y \in \mathcal { y }$ . On a pool of data, every label vector is correct for at most one of them. We therefore build the family of each candidate separately, from short descriptions of predictors.

Start from any full-set learner for $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ that uses M examples and has expected error below $1 / ( \ell + 1 )$ , and run the boosting procedure on a pool of $N \geq 2 M$ inputs, with an intersection $g \in \mathcal { T } _ { \ell + 1 } ( \mathcal { H } )$ as the target. After $T = O ( \log ( N / M ) )$ rounds at most M positions are still wrong, and we store them as corrections. The resulting description of $g$ consists of $O ( M T )$ pairs $( i , A )$ , each a position of the pool together with the correct-label set of $g$ there, and it decodes to a vector that is correct wherever $g$ is defined.

For a candidate $h ,$ let $F _ { h , S }$ consist of the decoded vectors of all descriptions whose set at each position i is a nonempty subset of $h ( x _ { i } )$ . There are at most $2 ^ { \ell } - 1$ such sets at each

position, so

$$
\log | F _ { h , S } | = O \big ( M \log ^ { 2 } ( e N / M ) \big ) ,
$$

with no dependence on $| { \mathcal { V } } |$ . If $g$ is the intersection of $\ell + 1$ candidates, every set in the description of $g$ is contained in the corresponding set of each of them. The same description therefore yields one vector in the family of every candidate, correct wherever $g$ is defined, which is the intersection property.

The conversion of Section 3.2 then gives a one-correct-label learner whose sample size is linear in M (Theorem 56), and the packed weak learner supplies $M = O _ { \ell , \gamma } ( p + 1 )$ with $p = d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) )$ . Since log $| F _ { h , S } |$ depends on N only through the ratio $N / M$ , the bound has no logarithmic factor in $p$ (Section 13).

Finally, Appendix B bounds $d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) )$ by $O ( D \log ( D + 2 ) )$ , where D is the joint DS or the disjoint-neighbor dimension of $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ (Theorem 60).

## 3.4 The Exponent ℓ Is Necessary

The minimax step of Section 3.2 loses an ℓth root when it passes from $\nu _ { i }$ to $\nu _ { i } ^ { \ell } .$ . To show that this loss cannot be avoided in general, we construct a class in which, at an unseen input, the best single label is wrong with probability of order $1 / n$ , whereas $\ell + 1$ plausible targets have no common label only with probability of order $n ^ { - \ell }$

Let $\ell \geq 2 , \chi = \mathbb { N } ^ { \ell }$ and $\boldsymbol { \ y } = [ \boldsymbol { \ell } ]$ . For $t \in \mathbb { N } ^ { \ell }$ and a finite set $F \subseteq \mathbb { N } ^ { \ell }$ let

$$
f _ { F } ( x ) = 1 + \mathbb { 1 } \{ x \in F \} , \qquad h _ { t , F } ( x ) = \{ f _ { F } ( x ) \} \cup \{ i \in [ \ell ] : x _ { i } \neq t _ { i } \} .
$$

Thus $h _ { t , F }$ always accepts the binary label $f _ { F } ( x ) \in \{ 1 , 2 \}$ , and it excludes a label $i \neq f _ { F } ( x )$ exactly on the slice $x _ { i } = t _ { i }$ . For the lower bound, put the inputs uniformly on the grid $[ n ] ^ { \ell }$ draw $F \subseteq [ n ] ^ { \ell }$ by including every grid point independently with probability $1 / 2 ,$ draw t uniformly from $[ n ] ^ { \ell }$ , and let the demonstrator always show $f _ { F } ( x )$ . The demonstrations reveal nothing about $t ,$ since the demonstrated label $f _ { F } ( x )$ does not depend on $t .$

At an unseen input, $f _ { F } ( x )$ is still equally likely to be 1 or $2 { \mathrm { . } }$ , so each of these labels is incorrect with probability $1 / ( 2 n )$ , and every other label with probability $1 / n$ . Therefore, every label is incorrect with probability at least $1 / ( 2 n )$

With at most $c n ^ { \ell }$ examples, for a small constant c, a constant fraction of the test inputs is unseen, and the expected error there is of order $1 / n$ . A second-moment argument shows that, for some target, the loss is $\Omega ( 1 / n )$ with probability greater than $1 / 8$ . Taking n of order $1 / \varepsilon$ gives the lower bound $\Omega ( \varepsilon ^ { - \ell } )$ . The dimension stays bounded: an intersection of $\ell + 1$ targets excludes each label on at most $\ell + 1$ slices, and $d _ { \mathrm { N } } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) ) = ( \ell + 1 ) ^ { \ell }$ (Lemma 42). Independent copies of the construction give the linear factor in the dimension (Section 11).

To see how this construction relates to the upper bound, fix an unseen input x and draw $\ell + 1$ targets independently from the conditional distribution given the sample. Their correct-label sets at x have empty intersection exactly when each of the ℓ labels is excluded by at least one of them. For each label i, one of the $\ell + 1$ targets must have $t _ { i } = x _ { i }$ , and the coordinates of the targets are independent, so the empty-intersection probability is at most $( \ell + 1 ) ^ { \ell } / n ^ { \ell }$ . Assigning a diferent target to exclude each label gives a lower bound of $\Omega ( n ^ { - \ell } )$ for fixed ℓ. The construction thus matches the relation between $\nu _ { i }$ and $\nu _ { i } ^ { \ell }$ in the upper bound.

## 3.5 Agnostic Learning from One Correct Label

A demonstration $( x , y )$ has two readings: the hidden set was a large set containing $y ,$ or it was exactly $\{ y \}$ . In the realizable setting this does not matter, since every demonstrated label is correct. In the agnostic setting the benchmark $L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } )$ compares whole sets, so two distributions of sets can induce the same law of demonstrations, hence the same law of the learner’s output, while calling for diferent predictions. This mechanism is behind every impossibility result below, and the conditions of Theorem F are what rule it out.

Where the conditions come from. The class of Theorem 67, with values $\{ 1 , 2 \} , \{ 1 , 3 \}$ and {2, 3} at a single input, shows the mechanism at work: three distributions of sets demonstrate a uniform label, yet each makes a diferent label the worst prediction, so every learner has excess error at least $1 / 9$ under one of them. The root of the problem is that no label lies in all three large sets: the learner must bet on which large set is hidden, and demonstrations cannot inform the bet. The dimensions of $\mathcal { T } _ { 3 } ( \mathcal { H } )$ do not see this; they coincide with those for the learnable class of the three singletons $x \mapsto \{ j \}$ (Remark 68).

The conditions of Theorem F make such bets unnecessary. Call an input mixed if some hypothesis has two or more labels there, and let $\mathcal { \partial } _ { x }$ be the set of labels a with $h ( x ) = \{ a \}$ for some $h \in { \mathcal { H } }$ . Local compatibility places $\mathcal { \partial } _ { x }$ inside every large value at x, so a prediction in $\mathcal { \partial } _ { x }$ is correct whenever the hidden set is a large value of the class, whichever it is. Local compatibility is necessary by the same mechanism: if a singleton $\{ a \}$ and a large set $B \not \ni { a }$ occur at one input, a mixture of $\{ a \}$ and B and a distribution of singletons demonstrate identically, yet the first calls for a label of B and the second for $a$ (Proposition 65). Nondegeneracy, two total hypotheses with everywhere diferent singleton values, makes $| \mathcal { V } _ { x } | \geq 2$ at every input, so safe labels always exist (for the class above, $\ y _ { x } = \varnothing )$ ; it also supplies the anchors of the lower bound below.

A learner predicting in ${ \mathcal { V } } _ { x }$ then needs to know about $h ( x )$ only whether it is undefined, large, or a singleton $\{ a \}$ , and which a. This is the collapsed value $\widetilde { h } ( \boldsymbol { x } )$ : the collapse forgets the content of large sets because local compatibility makes it irrelevant. At a mixed input, $\mathcal { \partial } _ { x }$ lies in a set of at most ℓ labels, so $\tilde { h } ( x ) \in \mathcal { V } _ { x } \cup \{ \circ \}$ takes at most $\ell + 1$ defined values: the mixed region is a multiclass problem over a bounded alphabet even when $\mathcal { V }$ is infinite. On the remaining pure inputs every defined value is a singleton, so a hypothesis’s error on demonstrated labels is at most its exact-set mismatch error. An agnostic multiclass learner therefore sufices. Both regions are determined by H alone, and both are governed by $d _ { \mathrm { D S } } ^ { \cap } ( \widetilde { \mathcal { H } } )$

Necessity. If $d _ { \mathrm { D S } } ^ { \cap } ( \widetilde { \mathcal { H } } ) = \infty$ , the DS dimension is infinite on one of the regions. On the pure region the realizable multiclass lower bound applies (Lemma 74). On the mixed region the bounded alphabet turns large DS witnesses into large Natarajan cubes, from which nondegeneracy and local compatibility extract an anchored cube (Lemma 73): inputs $x _ { 1 } , \ldots , x _ { n }$ , labels $a _ { i } \neq b _ { i }$ , an anchor with value $\{ b _ { i } \}$ at every $x _ { i } .$ , and for each $T \subseteq [ n ]$ a hypothesis $h _ { T }$ equal to $\{ a _ { i } \}$ on $T$ and containing $b _ { i }$ of $T$ . Put each i into $T$ with probability $1 / 4$ and demonstrate $a _ { i }$ on $T$ and $b _ { i }$ of $T$ . Read as the sets of $h _ { T }$ , the data are realizable, so the benchmark is zero and only $a _ { i }$ is correct on $T ;$ read as singletons, the anchor is wrong only on $T _ { i }$ so the benchmark is about $1 / 4$ , while $a _ { i }$ is wrong at three quarters of the inputs. $\mathrm { A t }$ an unseen $x _ { j }$ the learner cannot tell whether $j \in T$ , and one of the two readings gives it excess error at least $1 / 8$ (Lemma 72).

Suficiency. The learner routes each example to the learner of its region; as $L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } )$ is at least the mass-weighted sum of the regional benchmarks, it sufices to be competitive on each region of non-negligible mass (Lemma 81). On the pure region a hypothesis matches the hidden set only if it is the singleton of the demonstrated label, so an agnostic multiclass learner on the demonstrated pairs sufices (Theorems 60 and D with $\ell = 1 ;$ ; Section 14.5).

On the mixed region the learner predicts from ${ \mathcal { V } } _ { x }$ and competes with a comparator h whose set disagreement it cannot observe. The way out is to compare only where needed: where $h ( x )$ is large, local compatibility makes the learner correct whenever $h$ is; elsewhere the comparison is the ordinary multiclass regret on the demonstrated pairs, with undefined values of h counted as wrong. This observable regret bounds the excess error over $h$ (Lemma 76), and $h$ acts as a sleeping expert [FSSW97] that abstains on its large values. Running specialists over a finite cover of $\widetilde { \mathcal { H } }$ on the pool, whose size is independent of |Y| by the bounded alphabet, and a leave-one-out argument make the expected regret against every h vanish (Lemma 78). Validation cannot boost the confidence, since the regret involves the unknown h and may be negative. Instead, the learner averages independent runs, predicts a most frequently demonstrated label of $\mathcal { \partial } _ { x }$ at the inputs in a fresh sample, and independently rounds the averaged predictions elsewhere. Averaging concentrates the regret; with high probability the fresh sample contains every input of large mass, so the rounding error also concentrates (Lemma 79). Gluing the two regional learners proves Theorem F (Section 14.7).

## 3.6 Impossibility Results

Impossibility of scale-free dimension. We partition the input space into infinitely many blocks, where block $j$ uses the labels $\left[ s _ { j } \right]$ with $s _ { j } = j + 2$ . Each target in the class ${ \mathcal { H } } _ { j }$ excludes one label at every input of block $j$ and accepts all available labels on the other blocks; the excluded label is 1 at all but finitely many inputs. Each ${ \mathcal { H } } _ { j }$ is unlearnable, because these finitely many exceptions can lie anywhere in the block, so at an unseen input of block $j$ the excluded label can be uniform on $\left[ s _ { j } \right]$ . Uniform random guessing on block $j ,$ however, has expected error $1 / s _ { j }$ . Consequently, every union $\cup _ { j } \nu _ { j }$ of finite subclasses $\mathcal { V } _ { j } \subseteq \mathcal { H } _ { j }$ is learnable: at any prescribed accuracy and confidence, random guessing controls the error on the later blocks, and on the finitely many earlier blocks only finitely many exceptional inputs need to be handled. If a parameter D with finite witnesses characterized learnability, each unlearnable class ${ \mathcal { H } } _ { j }$ would have $\mathsf { D } ( \mathcal { H } _ { j } ) = \infty$ . The finite-witness property would then allow us to choose finite subclasses $\mathcal { V } _ { j } \subseteq \mathcal { H } _ { j }$ with $\mathsf { D } ( \mathcal { V } _ { j } ) \geq j$ , forcing the learnable union $\cup _ { j } \nu _ { j }$ to

have infinite parameter, a contradiction.

Impossibility of scale-sensitive dimension. Let $W = \{ 2 , 3 , \dots \}$ and consider the class $\kappa _ { \infty }$ of targets $h _ { F , B } ( x ) = \{ \mathbb { 1 } \{ x \in F \} \} \cup ( W \setminus B )$ , where $F \subseteq \mathbb { N }$ and $B \subseteq W$ are finite. Each target accepts one binary label at each input and every auxiliary label outside its forbidden set B. The class $\kappa _ { \infty }$ is unlearnable under one-correct-label feedback: demonstrations may reveal only binary labels, leaving unseen binary answers unpredictable, while a suitable finite forbidden set can invalidate almost all likely auxiliary predictions. The subclass G of targets with $2 \notin B$ is learnable by always predicting 2. For every finite subclass $\nu \subseteq \kappa _ { \infty }$ , the union of the forbidden sets of its targets is finite, so some auxiliary label is valid everywhere for all targets in $\nu _ { \mathrm { { ; } } }$ and swapping this label with 2 maps V into $\mathcal { G }$ . Hence, if each $\mathsf { D } _ { \alpha }$ has finite witnesses and is relabeling invariant, then $\mathsf { D } _ { \alpha } ( { \cal K } _ { \infty } ) = \mathsf { D } _ { \alpha } ( { \mathcal G } )$ at every scale $\alpha .$ . Such a family therefore assigns identical values to a learnable class and an unlearnable class, and cannot characterize learnability. This construction is inspired by [LBD24].

## 4 Conclusion

We studied PAC learning when each example may come with several correct labels and predicting any one of them sufices. With at most ℓ correct labels per example, the generalized Natarajan dimension characterizes learnability from full sets and from bandit feedback when Y is finite (Theorem $\mathrm { A } )$ , and the packed dimension characterizes learnability from full sets when Y is countable (Theorem D). In all these cases, realizable and agnostic learnability coincide, as in the multiclass case [DMY16, HKLM22, HMS25].

Learning from demonstrations, where each training example reveals only one of its correct labels, is governed by a diferent class. In the realizable setting, for finite and countable Y alike, H is learnable from demonstrations if and only if its intersection class $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ is learnable from full sets (Theorems B and E). This condition can fail already with three labels, for a class learnable from a single full-set example or even a single bandit answer (Example 15). When it holds, the sample complexity is at most of order $\varepsilon ^ { - \ell }$ up to logarithmic factors, and this exponent cannot be improved in general (Theorem C).

Agnostic learning from demonstrations behaves diferently. Already on a single input, two classes can have intersection classes with the same values of the dimensions that characterize realizable learning, while only one of them is agnostically learnable (Theorem F and Remark 68). For nondegenerate classes, agnostic learnability is instead characterized by finiteness of the DS dimension of the collapsed class $\widetilde { \mathcal { H } }$ , which merges every set of at least two correct labels into one new label, together with local compatibility (Theorem F).

Finally, the uniform bound ℓ on the number of correct labels is essential. Without it, it is impossible to characterize learnability by any parameter with finite witnesses (Theorem G).

We conclude with two open problems that arise naturally from our results and that we believe are worth exploring further.

⋆ The DS dimension of intersection classes. Over a countably infinite label space, must an intersection class $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ that is learnable from full sets have finite joint DS dimension? By Theorem E, this asks whether finiteness of $d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { T } _ { \ell + 1 } ( \mathcal { H } ) )$ , which sufices for learning from demonstrations, is also necessary. For finite label spaces it is (Theorem B). The analogous statement fails for general classes, since the class $\mathcal { H } ^ { + }$ of Appendix A is learnable from full sets but has infinite joint DS dimension (Theorem 91). Intersection classes, however, are more constrained. If $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ is learnable from full sets, then so is $\mathcal { T } _ { R } ( \mathcal { H } )$ for every fixed R (Corollary 58), so a counterexample must remain learnable from full sets after the intersections of any fixed number of hypotheses of H are added.

⋆ Scale-sensitive characterizations. Suppose that the correct-label sets are finite but not uniformly bounded. Is there a family $( \mathtt { D } _ { \varepsilon } ) _ { \varepsilon > 0 }$ of parameters with finite witnesses such that, under full-set feedback or under demonstrations, H is PAC learnable if and only if $\mathsf { D } _ { \varepsilon } ( \mathcal { H } ) < \infty$ for every $\varepsilon > 0 ?$ For real-valued classes, the fat-shattering dimension plays exactly this role. Theorem G rules out a single such parameter. Its last part also rules out families of relabeling-invariant parameters with finite witnesses, but only when the correct-label sets may be infinite and the feedback consists of demonstrations, so the case of finite sets remains open.

## AI Statement

The results for finite label spaces under full-set and bandit feedback, in both the realizable and the agnostic setting (Theorem A), were proved by the authors. The other results, as well as the writing of the paper, were developed with the help of Claude (Anthropic) and ChatGPT (OpenAI).

## 5 Preliminaries

We start with multi-label hypotheses and the three feedback models, continue with partial intersection classes and one-inclusion hypergraphs, and conclude with the four dimensions and a bound for plurality votes. We follow the notation of [AHHM21] for partial concepts and of $\mathrm { [ B C D ^ { + } 2 2 ] }$ for projections and one-inclusion graphs.

Write $\mathbb { N } = \{ 1 , 2 , \ldots \} , [ n ] = \{ 1 , \ldots , n \}$ , and $[ 0 ] = \varnothing$ ; all logarithms are natural, and sample sizes are rounded up to integers. The letter c denotes a positive constant that may increase from line to line. In a result whose statement declares some parameters fixed (for instance ℓ and γ), c depends only on these parameters, and never on the class, the alphabet size, the dimension, or the sample size; the notation $O _ { \ell } ( \cdot )$ makes this dependence explicit.

## 5.1 Multi-Label Hypotheses

In the main text, the instance space X and the label space Y are nonempty countable sets with their discrete σ-algebras.<sup>3</sup> Throughout the positive results we fix an integer $\ell \geq 1$ , the maximum number of correct labels, and put

$$
\mathcal { A } = \{ A \subseteq \mathcal { y } : 1 \leq | A | \leq \ell \} .\tag{1}
$$

When $\mathcal { V }$ is finite we write $k = | \mathcal { D } |$ and assume $\ell \leq k !$ ; the choice $\ell = k$ puts no bound on the number of correct labels. The number of possible correct-label sets is then

$$
q = | { \cal { A } } | = \sum _ { j = 1 } ^ { \ell } { \binom { k } { j } } .\tag{2}
$$

A multi-label hypothesis is a function $h : \mathcal { X } \to \mathcal { A } \cup \{ \star \}$ , and h is undefined at x if $h ( x ) = \star$ The support of h is supp $\langle h \rangle = \{ x \in \mathcal { X } : h ( x ) \neq \star \}$ , and $h$ is total if $\operatorname { s u p p } ( h ) = \mathcal { X }$ . For $x \in \operatorname { s u p p } ( h )$ , the set $h ( x )$ is the correct-label set of $x ,$ and its elements are the correct labels of x. A class is a set $\mathcal { H } \subseteq ( \mathcal { A } \cup \{ \star \} ) ^ { \mathcal { X } }$

A predictor is any function $g : \mathcal { X }  \mathcal { Y }$ (learners may be improper), and it is valid at x for h if $g ( x ) \in h ( x )$ . A randomized prediction p assigns to every $x \in \mathcal { X }$ a probability distribution $p ( \cdot \mid x )$ on $\mathcal { V } _ { : }$ , and we write $\begin{array} { r } { p ( A \mid x ) = \sum _ { y \in A } p ( y \mid x ) } \end{array}$

## 5.2 Learning Models

Realizable learning. Fix a target $h ^ { \star } \in { \mathcal { H } }$ and a distribution P on X with $P ( \operatorname { s u p p } ( h ^ { \star } ) ) = 1$ Since data are drawn only from the support of the target, undefined values let a class encode assumptions on the data, as for partial concept classes [AHHM21]. The learner outputs a single label, which counts as correct if it is one of the correct labels. Thus the errors of a predictor g and of a randomized prediction $p$ are

$$
L _ { P , h ^ { \star } } ( g ) = \operatorname* { P r } _ { x \sim P } \bigl [ g ( x ) \notin h ^ { \star } ( x ) \bigr ] , \qquad L _ { P , h ^ { \star } } ( p ) = \mathbb { E } _ { x \sim P } \bigl [ 1 - p ( h ^ { \star } ( x ) \mid x ) \bigr ] .\tag{3}
$$

The training inputs $x _ { 1 } , \ldots , x _ { m }$ are drawn independently from $P ,$ and the learner receives one of the following kinds of feedback.

(i) Full-set feedback. The learner observes the sample $\sigma = ( ( x _ { i } , h ^ { \star } ( x _ { i } ) ) ) _ { i = 1 } ^ { m } \sim \mathcal { D } _ { P , h ^ { \star } } ^ { m }$ , where $\mathcal { D } _ { P , h }$ denotes the law of $( x , h ( x ) )$ for $x \sim P$

(ii) Bandit feedback. After seeing $x _ { i }$ , the learner chooses a label $u _ { i } \in \mathcal { V }$ , which may depend on all previous inputs and answers, and observes only $b _ { i } = 1 \{ u _ { i } \in h ^ { \star } ( x _ { i } ) \}$ . Our bandit learners use a special case of this protocol: when $\mathcal { V }$ is finite, they draw the queries uniformly at random, independently of the inputs and of each other, as in the batch model of [HMMS26b].

(iii) One-correct-label feedback. The sample $\sigma = \left( ( x _ { 1 } , y _ { 1 } ) , \dots , ( x _ { m } , y _ { m } ) \right)$ consists of independent draws from a joint distribution D on $\mathcal { X } \times \mathcal { V }$ whose marginal on X is P and which satisfies $\operatorname* { P r } _ { ( x , y ) \sim \mathcal { D } } [ y \in h ^ { \star } ( x ) ] = 1 ;$ ; we say that such a distribution is compatible with $h ^ { \star }$ and we call $y _ { i } \mathrm { ~ a ~ }$ demonstrated label. The conditional law of $y _ { i }$ is otherwise arbitrary; for instance, the demonstrator may always show the same correct label.

When $\ell = 1$ , full-set and one-correct-label feedback coincide, and the model is multiclass PAC learning of partial concept classes $\mathrm { [ A H H M 2 1 , B C D ^ { + } 2 2 ] }$

Learners may be randomized: a learner may use an independent seed $\omega \sim \mathrm { U n i f } ( [ 0 , 1 ] )$ and its output is then a random predictor, drawn once. A learner may also depend on ε and $\delta ,$ , and it may ignore surplus observations.<sup>4</sup>

Definition 1 (PAC learnability). Fix one of the three feedback models. A class H is realizably $P A C$ learnable if for every $\varepsilon , \delta \in ( 0 , 1 )$ there are an integer m and a learner such that the following holds. Consider any target $h ^ { \star } \in { \mathcal { H } }$ and any distribution P on $\mathcal { X }$ with $P ( \operatorname { s u p p } ( h ^ { \star } ) ) = 1$ (and, for one-correct-label feedback, any joint distribution D as above). With probability at least $1 - \delta$ over m training examples and the seed, the learner outputs a predictor $\widehat g$ with $L _ { P , h ^ { \star } } ( { \widehat { g } } ) \leq \varepsilon$ . The least such m is the sample complexity, denoted $m _ { \mathcal { H } } ^ { \mathrm { f u l l } } ( \varepsilon , \delta )$ $m _ { \mathcal { H } } ^ { \mathrm { b a n d } } ( \varepsilon , \delta )$ , or $m _ { \mathcal { H } } ^ { \mathrm { o n e } } ( \varepsilon , \delta )$ ; it equals $\infty$ if no finite m works.

We pass between PAC guarantees and bounds on the expected error by two implications, valid for every random error $Z \in [ 0 , 1 ]$

$$
\operatorname* { P r } [ Z > \varepsilon ] \le \delta \implies \mathbb { E } [ Z ] \le \varepsilon + \delta , \qquad \mathbb { E } [ Z ] \le \varepsilon \delta \implies \operatorname* { P r } [ Z > \varepsilon ] \le \delta .\tag{4}
$$

The first holds because $Z \leq \varepsilon + 1 \{ Z > \varepsilon \}$ , and the second is Markov’s inequality.

Agnostic learning. In the agnostic setting, examples $( x , A )$ are drawn from an arbitrary distribution D on $\mathcal { X } \times \mathcal { A }$ . Under full-set feedback the learner observes the pairs $( x _ { i } , A _ { i } )$ and under bandit feedback it observes $x _ { i }$ and the answer 1 $\{ u _ { i } \in A _ { i } \}$ to its query. Under one-correct-label feedback it observes only pairs $( x _ { i } , y _ { i } )$ , where the conditional law of the demonstrated label y given $( x , A )$ is supported on A and otherwise arbitrary. The error of a predictor and the benchmark are

$$
L _ { \mathcal { D } } ( g ) = \operatorname* { P r } _ { ( x , A ) \sim \mathcal { D } } [ g ( x ) \notin A ] , \qquad L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } ) = \operatorname* { i n f } _ { h \in \mathcal { H } ( x , A ) \sim \mathcal { D } } [ h ( x ) \neq A ] .\tag{5}
$$

The benchmark compares the whole correct-label set of h with the sampled set, and, as in [AHHM21], it counts $h ( x ) = \star \mathrm { a s }$ a mismatch. It can therefore be much larger than the best achievable error. For example, with one input x, $\mathcal { H } = \{ h \} , h ( x ) = \{ 1 , 2 \}$ , and $A = \{ 1 \}$ almost surely, we have $L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } ) = 1$ , although predicting 1 has zero error.

Definition 2 (Agnostic PAC learnability). Fix one of the three feedback models. A class H is agnostically PAC learnable if for every $\varepsilon , \delta \in ( 0 , 1 )$ a finite number of examples sufices for some learner to ensure $\operatorname* { P r } [ L _ { \mathcal { D } } ( \widehat { g } ) > L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } ) + \varepsilon ] \leq \delta$ for every $\mathcal { D }$ (and, for one-correct-label feedback, every conditional law of the demonstrated label). Under full-set and bandit feedback, the least such number is denoted $m _ { \mathcal { H } } ^ { \mathrm { f u l l , a g } } ( \varepsilon , \delta )$ and $m _ { \mathcal { H } } ^ { \mathrm { b a n d , a g } } ( \varepsilon , \delta )$ , respectively.

Under one-correct-label feedback we need no notation for the sample complexity, since Section 14 characterizes learnability itself. A realizable experiment is the agnostic experiment with $A = h ^ { \star } ( x )$ , whose benchmark is zero. Hence agnostic PAC learnability implies realizable PAC learnability under each feedback model.

## 5.3 Projections, Intersections, and One-Inclusion Hypergraphs

We use sequences of inputs rather than sets, because the inputs of a learning problem form a sequence $\mathrm { [ B C D ^ { + } 2 2 ] }$ . For a sequence $S = ( x _ { 1 } , \ldots , x _ { n } ) \in { \mathcal { X } } ^ { n }$ , the projection of H to $S$ records the correct-label sets on $S$ of the hypotheses defined on all of S:

$$
\mathcal H | _ { S } = \left\{ ( h ( x _ { 1 } ) , \ldots , h ( x _ { n } ) ) : h \in \mathcal H , \ \{ x _ { 1 } , \ldots , x _ { n } \} \subseteq \operatorname { s u p p } ( h ) \right\} \subseteq { \mathcal A } ^ { n } .\tag{6}
$$

We call the elements of $\mathcal { H } | _ { S }$ patterns. For a pattern $v = ( v _ { 1 } , \ldots , v _ { n } )$ and $i \in [ n ] , v _ { - i }$ denotes v with its ith coordinate deleted, and for $V \subseteq A ^ { n }$ and $I \subseteq [ n ] , V | _ { I }$ is the coordinate restriction of $V \mathrm { ~ t o ~ } I$ . We write $\sigma$ for labeled samples and S for sequences of unlabeled inputs, which we call pools; a pool may contain repeated inputs.

Partial intersection classes. If every demonstrated label of a sample lies in $h _ { 1 } ( x ) \cap h _ { 2 } ( x )$ then the sample is consistent with both $h _ { 1 }$ and $h _ { 2 } .$ and the learner cannot tell them apart. More generally, a label in $h _ { 1 } ( x ) \cap \cdots \cap h _ { r } ( x )$ is a valid demonstration for each $h _ { a }$ , so a learner from one correct label must also succeed on samples generated by such intersections (Section 9). For an integer $R \geq 1$ , the partial intersection class $\mathcal { T } _ { R } ( \mathcal { H } )$ consists of all hypotheses g of the form

$$
g ( x ) = \left\{ \bigcap _ { a = 1 } ^ { r } h _ { a } ( x ) , \quad x \in B , \right.\tag{7}
$$

where $1 \leq r \leq R , h _ { 1 } , \ldots , h _ { r } \in \mathcal { H }$ , and $B \subseteq \textstyle \bigcap _ { a = 1 } ^ { r } \operatorname { s u p p } ( h _ { a } )$ is any set on which the intersection $\cap _ { a } h _ { a } ( x )$ is nonempty. We call $\left( h _ { 1 } , \ldots , h _ { r } \right)$ a representing tuple of $g ;$ repeating one of its members pads it to length exactly R. The class $\mathcal { T } _ { R } ( \mathcal { H } )$ is closed under domain restrictions, that is, it contains the restriction of each of its members to every subset of its support, but in general it is not closed under further intersections. All one-correct-label results concern $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ , the intersections of at most ℓ+1 hypotheses of H. Its correct-label sets are nonempty subsets of correct-label sets of $\mathcal { H } .$ , so they still have at most ℓ labels.

One-inclusion hypergraphs. For a finite nonempty $V \subseteq A ^ { n }$ , the one-inclusion hypergraph of V has vertex set V and hyperedges

$$
e _ { i , u } = \{ v \in V : v _ { - i } = u \} , \qquad i \in [ n ] , \ u \in V | _ { [ n ] \backslash \{ i \} } .\tag{8}
$$

This is the one-inclusion graph of [HLW94], in the hypergraph form of [RBR06] and $\mathrm { [ B C D ^ { + } 2 2 ] }$ with patterns of correct-label sets in place of labels. The vertices of $e _ { i , u }$ are the patterns that agree, as whole correct-label sets, at every coordinate other than $i ,$ and i is the direction of $e _ { i , u }$ . Hyperedges are indexed by the pair $( i , u )$ , so equal vertex sets in diferent directions count as diferent hyperedges.

## 5.4 Dimensions

We now define the four dimensions of this paper. Each of them is the largest n such that some sequence of n distinct inputs carries a certain finite witness; it is 0 if no witness of positive size exists, and $\infty$ if witnesses of every size exist. In all four cases the witness consists of patterns, and it contains groups of patterns that agree at all inputs but one and have no common label at the remaining input, so that no single prediction there is correct for a whole group. The definitions difer in how many such groups they require and how the groups are arranged.

The first dimension extends the Natarajan dimension and asks for a full product of conflicting families.

Definition 3 (Generalized Natarajan dimension). A sequence $S = ( x _ { 1 } , \ldots , x _ { n } ) $ of distinct inputs is product-shattered by H if there are finite nonempty families $\mathcal { A } _ { 1 } , \ldots , \mathcal { A } _ { n } \subseteq \mathcal { A }$ such that

$$
\bigcap _ { A \in A _ { i } } A = \varnothing \quad ( i \in [ n ] ) \qquad \mathrm { a n d } \qquad \mathcal { A } _ { 1 } \times \dots \times \mathcal { A } _ { n } \subseteq \mathcal { H } | _ { S } .
$$

The product $\mathcal { A } _ { 1 } \times \cdots \times \mathcal { A } _ { n }$ is a product witness. The generalized Natarajan dimension $d _ { \mathrm { N } } ^ { \cap } ( \mathcal { H } )$ is the largest size of a product-shattered sequence.

In a product witness, every input carries a family of correct-label sets with no common label, and every combination of choices is realized. The next dimension asks for less, in the spirit of the pseudo-cubes of $\mathrm { [ B C D ^ { + } 2 2 ] }$ : whenever the correct-label sets at all other inputs are revealed, the remaining candidates at the hidden input still have no common label.

Definition 4 (Joint DS dimension). Let $S = ( x _ { 1 } , \ldots , x _ { n } ) \in { \mathcal { X } } ^ { n }$ . A finite nonempty set $V \subseteq \mathcal { H } | _ { S }$ is a DS witness if every hyperedge of its one-inclusion hypergraph has an empty intersection in its own direction:

$$
\bigcap _ { w \in e _ { i , v _ { - i } } } w _ { i } = \varnothing \qquad ( v \in V , \ i \in [ n ] ) .
$$

The joint DS dimension $d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { H } )$ is the largest n for which some sequence of n distinct inputs carries a DS witness.

A stronger local condition asks that a single other candidate in the hyperedge already rules out every label of the current one. The resulting dimension is at most the joint DS dimension (Lemma 9), so upper bounds in terms of it, as in Theorem E, are stronger.

Definition 5 (Disjoint-neighbor dimension). Let $S = ( x _ { 1 } , \ldots , x _ { n } ) \in { \mathcal { X } } ^ { n }$ . A finite nonempty set $V \subseteq \mathcal { H } | _ { S }$ is a disjoint-neighbor witness if for every $v \in V$ and $i \in [ n ]$ there is $w \in V$ with

$$
w _ { - i } = v _ { - i } \qquad \mathrm { a n d } \qquad w _ { i } \cap v _ { i } = \emptyset .
$$

The disjoint-neighbor dimension $d _ { \mathrm { p a i r } } ( \mathcal { H } )$ is the largest n for which some sequence of n distinct inputs carries a disjoint-neighbor witness.

When $\ell = 1$ , all correct-label sets are singletons: then $d _ { \mathrm { N } } ^ { \bigcap }$ is the Natarajan dimension [Nat89], and $d _ { \mathrm { D S } } ^ { \cap }$ and $d _ { \mathrm { p a i r } }$ both equal the DS dimension [DSS14]. Already in this case $d _ { \mathrm { N } } ^ { \bigcap }$ can be smaller than the other two.

Example 6. Let $\mathcal { X } = \{ x _ { 1 } , x _ { 2 } \}$ , and let H consist of the six total hypotheses whose patterns on $( x _ { 1 } , x _ { 2 } )$ are $( 1 , 2 ) , ( 3 , 2 ) , ( 3 , 4 ) , ( 5 , 4 ) , ( 5 , 6 ) , ( 1 , 6 )$ , where each label stands for a singleton. The one-inclusion hypergraph of $\mathcal { H } | _ { ( x _ { 1 } , x _ { 2 } ) }$ is a hexagon, and every pattern has a neighbor with a diferent label in each direction, so ${ d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { H } ) } = d _ { \mathrm { p a i r } } ( \mathcal { H } ) = 2$ . On the other hand, the first labels 1, 3 and 5 occur with the second labels $\{ 2 , 6 \} , \{ 2 , 4 \}$ and $\{ 4 , 6 \}$ , respectively. No two of these sets share two labels, so $\mathcal { H } | _ { ( x _ { 1 } , x _ { 2 } ) }$ contains no product of two pairs, and $d _ { \mathrm { N } } ^ { \cap } ( \mathcal { H } ) = 1$

For larger $\ell ,$ the joint DS dimension difers from the DS dimension of the class obtained by viewing each correct-label set as a single label: our hyperedges must have empty intersections, while the ordinary DS dimension only requires at least two distinct sets in each hyperedge.

With sets of size two or more, an empty intersection in a hyperedge can force very few errors: if all members of a large hyperedge but one contain the label 1, then predicting 1 is wrong for only one of them. The last dimension, designed for countable label spaces, therefore asks for many disjoint small conflicts, which must cover a constant fraction of a weighted list of patterns in every direction. The fraction is a coverage parameter $\gamma _ { ; }$ which we fix once and for all in the range

$$
0 < \gamma < \frac { 1 } { \ell + 1 } .\tag{9}
$$

Definition 7 (Packed dimension). Let $S = ( x _ { 1 } , \ldots , x _ { n } ) \in { \mathcal { X } } ^ { n }$ . A γ-packed witness on S consists of a nonempty indexed list $v ^ { 1 } , \ldots , v ^ { M } \in \mathcal { H } | _ { S }$ , with repetitions allowed, and, for every direction $i \in [ n ]$ , a collection $B _ { i }$ of pairwise disjoint subsets of [M], called conflict blocks, such that every $B \in B _ { i }$ satisfies

$$
2 \leq | { \cal B } | \leq \ell + 1 , \qquad v _ { j } ^ { a } = v _ { j } ^ { b } ( a , b \in { \cal B } , \ j \neq i ) , \qquad \bigcap _ { a \in { \cal B } } v _ { i } ^ { a } = \emptyset ,
$$

and the blocks of each direction cover at least $\textrm { a } \gamma$ fraction of the list:

$$
{ \Big | } \bigcup _ { B \in B _ { i } } B { \Big | } \geq \gamma M \qquad ( i \in [ n ] ) .
$$

The packed dimension $d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathcal { H } )$ is the largest n for which some sequence of n distinct inputs carries a γ-packed witness. For another coverage parameter $0 < \rho \le 1$ , the dimension $d _ { \mathrm { p a c k } , \rho } ^ { \ n } ( \mathcal { H } )$ is defined in the same way, with $\gamma$ replaced by $\rho .$

Blocks in diferent directions may overlap. The repetitions in the list encode rational weights, and they are kept when a witness is projected to fewer coordinates. The upper bound on $\gamma$ serves the weak learner of Section 12: when $d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathcal { H } ) < \infty$ , its expected error from m examples is at most $\gamma + d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathcal { H } ) / ( m + 1 )$ (Lemma 50), which is below the threshold $1 / ( \ell + 1 )$ of Lemma 11 for suficiently large m. Other coverage levels $\rho$ appear mainly in comparisons with DS-type dimensions (Theorem 60).

We now compare the four dimensions. The comparison, and many later arguments, rest on the following simple fact. A finite family of sets is a minimal empty-intersection family if its intersection is empty and every proper subfamily has a nonempty intersection.

Lemma 8 (Small empty-intersection families). If a family of nonempty sets of size at most ℓ has an empty intersection, then some subfamily of at most $\ell + 1$ members has an empty intersection. Consequently, every minimal empty-intersection family of such sets has between two and $\ell + 1$ members.

Proof. Choose one member A of the family, and for each $y \in A$ choose a member that does not contain $y .$ . These at most $1 + | A | \leq \ell + 1$ sets have an empty intersection. A minimal family has at least two members, because each set is nonempty. □

Lemma 9 (Comparisons and projections). For every class $\mathcal { H }$

$$
\begin{array} { r } { d _ { \mathrm { N } } ^ { \cap } ( \mathcal { H } ) \leq d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { H } ) , \qquad d _ { \mathrm { p a i r } } ( \mathcal { H } ) \leq d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { H } ) , \qquad d _ { \mathrm { N } } ^ { \cap } ( \mathcal { H } ) \leq d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathcal { H } ) . } \end{array}\tag{10}
$$

The last inequality holds for $d _ { \mathrm { p a c k } , \rho } ^ { \cap }$ with every $0 < \rho \leq 1$ . Moreover, if V is a DS witness on $S$ and I is a nonempty set of coordinates, then $V { \big \vert } _ { I }$ is a DS witness on the corresponding subsequence; and a packed witness restricted to I, with all indices of the list kept, is a packed witness on that subsequence.

Proof. Take a product witness $\mathcal { A } _ { 1 } \ \times \ \cdots \times \mathcal { A } _ { n }$ and shrink each $\mathbf { \mathcal { A } } _ { i }$ to a minimal emptyintersection family, which by Lemma 8 has between two and $\ell + 1$ members. The product $V = \mathcal { A } _ { 1 } \times \cdots \times \mathcal { A } _ { n }$ is a DS witness, because the hyperedge of $v \in V$ in direction i is $\{ w \in V : w _ { - i } = v _ { - i } \}$ and its ith coordinates range over all of $A _ { i } ;$ this proves the first inequality. A disjoint-neighbor witness is a DS witness, because the hyperedge $e _ { i , v _ { - i } }$ contains v and a pattern whose ith coordinate is disjoint from $v _ { i } ;$ this proves the second inequality. Finally, if we list every element of the product V once, the hyperedges in direction i are disjoint blocks of size at most $\ell + 1$ with empty intersections that cover the whole list, so $V$ is a packed witness for every coverage parameter.

For the projection of a DS witness, let $u \in V | _ { I }$ and $i \in I$ , and choose $v \in V$ with $v | _ { I } = u$ The hyperedge of u in direction i contains the projection of the hyperedge of v in direction $i ,$ which has the same sets at coordinate i. An intersection over more sets can only be smaller, so it remains empty. For packed witnesses, the equalities away from each retained direction, the empty intersections, and the covered fractions are all preserved when the list is restricted to I. □

Disjoint products. For lower-bound constructions we combine classes on disjoint instance spaces (Section 11). Let X be the disjoint union of $\mathcal { X } _ { 1 } , \ldots , \mathcal { X } _ { b }$ , and for each $j \in [ b ]$ let $\mathcal { H } _ { j } \subseteq ( \boldsymbol { \mathcal { A } } \cup \{ \star \} ) ^ { \mathcal { X } _ { j } }$ . The disjoint product of $\mathcal { H } _ { 1 } , \ldots , \mathcal { H } _ { b }$ is the class

$$
\{ h \in ( { \mathcal { A } } \cup \{ { \star } \} ) ^ { \mathcal { X } } : h | _ { \mathcal { X } _ { j } } \in \mathcal { H } _ { j } { \mathrm { ~ f o r ~ a l l ~ } } j \in [ b ] \} ;
$$

thus a member is obtained by choosing a member of each ${ \mathcal { H } } _ { j }$ independently, undefined values included. The four dimensions add up under this operation.

Lemma 10 (Disjoint products). Let H be the disjoint product of nonempty classes $\mathcal { H } _ { 1 } , \dots , \mathcal { H } _ { b }$ $b \geq 1$ , over a common label space, with the same bound ℓ. For each $\mathsf { D } \in \{ d _ { \mathrm { N } } ^ { \cap } , d _ { \mathrm { D S } } ^ { \cap } , d _ { \mathrm { p a i r } } , d _ { \mathrm { p a c k } , \gamma } ^ { \cap } \}$

$$
\mathsf { D } ( \mathcal { H } ) = \sum _ { j = 1 } ^ { b } \mathsf { D } ( \mathcal { H } _ { j } ) .
$$

If moreover $\mathcal { H } _ { 1 } , \ldots , \mathcal { H } _ { b }$ are total, then $\begin{array} { r } { \mathsf { D } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) ) = \sum _ { j = 1 } ^ { b } \mathsf { D } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } _ { j } ) ) } \end{array}$

Proof. For the upper bound, take a witness on S and split its coordinates according to the component that contains each input. Every pattern of H on S restricts to a pattern of ${ \mathcal { H } } _ { j }$ on the part of $S$ in component $j ,$ so it sufices to check that each type of witness projects to a witness of the same type. A product witness projects to a product witness, and DS and packed witnesses (the latter with all indices kept) project correctly by Lemma 9. For a disjoint-neighbor witness $V$ and coordinates I, if w is a disjoint neighbor of v in a direction $i \in I$ , then $w | _ { I }$ is a disjoint neighbor of $v | _ { I }$ in direction i. Thus the part of a witness in component j has at most $\mathsf { D } ( \mathcal { H } _ { j } )$ coordinates, and summing over j gives the upper bound.

For the lower bound, take finite witnesses in the components (a single empty-coordinate pattern for a zero-dimensional component) and form their Cartesian product, whose elements are patterns of H because the components of a member of H are chosen independently. Fix a coordinate i in component j. A hyperedge of the product in direction i fixes the other components, so its sets at i are those of a hyperedge of the witness of component j. Hence products, empty intersections and disjoint neighbors at i are supplied by that witness. For packed witnesses, index the product list by tuples of indices, and in direction i copy every block of component $j$ for each choice of the indices of the other components; these blocks are disjoint and cover the same fraction of the list. The product witness has the sum of the component sizes, and taking arbitrarily large witnesses handles infinite values.

For the intersection classes, it sufices to show that $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ is the disjoint product of the classes $\mathbb { Z } _ { \ell + 1 } ( \mathcal { H } _ { j } )$ . Pad all representing tuples to length $\ell + 1$ . A tuple of members of H restricts to a tuple of members of ${ \mathcal { H } } _ { j }$ in each component, and conversely, tuples chosen in the components combine position by position into a tuple of members of $\mathcal { H } .$ . Since the components are total, the only constraint on the domain B in (7) is that the intersection be nonempty on $B { \mathrm { , } }$ and this holds exactly when it holds on every $B \cap \mathcal { X } _ { j }$ . So the domains also factor across the components. □

## 5.5 Plurality Votes

A binary majority vote is wrong only if at least half of the votes are wrong; with up to ℓ correct labels, the fraction one half becomes $1 / ( \ell + 1 )$ . A plurality label is a label that receives the largest number of votes,<sup>5</sup> and a most probable label of a distribution on Y is a label of maximum probability (one exists, since only finitely many labels have mass larger than half of the supremum of the masses); ties are broken by a fixed enumeration of Y.

Lemma 11 (Most probable labels). Let A be a set of at most ℓ labels. If a most probable label of a distribution on Y lies outside A, then the distribution assigns mass at least $1 / ( \ell + 1 )$ to $\mathcal { V } \backslash A$ . Consequently, the error of a most probable label is at most ℓ + 1 times the error of the randomized prediction. Likewise, a plurality vote is invalid at an input only if at least a $1 / ( \ell + 1 )$ fraction of the votes there are invalid, so its error is at most ℓ + 1 times the average error of the voters.

Proof. Let a be the mass of $\mathcal { V } \setminus A$ . The most probable label lies outside A, so its mass is at most a, and each of the at most ℓ labels in A has mass at most that of the most probable label. Thus $1 - a \le \ell a$ , that is, $a \ge 1 / ( \ell + 1 )$ . Applying this at every input with $A = h ^ { \star } ( x )$ and taking expectations over $x \sim P$ gives the consequence for a most probable label; for plurality votes, apply the same argument to the empirical distribution of the votes. □

## 6 Examples

We now compute the dimensions of Section 5.4 for four classes. In all of them every correctlabel set has at most two labels, so $\ell = 2$ and the parameter $\gamma$ of $d _ { \mathrm { p a c k } , \gamma } ^ { \cap }$ ranges over $( 0 , 1 / 3 )$ We start with two geometric classes on the real line, in which two labels are correct exactly at boundary points.<sup>6</sup> For both, $d _ { \mathrm { N } } ^ { \cap } , d _ { \mathrm { D S } } ^ { \cap }$ and $d _ { \mathrm { p a i r } }$ equal the corresponding parameter of the classical single-label class, so the extra correct labels at the boundaries cost nothing.

Example 12 (Unions of intervals). Let $\mathcal { X } = \mathbb { R } , \mathcal { Y } = \{ 0 , 1 \}$ , and $r \geq 1$ . For a union $U$ of at most r disjoint closed intervals, the empty union included, let $h _ { U } ( x ) = \{ 1 \}$ if x is an interior point of $U , h _ { U } ( x ) = \{ 0 , 1 \}$ if x is an endpoint of one of its intervals, and $h _ { U } ( x ) = \{ 0 \}$ otherwise. The class ${ \mathcal { H } } _ { \mathrm { i n t } }$ of all these hypotheses satisfies

$$
d _ { \mathrm { N } } ^ { \cap } ( \mathcal { H } _ { \mathrm { i n t } } ) = d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { H } _ { \mathrm { i n t } } ) = d _ { \mathrm { p a i r } } ( \mathcal { H } _ { \mathrm { i n t } } ) = 2 r .
$$

For the lower bound, take points $x _ { 1 } < \cdots < x _ { 2 r }$ . A labeling of these points by 0 and 1 has at most r runs of ones, and choosing for each run a closed interval that contains the run in its interior and no other $x _ { i }$ (these intervals are disjoint, since distinct runs are separated by a point labeled 0) realizes the corresponding pattern of singletons. Hence $x _ { 1 } , \ldots , x _ { 2 r }$ are product-shattered with the families $\{ \{ 0 \} , \{ 1 \} \}$ , and the realized patterns form a disjoint-neighbor witness.

For the upper bound, select the label 0 at every endpoint, turning each $h _ { U }$ into the indicator of the interior of U. In a DS witness of ${ \mathcal { H } } _ { \mathrm { i n t } }$ , the sets along each hyperedge have no common label, so they include both {0} and {1}. Patterns that agree outside a coordinate still agree there after the selection, so the selection maps the witness to a DS witness of the binary class of these indicators. For binary classes the DS dimension is the VC dimension, and the VC dimension of interiors of unions of at most r intervals is at most $2 r$ , because the alternating labeling $1 , 0 , 1 , \ldots , 0 , 1$ of $2 r + 1$ points needs $r + 1$ intervals. Hence $d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { H } _ { \mathrm { i n t } } ) \leq 2 r$ and by Lemma 9 the same bound holds for $d _ { \mathrm { N } } ^ { \cap }$ and $d _ { \mathrm { p a i r } }$

Example 13 (Nearest-site classification). Let $\chi = \mathbb { R }$ and $\ y = [ k ]$ with $k \geq 2$ . For sites $s _ { 1 } < \cdots < s _ { k }$ , let $h _ { s } ( x )$ be the set of indices $j$ that minimize $| x - s _ { j } |$ . Thus every input has a single nearest site, except the midpoint of two adjacent sites, where both of their indices are correct. The class ${ \mathcal { H } } _ { \mathrm { s i t e } }$ of all these hypotheses satisfies

$$
d _ { \mathrm { N } } ^ { \cap } ( \mathcal { H } _ { \mathrm { s i t e } } ) = d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { H } _ { \mathrm { s i t e } } ) = d _ { \mathrm { p a i r } } ( \mathcal { H } _ { \mathrm { s i t e } } ) = k - 1 .
$$

For the lower bound, fix sites $s _ { 1 } < \cdots < s _ { k }$ and let $x _ { j } = ( s _ { j } + s _ { j + 1 } ) / 2$ for $j \in [ k - 1 ]$ . Sites are recovered from $s _ { 1 }$ and their midpoints $m _ { j }$ by $s _ { j + 1 } = 2 m _ { j } - s _ { j }$ . Given signs, keep $s _ { 1 }$ , move the jth midpoint to $m _ { j } = x _ { j } \pm \delta$ for every $j ,$ , and recompute the sites. Each recomputed site moves by at most 2δ more than the previous one, hence by at most 2kδ, so for small $\delta > 0$ the sites stay in order and $x _ { j }$ remains closest to the jth or the $( j + 1 )$ st site. If the jth midpoint moved right, then $x _ { j }$ lies to its left and the correct-label set at $x _ { j }$ is $\{ j \}$ ; if it moved left, this set is $\{ j + 1 \}$ . Hence $x _ { 1 } , \ldots , x _ { k - 1 }$ are product-shattered with the families $\{ \{ j \} , \{ j + 1 \} \}$ , and the realized patterns form a disjoint-neighbor witness.

For the upper bound, take a DS witness of $\mathcal { H } _ { \mathrm { s i t e } } ;$ reordering its inputs, we may assume they are $x _ { 1 } < \cdots < x _ { n }$ . Select the smallest label of each correct-label set. The sets of a hyperedge cannot all have the same smallest label, since that label would be common to all of them, so the selected sequences form a DS witness of singletons. Moreover, each selected sequence is nondecreasing, because the smallest index of a nearest site does not decrease as the input moves to the right.

Choose a selected sequence $f = ( f _ { 1 } , \ldots , f _ { n } )$ with minimum sum of entries. In each direction $i , f$ has a neighbor $f ^ { \prime }$ that difers from it only at coordinate i, and $f _ { i } ^ { \prime } > f _ { i }$ by the choice of $f .$ Since $f ^ { \prime }$ is also nondecreasing and agrees with f of coordinate $i , f _ { i } < f _ { i } ^ { \prime } \leq f _ { i + 1 } ^ { \prime } = f _ { i + 1 }$ for $i < n$ , and $f _ { n } < f _ { n } ^ { \prime } \leq k$ . Hence $1 \leq f _ { 1 } < \cdot \cdot \cdot < f _ { n } < k$ , so $n \leq k - 1$ , and by Lemma 9 the same bound holds for $d _ { \mathrm { N } } ^ { \cap }$ and $d _ { \mathrm { p a i r } }$

The next two examples separate the roles of the diferent dimensions. The first gives classes with disjoint-neighbor dimension zero while the other three dimensions can be arbitrarily large or infinite. In particular, this shows that finite disjoint-neighbor dimension does not imply learnability from full sets. This contrasts with the fact that finite disjoint-neighbor dimension of the intersection class implies learnability from one correct label. The second gives a three-label class whose four dimensions are at most one. It is learnable from a single full-set example, and even from a single bandit answer, but not from any number of demonstrations. Thus it separates full-set and bandit learnability from one-correct-label learnability.

Example 14 (Triangle products). Let $\mathscr { y } = [ 3 ]$ and $\mathcal { T } = \{ \{ 1 , 2 \} , \{ 1 , 3 \} , \{ 2 , 3 \} \}$ , and let $\mathcal { H } _ { \triangle }$ be the class of all total hypotheses on $\mathcal { X }$ whose correct-label sets lie in T. If $\chi = [ d ]$ , then

$$
\begin{array} { r } { d _ { \mathrm { N } } ^ { \cap } ( \mathcal { H } _ { \triangle } ) = d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { H } _ { \triangle } ) = d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathcal { H } _ { \triangle } ) = d _ { \mathrm { p a i r } } ( \mathcal { Z } _ { 2 } ( \mathcal { H } _ { \triangle } ) ) = d , \qquad d _ { \mathrm { p a i r } } ( \mathcal { H } _ { \triangle } ) = 0 . } \end{array}
$$

Indeed, the three sets of $\tau$ have no common label, so $\mathcal { T } ^ { d }$ is a product witness, and by Lemma 9 the first three dimensions are at least $d ;$ none of them exceeds $| { \mathcal { X } } | = d$ . On the other hand, any two sets of $\tau$ share a label, so no pattern has a disjoint neighbor, and $d _ { \mathrm { p a i r } } ( \mathcal { H } _ { \triangle } ) = 0$

Intersections behave diferently, because every singleton in Y is the intersection of two sets of $\tau$ . Choosing these intersections independently at the d inputs shows that $\mathcal { T } _ { 2 } ( \mathcal { H } _ { \triangle } )$ contains every total hypothesis with singleton correct-label sets, and the patterns of these hypotheses form a disjoint-neighbor witness. Thus $d _ { \mathrm { p a i r } } ( \mathcal { T } _ { 2 } ( \mathcal { H } _ { \triangle } ) ) = d$ , and the same holds for $\begin{array} { r } { \mathbb { Z } _ { 3 } ( \mathcal { H } _ { \triangle } ) \supseteq \mathbb { Z } _ { 2 } ( \mathcal { H } _ { \triangle } ) } \end{array}$

If $\mathcal { X } = \mathbb { N } .$ , the same arguments give $d _ { \mathrm { p a i r } } ( \mathcal { H } _ { \triangle } ) = 0$ and infinite values of the other dimensions, so by Theorem $\mathrm { A }$ the class is not learnable even from full sets. Thus finite disjoint-neighbor dimension of the original class does not imply full-set learnability. In this example, $d _ { \mathrm { p a i r } } ( \mathcal { T } _ { 3 } ( \mathcal { H } _ { \triangle } ) ) = \infty$ , so the suficient condition for one-correct-label learnability in Theorem E does not hold.

Example 15 (A three-label separation). Let $\chi =  { \mathbb { N } }$ and $\ y = [ 3 ]$ . For every $b : \mathbb { N }  \{ 0 , 1 \}$ let $h _ { b } ( x ) = \{ 1 , 2 + b ( x ) \}$ , so that $h _ { b } ( x )$ is $\{ 1 , 2 \}$ where $b ( x ) = 0$ and $\{ 1 , 3 \}$ where $b ( x ) = 1$ and let $g ( x ) = \{ 2 , 3 \}$ for all $x .$ . Thus every $h _ { b }$ accepts the label 1 at every input, and g rejects it everywhere. The class $\mathcal { H } _ { \mathrm { b i t } } = \{ h _ { b } : b \in \{ 0 , 1 \} ^ { \mathbb { N } } \} \cup \{ g \}$ , a subclass of $\mathcal { H } _ { \triangle }$ on $\chi = \mathbb { N }$ , satisfies

$$
d _ { \mathrm { N } } ^ { \cap } ( \mathcal { H } _ { \mathrm { b i t } } ) = d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { H } _ { \mathrm { b i t } } ) = d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathcal { H } _ { \mathrm { b i t } } ) = 1 , \qquad d _ { \mathrm { p a i r } } ( \mathcal { H } _ { \mathrm { b i t } } ) = 0 .
$$

Indeed, the three sets of $\tau$ occur at every input and have no common label, so every input is product-shattered with the family $\tau ,$ and by Lemma 9 the first three dimensions are at least one. On two or more inputs, however, patterns that agree at all inputs but one still agree at some input. Since only $g$ takes the value {2, 3}, such patterns either all equal the pattern of $g$ or all come from hypotheses $h _ { b }$ and share the label 1. Hence no hyperedge has an empty intersection and there are no conflict blocks, so there are no DS or packed witnesses on two or more inputs, and $d _ { \mathrm { N } } ^ { \cap } ( \mathcal { H } _ { \mathrm { b i t } } ) \leq d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { H } _ { \mathrm { b i t } } ) = 1$ by Lemma 9. As for $\mathcal { H } _ { \triangle }$ , any two sets of $\tau$ share a label, so $d _ { \mathrm { p a i r } } ( \mathcal { H } _ { \mathrm { b i t } } ) = 0$

The class is easy to learn from full sets and from bandit feedback. The correct-label set of one example, or the answer to the query 1 at one example, reveals whether the target is $g$ or one of the hypotheses $h _ { b }$ . The learner that then predicts 2 at every input if the target is $^ { g , }$ and 1 at every input otherwise, has error zero. Thus $m _ { \mathcal { H } _ { \mathrm { b i t } } } ^ { \mathrm { f u l l } } ( \varepsilon , \delta ) \leq 1$ and $m _ { \mathcal { H } _ { \mathrm { b i t } } } ^ { \mathrm { b a n d } } ( \varepsilon , \delta ) \leq 1$ for all $\varepsilon , \delta \in ( 0 , 1 )$

However, ${ \mathcal { H } } _ { \mathrm { b i t } }$ is not learnable from one correct label. The label $2 + b ( x )$ is correct for both $h _ { b }$ and $g$ , so a demonstrator who always shows it produces the same sample for both targets. Fix m and $n .$ , choose $b ( 1 ) , \ldots , b ( n )$ as independent fair bits and $b ( x ) = 0$ for $x > n$ let P be uniform on $[ n ]$ , and let the demonstrator always show $2 + b ( x )$ . Let the target be $g$ with probability $1 / 3$ and $h _ { b }$ with probability $2 / 3$ , independently of b. Since the sample does not depend on this choice, the target is still $g$ with probability $1 / 3$ given the sample and the learner’s seed, and at a test input absent from the m training inputs, $b ( x )$ is still a fair bit, independent of this choice. At such an input, the label 1 is wrong with probability $1 / 3$ , namely when the target is $g _ { ; }$ , and each of the labels 2 and 3 is wrong with probability $\begin{array} { r } { \frac { 2 } { 3 } \cdot \frac { 1 } { 2 } = \frac { 1 } { 3 } } \end{array}$ , since it is wrong for $h _ { b }$ at exactly one value of $b ( x )$ . The test input is absent from the sample with probability $\textstyle ( 1 - { \frac { 1 } { n } } ) ^ { m }$ , so, averaged over b and the choice of the target, every learner using m examples has expected error at least $\textstyle { \frac { 1 } { 3 } } ( 1 - { \frac { 1 } { n } } ) ^ { m }$ , and the same holds for some fixed b and target, with the demonstrator showing $2 + b ( x )$ . By (4), a learner whose error is at most $1 / 8$ with probability at least $7 / 8$ has expected error at most $1 / 4$ , which is less than $\begin{array} { r } { \frac { 1 } { 3 } ( 1 - \frac { 1 } { n } ) ^ { m } } \end{array}$ once n is large. Hence $m _ { \mathcal { H } _ { \mathrm { b i t } } } ^ { \mathrm { o n e } } ( 1 / 8 , 1 / 8 ) = \infty$

All four dimensions of the intersection class are infinite, already for $\mathcal { T } _ { 2 } ( \mathcal { H } _ { \mathrm { b i t } } )$ . Indeed, $h _ { b } ( x ) \cap g ( x ) = \{ 2 + b ( x ) \}$ at every input, so $\mathcal { T } _ { 2 } ( \mathcal { H } _ { \mathrm { b i t } } )$ contains every total hypothesis with values in $\{ \{ 2 \} , \{ 3 \} \}$ . Thus every sequence of distinct inputs is product-shattered by $\mathcal { T } _ { 2 } ( \mathcal { H } _ { \mathrm { b i t } } )$ with the families $\{ \{ 2 \} , \{ 3 \} \}$ , and the realized patterns form a disjoint-neighbor witness. By Lemma 9, all four dimensions of $\mathcal { T } _ { 2 } ( \mathcal { H } _ { \mathrm { b i t } } )$ are infinite, and so are those of ${ \mathcal { T } } _ { 3 } ( { \mathcal { H } } _ { \mathrm { b i t } } ) \supseteq { \mathcal { T } } _ { 2 } ( { \mathcal { H } } _ { \mathrm { b i t } } )$ In particular, $d _ { \mathrm { N } } ^ { \cap } ( \mathbb { Z } _ { 3 } ( \mathcal { H } _ { \mathrm { b i t } } ) ) = \infty$ , so by Theorem $\mathrm { A } , \mathcal { T } _ { 3 } ( \mathcal { H } _ { \mathrm { b i t } } )$ is not learnable from full sets: the condition of Theorem B fails, although the four dimensions of ${ \mathcal { H } } _ { \mathrm { b i t } }$ itself are at most one.

## 7 Full-Set Learning with Finite Labels

In this section the label space is finite, $k = | \mathcal { V } | < \infty$ , and $1 \ \leq \ \ell \ \leq \ k ;$ ; the class H is nonempty and may be partial, and we write $d = d _ { \mathrm { N } } ^ { \Omega } ( \mathcal { H } )$ . When the label space is finite, realizable and agnostic learnability with full-set feedback coincide, as in the multiclass case [DMY16, HKLM22, HMS25], and both are characterized by either dimension.

Theorem 16 (Full-set learning with finite labels). Let $k = | \mathcal { V } | < \infty$ and let $\mathcal { H } \subseteq ( \mathcal { A } \cup \{ \star \} ) ^ { \mathcal { X } }$ be a nonempty class. The following conditions are equivalent:

(i) $d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { H } ) < \infty ;$

(ii) $d _ { \mathrm { N } } ^ { \Omega } ( \mathcal { H } ) < \infty ;$

(iii) H is realizably PAC learnable with full-set feedback;

(iv) H is agnostically PAC learnable with full-set feedback, with the benchmark (5).

The main tool is a deterministic rule that predicts the labels of a known pool of n inputs one at a time and makes $O _ { \ell } ( d \log ( n k ) )$ mistakes, whatever the target and the order of presentation (Theorem 19). It is an analogue of the halving algorithm, and its potential counts the products of conflicting sets that the remaining hypotheses still realize. This technique was initially inspired by work on transductive and adversary-directed online learning [HRSS24, HS25, HS26].

All our upper bounds come from this rule. A leave-one-out argument turns it into realizable learners, both in expectation and with high probability (Section 7.4). Since the rule changes its state only after a mistake, a run is determined by the few positions where it makes a mistake. Listing the possible runs therefore gives a small finite cover of the class on every pool (Section 7.2), and exponential weights over this cover give the agnostic learner (Section 7.5). For the lower bounds (Section 7.3), a DS witness or a product witness leaves the correct-label set of an unseen test input undetermined, so that every prediction there is wrong with constant probability. Section 7.6 assembles the proof of Theorem 16, and Section 8 adds two further equivalent conditions for bandit feedback.

## 7.1 A Transductive Mistake Bound

As in the one-inclusion graph approach of [HLW94], we first solve a transductive problem, in which the inputs are known in advance. In a transductive prediction problem the learner receives an unlabeled pool $S = ( x _ { 1 } , \ldots , x _ { n } ) \in { \mathcal { X } } ^ { n }$ , possibly with repetitions. Positions $i \in [ n ]$ are then presented one at a time, in any order and possibly more than once. When position i is presented, the learner predicts a label in $\mathcal { V } _ { : }$ , and then, if $x _ { i } \in \operatorname { s u p p } ( h ^ { \star } )$ , it observes the correct-label set $h ^ { \star } ( x _ { i } )$ . A mistake is a prediction outside $h ^ { \star } ( x _ { i } )$ at a position with $x _ { i } \in \operatorname { s u p p } ( h ^ { \star } )$ .

Our rule keeps a set of candidate patterns on the pool. Let

$$
W _ { 0 } = \{ ( h ( x _ { 1 } ) , \ldots , h ( x _ { n } ) ) : h \in \mathcal { H } \} \subseteq ( A \cup \{ \star \} ) ^ { n } ,
$$

and call its elements partial patterns. Unlike $\mathcal { H } | _ { S }$ , the set $W _ { 0 }$ also keeps the patterns of hypotheses that are undefined on part of the pool; the covers of Section 7.2 need them. For $W \subseteq W _ { 0 }$ , a position i and a set $A \in { \mathcal { A } }$ , put

$$
W _ { i , A } = \{ v \in W : v _ { i } = A \} .\tag{11}
$$

The halving algorithm counts the hypotheses that are still consistent, and every mistake removes at least half of them. Here the number of patterns can be exponential in n, so counting them would give a mistake bound linear in $n .$ We count instead the products of minimal empty-intersection families, placed at a few positions of the pool, that $W$ still realizes. Each such product is a product witness of $\mathcal { H } .$ , so, as in the Sauer–Shelah lemma, their number $\Phi ( W )$ is at most a polynomial of degree d in $n$ . The main inequality (Lemma 17) says that if $\boldsymbol { B }$ is a minimal empty-intersection family of $r$ sets, then the potentials $\Phi ( W _ { i , A } ) , A \in \mathfrak { B }$ , sum to at most $( r - 1 ) \Phi ( W )$ . Since $r \leq \ell + 1$ , the sets that keep more than an $\ell / ( \ell + 1 )$ fraction of the potential share a label, and a mistake on this label multiplies the potential by at most $\ell / ( \ell + 1 )$

We now make this precise. Let C be the collection of all minimal empty-intersection families contained in A. By Lemma 8, each of them has between 2 and $\ell + 1$ members, and therefore, with $q = | A |$

$$
\Gamma = | \mathfrak { C } | \leq \sum _ { r = 2 } ^ { \operatorname* { m i n } \{ \ell + 1 , q \} } { \binom { q } { r } } .\tag{12}
$$

Fix a pool $S = ( x _ { 1 } , \ldots , x _ { n } ) $ . For $W \subseteq W _ { 0 }$ , a product specification of $W$ consists of a set $T \subseteq [ n ]$ , its support, and families $\boldsymbol { B } _ { i } \in \mathfrak { C }$ for $i \in T$ , such that the product $\prod _ { i \in T } B _ { i }$ is realized by patterns in W that are defined on T:

$$
\prod _ { i \in T } { \mathcal { B } } _ { i } \subseteq \{ ( v _ { i } ) _ { i \in T } : v \in W , v _ { i } \neq \star { \mathrm { f o r ~ e v e r y ~ } } i \in T \} .\tag{13}
$$

The potential $\Phi ( W )$ is the number of product specifications of $W ;$ the empty specification is counted once when $W \neq \emptyset$ , and $\Phi ( \varnothing ) = 0$ . Define

$$
\Pi _ { d } ( n ) = \sum _ { j = 0 } ^ { \operatorname* { m i n } \{ d , n \} } { \binom { n } { j } } \Gamma ^ { j } ,\tag{14}
$$

where $\Gamma ^ { 0 } = 1$ . If $d < \infty$ , then

$$
1 \leq \Phi ( W ) \leq \Pi _ { d } ( n ) \qquad ( W \neq \emptyset ) .\tag{15}
$$

Indeed, the lower bound counts the empty specification. For the upper bound, note first that repeated inputs cannot both lie in a support: if $i \neq i ^ { \prime }$ and $x _ { i } = x _ { i ^ { \prime } }$ , then every $v \in W _ { 0 }$ has $v _ { i } = v _ { i ^ { \prime } }$ , whereas the product contains vectors with diferent sets at i and $i ^ { \prime } ,$ , since every family in C has at least two members. So the inputs $x _ { i } , i \in T$ , are distinct, and the right side of (13) is contained in ${ \mathcal { H } } | _ { ( x _ { i } ) _ { i \in T } }$ . A specification with $| T | > d$ would thus show that these inputs are product-shattered, contradicting $d _ { \mathrm { N } } ^ { \Omega } ( \mathcal { H } ) = d$ . Hence $| T | \leq \operatorname* { m i n } \{ d , n \}$ , and there are at most $\binom { n } { j } \Gamma ^ { j }$ specifications with $| T | = j$

Lemma 17 (Product-counting inequality). Let $W \subseteq W _ { 0 }$ , let $i \in [ n ]$ , and let $B \in { \mathfrak { C } }$ have r members. Then

$$
\sum _ { A \in B } \Phi ( W _ { i , A } ) \leq ( r - 1 ) \Phi ( W ) .\tag{16}
$$

Proof. Every specification of $W _ { i , A }$ is a specification of $W$ . Its support does not contain $i ,$ because all vectors of $W _ { i , A }$ have the same set A at $i ,$ while each family in C has at least two members. Let U be the set of specifications of W whose support does not contain $i ,$ and for $p \in \mathcal { U }$ let $r _ { p }$ be the number of sets $A \in B$ such that $p$ is a specification of $W _ { i , A }$ . Since $r _ { p } \leq r - 1 + \mathbb { 1 } \{ r _ { p } = r \}$

$$
\sum _ { A \in B } \Phi ( W _ { i , A } ) = \sum _ { p \in \mathcal { U } } r _ { p } \leq ( r - 1 ) | \mathcal { U } | + | \{ p \in \mathcal { U } : r _ { p } = r \} | .
$$

Algorithm 1: Conservative transductive rule   
Input: a pool $S = ( x _ { 1 } , \ldots , x _ { n } ) $ ; positions of S presented one at a time.   
1. Set $W = W _ { 0 } .$   
2. When position i is presented, predict $y _ { i } ( W )$   
3. If a correct-label set $A = h ^ { \star } ( x _ { i } )$ is revealed and $y _ { i } ( W ) \notin A$ , replace $W$ by $W _ { i , A }$   
Otherwise keep W.

Now let $p \in \mathcal { U }$ have support $T$ and $r _ { p } = r$ . Then every vector of the product specified by $p ,$ combined with every $A \in B$ at coordinate i, is realized by some $v \in W$ that is defined on $T \cup \{ i \}$ . So adding the family B at position $i$ extends $p$ to a specification of $W$ with support $T \cup \{ i \}$ . Distinct p give distinct extensions, and none of them lies in $\mathcal { U } .$ . Therefore $\Phi ( W ) \geq | \mathcal { U } | + | \{ p \in \mathcal { U } : r _ { p } = r \} |$ , and since $r \geq 2$ , the right side of the display is at most $( r - 1 ) \Phi ( W )$ □

The rule predicts a label shared by all the sets that keep a large part of the potential. For $W \subseteq W _ { 0 }$ and $i \in [ n ]$ , let

$$
\mathcal { G } = \Big \{ A \in \mathcal { A } : \Phi ( W _ { i , A } ) > \frac { \ell } { \ell + 1 } \Phi ( W ) \Big \} ,
$$

and let $y _ { i } ( W )$ be the first label of $\cap _ { A \in { \mathcal { G } } } A$ in the fixed enumeration of $\mathcal { V } ,$ or the first label of ${ \mathcal { V } } { \mathrm { ~ i f ~ } } { \mathcal { G } } = \emptyset$ (for instance, if $W = \varnothing )$ . The next lemma shows that the intersection is nonempty, so $y _ { i } ( W )$ is well defined.

Lemma 18 (A prediction that shrinks the potential on a mistake). Let $W \subseteq W _ { 0 }$ be nonempty and let $i \in [ n ]$ $I f \mathcal { G } \neq \emptyset$ , then $\cap _ { A \in { \mathcal { G } } } A \neq \emptyset$ . Moreover,

$$
y _ { i } ( W ) \not \in A \implies \Phi ( W _ { i , A } ) \leq \frac \ell { \ell + 1 } \Phi ( W ) \qquad ( A \in \mathcal { A } ) .\tag{17}
$$

Proof. Suppose that $\mathcal { G } \neq \emptyset$ and $\cap _ { A \in { \mathcal { G } } } A = \emptyset$ . By Lemma $8 , { \mathcal { G } }$ contains a family $B \in { \mathfrak { C } }$ with some number $r \leq \ell + 1$ of members. Then $r \ell / ( \ell + 1 ) \geq r - 1$ , and so

$$
\sum _ { A \in B } \Phi ( W _ { i , A } ) > \frac { r \ell } { \ell + 1 } \Phi ( W ) \geq ( r - 1 ) \Phi ( W ) ,
$$

which contradicts Lemma 17. Hence the intersection is nonempty. If $y _ { i } ( W ) \notin A$ , then $A \notin { \mathcal { G } }$ 2 which is (17). □

We call W the version space of the rule. A rule is conservative if it changes its state only after a mistake, as Algorithm 1 does. Its mistake bound is the logarithm of the initial

potential in base $1 + 1 / \ell \colon$

$$
M _ { d } ( n ) = \left\lceil \frac { \log \Pi _ { d } ( n ) } { \log ( 1 + 1 / \ell ) } \right\rceil .\tag{18}
$$

Both $\Pi _ { d } ( n )$ and $M _ { d } ( n )$ are nondecreasing in n. For finite $d ,$ since ${ \textstyle \binom { n } { j } } \leq n ^ { j } \leq { \textstyle \binom { d } { j } } n ^ { j }$ for $j \leq d ,$ the binomial theorem gives $\begin{array} { r } { \Pi _ { d } ( n ) \le \sum _ { j \le d } { \binom { d } { j } } ( n \Gamma ) ^ { j } = ( 1 + n \Gamma ) ^ { d } } \end{array}$ , and hence

$$
M _ { d } ( n ) \leq \left\lceil \frac { d \log ( 1 + n \Gamma ) } { \log ( 1 + 1 / \ell ) } \right\rceil .\tag{19}
$$

Theorem 19 (Conservative transductive mistake bound). Assume $d < \infty$ , and let S be a pool of n inputs. Algorithm 1 is deterministic and conservative. For every target $h ^ { \star } \in { \mathcal { H } }$ and every order of presentation, with repetitions allowed, it makes at most $M _ { d } ( n )$ mistakes.

Proof. The rule is clearly deterministic and conservative. Let $v ^ { \star } = ( h ^ { \star } ( x _ { 1 } ) , \ldots , h ^ { \star } ( x _ { n } ) ) \in W _ { 0 }$ An update after a mistake at i uses $A = h ^ { \star } ( x _ { i } ) = v _ { i } ^ { \star }$ , so $v ^ { \star }$ stays in W. Hence W is never empty, and $\Phi ( W ) \geq 1$ by (15). On a mistake we have $y _ { i } ( W ) \not \in A$ , so by (17) the update multiplies $\Phi ( W )$ by at most $\ell / ( \ell + 1 )$ , and all other steps leave W unchanged. Thus after s mistakes

$$
1 \leq \Phi ( W ) \leq \Pi _ { d } ( n ) \left( { \frac { \ell } { \ell + 1 } } \right) ^ { s } ,
$$

where the second inequality uses (15) for $W _ { 0 }$ . Taking logarithms, and using that s is an integer, gives $s \leq M _ { d } ( n )$ □

Remark 20 (Labeling the one-inclusion hypergraph). When all correct-label sets are singletons, a labeling of the hyperedges of the one-inclusion hypergraph by labels of their vertices is an orientation [HLW94, DSS14, BCD<sup>+</sup>22], and Theorem 19 gives the analogue of an orientation with small out-degree. Precisely, assume $d < \infty$ and let $S \in \mathcal { X } ^ { n }$ . For every finite nonempty $V \subseteq \mathcal { H } | _ { S }$ there is a map $\phi$ from the hyperedges of the one-inclusion hypergraph of V to $\mathcal { V }$ such that

$$
\operatorname* { m a x } _ { v \in V } \sum _ { i = 1 } ^ { n } \mathbb { 1 } \{ \phi ( e _ { i , v _ { - i } } ) \notin v _ { i } \} \leq M _ { d } ( n ) .\tag{20}
$$

Indeed, all vertices of $e _ { i , v . }$ agree at the coordinates $1 , \ldots , i - 1$ ; let $\phi ( e _ { i , v _ { - i } } )$ be the prediction of Algorithm 1 at position i after positions $1 , \ldots , i - 1$ are presented in this order with these common sets. For each $v \in V$ , the labels $\phi ( e _ { i , v _ { - i } } ) , i \in [ n ]$ , are then the predictions of Algorithm 1 in the order $1 , \ldots , n$ against a target realizing v, and Theorem 19 gives (20).

## 7.2 Finite Covers

A finite cover of H on a pool $S = ( x _ { 1 } , \ldots , x _ { n } ) $ is a finite family $F \subseteq \mathcal { V } ^ { n }$ with the following covering property: for every $h \in \mathcal H$ some $f \in F$ satisfies $f ( i ) \in h ( x _ { i } )$ at every position i with $x _ { i } \in \operatorname { s u p p } ( h )$ . Because Algorithm 1 is conservative, its run against a target is determined by the positions of its mistakes and the sets revealed there, and there are few such runs; correcting the wrong predictions of each run gives the cover.

Lemma 21 (Finite cover of a partial class). Assume $d < \infty$ . For every pool $S = ( x _ { 1 } , \ldots , x _ { n } ) $ there is a finite cover $F _ { S } \subseteq \mathcal { V } ^ { n }$ of H on S, which depends only on S and H, such that

$$
| F _ { S } | \leq \sum _ { j = 0 } ^ { \operatorname* { m i n } \{ n , M _ { d } ( n ) \} } { \binom { n } { j } } ( q k ) ^ { j } \leq ( 1 + n q k ) ^ { M _ { d } ( n ) } .\tag{21}
$$

Proof. We simulate Algorithm 1 with the positions presented in the order $1 , \ldots , n ,$ , guessing where it makes mistakes. For every set $I \subseteq [ n ]$ with $| I | \leq M _ { d } ( n )$ and every choice of $A _ { i } \in { \mathcal { A } }$ and $b _ { i } \in A _ { i }$ for $i \in I$ , we form one vector $f \in \mathcal { V } ^ { n }$ as follows. Start with $W = W _ { 0 }$ . At a position $i \in I$ , set $f ( i ) = b _ { i }$ and replace W by $W _ { i , A _ { i } }$ . At a position $i \not \in I$ , set $f ( i ) = y _ { i } ( W )$ and keep W. Let $F _ { S }$ be the set of all vectors obtained in this way; the construction uses only S and H. For $| I | = j$ there are $\binom { n } { j }$ choices of I and at most $( q k ) ^ { j }$ choices of the pairs $( A _ { i } , b _ { i } )$ , which gives the first bound in (21). The second follows from ${ \binom { n } { j } } \leq n ^ { j }$ and $\begin{array} { r } { \sum _ { j = 0 } ^ { M } ( n q k ) ^ { j } \leq ( 1 + n q k ) ^ { M } } \end{array}$ for $M = M _ { d } ( n )$

Now fix $h \in \mathcal H$ . Run Algorithm 1 against the target h in the order $1 , \ldots , n ,$ , and let I be the set of positions where it makes a mistake; $| I | \leq M _ { d } ( n )$ by Theorem 19. Choose this I, the sets $A _ { i } = h ( x _ { i } )$ , and any labels $b _ { i } \in h ( x _ { i } )$ for $i \in I .$ . The simulation and the run then update at the same positions with the same sets, so by induction on the position they have the same version space at every step. At a position $i \in I$ we have $f ( i ) = b _ { i } \in h ( x _ { i } )$ . At a position $i \not \in I$ with $x _ { i } \in \operatorname { s u p p } ( h )$ , the value $f ( i )$ is the prediction of Algorithm 1, which is not a mistake. Hence $f ( i ) \in h ( x _ { i } )$ at every position with $x _ { i } \in \operatorname { s u p p } ( h )$ □

## 7.3 Lower Bounds

Both bounds draw the target at random from a witness, so that the correct-label set at a test input missing from the sample is not determined by the sample. The first bound is reused in Sections 8 and 10 for the other two feedback models.

Proposition 22 (DS lower bound). Let $k < \infty$ and let H be a partial class with $d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { H } ) \geq$ $n \geq 2$ . For every learner that uses m full-set examples there exist a distribution $P$ and $h ^ { \star } \in { \mathcal { H } }$ with $P ( \operatorname { s u p p } ( h ^ { \star } ) ) = 1$ such that

$$
\mathbb { E } _ { \sigma \sim \mathcal { D } _ { P , h ^ { \star } } ^ { m } } \left[ L _ { P , h ^ { \star } } ( \widehat { g } ) \right] \geq \frac { 1 } { q } \left( 1 - \frac { 1 } { n } \right) ^ { m } ,\tag{22}
$$

where $\widehat g$ is the returned predictor. The same holds for every learner that uses m examples with bandit feedback, and for every learner that uses m examples with one-correct-label feedback, where the demonstrated label is drawn uniformly from $h ^ { \star } ( x )$

Proof. Since $d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { H } ) \geq n$ , the projection part of Lemma 9 gives distinct inputs $x _ { 1 } , \ldots , x _ { n }$ and a DS witness $V \subseteq { \mathcal { H } } | _ { ( x _ { 1 } , . . . , x _ { n } ) }$ . For each $v \in V$ fix $h _ { v } \in \mathcal { H }$ with $( h _ { v } ( x _ { 1 } ) , \ldots , h _ { v } ( x _ { n } ) ) = v$ Let P be uniform on $\{ x _ { 1 } , \ldots , x _ { n } \}$ , so that $P ( \operatorname { s u p p } ( h _ { v } ) ) = 1$ . We choose v uniformly from V, use the target $h _ { v } .$ , and prove (22) on average over $v ;$ then it holds for some $h _ { v }$

Draw a test point $x _ { i } \sim P$ independently of the sample, and condition on the event that $x _ { i }$ is not among the training inputs. Give the learner $v _ { - i }$ as well; this can only help it. The training observations are functions of $v _ { - i } .$ the training inputs and the seed, so the conditional law of $v$ is uniform on the hyperedge $e _ { i , v _ { - i } }$ . Distinct vertices of this hyperedge difer at coordinate $i ,$ so it has at most q vertices, and by the definition of a DS witness every label $y$ is missing from $w _ { i }$ for some $w \in e _ { i , v _ { - i } }$ . Hence any prediction at $x _ { i }$ , deterministic or randomized, is wrong with conditional probability at least $1 / | e _ { i , v _ { - i } } | \geq 1 / q$ . Finally, a uniform test point is missing from $m$ independent uniform training points with probability $( 1 - 1 / n ) ^ { m }$ . Averaging over the target, the sample and the test point gives (22).

For the other feedback models, note that a full-set learner can answer every membership query $\mathbb { 1 } \{ u \in h ^ { \star } ( x _ { j } ) \}$ and can draw a label uniformly from $h ^ { \star } ( x _ { j } )$ with its own seed. So every bandit learner, and every one-correct-label learner under the uniform law of demonstrations, is simulated by a full-set learner with the same output, and the bound transfers. A lower bound for one admissible law of the demonstrations is enough, since a learner must succeed under all of them. □

The second bound replaces the factor $1 / q$ by $1 / ( \ell + 1 )$ and gives the linear dependence of the sample complexity on d.

Proposition 23 (Product-witness lower bound). $I f d _ { \mathrm { N } } ^ { \cap } ( \mathcal { H } ) \geq n \geq 2$ , then for every learner that uses m full-set examples there are P and $h ^ { \star } \in { \mathcal { H } }$ with $P ( \operatorname { s u p p } ( h ^ { \star } ) ) = 1$ such that

$$
\mathbb { E } \big [ L _ { P , h ^ { \star } } ( \widehat { g } ) \big ] \geq \frac { 1 } { \ell + 1 } \left( 1 - \frac { 1 } { n } \right) ^ { m } .\tag{23}
$$

Consequently, $i f d \ge 2$ , then for $0 < \varepsilon \le 1 / [ 8 ( \ell + 1 ) ]$ and $0 < \delta < 1 / [ 4 ( \ell + 1 ) ]$

$$
m _ { \mathcal { H } } ^ { \mathrm { f u l l } } ( \varepsilon , \delta ) > \frac { d - 1 } { 8 ( \ell + 1 ) \varepsilon } ,\tag{24}
$$

and $i f d = \infty$ , this holds with every integer $n \geq 2$ in place of d.

Proof. Take a product-shattered sequence $x _ { 1 } , \ldots , x _ { n }$ with families $\boldsymbol { B } _ { 1 } , \ldots , \boldsymbol { B } _ { n }$ , and replace each family by a minimal empty-intersection subfamily; by Lemma 8, these have between 2 and $\ell + 1$ members. Choose the sets $v _ { i } \in B _ { i }$ independently and uniformly, fix one target $h _ { v } \in \mathcal { H }$ realizing each product vector $v ,$ and let $P$ be uniform on $\{ x _ { 1 } , \ldots , x _ { n } \}$ . Since the sets at distinct inputs are independent, the set $v _ { i }$ at an input $x _ { i }$ absent from the training sample is independent of all observed sets, and it is uniform on $B _ { i }$ . Every label is excluded by some member of $B _ { i }$ , so the conditional probability of an error at $x _ { i }$ is at least $1 / | B _ { i } | \geq 1 / ( \ell + 1 )$ The test input is missing with probability $( 1 - 1 / n ) ^ { m }$ , which gives (23).

For the PAC statement we make the unseen inputs rare. Fix the set at $x _ { n }$ to one member of $B _ { n }$ and keep the independent uniform choices at $x _ { 1 } , \ldots , x _ { n - 1 }$ . Let $a = 4 ( \ell + 1 ) \varepsilon \leq 1 / 2$ put mass $a / ( n - 1 )$ on each of $x _ { 1 } , \ldots , x _ { n - 1 }$ and the remaining mass $1 - a$ on $x _ { n }$ , and let $Z$ be the part of the risk contributed by $x _ { 1 } , \ldots , x _ { n - 1 }$ . The argument above gives

$$
0 \leq Z \leq a , \qquad \mathbb { E } [ Z ] \geq { \frac { a } { \ell + 1 } } \left( 1 - { \frac { a } { n - 1 } } \right) ^ { m } .
$$

If $m \leq ( n - 1 ) / [ 8 ( \ell + 1 ) \varepsilon ] = ( n - 1 ) / ( 2 a )$ , then $( 1 - a / ( n - 1 ) ) ^ { m } \geq 1 - m a / ( n - 1 ) \geq 1 / 2 .$ and so $\mathbb { E } [ Z ] ~ \ge ~ 2 \varepsilon$ . The reverse Markov inequality for $Z ~ \in ~ [ 0 , a ]$ , which follows from $Z \leq \varepsilon + a \mathbb { 1 } \{ Z > \varepsilon \}$ , then gives

$$
\operatorname* { P r } [ Z > \varepsilon ] \ge \frac { \mathbb { E } [ Z ] - \varepsilon } { a } \ge \frac { \varepsilon } { a } = \frac { 1 } { 4 ( \ell + 1 ) } > \delta .
$$

The risk is at least $Z ,$ and some fixed target attains at least this average failure probability. Taking $n = d$ gives (24). □

Proposition 23 holds for every label space, since its proof does not use the finiteness of $\mathcal { V } .$ The term $d - 1$ cannot be replaced by d for all classes: a class of two distinct constant singleton hypotheses has $d _ { \mathrm { N } } ^ { \cap } = 1$ and is identified from one example.

## 7.4 Realizable Full-Set Learning

We first turn Algorithm 1 into an inductive learner by the leave-one-out argument of [HLW94]: the learner places the query among the training inputs in a random order, and its expected risk is the average number of mistakes per position of Algorithm 1.

Theorem 24 (Expected risk). Assume $d < \infty$ . For every m $\geq 0$ there is a full-set learner such that for every $h ^ { \star } \in { \mathcal { H } }$ and every P with $P ( \operatorname { s u p p } ( h ^ { \star } ) ) = 1$

$$
\mathbb { E } \underset { \omega \sim \mathrm { U n i f } ( [ 0 , 1 ] ) } { \sim } \left[ L _ { P , h ^ { \star } } \left( \widehat { g } \right) \right] \leq \frac { M _ { d } ( m + 1 ) } { m + 1 } .\tag{25}
$$

Proof. We describe the learner and then compare it with a single run of Algorithm 1 on $m + 1$ examples.

The learner. Given $\sigma = \left( ( x _ { 1 } , A _ { 1 } ) , \dots , ( x _ { m } , A _ { m } ) \right)$ and a query x, let $n = m + 1$ and add x to the training inputs as position n. Use the seed to draw a uniformly random ordering of the n positions. Run Algorithm 1 on the pool listed in this order, revealing the observed sets $A _ { j }$ at the training positions that come before the query, and return the prediction at the query. The ordering is a function of the seed and the query, so for fixed $\sigma$ and $\omega$ the learner returns a fixed predictor ${ \widehat { g } } ;$ the agnostic learners below use the seed in the same way.

Analysis. Draw n independent examples $( x _ { j } , h ^ { \star } ( x _ { j } ) )$ ; almost surely all $x _ { j }$ lie in $\operatorname { s u p p } ( h ^ { \star } )$ List them in a uniformly random order and run Algorithm 1 on the whole sequence, revealing every set. The prediction at the t-th listed position depends on the observed sets only through those revealed before it, so it is the prediction of the learner trained on the other $n - 1$ examples with the t-th one as query. By exchangeability, the loss of the learner at a test point has the same law as the loss at a uniformly chosen listed position. For each fixed sequence and order, Algorithm 1 makes at most $M _ { d } ( n )$ mistakes by Theorem 19, so the average loss over a uniform position is at most $M _ { d } ( n ) / n$ . Averaging over the examples and the order gives (25). □

Markov’s inequality turns Theorem 24 into a PAC bound with a factor $1 / \delta$ . To get $\log ( 1 / \delta )$ instead, we use the leave-one-out conversion of $\left[ \mathrm { A A C S Z 2 3 } \right]$ . Write $\sigma ^ { - i }$ for a sample σ of size n with its i-th example deleted, and $\sigma _ { \leq t }$ for its first t examples. The conversion takes a deterministic, permutation-invariant rule $R ( \sigma ; x )$ that makes few leave-one-out mistakes, that is, such that on every realizable sample $\sigma = ( ( x _ { i } , h ^ { \star } ( x _ { i } ) ) ) _ { i = 1 } ^ { n }$

$$
\sum _ { i = 1 } ^ { n } \mathbb { 1 } \{ R ( \sigma ^ { - i } ; x _ { i } ) \notin h ^ { \star } ( x _ { i } ) \} \le M _ { n } ,\tag{26}
$$

and it shows that, with high probability, the rules $R ( \sigma _ { \leq t } ; \cdot ) , n / 4 \leq t < n$ , have small average risk. Their theorem allows a general {0, 1}-valued loss, so it applies to the loss $\mathbb { 1 } \{ y \notin A \}$ . We obtain a suitable rule by running Algorithm 1 on the distinct sample inputs and the query, listed in a fixed order on $\mathcal { X }$

Theorem 25 (High-probability bound). Fix any total order on X. Assume $d < \infty$ , let $\delta \in ( 0 , 1 )$ , and let n be a positive multiple of 4. There is a deterministic learner such that for every $h ^ { \star } \in { \mathcal { H } }$ and every P with $P ( \operatorname { s u p p } ( h ^ { \star } ) ) = 1$ , with probability at least $1 - \delta$ over $\sigma \sim \mathcal { D } _ { P , h ^ { \star } } ^ { n }$ , the returned predictor $\widehat g$ satisfies

$$
L _ { P , h ^ { \star } } ( \widehat g ) \leq \frac { 4 . 8 2 ( \ell + 1 ) } { n } \left( M _ { d } ( n ) + \log \frac { 2 } { \delta } \right) .\tag{27}
$$

Proof. We build a rule that satisfies (26) with $M _ { n } = M _ { d } ( n )$ , apply the conversion, and take a plurality vote.

A leave-one-out rule. Given a sample σ and a query x, define $R ( \sigma ; x )$ as follows. If σ is not realizable by ${ \mathcal { H } } ,$ return a fixed label. If x is one of the sample inputs, return the first label of its observed set. Otherwise, list the distinct sample inputs together with x in increasing order and run Algorithm 1 on this pool, presenting only the inputs smaller than $x ,$ in increasing order and with their observed sets; return its prediction at x. This rule depends only on the sample as an unordered collection, so it is permutation invariant.

The leave-one-out condition. Fix a realizable sample $\sigma = ( ( x _ { i } , h ^ { \star } ( x _ { i } ) ) ) _ { i = 1 } ^ { n }$ . If $x _ { i }$ occurs at least twice in $\sigma _ { : }$ , then it occurs in $\sigma ^ { - i }$ , and $R ( \sigma ^ { - i } ; x _ { i } )$ is correct. If $x _ { i }$ occurs once, the pool used by $R ( \sigma ^ { - i } ; x _ { i } )$ is the list of distinct inputs of $\sigma ,$ the same for all such $i ,$ and the inputs smaller than $x _ { i }$ all remain in $\sigma ^ { - i }$ . So $R ( \sigma ^ { - i } ; x _ { i } )$ is the prediction at $x _ { i }$ of a single run of Algorithm 1 on the distinct inputs of σ in increasing order, with every set revealed. This pool has at most $n$ positions, so by Theorem 19 and the monotonicity of $M _ { d }$ , the run makes at most $M _ { d } ( n )$ mistakes. Thus (26) holds with $M _ { n } = M _ { d } ( n )$

Applying the conversion. For a fixed target, we apply Theorem 2.1 of [AACSZ23] on the input space $\operatorname { s u p p } ( h ^ { \star } )$ , with the loss 1 $\{ y \notin A \}$ and the rule R restricted there.<sup>7</sup> Every realizable sample there satisfies (26) with $M _ { n } = M _ { d } ( n )$ , and the rule itself is defined on all of $\mathcal { X }$ and does not depend on $h ^ { \star }$ . The theorem gives, with probability at least $1 - \delta$

$$
\frac { 4 } { 3 n } \sum _ { t = n / 4 } ^ { n - 1 } L _ { P , h ^ { \star } } \left( R ( \sigma _ { \leq t } ; \cdot ) \right) \leq \frac { 4 . 8 2 } { n } \left( M _ { d } ( n ) + \log \frac { 2 } { \delta } \right) ,\tag{28}
$$

where the left side is the average risk of the $3 n / 4$ rules.

Plurality. Let ${ \widehat { g } } ( x )$ be the plurality of the $3 n / 4$ labels $R ( \sigma _ { \leq t } ; x ) , n / 4 \leq t < n$ , and let $N ( x )$ be the number of these labels outside $h ^ { \star } ( x )$ . By Lemma 11 applied to the votes, $\mathbb { 1 } \{ \widehat { g } ( x ) \ \notin \ h ^ { \star } ( x ) \} \ \le \ ( \ell + 1 ) N ( x ) / ( 3 n / 4 )$ . Integrating over $x \sim P$ and using (28) proves (27). □

Corollary 26 (Realizable sample complexity). $I f d = d _ { \mathrm { N } } ^ { \cap } ( \mathcal { H } ) < \infty$ , then

$$
m _ { \mathcal { H } } ^ { \mathrm { f u l l } } ( \varepsilon , \delta ) \leq \frac { c } { \varepsilon } \Big ( d \log \frac { ( d + 1 ) ( k + 1 ) } { \varepsilon } + \log \frac { 2 } { \delta } \Big ) ,
$$

where c depends only on $\ell .$

Proof. We have $q \leq ( k + 1 ) ^ { \ell }$ and $\Gamma \leq ( q + 1 ) ^ { \ell + 1 }$ , so (19) gives $M _ { d } ( n ) \leq c d \log ( 2 n ( k + 1 ) )$ for $d \geq 1$ , while $M _ { 0 } ( n ) = 0$ . Thus the right side of (27) is at most ε as soon as $n \geq$ $c ( d \log ( 2 n ( k + 1 ) ) + \log ( 2 / \delta ) ) / \varepsilon$ . Since $n \geq 2 a \log ( 2 a b )$ implies $n \geq a \log ( b n )$ whenever $a b \geq e ,$ , a routine calculation shows that this holds, with a larger $c ,$ for every n divisible by 4 that is at least the bound of the corollary. □

## 7.5 Agnostic Full-Set Learning

Recall that D is a distribution on $\mathcal { X } \times \mathcal { A }$ and that the benchmark $L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } )$ in (5) counts an undefined value as a mismatch. The cover of Lemma 21 depends only on the inputs, so its members can serve as experts before any set is seen, and exponential weights over them compete with every hypothesis in H.

Theorem 27 (Expected excess risk). Assume d $< \infty$ . For every $m \geq 0$ there is a randomized full-set learner such that for every D,

$$
\mathbb { E } _ { \omega \sim \Pi _ { \mathrm { n i f } } ^ { \sigma \sim \mathcal { D } ^ { m } } [ 0 , 1 ] ) } \left[ L _ { \mathcal { D } } ( \widehat { g } ) \right] \leq L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } ) + \sqrt { \frac { 2 M _ { d } ( m + 1 ) \log ( 1 + ( m + 1 ) q k ) } { m + 1 } } .\tag{29}
$$

Proof. Let $n = m + 1$ . We first bound the regret on a known pool and then pass to the inductive learner.

Exponential weights on a known pool. Fix a pool S of n inputs with sets $A _ { 1 } , \ldots , A _ { n } \in$ ${ \mathcal { A } } ,$ revealed one at a time in the order of the pool, and let $F _ { S }$ be the cover of Lemma 21. Run exponential weights with the vectors $f \in F _ { S }$ as experts and the losses $\lambda _ { i } ( f ) = \mathbb { 1 } \{ f ( i ) \notin A _ { i } \}$ at position $i ,$ the learner draws an expert with probability proportional to its current weight and predicts its label $\widehat { Y } _ { i }$ . The standard Hedge bound for losses in [0, 1], with a suitable learning rate, gives for every $f \in F _ { S }$

$$
\mathbb { E } _ { \omega \sim \mathrm { U n i f } ( [ 0 , 1 ] ) } \left[ \sum _ { i = 1 } ^ { n } 1 \{ \widehat { Y } _ { i } \notin A _ { i } \} \right] \leq \sum _ { i = 1 } ^ { n } \lambda _ { i } ( f ) + \sqrt { 2 n \log | F _ { S } | } .\tag{30}
$$

For $h \in { \mathcal { H } }$ , let $f _ { h } \in F _ { S }$ be given by the covering property. Then for every position $i ,$

$$
\mathbb { 1 } \{ f _ { h } ( i ) \notin A _ { i } \} \le \mathbb { 1 } \{ h ( x _ { i } ) \neq A _ { i } \} ,
$$

because $h ( x _ { i } ) = A _ { i }$ implies $x _ { i } \in \operatorname { s u p p } ( h )$ and hence $f _ { h } ( i ) \in h ( x _ { i } ) = A _ { i }$

The inductive learner. Given σ and a query x, add x to the training inputs and list the n positions in a uniformly random order. Run exponential weights with the cover of this ordered pool, processing the training examples before the query, and return its prediction at the query. Since the cover depends only on the inputs and their order, the argument of Theorem 24 shows that the expected test loss equals the expected average loss of exponential weights over n independent examples from D. Applying (30) with $f = f _ { h }$ for a fixed $h \in { \mathcal { H } }$ dividing by n and taking expectations gives

$$
\mathbb { E } \big [ L _ { \mathcal { D } } ( \widehat { g } ) \big ] \leq \operatorname* { P r } _ { ( x , A ) \sim \mathcal { D } } [ h ( x ) \neq A ] + \sqrt { \frac { 2 \log \operatorname* { s u p } _ { S \in \mathcal { X } ^ { n } } | F _ { S } | } { n } } ,
$$

and the theorem follows from (21) by taking the infimum over $h \in { \mathcal { H } }$

A standard confidence-boosting step converts this expected bound into a high-probability one: we train independent candidates and keep the one with the smallest validation loss.

Proposition 28 (From expected to high-probability excess risk). Let $\varepsilon , \delta \in ( 0 , 1 )$ , and suppose a full-set learner trained on m examples satisfies $\mathbb { E } [ L _ { \mathcal { D } } ( \widehat { g } ) ] \leq L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } ) + \varepsilon / 8$ for every D. Run it independently on b blocks of m training examples, and select the candidate with

the smallest empirical loss on a common independent validation set of v examples, where

$$
b = \left\lceil { \frac { 8 } { 3 \varepsilon } } \log { \frac { 2 } { \delta } } \right\rceil , \qquad v = \left\lceil { \frac { 8 } { \varepsilon ^ { 2 } } } \log { \frac { 4 b } { \delta } } \right\rceil .
$$

With probability at least $1 - \delta$ , the selected predictor has risk at most $L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } ) + \varepsilon$

Proof. Let $a = L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } )$ ; we may assume $a + \varepsilon < 1$ , since otherwise there is nothing to prove. By Markov’s inequality, one run has risk at most $a + \varepsilon / 2$ with probability at least

$$
1 - { \frac { a + \varepsilon / 8 } { a + \varepsilon / 2 } } = { \frac { 3 \varepsilon / 8 } { a + \varepsilon / 2 } } \geq { \frac { 3 \varepsilon } { 8 } } ,
$$

since $a + \varepsilon / 2 < 1$ . Hence, except with probability at most $( 1 - 3 \varepsilon / 8 ) ^ { b } \leq \delta / 2$ , some candidate has risk at most $a + \varepsilon / 2$ . Conditionally on the candidates ${ \widehat { g } } _ { 1 } , \ldots , { \widehat { g } } _ { b }$ , the validation losses of each candidate are independent Bernoulli variables. Let $\widehat { L } _ { j }$ be the empirical validation loss of $\widehat { g } _ { j }$ . By Hoefding’s inequality and a union bound over $j \leq b _ { i }$

$$
\operatorname* { m a x } _ { j \leq b } \mathopen { } \mathclose \bgroup \left| \widehat { L } _ { j } - L _ { \mathcal { D } } ( \widehat { g } _ { j } ) \aftergroup \egroup \right| \leq \varepsilon / 4
$$

with probability at least $1 - \delta / 2$ . On both events, the risk of the selected candidate exceeds the smallest risk among the candidates by at most $\varepsilon / 2$ , so it is at most $a + \varepsilon$ □

## 7.6 Proof of Theorem 16

Proof of Theorem $1 6 . \ ( \mathrm { i } ) { \Rightarrow } ( \mathrm { i i } )$ follows from $d _ { \mathrm { N } } ^ { \cap } ( \mathcal { H } ) \leq d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { H } )$ in Lemma 9.

(ii)⇒(iii) follows from Corollary 26, which bounds $m _ { \mathcal { H } } ^ { \mathrm { f u l l } } ( \varepsilon , \delta )$ for all $\varepsilon , \delta \in ( 0 , 1 )$

(ii)⇒(iv). By (19), the excess term in (29) tends to zero as $m \to \infty$ . So for m large the learner of Theorem 27 satisfies the assumption of Proposition 28, which gives agnostic PAC learnability.

$( \mathrm { i v } ) { \Rightarrow } ( \mathrm { i i i } )$ . Every realizable distribution $\mathcal { D } _ { P , h }$ ⋆ has benchmark zero and satisfies $L _ { \mathcal { D } } ( g ) =$ $L _ { P , h ^ { \star } } ( g )$ for every predictor g (Section 5.2), so an agnostic learner is a realizable learner with the same sample size.

$( \mathrm { i i i } ) { \Rightarrow } ( \mathrm { i } )$ . Suppose $d _ { \mathrm { D S } } ^ { \Omega } ( \mathcal { H } ) = \infty$ , and let $\varepsilon = \delta = 1 / ( 5 q )$ , so that $\varepsilon + \delta < 1 / ( 2 q )$ . Fix any learner with m examples. By Proposition 22 with n large enough that $( 1 - 1 / n ) ^ { m } \geq 1 / 2$ some target has expected risk $\mathbb { E } [ L ] \ \geq \ 1 / ( 2 q )$ . Since the risk L is at most 1, we have $\mathbb { E } [ L ] \leq \varepsilon + \operatorname* { P r } [ L > \varepsilon ]$ , and so

$$
\mathrm { P r } [ L > \varepsilon ] \ge \frac { 1 } { 2 q } - \frac { 1 } { 5 q } > \delta .
$$

So no finite sample size works, and H is not realizably PAC learnable.

## 8 Bandit Learning with Finite Labels

We now show that when the label space is finite, $k = | \mathcal { V } | < \infty$ , learning from bandit feedback (Section 5.2) is characterized by the same conditions as learning from full sets.

Theorem 29 (Bandit learning with finite labels). Let $k = | \mathcal { V } | < \infty$ and let $\mathcal { H } \subseteq ( \mathcal { A } \cup \{ \star \} ) ^ { \mathcal { X } }$ be a nonempty partial class. Conditions $( i ) - ( i v )$ of Theorem 16 are equivalent to each of the following conditions:

(v) H is realizably PAC learnable with bandit feedback;

(vi) H is agnostically PAC learnable with bandit feedback.

Moreover, suppose that $d = d _ { \mathrm { N } } ^ { \cap } ( \mathcal { H } ) < \infty$ , and let $q = | { \mathcal { A } } |$ be the number of possible correct-label sets. For every $m \geq 0$ , with $n = m + 1$ , there are bandit learners with m examples whose expected risks are at most

$$
\operatorname* { m i n } \left\{ 1 , { \frac { k M _ { d } ( n ) \log ( 1 + n q k ) } { n } } \right\} \qquad a n d \qquad L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } ) + \sqrt { \frac { 2 k M _ { d } ( n ) \log ( 1 + n q k ) } { n } }\tag{31}
$$

in the realizable and the agnostic case, respectively, where $M _ { d } ( n )$ is the mistake bound (18) of Theorem 19 and $L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } )$ is the exact-set mismatch benchmark in (5). Consequently, $m _ { \mathcal { H } } ^ { \mathrm { b a } }$ nd and $m _ { \mathcal { H } } ^ { \mathrm { b a n d , a g } }$ satisfy the bounds (41) and (42) of Proposition 35, where $m _ { \mathrm { B } } ( r )$ is the least m for which the corresponding excess term in (31) is at most r.

All our learners use the finite cover $F _ { S }$ of Lemma 21 as a set of experts. Since the cover is built from the inputs $S = ( x _ { 1 } , \ldots , x _ { n } ) $ alone, the experts are available before any correct-label set is seen. The dificulty is that bandit feedback reveals their losses only partly. A query u at position i tests only the experts recommending $u ,$ and it tests them all at once: if the answer is negative, all of them are wrong at $i ,$ and if it is positive, all of them are right there.

We first use this on a known pool, where the learner chooses its queries after seeing the whole pool (Section 8.1). We then treat the inductive problem, in which the training queries are made before the test input is known (Section 8.2), and finally turn the bounds on the expected risk into PAC bounds by validation (Section 8.3).

## 8.1 Transductive Bounds

In the transductive problem of Section 7.1 with bandit feedback, the learner knows the pool $S = ( x _ { 1 } , \ldots , x _ { n } ) $ in advance; when position i is presented, it predicts $\hat { y } _ { i } \in \mathcal { V }$ and observes only $\mathbb { 1 } \{ \hat { y } _ { i } \in A _ { i } \}$ . On such a pool, the cover turns the halving algorithm into a bandit learner: predict the most popular label among the surviving experts, and discard its supporters after a mistake.

Proposition 30 (Transductive mistake bound). Assume $d < \infty$ and $k \geq 2$ . For every pool S of size n there is a deterministic online bandit learner such that, for every $h ^ { \star } \in { \mathcal { H } }$ with $x _ { 1 } , \ldots , x _ { n } \in \mathrm { s u p p } ( h ^ { \star } )$ and $A _ { i } = h ^ { \star } ( x _ { i } )$ , the number of mistakes is at most

$$
\left\lceil \frac { \log | F _ { S } | } { - \log ( 1 - 1 / k ) } \right\rceil \leq \lceil k M _ { d } ( n ) \log ( 1 + n q k ) \rceil .\tag{32}
$$

Proof. The learner keeps a set of surviving experts, initially $F _ { S }$ . At each round it predicts a label recommended by the largest number of surviving experts. After a negative answer it eliminates every surviving expert f with $f ( i ) = { \hat { y } } _ { i }$ , and after a positive answer it changes nothing. Since $F _ { S }$ is a cover, some $f \in F _ { S }$ satisfies $f ( i ) \in h ^ { \star } ( x _ { i } )$ for every $i ,$ and this expert is never eliminated. On a mistake, all the surviving experts that recommended $\hat { y } _ { i }$ are eliminated, and since $\hat { y } _ { i }$ is a plurality label among k labels, they form at least a $1 / k$ fraction of the survivors. Thus after r mistakes we have $1 \leq | F _ { S } | ( 1 - 1 / k ) ^ { r }$ . Taking logarithms gives the first inequality in (32), and the second follows from (21) and $- \log ( 1 - 1 / k ) \geq 1 / k$ □

Without realizability, the same cover gives a regret bound through the EXP4 algorithm of [ACBFS02].

Remark 31 (Transductive regret). For any pool $S = ( x _ { 1 } , \ldots , x _ { n } ) $ and nonempty finite cover $F _ { S }$ of H on S, there is a randomized online bandit learner such that, for every sequence $A _ { 1 } , \ldots , A _ { n } \in { \mathcal { A } }$ fixed independently of its randomness,

$$
\mathbb { E } \left[ \sum _ { i = 1 } ^ { n } \mathbb { 1 } \{ { \hat { y } } _ { i } \notin A _ { i } \} \right] \leq \operatorname* { i n f } _ { h \in \mathcal { H } } \sum _ { i = 1 } ^ { n } \mathbb { 1 } \{ h ( x _ { i } ) \neq A _ { i } \} + C { \sqrt { k n \log | F _ { S } | } } ,\tag{33}
$$

where $C$ is universal. Indeed, EXP4 run with the members of $F _ { S }$ and the uniform expert as experts and with rewards 1 $\{ y \in A _ { i } \}$ has regret at most $C \sqrt { k n \log | F _ { S } | }$ against the best member of $F _ { S } \ [ \mathrm { A C B F S 0 2 }$ , Theorem 7.1] (when $| F _ { S } | = 1$ , the learner simply follows its only member). Since $F _ { S }$ is a cover, every $h \in \mathcal H$ has a member $f _ { h }$ with 1 $\{ f _ { h } ( i ) \notin A _ { i } \} \le \mathbb { 1 } \{ h ( x _ { i } ) \neq $ $A _ { i } \}$ for every $i \in [ n ]$ , as in the proof of Theorem 27, so the regret against $f _ { h }$ bounds the regret against $h$

## 8.2 Inductive Bounds

The bounds above assume that the whole pool, including the test input, is known in advance: the experts depend on it, and in Proposition 30 so do the queries. In the inductive problem the test input arrives only after training, so the queries can no longer be chosen by the experts. Our learners therefore query an independent uniform label at each training input and record the answers. Once the test input arrives, they build the cover of the whole pool, replay the recorded answers in a uniformly random order, and predict when the test input is reached. As in the proof of Theorem 24, the random order reduces the test loss to the average loss of a complete run.

A uniform query u at an input with correct-label set A exposes an expert f if u is the label of $f$ at this input and $u \not \in A$ . This happens with probability $1 / k$ if $f$ is wrong at this input, and never if $f$ is right. In the realizable case we eliminate the exposed experts. Each input then removes, in expectation, a $1 / k$ fraction of the surviving experts that are wrong there, while an expert that is right at every input is never removed, so the expected number of mistakes along a run is at most k log $| F |$ . In the agnostic case, k times the indicator of exposure is an unbiased estimate of the loss of $f ,$ and we run exponential weights with these estimates.

Theorem 32 (Inductive bandit learners). Suppose that for each n and each pool S of size n there is a finite cover $F _ { S } \subseteq { \mathcal { V } } ^ { n }$ of H on S that depends only on S and satisfies $| F _ { S } | \le L _ { n }$ . Let $n = m + 1$ . There are randomized bandit learners with m examples, whose training queries are independent, uniform on $\mathcal { V }$ , and independent of the training inputs and of the test input, such that the returned predictor $\widehat g$ satisfies

$$
\mathbb { E } \left[ L _ { P , h ^ { \star } } ( \widehat { g } ) \right] \leq \operatorname* { m i n } \left\{ 1 , \frac { k \log L _ { n } } { n } \right\}
$$

$$
( r e a l i z a b l e ) ,\tag{34}
$$

$$
\mathbb { E } \big [ L _ { \mathcal { D } } ( \widehat { g } ) \big ] \leq L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } ) + \sqrt { \frac { 2 k \log L _ { n } } { n } }
$$

$$
( a g n o s t i c ) ,\tag{35}
$$

$f o r$ every distribution P and $h ^ { \star } \in { \mathcal { H } }$ with $P ( \operatorname { s u p p } ( h ^ { \star } ) ) = 1$ , and for every distribution $\mathcal { D }$ on $\mathcal { X } \times \mathcal { A }$ , respectively. The expectations are over the m examples, drawn from $\mathcal { D } _ { P , h }$ ⋆ and from $\mathcal { D } _ { \perp }$ , and over the learner’s seed.

Proof. Throughout, $A _ { i }$ denotes the correct-label set of $x _ { i } ,$ which the learner never sees.

The learners. On each training input $x _ { i } , i \in [ m ]$ , the learner draws $u _ { i } \sim \mathrm { U n i f } ( \mathcal { V } )$ and observes $b _ { i } = \mathbb { 1 } \{ u _ { i } \in A _ { i } \}$ . Given a test input x, it puts $x _ { n } = x$ , draws a uniformly random permutation π of $[ n ]$ , and lists the inputs in this order, ${ z } _ { t } = { x } _ { \pi ( t ) }$ for $t \in [ n ]$ . Let $\tau = \pi ^ { - 1 } ( n )$ be the rank of the test input, so that $z _ { \tau } ~ = ~ x$ , let $F = F _ { ( z _ { 1 } , \ldots , z _ { n } ) }$ , and for $t \neq \tau$ write $\left( \tilde { u } _ { t } , \tilde { b } _ { t } \right) = \left( u _ { \pi ( t ) } , b _ { \pi ( t ) } \right)$ . We say that $f \in F$ is exposed at rank t if $\tilde { u } _ { t } = f ( t )$ and $\tilde { b } _ { t } = 0$ . Both learners process the ranks $t = 1 , \ldots , \tau - 1$ in this order and then predict at rank τ.

In the realizable case, let $V _ { 1 } = F$ , and obtain $V _ { t + 1 }$ by removing from $V _ { t }$ every expert exposed at rank t. At rank $\tau ,$ draw f uniformly from $V _ { \tau }$ and output $f ( \tau )$

In the agnostic case, let $w _ { 1 } ( f ) = 1$ for $f \in F$ , and at rank t update

$$
w _ { t + 1 } ( f ) = w _ { t } ( f ) \exp \bigl ( - \eta \hat { \lambda } _ { t } ( f ) \bigr ) , \qquad \hat { \lambda } _ { t } ( f ) = k \mathbb { 1 } \{ \tilde { u } _ { t } = f ( t ) , \tilde { b } _ { t } = 0 \} ,
$$

so that $\hat { \lambda } _ { t } ( f )$ is k times the indicator that $f$ is exposed at rank t, with $\eta = \sqrt { 2 \log L _ { n } / ( k n ) }$ (if $L _ { n } = 1$ , the family has a single member and the learner outputs its label). At rank $\tau .$ , draw f with probability proportional to $w _ { \tau } ( f )$ and output $f ( \tau )$

Reduction to a complete run. We claim that the expected test loss equals the average, over $t \in [ n ]$ , of the expected loss at rank t of the rule run through all n ranks, where the loss at rank t is computed before the update at rank t. To see this, draw the n examples, training and test, together with independent uniform queries $u _ { 1 } , \ldots , u _ { n } ;$ the query $u _ { n }$ is never used by the learner, and we define $( \tilde { u } _ { \tau } , \tilde { b } _ { \tau } )$ as for the other ranks. The triples $( x _ { i } , A _ { i } , u _ { i } )$ are i.i.d., so the permuted triples are again i.i.d. and independent of $\pi _ { : }$ , and in particular of the rank $\tau$ which is uniform on $[ n ]$ . Up to rank $\tau - 1$ the learner coincides with the complete run, which does not depend on $\tau .$ , and averaging over $\tau$ proves the claim. Below we write $A _ { t } ^ { \prime }$ for the correct-label set of $z _ { t }$ and argue for a fixed sequence $z _ { 1 } , \ldots , z _ { n } , A _ { 1 } ^ { \prime } , \ldots , A _ { n } ^ { \prime }$ , averaging over it only at the end. Then $F$ is fixed, and the queries $\tilde { u } _ { 1 } , \ldots , \tilde { u } _ { n }$ are independent and uniform on $\mathcal { V }$

The realizable bound. Here $A _ { t } ^ { \prime } = h ^ { \star } ( z _ { t } )$ for every $t ,$ since $P ( \operatorname { s u p p } ( h ^ { \star } ) ) = 1$ . Since $F$ is a cover, some expert $f$ satisfies $f ( t ) \in A _ { t } ^ { \prime }$ at every rank, and it is never exposed, because $\tilde { u } _ { t } = f ( t )$ forces $\tilde { b } _ { t } = 1$ . Thus every $V _ { t }$ is nonempty. Let

$$
\rho _ { t } = { \frac { 1 } { | V _ { t } | } } \sum _ { f \in V _ { t } } \mathbb { 1 } \{ f ( t ) \notin A _ { t } ^ { \prime } \}
$$

be the fraction of $V _ { t }$ that is wrong at rank $t ;$ given $V _ { t } .$ the rule makes a mistake at rank t with probability exactly $\rho _ { t }$ . An expert in $V _ { t }$ is exposed at rank t exactly when it is wrong there and its label equals $\tilde { u } _ { t }$ . Since $\tilde { u } _ { t }$ is uniform and independent of $V _ { t }$ , the expected fraction of $V _ { t }$ eliminated at rank t is $\rho _ { t } / k$ . Applying $- \log ( 1 - s ) \geq s$ to $s = 1 - | V _ { t + 1 } | / | V _ { t } | < 1$ and taking conditional expectations gives

$$
\mathbb { E } \big [ \log | V _ { t } | - \log | V _ { t + 1 } | \bigm | V _ { t } \big ] \geq \frac { \rho _ { t } } { k } .
$$

Summing over $t ,$ the left sides telescope, and since $| V _ { 1 } | \leq L _ { n }$ and $| V _ { n + 1 } | \geq 1$ 2

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { n } \rho _ { t } \right] \leq k \log L _ { n } .\tag{36}
$$

By the reduction, the expected test loss is $\mathbb { E } [ { \frac { 1 } { n } } \sum _ { t } \rho _ { t } ]$ , and (36) proves (34).

The agnostic bound. Write $\lambda _ { t } ( f ) = \mathbb { 1 } \{ f ( t ) \notin A _ { t } ^ { \prime } \}$ for the loss of $f$ at rank $t ,$ and let $p _ { t } ( f ) = w _ { t } ( f ) / W _ { t }$ with $\begin{array} { r } { W _ { t } = \sum _ { f \in F } w _ { t } ( f ) } \end{array}$ . The weights $p _ { t }$ are determined by the past, that is, by $\tilde { u } _ { 1 } , \ldots , \tilde { u } _ { t - 1 }$ , and given the past the rule makes a mistake at rank t with probability $\sum _ { f } p _ { t } ( f ) \lambda _ { t } ( f )$ . Since $\tilde { b } _ { t } = 0$ exactly when $\tilde { u } _ { t } \notin A _ { t } ^ { \prime } .$ , and $\tilde { u } _ { t }$ is uniform and independent of the past,

$$
\mathbb { E } \big [ \hat { \lambda } _ { t } ( f ) \big | \mathrm { \ p a s t } \big ] = \lambda _ { t } ( f ) , \qquad \mathbb { E } \big [ \hat { \lambda } _ { t } ( f ) ^ { 2 } \big | \mathrm { \ p a s t } \big ] = k \lambda _ { t } ( f ) \le k .\tag{37}
$$

Using $e ^ { - x } \leq 1 - x + x ^ { 2 } / 2$ for $x \geq 0$ and then lo $\operatorname { g } ( 1 + y ) \leq y$ , we get

$$
\log \frac { W _ { t + 1 } } { W _ { t } } = \log \sum _ { f } p _ { t } ( f ) e ^ { - \eta \hat { \lambda } _ { t } ( f ) } \leq - \eta \sum _ { f } p _ { t } ( f ) \hat { \lambda } _ { t } ( f ) + \frac { \eta ^ { 2 } } { 2 } \sum _ { f } p _ { t } ( f ) \hat { \lambda } _ { t } ( f ) ^ { 2 } .
$$

On the other hand, log $\begin{array} { r } { \varsigma ( W _ { n + 1 } / W _ { 1 } ) \geq - \eta \sum _ { t } \hat { \lambda } _ { t } ( f ) - \log | F | } \end{array}$ for each fixed $f \in F$ . We sum the upper bound over t and compare it with this lower bound. Because p is determined by the past, (37) gives $\begin{array} { r } { \mathbb { E } [ \sum _ { f } p _ { t } ( f ) \hat { \lambda } _ { t } ( f ) ] = \mathbb { E } [ \sum _ { f } p _ { t } ( f ) \lambda _ { t } ( f ) ] } \end{array}$ and $\mathbb { E } [ \sum _ { f } p _ { t } ( \bar { f } ) \hat { \lambda } _ { t } ( f ) ^ { 2 } ] \leq k$ , so taking expectations and using the choice of η we obtain

$$
\mathbb { E } \left[ \sum _ { t = 1 } ^ { n } \sum _ { f ^ { \prime } \in F } p _ { t } ( f ^ { \prime } ) \lambda _ { t } ( f ^ { \prime } ) \right] \leq \sum _ { t = 1 } ^ { n } \lambda _ { t } ( f ) + \frac { \log L _ { n } } { \eta } + \frac { \eta k n } { 2 } = \sum _ { t = 1 } ^ { n } \lambda _ { t } ( f ) + \sqrt { 2 k n \log L _ { n } } .\tag{38}
$$

Now fix $h \in { \mathcal { H } }$ , and let $f _ { h } \in F$ satisfy $f _ { h } ( t ) \in h ( z _ { t } )$ at every rank t with $z _ { t } \in \operatorname { s u p p } ( h )$ . As in the proof of Theorem 27, $\lambda _ { t } ( f _ { h } ) \leq \mathbb { 1 } \{ h ( z _ { t } ) \neq A _ { t } ^ { \prime } \}$ for every t. By the reduction, dividing (38) with $f = f _ { h }$ by n and averaging over the i.i.d. pool gives

$$
\mathbb { E } [ L _ { \mathcal { D } } ( \widehat { g } ) ] \leq \operatorname* { P r } _ { ( x , A ) \sim \mathcal { D } } [ h ( x ) \neq A ] + \sqrt { \frac { 2 k \log L _ { n } } { n } } .
$$

Taking the infimum over $h \in \mathcal H$ proves (35).

With the covers of Lemma 21, this gives the rates of Theorem 29.

Corollary 33 (Bandit rates). If $d = d _ { \mathrm { N } } ^ { \cap } ( \mathcal { H } ) < \infty$ , then the bounds (31) of Theorem 29 hold. For fixed k, ℓ and d, where ℓ bounds the sizes of the correct-label sets, both excess terms tend to zero as $n \to \infty$

Proof. Apply Theorem 32 with the covers of Lemma 21 and $L _ { n } = ( 1 + n q k ) ^ { M _ { d } ( n ) }$ , which is allowed by (21). By (19), $M _ { d } ( n ) = O ( \log ( n + 1 ) )$ ) for fixed k, ℓ and $d ,$ so the two excess terms are $O ( \log ^ { 2 } ( n + 1 ) / n )$ and $O ( \log ( n + 1 ) / \sqrt { n } )$ □

## 8.3 Confidence Amplification

We now turn the bounds on the expected risk into PAC bounds by repetition and validation, as in Proposition 28 for full sets: we run the learner on several independent blocks of examples and select one of the resulting candidates on a fresh validation block. Bandit feedback sufices for the selection. For candidates produced by any algorithm, one uniform query per validation example gives an unbiased estimate of the risk of every candidate at once, at the price of a factor k in the variance.

Lemma 34 (Validation with uniform queries). Let $s , v \geq 1$ be integers and $\varepsilon , \delta \in ( 0 , 1 )$ . Let $g _ { 1 } , \ldots , g _ { s }$ be predictors, possibly random, that are independent of the validation examples $( x _ { i } , A _ { i } ) \sim \mathcal { D } , i \in [ v ]$ . At each validation example query an independent $u _ { i } \sim \mathrm { U n i f } ( \mathcal { V } )$ and observe $b _ { i } = \mathbb { 1 } \{ u _ { i } \in A _ { i } \}$ . Define

$$
\widehat { L } _ { j } = \frac { k } { v } \sum _ { i = 1 } ^ { v } \mathbb { 1 } \{ u _ { i } = g _ { j } ( x _ { i } ) , \ b _ { i } = 0 \} \qquad ( j \in [ s ] ) .\tag{39}
$$

Then $\mathbb { E } [ \widehat { L } _ { j } \mid g _ { 1 } , \dots , g _ { s } ] = L _ { \mathcal { D } } ( g _ { j } )$ , and there is an absolute constant c with the following two properties. With probability at least $1 - \delta$ ，

$$
\operatorname* { m a x } _ { j \leq s } \lvert \widehat { L } _ { j } - L _ { \mathcal { D } } ( g _ { j } ) \rvert \leq c \left( \sqrt { \frac { k \log ( 2 s / \delta ) } { v } } + \frac { k \log ( 2 s / \delta ) } { v } \right) .\tag{40}
$$

Moreover, $i f v \ge c ( k / \varepsilon ) \log ( 2 s / \delta )$ , then with probability at least $1 - \delta$ every j with $L _ { \mathcal { D } } ( g _ { j } ) \le \varepsilon / 4$ has $\widehat { L } _ { j } < \varepsilon / 2$ , and every j with $L _ { \mathcal { D } } ( g _ { j } ) \ge \varepsilon$ has $\widehat { L } _ { j } > \varepsilon / 2$

Proof. Condition on $g _ { 1 } , \ldots , g _ { s } ;$ since the claims hold for every value of the predictors, they also hold unconditionally. Fix j and put $Z _ { i , j } = 1 \{ u _ { i } = g _ { j } ( x _ { i } ) , b _ { i } = 0 \}$ . This is the indicator of the event that $g _ { j } ( x _ { i } ) \notin A _ { i }$ and $u _ { i } = g _ { j } ( x _ { i } )$ , so $Z _ { 1 , j } , \dotsc , Z _ { v , j }$ are independent Bernoulli variables with mean $L _ { \mathcal { D } } ( g _ { j } ) / k$ . Hence $k Z _ { i , j }$ has mean $L _ { \mathcal { D } } ( g _ { j } )$ , lies in $[ 0 , k ]$ , and has variance at most $k L _ { \mathcal { D } } ( g _ { j } ) \le k$ , and Bernstein’s inequality with a union bound over the s candidates gives (40). For the threshold test, note that the threshold $\varepsilon / ( 2 k )$ for the average $\textstyle { \frac { 1 } { v } } \sum _ { i } Z _ { i , j }$ is at least twice its mean when $L _ { \mathcal { D } } ( g _ { j } ) \le \varepsilon / 4$ , and at most half its mean when $L _ { \mathcal { D } } ( g _ { j } ) \ge \varepsilon$ . In both cases the multiplicative Chernof bound shows that the average falls on the wrong side of the threshold with probability at most $\exp ( - v \varepsilon / ( c k ) )$ , after enlarging c if necessary, and a union bound over the s candidates finishes the proof. □

Proposition 35 (Confidence amplification). Let $m _ { \mathrm { B } }$ be any function such that, for every $r > 0$ , some bandit learner with $m _ { \mathrm { B } } ( r )$ examples has expected risk at most r in the realizable case, or at most $L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } ) + r$ in the agnostic case. Then, respectively,

$$
m _ { \mathcal { H } } ^ { \mathrm { b a n d } } ( \varepsilon , \delta ) = O \left( m _ { \mathrm { B } } ( \varepsilon / 1 6 ) \log \frac 2 \delta + \frac { k } { \varepsilon } \log \frac { 2 + \log ( 2 / \delta ) } { \delta } \right) ,\tag{41}
$$

$$
m _ { \mathcal { H } } ^ { \mathrm { b a n d , a g } } ( \varepsilon , \delta ) = O \left( \frac { m _ { \mathrm { B } } ( \varepsilon / 8 ) } { \varepsilon } \log \frac { 2 } { \delta } + \frac { k } { \varepsilon ^ { 2 } } \log \frac { 2 + \varepsilon ^ { - 1 } \log ( 2 / \delta ) } { \delta } \right) ,\tag{42}
$$

where the constants in $O ( \cdot )$ are universal.

Proof. In both cases the learner runs the given learner on s disjoint blocks of $m _ { \mathrm { B } } ( \cdot )$ examples and selects one of the resulting candidates $g _ { 1 } , \ldots , g _ { s }$ with Lemma 34, applied with failure probability $\delta / 2$ on one further block of v examples. The total number of examples is $s m _ { \mathrm { B } } ( \cdot ) + v$

Realizable case. By Markov’s inequality, a run with expected risk at most $\varepsilon / 1 6$ has risk at most $\varepsilon / 4$ with probability at least $3 / 4$ . With $s = \lceil \log _ { 4 } ( 2 / \delta ) \rceil$ independent runs, some candidate has risk at most $\varepsilon / 4$ with probability at least $1 - \delta / 2$ . Take $v = O ( ( k / \varepsilon ) \log ( 4 s / \delta ) )$ as in the threshold test of Lemma 34, and output a candidate with $\widehat { L } _ { j } < \varepsilon / 2$ . On both events such a candidate exists, and every such candidate has risk less than ε.

Agnostic case. Put $a = L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } ) . \mathrm { I f } \ a + \varepsilon \geq 1$ there is nothing to prove, so assume $a + \varepsilon < 1$ By Markov’s inequality, a run with expected risk at most $a + \varepsilon / 8$ has risk at most $a + \varepsilon / 2$ with probability at least $1 - ( a + \varepsilon / 8 ) / ( a + \varepsilon / 2 ) = ( 3 \varepsilon / 8 ) / ( a + \varepsilon / 2 ) \geq 3 \varepsilon / 8$ , as in the proof of Proposition 28. With $\begin{array} { r } { s = \lceil \frac { 8 } { 3 \varepsilon } \log \frac { 2 } { \delta } \rceil } \end{array}$ independent runs, some candidate $g _ { j } .$ ⋆ has risk at most $a + \varepsilon / 2$ with probability at least $1 - \delta / 2$ . Take $v = O ( ( k / \varepsilon ^ { 2 } )$ log $( 4 s / \delta ) )$ , so that (40) holds with accuracy $\varepsilon / 4$ , and output a candidate $g _ { \hat { j } }$ minimizing $\widehat { L } _ { j }$ . On both events,

$$
L _ { \mathcal { D } } ( g _ { \widehat { j } } ) \leq \widehat { L } _ { \widehat { j } } + \frac { \varepsilon } { 4 } \leq \widehat { L } _ { j ^ { \star } } + \frac { \varepsilon } { 4 } \leq L _ { \mathcal { D } } ( g _ { j ^ { \star } } ) + \frac { \varepsilon } { 2 } \leq a + \varepsilon .
$$

In both cases $\log ( 4 s / \delta )$ is at most a constant multiple of the logarithm in (41) and (42), respectively, which proves the two bounds. □

The training term in (41) depends multiplicatively on $\log ( 2 / \delta )$ , because the given learner is run on $s = \lceil \log _ { 4 } ( 2 / \delta ) \rceil$ blocks. In contrast, the realizable full-set bound of Theorem 25 depends additively on $\log ( 1 / \delta )$ , and so does elimination with uniform queries on a known pool. To see this, fix a pool $( x _ { 1 } , \ldots , x _ { N } )$ with correct-label sets $A _ { 1 } , \ldots , A _ { N }$ and a family $F \subseteq \mathcal { V } ^ { N }$ of experts. Let I be a uniformly random subset of [N] of size m, query an independent $u _ { i } \sim \mathrm { U n i f } ( \mathcal { V } )$ at each $i \in I$ , and observe $b _ { i } = \mathbb { 1 } \{ u _ { i } \in A _ { i } \}$ . Call $f \in F$ a survivor if no $i \in I$ has $f ( i ) = u _ { i }$ and $b _ { i } = 0$ , that is, if no query exposes $f .$

Proposition 36 (Survivors on a fixed pool). In the setting above, let $1 \leq m \leq N$ and $\delta \in ( 0 , 1 )$ , suppose that $A _ { i } = h ^ { \star } ( x _ { i } )$ for all i, where $h ^ { \star } \in { \mathcal { H } }$ satisfies $x _ { 1 } , \ldots , x _ { N } \in \mathrm { s u p p } ( h ^ { \star } )$ and suppose that F contains an expert $f ^ { \star }$ with $f ^ { \star } ( i ) \in A _ { i }$ for all i. Then $f ^ { \star }$ survives, and with probability at least $1 - \delta$ every survivor $f$ satisfies

$$
{ \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } \mathbb { 1 } \{ f ( i ) \notin A _ { i } \} \leq { \frac { k } { m } } \left( \log | F | + \log { \frac { 1 } { \delta } } \right) .\tag{43}
$$

Proof. If $f ( i ) = u _ { i }$ and $f ( i ) \in A _ { i }$ , then $b _ { i } = 1$ ; hence $f ^ { \star }$ survives. Fix $f \in F$ , let $p$ be the left side of (43), and put $a _ { i } = 1 - 1 / k { \mathrm { ~ i f ~ } } f ( i ) \notin A _ { i }$ and $a _ { i } = 1$ otherwise, so that the average of $a _ { 1 } , \ldots , a _ { N }$ is $1 - p / k$ . Given I, the expert $f$ survives exactly when $u _ { i } \neq f ( i )$ at every $i \in I$ with $f ( i ) \not \in A _ { i }$ , which has probability $\Pi _ { i \in I } a _ { i }$ . By Maclaurin’s inequality, the average of the products of m distinct entries of a list of nonnegative numbers is at most the mth power of their average, and therefore

$$
\operatorname* { P r } [ f { \mathrm { ~ s u r v i v e s } } ] = { \binom { N } { m } } ^ { - 1 } \sum _ { | I | = m } \prod _ { i \in I } a _ { i } \leq \left( { \frac { 1 } { N } } \sum _ { i = 1 } ^ { N } a _ { i } \right) ^ { m } = ( 1 - p / k ) ^ { m } \leq e ^ { - m p / k } .
$$

Thus every $f$ with $\begin{array} { r } { p > \frac { k } { m } ( \log | F | + \log \frac { 1 } { \delta } ) } \end{array}$ survives with probability less than $\delta / | F |$ , and a union bound over F proves the claim. □

We leave open whether an inductive bandit learner can achieve an additive dependence on $\log ( 1 / \delta )$ in the realizable case. Even with singleton correct-label sets, the confidence parameter has nontrivial efects in bandit PAC learning $\mathrm { [ Z Y W 2 6 ] }$

## 8.4 Proof of Theorem 29

Proof of Theorem 29. By Theorem 16, $( \mathrm { i } ) { - } ( \mathrm { i } \mathrm { v } )$ are equivalent. We show $( \mathrm { i i } ) { \Rightarrow } ( \mathrm { v i } ) { \Rightarrow } ( \mathrm { v } ) { \Rightarrow } ( \mathrm { i } )$

Assume (ii), and let $d = d _ { \mathrm { N } } ^ { \Omega } ( \mathcal { H } )$ . By Corollary 33, the bounds (31) hold, and their excess terms tend to zero, so $m _ { \mathrm { B } } ( r )$ is finite for every $r > 0$ . Proposition 35 then gives (41) and (42), and in particular (vi).

As noted in Section 5.2, agnostic PAC learnability implies realizable PAC learnability under every feedback model, so (vi) implies (v). Finally, a learner with bandit feedback sees less than a full-set learner, and Proposition 22 covers bandit learners explicitly; hence the proof of $( \mathrm { i i i } ) { \Rightarrow } ( \mathrm { i } )$ in Theorem 16 shows that (v) implies (i). □

## 9 A Reduction between Full-Set and One-Correct-Label Feedback

In this section we relate one-correct-label learning of a class H to full-set learning of its partial intersection class $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ , defined in (7). One direction follows from a union bound: a correct label of $\begin{array} { r } { g = \bigcap _ { a = 1 } ^ { r } h _ { a } } \end{array}$ at x is a valid demonstration for every $h _ { a } .$ , so a one-correct-label learner for H also learns the intersection class from full sets (Lemma 37).

The converse occupies the rest of the section. Theorem 39 turns finite families of singlelabel predictions with an intersection property into a one-correct-label learner with small expected loss, and Lemma 40 turns this into a high-probability guarantee. The families themselves are constructed for finite label spaces in Section 10 and for countable label spaces in Section 13.

Lemma 37 (Common demonstrations). For every integer $R \geq 1$ and $\varepsilon , \delta \in ( 0 , 1 )$

$$
m _ { \mathcal { T } _ { R } ( \mathcal { H } ) } ^ { \mathrm { f u l l } } ( \varepsilon , \delta ) \leq m _ { \mathcal { H } } ^ { \mathrm { o n e } } ( \varepsilon / R , \delta / R ) .
$$

Proof. Let $m = m _ { \mathcal { H } } ^ { \mathrm { o n e } } ( \varepsilon / R , \delta / R )$ ; if $m = \infty$ there is nothing to prove. Fix a one-correct-label learner for H that uses m examples and has accuracy $\varepsilon / R$ and confidence $1 - \delta / R$ . Our full-set learner receives $( x _ { 1 } , A _ { 1 } ) , \ldots , ( x _ { m } , A _ { m } )$ , replaces each $A _ { i }$ by its first element $y _ { i }$ in the fixed enumeration of Y, and runs the one-correct-label learner on $( x _ { 1 } , y _ { 1 } ) , \dots , ( x _ { m } , y _ { m } )$

Let $g \in \mathbb { Z } _ { R } ( \mathcal { H } )$ and let P be a distribution with $P ( \operatorname { s u p p } ( g ) ) = 1$ . By $\begin{array} { r } { ( \overleftrightarrow { \boldsymbol { \mathrm { \Lambda } } } ) , g ( x ) = \bigcap _ { a = 1 } ^ { r } h _ { a } ( x ) } \end{array}$ for every $x \in$ supp(g), where $r \leq R , h _ { 1 } , \ldots , h _ { r } \in { \mathcal { H } }$ and sup $\begin{array} { r } { \mathfrak { d } \big ( g \big ) \subseteq \bigcap _ { a = 1 } ^ { r } \operatorname { s u p p } ( h _ { a } ) } \end{array}$ . The pairs $( x _ { i } , y _ { i } )$ are i.i.d. draws from the law D of $( x , y ( x ) )$ , where $x \sim P$ and $y ( x )$ is the first element of $g ( x )$ . This law has marginal $P _ { \mathrm { : } }$ , and for every $a \in [ r ]$ we have $P ( \operatorname { s u p p } ( h _ { a } ) ) = 1$ and $y ( x ) \in g ( x ) \subseteq h _ { a } ( x )$ for every $x \in \operatorname { s u p p } ( g )$ . Thus D is a valid one-correct-label law for each of the targets $h _ { 1 } , \ldots , h _ { r } ,$ so by a union bound over $a \in [ r ]$ , with probability at least $1 - r \delta / R \geq 1 - \delta$ the output $\widehat { f }$ satisfies $L _ { P , h _ { a } } ( \widehat { f } ) \leq \varepsilon / R$ for every a. Since ${ \widehat { f } } ( x ) \not \in g ( x )$ implies

${ \widehat { f } } ( x ) \not \in h _ { a } ( x )$ for some a, we get

$$
L _ { P , g } ( \widehat { f } ) \leq \sum _ { a = 1 } ^ { r } L _ { P , h _ { a } } ( \widehat { f } ) \leq \frac { r \varepsilon } { R } \leq \varepsilon .
$$

We now turn to the converse. A demonstrated label $y \in h ^ { \star } ( x )$ reveals nothing about the other labels of $h ^ { \star } ( x )$ , so consistency with the demonstrations does not force good predictions. Our learner works with a pool S of N inputs, the training inputs together with the test input, and uses the demonstrations at a random half of the positions. With each candidate h it associates a small family $F _ { h , S }$ of single-label predictions on the pool. Among hypotheses consistent with the demonstrations, it retains h only if every member of $F _ { h , S }$ that is correct for h on the labeled half has small loss against h on the whole pool. A member with large loss is rarely correct on a random half, so a union bound over $F _ { h ^ { \star } , S }$ shows that the target is retained with high probability.

The families must have an intersection property: any $\ell + 1$ candidates share a member that predicts a common label wherever their correct-label sets intersect. For $\ell + 1$ retained candidates, the shared member is correct for all of them on the labeled half and hence has small loss against each on the whole pool. Their correct-label sets therefore intersect at all but a small fraction of the unlabeled positions. A minimax argument on these positions turns this into randomized predictions with small average loss against every retained candidate, and in particular against the target.

For a multiset $S = ( x _ { 1 } , \dots , x _ { N } )$ of inputs we write supp $( S ) = \{ x _ { 1 } , \ldots , x _ { N } \}$ for the set of its distinct elements, and we say that a hypothesis h is defined on S if $\operatorname { s u p p } ( S ) \subseteq \operatorname { s u p p } ( h )$

Definition 38 (Prediction families with the intersection property). Fix $N \geq 2$ . Suppose that for every multiset S of N inputs and every $h \in \mathcal H$ defined on S we are given a family

$$
F _ { h , S } \subseteq \{ f : \operatorname { s u p p } ( S ) \to y \} , \qquad | F _ { h , S } | \leq B _ { N } ,
$$

where $B _ { N } \geq 1$ does not depend on S and h. These families have the intersection property if for every $S$ and all $h _ { 1 } , \ldots , h _ { \ell + 1 } \in \mathcal { H }$ defined on S there is a function $f \in \bigcap _ { a = 1 } ^ { \ell + 1 } F _ { h _ { a } , S }$ such that

$$
\bigcap _ { a = 1 } ^ { \ell + 1 } h _ { a } ( x ) \neq \varnothing \quad \Longrightarrow \quad f ( x ) \in \bigcap _ { a = 1 } ^ { \ell + 1 } h _ { a } ( x ) \qquad ( x \in \operatorname { s u p p } ( S ) ) .\tag{44}
$$

The main step of the converse is the following theorem.

Theorem 39 (Conversion from prediction families). Let H be a nonempty class and $N \geq$ 2, and suppose that families $F _ { h , S }$ with size bound $B _ { N }$ have the intersection property of Definition 38. Then there is a one-correct-label learner that uses $N - 1$ examples and outputs a randomized prediction rule pb with the following property. For every $h ^ { \star } \in { \mathcal { H } }$ , every distribution P on X with $P ( \operatorname { s u p p } ( h ^ { \star } ) ) = 1$ , and every law D on $\mathcal { X } \times \mathcal { V }$ with marginal P and

Pr<sub>(x,y)∼D</sub>[y ∈ h<sup>⋆</sup>(x)] = 1,

$$
\mathbb { E } \big [ L _ { P , h ^ { \star } } ( \widehat { p } ) \big ] \leq \operatorname* { m i n } \left\{ 1 , \ 2 \left( \frac { ( \ell + 1 ) ( \log B _ { N } + \log N ) } { \lfloor N / 2 \rfloor } \right) ^ { 1 / \ell } + \frac { 1 } { N } \right\} ,\tag{45}
$$

where the expectation is over the training examples drawn from D and the seed of the learner.

Proof. We first define the retained class and the learner, and then analyze an experiment on N examples in which the test position is uniform among the unlabeled ones. Let

$$
t = \lfloor N / 2 \rfloor , \qquad \alpha = \frac { \log B _ { N } + \log N } { t } , \qquad \beta = \left( \frac { N ( \ell + 1 ) \alpha } { N - t } \right) ^ { 1 / \ell } .
$$

Here t is the number of labeled positions, α is chosen so that $B _ { N } e ^ { - t \alpha } = 1 / N$ , and $\beta$ is chosen so that $\beta ^ { \ell }$ is the bound in (50) below. For a multiset ${ \cal S } = ( x _ { 1 } , \dots , x _ { N } )$ , a function $f : \operatorname { s u p p } ( S )  \mathcal { Y }$ and a hypothesis h defined on $S _ { ; }$ , let

$$
L _ { S } ( f , h ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { 1 } \{ f ( x _ { i } ) \notin h ( x _ { i } ) \} .
$$

The retained class. Fix a pool ${ \cal S } = ( x _ { 1 } , \dots , x _ { N } )$ , a set $J \subseteq [ N ]$ of t observed positions, and labels $( y _ { i } ) _ { i \in J }$ . Write $U = [ N ] \backslash J$ for the unobserved positions and $S _ { U } = \{ x _ { i } : i \in U \}$ for their inputs. Let $\mathcal { H } _ { J }$ be the set of $h \in { \mathcal { H } }$ that are defined on $S _ { \mathrm { { ; } } }$ satisfy $y _ { i } \in h ( x _ { i } )$ for every $i \in J$ , and satisfy

$$
{ \mathrm { ~ f o r ~ e v e r y ~ } } f \in F _ { h , S } : \qquad f ( x _ { i } ) \in h ( x _ { i } ) { \mathrm { ~ f o r ~ a l l ~ } } i \in J \quad \Longrightarrow \quad L _ { S } ( f , h ) \leq \alpha .\tag{46}
$$

The class $\mathcal { H } _ { J }$ depends only on the multiset $S$ and the multiset of observed examples $\{ ( x _ { i } , y _ { i } )$ $i \in J \}$ . We show below that if $\mathcal { H } _ { J } \neq \emptyset$ , there are distributions $( p _ { z } ) _ { z \in S _ { U } }$ on $\mathcal { V }$ such that

$$
\frac { 1 } { | U | } \sum _ { i \in U } \bigl ( 1 - p _ { x _ { i } } ( h ( x _ { i } ) ) \bigr ) \le \beta \qquad \mathrm { f o r ~ e v e r y ~ } h \in \mathcal { H } _ { J } ,\tag{47}
$$

where $\begin{array} { r } { p _ { z } ( A ) = \sum _ { y \in A } p _ { z } ( y ) } \end{array}$ . Fix a rule that selects such distributions as a function of $S$ and the multiset of observed examples (when $\mathcal { H } _ { J } = \emptyset$ , every $p _ { z }$ is a point mass on the first label of Y); since $\mathcal { X }$ and $\mathcal { V }$ are countable, this rule is measurable.

The learner. Algorithm 2 places the test input at the last position of the pool and uses the demonstrations at a random set of t training positions.

Since $J \subseteq [ N - 1 ]$ , the test position N lies in $U$ , so $p _ { x }$ is defined in step 5. Moreover, the seed determines J before the test input is seen, so for a fixed seed and fixed training examples the map $x \mapsto { \widehat { p } } ( \cdot \mid x )$ is a randomized prediction.

Algorithm 2: One-correct-label learner of Theorem 39   
Input: training examples $( x _ { 1 } , y _ { 1 } ) , \dotsc , ( x _ { N - 1 } , y _ { N - 1 } ) \dotsc$ a test input $x .$   
1. Set $x _ { N } = x$ and ${ \cal S } = ( x _ { 1 } , \dots , x _ { N } )$   
2. Using the seed, draw J uniformly among the t-element subsets of $[ N - 1 ]$ . Let   
$U = [ N ] \backslash J .$   
3. Form the retained class $\mathcal { H } _ { J }$ from $S$ and $( x _ { i } , y _ { i } ) _ { i \in J } ,$ as in (46).   
4. Select distributions $( p _ { z } ) _ { z \in S _ { U } }$ satisfying (47) by the fixed rule.   
5. Output ${ \widehat { p } } ( \cdot \mid x ) = p _ { x }$

Reduction to a transductive experiment. For the analysis, give the test input $x _ { N } \sim P$ a label $y _ { N }$ drawn from the conditional law of $\mathcal { D }$ given $x _ { N }$ . The learner does not use $y _ { N }$ , and the pairs $( x _ { 1 } , y _ { 1 } ) , \dotsc , ( x _ { N } , y _ { N } )$ are now i.i.d. from $\mathcal { D } ;$ almost surely, $h ^ { \star }$ is defined on S and $y _ { i } \in h ^ { \star } ( x _ { i } )$ for every $i \in [ N ]$ . The learner observes the positions in a uniform t-subset J of $[ N - 1 ]$ and is tested at position $N$ . For a set $J$ of t observed positions and a test position $I \notin J$ , the loss $1 - p _ { x _ { I } } ( h ^ { \star } ( x _ { I } ) )$ depends on the examples only through the multiset $S _ { i }$ the multiset $\{ ( x _ { i } , y _ { i } ) : i \in J \}$ and $x _ { I }$ . Since the N pairs are i.i.d., relabeling the positions does not change their joint law, so the expected loss is the same for every such pair $( J , I )$ . We may therefore draw $J$ uniformly among the t-subsets of $[ N ]$ , independently of the examples, and then I uniformly from $U _ { : }$ , without changing the expected loss. From now on, J is drawn in this way. Conditional on the examples and on $J ,$ the predictions $( p _ { z } ) _ { z \in S _ { U } }$ are fixed, and

$$
\mathbb { E } _ { I } \big [ 1 - p _ { x _ { I } } ( h ^ { \star } ( x _ { I } ) ) \big ] = \frac { 1 } { | U | } \sum _ { i \in U } \big ( 1 - p _ { x _ { i } } ( h ^ { \star } ( x _ { i } ) ) \big ) .\tag{48}
$$

It remains to bound this average.

The true target is retained. We first show that

$$
\operatorname* { P r } _ { J } \bigl [ h ^ { \star } \notin \mathcal { H } _ { J } \mid ( x _ { i } , y _ { i } ) _ { i = 1 } ^ { N } \bigr ] \leq \frac { 1 } { N } .\tag{49}
$$

Since all demonstrated labels are correct for $h ^ { \star }$ , the target is discarded only if some $f \in$ $F _ { h ^ { \star } , S }$ with $L _ { S } ( f , h ^ { \star } ) > \alpha$ is correct for $h ^ { \star }$ at every position of J. Fix such an $f ,$ and let $b = N L _ { S } ( f , h ^ { \star } )$ be the number of positions $i \in [ N ]$ with $f ( x _ { i } ) \not \in h ^ { \star } ( x _ { i } )$ . Since $J$ is a uniform t-subset of $[ N ]$

$$
\operatorname* { P r } _ { J } \big [ f ( x _ { i } ) \in h ^ { \star } ( x _ { i } ) \mathrm { ~ f o r ~ a l l ~ } i \in J \big ] = \frac { { \binom { N - b } { t } } } { { \binom { N } { t } } } \leq \Big ( 1 - \frac { b } { N } \Big ) ^ { t } \leq \exp \bigl ( - t L _ { S } ( f , h ^ { \star } ) \bigr ) < e ^ { - t \alpha } .
$$

The family $F _ { h ^ { \star } , S }$ is determined by S alone, independently of J and of the labels, and has at most $B _ { N }$ members. A union bound over it bounds the probability of discarding the target by $B _ { N } e ^ { - t \alpha } = 1 / N$ , which proves (49).

Empty intersections are rare on U. Fix the examples and $J ,$ and let $h _ { 1 } , \ldots , h _ { \ell + 1 } \in \mathcal { H } _ { J }$ repetitions allowed. All of them accept every demonstrated label, so $\begin{array} { r } { y _ { i } \in \bigcap _ { a = 1 } ^ { \ell + 1 } h _ { a } ( x _ { i } ) } \end{array}$ for every $i \in J$ . Hence the function $f \in \bigcap _ { a = 1 } ^ { \ell + 1 } F _ { h _ { a } , S }$ given by the intersection property (44) is correct for every $h _ { a }$ at every position of $J _ { ; }$ , and (46) gives $L _ { S } ( f , h _ { a } ) \leq \alpha$ for every $a \in [ \ell + 1 ]$ At a position where $\begin{array} { r } { \bigcap _ { a } h _ { a } ( x _ { i } ) = \emptyset } \end{array}$ , the label $f ( x _ { i } )$ is incorrect for at least one $h _ { a }$ . Therefore

$$
\begin{array} { r l r } {  { \frac { 1 } { | U | } \sum _ { i \in U } \mathbb { 1 } \big \{ \bigcap _ { a = 1 } ^ { \ell + 1 } h _ { a } ( x _ { i } ) = \emptyset \big \} \leq \frac { 1 } { | U | } \sum _ { i \in U } \sum _ { a = 1 } ^ { \ell + 1 } \mathbb { 1 } \{ f ( x _ { i } ) \notin h _ { a } ( x _ { i } ) \} } } \\ & { } & { \leq \frac { N } { | U | } \sum _ { a = 1 } ^ { \ell + 1 } L _ { S } ( f , h _ { a } ) \leq \frac { N ( \ell + 1 ) \alpha } { N - t } = \beta ^ { \ell } , } \end{array}\tag{50}
$$

where the second inequality extends the sum from U to [N].

The minimax game on U. We prove (47) first for a finite nonempty subset $\mathcal { H } _ { 0 }$ of $\mathcal { H } _ { J }$ Only the finitely many labels in the sets $h ( z ) , h \in \mathcal { H } _ { 0 } , z \in S _ { U }$ , can be correct, so we obtain a finite zero-sum game: the predictor chooses a function $g$ from $S _ { U }$ to these labels, the adversary chooses $h \in \mathcal { H } _ { 0 }$ , and the payof is the average loss $\begin{array} { r } { \frac { 1 } { | U | } \sum _ { i \in U } \mathbb { 1 } \{ g ( x _ { i } ) \notin h ( x _ { i } ) \} } \end{array}$ . Let $\mu$ be a mixed strategy of the adversary, that is, a distribution on $\mathcal { H } _ { 0 }$ . Against $\mu ,$ , the best label at $z \in S _ { U }$ is wrong with probability

$$
\nu _ { z } = 1 - \operatorname* { m a x } _ { y \in \mathcal { V } } \operatorname* { P r } _ { h \sim \mu } [ y \in h ( z ) ] ,
$$

so the function that predicts such a label at every z has expected payof $\textstyle { \frac { 1 } { | U | } } \sum _ { i \in U } \nu _ { x _ { i } }$

To bound $\nu _ { z }$ , draw $h _ { 1 } , \ldots , h _ { \ell + 1 }$ independently from $\mu ,$ and condition on $h _ { 1 } ( z ) \ =$ $\{ y ^ { 1 } , \ldots , y ^ { r } \}$ ; here $1 \leq r \leq \ell .$ , since every correct-label set has at most ℓ elements. For each $j \in [ r ]$ , the hypothesis $h _ { j + 1 }$ excludes $y ^ { j }$ with probability $\operatorname* { P r } _ { h \sim \mu } [ y ^ { j } \notin h ( z ) ] \geq \nu _ { z }$ . These $r$ events are independent, and if all of them occur, no label of $h _ { 1 } ( z )$ lies in $\cap _ { a } h _ { a } ( z )$ . Conditional on $h _ { 1 } ( z )$ , the empty-intersection probability is therefore at least $\nu _ { z } ^ { r } \geq \nu _ { z } ^ { \ell }$ . Averaging over $h _ { 1 }$ gives

$$
\operatorname* { P r } _ { h _ { 1 } , \ldots , h _ { \ell + 1 } \sim \mu } \biggl [ \bigcap _ { a = 1 } ^ { \ell + 1 } h _ { a } ( z ) = \varnothing \biggr ] \geq \nu _ { z } ^ { \ell } .\tag{51}
$$

The bound (50) holds for every choice of $h _ { 1 } , \ldots , h _ { \ell + 1 } \in \mathcal { H } _ { 0 }$ . Taking its expectation over $h _ { 1 } , \ldots , h _ { \ell + 1 } \sim \mu$ and using (51) gives $\begin{array} { r } { \frac { 1 } { | U | } \sum _ { i \in U } \nu _ { x _ { i } } ^ { \ell } \le \beta ^ { \ell } } \end{array}$ , and by Jensen’s inequality,

$$
{ \frac { 1 } { | U | } } \sum _ { i \in U } \nu _ { x _ { i } } \leq \beta .\tag{52}
$$

So every mixed strategy of the adversary has a response with expected payof at most $\beta$ By the minimax theorem [vN28], some mixed strategy of the predictor has expected payof at most $\beta$ against every $h \in \mathcal { H } _ { 0 }$ . Its marginals $p _ { z } , z \in S _ { U }$ , therefore satisfy (47) for every $h \in \mathcal { H } _ { 0 }$ , and taking $\mathcal { H } _ { 0 } = \mathcal { H } _ { J }$ proves (47) when $\mathcal { H } _ { J }$ is finite.

Infinite retained classes. If $\mathcal { H } _ { J }$ is infinite, we use compactness. The set of subprobability distributions

$$
\Delta _ { \leq 1 } ( \mathcal { Y } ) = \Big \{ \theta \in [ 0 , 1 ] ^ { \mathcal { Y } } : \sum _ { y \in F } \theta ( y ) \leq 1 \mathrm { ~ f o r ~ e v e r y ~ f i n i t e ~ } F \subseteq \mathcal { Y } \Big \}
$$

is closed in the product topology, hence compact, and $\mathrm { S O }$ is its finite product over $S _ { U }$ . In this product impose, for each $h \in \mathcal { H } _ { J }$ , the constraint

$$
\frac { 1 } { | U | } \sum _ { i \in U } \sum _ { y \in h ( x _ { i } ) } p _ { x _ { i } } ( y ) \geq 1 - \beta .\tag{53}
$$

Each constraint involves finitely many coordinates and is therefore closed, and every finite collection of constraints is satisfiable by the finite case above. By compactness, some $( p _ { z } ) _ { z \in S _ { U } }$ satisfies all the constraints. Assigning the missing mass $\begin{array} { r } { 1 - \sum _ { y } p _ { z } ( y ) } \end{array}$ of each $p _ { z }$ to a fixed label does not decrease any $p _ { z } ( h ( z ) )$ ), and gives distributions satisfying (47).

Putting the pieces together. By (47), the average in (48) is at most $\beta$ when $h ^ { \star } \in { \mathcal { H } } _ { J }$ and at most one otherwise. Taking expectations and using (49), we obtain

$$
\begin{array} { r l r } {  { \mathbb { E } \big [ L _ { P , h ^ { \star } } ( \widehat { p } ) \big ] \leq \beta + \frac { 1 } { N } = \bigg ( \frac { N ( \ell + 1 ) ( \log B _ { N } + \log N ) } { ( N - t ) t } \bigg ) ^ { 1 / \ell } + \frac { 1 } { N } } } \\ & { } & { \leq 2 \bigg ( \frac { ( \ell + 1 ) ( \log B _ { N } + \log N ) } { t } \bigg ) ^ { 1 / \ell } + \frac { 1 } { N } , } \end{array}
$$

where we used $N / ( N - t ) \ \leq \ 2$ and $2 ^ { 1 / \ell } \ \leq \ 2$ . Since the loss is also at most one, this proves (45). □

Theorem 39 bounds the expected loss of a randomized prediction rule. To obtain a high-probability guarantee for a single predictor, we run the learner on independent blocks of examples and predict a most probable label of the average output.

Lemma 40 (Confidence amplification). Let $\varepsilon , \delta \in ( 0 , 1 )$ . Suppose that a one-correct-label learner for H uses n examples and outputs a randomized prediction rule $\widehat { p }$ with $\mathbb { E } [ L _ { P , h ^ { \star } } ( \widehat { p } ) ] \leq$ $\varepsilon / [ 3 2 ( \ell + 1 ) ^ { 2 } ]$ for every $h ^ { \star } \in { \mathcal { H } }$ , every distribution P with $P ( \operatorname { s u p p } ( h ^ { \star } ) ) = 1$ , and every law D on $\mathcal { X } \times \mathcal { V }$ with marginal $P$ and $\operatorname* { P r } _ { ( x , y ) \sim \mathcal { D } } [ y \in h ^ { \star } ( x ) ] = 1$ . Then

$$
m _ { \mathcal { H } } ^ { \mathrm { o n e } } ( \varepsilon , \delta ) \leq n \left[ 2 4 ( \ell + 1 ) \log ( 2 / \delta ) \right] = O _ { \ell } \big ( n \log ( 2 / \delta ) \big ) .
$$

Proof. Put $s = \ell + 1$ and $b = \lceil 2 4 s \log ( 2 / \delta ) \rceil$ . Run the learner on $b$ disjoint blocks of n examples with independent seeds, and let $p _ { 1 } , \ldots , p _ { b }$ be the outputs. Call $p _ { j }$ good if $L _ { P , h ^ { \star } } ( p _ { j } ) \leq \varepsilon / ( 4 s )$ , and bad otherwise. By Markov’s inequality, each $p _ { j }$ is bad with probability at most $1 / ( 8 s )$ , independently of the others, so the number of bad $p _ { j }$ is stochastically dominated by Bin $\left( b , 1 / ( 8 s ) \right)$ , whose mean is $b / ( 8 s )$ . By a multiplicative Chernof bound,

$$
\operatorname* { P r } \bigl [ \mathrm { a t } \ \mathrm { l e a s t } \ b / ( 4 s ) \ \mathrm { o f } \ \mathrm { t h e } \ p _ { j } \ \mathrm { a r e } \ \mathrm { b a d } \bigr ] \leq e ^ { - b / ( 2 4 s ) } \leq \delta .
$$

Let $\begin{array} { r } { \bar { p } = \frac { 1 } { b } \sum _ { j = 1 } ^ { b } p _ { j } } \end{array}$ , let ${ \widehat { g } } ( x )$ be a most probable label of $\bar { p } ( \cdot \mid x )$ , and assume that fewer than $b / ( 4 s )$ of the $p _ { j }$ are bad. I ${ \mathfrak { f } } ~ { \widehat { g } } ( x ) \not \in h ^ { \star } ( x )$ , then $\bar { p } ( \mathcal { V } \setminus h ^ { \star } ( x ) \mid x ) \ge 1 / s$ by Lemma 11. The bad $p _ { j }$ contribute less than $1 / ( 4 s )$ to this mass, so the good ones contribute more than $3 / ( 4 s )$ Hence

$$
\mathbb { 1 } \{ { \widehat { g } } ( x ) \notin h ^ { \star } ( x ) \} \leq { \frac { 4 s } { 3 b } } \sum _ { j : p _ { j } { \mathrm { ~ g o o d } } } \bigl ( 1 - p _ { j } ( h ^ { \star } ( x ) \mid x ) \bigr ) .
$$

Taking expectations over $x \sim P$ and using $L _ { P , h ^ { \star } } ( p _ { j } ) \ \leq \ \varepsilon / ( 4 s )$ for every good $p _ { j }$ gives $\begin{array} { r } { L _ { P , h ^ { \star } } ( \widehat { g } ) \leq \frac { 4 s } { 3 b } \cdot b \cdot \frac { \varepsilon } { 4 s } < \varepsilon } \end{array}$ . This holds with probability at least $1 - \delta$ , and the learner uses nb examples. □

## 10 One-Correct-Label Learning with Finite Labels

Here we prove Theorem B, in the following stronger form that adds two equivalent conditions and a bound on the expected loss; throughout, $k = | \mathcal { V } | < \infty$ . By the conversion of Section 9, it sufices to attach to every hypothesis a small family of single-label predictions with the intersection property. With finitely many labels, one family serves all hypotheses at once: a finite cover of $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ on the pool, as constructed in Section 7. Its size grows with $k ,$ which is why Section 13 gives each hypothesis a family of its own when $\mathcal { V }$ is infinite.

Theorem 41 (One-correct-label learning with finite labels). Let $| \mathcal { y } | = k < \infty$ , and let $\mathcal { H } \subseteq ( \mathcal { A } \cup \{ \star \} ) ^ { \dot { \mathcal { X } } }$ be nonempty. The following are equivalent:

(i) H is realizably one-correct-label PAC learnable;

(ii) $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ is realizably full-set PAC learnable;

(iii) $d _ { \mathrm { N } } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) ) < \infty ;$

(iv) $d _ { \mathrm { D S } } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) ) < \infty ;$

(v) $d _ { \mathrm { p a i r } } ( \mathcal { T } _ { \ell + 1 } ( \mathcal { H } ) ) < \infty ;$

(vi) $d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) ) < \infty .$

Moreover, let $d = d _ { \mathrm { N } } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) ) < \infty$ . There is a constant $^ { c , }$ depending only on $\ell ,$ such that the following hold.

(a) For every $m \geq 1$ there is a learner that uses m one-correct-label examples and outputs a randomized prediction rule $\widehat { p } _ { m }$ such that

$$
\mathbb { E } \left[ L _ { P , h ^ { \star } } ( \widehat { p } _ { m } ) \right] \leq \operatorname* { m i n } \left\{ 1 , \ c \left[ \frac { ( d + 1 ) \log ^ { 2 } ( 2 ( m + 1 ) ( k + 1 ) ) } { m + 1 } \right] ^ { 1 / \ell } \right\}\tag{54}
$$

for every $h ^ { \star } \in { \mathcal { H } }$ and every law D compatible with $h ^ { \star }$ whose marginal P satisfies $P ( \operatorname { s u p p } ( h ^ { \star } ) ) = 1$ , where the expectation is over the examples and the seed of the learner.

(b) The sample complexity satisfies

$$
m _ { \mathcal { H } } ^ { \mathrm { o n e } } ( \varepsilon , \delta ) \leq \left\lceil c \left( d + 1 \right) \varepsilon ^ { - \ell } \log ^ { 2 } \left( \frac { ( d + 2 ) ( k + 1 ) } { \varepsilon } \right) \log \frac { 2 } { \delta } \right\rceil .\tag{55}
$$

Both bounds also hold with $d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { T } _ { \ell + 1 } ( \mathcal { H } ) )$ in place of $d ,$ and if $d = 0$ , then $m _ { \mathcal { H } } ^ { \mathrm { o n e } } ( \varepsilon , \delta ) = 0$

The proof rests on one observation: the partial intersection of any $\ell + 1$ hypotheses of H on a pool is itself a member of $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ . A cover of $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ therefore contains, for any ℓ + 1 hypotheses, a vector that is correct for all of them wherever their correct-label sets intersect, which is exactly the intersection property. By Lemma 21 the cover has log $B _ { N } = O _ { \ell } ( ( d + 1 ) \log ^ { 2 } ( N k ) )$ on a pool of N inputs, Theorem 39 turns this into (54), and Lemma 40 turns (54) into (55).

Proof. Throughout the proof, c denotes a constant that depends only on $\ell ;$ its value may change between occurrences.

The equivalences. By Lemma 37 with $R = \ell + 1$ , (i) implies (ii). Theorem 16 applied to $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ shows that (ii), (iii) and (iv) are equivalent, and since a finite label space is countable, Theorem 45 shows that (ii) and (vi) are equivalent. Moreover, (iv) implies (v) because $d _ { \mathrm { p a i r } } \leq d _ { \mathrm { D S } } ^ { \cap }$ by Lemma 9, and (v) implies (vi) by Theorem 60. It remains to show that (iii) implies (i), and this follows from the learner constructed below, which also gives the two bounds.

Prediction families from a finite cover. Assume $d < \infty$ , and fix $N \geq 2$ and a multiset S of N inputs. We list S in a fixed total order on $\mathcal { X }$ , so that the ordered pool depends only on $S _ { ; }$ , and let $F _ { S }$ be the finite cover of $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ on this pool given by Lemma 21. Reading each vector at the first occurrence of every input, we regard $F _ { S }$ as a family of functions on supp(S). By (21),

$$
| F _ { S } | \le B _ { N } : = ( 1 + N q k ) ^ { M _ { d } ( N ) } ,\tag{56}
$$

where $q = | \mathcal { A } | \leq ( k + 1 ) ^ { \ell }$ is the number of possible correct-label sets and $M _ { d }$ is the mistake bound (18) of Algorithm 1 for the class $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$

Set $F _ { h , S } = F _ { S }$ for every $h \in \mathcal H$ defined on S. Given $h _ { 1 } , \ldots , h _ { \ell + 1 } \in \mathcal { H }$ defined on S, let $g$ be their intersection restricted to the inputs $x \in \operatorname { s u p p } ( S )$ at which $\textstyle \bigcap _ { a } h _ { a } ( x ) \neq \emptyset$ . Then

$g \in \mathcal { T } _ { \ell + 1 } ( \mathcal { H } )$ by (7), so the covering property gives $\textstyle f \in F _ { S } = \bigcap _ { a } F _ { h _ { a } , S }$ with $f ( x ) \in \bigcap _ { a } h _ { a } ( x )$ at every such input. This is (44), so the families $F _ { h , S }$ have the intersection property with size bound $B _ { N }$

Expected loss. By (12), $\Gamma \leq ( q + 1 ) ^ { \ell + 1 }$ , so (19) and $q \leq ( k + 1 ) ^ { \ell }$ give $M _ { d } ( N ) \leq c ( d +$ 1) $\log ( 2 N ( k + 1 ) )$ . Hence, by (56),

$$
\log B _ { N } \leq M _ { d } ( N ) \log ( 1 + N q k ) \leq c ( d + 1 ) \log ^ { 2 } ( 2 N ( k + 1 ) ) .
$$

Apply Theorem 39 with $N = m + 1$ , which gives a learner that uses m examples. Since $\lfloor N / 2 \rfloor \ge N / 3$ and log $N \leq \log ^ { 2 } ( 2 N ( k + 1 ) )$ , the term with the ℓth root in (45) is at most $\bar { c } [ ( \dot { d } + 1 ) \log ^ { 2 } ( 2 N ( k + 1 ) ) / N ] ^ { 1 / \ell }$ . This quantity is at least $N ^ { - 1 / \ell } \ge 1 / N$ , so increasing c absorbs the additive term $1 / N$ , and (54) follows.

Confidence. Lemma 40 needs a learner with expected loss at most $\varepsilon / [ 3 2 ( \ell + 1 ) ^ { 2 } ]$ , so it remains to find a sample size n at which the right side of (54) is this small. Let $a =$ $\log ( ( d + 2 ) ( k + 1 ) / \varepsilon )$ , which bounds each of $\log ( d + 2 )$ , $\log ( k + 1 )$ and $\log ( 1 / \varepsilon )$ and exceeds 1, and let $n = \lceil C ( d + 1 ) \varepsilon ^ { - \ell } a ^ { 2 } \rceil$ for a parameter $C \geq 1$ . Then $n \leq 2 C ( d + 1 ) \varepsilon ^ { - \ell } a ^ { 2 }$ , and since log $a \leq a$ , we get log $; ( 2 ( n + 1 ) ( k + 1 ) ) \le \log ( 8 C ) + ( \ell + 4 ) a \le c ( 1 + \log C ) a$ . Hence

$$
\frac { ( d + 1 ) \log ^ { 2 } ( 2 ( n + 1 ) ( k + 1 ) ) } { n + 1 } \leq \frac { c ( 1 + \log C ) ^ { 2 } a ^ { 2 } ( d + 1 ) } { C ( d + 1 ) \varepsilon ^ { - \ell } a ^ { 2 } } = \frac { c ( 1 + \log C ) ^ { 2 } } { C } \varepsilon ^ { \ell } .
$$

Choosing C large enough in terms of ℓ therefore makes the right side of (54) with $m = n$ at most $\varepsilon / [ 3 2 ( \ell + 1 ) ^ { 2 } ]$ . Lemma 40 now gives $m _ { \mathcal { H } } ^ { \mathrm { o n e } } ( \varepsilon , \delta ) \leq n \lceil 2 4 ( \ell + 1 ) \log ( 2 / \delta ) \rceil$ , which gives (55) because $n \leq 2 C ( d + 1 ) \varepsilon ^ { - \ell } a ^ { 2 }$ . Finally, $d _ { \mathrm { N } } ^ { \cap } \leq d _ { \mathrm { D S } } ^ { \cap }$ by Lemma 9 and both bounds increase with $d ,$ so they also hold with $d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { T } _ { \ell + 1 } ( \mathcal { H } ) )$ in place of $d .$

The case $d = 0$ . Let $d = 0$ and fix $x \in \mathcal { X }$ . We claim that the sets $h ( x )$ , over all $h \in \mathcal H$ defined at $x ,$ , have a common label. Otherwise, by Lemma 8, at most $\ell + 1$ of them already have an empty intersection. Since $\mathcal { H } \subseteq \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } )$ , these sets form a product witness of size one, that is, a family of correct-label sets at the single input x with no common label, which contradicts $d = 0$ . The predictor that outputs at x the first common label in the fixed enumeration of $\mathcal { V } .$ , and a fixed label if no hypothesis is defined at x, uses no examples and is correct at every input of $\operatorname { s u p p } ( h ^ { \star } )$ , for every $h ^ { \star } \in { \mathcal { H } }$ . This finishes the proof. □

## 11 Worst-Case One-Correct-Label Sample Complexity

The upper bound of Theorem 41 grows as $\varepsilon ^ { - \ell }$ in the accuracy. In this section we show that, for $\ell \geq 2$ , this exponent cannot be improved in general. The construction has two steps. We first define a base class $\mathcal { G } _ { \ell }$ on the grid $\mathbb { N } ^ { \ell }$ , independent of the accuracy, whose partial

intersection class $\pmb { \mathscr { T } } _ { \ell + 1 } ( \pmb { \mathscr { G } } _ { \ell } )$ has generalized Natarajan dimension $( \ell + 1 ) ^ { \ell }$ . Independent copies of $\mathcal { G } _ { \ell }$ then give a lower bound that is linear in the dimension.

The base class. Fix $\ell \geq 2$ . The instance space is $\mathbb { N } ^ { \ell }$ and the label space is $[ \ell ] ;$ ; we write $\boldsymbol { x } = ( x _ { 1 } , \dots , x _ { \ell } )$ for $x \in \mathbb { N } ^ { \ell }$ . For $t \in \mathbb { N } ^ { \ell }$ and a finite set $F \subseteq \mathbb { N } ^ { \ell }$ define

$$
\begin{array} { r } { f _ { F } ( x ) = 1 + \mathbb { 1 } \{ x \in F \} , \qquad h _ { t , F } ( x ) = \{ f _ { F } ( x ) \} \cup \{ i \in [ \ell ] : x _ { i } \neq t _ { i } \} , } \end{array}\tag{57}
$$

and let $\mathcal { G } _ { \ell } = \{ h _ { t , F } : t \in \mathbb { N } ^ { \ell } , F \subseteq \mathbb { N } ^ { \ell }$ finite}. Since $f _ { F } ( x ) \in \{ 1 , 2 \} \subseteq [ \ell ]$ , every $h _ { t , F } ( x )$ is a nonempty subset of $[ \ell ] .$ , so $\mathcal { G } _ { \ell }$ is a total class whose correct-label sets have at most ℓ labels. Thus each target has a binary label $f _ { F } ( x )$ , which is always correct, and any other label i is excluded only on the slice where the ith coordinate equals $t _ { i }$

$$
i \not \in h _ { t , F } ( x ) \quad \Longrightarrow \quad x _ { i } = t _ { i } .\tag{58}
$$

Lemma 42 (Dimension of the base class). $d _ { \mathrm { N } } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { G } _ { \ell } ) ) = ( \ell + 1 ) ^ { \ell }$

Proof. For the lower bound, let $S = [ \ell + 1 ] ^ { \ell }$ , fix any labeling $u : S  \{ 1 , 2 \}$ , and let $F = \{ x \in S : u ( x ) = 2 \}$ , so that $f _ { F } = u$ on S. For $a \in [ \ell + 1 ]$ let $t ^ { ( a ) } = ( a , \ldots , a )$ . Every $h _ { t ^ { ( a ) } , F } ( x )$ contains $u ( x )$ , and a label $i \neq u ( x )$ is excluded by the constituent with $a = x _ { i }$ 2 because then $x _ { i } = t _ { i } ^ { ( a ) }$ and $f _ { F } ( x ) \neq i$ . Therefore

$$
\bigcap _ { a = 1 } ^ { \ell + 1 } h _ { t ^ { ( a ) } , F } ( x ) = \{ u ( x ) \} \qquad ( x \in S ) .
$$

Thus $\pmb { \mathscr { T } } _ { \ell + 1 } ( \pmb { \mathscr { G } } _ { \ell } )$ realizes every pattern of the singletons $\{ 1 \} , \{ 2 \}$ on the $( \ell + 1 ) ^ { \ell }$ inputs of S, and S is product-shattered with the families $\{ \{ 1 \} , \{ 2 \} \}$

For the upper bound, we show that a product-shattered set takes at most $\ell + 1$ values in each coordinate. Let a finite set $S$ be product-shattered by $\pmb { \mathscr { T } } _ { \ell + 1 } ( \pmb { \mathscr { G } } _ { \ell } )$ with families $\mathcal { A } _ { x }$ $x \in S .$ , and fix a label $i \in [ \ell ]$ . Since the sets in $\mathcal { A } _ { x }$ have no common label, some $C _ { x } \in \mathcal { A } _ { x }$ excludes i, and product shattering gives $g \in { \mathcal { T } } _ { \ell + 1 } ( { \mathcal { G } } _ { \ell } )$ with $g ( x ) = C _ { x }$ for all $x \in S$ . Write $\begin{array} { r } { g ( x ) = \bigcap _ { a = 1 } ^ { r } h _ { t ^ { ( a ) } , F ^ { ( a ) } } ( x ) } \end{array}$ for $x \in S$ , with $r \leq \ell + 1$ . For each $x \in S$ we have $i \not \in g ( x )$ , so some constituent $h _ { t ^ { ( a ) } , F ^ { ( a ) } }$ excludes i at $x ,$ and (58) gives $x _ { i } = t _ { i } ^ { ( a ) }$ . Hence

$$
\{ x _ { i } : x \in S \} \subseteq \{ t _ { i } ^ { ( 1 ) } , \ldots , t _ { i } ^ { ( r ) } \} .
$$

This holds for every $i \in [ \ell ]$ , so S lies in a product of ℓ sets of size at most $\ell + 1$ , and $| S | \leq ( \ell + 1 ) ^ { \ell }$ □

Independent copies. For an integer $b \geq 1$ , the instance space is $[ b ] \times \mathbb { N } ^ { \ell }$ and the label space is [ℓ]. A target is given by parameters $\mathbf t = ( t ^ { ( 1 ) } , \dots , t ^ { ( b ) } ) \in (  { \mathbb { N } } ^ { \ell } ) ^ { b }$ and a finite set $F \subseteq [ b ] \times \mathbb { N } ^ { \ell }$

$$
\begin{array} { r } { f _ { F } ( j , x ) = 1 + \mathbb { 1 } \{ ( j , x ) \in F \} , \qquad h _ { \mathsf { t } , F } ( j , x ) = \{ f _ { F } ( j , x ) \} \cup \{ i \in [ \ell ] : x _ { i } \neq t _ { i } ^ { ( j ) } \} . } \end{array}\tag{59}
$$

Let $\mathcal { H } _ { b }$ be the class of all these targets. A finite set $F \subseteq [ b ] \times \mathbb { N } ^ { \ell }$ amounts to a finite set $\{ x : ( j , x ) \in F \} \subseteq \mathbb { N } ^ { \ell }$ for each block $j ,$ , so $\mathcal { H } _ { b }$ is the disjoint product of b copies of the total class $\mathcal { G } _ { \ell } .$ , and Lemmas 10 and 42 give

$$
d _ { \mathrm { N } } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } _ { b } ) ) = b ( \ell + 1 ) ^ { \ell } .\tag{60}
$$

Theorem 43 (Worst-case lower bound). Fix $\ell \geq 2$ . For every $b \geq 1$ and $0 < \varepsilon \le 1 / 2 5 6$

$$
m _ { \mathcal { H } _ { b } } ^ { \mathrm { o n e } } ( \varepsilon , 1 / 8 ) > \frac { b } { 4 ( 3 2 \varepsilon ) ^ { \ell } } .
$$

In particular, for every integer $d \geq 1$ and $b = \lceil d / ( \ell + 1 ) ^ { \ell } \rceil$ 2

$$
d \leq d _ { \mathrm { N } } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } _ { b } ) ) < d + ( \ell + 1 ) ^ { \ell } , \qquad m _ { \mathcal { H } _ { b } } ^ { \mathrm { o n e } } ( \varepsilon , 1 / 8 ) > \frac { d } { 4 ( 3 2 ( \ell + 1 ) \varepsilon ) ^ { \ell } } .
$$

The idea of the proof is that demonstrations reveal the binary labels but nothing about the slices. We put the inputs uniformly on the grid $[ b ] \times [ n ] ^ { \ell }$ , draw the binary labels as fair coins and the slice parameters $T _ { i } ^ { ( j ) }$ uniformly from [n], and let the demonstrator always show the binary label. A learner that sees at most a quarter of the grid must guess the binary label at most grid points, and about half of these guesses are wrong. A wrong guess i at $( j , x )$ is an error only if $x _ { i } = T _ { i } ^ { ( j ) }$ , and since the sample carries no information about $T _ { i } ^ { ( j ) }$ this happens with probability $1 / n$ . The expected loss is therefore of order $1 / n$ , so with n of order $1 / \varepsilon$ the learner needs a constant fraction of the $b n ^ { \ell }$ grid points, that is, of order $b \varepsilon ^ { - \ell }$ examples; the exponent ℓ is the dimension of the grid. A second-moment argument over the bℓ slices, which are independent given the sample, turns this into a bound that holds with constant probability.

Proof. Fix an integer $n \geq 1$ , put $G = [ b ] \times [ n ] ^ { \ell }$ and $N = | G | = b n ^ { \ell }$ , and fix a learner, possibly randomized, that uses $m \leq N / 4$ examples. We construct a random target and a compatible one-correct-label distribution under which the loss of this learner exceeds $1 / ( 1 6 n )$ with probability more than $1 / 8$

The hard distribution. Choose $F \subseteq G$ at random by including each point of G independently with probability $1 / 2$ , so that the values $f _ { F } ( j , x ) , ( j , x ) \in G ,$ are independent and uniform on $\{ 1 , 2 \}$ . Independently of F, let the entries of $\mathbf { T } = ( T _ { i } ^ { ( j ) } ) _ { j \in [ b ] , i \in [ \ell ] }$ be independent and uniform on [n]. The target is $h _ { \mathbf { T } , F } \in \mathcal { H } _ { b }$ , the marginal P is uniform on ${ \dot { G } } ,$ and the demonstrated label at $( j , x )$ is always $f _ { F } ( \boldsymbol { j } , \boldsymbol { x } )$ , which belongs to $h _ { \mathbf { T } , F } ( j , x )$ ; so for every realization of $F$ and T this is a one-correct-label distribution for the target. The learner receives a sample $\sigma$ of $m$ independent examples, and its internal randomness is a seed $\omega \sim \mathrm { U n i f } ( [ 0 , 1 ] )$ , independent of $F _ { ; }$ , T and σ.

Since the demonstrations depend on F but not on T, the entries of T remain independent and uniform on [n] conditional on $( F , \sigma , \omega )$ . Given $( \sigma , \omega )$ , the output of the learner is a fixed predictor, possibly randomized; let $p _ { i } ( j , x )$ be the probability that it predicts i at $( j , x )$ , so

that $p _ { i } ( j , x ) \geq 0$ and $\begin{array} { r } { \sum _ { i = 1 } ^ { \ell } p _ { i } ( j , x ) = 1 } \end{array}$

Disagreement with the binary labels. We first show that the output disagrees with $f _ { F }$ on a constant fraction of G. Let

$$
Q = \frac { 1 } { N } \sum _ { ( j , x ) \in G } \bigl ( 1 - p _ { f _ { F } ( j , x ) } ( j , x ) \bigr ) , \qquad \mu = \mathbb { E } [ Q ] ,
$$

so that $Q$ is the expected fraction of $G$ on which the output disagrees with $f _ { F }$ . At most m points of $G$ appear in $\sigma _ { \mathrm { { : } } }$ and at every other point $( j , x )$ the value $f _ { F } ( \boldsymbol { j } , \boldsymbol { x } )$ is still uniform on $\{ 1 , 2 \}$ given $( \sigma , \omega )$ , so that

$$
\mathbb { E } \big [ 1 - p _ { f _ { F } ( j , x ) } ( j , x ) \big | \sigma , \omega \big ] = 1 - \frac { p _ { 1 } ( j , x ) + p _ { 2 } ( j , x ) } { 2 } \geq \frac { 1 } { 2 } .
$$

Since every summand of Q lies in [0, 1] and $m \leq N / 4$

$$
0 \leq Q \leq 1 , \qquad \mu \geq \frac { N - m } { 2 N } \geq \frac { 3 } { 8 } .\tag{61}
$$

From disagreement to loss. Let L be the loss of the learner’s output against $h _ { \mathbf { T } , F }$ under P. By (59), a label $i \in [ \ell ]$ is invalid at $( j , x )$ exactly when $f _ { F } ( j , x ) \neq i$ and $x _ { i } = T _ { i } ^ { ( j ) }$ . We group the errors according to the slice that causes them: n $\begin{array} { r } { L = Z = \sum _ { j = 1 } ^ { b } \sum _ { i = 1 } ^ { \ell } Z _ { j , i } , } \end{array}$ , where

$$
Z _ { j , i } = { \frac { n } { N } } \sum _ { \substack { x \in [ n ] ^ { \ell } \colon \ x _ { i } = T _ { i } ^ { ( j ) } } } p _ { i } ( j , x ) \mathbb { 1 } \{ f _ { F } ( j , x ) \neq i \} .
$$

Conditional on $( F , \sigma , \omega )$ , the numbers $p _ { i } ( j , x )$ and $f _ { F } ( \boldsymbol { j } , \boldsymbol { x } )$ are fixed, so $Z _ { j , i }$ is a function of $T _ { i } ^ { ( j ) }$ alone, and the variables $Z _ { j , i }$ are conditionally independent. Each slice $\{ x \in [ n ] ^ { \ell } : x _ { i } = T _ { i } ^ { ( j ) } \}$ has $n ^ { \ell - 1 }$ points, so $0 \leq Z _ { j , i } \leq n \cdot n ^ { \ell - 1 } / N = 1 / b$ . Moreover, $\mathrm { P r } [ T _ { i } ^ { ( j ) } = x _ { i } \mid F , \sigma , \omega ] = 1 / n$ for every fixed $x \in [ n ] ^ { \ell }$ , and summing over $j$ and i gives

$$
\mathbb { E } \big [ Z \big | F , \sigma , \omega \big ] = \frac { 1 } { N } \sum _ { ( j , x ) \in G } \sum _ { i \neq f _ { F } ( j , x ) } p _ { i } ( j , x ) = Q .
$$

In particular $\mathbb { E } [ Z ] = \mu \geq 3 / 8$

A second-moment bound. By conditional independence, and since $0 \leq Z _ { j , i } \leq 1 / b$ implies $\begin{array} { r } { \operatorname { V a r } ( Z _ { j , i } \mid F , \sigma , \omega ) \le \frac { 1 } { b } \mathbb { E } [ Z _ { j , i } \mid F , \sigma , \omega ] } \end{array}$ 2

$$
\mathbb { E } \big [ Z ^ { 2 } \big | F , \sigma , \omega \big ] = Q ^ { 2 } + \sum _ { j = 1 } ^ { b } \sum _ { i = 1 } ^ { \ell } \operatorname { V a r } \big ( Z _ { j , i } \big | F , \sigma , \omega \big ) \leq Q ^ { 2 } + \frac { Q } { b } .
$$

Taking expectations and using $Q ^ { 2 } \leq Q$ and $b \geq 1$ gives $\mathbb { E } [ Z ^ { 2 } ] \le 2 \mu$ . Since $\mu / 6 \geq 1 / 1 6$ , the Paley–Zygmund inequality yields

$$
\operatorname* { P r } \left[ Z > { \frac { 1 } { 1 6 } } \right] \geq \operatorname* { P r } \left[ Z > { \frac { \mu } { 6 } } \right] \geq { \binom { 5 } { 6 } } ^ { 2 } { \frac { \mu ^ { 2 } } { \mathbb { E } \left[ Z ^ { 2 } \right] } } \geq { \frac { 2 5 } { 7 2 } } \mu > { \frac { 1 } { 8 } } .
$$

As $Z = n L$ , this means $\operatorname* { P r } [ L > 1 / ( 1 6 n ) ] > 1 / 8$ , where the probability is over $F , { \bf T } .$ , the sample $\sigma$ and the seed $\omega .$ . Since $F$ and T take finitely many values, some fixed realization $( F , \bf T )$ satisfies $\mathrm { P r } _ { \sigma , \omega } [ L > 1 / ( 1 6 n ) ] > 1 / 8$ , and this realization is a target in $\mathcal { H } _ { b }$ together with a compatible one-correct-label distribution.

Choice of n. Now let $0 < \varepsilon \le 1 / 2 5 6$ and choose $n = \lfloor 1 / ( 1 6 \varepsilon ) \rfloor$ . Then $n \geq 1 6 , 1 / ( 1 6 n ) \geq \varepsilon$ and $n \geq 1 / ( 3 2 \varepsilon )$ We have shown that every learner using $m \le b n ^ { \ell } / 4$ examples has $\operatorname* { P r } [ L > \varepsilon ] > 1 / 8$ for some target in $\mathcal { H } _ { b }$ and some compatible distribution, and therefore

$$
m _ { \mathcal { H } _ { b } } ^ { \mathrm { o n e } } ( \varepsilon , 1 / 8 ) > \frac { b n ^ { \ell } } { 4 } \geq \frac { b } { 4 ( 3 2 \varepsilon ) ^ { \ell } } .
$$

For the last assertion, let $b = \lceil d / ( \ell + 1 ) ^ { \ell } \rceil$ . Then $d \leq b ( \ell + 1 ) ^ { \ell } < d + ( \ell + 1 ) ^ { \ell }$ , so (60) gives $d \leq d _ { \mathrm { N } } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } _ { b } ) ) < d + ( \ell + 1 ) ^ { \ell }$ . Since $b \geq d / ( \ell + 1 ) ^ { \ell }$ , the bound just proved gives $m _ { \mathcal { H } _ { b } } ^ { \mathrm { o n e } } ( \varepsilon , 1 / 8 ) > d / ( 4 ( 3 2 ( \ell + 1 ) \varepsilon ) ^ { \ell } )$ . This finishes the proof. □

Remark 44 (The accuracy exponent is a worst-case statement). The exponent ℓ in Theorem 43 is forced for the worst class of a given dimension, not by the dimension alone: for $\ell \geq 2$ no lower bound proportional to $d \varepsilon ^ { - \ell }$ holds for every class with $d _ { \mathrm { N } } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) ) \geq d .$ Indeed, let $\mathcal { X } = \{ x _ { 0 } , x _ { 1 } , . . . , x _ { d } \}$ and $\boldsymbol { \ y } = [ \boldsymbol { \ell } ]$ , and let $\mathcal { H }$ consist of all h with $h ( x _ { j } ) \in \{ \{ 1 \} , \{ 2 \} \}$ for $j \in [ d ]$ and $h ( x _ { 0 } ) = [ \ell ] ;$ ; its largest correct-label set has ℓ labels. The inputs $x _ { 1 } , \ldots , x _ { d }$ are product-shattered with the families $\{ \{ 1 \} , \{ 2 \} \}$ , while $x _ { 0 }$ lies in no product-shattered set, because every member of $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ defined at $x _ { 0 }$ equals [ℓ] there. Thus $d _ { \mathrm { N } } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) ) = d$

This class is no harder than a binary class on d points: every label is correct at $x _ { 0 }$ , and at $x _ { 1 } , \ldots , x _ { d }$ the correct label is unique and is the one demonstrated. Let the learner output a predictor $g$ with $g ( x _ { 0 } ) = 1$ and $g ( x _ { j } ) \in \{ 1 , 2 \}$ for $j \in [ d ]$ that agrees with every demonstrated label at the inputs $x _ { 1 } , \ldots , x _ { d }$ of the sample; such a predictor exists, since the target’s labels at $x _ { j }$ are singletons. A predictor of this form is wrong only at inputs $x _ { j }$ with $j \in [ d ]$ , and there exactly when it disagrees with the demonstrated label; so if its loss exceeds $\varepsilon ,$ it agrees with all m demonstrations at these inputs with probability at most $( 1 - \varepsilon ) ^ { m }$ . A union bound over the $2 ^ { d }$ predictors gives

$$
m _ { \mathcal { H } } ^ { \mathrm { o n e } } ( \varepsilon , \delta ) \leq \left\lceil \frac { d \log 2 + \log ( 1 / \delta ) } { \varepsilon } \right\rceil ,
$$

which is of order $d / \varepsilon$ rather than $d \varepsilon ^ { - \ell } .$

## 12 Full-Set Learning with Countably Many Labels

We now let the label space Y be countable, and possibly infinite. The rates of Section 7 grow with $k = | \mathcal { D } |$ and become vacuous when Y is infinite, so the question is which dimension takes the place of the generalized Natarajan dimension. The answer is the packed dimension of Definition 7: it characterizes full-set learning, in both the realizable and the agnostic setting, with rates that do not depend on |Y|. Throughout the section we fix $\ell \geq 1$ and $\gamma$ as in (9).

Theorem 45 (Full-set learning with countably many labels). For every nonempty class $\mathcal { H } \subseteq ( \mathcal { A } \cup \{ \star \} ) ^ { \dot { \mathcal { X } } }$ the following are equivalent:

(i) H is realizably full-set PAC learnable;

(ii) H is agnostically full-set PAC learnable, with the benchmark $L ^ { \mathrm { s e t } }$ of (5);

(iii) $d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathcal { H } ) < \infty$

Moreover, $i f d = d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathcal { H } ) < \infty$ and $B = d + 1 + \log ( 2 / \delta )$ , then

$$
m _ { \mathcal { H } } ^ { \mathrm { f u l l } } ( \varepsilon , \delta ) \leq \left\lceil c \frac { B } { \varepsilon } \log ^ { 2 } \left( \frac { 2 B } { \varepsilon } \right) \right\rceil ,\tag{62}
$$

$$
m _ { \mathcal { H } } ^ { \mathrm { f u l l , a g } } ( \varepsilon , \delta ) \leq \left\lceil c \frac { B } { \varepsilon ^ { 2 } } \log ^ { 2 } ( \frac { 2 B } { \varepsilon } ) \right\rceil ,\tag{63}
$$

where c depends only on ℓ and $\gamma$ . These conditions imply $d _ { \mathrm { N } } ^ { \Omega } ( \mathcal { H } ) < \infty$ . Finally, for $\ell \geq 2$ finiteness of $d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { H } )$ neither implies nor is implied by realizable full-set PAC learnability, even for total classes with at most two correct labels per input.

The last assertion is proved in Appendix A (Theorems 90 and 91). Necessity of a finite packed dimension is a no-free-lunch argument: the conflict blocks of a packed witness on many inputs act like a shattered set and force a constant expected error (Section 12.1).

Suficiency follows the boosting-to-compression route, in three steps.

• A weak learner (Section 12.2). If $d = d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathcal { H } )$ is finite, then for every finite set of patterns and every distribution on it, all but at most d coordinates can be predicted from the other coordinates with error at most γ. A one-inclusion graph argument turns this into a learner with $O ( d + 1 )$ examples whose expected error is below $1 / ( \ell + 1 )$ .

• Compression (Section 12.3). Boosting the weak learner shows that every realizable sample of size N contains $O ( ( d + 1 ) \log N )$ examples from which a predictor that is valid on the whole sample can be reconstructed. The generalization bound for sample compression schemes then gives (62); its union bound over the $N ^ { O ( ( d + 1 ) \log N ) }$ short lists is the source of the second logarithm.

• The agnostic case (Section 12.4). The learner decodes every short list of examples and outputs a decoded predictor with the least empirical error.

## 12.1 The Lower Bound

When the test input is missing from the training sample, the learner cannot distinguish the targets of a conflict block, and no single label is correct for all of them. Since the blocks of each direction cover $\textrm { a } \gamma$ fraction of the witness, this costs a constant error.

Proposition 46. Suppose that H has $a \gamma$ -packed witness on $n \geq 2$ inputs. Then for every m and every full-set learner that uses m examples, possibly with randomized predictions, there are $h ^ { \star } \in { \mathcal { H } }$ and a distribution P with $P ( \operatorname { s u p p } ( h ^ { \star } ) ) = 1$ such that the output gb satisfies

$$
\mathbb { E } \big [ L _ { P , h ^ { \star } } ( \widehat { g } ) \big ] \geq \frac { \gamma } { \ell + 1 } \left( 1 - \frac { 1 } { n } \right) ^ { m } .\tag{64}
$$

Proof. Let $S = ( x _ { 1 } , \ldots , x _ { n } ) $ be the inputs of the witness, and let $v ^ { 1 } , \ldots , v ^ { M } \in \mathcal { H } | _ { S }$ and $\boldsymbol { B } _ { 1 } , \ldots , \boldsymbol { B } _ { n }$ be as in Definition 7. For each index $a \in [ M ]$ fix $h _ { a } \in \mathcal { H }$ that is defined on S and satisfies $( h _ { a } ( x _ { 1 } ) , \dots , h _ { a } ( x _ { n } ) ) = v ^ { a }$ . Let P be uniform on $\{ x _ { 1 } , \ldots , x _ { n } \}$ , so that $P ( \operatorname { s u p p } ( h _ { a } ) ) = 1$ for every a, and choose the target $h _ { a }$ with a uniform in [M].

Fix a direction i and a sequence of training inputs that avoids $x _ { i } ,$ and condition on it. Let $B \in B _ { i } . \mathrm { ~ A l l ~ } h _ { a }$ with $a \in B$ agree at the witness inputs other than $x _ { i } ,$ , so they produce the same training sample, and the learner predicts at $x _ { i }$ with the same distribution $p ( \cdot \mid x _ { i } )$ for all of them. Since $\cap _ { a \in B } v _ { i } ^ { a } = \emptyset$ , every label y lies outside $v _ { i } ^ { a }$ for some $a \in B$ , and averaging over $y \sim p ( \cdot \mid x _ { i } )$ gives

$$
\sum _ { a \in B } \left[ 1 - p ( v _ { i } ^ { a } \mid x _ { i } ) \right] \geq 1 .
$$

Thus each block contributes at least $1 / M$ to the error at $x _ { i }$ averaged over the uniform index $^ { a , }$ and the indices outside the blocks contribute nonnegatively. The blocks in $B _ { i }$ are disjoint, have at most $\ell + 1$ members each, and cover at least $\gamma M$ indices, so there are at least ${ \gamma } M / ( \ell + 1 )$ of them, and the averaged error at $x _ { i }$ is at least $\gamma / ( \ell + 1 )$ .

The test input is $x _ { i }$ with probability $1 / n$ , and $x _ { i }$ is absent from the training sample with probability $( 1 - 1 / n ) ^ { m }$ ; neither event depends on a. Averaging over i shows that the expected loss, averaged over a, is at least the right side of (64). Hence some fixed index a attains at least this average, and we take $h ^ { \star } = h _ { a }$ □

Corollary 47. If $d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathcal { H } ) = \infty .$ , then H is not realizably full-set PAC learnable.

Proof. Fix m and take a γ-packed witness on $n \geq \operatorname* { m a x } \{ 2 , 2 m \}$ inputs. Since $( 1 - 1 / n ) ^ { m } \geq$ $1 - m / n \geq 1 / 2$ , Proposition 46 gives a target with expected loss at least $\gamma / [ 2 ( \ell + 1 ) ]$ ]. On the other hand, if a learner with m examples had an $( \varepsilon , \delta )$ guarantee for $\varepsilon = \delta = \gamma / [ 8 ( \ell + 1 ) ]$ 2 then by (4) its expected loss would be at most $\varepsilon + \delta = \gamma / [ 4 ( \ell + 1 ) ]$ , a contradiction. □

## 12.2 A Weak Learner from the Packed Dimension

The weak learner is a randomized one-inclusion graph algorithm: it treats the test input as one more position of a pool and predicts its label from the correct-label sets at the other positions. Its error is governed by the following quantities. Let $S = ( x _ { 1 } , \ldots , x _ { n } ) $ be an ordered pool, possibly with repeated inputs, and let $V \subseteq \mathcal { H } | _ { S }$ be finite and nonempty. Write $\mathcal { E } _ { i } ( V )$ for the set of hyperedges of $V$ in direction i (see (8)); each $v \in V$ lies in exactly one of them. For a distribution $\mu$ on $V$ put

$$
b _ { i } ( V , \mu ) = \sum _ { e \in { \mathcal E } _ { i } ( V ) } \Bigl [ \mu ( e ) - \operatorname* { m a x } _ { y \in \mathcal y } \mu \bigl ( \{ v \in e : y \in v _ { i } \} \bigr ) \Bigr ] ,\tag{65}
$$

$$
\Lambda ( V , \mu ) = \sum _ { i = 1 } ^ { n } b _ { i } ( V , \mu ) .\tag{66}
$$

Only finitely many labels occur in $V .$ , so each maximum exists. If $v \sim \mu .$ , then $b _ { i } ( V , \mu )$ is the smallest expected error of a rule that predicts a label at coordinate i from $v _ { - i }$ . Indeed, such a rule chooses one label for each hyperedge in direction $i ,$ and the best choice is a label that lies in $v _ { i }$ for the largest mass of patterns $v$ of the hyperedge. In particular, $0 \leq b _ { i } ( V , \mu ) \leq 1$

The next two lemmas bound Λ by the packed dimension. The main idea is that, once rational weights are written as repetitions, each hyperedge in direction i contains disjoint conflict blocks that cover at least as much mass as its best label misses (Lemma 48). Thus, if $b _ { i } ( V , \mu ) > \gamma$ held in more than $d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathcal { H } )$ directions, the blocks of these directions would form a γ-packed witness, which is impossible (Lemma 49).

Lemma 48. Let $A _ { 1 } , \dotsc , A _ { K } \in { \mathcal { A } }$ . There are pairwise disjoint sets $B _ { 1 } , \ldots , B _ { s } \subseteq [ K ]$ such that $2 \leq | B _ { j } | \leq \ell + 1$ and $\cap _ { a \in B _ { i } } A _ { a } = \emptyset$ for every $j$ , and

$$
\Bigl | \bigcup _ { j = 1 } ^ { s } B _ { j } \Bigr | \geq K - \operatorname* { m a x } _ { y \in \mathcal { V } } \bigl | \{ a \in [ K ] : y \in A _ { a } \} \bigr | .
$$

Proof. As long as the sets at the remaining indices have empty intersection, remove a set of indices whose sets form a minimal family with empty intersection. By Lemma 8 each removed set has between 2 and $\ell + 1$ indices, and the removed sets are disjoint by construction. When the process stops, either no index remains or the remaining sets share a label $y ,$ and in the second case at most $| \{ a : y \in A _ { a } \} |$ indices remain. In both cases at least the claimed number of indices has been removed. □

Lemma 49. Suppose $d = d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathcal { H } ) < \infty$ . Let S be a pool of n inputs, let $V \subseteq \mathcal { H } | _ { S }$ be finite and nonempty, and let $\mu$ be a distribution on V . Then at most d coordinates i satisfy $b _ { i } ( V , \mu ) > \gamma$ . In particular,

$$
\Lambda ( V , \mu ) \leq d + n \gamma .\tag{67}
$$

Proof. We first dispose of repeated inputs: if $x _ { i } = x _ { j }$ for some $j \neq i$ , then $b _ { i } ( V , \mu ) = 0$ Indeed, every $v \in V$ is the pattern of a hypothesis, so $v _ { i } = v _ { j }$ . The members of a hyperedge in direction i agree at $j$ , hence also at $i ,$ so any label of their common set at i is correct for all of them, and every term of (65) vanishes.

Next, let $\mu$ take rational values with common denominator K. We show that $d + 1$ coordinates i with $b _ { i } ( V , \mu ) > \gamma$ would yield a γ-packed witness on $d + 1$ distinct inputs. Form a list of length $K$ that contains $K \mu ( v )$ copies of each $v \in V$ , and fix i. Applying Lemma 48, separately in each hyperedge $e \in \mathcal { E } _ { i } ( V )$ , to the sets $v _ { i }$ of the copies in e gives disjoint blocks of between 2 and $\ell + 1$ copies that agree outside i and have empty intersection at i. In e these blocks cover at least $K \mu ( e ) - K$ m $\operatorname { a x } _ { y } \mu ( \{ v \in e : y \in v _ { i } \} )$ copies, so the blocks of direction i together cover at least $K b _ { i } ( V , \mu )$ copies.

Now suppose that a set I of $d + 1$ coordinates satisfies $b _ { i } ( V , \mu ) > \gamma$ for all $i \in I$ . By the first step, the inputs $x _ { i }$ with $i \in I$ are distinct. Project every copy onto I, keeping all $K$ copies, and keep the blocks of the directions in I. The projected copies lie in $\mathcal { H } | _ { S _ { I } }$ , where $S _ { I }$ is the subpool indexed by I. The blocks are sets of copies, so projection leaves them disjoint and of the same sizes; the members of a block of direction i still agree at every coordinate of I other than $i ,$ and their sets at i, and hence their empty intersection, are unchanged. Since the blocks of each direction in I still cover more than $\gamma K$ copies, this is a γ-packed witness on $d + 1$ distinct inputs, a contradiction.

For a general $\mu ,$ each $b _ { i } ( V , \mu )$ is a continuous function of the finitely many masses of $\mu ,$ being a sum of linear terms minus maxima of finitely many linear terms. So if $b _ { i } ( V , \mu ) > \gamma$ held for $d + 1$ coordinates, it would also hold for them under a suficiently close rational distribution, which we have just excluded. Finally, $b _ { i } \leq 1$ for the at most d exceptional coordinates and $b _ { i } \leq \gamma$ for the others, which gives (67). □

Lemma 49 says that against every distribution on patterns some prediction rule makes few mistakes on the pool. To obtain a learner we need the quantifiers in the other order, a single rule that is good for every pattern; the minimax theorem provides it, and the leave-one-out argument of the one-inclusion graph algorithm turns its bound on the pool into a bound on the expected error.

Lemma 50. Suppose $d = d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathcal { H } ) < \infty$ . For every $m \geq 1$ there is a full-set learner that maps each sample σ of m examples to a randomized prediction $p _ { \sigma }$ such that

$$
\mathbb { E } _ { \sigma \sim \mathcal { D } _ { P , h ^ { \star } } ^ { m } } \left[ L _ { P , h ^ { \star } } ( p _ { \sigma } ) \right] \leq \gamma + \frac { d } { m + 1 }\tag{68}
$$

for every $h ^ { \star } \in { \mathcal { H } }$ and every P with $P ( \operatorname { s u p p } ( h ^ { \star } ) ) = 1$ . Moreover, $p _ { \sigma }$ is a deterministic function of $\sigma$ that does not change when the examples of σ are permuted.

Proof. Let $n = m + 1$ and fix an ordered pool $S = ( x _ { 1 } , \ldots , x _ { n } ) $ . A transductive rule on $S$ assigns to each direction $i \in [ n ]$ and each $u \in \mathcal { A } ^ { n - 1 } \mathrm { ~ a ~ }$ distribution $p _ { i , u }$ on $\mathcal { V }$ , and at position i of a pattern v it predicts with $p _ { i , v _ { - i } }$ . Its loss on v is the expected number of positions at which it is wrong,

$$
\sum _ { i = 1 } ^ { n } \bigl [ 1 - p _ { i , v _ { - i } } ( v _ { i } ) \bigr ] .
$$

We first show that some transductive rule on S has loss at most $d + n \gamma$ on every $v \in \mathcal { H } | _ { S }$

Let $V \subseteq \mathcal { H } | _ { S }$ be finite and nonempty, and let $\mathcal { \mathrm { y } } _ { V }$ be the finite set of labels that occur in $V .$ . Consider the finite zero-sum game in which the learner chooses a label in $\mathcal { { V } } _ { V }$ for every hyperedge of $V .$ , the adversary chooses $v \in V$ , and the loss is the number of positions i at which the label chosen for the hyperedge of v in direction i is not in $v _ { i }$ . Against a distribution $\mu$ on $V .$ , the learner can choose the labels of diferent hyperedges separately, and the label $y$ chosen for $e \in \mathcal { E } _ { i } ( V )$ contributes $\mu ( e ) - \mu ( \{ v \in e : y \in v _ { i } \} )$ to the expected loss. Labels outside $\mathcal { \mathrm { y } } _ { V }$ are correct for no pattern, so the maximum in (65) may be taken over $\mathcal { { V } } _ { V }$ , and the best response to $\mu$ has expected loss exactly $\Lambda ( V , \mu ) \leq d + n \gamma$ (Lemma 49). By the minimax theorem [vN28], the learner has a mixed strategy whose expected loss is at most $d + n \gamma$ against every $v \in V$ . This loss depends only on the law $p _ { i , u }$ of the label that the strategy chooses for each hyperedge $e _ { i , u }$ , so these laws, with the remaining $p _ { i , u }$ set arbitrarily, form a transductive rule with loss at most $d + n \gamma$ on every $v \in V$

The set $\mathcal { H } | _ { S }$ may be infinite, and we pass to it by compactness. When Y is infinite, the probability distributions on Y do not form a closed subset of $[ 0 , 1 ] ^ { y }$ , so we use the compact set $\mathcal { P } _ { \leq 1 } ( \mathcal { V } )$ of all $z \in [ 0 , 1 ] ^ { y }$ with $\textstyle \sum _ { y \in F } z _ { y } \leq 1$ for every finite $F \subseteq \mathcal { V }$ . Assign a vector $z ^ { i , u } \in \mathcal { P } _ { \leq 1 } ( \mathcal { Y } )$ to every pair $( i , u )$ , and let $Z$ be the compact product of these spaces. For $v \in \mathcal { H } | _ { S }$ , the constraint $\begin{array} { r } { \sum _ { i = 1 } ^ { n } [ 1 - \sum _ { y \in v _ { i } } z _ { y } ^ { i , v _ { - i } } ] \leq d + n \gamma } \end{array}$ involves finitely many coordinates, since each $v _ { i }$ is finite, so it defines a closed subset of $Z$ . By the finite case, every finitely many of these constraints are satisfied by some point of $Z _ { i }$ , and by the finite intersection property some point satisfies all of them. Adding the missing mass $1 - \textstyle \sum _ { y } z _ { y } ^ { i , u }$ of each $z ^ { i , u }$ to one fixed label gives distributions $p _ { i , u }$ and does not decrease the mass on any correct-label set, so the resulting rule has loss at most $d + n \gamma$ on every $v \in \mathcal { H } | _ { S }$

Now fix such a rule for every ordered pool of n inputs; since $\mathcal { X }$ is countable, there are countably many pools, so the resulting learner is measurable. Replace the rule on each pool S by its average over all permutations of the positions of $S _ { i }$ , applied to the correspondingly permuted pool and patterns. This preserves the bound on the loss, which is a sum over all positions, and makes the rules permutation-equivariant. Given $\boldsymbol { \sigma } = ( ( x _ { 1 } , A _ { 1 } ) , \dots , ( x _ { m } , A _ { m } ) )$ and a test input $x ,$ the learner forms the pool $( x _ { 1 } , \ldots , x _ { m } , x )$ and predicts with $p _ { n , ( A _ { 1 } , \dots , A _ { m } ) }$ By equivariance, this prediction does not change when the examples of $\sigma$ are permuted.

Finally, let $x _ { 1 } , \ldots , x _ { n }$ be i.i.d. from P and $v = \left( h ^ { \star } ( x _ { 1 } ) , \ldots , h ^ { \star } ( x _ { n } ) \right)$ . Almost surely all $x _ { j }$ lie in supp(h<sup>⋆</sup>), so $v \in \mathcal { H } | _ { S }$ . The expected loss of the learner is the expected error of the rule at position $n ,$ and by exchangeability and equivariance it equals the expected error at every position i. It is therefore $1 / n$ times the expected loss of the transductive rule on v, which is at most $( d + n \gamma ) / n = \gamma + d / ( m + 1 )$ □

## 12.3 Compression and the Realizable Upper Bound

A full-set sample $\sigma = \left( \left( x _ { 1 } , A _ { 1 } \right) , \ldots , \left( x _ { n } , A _ { n } \right) \right)$ is realizable by H if some $h \in \mathcal H$ satisfies $x _ { i } \in \operatorname { s u p p } ( h )$ and $h ( x _ { i } ) = A _ { i }$ for all i. We now boost the weak learner on a realizable sample until its most probable label is correct on the whole sample, following the boostingto-compression argument of [FS97], [MY16], [DMY16] and $\mathrm { [ C E H ^ { + } 2 6 ] }$ . As in the last two papers, training labels are sets and the loss is validity; Lemma 11 replaces the majority vote. Since the boosted predictor is determined by few examples of the sample, the generalization argument for sample compression schemes [LW86] then gives the realizable bound.

Boosting. We prove the boosting step once, for an arbitrary class $\mathcal { H } _ { \mathrm { a u x } } \subseteq ( \mathcal { A } \cup \{ \star \} ) ^ { \mathcal { X } }$ and with the number of rounds as a parameter, because Section 13 applies it to an intersection class. Fix $M \geq 1$ and a full-set learner for ${ \mathcal { H } } _ { \mathrm { a u x } }$ that uses M examples and whose randomized prediction has expected loss at most α, where $0 \leq \alpha < 1 / ( \ell + 1 )$ , for every target $h \in \mathcal { H } _ { \mathrm { a u x } }$ and every P with $P ( \operatorname { s u p p } ( h ) ) = 1$ . Averaging its prediction over its seed and over all orderings of its sample gives a rule $\sigma \mapsto p _ { \sigma }$ with the same guarantee that is symmetric, that is, $p _ { \sigma }$ depends only on the multiset of examples in σ. Put $\tau = 1 / ( \ell + 1 ) , \Delta = \tau - \alpha > 0$ and

$$
\eta = \frac { \Delta } { 2 ( \alpha + 1 ) } , \qquad \zeta = \eta \tau - \alpha ( e ^ { \eta } - 1 ) .\tag{69}
$$

Both constants depend only on ℓ and $\alpha ,$ and $\zeta > 0$ (see the proof below).

The next lemma is the multiplicative-weights analysis of boosting [FS97], run with weights on the positions of a pool. In each round the weak learner has small error on the weighted pool, so the total weight grows slowly, while by Lemma 11 the weight of a position at which the final most probable label is invalid grows fast. Here we only need long runs, after which no such position is left; the correction step at the end of the lemma is used in Section 13, where runs are shorter.

Lemma 51. Fix integers N, $T \geq 1$ . Let $S = ( x _ { 1 } , \ldots , x _ { n } ) $ be an ordered pool with $n \leq N$ let $h \in \mathcal { H } _ { \mathrm { a u x } }$ , and suppose that $D = \{ i \in [ n ] : x _ { i } \in \operatorname { s u p p } ( h ) \}$ is nonempty. Then there are multisets $\sigma _ { 1 } , \dots , \sigma _ { T } ,$ each consisting of M examples $( { { x } _ { i } } , { { h } \mathrm { ( } { { x } _ { i } } ) } )$ with $i \in D$ , such that the predictor

$$
f _ { 0 } ( x ) = a \ m o s t \ p r o b a b l e \ l a b e l \ o f \ \frac { 1 } { T } \sum _ { t = 1 } ^ { T } p _ { \sigma _ { t } } ( \cdot \mid x )
$$

satisfies $f _ { 0 } ( x _ { i } ) \not \in h ( x _ { i } )$ for at most $N e ^ { - \zeta T }$ positions $i \in D$ . Moreover, let $Q \subseteq D$ be the set of these positions, and let $f ( x )$ be the least label of $h ( x _ { i } )$ in the fixed enumeration of Y if $x = x _ { i }$ for some $i \in Q$ , and $f ( x ) = f _ { 0 } ( x )$ otherwise. Then $f ( x _ { i } ) \in h ( x _ { i } )$ for every $i \in D$ , and $f$ is determined by $\sigma _ { 1 } , \dots , \sigma _ { T }$ and the examples $( x _ { i } , h ( x _ { i } ) )$ with $i \in Q$

Proof. We first check that $\zeta > 0$ . Since $0 < \eta < 1$ , we have $e ^ { \eta } - 1 \le \eta + \eta ^ { 2 }$ , and since $\alpha \eta \le \Delta / 2$ , this gives $\zeta \ge \eta \Delta - \alpha \eta ^ { 2 } \ge \eta \Delta / 2 > 0$

Give every position $i \in D$ the weight $w _ { 1 } ( i ) = 1$ . At round t, let $P _ { t }$ be the distribution of $x _ { i }$ when $i \in D$ is drawn with probability proportional to $w _ { t } ( i )$ . Then $P _ { t } ( \operatorname { s u p p } ( h ) ) = 1$ , and for every sample σ the loss $L _ { P _ { t } , h } ( p _ { \sigma } )$ is the weighted average of $1 - p _ { \sigma } ( h ( x _ { i } ) \mid x _ { i } )$ over $i \in D$ A sample from $\mathcal { D } _ { P _ { t } , h } ^ { M }$ consists of M examples $( x _ { i } , h ( x _ { i } ) )$ with $i \in D$ , and by the weak-learning guarantee its expected loss is at most α. Hence some multiset $\sigma _ { t }$ of $M$ such examples has $L _ { P _ { t } , h } ( p _ { \sigma _ { t } } ) \leq \alpha$ . Put

$$
\lambda _ { t } ( i ) = 1 - p _ { \sigma _ { t } } ( h ( x _ { i } ) \mid x _ { i } ) \in [ 0 , 1 ] \qquad \mathrm { a n d } \qquad w _ { t + 1 } ( i ) = w _ { t } ( i ) e ^ { \eta \lambda _ { t } ( i ) } .
$$

Since $e ^ { \eta z } \leq 1 + ( e ^ { \eta } - 1 ) z$ for $0 \leq z \leq 1$ , the total weight grows in each round by a factor at most $1 + ( e ^ { \eta } - 1 ) \alpha \le \exp ( \alpha ( e ^ { \eta } - 1 ) )$ . As it starts at $| D | \leq N$ , after $T$ rounds it is at most $N \exp ( T \alpha ( e ^ { \eta } - 1 ) )$ .

Now suppose that $f _ { 0 } ( x _ { i } ) \not \in h ( x _ { i } )$ for some $i \in D$ . Since $| h ( x _ { i } ) | \leq \ell ;$ , Lemma 11 gives $\begin{array} { r } { T ^ { - 1 } \sum _ { t } \lambda _ { t } ( i ) \geq \tau . } \end{array}$ , so the final weight of i is at least $e ^ { \eta T \tau }$ . If there are $e _ { T }$ such positions, then

$$
e _ { T } e ^ { \eta T \tau } \le N e ^ { T \alpha ( e ^ { \eta } - 1 ) } , \qquad \mathrm { t h a t ~ i s , } \qquad e _ { T } \le N e ^ { - \zeta T } .
$$

For the correction step, note that equal inputs have equal correct-label sets under h and equal values under $f _ { 0 } ,$ , so a position whose input is also the input of a position in $Q$ lies in $Q$ itself. Hence $f ( x _ { i } ) = f _ { 0 } ( x _ { i } ) \in h ( x _ { i } )$ for $i \in D \backslash Q$ , while $f ( x _ { i } ) \in h ( x _ { i } )$ for $i \in Q$ by definition. The values of f depend only on the stated examples. □

A compression scheme. Assume $d = d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathcal { H } ) < \infty$ , and apply the construction above with $\mathcal { H } _ { \mathrm { a u x } } = \mathcal { H }$ and the learner of Lemma 50, where

$$
\alpha = \frac { 1 } { 2 } \Bigl ( \gamma + \frac { 1 } { \ell + 1 } \Bigr ) , \qquad M = \operatorname * { m a x } \Bigl \{ 1 , \Bigl [ \frac { d } { \alpha - \gamma } \Bigr ] \Bigr \} .
$$

By (68), the expected loss of this learner is at most $\gamma + d / ( M + 1 ) \leq \alpha$ , and $\alpha < 1 / ( \ell + 1 )$ so $\eta$ and $\zeta$ depend only on $\ell$ and $\gamma .$ . For a public bound N on the sample size put

$$
T _ { N } = \Big \lceil \frac { 2 \log ( 2 N ) } { \zeta } \Big \rceil , \qquad r _ { N } = M T _ { N } .\tag{70}
$$

The decoder Dec<sub>N</sub> splits an ordered list of $r _ { N }$ full-set examples into $T _ { N }$ consecutive groups $\sigma _ { 1 } , \dots , \sigma _ { T _ { N } }$ of M examples, and its output, the decoded predictor of the list, is a most probable label of $\begin{array} { r } { T _ { N } ^ { - 1 } \sum _ { t } p _ { \sigma _ { t } } ( \cdot \mid x ) } \end{array}$ at every x.

Lemma 52. Every nonempty realizable sample σ of size at most N has an ordered list of $r _ { N }$ of its examples, with repetitions allowed, whose decoded predictor is valid on all of $\sigma .$ Moreover, for fixed ℓ and $\gamma ,$

$$
r _ { N } \leq c ( d + 1 ) \log ( 2 N ) .\tag{71}
$$

Proof. Apply Lemma 51 with $T = T _ { N }$ rounds to the inputs of $\sigma$ and a hypothesis $h$ that realizes $\sigma$ . Then D consists of all positions, so the multisets $\sigma _ { t }$ consist of examples of $\sigma ,$ and since $N e ^ { - \zeta T _ { N } } \leq 1 / ( 4 N ) < 1$ , the predictor $f _ { 0 }$ is valid on all of $\sigma$ . List the multisets one after the other, in any order within each. Since each $p _ { \sigma _ { t } }$ depends only on the multiset $\sigma _ { t } ,$ the decoder returns $f _ { 0 }$ on this list. Finally, $M \leq c ( d + 1 )$ and $T _ { N } \le c \log ( 2 N )$ , which gives (71). □

In other words, the lists and Dec form a sample compression scheme of size $r _ { N }$ , in which the stored examples are ordered and may repeat.

Proof of Theorem $\mathit { 4 5 , }$ realizable part. Let $\sigma \sim \mathcal { D } _ { P , h ^ { \star } } ^ { N }$ . The decoded predictor of an index list in $[ N ] ^ { r _ { N } }$ is that of the list of examples it indexes. The learner goes through these index lists in a fixed order and outputs the decoded predictor $\widehat g$ of the first list whose decoded predictor is valid on all of $\sigma .$ . Such a list exists by Lemma 52, since $\sigma$ is realizable almost surely.

Fix a list, let J be its set of distinct indices, so that $| J | \le r _ { N }$ , and let $g$ be its decoded predictor. Conditional on the examples indexed by $J ,$ the predictor $g$ is fixed and the other $N - | J |$ examples are i.i.d. If $L _ { P , h ^ { \star } } ( g ) > \varepsilon$ , then $g$ is valid on all of them with probability at most $e ^ { - \varepsilon ( N - r _ { N } ) }$ . A union bound over the at most $N ^ { r _ { N } }$ lists gives

$$
\operatorname* { P r } \bigl [ L _ { P , h ^ { \star } } ( \widehat { g } ) > \varepsilon \bigr ] \leq \exp \bigl ( r _ { N } \log N - \varepsilon ( N - r _ { N } ) \bigr ) , \qquad N > r _ { N } .\tag{72}
$$

Now take $N = \lceil c B \varepsilon ^ { - 1 } \log ^ { 2 } ( 2 B / \varepsilon ) \rceil$ . By (71), the term $r _ { N }$ log N is of order $( d + 1 ) \log ^ { 2 } ( 2 N )$ and this is where the squared logarithm in (62) comes from. For a large enough $c ,$ a routine calculation gives $r _ { N } ( \log N + 1 ) \leq \varepsilon N / 2$ and log $( 1 / \delta ) \le \varepsilon N / 2$ . The first inequality implies $N > 2 r _ { N }$ , and together they bound the exponent in (72) by $- \varepsilon N / 2 \le - \log ( 1 / \delta )$ . This proves (62) and $( \mathrm { i i i } ) { \Rightarrow } ( \mathrm { i } )$ . Corollary 47 gives $( \mathrm { i } ) { \Rightarrow } ( \mathrm { i i i } )$ , and $d _ { \mathrm { N } } ^ { \cap } ( \mathcal { H } ) \leq d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathcal { H } )$ by Lemma 9.

## 12.4 Agnostic Full-Set Learning

The benchmark $L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } )$ in (5) compares the whole set $h ( x )$ with the observed set, so the examples on which a fixed $h \in \mathcal H$ equals the observed set are realizable by h. The compression scheme therefore applies to them: some decoded predictor is valid on these examples, and it is wrong on the sample only where h disagrees with the observed set. It remains to show that the empirical errors of the finitely many decoded predictors concentrate. This is the set-valued form of the compression-based agnostic reduction of [DMY16] and $\mathrm { [ C E H ^ { + } 2 6 ] }$

Lemma 53. For every $n \geq 1$ , let $r _ { n } \geq 1$ and let $\mathrm { D e c } _ { n }$ map ordered lists of $r _ { n }$ full-set examples to predictors. Assume that for every nonempty realizable sample $\sigma ^ { \prime }$ of size at most n there is an ordered list of $r _ { n }$ examples of $\sigma ^ { \prime }$ , with repetitions allowed, whose decoded predictor is valid on all of $\sigma ^ { \prime }$ . Then for $n > r _ { n }$ there is an agnostic full-set learner with n examples whose output satisfies, with probability at least $1 - \delta$

$$
L _ { \mathcal { D } } ( \widehat { g } ) \leq L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } ) + \frac { r _ { n } } { n } + \sqrt { \frac { r _ { n } \log n + \log ( 4 / \delta ) } { 2 ( n - r _ { n } ) } } + \sqrt { \frac { \log ( 2 / \delta ) } { 2 n } } .\tag{73}
$$

For $n \geq 2 r _ { n } .$ , the excess in (73) is $O ( \sqrt { ( r _ { n } \log n + \log ( 1 / \delta ) ) / n } )$ , the familiar form of agnostic compression bounds.

Proof. Let $\sigma = ( ( x _ { i } , A _ { i } ) ) _ { i = 1 } ^ { n } \sim \mathcal { D } ^ { n }$ and write $\begin{array} { r } { \widehat { L } _ { \sigma } ( g ) = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbb { 1 } \{ g ( x _ { i } ) \notin A _ { i } \} } \end{array}$ . The candidates are the decoded predictors of all index lists in $[ n ] ^ { r _ { n } }$ and one fixed fallback predictor, at most $n ^ { r _ { n } } + 1$ predictors in all. The learner outputs a candidate $\widehat g$ that minimizes $\widehat { L } _ { \sigma }$ , with ties broken by a fixed order. Note that $\widehat g$ does not depend on any hypothesis in $\mathcal { H } .$

Fix $h \in \mathcal H$ . The positions i with $h ( x _ { i } ) = A _ { i }$ form a realizable subsample (recall that $h ( x _ { i } ) \neq A _ { i }$ when $h ( x _ { i } ) = \star )$ . If this subsample is nonempty, some candidate is valid on it; if it is empty, every candidate has empirical loss at most 1. In both cases

$$
{ \widehat { L } } _ { \sigma } ( { \widehat { g } } ) \leq { \frac { 1 } { n } } \sum _ { i = 1 } ^ { n } \mathbb { 1 } \{ h ( x _ { i } ) \neq A _ { i } \} .\tag{74}
$$

Next, fix an index list, let J be its set of distinct indices, $t = | J | \leq r _ { n }$ , and let $g$ be its decoded predictor. Conditional on the examples indexed by $J ,$ the predictor $g$ is fixed and the other $n - t$ examples are i.i.d. from ${ \mathcal { D } } .$ . By Hoefding’s inequality [Hoe63], for every $u > 0$ 2

$$
\operatorname* { P r } \Bigl [ L _ { \mathcal { D } } ( g ) > \frac { 1 } { n - t } \sum _ { i \notin J } \mathbb { 1 } \left\{ g ( x _ { i } ) \notin A _ { i } \right\} + u \Bigr | ( x _ { i } , A _ { i } ) _ { i \in J } \Bigr ] \leq e ^ { - 2 ( n - t ) u ^ { 2 } } .
$$

Since $\widehat { L } _ { \sigma } ( g )$ is the average of the mean on J and the mean outside $J$ with weights $t / n$ and $( n - t ) / n$ , the mean outside $J$ exceeds $\widehat { L } _ { \sigma } ( g )$ by at most $t / n$ . Take

$$
u = \sqrt { \frac { r _ { n } \log n + \log ( 4 / \delta ) } { 2 ( n - r _ { n } ) } } ,
$$

so that $e ^ { - 2 ( n - t ) u ^ { 2 } } \leq \delta / ( 4 n ^ { r _ { n } } )$ ; the same bound holds for the fallback, with $J = \emptyset$ . A union bound over the candidates shows that, with probability at least $1 - \delta / 2$ , every candidate $g$ satisfies

$$
L _ { \mathcal { D } } ( g ) \leq \widehat { L } _ { \sigma } ( g ) + \frac { r _ { n } } { n } + u .
$$

By Hoefding’s inequality again, with probability at least $1 - \delta / 2$

$$
\frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbb { 1 } \{ h ( x _ { i } ) \neq A _ { i } \} \leq \operatorname* { P r } _ { ( x , A ) \sim \mathcal { D } } [ h ( x ) \neq A ] + \sqrt { \frac { \log ( 2 / \delta ) } { 2 n } } .
$$

Combining the last two displays with (74) gives (73) with $\operatorname* { P r } _ { ( x , A ) \sim { \mathcal { D } } } [ h ( x ) \neq A ]$ in place of $L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } )$ . Applying this to some $h \in { \mathcal { H } }$ with $\operatorname* { P r } _ { ( x , A ) \sim \mathcal { D } } [ h ( x ) \neq A ] \leq L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } ) + 1 / j$ shows that the event on which $L _ { \mathcal { D } } ( \widehat { \boldsymbol { g } } )$ exceeds the right side of (73) plus $1 / j$ has probability at most $\delta$ As $j \to \infty$ these events increase to the event on which (73) fails, so it too has probability at most $\delta .$ □

Proof of Theorem $\mathit { 4 5 , }$ agnostic part. Assume $d = d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathcal { H } ) < \infty$ . By Lemma $5 2 ,$ , the decoders $\mathrm { D e c } _ { n }$ of Section 12.3 satisfy the assumption of Lemma 53, with $r _ { n } \leq c ( d + 1 ) \log ( 2 n )$ Take

$$
n = \Big \lceil c \frac { B } { \varepsilon ^ { 2 } } \log ^ { 2 } \Big ( \frac { 2 B } { \varepsilon } \Big ) \Big \rceil .
$$

For a large enough $^ { c , }$ depending only on $\ell$ and $\gamma ,$ , we have $n > 2 r _ { n }$ , and the three excess terms in (73) sum to at most $\varepsilon ;$ as in the realizable case, the dominant term $\sqrt { r _ { n } \log n / n }$ with $r _ { n }$ log n of order $( d + 1 ) \log ^ { 2 } ( 2 n )$ , is what requires the factor $\log ^ { 2 } ( 2 B / \varepsilon ) / \varepsilon ^ { 2 }$ . This proves (63) and $( \mathrm { i i i } ) { \Rightarrow } ( \mathrm { i i } )$ . Finally, agnostic learnability implies realizable learnability (Section 5.2), which proves $( \mathrm { i i } ) { \Rightarrow } ( \mathrm { i } )$ □

Remark 54. By Theorem 45, finiteness of $d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathcal { H } )$ for one $\gamma$ in the range (9) is equivalent to finiteness for every such $\gamma$ . It is not equivalent to finiteness of $\mathrm { s u p } _ { \gamma } d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathcal { H } )$ , as the following learnable class shows. Let $\ell = 1$ , let $\mathcal { X }$ be infinite, and let $\mathcal { H }$ consist of the function that equals $\{ 0 \}$ everywhere and the functions that equal {1} at one input and {0} elsewhere. The class is learnable: predict 1 at an input seen with label set {1}, and 0 elsewhere. On the

other hand, $d _ { \mathrm { D S } } ^ { \Omega } ( \mathcal { H } ) = 1$ and

$$
d _ { \mathrm { p a c k } , \rho } ^ { \cap } ( \mathcal { H } ) = \lfloor 2 / \rho \rfloor - 1 \qquad ( 0 < \rho \leq 1 ) .
$$

Indeed, on n inputs a block in direction i must pair a copy of the zero pattern with a copy of the pattern that is {1} only at i. If these patterns have proportions $\mu _ { 0 } , \mu _ { 1 } , \ldots , \mu _ { n }$ in the list, then coverage ρ requires $\mu _ { 0 } , \mu _ { i } \geq \rho / 2$ for all i, and thus $( n + 1 ) \rho / 2 \leq 1$ . Conversely, the uniform list of the $n + 1$ patterns has coverage $2 / ( n + 1 )$ in every direction. This construction remains valid when the same class is viewed with a larger bound $\ell \geq 2 .$ , so the supremum is infinite in that case as well.

## 13 One-Correct-Label Learning with Countably Many Labels

We now return to one-correct-label feedback and drop the assumption that Y is finite; as everywhere in the paper, Y is countable. The main result of this section shows that onecorrect-label learnability of H is still equivalent to full-set learnability of $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ , and hence, by Theorem 45, to finiteness of $d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) )$ ).

Theorem 55. Let $\mathcal { H } \subseteq ( \mathcal { A } \cup \{ \star \} ) ^ { \mathcal { X } }$ be nonempty, and fix $\gamma$ as in (9). The following are equivalent:

(i) H is realizably one-correct-label PAC learnable;

(ii) $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ is realizably full-set PAC learnable;

(iii) $d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) ) < \infty .$

These conditions imply $d _ { \mathrm { N } } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) ) < \infty$ . Moreover, if $p = d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) ) < \infty$ , then

$$
m _ { \mathcal { H } } ^ { \mathrm { o n e } } ( \varepsilon , \delta ) \leq \left\lceil c \left( p + 1 \right) \varepsilon ^ { - \ell } \log ^ { 2 } ( 2 / \varepsilon ) \log ( 2 / \delta ) \right\rceil ,\tag{75}
$$

where c depends only on ℓ and $\gamma . \ I f p = 0$ , then $m _ { \mathcal { H } } ^ { \mathrm { o n e } } ( \varepsilon , \delta ) = 0$

Unlike the finite-label rate (55), the bound (75) does not depend on $| \mathcal { V } |$ , and it is linear in $p + 1$ . Call a full-set learner for $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ weak if its expected loss is below $1 / ( \ell + 1 )$ Theorem 55 is a consequence of the following conversion, which turns any weak learner into a one-correct-label learner whose sample size is linear in that of the weak learner.

Theorem 56 (Conversion of a weak full-set learner). Let $\mathcal { H } \subseteq ( \mathcal { A } \cup \{ \star \} ) ^ { \mathcal { X } }$ be nonempty. Let $M \geq 1$ and $0 \leq \alpha < 1 / ( \ell + 1 )$ , and suppose there is a full-set learner for $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ that uses M examples and whose randomized prediction has expected loss at most α for every target $g \in \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } )$ and every P with $P ( \operatorname { s u p p } ( g ) ) = 1$ . Then for every $N \geq 2 M$ there is a one-correct-label learner that uses $N - 1$ examples and whose randomized prediction $\widehat { p }$ satisfies, for every $h ^ { \star } \in { \mathcal { H } }$ , every P with $P ( \operatorname { s u p p } ( h ^ { \star } ) ) = 1$ , and every law D on $\mathcal { X } \times \mathcal { V }$ with marginal P and $\operatorname* { P r } _ { ( x , y ) \sim \mathcal { D } } [ y \in h ^ { \star } ( x ) ] = 1$ ，

$$
\mathbb { E } \big [ L _ { P , h ^ { \star } } ( \widehat { p } ) \big ] \leq \operatorname* { m i n } \left\{ 1 , \ c \left[ \frac { M \log ^ { 2 } ( e N / M ) } { N } \right] ^ { 1 / \ell } \right\} .\tag{76}
$$

Consequently,

$$
m _ { \mathcal { H } } ^ { \mathrm { o n e } } ( \varepsilon , \delta ) \leq \left\lceil c M \varepsilon ^ { - \ell } \log ^ { 2 } ( 2 / \varepsilon ) \log ( 2 / \delta ) \right\rceil .\tag{77}
$$

Here c depends only on ℓ and α.

The main idea is as follows. By Theorem 39, it sufices to construct prediction families with the intersection property whose size bound satisfies log $B _ { N } = O ( M \log ^ { 2 } ( e N / M ) )$ . The single cover of $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ used in Section 10 has size growing with |Y|, so we build a separate family $F _ { h , S }$ for each hypothesis h instead.

These families come from short descriptions. A short run of the boosting procedure of Section 12, followed by a few corrections, describes each $g \in \mathcal { T } _ { \ell + 1 } ( \mathcal { H } )$ on a pool of N inputs by ${ \cal O } ( M \log ( e N / M ) )$ ) pairs, each consisting of a position of the pool and a set of labels. The family $F _ { h , S }$ consists of the decoded predictors of all descriptions whose stored sets are nonempty subsets of the corresponding sets $h ( x _ { i } )$ ; there are at most $2 ^ { \ell } - 1$ such subsets at each position, whatever the size of Y. The description of $g = h _ { 1 } \cap \cdot \cdot \cdot \cap h _ { \ell + 1 }$ stores only the sets $g ( x _ { i } )$ , which are subsets of every $h _ { a } ( x _ { i } )$ , and this gives the intersection property. Finally, the count depends on N only through $N / M$ , so the sample size is linear in M.

## 13.1 Short Descriptions of Partial Intersections

We first encode each $g \in \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } )$ , on a pool of N inputs, by a short message, and decode the message to a predictor that is valid at every position of the pool where g is defined. Throughout, M, α and the learner are those of Theorem 56, and we apply the construction preceding Lemma 51 with $\mathcal { H } _ { \mathrm { a u x } } = \mathcal { T } _ { \ell + 1 } ( \mathcal { H } )$ . This gives a symmetric rule $\sigma \mapsto p _ { \sigma }$ and the constants $\eta , \zeta$ of (69), which depend only on ℓ and $\alpha$ . Recall that the lemma gives, after T rounds, T multisets of M examples whose averaged prediction is wrong at no more than $N e ^ { - \zeta T }$ positions of the pool, and that correcting these positions gives a predictor that is valid wherever the target is defined. Fix $N \geq 2 M$ and put

$$
\begin{array} { r } { T = \left\lceil \zeta ^ { - 1 } \log ( N / M ) \right\rceil . } \end{array}\tag{78}
$$

Since $N \geq 2 M$ , we have $T \geq 1$ and $N e ^ { - \zeta T } \leq M$ , so at most M corrections are needed.

Messages and the decoder. Fix an ordered pool ${ \cal S } = ( x _ { 1 } , \dots , x _ { N } )$ . A message is either the empty message or a tuple $( z _ { 1 } , \dots , z _ { T } ; Z )$ , where

• each $z _ { t }$ is a multiset of M pairs $( i , A ) \in [ N ] \times \mathcal { A }$ , and

$Z$ is a set of at most M such pairs with distinct positions $i .$

The decoder MsgDec maps the empty message to a fixed fallback predictor. It decodes a tuple $( z _ { 1 } , \dots , z _ { T } ; Z )$ in two steps. First, replacing every pair $( i , A )$ of $z _ { t }$ by the example $( x _ { i } , A )$ gives a multiset $\sigma _ { t }$ of $M$ examples, and $f _ { 0 } ( x )$ is a most probable label of $\textstyle T ^ { - 1 } \sum _ { t } p _ { \sigma _ { t } } ( \cdot \mid x )$ Second, the pairs of Z override $f _ { 0 } { \mathrm { : } }$ if $x = x _ { i }$ for some $( i , A ) \in Z$ , the decoder takes the least such i and returns the least label of its set A in the fixed enumeration of $\mathcal { V }$ ; otherwise it returns $f _ { 0 } ( x )$ . Thus every message decodes to a total predictor.

The message of a partial intersection. Let $g \in \mathcal { T } _ { \ell + 1 } ( \mathcal { H } )$ and $D = \{ i \in [ N ] : x _ { i } \in$ $\operatorname { s u p p } ( g ) \}$ . If D is empty, the message of $g$ is the empty message. Otherwise Lemma 51 with $T$ rounds gives multisets $\sigma _ { 1 } , \dots , \sigma _ { T }$ of M examples $( x _ { i } , g ( x _ { i } ) ) , i \in D$ , and a set $Q \subseteq D$ of at most $N e ^ { - \zeta T } \leq M$ positions, such that the corrected predictor $f$ of that lemma is valid at every position of $D$ . The message writes these examples in terms of positions: $z _ { t }$ is obtained from $\sigma _ { t }$ by replacing each example $( x , g ( x ) )$ by the pair $( i , g ( x _ { i } ) )$ , where i is the least position in D with $x _ { i } = x$ . With $Z = \{ ( i , g ( x _ { i } ) ) : i \in Q \}$ , the message of $g$ is $( z _ { 1 } , \dots , z _ { T } ; Z )$

This message decodes to $f .$ Indeed, the decoder turns each pair $( i , g ( x _ { i } ) )$ of $z _ { t }$ back into the example $( x _ { i } , g ( x _ { i } ) ) = ( x , g ( x ) )$ , so it recovers $\sigma _ { 1 } , \dots , \sigma _ { T }$ and hence $f _ { 0 }$ . Since $g$ is a function of the input, all positions of $Q$ with a given input $x _ { i }$ carry the same set $g ( x _ { i } )$ so at such an input the decoder returns the least label of $g ( x _ { i } )$ , as $f$ does; at every other input both predictors equal $f _ { 0 }$ . We record for later that every pair $( i , A )$ in the message of $g$ satisfies $A = g ( x _ { i } )$ .

## 13.2 Message Families of Single Hypotheses

Let $h \in { \mathcal { H } }$ be defined on the pool, that is, $x _ { i } \in \operatorname { s u p p } ( h )$ for all $i \in [ N ]$ . Let $\mathcal { M } _ { h , N }$ consist of the empty message and of all messages $( z _ { 1 } , \dots , z _ { T } ; Z )$ in which every pair $( i , A )$ satisfies $\emptyset \neq A \subseteq h ( x _ { i } )$ . Since Definition 38 uses functions on supp(S), we restrict the decoded predictors to the pool and set

$$
F _ { h , S } = \big \{ \mathrm { M s g D e c } _ { S } ( z ) | _ { \mathrm { s u p p } ( S ) } : z \in \mathcal { M } _ { h , N } \big \} .
$$

Allowing subsets of $h ( x _ { i } )$ , rather than $h ( x _ { i } )$ itself, is what makes the message of an intersection admissible for every constituent hypothesis.

Put $b _ { \ell } = 2 ^ { \ell } - 1$ . Since $| h ( x _ { i } ) | \leq \ell ;$ , at most $b _ { \ell }$ sets A are admissible at each position, so at most $N b _ { \ell }$ pairs are admissible. Counting the multisets of M admissible pairs for each $z _ { t } .$ , and the sets of $j \leq M$ admissible pairs with distinct positions for $Z _ { i }$ gives

$$
| F _ { h , S } | \leq | \mathcal { M } _ { h , N } | \leq B _ { N } : = 1 + \binom { N b _ { \ell } + M - 1 } { M } ^ { T } \sum _ { j = 0 } ^ { M } \binom { N } { j } b _ { \ell } ^ { j } .\tag{79}
$$

The elementary estimates $\binom { N b _ { \ell } + M - 1 } { M } \leq ( e ( b _ { \ell } + 1 ) N / M ) ^ { M }$ and $\begin{array} { r } { \sum _ { j \leq M } \binom { N } { j } b _ { \ell } ^ { j } \leq \sum _ { j \leq M } \binom { N b _ { \ell } } { j } \leq } \end{array}$

$( e N b _ { \ell } / M ) ^ { M }$ , together with $T \leq c \log ( e N / M )$ from (78) and $N \geq 2 M$ , give

$$
\log B _ { N } \leq c M \log ^ { 2 } ( e N / M ) .\tag{80}
$$

## 13.3 Proof of the Conversion Theorem

Proof of Theorem 56. Let S be a multiset of N inputs, and list its elements, with multiplicity, in the order of a fixed enumeration of the countable set X. This gives an ordered pool determined by $S ,$ and we use the families $F _ { h , S }$ of Section 13.2 for this pool. By (79), they have size bound $B _ { N }$

The intersection property. Let $h _ { 1 } , \ldots , h _ { \ell + 1 } \in \mathcal { H }$ be defined on S, and let B be the set of $x \in \operatorname { s u p p } ( S )$ with $\textstyle \bigcap _ { a } h _ { a } ( x ) \neq \emptyset$ . Let $\begin{array} { r } { g ( x ) = \bigcap _ { a } h _ { a } ( x ) } \end{array}$ for $x \in B$ and $g ( x ) = \star$ otherwise. Then $g \in \mathcal { T } _ { \ell + 1 } ( \mathcal { H } )$ , since this class is closed under domain restrictions. Let z be the message of g. Every pair $( i , A )$ in z has $A = g ( x _ { i } )$ , a nonempty subset of every $h _ { a } ( x _ { i } )$ , so $z \in \mathcal { M } _ { h _ { a } , N }$ for every a (this is trivial if z is empty). Hence $f = \mathrm { M s g D e c } _ { S } ( z ) | _ { \mathrm { s u p p } ( S ) }$ lies in $\bigcap _ { a } F _ { h _ { a } , S } .$ , and $\begin{array} { r } { f ( x ) \in g ( x ) = \bigcap _ { a } h _ { a } ( x ) } \end{array}$ for every $x \in B$ . This is the intersection property of Definition 38.

Expected loss. By Theorem 39 there is a learner with $N - 1$ examples whose expected loss is at most

$$
\operatorname* { m i n } \left\{ 1 , \ 2 \left( \frac { ( \ell + 1 ) ( \log B _ { N } + \log N ) } { \lfloor N / 2 \rfloor } \right) ^ { 1 / \ell } + \frac { 1 } { N } \right\} .
$$

We have log $N \leq M \log ( e N / M ) \leq M \log ^ { 2 } ( e N / M )$ , since $z \mapsto z \log ( e N / z )$ is nondecreasing on [1, N] and exceeds log N at $z ~ = ~ 1$ , and since log $( e N / M ) \ge 1$ . Together with (80) and $\vert N / 2 \vert \ge N / 3$ , this bounds the term with the root by $c [ M \log ^ { 2 } ( e N / M ) / N ] ^ { 1 / \ell }$ . Since $M \log ^ { 2 } ( e \bar { N } / M ) / N \geq 1 / N$ , the additive term $1 / N$ is absorbed by increasing c. This proves (76).

Confidence. Lemma 40 needs expected loss at most $\varepsilon / [ 3 2 ( \ell + 1 ) ^ { 2 } ]$ , and we choose N to achieve this. Let $a = \log ( 2 / \varepsilon )$ , and use $n = \lceil C M \varepsilon ^ { - \ell } a ^ { 2 } \rceil$ examples, so that $N = n + 1$ For $C \geq 6$ we have $N / M \ge x _ { 0 } : = C \varepsilon ^ { - \ell } a ^ { 2 } \ge e$ , and in particular $N \geq 2 M$ . Moreover, $\log ( e x _ { 0 } ) = 1 + \log C + \ell \log ( 1 / \varepsilon ) + 2 \log a \leq c ^ { \prime } ( 1 + \log C ) a .$ , where $c ^ { \prime }$ depends only on ℓ. Since $x \mapsto \log ^ { 2 } ( e x ) / x$ is nonincreasing for $x \geq e$ , the expression in brackets in (76) therefore satisfies

$$
\frac { M \log ^ { 2 } ( e N / M ) } { N } \leq \frac { \log ^ { 2 } ( e x _ { 0 } ) } { x _ { 0 } } \leq \frac { c ^ { \prime 2 } ( 1 + \log C ) ^ { 2 } \varepsilon ^ { \ell } } { C } .
$$

Choosing C large enough in terms of ℓ and α therefore makes the expected loss of the nexample learner at most $\varepsilon / [ 3 2 ( \ell + 1 ) ^ { 2 } ]$ . Lemma 40 now gives a learner with $n [ 2 4 ( \ell { + } 1 ) \log ( 2 / \delta ) ]$ examples, which proves (77). □

## 13.4 The Feedback Equivalence and the Packed Rate

Theorem 56 needs only a black-box weak learner, so it already gives the equivalence of the two feedback models. For the packed rate we feed it the learner of Lemma 50, which is weak already with $O ( p + 1 )$ examples.

Corollary 57 (One-correct-label versus full-set feedback). Let $\mathcal { H } \subseteq ( \mathcal { A } \cup \{ \star \} ) ^ { \mathcal { X } }$ be nonempty. Then H is realizably one-correct-label PAC learnable if and only if $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ is realizably full-set $P A C$ learnable. Moreover, let

$$
M = \operatorname* { m a x } \left\{ 1 , \ m _ { \mathcal { I } _ { \ell + 1 } ( \mathcal { H } ) } ^ { \mathrm { f u l l } } \left( \frac { 1 } { 8 ( \ell + 1 ) } , \frac { 1 } { 8 ( \ell + 1 ) } \right) \right\} .
$$

If $M < \infty ,$ then

$$
m _ { \mathcal { H } } ^ { \mathrm { o n e } } ( \varepsilon , \delta ) \leq c M \varepsilon ^ { - \ell } \log ^ { 2 } ( 2 / \varepsilon ) \log ( 2 / \delta ) ,\tag{81}
$$

where c depends only on ℓ.

Proof. By Lemma 37 with $R = \ell + 1$ , one-correct-label learnability of H implies full-set learnability of $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ . Conversely, if $M < \infty$ , a full-set PAC learner for $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ with accuracy and confidence parameters $1 / [ 8 ( \ell + 1 ) ]$ uses M examples and, by (4), has expected loss at most $1 / [ 4 ( \ell + 1 ) ]$ . Theorem 56 with $\alpha = 1 / [ 4 ( \ell + 1 ) ]$ gives (81), and in particular one-correct-label learnability. □

Proof of Theorem 55. Corollary 57 shows that (i) and (ii) are equivalent, and Theorem 45, applied to the class $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ , shows that (ii) and (iii) are equivalent. Since $d _ { \mathrm { N } } ^ { \cap } \leq d _ { \mathrm { p a c k } , \gamma } ^ { \cap }$ by Lemma 9, (iii) implies $d _ { \mathrm { N } } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) ) < \infty$

For the rate, let $p = d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) ) < \infty$ and put

$$
\alpha = \frac { 1 } { 2 } \Bigl ( \gamma + \frac { 1 } { \ell + 1 } \Bigr ) , \qquad M = \operatorname* { m a x } \left\{ 1 , \left\lceil \frac { p } { \alpha - \gamma } \right\rceil \right\} .
$$

The learner of Lemma 50 for $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ with M examples has expected loss at most $\gamma + p / ( M +$ $1 ) \leq \alpha < 1 / ( \ell + 1 )$ . Since $M \leq c ( p + 1 )$ and α depends only on ℓ and $\gamma , ~ ( 7 7 )$ gives (75).

Finally, let $p = 0$ . Then $d _ { \mathrm { N } } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) ) = 0$ by Lemma 9. The argument for the case $d = 0$ in the proof of Theorem 41 uses only this fact and Lemma 8, so it applies to every countable label space. It shows that at every input x the sets $h ( x )$ , over the $h \in \mathcal H$ defined at x, share a label. Predicting the first such label requires no examples and is valid on $\operatorname { s u p p } ( h ^ { \star } )$ for every $h ^ { \star } \in { \mathcal { H } }$ , so $m _ { \mathcal { H } } ^ { \mathrm { o n e } } ( \varepsilon , \delta ) = 0$ □

Corollary 58 (Every fixed intersection depth). If H is realizably one-correct-label PAC learnable, then $\mathcal { T } _ { R } ( \mathcal { H } )$ is realizably full-set PAC learnable for every $R \geq 1$ . For $R \geq \ell + 1$ the converse also holds. Moreover, if $p = d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) ) < \infty$ , then for every $R \geq 1$ and $\beta > 0$ there is a full-set learner for $\mathcal { T } _ { R } ( \mathcal { H } )$ with expected loss at most $\beta$ that uses at most $c \left( p + 1 \right)$ examples, where c depends only on ℓ, R, γ and $\beta$

Proof. The first assertion is Lemma 37. For $R \geq \ell + 1$ we have ${ \mathcal { T } } _ { \ell + 1 } ( { \mathcal { H } } ) \subseteq { \mathcal { T } } _ { R } ( { \mathcal { H } } )$ , so a full-set learner for $\mathcal { T } _ { R } ( \mathcal { H } )$ is also one for $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ , and Corollary 57 gives the converse. For the last assertion we may assume $\beta < 1$ . By Lemma 37 and by (75) with $\varepsilon = \delta = \beta / ( 2 R )$ , we have $m _ { \mathcal { T } _ { R } ( \mathcal { H } ) } ^ { \mathrm { f u l l } } ( \beta / 2 , \beta / 2 ) \le m _ { \mathcal { H } } ^ { \mathrm { o n e } } ( \beta / ( 2 R ) , \beta / ( 2 R ) ) \le c ^ { \prime } ( p + 1 )$ , where $c ^ { \prime }$ depends only on $\ell , R , \gamma$ and $\beta .$ The corresponding learner has expected loss at most $\beta$ by (4). □

Remark 59 (Intersection depth and the accuracy exponent). The characterization involves intersections of $\ell + 1$ hypotheses, while the rate has exponent ℓ. Both come from the same step in the proof of Theorem 39: the correct-label set of a first candidate has at most ℓ labels, and each of ℓ further independent candidates may exclude one of them. This gives the ℓth power in (51), and hence the factor $\varepsilon ^ { - \ell }$

## 13.5 Joint DS and Disjoint-Neighbor Bounds

We end the section with suficient conditions in terms of the joint DS dimension and the disjoint-neighbor dimension of $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ . They follow from Theorem 55 and the following comparison, proved in Appendix B, which bounds the packed dimension of $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ by either of these dimensions up to a logarithmic factor.

Theorem 60 (Comparison for partial intersection classes). Let $\mathcal { H } \subseteq ( \mathcal { A } \cup \{ \star \} ) ^ { \mathcal { X } }$ be nonempty, and fix $0 < \rho \leq 1$ . Let d be either $d _ { \mathrm { D S } } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) )$ or $d _ { \mathrm { p a i r } } ( \mathcal { T } _ { \ell + 1 } ( \mathcal { H } ) )$ . There is a constant $^ { c , }$ depending only on ℓ and $\rho ,$ , such that if $d < \infty$ , then

$$
\begin{array} { r } { d _ { \mathrm { p a c k } , \rho } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) ) \leq c d \log ( d + 2 ) . } \end{array}\tag{82}
$$

The bound holds for arbitrary sets X and Y, countable or not.

Corollary 61 (A joint DS or disjoint-neighbor bound). Let d be either $d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { T } _ { \ell + 1 } ( \mathcal { H } ) )$ or $d _ { \mathrm { p a i r } } ( \mathcal { T } _ { \ell + 1 } ( \mathcal { H } ) )$ . If $d < \infty$ , then H is realizably one-correct-label PAC learnable and

$$
m _ { \mathcal { H } } ^ { \mathrm { o n e } } ( \varepsilon , \delta ) \leq \left\lceil c \left( d + 1 \right) \log ( d + 2 ) \varepsilon ^ { - \ell } \log ^ { 2 } ( 2 / \varepsilon ) \log ( 2 / \delta ) \right\rceil ,\tag{83}
$$

where c depends only on ℓ. If $d = 0$ , then m $\imath _ { \mathcal { H } } ^ { \mathrm { o n e } } ( \varepsilon , \delta ) = 0$

Proof. Put $\rho _ { 0 } = 1 / [ 8 ( \ell + 1 ) ]$ , which satisfies (9). By Theorem 60 with $\rho = \rho _ { 0 }$ 2

$$
p _ { 0 } = d _ { \mathrm { p a c k } , \rho _ { 0 } } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) ) \leq c d \log ( d + 2 ) .
$$

Hence $p _ { 0 } + 1 \leq c ( d + 1 ) \log ( d + 2 )$ after enlarging $c ,$ and Theorem 55 with $\gamma = \rho _ { 0 }$ gives (75) with $p = p _ { 0 }$ . This proves (83). If $d = 0$ , then $p _ { 0 } = 0$ , and Theorem 55 gives $m _ { \mathcal { H } } ^ { \mathrm { o n e } } ( \varepsilon , \delta ) = 0$ .

## 14 Agnostic Learning from One Correct Label

Recall the agnostic model from Section 5.2. Examples $( x , A )$ are drawn from an arbitrary distribution D on $\mathcal { X } \times \mathcal { A }$ , but the learner observes only x and a label $y \in A$ chosen by the demonstrator, a conditional law of y given $( x , A )$ that is supported on A. The learner competes with the benchmark $L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } )$ of (5).

In the realizable setting, learnability from one correct label is governed by the intersection class $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ (Theorems B and E). In the agnostic setting it is not, because a demonstration does not reveal the set it came from: Section 14.2 gives a class whose intersection class has all four dimensions equal to one and which is still not learnable. For nondegenerate classes we show that learnability is characterized by a local compatibility condition together with finiteness of the DS dimension of the collapsed class, an ordinary multiclass class obtained by replacing every correct-label set with two or more labels by one new label. For $\ell = 1$ this recovers the characterization of agnostic learnability of partial concept classes [AHHM21] (Remark 83).

We state the characterization in Section 14.1 and show in Section 14.2 that nondegeneracy cannot be dropped; after some notation (Section 14.3), we prove necessity (Section 14.4), give learners for the two regions of the input space (Sections 14.5 and 14.6), and glue them together (Section 14.7).

## 14.1 Nondegenerate Classes and the Characterization

In binary classification, the two constant hypotheses disagree at every input. Our standing assumption asks for two hypotheses that play the same role.

Definition 62 (Nondegenerate classes). A class H is nondegenerate if it contains two total hypotheses $g _ { 0 } , g _ { 1 }$ with singleton values that difer at every input, that is, there are $b _ { 0 } , b _ { 1 } : \mathcal { X }  \mathcal { Y }$ with

$$
g _ { j } ( x ) = \{ b _ { j } ( x ) \} \quad ( j \in \{ 0 , 1 \} ) \qquad { \mathrm { a n d } } \qquad b _ { 0 } ( x ) \neq b _ { 1 } ( x ) \qquad ( x \in \mathcal { X } ) .
$$

Otherwise H is degenerate.

For instance, every class that contains two distinct constant singleton hypotheses is nondegenerate, regardless of its other members, which may be partial.

The characterization uses the following multiclass class. Fix a new label $\circ \notin \mathcal { V }$ . For $h \in { \mathcal { H } }$ , define

$$
\begin{array} { r } { \widetilde { h } ( x ) = \left\{ \begin{array} { l l } { a , } & { h ( x ) = \{ a \} , } \\ { \diamond , } & { h ( x ) \neq \star \mathrm { ~ a n d ~ } | h ( x ) | \geq 2 , } \\ { \star , } & { h ( x ) = \star , } \end{array} \right. \qquad \widetilde { \mathcal { H } } = \{ \widetilde { h } : h \in \mathcal { H } \} . } \end{array}\tag{84}
$$

The collapsed class $\widetilde { \mathcal { H } }$ keeps singleton and undefined values and turns every correct-label set with two or more labels into the label $\diamond ,$ which counts as defined. Reading each value a as the singleton $\{ a \}$ , we view $\widetilde { \mathcal { H } }$ as a class with $\ell = 1$ over the countable label space $\mathcal { V } \cup \{ \circ \}$ , with Natarajan and DS dimensions $d _ { \mathrm { N } } ^ { \cap } ( \widetilde { \mathcal { H } } )$ and $d _ { \mathrm { D S } } ^ { \cap } ( \widetilde { \mathcal { H } } )$ as in Section 5.4.

The collapse forgets which labels a large correct-label set contains. The next condition makes this information unnecessary.

Definition 63 (Local compatibility). A class H is locally compatible if all h, $h ^ { \prime } \in \mathcal { H }$ and $x \in \operatorname { s u p p } ( h ) \cap \operatorname { s u p p } ( h ^ { \prime } )$ such that $| h ( x ) | = 1$ and $| h ^ { \prime } ( x ) | \geq 2$ satisfy $h ( x ) \subseteq h ^ { \prime } ( x )$

In words, at every input, each correct-label set with two or more labels contains all singleton values of the class at that input. The condition is needed because demonstrations do not tell the learner whether a label came from a singleton or from a larger set. Under local compatibility, a singleton value of the class at x is a safe prediction whenever the hidden set at x is any value of the class with two or more labels, so the learner never needs to know which one.

Our main result in this section is the following characterization.

Theorem 64 (Agnostic learning from one correct label). Let X and Y be countable, and let $\mathcal { H } \subseteq ( \mathcal { A } \cup \{ \star \} ) ^ { \mathcal { X } }$ be nondegenerate and locally compatible. Then H is agnostically PAC learnable from one correct label if and only if

$$
d _ { \mathrm { D S } } ^ { \cap } ( \widetilde { \mathcal { H } } ) < \infty .
$$

Assuming local compatibility costs nothing, because it is necessary for every class.

Proposition 65 (Local compatibility is necessary). Let H be a class that is not locally compatible. For every m and every learner that uses m one-correct-label examples, there are a distribution D on $\mathcal { X } \times \mathcal { A }$ and a demonstrator such that the learner’s output $\widehat g$ satisfies

$$
\mathbb { E } \big [ L _ { \mathcal { D } } ( \widehat { g } ) \big ] \geq L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } ) + \frac { 1 } { 1 5 } .
$$

In particular, H is not agnostically PAC learnable from one correct label.

Together, Theorem 64 and Proposition 65 give the first part of Theorem F: a nondegenerate class is agnostically PAC learnable from one correct label if and only if it is locally compatible and $d _ { \mathrm { D S } } ^ { \Omega } ( \mathcal { \tilde { H } } ) < \infty$

Both directions of Theorem 64 treat two regions of the input space separately. The mixed region ${ \mathcal { X } } _ { \mathrm { m i x } }$ consists of the inputs at which some hypothesis has a correct-label set with at least two labels, and the pure region ${ \mathcal { X } } _ { \mathrm { p u r e } }$ is the rest. On the pure region every defined value is a singleton, and an agnostic multiclass learner sufices (Section 14.5). On the mixed region the learner predicts only singleton values of the class, which local compatibility makes safe, and runs specialists [FSSW97] over a finite cover of the collapsed class (Section 14.6). Section 14.7 glues the two learners, and necessity follows from a multiclass lower bound on the pure region and an indistinguishability argument on the mixed region (Section 14.4).

A binary reformulation. When every input belongs to the mixed region, the condition can be stated with a partial binary class on $\mathcal { X } \times \mathcal { V }$ . For $h \in { \mathcal { H } }$ define

$$
J _ { h } ( x , a ) = \left\{ { \begin{array} { l l } { 1 , } & { h ( x ) = \{ a \} , } \\ { 0 , } & { h ( x ) \neq \star { \mathrm { ~ a n d ~ } } h ( x ) \neq \{ a \} , } \\ { \star , } & { h ( x ) = \star , } \end{array} } \right. \qquad { \mathcal { I } } = \{ J _ { h } : h \in { \mathcal { H } } \} .
$$

As for partial concept classes [AHHM21], $\operatorname { V C } ( \mathcal { T } )$ is the largest n for which some n points of $\mathcal { X } \times \mathcal { V }$ are shattered by the members of J that are defined at all of them.

Corollary 66 (A VC criterion). Let X and Y be countable, let H be nondegenerate and locally compatible, and suppose that at every input some member of H has a correct-label set with at least two labels. Then H is agnostically PAC learnable from one correct label if and only if $\mathrm { V C } ( \mathcal { I } ) < \infty$

The proof is in Section 14.7. Undefined values must stay undefined in $J _ { h }$ (Remark 84).

## 14.2 The Degenerate Obstruction

We now show that nondegeneracy cannot be dropped from Theorem 64, and that the dimensions of $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ do not determine agnostic learnability. The reason is statistical: diferent distributions of correct-label sets can induce the same law of demonstrations, and hence the same law of the learner’s output, while a prediction that is good under one of them is bad under another. The following quantitative form of the second part of Theorem F adapts the indistinguishability argument of [PMBD26], given there for online learning with multiple correct answers, to the i.i.d. setting and the benchmark $L _ { \mathcal { D } } ^ { \mathrm { s e t } }$ of (5).

Theorem 67 (A degenerate class that is not agnostically learnable). Let $\ell = 2$ . There is a class H on a one-point input space with $| \mathcal { V } | = 3$ , in which every correct-label set has size two, that is not agnostically PAC learnable from one correct label: for every sample size and every learner there are a law D on $\mathcal { X } \times \mathcal { A }$ and a demonstrator such that the learner’s output $\widehat g$ satisfies

$$
\mathbb { E } \big [ L _ { \mathcal { D } } ( \widehat { g } ) \big ] \geq L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } ) + \frac 1 9 \qquad a n d \qquad \operatorname* { P r } \Big [ L _ { \mathcal { D } } ( \widehat { g } ) > L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } ) + \frac 1 { 1 8 } \Big ] \geq \frac 1 5 .
$$

Moreover, H is degenerate and locally compatible, $d _ { \mathrm { D S } } ^ { \cap } ( \widetilde { \mathcal { H } } ) = 0$ , and

$$
d _ { \mathrm { N } } ^ { \cap } ( \mathcal { Z } _ { 3 } ( \mathcal { H } ) ) = d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { Z } _ { 3 } ( \mathcal { H } ) ) = d _ { \mathrm { p a i r } } ( \mathcal { Z } _ { 3 } ( \mathcal { H } ) ) = d _ { \mathrm { p a c k } , \rho } ^ { \cap } ( \mathcal { Z } _ { 3 } ( \mathcal { H } ) ) = 1 \qquad ( 0 < \rho \leq 1 ) .
$$

Proof. Let $\mathcal { X } = \{ x \} , \mathcal { Y } = \{ 1 , 2 , 3 \}$ , and $\mathcal { H } = \{ h _ { 1 } , h _ { 2 } , h _ { 3 } \}$ with $h _ { i } ( x ) = \mathcal { y } \backslash \{ i \}$ . We construct three distributions under which the demonstrated label is uniform on $\mathcal { V } _ { : }$ while under the ith distribution the label i is the worst prediction. For $i \in [ 3 ]$ , let $\mathcal { D } _ { i }$ assign probability $2 / 3$ to $( x , y \setminus \{ i \} )$ and probability $1 / 3$ to $( x , \{ i \} )$ , and let the demonstrated label be a uniformly random element of the sampled set A. Under every $\mathcal { D } _ { i }$ the demonstrated label is indeed uniform: label i is demonstrated with probability $1 / 3$ , and each other label with probability $( 2 / 3 ) ( 1 / 2 ) = 1 / 3$

Thus the law of the demonstrated sample, and hence of the learner’s output, is the same under $\mathcal { D } _ { 1 } , \mathcal { D } _ { 2 }$ and $\mathcal { D } _ { 3 }$ . Let $p _ { j }$ be the probability that the learner predicts $j$ at $x ,$ over the sample and the seed. Since $p _ { 1 } + p _ { 2 } + p _ { 3 } = 1$ , there is $i ^ { \star }$ with $p _ { i ^ { \star } } \geq 1 / 3$ . Under $\mathcal { D } _ { i ^ { \star } }$ , predicting $i ^ { \star }$ has loss $2 / 3$ and predicting either other label has loss $1 / 3 { \mathrm { , } }$ , so

$$
\mathbb { E } \big [ L _ { \mathcal { D } _ { i ^ { \star } } } ( \widehat { g } ) \big ] = \frac { 2 } { 3 } p _ { i ^ { \star } } + \frac { 1 } { 3 } ( 1 - p _ { i ^ { \star } } ) = \frac { 1 } { 3 } + \frac { p _ { i ^ { \star } } } { 3 } \geq \frac { 4 } { 9 } .
$$

On the other hand, $h _ { i }$ ⋆ disagrees with the sampled set exactly when $A = \left\{ i ^ { \star } \right\}$ , which has probability $1 / 3$ , and the other two hypotheses always disagree with it. Hence $L _ { \mathcal { D } _ { i ^ { \star } } } ^ { \mathrm { s e t } } ( \mathcal { H } ) = 1 / 3$ which proves the bound in expectation.

For the bound in probability, condition on the sample and the seed, and let $r$ be the probability that the resulting prediction rule assigns to $i ^ { \star }$ at x $( \mathrm { s o } r \in \{ 0 , 1 \}$ for a deterministic predictor). Its excess loss under $\mathcal { D } _ { i } ,$ ⋆ is $Z = r / 3 \in [ 0 , 1 / 3 ]$ , and $\mathbb { E } [ Z ] \ge 1 / 9$ by the computation above. If $u = \mathrm { P r } [ Z > 1 / 1 8 ]$ , then

$$
{ \frac { 1 } { 9 } } \leq \mathbb { E } [ Z ] \leq { \frac { 1 } { 1 8 } } ( 1 - u ) + { \frac { 1 } { 3 } } u ,
$$

so $u \geq 1 / 5$

It remains to compute the dimensions. For $i \neq j$ the set $h _ { i } ( x ) \cap h _ { j } ( x )$ is a singleton, so $\mathcal { T } _ { 3 } ( \mathcal { H } )$ contains the three hypotheses with singleton values at x. Any two of them form a witness of size one for each of the four dimensions; in the packed witness the single block covers both indices of the list, so this works for every $0 < \rho \leq 1$ . Since there is only one input, no witness has size larger than one. Finally, no member of H has a singleton value, so H is degenerate and locally compatible. The collapsed class consists of the single hypothesis $x \mapsto \diamond$ . Its only pattern has no neighbor, so $d _ { \mathrm { D S } } ^ { \Omega } ( \tilde { \mathcal { H } } ) = 0$ □

The obstruction is specific to the agnostic setting: in the realizable case every demonstrated label is correct at the only input, so predicting it has zero loss. Characterizing agnostic learnability from one correct label for degenerate classes is an interesting open direction.

Remark 68 (The intersection class does not decide agnostic learnability). Let $\mathcal { H } _ { 0 }$ consist of the three hypotheses $x \mapsto \{ j \}$ on $\mathcal { X } = \{ x \} , \mathcal { Y } = \{ 1 , 2 , 3 \}$ . Every member of $\mathcal { T } _ { 3 } ( \mathcal { H } _ { 0 } )$ defined at x belongs to $\mathcal { H } _ { 0 }$ , so, as in the proof above, all four dimensions of $\mathcal { T } _ { 3 } ( \mathcal { H } _ { 0 } )$ equal one. Yet $\mathcal { H } _ { 0 }$ is agnostically learnable from one correct label, because the most frequently demonstrated label competes with the best constant. Indeed, let $t _ { j }$ be the probability that $j$ is demonstrated and $s _ { j } = \operatorname* { P r } [ A = \{ j \} ]$ ]. Every demonstrated label lies in $A _ { i }$ so $t _ { j } \geq s _ { j }$ , and the constant prediction $j$ has loss at most $1 - t _ { j }$ . By Hoefding’s inequality and a union bound over the three labels, with probability at least $1 - \delta$ the most frequent demonstrated label among ${ \cal O } ( \varepsilon ^ { - 2 } \log ( 1 / \delta ) )$ ) examples has loss at most $\begin{array} { r } { 1 - \operatorname* { m a x } _ { j } t _ { j } + \varepsilon \leq 1 - \operatorname* { m a x } _ { j } s _ { j } + \varepsilon = L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } _ { 0 } ) + \varepsilon } \end{array}$ . This agrees with Theorem 64: the class $\mathcal { H } _ { 0 }$ is nondegenerate and locally compatible, and $d _ { \mathrm { D S } } ^ { \Omega } ( \widetilde { \mathcal { H } } _ { 0 } ) = 1$

## 14.3 Regions and Local Alphabets

This subsection collects the notation and the combinatorial facts used in both directions of the proof of Theorem 64.

The two regions. We write $| h ( x ) | \geq 2$ to mean that h is defined at x and $h ( x )$ has at least two labels. The mixed region and the pure region are

$$
{ \mathcal { X } } _ { \operatorname* { m i x } } = { \big \{ } x \in { \mathcal { X } } : ~ | h ( x ) | \geq 2 ~ { \mathrm { f o r ~ s o m e } } ~ h \in { \mathcal { H } } { \big \} } , \qquad { \mathcal { X } } _ { \operatorname { p u r e } } = { \mathcal { X } } ~ { \backslash } ~ { \mathcal { X } } _ { \operatorname* { m i x } } .\tag{85}
$$

On the pure region every defined value of every hypothesis is a singleton, so $\widetilde { h } ( x ) = a$ means $h ( x ) = \{ a \}$ there, and the label ⋄ occurs only on the mixed region.

Local alphabets. The local alphabet $\mathcal { \partial } _ { x }$ of an input x is the set of labels a with $h ( x ) = \{ a \}$ for some $h \in { \mathcal { H } }$ . If H is nondegenerate and locally compatible, $x \in \mathcal { X } _ { \operatorname* { m i x } }$ , and $| h ( x ) | \geq 2$ then

$$
\{ b _ { 0 } ( x ) , b _ { 1 } ( x ) \} \subseteq { \mathcal { V } } _ { x } \subseteq h ( x ) ,\tag{86}
$$

where the first inclusion holds because $g _ { 0 } , g _ { 1 } \in \mathcal { H }$ , and the second is local compatibility. Hence every label of $\mathcal { \partial } _ { x }$ is correct whenever the hidden set is a value of a class member with at least two labels, and $2 \leq | \mathcal { V } _ { x } | \leq \ell$ on the mixed region. In particular, the collapsed values at a mixed input lie in $\mathcal { \partial } _ { x } \cup \{ \diamond \}$ , a set of at most $\ell + 1$ labels; this bound uses only local compatibility.

Restrictions and DS witnesses. For $Z \subseteq { \mathcal { X } }$ , the restriction of a class to $Z$ consists of the hypotheses that agree with a member of the class on $Z$ and are undefined outside Z. The collapsed class and its restrictions have singleton values. For such a class we identify a set $\{ a \}$ with the label a, so patterns are tuples of labels, and a DS witness is a pseudo-cube in the sense of [BCD<sup>+</sup>22]: a finite set V of patterns in which every $v \in V$ has, in every direction $i ,$ a neighbor in V that difers from v exactly at coordinate i. We need two simple facts about such classes.

Lemma 69 (Splitting and size of DS witnesses). Let C be a class with singleton values, and let $\mathcal { X } = Z _ { 1 } \cup Z _ { 2 }$ with $Z _ { 1 } \cap Z _ { 2 } = \emptyset$ . Let $\mathcal { C } _ { 1 }$ and $\mathcal { C } _ { 2 }$ be the restrictions of C to $Z _ { 1 }$ and $Z _ { 2 }$ . Then

$$
\begin{array} { r } { \operatorname* { m a x } \{ d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { C } _ { 1 } ) , d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { C } _ { 2 } ) \} \leq d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { C } ) \leq d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { C } _ { 1 } ) + d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { C } _ { 2 } ) . } \end{array}
$$

Moreover, every DS witness of C on n inputs has at least $2 ^ { n }$ patterns.

Proof. The patterns of $\mathcal { C } _ { j }$ on a sequence of inputs from $Z _ { j }$ are exactly those of ${ \mathcal { C } } _ { : }$ which gives the first inequality. For the second, let V be a DS witness of $\mathcal { C }$ on a sequence $S$ , and let $I _ { j }$ be the set of coordinates of the inputs of S that lie in $Z _ { j }$ . By Lemma 9, if $I _ { j }$ is nonempty then $V | _ { I _ { j } }$ is a DS witness on the corresponding subsequence, and its patterns are patterns of $\mathcal { C } _ { j }$ Hence $| I _ { j } | \leq d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { C } _ { j } )$ for $j = 1 , 2$

For the size, we induct on n; a witness on zero inputs is nonempty. Split a witness V on $n \geq 1$ inputs according to the label at the last coordinate. Every nonempty part, projected to the first $n - 1$ coordinates, is a DS witness, because neighbors in the first $n - 1$ directions have the same last label and so lie in the same part. There are at least two nonempty parts, since every pattern has a neighbor in the last direction. By induction, $| V | \geq 2 \cdot 2 ^ { n - 1 } . \qquad \bigsqcup$

When only boundedly many labels occur at each input, as on the mixed region, Natarajan’s lemma [Nat89] bounds the number of patterns, and the Natarajan and DS dimensions are finite together.

Lemma 70 (Bounded local alphabets). Let C be a class with singleton values such that, at every input x, the defined values $c ( x ) , c \in { \mathcal { C } }$ , lie in a set of at most $\ell + 1$ labels. $I f d _ { \mathrm { N } } ^ { \cap } ( { \mathcal { C } } ) \leq d ;$ then

$$
| { \mathcal { C } } | _ { S } | \leq n ^ { d } ( \ell + 1 ) ^ { 2 d }
$$

for every sequence $S ~ o f n \geq 1$ distinct inputs. Consequently, $d _ { \mathrm { N } } ^ { \cap } ( { \mathcal { C } } ) < \infty$ if and only if $d _ { \mathrm { D S } } ^ { \Omega } ( { \mathcal C } ) < \infty$

Proof. Map the possible labels at each input of S injectively into $[ \ell + 1 ]$ . This turns ${ \mathcal { C } } | _ { S }$ into a set of functions from $[ n ]$ to $[ \ell + 1 ]$ , and its Natarajan dimension is at most $d ,$ since a productshattered set of coordinates is a subsequence of S product-shattered by C. Natarajan’s lemma [Nat89] bounds the size of such a set by $n ^ { d } ( \ell + 1 ) ^ { 2 d }$ . For the equivalence, $d _ { \mathrm { N } } ^ { \cap } ( \mathcal { C } ) \leq d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { C } )$ by Lemma 9. Conversely, if $d _ { \mathrm { N } } ^ { \cap } ( \mathcal { C } ) \leq d < \infty$ , a DS witness on n inputs has at least $2 ^ { n }$ patterns by Lemma 69, so $2 ^ { n } \leq n ^ { d } ( \ell + 1 ) ^ { 2 d }$ , which bounds n. □

Learning on a region. The proof of suficiency learns on each region separately. For a set $R \subseteq { \mathcal { X } }$ , we say that H is agnostically learnable on R if the requirement of agnostic PAC learnability from one correct label holds for all distributions D on $\mathcal { X } \times \mathcal { A }$ with ${ \mathcal { D } } ( R \times { \mathcal { A } } ) = 1$ The benchmark is still $L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } )$ , which then depends only on the values of the hypotheses on R. We write $\mathcal { D } _ { \mathrm { o b s } }$ for the law of the observed pair $( x , y )$ , where $( x , A ) \sim \mathcal { D }$ and y is the demonstrated label.

## 14.4 Necessity

We now prove Proposition 65 and the necessity part of Theorem 64. The two indistinguishability arguments (Proposition 65 and Lemma 72) follow the pattern of Theorem 67: two distributions of correct-label sets induce the same law of demonstrations, but no prediction is good under both. We first show that local compatibility is necessary. For the DS condition, Lemma 69 shows that if $d _ { \mathrm { D S } } ^ { \Omega } ( \widetilde { \mathcal { H } } ) = \infty$ , then $\widetilde { \mathcal { H } }$ has arbitrarily large DS witnesses within one of the two regions. On the pure region it gives a realizable lower bound (Lemma 74); on the mixed region it gives large Natarajan cubes, which nondegeneracy and local compatibility turn into the anchored cubes defined below (Lemmas 72 and 73). Since excess errors are at most one, the first implication in (4) shows that an $( \varepsilon , \delta )$ agnostic guarantee gives expected excess error at most $\varepsilon + \delta$ , so each lower bound below rules out agnostic learnability.

Proof of Proposition 65. We use one input at which a singleton value lies outside a large correct-label set. Since H is not locally compatible, there are $x \in \mathcal { X } , h , h ^ { \prime } \in \mathcal { H }$ and a label s such that $h ( x ) = \{ s \}$ , the set $h ^ { \prime } ( x )$ has at least two labels, and $s \not \in h ^ { \prime } ( x )$ . Choose distinct $u , v \in h ^ { \prime } ( x )$ . Both distributions below put all their mass on the input x.

– Under $\mathcal { D } _ { 1 }$ , the set A is $h ^ { \prime } ( x )$ with probability $3 / 5$ and $\{ s \}$ with probability $2 / 5 ;$ when $A = h ^ { \prime } ( x )$ , the demonstrator shows u or v with probability $1 / 2$ each.

– Under $\mathcal { D } _ { 2 }$ , the set A is $\{ s \} , \{ u \}$ or $\{ v \}$ with probabilities $2 / 5 , 3 / 1 0$ and $3 / 1 0$ , and the demonstrator shows its only label.

In both cases the demonstrated label is s, u or v with probabilities $2 / 5 , 3 / 1 0$ and $3 / 1 0$ , so the sample, and hence the learner’s prediction ${ \widehat { g } } ( x )$ , has the same law under $\mathcal { D } _ { 1 }$ and $\mathcal { D } _ { 2 }$ . The hypothesis $h ^ { \prime }$ disagrees with A under $\mathcal { D } _ { 1 }$ only when $A = \{ s \}$ , and h disagrees with A under $\mathcal { D } _ { 2 }$ only when $A \neq \{ s \}$ , so $L _ { \mathcal { D } _ { 1 } } ^ { \mathrm { s e t } } ( \mathcal { H } ) \le 2 / 5$ and $L _ { \mathcal { D } _ { 2 } } ^ { \mathrm { s e t } } ( \mathcal { H } ) \le 3 / 5$

Let $p _ { s } , p _ { \mathrm { i n } }$ and $p _ { \mathrm { o u t } }$ be the probabilities that ${ \widehat { g } } ( x )$ equals s, lies in $h ^ { \prime } ( x )$ , or lies outside $h ^ { \prime } ( x ) \cup \{ s \}$ . Under $\mathcal { D } _ { 1 }$ , the prediction s is wrong with probability $3 / 5$ , a label of $h ^ { \prime } ( x )$ with probability $2 / 5 ,$ and any other label always. Under $\mathcal { D } _ { 2 }$ , the prediction s is wrong with probability $3 / 5 ,$ , a label of $h ^ { \prime } ( x )$ with probability at least $7 / 1 0$ , and any other label always. Hence the expected excess errors $e _ { 1 }$ and $e _ { 2 }$ under $\mathcal { D } _ { 1 }$ and $\mathcal { D } _ { 2 }$ satisfy

$$
e _ { 1 } \geq \frac { 1 } { 5 } p _ { s } + \frac { 3 } { 5 } p _ { \mathrm { o u t } } , \qquad e _ { 2 } \geq \frac { 1 } { 1 0 } p _ { \mathrm { i n } } + \frac { 2 } { 5 } p _ { \mathrm { o u t } } ,
$$

and therefore $\begin{array} { r } { \frac { 1 } { 3 } e _ { 1 } + \frac { 2 } { 3 } e _ { 2 } \geq \frac { 1 } { 1 5 } ( p _ { s } + p _ { \mathrm { i n } } + p _ { \mathrm { o u t } } ) = \frac { 1 } { 1 5 } } \end{array}$ . One of $e _ { 1 } , e _ { 2 }$ is at least $1 / 1 5$

Anchored cubes. The following configuration defeats every agnostic learner: a cube in which each hypothesis takes a prescribed singleton or contains a baseline label, together with one hypothesis that takes all baseline labels as singletons.

Definition 71 (Anchored cubes). An anchored cube of size n in H consists of distinct inputs $x _ { 1 } , \ldots , x _ { n }$ , labels $a _ { i } \neq b _ { i }$ for $i \in [ n ]$ , an anchor $g \in \mathcal { H }$ with $g ( x _ { i } ) = \{ b _ { i } \}$ for all i, and, for every $T \subseteq [ n ]$ , a hypothesis $h _ { T } \in \mathcal { H }$ such that

$$
h _ { T } ( x _ { i } ) = \{ a _ { i } \} \quad ( i \in T ) \qquad \mathrm { a n d } \qquad h _ { T } ( x _ { i } ) \neq \star , b _ { i } \in h _ { T } ( x _ { i } ) \quad ( i \notin T ) .
$$

The same demonstrations admit two explanations. In the first, the hidden sets are given by $h _ { T } .$ so the benchmark is zero and the learner must predict $a _ { i }$ on $T .$ . In the second, the hidden sets are the demonstrated singletons: predicting $a _ { i }$ is then wrong outside $T .$ , while the anchor keeps the benchmark small. We choose $T$ at random so that the sample does not reveal whether an unseen input belongs to $T$

Lemma 72 (Anchored cubes defeat every learner). If H has an anchored cube of size $n \geq \operatorname* { m a x } \{ 1 , 6 m \}$ , then for every learner that uses m one-correct-label examples there are a distribution D on $\mathcal { X } \times \mathcal { A }$ and a demonstrator such that

$$
\mathbb { E } \big [ L _ { \mathcal { D } } ( \widehat { g } ) \big ] \geq L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } ) + \frac { 1 } { 8 } .
$$

Consequently, if H has anchored cubes of every size, then it is not agnostically PAC learnable from one correct label.

Proof. Choose a random set $T \subseteq [ n ]$ that contains each i independently with probability $1 / 4$ In both distributions below, the input is uniform on $\{ x _ { 1 } , \ldots , x _ { n } \}$ and the demonstrated label at $x _ { i }$ is $a _ { i }$ if $i \in T$ and $b _ { i }$ otherwise. Under $\mathcal { D } _ { T }$ , the correct-label set at $x _ { i }$ is $h _ { T } ( x _ { i } )$ ; this is a realizable experiment with target $h _ { T }$ , so $L _ { \mathcal { D } _ { T } } ^ { \mathrm { s e t } } ( \mathcal { H } ) = 0$ . Under $\mathcal { D } _ { T } ^ { \prime }$ , the correct-label set is the demonstrated singleton; the anchor $g$ disagrees with it only at the inputs $x _ { i }$ with $i \in T$ , so $L _ { \mathcal { D _ { T } ^ { \prime } } } ^ { \mathrm { s e t } } ( \mathcal { H } ) \leq | T | / n$ . Both demonstrators are valid by the definition of an anchored cube, and the samples have the same law under $\mathcal { D } _ { T }$ and $\mathcal { D } _ { T } ^ { \prime }$

Draw a test index $j$ uniformly from $[ n ]$ , and let E be the event that $x _ { j }$ does not occur in the sample. Its probability is $u = ( 1 - 1 / n ) ^ { m } \geq 1 - m / n \geq 5 / 6$ . On $E$ , the event $j \in T$ is independent of the sample, the test index and the seed, and it has probability $1 / 4$ . Let r be the conditional probability, given $E _ { i }$ , that $\widehat { g } ( x _ { j } ) = a _ { j }$ . Under $\mathcal { D } _ { T } .$ , if $j \in T$ the only correct label is $a _ { j }$ , so

$$
\mathbb { E } \big [ L _ { \mathcal { D } _ { T } } ( \widehat { g } ) \big ] \geq \frac { u } { 4 } ( 1 - r ) .
$$

Under $\mathcal { D } _ { T } ^ { \prime }$ , at an unseen input the prediction $a _ { j }$ is wrong with probability $3 / 4$ and every other prediction with probability at least $1 / 4$ , so the conditional error on $E$ is at least $1 / 4 + r / 2$ Since $\mathbb { E } | T | / n = 1 / 4$

$$
\mathbb { E } \big [ L _ { \mathcal { D } _ { T } ^ { \prime } } ( \widehat { g } ) - L _ { \mathcal { D } _ { T } ^ { \prime } } ^ { \mathrm { s e t } } ( \mathcal { H } ) \big ] \geq u \Big ( \frac { 1 } { 4 } + \frac { r } { 2 } \Big ) - \frac { 1 } { 4 } = \frac { u r } { 2 } - \frac { 1 - u } { 4 } .
$$

These expectations include the choice of $T .$ . We weight the two bounds so that $r$ cancels:

$$
\frac { 2 } { 3 } \cdot \frac { u } { 4 } ( 1 - r ) + \frac { 1 } { 3 } \Big ( \frac { u r } { 2 } - \frac { 1 - u } { 4 } \Big ) = \frac { u } { 4 } - \frac { 1 } { 1 2 } \geq \frac { 5 } { 2 4 } - \frac { 2 } { 2 4 } = \frac { 1 } { 8 } .
$$

Hence for some fixed $T ,$ one of $\mathcal { D } _ { T }$ and $\mathcal { D } _ { T } ^ { \prime }$ has expected excess error at least $1 / 8$ . The consequence follows by taking $n \geq$ 6m and $\varepsilon + \delta < 1 / 8$ □

Nondegeneracy and local compatibility turn every Natarajan cube of the collapsed class into an anchored cube of a third of its size. The two anchors $g _ { 0 } , g _ { 1 }$ handle the coordinates at which the cube compares a singleton label with $\diamond$

Lemma 73 (Extracting anchored cubes). Let H be nondegenerate and locally compatible. If $\widetilde { \mathcal { H } }$ product-shatters a sequence of N distinct inputs, then H has an anchored cube of size at least $N / 3$

Proof. For a class with singleton values, a product witness on $x _ { 1 } , \ldots , x _ { \overset { N } { \underset {  } { \sim } } }$ contains a cube $\textstyle \prod _ { i = 1 } ^ { N } \{ c _ { i } , c _ { i } ^ { \prime } \}$ with $c _ { i } \neq c _ { i } ^ { \prime }$ in $\mathcal { V } \cup \{ \circ \}$ . Every vertex is the pattern of some $h$ with $h$ defined at all $x _ { i } .$ , and at most one of $c _ { i } , c _ { i } ^ { \prime }$ is ⋄. Partition the coordinates into the three groups below; a largest group has at least $N / 3$ coordinates, and each group yields an anchored cube. To define its hypotheses $h _ { T }$ , choose the indicated vertex on that group and fix arbitrary choices on all remaining coordinates.

– Both $c _ { i } , c _ { i } ^ { \prime }$ are labels of $\mathcal { V }$ . Put $a _ { i } = c _ { i }$ and $b _ { i } = c _ { i } ^ { \prime }$ . Let $h _ { T }$ take $\{ a _ { i } \}$ on $T$ and $\{ b _ { i } \}$ of $T _ { \ast }$ , and let the anchor be $h _ { \emptyset }$

– The symbols are $a _ { i } \in \mathcal V$ and $\diamond ,$ with $a _ { i } \neq b _ { 0 } ( x _ { i } )$ . Put $b _ { i } = b _ { 0 } ( x _ { i } )$ , and let the anchor be $g _ { 0 }$ . Let $h _ { T }$ take $\{ a _ { i } \}$ on $T$ and a set with at least two labels of $T ;$ by local compatibility with $g _ { 0 } ( x _ { i } ) = \{ b _ { i } \}$ , this set contains $b _ { i }$

– The symbols are $b _ { 0 } ( x _ { i } )$ and $\diamond .$ . Put $a _ { i } ~ = ~ b _ { 0 } ( x _ { i } )$ and $b _ { i } ~ = ~ b _ { 1 } ( x _ { i } )$ , which difer by nondegeneracy, and let the anchor be $g _ { 1 }$ . Let $h _ { T }$ take $\{ a _ { i } \}$ on $T$ and a set with at least two labels of $T$ , which contains $b _ { i }$ by local compatibility with $g _ { 1 }$

In each group the conditions of Definition 71 hold on the retained inputs.

On the pure region, large DS witnesses may use unboundedly many labels, so we need a realizable lower bound that does not depend on the number of labels. The bound $1 / q$ of Proposition 22 gives no positive constant when $\mathcal { V }$ is infinite; for singleton values, the same argument gives $1 / 2$

Lemma 74 (DS lower bound with singleton values). Let S be a sequence of n distinct inputs, and let $V \subseteq \mathcal { H } | _ { S }$ be a DS witness whose patterns consist of singletons. For every learner that uses m one-correct-label examples there are $h ^ { \star } \in { \mathcal { H } }$ and a distribution P with $P ( \operatorname { s u p p } ( h ^ { \star } ) ) = 1$ such that

$$
\mathbb { E } \big [ L _ { P , h ^ { \star } } ( \widehat { g } ) \big ] \geq \frac { 1 } { 2 } \Big ( 1 - \frac { 1 } { n } \Big ) ^ { m } .
$$

Proof. With singleton targets every demonstration is the correct label itself, so this is ordinary realizable multiclass learning. We follow the proof of Proposition 22, with P uniform on the inputs of $S$ and a uniformly random target pattern $v \in V$ . At a test input $x _ { i }$ missing from the sample, given $v _ { - i }$ , the target is uniform on the hyperedge $e _ { i , v _ { - i } }$ . This hyperedge has at least two patterns, and they have distinct labels at coordinate $i ,$ so every prediction is wrong with conditional probability at least $1 - 1 / | e _ { i , v _ { - i } } | \geq 1 / 2$ . The rest of the proof is unchanged, and it does not use the finiteness of $\mathcal { V }$ □

We can now prove the necessity part of Theorem 64.

Proof of necessity in Theorem $6 4$ . Let $\mathcal { H }$ be nondegenerate and locally compatible, and suppose that $d _ { \mathrm { D S } } ^ { \Omega } ( \widetilde { \mathcal { H } } ) = \infty$ . By Lemma 69, the restriction of $\widetilde { \mathcal { H } }$ to ${ \mathcal { X } } _ { \mathrm { p u r e } }$ or its restriction to ${ \mathcal { X } } _ { \mathrm { m i x } }$ has infinite DS dimension. We show that $\mathcal { H }$ is not agnostically PAC learnable in either case.

The pure region. Let V be a DS witness of $\widetilde { \mathcal { H } }$ on n inputs of ${ \mathcal { X } } _ { \mathrm { p u r e } }$ . There $\widetilde { h } ( x ) = a$ means $h ( x ) = \{ a \}$ , so $V$ , read as a set of singleton patterns, is also a DS witness in $\mathcal { H } | _ { S }$ , where $S$ is the sequence of these inputs. A realizable experiment is an agnostic experiment with benchmark zero, so Lemma 74 with $n \geq 2 m { \mathrm { ~ g i v e s } }$ , for every learner with m examples, a distribution under which the expected excess error is at least $\begin{array} { r } { \frac 1 2 ( 1 - \frac { 1 } { n } ) ^ { m } \geq \frac 1 2 ( 1 - \frac { \bar { m } } { n } ) \geq \frac 1 4 } \end{array}$ Since such witnesses exist for every n, H is not agnostically learnable.

The mixed region. By (86), the restriction of $\widetilde { \mathcal { H } }$ to ${ \mathcal { X } } _ { \mathrm { m i x } }$ has at most $\ell + 1$ possible values at each input. Its DS dimension is infinite, so by Lemma $7 0$ its Natarajan dimension is infinite as well. A sequence product-shattered by this restriction is product-shattered by $\widetilde { \mathcal { H } }$ , so Lemma 73 gives anchored cubes of every size, and Lemma 72 shows that $\mathcal { H }$ is not agnostically learnable. □

## 14.5 Learning on the Pure Region

We now prove suficiency, one region at a time. On the pure region every defined value is a singleton, so a hypothesis can match the hidden set only by predicting the demonstrated label. Its error on the demonstrated labels is therefore at most its exact-set mismatch error. An ordinary agnostic multiclass learner sufices; we obtain one from our full-set results with $\ell = 1$

Lemma 75 (Learning on the pure region). $I f d _ { \mathrm { D S } } ^ { \Omega } ( \widetilde { \mathcal { H } } ) < \infty$ , then H is agnostically learnable on ${ \mathcal { X } } _ { \mathrm { p u r e } }$

Proof. Let C be the class of restrictions of the members of $\mathcal { H }$ to ${ \mathcal { X } } _ { \mathrm { p u r e } } ,$ viewed as a class on the countable instance space ${ \mathcal { X } } _ { \mathrm { p u r e } }$ . Its values are singletons or ⋆, so $\mathcal { C }$ is a class with $\ell = 1$ , and on ${ \mathcal { X } } _ { \mathrm { p u r e } }$ the collapsed hypothesis $\widetilde { h }$ is h with each value $\{ a \}$ written as a. Hence $d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { C } ) \leq d _ { \mathrm { D S } } ^ { \cap } ( \widetilde { \mathcal { H } } ) < \infty$ by Lemma 69.

With $\ell = 1$ the intersection class adds nothing: two singletons intersect only if they are equal, so every member of $\mathcal { T } _ { 2 } ( \mathcal { C } )$ agrees on its support with a member of C, and ${ \mathcal { C } } \subseteq { \mathcal { T } } _ { 2 } ( { \mathcal { C } } )$ The two classes therefore have the same projections, and in particular $d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { T } _ { 2 } ( \mathcal { C } ) ) < \infty$ Theorem 60 with $\rho = \gamma = 1 / 4$ (this $\gamma$ satisfies (9)) gives $d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathcal { C } ) = d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathcal { T } _ { 2 } ( \mathcal { C } ) ) < \infty$ , and Theorem 45 then gives an agnostic full-set learner for C. With $\ell = 1$ a full-set example is a pair $( x , \{ y \} )$ , so this is an ordinary agnostic multiclass learner: for all $\varepsilon , \delta$ there is a finite sample size such that for every distribution of pairs $( x , y )$ on $\mathcal { X } _ { \mathrm { p u r e } } \times \mathcal { Y }$ , with probability at least $1 - \delta$ , the output $\widehat g$ satisfies

$$
\operatorname* { P r } [ { \widehat { g } } ( x ) \neq y ] \leq \operatorname* { i n f } _ { h \in { \mathcal { H } } } \operatorname* { P r } [ h ( x ) \neq \{ y \} ] + \varepsilon ,
$$

where an undefined value counts as an error. We run it on the observed pairs, drawn from $\mathcal { D } _ { \mathrm { o b s } }$

It remains to pass from the demonstrated label to the hidden set. For every $( x , A )$ with $x \in \mathcal { X } _ { \mathrm { p u r e } }$ , every $y \in A$ , every $h \in { \mathcal { H } }$ and every label $z ,$

$$
\mathbb { 1 } \{ h ( x ) \neq \{ y \} \} \leq \mathbb { 1 } \{ h ( x ) \neq A \} \qquad { \mathrm { a n d ~ } } \qquad \mathbb { 1 } \{ z \notin A \} \leq \mathbb { 1 } \{ z \neq y \} .
$$

For the first inequality, if $h ( x ) = A$ then A is a defined value at a pure input, hence a singleton, and so $A = \{ y \}$ The second holds because $y \in A$ . Taking expectations, with probability at least $1 - \delta$

$$
L _ { \mathcal { D } } ( \widehat { g } ) \leq \operatorname* { P r } [ \widehat { g } ( x ) \neq y ] \leq \operatorname* { i n f } _ { h \in \mathcal { H } } \operatorname* { P r } \big [ h ( x ) \neq \{ y \} \big ] + \varepsilon \leq L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } ) + \varepsilon .
$$

## 14.6 Learning on the Mixed Region

On the mixed region a demonstrated label may come from a correct-label set with several elements, and the learner cannot tell which. By (86), every label of ${ \mathcal { V } } _ { x }$ is correct whenever the hidden set is a value of a class member with at least two labels, so we restrict predictions to $\mathcal { \partial } _ { x }$ . The argument has three steps. First, we bound the hidden excess risk against any $h \in \mathcal H$ by a comparison $\Delta _ { h }$ that depends only on the law of the observed pairs (Lemma 76). Second, specialists over a finite cover of the collapsed class, combined with a leave-one-out argument, make this comparison small in expectation (Lemma 78). Third, we obtain a single predictor with high confidence (Lemma 79). Ordinary validation does not work in this step, since the comparison can be negative and evaluating it requires the unknown comparator $h ;$ instead we average independent runs, predict a most frequently demonstrated label at inputs of large mass, and round randomly elsewhere.

Throughout this subsection $\mathcal { H }$ is nondegenerate and locally compatible, and $\mathcal { D }$ is a distribution on ${ \mathcal { X } } _ { \operatorname* { m i x } } \times { \mathcal { A } } .$ A randomized prediction $p$ is admissible if $p ( \cdot \mid x )$ is supported on $\mathcal { \partial } _ { x }$ for every $x \in \mathcal { X } _ { \operatorname* { m i x } }$ , and a predictor $g$ is admissible if $g ( x ) \in \mathcal { V } _ { x } ;$ we identify $g$ with the randomized prediction that puts mass one on $g ( x )$ . As in (3), the risk of $p$ is $L _ { \mathcal { D } } ( p ) = \mathbb { E } _ { ( x , A ) \sim \mathcal { D } } [ 1 - p ( A \mid x ) ]$

An observable comparison. For $h \in { \mathcal { H } }$ , an admissible $p ,$ and an observed pair $( x , y )$ , let

$$
r _ { h } ( p ; x , y ) = \left\{ \begin{array} { l l } { \mathbb { 1 } \{ y = a \} - p ( y \mid x ) , } & { h ( x ) = \{ a \} , } \\ { 0 , } & { \lvert h ( x ) \rvert \geq 2 , } \\ { - p ( y \mid x ) , } & { h ( x ) = \star , } \end{array} \right. \qquad \Delta _ { h } ( p ) = \mathbb { E } _ { ( x , y ) \sim { D _ { \mathrm { o b s } } } } \big [ r _ { h } ( p ; x , y ) \big ] .
$$

At a singleton value, $r _ { h }$ compares how often h and $p$ agree with the demonstrated label, as in ordinary multiclass learning. At a value with two or more labels it is zero, since an admissible prediction is correct whenever this value is. At an undefined value, h counts as wrong and the learner’s agreement $p ( y \mid x )$ is subtracted. The comparison $\Delta _ { h } ( p )$ is afine in $p$ and lies in $[ - 1 , 1 ]$ ; it can be negative.

Lemma 76 (Observable comparison). For every $h \in \mathcal H$ and every admissible $p ,$

$$
L _ { \mathcal { D } } ( p ) - \operatorname* { P r } _ { ( x , A ) \sim \mathcal { D } } [ h ( x ) \neq A ] \leq \Delta _ { h } ( p ) .
$$

Proof. It sufices to show that $1 - p ( A \mid x ) - \mathbb { 1 } \left\{ h ( x ) \neq A \right\} \leq r _ { h } ( p ; x , y )$ whenever $y \in A$ If $h ( x ) = \{ a \} = A$ , then $y = a$ and both sides equal $1 - p ( a \mid x )$ . If $h ( x )$ is a singleton $\{ a \} \neq A$ , or $h ( x ) = \star$ , the left side is $- p ( A \mid x ) \leq - p ( y \mid x )$ , which is at most the right side. If $| h ( x ) | \geq 2$ and $h ( x ) = A$ , then $\mathcal { V } _ { x } \subseteq A$ by (86), so $p ( A \mid x ) = 1$ and both sides are zero. If $| h ( x ) | \geq 2$ and $h ( x ) \neq A$ , the left side is $- p ( A \mid x ) \leq 0$ □

We will also need the comparison input by input. Let $\mu _ { x }$ be the probability of x under $\mathcal { D } _ { \mathrm { o b s } }$ , let $\pi _ { x }$ be the conditional law of the demonstrated label given x (any law if $\mu _ { x } = 0 )$ , and let $\begin{array} { r } { \pi _ { x } ( p ) = \sum _ { a } \pi _ { x } ( a ) p ( a \mid x ) } \end{array}$ be the probability that a prediction drawn from $p ( \cdot \mid x )$ equals the demonstrated label. Let $c _ { h } ( x ) = \pi _ { x } ( a ) { \mathrm { ~ i f ~ } } h ( x ) = \{ a \}$ and $c _ { h } ( x ) = 0$ otherwise, and let $w _ { h } ( x ) = 0 \mathrm { ~ i f ~ } | h ( x ) | \geq 2$ and $w _ { h } ( x ) = 1$ otherwise. Then, directly from the definition of $r _ { h }$

$$
\Delta _ { h } ( p ) = \sum _ { x } \mu _ { x } \big ( c _ { h } ( x ) - w _ { h } ( x ) \pi _ { x } ( p ) \big ) .\tag{87}
$$

Only $\pi _ { x } ( p )$ depends on the prediction, and its coeficient is nonpositive, so increasing agreement with the demonstrated label cannot increase the comparison against any h. This lets us choose predictions at inputs of large mass without knowing the comparator.

Specialists over a finite cover. To make the comparison small we need finitely many candidate predictors on each pool of inputs, and the finite cover of Section 7 provides them. Let d be the Natarajan dimension of the restriction of $\widetilde { \mathcal { H } }$ to ${ \mathcal { X } } _ { \mathrm { m i x } } ;$ it is at most $d _ { \mathrm { D S } } ^ { \cap } ( \widetilde { \mathcal { H } } )$ by Lemma 9. At each $x \in \mathcal { X } _ { \operatorname* { m i x } }$ the collapsed values lie in $\mathcal { \partial } _ { x } \cup \{ \diamond \}$ , so an injection $\iota _ { x }$ of this set into $[ \ell + 1 ]$ turns the restriction into a class with singleton values over $[ \ell + 1 ]$ , still of Natarajan dimension d. Lemma 21, applied to this class,<sup>8</sup> gives for every pool $S = ( x _ { 1 } , \ldots , x _ { n } ) $ of inputs in ${ \mathcal { X } } _ { \mathrm { m i x } }$ a set $F _ { S } \subseteq [ \ell + 1 ] ^ { n }$ , depending only on $S$ and H, with the following covering property: for every $h \in \mathcal H$ some $f \in F _ { S }$ satisfies $f ( i ) = \iota _ { x _ { i } } ( \widetilde { h } ( x _ { i } ) )$ at every position i with $x _ { i } \in \operatorname { s u p p } ( h )$ Its size satisfies

$$
\log | F _ { S } | \le M _ { d } ( n ) \log \bigl ( 1 + n ( \ell + 1 ) ^ { 2 } \bigr ) \le \Lambda _ { n } , \qquad \Lambda _ { n } = c ( d + 1 ) \log ^ { 2 } ( n + 1 ) ,\tag{88}
$$

where c depends only on $\ell .$

Each $f \in F _ { S }$ acts as a specialist: at a position i with $f ( i ) = \iota _ { x _ { i } } ( a )$ for some $a \in \mathcal { V } _ { x _ { i } }$ , it is awake and predicts $a ;$ at all other positions it is asleep. We run a multiplicative-weights version of the specialists algorithm of [FSSW97]. As in Theorem 27, the learner predicts by weighting experts, but here only awake experts contribute, and each weight is updated by the expert’s gain relative to the learner. Each $f \in F _ { S }$ starts with weight $1 / | F _ { S } |$ , and the positions are processed in the order of the pool. At position $i ,$ if some specialist is awake, $p _ { i } ( \cdot \mid x _ { i } )$ is the weighted distribution of the labels of the awake specialists; otherwise it puts mass one on a fixed label of $\mathcal { \mathrm { { y } } } _ { { x } _ { i } }$ , which is nonempty. After $y _ { i }$ is revealed, an awake specialist with label a receives the payof $r _ { f , i } = \mathbb { 1 } \left\{ y _ { i } = a \right\} - p _ { i } ( y _ { i } \mid x _ { i } )$ , and a sleeping specialist receives $r _ { f , i } = 0$ . For a learning rate $0 < \beta \le 1 / 2$ , every weight is then multiplied by $1 + \beta r _ { f , i }$ . All predictions $p _ { i }$ are admissible.

The next lemma transfers the usual specialist regret bound to $r _ { h }$ , using a member of $F _ { S }$ that covers h.

Lemma 77 (Specialist bound). For every pool $S = ( x _ { 1 } , \ldots , x _ { n } ) $ of inputs in ${ \mathcal { X } } _ { \mathrm { m i x } }$ , all labels $y _ { 1 } , \dots , y _ { n } \in \mathcal { V }$ , and every $h \in \mathcal H$

$$
\sum _ { i = 1 } ^ { n } r _ { h } ( p _ { i } ; x _ { i } , y _ { i } ) \leq \frac { \Lambda _ { n } } { \beta } + \beta n .
$$

Proof. By the definition of $p _ { i }$ , the payofs at every position have weighted sum zero, so the weights always sum to one and each of them is at most one. Since $\lvert \beta r _ { f , i } \rvert \le 1 / 2$ and $\log ( 1 + z ) \geq z - z ^ { 2 }$ for $| z | \le 1 / 2$ , the final weight $W _ { f }$ of a specialist $f$ satisfies

$$
0 \geq \log W _ { f } \geq - \log | F _ { S } | + \beta \sum _ { i = 1 } ^ { n } r _ { f , i } - \beta ^ { 2 } n .
$$

Thus every specialist has total payof at most log $| F _ { S } | / \beta + \beta n$ . Now let $f$ cover h. We claim that $r _ { h } ( p _ { i } ; x _ { i } , y _ { i } ) \le r _ { f , i }$ at every position. Where $h ( x _ { i } ) = \{ a \}$ , the specialist f is awake and predicts a, so the two payofs are equal. Where $h ( x _ { i } )$ has two or more labels, $f ( i ) = \iota _ { x _ { i } } ( \diamond )$ 2 so $f$ is asleep and both payofs are zero. Where h is undefined, $r _ { h } = - p _ { i } ( y _ { i } \mid x _ { i } )$ , while $r _ { f , i }$ is either 0 or $\mathbb { 1 } \{ y _ { i } = a \} - p _ { i } ( y _ { i } \mid x _ { i } )$ . Summing over i and using (88) proves the lemma. □

An expected comparison bound. A leave-one-out argument turns the specialist bound into an inductive learner, exactly as in the proofs of Theorems 24 and 27.

Lemma 78 (Expected comparison). Let $n \geq 4 ( \Lambda _ { n } + 1 )$ . There is a learner that uses $n - 1$ observed examples and returns an admissible randomized prediction $p$ such that for every distribution of pairs on ${ \mathcal { X } } _ { \operatorname* { m i x } } \times { \mathcal { V } }$ and every $h \in \mathcal H$

$$
\mathbb { E } \big [ \Delta _ { h } ( p ) \big ] \leq 2 \sqrt { \frac { \Lambda _ { n } + 1 } { n } } .
$$

Proof. Let $\beta = \sqrt { ( \Lambda _ { n } + 1 ) / n } \le 1 / 2$ . Given $n - 1$ observed pairs and a query $x \in \mathcal { X } _ { \operatorname* { m i x } }$ , add $x$ to the training inputs as an n-th position. For each ordering of the n positions, run the specialists algorithm on the pool listed in this order, revealing the labels of the training positions that come before the query, and take its prediction at the query. Let $p ( \cdot \mid x )$ be the average of these predictions over all n! orderings; it is admissible and is a deterministic function of the sample.

Now draw n independent pairs, fix an ordering, and run the specialists algorithm once on all positions, revealing every label. The prediction at a position depends only on labels revealed earlier, so it is the prediction used by the learner for this ordering when that position is the query and the other $n - 1$ pairs form the sample. Let $p ^ { ( j ) }$ be the output of the learner trained on all pairs but the j-th. Averaging over orderings, and using that $r _ { h }$ is afine in the prediction, Lemma $7 7$ gives

$$
\sum _ { j = 1 } ^ { n } r _ { h } \big ( p ^ { ( j ) } ; x _ { j } , y _ { j } \big ) \le \frac { \Lambda _ { n } } { \beta } + \beta n ,
$$

where $( x _ { j } , y _ { j } )$ is the j-th pair. The j-th pair is independent of the others, so each summand has expectation $\mathbb { E } [ \Delta _ { h } ( p ) ]$ . Dividing by n gives $\mathbb { E } [ \Delta _ { h } ( p ) ] \le \Lambda _ { n } / ( \beta n ) + \beta \le 2 \beta$ □

Since $\Lambda _ { n } = O _ { \ell } ( ( d + 1 ) \log ^ { 2 } ( n + 1 ) )$ , the bound of Lemma 78 tends to zero, so for every $\alpha > 0$ a finite sample size gives $\mathbb { E } [ \Delta _ { h } ( p ) ] \le \alpha$ for all distributions and all h at once.

High confidence for a single predictor. It remains to pass from expectation to high probability. The learner of the next lemma does not depend on $h .$ , but its guarantee holds for each fixed h separately rather than uniformly over $\mathcal { H } ;$ this sufices, because the proof of Proposition 80 applies it to a single near-optimal $h$

Lemma 79 (Confidence). Let $d < \infty$ . For all $\varepsilon , \delta \in ( 0 , 1 )$ there is a learner that uses finitely many observed examples and returns a random admissible predictor $\widehat g$ such that for every distribution of pairs on ${ \mathcal { X } } _ { \operatorname* { m i x } } \times { \mathcal { V } }$ and every $h \in \mathcal H$

$$
\operatorname* { P r } \left[ \Delta _ { h } ( { \widehat { g } } ) > \varepsilon \right] \leq \delta .
$$

Proof. Let $\alpha = \varepsilon / 4$ , and let $m _ { 0 }$ be the sample size for which the learner of Lemma $7 8$ satisfies $\mathbb { E } [ \Delta _ { h } ( p ) ] \le \alpha$ . Set

$$
K = \Big \lceil \frac { 2 } { \alpha ^ { 2 } } \log \frac { 4 } { \delta } \Big \rceil , \quad \tau = \operatorname* { m i n } \Big \{ \frac { 1 } { 2 } , \frac { 2 \alpha ^ { 2 } } { \log ( 4 / \delta ) } \Big \} , \quad u = \Big \lceil \frac { 1 } { \tau } \log \frac { 4 } { \delta \tau } \Big \rceil , \quad v = \Big \lceil \frac { 2 } { \alpha ^ { 2 } } \Big ( u \log \ell + \log \frac { 8 } { \delta } \Big ) \Big \rceil .
$$

The learner uses three independent blocks of $K m _ { 0 }$ , u and v examples, in this order. Fix a distribution and $h \in { \mathcal { H } }$ . We show that each of the four steps below fails with probability at most $\delta / 4$

Step 1 (averaging). Train K randomized predictions $p _ { 1 } , \ldots , p _ { K }$ on separate parts of the first block, each of $m _ { 0 }$ examples, and let $\bar { p }$ be their average. The values $\Delta _ { h } ( p _ { j } )$ are independent, lie in $[ - 1 , 1 ]$ , and have mean at most $\alpha$ . Since $\Delta _ { h }$ is afine, Hoefding’s inequality gives $\Delta _ { h } ( \bar { p } ) \leq$ 2α except with probability $e ^ { - K \alpha ^ { 2 } / 2 } \le \delta / 4$

Step 2 (heavy inputs). Let U be the set of inputs of the second block. At most $1 / \tau$ inputs have $\mu _ { x } > \tau .$ , and each of them is missed with probability at most $e ^ { - u \tau }$ . So, except with probability $\tau ^ { - 1 } e ^ { - u \tau } \leq \delta / 4$ , every input outside $U$ has mass at most $\tau$

Step 3 (most frequent labels on $U )$ . For $x \in U$ , let $\widehat { a } _ { x }$ be the label of $\mathcal { \partial } _ { x }$ that is demonstrated most often at x in the third block, with ties broken by the fixed enumeration of $\mathcal { V } .$ . This choice maximizes the empirical frequency of $\{ x \in U , \ y = a ( x ) \}$ over the at most $\ell ^ { u }$ maps a with $a ( x ) \in \mathcal { V } _ { x }$ . For a fixed map $a ,$ , the event has probability $\begin{array} { r } { \sum _ { x \in U } \mu _ { x } \pi _ { x } ( a ( x ) ) } \end{array}$ . By Hoefding’s inequality and a union bound, conditionally on $U$ , all these frequencies are within $\alpha / 2$ of their probabilities except with probability $2 \ell ^ { u } e ^ { - v \alpha ^ { 2 } / 2 } \le \delta / 4$ . On this event, comparing $\widehat { a }$ with a map that picks a most probable label of $\pi _ { x }$ in $\mathcal { \partial } _ { x }$ at each $x \in U$ gives

$$
\sum _ { x \in U } \mu _ { x } \Big ( \operatorname* { m a x } _ { a \in \mathcal { Y } _ { x } } \pi _ { x } ( a ) - \pi _ { x } ( \widehat { a } _ { x } ) \Big ) \leq \alpha .
$$

Now let $p ^ { U }$ agree with $\bar { p }$ outside $U$ and put mass one on $\widehat { a } _ { x }$ at $x \in U$ . Since $\bar { p }$ is admissible, $\pi _ { x } ( \bar { p } ) \leq \operatorname* { m a x } _ { a \in \mathcal { Y } _ { x } } \pi _ { x } ( a )$ , and since $w _ { h } ( x ) \in \{ 0 , 1 \}$ , (87) gives

$$
\Delta _ { h } ( p ^ { U } ) - \Delta _ { h } ( \bar { p } ) = \sum _ { x \in U } \mu _ { x } w _ { h } ( x ) \big ( \pi _ { x } ( \bar { p } ) - \pi _ { x } ( \widehat { a } _ { x } ) \big ) \leq \alpha .
$$

Step 4 (rounding). Let $\widehat { \boldsymbol { g } } ( \boldsymbol { x } ) = \widehat { \boldsymbol { a } } _ { x }$ for $x \in U$ , and for every other $x \in \mathcal { X } _ { \operatorname* { m i x } }$ draw ${ \widehat { g } } ( x )$ from $\bar { p } ( \cdot \mid x )$ , independently across inputs and once for all, using the seed. Conditionally on the examples, (87) shows that $\Delta _ { h } ( \widehat { g } ) - \Delta _ { h } ( p ^ { U } )$ is a sum over $x \not \in U$ of independent terms of mean zero, and the term at x lies in an interval of length at most $\mu _ { x }$ . If every $x \not \in U$ has $\mu _ { x } \leq \tau$ , then $\textstyle \sum _ { x \not \in U } \mu _ { x } ^ { 2 } \leq \tau$ , and Hoefding’s inequality<sup>9</sup> gives $\Delta _ { h } ( \widehat { g } ) \leq \Delta _ { h } ( p ^ { U } ) + \alpha$ except with probability $\stackrel { \cdot } { e } ^ { - 2 \alpha ^ { 2 } / \tau } \leq \delta / 4$

By a union bound over the four steps, with probability at least $1 - \delta$ we have $\Delta _ { h } ( \widehat { g } ) \leq$ $2 \alpha + \alpha + \alpha = \varepsilon$ □

For standard Borel instance spaces, Section E.5 proves the same guarantee by rounding on finitely many Borel sets in place of individual inputs.

Proposition 80 (Learning on the mixed region). Let H be nondegenerate and locally compatible, and suppose the restriction of $\widetilde { \mathcal { H } }$ to ${ \mathcal { X } } _ { \mathrm { m i x } }$ has finite Natarajan dimension, for instance because $d _ { \mathrm { D S } } ^ { \cap } ( \widetilde { \mathcal { H } } ) < \infty$ . Then H is agnostically learnable on ${ \mathcal { X } } _ { \mathrm { m i x } }$

Proof. Given $\varepsilon , \delta ,$ run the learner of Lemma 79 with accuracy $\varepsilon / 2$ on the observed pairs. For the hidden distribution D at hand, fix $h \in { \mathcal { H } }$ with $\Pr _ { ( x , A ) \sim \mathcal { D } } [ h ( x ) \neq A ] \leq L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } ) + \varepsilon / 2 ;$ the learner does not use it. With probability at least $1 - \delta , \Delta _ { h } ( \widehat { g } ) \leq \varepsilon / 2$ , and Lemma 76, applied to the admissible predictor ${ \widehat { g } } ,$ gives $L _ { \mathcal { D } } ( \widehat { g } ) \leq \operatorname* { P r } _ { ( x , A ) \sim \mathcal { D } } [ h ( x ) \neq A ] + \varepsilon / 2 \leq L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } ) + \varepsilon$ □

## 14.7 Proof of the Characterization

The two regions depend only on ${ \mathcal { H } } ,$ so the learner can assign each input to its region and pass each regional learner the examples from its region.

Lemma 81 (Gluing). If H is agnostically learnable on ${ \mathcal { X } } _ { \mathrm { p u r e } }$ and on ${ \mathcal { X } } _ { \mathrm { m i x } }$ , then H is agnostically PAC learnable from one correct label.

Proof. Fix $\varepsilon , \delta .$ . For each region $R _ { : }$ , let $m _ { R }$ be the sample size of a learner on R with accuracy $\varepsilon / 4$ and confidence $\delta / 4$ , let $m _ { * } = \operatorname* { m a x } \{ 1 , m _ { \mathrm { p u r e } } , m _ { \mathrm { m i x } } \}$ , and draw

$$
N = \left\lceil \frac { 3 2 } { \varepsilon } \Big ( m _ { * } + \log \frac { 8 } { \delta } \Big ) \right\rceil
$$

examples. For each region $R ,$ pass its first $m _ { R }$ examples to the regional learner; if fewer are available, use an arbitrary predictor on $R .$ The two learners use independent seeds, and the final predictor uses the output for the region containing its input. Conditional on which sample positions fall in $R ,$ the retained examples are independent draws from the conditional law $\mathcal { D } _ { R }$ of $( x , A )$ given $x \in R$ , with the same demonstrator, so the regional guarantee applies whenever enough examples are retained.

Let $\mu _ { R } = \operatorname* { P r } _ { \mathcal { D } } [ x \in R ]$ . A region of mass below $\varepsilon / 4$ costs at most its mass, so only regions of larger mass need a guarantee. If $\mu _ { R } \geq \varepsilon / 4$ , the number of examples in R has mean $N \mu _ { R } \geq$

$8 ( m _ { * } + \log ( 8 / \delta ) ) > 2 m _ { R }$ , so by a multiplicative Chernof bound fewer than $m _ { R }$ examples are retained with probability at most $e ^ { - N \bar { \mu _ { R } } / 8 } \le e ^ { - \log ( 8 / \delta ) } = \delta / 8$ . Each regional learner fails with probability at most $\delta / 4$ . Hence, with probability at least $1 - 2 \cdot \delta / 8 - 2 \cdot \delta / 4 \geq 1 - \delta$ , every region with $\mu _ { R } \geq \varepsilon / 4$ satisfies $L _ { \mathcal { D } _ { R } } ( \widehat { g } ) \leq L _ { \mathcal { D } _ { R } } ^ { \mathrm { s e t } } ( \mathcal { H } ) + \varepsilon / 4$

The infimum of a sum is at least the sum of the infima, so $\begin{array} { r } { \sum _ { R } \mu _ { R } L _ { \mathcal { D } _ { R } } ^ { \mathrm { s e t } } ( \mathcal { H } ) \le L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } ) } \end{array}$ 2 where regions of mass zero contribute zero. On the same event, therefore,

$$
L _ { \mathcal { D } } ( \widehat { g } ) \leq \sum _ { R : \mu _ { R } \geq \varepsilon / 4 } \mu _ { R } \Big ( L _ { \mathcal { D } _ { R } } ^ { \mathrm { s e t } } ( \mathcal { H } ) + \frac { \varepsilon } { 4 } \Big ) + \sum _ { R : \mu _ { R } < \varepsilon / 4 } \mu _ { R } \leq L _ { \mathcal { D } } ^ { \mathrm { s e t } } ( \mathcal { H } ) + \frac { 3 \varepsilon } { 4 } .
$$

Proof of Theorem $6 4$ . Necessity was proved in Section 14.4. For suficiency, let $\mathcal { H }$ be nondegenerate and locally compatible with $d _ { \mathrm { D S } } ^ { \Omega } ( \tilde { \mathcal { H } } ) < \infty$ . Then H is agnostically learnable on ${ \mathcal { X } } _ { \mathrm { p u r e } }$ by Lemma 75 and on ${ \mathcal { X } } _ { \mathrm { m i x } }$ by Proposition 80, and Lemma 81 combines the two learners.

Finally, we prove Corollary 66 by comparing the collapsed class with the binary class $\mathcal { I }$

Lemma 82 (Collapsed and binary classes). For every class H, $d _ { \mathrm { N } } ^ { \cap } ( \widetilde { \mathcal { H } } ) \leq \mathrm { V C } ( \mathcal { I } )$ . If H is locally compatible and $\mathcal { X } _ { \mathrm { m i x } } = \mathcal { X }$ , then $\mathrm { V C } ( \mathcal { I } ) < \infty$ if and only if $d _ { \mathrm { D S } } ^ { \Omega } ( \mathcal { \tilde { H } } ) < \infty$

Proof. Take a cube $\textstyle \prod _ { i = 1 } ^ { n } \left\{ c _ { i } , c _ { i } ^ { \prime } \right\}$ realized by $\widetilde { \mathcal { H } }$ on distinct inputs $x _ { 1 } , \ldots , x _ { n } .$ as in the proof of Lemma 73. At each coordinate at least one of $c _ { i } , c _ { i } ^ { \prime }$ is a label of $y ;$ call it $a _ { i } . \mathrm { ~ A ~ }$ hypothesis h realizing a vertex has $J _ { h } ( x _ { i } , a _ { i } ) = 1$ if the vertex takes $a _ { i }$ at i and $J _ { h } ( x _ { i } , a _ { i } ) = 0$ otherwise. So the points $( x _ { i } , a _ { i } )$ are shattered by $\mathcal { I }$ , which proves the inequality.

Now let H be locally compatible with ${ \mathcal { X } } _ { \mathrm { m i x } } ~ = ~ { \mathcal { X } } .$ , and suppose that $\mathcal { I }$ shatters $( x _ { 1 } , a _ { 1 } ) , \ldots , ( x _ { n } , a _ { n } )$ . The inputs are distinct, because $J _ { h } ( x , a ) = J _ { h } ( x , a ^ { \prime } ) = 1$ forces $a = a ^ { \prime }$ The $2 ^ { n }$ binary patterns come from hypotheses defined at all $x _ { i }$ , and two of them with diferent binary patterns have diferent collapsed patterns on $S = ( x _ { 1 } , \ldots , x _ { n } ) $ , since exactly one of them takes $a _ { i }$ at a coordinate i where they difer. So $| { \\widetilde { \mathcal { H } } } | _ { S } | \geq 2 ^ { n }$ . On the other hand, since every input is mixed, local compatibility puts the collapsed values at each input in $\mathcal { \partial } _ { x } \cup \{ \diamond \}$ a set of at most $\ell + 1$ labels (Section 14.3), so Lemma 70 gives $2 ^ { n } \leq n ^ { d } ( \ell + 1 ) ^ { 2 d }$ if $d _ { \mathrm { N } } ^ { \cap } ( \widetilde { \mathcal { H } } ) \leq d$ Hence $\operatorname { V C } ( \mathcal { T } )$ is finite if and only if $d _ { \mathrm { N } } ^ {  } (  )$ is, and by Lemma 70 if and only if $d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { \hat { H } } )$ is.

Proof of Corollary 66. Under the assumption of the corollary every input has a value with at least two labels, so $\mathcal { X } _ { \mathrm { m i x } } = \mathcal { X }$ . Lemma 82 then shows that $\mathrm { V C } ( \mathcal { I } ) < \infty$ if and only if $d _ { \mathrm { D S } } ^ { \cap } ( \widetilde { \mathcal { H } } ) < \infty .$ , and Theorem 64 completes the proof. □

Remark 83 (Partial concept classes). For $\ell = 1$ every defined value is a singleton, so the mixed region is empty, local compatibility holds vacuously, and no anchors are needed. The proof of Lemma $7 5$ with $\mathcal { X } _ { \mathrm { p u r e } } = \mathcal { X }$ gives a learner when $d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { H } ) < \infty .$ and Lemma 74, applied to DS witnesses of $\mathcal { H }$ on the whole of $\mathcal { X }$ , rules out learning when $d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { H } ) = \infty$ . Thus, for $\ell = 1$ and without nondegeneracy, $\mathcal { H }$ is agnostically learnable if and only if $d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { H } ) < \infty$ . For $\mathcal { V } = \{ 0 , 1 \}$ a DS witness is a full binary cube, so the DS dimension is the VC dimension, and we recover the characterization of agnostic learnability of partial concept classes by finite VC dimension [AHHM21].

Remark 84 (Undefined values cannot be coded as zero). In Corollary 66, an undefined value of h must stay undefined in $J _ { h }$ . Let $\mathcal { X } = \mathbb { N } , \mathcal { Y } = \{ 0 , 1 \}$ and $\ell = 2 .$ , and let H consist of the constant hypotheses {0}, {1} and $\{ 0 , 1 \}$ , and, for every finite $F \subseteq \mathbb { N }$ , the hypothesis that equals {0} on $F$ and is undefined elsewhere. This class is nondegenerate, locally compatible, and every input has a value with two labels. On n distinct inputs the only patterns of $\widetilde { \mathcal { H } }$ are the constant ones $0 ^ { n } , 1 ^ { n }$ and $\diamond ^ { n }$ , so $d _ { \mathrm { D S } } ^ { \Omega } ( \widetilde { \mathcal { H } } ) = 1$ and $\mathcal { H }$ is agnostically learnable. Coding undefined values as zero, however, turns $x \mapsto J _ { h } ( x , 0 )$ into the indicators of all finite sets, a class of infinite VC dimension.

## 15 Limits of Dimension Characterizations

All other sections assume that every correct-label set has at most ℓ labels. In this section we drop this uniform bound and prove two impossibility results for characterizations of PAC learnability by dimensions. The first rules out scale-free characterizations, which use a single parameter independent of the accuracy and the confidence, as well as scale-free sample complexity bounds; it holds under both full-set and one-correct-label feedback. The second concerns scale-sensitive profiles, which are families of parameters indexed by a scale $\alpha \in ( 0 , 1 )$ . Under one-correct-label feedback, we show that every such profile whose members are determined by their finite subclasses and ignore the names of the labels (Definition 85) takes the same value on a class learnable from zero examples, on learnable classes of arbitrarily large sample complexity, and on a class that is not learnable. Our approach is related to [LBD24], who showed that no notion of dimension characterizes the learnability of classes of distributions under natural requirements (see also [LBD26]).

Setting. In this section a hypothesis is a total function h from $\mathcal { X }$ to the nonempty subsets of Y. In Section 15.1 every correct-label set is finite, but there is no common bound on their sizes; in Section 15.2 every correct-label set has finite complement in Y. The errors (3), the feedback models, and Definition 1 apply without change. For a bijection $\pi : \mathcal { V }  \mathcal { V }$ and a hypothesis $h ,$ , let $( \pi h ) ( x ) = \{ \pi ( y ) : y \in h ( x ) \}$ and $\pi \mathcal { H } = \{ \pi h : h \in \mathcal { H } \}$ . The parameters that we rule out are of the following kind.

Definition 85. A parameter D assigns a value $\mathrm { D } ( \mathcal { H } ) \in \mathbb { N } _ { 0 } \cup \{ \infty \}$ , where $ { \mathbb { N } } _ { 0 } = \{ 0 \} \cup  { \mathbb { N } }$ , to every nonempty class H under consideration.

(i) D has the finite-witness property if for every nonempty class $\mathcal { H } .$

$$
\begin{array} { r } { \mathrm { D } ( \mathcal { H } ) = \operatorname* { s u p } \bigl \{ \mathrm { D } ( \mathcal { V } ) : \mathcal { V } \subseteq \mathcal { H } , \ 1 \leq | \mathcal { V } | < \infty \bigr \} . } \end{array}\tag{89}
$$

Such a parameter is monotone: $\mathrm { D } ( \mathcal { H ^ { \prime } } ) \leq \mathrm { D } ( \mathcal { H } )$ whenever $\mathcal { H } ^ { \prime } \subseteq \mathcal { H }$

(ii) D is relabeling invariant if $\mathrm { D } ( \pi \mathcal { H } ) = \mathrm { D } ( \mathcal { H } )$ for every class H and every bijection $\pi : \mathcal { V } \to \mathcal { V }$

(iii) A scale-sensitive profile is a family $\mathbf { D } = ( \mathrm { D } _ { \alpha } ) _ { \alpha \in ( 0 , 1 ) }$ of parameters, which need not be related to each other.

Minimax risk. Fix the feedback model, and write $m _ { \mathcal { H } } ( \varepsilon , \delta )$ for the corresponding sample complexity $m _ { \mathcal { H } } ^ { \mathrm { f u l l } } ( \varepsilon , \delta )$ or $m _ { \mathcal { H } } ^ { \mathrm { o n e } } ( \varepsilon , \delta )$ . A data law for a target $h ^ { \star }$ is a distribution P on X under full-set feedback, and a distribution D with marginal $P$ that is compatible with $h ^ { \star }$ under one-correct-label feedback. For an integer $m \geq 0$ , the minimax risk with m examples is

$$
\mathfrak { L } _ { m } ( \mathcal { H } ) = \operatorname* { i n f } _ { \mathrm { l e a r n e r s } } \ \operatorname* { s u p } _ { h ^ { \star } \in \mathcal { H } \mathrm { ~ \mathfrak { d a t a l a w s } ~ } } \mathbb { E } \big [ L _ { P , h ^ { \star } } ( \widehat { g } ) \big ] ,
$$

where the infimum is over learners that use m examples and the expectation is over the sample and the seed. By (4), an $( \varepsilon , \delta ) – \mathrm { P A C }$ guarantee with m examples gives $\mathfrak { L } _ { m } ( \mathcal { H } ) \leq \varepsilon + \delta \mathrm { : }$ hence a class with $\mathfrak { L } _ { m } ( \mathcal { H } ) \geq c > 0$ for every m is not realizably PAC learnable.

Both constructions below use labels in $ { \mathbb { N } } _ { 0 }$ , and then the two feedback models are related by a simple reduction. A full-set learner can pass the least label of each observed set, which is a valid demonstration, to a one-correct-label learner. Hence every one-correct-label guarantee is also a full-set guarantee, and every full-set lower bound is also a one-correct-label lower bound.

## 15.1 Scale-Free Impossibility

Theorem 86 (Scale-free impossibility). Fix either full-set or one-correct-label feedback. Let X and Y be countably infinite, and consider countable classes H of total hypotheses with $1 \leq | h ( x ) | < \infty$ for all $h \in { \mathcal { H } }$ and $x \in \mathcal { X }$

(i) There is no parameter D with the finite-witness property such that $\mathrm { D } ( \mathcal { H } ) < \infty$ if and only if H is realizably PAC learnable.

(ii) There is no pair $( \mathrm { D } , \Psi )$ of a parameter D that is finite on every realizably learnable class and a function $\Psi : \mathbb { N } _ { 0 } \times ( 0 , 1 ) ^ { 2 }  [ 0 , \infty )$ such that

$$
m _ { \mathcal { H } } ( \varepsilon , \delta ) \leq \Psi \big ( \mathrm { D } ( \mathcal { H } ) , \varepsilon , \delta \big )\tag{90}
$$

for every realizably learnable class H and $a l l \varepsilon , \delta \in ( 0 , 1 )$

The proof uses classes $\mathcal { H } _ { j }$ , each on its own block of inputs, whose minimax error is $1 / s _ { j }$ at every sample size, where $s _ { j } = j + 2 ;$ in particular, no ${ \mathcal { H } } _ { j }$ is learnable. A finite subclass of ${ \mathcal { H } } _ { j }$ , however, departs from a default target at only finitely many inputs, and a learner that memorizes its sample is wrong on block $j$ only at the departures that it has not seen. Since the error floors $1 / s _ { j }$ tend to $0 ,$ at any given accuracy all but finitely many blocks can be handled by uniform guessing, so every union of finite subclasses $\mathcal { V } _ { j } \subseteq \mathcal { H } _ { j }$ is learnable. A parameter with finite witnesses that is finite exactly on learnable classes is infinite on each ${ \mathcal { H } } _ { j }$ , hence at least $j$ on some finite $\nu _ { j }$ , and therefore infinite on the learnable union; this proves part (i). For part (ii) we choose the finite subclasses diagonally, so that at accuracy of order $1 / s _ { j }$ the subclass $\nu _ { j }$ needs more examples than Ψ allows for parameter values up to $j .$

Construction: blocks with decreasing error floors. Since $\mathcal { X }$ and $\mathcal { V }$ are countably infinite, we may take $\mathcal { X } = \mathbb { N } \times \mathbb { N }$ and $\mathcal { V } = \mathbb { N }$ , and call $\{ k \} \times$ N the kth block. For $j \in \mathbb N$ and each function $a : \mathbb { N } \to [ s _ { j } ]$ that equals 1 outside a finite set, define

$$
h _ { j , a } ( k , t ) = \left\{ \begin{array} { l l } { [ s _ { j } ] \backslash \{ a ( t ) \} , } & { k = j , } \\ { [ s _ { k } ] , } & { k \neq j , } \end{array} \right.\tag{91}
$$

and let $\mathcal { H } _ { j }$ be the class of these targets. Thus a target in ${ \mathcal { H } } _ { j }$ excludes the label $a ( t )$ at each input $( j , t )$ of its active block and accepts all of $\left[ s _ { k } \right]$ on every other block k. Each ${ \mathcal { H } } _ { j }$ is countable, and its correct-label sets are finite and nonempty. By the reduction above, it sufices to prove lower bounds under full-set feedback and learning guarantees under one-correct-label feedback; each then holds under both models.

Lemma 87 (Error floors and finite unions).

(i) For every $j \geq 1$ and every integer m $\geq 0 _ { i }$

$$
\mathfrak { L } _ { m } ( \mathcal { H } _ { j } ) = \frac { 1 } { s _ { j } } .\tag{92}
$$

In particular, $\mathcal { H } _ { j }$ is not realizably PAC learnable.

(ii) Every nonempty class $\begin{array} { r } { \mathcal { H } = \bigcup _ { j \geq 1 } \mathcal { V } _ { j } } \end{array}$ with finite $\mathcal { V } _ { j } \subseteq \mathcal { H } _ { j }$ is realizably PAC learnable.

Proof. The lower bound in (92). Fix a full-set learner that uses m examples, and an integer $M > m$ . Choose $a ( 1 ) , \dots , a ( M )$ independently and uniformly from $\left[ s _ { j } \right]$ , set $a ( t ) = 1$ for $t > M$ , and let $P$ be the uniform distribution on $( j , 1 ) , \ldots , ( j , M )$ . The sample reveals the values of a only at the sampled inputs. Hence, at a test input that does not appear in the sample, the excluded label is uniform on $\left[ s _ { j } \right]$ and independent of the sample and the seed; a prediction in $[ s _ { j } ]$ is then invalid with probability $1 / s _ { j }$ , and a prediction outside $\left[ s _ { j } \right]$ is always invalid. A test input is absent from the sample with probability $( 1 - 1 / M ) ^ { m }$ , so averaging over the random choice of a gives

$$
\operatorname* { s u p } _ { a } \mathbb { E } \big [ L _ { P , h _ { j , a } } ( \widehat { g } ) \big ] \geq \frac { 1 } { s _ { j } } \Big ( 1 - \frac { 1 } { M } \Big ) ^ { m } .\tag{93}
$$

Letting $M \to \infty$ proves $\mathfrak { L } _ { m } ( \mathcal { H } _ { j } ) \ge 1 / s _ { j }$ for every $m .$ , and in particular ${ \mathcal { H } } _ { j }$ is not realizably PAC learnable.

The upper bound in (92). The learner that ignores the sample and outputs $g _ { \omega } ( k , t ) =$ $1 + \lfloor s _ { k } \omega \rfloor$ , with the seed $\omega$ uniform on $[ 0 , 1 )$ , predicts a uniform label of $\left[ s _ { k } \right]$ on block $k$ For a target $h _ { j , a } \in \mathcal { H } _ { j }$ , this label is always valid outside block $j$ and is invalid at $( j , t )$ with probability $1 / s _ { j }$ . Hence its expected error is at most $1 / s _ { j }$

Unions of finite subclasses. Let each $\mathcal { V } _ { j } \subseteq \mathcal { H } _ { j }$ be finite. The targets in $\nu _ { j }$ exclude a label other than 1 only at inputs $( j , t )$ with t in the finite set

$$
E _ { j } = \{ t \in \mathbb { N } : \ a ( t ) \neq 1 \ \mathrm { f o r ~ s o m e } \ h _ { j , a } \in \mathcal { V } _ { j } \} , \qquad \mathrm { a n d ~ w e ~ p u t } \qquad \mathcal { E } _ { J } = \bigcup _ { j \leq J } \bigl ( \{ j \} \times E _ { j } \bigr ) .
$$

Fix J and let the seed ω be uniform on [0, 1). The learner predicts

$$
\widehat { g } ( j , t ) = \left\{ \begin{array} { l l } { \mathrm { t h e ~ f i r s t ~ l a b e l ~ d e m o n s t r a t e d ~ a t } ( j , t ) , } & { \mathrm { i f ~ } ( j , t ) \mathrm { ~ a p p e a r s ~ i n ~ t h e ~ s a m p l e } , } \\ { 2 , } & { \mathrm { i f ~ } ( j , t ) \mathrm { ~ i s ~ u n s e e n ~ a n d ~ } j \leq J , } \\ { 1 + \lfloor s _ { j } \omega \rfloor , } & { \mathrm { i f ~ } ( j , t ) \mathrm { ~ i s ~ u n s e e n ~ a n d ~ } j > J . } \end{array} \right.
$$

Let the target be $h _ { j ^ { \prime } , a } \in \mathcal { V } _ { j ^ { \prime } }$ . Demonstrated labels are correct. At an unseen input of a block $k \neq j ^ { \prime }$ , the target accepts all of $\left[ s _ { k } \right]$ , and the prediction lies in $\left[ s _ { k } \right]$ (for the label 2 because $s _ { k } \geq 3 )$ . On block $j ^ { \prime }$ there are two cases. If $j ^ { \prime } \leq J ,$ , the label 2 is correct at $( j ^ { \prime } , t )$ unless $a ( t ) = 2$ , and then $( j ^ { \prime } , t ) \in \mathcal { E } _ { J } . \mathrm { ~ } \operatorname { I f ~ } j ^ { \prime } > J$ , the expected error on block $j ^ { \prime }$ is at most $1 / s _ { j ^ { \prime } } \leq 1 / s _ { J + 1 }$ , as in the upper bound above. Hence, for every $h ^ { \star } \in { \mathcal { H } }$ and every data law,

$$
\mathbb { E } \big [ L _ { P , h ^ { \star } } ( \widehat { g } ) \big ] \leq \sum _ { x \in \mathcal { E } _ { J } } p _ { x } ( 1 - p _ { x } ) ^ { m } + \frac { 1 } { s _ { J + 1 } } \leq \frac { \vert \mathcal { E } _ { J } \vert } { m + 1 } + \frac { 1 } { s _ { J + 1 } } ,\tag{94}
$$

where $p _ { x } = P ( \{ x \} )$ , and the second inequality uses $p ( 1 - p ) ^ { m } \leq 1 / ( m + 1 )$ for $p \in [ 0 , 1 ]$ . Given $\varepsilon , \delta ,$ choose J with $1 / s _ { J + 1 } \leq \varepsilon \delta / 2$ , and then m with $| \mathcal { E } _ { J } | / ( m + 1 ) \le \varepsilon \delta / 2$ . The expected error is then at most $\varepsilon \delta .$ , and the second implication in (4) proves realizable PAC learnability under one-correct-label feedback, and hence under full-set feedback. □

Proof of Theorem 86. For part (i), suppose that D has the stated properties. By Lemma $8 7 ( \mathrm { i } )$ no ${ \mathcal { H } } _ { j }$ is learnable, so $\mathrm { D } ( \mathcal { H } _ { j } ) = \infty$ , and the finite-witness property gives a finite nonempty subclass $\mathcal { V } _ { j } \subseteq \mathcal { H } _ { j }$ with $\mathrm { D } ( \mathcal { V } _ { j } ) \geq j$ . By Lemma $8 7 ( \mathrm { i i } )$ , the class $\mathcal { H } = \cup _ { j } \mathcal { V } _ { j }$ is realizably PAC learnable. However, monotonicity gives $\mathrm { D } ( \mathcal { H } ) \geq \operatorname* { s u p } _ { j } \mathrm { D } ( \mathcal { V } _ { j } ) = \infty$ , a contradiction.

For part (ii), suppose that D and Ψ satisfy (90), and set

$$
\varepsilon _ { j } = \delta _ { j } = \frac { 1 } { 8 s _ { j } } , \qquad q _ { j } = 1 + \Bigl [ \operatorname* { m a x } _ { d \in \{ 0 , \ldots , j \} } { \Psi ( d , \varepsilon _ { j } , \delta _ { j } ) } \Bigl ] , \qquad M _ { j } = 2 q _ { j } ,\tag{95}
$$

so that $q _ { j } > \Psi ( d , \varepsilon _ { j } , \delta _ { j } )$ for every $d \leq j$ . Let $\nu _ { j }$ consist of the targets $h _ { j , a } \in \mathcal { H } _ { j }$ with arbitrary values of a on $[ M _ { j } ]$ and $a ( t ) = 1$ for $t > M _ { j }$ These classes are finite, so $\mathcal { H } = \cup _ { i } \mathcal { V } _ { j }$ is realizably PAC learnable by Lemma $8 7 ( \mathrm { i i } )$ . For a learner that uses $m \leq q _ { j }$ examples, the argument for (93) with $M = M _ { j }$ uses only targets in $\nu _ { j }$ , and gives

$$
\operatorname* { s u p } _ { h ^ { \star } \in \mathcal { V } _ { j } } \operatorname* { s u p } _ { \mathrm { \ d a t a \ l a w s } } \mathbb { E } \big [ L _ { P , h ^ { \star } } ( \widehat { g } ) \big ] \geq \frac { 1 } { s _ { j } } \Big ( 1 - \frac { 1 } { M _ { j } } \Big ) ^ { m } \geq \frac { 1 } { s _ { j } } \Big ( 1 - \frac { m } { M _ { j } } \Big ) \geq \frac { 1 } { 2 s _ { j } } .
$$

An $( \varepsilon _ { j } , \delta _ { j } )$ -PAC guarantee would instead give expected error at most $\varepsilon _ { j } + \delta _ { j } = 1 / ( 4 s _ { j } )$ , by

(4). Hence $m _ { \mathcal { H } } ( \varepsilon _ { j } , \delta _ { j } ) > q _ { j }$ . Since $d _ { 0 } = \mathrm { D } ( \mathcal { H } )$ is finite, choosing $j \geq \operatorname* { m a x } \{ 1 , d _ { 0 } \}$ yields

$$
m _ { \mathcal { H } } ( \varepsilon _ { j } , \delta _ { j } ) > q _ { j } > \Psi ( d _ { 0 } , \varepsilon _ { j } , \delta _ { j } ) ,
$$

which contradicts (90).

## 15.2 Scale-Sensitive Impossibility

Theorem 88 (Scale-sensitive impossibility). Under one-correct-label feedback, there are countable input and label spaces and countable classes $\mathcal { G } \subseteq \mathcal { K } _ { N } \subseteq \mathcal { K } _ { \infty } ~ ( N \in \mathbb { N } )$ with the following properties.

(i) A single predictor has error zero for every target in $\mathcal { G } _ { i }$ , so $\mathcal { G }$ is learnable from zero examples.

(ii) Each $\kappa _ { N }$ is realizably PAC learnable, and $m _ { K _ { N } } ( 1 / 1 6 , 1 / 1 6 ) > \lfloor N / 2 \rfloor \ f o r \ N \ge 2$

(iii) The class $\kappa _ { \infty }$ is not realizably PAC learnable, and $\mathfrak { L } _ { m } ( { K } _ { \infty } ) = 1 / 2$ for every integer $m \geq 0$

(iv) Every scale-sensitive profile D whose coordinates have the finite-witness property and are relabeling invariant satisfies $\mathbf { D } ( { \mathcal { G } } ) = \mathbf { D } ( { \mathcal { K } } _ { N } ) = \mathbf { D } ( { \mathcal { K } } _ { \infty } )$ for all N.

Consequently, no such profile characterizes realizable PAC learnability, and no finite-valued function of such a profile bounds the sample complexity on learnable classes.

Each target accepts, at every input x, one binary label $\mathbb { 1 } \{ x \in F \}$ , for a finite set F of inputs, together with all but finitely many auxiliary labels $2 , 3 , . . . ,$ and the accepted auxiliary labels are the same at every input. Hence any finite set of targets has a common auxiliary label that is correct everywhere, and the relabeling that swaps this label with 2 maps the finite set into the class $\mathcal { G }$ of targets that accept 2. The class $\mathcal { G }$ is learnable from zero examples, and by relabeling invariance and finite witnesses every coordinate of the profile takes the same value on $\mathcal { G }$ as on the whole class.

The whole class, in contrast, is not learnable from demonstrations. The demonstrator may show only binary labels, which are fair bits at unseen inputs, so a binary prediction there is wrong half of the time. An auxiliary prediction does not help either, because the target may exclude, after the learner is fixed, the auxiliary labels that the learner is likely to predict. Adding to $\mathcal { G }$ the targets with $F \subseteq [ N ]$ gives the intermediate classes $\kappa _ { N }$ , which are learnable but whose sample complexity grows with $N$

Construction: cofinite correct-label sets. Take $\chi = \mathbb { N }$ and $\mathcal { V } = \mathbb { N } _ { 0 }$ , and let $W =$ $\{ 2 , 3 , \ldots \}$ be the set of auxiliary labels. For a finite set $F \subseteq \mathbb { N }$ and a finite set $B \subseteq W$ , define

$$
h _ { F , B } ( x ) = \left\{ \mathbb { 1 } \{ x \in F \} \right\} \cup ( W \setminus B ) ,\tag{96}
$$

and set

$$
\begin{array} { r l } & { { \mathcal K } _ { \infty } = \{ h _ { F , B } : ~ F \subseteq \mathbb { N } \mathrm { ~ a n d ~ } B \subseteq W \mathrm { ~ b o t h ~ f i n i t e } \} , } \\ & { ~ \mathcal { G } = \{ h _ { F , B } \in { \mathcal K } _ { \infty } : ~ 2 \notin B \} , } \\ & { { \mathcal K } _ { N } = \mathcal { G } \cup \{ h _ { F , B } \in { \mathcal K } _ { \infty } : ~ F \subseteq [ N ] \} . } \end{array}\tag{97}
$$

These classes are countable, every correct-label set has finite complement, and $\begin{array} { r } { \bigcup _ { N > 1 } \mathcal { K } _ { N } = \mathcal { K } _ { \infty } } \end{array}$ The pair $( F , B )$ is determined by $h _ { F , B }$ , since $F = \{ x : 1 \in h _ { F , B } ( x ) \}$ and $B = W \bar { \backslash } h _ { F , B } ( x )$ for every x.

Proof of Theorem 88. Identical profiles. Let $\mathcal { V } = \{ h _ { F _ { 1 } , B _ { 1 } } , \dotsc , h _ { F _ { r } , B _ { r } } \} \subseteq \mathcal { K } _ { \infty }$ be finite and nonempty. Choose $y \in W \setminus ( \{ 2 \} \cup \bigcup _ { i } B _ { i } )$ , and let π swap 2 and y and fix all other labels. Then $\pi h _ { F _ { i } , B _ { i } } = h _ { F _ { i } , \pi ( B _ { i } ) }$ and $2 \notin \pi ( B _ { i } )$ , so π $\mathcal { V } \subseteq \mathcal { G }$ . For every relabeling invariant parameter D with the finite-witness property, it follows that

$$
\operatorname { D } ( { \mathcal { V } } ) = \operatorname { D } ( \pi { \mathcal { V } } ) \leq \operatorname { D } ( { \mathcal { G } } ) .
$$

Taking the supremum over all finite nonempty $\nu \subseteq \kappa _ { \infty }$ and using (89) gives $\mathrm { D } ( K _ { \infty } ) \le \mathrm { D } ( \mathcal { G } )$ and the inclusions $\mathcal { G } \subseteq \mathcal { K } _ { N } \subseteq \mathcal { K } _ { \infty }$ together with monotonicity give equality for every $\kappa _ { N }$ Applying this to each coordinate $\mathrm { D } _ { \alpha }$ proves part (iv).

Learnability of $\mathcal { G }$ and $\kappa _ { N }$ . The constant predictor 2 has error zero for every target in ${ \mathcal { G } } _ { : }$ which proves part (i). For $\kappa _ { N }$ , consider the deterministic learner that outputs

$$
\left\{ \begin{array} { l l } { \mathrm { t h e ~ c o n s t a n t ~ } a , } & { \mathrm { i f ~ } a \mathrm { ~ i s ~ t h e ~ f i r s t ~ a u x i l i a r y ~ l a b e l ~ i n ~ t h e ~ s a m p l e , } } \\ { \mathrm { t h e ~ c o n s t a n t ~ 2 , } } & { \mathrm { i f ~ n o ~ a u x i l i a r y ~ l a b e l ~ a p p e a r s , ~ b u t ~ s o m e ~ } ( x , 1 ) \mathrm { ~ w i t h ~ } x > N \mathrm { ~ d o e s , } } \\ { g _ { T } , } & { \mathrm { o t h e r w i s e , } } \end{array} \right.
$$

where $g _ { T } ( x ) = \mathbb { 1 } \{ x \in T \}$ and $T \subseteq [ N ]$ is the set of inputs demonstrated with label 1. Let the target be $h _ { F , B } \in \mathcal { K } _ { N }$ . In the first case, a is correct at some input, so $a \notin B$ and a is correct at every input. In the second case, $x \in F$ and $x > N$ , so $F \nsubseteq { \mathbb { N } } ]$ ; hence $h _ { F , B } \in \mathcal { G }$ and 2 is correct at every input. In the last case, every demonstrated label is binary and hence equals $\mathbb { 1 } \{ x \in F \}$ , so $g _ { T }$ agrees with all demonstrations, and $g _ { T }$ lies in the fixed family $\mathcal { Q } _ { N } = \{ g _ { T } : T \subseteq [ N ] \}$ of size $2 ^ { N }$ . Since a demonstrated label is correct, an invalid prediction never agrees with it; so a function $g \in \mathcal { Q } _ { N }$ of error $r > \varepsilon$ agrees with each demonstration with probability at most $1 - r ,$ and

$$
\Pr \left[ g ( x _ { i } ) = y _ { i } { \mathrm { ~ f o r ~ a l l ~ } } i \in [ m ] \right] \leq ( 1 - r ) ^ { m } \leq e ^ { - m \varepsilon } .
$$

A union bound over $\mathcal { Q } _ { N }$ bounds the failure probability by $2 ^ { N } e ^ { - m \varepsilon }$ , uniformly over all targets $\kappa _ { N }$ and all data laws. Hence $m _ { K _ { N } } ( \varepsilon , \delta ) \leq \lceil ( N \log 2 + \log ( 1 / \delta ) ) / \varepsilon \rceil < \infty$

Lower bounds for $\kappa _ { N }$ and $\kappa _ { \infty }$ . Fix a learner that uses m examples, an integer $N \geq 2$ and $\eta > 0$ . Choose $F \subseteq [ N ]$ by independent fair membership bits, and let $\mathcal { D } _ { F }$ be the joint law of $( x , 1 1 \{ x \in F \} )$ ) for x uniform on $[ N ]$ ; for every finite $B \subseteq W , { \mathcal { D } } _ { F }$ is compatible with $h _ { F , B }$ . Draw $\sigma \sim { \cal D } _ { F } ^ { m }$ , an independent test input $x \sim \mathrm { U n i f } ( [ N ] )$ , and the seed $\omega ,$ , and write $Z = { \widehat { g } } ( x )$ for the learner’s prediction at x. Since the joint law of $( F , \sigma , x , \omega )$ does not depend on $B$ , we may choose B after the learner. As $Z$ takes values in $ { \mathbb { N } } _ { 0 }$ , countable additivity gives an integer $L \geq 2$ with $\operatorname* { P r } [ Z > L ] \leq \eta$ , and we set $B = \{ 2 , \ldots , L \}$ , so that

$$
\operatorname* { P r } [ Z \in W \setminus B ] \leq \eta .\tag{98}
$$

Let $E$ be the event that the test input does not appear in $\sigma ;$ then $\mathrm { P r } [ E ] = ( 1 - 1 / N ) ^ { m }$ Conditional on $E , \sigma , x$ , and $\omega ,$ the unseen bit $\mathbb { 1 } \{ x \in F \}$ is fair, so $Z$ difers from it with probability at least $1 / 2$ (exactly $1 / 2$ if Z is binary, and 1 if Z is auxiliary). A prediction that difers from the bit is valid only if $Z \in W \setminus B$ . Therefore, with $P$ the uniform distribution on [N], (98) gives

$$
\mathbb { E } _ { F , \sigma , \omega } \big [ L _ { P , h _ { F } , B } ( \widehat { g } ) \big ] \geq \frac { \mathrm { P r } [ E ] } { 2 } - \mathrm { P r } [ Z \in W \setminus B ] \geq \frac { 1 } { 2 } \Big ( 1 - \frac { 1 } { N } \Big ) ^ { m } - \eta .
$$

Hence some $h _ { F , B } \in \mathcal { K } _ { N }$ with $F \subseteq [ N ]$ has expected error at least this large. Letting $\eta  0$ and taking the infimum over learners gives

$$
\mathfrak { L } _ { m } ( { K } _ { N } ) \ge \frac { 1 } { 2 } \Big ( 1 - \frac { 1 } { N } \Big ) ^ { m } .\tag{99}
$$

For $m \leq \lfloor N / 2 \rfloor$ , Bernoulli’s inequality bounds the right-hand side from below by $1 / 4$ , whereas a $( 1 / 1 6 , 1 / 1 6 ) – \mathrm { P A C }$ guarantee would give expected error at most $1 / 8$ by (4). This proves $m _ { K _ { N } } ( 1 / 1 6 , 1 / 1 6 ) > \lfloor N / 2 \rfloor$ and completes part (ii).

Since ${ \cal K } _ { N } \subseteq { \cal K } _ { \infty }$ , letting $N  \infty$ in (99) gives $\mathfrak { L } _ { m } ( { K } _ { \infty } ) \ge 1 / 2$ . Conversely, the learner that ignores the sample and outputs a uniformly random constant predictor in $\{ 0 , 1 \}$ has expected error exactly $1 / 2$ for every target, since exactly one of 0 and 1 is correct at each input. Hence $\mathfrak { L } _ { m } ( { K } _ { \infty } ) = 1 / 2$ for every $m ,$ and $\kappa _ { \infty }$ is not realizably PAC learnable, which proves part (iii).

Finally, by part $( \mathrm { i v } )$ the common profile does not distinguish the learnable class G from the unlearnable class $\kappa _ { \infty }$ . Nor does it bound the sample complexity: at $( \varepsilon , \delta ) = ( 1 / 1 6 , 1 / 1 6 )$ such a bound would take the same value on every $\kappa _ { N }$ , contradicting part (ii) as $N  \infty$ .

Remark 89 (Full-set feedback for the cofinite construction). The scale-sensitive counterexample is specific to one-correct-label feedback. Every full correct-label set of $h _ { F , B }$ contains an auxiliary label $a \in W \setminus B$ , and such a label is correct at every input. Thus one observed set sufices: the learner that outputs its least auxiliary label at every input has error zero for every target in $\kappa _ { \infty }$ . The scale-free obstruction of Theorem $8 6 .$ , in contrast, holds under full-set feedback as well.

## References

[AACSZ23] Ishaq Aden-Ali, Yeshwanth Cherapanamjeri, Abhishek Shetty, and Nikita Zhivotovskiy. Optimal PAC bounds without uniform convergence. In 2023 IEEE 64th Annual Symposium on Foundations of Computer Science (FOCS), pages 1203–1223. IEEE, 2023.

[ACBFS02] Peter Auer, Nicolò Cesa-Bianchi, Yoav Freund, and Robert E. Schapire. The nonstochastic multiarmed bandit problem. SIAM Journal on Computing, 32(1):48–77, 2002.

[ADD<sup>+</sup>24] Julian Asilis, Siddartha Devic, Shaddin Dughmi, Vatsal Sharan, and Shang-Hua Teng. Regularization and optimal multiclass learning. In Shipra Agrawal and Aaron Roth, editors, Proceedings of Thirty Seventh Conference on Learning Theory, volume 247 of Proceedings of Machine Learning Research, pages 260– 310. PMLR, 30 Jun–03 Jul 2024.

[AHHM21] Noga Alon, Steve Hanneke, Ron Holzman, and Shay Moran. A theory of PAC learnability of partial concept classes. In 2021 IEEE 62nd Annual Symposium on Foundations of Computer Science (FOCS), pages 658–671. IEEE, 2021.

[BBCB<sup>+</sup>25] Marco Bressan, Nataly Brukhim, Nicolò Cesa-Bianchi, Emmanuel Esposito, Yishay Mansour, Shay Moran, and Maximilian Thiessen. A fine-grained characterization of PAC learnability. In Proceedings ofThirty Eighth Conference on Learning Theory, volume 291 of Proceedings of Machine Learning Research, pages 641–676. PMLR, 2025.

[BBDG<sup>+</sup>21] Kaspars Balodis, Shalev Ben-David, Mika Göös, Siddhartha Jain, and Robin Kothari. Unambiguous DNFs and Alon-Saks-Seymour. In 2021 IEEE 62nd Annual Symposium on Foundations of Computer Science (FOCS), pages 116– 124. IEEE, 2021.

[BCD<sup>+</sup>22] Nataly Brukhim, Daniel Carmon, Irit Dinur, Shay Moran, and Amir Yehudayof. A characterization of multiclass learnability. In 2022 IEEE 63rd Annual Symposium on Foundations of Computer Science (FOCS), pages 943–955. IEEE, 2022.

[BDCBHL95] Shai Ben-David, Nicolò Cesa-Bianchi, David Haussler, and Philip M. Long. Characterizations of learnability for classes of {0, . . . , n}-valued functions. Journal of Computer and System Sciences, 50(1):74–86, 1995.

[BDHM<sup>+</sup>19] Shai Ben-David, Pavel Hrubeš, Shay Moran, Amir Shpilka, and Amir Yehudayof. Learnability can be undecidable. Nature Machine Intelligence, 1(1):44–48, 2019.

[BEHW89] Anselm Blumer, Andrzej Ehrenfeucht, David Haussler, and Manfred K. Warmuth. Learnability and the Vapnik-Chervonenkis dimension. Journal of the ACM, 36(4):929–965, 1989.

[BHM23] Nataly Brukhim, Steve Hanneke, and Shay Moran. Improper multiclass boosting. In Proceedings of Thirty Sixth Conference on Learning Theory, volume 195 of Proceedings of Machine Learning Research, pages 5433–5452. PMLR, 2023.

[CEH<sup>+</sup>26] Alon Cohen, Liad Erez, Steve Hanneke, Tomer Koren, Yishay Mansour, Shay Moran, and Qian Zhang. Sample complexity of agnostic multiclass classification: Natarajan dimension strikes back. In Proceedings of the 58th Annual ACM Symposium on Theory of Computing (STOC), pages 722–733. ACM, 2026.

[CP23] Moses Charikar and Chirag Pabbaraju. A characterization of list learnability. In Proceedings of the 55th Annual ACM Symposium on Theory of Computing (STOC), pages 1713–1726. ACM, 2023.

[CST11] Timothee Cour, Ben Sapp, and Ben Taskar. Learning from partial labels. Journal of Machine Learning Research, 12:1501–1536, 2011.

[DH13] Amit Daniely and Tom Helbertal. The price of bandit information in multiclass online classification. In Proceedings of the 26th Annual Conference on Learning Theory, volume 30 of Proceedings of Machine Learning Research, pages 93–104. PMLR, 2013.

[DMY16] Ofir David, Shay Moran, and Amir Yehudayof. Supervised learning through the lens of compression. In Advances in Neural Information Processing Systems, volume 29, pages 2784–2792, 2016.

[DSBDSS11] Amit Daniely, Sivan Sabato, Shai Ben-David, and Shai Shalev-Shwartz. Multiclass learnability and the ERM principle. In Proceedings of the 24th Annual Conference on Learning Theory, volume 19 of Proceedings of Machine Learning Research, pages 207–232. PMLR, 2011.

[DSS14] Amit Daniely and Shai Shalev-Shwartz. Optimal learners for multiclass problems. In Proceedings of The 27th Conference on Learning Theory, volume 35 of Proceedings of Machine Learning Research, pages 287–316. PMLR, 2014.

[ECK<sup>+</sup>24a] Liad Erez, Alon Cohen, Tomer Koren, Yishay Mansour, and Shay Moran. Fast rates for bandit PAC multiclass classification. In Advances in Neural Information Processing Systems, volume 37, pages 75152–75176, 2024.

[ECK<sup>+</sup>24b] Liad Erez, Alon Cohen, Tomer Koren, Yishay Mansour, and Shay Moran. The real price of bandit information in multiclass classification. In Proceedings of Thirty Seventh Conference on Learning Theory, volume 247 of Proceedings of Machine Learning Research, pages 1573–1598. PMLR, 2024.

[FHMM24] Yuval Filmus, Steve Hanneke, Idan Mehalel, and Shay Moran. Bandit-feedback online multiclass classification: Variants and tradeofs. In Advances in Neural Information Processing Systems, volume 37, pages 129605–129639, 2024.

[Fox26] Jacob Fox. A note on the Alon-Saks-Seymour problem. arXiv preprint arXiv:2605.28915, 2026.

[FS97] Yoav Freund and Robert E. Schapire. A decision-theoretic generalization of on-line learning and an application to boosting. Journal of Computer and System Sciences, 55(1):119–139, 1997.

[FSSW97] Yoav Freund, Robert E. Schapire, Yoram Singer, and Manfred K. Warmuth. Using and combining predictors that specialize. In Proceedings of the 29th Annual ACM Symposium on Theory of Computing (STOC), pages 334–343, 1997.

[HKLM22] Max Hopkins, Daniel M. Kane, Shachar Lovett, and Gaurav Mahajan. Realizable learning is all you need. In Proceedings of Thirty Fifth Conference on Learning Theory, volume 178 of Proceedings of Machine Learning Research, pages 3015–3069. PMLR, 2022.

[HLW94] David Haussler, Nick Littlestone, and Manfred K. Warmuth. Predicting {0, 1}-functions on randomly drawn points. Information and Computation, 115(2):248–292, 1994.

[HMM26] Steve Hanneke, Idan Mehalel, and Shay Moran. Sample complexity of autoregressive reasoning: Chain-of-thought vs. end-to-end. arXiv preprint arXiv:2604.12013, 2026.

[HMMS26a] Steve Hanneke, Qinglin Meng, Shay Moran, and Amirreza Shaeiri. An optimal Sauer lemma over k-ary alphabets. arXiv preprint arXiv:2604.12952, 2026.

[HMMS26b] Steve Hanneke, Qinglin Meng, Shay Moran, and Amirreza Shaeiri. PAC learning with bandit feedback: Sharp sample complexity in the realizable setting. In Advances in Neural Information Processing Systems, 2026. To appear. arXiv:2605.25678.

[HMS25] Steve Hanneke, Qinglin Meng, and Amirreza Shaeiri. Representation preserving multiclass agnostic to realizable reduction. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 21995–22008. PMLR, 2025.

[Hoe63] Wassily Hoefding. Probability inequalities for sums of bounded random variables. Journal of the American Statistical Association, 58(301):13–30, 1963.

[HRSS24] Steve Hanneke, Vinod Raman, Amirreza Shaeiri, and Unique Subedi. Multiclass transductive online learning. In Advances in Neural Information Processing Systems, volume 37, pages 60403–60428, 2024.

[HS12] Hao Huang and Benny Sudakov. A counterexample to the Alon-Saks-Seymour conjecture and related problems. Combinatorica, 32(2):205–219, 2012.

[HS25] Steve Hanneke and Amirreza Shaeiri. A trichotomy for list transductive online learning. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research. PMLR, 2025.

[HS26] Steve Hanneke and Amirreza Shaeiri. A theory of adversary-directed online learning. In Advances in Neural Information Processing Systems, 2026.

[HSW25a] Steve Hanneke, Amirreza Shaeiri, and Hongao Wang. For universal multiclass online learning, bandit feedback and full supervision are equivalent. In Proceedings of the 36th International Conference on Algorithmic Learning Theory, volume 272 of Proceedings of Machine Learning Research. PMLR, 2025.

[HSW25b] Steve Hanneke, Amirreza Shaeiri, and Hongao Wang. Non-uniform multiclass learning with bandit feedback. In Advances in Neural Information Processing Systems, volume 38, 2025.

[HSZ25] Steve Hanneke, Amirreza Shaeiri, and Qian Zhang. Universal rates for multiclass learning with bandit feedback. In Proceedings of Thirty Eighth Conference on Learning Theory, volume 291 of Proceedings of Machine Learning Research, pages 2704–2756. PMLR, 2025.

[HY23] Steve Hanneke and Liu Yang. Bandit learnability can be undecidable. In Proceedings of Thirty Sixth Conference on Learning Theory, volume 195 of Proceedings of Machine Learning Research, pages 5813–5849. PMLR, 2023.

[JLB<sup>+</sup>26] Nirmit Joshi, Gene Li, Siddharth Bhandari, Shiva Prasad Kasiviswanathan, Cong Ma, and Nathan Srebro. Learning to answer from correct demonstrations. In International Conference on Learning Representations, 2026.

[JVB<sup>+</sup>25] Nirmit Joshi, Gal Vardi, Adam Block, Surbhi Goel, Zhiyuan Li, Theodor Misiakiewicz, and Nathan Srebro. A theory of learning with autoregressive chain of thought. In Proceedings of Thirty Eighth Conference on Learning Theory, volume 291 of Proceedings of Machine Learning Research, pages 3161– 3212. PMLR, 2025.

[Kec95] Alexander S. Kechris. Classical Descriptive Set Theory, volume 156 of Graduate Texts in Mathematics. Springer-Verlag, New York, 1995.

[KM24] Jon Kleinberg and Sendhil Mullainathan. Language generation in the limit. In Advances in Neural Information Processing Systems, volume 37, pages 66058–66079, 2024.

[Kos26] Vanessa Kosoy. Ambiguous online learning. In Proceedings of Thirty Ninth Conference on Learning Theory, volume 336 of Proceedings of Machine Learning Research, pages 4229–4266. PMLR, 2026.

[KSST08] Sham M. Kakade, Shai Shalev-Shwartz, and Ambuj Tewari. Eficient bandit algorithms for online multiclass prediction. In Proceedings of the 25th International Conference on Machine Learning (ICML), pages 440–447. ACM, 2008.

[LBD24] Tosca Lechner and Shai Ben-David. Inherent limitations of dimensions for characterizing learnability of distribution classes. In Proceedings of Thirty Seventh Conference on Learning Theory, volume 247 of Proceedings of Machine Learning Research, pages 3353–3374. PMLR, 2024.

[LBD26] Tosca Lechner and Shai Ben-David. No scale sensitive dimension for distribution learning. In Proceedings of the 37th International Conference on Algorithmic Learning Theory, volume 313 of Proceedings of Machine Learning Research, pages 1–27. PMLR, 2026.

[LD14] Li-Ping Liu and Thomas G. Dietterich. Learnability of the superset label learning problem. In Proceedings of the 31st International Conference on Machine Learning, volume 32 of Proceedings of Machine Learning Research, pages 1629–1637. PMLR, 2014.

[Lon20] Philip M. Long. New bounds on the price of bandit feedback for mistakebounded online multiclass learning. Theoretical Computer Science, 808:159–163, 2020.

[LW86] Nick Littlestone and Manfred K. Warmuth. Relating data compression and learnability. Technical report, Department of Computer and Information Sciences, University of California at Santa Cruz, 1986.

[Mal24] Eran Malach. Auto-regressive next-token predictors are universal learners. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 34417–34431. PMLR, 2024.

[MV09] Dhruv Mubayi and Sundar Vishwanathan. Bipartite coverings and the chromatic number. The Electronic Journal of Combinatorics, 16(1):N34, 2009.

[MY16] Shay Moran and Amir Yehudayof. Sample compression schemes for VC classes. Journal of the ACM, 63(3):21:1–21:10, 2016.

[Nat89] Balas K. Natarajan. On learning sets and functions. Machine Learning, 4(1):67–97, 1989.

[O’D14] Ryan O’Donnell. Analysis of Boolean Functions. Cambridge University Press, 2014.

[Pab26a] Chirag Pabbaraju. The optimal sample complexity of multiclass and list learning. arXiv preprint arXiv:2604.24749, 2026.

[Pab26b] Chirag Pabbaraju. Optimal unambiguous DNFs and Alon-Saks-Seymour. arXiv preprint arXiv:2608.02533, 2026.

[PMBD26] Alireza F. Pour, Farnam Mansouri, and Shai Ben-David. Learning with multiple correct answers – regret bounds under diferent feedback models. arXiv preprint arXiv:2602.09402, 2026.

[RBR06] Benjamin I. P. Rubinstein, Peter L. Bartlett, and J. Hyam Rubinstein. Shifting, one-inclusion mistake bounds and tight multiclass expected risk bounds. In Advances in Neural Information Processing Systems, volume 19, pages 1193– 1200, 2006.

[RRS<sup>+</sup>24] Ananth Raman, Vinod Raman, Unique Subedi, Idan Mehalel, and Ambuj Tewari. Multiclass online learnability under bandit feedback. In Proceedings of The 35th International Conference on Algorithmic Learning Theory, volume 237 of Proceedings of Machine Learning Research, pages 997–1012. PMLR, 2024.

[RST24] Vinod Raman, Unique Subedi, and Ambuj Tewari. Online learning with set-valued feedback. In Proceedings of Thirty Seventh Conference on Learning Theory, volume 247 of Proceedings of Machine Learning Research, pages 4381– 4412. PMLR, 2024.

[SFLT26] Shihao Shao, Cong Fang, Zhouchen Lin, and Dacheng Tao. Partial feedback online learning. arXiv preprint arXiv:2601.21462, 2026.

[Val84] Leslie G. Valiant. A theory of the learnable. Communications of the ACM, 27(11):1134–1142, 1984.

[VC71] Vladimir N. Vapnik and Alexey Ya. Chervonenkis. On the uniform convergence of relative frequencies of events to their probabilities. Theory of Probability & Its Applications, 16(2):264–280, 1971.

[VC74] Vladimir N. Vapnik and Alexey Ya. Chervonenkis. Theory of Pattern Recognition. Nauka, Moscow, 1974.

[vN28] John von Neumann. Zur Theorie der Gesellschaftsspiele. Mathematische Annalen, 100(1):295–320, 1928.

[ZYW26] Guangjian Zhang, Zhengyi Yang, and Xiaoyang Wang. Bandit multiclass PAC learning: Structural baselines and a confidence direct sum. arXiv preprint arXiv:2609.29694, 2026.

[ZZ14] Min-Ling Zhang and Zhi-Hua Zhou. A review on multi-label learning algorithms. IEEE Transactions on Knowledge and Data Engineering, 26(8):1819–1837, 2014.

## A Full-Set Learnability and the Joint DS Dimension Are Incomparable

This appendix proves the last assertion of Theorem 45: for full-set feedback and $\ell \geq 2$ , the joint DS dimension is neither necessary nor suficient for learnability. We give two total classes with at most two correct labels per input, so both lie in the regime $\ell \geq 2$ of that theorem. The first has joint DS dimension one but is not full-set PAC learnable (Theorem 90); the second has infinite joint DS dimension but is full-set PAC learnable (Theorem 91).

Both constructions use $\chi = \mathbb { N }$ and a countable label space Y that contains 0, 1, ⊥, and a fresh symbol ⟨t⟩ for each finite tuple t used in the construction. We call these symbols tuple labels; they are distinct from each other and from 0, 1, ⊥, and a tuple label reveals its entries. In both classes a hypothesis accepts, at each input $i \leq n$ , a bit $b _ { i }$ and a tuple label that encodes $b _ { i }$ together with some side information about the other inputs. The two constructions difer only in this side information. In the first, it couples the bits in pairs, which destroys DS witnesses but tells the learner almost nothing about the bit at an unseen input. In the second, it hands the learner a predictor that is consistent with the whole sample, but leaves each bit free to flip, which produces large witnesses.

## A.1 Joint DS Dimension One Does Not Imply Full-Set Learnability

Fix $n \geq 2$ , a vector $b \in \{ 0 , 1 \} ^ { n }$ , and integers $u _ { i j } \in [ n ]$ for $1 \leq i < j \leq n$ . Define the ofsets

$$
r _ { i j } = u _ { i j } , \qquad r _ { j i } = u _ { i j } + b _ { i } b _ { j } \qquad ( i < j ) ,\tag{100}
$$

so that the two ofsets of a pair difer exactly by $b _ { i } b _ { j } .$ , and write $r _ { i } = ( r _ { i j } ) _ { j \in [ n ] \backslash \{ i \} }$ , with the indices in increasing order. Define the total hypothesis

$$
h _ { n , b , u } ( i ) = \left\{ \{ \begin{array} { l l } { \{ b _ { i } , \langle n , i , b _ { i } , r _ { i } \rangle \} , } & { i \leq n , } \\ { \{ 0 \} , } & { i > n , } \end{array} \right.\tag{101}
$$

and let $\mathcal { H } ^ { - }$ consist of all these hypotheses, over every $n \geq 2$ and all choices of b and u.

Every correct label at an input $i \leq n$ determines $b _ { i } .$ since it is either $b _ { i }$ itself or a tuple label whose third entry is $b _ { i }$ . Thus predicting a correct label at i is as hard as guessing $b _ { i }$ . If the tuple labels recorded only $( n , i , b _ { i } )$ , the class would contain product witnesses of every size. The ofsets destroy them: the tuple labels at two inputs $i < j$ reveal $r _ { i j }$ and $r _ { j i }$ , hence the product $b _ { i } b _ { j }$ , and this coupling of the two bits rules out DS witnesses on two inputs.

The coupling is of little use to a learner. It does not see the tuple label at the test input $i ,$ and the only trace of $b _ { i }$ in its sample is the ofset $r _ { j i } = u _ { i j } + b _ { i }$ at observed inputs $j > i$ with $b _ { j } = 1$ . When $u _ { i j }$ is uniform on [n], this ofset reveals $b _ { i }$ only if it equals 1 or $n + 1$ which happens with probability $1 / n$ . The lower bound therefore takes n much larger than the sample size, so that the test input is unseen and its bit is almost uniform given the sample.

Theorem 90. The class $\mathcal { H } ^ { - }$ satisfies

$$
d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { H } ^ { - } ) = d _ { \mathrm { N } } ^ { \cap } ( \mathcal { H } ^ { - } ) = 1 ,
$$

and the fixed list $\{ 0 , 1 \}$ meets every correct-label set of $\mathcal { H } ^ { - }$ . Nevertheless, for every sample size m, under full-set feedback,

$$
\operatorname* { i n f } _ { \mathcal { L } } \operatorname* { s u p } _ { h \in \mathcal { H } ^ { - } , P } \mathbb { E } \big [ L _ { P , h } ( \mathcal { L } ( \sigma ) ) \big ] = \frac { 1 } { 2 } ,\tag{102}
$$

where the infimum is over all learners $\mathcal { L }$ that receive m examples, including randomized learners and randomized predictions. In particular, H<sup>−</sup> is not full-set PAC learnable.

Proof. By Lemma 9, the projection of a DS witness to two of its coordinates is again a DS witness, so for the dimensions it sufices to rule out DS witnesses on two inputs. We do this first for hypotheses with a common value of $n _ { \mathrm { { ; } } }$ and then in general.

Two inputs and a fixed n. Fix n and two inputs $i < j \le n$ , and suppose that the patterns on $( i , j )$ of the hypotheses $h _ { n , b , u }$ with $( b , u )$ in a finite nonempty set Θ form a DS witness. Tildes carry over to ofsets, so $\widetilde { r }$ is defined from $( \widetilde { b } , \widetilde { u } )$ by (100). Since n is fixed, two members $( b , u ) , ( \widetilde { b } , \widetilde { u } )$ of Θ have the same correct-label set at i if and only if $( \widetilde { b } _ { i } , \widetilde { r } _ { i } ) = ( b _ { i } , r _ { i } )$ . The DS condition at $j$ says that, for every $( b , u ) \in \Theta$ , the members that agree with $( b , u )$ at i have no common correct label at $j \colon$

$$
\begin{array} { c } { { \displaystyle \bigcap _ { ( \widetilde { b } , \widetilde { u } ) \in \Theta } h _ { n , \widetilde { b } , \widetilde { u } } ( j ) = \emptyset . } } \\ { { \displaystyle ( \widetilde { b } , \widetilde { u } ) \in \Theta } } \\ { { ( \widetilde { b } _ { i } , \widetilde { r } _ { i } ) = ( b _ { i } , r _ { i } ) } } \end{array}\tag{103}
$$

The DS condition at i says the same with the roles of i and $j$ swapped: the members that agree with $( b , u )$ at $j$ have no common correct label at i.

Let $q = \operatorname* { m a x } \{ r _ { i j } : ( b , u ) \in \Theta \}$ . The idea is that, when $b _ { j } = 1$ , changing $b _ { i }$ from 1 to 0 while keeping the correct-label set at $j ,$ , and hence the ofset $r _ { j i } = r _ { i j } + b _ { i }$ , raises $r _ { i j }$ by one. At the maximum q this move is impossible, and the DS conditions propagate the obstruction to the other three bit pairs. We show in turn that no $( b , u ) \in \Theta$ with $r _ { i j } = q$ has $( b _ { i } , b _ { j } )$ equal to (1, 1), (1, 0), (0, 0), or (0, 1). Since q is attained, this is a contradiction. In each case, fix such a $( b , u )$

Case 1: $( b _ { i } , b _ { j } ) = ( 1 , 1 )$ . Then $r _ { j i } = q + 1$ . Every member that agrees with $( b , u )$ at $j$ has $\widetilde { b } _ { j } = 1$ and $\widetilde { r } _ { i j } + \bar { b } _ { i } = \widetilde { r } _ { j i } = q + 1$ , and since $\widetilde { r } _ { i j } \le q$ , also $\widetilde { b } _ { i } = 1$ . These members share the label 1 at i, contradicting the DS condition at i.

Case 2: $( b _ { i } , b _ { j } ) = ( 1 , 0 )$ . Every member that agrees with $( b , u )$ at i has $\widetilde { b } _ { i } = 1$ and $\widetilde { r } _ { i j } = q$ so $\widetilde { b } _ { j } = 0$ by Case 1. These members share the label 0 at $j ,$ contradicting the DS condition at $j .$

Case 3: $( b _ { i } , b _ { j } ) = ( 0 , 0 )$ . Then $r _ { j i } = q$ . Every member that agrees with $( b , u )$ at $j$ has $\widetilde { b } _ { j } = 0$ , hence $\widetilde { r } _ { i j } = \widetilde { r } _ { j i } = q$ , so $\widetilde { b } _ { i } = 0$ by Case 2. These members share the label 0 at $i ,$

contradicting the DS condition at $i .$

Case $\begin{array} { r l } { \small { 4 } \colon ( b _ { i } , b _ { j } ) = ( 0 , 1 ) } \end{array}$ . Every member that agrees with $( b , u )$ at i has $\widetilde { b } _ { i } = 0$ and $\widetilde { r } _ { i j } = q$ so $\widetilde { b } _ { j } = 1$ by Case 3. These members share the label 1 at $j ,$ , contradicting the DS condition at $j .$

Two inputs in general. Now let $i < j$ be arbitrary inputs, and suppose that the patterns on $( i , j )$ of the hypotheses $h _ { n , b , u }$ with $( n , b , u )$ in a finite nonempty set form a DS witness. We distinguish two cases.

If every triple has $n \geq j$ , the tuple labels at both inputs reveal $n ,$ so patterns that agree at either input come from the same value of n. Hence the patterns with any one value of n form a DS witness on $( i , j )$ by themselves, which is impossible by the fixed-n case.

Otherwise some triple has $n < j$ , and the DS condition at j fails at it. Indeed, if $n < i$ l, its correct-label set at i is $\{ 0 \}$ , so every member that agrees with it at i also has $n < i ;$ if $i \le n < j$ , its tuple label at i reveals n, so every member that agrees with it at i has the same n. In both cases all these members have the correct-label set {0} at $j .$

The dimensions. Thus $\mathcal { H }$ <sup>−</sup> has no DS witness on two inputs, and $d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { H } ^ { - } ) \leq 1$ . At a single input i, two hypotheses with $n \geq i$ and opposite bits $b _ { i }$ have disjoint correct-label sets, since their tuple labels also difer. Hence $d _ { \mathrm { N } } ^ { \Omega } ( \mathcal { H } ^ { - } ) \geq 1$ , and the inequality $d _ { \mathrm { N } } ^ { \cap } ( \mathcal { H } ^ { - } ) \leq d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { H } ^ { - } )$ of Lemma 9 gives the asserted equalities.

The upper bound in (102). Every correct-label set of $\mathcal { H }$ <sup>−</sup> contains exactly one of the labels 0 and 1, so a uniformly random constant predictor with value in $\{ 0 , 1 \}$ has expected error exactly $1 / 2$ for every target and every distribution.

A random target. For the lower bound, fix n and let $P _ { n }$ be uniform on [n]. Choose

$$
B _ { i } \sim \operatorname { U n i f } ( \{ 0 , 1 \} ) ( i \in [ n ] ) , \qquad U _ { i j } \sim \operatorname { U n i f } ( [ n ] ) ( i < j ) ,
$$

all independent, take the target $h _ { n , B , U }$ , and draw m independent full-set examples from it. Write $R _ { j k }$ for the ofsets defined by (100) with B and U in place of b and u. Let $T$ be the set of distinct training inputs, and let I be an independent test input with law $P _ { n }$ . On the event $\{ I \notin T \}$ , the sample reveals $B _ { j }$ and the ofsets $R _ { j k } , k \neq j$ , at each $j \in T$ . Of these, only the ofsets

$$
R _ { j I } = U _ { I j } + B _ { I } , \qquad j \in J = \{ j \in T : \ j > I , \ B _ { j } = 1 \} ,
$$

depend on $B _ { I }$ . Such an ofset is uniform on [n] if $B _ { I } = 0$ and uniform on $\{ 2 , \ldots , n + 1 \}$ if $B _ { I } = 1$ , so it is informative only outside $\{ 2 , \ldots , n \}$ . Define the event

$$
E = \{ I \not \in T \} \cap \bigcap _ { j \in J } \{ R _ { j I } \in \{ 2 , \dots , n \} \} ,\tag{104}
$$

which is determined by the sample and I.

Condition on the training inputs, on a test input $I \notin T$ , and on the observed bits $B _ { j }$ $j \in T ;$ ; this fixes $J ,$ and $B _ { I }$ is still uniform. The block $( B _ { I } , ( U _ { I j } ) _ { j \in J } )$ consists of independent variables, and it is independent of all other observed ofsets, which are functions of the bits $B _ { k } , k \neq I$ , and of the variables $U _ { i ^ { \prime } j ^ { \prime } }$ outside this block. This gives two facts.

(a) $\operatorname* { P r } [ E ] \geq ( 1 - 1 / n ) ^ { 2 m }$ . Indeed, $\operatorname* { P r } [ I \notin T ] = ( 1 - 1 / n ) ^ { m }$ . Given $I \notin T$ and either value of $B _ { I }$ , the at most m ofsets $R _ { j I } , j \in J$ , are conditionally independent, and each lies in $\{ 2 , \ldots , n \}$ with probability $1 - 1 / n$

(b) On E, the bit $B _ { I }$ is uniform on $\{ 0 , 1 \}$ given the sample and I. Indeed, every observed value $R _ { j I } \in \{ 2 , \ldots , n \} , j \in J _ { \mathrm { ~ } }$ , has likelihood $1 / n$ under both values of $B _ { I }$ , and the rest of the sample does not depend on $B _ { I }$ . The same holds given also the independent randomness of the learner.

It remains to show that a correct prediction at I reveals $B _ { I }$ . For a label $y ,$ let $\beta ( y ) \in \{ 0 , 1 \}$ be the bit it points to: $\beta ( 1 ) = 1 , \beta ( 0 ) = \beta ( \perp ) = 0$ , and $\beta ( \langle n ^ { \prime } , k , a , w \rangle ) = a$ . Set $\beta ( y ) = 0$ for all remaining labels. Both correct labels at $I \in [ n ]$ point to $B _ { I } ,$ so for every label $y ,$

$$
\mathbb { 1 } \{ y \notin h _ { n , B , U } ( I ) \} \geq \mathbb { 1 } \{ \beta ( y ) \neq B _ { I } \} .
$$

Whatever the learner predicts at I is a function of the sample, of I, and of randomness independent of $B _ { I } . \mathrm { \ B y \ ( b ) }$ , on E the prediction is incorrect with conditional probability at least $1 / 2$ . Averaging over the target, the sample, and the randomness of the learner, and using (a),

$$
\mathbb { E } \left[ L _ { P _ { n } , h _ { n , B , U } } ( \mathcal { L } ( \sigma ) ) \right] \geq \frac { 1 } { 2 } \operatorname* { P r } [ E ] \geq \frac { 1 } { 2 } ( 1 - 1 / n ) ^ { 2 m } .\tag{105}
$$

Some fixed target attains at least this average. Hence, for a fixed learner and a fixed $m _ { \mathrm { : } }$ , the worst-case expected error is at least ${ \textstyle \frac { 1 } { 2 } } \big ( 1 - 1 / n \big ) ^ { 2 m }$ for every $n ,$ and thus at least $1 / 2$ . Together with the upper bound, this proves (102). In particular, $\mathcal { H } ^ { - }$ is not full-set PAC learnable: by (4), a PAC guarantee with $\varepsilon = \delta = 1 / 8$ would bound the expected error by $1 / 4$ uniformly.

## A.2 A Full-Set Learnable Class with Infinite Joint DS Dimension

We identify a vector $a \in \{ 0 , 1 \} ^ { n }$ with the function $x \mapsto a _ { x }$ on [n]. For $n \geq 1$ , choose a vector $b \in \{ 0 , 1 \} ^ { n }$ and functions $f _ { 1 } , \ldots , f _ { n } : [ n ] \to \{ 0 , 1 \}$ for which there is a permutation $\pi$ of $[ n ]$ such that

$$
f _ { \pi ( t ) } ( \pi ( s ) ) = b _ { \pi ( s ) } \qquad ( 1 \leq s \leq t \leq n ) .\tag{106}
$$

Thus each function $f _ { i }$ is correct at its own input and at all inputs that precede it in this order. Define the total hypothesis

$$
h _ { n , b , f } ( i ) = \left\{ \begin{array} { l l } { \{ b _ { i } , \langle n , i , f _ { i } \rangle \} , } & { i \leq n , } \\ { \{ \bot \} , } & { i > n , } \end{array} \right.\tag{107}
$$

and let $\mathcal { H } ^ { + }$ consist of all these hypotheses.

The tuple label at i reveals a function $f _ { i }$ that predicts the bits at all inputs, but $f _ { i }$ is reliable only at i and at the inputs that precede it in a hidden order. The hidden order hurts the dimension but not the learner. On the one hand, an input i may always be moved to the end of the order, where no other function needs to be correct at it, so its bit can be flipped without changing the correct-label sets elsewhere; this produces disjoint-neighbor witnesses of every size. On the other hand, the observed input that comes last in the order supplies a function that is correct at every observed input, so some predictor supplied by a single example is consistent with the whole sample. A learner that outputs such a predictor is a sample compression scheme of size one, and it generalizes at the usual rate.

Theorem 91. The class $\mathcal { H } ^ { + }$ satisfies $d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { H } ^ { + } ) = \infty$ , and it has a deterministic full-set learner such that, for every target $h \in \mathcal { H } ^ { + }$ , every distribution $P$ on $\mathbb { N } _ { i }$ , every $m \geq 1$ , and every $0 < \varepsilon < 1$ ，

$$
\operatorname* { P r } \left[ L _ { P , h } ( \widehat { g } _ { m } ) > \varepsilon \right] \leq \operatorname* { m i n } \big \{ 1 , m ( 1 - \varepsilon ) ^ { m - 1 } \big \} ,\tag{108}
$$

where ${ \widehat { g } } _ { m }$ is its output on m examples. In particular, $\mathcal { H } ^ { + }$ is full-set PAC learnable.

Proof. We first construct witnesses of every size, and then analyze the learner described above.

Arbitrarily large witnesses. Fix n, and let $\mathcal { H } _ { n } ^ { + }$ consist of the hypotheses in $\mathcal { H } ^ { + }$ with this value of n. This class is finite, and it is nonempty because $f _ { i } = b$ for every i satisfies (106). We show that the patterns of $\mathcal { H } _ { n } ^ { + }$ on $( 1 , \ldots , n )$ form a disjoint-neighbor witness, so that $d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { H } ^ { + } ) \geq d _ { \mathrm { p a i r } } ( \mathcal { H } ^ { + } ) \geq n$ by Lemma 9; since n is arbitrary, this gives $d _ { \mathrm { D S } } ^ { \cap } ( \mathcal { H } ^ { + } ) = \infty$

Take a member $h _ { n , b , f }$ of $\mathcal { H } _ { n } ^ { + }$ , a permutation π as in (106), and an input $i \in [ n ]$ . Let $b ^ { \prime }$ be b with coordinate i flipped, and let $f _ { i } ^ { \prime } = b ^ { \prime }$ and $f _ { j } ^ { \prime } = f _ { j }$ for $j \neq i$ . These parameters satisfy (106) with the order $\pi ^ { \prime }$ obtained from π by moving i to the last position. Indeed, for $j \neq i$ the inputs that precede $j$ under $\pi ^ { \prime }$ also precede it under $\pi$ and difer from i, and $b ^ { \prime }$ agrees with b there and at $j$ , so $f _ { j } ^ { \prime } = f _ { j }$ is still correct at all of them. The input i is last under $\pi ^ { \prime } .$ and $f _ { i } ^ { \prime } = b ^ { \prime }$ is correct everywhere.

Therefore $h _ { n , b ^ { \prime } , f ^ { \prime } } \in \mathcal { H } _ { n } ^ { + }$ . It has the same correct-label set as $h _ { n , b , f }$ at every $j \neq i$ , since $b _ { j } ^ { \prime } = b _ { j }$ and $f _ { j } ^ { \prime } = f _ { j } . \mathrm { A t } i$ , the correct-label sets $\{ b _ { i } , \langle n , i , f _ { i } \rangle \}$ and $\{ 1 - b _ { i } , \langle n , i , b ^ { \prime } \rangle \}$ are disjoint: their bits difer, and so do their tuple labels, because $f _ { i } ( i ) = b _ { i } \neq b _ { i } ^ { \prime }$ . This proves the disjoint-neighbor property.

The learning rule. An example whose correct-label set contains a tuple label $c = \langle n , i , f \rangle$ supplies the predictor $g _ { c } : \mathbb { N }  \{ 0 , 1 , \bot \}$ given by $g _ { c } ( x ) = f ( x )$ for $x \leq n$ and $g _ { c } ( x ) = \bot \ \mathrm { f o r }$ $x > n ;$ an example with correct-label set $\{ \bot \}$ supplies the predictor $g _ { \perp }$ that is identically $\perp$ Call a predictor $g$ valid on a sample $\sigma = ( ( x _ { t } , h ( x _ { t } ) ) ) _ { t = 1 } ^ { m }$ if $g ( x _ { t } ) \in h ( x _ { t } )$ for all $t \in [ m ]$ . The learner returns the first supplied predictor, in the order of the examples, that is valid on the sample, and an arbitrary predictor if there is none.

On every sample from a target $h = h _ { n , b , f } \in \mathcal { H } ^ { + }$ , some supplied predictor is valid, so the learner always returns one. If every observed correct-label set is {⊥}, then $g _ { \perp }$ is valid.

Otherwise, fix a permutation π as in (106) for the target, and take the observed input in [n] that comes last under π. By (106), the function it supplies is correct at every observed input in [n], and its extension by ⊥ is correct at every observed input outside [n].

Generalization. Fix the target and the distribution, and let $g _ { t }$ be the predictor supplied by the tth example. Conditional on the tth example, $g _ { t }$ is fixed and the other $m - 1$ examples are still independent with law $\mathcal { D } _ { P , h }$ . Hence, if $L _ { P , h } ( g _ { t } ) > \varepsilon$ , then $g _ { t }$ is correct at all of them with probability at most $( 1 - \varepsilon ) ^ { m - 1 }$ , and

$$
\operatorname* { P r } \left[ L _ { P , h } ( g _ { t } ) > \varepsilon { \mathrm { ~ a n d ~ } } g _ { t } { \mathrm { ~ i s ~ v a l i d ~ o n ~ t h e ~ s a m p l e } } \right] \leq ( 1 - \varepsilon ) ^ { m - 1 } .
$$

The output is one of the $g _ { t }$ and is valid on the sample, so a union bound over $t \in [ m ]$ proves (108). Since $m ( 1 - \varepsilon ) ^ { m - 1 } \leq m e ^ { - \varepsilon ( m - 1 ) }$ , the right-hand side is at most δ once $m \geq C \varepsilon ^ { - 1 } \big ( \log ( 2 / \varepsilon ) + \log ( 2 / \delta ) \big )$ for a universal constant $C ,$ so $m _ { \mathcal { H } ^ { + } } ^ { \mathrm { f u l l } } ( \varepsilon , \delta ) = O \bigl ( \varepsilon ^ { - 1 } ( \log ( 2 / \varepsilon ) +$ log $\left( 2 / \delta \right) )$ □

## B Comparing Packed and DS-Type Dimensions of Intersection Classes

This appendix proves Theorem 60. Throughout, $\ell \geq 1$ and $0 < \rho \leq 1$ are fixed, and we use the projections, partial intersection classes, one-inclusion hypergraphs and dimensions of Sections 5.3 and 5.4. The class in question is $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ : its correct-label sets have at most ℓ labels, and the conflict blocks of its packed witnesses have at most $\ell + 1$ members. The heart of the proof is the following extraction theorem.

Theorem 92 (Witness extraction). Fix $\ell \geq 1$ and $0 < \rho \leq 1$ . There is a constant $c _ { \ell , \rho } > 0$ with the following property. For every ρ-packed witness $o f \operatorname { \mathcal { T } } _ { \ell + 1 } ( \mathcal { H } )$ on a sequence of $n \geq 1$ distinct inputs, and every choice of representing tuples for the patterns of its list, there are a set U of inputs of this sequence and a disjoint-neighbor witness $W \subseteq \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) | _ { U }$ such that

$$
| U | \geq c _ { \ell , \rho } \frac { n } { \log ( n + 2 ) } ,
$$

and every pattern of W is represented by at most $\ell + 1$ hypotheses drawn from the given representing tuples.

The proof is constructive: a finite procedure finds U and W from the given witness and tuples. With $d = d _ { \mathrm { p a i r } } ( \mathscr { T } _ { \ell + 1 } ( \mathcal { H } ) )$ , the theorem gives $c _ { \ell , \rho } n / \log ( n + 2 ) \leq d$ for every ρ-packed witness on n inputs, and (82) follows by a short calculation; since $d _ { \mathrm { p a i r } } \leq d _ { \mathrm { D S } } ^ { \cap }$ by Lemma 9, so does the bound in terms of the DS dimension (Section B.3). The whole argument takes place inside finitely many hypotheses and labels, so X and Y may be arbitrary sets.

Outline. Fix a ρ-packed witness on n inputs with list $\boldsymbol { v } ^ { 1 } , \ldots , \boldsymbol { v } ^ { M }$ (Definition 7). In each direction its conflict blocks have at most $\ell + 1$ members and cover at least $\rho M$ entries, so there are at least $\rho M n / ( \ell + 1 )$ blocks in all. Section B.1 states two lemmas that turn suitable configurations into disjoint-neighbor witnesses. The first applies to a product-like family of patterns indexed by $[ r ] ^ { n }$ in which, at every input i, the patterns with the same ith coordinate share a common label. The second applies to many conflict blocks whose sets at the varying input all meet one list of bounded width, and it yields a witness whose size is proportional to the number of blocks per pattern.

The main step, in Section B.2, supplies such lists. Every pattern receives a list table of constant width that is determined by its history: its correct-label sets on a common initial set of inputs and on a few further inputs that the procedure records. The patterns of a conflict block difer only at the varying input, so unless this input is recorded, they have the same history and share one list. The reduction either finds a witness on $t + 1$ inputs or records O(t log n) inputs per pattern, and the records must spoil only a small fraction of the blocks; so we take t of order n/ log n, which is the source of the logarithm in Theorem 92. Section B.3 combines the three lemmas, and Sections B.4 and B.5 prove the two extraction lemmas.

## B.1 Two Extraction Lemmas

We first need a few definitions.

Localization. Fix a ρ-packed witness of $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ on distinct inputs $x _ { 1 } , \ldots , x _ { n }$ , which we call the pool, with list $\boldsymbol { v } ^ { 1 } , \ldots , \boldsymbol { v } ^ { M }$ , collections $B _ { i }$ of conflict blocks, and a representing tuple of at most $\ell + 1$ hypotheses of H for each $v ^ { a }$ . These hypotheses are defined on the whole pool, since their intersection $v ^ { a }$ is. Let $\mathcal { H } _ { 0 }$ be the finite class of their restrictions to the pool, identify $x _ { i }$ with $i ,$ and write $\mathcal { V } _ { 0 }$ for the finite set of labels used by $\mathcal { H } _ { 0 }$ . Every member of $\mathcal { T } _ { R } ( \mathcal { H } _ { 0 } )$ then corresponds to a member of $\mathcal { T } _ { R } ( \mathcal { H } )$ supported on the pool, so every witness built from $\mathcal { T } _ { R } ( \mathcal { H } _ { 0 } )$ on $[ n ]$ is a witness of the same type for $\mathcal { T } _ { R } ( \mathcal { H } )$ on the corresponding inputs, with representing tuples drawn from $\mathcal { H } _ { 0 }$ . The lemmas below hold for every finite class $\mathcal { H } _ { 0 }$ of this kind, and all ties are broken by fixed orders on the inputs, the labels and the hypothesis tuples.

Partial patterns. The partial pattern of $g \in { \mathcal { T } } _ { R } ( { \mathcal { H } } _ { 0 } )$ is the vector $( g ( 1 ) , \ldots , g ( n ) ) \in$ $( { \cal { A } } \cup \{ \star \} ) ^ { n }$ , with ⋆ exactly at the inputs outside supp(g); a partial pattern without ⋆ is a pattern on the pool. For partial patterns v, w and an input $i , v _ { - i } = w _ { - i }$ means that v and w agree at every input other than $i ;$ in particular, they are undefined at the same inputs other than i. A representing tuple of $^ { g , }$ together with supp(g), determines its partial pattern.

Products with common labels. The first configuration has its patterns indexed by all vectors $b \in [ r ] ^ { n }$ . At an input $i ,$ the correct-label sets of the patterns with $b _ { i } = j$ form the jth group. These sets may vary with the other coordinates of $b ,$ but they all contain a fixed label $a _ { i } ^ { j }$ , the common label of the group, and they all lie in a set $Y _ { i } ^ { j }$ , where $Y _ { i } ^ { 1 } , \ldots , Y _ { i } ^ { r }$ have an empty intersection; thus no label is correct for all groups at i. If every set depended only on $b _ { i } .$ the patterns would form a product witness (Definition 3). The next lemma recovers such a product of fixed sets on a constant fraction of the inputs, even when the given patterns are intersections of many hypotheses. It is proved in Section B.4.

Lemma 93 (Extraction from a product with common labels). Fix integers ℓ, $R \geq 1$ and $r \geq 2$ . There is a constant $a _ { \ell , R , r } > 0$ with the following property. Let $\mathcal { H } _ { 0 }$ be a finite class of hypotheses defined on a finite pool, with correct-label sets of size at most ℓ. Suppose that, on n inputs of the pool, which we identify with $[ n ]$ , sets $Y _ { i } ^ { j }$ , labels $a _ { i } ^ { j }$ , and patterns $A ^ { b } \in \mathcal { T } _ { R } ( \mathcal { H } _ { 0 } ) | _ { [ n ] }$ $b \in [ r ] ^ { n }$ , satisfy

$$
a _ { i } ^ { b _ { i } } \in A _ { i } ^ { b } \subseteq Y _ { i } ^ { b _ { i } } , \qquad \bigcap _ { j = 1 } ^ { r } Y _ { i } ^ { j } = \emptyset \qquad ( i \in [ n ] ) .
$$

Given representing tuples for these patterns, one can construct a set $U \subseteq [ n ]$ of at least $\boldsymbol { a } _ { \ell , R , r } \boldsymbol { n }$ inputs and nonempty sets $T _ { i } ^ { j } , i \in U , j \in [ r ]$ , such that

$$
\prod _ { i \in U } \{ T _ { i } ^ { j } : j \in [ r ] \} \subseteq \mathcal { Z } _ { \ell } ( \mathcal { H } _ { 0 } ) | _ { U } , \qquad \bigcap _ { j = 1 } ^ { r } T _ { i } ^ { j } = \emptyset \quad ( i \in U ) .
$$

This product is a DS witness. For $r = 2$ , it is also a disjoint-neighbor witness.

For $r > 2$ , the product need not be a disjoint-neighbor witness. A second application of the lemma, to a binary product of intersections of its sets, turns it into one.

Corollary 94 (Binary extraction). Under the hypotheses of Lemma ${ \mathit { 9 3 } } ,$ one can construct a disjoint-neighbor witness in $\mathcal { T } _ { \ell } ( \mathcal { H } _ { 0 } )$ on at least $a _ { \ell , \ell ^ { 2 } , 2 } a _ { \ell , R , r } n$ inputs.

Proof. Let U and $T _ { i } ^ { j }$ be given by Lemma 93, and fix $i \in U$ . The sets $T _ { i } ^ { 1 } , \ldots , T _ { i } ^ { r }$ are nonempty, have at most ℓ labels, being values of members of $\mathcal { T } _ { \ell } ( \mathcal { H } _ { 0 } )$ , and have an empty intersection. By Lemma $^ { 8 , }$ some minimal empty-intersection family among them has between two and $\ell + 1$ members. Let $Z _ { i } ^ { 2 }$ be its member with the largest index and $Z _ { i } ^ { 1 }$ the intersection of the others; then $Z _ { i } ^ { 1 } \neq \emptyset$ by minimality and $Z _ { i } ^ { 1 } \cap Z _ { i } ^ { 2 } = \emptyset$ . Every pattern of the binary product $\textstyle \prod _ { i \in U } \{ Z _ { i } ^ { 1 } , Z _ { i } ^ { 2 } \}$ is the intersection of ℓ patterns of the product of the lemma. Indeed, given a choice of $Z _ { i } ^ { 1 }$ or $Z _ { i } ^ { 2 }$ at every $i \in U$ , write the chosen set at i as $T _ { i } ^ { j _ { i , 1 } } \cap \cdots \cap T _ { i } ^ { j _ { i , \ell } }$ , repeating indices if it is the intersection of fewer than $\ell$ of the sets. For $k \in [ \ell ]$ , the pattern that takes the set $T _ { i } ^ { j _ { i , k } }$ at every $i \in U$ belongs to the product, because the product allows independent choices at diferent inputs, and the intersection of these ℓ patterns is the given pattern of the binary product. Each pattern of the product of the lemma is represented by at most ℓ hypotheses of $\mathcal { H } _ { 0 } .$ , so

$$
\prod _ { i \in U } \{ Z _ { i } ^ { 1 } , Z _ { i } ^ { 2 } \} \subseteq \mathbb { Z } _ { \ell ^ { 2 } } ( \mathcal { H } _ { 0 } ) | _ { U } .
$$

Lemma 93, with $R = \ell ^ { 2 } , r = 2 , Y _ { i } ^ { j } = Z _ { i } ^ { j }$ , and any label of $Z _ { i } ^ { j }$ as $a _ { i } ^ { j }$ , now gives a binary product in $\mathcal { T } _ { \ell } ( \mathcal { H } _ { 0 } )$ with disjoint sets at every input, that is, a disjoint-neighbor witness, on at least $a _ { \ell , \ell ^ { 2 } , 2 } | U | \geq a _ { \ell , \ell ^ { 2 } , 2 } a _ { \ell , R , r } n$ inputs. Both applications keep representing tuples drawn from $\mathcal { H } _ { 0 }$ □

Lists. Lists of labels are standard in multiclass learning $\mathrm { [ B C D ^ { + } 2 2 }$ , CP23]; here a list succeeds when it hits a correct-label set. A list table $Q$ on the pool assigns a nonempty set $Q ( i ) \subseteq \mathcal { V } _ { 0 }$ to every input $i \in [ n ]$ , and its width is $\operatorname* { m a x } _ { i } | Q ( i ) |$ . It hits a set A at i if $Q ( i ) \cap A \neq \emptyset$ , and misses it otherwise. For a pattern v on the pool, we count the misses,

$$
\operatorname { m i s s } ( Q , v ) = { \big | } \{ i \in [ n ] : Q ( i ) \cap v _ { i } = \emptyset \} { \big | } .
$$

The second configuration consists of many conflict blocks whose sets at the varying input all meet a list of bounded width. The list is required to hit the sets only at the varying input of each block; this local condition lets us, in Section B.3, discard diferent blocks in diferent directions. The lemma is proved in Section B.5.

Lemma 95 (Extraction under a bounded list). Fix integers $\ell , R , L \geq 1$ . There is a constant $\beta _ { \ell , R , L } > 0$ with the following property. Let $\mathcal { H } _ { 0 }$ be a finite class of hypotheses defined on a finite pool $[ n ]$ , with correct-label sets of size at most ℓ. Let $\boldsymbol { v } ^ { 1 } , \ldots , \boldsymbol { v } ^ { M }$ be partial patterns of $\mathcal { T } _ { R } ( \mathcal { H } _ { 0 } )$ , with repetitions allowed and with given representing tuples, and let Q be a list table of width at most L. For each input $i ,$ let $B _ { i }$ be a family of pairwise disjoint subsets of [M] such that every $B \in B _ { i }$ satisfies

$$
\begin{array} { l } { { 2 \leq | B | \leq \ell + 1 , \qquad v _ { - i } ^ { a } = v _ { - i } ^ { a ^ { \prime } } \quad ( a , a ^ { \prime } \in B ) , } } \\ { { \displaystyle \bigcap _ { a \in B } v _ { i } ^ { a } = \emptyset , \qquad v _ { i } ^ { a } \cap Q ( i ) \neq \emptyset \quad ( a \in B ) . } } \end{array}
$$

If $\begin{array} { r } { N = \sum _ { i } \left| B _ { i } \right| > 0 } \end{array}$ , one can construct a disjoint-neighbor witness in $\mathcal { T } _ { \ell } ( \mathcal { H } _ { 0 } )$ on at least $\beta _ { \ell , R , L } N / M$ inputs.

## B.2 List Reduction

Lemma 95 needs one list table of bounded width for all the patterns of a block. The next lemma assigns to every pattern a list table that is determined by its history, in such a way that two patterns that difer only at an input recorded by neither of them have the same history; the procedure either does this or finds a disjoint-neighbor witness of the desired size.

Lemma 96 (Conservative list reduction). Fix $\ell \geq 1$ and $0 < \eta < 1$ . There exist an integer $J = J _ { \ell } \ge 1$ and constants $L _ { * } = L _ { * } ( \ell )$ and $K = K _ { \ell , \eta }$ with the following property. Let $\boldsymbol { v } ^ { 1 } , \ldots , \boldsymbol { v } ^ { M }$ be patterns of $\mathbb { Z } _ { \ell + 1 } ( \mathcal { H } _ { 0 } )$ on $[ n ]$ , with repetitions allowed, and let $t \geq 1$ be an integer such that

$$
u : = \lceil t / \eta \rceil < n , \qquad J ( t + 1 ) \leq n .
$$

A finite procedure either constructs a disjoint-neighbor witness in $\mathbb { Z } _ { \ell + 1 } ( \mathcal { H } _ { 0 } )$ on at least $t + 1$ inputs, or assigns to every index $a \in [ M ]$ a history $\tau _ { a }$ , a record set $T _ { a } \subseteq [ n ]$ , and a list table $Q _ { a }$ of width at most $L _ { * }$ such that, with $\Lambda = K ( t + 1 ) \log ( 2 e n )$ ，

$$
| T _ { a } | \leq u + \Lambda , \qquad \sum _ { a = 1 } ^ { M } \operatorname { m i s s } ( Q _ { a } , v ^ { a } ) \leq M ( \eta n + \Lambda ) .\tag{109}
$$

The history $\tau _ { a }$ determines the list table $Q _ { a }$ , and for all $a , b \in [ M ]$ and $i \in [ n ]$ 2

$$
v _ { - i } ^ { a } = v _ { - i } ^ { b } , \quad i \notin T _ { a } \cup T _ { b } \quad \Longrightarrow \quad \tau _ { a } = \tau _ { b } .\tag{110}
$$

Proof idea. The proof has two phases. In the first, suppose that no $t + 1$ inputs carry a disjoint-neighbor witness. Then the patterns on any $t + 1$ inputs can be deleted one hyperedge at a time, and each deletion assigns a list of at most ℓ labels to an observation. Taking unions of these lists over the subsets of a random set of u inputs gives lists of width $e ^ { O ( t ) }$ , and a leave-one-out argument bounds their average number of misses by $\eta n$ . In the second phase, we reduce the width to a constant in $O ( \log ( t + 2 ) )$ ) passes. A pass fixes the new list at an input before it looks at the correct-label set there, and it records the input only when the new list misses a set that the old list hit. The records are controlled by a potential that counts, with weights, the binary products realized by the remaining candidate patterns: if one of these products uses $J ( t + 1 )$ inputs, Lemma 93 gives the witness, and otherwise every record divides the potential by a large factor. The history of a pattern consists of its sets at the initial and the recorded inputs. At an unrecorded input every pattern undergoes the same deletion step, and this gives the stability (110).

Proof. Choose an integer $J \ge \operatorname* { m a x } \{ 1 , 1 / a _ { \ell , \ell + 1 , 2 } \}$ , where $_ { a _ { \ell , \ell + 1 , 2 } }$ is the constant of Lemma $9 3 .$ and an integer $L _ { * } \ge \operatorname* { m a x } \{ 1 6 , \ell ^ { 4 } \}$ . We fix in advance all orders and tie-breaking rules, the same for every index $a \in [ M ]$ . Constants denoted by $C$ depend only on $\ell , \eta$ and $J ,$ constants denoted by $C _ { \ell }$ depend only on $\ell ,$ and both may change from line to line.

Deletion and initialization. For every set $I \subseteq [ n ]$ of $t + 1$ inputs, we first fix a rule that assigns lists to observations on $I ,$ that is, to pairs $( i , w )$ of an input $i \in I$ and a pattern w on $I \setminus \{ i \}$ . Start with the finite set of patterns of $\mathbb { Z } _ { \ell + 1 } ( \mathcal { H } _ { 0 } )$ on $I ,$ , and repeat the following while some patterns remain. Choose a remaining pattern v and an input $i \in I$ such that $v _ { i }$ meets $w _ { i }$ for every remaining pattern w with w $\mathbf { \Phi } _ { - i } = v _ { - i }$ . Assign the list $v _ { i }$ to the observation $( i , v _ { - i } )$ , and delete all remaining patterns with this observation, that is, the hyperedge $e _ { i , v _ { - i } }$ of the remaining set. If at some point no such choice exists while a nonempty set remains, then every remaining pattern has, in every direction $i \in I$ , a remaining neighbor whose set at i is disjoint from its own; so the remaining set is a disjoint-neighbor witness in $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } _ { 0 } )$ on $t + 1$ inputs, and we return it. Otherwise the process deletes every pattern on I. After a deletion no remaining pattern has the observation $( i , v _ { - i } )$ , so each observation receives at most one list, and we give every unassigned observation a fixed list consisting of one label of $\mathcal { V } _ { 0 }$ . All lists have at most ℓ labels. Moreover, the list $v _ { i }$ meets the set at i of every pattern deleted together with $v .$ . Hence every pattern w on I has an input $i \in I ,$ , the one at which it was deleted, such that the list assigned to the observation $( i , w _ { - i } )$ meets $w _ { i }$

These rules give the initial lists. For a set $I _ { 0 } \subseteq [ n ]$ of u inputs and a pattern v on $[ n ]$ , let $Q _ { v , I _ { 0 } } ( i ) = v _ { i } { \mathrm { ~ f o r ~ } } i \in I _ { 0 }$ , and for $i \notin I _ { 0 }$ let $Q _ { v , I _ { 0 } } ( i )$ be the union, over all sets $I ^ { \prime } \subseteq I _ { 0 }$ of t inputs, of the list assigned by the rule on $I ^ { \prime } \cup \{ i \}$ to the observation $( i , v | _ { I ^ { \prime } } )$ . Since ${ \binom { u } { t } } \leq ( e u / t ) ^ { t }$ and $u \leq 2 t / \eta$ , every such list table has width at most

$$
L _ { 0 } : = \operatorname* { m a x } \Bigl \{ \ell , \ell { \binom { u } { t } } \Bigr \} \le \exp \bigl ( C ( t + 1 ) \bigr ) .\tag{111}
$$

We now bound the misses for a random $I _ { 0 }$ . Fix a pattern v on [n] and a set $I _ { 1 } \subseteq [ n ]$ of $u + 1$ inputs, which exists since $u < n$ . We claim that at most t inputs $i \in I _ { 1 }$ have $Q _ { v , I _ { 1 } \backslash \{ i \} } ( i ) \cap v _ { i } = \emptyset$ . Indeed, otherwise let $I \subseteq I _ { 1 }$ be a set of $t + 1$ such inputs. The pattern $v | _ { I }$ was deleted by the rule on I at some input $i \in I .$ , so the list assigned to the observation $( i , v | _ { I \backslash \{ i \} } )$ meets $v _ { i }$ . Since $I \setminus \{ i \}$ is a set of t inputs of $I _ { 1 } \setminus \{ i \}$ , this list is contained in $Q _ { v , I _ { 1 } \backslash \{ i \} } ( i )$ , a contradiction. Now let $I _ { 0 }$ be a uniformly random set of u inputs. Equivalently, we may first draw a uniform set $I _ { 1 }$ of $u + 1$ inputs, then choose i uniformly from $I _ { 1 }$ and set $I _ { 0 } = I _ { 1 } \setminus \{ i \}$ . This gives the same joint law for $I _ { 0 }$ and a uniform input $i \notin I _ { 0 }$ . Since $Q _ { v , I _ { 0 } }$ never misses on $I _ { 0 } \mathrm { { . } }$ the claim gives

$$
\mathbb { E } \big [ \mathrm { m i s s } ( Q _ { v , I _ { 0 } } , v ) \big ] \leq \frac { t ( n - u ) } { u + 1 } \leq \eta n ,
$$

where the last inequality uses $u \geq t / \eta$ . Choose one set $I _ { 0 }$ that minimizes $\begin{array} { r } { \sum _ { a } \operatorname* { m i s s } ( Q _ { v ^ { a } , I _ { 0 } } , v ^ { a } ) } \end{array}$ ; the same set is used for every index. Write $Q _ { a , 0 } = Q _ { v ^ { a } , I _ { 0 } }$ . By averaging,

$$
\sum _ { a = 1 } ^ { M } \mathrm { m i s s } ( Q _ { a , 0 } , v ^ { a } ) \leq \eta M n .\tag{112}
$$

The set $I _ { 0 }$ is the first part of every record set $T _ { a }$ , and the sets $v _ { i } ^ { a } , i \in I _ { 0 }$ , are the first part of every history $\tau _ { a }$

Encoded profiles. The initial lists have few misses in total, but their width may be exponential in t, and we reduce it in passes. To decide which labels a shorter list must keep, we measure how many binary products the remaining candidate patterns can still realize, since a large binary product is exactly what Lemma 93 needs. Products are counted after coding every set by its trace on the old list, and a random coloring of the labels outside the list makes this coding faithful for disjointness: retained sets with disjoint codes are disjoint.

Formally, a pass starts from a list table $Q$ of width at most $L \geq 2$ . The reset class of Q is obtained by restricting every partial pattern A of $\mathbb { Z } _ { \ell + 1 } ( \mathcal { H } _ { 0 } )$ on $[ n ]$ to its hit domain,

$$
A \longmapsto A { \big | } _ { \{ i \in \operatorname { s u p p } ( A ) : A _ { i } \cap Q ( i ) \neq \emptyset \} } ;
$$

it consists of partial patterns of $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } _ { 0 } )$ , the one with empty support included. Thus every set attained at an input i by a member of the reset class meets $Q ( i )$ . Independently at every input i, color each label $y \in \mathcal { V } _ { 0 } \backslash Q ( i )$ by a uniform label $z _ { i } ( y ) \in Q ( i )$ . A set E at i is retained if

$$
E _ { Q } : = E \cap Q ( i ) \neq \emptyset \qquad { \mathrm { a n d } } \qquad z _ { i } ( E \setminus Q ( i ) ) \subseteq E _ { Q } ,
$$

and it is then encoded by $E _ { Q } ;$ a set that is not retained becomes ⋆. For a set V of partial patterns, let $V _ { z }$ be the set of their encodings. A profile π of $V _ { z }$ consists of a set $\operatorname { s u p p } ( \pi )$ of inputs and, at every $i \in \operatorname { s u p p } ( \pi )$ , an unordered pair of disjoint nonempty encoded sets, such that every choice of one set from each pair is realized by a member of $V _ { z }$ defined on all of $\operatorname { s u p p } ( \pi )$ . The profile with empty support is a profile of $V _ { z }$ whenever $V \neq \emptyset$ . Let $\Pi ( V _ { z } )$ be the set of profiles of $V _ { z }$

Extraction from a large profile. Before every pass, we enumerate the profiles of $V _ { z }$ for the reset class V of the current list and for all of the finitely many colorings z. Suppose that some profile π has $| \operatorname { s u p p } ( \pi ) | \geq J ( t + 1 )$ . Restrict it to $J ( t + 1 )$ inputs of its support, identified with $[ J ( t + 1 ) ]$ , order the two sets of each pair, and for every $b \in [ 2 ] ^ { J ( t + 1 ) }$ choose a member $A ^ { b }$ of the reset class whose encoding realizes the choice b. These are partial patterns of $\mathbb { Z } _ { \ell + 1 } ( \mathcal { H } _ { 0 } )$ defined on all $J ( t + 1 )$ inputs, and we keep representing tuples for them. At an input i of the restricted support, let $a _ { i } ^ { j }$ be any label of the jth encoded set and $\textstyle Y _ { i } ^ { j } = \bigcup _ { b : b _ { i } = j } A _ { i } ^ { b }$ , for $j \in [ 2 ]$ An encoded set is contained in every set it encodes, so $a _ { i } ^ { b _ { i } } \in A _ { i } ^ { b } \subseteq Y _ { i } ^ { b _ { i } }$ . Moreover $Y _ { i } ^ { 1 } \cap Y _ { i } ^ { 2 } = \emptyset$ a common label in $Q ( i )$ would lie in both encoded sets, and a common label $y \notin Q ( i )$ would have its color $z _ { i } ( y )$ in both, whereas the two encoded sets are disjoint. Lemma 93, with $R = \ell + 1$ and $r = 2$ , therefore constructs a disjoint-neighbor witness in ${ \mathscr T } _ { \ell } ( \mathcal { H } _ { 0 } ) \subseteq { \mathscr T } _ { \ell + 1 } ( \mathcal { H } _ { 0 } )$ on at least $a _ { \ell , \ell + 1 , 2 } J ( t + 1 ) \geq t + 1$ inputs, which we return.

Otherwise, for every coloring, every profile of the reset class has support of size less than $J ( t + 1 )$ . The same bound holds for every class obtained from the reset class by conditioning on correct-label sets and deleting inputs. Indeed, the realizers of a profile of such a class are restrictions of members of the reset class, and their encodings at the inputs of the support do not depend on the colors at the deleted inputs; so the profile is also a profile of the reset class under any extension of the coloring. This bound on the profiles controls the cost of the updates below.

The profile potential. During a pass with list table $Q$ of width at most $L ,$ the states are nonempty sets $V$ of partial patterns on the unprocessed inputs obtained from the reset class in this way. For such V put

$$
\Psi _ { Q } ( V ) = \mathbb { E } _ { z } \Big [ \sum _ { \pi \in \Pi ( V _ { z } ) } \lambda ^ { | \operatorname { s u p p } ( \pi ) | } \Big ] , \qquad \lambda = L ^ { 2 ( \ell - 1 ) } ,
$$

where z is the random coloring of the unprocessed inputs. The profile with empty support gives $\Psi _ { Q } ( V ) \geq 1$ . At each input, an encoded set is a nonempty subset of $Q ( i )$ with at most ℓ labels, so there are at most $C _ { \ell } ( L + 1 ) ^ { 2 \ell }$ pairs of encoded sets. Supports have fewer than $J ( t + 1 ) \leq n$ inputs, so bounding the number of supports of size $j$ by $\binom { n } { j }$ and using $\begin{array} { r } { \sum _ { j \le J ( t + 1 ) } { \binom { n } { j } } \le \left( e n / ( J ( t + 1 ) ) \right) ^ { J ( t + 1 ) } } \end{array}$ , valid since $J ( t + 1 ) \leq n$ , gives

$$
1 \leq \Psi _ { Q } ( V ) , \qquad \log \Psi _ { Q } ( V ) \leq C _ { \ell } J ( t + 1 ) \Bigl [ \log \frac { e n } { J ( t + 1 ) } + \log ( L + 1 ) \Bigr ] .\tag{113}
$$

Throughout the pass, $L , \lambda .$ , and the coloring law stay fixed; the coloring law is restricted to the unprocessed inputs as inputs are deleted. Deleting an input does not increase the potential, since every profile of the smaller class is a profile of the larger one with the same support.

For a set A attained at an unprocessed input i by a member of $V$ , the exact child $V _ { A }$ is the set of members of $V$ whose set at i is exactly A, with i deleted. The next inequality is the core of the proof: conditioning on disjoint sets splits the potential.

Claim 97. $I f A _ { 1 } , \ldots , A _ { k }$ are pairwise disjoint sets attained at an unprocessed input i $b y$ members of a state $V$ , then

$$
\sum _ { j = 1 } ^ { k } \Psi _ { Q } ( V _ { A _ { j } } ) \leq \Psi _ { Q } ( V ) .\tag{114}
$$

Proof. Fix the colors at all unprocessed inputs other than $i ;$ the encoded children $( V _ { A _ { j } } ) _ { \lambda }$ depend only on them. Let $\pi$ be a profile of exactly $k _ { \pi } \geq 1$ of these children, so that π contributes $k _ { \pi } \lambda ^ { | \mathrm { s u p p } ( \pi ) | }$ to the left-hand side. We show that, in expectation over the colors at $i ,$ the profile π and its extensions to i contribute at least as much to $\Psi _ { Q } ( V )$

First, $\pi$ is a profile of $V _ { z }$ whatever the colors at $i ,$ since its realizers come from members of $V$ . Next, consider two children that realize $\pi ,$ corresponding to $A _ { j }$ and $A _ { j ^ { \prime } }$ . If both sets are retained at $i ,$ their encodings are nonempty and disjoint, and adding this pair at i extends $\pi$ to a profile of $V _ { z }$ on $\operatorname { s u p p } ( \pi ) \cup \{ i \}$ : either choice at $i ,$ together with any choice in $\pi ,$ is realized by the corresponding child. Diferent pairs give diferent extensions.

It remains to bound the probability that both sets are retained. Every $A _ { j }$ meets $Q ( i )$ and has at most ℓ labels, so it has at most $\ell - 1$ labels outside $Q ( i )$ , each of which receives a color in $A _ { j } \cap Q ( i )$ with probability at least $1 / L ;$ ; hence $A _ { j }$ is retained with probability at least $L ^ { - ( \ell - 1 ) }$ . The retention events of $A _ { j }$ and $A _ { j ^ { \prime } }$ depend on the colors of disjoint sets of labels, so they are independent, and both sets are retained with probability at least $L ^ { - 2 ( \ell - 1 ) } = 1 / \lambda$ Each extension therefore has expected weight at least $\bar { \lambda } ^ { | \mathrm { s u p p } ( \pi ) | \bar { + 1 } } / \lambda = \lambda ^ { | \mathrm { s u p p } ( \pi ) | }$ , and the contribution of $\pi$ and its extensions is at least

$$
\Bigl ( 1 + \binom { k _ { \pi } } { 2 } \Bigr ) \lambda ^ { | \operatorname { s u p p } ( \pi ) | } \geq k _ { \pi } \lambda ^ { | \operatorname { s u p p } ( \pi ) | } .
$$

Deleting i from an extension of $\pi$ gives back $\pi ,$ so diferent profiles have diferent extensions. Summing over $\pi$ and averaging over the remaining colors proves (114). □

Conservative updates. Inequality (114) allows a short list to hit every set whose exact child keeps a large fraction of the potential. We now describe one pass for a target pattern $v$ (one of the $v ^ { a } )$ , a list table $Q$ of width at most $L ,$ and an integer $m \geq 1$ . Let $V$ be the reset class of $Q ,$ and process the inputs in a fixed order. At input $i ,$ call a set $A$ attained at i by a member of V heavy if $\Psi _ { Q } ( V _ { A } ) > \Psi _ { Q } ( V ) / ( m + 1 )$ . By (114), a family of pairwise disjoint heavy sets has at most m members. Let $Q ^ { \prime } ( i )$ be the union of a maximal such family, chosen by the fixed rule as a function of the state, or a fixed singleton if no set is heavy. Then $| Q ^ { \prime } ( i ) | \leq \ell m$ , and by maximality $Q ^ { \prime } ( i )$ meets every heavy set. The list $Q ^ { \prime } ( i )$ is computed and stored before the target set $v _ { i }$ is examined.

Now $v _ { i }$ is examined. If $Q ( i )$ meets $v _ { i }$ and $Q ^ { \prime } ( i )$ misses it, record the pair $( i , v _ { i } )$ and replace $V$ by $V _ { v _ { i } }$ . Otherwise record nothing and delete input i from all members of $V ,$ without conditioning. In both cases the restriction of v to its hit domain, with the processed inputs deleted, stays in $V ;$ so $V$ is never empty, and $\Psi _ { Q } ( V ) \geq 1$ throughout the pass. In particular, at a recorded input the set $v _ { i }$ is attained. It is not heavy, since $Q ^ { \prime } ( i )$ misses it, so conditioning on it divides the potential by at least $m + 1$ , while every other step is a deletion and does not increase the potential. By (113), the number of inputs recorded in one pass is at most

$$
\frac { C _ { \ell } J ( t + 1 ) } { \log ( m + 1 ) } \Bigl [ \log \frac { e n } { J ( t + 1 ) } + \log ( L + 1 ) \Bigr ] .\tag{115}
$$

A new miss, that is, an input at which $Q$ hits $v _ { i }$ and $Q ^ { \prime }$ misses it, is always recorded. Hence miss $( Q ^ { \prime } , v )$ exceeds miss $( Q , v )$ by at most the number of inputs recorded in the pass.

Reduction to constant width. We now choose the parameters so that the width decreases quickly and the total number of records remains bounded. Start from the width bound $L = L _ { 0 }$ of (111). While $L > L _ { * }$ , run a pass with

$$
m = \Big \lceil \frac { \sqrt { L } } { \ell } \Big \rceil , \qquad L _ { \mathrm { n e x t } } = 2 \sqrt { L } ,
$$

and replace L by $L _ { \mathrm { n e x t } }$ . Since $L > L _ { * } \ge \ell ^ { 4 }$ , we have $\ell \leq L ^ { 1 / 4 }$ , hence $\ell m \leq \sqrt L + \ell \leq 2 \sqrt L .$ , so the new list has width at most $L _ { \mathrm { n e x t } }$ , and $\log ( m { + } 1 ) \geq \log ( \sqrt { L } / \ell ) \geq \frac { 1 } { 4 }$ log L. The width bounds $L _ { 0 } , L _ { 1 } , . . .$ . are the same for every index, and they satisfy log $L _ { j + 1 } \stackrel { - } { - } \log 4 = \textstyle { \frac { 1 } { 2 } } ( \log L _ { j } - \log 4 )$ A pass runs only while $L _ { j } > L _ { * } \ge 1 6$ , that is, while log $L _ { j } - \log 4 > \log$ 4, so over the passes the numbers log $L _ { j } - \log 4$ halve from one pass to the next, start below log $L _ { 0 } \leq C ( t + 1 )$ by (111), and stay above log 4. Hence, with $j$ running over the passes,

$$
\# \{ \mathrm { p a s s e s } \} \leq C \log ( 2 t + 2 ) , \qquad \sum _ { j } \frac { 1 } { \log L _ { j } } \leq \sum _ { j } \frac { 1 } { \log L _ { j } - \log 4 } \leq \frac { 2 } { \log 4 } ,
$$

since the terms of the second sum double from one pass to the next and the last is less than $1 / \log 4 .$

If, for some index, the scan before some pass finds a profile with support of size at least $J ( t + 1 )$ , we return the witness extracted from it. Otherwise, by (115), log $\left( L + 1 \right) \leq 2 \log L$ $\log ( m + 1 ) \geq { \frac { 1 } { 4 } } \log L$ , and $J ( t + 1 ) \geq t + 1$ , a pass with width bound L records at most

$$
C ( t + 1 ) \Big [ \frac { \log ( e n / ( t + 1 ) ) } { \log L } + 1 \Big ]
$$

inputs. Summing over the passes with the two bounds above, the passes record at most

$$
C ( t + 1 ) \Big [ \log \frac { e n } { t + 1 } + \log ( 2 t + 2 ) \Big ] = C ( t + 1 ) \log ( 2 e n ) \le \Lambda
$$

inputs per index, for a suitable $K = K _ { \ell , \eta }$ . Let $T _ { a }$ consist of $I _ { 0 }$ and the inputs recorded in the passes for the index $a ,$ , and let $Q _ { a }$ be the final list table, of width at most $L _ { * }$ . Then $| T _ { a } | \leq u + \Lambda$ . Every new miss is recorded, so miss $( Q _ { a } , v ^ { a } ) \leq \operatorname* { m i s s } ( Q _ { a , 0 } , v ^ { a } ) + \Lambda$ , and (112) gives the second bound in (109).

Stability. The history $\tau _ { a }$ consists of the sets $v _ { i } ^ { a } , i \in I _ { 0 }$ , and of the pairs recorded in all passes, each with its pass index. It determines every list and every state of the run for the index $a .$ Indeed, the sets on $I _ { 0 }$ determine $Q _ { a , 0 }$ , and the width bounds are fixed in advance. Within a pass, the state before an input i determines $Q ^ { \prime } ( i )$ ; then the history tells whether i is recorded, and if so, the set to condition on; otherwise i is deleted. In particular, equal histories give equal final lists.

Now let $v _ { - i } ^ { a } = v _ { - i } ^ { b }$ and $i \notin T _ { a } \cup T _ { b }$ . Since $I _ { 0 } \subseteq T _ { a }$ , the set $I _ { 0 }$ avoids $i ,$ so the two runs see the same sets on $I _ { 0 }$ and start from the same list. Suppose that a pass starts from the same list in both runs. Then the two reset classes coincide, and we show by induction over the inputs that the two states coincide. Equal states give the same list $Q ^ { \prime } ( j )$ at the next input $j$ $\mathrm { I f } ~ j \neq i$ , then $v _ { j } ^ { a } = v _ { j } ^ { b }$ , so the two runs make the same decision, record the same pair if any, and reach the same state. If $j = i$ , neither run records $i ,$ so both delete it. Hence the pass records the same pairs and produces the same list in both runs. Induction over the passes gives $\tau _ { a } = \tau _ { b }$ , which proves (110). □

## B.3 Proof of the Comparison Theorem

We now prove Theorem 92 and then deduce Theorem 60. The main point is the choice of the target size t in Lemma 96: records and misses cost $\eta n + O ( t \log n )$ inputs per pattern, while the packed witness has $\rho n / ( \ell + 1 )$ blocks per pattern on average, so with t a small multiple of $n /$ log n most blocks survive, and Lemma 95 gives a witness on order n inputs, more than the $t + 1$ we need.

Proof of Theorem 92. Fix a $\rho -$ -packed witness of $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ on $n \geq 1$ inputs, with list $\boldsymbol { v } ^ { 1 } , \ldots , \boldsymbol { v } ^ { M }$ and collections $\boldsymbol { B } _ { 1 } , \ldots , \boldsymbol { B } _ { n }$ of conflict blocks, and fix the representing tuples of its patterns. $\mathrm { A s }$ in Section B.1, we localize to the finite class $\mathcal { H } _ { 0 }$ of the hypotheses in these tuples and identify the inputs with $[ n ]$ ; thus $\boldsymbol { v } ^ { 1 } , \ldots , \boldsymbol { v } ^ { M }$ are patterns of $\mathbb { Z } _ { \ell + 1 } ( \mathcal { H } _ { 0 } )$ defined on all of $[ n ]$

Constants. Put $\alpha = \rho / ( \ell + 1 )$ and $\eta = \alpha / 4$ , let $J , L ,$ <sub>∗</sub> and $K$ be the constants of Lemma 96 for ℓ and $\eta ,$ , and write $\beta = \beta _ { \ell , \ell + 1 , L }$ for the constant of Lemma 95. All of them depend only on ℓ and $\rho .$ Choose $c > 0$ with

$$
c \leq \operatorname* { m i n } \Bigl \{ \frac { \eta } { 4 } , \ \frac { 1 } { 4 J } , \ \frac { \alpha } { 4 ( 2 / \eta + 4 K ) } , \ \frac { \beta \alpha } { 4 } \Bigr \} ,
$$

and set

$$
t = \left\lfloor { \frac { c n } { \log ( 2 e n ) } } \right\rfloor .
$$

We construct a disjoint-neighbor witness on more than $c n / \log ( 2 e n )$ inputs whose patterns are intersections of at most $\ell + 1$ hypotheses of $\mathcal { H } _ { 0 }$ . This proves the theorem with $c _ { \ell , \rho } = c / 2$ since $\log ( 2 e n ) \leq 2 \log ( n + 2 )$ for $n \geq 1$

Target size zero. If $t = 0$ , then $c n / \log ( 2 e n ) < 1$ , and a witness on one input sufices. The blocks of direction 1 cover at least $\rho M > 0$ indices, so there is a block $B \in B _ { 1 }$ . Its sets $v _ { 1 } ^ { a }$ are intersections of up to $\ell + 1$ hypotheses, while the witness must lie in $\mathcal { T } _ { \ell } ( \mathcal { H } _ { 0 } )$ , so we pass to single hypotheses. The sets $g ( 1 )$ , over the hypotheses $g$ in the tuples of the patterns $v ^ { a }$ $a \in B .$ are nonempty sets of at most ℓ labels, and they have an empty intersection because $\cap _ { a \in B } v _ { 1 } ^ { a } = \emptyset$ . By Lemma 8 they contain a minimal empty-intersection family $A _ { 1 } , \ldots , A _ { r }$ with $2 \leq r \leq \ell + 1$ . The sets $\cap _ { j < r } A _ { j }$ and $A _ { r }$ are then disjoint and nonempty, and they are the values at input 1 of intersections of at most ℓ hypotheses of $\mathcal { H } _ { \mathrm { 0 } } ;$ as one-coordinate patterns, they form a disjoint-neighbor witness in $\mathcal { T } _ { \ell } ( \mathcal { H } _ { 0 } )$ on one input.

Applying the list reduction. Suppose now that $t \geq 1$ . Then $t + 1 \leq 2 t \leq 2 c n / \log ( 2 e n )$ $\operatorname { A s } t \geq 1$ forces $n \geq 2$ , the bound $c \leq \eta / 4$ gives $u = \lceil t / \eta \rceil \leq n / 4 + 1 < n$ , and the bound $c \leq 1 / ( 4 J )$ gives $J ( t + 1 ) \le n / 2$ . So Lemma 96 applies to the patterns $\boldsymbol { v } ^ { 1 } , \ldots , \boldsymbol { v } ^ { M }$ with target size t. If it constructs a disjoint-neighbor witness in $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } _ { 0 } )$ on at least $t + 1 > c n / \log ( 2 e n )$ inputs, we return this witness. Otherwise it assigns to every index $a \in \ [ M ]$ a history $\tau _ { a }$ , a record set $T _ { a } .$ , and a list $Q _ { a }$ of width at most $L _ { * }$ that satisfy (109) and (110), with $\Lambda = K ( t + 1 ) \log ( 2 e n )$ . It remains to find, in this case, a family of blocks to which Lemma 95 applies.

Discarding blocks hit by records or misses. The blocks of direction i have at most $\ell + 1$ members and cover at least $\rho M$ indices, so there are at least $\alpha M$ of them, and at least $\alpha M n$ blocks in all. Discard a block $B \in B _ { i }$ if some $a \in B$ records i, that is, $i \in T _ { a }$ , or has a miss at $i ,$ , that is, $Q _ { a } ( i ) \cap v _ { i } ^ { a } = \emptyset$ . Since the blocks of direction i are disjoint, an index a with a record or a miss at i lies in at most one block of direction i. By (109), the number of discarded blocks is therefore at most

$$
\sum _ { a = 1 } ^ { M } \bigl ( | T _ { a } | + \operatorname { m i s s } ( Q _ { a } , v ^ { a } ) \bigr ) \leq M ( u + \eta n + 2 \Lambda ) .
$$

Since $u \leq t / \eta + 1 \leq 2 t / \eta \leq 2 c n / \eta$ and $2 \Lambda = 2 K ( t + 1 ) \log ( 2 e n ) \leq 4 K c n$ , the choice of c gives

$$
u + \eta n + 2 \Lambda \le \left[ \eta + c \Bigl ( \frac { 2 } { \eta } + 4 K \Bigr ) \right] n \le \frac { \alpha } { 2 } n .
$$

Thus at least $\alpha M n / 2$ blocks remain. For every remaining block $B \in B _ { i }$ and every $a \in B$ , we have $i \not \in T _ { a }$ and $Q _ { a } ( i ) \cap v _ { i } ^ { a } \neq \emptyset$

Grouping the patterns by their histories. Partition the indices $[ M ]$ into groups of indices with the same history. Let $B \in B _ { i }$ be a remaining block and $a , b \in B$ . Then $v _ { - i } ^ { a } = v _ { - i } ^ { b }$ and $i \notin { T _ { a } \cup T _ { b } } , \mathrm { s o } \tau _ { a } = \tau _ { b }$ by (110). Hence every remaining block lies in a single group. Since the history determines the list, all indices of a group G have a common list $Q _ { G }$ of width at most $L _ { * }$ . Let $M _ { G } = | G |$ , and let $N _ { G }$ be the number of remaining blocks contained in G. Since $\textstyle \sum _ { G } M _ { G } = M$ and $\textstyle \sum _ { G } N _ { G } \geq \alpha M n / 2$ , weighted averaging gives a group G satisfying

$$
\frac { N _ { G } } { M _ { G } } \geq \frac { \alpha } { 2 } n > 0 .
$$

Applying the bounded-list lemma. Fix this group $G ,$ and for each $i \in [ n ]$ let $B _ { i } ^ { G }$ be the set of remaining blocks of direction i contained in $G .$ . Consider the patterns $v ^ { a } , a \in G$ , which are patterns of $\mathbb { Z } _ { \ell + 1 } ( \mathcal { H } _ { 0 } )$ defined on all of [n], with their representing tuples, and the list $Q _ { G }$ Every $B \in B _ { i } ^ { G }$ is a conflict block of the packed witness, so $2 \leq | \boldsymbol { B } | \leq \ell + 1$ , its patterns agree outside i, and their sets at i have an empty intersection; moreover, $Q _ { G } ( i ) = Q _ { a } ( i )$ hits $v _ { i } ^ { a }$ for every $a \in B$ . The blocks of $B _ { i } ^ { G }$ are disjoint, and $\begin{array} { r } { \sum _ { i } | B _ { i } ^ { G } | = N _ { G } > 0 } \end{array}$ . Thus Lemma $9 5$ with $R = \ell + 1$ and $L = L _ { * }$ , applied to the $M _ { G }$ patterns of $G ,$ constructs a disjoint-neighbor witness in $\mathcal { T } _ { \ell } ( \mathcal { H } _ { 0 } )$ on at least

$$
\beta \frac { N _ { G } } { M _ { G } } \geq \frac { \beta \alpha } { 2 } n \geq 2 c n \geq t + 1
$$

inputs, where the second inequality uses $c \leq \beta \alpha / 4$ and the last uses $t + 1 \leq 2 c n / \log ( 2 e n )$ This finishes the proof. □

We now deduce the comparison theorem.

Proof of Theorem 60. We first take $d = d _ { \mathrm { p a i r } } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) ) < \infty$ . Let a ρ-packed witness of $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ on $n \geq 1$ distinct inputs be given. Theorem 92, applied with any choice of representing tuples, gives a disjoint-neighbor witness of $\mathscr { T } _ { \ell + 1 } ( \mathcal { H } )$ on a subsequence $U$ of these inputs with $| U | \geq c _ { \ell , \rho } n / \log ( n + 2 ) > 0$ . Hence

$$
d \geq | U | \geq c _ { \ell , \rho } { \frac { n } { \log ( n + 2 ) } } .
$$

If $d = 0 .$ , this rules out every $\rho -$ -packed witness of positive size, so $d _ { \mathrm { p a c k } , \rho } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) ) = 0$ and (82) holds. Let $d \geq 1$ . Since log $\left( n + 2 \right) \leq \log ( 2 e n )$ , we have

$$
n \leq C d \log ( 2 e n ) , \qquad C = \operatorname* { m a x } \{ 1 , 1 / c _ { \ell , \rho } \} .
$$

Put $x = n / d ,$ , so that $\log ( 2 e n ) = \log ( 2 e d ) + \log x$ . The tangent-line inequality log $x \ \leq$ $x / ( 2 C ) + \log ( 2 C ) - 1$ gives

$$
x \leq C \big [ \log ( 2 e d ) + \log x \big ] \leq \frac { x } { 2 } + C \log ( 4 C d ) ,
$$

so $x \leq 2 C \log ( 4 C d )$ . Since $\log ( d + 2 ) \geq \log 3 > 1$ , we have log $\ ( 4 C d ) \leq \log ( 4 C ) + \log ( d + 2 ) \leq$

$( 1 + \log ( 4 C ) ) \log ( d + 2 )$ , and therefore

$$
\begin{array} { r } { n \leq 2 C ( 1 + \log ( 4 C ) ) d \log ( d + 2 ) . } \end{array}
$$

The constant depends only on ℓ and $\rho ,$ and n was the size of an arbitrary ρ-packed witness.   
Taking the supremum over all witnesses proves (82) for the disjoint-neighbor dimension.

Now let $d = d _ { \mathrm { D S } } ^ { \cap } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) ) < \infty$ . By Lemma 9, $d _ { \mathrm { p a i r } } ( \mathbb { Z } _ { \ell + 1 } ( \mathcal { H } ) ) \leq d .$ so the disjoint-neighbor case applies, and the bound in terms of d follows because $z \mapsto z \log ( z + 2 )$ is increasing for $z \geq 0$ □

## B.4 Proof of the Common-Label Lemma

In this subsection we prove Lemma 93. For a hypothesis $h \in \mathcal { H } _ { 0 }$ and an input i of the pool, we write $h ( i )$ for the correct-label set of h at i. We first prove two auxiliary results: a coverage lemma for partial classes over [r], and a noise inequality on $[ r ] ^ { n }$ . We then prove the common-label lemma by induction on the sizes of prescribed sets, called cores.

Partial finite-alphabet coverage. Fix $r \geq 2$ . A partial class on $[ n ]$ over the alphabet $[ r ]$ is a set ${ \mathcal { F } } \subseteq ( [ r ] \cup \{ \star \} ) ^ { [ n ] }$ , where $f _ { i } = \star$ means that f is undefined at $i ,$ and supp $( f ) = \{ i : f _ { i } \neq \star \}$ A set $U \subseteq [ n ]$ is shattered by $\mathcal { F }$ if every vector $b \in [ r ] ^ { U }$ is realized on $U$ , that is, some $f \in { \mathcal { F } }$ with $U \subseteq \operatorname { s u p p } ( f )$ has $f _ { i } = b _ { i }$ for all $i \in U$ . The empty set is shattered exactly when $\mathcal { F }$ is nonempty. For $r = 2$ , the largest size of a shattered set is the VC dimension of a partial concept class [AHHM21].

Lemma 98 (Partial coverage). Let $n \geq 1 , r \geq 2$ , and $0 < \kappa \leq 1 / r$ , and let ${ \mathcal { F } } \subseteq ( [ r ] \cup \{ \star \} ) ^ { [ n ] }$ be nonempty. Let D be the largest size $o f a$ set shattered by $\mathcal { F }$ . Suppose that $f o r$ every $b \in [ r ] ^ { n }$ there is $f ^ { b } \in { \mathcal { F } }$ with support $E _ { b }$ such that $f _ { i } ^ { b } = b _ { i }$ for all $i \in E _ { b }$ , and that

$$
{ \frac { 1 } { r ^ { n } } } \sum _ { b \in [ r ] ^ { n } } | E _ { b } | \geq { \Big ( } 1 - { \frac { 1 } { r } } + \kappa { \Big ) } n .
$$

Then $D \geq 1$ and

$$
n \leq C _ { r , \kappa } D , \qquad C _ { r , \kappa } = 4 a _ { r } ^ { 2 } \kappa ^ { - 4 } , \qquad a _ { r } = \frac { 1 } { \log ( r / ( r - 1 ) ) } .
$$

A shattered set of size at least $n / C _ { r , \kappa } ,$ together with a member of $\mathcal { F }$ realizing each vector on $i t ,$ can be found by finite enumeration.

For fixed lists $g _ { 1 } , \ldots , g _ { n } \subseteq [ r ]$ of size $r - 1$ , a uniform vector $b \in [ r ] ^ { n }$ has $b _ { i } \in g _ { i }$ at a $1 - 1 / r$ fraction of the coordinates on average; the lemma says that a class agreeing with random vectors on noticeably more coordinates shatters a set of linear size. The main idea is to compare two bounds on a family $\mathcal { G }$ of sequences of lists of $r - 1$ symbols such that every member of $\mathcal { F }$ is covered by some sequence $g \in { \mathcal { G } }$ , that is, $f _ { i } \in g _ { i }$ for all $i \in \operatorname { s u p p } ( f )$ On the one hand, for a uniformly random vector $b ,$ a concentration bound shows that the best sequence in $\mathcal { G }$ satisfies $b _ { i } \in g _ { i }$ at no more than $( 1 - 1 / r ) n + \sqrt { n \log | \mathcal { G } | / 2 }$ coordinates on average, so the hypothesis forces log $| \mathcal { G } | \geq 2 \kappa ^ { 2 } n$ . On the other hand, we construct G by reading each member of $\mathcal { F }$ coordinate by coordinate. At each step we propose a list excluding the symbol whose conditional class has the fewest shattered sets. If the member takes that symbol, we record the coordinate and change the list to cover it. Each record divides the number of shattered sets by at least $r / ( r - 1 )$ . Since $\mathcal { F }$ shatters at most $( e n / D ) ^ { D }$ sets, there are $O ( D \log ( n / D ) )$ records, and the records determine the sequence of lists; this gives log $| \mathcal { G } | = O ( D \log ^ { 2 } ( n / D ) )$ , and the two bounds together give $n = O ( D )$

Proof. The potential. For a partial class $V$ on a set of coordinates, let $\Phi ( V )$ be the number of sets shattered by $V _ { : }$ , the empty set included; thus $\Phi ( V ) \geq 1$ whenever $V \neq \emptyset$ . For a coordinate i and a symbol $c \in [ r ]$ , let $V _ { c }$ consist of the members of $V$ with value c at i, with coordinate i deleted, and let $V _ { - i }$ consist of all members of V with coordinate i deleted. Then

$$
\sum _ { c = 1 } ^ { r } \Phi ( V _ { c } ) \leq ( r - 1 ) \Phi ( V ) , \qquad \Phi ( V _ { - i } ) \leq \Phi ( V ) .\tag{116}
$$

The second inequality holds because every set shattered by $V _ { - i }$ is shattered by V . For the first, let U be a set of coordinates other than i that is shattered by exactly $\nu \geq 1$ of the classes $V _ { c }$ . Then $U$ is shattered by $V .$ , and if $\nu = r$ , so is $U \cup \{ i \}$ . Thus $U$ contributes ν to the left-hand side, and it accounts for one set counted by $\Phi ( V )$ if $\nu < r$ and for two if $\nu = r$ Since $\nu \leq r - 1$ in the first case and $r \leq 2 ( r - 1 )$ in the second, the contribution of $U$ is at most $r - 1$ times the number of sets it accounts for. Distinct sets $U$ account for distinct sets counted by $\Phi ( V )$ , which proves the claim.

Lists from few records. Fix an order on the coordinates and a rule for breaking ties. For each $f \in { \mathcal { F } }$ we read the coordinates in order and build a sequence $g = ( g _ { 1 } , \dotsc , g _ { n } )$ of lists of $r - 1$ symbols that covers $f .$ . The state is a partial class V on the unread coordinates, initially $\mathcal { F }$ . At coordinate i, let c be a symbol minimizing $\Phi ( V _ { c } )$ . If $f _ { i } \neq c ,$ put $g _ { i } = [ r ] \setminus \{ c \}$ and replace V by $V _ { - i }$ . If $f _ { i } = c ,$ record $i ,$ put $g _ { i } = [ r ] \setminus \{ c ^ { \prime } \}$ for the least $c ^ { \prime } \neq c ,$ and replace $V$ by $V _ { c }$ . In both cases $f _ { i } \in g _ { i }$ if f is defined at i, and the restriction of $f$ to the unread coordinates stays in $V$ , so that $\Phi ( V ) \geq 1$ throughout. By (116), the minimizing symbol satisfies $\Phi ( V _ { c } ) \leq ( 1 - 1 / r ) \Phi ( V )$ , so every record multiplies $\Phi ( V )$ by at most $1 - 1 / r$ , while the other steps do not increase it. Hence every f has at most $\mu : = \lfloor a _ { r } \log \Phi ( { \mathcal { F } } ) \rfloor$ records.

The recorded coordinates determine the whole run: given them, we can replay the procedure, conditioning on the current minimizing symbol at a recorded coordinate and deleting the coordinate otherwise. Thus the sequences of lists form a family $\mathcal { G }$ that covers every member of $\mathcal { F }$ . Since every shattered set has at most D elements,

$$
| \mathcal { G } | \leq \sum _ { \nu = 0 } ^ { \operatorname* { m i n } \{ \mu , n \} } { \binom { n } { \nu } } , \qquad \Phi ( \mathcal { F } ) \leq \sum _ { \nu = 0 } ^ { D } { \binom { n } { \nu } } .\tag{117}
$$

The lower bound on $| \mathcal G |$ . Let b be uniform on $[ r ] ^ { n }$ . For a fixed $g \in { \mathcal { G } }$ , the number of i with $\mathbf { b } _ { i } \in g _ { i }$ is a sum of n independent Bernoulli variables with mean $1 - 1 / r$ . By Hoefding’s

lemma and the standard bound on the expected maximum of finitely many subgaussian variables,

$$
\mathbb { E } \operatorname* { m a x } _ { g \in \mathcal { G } } \left| \left\{ i : \mathbf { b } _ { i } \in g _ { i } \right\} \right| \leq \Big ( 1 - \frac { 1 } { r } \Big ) n + \sqrt { \frac { n \log | \mathcal { G } | } { 2 } } .
$$

On the other hand, $f ^ { \mathbf { b } }$ is covered by some $g \in { \mathcal { G } }$ , and ${ \bf b } _ { i } = f _ { i } ^ { { \bf b } } \in g _ { i }$ for every $i \in E _ { \mathbf { b } }$ ; thus $| E _ { \mathbf { b } } |$ is at most the maximum on the left. By the hypothesis, $\kappa n \leq \sqrt { n \log | \mathcal { G } | / 2 }$ , that is,

$$
2 \kappa ^ { 2 } n \leq \log | \mathcal { G } | .\tag{118}
$$

In particular $| \mathcal { G } | > 1$ , so $\mu \geq 1$ by (117). Hence $\Phi ( \mathcal { F } ) > 1$ , and $\mathcal { F }$ shatters a nonempty set, that is, $D \geq 1$

The upper bound on $| \mathcal G |$ . Put $\xi = n / D \ge 1 . ~ \mathrm { I f } ~ \xi \le 1 6 a _ { r } ^ { 2 }$ , the claim holds, because $\kappa \leq 1 / 2$ gives $C _ { r , \kappa } \geq 6 4 a _ { r } ^ { 2 }$ . Otherwise, (117) and the bound $\begin{array} { r } { \sum _ { \nu \leq D } \binom { n } { \nu } \leq ( e n / D ) ^ { D } } \end{array}$ give

$$
\mu \leq \mu _ { 0 } : = a _ { r } D \log ( e \xi ) < \frac { n } { 2 } ,
$$

where the last inequality follows from $\log ( e \xi ) \leq 2 { \sqrt { \xi } }$ and $\xi > 1 6 a _ { r } ^ { 2 }$ . The function $\nu \mapsto$ $\nu \log ( e n / \nu )$ is increasing on $( 0 , n ]$ , and $a _ { r } \log ( e \xi ) \geq 1$ ; hence, by (117),

$$
\log | \mathcal { G } | \leq \mu \log \frac { e n } { \mu } \leq \mu _ { 0 } \log \frac { e n } { \mu _ { 0 } } = a _ { r } D \log ( e \xi ) \log \frac { e \xi } { a _ { r } \log ( e \xi ) } \leq a _ { r } D \log ^ { 2 } ( e \xi ) .
$$

Together with (118), this gives $\xi \le a _ { r } \log ^ { 2 } ( e \xi ) / ( 2 \kappa ^ { 2 } )$ . Since $( 1 + \tau ) e ^ { - \tau / 4 } \leq 2$ for all $\tau \geq 0$ (its maximum, attained at $\tau = 3$ , is $4 e ^ { - 3 / 4 } )$ , taking $\tau = \log \xi$ gives $\log ^ { 2 } ( e \xi ) \leq 4 \sqrt { \xi }$ . Therefore $\sqrt { \xi } \leq 2 a _ { r } \kappa ^ { - 2 }$ , that is, $n \leq 4 a _ { r } ^ { 2 } \kappa ^ { - 4 } D$

Finally, $\mathcal { F }$ is finite, so a largest shattered set and a member realizing each vector on it can be found by enumeration. □

A finite-alphabet noise inequality. For the light inputs in the proof below, we need a form of the hypercontractive inequality [O’D14] on the product space $[ r ] ^ { n }$ . We include the short proof.

Lemma 99 (Noise inequality). Fix $r \geq 2$ and $0 < \theta < 1$ . There is $p \in ( 1 , 2 )$ , depending only on r and $\theta ,$ such that the following holds for every $n \geq 0$ . Let u, $\mathbf { v } \in [ r ] ^ { n }$ be random vectors whose coordinate pairs $\left( \mathbf { u } _ { i } , \mathbf { v } _ { i } \right)$ are independent, and each pair is a common uniform symbol with probability θ and two independent uniform symbols otherwise. Then for all $f , g : [ r ] ^ { n } \to \{ 0 , 1 \}$

$$
\begin{array} { r } { \mathbb { E } \big [ f ( \mathbf { u } ) g ( \mathbf { v } ) \big ] \leq \big ( \mathbb { E } [ f ( \mathbf { u } ) ] \mathbb { E } [ g ( \mathbf { v } ) ] \big ) ^ { 1 / p } . } \end{array}
$$

Proof. We use the uniform probability measure on $[ r ]$ and on $[ r ] ^ { n }$ , and write $\| f \| _ { p } = ( \mathbb { E } | f | ^ { p } ) ^ { 1 / p }$ The proof has two steps: we first prove a one-coordinate bound for the noise operator, and then extend it to $[ r ] ^ { n }$ by the standard tensorization argument.

One coordinate. Let $f \geq 0$ be a function on [r], and write $f = a + b$ with $a = \mathbb { E } f$ and

$\mathbb { E } b = 0$ . We claim that

$$
\| f \| _ { p } ^ { 2 } \geq a ^ { 2 } + ( p - 1 ) \| b \| _ { p } ^ { 2 } \geq a ^ { 2 } + ( p - 1 ) r ^ { 1 - 2 / p } \| b \| _ { 2 } ^ { 2 } , \qquad 1 < p \leq 2 .\tag{119}
$$

If $a = 0$ , then $f = 0$ and there is nothing to prove. Otherwise, for $0 \leq \tau < 1$ put $z _ { \tau } = a + \tau b .$ which equals $( 1 - \tau ) a + \tau f$ and is therefore positive, and let $\phi ( \tau ) = \mathbb { E } z _ { \tau } ^ { p }$ and $\psi ( \tau ) = \phi ( \tau ) ^ { 2 / p }$ Diferentiating twice, and dropping a term that is nonnegative because $p \leq 2$ , we obtain

$$
\begin{array} { r } { \psi ^ { \prime \prime } ( \tau ) \geq 2 ( p - 1 ) \phi ( \tau ) ^ { ( 2 - p ) / p } \mathbb { E } \left[ b ^ { 2 } z _ { \tau } ^ { p - 2 } \right] . } \end{array}
$$

Hölder’s inequality with exponents $2 / p$ and $2 / ( 2 - p )$ , applied to $| b | ^ { p } = ( b ^ { 2 } z _ { \tau } ^ { p - 2 } ) ^ { p / 2 } z _ { \tau } ^ { p ( 2 - p ) / 2 }$ shows that $\begin{array} { r } { \phi ( \tau ) ^ { ( 2 - p ) / p } \mathbb { E } \big [ b ^ { 2 } z _ { \tau } ^ { p - 2 } \big ] \geq \| b \| _ { p } ^ { 2 } . } \end{array}$ . Since $\psi ( 0 ) = a ^ { 2 }$ and $\psi ^ { \prime } ( 0 ) = 2 a ^ { 2 - p } \cdot a ^ { p - 1 } \mathbb { E } b = 0$ integrating twice gives $\psi ( \tau ) \geq a ^ { 2 } + ( p - 1 ) \tau ^ { 2 } \| b \| _ { p } ^ { 2 } ,$ and letting $\tau  1$ proves the first inequality in (119). The second follows from $\| b \| _ { p } \geq r ^ { 1 / 2 - \operatorname { i } / p } \| b \| _ { 2 }$ , which is the comparison between the $\ell _ { p }$ and $\ell _ { 2 }$ norms of the vector $( b ( 1 ) , \ldots , b ( r ) )$ .

For $0 \leq \lambda \leq 1$ , let $T _ { \lambda } f = \lambda f + ( 1 - \lambda ) \mathbb { E } f$ . Choose $p < 2$ so close to 2 that $( p - 1 ) r ^ { 1 - 2 / p } \geq \theta ;$ this is possible because the left-hand side tends to 1 as $p  2$ . Since $T _ { \sqrt { \theta } } f = a + { \sqrt { \theta } } b$ and $\mathbb { E } b = 0 , ( 1 1 9 )$ gives

$$
\| T _ { \sqrt { \theta } } f \| _ { 2 } ^ { 2 } = a ^ { 2 } + \theta \| b \| _ { 2 } ^ { 2 } \leq \| f \| _ { p } ^ { 2 } .
$$

Tensorization. On $[ r ] ^ { n }$ , let $T _ { \lambda }$ act as the one-coordinate operator in every coordinate; it maps nonnegative functions to nonnegative functions. We show $\| T _ { \sqrt { \theta } } f \| _ { 2 } \ \leq \ \| f \| _ { p }$ for $f \geq 0$ by induction on $n .$ . Split the coordinates into a block x of $n - 1$ coordinates and one further coordinate $y ,$ and write $T _ { x }$ and $T _ { y }$ for the operators $T _ { \sqrt { \theta } }$ acting on these blocks, so that $T _ { \sqrt { \theta } } = T _ { x } T _ { y }$ . The induction hypothesis, applied for each fixed y, Minkowski’s integral inequality (which applies because $p \leq 2 )$ , and the one-coordinate bound, applied for each fixed x, give

$$
\| T _ { x } T _ { y } f \| _ { L _ { y } ^ { 2 } ( L _ { x } ^ { 2 } ) } \leq \| T _ { y } f \| _ { L _ { y } ^ { 2 } ( L _ { x } ^ { p } ) } \leq \| T _ { y } f \| _ { L _ { x } ^ { p } ( L _ { y } ^ { 2 } ) } \leq \| f \| _ { L _ { x } ^ { p } ( L _ { y } ^ { p } ) } = \| f \| _ { p } .
$$

The correlated pair. Given u, each $\mathbf { v } _ { i }$ equals $\mathbf { u } _ { i }$ with probability θ and is uniform otherwise, so $\mathbb { E } [ g ( \mathbf { v } ) \mid \mathbf { u } ] = T _ { \boldsymbol { \theta } } g ( \mathbf { u } )$ . The operators $T _ { \lambda }$ are self-adjoint and satisfy $T _ { \lambda } T _ { \lambda ^ { \prime } } = T _ { \lambda \lambda ^ { \prime } }$ . Hence, by the Cauchy–Schwarz inequality and the bound just proved,

$$
\begin{array} { r } { \mathbb { E } \big [ f ( \mathbf { u } ) g ( \mathbf { v } ) \big ] = \langle f , T _ { \theta } g \rangle = \langle T _ { \sqrt { \theta } } f , T _ { \sqrt { \theta } } g \rangle \leq \| f \| _ { p } \| g \| _ { p } = \big ( \mathbb { E } f \mathbb { E } g \big ) ^ { 1 / p } , } \end{array}
$$

where the last equality holds because $f$ and $g$ are indicator functions.

Extraction by core refinement. We now prove Lemma 93, using the notation of its statement. The main idea is to replace each common label $a _ { i } ^ { j }$ by cores, one for each of the R positions of a representing tuple, and to induct on the deficiency, the total number of labels by which the cores fall short of size ℓ. Fix, for every vector $b ,$ a representing tuple whose hypotheses contain the cores of the groups chosen by $b ,$ and draw b uniformly at random. Call an input heavy if, given its group, the hypothesis at some position of the random tuple contains a fixed label outside its core with probability at least a small constant δ. If at least half of the inputs are heavy, we add such a label to a core at each heavy input. This lowers the deficiency. The chosen tuples remain compatible with the enlarged cores on more than a $1 - 1 / r$ fraction of the heavy inputs on average. Lemma 98 then gives a large set of inputs on which we can apply the induction hypothesis. If at least half of the inputs are light, every label outside a core rarely lies in the corresponding hypothesis. We then intersect hypotheses taken from a few correlated copies of the random vector. When the copies agree on group j at input $i ,$ the intersection contains a fixed set $T _ { i } ^ { j }$ , defined by the cores of that group. The noise inequality shows that, conditional on this agreement, the intersection equals $\boldsymbol { T } _ { i } ^ { j }$ with probability close to one. Lemma 98, applied to these intersections, then yields the product of the sets $T _ { i } ^ { j }$ on a constant fraction of the inputs.

Proof of Lemma 93. Represent each pattern $A ^ { b }$ by a tuple of exactly R hypotheses of $\mathcal { H } _ { 0 }$ repeating hypotheses if needed. If $n = 0$ , the unique pattern on the empty set is the required witness. If $\ell = 1$ , all correct-label sets are singletons, so a nonempty intersection of hypotheses equals each of them on its support. Then $\bar { A } _ { i } ^ { b } = \{ a _ { i } ^ { b _ { i } } \}$ , and the patterns $A ^ { b }$ themselves form the product $\textstyle \prod _ { i \in [ n ] } \left\{ \{ a _ { i } ^ { j } \} : j \in [ r ] \right\}$ , which lies in $\mathcal { T } _ { 1 } ( \mathcal { H } _ { 0 } ) | _ { [ n ] } ;$ its sets at each input i have empty intersection, since $a _ { i } ^ { j } \in Y _ { i } ^ { j }$ . Thus we may take $a _ { 1 , R , r } = 1$ , and we assume from now on that $\ell \geq 2$ and $n \geq 1$

Constants. Set $\theta = 1 - 1 / ( 8 r )$ , let $p \in ( 1 , 2 )$ be given by Lemma 99 for r and $\theta ,$ and put $\epsilon = 2 / p - 1 > 0$ . Fix $0 < \delta < 1$ such that

$$
2 \ell \delta + \ell \delta ^ { \epsilon } \leq \frac { 1 } { 8 r } .\tag{120}
$$

With the constants $C _ { r , \kappa }$ of Lemma 98, write

$$
c _ { \ell , R } = \sum _ { \nu = 1 } ^ { \operatorname* { m i n } \{ \ell , R \} } \binom { R } { \nu } , \qquad C _ { H } = C _ { r , \delta / r } , \qquad C _ { L } = C _ { r , 1 / ( 2 r ) } ,
$$

and $\zeta _ { \Delta } = 1 / \left( 2 c _ { \ell , R } C _ { L } ( 2 C _ { H } ) ^ { \Delta } \right)$ for every integer $\Delta \geq 0$

Cores and the induction statement. Let $Y _ { i } ^ { j }$ be sets with $\textstyle \bigcap _ { i = 1 } ^ { r } Y _ { i } ^ { j } = \varnothing$ at every input $i \in [ n ]$ . A core system consists of nonempty sets $C _ { s } ^ { j } ( i )$ , for $i \in [ n ]$ , positions $s \in [ R ]$ , and groups $j \in [ r ]$ , such that

$$
| C _ { s } ^ { j } ( i ) | \leq \ell , \qquad \varnothing \ne \bigcap _ { s = 1 } ^ { R } C _ { s } ^ { j } ( i ) \subseteq Y _ { i } ^ { j } .\tag{121}
$$

A tuple $( h _ { 1 } , \ldots , h _ { R } )$ of hypotheses of $\mathcal { H } _ { 0 }$ is compatible with group j at i if

$$
C _ { s } ^ { j } ( i ) \subseteq h _ { s } ( i ) \quad ( s \in [ R ] ) , \qquad \varnothing \neq \bigcap _ { s = 1 } ^ { R } h _ { s } ( i ) \subseteq Y _ { i } ^ { j } .
$$

Since the sets $Y _ { i } ^ { j }$ need only have an empty total intersection, a tuple may be compatible with several groups at the same input. The deficiency at i is

$$
\Delta ( i ) = \sum _ { j = 1 } ^ { r } \sum _ { s = 1 } ^ { R } \bigl ( \ell - | C _ { s } ^ { j } ( i ) | \bigr ) .
$$

We prove the following statement by induction on the integer $\Delta \geq 0$ , for every $n \geq 1$ and all such sets $Y _ { i } ^ { j }$ and cores.

Suppose that $\Delta ( i ) \le \Delta$ for every $i \in \left\lceil n \right\rceil$ , and that for every $b \in [ r ] ^ { n }$ there is a tuple $( h _ { b , 1 } , \ldots , h _ { b , R } )$ of hypotheses of $\mathcal { H } _ { 0 }$ that is compatible with group $b _ { i }$ at every $i \in [ n ]$ . Then one can construct a set $U \subseteq [ n ]$ with $| U | \ge \zeta _ { \Delta } n$ and nonempty sets $T _ { i } ^ { j }$ such that

$$
\prod _ { i \in U } \{ T _ { i } ^ { j } : j \in [ r ] \} \subseteq \mathcal { Z } _ { \ell } ( \mathcal { H } _ { 0 } ) | _ { U } , \qquad \bigcap _ { j = 1 } ^ { r } T _ { i } ^ { j } = \emptyset \quad ( i \in U ) .
$$

The lemma follows from the case $C _ { s } ^ { j } ( i ) = \{ a _ { i } ^ { j } \}$ . Indeed, (121) holds, since $a _ { i } ^ { j } \in A _ { i } ^ { b } \subseteq Y _ { i } ^ { j }$ for every b with $b _ { i } = j$ . The representing tuple of $A ^ { b }$ is compatible with group $b _ { i }$ at every $i ,$ since each of its hypotheses contains $A _ { i } ^ { b } \ni a _ { i } ^ { b _ { i } }$ and their intersection is $A _ { i } ^ { \bar { b } } \subseteq Y _ { i } ^ { b _ { i } }$ . All deficiencies equal $r R ( \ell - 1 )$ , so the statement gives the lemma with

$$
a _ { \ell , R , r } = \zeta _ { r R ( \ell - 1 ) } = \frac { 1 } { 2 c _ { \ell , R } C _ { L } ( 2 C _ { H } ) ^ { r R ( \ell - 1 ) } } .
$$

Heavy and light inputs. Assume the hypotheses of the induction statement, and let b be uniform on $[ r ] ^ { n }$ . Call an input i heavy if some group $j ,$ position s, and label $y \notin C _ { s } ^ { j } ( i )$ satisfy

$$
\begin{array} { r } { \operatorname* { P r }  { \left[ y \in h _ { \mathbf { b } , s } ( i ) \ \middle \vert \ \mathbf { b } _ { i } = j \right] } \geq \delta , } \end{array}\tag{122}
$$

and light otherwise. If $\Delta = 0$ , every core has ℓ labels, so a core contained in a correct-label set equals it, and no input is heavy. Either at least $n / 2$ inputs are heavy, or at least $n / 2$ inputs are light; we treat the two cases in turn.

Heavy inputs. Suppose that the set $I _ { \mathrm { h } }$ of heavy inputs has at least $n / 2$ elements; then $\Delta \geq 1$ . At each $i \in I _ { \mathrm { h } }$ , choose a triple $( j , s , y )$ satisfying (122) and add y to $C _ { s } ^ { j } ( i )$ . Since the probability in (122) is positive, some b with $b _ { i } = j$ has $y \in h _ { b , s } ( i )$ . The tuple of this b is compatible with group j at i, so the enlarged core is contained in $h _ { b , s } ( i )$ and still has at most ℓ labels. The intersection of the enlarged cores remains nonempty because it contains the previous intersection. Each enlarged core is contained in the corresponding hypothesis of this tuple, so their intersection is still contained in $\cap _ { s } h _ { b , s } ( i ) \subseteq Y _ { i } ^ { j }$ . The other cores are unchanged. Thus the enlarged cores on $I _ { \mathrm { h } }$ satisfy (121), and their deficiency is at most $\Delta - 1$ at every input of $I _ { \mathrm { h } }$

It remains to find a large set on which every vector has a tuple compatible with the enlarged cores. Let $\mathcal { F }$ be the partial class on $I _ { \mathrm { h } }$ consisting of all $f$ for which some tuple of hypotheses of $\mathcal { H } _ { 0 }$ is compatible, for the enlarged cores, with group $f _ { i }$ at every $i \in \operatorname { s u p p } ( f )$ Fix $i \in I _ { \mathrm { h } }$ with chosen triple $( j , s , y )$ . If $\mathbf { b } _ { i } \neq j$ , the corresponding cores are unchanged, so the tuple of b remains compatible at i. If $\mathbf { b } _ { i } = j$ , it remains compatible exactly when $y \in h _ { \mathbf { b } , s } ( i )$ an event of conditional probability at least $\delta .$ . Hence the expected number of inputs of $I _ { \mathrm { h } }$ at which the tuple of b remains compatible is at least $( 1 - 1 / r + \delta / r ) | I _ { \mathrm { h } } |$ . For each vector $b ^ { \prime } \in [ r ] ^ { I _ { \mathrm { h } } }$ , choose an extension $b \in [ r ] ^ { n }$ that maximizes the number of such inputs, and let $f ^ { b ^ { \prime } }$ agree with $b ^ { \prime }$ at these inputs and be undefined at the other inputs of $I _ { \mathrm { h } }$ . Then $f ^ { b ^ { \prime } } \in \mathcal { F }$ , and since a maximum is at least an average, the supports of the functions $f ^ { b ^ { \prime } }$ have average size at least $( 1 - 1 / r + \delta / r ) | I _ { \mathrm { h } } |$ . Lemma $9 8$ , with $\kappa = \delta / r$ , gives a set $U \subseteq I _ { \mathrm { h } }$ shattered by $\mathcal { F }$ with

$$
| U | \geq { \frac { | I _ { \mathrm { h } } | } { C _ { H } } } \geq { \frac { n } { 2 C _ { H } } } .
$$

Shattering means that every vector of $[ r ] ^ { U }$ has a tuple compatible with the enlarged cores at every input of $U$ . The induction hypothesis for $\Delta - 1$ , applied on $U$ (relabeled as $[ | U | ] )$ constructs the required product on at least

$$
\zeta _ { \Delta - 1 } | U | \geq { \frac { \zeta _ { \Delta - 1 } } { 2 C _ { H } } } n = \zeta _ { \Delta } n
$$

inputs.

Light inputs: a certificate. Suppose now that at least $n / 2$ inputs are light. For a light input i and a position $s \in [ R ]$ , put $\begin{array} { r } { E _ { s } ( i ) = \bigcap _ { j = 1 } ^ { r } C _ { s } ^ { j } ( i ) } \end{array}$ . These sets have empty intersection, since by (121)

$$
\bigcap _ { s = 1 } ^ { R } E _ { s } ( i ) = \bigcap _ { j = 1 } ^ { r } \bigcap _ { s = 1 } ^ { R } C _ { s } ^ { j } ( i ) \subseteq \bigcap _ { j = 1 } ^ { r } Y _ { i } ^ { j } = \varnothing .
$$

We find a small subfamily with empty intersection. Some $E _ { s } ( i )$ has at most $\ell - 1$ labels. Indeed, if all of them had ℓ labels, then each core $C _ { s } ^ { j } ( i )$ , which contains $E _ { s } ( i )$ and has at most ℓ labels, would equal $E _ { s } ( i )$ and so would not depend on $j ;$ the nonempty set $\cap _ { s } C _ { s } ^ { 1 } ( i )$ would then lie in every $Y _ { i } ^ { j }$ , a contradiction. Take such an $s ,$ and for each label of $E _ { s } ( i )$ choose a position $s ^ { \prime }$ such that $E _ { s ^ { \prime } } ( i )$ omits it. This gives a nonempty certificate $S _ { i } \subseteq [ R ]$ with $| S _ { i } | \le \ell$ and $\begin{array} { r } { \bigcap _ { s \in S _ { i } } E _ { s } ( i ) = \emptyset } \end{array}$

There are at most $c _ { \ell , R }$ possible certificates, so some set $I _ { \mathrm { l } }$ of light inputs with a common certificate S satisfies $\lvert I _ { \mathrm { l } } \rvert \ge n / ( 2 c _ { \ell , R } )$ . List $S$ as $s _ { 1 } , \ldots , s _ { \nu } ;$ if $S$ is a singleton, repeat its member and set $\nu = 2$ . Thus $2 \leq \nu \leq \ell$ . For $i \in I _ { \mathrm { l } }$ and $j \in [ r ]$ , the sets

$$
T _ { i } ^ { j } = \bigcap _ { a = 1 } ^ { \nu } C _ { s _ { a } } ^ { j } ( i )
$$

are nonempty, since they contain $\cap _ { s } C _ { s } ^ { j } ( i )$ , and $\begin{array} { r } { \bigcap _ { j } T _ { i } ^ { j } = \bigcap _ { s \in S } E _ { s } ( i ) = \emptyset } \end{array}$ . We now show that intersections of $\nu$ hypotheses realize the product of these sets on a large subset of $I _ { 1 }$

Light inputs: correlated copies. Draw a uniform hub b $\in [ r ] ^ { n }$ and copies $\mathbf { b } ^ { ( 1 ) } , \ldots , \mathbf { b } ^ { ( \nu ) } \in$ $[ r ] ^ { n }$ , independently across inputs, as follows: at each input, with probability $\theta$ all copies equal the symbol of the hub, and otherwise they are independent uniform symbols, independent also of the hub. Each copy is uniform on $[ r ] ^ { n }$ , and any two copies form a pair as in Lemma 99. Put

$$
P = \bigcap _ { a = 1 } ^ { \nu } h _ { \mathbf { b } ^ { ( a ) } , s _ { a } } ,
$$

restricted to the inputs where this intersection is nonempty. The same $\nu \leq \ell$ hypotheses of $\mathcal { H } _ { 0 }$ are intersected at every input, so every outcome of $P$ belongs to $\mathcal { T } _ { \ell } ( \mathcal { H } _ { 0 } )$

Fix $i \in I _ { \mathrm { l } }$ and $j \in [ r ]$ . Condition on $\mathbf { b } _ { i } = j$ and on the event that all copies have symbol $j$ at $i ;$ given $\mathbf { b } _ { i } = j$ , this event has probability at least $\theta .$ . The other inputs keep their joint law, so each copy is distributed as a uniform vector conditioned on having symbol $j$ at $i ,$ , and the bounds that define lightness apply to it. Moreover, the tuple of each copy is compatible with group $j$ at $i ,$ so $h _ { \mathbf { b } ^ { ( a ) } , s _ { a } } ( i ) \supseteq C _ { s _ { a } } ^ { j } ( i )$ for every $^ { a , }$ and hence $T _ { i } ^ { j } \subseteq P _ { i }$ . All probabilities in the rest of this paragraph are conditional on these events, and we bound the probability that $P _ { i }$ contains a label outside $T _ { i } ^ { j }$ . Write $C ^ { \prime } = C _ { s _ { 1 } } ^ { j } ( i )$ and $C ^ { \prime \prime } = C _ { s _ { 2 } } ^ { j } ( i )$ , and note that $P _ { i } \subseteq h _ { \mathbf { b } ^ { ( 1 ) } , s _ { 1 } } ( i ) \cap h _ { \mathbf { b } ^ { ( 2 ) } , s _ { 2 } } ( i )$ . A label of $P _ { i }$ outside $T _ { i } ^ { j }$ is of one of three kinds.

(i) A label of $C ^ { \prime } \setminus T _ { i } ^ { j }$ . It lies outside $C _ { s _ { a } } ^ { j } ( i )$ for some $^ { a , }$ so it lies in $P _ { i } \subseteq h _ { \mathbf { b } ^ { ( a ) } , s _ { a } } ( i )$ with probability less than δ.

(ii) A label of $C ^ { \prime \prime } \backslash C ^ { \prime }$ . It lies in $P _ { i }$ only if it lies in $h _ { \mathbf { b } ^ { ( 1 ) } , s _ { 1 } } ( i )$ outside $C ^ { \prime }$ , which again has probability less than $\delta .$

(iii) A label $y \notin C ^ { \prime } \cup C ^ { \prime \prime }$ . It lies in $P _ { i }$ only if it lies in both $h _ { \mathbf { b } ^ { ( 1 ) } , s _ { 1 } } ( i )$ and $h _ { \mathbf { b } ^ { ( 2 ) } , s _ { 2 } } ( i )$

$\mathrm { A s } | C ^ { \prime } | , | C ^ { \prime \prime } | \leq \ell .$ , labels of kinds (i) and (ii) contribute at most 2ℓδ to the probability that $P _ { i } \neq T _ { i } ^ { j }$ . For a label $y$ of kind (iii) and $a \in \{ 1 , 2 \}$ , let $p _ { a , y }$ be the probability that $y \in h _ { \mathbf { b } ^ { ( a ) } , s _ { a } } ( i )$ Then $p _ { a , y } < \delta$ by lightness, and $\begin{array} { r } { \sum _ { y } p _ { a , y } \le \ell } \end{array}$ , because correct-label sets have at most ℓ labels. The two events $y \in h _ { \mathbf { b } ^ { ( a ) } , s _ { a } } ( i ) , a = 1 , 2$ , are determined by the coordinates of $\mathbf { b } ^ { ( 1 ) }$ and $\mathbf { b } ^ { ( 2 ) }$ at the other $n - 1$ inputs, which form a pair as in Lemma 99. That lemma, applied to the two indicator functions, gives

$$
\sum _ { y \notin { \cal C } ^ { \prime } \cup { \cal C } ^ { \prime \prime } } \operatorname* { P r } \lbrack y \in h _ { \mathbf { b } ^ { ( 1 ) } , s _ { 1 } } ( i ) \cap h _ { \mathbf { b } ^ { ( 2 ) } , s _ { 2 } } ( i ) \rbrack \leq \sum _ { y \notin { \cal C } ^ { \prime } \cup { \cal C } ^ { \prime \prime } } ( p _ { 1 , y } p _ { 2 , y } ) ^ { 1 / p } \leq \delta ^ { \epsilon } \sum _ { y \notin { \cal C } ^ { \prime \prime } \cup { \cal C } ^ { \prime \prime } } \sqrt { p _ { 1 , y } p _ { 2 , y } } \leq \ell \delta ^ { \epsilon } ,
$$

where the second inequality uses $( p _ { 1 , y } p _ { 2 , y } ) ^ { 1 / p - 1 / 2 } \ : \leq \ : \delta ^ { 2 / p - 1 } = \delta ^ { \epsilon }$ , and the last one is the Cauchy–Schwarz inequality. This also applies when $s _ { 1 } = s _ { 2 }$ , since the two hypotheses come from diferent copies. By (120), the conditional probability that $P _ { i } \neq T _ { i } ^ { j }$ is at most $1 / ( 8 r )$ Removing the conditioning gives

Pr-all copies agree with the hub at i, and $P _ { i } = T _ { i } ^ { \mathbf { b } _ { i } } \Big ] \geq \theta \Big ( 1 - \frac { 1 } { 8 r } \Big ) = \theta ^ { 2 } \geq 1 - \frac { 1 } { 4 r } ,$

(123)

and the right-hand side exceeds $1 - 1 / r + 1 / ( 2 r )$

Light inputs: coverage. Call an input $i \in I _ { \mathrm { l } }$ good for an outcome of the hub and the copies if the event in (123) occurs at i. Let ${ \mathcal { F } } ^ { \prime }$ be the partial class on $I _ { \mathrm { l } }$ consisting of all $f$ for which some outcome of $P$ satisfies $P _ { i } = T _ { i } ^ { f _ { i } }$ for every $i \in \operatorname { s u p p } ( f )$ ; when some sets $T _ { i } ^ { j }$ coincide, several symbols are allowed at i. For each vector $b ^ { \prime } \in [ r ] ^ { I _ { 1 } }$ , condition the hub to agree with $b ^ { \prime }$ on $I _ { \mathrm { l } }$ . Choose the remaining values of the hub and the copies so that the number of good inputs is at least its conditional expectation. Let $f ^ { b ^ { \prime } }$ agree with $b ^ { \prime }$ at these good inputs and be undefined elsewhere on $I _ { \mathrm { l } } .$ Then $f ^ { b ^ { \prime } } \in \mathcal { F } ^ { \prime }$ , and by (123) the supports of these functions have average size at least $( 1 - 1 / r + 1 / ( 2 r ) ) | I _ { 1 } |$ . Lemma 98, with $\kappa = 1 / ( 2 r )$ gives a set $U \subseteq I _ { 1 }$ shattered by ${ \mathcal { F } } ^ { \prime }$ with

$$
| U | \geq \frac { | I _ { 1 } | } { C _ { L } } \geq \frac { n } { 2 c _ { \ell , R } C _ { L } } \geq \zeta _ { \Delta } n .
$$

Shattering means that for every vector $b ^ { \prime \prime } \in [ r ] ^ { U }$ some outcome of $P$ satisfies $P _ { i } = T _ { i } ^ { b _ { i } ^ { \prime \prime } }$ for all $i \in U$ . Restricted to $U _ { \textrm { \scriptsize i } }$ these outcomes therefore form exactly the product $\textstyle \prod _ { i \in U } \{ T _ { i } ^ { j } : j \in [ r ] \}$ and each of them is the restriction to $U$ of a member of $\mathcal { T } _ { \ell } ( \mathcal { H } _ { 0 } )$ defined on U. Since $\cap _ { j } T _ { i } ^ { j } = \emptyset$ for every $i \in U$ , this proves the induction statement in the light case.

Closing the induction. If $\Delta = 0$ , all inputs are light, and the light case applies. If $\Delta \geq 1$ either the heavy case reduces the statement to $\Delta - 1$ on a set of at least $n / ( 2 C _ { H } )$ inputs, or the light case constructs the product directly. This proves the induction statement for every $\Delta$ , and hence the lemma with $a _ { \ell , R , r } = \zeta _ { r R ( \ell - 1 ) }$ . All choices range over finite sets, so the construction is a finite procedure, and every pattern of the product is represented by a tuple of at most ℓ hypotheses of $\mathcal { H } _ { 0 }$

The product is a DS witness: in direction i, fixing the sets at all other inputs of $U$ leaves exactly the distinct sets $T _ { i } ^ { j } , j \in [ r ]$ , at input i, and their intersection is empty; this holds also when sets of diferent groups coincide. For $r = 2$ , the sets $T _ { i } ^ { 1 }$ and $T _ { i } ^ { 2 }$ are disjoint, so every pattern of the product has a disjoint neighbor in every direction, and the product is also a disjoint-neighbor witness. □

## B.5 Proof of the Bounded-List Lemma

We now prove Lemma 95. The blocks may lie in all directions and conflict through arbitrary labels, so we first hash the correct-label sets into codes over a finite alphabet; then most blocks conflict through one fixed family $\mathcal { I }$ of r codes. Averaging over the multiplicities of the coded patterns gives a family of words over $[ r ]$ with many complete hyperedges per word, and compression turns it into a down-set, which projects onto a full cube $[ r ] ^ { \bar { E } }$ with $| E |$ of order $N / M ;$ undoing the compressions preserves this projection. Reserved symbols of the codes then give common labels on $E ,$ and Corollary 94 extracts the witness.

Proof of Lemma $g { 5 } .$ . Identify the pool with $[ n ] ;$ as before, every correct-label set below is a subset of the finite set ${ \mathcal { V } } _ { 0 }$ of labels used by $\mathcal { H } _ { 0 }$ . All tie-breaking rules are fixed in advance. Every pattern selected below is the restriction of some $v ^ { a }$ to a set of coordinates at which it is defined, and it is represented by the given representing tuple of $v ^ { a } ;$ since $\mathcal { T } _ { R } ( \mathcal { H } _ { 0 } )$ is closed under domain restrictions, it is a pattern of $\mathcal { T } _ { R } ( \mathcal { H } _ { 0 } )$ .

Encoding the correct-label sets. Order each list, and let $\iota _ { i } : Q ( i )  [ | Q ( i ) | ]$ be the resulting bijection; we call the symbols of $[ L ]$ reserved. Put

$$
N _ { 0 } = \ell { ( \ell + 1 ) } , \qquad m _ { 0 } = N _ { 0 } ( N _ { 0 } - 1 ) ,
$$

so that $\binom { N _ { 0 } } { 2 } / m _ { 0 } = 1 / 2$ . Independently for every coordinate i and every label $y \in \mathcal { V } _ { 0 } \setminus Q ( i )$ choose a uniform value $\chi _ { i } ( y ) \in [ m _ { 0 } ]$ , and extend $\iota _ { i }$ by $\iota _ { i } ( y ) = L + \chi _ { i } ( y )$ . Thus every label receives a symbol in $[ L + m _ { 0 } ]$ , the labels of $Q ( i )$ receive distinct reserved symbols, and no other label receives a reserved symbol. The code of a correct-label set A at i is

$$
\phi _ { i } ( A ) = \iota _ { i } ( A ) = \{ \iota _ { i } ( y ) : y \in A \} \qquad { \mathrm { i f ~ } } A \cap Q ( i ) \neq \varnothing ,
$$

and $\phi _ { i } ( A ) = { \star \operatorname { i f } { \cal A } }$ misses $Q ( i )$ ; we also put $\phi _ { i } ( \star ) = \star$ . Every code other than ⋆ has between one and ℓ symbols and contains a reserved symbol, so it belongs to the alphabet

$$
\mathcal Z = \big \{ Z \subseteq [ L + m _ { 0 } ] : 1 \leq | Z | \leq \ell , \ Z \cap [ L ] \neq \emptyset \big \} ,
$$

which depends only on ℓ and L. For a partial pattern v we call ${ \boldsymbol { \phi } } ( { \boldsymbol { v } } ) = ( \phi _ { 1 } ( v _ { 1 } ) , \dots , \phi _ { n } ( v _ { n } ) )$ its coded pattern.

A common label of sets that meet the list gives a common symbol of their codes, but hash collisions can create further common symbols. Call a block surviving if the codes of its sets have an empty intersection; we show that at least half of the blocks survive in expectation. Let $B \in B _ { i }$ . Its sets $v _ { i } ^ { a } , a \in B$ , all meet $Q ( i )$ , so their codes are not $\star ,$ and together they contain at most $| B | \ell \leq N _ { 0 }$ labels outside $Q ( i )$ . By the union bound over pairs, these labels receive distinct hash values with probability at least $1 - \binom { N _ { 0 } } { 2 } / m _ { 0 } = 1 / 2$ , and then $B$ survives. Indeed, a common reserved symbol would be the index of a label of $Q ( i )$ lying in every $v _ { i } ^ { a }$ $a \in B$ . A common symbol $L + w$ would come, in every $v _ { i } ^ { a }$ , from a label outside $Q ( i )$ with hash value $w$ , and since these labels have distinct hash values, it would come from one label lying in every $v _ { i } ^ { a }$ . Either case contradicts $\bigcap { } _ { a \in B } v _ { i } ^ { a } = \varnothing$ . Hence we may fix hash values for which at least $N / 2$ blocks survive; from now on the codes are fixed.

The codes are nonempty sets of at most ℓ symbols. By Lemma 8, the distinct codes of a surviving block therefore contain a minimal empty-intersection family, which has between two and $\ell + 1$ members. The number of such families of symbols of $\mathcal { Z }$ is at most

$$
\Gamma = \sum _ { j = 2 } ^ { \ell + 1 } { \binom { | \mathcal { Z } | } { j } } ,
$$

which depends only on ℓ and L. Assign to every surviving block one such family contained in its codes, and fix a family $\mathcal { I } = \{ Z _ { 1 } , \ldots , Z _ { r } \}$ , with $2 \leq r \leq \ell + 1$ , that is assigned to at least $N / ( 2 \Gamma )$ blocks. The selected blocks now have the same family $\mathcal { I }$ of conflicting codes, even when they lie in diferent directions.

Selecting complete collections. We now pass from the blocks to the coded patterns. Let V be the set of distinct coded patterns $\phi ( v ^ { a } ) , a \in [ M ]$ , and let $\omega ( w )$ be the number of indices $a \in [ M ]$ with $\phi ( v ^ { a } ) = w$ , so that $\textstyle \sum _ { w \in V } \omega ( w ) = M$ . A complete collection in direction i is a set of r members of V that agree at every coordinate other than i and take the values $Z _ { 1 } , \ldots , Z _ { r }$ at $i .$ Its members difer only at $i ,$ so a complete collection determines its direction. For $V ^ { \prime } \subseteq V$ we write $\mathcal { E } ( V ^ { \prime } )$ for the set of complete collections contained in $V ^ { \prime }$

Every block $B \in B _ { i }$ assigned to J determines a complete collection $e _ { B }$ in direction $i .$ Indeed, the coded patterns $\phi ( v ^ { a } ) , a \in B$ , agree of $i ,$ because the partial patterns $v ^ { a }$ agree of $i ,$ including their domains; and their codes at i include $Z _ { 1 } , \ldots , Z _ { r }$ , so those among them with these codes at i form a complete collection. Now fix $e \in { \mathcal { E } } ( V )$ with direction i. The blocks $B$ with $e _ { B } = e$ belong to $B _ { i } .$ , so they are pairwise disjoint, and each of them contains, for every $w \in e ,$ an index a with $\phi ( v ^ { a } ) = w$ . Hence at most min $\boldsymbol { \mathrm { 1 } } _ { w \in e } \omega ( w )$ blocks determine $e ,$ and summing over e gives

$$
\sum _ { e \in \mathcal { E } ( V ) } \operatorname* { m i n } _ { w \in e } \omega ( w ) \geq \frac { N } { 2 \Gamma } .\tag{124}
$$

We remove the multiplicities by splitting V into layers. For an integer $s \geq 1$ , let $V _ { s } = \{ w \in V : \omega ( w ) \geq s \}$ . A complete collection e lies in $V _ { s }$ exactly when min $. w \in e \omega ( w ) \geq s$ so

$$
\sum _ { s \geq 1 } | { \mathcal { E } } ( V _ { s } ) | = \sum _ { e \in { \mathcal { E } } ( V ) } \operatorname* { m i n } _ { w \in e } \omega ( w ) , \qquad \sum _ { s \geq 1 } | V _ { s } | = \sum _ { w \in V } \omega ( w ) = M .
$$

By weighted averaging and (124), some nonempty layer satisfies

$$
\frac { | \mathcal { E } ( V _ { s } ) | } { | V _ { s } | } \geq \frac { N } { 2 \Gamma M } = : \xi .\tag{125}
$$

Next we fix the values outside $\mathcal { I }$ . Partition $V _ { s }$ according to the set of coordinates at which the value lies outside $\mathcal { I }$ (the value ⋆ included) and the values at these coordinates. The members of a complete collection agree of its direction and take values in $\mathcal { I }$ at it, so each complete collection lies in one part, and by the same averaging some part $V ^ { \prime }$ satisfies $| \mathcal { E } ( V ^ { \prime } ) | \geq \xi | V ^ { \prime } |$ . Let I be the set of coordinates at which the members of $V ^ { \prime }$ take values in $\mathcal { I }$ Deleting the other coordinates, at which all members of $V ^ { \prime }$ agree, and identifying $Z _ { j }$ with j for $j \in [ r ]$ , we obtain a family $F \subseteq [ r ] ^ { I }$ of words with $| F | = | V ^ { \prime } |$ . The complete collections of $V ^ { \prime }$ become the hyperedges of the one-inclusion hypergraph of $F$ with r vertices, that is, the sets of $r$ words of $F$ that agree of some $i \in I$ and take all values of [r] at i. We still call these hyperedges complete collections and write $\mathcal { E } ( F )$ for their set, so that

$$
| \mathcal { E } ( F ) | \geq \xi | F | .
$$

Compression and recovery of a product. It remains to find a large set of coordinates on which F projects onto a full cube, and we do this by compression, or shifting, a standard technique for one-inclusion graphs [HLW94]. For a family $G \subseteq [ r ] ^ { I }$ and $i \in I$ , the compression of G in direction i replaces, in every hyperedge of G in direction i with ν words, the ν values at i by $1 , \ldots , \nu .$ . It keeps the number of words, the set of restrictions of the words to $I \setminus \{ i \}$ and the sizes of the hyperedges in direction i.

Claim 100. Compression keeps |G| and does not decrease $| \mathcal { E } ( G ) |$

Proof. Compression in direction i keeps the sizes of the hyperedges in direction i, hence the number of complete collections in direction i. Let $i ^ { \prime } \neq i ,$ fix the values at all coordinates other than i and $i ^ { \prime } { . }$ , and for $j \in [ r ]$ let $\Sigma _ { j } \subseteq [ r ]$ be the set of values at $i$ of the words of $G$ with these values and with value $j$ at $i ^ { \prime } .$ . The complete collections in direction $i ^ { \prime }$ among these words correspond to the values at i that occur with every $j .$ . Compression replaces each $\Sigma _ { j }$ by $\{ 1 , \dotsc , | \Sigma _ { j } | \}$ , so their number changes from

$$
\Bigl | \bigcap _ { j = 1 } ^ { r } \Sigma _ { j } \Bigr | \qquad \mathrm { t o } \qquad \operatorname* { m i n } _ { j } | \Sigma _ { j } | \geq \Bigl | \bigcap _ { j = 1 } ^ { r } \Sigma _ { j } \Bigr | .
$$

Summing over the fixed values proves the claim.

Every compression that changes the family strictly decreases the sum of the coordinate values of its words. Repeated compression therefore stops after finitely many steps at a family $F _ { * }$ fixed by every compression, and $| \mathcal { E } ( F _ { * } ) | \ge \xi | F _ { * } |$ by the claim. Every hyperedge of $F _ { * }$ in a direction i takes the values $1 , \ldots , \nu$ at i for some ν. So $F _ { * }$ is a down-set: lowering a value greater than one of a word of $F _ { * }$ by one gives a word of $F _ { * }$ , and repeating this shows that $F _ { * }$ contains every word that is coordinatewise at most a word of $F _ { * }$ . In particular, a hyperedge of $F _ { * }$ in direction $i$ is a complete collection exactly when it contains a word with value r at $i ,$ and this word is unique. Hence

$$
\sum _ { w \in F _ { * } } | \{ i \in I : w _ { i } = r \} | = | { \mathcal { E } } ( F _ { * } ) | \geq \xi | F _ { * } | ,
$$

and some word $w ^ { \ast } \in F _ { \ast }$ takes the value r at every coordinate of a set $E \subseteq I$ with $| E | \geq \xi$ As $F _ { * }$ is a down-set, it contains, for every $b \in [ r ] ^ { E }$ , the word that equals b on $E$ and $w ^ { * }$ of $E$

Thus $F _ { * } | _ { E } = [ r ] ^ { E }$

We now undo the compressions one at a time, in reverse order, and show that each family G in the sequence from $F$ to $F _ { * }$ satisfies $G | _ { E } = [ r ] ^ { E }$ . Suppose that $G ^ { \prime }$ is the compression of $G$ in direction i and $G ^ { \prime } | _ { E } = [ r ] ^ { E }$ . If $i \not \in E$ , then $G | _ { E } = G ^ { \prime } | _ { E }$ , since compression keeps the restrictions to $I \setminus \{ i \}$ . I ${ \mathrm { ~ f ~ } } i \in E$ , fix $b ^ { \prime } \in \lceil r \rceil ^ { E \setminus \{ i \} }$ , and let $w \in G ^ { \prime }$ restrict to $b ^ { \prime }$ on $E \setminus \{ i \}$ and take the value r at i. The hyperedge of $w$ in direction i in $G ^ { \prime }$ takes the values $1 , \ldots , \nu$ at i and contains $r ,$ so it has r words. The words of G that agree with w of i form a hyperedge of the same size, so they take all $r$ values at i, and they restrict to all $r$ extensions of $b ^ { \prime }$ to $E$ As $b ^ { \prime }$ was arbitrary, $G | _ { E } = [ r ] ^ { E }$ . In particular, $F | _ { E } = [ r ] ^ { E }$

Recovering groups with a common label. Each word of $F$ comes from a coded pattern of $V ^ { \prime }$ by deleting coordinates at which all members of $V ^ { \prime }$ agree. Hence, by the last paragraph, for every $b \in [ r ] ^ { E }$ there is an index $a ( b ) \in [ M ]$ with $\phi _ { i } ( v _ { i } ^ { a ( b ) } ) = Z _ { b _ { i } }$ for all $i \in E .$ . In particular, $\boldsymbol { v } ^ { a ( b ) }$ is defined on $E$ , and we let $A ^ { b }$ be its restriction to $E _ { i }$ a pattern of $\mathcal { T } _ { R } ( \mathcal { H } _ { 0 } )$ represented by the representing tuple of $v ^ { a ( b ) }$ . For every $j \in [ r ]$ , fix a reserved symbol $z _ { j } \in Z _ { j } \cap [ L ]$ , and for $i \in E$ put

$$
a _ { i } ^ { j } = \iota _ { i } ^ { - 1 } ( z _ { j } ) , \qquad Y _ { i } ^ { j } = \bigcup _ { b : b _ { i } = j } A _ { i } ^ { b } .
$$

The label $a _ { i } ^ { j }$ is well defined and lies in $A _ { i } ^ { b }$ whenever $b _ { i } = j$ . Indeed, for such b the symbol $z _ { j }$ belongs to $Z _ { j } = \iota _ { i } ( A _ { i } ^ { b } )$ , and the only label with the reserved symbol $z _ { j }$ is the label of $Q ( i )$ with index $z _ { j } ;$ so this label exists and lies in $A _ { i } ^ { b }$ . Hence

$$
a _ { i } ^ { b _ { i } } \in A _ { i } ^ { b } \subseteq Y _ { i } ^ { b _ { i } } \qquad ( b \in [ r ] ^ { E } , \ i \in E ) .
$$

Moreover, $\textstyle \bigcap _ { i = 1 } ^ { r } Y _ { i } ^ { j } = \varnothing$ for every $i \in E$ . Indeed, a label y in this intersection lies in some $A _ { i } ^ { b }$ with $b _ { i } = j$ , for every $j$ , so its symbol $\iota _ { i } ( y )$ belongs to every $Z _ { j } { \mathrm { ; } }$ ; this contradicts $\textstyle \bigcap _ { j = 1 } ^ { r } Z _ { j } = \varnothing$

Thus the hypotheses of Lemma 93 hold on the $| E |$ inputs of $E$ , with the representing tuples of the partial patterns $v ^ { a ( b ) }$ , and Corollary 94 constructs a disjoint-neighbor witness in $\mathcal { T } _ { \ell } ( \mathcal { H } _ { 0 } )$ on at least $a _ { \ell , \ell ^ { 2 } , 2 } a _ { \ell , R , r } | E |$ inputs, whose patterns are represented by hypotheses of $\mathcal { H } _ { 0 }$ Since $| E | \ge \xi = N / ( 2 \Gamma M )$ and $2 \leq r \leq \ell + 1$ , this is at least $\beta _ { \ell , R , L } N / M$ with

$$
\beta _ { \ell , R , L } = \frac { a _ { \ell , \ell ^ { 2 } , 2 } } { 2 \Gamma } \operatorname* { m i n } _ { 2 \leq r \leq \ell + 1 } a _ { \ell , R , r } ,
$$

which depends only on $\ell , R ,$ and L. This finishes the proof.

## C Connection with Cost-Sensitive PAC Learning

For finite label spaces, the realizable full-set characterization can also be derived from the cost-sensitive framework of [BBCB<sup>+</sup>25]. We first encode partial set-valued hypotheses as total hypotheses over a finite alphabet, and then relate the cube dimensions of [BBCB<sup>+</sup>25] to $d _ { \mathrm { N } } ^ { \cap }$ . Combined with their characterization, this gives a second proof that realizable full-set learnability is equivalent to $d _ { \mathrm { N } } ^ { \Omega } ( \mathcal { H } ) < \infty$ (Theorem A). Their eliminators also give a samplecomplexity bound, which we compare with our direct construction; neither bound dominates the other.

Throughout this appendix, $k = | \mathcal { y } | < \infty , 1 \leq \ell \leq k$ , and $\mathcal { H } \neq \emptyset$ . As in $( 1 ) - ( 2 )$ , A is the collection of nonempty sets of at most ℓ labels, and $q = | A |$

## C.1 Encoding the Learning Problem

Recall that $h ( x ) = \star$ means that h is undefined at x. Put $\Sigma = { \mathcal { A } } \cup \{ \star \}$ , and let $\mathcal { H } _ { \Sigma } \subseteq \Sigma ^ { \mathcal { X } }$ consist of the same functions as ${ \mathcal { H } } ,$ now with ⋆ treated as an ordinary target symbol. The actions also range over Σ. Fix $s ( A ) \in A$ for each $A \in { \mathcal { A } }$ and any $s ( \star ) \in \mathcal { V }$ , and call $s ( B )$ the label decoded from the action B. The cost of the action B against the target symbol A is

$$
w ( B , A ) = \left\{ { \begin{array} { l l } { \mathbb { 1 } \{ s ( B ) \notin A \} , } & { A \in { \mathcal { A } } , } \\ { 0 , } & { A = \star , } \end{array} } \right. \qquad B \in \Sigma .\tag{126}
$$

This cost has zero diagonal, as required in $\mathrm { [ B B C B ^ { + } 2 5 ] }$ . For every $x \in \operatorname { s u p p } ( h )$ , the cost $w ( B , h ( x ) ) = \mathbb { 1 } \{ s ( B ) \notin h ( x ) \}$ } is exactly the validity loss of the decoded label $s ( B )$ , and every label $y \in \mathcal { V }$ is decoded from the action $\{ y \}$ , since $s ( \{ y \} ) = y$

In the cost-sensitive problem, the input distribution may assign mass to points where the original target is undefined. We first show that this does not change qualitative learnability.

Lemma 101. The class H is realizably PAC learnable from full sets if and only if $\mathcal { H } _ { \Sigma }$ is (w, 0)-learnable in the sense of $\left[ B B C B ^ { + } \mathcal { Q } 5 \right]$

Proof. Suppose first that $\mathcal { H } _ { \Sigma }$ is $( w , 0 )$ -learnable. A full-set sample labeled by $h \in \mathcal H$ is a sample for $\mathcal { H } _ { \Sigma }$ whose inputs lie in supp(h). Run the cost-sensitive learner on it at accuracy $\varepsilon / ( \ell + 1 )$ and decode its predictions by s. Since the cost is the validity loss on supp(h), this gives a full-set learner. If the learner outputs distributions over actions, we predict a most probable decoded label, which by Lemma 11 multiplies the error by at most $\ell + 1$

Conversely, let $m = m _ { \mathcal { H } } ^ { \mathrm { f u l l } } ( \varepsilon , \delta / 2 )$ . Fix a target $h \in { \mathcal { H } }$ and an arbitrary input distribution $P ,$ and write $p = P ( \operatorname { s u p p } ( h ) )$ . The idea is to discard the examples where the target is undefined. Draw

$$
N = \left\lceil { \frac { 2 } { \varepsilon } } \left( m + 4 \log { \frac { 2 } { \delta } } \right) \right\rceil
$$

examples and discard those labeled ⋆. If at least m examples remain, run the full-set learner on the first $m$ of them and encode the returned predictor g by the singleton actions $\{ g ( x ) \}$ ; otherwise return a fixed prediction.

If $p \leq \varepsilon .$ , every predictor has w-risk at most ε, because the cost vanishes outside $\operatorname { s u p p } ( h )$ Now suppose that $p > \varepsilon$ . Then $N p > 2 m + 8 \log ( 2 / \delta )$ , and since the number $Z$ of retained examples has mean $N p$ , a Chernof bound gives

$$
\operatorname* { P r } [ Z < m ] \leq \exp ( - N p / 8 ) \leq \delta / 2 .
$$

Conditional on the retained positions, the retained inputs are independent draws from $P$ conditioned on supp(h). Thus, conditional on $Z \geq m$ , with probability at least $1 - \delta / 2$ the predictor $g$ has validity error at most ε under this conditional distribution. Its w-risk is p times this error, hence at most $p \varepsilon \leq \varepsilon$ . The two failure probabilities sum to at most δ.

## C.2 Conflicts and Cube Dimensions

In the terminology of $\mathrm { [ B B C B ^ { + } 2 5 ] }$ , a family $K \subseteq \Sigma$ is dice-attainable at threshold zero if

$$
\forall \mu \in \Delta ( K ) \ \exists \nu \in \Delta ( \Sigma ) : \quad \mathbb { E } _ { A \sim \mu , B \sim \nu } \big [ w ( B , A ) \big ] = 0 ,
$$

and it is obstructed otherwise. Since ⋆ costs nothing, only the defined symbols of K matter: put $\begin{array} { r } { I ( K ) = \bigcap _ { A \in K \backslash \{ \star \} } A } \end{array}$ , with the intersection of the empty family interpreted as Y. We claim that K is dice-attainable at threshold zero exactly when $I ( K ) \neq \emptyset$

Indeed, a label $y \in I ( K )$ gives the action $\{ y \}$ , which has zero cost against every symbol of K. Conversely, if $I ( K ) = \emptyset$ , take the target symbol uniformly from $K \setminus \{ \star \}$ . Every decoded label is excluded by at least one of these sets, so every action distribution incurs cost at least $1 / | K \setminus \{ \star \} | > 0$ . Thus the obstructed families are precisely

$$
\mathfrak { I } _ { 0 } = \{ K \subseteq \Sigma : I ( K ) = \varnothing \} .\tag{127}
$$

Recall the collection C of minimal empty-intersection families in $A$ and its size Γ from (12). By Lemma $^ { 8 , }$ every $J \in { \mathfrak { C } }$ has between two and $\ell + 1$ members. Every member of C is obstructed, and every $K \in \mathfrak { I } _ { 0 }$ contains some $J \in \mathfrak { C }$ , namely a minimal subfamily of $K \setminus \{ \star \}$ with empty intersection. Hence the minimal obstructed families are exactly the members of C, and none of them contains ⋆.

For $J \subseteq \Sigma$ with $| J | \geq 2$ , the J-cube dimension $d _ { J }$ of $\mathcal { H } _ { \Sigma }$ is the supremum of n such that $J ^ { n } \subseteq \mathcal { H } _ { \Sigma } | _ { S } = \{ ( h ( x _ { 1 } ) , \ldots , h ( x _ { n } ) ) : h \in \mathcal { H } _ { \Sigma } \}$ for some sequence $S = ( x _ { 1 } , \ldots , x _ { n } ) $ of distinct inputs, with entries equal to ⋆ allowed $[ \mathrm { B B C B ^ { + } 2 5 }$ , Definition 2]. If $J \subseteq A .$ , the cube uses only defined values, so $J ^ { n } \subseteq { \mathcal { H } } _ { \Sigma } | _ { S }$ is equivalent to $J ^ { n } \subseteq { \mathcal { H } } | _ { S }$ . The next lemma shows that $d _ { \mathrm { N } } ^ { \bigcap }$ is sandwiched between the cube dimensions of the conflict families.

Lemma 102. With the maximum and sum over an empty family both interpreted as zero,

$$
\operatorname* { m a x } _ { J \in { \mathfrak { C } } } d _ { J } \ \leq \ d _ { \mathrm { N } } ^ { \cap } ( { \mathcal { H } } ) \ \leq \ \sum _ { J \in { \mathfrak { C } } } d _ { J } \ \leq \ \Gamma \operatorname* { m a x } _ { J \in { \mathfrak { C } } } d _ { J } .\tag{128}
$$

Moreover, all the dimensions $d _ { K } , K \in \mathfrak { I } _ { 0 }$ , are finite if and only if all the dimensions $d _ { J } ,$ $J \in { \mathfrak { C } }$ , are finite.

Proof. A J-cube with $J \in { \mathfrak { C } }$ is a product witness, with the family J at every input (Definition 3); this proves the first inequality. For the second, take a product witness $\mathcal { A } _ { 1 } \times \cdots \times \mathcal { A } _ { n }$ and choose a minimal empty-intersection family $J _ { i } \subseteq A _ { i }$ at every coordinate. Group the coordinates according to the chosen family: for $J \in \mathfrak { C } .$ let $T _ { J } = \{ i : J _ { i } = J \}$ . The projection onto $T _ { J }$ contains a J-cube, so $| T _ { J } | \leq d _ { J }$ . Summing gives $n \leq \sum _ { J } d _ { J }$ , and the second inequality follows because the witness was arbitrary. The last inequality follows by bounding each summand by the maximum.

For the final assertion, every obstructed family K contains a $J \in \mathfrak { C }$ . A K-cube contains a J-cube on the same inputs, so $d _ { K } \leq d _ { J }$ . Conversely, every $J \in \mathfrak { C }$ is itself obstructed. □

Theorem 4 of $\mathrm { [ B B C B ^ { + } 2 5 ] }$ characterizes $( w , 0 )$ -learnability by finiteness of the cube dimensions of all obstructed families. Combined with the two lemmas, it gives the characterization, for partial hypotheses as well.

Corollary 103. The following are equivalent:

(i) H is realizably PAC learnable from full sets;

(ii) $d _ { J } < \infty$ for every $J \in \mathfrak { C } ;$

(iii) $d _ { \mathrm { N } } ^ { \Omega } ( \mathcal { H } ) < \infty .$

Proof. By Lemma 101 and Theorem 4 of $\mathrm { [ B B C B ^ { + } 2 5 ] }$ , (i) holds if and only if $d _ { K } < \infty$ for every $K \in \mathfrak { I } _ { 0 }$ , which by Lemma 102 is equivalent to (ii). Since the alphabet is finite, $\Gamma < \infty$ so the sum in (128) has finitely many terms and (128) shows that (ii) is equivalent to (iii).

## C.3 Quantitative Bounds

The reduction also yields a sample-complexity bound. Here the eliminators of $\mathrm { [ B B C B ^ { + } 2 5 ] }$ are trained directly on full-set samples, whose inputs lie in the support of the target, so the filtering step of Lemma 101 is not needed. Assume $d = d _ { \mathrm { N } } ^ { \cap } ( \mathcal { H } ) < \infty .$ , and put

$$
\mathfrak { C } _ { + } = \{ J \in \mathfrak { C } : d _ { J } > 0 \} , \qquad \Gamma _ { + } = | \mathfrak { C } _ { + } | , \qquad D _ { \mathrm { s u m } } = \sum _ { J \in \mathfrak { C } } d _ { J } .
$$

Proposition 104. $I f \Gamma _ { + } \geq 1$ , then

$$
m _ { \mathcal { H } } ^ { \mathrm { f u l l } } ( \varepsilon , \delta ) \leq O _ { \ell } \bigg ( \frac { D _ { \mathrm { s u m } } + \Gamma _ { + } \log ( 2 \Gamma _ { + } / \delta ) } { \varepsilon } \bigg ) .\tag{129}
$$

$I f \Gamma _ { + } = 0 $ , a single predictor, fixed in advance, is valid on supp(h) for every $h \in \mathcal H$

The learner trains one eliminator for each $J \in \mathfrak { C }$ and predicts a common label of the sets that survive. Only the families in C are needed, since an eliminator for J excludes a member of every obstructed family containing J.

Proof. Fix a target $h \in { \mathcal { H } }$ and a distribution P supported on $\operatorname { s u p p } ( h )$ . A J-eliminator is a function $e _ { J } : \mathcal { X } \to J$ , and its error is $\operatorname* { P r } _ { x \sim P } [ e _ { J } ( x ) = h ( x ) ]$ . Theorem 10 of $\mathrm { [ B B C B ^ { + } 2 5 ] }$ gives error at most η with confidence $1 - \xi$ from $O ( ( d _ { J } + \log ( 1 / \xi ) ) / \eta )$ examples. If an eliminator

is randomized, taking a most probable symbol multiplies its error by at most $| J | \leq \ell + 1$ , a factor absorbed in (129).

If $d _ { J } = 0$ , then at every input some member of J is absent from $\{ h ^ { \prime } ( x ) : h ^ { \prime } \in \mathcal { H } \}$ . Fix an order on J and choose the first such member as $e _ { J } ( x )$ ; this eliminator has zero error and needs no examples. For $J \in { \mathfrak { C } } _ { + }$ , put

$$
a _ { J } = d _ { J } + \log \frac { 2 \Gamma _ { + } } { \delta } , \qquad T = \sum _ { J \in \mathfrak { C } _ { + } } a _ { J } , \qquad \eta _ { J } = \frac { \varepsilon a _ { J } } { T } , \qquad \xi _ { J } = \frac { \delta } { \Gamma _ { + } } .
$$

Since $( d _ { J } + \log ( 1 / \xi _ { J } ) ) / \eta _ { J } \leq a _ { J } / \eta _ { J } = T / \varepsilon$ , each eliminator requires $O _ { \ell } ( T / \varepsilon )$ examples. Train them all on the same full-set sample, viewed as labeled examples for $\mathcal { H } _ { \Sigma }$ . A union bound shows that, with probability at least $1 - \delta$ , every trained eliminator has error at most $\eta _ { J } ;$ independence between the eliminators is not required.

For a test input x, remove all the excluded symbols and let

$$
{ \mathcal { R } } ( x ) = { \mathcal { A } } \setminus \{ e _ { J } ( x ) : J \in \mathfrak { C } \} .
$$

No $J \in \mathfrak { C }$ is contained in $\mathcal { R } ( x )$ , since its chosen symbol $e _ { J } ( x )$ was removed. Thus, if $\mathcal { R } ( x ) \neq \emptyset$ , its members have a common label, for otherwise $\mathcal { R } ( x )$ would contain a minimal empty-intersection family, which lies in C. Predict the first such label, or a fixed label if $\mathcal { R } ( x ) = \emptyset$ . Whenever no eliminator excludes the true set $h ( x )$ , that set remains in $\mathcal { R } ( x )$ and the prediction is valid. On the simultaneous success event, the risk is therefore at most $\begin{array} { r } { \sum _ { J \in \mathfrak { C } _ { + } } \eta _ { J } = \varepsilon . \ \mathrm { I f } \ \Gamma _ { + } = 0 } \end{array}$ , all eliminators are exact and the same construction needs no examples. □

Comparison with Corollary 26. Our direct bound is

$$
m _ { \mathcal { H } } ^ { \mathrm { f u l l } } ( \varepsilon , \delta ) \leq O _ { \ell } \left( \frac { d \log ( ( d + 1 ) ( k + 1 ) / \varepsilon ) + \log ( 2 / \delta ) } { \varepsilon } \right) .\tag{130}
$$

By (128), $D _ { \mathrm { s u m } } \geq d .$ , and since only the $\Gamma _ { + }$ families in ${ \mathfrak { C } } _ { + }$ contribute to $D _ { \mathrm { s u m } }$ , each with $d _ { J } \leq d _ { \colon }$ , also $D _ { \mathrm { s u m } } \leq \Gamma _ { + } d _ { }$ . Neither bound dominates the other. When few families have positive dimension, (129) can be smaller, and it has no logarithmic factor in d or $1 / \varepsilon ;$ the inverse-linear dependence on ε is already present in $[ \mathrm { B B C B ^ { + } 2 5 }$ , Corollary 13]. In exchange, it sums contributions from the fixed conflict families.

Our bound avoids this sum because the potential of Lemma 17 counts products in which the conflict family may vary from one coordinate to another. Its mistake bound on n inputs is $O _ { \ell } ( d \log ( 1 + n \Gamma ) )$ for $d \geq 1$ . Since $q \leq ( k + 1 ) ^ { \ell }$ and $\Gamma \leq ( q + 1 ) ^ { \ell + 1 }$ , we have log $\Gamma = O _ { \ell } ( \log ( k + 1 ) )$ , which gives the logarithmic dependence on k in (130) for fixed ℓ.

The sum over conflict families can be much larger than d. For $k \geq 2$ and $d \geq 1$ , take all functions from a d-point domain to ${ \mathcal { A } } .$ Then $d _ { \mathrm { N } } ^ { \cap } = d _ { J } = d$ for every $J \in \mathfrak { C } , \mathrm { { s o } } \ D _ { \mathrm { { s u m } } } = \Gamma d$ Moreover, $\Gamma \geq { \binom { k } { 2 } }$ because each pair of distinct singleton sets is a minimal conflict family. This compares the two upper bounds; it does not show that the family factor is necessary.

The constructions also difer in what they provide. A PAC J-eliminator excludes one target symbol for a fixed family J. Lemmas 17 and 18 give a learner that predicts labels in Y and changes its state only after a mistake. It satisfies the transductive guarantee of Theorem 19, and its possible runs form the finite cover of Lemma 21, used in the agnostic and bandit arguments.

The reduction above concerns the realizable full-set equivalence. In general, demonstrations and bandit answers do not reveal the encoded target symbol, while countable label spaces need not admit the finite collection of conflict families used here. The main text treats these settings with additional constructions.

## D Hypergraph Coloring with Forbidden Crossings

The counting argument behind Theorem 19 has a purely combinatorial counterpart, which we present here. It bounds the chromatic number of a uniform hypergraph whose hyperedges are covered by a few complete multipartite hypergraphs, no two of which cross. For graphs whose edges are partitioned into complete bipartite graphs, this is the setting of the Alon– Saks–Seymour problem, and Remark 107 compares our bound with the known ones.

Throughout, $\chi ( G )$ denotes the weak chromatic number of a hypergraph $G ,$ the minimum number of colors in a proper vertex coloring, that is, one with no monochromatic hyperedge. For $W \subseteq V ( G )$ we write $G [ W ]$ for the induced hypergraph on $W _ { i }$ , and we call W independent if it contains no hyperedge. For pairwise disjoint nonempty sets $A ^ { 1 } , \ldots , A ^ { r }$ , we write $K ^ { ( r ) } ( A ^ { 1 } , \ldots , A ^ { r } )$ for the complete r-partite r-uniform hypergraph with these vertex classes; its hyperedges are the sets that contain exactly one vertex of each $A ^ { a }$ . Two such hypergraphs cross if every vertex class of one meets every vertex class of the other.

Theorem 105. Let $r \geq 2$ and $m \geq 1$ be integers, let $G$ be an r-uniform hypergraph, and let $K _ { i } = K ^ { ( r ) } ( A _ { i } ^ { 1 } , \cdot \cdot \cdot , A _ { i } ^ { r } ) , i \in [ m ]$ , be complete r-partite r-uniform hypergraphs such that every hyperedge of G is a hyperedge of some $K _ { i }$ . Suppose that no two of the $K _ { i }$ cross, that is, for every pair $i \neq j$ there are $a , b \in [ r ]$ such that

$$
A _ { i } ^ { a } \cap A _ { j } ^ { b } = \varnothing .\tag{131}
$$

Then, with $\alpha = ( r - 1 ) / r$ and $L = \lceil \log _ { 1 / \alpha } m \rceil + 1$

$$
\chi ( G ) \leq \sum _ { t = 0 } ^ { L } m ^ { t } \alpha ^ { \binom { t } { 2 } } .
$$

In particular,

$$
\chi ( G ) \leq \exp \biggl ( \frac { ( \log m ) ^ { 2 } } { 2 \log \left( \frac { r } { r - 1 } \right) } + O _ { r } \bigl ( \log ( m + 1 ) \bigr ) \biggr ) .\tag{132}
$$

The proof follows the potential argument of Section 7.1. For $W \subseteq V ( G )$ , call an index $i \in [ m ]$ active on W if $W$ meets every class of $K _ { i } .$ , and let $\nu ( W )$ be the number of indices active on $W ;$ this count replaces the potential Φ. Inside a class of an active $K _ { i }$ , the index i is no longer active, and since no two of the $K _ { j }$ cross, every other active index remains active in at most $r - 1$ of the r classes. This is the analogue of Lemma 17: some class of $K _ { i }$ keeps at most an α-fraction of the other active indices. Removing one such class for each active index leaves an independent set, which gets a single color, and recursing on the removed classes gives depth $O _ { r } ( \log m )$ and the stated bound.

Proof. If $W \subseteq V ( G )$ contains a hyperedge, this hyperedge belongs to some $K _ { i }$ and has a vertex in each $A _ { i } ^ { a }$ , so i is active on $W$ . Hence $\nu ( W ) = 0$ implies that W is independent. Moreover, ν is monotone: $W ^ { \prime } \subseteq W$ implies $\nu ( W ^ { \prime } ) \leq \nu ( W )$

A contraction inequality. Fix W with $s = \nu ( W ) \geq 1$ , and let i be active on W. We claim that

$$
\sum _ { a = 1 } ^ { r } \nu ( W \cap A _ { i } ^ { a } ) \leq ( r - 1 ) ( s - 1 ) .\tag{133}
$$

Indeed, the classes $A _ { i } ^ { 1 } , \ldots , A _ { i } ^ { r }$ are pairwise disjoint, so each set $W \cap A _ { i } ^ { a }$ misses the other classes of $K _ { i }$ , and i is inactive on it. An index that is inactive on W is inactive on all these subsets as well. Finally, an index $j \neq i$ that is active on W is active on at most $r - 1$ of the sets $W \cap A _ { i } ^ { 1 } , \dotsc , W \cap A _ { i } ^ { r }$ , since otherwise W $\cap A _ { i } ^ { a } \cap A _ { j } ^ { b } \neq \emptyset$ for all $a , b \in [ r ]$ , contrary to (131). Since there are $s - 1$ such indices $j ,$ summing over all indices gives (133).

A recursive coloring. The r integers on the left of (133) sum to at most $( r - 1 ) ( s - 1 )$ so the smallest of them is at most $\alpha ( s - 1 )$ , and we may choose $a _ { i } \in [ r ]$ such that

$$
\nu ( W \cap A _ { i } ^ { a _ { i } } ) \leq \left\lfloor \alpha ( s - 1 ) \right\rfloor .\tag{134}
$$

We make this choice for every index active on W. Let $I ( W )$ be the set of these s indices, and put

$$
W _ { 0 } = W \setminus \bigcup _ { i \in I ( W ) } A _ { i } ^ { a _ { i } } .
$$

The set $W _ { 0 }$ is independent, because every hyperedge contained in W lies in some $K _ { i }$ with $i \in I ( W )$ and therefore meets $A _ { i } ^ { a _ { i } }$ . Partition $W \backslash W _ { 0 }$ into at most s disjoint sets $W _ { i } \subseteq W \cap A _ { i } ^ { a _ { i } }$ $i \in I ( W )$ ; by (134) and monotonicity, $\nu ( W _ { i } ) \leq \lfloor \alpha ( s - 1 ) \rfloor$ . Give $W _ { 0 }$ one color, and color each $G [ W _ { i } ]$ recursively with its own palette, the palettes being disjoint. The resulting coloring of $G [ W ]$ is proper, since each color class lies either in $W _ { 0 }$ or in one of the sets $W _ { i }$ , where it is a color class of a proper coloring of $G [ W _ { i } ]$

To count the colors, let $F ( 0 ) = 1$ and

$$
F ( s ) = 1 + s F \bigl ( \lfloor \alpha ( s - 1 ) \rfloor \bigr ) \qquad ( s \geq 1 ) .
$$

Since $\lfloor \alpha ( s - 1 ) \rfloor$ is smaller than s and nondecreasing in $s ,$ an induction on s shows that F is nondecreasing. We now show by induction on $\nu ( W )$ that $\chi ( G [ W ] ) \leq F ( \nu ( W ) )$ for every $W \subseteq V ( G ) . { \mathrm { ~ I f ~ } } \nu ( W ) = 0$ , then $W$ is independent and one color sufices. If $\nu ( W ) = s \geq 1$ , the coloring above uses one color on $W _ { 0 }$ and, by the induction hypothesis, at most $F ( \nu ( W _ { i } ) ) \leq$ $F ( \lfloor \alpha ( s - 1 ) \rfloor )$ colors on each of the at most s sets $W _ { i }$ , hence at most $1 + s F ( \lfloor \alpha ( s - 1 ) \rfloor ) = F ( s )$ colors in total. Monotonicity of $F$ is needed here because we only know an upper bound on $\nu ( W _ { i } )$ . Since $\nu ( V ( G ) ) \leq m$ , we obtain $\chi ( G ) \leq F ( m )$

Solving the recurrence. Let $s _ { 0 } = m$ , let $s _ { j + 1 } = \lfloor \alpha ( s _ { j } - 1 ) \rfloor$ while $s _ { j } \geq 1$ , and keep $s _ { j + 1 } = 0$ once $s _ { j } = 0$ . Then $s _ { j } \le m \alpha ^ { j }$ for every j, and since $L - 1 \ge \log _ { 1 / \alpha } m$ , we get $s _ { L - 1 } \leq 1$ and hence $s _ { L } = 0$ . Unrolling $F ( s _ { j } ) = 1 + s _ { j } F ( s _ { j + 1 } )$ until the sequence reaches zero, and then using $s _ { j } \leq m \alpha ^ { j }$ , gives

$$
F ( m ) \leq \sum _ { t = 0 } ^ { L } \prod _ { j = 0 } ^ { t - 1 } s _ { j } \leq \sum _ { t = 0 } ^ { L } m ^ { t } \alpha ^ { \binom { t } { 2 } } .
$$

This is the first bound of the theorem. For the second, put $\beta = \log ( 1 / \alpha ) = \log ( r / ( r - 1 ) )$ The logarithm of $m ^ { t } \alpha ^ { ( t ) }$ is t log $m - { \textstyle \frac { \beta } { 2 } } t ( t - 1 )$ , a quadratic in t whose maximum over the reals is $\begin{array} { r } { \frac { ( \log m ) ^ { 2 } } { 2 \beta } + \frac { 1 } { 2 } \log m + \frac { \beta } { 8 } } \end{array}$ . Since there are $L + 1$ terms,

$$
\log F ( m ) \leq { \frac { ( \log m ) ^ { 2 } } { 2 \beta } } + { \frac { 1 } { 2 } } \log m + { \frac { \beta } { 8 } } + \log ( L + 1 ) = { \frac { ( \log m ) ^ { 2 } } { 2 \beta } } + O _ { r } { \bigl ( } \log ( m + 1 ) { \bigr ) } .
$$

This finishes the proof.

When the hyperedge sets of the $K _ { i }$ are disjoint, no two of them cross, which gives the following corollary.

Corollary 106 (Multipartite edge partitions). Fix $r \geq 2$ . If the hyperedge set of an r-uniform hypergraph G is partitioned into the hyperedge sets of $m \geq 1$ complete r-partite r-uniform hypergraphs, then $\chi ( G )$ satisfies (132).

Proof. Let the complete r-partite hypergraphs be $K _ { i } = K ^ { ( r ) } ( A _ { i } ^ { 1 } , \cdot \cdot \cdot , A _ { i } ^ { r } ) , i \in [ m ]$ , and fix $i \neq j$ . If all $r ^ { 2 }$ intersections $A _ { i } ^ { a } \cap A _ { j } ^ { b }$ were nonempty, we could choose $v _ { a } \in A _ { i } ^ { a } \cap A _ { j } ^ { a }$ for each $a \in [ r ]$ . These vertices are distinct, since the classes of $K _ { i }$ are disjoint, and $\{ v _ { 1 } , \ldots , v _ { r } \}$ is a hyperedge of both $K _ { i }$ and $K _ { j }$ , contrary to the disjointness of their hyperedge sets. Thus $K _ { i }$ and $K _ { j }$ do not cross, and Theorem 105 applies. □

Remark 107. For $r = 2$ , Corollary 106 concerns graphs whose edge set is partitioned into m complete bipartite graphs, the setting of the Alon–Saks–Seymour problem. For such graphs, [MV09] proved a quasipolynomial upper bound with leading exponent $( \log m ) ^ { 2 } / ( 2 \log 2 )$ which is also what Theorem 105 gives for $r = 2$ , and [Fox26] improved the exponent to $( \log m ) ^ { 2 } / ( 4 \log 2 ) + O ( \log ( m + 1 ) )$ . For graph edge partitions, therefore, Theorem 105 is weaker than the best known bound; it extends the quasipolynomial order to $r \geq 3$ and to covers whose parts may overlap but do not cross. In the other direction, Alon, Saks and Seymour conjectured that $\chi \leq m + 1$ for such graphs, and [HS12] disproved this conjecture. Strengthening the lower bound of $[ \mathrm { B B D G ^ { + } 2 1 } ] , [ \mathrm { P a b 2 6 b } ]$ constructed such graphs with chromatic number $\exp ( \Omega ( ( \log m ) ^ { 2 } ) )$ . Thus the exponent $( \log m ) ^ { 2 }$ is necessary up to a constant factor already when $r = 2$

## E Standard Borel Instance Spaces

The main proofs assume that X is countable, so that the rules a learner builds on a finite pool, such as the transductive rules of Lemma 50 and the minimax predictions of Theorem 39, need no measurability conditions. This appendix extends Theorems A–F to standard Borel instance spaces under two mild assumptions on H. The label space stays countable, and the sample-complexity bounds of Theorems A–E are unchanged. We choose the pool rules measurably and, for the agnostic demonstration learner, replace independent rounding at individual inputs by rounding on finitely many Borel sets.

## E.1 Assumptions and Statement

Let X be a nonempty standard Borel space, let Y be a nonempty countable discrete space, and fix $\ell \geq 1$ . Equip $A \cup \{ \star \}$ with its discrete σ-algebra. We may assume $\mathcal { H } \neq \emptyset$ , since otherwise the learning requirements are vacuous. We impose the following assumptions on H:

(i) Every $h \in \mathcal H$ is Borel measurable.

(ii) For every $n \geq 1$ and $v \in ( \mathcal { A } \cup \{ \star \} ) ^ { n }$ , the set

$$
E _ { n , v } = \left\{ ( x _ { 1 } , \dots , x _ { n } ) \in { \mathcal { X } } ^ { n } : ( h ( x _ { 1 } ) , \dots , h ( x _ { n } ) ) = v { \mathrm { ~ f o r ~ s o m e ~ } } h \in { \mathcal { H } } \right\}
$$

is Borel.

Assumption (ii), unlike (i), is a condition on the class as a whole: whether H realizes a given finite pattern must depend measurably on the inputs. Both assumptions hold when X is countable or H is a countable class of Borel functions, but we do not require countability.

Two facts about standard Borel spaces are used throughout: equality of inputs is a Borel condition, and X admits a Borel injection into [0, 1], hence a Borel linear order. We use this order wherever the proofs sort a finite pool by an enumeration of $x ;$ sorting finitely many inputs is Borel.

Theorem 108. Let X be a standard Borel space and let H satisfy assumptions $( \mathrm { i } ) \mathrm { - } ( \mathrm { i i } )$ . Take all distributions to be Borel probability measures and all domain restrictions, including those in the intersection classes, to be restrictions to Borel sets. Require the prediction map of a learner, or each of its label probabilities for randomized predictions, to be jointly Borel in the sample, the random seed and the test input. Then the conclusions of Theorems A–F remain valid, with the same sample-complexity bounds in Theorems A–E.

By Fubini’s theorem, these conventions make the error of a learner’s output a Borel function of the sample and the seed, so the risks and success events are measurable. The necessity directions need nothing new, since they use finite witnesses and finitely supported distributions; the explicit lower-bound and separating classes, including those of Theorem G, have countable domains; and the comparison in Appendix B is combinatorial. It remains to make the upper-bound constructions measurable. On a general X , Theorem 39 also needs its prediction families to depend measurably on the pool; Section E.3 states these conditions and checks them for the families of Sections 10 and 13.

Intersection classes. Restricting the domains in $( 7 )$ to Borel sets changes no finite trace, and hence no dimension, because a restriction on a finite pool can be realized by restricting to the finitely many inputs where the desired pattern is defined. Assumption (i) passes to $\mathcal { T } _ { R } ( \mathcal { H } )$ , since the constituent hypotheses and the restricting domain are Borel.

For assumption (ii), fix a pool size n and a partial pattern $v \in ( \mathcal { A } \cup \{ \star \} ) ^ { n }$ . Call a tuple $( w ^ { 1 } , \ldots , w ^ { r } )$ of patterns in $( { \cal { A } } \cup \{ \star \} ) ^ { n }$ compatible with v if

$$
w _ { i } ^ { a } \neq \star \mathrm { f o r ~ a l l ~ } a \in [ r ] \quad \mathrm { a n d } \quad \bigcap _ { a = 1 } ^ { r } w _ { i } ^ { a } = v _ { i } \qquad \mathrm { w h e n e v e r ~ } v _ { i } \neq \star ,
$$

with no condition at the positions where $v _ { i } = \star$ . Then v is realized by $\mathcal { T } _ { R } ( \mathcal { H } )$ on $( x _ { 1 } , \ldots , x _ { n } )$ if and only if $x _ { i } = x _ { j }$ implies $v _ { i } = v _ { j }$ and

$$
( x _ { 1 } , \ldots , x _ { n } ) \in \bigcup _ { r = 1 } ^ { R } \bigcup _ { ( w ^ { 1 } , \ldots , w ^ { r } ) { \mathrm { ~ c o m p a t i b l e ~ w i t h ~ } } v } \bigcap _ { a = 1 } ^ { r } E _ { n , w ^ { a } } .
$$

Indeed, if $h _ { 1 } , \ldots , h _ { r }$ realize a compatible tuple, restricting their intersection to the finite set $\{ x _ { i } : v _ { i } \neq \star \}$ realizes $v ,$ where the first condition ensures that no input carries both a defined and an undefined entry of $v ;$ conversely, a representing tuple of an intersection realizing v realizes a compatible tuple. The first condition is Borel, and the union is countable because $A \cup \{ \star \}$ is countable, so this event is Borel by (ii).

## E.2 Choosing the Pool Rules Measurably

The compactness arguments in Lemma 50 and Theorem 39 choose, for each pool, a point satisfying countably many closed constraints, and the pool determines which constraints are in force. The following lemma makes this choice Borel whenever the pool determines the constraints in a Borel way.

Lemma 109 (Borel choice under closed constraints). Let K be a compact metrizable space and $C _ { 1 } , C _ { 2 } , . . .$ . closed subsets of K, and for $b \in \{ 0 , 1 \} ^ { \mathbb { N } }$ let $\begin{array} { r } { K _ { b } = \bigcap _ { j : b _ { i } = 1 } C _ { j } } \end{array}$ . Let T be a standard Borel space and $S \mapsto b ( S )$ a Borel map from T to $\{ 0 , 1 \} ^ { \mathbb { N } }$ such that $K _ { b ( S ) } \neq \emptyset$ for every $S \in T$ . Then there is a Borel map $\zeta : T  K$ with $\zeta ( S ) \in K _ { b ( S ) }$ for every $S \in T$

Proof. For each $j ,$ , the set of pairs $( b , z )$ with $b _ { j } = 0 \mathrm { ~ o r ~ } z \in C _ { j }$ is closed, so the relation $\{ ( b , z ) : z \in K _ { b } \}$ is closed in $\{ 0 , 1 \} ^ { \mathbb { N } } \times K$ . Its projection $\{ b : K _ { b } \neq \emptyset \}$ is closed, being the projection of a closed set along a compact factor, and each section $K _ { b }$ is compact. The Arsenin–Kunugui theorem [Kec95, Theorem 18.18] therefore gives a Borel map $b \mapsto z _ { b } \in K _ { b }$ on this projection, and $\zeta ( S ) = z _ { b ( S ) }$ is a composition of Borel maps. □

The packed weak learner. Let $d = d _ { \mathrm { p a c k } , \gamma } ^ { \cap } ( \mathcal { H } ) < \infty$ and fix the pool size n. The set $\mathcal { P } _ { \leq 1 } ( \mathcal { V } )$ of sub-probability vectors from the proof of Lemma 50 is closed in $[ 0 , 1 ] ^ { y }$ with the product topology, so the product K of its copies indexed by the countably many pairs $( i , u )$ with $i \in [ n ]$ and $u \in \mathcal { A } ^ { n - 1 }$ is compact metrizable. Each pattern $v \in \mathcal A ^ { n }$ gives the closed constraint

$$
\sum _ { i = 1 } ^ { n } \left( 1 - \sum _ { y \in v _ { i } } z _ { y } ^ { i , v _ { - i } } \right) \leq d + n \gamma ,
$$

which is in force on a pool $S = ( x _ { 1 } , \ldots , x _ { n } ) $ exactly when $v \in \mathcal { H } | _ { S }$ , that is, when $S \in E _ { n , v }$ Enumerating $\mathcal { A } ^ { n }$ as $v _ { 1 } , v _ { 2 } , . . . ,$ the map $b _ { j } ( S ) \ : = \ : 1 \{ S \in E _ { n , v _ { j } } \}$ is Borel by (ii), and the argument of Lemma 50 shows that the constraints in force are satisfiable, so Lemma 109 gives a Borel choice of a rule for each pool. The rest of that proof stays Borel: completing the missing mass $1 - \textstyle \sum _ { y } z _ { y } ^ { i , u }$ , a countable sum of coordinates, and averaging over the finitely many permutations of the pool, which gives the symmetric kernel of the leave-one-out argument.

## E.3 Finite Rules, Compression, and Prediction Families

Apart from the reduction, which uses Lemma 109 again, the remaining constructions use on each pool only countably many tests of the form $S \in E _ { n , v }$ and finitely many comparisons of inputs or of Borel quantities, so their measurability can be read of from their definitions.

Zero-example predictors. If the relevant Natarajan or packed dimension is zero, all the correct-label sets $h ( x )$ at a fixed input x share a label, since otherwise Lemma 8 gives a witness on one input. For $y \in \mathcal { V }$ , the set of inputs at which y belongs to all these sets is

$$
\mathcal { X } \setminus \bigcup _ { \stackrel { A \in \mathcal { A } } { y \notin A } } E _ { 1 , A } ,
$$

where $E _ { 1 , A }$ is the set of inputs x with $h ( x ) = A$ for some $h \in \mathcal H$ . This set is Borel, so predicting the first such label in the enumeration of Y is Borel. At inputs where every hypothesis is undefined, all labels qualify, and any prediction is allowed there.

Finite label spaces. For finite Y, the conservative rule of Section 7 uses only which partial patterns are realized on its pool. Each such test is Borel by (ii), so the potential, the updates and the fixed tie-breaking rules are Borel, and enumerating the possible runs gives Borel covers. The exponential-weights and bandit learners built from these covers are therefore measurable.

Compression. The compression constructions are measurable once the full-set weak kernel is jointly Borel. On a finite sample, each boosting round chooses among finitely many multisets of examples, and we take the first one that satisfies the required loss inequality. Reconstruction takes finite averages, chooses the first most probable label, and corrects

the predictions at finitely many sample inputs, which is Borel because equality of inputs is.   
Empirical risk minimization and validation compare finitely many Borel quantities.

Prediction families. On an ordered pool $S = ( x _ { 1 } , \dots , x _ { N } )$ , identify a function on supp(S) with its vector of values in $\mathcal { V } ^ { N }$ . For a general X we add the following conditions to the hypotheses of Theorem 39:

(a) the family $F _ { h , S }$ depends only on S and the pattern $v = ( h ( x _ { 1 } ) , \ldots , h ( x _ { N } ) ) \in \mathcal { A } ^ { N }$

(b) for each such v, this family is a Borel function of S, with values in the countable space of finite subsets of $\mathcal { V } ^ { N }$

(c) its members agree at repeated inputs of $S ;$

(d) permuting the pool and the pattern permutes the coordinates of every vector in the family.

The finite-label families of Section 10 and the message families of Section 13.2 satisfy them. For both, sort the pool and its pattern by the Borel order, construct the family there, and return its vectors to the original order; this gives (d). The finite-label families enumerate the runs of the conservative rule, reading a cover vector at the first occurrence of each input, which gives (c); these runs depend only on the pattern and on realization tests, which gives (a) and (b). The message families decode the admissible messages of Section 13.2; for a fixed pattern there are finitely many of them, their admissibility is determined by the pattern, and the decoder is a compression reconstruction, hence jointly Borel by the previous paragraph. The size bounds and intersection properties are exactly those already proved.

The reduction. It remains to make the minimax choice in the proof of Theorem 39 measurable. Recall that the learner observes the labels at the positions J of a pool $S =$ $( x _ { 1 } , \ldots , x _ { N } )$ , that $U = [ N ] \backslash J _ { : }$ and that it needs distributions $( p _ { z } ) _ { z \in \operatorname { s u p p } ( U ) }$ satisfying (47) for every retained hypothesis, where $\beta$ is the constant defined at the start of that proof. By condition (a), membership of h in $\mathcal { H } _ { J }$ depends only on its pattern $v \in \mathcal { A } ^ { N }$ on S, and v is retained if $S \in E _ { N , v } , y _ { i } \in v _ { i }$ for every $i \in J$ , and (46) holds for every member of its family; each of these tests is Borel. Since the set supp(U) varies with the pool, we index the distributions by positions instead and work in the compact metrizable space $\Delta _ { \leq 1 } ( \mathcal { Y } ) ^ { N }$ We require $p _ { i } ~ = ~ p _ { j }$ whenever $x _ { i } ~ = ~ x _ { j }$ , and for each retained pattern v we impose the constraint (53),

$$
\frac { 1 } { | U | } \sum _ { i \in U } \sum _ { y \in v _ { i } } p _ { i } ( y ) \geq 1 - \beta .
$$

These constraints are closed, the set of constraints in force is a Borel function of the data, and the proof of Theorem 39 shows that they are satisfiable when some pattern is retained. Lemma 109 therefore gives a Borel choice, and completing the missing mass violates no constraint. If no pattern is retained, a Borel event, we use the point mass on the fixed label. No measurable selection of a hypothesis from H is needed.

The proof also needs this choice to depend only on the multiset S and the multiset $\{ ( x _ { i } , y _ { i } ) : i \in J \}$ of observed examples. We therefore apply the selection to a canonical arrangement of the data, a Borel function of these two multisets: the observed examples first, sorted by the Borel order on inputs and then by the enumeration of $\mathcal { V } ,$ followed by the unobserved inputs in sorted order. By condition (d), the retained patterns are rearranged accordingly.

## E.4 Randomization and the Remaining Arguments

A jointly Borel kernel $p _ { \sigma } ( \cdot \mid x )$ can be realized as a measurable predictor with a single random seed. Write $\mathcal { V } = \{ y _ { 1 } , y _ { 2 } , . . . \}$ , take $\xi \sim \operatorname { U n i f } ( [ 0 , 1 ) )$ independent of everything else, and let $g _ { \sigma , \xi } ( x )$ be the first label $y _ { j }$ such that

$$
\xi < \sum _ { a = 1 } ^ { j } p _ { \sigma } ( y _ { a } \mid x ) .
$$

The partial sums increase to 1, so such a label exists. The map $( \sigma , \xi , x ) \mapsto g _ { \sigma , \xi } ( x )$ is jointly Borel with label distribution $p _ { \sigma } ( \cdot \mid x )$ at every input, so it has the same expected risk as the kernel; the error is evaluated at a single test input, so predictions at distinct inputs need not be independent. Conversely, averaging a learner over its seed preserves joint measurability by Fubini’s theorem.

The probability arguments for Theorems A–E now concern Borel random variables on standard Borel spaces. In particular, the conditional law of the demonstrated label given the input, used in the reduction, exists by the disintegration theorem [Kec95], so their exchangeability, concentration, compression and confidence-amplification arguments apply as written. It remains to extend the agnostic demonstration learner of Theorem F.

## E.5 Agnostic Learning from Demonstrations

Assume that H is nondegenerate and locally compatible, with $d _ { \mathrm { D S } } ^ { \Omega } ( \widetilde { \mathcal { H } } ) < \infty$ . We first check that the learner of Lemma 78 is measurable. The only probability argument that needs a diferent construction is the rounding step of Lemma 79. We replace its independent choices at individual inputs by independent choices on the gaps between finitely many sampled inputs.

The regions and the learner. The two regions are Borel, and membership in each local alphabet is Borel in the input, since

$$
\mathcal { X } _ { \operatorname* { m i x } } = \bigcup _ { \stackrel { A \in \mathcal { A } } { | A | \geq 2 } } E _ { 1 , A } , \qquad \{ x \in \mathcal { X } _ { \operatorname* { m i x } } : a \in \mathcal { V } _ { x } \} = \mathcal { X } _ { \operatorname* { m i x } } \cap E _ { 1 , \{ a \} } .
$$

The collapsed class inherits assumptions $( \mathrm { i } ) { - } ( \mathrm { i i } )$ : its hypotheses are Borel, and each finitepattern realization event is a countable union of original realization events. Restriction to either Borel region preserves these properties.

On the mixed region, enumerate $\mathcal { \partial } _ { x }$ in the fixed order on $\mathcal { V } ,$ , encode its labels by their ranks in [ℓ], and encode ⋄ by $\ell + 1$ . This encoding is Borel in x. A realization event for the encoded class is a countable union, over original patterns, of the corresponding events $E _ { n , v }$ intersected with the Borel conditions specifying the encoding. Thus the encoded class also satisfies (i)–(ii). Its finite covers are Borel by Section E.3, and the specialist updates and finite permutation averages give a jointly Borel kernel in Lemma 78. The learner on the pure region is already covered by the full-set extension with $\ell = 1$

Gaps of a sample. Fix a Borel injection $t : \mathcal { X } _ { \mathrm { m i x } }  [ 0 , 1 ]$ . Given inputs $X _ { 1 } , \dots , X _ { u }$ , let $U = \{ X _ { 1 } , \ldots , X _ { u } \}$ and list $t ( U )$ as $z _ { 1 } < \dots < z _ { s }$ . Put $z _ { 0 } = - \infty , z _ { s + 1 } = + \infty$ , and

$$
C _ { j } = \{ x \in \mathcal { X } _ { \operatorname* { m i x } } : z _ { j } < t ( x ) < z _ { j + 1 } \} , \qquad 0 \leq j \leq s .
$$

These gaps partition ${ \mathcal { X } } _ { \operatorname* { m i x } } \setminus U$ . Their membership relations are Borel in the sample and the test input. The next lemma shows that a fresh sample makes all their masses small, even when the input distribution has atoms.

Lemma 110. Let $X _ { 1 } , \dots , X _ { u }$ be independent with law $\mu$ on ${ \mathcal { X } } _ { \mathrm { m i x } }$ , where $u \geq 1$ . For every $0 < \tau < 1$

$$
\operatorname* { P r } \big [ \operatorname* { m a x } _ { 0 \leq j \leq s } \mu ( C _ { j } ) > \tau \big ] \leq ( u + 1 ) ( 1 - \tau ) ^ { u } \leq ( u + 1 ) e ^ { - u \tau } .
$$

Proof. Let F be the distribution function of $t ( X )$ for $X ~ \sim ~ \mu$ Couple the sample as $t ( X _ { i } ) ~ = ~ F ^ { - 1 } ( V _ { i } )$ , where the $V _ { i }$ are independent uniform variables on $( 0 , 1 )$ and $F ^ { - 1 }$ is the generalized inverse. The gap between consecutive observed values $a \ < \ b$ has mass $F ( b - ) - F ( a )$ . The interval $( F ( a ) , F ( b - ) )$ contains no $V _ { i } ,$ so its length is at most the largest spacing between consecutive ordered $V _ { i } \mathrm { { ^ { s } } }$ , including the two exterior spacings. The same holds for an exterior gap, with $F ( - \infty ) = 0$ and $F ( + \infty - ) = 1$

The $u + 1$ uniform spacings have a joint distribution uniform on the simplex of nonnegative vectors summing to one. Each spacing therefore exceeds $\tau$ with probability $( 1 - \tau ) ^ { u }$ , by translating that coordinate by $\tau$ and rescaling the simplex. A union bound proves the claim. □

Completing the confidence argument. We now extend Lemma $7 9$ to standard Borel instance spaces. Let $\mu$ be the input marginal on the mixed region and choose a Borel version $\pi _ { x }$ of the conditional law of the demonstrated label. With $c _ { h } , w _ { h }$ and $\pi _ { x } ( p )$ as in Section 14.6, the conditional comparison becomes

$$
\Delta _ { h } ( p ) = \int \left( c _ { h } ( x ) - w _ { h } ( x ) \pi _ { x } ( p ) \right) d \mu ( x ) .\tag{135}
$$

All these functions are Borel. Keep $\alpha = \varepsilon / 4 , m _ { 0 }$ , K and $\tau$ from the proof of Lemma 79, and replace its second block size by

$$
u = \left\lceil { \frac { 2 } { \tau } } \log { \frac { 8 } { \delta \tau } } \right\rceil .
$$

Use the same formula for the third block size v, with this new value of $u .$

Step 1 is unchanged: averaging the K kernels gives $\Delta _ { h } ( \bar { p } ) \leq 2$ α except with probability $\delta / 4$ , for each fixed h. Use the second block to form $U$ and the gaps $C _ { j }$ . Lemma 110 gives max<sub>j</sub> $\mu ( C _ { j } ) \leq \tau$ except with probability $( u + 1 ) e ^ { - u \tau } \leq \delta / 4$

Step 3 is also unchanged, since U is finite: use the third block to choose a most frequently demonstrated label $\widehat { a } _ { x } \in \mathcal { V } _ { x }$ at each $x \in U$ . Let $p ^ { U }$ equal $\bar { p }$ of U and put mass one on $\widehat { a } _ { x }$ at $x \in U$ . The same union bound over at most $\ell ^ { u }$ maps on $U$ , followed by (135), gives

$$
\Delta _ { h } ( p ^ { U } ) \leq \Delta _ { h } ( \bar { p } ) + \alpha
$$

except with probability $\delta / 4$

For the rounding step, draw one independent uniform seed $\xi _ { j }$ for each gap $C _ { j }$ . At every $x \in C _ { j } .$ , sample from $\bar { p } ( \cdot \mid x )$ by the inverse-CDF rule of Section E.4, using the same seed $\xi _ { j }$ throughout this gap. On U, predict $\widehat { a } _ { x }$ . This defines an admissible predictor $\widehat g$ that is jointly Borel in the examples, the seeds and the test input. There are at most $u + 1$ seeds, which can be encoded in a single uniform seed.

Fix h and condition on all examples. The contribution of gap $C _ { j }$ to the change in comparison is

$$
Z _ { j } = \int _ { C _ { j } } w _ { h } ( x ) \big ( \pi _ { x } ( \bar { p } ) - \pi _ { x } ( \widehat { g } ) \big ) d \mu ( x ) .
$$

At each $x \in C _ { j }$ , the rounded label has law $\bar { p } ( \cdot \mid x )$ , so Fubini’s theorem gives $\mathbb { E } [ Z _ { j } \mid$ examples] = 0. The $Z _ { j }$ are independent, since they use diferent seeds. Their range lengths are at most $\mu ( C _ { j } )$ : the first term in each integral is fixed, while $\begin{array} { r } { \int _ { C _ { i } } w _ { h } ( x ) \pi _ { x } ( \widehat { g } ) d \mu ( x ) } \end{array}$ lies in $[ 0 , \mu ( C _ { j } ) ]$ . On the event that all gaps have mass at most $\tau ,$ we have $\begin{array} { r } { \sum _ { j } \mu ( C _ { j } ) ^ { 2 } \leq \tau } \end{array}$ . Hence Hoefding’s inequality gives

$$
\begin{array} { r } { \operatorname* { P r } \big [ \Delta _ { h } ( \widehat { g } ) - \Delta _ { h } ( p ^ { U } ) > \alpha \mid \mathrm { e x a m p l e s } \big ] = \operatorname* { P r } \big [ \sum _ { j } Z _ { j } > \alpha \mid \mathrm { e x a m p l e s } \big ] \le e ^ { - 2 \alpha ^ { 2 } / \tau } \le \delta / 4 . } \end{array}
$$

The four failure events have total probability at most $\delta .$ . Outside their union, $\Delta _ { h } ( \widehat { g } ) \cong$ $2 \alpha + \alpha + \alpha = \varepsilon$ . This proves the confidence lemma on a standard Borel instance space.

The observable comparison now gives the mixed-region guarantee by choosing a fixed nearoptimal hypothesis, as in Proposition 80. Since the regions are Borel, the gluing argument of Lemma 81 combines this learner with the pure-region learner measurably. Necessity and the degenerate obstruction use finite witnesses and finitely supported distributions, so they need no change. This proves the extension of Theorem F and completes the proof of Theorem 108.